"""
统一入库管道 — 文件/目录/文本 → 自动切分 → SurrealDB + zvec。

提供三个入口：
  ingest_file(path)        — 单个文件入库
  ingest_directory(path)   — 批量目录遍历入库
  ingest_text(title, content, chunk_type) — 内存文本入库（供 LLM 编译后调用）

每个文件：
  1. 读取内容
  2. chunker.chunk_file() 切分为语义 chunk
  3. wiki_runtime.create_document() 写入 SurrealDB（含图/元数据）
  4. zvec_client.ingest_chunks() 写入向量库
  5. 返回结构化结果 {status, doc_id, chunks, ...}
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import chunker
import wiki_runtime as wr
import zvec_client

# 可选导入：本体模块（需 surrealdb）
try:
    import ontology
    _ONTOLOGY_AVAILABLE = True
except ImportError:
    _ONTOLOGY_AVAILABLE = False


# ====================================================================== #
#  支持的文件扩展名
# ====================================================================== #
SUPPORTED_EXTS: set[str] = {
    # Markdown
    ".md", ".markdown", ".mdown", ".mkd",
    # 纯文本
    ".txt", ".rst", ".org",
    # 代码
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
    # 结构化
    ".yaml", ".yml", ".json", ".toml", ".xml",
    # 配置
    ".cfg", ".ini", ".conf", ".env",
    # W3.3 HTML 网页
    ".html", ".htm",
    # W3.2 PDF 文档（需 pdfplumber）
    ".pdf",
}

# 跳过目录和文件
SKIP_DIRS: set[str] = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".DS_Store", ".idea", ".vscode", "dist", "build",
    "wiki_data.db",  # SurrealDB 数据目录
}
SKIP_FILES: set[str] = {
    ".gitignore", ".gitattributes", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "Cargo.lock", "Pipfile.lock", "poetry.lock",
}


# ====================================================================== #
#  单文件入库
# ====================================================================== #

def ingest_file(
    file_path: str,
    topic: str | None = None,
    author: str | None = None,
    skip_existing: bool = True,
    skip_zvec: bool = False,
    commit_hash: str | None = None,
) -> dict:
    """将单个文件入库为 Wiki 文档。

    Args:
        file_path: 文件路径
        topic: 可选主题 ID
        author: 可选作者
        skip_existing: 是否跳过已存在（按标题匹配）
        skip_zvec: 跳过向量库写入（仅写 SurrealDB）

    Returns:
        {
            "status": "ok" | "skipped" | "error",
            "file": str,
            "doc_id": str | None,
            "title": str,
            "chunks": int,
            "vectors": int,
            "reason": str | None,  # 仅 skipped/error
        }
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return {"status": "error", "file": str(file_path), "reason": "文件不存在"}

    name = path.stem
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        return {"status": "skipped", "file": str(file_path), "reason": f"不支持的文件类型: {ext}"}

    # 读取内容
    if ext == ".pdf":
        # W3.2：PDF 解析（pdfplumber 懒加载，未安装则明确报错）
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            return {
                "status": "error", "file": str(file_path),
                "reason": "未安装 pdfplumber，请先 `pip install pdfplumber`",
            }
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            content = "\n\n".join(pages)
        except Exception as e:
            return {"status": "error", "file": str(file_path), "reason": f"PDF 解析失败: {e}"}
    else:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"status": "error", "file": str(file_path), "reason": f"读取失败: {e}"}

    if not content.strip():
        return {"status": "skipped", "file": str(file_path), "reason": "文件为空"}

    # 检查重复（按标题）
    if skip_existing:
        existing = wr.find_documents_by_title(name)
        if existing:
            return {
                "status": "skipped",
                "file": str(file_path),
                "doc_id": existing[0]["id"],
                "title": name,
                "reason": f"已存在同名文档 ({existing[0]['id']})",
            }

    # 确定文档类型用于 chunk 切分
    ftype = chunker.detect_type(str(path))

    # 切分
    if ext == ".pdf":
        # PDF 已解析为纯文本，按是否含标题层级选择 md/text
        ctype = "md" if any(l.strip().startswith("## ")
                            for l in content.split("\n")[:40]) else "text"
        chunks = chunker.chunk_text(content, name, ctype)
    else:
        chunks = chunker.chunk_file(str(path), content)
    if not chunks:
        return {"status": "error", "file": str(file_path), "reason": "切分后无有效 chunk"}

    # W3.1：chunk 级概念标注（写入 chunk metadata，随向量入库供检索使用）
    _annotate_chunk_concepts(chunks)

    # 自动摘要：取第一个 chunk 的前 200 字
    summary = chunks[0].text[:200] if chunks else ""

    # 自动标签：根据扩展名
    tags = _infer_tags(str(path))

    # 代码文件：提取结构化符号索引（索引 D：函数/类/调用链），随文档落库
    code_symbols = None
    if ftype == "code":
        try:
            import code_analyzer
            code_symbols = code_analyzer.analyze_code(content, name + ext)
        except Exception:
            code_symbols = None

    try:
        # 写入 SurrealDB
        doc = wr.create_document(
            title=name,
            content=content,
            summary=summary,
            topic_id=topic,
            author=author,
            doc_key=name.lower(),
            tags=tags,
            code_symbols=code_symbols,
            commit_hash=commit_hash,
        )
    except Exception as e:
        return {"status": "error", "file": str(file_path), "reason": f"SurrealDB 写入失败: {e}"}

    doc_id = doc.get("id", "") if doc else ""

    # 概念标注（P1-1）
    concept_ids = _annotate_concepts(name, content) or []
    if concept_ids and doc_id:
        _update_document_concepts(doc_id, concept_ids)

    # 写入向量库
    vector_count = 0
    if not skip_zvec and doc_id:
        try:
            zvec_chunks = chunker.chunks_to_zvec(chunks, doc_id, name)
            vector_count = zvec_client.ingest_chunks(zvec_chunks)
        except Exception as e:
            # 向量入库失败不阻塞文档入库，返回 warning
            return {
                "status": "ok",
                "file": str(file_path),
                "doc_id": doc_id,
                "title": name,
                "chunks": len(chunks),
                "vectors": 0,
                "warning": f"向量入库失败: {e}",
            }

    return {
        "status": "ok",
        "file": str(file_path),
        "doc_id": doc_id,
        "title": name,
        "chunks": len(chunks),
        "vectors": vector_count,
    }


# ====================================================================== #
#  目录批量入库
# ====================================================================== #

def ingest_directory(
    dir_path: str,
    topic: str | None = None,
    author: str | None = None,
    recursive: bool = True,
    skip_existing: bool = True,
    skip_zvec: bool = False,
    max_files: int = 500,
) -> dict:
    """递归遍历目录，将支持的文件批量入库。

    Args:
        dir_path: 目录路径
        topic: 可选主题 ID
        author: 可选作者
        recursive: 是否递归子目录
        skip_existing: 是否跳过已存在
        skip_zvec: 跳过向量库
        max_files: 最大处理文件数（防止误操作处理超大仓库）

    Returns:
        {
            "total": int,
            "succeeded": int,
            "skipped": int,
            "failed": int,
            "results": [{...}, ...]
        }
    """
    base = Path(dir_path).resolve()
    if not base.is_dir():
        return {"error": f"{dir_path} 不是目录"}

    files: list[Path] = []
    if recursive:
        for root, dirs, filenames in os.walk(base):
            # 跳过目录
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in filenames:
                if fname in SKIP_FILES:
                    continue
                fp = Path(root) / fname
                if fp.suffix.lower() in SUPPORTED_EXTS:
                    files.append(fp)
    else:
        for fp in base.iterdir():
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTS:
                files.append(fp)

    # 限制数量
    files = files[:max_files]

    results: list[dict] = []
    for fp in files:
        result = ingest_file(
            str(fp),
            topic=topic,
            author=author,
            skip_existing=skip_existing,
            skip_zvec=skip_zvec,
        )
        results.append(result)

    succeeded = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")

    return {
        "total": len(results),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


# ====================================================================== #
#  内存文本入库（供 LLM 编译后调用）
# ====================================================================== #

def ingest_text(
    title: str,
    content: str,
    chunk_type: str | None = None,
    topic: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
    skip_zvec: bool = False,
    commit_hash: str | None = None,
) -> dict:
    """将内存中的文本入库为 Wiki 文档（不需要文件）。

    供 LLM compile_document 后自动同步向量使用。

    Args:
        title: 文档标题
        content: 文本内容
        chunk_type: "md" | "code" | "text" | "yaml"（None 则自动检测）
        topic: 可选主题
        author: 可选作者
        tags: 可选标签
        skip_zvec: 跳过向量库

    Returns:
        {status, doc_id, title, chunks, vectors}
    """
    if not content.strip():
        return {"status": "error", "reason": "内容为空"}

    # 自动检测类型
    if chunk_type is None:
        # 简单启发式：包含 ## 标题 → md，包含 def/class → code
        if any(line.strip().startswith("## ") for line in content.split("\n")[:20]):
            chunk_type = "md"
        elif any(
            line.strip().startswith(("def ", "class ", "function ", "import ", "from "))
            for line in content.split("\n")[:20]
        ):
            chunk_type = "code"
        else:
            chunk_type = "text"

    # 切分
    chunks = chunker.chunk_text(content, title, chunk_type)
    if not chunks:
        return {"status": "error", "reason": "切分后无有效 chunk"}

    # W3.1：chunk 级概念标注
    _annotate_chunk_concepts(chunks)

    # 自动摘要
    summary = chunks[0].text[:200] if chunks else ""

    # 代码内容：提取结构化符号索引（索引 D），随文档落库
    code_symbols = None
    if chunk_type == "code":
        try:
            import code_analyzer
            code_symbols = code_analyzer.analyze_code(content, title)
        except Exception:
            code_symbols = None

    try:
        # 写入 SurrealDB
        doc = wr.create_document(
            title=title,
            content=content,
            summary=summary,
            topic_id=topic,
            author=author,
            tags=tags,
            code_symbols=code_symbols,
            commit_hash=commit_hash,
        )
    except Exception as e:
        return {"status": "error", "reason": f"SurrealDB 写入失败: {e}"}

    doc_id = doc.get("id", "") if doc else ""

    # 概念标注（P1-1）
    concept_ids = _annotate_concepts(title, content) or []
    if concept_ids and doc_id:
        _update_document_concepts(doc_id, concept_ids)

    # 写入向量库
    vector_count = 0
    if not skip_zvec and doc_id:
        try:
            zvec_chunks = chunker.chunks_to_zvec(chunks, doc_id, title)
            vector_count = zvec_client.ingest_chunks(zvec_chunks)
        except Exception as e:
            return {
                "status": "ok",
                "doc_id": doc_id,
                "title": title,
                "chunks": len(chunks),
                "vectors": 0,
                "warning": f"向量入库失败: {e}",
            }

    return {
        "status": "ok",
        "doc_id": doc_id,
        "title": title,
        "chunks": len(chunks),
        "vectors": vector_count,
    }


# ====================================================================== #
#  辅助
# ====================================================================== #

def _infer_tags(file_path: str) -> list[str]:
    """根据文件路径和类型推断标签。"""
    path = Path(file_path)
    ext = path.suffix.lower()
    tags: list[str] = []

    # 扩展名标签
    ext_tag_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust",
        ".md": "markdown", ".yaml": "yaml", ".json": "json",
        ".cpp": "cpp", ".c": "c", ".h": "c",
        ".html": "html", ".htm": "html", ".pdf": "pdf",
    }
    if ext in ext_tag_map:
        tags.append(ext_tag_map[ext])

    # 目录名标签（父目录名）
    parent_name = path.parent.name.lower()
    if parent_name and parent_name not in (".", "src", "lib", "docs"):
        tags.append(parent_name)

    return tags


# ====================================================================== #
#  概念标注（P1-1：入库时标注 chunk 所属概念）
# ====================================================================== #

def _annotate_concepts(title: str, content: str) -> list[str] | None:
    """判断文档属于哪些概念，返回 concept_id 列表。

    优先用 LLM 多选题式标注（从本体概念列表中选择），
    LLM 不可达或失败时兜底为关键词子串匹配。
    仅在 ontology 模块可用时执行。
    """
    if not _ONTOLOGY_AVAILABLE:
        return None
    try:
        # 获取已有概念列表
        concepts = ontology.list_concepts()
        if not concepts:
            return None

        name_to_id = {c["name"]: c["id"] for c in concepts}

        # 优先：LLM 多选题标注
        try:
            from llm import annotate_document_concepts
            names = annotate_document_concepts(title, content[:800], concepts)
            matched = [name_to_id[n] for n in names if n in name_to_id]
            if matched:
                return matched
        except Exception:
            pass

        # 兜底：关键词子串匹配（不需要 LLM 调用）
        content_lower = (title + " " + content[:500]).lower()
        matched = [c["id"] for c in concepts if c["name"].lower() in content_lower]
        return matched if matched else None
    except Exception:
        return None


def _annotate_chunk_concepts(chunks: list) -> None:
    """W3.1：对切分后的 chunk 做概念标注，结果写入 chunk.metadata['concept_ids']。

    采用确定性的关键词子串匹配（不调用 LLM，开销低），
    标注后的概念随 chunks_to_zvec 进入向量库字段，供检索候选回带 chunk 级概念。
    """
    if not _ONTOLOGY_AVAILABLE:
        return
    try:
        concepts = ontology.list_concepts()
    except Exception:
        return
    if not concepts:
        return
    name_to_id = {c["name"].lower(): c["id"] for c in concepts if c.get("name")}
    if not name_to_id:
        return
    for ch in chunks:
        text = f"{ch.heading} {ch.text}".lower()
        matched = [cid for nm, cid in name_to_id.items() if len(nm) >= 2 and nm in text]
        if matched:
            ch.metadata = {**(ch.metadata or {}), "concept_ids": matched}


def propose_ontology_from_directory(dir_path: str, draft_path: str | None = None) -> dict:
    """W3.6：AST → 本体草案。

    遍历代码目录，用 code_analyzer 提取函数/类（=概念）与调用关系（=depends 边），
    生成一份 concept 草案 YAML（不自动写入线上本体，需人工 review 合入）。

    返回 {status, concepts, relations, draft_path}。
    """
    base = Path(dir_path).resolve()
    if not base.is_dir():
        return {"status": "error", "reason": f"{dir_path} 不是目录"}
    try:
        import code_analyzer
    except Exception as e:
        return {"status": "error", "reason": f"code_analyzer 不可用: {e}"}

    concepts: dict[str, dict] = {}
    relations: list[dict] = []
    seen_relations: set[str] = set()

    def _add_concept(name: str, ctype: str):
        if name and name not in concepts:
            concepts[name] = {
                "name": name,
                "type": ctype,
                "description": f"由代码 {ctype} 自动提取（待 review）",
            }

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fp = Path(root) / fname
            if fp.suffix.lower() not in {".py", ".js", ".ts", ".java", ".go", ".rs",
                                         ".cpp", ".c", ".h", ".hpp", ".cs", ".rb"}:
                continue
            try:
                a = code_analyzer.analyze_file(str(fp))
            except Exception:
                a = None
            if not a:
                continue
            for sym in a.get("symbols", []):
                ctype = "component" if sym.get("kind") == "class" else "method"
                _add_concept(sym["name"], ctype)
            for call in a.get("calls", []):
                caller, callee = call.get("caller"), call.get("callee")
                if not caller or not callee:
                    continue
                key = f"{caller}->{callee}"
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                _add_concept(caller, "method")
                _add_concept(callee, "method")
                relations.append({"from": caller, "to": callee, "type": "depends"})

    draft = {
        "concepts": list(concepts.values()),
        "relations": relations,
    }
    if draft_path is None:
        draft_path = str(base / "ontology_draft.yaml")
    try:
        import yaml
        with open(draft_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(draft, f, allow_unicode=True, sort_keys=False)
    except Exception:
        # 无 yaml 依赖则写 JSON
        import json
        with open(draft_path.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(draft, f, ensure_ascii=False, indent=2)
        draft_path = draft_path.replace(".yaml", ".json")

    return {
        "status": "ok",
        "concepts": len(concepts),
        "relations": len(relations),
        "draft_path": draft_path,
    }


def _update_document_concepts(doc_id: str, concept_ids: list[str] | None) -> None:
    """将概念绑定写入 SurrealDB document 的 concept_ids 字段。

    注意：调用方传入的 doc_id 经 _enrich_document 处理后已去掉 `document:` 前缀，
    此处必须规范化回 `document:{id}` record id，否则 SurrealDB 会把裸 key 当成表名，
    UPDATE 静默空操作（document.concept_ids 永远写不进）。
    concept_ids 传空列表表示清空旧绑定。
    """
    if not _ONTOLOGY_AVAILABLE or concept_ids is None:
        return
    rid = doc_id if ":" in doc_id else f"document:{doc_id}"
    try:
        from db import get_db
        db = get_db()
        db.query(f"UPDATE {rid} SET concept_ids = $ids", {"ids": concept_ids})
        # 建立 concept_binding 边
        for cid in concept_ids:
            ontology.bind_concept(cid, doc_id, binding_type="inferred")
    except Exception:
        pass


def repair_concept_bindings(limit: int = 200) -> dict:
    """对现有文档重新标注概念（使用 LLM 多选题标注，含关键词兜底）。

    用于修复概念绑定稀疏问题，使 Re-Rank 的概念距离因子真正生效。
    返回 {scanned, updated, errors}。
    """
    if not _ONTOLOGY_AVAILABLE:
        return {"scanned": 0, "updated": 0, "errors": ["ontology 不可用"]}
    try:
        docs = wr.list_documents(limit=limit)
    except Exception as e:
        return {"scanned": 0, "updated": 0, "errors": [str(e)]}

    scanned = 0
    updated = 0
    errors: list[str] = []
    for doc in docs:
        doc_id = doc.get("id", "")
        title = doc.get("title", "")
        content = doc.get("content", "")
        if not doc_id:
            continue
        scanned += 1
        try:
            new_ids = _annotate_concepts(title, content) or []
            old_ids = doc.get("concept_ids") or []
            if set(new_ids) != set(old_ids):
                _update_document_concepts(doc_id, new_ids)
                updated += 1
        except Exception as e:
            errors.append(f"{doc_id}: {e}")
    return {"scanned": scanned, "updated": updated, "errors": errors}


# ====================================================================== #
#  LLM compile_document 后自动同步向量（供 app.py 调用）
# ====================================================================== #

def sync_vectors_for_document(doc_id: str) -> dict:
    """为已有的 SurrealDB 文档重建向量索引。

    流程：get_document → chunk_text → zvec ingest。
    若文档已有旧向量分块，先删除再重建。

    Args:
        doc_id: SurrealDB 文档 ID（如 "document:transformer"）

    Returns:
        {status, doc_id, chunks_added, old_removed}
    """
    doc = wr.get_document(doc_id)
    if not doc:
        return {"status": "error", "reason": f"文档 {doc_id} 不存在"}

    title = doc.get("title", "Untitled")
    content = doc.get("content", "")
    if not content:
        return {"status": "error", "reason": "文档内容为空"}

    # 删除旧向量分块
    old_removed = 0
    try:
        old_removed = zvec_client.delete_by_document_id(doc_id)
    except Exception:
        pass

    # 重新切分
    # 检测类型：如果内容包含 ## 标题，用 md 模式
    if any(line.strip().startswith("## ") for line in content.split("\n")[:30]):
        chunks = chunker.chunk_text(content, title, "md")
    else:
        chunks = chunker.chunk_text(content, title, "text")

    if not chunks:
        return {"status": "error", "reason": "切分后无有效 chunk"}

    # 向量入库
    try:
        zvec_chunks = chunker.chunks_to_zvec(chunks, doc_id, title)
        added = zvec_client.ingest_chunks(zvec_chunks)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"向量入库失败: {e}",
            "chunks_prepared": len(chunks),
        }

    return {
        "status": "ok",
        "doc_id": doc_id,
        "chunks_added": added,
        "old_removed": old_removed,
    }
