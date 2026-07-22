"""
文档智能切分器 — 按文档类型做语义完整的 chunk。

支持：
  - Markdown：按 ## 标题 + 段落切分，不切断代码块和表格
  - 代码（Python/JS/Java）：按函数/类边界切分（正则），不切断函数体
  - 纯文本：按段落（双换行）切分
  - YAML/JSON：整文件单 chunk

每个 chunk 带元数据：
  - chunk_id, chunk_index, chunk_type (md/code/text/yaml)
  - file_path, heading (所属标题), start_line, end_line
  - parent_chunk_id (超长 chunk 二次切分时)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    """一个语义完整的文档分块。"""
    chunk_id: str
    chunk_index: int
    chunk_type: str          # "md" | "code" | "text" | "yaml"
    text: str                # chunk 原文
    file_path: str           # 来源文件路径
    heading: str = ""        # 所属的 Markdown 标题（代码为函数/类名）
    start_line: int = 1
    end_line: int = 1
    parent_chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


# ====================================================================== #
#  文件类型检测
# ====================================================================== #

# 代码文件扩展名 → 语言
CODE_EXTS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
}

# Markdown 扩展名
MD_EXTS: set[str] = {".md", ".markdown", ".mdown", ".mkd"}

# 结构化文本（整文件单 chunk）
STRUCTURED_EXTS: set[str] = {".yaml", ".yml", ".json", ".toml", ".xml", ".cfg", ".ini", ".conf"}


def detect_type(file_path: str) -> str:
    """根据扩展名检测文件类型。"""
    ext = Path(file_path).suffix.lower()
    if ext in MD_EXTS:
        return "md"
    if ext in CODE_EXTS:
        return "code"
    if ext in STRUCTURED_EXTS:
        return "yaml"
    return "text"


def detect_language(file_path: str) -> str:
    """根据扩展名推断编程语言。"""
    ext = Path(file_path).suffix.lower()
    return CODE_EXTS.get(ext, "text")


# ====================================================================== #
#  Markdown 切分
# ====================================================================== #

# 匹配 Markdown 标题（# ~ ######）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# 匹配代码块边界
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _split_markdown(content: str, file_path: str, max_chunk_chars: int = 2000) -> list[Chunk]:
    """按 ## 标题切分 Markdown，不切断代码块和表格。

    策略：
    1. 先找所有 ## 标题位置
    2. 每个 ## 标题作为一个 section
    3. section 太大时按段落二次切分
    4. 代码块（```...```）和表格保持完整
    """
    lines = content.split("\n")
    # 找到所有 ## 标题的行号（0-based）
    heading_positions: list[tuple[int, str, str]] = []  # (line_idx, level, text)
    in_code_block = False

    for i, line in enumerate(lines):
        if _CODE_FENCE_RE.match(line.strip()):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = _HEADING_RE.match(line.strip())
        if m:
            heading_positions.append((i, m.group(1), m.group(2).strip()))

    chunks: list[Chunk] = []
    chunk_idx = 0

    if not heading_positions:
        # 没有 ## 标题，整篇作为一个 chunk（或按段落切）
        return _split_plain_text(content, file_path, max_chunk_chars)

    # 处理第一个标题之前的内容（前言）
    first_h = heading_positions[0]
    if first_h[0] > 0:
        preamble = "\n".join(lines[:first_h[0]]).strip()
        if preamble:
            chunk_idx += 1
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="md",
                text=preamble,
                file_path=file_path,
                heading="(前言)",
                start_line=1,
                end_line=first_h[0],
            ))

    # 处理每个 section
    for idx, (line_idx, level, heading_text) in enumerate(heading_positions):
        start = line_idx
        end = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines).strip()

        if not section_text:
            continue

        # 如果 section 太长，按段落二次切分
        if len(section_text) > max_chunk_chars:
            sub_chunks = _split_section_by_paragraphs(
                section_text, file_path, heading_text, start + 1, chunk_idx, max_chunk_chars
            )
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)
        else:
            chunk_idx += 1
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="md",
                text=section_text,
                file_path=file_path,
                heading=heading_text,
                start_line=start + 1,
                end_line=end,
            ))

    return chunks


def _split_section_by_paragraphs(
    text: str, file_path: str, heading: str,
    base_line: int, start_idx: int, max_chars: int,
) -> list[Chunk]:
    """将超长 section 按段落二次切分，不切断代码块。"""
    # 按双换行切分，但合并代码块内的段落
    paragraphs = _smart_split_paragraphs(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buf_len = 0
    chunk_idx = start_idx
    current_line = base_line

    for para in paragraphs:
        para_len = len(para)
        if buf_len + para_len > max_chars and buffer:
            # 输出当前 buffer
            chunk_idx += 1
            chunk_text = "\n\n".join(buffer)
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="md",
                text=chunk_text,
                file_path=file_path,
                heading=heading,
                start_line=current_line,
                end_line=current_line + chunk_text.count("\n"),
            ))
            current_line += chunk_text.count("\n") + 2
            buffer = []
            buf_len = 0
        buffer.append(para)
        buf_len += para_len

    # 输出剩余 buffer
    if buffer:
        chunk_idx += 1
        chunk_text = "\n\n".join(buffer)
        chunks.append(Chunk(
            chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
            chunk_index=chunk_idx,
            chunk_type="md",
            text=chunk_text,
            file_path=file_path,
            heading=heading,
            start_line=current_line,
            end_line=current_line + chunk_text.count("\n"),
        ))

    return chunks


def _smart_split_paragraphs(text: str) -> list[str]:
    """按双换行切分段落，但保持代码块完整。"""
    in_fence = False
    current: list[str] = []
    result: list[str] = []

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            if not in_fence:
                # 代码块结束
                result.append("\n".join(current))
                current = []
            continue
        if in_fence:
            current.append(line)
            continue
        if line.strip() == "" and current:
            result.append("\n".join(current))
            current = []
        else:
            current.append(line)

    if current:
        result.append("\n".join(current))

    return [p for p in result if p.strip()]


# ====================================================================== #
#  代码切分（基于正则的函数/类边界）
# ====================================================================== #

# 匹配 Python 函数/类定义
_PY_FUNC_RE = re.compile(
    r"^(\s*)(def |class |async def )(\w+)\s*[\(:]",
    re.MULTILINE,
)

# 匹配 JavaScript/TypeScript 函数/类定义
_JS_FUNC_RE = re.compile(
    r"^(\s*)(function |class |const \w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)|"
    r"(?:export\s+)?(?:async\s+)?function\s+\w+)",
    re.MULTILINE,
)

# 匹配 Java 方法/类定义
_JAVA_FUNC_RE = re.compile(
    r"^(\s*)(public |private |protected |static |abstract |final |synchronized )*"
    r"(class |interface |enum |@\w+\s+)?[\w<>[\],\s]+\s+(\w+)\s*\([^)]*\)\s*(?:throws[\w\s,]+)?\s*\{",
    re.MULTILINE,
)

# 语言 → (正则, 缩进敏感的group_idx)
_LANG_PATTERNS: dict[str, re.Pattern] = {
    "python": _PY_FUNC_RE,
    "javascript": _JS_FUNC_RE,
    "typescript": _JS_FUNC_RE,
    "java": _JAVA_FUNC_RE,
    "go": re.compile(r"^func\s+(\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE),
    "rust": re.compile(r"^(\s*)(pub\s+)?(fn |struct |enum |trait |impl )", re.MULTILINE),
    "cpp": re.compile(r"^(\s*)([\w:]+\s+)?(\w+)\s*\([^)]*\)\s*(const\s*)?\{", re.MULTILINE),
    "c": re.compile(r"^(\s*)([\w*]+\s+)?(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE),
}


def _split_code(content: str, file_path: str, language: str,
                max_chunk_chars: int = 2000) -> list[Chunk]:
    """按函数/类边界切分代码。

    策略：
    1. 用语言特定的正则匹配函数/类定义
    2. 找到每个定义的位置
    3. 函数之间的内容（top-level import/comment）归入前一个 chunk
    4. 超大函数按逻辑块二次切分
    """
    pattern = _LANG_PATTERNS.get(language)
    if not pattern:
        # 未知语言，按段落切
        return _split_plain_text(content, file_path, max_chunk_chars)

    lines = content.split("\n")
    # 找到所有函数/类定义的行号（0-based）
    func_positions: list[tuple[int, str]] = []  # (line_idx, func_name)

    for match in pattern.finditer(content):
        line_idx = content[:match.start()].count("\n")
        # 提取函数/类名
        groups = match.groups()
        # 函数名通常是最后一个非空 group
        func_name = ""
        for g in reversed(groups):
            if g and g.strip() and not g.strip().startswith(("def ", "class ", "fn ", "function ")):
                func_name = g.strip()
                break
        if not func_name:
            func_name = match.group(0).split()[0] if match.group(0).split() else "unknown"
        func_positions.append((line_idx, func_name))

    if not func_positions:
        return _split_plain_text(content, file_path, max_chunk_chars)

    chunks: list[Chunk] = []
    chunk_idx = 0

    # 第一个函数之前的 top-level 内容
    first_func = func_positions[0]
    if first_func[0] > 0:
        preamble = "\n".join(lines[:first_func[0]]).strip()
        if preamble and not _is_all_imports_or_comments(preamble, language):
            chunk_idx += 1
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="code",
                text=preamble,
                file_path=file_path,
                heading=f"({language} top-level)",
                start_line=1,
                end_line=first_func[0],
            ))

    # 每个函数/类作为一个 chunk
    for idx, (line_idx, func_name) in enumerate(func_positions):
        start = line_idx
        end = func_positions[idx + 1][0] if idx + 1 < len(func_positions) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()

        if not section_text:
            continue

        if len(section_text) > max_chunk_chars:
            # 超大函数，按缩进块的逻辑边界切
            sub = _split_large_function(section_text, file_path, func_name, start + 1, chunk_idx, max_chunk_chars, language)
            chunks.extend(sub)
            chunk_idx += len(sub)
        else:
            chunk_idx += 1
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="code",
                text=section_text,
                file_path=file_path,
                heading=func_name,
                start_line=start + 1,
                end_line=end,
                metadata={"language": language},
            ))

    return chunks


def _is_all_imports_or_comments(text: str, language: str) -> bool:
    """检查文本是否全是 import/comment 行（可合并到下一个 chunk 的前言）。"""
    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "//", "/*", "*", "*/")):
            continue
        if language == "python" and (stripped.startswith(("import ", "from "))):
            continue
        if language in ("javascript", "typescript") and (stripped.startswith(("import ", "const ", "let ", "var "))):
            continue
        return False
    return True


def _split_large_function(
    text: str, file_path: str, func_name: str,
    base_line: int, start_idx: int, max_chars: int, language: str,
) -> list[Chunk]:
    """将超大函数按缩进块边界二次切分。"""
    lines = text.split("\n")
    # 找顶层缩进块（缩进 = 函数体基础缩进 + 1 级）
    base_indent = 999
    for line in lines[1:]:  # 跳过函数签名行
        stripped = line.rstrip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent < base_indent and indent > 0:
            base_indent = indent

    if base_indent == 999:
        base_indent = 4  # fallback

    chunks: list[Chunk] = []
    buffer_lines: list[str] = [lines[0]]  # 函数签名永远在第一个 chunk
    buf_len = len(lines[0])
    chunk_idx = start_idx
    current_line = base_line

    for line in lines[1:]:
        stripped = line.rstrip()
        indent = len(line) - len(line.lstrip()) if stripped else 0

        # 在顶层缩进处分块
        if stripped and indent <= base_indent and buf_len > max_chars * 0.3:
            chunk_idx += 1
            chunk_text = "\n".join(buffer_lines)
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="code",
                text=chunk_text,
                file_path=file_path,
                heading=f"{func_name} (part {len(chunks) + 1})",
                start_line=current_line,
                end_line=current_line + chunk_text.count("\n"),
                metadata={"language": language},
            ))
            current_line += chunk_text.count("\n") + 1
            buffer_lines = [f"# ... continuing {func_name} ..."]
            buf_len = len(buffer_lines[0])

        buffer_lines.append(line)
        buf_len += len(line) + 1

    # 输出剩余
    if len(buffer_lines) > 1:  # 有超过签名的内容
        chunk_idx += 1
        chunk_text = "\n".join(buffer_lines)
        chunks.append(Chunk(
            chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
            chunk_index=chunk_idx,
            chunk_type="code",
            text=chunk_text,
            file_path=file_path,
            heading=f"{func_name} (part {len(chunks) + 1})" if len(chunks) > 0 else func_name,
            start_line=current_line,
            end_line=current_line + chunk_text.count("\n"),
            metadata={"language": language},
        ))

    return chunks


# ====================================================================== #
#  纯文本 / 结构化文本切分
# ====================================================================== #

def _split_plain_text(content: str, file_path: str, max_chunk_chars: int = 2000) -> list[Chunk]:
    """按段落（双换行）切分纯文本。"""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buf_len = 0
    chunk_idx = 0
    start_line = 1

    for para in paragraphs:
        para_len = len(para)
        if buf_len + para_len > max_chunk_chars and buffer:
            chunk_idx += 1
            chunk_text = "\n\n".join(buffer)
            chunks.append(Chunk(
                chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
                chunk_index=chunk_idx,
                chunk_type="text",
                text=chunk_text,
                file_path=file_path,
                start_line=start_line,
                end_line=start_line + chunk_text.count("\n"),
            ))
            start_line += chunk_text.count("\n") + 2
            buffer = []
            buf_len = 0
        buffer.append(para)
        buf_len += para_len

    if buffer:
        chunk_idx += 1
        chunk_text = "\n\n".join(buffer)
        chunks.append(Chunk(
            chunk_id=f"{Path(file_path).stem}_c{chunk_idx}",
            chunk_index=chunk_idx,
            chunk_type="text",
            text=chunk_text,
            file_path=file_path,
            start_line=start_line,
            end_line=start_line + chunk_text.count("\n"),
        ))

    return chunks


def _split_structured(content: str, file_path: str) -> list[Chunk]:
    """结构化文本（YAML/JSON/TOML）整文件单 chunk。"""
    name = Path(file_path).stem
    return [Chunk(
        chunk_id=f"{name}_c1",
        chunk_index=1,
        chunk_type="yaml",
        text=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
    )]


# ====================================================================== #
#  主入口
# ====================================================================== #

def chunk_file(file_path: str, content: str | None = None,
               max_chunk_chars: int = 2000) -> list[Chunk]:
    """对单个文件做智能切分。

    Args:
        file_path: 文件路径（用于检测类型和提取元数据）
        content: 文件内容（若为 None 则从 file_path 读取）
        max_chunk_chars: 单个 chunk 最大字符数

    Returns:
        Chunk 列表
    """
    if content is None:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")

    if not content.strip():
        return []

    ftype = detect_type(file_path)

    if ftype == "md":
        return _split_markdown(content, file_path, max_chunk_chars)
    elif ftype == "code":
        language = detect_language(file_path)
        return _split_code(content, file_path, language, max_chunk_chars)
    elif ftype == "yaml":
        return _split_structured(content, file_path)
    else:
        return _split_plain_text(content, file_path, max_chunk_chars)


def chunk_text(content: str, title: str = "document",
               chunk_type: str = "text",
               max_chunk_chars: int = 2000) -> list[Chunk]:
    """对内存中的文本做智能切分（无文件路径时使用）。

    Args:
        content: 文本内容
        title: 用于生成 chunk_id 的标题
        chunk_type: 类型提示 ("md", "code", "text", "yaml")
        max_chunk_chars: 单个 chunk 最大字符数

    Returns:
        Chunk 列表
    """
    # 用 fake 路径来复用切分逻辑
    ext_map = {"md": ".md", "code": ".py", "yaml": ".yaml", "text": ".txt"}
    fake_path = f"{title}{ext_map.get(chunk_type, '.txt')}"

    if chunk_type == "md":
        return _split_markdown(content, fake_path, max_chunk_chars)
    elif chunk_type == "code":
        return _split_code(content, fake_path, "python", max_chunk_chars)
    elif chunk_type == "yaml":
        return _split_structured(content, fake_path)
    else:
        return _split_plain_text(content, fake_path, max_chunk_chars)


# ====================================================================== #
#  zvec 格式转换
# ====================================================================== #

def chunks_to_zvec(chunks: list[Chunk], document_id: str, title: str) -> list[dict]:
    """将 Chunk 列表转换为 zvec 入库格式。

    Args:
        chunks: Chunk 列表
        document_id: SurrealDB 文档 ID（如 "document:transformer"）
        title: 文档标题

    Returns:
        zvec ingest_chunks 所需格式的 dict 列表
    """
    return [
        {
            "id": ch.chunk_id,
            "text": ch.text,
            "fields": {
                "document_id": document_id,
                "title": title,
                "heading": ch.heading or title,
                "content": ch.text,
            },
        }
        for ch in chunks
    ]
