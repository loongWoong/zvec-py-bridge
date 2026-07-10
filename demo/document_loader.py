"""文档加载器 — 支持 Markdown、DOCX、PDF、XLSX 格式的文档解析与切分。

每种格式有独立的解析函数，按文件扩展名分发。解析库延迟导入，
缺库时返回清晰错误而非崩溃。纯文本文件作为 fallback。
"""
from __future__ import annotations

import io
import re


def parse_file(filename: str, content: bytes) -> dict:
    """解析上传的文件，返回 {title, content, chunks, source_type, source_file}。

    chunks: [{heading, content}, ...]
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    title = filename.rsplit(".", 1)[0] if "." in filename else filename

    if ext == "md":
        text = content.decode("utf-8", errors="replace")
        chunks = parse_markdown(text)
        source_type = "markdown"
    elif ext == "docx":
        chunks = parse_docx(content)
        source_type = "docx"
    elif ext == "pdf":
        chunks = parse_pdf(content)
        source_type = "pdf"
    elif ext in ("xlsx", "xls"):
        chunks = parse_xlsx(content)
        source_type = "xlsx"
    else:
        # 纯文本 fallback
        text = content.decode("utf-8", errors="replace")
        chunks = [{"heading": "", "content": text.strip()}]
        source_type = "text"

    full_content = "\n\n".join(c["content"] for c in chunks)
    return {
        "title": title,
        "content": full_content,
        "chunks": chunks,
        "source_type": source_type,
        "source_file": filename,
    }


# ====================================================================== #
#  Markdown
# ====================================================================== #
def parse_markdown(text: str) -> list[dict]:
    """按标题切分 Markdown，保留 heading；长段进一步按句号拆分。"""
    chunks: list[dict] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        if re.match(r"^#{1,6}\s", line):
            # 保存上一个 section
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({"heading": current_heading, "content": content})
            current_heading = re.sub(r"^#{1,6}\s*", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # 最后一个 section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({"heading": current_heading, "content": content})

    if not chunks:
        chunks = [{"heading": "", "content": text.strip()}]

    # 拆分过长的 chunk
    result: list[dict] = []
    for chunk in chunks:
        if len(chunk["content"]) > 500:
            result.extend(split_long_chunk(chunk))
        else:
            result.append(chunk)
    return result


# ====================================================================== #
#  DOCX
# ====================================================================== #
def parse_docx(content: bytes) -> list[dict]:
    """用 python-docx 提取段落，识别 Heading 样式作为 heading。"""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        return [{"heading": "", "content": "(缺少 python-docx 依赖，无法解析 DOCX)"}]

    doc = DocxDocument(io.BytesIO(content))
    chunks: list[dict] = []
    current_heading = ""
    current_lines: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            if current_lines:
                chunks.append({"heading": current_heading, "content": "\n".join(current_lines)})
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)

    if current_lines:
        chunks.append({"heading": current_heading, "content": "\n".join(current_lines)})

    if not chunks:
        chunks = [{"heading": "", "content": "(空文档)"}]

    # 拆分过长的 chunk
    result: list[dict] = []
    for chunk in chunks:
        if len(chunk["content"]) > 500:
            result.extend(split_long_chunk(chunk))
        else:
            result.append(chunk)
    return result


# ====================================================================== #
#  PDF
# ====================================================================== #
def parse_pdf(content: bytes) -> list[dict]:
    """用 pypdf 逐页提取文本，每页一个 chunk。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        return [{"heading": "", "content": "(缺少 pypdf 依赖，无法解析 PDF)"}]

    reader = PdfReader(io.BytesIO(content))
    chunks: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append({"heading": f"第{i + 1}页", "content": text.strip()})

    if not chunks:
        chunks = [{"heading": "", "content": "(无法提取文本，可能是扫描件)"}]

    # 拆分过长的 chunk
    result: list[dict] = []
    for chunk in chunks:
        if len(chunk["content"]) > 500:
            result.extend(split_long_chunk(chunk))
        else:
            result.append(chunk)
    return result


# ====================================================================== #
#  XLSX
# ====================================================================== #
def parse_xlsx(content: bytes) -> list[dict]:
    """用 openpyxl 逐工作表提取，行用 | 连接。"""
    try:
        from openpyxl import load_workbook
    except ImportError:
        return [{"heading": "", "content": "(缺少 openpyxl 依赖，无法解析 XLSX)"}]

    wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    chunks: list[dict] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            content_text = "\n".join(rows)
            chunks.append({"heading": f"工作表: {sheet_name}", "content": content_text})

    if not chunks:
        chunks = [{"heading": "", "content": "(空表格)"}]

    # 拆分过长的 chunk
    result: list[dict] = []
    for chunk in chunks:
        if len(chunk["content"]) > 500:
            result.extend(split_long_chunk(chunk))
        else:
            result.append(chunk)
    return result


# ====================================================================== #
#  通用切分
# ====================================================================== #
def split_long_chunk(chunk: dict, max_chars: int = 500) -> list[dict]:
    """将过长的 chunk 按句号/换行切分为更小的片段。"""
    content = chunk["content"]
    heading = chunk["heading"]
    # 按中英文句号、问号、换行切分
    sentences = re.split(r"(?<=[。！？\n.!?])", content)
    result: list[dict] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            result.append({"heading": heading, "content": current.strip()})
            current = sent
        else:
            current += sent
    if current.strip():
        result.append({"heading": heading, "content": current.strip()})
    return result if result else [chunk]
