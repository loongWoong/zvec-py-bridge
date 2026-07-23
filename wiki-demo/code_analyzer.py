"""code_analyzer — 轻量级代码结构化索引（对应 AI推理引擎.md Step 1 / Step 4 索引 D / 策略5）。

不依赖 tree-sitter / LSP / ctags 等重型依赖，采用「正则 + 缩进/括号启发式」提取：

  - 函数 / 方法 / 类（名称、签名、行号范围）
  - 调用关系（caller → callee 调用链）
  - 导入（import）与符号表
  - 文件名 / 语言

目标：为「索引 D：结构化索引（代码专用）/ AST / 调用链 / 文件树 / 符号表」提供
可落库的轻量替代实现，支撑策略5 [调用链] 的精确函数/类/调用关系定位，
并产出带 file_path / function_name 的结构化命中，供 Re-Rank 的结构相关性因子使用。

纯标准库，零外部依赖。
"""
from __future__ import annotations

import os
import re

# 扩展名 → 语言
LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c", ".h": "c", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".m": "objc",
}

# 不应被当作函数/方法调用的关键字
_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "await", "yield",
    "function", "class", "def", "import", "from", "export", "default",
    "new", "typeof", "instanceof", "throw", "else", "do", "try", "finally",
    "with", "var", "let", "const", "async", "print", "assert", "sizeof",
    "len", "range", "int", "float", "str", "bool", "list", "dict", "set",
    "map", "void", "public", "private", "protected", "static", "final",
    "ifdef", "ifndef", "elif", "foreach", "in", "not", "and", "or",
}

# ── 各语言声明正则（命名组 name + 缩进组 indent）──
_DECL_PATTERNS = {
    "python": [
        (re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:"), "class"),
        (re.compile(r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\("), "function"),
    ],
    "javascript": [
        (re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_$]\w*)\s*(?:extends\s+[\w.]+)?"), "class"),
        (re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$]\w*)\s*\("), "function"),
        (re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"), "function"),
        (re.compile(r"^(?P<indent>\s*)(?:async\s+)?(?P<name>[A-Za-z_$]\w*)\s*\([^)]*\)\s*\{"), "method"),
    ],
    "typescript": [
        (re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_$]\w*)\s*(?:extends\s+[\w.]+)?"), "class"),
        (re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$]\w*)\s*\("), "function"),
        (re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"), "function"),
        (re.compile(r"^(?P<indent>\s*)(?:async\s+)?(?P<name>[A-Za-z_$]\w*)\s*\([^)]*\)\s*\{"), "method"),
    ],
}
# 其它语言回退到通用（类/函数/方法）模式
_GENERIC_DECL = [
    (re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_$]\w*)"), "class"),
    (re.compile(r"^(?P<indent>\s*)(?:function|func|def|sub)\s+(?P<name>[A-Za-z_$]\w*)\s*\("), "function"),
    (re.compile(r"^(?P<indent>\s*)(?:const|let|var|val|fun)\s+(?P<name>[A-Za-z_$]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"), "function"),
    (re.compile(r"^(?P<indent>\s*)(?P<name>[A-Za-z_$]\w*)\s*\([^)]*\)\s*\{"), "method"),
]

# 调用提取：匹配 IDENT( ，但前面不能是 . 或字母数字（避免 self.foo / a.b() 误抓）
_CALL_RE = re.compile(r"(?<![.\w$])([A-Za-z_]\w*)\s*\(")
_IMPORT_PY = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))")
_IMPORT_JS = re.compile(r"""import\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]""")
_IMPORT_REQUIRE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")


def detect_language(filename: str) -> str:
    """根据文件名推断语言（未知返回 'text'）。"""
    ext = os.path.splitext(filename)[1].lower()
    return LANG_BY_EXT.get(ext, "text")


def _scan_declarations(lines: list[str], lang: str) -> list[tuple]:
    """扫描声明，返回 [(line_idx, name, kind, indent), ...]。"""
    patterns = _DECL_PATTERNS.get(lang, _GENERIC_DECL)
    decls: list[tuple] = []
    for i, line in enumerate(lines):
        for pat, kind in patterns:
            m = pat.match(line)
            if m:
                name = m.group("name")
                if name in _KEYWORDS:
                    continue
                indent = len(m.group("indent"))
                decls.append((i, name, kind, indent))
                break
    return decls


def _block_end(decls: list[tuple], idx: int, n_lines: int, lang: str) -> int:
    """估算声明块结束行（用于截取 signature + 调用体）。"""
    i, _name, _kind, indent = decls[idx]
    end = min(i + 60, n_lines)  # 最多看 60 行
    if lang == "python":
        for j in range(idx + 1, len(decls)):
            if decls[j][3] <= indent:
                end = decls[j][0]
                break
    elif idx + 1 < len(decls):
        end = min(decls[idx + 1][0], end)
    return end


def _extract_calls(snippet: str, self_name: str) -> list[str]:
    """从一段代码中提取调用（排除关键字与自身声明名）。"""
    calls: list[str] = []
    for m in _CALL_RE.finditer(snippet):
        callee = m.group(1)
        if callee in _KEYWORDS or callee == self_name:
            continue
        calls.append(callee)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _extract_imports(lines: list[str], lang: str) -> list[str]:
    modules: list[str] = []
    for line in lines:
        if lang == "python":
            m = _IMPORT_PY.match(line)
            if m:
                if m.group(1):
                    modules.append(m.group(1))
                elif m.group(2):
                    for part in re.split(r"[,\s]+", m.group(2)):
                        part = part.strip()
                        if part and part not in ("import",):
                            modules.append(part.split(".")[0])
        else:
            for pat in (_IMPORT_JS, _IMPORT_REQUIRE):
                m = pat.search(line)
                if m:
                    modules.append(m.group(1))
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for mod in modules:
        if mod not in seen:
            seen.add(mod)
            out.append(mod)
    return out


def analyze_code(content: str, filename: str = "") -> dict:
    """解析单段代码，返回结构化索引。

    Returns:
        {
          "filename": str,
          "language": str,
          "functions": [{name, kind, line, end_line, signature, calls:[...]}],
          "classes":   [{name, line, end_line, bases:[...], methods:[...]}],
          "symbols":   [所有 functions+classes 扁平化],
          "calls":     [{caller, callee}],   # 调用链边
          "imports":   [str],
        }
    """
    lang = detect_language(filename) if filename else "text"
    lines = content.splitlines()
    decls = _scan_declarations(lines, lang)

    functions: list[dict] = []
    classes: list[dict] = []
    symbols: list[dict] = []
    calls: list[dict] = []
    class_stack: list[str] = []  # 当前所属类名（按缩进入栈，近似）

    for idx, (i, name, kind, indent) in enumerate(decls):
        end = _block_end(decls, idx, len(lines), lang)
        snippet = "\n".join(lines[i:end])
        signature = lines[i].strip()
        callees = _extract_calls(snippet, name)

        if kind == "class":
            bases = []
            bm = re.search(r"class\s+\w+\s*\(([^)]*)\)", lines[i])
            if bm and bm.group(1).strip():
                bases = [b.strip().split(".")[-1] for b in bm.group(1).split(",") if b.strip()]
            entry = {
                "name": name, "kind": "class", "line": i + 1, "end_line": end,
                "signature": signature, "bases": bases, "methods": [],
                "calls": callees,
            }
            classes.append(entry)
            symbols.append(entry)
            # 类内调用：caller 记为类名
            for c in callees:
                calls.append({"caller": name, "callee": c})
        else:
            entry = {
                "name": name, "kind": kind, "line": i + 1, "end_line": end,
                "signature": signature, "calls": callees,
            }
            functions.append(entry)
            symbols.append(entry)
            for c in callees:
                calls.append({"caller": name, "callee": c})

    return {
        "filename": filename,
        "language": lang,
        "functions": functions,
        "classes": classes,
        "symbols": symbols,
        "calls": calls,
        "imports": _extract_imports(lines, lang),
    }


def analyze_file(path: str) -> dict | None:
    """读取文件并解析；失败返回 None。"""
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None
    analysis = analyze_code(content, os.path.basename(p))
    analysis["path"] = p
    return analysis


def _symbol_score(name: str, token: str) -> float:
    """单个 token 对符号名的匹配分。"""
    name_l = name.lower()
    token_l = token.lower()
    if name_l == token_l:
        return 1.0
    if name_l.startswith(token_l):
        return 0.8
    if token_l in name_l:
        return 0.6
    # 驼峰/下划线拆词
    parts = re.split(r"[_\s]+|(?<=[a-z])(?=[A-Z])", name_l)
    if token_l in parts:
        return 0.5
    return 0.0


def search_symbols(analysis: dict, query: str, max_hits: int = 20) -> list[dict]:
    """在符号表中检索与 query 相关的符号，返回按分数排序的符号列表。

    匹配层级：精确名 > 前缀 > 子串 > 拆词；若符号名无命中，则回退到
    snippet 子串 grep（对应策略2 grep 行为），保证至少能定位到相关代码。
    """
    tokens = [t for t in re.findall(r"[A-Za-z_]\w*", query.lower()) if len(t) >= 2]
    if not tokens:
        tokens = [query.lower()] if query.strip() else []

    scored: list[dict] = []
    for sym in analysis.get("symbols", []):
        best = 0.0
        matched = ""
        for tok in tokens:
            s = _symbol_score(sym["name"], tok)
            if s > best:
                best = s
                matched = tok
        if best > 0:
            scored.append({**sym, "score": best, "matched": matched})

    if not scored:
        # 回退：snippet 子串 grep
        q = query.strip().lower()
        if q:
            for sym in analysis.get("symbols", []):
                snippet = sym.get("signature", "").lower()
                if q in snippet:
                    scored.append({**sym, "score": 0.4, "matched": q})

    scored.sort(key=lambda x: (-x["score"], x["line"]))
    return scored[:max_hits]


def format_symbol_hits(analysis: dict, query: str, max_hits: int = 20) -> str:
    """将命中符号格式化为可喂给 LLM 的紧凑文本索引。"""
    matches = search_symbols(analysis, query, max_hits)
    if not matches:
        return ""
    fname = analysis.get("filename") or analysis.get("path") or ""
    lines_out = [f"文件: {fname} (语言: {analysis.get('language', '')})"]
    for m in matches:
        lines_out.append(f"  L{m['line']}  {m['kind']} {m['name']}")
        if m.get("signature"):
            lines_out.append(f"      {m['signature']}")
        if m.get("calls"):
            lines_out.append(f"      调用: {', '.join(m['calls'][:12])}")
    return "\n".join(lines_out)


def grep_in_analysis(analysis: dict, pattern: str, max_hits: int = 20) -> list[dict]:
    """在分析的原始文件中按正则匹配行（对应策略2 grep，返回 行号+内容）。"""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return []
    # 需要原始文本：analyze_file 场景才有；analyze_code 场景用 symbols 的 signature
    hits: list[dict] = []
    path = analysis.get("path")
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for ln, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append({"line": ln, "content": line.rstrip("\n")})
                        if len(hits) >= max_hits:
                            break
        except Exception:
            pass
    return hits


if __name__ == "__main__":
    # 简单自测：解析本文件自身
    a = analyze_file(__file__)
    if a:
        print(f"language={a['language']} symbols={len(a['symbols'])} calls={len(a['calls'])}")
        for s in a["symbols"][:10]:
            print(f"  L{s['line']} {s['kind']} {s['name']} calls={s['calls'][:5]}")
