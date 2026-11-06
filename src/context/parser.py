"""PSI-style project parser: file scan + symbol extraction.

Python files are parsed with the real `ast` module; other languages get a
pragmatic regex-based symbol scan. Everything is bounded (file count, file
size, symbols per file) so huge repos can't wedge the server.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

MAX_FILE_SIZE = 750_000
MAX_FILES = 1500
MAX_SYMBOLS_PER_FILE = 200
MAX_VIEW_LINES = 3000

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", ".next",
    ".nuxt", "target", ".idea", ".gradle", ".jeet", "database", "coverage",
    ".tox", "out", "bin", "obj", ".cache", "Unselected files",
}

LANGUAGE_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".cs": "csharp", ".swift": "swift",
    ".md": "markdown", ".rst": "markdown", ".txt": "text",
    ".json": "json", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".html": "html", ".css": "css", ".scss": "css", ".sql": "sql", ".sh": "shell",
    ".dockerfile": "docker", ".xml": "xml", ".vue": "vue", ".svelte": "svelte",
}
TEXT_EXTS = set(LANGUAGE_BY_EXT) | {".gitignore", ".editorconfig", ".cfg", ".ini"}

_GENERIC_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:public\s+|private\s+|internal\s+)?"
    r"(?:async\s+)?(function\*?|func|def|class|interface|struct|enum|trait|type|const|let|var)\s+"
    r"`?([A-Za-z_][A-Za-z0-9_.]*)`?", re.M)
_JS_ARROW_RE = re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*(?:async\s*)?\([^)]*\)\s*=>", re.M)
_MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.M)


def language_for(path: Path) -> str:
    if path.name.lower() == "dockerfile":
        return "docker"
    return LANGUAGE_BY_EXT.get(path.suffix.lower(), "text")


def extract_symbols(path: Path, language: str | None = None) -> list[dict]:
    language = language or language_for(path)
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    symbols: list[dict] = []
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                symbols.append({"name": node.name, "kind": "function",
                                "line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno),
                                "detail": f"({', '.join(args)})"})
            elif isinstance(node, ast.ClassDef):
                symbols.append({"name": node.name, "kind": "class",
                                "line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno),
                                "detail": f"class {node.name}"})
    elif language == "markdown":
        for match in _MD_HEADING_RE.finditer(source):
            symbols.append({"name": match.group(2).strip(), "kind": "section",
                            "line": source.count("\n", 0, match.start()) + 1,
                            "end_line": None, "detail": "#" * len(match.group(1))})
    else:
        for match in _GENERIC_SYMBOL_RE.finditer(source):
            symbols.append({"name": match.group(2), "kind": match.group(1).replace("*", ""),
                            "line": source.count("\n", 0, match.start()) + 1,
                            "end_line": None, "detail": ""})
        if language in ("javascript", "typescript", "vue", "svelte"):
            for match in _JS_ARROW_RE.finditer(source):
                symbols.append({"name": match.group(1), "kind": "function",
                                "line": source.count("\n", 0, match.start()) + 1,
                                "end_line": None, "detail": "arrow"})
    symbols.sort(key=lambda s: s["line"])
    return symbols[:MAX_SYMBOLS_PER_FILE]


def _iter_project_files(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".git")]
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() not in TEXT_EXTS and path.name.lower() != "dockerfile":
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_SIZE:
                continue
            yield path, size
            count += 1
            if count >= MAX_FILES:
                return


def scan_project(root: Path | str) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        return {"root": str(root), "count": 0, "total_bytes": 0, "files": [],
                "error": f"Not a directory: {root}"}
    files: list[dict] = []
    total_bytes = 0
    for path, size in _iter_project_files(root):
        rel = path.relative_to(root).as_posix()
        language = language_for(path)
        symbols = extract_symbols(path, language) if language in (
            "python", "javascript", "typescript", "markdown", "go", "rust", "java",
            "c", "cpp", "ruby", "php", "csharp", "kotlin", "swift", "shell") else []
        files.append({
            "path": rel,
            "size": size,
            "language": language,
            "symbols": symbols,
            "symbol_count": len(symbols),
        })
        total_bytes += size
    files.sort(key=lambda f: f["path"])
    return {"root": str(root), "count": len(files), "total_bytes": total_bytes, "files": files}


def safe_resolve(root: Path | str, relative: str) -> Path:
    """Resolve `relative` inside `root`; refuse path escapes."""
    root = Path(root).resolve()
    candidate = (root / relative).resolve() if not Path(relative).is_absolute() else Path(relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes project root: {relative!r}")
    return candidate


def read_file_view(root: Path | str, relative: str, start: int = 1,
                   end: int | None = None) -> dict:
    target = safe_resolve(root, relative)
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {relative}")
    text = target.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    total = len(lines)
    start = max(1, int(start or 1))
    end = min(total, int(end)) if end else min(total, start + MAX_VIEW_LINES - 1)
    language = language_for(target)
    view = [{"n": i + 1, "text": lines[i]} for i in range(start - 1, end)]
    return {
        "path": target.relative_to(Path(root).resolve()).as_posix() if target != Path(root).resolve() else target.name,
        "language": language,
        "total_lines": total,
        "start": start,
        "end": end,
        "truncated": end < total,
        "lines": view,
        "symbols": extract_symbols(target, language),
    }
