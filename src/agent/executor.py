"""Sandboxed tool belt for the Junie-clone agent.

Every tool is confined to the project root (path escapes are rejected) and
commands run WITHOUT a shell against a strict allowlist.
"""
from __future__ import annotations

import difflib
import shlex
import subprocess
from pathlib import Path

from ..context.parser import safe_resolve

ALLOWED_COMMAND_PREFIXES = [
    "pytest", "python -m pytest", "python3 -m pytest",
    "python -m compileall", "python3 -m compileall",
    "python -m py_compile", "python3 -m py_compile",
    "npm test", "npm run test", "npm run lint", "npm run build",
    "ls", "ls -la", "git status", "git diff", "git log", "git show",
    "cat", "head", "tail", "wc", "grep", "find",
]
COMMAND_TIMEOUT = 120
MAX_OUTPUT_CHARS = 4000
MAX_WRITE_BYTES = 1_000_000


class ToolError(Exception):
    pass


def list_dir(root: Path, relative: str = ".") -> list[dict]:
    target = safe_resolve(root, relative)
    if not target.is_dir():
        raise ToolError(f"Not a directory: {relative}")
    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:300]:
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else None,
        })
    return entries


def read_file(root: Path, relative: str, max_bytes: int = 200_000) -> dict:
    target = safe_resolve(root, relative)
    if not target.is_file():
        raise ToolError(f"No such file: {relative}")
    raw = target.read_bytes()[:max_bytes]
    return {"path": relative, "content": raw.decode("utf-8", errors="ignore"),
            "truncated": target.stat().st_size > max_bytes}


def write_file(root: Path, relative: str, content: str) -> dict:
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        raise ToolError(f"Refusing to write > {MAX_WRITE_BYTES} bytes.")
    target = safe_resolve(root, relative)
    existed = target.exists()
    old = target.read_text(encoding="utf-8", errors="ignore") if existed else ""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    diff_lines = list(difflib.unified_diff(old.splitlines(), content.splitlines(),
                                           fromfile=f"a/{relative}", tofile=f"b/{relative}", lineterm=""))
    return {"path": relative, "created": not existed,
            "bytes": len(content.encode("utf-8")),
            "diff": "\n".join(diff_lines[:400])}


def _command_allowed(cmd: str) -> bool:
    normalized = " ".join(shlex.split(cmd))
    return any(normalized == prefix or normalized.startswith(prefix + " ")
               for prefix in ALLOWED_COMMAND_PREFIXES)


def run_command(root: Path, cmd: str, timeout: int = COMMAND_TIMEOUT) -> dict:
    if not _command_allowed(cmd):
        raise ToolError(
            f"Command not in allowlist: {cmd!r}. "
            f"Allowed prefixes: {', '.join(sorted(set(ALLOWED_COMMAND_PREFIXES)))}")
    try:
        proc = subprocess.run(shlex.split(cmd), cwd=str(root), capture_output=True,
                              text=True, timeout=timeout, shell=False)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"Command timed out after {timeout}s: {cmd}") from exc
    except FileNotFoundError as exc:
        raise ToolError(f"Executable not found for: {cmd}") from exc
    stdout = (proc.stdout or "")[-MAX_OUTPUT_CHARS:]
    stderr = (proc.stderr or "")[-MAX_OUTPUT_CHARS:]
    return {"command": cmd, "returncode": proc.returncode,
            "stdout": stdout, "stderr": stderr, "ok": proc.returncode == 0}
