"""Shadow — Private AI Assistant · FastAPI server.

Run:  python src/main.py            (or)  uvicorn src.main:app --port 8000
UI:   http://localhost:8000
"""
from __future__ import annotations

import ast
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src import db  # noqa: E402
from src.agent import agent as agent_module  # noqa: E402
from src.config import load_config, public_settings, save_local_config, KEY_NAMES  # noqa: E402
from src.context import indexer, parser  # noqa: E402
from src.models import wrapper  # noqa: E402

STATIC_DIR = ROOT / "static"

app = FastAPI(title="Shadow — Private AI Assistant", version="1.1.0")

_scan_cache: dict[str, tuple[float, dict]] = {}


@app.exception_handler(wrapper.ModelError)
async def model_error_handler(_request, exc: wrapper.ModelError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# ── Health & models ──────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "app": "shadow", "time": db.now_iso()}


@app.get("/api/models")
def list_models() -> dict:
    cfg = load_config()
    return {"models": wrapper.model_availability(cfg), "default_model": cfg.get("default_model"),
            "offline_mode": bool(cfg.get("offline_mode"))}


class DefaultModelIn(BaseModel):
    key: str


@app.post("/api/models/default")
def set_default_model(payload: DefaultModelIn) -> dict:
    cfg = load_config()
    if payload.key not in (cfg.get("models") or {}):
        raise HTTPException(400, f"Unknown model: {payload.key}")
    save_local_config({"default_model": payload.key})
    return {"ok": True, "default_model": payload.key}


# ── Settings ─────────────────────────────────────────────────────

class SettingsIn(BaseModel):
    project_path: str | None = None
    offline_mode: bool | None = None
    default_model: str | None = None
    monthly_budget: float | None = None
    warning_threshold: float | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    api_keys: dict[str, str] | None = None   # only non-empty values are written
    clear_keys: list[str] | None = None      # names to wipe back to empty


@app.get("/api/settings")
def get_settings() -> dict:
    return public_settings(load_config())


@app.post("/api/settings")
def update_settings(payload: SettingsIn) -> dict:
    updates: dict = {}
    if payload.project_path is not None:
        updates["project_path"] = payload.project_path.strip() or "."
    if payload.offline_mode is not None:
        updates["offline_mode"] = bool(payload.offline_mode)
    if payload.default_model is not None:
        updates["default_model"] = payload.default_model
    credits: dict = {}
    if payload.monthly_budget is not None:
        credits["monthly_budget"] = max(0.0, float(payload.monthly_budget))
    if payload.warning_threshold is not None:
        credits["warning_threshold"] = max(0.0, float(payload.warning_threshold))
    if credits:
        updates["credits"] = credits
    ollama: dict = {}
    if payload.ollama_base_url is not None:
        ollama["base_url"] = payload.ollama_base_url.strip()
    if payload.ollama_model is not None:
        ollama["model"] = payload.ollama_model.strip()
    if ollama:
        updates["ollama"] = ollama

    if payload.api_keys:
        keys = updates.setdefault("api_keys", {})
        for k, v in payload.api_keys.items():
            if k in KEY_NAMES and v and v.strip():
                keys[k] = v.strip()
    if payload.clear_keys:
        keys = updates.setdefault("api_keys", {})
        for k in payload.clear_keys:
            if k in KEY_NAMES:
                keys[k] = ""
    if updates:
        save_local_config(updates)
    _scan_cache.clear()
    return public_settings(load_config())


# ── Chat ─────────────────────────────────────────────────────────

class OpenFileIn(BaseModel):
    path: str
    line: int | None = None


class ChatIn(BaseModel):
    session_id: str | None = None
    model: str | None = None
    message: str
    include_context: bool = True
    include_open_files: bool = True
    open_files: list[OpenFileIn] = []


_SYSTEM = ("You are Shadow, a private IDE copilot. Answer in markdown, be concise and code-first. "
           "Prefer concrete code over essays. Never invent file paths — use the provided context.")


def _open_file_blocks(root: Path, open_files: list[OpenFileIn]) -> str:
    blocks = []
    for entry in open_files[:4]:
        try:
            target = parser.safe_resolve(root, entry.path)
        except ValueError:
            continue
        if not target.is_file():
            continue
        language = parser.language_for(target)
        try:
            lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        cursor = entry.line or 0
        header = f"[Open file: {entry.path}] [Language: {language}]" + (f" [Cursor line: {cursor}]" if cursor else "")
        body, budget = [], 0
        for i, text in enumerate(lines[:400]):
            budget += len(text) + 8
            if budget > 6000:
                body.append(f"  … ({len(lines) - i} more lines)")
                break
            marker = ">>" if (i + 1) == cursor else "  "
            body.append(f"{marker}{i + 1:>5}| {text}")
        blocks.append(header + "\n" + "\n".join(body))
    return "\n\n".join(blocks)


@app.post("/api/chat")
def chat(payload: ChatIn) -> dict:
    cfg = load_config()
    root = Path(cfg["project_path"])
    model_key = payload.model or cfg.get("default_model", "demo")
    message = payload.message.strip()
    if not message:
        raise HTTPException(400, "Empty message.")

    session_id = payload.session_id
    if session_id and db.get_session(session_id) is None:
        session_id = None
    if not session_id:
        session_id = db.create_session(title=message[:80], model=model_key)

    parts: list[str] = []
    if payload.include_context:
        try:
            hits = indexer.search(message, root, k=4)
        except Exception:
            hits = []
        if hits:
            snippet_lines = ["Relevant snippets from the project (semantic search):"]
            for hit in hits:
                snippet_lines.append(f"- {hit['path']} (score {hit['score']})")
                for snip in hit["snippet"]:
                    snippet_lines.append(f"    L{snip['n']}: {snip['text']}")
            parts.append("\n".join(snippet_lines))
    if payload.include_open_files and payload.open_files:
        blocks = _open_file_blocks(root, payload.open_files)
        if blocks:
            parts.append(blocks)

    history = db.list_messages(session_id, limit=12)
    if history:
        hist = []
        for msg in history:
            hist.append(f"{msg['role'].capitalize()}: {msg['content'][:1500]}")
        parts.append("Conversation so far:\n" + "\n".join(hist))

    parts.append(f"User request:\n{message}")
    prompt = "\n\n".join(parts)

    db.add_message(session_id, "user", message, model=model_key)
    result = wrapper.call_model(model_key, prompt, system_prompt=_SYSTEM, kind="chat", max_tokens=4096)
    db.add_message(session_id, "assistant", result["text"], model=result["model_key"],
                   prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"],
                   cost=result["cost"], latency_ms=result["latency_ms"])

    return {
        "session_id": session_id,
        "message": {"role": "assistant", "content": result["text"],
                    "model": result["model_key"],
                    "model_label": result["model_label"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "cost": result["cost"],
                    "latency_ms": result["latency_ms"],
                    "free": result["free"],
                    "notice": result.get("notice")},
        "credits": db.usage_summary(),
    }


@app.get("/api/sessions")
def sessions() -> dict:
    return {"sessions": db.list_sessions()}


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: str) -> dict:
    if db.get_session(session_id) is None:
        raise HTTPException(404, "Session not found.")
    return {"messages": db.list_messages(session_id)}


@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str) -> dict:
    return {"ok": db.delete_session(session_id)}


# ── Credits ──────────────────────────────────────────────────────

@app.get("/api/credits")
def credits() -> dict:
    return db.usage_summary()


# ── Context ──────────────────────────────────────────────────────

def _project_scan(root: Path) -> dict:
    key = str(root)
    cached = _scan_cache.get(key)
    if cached and time.time() - cached[0] < 5.0:
        return cached[1]
    data = parser.scan_project(root)
    _scan_cache[key] = (time.time(), data)
    return data


@app.get("/api/context/files")
def context_files() -> dict:
    cfg = load_config()
    return _project_scan(Path(cfg["project_path"]))


@app.get("/api/context/file")
def context_file(path: str, start: int = 1, end: int | None = None) -> dict:
    cfg = load_config()
    root = Path(cfg["project_path"])
    try:
        return parser.read_file_view(root, path, start, end)
    except ValueError as exc:
        raise HTTPException(403, str(exc))
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {path}")


@app.get("/api/context/search")
def context_search(q: str, k: int = 6) -> dict:
    cfg = load_config()
    root = Path(cfg["project_path"])
    return {"query": q, "results": indexer.search(q, root, k=min(max(k, 1), 20))}


@app.get("/api/context/status")
def context_status() -> dict:
    cfg = load_config()
    return indexer.index_status(Path(cfg["project_path"]))


@app.post("/api/context/reindex")
def context_reindex() -> dict:
    cfg = load_config()
    return indexer.build_index(Path(cfg["project_path"]))


# ── AI tooling: tests, docstrings, completion ────────────────────

def _find_function(source: str, name: str | None):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None, None
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return None, tree, None
    target = next((f for f in functions if f.name == name), functions[0])
    segment = ast.get_source_segment(source, target)
    return target, tree, segment


def _strip_fences(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return match.group(1).strip("\n") if match else text.strip()


class TestsIn(BaseModel):
    file_path: str
    function_name: str | None = None
    model: str | None = None
    save: bool = False


@app.post("/api/generate-tests")
def generate_tests(payload: TestsIn) -> dict:
    cfg = load_config()
    root = Path(cfg["project_path"])
    target = parser.safe_resolve(root, payload.file_path)
    if not target.is_file():
        raise HTTPException(404, f"File not found: {payload.file_path}")
    source = target.read_text(encoding="utf-8", errors="ignore")
    func, _tree, segment = _find_function(source, payload.function_name)
    subject = segment if segment else source[:4000]
    fname = func.name if func else (payload.function_name or "module")

    prompt = (f"Generate pytest unit tests for this Python code. Cover happy path, edge cases "
              f"and error handling. Output ONLY the test file content, no explanations.\n\n"
              f"```python\n{subject}\n```")
    model_key = payload.model or cfg.get("default_model", "demo")
    result = wrapper.call_model(model_key, prompt, kind="test-generation", max_tokens=2500)
    code = _strip_fences(result["text"])

    saved_path = None
    if payload.save:
        module = Path(payload.file_path).with_suffix("").as_posix().replace("/", ".")
        if module.startswith("src."):
            module = module[4:]
        tests_dir = root / "tests"
        tests_dir.mkdir(exist_ok=True)
        saved = tests_dir / f"test_{fname}.py"
        header = f"# Generated by Shadow from {payload.file_path}\n# Tests for {module}.{fname}\n\n"
        saved.write_text(header + code + "\n", encoding="utf-8")
        saved_path = saved.relative_to(root).as_posix()
    return {"code": code, "function": fname, "saved_path": saved_path,
            "cost": result["cost"], "latency_ms": result["latency_ms"], "model": result["model_key"]}


class DocstringIn(BaseModel):
    file_path: str
    symbol: str | None = None
    model: str | None = None
    save: bool = False


@app.post("/api/generate-docstring")
def generate_docstring(payload: DocstringIn) -> dict:
    cfg = load_config()
    root = Path(cfg["project_path"])
    target = parser.safe_resolve(root, payload.file_path)
    if not target.is_file():
        raise HTTPException(404, f"File not found: {payload.file_path}")
    source = target.read_text(encoding="utf-8", errors="ignore")
    func, tree, segment = _find_function(source, payload.symbol)
    if func is None:
        raise HTTPException(400, "No function found to document.")

    prompt = (f"Write a Google-style docstring for this Python function. Return ONLY the docstring "
              f"text (the triple-quoted content), no code fences, no extra prose.\n\n```python\n{segment}\n```")
    model_key = payload.model or cfg.get("default_model", "demo")
    result = wrapper.call_model(model_key, prompt, kind="docstring", max_tokens=1200)
    doc = result["text"].strip().strip('"').strip()
    if doc.startswith("'''") or doc.startswith('"""'):
        doc = doc.strip('"').strip("'")

    saved = False
    if payload.save:
        saved = _apply_docstring(target, source, func, doc)
    return {"docstring": doc, "function": func.name, "saved": saved,
            "cost": result["cost"], "latency_ms": result["latency_ms"], "model": result["model_key"]}


def _apply_docstring(target: Path, source: str, func: ast.AST, doc: str) -> bool:
    lines = source.splitlines()
    indent = " " * (func.body[0].col_offset if func.body else func.col_offset + 4)  # type: ignore[attr-defined]
    new_block = [f'{indent}"""']
    for raw in doc.splitlines() or [""]:
        new_block.append(f"{indent}{raw.rstrip()}" if raw.strip() else "")
    new_block.append(f'{indent}"""')

    first = func.body[0]  # type: ignore[attr-defined]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        start, end = first.lineno - 1, (first.end_lineno or first.lineno)
        lines[start:end] = new_block
    else:
        insert_at = first.lineno - 1
        lines[insert_at:insert_at] = new_block
    target.write_text("\n".join(lines) + ("\n" if source.endswith("\n") else ""), encoding="utf-8")
    return True


class CompletionIn(BaseModel):
    file_path: str
    line: int
    prefix: str = ""
    model: str | None = None


@app.post("/api/completion")
def completion(payload: CompletionIn) -> dict:
    cfg = load_config()
    root = Path(cfg["project_path"])
    target = parser.safe_resolve(root, payload.file_path)
    if not target.is_file():
        raise HTTPException(404, f"File not found: {payload.file_path}")
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()[: payload.line]
    context = "\n".join(lines)[-4000:]
    prompt = (f"Complete the code at the cursor position (end of the text below). "
              f"Output ONLY the completion — no explanations, no fences.\n\n```\n{context}\n```")
    model_key = payload.model or cfg.get("default_model", "demo")
    result = wrapper.call_model(model_key, prompt, kind="completion", max_tokens=800)
    return {"completion": result["text"], "model": result["model_key"],
            "cost": result["cost"], "latency_ms": result["latency_ms"]}


# ── Shadow Agent ─────────────────────────────────────────────────

class AgentIn(BaseModel):
    task: str
    model: str | None = None
    auto_apply: bool = False


@app.post("/api/agent/run")
def agent_run(payload: AgentIn) -> dict:
    task = payload.task.strip()
    if not task:
        raise HTTPException(400, "Task is empty.")
    transcript = agent_module.run_agent(task, model_key=payload.model, auto_apply=payload.auto_apply)
    return {"transcript": transcript, "credits": db.usage_summary()}


# ── Static UI ────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "8000")),
                reload=False)
