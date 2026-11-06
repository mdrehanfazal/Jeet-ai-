"""Shadow Agent orchestrator: plan → execute → verify → refine.

Safety model
------------
* Every tool is confined to the configured project root (path escapes rejected).
* `run_command` executes WITHOUT a shell, against a strict allowlist.
* `auto_apply=False` → read-only tools run, but writes/commands are logged as
  *proposals* and never touch disk or processes.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..config import load_config
from ..context import indexer
from ..context.parser import scan_project
from ..models import wrapper
from . import executor, planner

AGENT_SYSTEM = (
    "You are Shadow Agent, an autonomous senior coding agent inside a private IDE. "
    "You work in ONE project rooted at the project root — never escape it.\n"
    "Each turn respond with STRICT JSON only (no prose, no fences):\n"
    '{"thought": "why this next step", "action": "<tool>", "args": {...}}\n\n'
    "Tools:\n"
    '  list_files  {"path": "."}                          — list a directory\n'
    '  read_file   {"path": "src/foo.py"}                 — read a text file\n'
    '  write_file  {"path": "...", "content": "..."}      — create/overwrite a file\n'
    '  run_command {"command": "python -m pytest -q"}     — allowlisted commands only\n'
    '  finish      {"summary": "what you did and why"}    — end the task\n\n'
    "Rules: small correct steps; prefer reading before writing; run tests when "
    "tests exist; never fabricate file paths you have not seen — list/read first; "
    "when the task is done or verification passes, call finish."
)

REFINE_SYSTEM = (
    "You are a senior engineer fixing a failing test suite. Respond with STRICT JSON only:\n"
    '{"explanation": "root cause in one sentence", "files": [{"path": "...", "content": "FULL corrected file"}]}\n'
    "Only include files you are confident about. Empty files array if unsure."
)


def build_context(root: Path, task: str, max_files: int = 400) -> str:
    scan = scan_project(root)
    lines = [f"Project root: {scan['root']}  ({scan['count']} files, {scan['total_bytes'] // 1024} KB)",
             "", "Files:"]
    for entry in scan["files"][:max_files]:
        lines.append(f"  {entry['path']} ({entry['language']})")
    if scan["count"] > max_files:
        lines.append(f"  … and {scan['count'] - max_files} more")
    try:
        hits = indexer.search(task, root, k=5)
    except Exception:
        hits = []
    if hits:
        lines += ["", "Relevant snippets (semantic search):"]
        for hit in hits:
            lines.append(f"- {hit['path']}  (score {hit['score']})")
            for snip in hit["snippet"]:
                lines.append(f"    L{snip['n']}: {snip['text']}")
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def _parse_action(text: str) -> dict | None:
    import json, re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        match = re.search(r"\{.*\}", text, re.S)
        candidate = match.group(0) if match else None
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("action"), str):
        return None
    data.setdefault("args", {})
    data.setdefault("thought", "")
    return data


def _exec_action(root: Path, name: str, args: dict, auto_apply: bool) -> dict:
    if name == "list_files":
        return {"ok": True, "output": executor.list_dir(root, args.get("path", "."))}
    if name == "read_file":
        result = executor.read_file(root, str(args.get("path", "")))
        return {"ok": True, "output": _truncate(result["content"], 6000),
                "truncated": result["truncated"]}
    if name == "write_file":
        path, content = str(args.get("path", "")), str(args.get("content", ""))
        if not auto_apply:
            return {"ok": True, "proposed": True,
                    "output": f"[preview] would write {path} ({len(content)} chars) — auto-apply is OFF"}
        if not path:
            raise executor.ToolError("write_file requires a path")
        return {"ok": True, "output": executor.write_file(root, path, content)}
    if name == "run_command":
        command = str(args.get("command", ""))
        if not auto_apply:
            return {"ok": True, "proposed": True,
                    "output": f"[preview] would run: {command} — auto-apply is OFF"}
        result = executor.run_command(root, command)
        return {"ok": result["ok"], "output": result}
    raise executor.ToolError(f"Unknown tool: {name}")


def _run_pytest(root: Path) -> dict | None:
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return None
    try:
        return executor.run_command(root, "python -m pytest -q")
    except executor.ToolError as exc:
        return {"ok": False, "command": "python -m pytest -q", "stdout": "", "stderr": str(exc),
                "returncode": -1}


def _demo_proposal(root: Path, task: str, steps: list[dict], auto_apply: bool) -> tuple[list[dict], list[str]]:
    actions: list[dict] = []
    files_changed: list[str] = []
    if not auto_apply:
        actions.append({"iteration": 1, "thought": "Offline demo brain cannot edit the project on its own.",
                        "tool": "preview", "args": {},
                        "ok": True, "proposed": True,
                        "output": "Auto-apply is OFF — plan only, nothing was written."})
        return actions, files_changed
    slug = "".join(c if c.isalnum() else "-" for c in task.lower())[:48].strip("-") or "task"
    rel = f"agent_output/proposal-{time.strftime('%Y%m%d-%H%M%S')}-{slug}.md"
    step_lines = "\n".join(f"{i + 1}. **{s['title']}** — {s.get('detail', '')}"
                           for i, s in enumerate(steps))
    content = f"# Agent proposal — {task.strip()[:120]}\n\n_Generated {time.strftime('%Y-%m-%d %H:%M')} by the offline demo brain._\n\n## Suggested plan\n\n{step_lines}\n\n## Next step\n\nConnect a real model (Settings → Models) and re-run the agent to have it executed end-to-end.\n"
    try:
        result = executor.write_file(root, rel, content)
        actions.append({"iteration": 1, "thought": "Offline brain writes a reviewable proposal instead of blind edits.",
                        "tool": "write_file", "args": {"path": rel}, "ok": True,
                        "output": f"Wrote {result['path']} ({result['bytes']} bytes)"})
        files_changed.append(rel)
    except executor.ToolError as exc:
        actions.append({"iteration": 1, "thought": "Could not write proposal.", "tool": "write_file",
                        "args": {"path": rel}, "ok": False, "output": str(exc)})
    return actions, files_changed


def run_agent(task: str, model_key: str | None = None, auto_apply: bool = False,
              max_iters: int = 8) -> dict:
    started = time.time()
    cfg = load_config()
    root = Path(cfg["project_path"])
    registry = wrapper.get_model_registry(cfg)
    key = model_key if model_key in registry else cfg.get("default_model", "demo")
    provider = registry.get(key, {}).get("provider", "demo")
    demo_mode = provider == "demo"

    context = build_context(root, task)
    plan_info = planner.plan_task(task, key, context)
    plan = plan_info["plan"]

    transcript: dict = {
        "task": task,
        "model_key": key,
        "model_label": registry.get(key, {}).get("label", key),
        "demo": demo_mode,
        "auto_apply": auto_apply,
        "project_root": str(root),
        "plan": plan,
        "actions": [],
        "files_changed": [],
        "verify": None,
        "refine": None,
        "summary": "",
        "cost": float(plan_info.get("cost", 0.0)),
        "total_tokens": 0,
        "warnings": [],
    }

    if demo_mode:
        actions, files_changed = _demo_proposal(root, task, plan["steps"], auto_apply)
        transcript["actions"] = actions
        transcript["files_changed"] = files_changed
        verify = _run_pytest(root) if auto_apply else None
        transcript["verify"] = verify
        transcript["summary"] = ("Demo mode: plan created" +
                                 (" and proposal written. " if files_changed else ". ") +
                                 "Connect a cloud/local model for full autonomous execution.")
        transcript["duration_ms"] = int((time.time() - started) * 1000)
        return transcript

    history: list[dict] = []
    files_changed: list[str] = []
    proposed_files: list[dict] = []
    summary_text = ""

    for iteration in range(1, max_iters + 1):
        history_block = "\n".join(
            f"Step {h['iteration']}: {h['tool']}({_truncate(str(h['args']), 200)})"
            f" -> ok={h['ok']} output={_truncate(str(h['output']), 900)}"
            for h in history[-6:]
        ) or "(no steps yet)"
        prompt = (
            f"Task:\n{task}\n\nPlan:\n" +
            "\n".join(f"{i + 1}. {s['title']} — {s.get('detail', '')}" for i, s in enumerate(plan["steps"])) +
            f"\n\nProject snapshot:\n{context[:5000]}\n\nProgress so far:\n{history_block}\n\n"
            "Decide the next action. JSON only."
        )
        try:
            result = wrapper.call_model(key, prompt, system_prompt=AGENT_SYSTEM,
                                        kind="agent", max_tokens=2000)
        except wrapper.ModelError as exc:
            transcript["warnings"].append(f"Model call failed at step {iteration}: {exc.detail}")
            break
        transcript["cost"] += result["cost"]
        transcript["total_tokens"] += result["total_tokens"]

        action = _parse_action(result["text"])
        if action is None:
            transcript["warnings"].append(f"Could not parse model output at step {iteration}; stopping.")
            transcript["actions"].append({"iteration": iteration, "thought": "unparseable output",
                                          "tool": "parse_error", "args": {}, "ok": False,
                                          "output": _truncate(result["text"], 800)})
            break

        tool, args, thought = action["action"], action["args"], action["thought"]
        if tool == "finish":
            summary_text = str(args.get("summary", "Task complete."))
            transcript["actions"].append({"iteration": iteration, "thought": thought,
                                          "tool": "finish", "args": args, "ok": True,
                                          "output": summary_text})
            break

        try:
            outcome = _exec_action(root, tool, args, auto_apply)
        except (executor.ToolError, ValueError, FileNotFoundError) as exc:
            outcome = {"ok": False, "output": str(exc)}

        entry = {"iteration": iteration, "thought": thought, "tool": tool, "args": args,
                 "ok": bool(outcome.get("ok")),
                 "proposed": bool(outcome.get("proposed")),
                 "output": outcome.get("output")}
        transcript["actions"].append(entry)
        history.append({**entry, "output": _truncate(str(outcome.get("output") or ""), 1500)})

        if tool == "write_file" and outcome.get("proposed"):
            proposed_files.append({"path": str(args.get("path", "")),
                                   "content": str(args.get("content", ""))})
        elif tool == "write_file" and outcome.get("ok"):
            out = outcome["output"]
            if isinstance(out, dict):
                files_changed.append(out.get("path", str(args.get("path", ""))))

    transcript["files_changed"] = files_changed
    transcript["proposed_files"] = proposed_files

    verify = _run_pytest(root) if auto_apply else None
    transcript["verify"] = verify

    # One refine pass if tests fail and we can write files
    if verify is not None and not verify.get("ok") and auto_apply:
        failure_excerpt = _truncate((verify.get("stdout") or "") + "\n" + (verify.get("stderr") or ""), 3500)
        refine_prompt = (f"Tests fail after working on this task:\n{task}\n\n"
                         f"pytest output:\n{failure_excerpt}\n\nFix the failures.")
        try:
            refine = wrapper.call_model(key, refine_prompt, system_prompt=REFINE_SYSTEM,
                                        kind="agent-refine", max_tokens=4000)
            transcript["cost"] += refine["cost"]
            transcript["total_tokens"] += refine["total_tokens"]
            applied: list[str] = []
            explanation = ""
            if refine["text"]:
                import json, re
                match = re.search(r"\{.*\}", refine["text"], re.S)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        explanation = str(data.get("explanation", ""))
                        for file_obj in data.get("files", [])[:10]:
                            path, content = file_obj.get("path"), file_obj.get("content")
                            if path and content is not None:
                                executor.write_file(root, str(path), str(content))
                                applied.append(str(path))
                    except (json.JSONDecodeError, executor.ToolError, ValueError):
                        pass
            verify2 = _run_pytest(root)
            transcript["refine"] = {"explanation": explanation, "applied": applied,
                                    "verify_ok_after": bool(verify2 and verify2.get("ok"))}
            transcript["verify"] = verify2 or verify
            transcript["files_changed"] = files_changed + applied
        except wrapper.ModelError as exc:
            transcript["warnings"].append(f"Refine pass failed: {exc.detail}")

    if not summary_text:
        summary_text = ("Task complete." if (not verify or verify.get("ok"))
                        else "Task finished with failing tests — see verify output.")
    transcript["summary"] = summary_text
    transcript["duration_ms"] = int((time.time() - started) * 1000)
    return transcript
