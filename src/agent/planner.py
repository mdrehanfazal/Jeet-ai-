"""Shadow Agent planner: task → structured plan.

With a cloud/local model the plan is produced by the model (JSON mode, one
repair attempt). On the offline Demo brain — or on any parse failure — a
keyword heuristic produces a sane default plan instead of crashing.
"""
from __future__ import annotations

import json
import re

from ..config import load_config
from ..models import wrapper

SYSTEM_PROMPT = (
    "You are Shadow Agent, an autonomous senior coding agent embedded in a private IDE. "
    "Given a task and a snapshot of the project, produce a concise execution plan.\n"
    "Respond with STRICT JSON only, no prose, matching:\n"
    '{"summary": "one sentence", "steps": [{"title": "...", "detail": "..."}]}\n'
    "3 to 7 steps. Order them: understand → create/change → verify → refine."
)


def heuristic_plan(task: str) -> dict:
    t = task.lower()
    steps = [{"title": "Recon project", "detail": "Scan structure and open the relevant files."}]
    if any(w in t for w in ("test", "pytest", "unit test")):
        steps.append({"title": "Design test matrix", "detail": "Enumerate happy path, edge cases and failure modes."})
        steps.append({"title": "Implement tests", "detail": "Write pytest modules under tests/."})
        steps.append({"title": "Run & iterate", "detail": "python -m pytest -q; fix failures."})
    elif any(w in t for w in ("doc", "readme", "documentation")):
        steps.append({"title": "Audit current docs", "detail": "Find gaps against the actual code."})
        steps.append({"title": "Draft documentation", "detail": "Write accurate, example-driven docs."})
    elif any(w in t for w in ("fix", "bug", "error", "crash")):
        steps.append({"title": "Reproduce", "detail": "Minimal failing case; capture the traceback."})
        steps.append({"title": "Localize & patch", "detail": "Fix the root cause, not the symptom."})
        steps.append({"title": "Regression check", "detail": "Run the suite; add a regression test."})
    elif any(w in t for w in ("add", "implement", "create", "build", "feature")):
        steps.append({"title": "Design the change", "detail": "Smallest coherent slice; match codebase idioms."})
        steps.append({"title": "Implement", "detail": "Write the code behind the new capability."})
        steps.append({"title": "Verify", "detail": "Compile/lint/test; iterate until green."})
    elif any(w in t for w in ("refactor", "clean", "simplify")):
        steps.append({"title": "Identify hotspots", "detail": "Long functions, duplication, unclear names."})
        steps.append({"title": "Refactor in small steps", "detail": "Behaviour-preserving edits; tests stay green."})
    else:
        steps.append({"title": "Formulate approach", "detail": "Clarify unknowns; pick the smallest viable solution."})
        steps.append({"title": "Implement", "detail": "Apply the change in coherent increments."})
        steps.append({"title": "Verify & summarize", "detail": "Run available checks; summarize what changed."})
    return {"summary": task.strip()[:140] or "Untitled task", "steps": steps}


def _extract_json(text: str) -> dict | None:
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
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    cleaned = []
    for step in steps[:7]:
        if isinstance(step, dict) and step.get("title"):
            cleaned.append({"title": str(step["title"])[:120],
                            "detail": str(step.get("detail", ""))[:400]})
    if not cleaned:
        return None
    return {"summary": str(data.get("summary", ""))[:200], "steps": cleaned}


def plan_task(task: str, model_key: str | None, context: str) -> dict:
    """Returns {plan: {summary, steps}, demo: bool, cost: float, latency_ms: int}."""
    cfg = load_config()
    registry = wrapper.get_model_registry(cfg)
    key = model_key if model_key in registry else cfg.get("default_model", "demo")
    provider = registry.get(key, {}).get("provider", "demo")

    if provider == "demo":
        return {"plan": heuristic_plan(task), "demo": True, "cost": 0.0, "latency_ms": 0,
                "model_key": "demo"}

    prompt = f"Task:\n{task}\n\nProject snapshot:\n{context[:6000]}\n\nProduce the plan now."
    try:
        result = wrapper.call_model(key, prompt, system_prompt=SYSTEM_PROMPT,
                                    kind="agent-plan", max_tokens=1200)
    except wrapper.ModelError:
        return {"plan": heuristic_plan(task), "demo": False, "cost": 0.0,
                "latency_ms": 0, "model_key": key, "fallback": "model-unavailable"}

    parsed = _extract_json(result["text"])
    if parsed is None:
        parsed = heuristic_plan(task)
        fallback = "unparseable-plan"
    else:
        fallback = None
    return {"plan": parsed, "demo": False, "cost": result["cost"],
            "latency_ms": result["latency_ms"], "model_key": key, "fallback": fallback}
