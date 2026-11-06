"""Ollama / LM-Studio style local model integration (OpenAI-compatible fallback)."""
from __future__ import annotations

import httpx


class LocalModelError(Exception):
    pass


def ollama_available(base_url: str, timeout: float = 1.5) -> bool:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def generate(base_url: str, model: str, prompt: str,
             system_prompt: str | None = None, timeout: float = 120.0) -> dict:
    """Call Ollama's /api/generate. Returns dict with text + token counts (estimated if absent)."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system_prompt:
        payload["system"] = system_prompt
    try:
        resp = httpx.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        raise LocalModelError(
            f"Cannot reach Ollama at {base_url} — is it running? ({exc}). "
            f"Start it with `ollama serve` and pull a model: `ollama pull {model}`."
        ) from exc
    if resp.status_code != 200:
        raise LocalModelError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    prompt_tokens = int(data.get("prompt_eval_count") or max(1, len(prompt) // 4))
    completion_tokens = int(data.get("eval_count") or max(1, len(data.get("response", "")) // 4))
    return {"text": data.get("response", "").strip(),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
