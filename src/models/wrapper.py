"""Unified model gateway: call_model(model_key, prompt, ...)

Provider dispatch → demo (offline heuristics) · ollama (local) ·
openai / anthropic / google (cloud, credit-metered).
Every call is logged to the usage_log for the credits system.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .. import db
from ..config import load_config
from . import demo as demo_model
from . import local as local_model


class ModelError(Exception):
    """Raised when a model call fails. `detail` is safe to show in the UI."""

    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def get_model_registry(cfg: dict | None = None) -> dict[str, dict]:
    cfg = cfg or load_config()
    return cfg.get("models") or {}


def model_availability(cfg: dict | None = None) -> list[dict[str, Any]]:
    """Per-model status for the Models tab (never leaks the key itself)."""
    cfg = cfg or load_config()
    keys = cfg.get("api_keys", {})
    stats = db.model_stats()
    registry = get_model_registry(cfg)
    offline = bool(cfg.get("offline_mode"))
    out: list[dict[str, Any]] = []
    for key, mdl in registry.items():
        provider = mdl.get("provider", "demo")
        if provider in ("openai", "anthropic", "google"):
            key_name = {"openai": "openai", "anthropic": "anthropic", "google": "google"}[provider]
            configured = bool((keys.get(key_name) or "").strip())
            status = "offline" if offline else ("connected" if configured else "missing_key")
        elif provider == "openai_compat":
            configured = bool((keys.get(mdl.get("key_name", "")) or "").strip())
            status = "offline" if offline else ("connected" if configured else "missing_key")
        elif provider == "ollama":
            configured = local_model.ollama_available(cfg.get("ollama", {}).get("base_url", "http://localhost:11434"))
            status = "connected" if configured else "unreachable"
        else:
            configured, status = True, "connected"
        out.append({
            "key": key,
            "label": mdl.get("label", key),
            "provider": provider,
            "model": mdl.get("model") or cfg.get("ollama", {}).get("model", ""),
            "best_for": mdl.get("best_for", ""),
            "in_price": mdl.get("in_price", 0.0),
            "out_price": mdl.get("out_price", 0.0),
            "free": provider in ("demo", "ollama") or bool(mdl.get("free")),
            "configured": configured,
            "status": status,
            "is_default": key == cfg.get("default_model"),
            "stats": stats.get(key, {"calls": 0, "tokens": 0, "cost": 0.0, "avg_latency": 0}),
        })
    return out


def _call_openai(cfg: dict, mdl: dict, prompt: str, system_prompt: str | None,
                 max_tokens: int) -> tuple[str, int, int]:
    api_key = (cfg.get("api_keys", {}).get("openai") or "").strip()
    if not api_key:
        raise ModelError("OpenAI API key not configured. Add it in Settings → API keys, or use the Demo/Ollama model.", 400)
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": prompt}]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {"model": mdl.get("model", "gpt-5"), "messages": messages}

    def _post(token_param: str) -> httpx.Response:
        return httpx.post("https://api.openai.com/v1/chat/completions",
                          json={**body, token_param: max_tokens},
                          headers=headers, timeout=120)

    resp = _post("max_completion_tokens")
    if resp.status_code == 400 and "max_completion_tokens" in resp.text:
        resp = _post("max_tokens")
    if resp.status_code != 200:
        raise ModelError(f"OpenAI error (HTTP {resp.status_code}): {resp.text[:400]}")
    data = resp.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    usage = data.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or estimate_tokens(prompt + (system_prompt or "")))
    ct = int(usage.get("completion_tokens") or estimate_tokens(text))
    return text, pt, ct


def _call_openai_compat(cfg: dict, mdl: dict, prompt: str, system_prompt: str | None,
                        max_tokens: int) -> tuple[str, int, int]:
    """Generic OpenAI-compatible chat completions (Groq, OpenRouter, Mistral…)."""
    key_name = mdl.get("key_name", "openai")
    api_key = (cfg.get("api_keys", {}).get(key_name) or "").strip()
    base_url = (mdl.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ModelError(f"Model '{mdl.get('label', key_name)}' is missing its base_url.", 500)
    if not api_key:
        hints = {"groq": "https://console.groq.com (free, no card)",
                 "openrouter": "https://openrouter.ai/keys (free, no card)",
                 "mistral": "https://console.mistral.ai (free Experiment tier)"}
        raise ModelError(f"{key_name.capitalize()} API key not configured. Get a FREE key at {hints.get(key_name, 'the provider console')} — then paste it in Settings → API keys.", 400)
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": prompt}]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://github.com/mdrehanfazal/Jeet-ai-", "X-Title": "Shadow"}
    body = {"model": mdl.get("model"), "messages": messages, "max_tokens": max_tokens}
    try:
        resp = httpx.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=120)
    except httpx.HTTPError as exc:
        raise ModelError(f"Cannot reach {base_url}: {exc}") from exc
    if resp.status_code != 200:
        raise ModelError(f"{key_name.capitalize()} error (HTTP {resp.status_code}): {resp.text[:400]}")
    data = resp.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    usage = data.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or estimate_tokens(prompt + (system_prompt or "")))
    ct = int(usage.get("completion_tokens") or estimate_tokens(text))
    return text, pt, ct


def _call_anthropic(cfg: dict, mdl: dict, prompt: str, system_prompt: str | None,
                    max_tokens: int) -> tuple[str, int, int]:
    api_key = (cfg.get("api_keys", {}).get("anthropic") or "").strip()
    if not api_key:
        raise ModelError("Anthropic API key not configured. Add it in Settings → API keys, or use the Demo/Ollama model.", 400)
    body: dict[str, Any] = {
        "model": mdl.get("model", "claude-sonnet-4-5"),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt
    try:
        resp = httpx.post("https://api.anthropic.com/v1/messages", json=body,
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                   "Content-Type": "application/json"},
                          timeout=120)
    except httpx.HTTPError as exc:
        raise ModelError(f"Cannot reach Anthropic API: {exc}") from exc
    if resp.status_code != 200:
        raise ModelError(f"Anthropic error (HTTP {resp.status_code}): {resp.text[:400]}")
    data = resp.json()
    text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text").strip()
    usage = data.get("usage") or {}
    pt = int(usage.get("input_tokens") or estimate_tokens(prompt))
    ct = int(usage.get("output_tokens") or estimate_tokens(text))
    return text, pt, ct


def _call_google(cfg: dict, mdl: dict, prompt: str, system_prompt: str | None,
                 max_tokens: int) -> tuple[str, int, int]:
    api_key = (cfg.get("api_keys", {}).get("google") or "").strip()
    if not api_key:
        raise ModelError("Google API key not configured. Add it in Settings → API keys, or use the Demo/Ollama model.", 400)
    model_name = mdl.get("model", "gemini-2.5-pro")
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    try:
        resp = httpx.post(url, json=body, params={"key": api_key}, timeout=120)
    except httpx.HTTPError as exc:
        raise ModelError(f"Cannot reach Google Generative Language API: {exc}") from exc
    if resp.status_code != 200:
        raise ModelError(f"Google error (HTTP {resp.status_code}): {resp.text[:400]}")
    data = resp.json()
    candidates = data.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(part.get("text", "") for part in parts).strip()
    meta = data.get("usageMetadata") or {}
    pt = int(meta.get("promptTokenCount") or estimate_tokens(prompt + (system_prompt or "")))
    ct = int(meta.get("candidatesTokenCount") or estimate_tokens(text))
    return text, pt, ct


def _call_ollama(cfg: dict, mdl: dict, prompt: str, system_prompt: str | None) -> tuple[str, int, int]:
    ollama_cfg = cfg.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    model_name = (mdl.get("model") or "").strip() or ollama_cfg.get("model", "llama3.1:latest")
    try:
        result = local_model.generate(base_url, model_name, prompt, system_prompt)
    except local_model.LocalModelError as exc:
        raise ModelError(str(exc), 502) from exc
    return result["text"], result["prompt_tokens"], result["completion_tokens"]


def call_model(model_key: str | None, prompt: str, system_prompt: str | None = None,
               kind: str = "chat", max_tokens: int = 4096) -> dict[str, Any]:
    """Call the selected model and meter the usage. Never raises raw HTTP errors."""
    cfg = load_config()
    registry = get_model_registry(cfg)
    key = model_key if model_key in registry else cfg.get("default_model", "demo")
    if key not in registry:
        key = "demo"
    mdl = registry[key]
    provider = mdl.get("provider", "demo")

    forced_offline = False
    if cfg.get("offline_mode") and provider in ("openai", "anthropic", "google", "openai_compat"):
        mdl = registry.get("ollama", mdl)
        provider, key, forced_offline = "ollama", "ollama", True

    start = time.perf_counter()
    if provider == "demo":
        text = demo_model.generate(prompt, system_prompt)
        pt, ct = estimate_tokens((system_prompt or "") + prompt), estimate_tokens(text)
    elif provider == "ollama":
        text, pt, ct = _call_ollama(cfg, mdl, prompt, system_prompt)
    elif provider == "openai_compat":
        text, pt, ct = _call_openai_compat(cfg, mdl, prompt, system_prompt, max_tokens)
    elif provider == "openai":
        text, pt, ct = _call_openai(cfg, mdl, prompt, system_prompt, max_tokens)
    elif provider == "anthropic":
        text, pt, ct = _call_anthropic(cfg, mdl, prompt, system_prompt, max_tokens)
    elif provider == "google":
        text, pt, ct = _call_google(cfg, mdl, prompt, system_prompt, max_tokens)
    else:
        raise ModelError(f"Unknown provider '{provider}' for model '{key}'.", 500)

    latency_ms = int((time.perf_counter() - start) * 1000)
    free = provider in ("demo", "ollama") or bool(mdl.get("free"))
    cost = 0.0 if free else (pt / 1000.0 * float(mdl.get("in_price", 0.0)) +
                              ct / 1000.0 * float(mdl.get("out_price", 0.0)))

    db.log_usage(model=key, kind=kind, prompt_tokens=pt, completion_tokens=ct,
                 cost=cost, latency_ms=latency_ms)

    result: dict[str, Any] = {
        "text": text,
        "model_key": key,
        "model_label": mdl.get("label", key),
        "provider": provider,
        "kind": kind,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "cost": round(cost, 6),
        "latency_ms": latency_ms,
        "free": free,
    }
    if forced_offline:
        result["notice"] = "Offline mode is ON — request was routed to local Ollama instead of the cloud model."
    return result
