"""Configuration loader for Shadow — Private AI Assistant.

Reads ``config.yaml`` (committed defaults) and merges ``config.local.yaml``
(gitignored secrets) plus environment variables on top of it.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
LOCAL_CONFIG_PATH = ROOT / "config.local.yaml"

DEFAULTS: dict[str, Any] = {
    "app": {"name": "Shadow — Private AI Assistant", "version": "1.1.0"},
    "project_path": ".",
    "offline_mode": False,
    "default_model": "demo",
    "credits": {"monthly_budget": 35.0, "warning_threshold": 5.0},
    "ollama": {"base_url": "http://localhost:11434", "model": "llama3.1:latest"},
    "api_keys": {"openai": "", "anthropic": "", "google": "",
                 "groq": "", "openrouter": "", "mistral": ""},
    "models": {},
}

KEY_NAMES = ("openai", "anthropic", "google", "groq", "openrouter", "mistral")

_ENV_KEY_MAP = {
    "OPENAI_API_KEY": "openai",
    "ANTHROPIC_API_KEY": "anthropic",
    "GOOGLE_API_KEY": "google",
    "GEMINI_API_KEY": "google",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def load_config() -> dict[str, Any]:
    cfg = _deep_merge(DEFAULTS, _read_yaml(CONFIG_PATH))
    cfg = _deep_merge(cfg, _read_yaml(LOCAL_CONFIG_PATH))

    # Environment variable overrides (never written to disk)
    for env_var, key_name in _ENV_KEY_MAP.items():
        env_val = os.environ.get(env_var)
        if env_val:
            cfg.setdefault("api_keys", {})[key_name] = env_val

    env_project = os.environ.get("JEET_PROJECT_PATH")
    if env_project:
        cfg["project_path"] = env_project

    project = Path(os.path.expanduser(str(cfg.get("project_path") or ".")))
    if not project.is_absolute():
        project = (ROOT / project).resolve()
    cfg["project_path"] = str(project)
    return cfg


def project_root() -> Path:
    return Path(load_config()["project_path"])


def save_local_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into config.local.yaml and return the new local config."""
    current = _read_yaml(LOCAL_CONFIG_PATH)
    merged = _deep_merge(current, updates)
    with LOCAL_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, default_flow_style=False, sort_keys=False)
    return merged


def public_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Settings payload safe to send to the UI (never includes raw keys)."""
    keys = cfg.get("api_keys", {})
    return {
        "app": cfg.get("app", {}),
        "project_path": cfg.get("project_path"),
        "offline_mode": bool(cfg.get("offline_mode")),
        "default_model": cfg.get("default_model"),
        "credits": cfg.get("credits", {}),
        "ollama": cfg.get("ollama", {}),
        "api_keys": {name: bool((keys.get(name) or "").strip()) for name in KEY_NAMES},
        "config_files": {
            "defaults": str(CONFIG_PATH),
            "local": str(LOCAL_CONFIG_PATH),
        },
    }
