"""End-to-end API tests — runs entirely on the free offline Demo brain.

Uses an isolated SQLite DB and a temporary project directory so nothing
touches your real data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    project = tmp_path / "sample_project"
    project.mkdir()
    (project / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (project / "helper.js").write_text(
        "export function greet(name) { return 'hi ' + name; }\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JEET_DB_PATH", str(tmp_path / "usage.db"))
    monkeypatch.setenv("JEET_INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("JEET_PROJECT_PATH", str(project))
    import src.config as config_module  # noqa: WPS433  (config is loaded lazily from env each call)
    monkeypatch.setattr(config_module, "LOCAL_CONFIG_PATH", tmp_path / "config.local.yaml")
    from src.main import app  # noqa: WPS433

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_models_listed(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    keys = {m["key"] for m in resp.json()["models"]}
    assert {"demo", "gpt-5", "claude-4.5-sonnet", "gemini-2.5-pro", "ollama",
            "gemini-2.5-flash", "groq-gpt-oss-120b", "openrouter-deepseek-r1",
            "mistral-small"} <= keys


def test_free_models_are_free_and_unmetered(client):
    resp = client.get("/api/models")
    by_key = {m["key"]: m for m in resp.json()["models"]}
    for key in ("gemini-2.5-flash", "groq-gpt-oss-120b", "openrouter-deepseek-r1", "mistral-small"):
        assert by_key[key]["free"] is True
        assert by_key[key]["status"] in ("missing_key", "connected")  # no key in tests


def test_free_model_without_key_fails_without_network(client):
    resp = client.post("/api/chat", json={"message": "hi", "model": "groq-gpt-oss-120b",
                                          "include_context": False})
    assert resp.status_code == 400
    assert "console.groq.com" in resp.json()["detail"]


def test_chat_demo_and_credits_metering(client):
    before = client.get("/api/credits").json()["calls"]
    resp = client.post("/api/chat", json={"message": "Explain this project", "model": "demo",
                                          "include_context": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"]["content"]
    assert body["message"]["free"] is True
    after = client.get("/api/credits").json()
    assert after["calls"] == before + 1


def test_chat_sessions_roundtrip(client):
    resp = client.post("/api/chat", json={"message": "hello", "model": "demo"})
    session_id = resp.json()["session_id"]
    messages = client.get(f"/api/sessions/{session_id}/messages").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    delete = client.delete(f"/api/sessions/{session_id}").json()
    assert delete["ok"] is True


def test_context_files_and_view(client):
    files = client.get("/api/context/files").json()
    paths = {f["path"] for f in files["files"]}
    assert "calculator.py" in paths
    view = client.get("/api/context/file", params={"path": "calculator.py"}).json()
    assert any(sym["name"] == "add" for sym in view["symbols"])


def test_context_path_escape_rejected(client):
    resp = client.get("/api/context/file", params={"path": "../../etc/passwd"})
    assert resp.status_code in (403, 404)


def test_semantic_search(client):
    resp = client.post("/api/context/reindex")
    assert resp.status_code == 200
    results = client.get("/api/context/search", params={"q": "add numbers"}).json()["results"]
    assert any(r["path"] == "calculator.py" for r in results)


def test_generate_tests_demo(client):
    resp = client.post("/api/generate-tests",
                       json={"file_path": "calculator.py", "function_name": "add", "model": "demo"})
    assert resp.status_code == 200, resp.text
    assert "def test_add" in resp.json()["code"]


def test_generate_docstring_demo(client):
    resp = client.post("/api/generate-docstring",
                       json={"file_path": "calculator.py", "symbol": "add", "model": "demo"})
    assert resp.status_code == 200, resp.text
    assert "add" in resp.json()["docstring"]


def test_agent_demo_run(client):
    resp = client.post("/api/agent/run",
                       json={"task": "Write tests for calculator.py", "model": "demo",
                             "auto_apply": False})
    assert resp.status_code == 200, resp.text
    transcript = resp.json()["transcript"]
    assert transcript["plan"]["steps"]
    assert transcript["demo"] is True


def test_settings_roundtrip(client):
    resp = client.post("/api/settings", json={"offline_mode": True,
                                              "monthly_budget": 42.0})
    assert resp.status_code == 200
    # Local config is redirected to a temp file by the fixture — your real
    # config.local.yaml is never touched, and the flip-back is just hygiene.
    assert resp.json()["offline_mode"] is True
    assert resp.json()["credits"]["monthly_budget"] == 42.0
    client.post("/api/settings", json={"offline_mode": False, "monthly_budget": 35.0})
