# Jeet — Private AI Assistant

> Personal clone of the JetBrains AI Assistant premium experience (AI Ultimate tier), rebuilt from scratch for local, private use.
> **Not affiliated with JetBrains. For personal use only — do not distribute.**

Everything runs on your machine. A **Demo brain** (offline heuristics) works out of the box with zero keys; plug in an **OpenAI / Anthropic / Google** key or a local **Ollama** model for full-power completions, chat and the autonomous agent.

## Features

| # | Feature | Status |
|---|---------|--------|
| A | Multi-model selection & switching (GPT-5 · Claude 4.5 Sonnet · Gemini 2.5 Pro · Ollama · Demo) | ✅ per-conversation picker, cost tracking |
| B | PSI-style context awareness — project scan, Python **AST symbol table**, open tabs + cursor line injection, **TF-IDF semantic index** | ✅ |
| C | AI code generation & completion (inline complete button at cursor) | ✅ |
| D | Test & docstring generation with one-click save | ✅ pytest skeletons · Google-style docstrings |
| E | AI chat with persistent history (SQLite), sessions sidebar | ✅ |
| F | **Junie-clone agent**: Plan → Execute → Verify → Refine, sandboxed to project root, allowlisted shell-free commands | ✅ |
| G | Credit system simulation (35 credits/month, 1 credit = $1, low-balance warning) | ✅ |
| H | Local model support (Ollama offline) + offline mode switch | ✅ |

## Quickstart

```bash
cd Jeet-ai-
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py            # → http://localhost:8000
```

(Windows: `py -m venv .venv && .venv\Scripts\activate`.)

Open **http://localhost:8000** — the UI is served by the same server (no Node build step).

### 60-second tour
1. **Chat** — ask "Explain this project". Toggle *Project context* / *Open files* chips to control what's injected.
2. **Files** — browse the tree, click a line to set the AI cursor, try semantic search ("credit deduction logic"), hit **✨ Complete @ cursor**, **🧪 Generate tests**, **📝 Add docstring**.
3. **Agent** — `Add a healthz endpoint with a test`. Preview mode (auto-apply **off**) never writes; enable it to execute. Verify step runs `pytest -q` automatically when `tests/` exists.
4. **Credits** — ring gauge, per-model spend, daily bars, recent usage. Demo/Ollama cost 0.
5. **Models** — status per model (connected / missing key / Ollama offline), one-click default.
6. **Settings** — project root, offline mode, monthly budget, API keys → saved to `config.local.yaml` (gitignored).

## Configuration

Precedence: `config.yaml` (committed defaults) → `config.local.yaml` (gitignored secrets/overrides) → env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `JEET_PROJECT_PATH`, `PORT`).

```yaml
# config.local.yaml (created automatically by the Settings page)
project_path: "/home/you/your-project"
offline_mode: false
default_model: "claude-4.5-sonnet"
credits: { monthly_budget: 35.0, warning_threshold: 5.0 }
api_keys: { openai: "sk-…", anthropic: "sk-ant-…", google: "AIza…" }
ollama:  { base_url: "http://localhost:11434", model: "llama3.1:latest" }
```

**Ollama (free, offline, unlimited):** `ollama serve` then `ollama pull llama3.1`. Flip *Offline mode* in Settings and everything routes to it.

## Architecture

```
┌────────────┐   HTTP/JSON   ┌──────────────────────────────┐
│ static/    │ ◄───────────► │ src/main.py (FastAPI)        │
│ index.html │               │  ├─ src/models/wrapper.py    │──► OpenAI · Anthropic · Gemini
│ (SPA)      │               │  ├─ src/models/local.py      │──► Ollama  (http://localhost:11434)
└────────────┘               │  ├─ src/models/demo.py       │──► Offline heuristic brain
                             │  ├─ src/context/parser.py    │    AST (Python) + regex (JS/TS/…)
                             │  ├─ src/context/indexer.py   │    TF-IDF vector search (.jeet/index.json)
                             │  ├─ src/agent/{planner,executor,agent}.py
                             │  └─ src/db.py                │    SQLite: usage_log + chat_*
                             └──────────────────────────────┘
```

**Safety model (agent):** all tools confined to `project_path` (path escapes rejected) · commands run **without a shell** against an allowlist (`pytest`, `python -m pytest`, `python -m compileall`, `npm test`, `ls`, `git status|diff|log`…) · `auto_apply=false` logs every write/command as a proposal, nothing touches disk · one automatic refine pass on failing tests.

**Privacy:** no telemetry, no external calls except the model API you configure. Indexes and DBs stay in `.jeet/` and `database/` (both gitignored).

## API surface

`GET  /api/health · /api/models · /api/credits · /api/settings · /api/context/{files,file,search,status} · /api/sessions[/{id}[/messages]]`
`POST /api/chat · /api/agent/run · /api/generate-tests · /api/generate-docstring · /api/completion · /api/context/reindex · /api/settings · /api/models/default`
`DELETE /api/sessions/{id}`

## Tests

```bash
pip install pytest
python -m pytest tests/ -q     # 11 tests, fully offline, isolated temp DB/project
```

## Costs note

Cloud prices live in `config.yaml` under `models.*` as **credits per 1K tokens** (1 credit = $1): GPT-5 ≈ $1.25/$10 per Mtok, Claude 4.5 Sonnet $3/$15, Gemini 2.5 Pro ≈ $1.25/$10. Local/demo models are free. Verify current pricing on each provider's site — these are placeholders for the simulation.

## Legal

Personal use only. Respect each provider's Terms of Service. Not affiliated with or endorsed by JetBrains, OpenAI, Anthropic or Google.
