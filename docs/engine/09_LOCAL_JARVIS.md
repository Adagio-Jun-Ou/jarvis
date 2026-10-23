# Jarvis, local — running the whole system on one machine

The engine that was built for Databricks runs unchanged on a personal
machine: the platform sat behind four seams (a `sql` callable, a DB
connection, an LLM executor, a browser backend), and local mode
substitutes each.

```powershell
# one-time
pip install -r requirements.txt
pip install faster-whisper edge-tts mss pynput pytesseract playwright   # optional arms
playwright install chromium
# secrets: ~/.jarvis/secrets/api_keys.env  (KEY=value lines)
#   OPENROUTER_API_KEY=sk-or-...
python run_local.py          # http://127.0.0.1:8765
```

| Concern | Local substitute | Knob |
|---|---|---|
| LLM | OpenRouter (OpenAI wire, real streaming, native tool calls) | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (default `z-ai/glm-5.3-flash`), `ALEX_LLM_BACKEND=openrouter\|ollama\|mock` |
| Store | SQLite, WAL, one file — survives restarts | `ALEX_DB_PATH` (default `~/.jarvis/alex.db`) |
| Identity | the dev token the UI self-bootstraps (single user) | `ALEX_ENV=dev` |
| Embeddings | local Ollama `nomic-embed-text` | `ALEX_EMBED_MODEL`, `ALEX_EMBED_URL` |
| Browser | local Playwright Chromium | `ALEX_BROWSER=playwright\|memory` |
| Desktop | mss + pynput (+ Tesseract OCR) | tools `desktop_*`; acts are policy-ASK |
| Voice | faster-whisper on CUDA (large-v3) / CPU (base); edge-tts | `ALEX_WHISPER_MODEL`, `ALEX_TTS_VOICE` |
| Cost | governor budgets + `ALEX_RUN_MAX_TOKENS`, approved model catalog | `ALEX_RUN_*` |

Everything enterprise-grade still applies: the constitution, ACLs
(one admin user), the execution governor, tool policy (allow/ask/deny —
desktop acts and installs ask), plan/acceptance/questions on the task
board, standing orders with the scheduler, swarm sub-agents with
inherited budgets, per-turn run metrics, telemetry export.

## When something is missing

    python scripts/engine/doctor.py

Prints the configured backend and whether its key is present (never the
key itself), the durable-store path and whether it is writable, which
optional arms (browser/voice/screen) are installed, how many adapted
Jarvis tools registered, the dev ACL capability count, and the
constitution article count. Exit 0 means the core checks pass.

Model routes on OpenRouter live in `agent/models.py` (`OPENROUTER_CATALOG`);
the approved-endpoint gate applies to them exactly as it did to serving
endpoints — a cost guardrail matters more on a personal card.
