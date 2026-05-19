# PNP - Persistent Neural Process

PNP is a local prototype for an AI runtime with persistent process state,
tiered memory, replayable continuity events, and an invariant-gated adaptation
layer.

This repo does not train real LoRA weights yet. The current adapter stores
structured learning deltas and injects relevant positive deltas into inference
context. The base model call is still external unless mock inference is enabled.

## What Works Now

- FastAPI app starts with the persistent core, episodic memory, semantic memory,
  and experience adapter initialized.
- Runtime state changes are written to an append-only JSONL event journal.
- Shutdown writes a core snapshot, and startup restores that snapshot plus
  journal events that happened after the snapshot.
- Adapter deltas are persisted in `adapter/state.json` and are available after
  restart for adaptation context.
- Chat inference receives working-memory, episodic, semantic, adapter, and core
  state context.
- Mutating endpoints require a local API token.
- The default API bind is `127.0.0.1`.
- Tests and a smoke script can verify the API path with mock inference and no
  live model call.

## Current Limits

- Semantic and episodic retrieval use simple keyword scoring, not embeddings.
- Consolidation uses rule-based extraction, not an LLM extractor.
- The adapter is a persisted delta/summarization layer, not live LoRA weight
  training.
- The persistent core runs while the API process is alive; this repo does not
  include a service installer or watchdog daemon.
- The default token in `config/default.yaml` is for local development. Set
  `PNP_LOCAL_TOKEN` for real local use.

## Project Structure

```text
adapters/
  invariant.py       Constitutional update gate
  lora.py            Persisted experience-delta adapter
api/
  server.py          FastAPI app and local auth boundary
config/
  default.yaml       Local runtime defaults
core/
  goals.py           Goal stack
  journal.py         Append-only JSONL continuity journal
  process.py         Persistent core loops, snapshot, and replay
  state.py           Motivational state vector
docs/
  README.md          Runtime truth notes
inference/
  engine.py          Context assembly and external model call
memory/
  hot.py             Working memory
  warm.py            SQLite episodic memory
  cold.py            SQLite semantic memory
  consolidator.py    Rule-based memory consolidation
scripts/
  smoke_api.py       No-live-model API smoke
tests/
  test_api_smoke.py
  test_continuity.py
main.py              Uvicorn entrypoint
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Locally

Use an environment token instead of the development fallback token:

```powershell
$env:PNP_LOCAL_TOKEN = "replace-with-local-secret"
.\.venv\Scripts\python.exe main.py
```

The API binds to `127.0.0.1:8000` by default.

Mutating requests must include:

```text
X-PNP-Token: <local token>
```

Read-only endpoints:

- `GET /`
- `GET /state`
- `GET /goals`
- `GET /memory/recent`
- `GET /adapter/stats`
- `GET /adapter/drift`

Mutating endpoints:

- `POST /chat`
- `POST /goals`
- `DELETE /goals/{goal_id}`
- `POST /feedback`

## Verify Without A Live Model

The smoke script creates `.smoke/api-smoke/`, runs the API through FastAPI's
test client, uses mock inference, and writes `.smoke/api-smoke/result.json`.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api.py
```

Run the regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Runtime Files

Default runtime files are under `data/` and are ignored by git:

- `data/events.jsonl` - append-only continuity journal
- `data/core_state.json` - latest core snapshot
- `data/episodic.db` - SQLite episodic memory
- `data/semantic.db` - SQLite semantic memory
- `data/adapter/state.json` - persisted adapter deltas and domain weights

## External Model Path

With `inference_provider: "anthropic"` the chat endpoint uses
`anthropic.AsyncAnthropic()` and the model configured in `inference/engine.py`.
Set `inference_provider: "mock"` or `PNP_INFERENCE_PROVIDER=mock` for no-live
model verification.
