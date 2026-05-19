# PNP - Persistent Neural Process

PNP is a local runtime for an AI process with persistent core state, tiered
memory, replayable continuity events, vector retrieval, and a trainable
low-rank adaptation layer.

The project is intentionally local-first. It can verify startup, memory,
adapter training, API auth, restart replay, and watchdog behavior without a
live model call. For real generation, set `inference_provider` and `model_id`
for the provider you want.

## What Works Now

- FastAPI startup initializes the persistent core, episodic memory, semantic
  memory, and experience adapter.
- Runtime state changes are written to an append-only JSONL event journal.
- Shutdown writes a core snapshot, and startup restores the snapshot plus
  journal events that happened after the snapshot.
- Working, episodic, and semantic memory use deterministic local vector
  embeddings and cosine retrieval.
- Adapter deltas are persisted in `adapter/state.json`.
- Adapter feedback trains a persisted low-rank additive adapter model in
  `adapter/low_rank_adapter.json`.
- Chat inference receives working-memory, episodic, semantic, adapter, and core
  state context.
- Mutating endpoints require a local API token.
- The default API bind is `127.0.0.1`.
- A watchdog supervisor can restart the local API process if it exits or fails
  repeated health checks.
- Tests and smoke scripts verify the API, continuity, vector retrieval, adapter
  training, and supervisor paths without a live model call.

## Boundaries

- The low-rank adapter is real trainable local adapter math over PNP text
  embeddings. It influences provider context and scoring; it does not modify
  external provider weights.
- The default embedder is deterministic and dependency-free. It is vector
  retrieval, not transformer embedding quality.
- Consolidation uses rule-based fact extraction. No separate LLM extractor is
  called during tests or smokes.
- `scripts/start_supervisor.ps1` runs a foreground watchdog. It is not a Windows
  Service installer.
- The default token in `config/default.yaml` is for local development. Set
  `PNP_LOCAL_TOKEN` for real local use.

## Project Structure

```text
adapters/
  invariant.py       Constitutional update gate
  lora.py            Experience delta store and adapter interface
  low_rank.py        Trainable low-rank adapter math
api/
  server.py          FastAPI app and local auth boundary
config/
  default.yaml       Local runtime defaults
core/
  goals.py           Goal stack
  journal.py         Append-only JSONL continuity journal
  process.py         Persistent core loops, snapshot, and replay
  state.py           Motivational state vector
daemon/
  supervisor.py      Local watchdog supervisor
docs/
  README.md          Runtime truth notes
inference/
  engine.py          Context assembly and external model call
memory/
  embedding.py       Deterministic local text embeddings
  hot.py             Working memory
  warm.py            SQLite episodic memory with vectors
  cold.py            SQLite semantic memory with vectors
  consolidator.py    Rule-based memory consolidation
scripts/
  smoke_api.py       No-live-model API smoke
  smoke_supervisor.py
  start_supervisor.ps1
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
- `GET /memory/semantic`
- `GET /adapter/stats`
- `GET /adapter/drift`

Mutating endpoints:

- `POST /chat`
- `POST /goals`
- `DELETE /goals/{goal_id}`
- `POST /feedback`
- `POST /adapter/train`

## Run With Watchdog

```powershell
$env:PNP_LOCAL_TOKEN = "replace-with-local-secret"
.\scripts\start_supervisor.ps1
```

The supervisor starts `main.py`, checks `GET /`, and restarts the process after
crashes or repeated health failures.

## Verify Without A Live Model

The API smoke creates `.smoke/api-smoke/`, runs the API through FastAPI's test
client, uses mock inference, and writes `.smoke/api-smoke/result.json`.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_api.py
```

The supervisor smoke verifies restart behavior with a child process that exits:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_supervisor.py
```

Run the regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Runtime Files

Default runtime files are under `data/` and are ignored by git:

- `data/events.jsonl` - append-only continuity journal
- `data/core_state.json` - latest core snapshot
- `data/episodic.db` - SQLite episodic memory and embeddings
- `data/semantic.db` - SQLite semantic memory and embeddings
- `data/adapter/state.json` - persisted adapter deltas and domain weights
- `data/adapter/low_rank_adapter.json` - trained low-rank adapter weights

## Provider Path

Supported provider selectors:

- `mock` - no-live-model verification.
- `anthropic` - Anthropic Messages API through the installed SDK.
- `openai_compatible` - `/v1/chat/completions` compatible HTTP providers,
  including many local gateways.
- `ollama` - local Ollama `/api/chat`.

Set `PNP_INFERENCE_PROVIDER` and `PNP_MODEL_ID` to override config at runtime.
