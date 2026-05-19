# PNP - Persistent Neural Process

PNP is a local runtime for a continuously running AI process. It keeps a
persistent core alive between requests, stores working/episodic/semantic memory,
journals mutating events, and routes inference through configurable providers.

This repo does not modify private provider weights. It trains and persists a
local low-rank adapter over experience deltas, then uses those learned signals as
part of retrieval and inference context.

## What Runs

- `main.py` starts the FastAPI app and persistent core.
- `api/server.py` exposes local API endpoints for chat, state, goals, memory,
  feedback, adapter training, and adapter inspection.
- `core/process.py` runs heartbeat, motivational state, salience, goals,
  snapshots, and journal replay.
- `memory/` contains working memory, SQLite episodic memory, SQLite semantic
  memory, deterministic local embedding retrieval, and consolidation.
- `adapters/lora.py` stores feedback deltas, gates them with invariants, trains a
  persisted low-rank adapter model, and returns adaptation context.
- `inference/providers.py` supports `mock`, `anthropic`, `openai_compatible`,
  and `ollama` providers.
- `daemon/supervisor.py` can restart a crashed child process and perform health
  checks.

## Local Defaults

The default config is intentionally local-first:

- API host: `127.0.0.1`
- Inference provider: `mock`
- Model id: `local-model`
- Mutating endpoint token: `local_api_token` from config or `PNP_LOCAL_TOKEN`

The mock provider lets the API, memory, journaling, replay, and adapter training
be verified without a live model call.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Open API docs at `http://127.0.0.1:8000/docs`.

For a local Ollama model, set any installed model id. On this machine the smoke
uses `qwen2.5-coder:7b`:

```powershell
$env:PNP_INFERENCE_PROVIDER = "ollama"
$env:PNP_MODEL_ID = "qwen2.5-coder:7b"
$env:PNP_LOCAL_TOKEN = "change-this-local-token"
.\.venv\Scripts\python.exe main.py
```

OpenAI-compatible servers can be used with:

```powershell
$env:PNP_INFERENCE_PROVIDER = "openai_compatible"
$env:PNP_OPENAI_COMPATIBLE_BASE = "http://127.0.0.1:8001/v1"
$env:PNP_MODEL_ID = "your-local-model"
```

## API Examples

Read-only endpoints do not need the token:

```powershell
curl.exe http://127.0.0.1:8000/
curl.exe http://127.0.0.1:8000/state
curl.exe http://127.0.0.1:8000/goals
curl.exe "http://127.0.0.1:8000/memory/semantic?q=concise+notes"
```

Mutating endpoints require `X-PNP-Token`:

```powershell
$token = "dev-local-token-change-me"

curl.exe -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"message\":\"What do you remember about concise notes?\",\"concepts\":[\"memory\"]}"

curl.exe -X POST http://127.0.0.1:8000/goals `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"description\":\"Keep PNP verification green\",\"priority\":\"HIGH\"}"

curl.exe -X POST http://127.0.0.1:8000/feedback `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"content\":\"Use concise implementation notes.\",\"feedback\":0.8,\"domain\":\"user_preferences\",\"confidence\":0.8}"

curl.exe -X POST http://127.0.0.1:8000/adapter/train `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"epochs\":5}"
```

Protected mutating endpoints include `/chat`, `/goals`, goal progress/deletion,
manual episodic memory writes, manual consolidation, `/feedback`, and
`/adapter/train`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe scripts\smoke_api.py
.\.venv\Scripts\python.exe scripts\smoke_supervisor.py
.\.venv\Scripts\python.exe scripts\smoke_ollama_qwen.py
```

`scripts\smoke_api.py` writes artifacts under `.smoke\api-smoke\`.
`scripts\smoke_ollama_qwen.py` writes artifacts under
`.smoke\ollama-qwen-smoke\` and can be pointed at another installed Ollama model
with `PNP_SMOKE_OLLAMA_MODEL`.

## Runtime Truth

Implemented:

- FastAPI startup constructs the persistent core, memory layers, and adapter.
- Local mock inference exercises `/chat` without a live provider.
- Provider routing supports Anthropic, OpenAI-compatible chat servers, Ollama,
  and mock.
- Semantic memory retrieval is injected into chat context when matching facts
  exist.
- Episodic memory retrieval and adapter context are also injected into chat.
- Mutating endpoints are token-protected by default.
- Runtime events are appended to a JSONL journal.
- Shutdown writes a core snapshot.
- Startup restores the snapshot and replays newer journal events, including
  interactions, goal creation, goal progress, goal completion, and
  consolidation counts.
- Adapter deltas persist across restart.
- Adapter training persists a local low-rank adapter model and exposes
  `/adapter/train`.
- The supervisor can restart a crashed process and run health checks.

Current limits:

- PNP does not train or modify external provider weights, including Claude,
  OpenAI-compatible hosted models, or Ollama model files.
- The local adapter is structured low-rank adapter persistence over local
  embeddings, not PEFT/gradient training inside a transformer runtime.
- Local embedding retrieval uses deterministic TF-IDF/LSA style vectors and
  hash fallback, not a transformer embedding model.
- The supervisor is a process supervisor, not an installed Windows Service.
- Streaming chat responses are not implemented.
