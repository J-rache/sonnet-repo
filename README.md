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
- `project/continuity.py` owns project-wide memory, participant continuity
  lanes, temporary seat bindings, shared toolbox entries, lessons learned, and
  project archive/restore.
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

## Install As A Standalone Local App

The install script creates a per-user app under `%LOCALAPPDATA%\PNP` by default:

- copies the tracked application source into `app\`
- creates an isolated `.venv\`
- installs Python dependencies from `requirements.txt`
- installs the package entry point in that venv
- writes an installed config with data paths under the install root
- writes Start, Supervisor, Smoke, Status, and Docs launchers
- creates Desktop shortcuts unless `-NoDesktopShortcuts` is passed

Default mock install:

```powershell
.\install.ps1
```

Install using any local Ollama model:

```powershell
.\install.ps1 -Provider ollama -ModelId qwen2.5-coder:7b
```

Install with optional local PEFT/LoRA training dependencies for a trainable
Hugging Face-style sync model:

```powershell
.\install.ps1 -Provider ollama -ModelId qwen2.5-coder:7b `
  -InstallTrainingDeps `
  -SyncModelBase "C:\path\to\local\hf-causal-lm"
```

Install without Desktop shortcuts:

```powershell
.\install.ps1 -NoDesktopShortcuts
```

Installed launchers:

```powershell
& "$env:LOCALAPPDATA\PNP\PNP-Start.ps1"
& "$env:LOCALAPPDATA\PNP\PNP-Supervisor.ps1"
& "$env:LOCALAPPDATA\PNP\PNP-Status.ps1"
& "$env:LOCALAPPDATA\PNP\PNP-Smoke.ps1"
```

The installed config is at `%LOCALAPPDATA%\PNP\config\installed.yaml`. The API
token is stored there as `local_api_token`; `PNP_LOCAL_TOKEN` can override it.

## Sync Model Adapter

PNP treats the durable brain as journal + memory + adapter state, not as a
single provider's hidden weights. On every adapter train, it writes:

- `sync_model_pack.json`: model/provider binding, delta counts, low-rank metrics,
  and optional PEFT LoRA metadata
- `sync_model_context.md`: compact learned state injected into future inference
  context
- `low_rank_adapter.json`: the dependency-free local low-rank adapter weights

For Ollama/GGUF models, the sync model binds through this persisted adapter
pack and context. For local Hugging Face-style causal-LM runtimes, set
`sync_model_adapter_backend: peft_lora` plus `sync_model_base_model` and install
the `train` extras; PNP will train and save a real PEFT LoRA adapter under the
adapter directory.

## Project Continuity

One running service can hold a whole Mystro-style project. Project state is
separate from participant state:

- Project memory stores facts, decisions, goals, current state, history,
  stenographer summaries, and archive metadata.
- Participant continuity is keyed by `project_id + participant_identity` and
  owns that participant's journal, episodic/semantic memory, adapter/sync state,
  and resume point.
- Seat bindings are temporary routing, for example `seat-A -> participant-X`.
  Seat ids never own memory.
- A participant that moves seats keeps its own lane. A new participant entering
  an old seat starts or restores its own lane, not the old occupant's lane.
- A first-time participant is hydrated from stenographer project summary/history.
  A returning participant restores its own lane first, then receives any newer
  project summary update.
- Shared toolbox entries and lessons learned are project-scoped. Toolbox entries
  can name allowed participants, are not seat-scoped, and are not promoted to
  global tools or global lessons by default.
- Project archive/restore includes project memory, participant lanes, seat
  bindings, toolbox entries, lessons, and archive metadata.

## Live Mystro Table Smoke

When `C:\Users\Jae\Desktop\mystro-table` is running and Ollama has
`qwen2.5-coder:7b` installed, this repo includes a live end-to-end smoke that
loads Mystro participant seats, binds those seats to PNP participant identities,
runs Qwen-backed PNP chat, checks seat-move continuity, checks replacement-seat
isolation, records a project toolbox tool and lesson, archives/restores the
project, and can re-check continuity after a PNP restart:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_mystro_table_sync.py
.\.venv\Scripts\python.exe scripts\smoke_mystro_table_sync.py --verify-existing
```

The smoke writes its evidence to `.smoke\mystro_table_sync\result.json`. It is
optional and specific to a local Mystro Table + Ollama setup; PNP itself remains
provider-neutral and can use any configured provider/runtime.

## Run From Source

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
`GET /adapter/sync` shows the current sync-model pack and context.

Project continuity endpoints use the same local token for mutations:

```powershell
curl.exe -X POST http://127.0.0.1:8000/projects `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"project_id\":\"mystro-project\"}"

curl.exe -X POST http://127.0.0.1:8000/projects/mystro-project/stenographer/summary `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"summary\":\"Current table state and verified project history.\"}"

curl.exe -X POST http://127.0.0.1:8000/projects/mystro-project/seats/seat-A/bind `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"participant_identity\":\"provider:model:worker-1\",\"model_id\":\"qwen2.5-coder:7b\"}"

curl.exe -X POST http://127.0.0.1:8000/projects/mystro-project/seats/seat-A/chat `
  -H "Content-Type: application/json" `
  -H "X-PNP-Token: $token" `
  -d "{\"message\":\"Continue from your participant lane and project history.\"}"
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe scripts\smoke_api.py
.\.venv\Scripts\python.exe scripts\smoke_supervisor.py
.\.venv\Scripts\python.exe scripts\smoke_ollama_qwen.py
.\scripts\smoke_install.ps1
```

`scripts\smoke_api.py` writes artifacts under `.smoke\api-smoke\`.
`scripts\smoke_ollama_qwen.py` writes artifacts under
`.smoke\ollama-qwen-smoke\` and can be pointed at another installed Ollama model
with `PNP_SMOKE_OLLAMA_MODEL`.
`scripts\smoke_install.ps1` performs a real install into `.install-smoke\PNP`,
installs dependencies, runs the installed smoke launcher, and removes the smoke
install unless `-Keep` is passed.

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
- Adapter training writes a sync-model adapter pack and reloadable sync context
  that binds prior sessions to whichever model/provider is selected on startup.
- Optional PEFT/LoRA training is wired for local trainable Hugging Face-style
  runtimes when training dependencies and a local base model are configured.
- Project continuity supports many participant lanes inside one service
  instance, keyed by `project_id + participant_identity` rather than seat.
- Seat bindings are a remappable routing layer. Moving a participant to another
  seat preserves that participant's lane; replacing a seat occupant does not
  inherit the old occupant's continuity.
- First-time project participants hydrate from stenographer summary/history.
  Returning participants restore their own lane and then take newer project
  summary updates.
- Project-scoped shared toolbox and lessons learned persist separately from
  participant-specific memory and adapter deltas.
- Project archive/restore preserves project memory, participant lanes, seat
  bindings, toolbox entries, lessons, and archive metadata.
- The supervisor can restart a crashed process and run health checks.
- `install.ps1` installs PNP as a per-user standalone local app with its own
  venv, installed config, launchers, optional Desktop shortcuts, and smoke
  command.

Runtime boundaries:

- PNP does not claim access to hidden provider weights. Provider sync happens
  through the durable sync pack, retrieved memory, and adapter context.
- PEFT/LoRA weight training is available for local trainable model runtimes, not
  for runtimes such as Ollama/GGUF that expose inference but not gradients.
- Local embedding retrieval uses deterministic TF-IDF/LSA style vectors and
  hash fallback, not a transformer embedding model.
- The supervisor is a process supervisor, not an installed Windows Service.
- Streaming chat responses are not implemented.
