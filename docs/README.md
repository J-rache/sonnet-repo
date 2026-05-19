# Runtime Truth

This repo proves a local Persistent Neural Process runtime slice. It is not a
hosted production service and it is not a way to alter private weights owned by
external model providers.

Implemented behavior:

- The API starts `PersistentCore`, `EpisodicMemory`, `SemanticMemory`, and
  `ExperienceAdapter`.
- The default API host is `127.0.0.1`.
- Mutating endpoints require `X-PNP-Token` by default.
- Runtime mutation events are appended to a durable JSONL journal.
- Core shutdown writes a snapshot.
- Core startup restores the snapshot and replays newer journal events.
- Working, episodic, and semantic memory use deterministic local vector
  retrieval.
- Adapter deltas are saved to disk and loaded on restart.
- Adapter feedback trains a persisted local low-rank additive adapter.
- Adapter training writes a durable sync-model pack plus reloadable sync context
  that binds prior sessions to the selected model/provider on startup.
- A PEFT/LoRA backend exists for local trainable Hugging Face-style runtimes
  when optional training dependencies and a local base model are configured.
- `/adapter/train` exposes a token-protected manual adapter-training hook.
- `/adapter/sync` exposes the current sync-model pack and context.
- Chat context includes working memory, episodic recall, semantic recall, and
  adapter context when those sources have matches.
- Mock inference can exercise `/chat` without any live provider.
- Provider adapters exist for mock, Anthropic, OpenAI-compatible chat servers,
  and Ollama.
- Consolidation can use the configured provider for JSON fact extraction and
  falls back to local rules when no live provider is configured.
- Project continuity supports many participant lanes inside one running service.
- Participant continuity is keyed by `project_id + participant_identity`; seat
  ids are only temporary routing bindings.
- New project participants hydrate from stenographer summary/history. Returning
  participants restore their own lane first and then receive newer project
  summary updates.
- Project memory, participant continuity, seat bindings, shared toolbox, and
  lessons learned are persisted as separate layers.
- Project archive/restore includes project memory, participant lanes, seat
  bindings, toolbox entries, lessons, and archive metadata.
- `scripts/smoke_mystro_table_sync.py` performs an optional live Mystro Table
  integration smoke against `127.0.0.1:8787`, installed PNP on `127.0.0.1:8000`,
  and Ollama `qwen2.5-coder:7b`, including multi-seat participant binding,
  model-backed chat, seat-move continuity, replacement-seat isolation, toolbox,
  lessons, archive/restore, and a restart verification mode.
- The process supervisor can restart a crashed child process and use `GET /` as
  a health probe.
- `install.ps1` installs the app into a per-user standalone layout with copied
  source, a dedicated venv, installed config, launchers, dependency install, and
  optional Desktop shortcuts.

Runtime boundaries:

- Hidden provider weights are not modified. Providers without a local training
  surface are synchronized through context, memory, and adapter state.
- PEFT/LoRA weight training requires a local trainable model runtime; Ollama/GGUF
  inference endpoints do not expose gradients to update model files.
- Embedding retrieval is deterministic local vector scoring, not transformer
  embedding quality.
- The supervisor is not installed as a Windows Service.
- Streaming responses are not implemented.
