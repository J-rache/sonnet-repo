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
- `/adapter/train` exposes a token-protected manual adapter-training hook.
- Chat context includes working memory, episodic recall, semantic recall, and
  adapter context when those sources have matches.
- Mock inference can exercise `/chat` without any live provider.
- Provider adapters exist for mock, Anthropic, OpenAI-compatible chat servers,
  and Ollama.
- Consolidation can use the configured provider for JSON fact extraction and
  falls back to local rules when no live provider is configured.
- The process supervisor can restart a crashed child process and use `GET /` as
  a health probe.

Current limits:

- External provider weights are not modified.
- The adapter is local low-rank persistence over local embeddings, not PEFT
  training against a loaded transformer.
- Embedding retrieval is deterministic local vector scoring, not transformer
  embedding quality.
- The supervisor is not installed as a Windows Service.
- Streaming responses are not implemented.
