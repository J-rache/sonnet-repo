# Runtime Truth

This repo currently proves a local PNP runtime slice, not a hosted production
system or a way to modify external provider weights.

Implemented runtime behavior:

- FastAPI startup constructs `PersistentCore`, `EpisodicMemory`,
  `SemanticMemory`, and `ExperienceAdapter`.
- Mutating API endpoints require `X-PNP-Token`.
- Runtime mutation events are appended to a JSONL journal.
- Core shutdown writes a snapshot, and startup replays journal events newer
  than the snapshot.
- Working, episodic, and semantic memory use deterministic local embeddings for
  vector retrieval.
- Adapter deltas are saved to disk and loaded on restart.
- Adapter feedback trains a persisted low-rank additive adapter model over
  local embeddings.
- Chat context includes semantic-memory retrieval when matching facts exist.
- Mock inference can exercise `/chat` without any live provider.
- Provider adapters exist for mock, Anthropic, OpenAI-compatible chat
  completions, and Ollama.
- The supervisor can restart a crashed child process and can use `GET /` as a
  health probe.

Not implemented:

- External provider weight updates.
- Transformer embedding model quality.
- Windows Service installation.
- Streaming response delivery.
