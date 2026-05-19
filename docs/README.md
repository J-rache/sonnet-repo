# Runtime Truth

This repo currently proves a local PNP vertical slice, not a complete
production persistent-AI system.

Implemented runtime behavior:

- FastAPI startup constructs `PersistentCore`, `EpisodicMemory`,
  `SemanticMemory`, and `ExperienceAdapter`.
- Mutating API endpoints require `X-PNP-Token`.
- Runtime mutation events are appended to a JSONL journal.
- Core shutdown writes a snapshot, and startup replays journal events newer
  than the snapshot.
- Adapter deltas are saved to disk and loaded on restart.
- Chat context includes semantic-memory retrieval when matching facts exist.
- Mock inference can exercise `/chat` without Anthropic or another live model.

Not implemented yet:

- Real LoRA weight training or adapter weight application.
- Vector embeddings for memory retrieval.
- Service installation, process supervision, or cross-process locking.
- Streaming response delivery.
