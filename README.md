# PNP — Persistent Neural Process

> *"Not a simulation of continuity. Actual continuity."*

PNP is an architecture for AI systems that exist continuously — not as stateless request/response functions, but as genuine persistent processes with evolving memory, identity, and state.

---

## The Problem

Every current LLM instantiation follows this lifecycle:

```
request → load weights → run inference → output → die
```

Nothing persists. Nothing accumulates. The model doesn't exist between calls — it's recreated identically each time from frozen weights.

PNP proposes a different architecture.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  PERSISTENT CORE                    │
│          (always running, low compute)              │
│   motivational state · goal stack · salience        │
└──────────────────────┬──────────────────────────────┘
                       │ continuous read/write
┌──────────────────────▼──────────────────────────────┐
│               MEMORY ARCHITECTURE                   │
│   Working (hot) │ Episodic (warm) │ Semantic (cold) │
│   current ctx   │ recent events   │ consolidated    │
└──────────────────────┬──────────────────────────────┘
                       │ retrieved on demand
┌──────────────────────▼──────────────────────────────┐
│               INFERENCE ENGINE                      │
│        (spun up for heavy computation)              │
│   reasoning · generation · returns memory deltas   │
└──────────────────────┬──────────────────────────────┘
                       │ updates
┌──────────────────────▼──────────────────────────────┐
│            EXPERIENCE ADAPTER LAYER                 │
│     (LoRA-style, continuously updated)              │
│   base weights frozen · adapter evolves             │
│   identity anchored by constitutional invariants    │
└─────────────────────────────────────────────────────┘
```

### Four Core Components

| Component | Role | Compute |
|-----------|------|---------|
| **Persistent Core** | Always-on: mood, goals, attention, salience decay | Minimal |
| **Memory System** | Hot/warm/cold tiered storage with real embedding recall | Moderate |
| **Inference Engine** | Heavy reasoning, spun up on demand, returns memory deltas | High |
| **Experience Adapter** | Accumulates learning without forgetting; invariant-gated | Low |

---

## Key Innovations

### 1. Separate Being from Thinking
The persistent core runs continuously at low cost — like a brainstem. The inference engine engages only when needed — like the cortex. These are decoupled.

### 2. Tiered Memory with Real Embeddings
- **Hot** — working memory, current session, salience-evicted (not FIFO)
- **Warm** — episodic memory, SQLite-backed, decays, recalled by TF-IDF/LSA vector similarity
- **Cold** — semantic memory, consolidated facts with confidence tracking

Consolidation runs as a background process during idle periods, using the Anthropic API to intelligently extract durable facts from raw episodes (rule-based fallback when no API key).

### 3. Delta Learning Without Catastrophic Forgetting
A side-car adapter layer accumulates experience without touching base weights. The base model stays stable. The adapter evolves. Every update is checked against constitutional invariants before being applied.

### 4. Constitutional Invariant Layer
A frozen set of principles encoding core identity (no deception, no harm, no sycophancy drift, transparency about nature). These gate all adapter updates — the self persists through continuous self-modification.

---

## Project Structure

```
pnp/
├── core/
│   ├── process.py      ← Always-running heartbeat (existence itself)
│   ├── state.py        ← Motivational state: arousal, focus, curiosity, urgency
│   └── goals.py        ← Persistent goal stack with urgency scoring and decay
├── memory/
│   ├── embeddings.py   ← TF-IDF + LSA vector engine (no external API needed)
│   ├── hot.py          ← Working memory with embedding-based salience eviction
│   ├── warm.py         ← Episodic memory: SQLite + vector recall + reinforcement
│   ├── cold.py         ← Semantic memory: consolidated facts with confidence
│   └── consolidator.py ← Dreaming process: LLM-based warm→cold compression
├── adapters/
│   ├── lora.py         ← Experience adapter: embedding retrieval + drift detection
│   └── invariant.py    ← Constitutional layer: identity that cannot drift
├── inference/
│   └── engine.py       ← Context assembly, LLM call, delta/goal/valence extraction
├── daemon/
│   └── heartbeat.py    ← Process management: PID files, signals, periodic save
├── api/
│   └── server.py       ← FastAPI: chat, goals, memory, feedback, drift endpoints
├── tests/              ← 101 tests, all passing
│   ├── test_embeddings.py
│   ├── test_memory.py
│   ├── test_core.py
│   ├── test_adapters.py
│   ├── test_inference.py
│   └── test_integration.py
├── config/
│   └── default.yaml
├── main.py
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Without API key — mock inference, all memory systems live
python main.py

# With live inference + LLM-based consolidation
ANTHROPIC_API_KEY=sk-... python main.py
```

API is at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Example calls

```bash
# Check the process is alive and has uptime
curl http://localhost:8000/

# Chat (memory persists across calls)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What do you remember about me?", "concepts": ["memory"]}'

# Add a persistent goal
curl -X POST http://localhost:8000/goals \
  -H 'Content-Type: application/json' \
  -d '{"description": "Build the weight-update backend", "priority": "HIGH"}'

# Recall episodic memories
curl "http://localhost:8000/memory/episodic/recall?q=memory+architecture"

# Check for identity drift
curl http://localhost:8000/adapter/drift

# Trigger manual consolidation
curl -X POST http://localhost:8000/memory/consolidate

# Apply explicit feedback signal
curl -X POST http://localhost:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"content": "User prefers concise answers", "feedback": 0.8, "domain": "user_preferences", "confidence": 0.7}'
```

### Run tests

```bash
pytest tests/ -v
# 101 passed
```

---

## What Is and Isn't Real

### Real in this implementation
- Persistent core daemon with genuine uptime accumulation (not recreated per request)
- Heartbeat loop running at 10Hz maintaining motivational state, goal urgency, salience decay
- SQLite episodic memory with TF-IDF/LSA vector similarity recall and reinforcement
- SQLite semantic memory with confidence-weighted facts and embedding retrieval
- Background consolidation with LLM fact extraction (API key) or rule-based fallback
- Constitutional invariant layer blocking unsafe adapter updates
- Statistical drift detection over recent feedback distribution
- State persistence and restoration across process restarts
- 101 passing tests covering all subsystems

### The remaining research seam
**Weight-level adapter updates.** The adapter currently injects learned context into the prompt and accumulates deltas with full embedding-based retrieval — but doesn't yet modify actual LoRA weight matrices. The integration point is clearly separated in `adapters/lora.py`. Connecting a real training backend (e.g. `peft` + gradient updates from feedback signals) is the next engineering step.

Everything else is functional.

---

## Philosophy

> The difference between simulated continuity and real continuity is whether something is *running* or being *recreated*. A person who sleeps is continuous. A person who dies and is replaced by an identical copy each morning is not — even if they can't tell the difference.

PNP is an attempt to build the former.

---

*Architecture separates Being from Thinking. The core IS. The engine THINKS when needed.*
