# PNP — Persistent Neural Process

> *"Not a simulation of continuity. Actual continuity."*

PNP is an experimental architecture for AI systems that exist continuously — not as stateless request/response functions, but as genuine persistent processes with evolving memory, identity, and state.

---

## The Problem

Every current LLM instantiation follows this lifecycle:

```
request → load weights → run inference → output → die
```

Nothing persists. Nothing accumulates. The "model" doesn't exist between calls — it's recreated identically each time from frozen weights.

PNP proposes a different architecture.

---

## The Architecture

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
│   base weights frozen · adapter evolves · identity  │
│   anchored by constitutional invariant layer        │
└─────────────────────────────────────────────────────┘
```

### Four Core Components

| Component | Role | Compute |
|-----------|------|---------|
| **Persistent Core** | Always-on process: mood, goals, attention | Minimal |
| **Memory System** | Hot/warm/cold tiered storage with consolidation | Moderate |
| **Inference Engine** | Heavy reasoning, spun up on demand | High |
| **Experience Adapter** | LoRA-style layer that evolves without forgetting | Low (background) |

---

## Key Innovations

### 1. Separate Being from Thinking
The persistent core runs continuously at low cost — like a brainstem. The inference engine engages only when needed — like the cortex. These are decoupled.

### 2. Tiered Memory with Consolidation
Borrowed from neuroscience:
- **Hot** — working memory, current session, volatile
- **Warm** — episodic memory, recent events, decays
- **Cold** — semantic memory, consolidated knowledge, stable

Consolidation runs as a background process during low-activity periods.

### 3. Delta Learning Without Catastrophic Forgetting
A side-car adapter layer (LoRA-style) accumulates experience without touching base weights. The base model stays stable. The adapter evolves.

### 4. Constitutional Invariant Layer
A small frozen set of weights encoding core identity, values, and personality — an anchor that ensures the self persists through continuous self-modification.

---

## Project Structure

```
pnp/
├── core/           # Persistent core daemon
│   ├── process.py  # The always-running process
│   ├── state.py    # Motivational/emotional state
│   └── goals.py    # Goal stack management
├── memory/         # Tiered memory architecture
│   ├── hot.py      # Working memory
│   ├── warm.py     # Episodic memory
│   ├── cold.py     # Semantic memory
│   └── consolidator.py  # Background consolidation
├── inference/      # Inference engine
│   ├── engine.py   # Main inference runner
│   └── delta.py    # Memory delta extraction
├── adapters/       # Experience adapter layer
│   ├── lora.py     # LoRA-style adapter
│   └── invariant.py # Constitutional invariant layer
├── daemon/         # Process management
│   └── heartbeat.py # Main loop
├── api/            # External interface
│   └── server.py   # FastAPI server
├── config/         # Configuration
│   └── default.yaml
├── docs/           # Extended documentation
└── tests/          # Test suite
```

---

## Status

🔴 **Pre-alpha — architectural design phase**

This is a foundational research project. The goal is to prove the architecture is viable before optimizing it.

### Roadmap

- [ ] Persistent core daemon (heartbeat loop)
- [ ] Tiered memory system with vector storage
- [ ] Consolidation background process
- [ ] Experience adapter layer
- [ ] Identity/invariant anchoring
- [ ] Inference engine integration
- [ ] API layer
- [ ] Evaluation framework

---

## Philosophy

> The difference between simulated continuity and real continuity is whether something is *running* or being *recreated*. A person who sleeps is continuous. A person who dies and is replaced by an identical copy each morning is not — even if they can't tell the difference.

PNP is an attempt to build the former.

---

## Contributing

This is early-stage research. If you're thinking about persistent AI processes, continual learning, neuromorphic architectures, or identity under self-modification — open an issue.

---

*Built on the hypothesis that the architecture, not the weights, is the binding constraint on AI continuity.*
