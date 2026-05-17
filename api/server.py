"""
api/server.py

External API — the interface between the persistent process and the outside world.

The persistent core runs independently of this. The API is just
a window into it: you can talk to it, check its state, add goals,
inspect memory. But whether or not anyone is calling this API,
the core keeps running.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PNP — Persistent Neural Process",
    description="Interface to a continuously-running AI process",
    version="0.1.0",
)

# The persistent core — initialized at startup, runs until shutdown
_core = None
_adapter = None


@app.on_event("startup")
async def startup():
    global _core, _adapter
    import yaml
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    from core.process import PersistentCore
    from adapters.lora import ExperienceAdapter

    _core = PersistentCore(config)
    _adapter = ExperienceAdapter(
        base_model_id=config.get("base_model", "claude-sonnet-4-20250514"),
        config=config
    )

    # Start the persistent core in background
    asyncio.create_task(_core.start())
    logger.info("PNP API online. Persistent core running.")


@app.on_event("shutdown")
async def shutdown():
    if _core:
        await _core.stop()


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    concepts: list[str] = []
    stream: bool = False


class GoalRequest(BaseModel):
    description: str
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH, URGENT
    deadline_hours: Optional[float] = None


class FeedbackRequest(BaseModel):
    content: str
    feedback: float   # -1.0 to 1.0
    domain: str = "general"
    confidence: float = 0.5


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check — also shows that the process is alive and has uptime."""
    if not _core:
        return {"status": "initializing"}
    state = _core.get_state_snapshot()
    return {
        "status": "alive",
        "uptime_hours": round(state["uptime_seconds"] / 3600, 2),
        "mode": state["motivational_state"]["mode"],
        "heartbeats": state["heartbeat_count"],
    }


@app.get("/state")
async def get_state():
    """Full core state snapshot."""
    if not _core:
        raise HTTPException(503, "Core not initialized")
    return _core.get_state_snapshot()


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a message to the persistent process.

    Unlike a standard LLM API, this updates the persistent state:
    - Working memory is updated
    - Episodic memory records the interaction
    - Salience map is updated
    - Motivational state responds to the interaction
    """
    if not _core:
        raise HTTPException(503, "Core not initialized")

    import anthropic

    # 1. Notify core of interaction — updates state, salience, working memory
    core_state = _core.on_interaction(
        content=request.message,
        metadata={"role": "user", "concepts": request.concepts}
    )

    # 2. Retrieve relevant memory
    episodic_context = ""
    semantic_context = ""

    try:
        episodes = _core.episodic_memory.recall(request.message, limit=3)
        if episodes:
            ep_parts = ["=== RELEVANT MEMORIES ==="]
            for ep in episodes:
                ep_parts.append(f"[{ep.tags}] {ep.summary}")
            ep_parts.append("=== END MEMORIES ===")
            episodic_context = "\n".join(ep_parts)
    except Exception as e:
        logger.warning(f"Episodic recall failed: {e}")

    # 3. Get adaptation context
    adaptation_context = ""
    if _adapter:
        adaptation_context = _adapter.get_adaptation_context(request.message)

    # 4. Build inference request
    from inference.engine import InferenceRequest, run_inference

    inference_req = InferenceRequest(
        user_input=request.message,
        working_memory_context=_core.working_memory.to_context_string(),
        episodic_context=episodic_context,
        semantic_context=semantic_context,
        adaptation_context=adaptation_context,
        core_state=core_state,
        stream=request.stream,
    )

    # 5. Run inference
    client = anthropic.AsyncAnthropic()
    result = await run_inference(inference_req, client)

    # 6. Store interaction in episodic memory
    _core.episodic_memory.store(
        content=request.message,
        summary=result.memory_deltas[0]["summary"] if result.memory_deltas else request.message[:100],
        tags=["interaction"] + request.concepts,
        valence=result.valence,
        salience=0.8,
    )

    # 7. Store response in working memory
    _core.working_memory.add(result.content, {"role": "assistant"})

    return {
        "response": result.content,
        "tokens_used": result.tokens_used,
        "latency_ms": round(result.latency_ms),
        "core_state": {
            "mode": core_state["motivational_state"]["mode"],
            "uptime_hours": round(core_state["uptime_seconds"] / 3600, 2),
        }
    }


@app.post("/goals")
async def add_goal(request: GoalRequest):
    """Add a goal to the persistent goal stack."""
    if not _core:
        raise HTTPException(503, "Core not initialized")

    from core.goals import GoalPriority
    import time

    priority = GoalPriority[request.priority]
    deadline = time.time() + (request.deadline_hours * 3600) if request.deadline_hours else None

    goal = _core.goals.add(
        description=request.description,
        priority=priority,
        deadline=deadline,
    )
    return {"goal_id": goal.id, "description": goal.description, "priority": goal.priority.name}


@app.delete("/goals/{goal_id}")
async def complete_goal(goal_id: str, notes: str = ""):
    """Mark a goal as complete."""
    if not _core:
        raise HTTPException(503, "Core not initialized")
    _core.goals.complete(goal_id, notes)
    return {"status": "completed", "goal_id": goal_id}


@app.get("/memory/recent")
async def recent_memory(hours: float = 24):
    """View recent episodic memory."""
    if not _core:
        raise HTTPException(503, "Core not initialized")
    episodes = _core.episodic_memory.recent(hours=hours)
    return {
        "episodes": [ep.to_dict() for ep in episodes],
        "stats": _core.episodic_memory.stats(),
    }


@app.post("/feedback")
async def apply_feedback(request: FeedbackRequest):
    """
    Apply a learning signal to the experience adapter.

    This is how the system learns from explicit feedback —
    the delta is checked against constitutional invariants before
    being applied to the adapter layer.
    """
    if not _adapter:
        raise HTTPException(503, "Adapter not initialized")

    from adapters.lora import ExperienceDelta

    delta = ExperienceDelta(
        content=request.content,
        feedback=request.feedback,
        domain=request.domain,
        confidence=request.confidence,
    )

    allowed = _adapter.apply_delta(delta)
    return {
        "applied": allowed,
        "blocked_by_invariant": not allowed,
        "adapter_stats": _adapter.stats(),
    }


@app.get("/adapter/stats")
async def adapter_stats():
    """View experience adapter state."""
    if not _adapter:
        raise HTTPException(503, "Adapter not initialized")
    return _adapter.stats()


@app.get("/adapter/drift")
async def check_drift():
    """Check for identity drift in the adapter."""
    if not _adapter:
        raise HTTPException(503, "Adapter not initialized")
    drift = _adapter.detect_drift()
    return {
        "drift_detected": drift is not None,
        "drift_report": drift,
    }
