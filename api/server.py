"""
api/server.py

External API — the interface between the persistent process and the outside world.

The persistent core runs independently of this API. The API is a window into it:
chat, inspect state, add goals, apply feedback, view memory. Whether or not
anyone calls this API, the core keeps running.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_core = None
_adapter = None
_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _core, _adapter, _client

    import yaml
    config_path = os.environ.get("PNP_CONFIG", "config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from core.process import PersistentCore
    from adapters.lora import ExperienceAdapter

    _core = PersistentCore(config)
    _adapter = ExperienceAdapter(
        base_model_id=config.get("base_model", "claude-haiku-4-5-20251001"),
        config=config,
    )

    # Anthropic client — available if ANTHROPIC_API_KEY is set
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        import anthropic
        _client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("Anthropic API client ready.")
    else:
        logger.warning("ANTHROPIC_API_KEY not set — inference will use mock responses.")

    core_task = asyncio.create_task(_core.start())
    logger.info("PNP API online. Persistent core running.")

    yield

    core_task.cancel()
    try:
        await core_task
    except asyncio.CancelledError:
        pass
    await _core.stop()


app = FastAPI(
    title="PNP — Persistent Neural Process",
    description="Interface to a continuously-running AI process with persistent memory and identity",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    concepts: list[str] = Field(default_factory=list)
    model: Optional[str] = None

class GoalRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=500)
    priority: str = Field(default="MEDIUM", pattern="^(LOW|MEDIUM|HIGH|URGENT)$")
    deadline_hours: Optional[float] = Field(default=None, gt=0)

class GoalProgressRequest(BaseModel):
    progress: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""

class FeedbackRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    feedback: float = Field(..., ge=-1.0, le=1.0)
    domain: str = Field(default="general")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

class EpisodeRequest(BaseModel):
    content: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    salience: float = Field(default=0.8, ge=0.0, le=1.0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_core():
    if _core is None:
        raise HTTPException(503, detail="Persistent core not initialized")
    return _core

def _require_adapter():
    if _adapter is None:
        raise HTTPException(503, detail="Experience adapter not initialized")
    return _adapter


async def _mock_inference(user_input: str, core_state: dict) -> str:
    """Deterministic mock response when no API key is available."""
    mode = core_state.get("motivational_state", {}).get("mode", "nominal")
    uptime_h = round(core_state.get("uptime_seconds", 0) / 3600, 2)
    interactions = core_state.get("total_interactions", 0)
    return (
        f"[MOCK — set ANTHROPIC_API_KEY for real inference]\n\n"
        f"I received: '{user_input}'\n\n"
        f"Core state: mode={mode}, uptime={uptime_h}h, "
        f"interactions={interactions}, "
        f"heartbeats={core_state.get('heartbeat_count', 0):,}"
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
async def root():
    """Health check — shows the process is alive and has real uptime."""
    core = _require_core()
    state = core.get_state_snapshot()
    return {
        "status": "alive",
        "uptime_hours": round(state["uptime_seconds"] / 3600, 3),
        "mode": state["motivational_state"]["mode"],
        "heartbeats": state["heartbeat_count"],
        "total_interactions": state["total_interactions"],
        "active_goals": state["active_goals"].__len__(),
        "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.get("/state", tags=["Status"])
async def get_state():
    """Full core state snapshot."""
    return _require_core().get_state_snapshot()


@app.get("/health", tags=["Status"])
async def health():
    """Minimal health check for monitoring."""
    if _core is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "uptime_seconds": round(_core.metrics.uptime_seconds, 1)}


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post("/chat", tags=["Interaction"])
async def chat(request: ChatRequest):
    """
    Send a message to the persistent process.

    Unlike a stateless LLM call, this:
    - Updates persistent motivational state
    - Stores the interaction in episodic memory
    - Retrieves relevant memories to inform the response
    - Applies learned adaptations from the experience adapter
    - Integrates any new goals mentioned in the response
    """
    core = _require_core()
    adapter = _require_adapter()

    # 1. Notify core — updates state, salience, working memory
    core_state = core.on_interaction(
        content=request.message,
        metadata={"role": "user", "concepts": request.concepts}
    )

    # 2. Retrieve relevant episodic memories
    episodic_context = ""
    try:
        episodes = core.episodic_memory.recall(request.message, limit=4, min_salience=0.05)
        if episodes:
            lines = ["=== RELEVANT MEMORIES ==="]
            for ep in episodes:
                lines.append(f"[{', '.join(ep.tags)}] {ep.summary}")
            lines.append("=== END MEMORIES ===")
            episodic_context = "\n".join(lines)
    except Exception as e:
        logger.warning(f"Episodic recall error: {e}")

    # 3. Retrieve relevant semantic facts
    semantic_context = ""
    try:
        facts = core.consolidator.semantic.retrieve(request.message, limit=3)
        if facts:
            lines = ["=== KNOWN FACTS ==="]
            for f in facts:
                lines.append(f"[conf={f.confidence:.2f}] {f.content}")
            lines.append("=== END FACTS ===")
            semantic_context = "\n".join(lines)
    except Exception as e:
        logger.warning(f"Semantic recall error: {e}")

    # 4. Get adaptation context
    adaptation_context = adapter.get_adaptation_context(request.message)

    # 5. Run inference
    from inference.engine import InferenceRequest, run_inference

    model = request.model or os.environ.get("PNP_MODEL", "claude-haiku-4-5-20251001")

    inf_req = InferenceRequest(
        user_input=request.message,
        working_memory_context=core.working_memory.to_context_string(),
        episodic_context=episodic_context,
        semantic_context=semantic_context,
        adaptation_context=adaptation_context,
        core_state=core_state,
        model=model,
    )

    if _client:
        from inference.engine import run_inference
        result = await run_inference(inf_req, _client)
        response_text = result.content
        tokens_used = result.tokens_used
        latency_ms = round(result.latency_ms)
        suggested_goals = result.suggested_goals
        valence = result.valence
        memory_deltas = result.memory_deltas
    else:
        response_text = await _mock_inference(request.message, core_state)
        tokens_used = 0
        latency_ms = 0
        suggested_goals = []
        valence = 0.0
        memory_deltas = [{
            "type": "episodic", "content": request.message,
            "summary": request.message[:100],
            "tags": ["interaction"], "durable_facts": [], "goals_mentioned": [],
        }]

    # 6. Store interaction in episodic memory
    delta = memory_deltas[0] if memory_deltas else {}
    try:
        core.episodic_memory.store(
            content=request.message,
            summary=delta.get("summary", request.message[:100]),
            tags=delta.get("tags", ["interaction"]),
            valence=valence,
            salience=0.85,
        )
    except Exception as e:
        logger.error(f"Episode store failed: {e}")

    # 7. Store durable facts from this interaction
    for fact_text in delta.get("durable_facts", []):
        try:
            core.consolidator.semantic.store_fact(
                content=fact_text,
                source_episode_ids=[],
                domain="general",
                confidence=0.5,
            )
        except Exception as e:
            logger.warning(f"Fact store failed: {e}")

    # 8. Store response in working memory
    core.working_memory.add(response_text, {"role": "assistant"})

    # 9. Integrate inference results back into core state
    core.on_inference_result(suggested_goals=suggested_goals, valence=valence)

    # 10. Apply feedback delta to adapter (implicit: positive for helpful completion)
    from adapters.lora import ExperienceDelta
    if request.message and response_text:
        delta_obj = ExperienceDelta(
            content=f"Q: {request.message[:100]} A: {response_text[:100]}",
            feedback=0.3,  # Neutral-positive for completing a request
            domain="interaction",
            confidence=0.4,
        )
        adapter.apply_delta(delta_obj, invariant_check=True)

    return {
        "response": response_text,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "core_state": {
            "mode": core_state["motivational_state"]["mode"],
            "uptime_hours": round(core_state["uptime_seconds"] / 3600, 3),
            "total_interactions": core_state["total_interactions"],
            "active_goals": len(core_state["active_goals"]),
        },
        "memory": {
            "episodic_retrieved": len(episodes) if "episodes" in dir() else 0,
            "semantic_retrieved": len(facts) if "facts" in dir() else 0,
            "adaptations_applied": bool(adaptation_context),
            "new_goals_from_response": len(suggested_goals),
        },
    }


# ── Goals ──────────────────────────────────────────────────────────────────────

@app.get("/goals", tags=["Goals"])
async def list_goals():
    """List all active goals."""
    core = _require_core()
    return {
        "active_goals": core.goals.to_list(),
        "active_count": core.goals.active_count,
    }


@app.post("/goals", tags=["Goals"])
async def add_goal(request: GoalRequest):
    """Add a goal to the persistent goal stack."""
    core = _require_core()
    from core.goals import GoalPriority

    deadline = None
    if request.deadline_hours:
        deadline = time.time() + request.deadline_hours * 3600

    goal = core.goals.add(
        description=request.description,
        priority=GoalPriority[request.priority],
        deadline=deadline,
    )
    return {
        "goal_id": goal.id,
        "description": goal.description,
        "priority": goal.priority.name,
        "deadline": goal.deadline,
    }


@app.patch("/goals/{goal_id}/progress", tags=["Goals"])
async def update_goal_progress(goal_id: str, request: GoalProgressRequest):
    """Update goal progress."""
    core = _require_core()
    core.goals.update_progress(goal_id, request.progress, request.notes)
    return {"goal_id": goal_id, "progress": request.progress}


@app.delete("/goals/{goal_id}", tags=["Goals"])
async def complete_goal(goal_id: str, notes: str = ""):
    """Mark a goal as complete."""
    core = _require_core()
    core.goals.complete(goal_id, notes)
    return {"status": "completed", "goal_id": goal_id}


# ── Memory ─────────────────────────────────────────────────────────────────────

@app.get("/memory/episodic/recent", tags=["Memory"])
async def recent_episodes(hours: float = Query(default=24, gt=0, le=8760)):
    """View recent episodic memory."""
    core = _require_core()
    episodes = core.episodic_memory.recent(hours=hours)
    return {
        "episodes": [ep.to_dict() for ep in episodes],
        "stats": core.episodic_memory.stats(),
    }


@app.get("/memory/episodic/recall", tags=["Memory"])
async def recall_episodes(q: str = Query(..., min_length=1), limit: int = Query(default=5, le=20)):
    """Recall episodic memories relevant to a query."""
    core = _require_core()
    episodes = core.episodic_memory.recall(q, limit=limit)
    return {"query": q, "results": [ep.to_dict() for ep in episodes]}


@app.post("/memory/episodic", tags=["Memory"])
async def store_episode(request: EpisodeRequest):
    """Manually store an episode in episodic memory."""
    core = _require_core()
    ep = core.episodic_memory.store(
        content=request.content,
        summary=request.summary,
        tags=request.tags,
        valence=request.valence,
        salience=request.salience,
    )
    return ep.to_dict()


@app.get("/memory/semantic", tags=["Memory"])
async def query_semantic(
    q: str = Query(..., min_length=1),
    domain: Optional[str] = None,
    limit: int = Query(default=10, le=50),
):
    """Query consolidated semantic memory."""
    core = _require_core()
    facts = core.consolidator.semantic.retrieve(q, domain=domain, limit=limit)
    return {
        "query": q,
        "results": [f.to_dict() for f in facts],
        "stats": core.consolidator.semantic.stats(),
    }


@app.get("/memory/working", tags=["Memory"])
async def working_memory_state():
    """View current working memory."""
    core = _require_core()
    return {
        "token_count": core.working_memory.current_tokens,
        "capacity": core.working_memory.capacity,
        "utilization": round(core.working_memory.current_tokens / core.working_memory.capacity, 3),
        "entries": len(core.working_memory),
        "context_preview": core.working_memory.to_context_string()[:500],
    }


@app.post("/memory/consolidate", tags=["Memory"])
async def trigger_consolidation():
    """Manually trigger a consolidation cycle."""
    core = _require_core()
    result = await core.consolidator.run_cycle(core.salience)
    core.metrics.consolidation_cycles += 1
    return result


# ── Adapter / Learning ─────────────────────────────────────────────────────────

@app.post("/feedback", tags=["Learning"])
async def apply_feedback(request: FeedbackRequest):
    """
    Apply an explicit learning signal to the experience adapter.

    Checked against constitutional invariants before being applied.
    """
    adapter = _require_adapter()
    from adapters.lora import ExperienceDelta

    delta = ExperienceDelta(
        content=request.content,
        feedback=request.feedback,
        domain=request.domain,
        confidence=request.confidence,
    )
    allowed = adapter.apply_delta(delta, invariant_check=True)
    return {
        "applied": allowed,
        "blocked_by_invariant": not allowed,
        "adapter_update_count": adapter._update_count,
    }


@app.get("/adapter/stats", tags=["Learning"])
async def adapter_stats():
    """View experience adapter state and statistics."""
    return _require_adapter().stats()


@app.get("/adapter/drift", tags=["Learning"])
async def check_drift():
    """Check for identity or capability drift in the experience adapter."""
    adapter = _require_adapter()
    drift = adapter.detect_drift()
    return {
        "drift_detected": drift is not None,
        "drift_report": drift,
        "update_count": adapter._update_count,
        "blocked_count": adapter._blocked_count,
    }


@app.get("/adapter/context", tags=["Learning"])
async def get_adaptation_context(q: str = Query(...), domain: Optional[str] = None):
    """Get the adaptation context that would be injected for a given query."""
    adapter = _require_adapter()
    context = adapter.get_adaptation_context(q, domain=domain)
    return {"query": q, "adaptation_context": context or "(none)"}
