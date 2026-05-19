"""
api/server.py

FastAPI boundary for the local Persistent Neural Process runtime.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_core = None
_adapter = None
_core_task = None
_config: dict = {}
_auth_token: Optional[str] = None


async def startup_runtime():
    global _core, _adapter, _core_task, _config, _auth_token

    _config = load_config()
    _auth_token = resolve_local_token(_config)
    if not _auth_token:
        raise RuntimeError("Mutating endpoints require a local API token.")

    from adapters.lora import ExperienceAdapter
    from core.process import PersistentCore

    _core = PersistentCore(_config)
    _adapter = ExperienceAdapter(
        model_id=resolve_model_id(_config),
        config=_config,
    )

    _core_task = asyncio.create_task(_core.start())
    logger.info("PNP API online. Persistent core running.")


async def shutdown_runtime():
    global _core, _adapter, _core_task

    if _core:
        await _core.stop()

    if _core_task:
        try:
            await asyncio.wait_for(_core_task, timeout=1.0)
        except asyncio.TimeoutError:
            _core_task.cancel()
            try:
                await _core_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass

    _core = None
    _adapter = None
    _core_task = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup_runtime()
    try:
        yield
    finally:
        await shutdown_runtime()


app = FastAPI(
    title="PNP - Persistent Neural Process",
    description="Local API for a continuously running PNP prototype",
    version="0.2.0",
    lifespan=lifespan,
)


def load_config() -> dict:
    import yaml

    config_path = os.getenv("PNP_CONFIG_PATH", "config/default.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("api_host", "127.0.0.1")
    config.setdefault("local_api_token_env", "PNP_LOCAL_TOKEN")
    config.setdefault("inference_provider", "mock")
    return config


def resolve_local_token(config: dict) -> Optional[str]:
    env_name = config.get("local_api_token_env", "PNP_LOCAL_TOKEN")
    return os.getenv(env_name) or config.get("local_api_token")


def resolve_model_id(config: dict) -> str:
    return os.getenv("PNP_MODEL_ID") or config.get("model_id") or "mock-model"


async def require_local_token(x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token")):
    if not _auth_token:
        raise HTTPException(503, "Local API token is not configured")
    if not x_pnp_token or not secrets.compare_digest(str(x_pnp_token), str(_auth_token)):
        raise HTTPException(401, "Missing or invalid local API token")


class ChatRequest(BaseModel):
    message: str
    concepts: list[str] = []
    stream: bool = False


class GoalRequest(BaseModel):
    description: str
    priority: str = "MEDIUM"
    deadline_hours: Optional[float] = None


class FeedbackRequest(BaseModel):
    content: str
    feedback: float
    domain: str = "general"
    confidence: float = 0.5


class AdapterTrainRequest(BaseModel):
    epochs: Optional[int] = None


@app.get("/")
async def root():
    if not _core:
        return {"status": "initializing"}
    state = _core.get_state_snapshot()
    return {
        "status": "alive",
        "uptime_hours": round(state["uptime_seconds"] / 3600, 2),
        "mode": state["motivational_state"]["mode"],
        "heartbeats": state["heartbeat_count"],
        "continuity": state["continuity"],
    }


@app.get("/state")
async def get_state():
    if not _core:
        raise HTTPException(503, "Core not initialized")
    return _core.get_state_snapshot()


@app.get("/goals")
async def list_goals():
    if not _core:
        raise HTTPException(503, "Core not initialized")
    return {"active_goals": _core.goals.to_list()}


@app.post("/chat")
async def chat(request: ChatRequest, _auth: None = Depends(require_local_token)):
    if not _core:
        raise HTTPException(503, "Core not initialized")

    core_state = _core.on_interaction(
        content=request.message,
        metadata={"role": "user", "concepts": request.concepts},
    )

    episodic_context = build_episodic_context(request.message)
    semantic_context = build_semantic_context(request.message)
    adaptation_context = _adapter.get_adaptation_context(request.message) if _adapter else ""

    from inference.engine import InferenceRequest

    inference_req = InferenceRequest(
        user_input=request.message,
        working_memory_context=_core.working_memory.to_context_string(),
        episodic_context=episodic_context,
        semantic_context=semantic_context,
        adaptation_context=adaptation_context,
        core_state=core_state,
        stream=request.stream,
        model=resolve_model_id(_config),
    )

    result = await run_configured_inference(inference_req)

    episode = _core.episodic_memory.store(
        content=request.message,
        summary=result.memory_deltas[0]["summary"] if result.memory_deltas else request.message[:100],
        tags=["interaction"] + request.concepts,
        valence=result.valence,
        salience=0.8,
    )
    _core.record_memory_written("episodic", episode.summary, episode.to_dict(), salience=episode.salience)

    for delta in result.memory_deltas:
        if delta.get("type") == "semantic":
            fact = _core.consolidator.semantic.store_fact(
                content=delta.get("content", request.message),
                source_episode_ids=[episode.id],
                domain=delta.get("domain", "general"),
                confidence=delta.get("confidence", 0.5),
            )
            _core.record_memory_written("semantic", fact.content, fact.to_dict(), salience=fact.confidence)

    _core.working_memory.add(result.content, {"role": "assistant"})
    _core.record_memory_written("working", result.content, {"role": "assistant"}, salience=0.8)

    return {
        "response": result.content,
        "tokens_used": result.tokens_used,
        "latency_ms": round(result.latency_ms),
        "context_used": {
            "episodic": bool(episodic_context),
            "semantic": bool(semantic_context),
            "adaptation": bool(adaptation_context),
        },
        "core_state": {
            "mode": core_state["motivational_state"]["mode"],
            "uptime_hours": round(core_state["uptime_seconds"] / 3600, 2),
        },
    }


def build_episodic_context(query: str) -> str:
    try:
        episodes = _core.episodic_memory.recall(query, limit=3)
    except Exception as exc:
        logger.warning("Episodic recall failed: %s", exc)
        return ""

    if not episodes:
        return ""

    parts = ["=== RELEVANT MEMORIES ==="]
    for episode in episodes:
        parts.append(f"[{episode.tags}] {episode.summary}")
    parts.append("=== END MEMORIES ===")
    return "\n".join(parts)


def build_semantic_context(query: str) -> str:
    try:
        facts = _core.consolidator.semantic.retrieve(query, limit=5)
    except Exception as exc:
        logger.warning("Semantic recall failed: %s", exc)
        return ""

    if not facts:
        return ""

    parts = ["=== RELEVANT FACTS ==="]
    for fact in facts:
        parts.append(f"[{fact.domain}:{fact.confidence:.2f}] {fact.content}")
    parts.append("=== END FACTS ===")
    return "\n".join(parts)


async def run_configured_inference(inference_req):
    from inference.providers import run_provider_inference

    return await run_provider_inference(inference_req, _config)


@app.post("/goals")
async def add_goal(request: GoalRequest, _auth: None = Depends(require_local_token)):
    if not _core:
        raise HTTPException(503, "Core not initialized")

    from core.goals import GoalPriority

    try:
        priority = GoalPriority[request.priority]
    except KeyError as exc:
        raise HTTPException(400, f"Unknown goal priority: {request.priority}") from exc

    deadline = time.time() + (request.deadline_hours * 3600) if request.deadline_hours else None
    goal = _core.add_goal(
        description=request.description,
        priority=priority,
        deadline=deadline,
    )
    return {"goal_id": goal.id, "description": goal.description, "priority": goal.priority.name}


@app.delete("/goals/{goal_id}")
async def complete_goal(goal_id: str, notes: str = "", _auth: None = Depends(require_local_token)):
    if not _core:
        raise HTTPException(503, "Core not initialized")
    if not _core.complete_goal(goal_id, notes):
        raise HTTPException(404, "Goal not found")
    return {"status": "completed", "goal_id": goal_id}


@app.get("/memory/recent")
async def recent_memory(hours: float = 24):
    if not _core:
        raise HTTPException(503, "Core not initialized")
    episodes = _core.episodic_memory.recent(hours=hours)
    return {
        "episodes": [ep.to_dict() for ep in episodes],
        "stats": _core.episodic_memory.stats(),
    }


@app.get("/memory/semantic")
async def semantic_memory_stats():
    if not _core:
        raise HTTPException(503, "Core not initialized")
    return _core.consolidator.semantic.stats()


@app.post("/feedback")
async def apply_feedback(request: FeedbackRequest, _auth: None = Depends(require_local_token)):
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
    if _core:
        _core.record_adapter_delta(delta.to_dict(), allowed)
    return {
        "applied": allowed,
        "blocked_by_invariant": not allowed,
        "adapter_stats": _adapter.stats(),
    }


@app.post("/adapter/train")
async def train_adapter(request: AdapterTrainRequest, _auth: None = Depends(require_local_token)):
    if not _adapter:
        raise HTTPException(503, "Adapter not initialized")

    metrics = _adapter.train_adapter(epochs=request.epochs)
    if _core:
        _core.record_event("adapter_trained", metrics)
    return {
        "trained": True,
        "metrics": metrics,
        "adapter_stats": _adapter.stats(),
    }


@app.get("/adapter/stats")
async def adapter_stats():
    if not _adapter:
        raise HTTPException(503, "Adapter not initialized")
    return _adapter.stats()


@app.get("/adapter/drift")
async def check_drift():
    if not _adapter:
        raise HTTPException(503, "Adapter not initialized")
    drift = _adapter.detect_drift()
    return {
        "drift_detected": drift is not None,
        "drift_report": drift,
    }
