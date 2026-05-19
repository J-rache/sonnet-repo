"""
api/server.py

FastAPI interface for the Persistent Neural Process runtime.

The persistent core owns continuity. This API is the local control plane for
chat, state inspection, goals, memory, and adapter feedback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_config: dict = {}
_core = None
_adapter = None


def _get_local_token() -> str:
    """Return the configured local API token from env or config."""
    token_env = _config.get("local_api_token_env", "PNP_LOCAL_TOKEN") if _config else "PNP_LOCAL_TOKEN"
    token = os.environ.get(token_env, "")
    if not token and _config:
        token = _config.get("local_api_token", "")
    return token


def _check_auth(x_pnp_token: Optional[str]) -> bool:
    required = _get_local_token()
    if not required:
        return True
    return x_pnp_token == required


def _require_auth(x_pnp_token: Optional[str]) -> None:
    if not _check_auth(x_pnp_token):
        raise HTTPException(401, detail="Invalid or missing X-PNP-Token")


def _configured_provider() -> str:
    return (os.environ.get("PNP_INFERENCE_PROVIDER") or _config.get("inference_provider", "mock")).lower()


def _configured_model(request_model: Optional[str] = None) -> str:
    return (
        request_model
        or os.environ.get("PNP_MODEL_ID")
        or os.environ.get("PNP_MODEL")
        or _config.get("model_id")
        or _config.get("base_model")
        or "local-model"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _core, _adapter, _config

    import yaml
    from adapters.lora import ExperienceAdapter
    from core.process import PersistentCore

    config_path = os.environ.get("PNP_CONFIG_PATH") or os.environ.get("PNP_CONFIG", "config/default.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _config = config
    _core = PersistentCore(config)
    _adapter = ExperienceAdapter(
        base_model_id=_configured_model(),
        config=config,
    )

    provider = _configured_provider()
    logger.info("Inference provider configured: %s model=%s", provider, _configured_model())

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
    title="PNP - Persistent Neural Process",
    description="Local interface to a persistent AI process with memory, goals, and adapter feedback",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    concepts: list[str] = Field(default_factory=list)
    model: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=4096)


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


class AdapterTrainRequest(BaseModel):
    epochs: Optional[int] = Field(default=None, ge=1, le=500)


class EpisodeRequest(BaseModel):
    content: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    salience: float = Field(default=0.8, ge=0.0, le=1.0)


def _require_core():
    if _core is None:
        raise HTTPException(503, detail="Persistent core not initialized")
    return _core


def _require_adapter():
    if _adapter is None:
        raise HTTPException(503, detail="Experience adapter not initialized")
    return _adapter


def _provider_has_credentials(provider: str) -> bool:
    providers = _config.get("providers", {})
    provider_config = providers.get(provider, {})
    if provider == "mock":
        return True
    if provider == "anthropic":
        key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
        return bool(os.environ.get(key_env) or provider_config.get("api_key"))
    if provider in {"openai", "openai_compatible", "vllm", "lmstudio"}:
        key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
        return bool(os.environ.get(key_env) or provider_config.get("api_key"))
    if provider == "ollama":
        return True
    return False


@app.get("/", tags=["Status"])
async def root():
    core = _require_core()
    state = core.get_state_snapshot()
    provider = _configured_provider()
    return {
        "status": "alive",
        "uptime_hours": round(state["uptime_seconds"] / 3600, 3),
        "mode": state["motivational_state"]["mode"],
        "heartbeats": state["heartbeat_count"],
        "total_interactions": state["total_interactions"],
        "active_goals": len(state["active_goals"]),
        "inference_provider": provider,
        "model": _configured_model(),
        "provider_configured": _provider_has_credentials(provider),
    }


@app.get("/state", tags=["Status"])
async def get_state():
    return _require_core().get_state_snapshot()


@app.get("/health", tags=["Status"])
async def health():
    if _core is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "uptime_seconds": round(_core.metrics.uptime_seconds, 1)}


@app.post("/chat", tags=["Interaction"])
async def chat(request: ChatRequest, x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token")):
    _require_auth(x_pnp_token)
    core = _require_core()
    adapter = _require_adapter()

    core_state = core.on_interaction(
        content=request.message,
        metadata={"role": "user", "concepts": request.concepts},
    )

    episodes = []
    episodic_context = ""
    try:
        episodes = core.episodic_memory.recall(request.message, limit=4, min_salience=0.05)
        if episodes:
            lines = ["=== RELEVANT MEMORIES ==="]
            lines.extend(f"[{', '.join(ep.tags)}] {ep.summary}" for ep in episodes)
            lines.append("=== END MEMORIES ===")
            episodic_context = "\n".join(lines)
    except Exception as e:
        logger.warning("Episodic recall error: %s", e)

    facts = []
    semantic_context = ""
    try:
        facts = core.consolidator.semantic.retrieve(request.message, limit=3)
        if facts:
            lines = ["=== KNOWN FACTS ==="]
            lines.extend(f"[conf={fact.confidence:.2f}] {fact.content}" for fact in facts)
            lines.append("=== END FACTS ===")
            semantic_context = "\n".join(lines)
    except Exception as e:
        logger.warning("Semantic recall error: %s", e)

    adaptation_context = adapter.get_adaptation_context(request.message)

    from inference.engine import InferenceRequest
    from inference.providers import run_provider_inference

    inf_req = InferenceRequest(
        user_input=request.message,
        working_memory_context=core.working_memory.to_context_string(),
        episodic_context=episodic_context,
        semantic_context=semantic_context,
        adaptation_context=adaptation_context,
        core_state=core_state,
        model=_configured_model(request.model),
        max_tokens=request.max_tokens,
    )

    try:
        result = await run_provider_inference(inf_req, _config)
    except Exception as e:
        provider = _configured_provider()
        logger.exception("Inference provider failed: %s", provider)
        raise HTTPException(502, detail=f"Inference provider '{provider}' failed: {e}") from e

    delta = result.memory_deltas[0] if result.memory_deltas else {}
    try:
        core.episodic_memory.store(
            content=request.message,
            summary=delta.get("summary", request.message[:100]),
            tags=delta.get("tags", ["interaction"]),
            valence=result.valence,
            salience=0.85,
        )
    except Exception as e:
        logger.error("Episode store failed: %s", e)

    for fact_text in delta.get("durable_facts", []):
        try:
            core.consolidator.semantic.store_fact(
                content=fact_text,
                source_episode_ids=[],
                domain="general",
                confidence=0.5,
            )
        except Exception as e:
            logger.warning("Fact store failed: %s", e)

    core.working_memory.add(result.content, {"role": "assistant"})
    core.on_inference_result(suggested_goals=result.suggested_goals, valence=result.valence)

    from adapters.lora import ExperienceDelta

    if request.message and result.content:
        delta_obj = ExperienceDelta(
            content=f"Q: {request.message[:100]} A: {result.content[:100]}",
            feedback=0.3,
            domain="interaction",
            confidence=0.4,
        )
        if adapter.apply_delta(delta_obj, invariant_check=True):
            try:
                core.journal.append(
                    "adapter_delta_applied",
                    {"domain": "interaction", "feedback": 0.3, "source": "chat"},
                )
            except Exception:
                pass

    return {
        "response": result.content,
        "tokens_used": result.tokens_used,
        "latency_ms": round(result.latency_ms),
        "provider": result.metadata.get("provider", _configured_provider()),
        "model": result.metadata.get("model", inf_req.model),
        "context_used": {
            "episodic": len(episodes) > 0,
            "semantic": len(facts) > 0,
            "adaptation": bool(adaptation_context),
            "working_memory": core.working_memory.current_tokens > 0,
        },
        "core_state": {
            "mode": core_state["motivational_state"]["mode"],
            "uptime_hours": round(core_state["uptime_seconds"] / 3600, 3),
            "total_interactions": core_state["total_interactions"],
            "active_goals": len(core_state["active_goals"]),
        },
        "memory": {
            "episodic_retrieved": len(episodes),
            "semantic_retrieved": len(facts),
            "adaptations_applied": bool(adaptation_context),
            "new_goals_from_response": len(result.suggested_goals),
        },
    }


@app.get("/goals", tags=["Goals"])
async def list_goals():
    core = _require_core()
    return {
        "active_goals": core.goals.to_list(),
        "active_count": core.goals.active_count,
    }


@app.post("/goals", tags=["Goals"])
async def add_goal(request: GoalRequest, x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token")):
    _require_auth(x_pnp_token)
    core = _require_core()
    from core.goals import GoalPriority

    deadline = time.time() + request.deadline_hours * 3600 if request.deadline_hours else None
    goal = core.add_goal(
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
async def update_goal_progress(
    goal_id: str,
    request: GoalProgressRequest,
    x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token"),
):
    _require_auth(x_pnp_token)
    core = _require_core()
    core.goals.update_progress(goal_id, request.progress, request.notes)
    try:
        core.journal.append(
            "goal_progress_updated",
            {"goal_id": goal_id, "progress": request.progress, "notes": request.notes},
        )
    except Exception:
        pass
    return {"goal_id": goal_id, "progress": request.progress}


@app.delete("/goals/{goal_id}", tags=["Goals"])
async def complete_goal(
    goal_id: str,
    notes: str = "",
    x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token"),
):
    _require_auth(x_pnp_token)
    core = _require_core()
    core.goals.complete(goal_id, notes)
    try:
        core.journal.append("goal_completed", {"goal_id": goal_id, "notes": notes})
    except Exception:
        pass
    return {"status": "completed", "goal_id": goal_id}


@app.get("/memory/recent", tags=["Memory"])
@app.get("/memory/episodic/recent", tags=["Memory"])
async def recent_episodes(hours: float = Query(default=24, gt=0, le=8760)):
    core = _require_core()
    episodes = core.episodic_memory.recent(hours=hours)
    return {
        "episodes": [ep.to_dict() for ep in episodes],
        "stats": core.episodic_memory.stats(),
    }


@app.get("/memory/episodic/recall", tags=["Memory"])
async def recall_episodes(q: str = Query(..., min_length=1), limit: int = Query(default=5, le=20)):
    core = _require_core()
    episodes = core.episodic_memory.recall(q, limit=limit)
    return {"query": q, "results": [ep.to_dict() for ep in episodes]}


@app.post("/memory/episodic", tags=["Memory"])
async def store_episode(
    request: EpisodeRequest,
    x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token"),
):
    _require_auth(x_pnp_token)
    core = _require_core()
    ep = core.episodic_memory.store(
        content=request.content,
        summary=request.summary,
        tags=request.tags,
        valence=request.valence,
        salience=request.salience,
    )
    try:
        core.journal.append("memory_written", {"type": "episodic", "episode_id": ep.id})
    except Exception:
        pass
    return ep.to_dict()


@app.get("/memory/semantic", tags=["Memory"])
async def query_semantic(
    q: str = Query(..., min_length=1),
    domain: Optional[str] = None,
    limit: int = Query(default=10, le=50),
):
    core = _require_core()
    facts = core.consolidator.semantic.retrieve(q, domain=domain, limit=limit)
    return {
        "query": q,
        "results": [f.to_dict() for f in facts],
        "stats": core.consolidator.semantic.stats(),
    }


@app.get("/memory/working", tags=["Memory"])
async def working_memory_state():
    core = _require_core()
    return {
        "token_count": core.working_memory.current_tokens,
        "capacity": core.working_memory.capacity,
        "utilization": round(core.working_memory.current_tokens / core.working_memory.capacity, 3),
        "entries": len(core.working_memory),
        "context_preview": core.working_memory.to_context_string()[:500],
    }


@app.post("/memory/consolidate", tags=["Memory"])
async def trigger_consolidation(x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token")):
    _require_auth(x_pnp_token)
    core = _require_core()
    return await core.run_consolidation_cycle()


@app.post("/feedback", tags=["Learning"])
async def apply_feedback(
    request: FeedbackRequest,
    x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token"),
):
    _require_auth(x_pnp_token)
    adapter = _require_adapter()
    from adapters.lora import ExperienceDelta

    delta = ExperienceDelta(
        content=request.content,
        feedback=request.feedback,
        domain=request.domain,
        confidence=request.confidence,
    )
    allowed = adapter.apply_delta(delta, invariant_check=True)
    try:
        _require_core().journal.append(
            "adapter_delta_applied" if allowed else "adapter_delta_blocked",
            {"domain": request.domain, "feedback": request.feedback, "source": "feedback"},
        )
    except Exception:
        pass
    return {
        "applied": allowed,
        "blocked_by_invariant": not allowed,
        "adapter_update_count": adapter._update_count,
    }


@app.post("/adapter/train", tags=["Learning"])
async def train_adapter(
    request: AdapterTrainRequest,
    x_pnp_token: Optional[str] = Header(default=None, alias="X-PNP-Token"),
):
    _require_auth(x_pnp_token)
    adapter = _require_adapter()
    metrics = adapter.train_adapter(epochs=request.epochs)
    try:
        _require_core().journal.append("adapter_trained", metrics)
    except Exception:
        pass
    return {
        "trained": metrics.get("sample_count", 0) > 0,
        "metrics": metrics,
        "adapter_stats": adapter.stats(),
    }


@app.get("/adapter/stats", tags=["Learning"])
async def adapter_stats():
    return _require_adapter().stats()


@app.get("/adapter/sync", tags=["Learning"])
async def adapter_sync_state():
    adapter = _require_adapter()
    return {
        "sync_model_adapter": adapter.sync_model_state(),
        "sync_model_context": adapter.get_sync_model_context(),
    }


@app.get("/adapter/drift", tags=["Learning"])
async def check_drift():
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
    adapter = _require_adapter()
    context = adapter.get_adaptation_context(q, domain=domain)
    return {"query": q, "adaptation_context": context or "(none)"}
