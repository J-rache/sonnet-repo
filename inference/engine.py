"""
inference/engine.py

The inference engine is called on demand for heavy computation.

It receives context assembled from memory and core state, runs a model call, and
returns both output and memory deltas: what should be stored or learned from the
interaction.
"""

from dataclasses import dataclass
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Everything the inference engine needs to respond."""

    user_input: str
    working_memory_context: str
    episodic_context: str
    semantic_context: str
    adaptation_context: str
    core_state: dict
    system_prompt_override: Optional[str] = None
    stream: bool = True
    max_tokens: int = 2048
    model: str = "mock-model"


@dataclass
class InferenceResult:
    """Output from the inference engine."""

    content: str
    tokens_used: int
    latency_ms: float
    memory_deltas: list[dict]
    suggested_goals: list[dict]
    valence: float
    metadata: dict


def build_system_prompt(core_state: dict) -> str:
    """
    Build the system prompt incorporating local runtime context.
    """
    mode = core_state.get("motivational_state", {}).get("mode", "nominal")
    uptime_hours = round(core_state.get("uptime_seconds", 0) / 3600, 1)
    active_goals = core_state.get("active_goals", [])
    consolidation_cycles = core_state.get("consolidation_cycles", 0)

    goal_text = ""
    if active_goals:
        goal_summaries = [f"- {g['description']} (priority: {g['priority']})" for g in active_goals[:3]]
        goal_text = "\n\nActive goals:\n" + "\n".join(goal_summaries)

    return f"""You are operating inside a local Persistent Neural Process prototype.

The local persistent core for this API process has been running for {uptime_hours}
hours and has recorded {consolidation_cycles} memory consolidation cycles. The
current operational mode is: {mode}.

You may receive working memory, episodic memory, semantic memory, adapter
context, and active goals. Use that context when relevant. Do not imply more
continuity, autonomy, training, or model access than the provided runtime state
proves.

Be concise, honest, and direct. Acknowledge uncertainty and implementation
limits when they matter.{goal_text}"""


def build_full_context(request: InferenceRequest) -> list[dict]:
    """Assemble the full message context for inference."""
    context_parts = []

    if request.working_memory_context:
        context_parts.append(request.working_memory_context)

    if request.episodic_context:
        context_parts.append(request.episodic_context)

    if request.semantic_context:
        context_parts.append(request.semantic_context)

    if request.adaptation_context:
        context_parts.append(request.adaptation_context)

    full_context = "\n\n".join(filter(None, context_parts))

    if full_context:
        return [{
            "role": "user",
            "content": f"[PERSISTENT CONTEXT]\n{full_context}\n[END CONTEXT]\n\n{request.user_input}",
        }]

    return [{"role": "user", "content": request.user_input}]


async def run_inference(request: InferenceRequest, api_client) -> InferenceResult:
    """
    Run inference with full persistent context.

    api_client is the LLM API client, such as Anthropic, OpenAI, or a compatible
    local model wrapper.
    """
    t_start = time.monotonic()

    system_prompt = request.system_prompt_override or build_system_prompt(request.core_state)
    messages = build_full_context(request)

    logger.debug(
        "Running inference. Context tokens estimate: %s",
        sum(len(m["content"]) // 4 for m in messages),
    )

    response = await api_client.messages.create(
        model=request.model,
        max_tokens=request.max_tokens,
        system=system_prompt,
        messages=messages,
    )

    content = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    latency_ms = (time.monotonic() - t_start) * 1000
    memory_deltas = extract_memory_deltas(request.user_input, content)
    valence = estimate_valence(content)

    return InferenceResult(
        content=content,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        memory_deltas=memory_deltas,
        suggested_goals=[],
        valence=valence,
        metadata={
            "model": request.model,
            "system_prompt_tokens": len(system_prompt) // 4,
        },
    )


def extract_memory_deltas(user_input: str, response_content: str) -> list[dict]:
    """
    Extract what should be remembered from this interaction.

    This prototype uses simple heuristics; it does not call another model for
    memory extraction.
    """
    deltas = []

    summary = f"User asked about: {user_input[:100]}. Responded with: {response_content[:200]}"
    deltas.append({
        "type": "episodic",
        "content": user_input,
        "summary": summary,
        "tags": ["interaction"],
    })

    if any(phrase in user_input.lower() for phrase in ["remember", "note that", "i prefer", "i like", "i hate"]):
        deltas.append({
            "type": "semantic",
            "content": user_input,
            "domain": "user_preferences",
            "confidence": 0.7,
        })

    return deltas


def estimate_valence(content: str) -> float:
    """
    Estimate emotional valence with a small keyword heuristic.
    """
    positive_words = {"great", "excellent", "wonderful", "thanks", "helpful", "good", "love"}
    negative_words = {"bad", "wrong", "mistake", "sorry", "fail", "error", "hate"}

    words = set(content.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)

    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)
