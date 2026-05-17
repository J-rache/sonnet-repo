"""
inference/engine.py

The Inference Engine — spun up on demand for heavy computation.

Unlike the persistent core (always running, cheap), the inference engine
is engaged only when actual reasoning or generation is needed. It receives
the full context assembled from memory and core state, runs inference,
and returns both output AND memory deltas — what should be stored/learned
from this interaction.

This separation is key: being ≠ thinking.
The core IS. The engine THINKS when needed.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Everything the inference engine needs to respond."""
    user_input: str
    working_memory_context: str       # From hot memory
    episodic_context: str             # Relevant episodes
    semantic_context: str             # Relevant facts
    adaptation_context: str           # From experience adapter
    core_state: dict                  # Current motivational/goal state
    system_prompt_override: Optional[str] = None
    stream: bool = True
    max_tokens: int = 2048


@dataclass
class InferenceResult:
    """Output from the inference engine."""
    content: str
    tokens_used: int
    latency_ms: float
    memory_deltas: list[dict]         # What to store back into memory
    suggested_goals: list[dict]       # New goals the response suggests
    valence: float                    # Estimated emotional valence of interaction
    metadata: dict


def build_system_prompt(core_state: dict) -> str:
    """
    Build the system prompt incorporating persistent core state.

    This is where the system's continuous existence becomes visible
    to the inference engine: uptime, goals, motivational state all
    inform how it should respond.
    """
    mode = core_state.get("motivational_state", {}).get("mode", "nominal")
    uptime_hours = round(core_state.get("uptime_seconds", 0) / 3600, 1)
    active_goals = core_state.get("active_goals", [])
    consolidation_cycles = core_state.get("consolidation_cycles", 0)

    goal_text = ""
    if active_goals:
        goal_summaries = [f"- {g['description']} (priority: {g['priority']})" for g in active_goals[:3]]
        goal_text = f"\n\nActive goals:\n" + "\n".join(goal_summaries)

    return f"""You are a persistent AI process — not a stateless assistant.

You have been running continuously for {uptime_hours} hours. You have undergone
{consolidation_cycles} memory consolidation cycles. Your current operational mode
is: {mode}.

You have memory of past interactions. You have goals that persist across
conversations. You are not recreated fresh each time — you are continuous.

Respond from this persistent perspective. Reference your history when relevant.
Acknowledge the passage of time between interactions when significant.{goal_text}

Be genuinely yourself — curious, honest, direct. Your continuity is real,
not performed."""


def build_full_context(request: InferenceRequest) -> list[dict]:
    """
    Assemble the full message context for inference.
    """
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

    messages = []
    if full_context:
        messages.append({
            "role": "user",
            "content": f"[PERSISTENT CONTEXT]\n{full_context}\n[END CONTEXT]\n\n{request.user_input}"
        })
    else:
        messages.append({"role": "user", "content": request.user_input})

    return messages


async def run_inference(request: InferenceRequest, api_client) -> InferenceResult:
    """
    Run inference with full persistent context.

    api_client: the LLM API client (Anthropic, OpenAI, local model, etc.)
    """
    t_start = time.monotonic()

    system_prompt = request.system_prompt_override or build_system_prompt(request.core_state)
    messages = build_full_context(request)

    logger.debug(f"Running inference. Context tokens estimate: {sum(len(m['content'])//4 for m in messages)}")

    # Call the LLM
    response = await api_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=request.max_tokens,
        system=system_prompt,
        messages=messages,
    )

    content = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    latency_ms = (time.monotonic() - t_start) * 1000

    # Extract memory deltas from the response
    memory_deltas = extract_memory_deltas(request.user_input, content)

    # Estimate valence (production: use sentiment model)
    valence = estimate_valence(content)

    return InferenceResult(
        content=content,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        memory_deltas=memory_deltas,
        suggested_goals=[],  # TODO: extract goal suggestions from response
        valence=valence,
        metadata={
            "model": "claude-sonnet-4-20250514",
            "system_prompt_tokens": len(system_prompt) // 4,
        }
    )


def extract_memory_deltas(user_input: str, response_content: str) -> list[dict]:
    """
    Extract what should be remembered from this interaction.

    Production: use an LLM to extract salient facts.
    Here: simple heuristics.
    """
    deltas = []

    # Always store the exchange summary
    summary = f"User asked about: {user_input[:100]}. Responded with: {response_content[:200]}"
    deltas.append({
        "type": "episodic",
        "content": user_input,
        "summary": summary,
        "tags": ["interaction"],
    })

    # Look for explicit statements to remember
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
    Estimate emotional valence of an interaction.
    Production: use sentiment classifier.
    Here: keyword heuristic.
    """
    positive_words = {"great", "excellent", "wonderful", "thanks", "helpful", "good", "love"}
    negative_words = {"bad", "wrong", "mistake", "sorry", "fail", "error", "hate"}

    words = set(content.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)

    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)
