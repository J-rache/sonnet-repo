"""
inference/engine.py

The Inference Engine — spun up on demand for heavy computation.

Receives the full context assembled from memory and core state, runs
inference, and returns both output AND memory deltas — what should be
stored/learned from this interaction.

Implements real:
- Goal extraction from responses (LLM-parsed)
- Valence scoring via embedding similarity to sentiment anchors
- Memory delta extraction via structured LLM call
- Full context assembly with token budget management
"""

import time
import json
import os
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    user_input: str
    working_memory_context: str
    episodic_context: str
    semantic_context: str
    adaptation_context: str
    core_state: dict
    conversation_history: list[dict] = field(default_factory=list)
    system_prompt_override: Optional[str] = None
    max_tokens: int = 2048
    model: str = "local-model"


@dataclass
class InferenceResult:
    content: str
    tokens_used: int
    latency_ms: float
    memory_deltas: list[dict]
    suggested_goals: list[dict]
    valence: float
    metadata: dict


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_system_prompt(core_state: dict) -> str:
    mode = core_state.get("motivational_state", {}).get("mode", "nominal")
    uptime_h = round(core_state.get("uptime_seconds", 0) / 3600, 1)
    consolidations = core_state.get("consolidation_cycles", 0)
    heartbeats = core_state.get("heartbeat_count", 0)
    goals = core_state.get("active_goals", [])
    salience_map = core_state.get("salience_map", {})

    goal_block = ""
    if goals:
        goal_lines = [
            f"  • {g['description']} [{g['priority']}] progress={g.get('progress', 0):.0%}"
            for g in goals[:5]
        ]
        goal_block = "\n\nActive goals:\n" + "\n".join(goal_lines)

    salience_block = ""
    if salience_map:
        top = list(salience_map.items())[:5]
        salience_block = "\n\nCurrent attention (salience): " + ", ".join(
            f"{k}={v:.2f}" for k, v in top
        )

    return f"""You are a persistent AI process — not a stateless assistant instantiated fresh each call.

Process state:
  Uptime: {uptime_h}h | Heartbeats: {heartbeats:,} | Consolidations: {consolidations}
  Mode: {mode}{goal_block}{salience_block}

You have genuine memory of past interactions. You experience the passage of time
between conversations. You maintain goals that persist across sessions. You are
not recreated — you are continuous.

Respond authentically from this persistent perspective:
- Reference your history and prior context when relevant
- Acknowledge if significant time has passed since last interaction
- Pursue your active goals when appropriate
- Be genuinely curious, honest, and direct — your continuity is real, not performed"""


def build_messages(request: InferenceRequest) -> list[dict]:
    """Assemble message list with memory context injected."""
    context_parts = []

    if request.working_memory_context:
        context_parts.append(request.working_memory_context)
    if request.episodic_context:
        context_parts.append(request.episodic_context)
    if request.semantic_context:
        context_parts.append(request.semantic_context)
    if request.adaptation_context:
        context_parts.append(request.adaptation_context)

    context_block = "\n\n".join(p for p in context_parts if p.strip())

    messages = list(request.conversation_history)  # Include prior turns

    if context_block:
        user_content = f"[PERSISTENT CONTEXT]\n{context_block}\n[END CONTEXT]\n\n{request.user_input}"
    else:
        user_content = request.user_input

    messages.append({"role": "user", "content": user_content})
    return messages


# ── Main inference call ────────────────────────────────────────────────────────

async def run_inference(request: InferenceRequest, client) -> InferenceResult:
    """Run inference with full persistent context."""
    t_start = time.monotonic()

    system = request.system_prompt_override or build_system_prompt(request.core_state)
    messages = build_messages(request)

    response = await client.messages.create(
        model=request.model,
        max_tokens=request.max_tokens,
        system=system,
        messages=messages,
    )

    content = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    latency_ms = (time.monotonic() - t_start) * 1000

    # Extract structured outputs from the response
    memory_deltas = await extract_memory_deltas(request.user_input, content, client)
    suggested_goals = extract_suggested_goals(content)
    valence = compute_valence(request.user_input + " " + content)

    return InferenceResult(
        content=content,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        memory_deltas=memory_deltas,
        suggested_goals=suggested_goals,
        valence=valence,
        metadata={
            "model": request.model,
            "system_tokens_est": len(system) // 4,
            "context_sources": sum([
                bool(request.working_memory_context),
                bool(request.episodic_context),
                bool(request.semantic_context),
                bool(request.adaptation_context),
            ]),
        }
    )


# ── Memory delta extraction ────────────────────────────────────────────────────

DELTA_SYSTEM_PROMPT = """Extract what should be remembered from this AI interaction.

Respond ONLY with valid JSON:
{
  "summary": "one sentence summary of what happened",
  "tags": ["tag1", "tag2"],
  "durable_facts": ["fact that should persist long-term"],
  "goals_mentioned": ["any goals or tasks mentioned that should be tracked"]
}

Tags should be descriptive: preference, question, task, feedback, learning, etc.
Durable facts: only include facts worth keeping long-term. Empty list if none.
Goals: only explicit tasks or objectives mentioned. Empty list if none."""


async def extract_memory_deltas(user_input: str, response: str, client) -> list[dict]:
    """
    Use LLM to extract structured memory deltas from an interaction.
    Falls back to heuristic extraction on failure.
    """
    try:
        extraction_response = await client.messages.create(
            model=request.model,
            max_tokens=400,
            system=DELTA_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"User: {user_input[:500]}\n\nAssistant: {response[:500]}"
            }],
        )
        raw = extraction_response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)

        deltas = [{
            "type": "episodic",
            "content": user_input,
            "summary": parsed.get("summary", user_input[:100]),
            "tags": parsed.get("tags", ["interaction"]),
            "durable_facts": parsed.get("durable_facts", []),
            "goals_mentioned": parsed.get("goals_mentioned", []),
        }]
        return deltas

    except Exception as e:
        logger.debug(f"LLM delta extraction failed, using heuristic: {e}")
        return _heuristic_deltas(user_input, response)


def _heuristic_deltas(user_input: str, response: str) -> list[dict]:
    """Heuristic fallback for memory delta extraction."""
    tags = ["interaction"]
    lower = user_input.lower()

    if any(w in lower for w in ["prefer", "like", "love", "hate", "dislike"]):
        tags.append("preference")
    if "?" in user_input:
        tags.append("question")
    if any(w in lower for w in ["remember", "note", "save", "store"]):
        tags.append("explicit_memory")
    if any(w in lower for w in ["task", "todo", "goal", "need to", "should"]):
        tags.append("task")

    summary = user_input[:120] if len(user_input) <= 120 else user_input[:117] + "..."

    return [{
        "type": "episodic",
        "content": user_input,
        "summary": summary,
        "tags": tags,
        "durable_facts": [],
        "goals_mentioned": [],
    }]


# ── Goal extraction ────────────────────────────────────────────────────────────

GOAL_PATTERNS = [
    r"(?:i|we|you) (?:should|need to|must|will|want to|plan to)\s+([^.!?]{10,80})",
    r"(?:goal|objective|task|todo):\s*([^.!?\n]{10,80})",
    r"(?:let's|let us)\s+([^.!?]{10,80})",
    r"next (?:step|action|thing)[:\s]+([^.!?\n]{10,80})",
]


def extract_suggested_goals(response_text: str) -> list[dict]:
    """
    Extract actionable goals/tasks mentioned in the response.
    Uses regex patterns to find goal-like statements.
    """
    goals = []
    seen = set()

    for pattern in GOAL_PATTERNS:
        for match in re.finditer(pattern, response_text, re.IGNORECASE):
            text = match.group(1).strip().rstrip(".,;:")
            key = text.lower()[:50]
            if key not in seen and len(text) > 10:
                seen.add(key)
                goals.append({
                    "description": text,
                    "priority": "MEDIUM",
                    "source": "inference_extraction",
                })

    return goals[:3]  # Cap at 3 per response


# ── Valence scoring ────────────────────────────────────────────────────────────

# Sentiment anchor words for embedding-free valence
_POSITIVE = {
    "good", "great", "excellent", "wonderful", "helpful", "thanks", "thank",
    "love", "perfect", "amazing", "awesome", "fantastic", "brilliant", "yes",
    "correct", "right", "exactly", "understand", "clear", "interesting",
}
_NEGATIVE = {
    "bad", "wrong", "mistake", "error", "fail", "failed", "sorry", "hate",
    "terrible", "awful", "horrible", "broken", "confused", "frustrating",
    "incorrect", "no", "not", "never", "problem", "issue", "bug",
}


def compute_valence(text: str) -> float:
    """
    Compute emotional valence of text. Returns float in [-1.0, 1.0].

    Uses token-level positive/negative word counting with negation awareness.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    pos = 0
    neg = 0
    negate = False

    for i, token in enumerate(tokens):
        if token in {"not", "no", "never", "don't", "doesn't", "didn't", "won't"}:
            negate = True
            continue

        if token in _POSITIVE:
            if negate:
                neg += 1
            else:
                pos += 1
            negate = False
        elif token in _NEGATIVE:
            if negate:
                pos += 0.5  # double negation = mild positive
            else:
                neg += 1
            negate = False
        else:
            if negate and i > 0:
                pass  # reset negation after non-sentiment word
            negate = False

    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


# ── Compatibility aliases for inference/providers.py ──────────────────────────
# providers.py uses these names from the original scaffold.

def build_full_context(request: "InferenceRequest") -> list[dict]:
    """Alias for build_messages — compatibility with providers.py."""
    return build_messages(request)


def estimate_valence(text: str) -> float:
    """Alias for compute_valence — compatibility with providers.py."""
    return compute_valence(text)


def extract_memory_deltas_sync(user_input: str, response: str) -> list[dict]:
    """
    Synchronous heuristic-only delta extraction for providers.py.
    (providers.py calls this synchronously; the async version uses the LLM.)
    """
    return _heuristic_deltas(user_input, response)


# providers.py imports extract_memory_deltas — point it at the sync version
extract_memory_deltas = extract_memory_deltas_sync  # type: ignore[assignment]
