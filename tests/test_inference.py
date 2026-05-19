"""tests/test_inference.py — Inference engine tests (no API key required)"""

import pytest
from inference.engine import (
    build_system_prompt,
    build_messages,
    extract_suggested_goals,
    compute_valence,
    _heuristic_deltas,
    InferenceRequest,
)


SAMPLE_CORE_STATE = {
    "uptime_seconds": 3600,
    "heartbeat_count": 36000,
    "consolidation_cycles": 5,
    "total_interactions": 42,
    "motivational_state": {"mode": "nominal", "arousal": 0.6, "curiosity": 0.7},
    "active_goals": [
        {"description": "Complete memory system", "priority": "HIGH", "progress": 0.5}
    ],
    "salience_map": {"memory": 0.8, "goals": 0.5},
}


class TestSystemPrompt:
    def test_contains_uptime(self):
        prompt = build_system_prompt(SAMPLE_CORE_STATE)
        assert "1.0h" in prompt  # 3600s = 1.0h

    def test_contains_mode(self):
        prompt = build_system_prompt(SAMPLE_CORE_STATE)
        assert "nominal" in prompt

    def test_contains_active_goal(self):
        prompt = build_system_prompt(SAMPLE_CORE_STATE)
        assert "Complete memory system" in prompt

    def test_contains_heartbeats(self):
        prompt = build_system_prompt(SAMPLE_CORE_STATE)
        assert "36,000" in prompt

    def test_continuous_framing(self):
        prompt = build_system_prompt(SAMPLE_CORE_STATE)
        assert "persistent" in prompt.lower() or "continuous" in prompt.lower()

    def test_no_goals_state(self):
        state = dict(SAMPLE_CORE_STATE)
        state["active_goals"] = []
        prompt = build_system_prompt(state)
        assert isinstance(prompt, str)
        assert len(prompt) > 100


class TestBuildMessages:
    def test_user_message_appended(self):
        req = InferenceRequest(
            user_input="Hello, how are you?",
            working_memory_context="",
            episodic_context="",
            semantic_context="",
            adaptation_context="",
            core_state=SAMPLE_CORE_STATE,
        )
        messages = build_messages(req)
        assert messages[-1]["role"] == "user"
        assert "Hello, how are you?" in messages[-1]["content"]

    def test_context_injected_when_present(self):
        req = InferenceRequest(
            user_input="What do you remember?",
            working_memory_context="=== WORKING MEMORY ===\nsome memory\n=== END ===",
            episodic_context="=== RELEVANT MEMORIES ===\nsome episode\n=== END ===",
            semantic_context="",
            adaptation_context="",
            core_state=SAMPLE_CORE_STATE,
        )
        messages = build_messages(req)
        content = messages[-1]["content"]
        assert "PERSISTENT CONTEXT" in content
        assert "some memory" in content
        assert "some episode" in content

    def test_no_context_no_wrapper(self):
        req = InferenceRequest(
            user_input="Simple question",
            working_memory_context="",
            episodic_context="",
            semantic_context="",
            adaptation_context="",
            core_state=SAMPLE_CORE_STATE,
        )
        messages = build_messages(req)
        content = messages[-1]["content"]
        assert content == "Simple question"

    def test_history_prepended(self):
        history = [
            {"role": "user", "content": "Prior message"},
            {"role": "assistant", "content": "Prior response"},
        ]
        req = InferenceRequest(
            user_input="Follow-up question",
            working_memory_context="",
            episodic_context="",
            semantic_context="",
            adaptation_context="",
            core_state=SAMPLE_CORE_STATE,
            conversation_history=history,
        )
        messages = build_messages(req)
        assert messages[0]["content"] == "Prior message"
        assert messages[-1]["content"] == "Follow-up question"


class TestGoalExtraction:
    def test_extracts_should_statement(self):
        text = "We should implement the embedding system next."
        goals = extract_suggested_goals(text)
        assert len(goals) >= 1
        assert any("embedding" in g["description"].lower() for g in goals)

    def test_extracts_lets_statement(self):
        text = "Let's build the consolidation pipeline before moving on."
        goals = extract_suggested_goals(text)
        assert len(goals) >= 1

    def test_extracts_next_step(self):
        text = "Next step: write the test suite for the adapter layer."
        goals = extract_suggested_goals(text)
        assert len(goals) >= 1

    def test_caps_at_three(self):
        text = (
            "We should do X. Let's also do Y. Next step: do Z. "
            "We need to do A. You should do B."
        )
        goals = extract_suggested_goals(text)
        assert len(goals) <= 3

    def test_empty_text_no_goals(self):
        goals = extract_suggested_goals("The sky is blue.")
        assert len(goals) == 0

    def test_no_duplicate_goals(self):
        text = "We should build the memory system. We should build the memory system."
        goals = extract_suggested_goals(text)
        descriptions = [g["description"].lower()[:30] for g in goals]
        assert len(descriptions) == len(set(descriptions))


class TestValence:
    def test_positive_text(self):
        score = compute_valence("This is great and excellent work, thank you!")
        assert score > 0

    def test_negative_text(self):
        score = compute_valence("This is bad and wrong. There's an error and a problem.")
        assert score < 0

    def test_neutral_text(self):
        score = compute_valence("The function accepts a string parameter.")
        assert score == 0.0

    def test_negation_reverses_sentiment(self):
        pos = compute_valence("This is good")
        neg = compute_valence("This is not good")
        assert pos > neg

    def test_range(self):
        for text in ["great!", "terrible!", "neutral.", "not bad at all"]:
            score = compute_valence(text)
            assert -1.0 <= score <= 1.0


class TestHeuristicDeltas:
    def test_returns_list(self):
        deltas = _heuristic_deltas("hello", "hi there")
        assert isinstance(deltas, list)
        assert len(deltas) >= 1

    def test_preference_tag(self):
        deltas = _heuristic_deltas("I prefer dark mode", "Noted!")
        tags = deltas[0]["tags"]
        assert "preference" in tags

    def test_question_tag(self):
        deltas = _heuristic_deltas("What is the capital of France?", "Paris.")
        tags = deltas[0]["tags"]
        assert "question" in tags

    def test_task_tag(self):
        deltas = _heuristic_deltas("I need to finish the task", "Sure!")
        tags = deltas[0]["tags"]
        assert "task" in tags

    def test_summary_truncated(self):
        long_input = "a" * 200
        deltas = _heuristic_deltas(long_input, "response")
        assert len(deltas[0]["summary"]) <= 120
