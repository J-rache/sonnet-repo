import json
from pathlib import Path

from fastapi.testclient import TestClient
import yaml


TOKEN = "test-local-token"


def write_config(tmp_path: Path) -> Path:
    data_dir = tmp_path / "runtime"
    config = {
        "base_model": "mock-model",
        "data_dir": str(data_dir),
        "working_memory_capacity": 2048,
        "episodic_db_path": str(data_dir / "episodic.db"),
        "semantic_db_path": str(data_dir / "semantic.db"),
        "adapter_path": str(data_dir / "adapter"),
        "core_state_path": str(data_dir / "core_state.json"),
        "journal_path": str(data_dir / "events.jsonl"),
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "api_log_level": "warning",
        "local_api_token": TOKEN,
        "inference_provider": "mock",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def event_types(path: Path) -> set[str]:
    return {
        json.loads(line)["type"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_api_smoke_without_live_model_call(monkeypatch, tmp_path):
    config_path = write_config(tmp_path)
    monkeypatch.setenv("PNP_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PNP_INFERENCE_PROVIDER", "mock")

    from api import server

    headers = {"X-PNP-Token": TOKEN}
    with TestClient(server.app) as client:
        assert client.get("/").json()["status"] == "alive"
        assert client.get("/state").status_code == 200
        assert client.get("/goals").json()["active_goals"] == []
        assert client.post("/goals", json={"description": "blocked"}).status_code == 401

        goal_response = client.post(
            "/goals",
            json={"description": "verify local continuity", "priority": "HIGH"},
            headers=headers,
        )
        assert goal_response.status_code == 200
        goal_id = goal_response.json()["goal_id"]
        assert client.delete(f"/goals/{goal_id}", headers=headers).status_code == 200

        server._core.consolidator.semantic.store_fact(
            content="Jae prefers concise implementation notes.",
            source_episode_ids=[],
            domain="user_preferences",
            confidence=0.9,
        )

        feedback_response = client.post(
            "/feedback",
            json={
                "content": "Use concise implementation notes when explaining work.",
                "feedback": 0.8,
                "domain": "user_preferences",
                "confidence": 0.9,
            },
            headers=headers,
        )
        assert feedback_response.status_code == 200
        assert feedback_response.json()["applied"] is True

        chat_response = client.post(
            "/chat",
            json={
                "message": "remember concise implementation notes",
                "concepts": ["concise", "implementation"],
            },
            headers=headers,
        )
        assert chat_response.status_code == 200
        body = chat_response.json()
        assert body["response"].startswith("Mock inference response")
        assert body["context_used"]["semantic"] is True
        assert body["context_used"]["adaptation"] is True

        recent = client.get("/memory/recent").json()
        assert recent["stats"]["total_episodes"] >= 1

    journal_path = tmp_path / "runtime" / "events.jsonl"
    assert {
        "interaction_received",
        "goal_added",
        "goal_completed",
        "memory_written",
        "adapter_delta_applied",
    }.issubset(event_types(journal_path))
