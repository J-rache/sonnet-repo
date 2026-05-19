from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TOKEN = "smoke-local-token"


def main() -> int:
    artifact_dir = REPO_ROOT / ".smoke" / "api-smoke"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    runtime_dir = artifact_dir / "runtime"
    runtime_dir.mkdir(parents=True)

    config = {
        "model_id": "mock-model",
        "data_dir": str(runtime_dir),
        "working_memory_capacity": 2048,
        "episodic_db_path": str(runtime_dir / "episodic.db"),
        "semantic_db_path": str(runtime_dir / "semantic.db"),
        "adapter_path": str(runtime_dir / "adapter"),
        "core_state_path": str(runtime_dir / "core_state.json"),
        "journal_path": str(runtime_dir / "events.jsonl"),
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "api_log_level": "warning",
        "local_api_token": TOKEN,
        "inference_provider": "mock",
    }
    config_path = artifact_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    os.environ["PNP_CONFIG_PATH"] = str(config_path)
    os.environ["PNP_INFERENCE_PROVIDER"] = "mock"

    from fastapi.testclient import TestClient
    from api import server

    headers = {"X-PNP-Token": TOKEN}
    result: dict = {"checks": []}

    with TestClient(server.app) as client:
        root = client.get("/")
        result["checks"].append(["GET /", root.status_code])

        unauthorized = client.post("/goals", json={"description": "blocked"})
        result["checks"].append(["POST /goals without token", unauthorized.status_code])

        goal = client.post(
            "/goals",
            json={"description": "smoke continuity goal", "priority": "HIGH"},
            headers=headers,
        )
        result["checks"].append(["POST /goals with token", goal.status_code])
        goal_id = goal.json()["goal_id"]

        goals = client.get("/goals")
        result["checks"].append(["GET /goals", goals.status_code])

        completed = client.delete(f"/goals/{goal_id}", headers=headers)
        result["checks"].append(["DELETE /goals/{goal_id} with token", completed.status_code])

        server._core.consolidator.semantic.store_fact(
            content="Smoke fact: concise notes should be retrieved.",
            source_episode_ids=[],
            domain="smoke",
            confidence=0.9,
        )

        feedback = client.post(
            "/feedback",
            json={
                "content": "Use concise notes in smoke responses.",
                "feedback": 0.8,
                "domain": "smoke",
                "confidence": 0.8,
            },
            headers=headers,
        )
        result["checks"].append(["POST /feedback with token", feedback.status_code])

        train = client.post("/adapter/train", json={"epochs": 5}, headers=headers)
        result["checks"].append(["POST /adapter/train with token", train.status_code])
        result["adapter_train"] = train.json()["metrics"]

        chat = client.post(
            "/chat",
            json={"message": "remember concise notes", "concepts": ["concise"]},
            headers=headers,
        )
        result["checks"].append(["POST /chat mock inference", chat.status_code])
        result["chat"] = chat.json()

        recent = client.get("/memory/recent")
        result["checks"].append(["GET /memory/recent", recent.status_code])
        result["recent_stats"] = recent.json()["stats"]

        state = client.get("/state")
        result["checks"].append(["GET /state", state.status_code])

        adapter_stats = client.get("/adapter/stats")
        result["checks"].append(["GET /adapter/stats", adapter_stats.status_code])
        result["adapter_stats"] = adapter_stats.json()

        semantic_stats = client.get("/memory/semantic")
        result["checks"].append(["GET /memory/semantic", semantic_stats.status_code])
        result["semantic_stats"] = semantic_stats.json()

    journal_path = runtime_dir / "events.jsonl"
    event_types = [
        json.loads(line)["type"]
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result["event_types"] = sorted(set(event_types))
    result["artifact_dir"] = str(artifact_dir)
    result["journal_path"] = str(journal_path)
    result["ok"] = all(status in {200, 401} for _, status in result["checks"])

    result_path = artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
