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
        "project_data_dir": str(runtime_dir / "projects"),
        "project_archive_dir": str(runtime_dir / "project_archives"),
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
    os.environ["PNP_LOCAL_TOKEN"] = TOKEN

    from fastapi.testclient import TestClient
    from api import server

    headers = {"X-PNP-Token": TOKEN}
    result: dict = {"checks": []}

    with TestClient(server.app) as client:
        root = client.get("/")
        result["checks"].append(["GET /", root.status_code])

        setup = client.get("/setup/status")
        result["checks"].append(["GET /setup/status", setup.status_code])
        result["setup_status"] = setup.json()

        demo = client.post(
            "/setup/demo",
            json={"project_id": "smoke-demo-project", "title": "Smoke Demo Project"},
            headers=headers,
        )
        result["checks"].append(["POST /setup/demo", demo.status_code])
        result["demo_project_has_lanes"] = "demo:alpha" in demo.json().get("participant_lanes", {})

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

        progress_blocked = client.patch(f"/goals/{goal_id}/progress", json={"progress": 0.25})
        result["checks"].append(["PATCH /goals/{goal_id}/progress without token", progress_blocked.status_code])

        progress = client.patch(
            f"/goals/{goal_id}/progress",
            json={"progress": 0.5, "notes": "halfway through smoke"},
            headers=headers,
        )
        result["checks"].append(["PATCH /goals/{goal_id}/progress with token", progress.status_code])

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

        episode_blocked = client.post(
            "/memory/episodic",
            json={"content": "blocked episode", "summary": "blocked episode"},
        )
        result["checks"].append(["POST /memory/episodic without token", episode_blocked.status_code])

        episode = client.post(
            "/memory/episodic",
            json={
                "content": "Manual smoke episode about concise notes.",
                "summary": "Manual smoke episode.",
                "tags": ["smoke"],
            },
            headers=headers,
        )
        result["checks"].append(["POST /memory/episodic with token", episode.status_code])

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

        adapter_sync = client.get("/adapter/sync")
        result["checks"].append(["GET /adapter/sync", adapter_sync.status_code])
        result["adapter_sync"] = adapter_sync.json()

        consolidate_blocked = client.post("/memory/consolidate")
        result["checks"].append(["POST /memory/consolidate without token", consolidate_blocked.status_code])

        semantic_stats = client.get("/memory/semantic", params={"q": "concise notes"})
        result["checks"].append(["GET /memory/semantic", semantic_stats.status_code])
        result["semantic_stats"] = semantic_stats.json()

        project = client.post("/projects", json={"project_id": "smoke-project"}, headers=headers)
        result["checks"].append(["POST /projects with token", project.status_code])

        summary = client.post(
            "/projects/smoke-project/stenographer/summary",
            json={"summary": "Stenographer smoke summary: seat identity is temporary routing."},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/stenographer/summary", summary.status_code])

        bind_a = client.post(
            "/projects/smoke-project/seats/seat-A/bind",
            json={"participant_identity": "participant-X", "model_id": "mock-model"},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/seats/{seat}/bind X", bind_a.status_code])

        participant_memory = client.post(
            "/projects/smoke-project/participants/participant-X/memory",
            json={
                "content": "Participant X smoke-only continuity note.",
                "summary": "Participant X smoke-only continuity note.",
                "tags": ["smoke"],
            },
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/participants/{id}/memory", participant_memory.status_code])

        bind_c = client.post(
            "/projects/smoke-project/seats/seat-C/bind",
            json={"participant_identity": "participant-X", "model_id": "mock-model"},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/seats/{seat}/bind move", bind_c.status_code])

        bind_z = client.post(
            "/projects/smoke-project/seats/seat-A/bind",
            json={"participant_identity": "participant-Z", "model_id": "mock-model"},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/seats/{seat}/bind Z", bind_z.status_code])

        x_context = client.get("/projects/smoke-project/seats/seat-C/continuity", params={"q": "smoke-only"})
        z_context = client.get("/projects/smoke-project/seats/seat-A/continuity", params={"q": "smoke-only"})
        result["checks"].append(["GET participant X continuity", x_context.status_code])
        result["checks"].append(["GET participant Z continuity", z_context.status_code])
        result["project_continuity_isolated"] = (
            "Participant X smoke-only continuity note" in x_context.json().get("context", "")
            and "Participant X smoke-only continuity note" not in z_context.json().get("context", "")
        )

        tool = client.post(
            "/projects/smoke-project/toolbox/tools",
            json={"name": "smoke verifier", "description": "Project-scoped smoke helper", "source": "created_on_demand"},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/toolbox/tools", tool.status_code])
        tool_verified = client.post(
            f"/projects/smoke-project/toolbox/tools/{tool.json()['id']}/verify",
            json={"status": "verified"},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/toolbox/tools/{tool_id}/verify", tool_verified.status_code])

        lesson = client.post(
            "/projects/smoke-project/lessons",
            json={
                "content": "Do not reuse old occupant continuity when a new participant takes the same seat.",
                "source": "smoke_api",
                "confidence": 0.9,
                "tags": ["seat_binding"],
            },
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/lessons", lesson.status_code])
        lesson_verified = client.post(
            f"/projects/smoke-project/lessons/{lesson.json()['id']}/verify",
            json={"status": "verified"},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/lessons/{lesson_id}/verify", lesson_verified.status_code])

        package = client.get("/projects/smoke-project/package")
        result["checks"].append(["GET /projects/{id}/package", package.status_code])
        result["package_has_participant_lanes"] = "participant-X" in package.json().get("participant_lanes", {})

        project_chat = client.post(
            "/projects/smoke-project/seats/seat-A/chat",
            json={"message": "remember project continuity smoke", "max_tokens": 64},
            headers=headers,
        )
        result["checks"].append(["POST /projects/{id}/seats/{seat}/chat", project_chat.status_code])
        result["project_chat"] = project_chat.json()

        archive = client.post("/projects/smoke-project/archive", headers=headers)
        result["checks"].append(["POST /projects/{id}/archive", archive.status_code])
        result["project_archive"] = archive.json()

    journal_path = runtime_dir / "events.jsonl"
    event_types = [
        json.loads(line)["type"]
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result["event_types"] = sorted(set(event_types))
    result["artifact_dir"] = str(artifact_dir)
    result["journal_path"] = str(journal_path)
    result["ok"] = (
        all(status in {200, 401} for _, status in result["checks"])
        and result.get("project_continuity_isolated", False)
        and result.get("demo_project_has_lanes", False)
        and result.get("package_has_participant_lanes", False)
    )

    result_path = artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
