from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from project.continuity import ProjectContinuityService


TOKEN = "project-test-token"


def project_config(tmp_path: Path) -> dict:
    data_dir = tmp_path / "runtime"
    return {
        "model_id": "mock-model",
        "base_model": "mock-model",
        "data_dir": str(data_dir),
        "project_data_dir": str(data_dir / "projects"),
        "project_archive_dir": str(data_dir / "project_archives"),
        "working_memory_capacity": 2048,
        "embedding_dimensions": 64,
        "episodic_db_path": str(data_dir / "single_episodic.db"),
        "semantic_db_path": str(data_dir / "single_semantic.db"),
        "adapter_path": str(data_dir / "single_adapter"),
        "embed_path": str(data_dir / "single_embeddings.pkl"),
        "journal_path": str(data_dir / "single_events.jsonl"),
        "core_state_path": str(data_dir / "single_core_state.json"),
        "adapter_training_epochs": 4,
        "api_host": "127.0.0.1",
        "api_port": 8000,
        "local_api_token": TOKEN,
        "inference_provider": "mock",
    }


def test_participant_keeps_continuity_after_seat_move_and_new_occupant_is_isolated(tmp_path):
    config = project_config(tmp_path)
    service = ProjectContinuityService(config)
    project_id = "mystro-project"

    service.record_stenographer_summary(
        project_id,
        summary="Stenographer: repo work is focused on project continuity.",
        history_package="Project history says seat identity is temporary routing only.",
    )

    first = service.bind_seat(project_id, "seat-A", "participant-X", model_id="mock-model")
    assert first["lane_key"] == f"{project_id}:participant-X"
    assert first["hydration"]["source"] == "stenographer_summary"

    lane_x = service.participant_lane(project_id, "participant-X")
    lane_x.store_episode(
        content="Participant X private continuity note: use the exact Rust verifier.",
        summary="Participant X private Rust verifier habit.",
        tags=["participant_private"],
    )
    assert lane_x.apply_delta(
        "Participant X private adapter habit: exact Rust verifier first.",
        feedback=0.9,
        domain="participant_private",
        confidence=0.9,
        invariant_check=False,
    )

    moved = service.bind_seat(project_id, "seat-C", "participant-X", model_id="mock-model")
    assert moved["moved_from"] == ["seat-A"]
    assert "seat-A" not in service.seat_bindings(project_id)
    assert service.participant_for_seat(project_id, "seat-C") == "participant-X"

    replacement = service.bind_seat(project_id, "seat-A", "participant-Z", model_id="mock-model")
    assert replacement["previous_participant"] is None
    assert replacement["lane_key"] == f"{project_id}:participant-Z"

    x_context = service.participant_lane(project_id, "participant-X").build_context("Rust verifier").to_block()
    z_context = service.participant_lane(project_id, "participant-Z").build_context("Rust verifier").to_block()
    assert "Participant X private Rust verifier habit" in x_context
    assert "Participant X private Rust verifier habit" not in z_context
    assert "Stenographer: repo work is focused on project continuity" in z_context


def test_first_join_hydrates_from_stenographer_and_returning_participant_restores_lane(tmp_path):
    config = project_config(tmp_path)
    project_id = "returning-project"
    service = ProjectContinuityService(config)
    service.record_stenographer_summary(
        project_id,
        summary="Stenographer: bootstrap from shared project history before private learning.",
        history_package="History package: decisions and current state are project-wide.",
    )

    joined = service.bind_seat(project_id, "seat-B", "participant-returning", model_id="mock-model")
    assert joined["lane_existed"] is False
    lane = service.participant_lane(project_id, "participant-returning")
    assert lane.manifest["hydrated_from"] == "stenographer_summary"
    assert "bootstrap from shared project history" in lane.build_context("bootstrap history").to_block()

    lane.store_episode(
        content="Returning participant private note: resume in api/server.py.",
        summary="Resume in api/server.py for this participant.",
        tags=["resume"],
    )
    lane.set_resume_point({"file": "api/server.py", "next_step": "wire project API"})

    service_restarted = ProjectContinuityService(config)
    service_restarted.record_stenographer_summary(
        project_id,
        summary="Stenographer fresh update: tests should cover seat movement.",
    )
    rebound = service_restarted.bind_seat(project_id, "seat-D", "participant-returning", model_id="mock-model")
    assert rebound["lane_existed"] is True

    restored_lane = service_restarted.participant_lane(project_id, "participant-returning")
    assert restored_lane.manifest["resume_point"]["file"] == "api/server.py"
    context = restored_lane.build_context("api server tests seat movement").to_block()
    assert "Resume in api/server.py for this participant" in context
    assert "tests should cover seat movement" in context


def test_shared_toolbox_lessons_and_archive_restore_are_project_scoped(tmp_path):
    config = project_config(tmp_path)
    project_id = "archive-project"
    service = ProjectContinuityService(config)
    service.record_stenographer_summary(project_id, "Stenographer: archive must include all project layers.")
    service.bind_seat(project_id, "seat-A", "participant-A", model_id="mock-model")

    tool = service.add_tool(
        project_id,
        name="project verifier",
        description="Runs the project continuity smoke tests.",
        command="python -m pytest tests/test_project_continuity.py",
        source="created_on_demand",
        created_by="participant-A",
        useful=True,
    )
    assert tool["scope"] == "project"
    assert service.bind_seat(project_id, "seat-C", "participant-A")["moved_from"] == ["seat-A"]
    assert service.list_tools(project_id)[0]["id"] == tool["id"]

    lane = service.participant_lane(project_id, "participant-A")
    before_lessons = lane.episodic_memory.stats()["total_episodes"]
    lesson = service.add_lesson(
        project_id,
        content="Seat ids must never be used as participant continuity keys.",
        source="test_project_continuity",
        confidence=0.95,
        tags=["seat_binding", "identity"],
    )
    after_lessons = lane.episodic_memory.stats()["total_episodes"]
    assert after_lessons == before_lessons
    assert service.retrieve_lessons(project_id, "participant continuity keys")[0]["id"] == lesson["id"]

    archive = service.archive_project(project_id)
    archive_path = Path(archive["archive_path"])
    assert archive_path.exists()
    service.add_tool(
        project_id,
        name="post-archive temporary tool",
        description="This should disappear after restore.",
    )
    assert len(service.list_tools(project_id)) == 2

    restored_service = ProjectContinuityService(config)
    restored = restored_service.restore_project(project_id, str(archive_path))
    assert restored["restored"] is True
    restored_project = restored_service.get_project(project_id)
    assert "participant-A" in restored_project["participants"]
    assert len(restored_service.list_tools(project_id)) == 1
    assert restored_service.list_tools(project_id)[0]["name"] == "project verifier"
    assert restored_service.list_lessons(project_id)[0]["content"].startswith("Seat ids must never")
    assert restored_service.participant_lane(project_id, "participant-A").episodic_memory.stats()["total_episodes"] >= 1


def test_project_api_routes_preserve_identity_and_do_not_break_single_lane(monkeypatch, tmp_path):
    config = project_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("PNP_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PNP_INFERENCE_PROVIDER", "mock")

    from api import server

    headers = {"X-PNP-Token": TOKEN}
    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200
        assert client.post("/projects", json={"project_id": "api-project"}, headers=headers).status_code == 200
        assert client.post(
            "/projects/api-project/stenographer/summary",
            json={"summary": "Stenographer: API project history is available to new participants."},
            headers=headers,
        ).status_code == 200
        bind_x = client.post(
            "/projects/api-project/seats/seat-A/bind",
            json={"participant_identity": "participant-X", "model_id": "mock-model"},
            headers=headers,
        )
        assert bind_x.status_code == 200
        assert bind_x.json()["hydration"]["source"] == "stenographer_summary"

        assert client.post(
            "/projects/api-project/participants/participant-X/memory",
            json={
                "content": "Participant X API-only note.",
                "summary": "Participant X API-only note.",
                "tags": ["private"],
            },
            headers=headers,
        ).status_code == 200
        assert client.post(
            "/projects/api-project/seats/seat-C/bind",
            json={"participant_identity": "participant-X", "model_id": "mock-model"},
            headers=headers,
        ).json()["moved_from"] == ["seat-A"]
        assert client.post(
            "/projects/api-project/seats/seat-A/bind",
            json={"participant_identity": "participant-Z", "model_id": "mock-model"},
            headers=headers,
        ).status_code == 200

        z_context = client.get("/projects/api-project/seats/seat-A/continuity?q=API-only").json()["context"]
        x_context = client.get("/projects/api-project/seats/seat-C/continuity?q=API-only").json()["context"]
        assert "Participant X API-only note" not in z_context
        assert "Participant X API-only note" in x_context

        tool_response = client.post(
            "/projects/api-project/toolbox/tools",
            json={"name": "api smoke helper", "description": "Useful project tool", "source": "created_on_demand"},
            headers=headers,
        )
        assert tool_response.status_code == 200
        assert client.get("/projects/api-project/toolbox").json()["tools"][0]["name"] == "api smoke helper"

        lesson_response = client.post(
            "/projects/api-project/lessons",
            json={
                "content": "Do not hydrate new participants from old seat occupants.",
                "source": "api-test",
                "confidence": 0.9,
                "tags": ["hydration"],
            },
            headers=headers,
        )
        assert lesson_response.status_code == 200
        assert client.get("/projects/api-project/lessons?q=hydrate").json()["lessons"][0]["source"] == "api-test"

        chat_response = client.post(
            "/projects/api-project/seats/seat-A/chat",
            json={"message": "remember this project API continuity check", "max_tokens": 64},
            headers=headers,
        )
        assert chat_response.status_code == 200
        assert chat_response.json()["participant_identity"] == "participant-Z"
        assert chat_response.json()["provider"] == "mock"

        single_lane_chat = client.post(
            "/chat",
            json={"message": "single lane still works"},
            headers=headers,
        )
        assert single_lane_chat.status_code == 200
        assert single_lane_chat.json()["provider"] == "mock"
