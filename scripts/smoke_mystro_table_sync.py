from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_PARTICIPANTS = [
    {
        "seat": "seat-5",
        "participant_identity": "continuity-probe-7f3a9c",
        "display_name": "Subagent Einstein",
        "continuity_phrase": "PNP continuity verified across Mystro Table handoff",
    },
    {
        "seat": "seat-6",
        "participant_identity": "continuity_probe_jae_codex_20260519_b",
        "display_name": "Subagent Averroes",
        "continuity_phrase": "Mystro Table continuity verified across PNP participant handoff",
    },
    {
        "seat": "seat-7",
        "participant_identity": "codex-pnp-continuity-observer-3",
        "display_name": "Subagent Kuhn",
        "continuity_phrase": "Mystro Table continuity check acknowledged by independent participant three.",
    },
]
REPLACEMENT = {
    "seat": "seat-5",
    "participant_identity": "continuity_probe_replacement_delta",
    "display_name": "Replacement Delta",
    "continuity_phrase": "Replacement participant must not inherit the first participant's private sync phrase.",
}
QWEN_MODEL = "qwen2.5-coder:7b"


class SmokeFailure(RuntimeError):
    pass


def _json_bytes(payload: dict[str, Any] | None) -> bytes | None:
    if payload is None:
        return None
    return json.dumps(payload).encode("utf-8")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> Any:
    body = _json_bytes(payload)
    req_headers = {"Accept": "application/json"}
    if body is not None:
        req_headers["Content-Type"] = "application/json"
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SmokeFailure(f"{method} {url} failed: {exc.reason}") from exc


def get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    return request_json("GET", url, headers=headers, timeout=timeout)


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    return request_json("POST", url, payload=payload, headers=headers, timeout=timeout)


def patch_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    return request_json("PATCH", url, payload=payload, headers=headers, timeout=timeout)


def url_join(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def query_url(base: str, path: str, **params: str) -> str:
    return url_join(base, path) + "?" + urllib.parse.urlencode(params)


def discover_pnp_token() -> str:
    token = os.environ.get("PNP_LOCAL_TOKEN", "").strip()
    if token:
        return token

    start_script = Path(os.environ.get("LOCALAPPDATA", "")) / "PNP" / "PNP-Start.ps1"
    if start_script.exists():
        text = start_script.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\$env:PNP_LOCAL_TOKEN\s*=\s*'([^']+)'", text)
        if match:
            return match.group(1)

    installed_config = Path(os.environ.get("LOCALAPPDATA", "")) / "PNP" / "config" / "installed.yaml"
    if installed_config.exists():
        for line in installed_config.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("local_api_token:"):
                return line.split(":", 1)[1].strip().strip("'\"")

    config = Path("config") / "default.yaml"
    if config.exists():
        for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("local_api_token:"):
                return line.split(":", 1)[1].strip().strip("'\"")

    raise SmokeFailure("PNP token not found in PNP_LOCAL_TOKEN, installed config, or default config.")


def pnp_headers() -> dict[str, str]:
    return {"X-PNP-Token": discover_pnp_token()}


def mystro_headers() -> dict[str, str]:
    token = os.environ.get("MYSTRO_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def compact_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_services(args: argparse.Namespace) -> dict[str, Any]:
    pnp_health = get_json(url_join(args.pnp_url, "/health"))
    mystro_health = get_json(url_join(args.mystro_url, "/api/health"))
    tags = get_json(url_join(args.ollama_url, "/api/tags"), timeout=20)
    model_names = [item.get("name", "") for item in tags.get("models", [])]
    assert_true(QWEN_MODEL in model_names, f"{QWEN_MODEL} is not installed in Ollama.")
    return {
        "pnp_health": pnp_health,
        "mystro_health": mystro_health,
        "ollama_model_count": len(model_names),
        "qwen_available": True,
    }


def fill_mystro_seat(args: argparse.Namespace, participant: dict[str, str]) -> dict[str, Any]:
    payload = {
        "providerId": "ollama",
        "instanceLabel": f"{participant['display_name']} / {QWEN_MODEL}",
        "currentFocus": (
            f"PNP sync lane for {participant['participant_identity']}; "
            "continuity is keyed by participant identity, not this seat."
        ),
    }
    response = patch_json(
        url_join(args.mystro_url, f"/api/seats/{participant['seat']}/occupant"),
        payload,
        headers=mystro_headers(),
    )
    occupant = response["seat"]["occupant"]
    assert_true(occupant["providerId"] == "ollama", f"{participant['seat']} was not filled by Ollama.")
    assert_true(occupant["connectionStatus"] == "connected", f"{participant['seat']} Ollama is not connected.")
    return response


def bind_pnp_seat(args: argparse.Namespace, project_id: str, participant: dict[str, str]) -> dict[str, Any]:
    payload = {
        "participant_identity": participant["participant_identity"],
        "display_name": participant["display_name"],
        "provider": "ollama",
        "model_id": QWEN_MODEL,
        "metadata": {
            "mystro_seat": participant["seat"],
            "live_smoke": True,
            "continuity_phrase": participant["continuity_phrase"],
        },
    }
    return post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/seats/{participant['seat']}/bind"),
        payload,
        headers=pnp_headers(),
    )


def project_chat(args: argparse.Namespace, project_id: str, seat: str, message: str) -> dict[str, Any]:
    response = post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/seats/{seat}/chat"),
        {"message": message, "concepts": ["mystro", "participant_continuity", "qwen_sync"], "max_tokens": 80},
        headers=pnp_headers(),
        timeout=120,
    )
    assert_true(response.get("provider") == "ollama", f"{seat} did not route through the Ollama provider.")
    assert_true(response.get("model") == QWEN_MODEL, f"{seat} did not use {QWEN_MODEL}: {response.get('model')}")
    assert_true(bool(response.get("response", "").strip()), f"{seat} returned an empty model response.")
    return response


def create_and_dispatch_mystro_task(args: argparse.Namespace, project_id: str, seat: str) -> dict[str, Any]:
    task_packet = post_json(
        url_join(args.mystro_url, "/api/tasks"),
        {
            "title": f"PNP live Qwen sync check {project_id}",
            "details": (
                "Use the local Ollama model for a concise verification note. "
                f"The selected sync model for this test is {QWEN_MODEL}."
            ),
            "assignedSeatId": seat,
            "toolIds": ["ollama-generate"],
            "acceptanceCriteria": [f"Dispatch model is {QWEN_MODEL}"],
        },
        headers=mystro_headers(),
        timeout=60,
    )
    task_id = task_packet["task"]["id"]
    dispatch_packet = post_json(
        url_join(args.mystro_url, f"/api/tasks/{task_id}/dispatch"),
        {},
        headers=mystro_headers(),
        timeout=130,
    )
    dispatch = dispatch_packet["dispatch"]
    assert_true(dispatch.get("delivered") is True, f"Mystro dispatch was not delivered: {dispatch}")
    assert_true(dispatch.get("model") == QWEN_MODEL, f"Mystro dispatched {dispatch.get('model')} instead of {QWEN_MODEL}.")
    assert_true(bool(dispatch.get("output", "").strip()), "Mystro Ollama dispatch returned empty output.")
    return dispatch_packet


def create_or_select_mystro_project(args: argparse.Namespace) -> dict[str, Any]:
    if args.use_active_mystro_project:
        room = get_json(url_join(args.mystro_url, "/api/room"))
        active = room.get("activeProject") or {}
        assert_true(active.get("pnpProjectId"), "Mystro active project did not expose pnpProjectId.")
        return {
            "created": False,
            "project": active,
            "activeProjectId": active["id"],
        }

    name = f"PNP Live Sync Smoke {int(time.time())}"
    packet = post_json(
        url_join(args.mystro_url, "/api/projects"),
        {"name": name, "activate": True},
        headers=mystro_headers(),
    )
    project = packet["project"]
    assert_true(packet["activeProjectId"] == project["id"], "Mystro did not activate the smoke project.")
    assert_true(project.get("pnpProjectId"), "Mystro project did not expose pnpProjectId.")
    return {"created": True, **packet}


def select_mystro_project(args: argparse.Namespace, project_id: str) -> dict[str, Any]:
    return patch_json(
        url_join(args.mystro_url, "/api/projects/active"),
        {"projectId": project_id},
        headers=mystro_headers(),
    )


def continuity_context(args: argparse.Namespace, project_id: str, seat: str, query: str) -> dict[str, Any]:
    return get_json(query_url(args.pnp_url, f"/projects/{project_id}/seats/{seat}/continuity", q=query))


def participant_context(args: argparse.Namespace, project_id: str, participant: str, query: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(participant, safe="")
    return get_json(query_url(args.pnp_url, f"/projects/{project_id}/participants/{encoded}", q=query))


def run_exercise(args: argparse.Namespace) -> dict[str, Any]:
    service_checks = check_services(args)
    pnp_auth = pnp_headers()

    mystro_project = create_or_select_mystro_project(args)
    selected_mystro_project = mystro_project["project"]
    project_id = args.project_id or selected_mystro_project["pnpProjectId"]

    mystro_room_before = get_json(url_join(args.mystro_url, "/api/room"))
    assert_true(
        mystro_room_before.get("activeProject", {}).get("pnpProjectId") == project_id,
        "Mystro active project and PNP smoke project id diverged.",
    )
    last_thoughts = get_json(url_join(args.mystro_url, "/api/resume/last-thoughts"))
    assert_true(last_thoughts.get("summary", {}).get("pnpProjectId") == project_id, "Last-thoughts package did not use selected project id.")
    post_json(
        url_join(args.mystro_url, "/api/stenographer/notes"),
        {
            "kind": "live_pnp_sync_smoke",
            "summary": f"Starting live PNP sync smoke for {project_id}.",
            "details": "Three participant identities are being bound to Mystro seats with Ollama/Qwen and PNP continuity lanes.",
            "seatId": "stenographer",
            "touchedFiles": ["project-room/seat-registry/seats.json"],
        },
        headers=mystro_headers(),
    )

    post_json(
        url_join(args.pnp_url, "/projects"),
        {"project_id": project_id, "title": "Mystro Table live PNP sync smoke"},
        headers=pnp_auth,
    )
    summary = (
        "Live Mystro Table PNP sync smoke. The project uses participant identity as the continuity key, "
        "seat ids only as temporary routing, and qwen2.5-coder:7b as the local Ollama sync model for this run."
    )
    history_package = "\n".join(
        [
            f"Mystro health: {compact_json(service_checks['mystro_health'])}",
            f"Last thoughts keys: {', '.join(sorted(last_thoughts.keys())) if isinstance(last_thoughts, dict) else 'unknown'}",
            "Participants:",
            *[
                f"- {p['participant_identity']} on {p['seat']} with {QWEN_MODEL}"
                for p in DEFAULT_PARTICIPANTS
            ],
        ]
    )
    post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/stenographer/summary"),
        {
            "summary": summary,
            "history_package": history_package,
            "source": "mystro-stenographer-live-smoke",
            "confidence": 0.95,
        },
        headers=pnp_auth,
    )
    post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/state"),
        {
            "state": {
                "source": "live_mystro_table_sync_smoke",
                "local_model": QWEN_MODEL,
                "continuity_key": "project_id + participant_identity",
            }
        },
        headers=pnp_auth,
    )

    mystro_seats = {}
    pnp_bindings = {}
    pnp_chats = {}
    for participant in DEFAULT_PARTICIPANTS:
        mystro_seats[participant["seat"]] = fill_mystro_seat(args, participant)
        bind = bind_pnp_seat(args, project_id, participant)
        assert_true(bind["lane_existed"] is False, f"{participant['participant_identity']} unexpectedly existed before first bind.")
        assert_true(bind["hydration"]["source"] == "stenographer_summary", "First-time participant did not hydrate from stenographer summary.")
        pnp_bindings[participant["seat"]] = bind
        pnp_chats[participant["seat"]] = project_chat(
            args,
            project_id,
            participant["seat"],
            f"Give one sentence confirming your participant lane is {participant['participant_identity']} and not owned by the chair.",
        )

    private_phrase = f"alpha private continuity marker {int(time.time())}"
    alpha = DEFAULT_PARTICIPANTS[0]
    post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/participants/{alpha['participant_identity']}/memory"),
        {
            "content": private_phrase,
            "summary": private_phrase,
            "tags": ["private_lane_marker", "live_smoke"],
            "salience": 0.95,
        },
        headers=pnp_auth,
    )
    post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/participants/{alpha['participant_identity']}/feedback"),
        {
            "content": "Prefer participant identity over seat identity when reconnecting a Mystro Table lane.",
            "feedback": 0.8,
            "domain": "mystro_continuity",
            "confidence": 0.8,
        },
        headers=pnp_auth,
    )

    alpha_moved = dict(alpha)
    alpha_moved["seat"] = "seat-8"
    move_response = bind_pnp_seat(args, project_id, alpha_moved)
    assert_true(move_response["lane_existed"] is True, "Moved participant did not reconnect to the existing lane.")
    assert_true("seat-5" in move_response["moved_from"], "Seat move did not remove the old seat binding.")
    moved_context = continuity_context(args, project_id, "seat-8", private_phrase)
    assert_true(private_phrase in compact_json(moved_context), "Moved participant continuity did not contain the private marker.")

    replacement = dict(REPLACEMENT)
    fill_mystro_seat(args, replacement)
    replacement_bind = bind_pnp_seat(args, project_id, replacement)
    assert_true(replacement_bind["lane_existed"] is False, "Replacement participant unexpectedly reused an existing lane.")
    replacement_context = continuity_context(args, project_id, "seat-5", private_phrase)
    replacement_text = compact_json(replacement_context)
    assert_true(replacement_context["participant_identity"] == replacement["participant_identity"], "Replacement seat bound to the wrong participant.")
    assert_true(private_phrase not in replacement_text, "Replacement participant inherited the prior occupant private marker.")
    assert_true(replacement_bind["hydration"]["source"] == "stenographer_summary", "Replacement did not hydrate from project history.")

    tool = post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/toolbox/tools"),
        {
            "name": "mystro-pnp-live-sync-smoke",
            "description": "Reusable live smoke for Mystro Table chair routing into PNP participant continuity lanes.",
            "tool_type": "smoke",
            "command": "python scripts/smoke_mystro_table_sync.py",
            "path": "scripts/smoke_mystro_table_sync.py",
            "source": "live-smoke",
            "created_by": alpha["participant_identity"],
            "allowed_participants": [p["participant_identity"] for p in DEFAULT_PARTICIPANTS] + [replacement["participant_identity"]],
            "metadata": {"local_model": QWEN_MODEL, "mystro_url": args.mystro_url},
            "useful": True,
        },
        headers=pnp_auth,
    )
    toolbox = get_json(url_join(args.pnp_url, f"/projects/{project_id}/toolbox"))
    assert_true(any(item["id"] == tool["id"] for item in toolbox["tools"]), "Project toolbox did not retain the new smoke tool.")

    lesson_phrase = "Mystro seat ids are routing only; PNP sync lanes must follow participant identity."
    lesson = post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/lessons"),
        {
            "content": lesson_phrase,
            "source": "live-mystro-pnp-sync-smoke",
            "confidence": 0.95,
            "tags": ["mystro", "seat-binding", "participant-continuity"],
        },
        headers=pnp_auth,
    )
    lessons = get_json(query_url(args.pnp_url, f"/projects/{project_id}/lessons", q="seat participant continuity"))
    assert_true(any(item["id"] == lesson["id"] for item in lessons["lessons"]), "Lesson retrieval did not return the recorded lesson.")
    alpha_context_for_lesson = participant_context(args, project_id, alpha["participant_identity"], lesson_phrase)
    assert_true(lesson_phrase not in compact_json(alpha_context_for_lesson.get("context", "")), "Project lesson polluted participant-only context.")

    mystro_dispatch = create_and_dispatch_mystro_task(args, project_id, "seat-5")

    archive = post_json(url_join(args.pnp_url, f"/projects/{project_id}/archive"), {}, headers=pnp_auth, timeout=90)
    archive_path = Path(archive["archive_path"])
    assert_true(archive_path.exists(), f"Archive was not written: {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
    expected = {"project.json", "participants.json", "seats.json", "toolbox.json", "lessons.json"}
    assert_true(expected.issubset(names), f"Archive missing required files: {sorted(expected - names)}")
    assert_true(any(name.startswith("participants/") for name in names), "Archive did not include participant lane directories.")
    restore = post_json(
        url_join(args.pnp_url, f"/projects/{project_id}/restore"),
        {"archive_path": str(archive_path), "overwrite": True},
        headers=pnp_auth,
        timeout=90,
    )
    restored_project = get_json(url_join(args.pnp_url, f"/projects/{project_id}"))
    assert_true(restored_project["toolbox_count"] >= 1, "Restored project lost toolbox entries.")
    assert_true(restored_project["lesson_count"] >= 1, "Restored project lost lessons.")
    assert_true(alpha["participant_identity"] in restored_project["participants"], "Restored project lost alpha participant lane.")

    result = {
        "ok": True,
        "mode": "exercise",
        "project_id": project_id,
        "mystro_project": {
            "id": selected_mystro_project["id"],
            "name": selected_mystro_project["name"],
            "pnpProjectId": selected_mystro_project["pnpProjectId"],
            "created_by_smoke": mystro_project["created"],
        },
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": service_checks,
        "participants": DEFAULT_PARTICIPANTS,
        "replacement": replacement,
        "private_marker": private_phrase,
        "mystro_room_before_seat_count": len(mystro_room_before.get("seats", [])),
        "mystro_bound_seats": {
            seat: {
                "provider": packet["seat"]["occupant"]["providerId"],
                "status": packet["seat"]["occupant"]["connectionStatus"],
                "label": packet["seat"]["occupant"]["instanceLabel"],
            }
            for seat, packet in mystro_seats.items()
        },
        "pnp_chat_models": {
            seat: {"provider": chat["provider"], "model": chat["model"], "latency_ms": chat["latency_ms"]}
            for seat, chat in pnp_chats.items()
        },
        "seat_move": {
            "moved_participant": alpha["participant_identity"],
            "from": "seat-5",
            "to": "seat-8",
            "moved_from": move_response["moved_from"],
        },
        "replacement_isolation": {
            "seat": "seat-5",
            "participant_identity": replacement["participant_identity"],
            "private_marker_absent": private_phrase not in replacement_text,
        },
        "toolbox_tool_id": tool["id"],
        "lesson_id": lesson["id"],
        "mystro_dispatch": {
            "task_id": mystro_dispatch["task"]["id"],
            "delivered": mystro_dispatch["dispatch"]["delivered"],
            "model": mystro_dispatch["dispatch"]["model"],
            "status": mystro_dispatch["dispatch"]["status"],
        },
        "archive": {
            "path": str(archive_path),
            "contains_required_project_layers": True,
            "restore": restore,
        },
    }
    write_result(args.out, result)
    return result


def run_verify_existing(args: argparse.Namespace) -> dict[str, Any]:
    result = load_result(args.out)
    project_id = result["project_id"]
    mystro_project = result.get("mystro_project", {})
    if mystro_project.get("id"):
        selected = select_mystro_project(args, mystro_project["id"])
        assert_true(selected["project"]["pnpProjectId"] == project_id, "Mystro restart verification selected the wrong PNP project id.")
    alpha = result["participants"][0]
    service_checks = check_services(args)
    bind = bind_pnp_seat(args, project_id, {**alpha, "seat": "seat-8"})
    assert_true(bind["lane_existed"] is True, "Returning participant did not reconnect to the saved lane after restart.")
    context = continuity_context(args, project_id, "seat-8", result["private_marker"])
    assert_true(result["private_marker"] in compact_json(context), "Returning participant lost private continuity marker after restart.")
    chat = project_chat(
        args,
        project_id,
        "seat-8",
        "Return one short sentence confirming this participant lane survived the PNP restart.",
    )
    toolbox = get_json(url_join(args.pnp_url, f"/projects/{project_id}/toolbox"))
    lessons = get_json(query_url(args.pnp_url, f"/projects/{project_id}/lessons", q="seat participant continuity"))
    assert_true(any(item["id"] == result["toolbox_tool_id"] for item in toolbox["tools"]), "Toolbox tool missing after restart.")
    assert_true(any(item["id"] == result["lesson_id"] for item in lessons["lessons"]), "Lesson missing after restart.")
    result["restart_verification"] = {
        "ok": True,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": service_checks,
        "returning_bind_lane_existed": bind["lane_existed"],
        "chat_model": chat["model"],
        "chat_provider": chat["provider"],
    }
    write_result(args.out, result)
    return result["restart_verification"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Mystro Table + PNP project continuity sync smoke.")
    parser.add_argument("--pnp-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mystro-url", default="http://127.0.0.1:8787")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--use-active-mystro-project", action="store_true", help="Use Mystro's current active project instead of creating a named smoke project.")
    parser.add_argument("--out", type=Path, default=Path(".smoke") / "mystro_table_sync" / "result.json")
    parser.add_argument("--verify-existing", action="store_true", help="Verify an existing project result after a PNP restart.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = run_verify_existing(args) if args.verify_existing else run_exercise(args)
    except SmokeFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
