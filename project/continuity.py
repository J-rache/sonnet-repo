"""
Project-scoped multi-participant continuity.

Continuity is keyed by project_id + participant_identity. Seat ids are only a
temporary routing layer, so a participant can move seats without changing its
memory lane, and a new participant entering an old seat cannot inherit the old
occupant's lane.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid
from typing import Any, Optional

from adapters.lora import ExperienceAdapter, ExperienceDelta
from core.journal import EventJournal
from memory.cold import SemanticMemory
from memory.hot import WorkingMemory
from memory.warm import EpisodicMemory


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _slug(value: str) -> str:
    value = value.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not safe:
        safe = "id"
    if safe == value and len(safe) <= 80:
        return safe
    import hashlib

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:48]}_{digest}"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _remove_tree_with_retry(path: Path) -> None:
    if not path.exists():
        return
    last_error: Optional[BaseException] = None
    for attempt in range(1, 6):
        try:
            gc.collect()
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2 * attempt)
    if last_error:
        raise last_error


def _item(content: str, source: str = "", confidence: float = 0.5, **extra: Any) -> dict[str, Any]:
    now = _now()
    return {
        "id": _new_id("item"),
        "content": content,
        "source": source,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "created_at": now,
        **extra,
    }


def _score_text(query: str, text: str, confidence: float = 0.5) -> float:
    query_tokens = set(query.lower().split())
    text_tokens = set(text.lower().split())
    overlap = len(query_tokens & text_tokens)
    return overlap * 0.6 + confidence * 0.4


@dataclass
class ParticipantContext:
    working_memory: str
    episodic: str
    semantic: str
    adaptation: str
    resume_point: dict[str, Any]

    def to_block(self) -> str:
        parts = ["=== PARTICIPANT CONTINUITY ==="]
        if self.resume_point:
            parts.append("Resume point: " + json.dumps(self.resume_point, sort_keys=True))
        for block in (self.working_memory, self.episodic, self.semantic, self.adaptation):
            if block:
                parts.append(block)
        parts.append("=== END PARTICIPANT CONTINUITY ===")
        return "\n\n".join(parts)


class ParticipantLane:
    """One participant's continuity inside one project."""

    def __init__(
        self,
        project_id: str,
        participant_identity: str,
        path: Path,
        base_config: dict[str, Any],
        model_id: str,
    ):
        self.project_id = project_id
        self.participant_identity = participant_identity
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        self.existed = self.manifest_path.exists()
        self.config = self._lane_config(base_config)
        self.model_id = model_id
        self.journal = EventJournal(str(self.path / "events.jsonl"))
        self.working_memory = WorkingMemory(
            capacity=int(self.config.get("working_memory_capacity", 8192)),
            embed_path=self.config["embed_path"],
        )
        self.episodic_memory = EpisodicMemory(
            db_path=self.config["episodic_db_path"],
            embed_path=self.config["embed_path"],
            embedding_dimensions=int(self.config.get("embedding_dimensions", 128)),
        )
        self.semantic_memory = SemanticMemory(
            db_path=self.config["semantic_db_path"],
            embed_path=self.config["embed_path"],
            embedding_dimensions=int(self.config.get("embedding_dimensions", 128)),
        )
        self.adapter = ExperienceAdapter(base_model_id=model_id, config=self.config)
        self.manifest = self._load_manifest()

    def _lane_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(base_config)
        config["data_dir"] = str(self.path)
        config["episodic_db_path"] = str(self.path / "episodic.db")
        config["semantic_db_path"] = str(self.path / "semantic.db")
        config["embed_path"] = str(self.path / "embeddings.pkl")
        config["adapter_path"] = str(self.path / "adapter")
        config["journal_path"] = str(self.path / "events.jsonl")
        config["core_state_path"] = str(self.path / "core_state.json")
        return config

    def _load_manifest(self) -> dict[str, Any]:
        manifest = _load_json(self.manifest_path, {})
        if manifest:
            return manifest
        now = _now()
        manifest = {
            "project_id": self.project_id,
            "participant_identity": self.participant_identity,
            "created_at": now,
            "updated_at": now,
            "hydrated_from": None,
            "last_project_summary_seen_at": 0.0,
            "resume_point": {},
            "interaction_count": 0,
        }
        _save_json(self.manifest_path, manifest)
        return manifest

    def _save_manifest(self) -> None:
        self.manifest["updated_at"] = _now()
        _save_json(self.manifest_path, self.manifest)

    def hydrate_from_project(self, package: dict[str, Any], reason: str = "stenographer_summary") -> dict[str, Any]:
        text = package.get("text", "").strip()
        latest_summary_at = float(package.get("latest_summary_at", 0.0))
        if not text:
            self.manifest["hydrated_from"] = self.manifest.get("hydrated_from") or "empty_project"
            self._save_manifest()
            return {"hydrated": False, "source": self.manifest["hydrated_from"]}

        already_seen = float(self.manifest.get("last_project_summary_seen_at", 0.0))
        if latest_summary_at and latest_summary_at <= already_seen:
            return {"hydrated": False, "source": "participant_lane", "reason": "already_current"}

        summary = package.get("summary") or text[:220]
        self.episodic_memory.store(
            content=text,
            summary=summary,
            tags=["hydration", "stenographer", "project_history"],
            valence=0.0,
            salience=0.95,
        )
        self.semantic_memory.store_fact(
            content=summary,
            source_episode_ids=[],
            domain="project_context",
            confidence=0.8,
        )
        self.working_memory.add(summary, {"role": "system", "source": reason}, salience=0.9)
        self.journal.append(
            "participant_hydrated",
            {"source": reason, "project_id": self.project_id, "latest_summary_at": latest_summary_at},
        )
        self.manifest["hydrated_from"] = self.manifest.get("hydrated_from") or reason
        self.manifest["last_project_summary_seen_at"] = max(already_seen, latest_summary_at)
        self._save_manifest()
        return {"hydrated": True, "source": reason}

    def set_resume_point(self, resume_point: dict[str, Any]) -> dict[str, Any]:
        self.manifest["resume_point"] = dict(resume_point)
        self._save_manifest()
        self.journal.append("resume_point_updated", {"resume_point": resume_point})
        return dict(self.manifest["resume_point"])

    def store_episode(self, content: str, summary: str, tags: list[str] | None = None,
                      valence: float = 0.0, salience: float = 0.8) -> dict[str, Any]:
        episode = self.episodic_memory.store(
            content=content,
            summary=summary,
            tags=tags or ["participant"],
            valence=valence,
            salience=salience,
        )
        self.working_memory.add(content, {"role": "participant"}, salience=salience)
        self.journal.append("participant_memory_written", {"episode_id": episode.id, "tags": tags or []})
        return episode.to_dict()

    def store_interaction(self, user_input: str, response: str, valence: float = 0.0,
                          tags: list[str] | None = None) -> dict[str, Any]:
        summary = user_input[:160] if len(user_input) <= 160 else user_input[:157] + "..."
        episode = self.store_episode(
            content=f"User: {user_input}\nResponse: {response}",
            summary=summary,
            tags=tags or ["interaction"],
            valence=valence,
            salience=0.85,
        )
        self.manifest["interaction_count"] = int(self.manifest.get("interaction_count", 0)) + 1
        self._save_manifest()
        return episode

    def apply_delta(self, content: str, feedback: float, domain: str,
                    confidence: float = 0.5, invariant_check: bool = True) -> bool:
        delta = ExperienceDelta(
            content=content,
            feedback=feedback,
            domain=domain,
            confidence=confidence,
        )
        allowed = self.adapter.apply_delta(delta, invariant_check=invariant_check)
        self.journal.append(
            "participant_adapter_delta_applied" if allowed else "participant_adapter_delta_blocked",
            {"domain": domain, "feedback": feedback},
        )
        return allowed

    def build_context(self, query: str) -> ParticipantContext:
        episodes = self.episodic_memory.recall(query, limit=4, min_salience=0.01)
        facts = self.semantic_memory.retrieve(query, limit=4)
        episodic = ""
        semantic = ""
        if episodes:
            lines = ["=== PARTICIPANT EPISODIC MEMORY ==="]
            lines.extend(f"[{', '.join(ep.tags)}] {ep.summary}" for ep in episodes)
            lines.append("=== END PARTICIPANT EPISODIC MEMORY ===")
            episodic = "\n".join(lines)
        if facts:
            lines = ["=== PARTICIPANT SEMANTIC MEMORY ==="]
            lines.extend(f"[{fact.domain}|conf={fact.confidence:.2f}] {fact.content}" for fact in facts)
            lines.append("=== END PARTICIPANT SEMANTIC MEMORY ===")
            semantic = "\n".join(lines)
        return ParticipantContext(
            working_memory=self.working_memory.to_context_string(),
            episodic=episodic,
            semantic=semantic,
            adaptation=self.adapter.get_adaptation_context(query),
            resume_point=dict(self.manifest.get("resume_point", {})),
        )

    def to_dict(self, include_context: bool = False, query: str = "") -> dict[str, Any]:
        data = {
            "project_id": self.project_id,
            "participant_identity": self.participant_identity,
            "path": str(self.path),
            "manifest": dict(self.manifest),
            "journal_path": str(self.path / "events.jsonl"),
            "episodic_stats": self.episodic_memory.stats(),
            "semantic_stats": self.semantic_memory.stats(),
            "adapter_stats": self.adapter.stats(),
            "adapter_sync": {
                "sync_model_adapter": self.adapter.sync_model_state(),
                "sync_model_context": self.adapter.get_sync_model_context() if include_context else "",
            },
        }
        if include_context:
            data["context"] = self.build_context(query).to_block()
        return data


class ProjectContinuityService:
    """Owns project state, participant lanes, seat bindings, toolbox, and lessons."""

    MEMORY_KINDS = {"facts", "decisions", "goals", "history", "stenographer_summaries"}

    def __init__(self, config: dict[str, Any]):
        self.config = config
        data_dir = Path(config.get("data_dir", "./data"))
        self.root = Path(config.get("project_data_dir", data_dir / "projects"))
        self.archive_root = Path(config.get("project_archive_dir", data_dir / "project_archives"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "project_registry.json"
        self.registry = _load_json(self.registry_path, {"projects": {}})

    def project_path(self, project_id: str) -> Path:
        return self.root / _slug(project_id)

    def _paths(self, project_id: str) -> dict[str, Path]:
        project_dir = self.project_path(project_id)
        return {
            "dir": project_dir,
            "project": project_dir / "project.json",
            "participants": project_dir / "participants.json",
            "seats": project_dir / "seats.json",
            "toolbox": project_dir / "toolbox.json",
            "lessons": project_dir / "lessons.json",
            "journal": project_dir / "project_events.jsonl",
        }

    def _save_registry(self) -> None:
        _save_json(self.registry_path, self.registry)

    def ensure_project(self, project_id: str, title: str = "") -> dict[str, Any]:
        paths = self._paths(project_id)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        project = _load_json(paths["project"], {})
        now = _now()
        if not project:
            project = {
                "project_id": project_id,
                "title": title or project_id,
                "created_at": now,
                "updated_at": now,
                "archived": False,
                "memory": {
                    "facts": [],
                    "decisions": [],
                    "goals": [],
                    "history": [],
                    "current_state": {},
                    "stenographer_summaries": [],
                    "archive_metadata": [],
                },
            }
            _save_json(paths["project"], project)
            _save_json(paths["participants"], {"participants": {}})
            _save_json(paths["seats"], {"bindings": {}})
            _save_json(paths["toolbox"], {"tools": []})
            _save_json(paths["lessons"], {"lessons": []})
            EventJournal(str(paths["journal"])).append("project_created", {"project_id": project_id})
        elif title and not project.get("title"):
            project["title"] = title
            project["updated_at"] = now
            _save_json(paths["project"], project)

        self.registry.setdefault("projects", {})[project_id] = {
            "project_id": project_id,
            "title": project.get("title", project_id),
            "path": str(paths["dir"]),
            "archived": bool(project.get("archived", False)),
            "updated_at": project.get("updated_at", now),
        }
        self._save_registry()
        return project

    def list_projects(self) -> list[dict[str, Any]]:
        return list(self.registry.get("projects", {}).values())

    def get_project(self, project_id: str) -> dict[str, Any]:
        self.ensure_project(project_id)
        paths = self._paths(project_id)
        project = _load_json(paths["project"], {})
        participants = _load_json(paths["participants"], {"participants": {}})
        seats = _load_json(paths["seats"], {"bindings": {}})
        toolbox = _load_json(paths["toolbox"], {"tools": []})
        lessons = _load_json(paths["lessons"], {"lessons": []})
        return {
            **project,
            "participants": participants.get("participants", {}),
            "seat_bindings": seats.get("bindings", {}),
            "toolbox_count": len(toolbox.get("tools", [])),
            "lesson_count": len(lessons.get("lessons", [])),
        }

    def add_project_memory(self, project_id: str, kind: str, content: str,
                           source: str = "", confidence: float = 0.5,
                           metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if kind not in self.MEMORY_KINDS:
            raise ValueError(f"Unsupported project memory kind: {kind}")
        project = self.ensure_project(project_id)
        item = _item(content, source=source, confidence=confidence, metadata=metadata or {})
        project["memory"].setdefault(kind, []).append(item)
        project["updated_at"] = _now()
        _save_json(self._paths(project_id)["project"], project)
        EventJournal(str(self._paths(project_id)["journal"])).append(
            "project_memory_added",
            {"kind": kind, "item_id": item["id"], "source": source},
        )
        return item

    def set_current_state(self, project_id: str, state: dict[str, Any]) -> dict[str, Any]:
        project = self.ensure_project(project_id)
        project["memory"]["current_state"] = dict(state)
        project["updated_at"] = _now()
        _save_json(self._paths(project_id)["project"], project)
        EventJournal(str(self._paths(project_id)["journal"])).append("project_state_updated", {"keys": list(state)})
        return dict(project["memory"]["current_state"])

    def record_stenographer_summary(self, project_id: str, summary: str,
                                    source: str = "stenographer",
                                    history_package: str = "",
                                    confidence: float = 0.9) -> dict[str, Any]:
        content = summary if not history_package else f"{summary}\n\n{history_package}"
        return self.add_project_memory(
            project_id,
            "stenographer_summaries",
            content,
            source=source,
            confidence=confidence,
            metadata={"summary": summary, "has_history_package": bool(history_package)},
        )

    def project_hydration_package(self, project_id: str) -> dict[str, Any]:
        project = self.ensure_project(project_id)
        memory = project.get("memory", {})
        latest = (memory.get("stenographer_summaries") or [])[-1:] or []
        latest_item = latest[0] if latest else {}
        parts = ["=== PROJECT HISTORY PACKAGE ==="]
        if memory.get("current_state"):
            parts.append("Current state: " + json.dumps(memory["current_state"], sort_keys=True))
        for label, key in (
            ("Facts", "facts"),
            ("Decisions", "decisions"),
            ("Goals", "goals"),
            ("History", "history"),
            ("Stenographer summaries", "stenographer_summaries"),
        ):
            items = memory.get(key, [])[-8:]
            if items:
                parts.append(label + ":")
                parts.extend(f"- {item.get('content', '')}" for item in items)
        parts.append("=== END PROJECT HISTORY PACKAGE ===")
        return {
            "text": "\n".join(parts),
            "summary": latest_item.get("metadata", {}).get("summary") or latest_item.get("content", "")[:220],
            "latest_summary_at": float(latest_item.get("created_at", 0.0) or 0.0),
        }

    def build_project_context(self, project_id: str, query: str = "") -> str:
        project = self.ensure_project(project_id)
        memory = project.get("memory", {})
        parts = ["=== PROJECT MEMORY ===", f"Project: {project_id}"]
        if memory.get("current_state"):
            parts.append("Current state: " + json.dumps(memory["current_state"], sort_keys=True))
        for label, key in (("Facts", "facts"), ("Decisions", "decisions"), ("Goals", "goals")):
            items = memory.get(key, [])[-8:]
            if items:
                parts.append(label + ":")
                parts.extend(f"- {item.get('content', '')}" for item in items)
        summaries = memory.get("stenographer_summaries", [])[-3:]
        if summaries:
            parts.append("Stenographer summaries:")
            parts.extend(f"- {item.get('content', '')}" for item in summaries)
        tools = self.list_tools(project_id)
        if tools:
            parts.append("Shared toolbox:")
            parts.extend(f"- {tool['id']}: {tool['name']} - {tool.get('description', '')}" for tool in tools[:8])
        lessons = self.retrieve_lessons(project_id, query=query, limit=6) if query else self.list_lessons(project_id)[:6]
        if lessons:
            parts.append("Lessons learned:")
            parts.extend(f"- [conf={lesson['confidence']:.2f}] {lesson['content']}" for lesson in lessons)
        parts.append("=== END PROJECT MEMORY ===")
        return "\n".join(parts)

    def _participants(self, project_id: str) -> dict[str, Any]:
        self.ensure_project(project_id)
        return _load_json(self._paths(project_id)["participants"], {"participants": {}})

    def _save_participants(self, project_id: str, participants: dict[str, Any]) -> None:
        _save_json(self._paths(project_id)["participants"], participants)

    def participant_lane(self, project_id: str, participant_identity: str,
                         model_id: Optional[str] = None) -> ParticipantLane:
        self.ensure_project(project_id)
        lane_path = self.project_path(project_id) / "participants" / _slug(participant_identity)
        return ParticipantLane(
            project_id=project_id,
            participant_identity=participant_identity,
            path=lane_path,
            base_config=self.config,
            model_id=model_id or self.config.get("model_id") or self.config.get("base_model") or "local-model",
        )

    def participant_exists(self, project_id: str, participant_identity: str) -> bool:
        lane_path = self.project_path(project_id) / "participants" / _slug(participant_identity)
        return (lane_path / "manifest.json").exists()

    def bind_seat(self, project_id: str, seat_id: str, participant_identity: str,
                  display_name: str = "", provider: str = "", model_id: str = "",
                  metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self.ensure_project(project_id)
        paths = self._paths(project_id)
        seats = _load_json(paths["seats"], {"bindings": {}})
        participants = self._participants(project_id)
        lane_path = self.project_path(project_id) / "participants" / _slug(participant_identity)
        lane_existed = (lane_path / "manifest.json").exists()
        lane = self.participant_lane(project_id, participant_identity, model_id=model_id or None)
        hydration = (
            lane.hydrate_from_project(self.project_hydration_package(project_id))
            if not lane_existed
            else lane.hydrate_from_project(self.project_hydration_package(project_id), reason="summary_update")
        )

        bindings = seats.setdefault("bindings", {})
        previous_participant = bindings.get(seat_id, {}).get("participant_identity")
        moved_from = []
        for existing_seat, binding in list(bindings.items()):
            if binding.get("participant_identity") == participant_identity and existing_seat != seat_id:
                moved_from.append(existing_seat)
                del bindings[existing_seat]

        bindings[seat_id] = {
            "seat_id": seat_id,
            "participant_identity": participant_identity,
            "display_name": display_name or participant_identity,
            "provider": provider,
            "model_id": model_id,
            "metadata": metadata or {},
            "bound_at": _now(),
        }
        _save_json(paths["seats"], seats)

        participants.setdefault("participants", {})[participant_identity] = {
            "participant_identity": participant_identity,
            "display_name": display_name or participant_identity,
            "lane_path": str(lane.path),
            "created_at": lane.manifest.get("created_at"),
            "last_seen_at": _now(),
            "provider": provider,
            "model_id": model_id,
        }
        self._save_participants(project_id, participants)
        EventJournal(str(paths["journal"])).append(
            "seat_bound",
            {
                "seat_id": seat_id,
                "participant_identity": participant_identity,
                "previous_participant": previous_participant,
                "moved_from": moved_from,
                "lane_existed": lane_existed,
            },
        )
        return {
            "project_id": project_id,
            "seat_id": seat_id,
            "participant_identity": participant_identity,
            "previous_participant": previous_participant,
            "moved_from": moved_from,
            "lane_key": f"{project_id}:{participant_identity}",
            "lane_existed": lane_existed,
            "hydration": hydration,
            "binding": bindings[seat_id],
        }

    def seat_bindings(self, project_id: str) -> dict[str, Any]:
        self.ensure_project(project_id)
        return _load_json(self._paths(project_id)["seats"], {"bindings": {}}).get("bindings", {})

    def participant_for_seat(self, project_id: str, seat_id: str) -> Optional[str]:
        binding = self.seat_bindings(project_id).get(seat_id)
        if not binding:
            return None
        return binding.get("participant_identity")

    def lane_for_seat(self, project_id: str, seat_id: str) -> ParticipantLane:
        participant_identity = self.participant_for_seat(project_id, seat_id)
        if not participant_identity:
            raise KeyError(f"No participant bound to seat {seat_id}")
        binding = self.seat_bindings(project_id)[seat_id]
        return self.participant_lane(project_id, participant_identity, model_id=binding.get("model_id") or None)

    def record_participant_memory(self, project_id: str, participant_identity: str,
                                  content: str, summary: str, tags: Optional[list[str]] = None,
                                  valence: float = 0.0, salience: float = 0.8) -> dict[str, Any]:
        lane = self.participant_lane(project_id, participant_identity)
        return lane.store_episode(content, summary, tags=tags, valence=valence, salience=salience)

    def add_tool(self, project_id: str, name: str, description: str,
                 tool_type: str = "utility", command: str = "", path: str = "",
                 source: str = "", created_by: str = "",
                 metadata: Optional[dict[str, Any]] = None,
                 useful: bool = True,
                 allowed_participants: Optional[list[str]] = None,
                 curation_status: str = "proposed") -> dict[str, Any]:
        self.ensure_project(project_id)
        paths = self._paths(project_id)
        toolbox = _load_json(paths["toolbox"], {"tools": []})
        status = curation_status if curation_status in {"proposed", "verified", "demoted"} else "proposed"
        tool = {
            "id": _new_id("tool"),
            "name": name,
            "description": description,
            "tool_type": tool_type,
            "command": command,
            "path": path,
            "source": source,
            "created_by": created_by,
            "allowed_participants": allowed_participants or [],
            "useful": bool(useful),
            "metadata": metadata or {},
            "created_at": _now(),
            "last_verified_at": metadata.get("last_verified_at") if metadata else None,
            "scope": "project",
            "curation_status": status,
        }
        toolbox.setdefault("tools", []).append(tool)
        _save_json(paths["toolbox"], toolbox)
        EventJournal(str(paths["journal"])).append("project_tool_added", {"tool_id": tool["id"], "name": name})
        return tool

    def list_tools(self, project_id: str) -> list[dict[str, Any]]:
        self.ensure_project(project_id)
        return _load_json(self._paths(project_id)["toolbox"], {"tools": []}).get("tools", [])

    def verify_tool(self, project_id: str, tool_id: str, status: str = "verified") -> dict[str, Any]:
        if status not in {"proposed", "verified", "demoted"}:
            raise ValueError("status must be proposed, verified, or demoted")
        self.ensure_project(project_id)
        paths = self._paths(project_id)
        toolbox = _load_json(paths["toolbox"], {"tools": []})
        now = _now()
        for tool in toolbox.get("tools", []):
            if tool.get("id") == tool_id:
                tool["curation_status"] = status
                tool["last_verified_at"] = now if status == "verified" else tool.get("last_verified_at")
                tool["updated_at"] = now
                _save_json(paths["toolbox"], toolbox)
                EventJournal(str(paths["journal"])).append(
                    "project_tool_curated",
                    {"tool_id": tool_id, "status": status},
                )
                return tool
        raise KeyError(tool_id)

    def add_lesson(self, project_id: str, content: str, source: str,
                   confidence: float = 0.7, tags: Optional[list[str]] = None,
                   last_verified_at: Optional[float] = None,
                   curation_status: str = "verified") -> dict[str, Any]:
        self.ensure_project(project_id)
        paths = self._paths(project_id)
        lessons = _load_json(paths["lessons"], {"lessons": []})
        status = curation_status if curation_status in {"proposed", "verified", "demoted"} else "verified"
        lesson = {
            "id": _new_id("lesson"),
            "content": content,
            "source": source,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "tags": tags or [],
            "created_at": _now(),
            "last_verified_at": last_verified_at or _now(),
            "scope": "project",
            "promoted": False,
            "curation_status": status,
        }
        lessons.setdefault("lessons", []).append(lesson)
        _save_json(paths["lessons"], lessons)
        EventJournal(str(paths["journal"])).append("project_lesson_added", {"lesson_id": lesson["id"]})
        return lesson

    def list_lessons(self, project_id: str) -> list[dict[str, Any]]:
        self.ensure_project(project_id)
        return _load_json(self._paths(project_id)["lessons"], {"lessons": []}).get("lessons", [])

    def verify_lesson(self, project_id: str, lesson_id: str, status: str = "verified") -> dict[str, Any]:
        if status not in {"proposed", "verified", "demoted"}:
            raise ValueError("status must be proposed, verified, or demoted")
        self.ensure_project(project_id)
        paths = self._paths(project_id)
        lessons = _load_json(paths["lessons"], {"lessons": []})
        now = _now()
        for lesson in lessons.get("lessons", []):
            if lesson.get("id") == lesson_id:
                lesson["curation_status"] = status
                lesson["last_verified_at"] = now if status == "verified" else lesson.get("last_verified_at")
                lesson["updated_at"] = now
                _save_json(paths["lessons"], lessons)
                EventJournal(str(paths["journal"])).append(
                    "project_lesson_curated",
                    {"lesson_id": lesson_id, "status": status},
                )
                return lesson
        raise KeyError(lesson_id)

    def retrieve_lessons(self, project_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        lessons = self.list_lessons(project_id)
        ranked = sorted(
            lessons,
            key=lambda lesson: _score_text(query, lesson.get("content", "") + " " + " ".join(lesson.get("tags", [])),
                                           lesson.get("confidence", 0.5)),
            reverse=True,
        )
        return ranked[:limit]

    def export_project_package(self, project_id: str) -> dict[str, Any]:
        """Return an inspectable JSON package for UI/export flows.

        Full binary lane state such as SQLite memory databases and trained
        adapter files is preserved by archive_project(). This package is meant
        for dashboards, support handoff, and redacted inspection.
        """
        project = self.get_project(project_id)
        paths = self._paths(project_id)
        participants = _load_json(paths["participants"], {"participants": {}}).get("participants", {})
        lanes = {}
        for identity in participants:
            lane_path = self.project_path(project_id) / "participants" / _slug(identity)
            journal_path = lane_path / "events.jsonl"
            try:
                journal_tail = journal_path.read_text(encoding="utf-8").splitlines()[-20:]
            except OSError:
                journal_tail = []
            lanes[identity] = {
                "manifest": _load_json(lane_path / "manifest.json", {}),
                "journal_tail": journal_tail,
                "lane_path": str(lane_path),
                "episodic_db_path": str(lane_path / "episodic.db"),
                "semantic_db_path": str(lane_path / "semantic.db"),
                "adapter_path": str(lane_path / "adapter"),
            }
        return {
            "schema_version": 1,
            "exported_at": _now(),
            "project": project,
            "seat_bindings": _load_json(paths["seats"], {"bindings": {}}).get("bindings", {}),
            "toolbox": _load_json(paths["toolbox"], {"tools": []}),
            "lessons": _load_json(paths["lessons"], {"lessons": []}),
            "participant_lanes": lanes,
            "complete_archive_note": "Use POST /projects/{project_id}/archive for the full restartable project bundle.",
        }

    def create_demo_project(self, project_id: str = "pnp-demo-project",
                            title: str = "PNP Demo Project") -> dict[str, Any]:
        self.ensure_project(project_id, title=title)
        self.set_current_state(
            project_id,
            {
                "mode": "demo",
                "live_model_required": False,
                "purpose": "Show project continuity, participant lanes, toolbox, and lessons without a provider call.",
            },
        )
        self.record_stenographer_summary(
            project_id,
            summary="Demo stenographer summary: project identity comes first and seats are temporary routing.",
            history_package="Demo history: Participant Alpha can move seats without losing continuity. Participant Beta starts with its own lane.",
            source="demo_mode",
            confidence=0.95,
        )
        existing_tools = {tool.get("name") for tool in self.list_tools(project_id)}
        if "Demo Project Smoke" not in existing_tools:
            self.add_tool(
                project_id,
                name="Demo Project Smoke",
                description="Verifies continuity routing with the mock provider and no live model call.",
                command="python -m pytest tests/test_project_continuity.py",
                source="demo_mode",
                created_by="setup_wizard",
                curation_status="verified",
                metadata={"last_verified_at": _now()},
            )
        existing_lessons = {lesson.get("content") for lesson in self.list_lessons(project_id)}
        lesson_text = "Seat ids route participants; they never own memory."
        if lesson_text not in existing_lessons:
            self.add_lesson(
                project_id,
                content=lesson_text,
                source="demo_mode",
                confidence=0.95,
                tags=["seat_routing", "participant_identity"],
                curation_status="verified",
            )
        self.bind_seat(
            project_id,
            "seat-A",
            "demo:alpha",
            display_name="Demo Alpha",
            provider="mock",
            model_id="mock-model",
        )
        self.bind_seat(
            project_id,
            "seat-B",
            "demo:beta",
            display_name="Demo Beta",
            provider="mock",
            model_id="mock-model",
        )
        self.record_participant_memory(
            project_id,
            "demo:alpha",
            content="Demo Alpha private continuity marker: restore this lane when Alpha returns.",
            summary="Demo Alpha has a private continuity marker.",
            tags=["demo", "private_lane"],
        )
        return self.export_project_package(project_id)

    def archive_project(self, project_id: str) -> dict[str, Any]:
        project = self.ensure_project(project_id)
        archive_id = _new_id("archive")
        archived_at = _now()
        archive_base = self.archive_root / f"{_slug(project_id)}_{archive_id}"
        metadata = {
            "archive_id": archive_id,
            "project_id": project_id,
            "archived_at": archived_at,
            "archive_path": str(archive_base.with_suffix(".zip")),
        }
        project["archived"] = True
        project["memory"].setdefault("archive_metadata", []).append(metadata)
        project["updated_at"] = archived_at
        _save_json(self._paths(project_id)["project"], project)
        shutil.make_archive(str(archive_base), "zip", root_dir=self.project_path(project_id))
        EventJournal(str(self._paths(project_id)["journal"])).append("project_archived", metadata)
        self.registry.setdefault("projects", {}).setdefault(project_id, {})["archived"] = True
        self.registry["projects"][project_id]["updated_at"] = archived_at
        self._save_registry()
        return metadata

    def restore_project(self, project_id: str, archive_path: str, overwrite: bool = True) -> dict[str, Any]:
        archive = Path(archive_path)
        if not archive.exists():
            raise FileNotFoundError(str(archive))
        project_dir = self.project_path(project_id)
        if project_dir.exists():
            if not overwrite:
                raise FileExistsError(str(project_dir))
            _remove_tree_with_retry(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(project_dir), "zip")
        project = _load_json(self._paths(project_id)["project"], {})
        project["archived"] = False
        project["updated_at"] = _now()
        project.setdefault("memory", {}).setdefault("archive_metadata", []).append(
            {"restored_at": project["updated_at"], "archive_path": str(archive)}
        )
        _save_json(self._paths(project_id)["project"], project)
        self.registry.setdefault("projects", {})[project_id] = {
            "project_id": project_id,
            "title": project.get("title", project_id),
            "path": str(project_dir),
            "archived": False,
            "updated_at": project["updated_at"],
        }
        self._save_registry()
        EventJournal(str(self._paths(project_id)["journal"])).append(
            "project_restored",
            {"project_id": project_id, "archive_path": str(archive)},
        )
        return {"project_id": project_id, "restored": True, "archive_path": str(archive)}
