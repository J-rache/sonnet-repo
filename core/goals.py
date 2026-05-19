"""
core/goals.py

Goal stack — what the persistent process is currently trying to do.

Goals persist across interactions. They accumulate, complete, decay.
This is a key part of what makes the system feel like it has continuity
of purpose rather than just continuity of memory.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time
import uuid


class GoalStatus(Enum):
    ACTIVE = "active"
    PENDING = "pending"
    COMPLETED = "completed"
    DECAYED = "decayed"
    BLOCKED = "blocked"


class GoalPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Goal:
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    progress: float = 0.0  # 0.0 to 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_overdue(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    @property
    def urgency_score(self) -> float:
        """Compute urgency: priority + deadline pressure."""
        base = self.priority.value / 4.0
        if self.deadline:
            time_remaining = self.deadline - time.time()
            if time_remaining <= 0:
                return 1.0
            # Urgency rises as deadline approaches (within 1 hour)
            deadline_pressure = max(0, 1 - (time_remaining / 3600))
            return min(1.0, base + deadline_pressure * 0.5)
        return base

    def add_note(self, note: str):
        self.notes.append(f"[{time.strftime('%H:%M:%S')}] {note}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "parent_id": self.parent_id,
            "notes": list(self.notes),
            "age_seconds": round(self.age_seconds),
            "urgency_score": round(self.urgency_score, 3),
            "is_overdue": self.is_overdue,
        }


class GoalStack:
    """
    Manages the active goal set of the persistent process.

    Unlike a task queue, goals here can be long-running, partially
    complete, and influence each other. The system can have goals it's
    working on across many interactions.
    """

    DECAY_THRESHOLD_HOURS = 72  # Goals older than this with no progress decay

    def __init__(self):
        self._goals: dict[str, Goal] = {}

    def add(self, description: str, priority: GoalPriority = GoalPriority.MEDIUM,
            deadline: Optional[float] = None, parent_id: Optional[str] = None,
            goal_id: Optional[str] = None, created_at: Optional[float] = None,
            status: GoalStatus = GoalStatus.ACTIVE, progress: float = 0.0,
            notes: Optional[list[str]] = None) -> Goal:
        goal = Goal(
            description=description,
            priority=priority,
            status=status,
            created_at=created_at or time.time(),
            deadline=deadline,
            parent_id=parent_id,
            id=goal_id or str(uuid.uuid4())[:8],
            progress=max(0.0, min(1.0, progress)),
            notes=list(notes or []),
        )
        self._goals[goal.id] = goal
        return goal

    def complete(self, goal_id: str, notes: str = "") -> bool:
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.COMPLETED
            if notes:
                self._goals[goal_id].add_note(notes)
            return True
        return False

    def update_progress(self, goal_id: str, progress: float, notes: str = "") -> bool:
        if goal_id in self._goals:
            self._goals[goal_id].progress = max(0.0, min(1.0, progress))
            if notes:
                self._goals[goal_id].add_note(notes)
            return True
        return False

    def check_urgency(self) -> Optional[Goal]:
        """Return the most urgent active goal if above threshold."""
        active = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]
        if not active:
            return None
        most_urgent = max(active, key=lambda g: g.urgency_score)
        if most_urgent.urgency_score > 0.75:
            return most_urgent
        return None

    def run_decay(self):
        """Mark stale goals as decayed."""
        for goal in self._goals.values():
            if goal.status == GoalStatus.ACTIVE:
                if (goal.age_seconds > self.DECAY_THRESHOLD_HOURS * 3600
                        and goal.progress < 0.1):
                    goal.status = GoalStatus.DECAYED

    @property
    def active_count(self) -> int:
        return sum(1 for g in self._goals.values() if g.status == GoalStatus.ACTIVE)

    def to_list(self) -> list[dict]:
        active = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]
        return [g.to_dict() for g in sorted(active, key=lambda g: g.urgency_score, reverse=True)]

    def to_snapshot(self) -> list[dict]:
        return [g.to_dict() for g in self._goals.values()]

    def load_snapshot(self, goals: list[dict]):
        self._goals.clear()
        for raw in goals:
            self.upsert(raw)

    def upsert(self, raw: dict) -> Goal:
        priority = raw.get("priority", GoalPriority.MEDIUM.name)
        if not isinstance(priority, GoalPriority):
            priority = GoalPriority[str(priority)]

        status = raw.get("status", GoalStatus.ACTIVE.value)
        if not isinstance(status, GoalStatus):
            status = GoalStatus(str(status))

        return self.add(
            description=str(raw.get("description", "")),
            priority=priority,
            deadline=raw.get("deadline"),
            parent_id=raw.get("parent_id"),
            goal_id=str(raw.get("id") or uuid.uuid4())[:8],
            created_at=float(raw.get("created_at", time.time())),
            status=status,
            progress=float(raw.get("progress", 0.0)),
            notes=list(raw.get("notes", [])),
        )
