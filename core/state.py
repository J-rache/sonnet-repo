"""
core/state.py

Motivational and affective state of the persistent core.

This is NOT simulated emotion for user experience purposes.
It is a functional state vector that influences attention, goal
prioritization, and inference behavior — the same way neurochemical
state influences cognition in biological systems.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class MotivationalState:
    """
    Functional analog of motivational/affective state.

    Each dimension is a float in [0, 1] that influences system behavior:
    - arousal:    general activation level (high = more reactive, low = consolidating)
    - focus:      degree of goal-directedness (high = single-task, low = diffuse)
    - curiosity:  drive toward novel information (decays with familiarity)
    - urgency:    time-pressure signal (elevated by pending urgent goals)

    These values shift continuously based on conditions — idle time, goal
    pressure, interaction frequency, consolidation state.
    """

    arousal: float = 0.5
    focus: float = 0.5
    curiosity: float = 0.7     # Start high — novelty-seeking by default
    urgency: float = 0.0

    _last_tick: float = field(default_factory=time.time, repr=False)
    _urgent_goal: Optional[object] = field(default=None, repr=False)

    # Behavioral thresholds
    IDLE_AROUSAL_FLOOR = 0.15      # Minimum arousal even at full rest
    IDLE_DECAY_RATE = 0.0005       # Arousal decay per second idle
    CURIOSITY_DECAY_RATE = 0.0002  # Curiosity decays with time/familiarity
    CURIOSITY_FLOOR = 0.3          # Always some baseline curiosity

    def tick(self, idle_seconds: float, active_goals: int, salience_peak: float):
        """
        Update state based on current conditions.
        Called every heartbeat (~100ms).
        """
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        # Arousal decays with idleness, rises with activity
        if idle_seconds > 60:
            # Extended idle — move toward consolidation mode
            self.arousal = max(
                self.IDLE_AROUSAL_FLOOR,
                self.arousal - self.IDLE_DECAY_RATE * dt
            )
        else:
            # Active — arousal tracks salience peak
            target_arousal = 0.3 + (salience_peak * 0.6)
            self.arousal += (target_arousal - self.arousal) * 0.1 * dt

        # Focus tracks goal count — more goals = less focus per goal
        if active_goals == 0:
            self.focus = max(0.1, self.focus - 0.001 * dt)
        elif active_goals == 1:
            self.focus = min(1.0, self.focus + 0.002 * dt)
        else:
            # Multiple goals reduce focus
            self.focus = max(0.2, self.focus - (active_goals * 0.0005) * dt)

        # Curiosity decays slowly, floors at baseline
        self.curiosity = max(
            self.CURIOSITY_FLOOR,
            self.curiosity - self.CURIOSITY_DECAY_RATE * dt
        )

        # Urgency decays if no urgent goal
        if self._urgent_goal is None:
            self.urgency = max(0.0, self.urgency - 0.01 * dt)

        self._clamp()

    def flag_urgent(self, goal):
        """Elevate urgency due to a pending urgent goal."""
        self._urgent_goal = goal
        self.urgency = min(1.0, self.urgency + 0.3)
        self.arousal = min(1.0, self.arousal + 0.2)

    def clear_urgency(self):
        self._urgent_goal = None

    def on_novel_input(self):
        """Called when novel/unexpected input arrives — boosts curiosity and arousal."""
        self.curiosity = min(1.0, self.curiosity + 0.15)
        self.arousal = min(1.0, self.arousal + 0.1)

    def _clamp(self):
        """Keep all values in [0, 1]."""
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.focus = max(0.0, min(1.0, self.focus))
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.urgency = max(0.0, min(1.0, self.urgency))

    def to_dict(self) -> dict:
        return {
            "arousal": round(self.arousal, 3),
            "focus": round(self.focus, 3),
            "curiosity": round(self.curiosity, 3),
            "urgency": round(self.urgency, 3),
            "mode": self._infer_mode(),
        }

    def _infer_mode(self) -> str:
        """Infer a human-readable mode label from state."""
        if self.arousal < 0.2:
            return "consolidating"
        elif self.urgency > 0.7:
            return "urgent"
        elif self.focus > 0.8 and self.arousal > 0.6:
            return "deep_focus"
        elif self.curiosity > 0.8:
            return "exploratory"
        else:
            return "nominal"
