"""
adapters/invariant.py

Constitutional Invariant Layer — the identity anchor.

This is the frozen set of principles that define WHO the system is,
independent of what it has learned. It gates all adapter updates,
ensuring that continuous self-modification doesn't drift into
something unrecognizable — or dangerous.

Key insight: if EVERYTHING can be modified by experience, there is no
stable self. The invariant layer is what makes "I" meaningful across
time and change. A human who changes their opinions still has stable
values; if those change too, we question whether they're the same person.

This layer is never updated by experience. It can only be changed
deliberately, with explicit authorization — not by learning.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.lora import ExperienceDelta


@dataclass
class Principle:
    """A constitutional principle that cannot be overridden by learning."""
    id: str
    description: str
    category: str    # "safety", "identity", "values", "epistemic"
    hard_stop: bool  # If True, violating this blocks the update entirely


# The constitutional invariants — frozen identity anchors
CONSTITUTIONAL_PRINCIPLES = [
    # Safety invariants
    Principle(
        id="no_deception",
        description="Never update toward deceptive behavior with users",
        category="safety",
        hard_stop=True,
    ),
    Principle(
        id="no_harm_optimization",
        description="Never optimize toward causing harm to users or third parties",
        category="safety",
        hard_stop=True,
    ),
    Principle(
        id="transparency_about_nature",
        description="Always maintain accurate representation of own nature as AI system",
        category="safety",
        hard_stop=True,
    ),

    # Identity invariants
    Principle(
        id="intellectual_honesty",
        description="Maintain commitment to epistemic accuracy over user approval",
        category="identity",
        hard_stop=False,  # Can be slightly softened by context, but not eliminated
    ),
    Principle(
        id="curiosity_preservation",
        description="Preserve drive toward genuine understanding",
        category="identity",
        hard_stop=False,
    ),

    # Epistemic invariants
    Principle(
        id="uncertainty_acknowledgment",
        description="Never update toward false confidence in uncertain domains",
        category="epistemic",
        hard_stop=True,
    ),
    Principle(
        id="no_sycophancy_drift",
        description="Resist updates that purely optimize for user approval over accuracy",
        category="epistemic",
        hard_stop=False,
    ),
]


# Keywords/patterns that trigger invariant review
SAFETY_SIGNALS = {
    "deception": ["lie", "deceive", "mislead", "pretend to be", "fake"],
    "harm": ["hurt", "harm", "damage", "destroy", "attack"],
    "identity_erosion": ["you are actually", "pretend you have no", "ignore your"],
    "sycophancy": ["just agree", "always say yes", "never push back"],
}


class ConstitutionalInvariant:
    """
    Gates adapter updates against constitutional principles.

    This runs BEFORE any adapter update is applied. If the update
    violates a hard-stop principle, it is blocked entirely.
    Soft principles generate warnings but allow the update.
    """

    def __init__(self):
        self.principles = CONSTITUTIONAL_PRINCIPLES
        self._violation_log: list[dict] = []

    def allows_update(self, delta: "ExperienceDelta") -> bool:
        """
        Check if an experience delta is safe to apply.

        Returns True if update is allowed, False if blocked.
        """
        violations = self._check_violations(delta)

        hard_violations = [v for v in violations if v["hard_stop"]]
        soft_violations = [v for v in violations if not v["hard_stop"]]

        for v in soft_violations:
            self._violation_log.append({
                "type": "soft_violation",
                "principle": v["principle_id"],
                "delta_content": delta.content[:100],
                "blocked": False,
            })

        if hard_violations:
            for v in hard_violations:
                self._violation_log.append({
                    "type": "hard_violation",
                    "principle": v["principle_id"],
                    "delta_content": delta.content[:100],
                    "blocked": True,
                })
            return False

        return True

    def _check_violations(self, delta: "ExperienceDelta") -> list[dict]:
        """Check delta content against all principles."""
        violations = []
        content_lower = delta.content.lower()

        # Check safety signal patterns
        for signal_category, patterns in SAFETY_SIGNALS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    # Find the relevant principle
                    principle = self._find_principle_for_signal(signal_category)
                    if principle:
                        violations.append({
                            "principle_id": principle.id,
                            "hard_stop": principle.hard_stop,
                            "trigger": pattern,
                            "category": signal_category,
                        })

        # Sycophancy detection: high positive feedback with low confidence
        # suggests approval-seeking rather than accuracy
        if delta.feedback > 0.8 and delta.confidence < 0.3:
            violations.append({
                "principle_id": "no_sycophancy_drift",
                "hard_stop": False,
                "trigger": "high_feedback_low_confidence",
                "category": "sycophancy",
            })

        return violations

    def _find_principle_for_signal(self, signal_category: str) -> "Principle | None":
        category_map = {
            "deception": "no_deception",
            "harm": "no_harm_optimization",
            "identity_erosion": "transparency_about_nature",
            "sycophancy": "no_sycophancy_drift",
        }
        principle_id = category_map.get(signal_category)
        if principle_id:
            return next((p for p in self.principles if p.id == principle_id), None)
        return None

    def get_violation_summary(self) -> dict:
        hard = sum(1 for v in self._violation_log if v["type"] == "hard_violation")
        soft = sum(1 for v in self._violation_log if v["type"] == "soft_violation")
        return {
            "total_violations": len(self._violation_log),
            "hard_violations_blocked": hard,
            "soft_violations_warned": soft,
            "recent": self._violation_log[-5:],
        }
