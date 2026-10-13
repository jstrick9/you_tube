"""Hard ceilings for per-stage recovery so config edits cannot create retry storms."""
from __future__ import annotations

from dataclasses import asdict, dataclass

MAX_SCRIPT_DRAFTS = 4
MAX_VISUAL_REPAIRS = 2
MAX_SELECTION_ROUNDS = 3
MAX_TOPICS_PER_VIDEO = 6


def bounded_attempts(value, default: int, maximum: int, minimum: int = 0) -> int:
    if isinstance(value, bool):
        value = default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return min(maximum, max(minimum, number))


@dataclass(frozen=True)
class RecoveryBudgets:
    script_drafts: int = 4
    visual_repairs: int = 2
    selection_rounds: int = 3
    topics_per_video: int = 6

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RecoveryBudgets":
        cfg = cfg if isinstance(cfg, dict) else {}
        content = cfg.get("content") if isinstance(cfg.get("content"), dict) else {}
        qa = cfg.get("qa") if isinstance(cfg.get("qa"), dict) else {}
        return cls(
            script_drafts=bounded_attempts(content.get("max_drafts"), 4, MAX_SCRIPT_DRAFTS, 1),
            visual_repairs=bounded_attempts(qa.get("max_repairs"), 2, MAX_VISUAL_REPAIRS, 0),
            selection_rounds=bounded_attempts(content.get("selection_rounds"), 3, MAX_SELECTION_ROUNDS, 1),
            topics_per_video=bounded_attempts(content.get("max_topic_attempts_per_video"), 6,
                                              MAX_TOPICS_PER_VIDEO, 1),
        )

    def report(self) -> dict[str, int]:
        return asdict(self)
