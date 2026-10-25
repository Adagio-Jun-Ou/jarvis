"""
Review policy — the strict L3 (async judge) preset.

The agent's runtime always reads the active constitutional articles
from the control plane; this object only tunes WHEN the judge fires
and which model judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ReviewMode = Literal["off", "triggered", "always"]


@dataclass
class ReviewPolicy:
    """How strictly the async judge reviews each turn.

    * `off` — the L3 judge never runs. Not recommended for any
      production agent that may take consequential actions.
    * `triggered` — the judge runs only for turns that called a risky
      tool (the agent's default; matches `execution_review.STRICT_TRIGGERS`).
    * `always` — the judge runs on every turn. Highest cost; useful
      for the first 30 days of a new agent or for agents with
      high-impact actions.
    """

    mode: ReviewMode = "triggered"
    strict_model: str = "databricks-claude-haiku-4-5"  # small, strict, cheap
    extra_triggers: list[str] = field(default_factory=list)
    # Articles are always included by the runtime; this object only
    # tunes WHEN the reviewer fires and WHICH small model judges.

    def to_payload(self) -> dict:
        return {
            "mode": self.mode,
            "strict_model": self.strict_model,
            "extra_triggers": list(self.extra_triggers),
        }
