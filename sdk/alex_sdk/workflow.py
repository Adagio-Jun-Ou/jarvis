"""
Workflow builder — typed steps that map to capabilities.

A workflow is a sequence of steps. Each step is bound to either a
governed tool (declared via `@tool`) or a capability name. Steps
marked with `confirm()` are two-phase: the first call stages the
action, the user's click on the UI's confirmation card executes it.

Workflows are validated at build time. The agent refuses to deploy
an agent whose workflows reference undeclared capabilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    name: str
    capability: str = ""
    tool: Any | None = None
    confirm: bool = False
    description: str = ""


class Workflow:
    """A named sequence of governed steps.

    Use `step()` for ordinary steps and `confirm()` for human-gated
    ones. The agent will emit a `confirmation_required` event when
    the runtime reaches a `confirm()` step; the user clicks the UI's
    card to execute the action.
    """

    def __init__(self, name: str, *, description: str = "") -> None:
        self.name = name
        self.description = description
        self.steps: list[Step] = []

    def step(
        self, name: str, *, capability: str = "", tool: Any = None, description: str = ""
    ) -> Workflow:
        cap = capability or _capability_of(tool)
        self.steps.append(
            Step(name=name, capability=cap, tool=tool, confirm=False, description=description)
        )
        return self

    def confirm(
        self, name: str, *, capability: str = "", tool: Any = None, description: str = ""
    ) -> Workflow:
        cap = capability or _capability_of(tool)
        self.steps.append(
            Step(name=name, capability=cap, tool=tool, confirm=True, description=description)
        )
        return self

    def validate(self, declared_caps: Iterable[str]) -> list[str]:
        """Return a list of validation errors (empty == ok)."""
        errs: list[str] = []
        declared = set(declared_caps)
        for s in self.steps:
            if s.capability and s.capability not in declared:
                errs.append(
                    f"workflow '{self.name}' step '{s.name}' uses "
                    f"undeclared capability '{s.capability}'"
                )
        return errs

    def to_n8n(self) -> dict:
        """Emit an n8n workflow definition (registered in
        `media.control.workflow_catalog`). Confirm steps carry
        `requires_dual_approval=true`."""
        return {
            "name": self.name,
            "description": self.description,
            "nodes": [
                {
                    "name": s.name,
                    "capability": s.capability,
                    "requires_dual_approval": s.confirm,
                    "description": s.description,
                }
                for s in self.steps
            ],
        }

    def to_native_plan(self) -> dict:
        """Emit a native agent plan (the runtime executes steps in
        order; confirm steps pause for a human click)."""
        return {
            "name": self.name,
            "description": self.description,
            "plan": [
                {
                    "name": s.name,
                    "capability": s.capability,
                    "two_phase": s.confirm,
                    "description": s.description,
                }
                for s in self.steps
            ],
        }


def _capability_of(tool: Any) -> str:
    if tool is None:
        return ""
    meta = getattr(tool, "__alex__", None)
    if isinstance(meta, dict):
        return str(meta.get("capability") or "")
    return ""


def workflow(name: str, *, description: str = "") -> Workflow:
    """Entry point: `weekly = workflow("weekly-brief").step(...)`."""
    return Workflow(name, description=description)
