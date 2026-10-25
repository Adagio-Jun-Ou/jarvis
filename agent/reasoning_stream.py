"""
Visible reasoning stream.

When the user toggles "Show thinking", the agent switches to an
open-weights reasoner (the route from `agent.models.open_reasoning_route()`)
and streams the full chain-of-thought to the UI as `reasoning` SSE events.
After the turn, the raw thinking is archived to `media.control.thinking_traces`
(append-only) and a small decision digest is computed for future turns.
"""

from __future__ import annotations

import logging

log = logging.getLogger("agent.reasoning_stream")


REASONING_SYSTEM = (
    "Think step by step before each response. The user has asked to see your "
    "chain-of-thought, so structure it as numbered steps and emit the steps "
    "as separate 'reasoning' events. Keep conclusions short."
)


def wants_visible_reasoning(custom_inputs: dict | None) -> bool:
    """Did the user ask for visible thinking this turn?"""
    if not custom_inputs:
        return False
    return bool(
        custom_inputs.get("show_reasoning") or custom_inputs.get("visible_reasoning")
    )


def reasoner_for_visible_mode() -> str:
    """The route the agent uses when the user asked for visible reasoning.
    Returns the endpoint name (a string)."""
    # Late import to avoid a cycle with agent.models
    from agent.models import open_reasoning_route

    return open_reasoning_route().endpoint
