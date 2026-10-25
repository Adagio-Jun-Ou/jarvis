"""
The governed workspace agent.

The canonical seam is :class:`agent.vm.VM`. This module is a thin
re-export so ``from agent.agent import make_vm`` keeps working; the VM
is where the agent loop, the tool registry, the browser OS and the
control stack are actually composed.

The pre-VM ``AlexAgent`` implementation that used to live here has been
removed — it was a second, unused agent with its own copy of the
constitution wiring, and keeping it meant every control added to the VM
had to be reasoned about twice.
"""

from __future__ import annotations

from agent.vm import VM, HarnessConfig, VMSession, make_vm

__all__ = ["VM", "HarnessConfig", "VMSession", "make_vm"]
