"""
Core SDK: Agent, @tool, @capability, @skill.

`@capability` declares a capability the agent will use. Capabilities
must be declared before any tool that uses them — `_validate()` checks
this at build time and refuses to deploy otherwise.

`@tool` binds a function as a governed tool. The capability is enforced
at runtime by the agent's `guardrails.guarded(...)` gate.

`@skill` authors a skill by hand. The learning loop authors the rest.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

log = logging.getLogger("alex_sdk.core")


# Process-wide registry. We use a class-level container with a lock
# so multiple threads can register concurrently without races.
class _Registry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tools: dict[str, Any] = {}
        self._capabilities: set[str] = set()
        self._skills: dict[str, dict] = {}

    def add_capability(self, name: str) -> None:
        with self._lock:
            self._capabilities.add(name)

    def has_capability(self, name: str) -> bool:
        with self._lock:
            return name in self._capabilities

    def add_tool(self, name: str, tool_obj: Any) -> None:
        with self._lock:
            self._tools[name] = tool_obj

    def add_skill(self, name: str, info: dict) -> None:
        with self._lock:
            self._skills[name] = info

    def tool_names(self) -> list[str]:
        with self._lock:
            return list(self._tools.keys())

    def capabilities(self) -> set[str]:
        with self._lock:
            return set(self._capabilities)

    def skill_names(self) -> list[str]:
        with self._lock:
            return list(self._skills.keys())


_REGISTRY = _Registry()


def capability(name: str, *, delegable: bool = False, requires_confirm: bool = False) -> str:
    """Declare a capability. The capability is registered to the
    control plane on deploy and enforced at runtime by the agent's
    dual-ACL gate. Nothing acts without one."""
    _REGISTRY.add_capability(name)
    log.debug(
        "capability registered: %s (delegable=%s, confirm=%s)", name, delegable, requires_confirm
    )
    return name


def tool(
    capability: str,
    *,
    reviewed: bool = False,
    confirm: bool = False,
    description: str | None = None,
):
    """Bind a function as a governed tool.

    `capability` is the name of a declared capability. The agent
    refuses to register the tool if the capability is not declared.
    `reviewed=True` routes calls through the strict async judge.
    `confirm=True` makes the call two-phase: the first call returns a
    token; the user's click calls the tool again with the token to
    actually execute.
    """

    def deco(fn: Callable) -> Callable:
        if not _REGISTRY.has_capability(capability):
            raise ValueError(
                f"tool '{fn.__name__}' uses undeclared capability "
                f"'{capability}'. Declare it with @capability('{capability}') first."
            )

        @wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped.__alex__ = {
            "capability": capability,
            "reviewed": reviewed,
            "confirm": confirm,
            "description": description or (fn.__doc__ or "").strip(),
        }
        wrapped.name = fn.__name__
        _REGISTRY.add_tool(fn.__name__, wrapped)
        return wrapped

    return deco


def skill(name: str, *, trigger: str, uses: list[str]):
    """Author a skill by hand. `uses` is the list of capabilities the
    skill relies on; all must be declared/approved. The build step
    checks this."""

    def deco(fn: Callable) -> Callable:
        for cap in uses:
            if not _REGISTRY.has_capability(cap):
                raise ValueError(f"skill '{name}' uses undeclared capability '{cap}'")
        _REGISTRY.add_skill(name, {"trigger": trigger, "uses": uses, "fn": fn})
        return fn

    return deco


@dataclass
class Agent:
    """The agent's client-side handle.

    Construction is side-effect-free; `deploy()` is the moment the
    governance check runs. `dev()` runs the same validation against a
    local dev catalog and does NOT ship to production."""

    name: str
    models: list[str] = field(default_factory=lambda: ["claude-sonnet-4-5"])
    tools: list = field(default_factory=list)
    workflows: list = field(default_factory=list)
    memory: str = "vector+graph"  # 'none'|'vector'|'graph'|'vector+graph'
    learning: bool = True
    review: str = "triggered"  # 'off'|'triggered'|'always'
    instance_scope: str = "default"
    description: str = ""

    def _validate(self) -> None:
        """Build-time governance check — fail early, not in prod."""
        errors: list[str] = []
        for t in self.tools:
            meta = getattr(t, "__alex__", None)
            if not meta:
                errors.append(f"tool '{getattr(t, '__name__', t)}' is not a @tool")
                continue
            cap = meta["capability"]
            if not _REGISTRY.has_capability(cap):
                errors.append(f"tool '{t.__name__}' uses undeclared capability '{cap}'")
        for w in self.workflows:
            errors.extend(w.validate(_REGISTRY.capabilities()))
        if errors:
            raise ValueError("Governance validation failed:\n  - " + "\n  - ".join(errors))

    def deploy(self, target: str = "prod") -> dict:
        """Validate governance, register capabilities/tools/workflows,
        run the eval + constitutional gate, then ship to a serving
        endpoint. Wraps `agent.deploy.deploy`."""
        self._validate()
        log.info(
            "agent '%s' validated: %d tools, %d workflows",
            self.name,
            len(self.tools),
            len(self.workflows),
        )
        # In production, this calls agent.deploy.log_and_register
        # followed by run_eval and deploy. We import lazily so the SDK
        # works without the agent module on the build machine.
        try:
            from agent.deploy import deploy, log_and_register, run_eval
        except ImportError as exc:
            raise RuntimeError(
                "agent module not on PYTHONPATH; install the agent alongside the SDK"
            ) from exc
        uri = log_and_register()
        result = run_eval(uri)
        if not result.get("passed"):
            raise RuntimeError(
                f"constitutional gate failed (correctness={result.get('correctness')}); "
                f"refusing to deploy"
            )
        endpoint = deploy(uri, endpoint_name=f"alex-{self.name}", eval_result=result)
        return {
            "endpoint": endpoint.get("endpoint") or f"alex-{self.name}",
            "target": target,
            "correctness": result.get("correctness"),
        }

    def dev(self) -> None:
        """Local loop against the dev catalog with a mocked identity.

        This does not ship anything. It validates the build, then
        prints the URL a developer can hit locally.
        """
        self._validate()
        log.info(
            "agent '%s' live at http://localhost:8770 " "(mock identity, dev catalog)", self.name
        )
        print(f"[alex dev] '{self.name}' ready at http://localhost:8770")
