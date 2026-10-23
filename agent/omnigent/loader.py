"""YAML loader for :class:`AgentDef` (vendored, adapted).

Adapted from ``omnigent/inner/loader.py``
(https://github.com/omnigent-ai/omnigent, Apache-2.0).  The
adaptations:

* **Removed** every Claude/Codex/Cursor/Goose/Kimi/Kiro/Antigravity
  harness-specific field, the ``OSEnvSpec`` and ``TerminalEnvSpec``
  sub-loader, the ``CredentialProxySpec`` loader, the ``Skill`` /
  ``SkillTool`` filesystem scanner, and the upstream spec-validator.
* **Kept** the public function :func:`load_agent_def` and the
  minimum YAML keys Alex 2.0 actually uses (``name``, ``prompt``,
  ``instructions``, ``tools``, ``policies``, ``executor``, ``params``,
  ``metadata``).
* **Adapted** the tool loader to recognise ``callable: mod.func``
  dotted-path resolution against the ``agent.tools`` namespace.
  This is the path Alex's :class:`agent.tools.*` functions enter the
  :class:`AgentDef`; in code the same function is exposed as
  :func:`agent.omnigent._alex_bridge.build_agent_def`, which is what
  :mod:`agent.agent` calls.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from .datamodel import AgentDef, ExecutorSpec, ParamDef
from .tools import (
    AgentTool,
    FunctionTool,
    HandoffTool,
    InheritedTool,
    MCPTool,
    SelfAgentTool,
    SkillTool,
    Tool,
)

logger = logging.getLogger(__name__)


def _resolve_callable(dotted: str) -> Any:
    """Resolve a ``package.module.func`` dotted path to the function.

    Raises ``ImportError`` if any segment is missing; ``AttributeError``
    if the leaf doesn't exist on the parent module.
    """
    if ":" in dotted:
        mod, _, attr = dotted.partition(":")
    else:
        mod, _, attr = dotted.rpartition(".")
    if not mod or not attr:
        raise ImportError(
            f"callable path must be 'package.module.func': got {dotted!r}"
        )
    module = importlib.import_module(mod)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{dotted!r}: no attribute {attr!r} on {mod!r}") from exc


def _load_tool(name: str, spec: Any) -> Tool:
    """Materialise a :class:`Tool` from a YAML entry.

    Recognised shapes:

    * ``{type: function, callable: mod.func}`` → :class:`FunctionTool`
    * ``{type: mcp, url: https://...}``        → :class:`MCPTool`
    * ``{type: agent, prompt: ...}``           → :class:`AgentTool`
    * ``{type: skill, path: ...}``             → :class:`SkillTool`
    * ``{type: handoff, target_agent: ...}``   → :class:`HandoffTool`
    * ``self`` (string)                        → :class:`SelfAgentTool`
    * ``{type: inherit}``                      → :class:`InheritedTool`
    """
    if spec == "self":
        return SelfAgentTool(name=name, description=f"self:{name}")
    if not isinstance(spec, dict):
        raise ValueError(
            f"tool {name!r}: expected mapping or 'self', got {type(spec).__name__}"
        )

    kind = spec.get("type", "function")
    if kind == "function":
        callable_path = spec.get("callable")
        if not callable_path:
            raise ValueError(
                f"tool {name!r}: 'callable' is required for function tools"
            )
        return FunctionTool(
            name=name,
            description=spec.get("description"),
            callable=_resolve_callable(callable_path),
            scopes=set(spec.get("scopes") or []),
        )
    if kind == "mcp":
        return MCPTool(
            name=name,
            description=spec.get("description"),
            url=spec.get("url"),
            command=spec.get("command"),
            args=spec.get("args"),
            env=spec.get("env"),
            tools=spec.get("tools"),
            tool_name=spec.get("tool_name"),
            profile=spec.get("profile"),
            databricks_server=spec.get("databricks_server"),
            headers=spec.get("headers"),
        )
    if kind == "agent":
        return AgentTool(
            name=name,
            description=spec.get("description"),
            prompt=spec.get("prompt"),
            tools={},
            pass_history=bool(spec.get("pass_history", False)),
        )
    if kind == "skill":
        return SkillTool(
            name=name,
            description=spec.get("description"),
            path=spec.get("path"),
        )
    if kind == "handoff":
        return HandoffTool(
            name=name,
            description=spec.get("description"),
            target_agent=spec.get("target_agent"),
            pass_history=bool(spec.get("pass_history", True)),
        )
    if kind == "inherit":
        return InheritedTool(name=name, description=spec.get("description"))
    raise ValueError(f"tool {name!r}: unknown type {kind!r}")


def _load_policy(name: str, spec: Any) -> Any:
    """Materialise a :class:`Policy` from a YAML entry.

    The current Alex 2.0 constitution is :class:`FunctionPolicy`-
    shaped: ``{type: function, callable: mod.func}``.  Future
    PromptPolicy / custom-policy YAML is intentionally not added here
    because the bridge (see ``_alex_bridge.py``) is the canonical
    construction site.
    """
    from .policies import FunctionPolicy

    if not isinstance(spec, dict):
        raise ValueError(
            f"policy {name!r}: expected mapping, got {type(spec).__name__}"
        )
    kind = spec.get("type", "function")
    if kind != "function":
        raise ValueError(
            f"policy {name!r}: only type=function is supported by the vendored loader"
        )
    callable_path = spec.get("callable")
    if not callable_path:
        raise ValueError(
            f"policy {name!r}: 'callable' is required for function policies"
        )
    on = spec.get("on") or ["request", "response", "tool_call"]
    return FunctionPolicy(
        name=name,
        on=on,
        callable=_resolve_callable(callable_path),
        config=spec.get("config") or {},
    )


def load_agent_def(path: str) -> AgentDef:
    """Load an :class:`AgentDef` from a YAML file.

    The YAML schema recognised by the vendored copy is intentionally
    narrow; the full upstream schema (with ``os_env``,
    ``terminals``, ``credential_proxy``, ``spawn``, ``policy_*``,
    etc.) is not supported and would raise on load.  See
    ``docs/OMNIGENT_INTEGRATION.md`` for the supported key list.
    """
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    name = raw.get("name")
    if not name:
        raise ValueError(f"{path}: 'name' is required")

    agent = AgentDef(
        name=name,
        prompt=raw.get("prompt"),
        instructions=raw.get("instructions"),
        executor=_maybe_executor_spec(raw.get("executor")),
        metadata=raw.get("metadata") or {},
    )

    for tool_name, tool_spec in (raw.get("tools") or {}).items():
        agent.tools[tool_name] = _load_tool(tool_name, tool_spec)

    for policy_name, policy_spec in (raw.get("policies") or {}).items():
        agent.policies[policy_name] = _load_policy(policy_name, policy_spec)

    for param_name, param_spec in (raw.get("params") or {}).items():
        if not isinstance(param_spec, dict):
            param_spec = {"default": param_spec}
        agent.params[param_name] = ParamDef(
            type=param_spec.get("type", "string"),
            description=param_spec.get("description"),
            default=param_spec.get("default"),
        )

    return agent


def _maybe_executor_spec(raw: Any) -> ExecutorSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"executor: expected mapping, got {type(raw).__name__}")
    return ExecutorSpec(
        model=raw.get("model"),
        harness=raw.get("harness"),
        profile=raw.get("profile"),
        auth=raw.get("auth"),
    )
