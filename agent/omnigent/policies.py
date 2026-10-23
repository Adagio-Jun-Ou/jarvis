"""Policy type hierarchy for Omnigent (vendored, adapted).

This is an adapted subset of ``omnigent/inner/policies.py`` from
https://github.com/omnigent-ai/omnigent (Apache-2.0).  The
adaptations:

* **Kept**: :class:`Policy` (base), :class:`FunctionPolicy`,
  :class:`PolicyResult`, :class:`PolicyAction`,
  :class:`PolicyRuntimeContext`, plus the helper event builder and
  the sync/async callable dispatch.
* **Trimmed**: :class:`PromptPolicy` is exposed as a thin stub that
  always returns ALLOW — Alex 2.0's constitution is
  :class:`agent.constitution.ConstitutionalInterceptor`, which is
  registered as a :class:`FunctionPolicy` and runs as our L2 layer.
  Upstream's PromptPolicy calls a real LLM (Databricks FM API) to
  evaluate prompts; Alex doesn't need that path because the four
  constitution layers are simpler and deterministic.
* **Trimmed**: the LLM-driven ``evaluate`` chain on
  :class:`PromptPolicy` is replaced with a deterministic allow.
  Field shape is preserved so a future swap is a one-line import
  change.

The convention is: a policy returns ``PolicyResult(action=DENY)`` to
block.  Alex's :class:`ConstitutionalInterceptor` is wrapped by
``agent/omnigent/_alex_bridge.py::ConstitutionPolicy`` which raises
``ConstitutionalBlock`` on DENY so the executor yields
``ToolCallRequest(name=tool)`` but the bridge converts it to a
``law_block`` SSE event instead of executing.
"""

from __future__ import annotations

import enum
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from .datamodel import AgentDef, ExecutorSpec
from .executor import (
    Executor,
    ExecutorConfig,
)

PolicyContent: TypeAlias = Any
PolicyContext: TypeAlias = dict[str, Any]
PolicyCallable: TypeAlias = Callable[..., Any]
FactoryParams: TypeAlias = dict[str, Any]
PolicyCallableResult: TypeAlias = Any
PolicyResponsePayload: TypeAlias = dict[str, Any]

_POLICY_SYSTEM_PROMPT = "You are an Omnigent policy evaluator. Return exactly one JSON object and nothing else."


class PolicyAction(enum.Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PolicyResult:
    action: PolicyAction = PolicyAction.ALLOW
    reason: str | None = None
    set_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class PolicyRuntimeContext:
    default_executor: Executor | None = None
    default_executor_spec: ExecutorSpec | None = None
    default_executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    default_executor_factory: Callable[[], Executor | None] | None = None
    executor_factory: Callable[[AgentDef], Executor] | None = None


@dataclass
class Policy:
    name: str | None = None
    on: list[Literal["request", "response", "tool_call", "tool_result"]] = field(
        default_factory=lambda: ["request", "response"]
    )
    _runtime_context: PolicyRuntimeContext | None = field(
        default=None, init=False, repr=False
    )
    _session: Any | None = field(default=None, init=False, repr=False)

    def bind_runtime(self, runtime_context: PolicyRuntimeContext) -> None:
        self._runtime_context = runtime_context

    async def evaluate(
        self,
        content: PolicyContent,
        phase: str,
        context: PolicyContext | None = None,
    ) -> PolicyResult:
        del context
        return PolicyResult(action=PolicyAction.ALLOW)

    def reset_turn(self) -> None:
        pass

    def _get_context(self, extra_context: PolicyContext | None = None) -> PolicyContext:
        labels: dict[str, str] = {}
        if self._session is not None:
            labels = dict(self._session.labels)
        context: PolicyContext = {"labels": labels}
        if extra_context:
            context.update(extra_context)
        return context


def _accepts_config(fn: PolicyCallable) -> bool:
    sig = inspect.signature(fn)
    positional_count = 0
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
    return positional_count >= 2


def _build_inner_event(
    content: PolicyContent,
    phase: str,
    context: PolicyContext,
) -> dict[str, Any]:
    tool_name: str | None = None
    if phase == "tool_call" and isinstance(content, dict):
        raw_tool = content.get("tool")
        if isinstance(raw_tool, str):
            tool_name = raw_tool
    elif phase == "tool_result":
        raw_tool_name = context.get("tool_name")
        if isinstance(raw_tool_name, str):
            tool_name = raw_tool_name

    labels = context.get("labels")
    event: dict[str, Any] = {
        "type": phase,
        "target": tool_name,
        "data": content,
        "context": {
            "actor": {},
            **({"labels": labels} if labels is not None else {}),
        },
    }
    return event


def _call_policy_callable(
    fn: PolicyCallable,
    content: PolicyContent,
    phase: str,
    context: PolicyContext,
    config: dict[str, Any] | None = None,
) -> PolicyCallableResult:
    event = _build_inner_event(content, phase, context)
    if _accepts_config(fn):
        return fn(event, config or {})
    return fn(event)


async def _async_call_policy_callable(
    fn: PolicyCallable,
    content: PolicyContent,
    phase: str,
    context: PolicyContext,
    config: dict[str, Any] | None = None,
) -> PolicyCallableResult:
    event = _build_inner_event(content, phase, context)
    if _accepts_config(fn):
        return await fn(event, config or {})
    return await fn(event)


@dataclass
class FunctionPolicy(Policy):
    """A policy backed by a Python callable.

    The callable receives ``(event)`` or ``(event, config)`` where
    ``event`` is a dict and ``config`` is a runtime configuration dict.
    It must return ``{"action": "ALLOW"|"DENY"|"ASK", "reason": "..."}``
    or a :class:`PolicyResult`.
    """

    callable: PolicyCallable | None = None
    factory_params: FactoryParams = field(default_factory=dict)
    factory: PolicyCallable | None = field(default=None, repr=False)
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.factory_params and self.factory is None:
            raise ValueError(
                f"FunctionPolicy '{self.name or '<unnamed>'}': factory_params requires "
                f"factory to be set (needed for serialization and sub-agent copying)"
            )

    def reset_turn(self) -> None:
        if self.callable is not None and hasattr(self.callable, "reset_turn"):
            self.callable.reset_turn()

    def __copy__(self) -> FunctionPolicy:
        return FunctionPolicy(
            name=self.name,
            on=list(self.on),
            callable=(
                self.factory(**self.factory_params)
                if self.factory is not None
                else self.callable
            ),
            factory_params=dict(self.factory_params),
            factory=self.factory,
            config=dict(self.config) if self.config else None,
        )

    async def evaluate(
        self,
        content: PolicyContent,
        phase: str,
        context: PolicyContext | None = None,
    ) -> PolicyResult:
        if self.callable is None:
            return PolicyResult(action=PolicyAction.ALLOW)

        phase_context = {"configured_phases": list(self.on)}
        if context:
            phase_context.update(context)
        merged_context = self._get_context(phase_context)
        if inspect.iscoroutinefunction(self.callable):
            result = await _async_call_policy_callable(
                self.callable,
                content,
                phase,
                merged_context,
                self.config,
            )
        else:
            result = _call_policy_callable(
                self.callable,
                content,
                phase,
                merged_context,
                self.config,
            )

        if isinstance(result, PolicyResult):
            return result
        if isinstance(result, PolicyAction):
            return PolicyResult(action=result)
        if isinstance(result, dict):
            return _coerce_v0_dict_to_result(result)
        # A forgotten `return`, a bare string, a bool — any of these
        # used to fall through to ALLOW. The constitution gates tool
        # execution through this path, so an unrecognized return is a
        # coding fault that must be loud, not a silent pass.
        raise ValueError(
            f"FunctionPolicy callable returned {type(result).__name__}; "
            "expected PolicyResult, PolicyAction, or a dict with "
            f"'action'/'result'. Got: {result!r:.200}"
        )


def _coerce_v0_dict_to_result(raw: dict[str, Any]) -> PolicyResult:
    result_raw = raw.get("action") if "action" in raw else raw.get("result")
    if result_raw is None:
        raise ValueError(
            f"FunctionPolicy dict return missing 'action'/'result' key; got {raw!r}",
        )
    action = PolicyAction(str(result_raw).lower())
    return PolicyResult(
        action=action,
        reason=raw.get("reason"),
    )


@dataclass
class PromptPolicy(Policy):
    """Stub of the upstream ``PromptPolicy`` that always ALLOWs.

    Alex 2.0 doesn't use LLM-evaluated prompt policies (its
    :mod:`agent.constitution` does that work deterministically), so
    this class exists only to keep the public surface import-shape
    compatible with upstream.  A future swap to a real
    LLM-evaluating policy is a one-line change in
    :mod:`agent.omnigent._alex_bridge`.
    """

    prompt: str | None = None
    executor: Executor | ExecutorSpec | None = None
    allow_set_labels: bool = False
    allowed_label_keys: list[str] | None = None

    async def evaluate(
        self,
        content: PolicyContent,
        phase: str,
        context: PolicyContext | None = None,
    ) -> PolicyResult:
        del context
        del content
        del phase
        return PolicyResult(action=PolicyAction.ALLOW)
