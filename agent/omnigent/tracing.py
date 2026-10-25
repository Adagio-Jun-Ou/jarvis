"""MLflow tracing integration for Omnigent (vendored, adapted).

Adapted from ``omnigent/inner/tracing.py`` (Apache-2.0).  The
upstream tracing module is built on OpenTelemetry; Alex 2.0 vendors
a sibling built on MLflow 3.x (``mlflow==3.1.0`` in
``agent/requirements.txt``) because MLflow tracing is the primary
eval/observability tool in the Alex bundle, and MLflow Traces are
visible in the same Databricks workspace where the agent runs.

If MLflow is not importable (e.g. in a unit test that mocks
``mlflow`` away), the helpers degrade to a no-op that just logs span
events through the stdlib ``logging`` module so the agent loop
remains observable in tests.

Span hierarchy mirrors the upstream design::

    agent:<name>     span_kind=AGENT
    ├── tool:<name>  span_kind=TOOL
    └── policy:<name> span_kind=GUARDRAIL

LLM-level spans are not emitted from this module — those are emitted
inside the executor by the underlying OpenAI/Ollama SDK.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("alex.omnigent.tracing")

_tracing_enabled: bool = False


def enable_tracing() -> None:
    global _tracing_enabled
    _tracing_enabled = True
    try:
        import mlflow

        mlflow.set_tracking_uri("databricks")
    except Exception:  # noqa: BLE001 — mlflow may be uninstalled in unit tests; that's fine
        logger.debug("mlflow not importable; tracing enabled in logging-only mode")


def disable_tracing() -> None:
    global _tracing_enabled
    _tracing_enabled = False


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def _safe_serialize(value: Any, max_len: int = 4000) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "...(truncated)"
        return value
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v, max_len) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v, max_len) for v in value]
    try:
        s = json.dumps(value, default=str)
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return s
    except Exception:  # noqa: BLE001
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return s


class TracingContext:
    """Holds the active span stack for a single session/turn.

    Spans are created against MLflow when ``mlflow`` is importable and
    ``_tracing_enabled`` is True, otherwise we emit a ``logger.debug``
    line that includes the span name, kind, and inputs.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._root_span: Any = None
        self._current_span: Any = None
        self._inherited_parent: Any = None
        self.enabled: bool = True
        self.session_id: str | None = session_id

    @property
    def active(self) -> bool:
        return self.enabled and self._root_span is not None

    def _start_span(self, name: str, span_type: str, attributes: dict[str, Any]) -> Any:
        if not (_tracing_enabled and self.enabled):
            logger.debug("span %s kind=%s attrs=%s", name, span_type, attributes)
            return _NoopSpan(name, span_type)
        try:
            from mlflow.tracing import start_span

            return start_span(name=name, span_type=span_type, attributes=attributes)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlflow.start_span failed (%s); falling back to noop", exc)
            return _NoopSpan(name, span_type)

    def _end_span(
        self, span: Any, status: str = "OK", error: str | None = None
    ) -> None:
        if span is None:
            return
        if isinstance(span, _NoopSpan):
            span.end(status=status, error=error)
            return
        try:
            from mlflow.tracing import end_span

            if error:
                end_span(status="ERROR", error=error)
            else:
                end_span(status=status)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlflow.end_span failed (%s); span dropped", exc)

    def start_agent_span(
        self,
        agent_name: str,
        user_message: str,
        model: str | None = None,
    ) -> Any:
        attrs: dict[str, Any] = {
            "agent.name": agent_name,
            "input.value": _safe_serialize(user_message),
        }
        if self.session_id:
            attrs["session.id"] = self.session_id
        if model:
            attrs["llm.model_name"] = model
        span = self._start_span(f"agent:{agent_name}", "AGENT", attrs)
        if self._root_span is None:
            self._root_span = span
        self._current_span = span
        return span

    def end_agent_span(
        self,
        span: Any,
        response: str | None,
        error: str | None = None,
    ) -> None:
        if span is None:
            return
        try:
            from mlflow.tracing import set_span_outputs

            if response is not None:
                set_span_outputs(_safe_serialize(response))
        except Exception:  # noqa: BLE001
            pass
        self._end_span(span, status="ERROR" if error else "OK", error=error)
        if span is self._root_span:
            self._root_span = None
            self._current_span = self._inherited_parent
        elif span is self._current_span:
            self._current_span = self._inherited_parent

    def start_tool_span(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        attrs: dict[str, Any] = {
            "tool.name": tool_name,
            "input.value": _safe_serialize(tool_args),
        }
        if self.session_id:
            attrs["session.id"] = self.session_id
        span = self._start_span(f"tool:{tool_name}", "TOOL", attrs)
        self._current_span = span
        return span

    def end_tool_span(
        self,
        span: Any,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        parent_span: Any = None,
    ) -> None:
        if span is None:
            return
        try:
            from mlflow.tracing import set_span_outputs

            set_span_outputs(_safe_serialize(result))
        except Exception:  # noqa: BLE001
            pass
        self._end_span(span, status="ERROR" if error else "OK", error=error)
        if span is self._current_span:
            self._current_span = parent_span

    def start_policy_span(
        self,
        policy_name: str,
        phase: str,
        content: Any = None,
    ) -> Any:
        attrs: dict[str, Any] = {
            "policy.name": policy_name,
            "policy.phase": phase,
            "input.value": _safe_serialize(content),
        }
        if self.session_id:
            attrs["session.id"] = self.session_id
        return self._start_span(f"policy:{policy_name}", "GUARDRAIL", attrs)

    def end_policy_span(
        self,
        span: Any,
        action: str = "allow",
        reason: str | None = None,
    ) -> None:
        if span is None:
            return
        try:
            from mlflow.tracing import set_span_attributes

            set_span_attributes(
                {"policy.action": action, "policy.reason": reason or ""}
            )
        except Exception:  # noqa: BLE001
            pass
        self._end_span(span, status="ERROR" if action == "deny" else "OK")

    def create_child_context(self) -> TracingContext:
        child = TracingContext(session_id=self.session_id)
        child.enabled = self.enabled
        child._current_span = self._current_span
        child._inherited_parent = self._current_span
        return child


class _NoopSpan:
    def __init__(self, name: str, span_type: str) -> None:
        self.name = name
        self.span_type = span_type

    def end(self, status: str = "OK", error: str | None = None) -> None:
        if error:
            logger.debug(
                "span %s (%s) status=%s error=%s",
                self.name,
                self.span_type,
                status,
                error,
            )
        else:
            logger.debug("span %s (%s) status=%s", self.name, self.span_type, status)
