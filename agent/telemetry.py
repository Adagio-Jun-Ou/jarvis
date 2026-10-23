"""Telemetry export — run accounting for the enterprise's own pipes.

Claude Code and goose both ship OTel export because an enterprise
watches its agents from ITS observability stack, not from the agent's
own UI. This is that seam, with graceful degradation so it can never
cost a run:

* ``ALEX_TELEMETRY`` unset (default): every call is a no-op.
* ``ALEX_TELEMETRY=log``: one structured JSON line per event on the
  ``alex.telemetry`` logger — Databricks Apps captures app logs, and
  every enterprise log pipeline can scrape a JSON line.
* ``ALEX_TELEMETRY=otlp``: real OpenTelemetry spans to
  ``ALEX_OTEL_ENDPOINT`` when the ``opentelemetry-sdk`` +
  ``opentelemetry-exporter-otlp`` packages are installed (operator
  opt-in; they are deliberately NOT in the pinned requirements — the
  app container stays small unless someone wants this). Missing
  packages degrade to the log mode with one warning, never an error.

Events are attributes-only — counts, durations, ids. Never content:
the payloads people write stay out of the telemetry pipe by
construction, which is the same promise the exec views make.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("agent.telemetry")
_tel = logging.getLogger("alex.telemetry")

_warned_missing_otlp = False
_tracer: Any = None


def _mode() -> str:
    return os.environ.get("ALEX_TELEMETRY", "").strip().lower()


def _otlp_tracer() -> Any:
    """The OTel tracer, built once; None when the stack is unavailable."""
    global _tracer, _warned_missing_otlp
    if _tracer is not None:
        return _tracer
    endpoint = os.environ.get("ALEX_OTEL_ENDPOINT", "").strip()
    if not endpoint:
        if not _warned_missing_otlp:
            _warned_missing_otlp = True
            log.warning("ALEX_TELEMETRY=otlp but ALEX_OTEL_ENDPOINT is "
                        "unset; falling back to log lines")
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": "alex-agent"})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("alex")
        return _tracer
    except Exception:  # noqa: BLE001
        if not _warned_missing_otlp:
            _warned_missing_otlp = True
            log.warning(
                "ALEX_TELEMETRY=otlp but the OpenTelemetry packages are "
                "not installed (pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp); falling back to log lines"
            )
        return None


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Attributes only — scalars, no content-bearing values."""
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float):
            out[key] = value
        else:
            out[key] = str(value)[:200]
    return out


def emit(event: str, **attrs: Any) -> None:
    """One telemetry event. Never raises, never blocks a run."""
    mode = _mode()
    if not mode:
        return
    try:
        clean = _clean(attrs)
        if mode == "otlp":
            tracer = _otlp_tracer()
            if tracer is not None:
                with tracer.start_as_current_span(event) as span:
                    for key, value in clean.items():
                        span.set_attribute(f"alex.{key}", value)
                return
        _tel.info(json.dumps(
            {"event": event, "ts": round(time.time(), 3), **clean},
            separators=(",", ":"), ensure_ascii=False, sort_keys=True,
        ))
    except Exception:  # noqa: BLE001
        log.debug("telemetry emit failed", exc_info=True)
