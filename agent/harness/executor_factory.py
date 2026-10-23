"""
executor_for — the single place that decides which :class:`Executor`
the VM uses.

The choice is the LLM BACKEND (``alex.llm.llm_backend()``: explicit
``ALEX_LLM_BACKEND``, else openrouter when ``OPENROUTER_API_KEY`` is
set, else the local Ollama daemon) — decoupled from ``mode`` (is this
dev?), which only ever said where the *data* lives. The Databricks
executor remains selectable for a workspace deployment; the local
machine never needs it.

* ``openrouter``: :class:`agent.omnigent.OpenRouterExecutor`
* ``ollama``: :class:`agent.omnigent.LocalOllamaExecutor`
* ``databricks``: :class:`agent.omnigent.DatabricksExecutor`
* ``mock``: :class:`agent.omnigent.MockExecutor`
* Tests can pass ``executor=...`` directly to the VM.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("alex.harness.executor_factory")

_MODES = {"dev", "staging", "production"}


def executor_for(mode: str | None = None, backend: str | None = None) -> Any:
    """Return the right Omnigent :class:`Executor`.

    ``mode`` is validated (dev/staging/production) but no longer
    picks the LLM; ``backend`` overrides ``alex.llm.llm_backend()``.
    """
    if mode is None:
        from alex.config import get_settings

        mode = get_settings().mode
    if mode not in _MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    if backend is None:
        from alex.llm import llm_backend

        backend = llm_backend()

    if backend == "openrouter":
        from agent.omnigent import OpenRouterExecutor

        logger.info("harness: OpenRouter executor")
        return OpenRouterExecutor()
    if backend == "ollama":
        from agent.omnigent import LocalOllamaExecutor

        logger.info("harness: local Ollama executor")
        return LocalOllamaExecutor()
    if backend == "databricks":
        from agent.omnigent import DatabricksExecutor

        logger.info("harness: Databricks Model Serving executor")
        return DatabricksExecutor()
    if backend == "mock":
        from agent.omnigent import MockExecutor

        return MockExecutor()
    raise ValueError(f"unknown LLM backend: {backend!r}")


__all__ = ["executor_for"]
