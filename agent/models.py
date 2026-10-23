"""
Multi-model selection for the Alex agent.

Every model the agent may call is a real, deployed Mosaic AI serving
endpoint. The model catalog is data — adding a model is a one-line
change here + a corresponding route in `governance/ai_gateway.yml`.

Production:
  * Endpoints are real Databricks Model Serving endpoints.
  * Every endpoint sits behind AI Gateway (rate limits, PII guardrails,
    payload logging).
  * The default model is the one the system recommends unless the user
    explicitly picks another, or the 'auto' heuristic picks a cheaper
    one for short or factual turns.

Dev / test:
  * The agent endpoint name is read from the env (`ALEX_DEFAULT_LLM_ENDPOINT`)
    or from `alex.config.Settings`. The same field is used in production,
    so swapping endpoints is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    endpoint: str
    tier: str  # 'frontier' | 'fast' | 'cheap' | 'open_reasoning'
    supports_tools: bool
    notes: str


# The approved catalog. Names are real Databricks serving endpoint names
# (Mosaic AI). Adding a model = a row in ai_gateway.yml + a row here.
MODEL_CATALOG: dict[str, ModelRoute] = {
    # The cost spine: GLM 5.3 Flash (Databricks-hosted, pay-per-token,
    # endpoint databricks-glm-5-3-flash) carries routine turns; GLM 5.3
    # (documented function calling + parallel tool calls) carries the
    # frontier ones. Both sit behind AI Gateway like everything else.
    "glm-flash": ModelRoute(
        "databricks-glm-5-3-flash",
        "fast",
        True,
        "Default. Cost-controlled agentic turns; MoE, agentic tool use.",
    ),
    "glm": ModelRoute(
        "databricks-glm-5-3",
        "frontier",
        True,
        "Frontier GLM: function calling, parallel tools, long analysis.",
    ),
    "claude-sonnet": ModelRoute(
        "databricks-claude-sonnet-4-5",
        "frontier",
        True,
        "Agentic workflows, tool use (explicit pick).",
    ),
    "claude-opus": ModelRoute(
        "databricks-claude-opus-4-5",
        "frontier",
        True,
        "Deep reasoning: contracts, rights analysis.",
    ),
    "llama-3-3-70b": ModelRoute(
        "databricks-meta-llama-3-3-70b-instruct",
        "fast",
        True,
        "Fast/cheap; fallback route.",
    ),
    "claude-haiku": ModelRoute(
        "databricks-claude-haiku-4-5",
        "fast",
        True,
        "Cheap summarization and the async critic.",
    ),
    "open-reasoning": ModelRoute(
        "databricks-qwen3-235b-thinking",
        "open_reasoning",
        True,
        "Open weights; visible chain-of-thought for execution.",
    ),
}

# OpenRouter routes — the local machine's catalog. Same logical keys
# where the model is the same, so the UI picker and resolve_model are
# backend-agnostic; the approved-endpoint gate accepts both families.
OPENROUTER_CATALOG: dict[str, ModelRoute] = {
    "or-glm-flash": ModelRoute(
        "z-ai/glm-5.3-flash", "fast", True,
        "Default on OpenRouter. Cost-controlled agentic turns.",
    ),
    "or-glm": ModelRoute(
        "z-ai/glm-5.3", "frontier", True,
        "Frontier GLM: function calling, long analysis.",
    ),
    "or-claude-sonnet": ModelRoute(
        "anthropic/claude-sonnet-4.5", "frontier", True,
        "Agentic workflows, tool use (explicit pick).",
    ),
    "or-claude-haiku": ModelRoute(
        "anthropic/claude-haiku-4.5", "fast", True,
        "Cheap summarization and the async critic.",
    ),
    "or-kimi": ModelRoute(
        "moonshotai/kimi-k2", "frontier", True,
        "Code, math, analysis (the Jarvis backend model).",
    ),
    "or-qwen-thinking": ModelRoute(
        "qwen/qwen3-235b-a22b-thinking-2507", "open_reasoning", True,
        "Open weights; visible chain-of-thought.",
    ),
}
MODEL_CATALOG.update(OPENROUTER_CATALOG)


def _backend() -> str:
    import os

    explicit = os.environ.get("ALEX_LLM_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    return "openrouter" if os.environ.get("OPENROUTER_API_KEY", "").strip() else "ollama"


def _pick(databricks_key: str, openrouter_key: str) -> str:
    return openrouter_key if _backend() == "openrouter" else databricks_key


DEFAULT_MODEL = "glm"
CHEAP_MODEL = "glm-flash"
OPEN_REASONING_MODEL = "open-reasoning"

# Cheap heuristic router for 'auto' mode: burn frontier tokens only when
# the turn actually needs them. (Gateway intelligent routing can replace
# this; the heuristic is a fallback.)
_FRONTIER_TRIGGERS = (
    "contract",
    "rights",
    "license",
    "legal",
    "analyze",
    "plan",
    "strategy",
    "write",
    "draft",
    "compare",
    "why",
    "summarize the rights",
)


def resolve_model(requested: str | None, last_user_msg: str) -> ModelRoute:
    """Return the ModelRoute for a user request. The 'auto' mode picks by
    a cheap heuristic. A specific id must exist in the catalog."""
    if requested and requested != "auto":
        route = MODEL_CATALOG.get(requested)
        if route is None:
            raise ValueError(
                f"Model '{requested}' is not in the approved catalog: "
                f"{sorted(MODEL_CATALOG)}"
            )
        return route
    text = (last_user_msg or "").lower()
    needs_frontier = len(text) > 400 or any(t in text for t in _FRONTIER_TRIGGERS)
    if needs_frontier:
        return MODEL_CATALOG[_pick(DEFAULT_MODEL, "or-glm")]
    return MODEL_CATALOG[_pick(CHEAP_MODEL, "or-glm-flash")]


def open_reasoning_route() -> ModelRoute:
    """The route used when visible reasoning is requested. The 'open weights'
    model whose full chain-of-thought is appropriate to surface."""
    return MODEL_CATALOG[_pick(OPEN_REASONING_MODEL, "or-qwen-thinking")]
