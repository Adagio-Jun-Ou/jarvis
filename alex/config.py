"""
Centralized configuration for the Alex system.

Every other module reads from `alex.config.settings`. The settings object is
populated once at process start, and downstream code never reaches into
`os.environ` again. This makes it possible to:
  * lock tests to a fixed config (no surprise env vars),
  * surface config errors at import time, not 30 minutes into a job,
  * document the full surface area in one place.

Three "modes" are recognized:
  * "production" — Databricks App runtime, OBO tokens, real warehouses.
  * "staging"    — same as production but pointed at dev/staging targets.
  * "dev"        — local FastAPI; signed dev JWTs; in-memory or sqlite
                   stand-ins for UC, Vector Search, and Lakebase. All
                   external services are mocked via the `alex.testing` module.

The mode is selected by the ALEX_ENV env var. Anything else is an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(RuntimeError):
    """Raised when the runtime configuration is missing or contradictory."""


@dataclass(frozen=True)
class Settings:
    mode: str
    catalog: str = "media"
    alex_schema: str = "alex"
    control_schema: str = "control"
    content_schema: str = "content"
    audience_schema: str = "audience"
    ops_schema: str = "ops"
    agent_endpoint: str = "alex-agent"
    warehouse_id: str = ""
    lakebase_instance: str = "alex-oltp"
    # NOTE: there is deliberately no vector-search-index setting.
    # Retrieval runs on the enterprise_docs pgvector table (Lakebase);
    # a config knob naming a Databricks Vector Search index nothing
    # queries would be a promise the architecture does not keep.
    genie_space_id: str = ""
    # OBO / JWT
    jwt_audience: str = "alex-workspace"
    jwt_issuer: str = "https://databricks.local/oidc"
    jwt_dev_secret: str = "dev-secret-do-not-use-in-prod"  # dev mode only
    # LLM
    # GLM 5.3 Flash: the cost-controlled Databricks-hosted default
    # (pay-per-token; ~1M context; agentic tool use). Overridable via
    # ALEX_DEFAULT_LLM_ENDPOINT; must be in the approved catalog.
    default_llm_endpoint: str = "databricks-glm-5-3-flash"
    critic_llm_endpoint: str = "databricks-claude-haiku-4-5"
    # Behaviour
    constitution_cache_ttl: int = 300
    eval_pass_threshold: float = 0.8
    auto_review_sample_rate: float = 0.10
    upload_max_bytes: int = 25 * 1024 * 1024
    # Stream
    sse_keepalive_seconds: int = 15
    # Secrets scope names (Databricks Secrets)
    secrets_scope: str = "alex"
    secret_browser_vault_key: str = "browser_vault_key"
    secret_n8n_webhook_token: str = "n8n_webhook_token"
    secret_graph_client_secret: str = "graph_client_secret"
    extra: dict = field(default_factory=dict)


def _openrouter() -> bool:
    """The LLM backend is OpenRouter: explicit ALEX_LLM_BACKEND, or a key
    present with no explicit choice. Mirrors alex.llm.llm_backend without
    importing it (config must stay import-free)."""
    explicit = os.environ.get("ALEX_LLM_BACKEND", "").strip().lower()
    if explicit:
        return explicit == "openrouter"
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise ConfigError(f"Required environment variable {name!r} is not set")
    return v


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a valid number") from e
    if not (lo <= v <= hi):
        raise ConfigError(f"{name}={v} out of range [{lo}, {hi}]")
    return v


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a valid integer") from e
    if not (lo <= v <= hi):
        raise ConfigError(f"{name}={v} out of range [{lo}, {hi}]")
    return v


def load() -> Settings:
    mode = os.environ.get("ALEX_ENV", "").strip().lower()
    if not mode:
        # Unset defaults to dev for local friendliness.
        mode = "dev"
    elif mode not in {"production", "staging", "dev"}:
        raise ValueError(
            f"ALEX_ENV={mode!r} is not a recognized mode; "
            "expected one of: production, staging, dev (or unset for dev)"
        )
    return Settings(
        mode=mode,
        catalog=os.environ.get("ALEX_CATALOG", "media"),
        alex_schema=os.environ.get("ALEX_ALEX_SCHEMA", "alex"),
        control_schema=os.environ.get("ALEX_CONTROL_SCHEMA", "control"),
        content_schema=os.environ.get("ALEX_CONTENT_SCHEMA", "content"),
        audience_schema=os.environ.get("ALEX_AUDIENCE_SCHEMA", "audience"),
        ops_schema=os.environ.get("ALEX_OPS_SCHEMA", "ops"),
        agent_endpoint=os.environ.get("AGENT_ENDPOINT", "alex-agent"),
        warehouse_id=os.environ.get("WAREHOUSE_ID", ""),
        lakebase_instance=os.environ.get("LAKEBASE_INSTANCE", "alex-oltp"),
        genie_space_id=os.environ.get("ALEX_GENIE_SPACE_ID", ""),
        jwt_audience=os.environ.get("ALEX_JWT_AUDIENCE", "alex-workspace"),
        jwt_issuer=os.environ.get("ALEX_JWT_ISSUER", "https://databricks.local/oidc"),
        jwt_dev_secret=os.environ.get(
            "ALEX_JWT_DEV_SECRET", "dev-secret-do-not-use-in-prod"
        ),
        default_llm_endpoint=os.environ.get(
            "ALEX_DEFAULT_LLM_ENDPOINT",
            (
                os.environ.get("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
                if _openrouter()
                else "databricks-glm-5-3-flash"
            ),
        ),
        critic_llm_endpoint=os.environ.get(
            "ALEX_CRITIC_LLM_ENDPOINT",
            "anthropic/claude-haiku-4.5" if _openrouter() else "databricks-claude-haiku-4-5",
        ),
        # Behaviour knobs: env-tunable with validated ranges so a bad value
        # (e.g. a sample rate of 2.0 or a negative upload cap) is caught at
        # load time rather than 30 minutes into a job.
        constitution_cache_ttl=_env_int(
            "ALEX_CONSTITUTION_CACHE_TTL", 300, 1, 86_400
        ),
        eval_pass_threshold=_env_float(
            "ALEX_EVAL_PASS_THRESHOLD", 0.8, 0.0, 1.0
        ),
        auto_review_sample_rate=_env_float(
            "ALEX_AUTO_REVIEW_SAMPLE_RATE", 0.10, 0.0, 1.0
        ),
        upload_max_bytes=_env_int(
            "ALEX_UPLOAD_MAX_BYTES", 25 * 1024 * 1024, 1, 1024 * 1024 * 1024
        ),
        sse_keepalive_seconds=_env_int(
            "ALEX_SSE_KEEPALIVE_SECONDS", 15, 1, 3600
        ),
    )


# Lazy-initialized singleton. Importing modules call `get_settings()` which
# loads once per process. Tests call `reset_settings()` between cases to swap
# the env.
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load()
    return _settings


def reset_settings(**overrides) -> Settings:
    """Test helper: build a Settings with explicit overrides, ignoring env.

    The base values come from a fresh load() so test-isolation doesn't
    depend on whatever the OS environment happens to contain."""
    base = load()
    # Build a new Settings with overrides applied on top of the loaded base.
    merged: dict = {**base.__dict__}
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    global _settings
    s = Settings(**merged)
    _settings = s
    return s
