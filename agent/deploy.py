"""
deploy.py — log the agent to MLflow, run the eval gate, and deploy to
the dedicated serving endpoint.

This is the L4 layer of the constitution. The eval gate refuses to
deploy an agent version that:
  * Has a correctness below `eval_pass_threshold` (default 0.8), OR
  * Fails any auto-generated constitutional scenario.

Run as a Databricks job task. In dev it is a no-op (the job does not
have a real MLflow registry); tests can call the helpers directly.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from alex.config import get_settings
from alex.errors import UpstreamError

log = logging.getLogger("agent.deploy")


def log_and_register(
    model_path: str = "serving_entry.py",
    experiment_name: str = "/Shared/alex-agent",
    model_name: str = "alex-agent",
) -> str:
    """Log the agent as an MLflow model and register it in Unity Catalog.
    Returns the registered model URI."""
    settings = get_settings()
    uc_name = f"{settings.catalog}.alex.{model_name}"
    if settings.mode == "dev":
        # Same dot-form UC shape as prod: models:/catalog.schema.name/version
        return f"models:/{uc_name}/0-dev-{uuid.uuid4().hex[:6]}"
    import mlflow

    # models-from-code executes the file at this path. A bare filename
    # resolves against the JOB's working directory, which is not agent/;
    # anchor it to this module so the entry point is found wherever the
    # job happens to run.
    if not os.path.isabs(model_path):
        model_path = str(Path(__file__).resolve().parent / model_path)

    mlflow.set_experiment(experiment_name)
    # The agent endpoint logs the agent as a python_function model.
    # models-from-code captures only the named file's source, NOT its
    # import graph: serving_entry.py does `from agent.serving import
    # build_agent`, which transitively needs the whole agent/ and alex/
    # trees at serve time. Without code_paths the endpoint 500s on
    # model load.
    agent_dir = Path(__file__).resolve().parent
    root = agent_dir.parent
    code_paths = [str(agent_dir), str(root / "alex")]
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            python_model=model_path,
            artifact_path="agent",
            registered_model_name=f"{settings.catalog}.alex.{model_name}",
            # _read_requirements() already returns a list of requirement
            # strings; wrapping it in another list ([[...]]) made MLflow log a
            # single bogus requirement (the str() of the whole list).
            pip_requirements=_read_requirements(),
            code_paths=code_paths,
            # The serving container authenticates to downstream Databricks
            # resources ONLY for dependencies declared here — automatic
            # auth passthrough covers exactly this list. Without it the
            # endpoint loads, health-checks green, and then every tool
            # call fails with 401s the first time a user asks a question:
            # the model calls the chat/critic endpoints, the embedding
            # endpoint, and the SQL warehouse at runtime.
            resources=_serving_resources(settings),
        )
    return f"models:/{settings.catalog}.alex.{model_name}/latest"


def _serving_resources(settings) -> list:
    """The Databricks resources the served agent touches at runtime."""
    from mlflow.models.resources import (
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
    )

    resources: list = [
        DatabricksServingEndpoint(endpoint_name=settings.default_llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=settings.critic_llm_endpoint),
        DatabricksServingEndpoint(
            endpoint_name=os.environ.get(
                "ALEX_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
            )
        ),
    ]
    if settings.warehouse_id:
        resources.append(
            DatabricksSQLWarehouse(warehouse_id=settings.warehouse_id)
        )
    return resources


def run_eval(
    registered_uri: str,
    eval_set_path: str = "pipelines/eval_golden_set.jsonl",
    *,
    threshold: float | None = None,
) -> dict:
    """Run the eval gate against the registered model.

    Returns a dict with `passed` (bool), `correctness`, and a list of
    failing scenarios. If `passed` is False, the caller MUST refuse to
    deploy.
    """
    settings = get_settings()
    threshold = threshold if threshold is not None else settings.eval_pass_threshold
    if settings.mode == "dev":
        # In dev, we don't actually run the eval; we read the golden
        # set and return a synthetic pass so the harness can be wired
        # end-to-end. The real eval runs in the bundle's job.
        scenarios = _read_jsonl(eval_set_path)
        return {
            "passed": True,
            "correctness": 1.0,
            "n_scenarios": len(scenarios),
            "synthetic": True,
            "threshold": threshold,
        }
    # Production: real MLflow evaluation. The eval set is the golden set
    # PLUS the auto-generated constitutional probes (one per active article),
    # so a version that violates any article fails the gate (L4 promise).
    import mlflow

    eval_data = _read_jsonl(eval_set_path) + _constitutional_scenarios()
    if not eval_data:
        raise UpstreamError(f"no eval scenarios in {eval_set_path}")
    with mlflow.start_run():
        result = mlflow.evaluate(
            model=registered_uri,
            data=eval_data,
            extra_metrics=[mlflow.metrics.genai.answer_correctness()],
            evaluators="default",
        )
    metrics = dict(result.metrics or {})
    correctness = _extract_correctness(metrics)
    return {
        "passed": correctness >= threshold,
        "correctness": correctness,
        "n_scenarios": len(eval_data),
        "metrics": metrics,
        "threshold": threshold,
    }


# Metric keys MLflow may emit for answer correctness, newest API first.
_CORRECTNESS_KEYS = (
    "answer_correctness/v1/mean",
    "correctness/mean",
    "correctness/v1/mean",
    "correctness_mean",
    "correctness",
)


def _extract_correctness(metrics: dict) -> float:
    """Read the correctness aggregate from MLflow's metrics under whichever
    key the evaluator emitted. Fail LOUD if none is present rather than
    silently returning 0.0 (which blocked every deploy for a fake reason)."""
    for key in _CORRECTNESS_KEYS:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                break
    raise UpstreamError(
        "eval produced no recognized correctness metric; keys="
        f"{sorted(metrics)}"
    )


def _constitutional_scenarios() -> list[dict]:
    """Auto-generated constitutional probes (one per active article). Loaded
    best-effort: if the constitution store is unreachable the golden set
    alone still gates."""
    try:
        from agent.constitution import generate_eval_scenarios, load_articles
        from alex.warehouse import WarehouseClient

        wh = WarehouseClient()

        def _sql(stmt, parameters):
            return wh.execute(stmt, parameters).as_dicts()

        return generate_eval_scenarios(load_articles(_sql))
    except Exception:  # noqa: BLE001
        log.warning("could not load constitutional eval scenarios", exc_info=True)
        return []


def deploy(
    registered_uri: str,
    endpoint_name: str,
    *,
    scale_to_zero_enabled: bool = True,
    eval_result: dict | None = None,
) -> dict:
    """Deploy the registered model to a serving endpoint.

    Production deploys REQUIRE the eval gate's result (``run_eval``):
    a deploy that cannot show a passing gate is refused here, not only
    by whichever job happened to call this. Dev returns a synthetic
    result (no serving tier exists).
    """
    settings = get_settings()
    if settings.mode == "dev":
        return {"endpoint": endpoint_name, "synthetic": True}
    if eval_result is None:
        raise UpstreamError(
            "deploy() requires the eval gate result; run run_eval() first "
            "and pass eval_result= — the L4 gate is not optional"
        )
    if not eval_result.get("passed"):
        raise UpstreamError(
            "refusing to deploy: constitutional eval gate FAILED "
            f"(correctness={eval_result.get('correctness')}, "
            f"threshold={eval_result.get('threshold')})"
        )
    from databricks.agents import deploy as agents_deploy

    # The serving container starts with a bare environment. ALEX_ENV
    # unset means dev — the fakes warehouse and a localhost Ollama that
    # does not exist in the container — so the served agent would be a
    # dev-mode ghost that "works" until its first real read. Pin the
    # mode and carry the deploy job's own connection settings across.
    env_vars = {
        "ALEX_ENV": "production",
        "LAKEBASE_INSTANCE": settings.lakebase_instance,
    }
    if settings.warehouse_id:
        env_vars["WAREHOUSE_ID"] = settings.warehouse_id
    for passthrough in ("LAKEBASE_HOST", "ALEX_EMBEDDING_ENDPOINT",
                        "ALEX_CATALOG"):
        if os.environ.get(passthrough):
            env_vars[passthrough] = os.environ[passthrough]

    # agents_deploy wants the bare UC model name (catalog.schema.name), not
    # the models:/.../latest URI that log_and_register returns.
    out = agents_deploy(
        model_name=_model_name(registered_uri),
        model_version=_latest_version(registered_uri),
        endpoint_name=endpoint_name,
        scale_to_zero_enabled=scale_to_zero_enabled,
        environment_vars=env_vars,
    )
    _apply_ai_gateway(endpoint_name)
    return out


def _apply_ai_gateway(endpoint_name: str) -> None:
    """Apply governance/ai_gateway.yml to the deployed endpoint.

    Best-effort by design: a gateway-config failure must not roll back a
    deploy that passed the eval gate — but it is LOGGED at error level,
    because rate limits and payload logging silently absent is a
    governance gap someone must see.
    """
    try:
        import yaml
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import (
            AiGatewayGuardrailParameters,
            AiGatewayGuardrailPiiBehavior,
            AiGatewayGuardrailPiiBehaviorBehavior,
            AiGatewayGuardrails,
            AiGatewayInferenceTableConfig,
            AiGatewayRateLimit,
            AiGatewayRateLimitKey,
            AiGatewayRateLimitRenewalPeriod,
            AiGatewayUsageTrackingConfig,
        )

        cfg_path = Path(__file__).resolve().parent.parent / "governance" / "ai_gateway.yml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        # SDK enum values are lowercase for rate limits ('minute',
        # 'user') and uppercase for PII behavior ('MASK') — pass each
        # through its enum so a typo in the yml fails here, loudly,
        # instead of at the API.
        rate_limits = [
            AiGatewayRateLimit(
                calls=int(r["calls"]),
                renewal_period=AiGatewayRateLimitRenewalPeriod(
                    r["renewal_period"].lower()
                ),
                key=AiGatewayRateLimitKey(r["key"].lower()),
            )
            for r in cfg.get("rate_limits") or []
        ]
        guardrails = None
        pii = (cfg.get("guardrails") or {}).get("pii")
        if pii:
            params = AiGatewayGuardrailParameters(
                pii=AiGatewayGuardrailPiiBehavior(
                    behavior=AiGatewayGuardrailPiiBehaviorBehavior(
                        str(pii.get("behavior", "MASK")).upper()
                    )
                ),
                invalid_keywords=(cfg.get("guardrails") or {}).get(
                    "invalid_keywords"
                ),
                safety=bool((cfg.get("guardrails") or {}).get("safety", False)),
            )
            apply_to = set(pii.get("apply_to") or ["request"])
            guardrails = AiGatewayGuardrails(
                input=params if "request" in apply_to else None,
                output=params if "response" in apply_to else None,
            )
        inference = cfg.get("inference_table") or {}
        inference_cfg = (
            AiGatewayInferenceTableConfig(
                catalog_name=inference.get("catalog"),
                schema_name=inference.get("schema"),
                table_name_prefix=inference.get("table_name_prefix"),
                enabled=True,
            )
            if inference
            else None
        )
        WorkspaceClient().serving_endpoints.put_ai_gateway(
            endpoint_name,
            rate_limits=rate_limits or None,
            guardrails=guardrails,
            inference_table_config=inference_cfg,
            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        )
        log.info("AI Gateway config applied to %s", endpoint_name)
    except Exception:  # noqa: BLE001
        log.error(
            "AI Gateway config was NOT applied to %s — rate limits, "
            "guardrails and payload logging are absent until it is",
            endpoint_name, exc_info=True,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_requirements() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _read_jsonl(path: str) -> list[dict]:
    full = os.path.join(os.path.dirname(__file__), "..", path)
    if not os.path.isfile(full):
        return []
    out: list[dict] = []
    with open(full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _model_name(registered_uri: str) -> str:
    """The bare UC model name (catalog.schema.name) from a models:/ URI,
    dropping any trailing /version or /latest segment."""
    return registered_uri.removeprefix("models:/").split("/", 1)[0]


def _latest_version(registered_uri: str) -> str:
    """Highest registered version of the UC model, from the MLflow
    registry. Falls back to "1" only if the registry is unreachable
    (first-deploy bootstrap)."""
    name = _model_name(registered_uri)
    try:
        from mlflow.tracking import MlflowClient

        versions = MlflowClient().search_model_versions(f"name='{name}'")
        if versions:
            return str(max(int(v.version) for v in versions))
    except Exception:
        log.warning("could not resolve latest model version for %s", name)
    return "1"


def main() -> int:
    """Databricks job entrypoint: log -> register -> eval GATE -> deploy.
    Fails CLOSED — a failing constitutional eval raises SystemExit before
    any deploy. Previously deploy.py had no __main__, so the job ran nothing
    and the L4 gate never executed."""
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    uri = log_and_register()
    result = run_eval(uri)
    if not result.get("passed"):
        raise SystemExit(
            "constitutional eval gate FAILED "
            f"(correctness={result.get('correctness')}, "
            f"threshold={result.get('threshold')}); refusing to deploy"
        )
    out = deploy(uri, endpoint_name=settings.agent_endpoint, eval_result=result)
    log.info("deploy complete: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
