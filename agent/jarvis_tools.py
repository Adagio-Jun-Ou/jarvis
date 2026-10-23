"""Jarvis's native tools, registered into the engine's governed registry.

Jarvis ships ~146 tools as ``tools.base.Tool`` subclasses: async
``execute(**params)``, a JSON-schema ``parameters`` block, a
``requires_permission`` flag, and a category in ``core.tool_registry``.
The engine's registry wants a sync callable with a Python signature and
a capability string. This adapter is the bridge, so every Jarvis tool
runs under the engine's constitution, ACL, governor, tool policy and
audit — without touching a line of Jarvis.

Mapping:
* capability ``jarvis.<category>.read`` for plain tools (power 0) and
  ``jarvis.<category>.act`` for tools that declared
  ``requires_permission`` (power 2, policy-ASK by default — a
  supervised chat proceeds and says so, an unattended run parks on
  approval; the same gate Jarvis's PermissionManager expressed);
* results of tools that read the outside world (web, browser, screen,
  data) are marked untrusted and fenced;
* a synthesized signature (required params without defaults, typed from
  the JSON schema) so the engine's arg validation and schema derivation
  work, plus the ORIGINAL JSON schema as ``input_schema`` so the model
  sees Jarvis's parameter descriptions and enums;
* coroutines run on a dedicated thread with their own loop — the engine
  invokes tools from inside a running loop.

Name clashes with engine tools are resolved by prefixing ``jarvis_``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from typing import Any

log = logging.getLogger("agent.jarvis_tools")

#: A Jarvis tool that blocks past its own timeout must not hold an
#: engine tool slot forever (safe_execute bounds most tools; this is
#: the backstop for coroutines that ignore cancellation).
_TOOL_JOIN_TIMEOUT_S = float(os.environ.get("JARVIS_TOOL_TIMEOUT_S", "600"))

_JSON_TO_PY: dict[str, type] = {
    "string": str, "integer": int, "number": float, "boolean": bool,
    "array": list, "object": dict,
}
_UNTRUSTED_CATEGORIES = {"web", "browser", "screen", "data", "multimodal"}
#: File-reading tools whose result can carry third-party text. The
#: engine marks its own ``view``/``read_inbox`` untrusted for the same
#: reason; local files are an injection path too.
_UNTRUSTED_TOOLS = {
    "read_file", "list_files", "native_read", "native_list",
    "project_file_read", "read_sandbox_file", "view_sandbox_results",
}
#: Jarvis tools whose own ``requires_permission`` flag understates their
#: reach (``shell`` runs arbitrary commands but declares False; the
#: project tools write and run; approve/deny resolve someone else's
#: sandbox request). They are promoted to ``.act`` (power 2, policy-ASK)
#: regardless of the flag — the engine's ``compute.bash`` precedent.
_ALWAYS_ACT = {
    "shell", "run_shell", "execute_command",
    "approve_permission", "deny_permission",
    "project_init", "project_decompose", "project_execute",
    "project_file_write", "project_test", "project_commit",
    "sandbox_explore", "tool_practice",
    "notes", "reminders",
    "image_download", "ingest_music_library",
    "generate_composition", "generate_variation",
}
#: Fallback category per Jarvis tool family (core.tool_registry does
#: not catalogue every tool). Without this, 56 tools collapsed into
#: one ``jarvis.misc.*`` capability — a least-privilege problem.
_FAMILY_CATEGORY: dict[str, str] = {
    "tools.system": "system",
    "tools.system_sense": "system",
    "tools.web": "web",
    "tools.productivity": "productivity",
    "tools.spatial": "spatial",
    "tools.art_philosophy": "art",
    "tools.physics_sim": "physics",
    "tools.data_tools": "data",
    "tools.free_time": "learning",
    "tools.screen_tools": "screen",
    "tools.browser_tools": "browser",
    "tools.terminal": "terminal",
    "tools.native_files": "files",
    "tools.insight_search": "knowledge",
    "tools.tool_discovery": "meta",
    "tools.evolution_tools": "evolution",
    "tools.music": "music",
    "tools.project": "project",
    "tools.oracle": "knowledge",
    "tools.cognitive_ai": "knowledge",
}
_REGISTERED: dict[str, str] = {}   # engine name -> jarvis name


def _run_coro(coro):
    """Run a coroutine to completion from a thread that may itself be
    inside a running event loop (the engine's tool seam is)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def _worker():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_worker, name="jarvis-tool", daemon=True)
    t.start()
    t.join(_TOOL_JOIN_TIMEOUT_S)
    if t.is_alive():
        raise TimeoutError(
            f"Jarvis tool did not finish within {_TOOL_JOIN_TIMEOUT_S:.0f}s"
        )
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _result_to_dict(res: Any) -> dict:
    """ToolResult / str / dict -> the engine's {"ok": ...} shape."""
    if hasattr(res, "success"):
        out: dict[str, Any] = {"ok": bool(res.success)}
        if getattr(res, "output", None):
            out["output"] = res.output
        if getattr(res, "result", None):
            out["result"] = res.result
        if getattr(res, "error", None):
            out["error"] = res.error
        return out
    if isinstance(res, dict):
        return {"ok": not res.get("error"), **res}
    return {"ok": True, "output": "" if res is None else str(res)}


def _make_callable(tool: Any):
    schema = getattr(tool, "parameters", None) or {}
    props: dict = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    def call(**kwargs):
        kwargs.pop("user_email", None)
        kwargs.pop("instance_scope", None)
        try:
            return _result_to_dict(_run_coro(tool.safe_execute(**kwargs)))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    params = []
    for pname, pdef in props.items():
        if not pname.isidentifier() or pname in ("user_email", "instance_scope"):
            continue
        ann = _JSON_TO_PY.get(str((pdef or {}).get("type", "string")), str)
        default = inspect.Parameter.empty if pname in required else None
        params.append(inspect.Parameter(
            pname, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=ann
        ))
    params.append(inspect.Parameter("user_email", inspect.Parameter.KEYWORD_ONLY, default=""))
    params.append(inspect.Parameter("instance_scope", inspect.Parameter.KEYWORD_ONLY, default="default"))
    call.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    call.__name__ = str(getattr(tool, "name", "jarvis_tool"))
    call.__doc__ = str(getattr(tool, "description", "") or "")
    return call


def _category_of(name: str, family: str = "") -> str:
    try:
        from core.tool_registry import get_tool_info

        info = get_tool_info(name)
        if info is not None and getattr(info, "category", None) is not None:
            return str(info.category.value)
    except Exception:  # noqa: BLE001
        pass
    return _FAMILY_CATEGORY.get(family, "misc")


def discover() -> list[Any]:
    """Every Jarvis Tool instance the boot sequence would register,
    built the same way run.py builds them, each family independently
    guarded so one missing optional dependency costs one family."""
    found: list[Any] = []

    def _add(items, modname=""):
        for it in items or []:
            if it is None:
                continue
            inst = it() if isinstance(it, type) else it
            if hasattr(inst, "execute") and getattr(inst, "name", None):
                # Provenance for the capability fallback (the shared
                # tool registry does not catalogue every Jarvis tool).
                if modname and not getattr(inst, "_jarvis_family", None):
                    inst._jarvis_family = modname
                found.append(inst)

    def _music_tools(m):
        bag: dict = {}
        m.register_music_tools(bag)
        return list(bag.values())

    def _project_tools(m):
        return [
            m.ProjectInitTool(), m.ProjectStatusTool(),
            m.ProjectDecomposeTool(), m.ProjectExecuteTool(),
            m.ProjectFileReadTool(), m.ProjectFileWriteTool(),
            m.ProjectTestTool(), m.ProjectCommitTool(),
        ]

    families = [
        ("tools.system", lambda m: m.SystemTools().get_tools()),
        ("tools.system_sense", lambda m: [m.SystemSenseTool(), m.SystemWatchTool()]),
        ("tools.web", lambda m: m.WebTools().get_tools()),
        ("tools.productivity", lambda m: m.ProductivityTools().get_tools()),
        ("tools.spatial", lambda m: m.SpatialTools().get_tools()),
        ("tools.art_philosophy", lambda m: m.ArtPhilosophyTools().get_tools()),
        ("tools.physics_sim", lambda m: m.PhysicsSimTools().get_tools()),
        # tools.meta_tools is intentionally absent: its tools need a
        # Jarvis LLM client (SelfReflectionEngine.llm.chat), which the
        # engine does not provide. The engine has its own review surface.
        ("tools.data_tools", lambda m: m.get_data_tools()),
        ("tools.free_time", lambda m: m.get_free_time_tools()),
        ("tools.screen_tools", lambda m: m.SCREEN_TOOLS),
        ("tools.browser_tools", lambda m: m.BROWSER_TOOLS),
        ("tools.terminal", lambda m: m.TERMINAL_TOOLS),
        ("tools.native_files", lambda m: m.TOOLS),
        ("tools.insight_search", lambda m: m.TOOLS),
        ("tools.tool_discovery", lambda m: m.TOOL_DISCOVERY_TOOLS),
        ("tools.evolution_tools", lambda m: m.EVOLUTION_TOOLS),
        # register_music_tools fills a dict (or a registry with
        # register_tools); it returns the count, not the tools.
        ("tools.music", _music_tools),
        ("tools.project", _project_tools),
        ("tools.oracle", lambda m: [m.OraclePredictTool()]),
        ("tools.cognitive_ai", lambda m: [m.CognitivePsychologyTool(), m.MachineAlignmentTool(), m.NeuroscienceTool()]),
    ]
    import importlib

    for modname, build in families:
        try:
            mod = importlib.import_module(modname)
            _add(build(mod), modname)
        except Exception as exc:  # noqa: BLE001
            log.info("jarvis tool family %s skipped: %s", modname, exc)
    # Deduplicate by name, first wins.
    seen: dict[str, Any] = {}
    for t in found:
        seen.setdefault(str(t.name), t)
    return list(seen.values())


def register_jarvis_tools(tools: list[Any] | None = None) -> dict[str, str]:
    """Register Jarvis tools into the engine registry + guardrail maps.
    Idempotent. Returns {engine_name: capability}."""
    from agent import tool_policy
    from agent.execution_governor import CAPABILITY_POWER
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY, ToolSpec

    out: dict[str, str] = {}
    for tool in (tools if tools is not None else discover()):
        jname = str(tool.name)
        name = jname if jname not in TOOL_REGISTRY or _REGISTERED.get(jname) == jname else f"jarvis_{jname}"
        if name in TOOL_REGISTRY and _REGISTERED.get(name) == jname:
            out[name] = TOOL_REGISTRY[name].capability
            continue
        family = str(getattr(tool, "_jarvis_family", "") or "")
        category = _category_of(jname, family).replace(".", "_")
        acts = (
            bool(getattr(tool, "requires_permission", False))
            or jname in _ALWAYS_ACT
        )
        capability = f"jarvis.{category}.{'act' if acts else 'read'}"
        spec = ToolSpec(
            name=name,
            capability=capability,
            callable=_make_callable(tool),
            description=str(getattr(tool, "description", "") or jname)[:400],
            return_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            constitution_articles=("A3",) + (("A8",) if acts else ()),
            untrusted_output=(
                category in _UNTRUSTED_CATEGORIES or jname in _UNTRUSTED_TOOLS
            ),
            input_schema=getattr(tool, "parameters", None) or None,
            deferred=True,
        )
        TOOL_REGISTRY[name] = spec
        TOOL_CAPABILITY[name] = capability
        CAPABILITY_POWER.setdefault(capability, 2 if acts else 0)
        if acts:
            tool_policy.DEFAULT_ASK_CAPABILITIES = frozenset(
                set(tool_policy.DEFAULT_ASK_CAPABILITIES) | {capability}
            )
        _REGISTERED[name] = jname
        out[name] = capability
    log.info("jarvis tools registered: %d", len(out))
    return out


def unregister_jarvis_tools() -> int:
    """Remove every adapted tool (tests, or a hot reload).

    Also drops capability power and ask-policy entries that no
    remaining tool uses, so a reload cannot leave stale governance
    behind (a removed ``jarvis.*.act`` must not stay policy-ASK).
    """
    from agent import tool_policy
    from agent.execution_governor import CAPABILITY_POWER
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY

    removed_caps: set[str] = set()
    n = 0
    for name in list(_REGISTERED):
        spec = TOOL_REGISTRY.pop(name, None)
        TOOL_CAPABILITY.pop(name, None)
        _REGISTERED.pop(name, None)
        n += 1
        if spec is not None:
            removed_caps.add(spec.capability)
    remaining = {s.capability for s in TOOL_REGISTRY.values()}
    stale = removed_caps - remaining
    for capability in stale:
        CAPABILITY_POWER.pop(capability, None)
    if stale:
        tool_policy.DEFAULT_ASK_CAPABILITIES = frozenset(
            c for c in tool_policy.DEFAULT_ASK_CAPABILITIES if c not in stale
        )
    return n
