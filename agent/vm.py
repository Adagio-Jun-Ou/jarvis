"""
agent.vm — the VM (the canonical Alex 2.0 process model).

A :class:`VM` owns everything one Alex session needs:

* an LLM executor (Ollama Cloud in dev, Databricks Model Serving
  in production — selected by :func:`agent.harness.executor_for`)
* a browser OS backend (in-memory in dev, Playwright in
  production)
* a session identity (the OBO user email + instance scope)
* a constitution interceptor (L2 of the four-layer constitution)
* an event pump (the SSE event subscriber registry)
* an audit log (writes to ``media.alex.agent_actions`` and
  ``media.control.constitution_violations``)
* a tools registry (the agent's :class:`AgentDef` is built from
  :data:`agent.tools.registry.TOOL_REGISTRY`)

The VM exposes one method per user-facing operation.  The
FastAPI routes, the SDK, and the test harness construct a VM and
call those methods.  The VM is the *only* place the agent loop,
the browser OS, and the audit log are composed.

Why a class and not a module?  A class lets us hang the executor,
the event pump, the session, and the browser OS off ``self`` so
``ReactLoop`` can reach them via ``self._invoke_tool(request)``
without a global registry.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.agentic_browser import (
    browser_click,
    browser_extract,
    browser_navigate,
    browser_open_in_user_session,
    browser_screenshot,
    browser_submit,
    browser_type,
)
from agent.browser_os import BrowserSessionState, browser_os_for_mode
from agent.constitution import (
    ConstitutionalInterceptor,
    build_constitution_prompt,
    load_articles,
    make_violation_logger,
)
from agent.harness import EventPump, executor_for
from agent.harness.constitution_adapter import ConstitutionAdapter
from agent.harness.react_loop import ReactLoop
from agent.omnigent import (
    AgentDef,
    Executor,
    ToolCallRequest,
)
from agent.persona import PERSONA, denial_payload, extract_durable_fact
from agent.tools import TOOL_REGISTRY
from agent.tools.registry import ToolSpec
from alex.config import get_settings
from alex.errors import ConstitutionalBlock, Denied
from alex.events import (
    artifact_event,
    context_event,
    error_event,
    governor_event,
    law_block_event,
    steering_event,
    tool_event,
)

logger = logging.getLogger("alex.vm")


# The full ``media.alex.agent_actions`` column set, matching the
# governance DDL (governance/permissions.sql).  The governance
# columns (id/actor/action_type/tool_name/tool_args/outcome) are the
# source of truth; the harness columns (action_id/tool/capability/
# user_email/instance_scope/input_json/output_json/ok) are kept for
# the VM/browser tooling that reads them.
_AGENT_ACTIONS_COLUMNS = [
    "id",
    "ts",
    "actor",
    "session_id",
    "action_type",
    "target_id",
    "detail",
    "outcome",
    "tool_name",
    "tool_args",
    "action_id",
    "tool",
    "capability",
    "user_email",
    "instance_scope",
    "input_json",
    "output_json",
    "ok",
]


# Per-(user, warehouse) cache of the effective-capability lookup.
# Keyed by (user_email, id(warehouse)); entries expire after
# _ACL_TTL_S seconds.  Tests clear it via _acl_cache_clear().
_ACL_TTL_S = float(os.environ.get("ALEX_ACL_TTL_S", "60"))
# Model tool-selection accuracy degrades sharply past roughly this many
# tools — and every schema is prompt the model re-reads on every turn.
# The cap is on the WORKING SET as a whole, not the remote portion:
# capping only remote tools still let the total run away.
MAX_TOOLS_IN_PROMPT = int(os.environ.get("ALEX_MAX_TOOLS", "100"))
MAX_REMOTE_TOOLS = int(os.environ.get("ALEX_MAX_REMOTE_TOOLS", "40"))
#: Deferred (library) tools admitted to one turn by relevance.
MAX_DEFERRED_IN_PROMPT = int(os.environ.get("ALEX_MAX_DEFERRED_TOOLS", "24"))
#: How long a VM reuses its discovered MCP tool set. The turn's working
#: set is rebuilt per turn (deferred admission), and discovery loads the
#: principal + asks every server: without a TTL that work ran once per
#: message. A revoked remote capability still bites at call time (the
#: caller re-resolves it), so the cache only affects exposure.
MCP_DISCOVERY_TTL_S = float(os.environ.get("ALEX_MCP_DISCOVERY_TTL_S", "300"))


def score_tool(spec: Any, words: set[str]) -> int:
    """Lexical relevance of a tool to a request: name words count double."""
    name_words = set(str(spec.name).lower().replace("_", " ").split())
    desc_words = set(str(spec.description or "").lower().replace("_", " ").split())
    return 2 * len(words & name_words) + len(words & desc_words)
_ACL_CACHE: dict[tuple[str, str], tuple[float, set[str] | None]] = {}
_ACL_LOCK = threading.Lock()


def _acl_cache_clear() -> None:
    _ACL_CACHE.clear()


_prod_warehouse = None


def _vm_warehouse():
    """The warehouse the VM uses for ACL lookups and the memory-mirror
    write. Dev/test: the in-memory fake. Production: the real Databricks SQL
    warehouse. Using the fake unconditionally meant the ACL query always saw
    an empty table in prod (falling back to exposing every tool) and the
    mirror write silently no-op'd."""
    from alex import testing as _testing

    if _testing.is_dev():
        return _testing.get_fakes().warehouse
    global _prod_warehouse
    if _prod_warehouse is None:
        from alex.warehouse import WarehouseClient

        _prod_warehouse = WarehouseClient()
    return _prod_warehouse


def _tool_produced(result: Any) -> bool:
    """Did this call actually yield something?

    Stagnation detection needs to tell "searched and found nothing, six
    times" from "did six useful things". An explicit ``ok=False``, an
    error, or an empty collection is not progress.
    """
    if result is None:
        return False
    if isinstance(result, dict):
        if result.get("ok") is False or result.get("error"):
            return False
        for key in ("results", "rows", "tasks", "inbox", "documents",
                    "edges", "memories", "symbols", "tools", "orders",
                    "practices", "matches", "facts"):
            if key in result:
                return bool(result[key])
        return bool(result)
    if isinstance(result, (list, tuple, set, str)):
        return len(result) > 0
    return True


def _result_preview(result: Any, limit: int = 240) -> str:
    """A short, human-readable one-line preview of a tool result for the
    UI activity rail. Prefers a message/error/path field, else a compact
    JSON snippet."""
    try:
        if isinstance(result, dict):
            for key in ("message", "error", "download", "path"):
                if result.get(key):
                    s = str(result[key])
                    return s[:limit]
            if "rows" in result and isinstance(result["rows"], list):
                return f"{len(result['rows'])} rows"
            if "memories" in result and isinstance(result["memories"], list):
                return f"{len(result['memories'])} memories"
            if "tasks" in result and isinstance(result["tasks"], list):
                return f"{len(result['tasks'])} tasks"
            if "inbox" in result and isinstance(result["inbox"], list):
                return f"{len(result['inbox'])} messages"
            s = json.dumps(result, default=str, ensure_ascii=False)
        elif isinstance(result, (list, tuple)):
            return f"{len(result)} items"
        else:
            s = str(result)
    except Exception:  # noqa: BLE001
        return ""
    s = " ".join(s.split())
    return s[:limit] + ("…" if len(s) > limit else "")


class TurnAborted(Exception):
    """A stop was requested; the turn unwinds at its next safe point.

    Deliberately not an error of the model or a tool: it is the
    crew's verdict on the run, and the runner converts it into the
    graceful 'stopped' bookkeeping."""


@dataclass
class VMSession:
    """The long-lived per-user VM state.

    Mirrors the fields the existing ``AlexAgent`` carried, but is
    dataclass-shaped so the SDK and the tests can construct one
    directly.
    """

    session_id: str
    user_email: str
    instance_scope: str
    constitution: ConstitutionalInterceptor
    audit_log: Any  # callable: appends a row to media.alex.agent_actions
    history: list[dict[str, Any]] = field(default_factory=list)
    # The <=80-token decision digest from the last visible-reasoning
    # turn.  Re-injected into the next turn's context (T0 reasoning
    # block); the raw chain-of-thought is only in the archive.
    last_reasoning_digest: str = ""


@dataclass
class HarnessConfig:
    """The configuration the VM reads at construction time."""

    mode: str = "dev"  # 'dev' | 'staging' | 'production'
    executor: Executor | None = None
    browser_os: Any | None = None
    warehouse: Any | None = None
    audit_log: Any | None = None
    constitution: ConstitutionalInterceptor | None = None
    # A hard upper bound on this VM's capabilities, intersected with the
    # user's ACL. It can only narrow, never widen: a standing order's
    # declared list, or a swarm sub-agent's brief, caps what this VM may
    # do below what the person themselves could. None means "the ACL
    # alone decides". Enforced in _allowed_capabilities so it holds for
    # tool exposure AND every later filter that reads it.
    capability_ceiling: frozenset[str] | None = None
    # Plan mode: the agent may look at anything it could normally look
    # at, and change nothing. Implemented as one more ceiling — the
    # read-only predicate below — so it composes with the ACL and any
    # delegated ceiling through the same single enforcement point.
    # The pattern the strong harnesses ship (OpenCode's plan/build
    # stages): draft the approach with full read access, then a person
    # flips to build. Structural, not prompt-level: a mutating tool is
    # not exposed at all, so no phrasing reaches it.
    plan_mode: bool = False
    # A hard upper bound on this VM's per-turn resource budget, applied
    # elementwise over the autonomy-derived Budget. Like the capability
    # ceiling it can only NARROW: a swarm child runs on a slice of its
    # parent's remaining allowance instead of a fresh full budget —
    # without this, one delegate call could legally spend 10x the
    # parent's entire meter.
    budget_ceiling: Any | None = None


def _capability_is_readonly(capability: str) -> bool:
    """Whether a capability can change anything outside the model's own
    context. The list is deny-by-default: an unknown capability is
    treated as mutating, because the cost of wrongly hiding a read tool
    in plan mode is a weaker plan, while the cost of wrongly exposing a
    write tool is plan mode being a lie."""
    cap = (capability or "").lower()
    if cap.endswith(".read") or cap.endswith(".search") or cap.endswith(".query"):
        return True
    # Read-shaped capabilities whose names don't say so.
    if cap in {
        "knowledge.read", "memory.read", "learning.read", "browser.extract",
        "browser.navigate",  # navigation renders; submit/act are separate caps
        "code.read",
    }:
        return True
    return False


#: Tools exposed in plan mode even though their capability is mutating:
#: they write ONLY the plan rows, and producing the plan is the entire
#: point of plan mode. Tool-level, not capability-level, because
#: tasks.update also powers update_task_status — which moves real work.
#: ask_user rides along: clarifying a genuine decision with the crew is
#: exactly what a planning pass should do before the plan hardens, and
#: the tool writes only a question row + a thread message.
_PLAN_MODE_TOOL_EXCEPTIONS = frozenset(
    {"plan_task", "check_plan_step", "ask_user"}
)


class VM:
    """The canonical Alex 2.0 process model.

    A VM is created once per Alex session.  The same VM powers the
    FastAPI ``/api/chat`` route, the ``/api/browser/*`` routes, and
    the SDK's ``Alex.run_turn()``.  The class is intentionally
    synchronous on the public surface (every public method blocks
    until it returns); the executor runs the LLM stream in a
    worker thread, but the VM's interface is plain Python so the
    FastAPI route can wrap it in a thread pool when needed.
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_email: str,
        instance_scope: str = "default",
        config: HarnessConfig | None = None,
    ) -> None:
        self._lock = threading.RLock()
        # One conversation = one turn at a time (see run_turn). RLock so
        # a same-thread re-entry (a tool driving its own VM) cannot
        # deadlock; cross-thread calls queue.
        self._turn_lock = threading.RLock()
        self._governor = None
        cfg = config or HarnessConfig()
        self._config = cfg
        self._session = VMSession(
            session_id=session_id,
            user_email=user_email,
            instance_scope=instance_scope,
            constitution=cfg.constitution
            or _live_constitution(session_id, user_email, cfg.warehouse),
            # Fall back to the prod-aware warehouse (real in production, fake
            # in dev) rather than a throwaway InMemoryWarehouse — otherwise
            # the A3 audit trail is lost in prod whenever no warehouse is
            # injected via config.
            audit_log=cfg.audit_log
            or _DefaultAuditLog(cfg.warehouse or _vm_warehouse()),
        )
        # The executor is per-VM so the LLM connection is per-session.
        # The factory returns a fresh executor; production code may
        # pass one in via ``config.executor`` to share a connection
        # across VMs.
        self._executor: Executor = cfg.executor or executor_for(cfg.mode)
        # Mode-aware: dev gets the in-memory simulator, production gets
        # Playwright. Hardcoding InMemoryBrowserOS here meant production
        # browsed an HTML-string store while reporting success.
        self._browser_os = cfg.browser_os or browser_os_for_mode()
        self._pump = EventPump(name=f"vm:{session_id}")
        # Per-turn tool-chain trace for the workflow summary event
        # (reset by run_turn; appended by _invoke_tool).
        self._turn_tool_calls: list[dict[str, str]] = []
        # Mid-run steering mailbox (see steer/drain_steering). A human
        # on a task crew can course-correct the agent between tool
        # passes; the notes are queued here and drained at safe points
        # in the ReAct loop so no note interrupts an in-flight tool.
        self._steering: list[tuple[str, str]] = []
        # True only while run_turn is executing; steer() is a no-op
        # (returns False) outside a turn so the board knows a post is
        # just a thread message, not a delivered instruction.
        self._turn_active: bool = False
        # Set by request_stop(); checked at the loop's safe points.
        self._abort: bool = False
        # Policy-flagged tools already announced on this VM's stream
        # (supervised sessions warn once per tool, not per call).
        self._policy_said: set[str] = set()
        # Deferred tools this VM has pulled into its working set, in
        # pin order (dict preserves insertion; the oldest is evicted
        # first once the working set is full).
        self._pinned: dict[str, None] = {}
        # MCP discovery cache for the per-turn working-set rebuild.
        self._remote_tools_cache: dict[str, Any] | None = None
        self._remote_tools_at: float = 0.0
        self._agent_def = self._build_agent_def()
        self._closed = False
        # Register the VM's browser OS as the fakes' singleton so
        # the agentic browser tools (which resolve the browser OS
        # via ``alex.testing.get_fakes().browser_os``) find this
        # VM's instance.  We also keep a per-user mapping so the
        # ``_get_browser_os(user_email=...)`` helper can route
        # multi-user test setups.
        from alex import testing as _testing

        fakes = _testing.get_fakes()
        fakes.browser_os = self._browser_os
        fakes.audit_log = self._session.audit_log
        fakes.current_vm_pump = self._pump
        existing = getattr(fakes, "browser_os_per_user", None) or {}
        existing = dict(existing)
        existing[user_email] = self._browser_os
        fakes.browser_os_per_user = existing
        # Seed the fakes' warehouse with the dev schema the
        # ``agentic_browser`` module's audit log needs.  In
        # production the warehouse is a real Unity Catalog and the
        # DDL is declarative.
        register = getattr(fakes.warehouse, "register", None)
        if callable(register):
            try:
                register("media.alex.agent_actions", _AGENT_ACTIONS_COLUMNS)
            except Exception:  # noqa: BLE001
                pass

    # -- properties ------------------------------------------------------
    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def user_email(self) -> str:
        return self._session.user_email

    @property
    def instance_scope(self) -> str:
        return self._session.instance_scope

    @property
    def pump(self) -> EventPump:
        return self._pump

    @property
    def executor(self) -> Executor:
        return self._executor

    @property
    def browser_os(self):
        return self._browser_os

    @property
    def agent_def(self) -> AgentDef:
        return self._agent_def

    # -- subscription ---------------------------------------------------
    def subscribe_sse(self, subscriber) -> Any:
        """Register an SSE subscriber. Returns an unsubscribe callable."""
        return self._pump.subscribe(subscriber)

    # -- mid-run steering -------------------------------------------------
    def steer(self, text: str, *, author: str = "human") -> bool:
        """Queue a steering note for the running turn.

        Returns False when there is no turn to steer — the caller (the
        task board) then knows the message is just a thread post, not a
        course correction, and does not have to guess.

        Notes are delivered at the *next safe point*: between ReAct
        iterations, never mid-tool. A note arriving after the loop has
        finished is not delivered, and that is also reported by the
        return value so nobody believes a note was read when it wasn't.

        The active-check and the append are one atomic step: checking
        the flag before taking the lock left a window where a note was
        accepted just as the turn ended — queued after the final drain,
        never read, and the caller told it was delivered.
        """
        text = (text or "").strip()
        if not text or self._closed:
            return False
        with self._lock:
            if not self._turn_active:
                return False
            self._steering.append((author, text))
        self._pump.emit(steering_event(author=author, text=text, delivered=False))
        return True

    def drain_steering(self) -> list[tuple[str, str]]:
        """Take every queued note. Called by the loop between iterations.

        Doubles as the stop checkpoint: when a stop has been requested,
        this raises TurnAborted at the same safe point, so the loop
        unwinds between iterations instead of a "stopped" run continuing
        to make model passes and execute tools."""
        with self._lock:
            if self._abort:
                raise TurnAborted()
            out = self._steering
            self._steering = []
        return out

    def request_stop(self, reason: str = "stopped by the crew") -> bool:
        """Abort the running turn at its next safe point.

        Idempotent and honest: returns False when there is nothing
        running to abort. Stop is not an interrupt-kernel — the current
        tool completes (killing one mid-flight is its own failure
        class), but no further model pass happens and no further tool
        runs."""
        with self._lock:
            if not self._turn_active or self._abort:
                return False
            self._abort = True
        self._pump.emit(error_event("run_stopping", f"stop requested: {reason}"))
        return True

    @property
    def abort_requested(self) -> bool:
        with self._lock:
            return self._abort

    def credit_wait(self, seconds: float) -> None:
        """Credit time this turn spent parked on a human (ask_user) back
        to the governor's wall clock. No-op before the first turn."""
        gov = self._governor
        if gov is not None:
            try:
                gov.credit_wait(seconds)
            except Exception:  # noqa: BLE001
                logger.debug("wait credit failed", exc_info=True)

    def last_governor_state(self) -> dict | None:
        """The most recent turn's governor accounting (budgets, counters,
        halt verdict) — what the run consumed, for callers deciding
        whether a turn ended because the WORK ended or the BUDGET did."""
        gov = self._governor
        try:
            return gov.state() if gov is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _end_turn_steering(self) -> list[tuple[str, str]]:
        """Close the mailbox atomically with the turn flag.

        The turn thread calls this once, under the lock: flag off and
        mailbox emptied as one step, so a concurrent steer() either
        sees the flag up (and its note is in the returned list) or
        sees it down (and returns False honestly). No note can land
        between the two.
        """
        with self._lock:
            self._turn_active = False
            out = self._steering
            self._steering = []
        return out

    def _policy_gate(self, request, spec, governor):
        """The allow/ask/deny decision for one tool call, applied.

        Returns a refusal payload to hand the model, or None to
        proceed. Never raises — a policy-layer bug must fail as a
        refusal of THIS call, not as a crashed turn.
        """
        try:
            from agent import tool_policy

            verdict = tool_policy.decision(spec.name, spec.capability)
            if verdict == tool_policy.ALLOW:
                return None
            if verdict == tool_policy.DENY:
                if governor is not None:
                    governor.record_denial(
                        request.name, spec.capability, "workspace policy"
                    )
                self._pump.emit(error_event(
                    "policy_denied",
                    f"{request.name} is denied by workspace policy",
                ))
                self._session.audit_log(
                    user_email=self._session.user_email,
                    session_id=self._session.session_id,
                    tool=request.name,
                    capability=spec.capability,
                    args=request.args or {},
                    outcome="denied:policy",
                )
                self._record_workflow_step(request.name, "denied")
                return {
                    "ok": False,
                    "error": (
                        f"{request.name} is denied by workspace policy. "
                        "Do not look for another route to the same action; "
                        "tell the user an operator would have to change "
                        f"tool_policy.{spec.name}."
                    ),
                }
            # ASK.
            from agent.task_plan import task_for_session

            task_id = task_for_session(self.session_id)
            if task_id:
                from agent import task_questions
                from agent import tool_policy as tp

                approval = task_questions.request_approval(
                    task_id,
                    tp.approval_question(spec.name, request.args or {}),
                    vm=self,
                )
                self._session.audit_log(
                    user_email=self._session.user_email,
                    session_id=self._session.session_id,
                    tool=request.name,
                    capability=spec.capability,
                    args=request.args or {},
                    outcome=(
                        "approved:policy" if approval.get("approved")
                        else "denied:policy_no_approval"
                    ),
                )
                if approval.get("approved"):
                    return None
                if governor is not None:
                    governor.record_denial(
                        request.name, spec.capability, "approval refused"
                    )
                who = approval.get("answered_by")
                return {
                    "ok": False,
                    "error": (
                        f"{request.name} was not approved"
                        + (f" ({who} said no)" if who else
                           f" ({approval.get('reason', 'no approval')})")
                        + ". Do not retry it or route around it; work what "
                        "you can without this action and say in your "
                        "summary that it awaits a human decision."
                    ),
                }
            # Supervised (chat/worker) session: proceed, but say it.
            if spec.name not in self._policy_said:
                self._policy_said.add(spec.name)
                self._pump.emit(governor_event(
                    action="warn",
                    reason=(
                        f"{spec.name} is a policy-flagged action; it ran "
                        "because this session is supervised. Operators can "
                        f"require approval via tool_policy.{spec.name}."
                    ),
                    pathology="policy",
                ))
            return None
        except Exception:  # noqa: BLE001
            logger.exception("tool policy gate failed; refusing the call")
            return {
                "ok": False,
                "error": (
                    f"the policy layer could not evaluate {request.name}; "
                    "the call was refused (fail closed). Try a different "
                    "approach or report the problem."
                ),
            }

    # -- tool invocation ------------------------------------------------
    def pin_tools(self, names: list[str]) -> list[str]:
        """Make deferred tools callable now: they join this VM's working
        set immediately (the bridge re-reads the set every pass) and
        stay pinned for later turns.

        Applies the same exposure filters as ``_build_agent_def`` (ACL
        capability, plan mode) and keeps the pinned set bounded — the
        oldest pin is evicted once ``MAX_DEFERRED_IN_PROMPT`` is
        reached, so a long session cannot inflate every model pass.
        Returns the names actually pinned.
        """
        from agent.omnigent import FunctionTool

        allowed = self._allowed_capabilities()
        plan_mode = bool(self._config.plan_mode)
        pinned: list[str] = []
        for name in names:
            spec = TOOL_REGISTRY.get(name)
            if spec is None or not getattr(spec, "deferred", False):
                continue
            if allowed is not None and spec.capability not in allowed:
                continue
            if plan_mode and not (
                _capability_is_readonly(spec.capability)
                or spec.name in _PLAN_MODE_TOOL_EXCEPTIONS
            ):
                continue
            # Re-pinning refreshes recency (dict insertion order).
            self._pinned.pop(name, None)
            self._pinned[name] = None
            while len(self._pinned) > MAX_DEFERRED_IN_PROMPT:
                oldest = next(iter(self._pinned))
                del self._pinned[oldest]
            if self._agent_def is not None and name not in self._agent_def.tools:
                self._agent_def.tools[name] = FunctionTool(
                    name=spec.name, description=spec.description,
                    callable=spec.callable, scopes={spec.capability},
                    input_schema=getattr(spec, "input_schema", None),
                )
            pinned.append(name)
        return pinned

    def _focused_deferred(self, focus: str) -> set[str]:
        """Deferred tools that earn a place in this turn: pinned ones plus
        the best lexical matches to the request, capped."""
        chosen = set(self._pinned)
        words = {w for w in (focus or "").lower().replace("_", " ").split() if len(w) >= 3}
        if words:
            ranked = sorted(
                ((score_tool(spec, words), spec.name)
                 for spec in TOOL_REGISTRY.values() if getattr(spec, "deferred", False)),
                key=lambda t: -t[0],
            )
            for score, name in ranked:
                if score <= 0 or len(chosen) >= MAX_DEFERRED_IN_PROMPT:
                    break
                chosen.add(name)
        return chosen

    def _build_agent_def(self, focus: str = "") -> AgentDef:
        """Build the :class:`AgentDef` from the tool registry."""
        from agent.omnigent import FunctionTool

        def _session_getter() -> VMSession:
            return self._session

        adapter = ConstitutionAdapter(
            session_getter=_session_getter,
            on_block=lambda payload, exc: self._pump.emit(
                law_block_event(
                    article=getattr(exc, "article_id", "?"), detail=str(exc)
                )
            ),
        )

        allowed = self._allowed_capabilities()
        plan_mode = bool(self._config.plan_mode)
        focused = self._focused_deferred(focus)
        defs: dict[str, Any] = {}
        for spec in TOOL_REGISTRY.values():
            if getattr(spec, "deferred", False) and spec.name not in focused:
                continue
            # Expose only tools the user can actually run. This is both
            # defense in depth (guardrails still gate at call time) and a
            # big latency win: every excluded schema is prompt tokens the
            # LLM doesn't have to re-read on each call.
            if allowed is not None and spec.capability not in allowed:
                continue
            # Plan mode: read-only working set plus the plan tools.
            if plan_mode and not (
                _capability_is_readonly(spec.capability)
                or spec.name in _PLAN_MODE_TOOL_EXCEPTIONS
            ):
                continue
            defs[spec.name] = FunctionTool(
                name=spec.name,
                description=spec.description,
                callable=spec.callable,
                scopes={spec.capability},
                input_schema=getattr(spec, "input_schema", None),
            )
        # Remote tools fill whatever room is left after the built-ins.
        # Built-ins win the space: they are governed, audited and known,
        # and a remote duplicate of one is pure prompt cost.
        # Plan mode exposes NO remote tools: we cannot prove a remote
        # tool is read-only, and unknown means mutating here.
        builtin_names = set(defs)
        room = 0 if plan_mode else max(MAX_TOOLS_IN_PROMPT - len(defs), 0)
        try:
            remote = {} if plan_mode else self._cached_mcp_tools()
        except Exception:  # noqa: BLE001
            # Belt and braces around the discovery path: a broken
            # integration must never cost the agent its own tools.
            logger.warning("MCP discovery failed; continuing with built-ins")
            remote = {}
        added = 0
        for name, tool in remote.items():
            if added >= room:
                break
            parsed = name.split("__")
            if len(parsed) >= 3 and parsed[-1] in builtin_names:
                # The same capability already exists natively; exposing it
                # twice teaches the model to pick at random.
                continue
            if defs.setdefault(name, tool) is tool:
                added += 1
        if len(remote) > added:
            logger.info(
                "exposing %d of %d remote tools (%d built-in already in the "
                "working set, cap %d)",
                added, len(remote), len(builtin_names), MAX_TOOLS_IN_PROMPT,
            )
        return AgentDef(
            name="alex",
            prompt=PERSONA,
            tools=defs,
            policies={"constitution": adapter},
        )

    def _cached_mcp_tools(self) -> dict[str, Any]:
        """``_mcp_tools`` with a short-lived per-VM cache.

        Deferred admission rebuilds the working set on every turn; MCP
        discovery (principal + every server) must not ride along. The
        call itself still re-resolves authorization, so a revoked
        capability is enforced at the next call.
        """
        now = time.time()
        if (
            self._remote_tools_cache is not None
            and now - self._remote_tools_at < MCP_DISCOVERY_TTL_S
        ):
            return self._remote_tools_cache
        tools = self._mcp_tools()
        self._remote_tools_cache = tools
        self._remote_tools_at = now
        return tools

    def _mcp_tools(self) -> dict[str, Any]:
        """The remote MCP tools this user's VM should expose.

        Each becomes an ordinary :class:`FunctionTool`, so a remote tool
        is gated, audited and streamed exactly like a built-in one — the
        model cannot tell the difference, and neither can the guardrails.
        A discovery failure yields no tools rather than failing the turn.
        """
        from agent.omnigent import FunctionTool

        try:
            from agent import access, mcp_client

            principal = access.load_principal(
                _module_sql, self._session.user_email
            )
            remote = mcp_client.tools_for(_module_sql, principal)
        except Exception:  # noqa: BLE001
            logger.debug("MCP discovery unavailable", exc_info=True)
            return {}

        out: dict[str, Any] = {}
        for tool in remote[:MAX_REMOTE_TOOLS]:
            out[tool.name] = FunctionTool(
                name=tool.name,
                description=tool.description,
                callable=self._mcp_caller(tool),
                scopes={tool.capability},
            )
        if len(remote) > MAX_REMOTE_TOOLS:
            logger.warning(
                "%d remote tools available; exposing the first %d — "
                "model accuracy degrades past this point",
                len(remote), MAX_REMOTE_TOOLS,
            )
        return out

    def _mcp_caller(self, tool):
        """Bind one remote tool to a synchronous callable.

        Authorization is re-resolved at call time, not captured at
        discovery time: a capability revoked mid-session must take
        effect on the next call, not the next login.
        """

        def _call(**kwargs: Any) -> Any:
            from agent import access, mcp_client

            principal = access.load_principal(
                _module_sql, self._session.user_email
            )
            return mcp_client.call(_module_sql, principal, tool.name, kwargs)

        _call.__name__ = tool.name
        _call.__doc__ = tool.description
        return _call

    def _lazy_governor(self):
        """The session's governor, built on first use.

        Only `run_turn` used to build one, so every other way into a
        tool ran ungoverned: the `/api/tools/{name}` route, the SDK,
        and a LangGraph node calling through `langchain_tools`. Those
        are runs too — a caller looping on the invoke endpoint is
        exactly the shape this layer exists to stop. Absence of a
        governor must never mean absence of governance, so one is
        built on first use and then shared for the session.
        """
        if self._governor is None:
            self._governor = self._make_governor(
                goal=f"direct tool use in session {self._session.session_id}"
            )
        return self._governor

    def _make_governor(self, *, goal: str):
        """Build this turn's governor, sized by the user's autonomy level.

        The autonomy preference is a real control, not a label: it moves
        the tool-call, turn and time ceilings, so "suggest" genuinely
        stops sooner than "autopilot".
        """
        from agent.execution_governor import Budget, ExecutionGovernor

        autonomy = "co-pilot"
        try:
            from agent import preferences

            autonomy = preferences.autonomy_of(
                _module_sql, self._session.user_email
            )
        except Exception:  # noqa: BLE001
            logger.debug("autonomy preference unavailable", exc_info=True)

        def _emit(event: dict[str, Any]) -> None:
            # Governor decisions go into the same stream as tool calls and
            # retrievals, so a halted run explains itself where the user
            # is already looking rather than appearing to stall.
            self._pump.emit(
                governor_event(
                    action=event.get("action", ""),
                    reason=event.get("reason", ""),
                    pathology=event.get("pathology", ""),
                    detail=event.get("detail") or {},
                    state=event.get("state") or {},
                )
            )

        budget = Budget.for_autonomy(autonomy)
        ceiling = self._config.budget_ceiling
        if ceiling is not None:
            # Narrow only. Zero-or-negative ceiling fields mean "no
            # opinion" (max_total_tokens uses 0 as off, so a positive
            # ceiling engages the token bound even when the base is off).
            budget.max_seconds = min(budget.max_seconds, ceiling.max_seconds)
            budget.max_tool_calls = min(
                budget.max_tool_calls, ceiling.max_tool_calls
            )
            budget.max_llm_turns = min(
                budget.max_llm_turns, ceiling.max_llm_turns
            )
            if ceiling.max_total_tokens > 0:
                budget.max_total_tokens = (
                    min(budget.max_total_tokens, ceiling.max_total_tokens)
                    if budget.max_total_tokens > 0
                    else ceiling.max_total_tokens
                )
        return ExecutionGovernor(
            goal=goal,
            budget=budget,
            run_id=self._session.session_id,
            on_event=_emit,
        )

    @property
    def governor(self):
        """This turn's governor, for tests and the oversight surface."""
        return self._governor

    def _allowed_capabilities(self) -> set[str] | None:
        """The user's effective capability set, or None when the ACL
        store is unavailable (offline dev without a seed) — in which
        case every registered tool stays exposed and the guardrails
        gate remains the enforcement point.

        The lookup is cached per (user, warehouse) for
        :data:`_ACL_TTL_S` seconds so a VM rebuild (or a burst of VMs
        for the same user) doesn't re-query the ACL each time. The
        warehouse half of the key is a *stable* identity (module-level
        singleton identity, falling back to the repr): ``id(wh)``
        changes meaning when an object is freed and a new one lands on
        the same address, which would serve one warehouse's cached
        capabilities for a different warehouse.
        """
        try:
            wh = _vm_warehouse()
        except Exception:  # noqa: BLE001
            # The same fail-closed rule as _query_capabilities, one
            # level up: in production, failing to *acquire* the ACL
            # warehouse (bad config, credential error) must deny every
            # tool, not expose them all. Only offline dev stays
            # permissive, where "no warehouse" means "no seed".
            from alex.testing import is_dev

            if is_dev():
                logger.debug(
                    "no ACL warehouse in dev; leaving the tool list unfiltered",
                    exc_info=True,
                )
                return None
            logger.error(
                "ACL warehouse unavailable for %s — refusing every tool. "
                "An agent that can do nothing is the correct outcome of a "
                "control-plane outage.",
                self._session.user_email, exc_info=True,
            )
            return set()
        now = time.monotonic()
        wh_key = self._warehouse_cache_key(wh)
        key = (self._session.user_email, wh_key)
        with _ACL_LOCK:
            # Evict expired entries on read: the cache only ever grew,
            # so a long-lived process accumulated an entry per user
            # forever.
            for k in [
                k for k, (ts, _c) in _ACL_CACHE.items()
                if now - ts >= _ACL_TTL_S
            ]:
                _ACL_CACHE.pop(k, None)
            cached = _ACL_CACHE.get(key)
            if cached is not None and now - cached[0] < _ACL_TTL_S:
                return self._apply_ceiling(cached[1])
        caps = self._query_capabilities(wh)
        with _ACL_LOCK:
            _ACL_CACHE[key] = (now, caps)
        return self._apply_ceiling(caps)

    def _apply_ceiling(self, caps: set[str] | None) -> set[str] | None:
        """Intersect the ACL with this VM's configured ceiling.

        The ceiling is what makes a delegated context (a standing order,
        a swarm sub-agent) run with LESS than the person could — the
        intersection can only narrow. When the ACL store is permissive
        (dev, caps is None) the ceiling alone binds, because a ceiling
        that evaporates exactly when the ACL is absent protects nothing.
        The cache stores the RAW lookup, so two VMs for the same user
        with different ceilings cannot poison each other.
        """
        ceiling = self._config.capability_ceiling
        if ceiling is None:
            return caps
        if caps is None:
            return set(ceiling)
        return caps & ceiling

    @staticmethod
    def _warehouse_cache_key(wh) -> str:
        """A stable identity for the ACL warehouse.

        The module-level singleton path returns one long-lived object,
        so its ``id`` is stable for the process life — but an injected
        or short-lived warehouse would have its ``id`` reused by an
        unrelated successor, pairing stale capabilities with a new
        store. The repr encodes type plus identity, which is stable
        for a live object and never collides with a successor.
        """
        return f"{type(wh).__module__}.{type(wh).__qualname__}@{id(wh)}"

    def _query_capabilities(self, wh) -> set[str] | None:
        """The user's granted capabilities.

        ``None`` means *do not filter* and is returned only where that is
        safe. It used to be returned in two cases that are not:

        * **No rows.** A user with no entitlements got ``None`` and was
          therefore offered every tool — the exact inverse of the intent.
          An empty grant is a real answer: it means nothing.
        * **A failed query.** Any exception looked like "offline dev
          without a seed", so a control-plane outage in production
          escalated every user to every capability, including
          ``run_python`` and ``vm_exec``.

        Both now deny. The VM filters the tool list AND re-checks the
        capability at call time (`_invoke_tool`), so a direct API/SDK
        call cannot bypass the ACL by skipping exposure; this lookup is
        the single source both paths use. It has to fail closed.
        The permissive answer survives for exactly one case: dev with no
        ACL table at all, where the alternative is a local agent that can
        do nothing.
        """
        try:
            res = wh.execute(
                "SELECT capability FROM media.control.effective_employee_acl "
                "WHERE user_email = :u",
                {"u": self._session.user_email},
            )
            rows = res.as_dicts()
        except Exception:  # noqa: BLE001
            from alex.testing import is_dev

            if is_dev():
                logger.debug(
                    "ACL store unreadable in dev; leaving the tool list "
                    "unfiltered", exc_info=True,
                )
                return None
            logger.error(
                "ACL store unreadable for %s — refusing every tool. An "
                "agent that can do nothing is the correct outcome of a "
                "control-plane outage.",
                self._session.user_email, exc_info=True,
            )
            return set()
        return {r["capability"] for r in rows if r.get("capability")}

    def _invoke_tool(self, request: ToolCallRequest) -> Any:
        """Run a single tool call.  The VM is the single point that:

        1. Resolves the tool by name from :data:`TOOL_REGISTRY`.
        2. Emits a ``tool`` event through the pump.
        3. Audit-logs the call (A3).
        4. Calls the function.
        5. Emits a final ``artifact`` event with the result.
        """
        spec: ToolSpec | None = TOOL_REGISTRY.get(request.name)
        if spec is None and request.name.startswith("mcp__"):
            # Remote MCP tools live in the AgentDef, not the registry —
            # and resolving only the registry meant every MCP call the
            # MODEL made came back unknown_tool: the tools were shown
            # and never runnable. A transient spec routes them through
            # the same governor/constitution/audit path as built-ins.
            # untrusted_output is unconditional: the result is produced
            # by a server this process does not control.
            remote_tool = (self._agent_def.tools or {}).get(request.name)
            if remote_tool is not None:
                spec = ToolSpec(
                    name=request.name,
                    capability=next(
                        iter(getattr(remote_tool, "scopes", None) or []),
                        "mcp.use",
                    ),
                    callable=remote_tool.callable,
                    description=remote_tool.description or request.name,
                    untrusted_output=True,
                )
        if spec is None:
            # Unknown tools are still steps the run took: record them
            # with the governor or a caller looping on garbage names is
            # invisible to every bound (the exact threat the layer
            # exists for), and audit the attempt so oversight sees it.
            self._pump.emit(
                error_event(
                    "unknown_tool", f"tool {request.name!r} is not in the registry"
                )
            )
            governor = self._lazy_governor()
            if governor is not None:
                governor.record(
                    kind="tool", name=request.name, ok=False, produced=False
                )
            self._record_workflow_step(request.name, "error")
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability="unknown",
                args=request.args or {},
                outcome="error:unknown_tool",
            )
            return {"ok": False, "error": "unknown_tool"}

        # Dual-ACL at CALL time, not only at exposure time. Built-in
        # tools re-check inside their guarded() wrapper; adapted tools
        # (Jarvis's native tools) are raw callables with no wrapper, so
        # this is their only ACL gate — and a direct invoke (the
        # /api/tools route, the SDK) never passes through exposure.
        allowed = self._allowed_capabilities()
        if allowed is not None and spec.capability not in allowed:
            self._pump.emit(error_event(
                "missing_capability",
                f"{request.name} needs {spec.capability}, which this user lacks",
            ))
            self._record_workflow_step(request.name, "denied")
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome="denied:missing capability",
            )
            return {
                "ok": False,
                "error": f"missing capability {spec.capability}",
            }

        # Plan mode holds at CALL time too, not only at exposure time:
        # exposure filtering is what the model sees, but a direct invoke
        # (the /api/tools route, the SDK) reaches this method without
        # ever seeing the filtered list.
        if self._config.plan_mode and not (
            _capability_is_readonly(spec.capability)
            or request.name in _PLAN_MODE_TOOL_EXCEPTIONS
        ):
            self._pump.emit(error_event(
                "plan_mode", f"{request.name} changes things; this session "
                "is in plan mode (read + plan only)",
            ))
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome="denied:plan_mode",
            )
            return {
                "ok": False,
                "error": "plan mode: this session may read and plan, not "
                         "act. Put the step in the plan instead.",
            }

        # Stop gate (crew verdict): a stopped run executes nothing
        # further. Checked before the governor so a halt-after-stop
        # reads as "stopped", not a pathology, in the audit trail.
        if self.abort_requested:
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome="denied:run stopped",
            )
            return {
                "ok": False,
                "stopped": True,
                "error": "the crew stopped this run; no further tools run",
                "guidance": (
                    "The run has been stopped. Do not attempt further "
                    "actions; summarize what was completed."
                ),
            }

        # Governor (L1.5): is this RUN still behaving like work?
        # Distinct from the ACL and the constitution, which ask whether an
        # action is permitted. Every step here may be individually allowed
        # while the run as a whole has started looping, drifting, or
        # working around a refusal.
        governor = self._lazy_governor()
        if governor is not None:
            verdict = governor.before_tool(
                request.name, request.args or {}, spec.capability
            )
            if verdict.halting:
                self._pump.emit(
                    error_event("run_halted", verdict.reason)
                )
                self._session.audit_log(
                    user_email=self._session.user_email,
                    session_id=self._session.session_id,
                    tool=request.name,
                    capability=spec.capability,
                    args=request.args or {},
                    outcome=f"halted:{verdict.pathology}",
                )
                return {
                    "ok": False,
                    "halted": True,
                    "pathology": verdict.pathology,
                    "error": verdict.reason,
                    "guidance": (
                        "Stop and report what you have so far, including what "
                        "is blocking you. Do not try another route to the same "
                        "action."
                    ),
                }

        # Constitution (L2): evaluate the adapter before executing.
        # A DENY (or a raised ConstitutionalBlock) stops the call.
        denial = self._constitution_denial(request, spec)
        if denial is not None:
            reason, article = denial
            if governor is not None:
                governor.record_denial(
                    request.name, spec.capability, reason
                )
            self._pump.emit(error_event("constitution_block", reason))
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome=f"denied:{reason}",
            )
            self._record_workflow_step(request.name, "denied")
            return denial_payload(reason, article=article)

        # Workspace tool policy (L2.5): allow / ask / deny. The
        # risk-graded layer the strong harnesses ship — deny refuses
        # outright; ask parks an UNATTENDED board run on a one-click
        # crew approval (fail closed), while a supervised session
        # proceeds with the act said out loud on the stream.
        policy_refusal = self._policy_gate(request, spec, governor)
        if policy_refusal is not None:
            return policy_refusal

        # Resolve kwargs from the request.  Every browser tool in
        # the registry takes ``session_id``, ``tab_id``, ``user_email``,
        # plus its own args.  The convention is the tool's
        # ``__call__`` accepts keyword arguments; missing ones are
        # filled in by the VM.
        # Validate args against the tool's signature BEFORE the audit row
        # and the tool event, so a malformed call (a) comes back to the
        # model as a correctable error instead of a Python TypeError,
        # (b) is recorded as a failed step with the governor — a caller
        # hammering malformed args used to be invisible to every bound —
        # and (c) does not leave an orphaned "started" audit row that
        # makes the call look in-flight forever.
        problem = _validate_tool_args(spec, request.args or {})
        if problem is not None:
            if governor is not None:
                governor.record(
                    kind="tool", name=request.name,
                    args=request.args or {}, capability=spec.capability,
                    ok=False, produced=False,
                )
            self._pump.emit(error_event("invalid_tool_args", problem))
            self._record_workflow_step(request.name, "error")
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome=f"error:invalid args: {problem}",
            )
            return {"ok": False, "error": f"invalid arguments: {problem}"}

        # Audit: log the start of the call (A3).
        self._session.audit_log(
            user_email=self._session.user_email,
            session_id=self._session.session_id,
            tool=request.name,
            capability=spec.capability,
            args=request.args or {},
            outcome="started",
        )
        self._pump.emit(tool_event(request.name, request.args or {}))

        try:
            kwargs = dict(request.args or {})
            # Identity kwargs are ALWAYS the session's — a model-supplied
            # user_email/instance_scope must never win (impersonation).
            kwargs["user_email"] = self._session.user_email
            kwargs["instance_scope"] = self._session.instance_scope
            # The actual guardrails gate lives inside the tool
            # implementation (it inspects TOOL_CAPABILITY and
            # checks media.control.employee_acl).  The VM doesn't
            # double-gate; the constitution interceptor already
            # ran in the L2 layer.
            # Tools that themselves need the invoking VM (delegation)
            # find it via this contextvar rather than a parameter the
            # model could set. Reset in all cases so a pooled thread
            # cannot leak one VM's identity into another's call.
            from agent import swarm as _swarm

            _vm_token = _swarm.set_current_vm(self)
            try:
                result = spec.callable(**kwargs)
            finally:
                _swarm.reset_current_vm(_vm_token)
        except ConstitutionalBlock as exc:
            if governor is not None:
                governor.record_denial(request.name, spec.capability, str(exc))
            self._pump.emit(
                law_block_event(
                    article=getattr(exc, "article_id", "?"), detail=str(exc)
                )
            )
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome=f"denied:{exc}",
            )
            self._record_workflow_step(request.name, "denied")
            return denial_payload(
                str(exc), article=getattr(exc, "article_id", None)
            )
        except Denied as exc:
            if governor is not None:
                governor.record_denial(request.name, spec.capability, str(exc))
            self._pump.emit(error_event("acl_denied", str(exc)))
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome=f"denied:{exc}",
            )
            self._record_workflow_step(request.name, "denied")
            return denial_payload(
                str(exc),
                how_to_proceed=(
                    f"The user lacks the {spec.capability!r} capability. "
                    "Do not retry. Tell the user which permission is "
                    "missing, or use a tool they are allowed to run."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if governor is not None:
                governor.record(
                    kind="tool", name=request.name, args=request.args,
                    capability=spec.capability, ok=False, produced=False,
                )
            self._pump.emit(error_event("tool_error", str(exc)))
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool=request.name,
                capability=spec.capability,
                args=request.args or {},
                outcome=f"error:{exc}",
            )
            self._record_workflow_step(request.name, "error")
            return {"ok": False, "error": str(exc)}

        if governor is not None:
            # `produced` is what stagnation detection keys on: a call that
            # created, changed or returned something is progress; one that
            # came back empty or with ok=False is not.
            governor.record(
                kind="tool",
                name=request.name,
                args=request.args,
                capability=spec.capability,
                ok=True,
                produced=_tool_produced(result),
                untrusted=spec.untrusted_output,
                # The observation digest — what tells the stuck detector
                # "same call, same answer" (a loop) from "same call, new
                # answer" (polling).
                result=result,
            )
        self._session.audit_log(
            user_email=self._session.user_email,
            session_id=self._session.session_id,
            tool=request.name,
            capability=spec.capability,
            args=request.args or {},
            outcome="ok",
            result=result,
        )
        # Feed the interceptor's per-session procedure state (A-series
        # procedure articles check required prior steps).
        try:
            record = getattr(self._session.constitution, "record", None)
            if callable(record):
                record(request.name, request.args or {}, result)
        except Exception:  # noqa: BLE001
            logger.exception("constitution record failed")
        # Surface the result as an artifact WITH a compact preview so the
        # UI activity rail can show what the tool returned (like a coding
        # agent showing command output).
        self._pump.emit(
            artifact_event(
                path=f"tool://{request.name}",
                kind="tool_result",
                preview=_result_preview(result),
            )
        )
        self._record_workflow_step(request.name, "ok")
        # The registry's contract: a deferred tool also enters a turn
        # "by having been used before". Pin on successful use (bounded
        # and ACL/plan-filtered inside pin_tools) so a multi-turn task
        # does not lose the tool it just proved it needs.
        if getattr(spec, "deferred", False):
            try:
                self.pin_tools([spec.name])
            except Exception:  # noqa: BLE001
                logger.debug("pin-after-use failed", exc_info=True)
        return result

    def _record_workflow_step(self, tool: str, outcome: str) -> None:
        """Append one step to the per-turn tool-chain trace (item for
        the workflow-summary artifact).  Never raises."""
        try:
            self._turn_tool_calls.append({"tool": tool, "outcome": outcome})
        except Exception:  # noqa: BLE001
            logger.debug("workflow step record failed", exc_info=True)

    def _emit_workflow_summary(self) -> None:
        """After a turn that used >=2 tools, emit ONE compact artifact
        event summarizing the tool chain (names in order + outcome) so
        the UI inspector and the audit trail tell the workflow story."""
        steps = list(self._turn_tool_calls)
        if len(steps) < 2:
            return
        try:
            chain = " -> ".join(
                f"{s['tool']}:{s['outcome']}" for s in steps
            )
            ev = artifact_event(
                path=f"workflow://{self._session.session_id}",
                kind="workflow_summary",
            )
            ev.extras["workflow"] = {
                "tools": [s["tool"] for s in steps],
                "steps": steps,
                "chain": chain[:600],
            }
            self._pump.emit(ev)
        except Exception:  # noqa: BLE001
            logger.debug("workflow summary emit failed", exc_info=True)

    def _constitution_denial(
        self, request: ToolCallRequest, spec: ToolSpec
    ) -> tuple[str, str | None] | None:
        """Evaluate the constitution adapter for a tool call.

        Returns ``(reason, article_id_or_None)`` when the policy says
        DENY, or ``None`` when the call is allowed (or no adapter is
        wired).  A crash inside the policy fails closed.
        """
        policy = (self._agent_def.policies or {}).get("constitution")
        fn = getattr(policy, "callable", None)
        if not callable(fn):
            return None
        event = {
            "type": "tool_call",
            "target": request.name,
            "data": {"tool": request.name, "args": request.args or {}},
            "context": {"actor": {}, "labels": {}},
        }
        try:
            result = fn(event)
        except ConstitutionalBlock as exc:
            return str(exc), getattr(exc, "article_id", None)
        except Exception:  # noqa: BLE001
            logger.exception("constitution evaluation failed; failing closed")
            return "constitution evaluation failed", None
        action = getattr(result, "action", None)
        reason = getattr(result, "reason", None)
        article = getattr(result, "article_id", None)
        if isinstance(result, dict):
            action = result.get("action", action)
            reason = result.get("reason", reason)
            article = result.get("article_id", article)
        action_name = str(getattr(action, "value", action) or "").lower()
        if action_name == "deny":
            return str(reason or "denied by constitution"), article
        return None

    # -- chat turn -------------------------------------------------------
    def run_turn(
        self,
        *,
        text: str,
        history: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        show_reasoning: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Serialized entry point for one chat turn.

        A VM is one conversation (and one working set, governor, and
        session history). Two overlapping calls — a double-submit, two
        tabs, a route plus a task runner — used to interleave those
        mutable fields: replies crossed, governors were replaced
        mid-run, and messages landed out of order. Turns on one VM are
        therefore serialized; different VMs (the swarm's workers) still
        run in parallel.
        """
        with self._turn_lock:
            return self._run_turn_locked(
                text=text,
                history=history,
                system_prompt=system_prompt,
                show_reasoning=show_reasoning,
            )

    def _run_turn_locked(
        self,
        *,
        text: str,
        history: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        show_reasoning: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run one chat turn and return ``(final_response, history)``.

        The caller passes the user's *new* message text; the VM
        appends it to the session history, runs a ReAct loop, and
        returns the updated history (so the SDK can persist it).

        When ``show_reasoning`` is true (off by default), the VM
        emits ``reasoning`` SSE events for the turn, archives the
        full trace to ``media.control.thinking_traces`` via
        :func:`agent.thinking.archive`, and stores a <=80-token
        digest that the *next* turn re-injects into context.
        """
        if self._closed:
            raise RuntimeError("VM is closed")
        # One governor per turn: bounds and pathology detection are
        # per-run, so a long session cannot accumulate its way past them.
        self._governor = self._make_governor(goal=text)
        if any(getattr(sp, "deferred", False) for sp in TOOL_REGISTRY.values()):
            self._agent_def = self._build_agent_def(focus=text)
        if history is None:
            history = list(self._session.history)
        history.append(
            {
                "role": "user",
                "content": text,
                "metadata": {"session_id": self._session.session_id},
            }
        )
        if show_reasoning:
            from alex.events import reasoning_event

            self._pump.emit(
                reasoning_event("plan", f"Considering the request: {text[:200]}")
            )
        if system_prompt is None:
            system_prompt = self._build_system_prompt(
                text, show_reasoning=show_reasoning
            )
        history = self._compact_history(history)
        self._turn_tool_calls = []
        with self._lock:
            self._steering = []
            # A fresh turn clears any prior stop: a VM reused for the
            # next turn (continuation run, new request on the same
            # session) works again.
            self._abort = False
        loop = ReactLoop(
            executor=self._executor,
            agent_def=self._agent_def,
            history=history,
            system_prompt=system_prompt,
            pump=self._pump,
            constitution_adapter=self._agent_def.policies["constitution"],
            poll_steering=self.drain_steering,
            # llm_turns was a permanent zero before this hook existed:
            # the budget clause could never fire and oversight reported
            # every run as having thought zero times.
            on_llm_pass=(
                lambda usage=None: self._governor.record_llm_pass(usage)
            )
            if self._governor is not None else None,
        )
        # Hand the loop a back-reference so its _dispatch_tool_call
        # can reach the VM's _invoke_tool.
        loop._vm = self  # type: ignore[attr-defined]
        with self._lock:
            self._steering = []
            self._turn_active = True
        try:
            # The governor's budget is THE authority on how long a run
            # may think: the loop's iteration cap comes from it, so
            # "autopilot" genuinely gets more passes than "suggest"
            # instead of both hitting a hard-coded 8 first.
            max_turns = 8
            try:
                max_turns = int(self._governor.budget.max_llm_turns)
            except Exception:  # noqa: BLE001
                logger.debug("governor budget unavailable; using default")
            last, _ = loop.run(max_turns=max_turns)
        finally:
            # Atomic with the flag: a note that won the race into the
            # mailbox in the instant the loop was ending is drained
            # here or not accepted at all — never accepted-then-lost.
            # Say so on the stream; the next turn clears the mailbox,
            # so this is its only chance to surface.
            for author, note in self._end_turn_steering():
                self._pump.emit(
                    steering_event(
                        author=author, text=note, delivered=False,
                    )
                )
                self._pump.emit(
                    error_event(
                        "steering_not_delivered",
                        f"a note from {author} arrived as the turn was "
                        "ending and was not read; it is in the task "
                        "thread and the next run will see it",
                    )
                )
        last = self._inline_review(last)
        self._emit_workflow_summary()
        self._session.history = list(history)
        self._capture_learned_fact(text)
        if show_reasoning:
            self._finish_visible_reasoning(text, last)
        return last, history

    def _compact_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Budgeted context compaction (agent/harness/compaction.py):
        aux-model summary of the middle, protected head (the task) and
        tail (the working set), stale tool payloads pruned first, and a
        lineage-stamped digest chain across repeated compactions. Falls
        back to the deterministic first-line digest when no summarizer
        answers, and to the plain history when compaction itself fails —
        compaction must never cost a turn."""
        try:
            from agent.harness.compaction import compact_history

            # Task sessions carry a live plan; re-read it at compaction
            # time so the digest head shows the plan as it IS, not as it
            # stood when the turn began.
            live_note = ""
            try:
                from agent.task_plan import prompt_block, task_for_session

                bound = task_for_session(self.session_id)
                if bound:
                    live_note = prompt_block(bound)
            except Exception:  # noqa: BLE001
                logger.debug("live plan for compaction unavailable",
                             exc_info=True)
            compacted, ran = compact_history(
                history, llm_invoke=self._judge_invoke(),
                live_note=live_note,
            )
            if ran:
                self._pump.emit(
                    context_event(
                        source="compaction",
                        items=[],
                        summary=(
                            f"compressed the conversation from {len(history)} "
                            f"to {len(compacted)} messages (decisions and "
                            "identifiers preserved in the digest)"
                        ),
                    )
                )
            return compacted
        except Exception:  # noqa: BLE001
            logger.warning("compaction failed; running uncompacted",
                           exc_info=True)
            return _prune_history(history)

    def _inline_review(self, draft: str) -> str:
        """L3 of the constitution: critique the answer before it ships.

        The constitution is enforced at four layers — the prompt (L1),
        the interceptor before each tool call (L2), this review (L3), and
        the deploy gate (L4). L3 was documented and implemented but never
        connected to the VM; it only ever ran in the pre-VM agent, so the
        live path had three of the four.

        It runs only when the turn used a consequential tool
        (``execution_review.STRICT_TRIGGERS``), so an ordinary question
        pays nothing. A ``block`` verdict replaces the answer rather than
        editing it: the point of a blocking article is that the response
        does not go out.
        """
        tools = [
            step.get("tool", "")
            for step in self._turn_tool_calls
            if step.get("outcome") == "ok"
        ]
        if not draft or not tools:
            return draft
        try:
            from agent import execution_review

            if not execution_review.should_review(tools):
                return draft
            from agent.constitution import load_articles

            articles = load_articles(_module_sql)
            verdict = execution_review.review(
                draft=draft,
                tool_calls=tools,
                articles=articles,
                judge_invoke=self._judge_invoke(),
            )
        except Exception:  # noqa: BLE001
            # A review that cannot run must not silently pass the answer
            # as reviewed, but it must also not lose the user's work.
            logger.warning("inline review unavailable", exc_info=True)
            return draft

        if verdict.verdict == "ok":
            return draft
        self._pump.emit(
            law_block_event(
                article=verdict.article or "?",
                detail=verdict.reason or "the reviewer withheld this answer",
            )
        )
        self._session.audit_log(
            user_email=self._session.user_email,
            session_id=self._session.session_id,
            tool="inline_review",
            capability="agent.review",
            args={"tools": tools},
            outcome=f"{verdict.verdict}:{verdict.article or ''}",
        )
        if verdict.verdict == "block":
            return (
                "I stopped short of sending that. The review step flagged it "
                f"against {verdict.article or 'the constitution'}: "
                f"{verdict.reason or 'it did not pass review'}. "
                "Tell me how you would like to proceed."
            )
        return draft

    def _judge_invoke(self):
        """A plain ``messages -> text`` callable for the review critic."""

        def _invoke(messages):
            from alex.llm import get_llm

            result = get_llm().invoke(messages)
            return getattr(result, "text", result)

        return _invoke

    def _capture_learned_fact(self, text: str) -> None:
        """Knowledge accumulation: persist at most one durable fact or
        preference from the user's message (heuristic, no extra LLM
        call), surface it as an SSE ``memory`` event, and audit-log the
        write.  Forget requests suppress capture (see
        :func:`agent.persona.extract_durable_fact`).  Failures never
        break the turn."""
        try:
            found = extract_durable_fact(text)
            if found is None:
                return
            kind, content = found
            from agent import vector_memory
            from agent.memory import connection
            from alex.events import Event

            with connection() as conn:
                bead = vector_memory.store(
                    conn,
                    self._session.user_email,
                    kind,
                    content,
                    source_ref=f"session:{self._session.session_id}",
                )
            # Mirror to the Delta-side table so the transparency panel
            # and /api/sdk/recall (which read the mirror) see it too;
            # in production the consolidation job also maintains this.
            try:
                _vm_warehouse().execute(
                    "INSERT INTO media.control.vector_memory_mirror "
                    "(id, owner_email, kind, content, salience, source_ref, "
                    " created_at, superseded_by) "
                    "VALUES (:id, :u, :k, :c, :s, :ref, current_timestamp(), NULL)",
                    {
                        "id": str(uuid.uuid4()),
                        "u": self._session.user_email,
                        "k": kind,
                        "c": content,
                        "s": 0.7,
                        "ref": f"session:{self._session.session_id}",
                    },
                )
            except Exception:  # noqa: BLE001
                logger.debug("mirror write skipped (table unavailable)")
            self._pump.emit(
                Event(
                    type="memory",
                    kind=kind,
                    text=content,
                    extras={"bead_id": bead.id, "learned": True},
                )
            )
            self._session.audit_log(
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                tool="memory_capture",
                capability="memory.write",
                args={"kind": kind, "content": content},
                outcome="ok",
                result=bead.id,
            )
        except Exception:  # noqa: BLE001
            logger.exception("learned-fact capture failed")

    def _finish_visible_reasoning(self, user_text: str, final: str) -> None:
        """Emit the concluding ``reasoning`` event, archive the full
        trace, and keep only the digest for future context."""
        try:
            from alex.events import reasoning_event

            self._pump.emit(reasoning_event("conclude", (final or "")[:400]))
            from agent import thinking

            raw = (
                f"user: {user_text}\n"
                f"assistant (final): {final}\n"
                f"history_len: {len(self._session.history)}"
            )
            digest = thinking.digest_only_for_context({"summary": final or ""})
            wh = self._resolve_warehouse()
            register = getattr(wh, "register", None)
            if callable(register):
                try:
                    register(
                        "media.control.thinking_traces",
                        [
                            "id",
                            "ts",
                            "user_email",
                            "session_id",
                            "turn_no",
                            "phase",
                            "raw_thinking",
                            "digest",
                        ],
                    )
                except Exception:  # noqa: BLE001
                    pass

            def _sql(query: str, parameters: dict | None = None):
                return wh.execute(query, parameters or {})

            thinking.archive(
                _sql,
                self._session.user_email,
                self._session.session_id,
                "turn",
                raw,
                digest,
            )
            self._session.last_reasoning_digest = digest
        except Exception:  # noqa: BLE001
            logger.exception("visible-reasoning archive failed")

    def _resolve_warehouse(self):
        wh = self._config.warehouse
        if wh is None:
            from alex import testing as _testing

            wh = _testing.get_fakes().warehouse
        return wh

    def _constitution_block(self) -> str:
        """The law, as text, for the T0 slice.

        L1 is the layer that tells the model what the rules *are*, and it
        was the other half of the same omission as the interceptor: the
        context engine has always accepted a `constitution_block`, and
        nothing ever passed one, so `build_constitution_prompt` was
        reachable only from its own unit test. The model was being asked
        to obey a constitution it had never been shown, and the first it
        heard of any article was a tool call coming back refused.

        Reuses the articles already loaded onto the session's interceptor
        rather than re-reading them, so L1 and L3 cannot disagree about
        what the law says.
        """
        articles = getattr(self._session.constitution, "articles", None) or []
        if not articles:
            return ""
        try:
            return build_constitution_prompt(articles)
        except Exception:  # noqa: BLE001
            logger.debug("could not render the constitution prompt", exc_info=True)
            return ""

    def _build_system_prompt(
        self, last_user_msg: str, *, show_reasoning: bool = False
    ) -> str:
        """Assemble the turn's system prompt via the context engine.

        Wires RAG/context (standing orders, open tasks, recalled
        memories, KG edges) into the chat path.  Any backend failure
        degrades to the static base prompt so the VM keeps working
        fully offline against the dev fakes.
        """
        base = self._agent_def.prompt or "You are Alex, an enterprise AI agent."
        try:
            from agent.context_engine import load_context

            wh = self._resolve_warehouse()

            def _sql(query: str, parameters: dict | None = None):
                res = wh.execute(query, parameters=parameters or {})
                as_dicts = getattr(res, "as_dicts", None)
                return as_dicts() if callable(as_dicts) else (res or [])

            # Learned skills (learning loop): compact per-user hints
            # matched against the message.  Failures degrade to "".
            skill_hints = ""
            try:
                from agent.learning_loop import active_skill_hints

                skill_hints = active_skill_hints(
                    _sql, self._session.user_email, last_user_msg
                )
            except Exception:  # noqa: BLE001
                logger.debug("skill hints unavailable", exc_info=True)

            # Retrieval depth: instance-library gist + company-first
            # merged retrieval.  Fully offline-safe (empty on failure).
            workspace_block = ""
            try:
                from agent.expertise import merge_retrieval, to_company_first_prompt
                from agent.instance_library import context_gist, library_for

                library = library_for(_sql, self._session.user_email)
                gist = context_gist(library)
                company_hits = [
                    {
                        "kind": "company",
                        "text": e.get("pinned_summary") or e.get("path") or "",
                    }
                    for e in library
                ]
                merged = merge_retrieval(company_hits, [])
                workspace_block = "\n\n".join(
                    s for s in (gist, to_company_first_prompt(merged)) if s
                )
            except Exception:  # noqa: BLE001
                logger.debug("instance library unavailable", exc_info=True)

            # Visible reasoning: inject only the prior turn's digest
            # (<=80 tokens); the raw chain-of-thought never re-enters
            # the context.
            reasoning_parts: list[str] = []
            if self._session.last_reasoning_digest:
                reasoning_parts.append(
                    "PRIOR TURN DECISION DIGEST: "
                    + self._session.last_reasoning_digest
                )
            if show_reasoning:
                from agent.reasoning_stream import REASONING_SYSTEM

                reasoning_parts.append(REASONING_SYSTEM)

            # Goal anticipation: a deterministic "what likely matters
            # now" slice from cheap warehouse signals.  Fail-soft to "".
            anticipation = ""
            try:
                from agent.anticipation import anticipation_block

                anticipation = anticipation_block(
                    _sql, self._session.user_email
                )
            except Exception:  # noqa: BLE001
                logger.debug("anticipation unavailable", exc_info=True)

            # Taught routines: when the message matches a routine the
            # user taught (POST /api/teach), inject its steps so the
            # model executes the workflow with tools.  Fail-soft to "".
            routine_block = ""
            try:
                from agent.learn_with_you import (
                    match_routine,
                    routine_context_block,
                )

                matched = match_routine(
                    _sql, self._session.user_email, last_user_msg
                )
                if matched:
                    routine_block = routine_context_block(matched)
            except Exception:  # noqa: BLE001
                logger.debug("routine match unavailable", exc_info=True)

            # Wrap the context adapters so what the agent consulted is
            # *shown*, not merely used. Without this the grounding for an
            # answer is invisible: the passages that produced a cited
            # claim never reach the UI.
            def _seen_retrieve(email: str, text: str) -> list[dict[str, Any]]:
                hits = _rag_retrieve(email, text)
                if hits:
                    self._pump.emit(
                        context_event(
                            source="knowledge",
                            items=[
                                {
                                    "n": i,
                                    "title": (h.get("citation") or {}).get("title", ""),
                                    "source_ref": (h.get("citation") or {}).get(
                                        "source_ref", ""
                                    ),
                                    "classification": (h.get("citation") or {}).get(
                                        "classification", ""
                                    ),
                                    "score": h.get("score"),
                                    "why": h.get("why"),
                                    "text": str(h.get("text") or "")[:600],
                                }
                                for i, h in enumerate(hits, start=1)
                            ],
                            summary=(
                                f"retrieved {len(hits)} passage"
                                f"{'' if len(hits) == 1 else 's'} "
                                "from enterprise knowledge"
                            ),
                        )
                    )
                return hits

            def _seen_memory(email: str, limit: int = 5, query: str = ""):
                beads = _memory_recall(email, limit=limit, query=query)
                if beads:
                    self._pump.emit(
                        context_event(
                            source="memory",
                            items=[
                                {"kind": k, "content": str(v)[:300]} for k, v in beads
                            ],
                            summary=(
                                f"recalled {len(beads)} memor"
                                f"{'y' if len(beads) == 1 else 'ies'} about you"
                            ),
                        )
                    )
                return beads

            def _seen_kg(email: str, text: str) -> list[dict[str, Any]]:
                edges = _kg_query(email, text)
                if edges:
                    self._pump.emit(
                        context_event(
                            source="graph",
                            items=[
                                {
                                    "src": e.get("src", ""),
                                    "rel": e.get("rel", ""),
                                    "dst": e.get("dst", ""),
                                    "confidence": e.get("confidence"),
                                }
                                for e in edges[:12]
                            ],
                            summary=(
                                f"linked {len(edges)} fact"
                                f"{'' if len(edges) == 1 else 's'} "
                                "from the knowledge graph"
                            ),
                        )
                    )
                return edges

            # The principal gates which practices are in force for this
            # person; without it the guidance slice stays empty rather
            # than showing rules they may not see.
            try:
                from agent import access

                _principal = access.load_principal(
                    _sql, self._session.user_email
                )
            except Exception:  # noqa: BLE001
                logger.debug("principal unavailable for guidance", exc_info=True)
                _principal = None

            # The user's own custom instructions ride in the T0 workspace
            # block, framed as preference: they steer tone and habit but
            # can never outrank the constitution or a mandatory practice.
            try:
                from agent import preferences

                _custom = preferences.custom_instruction_block(
                    _sql, self._session.user_email
                )
                if _custom:
                    workspace_block = (
                        "\n\n".join([workspace_block, _custom])
                        if workspace_block
                        else _custom
                    )
            except Exception:  # noqa: BLE001
                logger.debug("custom instructions unavailable", exc_info=True)

            ctx = load_context(
                system_prompt=base,
                principal=_principal,
                user_email=self._session.user_email,
                session_id=self._session.session_id,
                last_user_msg=last_user_msg,
                sql=_sql,
                memory_recall=_seen_memory,
                kg_query=_seen_kg,
                retrieve=_seen_retrieve,
                workspace_block=workspace_block,
                skill_hints=skill_hints,
                reasoning_block="\n\n".join(reasoning_parts),
                constitution_block=self._constitution_block(),
            )
            slices = [
                ctx.get("T0") or "",
                ctx.get("T1") or "",
                anticipation,
                routine_block,
                ctx.get("T2") or "",
                ctx.get("T3") or "",
            ]
            prompt = "\n\n".join(s for s in slices if s) or base
            # Request observability — the practitioner consensus on
            # production agents: a team must be able to inspect what
            # entered a given model request and which tools were on the
            # table, not infer it from outputs. One event per turn,
            # before the first model call, carrying the assembled
            # prompt's shape (never its content — slices can hold
            # confidential passages; sizes are the observable).
            try:
                slice_names = ("T0 system+law", "T1 directives",
                               "anticipation", "routine", "T2 session",
                               "T3 this turn")
                items = [
                    {"slice": name, "chars": len(s)}
                    for name, s in zip(slice_names, slices, strict=True)
                    if s
                ]
                n_tools = len(self._agent_def.tools)
                self._pump.emit(
                    context_event(
                        source="request",
                        items=items,
                        summary=(
                            f"model request assembled: {len(prompt):,} chars "
                            f"of context · {n_tools} tools exposed"
                            + (" · plan mode" if self._config.plan_mode else "")
                        ),
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("request-observability emit failed", exc_info=True)
            return prompt
        except Exception:  # noqa: BLE001
            logger.exception("context engine failed; using base prompt")
            return base

    # -- browser operations (the public surface) ------------------------
    def open_browser_session(self) -> BrowserSessionState:
        s = self._browser_os.create_session(
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        self._pump.emit(
            artifact_event(
                path=f"browser://{s.session_id}/opened", kind="browser_session"
            )
        )
        return s

    def list_browser_sessions(self) -> list[BrowserSessionState]:
        return self._browser_os.list_sessions(self._session.user_email)

    def close_browser_session(self, session_id: str) -> None:
        self._browser_os.close_session(session_id)
        self._pump.emit(
            artifact_event(
                path=f"browser://{session_id}/closed", kind="browser_session"
            )
        )

    def open_browser_tab(self, session_id: str, *, url: str = "about:blank") -> Any:
        tab = self._browser_os.create_tab(session_id, url=url)
        self._pump.emit(
            artifact_event(
                path=f"browser://{session_id}/{tab.tab_id}/opened", kind="browser_tab"
            )
        )
        return tab

    def list_browser_tabs(self, session_id: str) -> list[Any]:
        return self._browser_os.list_tabs(session_id)

    def close_browser_tab(self, session_id: str, tab_id: str) -> None:
        self._browser_os.close_tab(session_id, tab_id)
        self._pump.emit(
            artifact_event(
                path=f"browser://{session_id}/{tab_id}/closed", kind="browser_tab"
            )
        )

    def navigate_browser(
        self, session_id: str, tab_id: str, *, url: str
    ) -> dict[str, Any]:
        res = browser_navigate(
            session_id=session_id,
            tab_id=tab_id,
            url=url,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        if res.get("ok"):
            self._pump.emit(
                artifact_event(
                    path=f"browser://{session_id}/{tab_id}/navigated",
                    kind="page",
                )
            )
        return res

    def click_browser(
        self, session_id: str, tab_id: str, *, selector: str
    ) -> dict[str, Any]:
        res = browser_click(
            session_id=session_id,
            tab_id=tab_id,
            selector=selector,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        if res.get("ok"):
            self._pump.emit(
                artifact_event(
                    path=f"browser://{session_id}/{tab_id}/clicked",
                    kind="click",
                )
            )
        return res

    def type_browser(
        self,
        session_id: str,
        tab_id: str,
        *,
        selector: str,
        text: str,
    ) -> dict[str, Any]:
        res = browser_type(
            session_id=session_id,
            tab_id=tab_id,
            selector=selector,
            text=text,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        if res.get("ok"):
            self._pump.emit(
                artifact_event(
                    path=f"browser://{session_id}/{tab_id}/typed",
                    kind="form_value",
                )
            )
        return res

    def extract_browser(
        self, session_id: str, tab_id: str, *, selector: str = "body"
    ) -> dict[str, Any]:
        res = browser_extract(
            session_id=session_id,
            tab_id=tab_id,
            selector=selector,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        self._pump.emit(
            artifact_event(
                path=f"browser://{session_id}/{tab_id}/extract",
                kind="page_text",
            )
        )
        return res

    def screenshot_browser(self, session_id: str, tab_id: str) -> dict[str, Any]:
        res = browser_screenshot(
            session_id=session_id,
            tab_id=tab_id,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        self._pump.emit(
            artifact_event(
                path=f"browser://{session_id}/{tab_id}/screenshot",
                kind="png",
            )
        )
        return res

    def submit_browser(
        self,
        session_id: str,
        tab_id: str,
        *,
        selector: str = "form",
        confirmation_token: str | None = None,
    ) -> dict[str, Any]:
        # The confirmation_required SSE event (with the token) is
        # emitted inside browser_submit's staging phase — that is the
        # only channel the token travels on. It is deliberately absent
        # from the returned dict: this method's result reaches the
        # model, and a model-readable token is a self-approving one.
        return browser_submit(
            session_id=session_id,
            tab_id=tab_id,
            selector=selector,
            confirmation_token=confirmation_token,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )

    def mirror_browser(
        self, session_id: str, tab_id: str, *, mode: str
    ) -> dict[str, Any]:
        res = browser_open_in_user_session(
            session_id=session_id,
            tab_id=tab_id,
            mode=mode,
            user_email=self._session.user_email,
            instance_scope=self._session.instance_scope,
        )
        self._pump.emit(
            artifact_event(
                path=f"browser://{session_id}/{tab_id}/mirror",
                kind="user_control",
            )
        )
        return res

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            close = getattr(self._executor, "close", None)
            if callable(close):
                # The abstract Executor.close is async; the concrete
                # MockExecutor.close is sync.  Detect by inspecting
                # the coroutine-ness of the return value.
                import inspect

                result = close()
                if inspect.iscoroutine(result):
                    # Await it wherever we are: close() is called from
                    # routes, worker threads and context managers alike.
                    try:
                        from agent.harness.sync_bridge import run_coroutine

                        run_coroutine(result, timeout=30.0)
                    except Exception:  # noqa: BLE001
                        logger.exception("error awaiting executor.close()")
        except Exception:  # noqa: BLE001
            logger.exception("error closing executor")

    def __enter__(self) -> VM:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Context-engine adapters (memory + knowledge graph)
# ---------------------------------------------------------------------------


def _prune_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap the in-context history to the last N messages
    (``ALEX_HISTORY_MAX_MSGS``, default 20).

    Earlier messages are collapsed into a single digest entry so the
    model keeps the gist of prior decisions without paying for every
    old token.

    A prior digest head is folded into the new one **in full**, not
    by its first line: the head's own text promises "decisions below
    still stand", and re-pruning used to keep only the banner while
    every actual bullet from the previous digest evaporated — so
    anything decided more than two prune windows ago was silently
    forgotten, one turn at a time.
    """
    try:
        max_msgs = int(os.environ.get("ALEX_HISTORY_MAX_MSGS", "20"))
    except ValueError:
        max_msgs = 20
    if max_msgs <= 0 or len(history) <= max_msgs:
        return history
    earlier, recent = history[:-max_msgs], history[-max_msgs:]
    lines: list[str] = []
    for m in earlier:
        role = str(m.get("role", "?"))
        content = str(m.get("content") or "").strip()
        if role == "system" and content.startswith(
            "EARLIER CONVERSATION DIGEST"
        ):
            # A previous digest: carry its bullets forward verbatim,
            # minus the banner it came with.
            body = content.split("\n", 1)[-1].strip()
            if body:
                lines.append(body)
            continue
        first = content.splitlines()[0] if content else ""
        if first:
            lines.append(f"- {role}: {first[:120]}")
    digest = "\n".join(lines)[-1200:]
    head = {
        "role": "system",
        "content": (
            "EARLIER CONVERSATION DIGEST (older turns pruned; decisions "
            "below still stand):\n" + digest
        ),
    }
    return [head, *recent]


def _validate_tool_args(spec: ToolSpec, args: dict) -> str | None:
    """Check the model-supplied args against the tool's signature.

    Returns a human/model-readable problem string, or None when the
    call is well-formed.  Identity kwargs are VM-injected and neither
    required from nor accepted as overrides by the model."""
    import inspect

    try:
        sig = inspect.signature(spec.callable)
    except (TypeError, ValueError):
        return None
    params = sig.parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    identity = {"user_email", "instance_scope"}
    unknown = [
        k for k in args if k not in params and k not in identity
    ]
    if unknown and not accepts_kwargs:
        valid = [n for n in params if n not in identity and n != "kwargs"]
        return f"unknown argument(s) {unknown}; this tool accepts: {valid}"
    missing = [
        n
        for n, p in params.items()
        if n not in identity
        and p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and n not in args
    ]
    if missing:
        return f"missing required argument(s): {missing}"

    # Type-check against annotations, coercing where safe.  Small local
    # models routinely mix types (k="favorite reporting day"); a
    # correctable error message beats a Python crash inside the tool.
    for name, value in list(args.items()):
        p = params.get(name)
        if p is None or p.annotation is inspect.Parameter.empty:
            continue
        ann = p.annotation
        want = {int: int, float: float, bool: bool, str: str}.get(
            ann if isinstance(ann, type) else None
        )
        if want is None or isinstance(value, want):
            continue
        if want in (int, float) and isinstance(value, str):
            try:
                args[name] = want(value.strip())
                continue
            except ValueError:
                pass
        if want is int and isinstance(value, float) and value.is_integer():
            args[name] = int(value)
            continue
        if want is str:
            args[name] = str(value)
            continue
        return (
            f"argument {name!r} must be {want.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )
    return None


# Relevance floor for recalled beads: 0.6*lexical + 0.4*salience must
# clear this, and the bead must share at least one term with the
# message, before it earns prompt tokens.
_RECALL_FLOOR = 0.25


def _memory_recall(
    user_email: str, limit: int = 5, query: str = ""
) -> list[tuple[str, str]]:
    """Adapter over :func:`agent.vector_memory.recall` for the
    context engine — returns ``list[(kind, content)]``.

    When ``query`` (the current user message) is supplied, beads are
    scored ``0.6*lexical + 0.4*salience`` (the lexical term-overlap
    ranking shared with ``/api/sdk/recall``); beads with no term
    overlap or a combined score below the floor are excluded, so
    high-salience but irrelevant memories stay out of the prompt.
    Any failure returns ``[]``.
    """
    limit = max(1, min(int(limit), 5))
    try:
        from agent import vector_memory
        from agent.memory import connection

        with connection() as conn:
            beads = vector_memory.recall(conn, user_email, k=max(limit * 5, 25))
        terms = [t for t in query.lower().split() if len(t) > 2]
        if not terms:
            return [(b.kind, b.content) for b in beads[:limit]]
        scored: list[tuple[float, Any]] = []
        for b in beads:
            content = (b.content or "").lower()
            lexical = min(sum(t in content for t in terms) / len(terms), 1.0)
            if lexical <= 0:
                continue
            combined = 0.6 * lexical + 0.4 * float(b.salience or 0.0)
            if combined < _RECALL_FLOOR:
                continue
            scored.append((combined, b))
        scored.sort(key=lambda x: -x[0])
        return [(b.kind, b.content) for _, b in scored[:limit]]
    except Exception:  # noqa: BLE001
        logger.debug("memory recall unavailable", exc_info=True)
        return []


def _live_constitution(
    session_id: str, user_email: str, warehouse: Any | None = None
) -> ConstitutionalInterceptor:
    """The L2 interceptor, loaded with the constitution actually in force.

    The default used to be an interceptor with **no articles and a no-op
    violation logger**, and nothing ever replaced it: `make_vm` does not
    pass `constitution=`, so every VM the application built ran an L2
    check that was structurally incapable of denying anything and never
    recorded a violation. The layer was in the design, in the docs, and
    in the call path — it just had nothing to enforce.

    Articles are read once per VM (VMs are cached per session), and a
    failure to read them degrades to the empty interceptor rather than
    failing the session: L3 review and the governor still apply, and a
    chat that refuses to start because a table is missing is its own
    kind of outage. The read failure is logged at warning level because
    silently running with no constitution is exactly the state this
    function exists to end.
    """
    def sql(query: str, parameters: dict | None = None):
        if warehouse is not None:
            res = warehouse.execute(query, parameters or {})
            as_dicts = getattr(res, "as_dicts", None)
            return as_dicts() if callable(as_dicts) else list(res or [])
        return _module_sql(query, parameters)

    articles: list[dict] = []
    log_violation = None
    try:
        articles = load_articles(sql)
        log_violation = make_violation_logger(sql, user_email, session_id)
    except Exception:  # noqa: BLE001
        logger.warning(
            "could not load the constitution for session %s; L2 will not "
            "enforce this session", session_id, exc_info=True,
        )
    return ConstitutionalInterceptor(
        articles=articles,
        log_violation=log_violation or (lambda *a, **k: None),
        session_id=session_id,
        user_email=user_email,
    )


def _module_sql(query: str, parameters: dict | None = None):
    """Warehouse reader for module-level adapters (which have no ``self``
    and therefore no resolved warehouse of their own)."""
    from agent.warehouse import make_sql

    return make_sql()(query, parameters or {})


def _rag_retrieve(user_email: str, text: str) -> list[dict[str, Any]]:
    """Adapter over :func:`agent.rag.retrieve` for the context engine.

    Runs hybrid (dense + BM25 + graph-expanded) retrieval over the
    enterprise corpus the user may read, so every turn is grounded in
    organisational knowledge with citations attached. Any failure
    returns ``[]`` — a missing corpus degrades the turn to the base
    prompt rather than breaking it.
    """
    try:
        from agent import access, rag
        from agent.memory import connection

        principal = access.load_principal(_module_sql, user_email)
        with connection() as conn:
            hits = rag.retrieve(
                conn, text, user_email=user_email, k=4, principal=principal
            )
        return [h.to_dict() for h in hits]
    except Exception:  # noqa: BLE001
        logger.debug("enterprise retrieval unavailable", exc_info=True)
        return []


def _kg_query(user_email: str, text: str) -> list[dict[str, Any]]:
    """Adapter over :mod:`agent.enterprise_graph` for the context engine.

    This used to call the legacy per-user graph
    (`knowledge_graph.entity_context`), which is a raw psycopg read of a
    Lakebase table — in dev there is no such connection, so the KG
    channel of every context assembly was silently empty exactly where
    the rest of the system was being developed and tested. The modern
    graph has a dev backend, embedded entity resolution, temporal edges
    and the enterprise ∪ user scope model; the context engine now sees
    the same graph everything else writes.

    Any failure returns ``[]`` so a missing backend degrades to the base
    prompt."""
    try:
        from agent.enterprise_graph import subgraph
        from agent.memory import connection

        with connection() as conn:
            return subgraph(conn, text, user_email=user_email).get("edges") or []
    except Exception:  # noqa: BLE001
        logger.debug("kg query unavailable", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Default audit log + VM factory used by the SDK and the FastAPI routes
# ---------------------------------------------------------------------------


class _DefaultAuditLog:
    """Writes a row to ``media.alex.agent_actions`` (constitution A3).

    The dev backend is :class:`alex.warehouse.InMemoryWarehouse`; the
    production backend is the Databricks SDK's statement-execution
    client.  The VM doesn't care which it is — it calls
    ``wh.execute(sql, params)`` and either backend handles it.
    """

    def __init__(self, warehouse: Any) -> None:
        self._wh = warehouse
        self._seed_if_needed()

    def _seed_if_needed(self) -> None:
        # The dev ``InMemoryWarehouse`` starts empty; the agent
        # tool surface and the audit log both expect
        # ``media.alex.agent_actions`` to exist.  In production the
        # warehouse is a real Unity Catalog and the DDL is
        # declarative, so ``register`` is a no-op.
        register = getattr(self._wh, "register", None)
        if not callable(register):
            return
        try:
            register("media.alex.agent_actions", _AGENT_ACTIONS_COLUMNS)
        except Exception:  # noqa: BLE001
            # Already registered or the warehouse doesn't support
            # dev seeding; the audit-log writes are best-effort.
            pass

    @staticmethod
    def _normalize_outcome(outcome: str) -> str:
        for known in ("ok", "denied", "error", "revoked", "started"):
            if outcome == known or outcome.startswith(f"{known}:"):
                return known
        return "error"

    def __call__(self, **kwargs: Any) -> None:
        tool = kwargs.get("tool", "")
        outcome = str(kwargs.get("outcome", ""))
        try:
            input_json = json.dumps(
                kwargs.get("args") or {}, default=str, ensure_ascii=False
            )[:2000]
        except Exception:  # noqa: BLE001
            input_json = "{}"
        result = kwargs.get("result")
        output_json = str(result if result is not None else outcome)[:2000]
        action_id = f"act_{uuid.uuid4()}"
        try:
            self._wh.execute(
                "INSERT INTO media.alex.agent_actions "
                "(id, actor, action_type, detail, outcome, tool_name, tool_args, "
                " action_id, session_id, tool, capability, user_email, instance_scope, "
                " input_json, output_json, ok, ts) "
                "VALUES (:aid, :u, :atype, :out, :outcome, :tool, :in, "
                "        :aid, :sid, :tool, :cap, :u, :scope, :in, :out, :ok, "
                "        current_timestamp())",
                {
                    "aid": action_id,
                    "atype": "tool_call",
                    "sid": kwargs.get("session_id", ""),
                    "tool": tool,
                    "cap": kwargs.get("capability", f"alex:tool:{tool}"),
                    "u": kwargs.get("user_email", ""),
                    "scope": kwargs.get("instance_scope", ""),
                    "in": input_json,
                    "out": output_json,
                    "outcome": self._normalize_outcome(outcome),
                    "ok": 1 if outcome.startswith("ok") else 0,
                },
            )
        except Exception:  # noqa: BLE001
            # Best-effort: the agent's audit log must not block the
            # user-facing operation.  Errors are visible in the
            # warehouse's error log.
            logger.exception("audit log write failed for %s", tool)


def make_vm(
    *,
    session_id: str,
    user_email: str,
    instance_scope: str = "default",
    mode: str | None = None,
    executor: Executor | None = None,
    browser_os: Any | None = None,
    warehouse: Any | None = None,
    capability_ceiling: frozenset[str] | None = None,
    plan_mode: bool = False,
    budget_ceiling: Any | None = None,
) -> VM:
    """Construct a :class:`VM` with sensible defaults.

    ``mode`` defaults to the value of :data:`alex.config.Settings.mode`
    (which falls back to ``"dev"`` when ``ALEX_ENV`` is unset).
    ``executor``, ``browser_os``, and ``warehouse`` are injected
    when the caller wants a different backend (tests, the SDK).

    When ``warehouse`` is not supplied, the VM uses the prod-aware
    warehouse: the fakes' shared ``InMemoryWarehouse`` in dev (so the
    audit log and the tools read/write the same store the tests inspect)
    and the real Databricks SQL warehouse in production. Defaulting to the
    fake unconditionally silently discarded the audit trail in prod.
    """
    if mode is None:
        mode = get_settings().mode
    if warehouse is None:
        warehouse = _vm_warehouse()
    return VM(
        session_id=session_id,
        user_email=user_email,
        instance_scope=instance_scope,
        config=HarnessConfig(
            mode=mode,
            executor=executor,
            browser_os=browser_os,
            warehouse=warehouse,
            capability_ceiling=capability_ceiling,
            plan_mode=plan_mode,
            budget_ceiling=budget_ceiling,
        ),
    )


__all__ = [
    "VM",
    "VMSession",
    "HarnessConfig",
    "TOOL_REGISTRY",
    "ToolSpec",
    "make_vm",
]
