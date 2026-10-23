"""Swarm execution — one agent delegating to named, bounded sub-agents.

The shape is orchestrator–worker, which the published multi-agent work
(Anthropic's research system among others) converges on for a reason:
one lead agent plans and delegates; workers execute in **isolation**,
each with a self-contained brief and a fresh context; workers never talk
to each other; every decision about what happens next belongs to the
orchestrator. Peer-to-peer chatter is where multi-agent systems become
unaccountable, so it is not offered.

Every sub-agent is a full :class:`agent.vm.VM` — the same tool registry,
the same seven-layer control stack — bounded three ways on top of it:

* **Capability ceiling.** A worker runs with the *intersection* of the
  parent VM's effective capabilities and what its brief requests, minus
  :data:`DELEGATE_CAPABILITY` always — a worker can never delegate, so
  the swarm is depth-one by construction and the concurrency cap cannot
  be multiplied away by recursion. Enforced by
  ``HarnessConfig.capability_ceiling`` inside the child VM itself, not by
  this module's goodwill.
* **Concurrency.** At most :data:`MAX_CONCURRENT` workers run at once.
* **Fan-out.** At most :data:`MAX_PER_TASK` workers per delegation.
  Published guidance lands on batches of 5–10; ten is the hard policy
  ceiling here, configured, not hoped.

Visibility is a first-class output, not a debug feature: every child
event — thinking, tool calls, artifacts, denials — is forwarded onto the
**parent's** event pump as a ``subagent`` event carrying the worker's
name, so the person watching the run sees each worker's lane live, and
the supervision layer reviews the same record afterwards.
"""

from __future__ import annotations

import json
import logging
import os as _os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from alex.events import subagent_event

log = logging.getLogger("agent.swarm")

#: The capability that lets an agent delegate at all — and the one no
#: sub-agent ever receives.
DELEGATE_CAPABILITY = "agents.delegate"

#: Hard policy bounds. Environment-overridable for operators, clamped so
#: an override cannot raise them past the policy ceiling of ten.
MAX_CONCURRENT = min(10, int(_os.environ.get("ALEX_SWARM_CONCURRENT", "10")))
MAX_PER_TASK = min(10, int(_os.environ.get("ALEX_SWARM_PER_TASK", "10")))
#: Wall-clock bound per worker, on top of its own governor's budgets.
WORKER_TIMEOUT_S = float(_os.environ.get("ALEX_SWARM_WORKER_TIMEOUT", "300"))

#: Child event types worth forwarding to the parent's stream. `text` is
#: deliberately included — a worker's answer streaming into its lane is
#: what makes the swarm watchable rather than a spinner with names.
_FORWARDED = frozenset(
    {"text", "reasoning", "tool", "artifact", "error", "law_block", "governor"}
)

# The VM currently executing a tool, set by VM._invoke_tool around the
# call. This is how a registry tool (a plain callable) reaches the parent
# VM that invoked it without the model being able to name a different
# one.
_CURRENT_VM: ContextVar[Any | None] = ContextVar("alex_current_vm", default=None)


def set_current_vm(vm: Any | None):
    return _CURRENT_VM.set(vm)


def reset_current_vm(token) -> None:
    try:
        _CURRENT_VM.reset(token)
    except ValueError:  # token from another context; nothing to restore
        pass


def current_vm() -> Any | None:
    return _CURRENT_VM.get()


@dataclass
class SubagentSpec:
    """One worker's self-contained brief.

    ``brief`` must stand alone: the worker starts with a fresh context
    and does not know the other workers exist. ``capabilities`` is what
    the brief *requests*; what it gets is the intersection with the
    parent's own effective set.
    """

    name: str
    brief: str
    capabilities: list[str] = field(default_factory=list)


@dataclass
class SubagentResult:
    name: str
    status: str  # 'done' | 'failed' | 'timeout'
    output: str = ""
    error: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _parse_specs(tasks: Any) -> list[SubagentSpec]:
    """Model-supplied task list -> validated specs.

    Names are deduplicated because they key the UI lanes and the result
    map; two workers sharing a name would silently merge their records.
    """
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tasks must be a JSON list of objects: {exc}") from exc
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    if len(tasks) > MAX_PER_TASK:
        raise ValueError(
            f"{len(tasks)} sub-agents requested; the per-task policy limit "
            f"is {MAX_PER_TASK}. Split the work into batches."
        )
    specs: list[SubagentSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(tasks):
        if not isinstance(raw, dict):
            raise ValueError(f"task {i} is not an object")
        name = str(raw.get("name") or f"worker-{i + 1}").strip()[:48]
        base, n = name, 2
        while name in seen:
            name, n = f"{base}-{n}", n + 1
        seen.add(name)
        brief = str(raw.get("brief") or raw.get("task") or "").strip()
        if not brief:
            raise ValueError(f"task {i} ({name}) has no brief")
        caps = [str(c).strip() for c in (raw.get("capabilities") or []) if str(c).strip()]
        specs.append(SubagentSpec(name=name, brief=brief, capabilities=caps))
    return specs


def _child_ceiling(parent_vm: Any, requested: list[str]) -> frozenset[str]:
    """What one worker may do: parent ∩ requested, never the delegate cap.

    An empty request inherits the parent's set (minus delegation) — the
    orchestrator narrowing per-worker is encouraged but optional. The
    delegate capability is stripped unconditionally: workers that can
    spawn workers turn a bounded fan-out into an unbounded tree.
    """
    parent_caps = parent_vm._allowed_capabilities()
    if parent_caps is None:
        # Permissive dev with no ACL store. The ceiling is then the
        # request itself; with no request, the registry minus delegation.
        from agent.tools import TOOL_REGISTRY

        parent_caps = {spec.capability for spec in TOOL_REGISTRY.values()}
    allowed = set(parent_caps)
    if requested:
        allowed &= set(requested)
    allowed.discard(DELEGATE_CAPABILITY)
    return frozenset(allowed)


def _child_budget(parent_vm: Any, n_workers: int):
    """One worker's budget ceiling: a slice of the PARENT's remaining
    allowance, not a fresh full budget.

    Before this, every child built its own ``Budget.for_autonomy`` —
    10 workers x 40 tool calls charged to a single parent step, so a
    parent at 90% of its meter could legally multiply its remaining
    spend by the fan-out. Tool calls and model passes divide across the
    batch (they add up); wall clock does not (workers run in parallel).
    Small floors keep every worker viable — the parent chose to
    delegate, and a zero-budget child is a guaranteed failure billed as
    a worker.

    None when the parent has no live governor (direct test drives):
    the child then budgets itself as before.
    """
    gov = getattr(parent_vm, "governor", None) or getattr(
        parent_vm, "_governor", None
    )
    if gov is None:
        return None
    try:
        from agent.execution_governor import Budget

        base = gov.budget
        n = max(1, int(n_workers))
        remaining_tools = max(0, base.max_tool_calls - gov.tool_calls)
        remaining_turns = max(0, base.max_llm_turns - gov.llm_turns)
        remaining_s = max(0.0, base.max_seconds - gov.elapsed)
        return Budget(
            max_seconds=max(30.0, min(WORKER_TIMEOUT_S, remaining_s)),
            max_tool_calls=max(4, remaining_tools // n),
            max_llm_turns=max(3, remaining_turns // n + 1),
        )
    except Exception:  # noqa: BLE001
        log.debug("child budget derivation failed", exc_info=True)
        return None


def _worker_prompt(spec: SubagentSpec) -> str:
    """The brief, framed so the worker returns data, not conversation."""
    return (
        f"You are sub-agent '{spec.name}', one worker in a swarm. Your "
        "output goes to the lead agent that delegated to you, not to a "
        "person — return your findings directly and completely, with no "
        "greetings and no questions back. You cannot ask for "
        "clarification; state assumptions explicitly instead.\n\n"
        f"YOUR TASK:\n{spec.brief}"
    )


def _run_one(
    parent_vm: Any, spec: SubagentSpec, *, vm_factory=None, budget=None
) -> SubagentResult:
    """Execute one worker in its own VM, forwarding every step."""
    pump = parent_vm.pump
    started = time.monotonic()
    pump.emit(
        subagent_event(name=spec.name, phase="spawned", text=spec.brief[:280])
    )

    ceiling = _child_ceiling(parent_vm, spec.capabilities)
    try:
        if vm_factory is None:
            from agent.vm import make_vm as vm_factory  # type: ignore
        child = vm_factory(
            session_id=f"{parent_vm.session_id}:swarm:{spec.name}",
            user_email=parent_vm.user_email,
            capability_ceiling=ceiling,
            budget_ceiling=budget,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("swarm worker %s could not start a VM", spec.name)
        pump.emit(subagent_event(name=spec.name, phase="failed", text=str(exc)))
        return SubagentResult(
            name=spec.name, status="failed", error=f"could not start: {exc}",
            elapsed_s=time.monotonic() - started,
        )

    def forward(event: Any) -> None:
        payload = event if isinstance(event, dict) else _event_dict(event)
        if payload.get("type") in _FORWARDED:
            pump.emit(
                subagent_event(
                    name=spec.name, phase=str(payload.get("type")), body=payload
                )
            )

    try:
        child.subscribe_sse(forward)
        text, _history = child.run_turn(text=_worker_prompt(spec))
        output = (text or "").strip()
        elapsed = time.monotonic() - started
        pump.emit(
            subagent_event(
                name=spec.name, phase="done",
                text=(output[:280] or "Finished with no output."),
            )
        )
        return SubagentResult(
            name=spec.name, status="done", output=output, elapsed_s=elapsed
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("swarm worker %s failed", spec.name)
        elapsed = time.monotonic() - started
        pump.emit(subagent_event(name=spec.name, phase="failed", text=str(exc)))
        return SubagentResult(
            name=spec.name, status="failed", error=str(exc), elapsed_s=elapsed
        )
    finally:
        try:
            child.close()
        except Exception:  # noqa: BLE001
            log.debug("swarm worker %s close failed", spec.name, exc_info=True)


def run_swarm(
    parent_vm: Any, tasks: Any, *, vm_factory=None, timeout_s: float | None = None
) -> dict[str, Any]:
    """Run a batch of workers and return every result, good and bad.

    Results are keyed by worker name and always complete: a worker that
    failed or timed out appears with its status rather than vanishing,
    because the orchestrator reasons over the whole batch and a silent
    hole reads as "nothing to report" when it means "nobody looked".
    """
    specs = _parse_specs(tasks)
    window = WORKER_TIMEOUT_S if timeout_s is None else timeout_s
    results: dict[str, SubagentResult] = {}

    # No `with` block: the context manager joins every worker thread on
    # exit, so one straggler blocked inside a long model call kept the
    # ORCHESTRATOR waiting long past the window — a timeout that does
    # not time out. The pool is released without waiting; stragglers
    # finish (and close their VMs in _run_one's finally) on their own.
    pool = ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT, len(specs)),
        thread_name_prefix="swarm",
    )
    try:
        # Every worker draws from the same slice of the parent's
        # remaining allowance, computed once for the batch.
        child_budget = _child_budget(parent_vm, len(specs))
        futures = {
            pool.submit(
                _run_one, parent_vm, spec,
                vm_factory=vm_factory, budget=child_budget,
            ): spec
            for spec in specs
        }
        deadline = time.monotonic() + window
        try:
            for future in as_completed(futures, timeout=window + 30):
                spec = futures[future]
                try:
                    results[spec.name] = future.result(
                        timeout=max(0.1, deadline - time.monotonic())
                    )
                except Exception as exc:  # noqa: BLE001
                    results[spec.name] = SubagentResult(
                        name=spec.name, status="timeout", error=str(exc)
                    )
                    parent_vm.pump.emit(
                        subagent_event(name=spec.name, phase="failed", text=str(exc))
                    )
        except TimeoutError:
            # as_completed itself timing out (some workers never finished)
            # must NOT discard the results already in hand — the whole
            # promise of this function is that every worker appears in
            # the batch, and an exception here vanished all of them at
            # once. The stragglers get their 'timeout' rows below.
            log.warning(
                "swarm window elapsed with %d/%d workers unfinished",
                len(specs) - len(results), len(specs),
            )
            for future, spec in futures.items():
                if spec.name not in results:
                    future.cancel()
                    parent_vm.pump.emit(
                        subagent_event(
                            name=spec.name, phase="failed",
                            text="timed out; the swarm window elapsed",
                        )
                    )
    finally:
        pool.shutdown(wait=False, cancel_futures=False)

    for spec in specs:  # a future that never completed still gets a row
        results.setdefault(
            spec.name,
            SubagentResult(name=spec.name, status="timeout", error="never finished"),
        )

    done = sum(1 for r in results.values() if r.status == "done")
    return {
        "ok": done == len(specs),
        "workers": len(specs),
        "succeeded": done,
        "results": [results[s.name].to_dict() for s in specs],
    }


def _event_dict(event: Any) -> dict[str, Any]:
    to_json = getattr(event, "to_json", None)
    if callable(to_json):
        try:
            return json.loads(to_json())
        except (TypeError, ValueError):
            return {}
    return {}


# ---------------------------------------------------------------------------
# The registry tool
# ---------------------------------------------------------------------------


def delegate_subagents(
    tasks: Any, *, user_email: str = "", instance_scope: str = "default"
) -> dict[str, Any]:
    """Spawn up to ten named sub-agents to work parts of a task in parallel.

    Each task is an object: {"name": str, "brief": str, "capabilities":
    [str, ...]}. Briefs must be self-contained — workers start fresh and
    cannot see each other or ask questions. Workers get at most YOUR
    capabilities, narrowed to what each brief requests, and can never
    delegate further.
    """
    parent = current_vm()
    if parent is None:
        # Reachable only if the tool is invoked outside a VM, which the
        # registry wiring never does — but a tool that guesses its parent
        # would let a caller pick whose capabilities to inherit.
        return {"ok": False, "error": "delegation requires a running agent session"}
    try:
        return run_swarm(parent, tasks)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
