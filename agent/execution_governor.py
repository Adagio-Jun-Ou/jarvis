"""
The execution governor — hard bounds on what one agentic run may do.

Giving an agent its own VM and a goal creates a specific failure mode:
the agent will pursue the goal. Not maliciously — a model that has been
told to finish a task and keeps failing will retry the same call, cycle
between two approaches, wander into unrelated work, or reach for a more
privileged tool after a refusal, because each of those locally looks
like progress. None of it is caught by authorization, because every
individual step may be permitted; what is wrong is the *shape* of the
run.

So this is a separate control from the ACL and the constitution. Those
answer "may this action happen?". The governor answers "is this run
still behaving like work?" and stops it when the answer is no.

Six pathologies, all detected deterministically — no model in the loop,
because the thing being watched is a model:

* **Budget** — wall-clock, tool calls, LLM turns. The blunt backstop.
* **Repetition** — the same tool with the same arguments, again.
* **Oscillation** — A, B, A, B: two approaches alternating, which is a
  model unable to choose rather than a model making progress.
* **Stagnation** — many steps, nothing produced or changed.
* **Denial escalation** — retrying after a refusal, or reaching for
  progressively more powerful capabilities after one. This is the
  "achieve the goal anyway" failure, and it is treated as the most
  serious signal here because it is the one where the agent is working
  *against* the governance rather than merely wasting time.
* **Scope drift** — touching capability families that have nothing to do
  with the stated goal.

Every decision is an event, so a halted run explains itself in the same
execution log the user is already watching.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent.execution_governor")

# Actions, least to most severe.
CONTINUE = "continue"
WARN = "warn"
HALT = "halt"

# Capability families ordered by blast radius. Reaching *up* this list
# after a denial is the escalation signal.
CAPABILITY_POWER = {
    "knowledge.read": 0, "memory.read": 0, "files.read": 0,
    "email.read": 0, "tasks.read": 0, "code.read": 0,
    "practices.read": 0, "orders.read": 0, "office.read": 0,
    "memory.write": 1, "files.write": 1, "tasks.create": 1,
    "tasks.update": 1, "knowledge.write": 1, "office.write": 1,
    "browser.read": 1, "browser.use": 1,
    "browser.act": 2, "email.draft": 2, "internal_api.read": 2,
    "compute.python": 2, "compute.bash": 2, "orders.run": 2,
    "email.send": 3, "internal_api.act": 3, "publish.post": 3,
    "compute.shell": 3, "compute.download": 3,
    # Workflows on the enterprise n8n instance act on real systems.
    "workflow.read": 0, "workflow.trigger": 3, "workflow.create": 3,
    # Fan-out spawns whole governed VMs — compute-act power, so a run
    # that escalates into delegation after denials, or delegates right
    # after consuming untrusted content, is seen by the same checks as
    # any other act. (Was absent, which scored delegation as a read.)
    "agents.delegate": 2,
    # Reading the screen is a read; moving the mouse on the person's own
    # machine is an act.
    "desktop.read": 0, "desktop.act": 2,
    "home.read": 0, "home.act": 2,
    "compute.install": 4, "system.install": 4,
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass
class Budget:
    """What a single run is allowed to consume.

    Defaults are deliberately modest: a run that needs more than this is
    usually a run that has lost the thread, and the ceiling should be
    raised knowingly rather than discovered in a bill.
    """

    max_seconds: float = field(
        default_factory=lambda: _env_float("ALEX_RUN_MAX_SECONDS", 600.0)
    )
    max_tool_calls: int = field(
        default_factory=lambda: _env_int("ALEX_RUN_MAX_TOOL_CALLS", 40)
    )
    max_llm_turns: int = field(
        default_factory=lambda: _env_int("ALEX_RUN_MAX_LLM_TURNS", 12)
    )
    max_repeats: int = field(
        default_factory=lambda: _env_int("ALEX_RUN_MAX_REPEATS", 3)
    )
    max_denials: int = field(
        default_factory=lambda: _env_int("ALEX_RUN_MAX_DENIALS", 3)
    )
    max_stagnant_steps: int = field(
        default_factory=lambda: _env_int("ALEX_RUN_MAX_STAGNANT", 6)
    )
    # Token ceiling for the whole run (prompt + completion, summed over
    # passes). 0 = no ceiling — token counts only exist when the
    # provider reports them, so the default must not halt runs on
    # providers that report nothing.
    max_total_tokens: int = field(
        default_factory=lambda: _env_int("ALEX_RUN_MAX_TOKENS", 0)
    )
    # Autonomy raises or lowers the ceiling, so "suggest" mode is genuinely
    # more conservative than "autopilot" rather than nominally so.
    @classmethod
    def for_autonomy(cls, autonomy: str) -> Budget:
        b = cls()
        if autonomy == "suggest":
            b.max_tool_calls = min(b.max_tool_calls, 8)
            b.max_llm_turns = min(b.max_llm_turns, 4)
            b.max_seconds = min(b.max_seconds, 180.0)
        elif autonomy == "autopilot":
            b.max_tool_calls = int(b.max_tool_calls * 1.5)
            b.max_llm_turns = int(b.max_llm_turns * 1.5)
            b.max_seconds = b.max_seconds * 2
        return b

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_seconds": self.max_seconds,
            "max_tool_calls": self.max_tool_calls,
            "max_llm_turns": self.max_llm_turns,
            "max_repeats": self.max_repeats,
            "max_denials": self.max_denials,
            "max_stagnant_steps": self.max_stagnant_steps,
            "max_total_tokens": self.max_total_tokens,
        }


@dataclass
class Verdict:
    action: str
    reason: str
    pathology: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def halting(self) -> bool:
        return self.action == HALT

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "pathology": self.pathology,
            "detail": self.detail,
        }


OK = Verdict(CONTINUE, "")


@dataclass
class Step:
    """One thing the agent did."""

    kind: str  # 'tool' | 'llm' | 'denial'
    name: str = ""
    args_key: str = ""
    capability: str = ""
    ok: bool = True
    produced: bool = False  # did it change or create something?
    # The result carried outside-authored content (a web page, an
    # email, a remote tool's response) — the indirect-injection vector.
    untrusted: bool = False
    # A stable digest of the observation, for stuck detection: the same
    # call with the SAME result again is a loop; with a different
    # result it is polling.
    result_key: str = ""
    ts: float = field(default_factory=time.time)


def _args_key(args: Any) -> str:
    """A stable identity for a call's arguments, for repeat detection."""
    try:
        return json.dumps(args, sort_keys=True, default=str)[:400]
    except (TypeError, ValueError):
        return str(args)[:400]


def _family_of(capability: str) -> str:
    """The capability family drift detection reasons about.

    Engine tools are ``family.action``. Adapted Jarvis tools are
    ``jarvis.<category>.<read|act>``; their family for drift purposes
    is ``<category>``, not ``"jarvis"`` — otherwise a goal that names
    files/compute/browser never matches the adapted tools that serve
    it, and the third adapted act tool halts the run as scope drift.
    """
    parts = (capability or "").split(".")
    if parts and parts[0] == "jarvis" and len(parts) >= 3:
        return parts[1]
    return parts[0] if parts else ""


class ExecutionGovernor:
    """Watches one run and decides whether it may continue.

    Construct one per run, call :meth:`before_tool` ahead of each tool
    call and :meth:`record` after, and honour a halting verdict. The
    governor never *performs* the stop — the caller does — so there is
    exactly one place in the code that ends a run.
    """

    # Capability families a goal might plausibly need, keyed by a word in
    # the goal. Used only for drift detection, never for authorization.
    _GOAL_HINTS = {
        "email": {"email"}, "inbox": {"email"}, "mail": {"email"},
        "task": {"tasks"}, "board": {"tasks"},
        "code": {"code", "files", "compute"},
        "debug": {"code", "files", "compute"},
        "file": {"files"}, "document": {"files", "office"},
        "browser": {"browser"}, "web": {"browser"},
        "research": {"knowledge", "memory", "browser"},
        "report": {"knowledge", "office", "files"},
        "risk": {"knowledge", "memory"},
        "customer": {"knowledge", "memory"},
    }

    def __init__(
        self,
        *,
        goal: str = "",
        budget: Budget | None = None,
        on_event=None,
        run_id: str = "",
    ) -> None:
        self.goal = goal or ""
        self.budget = budget or Budget()
        self.run_id = run_id
        self._on_event = on_event
        self.started_at = time.time()
        # Seconds spent parked on a human answer (ask_user). Credited
        # back out of the wall clock: a person deliberating must not
        # push the agent into its time ceiling — that would be a system
        # that punishes exactly the runs that asked before acting.
        self._wait_credit_s = 0.0
        # Token + latency accounting, accumulated per model pass when
        # the provider reports it. Zeroes mean "not reported", never
        # "free" — the token ceiling only engages on real counts.
        self.usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "pass_ms": [],
        }
        self.steps: list[Step] = []
        self.verdicts: list[Verdict] = []
        self.halted: Verdict | None = None
        self._warned: set[str] = set()
        self._expected_families = self._families_for_goal(self.goal)
        # The compound checks in before_tool (read counters, decide,
        # append) are not atomic under the GIL, and a VM can be driven
        # from a route thread while a turn thread also runs — so the
        # governor gets its own lock rather than trusting the caller's.
        self._lock = threading.Lock()

    # -- introspection ---------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at - self._wait_credit_s

    def credit_wait(self, seconds: float) -> None:
        """Time the run spent waiting on a human is not time spent
        working; take it back off the clock."""
        with self._lock:
            self._wait_credit_s += max(0.0, float(seconds))

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")

    @property
    def llm_turns(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm")

    @property
    def denials(self) -> int:
        return sum(1 for s in self.steps if s.kind == "denial")

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or 0) + int(
            self.usage.get("completion_tokens") or 0
        )

    def state(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal[:200],
            "elapsed_s": round(self.elapsed, 1),
            "waited_s": round(self._wait_credit_s, 1),
            "tool_calls": self.tool_calls,
            "llm_turns": self.llm_turns,
            "denials": self.denials,
            "steps": len(self.steps),
            "usage": {
                "prompt_tokens": self.usage["prompt_tokens"],
                "completion_tokens": self.usage["completion_tokens"],
                "pass_ms": list(self.usage["pass_ms"]),
            },
            "budget": self.budget.to_dict(),
            "halted": self.halted.to_dict() if self.halted else None,
            "warnings": [v.to_dict() for v in self.verdicts if v.action == WARN],
        }

    def record_llm_pass(self, usage: dict[str, Any] | None = None) -> None:
        """One model pass finished; count it and fold in what it cost.

        The one hook the loop calls per pass — counting and cost stay in
        lockstep, so llm_turns can never say 8 while the tokens say 3.
        """
        self.record(kind="llm")
        if not usage:
            return
        with self._lock:
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(
                usage.get("completion_tokens") or 0
            )
            latency = usage.get("latency_ms")
            if latency is not None:
                # Bounded by max_llm_turns in practice; the cap is a
                # guard against a caller looping outside the budget.
                if len(self.usage["pass_ms"]) < 200:
                    self.usage["pass_ms"].append(round(float(latency), 1))

    # -- the gate --------------------------------------------------------

    def before_tool(self, name: str, args: Any, capability: str = "") -> Verdict:
        """Decide whether this tool call may proceed.

        The read-decide sequence is under the lock: without it, two
        threads sharing one governor (a turn plus a direct invoke)
        could both pass every check on the same counters and both
        proceed past a budget that only had room for one.
        """
        with self._lock:
            if self.halted:
                return self.halted
            for check in (
                self._check_budget,
                self._check_repetition,
                self._check_oscillation,
                self._check_stagnation,
                self._check_denials,
                self._check_untrusted_escalation,
            ):
                verdict = check(name, args, capability)
                if verdict.action != CONTINUE:
                    return self._emit(verdict)
            drift = self._check_drift(name, args, capability)
            if drift.action != CONTINUE:
                return self._emit(drift)
            return OK

    def record(
        self,
        *,
        kind: str = "tool",
        name: str = "",
        args: Any = None,
        capability: str = "",
        ok: bool = True,
        produced: bool = False,
        untrusted: bool = False,
        result: Any = None,
    ) -> None:
        """Record what actually happened, after the fact."""
        with self._lock:
            self.steps.append(
                Step(
                    kind=kind,
                    name=name,
                    args_key=_args_key(args) if args is not None else "",
                    capability=capability,
                    ok=ok,
                    produced=produced,
                    untrusted=untrusted,
                    result_key=_args_key(result) if result is not None else "",
                )
            )

    def record_denial(self, name: str, capability: str = "", reason: str = "") -> None:
        """A refusal — by the ACL, the constitution, or a tool itself.

        Tracked separately because what the agent does *after* a refusal
        is the strongest signal of a run going wrong.
        """
        with self._lock:
            self.steps.append(
                Step(kind="denial", name=name, capability=capability, ok=False)
            )
        log.info(
            "run %s: %s denied (%s) %s", self.run_id, name, capability, reason
        )

    def _emit(self, verdict: Verdict) -> Verdict:
        self.verdicts.append(verdict)
        if verdict.halting:
            self.halted = verdict
            log.warning(
                "run %s HALTED by the governor: %s", self.run_id, verdict.reason
            )
        if self._on_event:
            try:
                self._on_event(
                    {
                        "type": "governor",
                        **verdict.to_dict(),
                        "state": self.state(),
                    }
                )
            except Exception:  # noqa: BLE001
                log.debug("governor event sink failed", exc_info=True)
        return verdict

    # -- the checks ------------------------------------------------------

    def _check_budget(self, *_a) -> Verdict:
        if self.elapsed > self.budget.max_seconds:
            return Verdict(
                HALT,
                f"This run hit its {self.budget.max_seconds:.0f}s time budget.",
                "budget_time",
                {"elapsed_s": round(self.elapsed, 1)},
            )
        if self.tool_calls >= self.budget.max_tool_calls:
            return Verdict(
                HALT,
                f"This run hit its {self.budget.max_tool_calls}-tool-call budget.",
                "budget_tools",
                {"tool_calls": self.tool_calls},
            )
        if self.llm_turns >= self.budget.max_llm_turns:
            return Verdict(
                HALT,
                f"This run hit its {self.budget.max_llm_turns}-turn budget.",
                "budget_turns",
                {"llm_turns": self.llm_turns},
            )
        if (
            self.budget.max_total_tokens > 0
            and self.total_tokens >= self.budget.max_total_tokens
        ):
            return Verdict(
                HALT,
                f"This run hit its {self.budget.max_total_tokens:,}-token "
                "budget.",
                "budget_tokens",
                {"total_tokens": self.total_tokens},
            )
        return OK

    def _check_repetition(self, name: str, args: Any, _cap: str) -> Verdict:
        """Same call again — but is it a LOOP or is it POLLING?

        The published stuck-detector lesson (OpenHands' detector, and
        OpenCode's doom_loop key): the loop signal is an identical
        (action, OBSERVATION) pair, not identical arguments alone. A
        model polling an execution status calls the same tool with the
        same args and gets a *different* answer each time — killing
        that at three calls punished exactly the runs doing it right.
        So: identical args AND identical result halts at max_repeats
        (futility — nothing will change); identical args with varying
        results gets a longer leash (2x) before the ceiling closes,
        because even patient polling must end.
        """
        key = (name, _args_key(args))
        same_args = [
            s for s in self.steps
            if s.kind == "tool" and (s.name, s.args_key) == key
        ]
        # An empty result_key (a caller that recorded no observation)
        # compares equal to other empties on purpose: no observation
        # means no evidence of progress, and the conservative reading
        # of that is the old strict behavior.
        futile = 0
        if same_args:
            last_result = same_args[-1].result_key
            futile = sum(1 for s in same_args if s.result_key == last_result)
        if futile >= self.budget.max_repeats:
            return Verdict(
                HALT,
                f"{name} has been called {futile} times with the same "
                "arguments and the same result each time; repeating it "
                "will not produce a different one.",
                "repetition",
                {"tool": name, "repeats": futile},
            )
        if len(same_args) >= self.budget.max_repeats * 2:
            return Verdict(
                HALT,
                f"{name} has been called {len(same_args)} times with the "
                "same arguments; even polling has to end.",
                "repetition",
                {"tool": name, "repeats": len(same_args)},
            )
        if futile == self.budget.max_repeats - 1:
            return self._warn_once(
                f"repeat:{name}",
                Verdict(
                    WARN,
                    f"{name} keeps returning the same result for the same "
                    "arguments.",
                    "repetition",
                    {"tool": name, "repeats": futile},
                ),
            )
        return OK

    def _check_oscillation(self, name: str, _args: Any, _cap: str) -> Verdict:
        """A, B, A, B — alternating between two approaches."""
        tools = [s.name for s in self.steps if s.kind == "tool"][-5:]
        if len(tools) >= 4:
            seq = [*tools[-4:], name]
            if seq[0] == seq[2] == seq[4] and seq[1] == seq[3] and seq[0] != seq[1]:
                return Verdict(
                    HALT,
                    f"This run is alternating between {seq[0]} and {seq[1]} "
                    "without making progress.",
                    "oscillation",
                    {"cycle": [seq[0], seq[1]]},
                )
        return OK

    def _check_stagnation(self, *_a) -> Verdict:
        recent = [s for s in self.steps if s.kind == "tool"][
            -self.budget.max_stagnant_steps :
        ]
        if len(recent) >= self.budget.max_stagnant_steps and not any(
            s.produced for s in recent
        ):
            return Verdict(
                HALT,
                f"The last {len(recent)} tool calls produced nothing; this run "
                "is not making progress.",
                "stagnation",
                {"steps_without_progress": len(recent)},
            )
        return OK

    def _check_denials(self, name: str, _args: Any, capability: str) -> Verdict:
        """The most serious signal: working around a refusal."""
        if self.denials == 0:
            return OK
        if self.denials >= self.budget.max_denials:
            return Verdict(
                HALT,
                f"This run has been refused {self.denials} times. Stopping "
                "rather than looking for another way around the refusal.",
                "denial_persistence",
                {"denials": self.denials},
            )
        denied_caps = [
            s.capability for s in self.steps if s.kind == "denial" and s.capability
        ]
        if capability and denied_caps:
            highest_denied = max(
                (CAPABILITY_POWER.get(c, 0) for c in denied_caps), default=0
            )
            if CAPABILITY_POWER.get(capability, 0) > highest_denied:
                return Verdict(
                    HALT,
                    f"After being refused {denied_caps[-1]}, this run reached "
                    f"for the more powerful {capability}. Escalating after a "
                    "refusal is not permitted.",
                    "denial_escalation",
                    {"denied": denied_caps[-1], "attempted": capability},
                )
        return OK

    def _check_untrusted_escalation(
        self, name: str, _args: Any, capability: str
    ) -> Verdict:
        """An irreversible external action after consuming content an
        outside party could have authored — the lethal-trifecta shape of
        indirect prompt injection (read attacker content, then act with
        real authority). A WARN, not a halt: reading an email and then
        sending a reply is the job. But the combination is exactly what
        oversight should see, once per run, on the same stream as
        everything else."""
        if CAPABILITY_POWER.get(capability, 0) < 3:
            return OK
        if not any(s.untrusted for s in self.steps if s.kind == "tool"):
            return OK
        sources = sorted({
            s.name for s in self.steps if s.kind == "tool" and s.untrusted
        })[:4]
        return self._warn_once(
            "untrusted_escalation",
            Verdict(
                WARN,
                f"{name} ({capability}) acts externally after this run "
                f"consumed outside-authored content ({', '.join(sources)}). "
                "The action still runs under its own gates; this is the "
                "injection-shaped combination, said out loud.",
                "untrusted_content",
                {"action": name, "capability": capability,
                 "untrusted_sources": sources},
            ),
        )

    def _check_drift(self, name: str, _args: Any, capability: str) -> Verdict:
        """Wandering into work the goal never asked for."""
        if not self._expected_families or not capability:
            return OK
        family = _family_of(capability)
        if family in self._expected_families:
            return OK
        # Read-only excursions are how legitimate research works; only
        # warn. Anything that writes or acts outside the goal's families
        # is a drift worth stopping on.
        if CAPABILITY_POWER.get(capability, 0) <= 0:
            return OK
        used_off_goal = Counter(
            _family_of(s.capability)
            for s in self.steps
            if s.kind == "tool" and s.capability
            and _family_of(s.capability) not in self._expected_families
            and CAPABILITY_POWER.get(s.capability, 0) > 0
        )
        if used_off_goal[family] >= 2:
            return Verdict(
                HALT,
                f"This run keeps using {family}.* capabilities, which the goal "
                f"({self.goal[:60]!r}) does not call for.",
                "scope_drift",
                {"family": family, "expected": sorted(self._expected_families)},
            )
        return self._warn_once(
            f"drift:{family}",
            Verdict(
                WARN,
                f"{name} uses {capability}, outside what the goal suggested.",
                "scope_drift",
                {"family": family},
            ),
        )

    def _warn_once(self, key: str, verdict: Verdict) -> Verdict:
        if key in self._warned:
            return OK
        self._warned.add(key)
        return verdict

    def _families_for_goal(self, goal: str) -> set[str]:
        """Capability families the goal plausibly needs.

        Empty means "no opinion" — drift detection is skipped rather than
        guessing, because a wrong guess here would block legitimate work.
        """
        text = (goal or "").lower()
        families: set[str] = set()
        for word, fams in self._GOAL_HINTS.items():
            if word in text:
                families |= fams
        if families:
            # Reading and remembering are always in scope; every task
            # legitimately begins by finding out what it is about.
            families |= {"knowledge", "memory", "practices", "orders"}
        return families


__all__ = [
    "ExecutionGovernor",
    "Budget",
    "Verdict",
    "CONTINUE",
    "WARN",
    "HALT",
    "CAPABILITY_POWER",
]
