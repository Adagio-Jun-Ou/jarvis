"""The serving contract — how Alex is invoked on Databricks.

Model Serving does not care what runs inside the agent. It cares about
one interface: MLflow's ``ResponsesAgent``, which is what Agent
Evaluation, the AI Playground, Agent Monitoring and the review app all
speak. Implementing it is what makes Alex deployable; it is also what
makes the choice of agent framework a private decision rather than a
platform commitment, because any loop that can answer ``predict`` is a
valid agent.

That is the whole reason this module exists rather than a LangGraph
rewrite. The framework question and the deployment question look like
the same question and are not: adopting LangGraph would not have
deployed anything by itself, and implementing this deploys the agent we
already have. See ``docs/ARCHITECTURE.md``.

The translation is kept as **pure functions** below, with the MLflow
subclass a thin shell over them. MLflow is a heavy import that only
exists in the serving image, so the logic that decides what the caller
sees has to be testable without it.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("agent.serving")

# Events the VM emits that a serving caller should see as visible work,
# mapped to the Responses item type that represents them. Anything not
# named here is deliberately withheld: `context` carries retrieved
# passages, `reasoning` carries the model's private deliberation, and
# neither belongs in a serving response by default — they are for the
# Transparency surface, where the user has an identity and a scope.
_TOOL_EVENTS = frozenset({"tool"})
_REFUSAL_EVENTS = frozenset({"law_block", "error"})


def new_item_id(prefix: str = "msg") -> str:
    """A stable id for one output item. Streaming deltas and the final
    ``done`` item must share it, or the client cannot assemble them."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def text_delta(item_id: str, text: str) -> dict[str, Any]:
    """One streamed chunk of assistant text."""
    return {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "delta": text,
    }


def output_text_item(item_id: str, text: str) -> dict[str, Any]:
    """The completed assistant message."""
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def function_call_item(
    item_id: str, name: str, args: dict | None, call_id: str
) -> dict[str, Any]:
    """A tool call, surfaced so evaluation and monitoring can see which
    capability was exercised — the audit ledger is the system of record,
    but a reviewer reading a trace should not have to leave it."""
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args or {}, ensure_ascii=False),
    }


def function_call_output_item(
    item_id: str, call_id: str, output: Any
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call_output",
        "call_id": call_id,
        "output": _stringify(output),
    }


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def done_item(item: dict[str, Any]) -> dict[str, Any]:
    """Wrap a completed item in the event that closes it."""
    return {"type": "response.output_item.done", "item": item}


def user_text(messages: list[dict[str, Any]]) -> str:
    """The new user message from a Responses-shaped input list.

    Serving hands over the whole conversation each call, while the VM
    owns its own history and wants only the new turn. Taking the last
    user message is what reconciles the two.
    """
    for message in reversed(messages or []):
        if (message or {}).get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
                and part.get("type") in ("input_text", "text", "output_text")
            ]
            if parts:
                return "\n".join(p for p in parts if p)
    return ""


def prior_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything before the new user message, in the VM's own shape.

    Dropped rather than translated: tool calls and their outputs. The VM
    reconstructs those from its own session, and replaying a caller's
    account of what tools ran would let a caller *assert* that a
    consequential tool already succeeded — a confirmation it never gave.
    """
    out: list[dict[str, Any]] = []
    for message in (messages or [])[:-1]:
        role = (message or {}).get("role")
        if role not in ("user", "assistant", "system"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("text")
            )
        if content:
            out.append({"role": role, "content": str(content)})
    return out


def event_to_items(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one VM event into zero or more Responses output items.

    Returns a list because a completed tool call is two items — the call
    and its output — and because most events translate to none at all.
    """
    kind = (event or {}).get("type")
    if kind in _TOOL_EVENTS:
        call_id = f"call_{uuid.uuid4().hex[:20]}"
        items = [
            function_call_item(
                new_item_id("fc"), event.get("tool") or "unknown",
                event.get("args"), call_id,
            )
        ]
        if event.get("result") is not None:
            items.append(
                function_call_output_item(
                    new_item_id("fco"), call_id, event["result"]
                )
            )
        return items
    if kind in _REFUSAL_EVENTS:
        # A refusal is the answer, not an exception. Serving callers that
        # only read assistant text would otherwise see an empty response
        # and report the agent as broken rather than as governed.
        detail = event.get("detail") or event.get("message") or ""
        article = event.get("article")
        text = (
            f"Refused under constitution article {article}: {detail}"
            if article else f"Could not complete this: {detail}"
        )
        return [output_text_item(new_item_id("refusal"), text)]
    return []


def make_vm_for_request(
    session_id: str, user_email: str, *, plan_mode: bool = False
):
    """Build the VM for one serving request."""
    from agent.vm import make_vm

    return make_vm(
        session_id=session_id,
        user_email=user_email,
        instance_scope="default",
        plan_mode=plan_mode,
    )


def _custom_inputs(request: Any) -> dict[str, Any]:
    """The request's custom_inputs, whatever shape it arrived in.

    Only NON-identity flags may be honoured from here. Identity comes
    from the platform context (resolve_identity); a mode flag like
    plan_mode is safe from the payload because switching it on can only
    NARROW what the VM may do — a caller restricting themselves is not
    an escalation, and off is the default.
    """
    if isinstance(request, dict):
        raw = request.get("custom_inputs")
    else:
        raw = getattr(request, "custom_inputs", None)
    return raw if isinstance(raw, dict) else {}


def resolve_identity(context: Any, default_email: str = "") -> tuple[str, str]:
    """``(session_id, user_email)`` for a serving request.

    Identity comes from the request *context*, never from the payload:
    on Databricks the signed-in user arrives out of band, and letting a
    caller name themselves in the body would make every ACL in the
    system advisory. MLflow's per-conversation id is exactly the session
    identity the VM wants — the same conversation reaching the same VM
    is what makes its history and its governor budget mean anything.
    """
    session_id = getattr(context, "conversation_id", None) or f"serving-{uuid.uuid4().hex[:12]}"
    user_email = getattr(context, "user_id", None) or default_email
    return str(session_id), str(user_email)


class AlexResponsesAgent:
    """Alex behind MLflow's ``ResponsesAgent`` contract.

    Deliberately not a subclass at import time: ``mlflow`` exists in the
    serving image and not in the test environment, and a module that
    cannot be imported without it could not be unit-tested at all. The
    real base class is mixed in by :func:`build_agent` when serving.
    """

    def __init__(self, default_user_email: str = "") -> None:
        self._default_user_email = default_user_email

    # -- the contract ---------------------------------------------------

    def predict(self, request: Any) -> dict[str, Any]:
        """Answer one turn. Non-streaming callers get the whole response."""
        items: list[dict[str, Any]] = []
        for event in self.predict_stream(request):
            if event.get("type") == "response.output_item.done":
                items.append(event["item"])
        return {"output": items}

    def predict_stream(self, request: Any) -> Iterator[dict[str, Any]]:
        """Answer one turn, streaming.

        Text is streamed as it is generated; tool calls surface as they
        happen. The VM runs on a worker thread because its loop is
        synchronous and the events arrive on a queue.
        """
        import queue
        import threading

        messages = _request_messages(request)
        context = getattr(request, "context", None)
        session_id, user_email = resolve_identity(
            context, self._default_user_email
        )
        # custom_inputs used to be read by NOTHING here — the app proxy
        # dutifully forwarded flags that died at this boundary, so plan
        # mode was silently a dev-only feature in production.
        extras = _custom_inputs(request)
        vm = make_vm_for_request(
            session_id, user_email,
            plan_mode=bool(extras.get("plan_mode")),
        )

        events: queue.Queue = queue.Queue()
        finished = object()
        unsubscribe = vm.subscribe_sse(events.put)
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                text, _history = vm.run_turn(
                    text=user_text(messages),
                    history=prior_history(messages),
                )
                result["text"] = text
            except Exception as exc:  # noqa: BLE001
                log.exception("serving turn failed")
                result["error"] = str(exc)
            finally:
                events.put(finished)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        text_id = new_item_id()
        streamed: list[str] = []
        try:
            while True:
                event = events.get()
                if event is finished:
                    break
                payload = _as_dict(event)
                if payload.get("type") == "text" and payload.get("text"):
                    streamed.append(payload["text"])
                    yield text_delta(text_id, payload["text"])
                    continue
                for item in event_to_items(payload):
                    yield done_item(item)
        finally:
            if callable(unsubscribe):
                unsubscribe()
            worker.join(timeout=5)
            try:
                vm.close()
            except Exception:  # noqa: BLE001
                log.debug("VM close failed after serving turn", exc_info=True)

        # The final item carries the whole answer even when it was
        # streamed, because a client that reconnected mid-turn has only
        # this. Prefer run_turn's return value: inline review may have
        # replaced the text that was streamed.
        final = result.get("text") or "".join(streamed)
        if result.get("error") and not final:
            final = f"Could not complete this: {result['error']}"
        yield done_item(output_text_item(text_id, final))


def _as_dict(event: Any) -> dict[str, Any]:
    """VM events arrive as :class:`alex.events.Event`; serving wants dicts."""
    if isinstance(event, dict):
        return event
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    to_json = getattr(event, "to_json", None)
    if callable(to_json):
        try:
            return json.loads(to_json())
        except (TypeError, ValueError):
            return {}
    return {}


def _request_messages(request: Any) -> list[dict[str, Any]]:
    if isinstance(request, dict):
        raw = request.get("input") or request.get("messages") or []
    else:
        raw = getattr(request, "input", None) or getattr(request, "messages", None) or []
    out = []
    for message in raw:
        if isinstance(message, dict):
            out.append(message)
        elif hasattr(message, "model_dump"):
            out.append(message.model_dump())
    return out


def build_agent(default_user_email: str = ""):
    """The object MLflow logs and serves.

    Mixes :class:`AlexResponsesAgent` with MLflow's ``ResponsesAgent`` at
    call time so the module imports cleanly without MLflow installed.
    """
    from mlflow.pyfunc import ResponsesAgent

    class _Served(AlexResponsesAgent, ResponsesAgent):
        pass

    return _Served(default_user_email=default_user_email)
