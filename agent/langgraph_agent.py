"""A LangGraph agent driven by Alex's model and Alex's governed tools.

Alex's own loop is `agent/vm.py`, and this does not replace it — see
"Why the agent loop is ours" in `docs/ARCHITECTURE.md`. What this gives
you is the other direction: a real LangGraph graph, so a team that builds
on LangGraph can run Alex's tools inside their own orchestration, and so
the option is a working thing rather than a claim in a README.

Two pieces:

* :class:`AlexChatModel` — Alex's LLM behind LangChain's
  ``BaseChatModel``, including tool-calling. Whatever `alex.llm` is
  routed to (Ollama locally, Databricks serving in production) is what
  the graph runs on, so the graph does not acquire its own model
  configuration to drift from the rest of the system.
* :func:`build_agent` — ``create_react_agent`` over
  :func:`agent.langchain_tools.governed_tools`.

The governance point is the same one that module makes: the tools are
the VM's, so the ACL, the governor, the constitution and the audit log
all still apply inside somebody else's graph. A LangGraph agent here is
not a way around the control stack; it is a different way into it.

Both LangChain and LangGraph are optional dependencies.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

log = logging.getLogger("agent.langgraph_agent")

_MISSING = (
    "langgraph is not installed. It is an optional dependency: "
    "`pip install langgraph langchain-core` to drive Alex's governed "
    "tools from a LangGraph agent."
)


def langgraph_available() -> bool:
    try:
        import langgraph.prebuilt  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def to_alex_message(message: Any) -> dict[str, Any]:
    """One LangChain message in the wire shape `alex.llm` sends.

    The OpenAI/Ollama chat shape, which is what both backends speak.
    Tool results carry the tool's *name* as well as its content: a model
    reading a bare result has to infer which call it answered, and with
    parallel tool calls that inference is a guess.
    """
    kind = getattr(message, "type", None) or ""
    content = getattr(message, "content", "") or ""
    if not isinstance(content, str):
        # Multi-part content (text + images). Keep the text; this model
        # path is text-only and silently dropping everything would be
        # worse than dropping the parts we cannot send.
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        content = "\n".join(x for x in parts if x)

    if kind == "system":
        return {"role": "system", "content": content}
    if kind == "human":
        return {"role": "user", "content": content}
    if kind == "tool":
        out = {"role": "tool", "content": content}
        name = getattr(message, "name", "")
        if name:
            out["name"] = name
        call_id = getattr(message, "tool_call_id", "")
        if call_id:
            out["tool_call_id"] = call_id
        return out

    # Assistant. Tool calls have to survive the round trip or the model
    # cannot see what it already asked for and will ask again.
    out = {"role": "assistant", "content": content}
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        out["tool_calls"] = [
            {
                "id": c.get("id") or "",
                "type": "function",
                "function": {
                    "name": c.get("name", ""),
                    "arguments": c.get("args", {}) or {},
                },
            }
            for c in calls
        ]
    return out


def _to_langchain_ai(response: Any):
    """Alex's ``LLMResponse`` as a LangChain ``AIMessage``."""
    from langchain_core.messages import AIMessage

    calls = []
    for call in getattr(response, "tool_calls", None) or []:
        calls.append(
            {
                "name": call.name,
                "args": dict(call.args or {}),
                # LangGraph pairs the ToolMessage back to the call by id,
                # so an empty one would silently break multi-tool turns.
                "id": call.id or f"call_{uuid.uuid4().hex[:16]}",
                "type": "tool_call",
            }
        )
    return AIMessage(content=getattr(response, "content", "") or "", tool_calls=calls)


# ---------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------


def make_chat_model(*, temperature: float = 0.1, llm: Any = None):
    """Alex's LLM as a LangChain ``BaseChatModel``.

    Built inside a function rather than at module scope so importing this
    module does not require LangChain — the class has to subclass
    ``BaseChatModel``, which means the dependency would otherwise be
    needed at import time by everything that touches the agent package.
    """
    if not langchain_core_available():
        raise ImportError(_MISSING)

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class AlexChatModel(BaseChatModel):
        """Whatever `alex.llm` is routed to, behind LangChain's interface."""

        temperature: float = 0.1
        bound_tools: list[dict] = []

        @property
        def _llm_type(self) -> str:
            return "alex"

        def bind_tools(self, tools, **kwargs: Any):  # type: ignore[override]
            converted = [convert_to_openai_tool(t) for t in tools]
            return self.__class__(
                temperature=self.temperature, bound_tools=converted
            )

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            from alex.llm import get_llm

            client = llm or get_llm()
            response = client.invoke(
                [to_alex_message(m) for m in messages],
                tools=self.bound_tools or None,
                temperature=self.temperature,
            )
            return ChatResult(
                generations=[ChatGeneration(message=_to_langchain_ai(response))]
            )

    return AlexChatModel(temperature=temperature)


def langchain_core_available() -> bool:
    from agent.langchain_tools import langchain_available

    return langchain_available()


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def build_agent(vm, *, tool_names: list[str] | None = None, temperature: float = 0.1):
    """A LangGraph ReAct agent over this VM's governed tools.

    The VM is required, not optional: it is what carries the principal,
    the ACL, the governor and the audit sink. A graph built without one
    would have no way to be governed, which is the only reason to build
    it here rather than in the caller's own code.
    """
    if not langgraph_available():
        raise ImportError(_MISSING)
    from langgraph.prebuilt import create_react_agent

    from agent.langchain_tools import governed_tools

    tools = governed_tools(vm, names=tool_names)
    if not tools:
        # Better than a graph that silently cannot act: an agent with no
        # tools looks like a model that will not use them.
        log.warning(
            "no tools exported for %s — the principal's ACL allows none of "
            "the registry, so this graph can only talk",
            getattr(vm, "user_email", "?"),
        )
    return create_react_agent(make_chat_model(temperature=temperature), tools)


def run(vm, text: str, *, tool_names: list[str] | None = None) -> str:
    """One turn through the graph. Returns the final assistant text."""
    graph = build_agent(vm, tool_names=tool_names)
    result = graph.invoke({"messages": [("user", text)]})
    messages = result.get("messages") or []
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and getattr(message, "content", ""):
            return str(message.content)
    return ""
