"""
Omnigent — open-source AI agent framework, vendored as the agent-loop
backbone of Alex 2.0.

This package is an adapted subset of the public Omnigent project
(https://github.com/omnigent-ai/omnigent), Apache-2.0.  We vendor only
the surfaces that are useful to Alex (the datamodel, the executor
interface, the tool/policy/tracing primitives, and a Databricks model
executor) and adapt them so they work against our existing
``alex.llm.OllamaCloudLLM`` (dev) and ``alex.llm.DatabricksServingLLM``
(prod) and against MLflow tracing (which is already in
``agent/requirements.txt``).  We do NOT vendor the Claude/Codex/Cursor/
Goose/Kimi/Kiro/Antigravity/ACP/IPC subprocess executors, the
``bwrap``/``seatbelt``/``_seccomp`` sandbox backends, the
``credential_proxy`` machinery, the ``~/.databrickscfg`` resolver, or
the ``omni`` CLI / web UI — Alex deploys as a Databricks App on top of
the same Unity Catalog + Lakebase it owns, and does not need any of
that surface.

See ``LICENSE`` in this directory and ``docs/OMNIGENT_INTEGRATION.md``
at the repo root for what was kept, what was adapted, and what was
removed.  The public surface mirrors the upstream ``omnigent.inner.*``
modules so a future migration to a full upstream install is
mechanical.

Public symbols re-exported here:

* **Datamodel** — :class:`AgentDef`, :class:`History`, :class:`Message`,
  :class:`AskRequest`, :class:`Memory`, :class:`MemoryConfig`,
  :class:`Credentials`, :class:`ParamDef`, :class:`ExecutorSpec`,
  :class:`SessionState`.
* **Executor** — :class:`Executor`, :class:`ExecutorConfig`,
  :class:`ExecutorEvent`, :class:`TextChunk`, :class:`ReasoningChunk`,
  :class:`ToolCallRequest`, :class:`ToolCallComplete`,
  :class:`TurnComplete`, :class:`TurnCancelled`, :class:`ExecutorError`,
  :class:`MockExecutor`, plus our :class:`DatabricksExecutor` and
  :class:`LocalOllamaExecutor`.
* **Tools** — :class:`Tool`, :class:`FunctionTool`, :class:`MCPTool`,
  :class:`AgentTool`, :class:`SkillTool`, :class:`HandoffTool`,
  :class:`InheritedTool`, :class:`SelfAgentTool`.
* **Policies** — :class:`Policy`, :class:`FunctionPolicy`,
  :class:`PromptPolicy`, :class:`PolicyResult`, :class:`PolicyAction`,
  :class:`PolicyRuntimeContext`.
* **Tracing** — :func:`enable_tracing`, :func:`disable_tracing`,
  :func:`is_tracing_enabled`, :class:`TracingContext`.
* **Loader** — :func:`load_agent_def`.
* **Bridge** — :func:`alex_bridge.build_agent_def`, :func:`alex_bridge.run_turn`.
"""

from __future__ import annotations

import hashlib as _fips_safe_hashlib

_fips_safe_orig_md5 = _fips_safe_hashlib.md5


def _fips_safe_md5(*args, **kwargs):
    kwargs.setdefault("usedforsecurity", False)
    return _fips_safe_orig_md5(*args, **kwargs)


_fips_safe_hashlib.md5 = _fips_safe_md5  # type: ignore[assignment]

from . import _env_compat  # noqa: E402,F401
from ._env_compat import mirror_legacy_env as _mirror_legacy_env  # noqa: E402

_mirror_legacy_env()

from .databricks_executor import DatabricksExecutor  # noqa: E402
from .datamodel import (  # noqa: E402
    AgentDef,
    AskRequest,
    Connection,
    Credentials,
    History,
    Memory,
    MemoryConfig,
    Message,
    ParamDef,
    SessionState,
)
from .executor import (  # noqa: E402
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    MessageSplit,
    MockExecutor,
    ReasoningChunk,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolCallStatus,
    TurnCancelled,
    TurnComplete,
    classify_tool_result,
    iterate_blocking_stream,
    split_transient_tail,
)
from .loader import load_agent_def  # noqa: E402
from .local_ollama_executor import LocalOllamaExecutor  # noqa: E402
from .openrouter_executor import OpenRouterExecutor  # noqa: E402
from .policies import (  # noqa: E402
    FunctionPolicy,
    Policy,
    PolicyAction,
    PolicyResult,
    PolicyRuntimeContext,
    PromptPolicy,
)
from .tools import (  # noqa: E402
    AgentTool,
    CancellableFunctionTool,
    FunctionTool,
    HandoffTool,
    InheritedTool,
    MCPTool,
    SelfAgentTool,
    SkillTool,
    Tool,
)
from .tracing import (  # noqa: E402
    TracingContext,
    disable_tracing,
    enable_tracing,
    is_tracing_enabled,
)

__all__ = [
    "AgentDef",
    "AgentTool",
    "AskRequest",
    "CancellableFunctionTool",
    "Connection",
    "Credentials",
    "DatabricksExecutor",
    "Executor",
    "ExecutorConfig",
    "ExecutorError",
    "ExecutorEvent",
    "FunctionPolicy",
    "FunctionTool",
    "History",
    "HandoffTool",
    "InheritedTool",
    "LocalOllamaExecutor",
    "OpenRouterExecutor",
    "MCPTool",
    "Memory",
    "MemoryConfig",
    "Message",
    "MessageSplit",
    "MockExecutor",
    "ParamDef",
    "Policy",
    "PolicyAction",
    "PolicyResult",
    "PolicyRuntimeContext",
    "PromptPolicy",
    "ReasoningChunk",
    "SelfAgentTool",
    "SessionState",
    "SkillTool",
    "TextChunk",
    "Tool",
    "ToolCallComplete",
    "ToolCallRequest",
    "ToolCallStatus",
    "TracingContext",
    "TurnCancelled",
    "TurnComplete",
    "classify_tool_result",
    "disable_tracing",
    "enable_tracing",
    "is_tracing_enabled",
    "iterate_blocking_stream",
    "load_agent_def",
    "split_transient_tail",
]
