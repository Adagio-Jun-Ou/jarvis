"""Core data model types for Omnigent (vendored, adapted).

This is an adapted subset of ``omnigent/inner/datamodel.py`` from
https://github.com/omnigent-ai/omnigent (Apache-2.0).  The
adaptations:

* **Removed**: the entire ``OSEnvSpec`` / ``TerminalEnvSpec`` /
  ``CredentialProxySpec`` / ``CredentialSourceSpec`` /
  ``CredentialProxyEntry`` / ``OSEnvSandboxSpec`` family.  Alex 2.0
  deploys as a Databricks App behind a Unity Catalog policy and does
  not run subprocess sandboxes.
* **Trimmed**: ``AgentDef`` keeps only the fields Alex 2.0 actually
  reads (``name``, ``prompt``, ``tools``, ``policies``, ``params``,
  ``executor``, ``skills``, ``metadata``).  The dropped fields are
  upstream-specific to Claude/Codex/Cursor harnesses that we do not
  vendor.
* **Kept**: ``Message``, ``History``, ``Connection``, ``Memory``,
  ``MemoryConfig``, ``Credentials``, ``ParamDef``, ``SessionState``,
  ``ExecutorSpec``, ``LabelSchemaRule``, ``AgentDef.from_yaml``.

The file is otherwise structurally identical to the upstream source
so a future swap back to a full upstream install is a one-line
``import omnigent.inner.datamodel`` away.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias

MessageContent: TypeAlias = str | dict[str, Any]
MessageMetadata: TypeAlias = dict[str, Any]
AskContent: TypeAlias = Any
LabelValue: TypeAlias = Any
ParamDefault: TypeAlias = Any

ToolRegistry: TypeAlias = dict[str, Any]
PolicyRegistry: TypeAlias = dict[str, Any]
SkillRegistry: TypeAlias = dict[str, Any]
AgentMetadata: TypeAlias = dict[str, Any]
Workflow: TypeAlias = Any


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool_call", "tool_result"]
    content: MessageContent
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str | None = None
    metadata: MessageMetadata = field(default_factory=dict)


@dataclass
class AskRequest:
    reason: str
    phase: str
    content: AskContent = None
    session_id: str | None = None
    session_label: str | None = None


class History:
    def __init__(self, name: str = "self") -> None:
        self.name = name
        self.messages: list[Message] = []

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def get_context_window(self, max_tokens: int | None = None) -> list[Message]:  # noqa: ARG002
        return list(self.messages)

    def search(self, query: str) -> list[Message]:
        results: list[Message] = []
        for msg in self.messages:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if query.lower() in text.lower():
                results.append(msg)
        return results

    def as_text(self) -> str:
        return "\n".join(
            f"[{m.role}] {m.content if isinstance(m.content, str) else m.content!r}"
            for m in self.messages
        )

    def __len__(self) -> int:
        return len(self.messages)


class Connection:
    def __init__(self, name: str = "primary", conn_type: str = "chat") -> None:
        import asyncio

        self.id: str = str(uuid.uuid4())
        self.name = name
        self.type = conn_type
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._outbox: asyncio.Queue[str] = asyncio.Queue()

    async def receive(self) -> Message:
        return await self._inbox.get()

    def try_receive_nowait(self) -> Message | None:
        import asyncio

        try:
            return self._inbox.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def send(self, content: str) -> None:
        await self._outbox.put(content)

    async def inject_user_message(self, text: str) -> None:
        msg = Message(role="user", content=text, source=self.name)
        await self._inbox.put(msg)

    async def read_agent_response(self) -> str:
        return await self._outbox.get()


@dataclass
class MemoryConfig:
    scope: Literal["per_session", "per_user", "cross_user"] = "per_session"


class Memory:
    def __init__(self, name: str, scope: str = "per_session") -> None:
        self.name = name
        self.scope = scope
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    def peek(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        if prefix is None:
            return list(self._store.keys())
        return [k for k in self._store if k.startswith(prefix)]

    async def search(self, query: str, limit: int = 10) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for k, v in self._store.items():
            if query.lower() in v.lower():
                results.append((k, v))
                if len(results) >= limit:
                    break
        return results


@dataclass
class Credentials:
    token: str | None = None
    scopes: set[str] = field(default_factory=set)
    principal: str | None = None
    expires_at: datetime | None = None

    def attenuate(self, scopes: set[str]) -> Credentials:
        if not scopes.issubset(self.scopes):
            raise ValueError(
                f"Cannot attenuate: {scopes - self.scopes} not in current scopes"
            )
        return Credentials(
            token=self.token,
            scopes=scopes,
            principal=self.principal,
            expires_at=self.expires_at,
        )


@dataclass
class ParamDef:
    type: str = "string"
    description: str | None = None
    default: ParamDefault = None


class SessionState(enum.Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SLEEPING = "SLEEPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ExecutorSpec:
    model: str | None = None
    harness: str | None = None
    profile: str | None = None
    auth: object | None = None


@dataclass
class LabelSchemaRule:
    values: list[str] = field(default_factory=list)

    def normalize(self, value: LabelValue) -> str | None:
        candidate = str(value)
        return candidate if candidate in self.values else None

    def allows(self, new_value: str) -> bool:
        return new_value in self.values


@dataclass
class AgentDef:
    name: str | None = None
    prompt: str | None = None
    instructions: str | None = None
    tools: ToolRegistry = field(default_factory=dict)
    input_type: str | None = None
    output_type: str | None = None
    executor: ExecutorSpec | None = None
    policies: PolicyRegistry = field(default_factory=dict)
    params: dict[str, ParamDef] = field(default_factory=dict)
    memories: dict[str, MemoryConfig] = field(default_factory=dict)
    async_enabled: bool = True
    cancellable: bool = True
    workflow: Workflow = None
    metadata: AgentMetadata = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    label_schema: dict[str, LabelSchemaRule] = field(default_factory=dict)
    ask_timeout: float | None = None
    policy_transparency: bool = False

    @staticmethod
    def from_yaml(path: str) -> AgentDef:
        from .loader import load_agent_def

        return load_agent_def(path)
