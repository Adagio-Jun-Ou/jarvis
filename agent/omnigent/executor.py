"""Executor adapter interface for Omnigent (vendored, adapted).

This is an adapted subset of ``omnigent/inner/executor.py`` from
https://github.com/omnigent-ai/omnigent (Apache-2.0).  The
adaptations:

* **Kept** as-is: ``ExecutorConfig``, every event dataclass
  (``TextChunk``, ``ReasoningChunk``, ``ToolCallRequest``,
  ``ToolCallComplete``, ``TurnComplete``, ``TurnCancelled``,
  ``CompactionComplete``, ``ExecutorError``), ``iterate_blocking_stream``,
  ``MessageSplit`` / ``split_transient_tail``,
  ``classify_tool_result`` / ``ToolResultClassification``, and
  ``MockExecutor``.
* **Trimmed**: removed the ``MessageSplit``-related ``CompactionComplete``
  persistence hook (we persist our own ``History`` through
  ``agent/session_manager.py``).

The base :class:`Executor` is unchanged in shape: subclasses
implement ``async def run_turn(...) -> AsyncIterator[ExecutorEvent]``.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

Message: TypeAlias = dict[str, Any]
ToolSpec: TypeAlias = dict[str, Any]
ToolArgs: TypeAlias = dict[str, Any]
ToolResult: TypeAlias = Any
ToolCallMetadata: TypeAlias = dict[str, Any]
ExecutorExtra: TypeAlias = dict[str, Any]
EnqueuedContent: TypeAlias = Any
ProviderStreamItem: TypeAlias = Any


@runtime_checkable
class _ClosableIterator(Protocol):
    def close(self) -> None: ...


@dataclass
class ExecutorConfig:
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 100000
    extra: ExecutorExtra = field(default_factory=dict)


@dataclass
class ExecutorEvent:
    pass


@dataclass
class TextChunk(ExecutorEvent):
    text: str


@dataclass
class ReasoningChunk(ExecutorEvent):
    delta: str
    event_type: str


@dataclass
class ToolCallRequest(ExecutorEvent):
    name: str
    args: ToolArgs = field(default_factory=dict)
    metadata: ToolCallMetadata = field(default_factory=dict)


@dataclass
class TurnComplete(ExecutorEvent):
    response: str | None = None
    modified_by_policy: bool = False
    continue_turn: bool = False
    usage: dict[str, Any] | None = None


@dataclass
class PassUsage(ExecutorEvent):
    """Token accounting for ONE model pass, emitted when the provider
    reports it. Separate from TurnComplete because tool-call passes end
    the stream without a TurnComplete — and those are most of a working
    turn's passes; usage that only rode the final pass undercounted the
    run several-fold."""

    usage: dict[str, Any] = field(default_factory=dict)


class ToolCallStatus(str, enum.Enum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class ToolCallComplete(ExecutorEvent):
    name: str
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    result: ToolResult = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: ToolCallMetadata = field(default_factory=dict)


@dataclass
class CompactionComplete(ExecutorEvent):
    summary: str
    token_count: int
    model: str | None = None
    compacted_messages: list[dict[str, Any]] | None = None


@dataclass
class TurnCancelled(ExecutorEvent):
    reason: str = "user_cancelled"
    phase: str = "model"


@dataclass
class ExecutorError(ExecutorEvent):
    message: str
    retryable: bool = False


def _close_stream_quietly(stream: Iterator[ProviderStreamItem]) -> None:
    if isinstance(stream, _ClosableIterator):
        with contextlib.suppress(Exception):
            stream.close()


@dataclass(frozen=True)
class _StreamItem:
    payload: ProviderStreamItem


@dataclass(frozen=True)
class _StreamError:
    exc: BaseException


@dataclass(frozen=True)
class _StreamDone:
    pass


async def iterate_blocking_stream(
    stream: Iterator[ProviderStreamItem],
) -> AsyncIterator[ProviderStreamItem]:
    import queue as sync_queue

    signal_queue: sync_queue.Queue[_StreamItem | _StreamError | _StreamDone] = (
        sync_queue.Queue()
    )
    stop = threading.Event()

    def _worker() -> None:
        try:
            for item in stream:
                if stop.is_set():
                    break
                signal_queue.put(_StreamItem(payload=item))
        except Exception as exc:  # noqa: BLE001
            signal_queue.put(_StreamError(exc=exc))
        finally:
            _close_stream_quietly(stream)
            signal_queue.put(_StreamDone())

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    loop = asyncio.get_running_loop()
    try:
        while True:
            # Block in a worker thread on the queue rather than spinning
            # a 1ms poll: the poll woke this coroutine a thousand times a
            # second for the entire duration of every model stream. The
            # short timeout keeps cancellation responsive.
            try:
                signal = await loop.run_in_executor(
                    None, lambda: signal_queue.get(timeout=0.25)
                )
            except sync_queue.Empty:
                continue
            if isinstance(signal, _StreamItem):
                yield signal.payload
                continue
            if isinstance(signal, _StreamError):
                raise signal.exc
            return
    finally:
        stop.set()
        _close_stream_quietly(stream)
        thread.join(timeout=0.5)


@dataclass(frozen=True)
class MessageSplit:
    persisted: list[Message]
    transient: list[Message]


def split_transient_tail(messages: list[Message]) -> MessageSplit:
    split_idx = len(messages)
    while split_idx > 0:
        meta = messages[split_idx - 1].get("metadata", {})
        if isinstance(meta, dict) and meta.get("framework"):
            split_idx -= 1
            continue
        break
    return MessageSplit(
        persisted=messages[:split_idx],
        transient=messages[split_idx:],
    )


@dataclass(frozen=True)
class ToolResultClassification:
    status: ToolCallStatus
    error: str


def classify_tool_result(
    result: ToolResult,
    *,
    fallback_to_string: bool = False,
) -> ToolResultClassification:
    if result is None:
        return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")

    if isinstance(result, dict):
        if result.get("cancelled"):
            return ToolResultClassification(
                status=ToolCallStatus.CANCELLED,
                error=str(result.get("reason", "cancelled")),
            )
        if result.get("error"):
            return ToolResultClassification(
                status=ToolCallStatus.ERROR,
                error=str(result["error"]),
            )
        if result.get("blocked"):
            return ToolResultClassification(
                status=ToolCallStatus.BLOCKED,
                error=str(result.get("reason", "BLOCKED")),
            )
        for key in ("content", "result", "output", "text"):
            if key in result:
                nested = classify_tool_result(
                    result[key],
                    fallback_to_string=fallback_to_string,
                )
                if nested.error:
                    return nested
        return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")

    if isinstance(result, list):
        for item in result:
            nested = classify_tool_result(
                item,
                fallback_to_string=fallback_to_string,
            )
            if nested.error:
                return nested
        return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")

    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")
        try:
            parsed = json.loads(stripped)
        except (TypeError, json.JSONDecodeError):
            if fallback_to_string:
                return ToolResultClassification(
                    status=ToolCallStatus.ERROR,
                    error=stripped,
                )
            return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")
        nested = classify_tool_result(
            parsed,
            fallback_to_string=fallback_to_string,
        )
        if nested.error:
            return nested
        if fallback_to_string:
            return ToolResultClassification(
                status=ToolCallStatus.ERROR,
                error=stripped,
            )
        return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")

    if hasattr(result, "model_dump"):
        return classify_tool_result(
            result.model_dump(by_alias=True, exclude_none=True),
            fallback_to_string=fallback_to_string,
        )

    if hasattr(result, "__dict__"):
        return classify_tool_result(
            vars(result),
            fallback_to_string=fallback_to_string,
        )

    if fallback_to_string:
        return ToolResultClassification(status=ToolCallStatus.ERROR, error=str(result))
    return ToolResultClassification(status=ToolCallStatus.SUCCESS, error="")


class Executor:
    """Abstract interface for LLM backends and agent harnesses.

    Subclass this and implement :meth:`run_turn` for each backend.
    """

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        raise NotImplementedError
        yield  # pragma: no cover

    def supports_streaming(self) -> bool:
        return False

    def supports_tool_calling(self) -> bool:
        return True

    def handles_tools_internally(self) -> bool:
        return False

    def max_context_tokens(self) -> int | None:
        return None

    async def close_session(self, session_key: str) -> None:
        return

    async def interrupt_session(self, session_key: str) -> bool:
        return False

    async def enqueue_session_message(
        self, session_key: str, content: EnqueuedContent
    ) -> bool:
        return False

    def supports_live_message_queue(self) -> bool:
        return False

    def supports_tool_boundary_interrupt(self) -> bool:
        return False

    def supports_stepwise_internal_turns(self) -> bool:
        return False

    async def close(self) -> None:
        return


class MockExecutor(Executor):
    def __init__(self) -> None:
        self._turns: list[list[ExecutorEvent]] = []

    def enqueue_response(self, text: str) -> None:
        self._turns.append([TurnComplete(response=text)])

    def enqueue_tool_call(
        self,
        tool_name: str,
        args: ToolArgs | None = None,
        follow_up_response: str | None = None,
    ) -> None:
        events: list[ExecutorEvent] = [
            ToolCallRequest(name=tool_name, args=args or {}),
        ]
        self._turns.append(events)
        if follow_up_response is not None:
            self._turns.append([TurnComplete(response=follow_up_response)])

    def enqueue_events(self, events: list[ExecutorEvent]) -> None:
        self._turns.append(events)

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        if not self._turns:
            yield TurnComplete(response="[MockExecutor: no more scripted turns]")
            return
        events = self._turns.pop(0)
        for event in events:
            yield event
