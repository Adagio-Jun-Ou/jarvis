"""OpenCode-style agent executor for Jarvis.

Implements the continuous agent loop pattern from OpenCode:
- Iterates until task completion (not single-turn)
- Streams thinking/reasoning content
- Handles tool calls with auto-execution
- Manages conversation turns with proper finish detection

This integrates GLM-5's thinking capabilities with Jarvis tools.
"""
import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable

from core.helpers import get_response_content, get_tool_call_info, safe_error_message
from core.permissions import PermissionLevel

logger = logging.getLogger(__name__)


class ToolExecutionGuard:
    """Rate limiting, timeout, and validation for tool execution."""

    DEFAULT_TIMEOUT = 120.0
    MAX_INPUT_LENGTH = 50000
    MAX_CONCURRENT_PER_TOOL = 5

    def __init__(self):
        self._tool_timeouts: dict[str, float] = {}
        self._rate_limits: dict[str, deque] = {}
        self._concurrent_counts: dict[str, int] = {}
        self._rate_limit_per_minute: int = 60
        self._lock = threading.Lock()

    def set_tool_timeout(self, tool_name: str, timeout_seconds: float) -> None:
        """Set a per-tool timeout override."""
        self._tool_timeouts[tool_name] = max(1.0, min(3600.0, timeout_seconds))

    def get_timeout(self, tool_name: str) -> float:
        """Get timeout for a tool (default if not set)."""
        return self._tool_timeouts.get(tool_name, self.DEFAULT_TIMEOUT)

    def set_rate_limit(self, calls_per_minute: int) -> None:
        """Set global rate limit (calls per minute per tool)."""
        self._rate_limit_per_minute = max(1, calls_per_minute)

    def check_rate_limit(self, tool_name: str) -> bool:
        """Check if a tool call is within rate limits. Returns True if allowed."""
        now = time.time()
        window = 60.0
        with self._lock:
            if tool_name not in self._rate_limits:
                self._rate_limits[tool_name] = deque()
            q = self._rate_limits[tool_name]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= self._rate_limit_per_minute:
                return False
            q.append(now)
            return True

    def check_concurrency(self, tool_name: str) -> bool:
        """Check if concurrent calls for a tool are within limits."""
        with self._lock:
            current = self._concurrent_counts.get(tool_name, 0)
            return current < self.MAX_CONCURRENT_PER_TOOL

    def acquire_concurrency(self, tool_name: str) -> None:
        """Increment concurrent call counter."""
        with self._lock:
            self._concurrent_counts[tool_name] = self._concurrent_counts.get(tool_name, 0) + 1

    def release_concurrency(self, tool_name: str) -> None:
        """Decrement concurrent call counter."""
        with self._lock:
            current = self._concurrent_counts.get(tool_name, 0)
            if current > 0:
                self._concurrent_counts[tool_name] = current - 1

    def validate_input(self, tool_name: str, tool_input: dict) -> tuple[bool, list[str]]:
        """Validate tool input parameters.
        
        Returns (is_valid, list_of_errors).
        """
        errors = []
        if not isinstance(tool_input, dict):
            errors.append(f"Tool input must be a dict, got {type(tool_input).__name__}")
            return False, errors

        for key, value in tool_input.items():
            if isinstance(value, str) and len(value) > self.MAX_INPUT_LENGTH:
                errors.append(f"Parameter '{key}' exceeds max length ({self.MAX_INPUT_LENGTH} chars)")

        total_size = len(json.dumps(tool_input))
        if total_size > self.MAX_INPUT_LENGTH * 2:
            errors.append(f"Total input size ({total_size} bytes) exceeds limit")

        return len(errors) == 0, errors

    def get_stats(self) -> dict:
        """Get guard statistics."""
        with self._lock:
            return {
                "tool_timeouts": dict(self._tool_timeouts),
                "rate_tracked_tools": list(self._rate_limits.keys()),
                "rate_limit_per_minute": self._rate_limit_per_minute,
                "concurrent_counts": dict(self._concurrent_counts),
            }


_guard: ToolExecutionGuard | None = None
_guard_lock = threading.Lock()


def get_tool_execution_guard() -> ToolExecutionGuard:
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = ToolExecutionGuard()
    return _guard


class FinishReason(Enum):
    """Reasons for finishing a turn."""
    STOP = "stop"                    # Natural end
    TOOL_CALLS = "tool-calls"        # Tool call requested
    MAX_TOKENS = "max-tokens"        # Hit token limit
    ERROR = "error"                  # Error occurred
    UNKNOWN = "unknown"


class ToolStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    PERMISSION_DENIED = "permission-denied"


@dataclass
class ReasoningPart:
    """A reasoning/thinking content part."""
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class TextPart:
    """A text content part."""
    text: str = ""


@dataclass
class ToolPart:
    """A tool call part."""
    tool_name: str
    call_id: str
    input: dict = field(default_factory=dict)
    status: ToolStatus = ToolStatus.PENDING
    output: str = ""
    error: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class AssistantMessage:
    """An assistant message with multiple parts."""
    role: str = "assistant"
    parts: list = field(default_factory=list)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    tool_calls: list[ToolPart] = field(default_factory=list)
    reasoning: ReasoningPart | None = None
    text: str = ""


@dataclass
class AgentStep:
    """A single step in the agent loop."""
    step_number: int
    assistant_message: AssistantMessage
    tool_results: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Result of a complete agent execution."""
    steps: list[AgentStep] = field(default_factory=list)
    total_tool_calls: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    finish_reason: FinishReason = FinishReason.STOP
    metadata: dict = field(default_factory=dict)


class AgentLoop:
    """OpenCode-style continuous agent loop.
    
    Executes LLM calls in a loop, handling tool calls automatically
    until the model signals completion.
    
    Usage:
        loop = AgentLoop(
            llm_client=backend_llm,
            tool_registry=tool_registry,
            permission_manager=permissions,
        )
        
        result = await loop.execute(
            messages=[{"role": "user", "content": "..."}],
            system_prompt="...",
            max_steps=20,
        )
    """
    
    def __init__(
        self,
        llm_client,
        tool_registry: dict[str, Any],
        permission_manager=None,
        max_steps: int = 50,
        show_thinking: bool = True,
        show_tool_calls: bool = True,
        auto_approve_tools: bool = False,
        on_step: Callable[[AgentStep], None] | None = None,
        on_tool_call: Callable[[ToolPart], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        plan_first: bool = False,
        verify_steps: bool = False,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.permissions = permission_manager
        self.max_steps = max_steps
        self.show_thinking = show_thinking
        self.show_tool_calls = show_tool_calls
        self.auto_approve_tools = auto_approve_tools
        self.on_step = on_step
        self.on_tool_call = on_tool_call
        self.on_thinking = on_thinking
        self.plan_first = plan_first
        self.verify_steps = verify_steps
    
    async def execute(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[str] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AgentResult:
        """Execute the agent loop until completion.
        
        Args:
            messages: Conversation history
            system_prompt: System instructions
            tools: Optional list of tool names to enable
            temperature: LLM temperature
            max_tokens: Maximum output tokens
            
        Returns:
            AgentResult with all steps and final state
        """
        start_time = time.time()
        result = AgentResult()
        
        # tools parameter can be:
        # - None: use all available tools  
        # - list[str]: list of tool names
        # - list[dict]: pre-built tool schemas (passed through)
        if tools is None or (isinstance(tools, list) and len(tools) > 0 and isinstance(tools[0], str)):
            available_tools = self._build_tool_schemas_from_names(tools)
        else:
            available_tools = tools
        
        while len(result.steps) < self.max_steps:
            step_number = len(result.steps) + 1
            step_start = time.time()
            
            assistant_message = await self._execute_turn(
                messages=messages,
                system_prompt=system_prompt,
                tools=available_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                show_thinking=self.show_thinking,
            )
            
            tool_results = []
            
            if assistant_message.tool_calls:
                if len(assistant_message.tool_calls) > 1:
                    tool_results = await self._execute_tools_parallel(assistant_message.tool_calls)
                else:
                    tool_result = await self._execute_tool(assistant_message.tool_calls[0])
                    tool_results = [tool_result]
                result.total_tool_calls += len(assistant_message.tool_calls)
            
            step = AgentStep(
                step_number=step_number,
                assistant_message=assistant_message,
                tool_results=tool_results,
                duration_ms=(time.time() - step_start) * 1000,
            )
            result.steps.append(step)
            
            if self.on_step:
                self.on_step(step)
            
            if assistant_message.finish_reason != FinishReason.TOOL_CALLS:
                result.finish_reason = assistant_message.finish_reason
                break
            
            messages = self._append_step_to_messages(messages, step)
            
            # Context compaction between steps: without this, long
            # tool-executing runs grow the message list without bound.
            messages, did_summarize = await self.summarize_if_needed(messages)
            if did_summarize:
                logger.info(
                    f"AgentLoop compacted context to {len(messages)} messages "
                    f"after step {step_number}"
                )
            
            if assistant_message.reasoning:
                result.total_tokens += len(assistant_message.reasoning.text.split()) * 2
            if assistant_message.text:
                result.total_tokens += len(assistant_message.text.split()) * 2
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    async def execute_with_planning(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[str] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        objective: str | None = None,
    ) -> AgentResult:
        """Execute with Plan-Execute-Verify pattern.
        
        First generates a structured plan via AgenticHarness,
        then executes each step through the normal agent loop,
        with verification between steps.
        
        Args:
            messages: Conversation history
            system_prompt: System instructions
            tools: Optional list of tool names
            temperature: LLM temperature
            max_tokens: Maximum output tokens
            objective: Task objective for planning (extracted from last user message if None)
            
        Returns:
            AgentResult with all steps and verification metadata
        """
        if objective is None:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    objective = msg.get("content", "")[:500]
                    break
            if not objective:
                objective = "Complete the requested task"
        
        from core.agentic_harness import AgenticHarness, create_plan_from_objective, StepStatus
        
        harness = AgenticHarness()
        available_tools = list(self.tools.keys()) if self.tools else []
        heuristic_steps = create_plan_from_objective(objective, available_tools)
        plan = harness.create_plan(objective, heuristic_steps, max_iterations=self.max_steps)
        
        start_time = time.time()
        result = AgentResult()
        
        tool_schemas = self._build_tool_schemas_from_names(tools)
        
        for step_data in plan.steps:
            if len(result.steps) >= self.max_steps:
                break
            
            step_number = len(result.steps) + 1
            step_start = time.time()
            
            deps_met = all(
                dep_id in {s.id for s in plan.steps if s.status.value in ("success", "verified", "skipped")}
                for dep_id in step_data.dependencies
            )
            if not deps_met:
                step_data.status = StepStatus.SKIPPED
                continue
            
            user_msg = {
                "role": "user",
                "content": f"Execute step: {step_data.description}\nTool: {step_data.tool_name}\nParams: {step_data.tool_params}",
            }
            step_messages = list(messages) + [user_msg]
            
            assistant_message = await self._execute_turn(
                messages=step_messages,
                system_prompt=system_prompt,
                tools=tool_schemas,
                temperature=temperature,
                max_tokens=max_tokens,
                show_thinking=self.show_thinking,
            )
            
            tool_results = []
            if assistant_message.tool_calls:
                for tool_part in assistant_message.tool_calls:
                    tool_result = await self._execute_tool(tool_part)
                    tool_results.append(tool_result)
                    result.total_tool_calls += 1
            elif step_data.tool_name and step_data.tool_name in self.tools:
                tool_result = await self._execute_tool_raw(step_data.tool_name, step_data.tool_params)
                tool_results.append(tool_result)
                result.total_tool_calls += 1
            
            if tool_results:
                success = any(r.get("success", False) for r in tool_results)
                if success:
                    step_data.status = StepStatus.SUCCESS
                else:
                    step_data.status = StepStatus.FAILED
            elif assistant_message.text:
                step_data.status = StepStatus.SUCCESS
            
            if self.verify_steps and step_data.status.value == "success":
                verified = self._verify_step_result(step_data, tool_results, assistant_message)
                if verified:
                    step_data.status = StepStatus.VERIFIED
                    step_data.verification_result = "passed"
                else:
                    step_data.verification_result = "failed"
            
            step = AgentStep(
                step_number=step_number,
                assistant_message=assistant_message,
                tool_results=tool_results,
                duration_ms=(time.time() - step_start) * 1000,
            )
            result.steps.append(step)
            
            if self.on_step:
                self.on_step(step)
            
            if assistant_message.finish_reason != FinishReason.TOOL_CALLS:
                break
            
            messages = self._append_step_to_messages(messages, step)
        
        result.finish_reason = FinishReason.STOP
        result.duration_ms = (time.time() - start_time) * 1000
        
        completed = sum(1 for s in plan.steps if s.status.value in ("success", "verified"))
        result.metadata = {
            "plan_id": plan.id,
            "total_planned_steps": len(plan.steps),
            "completed_steps": completed,
            "plan_progress": plan.progress,
            "plan_complete": plan.is_complete,
        }
        
        return result
    
    def _verify_step_result(
        self,
        step_data: Any,
        tool_results: list[dict],
        assistant_message: AssistantMessage,
    ) -> bool:
        """Verify a step's result is valid using heuristic checks."""
        if step_data.expected_outcome and assistant_message.text:
            expected_lower = step_data.expected_outcome.lower()
            response_lower = assistant_message.text.lower()
            if any(word in response_lower for word in expected_lower.split() if len(word) > 3):
                return True
        
        for tr in tool_results:
            if not tr.get("success", False):
                return False
            output = tr.get("output", "")
            if isinstance(output, dict) and output.get("error"):
                return False
            if isinstance(output, str) and not output.strip():
                return False
        
        return bool(tool_results) or bool(assistant_message.text)

    async def _verify_step_with_llm(
        self,
        step_data: Any,
        tool_results: list[dict],
        assistant_message: AssistantMessage,
        expected_outcome: str = "",
    ) -> bool:
        """Verify step result using LLM assessment.

        Uses the backend LLM to check if the step output satisfies
        the expected outcome. Falls back to heuristic verification.
        """
        if not hasattr(self.llm, 'chat'):
            return self._verify_step_result(step_data, tool_results, assistant_message)

        output_text = assistant_message.text or ""
        tool_output = ""
        for tr in tool_results:
            if tr.get("success"):
                raw = tr.get("output", "")
                tool_output += (str(raw) if isinstance(raw, str) else json.dumps(raw)[:500]) + "\n"

        if not output_text and not tool_output:
            return False

        prompt = f"""Verify if this step's result satisfies the expected outcome.

Step: {step_data.description if hasattr(step_data, 'description') else 'unknown'}
Expected: {expected_outcome or (step_data.expected_outcome if hasattr(step_data, 'expected_outcome') else 'task completed')}

Result:
{output_text[:500]}
{tool_output[:300]}

Answer ONLY: PASS or FAIL"""

        try:
            response = await asyncio.to_thread(
                self.llm.chat,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response if isinstance(response, str) else str(response)
            return "PASS" in text.upper()
        except Exception:
            return self._verify_step_result(step_data, tool_results, assistant_message)
    
    async def execute_stream(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[str] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict]:
        """Execute the agent loop with streaming output.
        
        Yields events for each piece of content:
        - {"type": "thinking", "text": "..."} 
        - {"type": "text", "text": "..."}
        - {"type": "tool_call", "tool": "...", "input": {...}}
        - {"type": "tool_result", "tool": "...", "output": "..."}
        - {"type": "step_complete", "step": N}
        - {"type": "complete", "finish_reason": "..."}
        """
        # tools parameter can be:
        # - None: use all available tools
        # - list[str]: list of tool names
        # - list[dict]: pre-built tool schemas (passed through)
        if tools is None or (isinstance(tools, list) and len(tools) > 0 and isinstance(tools[0], str)):
            available_tools = self._build_tool_schemas_from_names(tools)
        else:
            available_tools = tools
        step_count = 0
        
        while step_count < self.max_steps:
            step_count += 1
            tool_calls = []
            finish_reason = FinishReason.UNKNOWN
            
            async for event in self._stream_turn(
                messages=messages,
                system_prompt=system_prompt,
                tools=available_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if event.get("type") == "tool_call":
                    tool_calls.append(event)
                elif event.get("type") == "finish_reason":
                    finish_reason = FinishReason(event.get("reason", "unknown"))
                
                yield event
            
            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                tool_input = tc.get("input", {})
                
                tool_result = await self._execute_tool_raw(tool_name, tool_input)
                yield {"type": "tool_result", "tool": tool_name, **tool_result}
                
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tc.get("call_id", f"call_{tool_name}"),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input),
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("call_id", f"call_{tool_name}"),
                    "content": tool_result.get("output", ""),
                })
            
            yield {"type": "step_complete", "step": step_count}
            
            if finish_reason != FinishReason.TOOL_CALLS:
                yield {"type": "complete", "finish_reason": finish_reason.value, "steps": step_count}
                break
            
            messages = self._prepare_messages_for_next_turn(messages)
        
        # If we exited the loop without completing (hit max_steps)
        if step_count >= self.max_steps and finish_reason == FinishReason.TOOL_CALLS:
            yield {"type": "complete", "finish_reason": "max_iterations", "steps": step_count}
    
    async def _execute_turn(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        temperature: float,
        max_tokens: int,
        show_thinking: bool,
    ) -> AssistantMessage:
        """Execute a single turn of the agent loop."""
        assistant_message = AssistantMessage()
        
        try:
            from core.llm import subsystem_context
            if hasattr(self.llm, 'chat_stream_with_tools'):
                async with subsystem_context("AgentLoop"):
                    stream = self.llm.chat_stream_with_tools(
                        user_input="",
                        system_prompt=system_prompt,
                        conversation_history=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                
                thinking_parts = []
                response_parts = []
                tool_calls_list = []
                
                async for chunk in stream:
                    if chunk.get("type") == "thinking" or chunk.get("type") == "reasoning":
                        text = chunk.get("text", "")
                        thinking_parts.append(text)
                        if self.on_thinking:
                            self.on_thinking(text)
                    
                    elif chunk.get("type") == "text" or chunk.get("type") == "content":
                        text = chunk.get("text", "") or chunk.get("content", "")
                        response_parts.append(text)
                    
                    elif chunk.get("type") == "tool_call":
                        tool_calls_list.append(chunk)
                    
                    elif chunk.get("type") == "finish":
                        reason = chunk.get("reason", "stop")
                        try:
                            assistant_message.finish_reason = FinishReason(reason)
                        except ValueError:
                            assistant_message.finish_reason = FinishReason.STOP
                
                thinking_text = "".join(thinking_parts)
                response_text = "".join(response_parts)
                
                if thinking_text:
                    assistant_message.reasoning = ReasoningPart(
                        text=thinking_text,
                        start_time=time.time(),
                        end_time=time.time(),
                    )
                    if show_thinking:
                        assistant_message.parts.append(assistant_message.reasoning)
                
                if response_text:
                    assistant_message.text = response_text
                    assistant_message.parts.append(TextPart(text=response_text))
                
                for tc in tool_calls_list:
                    tool_part = ToolPart(
                        tool_name=tc.get("name", tc.get("tool", "")),
                        call_id=tc.get("id", tc.get("call_id", "")),
                        input=tc.get("arguments", tc.get("input", {})),
                        status=ToolStatus.PENDING,
                    )
                    assistant_message.tool_calls.append(tool_part)
                    assistant_message.parts.append(tool_part)
                
                if assistant_message.tool_calls:
                    assistant_message.finish_reason = FinishReason.TOOL_CALLS
                
            else:
                response = await self.llm.chat_with_tools(
                    user_input="",
                    system_prompt=system_prompt,
                    conversation_history=messages,
                    tools=tools,
                )
                
                assistant_message.text = get_response_content(response)
                assistant_message.parts.append(TextPart(text=assistant_message.text))
                
                if "tool_calls" in response.get("message", {}):
                    for tc in response["message"]["tool_calls"]:
                        try:
                            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        _, name = get_tool_call_info(tc)
                        tool_part = ToolPart(
                            tool_name=name or tc.get("function", {}).get("name", ""),
                            call_id=tc.get("id", ""),
                            input=args,
                            status=ToolStatus.PENDING,
                        )
                        assistant_message.tool_calls.append(tool_part)
                        assistant_message.parts.append(tool_part)
                    
                    assistant_message.finish_reason = FinishReason.TOOL_CALLS
                else:
                    assistant_message.finish_reason = FinishReason.STOP
        
        except Exception as e:
            logger.error(f"Agent turn error: {e}")
            assistant_message.finish_reason = FinishReason.ERROR
            assistant_message.parts.append(TextPart(text=f"[Error: {e}]"))
        
        return assistant_message
    
    async def _stream_turn(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[dict]:
        """Stream a single turn of the agent loop."""
        try:
            if hasattr(self.llm, 'chat_stream_with_tools'):
                stream = self.llm.chat_stream_with_tools(
                    user_input="",
                    system_prompt=system_prompt,
                    conversation_history=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                async for chunk in stream:
                    chunk_type = chunk.get("type", "")
                    
                    if chunk_type in ("thinking", "reasoning"):
                        yield {"type": "thinking", "text": chunk.get("text", "")}
                    
                    elif chunk_type in ("text", "content"):
                        yield {"type": "text", "text": chunk.get("text", "") or chunk.get("content", "")}
                    
                    elif chunk_type == "tool_call":
                        yield {
                            "type": "tool_call",
                            "tool": chunk.get("name", chunk.get("tool", "")),
                            "call_id": chunk.get("id", chunk.get("call_id", "")),
                            "input": chunk.get("arguments", chunk.get("input", {})),
                        }
                    
                    elif chunk_type == "finish":
                        yield {"type": "finish_reason", "reason": chunk.get("reason", "stop")}
            
            else:
                response = await self.llm.chat_with_tools(
                    user_input="",
                    system_prompt=system_prompt,
                    conversation_history=messages,
                    tools=tools,
                )
                
                content = get_response_content(response)
                if content:
                    yield {"type": "text", "text": content}
                
                tool_calls = response.get("message", {}).get("tool_calls", [])
                for tc in tool_calls:
                    name, args = get_tool_call_info(tc)
                    try:
                        args_dict = json.loads(tc.get("function", {}).get("arguments", "{}")) if isinstance(args, str) else (args or {})
                    except json.JSONDecodeError:
                        args_dict = {}
                    yield {
                        "type": "tool_call",
                        "tool": name or tc.get("function", {}).get("name", ""),
                        "call_id": tc.get("id", ""),
                        "input": args_dict,
                    }
                
                yield {"type": "finish_reason", "reason": "tool-calls" if tool_calls else "stop"}
        
        except Exception as e:
            yield {"type": "error", "error": safe_error_message(e)}
            yield {"type": "finish_reason", "reason": "error"}
    
    async def _execute_tool(self, tool_part: ToolPart) -> dict:
        """Execute a tool and return the result."""
        tool_part.status = ToolStatus.RUNNING
        tool_part.start_time = time.time()
        
        if self.on_tool_call:
            self.on_tool_call(tool_part)
        
        result = await self._execute_tool_raw(tool_part.tool_name, tool_part.input)
        
        tool_part.end_time = time.time()
        
        if result.get("success", False):
            tool_part.status = ToolStatus.COMPLETED
            tool_part.output = result.get("output", "")
        else:
            tool_part.status = ToolStatus.ERROR
            tool_part.error = result.get("error", "Unknown error")
        
        return result
    
    async def _execute_tools_parallel(self, tool_calls: list[ToolPart], max_concurrent: int = 3) -> list[dict]:
        """Execute multiple tool calls in parallel.

        Limits concurrency to avoid resource exhaustion.
        Read-only tools (search, read, list) are safe to parallelize.
        Write tools (write, delete, execute) run sequentially for safety.

        Args:
            tool_calls: List of ToolPart objects to execute
            max_concurrent: Maximum concurrent tool executions

        Returns:
            List of tool result dicts
        """
        WRITE_TOOLS = {
            'write_file', 'write', 'edit', 'create', 'delete', 'remove',
            'terminal_execute', 'shell', 'execute', 'install_package',
            'self_edit', 'modify_settings',
        }

        read_only = []
        write_calls = []
        for i, tc in enumerate(tool_calls):
            if tc.tool_name in WRITE_TOOLS:
                write_calls.append((i, tc))
            else:
                read_only.append((i, tc))

        results = [None] * len(tool_calls)

        if read_only:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def _run_with_sem(orig_idx, tool_part):
                async with semaphore:
                    return orig_idx, await self._execute_tool(tool_part)

            tasks = [_run_with_sem(idx, tc) for idx, tc in read_only]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    logger.warning(f"Parallel tool execution error: {safe_error_message(item)}")
                else:
                    idx, result = item
                    results[idx] = result

        for idx, tc in write_calls:
            results[idx] = await self._execute_tool(tc)

        return [r for r in results if r is not None]
    
    async def _execute_tool_raw(self, tool_name: str, tool_input: dict | str) -> dict:
        """Execute a tool by name with given input.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool arguments (dict or JSON string)
            
        Returns:
            Dict with success status and output/error
        """
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        # Ensure tool_input is a dict
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed JSON in tool '{tool_name}' args: {e}")
                return {"success": False, "error": f"Invalid tool arguments: JSON parse error", "tool": tool_name}
        
        if not isinstance(tool_input, dict):
            logger.warning(f"Non-dict tool args for '{tool_name}': {type(tool_input)}")
            return {"success": False, "error": f"Invalid tool arguments: expected dict, got {type(tool_input).__name__}", "tool": tool_name}
        
        # Guard: validate input size
        guard = get_tool_execution_guard()
        valid, errors = guard.validate_input(tool_name, tool_input)
        if not valid:
            return {"success": False, "error": f"Input validation failed: {'; '.join(errors)}"}
        
        # Guard: rate limit check
        if not guard.check_rate_limit(tool_name):
            return {"success": False, "error": f"Rate limit exceeded for tool '{tool_name}', please retry later"}
        
        # Guard: concurrency check
        if not guard.check_concurrency(tool_name):
            return {"success": False, "error": f"Too many concurrent calls for tool '{tool_name}'"}
        
        guard.acquire_concurrency(tool_name)
        
        tool = self.tools[tool_name]
        timeout = guard.get_timeout(tool_name)
        
        # SECURITY: Shell tools require permission during autonomous operations
        # Allow during user-initiated operations for normal Jarvis functionality
        shell_tools = {'terminal_execute', 'terminal_mode', 'install_package', 'shell', 'run_shell', 'execute_command'}
        
        if self.permissions and getattr(tool, 'requires_permission', False):
            # Check if this is an autonomous operation (daydream, free_time, background)
            # These operations should ALWAYS require explicit permission for shell access
            is_autonomous = getattr(self, '_autonomous_mode', False)
            
            # Permission key must match what JarvisAgent.execute_tool and the
            # /api/permissions endpoint use. This used to ask for
            # f"tool:{tool_name}", which matches neither the exact-key store nor
            # _get_config_permission's prefix map ("shell"/"file_*"/"web"), so
            # check_permission could only ever return None: both grants and
            # denies were invisible and the `== deny` branch below was dead code.
            action_key = f"{tool_name}:{tool_input.get('action', '')}" if tool_input else tool_name
            permission = self.permissions.check_permission(action_key)
            if permission is None:
                permission = self.permissions.check_permission(tool_name)

            # Shell tools during autonomous operation: deny unless explicitly
            # granted. There is no user present to answer a prompt, so an
            # ungranted permission (None) must mean "no", not "proceed".
            if tool_name in shell_tools and is_autonomous:
                if permission != PermissionLevel.ALLOW_ALWAYS and permission != PermissionLevel.ALLOW_SESSION:
                    logger.warning(
                        f"Shell tool '{tool_name}' blocked: autonomous operation requires "
                        f"an explicit prior grant (permission={permission})"
                    )
                    guard.release_concurrency(tool_name)
                    return {"success": False, "error": "Permission denied - shell access requires explicit approval during autonomous operations"}
            elif not self.auto_approve_tools:
                # Interactive path: an explicit deny blocks. An ungranted
                # permission (None) is still allowed here so the web agent-loop
                # keeps working, but it is surfaced — see NEXT_STEPS for the
                # key-scheme unification this is waiting on.
                if permission == PermissionLevel.DENY:
                    guard.release_concurrency(tool_name)
                    return {"success": False, "error": "Permission denied"}
                if permission is None:
                    logger.info(
                        f"Tool '{tool_name}' requires permission but none is recorded; "
                        f"allowing on the interactive path"
                    )

        try:
            if hasattr(tool, 'execute'):
                # Try safe_execute first (adds timeout and structured error handling)
                if hasattr(tool, 'safe_execute'):
                    try:
                        tool_result = await asyncio.wait_for(
                            tool.safe_execute(**tool_input),
                            timeout=timeout,
                        )
                        if isinstance(tool_result, dict) and 'success' in tool_result:
                            guard.release_concurrency(tool_name)
                            return tool_result
                        from tools.base import ToolResult
                        if isinstance(tool_result, ToolResult):
                            guard.release_concurrency(tool_name)
                            return tool_result.to_dict()
                    except asyncio.TimeoutError:
                        guard.release_concurrency(tool_name)
                        return {"success": False, "error": f"Tool '{tool_name}' timed out after {timeout}s"}
                    except Exception as safe_exec_err:
                        logger.debug(f"safe_execute failed for '{tool_name}': {safe_exec_err}")

                try:
                    if asyncio.iscoroutinefunction(tool.execute):
                        result = await asyncio.wait_for(
                            tool.execute(**tool_input),
                            timeout=timeout,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(tool.execute, **tool_input),
                            timeout=timeout,
                        )
                except asyncio.TimeoutError:
                    guard.release_concurrency(tool_name)
                    return {"success": False, "error": f"Tool '{tool_name}' timed out after {timeout}s"}
                
                if isinstance(result, dict):
                    guard.release_concurrency(tool_name)
                    return {"success": True, "output": result}
                elif isinstance(result, str):
                    guard.release_concurrency(tool_name)
                    return {"success": True, "output": result}
                else:
                    guard.release_concurrency(tool_name)
                    return {"success": True, "output": str(result)}
            
            guard.release_concurrency(tool_name)
            return {"success": False, "error": f"Tool {tool_name} has no execute method"}
        
        except Exception as e:
            guard.release_concurrency(tool_name)
            logger.error(f"Tool execution error [{tool_name}]: {e}")
            return {"success": False, "error": safe_error_message(e)}

    def _build_tool_schemas_from_names(self, tool_names: list[str] | None) -> list[dict]:
        """Build tool schemas for LLM from tool names.
        
        Args:
            tool_names: List of tool names, or None for all tools
            
        Returns:
            List of tool schema dicts for LLM
        """
        if tool_names is None:
            tools_to_use = list(self.tools.keys())
        else:
            tools_to_use = tool_names
        
        schemas = []
        for name in tools_to_use:
            if name not in self.tools:
                continue
            
            tool = self.tools[name]
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'parameters', {"type": "object", "properties": {}}),
                }
            }
            schemas.append(schema)
        
        return schemas

    def _append_step_to_messages(self, messages: list[dict], step: AgentStep) -> list[dict]:
        """Add a completed step to the message history.

        Order matters: the assistant message that declares `tool_calls` must come
        BEFORE the tool messages answering it. This used to append each tool
        result inside the loop and the assistant message only afterwards, so the
        history handed back to the model on the next turn had every tool result
        preceding the call it responded to — tool_call_ids referring forward to a
        message that had not appeared yet.

        Every declared tool_call also gets exactly one tool message. Previously a
        call with empty/failed output produced none, leaving a dangling
        tool_call_id that strict providers reject.
        """
        new_messages = list(messages)

        tool_calls = []
        tool_messages = []

        for tc in step.assistant_message.tool_calls:
            tool_calls.append({
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.tool_name,
                    "arguments": json.dumps(tc.input),
                }
            })

            content = tc.output
            if not content:
                error = getattr(tc, "error", None)
                content = f"Error: {error}" if error else ""
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": content,
            })

        assistant_msg = {"role": "assistant", "content": step.assistant_message.text}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        new_messages.append(assistant_msg)
        new_messages.extend(tool_messages)

        return new_messages
    
    def _prepare_messages_for_next_turn(self, messages: list[dict]) -> list[dict]:
        """Clean up messages for the next turn."""
        return messages

    async def summarize_if_needed(
        self,
        messages: list[dict],
        max_messages: int = 50,
        keep_recent: int = 10,
    ) -> tuple[list[dict], bool]:
        """Summarize conversation if it exceeds max_messages.
        
        OpenCode pattern: summarize older messages to reduce context while preserving
        important context about what was done.
        
        Args:
            messages: Current message history
            max_messages: Maximum messages before summarization
            keep_recent: Number of recent messages to keep intact
            
        Returns:
            (summarized_messages, was_summarized)
        """
        if len(messages) <= max_messages:
            return messages, False
        
        recent_messages = messages[-keep_recent:]
        old_messages = messages[:-keep_recent]
        
        summary_text = "Previous conversation summary:\n"
        for msg in old_messages[-20:]:  # Summarize last 20 of older messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            if role == "user":
                summary_text += f"User asked: {content}\n"
            elif role == "assistant":
                summary_text += f"Assistant: {content}\n"
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                summary_text += f"Tool {tool_name}: {content[:100]}\n"
        
        summary_msg = {
            "role": "system",
            "content": summary_text,
        }
        
        return [summary_msg] + recent_messages, True


def create_agent_loop(
    llm_client,
    tools: dict[str, Any],
    permissions=None,
    **kwargs,
) -> AgentLoop:
    """Create an OpenCode-style agent loop.
    
    Args:
        llm_client: LLM client with streaming support
        tools: Dictionary of tool name -> tool instance
        permissions: Optional permission manager
        **kwargs: Additional AgentLoop options
        
    Returns:
        Configured AgentLoop instance
    """
    return AgentLoop(
        llm_client=llm_client,
        tool_registry=tools,
        permission_manager=permissions,
        **kwargs,
    )


def create_agent_loop_from_agent(agent) -> AgentLoop:
    """Factory function to create an AgentLoop from a JarvisAgent.
    
    This integrates the OpenCode-style agent loop with Jarvis's existing
    agent infrastructure, including:
    - Tool registry (all registered tools)
    - Permission manager
    - Backend LLM for reliable tool calling
    - Auto-approve setting from config
    
    Args:
        agent: JarvisAgent instance
        
    Returns:
        Configured AgentLoop ready for execution
    """
    return AgentLoop(
        llm_client=agent.backend_llm if hasattr(agent, 'backend_llm') and agent.backend_llm else agent.llm,
        tool_registry=agent.tools,
        permission_manager=agent.permissions,
        max_steps=getattr(agent.config.agent, 'max_agentic_iterations', 10),
        show_thinking=getattr(agent.config.agent, 'show_thinking', True),
        show_tool_calls=getattr(agent.config.agent, 'show_tool_calls', True),
        auto_approve_tools=getattr(agent.config.agent, 'auto_approve_tools', False),
        plan_first=getattr(agent.config.agent, 'plan_first', False),
        verify_steps=getattr(agent.config.agent, 'verify_steps', False),
        on_step=lambda step: logger.debug(f"[AgentLoop] Step {step.step_number}: {len(step.tool_results)} tool results"),
        on_tool_call=lambda tc: logger.info(f"[AgentLoop] Tool call: {tc.tool_name}"),
    )
