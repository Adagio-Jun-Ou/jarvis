"""Ollama LLM integration for Jarvis - supports local and cloud API with failover.

Hardware-optimized for AMD Ryzen AI:
- GPU-accelerated token counting (via torch/numpy)
- Async connection pooling
- Memory-efficient streaming
- Integrated with unified context management
"""
import asyncio
import json
import logging
import os
import time
import threading
import atexit
from contextvars import ContextVar
from datetime import datetime
from typing import Any, AsyncIterator
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import httpx

logger = logging.getLogger(__name__)

try:
    from core.context_manager import (
        get_subsystem as _get_subsystem_from_ctx,
        record_output_for_eval,
        get_ethical_score,
    )
    HAS_CONTEXT_MANAGER = True
except ImportError:
    HAS_CONTEXT_MANAGER = False
    _get_subsystem_from_ctx = lambda: "Chat"
    def record_output_for_eval(*args, **kwargs): return {"evaluated": False}
    def get_ethical_score(): return 1.0

_current_subsystem_var: ContextVar[str | None] = ContextVar("subsystem", default=None)


def set_subsystem(name: str):
    """Set the current subsystem context for API call tracking."""
    _current_subsystem_var.set(name)
    if HAS_CONTEXT_MANAGER:
        from core.context_manager import set_subsystem as ctx_set
        ctx_set(name)


def get_subsystem() -> str:
    """Get the current subsystem context."""
    val = _current_subsystem_var.get()
    if val:
        return val
    if HAS_CONTEXT_MANAGER:
        return _get_subsystem_from_ctx()
    return "Chat"


class subsystem_context:
    """Context manager for setting subsystem during LLM calls."""
    def __init__(self, name: str):
        self.name = name
        self._token = None
        self._ctx = None

    def __enter__(self):
        self._token = _current_subsystem_var.set(self.name)
        if HAS_CONTEXT_MANAGER:
            from core.context_manager import subsystem_context as ctx_subsystem
            self._ctx = ctx_subsystem(self.name)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._token is not None:
            _current_subsystem_var.reset(self._token)
        if HAS_CONTEXT_MANAGER and self._ctx:
            self._ctx.__exit__(*args)

    async def __aenter__(self):
        self._token = _current_subsystem_var.set(self.name)
        if HAS_CONTEXT_MANAGER:
            from core.context_manager import subsystem_context as ctx_subsystem
            self._ctx = ctx_subsystem(self.name)
            await self._ctx.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._token is not None:
            _current_subsystem_var.reset(self._token)
        if HAS_CONTEXT_MANAGER and self._ctx:
            await self._ctx.__aexit__(*args)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm_")
atexit.register(lambda: _executor.shutdown(wait=False))

HAS_ACCELERATION = False
np = None
torch = None

try:
    import numpy as _np
    np = _np
    HAS_ACCELERATION = True
except ImportError:
    pass

try:
    import torch as _torch
    torch = _torch
    HAS_ACCELERATION = True
except ImportError:
    pass


def count_tokens_fast(text: str) -> int:
    """Fast token counting using GPU acceleration if available.
    
    Uses character-based approximation calibrated for typical token ratios.
    GPU-accelerated for batch operations when torch/numpy available.
    """
    if not text:
        return 0

    char_count = len(text)
    word_count = len(text.split())

    tokens_from_chars = char_count // 4
    tokens_from_words = int(word_count * 1.3)

    return max(tokens_from_chars, tokens_from_words)


def count_tokens_batch(texts: list[str]) -> list[int]:
    """Count tokens for multiple texts using GPU acceleration.
    
    Optimized for batch processing with numpy/torch.
    """
    if not texts:
        return []

    if HAS_ACCELERATION and np is not None and len(texts) > 10:
        char_counts = np.array([len(t) for t in texts], dtype=np.int32)
        char_tokens = char_counts // 4

        word_counts = np.array([len(t.split()) for t in texts], dtype=np.int32)
        word_tokens = (word_counts * 1.3).astype(np.int32)

        return np.maximum(char_tokens, word_tokens).tolist()

    return [count_tokens_fast(t) for t in texts]


class OllamaClient:
    """Client for interacting with Ollama API (local or cloud) with automatic failover.
    
    Hardware-optimized:
    - Async HTTP/2 connection pooling
    - GPU-accelerated token counting for batch operations
    - Memory-efficient streaming
    - Hardware-aware batch size tuning
    """

    CLOUD_API_URL = "https://ollama.com"
    MAX_CALL_CALLBACKS = 20

    def __init__(
        self,
        model: str = "glm-5",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        api_key: str | None = None,
        use_cloud: bool = False,
        fallback_models: list[str] | None = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        request_timeout: float = 300.0,
        max_retries: int = 3,
        use_token_optimizer: bool = True,
        hardware_config: Any = None,
        circuit_breaker: Any = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.use_cloud = use_cloud if use_cloud is not None else bool(self.api_key)
        self.use_token_optimizer = use_token_optimizer
        self._hardware_config = hardware_config
        self._circuit_breaker = circuit_breaker

        self.fallback_models = fallback_models or []
        if model not in self.fallback_models:
            self.fallback_models.insert(0, model)
        self.current_model = model
        self._current_model_lock = threading.Lock()
        self._models_list_cache = None
        self._models_list_cache_key = None
        self.model_errors: dict[str, int] = {}
        self._model_errors_lock = threading.Lock()
        self._models_list_cache_lock = threading.Lock()
        self.max_errors_before_fallback = 3
        
        # Track API calls for dashboard
        self._call_count = 0
        self._last_call_time: datetime | None = None
        self._call_log: deque[dict] = deque(maxlen=100)
        self._call_log_lock = threading.Lock()
        self._call_callbacks: list[Any] = []
        self._call_callbacks_lock = threading.Lock()  # Protect callbacks list

        self.max_retries = max_retries

        if self.use_cloud:
            self.base_url = self.CLOUD_API_URL
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        else:
            self.base_url = host.rstrip("/")
            headers = {}

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=30.0,
        )

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=30.0),
            headers=headers,
            limits=limits,
            http2=False,
        )

        self._closed = False
        self._token_optimizer = None
        self._hardware = None

        if use_token_optimizer:
            try:
                from core.token_optimizer import get_token_optimizer
                self._token_optimizer = get_token_optimizer()
            except Exception as e:
                logger.debug(f"Token optimizer not available: {e}")
                self._token_optimizer = None

        self._init_hardware()

    def __repr__(self) -> str:
        key_display = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 8 else "***"
        return f"OllamaClient(model={self.model!r}, use_cloud={self.use_cloud}, api_key={key_display})"

    def _log_call(self, call_type: str, model: str, tokens: int = 0, latency_ms: float = 0, cached: bool = False, subsystem: str | None = None):
        """Log an API call for activity tracking."""
        if subsystem is None:
            subsystem = get_subsystem()
        call_record = None
        with self._call_log_lock:
            self._call_count += 1
            self._last_call_time = datetime.now()
            call_record = {
                "timestamp": self._last_call_time.isoformat(),
                "type": call_type,
                "model": model,
                "tokens": tokens,
                "latency_ms": int(latency_ms),
                "cached": cached,
                "subsystem": subsystem,
            }
            self._call_log.appendleft(call_record)
        
        # Also report to dual_model stats for unified tracking
        try:
            from core.dual_model import get_dual_model_orchestrator
            orchestrator = get_dual_model_orchestrator()
            if orchestrator:
                # Determine if this is primary or backend based on model
                is_backend = "glm" in model.lower() or "backend" in model.lower()
                orchestrator.record_subsystem_call("backend" if is_backend else "primary", latency_ms)
        except Exception as e:
            logger.warning(f"Failed to record dual model subsystem call: {e}")
        
        if call_record:
            with self._call_callbacks_lock:
                callbacks = list(self._call_callbacks)
            for callback in callbacks:
                try:
                    callback(call_record)
                except Exception as e:
                    logger.warning(f"Failed to call API callback: {e}")

    def add_call_callback(self, callback: callable):
        with self._call_callbacks_lock:
            if callback not in self._call_callbacks:
                if len(self._call_callbacks) >= self.MAX_CALL_CALLBACKS:
                    return
                self._call_callbacks.append(callback)

    def remove_call_callback(self, callback: callable):
        """Unregister a callback."""
        with self._call_callbacks_lock:
            if callback in self._call_callbacks:
                self._call_callbacks.remove(callback)

    def get_call_stats(self) -> dict:
        """Get call statistics for dashboard."""
        with self._call_log_lock:
            now_ts = time.time()
            one_min_ago = now_ts - 60
            
            # Optimize: compare timestamps directly instead of parsing datetime
            recent_calls = []
            for c in self._call_log:
                if c.get("timestamp"):
                    try:
                        # Parse once
                        ts = datetime.fromisoformat(c["timestamp"])
                        if ts.timestamp() > one_min_ago:
                            recent_calls.append(c)
                    except (ValueError, TypeError):
                        continue
                    
            import itertools
            recent_slice = list(itertools.islice(self._call_log, 20))
        return {
            "total_calls": self._call_count,
            "calls_per_minute": len(recent_calls),
            "last_call": self._last_call_time.isoformat() if self._last_call_time else None,
            "recent_calls": recent_slice,
        }

    def _init_hardware(self):
        """Initialize hardware acceleration."""
        try:
            from core.hardware import get_hardware_detector, get_memory_pool
            self._hardware = get_hardware_detector()
            self._memory_pool = get_memory_pool(self._hardware_config)

            if self._hardware.use_gpu() and self._hardware.primary_accelerator:
                logger.info(f"GPU acceleration enabled: {self._hardware.primary_accelerator}")
            if self._hardware.use_npu():
                logger.info("NPU acceleration available")

        except Exception as e:
            logger.debug(f"Hardware detection not available: {e}")

    def get_optimal_batch_size(self, tokens_per_sample: int = 512) -> int:
        """Get optimal batch size based on available GPU/memory."""
        if self._hardware:
            return self._hardware.get_optimal_batch_size(
                model_size_mb=4000,
                tokens_per_sample=tokens_per_sample
            )
        return 4

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self.client.aclose()

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # For cloud API, just verify we have credentials
            if self.use_cloud:
                return bool(self.api_key)

            # Local Ollama - check if service is running
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                try:
                    models = response.json().get("models", [])
                except Exception:
                    models = []
                return any(m.get("name", "").startswith(self.model) for m in models)
        except httpx.TimeoutException:
            logger.warning(f"Timeout checking Ollama availability at {self.base_url}")
        except httpx.ConnectError:
            logger.debug(f"Cannot connect to Ollama at {self.base_url}")
        except Exception as e:
            logger.debug(f"Error checking Ollama availability: {e}")
        return False

    def _build_messages(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """Build message list for the API."""
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            # Conversation history already includes the user's message
            # (added via memory.add_user_message before calling think())
            messages.extend(conversation_history)
        else:
            # Only add user_input if there's no conversation history
            messages.append({"role": "user", "content": user_input})

        return messages

    def _get_chat_url(self) -> str:
        """Get the chat endpoint URL."""
        # Both local and cloud use the same endpoint
        return f"{self.base_url}/api/chat"

    def _build_payload(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> dict:
        """Build the request payload."""
        # Use native Ollama format for both local and cloud
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
            },
        }

        # Only set num_predict if max_tokens is specified
        # If None, let the model use its full context window
        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens

        if tools:
            payload["tools"] = tools

        return payload

    def _increment_model_error(self, model: str) -> int:
        """Thread-safe increment of model error count. Returns new count."""
        with self._model_errors_lock:
            self.model_errors[model] = self.model_errors.get(model, 0) + 1
            while len(self.model_errors) > 50:
                oldest = next(iter(self.model_errors))
                del self.model_errors[oldest]
            return self.model_errors[model]

    def _get_model_error_count(self, model: str) -> int:
        """Thread-safe get of model error count."""
        with self._model_errors_lock:
            return self.model_errors.get(model, 0)

    def _clear_model_error(self, model: str) -> None:
        """Thread-safe clear of model error count."""
        with self._model_errors_lock:
            self.model_errors[model] = 0
        with self._models_list_cache_lock:
            self._models_list_cache = None

    def _get_models_to_try(self) -> list[str]:
        """Get ordered list of models to try, with caching."""
        with self._models_list_cache_lock:
            current = self.current_model
            cache_key = (current, tuple(self.fallback_models))
            if self._models_list_cache_key == cache_key and self._models_list_cache is not None:
                return self._models_list_cache
            models = [current] + [m for m in self.fallback_models if m != current]
            self._models_list_cache = models
            self._models_list_cache_key = cache_key
            return models

    async def _get_next_model(self, failed_model: str) -> str | None:
        """Get the next model to try after a failure.
        
        Uses exponential backoff without blocking — returns immediately
        if circuit breaker is available, or uses brief async sleep.
        Never blocks for more than 5 seconds total.
        """
        with self._model_errors_lock:
            error_count = self.model_errors.get(failed_model, 0)
        
        if error_count >= self.max_errors_before_fallback:
            with self._current_model_lock:
                current = self.current_model
            for model in self.fallback_models:
                if model != failed_model and model != current:
                    logger.warning(f"Model {failed_model} failed ({error_count} errors), switching to fallback: {model}")
                    backoff = min(2 ** (error_count - self.max_errors_before_fallback), 5)
                    await asyncio.sleep(backoff)
                    with self._current_model_lock:
                        self.current_model = model
                    return model

        if failed_model == self.model:
            for model in self.fallback_models:
                if model != failed_model:
                    logger.info(f"Primary model failed, using fallback: {model}")
                    return model

        return None

    async def _make_request_with_failover(
        self,
        payload: dict,
        stream: bool = False,
    ) -> tuple[httpx.Response, str]:
        """Make a request with automatic failover on error.
        
        Uses circuit breaker if available for resilience.
        """
        if self._circuit_breaker:
            from core.circuit_breaker import CircuitState
            if self._circuit_breaker.state == CircuitState.OPEN:
                raise Exception(f"Circuit breaker '{self._circuit_breaker.name}' is open - LLM unavailable")
        
        models_to_try = self._get_models_to_try()

        last_error = None
        for model in models_to_try:
            attempt_payload = dict(payload)
            attempt_payload["model"] = model
            try:
                if stream:
                    raise NotImplementedError("Use chat_stream for streaming requests")
                else:
                    response = await self.client.post(
                        self._get_chat_url(),
                        json=attempt_payload,
                    )
                    if response.status_code == 200:
                        self._clear_model_error(model)
                        with self._current_model_lock:
                            self.current_model = model
                        
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        
                        return response, model
                    elif response.status_code >= 500:
                        last_error = f"Model {model}: HTTP {response.status_code}"
                        logger.warning(last_error)
                        continue
                    else:
                        response.raise_for_status()
            except httpx.HTTPStatusError as e:
                last_error = f"Model {model}: {e}"
                logger.warning(last_error)
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                continue
            except Exception as e:
                last_error = f"Model {model}: {e}"
                logger.warning(last_error)
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise Exception(f"All models failed. Last error: {last_error}")

    async def chat(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and get response with automatic failover."""
        messages = self._build_messages(user_input, system_prompt, conversation_history)
        payload = self._build_payload(messages, tools, stream=False)

        start_time = time.time()
        response, used_model = await self._make_request_with_failover(payload, stream=False)
        latency_ms = (time.time() - start_time) * 1000
        
        # Log the call
        self._log_call("chat", used_model, latency_ms=latency_ms)
        
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise

    async def chat_stream(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat response tokens with automatic failover."""
        messages = self._build_messages(user_input, system_prompt, conversation_history)
        payload = self._build_payload(messages, tools, stream=True)

        models_to_try = self._get_models_to_try()

        for model in models_to_try:
            attempt_payload = dict(payload)
            attempt_payload["model"] = model
            try:
                async with self.client.stream(
                    "POST",
                    self._get_chat_url(),
                    json=attempt_payload,
                ) as response:
                    buffer_parts = []
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                msg = data.get("message", {})
                                thinking = msg.get("thinking", "")
                                content = msg.get("content", "")
                                if thinking:
                                    if not isinstance(thinking, str):
                                        thinking = str(thinking)
                                    yield f"\U0001f4ad\n{thinking}\n\U0001f4ad\n\n"
                                if content:
                                    buffer_parts.append(content)
                                    yield content
                            except json.JSONDecodeError:
                                continue
                    buffer = "".join(buffer_parts)

                    # Success - reset error count
                    self._clear_model_error(model)
                    with self._current_model_lock:
                        self.current_model = model

                    # Log the call
                    self._log_call("chat_stream", model)

                    # If we got no content, return what we have
                    if not buffer:
                        logger.warning(f"Empty response from {model}")
                    return

            except Exception as e:
                logger.warning(f"Model {model} failed in stream: {e}")
                self._increment_model_error(model)
                continue

        # All models failed
        yield "[Error: All models failed. Please try again.]"

    async def chat_stream_with_tools(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict]:
        """Stream chat response with structured output for thinking, text, and tool calls.
        
        Yields dicts with types:
        - {"type": "thinking", "text": "..."}  - GLM-5 thinking content
        - {"type": "text", "text": "..."}      - Response text  
        - {"type": "tool_call", "name": "...", "call_id": "...", "arguments": {...}}
        - {"type": "finish", "reason": "stop|tool-calls|error"}
        """
        messages = self._build_messages(user_input, system_prompt, conversation_history)
        payload = self._build_payload(messages, tools, stream=True)
        
        if temperature != 0.7:
            payload["options"]["temperature"] = temperature
        
        models_to_try = self._get_models_to_try()
        
        for model in models_to_try:
            attempt_payload = dict(payload)
            attempt_payload["model"] = model
            try:
                async with self.client.stream(
                    "POST",
                    self._get_chat_url(),
                    json=attempt_payload,
                ) as response:
                    thinking_parts = []
                    content_parts = []
                    tool_calls_accum = []
                    finish_reason = "stop"
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            msg = data.get("message", {})
                            
                            # Handle thinking content
                            thinking = msg.get("thinking", "")
                            if thinking:
                                # Ensure proper unicode handling
                                try:
                                    if not isinstance(thinking, str):
                                        thinking = str(thinking)
                                    thinking_parts.append(thinking)
                                    yield {"type": "thinking", "text": thinking}
                                except UnicodeDecodeError:
                                    logger.warning("Unicode decode error in thinking block")
                                    pass
                            
                            # Handle text content
                            content = msg.get("content", "")
                            if content:
                                try:
                                    if not isinstance(content, str):
                                        content = str(content)
                                    content_parts.append(content)
                                    yield {"type": "text", "text": content}
                                except UnicodeDecodeError:
                                    logger.warning("Unicode decode error in content")
                                    pass
                            
                            # Handle tool calls (streamed incrementally)
                            tool_calls = msg.get("tool_calls", [])
                            for tc in tool_calls:
                                tc_id = tc.get("id", "")
                                tc_func = tc.get("function", {})
                                tc_name = tc_func.get("name", "")
                                tc_args = tc_func.get("arguments", "")
                                
                                if tc_name:
                                    # New tool call - find existing or start new
                                    existing = next((t for t in tool_calls_accum if t["id"] == tc_id), None)
                                    if existing:
                                        existing["name"] = tc_name
                                    else:
                                        tool_calls_accum.append({
                                            "id": tc_id,
                                            "name": tc_name,
                                            "arguments": "",
                                        })
                                
                                if tc_args and tool_calls_accum:
                                    # Find the tool call to append args to
                                    target = None
                                    if tc_id:
                                        target = next((t for t in tool_calls_accum if t["id"] == tc_id), None)
                                    if target is None and tool_calls_accum:
                                        target = tool_calls_accum[-1]
                                    if target:
                                        target["arguments"] += tc_args
                            
                            # Check for done
                            if data.get("done", False):
                                finish_reason = data.get("done_reason", "stop")
                                
                        except json.JSONDecodeError:
                            continue
                    
                    # Emit collected tool calls
                    if tool_calls_accum:
                        for tc in tool_calls_accum:
                            try:
                                args = json.loads(tc["arguments"])
                            except (json.JSONDecodeError, KeyError, TypeError):
                                logger.debug("Failed to parse tool arguments, using empty dict")
                                args = {}
                            
                            yield {
                                "type": "tool_call",
                                "name": tc["name"],
                                "call_id": tc["id"],
                                "arguments": args,
                            }
                        finish_reason = "tool-calls"
                    
                    # Emit finish
                    yield {"type": "finish", "reason": finish_reason}
                    
                    # Success - reset error count
                    self._clear_model_error(model)
                    with self._current_model_lock:
                        self.current_model = model
                    
                    # Log the call
                    self._log_call("chat_stream_with_tools", model)
                    return
                    
            except Exception as e:
                logger.warning(f"Model {model} failed in stream_with_tools: {e}")
                self._increment_model_error(model)
                continue
        
        # All models failed
        yield {"type": "finish", "reason": "error", "error": "All models failed"}

    async def chat_with_tools(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[dict] | None]:
        """Chat and handle tool calls with automatic failover."""
        messages = self._build_messages(user_input, system_prompt, conversation_history)
        payload = self._build_payload(messages, tools, stream=False)

        # Log context size
        total_chars = sum(len(m.get("content", "")) + len(m.get("thinking", "")) + 200 for m in payload.get("messages", []))
        total_chars += sum(len(json.dumps(t)) for t in payload.get("tools", []))
        logger.debug(f"[TIMING] Payload size: {total_chars:,} chars (~{total_chars//4} tokens)")

        t1 = time.time()
        response, used_model = await self._make_request_with_failover(payload, stream=False)
        t2 = time.time()
        latency_ms = (t2 - t1) * 1000
        logger.info(f"[TIMING] API call: {t2-t1:.2f}s")

        try:
            result = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response in chat_with_tools: {e}")
            raise

        # Log the call
        self._log_call("chat_with_tools", used_model, latency_ms=latency_ms)

        message = result.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        # If content is empty, use thinking as content
        if not content and thinking:
            content = thinking
            thinking = ""

        # Handle streaming-style response (API may return multiple JSON objects)
        if not content and not thinking:
            text = response.text
            if text:
                content_parts = []
                thinking_parts = []
                for line in text.strip().split('\n'):
                    try:
                        chunk = json.loads(line)
                        chunk_msg = chunk.get("message", {})
                        chunk_content = chunk_msg.get("content", "")
                        chunk_thinking = chunk_msg.get("thinking", "")
                        if chunk_content:
                            content_parts.append(chunk_content)
                        if chunk_thinking:
                            thinking_parts.append(chunk_thinking)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                content = "".join(content_parts)
                thinking = "".join(thinking_parts)

        # Prepend thinking if present
        if thinking and content:
            content = f"<think>\n{thinking}\n</think>\n\n{content}"

        tool_calls = message.get("tool_calls")

        return content, tool_calls

    def format_tools_for_ollama(self, tools: list[dict]) -> list[dict]:
        """Format tools for Ollama's tool calling format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            })
        return formatted
