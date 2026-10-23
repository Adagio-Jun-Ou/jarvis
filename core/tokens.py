"""Token counting and context optimization for Jarvis.

Provides accurate token counting without external dependencies.
Supports multiple model context windows with configurable thresholds.

MEMORY CONFIGURATION:
- compression_threshold: When to start compressing (0.0-1.0 of max_tokens)
- proactive_summarize: When to proactively summarize (0.0-1.0 of max_tokens)
- reserved_output: Tokens reserved for model output
"""
import re
import logging
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

_token_counter_cache: dict[str, "TokenCounter"] = {}
_token_counter_lock = threading.Lock()


@dataclass
class ModelContextWindow:
    """Context window configuration for a model.
    
    Attributes:
        name: Model identifier
        max_tokens: Maximum context window size
        input_cost_per_1k: Cost per 1K input tokens
        output_cost_per_1k: Cost per 1K output tokens
        reserved_output: Tokens reserved for model output
        compression_threshold: Fraction of max_tokens before compression (0.0-1.0)
        proactive_summarize: Fraction before proactive summarization (0.0-1.0)
        supports_vision: Whether model supports image inputs
        supports_thinking: Whether model outputs thinking/reasoning
    """
    name: str
    max_tokens: int
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    reserved_output: int = 4096
    compression_threshold: float = 0.85
    proactive_summarize: float = 0.50
    supports_vision: bool = False
    supports_thinking: bool = False


KNOWN_MODELS: dict[str, ModelContextWindow] = {
    "qwen3.5": ModelContextWindow(
        "qwen3.5", 131072,
        compression_threshold=0.90,
        proactive_summarize=0.55,
        supports_vision=True,
    ),
    "qwen3.5:397b": ModelContextWindow(
        "qwen3.5:397b", 131072,
        compression_threshold=0.90,
        proactive_summarize=0.55,
        supports_vision=True,
    ),
    "qwen3-vl": ModelContextWindow(
        "qwen3-vl", 32768,
        compression_threshold=0.85,
        proactive_summarize=0.50,
        supports_vision=True,
    ),
    "qwen3-vl:235b": ModelContextWindow(
        "qwen3-vl:235b", 32768,
        compression_threshold=0.85,
        proactive_summarize=0.50,
        supports_vision=True,
    ),
    "glm-5": ModelContextWindow(
        "glm-5", 131072,
        compression_threshold=0.90,
        proactive_summarize=0.50,
        supports_thinking=True,
    ),
    "glm-4": ModelContextWindow(
        "glm-4", 131072,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "kimi-k2.5": ModelContextWindow(
        "kimi-k2.5", 131072,
        compression_threshold=0.90,
        proactive_summarize=0.50,
    ),
    "llama3": ModelContextWindow(
        "llama3", 8192,
        compression_threshold=0.80,
        proactive_summarize=0.50,
    ),
    "llama3.1": ModelContextWindow(
        "llama3.1", 131072,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "llama3.2": ModelContextWindow(
        "llama3.2", 131072,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "mistral": ModelContextWindow(
        "mistral", 32768,
        compression_threshold=0.80,
        proactive_summarize=0.50,
    ),
    "mistral-nemo": ModelContextWindow(
        "mistral-nemo", 131072,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "qwen2.5": ModelContextWindow(
        "qwen2.5", 131072,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "codellama": ModelContextWindow(
        "codellama", 16384,
        compression_threshold=0.75,
        proactive_summarize=0.50,
    ),
    "deepseek-coder": ModelContextWindow(
        "deepseek-coder", 16384,
        compression_threshold=0.75,
        proactive_summarize=0.50,
    ),
    "phi3": ModelContextWindow(
        "phi3", 128000,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "gemma2": ModelContextWindow(
        "gemma2", 8192,
        compression_threshold=0.75,
        proactive_summarize=0.50,
    ),
    "command-r": ModelContextWindow(
        "command-r", 128000,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
    "claude-3": ModelContextWindow(
        "claude-3", 200000,
        compression_threshold=0.90,
        proactive_summarize=0.50,
    ),
    "gpt-4": ModelContextWindow(
        "gpt-4", 128000,
        compression_threshold=0.85,
        proactive_summarize=0.50,
    ),
}

DEFAULT_MODEL_CONFIG = ModelContextWindow(
    "default", 8192,
    compression_threshold=0.80,
    proactive_summarize=0.50,
)


class TokenCounter:
    """Token counter with multiple estimation methods.

    Provides model-specific context management with:
    - Per-model compression thresholds
    - Proactive summarization triggers
    - Unified token counting
    """

    AVG_CHARS_PER_TOKEN = 4.0
    AVG_CHARS_PER_TOKEN_CJK = 2.0

    def __init__(self, model: str | None = None):
        self.model = model
        self._config = self._get_model_config(model)
        self.max_context = self._config.max_tokens
        self.reserved_output = self._config.reserved_output
        self.compression_threshold = self._config.compression_threshold
        self.proactive_summarize = self._config.proactive_summarize
        self._tiktoken_available: bool | None = None
        self._cl100k_base = None

    def _get_model_config(self, model: str | None) -> ModelContextWindow:
        """Get model configuration, matching partial names."""
        if not model:
            return DEFAULT_MODEL_CONFIG
        model_lower = model.lower()
        for name, config in KNOWN_MODELS.items():
            if name in model_lower or model_lower in name:
                return config
        return DEFAULT_MODEL_CONFIG

    @property
    def compression_threshold_tokens(self) -> int:
        """Get the token count at which compression should start."""
        return int(self.max_context * self.compression_threshold)

    @property
    def proactive_summarize_tokens(self) -> int:
        """Get the token count for proactive summarization."""
        return int(self.max_context * self.proactive_summarize)

    def needs_compression(self, messages: list[dict]) -> bool:
        """Check if context needs compression."""
        return self.count_messages(messages) >= self.compression_threshold_tokens

    def needs_proactive_summarize(self, messages: list[dict]) -> bool:
        """Check if context needs proactive summarization."""
        return self.count_messages(messages) >= self.proactive_summarize_tokens

    def _check_tiktoken(self) -> bool:
        if self._tiktoken_available is not None:
            return self._tiktoken_available
        try:
            import tiktoken
            self._tiktoken_available = True
            self._cl100k_base = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._tiktoken_available = False
        return self._tiktoken_available

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using best available method."""
        if not text:
            return 0

        if self._check_tiktoken() and self._cl100k_base is not None:
            return len(self._cl100k_base.encode(text))

        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using heuristics."""
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
        other_chars = len(text) - cjk_chars

        estimated = (
            other_chars / self.AVG_CHARS_PER_TOKEN +
            cjk_chars / self.AVG_CHARS_PER_TOKEN_CJK
        )

        words = len(text.split())
        word_estimate = words * 1.3

        code_chars = len(re.findall(r'[{}()\[\];:,.]', text))
        code_penalty = code_chars * 0.2

        estimated += code_penalty

        return int((estimated + word_estimate) / 2) + 1

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += 4
            if "role" in msg:
                total += len(msg["role"]) // 4 + 1
            if "name" in msg:
                total += len(msg["name"]) // 4 + 1
            if "content" in msg:
                total += self.count_tokens(msg["content"])
            if "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    total += self.count_tokens(str(call))
        return total

    def get_available_context(self, messages: list[dict]) -> int:
        """Get remaining context window space."""
        used = self.count_messages(messages)
        return max(0, self.max_context - self.reserved_output - used)

    def can_fit(self, messages: list[dict], additional_tokens: int = 0) -> bool:
        """Check if messages fit in context window."""
        return self.count_messages(messages) + additional_tokens <= self.max_context - self.reserved_output

    def trim_messages(
        self,
        messages: list[dict],
        target_tokens: int | None = None,
        keep_system: bool = True,
        keep_recent: int = 2,
    ) -> list[dict]:
        """Trim messages to fit within target token count."""
        target = target_tokens or (self.max_context - self.reserved_output)

        if self.count_messages(messages) <= target:
            return messages

        result = []
        system_messages = []
        recent_messages = []

        for msg in messages:
            if keep_system and msg.get("role") == "system":
                system_messages.append(msg)
            else:
                recent_messages.append(msg)

        result.extend(system_messages)

        recent_to_keep = recent_messages[-keep_recent * 2:] if keep_recent else []
        older_messages = recent_messages[:-keep_recent * 2] if keep_recent else recent_messages

        result.extend(recent_to_keep)

        current_tokens = self.count_messages(result)
        target - current_tokens

        for msg in reversed(older_messages):
            msg_tokens = self.count_messages([msg])
            if current_tokens + msg_tokens <= target:
                result.insert(len(system_messages), msg)
                current_tokens += msg_tokens
            else:
                break

        return result

    def optimize_context(
        self,
        messages: list[dict],
        importance_scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """Optimize context by keeping important messages."""
        if not importance_scores or self.count_messages(messages) <= self.max_context - self.reserved_output:
            return self.trim_messages(messages)

        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        scored_indices = []
        for i, msg in enumerate(other_messages):
            score = importance_scores.get(i, 0.5)
            recency_bonus = (i / len(other_messages)) * 0.3
            scored_indices.append((i, score + recency_bonus))

        scored_indices.sort(key=lambda x: x[1], reverse=True)

        target = self.max_context - self.reserved_output
        current_tokens = self.count_messages(system_messages)
        selected_indices = set()

        for idx, score in scored_indices:
            msg_tokens = self.count_messages([other_messages[idx]])
            if current_tokens + msg_tokens <= target:
                selected_indices.add(idx)
                current_tokens += msg_tokens

        sorted_indices = sorted(selected_indices)
        result = system_messages + [other_messages[i] for i in sorted_indices]

        return result


def get_model_config(model: str) -> ModelContextWindow:
    """Get configuration for a known model, returns default if not found."""
    if not model:
        return DEFAULT_MODEL_CONFIG
    model_lower = model.lower()
    for name, config in KNOWN_MODELS.items():
        if name in model_lower or model_lower in name:
            return config
    return DEFAULT_MODEL_CONFIG


def get_token_counter(model: str | None = None) -> TokenCounter:
    """Get a cached TokenCounter instance for the given model."""
    global _token_counter_cache, _token_counter_lock
    cache_key = model or "default"
    with _token_counter_lock:
        if cache_key not in _token_counter_cache:
            _token_counter_cache[cache_key] = TokenCounter(model)
            if len(_token_counter_cache) > 20:
                oldest_key = next(iter(_token_counter_cache))
                del _token_counter_cache[oldest_key]
        return _token_counter_cache[cache_key]


def get_compression_threshold(model: str) -> int:
    """Get the token threshold for compression for a model."""
    config = get_model_config(model)
    return int(config.max_tokens * config.compression_threshold)


def get_proactive_summarize_threshold(model: str) -> int:
    """Get the token threshold for proactive summarization."""
    config = get_model_config(model)
    return int(config.max_tokens * config.proactive_summarize)


def count_tokens(text: str, model: str | None = None) -> int:
    """Convenience function to count tokens for text."""
    counter = get_token_counter(model)
    return counter.count_tokens(text)


def count_messages_tokens(messages: list[dict], model: str | None = None) -> int:
    """Convenience function to count tokens in messages."""
    counter = get_token_counter(model)
    return counter.count_messages(messages)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """Estimate cost for a request."""
    config = get_model_config(model)
    input_cost = (input_tokens / 1000) * config.input_cost_per_1k
    output_cost = (output_tokens / 1000) * config.output_cost_per_1k
    return input_cost + output_cost


def get_all_model_configs() -> dict[str, ModelContextWindow]:
    """Get all known model configurations."""
    return KNOWN_MODELS.copy()
