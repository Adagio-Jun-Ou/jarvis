"""Configuration management for Jarvis."""
import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    model: str = "glm-5"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int | None = None  # None = use model's maximum context
    api_key: str | None = None  # Set via OLLAMA_API_KEY env var or config
    use_cloud: bool = False  # Set to True to use Ollama cloud API
    vision_model: str | None = None  # Vision-capable model for image analysis (e.g., "qwen3-vl", "llava")
    fallback_models: list[str] = ["glm-5"]  # Models to try if primary fails
    timeout: int = 120
    stream_timeout: int = 300


class BackendLLMConfig(BaseModel):
    """Configuration for backend/CLI LLM (Kimi-K2.7-Code for code, analysis, refinement)."""
    model: str = "kimi-k2.7-code"
    use_cloud: bool = True
    api_key: str | None = None
    fallback_models: list[str] = ["glm-5.3-flash", "glm-5.1"]  # Primary model takes over on failure
    temperature: float = 0.3  # Lower for code/analysis tasks
    timeout: int = 180


class ConceptLLMConfig(BaseModel):
    """Configuration for concept LLM - high-level pattern recognition and abstraction.
    
    In the dual-model architecture, concept tasks are handled by the backend LLM (GLM-5).
    This config is kept for backward compatibility but routes to backend_llm.
    """
    model: str = "glm-5"  # Now uses backend LLM for concept tasks
    use_cloud: bool = True
    api_key: str | None = None
    temperature: float = 0.5  # Moderate - creative but focused
    timeout: int = 300  # Longer for complex reasoning
    enabled: bool = False  # Disabled - tasks routed to backend_llm


class AgentConfig(BaseModel):
    name: str = "Jarvis"
    max_history_turns: int = 20
    max_history_steps: int = 100  # Maximum steps in execution history
    tool_timeout: float = 60.0
    max_concurrent_tools: int = 5
    max_agentic_iterations: int = 5
    tool_timeouts: dict[str, float] = {
        "web_search": 30.0,
        "web_fetch": 60.0,
        "shell": 30.0,
        "vision": 120.0,
        "transcribe": 180.0,
        "docker": 60.0,
        "imagine": 10.0,
        "default": 60.0,
    }


class VoiceConfig(BaseModel):
    enabled: bool = True
    wake_word: str = "hey jarvis"
    tts_engine: str = "espeak"
    stt_model: str = "base"


class PermissionConfig(BaseModel):
    shell_commands: str = "ask"
    file_delete: str = "ask"
    file_write: str = "ask"
    web_access: str = "allow"


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_context_tokens: int | None = None  # None = dynamic, adapts to conversation needs
    long_term_storage: str = "~/.jarvis/memory"


class WebSearchConfig(BaseModel):
    enabled: bool = True
    provider: str = "duckduckgo"


class SystemToolsConfig(BaseModel):
    enabled: bool = True
    sandbox_dir: str = "~"


class ToolsConfig(BaseModel):
    web_search: WebSearchConfig = WebSearchConfig()
    system: SystemToolsConfig = SystemToolsConfig()


class InterfaceConfig(BaseModel):
    default: str = "cli"
    web_port: int = 8080


class ThinkingConfig(BaseModel):
    """Configuration for proactive thinking engine."""
    enabled: bool = True
    mode: str = "supportive"  # supportive, analytical, creative
    suggestion_interval: int = 30  # seconds between background thinking
    confidence_threshold: float = 0.7  # minimum confidence to suggest
    max_suggestions_per_turn: int = 3
    pause_detection: bool = True  # detect natural conversation pauses


class ContextConfig(BaseModel):
    """Configuration for dynamic context management."""
    max_tokens: int | None = None  # None = use model's maximum
    history_strategy: str = "smart"  # smart, recent, summarized, semantic
    importance_tracking: bool = True  # track important topics


class DockerConfig(BaseModel):
    """Configuration for Docker automation."""
    enabled: bool = True
    socket: str = "/var/run/docker.sock"


class N8NConfig(BaseModel):
    """Configuration for n8n workflow automation."""
    enabled: bool = True
    url: str = "http://localhost:5678"
    api_key: str | None = None  # Set via N8N_API_KEY env var


class AutomationConfig(BaseModel):
    """Configuration for automation tools."""
    docker: DockerConfig = DockerConfig()
    n8n: N8NConfig = N8NConfig()


class IdentityConfig(BaseModel):
    """Configuration for assistant identity."""
    name: str = "Jarvis"
    voice_engine: str = "auto"  # auto, pyttsx3, espeak, neural
    voice_id: str | None = None  # Specific voice ID or name
    voice_rate: int = 180
    voice_pitch: float = 1.0


class HardwareConfig(BaseModel):
    """Configuration for hardware acceleration."""
    gpu_enabled: bool = True
    npu_enabled: bool = True
    gpu_memory_mb: int = 4096
    reserved_system_mb: int = 4096
    max_cache_mb: int = 8192
    preferred_backend: str = "auto"
    enable_fp16: bool = True
    enable_int8: bool = True
    tensor_parallel_size: int = 1
    batch_size: int | None = None
    accelerator_priority: list[str] = ["npu", "rocm", "cuda", "vulkan", "cpu"]
    npu_int8_quantize: bool = True
    npu_model_cache: str = "~/.jarvis/npu_models"
    onnx_optimization_level: str = "all"


class SecurityConfig(BaseModel):
    """Configuration for security and authentication."""
    require_auth: bool = True  # Require API key for all non-localhost access
    api_key: str | None = None  # Set via JARVIS_API_KEY env var
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    # Per-IP cap, enforced by auth_middleware.SecurityMiddleware (loopback
    # exempt). The web dashboard polls aggressively (flow visualization at
    # 200ms alone is 300 req/min), so this default matches the web layer's
    # historical value rather than the never-enforced 120.
    rate_limit_requests_per_minute: int = 2000
    max_request_size_mb: int = 10
    xss_protection: bool = True
    csrf_protection: bool = True


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    backend_llm: BackendLLMConfig = BackendLLMConfig()
    concept_llm: ConceptLLMConfig = ConceptLLMConfig()
    agent: AgentConfig = AgentConfig()
    voice: VoiceConfig = VoiceConfig()
    permissions: PermissionConfig = PermissionConfig()
    memory: MemoryConfig = MemoryConfig()
    tools: ToolsConfig = ToolsConfig()
    interface: InterfaceConfig = InterfaceConfig()
    thinking: ThinkingConfig = ThinkingConfig()
    context: ContextConfig = ContextConfig()
    automation: AutomationConfig = AutomationConfig()
    identity: IdentityConfig = IdentityConfig()
    hardware: HardwareConfig = HardwareConfig()
    security: SecurityConfig = SecurityConfig()

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            print(f"Config file not found at {config_path}, using defaults")
            return cls()

        with open(config_path, encoding='utf-8') as f:
            content = f.read()
        SAFE_ENV_VARS = {"OLLAMA_API_KEY", "JARVIS_HOME", "HOME", "USER", "PATH"}
        for var in SAFE_ENV_VARS:
            val = os.environ.get(var, "")
            if val:
                content = content.replace(f"${{{var}}}", val)
                content = content.replace(f"${var}", val)
        data = yaml.safe_load(content) or {}

        if "llm" not in data:
            data["llm"] = {}
        if os.environ.get("OLLAMA_API_KEY") and not data.get("llm", {}).get("api_key"):
            data["llm"]["api_key"] = os.environ.get("OLLAMA_API_KEY")
        if os.environ.get("OLLAMA_API_KEY") and not data.get("llm", {}).get("use_cloud"):
            data["llm"]["use_cloud"] = True

        if "backend_llm" not in data:
            data["backend_llm"] = {}
        if data.get("llm", {}).get("api_key") and not data.get("backend_llm", {}).get("api_key"):
            data["backend_llm"]["api_key"] = data["llm"]["api_key"]

        if "concept_llm" not in data:
            data["concept_llm"] = {}
        if data.get("llm", {}).get("api_key") and not data.get("concept_llm", {}).get("api_key"):
            data["concept_llm"]["api_key"] = data["llm"]["api_key"]

        return cls(**data)

    def get_storage_path(self) -> Path:
        """Get the storage path for persistent data."""
        path = Path(self.memory.long_term_storage).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_permissions_path(self) -> Path:
        """Get the path for stored permissions."""
        return self.get_storage_path() / "permissions.json"
