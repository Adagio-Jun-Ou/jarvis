#!/usr/bin/env python3
"""Jarvis launcher with proper logging configuration."""
import os

# Disable ALL GPU backends BEFORE any imports to prevent ROCm crashes
# Must be set BEFORE any library that might initialize GPU context
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''
os.environ['HSA_OVERRIDE_GFX_VERSION'] = ''  # Disable AMD GPU detection
os.environ['OPENCL_PLATFORMS'] = ''  # Prevent OpenCL GPU detection
os.environ['ROCM_HOME'] = ''  # Disable ROCm home
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU messages
os.environ['DISABLE_ONEDNN'] = '1'  # Disable oneDNN which might trigger GPU

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Isolate the torch inductor cache. Hardcoding /tmp put it on the wrong volume
# (or an unwritable one) on non-POSIX hosts; honour the platform temp dir, and
# let JARVIS_CACHE_DIR override it when the system drive is short on space.
_cache_root = Path(os.environ.get("JARVIS_CACHE_DIR") or tempfile.gettempdir())
os.environ.setdefault('TORCHINDUCTOR_CACHEDIR', str(_cache_root / "jarvis_torch_cache"))

# Windows consoles default to a legacy codepage (cp1252). Jarvis prints emoji in
# its banner, status lines and logs, so any of those would raise
# UnicodeEncodeError and kill the process — after every subsystem had already
# started. Force UTF-8 on the standard streams before anything writes to them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream
        pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load API keys from secrets file before anything else
_secrets_file = Path.home() / ".jarvis" / "secrets" / "api_keys.env"
if _secrets_file.exists():
    for line in _secrets_file.read_text(encoding='utf-8').strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            sep_idx = line.find("=")
            key = line[:sep_idx].strip()
            value = line[sep_idx + 1:].strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def setup_logging(debug: bool = False) -> None:
    """Configure logging for Jarvis."""
    level = logging.DEBUG if debug else logging.INFO

    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Explicit UTF-8: log records carry emoji, and the default here is
            # the platform codepage, which raises on the first non-ASCII record.
            logging.FileHandler(log_dir / "jarvis.log", encoding="utf-8"),
        ],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Jarvis - Personal AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interface", "-i",
        choices=["cli", "terminal", "voice", "web"],
        default=None,
        help="Interface to use (default: from config or cli)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model to use",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for web interface",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--conversational-speech",
        action="store_true",
        help="Enable conversational speech model for voice interface",
    )
    parser.add_argument(
        "--no-n8n",
        action="store_true",
        help="Disable n8n auto-start",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run as a Model Context Protocol (MCP) stdio server for external clients",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        default=True,
        help="Force HTTPS (default: True, requires SSL certs)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=False,
        help="Allow HTTP (not recommended for production)",
    )
    args = parser.parse_args()

    # MCP stdio mode: the real stdout is reserved for JSON-RPC frames.
    # Redirect sys.stdout BEFORE logging is configured so no banner,
    # print or log line can corrupt the protocol stream.
    if args.mcp:
        sys.stdout = sys.stderr

    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger("jarvis")

    # Pre-boot validation
    try:
        from core.startup_validator import StartupValidator
        validator = StartupValidator(base_path=Path(__file__).parent)
        validator.validate()
        validator.print_report()
        summary = validator.get_summary()
        if not summary["ready"]:
            logger.error(f"Startup validation failed: {summary['errors']} errors")
    except Exception as e:
        logger.debug(f"Startup validator skipped: {e}")

    # Import after logging is configured
    from core.agent import JarvisAgent
    from core.config import Config

    # Load configuration
    config = Config.load(args.config)

    # Override config with command line arguments
    if args.interface:
        config.interface.default = args.interface
    if args.model:
        config.llm.model = args.model
    if args.port:
        config.interface.web_port = args.port
    if args.no_n8n:
        config.automation.n8n.enabled = False
    
    # Force HTTPS unless --insecure is specified
    if args.insecure:
        args.https = False

    logger.info(f"Starting {config.identity.name} with {config.interface.default} interface")
    logger.info(f"Using model: {config.llm.model}")
    logger.info(f"Thinking mode: {config.thinking.mode}")

    # Track background tasks for cleanup
    background_tasks: list[asyncio.Task] = []
    
    # Create agent
    agent = JarvisAgent(config)

    # Get API keys from environment
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_engine_id = os.environ.get("GOOGLE_ENGINE_ID")
    n8n_api_key = os.environ.get("N8N_API_KEY")

    # Begin batch registration to avoid repeated prompt rebuilds
    agent.begin_batch_registration()

    # Register tools
    from tools.system import SystemTools
    from tools.web import WebTools
    from tools.productivity import ProductivityTools
    from tools.home import HomeTools
    from tools.spatial import SpatialTools
    from tools.art_philosophy import ArtPhilosophyTools
    from tools.physics_sim import PhysicsSimTools
    from tools.docker import DockerTool, DockerComposeTool
    from tools.n8n import N8NTool
    from tools.meta_tools import get_meta_tools
    from tools.data_tools import get_data_tools
    from tools.free_time import get_free_time_tools
    from core.multimodal import MultimodalManager, VisionTool, AudioTranscriptionTool
    from core.vision_tokens import WebcamCapture, WebcamTool
    from core.imagination import ImaginationEngine, ImaginationTool
    from core.dream import DreamEngine, DreamModeTool
    from core.growth import SelfGrowthEngine, SelfGrowthTool

    system_tools = SystemTools(sandbox_dir=config.tools.system.sandbox_dir)
    web_tools = WebTools(google_api_key=google_api_key, google_engine_id=google_engine_id)
    productivity_tools = ProductivityTools(storage_path=config.get_storage_path())
    home_tools = HomeTools()
    spatial_tools = SpatialTools()
    art_philosophy_tools = ArtPhilosophyTools()
    physics_sim_tools = PhysicsSimTools()

    agent.register_tools(system_tools.get_tools())
    agent.register_tools(web_tools.get_tools())
    agent.register_tools(productivity_tools.get_tools())
    agent.register_tools(home_tools.get_tools())
    agent.register_tools(spatial_tools.get_tools())
    agent.register_tools(art_philosophy_tools.get_tools())
    agent.register_tools(physics_sim_tools.get_tools())

    # Automation tools
    n8n_tool = None  # Track for daydream integration
    if config.automation.docker.enabled:
        try:
            docker_tool = DockerTool(docker_socket=config.automation.docker.socket)
            docker_compose_tool = DockerComposeTool()
            agent.register_tools([docker_tool, docker_compose_tool])
            logger.info("Docker tools queued")
        except Exception as e:
            logger.warning(f"Docker tools not available: {e}")

    if config.automation.n8n.enabled:
        try:
            n8n_tool = N8NTool(
                base_url=config.automation.n8n.url,
                api_key=n8n_api_key or config.automation.n8n.api_key,
            )
            agent.register_tools([n8n_tool])
            logger.info("n8n tool queued")
        except Exception as e:
            logger.warning(f"n8n tools not available: {e}")

    data_tools = get_data_tools()
    agent.register_tools(data_tools)
    logger.info("Data analysis tools queued")

    try:
        from tools.oracle import OraclePredictTool
        oracle_tool = OraclePredictTool()
        agent.register_tools([oracle_tool])
        logger.info("Oracle prediction tool queued")
    except ImportError as e:
        logger.warning(f"Oracle prediction tool not available: {e}")

    try:
        from tools.cognitive_ai import CognitivePsychologyTool, MachineAlignmentTool, NeuroscienceTool
        agent.register_tools([CognitivePsychologyTool(), MachineAlignmentTool(), NeuroscienceTool()])
        logger.info("Cognitive-AI tools queued")
    except ImportError as e:
        logger.warning(f"Cognitive-AI tools not available: {e}")

    agent.end_batch_registration()
    logger.info(f"Batch registered {len(agent.tools)} core tools")

    # Start the shared embedding server before anything asks for an embedding.
    # EmbeddingInterconnect is designed around a single shared model reached over
    # IPC, but nothing ever launched the server, so every process — Jarvis itself
    # plus each sub-agent worker — silently fell back to loading its own copy of
    # the ~239M-parameter Jina model (~1GB resident each). Best-effort: if it
    # cannot start, the client's local fallback still applies.
    try:
        from core.embedding_server import start_server_if_needed

        await start_server_if_needed()
    except Exception as e:
        logger.warning(
            f"Shared embedding server unavailable, each consumer will load its "
            f"own model: {e}"
        )

    # Start agent
    await agent.start()

    # Initialize API Call Manager for coordinated LLM calls
    from core.api_manager import get_api_manager
    api_manager = get_api_manager(llm_client=agent.llm)
    await api_manager.start()
    logger.info("API Call Manager started for coordinated LLM calls")

    # Initialize Model Coordinator for inter-model communication
    from core.model_coordinator import get_model_coordinator
    from core.backend_llm import get_backend_llm
    
    # Get separate backend LLM (GLM-5) for reasoning/code tasks
    backend_llm = get_backend_llm(config)
    
    model_coordinator = get_model_coordinator(
        primary_llm=agent.llm,
        backend_llm=backend_llm,
    )
    # Attach shared resources
    if hasattr(agent, 'memory'):
        model_coordinator.shared_context.attach_memory(agent.memory)
    logger.info(f"Model Coordinator initialized: {list(model_coordinator.models.keys())}")
    
    # Initialize Dual Model Orchestrator for seamless model collaboration
    from core.dual_model import init_dual_model
    dual_model = init_dual_model(
        primary_llm=agent.llm,
        backend_llm=backend_llm,
        model_coordinator=model_coordinator,
    )
    await dual_model.start()
    logger.info("Dual-Model Orchestrator initialized")
    logger.info(f"  Primary tasks: chat, creative, vision, abstraction, meta-reasoning, patterns")
    logger.info(f"  Backend tasks: code, math, analysis, reasoning, refinement, synthesis")
    
    # Attach dual-model to agent for chat routing
    agent.attach_dual_model(backend_llm, dual_model, model_coordinator)
    logger.info("  Chat routing: dual-model enabled for complex queries")
    
    # Attach Prefrontal Cortex for meta-reasoning
    agent.attach_prefrontal_cortex()
    logger.info("Prefrontal Cortex attached for meta-reasoning")
    
    # Start Prefrontal Cortex background learning (async)
    async def start_prefrontal_cortex():
        if hasattr(agent, '_prefrontal_cortex') and agent._prefrontal_cortex:
            try:
                await agent._prefrontal_cortex.start()
                logger.info("Prefrontal Cortex started - observing and learning")
            except Exception as e:
                logger.error(f"Prefrontal Cortex start failed: {e}")
    
    pfc_task = asyncio.create_task(start_prefrontal_cortex())
    background_tasks.append(pfc_task)
    
    # Initialize Dual Subsystem Integration for core intelligence subsystems
    from core.dual_subsystem import get_dual_subsystem_integration, SubsystemTask
    dual_subsystem = get_dual_subsystem_integration(
        primary_llm=agent.llm,
        backend_llm=backend_llm,
        model_coordinator=model_coordinator,
    )
    logger.info("Dual Subsystem Integration initialized")
    logger.info(f"  Daydream: Primary (explore) → Backend (evaluate)")
    logger.info(f"  Erdős: Backend (math) + Parallel (insights)")
    logger.info(f"  Evolution: Primary (propose) → Backend (analyze)")
    logger.info(f"  Self-Refine: Backend (code) + Parallel (fixes)")
    logger.info(f"  Insight: Parallel (extract) → Backend (guide)")
    logger.info(f"  Learning: Parallel (discover) → Backend (evaluate)")
    logger.info(f"  Proactive: Primary (curious) + Parallel (reflect)")
    
    # Initialize Intelligent Orchestrator for complex task decomposition
    from core.intelligent_orchestrator import initialize_intelligent_orchestrator
    intelligent_orchestrator = initialize_intelligent_orchestrator(
        llm_client=agent.llm,
        backend_llm=backend_llm,
        dual_model=dual_model,
    )
    logger.info("Intelligent Orchestrator initialized")
    logger.info(f"  Task decomposition: Backend breaks down complex requests")
    logger.info(f"  Model agents: Primary (creative/chat) + Backend (code/math/analysis)")
    logger.info(f"  Parallel execution: Independent subtasks run concurrently")
    
    # Attach orchestrator to agent for complex task handling
    agent.attach_intelligent_orchestrator(intelligent_orchestrator)
    logger.info("Intelligent Orchestrator attached to agent")

    # Register additional tools after start
    agent.begin_batch_registration()

    if agent.reflection_engine:
        meta_tools = get_meta_tools(agent.reflection_engine)
        agent.register_tools(meta_tools)
        logger.info("Meta-cognitive tools queued")

    # Task decomposition (Claude Code TodoWrite semantics): the agent's
    # explicit working list, injected into the prompt and consulted by
    # the completion evaluator while tasks remain unfinished.
    try:
        from tools.task_list import UpdateTasksTool
        agent.register_tools([UpdateTasksTool()])
        logger.info("Task-list tool queued (update_tasks)")
    except Exception as e:
        logger.warning(f"Task-list tool not available: {e}")

    # Multimodal
    multimodal = MultimodalManager(
        llm_client=agent.llm,
        whisper_model=config.voice.stt_model,
        vision_model=config.llm.vision_model,
    )
    vision_tool = VisionTool(multimodal.vision)
    audio_tool = AudioTranscriptionTool(multimodal.audio)
    agent.register_tools([vision_tool, audio_tool])
    logger.info("Multimodal tools queued (vision, audio)")

    # Webcam — wired to the multimodal vision processor so captures are
    # actually analyzed, not just measured ("640x480 pixels")
    try:
        webcam_capture = WebcamCapture()
        webcam_tool = WebcamTool(webcam_capture, vision_processor=multimodal.vision)
        agent.register_tools([webcam_tool])
        logger.info("Webcam tool queued (vision-enabled)")
    except Exception as e:
        logger.debug(f"Webcam not available: {e}")

# Imagination
    imagination = ImaginationEngine()
    imagination_tool = ImaginationTool(imagination)
    agent.register_tools([imagination_tool])
    logger.info("4D Imagination engine queued")

    # Dream mode
    dream = DreamEngine(llm_client=agent.llm)
    dream_tool = DreamModeTool(dream)
    agent.register_tools([dream_tool])
    logger.info("Dream mode queued")

    # Self-growth
    growth = SelfGrowthEngine()
    growth_tool = SelfGrowthTool(growth)
    agent.register_tools([growth_tool])
    logger.info("Self-growth engine queued")

    # Security tools
    security_components = {}
    try:
        from tools.security_tools import get_security_tools
        security_tools, security_components = get_security_tools()
        agent.register_tools(security_tools)
        logger.info("Security suite queued (6 tools)")
    except Exception as e:
        logger.debug(f"Security tools not available: {e}")

    # Auto-refine
    try:
        from core.auto_refine import AutoRefineManager, AutoRefineTool
        refine = AutoRefineManager(llm_client=agent.llm, base_path=Path(__file__).parent)
        refine_tool = AutoRefineTool(refine)
        agent.register_tools([refine_tool])
        logger.info("Auto-refine system queued")
    except Exception as e:
        logger.warning(f"Auto-refine not available: {e}")

    # Refinement tools (OpenCode-style self-improvement)
    try:
        from tools.refine_tools import get_refine_tools
        refine_tools = get_refine_tools(Path(__file__).parent)
        agent.register_tools(refine_tools)
        logger.info(f"Refinement tools queued ({len(refine_tools)} tools)")
    except Exception as e:
        logger.debug(f"Refinement tools not available: {e}")

    # Voice tokens
    try:
        from core.voice_tokens import VoiceTokenManager
        voice_manager = VoiceTokenManager()
        setattr(agent, 'voice_tokens', voice_manager)
        logger.info("Voice token system queued")
    except Exception as e:
        logger.debug(f"Voice tokens not available: {e}")

    # Intuition bridge
    try:
        from core.intuition_bridge import IntuitionBridge
        bridge = IntuitionBridge()
        agent.intuition_bridge = bridge
        logger.info("Intuition bridge queued")
    except Exception as e:
        logger.warning(f"Intuition bridge not available: {e}")

    # Evolution & learning tools
    try:
        from tools.evolution_tools import register_evolution_tools
        from tools.learning_tools import ErdosMathTool, DefenseTool, LearningTool
        register_evolution_tools(agent)
        agent.register_tools([ErdosMathTool(), DefenseTool(), LearningTool()])
        logger.info("Evolution & learning tools queued")
    except Exception as e:
        logger.warning(f"Evolution tools not available: {e}")

    # Tool discovery tools
    try:
        from tools.tool_discovery import register_tool_discovery_tools
        count = register_tool_discovery_tools(agent)
        logger.info(f"Tool discovery tools queued ({count} tools)")
    except Exception as e:
        logger.debug(f"Tool discovery tools not available: {e}")

    # Free time sandbox tools
    try:
        free_time_tools = get_free_time_tools()
        agent.register_tools(free_time_tools)
        logger.info(f"Free time sandbox tools queued ({len(free_time_tools)} tools)")
    except Exception as e:
        logger.debug(f"Free time tools not available: {e}")

    # Project tools for agentic work
    try:
        from tools.project import register_project_tools
        project_tools = register_project_tools()
        agent.register_tools(project_tools)
        logger.info(f"Project tools registered ({len(project_tools)} tools)")
    except Exception as e:
        logger.debug(f"Project tools not available: {e}")

    # Screen vision tools
    try:
        from tools.screen_tools import register_screen_tools
        screen_count = register_screen_tools(agent)
        if screen_count > 0:
            logger.info(f"Screen vision tools queued ({screen_count} tools)")
    except Exception as e:
        logger.debug(f"Screen vision tools not available: {e}")

    # Music generation and tokenization tools
    try:
        from tools.music import register_music_tools
        music_count = register_music_tools(agent.tools)
        if music_count > 0:
            logger.info(f"Music tools registered ({music_count} tools)")
    except Exception as e:
        logger.debug(f"Music tools not available: {e}")

    # Browser automation tools
    try:
        from tools.browser_tools import register_browser_tools
        browser_count = register_browser_tools(agent)
        if browser_count > 0:
            logger.info(f"Browser automation tools queued ({browser_count} tools)")
    except Exception as e:
        logger.debug(f"Browser automation tools not available: {e}")

    # Web UI self-refinement tools
    try:
        from tools.web_ui_refine import TOOLS as WEB_UI_REFINE_TOOLS
        web_ui_tools = [tool_cls() for tool_cls in WEB_UI_REFINE_TOOLS]
        agent.register_tools(web_ui_tools)
        if web_ui_tools:
            logger.info(f"Web UI refinement tools queued ({len(web_ui_tools)} tools)")
    except Exception as e:
        logger.warning(f"Web UI refinement tools not available: {e}")

    # UI Sensing tools - Jarvis can see its own UIs
    try:
        from tools.ui_sense import TOOLS as UI_SENSE_TOOLS, set_web_interface, log_ui_event
        agent.register_tools(UI_SENSE_TOOLS)
        logger.info(f"UI sensing tools queued ({len(UI_SENSE_TOOLS)} tools)")
    except Exception as e:
        logger.warning(f"UI sensing tools not available: {e}")

    # System Sensing tools - Jarvis can see its entire runtime environment
    # These are always-available core tools for agentic awareness
    try:
        from tools.system_sense import (
            TOOLS as SYSTEM_SENSE_TOOLS,
            set_agent_for_system_sense,
            set_web_interface_for_system_sense,
            integrate_with_security,
        )
        agent.register_tools(SYSTEM_SENSE_TOOLS)
        set_agent_for_system_sense(agent)
        
        # Integrate system monitoring with security tools
        if security_components:
            integration_status = integrate_with_security(security_components)
            logger.info(f"System sensing tools queued ({len(SYSTEM_SENSE_TOOLS)} tools, security integration: {integration_status})")
        else:
            logger.info(f"System sensing tools queued ({len(SYSTEM_SENSE_TOOLS)} tools)")
    except Exception as e:
        logger.warning(f"System sensing tools not available: {e}")

    # Tool Creation tools - Jarvis can create and test its own tools
    try:
        from tools.tool_creation import TOOLS as TOOL_CREATION_TOOLS
        agent.register_tools(TOOL_CREATION_TOOLS)
        logger.info(f"Tool creation tools queued ({len(TOOL_CREATION_TOOLS)} tools)")
    except Exception as e:
        logger.warning(f"Tool creation tools not available: {e}")

    # Insight Search tools - Search through accumulated knowledge
    try:
        from tools.insight_search import TOOLS as INSIGHT_SEARCH_TOOLS
        agent.register_tools(INSIGHT_SEARCH_TOOLS)
        logger.info(f"Insight search tools queued ({len(INSIGHT_SEARCH_TOOLS)} tools)")
    except Exception as e:
        logger.warning(f"Insight search tools not available: {e}")

    # Native terminal tools (like OpenCode)
    try:
        from tools.terminal import register_terminal_tools, get_jarvis_computer
        computer = get_jarvis_computer()
        terminal_count = register_terminal_tools(agent, computer)
        if terminal_count > 0:
            logger.info(f"Native terminal tools queued ({terminal_count} tools)")
    except Exception as e:
        logger.debug(f"Terminal tools not available: {e}")

    # Subagent Task tool (isolated context, summary-only return)
    try:
        from tools.subagent_task import register_subagent_tools
        register_subagent_tools(agent, agent)
    except Exception as e:
        logger.debug(f"Subagent task tool not available: {e}")

    # Model-triggered context compaction (Claude Code /compact semantics)
    try:
        from tools.compact_context import register_compact_tools
        register_compact_tools(agent, agent)
    except Exception as e:
        logger.debug(f"Compact context tool not available: {e}")

    agent.end_batch_registration()

    # Load skills (OpenClaw-style)
    try:
        from core.skill_loader import get_skill_loader
        loader = get_skill_loader(
            jarvis_path=Path(__file__).parent,
            workspace_path=config.get_storage_path().parent if hasattr(config, 'get_storage_path') else Path.cwd()
        )

        loader.load_all_skills()
        loaded = len(loader.get_loaded_skills())
        tool_count = loader.register_tools_with_agent(agent)
        logger.info(f"Skills loaded: {loaded} skills, {tool_count} tools registered")

        skill_content = loader.get_system_prompt_content()
        if skill_content:
            setattr(agent, '_skill_content', skill_content)

    except Exception as e:
        logger.warning(f"Skills not loaded: {e}")

    # Start background systems
    try:
        from core.evolution_orchestrator import get_evolution_orchestrator
        from core.erdos_engine import get_erdos_engine
        from core.defense import get_defense_agent
        from core.learning_pipeline import get_learning_pipeline
        from core.memory_cleanup import get_memory_manager
        from core.daydream_engine import get_daydream_engine
        from core.system_integrator import get_system_integrator
        from core.state_sync import get_state_synchronizer
        from core.observability import get_observability
        from core.free_time import get_free_time_engine
        from core.self_refine import get_self_refine_agent
        
        # ══════════════════════════════════════════════════════════════════
        # BI-MODEL ARCHITECTURE
        # ─────────────────────────────────────────────────────────────────
        # Primary (Kimi K2.5): Chat, creative, user-facing interactions
        # Backend (GLM-5): Code, math, reasoning, analysis, refinement
        # Model Coordinator: Routes tasks and enables cross-model collaboration
        # ══════════════════════════════════════════════════════════════════
        
        # Parallel initialization of subsystems (file I/O bound)
        import concurrent.futures
        max_workers = min(8, os.cpu_count() or 4)
        init_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="subsys_init_")
        
        logger.info("Initializing subsystems in parallel...")
        init_futures = {
            'orchestrator': init_pool.submit(get_evolution_orchestrator),
            'erdos': init_pool.submit(get_erdos_engine, backend_llm),
            'defense': init_pool.submit(get_defense_agent),
            'pipeline': init_pool.submit(get_learning_pipeline, backend_llm),
            'memory_mgr': init_pool.submit(get_memory_manager),
            'daydream': init_pool.submit(get_daydream_engine),
            'integrator': init_pool.submit(get_system_integrator),
            'state_sync': init_pool.submit(get_state_synchronizer),
            'observability': init_pool.submit(get_observability),
            'free_time': init_pool.submit(get_free_time_engine, backend_llm),
            'self_refine': init_pool.submit(get_self_refine_agent, backend_llm),
        }
        
        # Wait for all initializations
        orchestrator = init_futures['orchestrator'].result()
        logger.debug("  ✓ evolution_orchestrator")
        
        erdos = init_futures['erdos'].result()
        logger.debug("  ✓ erdos_engine")
        
        defense = init_futures['defense'].result()
        logger.debug("  ✓ defense_agent")
        
        pipeline = init_futures['pipeline'].result()
        logger.debug("  ✓ learning_pipeline")
        
        memory_mgr = init_futures['memory_mgr'].result()
        logger.debug("  ✓ memory_manager")
        
        daydream = init_futures['daydream'].result()
        logger.debug("  ✓ daydream_engine")
        
        integrator = init_futures['integrator'].result()
        logger.debug("  ✓ system_integrator")
        
        state_sync = init_futures['state_sync'].result()
        logger.debug("  ✓ state_sync")
        
        observability = init_futures['observability'].result()
        logger.debug("  ✓ observability")
        
        free_time = init_futures['free_time'].result()
        logger.debug("  ✓ free_time_engine")
        
        self_refine = init_futures['self_refine'].result()
        logger.debug("  ✓ self_refine_agent")
        
        init_pool.shutdown(wait=False)
        logger.info("All subsystems initialized")
        
        # Daydream Engine - needs idle status callback
        daydream.get_idle_status = lambda: agent.scheduler.is_idle() if agent.scheduler else True
        
        # Attach defense agent to learning pipeline for threat analysis
        pipeline.attach_defense_agent(defense)

        # Register subsystems
        orchestrator.attach_subsystem('erdos_engine', erdos)
        orchestrator.attach_subsystem('defense', defense)
        orchestrator.attach_subsystem('learning_pipeline', pipeline)
        memory_mgr.register_collection('history', erdos.history)
        memory_mgr.register_collection('sessions', pipeline.sessions)

        integrator.register_subsystem('evolution_orchestrator', orchestrator, dependencies=[])
        integrator.register_subsystem('erdos_engine', erdos, dependencies=['evolution_orchestrator'])
        integrator.register_subsystem('defense', defense, dependencies=[])
        integrator.register_subsystem('learning_pipeline', pipeline, dependencies=['erdos_engine', 'defense'])
        integrator.register_subsystem('daydream_engine', daydream, dependencies=['erdos_engine'])
        integrator.register_subsystem('state_sync', state_sync, dependencies=[])
        integrator.register_subsystem('observability', observability, dependencies=[])

        # Attach BioEvolve MAP-Elites engine to evolution orchestrator
        try:
            orchestrator.attach_bioevolve_engine()
            logger.info("  ✓ bioevolve_engine attached")
        except Exception as e:
            logger.warning(f"  ✗ bioevolve_engine unavailable: {e}")

        # Register health checks safely (avoid NameError if init fails)
        if 'erdos' in dir():
            observability.register_health_check('erdos_engine', lambda: (erdos._running, {'problems': len(erdos.problems)}))
        if 'pipeline' in dir():
            observability.register_health_check('learning_pipeline', lambda: (pipeline._running, {'sessions': len(pipeline.sessions)}))
        if 'defense' in dir():
            observability.register_health_check('defense', lambda: (defense._running, {'detected': defense._stats.get('threats_detected', 0)}))
        if 'daydream' in dir():
            observability.register_health_check('daydream_engine', lambda: (daydream._running, {'count': daydream._daydream_count}))
        if 'free_time' in dir():
            observability.register_health_check('free_time', lambda: (True, {'sessions': free_time._session_count, 'state': free_time._state.value}))
        if 'self_refine' in dir():
            observability.register_health_check('self_refine', lambda: (self_refine._running, {'tasks': len(self_refine._tasks)}))
        observability.register_health_check('dual_model', lambda: (dual_model._running, dual_model.get_stats()))
        observability.register_health_check('model_coordinator', lambda: (True, {'connected': len(model_coordinator.get_connected_models())}))
        observability.register_health_check('bioevolve', lambda: (orchestrator._bioevolve_engine is not None, orchestrator.get_bioevolve_status()))
        
        # Context cascade health check
        try:
            from core.context_cascade import get_context_cascade
            _cascade = get_context_cascade()
            observability.register_health_check('context_cascade', lambda: _cascade.health_check())
        except Exception as e:
            logger.debug(f"Context cascade health check skipped: {e}")
        
        # System Sense health check for agentic monitoring
        try:
            from tools.system_sense import get_system_sense_tool
            _system_sense = get_system_sense_tool()
            observability.register_health_check('system_sense', lambda: (True, {'agent_connected': _system_sense._agent is not None}))
        except Exception as e:
            logger.debug(f"System sense health check skipped: {e}")
        
        # Start subsystems in parallel
        logger.info("Starting subsystems...")
        await asyncio.gather(
            state_sync.start(),
            observability.start(),
            erdos.start_background_learning(),
            defense.start_monitoring(),
            pipeline.start_pipeline(),
            orchestrator.start_evolution(),
            self_refine.start(),
        )
        logger.info("All subsystems started")
        
        # Register telemetry collectors and degradation monitors
        try:
            from core.telemetry_aggregator import get_telemetry_aggregator, register_standard_collectors
            from core.graceful_degradation import get_degradation_manager, SubsystemStatus
            register_standard_collectors()
            degradation = get_degradation_manager()
            degradation.mark_healthy("memory_router")
            degradation.mark_healthy("adaptive_context")
            degradation.mark_healthy("prompt_sandbox")
            degradation.mark_healthy("attention_queue")
            degradation.mark_healthy("concept_drift")
            degradation.mark_healthy("load_balancer")
            degradation.mark_healthy("self_reflection")
            degradation.mark_healthy("memory_consolidation")
            degradation.mark_healthy("supervisor")
            logger.info("Telemetry aggregator + degradation manager initialized")
        except Exception as e:
            logger.debug(f"Telemetry/degradation initialization skipped: {e}")
        
        # Log dual-model architecture status
        logger.info("=" * 60)
        logger.info("DUAL-MODEL ARCHITECTURE ACTIVE")
        logger.info("=" * 60)
        logger.info(f"  Primary (Kimi K2.5):  Chat, Creative, Vision, Meta-reasoning, Abstractions")
        logger.info(f"  Backend (GLM-5):      Code, Math, Analysis, Synthesis, Refinement")
        logger.info(f"  Model Coordinator:    Cross-model collaboration enabled")
        logger.info(f"  Connected models:     {model_coordinator.get_connected_models()}")
        logger.info("=" * 60)

        # Daydream callbacks
        from core.insight_bridge import get_insight_bridge as _get_insight_bridge_fn
        def on_insight(insight, problem, category, evolution_trigger):
            try:
                bridge = _get_insight_bridge_fn()
                bridge.process_insight(insight, problem or "daydream", category, None)
            except Exception:
                pass

        setattr(daydream, 'on_insight', on_insight)
        
        # Index existing insights on startup
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            if len(review._hash_index) == 0:
                logger.info("Indexing insights on startup...")
                count = review.index_from_bridge()
                logger.info(f"Indexed {count} insights from insight_bridge")
        except Exception as e:
            logger.debug(f"Could not index insights on startup: {e}")
        
        # Connect daydream to agent's activity tracking
        def get_idle_status():
            if agent.scheduler:
                return agent.scheduler.is_idle()
            return False
        
        def get_queue_size():
            if agent.scheduler:
                return len(agent.scheduler._active_tasks)
            return 0
        
        def get_pending_count():
            return 0  # No permission queue in current setup
        
        daydream.get_idle_status = get_idle_status
        daydream.get_queue_size = get_queue_size
        daydream.get_pending_count = get_pending_count
        
        # Set up daydream callback for math exploration
        async def daydream_math_exploration():
            try:
                result = await erdos.daydream_step()
                if result and result.get("status") != "not_running":
                    logger.debug(f"Daydream explored: {result.get('problem_id', 'unknown')}")
            except Exception as e:
                logger.debug(f"Daydream math exploration error: {e}")
        
        daydream._daydream_callback = daydream_math_exploration
        
        await daydream.start()
        
        # Connect tool registry to daydream for practice during idle
        daydream._tool_registry = agent.tools
        daydream._agent = agent  # Pass agent reference for tool usage tracking
        logger.info("- Daydream: tool registry connected for practice")

        # Proactive intelligence engine - uses PRIMARY for curiosity/creative, BACKEND for analysis
        try:
            from core.proactive_engine import get_proactive_engine
            proactive = get_proactive_engine(backend_llm)  # Use backend for proactive analysis
            proactive.get_idle_status = lambda: agent.scheduler.is_idle() if agent.scheduler else True
            await proactive.start()
            logger.info("- Proactive Engine: autonomous intelligence when idle")
        except Exception as e:
            logger.debug(f"Proactive engine not available: {e}")

        # Music Creation Engine - always-on background classical composition
        # PFC continuously observes acoustic and symbolic tokens
        try:
            from core.music_creation_engine import get_music_creation_engine
            music_creation = get_music_creation_engine()
            await music_creation.start()
            logger.info("- Music Creation: continuous composition with PFC observation")
        except Exception as e:
            logger.debug(f"Music creation engine not available: {e}")

        # Connect n8n to daydream integration
        if n8n_tool:
            try:
                from core.daydream_n8n import get_daydream_n8n
                daydream_n8n = get_daydream_n8n(n8n_tool)
                logger.info("- Daydream n8n: connected to workflow automation")
            except Exception as e:
                logger.debug(f"Daydream n8n integration not available: {e}")

        # Intuition bridge
        if hasattr(agent, 'intuition_bridge') and agent.intuition_bridge:
            agent.intuition_bridge.attach_erdos(erdos)
            agent.intuition_bridge.attach_daydream(daydream)
            agent.intuition_bridge.attach_learning(pipeline)
            bridge_task = asyncio.create_task(agent.intuition_bridge.start())
            background_tasks.append(bridge_task)
            logger.info("- Intuition Bridge: connected to Erdős, Daydream, Learning")

        # Memory cleanup
        from core.memory_cleanup import (
            create_conversation_cleanup_handler,
            create_cache_cleanup_handler,
            create_file_cleanup_handler,
            create_log_cleanup_handler,
            MemoryPressureLevel,
        )
        from core.memory_watchdog import get_memory_watchdog, register_for_cleanup

        memory_mgr.register_cleanup_handler(create_conversation_cleanup_handler(agent.memory))
        memory_mgr.register_cleanup_handler(create_cache_cleanup_handler(agent.tool_cache))

        uploads_dir = config.get_storage_path() / "uploads"
        memory_mgr.register_cleanup_handler(create_file_cleanup_handler(uploads_dir, max_age_hours=24))

        logs_dir = Path(__file__).parent / "logs"
        memory_mgr.register_cleanup_handler(create_log_cleanup_handler(logs_dir, max_files=10, max_age_days=7))

        async def memory_pressure_response(level, free_mb):
            if level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                logger.warning(f"Memory pressure {level.value}: {free_mb}MB free")

        memory_mgr.register_pressure_callback(memory_pressure_response)
        await memory_mgr.start()
        logger.info("- Memory Manager: periodic cleanup and pressure monitoring")
        
        watchdog = get_memory_watchdog()
        watchdog.set_max_memory(config.hardware.max_cache_mb / 1024 if hasattr(config.hardware, 'max_cache_mb') else 8.0)
        
        if hasattr(agent, 'tool_cache'):
            register_for_cleanup(agent.tool_cache)
        if hasattr(agent, 'memory') and hasattr(agent.memory, 'clear'):
            register_for_cleanup(agent.memory)
        
        try:
            from core.vector_memory import get_vector_memory
            vm = get_vector_memory()
            register_for_cleanup(vm)
        except Exception:
            pass
        
        try:
            from core.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            register_for_cleanup(kg)
        except Exception:
            pass
        
        await watchdog.start()
        logger.info("- Memory Watchdog: aggressive memory monitoring")

        # Runtime integrity monitoring. StartupValidator only runs pre-flight
        # checks once; this watches the files self-edit is forbidden to touch
        # for unexpected modification while Jarvis is running — which matters
        # precisely because Jarvis can rewrite its own code.
        try:
            from core.runtime_integrity import get_integrity_checker

            integrity = get_integrity_checker()
            integrity.start()
            logger.info(
                f"- Runtime Integrity: monitoring {len(integrity._file_hashes)} critical files"
            )
        except Exception as e:
            logger.warning(f"Runtime integrity checker unavailable: {e}")

        # Health monitoring
        from core.health import get_health_manager, HealthCheckResult
        
        health_mgr = get_health_manager()
        
        health_mgr.register("agent", dependencies=[])
        health_mgr.register("llm", dependencies=["agent"])
        health_mgr.register("memory", dependencies=["agent"])
        health_mgr.register("erdos_engine", dependencies=["memory"])
        health_mgr.register("daydream_engine", dependencies=["memory"])
        health_mgr.register("evolution", dependencies=["erdos_engine"])
        health_mgr.register("defense", dependencies=[])
        health_mgr.register("learning", dependencies=["erdos_engine"])
        
        health_mgr.mark_healthy("agent", {"tools": len(agent.tools)})
        health_mgr.mark_healthy("llm", {"model": config.llm.model})
        health_mgr.mark_healthy("memory", {"messages": len(agent.memory.get_messages_for_llm()) if hasattr(agent, 'memory') else 0})
        health_mgr.mark_healthy("erdos_engine", {"problems": len(erdos.problems)})
        health_mgr.mark_healthy("daydream_engine", {"running": daydream._running})
        health_mgr.mark_healthy("evolution", {"genes": len(orchestrator.genes)})
        health_mgr.mark_healthy("defense", {"threats": defense._stats.get('threats_detected', 0)})
        health_mgr.mark_healthy("learning", {"sessions": len(pipeline.sessions)})

        # Embeddings underpin vector memory, the knowledge graph and insight
        # retrieval. A failed model load degrades encode() to zero vectors,
        # which makes every pair perfectly similar — the subsystems then return
        # confident nonsense rather than failing. Report that state instead of
        # leaving it invisible.
        try:
            from core.embedding_interconnect import get_embedding_interconnect

            health_mgr.register("embeddings", dependencies=[])
            emb_health = get_embedding_interconnect().health()
            if emb_health.get("healthy", True):
                health_mgr.mark_healthy("embeddings", emb_health)
            else:
                health_mgr.mark_degraded("embeddings", emb_health.get("message", "degraded"), emb_health)
                logger.error(f"Embeddings degraded at startup: {emb_health.get('message')}")
        except Exception as e:
            logger.warning(f"Could not register embedding health: {e}")

        await health_mgr.start()
        logger.info("- Health Manager: subsystem health monitoring")
        
        # Circuit breakers for LLM resilience
        from core.circuit_breaker import get_circuit_breaker, CircuitConfig
        from core.backend_llm import set_circuit_breakers
        
        primary_breaker = get_circuit_breaker(
            "llm_primary",
            CircuitConfig(failure_threshold=5, timeout_seconds=60.0),
        )
        backend_breaker = get_circuit_breaker(
            "llm_backend",
            CircuitConfig(failure_threshold=3, timeout_seconds=30.0),
        )
        concept_breaker = get_circuit_breaker(
            "llm_concept",
            CircuitConfig(failure_threshold=3, timeout_seconds=60.0),
        )
        
        # Attach circuit breakers to LLM clients
        set_circuit_breakers(
            primary_breaker=primary_breaker,
            backend_breaker=backend_breaker,
            concept_breaker=concept_breaker,
        )
        logger.info("- Circuit Breakers: LLM resilience enabled (primary, backend, concept)")
        
        # Rate limiting
        from core.rate_limiter import get_rate_limiter, RateLimit
        rate_limiter = get_rate_limiter()
        rate_limiter.set_endpoint_limit("/api/chat", RateLimit(requests_per_second=5, requests_per_minute=50))
        rate_limiter.set_endpoint_limit("/ws", RateLimit(requests_per_second=10, requests_per_minute=100))
        logger.info("- Rate Limiter: API rate limiting enabled")
        
        # Audit logging
        from core.audit import get_audit_logger
        audit_logger = get_audit_logger()
        logger.info(f"- Audit Logger: security audit trail enabled ({audit_logger.log_dir})")
        
        # Metrics collection for bounded collections
        from core.metrics import get_metrics_collector
        metrics = get_metrics_collector()
        
        # Register bounded collections for monitoring
        metrics.register("learning_pipeline.sessions", pipeline.sessions, 100)
        metrics.register("defense.integrity_records", defense.integrity_records, 500)
        metrics.register("daydream.exploration_topics", daydream._exploration_topics, 20)
        metrics.register("orchestrator.workflow_history", orchestrator._workflow_history if hasattr(orchestrator, '_workflow_history') else [], 50)
        logger.info("- Metrics: Bounded collection monitoring enabled")

        # Initialize critical singleton modules that may be lazily accessed
        try:
            from core.unified_context_engine import get_context_engine
            context_engine = get_context_engine()
            logger.debug("  ✓ unified_context_engine")
        except Exception as e:
            logger.debug(f"  ✗ unified_context_engine: {e}")

        try:
            from core.event_bus import get_event_bus
            event_bus = get_event_bus()
            logger.debug("  ✓ event_bus")
        except Exception as e:
            logger.debug(f"  ✗ event_bus: {e}")

        try:
            from core.cognitive_load_balancer import get_load_balancer
            load_balancer = get_load_balancer()
            logger.debug("  ✓ cognitive_load_balancer")
        except Exception as e:
            logger.debug(f"  ✗ cognitive_load_balancer: {e}")

        try:
            from core.goal_tracker import get_goal_tracker
            goal_tracker = get_goal_tracker()
            logger.debug("  ✓ goal_tracker")
        except Exception as e:
            logger.debug(f"  ✗ goal_tracker: {e}")

        try:
            from core.self_edit import get_self_edit_engine
            self_edit = get_self_edit_engine()
            logger.debug("  ✓ self_edit_engine")
        except Exception as e:
            logger.debug(f"  ✗ self_edit_engine: {e}")

        try:
            from core.conversation_summarizer import get_conversation_summarizer
            summarizer = get_conversation_summarizer()
            logger.debug("  ✓ conversation_summarizer")
        except Exception as e:
            logger.debug(f"  ✗ conversation_summarizer: {e}")

        try:
            from core.context_compressor import get_context_compressor
            compressor = get_context_compressor()
            logger.debug("  ✓ context_compressor")
        except Exception as e:
            logger.debug(f"  ✗ context_compressor: {e}")

        try:
            from core.memory_protection import get_memory_protection
            mem_protection = get_memory_protection()
            logger.debug("  ✓ memory_protection")
        except Exception as e:
            logger.debug(f"  ✗ memory_protection: {e}")

        logger.info("Critical subsystems initialized")

        logger.info("Background systems started:")
        logger.info("- System Integrator: unified subsystem orchestration")
        logger.info("- State Sync: cross-subsystem state management")
        logger.info("- Observability: metrics and health monitoring")
        logger.info("- Evolution: low-level background (every 4 hours)")
        logger.info("- Daydream: intensive exploration when idle")
        logger.info("- Defense review: every 5 minutes")
        logger.info("- Free Time: sandbox exploration (1/14 of idle time)")

        # Initialize composable workflow system
        try:
            from core.subsystem_stages import init_subsystem_workflows
            from core.workflow_shell import get_workflow_shell
            
            init_subsystem_workflows()
            
            workflow_shell = get_workflow_shell()
            pipeline_count = len(workflow_shell.list_pipelines())
            logger.info(f"- Workflow Shell: {pipeline_count} registered pipelines")
            logger.info("  Registered pipelines: " + ", ".join(workflow_shell.list_pipelines()))
        except Exception as e:
            logger.debug(f"Workflow system initialization: {e}")

        # Initialize multi-model agentic integration
        try:
            from core.multi_model_agentic import init_multi_model_agentic
            
            multi_model_agentic = init_multi_model_agentic(
                primary_llm=agent.llm,
                backend_llm=backend_llm,
                dual_model=dual_model,
            )
            agent.attach_multi_model_agentic(multi_model_agentic)
            logger.info("- Multi-Model Agentic: project work with dual-model routing")
        except Exception as e:
            logger.debug(f"Multi-model agentic initialization: {e}")

    except Exception as e:
        logger.warning(f"Background systems not fully available: {e}")

    # Integration Harness — wires runtime subsystems together
    try:
        from core.integration_harness import start_integration_harness
        harness_results = start_integration_harness()
        harness_healthy = sum(1 for v in harness_results.values() if v == "ok")
        logger.info(f"- Integration Harness: {harness_healthy}/{len(harness_results)} subsystems wired")
    except Exception as e:
        logger.debug(f"Integration harness not available: {e}")

    # Health Router — routes requests based on backend health
    try:
        from core.health_router import get_health_router
        health_router = get_health_router()
        health_router.register_backend("llm_primary", priority=True)
        health_router.register_backend("llm_backend", priority=True)
        logger.info("- Health Router: LLM primary + backend backends registered")
    except Exception as e:
        logger.debug(f"Health router not available: {e}")

    # Agent Memory Store — persistent KV store with TTL
    try:
        from core.agent_memory_store import get_agent_memory_store
        agent_memory = get_agent_memory_store()
        logger.info(f"- Agent Memory Store: {len(agent_memory.list_keys())} keys, {len(agent_memory.list_namespaces())} namespaces")
    except Exception as e:
        logger.debug(f"Agent memory store not available: {e}")

    # Unified Rate Limiter — combines token bucket + backpressure
    try:
        from core.unified_rate_limiter import get_unified_rate_limiter
        from core.rate_limiter import configure_endpoint_limits
        url = get_unified_rate_limiter()
        configured = configure_endpoint_limits(url._limiter)
        logger.info(f"- Unified Rate Limiter: {len(configured)} endpoint limits configured")
    except Exception as e:
        logger.debug(f"Unified rate limiter not available: {e}")

    # Persistent memory
    try:
        from core.persistent_memory import get_persistent_memory
        persistent_memory = get_persistent_memory()
        pm_task = asyncio.create_task(persistent_memory.start())
        background_tasks.append(pm_task)
        logger.info(f"Persistent memory: {len(persistent_memory.long_term_memories)} memories, {len(persistent_memory.user_facts)} facts")
    except Exception as e:
        logger.debug(f"Persistent memory not available: {e}")

    logger.info(f"Registered {len(agent.tools)} tools total")

    # Start supervisor for health monitoring and graceful shutdown
    from core.supervisor import get_supervisor
    supervisor = get_supervisor()
    supervisor.register_subsystem("agent", agent)
    
    async def shutdown_handler():
        logger.info("Supervisor shutdown triggered")
        # Cancel all background tasks
        for task in background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Stop Prefrontal Cortex
        if hasattr(agent, '_prefrontal_cortex') and agent._prefrontal_cortex:
            try:
                await agent._prefrontal_cortex.stop()
                logger.info("  Prefrontal Cortex stopped")
            except Exception as e:
                logger.warning(f"Failed to stop Prefrontal Cortex: {e}")
        # Stop api_manager if started
        if 'api_manager' in dir():
            try:
                await api_manager.stop()
                logger.info("  API Manager stopped")
            except Exception as e:
                logger.debug(f"API Manager stop: {e}")
        # Stop dual_model if started
        if 'dual_model' in dir():
            try:
                await dual_model.stop()
                logger.info("  Dual Model stopped")
            except Exception as e:
                logger.debug(f"Dual Model stop: {e}")
        await agent.stop()
    
    supervisor.register_shutdown_handler(shutdown_handler)
    await supervisor.start()
    logger.info("- Supervisor: health monitoring and auto-recovery enabled")

    # Start n8n if enabled
    n8n_manager = None
    if config.automation.n8n.enabled:
        try:
            from core.n8n_manager import get_n8n_manager
            n8n_manager = get_n8n_manager(port=int(config.automation.n8n.url.split(":")[-1]) if ":" in config.automation.n8n.url else 5678)
            started = await n8n_manager.start()
            if started:
                logger.info(f"- n8n: running on {config.automation.n8n.url}")
            else:
                logger.warning("- n8n: failed to start")
        except Exception as e:
            logger.warning(f"n8n manager not available: {e}")

    # Run interface
    try:
        if args.mcp:
            from core.mcp_server import run_stdio_server
            logger.info("Starting MCP stdio server (external client mode)")
            await run_stdio_server(agent)
            return
        if config.interface.default == "web":
            from interfaces.web_react import WebInterface
            import socket
            import subprocess
            
            # Check SSL certificates
            ssl_dir = Path(__file__).parent / "ssl"
            ssl_keyfile = ssl_dir / "key.pem"
            ssl_certfile = ssl_dir / "cert.pem"
            
            # Default to HTTPS (force_https=True unless --insecure)
            use_https = not args.insecure
            
            # Generate SSL certs if needed - HTTPS is mandatory for security
            if use_https and (not ssl_keyfile.exists() or not ssl_certfile.exists()):
                logger.info("Generating SSL certificates for HTTPS...")
                ssl_dir.mkdir(exist_ok=True)
                try:
                    subprocess.run([
                        "openssl", "req", "-x509", "-newkey", "rsa:2048",
                        "-keyout", str(ssl_keyfile),
                        "-out", str(ssl_certfile),
                        "-days", "365", "-nodes",
                        "-subj", "/CN=localhost"
                    ], check=True, capture_output=True)
                    logger.info("SSL certificates generated successfully")
                except Exception as e:
                    logger.error(f"Failed to generate SSL certificates: {e}")
                    logger.error("HTTPS is required. Install openssl or use existing certs.")
                    logger.error("To generate manually: openssl req -x509 -newkey rsa:2048 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj '/CN=localhost'")
                    raise RuntimeError("SSL certificate generation failed - HTTPS required for security")
            
            port = config.interface.web_port
            protocol = "https" if use_https else "http"
            
            # Get local IP
            local_ip = "localhost"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                pass
            
            # Get Tailscale hostname
            tailscale_host = None
            try:
                result = subprocess.run(
                    ["tailscale", "status", "--json"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    import json
                    ts_data = json.loads(result.stdout)
                    tailscale_host = ts_data.get("Self", {}).get("DNSName", "")
                    if tailscale_host:
                        tailscale_host = tailscale_host.rstrip(".")
            except Exception:
                pass
            
            # Print startup banner - Tailscale is PRIMARY for remote access
            print(f"\n{'='*60}")
            print(f"🚀 Jarvis v3.0 - Secure AI Assistant")
            print(f"{'='*60}")
            print(f"\n📡 Access URLs (HTTPS required):\n")
            if tailscale_host:
                print(f"   ⭐ Tailscale:  https://{tailscale_host}:{port}")
                print(f"      (Primary - accessible from anywhere on your tailnet)")
                print()
            print(f"   Local:        {protocol}://localhost:{port}")
            if local_ip != "localhost":
                print(f"   Local IP:     {protocol}://{local_ip}:{port}")
            print()
            print(f"🔒 HTTPS enabled - Secure connection required")
            if tailscale_host:
                print(f"🌐 Tailscale VPN detected - Remote access configured")
            
            if config.security.require_auth:
                print("🔐 Authentication enabled - set JARVIS_API_KEY environment variable")
            
            print()
            
            web_interface = WebInterface(
                agent, 
                port,
                require_auth=config.security.require_auth,
                force_https=use_https,
                rate_limit_per_minute=config.security.rate_limit_requests_per_minute,
            )
            
            # Connect UI sensing tools to the web interface
            try:
                from tools.ui_sense import set_web_interface as set_ui_sense_interface
                set_ui_sense_interface(web_interface)
                logger.info("UI sensing connected to web interface")
            except Exception as e:
                logger.warning(f"UI sensing connection failed: {e}")
            
            # Connect system sensing tools to the web interface
            try:
                from tools.system_sense import set_web_interface_for_system_sense
                set_web_interface_for_system_sense(web_interface)
                logger.info("System sensing connected to web interface")
            except Exception as e:
                logger.debug(f"System sensing web interface connection: {e}")
            
            await web_interface.run()
        elif config.interface.default == "voice":
            from interfaces.voice import VoiceInterface
            voice = VoiceInterface(agent, use_conversational_speech=args.conversational_speech)
            await voice.start()
        elif config.interface.default == "terminal":
            from interfaces.terminal import TerminalInterface
            terminal = TerminalInterface(agent)
            await terminal.run()
        else:
            from interfaces.cli import CLIInterface
            cli = CLIInterface(agent)
            await cli.start()
    except KeyboardInterrupt:
        logger.info("Shutting down via KeyboardInterrupt...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            supervisor_inst = globals().get('supervisor')
            if supervisor_inst:
                supervisor_inst.record_crash(e)
        except Exception:
            pass
        raise
    finally:
        # Clean up background tasks
        for task in background_tasks:
            if not task.done():
                task.cancel()
        # Stop subsystems
        if n8n_manager:
            try:
                await n8n_manager.stop()
            except Exception as e:
                logger.debug(f"n8n stop error: {e}")
        try:
            await supervisor.shutdown()
        except Exception as e:
            logger.debug(f"Supervisor shutdown error: {e}")
        try:
            await agent.stop()
        except Exception as e:
            logger.debug(f"Agent stop error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
