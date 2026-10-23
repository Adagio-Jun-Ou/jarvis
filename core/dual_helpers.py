"""Integration helpers for dual-model subsystems.

Provides easy integration points for core subsystems to use dual-model routing.
All functions require actual LLM clients - no fake fallback responses.
"""
import logging
import threading

logger = logging.getLogger(__name__)

_dual_integration = None
_dual_lock = threading.Lock()


class DualModelError(Exception):
    """Raised when dual model integration is required but unavailable."""
    pass


def get_integration():
    """Get the dual subsystem integration lazily.
    
    Returns the global instance that was initialized with LLM clients in run.py.
    
    Raises:
        DualModelError: If no integration instance exists (callers must handle gracefully)
    """
    global _dual_integration
    if _dual_integration is None:
        try:
            from core.dual_subsystem import get_dual_subsystem_integration
            _dual_integration = get_dual_subsystem_integration()
            if _dual_integration and _dual_integration.primary_llm:
                logger.debug("Dual subsystem integration retrieved successfully")
                return _dual_integration
            else:
                logger.error("Dual subsystem integration exists but has no LLM clients - integration incomplete")
                raise DualModelError(
                    "Dual subsystem integration incomplete: No LLM clients configured. "
                    "Ensure dual_model.init_dual_model() is called in run.py with valid LLM clients."
                )
        except Exception as e:
            logger.error(f"Dual subsystem integration not available: {e}")
            raise DualModelError(
                f"Dual subsystem integration not available: {e}. "
                f"Check that core/dual_subsystem.py exists and run.py initializes dual_model properly."
            )
    return _dual_integration


def check_integration_available() -> bool:
    """Check if dual model integration is available without raising.
    
    Returns True if integration is ready, False otherwise.
    """
    global _dual_integration
    if _dual_integration is None:
        try:
            from core.dual_subsystem import get_dual_subsystem_integration
            _dual_integration = get_dual_subsystem_integration()
            return _dual_integration is not None and _dual_integration.primary_llm is not None
        except Exception:
            return False
    return _dual_integration is not None and _dual_integration.primary_llm is not None


async def erdos_math_solve(problem: str, context: dict | None = None) -> str:
    """Solve a math problem using the backend model (GLM-5).
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.backend_llm:
        raise DualModelError(
            "Cannot solve math problem: Backend LLM (GLM-5) not configured. "
            "Math solving requires the backend model. Check dual_model initialization."
        )
    
    return await integration.call(
        subsystem="erdos",
        task=SubsystemTask.MATH_REASONING,
        prompt=problem,
        context=context,
        system_prompt="You are solving a mathematical problem. Show your reasoning step by step.",
    )


async def erdos_generate_insights(discovery: str, context: dict | None = None) -> str:
    """Generate insights from a discovery using both models.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm or not integration.backend_llm:
        raise DualModelError(
            "Cannot generate insights: Both Primary and Backend LLMs required. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="erdos",
        task=SubsystemTask.INSIGHT_GENERATION,
        prompt=f"What insights can we derive from this discovery?\n\n{discovery}",
        context=context,
    )


async def daydream_explore(topic: str, context: dict | None = None) -> str:
    """Explore a topic during daydreaming (creative then analytical).
    
    Primary generates ideas, Backend evaluates them.
    This is a SEQUENTIAL collaboration pattern.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm or not integration.backend_llm:
        raise DualModelError(
            "Cannot explore: Both Primary and Backend LLMs required for daydream exploration. "
            "Check dual_model initialization in run.py."
        )
    
    creative_result = await integration.call(
        subsystem="daydream",
        task=SubsystemTask.CREATIVE_EXPLORATION,
        prompt=f"Explore this topic creatively and generate novel ideas:\n\n{topic}",
        context=context,
        system_prompt="You are imagining possibilities. Be creative and generative.",
    )
    evaluation = await integration.call(
        subsystem="daydream",
        task=SubsystemTask.IDEA_EVALUATION,
        prompt=f"Evaluate these ideas for feasibility and potential:\n\n{creative_result}",
        context=context,
        system_prompt="You are evaluating ideas critically. Be analytical but fair.",
    )
    integration.track_sequential("daydream", SubsystemTask.CREATIVE_EXPLORATION, SubsystemTask.IDEA_EVALUATION)
    return f"Ideas:\n{creative_result}\n\nEvaluation:\n{evaluation}"


async def evolution_propose(area: str, context: dict | None = None) -> str:
    """Generate evolution proposal using primary (creative) model.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm:
        raise DualModelError(
            "Cannot propose evolution: Primary LLM not configured. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="evolution",
        task=SubsystemTask.PROPOSAL_GENERATION,
        prompt=f"Propose a creative improvement for: {area}",
        context=context,
        system_prompt="You are proposing improvements. Be creative and forward-thinking.",
    )


async def evolution_analyze(proposal: str, context: dict | None = None) -> str:
    """Analyze impact of proposal using backend (analytical) model.
    
    Call after evolution_propose for sequential collaboration pattern.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.backend_llm:
        raise DualModelError(
            "Cannot analyze evolution: Backend LLM not configured. "
            "Check dual_model initialization in run.py."
        )
    
    result = await integration.call(
        subsystem="evolution",
        task=SubsystemTask.IMPACT_ANALYSIS,
        prompt=f"Analyze the impact and risks of this proposal:\n\n{proposal}",
        context=context,
        system_prompt="You are analyzing proposals critically. Be thorough and identify risks.",
    )
    integration.track_sequential("evolution", SubsystemTask.PROPOSAL_GENERATION, SubsystemTask.IMPACT_ANALYSIS)
    return result


async def self_refine_analyze(code: str, context: dict | None = None) -> str:
    """Analyze code using backend model.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.backend_llm:
        raise DualModelError(
            "Cannot analyze code: Backend LLM not configured. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="self_refine",
        task=SubsystemTask.CODE_ANALYSIS,
        prompt=code,
        context=context,
        system_prompt="You are analyzing code. Identify issues, bugs, and potential improvements.",
    )


async def self_refine_fix(issue: str, context: dict | None = None) -> str:
    """Generate fix suggestions using both models (parallel).
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm or not integration.backend_llm:
        raise DualModelError(
            "Cannot generate fix: Both Primary and Backend LLMs required. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="self_refine",
        task=SubsystemTask.FIX_GENERATION,
        prompt=f"Suggest fixes for:\n\n{issue}",
        context=context,
    )


async def insight_extract(data: str, context: dict | None = None) -> str:
    """Extract patterns from data using both models.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm or not integration.backend_llm:
        raise DualModelError(
            "Cannot extract insights: Both Primary and Backend LLMs required. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="insight_bridge",
        task=SubsystemTask.PATTERN_EXTRACTION,
        prompt=f"Extract patterns and insights from:\n\n{data}",
        context=context,
    )


async def insight_guidance(pattern: str, context: dict | None = None) -> str:
    """Generate evolution guidance from pattern using backend.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.backend_llm:
        raise DualModelError(
            "Cannot generate guidance: Backend LLM not configured. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="insight_bridge",
        task=SubsystemTask.EVOLUTION_GUIDANCE,
        prompt=f"How should this pattern guide system evolution?\n\n{pattern}",
        context=context,
        system_prompt="You are an expert at translating insights into actionable guidance.",
    )


async def learning_discover(experience: str, context: dict | None = None) -> str:
    """Discover patterns from experience using both models.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm or not integration.backend_llm:
        raise DualModelError(
            "Cannot discover patterns: Both Primary and Backend LLMs required. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="learning",
        task=SubsystemTask.PATTERN_DISCOVERY,
        prompt=f"What patterns can we learn from this experience?\n\n{experience}",
        context=context,
    )


async def learning_evaluate(strategy: str, context: dict | None = None) -> str:
    """Evaluate strategy effectiveness using backend.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.backend_llm:
        raise DualModelError(
            "Cannot evaluate strategy: Backend LLM not configured. "
            "Check dual_model initialization in run.py."
        )
    
    result = await integration.call(
        subsystem="learning",
        task=SubsystemTask.STRATEGY_EVALUATION,
        prompt=f"Evaluate this strategy's effectiveness:\n\n{strategy}",
        context=context,
        system_prompt="You are evaluating strategies. Be critical and thorough.",
    )
    integration.track_sequential("learning", SubsystemTask.PATTERN_DISCOVERY, SubsystemTask.STRATEGY_EVALUATION)
    return result


async def proactive_curious(topic: str, context: dict | None = None) -> str:
    """Generate curiosity-driven exploration using primary.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm:
        raise DualModelError(
            "Cannot generate curiosity: Primary LLM not configured. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="proactive",
        task=SubsystemTask.CURIOSITY_GENERATION,
        prompt=f"What should we explore about: {topic}?",
        context=context,
        system_prompt="You are curious and creative. Suggest interesting directions to explore.",
    )


async def proactive_reflect(context_data: str, context: dict | None = None) -> str:
    """Self-reflection using both models.
    
    Raises:
        DualModelError: If dual model integration is not available
    """
    from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
    
    integration = get_dual_subsystem_integration()
    if not integration or not integration.primary_llm or not integration.backend_llm:
        raise DualModelError(
            "Cannot reflect: Both Primary and Backend LLMs required. "
            "Check dual_model initialization in run.py."
        )
    
    return await integration.call(
        subsystem="proactive",
        task=SubsystemTask.SELF_REFLECTION,
        prompt=f"Reflect on this context:\n\n{context_data}",
        context=context,
    )
