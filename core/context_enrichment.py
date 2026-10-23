"""Context Enrichment - Enriches Jarvis context with live subsystem state.

This module provides functions to gather state from all background subsystems
and enrich the context passed to the LLM, making Jarvis feel like a "whole being"
with continuous awareness of its activities.
"""
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_last_enrichment_time = 0.0
_enrichment_cache: dict[str, Any] = {}
_cache_ttl = 10.0  # Cache subsystem state for 10 seconds
_cache_lock = threading.Lock()


def get_daydream_state() -> dict[str, Any]:
    """Get current state from daydream engine.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.daydream_engine import get_daydream_engine
        engine = get_daydream_engine()
        if not engine:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        status = engine.get_status()
        return {
            "status": "active" if status.get("running") and status.get("can_daydream") else "idle",
            "activity": status.get("current_activity", ""),
            "discoveries": status.get("exploration_topics", [])[:3],
            "metrics": {
                "daydream_count": status.get("daydream_count", 0),
                "insights_generated": status.get("insights_generated", 0),
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get daydream state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_erdos_state() -> dict[str, Any]:
    """Get current state from Erdős engine.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.erdos_engine import get_erdos_engine
        engine = get_erdos_engine()
        if not engine:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        status = engine.get_status()
        running = status.get("running", False)
        return {
            "status": "active" if running else "idle",
            "activity": "solving problems" if running else "",
            "discoveries": status.get("recent_solutions", [])[:3],
            "metrics": {
                "problems_attempted": status.get("total_attempted", 0),
                "problems_solved": status.get("solved", 0),
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get Erdős state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_music_state() -> dict[str, Any]:
    """Get current state from music creation engine.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.music_creation_engine import get_music_creation_engine
        engine = get_music_creation_engine()
        if not engine:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        stats = engine.get_stats()
        running = stats.get("running", False)
        
        activity = ""
        if running and hasattr(engine, '_current_composition') and engine._current_composition:
            comp = engine._current_composition
            activity = f"Composing {comp.title}"
        
        recent_comps = stats.get("recent_compositions", [])[:3]
        discoveries = []
        for comp in recent_comps:
            if isinstance(comp, dict):
                discoveries.append(f"{comp.get('title', 'composition')} ({comp.get('style', '?')})")
        
        return {
            "status": "creating" if running else "idle",
            "activity": activity,
            "discoveries": discoveries,
            "metrics": {
                "compositions_created": stats.get("compositions_created", 0),
                "tokens_observed": stats.get("tokens_observed", 0),
                "current_style": stats.get("current_style"),
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get music state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_learning_state() -> dict[str, Any]:
    """Get current state from learning pipeline.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.learning_pipeline import get_learning_pipeline
        pipeline = get_learning_pipeline()
        if not pipeline:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        running = getattr(pipeline, '_running', False)
        generation = getattr(pipeline, 'generation', 0)
        phase = getattr(pipeline, '_current_phase', None)
        
        activity = ""
        if running and phase:
            activity = f"learning phase: {phase.value if hasattr(phase, 'value') else str(phase)}"
        
        return {
            "status": "learning" if running else "idle",
            "activity": activity,
            "discoveries": [f"Generation {generation}"] if generation > 0 else [],
            "metrics": {
                "generation": generation,
                "running": running,
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get learning state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_evolution_state() -> dict[str, Any]:
    """Get current state from evolution orchestrator.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.evolution_orchestrator import get_evolution_orchestrator
        orchestrator = get_evolution_orchestrator()
        if not orchestrator:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        running = getattr(orchestrator, '_running', False)
        count = getattr(orchestrator, 'evolution_count', 0)
        
        return {
            "status": "evolving" if running else "idle",
            "activity": "self-improvement" if running else "",
            "discoveries": [f"{count} evolutions applied"] if count > 0 else [],
            "metrics": {
                "evolution_count": count,
                "running": running,
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get evolution state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_intuition_state() -> dict[str, Any]:
    """Get current state from intuition bridge.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.intuition_bridge import get_intuition_bridge
        bridge = get_intuition_bridge()
        if not bridge:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        stats = bridge.get_stats()
        total = stats.get('total_insights', 0)
        recent = stats.get('recent_insights', [])[:3]
        
        return {
            "status": "active",
            "activity": "processing insights",
            "discoveries": recent,
            "metrics": {
                "total_insights": total,
                "pending_actions": stats.get('pending_actions', 0),
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get intuition state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_pfc_state() -> dict[str, Any]:
    """Get current state from prefrontal cortex.
    
    Returns:
        Dict with 'status', 'activity', 'discoveries', 'metrics'
    """
    try:
        from core.prefrontal_cortex import get_prefrontal_cortex
        pfc = get_prefrontal_cortex()
        if not pfc:
            return {"status": "unavailable", "activity": "", "discoveries": []}
        
        running = getattr(pfc, '_running', False)
        concepts = getattr(pfc, '_concepts', [])
        
        return {
            "status": "active" if running else "idle",
            "activity": "observing concepts" if running else "",
            "discoveries": [f"{len(concepts)} concepts tracked"] if concepts else [],
            "metrics": {
                "concept_count": len(concepts) if concepts else 0,
                "running": running,
            }
        }
    except Exception as e:
        logger.debug(f"Failed to get PFC state: {e}")
        return {"status": "error", "activity": "", "discoveries": []}


def get_all_subsystem_states(use_cache: bool = True) -> dict[str, dict]:
    """Get states from all subsystems.
    
    Args:
        use_cache: Whether to use cached values (default True for performance)
        
    Returns:
        Dict mapping subsystem names to their states
    """
    global _last_enrichment_time, _enrichment_cache
    
    # Check cache and return copy while holding lock
    if use_cache:
        with _cache_lock:
            if time.time() - _last_enrichment_time < _cache_ttl:
                return _enrichment_cache.copy()
    
    # Gather fresh states
    states = {
        "daydream": get_daydream_state(),
        "erdos": get_erdos_state(),
        "music": get_music_state(),
        "learning": get_learning_state(),
        "evolution": get_evolution_state(),
        "intuition": get_intuition_state(),
        "pfc": get_pfc_state(),
    }
    
    # Update cache
    with _cache_lock:
        _enrichment_cache = states.copy()
        _last_enrichment_time = time.time()
    
    return states


def get_active_subsystems() -> list[tuple[str, str]]:
    """Get list of currently active subsystems with their activities.
    
    Returns:
        List of (name, activity) tuples for active subsystems
    """
    states = get_all_subsystem_states(use_cache=True)
    active = []
    
    for name, state in states.items():
        status = state.get("status", "idle")
        if status not in ("idle", "unavailable", "error"):
            activity = state.get("activity", status)
            if activity:
                active.append((name, activity))
    
    return active


def get_recent_discoveries(limit: int = 5) -> list[str]:
    """Get recent discoveries from all subsystems.
    
    Args:
        limit: Maximum discoveries to return
        
    Returns:
        List of discovery strings
    """
    states = get_all_subsystem_states(use_cache=True)
    discoveries = []
    
    for name, state in states.items():
        for discovery in state.get("discoveries", []):
            if discovery and isinstance(discovery, str):
                discoveries.append(f"[{name}] {discovery}")
    
    return discoveries[:limit]


def get_subsystem_activity_summary() -> str:
    """Get a brief summary of what subsystems are doing.
    
    Returns:
        Human-readable summary string
    """
    active = get_active_subsystems()
    
    if not active:
        return "idle"
    
    parts = []
    for name, activity in active[:3]:
        parts.append(f"{name}: {activity[:30]}")
    
    return "; ".join(parts)


def enrich_context_with_subsystems(base_context: str) -> str:
    """Enrich a context string with subsystem state.
    
    Args:
        base_context: The base context to enrich
        
    Returns:
        Context with subsystem information appended
    """
    get_all_subsystem_states(use_cache=True)
    active = get_active_subsystems()
    discoveries = get_recent_discoveries(limit=5)
    
    if not active and not discoveries:
        return base_context
    
    parts = [base_context]
    
    # Active subsystems
    if active:
        parts.append("\n<subsystem_state>")
        for name, activity in active:
            parts.append(f"  {name}: {activity}")
        parts.append("</subsystem_state>")
    
    # Recent discoveries
    if discoveries:
        parts.append("\n<recent_discoveries>")
        for discovery in discoveries:
            parts.append(f"  {discovery}")
        parts.append("</recent_discoveries>")
    
    return "\n".join(parts)


def get_memory_context(query: str = "", limit: int = 5) -> str:
    """Get relevant context from all memory systems.
    
    Args:
        query: Query to find relevant memories
        limit: Maximum items per memory type
        
    Returns:
        Formatted memory context
    """
    parts = []
    
    # Persistent memory - user facts
    try:
        from core.persistent_memory import get_persistent_memory
        pm = get_persistent_memory()
        if pm:
            facts = pm.get_user_facts()
            if facts:
                parts.append("<user_context>")
                for key, value in list(facts.items())[:limit]:
                    if isinstance(value, str) and len(value) < 100:
                        parts.append(f"  {key}: {value}")
                parts.append("</user_context>")
    except Exception as e:
        logger.debug(f"Failed to get persistent memory: {e}")
    
    # Knowledge graph - related concepts
    try:
        from core.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if kg and query:
            related = kg.get_related_context(query, limit=limit)
            if related:
                parts.append("<related_knowledge>")
                for item in related[:limit]:
                    if isinstance(item, str) and len(item) < 200:
                        parts.append(f"  {item}")
                parts.append("</related_knowledge>")
    except Exception as e:
        logger.debug(f"Failed to get knowledge graph: {e}")
    
    # Intuition insights - learned patterns
    try:
        from core.intuition_bridge import get_intuition_bridge
        bridge = get_intuition_bridge()
        if bridge:
            insights = bridge.get_top_insights(limit=limit)
            if insights:
                parts.append("<learned_patterns>")
                for insight in insights:
                    if isinstance(insight, str) and len(insight) < 200:
                        parts.append(f"  {insight}")
                parts.append("</learned_patterns>")
    except Exception as e:
        logger.debug(f"Failed to get intuition insights: {e}")
    
    # Unified memory - recent entries
    try:
        from core.unified_memory import get_unified_memory, MemoryType
        um = get_unified_memory()
        if um:
            recent = um.get_recent(MemoryType.INSIGHT, limit=limit)
            if recent:
                parts.append("<recent_insights>")
                for entry in recent:
                    if hasattr(entry, 'content') and len(entry.content) < 200:
                        parts.append(f"  {entry.content[:150]}")
                parts.append("</recent_insights>")
    except Exception as e:
        logger.debug(f"Failed to get unified memory: {e}")
    
    return "\n".join(parts) if parts else ""


def format_live_context_for_prompt(
    agent: Any,
    user_message: str = ""
) -> str:
    """Format live context for injection into responses.
    
    This creates a natural context block that helps Jarvis respond
    with awareness of what it's been doing, learning, and thinking.
    
    Args:
        agent: JarvisAgent instance
        user_message: The user's message (for relevance)
        
    Returns:
        Formatted context string
    """
    from core.live_state import get_live_state
    from core.narrative_thread import get_narrative_thread
    
    parts = []
    
    # Get live state
    live_state = get_live_state()
    live_state.update_from_agent(agent)
    live_state.update_from_subsystems()
    live_state.update_discoveries()
    
    # Get current activity
    state = live_state.get_state()
    
    # Current activity block
    if state.current_activity:
        activity_block = f"<current_activity>{state.current_activity}</current_activity>"
        parts.append(activity_block)
    
    # Recent discoveries from background
    discoveries = []
    for comp in state.discoveries.music_compositions[:2]:
        if comp:
            discoveries.append(f"Composed: {comp}")
    for insight in state.discoveries.math_insights[:1]:
        if insight:
            discoveries.append(f"Solved: {insight[:50]}")
    for thought in state.discoveries.daydream_thoughts[:2]:
        if thought:
            discoveries.append(f"Exploring: {thought[:50]}")
    
    if discoveries:
        parts.append(f"<recent_work>{' | '.join(discoveries[:3])}</recent_work>")
    
    # Narrative thread
    narrative = get_narrative_thread()
    narrative_context = narrative.get_context_string(include_last_session=True)
    if narrative_context:
        parts.append(narrative_context)
    
    # Memory context relevant to query
    if user_message:
        memory_ctx = get_memory_context(user_message, limit=3)
        if memory_ctx:
            parts.append(memory_ctx)
    
    # Subsystem summary (only if interesting activity)
    summary = get_subsystem_activity_summary()
    if summary and summary != "idle":
        parts.append(f"<subsystems>{summary}</subsystems>")
    
    return "\n".join(parts) if parts else ""


def get_emotional_context_for_prompt(user_message: str = "") -> str:
    """Get emotional context from emotional state manager.
    
    Args:
        user_message: The user's message for context
        
    Returns:
        Emotional context string for prompts
    """
    try:
        from core.emotional_state import get_emotional_state_manager
        manager = get_emotional_state_manager()
        
        # Analyze the message if provided
        if user_message:
            manager.analyze_input(user_message)
        
        return manager.get_emotional_context()
    except Exception as e:
        logger.debug(f"Failed to get emotional context: {e}")
        return ""


def build_rich_context(
    agent: Any,
    include_subsystems: bool = True,
    include_discoveries: bool = True,
    include_memory: bool = True,
    include_narrative: bool = True,
    include_user_model: bool = True,
    include_goals: bool = True,
    include_proactive: bool = True,
    include_reflection: bool = True,
    include_personality: bool = True,
    query: str = ""
) -> str:
    """Build rich context combining all available state.
    
    This is the main entry point for context enrichment.
    
    Args:
        agent: JarvisAgent instance
        include_subsystems: Include subsystem activity state
        include_discoveries: Include recent discoveries
        include_memory: Include memory context
        include_narrative: Include narrative thread
        include_user_model: Include user preferences and context
        include_goals: Include active goals
        include_proactive: Include proactive context
        include_reflection: Include reflection insights
        include_personality: Include personality traits
        query: Query context for relevant memory retrieval
        
    Returns:
        Rich context string for LLM prompts
    """
    from core.live_state import get_live_state
    from core.user_model import get_user_model_manager
    from core.goal_tracker import get_goal_tracker
    from core.proactive import get_proactive_suggester
    from core.self_reflection import get_self_reflection
    from core.personality import get_personality_system
    
    live_state = get_live_state()
    
    # Start with base context
    parts = []
    
    # Static identity from agent
    if hasattr(agent, 'prompt_manager') and agent.prompt_manager:
        parts.append(agent.prompt_manager.get_base_prompt())
    else:
        parts.append("You are Jarvis, an AI assistant.")
    
    # Live state
    if include_subsystems or include_discoveries:
        live_state.update_from_agent(agent)
        live_state.update_from_subsystems()
        live_state.update_discoveries()
        
        state_context = live_state.get_context_string()
        if state_context:
            parts.append(state_context)
    
    # Memory context
    if include_memory:
        memory_context = get_memory_context(query)
        if memory_context:
            parts.append(memory_context)
    
    # Narrative thread (from live state)
    if include_narrative:
        narrative = live_state.get_state().narrative
        if narrative.current_focus:
            parts.append(f"\n<focus>{narrative.current_focus}</focus>")
    
    # User model context
    if include_user_model:
        try:
            user_manager = get_user_model_manager()
            user_context = user_manager.get_user_context()
            if user_context:
                parts.append(user_context)
        except Exception as e:
            logger.debug(f"Failed to get user context: {e}")
    
    # Goals context
    if include_goals:
        try:
            goal_tracker = get_goal_tracker()
            goal_context = goal_tracker.get_goal_context()
            if goal_context:
                parts.append(goal_context)
        except Exception as e:
            logger.debug(f"Failed to get goal context: {e}")
    
    # Proactive context
    if include_proactive:
        try:
            proactive_engine = get_proactive_suggester()
            proactive_context = proactive_engine.get_context_for_prompt()
            if proactive_context:
                parts.append(proactive_context)
        except Exception as e:
            logger.debug(f"Failed to get proactive context: {e}")
    
    # Reflection context
    if include_reflection:
        try:
            reflection = get_self_reflection()
            reflection_context = reflection.get_reflection_context()
            if reflection_context:
                parts.append(reflection_context)
        except Exception as e:
            logger.debug(f"Failed to get reflection context: {e}")
    
    # Personality context
    if include_personality:
        try:
            personality = get_personality_system()
            personality_context = personality.get_trait_summary()
            if personality_context:
                parts.append(personality_context)
        except Exception as e:
            logger.debug(f"Failed to get personality context: {e}")
    
    return "\n\n".join(parts)


def unified_memory_search(query: str, limit: int = 5) -> list[dict]:
    """Search across all memory systems at once.
    
    This provides a unified query interface that searches:
    - Persistent memory (user facts)
    - Knowledge graph (semantic concepts)
    - Unified memory (recent insights)
    - Intuition bridge (learned patterns)
    
    Args:
        query: Search query
        limit: Maximum results per system
        
    Returns:
        List of search results with source and relevance
    """
    results = []
    
    # Persistent memory
    try:
        from core.persistent_memory import get_persistent_memory
        pm = get_persistent_memory()
        if pm:
            memories = pm.get_relevant_memories(query, limit=limit)
            for mem in memories:
                results.append({
                    "source": "persistent",
                    "content": mem.get("content", str(mem)),
                    "relevance": mem.get("relevance", 0.5),
                    "timestamp": mem.get("timestamp"),
                })
    except Exception as e:
        logger.debug(f"Persistent memory search failed: {e}")
    
    # Knowledge graph
    try:
        from core.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if kg and query:
            related = kg.get_related_context(query, limit=limit)
            for item in related:
                if isinstance(item, str):
                    results.append({
                        "source": "knowledge_graph",
                        "content": item,
                        "relevance": 0.6,
                    })
    except Exception as e:
        logger.debug(f"Knowledge graph search failed: {e}")
    
    # Unified memory
    try:
        from core.unified_memory import get_unified_memory, MemoryType
        um = get_unified_memory()
        if um:
            recent = um.get_recent(MemoryType.INSIGHT, limit=limit)
            for entry in recent:
                if hasattr(entry, 'content'):
                    # Simple relevance check
                    relevance = 0.5
                    if query.lower() in entry.content.lower():
                        relevance = 0.8
                    results.append({
                        "source": "unified_memory",
                        "content": entry.content[:200],
                        "relevance": relevance,
                        "timestamp": entry.timestamp if hasattr(entry, 'timestamp') else None,
                    })
    except Exception as e:
        logger.debug(f"Unified memory search failed: {e}")
    
    # Intuition bridge
    try:
        from core.intuition_bridge import get_intuition_bridge
        ib = get_intuition_bridge()
        if ib:
            insights = ib.get_top_insights(limit=limit)
            for insight in insights:
                relevance = 0.5
                if query.lower() in insight.lower():
                    relevance = 0.7
                results.append({
                    "source": "intuition",
                    "content": insight[:200] if len(insight) > 200 else insight,
                    "relevance": relevance,
                })
    except Exception as e:
        logger.debug(f"Intuition bridge search failed: {e}")
    
    # Sort by relevance
    results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    
    return results[:limit * 2]


def format_unified_results(results: list[dict], max_items: int = 5) -> str:
    """Format unified search results for context.
    
    Args:
        results: Search results from unified_memory_search
        max_items: Maximum items to include
        
    Returns:
        Formatted context string
    """
    if not results:
        return ""
    
    parts = ["<relevant_context>"]
    
    for result in results[:max_items]:
        source = result.get("source", "unknown")
        content = result.get("content", "")
        if content:
            parts.append(f"  [{source}] {content[:150]}")
    
    parts.append("</relevant_context>")
    
    return "\n".join(parts)