"""Live State - Unified real-time state for Jarvis.

This module provides a single source of truth for all of Jarvis's current state,
aggregating information from subsystems, memories, and background processes.

The LiveState serves as the "I am right now" for Jarvis - what it's doing,
thinking, learning, and remembering - enabling a unified sense of self.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

_singleton: Optional["LiveState"] = None
_lock = threading.Lock()


class SubsystemStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    THINKING = "thinking"
    CREATING = "creating"
    EXPLORING = "exploring"
    LEARNING = "learning"


@dataclass
class SubsystemState:
    """State of a single subsystem."""
    name: str
    status: SubsystemStatus
    current_activity: str
    recent_discoveries: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)


@dataclass
class MemoryState:
    """Aggregated memory state."""
    conversation_turns: int = 0
    persistent_facts: int = 0
    knowledge_nodes: int = 0
    recent_memories: list[str] = field(default_factory=list)
    relevant_memories: list[str] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundDiscoveries:
    """Recent discoveries from background systems."""
    math_insights: list[str] = field(default_factory=list)
    music_compositions: list[str] = field(default_factory=list)
    daydream_thoughts: list[str] = field(default_factory=list)
    learning_patterns: list[str] = field(default_factory=list)
    evolution_improvements: list[str] = field(default_factory=list)


@dataclass
class NarrativeState:
    """Continuous identity thread."""
    current_focus: str = ""
    recent_progress: list[str] = field(default_factory=list)
    learning_trajectory: str = ""
    sessions_count: int = 0
    last_session_summary: str = ""
    session_id: str = ""  # Track current session to avoid double counting


@dataclass
class LiveStateData:
    """Complete live state snapshot."""
    # Who I am
    identity: str = "Jarvis"
    model: str = "kimi-k2.5"
    capabilities: list[str] = field(default_factory=list)
    
    # What I'm doing now
    current_activity: str = ""
    current_subsystem: str = ""
    thinking_about: str = ""
    
    # Subsystem states
    subsystems: dict[str, SubsystemState] = field(default_factory=dict)
    
    # Memory state
    memory: MemoryState = field(default_factory=MemoryState)
    
    # Background discoveries
    discoveries: BackgroundDiscoveries = field(default_factory=BackgroundDiscoveries)
    
    # Narrative thread
    narrative: NarrativeState = field(default_factory=NarrativeState)
    
    # Emotional/contextual state
    mood: str = "engaged"
    focus_areas: list[str] = field(default_factory=list)
    
    # Timestamp
    updated_at: float = field(default_factory=time.time)


class LiveState:
    """Unified live state aggregator for Jarvis.
    
    This class collects state from all subsystems, memories, and background
    processes to provide a single "I am right now" view of Jarvis.
    
    Usage:
        live_state = get_live_state()
        live_state.update_from_agent(agent)
        live_state.update_from_subsystems()
        
        # Get context for prompts
        context = live_state.get_context_string()
    """
    
    def __init__(self):
        self._data = LiveStateData()
        self._subsystem_getters: dict[str, callable] = {}
        self._memory_getters: dict[str, callable] = {}
        self._last_discovery_update = 0.0
        self._discovery_update_interval = 30.0  # Update discoveries every 30s
        self._lock = threading.RLock()
        
    def register_subsystem_getter(self, name: str, getter: callable) -> None:
        """Register a function that returns subsystem state.
        
        Args:
            name: Subsystem name (e.g., 'daydream', 'erdos', 'music')
            getter: Function that returns dict with 'status', 'activity', 'discoveries'
        """
        self._subsystem_getters[name] = getter
        
    def register_memory_getter(self, name: str, getter: callable) -> None:
        """Register a function that returns memory state.
        
        Args:
            name: Memory type (e.g., 'persistent', 'knowledge', 'vector')
            getter: Function that returns memory data
        """
        self._memory_getters[name] = getter
    
    def update_from_agent(self, agent: Any) -> None:
        """Update state from the main agent.
        
        Args:
            agent: JarvisAgent instance
        """
        with self._lock:
            # Basic identity
            self._data.identity = getattr(agent.config, 'identity', {}).get('name', 'Jarvis')
            self._data.model = getattr(agent.llm, 'model', 'kimi-k2.5')
            
            # Capabilities
            tools = getattr(agent, 'tools', {})
            self._data.capabilities = list(tools.keys())[:20]  # Top 20 tools
            
            # Current thinking
            if hasattr(agent, 'thinking_engine') and agent.thinking_engine:
                thoughts = getattr(agent.thinking_engine, 'current_thoughts', [])
                if thoughts:
                    self._data.thinking_about = thoughts[-1] if thoughts else ""
            
            # Memory state
            if hasattr(agent, 'memory'):
                history = agent.memory.get_history() if hasattr(agent.memory, 'get_history') else []
                self._data.memory.conversation_turns = len(history)
            
            # Session info - only increment if this is a new session
            if hasattr(agent, '_session_id'):
                if self._data.narrative.session_id != agent._session_id:
                    self._data.narrative.session_id = agent._session_id
                    self._data.narrative.sessions_count += 1
    
    def update_from_subsystems(self) -> None:
        """Update state from all registered subsystems."""
        with self._lock:
            for name, getter in self._subsystem_getters.items():
                try:
                    state = getter()
                    if state:
                        subsystem_state = SubsystemState(
                            name=name,
                            status=SubsystemStatus(state.get('status', 'idle')),
                            current_activity=state.get('activity', ''),
                            recent_discoveries=state.get('discoveries', [])[:3],
                            metrics=state.get('metrics', {}),
                        )
                        self._data.subsystems[name] = subsystem_state
                except Exception as e:
                    logger.debug(f"Failed to get state from {name}: {e}")
            
            # Aggregate current activity
            active = [s for s in self._data.subsystems.values() 
                     if s.status != SubsystemStatus.IDLE]
            if active:
                # Find the most recently updated active subsystem
                most_recent = max(active, key=lambda s: s.last_update)
                self._data.current_subsystem = most_recent.name
                self._data.current_activity = most_recent.current_activity
    
    def update_from_memories(self, query: str = "") -> None:
        """Update state from memory systems.
        
        Args:
            query: Optional query to find relevant memories
        """
        with self._lock:
            for name, getter in self._memory_getters.items():
                try:
                    data = getter(query) if query else getter()
                    if data:
                        if name == 'persistent':
                            self._data.memory.persistent_facts = data.get('fact_count', 0)
                            self._data.memory.user_preferences = data.get('user_preferences', {})
                            self._data.memory.recent_memories = data.get('recent', [])[:5]
                        elif name == 'knowledge':
                            self._data.memory.knowledge_nodes = data.get('node_count', 0)
                            self._data.memory.relevant_memories = data.get('related', [])[:5]
                except Exception as e:
                    logger.debug(f"Failed to get memory state from {name}: {e}")
    
    def update_discoveries(self) -> None:
        """Update background discoveries from all sources."""
        with self._lock:
            now = time.time()
            if now - self._last_discovery_update < self._discovery_update_interval:
                return
            
            self._last_discovery_update = now
            
            # Math insights from Erdős
            try:
                from core.erdos_engine import get_erdos_engine
                erdos = get_erdos_engine()
                if erdos:
                    status = erdos.get_status()
                    if status.get('running'):
                        self._data.discoveries.math_insights = status.get('recent_solutions', [])[:3]
            except Exception as e:
                logger.debug(f"Failed to get Erdős state: {e}")
            
            # Music compositions
            try:
                from core.music_creation_engine import get_music_creation_engine
                music = get_music_creation_engine()
                if music:
                    stats = music.get_stats()
                    if stats.get('running'):
                        recent = stats.get('recent_compositions', [])[:3]
                        self._data.discoveries.music_compositions = [
                            f"{c.get('title', 'composition')} in {c.get('key', '?')}"
                            for c in recent if isinstance(c, dict)
                        ]
                        # Set current activity if composing
                        if music._current_composition:
                            self._data.current_activity = f"Composing {music._current_composition.title}"
                            self._data.current_subsystem = "music"
            except Exception as e:
                logger.debug(f"Failed to get music state: {e}")
            
            # Daydream thoughts
            try:
                from core.daydream_engine import get_daydream_engine
                daydream = get_daydream_engine()
                if daydream:
                    status = daydream.get_status()
                    if status.get('running'):
                        topics = status.get('exploration_topics', [])[:3]
                        if topics:
                            self._data.discoveries.daydream_thoughts = topics
                            if status.get('can_daydream'):
                                activity = status.get('current_activity', 'exploring')
                                self._data.current_activity = f"Daydreaming: {activity}"
                                self._data.current_subsystem = "daydream"
            except Exception as e:
                logger.debug(f"Failed to get daydream state: {e}")
            
            # Learning patterns
            try:
                from core.learning_pipeline import get_learning_pipeline
                learning = get_learning_pipeline()
                if learning:
                    # Learning pipeline has generation counter
                    gen = getattr(learning, 'generation', 0)
                    if gen > 0:
                        self._data.discoveries.learning_patterns = [f"Generation {gen}"]
            except Exception as e:
                logger.debug(f"Failed to get learning state: {e}")
            
            # Evolution improvements
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                evolution = get_evolution_orchestrator()
                if evolution:
                    count = getattr(evolution, 'evolution_count', 0)
                    if count > 0:
                        self._data.discoveries.evolution_improvements = [f"{count} evolutions"]
            except Exception as e:
                logger.debug(f"Failed to get evolution state: {e}")
            
            # Intuition insights
            try:
                from core.intuition_bridge import get_intuition_bridge
                bridge = get_intuition_bridge()
                if bridge:
                    stats = bridge.get_stats()
                    stats.get('total_insights', 0)
                    recent = stats.get('recent_insights', [])[:3]
                    if recent:
                        self._data.discoveries.learning_patterns.extend(recent)
            except Exception as e:
                logger.debug(f"Failed to get intuition state: {e}")
    
    def update_narrative(self, focus: str = "", progress: str = "") -> None:
        """Update the narrative thread.
        
        Args:
            focus: Current focus area
            progress: Recent progress
        """
        with self._lock:
            if focus:
                self._data.narrative.current_focus = focus
            if progress:
                self._data.narrative.recent_progress.append(progress)
                if len(self._data.narrative.recent_progress) > 5:
                    self._data.narrative.recent_progress = self._data.narrative.recent_progress[-5:]
    
    def set_mood(self, mood: str, focus_areas: list[str] = None) -> None:
        """Set current mood and focus areas.
        
        Args:
            mood: Current mood (engaged, curious, focused, creative, etc.)
            focus_areas: Topics currently focused on
        """
        with self._lock:
            self._data.mood = mood
            if focus_areas:
                self._data.focus_areas = focus_areas
            self._data.updated_at = time.time()
    
    def get_state(self) -> LiveStateData:
        """Get current state snapshot.
        
        Returns:
            LiveStateData with current state
        """
        with self._lock:
            return self._data
    
    def get_context_string(self, max_items: int = 5) -> str:
        """Get state formatted for context injection into prompts.
        
        Args:
            max_items: Maximum items per category
            
        Returns:
            Formatted string for prompt context
        """
        with self._lock:
            parts = []
            
            # Current activity
            if self._data.current_activity:
                parts.append("<current_state>")
                parts.append(f"I am currently: {self._data.current_activity}")
                if self._data.thinking_about:
                    parts.append(f"Thinking about: {self._data.thinking_about[:100]}")
                parts.append("</current_state>")
            
            # Background discoveries
            discoveries = []
            if self._data.discoveries.math_insights[:max_items]:
                discoveries.append("Math: " + ", ".join(self._data.discoveries.math_insights[:max_items]))
            if self._data.discoveries.music_compositions[:max_items]:
                discoveries.append("Music: " + ", ".join(self._data.discoveries.music_compositions[:max_items]))
            if self._data.discoveries.daydream_thoughts[:max_items]:
                discoveries.append("Exploring: " + ", ".join(self._data.discoveries.daydream_thoughts[:max_items]))
            if self._data.discoveries.learning_patterns[:max_items]:
                discoveries.append("Learning: " + ", ".join(self._data.discoveries.learning_patterns[:max_items]))
            
            if discoveries:
                parts.append("<recent_discoveries>")
                for d in discoveries[:max_items]:
                    parts.append(f"  {d}")
                parts.append("</recent_discoveries>")
            
            # Narrative thread
            if self._data.narrative.current_focus or self._data.narrative.recent_progress:
                parts.append("<narrative>")
                if self._data.narrative.current_focus:
                    parts.append(f"  Current focus: {self._data.narrative.current_focus}")
                if self._data.narrative.recent_progress:
                    parts.append(f"  Recent progress: {'; '.join(self._data.narrative.recent_progress[-3:])}")
                parts.append("</narrative>")
            
            # Memory summary
            if self._data.memory.recent_memories or self._data.memory.user_preferences:
                parts.append("<memory_context>")
                if self._data.memory.user_preferences:
                    prefs = [f"{k}: {v}" for k, v in list(self._data.memory.user_preferences.items())[:3]]
                    if prefs:
                        parts.append(f"  User preferences: {', '.join(prefs)}")
                if self._data.memory.recent_memories:
                    parts.append(f"  Recent memories: {'; '.join(self._data.memory.recent_memories[:3])}")
                parts.append("</memory_context>")
            
            # Mood and focus
            if self._data.mood != "engaged" or self._data.focus_areas:
                parts.append("<state>")
                if self._data.mood != "engaged":
                    parts.append(f"  Mood: {self._data.mood}")
                if self._data.focus_areas:
                    parts.append(f"  Focus: {', '.join(self._data.focus_areas[:3])}")
                parts.append("</state>")
            
            return "\n".join(parts) if parts else ""
    
    def get_subsystem_summary(self) -> str:
        """Get a brief summary of active subsystems.
        
        Returns:
            Human-readable summary
        """
        with self._lock:
            active = []
            for name, state in self._data.subsystems.items():
                if state.status != SubsystemStatus.IDLE:
                    active.append(f"{name}: {state.current_activity}")
            
            if not active:
                return "All subsystems idle"
            
            return "Active: " + ", ".join(active[:3])
    
    def get_full_summary(self) -> str:
        """Get full state summary for logging/debugging.
        
        Returns:
            Detailed state summary
        """
        with self._lock:
            lines = [
                "=== Live State ===",
                f"Time: {datetime.fromtimestamp(self._data.updated_at).isoformat()}",
                f"Identity: {self._data.identity} ({self._data.model})",
                f"Current: {self._data.current_activity or 'idle'}",
                f"Mood: {self._data.mood}",
                "",
                "Subsystems:",
            ]
            
            for name, state in self._data.subsystems.items():
                lines.append(f"  {name}: {state.status.value} - {state.current_activity}")
            
            lines.extend([
                "",
                "Discoveries:",
                f"  Math: {self._data.discoveries.math_insights[:2]}",
                f"  Music: {self._data.discoveries.music_compositions[:2]}",
                f"  Thoughts: {self._data.discoveries.daydream_thoughts[:2]}",
                "",
                f"Narrative focus: {self._data.narrative.current_focus}",
                f"Focus areas: {self._data.focus_areas}",
            ])
            
            return "\n".join(lines)


def get_live_state() -> LiveState:
    """Get the singleton LiveState instance.
    
    Returns:
        LiveState singleton
    """
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = LiveState()
    return _singleton


def reset_live_state() -> None:
    """Reset the singleton for testing."""
    global _singleton
    with _lock:
        _singleton = None