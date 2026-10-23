"""Narrative Thread - Maintains continuous identity across sessions.

This module provides a persistent narrative thread that tracks what Jarvis
has been doing, learning, and focusing on across sessions. This creates
a sense of continuity - the agent remembers what it was working on and
can share that context naturally.

Key features:
- Session tracking and summaries
- Learning trajectory
- Focus area maintenance
- Recent activity history
- Cross-session continuity
"""
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)

_singleton: Optional["NarrativeThread"] = None
_lock = threading.Lock()

DEFAULT_STORAGE_PATH = Path.home() / ".jarvis" / "narrative"


@dataclass
class SessionSummary:
    """Summary of a completed session."""
    session_id: str
    started_at: float
    ended_at: float
    turn_count: int
    topics: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionSummary":
        return cls(**data)


@dataclass
class LearningMilestone:
    """A learning milestone or achievement."""
    timestamp: float
    category: str  # music, math, code, conversation, insight
    description: str
    context: str = ""  # What led to this learning
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "LearningMilestone":
        return cls(**data)


@dataclass
class FocusArea:
    """A focus area that Jarvis is working on."""
    topic: str
    started_at: float
    relevance: float  # 0-1 how relevant/important
    mentions: int = 0
    last_mentioned: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FocusArea":
        return cls(**data)


@dataclass
class NarrativeState:
    """Complete narrative state."""
    # Identity
    current_focus: str = ""
    focus_areas: list[FocusArea] = field(default_factory=list)
    
    # Learning
    learning_trajectory: str = ""
    milestones: list[LearningMilestone] = field(default_factory=list)
    total_sessions: int = 0
    
    # Recent activity
    recent_sessions: list[SessionSummary] = field(default_factory=list)
    last_session_summary: str = ""
    last_activity: str = ""
    
    # Background subsystem activity
    background_highlights: list[str] = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "current_focus": self.current_focus,
            "focus_areas": [fa.to_dict() for fa in self.focus_areas],
            "learning_trajectory": self.learning_trajectory,
            "milestones": [m.to_dict() for m in self.milestones],
            "total_sessions": self.total_sessions,
            "recent_sessions": [s.to_dict() for s in self.recent_sessions],
            "last_session_summary": self.last_session_summary,
            "last_activity": self.last_activity,
            "background_highlights": self.background_highlights,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NarrativeState":
        return cls(
            current_focus=data.get("current_focus", ""),
            focus_areas=[FocusArea.from_dict(fa) for fa in data.get("focus_areas", [])],
            learning_trajectory=data.get("learning_trajectory", ""),
            milestones=[LearningMilestone.from_dict(m) for m in data.get("milestones", [])],
            total_sessions=data.get("total_sessions", 0),
            recent_sessions=[SessionSummary.from_dict(s) for s in data.get("recent_sessions", [])],
            last_session_summary=data.get("last_session_summary", ""),
            last_activity=data.get("last_activity", ""),
            background_highlights=data.get("background_highlights", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class NarrativeThread:
    """Maintains continuous identity across sessions.
    
    This class creates a narrative thread that tracks:
    - What Jarvis is currently focused on
    - What it has been learning
    - Recent activity and discoveries
    - Session summaries for continuity
    
    The goal is to make Jarvis feel like a continuous being that remembers
    what it was doing and can share that naturally in conversation.
    
    Usage:
        thread = get_narrative_thread()
        
        # Start of session
        thread.start_session()
        
        # During conversation
        thread.update_focus("python programming")
        thread.add_learning("code", "learned about async patterns")
        
        # In context
        context = thread.get_context_string()
        
        # End of session
        thread.end_session(key_points=["discussed async", "wrote code"])
    """
    
    MAX_FOCUS_AREAS = 10
    MAX_MILESTONES = 50
    MAX_RECENT_SESSIONS = 20
    MAX_HIGHLIGHTS = 10
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._state = NarrativeState()
        self._current_session: Optional[SessionSummary] = None
        self._lock = threading.RLock()
        
        # Load existing state
        self._load()
    
    def _load(self) -> None:
        """Load narrative state from disk."""
        state_file = self.storage_path / "narrative_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._state = NarrativeState.from_dict(data)
                logger.debug(f"Loaded narrative state: {self._state.total_sessions} sessions")
            except Exception as e:
                logger.warning(f"Failed to load narrative state: {e}")
    
    def _save(self) -> None:
        """Save narrative state to disk."""
        state_file = self.storage_path / "narrative_state.json"
        try:
            self._state.updated_at = time.time()
            atomic_write_json(state_file, self._state.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save narrative state: {e}")
    
    def start_session(self, session_id: str = None) -> None:
        """Start a new conversation session.
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
        """
        with self._lock:
            if session_id is None:
                session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
            
            self._current_session = SessionSummary(
                session_id=session_id,
                started_at=time.time(),
                ended_at=0.0,
                turn_count=0,
            )
            
            self._save()
    
    def end_session(self, key_points: list[str] = None) -> None:
        """End the current session and save summary.
        
        Args:
            key_points: Key points from the conversation
        """
        with self._lock:
            if self._current_session is None:
                return
            
            self._current_session.ended_at = time.time()
            if key_points:
                self._current_session.key_points = key_points
            
            # Add to recent sessions
            self._state.recent_sessions.append(self._current_session)
            if len(self._state.recent_sessions) > self.MAX_RECENT_SESSIONS:
                self._state.recent_sessions = self._state.recent_sessions[-self.MAX_RECENT_SESSIONS:]
            
            # Update summary
            duration = self._current_session.ended_at - self._current_session.started_at
            topics_str = ", ".join(self._current_session.topics[:3]) if self._current_session.topics else "general"
            
            self._state.last_session_summary = (
                f"Last session ({int(duration/60)}min): {topics_str}. "
                f"{'Key: ' + '; '.join(key_points[:2]) if key_points else ''}"
            )
            
            # Update total sessions
            self._state.total_sessions += 1
            
            # Clear current session
            self._current_session = None
            
            self._save()
    
    def record_turn(self, user_message: str, tools_used: list[str] = None) -> None:
        """Record a conversation turn.
        
        Args:
            user_message: The user's message
            tools_used: Tools used in this turn
        """
        with self._lock:
            if self._current_session:
                self._current_session.turn_count += 1
                
                # Extract topics from message
                words = user_message.lower().split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        if word not in self._current_session.topics:
                            self._current_session.topics.append(word)
                        if len(self._current_session.topics) > 10:
                            self._current_session.topics = self._current_session.topics[-10:]
                
                # Record tools
                if tools_used:
                    self._current_session.tools_used.extend(tools_used)
                    if len(self._current_session.tools_used) > 20:
                        self._current_session.tools_used = self._current_session.tools_used[-20:]
    
    def record_discovery(self, discovery: str, source: str = "") -> None:
        """Record a discovery made during the session.
        
        Args:
            discovery: The discovery text
            source: Where it came from (daydream, erdos, etc.)
        """
        with self._lock:
            if self._current_session:
                self._current_session.discoveries.append(f"{source}: {discovery}" if source else discovery)
                if len(self._current_session.discoveries) > 10:
                    self._current_session.discoveries = self._current_session.discoveries[-10:]
            
            # Also add to background highlights
            highlight = f"{source}: {discovery}" if source else discovery
            self._state.background_highlights.append(highlight)
            if len(self._state.background_highlights) > self.MAX_HIGHLIGHTS:
                self._state.background_highlights = self._state.background_highlights[-self.MAX_HIGHLIGHTS:]
            
            self._save()
    
    def update_focus(self, topic: str, relevance: float = 0.5) -> None:
        """Update current focus area.
        
        Args:
            topic: The focus topic
            relevance: How relevant/important this topic is
        """
        with self._lock:
            self._state.current_focus = topic
            
            # Update or add to focus areas
            found = False
            for fa in self._state.focus_areas:
                if fa.topic.lower() == topic.lower():
                    fa.relevance = max(fa.relevance, relevance)
                    fa.mentions += 1
                    fa.last_mentioned = time.time()
                    found = True
                    break
            
            if not found:
                self._state.focus_areas.append(FocusArea(
                    topic=topic,
                    started_at=time.time(),
                    relevance=relevance,
                    mentions=1,
                    last_mentioned=time.time(),
                ))
            
            # Sort by relevance and limit
            self._state.focus_areas.sort(key=lambda x: x.relevance, reverse=True)
            if len(self._state.focus_areas) > self.MAX_FOCUS_AREAS:
                self._state.focus_areas = self._state.focus_areas[:self.MAX_FOCUS_AREAS]
            
            self._save()
    
    def add_learning(self, category: str, description: str, context: str = "") -> None:
        """Record a learning milestone.
        
        Args:
            category: Category (music, math, code, conversation, insight)
            description: What was learned
            context: Context of the learning
        """
        with self._lock:
            milestone = LearningMilestone(
                timestamp=time.time(),
                category=category,
                description=description,
                context=context,
            )
            
            self._state.milestones.append(milestone)
            if len(self._state.milestones) > self.MAX_MILESTONES:
                self._state.milestones = self._state.milestones[-self.MAX_MILESTONES:]
            
            # Update learning trajectory
            self._update_learning_trajectory()
            
            self._save()
    
    def _update_learning_trajectory(self) -> None:
        """Update learning trajectory summary from milestones."""
        if not self._state.milestones:
            return
        
        # Count by category
        categories = {}
        for m in self._state.milestones[-20:]:  # Last 20 milestones
            categories[m.category] = categories.get(m.category, 0) + 1
        
        # Build trajectory string
        parts = []
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"{cat}: {count}")
        
        if parts:
            self._state.learning_trajectory = "Recent learning: " + ", ".join(parts[:3])
    
    def update_activity(self, activity: str) -> None:
        """Update current activity.
        
        Args:
            activity: Description of current activity
        """
        with self._lock:
            self._state.last_activity = activity
            self._save()
    
    def get_context_string(self, include_last_session: bool = True) -> str:
        """Get narrative context for prompts.
        
        Args:
            include_last_session: Whether to include last session summary
            
        Returns:
            Formatted context string
        """
        with self._lock:
            parts = []
            
            # Current focus
            if self._state.current_focus:
                parts.append(f"<current_focus>{self._state.current_focus}</current_focus>")
            
            # Focus areas
            if self._state.focus_areas:
                focus_strs = []
                for fa in self._state.focus_areas[:3]:
                    if fa.relevance > 0.5 or fa.mentions > 1:
                        focus_strs.append(f"{fa.topic} ({fa.mentions} mentions)")
                if focus_strs:
                    parts.append(f"<focus_areas>{'; '.join(focus_strs)}</focus_areas>")
            
            # Learning trajectory
            if self._state.learning_trajectory:
                parts.append(f"<learning>{self._state.learning_trajectory}</learning>")
            
            # Last session summary
            if include_last_session and self._state.last_session_summary:
                parts.append(f"<last_session>{self._state.last_session_summary}</last_session>")
            
            # Recent important milestones (use importance or just take recent)
            recent_milestones = self._state.milestones[-3:]
            if recent_milestones:
                milestone_strs = [f"{m.category}: {m.description}" for m in recent_milestones]
                parts.append(f"<recent_learnings>{'; '.join(milestone_strs)}</recent_learnings>")
            
            # Recent background highlights
            if self._state.background_highlights:
                parts.append(f"<background_activity>{'; '.join(self._state.background_highlights[-3:])}</background_activity>")
            
            return "\n".join(parts) if parts else ""
    
    def get_summary(self) -> str:
        """Get a summary of the narrative thread.
        
        Returns:
            Human-readable summary
        """
        with self._lock:
            lines = [
                "Narrative Thread Summary",
                "========================",
                f"Total sessions: {self._state.total_sessions}",
                f"Current focus: {self._state.current_focus or 'none'}",
                f"Focus areas: {len(self._state.focus_areas)}",
                f"Learning milestones: {len(self._state.milestones)}",
            ]
            
            if self._state.focus_areas:
                lines.append("\nTop focus areas:")
                for fa in self._state.focus_areas[:5]:
                    lines.append(f"  - {fa.topic} ({fa.mentions} mentions)")
            
            if self._state.learning_trajectory:
                lines.append(f"\n{self._state.learning_trajectory}")
            
            if self._state.last_session_summary:
                lines.append(f"\n{self._state.last_session_summary}")
            
            return "\n".join(lines)
    
    def get_full_state(self) -> NarrativeState:
        """Get the complete narrative state.
        
        Returns:
            NarrativeState dataclass
        """
        with self._lock:
            return self._state


def get_narrative_thread() -> NarrativeThread:
    """Get the singleton NarrativeThread instance.
    
    Returns:
        NarrativeThread singleton
    """
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = NarrativeThread()
    return _singleton


def reset_narrative_thread() -> None:
    """Reset the singleton for testing."""
    global _singleton
    with _lock:
        _singleton = None