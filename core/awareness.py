"""Context visibility and awareness system for Jarvis.

This module provides users visibility into Jarvis's thinking process,
displaying relevant memories, confidence levels, and reasoning chains
when requested.

Key features:
- Show what Jarvis is thinking about
- Display relevant memories being used
- Indicate confidence levels for suggestions
- Show reasoning chain on request
"""
import threading
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AwarenessLevel(Enum):
    """Levels of awareness visibility."""
    MINIMAL = "minimal"       # Only show suggestions
    NORMAL = "normal"         # Show suggestions with brief context
    DETAILED = "detailed"     # Show suggestions with reasoning
    DEBUG = "debug"          # Show everything


@dataclass
class ThoughtVisibility:
    """Represents what's visible about a thought."""
    thought_id: str
    content_preview: str
    confidence: float
    suggestion_type: str
    reasoning_chain: list[str] = field(default_factory=list)
    related_memories: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_display(self, level: AwarenessLevel) -> str:
        """Format for display at the given awareness level.

        Args:
            level: The awareness level to display at

        Returns:
            Formatted string for display
        """
        if level == AwarenessLevel.MINIMAL:
            return self.content_preview

        lines = [f"💭 {self.content_preview}"]

        if level in [AwarenessLevel.NORMAL, AwarenessLevel.DETAILED, AwarenessLevel.DEBUG]:
            confidence_bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
            lines.append(f"   Confidence: [{confidence_bar}] {self.confidence:.0%}")

        if level in [AwarenessLevel.DETAILED, AwarenessLevel.DEBUG]:
            if self.reasoning_chain:
                lines.append("   Reasoning:")
                for step in self.reasoning_chain:
                    lines.append(f"     → {step}")

        if level == AwarenessLevel.DEBUG:
            lines.append(f"   Type: {self.suggestion_type}")
            lines.append(f"   ID: {self.thought_id}")
            if self.related_memories:
                lines.append("   Related memories:")
                for mem in self.related_memories[:3]:
                    lines.append(f"     • {mem[:100]}...")

        return "\n".join(lines)


@dataclass
class MemoryVisibility:
    """Represents what's visible about a memory."""
    memory_id: str
    content: str
    relevance_score: float
    timestamp: datetime
    source: str = "conversation"

    def to_display(self) -> str:
        """Format for display.

        Returns:
            Formatted string for display
        """
        relevance = f"[{self.relevance_score:.0%} relevant]"
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"📜 {relevance} {time_str}: {self.content[:100]}..."


class AwarenessManager:
    """Manages visibility into Jarvis's thinking process.

    This class provides a window into Jarvis's internal state,
    allowing users to understand what Jarvis is thinking about
    and why it makes certain suggestions.

    Example usage:
        awareness = AwarenessManager(storage_path=Path("~/.jarvis/awareness"))
        awareness.set_level(AwarenessLevel.DETAILED)

        # When a thought is generated:
        visibility = awareness.create_thought_visibility(thought, reasoning)
        print(visibility.to_display(awareness.level))

        # User asks "what are you thinking about?":
        print(awareness.get_current_focus())
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        level: AwarenessLevel = AwarenessLevel.NORMAL,
    ):
        """Initialize the awareness manager.

        Args:
            storage_path: Path to store awareness data
            level: Initial awareness level
        """
        self.storage_path = storage_path
        self.level = level

        self._lock = threading.RLock()

        # Current state
        self._current_thoughts: list[ThoughtVisibility] = []
        self._relevant_memories: list[MemoryVisibility] = []
        self._focus_topics: list[str] = []

        # Statistics
        self._thoughts_generated = 0
        self._suggestions_shown = 0
        self._memories_accessed = 0

    def set_level(self, level: AwarenessLevel | str) -> None:
        """Set the awareness level.

        Args:
            level: The awareness level (enum or string)
        """
        if isinstance(level, str):
            level_map = {
                "minimal": AwarenessLevel.MINIMAL,
                "normal": AwarenessLevel.NORMAL,
                "detailed": AwarenessLevel.DETAILED,
                "debug": AwarenessLevel.DEBUG,
            }
            self.level = level_map.get(level.lower(), AwarenessLevel.NORMAL)
        else:
            self.level = level

    def create_thought_visibility(
        self,
        thought: Any,
        reasoning: list[str] | None = None,
        related_memories: list[str] | None = None,
    ) -> ThoughtVisibility:
        """Create a visibility object for a thought.

        Args:
            thought: The Thought object
            reasoning: Optional reasoning chain
            related_memories: Optional list of related memory IDs

        Returns:
            A ThoughtVisibility object
        """
        with self._lock:
            self._thoughts_generated += 1

            visibility = ThoughtVisibility(
                thought_id=f"thought_{self._thoughts_generated}",
                content_preview=thought.content[:200] + "..." if len(thought.content) > 200 else thought.content,
                confidence=thought.confidence,
                suggestion_type=thought.suggestion_type.value,
                reasoning_chain=reasoning or [],
                related_memories=related_memories or [],
            )

            self._current_thoughts.append(visibility)
            if len(self._current_thoughts) > 20:
                self._current_thoughts = self._current_thoughts[-20:]

        return visibility

    def add_memory_visibility(
        self,
        content: str,
        relevance_score: float,
        timestamp: datetime,
        source: str = "conversation",
    ) -> MemoryVisibility:
        """Add a memory to the visibility system.

        Args:
            content: The memory content
            relevance_score: How relevant this memory is (0.0-1.0)
            timestamp: When the memory was created
            source: Source of the memory

        Returns:
            A MemoryVisibility object
        """
        with self._lock:
            self._memories_accessed += 1

            visibility = MemoryVisibility(
                memory_id=f"memory_{self._memories_accessed}",
                content=content,
                relevance_score=relevance_score,
                timestamp=timestamp,
                source=source,
            )

            self._relevant_memories.append(visibility)
            if len(self._relevant_memories) > 50:
                self._relevant_memories = self._relevant_memories[-50:]

        return visibility

    def set_focus_topics(self, topics: list[str]) -> None:
        """Set the current focus topics.

        Args:
            topics: List of topics Jarvis is focusing on
        """
        with self._lock:
            self._focus_topics = topics

    def get_current_focus(self) -> str:
        """Get a summary of what Jarvis is currently thinking about.

        Returns:
            A human-readable summary of current focus
        """
        with self._lock:
            focus_topics = list(self._focus_topics)
            current_thoughts = list(self._current_thoughts)
            relevant_memories = list(self._relevant_memories)

        lines = ["🧠 Current Focus"]

        if focus_topics:
            lines.append("Topics of interest:")
            for topic in focus_topics[:5]:
                lines.append(f"  • {topic}")
        else:
            lines.append("No specific focus at the moment.")

        if current_thoughts:
            lines.append("\nRecent thoughts:")
            for thought in current_thoughts[-3:]:
                lines.append(f"  💭 {thought.content_preview[:50]}...")

        if relevant_memories and self.level in [AwarenessLevel.DETAILED, AwarenessLevel.DEBUG]:
            lines.append("\nRelevant memories:")
            for mem in relevant_memories[-3:]:
                lines.append(f"  📜 {mem.content[:50]}...")

        return "\n".join(lines)

    def get_suggestion_context(self, thought: Any) -> str:
        """Get context for displaying a suggestion.

        Args:
            thought: The thought being suggested

        Returns:
            Context string for the suggestion
        """
        if self.level == AwarenessLevel.MINIMAL:
            return thought.content

        lines = [f"💭 {thought.content}"]

        if self.level in [AwarenessLevel.NORMAL, AwarenessLevel.DETAILED, AwarenessLevel.DEBUG]:
            # Add confidence indicator
            conf = thought.confidence
            if conf >= 0.8:
                conf_label = "high confidence"
            elif conf >= 0.6:
                conf_label = "moderate confidence"
            else:
                conf_label = "exploring"

            lines.append(f"   [{conf_label}]")

        if self.level in [AwarenessLevel.DETAILED, AwarenessLevel.DEBUG]:
            lines.append(f"   Type: {thought.suggestion_type.value}")
            if thought.topics:
                lines.append(f"   Related to: {', '.join(thought.topics[:3])}")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the awareness system.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return {
                "level": self.level.value,
                "thoughts_generated": self._thoughts_generated,
                "suggestions_shown": self._suggestions_shown,
                "memories_accessed": self._memories_accessed,
                "current_focus_topics": list(self._focus_topics),
                "recent_thought_count": len(self._current_thoughts),
                "relevant_memory_count": len(self._relevant_memories),
            }

    def record_suggestion_shown(self) -> None:
        """Record that a suggestion was displayed to the user."""
        with self._lock:
            self._suggestions_shown += 1

    def format_thought_for_display(self, thought: ThoughtVisibility) -> str:
        """Format a thought for display at the current awareness level.

        Args:
            thought: The thought to format

        Returns:
            Formatted string
        """
        return thought.to_display(self.level)

    def format_suggestion(self, content: str, confidence: float, thought_type: str) -> str:
        """Format a suggestion for display.

        Args:
            content: The suggestion content
            confidence: Confidence score
            thought_type: Type of thought

        Returns:
            Formatted string for display
        """
        if self.level == AwarenessLevel.MINIMAL:
            return f"💭 {content}"

        # Add visual indicators based on confidence
        if confidence >= 0.8:
            indicator = "✨"  # High confidence - sparkle
        elif confidence >= 0.6:
            indicator = "💡"  # Moderate - lightbulb
        else:
            indicator = "🤔"  # Lower - thinking

        return f"{indicator} {content}"

    def save_state(self) -> None:
        """Save the current state to disk."""
        if self.storage_path is None:
            return

        with self._lock:
            state = {
                "level": self.level.value,
                "thoughts_generated": self._thoughts_generated,
                "suggestions_shown": self._suggestions_shown,
                "memories_accessed": self._memories_accessed,
                "focus_topics": list(self._focus_topics),
            }

        self.storage_path.mkdir(parents=True, exist_ok=True)
        state_file = self.storage_path / "awareness_state.json"
        atomic_write_json(state_file, state)

    def load_state(self) -> bool:
        """Load state from disk.

        Returns:
            True if state was loaded, False otherwise
        """
        if self.storage_path is None:
            return False

        state_file = self.storage_path / "awareness_state.json"
        if not state_file.exists():
            return False

        try:
            state = read_json_with_integrity(state_file)

            with self._lock:
                self.level = AwarenessLevel(state.get("level", "normal"))
                self._thoughts_generated = state.get("thoughts_generated", 0)
                self._suggestions_shown = state.get("suggestions_shown", 0)
                self._memories_accessed = state.get("memories_accessed", 0)
                self._focus_topics = state.get("focus_topics", [])

            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear the current state."""
        with self._lock:
            self._current_thoughts.clear()
            self._relevant_memories.clear()
            self._focus_topics.clear()
