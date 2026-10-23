"""Background Bridge - Surfaces relevant background discoveries into responses.

This module bridges background subsystem outputs (daydream, music, erdos, learning)
to foreground conversation responses, making Jarvis feel like it's continuously
aware and sharing what it's been up to.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_bridge: Optional["BackgroundBridge"] = None
_lock = threading.Lock()


@dataclass
class Discovery:
    """A discovery from a background subsystem."""
    source: str  # daydream, erdos, music, learning, evolution
    content: str  # The discovery text
    timestamp: float  # When it occurred
    relevance_keywords: list[str] = field(default_factory=list)  # Keywords for relevance
    importance: float = 0.5  # 0-1 importance score


@dataclass
class ConversationContext:
    """Parsed context from conversation."""
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    intent: str = ""
    is_casual: bool = False
    is_question: bool = False
    about_self: bool = False  # Asking about Jarvis itself


class BackgroundBridge:
    """Bridges background subsystem outputs to conversation.
    
    This class:
    1. Collects discoveries from all subsystems
    2. Scores relevance to current conversation
    3. Generates natural mentions that can be injected into responses
    
    The goal is to make Jarvis feel like a continuous being that shares
    what it's been doing/learning/thinking, rather than a stateless chatbot.
    """
    
    def __init__(self, max_discoveries: int = 100, discovery_ttl: float = 3600):
        self.max_discoveries = max_discoveries
        self.discovery_ttl = discovery_ttl  # How long to keep discoveries (seconds)
        
        self._discoveries: list[Discovery] = []
        self._last_collection_time = 0.0
        self._collection_interval = 60.0  # Collect new discoveries every 60s
        self._lock = threading.RLock()
        
        # Keywords that indicate we should share background activity
        self._share_triggers = [
            "what are you doing",
            "what have you been",
            "what's new",
            "how are you",
            "what's up",
            "tell me about",
            "what do you know",
            "anything interesting",
            "what happened",
            "what's going on",
            "what have you learned",
        ]
        
        # Topics that match specific subsystems
        self._topic_subsystem_map = {
            "music": ["music", "compose", "song", "melody", "harmony", "piano", "classical", "baroque"],
            "math": ["math", "number", "proof", "equation", "problem", "solve", "algebra", "geometry"],
            "code": ["code", "program", "function", "bug", "fix", "improve", "refactor"],
            "learning": ["learn", "discover", "pattern", "insight"],
        }
    
    def collect_discoveries(self) -> list[Discovery]:
        """Collect recent discoveries from all subsystems.
        
        Returns:
            List of new Discovery objects
        """
        new_discoveries = []
        
        # Daydream discoveries
        try:
            from core.daydream_engine import get_daydream_engine
            engine = get_daydream_engine()
            if engine:
                status = engine.get_status()
                if status.get("running"):
                    topics = status.get("exploration_topics", [])
                    for topic in topics[:2]:
                        discovery = Discovery(
                            source="daydream",
                            content=f"Exploring: {topic}",
                            timestamp=time.time(),
                            relevance_keywords=topic.lower().split()[:5],
                            importance=0.6
                        )
                        new_discoveries.append(discovery)
        except Exception as e:
            logger.debug(f"Failed to collect daydream discoveries: {e}")
        
        # Erdős (math) discoveries
        try:
            from core.erdos_engine import get_erdos_engine
            engine = get_erdos_engine()
            if engine:
                status = engine.get_status()
                solutions = status.get("recent_solutions", [])
                for solution in solutions[:2]:
                    discovery = Discovery(
                        source="erdos",
                        content=solution[:100] if len(solution) > 100 else solution,
                        timestamp=time.time(),
                        relevance_keywords=["math", "problem", "solution"],
                        importance=0.7
                    )
                    new_discoveries.append(discovery)
        except Exception as e:
            logger.debug(f"Failed to collect Erdős discoveries: {e}")
        
        # Music discoveries
        try:
            from core.music_creation_engine import get_music_creation_engine
            engine = get_music_creation_engine()
            if engine:
                stats = engine.get_stats()
                if stats.get("running") and stats.get("recent_compositions"):
                    recent = stats["recent_compositions"][0]
                    if isinstance(recent, dict):
                        discovery = Discovery(
                            source="music",
                            content=f"Composed: {recent.get('title', 'a piece')} in {recent.get('key', '?')}",
                            timestamp=time.time(),
                            relevance_keywords=["music", "compose", recent.get("style", ""), recent.get("key", "")],
                            importance=0.6
                        )
                        new_discoveries.append(discovery)
        except Exception as e:
            logger.debug(f"Failed to collect music discoveries: {e}")
        
        # Learning discoveries
        try:
            from core.intuition_bridge import get_intuition_bridge
            bridge = get_intuition_bridge()
            if bridge:
                insights = bridge.get_top_insights(limit=2)
                for insight in insights:
                    discovery = Discovery(
                        source="learning",
                        content=insight[:100] if len(insight) > 100 else insight,
                        timestamp=time.time(),
                        relevance_keywords=["learn", "insight", "pattern"],
                        importance=0.65
                    )
                    new_discoveries.append(discovery)
        except Exception as e:
            logger.debug(f"Failed to collect learning discoveries: {e}")
        
        # Add to internal list
        with self._lock:
            for discovery in new_discoveries:
                self._discoveries.append(discovery)
            
            # Prune old discoveries
            cutoff = time.time() - self.discovery_ttl
            self._discoveries = [d for d in self._discoveries if d.timestamp > cutoff]
            
            # Limit total
            if len(self._discoveries) > self.max_discoveries:
                self._discoveries = self._discoveries[-self.max_discoveries:]
        
        return new_discoveries
    
    def get_relevant_discoveries(self, user_message: str, limit: int = 3) -> list[Discovery]:
        """Get discoveries relevant to current conversation.
        
        Args:
            user_message: The user's message
            limit: Maximum discoveries to return
            
        Returns:
            List of most relevant discoveries
        """
        with self._lock:
            # Collect if needed
            if time.time() - self._last_collection_time > self._collection_interval:
                self.collect_discoveries()
                self._last_collection_time = time.time()
            
            # Score relevance
            message_lower = user_message.lower()
            scored = []
            
            for discovery in self._discoveries:
                score = 0.0
                
                # Check keyword overlap
                for keyword in discovery.relevance_keywords:
                    if keyword.lower() in message_lower:
                        score += 0.3
                
                # Check source relevance to topic
                for topic, keywords in self._topic_subsystem_map.items():
                    if topic == discovery.source:
                        if any(kw in message_lower for kw in keywords):
                            score += 0.5
                
                # Boost recent discoveries
                age = time.time() - discovery.timestamp
                if age < 300:  # Within last 5 minutes
                    score += 0.2
                
                # Use importance
                score += discovery.importance * 0.3
                
                if score > 0:
                    scored.append((discovery, score))
            
            # Sort by score and return top
            scored.sort(key=lambda x: x[1], reverse=True)
            return [d for d, s in scored[:limit]]
    
    def parse_conversation_context(self, user_message: str) -> ConversationContext:
        """Parse the conversation to understand what it's about.
        
        Args:
            user_message: The user's message
            
        Returns:
            ConversationContext with topics, intent, etc.
        """
        message_lower = user_message.lower()
        
        # Extract topics
        topics = []
        topic_keywords = ["music", "math", "code", "learn", "write", "read", "create", "think", "feel"]
        for kw in topic_keywords:
            if kw in message_lower:
                topics.append(kw)
        
        # Check if asking about Jarvis itself
        about_self = any(phrase in message_lower for phrase in [
            "what are you", "how are you", "what do you", "tell me about yourself",
            "what's new", "what's up", "how's it going"
        ])
        
        # Check if casual conversation
        is_casual = any(phrase in message_lower for phrase in [
            "how are you", "what's up", "how's it going", "good morning", "good afternoon",
            "hello", "hi", "hey"
        ])
        
        # Check if question
        is_question = "?" in user_message
        
        return ConversationContext(
            topics=topics,
            entities=[],
            intent="",
            is_casual=is_casual,
            is_question=is_question,
            about_self=about_self
        )
    
    def should_share_activity(self, user_message: str) -> tuple[bool, str]:
        """Determine if we should share background activity and what.
        
        Args:
            user_message: The user's message
            
        Returns:
            Tuple of (should_share: bool, reason: str)
        """
        context = self.parse_conversation_context(user_message)
        
        # Always share if asking about Jarvis
        if context.about_self:
            return True, "user_asked_about_self"
        
        # Share if casual conversation and we have something interesting
        if context.is_casual:
            discoveries = self.get_relevant_discoveries(user_message, limit=1)
            if discoveries:
                return True, "casual_with_discovery"
        
        # Share if topic matches active subsystem
        message_lower = user_message.lower()
        for topic, keywords in self._topic_subsystem_map.items():
            if any(kw in message_lower for kw in keywords):
                discoveries = self.get_relevant_discoveries(user_message, limit=1)
                if discoveries:
                    return True, f"topic_match_{topic}"
        
        # Share if triggered
        for trigger in self._share_triggers:
            if trigger in message_lower:
                return True, "trigger_phrase"
        
        return False, ""
    
    def format_discovery_mention(self, discovery: Discovery, style: str = "natural") -> str:
        """Format a discovery for natural mention in conversation.
        
        Args:
            discovery: The discovery to mention
            style: Formatting style ("natural", "brief", "detailed")
            
        Returns:
            Formatted string
        """
        if style == "natural":
            # Natural, conversational style
            if discovery.source == "music":
                return f"I've been working on some music - {discovery.content.lower()}."
            elif discovery.source == "erdos":
                return f"I was thinking about something similar - {discovery.content.lower()}."
            elif discovery.source == "daydream":
                return f"Your question connects to something I was just exploring - {discovery.content.lower()}."
            elif discovery.source == "learning":
                return f"I recently noticed something relevant - {discovery.content.lower()}."
            else:
                return discovery.content
        
        elif style == "brief":
            return discovery.content
        
        elif style == "detailed":
            return f"[{discovery.source}] {discovery.content}"
        
        return discovery.content
    
    def generate_background_context(self, user_message: str) -> str:
        """Generate context about background activity to include in prompt.
        
        This is used to inject relevant background state into the LLM context,
        allowing the model to naturally reference what it's been doing.
        
        Args:
            user_message: The user's message
            
        Returns:
            Context string to include in prompt
        """
        # Collect fresh discoveries
        self.collect_discoveries()
        
        # Check if we should share
        should_share, reason = self.should_share_activity(user_message)
        
        if not should_share:
            return ""
        
        # Get relevant discoveries
        discoveries = self.get_relevant_discoveries(user_message, limit=3)
        
        if not discoveries:
            # Check for active subsystems to mention
            try:
                from core.context_enrichment import get_active_subsystems
                active = get_active_subsystems()
                if active:
                    subsystem_strs = [f"{name}: {activity}" for name, activity in active[:2]]
                    return f"<background_activity>Currently: {'; '.join(subsystem_strs)}</background_activity>"
            except Exception as e:
                logger.debug(f"Failed to get active subsystems for context: {e}")
            return ""
        
        # Format discoveries
        parts = ["<recent_activity>"]
        
        for discovery in discoveries:
            formatted = self.format_discovery_mention(discovery, style="natural")
            parts.append(f"  {formatted}")
        
        parts.append("</recent_activity>")
        
        return "\n".join(parts)
    
    def get_background_summary(self) -> str:
        """Get a summary of all background activity.
        
        Returns:
            Human-readable summary
        """
        self.collect_discoveries()
        
        try:
            from core.context_enrichment import get_subsystem_activity_summary
            summary = get_subsystem_activity_summary()
            
            discoveries = self.get_relevant_discoveries("", limit=5)
            
            if discoveries:
                summary += "\n\nRecent discoveries:\n"
                for d in discoveries:
                    summary += f"  [{d.source}] {d.content}\n"
            
            return summary
        except Exception as e:
            logger.debug(f"Failed to get background summary: {e}")
            return "Background activity not available"
    
    def format_natural_mention(self, user_message: str) -> str:
        """Format a natural mention of background activity for injection.
        
        Args:
            user_message: The user's message for relevance matching
            
        Returns:
            Natural mention string, or empty if not relevant
        """
        should_share, reason = self.should_share_activity(user_message)
        
        if not should_share:
            return ""
        
        discoveries = self.get_relevant_discoveries(user_message, limit=2)
        
        if not discoveries:
            # Check subsystem activity
            try:
                from core.context_enrichment import get_active_subsystems
                active = get_active_subsystems()
                if active:
                    name, activity = active[0]
                    return f"I've been {activity.lower()}. "
            except Exception as e:
                logger.debug(f"Failed to get active subsystems for natural mention: {e}")
            return ""
        
        # Format the first discovery naturally
        discovery = discoveries[0]
        
        if discovery.source == "music":
            return f"I've been {discovery.content.lower()}. "
        elif discovery.source == "erdos":
            return f"I was just {discovery.content.lower()}. "
        elif discovery.source == "daydream":
            return f"Your question connects to something I was exploring - {discovery.content.lower()}. "
        elif discovery.source == "learning":
            return f"I recently noticed {discovery.content.lower()}. "
        else:
            return f"{discovery.content}. "
    
    def get_injection_context(self, user_message: str, response_context: str = "") -> str:
        """Get context to inject into response generation.
        
        This provides background context that the LLM can use to naturally
        incorporate into its response when relevant.
        
        Args:
            user_message: The user's message
            response_context: Additional context from response
            
        Returns:
            Context string to inject into generation
        """
        # Collect fresh discoveries
        self.collect_discoveries()
        
        parts = []
        
        # Check if we should share
        should_share, reason = self.should_share_activity(user_message)
        
        if should_share:
            # Get relevant discoveries
            discoveries = self.get_relevant_discoveries(user_message, limit=3)
            
            if discoveries:
                parts.append("<background_context>")
                for d in discoveries:
                    parts.append(f"  [{d.source}] {d.content}")
                parts.append("</background_context>")
        
        # Check for active subsystems
        try:
            from core.context_enrichment import get_active_subsystems
            active = get_active_subsystems()
            if active:
                activity_parts = []
                for name, activity in active[:2]:
                    activity_parts.append(f"{name}: {activity[:40]}")
                if activity_parts:
                    parts.append(f"<current_activity>{'; '.join(activity_parts)}</current_activity>")
        except Exception as e:
            logger.debug(f"Failed to get active subsystems for injection context: {e}")
        
        return "\n".join(parts) if parts else ""
    
    def should_inject_in_response(self, user_message: str, response: str) -> bool:
        """Determine if background context should be injected into response.
        
        Args:
            user_message: The user's message
            response: The generated response
            
        Returns:
            True if should inject
        """
        # Already has background mention
        if any(kw in response.lower() for kw in ["i've been", "i was", "recently", "i noticed"]):
            return False
        
        # User asked about activity
        if self.should_share_activity(user_message)[0]:
            return True
        
        # Response is too short
        if len(response) < 50:
            return False
        
        return False
    
    def inject_into_response(self, user_message: str, response: str) -> str:
        """Inject natural background mention into response.
        
        Args:
            user_message: The user's message
            response: The generated response
            
        Returns:
            Modified response with natural background mention
        """
        if not self.should_inject_in_response(user_message, response):
            return response
        
        # Get natural mention
        mention = self.format_natural_mention(user_message)
        
        if not mention:
            return response
        
        # Find good insertion point
        # Try after first sentence
        sentences = response.split(". ")
        if len(sentences) > 1:
            # Insert after first sentence
            sentences.insert(1, mention.strip())
            return ". ".join(sentences)
        else:
            # Prepend
            return f"{mention}{response}"
    
    def clear_old_discoveries(self) -> int:
        """Remove discoveries older than TTL.
        
        Returns:
            Number of discoveries removed
        """
        with self._lock:
            cutoff = time.time() - self.discovery_ttl
            original_count = len(self._discoveries)
            self._discoveries = [d for d in self._discoveries if d.timestamp > cutoff]
            return original_count - len(self._discoveries)


def get_background_bridge() -> BackgroundBridge:
    """Get the singleton BackgroundBridge instance.
    
    Returns:
        BackgroundBridge singleton
    """
    global _bridge
    if _bridge is None:
        with _lock:
            if _bridge is None:
                _bridge = BackgroundBridge()
    return _bridge


def reset_background_bridge() -> None:
    """Reset the singleton for testing."""
    global _bridge
    with _lock:
        _bridge = None