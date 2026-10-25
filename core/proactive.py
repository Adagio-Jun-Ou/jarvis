"""Proactive Engine - Generate proactive suggestions and actions.

This module enables Jarvis to be helpful proactively instead of just reactive,
responding to user context, goals, and patterns.
"""

import logging
import threading
import time
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of proactive suggestions."""
    HELP_WITH_TASK = "help_with_task"
    CONTINUE_WORK = "continue_work"
    BREAK_REMINDER = "break_reminder"
    RELATED_TOPIC = "related_topic"
    GOAL_PROGRESS = "goal_progress"
    LEARNING_OFFER = "learning_offer"
    FILE_REVIEW = "file_review"
    TEST_RUN = "test_run"


@dataclass
class ProactiveSuggestion:
    """A proactive suggestion for the user."""
    suggestion_type: SuggestionType
    message: str
    priority: int  # 1-5, 5 is highest
    confidence: float  # 0.0-1.0
    context: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    shown: bool = False
    dismissed: bool = False


@dataclass
class Opportunity:
    """An opportunity to help the user."""
    description: str
    relevance: float  # 0.0-1.0
    action_suggestion: str
    requires_permission: bool = False
    context: dict = field(default_factory=dict)


class ProactiveEngine:
    """Generate proactive suggestions and actions."""
    
    MAX_SUGGESTION_HISTORY = 200
    MAX_MENTIONED_FILES = 10
    MAX_MENTIONED_TOPICS = 20
    
    # Patterns that suggest user needs help
    HELP_PATTERNS = [
        r"(?:trying to|attempting to|working on) (.+?)(?:\.|$)",
        r"(?:stuck on|blocked on|having trouble with) (.+?)(?:\.|$)",
        r"(?:can't|cannot) (?:figure out|understand|get) (.+?)(?:\.|$)",
        r"(?:need to|have to|must) (.+?)(?:\.|$)",
    ]
    
    # Activity patterns that suggest break needed
    INTENSE_WORK_PATTERNS = [
        "debug", "error", "fix", "stuck", "issue", "problem",
        "not working", "failing", "crash"
    ]
    
    # Time thresholds for proactive suggestions
    BREAK_REMINDER_THRESHOLD = 45 * 60  # 45 minutes
    GOAL_CHECK_THRESHOLD = 20 * 60  # 20 minutes
    
    def __init__(self):
        self._session_start = time.time()
        self._last_suggestion_time = time.time() - 120  # Allow first suggestion immediately
        self._suggestion_history: list[ProactiveSuggestion] = []
        self._dismissed_types: dict[str, int] = {}  # type -> dismiss count
        self._lock = threading.RLock()
        self._conversation_turns: int = 0
        self._last_activity_time = time.time()
        self._mentioned_files: deque = deque(maxlen=self.MAX_MENTIONED_FILES)
        self._mentioned_topics: deque = deque(maxlen=self.MAX_MENTIONED_TOPICS)
    
    def should_suggest(
        self,
        user_mood: str = "neutral",
        active_goals: list = None,
        recent_activity: dict = None
    ) -> tuple[bool, ProactiveSuggestion | None]:
        """Determine if should make a proactive suggestion.
        
        Args:
            user_mood: Current detected user mood
            active_goals: List of active goals
            recent_activity: Recent activity dict
            
        Returns:
            Tuple of (should_suggest, suggestion)
        """
        with self._lock:
            # Don't suggest too frequently
            min_suggestion_interval = 2 * 60  # 2 minutes
            if time.time() - self._last_suggestion_time < min_suggestion_interval:
                return False, None
            
            # For urgent users, offer immediate help
            if user_mood == "urgent":
                return True, ProactiveSuggestion(
                    suggestion_type=SuggestionType.HELP_WITH_TASK,
                    message="I see this is urgent. How can I help right now?",
                    priority=5,
                    confidence=0.9
                )
            
            # Don't suggest if user seems frustrated or angry
            if user_mood in ["frustrated", "angry"]:
                return False, None
            
            recent_activity = recent_activity or {}
            active_goals = active_goals or []
            
            # Check for opportunities
            opportunities = self.check_for_opportunities(
                user_mood, active_goals, recent_activity
            )
            
            if not opportunities:
                return False, None
            
            # Pick highest relevance opportunity
            best = max(opportunities, key=lambda o: o.relevance)
            
            # Create suggestion
            suggestion = ProactiveSuggestion(
                suggestion_type=SuggestionType.HELP_WITH_TASK,
                message=best.action_suggestion,
                priority=min(5, int(best.relevance * 5)),
                confidence=best.relevance,
                context=best.context
            )
            
            self._suggestion_history.append(suggestion)
            if len(self._suggestion_history) > self.MAX_SUGGESTION_HISTORY:
                self._suggestion_history = self._suggestion_history[-self.MAX_SUGGESTION_HISTORY:]
            self._last_suggestion_time = time.time()
            
            return True, suggestion
    
    def check_for_opportunities(
        self,
        user_mood: str = "neutral",
        active_goals: list = None,
        recent_activity: dict = None
    ) -> list[Opportunity]:
        """Check for opportunities to help.
        
        Returns list of opportunities sorted by relevance.
        """
        with self._lock:
            mentioned_files = list(self._mentioned_files)
            last_activity_time = self._last_activity_time

        opportunities = []
        recent_activity = recent_activity or {}
        active_goals = active_goals or []
        
        # Check for goal progress opportunity
        for goal in active_goals[:3]:
            if hasattr(goal, 'blockers') and goal.blockers:
                opportunities.append(Opportunity(
                    description=f"Help with blocked goal: {goal.description}",
                    relevance=0.9,
                    action_suggestion=f"I see you're blocked on '{goal.description}'. Would you like help brainstorming solutions?",
                    context={"goal_id": getattr(goal, 'goal_id', None)}
                ))
            elif hasattr(goal, 'tasks') and goal.tasks:
                pending = [t for t in goal.tasks if getattr(t, 'status', 'pending') == 'pending']
                if pending:
                    opportunities.append(Opportunity(
                        description=f"Continue progress on: {goal.description}",
                        relevance=0.7,
                        action_suggestion=f"You have {len(pending)} tasks remaining for '{goal.description}'. Want to work on '{pending[0].description}'?",
                        context={"goal_id": getattr(goal, 'goal_id', None)}
                    ))
        
        # Check for recently mentioned files
        if mentioned_files:
            opportunities.append(Opportunity(
                description="Review recently mentioned files",
                relevance=0.5,
                action_suggestion=f"I noticed you mentioned {', '.join(mentioned_files[:2])}. Would you like me to analyze them?",
                context={"files": mentioned_files[:3]}
            ))
        
        # Check for time-based opportunities
        time_since_last_activity = time.time() - last_activity_time
        if time_since_last_activity > 1800:  # 30 minutes
            opportunities.append(Opportunity(
                description="Resume work session",
                relevance=0.6,
                action_suggestion="Welcome back! Would you like to continue where we left off?",
                context={"minutes_away": int(time_since_last_activity / 60)}
            ))
        
        # Check for break reminder
        time_since_session_start = time.time() - self._session_start
        if time_since_session_start > self.BREAK_REMINDER_THRESHOLD:
            intense_work = any(
                pattern in str(recent_activity.get('last_input', '')).lower()
                for pattern in self.INTENSE_WORK_PATTERNS
            )
            if intense_work:
                opportunities.append(Opportunity(
                    description="Break reminder after intense work",
                    relevance=0.4,
                    action_suggestion="You've been working intensely. Consider taking a short break!",
                    context={"session_minutes": int(time_since_session_start / 60)}
                ))
        
        # Sort by relevance
        opportunities.sort(key=lambda o: o.relevance, reverse=True)
        return opportunities
    
    def generate_suggestion(
        self,
        user_model: dict,
        active_goals: list,
        recent_activity: dict
    ) -> ProactiveSuggestion | None:
        """Generate a helpful proactive suggestion."""
        opportunities = self.check_for_opportunities(
            user_model.get("mood", "neutral"),
            active_goals,
            recent_activity
        )
        
        if not opportunities:
            return None
        
        # Pick best opportunity
        best = opportunities[0]
        
        return ProactiveSuggestion(
            suggestion_type=SuggestionType.HELP_WITH_TASK,
            message=best.action_suggestion,
            priority=min(5, int(best.relevance * 5)),
            confidence=best.relevance,
            context=best.context
        )
    
    def anticipate_needs(self, user_input: str) -> list[str]:
        """Anticipate what user might need next.
        
        Returns list of anticipated needs/suggestions.
        """
        needs = []
        user_lower = user_input.lower()
        
        # Extract what user is working on
        for pattern in self.HELP_PATTERNS:
            match = re.search(pattern, user_lower)
            if match:
                topic = match.group(1).strip()
                
                # Anticipate research help
                needs.append(f"Research {topic}")
                needs.append(f"Explain {topic}")
                needs.append(f"Find examples of {topic}")
                
                # Check if it's code-related
                if any(word in user_lower for word in ["code", "function", "bug", "error", "debug"]):
                    needs.append("Review related code")
                    needs.append("Run tests")
                    needs.append("Check error logs")
                
                break
        
        # Check for specific patterns
        if "error" in user_lower or "bug" in user_lower:
            needs.append("Search for similar issues")
            needs.append("Analyze the error message")
        
        if "learn" in user_lower:
            needs.append("Provide learning resources")
            needs.append("Explain the basics")
        
        if "build" in user_lower or "create" in user_lower:
            needs.append("Suggest architecture")
            needs.append("Generate boilerplate")
        
        return needs[:3]  # Return top 3
    
    def update_activity(self, user_input: str, mentioned_files: list[str] = None):
        """Update activity tracking.
        
        Args:
            user_input: User's input text
            mentioned_files: Files mentioned in conversation
        """
        with self._lock:
            self._conversation_turns += 1
            self._last_activity_time = time.time()
            
            if mentioned_files:
                for f in mentioned_files:
                    if f not in self._mentioned_files:
                        self._mentioned_files.append(f)
            
            words = user_input.lower().split()
            for word in words:
                if len(word) > 4:
                    self._mentioned_topics.append(word)
    
    def dismiss_suggestion(self, suggestion_type: SuggestionType):
        """Mark a suggestion type as dismissed."""
        with self._lock:
            type_name = suggestion_type.value
            self._dismissed_types[type_name] = self._dismissed_types.get(type_name, 0) + 1
    
    def get_context_for_prompt(self) -> str:
        """Get proactive context for LLM prompt.
        
        Returns formatted context string.
        """
        with self._lock:
            conversation_turns = self._conversation_turns
            mentioned_files = list(self._mentioned_files)[:3]
            mentioned_topics = list(self._mentioned_topics)
        
        lines = []
        
        # Add conversation context
        if conversation_turns > 0:
            lines.append(f"Conversation turns: {conversation_turns}")
        
        # Add recently mentioned files
        if mentioned_files:
            lines.append(f"Recently mentioned files: {', '.join(mentioned_files)}")
        
        # Add top topics
        if mentioned_topics:
            from collections import Counter
            topic_counts = Counter(mentioned_topics)
            top_topics = [t for t, _ in topic_counts.most_common(5)]
            lines.append(f"Topics of interest: {', '.join(top_topics)}")
        
        if not lines:
            return ""
        
        return "<proactive_context>\n" + "\n".join(lines) + "\n</proactive_context>"
    
    def get_suggestion_history(self, limit: int = 10) -> list[ProactiveSuggestion]:
        """Get recent suggestion history."""
        with self._lock:
            return self._suggestion_history[-limit:]


# Singleton
_engine: ProactiveEngine | None = None
_lock = threading.Lock()


def get_proactive_suggester() -> ProactiveEngine:
    """Get singleton ProactiveEngine instance (suggestion engine).
    
    Note: This is the proactive SUGGESTION engine from proactive.py.
    For the proactive INTELLIGENCE engine (idle-time API calls),
    use get_proactive_engine() from proactive_engine.py.
    """
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = ProactiveEngine()
    return _engine


def get_proactive_engine() -> ProactiveEngine:
    """Deprecated: Use get_proactive_suggester() for clarity.
    
    This alias exists for backward compatibility but will be removed.
    Prefer get_proactive_suggester() to distinguish from
    get_proactive_engine() in proactive_engine.py.
    """
    return get_proactive_suggester()


def reset_proactive_engine() -> None:
    """Reset the singleton (for testing)."""
    global _engine
    with _lock:
        _engine = None