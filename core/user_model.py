"""User Model - Build and maintain model of user over time.

This module tracks user preferences, communication style, interests, expertise,
and goals across conversations to enable personalization and better assistance.

Key features:
- Learn preferences from conversations
- Track communication style preferences
- Detect expertise areas
- Track mentioned projects and goals
- Maintain cross-session persistence
"""
from __future__ import annotations

from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton
_user_model: Optional["UserModelManager"] = None
_lock = threading.Lock()

DEFAULT_STORAGE_PATH = Path.home() / ".jarvis" / "user_model"

# Memory bounds to prevent unbounded growth
MAX_PREFERENCES = 100
MAX_INTERESTS = 50
MAX_EXPERTISE_AREAS = 20
MAX_ACTIVE_GOALS = 10
MAX_MENTIONED_PROJECTS = 20
MAX_PATTERNS = 50
MAX_TOPICS_DISCUSSED = 100
MAX_TOPICS_PER_UPDATE = 5


class DetailLevel(Enum):
    """Preferred response detail level."""
    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"


class CommunicationStyle(Enum):
    """Communication style preferences."""
    CASUAL = "casual"
    FORMAL = "formal"
    TECHNICAL = "technical"
    MIXED = "mixed"


@dataclass
class UserPreference:
    """A learned user preference."""
    key: str
    value: str
    confidence: float  # 0.0-1.0
    source: str  # "explicit", "inferred", "observed"
    timestamp: float
    access_count: int = 0


@dataclass
class ExpertiseArea:
    """User's expertise in a domain."""
    domain: str
    level: str  # "beginner", "intermediate", "expert"
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class UserGoal:
    """A user's mentioned goal or project."""
    description: str
    status: str  # "active", "paused", "completed", "mentioned"
    first_mentioned: float
    last_mentioned: float
    mention_count: int = 1
    related_topics: list[str] = field(default_factory=list)
    progress_notes: list[str] = field(default_factory=list)


@dataclass
class ConversationPattern:
    """Pattern observed in user's conversations."""
    pattern_type: str  # "question_style", "topic_preference", "time_preference"
    pattern_value: str
    frequency: int
    last_seen: float = field(default_factory=time.time)


CURRENT_USER_MODEL_VERSION = 1


def migrate_user_model(data: dict, from_version: int) -> dict:
    """Migrate user model data to current schema version."""
    current_version = CURRENT_USER_MODEL_VERSION
    
    if from_version >= current_version:
        return data
    
    # Apply migrations in sequence
    for version in range(from_version, current_version):
        migration_func = USER_MODEL_MIGRATIONS.get(version)
        if migration_func:
            data = migration_func(data)
        data["schema_version"] = version + 1
    
    return data


def migrate_v0_to_v1(data: dict) -> dict:
    """Migrate from version 0 (no version) to version 1."""
    # Ensure all required fields exist
    if "schema_version" not in data:
        data["schema_version"] = 1
    if "user_id" not in data:
        data["user_id"] = "default"
    if "created_at" not in data:
        data["created_at"] = time.time()
    if "last_interaction" not in data:
        data["last_interaction"] = time.time()
    return data


USER_MODEL_MIGRATIONS = {
    0: migrate_v0_to_v1,
}


@dataclass
class UserModelData:
    """Complete user model data."""
    # Schema version for migration
    schema_version: int = 1
    
    # Identity
    user_id: str = "default"
    created_at: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    interaction_count: int = 0
    
    # Learned preferences
    preferences: dict[str, UserPreference] = field(default_factory=dict)
    
    # Communication style
    communication_style: CommunicationStyle = CommunicationStyle.MIXED
    detail_preference: DetailLevel = DetailLevel.MEDIUM
    uses_technical_terms: bool = False
    uses_emojis: bool = False
    asks_followup_questions: bool = True
    average_message_length: float = 50.0
    
    # Interests and expertise
    interests: dict[str, float] = field(default_factory=dict)  # topic -> score
    expertise_areas: dict[str, ExpertiseArea] = field(default_factory=dict)
    
    # Goals and projects
    active_goals: list[UserGoal] = field(default_factory=list)
    mentioned_projects: list[str] = field(default_factory=list)
    
    # Conversation patterns
    patterns: list[ConversationPattern] = field(default_factory=list)
    
    # Topics discussed
    topics_discussed: dict[str, int] = field(default_factory=dict)  # topic -> count
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_interaction": self.last_interaction,
            "interaction_count": self.interaction_count,
            "preferences": {k: asdict(v) for k, v in self.preferences.items()},
            "communication_style": self.communication_style.value,
            "detail_preference": self.detail_preference.value,
            "uses_technical_terms": self.uses_technical_terms,
            "uses_emojis": self.uses_emojis,
            "asks_followup_questions": self.asks_followup_questions,
            "average_message_length": self.average_message_length,
            "interests": self.interests,
            "expertise_areas": {k: asdict(v) for k, v in self.expertise_areas.items()},
            "active_goals": [asdict(g) for g in self.active_goals],
            "mentioned_projects": self.mentioned_projects,
            "patterns": [asdict(p) for p in self.patterns],
            "topics_discussed": self.topics_discussed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserModelData":
        """Create from dictionary with migration support."""
        # Get schema version, default to 0 for old data
        schema_version = data.get("schema_version", 0)
        
        # Apply migrations if needed
        data = migrate_user_model(data, schema_version)
        
        preferences = {}
        for k, v in data.get("preferences", {}).items():
            if isinstance(v, dict):
                preferences[k] = UserPreference(**v)
        
        expertise_areas = {}
        for k, v in data.get("expertise_areas", {}).items():
            if isinstance(v, dict):
                expertise_areas[k] = ExpertiseArea(**v)
        
        active_goals = []
        for g in data.get("active_goals", []):
            if isinstance(g, dict):
                active_goals.append(UserGoal(**g))
        
        patterns = []
        for p in data.get("patterns", []):
            if isinstance(p, dict):
                patterns.append(ConversationPattern(**p))
        
        return cls(
            user_id=data.get("user_id", "default"),
            created_at=data.get("created_at", time.time()),
            last_interaction=data.get("last_interaction", time.time()),
            interaction_count=data.get("interaction_count", 0),
            preferences=preferences,
            communication_style=CommunicationStyle(data.get("communication_style", "mixed")),
            detail_preference=DetailLevel(data.get("detail_preference", "medium")),
            uses_technical_terms=data.get("uses_technical_terms", False),
            uses_emojis=data.get("uses_emojis", False),
            asks_followup_questions=data.get("asks_followup_questions", True),
            average_message_length=data.get("average_message_length", 50.0),
            interests=data.get("interests", {}),
            expertise_areas=expertise_areas,
            active_goals=active_goals,
            mentioned_projects=data.get("mentioned_projects", []),
            patterns=patterns,
            topics_discussed=data.get("topics_discussed", {}),
        )


class UserModelManager:
    """Manages user model learning and retrieval.
    
    This class:
    1. Learns user preferences from conversations
    2. Tracks communication style patterns
    3. Detects expertise areas
    4. Tracks goals and projects
    5. Persists user model across sessions
    
    Usage:
        model = get_user_model_manager()
        
        # Update from conversation
        model.update_from_input("I prefer Python over JavaScript")
        model.update_from_input("Can you explain this in detail?")
        
        # Get preferences
        prefs = model.get_preferences()
        style = model.get_communication_style()
        
        # Get context for prompts
        context = model.get_user_context()
    """
    
    # Pattern definitions
    TECHNICAL_TERMS = {
        "function", "class", "method", "variable", "API", "endpoint",
        "algorithm", "data structure", "database", "server", "client",
        "async", "await", "callback", "promise", "stream", "buffer",
        "recursion", "iteration", "loop", "array", "object", "hash",
        "compile", "runtime", "debug", "test", "deploy", "container",
    }
    
    INTEREST_KEYWORDS = {
        "python", "javascript", "rust", "go", "java", "c++", "typescript",
        "machine learning", "AI", "web development", "mobile", "backend",
        "frontend", "devops", "cloud", "security", "database", "API",
        "music", "art", "writing", "gaming", "math", "science",
        "business", "marketing", "finance", "health", "fitness",
    }
    
    GOAL_INDICATORS = [
        (r"i want to (learn|build|create|make|develop) (.+?)(?:\.|$)", 2),
        (r"i need to (finish|complete|implement) (.+?)(?:\.|$)", 2),
        (r"my goal is (.+?)(?:\.|$)", 1),
        (r"i'm working on (.+?)(?:\.|$)", 1),
        (r"i'm trying to (.+?)(?:\.|$)", 1),
    ]
    
    PREFERENCE_INDICATORS = [
        (r"i prefer (.+?)(?:\s+over|\s+than|\s+instead|\.|$)", "preference"),
        (r"i like (.+?)(?:\s+for|\s+because|\.|$)", "positive"),
        (r"i dislike (.+?)(?:\s+because|\.|$)", "negative"),
        (r"i hate (.+?)(?:\s+because|\.|$)", "negative"),
        (r"i love (.+?)(?:\s+because|\.|$)", "positive"),
        (r"my favorite (.+?) is (.+?)(?:\.|$)", "favorite"),
        (r"i usually (.+?)(?:\.|$)", "habit"),
        (r"i always (.+?)(?:\.|$)", "habit"),
        (r"i never (.+?)(?:\.|$)", "negative"),
    ]
    
    MAX_MESSAGE_LENGTHS = 500
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._data: UserModelData = UserModelData(
            user_id="default",
            created_at=time.time(),
            last_interaction=time.time(),
        )
        
        self._lock = threading.RLock()
        self._message_lengths: list[int] = []
        
        # Load existing data
        self._load()
    
    def _load(self) -> None:
        """Load user model from disk."""
        model_file = self.storage_path / "user_model.json"
        if model_file.exists():
            try:
                data = read_json_with_integrity(model_file)
                self._data = UserModelData.from_dict(data)
                logger.debug(f"Loaded user model: {len(self._data.preferences)} preferences")
            except Exception as e:
                logger.warning(f"Failed to load user model: {e}")
    
    def _save(self) -> None:
        """Save user model to disk."""
        model_file = self.storage_path / "user_model.json"
        try:
            atomic_write_json(model_file, self._data.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save user model: {e}")
    
    def update_from_input(self, user_input: str, response: str = "") -> None:
        """Update user model from conversation input.
        
        Args:
            user_input: The user's message
            response: The assistant's response (optional)
        """
        with self._lock:
            self._data.interaction_count += 1
            self._data.last_interaction = time.time()
            
            # Track message length
            self._message_lengths.append(len(user_input))
            if len(self._message_lengths) > self.MAX_MESSAGE_LENGTHS:
                self._message_lengths = self._message_lengths[-self.MAX_MESSAGE_LENGTHS:]
            self._data.average_message_length = sum(self._message_lengths) / len(self._message_lengths)
            
            # Detect technical terms
            self._detect_technical_terms(user_input)
            
            # Detect interests
            self._detect_interests(user_input)
            
            # Detect preferences
            self._detect_preferences(user_input)
            
            # Detect goals
            self._detect_goals(user_input)
            
            # Detect communication style
            self._detect_communication_style(user_input)
            
            # Track topics
            self._track_topics(user_input)
            
            # Prune data to prevent unbounded growth
            self._prune_data()
            
            # Save periodically
            if self._data.interaction_count % 5 == 0:
                self._save()
    
    def _prune_data(self) -> None:
        """Prune data to stay within bounds."""
        # Prune preferences
        if len(self._data.preferences) > MAX_PREFERENCES:
            # Keep highest confidence preferences
            sorted_prefs = sorted(
                self._data.preferences.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
            self._data.preferences = dict(sorted_prefs[:MAX_PREFERENCES])
        
        # Prune interests
        if len(self._data.interests) > MAX_INTERESTS:
            sorted_interests = sorted(
                self._data.interests.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._data.interests = dict(sorted_interests[:MAX_INTERESTS])
        
        # Prune active goals (keep most recent)
        if len(self._data.active_goals) > MAX_ACTIVE_GOALS:
            self._data.active_goals = sorted(
                self._data.active_goals,
                key=lambda g: g.last_mentioned,
                reverse=True
            )[:MAX_ACTIVE_GOALS]
        
        # Prune mentioned projects
        if len(self._data.mentioned_projects) > MAX_MENTIONED_PROJECTS:
            self._data.mentioned_projects = self._data.mentioned_projects[-MAX_MENTIONED_PROJECTS:]
        
        # Prune patterns
        if len(self._data.patterns) > MAX_PATTERNS:
            self._data.patterns = sorted(
                self._data.patterns,
                key=lambda p: p.last_seen,
                reverse=True
            )[:MAX_PATTERNS]
        
        # Prune topics discussed
        if len(self._data.topics_discussed) > MAX_TOPICS_DISCUSSED:
            sorted_topics = sorted(
                self._data.topics_discussed.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._data.topics_discussed = dict(sorted_topics[:MAX_TOPICS_DISCUSSED])
    
    def _detect_technical_terms(self, text: str) -> None:
        """Detect technical term usage."""
        text_lower = text.lower()
        technical_count = sum(1 for term in self.TECHNICAL_TERMS if term in text_lower)
        
        if technical_count > 0:
            # Update running average
            # If user uses technical terms consistently, mark as True
            if technical_count >= 2:
                self._data.uses_technical_terms = True
    
    def _detect_interests(self, text: str) -> None:
        """Detect user interests from text."""
        text_lower = text.lower()
        
        for interest in self.INTEREST_KEYWORDS:
            if interest in text_lower:
                # Increase score
                current = self._data.interests.get(interest, 0.0)
                self._data.interests[interest] = min(1.0, current + 0.1)
    
    def _detect_preferences(self, text: str) -> None:
        """Detect explicit preferences."""
        text_lower = text.lower()
        
        for pattern, pref_type in self.PREFERENCE_INDICATORS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Use hash of full match to avoid truncation collisions
                import hashlib
                unique_key = f"{pref_type}_{hashlib.sha256(match.encode()).hexdigest()[:8]}"
                display_value = match[:100]  # Store full value for display
                
                if unique_key not in self._data.preferences:
                    self._data.preferences[unique_key] = UserPreference(
                        key=unique_key,
                        value=display_value,
                        confidence=0.7,
                        source="explicit",
                        timestamp=time.time(),
                    )
                else:
                    # Increase confidence
                    self._data.preferences[unique_key].confidence = min(1.0, self._data.preferences[unique_key].confidence + 0.1)
                    self._data.preferences[unique_key].access_count += 1
    
    def _detect_goals(self, text: str) -> None:
        """Detect user goals and projects."""
        text_lower = text.lower()
        
        for pattern, group in self.GOAL_INDICATORS:
            match = re.search(pattern, text_lower)
            if match:
                # Extract goal description from capture group
                goal_text = match.group(group).strip()[:100]
                
                if goal_text and len(goal_text) > 2:
                    # Check if already tracking
                    existing = None
                    for goal in self._data.active_goals:
                        if goal_text in goal.description or goal.description in goal_text:
                            existing = goal
                            break
                    
                    if existing:
                        existing.last_mentioned = time.time()
                        existing.mention_count += 1
                    else:
                        self._data.active_goals.append(UserGoal(
                            description=goal_text,
                            status="active",
                            first_mentioned=time.time(),
                            last_mentioned=time.time(),
                        ))
    
    def _detect_communication_style(self, text: str) -> None:
        """Detect communication style preferences."""
        # Check for emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
        
        if emoji_pattern.search(text):
            self._data.uses_emojis = True
        
        # Check for followup questions
        question_count = text.count('?')
        if question_count > 0:
            self._data.asks_followup_questions = True
        
        # Determine style based on patterns
        technical_score = 1.0 if self._data.uses_technical_terms else 0.0
        formal_indicators = len(re.findall(r'\b(would|could|should|may I|please)\b', text.lower()))
        casual_indicators = len(re.findall(r'\b(yeah|yep|nope|gonna|wanna|kinda)\b', text.lower()))
        
        if self._data.uses_technical_terms and technical_score > 0.5:
            self._data.communication_style = CommunicationStyle.TECHNICAL
        elif formal_indicators > casual_indicators * 2:
            self._data.communication_style = CommunicationStyle.FORMAL
        elif casual_indicators > formal_indicators * 2:
            self._data.communication_style = CommunicationStyle.CASUAL
    
    def _track_topics(self, text: str) -> None:
        """Track topics discussed."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Filter out common words
        stop_words = {'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been',
                      'would', 'could', 'should', 'might', 'about', 'after', 'before',
                      'there', 'their', 'what', 'when', 'where', 'which', 'while'}
        
        for word in words:
            if word not in stop_words:
                self._data.topics_discussed[word] = self._data.topics_discussed.get(word, 0) + 1
    
    def get_preferences(self) -> dict[str, str]:
        """Get user preferences as simple key-value dict.
        
        Returns:
            Dict of preference keys to values
        """
        with self._lock:
            return {k: v.value for k, v in self._data.preferences.items()}
    
    def get_communication_style(self) -> tuple[CommunicationStyle, DetailLevel]:
        """Get user's communication style preferences.
        
        Returns:
            Tuple of (CommunicationStyle, DetailLevel)
        """
        with self._lock:
            return self._data.communication_style, self._data.detail_preference
    
    def get_interests(self, limit: int = 10) -> list[str]:
        """Get user's top interests.
        
        Args:
            limit: Maximum interests to return
            
        Returns:
            List of interest topics sorted by score
        """
        with self._lock:
            sorted_interests = sorted(
                self._data.interests.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [topic for topic, score in sorted_interests[:limit]]
    
    def get_expertise_areas(self) -> dict[str, str]:
        """Get user's expertise areas.
        
        Returns:
            Dict of domain -> level
        """
        with self._lock:
            return {domain: area.level for domain, area in self._data.expertise_areas.items()}
    
    def get_active_goals(self) -> list[dict]:
        """Get user's active goals.
        
        Returns:
            List of goal dictionaries
        """
        with self._lock:
            return [
                {
                    "description": goal.description,
                    "status": goal.status,
                    "mention_count": goal.mention_count,
                }
                for goal in self._data.active_goals
                if goal.status == "active"
            ]
    
    def get_user_context(self) -> str:
        """Get formatted user context for prompts.
        
        Returns:
            XML-formatted user context string
        """
        with self._lock:
            parts = []
            
            # Communication style
            if self._data.communication_style != CommunicationStyle.MIXED:
                parts.append(f"<style>{self._data.communication_style.value}</style>")
            
            if self._data.detail_preference != DetailLevel.MEDIUM:
                parts.append(f"<detail>{self._data.detail_preference.value}</detail>")
            
            # Technical preference
            if self._data.uses_technical_terms:
                parts.append("<uses_technical>true</uses_technical>")
            
            # Top interests
            top_interests = self.get_interests(limit=3)
            if top_interests:
                parts.append(f"<interests>{', '.join(top_interests)}</interests>")
            
            # Active goals
            active_goals = self.get_active_goals()
            if active_goals:
                goal_strs = [g["description"][:50] for g in active_goals[:2]]
                parts.append(f"<goals>{'; '.join(goal_strs)}</goals>")
            
            # Recent topics
            recent_topics = sorted(
                self._data.topics_discussed.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            if recent_topics:
                parts.append(f"<recent_topics>{', '.join(t[0] for t in recent_topics)}</recent_topics>")
            
            # Preferences
            if self._data.preferences:
                top_prefs = sorted(
                    self._data.preferences.items(),
                    key=lambda x: x[1].confidence,
                    reverse=True
                )[:3]
                pref_strs = [f"{k.split('_', 1)[0]}: {v.value[:30]}" for k, v in top_prefs]
                if pref_strs:
                    parts.append(f"<preferences>{' | '.join(pref_strs)}</preferences>")
            
            return "\n".join(parts) if parts else ""
    
    def get_summary(self) -> str:
        """Get a human-readable summary of user model.
        
        Returns:
            Summary string
        """
        with self._lock:
            lines = [
                "User Model Summary",
                "==================",
                f"Interactions: {self._data.interaction_count}",
                f"Style: {self._data.communication_style.value}",
                f"Detail preference: {self._data.detail_preference.value}",
                f"Uses technical terms: {self._data.uses_technical_terms}",
                f"Uses emojis: {self._data.uses_emojis}",
                f"Avg message length: {self._data.average_message_length:.0f} chars",
                "",
                "Top interests:",
            ]
            
            for interest in self.get_interests(limit=5):
                lines.append(f"  - {interest}")
            
            if self._data.active_goals:
                lines.append("")
                lines.append("Active goals:")
                for goal in self._data.active_goals[:5]:
                    lines.append(f"  - {goal.description[:50]}")
            
            return "\n".join(lines)
    
    def update_goal_progress(self, goal_description: str, progress: str) -> None:
        """Update progress on a goal.
        
        Args:
            goal_description: Description of the goal
            progress: Progress note to add
        """
        with self._lock:
            for goal in self._data.active_goals:
                if goal_description.lower() in goal.description.lower():
                    goal.progress_notes.append(progress)
                    goal.last_mentioned = time.time()
                    self._save()
                    return
    
    def complete_goal(self, goal_description: str) -> None:
        """Mark a goal as completed.
        
        Args:
            goal_description: Description of the goal
        """
        with self._lock:
            for goal in self._data.active_goals:
                if goal_description.lower() in goal.description.lower():
                    goal.status = "completed"
                    self._save()
                    return
    
    def predict_interest(self, topic: str) -> float:
        """Predict how interested user would be in topic.
        
        Args:
            topic: Topic to predict interest for
            
        Returns:
            Interest score 0.0-1.0
        """
        with self._lock:
            topic_lower = topic.lower()
            
            # Direct match
            for interest, score in self._data.interests.items():
                if interest in topic_lower or topic_lower in interest:
                    return score
            
            # Related words
            words = set(topic_lower.split())
            total_score = 0.0
            match_count = 0
            
            for interest, score in self._data.interests.items():
                interest_words = set(interest.split())
                overlap = len(words & interest_words)
                if overlap > 0:
                    total_score += score * (overlap / max(len(words), len(interest_words)))
                    match_count += 1
            
            if match_count > 0:
                return total_score / match_count
            
            # Default based on interaction count
            return min(0.5, self._data.interaction_count * 0.01)
    
    def reset(self) -> None:
        """Reset user model for new user."""
        with self._lock:
            self._data = UserModelData(
                user_id="default",
                created_at=time.time(),
                last_interaction=time.time(),
            )
            self._message_lengths = []
            self._save()


def get_user_model_manager() -> UserModelManager:
    """Get the singleton UserModelManager instance.
    
    Returns:
        UserModelManager singleton
    """
    global _user_model
    if _user_model is None:
        with _lock:
            if _user_model is None:
                _user_model = UserModelManager()
    return _user_model


def reset_user_model() -> None:
    """Reset the singleton for testing."""
    global _user_model
    with _lock:
        if _user_model is not None:
            _user_model.reset()
        _user_model = None


def analyze_user_preferences(text: str) -> dict[str, str]:
    """Convenience function to analyze user input for preferences.
    
    Args:
        text: User input text
        
    Returns:
        Dict of preferences found
    """
    manager = get_user_model_manager()
    manager.update_from_input(text)
    return manager.get_preferences()


def get_user_context_for_prompt() -> str:
    """Convenience function to get user context.
    
    Returns:
        User context string
    """
    return get_user_model_manager().get_user_context()