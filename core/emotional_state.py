"""Emotional State Awareness for Jarvis.

This module tracks emotional/mood context in conversations to enable
empathetic and context-appropriate responses.

Key features:
- Detect user mood from text (excited, frustrated, curious, etc.)
- Track engagement level and sentiment trajectory
- Adjust response style based on emotional context
- Maintain emotional history for conversation continuity
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton
_emotional_state: Optional["EmotionalStateManager"] = None
_lock = threading.Lock()


class UserMood(Enum):
    """Possible user moods detected from conversation."""
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    CONFUSED = "confused"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    THOUGHTFUL = "thoughtful"
    URGENT = "urgent"


class EngagementLevel(Enum):
    """User engagement levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EmotionalSignals:
    """Signals extracted from user input."""
    mood: UserMood = UserMood.NEUTRAL
    sentiment: float = 0.0  # -1.0 to 1.0
    engagement: EngagementLevel = EngagementLevel.MEDIUM
    is_question: bool = False
    is_statement: bool = False
    is_command: bool = False
    has_emotional_words: bool = False
    exclamation_count: int = 0
    question_count: int = 0
    word_count: int = 0
    caps_ratio: float = 0.0
    avg_word_length: float = 0.0
    technical_terms: int = 0
    emotional_keywords: list[str] = field(default_factory=list)


@dataclass
class ResponseStyle:
    """Recommended response style based on emotional context."""
    detail_level: str = "medium"  # "short", "medium", "detailed"
    tone: str = "neutral"  # "warm", "neutral", "formal"
    energy: str = "medium"  # "low", "medium", "high"
    should_match_energy: bool = True
    should_be_careful: bool = False
    should_explore: bool = False
    should_provide_examples: bool = True
    follow_up_questions: bool = True
    technical_depth: str = "medium"  # "none", "low", "medium", "high"


@dataclass
class ConversationTurn:
    """A single conversation turn for emotional tracking."""
    timestamp: float
    user_input: str
    signals: EmotionalSignals
    response_length: int = 0
    topic: str = ""


class EmotionalStateManager:
    """Manages emotional state tracking across conversations.
    
    This class:
    1. Analyzes user input for emotional signals
    2. Tracks emotional trajectory over conversation
    3. Recommends response styles
    4. Maintains emotional context history
    
    Usage:
        manager = get_emotional_state_manager()
        signals = manager.analyze_input("I'm so excited about this project!")
        style = manager.get_response_style()
    """
    
    MAX_HISTORY = 50  # Maximum turns to track
    MIN_TURNS_FOR_PATTERN = 3  # Minimum turns before pattern detection
    
    # Mood detection patterns
    MOOD_PATTERNS = {
        UserMood.EXCITED: [
            r"\b(amazing|awesome|incredible|excited|exciting|wow|omg|fantastic)\b",
            r"\b(can't wait|so cool|love this|yay|hype)\b",
            r"!!+",
        ],
        UserMood.FRUSTRATED: [
            r"\b(frustrated|frustrating|annoying|annoyed|stupid|doesn't work|broken|ugh)\b",
            r"\b(why won't|not working|this sucks|terrible|awful)\b",
            r"\b(keep getting|keeps failing|again and again)\b",
        ],
        UserMood.CURIOUS: [
            r"\b(wonder|curious|interesting|how does|what if)\b",
            r"\b(tell me more|explain|understand|learn|explore)\b",
            r"\?.*\?",  # Multiple questions
        ],
        UserMood.CONFUSED: [
            r"\b(confused|don't understand|not sure|unclear|lost)\b",
            r"\b(what do you mean|huh|i don't get|makes no sense)\b",
            r"\b(help me understand|can you explain)\b",
        ],
        UserMood.HAPPY: [
            r"\b(happy|glad|pleased|delighted|wonderful|great)\b",
            r"\b(thanks|thank you|appreciate|grateful)\b",
            r"\b(working perfectly|love it|exactly what|i like)\b",
        ],
        UserMood.SAD: [
            r"\b(sad|unhappy|disappointed|sorry|regret)\b",
            r"\b(miss|missing|lost|gone|failed)\b",
            r"\b(feeling down|not good|upset|i'm sad)\b",
        ],
        UserMood.ANGRY: [
            r"\b(angry|mad|furious|outraged|hate)\b",
            r"\b(this is stupid|idiot|worthless|garbage|trash)\b",
            r"\b(i'?m angry|i'm so angry)\b",
        ],
        UserMood.THOUGHTFUL: [
            r"\b(thinking|considering|wondering|reflecting|pondering)\b",
            r"\b(maybe|perhaps|possibly|might|could)\b",
            r"\b(let me think|hmm|interesting point)\b",
        ],
        UserMood.URGENT: [
            r"\b(urgent|immediately|asap|right now|quickly)\b",
            r"\b(need this now|deadline|time sensitive)\b",
            r"\bhurry\b",
        ],
    }
    
    # Sentiment keywords with weights
    SENTIMENT_KEYWORDS = {
        # Positive
        "love": 0.8, "great": 0.6, "awesome": 0.7, "amazing": 0.8,
        "fantastic": 0.7, "excellent": 0.7, "wonderful": 0.7,
        "happy": 0.6, "glad": 0.5, "thanks": 0.4, "thank": 0.4,
        "perfect": 0.7, "brilliant": 0.7, "helpful": 0.5,
        "working": 0.3, "success": 0.6, "good": 0.4,
        # Negative
        "hate": -0.8, "stupid": -0.7, "terrible": -0.7, "awful": -0.7,
        "bad": -0.4, "wrong": -0.4, "broken": -0.5, "fail": -0.5,
        "frustrated": -0.6, "annoying": -0.5, "angry": -0.7,
        "confused": -0.3, "lost": -0.3, "problem": -0.3,
        "error": -0.4, "bug": -0.4, "issue": -0.2,
    }
    
    # Engagement indicators
    ENGAGEMENT_INDICATORS = {
        "high": [
            r"\?",
            r"\b(tell me|explain|show me|help me|let's|how about)\b",
            r"\!",
        ],
        "low": [
            r"^(ok|okay|k|yeah|yes|no|nope|sure|fine)$",
            r"^(ok|okay|k|yeah|yes|no|nope|sure|fine)\.$",
        ],
    }

    def __init__(self):
        self._history: deque[ConversationTurn] = deque(maxlen=self.MAX_HISTORY)
        self._current_mood: UserMood = UserMood.NEUTRAL
        self._current_sentiment: float = 0.0
        self._current_engagement: EngagementLevel = EngagementLevel.MEDIUM
        self._mood_history: deque[UserMood] = deque(maxlen=10)
        self._sentiment_trajectory: deque[float] = deque(maxlen=20)
        self._lock = threading.RLock()
    
    def analyze_input(self, user_input: str) -> EmotionalSignals:
        """Analyze user input for emotional signals.
        
        Args:
            user_input: The user's message
            
        Returns:
            EmotionalSignals with detected mood, sentiment, etc.
        """
        signals = EmotionalSignals()
        
        if not user_input or not user_input.strip():
            return signals
        
        text = user_input.strip()
        words = text.split()
        signals.word_count = len(words)
        
        # Detect mood
        signals.mood = self._detect_mood(text)
        signals.emotional_keywords = self._extract_emotional_keywords(text)
        signals.has_emotional_words = len(signals.emotional_keywords) > 0
        
        # Calculate sentiment
        signals.sentiment = self._calculate_sentiment(text)
        
        # Detect engagement
        signals.engagement = self._detect_engagement(text)
        
        # Count punctuation
        signals.exclamation_count = text.count('!')
        signals.question_count = text.count('?')
        
        # Detect patterns
        signals.is_question = '?' in text or any(
            text.lower().startswith(q) 
            for q in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can ', 'could ', 'would ', 'should ', 'is ', 'are ', 'do ', 'does ']
        )
        signals.is_statement = not signals.is_question and '.' in text
        signals.is_command = any(
            text.lower().startswith(cmd)
            for cmd in ['do ', 'run ', 'create ', 'make ', 'build ', 'fix ', 'delete ', 'remove ']
        )
        
        # Calculate caps ratio
        caps_count = sum(1 for c in text if c.isupper())
        alpha_count = sum(1 for c in text if c.isalpha())
        signals.caps_ratio = caps_count / alpha_count if alpha_count > 0 else 0.0
        
        # Average word length
        if words:
            signals.avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Count technical terms
        technical_patterns = [
            r'\b(function|class|method|variable|API|code|error|debug)\b',
            r'\b(database|server|client|request|response|endpoint)\b',
            r'\b(algorithm|data|structure|array|object|string)\b',
        ]
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            signals.technical_terms += len(matches)
        
        # Update internal state
        with self._lock:
            self._current_mood = signals.mood
            self._current_sentiment = signals.sentiment
            self._current_engagement = signals.engagement
            self._mood_history.append(signals.mood)
            self._sentiment_trajectory.append(signals.sentiment)
            
            # Add to history
            turn = ConversationTurn(
                timestamp=time.time(),
                user_input=text,
                signals=signals,
            )
            self._history.append(turn)
        
        return signals
    
    def _detect_mood(self, text: str) -> UserMood:
        """Detect user mood from text.
        
        Args:
            text: User input text
            
        Returns:
            Detected UserMood
        """
        text_lower = text.lower()
        mood_scores: dict[UserMood, int] = {}
        
        for mood, patterns in self.MOOD_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            if score > 0:
                mood_scores[mood] = score
        
        if not mood_scores:
            return UserMood.NEUTRAL
        
        # Return mood with highest score
        return max(mood_scores, key=mood_scores.get)
    
    def _extract_emotional_keywords(self, text: str) -> list[str]:
        """Extract emotional keywords from text.
        
        Args:
            text: User input text
            
        Returns:
            List of emotional keywords found
        """
        text_lower = text.lower()
        keywords = []
        
        for keyword in self.SENTIMENT_KEYWORDS:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score from -1.0 to 1.0.
        
        Args:
            text: User input text
            
        Returns:
            Sentiment score
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return 0.0
        
        score = 0.0
        count = 0
        
        for word in words:
            if word in self.SENTIMENT_KEYWORDS:
                score += self.SENTIMENT_KEYWORDS[word]
                count += 1
        
        # Normalize to -1 to 1
        if count == 0:
            return 0.0
        
        return max(-1.0, min(1.0, score / count))
    
    def _detect_engagement(self, text: str) -> EngagementLevel:
        """Detect user engagement level.
        
        Args:
            text: User input text
            
        Returns:
            EngagementLevel
        """
        text_lower = text.lower().strip()
        
        # Check for low engagement patterns
        for pattern in self.ENGAGEMENT_INDICATORS["low"]:
            if re.match(pattern, text_lower):
                return EngagementLevel.LOW
        
        # Check for high engagement patterns
        high_count = 0
        for pattern in self.ENGAGEMENT_INDICATORS["high"]:
            if re.search(pattern, text_lower):
                high_count += 1
        
        if high_count >= 2:
            return EngagementLevel.HIGH
        
        # Count words as proxy
        words = text.split()
        if len(words) >= 15:
            return EngagementLevel.HIGH
        elif len(words) >= 5:
            return EngagementLevel.MEDIUM
        else:
            return EngagementLevel.LOW
    
    def get_response_style(self) -> ResponseStyle:
        """Get recommended response style based on emotional context.
        
        Returns:
            ResponseStyle with recommendations
        """
        with self._lock:
            style = ResponseStyle()
            
            # Adjust based on mood
            if self._current_mood == UserMood.EXCITED:
                style.energy = "high"
                style.tone = "warm"
                style.should_match_energy = True
                style.detail_level = "medium"
            
            elif self._current_mood == UserMood.FRUSTRATED:
                style.energy = "low"
                style.tone = "neutral"
                style.should_be_careful = True
                style.detail_level = "short"
                style.follow_up_questions = False
            
            elif self._current_mood == UserMood.CURIOUS:
                style.energy = "medium"
                style.tone = "warm"
                style.should_explore = True
                style.detail_level = "detailed"
                style.follow_up_questions = True
            
            elif self._current_mood == UserMood.CONFUSED:
                style.energy = "medium"
                style.tone = "warm"
                style.should_provide_examples = True
                style.detail_level = "medium"
                style.follow_up_questions = True
            
            elif self._current_mood == UserMood.ANGRY:
                style.energy = "low"
                style.tone = "neutral"
                style.should_be_careful = True
                style.detail_level = "short"
                style.follow_up_questions = False
            
            elif self._current_mood == UserMood.SAD:
                style.energy = "low"
                style.tone = "warm"
                style.should_be_careful = True
                style.detail_level = "medium"
            
            elif self._current_mood == UserMood.HAPPY:
                style.energy = "medium"
                style.tone = "warm"
                style.should_match_energy = True
            
            elif self._current_mood == UserMood.THOUGHTFUL:
                style.energy = "medium"
                style.tone = "neutral"
                style.detail_level = "detailed"
                style.should_explore = True
            
            elif self._current_mood == UserMood.URGENT:
                style.energy = "high"
                style.tone = "neutral"
                style.detail_level = "short"
                style.follow_up_questions = False
            
            # Adjust based on engagement
            if self._current_engagement == EngagementLevel.HIGH:
                style.detail_level = "detailed"
                style.follow_up_questions = True
            elif self._current_engagement == EngagementLevel.LOW:
                style.detail_level = "short"
                style.follow_up_questions = False
            
            return style
    
    def get_emotional_context(self) -> str:
        """Get emotional context for prompts.
        
        Returns:
            XML-formatted emotional context
        """
        with self._lock:
            parts = []
            
            # Current mood
            parts.append(f"<mood>{self._current_mood.value}</mood>")
            
            # Sentiment
            sentiment_desc = "neutral"
            if self._current_sentiment > 0.3:
                sentiment_desc = "positive"
            elif self._current_sentiment < -0.3:
                sentiment_desc = "negative"
            parts.append(f"<sentiment>{sentiment_desc}</sentiment>")
            
            # Engagement
            parts.append(f"<engagement>{self._current_engagement.value}</engagement>")
            
            # Response guidance
            style = self.get_response_style()
            if style.should_be_careful:
                parts.append("<careful>User may be frustrated. Be helpful, concise.</careful>")
            elif style.should_explore:
                parts.append("<exploratory>User is curious. Can explore together.</exploratory>")
            elif style.should_match_energy:
                parts.append("<match_energy>Match user's energy level.</match_energy>")
            
            return "\n".join(parts)
    
    def get_mood_history(self, limit: int = 5) -> list[UserMood]:
        """Get recent mood history.
        
        Args:
            limit: Maximum moods to return
            
        Returns:
            List of recent moods
        """
        with self._lock:
            return list(self._mood_history)[-limit:]
    
    def get_sentiment_trajectory(self, limit: int = 10) -> list[float]:
        """Get sentiment trajectory over time.
        
        Args:
            limit: Maximum values to return
            
        Returns:
            List of sentiment values
        """
        with self._lock:
            return list(self._sentiment_trajectory)[-limit:]
    
    def should_show_empathy(self) -> bool:
        """Determine if empathy should be shown.
        
        Returns:
            True if should show empathy
        """
        with self._lock:
            # Show empathy for negative sentiment
            if self._current_sentiment < -0.3:
                return True
            
            # Show empathy for certain moods
            empathy_moods = {
                UserMood.FRUSTRATED,
                UserMood.SAD,
                UserMood.ANGRY,
                UserMood.CONFUSED,
            }
            
            return self._current_mood in empathy_moods
    
    def get_empathy_message(self) -> str | None:
        """Get appropriate empathy message if needed.
        
        Returns:
            Empathy message or None
        """
        if not self.should_show_empathy():
            return None
        
        with self._lock:
            if self._current_mood == UserMood.FRUSTRATED:
                return "I understand this can be frustrating. Let me help."
            elif self._current_mood == UserMood.SAD:
                return "I'm sorry to hear that. How can I help?"
            elif self._current_mood == UserMood.ANGRY:
                return "I understand you're upset. Let's work through this together."
            elif self._current_mood == UserMood.CONFUSED:
                return "I can help clarify this. Let me explain."
        
        return None
    
    def summarize_emotional_state(self) -> str:
        """Get a summary of current emotional state.
        
        Returns:
            Human-readable summary
        """
        with self._lock:
            lines = [
                f"Current mood: {self._current_mood.value}",
                f"Sentiment: {self._current_sentiment:.2f} ({'positive' if self._current_sentiment > 0.3 else 'negative' if self._current_sentiment < -0.3 else 'neutral'})",
                f"Engagement: {self._current_engagement.value}",
                f"Recent moods: {', '.join(m.value for m in list(self._mood_history)[-3:])}",
                f"Turn count: {len(self._history)}",
            ]
            
            style = self.get_response_style()
            lines.append(f"Recommended style: {style.tone} tone, {style.detail_level} detail")
            
            return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset emotional state for new conversation."""
        with self._lock:
            self._history.clear()
            self._current_mood = UserMood.NEUTRAL
            self._current_sentiment = 0.0
            self._current_engagement = EngagementLevel.MEDIUM
            self._mood_history.clear()
            self._sentiment_trajectory.clear()
    
    def update_from_physiology(self, arousal: float = 0.5, valence: float = 0.0) -> None:
        """Update emotional state from physiological signals (bio-homeostasis bridge).
        
        Arousal modulates engagement: high arousal → high engagement.
        Valence shifts sentiment toward positive or negative.
        
        Args:
            arousal: Physiological arousal level (0.0-1.0)
            valence: Pleasure/displeasure signal (-1.0 to 1.0)
        """
        with self._lock:
            if arousal > 0.7:
                self._current_engagement = EngagementLevel.HIGH
            elif arousal < 0.3:
                self._current_engagement = EngagementLevel.LOW
            
            blended_sentiment = self._current_sentiment * 0.8 + valence * 0.2
            self._current_sentiment = max(-1.0, min(1.0, blended_sentiment))
            
            if valence < -0.3 and self._current_mood == UserMood.NEUTRAL:
                self._current_mood = UserMood.FRUSTRATED
            elif valence > 0.3 and self._current_mood == UserMood.NEUTRAL:
                self._current_mood = UserMood.HAPPY


def get_emotional_state_manager() -> EmotionalStateManager:
    """Get the singleton EmotionalStateManager instance.
    
    Returns:
        EmotionalStateManager singleton
    """
    global _emotional_state
    if _emotional_state is None:
        with _lock:
            if _emotional_state is None:
                _emotional_state = EmotionalStateManager()
    return _emotional_state


def reset_emotional_state() -> None:
    """Reset the singleton for testing."""
    global _emotional_state
    with _lock:
        if _emotional_state is not None:
            _emotional_state.reset()
        _emotional_state = None


def analyze_emotional_signals(text: str) -> EmotionalSignals:
    """Convenience function to analyze emotional signals.
    
    Args:
        text: User input text
        
    Returns:
        EmotionalSignals
    """
    return get_emotional_state_manager().analyze_input(text)


def get_response_style_for_context() -> ResponseStyle:
    """Convenience function to get response style.
    
    Returns:
        ResponseStyle
    """
    return get_emotional_state_manager().get_response_style()


def get_emotional_context_for_prompt() -> str:
    """Convenience function to get emotional context.
    
    Returns:
        Emotional context string
    """
    return get_emotional_state_manager().get_emotional_context()