"""Self-Reflection - Analyze own behavior and improve.

This module enables Jarvis to reflect on conversations and learn from
interactions, improving response quality over time.
"""

import logging
import threading
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from core.helpers import atomic_write_json, read_json_with_integrity
from enum import Enum

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = Path.home() / ".jarvis" / "reflection"

# Memory bounds
MAX_CONVERSATION_HISTORY = 50
MAX_MISTAKES = 100
MAX_QUALITY_SCORES = 200
MAX_TOPICS_TRACKED = 50


class ResponseQuality(Enum):
    """Quality rating for responses."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNCLEAR = "unclear"


@dataclass
class Mistake:
    """Represents a mistake made by Jarvis."""
    timestamp: float
    context: str  # What was the situation
    action: str  # What Jarvis did
    issue: str  # What went wrong
    improvement: str  # How to improve
    category: str  # Category of mistake
    resolved: bool = False


@dataclass
class ConversationAnalysis:
    """Analysis of a conversation."""
    satisfaction_score: float  # 0.0-1.0
    user_engagement: str  # "high", "medium", "low"
    response_quality: ResponseQuality
    successful_interactions: int
    issues_encountered: int
    topics_covered: list[str]
    improvements_suggested: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflectionResult:
    """Result of self-reflection on a conversation."""
    analysis: ConversationAnalysis
    mistakes: list[Mistake]
    improvements: list[str]
    patterns_noticed: list[str]


class SelfReflection:
    """Reflect on own behavior and improve."""
    
    # Patterns indicating user dissatisfaction
    DISSATISFACTION_PATTERNS = [
        r"that'?s? not what i (wanted|asked|meant)",
        r"that'?s? wrong",
        r"that doesn'?t help",
        r"i don'?t understand",
        r"not (really|exactly|quite)",
        r"try again",
        r"no,?",
        r"(?:actually|actually,)",
    ]
    
    # Patterns indicating user satisfaction
    SATISFACTION_PATTERNS = [
        r"thanks?",
        r"perfect",
        r"exactly",
        r"that'?s? (great|helpful|perfect)",
        r"(?:cool|awesome|great)",
        r"that works",
    ]
    
    # Patterns indicating confusion
    CONFUSION_PATTERNS = [
        r"what do you mean",
        r"i don'?t get it",
        r"(?:can you|please) explain",
        r"huh\??",
        r"confusing",
    ]
    
    # Category mapping
    MISTAKE_CATEGORIES = {
        "understanding": "Failed to understand user intent",
        "accuracy": "Provided incorrect information",
        "relevance": "Response not relevant to context",
        "clarity": "Response unclear or confusing",
        "completeness": "Response incomplete",
        "style": "Response style inappropriate",
    }
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._conversation_history: list[dict] = []
        self._mistakes: list[Mistake] = []
        self._quality_scores: list[tuple[float, float]] = []  # (score, timestamp)
        self._improvement_count: dict[str, int] = {}
        
        self._load()
    
    def _load(self) -> None:
        """Load reflection data from disk."""
        mistakes_file = self.storage_path / "mistakes.json"
        if mistakes_file.exists():
            try:
                data = read_json_with_integrity(mistakes_file)
                self._mistakes = [
                    Mistake(**m) for m in data.get("mistakes", [])
                ]
            except Exception as e:
                logger.warning(f"Failed to load mistakes: {e}")
        
        quality_file = self.storage_path / "quality.json"
        if quality_file.exists():
            try:
                data = read_json_with_integrity(quality_file)
                self._quality_scores = [
                    (s, t) for s, t in data.get("scores", [])
                ]
            except Exception as e:
                logger.warning(f"Failed to load quality scores: {e}")
    
    def _save(self) -> None:
        """Save reflection data to disk."""
        mistakes_file = self.storage_path / "mistakes.json"
        with self._lock:
            mistakes_data = [
                {
                    "timestamp": m.timestamp,
                    "context": m.context,
                    "action": m.action,
                    "issue": m.issue,
                    "improvement": m.improvement,
                    "category": m.category,
                    "resolved": m.resolved
                }
                for m in self._mistakes[-MAX_MISTAKES:]
            ]
            quality_data = [
                [s, t] for s, t in self._quality_scores[-100:]
            ]
        try:
            atomic_write_json(mistakes_file, {"mistakes": mistakes_data})
        except Exception as e:
            logger.warning(f"Failed to save mistakes: {e}")
        
        quality_file = self.storage_path / "quality.json"
        try:
            atomic_write_json(quality_file, {"scores": quality_data})
        except Exception as e:
            logger.warning(f"Failed to save quality scores: {e}")
    
    def record_conversation_turn(self, user_input: str, jarvis_response: str):
        """Record a conversation turn for later analysis.
        
        Args:
            user_input: What the user said
            jarvis_response: What Jarvis responded
        """
        with self._lock:
            self._conversation_history.append({
                "user": user_input,
                "jarvis": jarvis_response,
                "timestamp": time.time()
            })
            
            # Keep only last MAX_CONVERSATION_HISTORY turns
            if len(self._conversation_history) > MAX_CONVERSATION_HISTORY:
                self._conversation_history = self._conversation_history[-MAX_CONVERSATION_HISTORY:]
    
    def analyze_conversation(self, turns: list[dict] = None) -> ReflectionResult:
        """Analyze how a conversation went.
        
        Args:
            turns: Conversation turns to analyze (uses stored history if None)
            
        Returns:
            ReflectionResult with analysis and suggestions
        """
        turns = turns or self._conversation_history
        
        if not turns:
            return ReflectionResult(
                analysis=ConversationAnalysis(
                    satisfaction_score=0.5,
                    user_engagement="unclear",
                    response_quality=ResponseQuality.UNCLEAR,
                    successful_interactions=0,
                    issues_encountered=0,
                    topics_covered=[]
                ),
                mistakes=[],
                improvements=[],
                patterns_noticed=[]
            )
        
        # Analyze satisfaction
        satisfaction_signals = []
        for turn in turns:
            user_msg = turn.get("user", "").lower()
            
            # Check for dissatisfaction
            for pattern in self.DISSATISFACTION_PATTERNS:
                if re.search(pattern, user_msg):
                    satisfaction_signals.append(-0.2)
            
            # Check for satisfaction
            for pattern in self.SATISFACTION_PATTERNS:
                if re.search(pattern, user_msg):
                    satisfaction_signals.append(0.2)
            
            # Check for confusion
            for pattern in self.CONFUSION_PATTERNS:
                if re.search(pattern, user_msg):
                    satisfaction_signals.append(-0.1)
        
        satisfaction_score = sum(satisfaction_signals) / len(turns) if turns else 0.5
        satisfaction_score = max(0.0, min(1.0, 0.5 + satisfaction_score))
        
        # Determine engagement
        avg_turn_length = sum(len(t.get("user", "")) for t in turns) / len(turns) if turns else 0
        if avg_turn_length > 100:
            engagement = "high"
        elif avg_turn_length > 50:
            engagement = "medium"
        else:
            engagement = "low"
        
        # Determine response quality
        quality_threshold = satisfaction_score
        if quality_threshold > 0.8:
            quality = ResponseQuality.EXCELLENT
        elif quality_threshold > 0.6:
            quality = ResponseQuality.GOOD
        elif quality_threshold > 0.4:
            quality = ResponseQuality.ACCEPTABLE
        else:
            quality = ResponseQuality.POOR
        
        # Extract topics
        topics = []
        topic_pattern = re.compile(r'\b[a-z]{4,}\b')
        for turn in turns:
            matches = topic_pattern.findall(turn.get("user", "").lower())
            topics.extend(matches)
        
        # Get most common topics
        from collections import Counter
        topic_counts = Counter(topics)
        topics_covered = [t for t, _ in topic_counts.most_common(5)]
        
        # Identify mistakes
        mistakes = self._identify_mistakes(turns)
        
        # Generate improvements
        improvements = self._generate_improvements(mistakes, satisfaction_score)
        
        # Notice patterns
        patterns = self._notice_patterns(turns)
        
        analysis = ConversationAnalysis(
            satisfaction_score=satisfaction_score,
            user_engagement=engagement,
            response_quality=quality,
            successful_interactions=sum(1 for s in satisfaction_signals if s > 0),
            issues_encountered=sum(1 for s in satisfaction_signals if s < 0),
            topics_covered=topics_covered
        )
        
        # Record quality score
        with self._lock:
            self._quality_scores.append((satisfaction_score, time.time()))
            if len(self._quality_scores) > MAX_QUALITY_SCORES:
                self._quality_scores = self._quality_scores[-MAX_QUALITY_SCORES:]
            self._save()
        
        return ReflectionResult(
            analysis=analysis,
            mistakes=mistakes,
            improvements=improvements,
            patterns_noticed=patterns
        )
    
    def _identify_mistakes(self, turns: list[dict]) -> list[Mistake]:
        """Identify mistakes from conversation turns."""
        mistakes = []
        
        for i, turn in enumerate(turns):
            user_msg = turn.get("user", "").lower()
            
            # Check for understanding mistakes
            if any(re.search(p, user_msg) for p in self.DISSATISFACTION_PATTERNS):
                mistakes.append(Mistake(
                    timestamp=turn.get("timestamp", time.time()),
                    context=turns[max(0, i-1)].get("user", "") if i > 0 else "",
                    action=turns[max(0, i-1)].get("jarvis", "") if i > 0 else "",
                    issue="User expressed dissatisfaction",
                    improvement="Better understand user intent before responding",
                    category="understanding"
                ))
            
            # Check for confusion
            if any(re.search(p, user_msg) for p in self.CONFUSION_PATTERNS):
                mistakes.append(Mistake(
                    timestamp=turn.get("timestamp", time.time()),
                    context=turns[max(0, i-1)].get("jarvis", "") if i > 0 else "",
                    action=turns[max(0, i-1)].get("jarvis", "") if i > 0 else "",
                    issue="User was confused",
                    improvement="Provide clearer explanations",
                    category="clarity"
                ))
        
        return mistakes
    
    def _generate_improvements(self, mistakes: list[Mistake], satisfaction: float) -> list[str]:
        """Generate improvement suggestions from mistakes."""
        improvements = []
        
        # Group mistakes by category
        mistake_categories = {}
        for mistake in mistakes:
            category = mistake.category
            if category not in mistake_categories:
                mistake_categories[category] = 0
            mistake_categories[category] += 1
        
        # Generate category-specific improvements
        for category, count in mistake_categories.items():
            if category == "understanding":
                improvements.append(f"Improve understanding ({count}x): Ask clarifying questions when intent is unclear")
            elif category == "clarity":
                improvements.append(f"Improve clarity ({count}x): Provide more structured and detailed explanations")
            elif category == "accuracy":
                improvements.append(f"Improve accuracy ({count}x): Verify information before presenting")
            elif category == "relevance":
                improvements.append(f"Improve relevance ({count}x): Better context awareness needed")
        
        # General improvements based on satisfaction
        if satisfaction < 0.5:
            improvements.append("Overall satisfaction low: Focus on being more helpful and responsive")
        
        return improvements
    
    def _notice_patterns(self, turns: list[dict]) -> list[str]:
        """Notice patterns in conversation."""
        patterns = []
        
        # Check for repeated questions
        questions = []
        for turn in turns:
            user_msg = turn.get("user", "")
            if "?" in user_msg:
                questions.append(user_msg.split("?")[0])
        
        if questions:
            from collections import Counter
            question_counts = Counter(questions)
            for q, count in question_counts.most_common(3):
                if count > 1:
                    patterns.append(f"Repeated question: '{q[:50]}...' asked {count} times")
        
        # Check for topic shifts
        topics = []
        for turn in turns:
            words = set(turn.get("user", "").lower().split())
            topics.append(words)
        
        if len(topics) > 1:
            # Calculate topic overlap between consecutive turns
            overlaps = []
            for i in range(len(topics) - 1):
                if topics[i] and topics[i+1]:
                    overlap = len(topics[i] & topics[i+1]) / max(len(topics[i] | topics[i+1]), 1)
                    overlaps.append(overlap)
            
            if overlaps and sum(overlaps) / len(overlaps) < 0.3:
                patterns.append("Frequent topic shifts detected")
        
        return patterns
    
    def identify_mistakes(self) -> list[Mistake]:
        """Identify mistakes made recently.
        
        Returns:
            List of recent mistakes
        """
        with self._lock:
            # Return mistakes from last 24 hours
            recent_threshold = time.time() - 24 * 60 * 60
            recent_mistakes = [
                m for m in self._mistakes
                if m.timestamp > recent_threshold
            ]
            return recent_mistakes
    
    def generate_improvement_plan(self, mistakes: list[Mistake] = None) -> str:
        """Generate plan to avoid future mistakes.
        
        Args:
            mistakes: Mistakes to plan for (uses identified mistakes if None)
            
        Returns:
            Improvement plan text
        """
        mistakes = mistakes or self.identify_mistakes()
        
        if not mistakes:
            return "No recent mistakes identified. Continue prioritizing clarity, accuracy, and helpfulness."
        
        # Group by category
        by_category = {}
        for mistake in mistakes:
            category = mistake.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(mistake)
        
        # Build plan
        lines = ["# Improvement Plan\n"]
        
        for category, category_mistakes in by_category.items():
            count = len(category_mistakes)
            lines.append(f"## {category.title()} ({count} issue{'s' if count > 1 else ''})")
            lines.append(f"- {self.MISTAKE_CATEGORIES.get(category, category)}")
            
            # Add specific improvements
            for mistake in category_mistakes[:3]:  # Top 3
                lines.append(f"  - {mistake.improvement}")
            lines.append("")
        
        return "\n".join(lines)
    
    def track_response_quality(self, response: str, user_feedback: str = None) -> None:
        """Track quality of responses.
        
        Args:
            response: The response given
            user_feedback: Optional explicit feedback
        """
        quality_score = 0.5  # Default neutral
        
        if user_feedback:
            user_lower = user_feedback.lower()
            
            negation_words = ["not", "isn't", "doesn't", "don't", "didn't", "wasn't", "aren't"]
            has_negation = any(word in user_lower for word in negation_words)
            has_negative = any(word in user_lower for word in ["wrong", "bad", "not helpful", "unclear", "terrible", "awful"])
            has_positive = any(word in user_lower for word in ["great", "perfect", "thanks", "helpful", "excellent", "good", "nice", "ok"])
            
            if has_negation and has_negative and not has_positive:
                quality_score = 0.6
            elif has_negation and has_positive and not has_negative:
                quality_score = 0.3
            elif has_negation:
                quality_score = 0.4
            elif has_positive:
                if any(word in user_lower for word in ["great", "perfect", "thanks", "helpful", "excellent"]):
                    quality_score = 0.8
                else:
                    quality_score = 0.6
            elif has_negative:
                quality_score = 0.2
            elif any(word in user_lower for word in ["confusing", "not sure", "maybe"]):
                quality_score = 0.4
        
        with self._lock:
            self._quality_scores.append((quality_score, time.time()))
            if len(self._quality_scores) > MAX_QUALITY_SCORES:
                self._quality_scores = self._quality_scores[-MAX_QUALITY_SCORES:]
            self._save()
    
    def get_average_quality(self, hours: int = 24) -> float:
        """Get average response quality over time period.
        
        Args:
            hours: Number of hours to average
            
        Returns:
            Average quality score 0.0-1.0
        """
        with self._lock:
            threshold = time.time() - hours * 60 * 60
            recent_scores = [s for s, t in self._quality_scores if t > threshold]
            
            if not recent_scores:
                return 0.5  # No data, assume neutral
            
            return sum(recent_scores) / len(recent_scores)
    
    def get_reflection_context(self) -> str:
        """Get reflection context for LLM prompts.
        
        Returns:
            Formatted context string
        """
        lines = []
        
        # Recent quality
        avg_quality = self.get_average_quality()
        lines.append(f"Recent response quality: {avg_quality:.0%}")
        
        # Recent mistakes
        recent_mistakes = self.identify_mistakes()
        if recent_mistakes:
            lines.append(f"Recent issues ({len(recent_mistakes)}):")
            for mistake in recent_mistakes[:3]:
                lines.append(f"  - {mistake.category}: {mistake.improvement}")
        
        # Improvement focus
        if avg_quality < 0.5:
            lines.append("Focus on: understanding user intent and providing clearer responses")
        elif avg_quality < 0.7:
            lines.append("Focus on: consistency and completeness in responses")
        else:
            lines.append("Continue: maintaining high response quality")
        
        return "<reflection>\n" + "\n".join(lines) + "\n</reflection>"
    
    def analyze_trends(self, days: int = 7) -> dict:
        """Analyze reflection trends over time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        with self._lock:
            threshold = time.time() - days * 24 * 60 * 60
            
            # Get recent data
            recent_mistakes = [m for m in self._mistakes if m.timestamp > threshold]
            recent_scores = [(s, t) for s, t in self._quality_scores if t > threshold]
            
            # Analyze common mistake categories
            category_counts: dict[str, int] = {}
            for mistake in recent_mistakes:
                category_counts[mistake.category] = category_counts.get(mistake.category, 0) + 1
            
            # Sort by frequency
            common_mistakes = sorted(
                category_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Calculate quality trend (improving or declining)
            quality_trend = "stable"
            if len(recent_scores) >= 4:
                # Split into two halves and compare averages
                mid = len(recent_scores) // 2
                first_half = [s for s, _ in recent_scores[:mid]]
                second_half = [s for s, _ in recent_scores[mid:]]
                
                if first_half and second_half:
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    
                    if second_avg > first_avg + 0.05:
                        quality_trend = "improving"
                    elif second_avg < first_avg - 0.05:
                        quality_trend = "declining"
            
            # Calculate average quality
            avg_quality = 0.5
            if recent_scores:
                avg_quality = sum(s for s, _ in recent_scores) / len(recent_scores)
            
            # Suggest improvement areas
            improvement_areas = []
            for category, count in common_mistakes:
                if count >= 3:
                    improvement_areas.append({
                        "category": category,
                        "description": self.MISTAKE_CATEGORIES.get(category, category),
                        "count": count
                    })
            
            return {
                "avg_quality": avg_quality,
                "quality_trend": quality_trend,
                "total_mistakes": len(recent_mistakes),
                "common_mistakes": common_mistakes,
                "improvement_areas": improvement_areas,
                "period_days": days,
                "data_points": len(recent_scores)
            }
    
    def get_trend_summary(self, days: int = 7) -> str:
        """Get human-readable trend summary.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Formatted trend summary
        """
        trends = self.analyze_trends(days)
        
        lines = [f"<trend_analysis period=\"{days} days\">"]
        lines.append(f"Quality: {trends['avg_quality']:.0%} ({trends['quality_trend']})")
        
        if trends['common_mistakes']:
            lines.append("Common issues:")
            for category, count in trends['common_mistakes'][:3]:
                lines.append(f"  - {category}: {count} occurrences")
        
        if trends['improvement_areas']:
            lines.append("Focus areas:")
            for area in trends['improvement_areas'][:3]:
                lines.append(f"  - {area['category']}: {area['description']}")
        
        lines.append("</trend_analysis>")
        return "\n".join(lines)


# Singleton
_reflection: SelfReflection | None = None
_lock = threading.Lock()


def get_self_reflection() -> SelfReflection:
    """Get singleton SelfReflection instance."""
    global _reflection
    if _reflection is None:
        with _lock:
            if _reflection is None:
                _reflection = SelfReflection()
    return _reflection


def reset_self_reflection() -> None:
    """Reset the singleton (for testing)."""
    global _reflection
    with _lock:
        _reflection = None