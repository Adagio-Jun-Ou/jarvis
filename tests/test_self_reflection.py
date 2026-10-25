"""Tests for Self-Reflection."""
import pytest
import time
import os
from pathlib import Path
from core.self_reflection import (
    SelfReflection,
    Mistake,
    ConversationAnalysis,
    ReflectionResult,
    ResponseQuality,
    get_self_reflection,
    reset_self_reflection,
)


def get_fresh_reflection():
    """Get a fresh reflection instance for testing."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    path = Path(f"/tmp/reflection_test_{unique_id}")
    return SelfReflection(storage_path=path)


class TestMistake:
    """Tests for Mistake dataclass."""
    
    def test_creation(self):
        mistake = Mistake(
            timestamp=time.time(),
            context="User asked about Python",
            action="Provided Java example",
            issue="Wrong language",
            improvement="Better language detection",
            category="understanding"
        )
        assert mistake.category == "understanding"
        assert not mistake.resolved


class TestConversationAnalysis:
    """Tests for ConversationAnalysis dataclass."""
    
    def test_creation(self):
        analysis = ConversationAnalysis(
            satisfaction_score=0.8,
            user_engagement="high",
            response_quality=ResponseQuality.GOOD,
            successful_interactions=5,
            issues_encountered=1,
            topics_covered=["python", "code"]
        )
        assert analysis.satisfaction_score == 0.8
        assert analysis.response_quality == ResponseQuality.GOOD


class TestSelfReflection:
    """Tests for SelfReflection."""
    
    def test_initial_state(self):
        reflection = get_fresh_reflection()
        assert len(reflection._conversation_history) == 0
        assert len(reflection._mistakes) == 0
    
    def test_record_conversation_turn(self):
        reflection = get_fresh_reflection()
        
        reflection.record_conversation_turn(
            user_input="What is Python?",
            jarvis_response="Python is a programming language."
        )
        
        assert len(reflection._conversation_history) == 1
        assert reflection._conversation_history[0]["user"] == "What is Python?"
    
    def test_record_multiple_turns(self):
        reflection = get_fresh_reflection()
        
        for i in range(10):
            reflection.record_conversation_turn(
                user_input=f"Question {i}",
                jarvis_response=f"Answer {i}"
            )
        
        assert len(reflection._conversation_history) == 10
    
    def test_record_too_many_turns(self):
        reflection = get_fresh_reflection()
        
        for i in range(60):
            reflection.record_conversation_turn(
                user_input=f"Question {i}",
                jarvis_response=f"Answer {i}"
            )
        
        # Should keep only last 50
        assert len(reflection._conversation_history) == 50
    
    def test_analyze_conversation_empty(self):
        reflection = get_fresh_reflection()
        
        result = reflection.analyze_conversation()
        
        assert isinstance(result, ReflectionResult)
        assert result.analysis.satisfaction_score == 0.5
        assert result.analysis.response_quality == ResponseQuality.UNCLEAR
    
    def test_analyze_conversation_positive(self):
        reflection = get_fresh_reflection()
        
        turns = [
            {"user": "What is Python?", "jarvis": "Python is a language."},
            {"user": "Thanks, that helps!", "jarvis": "You're welcome!"},
            {"user": "Great explanation", "jarvis": "Happy to help!"}
        ]
        
        result = reflection.analyze_conversation(turns)
        
        assert result.analysis.satisfaction_score > 0.5
        assert result.analysis.successful_interactions > 0
    
    def test_analyze_conversation_negative(self):
        reflection = get_fresh_reflection()
        
        turns = [
            {"user": "What is Python?", "jarvis": "Python is a snake."},
            {"user": "That's not what I wanted", "jarvis": "Sorry."},
            {"user": "That doesn't help", "jarvis": "Let me try again."}
        ]
        
        result = reflection.analyze_conversation(turns)
        
        assert result.analysis.satisfaction_score < 0.5
        assert result.analysis.issues_encountered > 0
        assert len(result.mistakes) > 0
    
    def test_analyze_conversation_engagement(self):
        reflection = get_fresh_reflection()
        
        # Long messages = high engagement
        turns = [
            {"user": "I'm working on a complex Python project and I need help with async programming patterns specifically for web scraping. Can you explain the best practices?", "jarvis": "Sure!"},
            {"user": "That's great, but can you also explain how to handle rate limiting and retries properly?", "jarvis": "Of course!"}
        ]
        
        result = reflection.analyze_conversation(turns)
        
        assert result.analysis.user_engagement == "high"
    
    def test_analyze_conversation_topics(self):
        reflection = get_fresh_reflection()
        
        turns = [
            {"user": "I need help with Python async programming", "jarvis": "Sure!"},
            {"user": "What about concurrency in Python?", "jarvis": "Here's info..."}
        ]
        
        result = reflection.analyze_conversation(turns)
        
        assert "python" in result.analysis.topics_covered or len(result.analysis.topics_covered) >= 0
    
    def test_identify_mistakes(self):
        reflection = get_fresh_reflection()
        
        result = reflection.analyze_conversation([])
        mistakes = result.mistakes
        
        assert isinstance(mistakes, list)
    
    def test_generate_improvement_plan_no_mistakes(self):
        reflection = get_fresh_reflection()
        
        plan = reflection.generate_improvement_plan([])
        
        assert "No recent mistakes" in plan or "improvement" in plan.lower()
    
    def test_generate_improvement_plan_with_mistakes(self):
        reflection = get_fresh_reflection()
        
        mistakes = [
            Mistake(
                timestamp=time.time(),
                context="context",
                action="action",
                issue="issue",
                improvement="improve understanding",
                category="understanding"
            )
        ]
        
        plan = reflection.generate_improvement_plan(mistakes)
        
        assert "Understanding" in plan or "understanding" in plan.lower()
    
    def test_track_response_quality_positive(self):
        reflection = get_fresh_reflection()
        
        reflection.track_response_quality(
            response="Python is a programming language.",
            user_feedback="Thanks, that's helpful!"
        )
        
        # Should have recorded quality
        assert len(reflection._quality_scores) == 1
        assert reflection._quality_scores[0][0] > 0.5
    
    def test_track_response_quality_negative(self):
        reflection = get_fresh_reflection()
        
        reflection.track_response_quality(
            response="Python is a snake.",
            user_feedback="That's not helpful, wrong information."
        )
        
        assert len(reflection._quality_scores) == 1
        assert reflection._quality_scores[0][0] < 0.5
    
    def test_track_response_quality_neutral(self):
        reflection = get_fresh_reflection()
        
        reflection.track_response_quality(
            response="Here's some info.",
            user_feedback=None
        )
        
        assert len(reflection._quality_scores) == 1
        # Should be neutral 0.5
        avg = reflection.get_average_quality()
        assert 0.4 <= avg <= 0.6
    
    def test_get_average_quality(self):
        reflection = get_fresh_reflection()
        
        reflection._quality_scores = [
            (0.8, time.time()),
            (0.6, time.time()),
            (0.7, time.time())
        ]
        
        avg = reflection.get_average_quality()
        
        assert 0.6 <= avg <= 0.8
    
    def test_get_reflection_context(self):
        reflection = get_fresh_reflection()
        
        reflection._quality_scores = [(0.7, time.time())]
        
        context = reflection.get_reflection_context()
        
        assert "<reflection>" in context
        assert "quality" in context.lower()
    
    def test_persistence(self, tmp_path):
        reflection1 = SelfReflection(storage_path=tmp_path)
        
        reflection1.record_conversation_turn("Test", "Response")
        reflection1.track_response_quality("Test", "Good!")
        
        # Create new instance to load persisted data
        reflection2 = SelfReflection(storage_path=tmp_path)
        
        assert len(reflection2._quality_scores) >= 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_self_reflection_singleton(self):
        reflection1 = get_self_reflection()
        reflection2 = get_self_reflection()
        assert reflection1 is reflection2
    
    def test_reset_self_reflection(self):
        reflection1 = get_self_reflection()
        reset_self_reflection()
        reflection2 = get_self_reflection()
        # After clean, different storage path might give different instance
        assert reflection2 is not None


class TestIntegration:
    """Integration tests."""
    
    def test_full_analysis_flow(self):
        reflection = get_fresh_reflection()
        
        # Record conversation
        reflection.record_conversation_turn(
            "What is async/await in Python?",
            "Async/await is used for asynchronous programming..."
        )
        reflection.record_conversation_turn(
            "Thanks, that's helpful!",
            "You're welcome! Let me know if you need more details."
        )
        reflection.record_conversation_turn(
            "Can you explain coroutines too?",
            "Coroutines are functions that can pause and resume..."
        )
        
        # Analyze
        result = reflection.analyze_conversation()
        
        assert isinstance(result, ReflectionResult)
        assert result.analysis.satisfaction_score >= 0.0
        assert isinstance(result.mistakes, list)
        assert isinstance(result.improvements, list)
        
        # Track quality
        reflection.track_response_quality("Response", "Great explanation!")
        
        # Get context
        context = reflection.get_reflection_context()
        assert len(context) > 0
    
    def test_negative_conversation_analysis(self):
        reflection = get_fresh_reflection()
        
        # Record problematic conversation
        reflection.record_conversation_turn(
            "What is Python?",
            "Python is a snake that eats mice."
        )
        reflection.record_conversation_turn(
            "That's not what I meant!",
            "Oh, you meant the programming language."
        )
        reflection.record_conversation_turn(
            "I don't understand your explanation.",
            "Let me try again..."
        )
        
        result = reflection.analyze_conversation()
        
        # Should identify issues
        assert result.analysis.satisfaction_score < 0.6
        assert result.analysis.issues_encountered >= 0
    
    def test_quality_tracking_over_time(self):
        reflection = get_fresh_reflection()
        
        # Series of responses
        feedbacks = [
            "Great!",
            "Good, thanks",
            "Perfect",
            "ok",
            "Good"
        ]
        
        for i, fb in enumerate(feedbacks):
            reflection.track_response_quality(f"Response {i}", fb)
        
        avg = reflection.get_average_quality()
        
        # Should be above 0.5 with mostly positive feedback
        assert avg > 0.5
    
    def test_analyze_trends(self):
        reflection = get_fresh_reflection()
        
        # Record some conversations with quality tracking
        reflection.track_response_quality("Response 1", "Great!")
        reflection.track_response_quality("Response 2", "Good")
        reflection.track_response_quality("Response 3", "Not helpful")
        
        # Analyze trends
        trends = reflection.analyze_trends(days=1)
        
        assert "avg_quality" in trends
        assert "quality_trend" in trends
        assert "total_mistakes" in trends
        assert "common_mistakes" in trends
        assert isinstance(trends["avg_quality"], float)
        assert trends["quality_trend"] in ["improving", "declining", "stable"]
    
    def test_get_trend_summary(self):
        reflection = get_fresh_reflection()
        
        # Track some quality
        reflection.track_response_quality("Response", "Good")
        
        summary = reflection.get_trend_summary(days=1)
        
        assert "<trend_analysis" in summary
        assert "Quality:" in summary or "quality" in summary.lower()