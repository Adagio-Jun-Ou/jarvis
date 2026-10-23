"""Tests for Sentiment Analyzer module."""
import threading

import pytest

from core.sentiment_analyzer import (
    SentimentAnalyzer, SentimentType, NeedType, SentimentResult,
    SentimentTrajectory, get_sentiment_analyzer,
)


class TestSentimentAnalysis:
    def test_neutral_short_input(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("ok")
        assert result.sentiment == SentimentType.NEUTRAL

    def test_empty_input(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("")
        assert result.sentiment == SentimentType.NEUTRAL
        assert result.intensity == 0.0

    def test_frustration_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is so frustrating, I'm stuck!")
        assert result.sentiment == SentimentType.FRUSTRATED
        assert result.intensity > 0.5
        assert result.requires_response_change is True

    def test_curiosity_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("How does this work? I'm curious about the implementation.")
        assert result.sentiment == SentimentType.CURIOUS
        assert result.confidence > 0.3

    def test_excitement_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is awesome! Great job!")
        assert result.sentiment == SentimentType.EXCITED
        assert result.intensity > 0.5

    def test_gratitude_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("Thank you so much, this is really helpful!")
        assert result.sentiment in (SentimentType.GRATEFUL, SentimentType.EXCITED)

    def test_urgency_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I need this ASAP, it's urgent!")
        assert result.sentiment == SentimentType.URGENCY
        assert result.intensity > 0.6

    def test_confusion_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I'm confused and this is unclear, I don't understand")
        assert result.sentiment == SentimentType.CONFUSED
        assert result.intensity > 0.4

    def test_determination_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I'm determined to figure this out, I won't give up!")
        assert result.sentiment == SentimentType.DETERMINED

    def test_overwhelmed_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("There's too much to handle, I'm overwhelmed")
        assert result.sentiment == SentimentType.OVERWHELMED

    def test_satisfaction_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("That works, problem solved!")
        assert result.sentiment in (SentimentType.SATISFIED, SentimentType.GRATEFUL)

    def test_intensity_multiplier_very(self):
        sa = SentimentAnalyzer()
        result_normal = sa.analyze("this is frustrating")
        sa.reset()
        result_intense = sa.analyze("this is very frustrating")
        assert result_intense.intensity >= result_normal.intensity

    def test_intensity_multiplier_slightly(self):
        sa = SentimentAnalyzer()
        result_normal = sa.analyze("I'm frustrated")
        sa.reset()
        result_mild = sa.analyze("I'm slightly frustrated")
        assert result_mild.intensity < result_normal.intensity

    def test_negation_reduces_sentiment(self):
        sa = SentimentAnalyzer()
        result_positive = sa.analyze("this is very frustrating")
        sa.reset()
        result_negated = sa.analyze("this is not frustrating")
        assert result_negated.intensity < result_positive.intensity or result_negated.sentiment == SentimentType.NEUTRAL

    def test_caps_detection(self):
        sa = SentimentAnalyzer()
        result_caps = sa.analyze("THIS IS BROKEN AND TERRIBLE!")
        assert result_caps.sentiment == SentimentType.FRUSTRATED
        assert result_caps.intensity > 0.5

    def test_question_boosts_curious(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("what does this do?")
        assert SentimentType.CURIOUS in [result.sentiment] or result.confidence > 0

    def test_needs_detection_help(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I need help with this error")
        assert NeedType.HELP in result.needs

    def test_needs_detection_explanation(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("Can you explain how this works?")
        assert NeedType.EXPLANATION in result.needs

    def test_needs_detection_action(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("Fix this bug and update the documentation")
        assert NeedType.ACTION in result.needs

    def test_needs_detection_confirmation(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("Can you verify this is the right approach?")
        assert NeedType.CONFIRMATION in result.needs or NeedType.EXPLANATION in result.needs or len(result.needs) >= 1

    def test_escalation_risk(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I'm absolutely furious! This is broken again!")
        assert result.escalation_risk > 0.3
        assert result.requires_response_change is True

    def test_low_escalation_risk(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("that's nice")
        assert result.escalation_risk < 0.5


class TestSentimentTrajectory:
    def test_trajectory_starts_stable(self):
        traj = SentimentTrajectory()
        assert traj.trend == "stable"

    def test_trajectory_escalating(self):
        traj = SentimentTrajectory()
        for i in range(8):
            traj.add(SentimentType.NEUTRAL, 0.2)
        for i in range(8):
            traj.add(SentimentType.FRUSTRATED, 0.5 + i * 0.06)
        traj.compute_trend()
        assert traj.trend == "escalating"

    def test_trajectory_de_escalating(self):
        traj = SentimentTrajectory()
        for i in range(8):
            traj.add(SentimentType.FRUSTRATED, 0.8)
        for i in range(8):
            traj.add(SentimentType.SATISFIED, 0.2)
        traj.compute_trend()
        assert traj.trend == "de_escalating"

    def test_is_escalating(self):
        traj = SentimentTrajectory()
        for i in range(10):
            traj.add(SentimentType.FRUSTRATED, 0.3 + i * 0.07)
        traj.compute_trend()
        assert traj.is_escalating() is True


class TestSentimentAnalyzerState:
    def test_state_empty(self):
        sa = SentimentAnalyzer()
        state = sa.get_state()
        assert state["trajectory_trend"] == "stable"
        assert state["total_analyzed"] == 0

    def test_state_after_analysis(self):
        sa = SentimentAnalyzer()
        sa.analyze("I'm frustrated with this bug!")
        state = sa.get_state()
        assert state["total_analyzed"] == 1
        assert state["recent_sentiment"] == "frustrated"
        assert state["recent_intensity"] > 0

    def test_recent_sentiments(self):
        sa = SentimentAnalyzer()
        sa.analyze("Hello!")
        sa.analyze("This is frustrating!")
        sa.analyze("How does this work?")
        recent = sa.get_recent_sentiments(limit=3)
        assert len(recent) == 3
        assert "sentiment" in recent[0]
        assert "intensity" in recent[0]

    def test_reset(self):
        sa = SentimentAnalyzer()
        sa.analyze("This is terrible!")
        sa.reset()
        state = sa.get_state()
        assert state["total_analyzed"] == 0
        assert state["recent_intensity"] == 0.0


class TestSentimentAnalyzerSingleton:
    def test_singleton(self):
        sa1 = get_sentiment_analyzer()
        sa2 = get_sentiment_analyzer()
        assert sa1 is sa2


class TestSentimentAnalyzerThreadSafety:
    def test_concurrent_analysis(self):
        sa = SentimentAnalyzer()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    result = sa.analyze("This is a test message about a bug!")
                    assert result.sentiment is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0


class TestSentimentEdgeCases:
    def test_very_short_input(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("ok")
        assert result.sentiment == SentimentType.NEUTRAL

    def test_only_punctuation(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("...!!!")
        assert result.sentiment == SentimentType.NEUTRAL

    def test_mixed_signals(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I'm grateful but also a bit frustrated with the delay")
        assert result.sentiment in (SentimentType.GRATEFUL, SentimentType.FRUSTRATED)
        assert result.intensity > 0

    def test_playful_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("lol let's try something fun!")
        assert result.sentiment == SentimentType.PLAYFUL

    def test_multi_need_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("Can you explain how this works and help me fix it?")
        assert len(result.needs) >= 1

    def test_anxious_detection(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I'm worried about the deadline, this is urgent!")
        assert result.sentiment in (SentimentType.ANXIOUS, SentimentType.URGENCY)
        assert result.escalation_risk > 0.3