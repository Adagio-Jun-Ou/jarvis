"""Tests for Emotional Memory Bridge."""

import pytest
import time
from core.emotional_memory import (
    EmotionalMemory, AffectiveTag, FeelingMilestone,
    get_emotional_memory, reset_emotional_memory,
)


class TestAffectiveTag:
    def test_default_tag(self):
        tag = AffectiveTag()
        assert tag.valence == 0.0
        assert tag.arousal == 0.5
        assert tag.quality == "neutral"

    def test_emotional_weight_low(self):
        tag = AffectiveTag(valence=0.0, arousal=0.5, dominance=0.5)
        assert tag.emotional_weight < 0.3

    def test_emotional_weight_high(self):
        tag = AffectiveTag(valence=-0.8, arousal=0.9, dominance=0.2)
        assert tag.emotional_weight > 0.4

    def test_emotional_weight_bounded(self):
        tag = AffectiveTag(valence=-1.0, arousal=1.0, dominance=0.0)
        assert tag.emotional_weight <= 1.0


class TestEmotionalMemory:
    def test_tag_interaction(self):
        em = EmotionalMemory()
        tag = em.tag_interaction(valence=0.5, arousal=0.7, quality="happy", topic="code review")
        assert tag.valence == 0.5
        assert tag.quality == "happy"

    def test_feeling_trajectory(self):
        em = EmotionalMemory()
        em.record_feeling_shift("tense", "calm", regulation="reappraisal", trigger="user_feedback")
        trajectory = em.get_feeling_trajectory()
        assert len(trajectory) == 1
        assert trajectory[0]["regulation"] == "reappraisal"

    def test_regulation_learning(self):
        em = EmotionalMemory()
        em.record_feeling_shift("overwhelmed", "calm", regulation="attentional_deployment", trigger="complex_task")
        history = em.get_regulation_history()
        assert len(history) == 1
        assert history[0]["strategy"] == "attentional_deployment"

    def test_best_regulation_for(self):
        em = EmotionalMemory()
        for _ in range(3):
            em.record_feeling_shift("tense", "calm", regulation="reappraisal", trigger="stress")
        em.record_feeling_shift("tense", "worried", regulation="suppression", trigger="stress")
        best = em.get_best_regulation_for("tense")
        assert best == "reappraisal"

    def test_best_regulation_unknown(self):
        em = EmotionalMemory()
        assert em.get_best_regulation_for("unknown_feeling") is None

    def test_recall_boost_congruent(self):
        em = EmotionalMemory()
        tags = [
            AffectiveTag(valence=0.5, arousal=0.7),
            AffectiveTag(valence=-0.3, arousal=0.3),
        ]
        boosts = em.get_recall_boost(current_valence=0.6, current_arousal=0.7, memory_tags=tags)
        assert len(boosts) == 2
        assert boosts[0] > 1.0

    def test_recall_boost_incongruent(self):
        em = EmotionalMemory()
        tags = [
            AffectiveTag(valence=-0.8, arousal=0.9),
        ]
        boosts = em.get_recall_boost(current_valence=0.8, current_arousal=0.2, memory_tags=tags)
        assert boosts[0] < 1.0

    def test_dominant_feeling_empty(self):
        em = EmotionalMemory()
        result = em.get_dominant_feeling()
        assert result["quality"] == "neutral"

    def test_dominant_feeling_with_data(self):
        em = EmotionalMemory()
        for _ in range(5):
            em.tag_interaction(valence=0.5, arousal=0.7, quality="happy")
        em.tag_interaction(valence=-0.3, arousal=0.5, quality="tense")
        result = em.get_dominant_feeling()
        assert result["quality"] == "happy"
        assert result["sample_size"] == 6

    def test_valence_trend(self):
        em = EmotionalMemory()
        for i in range(10):
            em.tag_interaction(valence=-0.5 + i * 0.12, arousal=0.5, quality="improving")
        result = em.get_dominant_feeling()
        assert result["valence_trend"] > 0.0

    def test_get_state(self):
        em = EmotionalMemory()
        em.tag_interaction(valence=0.3, arousal=0.6, quality="focused")
        state = em.get_state()
        assert "total_tags" in state
        assert "dominant_feeling" in state
        assert "session_duration_min" in state

    def test_recent_tags(self):
        em = EmotionalMemory()
        em.tag_interaction(valence=0.3, arousal=0.6, quality="focused", topic="coding")
        tags = em.get_recent_tags(limit=5)
        assert len(tags) == 1
        assert tags[0]["topic"] == "coding"

    def test_singleton(self):
        reset_emotional_memory()
        em1 = get_emotional_memory()
        em2 = get_emotional_memory()
        assert em1 is em2

    def test_regulation_history_filtered(self):
        em = EmotionalMemory()
        em.record_feeling_shift("tense", "calm", regulation="reappraisal", trigger="deadline")
        em.record_feeling_shift("happy", "content", regulation="acceptance", trigger="success")
        tense_history = em.get_regulation_history(feeling_quality="tense")
        assert len(tense_history) == 1
        assert tense_history[0]["strategy"] == "reappraisal"

    def test_prune_old_tags_removes_expired(self):
        em = EmotionalMemory()
        em.tag_interaction(valence=0.5, arousal=0.6, quality="happy")
        em._affective_tags[0].timestamp = time.time() - 100 * 3600
        em.tag_interaction(valence=0.3, arousal=0.5, quality="neutral")
        pruned = em.prune_old_tags(max_age_hours=48)
        assert pruned == 1
        assert len(em._affective_tags) == 1

    def test_prune_old_tags_keeps_recent(self):
        em = EmotionalMemory()
        for _ in range(10):
            em.tag_interaction(valence=0.3, arousal=0.5, quality="neutral")
        pruned = em.prune_old_tags(max_age_hours=48)
        assert pruned == 0
        assert len(em._affective_tags) == 10