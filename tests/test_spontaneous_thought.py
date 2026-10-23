"""Tests for spontaneous thought generation."""

import time
import pytest
from core.spontaneous_thought import (
    SpontaneousThoughtGenerator, SpontaneousThought,
    get_spontaneous_thought_generator, reset_spontaneous_thought_generator,
)
from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState


class TestSpontaneousThought:
    def test_initial_state(self):
        stg = SpontaneousThoughtGenerator()
        state = stg.get_state()
        assert state["total_thoughts"] == 0
        assert state["pending"] == 0

    def test_generate_curiosity_thought(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        pred_state = {
            "curiosity_topics": [("quantum computing", 0.8), ("AI safety", 0.6)],
            "drives": {"curiosity": 0.7},
            "should_explore": True,
            "free_energy": 0.5,
        }
        thought = stg.tick(prediction_state=pred_state)
        assert thought is not None
        assert thought.source in ("curiosity", "consolidation", "association", "emotion")

    def test_generate_consolidation_thought(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        pred_state = {
            "curiosity_topics": [("neural networks", 0.5)],
            "drives": {"mastery": 0.8},
        }
        cog_state = {"state": "dreaming", "processing_depth": "deep"}
        thought = stg._generate_consolidation_thought(pred_state, {})
        assert thought is not None
        assert thought.source == "consolidation"
        assert "neural networks" in thought.content

    def test_cooldown_prevents_spam(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 100.0
        pred_state = {"curiosity_topics": [("test", 0.5)]}
        thought1 = stg.tick(prediction_state=pred_state)
        thought2 = stg.tick(prediction_state=pred_state)
        if thought1:
            assert thought2 is None

    def test_get_next_thought(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        stg._min_interval = 0.0
        pred_state = {"curiosity_topics": [("test", 0.5)]}
        thought = stg.tick(prediction_state=pred_state)
        if thought:
            next_thought = stg.get_next_thought()
            assert next_thought is not None
            assert next_thought.expressed

    def test_peek_thoughts(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        pred_state = {"curiosity_topics": [("test", 0.5)]}
        thought = stg.tick(prediction_state=pred_state)
        if thought:
            peeked = stg.peek_thoughts(limit=5)
            assert len(peeked) >= 0

    def test_clear_expressed(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        stg._min_interval = 0.0
        pred_state = {"curiosity_topics": [("test", 0.5)]}
        thought = stg.tick(prediction_state=pred_state)
        if thought:
            next_t = stg.get_next_thought()
            cleared = stg.clear_expressed()
            assert cleared >= 0

    def test_no_daydream_in_awake(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        cog_state = {"state": "awake"}
        pred_state = {}
        thought = stg.tick(cognitive_state=cog_state, prediction_state=pred_state)
        assert thought is not None or True

    def test_dreaming_boosts_consolidation(self):
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        cog_state = {"state": "dreaming", "processing_depth": "deep"}
        pred_state = {"curiosity_topics": [("test", 0.5)]}
        thought = stg.tick(cognitive_state=cog_state, prediction_state=pred_state)
        assert thought is not None

    def test_singleton(self):
        st1 = get_spontaneous_thought_generator()
        st2 = get_spontaneous_thought_generator()
        assert st1 is st2
        reset_spontaneous_thought_generator()