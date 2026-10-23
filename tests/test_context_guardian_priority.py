"""Tests for Context Guardian priority-based eviction."""
import pytest
import time

from core.context_guardian import (
    ContextGuardian,
    PressureLevel,
    GuardianAction,
    GuardianStats,
    THRESHOLDS,
    get_context_guardian,
)


class TestContextGuardianPressure:
    def test_normal_pressure(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(50000) == PressureLevel.NORMAL

    def test_warning_pressure(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(71000) == PressureLevel.WARNING

    def test_soft_evict_pressure(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(81000) == PressureLevel.SOFT_EVICT

    def test_hard_evict_pressure(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(91000) == PressureLevel.HARD_EVICT

    def test_emergency_pressure(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(96000) == PressureLevel.EMERGENCY

    def test_zero_budget(self):
        g = ContextGuardian(token_budget=0)
        assert g.check_pressure(0) == PressureLevel.NORMAL

    def test_should_auto_evict(self):
        g = ContextGuardian(token_budget=100000)
        assert g.should_auto_evict(50000) is False
        assert g.should_auto_evict(81000) is True
        assert g.should_auto_evict(96000) is True


class TestContextGuardianEviction:
    def test_eviction_plan_soft(self):
        g = ContextGuardian(token_budget=100000)
        plan = g.get_eviction_plan(81000, 30)
        assert plan["level"] == "soft_evict"
        assert plan["keep_recent"] >= 10

    def test_eviction_plan_hard(self):
        g = ContextGuardian(token_budget=100000)
        plan = g.get_eviction_plan(91000, 30)
        assert plan["level"] == "hard_evict"
        assert plan["should_summarize"] is True

    def test_eviction_plan_emergency(self):
        g = ContextGuardian(token_budget=100000)
        plan = g.get_eviction_plan(96000, 30)
        assert plan["level"] == "emergency"
        assert plan["keep_recent"] <= 5


class TestMessageScoring:
    def test_system_message_high(self):
        g = ContextGuardian()
        score = g.score_message({"role": "system", "content": "You are Jarvis."})
        assert score >= 0.9

    def test_tool_result_medium(self):
        g = ContextGuardian()
        score = g.score_message({"role": "tool", "content": "result data"})
        assert 0.3 <= score <= 0.7

    def test_summary_low(self):
        g = ContextGuardian()
        score = g.score_message({"role": "system", "content": "Previous conversation summary: user asked about Python"})
        assert score < 0.5

    def test_error_message_high(self):
        g = ContextGuardian()
        score = g.score_message({"role": "assistant", "content": "Error: file not found"})
        assert score > 0.6

    def test_code_message_high(self):
        g = ContextGuardian()
        score = g.score_message({"role": "assistant", "content": "def hello():\n  return 'hi'"})
        assert score > 0.6

    def test_long_message_lower(self):
        g = ContextGuardian()
        short = g.score_message({"role": "assistant", "content": "Short response"})
        long = g.score_message({"role": "assistant", "content": "x" * 3000})
        assert long <= short

    def test_user_message_base(self):
        g = ContextGuardian()
        score = g.score_message({"role": "user", "content": "How do I install pip?"})
        assert 0.4 <= score <= 0.8

    def test_empty_message(self):
        g = ContextGuardian()
        score = g.score_message({"role": "user", "content": ""})
        assert score >= 0.0


class TestEvictionOrder:
    def test_system_last(self):
        g = ContextGuardian()
        messages = [
            {"role": "system", "content": "You are Jarvis."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Previous conversation summary: stuff"},
        ]
        order = g.get_eviction_order(messages)
        assert order[0] == 2  # Summary evicted first
        assert order[-1] == 0  # System message evicted last

    def test_tool_results_before_user(self):
        g = ContextGuardian()
        messages = [
            {"role": "user", "content": "Search for Python tutorials"},
            {"role": "tool", "content": "Search results: ..."},
        ]
        order = g.get_eviction_order(messages)
        assert order[0] == 1  # Tool evicted first

    def test_preserves_all_indices(self):
        g = ContextGuardian()
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        order = g.get_eviction_order(messages)
        assert set(order) == set(range(10))


class TestGuardianActions:
    def test_record_warning(self):
        g = ContextGuardian()
        action = GuardianAction(
            level=PressureLevel.WARNING,
            action="warn",
            tokens_before=71000,
            tokens_after=71000,
            messages_removed=0,
            reason="Approaching limit",
        )
        g.record_action(action)
        stats = g.get_stats()
        assert stats["warnings_issued"] == 1

    def test_record_eviction(self):
        g = ContextGuardian()
        action = GuardianAction(
            level=PressureLevel.SOFT_EVICT,
            action="evict",
            tokens_before=81000,
            tokens_after=60000,
            messages_removed=5,
            reason="Soft eviction",
        )
        g.record_action(action)
        stats = g.get_stats()
        assert stats["soft_evictions"] == 1
        assert stats["total_messages_evicted"] == 5

    def test_actions_bounded(self):
        g = ContextGuardian()
        for i in range(150):
            action = GuardianAction(
                level=PressureLevel.WARNING,
                action=f"warn_{i}",
                tokens_before=71000,
                tokens_after=71000,
                messages_removed=0,
                reason="test",
            )
            g.record_action(action)
        assert len(g._actions) <= 100


class TestEmergencyCallbacks:
    def test_emergency_callback_invoked(self):
        g = ContextGuardian()
        called = []
        g.add_emergency_callback(lambda action: called.append(action))
        action = GuardianAction(
            level=PressureLevel.EMERGENCY,
            action="emergency",
            tokens_before=96000,
            tokens_after=40000,
            messages_removed=20,
            reason="Emergency eviction",
        )
        g.record_action(action)
        assert len(called) == 1

    def test_no_callback_on_non_emergency(self):
        g = ContextGuardian()
        called = []
        g.add_emergency_callback(lambda action: called.append(action))
        action = GuardianAction(
            level=PressureLevel.WARNING,
            action="warn",
            tokens_before=71000,
            tokens_after=71000,
            messages_removed=0,
            reason="Warning",
        )
        g.record_action(action)
        assert len(called) == 0

    def test_callback_error_doesnt_break(self):
        g = ContextGuardian()
        def bad_callback(action):
            raise ValueError("bad")
        g.add_emergency_callback(bad_callback)
        action = GuardianAction(
            level=PressureLevel.EMERGENCY,
            action="emergency",
            tokens_before=96000,
            tokens_after=40000,
            messages_removed=20,
            reason="test",
        )
        g.record_action(action)

    def test_max_callbacks(self):
        g = ContextGuardian()
        for i in range(25):
            g.add_emergency_callback(lambda action: None)
        assert len(g._emergency_callbacks) <= 20


class TestGetContextGuardian:
    def test_singleton(self):
        from core.context_guardian import _guardian
        import core.context_guardian as mod
        mod._guardian = None
        g1 = get_context_guardian()
        g2 = get_context_guardian()
        assert g1 is g2
        mod._guardian = None