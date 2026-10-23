"""Tests for context system hardening: adaptive thresholds, pressure predictor, multi-budget guardian."""
import os
import tempfile
import threading
import time

import pytest

os.environ.setdefault("JARVIS_HOME", tempfile.mkdtemp())

from core.context_guardian import (
    ContextGuardian,
    PressureLevel,
    GuardianAction,
    AdaptiveThresholds,
    ContextPressurePredictor,
    MultiBudgetGuardian,
    get_context_guardian,
    get_multi_budget_guardian,
)


class TestAdaptiveThresholds:
    def test_initial_thresholds(self):
        at = AdaptiveThresholds()
        assert at.get_threshold(PressureLevel.WARNING) == 0.70
        assert at.get_threshold(PressureLevel.EMERGENCY) == 0.95

    def test_threshold_adjusts_up_on_frequent_evictions(self):
        at = AdaptiveThresholds()
        at._window_seconds = 1.0
        at._eviction_count_window = []
        for _ in range(5):
            at.record_eviction(PressureLevel.SOFT_EVICT)
        threshold = at.get_threshold(PressureLevel.SOFT_EVICT)
        assert threshold > 0.80

    def test_threshold_adjusts_down_on_stable(self):
        at = AdaptiveThresholds(cooldown_seconds=0.01)
        at.record_eviction(PressureLevel.SOFT_EVICT)
        at._adjustments[PressureLevel.SOFT_EVICT] = 0.05
        time.sleep(0.02)
        at.record_stable()
        threshold = at.get_threshold(PressureLevel.SOFT_EVICT)
        assert threshold < 0.85

    def test_max_adjustment_cap(self):
        at = AdaptiveThresholds(max_adjustment=0.05, adjustment_rate=0.03)
        for _ in range(20):
            at.record_eviction(PressureLevel.HARD_EVICT)
        threshold = at.get_threshold(PressureLevel.HARD_EVICT)
        assert threshold <= 0.90 + 0.05

    def test_get_status(self):
        at = AdaptiveThresholds()
        status = at.get_status()
        assert "current_thresholds" in status
        assert "base_thresholds" in status
        assert "adjustments" in status
        assert "eviction_rate_per_min" in status

    def test_thread_safety(self):
        at = AdaptiveThresholds()
        errors = []

        def writer():
            try:
                for _ in range(50):
                    at.record_eviction(PressureLevel.SOFT_EVICT)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    at.get_threshold(PressureLevel.WARNING)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestContextPressurePredictor:
    def test_initial_state(self):
        cpp = ContextPressurePredictor()
        assert cpp.get_trend() == "stable"

    def test_record_samples(self):
        cpp = ContextPressurePredictor()
        for i in range(10):
            cpp.record(1000 + i * 100)
        assert len(cpp._samples) == 10

    def test_predict_rising(self):
        cpp = ContextPressurePredictor()
        now = time.time()
        for i in range(10):
            cpp._samples.append((now - (10 - i), 10000 + i * 500))
        result = cpp.predict_time_to_pressure(budget=128000, target_ratio=0.80)
        assert result is not None
        assert result > 0

    def test_predict_insufficient_data(self):
        cpp = ContextPressurePredictor()
        cpp.record(1000)
        result = cpp.predict_time_to_pressure(budget=128000, target_ratio=0.80)
        assert result is None

    def test_predict_decreasing_returns_none(self):
        cpp = ContextPressurePredictor()
        now = time.time()
        for i in range(10):
            cpp._samples.append((now - (10 - i), 100000 - i * 5000))
        result = cpp.predict_time_to_pressure(budget=128000, target_ratio=0.80)
        assert result is None

    def test_predict_already_over(self):
        cpp = ContextPressurePredictor()
        now = time.time()
        for i in range(10):
            cpp._samples.append((now - (10 - i), 120000 + i * 100))
        result = cpp.predict_time_to_pressure(budget=128000, target_ratio=0.80)
        assert result == 0.0

    def test_trend_rising(self):
        cpp = ContextPressurePredictor()
        now = time.time()
        for i in range(5):
            cpp._samples.append((now - (5 - i), 10000 + i * 2000))
        assert cpp.get_trend() == "rising"

    def test_trend_falling(self):
        cpp = ContextPressurePredictor()
        now = time.time()
        for i in range(5):
            cpp._samples.append((now - (5 - i), 100000 - i * 2000))
        assert cpp.get_trend() == "falling"

    def test_trend_stable(self):
        cpp = ContextPressurePredictor()
        now = time.time()
        for i in range(5):
            cpp._samples.append((now - (5 - i), 50000 + i * 10))
        assert cpp.get_trend() == "stable"

    def test_sample_window(self):
        cpp = ContextPressurePredictor(sample_window=5)
        for i in range(20):
            cpp.record(i * 100)
        assert len(cpp._samples) <= 5

    def test_get_status(self):
        cpp = ContextPressurePredictor()
        cpp.record(50000)
        status = cpp.get_status()
        assert "sample_count" in status
        assert "trend" in status
        assert "latest_tokens" in status

    def test_thread_safety(self):
        cpp = ContextPressurePredictor()
        errors = []

        def writer():
            try:
                for i in range(100):
                    cpp.record(i * 100)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    cpp.get_trend()
                    cpp.predict_time_to_pressure(128000, 0.8)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestMultiBudgetGuardian:
    def test_create_guardian(self):
        mbg = MultiBudgetGuardian()
        g = mbg.get_guardian("primary", budget=64000)
        assert g is not None
        assert g.get_budget() == 64000

    def test_separate_budgets(self):
        mbg = MultiBudgetGuardian()
        g1 = mbg.get_guardian("primary", budget=64000)
        g2 = mbg.get_guardian("backend", budget=32000)
        assert g1.get_budget() != g2.get_budget()

    def test_same_guardian_returned(self):
        mbg = MultiBudgetGuardian()
        g1 = mbg.get_guardian("primary", budget=64000)
        g2 = mbg.get_guardian("primary", budget=128000)
        assert g1 is g2

    def test_check_all(self):
        mbg = MultiBudgetGuardian()
        mbg.get_guardian("primary", budget=64000)
        mbg.get_guardian("backend", budget=32000)
        results = mbg.check_all({"primary": 50000, "backend": 25000})
        assert "primary" in results
        assert "backend" in results
        assert results["primary"]["level"] in ["normal", "warning", "soft_evict", "hard_evict", "emergency"]

    def test_record_eviction(self):
        mbg = MultiBudgetGuardian()
        mbg.get_guardian("primary", budget=64000)
        mbg.record_eviction("primary", PressureLevel.SOFT_EVICT)

    def test_record_stable(self):
        mbg = MultiBudgetGuardian()
        mbg.get_guardian("primary", budget=64000)
        mbg.record_stable("primary")

    def test_record_tokens(self):
        mbg = MultiBudgetGuardian()
        mbg.get_guardian("primary", budget=64000)
        for i in range(10):
            mbg.record_tokens("primary", 10000 + i * 1000)

    def test_get_status(self):
        mbg = MultiBudgetGuardian()
        mbg.get_guardian("primary", budget=64000)
        status = mbg.get_status()
        assert "models" in status
        assert "primary" in status["models"]
        assert "model_count" in status

    def test_unknown_model_returns_empty(self):
        mbg = MultiBudgetGuardian()
        results = mbg.check_all({"unknown": 50000})
        assert "unknown" not in results

    def test_singleton(self):
        mbg1 = get_multi_budget_guardian()
        mbg2 = get_multi_budget_guardian()
        assert mbg1 is mbg2


class TestContextGuardianEnhanced:
    def test_score_system_message(self):
        g = ContextGuardian()
        assert g.score_message({"role": "system", "content": "Important"}) >= 0.9

    def test_score_tool_message(self):
        g = ContextGuardian()
        assert g.score_message({"role": "tool", "content": "result"}) == 0.5

    def test_score_summary_low(self):
        g = ContextGuardian()
        assert g.score_message({"role": "assistant", "content": "Previous conversation summary"}) <= 0.3

    def test_score_error_message(self):
        g = ContextGuardian()
        s1 = g.score_message({"role": "user", "content": "normal"})
        s2 = g.score_message({"role": "user", "content": "Error: failed"})
        assert s2 > s1

    def test_score_code_message(self):
        g = ContextGuardian()
        s1 = g.score_message({"role": "user", "content": "hello"})
        s2 = g.score_message({"role": "user", "content": "def foo(): return bar"})
        assert s2 > s1

    def test_score_long_message_lower(self):
        g = ContextGuardian()
        s1 = g.score_message({"role": "user", "content": "short"})
        s2 = g.score_message({"role": "user", "content": "x" * 3000})
        assert s2 < s1

    def test_eviction_order(self):
        g = ContextGuardian()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Previous conversation summary"},
            {"role": "user", "content": "hello"},
        ]
        order = g.get_eviction_order(msgs)
        assert order[0] == 2  # summary evicted first

    def test_emergency_callback(self):
        g = ContextGuardian()
        called = []
        g.add_emergency_callback(lambda a: called.append(a))
        action = GuardianAction(
            level=PressureLevel.EMERGENCY,
            action="emergency",
            tokens_before=100000,
            tokens_after=50000,
            messages_removed=50,
            reason="overflow",
        )
        g.record_action(action)
        assert len(called) == 1

    def test_max_callbacks(self):
        g = ContextGuardian()
        g._max_callbacks = 2
        g.add_emergency_callback(lambda a: None)
        g.add_emergency_callback(lambda a: None)
        g.add_emergency_callback(lambda a: None)
        assert len(g._emergency_callbacks) == 2

    def test_check_pressure_normal(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(50000) == PressureLevel.NORMAL

    def test_check_pressure_warning(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(75000) == PressureLevel.WARNING

    def test_check_pressure_emergency(self):
        g = ContextGuardian(token_budget=100000)
        assert g.check_pressure(98000) == PressureLevel.EMERGENCY

    def test_should_auto_evict(self):
        g = ContextGuardian(token_budget=100000)
        assert not g.should_auto_evict(50000)
        assert g.should_auto_evict(85000)

    def test_get_eviction_plan(self):
        g = ContextGuardian(token_budget=100000)
        plan = g.get_eviction_plan(85000, 50)
        assert "level" in plan
        assert "target_tokens" in plan
        assert "keep_recent" in plan

    def test_record_action_tracking(self):
        g = ContextGuardian()
        action = GuardianAction(
            level=PressureLevel.WARNING,
            action="warn",
            tokens_before=90000,
            tokens_after=90000,
            messages_removed=0,
            reason="approaching limit",
        )
        g.record_action(action)
        stats = g.get_stats()
        assert stats["warnings_issued"] == 1

    def test_singleton(self):
        g1 = get_context_guardian()
        g2 = get_context_guardian()
        assert g1 is g2