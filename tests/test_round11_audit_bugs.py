"""Tests for Round 11 audit bug fixes.

Covers:
- self_awareness.py: _assess_health buggy line comprehension, identify_blind_spots missing lock, register_multimodal unbounded
- knowledge_graph.py: add_insight no longer holds lock during I/O (add_node/add_edge are individually locked)
"""
import ast
import threading
import time
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# self_awareness.py — _assess_health buggy line comprehension
# ============================================================

class TestIntrospectionHealthAssessment:
    def test_assess_health_except_pass_detected(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = """def foo():
    try:
        x = 1
    except:
        pass
"""
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health < 1.0

    def test_assess_health_no_except_pass(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = """def foo():
    try:
        x = 1
    except ValueError:
        x = 0
"""
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health > 0.9

    def test_assess_health_no_deduction_triggers(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = """def foo():
    x = 1
    return x
"""
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health > 0.9

    def test_assess_health_todo_detected(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = """def foo():
    # TODO: fix this later
    pass
"""
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health < 1.0

    def test_assess_health_no_issues(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = """def foo():
    x = 1
    return x
"""
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health > 0.9

    def test_assess_health_clamped_at_0_1(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = "\n".join(["    # TODO" for _ in range(60)])
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health >= 0.1

    def test_assess_health_duplicate_except_lines(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        source = """try:
    x = 1
except:
    pass
try:
    y = 2
except:
    pass
"""
        tree = ast.parse(source)
        health = engine._assess_health(source, tree)
        assert health < 0.95


# ============================================================
# self_awareness.py — identify_blind_spots lock
# ============================================================

class TestMetaCognitiveBlindSpots:
    def test_identify_blind_spots_no_crash(self):
        from core.self_awareness import MetaCognitiveEngine

        engine = MetaCognitiveEngine()
        engine.record_reasoning_event("analytical", "test", "success", 0.3)
        engine.record_failure("test", "analytical", "error")

        spots = engine.identify_blind_spots()
        assert isinstance(spots, list)

    def test_identify_blind_spots_thread_safe(self):
        from core.self_awareness import MetaCognitiveEngine

        engine = MetaCognitiveEngine()
        errors = []

        def write_events():
            try:
                for i in range(50):
                    engine.record_reasoning_event("analytical", f"ctx_{i}", "ok", 0.4)
                    engine.record_failure(f"task_{i}", "approach", "err")
            except Exception as e:
                errors.append(e)

        def read_blind_spots():
            try:
                for _ in range(50):
                    spots = engine.identify_blind_spots()
                    assert isinstance(spots, list)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_events),
            threading.Thread(target=read_blind_spots),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_low_confidence_detection(self):
        from core.self_awareness import MetaCognitiveEngine

        engine = MetaCognitiveEngine()
        for _ in range(10):
            engine.record_reasoning_event("analytical", "test", "success", 0.2)

        spots = engine.identify_blind_spots()
        assert any("analytical" in s for s in spots)


# ============================================================
# self_awareness.py — register_multimodal bounds
# ============================================================

class TestRegisterMultimodal:
    def test_register_multimodal_dedupes(self):
        from core.self_awareness import SelfAwarenessEngine

        engine = SelfAwarenessEngine()
        engine.register_multimodal(["vision", "audio"])
        engine.register_multimodal(["vision", "haptic"])

        assert "vision" in engine._capabilities.multimodal
        assert engine._capabilities.multimodal.count("vision") == 1
        assert "audio" in engine._capabilities.multimodal
        assert "haptic" in engine._capabilities.multimodal

    def test_register_multimodal_bounded(self):
        from core.self_awareness import SelfAwarenessEngine

        engine = SelfAwarenessEngine()
        for i in range(30):
            engine.register_multimodal([f"cap_{i}"])

        assert len(engine._capabilities.multimodal) <= 20


# ============================================================
# knowledge_graph.py — add_insight no I/O under outer lock
# ============================================================

class TestKnowledgeGraphAddInsight:
    def test_add_insight_creates_nodes(self):
        from core.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        node_id = kg.add_insight(
            insight_id="test_1",
            content="A pattern involving primes and graph coloring",
            problem_id="Is there a bound?",
            category="number_theory",
        )

        assert node_id is not None
        assert len(kg.nodes) > 1

    def test_add_insight_with_strategy(self):
        from core.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        node_id = kg.add_insight(
            insight_id="test_2",
            content="Using induction on graph parameters",
            strategy="induction",
        )

        assert node_id is not None
        assert len(kg.nodes) > 1

    def test_add_insight_no_lock_deadlock(self):
        from core.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()

        errors = []

        def add_insights():
            try:
                for i in range(10):
                    kg.add_insight(
                        insight_id=f"ins_{i}",
                        content=f"Test insight {i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_insights) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors


# ============================================================
# Comprehensive regression: existing tests still pass
# ============================================================

class TestRound11Regressions:
    def test_introspection_module_still_works(self):
        from core.self_awareness import IntrospectionEngine

        engine = IntrospectionEngine()
        result = engine.introspect_module("self_awareness")
        assert result is not None
        assert result.lines_of_code > 0
        assert result.functions > 0

    def test_capability_registry_thread_safe(self):
        from core.self_awareness import CapabilityRegistry, CapabilityType

        registry = CapabilityRegistry()
        errors = []

        def register():
            try:
                for i in range(50):
                    registry.register_capability(
                        f"cap_{i}", f"desc_{i}", CapabilityType.CORE
                    )
            except Exception as e:
                errors.append(e)

        def read():
            try:
                for _ in range(50):
                    caps = registry.get_capabilities()
                    assert isinstance(caps, list)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register),
            threading.Thread(target=read),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors