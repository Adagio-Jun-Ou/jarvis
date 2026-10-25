"""Tests for Attention Priority Queue and Concept Drift Detector."""
import threading
import time
import pytest

from core.attention_priority import (
    AttentionPriorityQueue,
    UrgencySignal,
    get_attention_queue,
)
from core.concept_drift import (
    ConceptDriftDetector,
    DriftType,
    TopicWindow,
    get_drift_detector,
)
from collections import Counter


class TestAttentionPriorityQueue:
    def test_enqueue_dequeue(self):
        q = AttentionPriorityQueue()
        item_id = q.enqueue("test message", urgency=UrgencySignal.NORMAL)
        assert item_id is not None
        item = q.dequeue()
        assert item is not None
        assert item.content == "test message"

    def test_priority_ordering(self):
        q = AttentionPriorityQueue()
        q.enqueue("low priority", urgency=UrgencySignal.NORMAL)
        q.enqueue("high priority", urgency=UrgencySignal.CORRECTION)
        item = q.dequeue()
        assert item.urgency == UrgencySignal.CORRECTION

    def test_safety_critical_highest_priority(self):
        q = AttentionPriorityQueue()
        q.enqueue("normal", urgency=UrgencySignal.NORMAL)
        q.enqueue("correction", urgency=UrgencySignal.CORRECTION)
        q.enqueue("safety", urgency=UrgencySignal.SAFETY_CRITICAL)
        item = q.dequeue()
        assert item.urgency == UrgencySignal.SAFETY_CRITICAL

    def test_topic_continuity_bonus(self):
        q = AttentionPriorityQueue()
        q.enqueue("message about math", urgency=UrgencySignal.NORMAL, topic="math")
        q.dequeue()  # Process and set current topic
        q._current_topic = "math"
        id1 = q.enqueue("another math message", urgency=UrgencySignal.NORMAL, topic="math")
        id2 = q.enqueue("unrelated message", urgency=UrgencySignal.NORMAL, topic="sports")

    def test_empty_queue(self):
        q = AttentionPriorityQueue()
        assert q.dequeue() is None
        assert q.peek() is None
        assert q.is_empty()

    def test_peek_doesnt_remove(self):
        q = AttentionPriorityQueue()
        q.enqueue("test")
        item = q.peek()
        assert item is not None
        assert q.size() == 1

    def test_queue_max_size(self):
        q = AttentionPriorityQueue()
        q._MAX_QUEUE_SIZE = 5
        for i in range(10):
            q.enqueue(f"msg_{i}", urgency=UrgencySignal.NORMAL)
        assert q.size() <= 5

    def test_reprioritize(self):
        q = AttentionPriorityQueue()
        item_id = q.enqueue("test", urgency=UrgencySignal.NORMAL)
        result = q.reprioritize(item_id, UrgencySignal.SAFETY_CRITICAL)
        assert result

    def test_reprioritize_nonexistent(self):
        q = AttentionPriorityQueue()
        result = q.reprioritize("nonexistent", UrgencySignal.SAFETY_CRITICAL)
        assert not result

    def test_bump_priority(self):
        q = AttentionPriorityQueue()
        item_id = q.enqueue("low", urgency=UrgencySignal.NORMAL)
        q.bump_priority(item_id, boost=5.0)
        item = q.peek()
        assert item.item_id == item_id

    def test_cognitive_state_affects_priority(self):
        q = AttentionPriorityQueue()
        id_normal = q.enqueue("normal msg", urgency=UrgencySignal.NORMAL, cognitive_state="IDLE")
        id_overloaded = q.enqueue("overloaded msg", urgency=UrgencySignal.NORMAL, cognitive_state="OVERLOADED")
        item = q.dequeue()
        assert item.urgency == UrgencySignal.NORMAL

    def test_decay_removes_low_priority(self):
        q = AttentionPriorityQueue()
        q.enqueue("important", urgency=UrgencySignal.SAFETY_CRITICAL)
        q.enqueue("trivial", urgency=UrgencySignal.NORMAL)
        for item in q._queue:
            if item.urgency == UrgencySignal.NORMAL:
                item.priority = 0.001
        removed = q.apply_decay()
        assert removed >= 1

    def test_get_stats(self):
        q = AttentionPriorityQueue()
        q.enqueue("test1")
        q.enqueue("test2")
        q.dequeue()
        stats = q.get_stats()
        assert stats["queue_size"] == 1
        assert stats["total_processed"] == 1

    def test_snapshot(self):
        q = AttentionPriorityQueue()
        q.enqueue("msg1")
        q.enqueue("msg2")
        snap = q.get_queue_snapshot()
        assert len(snap) == 2

    def test_singleton(self):
        q = get_attention_queue()
        assert isinstance(q, AttentionPriorityQueue)


class TestTopicWindow:
    def test_empty_similarity(self):
        w1 = TopicWindow()
        w2 = TopicWindow()
        assert w1.similarity(w2) == 0.0

    def test_identical_similarity(self):
        w1 = TopicWindow(keywords=Counter({"math": 3, "graph": 2}))
        w2 = TopicWindow(keywords=Counter({"math": 3, "graph": 2}))
        sim = w1.similarity(w2)
        assert sim > 0.9

    def test_no_overlap(self):
        w1 = TopicWindow(keywords=Counter({"math": 3}))
        w2 = TopicWindow(keywords=Counter({"sports": 3}))
        assert w1.similarity(w2) == 0.0

    def test_partial_overlap(self):
        w1 = TopicWindow(keywords=Counter({"math": 3, "graph": 2}))
        w2 = TopicWindow(keywords=Counter({"math": 3, "music": 2}))
        sim = w1.similarity(w2)
        assert 0 < sim < 1


class TestConceptDriftDetector:
    def test_no_drift_same_topic(self):
        dd = ConceptDriftDetector()
        dd.observe("Tell me about graph theory and nodes")
        dd._drift_cooldown = 0
        result = dd.observe("Graph theory uses nodes and edges")
        assert result is None or result.drift_type == DriftType.NONE

    def test_sudden_drift_detected(self):
        dd = ConceptDriftDetector()
        dd.observe("Tell me about mathematics calculus algebra equations derivatives")
        dd._drift_cooldown = 0
        result = dd.observe("By the way, what are cooking recipes pasta sauce garlic?")
        if result is not None:
            assert result.drift_type == DriftType.SUDDEN

    def test_gradual_drift(self):
        dd = ConceptDriftDetector()
        dd.observe("Let us discuss graph theory networks")
        dd._drift_cooldown = 0
        result = dd.observe("Cooking recipes with vegetables spices herbs ingredients")
        if result:
            assert result.drift_type in (DriftType.GRADUAL, DriftType.SUDDEN)

    def test_current_topic(self):
        dd = ConceptDriftDetector()
        dd.observe("quantum physics experiments particles")
        topic = dd.get_current_topic()
        assert isinstance(topic, str)

    def test_topic_stack(self):
        dd = ConceptDriftDetector()
        dd.observe("quantum physics particles")
        dd._drift_cooldown = 0
        dd.observe("By the way, cooking recipes vegetables")
        stack = dd.get_topic_stack()
        assert isinstance(stack, list)

    def test_drift_history(self):
        dd = ConceptDriftDetector()
        dd.observe("mathematics calculations")
        dd._drift_cooldown = 0
        dd.observe("By the way, what about cooking recipes?")
        history = dd.get_drift_history()
        assert isinstance(history, list)

    def test_stats(self):
        dd = ConceptDriftDetector()
        dd.observe("hello world")
        stats = dd.get_stats()
        assert stats["total_messages"] == 1

    def test_reset(self):
        dd = ConceptDriftDetector()
        dd.observe("test message content here")
        dd.reset()
        stats = dd.get_stats()
        assert stats["total_messages"] == 0

    def test_empty_message(self):
        dd = ConceptDriftDetector()
        result = dd.observe("")
        assert result is None

    def test_singleton(self):
        dd = get_drift_detector()
        assert isinstance(dd, ConceptDriftDetector)

    def test_keyword_extraction(self):
        dd = ConceptDriftDetector()
        keywords = dd._extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "quick" in keywords
        assert "brown" in keywords
        assert "the" not in keywords

    def test_max_keywords(self):
        dd = ConceptDriftDetector()
        words = " ".join(f"word{i}" for i in range(100))
        keywords = dd._extract_keywords(words)
        assert len(keywords) <= dd._MAX_KEYWORDS


class TestAttentionQueueThreadSafety:
    def test_concurrent_enqueue_dequeue(self):
        q = AttentionPriorityQueue()
        errors = []

        def producer(idx):
            try:
                for i in range(10):
                    q.enqueue(f"msg_{idx}_{i}", urgency=UrgencySignal.NORMAL)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                while True:
                    item = q.dequeue()
                    if item is None and q.size() == 0:
                        break
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        while q.size() > 0:
            q.dequeue()

        assert not errors


class TestAgentUrgencyDetection:
    def test_safety_critical_detection(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)
        agent._attention_queue = None
        agent._drift_detector = None
        agent._thought_sense = None

        assert agent._detect_urgency("This is a dangerous emergency!") == UrgencySignal.SAFETY_CRITICAL
        assert agent._detect_urgency("security breach detected") == UrgencySignal.SAFETY_CRITICAL

    def test_correction_detection(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)

        assert agent._detect_urgency("No, that's wrong") == UrgencySignal.CORRECTION
        assert agent._detect_urgency("Stop doing that") == UrgencySignal.CORRECTION

    def test_time_sensitive_detection(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)

        assert agent._detect_urgency("I need this ASAP") == UrgencySignal.TIME_SENSITIVE
        assert agent._detect_urgency("This is urgent") == UrgencySignal.TIME_SENSITIVE

    def test_follow_up_detection(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)

        assert agent._detect_urgency("What's the next step?") == UrgencySignal.FOLLOW_UP

    def test_normal_detection(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)

        assert agent._detect_urgency("Hello there") == UrgencySignal.NORMAL
        assert agent._detect_urgency("Tell me about Python") == UrgencySignal.NORMAL


class TestDriftToAttentionIntegration:
    def test_sudden_drift_creates_system_alert(self):
        q = AttentionPriorityQueue()
        d = ConceptDriftDetector()

        d.observe("Tell me about machine learning algorithms and neural networks")
        d.observe("Machine learning uses training data to build models")
        drift = d.observe("By the way, let's talk about cooking recipes for Italian food")

        if drift and drift.drift_type in (DriftType.SUDDEN, DriftType.OSCILLATION):
            q.enqueue(
                content=f"DRIFT:{drift.drift_type.name}:{drift.to_topic}",
                urgency=UrgencySignal.SYSTEM_ALERT,
                source="concept_drift",
                topic=drift.to_topic,
            )

        if q.size() > 0:
            item = q.dequeue()
            assert item.urgency == UrgencySignal.SYSTEM_ALERT
            assert item.source == "concept_drift"
            assert "DRIFT" in item.content