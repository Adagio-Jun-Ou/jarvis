"""Daydream Sketch Pad - Quick insight capture and concept testing.

When Jarvis daydreams, it can:
1. Sketch down key insights quickly
2. Test concepts before committing
3. Validate ideas against known results
"""
import logging
from collections import deque
from core.helpers import atomic_write_json, read_json_with_integrity
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SketchType(Enum):
    INSIGHT = "insight"
    CONJECTURE = "conjecture"
    COUNTEREXAMPLE = "counterexample"
    METHOD_IDEA = "method_idea"
    CONNECTION = "connection"
    PATTERN = "pattern"
    QUESTION = "question"


class ConceptStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    REFUTED = "refuted"
    PROMOTED = "promoted"


@dataclass
class InsightSketch:
    id: str
    sketch_type: SketchType
    title: str
    content: str
    mathematical_context: str
    tags: list[str]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    tested: bool = False
    test_result: str | None = None
    status: ConceptStatus = ConceptStatus.DRAFT
    source_problem: str | None = None
    related_sketches: list[str] = field(default_factory=list)
    promoted_to: str | None = None


@dataclass
class ConceptTest:
    id: str
    sketch_id: str
    test_type: str
    test_input: Any
    expected_output: Any
    actual_output: Any | None = None
    passed: bool | None = None
    notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class DaydreamSketchPad:
    """Quick capture and testing for daydream insights.
    
    Features:
    - Fast insight sketching with minimal overhead
    - Quick concept tests (numerical, logical, counterexample)
    - Promotion system for validated insights
    - Connection discovery between sketches
    """

    MAX_SKETCHES = 200
    MAX_TESTS = 500

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "sketches"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.sketches: deque[InsightSketch] = deque(maxlen=self.MAX_SKETCHES)
        self.tests: deque[ConceptTest] = deque(maxlen=self.MAX_TESTS)
        self._sketch_counter = 0
        self._test_counter = 0

        self._lock = threading.RLock()

        self._load_state()

    def _load_state(self) -> None:
        state_file = self.storage_path / "sketch_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._sketch_counter = data.get("sketch_counter", 0)
                self._test_counter = data.get("test_counter", 0)
            except Exception as e:
                logger.warning(f"Could not load sketch state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "sketch_state.json"
        try:
            atomic_write_json(state_file, {
                    "sketch_counter": self._sketch_counter,
                    "test_counter": self._test_counter,
                    "total_sketches": len(self.sketches),
                    "total_tests": len(self.tests),
                })
        except Exception as e:
            logger.warning(f"Could not save sketch state: {e}")

    def sketch(
        self,
        sketch_type: SketchType,
        title: str,
        content: str,
        mathematical_context: str = "",
        tags: list[str] | None = None,
        confidence: float = 0.5,
        source_problem: str | None = None,
    ) -> InsightSketch:
        """Quickly sketch an insight."""
        with self._lock:
            self._sketch_counter += 1
            sketch_id = f"sketch_{self._sketch_counter:06d}"

        sketch = InsightSketch(
            id=sketch_id,
            sketch_type=sketch_type,
            title=title,
            content=content,
            mathematical_context=mathematical_context,
            tags=tags or [],
            confidence=confidence,
            source_problem=source_problem,
        )

        with self._lock:
            self.sketches.append(sketch)

        self._save_state()

        logger.debug(f"Sketched: {sketch_type.value} - {title[:50]}")
        return sketch

    def sketch_insight(self, content: str, context: str = "", confidence: float = 0.5) -> InsightSketch:
        """Quick insight capture."""
        return self.sketch(
            sketch_type=SketchType.INSIGHT,
            title=content[:50],
            content=content,
            mathematical_context=context,
            confidence=confidence,
        )

    def sketch_conjecture(self, statement: str, context: str = "") -> InsightSketch:
        """Sketch a mathematical conjecture."""
        return self.sketch(
            sketch_type=SketchType.CONJECTURE,
            title=statement[:50],
            content=statement,
            mathematical_context=context,
            confidence=0.3,
        )

    def sketch_method(self, name: str, description: str, context: str = "") -> InsightSketch:
        """Sketch a method idea."""
        return self.sketch(
            sketch_type=SketchType.METHOD_IDEA,
            title=name,
            content=description,
            mathematical_context=context,
            confidence=0.4,
        )

    def sketch_connection(self, concept_a: str, concept_b: str, reason: str) -> InsightSketch:
        """Sketch a connection between concepts."""
        return self.sketch(
            sketch_type=SketchType.CONNECTION,
            title=f"{concept_a} ↔ {concept_b}",
            content=reason,
            mathematical_context=f"Connection between {concept_a} and {concept_b}",
            tags=[concept_a, concept_b],
            confidence=0.5,
        )

    def test_numerical(
        self,
        sketch_id: str,
        test_values: list[tuple[Any, Any]],
        test_function: callable,
    ) -> ConceptTest:
        """Test a concept with numerical values."""
        sketch = self._get_sketch(sketch_id)
        if not sketch:
            raise ValueError(f"Sketch {sketch_id} not found")

        with self._lock:
            self._test_counter += 1
            test_id = f"test_{self._test_counter:06d}"

        passed_count = 0
        results = []

        for input_val, expected in test_values:
            try:
                actual = test_function(input_val)
                passed = actual == expected
                if passed:
                    passed_count += 1
                results.append({
                    "input": input_val,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                })
            except Exception as e:
                results.append({
                    "input": input_val,
                    "expected": expected,
                    "actual": str(e),
                    "passed": False,
                })

        total = len(test_values)
        all_passed = passed_count == total

        test = ConceptTest(
            id=test_id,
            sketch_id=sketch_id,
            test_type="numerical",
            test_input=test_values,
            expected_output=[e for _, e in test_values],
            actual_output=results,
            passed=all_passed,
            notes=f"Passed {passed_count}/{total} tests",
        )

        with self._lock:
            sketch.tested = True
            sketch.test_result = test.notes
            sketch.status = ConceptStatus.VALIDATED if all_passed else ConceptStatus.REFUTED

            self.tests.append(test)
            self._save_state()

        return test

    def test_counterexample(
        self,
        sketch_id: str,
        counterexample: Any,
        explanation: str = "",
    ) -> ConceptTest:
        """Test for counterexample to refute a conjecture."""
        sketch = self._get_sketch(sketch_id)
        if not sketch:
            raise ValueError(f"Sketch {sketch_id} not found")

        with self._lock:
            self._test_counter += 1
            test_id = f"test_{self._test_counter:06d}"

        test = ConceptTest(
            id=test_id,
            sketch_id=sketch_id,
            test_type="counterexample",
            test_input=counterexample,
            expected_output=None,
            actual_output=explanation,
            passed=True,
            notes=f"Counterexample found: {explanation}",
        )

        with self._lock:
            sketch.tested = True
            sketch.test_result = f"Counterexample: {explanation}"
            sketch.status = ConceptStatus.REFUTED
            sketch.confidence = 0.0

            self.tests.append(test)
            self._save_state()

        return test

    def test_prime_check(self, sketch_id: str, n: int, claim: bool) -> ConceptTest:
        """Test a prime number claim."""
        def is_prime(num: int) -> bool:
            if num < 2:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for i in range(3, int(num ** 0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True

        actual = is_prime(n)
        passed = actual == claim

        with self._lock:
            self._test_counter += 1
            test_id = f"test_{self._test_counter:06d}"

        sketch = self._get_sketch(sketch_id)

        test = ConceptTest(
            id=test_id,
            sketch_id=sketch_id,
            test_type="prime_check",
            test_input=n,
            expected_output=claim,
            actual_output=actual,
            passed=passed,
            notes=f"{n} is {'prime' if actual else 'not prime'}, claimed {'prime' if claim else 'not prime'}",
        )

        if sketch:
            with self._lock:
                sketch.tested = True
                sketch.test_result = test.notes
                sketch.status = ConceptStatus.VALIDATED if passed else ConceptStatus.REFUTED

                self.tests.append(test)
                self._save_state()

        return test

    def promote(self, sketch_id: str, promotion_target: str = "insight_bridge") -> bool:
        """Promote a validated sketch to a permanent store."""
        sketch = self._get_sketch(sketch_id)
        if not sketch:
            return False

        if sketch.status != ConceptStatus.VALIDATED:
            logger.warning(f"Cannot promote sketch {sketch_id}: status is {sketch.status.value}")
            return False

        if promotion_target == "insight_bridge":
            try:
                from core.insight_bridge import get_insight_bridge

                bridge = get_insight_bridge()

                bridge.process_insight(
                    raw_insight=sketch.content,
                    source_problem=sketch.source_problem or "daydream_sketch",
                    source_category=sketch.mathematical_context or "unknown",
                    problem_solved=False,
                )

                sketch.status = ConceptStatus.PROMOTED
                sketch.promoted_to = "insight_bridge"
                self._save_state()

                logger.info(f"Promoted sketch {sketch_id} to insight_bridge")
                return True

            except Exception as e:
                logger.error(f"Failed to promote sketch: {e}")
                return False

        return False

    def find_connections(self, min_shared_tags: int = 2) -> list[tuple[str, str, list[str]]]:
        """Find sketches that might be connected via shared tags."""
        connections = []

        for i, sketch_a in enumerate(self.sketches):
            for sketch_b in list(self.sketches)[i + 1:]:
                shared = set(sketch_a.tags) & set(sketch_b.tags)
                if len(shared) >= min_shared_tags:
                    connections.append((sketch_a.id, sketch_b.id, list(shared)))

        return connections

    def get_drafts(self) -> list[InsightSketch]:
        """Get all untested draft sketches."""
        return [s for s in self.sketches if s.status == ConceptStatus.DRAFT]

    def get_validated(self) -> list[InsightSketch]:
        """Get all validated sketches ready for promotion."""
        return [s for s in self.sketches if s.status == ConceptStatus.VALIDATED]

    def _get_sketch(self, sketch_id: str) -> InsightSketch | None:
        with self._lock:
            for s in self.sketches:
                if s.id == sketch_id:
                    return s
        return None

    def get_statistics(self) -> dict:
        with self._lock:
            by_type: dict[str, int] = {}
            by_status: dict[str, int] = {}

            for s in self.sketches:
                t = s.sketch_type.value
                by_type[t] = by_type.get(t, 0) + 1
                st = s.status.value
                by_status[st] = by_status.get(st, 0) + 1

            return {
                "total_sketches": len(self.sketches),
                "total_tests": len(self.tests),
                "validated": len(self.get_validated()),
                "drafts": len(self.get_drafts()),
                "by_type": by_type,
                "by_status": by_status,
            }


_sketch_pad: DaydreamSketchPad | None = None
_sketch_lock = threading.Lock()


def get_sketch_pad() -> DaydreamSketchPad:
    """Get global sketch pad instance (thread-safe)."""
    global _sketch_pad
    if _sketch_pad is None:
        with _sketch_lock:
            if _sketch_pad is None:
                _sketch_pad = DaydreamSketchPad()
    return _sketch_pad
