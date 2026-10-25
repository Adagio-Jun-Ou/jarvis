"""Enhanced Prefrontal Cortex - Advanced Reasoning Module.

Extends the base Prefrontal Cortex with:
- Multi-step reasoning chains
- Causal inference
- Hypothesis generation and testing
- Counterfactual reasoning
- Abstract pattern recognition
"""
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning the cortex can perform."""
    DEDUCTIVE = "deductive"  # General to specific
    INDUCTIVE = "inductive"  # Specific to general
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # Similar cases
    CAUSAL = "causal"  # Cause and effect
    COUNTERFACTUAL = "counterfactual"  # What if
    META_COGNITIVE = "meta_cognitive"  # Thinking about thinking


class ConfidenceLevel(Enum):
    """Confidence levels for conclusions."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class Hypothesis:
    """A testable hypothesis."""
    id: str
    statement: str
    reasoning_type: ReasoningType
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    confidence: float = 0.5
    tested: bool = False
    test_results: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    id: str
    goal: str
    steps: list[dict] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    conclusion: Optional[str] = None
    confidence: float = 0.0
    successful: bool = False
    
    def add_step(self, step_type: str, content: str, evidence: dict = None) -> None:
        """Add a reasoning step."""
        self.steps.append({
            "type": step_type,
            "content": content,
            "evidence": evidence or {},
            "timestamp": datetime.now().isoformat(),
        })


@dataclass
class CausalRelation:
    """A causal relationship between concepts."""
    cause: str
    effect: str
    strength: float  # 0.0 to 1.0
    mechanism: Optional[str] = None
    evidence: list[str] = field(default_factory=list)
    reversible: bool = False


class ReasoningEngine:
    """Advanced reasoning capabilities for Prefrontal Cortex.
    
    Provides:
    - Multi-step deductive/inductive reasoning
    - Hypothesis generation and testing
    - Causal inference
    - Analogical reasoning
    - Counterfactual analysis
    
    Usage:
        engine = ReasoningEngine()
        
        # Generate hypothesis
        hypothesis = engine.generate_hypothesis(
            observation="User is frustrated",
            context=["Multiple failed attempts", "Slow responses"]
        )
        
        # Test hypothesis
        result = engine.test_hypothesis(hypothesis)
        
        # Generate reasoning chain
        chain = engine.reason("Why is the system slow?")
    """
    
    MAX_HYPOTHESES = 50
    MAX_CAUSAL_GRAPH_SIZE = 100
    MAX_REASONING_HISTORY = 100
    
    def __init__(self):
        self._hypotheses: dict[str, Hypothesis] = {}
        self._causal_graph: dict[str, list[CausalRelation]] = {}
        self._reasoning_history: deque = deque(maxlen=self.MAX_REASONING_HISTORY)
        self._patterns: dict[str, int] = {}  # pattern -> count
        self._lock = threading.Lock()
        
        # Reasoning templates
        self._templates = self._init_templates()
    
    def _init_templates(self) -> dict[str, list[str]]:
        """Initialize reasoning templates."""
        return {
            "diagnostic": [
                "If {symptom}, then possible causes are {causes}",
                "Observing {observation} suggests {hypothesis}",
                "Given {evidence}, {conclusion} is likely",
            ],
            "causal": [
                "{cause} leads to {effect} because {mechanism}",
                "{event1} triggers {event2} through {process}",
                "When {condition}, {outcome} occurs",
            ],
            "analogical": [
                "{case1} is similar to {case2} in that {shared_features}",
                "Just as {analogy_source}, {analogy_target}",
                "{pattern} here resembles {known_pattern}",
            ],
            "counterfactual": [
                "If {counterfactual_condition}, then {counterfactual_outcome}",
                "Without {factor}, {alternative_outcome}",
                "Had {past_condition}, then {past_outcome}",
            ],
        }
    
    def generate_hypothesis(
        self,
        observation: str,
        context: list[str],
        reasoning_type: ReasoningType = ReasoningType.ABDUCTIVE,
    ) -> Hypothesis:
        """Generate a hypothesis to explain an observation.
        
        Args:
            observation: What we're trying to explain
            context: Relevant context/information
            reasoning_type: Type of reasoning to use
            
        Returns:
            A testable hypothesis
        """
        import hashlib
        hyp_id = hashlib.sha256(f"{observation}:{context}".encode()).hexdigest()[:12]
        
        # Generate hypothesis statement based on reasoning type
        if reasoning_type == ReasoningType.ABDUCTIVE:
            # Best explanation reasoning
            statement = self._abduce_best_explanation(observation, context)
        elif reasoning_type == ReasoningType.DEDUCTIVE:
            # Logical deduction
            statement = self._deduce_conclusion(observation, context)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            # Generalization from examples
            statement = self._induce_pattern(observation, context)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            # Similarity-based
            statement = self._analogize(observation, context)
        else:
            statement = f"Hypothesis: {observation} caused by {context[0] if context else 'unknown'}"
        
        hypothesis = Hypothesis(
            id=hyp_id,
            statement=statement,
            reasoning_type=reasoning_type,
            evidence_for=context,
            confidence=0.5,
        )
        
        with self._lock:
            if len(self._hypotheses) < self.MAX_HYPOTHESES:
                self._hypotheses[hyp_id] = hypothesis
        
        logger.debug(f"Generated hypothesis: {statement[:80]}...")
        return hypothesis
    
    def _abduce_best_explanation(self, observation: str, context: list[str]) -> str:
        """Generate best explanation for observation (abductive reasoning)."""
        # Pattern matching on context
        if any("slow" in c.lower() for c in context):
            return f"{observation} likely caused by performance bottleneck"
        elif any("error" in c.lower() for c in context):
            return f"{observation} caused by error in processing"
        elif any("memory" in c.lower() for c in context):
            return f"{observation} related to memory state"
        else:
            return f"{observation} explained by {context[0] if context else 'unknown factors'}"
    
    def _deduce_conclusion(self, observation: str, context: list[str]) -> str:
        """Deduce conclusion from premises (deductive reasoning)."""
        # Simple deductive pattern
        return f"Given {', '.join(context[:2])}, it follows that {observation}"
    
    def _induce_pattern(self, observation: str, context: list[str]) -> str:
        """Induce general pattern from examples (inductive reasoning)."""
        return f"Pattern: {observation} with common factors {', '.join(context[:3])}"
    
    def _analogize(self, observation: str, context: list[str]) -> str:
        """Draw analogy to known situations (analogical reasoning)."""
        return f"{observation} similar to known patterns: {', '.join(context[:2])}"
    
    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        test_data: Optional[dict] = None,
    ) -> dict:
        """Test a hypothesis against evidence.
        
        Args:
            hypothesis: The hypothesis to test
            test_data: Additional test data/evidence
            
        Returns:
            Test results with confidence update
        """
        results = {
            "tested": True,
            "test_time": datetime.now().isoformat(),
            "evidence_found": [],
            "evidence_against": [],
            "confidence_change": 0.0,
        }
        
        # Test against existing evidence
        supported = len(hypothesis.evidence_for)
        contradicted = len(hypothesis.evidence_against)
        
        # Update confidence based on evidence
        if supported > contradicted:
            hypothesis.confidence = min(1.0, hypothesis.confidence + 0.1)
            results["confidence_change"] = 0.1
        elif contradicted > supported:
            hypothesis.confidence = max(0.0, hypothesis.confidence - 0.15)
            results["confidence_change"] = -0.15
        
        hypothesis.tested = True
        hypothesis.test_results = results
        
        logger.debug(f"Tested hypothesis {hypothesis.id}: confidence={hypothesis.confidence:.2f}")
        return results
    
    def reason(
        self,
        goal: str,
        context: dict[str, Any],
        max_steps: int = 5,
    ) -> ReasoningChain:
        """Perform multi-step reasoning toward a goal.
        
        Args:
            goal: What we want to reason about
            context: Available context/evidence
            max_steps: Maximum reasoning steps
            
        Returns:
            A reasoning chain with steps and conclusion
        """
        import hashlib
        chain_id = hashlib.sha256(goal.encode()).hexdigest()[:12]
        
        chain = ReasoningChain(
            id=chain_id,
            goal=goal,
        )
        
        # Step 1: Analyze goal
        chain.add_step("analyze", f"Goal: {goal}", context)
        
        # Step 2: Gather evidence.
        # This used to read only a key literally named "evidence", so a caller
        # passing ordinary context — reason(goal, {"density": "increased"}) —
        # had all of it silently discarded and the hypothesis was generated from
        # an empty list. Honour an explicit "evidence" key when present, and
        # otherwise treat the context entries themselves as the evidence.
        raw_evidence = context.get("evidence") if isinstance(context, dict) else None
        if raw_evidence is None:
            evidence = (
                [f"{key}: {value}" for key, value in context.items()]
                if isinstance(context, dict) else []
            )
        elif isinstance(raw_evidence, list):
            evidence = raw_evidence
        else:
            evidence = [str(raw_evidence)]
        chain.add_step("gather", f"Evidence: {len(evidence)} items", {"evidence": evidence})

        # Step 3: Generate hypothesis
        hypothesis = self.generate_hypothesis(goal, evidence)
        chain.hypotheses.append(hypothesis)
        chain.add_step("hypothesize", hypothesis.statement, {"hypothesis_id": hypothesis.id})
        
        # Step 4: Test hypothesis
        test_result = self.test_hypothesis(hypothesis)
        chain.add_step("test", f"Test result: confidence={hypothesis.confidence:.2f}", test_result)
        
        # Step 5: Draw conclusion
        if hypothesis.confidence >= ConfidenceLevel.HIGH.value:
            chain.conclusion = hypothesis.statement
            chain.confidence = hypothesis.confidence
            chain.successful = True
        else:
            chain.conclusion = f"Insufficient confidence: {hypothesis.confidence:.2f}"
            chain.confidence = hypothesis.confidence
            chain.successful = False
        
        chain.add_step("conclude", chain.conclusion, {"confidence": chain.confidence})
        
        # Store in history
        with self._lock:
            self._reasoning_history.append(chain)
        
        return chain
    
    def add_causal_relation(
        self,
        cause: str,
        effect: str,
        strength: float = 0.7,
        mechanism: Optional[str] = None,
    ) -> None:
        """Add a causal relation to the graph.
        
        Args:
            cause: The cause
            effect: The effect
            strength: Strength of relationship (0-1)
            mechanism: How cause leads to effect
        """
        relation = CausalRelation(
            cause=cause,
            effect=effect,
            strength=strength,
            mechanism=mechanism,
        )
        
        with self._lock:
            if cause not in self._causal_graph:
                self._causal_graph[cause] = []
            
            if len(self._causal_graph) < self.MAX_CAUSAL_GRAPH_SIZE:
                self._causal_graph[cause].append(relation)
                logger.debug(f"Added causal relation: {cause} -> {effect} ({strength})")
    
    def find_causes(self, effect: str) -> list[CausalRelation]:
        """Find potential causes for an effect.
        
        Args:
            effect: The effect to find causes for
            
        Returns:
            List of causal relations, sorted by strength
        """
        causes = []
        
        with self._lock:
            for cause, relations in self._causal_graph.items():
                for relation in relations:
                    if relation.effect == effect:
                        causes.append(relation)
        
        # Sort by strength
        causes.sort(key=lambda r: r.strength, reverse=True)
        return causes
    
    def find_effects(self, cause: str) -> list[CausalRelation]:
        """Find potential effects of a cause.
        
        Args:
            cause: The cause to find effects for
            
        Returns:
            List of causal relations, sorted by strength
        """
        with self._lock:
            return sorted(
                self._causal_graph.get(cause, []),
                key=lambda r: r.strength,
                reverse=True,
            )
    
    def counterfactual(
        self,
        actual_situation: str,
        counterfactual_condition: str,
    ) -> dict:
        """Perform counterfactual reasoning.
        
        What would have happened if X were different?
        
        Args:
            actual_situation: What actually happened
            counterfactual_condition: What could have been different
            
        Returns:
            Counterfactual analysis
        """
        analysis = {
            "actual": actual_situation,
            "counterfactual": counterfactual_condition,
            "prediction": None,
            "confidence": 0.0,
            "reasoning": [],
        }
        
        # Use causal graph to predict
        effects = self.find_effects(counterfactual_condition)
        
        if effects:
            # Predict most likely effect
            most_likely = effects[0]
            analysis["prediction"] = most_likely.effect
            analysis["confidence"] = most_likely.strength
            analysis["reasoning"].append(
                f"Given {counterfactual_condition}, {most_likely.effect} would likely occur"
            )
        else:
            # No causal knowledge, use pattern matching
            analysis["prediction"] = f"Alternative outcome of {counterfactual_condition}"
            analysis["confidence"] = 0.3
            analysis["reasoning"].append("No causal model available, low confidence")
        
        return analysis
    
    def analogical_reasoning(
        self,
        source_case: str,
        target_case: str,
        known_outcome: str,
    ) -> dict:
        """Perform analogical reasoning.
        
        If case A is similar to case B, and we know outcome for A,
        what can we infer about B?
        
        Args:
            source_case: The known case
            target_case: The new case to reason about
            known_outcome: What happened in source case
            
        Returns:
            Analogical reasoning result
        """
        result = {
            "source": source_case,
            "target": target_case,
            "source_outcome": known_outcome,
            "predicted_outcome": None,
            "similarity": 0.0,
            "confidence": 0.0,
        }
        
        # Pattern matching
        pattern_key = f"{source_case[:50]}:{known_outcome[:50]}"
        with self._lock:
            count = self._patterns.get(pattern_key, 0)
            self._patterns[pattern_key] = count + 1
        
        # Similarity heuristic based on pattern frequency and text overlap
        source_words = set(source_case.lower().split())
        target_words = set(target_case.lower().split())
        word_overlap = len(source_words & target_words) / max(len(source_words | target_words), 1)
        pattern_strength = min(count / 10.0, 1.0) if count > 0 else 0.0
        result["similarity"] = 0.3 * word_overlap + 0.7 * pattern_strength
        
        # Predict outcome
        result["predicted_outcome"] = f"Analogous to {known_outcome}"
        result["confidence"] = result["similarity"] * 0.8
        
        return result
    
    def get_hypotheses(self) -> list[Hypothesis]:
        """Get all hypotheses."""
        with self._lock:
            return list(self._hypotheses.values())
    
    def get_causal_graph_stats(self) -> dict:
        """Get statistics about the causal graph."""
        with self._lock:
            return {
                "nodes": len(self._causal_graph),
                "edges": sum(len(relations) for relations in self._causal_graph.values()),
                "patterns": len(self._patterns),
                "reasoning_history": len(self._reasoning_history),
            }
    
    def clear(self) -> None:
        """Clear all reasoning state."""
        with self._lock:
            self._hypotheses.clear()
            self._causal_graph.clear()
            self._reasoning_history.clear()
            self._patterns.clear()
        logger.info("Reasoning engine cleared")


# Singleton
_reasoning_engine: Optional[ReasoningEngine] = None
_engine_lock = threading.Lock()


def get_reasoning_engine() -> ReasoningEngine:
    """Get the singleton reasoning engine."""
    global _reasoning_engine
    if _reasoning_engine is None:
        with _engine_lock:
            if _reasoning_engine is None:
                _reasoning_engine = ReasoningEngine()
    return _reasoning_engine