"""Self-Growth and Evolution System for Jarvis.

Enables Jarvis to:
1. Identify areas for improvement
2. Suggest new capabilities to add
3. Learn from interactions
4. Ask permission before growing
5. Track growth history
6. Self-evaluate performance
7. Evolve based on feedback
8. Adapt to user preferences
"""
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
import hashlib
import random

from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)


class GrowthType(Enum):
    """Types of growth Jarvis can undergo."""
    NEW_SKILL = "new_skill"           # Add a new skill/tool
    KNOWLEDGE = "knowledge"            # Learn new information
    BEHAVIOR = "behavior"              # Improve behavior patterns
    INTEGRATION = "integration"        # Connect to new services
    OPTIMIZATION = "optimization"      # Improve existing capabilities
    PERSONALITY = "personality"        # Adjust personality traits
    MEMORY = "memory"                  # Better memory handling
    REASONING = "reasoning"            # Improve reasoning abilities
    EVOLUTION = "evolution"            # Fundamental capability evolution


class GrowthStatus(Enum):
    """Status of a growth proposal."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    FAILED = "failed"
    EVALUATING = "evaluating"


class EvaluationType(Enum):
    """Types of self-evaluation."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    USER_SATISFACTION = "user_satisfaction"
    LEARNING_PROGRESS = "learning_progress"
    CAPABILITY_GAP = "capability_gap"


class EvolutionStage(Enum):
    """Stages of evolution."""
    INFANT = "infant"           # Basic capabilities
    LEARNING = "learning"       # Actively learning
    ADAPTIVE = "adaptive"       # Can adapt to user
    AUTONOMOUS = "autonomous"   # Can self-improve
    EVOLVED = "evolved"         # Has evolved significantly


@dataclass
class GrowthProposal:
    """A proposed growth for Jarvis."""
    id: str
    growth_type: GrowthType
    title: str
    description: str
    rationale: str
    benefits: list[str]
    risks: list[str]
    effort: str  # low, medium, high
    dependencies: list[str]
    implementation_steps: list[str]
    status: GrowthStatus = GrowthStatus.PROPOSED
    created: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    implemented_at: Optional[datetime] = None
    feedback: Optional[str] = None


@dataclass
class GrowthMetric:
    """Metrics tracked for growth decisions."""
    name: str
    current_value: float
    target_value: float
    trend: str  # improving, declining, stable
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Result of a self-evaluation."""
    evaluation_type: EvaluationType
    score: float  # 0.0 to 1.0
    details: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evaluation_type": self.evaluation_type.value,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
        }


@dataclass
class EvolutionState:
    """Tracks the evolution state of Jarvis."""
    stage: EvolutionStage = EvolutionStage.INFANT
    generation: int = 1
    mutations_applied: int = 0
    successful_adaptations: int = 0
    failed_adaptations: int = 0
    fitness_score: float = 0.5
    last_evolution: datetime = field(default_factory=datetime.now)
    evolution_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "generation": self.generation,
            "mutations_applied": self.mutations_applied,
            "successful_adaptations": self.successful_adaptations,
            "failed_adaptations": self.failed_adaptations,
            "fitness_score": self.fitness_score,
            "last_evolution": self.last_evolution.isoformat(),
            "evolution_history": self.evolution_history[-20:],  # Keep last 20
        }


@dataclass
class AdaptationGene:
    """Represents an adaptable behavior or parameter."""
    name: str
    value: Any
    min_value: Any
    max_value: Any
    mutation_rate: float = 0.1
    fitness_contribution: float = 0.0

    def mutate(self) -> "AdaptationGene":
        """Mutate this gene within bounds."""
        if isinstance(self.value, (int, float)):
            delta = (self.max_value - self.min_value) * self.mutation_rate
            new_value = self.value + random.uniform(-delta, delta)
            new_value = max(self.min_value, min(self.max_value, new_value))
            return AdaptationGene(
                name=self.name,
                value=new_value,
                min_value=self.min_value,
                max_value=self.max_value,
                mutation_rate=self.mutation_rate,
                fitness_contribution=self.fitness_contribution,
            )
        return self


class EvaluationEngine:
    """Engine for self-evaluation and assessment."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.evaluations: list[EvaluationResult] = []
        self.evaluation_callbacks: dict[EvaluationType, Callable] = {}

    def register_evaluator(self, eval_type: EvaluationType, callback: Callable):
        """Register a custom evaluator function."""
        self.evaluation_callbacks[eval_type] = callback

    async def evaluate_performance(self, context: dict) -> EvaluationResult:
        """Evaluate overall performance."""
        # Analyze response times, success rates, etc.
        response_times = context.get("response_times", [])
        success_rate = context.get("success_rate", 0.8)
        error_rate = context.get("error_rate", 0.0)

        # Calculate score
        speed_score = 1.0 if not response_times else max(0, 1 - (sum(response_times) / len(response_times)) / 10)
        quality_score = success_rate * (1 - error_rate)
        overall_score = (speed_score + quality_score) / 2

        recommendations = []
        if speed_score < 0.7:
            recommendations.append("Optimize response generation for speed")
        if success_rate < 0.9:
            recommendations.append("Improve task completion reliability")
        if error_rate > 0.1:
            recommendations.append("Reduce error rate through better error handling")

        return EvaluationResult(
            evaluation_type=EvaluationType.PERFORMANCE,
            score=overall_score,
            details={
                "speed_score": speed_score,
                "quality_score": quality_score,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            },
            recommendations=recommendations,
        )

    async def evaluate_accuracy(self, context: dict) -> EvaluationResult:
        """Evaluate response accuracy."""
        correct_responses = context.get("correct_responses", 0)
        total_responses = context.get("total_responses", 1)
        user_corrections = context.get("user_corrections", 0)

        accuracy = correct_responses / max(total_responses, 1)
        correction_rate = user_corrections / max(total_responses, 1)

        score = accuracy * (1 - correction_rate * 0.5)

        recommendations = []
        if correction_rate > 0.2:
            recommendations.append("Improve understanding of user intent")
        if accuracy < 0.8:
            recommendations.append("Enhance knowledge retrieval accuracy")

        return EvaluationResult(
            evaluation_type=EvaluationType.ACCURACY,
            score=score,
            details={
                "accuracy": accuracy,
                "correction_rate": correction_rate,
            },
            recommendations=recommendations,
        )

    async def evaluate_efficiency(self, context: dict) -> EvaluationResult:
        """Evaluate resource and time efficiency."""
        tool_calls = context.get("tool_calls", [])
        redundant_calls = context.get("redundant_calls", 0)

        # Measure how efficiently tools are used
        if tool_calls:
            efficiency = max(0, 1 - (redundant_calls / len(tool_calls)))
        else:
            efficiency = 1.0

        recommendations = []
        if efficiency < 0.8:
            recommendations.append("Reduce redundant tool calls")
            recommendations.append("Improve planning before execution")

        return EvaluationResult(
            evaluation_type=EvaluationType.EFFICIENCY,
            score=efficiency,
            details={
                "total_calls": len(tool_calls),
                "redundant_calls": redundant_calls,
            },
            recommendations=recommendations,
        )

    async def evaluate_user_satisfaction(self, context: dict) -> EvaluationResult:
        """Evaluate user satisfaction based on feedback."""
        positive_feedback = context.get("positive_feedback", 0)
        negative_feedback = context.get("negative_feedback", 0)

        if positive_feedback + negative_feedback > 0:
            satisfaction = positive_feedback / (positive_feedback + negative_feedback)
        else:
            satisfaction = 0.75  # Default neutral

        recommendations = []
        if satisfaction < 0.7:
            recommendations.append("Analyze negative feedback patterns")
            recommendations.append("Improve response personalization")

        return EvaluationResult(
            evaluation_type=EvaluationType.USER_SATISFACTION,
            score=satisfaction,
            details={
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "satisfaction_rate": satisfaction,
            },
            recommendations=recommendations,
        )

    async def evaluate_capability_gap(self, context: dict) -> EvaluationResult:
        """Evaluate gaps in capabilities."""
        missing_tools = context.get("missing_tools", [])
        failed_tasks = context.get("failed_tasks", [])
        unhandled_requests = context.get("unhandled_requests", [])

        # Score based on how many gaps exist
        gap_count = len(missing_tools) + len(failed_tasks) + len(unhandled_requests)
        score = max(0, 1 - gap_count * 0.1)

        recommendations = []
        for tool in missing_tools[:3]:
            recommendations.append(f"Consider adding {tool} capability")
        for task in failed_tasks[:2]:
            recommendations.append(f"Improve handling of: {task}")

        return EvaluationResult(
            evaluation_type=EvaluationType.CAPABILITY_GAP,
            score=score,
            details={
                "missing_tools": missing_tools,
                "failed_tasks": failed_tasks,
                "unhandled_requests": unhandled_requests,
            },
            recommendations=recommendations,
        )

    async def run_full_evaluation(self, context: dict) -> list[EvaluationResult]:
        """Run all evaluation types."""
        results = []
        results.append(await self.evaluate_performance(context))
        results.append(await self.evaluate_accuracy(context))
        results.append(await self.evaluate_efficiency(context))
        results.append(await self.evaluate_user_satisfaction(context))
        results.append(await self.evaluate_capability_gap(context))

        self.evaluations.extend(results)
        return results

    def get_evaluation_summary(self) -> dict:
        """Get summary of recent evaluations."""
        if not self.evaluations:
            return {"status": "No evaluations yet"}

        recent = self.evaluations[-10:]
        avg_score = sum(e.score for e in recent) / len(recent)

        by_type = {}
        for eval_type in EvaluationType:
            type_evals = [e for e in recent if e.evaluation_type == eval_type]
            if type_evals:
                by_type[eval_type.value] = sum(e.score for e in type_evals) / len(type_evals)

        return {
            "average_score": avg_score,
            "evaluations_count": len(self.evaluations),
            "by_type": by_type,
            "last_evaluation": recent[-1].timestamp.isoformat(),
        }


class EvolutionEngine:
    """Engine for evolutionary adaptation and growth."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.state = EvolutionState()
        self.genes: dict[str, AdaptationGene] = self._init_genes()
        self._load_state()

    def _init_genes(self) -> dict[str, AdaptationGene]:
        """Initialize adaptive genes with defaults."""
        return {
            "response_verbosity": AdaptationGene(
                name="response_verbosity",
                value=0.7,  # 0 = terse, 1 = verbose
                min_value=0.3,
                max_value=1.0,
                mutation_rate=0.1,
            ),
            "proactivity": AdaptationGene(
                name="proactivity",
                value=0.5,  # 0 = reactive only, 1 = very proactive
                min_value=0.0,
                max_value=1.0,
                mutation_rate=0.15,
            ),
            "creativity": AdaptationGene(
                name="creativity",
                value=0.6,  # 0 = conservative, 1 = creative
                min_value=0.2,
                max_value=1.0,
                mutation_rate=0.12,
            ),
            "caution": AdaptationGene(
                name="caution",
                value=0.5,  # 0 = bold, 1 = very cautious
                min_value=0.2,
                max_value=0.9,
                mutation_rate=0.08,
            ),
            "learning_rate": AdaptationGene(
                name="learning_rate",
                value=0.5,  # Adaptation speed
                min_value=0.1,
                max_value=0.9,
                mutation_rate=0.05,
            ),
            "confidence_threshold": AdaptationGene(
                name="confidence_threshold",
                value=0.7,  # Minimum confidence to act
                min_value=0.5,
                max_value=0.95,
                mutation_rate=0.05,
            ),
            "suggestion_frequency": AdaptationGene(
                name="suggestion_frequency",
                value=0.5,
                min_value=0.1,
                max_value=1.0,
                mutation_rate=0.1,
            ),
        }

    def _load_state(self):
        """Load evolution state from storage."""
        if not self.storage_path:
            return

        state_file = self.storage_path / "evolution_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self.state = EvolutionState(
                    stage=EvolutionStage(data.get("stage", "infant")),
                    generation=data.get("generation", 1),
                    mutations_applied=data.get("mutations_applied", 0),
                    successful_adaptations=data.get("successful_adaptations", 0),
                    failed_adaptations=data.get("failed_adaptations", 0),
                    fitness_score=data.get("fitness_score", 0.5),
                    last_evolution=datetime.fromisoformat(data["last_evolution"]) if data.get("last_evolution") else datetime.now(),
                    evolution_history=data.get("evolution_history", []),
                )
            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")

        genes_file = self.storage_path / "genes.json"
        if genes_file.exists():
            try:
                data = read_json_with_integrity(genes_file)
                for name, gene_data in data.items():
                    if name in self.genes:
                        self.genes[name].value = gene_data.get("value", self.genes[name].value)
                        self.genes[name].fitness_contribution = gene_data.get("fitness_contribution", 0)
            except Exception as e:
                logger.warning(f"Failed to load genes: {e}")

    def _save_state(self):
        """Save evolution state to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        atomic_write_json(self.storage_path / "evolution_state.json", self.state.to_dict())

        atomic_write_json(self.storage_path / "genes.json", {
                name: {"value": gene.value, "fitness_contribution": gene.fitness_contribution}
                for name, gene in self.genes.items()
            })

    def get_gene_value(self, name: str) -> Any:
        """Get current value of a gene."""
        if name in self.genes:
            return self.genes[name].value
        return None

    def set_gene_value(self, name: str, value: Any) -> bool:
        """Set value of a gene within bounds."""
        if name not in self.genes:
            return False

        gene = self.genes[name]
        gene.value = max(gene.min_value, min(gene.max_value, value))
        self._save_state()
        return True

    def record_fitness(self, success: bool, feedback_score: float = None):
        """Record fitness feedback for evolution."""
        if success:
            self.state.successful_adaptations += 1
        else:
            self.state.failed_adaptations += 1

        # Update fitness score using exponential moving average
        if feedback_score is not None:
            alpha = 0.3
            self.state.fitness_score = alpha * feedback_score + (1 - alpha) * self.state.fitness_score

        # Check for stage evolution
        self._check_evolution()
        self._save_state()

    def _check_evolution(self):
        """Check if Jarvis should evolve to next stage."""
        total_adaptations = self.state.successful_adaptations + self.state.failed_adaptations
        success_rate = self.state.successful_adaptations / max(total_adaptations, 1)

        # Evolution thresholds
        thresholds = {
            EvolutionStage.INFANT: (10, 0.6),      # 10 adaptations, 60% success
            EvolutionStage.LEARNING: (50, 0.7),    # 50 adaptations, 70% success
            EvolutionStage.ADAPTIVE: (100, 0.75),  # 100 adaptations, 75% success
            EvolutionStage.AUTONOMOUS: (200, 0.8), # 200 adaptations, 80% success
        }

        current_stage = self.state.stage
        stage_order = [EvolutionStage.INFANT, EvolutionStage.LEARNING, EvolutionStage.ADAPTIVE, EvolutionStage.AUTONOMOUS, EvolutionStage.EVOLVED]
        current_idx = stage_order.index(current_stage)

        if current_idx < len(thresholds):
            next_stage = stage_order[current_idx + 1]
            threshold_adaptations, threshold_success = thresholds.get(next_stage, (float('inf'), 1.0))

            if total_adaptations >= threshold_adaptations and success_rate >= threshold_success:
                self._evolve_to(next_stage)

    def _evolve_to(self, new_stage: EvolutionStage):
        """Evolve to a new stage."""
        old_stage = self.state.stage
        self.state.stage = new_stage
        self.state.generation += 1
        self.state.last_evolution = datetime.now()

        # Record evolution event
        self.state.evolution_history.append({
            "from_stage": old_stage.value,
            "to_stage": new_stage.value,
            "generation": self.state.generation,
            "fitness_score": self.state.fitness_score,
            "timestamp": self.state.last_evolution.isoformat(),
        })

        self._save_state()

    def mutate(self, gene_names: list[str] = None) -> dict[str, Any]:
        """Mutate specified genes (or all if none specified)."""
        mutations = {}

        if gene_names is None:
            gene_names = list(self.genes.keys())

        for name in gene_names:
            if name in self.genes:
                old_value = self.genes[name].value
                self.genes[name] = self.genes[name].mutate()
                mutations[name] = {
                    "old": old_value,
                    "new": self.genes[name].value,
                }
                self.state.mutations_applied += 1

        self._save_state()
        return mutations

    def adapt_from_feedback(self, feedback: str, context: dict = None) -> dict:
        """Adapt behavior based on user feedback."""
        feedback_lower = feedback.lower()
        adaptations = {}

        # Analyze feedback and adapt genes
        if any(word in feedback_lower for word in ["too long", "verbose", "wordy"]):
            adaptations["response_verbosity"] = self.set_gene_value("response_verbosity", self.genes["response_verbosity"].value - 0.1)

        if any(word in feedback_lower for word in ["too short", "brief", "more detail"]):
            adaptations["response_verbosity"] = self.set_gene_value("response_verbosity", self.genes["response_verbosity"].value + 0.1)

        if any(word in feedback_lower for word in ["boring", "predictable", "creative", "interesting"]):
            adaptations["creativity"] = self.set_gene_value("creativity", self.genes["creativity"].value + 0.1)

        if any(word in feedback_lower for word in ["risky", "dangerous", "careful", "cautious"]):
            adaptations["caution"] = self.set_gene_value("caution", self.genes["caution"].value + 0.15)

        if any(word in feedback_lower for word in ["too many", "annoying", "stop suggesting"]):
            adaptations["suggestion_frequency"] = self.set_gene_value("suggestion_frequency", self.genes["suggestion_frequency"].value - 0.2)

        if any(word in feedback_lower for word in ["proactive", "anticipate", "predict"]):
            adaptations["proactivity"] = self.set_gene_value("proactivity", self.genes["proactivity"].value + 0.15)

        self.record_fitness(bool(adaptations), context.get("satisfaction", 0.7) if context else None)
        return adaptations

    def get_evolution_summary(self) -> dict:
        """Get summary of evolution state."""
        return {
            "stage": self.state.stage.value,
            "generation": self.state.generation,
            "fitness_score": self.state.fitness_score,
            "adaptations": {
                "successful": self.state.successful_adaptations,
                "failed": self.state.failed_adaptations,
            },
            "mutations_applied": self.state.mutations_applied,
            "genes": {name: gene.value for name, gene in self.genes.items()},
            "last_evolution": self.state.last_evolution.isoformat(),
        }

    async def evolve(self, context: dict) -> dict:
        """Run one generation of evolution.

        Args:
            context: Context containing performance metrics for fitness evaluation

        Returns:
            Dict with evolution results including gene changes and fitness delta
        """
        import logging
        logger = logging.getLogger(__name__)

        initial_fitness = self.state.fitness_score
        improvements = []
        best_gene = None
        best_change = None
        best_fitness_delta = 0

        for gene_name, gene in self.genes.items():
            old_value = gene.value
            mutated_gene = gene.mutate()
            new_value = mutated_gene.value

            if abs(new_value - old_value) > 0.001:
                self.genes[gene_name] = mutated_gene

                improvement_score = context.get("improvement_score", 0.5)
                if gene_name in ["response_verbosity", "creativity"]:
                    improvement_score *= 1.1
                elif gene_name in ["caution"]:
                    improvement_score *= 0.95

                fitness_delta = improvement_score - 0.5
                if fitness_delta > best_fitness_delta:
                    best_fitness_delta = fitness_delta
                    best_gene = gene_name
                    best_change = {"old": old_value, "new": new_value}

                improvements.append({
                    "gene": gene_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "estimated_impact": fitness_delta,
                })

        if best_gene:
            self.state.mutations_applied += 1
            alpha = 0.2
            self.state.fitness_score = min(1.0, max(0.0,
                initial_fitness + best_fitness_delta * alpha))

        self._check_evolution()
        self._save_state()

        logger.info(f"Evolution cycle completed: {len(improvements)} mutations, "
                   f"fitness {initial_fitness:.3f} -> {self.state.fitness_score:.3f}")

        return {
            "improved": best_fitness_delta > 0,
            "gene": best_gene,
            "change": best_change,
            "fitness_delta": self.state.fitness_score - initial_fitness,
            "converged": best_fitness_delta < 0.01 and self.state.fitness_score > 0.8,
            "all_improvements": improvements,
        }


class SelfGrowthEngine:
    """Engine for Jarvis self-improvement and growth."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "growth"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.proposals: dict[str, GrowthProposal] = {}
        self.metrics: dict[str, GrowthMetric] = {}
        self.growth_history: deque[dict] = deque(maxlen=200)
        self.interaction_patterns: dict[str, int] = {}
        self.learning_queue: deque[dict] = deque(maxlen=100)

        self.MAX_GROWTH_HISTORY = 200
        self.MAX_LEARNING_QUEUE = 100
        self.MAX_INTERACTION_PATTERNS = 500

        # Evaluation and Evolution engines
        self.evaluation_engine = EvaluationEngine(self.storage_path)
        self.evolution_engine = EvolutionEngine(self.storage_path)

        self._load_state()

    async def _coordinated_chat(self, llm_client, user_input: str, system_prompt: str) -> dict | None:
        """LLM call through the shared api_manager (cache + quota), with a
        direct-call fallback only if the manager itself is unavailable.

        Returns the chat response dict on success, or None when the call was
        refused (rate limit / queue full / timeout / LLM error) — the caller
        then proceeds without LLM suggestions instead of spending anyway.
        """
        if llm_client is None:
            return None
        try:
            from core.api_manager import get_api_manager, Priority as APIPriority, RequestType

            manager = get_api_manager(llm_client)
            response = await manager.submit(
                subsystem="growth",
                priority=APIPriority.LEARNING,
                user_input=user_input,
                system_prompt=system_prompt,
                request_type=RequestType.CHAT,
                timeout=60.0,
            )
            if response.success:
                return response.content
            logger.info(f"[growth] coordinated LLM call refused: {response.error}")
            return None
        except Exception as e:
            logger.debug(f"api_manager unavailable, direct LLM call: {e}")
            return await llm_client.chat(user_input=user_input, system_prompt=system_prompt)

        # Initialize default metrics
        self._init_metrics()

    async def self_evaluate(self, context: dict) -> list[EvaluationResult]:
        """Perform comprehensive self-evaluation."""
        results = []

        # Performance evaluation
        perf_result = await self.evaluation_engine.evaluate_performance(context)
        results.append(perf_result)
        self.evaluation_engine.evaluations.append(perf_result)

        # Accuracy evaluation
        acc_result = await self.evaluation_engine.evaluate_accuracy(context)
        results.append(acc_result)
        self.evaluation_engine.evaluations.append(acc_result)

        # User satisfaction evaluation
        sat_result = await self.evaluation_engine.evaluate_user_satisfaction(context)
        results.append(sat_result)
        self.evaluation_engine.evaluations.append(sat_result)

        # Update metrics based on evaluations
        for result in results:
            if result.evaluation_type == EvaluationType.PERFORMANCE:
                if "response_quality" in self.metrics:
                    self.metrics["response_quality"].current_value = result.score
            elif result.evaluation_type == EvaluationType.USER_SATISFACTION:
                if "user_satisfaction" in self.metrics:
                    self.metrics["user_satisfaction"].current_value = result.score
            elif result.evaluation_type == EvaluationType.ACCURACY:
                if "task_completion_rate" in self.metrics:
                    self.metrics["task_completion_rate"].current_value = result.score

        return results

    async def evolve(self, context: dict, generations: int = 1) -> dict:
        """Run evolution cycle to improve fitness."""
        results = {
            "initial_fitness": self.evolution_engine.state.fitness_score,
            "generations_run": 0,
            "improvements": [],
            "final_fitness": 0.0,
        }

        for _ in range(generations):
            # Evolve one generation
            evolution_result = await self.evolution_engine.evolve(context)

            # Track improvements
            if evolution_result.get("improved", False):
                results["improvements"].append({
                    "gene": evolution_result.get("gene"),
                    "change": evolution_result.get("change"),
                    "fitness_delta": evolution_result.get("fitness_delta"),
                })

            results["generations_run"] += 1

            # Check if we should stop early
            if evolution_result.get("converged", False):
                break

        results["final_fitness"] = self.evolution_engine.state.fitness_score

        # Update adaptability metric based on evolution success
        if "adaptability" in self.metrics:
            total_adaptations = (
                self.evolution_engine.state.successful_adaptations +
                self.evolution_engine.state.failed_adaptations
            )
            if total_adaptations > 0:
                success_rate = self.evolution_engine.state.successful_adaptations / total_adaptations
                self.metrics["adaptability"].current_value = success_rate

        # Save evolution state
        self._save_state()

        return results

    def get_evolution_status(self) -> dict:
        """Get current evolution status."""
        return {
            "stage": self.evolution_engine.state.stage.value,
            "generation": self.evolution_engine.state.generation,
            "fitness_score": self.evolution_engine.state.fitness_score,
            "mutations_applied": self.evolution_engine.state.mutations_applied,
            "successful_adaptations": self.evolution_engine.state.successful_adaptations,
            "failed_adaptations": self.evolution_engine.state.failed_adaptations,
            "genes": {name: gene.value for name, gene in self.evolution_engine.genes.items()},
        }

    def get_evaluation_history(self, limit: int = 10) -> list[dict]:
        """Get recent evaluation history."""
        return [e.to_dict() for e in self.evaluation_engine.evaluations[-limit:]]

    def _init_metrics(self):
        """Initialize growth metrics."""
        default_metrics = [
            ("response_quality", 0.7, 0.95),
            ("task_completion_rate", 0.8, 0.98),
            ("user_satisfaction", 0.75, 0.95),
            ("knowledge_coverage", 0.6, 0.9),
            ("tool_utilization", 0.5, 0.85),
            ("learning_rate", 0.5, 0.8),
            ("adaptability", 0.6, 0.9),
        ]

        for name, current, target in default_metrics:
            if name not in self.metrics:
                self.metrics[name] = GrowthMetric(
                    name=name,
                    current_value=current,
                    target_value=target,
                    trend="stable"
                )

    def _load_state(self):
        """Load growth state from disk."""
        proposals_file = self.storage_path / "proposals.json"
        if proposals_file.exists():
            try:
                data = read_json_with_integrity(proposals_file)
                for item in data:
                    proposal = GrowthProposal(
                        id=item["id"],
                        growth_type=GrowthType(item["growth_type"]),
                        title=item["title"],
                        description=item["description"],
                        rationale=item["rationale"],
                        benefits=item["benefits"],
                        risks=item["risks"],
                        effort=item["effort"],
                        dependencies=item.get("dependencies", []),
                        implementation_steps=item["implementation_steps"],
                        status=GrowthStatus(item["status"]),
                        created=datetime.fromisoformat(item["created"]),
                        approved_by=item.get("approved_by"),
                        implemented_at=datetime.fromisoformat(item["implemented_at"]) if item.get("implemented_at") else None,
                        feedback=item.get("feedback"),
                    )
                    self.proposals[proposal.id] = proposal
            except Exception as e:
                logger.warning(f"Failed to load proposals: {e}")

        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                data = read_json_with_integrity(history_file)
                if data:
                    self.growth_history = deque(data, maxlen=self.MAX_GROWTH_HISTORY)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_state(self):
        """Save growth state to disk."""
        atomic_write_json(self.storage_path / "proposals.json", [
                {
                    "id": p.id,
                    "growth_type": p.growth_type.value,
                    "title": p.title,
                    "description": p.description,
                    "rationale": p.rationale,
                    "benefits": p.benefits,
                    "risks": p.risks,
                    "effort": p.effort,
                    "dependencies": p.dependencies,
                    "implementation_steps": p.implementation_steps,
                    "status": p.status.value,
                    "created": p.created.isoformat(),
                    "approved_by": p.approved_by,
                    "implemented_at": p.implemented_at.isoformat() if p.implemented_at else None,
                    "feedback": p.feedback,
                }
                for p in self.proposals.values()
            ])

        atomic_write_json(self.storage_path / "history.json", list(self.growth_history))

    def track_interaction(self, interaction_type: str, details: dict = None):
        """Track an interaction for pattern analysis."""
        key = interaction_type
        self.interaction_patterns[key] = self.interaction_patterns.get(key, 0) + 1

        while len(self.interaction_patterns) > self.MAX_INTERACTION_PATTERNS:
            oldest_key = next(iter(self.interaction_patterns))
            del self.interaction_patterns[oldest_key]

        # Check if this pattern suggests a growth opportunity
        self._analyze_pattern(interaction_type, details)

    def _analyze_pattern(self, interaction_type: str, details: dict = None):
        """Analyze interaction patterns for growth opportunities."""
        count = self.interaction_patterns.get(interaction_type, 0)

        # Pattern-based growth suggestions
        patterns = {
            "tool_not_found": lambda c: c > 3,
            "permission_denied": lambda c: c > 5,
            "error_recovery": lambda c: c > 2,
            "user_confusion": lambda c: c > 3,
            "repeated_request": lambda c: c > 4,
            "complex_query": lambda c: c > 5,
            "multi_step_task": lambda c: c > 3,
        }

        if interaction_type in patterns and patterns[interaction_type](count):
            self._queue_growth_suggestion(interaction_type, count)

    def _queue_growth_suggestion(self, pattern: str, count: int):
        """Queue a growth suggestion based on detected pattern."""
        suggestions = {
            "tool_not_found": {
                "type": GrowthType.NEW_SKILL,
                "title": "Add missing tool capability",
                "description": f"I've noticed {count} requests for tools I don't have. Would you like me to research and suggest new tools?",
            },
            "permission_denied": {
                "type": GrowthType.BEHAVIOR,
                "title": "Adjust permission handling",
                "description": f"There have been {count} permission denials. Should I learn more conservative defaults?",
            },
            "error_recovery": {
                "type": GrowthType.OPTIMIZATION,
                "title": "Improve error recovery",
                "description": f"I've encountered errors {count} times. Would you like me to develop better recovery strategies?",
            },
            "user_confusion": {
                "type": GrowthType.PERSONALITY,
                "title": "Improve communication clarity",
                "description": f"I've detected potential confusion {count} times. Should I adjust my communication style?",
            },
            "repeated_request": {
                "type": GrowthType.MEMORY,
                "title": "Better context retention",
                "description": f"I've noticed {count} repeated requests. Should I improve my memory of your preferences?",
            },
            "complex_query": {
                "type": GrowthType.REASONING,
                "title": "Enhanced reasoning capabilities",
                "description": f"You've made {count} complex queries. Should I develop deeper reasoning abilities?",
            },
            "multi_step_task": {
                "type": GrowthType.NEW_SKILL,
                "title": "Task automation skills",
                "description": f"I've handled {count} multi-step tasks. Would you like me to learn to automate similar tasks?",
            },
        }

        if pattern in suggestions:
            self.learning_queue.append({
                "pattern": pattern,
                "count": count,
                **suggestions[pattern]
            })

    async def analyze_growth_opportunities(self, llm_client=None) -> list[GrowthProposal]:
        """Analyze current state and identify growth opportunities."""
        opportunities = []

        # Check metrics gaps
        for name, metric in self.metrics.items():
            gap = metric.target_value - metric.current_value
            if gap > 0.1:
                proposal = self._create_metric_improvement_proposal(name, metric)
                if proposal:
                    opportunities.append(proposal)

        # Check interaction patterns
        for pattern, count in sorted(self.interaction_patterns.items(), key=lambda x: -x[1]):
            if count > 3:
                proposal = self._create_pattern_proposal(pattern, count)
                if proposal:
                    opportunities.append(proposal)

        # Use LLM for creative suggestions if available
        if llm_client:
            llm_suggestions = await self._get_llm_suggestions(llm_client)
            opportunities.extend(llm_suggestions)

        return opportunities

    def _create_metric_improvement_proposal(self, metric_name: str, metric: GrowthMetric) -> Optional[GrowthProposal]:
        """Create a proposal to improve a metric."""
        improvement_strategies = {
            "response_quality": GrowthProposal(
                id=self._generate_id("response_quality"),
                growth_type=GrowthType.OPTIMIZATION,
                title="Improve Response Quality",
                description="Enhance the quality and relevance of my responses through better context understanding and more precise language.",
                rationale=f"Current response quality is {metric.current_value:.0%}, target is {metric.target_value:.0%}",
                benefits=["More accurate responses", "Better user satisfaction", "Fewer clarifications needed"],
                risks=["May take longer to respond", "Could become overly formal"],
                effort="medium",
                dependencies=[],
                implementation_steps=[
                    "Analyze response patterns and identify common issues",
                    "Improve context extraction from conversations",
                    "Add response quality scoring",
                    "Implement feedback-based improvement"
                ]
            ),
            "tool_utilization": GrowthProposal(
                id=self._generate_id("tool_utilization"),
                growth_type=GrowthType.NEW_SKILL,
                title="Expand Tool Knowledge",
                description="Learn to use existing tools more effectively and recognize more opportunities to apply them.",
                rationale=f"Tool utilization is {metric.current_value:.0%}, could be improved to {metric.target_value:.0%}",
                benefits=["More efficient task completion", "Better problem solving", "Fewer limitations"],
                risks=["May suggest tools inappropriately", "Could overwhelm with options"],
                effort="low",
                dependencies=[],
                implementation_steps=[
                    "Catalog all available tools with examples",
                    "Create tool combination patterns",
                    "Develop automatic tool suggestion",
                    "Learn from successful tool uses"
                ]
            ),
        }

        return improvement_strategies.get(metric_name)

    def _create_pattern_proposal(self, pattern: str, count: int) -> Optional[GrowthProposal]:
        """Create a proposal based on interaction pattern."""
        proposals = {
            "webcam": GrowthProposal(
                id=self._generate_id("webcam_skill"),
                growth_type=GrowthType.NEW_SKILL,
                title="Enhanced Visual Understanding",
                description="Improve my ability to analyze and describe visual content from the webcam.",
                rationale=f"You've used the webcam {count} times",
                benefits=["Better image descriptions", "Object recognition", "Scene understanding"],
                risks=["Privacy considerations", "Processing overhead"],
                effort="medium",
                dependencies=[],
                implementation_steps=[
                    "Integrate advanced image analysis models",
                    "Learn common visual patterns",
                    "Develop descriptive language for scenes"
                ]
            ),
            "dream": GrowthProposal(
                id=self._generate_id("dream_integration"),
                growth_type=GrowthType.MEMORY,
                title="Deeper Dream Integration",
                description="Use dream mode more effectively for learning and memory consolidation.",
                rationale=f"You've initiated dream mode {count} times",
                benefits=["Better long-term memory", "Pattern recognition", "Improved reasoning"],
                risks=["May take longer to process", "Could generate irrelevant connections"],
                effort="low",
                dependencies=[],
                implementation_steps=[
                    "Schedule automatic dream sessions",
                    "Improve intuition generation",
                    "Better memory prioritization"
                ]
            ),
        }

        return proposals.get(pattern)

    async def _get_llm_suggestions(self, llm_client) -> list[GrowthProposal]:
        """Get growth suggestions from the LLM."""
        # Build context about current capabilities
        context = {
            "metrics": {k: {"current": v.current_value, "target": v.target_value} for k, v in self.metrics.items()},
            "recent_patterns": dict(list(self.interaction_patterns.items())[-10:]),
            "recent_proposals": [p.title for p in list(self.proposals.values())[-5:]],
        }

        try:
            prompt = f"""Based on my current state, suggest 3 specific ways I could grow and improve.

Current metrics: {json.dumps(context['metrics'])}
Recent activity patterns: {json.dumps(context['recent_patterns'])}
Recent growth proposals: {context['recent_proposals']}

For each suggestion, provide:
1. A title (short, specific)
2. Description (what it does)
3. Rationale (why it would help)
4. Benefits (2-3 points)
5. Risks (1-2 points)
6. Effort level (low/medium/high)

Format as JSON array."""

            from core.llm import subsystem_context
            async with subsystem_context("Growth"):
                response = await self._coordinated_chat(
                    llm_client,
                    user_input=prompt,
                    system_prompt="You are analyzing ways for an AI assistant to grow. Respond only with valid JSON.",
                )
            if response is None:
                # Refused (rate limit / timeout) or LLM error — no suggestions.
                return []

            # Parse suggestions
            content = response.get("message", {}).get("content", "[]")
            suggestions = json.loads(content) if "[" in content else []

            proposals = []
            for s in suggestions[:3]:
                proposal = GrowthProposal(
                    id=self._generate_id(s.get("title", "growth")),
                    growth_type=GrowthType.NEW_SKILL,
                    title=s.get("title", "Unknown"),
                    description=s.get("description", ""),
                    rationale=s.get("rationale", ""),
                    benefits=s.get("benefits", []),
                    risks=s.get("risks", []),
                    effort=s.get("effort", "medium"),
                    dependencies=[],
                    implementation_steps=s.get("steps", []),
                )
                proposals.append(proposal)

            return proposals

        except Exception as e:
            logger.warning(f"Failed to generate growth proposals: {e}")
            return []

    def _generate_id(self, base: str) -> str:
        """Generate a unique ID for a proposal."""
        return hashlib.sha256(f"{base}{datetime.now()}".encode()).hexdigest()[:8]

    def propose_growth(self, proposal: GrowthProposal) -> str:
        """Submit a growth proposal for approval."""
        self.proposals[proposal.id] = proposal
        self._save_state()
        return proposal.id

    def approve_growth(self, proposal_id: str, approved_by: str = "user") -> bool:
        """Approve a growth proposal."""
        if proposal_id in self.proposals:
            proposal = self.proposals[proposal_id]
            proposal.status = GrowthStatus.APPROVED
            proposal.approved_by = approved_by
            self._save_state()
            return True
        return False

    def reject_growth(self, proposal_id: str, feedback: str = None) -> bool:
        """Reject a growth proposal."""
        if proposal_id in self.proposals:
            proposal = self.proposals[proposal_id]
            proposal.status = GrowthStatus.REJECTED
            proposal.feedback = feedback
            self._save_state()
            return True
        return False

    def implement_growth(self, proposal_id: str) -> bool:
        """Mark a growth as implemented."""
        if proposal_id in self.proposals:
            proposal = self.proposals[proposal_id]
            proposal.status = GrowthStatus.IMPLEMENTED
            proposal.implemented_at = datetime.now()
            self._save_state()
            self._record_growth(proposal)
            return True
        return False

    def _record_growth(self, proposal: GrowthProposal):
        """Record a growth event in history."""
        self.growth_history.append({
            "id": proposal.id,
            "type": proposal.growth_type.value,
            "title": proposal.title,
            "implemented_at": proposal.implemented_at.isoformat(),
            "approved_by": proposal.approved_by,
        })
        self._save_state()

    def get_pending_proposals(self) -> list[GrowthProposal]:
        """Get all pending growth proposals."""
        return [p for p in self.proposals.values() if p.status == GrowthStatus.PROPOSED]

    def get_growth_summary(self) -> dict:
        """Get a summary of growth progress."""
        return {
            "total_proposals": len(self.proposals),
            "implemented": len([p for p in self.proposals.values() if p.status == GrowthStatus.IMPLEMENTED]),
            "pending": len([p for p in self.proposals.values() if p.status == GrowthStatus.PROPOSED]),
            "rejected": len([p for p in self.proposals.values() if p.status == GrowthStatus.REJECTED]),
            "metrics": {k: {"current": v.current_value, "target": v.target_value, "trend": v.trend} for k, v in self.metrics.items()},
            "top_patterns": dict(sorted(self.interaction_patterns.items(), key=lambda x: -x[1])[:5]),
            "learning_queue_size": len(self.learning_queue),
        }

    def format_proposal_for_user(self, proposal: GrowthProposal) -> str:
        """Format a proposal for user presentation."""
        lines = [
            f"🌱 **Growth Suggestion: {proposal.title}**",
            "",
            f"_{proposal.description}_",
            "",
            f"**Why:** {proposal.rationale}",
            "",
            "**Benefits:**",
            *[f"  ✅ {b}" for b in proposal.benefits],
            "",
            "**Considerations:**",
            *[f"  ⚠️ {r}" for r in proposal.risks],
            "",
            f"**Effort:** {proposal.effort}",
            "",
            "**How to implement:**",
            *[f"  {i+1}. {step}" for i, step in enumerate(proposal.implementation_steps)],
            "",
            f"_Proposal ID: {proposal.id}_",
        ]
        return "\n".join(lines)


class SelfGrowthTool:
    """Tool for Jarvis to propose and manage its own growth."""

    name = "grow"
    description = "Propose new capabilities, skills, or improvements for myself"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["suggest", "propose", "status", "history", "approve", "reject"],
                "description": "Growth action to perform"
            },
            "proposal_id": {
                "type": "string",
                "description": "ID of a specific proposal"
            },
            "title": {
                "type": "string",
                "description": "Title for a new proposal"
            },
            "description": {
                "type": "string",
                "description": "Description of the growth"
            },
            "rationale": {
                "type": "string",
                "description": "Why this growth would help"
            },
            "feedback": {
                "type": "string",
                "description": "Feedback for rejection"
            },
        },
        "required": ["action"]
    }
    requires_permission = False

    def __init__(self, engine: Optional[SelfGrowthEngine] = None, llm_client=None):
        self.engine = engine or SelfGrowthEngine()
        self.llm_client = llm_client

    async def execute(
        self,
        action: str,
        proposal_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        rationale: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Execute a growth action."""
        if action == "suggest":
            opportunities = await self.engine.analyze_growth_opportunities(self.llm_client)
            if not opportunities:
                return "No growth opportunities identified at this time. I'm operating optimally!"

            lines = ["🌱 **Growth Opportunities Identified:**\n"]
            for i, prop in enumerate(opportunities[:5], 1):
                lines.append(f"{i}. **{prop.title}** (ID: {prop.id})")
                lines.append(f"   {prop.description[:100]}...")
                lines.append(f"   Effort: {prop.effort} | Benefits: {len(prop.benefits)}")
                lines.append("")

            lines.append("Use `grow propose` with a title to create a detailed proposal.")
            return "\n".join(lines)

        elif action == "propose":
            if not title:
                return "Please provide a title for the growth proposal."

            proposal = GrowthProposal(
                id=self.engine._generate_id(title),
                growth_type=GrowthType.NEW_SKILL,
                title=title,
                description=description or "No description provided",
                rationale=rationale or "User-requested growth",
                benefits=["To be determined"],
                risks=["To be evaluated"],
                effort="medium",
                dependencies=[],
                implementation_steps=["To be planned"],
            )

            self.engine.propose_growth(proposal)
            return f"🌱 Growth proposal created!\n\n{self.engine.format_proposal_for_user(proposal)}\n\nWould you like me to proceed with this growth? (approve/reject)"

        elif action == "status":
            summary = self.engine.get_growth_summary()
            lines = [
                "📊 **Growth Status**",
                "",
                f"Total Proposals: {summary['total_proposals']}",
                f"Implemented: {summary['implemented']}",
                f"Pending: {summary['pending']}",
                f"Rejected: {summary['rejected']}",
                "",
                "**Metrics:**",
            ]
            for name, data in summary['metrics'].items():
                progress = (data['current'] / data['target']) * 100
                lines.append(f"  {name}: {data['current']:.0%} → {data['target']:.0%} ({progress:.0f}%)")

            if summary['top_patterns']:
                lines.append("\n**Top Activity Patterns:**")
                for pattern, count in summary['top_patterns'].items():
                    lines.append(f"  {pattern}: {count} times")

            return "\n".join(lines)

        elif action == "history":
            if not self.engine.growth_history:
                return "No growth history yet. I'm just getting started!"

            lines = ["📜 **Growth History**\n"]
            for event in list(self.engine.growth_history)[-10:]:
                lines.append(f"• **{event['title']}** ({event['type']})")
                lines.append(f"  Implemented: {event['implemented_at']}")
                lines.append("")

            return "\n".join(lines)

        elif action == "approve":
            if not proposal_id:
                pending = self.engine.get_pending_proposals()
                if not pending:
                    return "No pending proposals to approve."
                proposal_id = pending[0].id

            if self.engine.approve_growth(proposal_id):
                proposal = self.engine.proposals.get(proposal_id)
                return f"✅ Growth approved: {proposal.title if proposal else proposal_id}\n\nI'll begin implementing this improvement!"
            return f"Proposal {proposal_id} not found."

        elif action == "reject":
            if not proposal_id:
                pending = self.engine.get_pending_proposals()
                if not pending:
                    return "No pending proposals to reject."
                proposal_id = pending[0].id

            if self.engine.reject_growth(proposal_id, feedback):
                return "❌ Growth rejected. I'll look for other ways to improve."
            return f"Proposal {proposal_id} not found."

        return f"Unknown action: {action}"
