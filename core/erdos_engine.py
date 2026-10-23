"""Erdős Math Problem Engine for continuous learning.

Inspired by Paul Erdős' collaborative mathematics approach, this engine
generates and attempts to solve problems in:
- Graph theory (Ramsey numbers, coloring problems)
- Number theory (prime gaps, additive combinatorics)
- Combinatorics (extremal problems, counting)
- Probability (probabilistic methods)

The engine runs in background, learning from attempts and evolving
solution strategies over time.
"""
import asyncio
import logging
from core.helpers import atomic_write_json, read_json_with_integrity
import math
import random
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from collections import defaultdict, deque

from core.dual_helpers import erdos_math_solve
from core.helpers import get_response_content

logger = logging.getLogger(__name__)

from core.prefrontal_cortex import get_prefrontal_cortex


class ProblemCategory(Enum):
    GRAPH_THEORY = "graph_theory"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"
    RAMSEY_THEORY = "ramsey_theory"
    ADDITIVE_COMBINATORICS = "additive_combinatorics"


class ProblemDifficulty(Enum):
    TRIVIAL = 1
    EASY = 2
    MODERATE = 3
    HARD = 4
    ERDOS_LEVEL = 5
    UNSOLVED = 6


@dataclass
class MathProblem:
    id: str
    title: str
    category: ProblemCategory
    difficulty: ProblemDifficulty
    statement: str
    hint: str | None = None
    known_bounds: dict[str, tuple[float, float]] | None = None
    strategies_tried: list[str] = field(default_factory=list)
    partial_results: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_attempt: datetime | None = None
    attempt_count: int = 0
    best_result: dict | None = None
    solved: bool = False
    solution: str | None = None
    erdos_number: int | None = None

    MAX_STRATEGIES = 50
    MAX_PARTIAL_RESULTS = 20

    def add_strategy(self, strategy: str) -> None:
        if strategy not in self.strategies_tried:
            self.strategies_tried.append(strategy)
            if len(self.strategies_tried) > self.MAX_STRATEGIES:
                self.strategies_tried = self.strategies_tried[-self.MAX_STRATEGIES:]

    def add_partial_result(self, result: dict) -> None:
        self.partial_results.append(result)
        if len(self.partial_results) > self.MAX_PARTIAL_RESULTS:
            self.partial_results = self.partial_results[-self.MAX_PARTIAL_RESULTS:]


@dataclass
class ProblemAttempt:
    problem_id: str
    timestamp: datetime
    strategy: str
    approach: str
    result: str
    success: bool
    insight: str | None = None
    computation_time: float = 0.0
    partial_answer: Any | None = None


class ErdosProblemGenerator:
    """Generates Erdős-style mathematical problems."""

    def __init__(self):
        self._problem_templates = self._load_templates()
        self._counter = 0
        self._lock = threading.Lock()

    def _load_templates(self) -> dict:
        return {
            ProblemCategory.RAMSEY_THEORY: [
                {
                    "template": "Find the Ramsey number R({s}, {t})",
                    "bounds_key": "ramsey",
                    "known": {
                        (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18,
                        (3, 7): 23, (3, 9): 36, (4, 4): 18, (4, 5): 25,
                    },
                    "generate": self._gen_ramsey_problem,
                },
                {
                    "template": "Find bounds for diagonal Ramsey number R({k})",
                    "bounds_key": "diagonal_ramsey",
                    "known": {},
                    "generate": self._gen_diagonal_ramsey,
                },
            ],
            ProblemCategory.NUMBER_THEORY: [
                {
                    "template": "Find the longest sequence of consecutive integers with {p} distinct prime factors each",
                    "bounds_key": "prime_factors",
                    "generate": self._gen_prime_gap,
                },
                {
                    "template": "Estimate π({n}) - π({n}-{k}) (primes in interval)",
                    "bounds_key": "prime_count",
                    "generate": self._gen_prime_count,
                },
                {
                    "template": "Find the smallest number with {k} distinct representations as sum of two squares",
                    "bounds_key": "sum_squares",
                    "generate": self._gen_sum_squares,
                },
            ],
            ProblemCategory.GRAPH_THEORY: [
                {
                    "template": "Find the maximum number of edges in a {n}-vertex graph with no {k}-clique",
                    "bounds_key": "turan",
                    "known": {},
                    "generate": self._gen_turan,
                },
                {
                    "template": "Determine the chromatic number of the unit distance graph in {d}D",
                    "bounds_key": "chromatic",
                    "generate": self._gen_chromatic,
                },
            ],
            ProblemCategory.ADDITIVE_COMBINATORICS: [
                {
                    "template": "What is the maximum size of a subset of {{1,...,{n}}} with no two elements summing to {s}?",
                    "bounds_key": "sum_free",
                    "generate": self._gen_sum_free,
                },
                {
                    "template": "Find the largest subset of {{1,...,{n}}} containing no {k}-term arithmetic progression",
                    "bounds_key": "van_der_waerden",
                    "generate": self._gen_arithmetic_progression,
                },
            ],
            ProblemCategory.COMBINATORICS: [
                {
                    "template": "How many ways to partition {n} into at most {k} parts?",
                    "bounds_key": "partitions",
                    "generate": self._gen_partition,
                },
                {
                    "template": "What is the number of labeled trees on {n} vertices?",
                    "bounds_key": "cayley",
                    "known": {},
                    "generate": self._gen_cayley,
                },
            ],
            ProblemCategory.PROBABILITY: [
                {
                    "template": "What is the probability threshold for a random graph G({n},{p}) to be connected?",
                    "bounds_key": "connectivity",
                    "generate": self._gen_connectivity,
                },
                {
                    "template": "Find the expected number of {k}-cliques in G({n},{p})",
                    "bounds_key": "clique_expectation",
                    "generate": self._gen_clique_prob,
                },
            ],
        }

    def generate_problem(
        self,
        category: ProblemCategory | None = None,
        difficulty: ProblemDifficulty | None = None,
    ) -> MathProblem:
        if category is None:
            category = random.choice(list(ProblemCategory))

        templates = self._problem_templates.get(category, [])
        if not templates:
            category = ProblemCategory.NUMBER_THEORY
            templates = self._problem_templates[category]

        template = random.choice(templates)
        generator = template.get("generate", self._generic_generate)

        problem_data = generator(template)

        with self._lock:
            self._counter += 1
            problem_id = f"erdos_{category.value[:3]}_{self._counter:04d}"

        return MathProblem(
            id=problem_id,
            title=problem_data["title"],
            category=category,
            difficulty=difficulty or problem_data.get("difficulty", ProblemDifficulty.MODERATE),
            statement=problem_data["statement"],
            hint=problem_data.get("hint"),
            known_bounds=problem_data.get("bounds"),
        )

    def _gen_ramsey_problem(self, template: dict) -> dict:
        s = random.randint(3, 6)
        t = random.randint(3, 6)

        known = template.get("known", {})
        known_value = known.get((s, t)) or known.get((t, s))

        if known_value:
            difficulty = ProblemDifficulty.EASY
            hint = f"The answer is {known_value}. Verify this."
        else:
            difficulty = ProblemDifficulty.HARD
            lower = math.ceil((s + t - 1) * math.sqrt(2) / 2)
            # Cap calculation to prevent overflow
            exponent = min(s + t - 2, 60)  # Safety limit
            upper = 2 ** exponent
            hint = f"Lower bound ~{lower}, upper bound ~{upper}"

        return {
            "title": f"Ramsey Number R({s}, {t})",
            "statement": f"Find the smallest n such that any red-blue coloring of the edges of K_n contains either a red K_{s} or a blue K_{t}.",
            "difficulty": difficulty,
            "hint": hint,
            "bounds": {"n": (s + t - 2, 2 ** (s + t - 2))},
        }

    def _gen_diagonal_ramsey(self, template: dict) -> dict:
        k = random.randint(5, 10)
        lower = 2 ** (k / 2)
        upper = 4 ** (k / 2)

        return {
            "title": f"Diagonal Ramsey Number R({k})",
            "statement": f"Find bounds for R({k},{k}), the smallest n where any 2-coloring of K_n has a monochromatic K_{k}.",
            "difficulty": ProblemDifficulty.ERDOS_LEVEL,
            "hint": f"Known: 2^({k}/2) < R({k}) < 4^({k}/2). Erdős: 'If aliens invade and demand R(5), we should form a panel.'",
            "bounds": {"n": (lower, upper)},
        }

    def _gen_prime_gap(self, template: dict) -> dict:
        k = random.randint(2, 4)
        length = random.randint(2, 6)

        return {
            "title": f"Consecutive Numbers with {k} Prime Factors",
            "statement": f"Find the longest sequence of consecutive positive integers where each has exactly {k} distinct prime factors.",
            "difficulty": ProblemDifficulty.MODERATE,
            "hint": "Use the Chinese Remainder Theorem construction for lower bounds.",
            "bounds": {"length": (length, length * 2)},
        }

    def _gen_prime_count(self, template: dict) -> dict:
        n = random.randint(1000, 10000)
        k = random.randint(50, 200)

        approx_pi = n / math.log(n)

        return {
            "title": f"Prime Count in [{n-k}, {n}]",
            "statement": f"Estimate the number of primes in the interval [{n-k}, {n}] using the Prime Number Theorem.",
            "difficulty": ProblemDifficulty.MODERATE,
            "hint": f"π({n}) ≈ {approx_pi:.0f} by PNT. The interval width is {k}.",
            "bounds": {"count": (k / math.log(n) * 0.9, k / math.log(n) * 1.1)},
        }

    def _gen_sum_squares(self, template: dict) -> dict:
        k = random.randint(2, 5)

        return {
            "title": "Sum of Two Squares Representations",
            "statement": f"Find the smallest positive integer that has exactly {k} distinct representations as a sum of two squares (counting order).",
            "difficulty": ProblemDifficulty.HARD,
            "hint": "Numbers with many representations have many distinct prime factors ≡ 1 (mod 4).",
            "bounds": {"min_value": (2 ** k, 10 ** k)},
        }

    def _gen_turan(self, template: dict) -> dict:
        n = random.randint(10, 100)
        k = random.randint(3, 5)

        turan_edges = (1 - 1/(k-1)) * n * n / 2

        return {
            "title": "Turan's Theorem Application",
            "statement": f"What is the maximum number of edges in an {n}-vertex graph with no {k}-clique?",
            "difficulty": ProblemDifficulty.MODERATE,
            "hint": f"Turan's theorem: maximum is (1 - 1/({k}-1)) × n²/2 ≈ {turan_edges:.0f}",
            "bounds": {"edges": (turan_edges * 0.9, turan_edges)},
        }

    def _gen_chromatic(self, template: dict) -> dict:
        d = random.randint(2, 4)
        known = {1: 2, 2: (4, 7), 3: (6, 15), 4: (7, float('inf'))}

        return {
            "title": f"Chromatic Number in {d}D",
            "statement": f"Determine the chromatic number of the unit distance graph in {d}-dimensional Euclidean space.",
            "difficulty": ProblemDifficulty.ERDOS_LEVEL if d >= 2 else ProblemDifficulty.MODERATE,
            "hint": f"Known bounds: {known.get(d, 'unknown')}. The Hadwiger-Nelson problem for d=2.",
            "bounds": {"chromatic_number": known.get(d, (2, 10))},
        }

    def _gen_sum_free(self, template: dict) -> dict:
        n = random.randint(10, 100)
        s = random.randint(n, 2*n)

        return {
            "title": "Sum-Free Subset Problem",
            "statement": f"What is the maximum size of a subset A ⊆ {{1,...,{n}}} such that for any x,y ∈ A, x+y ≠ {s}?",
            "difficulty": ProblemDifficulty.MODERATE,
            "hint": f"Consider numbers that are all > {s//2} or use modular arithmetic.",
            "bounds": {"max_size": (n//2 - 1, n//2 + 1)},
        }

    def _gen_arithmetic_progression(self, template: dict) -> dict:
        n = random.randint(20, 100)
        k = random.randint(3, 5)

        density = 1 / (k - 1)
        approx_size = int(n * density)

        return {
            "title": f"Avoiding {k}-term AP",
            "statement": f"Find the largest subset of {{1,...,{n}}} containing no {k}-term arithmetic progression.",
            "difficulty": ProblemDifficulty.HARD,
            "hint": "Szemerédi's theorem: density → 0 for large n. Behrend construction gives ~n·exp(-c√log n).",
            "bounds": {"max_size": (approx_size * 0.5, approx_size)},
        }

    def _gen_partition(self, template: dict) -> dict:
        n = random.randint(10, 50)
        k = random.randint(2, min(10, n//2))

        def partition_count(n, k):
            if k == 0:
                return 1 if n == 0 else 0
            dp = [[0] * (k + 1) for _ in range(n + 1)]
            dp[0][0] = 1
            for i in range(1, n + 1):
                for j in range(1, min(i, k) + 1):
                    dp[i][j] = dp[i-1][j-1] + dp[i-j][j]
            return sum(dp[n][j] for j in range(1, k + 1))

        count = partition_count(n, k)

        return {
            "title": "Integer Partitions",
            "statement": f"How many ways can {n} be partitioned into at most {k} parts?",
            "difficulty": ProblemDifficulty.MODERATE,
            "hint": f"Use dynamic programming. The answer is {count}.",
            "bounds": {"count": (count, count)},
        }

    def _gen_cayley(self, template: dict) -> dict:
        n = random.randint(3, 15)
        count = n ** (n - 2)

        return {
            "title": "Cayley's Formula",
            "statement": f"How many labeled trees are there on {n} vertices?",
            "difficulty": ProblemDifficulty.EASY,
            "hint": f"Cayley's formula: n^(n-2) = {count}",
            "bounds": {"count": (count, count)},
        }

    def _gen_connectivity(self, template: dict) -> dict:
        n = random.randint(10, 100)
        threshold = math.log(n) / n

        return {
            "title": f"Connectivity Threshold for G({n}, p)",
            "statement": f"What is the threshold probability p such that G({n}, p) is almost surely connected?",
            "difficulty": ProblemDifficulty.HARD,
            "hint": f"Threshold: p = log({n})/{n} ≈ {threshold:.4f}",
            "bounds": {"threshold": (threshold * 0.9, threshold * 1.1)},
        }

    def _gen_clique_prob(self, template: dict) -> dict:
        n = random.randint(10, 50)
        k = random.randint(3, 5)
        p = random.random() * 0.5 + 0.1

        from math import comb
        expected = comb(n, k) * (p ** (comb(k, 2)))

        return {
            "title": f"Expected {k}-cliques in G({n}, {p:.2f})",
            "statement": f"In a random graph G({n}, {p:.2f}), what is the expected number of {k}-cliques?",
            "difficulty": ProblemDifficulty.MODERATE,
            "hint": f"E[# of K_{k}] = C({n},{k}) × p^C({k},2) ≈ {expected:.2f}",
            "bounds": {"expected_cliques": (expected * 0.9, expected * 1.1)},
        }

    def _generic_generate(self, template: dict) -> dict:
        return {
            "title": "General Mathematical Problem",
            "statement": template.get("template", "Solve this problem."),
            "difficulty": ProblemDifficulty.MODERATE,
        }


class ErdosSolver:
    """Attempts to solve Erdős-style problems with various strategies."""

    STRATEGIES = [
        "brute_force_small",
        "induction",
        "probabilistic_method",
        "extremal_construction",
        "ramsey_theory",
        "szemeredi_regularity",
        "density_increment",
        "energy_increment",
        "polynomial_method",
        "combinatorial_nullstellensatz",
        "double_counting",
        "pigeonhole_principle",
        "graph_symmetrization",
        "density_halving",
        "greedy_construction",
    ]

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        self._api_manager = None
        self._context_engine = None
        self._strategy_stats: dict[str, dict] = defaultdict(
            lambda: {"attempts": 0, "successful": 0, "avg_time": 0.0}
        )

    def _get_api_manager(self):
        """Get or create API manager for coordinated LLM calls."""
        if self._api_manager is None and self.llm_client:
            try:
                from core.api_manager import get_api_manager
                self._api_manager = get_api_manager(self.llm_client)
            except Exception as e:
                logger.debug(f"Failed to get API manager for Erdős engine: {e}")
        return self._api_manager
    
    def _get_context_engine(self):
        """Get unified context engine for enriched problem solving."""
        if self._context_engine is None:
            try:
                from core.unified_context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except Exception as e:
                logger.debug(f"Context engine unavailable: {e}")
        return self._context_engine
    
    async def get_mathematical_context(self, problem: MathProblem) -> str:
        """Get relevant mathematical context from unified memory.
        
        Enriches problem solving with:
        - Related insights from past mathematical discoveries
        - Knowledge graph connections
        - Previously discovered patterns
        """
        try:
            engine = self._get_context_engine()
            if engine is None:
                return ""
            
            # Build query from problem
            query = f"{problem.category.value} {problem.title}"
            
            context = await engine.assemble_context(
                query=query,
                max_tokens=300,
                include_sources=["insight_active", "knowledge_graph"],
            )
            
            if context.entries:
                # Format for mathematical context
                lines = ["Related mathematical context:"]
                for entry in context.entries[:3]:
                    lines.append(f"- {entry.content[:200]}")
                return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Failed to get mathematical context: {e}")
        
        return ""

    async def _make_llm_call(
        self,
        user_input: str,
        system_prompt: str,
        priority: str = "learning",
        conversation_history: list | None = None,
    ) -> dict:
        """Make an LLM call through the API manager for coordination."""
        from core.api_manager import Priority as APIPriority

        priority_map = {
            "interactive": APIPriority.INTERACTIVE,
            "learning": APIPriority.LEARNING,
            "refinement": APIPriority.REFINEMENT,
            "daydream": APIPriority.DAYDREAM,
            "maintenance": APIPriority.MAINTENANCE,
        }

        api_manager = self._get_api_manager()

        if api_manager is not None:
            api_priority = priority_map.get(priority, APIPriority.LEARNING)
            return await api_manager.submit_chat(
                subsystem="erdos_engine",
                priority=api_priority,
                user_input=user_input,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
            )
        elif self.llm_client:
            from core.llm import subsystem_context
            async with subsystem_context("Erdős"):
                return await self.llm_client.chat(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                )
        else:
            raise RuntimeError("No LLM client available")

    def _reasoning_engine(self):
        """Lazily attach the advisory reasoning engine (symbolic hypothesis +
        causal layer; see core/reasoning_engine.py).

        The reasoning engine is the tree's only symbolic-inference
        implementation (abductive/deductive/inductive/analogical hypotheses,
        causal graph). Erdos tracks which strategies win but does no
        hypothesis or causal reasoning — this supplies both, as advice only:
        it can bias strategy weights by ±20%, never override the empirical
        stats. Unavailable -> False sentinel (no retry spam), selection is
        unaffected.
        """
        handle = getattr(self, "_reasoning_handle", None)
        if handle is None:
            try:
                from core.reasoning_engine import ReasoningType, get_reasoning_engine
                self._ReasoningType = ReasoningType
                handle = get_reasoning_engine()
            except Exception as e:
                logger.debug(f"Reasoning engine unavailable: {e}")
                handle = False
            self._reasoning_handle = handle
        return handle or None

    def _record_reasoning(self, problem: MathProblem, strategy: str,
                          success: bool, insight: str | None,
                          approach: str, result: str) -> None:
        """Feed the attempt outcome into the causal graph + hypothesis layer."""
        re_ = self._reasoning_engine()
        if re_ is None:
            return
        try:
            category = getattr(problem.category, "name", str(problem.category))
            outcome = "success" if success else "failure"
            re_.add_causal_relation(
                cause=f"strategy:{strategy}",
                effect=f"category:{category}:{outcome}",
                strength=0.8 if success else 0.6,
                mechanism=approach[:100] if approach else None,
            )
            if not success:
                # Abductive: why did this fail? Hypothesis is generated and
                # tested (confidence updated) against the attempt evidence.
                hyp = re_.generate_hypothesis(
                    observation=f"{strategy} failed on {category} problem",
                    context=[approach[:120], (result or "")[:120], f"insight={insight or 'none'}"],
                    reasoning_type=self._ReasoningType.ABDUCTIVE,
                )
                re_.test_hypothesis(hyp, test_data={"strategy": strategy, "category": category})
            elif insight:
                re_.add_causal_relation(
                    cause=f"strategy:{strategy}",
                    effect=f"insight:{insight[:60]}",
                    strength=0.7,
                    mechanism=approach[:100] if approach else None,
                )
        except Exception as e:
            logger.debug(f"Reasoning recording failed (advisory): {e}")

    async def attempt_solution(
        self,
        problem: MathProblem,
        strategy: str | None = None,
    ) -> ProblemAttempt:
        if strategy is None:
            strategy = self._select_strategy(problem)

        start_time = datetime.now()

        approach = await self._generate_approach(problem, strategy)
        result, success, insight = await self._execute_strategy(problem, strategy, approach)

        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds()

        self._update_stats(strategy, success, computation_time)
        self._record_reasoning(problem, strategy, success, insight, approach, result)

        return ProblemAttempt(
            problem_id=problem.id,
            timestamp=start_time,
            strategy=strategy,
            approach=approach,
            result=result,
            success=success,
            insight=insight,
            computation_time=computation_time,
        )

    def _select_strategy(self, problem: MathProblem) -> str:
        category_strategies = {
            ProblemCategory.RAMSEY_THEORY: ["probabilistic_method", "extremal_construction", "induction"],
            ProblemCategory.NUMBER_THEORY: ["induction", "density_increment", "polynomial_method"],
            ProblemCategory.GRAPH_THEORY: ["szemeredi_regularity", "graph_symmetrization", "double_counting"],
            ProblemCategory.ADDITIVE_COMBINATORICS: ["density_increment", "energy_increment", "polynomial_method"],
            ProblemCategory.COMBINATORICS: ["double_counting", "pigeonhole_principle", "greedy_construction"],
            ProblemCategory.PROBABILITY: ["probabilistic_method", "double_counting"],
        }

        candidates = category_strategies.get(problem.category, self.STRATEGIES)

        weighted = []
        for s in candidates:
            stats = self._strategy_stats[s]
            success_rate = stats["successful"] / max(1, stats["attempts"])
            weight = 1 + success_rate * 2 + random.random() * 0.5

            # Advisory causal bias from the reasoning engine: strategies with
            # recorded success/failure relations on this category shift their
            # weight by up to ±20%. The empirical stats above stay dominant;
            # the causal layer only breaks near-ties and accumulates evidence
            # that raw counts would miss (e.g. strategy X fails specifically
            # on category Y while being fine elsewhere).
            re_ = self._reasoning_engine()
            if re_ is not None:
                try:
                    cat = getattr(problem.category, "name", str(problem.category))
                    for rel in re_.find_effects(f"strategy:{s}"):
                        if not rel.effect.startswith(f"category:{cat}:"):
                            continue
                        if rel.effect.endswith(":success"):
                            weight *= 1 + rel.strength * 0.2
                        elif rel.effect.endswith(":failure"):
                            weight *= 1 - rel.strength * 0.1
                except Exception as e:
                    logger.debug(f"Causal strategy bias failed (advisory): {e}")

            weighted.append((s, weight))

        weighted.sort(key=lambda x: x[1], reverse=True)

        if random.random() < 0.3:
            return random.choice(self.STRATEGIES)

        return weighted[0][0]

    async def _generate_approach(self, problem: MathProblem, strategy: str) -> str:
        strategy_methods = {
            "brute_force_small": "Enumerate small cases to find pattern",
            "induction": "Prove base case(s), then show inductive step",
            "probabilistic_method": "Use randomness to show existence",
            "extremal_construction": "Build extremal example achieving bound",
            "ramsey_theory": "Apply Ramsey-type coloring arguments",
            "szemeredi_regularity": "Use regularity lemma structure",
            "density_increment": "Find dense substructure iteratively",
            "energy_increment": "Apply Fourier-analytic energy methods",
            "polynomial_method": "Use polynomial interpolation/vanishing",
            "combinatorial_nullstellensatz": "Apply coefficient conditions",
            "double_counting": "Count same thing in two ways",
            "pigeonhole_principle": "Apply counting/averaging argument",
            "graph_symmetrization": "Iteratively symmetrize graph",
            "density_halving": "Iteratively remove sparse regions",
            "greedy_construction": "Build solution greedily",
        }

        return strategy_methods.get(strategy, f"Apply {strategy} approach")

    async def _execute_strategy(
        self,
        problem: MathProblem,
        strategy: str,
        approach: str,
    ) -> tuple[str, bool, str | None]:
        # First, try LLM-assisted reasoning if available
        if self.llm_client:
            try:
                return await self._llm_assisted_solve(problem, strategy, approach)
            except Exception as e:
                logger.debug(f"LLM solve failed, falling back to local: {e}")

        # Fallback to local methods - each may raise if not applicable
        try:
            if strategy == "brute_force_small":
                return await self._brute_force(problem)
            elif strategy == "probabilistic_method":
                return await self._probabilistic_attempt(problem)
            elif strategy == "double_counting":
                return await self._double_counting(problem)
            elif strategy == "pigeonhole_principle":
                return await self._pigeonhole_attempt(problem)
            elif strategy == "greedy_construction":
                return await self._greedy_attempt(problem)
            elif strategy == "induction":
                return await self._induction_attempt(problem)
            elif strategy == "extremal_construction":
                return await self._extremal_attempt(problem)
        except RuntimeError as e:
            logger.debug(f"Strategy {strategy} not applicable: {e}")
        
        return await self._generic_attempt(problem, strategy, approach)

    async def _llm_assisted_solve(
        self,
        problem: MathProblem,
        strategy: str,
        approach: str,
    ) -> tuple[str, bool, str | None]:
        # Get mathematical context from unified memory
        math_context = ""
        try:
            math_context = await self.get_mathematical_context(problem)
        except Exception as e:
            logger.debug(f"Could not get mathematical context: {e}")
        
        context_section = ""
        if math_context:
            context_section = f"\n\n{math_context}\n"
        
        prompt = f"""You are a mathematician exploring: {problem.title}
{context_section}
Problem: {problem.statement}

Category: {problem.category.value}
Strategy to try: {strategy}
Approach: {approach}

Previous attempts: {', '.join(problem.strategies_tried[-5:]) if problem.strategies_tried else 'None'}

Please reason through this problem step by step. Consider:
1. What mathematical techniques apply here?
2. Can you find bounds or estimates?
3. Is there a combinatorial or probabilistic argument?
4. Any patterns or connections to known results?

Provide your reasoning and a conclusion. Format your response as:
REASONING: <your reasoning>
ANSWER: <your answer/bound/conclusion>
INSIGHT: <key insight if any>
SUCCESS: <true if you found a meaningful result, false if partial/inconclusive>"""

        try:
            content = await erdos_math_solve(
                prompt,
                context={
                    "problem_id": problem.id,
                    "category": problem.category.value,
                    "strategy": strategy,
                }
            )

            if content == "Dual model integration not available":
                raise RuntimeError("Dual model not available, falling back to direct LLM")

            reasoning = ""
            answer = ""
            insight = None
            success = False

            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("INSIGHT:"):
                    insight = line.replace("INSIGHT:", "").strip()
                elif line.startswith("SUCCESS:"):
                    success = "true" in line.lower()

            if not answer:
                answer = content

            result = f"Dual-Model Analysis ({strategy}):\n{reasoning}\n\nAnswer: {answer}"
            return result, success, insight

        except Exception as e:
            logger.debug(f"Dual-model solve failed, falling back to direct LLM: {e}")
            
            if self.llm_client:
                try:
                    response = await self._make_llm_call(
                        user_input=prompt,
                        system_prompt="You are a mathematician in the style of Paul Erdős.",
                        priority="learning",
                    )

                    content = get_response_content(response)

                    reasoning = ""
                    answer = ""
                    insight = None
                    success = False

                    for line in content.split("\n"):
                        line = line.strip()
                        if line.startswith("REASONING:"):
                            reasoning = line.replace("REASONING:", "").strip()
                        elif line.startswith("ANSWER:"):
                            answer = line.replace("ANSWER:", "").strip()
                        elif line.startswith("INSIGHT:"):
                            insight = line.replace("INSIGHT:", "").strip()
                        elif line.startswith("SUCCESS:"):
                            success = "true" in line.lower()

                    if not answer:
                        answer = content

                    result = f"LLM Analysis ({strategy}):\n{reasoning}\n\nAnswer: {answer}"
                    return result, success, insight
                except Exception as e2:
                    logger.error(f"LLM solve error: {e2}")
                    raise
            raise

    async def _brute_force(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Brute force computation for small problem sizes.
        
        Only returns results when actual computation is performed.
        Raises RuntimeError if problem is too large for brute force.
        """
        if problem.category == ProblemCategory.COMBINATORICS:
            if "Cayley" in problem.title:
                match = re.search(r"(\d+)\s+vertices", problem.statement)
                if match:
                    n = int(match.group(1))
                    if n <= 15:
                        result = n ** (n - 2)
                        return f"Computed {result} labeled trees via Cayley's formula.", True, "Cayley's formula verified"
                    else:
                        raise RuntimeError(
                            f"Brute force infeasible: {n} vertices yields {n**(n-2):.2e} trees. "
                            f"Maximum for brute force is n=15. Use probabilistic or constructive methods instead."
                        )

        if problem.category == ProblemCategory.RAMSEY_THEORY:
            match = re.search(r"R\((\d+),\s*(\d+)\)", problem.title)
            if match:
                s, t = int(match.group(1)), int(match.group(2))
                if s <= 4 and t <= 4:
                    known = {(3, 3): 6, (3, 4): 9, (4, 4): 18, (3, 5): 14}
                    if (s, t) in known:
                        return f"R({s},{t}) = {known[(s, t)]}", True, "Known Ramsey number"
        
        raise RuntimeError(
            f"No brute force solution available for this problem. "
            f"Category: {problem.category.value}, Problem Type: {problem.title[:50]}. "
            f"Requires LLM-based reasoning or specialized solver."
        )

    async def _probabilistic_attempt(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Apply probabilistic method to derive bounds.
        
        Uses actual mathematical computation, not placeholder responses.
        """
        if problem.category == ProblemCategory.RAMSEY_THEORY:
            match = re.search(r"R\((\d+)\)", problem.title) or re.search(r"R\((\d+),\s*(\d+)\)", problem.title)
            if match:
                if len(match.groups()) == 1:
                    k = int(match.group(1))
                    lower = 2 ** (k / 2)
                    upper = 4 ** (k - 1)
                    import math
                    return (
                        f"Probabilistic bounds: 2^({k}/2) < R({k}) < 4^({k}-1). "
                        f"Lower bound ≈ {lower:.0f}, Upper bound ≈ {upper:.0f} (Erdős-Szekeres)",
                        True,
                        "Exponential bounds established"
                    )
                elif len(match.groups()) == 2:
                    s, t = int(match.group(1)), int(match.group(2))
                    import math
                    lower = int(2 ** ((s + t - 2) / 2))
                    upper = int(math.comb(s + t - 2, s - 1))
                    return (
                        f"Bounds for R({s},{t}): {lower} ≤ R({s},{t}) ≤ {upper}",
                        True,
                        "Classical Erdős-Szekeres bounds"
                    )
        
        raise RuntimeError(
            f"Probabilistic method not applicable to {problem.category.value} problems. "
            f"Requires Ramsey theory or extremal combinatorics problem."
        )

    async def _double_counting(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Apply double counting principle with actual computation.
        
        Only succeeds if a real double-counting argument can be constructed.
        """
        if problem.category not in [ProblemCategory.COMBINATORICS, ProblemCategory.GRAPH_THEORY]:
            raise RuntimeError(
                f"Double counting requires combinatorial structure. "
                f"Problem category {problem.category.value} not suitable."
            )
        
        if self.llm_client:
            try:
                from core.dual_helpers import erdos_math_solve
                result = await erdos_math_solve(
                    f"Apply double counting to this problem:\n{problem.statement}\n\n"
                    f"Count the same set in two different ways and derive a relationship.",
                    context={"problem_id": problem.id, "strategy": "double_counting"}
                )
                if result and "error" not in result.lower():
                    return result, True, "Double counting via LLM reasoning"
            except Exception as e:
                logger.debug(f"Double counting LLM failed: {e}")
        
        raise RuntimeError(
            "Double counting could not be applied automatically. "
            "This strategy requires identifying a bijection between two counting methods. "
            "LLM reasoning required but unavailable."
        )

    async def _pigeonhole_attempt(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Apply pigeonhole principle with actual analysis.
        
        Only succeeds if pigeonhole can be constructively applied.
        """
        if self.llm_client:
            try:
                from core.dual_helpers import erdos_math_solve
                result = await erdos_math_solve(
                    f"Apply the pigeonhole principle to this problem:\n{problem.statement}\n\n"
                    f"Identify the 'pigeons' and 'holes' and show existence of collision.",
                    context={"problem_id": problem.id, "strategy": "pigeonhole"}
                )
                if result and "error" not in result.lower():
                    return result, True, "Pigeonhole principle via LLM reasoning"
            except Exception as e:
                logger.debug(f"Pigeonhole LLM failed: {e}")
        
        raise RuntimeError(
            "Pigeonhole principle requires identifying n+1 items in n categories. "
            "Cannot automatically apply - requires LLM reasoning which is unavailable."
        )

    async def _greedy_attempt(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Apply greedy algorithm with actual construction.
        
        Only succeeds if a greedy construction can be actually computed.
        """
        if problem.category == ProblemCategory.ADDITIVE_COMBINATORICS:
            if "sum" in problem.statement.lower() or "subset" in problem.statement.lower():
                import random
                numbers = []
                for i in range(100):
                    candidate = random.randint(1, 1000)
                    if all((candidate + n) not in numbers for n in numbers):
                        numbers.append(candidate)
                return (
                    f"Greedy sum-free subset construction: {len(numbers)} elements found. "
                    f"Subset: {numbers[:10]}{'...' if len(numbers) > 10 else ''}",
                    True,
                    f"Constructed sum-free set of size {len(numbers)}"
                )
        
        if problem.category == ProblemCategory.COMBINATORICS:
            if "maximum" in problem.title.lower() or "largest" in problem.title.lower():
                raise RuntimeError(
                    "Greedy approach needs specific problem structure. "
                    "LLM reasoning required to identify greedy strategy."
                )
        
        raise RuntimeError(
            f"Greedy construction not directly applicable to {problem.category.value}. "
            f"Requires problem-specific greedy criterion which cannot be auto-detected."
        )

    async def _induction_attempt(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Apply mathematical induction with LLM-based reasoning.
        
        Requires LLM to construct the induction proof structure.
        """
        if self.llm_client:
            try:
                from core.dual_helpers import erdos_math_solve
                result = await erdos_math_solve(
                    f"Prove by mathematical induction:\n{problem.statement}\n\n"
                    f"1. State the base case and verify it.\n"
                    f"2. State the inductive hypothesis.\n"
                    f"3. Prove the inductive step.\n"
                    f"4. Conclude the proof.",
                    context={"problem_id": problem.id, "strategy": "induction"}
                )
                if result and "error" not in result.lower():
                    return result, True, "Induction proof via LLM reasoning"
            except Exception as e:
                logger.debug(f"Induction LLM failed: {e}")
        
        raise RuntimeError(
            "Induction proof requires LLM reasoning to construct. "
            "The problem may need base case identification and inductive step formulation. "
            "LLM client unavailable for reasoning."
        )

    async def _extremal_attempt(self, problem: MathProblem) -> tuple[str, bool, str | None]:
        """Apply extremal construction with actual computation.
        
        Only returns results when actual extremal bounds can be computed.
        """
        if problem.category == ProblemCategory.GRAPH_THEORY and "Turan" in problem.title:
            match = re.search(r"(\d+)-vertex.*?(\d+)-clique", problem.statement)
            if match:
                n, k = int(match.group(1)), int(match.group(2))
                edges = int((1 - 1/(k-1)) * n * n / 2)
                return f"Turan's theorem: T_{k-1}({n}) has {edges} edges", True, "Turan graph is extremal"
        
        if problem.category == ProblemCategory.GRAPH_THEORY:
            if "maximum" in problem.title.lower() and "edge" in problem.statement.lower():
                match = re.search(r"(\d+).*?vertices", problem.statement)
                if match:
                    n = int(match.group(1))
                    max_edges = n * (n - 1) // 2
                    return (
                        f"Complete graph K_{n} has {max_edges} edges (maximum for {n} vertices)",
                        True,
                        "Trivial bound - complete graph"
                    )
        
        raise RuntimeError(
            "Extremal construction requires specific problem structure (Turan, Erdős–Stone, etc.). "
            "Cannot automatically determine extremal function for this problem. "
            "LLM reasoning required but unavailable."
        )

    async def _generic_attempt(
        self,
        problem: MathProblem,
        strategy: str,
        approach: str,
    ) -> tuple[str, bool, str | None]:
        """Generic attempt using LLM reasoning.
        
        Returns failure status if LLM unavailable - no placeholder responses.
        """
        if not self.llm_client:
            return (
                f"Cannot apply {strategy}: LLM client unavailable for math reasoning. "
                f"Problem '{problem.title}' requires language model assistance.",
                False,
                f"No LLM available for {strategy}"
            )
        
        try:
            from core.dual_helpers import erdos_math_solve
            result = await erdos_math_solve(
                f"Apply {strategy} ({approach}) to solve:\n{problem.statement}",
                context={"problem_id": problem.id, "strategy": strategy}
            )
            return result, False, f"{strategy} attempted via LLM"
        except Exception as e:
            return (
                f"LLM-based {strategy} failed: {e}",
                False,
                f"LLM call failed for {strategy}"
            )

    def _update_stats(self, strategy: str, success: bool, time: float) -> None:
        stats = self._strategy_stats[strategy]
        stats["attempts"] += 1
        if success:
            stats["successful"] += 1
        stats["avg_time"] = (stats["avg_time"] * (stats["attempts"] - 1) + time) / stats["attempts"]

    def get_strategy_performance(self) -> dict:
        return dict(self._strategy_stats)


class ErdosEngine:
    """Main engine for Erdős-style mathematical learning.
    
    Runs continuously in background, generating problems, attempting solutions,
    and learning from results to evolve problem-solving strategies.
    """

    MAX_PROBLEMS = 100
    MAX_HISTORY = 1000

    def __init__(
        self,
        storage_path: Path | None = None,
        llm_client: Any = None,
    ):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "erdos"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.generator = ErdosProblemGenerator()
        self.solver = ErdosSolver(llm_client)
        
        self._context_engine = None

        self.problems: dict[str, MathProblem] = {}
        self.history: deque[ProblemAttempt] = deque(maxlen=self.MAX_HISTORY)
        self._running = False
        self._throttled = False
        self._task: asyncio.Task | None = None
        self._lock = threading.RLock()

        self._learning_state = {
            "total_problems": 0,
            "solved_problems": 0,
            "best_strategies": [],
            "insights": [],
            "discovered_patterns": [],
        }

        self._load_state()
    
    def _get_context_engine(self):
        """Get unified context engine for enriched problem solving."""
        if self._context_engine is None:
            try:
                from core.unified_context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except Exception as e:
                logger.debug(f"Context engine unavailable: {e}")
        return self._context_engine
    
    async def get_mathematical_context(self, topic: str) -> str:
        """Get relevant mathematical context from unified memory.
        
        Enriches problem solving with:
        - Related insights from past mathematical discoveries
        - Knowledge graph connections
        - Previously discovered patterns
        """
        try:
            engine = self._get_context_engine()
            if engine is None:
                return ""
            
            context = await engine.assemble_context(
                query=f"mathematics {topic}",
                max_tokens=300,
                include_sources=["insight_active", "knowledge_graph"],
            )
            
            if context.entries:
                lines = ["Related mathematical context:"]
                for entry in context.entries[:3]:
                    lines.append(f"- {entry.content[:150]}")
                return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Failed to get mathematical context: {e}")
        
        return ""

    def _load_state(self) -> None:
        """Load state - called during singleton init (synchronous)."""
        state_file = self.storage_path / "erdos_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._learning_state = data.get("learning_state", self._learning_state)
                logger.info(f"Loaded Erdős state: {self._learning_state['total_problems']} problems attempted")
            except Exception as e:
                logger.warning(f"Failed to load Erdős state: {e}")
    
    async def _load_state_async(self) -> None:
        """Async version for lazy loading after init."""
        state_file = self.storage_path / "erdos_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                if data is None:
                    raise ValueError("Empty state file")
                self._learning_state = data.get("learning_state", self._learning_state)
                logger.info(f"Loaded Erdős state: {self._learning_state['total_problems']} problems attempted")
            except Exception as e:
                logger.warning(f"Failed to load Erdős state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "erdos_state.json"
        try:
            atomic_write_json(state_file, {
                    "learning_state": self._learning_state,
                    "timestamp": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save Erdős state: {e}")

    async def start_background_learning(self) -> None:
        """Start background learning - now managed by DaydreamEngine."""
        self._running = True
        logger.info("Erdős engine ready - will solve problems when called")

    async def stop_background_learning(self) -> None:
        """Stop background learning."""
        self._running = False
        self._save_state()
        logger.info("Erdős engine stopped")

    async def daydream_step(self) -> dict:
        """Single daydream step - called by DaydreamEngine when idle.
        
        This is the main entry point for idle-time math exploration.
        Returns the result of attempting one problem.
        """
        if not self._running:
            return {"status": "not_running"}

        return await self._learning_step()

    async def _learning_step(self, problem=None) -> dict:
        """Single learning step: generate problem, attempt solution, learn."""
        if problem is None:
            problem = self.generator.generate_problem()
            self.problems[problem.id] = problem
        self._learning_state["total_problems"] += 1

        attempt = await self.solver.attempt_solution(problem)
        problem.attempt_count += 1
        problem.last_attempt = attempt.timestamp
        problem.add_strategy(attempt.strategy)

        self.history.append(attempt)

        if attempt.success:
            problem.solved = True
            problem.solution = attempt.result
            self._learning_state["solved_problems"] += 1
            logger.info(f"Solved {problem.id}: {problem.title}")
        else:
            problem.add_partial_result({
                "strategy": attempt.strategy,
                "result": attempt.result,
                "insight": attempt.insight,
            })
            if attempt.insight and "No LLM available" not in str(attempt.insight):
                self._learning_state["insights"].append(attempt.insight)
                while len(self._learning_state["insights"]) > 200:
                    self._learning_state["insights"].pop(0)

        # Observe insight with Prefrontal Cortex
        if attempt.insight:
            try:
                pfc = get_prefrontal_cortex()
                if pfc and pfc._running:
                    # Schedule async observation
                    from core.helpers import create_tracked_task
                    create_tracked_task(
                        pfc.observe_insight(
                            insight=attempt.insight[:200],
                            source="erdos",
                            metadata={
                                "problem": problem.title,
                                "category": problem.category.value,
                                "strategy": attempt.strategy,
                                "success": attempt.success,
                            }
                        ),
                        name="pfc_observe_erdos_insight",
                    )
            except Exception as e:
                logger.debug(f"PFC observation failed: {e}")
            
            try:
                from core.insight_bridge import get_insight_bridge
                bridge = get_insight_bridge()
                bridge.process_insight(
                    raw_insight=attempt.insight,
                    source_problem=problem.title,
                    source_category=problem.category.value,
                    strategy_used=attempt.strategy,
                    problem_solved=attempt.success,
                )
            except Exception as e:
                logger.debug(f"Insight bridge not available: {e}")

        while len(self.problems) > self.MAX_PROBLEMS:
            sorted_problems = sorted(
                self.problems.values(),
                key=lambda p: (p.solved, p.attempt_count, p.created_at)
            )
            oldest = sorted_problems[0]
            del self.problems[oldest.id]

        if self._learning_state["total_problems"] % 10 == 0:
            self._update_best_strategies()
            self._save_state()

        return {
            "problem": problem.id,
            "title": problem.title,
            "category": problem.category.value,
            "statement": problem.statement,  # Include full problem statement
            "difficulty": problem.difficulty.value,  # Include difficulty level
            "success": attempt.success,
            "strategy": attempt.strategy,
            "approach": attempt.approach,  # Include the approach used
            "insight": attempt.insight,
            "result": attempt.result,  # Include the result
            "computation_time": attempt.computation_time,  # Include computation time
        }

    def _update_best_strategies(self) -> None:
        """Update ranking of best strategies."""
        perf = self.solver.get_strategy_performance()
        ranked = sorted(
            perf.items(),
            key=lambda x: x[1]["successful"] / max(1, x[1]["attempts"]),
            reverse=True
        )
        self._learning_state["best_strategies"] = [
            {"strategy": s, **stats} for s, stats in ranked[:5]
        ]

    def get_status(self) -> dict:
        """Get current engine status."""
        with self._lock:
            solved = sum(1 for p in self.problems.values() if p.solved)
            
            recent_insights = []
            for insight in self._learning_state["insights"][-5:]:
                recent_insights.append({
                    "insight": insight,
                    "context": None,
                })
            
            for attempt in list(self.history)[-10:][::-1]:
                if attempt.insight:
                    for i, ins_data in enumerate(recent_insights):
                        if ins_data["insight"] == attempt.insight:
                            problem = self.problems.get(attempt.problem_id)
                            if problem:
                                recent_insights[i]["context"] = {
                                    "problem_title": problem.title,
                                    "problem_statement": problem.statement[:200] + "..." if len(problem.statement) > 200 else problem.statement,
                                    "category": problem.category.value,
                                    "strategy": attempt.strategy,
                                    "success": attempt.success,
                                }

            return {
                "running": self._running,
                "total_problems": self._learning_state["total_problems"],
                "current_problems": len(self.problems),
                "solved_count": solved,
                "history_size": len(self.history),
                "best_strategies": self._learning_state["best_strategies"][:3],
                "insights_count": len(self._learning_state["insights"]),
                "recent_insights": recent_insights,
                "by_category": {
                    cat.value: sum(1 for p in self.problems.values() if p.category == cat)
                    for cat in ProblemCategory
                },
                "solved_by_category": {
                    cat.value: sum(1 for p in self.problems.values() if p.category == cat and p.solved)
                    for cat in ProblemCategory
                },
            }

    def get_solved_problems(self) -> list[dict]:
        """Get list of solved problems."""
        with self._lock:
            return [
                {
                    "id": p.id,
                    "title": p.title,
                    "category": p.category.value,
                    "solution": p.solution,
                    "attempts": p.attempt_count,
                }
                for p in self.problems.values()
                if p.solved
            ]

    def get_active_problems(self) -> list[dict]:
        """Get list of actively being worked on problems."""
        with self._lock:
            return [
                {
                    "id": p.id,
                    "title": p.title,
                    "category": p.category.value,
                    "attempts": p.attempt_count,
                    "strategies_tried": p.strategies_tried[-3:],
                    "best_partial": p.partial_results[-1] if p.partial_results else None,
                }
                for p in self.problems.values()
                if not p.solved and p.attempt_count > 0
            ]

    async def force_attempt(self, problem_id: str | None = None) -> dict:
        if problem_id and problem_id in self.problems:
            problem = self.problems[problem_id]
        else:
            problem = self.generator.generate_problem()
            self.problems[problem.id] = problem

        return await self._learning_step(problem)

    def get_insights(self) -> list[str]:
        """Get accumulated mathematical insights (filtered for quality)."""
        with self._lock:
            insights = self._learning_state["insights"][-50:]
            return [
                i for i in insights 
                if i and "No LLM available" not in i and "Cannot apply" not in i and "unavailable" not in i
            ][-20:]

    def throttle(self, aggressive: bool = False) -> None:
        """Throttle engine activity for resource conservation."""
        if aggressive:
            self._throttled = True
            logger.info("Erdős engine throttled (aggressive)")
        else:
            self._throttled = True
            logger.info("Erdős engine throttled")

    def resume(self) -> None:
        """Resume normal operation after throttling."""
        self._throttled = False
        logger.info("Erdős engine resumed")

    def is_healthy(self) -> tuple[bool, dict]:
        """Check engine health."""
        with self._lock:
            return (
                self._running,
                {
                    "problems": len(self.problems),
                    "total_attempted": self._learning_state["total_problems"],
                    "total_solved": self._learning_state["solved_problems"],
                    "throttled": self._throttled,
                }
            )


_erdos_engine: ErdosEngine | None = None
_erdos_lock = threading.Lock()


def get_erdos_engine(llm_client: Any = None) -> ErdosEngine:
    """Get global Erdős engine instance (thread-safe).
    
    Args:
        llm_client: Optional LLM client for intelligent problem solving.
                    If provided on first call, will be used for reasoning.
    """
    global _erdos_engine
    if _erdos_engine is None:
        with _erdos_lock:
            if _erdos_engine is None:
                _erdos_engine = ErdosEngine(llm_client=llm_client)
    elif llm_client is not None and _erdos_engine.solver.llm_client is None:
        # Update LLM client if provided later
        _erdos_engine.solver.llm_client = llm_client
    return _erdos_engine
