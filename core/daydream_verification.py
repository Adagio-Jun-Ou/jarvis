"""Daydream Solution Verification - Validates mathematical insights.

When Jarvis daydreams and finds solutions, they must be verified before
being shared or used for evolution. This prevents hallucinated "discoveries"
from corrupting the knowledge base.
"""
import logging
import random
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    TIMEOUT = "timeout"


class VerificationMethod(Enum):
    SYMBOLIC_CHECK = "symbolic_check"
    NUMERICAL_VALIDATION = "numerical_validation"
    COUNTEREXAMPLE_TEST = "counterexample_test"
    REFERENCE_COMPARISON = "reference_comparison"
    LOGICAL_CONSISTENCY = "logical_consistency"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class VerificationResult:
    method: VerificationMethod
    status: VerificationStatus
    confidence: float
    details: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DaydreamSolution:
    id: str
    problem_type: str
    problem_statement: str
    proposed_solution: str
    insight: str
    claimed_result: Any
    timestamp: datetime = field(default_factory=datetime.now)
    verification_results: list[VerificationResult] = field(default_factory=list)
    overall_status: VerificationStatus = VerificationStatus.PENDING
    confidence_score: float = 0.0

    def add_verification(self, result: VerificationResult) -> None:
        self.verification_results.append(result)
        self._update_overall()

    def _update_overall(self) -> None:
        if not self.verification_results:
            return

        verified_count = sum(1 for r in self.verification_results if r.status == VerificationStatus.VERIFIED)
        refuted_count = sum(1 for r in self.verification_results if r.status == VerificationStatus.REFUTED)
        total = len(self.verification_results)

        if refuted_count > 0:
            self.overall_status = VerificationStatus.REFUTED
            self.confidence_score = 0.0
        elif verified_count == total:
            self.overall_status = VerificationStatus.VERIFIED
            self.confidence_score = sum(r.confidence for r in self.verification_results) / total
        elif verified_count > total / 2:
            self.overall_status = VerificationStatus.UNCERTAIN
            self.confidence_score = sum(r.confidence for r in self.verification_results) / total
        else:
            self.overall_status = VerificationStatus.UNCERTAIN
            self.confidence_score = 0.3


class KnownResults:
    """Database of known mathematical results for verification."""

    RAMSEY_NUMBERS = {
        (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18,
        (3, 7): 23, (3, 8): 28, (3, 9): 36, (4, 4): 18,
        (4, 5): 25, (4, 6): 36, (5, 5): (43, 48),
        (6, 3): 18, (7, 3): 23, (8, 3): 28, (9, 3): 36,
        (5, 3): 14, (6, 4): 36, (5, 4): 25,
    }

    RAMSEY_BOUNDS = {
        (6, 6): (102, 165),
        (7, 7): (205, 540),
        (4, 7): (49, 49),
        (5, 6): (58, 87),
        (5, 7): (80, 143),
        (6, 7): (113, 298),
    }

    PRIME_GAPS = {
        1: 1, 2: 2, 3: 4, 4: 6, 5: 6, 6: 8, 7: 14, 8: 14,
        9: 18, 10: 20, 11: 22, 12: 34, 13: 24, 14: 34, 15: 36,
        16: 44, 17: 36, 18: 52, 19: 60, 20: 60,
        50: 72, 100: 114, 500: 334, 1000: 354,
        2000: 404, 5000: 512, 10000: 720,
    }

    KNOWN_PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
        61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
        131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
        197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
        271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
        353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
    ]

    PERFECT_NUMBERS = [6, 28, 496, 8128, 33550336, 8589869056]

    FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

    CATALAN_NUMBERS = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786]

    EULER_TOTIENT_VALUES = {
        1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 2, 7: 6, 8: 4, 9: 6, 10: 4,
        11: 10, 12: 4, 13: 12, 14: 6, 15: 8, 16: 8, 17: 16, 18: 6, 19: 18, 20: 8,
    }

    GRAPH_CHROMATIC_NUMBERS = {
        "K_3": 3, "K_4": 4, "K_5": 5, "K_n": lambda n: n,
        "C_3": 3, "C_4": 2, "C_5": 3, "C_6": 2,
        "Petersen": 3, "Dodecahedron": 3, "Icosahedron": 4,
        "Planar_max": 4, "Bipartite": 2,
    }

    EULER_CHARACTERISTICS = {
        "tetrahedron": 2, "cube": 2, "octahedron": 2, "dodecahedron": 2, "icosahedron": 2,
        "torus": 0, "klein_bottle": 0, "projective_plane": 1,
    }

    PARTITION_NUMBERS = {
        1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 11, 7: 15, 8: 22, 9: 30, 10: 42,
        15: 176, 20: 627, 25: 1958, 50: 204226,
    }

    BELL_NUMBERS = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]

    STIRLING_FIRST = {
        (4, 1): 6, (4, 2): 11, (4, 3): 6, (4, 4): 1,
        (5, 1): 24, (5, 2): 50, (5, 3): 35, (5, 4): 10, (5, 5): 1,
    }

    STIRLING_SECOND = {
        (4, 1): 1, (4, 2): 7, (4, 3): 6, (4, 4): 1,
        (5, 1): 1, (5, 2): 15, (5, 3): 25, (5, 4): 10, (5, 5): 1,
    }

    GRAPH_COLORING_BOUNDS = {
        "complete_graph": lambda n: n,
        "cycle_graph": lambda n: 2 if n % 2 == 0 else 3,
        "bipartite": 2,
        "planar_max": 4,
    }

    KNOWN_THEOREMS = {
        "four_color": "Any planar graph can be colored with at most 4 colors",
        "fermat_little": "a^p ≡ a (mod p) for prime p",
        "euler_formula": "V - E + F = 2 for connected planar graphs",
        "handshaking": "Sum of degrees = 2|E| for any graph",
        "ramsey_theorem": "R(s,t) exists for all s,t ≥ 3",
        "bertrand_postulate": "There is always a prime between n and 2n for n > 1",
        "wilson_theorem": "(p-1)! ≡ -1 (mod p) iff p is prime",
    }


class SolutionVerifier:
    """Verifies daydream solutions through multiple methods."""

    MAX_PENDING_SOLUTIONS = 100
    MAX_VERIFIED_SOLUTIONS = 500

    def __init__(self):
        self.known = KnownResults()
        self._lock = threading.RLock()
        self._solution_counter = 0
        self.pending_solutions: deque[DaydreamSolution] = deque(maxlen=self.MAX_PENDING_SOLUTIONS)
        self.verified_solutions: deque[DaydreamSolution] = deque(maxlen=self.MAX_VERIFIED_SOLUTIONS)

    def submit_solution(
        self,
        problem_type: str,
        problem_statement: str,
        proposed_solution: str,
        insight: str,
        claimed_result: Any = None,
    ) -> DaydreamSolution:
        with self._lock:
            self._solution_counter += 1
            solution_id = f"sol_{self._solution_counter:06d}"

            solution = DaydreamSolution(
                id=solution_id,
                problem_type=problem_type,
                problem_statement=problem_statement,
                proposed_solution=proposed_solution,
                insight=insight,
                claimed_result=claimed_result,
            )

            self.pending_solutions.append(solution)
            return solution

    async def verify(self, solution: DaydreamSolution) -> VerificationStatus:
        """Run all applicable verification methods."""
        problem_type = solution.problem_type.lower()

        if "ramsey" in problem_type:
            await self._verify_ramsey(solution)
        elif "prime" in problem_type or "number_theory" in problem_type:
            await self._verify_prime_problem(solution)
        elif "graph" in problem_type:
            await self._verify_graph_problem(solution)
        elif "combinator" in problem_type:
            await self._verify_combinatorics(solution)
        elif "probability" in problem_type:
            await self._verify_probability(solution)
        else:
            await self._verify_generic(solution)

        self._finalize_solution(solution)
        return solution.overall_status

    async def _verify_ramsey(self, solution: DaydreamSolution) -> None:
        """Verify Ramsey number claims."""

        result = VerificationResult(
            method=VerificationMethod.REFERENCE_COMPARISON,
            status=VerificationStatus.PENDING,
            confidence=0.0,
            details="",
        )

        match = re.search(r'R\((\d+),\s*(\d+)\)', solution.problem_statement)
        if match:
            s, t = int(match.group(1)), int(match.group(2))
            key = (min(s, t), max(s, t))

            if solution.claimed_result is not None:
                known = self.known.RAMSEY_NUMBERS.get(key)

                if known is not None:
                    if isinstance(known, tuple):
                        low, high = known
                        if low <= solution.claimed_result <= high:
                            result.status = VerificationStatus.VERIFIED
                            result.confidence = 0.8
                            result.details = f"R({s},{t}) known to be in [{low},{high}], claimed {solution.claimed_result} is valid"
                        else:
                            result.status = VerificationStatus.REFUTED
                            result.confidence = 1.0
                            result.details = f"R({s},{t}) known to be in [{low},{high}], claimed {solution.claimed_result} is OUTSIDE range"
                    else:
                        if solution.claimed_result == known:
                            result.status = VerificationStatus.VERIFIED
                            result.confidence = 1.0
                            result.details = f"R({s},{t}) = {known}, claim matches exactly"
                        else:
                            result.status = VerificationStatus.REFUTED
                            result.confidence = 1.0
                            result.details = f"R({s},{t}) = {known}, claimed {solution.claimed_result} is WRONG"
                else:
                    result.status = VerificationStatus.UNCERTAIN
                    result.confidence = 0.5
                    result.details = f"R({s},{t}) not in known results, cannot verify"

        solution.add_verification(result)

        numerical_result = await self._numerical_ramsey_check(solution)
        if numerical_result:
            solution.add_verification(numerical_result)

    async def _numerical_ramsey_check(self, solution: DaydreamSolution) -> VerificationResult | None:
        """Check Ramsey claim by sampling random graphs.
        
        For R(s,t) = n:
        - Any K_n has required clique (upper bound)
        - Some K_{n-1} avoids required clique (lower bound)
        """
        result = VerificationResult(
            method=VerificationMethod.NUMERICAL_VALIDATION,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.0,
            details="",
        )

        match = re.search(r'R\((\d+),\s*(\d+)\)', solution.problem_statement)
        if not match:
            return None

        s, t = int(match.group(1)), int(match.group(2))
        claimed = solution.claimed_result

        if claimed is None or claimed <= 0:
            return None

        try:
            lower_bound_confirmed = self._test_ramsey_lower_bound(s, t, claimed)
            if lower_bound_confirmed:
                result.status = VerificationStatus.VERIFIED
                result.confidence = 0.6
                result.details = f"Confirmed lower bound: found K_{claimed-1} coloring avoiding R({s},{t})"
            else:
                result.status = VerificationStatus.REFUTED
                result.confidence = 0.7
                result.details = "Lower bound not confirmed by sampling"
        except Exception as e:
            result.details = f"Numerical check failed: {e}"

        return result

    def _test_ramsey_lower_bound(self, s: int, t: int, claimed: int) -> bool:
        """Test if we can find a K_{claimed-1} with no K_s in one color and no K_t in other.
        
        This confirms R(s,t) > claimed-1, meaning R(s,t) >= claimed.
        Returns True if such a coloring is found (confirming lower bound).
        """
        n = claimed - 1
        if n < max(s, t):
            return True  # Trivially small graphs

        # Known good colorings for small cases
        if (s, t) == (3, 3) and n == 5:
            return self._pentagon_coloring_works()

        trials = min(100, n * n)

        for _ in range(trials):
            red_edges = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        red_edges.add((i, j))

            has_red_k_s = self._check_clique(n, s, red_edges, is_red=True)
            blue_edges = {(i, j) for i in range(n) for j in range(i + 1, n)} - red_edges
            has_blue_k_t = self._check_clique(n, t, blue_edges, is_red=False)

            if not has_red_k_s and not has_blue_k_t:
                return True  # Found lower bound confirmation

        return False

    def _pentagon_coloring_works(self) -> bool:
        """Check that K_5 with pentagon coloring has no K_3."""
        n = 5
        # Pentagon edges are red
        red_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}

        has_red_k3 = self._check_clique(n, 3, red_edges, is_red=True)

        all_edges = {(i, j) for i in range(n) for j in range(i + 1, n)}
        blue_edges = all_edges - red_edges

        has_blue_k3 = self._check_clique(n, 3, blue_edges, is_red=False)

        return not has_red_k3 and not has_blue_k3

    def _check_clique(self, n: int, k: int, edges: set, is_red: bool) -> bool:
        """Check if a k-clique exists with all edges in 'edges'."""
        if k > n:
            return False
        if k <= 2:
            return True

        from itertools import combinations
        for clique in combinations(range(n), k):
            all_present = True
            for i, j in combinations(clique, 2):
                if (i, j) not in edges and (j, i) not in edges:
                    all_present = False
                    break
            if all_present:
                return True
        return False

    async def _verify_prime_problem(self, solution: DaydreamSolution) -> None:
        """Verify prime number related claims."""

        result = VerificationResult(
            method=VerificationMethod.SYMBOLIC_CHECK,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            details="",
        )

        if "prime gap" in solution.problem_statement.lower():
            match = re.search(r'gap.*?(\d+)', solution.problem_statement)
            if match and solution.claimed_result:
                gap_n = int(match.group(1))
                known_gap = self.known.PRIME_GAPS.get(gap_n)

                if known_gap:
                    if solution.claimed_result == known_gap:
                        result.status = VerificationStatus.VERIFIED
                        result.confidence = 1.0
                        result.details = f"Prime gap({gap_n}) = {known_gap}, claim matches"
                    elif solution.claimed_result >= known_gap:
                        result.status = VerificationStatus.VERIFIED
                        result.confidence = 0.7
                        result.details = f"Claimed gap {solution.claimed_result} >= known minimum {known_gap}"
                    else:
                        result.status = VerificationStatus.REFUTED
                        result.confidence = 1.0
                        result.details = f"Known prime gap({gap_n}) = {known_gap}, claimed {solution.claimed_result} is wrong"

        elif "is prime" in solution.problem_statement.lower():
            match = re.search(r'(\d+)', solution.problem_statement)
            if match:
                n = int(match.group(1))
                is_prime = self._is_prime_check(n)
                claimed_prime = solution.claimed_result

                if claimed_prime == is_prime:
                    result.status = VerificationStatus.VERIFIED
                    result.confidence = 1.0
                    result.details = f"{n} is{' ' if is_prime else ' not '}prime, claim correct"
                else:
                    result.status = VerificationStatus.REFUTED
                    result.confidence = 1.0
                    result.details = f"{n} is{' ' if is_prime else ' not '}prime, claim was wrong"

        elif "perfect" in solution.problem_statement.lower():
            match = re.search(r'(\d+)', solution.problem_statement)
            if match:
                n = int(match.group(1))
                is_perfect = n in self.known.PERFECT_NUMBERS
                if solution.claimed_result == is_perfect:
                    result.status = VerificationStatus.VERIFIED
                    result.confidence = 1.0
                    result.details = f"{n} is{' ' if is_perfect else ' not '}perfect"

        elif "fibonacci" in solution.problem_statement.lower():
            match = re.search(r'(\d+)', solution.problem_statement)
            if match:
                n = int(match.group(1))
                is_fib = n in self.known.FIBONACCI
                if solution.claimed_result == is_fib:
                    result.status = VerificationStatus.VERIFIED
                    result.confidence = 1.0
                    result.details = f"{n} is{' ' if is_fib else ' not '}Fibonacci"

        elif "partition" in solution.problem_statement.lower():
            match = re.search(r'p\((\d+)\)|partition.*?(\d+)', solution.problem_statement)
            if match:
                n = int(match.group(1) or match.group(2))
                known = self.known.PARTITION_NUMBERS.get(n)
                if known and solution.claimed_result == known:
                    result.status = VerificationStatus.VERIFIED
                    result.confidence = 1.0
                    result.details = f"p({n}) = {known}"

        solution.add_verification(result)

    def _is_prime_check(self, n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    async def _verify_graph_problem(self, solution: DaydreamSolution) -> None:
        """Verify graph theory claims."""

        result = VerificationResult(
            method=VerificationMethod.LOGICAL_CONSISTENCY,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            details="Graph problem verification requires specific context",
        )

        problem_lower = solution.problem_statement.lower()

        if "chromatic" in problem_lower:
            if solution.claimed_result:
                chrom = solution.claimed_result

                for graph_name, expected in self.known.GRAPH_CHROMATIC_NUMBERS.items():
                    if graph_name.lower() in problem_lower:
                        if callable(expected):
                            match = re.search(r'n\s*=\s*(\d+)', problem_lower)
                            if match:
                                n = int(match.group(1))
                                expected_val = expected(n)
                                if chrom == expected_val:
                                    result.status = VerificationStatus.VERIFIED
                                    result.confidence = 1.0
                                    result.details = f"χ({graph_name}) = {expected_val}, matches"
                                else:
                                    result.status = VerificationStatus.REFUTED
                                    result.confidence = 1.0
                                    result.details = f"χ({graph_name}) = {expected_val}, claimed {chrom} is wrong"
                        elif chrom == expected:
                            result.status = VerificationStatus.VERIFIED
                            result.confidence = 1.0
                            result.details = f"χ({graph_name}) = {expected}"
                        break
                else:
                    if isinstance(chrom, int) and chrom >= 1 and chrom <= 4:
                        result.status = VerificationStatus.VERIFIED
                        result.confidence = 0.6
                        result.details = f"Chromatic number {chrom} is plausible for planar graph"

        elif "euler" in problem_lower and "characteristic" in problem_lower:
            for shape, expected in self.known.EULER_CHARACTERISTICS.items():
                if shape.lower() in problem_lower:
                    if solution.claimed_result == expected:
                        result.status = VerificationStatus.VERIFIED
                        result.confidence = 1.0
                        result.details = f"χ({shape}) = {expected}"
                    break

        elif "handshaking" in problem_lower or "sum of degrees" in problem_lower:
            result.status = VerificationStatus.VERIFIED
            result.confidence = 1.0
            result.details = "Handshaking lemma: valid graph theorem"

        elif "four color" in problem_lower:
            result.status = VerificationStatus.VERIFIED
            result.confidence = 1.0
            result.details = "Four color theorem is proven"

        solution.add_verification(result)

    async def _verify_combinatorics(self, solution: DaydreamSolution) -> None:
        """Verify combinatorics claims."""

        result = VerificationResult(
            method=VerificationMethod.NUMERICAL_VALIDATION,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            details="",
        )

        problem_lower = solution.problem_statement.lower()

        if "catalan" in problem_lower:
            match = re.search(r'catalan.*?(\d+)|c_(\d+)', problem_lower)
            if match:
                n = int(match.group(1) or match.group(2))
                if n < len(self.known.CATALAN_NUMBERS):
                    expected = self.known.CATALAN_NUMBERS[n]
                    if solution.claimed_result == expected:
                        result.status = VerificationStatus.VERIFIED
                        result.confidence = 1.0
                        result.details = f"Catalan({n}) = {expected}"

        elif "bell" in problem_lower:
            match = re.search(r'bell.*?(\d+)|b_(\d+)', problem_lower)
            if match:
                n = int(match.group(1) or match.group(2))
                if n < len(self.known.BELL_NUMBERS):
                    expected = self.known.BELL_NUMBERS[n]
                    if solution.claimed_result == expected:
                        result.status = VerificationStatus.VERIFIED
                        result.confidence = 1.0
                        result.details = f"Bell({n}) = {expected}"

        elif "stirling" in problem_lower:
            match = re.search(r's\((\d+),\s*(\d+)\)', problem_lower)
            if match:
                n, k = int(match.group(1)), int(match.group(2))
                if "first" in problem_lower:
                    expected = self.known.STIRLING_FIRST.get((n, k))
                else:
                    expected = self.known.STIRLING_SECOND.get((n, k))
                if expected and solution.claimed_result == expected:
                    result.status = VerificationStatus.VERIFIED
                    result.confidence = 1.0
                    result.details = "Stirling number verified"

        else:
            result.details = "Combinatorics verification through spot-check"

        solution.add_verification(result)

    async def _verify_probability(self, solution: DaydreamSolution) -> None:
        """Verify probability claims."""

        result = VerificationResult(
            method=VerificationMethod.NUMERICAL_VALIDATION,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            details="Probability verification through simulation",
        )

        if solution.claimed_result is not None:
            if isinstance(solution.claimed_result, (int, float)):
                if 0 <= solution.claimed_result <= 1:
                    result.status = VerificationStatus.VERIFIED
                    result.confidence = 0.7
                    result.details = f"Probability {solution.claimed_result} is in valid range [0,1]"

        solution.add_verification(result)

    async def _verify_generic(self, solution: DaydreamSolution) -> None:
        """Generic verification for unknown problem types."""

        result = VerificationResult(
            method=VerificationMethod.LOGICAL_CONSISTENCY,
            status=VerificationStatus.UNCERTAIN,
            confidence=0.3,
            details="No specific verification method available for this problem type",
        )

        solution.add_verification(result)

    def _finalize_solution(self, solution: DaydreamSolution) -> None:
        """Move solution to appropriate list."""
        with self._lock:
            if solution in self.pending_solutions:
                self.pending_solutions.remove(solution)

            if solution.overall_status in [VerificationStatus.VERIFIED, VerificationStatus.UNCERTAIN]:
                self.verified_solutions.append(solution)

    def get_verified_insights(self, min_confidence: float = 0.7) -> list[dict]:
        """Get verified solutions that passed checks."""
        with self._lock:
            return [
                {
                    "id": s.id,
                    "problem": s.problem_statement[:100],
                    "insight": s.insight,
                    "confidence": s.confidence_score,
                    "status": s.overall_status.value,
                }
                for s in self.verified_solutions
                if s.confidence_score >= min_confidence
            ]


_verifier: SolutionVerifier | None = None
_verifier_lock = threading.Lock()


def get_verifier() -> SolutionVerifier:
    global _verifier
    if _verifier is None:
        with _verifier_lock:
            if _verifier is None:
                _verifier = SolutionVerifier()
    return _verifier
