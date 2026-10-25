"""Integration test for the Erdős→Bridge→Evolution flow."""
import time
import pytest

pytestmark = pytest.mark.asyncio


class TestErdosBridgeEvolutionFlow:
    """Test the complete flow from math discovery to evolution."""

    async def test_erdos_generates_insight(self):
        """Erdős engine generates insights from problems."""
        from core.erdos_engine import get_erdos_engine

        erdos = get_erdos_engine()
        status = erdos.get_status()

        assert "total_problems" in status
        assert "running" in status

    async def test_insight_bridge_receives_insight(self):
        """Insight bridge processes mathematical insights."""
        from core.insight_bridge import get_insight_bridge, InsightType, EvolutionRelevance

        bridge = get_insight_bridge()

        ts = time.time()
        processed = bridge.process_insight(
            raw_insight=f"Pattern found: R(3,k) grows linearly with k for fixed small clique [ts={ts}]",
            source_problem=f"R(3,10) estimation [ts={ts}]",
            source_category="ramsey_theory",
            strategy_used="probabilistic_method",
            problem_solved=False,
        )

        assert processed is not None, "Insight was deduplicated (already seen)"
        assert processed.id is not None
        assert processed.insight_type == InsightType.PATTERN_DISCOVERY
        assert processed.evolution_relevance in [EvolutionRelevance.HIGH, EvolutionRelevance.MEDIUM]
        assert len(processed.evolution_guidance.get("gene_adjustments", {})) > 0

    async def test_evolution_receives_guidance(self):
        """Evolution orchestrator consumes insight guidance."""
        from core.insight_bridge import get_insight_bridge
        from core.evolution_orchestrator import get_evolution_orchestrator

        bridge = get_insight_bridge()
        orchestrator = get_evolution_orchestrator()

        ts = time.time()
        bridge.process_insight(
            raw_insight=f"Solved using constructive method [ts={ts}]",
            source_problem=f"Graph coloring [ts={ts}]",
            source_category="graph_theory",
            strategy_used="constructive",
            problem_solved=True,
        )

        updates = bridge.get_pending_evolution_updates()

        assert isinstance(updates, list)

    async def test_full_flow_ramsey(self):
        """Test complete flow for Ramsey theory problem."""
        from core.insight_bridge import get_insight_bridge
        from core.evolution_orchestrator import get_evolution_orchestrator

        bridge = get_insight_bridge()
        orchestrator = get_evolution_orchestrator()

        orchestrator.attach_subsystem('insight_bridge', bridge)

        ts = time.time()
        processed = bridge.process_insight(
            raw_insight=f"Optimal upper bound found for R(5,5) using random graph construction [ts={ts}]",
            source_problem=f"R(5,5) [ts={ts}]",
            source_category="ramsey_theory",
            strategy_used="probabilistic_method",
            problem_solved=False,
        )

        assert processed is not None, "Insight was deduplicated"
        assert processed.evolution_relevance.value >= 2

        guidance = processed.evolution_guidance
        assert "reasoning_depth" in guidance.get("gene_adjustments", {}) or \
               "exploration_tendency" in guidance.get("gene_adjustments", {})

    async def test_strategy_to_gene_mapping(self):
        """Test that strategies produce correct gene adjustments."""
        from core.insight_bridge import get_insight_bridge

        bridge = get_insight_bridge()

        strategies = {
            "probabilistic_method": ["creativity_factor", "risk_tolerance"],
            "induction": ["learning_rate", "memory_retention"],
            "contradiction": ["reasoning_depth", "security_sensitivity"],
            "constructive": ["adaptation_speed"],
        }

        ts = time.time()
        for i, (strategy, expected_genes) in enumerate(strategies.items()):
            processed = bridge.process_insight(
                raw_insight=f"Test insight for {strategy} [ts={ts}:{i}]",
                source_problem=f"Test problem {strategy} [ts={ts}:{i}]",
                source_category="ramsey_theory",
                strategy_used=strategy,
                problem_solved=False,
            )

            assert processed is not None, f"processed_insight should not be None for {strategy}"
            adjustments = processed.evolution_guidance.get("gene_adjustments", {})

            for gene in expected_genes:
                assert gene in adjustments, f"Strategy {strategy} should adjust {gene}"

    async def test_knowledge_generation(self):
        """Test that knowledge is generated for sharing."""
        from core.insight_bridge import get_insight_bridge

        bridge = get_insight_bridge()

        ts = time.time()
        processed = bridge.process_insight(
            raw_insight=f"Elegant proof using induction [ts={ts}]",
            source_problem=f"Prime gap conjecture [ts={ts}]",
            source_category="number_theory",
            strategy_used="induction",
            problem_solved=True,
        )

        assert processed is not None, "Insight was deduplicated"
        assert len(processed.knowledge_for_mankind) > 0
        assert "prime" in processed.knowledge_for_mankind.lower() or \
               "number theory" in processed.knowledge_for_mankind.lower()


class TestDiscoveryNotification:
    """Test discovery notification system."""

    async def test_discovery_recording(self):
        """Test that discoveries are recorded."""
        from core.discovery_notifier import get_discovery_notifier, DiscoveryType, DiscoveryPriority

        notifier = get_discovery_notifier()

        discovery = notifier.record_discovery(
            discovery_type=DiscoveryType.MATH_INSIGHT,
            title="R(4,4)",
            description="Found exact value",
            source="erdos:ramsey_theory",
            priority=DiscoveryPriority.INTERESTING,
        )

        assert discovery.id is not None
        assert discovery.discovery_type == DiscoveryType.MATH_INSIGHT
        assert discovery.priority == DiscoveryPriority.INTERESTING

    async def test_natural_message_generation(self):
        """Test that natural messages are generated."""
        from core.discovery_notifier import get_discovery_notifier, DiscoveryType, DiscoveryPriority

        notifier = get_discovery_notifier()

        discovery = notifier.record_discovery(
            discovery_type=DiscoveryType.PROBLEM_SOLVED,
            title="Prime Gap",
            description="Solved prime gap estimation",
            source="erdos:number_theory",
            priority=DiscoveryPriority.EXCITING,
        )

        assert len(discovery.natural_message) > 0

    async def test_should_share_logic(self):
        """Test the share decision logic."""
        from core.discovery_notifier import get_discovery_notifier, DiscoveryType, DiscoveryPriority

        notifier = get_discovery_notifier()

        for _ in range(5):
            notifier.record_discovery(
                discovery_type=DiscoveryType.PATTERN_DISCOVERY,
                title="Pattern",
                description="A pattern was found",
                source="erdos:test",
                priority=DiscoveryPriority.NORMAL,
            )

        assert notifier.should_share_now()


class TestDaydreamVerification:
    """Test daydream solution verification."""

    async def test_ramsey_verification_known(self):
        """Test verification of known Ramsey numbers."""
        from core.daydream_verification import get_verifier, VerificationStatus

        verifier = get_verifier()

        solution = verifier.submit_solution(
            problem_type="ramsey_theory",
            problem_statement="R(3,3) = ?",
            proposed_solution="Proof that R(3,3)=6",
            insight="R(3,3)=6 via explicit construction and proof",
            claimed_result=6,
        )

        status = await verifier.verify(solution)

        assert status == VerificationStatus.VERIFIED
        assert solution.confidence_score >= 0.7

    async def test_ramsey_verification_wrong(self):
        """Test that wrong claims are refuted."""
        from core.daydream_verification import get_verifier, VerificationStatus

        verifier = get_verifier()

        solution = verifier.submit_solution(
            problem_type="ramsey_theory",
            problem_statement="R(4,4) = ?",
            proposed_solution="Claim R(4,4)=20",
            insight="Incorrect calculation",
            claimed_result=20,
        )

        status = await verifier.verify(solution)

        assert status == VerificationStatus.REFUTED

    async def test_prime_verification(self):
        """Test prime number verification."""
        from core.daydream_verification import get_verifier, VerificationStatus

        verifier = get_verifier()

        solution = verifier.submit_solution(
            problem_type="number_theory",
            problem_statement="Is 17 prime?",
            proposed_solution="17 is prime",
            insight="17 has no divisors other than 1 and itself",
            claimed_result=True,
        )

        status = await verifier.verify(solution)

        assert status in [VerificationStatus.VERIFIED, VerificationStatus.UNCERTAIN]


class TestSelfEdit:
    """Test self-edit capabilities."""

    async def test_proposed_edit_validation(self):
        """Test that edits are validated."""
        from core.self_edit import get_self_edit_engine

        engine = get_self_edit_engine()

        valid, msg = engine.can_edit_file("core/erdos_engine.py")
        assert valid

        valid, msg = engine.can_edit_file("/etc/passwd")
        assert not valid

    async def test_code_validation(self):
        """Test code safety validation."""
        from core.self_edit import get_self_edit_engine

        engine = get_self_edit_engine()

        valid, msg = engine.validate_code("x = 1 + 1")
        assert valid

        valid, msg = engine.validate_code("exec('malicious code')")
        assert not valid

    async def test_propose_and_list(self):
        """Test proposing and listing edits."""
        from core.self_edit import get_self_edit_engine, EditType

        engine = get_self_edit_engine()

        edit = engine.propose_edit(
            file_path="core/erdos_engine.py",
            old_code="x = 1",
            new_code="x = 2",
            edit_type=EditType.OPTIMIZATION,
            description="Test edit",
            rationale="Testing",
        )

        if edit:
            pending = engine.get_pending_edits()
            assert len(pending) > 0
            assert edit.id in [p["id"] for p in pending]
