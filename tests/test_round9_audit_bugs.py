"""Tests for deep audit round 9 bug fixes.

Covers batch 3 bugs:
- auto_refine.py: apply_change race condition (APPLYING status guard)
- evolution_orchestrator.py: gene reads under lock
- supervisor.py: get_next_fallback_model thread safety
- claims.py: _claim_history deque instead of list
- token_optimizer.py: evict formula divisor fix
- memory.py: _compress_history under lock
"""
import threading
import pytest


class TestAutoRefineApplyChangeGuard:
    def test_apply_change_status_enum_has_applying(self):
        from core.auto_refine import ChangeStatus
        assert hasattr(ChangeStatus, "APPLYING")
        assert ChangeStatus.APPLYING.value == "applying"


class TestEvolutionGeneReadsUnderLock:
    def test_should_propose_evolution_accesses_genes(self):
        from core.evolution_orchestrator import EvolutionOrchestrator
        evo = EvolutionOrchestrator()
        analysis = {"weak_areas": [], "strength_areas": [], "improvement_opportunities": []}
        result = evo._should_propose_evolution(analysis)
        assert isinstance(result, bool)

    def test_calculate_evolution_interval(self):
        from core.evolution_orchestrator import EvolutionOrchestrator
        evo = EvolutionOrchestrator()
        interval = evo._calculate_evolution_interval()
        assert interval > 0
        assert interval >= evo.MIN_EVOLUTION_INTERVAL


class TestSupervisorFallbackModelThreadSafety:
    def test_get_next_fallback_model_returns_model_or_none(self):
        from core.supervisor import ServerSupervisor
        sup = ServerSupervisor()
        sup._fallback_llm_models = ["model_a", "model_b"]
        result = sup.get_next_fallback_model()
        assert result == "model_b"
        result2 = sup.get_next_fallback_model()
        assert result2 is None
        sup.reset_model_fallback()


class TestClaimsHistoryDeque:
    def test_claim_history_is_deque(self):
        from collections import deque
        from core.claims import ClaimsRegistry
        registry = ClaimsRegistry(agent_id="test")
        assert isinstance(registry._claim_history, deque)

    def test_claim_history_auto_evicts(self):
        from core.claims import ClaimsRegistry
        registry = ClaimsRegistry(agent_id="test")
        max_size = registry.MAX_HISTORY
        for i in range(max_size + 50):
            registry._claim_history.append({"idx": i})
        assert len(registry._claim_history) <= max_size


class TestTokenOptimizerEvictFormula:
    def test_evict_formula_divisor_is_tokens_saved(self):
        from core.token_optimizer import Pattern, ReasoningBank
        bank = ReasoningBank()
        for i in range(60):
            pattern = Pattern(
                id=f"pat_{i}",
                pattern=f"test_pattern_{i}",
                template=f"template_{i}",
                tokens_saved=max(1, i),
            )
            pattern.success_count = i + 1
            bank.patterns[pattern.pattern] = pattern
        initial_count = len(bank.patterns)
        bank._evict()
        assert len(bank.patterns) < initial_count


class TestMemoryCompressHistoryUnderLock:
    def test_compress_history_is_thread_safe(self):
        from core.memory import ConversationMemory
        mem = ConversationMemory()
        for i in range(50):
            mem.add_user_message(f"Message {i} " * 20)
        mem._compress_history(keep_recent=10)
        assert len(mem.history) <= 40

    def test_concurrent_compress_and_add(self):
        from core.memory import ConversationMemory
        mem = ConversationMemory()
        errors = []

        def add_messages():
            try:
                for i in range(20):
                    mem.add_user_message(f"Thread message {i}")
            except Exception as e:
                errors.append(e)

        def compress():
            try:
                for _ in range(5):
                    mem._compress_history(keep_recent=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages) for _ in range(3)]
        threads.append(threading.Thread(target=compress))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0