"""Regression tests for routing background subsystems through the api_manager.

The audit item (NEXT_STEPS "API manager bypass") was cost exposure: background
subsystems called the LLM client directly, so the api_manager's per-priority
quotas, caching and queue backpressure never applied to them. These tests pin
the routing contract for the three modules migrated in this cycle:

- core/self_evaluation.py  (_llm_evaluate, _enhance_reflection_with_llm)
- core/growth.py           (_get_llm_suggestions)
- core/self_correction.py  (_decompose_with_llm / decompose_goal_async)

They also pin the sync/async decomposition fix: sync decompose_goal with an
async client used to leak an un-awaited coroutine and silently no-op.

All tests are offline: the api_manager is a stub and the LLM clients are
in-process fakes. The manager function is monkeypatched per test so the
process-wide singleton is never created or consulted.
"""
import asyncio
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeAsyncLLM:
    """Minimal async client shaped like OllamaClient (chat with kwargs)."""

    def __init__(self, content='{"scores": {}}'):
        self.content = content
        self.chat_calls: list[dict] = []

    async def chat(self, user_input: str = "", system_prompt: str = "", **kwargs):
        self.chat_calls.append({"user_input": user_input, "system_prompt": system_prompt})
        return {"message": {"content": self.content}}


class FakeManager:
    """Stands in for APICallManager, recording submit() calls."""

    def __init__(self, response):
        self.response = response
        self.submit_calls: list[dict] = []

    async def submit(self, **kwargs):
        self.submit_calls.append(kwargs)
        return self.response


def _patch_manager(monkeypatch, manager):
    monkeypatch.setattr("core.api_manager.get_api_manager", lambda llm_client=None: manager)


def _patch_no_manager(monkeypatch):
    """Simulate the manager being uninitializable (fresh process, no boot)."""
    def boom(llm_client=None):
        raise ValueError("LLM client required for first initialization")
    monkeypatch.setattr("core.api_manager.get_api_manager", boom)


def _api_response(content=None, success=True, error=None):
    from core.api_manager import APIResponse
    return APIResponse(
        request_id="test_req",
        success=success,
        content=content,
        error=error,
    )


# =========================================================================
# self_correction: sync/async decomposition fix
# =========================================================================

class TestSelfCorrectionDecomposition:
    def _engine(self, llm):
        from core.self_correction import SelfCorrectionEngine
        return SelfCorrectionEngine(
            storage_path=Path(tempfile.mkdtemp()), llm_client=llm,
        )

    def test_sync_decompose_goal_with_async_client_is_a_clean_noop(self):
        """Regression: sync decompose_goal called the async client's chat(),
        fed the un-awaited coroutine into re.search (TypeError), and the
        broad except swallowed it — LLM decomposition silently never worked
        and the coroutine leaked. It must now skip cleanly."""
        engine = self._engine(FakeAsyncLLM(content='[{"name": "s1"}]'))
        plan = engine.decompose_goal("analyze and test the code")
        assert len(plan.steps) > 0
        # The async LLM was never consumed from the sync path.
        assert engine.llm.chat_calls == []

    def test_sync_decompose_with_async_client_closes_the_coroutine(self):
        """The skipped coroutine must be closed, not left to GC with a
        'coroutine was never awaited' RuntimeWarning."""
        engine = self._engine(FakeAsyncLLM())

        class ChatOnlyAsync:
            def chat(self, *a, **kw):
                async def _c():
                    return {"message": {"content": "[]"}}
                return _c()

        engine.llm = ChatOnlyAsync()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = engine._decompose_with_llm_sync("goal", None)
        assert result is None

    def test_async_decompose_goal_uses_llm_via_direct_fallback(self, monkeypatch):
        """Without a manager, decompose_goal_async must actually await the
        client — the fix that makes LLM decomposition work at all."""
        engine = self._engine(FakeAsyncLLM(content='[{"name": "step_a", "description": "d"}]'))
        _patch_no_manager(monkeypatch)

        plan = asyncio.run(engine.decompose_goal_async("run the tests"))

        assert engine.llm.chat_calls, "LLM was never called on the async path"
        names = [s.name for s in plan.steps]
        assert "step_a" in names

    def test_async_decompose_goal_routes_through_manager(self, monkeypatch):
        manager = FakeManager(_api_response(
            content={"message": {"content": '[{"name": "mgr_step"}]'}}
        ))
        engine = self._engine(FakeAsyncLLM())
        _patch_manager(monkeypatch, manager)

        plan = asyncio.run(engine.decompose_goal_async("run the tests"))

        assert manager.submit_calls, "decomposition bypassed the api_manager"
        call = manager.submit_calls[0]
        assert call["subsystem"] == "self_correction"
        from core.api_manager import Priority
        assert call["priority"] == Priority.LEARNING
        names = [s.name for s in plan.steps]
        assert "mgr_step" in names

    def test_async_decompose_goal_manager_refusal_falls_back_to_heuristics(self, monkeypatch):
        """Rate-limit refusal must degrade to heuristic steps, not bypass."""
        manager = FakeManager(_api_response(success=False, error="Rate limit exceeded"))
        engine = self._engine(FakeAsyncLLM())
        _patch_manager(monkeypatch, manager)

        plan = asyncio.run(engine.decompose_goal_async("analyze and test the code"))

        assert manager.submit_calls
        assert engine.llm.chat_calls == [], "refused call must not bypass to direct"
        assert len(plan.steps) > 0  # heuristic fallback

    def test_execute_with_recovery_uses_async_decomposition(self, monkeypatch):
        """The production entry point must take the LLM-capable async path."""
        engine = self._engine(FakeAsyncLLM(content='[{"name": "s1", "action": "execute"}]'))
        _patch_no_manager(monkeypatch)
        engine.register_handler("execute", lambda params, context: {"output": "done"})

        seen = {}
        original_async = engine.decompose_goal_async

        async def spy(goal, context=None, **kw):
            seen["goal"] = goal
            return await original_async(goal, context, **kw)

        engine.decompose_goal_async = spy
        ok, result = asyncio.run(engine.execute_with_recovery("run the tests"))

        assert seen["goal"] == "run the tests"
        assert engine.llm.chat_calls, "LLM decomposition never ran"
        assert ok is True


# =========================================================================
# self_evaluation: coordinated evaluation
# =========================================================================

class TestSelfEvaluationCoordination:
    def _engine(self, llm):
        from core.self_evaluation import SelfEvaluationEngine
        return SelfEvaluationEngine(llm_client=llm, storage_path=Path(tempfile.mkdtemp()))

    @staticmethod
    def _record(engine):
        from core.self_evaluation import ActionRecord, ActionType, EthicalPrinciple
        return ActionRecord(
            id="rec_1",
            timestamp=datetime.now(),
            action_type=ActionType.TOOL_USE,
            description="test action",
            input_preview="in",
            output_preview="out",
            subsystem="test",
            principles_evaluated={EthicalPrinciple.HONESTY: 1.0},
            overall_score=0.8,
            flags=[],
        )

    def test_llm_evaluate_routes_through_manager(self, monkeypatch):
        manager = FakeManager(_api_response(
            content={"message": {"content": '{"scores": {}, "overall_score": 0.9, "flags": []}'}}
        ))
        engine = self._engine(FakeAsyncLLM())
        _patch_manager(monkeypatch, manager)

        result = asyncio.run(engine._llm_evaluate(self._record(engine)))

        assert manager.submit_calls
        assert manager.submit_calls[0]["subsystem"] == "self_evaluation"
        assert result["overall_score"] == 0.9

    def test_llm_evaluate_degrades_to_default_on_rate_limit(self, monkeypatch):
        """Quota exhausted -> neutral default scores, never a bypass."""
        manager = FakeManager(_api_response(success=False, error="Rate limit exceeded"))
        engine = self._engine(FakeAsyncLLM())
        _patch_manager(monkeypatch, manager)

        result = asyncio.run(engine._llm_evaluate(self._record(engine)))

        assert result == {"scores": {}, "overall_score": 0.8, "flags": []}
        assert engine.llm.chat_calls == []

    def test_llm_evaluate_direct_fallback_without_manager(self, monkeypatch):
        engine = self._engine(FakeAsyncLLM(content='{"overall_score": 0.55}'))
        _patch_no_manager(monkeypatch)

        result = asyncio.run(engine._llm_evaluate(self._record(engine)))

        assert engine.llm.chat_calls, "direct fallback never invoked the client"
        assert result["overall_score"] == 0.55

    def test_coordinated_chat_without_llm_returns_none(self):
        engine = self._engine(None)
        assert asyncio.run(engine._coordinated_chat("x", "y")) is None


# =========================================================================
# growth: coordinated suggestions
# =========================================================================

class TestGrowthCoordination:
    def _engine(self):
        from core.growth import SelfGrowthEngine
        return SelfGrowthEngine(storage_path=Path(tempfile.mkdtemp()))

    def test_llm_suggestions_route_through_manager(self, monkeypatch):
        manager = FakeManager(_api_response(
            content={"message": {"content": '[{"title": "T", "description": "D", "rationale": "R"}]'}}
        ))
        engine = self._engine()
        llm = FakeAsyncLLM()
        _patch_manager(monkeypatch, manager)

        proposals = asyncio.run(engine._get_llm_suggestions(llm))

        assert manager.submit_calls
        assert manager.submit_calls[0]["subsystem"] == "growth"
        assert proposals and proposals[0].title == "T"

    def test_llm_suggestions_empty_on_rate_limit(self, monkeypatch):
        manager = FakeManager(_api_response(success=False, error="Rate limit exceeded"))
        engine = self._engine()
        llm = FakeAsyncLLM()
        _patch_manager(monkeypatch, manager)

        proposals = asyncio.run(engine._get_llm_suggestions(llm))

        assert proposals == []
        assert llm.chat_calls == [], "refused call must not bypass to direct"

    def test_llm_suggestions_direct_fallback_without_manager(self, monkeypatch):
        engine = self._engine()
        llm = FakeAsyncLLM(content='[{"title": "FB", "description": "d", "rationale": "r"}]')
        _patch_no_manager(monkeypatch)

        proposals = asyncio.run(engine._get_llm_suggestions(llm))

        assert llm.chat_calls
        assert proposals and proposals[0].title == "FB"

    def test_llm_suggestions_without_client_returns_empty(self):
        engine = self._engine()
        proposals = asyncio.run(engine._get_llm_suggestions(None))
        assert proposals == []