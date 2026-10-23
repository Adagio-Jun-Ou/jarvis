"""Regression tests for the api_manager refusal contract.

Defect class: submit_chat() converted refusals (rate limit / queue full /
timeout / LLM error) into a response shaped exactly like a successful one
— ``{"message": {"content": "Rate limit exceeded"}}`` — so every module
that extracted content got the error *string as if the model had said it*.
Erdos would file "Rate limit exceeded" as a mathematical insight; the
meta engine would reflect on it; auto-refine would treat it as code
analysis.

The contract now:
- submit_chat marks refusals with ``_refused: True`` (successes with
  ``_refused: False``)
- get_response_content()/get_response_thinking() — the single chokepoint
  all 23 extraction sites in the 8 partially-adopted modules use —
  extract *default* from a refused response, so callers degrade to their
  neutral path instead of processing the error text
- a refusal never becomes a fake answer, and callers never re-call the
  LLM directly (which would bypass the quota the refusal enforces)
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _manager_refusing():
    """APICallManager wired to a submit() that always refuses."""
    from core.api_manager import APICallManager, APIResponse, Priority

    mgr = APICallManager.__new__(APICallManager)  # skip __init__ (background loops)
    mgr.submit = AsyncMock(return_value=APIResponse(
        request_id="t1", success=False, content=None, error="Rate limit exceeded"))

    async def _submit_chat(**kwargs):
        return await APICallManager.submit_chat(mgr, **kwargs)

    return mgr, _submit_chat


class TestSubmitChatRefusalFlag:
    @pytest.mark.asyncio
    async def test_refusal_marked(self):
        from core.api_manager import Priority

        mgr, submit_chat = _manager_refusing()
        result = await submit_chat(
            subsystem="test", priority=Priority.LEARNING,
            user_input="hi", system_prompt="")
        assert result["_refused"] is True
        assert result["message"]["content"] == "Rate limit exceeded"
        assert result["_from_cache"] is False

    @pytest.mark.asyncio
    async def test_success_marked_not_refused(self):
        from core.api_manager import APICallManager, APIResponse, Priority

        mgr = APICallManager.__new__(APICallManager)
        mgr.submit = AsyncMock(return_value=APIResponse(
            request_id="t2", success=True,
            content={"message": {"content": "real answer"}}))

        result = await APICallManager.submit_chat(
            mgr, subsystem="test", priority=Priority.LEARNING,
            user_input="hi", system_prompt="")
        assert result["_refused"] is False
        assert result["_from_cache"] is False
        assert result["message"]["content"] == "real answer"


class TestChokepointExtraction:
    def test_refused_response_extracts_default(self):
        from core.helpers import get_response_content

        refused = {"message": {"content": "Rate limit exceeded"},
                   "_refused": True, "_from_cache": False}
        assert get_response_content(refused) == ""
        assert get_response_content(refused, default="fallback") == "fallback"

    def test_refused_thinking_extracts_default(self):
        from core.helpers import get_response_thinking

        refused = {"message": {"content": "Request queue full"},
                   "_refused": True}
        assert get_response_thinking(refused) == ""
        assert get_response_thinking(refused, default="none") == "none"

    def test_genuine_response_extracts_normally(self):
        from core.helpers import get_response_content, get_response_thinking

        ok = {"message": {"content": "The answer is 42.", "thinking": "hmm"},
              "_refused": False}
        assert get_response_content(ok) == "The answer is 42."
        assert get_response_thinking(ok) == "hmm"

    def test_legacy_response_without_flag_untouched(self):
        """Responses from other paths (no _refused key) behave as before."""
        from core.helpers import get_response_content

        legacy = {"message": {"content": "plain"}}
        assert get_response_content(legacy) == "plain"

    def test_refusal_does_not_look_like_insight(self):
        """The exact incident shape: erdos-style consumers must NOT be able
        to turn a refusal into an 'insight'."""
        from core.helpers import get_response_content

        refusal = {"message": {"content": "Rate limit exceeded"},
                   "_refused": True, "_from_cache": False}
        content = get_response_content(refusal)
        assert content == ""  # no insight text to mine


class TestAdoptedModulesDegrade:
    def test_self_evaluation_coordinated_chat_degrades_on_refusal(self):
        """The reference adopter: refusal -> None (neutral default), never
        a direct LLM bypass."""
        import asyncio

        from core.self_evaluation import SelfEvaluationEngine

        engine = SelfEvaluationEngine.__new__(SelfEvaluationEngine)
        engine.llm = MagicMock()
        engine._subsystem_name = "self_evaluation"

        refused = {"message": {"content": "Rate limit exceeded"},
                   "_refused": True}
        # Simulate the manager refusing
        from core.api_manager import APIResponse
        engine_refusal = APIResponse(
            request_id="x", success=False, content=None, error="Rate limit exceeded")

        import core.self_evaluation as se
        from core.api_manager import get_api_manager, Priority as P, RequestType

        original = get_api_manager

        class FakeManager:
            async def submit(self, **kw):
                return engine_refusal

        import core.api_manager as am
        am.get_api_manager = lambda llm: FakeManager()
        try:
            result = asyncio.new_event_loop().run_until_complete(
                engine._coordinated_chat("input", "prompt"))
        finally:
            am.get_api_manager = original
        assert result is None  # degrade, do not bypass
