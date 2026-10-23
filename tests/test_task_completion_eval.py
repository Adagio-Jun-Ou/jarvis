"""Regression tests for the agentic-loop task-completion evaluation.

Two defects made _evaluate_task_completion — the decision point of the
whole agentic loop — silently dead, so every tool-using turn ended after
the first round of tools no matter how incomplete the work was:

1. The call passed ``stream=False`` to OllamaClient.chat, which has no
   ``stream`` parameter → TypeError → swallowed by the broad except →
   ``return True, "complete"``.
2. Even without (1), the response shape check was ``"content" in
   response``, but OllamaClient.chat returns the raw Ollama JSON
   (``{"message": {"content": ...}}``) with no top-level "content" key →
   always "No response from backend_llm" → complete.

Either one alone disabled the CONTINUE path. The logs showed only the
misleading "[AGENT] No response from backend_llm" warning.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _agent_with_backend(reply_content: str):
    """JarvisAgent stand-in with a fake OllamaClient-shaped backend_llm."""
    from core.agent import JarvisAgent

    agent = JarvisAgent.__new__(JarvisAgent)  # skip heavy __init__
    agent._recursive_self_model = None
    agent.backend_llm = AsyncMock()
    # OllamaClient.chat returns the RAW Ollama response shape:
    agent.backend_llm.chat = AsyncMock(return_value={
        "model": "glm-5",
        "created_at": "2026-08-31T00:00:00Z",
        "message": {"role": "assistant", "content": reply_content},
        "done": True,
    })
    return agent


class TestEvaluateTaskCompletion:
    def test_complete_response(self):
        agent = _agent_with_backend("COMPLETE")
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("list files", "Tool: list_files\nResult: a.txt, b.txt")
        )
        assert is_complete is True
        # No phantom kwarg: chat called with valid OllamaClient.chat params only
        kwargs = agent.backend_llm.chat.call_args.kwargs
        assert "stream" not in kwargs
        assert "user_input" in kwargs

    def test_continue_response_yields_next_action(self):
        agent = _agent_with_backend("CONTINUE: analyze the fetched data")
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion(
                "fetch and analyze data", "Tool: web_fetch\nResult: raw csv...")
        )
        assert is_complete is False
        assert "analyze the fetched data" in next_action
        kwargs = agent.backend_llm.chat.call_args.kwargs
        assert "stream" not in kwargs

    def test_bare_continue_gets_default_action(self):
        agent = _agent_with_backend("CONTINUE")
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("do the thing", "partial results")
        )
        assert is_complete is False
        assert next_action == "continue with the next logical step"

    def test_prose_answer_with_done_word(self):
        agent = _agent_with_backend("The task is done and all results are shown.")
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("x", "y")
        )
        assert is_complete is True

    def test_prose_incomplete_answer(self):
        agent = _agent_with_backend("The results are incomplete; more work is needed.")
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("x", "y")
        )
        assert is_complete is False
        assert next_action

    def test_no_backend_assumes_complete(self):
        from core.agent import JarvisAgent

        agent = JarvisAgent.__new__(JarvisAgent)
        agent.backend_llm = None
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("x", "y")
        )
        assert is_complete is True

    def test_error_defaults_to_complete_no_infinite_loop(self):
        agent = _agent_with_backend("whatever")
        agent.backend_llm.chat = AsyncMock(side_effect=RuntimeError("boom"))
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("x", "y")
        )
        assert is_complete is True

    def test_refused_evaluation_defaults_complete(self):
        """A refused api_manager-shaped response (empty content) must not
        be treated as an answer."""
        agent = _agent_with_backend("")
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("x", "y")
        )
        assert is_complete is True


class _PoorMetacogModel:
    """Stand-in RecursiveSelfModel whose type-2 metrics say confidence
    and accuracy diverge (index < 0)."""

    def compute_metacognition(self):
        return {"calibration": 0.2, "discrimination": -0.8,
                "index": -0.4, "n": 50, "reliable": True}


class TestMetacognitionGate:
    def test_poor_metacog_blocks_complete_with_errors(self):
        """When the self-model reliably tracks confidence/accuracy
        divergence, a COMPLETE verdict on failure-laden context is
        downgraded to a verification step."""
        agent = _agent_with_backend("COMPLETE")
        agent._recursive_self_model = _PoorMetacogModel()
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion(
                "list files", "Tool: list_files failed\nResult: error")
        )
        assert is_complete is False
        assert "verify" in next_action.lower()

    def test_poor_metacog_passes_clean_context(self):
        """The gate only fires when the context shows failures — a clean
        COMPLETE with poor metacognition still completes."""
        agent = _agent_with_backend("COMPLETE")
        agent._recursive_self_model = _PoorMetacogModel()
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion(
                "list files", "Tool: list_files\nResult: a.txt, b.txt")
        )
        assert is_complete is True

    def test_unreliable_metacog_never_gates(self):
        """Below the reliability threshold the gate is inert — no
        double-checking based on noise."""
        agent = _agent_with_backend("COMPLETE")

        class _UnreliableModel:
            def compute_metacognition(self):
                return {"calibration": None, "discrimination": None,
                        "index": None, "n": 5, "reliable": False}

        agent._recursive_self_model = _UnreliableModel()
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("x", "Tool failed: error")
        )
        assert is_complete is True


class TestOllamaShapeParity:
    def test_real_ollama_client_has_no_stream_param(self):
        """Pin the API shape: OllamaClient.chat must NOT accept stream=.
        If someone adds it, this test documents the contract change."""
        import inspect

        from core.llm import OllamaClient

        sig = inspect.signature(OllamaClient.chat)
        assert "stream" not in sig.parameters
        assert "user_input" in sig.parameters
        assert "system_prompt" in sig.parameters

    def test_get_response_content_extracts_ollama_shape(self):
        from core.helpers import get_response_content

        ollama_response = {
            "model": "glm-5",
            "message": {"role": "assistant", "content": "CONTINUE: run tests"},
        }
        assert get_response_content(ollama_response) == "CONTINUE: run tests"