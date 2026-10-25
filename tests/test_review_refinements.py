"""Regression tests for the memory-hub embedding honesty and the adaptive
loop's unified terminal permission gate.

Both fixes close divergent-behaviour holes found during the full-system
review + external research pass (Anthropic's "Building effective agents":
one safety chokepoint, poka-yoke tools, ground truth over confident
nonsense):

1. MemoryHub.encode() used to fabricate a *random* 768-dim vector when the
   embedding interconnect was unavailable, cache it like a real one, and let
   every retrieval score become plausible-looking noise. It must now return
   None, and conversation re-ranking must degrade to deterministic lexical
   (word-overlap) scoring instead.

2. AdaptiveAgentLoop._execute_shell_tool re-implemented only the HIGH-risk
   block of the terminal policy; MEDIUM commands (package installs, docker
   ops, script execution) ran ungated on this path while being gated on the
   tool path. The gate must now consult the terminal tool's own
   needs_permission() (single policy source) and the agent's
   PermissionManager (same contract as execute_tool): DENY refuses, no
   stored grant requires the user's approval, a stored allow proceeds.

All tests are offline: the embedder and permission stores are fakes.
"""
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =========================================================================
# MemoryHub: no fabricated embeddings
# =========================================================================

class TestMemoryHubEmbeddingHonesty:
    @staticmethod
    def _hub():
        from core.memory_hub import MemoryHub

        hub = MemoryHub()
        # Force "embedder unavailable".
        hub._embedding_interconnect = SimpleNamespace(encode=lambda text: None)
        return hub

    def test_encode_returns_none_without_embedder(self):
        hub = self._hub()
        assert hub.encode("hello world") is None

    def test_none_is_never_cached(self):
        """A missing embedding must not poison the cache slot."""
        hub = self._hub()
        hub.encode("hello world")
        assert hub._embedding_cache == {}

    def test_real_embedding_still_cached_and_dimension_truncatable(self):
        import numpy as np

        from core.memory_hub import MemoryHub

        hub = MemoryHub()
        vec = np.ones(768, dtype=np.float32)
        hub._embedding_interconnect = SimpleNamespace(encode=lambda text: vec)

        full = hub.encode("hello world", dim=768)
        assert full is not None and full.shape == (768,)
        truncated = hub.encode("hello world", dim=256)
        assert truncated is not None and truncated.shape == (256,)

        # Cached: same object returned, no re-encode.
        again = hub.encode("hello world", dim=256)
        assert again is truncated

    def test_no_random_vector_is_ever_produced(self, monkeypatch):
        """Guard the fix: encode() must not call any RNG fallback.

        Note: setting `_embedding_interconnect = None` re-triggers the lazy
        loader, so "unavailable" is simulated with a stub returning None.
        """
        import numpy as np

        from core.memory_hub import MemoryHub

        def fail_random(*args, **kwargs):
            raise AssertionError("encode() fell back to a random vector")

        monkeypatch.setattr(np.random, "randn", fail_random)
        hub = MemoryHub()
        hub._embedding_interconnect = SimpleNamespace(encode=lambda text: None)
        assert hub.encode("anything") is None

    def test_conversation_retrieval_degrades_to_lexical(self):
        """With no embedder, ranking must follow word overlap, not noise.

        Two messages, same recency: the one sharing words with the query
        must score higher than the unrelated one.
        """
        from core.memory_hub import MemoryHub

        hub = MemoryHub()
        hub._embedding_interconnect = None

        now = datetime.now()
        conv = SimpleNamespace(messages=[
            {"role": "user", "content": "the python pytest runner failed", "timestamp": now},
            {"role": "user", "content": "completely unrelated dinner plans", "timestamp": now},
        ])
        hub._conversation_memory = conv

        results = hub._retrieve_from_conversation("python pytest", top_k=5)
        assert results, "lexical fallback returned nothing"
        scores = {r.metadata.get("role_index", i): r.score for i, r in enumerate(results)}
        # The matching message must outrank the unrelated one whenever both
        # survive the score threshold; at minimum it must be present.
        matching = [r for r in results if "pytest" in r.content]
        assert matching, "word-overlap match missing from lexical fallback"
        unrelated = [r for r in results if "dinner" in r.content]
        if unrelated:
            assert matching[0].score > unrelated[0].score

    def test_lexical_similarity_is_deterministic_and_bounded(self):
        from core.memory_hub import MemoryHub

        sim_same = MemoryHub._lexical_similarity(
            MemoryHub._lexical_tokens("python pytest runner"),
            "the python pytest runner failed",
        )
        sim_none = MemoryHub._lexical_similarity(
            MemoryHub._lexical_tokens("python pytest runner"),
            "completely unrelated dinner plans",
        )
        assert 0.0 <= sim_none < sim_same <= 1.0
        assert sim_none == 0.0

    def test_vector_retrieval_does_not_burn_a_dead_encode(self):
        """_retrieve_from_vector used to encode the query and discard the
        result — a wasted model call (and a cached poison vector in the old
        fallback days). It must not call encode() at all; search() uses
        VectorMemory's own embedder."""
        from core.memory_hub import MemoryHub

        hub = MemoryHub()

        def fail_encode(*args, **kwargs):
            raise AssertionError("_retrieve_from_vector called encode()")

        hub.encode = fail_encode
        hub._vector_memory = SimpleNamespace(
            search=lambda query, k: [("id1", 0.9, {"text": "found"})],
            count=lambda: 1,
        )
        results = hub._retrieve_from_vector("anything", top_k=3)
        assert results and results[0].content == "found"


# =========================================================================
# AdaptiveAgentLoop: unified terminal permission gate
# =========================================================================

class TestAdaptiveLoopPermissionGate:
    @staticmethod
    def _loop(agent_tools=None, permission_result="unset"):
        from core.adaptive_agent_loop import AdaptiveAgentLoop

        agent = MagicMock()
        if agent_tools is not None:
            agent.tools = agent_tools
        else:
            agent.tools = {}
        agent.permissions = SimpleNamespace(
            check_permission=lambda key: permission_result,
        )
        loop = AdaptiveAgentLoop(agent=agent)
        return loop

    @staticmethod
    def _terminal_tool(needs=True):
        """A terminal tool stand-in mirroring TerminalExecuteTool's contract."""
        return SimpleNamespace(needs_permission=lambda command="", **kw: needs)

    def test_low_risk_command_runs_ungated(self):
        loop = self._loop(
            agent_tools={"terminal_execute": self._terminal_tool(needs=False)},
            permission_result=None,  # no stored grant at all
        )
        result = loop._execute_shell_tool("terminal_execute", {"command": "ls -la"}, 10.0)
        # ls is low-risk: executes (success on most hosts) — never gated.
        assert result.get("error") != "PERMISSION_REQUIRED"
        assert result.get("error") != "Permission denied"

    def test_medium_risk_requires_permission_without_grant(self):
        """pip install: medium risk on the tool path — the adaptive loop must
        demand approval too, not run it ungated."""
        loop = self._loop(
            agent_tools={"terminal_execute": self._terminal_tool(needs=True)},
            permission_result=None,  # PermissionManager: nothing stored
        )
        result = loop._execute_shell_tool(
            "terminal_execute", {"command": "pip install requests"}, 10.0
        )
        assert result["success"] is False
        assert result["error"] == "PERMISSION_REQUIRED"
        assert "pip install requests" in result["output"]

    def test_medium_risk_denied_by_stored_preference(self):
        from core.permissions import PermissionLevel

        loop = self._loop(
            agent_tools={"terminal_execute": self._terminal_tool(needs=True)},
            permission_result=PermissionLevel.DENY,
        )
        result = loop._execute_shell_tool(
            "terminal_execute", {"command": "pip install requests"}, 10.0
        )
        assert result["success"] is False
        assert result["error"] == "Permission denied"
        assert result["exit_code"] == -1

    def test_medium_risk_proceeds_with_stored_allow(self):
        from core.permissions import PermissionLevel

        loop = self._loop(
            agent_tools={"terminal_execute": self._terminal_tool(needs=True)},
            permission_result=PermissionLevel.ALLOW_SESSION,
        )
        result = loop._execute_shell_tool(
            "terminal_execute", {"command": "pip install requests"}, 10.0
        )
        assert result.get("error") != "PERMISSION_REQUIRED"
        assert result.get("error") != "Permission denied"

    def test_high_risk_still_hard_blocked_before_any_gate(self):
        loop = self._loop(
            agent_tools={"terminal_execute": self._terminal_tool(needs=False)},
            permission_result="allow",  # even an explicit allow must not matter
        )
        result = loop._execute_shell_tool(
            "terminal_execute", {"command": "rm -rf /"}, 10.0
        )
        assert result["success"] is False
        assert result["error"].startswith("BLOCKED")

    def test_needs_permission_failure_gates_fail_closed(self):
        """A broken policy oracle must refuse, not wave through."""
        class BrokenTool:
            def needs_permission(self, command="", **kw):
                raise RuntimeError("policy oracle exploded")

        loop = self._loop(
            agent_tools={"terminal_execute": BrokenTool()},
            permission_result=None,
        )
        result = loop._execute_shell_tool(
            "terminal_execute", {"command": "pip install requests"}, 10.0
        )
        assert result["success"] is False
        assert result["error"] == "PERMISSION_REQUIRED"

    def test_no_terminal_tool_registered_still_executes_medium(self):
        """If the registry lacks the tool object (e.g. custom toolsets), the
        gate has no policy oracle; the pre-existing behaviour (risk-based
        blocking only) applies rather than a hard failure."""
        loop = self._loop(agent_tools={}, permission_result=None)
        result = loop._execute_shell_tool(
            "terminal_execute", {"command": "pip install requests"}, 10.0
        )
        assert result.get("error") != "PERMISSION_REQUIRED"

    def test_real_terminal_tool_matches_agent_path_decision(self):
        """The gate must agree with the tool object it delegates to: for the
        real TerminalExecuteTool, needs_permission(pip install) is True and
        needs_permission(ls) is False — the loop gate follows both."""
        from tools.terminal import TerminalExecuteTool

        tool = TerminalExecuteTool()
        loop = self._loop(
            agent_tools={"terminal_execute": tool},
            permission_result=None,
        )
        gated = loop._execute_shell_tool(
            "terminal_execute", {"command": "pip install requests"}, 10.0
        )
        assert gated["error"] == "PERMISSION_REQUIRED"

        free = loop._execute_shell_tool(
            "terminal_execute", {"command": "ls"}, 10.0
        )
        assert free.get("error") != "PERMISSION_REQUIRED"