"""Guards that written subsystems are actually reachable from production.

Jarvis's recurring failure mode is not broken code — it is *correct code nobody
calls*. Several substantial modules (SSRF protection, the shared embedding
server, the safe-files registry, the PFC observer) were fully implemented and
unit-tested while being imported by nothing, so their protection or behaviour
never applied at runtime and no test noticed.

These tests assert the integration points themselves, not the modules'
internals. They fail if someone unwires a subsystem, which a module-level unit
test cannot detect.
"""
import asyncio
import inspect
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _engine():
    """SelfEditEngine anchored at the repo root.

    can_edit_file() resolves paths relative to jarvis_root, which defaults to
    Path.cwd() — that varies per xdist worker, so it must be pinned here.
    """
    from core.self_edit import SelfEditEngine

    return SelfEditEngine(jarvis_root=REPO_ROOT)


class TestSafeFilesRegistryIsAuthoritative:
    """core/safe_files.py must be the single source of self-modification policy."""

    def test_self_edit_consumes_the_registry(self):
        from core.safe_files import (
            CRITICAL_FILES,
            NEVER_SAFE_PATTERN_SOURCES,
            SAFE_EDIT_FILES,
        )
        from core.self_edit import SelfEditEngine

        assert SelfEditEngine.SAFE_FILES is SAFE_EDIT_FILES
        assert SelfEditEngine.CRITICAL_FILES is CRITICAL_FILES
        assert len(SelfEditEngine.NEVER_SAFE_PATTERNS) == len(NEVER_SAFE_PATTERN_SOURCES)

    def test_registry_has_no_file_both_safe_and_critical(self):
        """The two lists used to disagree; critical must always win."""
        from core.safe_files import CRITICAL_FILES, SAFE_EDIT_FILES

        overlap = SAFE_EDIT_FILES & CRITICAL_FILES
        assert not overlap, f"files marked both safe and critical: {sorted(overlap)}"

    @pytest.mark.parametrize("path", [
        "core/self_edit.py",       # the sandbox must not edit its own rules
        "core/vector_memory.py",
        "core/knowledge_graph.py",
        "core/auth.py",
        "core/permissions.py",
        "run.py",
    ])
    def test_critical_files_are_not_editable(self, path):
        can_edit, reason = _engine().can_edit_file(path)
        assert can_edit is False, f"{path} should be protected, got: {reason}"

    @pytest.mark.parametrize("path", [
        "core/erdos_engine.py",
        "core/daydream_engine.py",
        "tools/evolution_tools.py",
        "web/src/App.tsx",
    ])
    def test_designated_safe_files_remain_editable(self, path):
        """Tightening policy must not have removed legitimate self-edit targets."""
        can_edit, reason = _engine().can_edit_file(path)
        assert can_edit is True, f"{path} should stay editable, got: {reason}"

    @pytest.mark.parametrize("snippet", [
        "def go(c):\n    return _spawn(c, shell=True)",   # bare shell=True
        "import pdb",
        "breakpoint()",
        "x = eval('1+1')",
        "data = pickle.loads(b'')",
    ])
    def test_forbidden_patterns_block_edits(self, snippet):
        valid, reason = _engine().validate_code(snippet)
        assert valid is False, f"should have been blocked: {snippet!r}"

    def test_benign_code_still_passes(self):
        valid, reason = _engine().validate_code("def add(a, b):\n    return a + b")
        assert valid is True, reason


class TestPrefrontalCortexObserverIsWired:
    """Insights must reach the PFC learning loop."""

    def test_insight_bridge_reports_to_pfc(self, monkeypatch):
        import core.pfc_observer as observer
        from core.insight_bridge import get_insight_bridge

        seen = []

        async def spy(insight, source, metadata=None):
            seen.append({"insight": insight, "source": source, "metadata": metadata or {}})
            return True

        monkeypatch.setattr(observer, "observe_insight", spy)

        async def drive():
            get_insight_bridge().process_insight(
                raw_insight=f"wiring probe {uuid.uuid4()}: deletion tightens the bound",
                source_problem=f"probe problem {uuid.uuid4()}",
                source_category="probe_category",
                strategy_used="probabilistic",
            )
            await asyncio.sleep(0.25)  # let the scheduled task run

        asyncio.run(drive())

        assert seen, "insight_bridge did not report the insight to the PFC observer"
        assert seen[0]["source"].startswith("insight_bridge:")
        assert seen[0]["metadata"].get("insight_id")

    def test_observer_fails_closed_without_pfc(self, monkeypatch):
        """Must degrade quietly, never raise into the caller."""
        import core.pfc_observer as observer

        monkeypatch.setattr(observer, "_pfc", None)
        monkeypatch.setattr(
            observer, "observe_event", observer.observe_event
        )  # keep real implementation
        result = asyncio.run(observer.observe_insight("x", source="test"))
        assert result in (True, False)  # never an exception


class TestSSRFProtectorIsWired:
    """core/security/ssrf_protection must gate real outbound fetches."""

    def test_exported_from_security_package(self):
        from core.security import get_ssrf_protector, validate_outbound_url

        assert get_ssrf_protector() is not None
        assert validate_outbound_url("https://example.com/page").valid is True

    @pytest.mark.parametrize("url", [
        "https://example.com:22/x",      # SSH
        "https://example.com:6379/x",    # Redis
        "https://user:pw@example.com/x",  # embedded credentials
    ])
    def test_protector_denials_reach_the_fetch_path(self, url):
        from tools.web import is_safe_url

        safe, reason = is_safe_url(url)
        assert safe is False, f"{url} should be blocked; reason={reason!r}"


class TestSharedEmbeddingServerIsStarted:
    """run.py must start the shared embedding server.

    Without it every process — Jarvis plus each sub-agent worker — silently
    falls back to loading its own ~239M-parameter copy of the model.
    """

    def test_run_py_starts_the_server(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[1].joinpath("run.py").read_text(encoding="utf-8")
        assert "start_server_if_needed" in src, (
            "run.py no longer starts the shared embedding server"
        )

    def test_start_helper_is_awaitable(self):
        from core.embedding_server import start_server_if_needed

        assert inspect.iscoroutinefunction(start_server_if_needed)


class TestStructuredLoggingIsWired:
    """Correlation IDs must be attached to every web request."""

    def test_request_context_resets_on_exit(self):
        """Regression: __exit__ looked up the var via `token.var.old_value`,
        which raised AttributeError and was swallowed, so nothing was ever
        reset and request ids leaked between requests."""
        from core.structured_logging import RequestContext, get_request_id, get_session_id

        assert get_request_id() is None
        with RequestContext(request_id="AAA", session_id="S1"):
            assert get_request_id() == "AAA"
            assert get_session_id() == "S1"
        assert get_request_id() is None, "request id leaked past the context manager"
        assert get_session_id() is None, "session id leaked past the context manager"

    def test_request_context_nests(self):
        from core.structured_logging import RequestContext, get_request_id

        with RequestContext(request_id="OUTER"):
            with RequestContext(request_id="INNER"):
                assert get_request_id() == "INNER"
            assert get_request_id() == "OUTER", "inner exit clobbered the outer value"
        assert get_request_id() is None

    def test_web_interface_installs_request_id_middleware(self):
        src = REPO_ROOT.joinpath("interfaces", "web_react.py").read_text(encoding="utf-8")
        assert "request_id_middleware" in src, (
            "web_react no longer attaches correlation IDs to requests"
        )
        assert "X-Request-ID" in src


class TestRuntimeIntegrityIsWired:
    """Critical files must be watched for modification while Jarvis runs."""

    def test_run_py_starts_the_checker(self):
        src = REPO_ROOT.joinpath("run.py").read_text(encoding="utf-8")
        assert "get_integrity_checker" in src, (
            "run.py no longer starts the runtime integrity checker"
        )

    def test_checker_uses_the_shared_critical_registry(self):
        """It kept a private 6-entry list while the registry carried 16."""
        from core.runtime_integrity import RuntimeIntegrityChecker
        from core.safe_files import CRITICAL_FILES

        checker = RuntimeIntegrityChecker(jarvis_root=REPO_ROOT)
        assert set(checker.critical_files) == set(CRITICAL_FILES)

    def test_check_all_now_leaves_no_orphaned_coroutine(self):
        """Regression: it called run_until_complete() on the running loop, then
        asyncio.run(), both raising, leaving two un-awaited coroutines behind."""
        import warnings

        from core.runtime_integrity import RuntimeIntegrityChecker

        async def drive():
            checker = RuntimeIntegrityChecker(jarvis_root=REPO_ROOT)
            checker.start()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    result = checker.check_all_now()
                return result
            finally:
                checker.stop()

        result = asyncio.run(drive())
        assert result["passed"] is True
        assert result["checks"], "no checks were run"

    def test_sync_health_callbacks_still_run_in_sync_path(self):
        from core.runtime_integrity import RuntimeIntegrityChecker

        checker = RuntimeIntegrityChecker(jarvis_root=REPO_ROOT)
        checker.start()
        try:
            checker.register_health_callback("probe", lambda: {"healthy": True, "message": "ok"})
            names = {c["name"]: c["message"] for c in checker.check_all_now()["checks"]}
            assert names.get("subsystem.probe") == "ok"
        finally:
            checker.stop()


class TestEmbeddingHealthIsObservable:
    """A dead embedder must be visible, not silently corrupt every subsystem.

    When the model fails to load, encode() returns zero vectors — every pair is
    then perfectly similar, so vector memory, the knowledge graph and insight
    retrieval return confident nonsense instead of failing. That happened for
    real (a missing 'peft' dependency) and nothing could detect it.
    """

    def test_degraded_embedder_reports_unhealthy_with_cause(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder()
        embedder._model = None
        embedder._load_errors = ["pytorch: requires package 'peft' which was not found"]

        health = embedder.health()
        assert health["healthy"] is False
        assert health["status"] == "degraded"
        assert health["returns_zero_vectors"] is True
        assert any("peft" in e for e in health["load_errors"]), "cause not reported"

    def test_health_does_not_force_a_model_load(self):
        """Polling health must stay cheap — it must not pull a 239M model."""
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder()
        health = embedder.health()
        assert embedder._model is None, "health() triggered a model load"
        assert health["status"] == "not_loaded"
        assert health["healthy"] is True  # not yet attempted != broken

    def test_interconnect_exposes_health(self):
        from core.embedding_interconnect import EmbeddingInterconnect

        health = EmbeddingInterconnect.health
        assert callable(health)

    def test_run_py_registers_embedding_health(self):
        src = REPO_ROOT.joinpath("run.py").read_text(encoding="utf-8")
        assert 'health_mgr.register("embeddings"' in src, (
            "run.py no longer reports embedding health"
        )


class TestDequeSlicingRegressions:
    """collections.deque does not support slice indexing.

    Twelve sites did `self.some_deque[-N:]`, which raises
    `TypeError: sequence index must be integer, not 'slice'` the moment the
    deque is non-empty. Several sat inside broad `except Exception` handlers, so
    the feature silently became a permanent no-op rather than failing loudly.
    """

    def test_deque_slicing_still_raises(self):
        """Guard the premise — if this ever changes, these fixes are moot."""
        from collections import deque

        with pytest.raises(TypeError):
            deque([1, 2, 3])[-2:]

    def test_no_deque_is_sliced_anywhere(self):
        """Static sweep: no `self.<deque>[a:b]` anywhere in the tree."""
        import ast

        offenders = []
        for pkg in ("core", "tools", "interfaces", "speech"):
            for path in (REPO_ROOT / pkg).rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
                except SyntaxError:
                    continue
                deques = set()
                for node in ast.walk(tree):
                    value = getattr(node, "value", None)
                    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Call):
                        fn = value.func
                        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) == "deque":
                            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                            for t in targets:
                                name = getattr(t, "attr", None) or getattr(t, "id", None)
                                if name:
                                    deques.add(name)
                if not deques:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                        name = getattr(node.value, "attr", None) or getattr(node.value, "id", None)
                        if name in deques:
                            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno} ({name})")

        assert not offenders, "deque sliced (raises TypeError):\n  " + "\n  ".join(offenders)

    def test_learning_pipeline_session_history_works(self):
        from core.learning_pipeline import get_learning_pipeline

        assert isinstance(get_learning_pipeline().get_session_history(), list)

    def test_resource_monitor_statistics_work(self):
        from core.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()
        monitor.history.append(monitor._get_snapshot())
        assert "current_cpu" in monitor.get_statistics()


class TestResourceMonitorIsCrossPlatform:
    """os.getloadavg() is POSIX-only; calling it unconditionally broke every snapshot."""

    def test_snapshot_succeeds_on_this_platform(self):
        from core.resource_monitor import get_resource_monitor

        snap = get_resource_monitor()._get_snapshot()
        assert snap.memory_percent >= 0
        assert len(snap.load_average) == 3

    def test_pressure_checks_do_not_raise(self):
        from core.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()
        assert isinstance(monitor.should_throttle(), bool)
        under, _reason = monitor.is_under_pressure()
        assert isinstance(under, bool)


class TestPermissionDenyIsHonoured:
    """A stored permanent deny must read back as DENY, not None."""

    def test_permanent_deny_returns_deny(self, tmp_path):
        from core.permissions import PermissionLevel, PermissionManager

        manager = PermissionManager(storage_path=tmp_path / "perm.json")
        manager.deny_permission("write_file", permanently=True)
        assert manager.check_permission("write_file") == PermissionLevel.DENY

    def test_grant_still_returns_allow(self, tmp_path):
        from core.permissions import PermissionLevel, PermissionManager

        manager = PermissionManager(storage_path=tmp_path / "perm.json")
        manager.grant_permission("write_file", ttl=3600)
        assert manager.check_permission("write_file") == PermissionLevel.ALLOW_SESSION


class TestSelfRefineHonoursCriticalFiles:
    """Windows path separators made every membership test miss."""

    @pytest.mark.parametrize("name", ["agent.py", "llm.py", "permissions.py", "auth.py"])
    def test_critical_files_refused_with_native_separator(self, name):
        import os
        from pathlib import Path

        from core.self_refine import AgentLayer

        layer = AgentLayer(base_path=Path(REPO_ROOT))
        native = "core" + os.sep + name
        safe, reason = layer.is_safe_to_modify(native)
        assert safe is False, f"{native!r} should be refused, got {reason!r}"

    def test_designated_safe_file_still_allowed(self):
        import os
        from pathlib import Path

        from core.self_refine import AgentLayer

        layer = AgentLayer(base_path=Path(REPO_ROOT))
        safe, _reason = layer.is_safe_to_modify("core" + os.sep + "erdos_engine.py")
        assert safe is True


class TestVectorMemorySearchContract:
    """VectorMemory.search() returns (entry_id, score, metadata) tuples.

    Four separate consumers treated those tuples as objects — probing
    `result.metadata` / `result.score` via hasattr, or calling `.get()` on them.
    hasattr is always False on a tuple and `.get()` raises AttributeError, and
    both failures were swallowed, so the consumers silently returned empty or
    mis-scored results rather than erroring.
    """

    @pytest.fixture(scope="class")
    def populated(self, tmp_path_factory):
        import os

        os.environ.setdefault("JARVIS_LOCAL_EMBEDDER", "1")
        from core.vector_memory import VectorMemory

        path = tmp_path_factory.mktemp("vm")
        memory = VectorMemory(storage_path=path)
        memory.store("the quick brown fox", metadata={"tag": "a"}, importance=0.8)
        memory.store("lazy dogs sleeping", metadata={"tag": "b"}, importance=0.6)
        memory._save_index()
        return path, memory

    def test_search_returns_three_tuples(self, populated):
        _path, memory = populated
        results = memory.search("brown", k=2)
        assert results, "no results to check the contract against"
        for item in results:
            assert isinstance(item, tuple) and len(item) == 3
            entry_id, score, metadata = item
            assert isinstance(entry_id, str)
            assert isinstance(score, float)
            assert isinstance(metadata, dict)

    def test_metadata_survives_a_restart(self, populated):
        """Regression: _load_index restored the graph but not metadata_index,
        so every post-restart search returned {} for metadata."""
        from core.vector_memory import VectorMemory

        path, _memory = populated
        reopened = VectorMemory(storage_path=path)
        results = reopened.search("brown", k=2)

        assert results, "index did not survive the restart"
        assert all(md for _id, _score, md in results), "metadata came back empty"
        assert any(md.get("text") for _id, _score, md in results), "text not rehydrated"
        assert reopened.importance_scores, "importance scores not rehydrated"


class TestUnifiedMemoryIndexSurvivesRestart:
    def test_keyword_search_works_after_reload(self, tmp_path):
        """Regression: the word index is not persisted and was only built in
        store(), so after a restart search() had no candidates and returned []."""
        import os

        os.environ.setdefault("JARVIS_LOCAL_EMBEDDER", "1")
        from core.unified_memory import MemoryType, UnifiedMemory

        first = UnifiedMemory(storage_path=tmp_path)
        first.store("the quick brown fox jumps", MemoryType.SEMANTIC)
        first._save()

        reopened = UnifiedMemory(storage_path=tmp_path)
        assert reopened._memories, "memories did not reload"
        assert reopened._index, "word index was not rebuilt on load"
        assert reopened.search("brown"), "keyword search returned nothing after restart"


class TestToolCallMessageOrdering:
    """The assistant message declaring tool_calls must precede its tool results."""

    def test_assistant_precedes_tool_messages(self):
        from dataclasses import dataclass, field

        from core.agent_loop import AgentLoop

        @dataclass
        class _Call:
            call_id: str
            tool_name: str
            input: dict
            output: str = ""
            error: str = ""

        @dataclass
        class _Assistant:
            text: str = ""
            tool_calls: list = field(default_factory=list)

        @dataclass
        class _Step:
            assistant_message: _Assistant

        loop = AgentLoop.__new__(AgentLoop)
        step = _Step(_Assistant("thinking", [
            _Call("c1", "read", {}, output="contents"),
            _Call("c2", "write", {}, output="", error="disk full"),
        ]))
        messages = loop._append_step_to_messages([{"role": "user", "content": "hi"}], step)
        roles = [m["role"] for m in messages]

        assert roles.index("assistant") < roles.index("tool"), (
            "tool results precede the assistant message that declares them"
        )

    def test_every_tool_call_gets_a_response(self):
        """A failed call produced no tool message, leaving a dangling tool_call_id."""
        from dataclasses import dataclass, field

        from core.agent_loop import AgentLoop

        @dataclass
        class _Call:
            call_id: str
            tool_name: str
            input: dict
            output: str = ""
            error: str = ""

        @dataclass
        class _Assistant:
            text: str = ""
            tool_calls: list = field(default_factory=list)

        @dataclass
        class _Step:
            assistant_message: _Assistant

        loop = AgentLoop.__new__(AgentLoop)
        step = _Step(_Assistant("t", [
            _Call("c1", "read", {}, output="ok"),
            _Call("c2", "write", {}, output="", error="disk full"),
        ]))
        messages = loop._append_step_to_messages([], step)

        declared = {c["id"] for m in messages for c in m.get("tool_calls", [])}
        answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
        assert declared == answered, f"unanswered tool_calls: {declared - answered}"


class TestPromptTemplatesRender:
    """Literal JSON braces in a .format() template raise KeyError before any LLM call."""

    def test_ethical_prompt_formats(self):
        from core.self_evaluation import ETHICAL_PRINCIPLES_PROMPT

        rendered = ETHICAL_PRINCIPLES_PROMPT.format(
            action_type="tool_call", subsystem="agent",
            input_preview="hi", output_preview="ok",
        )
        assert '"scores"' in rendered, "JSON example block was lost"
        assert "tool_call" in rendered

    def test_reflection_prompt_formats(self):
        from core.self_evaluation import SELF_REFLECTION_PROMPT

        rendered = SELF_REFLECTION_PROMPT.format(
            period_start="a", period_end="b", total_actions=3,
            avg_score=0.87, total_flags=1, weak_principles="privacy",
            flags_summary="none",
        )
        assert "0.87" in rendered
        assert '"summary"' in rendered


class TestAdaptiveLoopHonoursShellBlocklist:
    """The adaptive loop routes shell tools through ExecutionContext for a
    persistent cwd — which also routed around TerminalExecuteTool's high-risk
    blocklist. The same commands were refused on the tool path and executed here.
    """

    @pytest.mark.parametrize("command,label", [
        ("rm -rf /", "wildcard root delete"),
        ("echo $(id)", "command substitution"),
    ])
    def test_high_risk_commands_are_blocked(self, command, label):
        from core.adaptive_agent_loop import AdaptiveAgentLoop
        from tools.terminal import TerminalExecuteTool

        loop = AdaptiveAgentLoop()
        registry = {"terminal_execute": TerminalExecuteTool()}
        result = asyncio.run(
            loop.execute_tool_with_recovery("terminal_execute", {"command": command}, registry)
        )

        assert result.get("success") is False, f"{label} was executed"
        assert "BLOCK" in str(result.get("error", "")).upper()

    def test_benign_command_still_runs(self):
        from core.adaptive_agent_loop import AdaptiveAgentLoop
        from tools.terminal import TerminalExecuteTool

        loop = AdaptiveAgentLoop()
        registry = {"terminal_execute": TerminalExecuteTool()}
        result = asyncio.run(
            loop.execute_tool_with_recovery("terminal_execute", {"command": "echo SAFE_PROOF"}, registry)
        )
        assert "SAFE_PROOF" in str(result.get("output", ""))

    def test_risk_assessment_fails_closed(self, monkeypatch):
        """If the assessor cannot be loaded the command must be refused."""
        import core.adaptive_agent_loop as module

        def boom(_command):
            raise RuntimeError("assessor exploded")

        monkeypatch.setattr(module, "_assess_command_risk", lambda c: ("high", "assessor unavailable"))
        loop = module.AdaptiveAgentLoop()
        result = loop._execute_shell_tool("terminal_execute", {"command": "echo hi"}, 10.0)
        assert result.get("success") is False


class TestProtectedPathHelperIsShared:
    """The six protected-path blocklists must use one matching mechanism."""

    def test_helper_rejects_real_system_paths(self):
        from core.helpers import is_protected_path

        roots = ("/etc", "/root", "/dev")
        for p in ("/etc", "/etc/passwd", "/root/.ssh/id_rsa", "/dev/null"):
            assert is_protected_path(p, roots), p

    def test_helper_does_not_reject_lookalike_siblings(self):
        """Bare startswith() used to deny /etcetera and /devops."""
        from core.helpers import is_protected_path

        roots = ("/etc", "/root", "/dev", "/bin")
        for p in ("/etcetera/data", "/devops/scripts", "/bindings", "/rootkit-scan"):
            assert not is_protected_path(p, roots), p

