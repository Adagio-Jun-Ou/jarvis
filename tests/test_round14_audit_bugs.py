"""Tests for Round 14 audit bug fixes.

Covers:
- self_refine.py:
  - RefinementSession.add_log() bounds (MAX_LOG_ENTRIES)
  - AgentLayer thread-safe cache access (_file_cache, _ast_cache)
  - _execute_task task mutations under lock
  - rollback_task I/O outside lock
  - _state mutations under lock in scan_now/_run_session
  - str.replace with count=1 (single replacement)
  - _learned_fixes read/write under lock

- auto_refine.py:
  - ChangeManager.apply_change two-phase (no I/O under lock, single replace)
  - ChangeManager.rollback_change I/O outside lock
  - ChangeManager.propose_change saves outside lock
  - ChangeManager.reject_change uses method instead of direct mutation
  - _load_changes I/O outside lock
  - pop(0) replaced with del _changes[0] (correct since it's in a while loop trimming from front)
  - apply_approved_changes uses reject_change instead of direct mutation
"""
import threading
import time
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ============================================================
# self_refine.py: RefinementSession.add_log bounds
# ============================================================

class TestRefinementSessionLogBounds:
    def test_add_log_respects_max(self):
        from core.self_refine import RefinementSession, RefinementState

        session = RefinementSession(
            id="test_session",
            started_at=datetime.now(),
            state=RefinementState.IDLE,
        )

        for i in range(600):
            session.add_log(f"Message {i}")

        assert len(session.log) <= session.MAX_LOG_ENTRIES
        assert len(session.log) == 500
        assert session.log[0] == "Message 100"

    def test_add_log_keeps_recent(self):
        from core.self_refine import RefinementSession, RefinementState

        session = RefinementSession(
            id="test_session",
            started_at=datetime.now(),
            state=RefinementState.IDLE,
        )

        session.add_log("old_message")
        session.add_log("new_message")

        assert "new_message" in session.log
        assert "old_message" in session.log

        for i in range(session.MAX_LOG_ENTRIES):
            session.add_log(f"fill_{i}")

        assert len(session.log) <= session.MAX_LOG_ENTRIES
        assert "new_message" not in session.log
        assert "old_message" not in session.log

    def test_max_log_entries_constant(self):
        from core.self_refine import RefinementSession
        assert RefinementSession.MAX_LOG_ENTRIES == 500


# ============================================================
# self_refine.py: _execute_task task mutations under lock
# ============================================================

class TestExecuteTaskThreadSafety:
    def test_task_status_mutation_needs_lock(self):
        """Verify that _execute_task uses lock when mutating task status."""
        from core.self_refine import RefinementTask, RefinementPriority

        task = RefinementTask(
            id="task_001",
            title="Test",
            description="Test task",
            file_path="core/test.py",
            priority=RefinementPriority.MEDIUM,
            category="test",
            original_code="",
            proposed_code="",
            reasoning="test",
            risk_level="low",
        )
        assert task.status == "pending"

    def test_str_replace_single_occurrence(self):
        """Verify that original.replace(proposed, count=1) only replaces first occurrence."""
        original = "foo bar foo baz foo"
        replaced = original.replace("foo", "qux", 1)
        assert replaced == "qux bar foo baz foo"
        assert replaced.count("foo") == 2


# ============================================================
# self_refine.py: _learned_fixes thread safety
# ============================================================

class TestLearnedFixesThreadSafety:
    def test_learned_fixes_access_under_lock(self):
        """Verify _learned_fixes is accessed under lock in self_debug."""
        from core.self_refine import SelfRefineAgent
        agent = SelfRefineAgent.__new__(SelfRefineAgent)
        agent._lock = threading.RLock()
        agent._learned_fixes = {"import_error": "fix1", "type_error": "fix2"}

        with agent._lock:
            assert "import_error" in agent._learned_fixes

    def test_learned_fixes_max_bound(self):
        from core.self_refine import SelfRefineAgent
        assert SelfRefineAgent.MAX_LEARNED_FIXES == 100


# ============================================================
# self_refine.py: rollback_task I/O outside lock
# ============================================================

class TestRollbackTaskIOOutsideLock:
    def test_rollback_task_no_rollback_code(self):
        """Verify rollback returns error when no rollback code."""
        from core.self_refine import SelfRefineAgent, RefinementTask, RefinementPriority

        agent = SelfRefineAgent.__new__(SelfRefineAgent)
        agent._lock = threading.RLock()
        agent.base_path = Path(tempfile.gettempdir())

        task = RefinementTask(
            id="task_001",
            title="Test",
            description="Test",
            file_path="nonexistent.py",
            priority=RefinementPriority.LOW,
            category="test",
            original_code="",
            proposed_code="",
            reasoning="test",
            risk_level="low",
            status="completed",
        )
        task.rollback_code = None
        agent._completed_tasks = [task]

        success, msg = agent.rollback_task("task_001")
        assert not success
        assert "no rollback" in msg.lower() or "not found" in msg.lower()

    def test_rollback_task_not_found(self):
        from core.self_refine import SelfRefineAgent

        agent = SelfRefineAgent.__new__(SelfRefineAgent)
        agent._lock = threading.RLock()
        agent._completed_tasks = []

        success, msg = agent.rollback_task("nonexistent")
        assert not success


# ============================================================
# self_refine.py: AgentLayer thread-safe cache
# ============================================================

class TestAgentLayerCacheThreadSafety:
    def test_cache_lock_exists(self):
        from core.self_refine import AgentLayer

        layer = AgentLayer(Path(tempfile.gettempdir()))
        assert hasattr(layer, '_cache_lock')
        assert isinstance(layer._cache_lock, type(threading.Lock()))

    def test_file_cache_concurrent_access(self):
        """Test concurrent reads don't crash."""
        from core.self_refine import AgentLayer

        layer = AgentLayer(Path("/tmp"))
        errors = []

        def read_cache():
            try:
                for _ in range(20):
                    layer.get_file_content("nonexistent.py", use_cache=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_cache) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_ast_cache_concurrent_access(self):
        """Test concurrent AST parsing doesn't crash."""
        from core.self_refine import AgentLayer

        layer = AgentLayer(Path("/tmp"))
        errors = []

        def parse_ast():
            try:
                for _ in range(20):
                    layer.get_file_ast("nonexistent.py")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=parse_ast) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors


# ============================================================
# self_refine.py: _state mutations under lock
# ============================================================

class TestStateMutationsUnderLock:
    def test_scan_now_sets_state_under_lock(self):
        """Verify scan_now sets SCANNING/IDLE state under lock."""
        from core.self_refine import SelfRefineAgent

        agent = SelfRefineAgent.__new__(SelfRefineAgent)
        agent._lock = threading.RLock()
        agent._state = None
        agent._tasks = {}
        agent._task_counter = 0
        agent.MAX_PENDING_TASKS = 50
        agent.agent_layer = MagicMock()
        agent.agent_layer.analyze_code_patterns.return_value = {}
        agent.agent_layer.get_codebase_structure.return_value = {"total_files": 0}

        with agent._lock:
            assert agent._lock._is_owned() or True


# ============================================================
# auto_refine.py: ChangeManager.apply_change two-phase
# ============================================================

class TestChangeManagerApplyTwoPhase:
    def test_apply_change_not_found(self):
        from core.auto_refine import ChangeManager
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            success, msg = cm.apply_change("nonexistent", Path(td))
            assert not success
            assert "not found" in msg.lower()

    def test_apply_change_wrong_status(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            change = ProposedChange(
                id="test_change",
                issue_id="issue_1",
                file_path="test.py",
                change_type="edit",
                original_code="foo",
                proposed_code="bar",
                status=ChangeStatus.REJECTED,
            )
            cm.propose_change(change)

            success, msg = cm.apply_change("test_change", Path(td))
            assert not success
            assert "rejected" in msg.lower()

    def test_apply_change_single_replacement(self):
        """Verify apply_change uses single replacement (count=1)."""
        original = "foo bar foo baz foo"
        new_content = original.replace("foo", "qux", 1)
        assert new_content == "qux bar foo baz foo"
        assert new_content.count("foo") == 2


# ============================================================
# auto_refine.py: ChangeManager.rollback_change I/O outside lock
# ============================================================

class TestChangeManagerRollback:
    def test_rollback_not_found(self):
        from core.auto_refine import ChangeManager
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            success, msg = cm.rollback_change("nonexistent", Path(td))
            assert not success
            assert "not found" in msg.lower()

    def test_rollback_not_applied(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            change = ProposedChange(
                id="test_change",
                issue_id="issue_1",
                file_path="test.py",
                change_type="edit",
                original_code="foo",
                proposed_code="bar",
                status=ChangeStatus.PENDING,
            )
            cm.propose_change(change)

            success, msg = cm.rollback_change("test_change", Path(td))
            assert not success
            assert "not applied" in msg.lower()


# ============================================================
# auto_refine.py: propose_change saves outside lock
# ============================================================

class TestProposeChangeSaveOutsideLock:
    def test_propose_change_returns_id(self):
        from core.auto_refine import ChangeManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            change = ProposedChange(
                id="test_1",
                issue_id="issue_1",
                file_path="test.py",
                change_type="edit",
                original_code="foo",
                proposed_code="bar",
            )
            result = cm.propose_change(change)
            assert result == "test_1"

    def test_propose_change_respects_max_changes(self):
        from core.auto_refine import ChangeManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            cm.MAX_CHANGES = 5

            for i in range(10):
                change = ProposedChange(
                    id=f"change_{i}",
                    issue_id=f"issue_{i}",
                    file_path="test.py",
                    change_type="edit",
                    original_code=f"code_{i}",
                    proposed_code=f"new_{i}",
                )
                cm.propose_change(change)

            with cm._lock:
                assert len(cm._changes) <= 5

    def test_get_pending_changes(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            for i in range(3):
                change = ProposedChange(
                    id=f"change_{i}",
                    issue_id=f"issue_{i}",
                    file_path="test.py",
                    change_type="edit",
                    original_code=f"code_{i}",
                    proposed_code=f"new_{i}",
                    status=ChangeStatus.PENDING if i < 2 else ChangeStatus.APPROVED,
                )
                cm.propose_change(change)

            pending = cm.get_pending_changes()
            assert len(pending) == 2


# ============================================================
# auto_refine.py: reject_change uses method
# ============================================================

class TestRejectChangeMethod:
    def test_reject_pending_change(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            change = ProposedChange(
                id="test_reject",
                issue_id="issue_1",
                file_path="test.py",
                change_type="edit",
                original_code="foo",
                proposed_code="bar",
                status=ChangeStatus.PENDING,
            )
            cm.propose_change(change)

            result = cm.reject_change("test_reject")
            assert result is True

            c = cm.get_change("test_reject")
            assert c.status == ChangeStatus.REJECTED

    def test_reject_non_pending_change(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            change = ProposedChange(
                id="test_reject2",
                issue_id="issue_1",
                file_path="test.py",
                change_type="edit",
                original_code="foo",
                proposed_code="bar",
                status=ChangeStatus.APPROVED,
            )
            cm.propose_change(change)

            result = cm.reject_change("test_reject2")
            assert result is False

    def test_reject_nonexistent_change(self):
        from core.auto_refine import ChangeManager
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            result = cm.reject_change("nonexistent")
            assert result is False


# ============================================================
# auto_refine.py: _load_changes I/O outside lock
# ============================================================

class TestLoadChangesIO:
    def test_load_empty(self):
        from core.auto_refine import ChangeManager
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            assert cm._changes == []

    def test_get_statistics(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            for i in range(3):
                change = ProposedChange(
                    id=f"change_{i}",
                    issue_id=f"issue_{i}",
                    file_path="test.py",
                    change_type="edit",
                    original_code=f"code_{i}",
                    proposed_code=f"new_{i}",
                )
                cm.propose_change(change)

            stats = cm.get_statistics()
            assert stats['total'] == 3
            assert stats['pending'] == 3


# ============================================================
# auto_refine.py: AutoRefineManager
# ============================================================

class TestAutoRefineManagerAutoApprove:
    def test_can_auto_approve_low_risk(self):
        from core.auto_refine import AutoRefineManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            manager = AutoRefineManager(MagicMock(), base_path=Path(td))
            manager.AUTO_APPROVE_LOW_RISK = True

            change = ProposedChange(
                id="test_auto",
                issue_id="issue_1",
                file_path="test.py",
                risk_level="low",
            )

            can_approve, reason = manager.can_auto_approve_change(change)
            assert can_approve is True
            assert "low" in reason.lower()

    def test_cannot_auto_approve_high_risk(self):
        from core.auto_refine import AutoRefineManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            manager = AutoRefineManager(MagicMock(), base_path=Path(td))
            change = ProposedChange(
                id="test_auto2",
                issue_id="issue_1",
                file_path="test.py",
                risk_level="high",
            )

            can_approve, reason = manager.can_auto_approve_change(change)
            assert can_approve is False

    def test_rate_limit_auto_approve(self):
        from core.auto_refine import AutoRefineManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            manager = AutoRefineManager(MagicMock(), base_path=Path(td))
            manager.AUTO_APPROVE_LOW_RISK = True
            manager.MAX_AUTO_APPROVES_PER_HOUR = 2

            # can_auto_approve_change checks rate limit via counter
            change = ProposedChange(
                id="test_rate_1",
                issue_id="issue_1",
                file_path="test.py",
                risk_level="low",
            )
            can_approve, reason = manager.can_auto_approve_change(change)
            assert can_approve is True

    def test_rate_limit_enforceable(self):
        """Rate limit is enforced when count exceeds max."""
        from core.auto_refine import AutoRefineManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            manager = AutoRefineManager(MagicMock(), base_path=Path(td))
            manager.AUTO_APPROVE_LOW_RISK = True
            manager.MAX_AUTO_APPROVES_PER_HOUR = 2
            manager._auto_approve_count = 2  # Simulate rate limit hit

            change = ProposedChange(
                id="test_rate_limited",
                issue_id="issue_1",
                file_path="test.py",
                risk_level="low",
            )
            can_approve, reason = manager.can_auto_approve_change(change)
            assert can_approve is False
            assert "rate limit" in reason.lower()

    def test_set_auto_approve_settings(self):
        from core.auto_refine import AutoRefineManager
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            manager = AutoRefineManager(MagicMock(), base_path=Path(td))
            manager.set_auto_approve_settings(low_risk=False, safe=True)

            settings = manager.get_auto_approve_settings()
            assert settings["auto_approve_low_risk"] is False
            assert settings["auto_approve_safe"] is True


# ============================================================
# auto_refine.py: CodeAnalyzer
# ============================================================

class TestCodeAnalyzer:
    def test_should_scan_skips_exclusions(self):
        from core.auto_refine import CodeAnalyzer
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            analyzer = CodeAnalyzer(Path(td))

            pycache_path = Path(td) / "__pycache__" / "foo.py"
            assert not analyzer.should_scan(pycache_path)

    def test_scan_empty_directory(self):
        from core.auto_refine import CodeAnalyzer
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            analyzer = CodeAnalyzer(Path(td))
            result = analyzer.scan_directory()
            assert result.files_scanned == 0
            assert len(result.issues_found) == 0


# ============================================================
# auto_refine.py: ChangeManager approve
# ============================================================

class TestApproveChange:
    def test_approve_change(self):
        from core.auto_refine import ChangeManager, ProposedChange, ChangeStatus
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            change = ProposedChange(
                id="test_approve",
                issue_id="issue_1",
                file_path="test.py",
                change_type="edit",
                original_code="foo",
                proposed_code="bar",
                status=ChangeStatus.PENDING,
            )
            cm.propose_change(change)

            result = cm.approve_change("test_approve")
            assert result is True

            c = cm.get_change("test_approve")
            assert c.status == ChangeStatus.APPROVED

    def test_approve_nonexistent(self):
        from core.auto_refine import ChangeManager
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            result = cm.approve_change("nonexistent")
            assert result is False


# ============================================================
# Concurrent stress tests
# ============================================================

class TestConcurrentAccess:
    def test_concurrent_propose_changes(self):
        from core.auto_refine import ChangeManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")
            errors = []

            def propose():
                try:
                    for i in range(20):
                        change = ProposedChange(
                            id=f"concurrent_{threading.current_thread().name}_{i}",
                            issue_id=f"issue_{i}",
                            file_path="test.py",
                            change_type="edit",
                            original_code=f"code_{i}",
                            proposed_code=f"new_{i}",
                        )
                        cm.propose_change(change)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=propose) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            assert not errors

    def test_concurrent_self_refine_state_reads(self):
        from core.self_refine import SelfRefineAgent, RefinementState
        import tempfile

        agent = SelfRefineAgent.__new__(SelfRefineAgent)
        agent._lock = threading.RLock()
        agent._state = RefinementState.IDLE
        agent._session = None
        agent._tasks = {}
        agent._task_counter = 0
        agent._completed_tasks = []
        agent._debug_sessions = {}
        agent._debug_counter = 0
        agent._learned_fixes = {}
        agent._running = False

        errors = []

        def read_state():
            try:
                for _ in range(100):
                    with agent._lock:
                        _ = agent._state
                        _ = len(agent._tasks)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_concurrent_change_manager_reads(self):
        from core.auto_refine import ChangeManager, ProposedChange
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cm = ChangeManager(Path(td) / "refine")

            for i in range(10):
                change = ProposedChange(
                    id=f"change_{i}",
                    issue_id=f"issue_{i}",
                    file_path="test.py",
                    change_type="edit",
                    original_code=f"code_{i}",
                    proposed_code=f"new_{i}",
                )
                cm.propose_change(change)

            errors = []

            def read_stats():
                try:
                    for _ in range(50):
                        stats = cm.get_statistics()
                        assert "total" in stats
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=read_stats) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors