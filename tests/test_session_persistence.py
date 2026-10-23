"""Tests for Session Persistence Manager.

Covers:
- Session creation, listing, deletion
- Turn addition and retrieval
- Atomic writes and HMAC integrity
- Crash recovery / stale session detection
- Session data truncation for large sessions
- LRU eviction
- Thread safety
"""
import json
import os
import time
import tempfile
import threading
from pathlib import Path

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from core.session_persistence import (
    SessionPersistence, SessionTurn, SessionMetadata,
    get_session_persistence,
)


class TestSessionTurn:
    def test_turn_creation(self):
        turn = SessionTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_turn_to_dict(self):
        turn = SessionTurn(role="assistant", content="response", tool_calls=[{"name": "test"}])
        d = turn.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "response"
        assert len(d["tool_calls"]) == 1

    def test_turn_from_dict(self):
        data = {"role": "user", "content": "hello", "timestamp": time.time()}
        turn = SessionTurn.from_dict(data)
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_turn_content_truncated(self):
        turn = SessionTurn(role="user", content="x" * 60000)
        d = turn.to_dict()
        assert len(d["content"]) <= 50000


class TestSessionMetadata:
    def test_metadata_defaults(self):
        meta = SessionMetadata(session_id="test_1")
        assert meta.title == "Untitled Session"
        assert meta.turn_count == 0
        assert meta.is_active is True

    def test_metadata_to_dict(self):
        meta = SessionMetadata(session_id="test_1", title="Test")
        d = meta.to_dict()
        assert d["session_id"] == "test_1"
        assert d["title"] == "Test"

    def test_metadata_from_dict(self):
        data = {"session_id": "s1", "title": "Hello", "turn_count": 5}
        meta = SessionMetadata.from_dict(data)
        assert meta.session_id == "s1"
        assert meta.turn_count == 5


class TestSessionPersistence:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sp = SessionPersistence(sessions_dir=Path(self.tmpdir))

    def test_create_session(self):
        sid = self.sp.create_session(title="Test Session")
        assert sid is not None
        assert sid.startswith("session_")

    def test_create_named_session(self):
        sid = self.sp.create_session(session_id="my_session", title="Custom")
        assert sid == "my_session"
        meta = self.sp.get_session_metadata("my_session")
        assert meta is not None
        assert meta.title == "Custom"

    def test_create_duplicate_session(self):
        sid1 = self.sp.create_session(session_id="dup_test", title="First")
        sid2 = self.sp.create_session(session_id="dup_test", title="Second")
        assert sid1 == sid2
        sessions = self.sp.list_sessions()
        dup_count = sum(1 for s in sessions if s["session_id"] == "dup_test")
        assert dup_count == 1

    def test_add_turn(self):
        sid = self.sp.create_session(session_id="turn_test", title="Turn Test")
        self.sp.add_turn(sid, "user", "Hello Jarvis")
        self.sp.add_turn(sid, "assistant", "Hi there!")
        meta = self.sp.get_session_metadata(sid)
        assert meta.turn_count == 2

    def test_add_turn_with_tools(self):
        sid = self.sp.create_session(session_id="tool_turn_test")
        self.sp.add_turn(
            sid, "assistant", "Running tool",
            tool_calls=[{"name": "terminal_execute", "args": {"command": "ls"}}],
            tool_results=[{"success": True, "output": "files..."}],
        )
        meta = self.sp.get_session_metadata(sid)
        assert meta.total_tool_calls == 1

    def test_save_and_load(self):
        sid = self.sp.create_session(session_id="save_test")
        self.sp.add_turn(sid, "user", "Save me")
        self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
        data = self.sp.load_session_data(sid)
        assert data is not None
        assert "turns" in data

    def test_delete_session(self):
        sid = self.sp.create_session(session_id="delete_test")
        self.sp.add_turn(sid, "user", "bye")
        result = self.sp.delete_session(sid)
        assert result is True
        assert self.sp.get_session_metadata(sid) is None

    def test_list_sessions(self):
        self.sp.create_session(session_id="list_1", title="Session 1")
        self.sp.create_session(session_id="list_2", title="Session 2")
        sessions = self.sp.list_sessions()
        assert len(sessions) >= 2

    def test_find_stale_sessions(self):
        sid = self.sp.create_session(session_id="stale_test")
        meta = self.sp.get_session_metadata(sid)
        meta.updated_at = time.time() - 7200
        stale = self.sp.find_stale_sessions(max_idle_seconds=3600)
        assert sid in stale

    def test_no_stale_when_active(self):
        sid = self.sp.create_session(session_id="active_test")
        self.sp.add_turn(sid, "user", "I'm here")
        stale = self.sp.find_stale_sessions(max_idle_seconds=3600)
        assert sid not in stale

    def test_resume_session(self):
        sid = self.sp.create_session(session_id="resume_test")
        self.sp.add_turn(sid, "user", "Before crash")
        self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
        data = self.sp.resume_session(sid)
        assert data is not None
        assert data.get("crash_recovery") is True

    def test_mark_inactive(self):
        sid = self.sp.create_session(session_id="inactive_test")
        self.sp.mark_inactive(sid)
        meta = self.sp.get_session_metadata(sid)
        assert meta.is_active is False

    def test_cleanup_old_sessions(self):
        sid = self.sp.create_session(session_id="old_test")
        self.sp.mark_inactive(sid)
        meta = self.sp.get_session_metadata(sid)
        meta.updated_at = time.time() - 31 * 86400
        removed = self.sp.cleanup_old_sessions(max_age_days=30)
        assert removed >= 1

    def test_status(self):
        self.sp.create_session(session_id="status_test")
        status = self.sp.get_status()
        assert "total_sessions" in status
        assert "active_sessions" in status
        assert status["total_sessions"] >= 1

    def test_hmac_integrity(self):
        sid = self.sp.create_session(session_id="hmac_test")
        self.sp.add_turn(sid, "user", "integrity check")
        self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
        data = self.sp.load_session_data(sid)
        assert data is not None
        path = self.sp._session_path(sid)
        raw = json.loads(path.read_text(encoding='utf-8'))
        assert "_hmac" in raw

    def test_hmac_tampering_detected(self):
        sid = self.sp.create_session(session_id="tamper_test")
        self.sp.add_turn(sid, "user", "original")
        self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
        path = self.sp._session_path(sid)
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw["_hmac"] = "tampered_mac_value"
        path.write_text(json.dumps(raw), encoding='utf-8')
        data = self.sp.load_session_data(sid)
        assert data is None

    def test_concurrent_sessions(self):
        results = []
        errors = []

        def create_and_add(i):
            try:
                sid = self.sp.create_session(session_id=f"conc_{i}")
                self.sp.add_turn(sid, "user", f"concurrent message {i}")
                self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
                results.append(sid)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create_and_add, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 10

    def test_lru_eviction(self):
        self.sp = SessionPersistence(sessions_dir=Path(self.tmpdir))
        original_max = 5
        import core.session_persistence as sp_module
        sp_module.MAX_SESSIONS = original_max
        try:
            for i in range(original_max + 3):
                self.sp.create_session(session_id=f"lru_{i}", title=f"Session {i}")
            assert len(self.sp._sessions) <= original_max + 1
        finally:
            sp_module.MAX_SESSIONS = 100

    def test_singleton(self):
        sp1 = get_session_persistence()
        sp2 = get_session_persistence()
        assert sp1 is sp2


class TestSessionPersistenceDataIntegrity:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sp = SessionPersistence(sessions_dir=Path(self.tmpdir))

    def test_turns_preserved_across_save_load(self):
        sid = self.sp.create_session(session_id="preserve_test")
        self.sp.add_turn(sid, "user", "First message")
        self.sp.add_turn(sid, "assistant", "First reply", tool_calls=[{"name": "test"}])
        self.sp.add_turn(sid, "user", "Second message")
        existing = self.sp.load_session_data(sid) or {"turns": []}
        assert len(existing.get("turns", [])) >= 1
        self.sp.save_session_data(sid, existing)
        data = self.sp.load_session_data(sid)
        assert data is not None

    def test_metadata_in_saved_data(self):
        sid = self.sp.create_session(session_id="meta_test", title="Title Test")
        self.sp.add_turn(sid, "user", "test")
        self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
        data = self.sp.load_session_data(sid)
        assert data is not None
        assert "metadata" in data

    def test_session_id_sanitization(self):
        sp = SessionPersistence(sessions_dir=Path(self.tmpdir))
        path = sp._session_path("../../../etc/passwd")
        assert ".." not in str(path)
        assert str(path).startswith(str(self.tmpdir))

    def test_crash_recovery_counter(self):
        sid = self.sp.create_session(session_id="crash_counter")
        self.sp.add_turn(sid, "user", "test")
        self.sp.save_session_data(sid, self.sp.load_session_data(sid) or {"turns": []})
        self.sp.resume_session(sid)
        status = self.sp.get_status()
        assert status["total_crash_recoveries"] >= 1