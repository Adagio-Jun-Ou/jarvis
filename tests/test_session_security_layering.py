"""Regression tests for the session_security advisory layering.

core/security/session_security.py implements five capabilities that
auth_middleware lacked (user-agent binding, concurrent-session limits,
revocation, fixation prevention, progressive brute-force lockout). It
is now layered into SessionManager following the SSRF protector's rules:

- Advisory only: it can ADD a denial (brute-force lockout refuses new
  session creation), never grant anything.
- Fail-open: if the layer is unavailable or raises, sessions work as
  before — a bug there can never lock the operator out.
- Contract preserved: callers still see Session objects and the same
  (is_valid, session, error) tuples; the one change is that
  create_session may raise PermissionError when the layer refuses,
  and validate_websocket_connection converts that to (False, None, msg).

The two other capabilities the layer owns (user-agent binding, its own
concurrent limits) apply to its OWN session store; wiring them into
SessionManager's store requires the caller contract to carry
user_agent, which is the "live auth path" change deferred to product
sign-off. What IS wired today: per-IP progressive brute-force lockout
(failed validations feed it; creation consults it; success clears it).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mw(require_auth=False, **kw):
    from core.auth_middleware import SecurityConfig, SecurityMiddleware

    cfg = SecurityConfig(require_auth=require_auth, **kw)
    return SecurityMiddleware(cfg)


def _reset_layer():
    """Fresh session_security singleton per test (it has global state)."""
    import core.security.session_security as ss

    ss._session_security = None
    return ss


class TestAdvisoryLayering:
    def test_layer_fails_open_when_unavailable(self, monkeypatch):
        """A broken/unavailable layer must never block session creation."""
        mw = _mw()
        monkeypatch.setattr(
            "core.security.session_security.get_session_security",
            lambda: (_ for _ in ()).throw(ImportError("gone")),
        )
        mw.session_manager._sec_layer = None
        mw.session_manager._sec_layer_failed = False
        session = mw.session_manager.create_session(client_ip="10.0.0.1")
        assert session.session_id

    def test_failed_validations_feed_progressive_lockout(self):
        """The incident shape: an IP hammering failed attempts gets new
        session creation refused — across sessions, not per-session."""
        ss = _reset_layer()
        mw = _mw(max_failed_attempts=1)

        # Create a session, then fail validations against it repeatedly.
        s1 = mw.session_manager.create_session(client_ip="203.0.113.9")
        for _ in range(12):
            mw.session_manager.record_failed_attempt(s1.session_id)

        # The layer's per-IP lockout must now refuse creation from that IP.
        with pytest.raises(PermissionError):
            mw.session_manager.create_session(client_ip="203.0.113.9")

    def test_successful_creation_clears_failure_streak(self):
        ss = _reset_layer()
        mw = _mw()

        # A few failures, then a successful login
        s1 = mw.session_manager.create_session(client_ip="203.0.113.10")
        for _ in range(3):
            mw.session_manager.record_failed_attempt(s1.session_id)
        s2 = mw.session_manager.create_session(client_ip="203.0.113.10")  # success
        assert s2.session_id != s1.session_id

    def test_lockout_is_per_ip(self):
        ss = _reset_layer()
        mw = _mw(max_failed_attempts=1)

        s1 = mw.session_manager.create_session(client_ip="203.0.113.11")
        for _ in range(12):
            mw.session_manager.record_failed_attempt(s1.session_id)

        # Same IP refused...
        with pytest.raises(PermissionError):
            mw.session_manager.create_session(client_ip="203.0.113.11")
        # ...different IP unaffected.
        s2 = mw.session_manager.create_session(client_ip="198.51.100.5")
        assert s2.session_id

    def test_loopback_not_special_cased_in_layer(self):
        """No loopback exemption in the layer's lockout — but note the
        local operator creates sessions without an IP in some paths,
        which must never be refused (None IP skips the advisory check)."""
        ss = _reset_layer()
        mw = _mw()
        session = mw.session_manager.create_session(client_ip=None)
        assert session.session_id


class TestContractPreserved:
    def test_validate_session_still_returns_same_tuple(self):
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(client_ip="192.0.2.1")
        ok, session, err = mw.session_manager.validate_session(s.session_id, "192.0.2.1")
        assert ok is True and session is s and err is None

    def test_validate_rejects_bad_id_as_before(self):
        mw = _mw()
        ok, session, err = mw.session_manager.validate_session("nonexistent", "1.2.3.4")
        assert ok is False and session is None and err

    def test_ws_connection_validation_converts_refusal(self):
        """validate_websocket_connection must turn the layer's PermissionError
        into a clean refusal tuple, not a 500."""
        ss = _reset_layer()
        mw = _mw()
        s1 = mw.session_manager.create_session(client_ip="203.0.113.12")
        for _ in range(12):
            mw.session_manager.record_failed_attempt(s1.session_id)

        ok, result, err = mw.validate_websocket_connection(
            headers={}, query_params={}, client_ip="203.0.113.12")
        assert ok is False
        assert "locked" in err.lower() or "attempt" in err.lower()


class TestUserAgentBinding:
    """The hijack-detection capability, now wired end to end.

    A session created with a user-agent pins a truncated snippet; a later
    validation presenting a DIFFERENT user-agent (cookie stolen to another
    browser) is refused. The comparison policy is the security layer's
    _verify_user_agent (allows minor version drift, refuses different
    browsers); the store is SessionManager's own — no dual-store tokens.
    """

    def test_creation_pins_ua_snippet(self):
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(
            client_ip="192.0.2.7", metadata={"user_agent": "Mozilla/5.0 Chrome/120.0"})
        assert s.metadata["user_agent_snippet"] == "Mozilla/5.0 Chrome/120.0"

    def test_same_ua_validates(self):
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(
            client_ip="192.0.2.8", metadata={"user_agent": "Mozilla/5.0 Chrome/120.0"})
        ok, sess, err = mw.session_manager.validate_session(
            s.session_id, "192.0.2.8", user_agent="Mozilla/5.0 Chrome/120.0.1")
        assert ok is True and err is None

    def test_different_ua_refused_as_hijack(self):
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(
            client_ip="192.0.2.9", metadata={"user_agent": "Mozilla/5.0 Chrome/120.0"})
        ok, sess, err = mw.session_manager.validate_session(
            s.session_id, "192.0.2.9", user_agent="python-requests/2.31")
        assert ok is False
        assert "user-agent mismatch" in err.lower()

    def test_no_ua_at_validation_degrades_open(self):
        """Old clients that don't present a UA still validate (advisory)."""
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(
            client_ip="192.0.2.10", metadata={"user_agent": "Mozilla/5.0"})
        ok, sess, err = mw.session_manager.validate_session(s.session_id, "192.0.2.10")
        assert ok is True

    def test_session_without_pinned_ua_never_fails_on_ua(self):
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(client_ip="192.0.2.11")
        ok, _, err = mw.session_manager.validate_session(
            s.session_id, "192.0.2.11", user_agent="python-requests/2.31")
        assert ok is True

    def test_ws_connection_pins_and_verifies_ua(self):
        ss = _reset_layer()
        mw = _mw()
        headers = {"user-agent": "Mozilla/5.0 Chrome/120.0"}
        ok, session_id, err = mw.validate_websocket_connection(
            headers=headers, query_params={}, client_ip="192.0.2.12")
        assert ok is True
        session = mw.session_manager._sessions[session_id]
        assert session.metadata["user_agent_snippet"].startswith("Mozilla/5.0")

    def test_ua_snippet_truncated_to_200(self):
        ss = _reset_layer()
        mw = _mw()
        long_ua = "X" * 500
        s = mw.session_manager.create_session(
            client_ip="192.0.2.13", metadata={"user_agent": long_ua})
        assert len(s.metadata["user_agent_snippet"]) == 200


class TestConcurrentSessionLimit:
    """The third capability: an IP holding more live sessions than the
    layer's max_concurrent_sessions gets new creations refused. Accounting
    uses SessionManager's own per-IP live set; loopback is exempt so the
    local operator can never lock themselves out."""

    def test_excess_sessions_refused(self):
        ss = _reset_layer()
        mw = _mw()
        limit = mw.session_manager._security_layer().max_concurrent_sessions  # default 5

        for i in range(limit):
            mw.session_manager.create_session(client_ip="203.0.113.20")
        with pytest.raises(PermissionError, match="[Cc]oncurrent"):
            mw.session_manager.create_session(client_ip="203.0.113.20")

    def test_ended_sessions_free_capacity(self):
        ss = _reset_layer()
        mw = _mw()
        limit = mw.session_manager._security_layer().max_concurrent_sessions

        sessions = [mw.session_manager.create_session(client_ip="203.0.113.21")
                    for _ in range(limit)]
        with pytest.raises(PermissionError):
            mw.session_manager.create_session(client_ip="203.0.113.21")

        mw.session_manager.end_session(sessions[0].session_id)
        s = mw.session_manager.create_session(client_ip="203.0.113.21")
        assert s.session_id

    def test_loopback_exempt(self):
        """The local operator opens multiple tabs/CLIs — never refused."""
        ss = _reset_layer()
        mw = _mw()
        for _ in range(10):
            s = mw.session_manager.create_session(client_ip="127.0.0.1")
            assert s.session_id

    def test_no_ip_not_limited(self):
        ss = _reset_layer()
        mw = _mw()
        for _ in range(8):
            s = mw.session_manager.create_session(client_ip=None)
            assert s.session_id

    def test_limit_is_per_ip(self):
        ss = _reset_layer()
        mw = _mw()
        limit = mw.session_manager._security_layer().max_concurrent_sessions

        for _ in range(limit):
            mw.session_manager.create_session(client_ip="203.0.113.22")
        # Different IP unaffected
        s = mw.session_manager.create_session(client_ip="198.51.100.9")
        assert s.session_id


class TestRevocationAndFixation:
    """The fourth + fifth capabilities: explicit revocation (with memory of
    the revocation) and fixation prevention (structural — ids are always
    server-generated, never client-supplied)."""

    def test_revoke_then_validate_reports_revoked(self):
        ss = _reset_layer()
        mw = _mw()
        s = mw.session_manager.create_session(client_ip="192.0.2.30")
        assert mw.session_manager.revoke_session(s.session_id) is True
        ok, _, err = mw.session_manager.validate_session(s.session_id, "192.0.2.30")
        assert ok is False
        assert err == "Session revoked"

    def test_revoke_unknown_session_returns_false(self):
        ss = _reset_layer()
        mw = _mw()
        assert mw.session_manager.revoke_session("never-existed") is False

    def test_revoke_ip_sessions_bulk(self):
        ss = _reset_layer()
        mw = _mw()
        a = mw.session_manager.create_session(client_ip="192.0.2.31")
        b = mw.session_manager.create_session(client_ip="192.0.2.31")
        c = mw.session_manager.create_session(client_ip="192.0.2.32")  # other IP
        count = mw.session_manager.revoke_ip_sessions("192.0.2.31")
        assert count == 2
        for sid in (a.session_id, b.session_id):
            ok, _, err = mw.session_manager.validate_session(sid, "192.0.2.31")
            assert ok is False and err == "Session revoked"
        # Other IP's session unaffected
        ok, _, _ = mw.session_manager.validate_session(c.session_id, "192.0.2.32")
        assert ok is True

    def test_revocation_memory_is_bounded(self):
        ss = _reset_layer()
        mw = _mw()
        mw.session_manager._MAX_REVOKED_IDS = 10
        ids = []
        for i in range(12):
            s = mw.session_manager.create_session(client_ip=None)
            mw.session_manager.revoke_session(s.session_id)
            ids.append(s.session_id)
        assert len(mw.session_manager._revoked_ids) == 10
        # Oldest revocations forgotten
        assert ids[0] not in mw.session_manager._revoked_ids
        assert ids[-1] in mw.session_manager._revoked_ids

    def test_fixation_prevented_server_generated_ids_only(self):
        """Session ids are always server-generated secrets; a client cannot
        fix a chosen id. Structural guarantee — create twice, ids differ and
        match token_urlsafe entropy."""
        ss = _reset_layer()
        mw = _mw()
        s1 = mw.session_manager.create_session(client_ip="192.0.2.33")
        s2 = mw.session_manager.create_session(client_ip="192.0.2.33")
        assert s1.session_id != s2.session_id
        # token_urlsafe(32) -> 43-char base64
        assert len(s1.session_id) >= 40
        # There is no API to create a session with a caller-chosen id
        import inspect
        from core.auth_middleware import SessionManager
        sig = inspect.signature(SessionManager.create_session)
        assert "session_id" not in sig.parameters