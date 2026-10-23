"""Session Security Hardening for Jarvis.

Provides comprehensive session security including:
- Secure session token generation (cryptographic)
- Session timeout and expiry enforcement
- Concurrent session limiting
- Session fixation prevention
- IP binding and user-agent verification
- Session revocation and audit trail
- Brute-force protection with progressive delays
"""

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    created_at: float
    last_accessed: float
    ip_address: str | None = None
    user_agent: str | None = None
    is_revoked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_accessed

    def is_expired(self, max_age_seconds: float, max_idle_seconds: float) -> bool:
        if self.age_seconds > max_age_seconds:
            return True
        if self.idle_seconds > max_idle_seconds:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "ip_address": self.ip_address,
            "user_agent_hash": hashlib.sha256(self.user_agent.encode()).hexdigest()[:16] if self.user_agent else None,
            "is_revoked": self.is_revoked,
            "age_seconds": round(self.age_seconds, 1),
            "idle_seconds": round(self.idle_seconds, 1),
        }


@dataclass
class LoginAttempt:
    ip_address: str
    user_id: str | None
    timestamp: float
    success: bool
    failure_reason: str | None = None


class SessionSecurityManager:
    """Comprehensive session security management."""

    DEFAULT_MAX_AGE_SECONDS = 86400  # 24 hours
    DEFAULT_MAX_IDLE_SECONDS = 3600  # 1 hour
    DEFAULT_MAX_CONCURRENT_SESSIONS = 5
    DEFAULT_MAX_LOGIN_ATTEMPTS = 10
    DEFAULT_LOGIN_WINDOW_SECONDS = 900  # 15 minutes
    DEFAULT_LOCKOUT_DURATION_SECONDS = 300  # 5 minutes
    TOKEN_BYTES = 32

    def __init__(self):
        self._sessions: dict[str, SessionInfo] = {}
        self._token_to_session: dict[str, str] = {}
        self._user_sessions: dict[str, list[str]] = defaultdict(list)
        self._login_attempts: dict[str, list[LoginAttempt]] = defaultdict(list)
        self._lockouts: dict[str, float] = {}
        self._revoked_tokens: set[str] = set()
        self._hmac_key = os.urandom(32)
        self._lock = threading.RLock()

        self.max_age_seconds = self.DEFAULT_MAX_AGE_SECONDS
        self.max_idle_seconds = self.DEFAULT_MAX_IDLE_SECONDS
        self.max_concurrent_sessions = self.DEFAULT_MAX_CONCURRENT_SESSIONS
        self.max_login_attempts = self.DEFAULT_MAX_LOGIN_ATTEMPTS
        self.login_window_seconds = self.DEFAULT_LOGIN_WINDOW_SECONDS
        self.lockout_duration_seconds = self.DEFAULT_LOCKOUT_DURATION_SECONDS

    def create_session(
        self,
        user_id: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Create a new secure session. Returns (session_id, session_token).
        
        The session_token should be given to the client.
        The session_id is used internally for tracking.
        """
        session_id = secrets.token_hex(16)
        raw_token = secrets.token_bytes(self.TOKEN_BYTES)
        token_hex = raw_token.hex()

        user_agent_snippet = user_agent[:200] if user_agent else None

        session = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            last_accessed=time.time(),
            ip_address=ip_address,
            user_agent=user_agent_snippet,
            metadata=metadata or {},
        )

        with self._lock:
            self._evict_expired_sessions()

            user_session_ids = self._user_sessions.get(user_id, [])
            active = [
                sid for sid in user_session_ids
                if sid in self._sessions and not self._sessions[sid].is_revoked
                and not self._sessions[sid].is_expired(self.max_age_seconds, self.max_idle_seconds)
            ]

            while len(active) >= self.max_concurrent_sessions:
                oldest_sid = active[0]
                if oldest_sid in self._sessions:
                    self._sessions[oldest_sid].is_revoked = True
                active = active[1:]

            self._sessions[session_id] = session
            self._token_to_session[token_hex] = session_id
            self._user_sessions[user_id] = active + [session_id]

        logger.info(f"Session created for user '{user_id}' from {ip_address}")
        self._record_login_attempt(ip_address, user_id, True)

        return session_id, token_hex

    def validate_session(
        self,
        session_token: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> tuple[bool, SessionInfo | None, str]:
        """Validate a session token. Returns (valid, session_info, error_message)."""
        with self._lock:
            session_id = self._token_to_session.get(session_token)
            if session_id is None:
                return False, None, "Invalid session token"

            session = self._sessions.get(session_id)
            if session is None:
                return False, None, "Session not found"

            if session.is_revoked:
                return False, None, "Session revoked"

            if session.is_expired(self.max_age_seconds, self.max_idle_seconds):
                if session.age_seconds > self.max_age_seconds:
                    return False, None, "Session expired (max age)"
                else:
                    return False, None, "Session expired (idle timeout)"

            if ip_address and session.ip_address:
                if ip_address != session.ip_address:
                    logger.warning(
                        f"IP mismatch for session {session.session_id}: "
                        f"expected {session.ip_address}, got {ip_address}"
                    )
                    return False, None, "IP address mismatch"

            if user_agent and session.user_agent:
                if not self._verify_user_agent(user_agent, session.user_agent):
                    logger.warning(
                        f"User-agent mismatch for session {session.session_id}"
                    )
                    return False, None, "User-agent mismatch"

            session.last_accessed = time.time()
            return True, session, ""

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_revoked = True
                logger.info(f"Session {session_id} revoked for user '{session.user_id}'")
                return True
        return False

    def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user. Returns count revoked."""
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            count = 0
            for sid in session_ids:
                session = self._sessions.get(sid)
                if session and not session.is_revoked:
                    session.is_revoked = True
                    count += 1
            return count

    def check_login_allowed(self, ip_address: str) -> tuple[bool, str]:
        """Check if a login attempt is allowed (not locked out).
        
        Returns (is_allowed, reason).
        """
        now = time.time()
        with self._lock:
            lockout_until = self._lockouts.get(ip_address)
            if lockout_until and now < lockout_until:
                remaining = int(lockout_until - now)
                return False, f"Account locked for {remaining} more seconds"

            attempts = self._login_attempts.get(ip_address, [])
            recent = [a for a in attempts if now - a.timestamp < self.login_window_seconds]
            recent_failures = [a for a in recent if not a.success]

            if len(recent_failures) >= self.max_login_attempts:
                lockout_seconds = min(
                    self.lockout_duration_seconds * (1 + len(recent_failures) // self.max_login_attempts),
                    3600,
                )
                self._lockouts[ip_address] = now + lockout_seconds
                return False, f"Too many failed attempts. Locked for {lockout_seconds}s"

            return True, ""

    def _record_login_attempt(self, ip_address: str, user_id: str | None, success: bool, reason: str | None = None) -> None:
        """Record a login attempt."""
        now = time.time()
        attempt = LoginAttempt(
            ip_address=ip_address,
            user_id=user_id,
            timestamp=now,
            success=success,
            failure_reason=reason,
        )
        with self._lock:
            self._login_attempts[ip_address].append(attempt)
            attempts = self._login_attempts[ip_address]
            self._login_attempts[ip_address] = [
                a for a in attempts if now - a.timestamp < self.login_window_seconds * 2
            ]

    def _hash_token(self, token: str) -> str:
        """Hash a session token for comparison."""
        return hmac.new(self._hmac_key, token.encode(), hashlib.sha256).hexdigest()

    def _verify_user_agent(self, current: str, stored: str) -> bool:
        """Verify user-agent consistency (allows minor version changes)."""
        current_parts = current.lower().split()
        stored_parts = stored.lower().split()
        if len(current_parts) != len(stored_parts):
            return False
        for cur, sto in zip(current_parts, stored_parts):
            if cur == sto:
                continue
            prefix_len = min(len(cur), len(sto)) // 2
            if cur[:prefix_len] != sto[:prefix_len]:
                return False
        return True

    def _evict_expired_sessions(self) -> None:
        """Remove expired and revoked sessions."""
        to_remove = []
        for sid, session in self._sessions.items():
            if session.is_revoked or session.is_expired(self.max_age_seconds, self.max_idle_seconds):
                to_remove.append(sid)

        for sid in to_remove:
            session = self._sessions.pop(sid)
            tokens_to_remove = [t for t, s in self._token_to_session.items() if s == sid]
            for t in tokens_to_remove:
                del self._token_to_session[t]
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    s for s in self._user_sessions[session.user_id] if s != sid
                ]

    def get_active_sessions(self, user_id: str | None = None) -> list[SessionInfo]:
        """Get active sessions, optionally filtered by user."""
        with self._lock:
            self._evict_expired_sessions()
            sessions = list(self._sessions.values())
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]
            return [s for s in sessions if not s.is_revoked]

    def get_stats(self) -> dict[str, Any]:
        """Get session security statistics."""
        with self._lock:
            self._evict_expired_sessions()
            total = len(self._sessions)
            active = sum(1 for s in self._sessions.values() if not s.is_revoked)
            users = len(set(s.user_id for s in self._sessions.values() if not s.is_revoked))
            lockouts = len(self._lockouts)
            recent_attempts = 0
            now = time.time()
            for attempts in self._login_attempts.values():
                recent_attempts += sum(1 for a in attempts if now - a.timestamp < self.login_window_seconds)
            return {
                "total_sessions": total,
                "active_sessions": active,
                "unique_users": users,
                "active_lockouts": lockouts,
                "recent_login_attempts": recent_attempts,
                "max_concurrent_per_user": self.max_concurrent_sessions,
                "max_age_seconds": self.max_age_seconds,
                "max_idle_seconds": self.max_idle_seconds,
            }


_session_security: SessionSecurityManager | None = None
_session_security_lock = threading.Lock()


def get_session_security() -> SessionSecurityManager:
    """Thread-safe singleton for SessionSecurityManager."""
    global _session_security
    if _session_security is None:
        with _session_security_lock:
            if _session_security is None:
                _session_security = SessionSecurityManager()
    return _session_security