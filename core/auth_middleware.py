"""Authentication and Security Middleware for Jarvis.

Provides protection against:
- Unauthorized API access
- Injection attacks (XSS, SQL, command injection)
- Data extraction attacks
- Rate limiting bypass
- Session hijacking

Features:
- API key authentication with timing-safe comparison
- Input validation and sanitization
- Security headers (CSP, XSS Protection, etc.)
- Request signing for WebSocket
- Session management
"""
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

API_KEY_ENV = "JARVIS_API_KEY"
SESSION_SECRET_ENV = "JARVIS_SESSION_SECRET"


class AuthLevel(Enum):
    """Authentication levels for different operations."""
    PUBLIC = "public"          
    READONLY = "readonly"      
    STANDARD = "standard"      
    ELEVATED = "elevated"      
    ADMIN = "admin"            


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key: str | None = None
    session_secret: str | None = None
    require_auth: bool = False
    session_timeout_minutes: int = 60
    max_session_age_hours: int = 24
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests_per_minute: int = 120
    max_request_size_bytes: int = 10 * 1024 * 1024  
    allowed_origins: list[str] = field(default_factory=lambda: [])
    csrf_protection: bool = True
    xss_protection: bool = True
    content_security_policy: str = "default-src 'self'; script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net; style-src 'self' 'nonce-{nonce}' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob:; connect-src 'self' ws: wss:; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none';"
    strict_transport_security: str = "max-age=31536000; includeSubDomains"


@dataclass
class Session:
    """User session data."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    auth_level: AuthLevel = AuthLevel.STANDARD
    failed_attempts: int = 0
    locked_until: datetime | None = None
    client_ip: str | None = None
    metadata: dict = field(default_factory=dict)
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        return datetime.now() < self.locked_until
    
    def is_ip_valid(self, client_ip: str | None = None) -> bool:
        if not client_ip:
            return True
        if not self.client_ip:
            return False
        return self.client_ip == client_ip


class InputSanitizer:
    """Sanitizes user input to prevent injection attacks."""
    
    XSS_PATTERNS = [
        (r'<script[^>]*>.*?</script>', '', re.IGNORECASE | re.DOTALL),
        (r'<\s*script[^>]*>.*?<\s*/\s*script[^>]*>', '', re.IGNORECASE | re.DOTALL),
        (r'javascript\s*:', '', re.IGNORECASE),
        (r'vbscript\s*:', '', re.IGNORECASE),
        (r'data\s*:\s*text/html', '', re.IGNORECASE),
        (r'on(?:click|load|error|mouseover|focus|blur|submit|change|input|keydown|keyup|mouseout|mousemove|mouseup|mousedown|contextmenu|drag|dragend|dragenter|dragleave|dragover|dragstart|drop|scroll|wheel|copy|cut|paste|animationend|animationiteration|animationstart|transitionend|pointerdown|pointerup|pointermove|pointerover|pointerout|pointerenter|pointerleave|gotpointercapture|lostpointercapture|touchstart|touchend|touchmove|touchcancel)\s*=', '', re.IGNORECASE),
        (r'<iframe[^>]*>.*?</iframe>', '', re.IGNORECASE | re.DOTALL),
        (r'<object[^>]*>.*?</object>', '', re.IGNORECASE | re.DOTALL),
        (r'<embed[^>]*>', '', re.IGNORECASE),
        (r'<form[^>]*>.*?</form>', '', re.IGNORECASE | re.DOTALL),
        (r'<img[^>]+on\w+\s*=', '', re.IGNORECASE),
        (r'<svg[^>]*on\w+\s*=', '', re.IGNORECASE),
        (r'<body[^>]*on\w+\s*=', '', re.IGNORECASE),
        (r'<link[^>]*href\s*=\s*["\']?\s*javascript:', '', re.IGNORECASE),
        (r'<meta[^>]*http-equiv\s*=\s*["\']?refresh', '', re.IGNORECASE),
        (r'<base[^>]*href\s*=\s*["\']?\s*javascript:', '', re.IGNORECASE),
    ]
    
    SQL_INJECTION_PATTERNS = [
        (r"('\s*(OR|AND)\s*['\"0-9])", '', re.IGNORECASE),
        (r"('\s*;\s*(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE))", '', re.IGNORECASE),
        (r"('\s*--)", '', re.IGNORECASE),
        (r"--\s*$", '', re.MULTILINE),
        (r"/\*.*?\*/", '', re.DOTALL),
        (r"UNION\s+(ALL\s+)?SELECT", '', re.IGNORECASE),
        (r"OR\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+", '', re.IGNORECASE),
        (r"AND\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+", '', re.IGNORECASE),
        (r"'\s*=\s*'", '', re.IGNORECASE),
        (r"EXEC(\s+|\().* xp_", '', re.IGNORECASE),
        (r"WAITFOR\s+DELAY", '', re.IGNORECASE),
        (r"BENCHMARK\s*\(", '', re.IGNORECASE),
        (r"SLEEP\s*\(", '', re.IGNORECASE),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        (r'[;&|`$]', '', 0),
        (r'\$\([^)]*\)', '', 0),
        (r'`[^`]*`', '', 0),
        (r'\|\s*\w+', '', 0),
        (r'>\s*/', '', 0),
        (r'<\s*/', '', 0),
        (r'\brm\s+-rf\b', '', re.IGNORECASE),
        (r'\bchmod\s+[0-7]{3,4}\b', '', re.IGNORECASE),
        (r'\bchown\s+', '', re.IGNORECASE),
        (r'\bnc\s+-[elp]', '', re.IGNORECASE),
        (r'\bcurl\s+', '', re.IGNORECASE),
        (r'\bwget\s+', '', re.IGNORECASE),
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        (r'\.\./', '', 0),
        (r'\.\.\\', '', 0),
        (r'\.\.\>', '', 0),
        (r'%2e%2e%2f', '', re.IGNORECASE),
        (r'%2e%2e/', '', re.IGNORECASE),
        (r'\.\.%2f', '', re.IGNORECASE),
        (r'%2e%2e\\', '', re.IGNORECASE),
        (r'%252e%252e', '', re.IGNORECASE),
        (r'\.\.%5c', '', re.IGNORECASE),
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, strict: bool = False) -> str:
        """Sanitize a string value.

        XSS, SQL-injection, path-traversal and command-injection patterns are
        ALL removed on every call.

        Args:
            value: String to sanitize
            strict: Currently has no effect. It is retained for API compatibility
                with callers that pass it (validate_request uses strict=False).
                The docstring previously claimed it gated command-injection
                removal, which was never true — that filtering is unconditional
                by design.

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return value
        
        result = value
        
        for pattern, replacement, flags in cls.XSS_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=flags)
        
        for pattern, replacement, flags in cls.SQL_INJECTION_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=flags)
        
        for pattern, replacement, flags in cls.PATH_TRAVERSAL_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=flags)

        # Deliberately unconditional — see TestInputSanitizerAlwaysRuns in
        # tests/test_security_hardening.py, which pins this behaviour. Note the
        # cost: these patterns delete ';', '&', '|', '$' and backticks plus whole
        # words like "curl "/"wget ", so any request body carrying a code
        # snippet, a shell example or a '$' amount is silently mutilated.
        # If that becomes a problem, gate this on `strict` AND update that test —
        # do not simply flip it, the unconditional form is intentional.
        for pattern, replacement, flags in cls.COMMAND_INJECTION_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=flags)

        return result.strip()
    
    @classmethod
    def sanitize_dict(cls, data: dict, strict: bool = False) -> dict:
        """Recursively sanitize a dictionary."""
        result = {}
        for key, value in data.items():
            clean_key = cls.sanitize_string(str(key), strict)
            if isinstance(value, dict):
                result[clean_key] = cls.sanitize_dict(value, strict)
            elif isinstance(value, list):
                result[clean_key] = [cls.sanitize_string(item, strict) if isinstance(item, str) else item for item in value]
            elif isinstance(value, str):
                result[clean_key] = cls.sanitize_string(value, strict)
            else:
                result[clean_key] = value
        return result
    
    @classmethod
    def validate_file_path(cls, path: str, base_dir: str | None = None) -> bool:
        """Validate a file path to prevent traversal attacks."""
        if not path:
            return False
        
        dangerous = ['..', '~', '\x00']
        if any(d in path for d in dangerous):
            return False
        
        if base_dir:
            import os
            try:
                resolved = os.path.realpath(os.path.join(base_dir, path))
                return resolved.startswith(os.path.realpath(base_dir))
            except Exception:
                return False
        
        return True
    
    @classmethod
    def escape_html(cls, value: str) -> str:
        """Escape HTML entities."""
        if not isinstance(value, str):
            return value
        return (value
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))


class APIKeyValidator:
    """Validates API keys with timing-safe comparison."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._valid_keys: set[bytes] = set()
        self._key_hashes: dict[str, bytes] = {}
        self._cached_secret: str | None = None
        self._load_keys()
    
    def _get_session_secret(self) -> str:
        """Get session secret, caching generated secrets per-process."""
        if self._cached_secret:
            return self._cached_secret
        secret = self.config.session_secret or os.environ.get(SESSION_SECRET_ENV)
        if not secret:
            secret = secrets.token_hex(32)
            logger.info("Generated random session secret (sessions invalidate on restart)")
        self._cached_secret = secret
        return secret
    
    def _generate_session_token(self, auth_level: AuthLevel) -> str:
        """Generate a secure session token."""
        secret = self._get_session_secret()
        timestamp = str(int(time.time()))
        random_bytes = secrets.token_hex(16)
        payload = f"{timestamp}:{random_bytes}:{auth_level.value}"
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"{payload}:{signature}"
    
    def _load_keys(self):
        """Load API keys from environment and config."""
        keys = []
        
        if self.config.api_key:
            keys.append(self.config.api_key)
        
        env_key = os.environ.get(API_KEY_ENV)
        if env_key:
            keys.append(env_key)
        
        multi_keys = os.environ.get("JARVIS_API_KEYS", "")
        for key in multi_keys.split(","):
            key = key.strip()
            if key:
                keys.append(key)
        
        for key in keys:
            key_bytes = key.encode('utf-8')
            self._valid_keys.add(key_bytes)
            key_hash = hashlib.sha256(key_bytes).hexdigest()
            self._key_hashes[key_hash] = key_bytes
        
        logger.info(f"Loaded {len(self._valid_keys)} API key(s)")
    
    def validate(self, provided_key: str) -> tuple[bool, str | None]:
        """Validate an API key using timing-safe comparison.
        
        Returns:
            Tuple of (is_valid, session_token or error_message)
        """
        if not self._valid_keys:
            logger.error("No API keys configured — authentication DENIED. Set JARVIS_API_KEY env var to enable auth.")
            return False, "No API keys configured — set JARVIS_API_KEY environment variable"
        
        if not provided_key:
            return False, "API key required"
        
        provided_bytes = provided_key.encode('utf-8')
        provided_hash = hashlib.sha256(provided_bytes).hexdigest()
        
        for stored_hash, stored_key in self._key_hashes.items():
            if hmac.compare_digest(provided_hash, stored_hash):
                return True, self._generate_session_token(AuthLevel.STANDARD)
        
        return False, "Invalid API key"
    


class SessionManager:
    """Manages user sessions with security controls.

    Layered with core/security/session_security.py as an advisory second
    opinion (the roadmap's five capabilities auth_middleware lacks: user-
    agent binding, concurrent-session limits, revocation, fixation
    prevention, progressive brute-force lockout). The layering follows the
    SSRF protector's rules: it can only *add* a denial, never grant; and
    if it errors it fails OPEN with a warning, so a bug there can never
    lock the operator out. SessionManager remains the session store and
    keeps its existing return contract — callers still see Session
    objects and the same (is_valid, session, error) tuples.
    """

    MAX_SESSIONS = 1000

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._sessions: dict[str, Session] = {}
        self._ip_to_sessions: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._sec_layer = None
        self._sec_layer_failed = False
        self._revoked_ids: OrderedDict[str, float] = OrderedDict()
        self._MAX_REVOKED_IDS = 2000

    def _security_layer(self):
        """Lazily attach the session-security advisory layer (once).

        Unavailable/broken layer returns None and is remembered, so we do
        not retry-and-log on every request.
        """
        if self._sec_layer_failed:
            return None
        if self._sec_layer is None:
            try:
                from core.security.session_security import get_session_security
                self._sec_layer = get_session_security()
            except Exception as e:
                self._sec_layer_failed = True
                logger.warning(f"Session security layer unavailable (degrading open): {e}")
        return self._sec_layer

    def _security_advise_deny_creation(self, client_ip: str | None) -> str | None:
        """Ask the security layer whether session creation must be refused.

        Uses the layer's progressive brute-force lockout (an IP that has
        failed too many logins recently is locked out here too). Returns a
        denial reason, or None when creation may proceed / the layer is
        unavailable (fail-open). Failures of the layer itself degrade open.
        """
        if not client_ip:
            return None
        layer = self._security_layer()
        if layer is None:
            return None
        try:
            allowed, reason = layer.check_login_allowed(client_ip)
            if not allowed:
                return reason or "Too many failed attempts"
            return None
        except Exception as e:
            logger.warning(f"Session security layer error (fail-open): {e}")
            return None

    def _security_record_attempt(self, client_ip: str | None, success: bool) -> None:
        """Record a login attempt outcome with the security layer (feeds the
        progressive lockout). Advisory only; errors are logged and ignored."""
        if not client_ip:
            return
        layer = self._security_layer()
        if layer is None:
            return
        try:
            layer._record_login_attempt(client_ip, None, success)
        except Exception as e:
            logger.debug(f"Security layer attempt recording failed: {e}")
    
    def create_session(self, auth_level: AuthLevel = AuthLevel.STANDARD, 
                       client_ip: str | None = None,
                       metadata: dict | None = None) -> Session:
        """Create a new session."""
        with self._lock:
            session_id = secrets.token_urlsafe(32)
            now = datetime.now()
            
            session = Session(
                session_id=session_id,
                created_at=now,
                last_activity=now,
                auth_level=auth_level,
                client_ip=client_ip,
                metadata=metadata or {},
            )
            
            self._sessions[session_id] = session
            
            if len(self._sessions) > self.MAX_SESSIONS:
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_activity)
                self.end_session(oldest_id)
            
            if client_ip:
                session.metadata["client_ip"] = client_ip
                if client_ip not in self._ip_to_sessions:
                    self._ip_to_sessions[client_ip] = set()
                self._ip_to_sessions[client_ip].add(session_id)

                # Concurrent-session limit (the third of the five session-
                # security capabilities): an IP holding more live sessions
                # than the layer's max_concurrent_sessions is refused. The
                # limit constant is the layer's policy; the accounting uses
                # THIS store's live per-IP set (evicted on end_session).
                # Loopback is exempt: the local operator (CLI, tests, dev
                # server, multiple tabs) must never lock themselves out.
                layer = self._security_layer()
                if layer is not None and client_ip not in ("127.0.0.1", "::1", "localhost"):
                    # `live` already includes the session being created, so
                    # exceeding the limit means live > max.
                    live = {
                        sid for sid in self._ip_to_sessions[client_ip]
                        if sid in self._sessions
                        and not self._sessions[sid].is_locked()
                    }
                    if len(live) > layer.max_concurrent_sessions:
                        self.end_session(session_id)
                        logger.warning(
                            f"Session creation refused: IP {client_ip} would hold "
                            f"{len(live)} live sessions (limit {layer.max_concurrent_sessions})")
                        raise PermissionError(
                            f"Too many concurrent sessions "
                            f"(limit {layer.max_concurrent_sessions})")

            # User-agent binding (hijack detection): pin a truncated UA
            # snippet at creation; validate_session refuses requests whose
            # UA changed mid-session (cookie stolen to a different browser).
            user_agent = session.metadata.get("user_agent")
            if user_agent:
                session.metadata["user_agent_snippet"] = str(user_agent)[:200]

            # Advisory brute-force check: an IP that has been hammering logins
            # gets its new sessions refused too. Failures degrade open (the
            # local store above stays authoritative for session creation).
            denial = self._security_advise_deny_creation(client_ip)
            if denial:
                self.end_session(session_id)
                logger.warning(f"Session creation refused by security layer: {denial}")
                raise PermissionError(denial)
            # Creation succeeded — that is a successful login for the layer's
            # per-IP attempt window (clears stale failure streaks).
            self._security_record_attempt(client_ip, success=True)

            logger.debug(f"Created session {session_id[:8]}... for IP {client_ip}")
            return session
    
    def validate_session(self, session_id: str, client_ip: str | None = None,
                         user_agent: str | None = None) -> tuple[bool, Session | None, str | None]:
        """Validate a session.

        When *user_agent* is provided and the session pinned one at creation,
        a mid-session UA change (stolen cookie replayed from a different
        browser) is refused — hijack detection. The comparison policy is the
        security layer's _verify_user_agent (single policy source); if the
        layer is unavailable the check degrades open.
        """
        with self._lock:
            if not session_id:
                return False, None, "Session required"

            # Revocation memory (bounded): a revoked id is never valid again,
            # even though the live session object is gone.
            if session_id in self._revoked_ids:
                return False, None, "Session revoked"

            session = self._sessions.get(session_id)
            if not session:
                return False, None, "Invalid session"

            if session.is_expired(self.config.session_timeout_minutes):
                self.end_session(session_id)
                return False, None, "Session expired"

            if session.is_locked():
                return False, None, "Account locked"

            if not session.is_ip_valid(client_ip):
                logger.warning(f"Session {session_id[:8]} IP mismatch: expected {session.client_ip}, got {client_ip}")
                return False, None, "Session IP mismatch"

            # User-agent binding (advisory, fail-open)
            pinned = session.metadata.get("user_agent_snippet")
            if user_agent and pinned:
                verifier = None
                layer = self._security_layer()
                if layer is not None:
                    verifier = layer._verify_user_agent
                else:
                    try:
                        from core.security.session_security import SessionSecurityManager
                        verifier = SessionSecurityManager._verify_user_agent
                    except Exception:
                        verifier = None
                if verifier is not None:
                    try:
                        if not verifier(user_agent, pinned):
                            logger.warning(
                                f"Session {session_id[:8]} UA mismatch: hijack suspected")
                            self.record_failed_attempt(session_id)
                            return False, None, "Session user-agent mismatch"
                    except Exception as e:
                        logger.debug(f"UA verification failed (fail-open): {e}")

            session.last_activity = datetime.now()
            return True, session, None
    
    def record_failed_attempt(self, session_id: str) -> bool:
        """Record a failed authentication attempt.

        Returns:
            True if session should be locked
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.failed_attempts += 1

            # Feed the advisory layer's per-IP progressive lockout as well —
            # repeated failures against *any* session from this IP eventually
            # refuse new session creation (see _security_advise_deny_creation).
            self._security_record_attempt(
                session.metadata.get("client_ip"), success=False)

            if session.failed_attempts >= self.config.max_failed_attempts:
                session.locked_until = datetime.now() + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                logger.warning(f"Session {session_id[:8]}... locked due to failed attempts")
                return True

            return False
    
    def end_session(self, session_id: str):
        """End a session and clean up IP tracking."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                client_ip = session.metadata.get("client_ip")
                if client_ip and client_ip in self._ip_to_sessions:
                    self._ip_to_sessions[client_ip].discard(session_id)
                    if not self._ip_to_sessions[client_ip]:
                        del self._ip_to_sessions[client_ip]
                logger.debug(f"Ended session {session_id[:8]}...")

    def revoke_session(self, session_id: str) -> bool:
        """Explicitly revoke a session (the fourth session_security
        capability): unlike end_session, the revocation is remembered, so a
        later validate attempt with the same id reports 'Session revoked'
        instead of 'Invalid session' — evidence someone tried to reuse it.

        Returns True if the session existed and was revoked.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.metadata["revoked"] = True
            self._revoked_ids[session_id] = time.time()
            # Bound the revocation memory: drop the oldest (revocations
            # expire after a full session-timeout anyway).
            while len(self._revoked_ids) > self._MAX_REVOKED_IDS:
                self._revoked_ids.popitem(last=False)
            self.end_session(session_id)
            logger.warning(f"Session {session_id[:8]}... explicitly revoked")
            return True

    def revoke_ip_sessions(self, client_ip: str) -> int:
        """Revoke every live session for an IP (bulk: compromised endpoint,
        forced logout everywhere). Returns count revoked."""
        with self._lock:
            sids = list(self._ip_to_sessions.get(client_ip, set()))
            count = 0
            for sid in sids:
                if self.revoke_session(sid):
                    count += 1
            return count
    
    def cleanup_expired(self):
        """Remove expired sessions using session timeout."""
        with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired(self.config.session_timeout_minutes)
            ]
            for sid in expired:
                self.end_session(sid)
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_active_count(self) -> int:
        """Get count of active sessions."""
        with self._lock:
            return len(self._sessions)


class SecurityMiddleware:
    """Main security middleware coordinator."""

    # Loopback is exempt from the per-IP request-rate window: this is a
    # personal assistant that legitimately talks to itself (CLI, tests,
    # local tools) and the window exists to stop remote floods, not to
    # throttle the local operator.
    _RATE_EXEMPT_IPS = frozenset({"127.0.0.1", "::1", "localhost"})

    # Bounded client map: each entry is a deque of monotonic timestamps.
    _MAX_RATE_CLIENTS = 1000

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self.api_key_validator = APIKeyValidator(self.config)
        self.session_manager = SessionManager(self.config)
        self.sanitizer = InputSanitizer()
        self._blocked_ips: OrderedDict[str, float] = OrderedDict()
        self._MAX_BLOCKED_IPS = 1000
        self._rate_windows: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()
        self._rate_seq = 0  # monotonic key for per-request window entries
        self._rate_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Per-IP request-rate enforcement (rate_limit_requests_per_minute)
    # ------------------------------------------------------------------
    # This config field existed but was never enforced: the web layer uses
    # core.rate_limiter for per-endpoint limits, and this middleware — the
    # owner of the field — silently ignored it. It is now a per-IP sliding
    # window over the monotonic clock (immune to wall-clock steps, matching
    # the circuit breakers).

    def _rate_limit_allowed(self, client_ip: str | None) -> bool:
        """Check and consume one request slot for *client_ip*.

        Returns True if the request is within
        ``config.rate_limit_requests_per_minute``. Loopback and unknown
        clients (no IP visible, e.g. in-process calls) are exempt.
        """
        if not client_ip or client_ip in self._RATE_EXEMPT_IPS:
            return True

        limit = self.config.rate_limit_requests_per_minute
        if limit <= 0:
            return True  # explicitly disabled

        now = time.monotonic()
        window_seconds = 60.0

        with self._rate_lock:
            stamps = self._rate_windows.get(client_ip)
            if stamps is None:
                stamps = OrderedDict()
                self._rate_windows[client_ip] = stamps

            # Sliding window: drop entries older than the window.
            while stamps and now - next(iter(stamps.values())) > window_seconds:
                stamps.popitem(last=False)

            if len(stamps) >= limit:
                # LRU-touch so eviction order follows activity.
                self._rate_windows.move_to_end(client_ip)
                return False

            # Key by a monotonic sequence, NOT by client_ip: this OrderedDict
            # is already per-client, so keying by client_ip kept exactly one
            # entry alive and the limit could never fire.
            self._rate_seq += 1
            stamps[self._rate_seq] = now
            self._rate_windows.move_to_end(client_ip)

            # Bound the client map; evict the least recently active client.
            while len(self._rate_windows) > self._MAX_RATE_CLIENTS:
                self._rate_windows.popitem(last=False)
            return True

    def _rate_limit_retry_after(self, client_ip: str | None) -> float:
        """Seconds until the oldest timestamp in *client_ip*'s window ages out."""
        if not client_ip:
            return 0.0
        with self._rate_lock:
            stamps = self._rate_windows.get(client_ip)
            if not stamps:
                return 0.0
            oldest = next(iter(stamps.values()))
            return max(0.0, 60.0 - (time.monotonic() - oldest))
    
    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for responses."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Strict-Transport-Security": self.config.strict_transport_security,
        }
        
        if self.config.xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"
        
        if self.config.content_security_policy:
            headers["Content-Security-Policy"] = self.config.content_security_policy
        
        return headers
    
    def validate_request(self, 
                        headers: dict[str, str],
                        body: dict | None = None,
                        client_ip: str | None = None) -> tuple[bool, dict, str | None]:
        """Validate an incoming request.
        
        Args:
            headers: Request headers
            body: Request body (will be sanitized)
            client_ip: Client IP address
            
        Returns:
            Tuple of (is_valid, sanitized_data, error_message)
        """
        if client_ip and self._is_blocked_ip(client_ip):
            return False, {}, "IP blocked"

        if not self._rate_limit_allowed(client_ip):
            retry_after = int(self._rate_limit_retry_after(client_ip)) + 1
            return False, {}, (
                f"Rate limit exceeded ({self.config.rate_limit_requests_per_minute} "
                f"requests/minute); retry after {retry_after}s"
            )

        if self.config.require_auth:
            api_key = headers.get("X-API-Key") or headers.get("Authorization", "").replace("Bearer ", "")
            
            if not api_key:
                return False, {}, "Authentication required"
            
            is_valid, result = self.api_key_validator.validate(api_key)
            if not is_valid:
                return False, {}, result
        
        if body and isinstance(body, dict):
            sanitized_body = self.sanitizer.sanitize_dict(body, strict=False)
        else:
            sanitized_body = body or {}
        
        return True, sanitized_body, None
    
    def validate_websocket_connection(self,
                                      headers: dict[str, str],
                                      query_params: dict[str, str],
                                      client_ip: str | None = None) -> tuple[bool, str | None, str | None]:
        """Validate a WebSocket connection.
        
        Returns:
            Tuple of (is_valid, session_token, error_message)
        """
        if client_ip and self._is_blocked_ip(client_ip):
            return False, None, "IP blocked"

        if not self._rate_limit_allowed(client_ip):
            return False, None, (
                f"Rate limit exceeded "
                f"({self.config.rate_limit_requests_per_minute} requests/minute)"
            )

        if self.config.require_auth:
            api_key = (
                headers.get("X-API-Key") or
                headers.get("Authorization", "").replace("Bearer ", "") or
                query_params.get("api_key") or
                query_params.get("token")
            )
            
            if not api_key:
                return False, None, "Authentication required for WebSocket"
            
            is_valid, result = self.api_key_validator.validate(api_key)
            if not is_valid:
                return False, None, result
            
            return True, result, None
        
        try:
            session = self.session_manager.create_session(
                auth_level=AuthLevel.STANDARD,
                client_ip=client_ip,
                metadata={"user_agent": headers.get("user-agent") or headers.get("User-Agent")},
            )
        except PermissionError as e:
            # The advisory security layer refused creation (progressive
            # brute-force lockout on this IP).
            return False, None, str(e)

        # Immediately re-validate with the presenting UA (pins the binding
        # and exercises the same policy future requests will use).
        ua = headers.get("user-agent") or headers.get("User-Agent")
        if ua:
            ok, _sess, err = self.session_manager.validate_session(
                session.session_id, client_ip, user_agent=ua)
            if not ok:
                self.session_manager.end_session(session.session_id)
                return False, None, err or "Session validation failed"
        return True, session.session_id, None
    
    def _is_blocked_ip(self, ip: str) -> bool:
        """Check if an IP is blocked."""
        return ip in self._blocked_ips

    def block_ip(self, ip: str) -> None:
        self._blocked_ips[ip] = time.time()
        if len(self._blocked_ips) > self._MAX_BLOCKED_IPS:
            self._blocked_ips.popitem(last=False)

    def unblock_ip(self, ip: str) -> None:
        self._blocked_ips.pop(ip, None)
    
    def sanitize_input(self, data: Any, strict: bool = False) -> Any:
        """Sanitize user input."""
        if isinstance(data, dict):
            return self.sanitizer.sanitize_dict(data, strict)
        elif isinstance(data, str):
            return self.sanitizer.sanitize_string(data, strict)
        elif isinstance(data, list):
            return [self.sanitize_input(item, strict) for item in data]
        return data
    
    def create_session(self, auth_level: AuthLevel = AuthLevel.STANDARD,
                       client_ip: str | None = None,
                       metadata: dict | None = None) -> Session:
        """Create a new session (metadata may carry "user_agent")."""
        return self.session_manager.create_session(auth_level, client_ip, metadata)

    def validate_session(self, session_id: str, client_ip: str | None = None,
                         user_agent: str | None = None) -> tuple[bool, Session | None]:
        """Validate a session (user_agent enables hijack detection)."""
        is_valid, session, _ = self.session_manager.validate_session(session_id, client_ip, user_agent)
        return is_valid, session


def create_auth_middleware(security_config: SecurityConfig | None = None):
    """Factory function to create security middleware."""
    config = security_config or SecurityConfig()
    return SecurityMiddleware(config)


_security_middleware: SecurityMiddleware | None = None
_middleware_lock = threading.Lock()


def get_security_middleware(config: SecurityConfig | None = None) -> SecurityMiddleware:
    """Get or create the global security middleware instance."""
    global _security_middleware
    if _security_middleware is None:
        with _middleware_lock:
            if _security_middleware is None:
                _security_middleware = SecurityMiddleware(config)
    return _security_middleware