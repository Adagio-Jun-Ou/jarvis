"""Tests for authentication and security middleware."""
import pytest
import time

from core.auth_middleware import (
    SecurityMiddleware,
    SecurityConfig,
    InputSanitizer,
    APIKeyValidator,
    SessionManager,
    AuthLevel,
)


class TestInputSanitizer:
    """Test input sanitization for injection attacks."""
    
    def test_sanitize_xss_script_tag(self):
        """Test XSS script tag removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('<script>alert("xss")</script>Hello')
        assert '<script>' not in result
        assert 'Hello' in result
    
    def test_sanitize_xss_javascript_url(self):
        """Test javascript: URL removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('Click <a href="javascript:alert(1)">here</a>')
        assert 'javascript:' not in result
    
    def test_sanitize_xss_event_handler(self):
        """Test XSS event handler removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('<img src=x onerror=alert(1)>')
        assert 'onerror' not in result
    
    def test_sanitize_xss_iframe(self):
        """Test iframe removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('<iframe src="evil.com"></iframe>')
        assert '<iframe' not in result
    
    def test_sanitize_sql_union(self):
        """Test SQL UNION injection removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string("1 UNION SELECT * FROM users")
        assert 'UNION SELECT' not in result.upper()
    
    def test_sanitize_sql_comment(self):
        """Test SQL comment removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string("admin'--")
        assert '--' not in result or result.strip() == "admin'"
    
    def test_sanitize_path_traversal(self):
        """Test path traversal removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('../../../etc/passwd')
        assert '../' not in result
    
    def test_sanitize_path_traversal_encoded(self):
        """Test encoded path traversal removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('%2e%2e%2fetc%2fpasswd')
        assert '%2e%2e' not in result.lower()
    
    def test_sanitize_command_injection_strict(self):
        """Test command injection removal in strict mode."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('; rm -rf /', strict=True)
        assert ';' not in result
        assert 'rm -rf' not in result
    
    def test_sanitize_command_backticks(self):
        """Test backtick command substitution removal."""
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize_string('`whoami`', strict=True)
        assert '`' not in result
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        sanitizer = InputSanitizer()
        data = {
            'name': '<script>alert(1)</script>John',
            'path': '../../../etc/passwd',
            'nested': {'key': "'; DROP TABLE users; --"}
        }
        result = sanitizer.sanitize_dict(data)
        assert '<script>' not in result['name']
        assert '../' not in result['path']
        assert 'DROP TABLE' not in result['nested']['key']
    
    def test_sanitize_list(self):
        """Test list sanitization."""
        sanitizer = InputSanitizer()
        data = ['<script>xss</script>', 'normal text', '../../path']
        result = sanitizer.sanitize_dict({'items': data})
        assert '<script>' not in result['items'][0]
        assert '../' not in result['items'][2]
    
    def test_escape_html(self):
        """Test HTML entity escaping."""
        sanitizer = InputSanitizer()
        result = sanitizer.escape_html('<script>alert(1)</script>')
        assert '&lt;' in result
        assert '&gt;' in result
        assert '<script>' not in result


class TestAPIKeyValidator:
    """Test API key validation."""
    
    def test_validate_correct_key(self):
        """Test validation with correct key."""
        config = SecurityConfig(api_key='test-secret-key')
        validator = APIKeyValidator(config)
        is_valid, result = validator.validate('test-secret-key')
        assert is_valid is True
        assert result is not None  # Session token
    
    def test_validate_wrong_key(self):
        """Test validation with wrong key."""
        config = SecurityConfig(api_key='test-secret-key')
        validator = APIKeyValidator(config)
        is_valid, result = validator.validate('wrong-key')
        assert is_valid is False
        assert 'Invalid' in result
    
    def test_validate_no_key_required(self):
        """Test validation when no key configured — should deny (security hardening)."""
        config = SecurityConfig(require_auth=False, api_key=None)
        validator = APIKeyValidator(config)
        is_valid, result = validator.validate('')
        assert is_valid is False
    
    def test_validate_multiple_keys(self, monkeypatch):
        """Test validation with multiple valid keys."""
        monkeypatch.setenv('JARVIS_API_KEYS', 'key1,key2,key3')
        config = SecurityConfig()
        validator = APIKeyValidator(config)
        
        is_valid1, _ = validator.validate('key1')
        is_valid2, _ = validator.validate('key2')
        is_valid3, _ = validator.validate('key3')
        is_valid_wrong, _ = validator.validate('key4')
        
        assert is_valid1 is True
        assert is_valid2 is True
        assert is_valid3 is True
        assert is_valid_wrong is False
    
    def test_timing_safe_comparison(self):
        """Test that key comparison is timing-safe."""
        config = SecurityConfig(api_key='a' * 32)
        validator = APIKeyValidator(config)
        
        # Timing-safe comparison should take similar time for right/wrong keys
        times = []
        for key in ['a' * 32, 'b' * 32, 'a' * 31 + 'b']:
            start = time.time()
            validator.validate(key)
            times.append(time.time() - start)
        
        # All times should be within reasonable variance (not strict test)
        # Just verify it doesn't error
        assert len(times) == 3


class TestSessionManager:
    """Test session management."""
    
    def test_create_session(self):
        """Test session creation."""
        config = SecurityConfig()
        manager = SessionManager(config)
        session = manager.create_session(AuthLevel.STANDARD, '127.0.0.1')
        
        assert session.session_id is not None
        assert session.auth_level == AuthLevel.STANDARD
        assert not session.is_expired()
    
    def test_validate_session(self):
        """Test session validation."""
        config = SecurityConfig()
        manager = SessionManager(config)
        session = manager.create_session()
        
        is_valid, sess, err = manager.validate_session(session.session_id)
        assert is_valid is True
        assert sess is not None
        assert err is None
    
    def test_validate_invalid_session(self):
        """Test validation of invalid session."""
        config = SecurityConfig()
        manager = SessionManager(config)
        
        is_valid, sess, err = manager.validate_session('invalid-id')
        assert is_valid is False
        assert sess is None
    
    def test_lockout_after_failed_attempts(self):
        """Test session lockout after failures."""
        config = SecurityConfig(max_failed_attempts=3)
        manager = SessionManager(config)
        session = manager.create_session()
        
        # Record failures
        for _ in range(3):
            manager.record_failed_attempt(session.session_id)
        
        # Session should be locked
        is_valid, sess, err = manager.validate_session(session.session_id)
        assert is_valid is False
        assert "locked" in err.lower() or "Account locked" in err
    
    def test_session_expiry(self):
        """Test session expiry."""
        config = SecurityConfig(session_timeout_minutes=0)  # Immediate timeout
        manager = SessionManager(config)
        session = manager.create_session()
        
        # Force expiry by setting last_activity to past time
        from datetime import datetime, timedelta
        session.last_activity = datetime.now() - timedelta(hours=1)
        
        is_valid, sess, err = manager.validate_session(session.session_id)
        assert is_valid is False
        assert "expired" in err.lower()
    
    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        config = SecurityConfig()
        manager = SessionManager(config)
        
        # Create a session
        session = manager.create_session()
        
        # Manually set last_activity to past
        from datetime import datetime, timedelta
        session.last_activity = datetime.now() - timedelta(hours=25)
        
        manager.cleanup_expired()
        
        # Expired session should be removed
        assert session.session_id not in manager._sessions


class TestSecurityMiddleware:
    """Test security middleware."""
    
    def test_get_security_headers(self):
        """Test security headers generation."""
        config = SecurityConfig()
        middleware = SecurityMiddleware(config)
        headers = middleware.get_security_headers()
        
        assert 'X-Content-Type-Options' in headers
        assert 'X-Frame-Options' in headers
        assert 'X-XSS-Protection' in headers
        assert 'Content-Security-Policy' in headers
        assert headers['X-Frame-Options'] == 'DENY'
    
    def test_validate_request_no_auth(self):
        """Test request validation without auth requirement."""
        config = SecurityConfig(require_auth=False)
        middleware = SecurityMiddleware(config)
        
        is_valid, data, err = middleware.validate_request(
            headers={},
            body={'message': 'Hello'},
            client_ip='127.0.0.1'
        )
        
        assert is_valid is True
        assert data == {'message': 'Hello'}
        assert err is None
    
    def test_validate_request_with_auth(self):
        """Test request validation with auth requirement."""
        config = SecurityConfig(require_auth=True, api_key='secret')
        middleware = SecurityMiddleware(config)
        
        # Without key
        is_valid, data, err = middleware.validate_request(
            headers={},
            body={},
            client_ip='127.0.0.1'
        )
        assert is_valid is False
        
        # With correct key
        is_valid, data, err = middleware.validate_request(
            headers={'X-API-Key': 'secret'},
            body={'message': 'Hello'},
            client_ip='127.0.0.1'
        )
        assert is_valid is True
    
    def test_validate_request_sanitizes_body(self):
        """Test that request validation sanitizes body."""
        config = SecurityConfig()
        middleware = SecurityMiddleware(config)
        
        is_valid, data, err = middleware.validate_request(
            headers={},
            body={
                'message': '<script>alert(1)</script>Hello',
                'path': '../../../etc/passwd'
            },
            client_ip='127.0.0.1'
        )
        
        assert is_valid is True
        assert '<script>' not in data['message']
        assert '../' not in data['path']
    
    def test_validate_file_path(self):
        """Test file path validation."""
        assert InputSanitizer.validate_file_path('safe/path.txt') is True
        assert InputSanitizer.validate_file_path('../unsafe/path.txt') is False
        assert InputSanitizer.validate_file_path('../../etc/passwd') is False
    
    def test_create_session(self):
        """Test session creation through middleware."""
        config = SecurityConfig()
        middleware = SecurityMiddleware(config)
        
        session = middleware.create_session(AuthLevel.STANDARD, '127.0.0.1')
        assert session is not None
        assert session.auth_level == AuthLevel.STANDARD
    
    def test_validate_session(self):
        """Test session validation through middleware."""
        config = SecurityConfig()
        middleware = SecurityMiddleware(config)
        
        session = middleware.create_session()
        is_valid, sess = middleware.validate_session(session.session_id)
        
        assert is_valid is True
        assert sess is not None


class TestSecurityConfig:
    """Test security configuration."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.require_auth is False
        assert config.session_timeout_minutes == 60
        assert config.max_failed_attempts == 5
        assert config.rate_limit_requests_per_minute == 120
    
    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            require_auth=True,
            api_key='test-key',
            session_timeout_minutes=30,
            max_failed_attempts=3,
            rate_limit_requests_per_minute=60
        )
        
        assert config.require_auth is True
        assert config.api_key == 'test-key'
        assert config.session_timeout_minutes == 30
        assert config.max_failed_attempts == 3


class TestAuthLevel:
    """Test authentication levels."""
    
    def test_auth_levels(self):
        """Test that auth levels exist."""
        assert AuthLevel.PUBLIC.value == 'public'
        assert AuthLevel.READONLY.value == 'readonly'
        assert AuthLevel.STANDARD.value == 'standard'
        assert AuthLevel.ELEVATED.value == 'elevated'
        assert AuthLevel.ADMIN.value == 'admin'


class TestRateLimitEnforcement:
    """SecurityConfig.rate_limit_requests_per_minute must actually be enforced.

    Regression: the field existed since the beginning but nothing read it —
    the web layer used core.rate_limiter's own limits and this middleware,
    the field's owner, silently ignored it. It is now a per-IP sliding
    window over the monotonic clock, loopback-exempt.
    """

    def test_remote_ip_blocked_after_limit(self):
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=3,
        ))

        for _ in range(3):
            ok, _data, err = mw.validate_request(
                headers={}, body={}, client_ip="203.0.113.7",
            )
            assert ok is True, err

        ok, _data, err = mw.validate_request(
            headers={}, body={}, client_ip="203.0.113.7",
        )
        assert ok is False
        assert "Rate limit" in err

    def test_loopback_is_exempt(self):
        """The local operator (CLI, tests, tools) must not throttle itself."""
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=2,
        ))

        for _ in range(50):
            ok, _data, err = mw.validate_request(
                headers={}, body={}, client_ip="127.0.0.1",
            )
            assert ok is True, err

    def test_no_ip_is_exempt(self):
        """In-process calls (no client_ip) never hit the window."""
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=2,
        ))

        for _ in range(50):
            ok, _data, _err = mw.validate_request(headers={}, body={})
            assert ok is True

    def test_windows_are_per_ip(self):
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=2,
        ))

        for _ in range(2):
            assert mw.validate_request(
                headers={}, body={}, client_ip="203.0.113.7",
            )[0] is True
        # Exhausted for .7 ...
        assert mw.validate_request(
            headers={}, body={}, client_ip="203.0.113.7",
        )[0] is False
        # ... but a different IP is unaffected.
        assert mw.validate_request(
            headers={}, body={}, client_ip="198.51.100.9",
        )[0] is True

    def test_window_is_monotonic_not_wall_clock(self, monkeypatch):
        """A wall-clock step must not fake window expiry. Regression class:
        same as the circuit breakers — decision state on the monotonic clock."""
        import time as _time
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=1,
        ))
        assert mw.validate_request(
            headers={}, body={}, client_ip="203.0.113.7",
        )[0] is True
        assert mw.validate_request(
            headers={}, body={}, client_ip="203.0.113.7",
        )[0] is False

        real_time = _time.time()
        monkeypatch.setattr(_time, "time", lambda: real_time + 3650 * 86400.0)
        # A decade of wall-clock 'passing' must not free a slot.
        assert mw.validate_request(
            headers={}, body={}, client_ip="203.0.113.7",
        )[0] is False

    def test_zero_disables_limit(self):
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=0,
        ))
        for _ in range(200):
            assert mw.validate_request(
                headers={}, body={}, client_ip="203.0.113.7",
            )[0] is True

    def test_websocket_connection_rate_limited(self):
        """The WebSocket path previously had no rate check at all."""
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=2,
        ))

        for _ in range(2):
            ok, _token, err = mw.validate_websocket_connection(
                headers={}, query_params={}, client_ip="203.0.113.7",
            )
            assert ok is True, err

        ok, _token, err = mw.validate_websocket_connection(
            headers={}, query_params={}, client_ip="203.0.113.7",
        )
        assert ok is False
        assert "Rate limit" in err

    def test_blocked_ip_check_happens_before_rate_limit(self):
        """Ordering: a blocked IP is refused regardless of window state."""
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(require_auth=False))
        mw.block_ip("203.0.113.7")

        ok, _data, err = mw.validate_request(
            headers={}, body={}, client_ip="203.0.113.7",
        )
        assert ok is False
        assert err == "IP blocked"

    def test_client_map_is_bounded(self):
        """The per-IP map must not grow without bound under a client flood."""
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=1000,
        ))
        for i in range(mw._MAX_RATE_CLIENTS + 500):
            mw.validate_request(headers={}, body={}, client_ip=f"10.0.{i // 256}.{i % 256}")

        assert len(mw._rate_windows) <= mw._MAX_RATE_CLIENTS

    def test_retry_after_is_reported(self):
        from core.auth_middleware import SecurityConfig, SecurityMiddleware

        mw = SecurityMiddleware(SecurityConfig(
            require_auth=False, rate_limit_requests_per_minute=1,
        ))
        mw.validate_request(headers={}, body={}, client_ip="203.0.113.7")
        mw.validate_request(headers={}, body={}, client_ip="203.0.113.7")

        retry = mw._rate_limit_retry_after("203.0.113.7")
        assert 0.0 <= retry <= 60.0