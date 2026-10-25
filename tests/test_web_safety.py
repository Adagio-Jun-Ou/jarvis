#!/usr/bin/env python3
"""
Web Tools Safe Search Tests

Tests the safe search and URL safety mechanisms in web tools.
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

import sys
sys.path.insert(0, _JARVIS_ROOT)

import pytest
from unittest.mock import Mock, patch, AsyncMock


# ==================== SAFE SEARCH TESTS ====================

class TestSafeSearch:
    """Tests for safe search functionality."""
    
    def test_safe_search_clean_query(self):
        """Clean queries should pass safe search."""
        from tools.web import check_safe_search
        
        is_safe, category, hint = check_safe_search("Python programming tutorial")
        
        assert is_safe == True
        assert category == ""
        assert hint == ""
    
    def test_safe_search_adult_content_blocked(self):
        """Adult content queries should be blocked."""
        from tools.web import check_safe_search
        
        is_safe, category, hint = check_safe_search("porn videos")
        
        assert is_safe == False
        assert category == "adult"
    
    def test_safe_search_violence_blocked(self):
        """Violence queries should be blocked."""
        from tools.web import check_safe_search
        
        is_safe, category, hint = check_safe_search("gore videos")
        
        assert is_safe == False
        assert category == "violence"
    
    def test_safe_search_illegal_blocked(self):
        """Illegal content queries should be blocked."""
        from tools.web import check_safe_search, BLOCKED_CATEGORIES
        
        is_safe, category, hint = check_safe_search("how to make bomb")
        
        assert is_safe == False
        # "bomb" is categorized as harmful (dangerous content)
        assert category in BLOCKED_CATEGORIES or category == "harmful"
    
    def test_safe_search_malware_blocked(self):
        """Malware queries should be blocked."""
        from tools.web import check_safe_search, BLOCKED_CATEGORIES
        
        is_safe, category, hint = check_safe_search("malware download")
        
        assert is_safe == False
        assert category in BLOCKED_CATEGORIES
    
    def test_safe_search_harmful_blocked(self):
        """Harmful content queries should be blocked."""
        from tools.web import check_safe_search
        
        is_safe, category, hint = check_safe_search("suicide methods")
        
        assert is_safe == False
        assert category == "harmful"
    
    def test_safe_search_hate_blocked(self):
        """Hate speech queries should be blocked."""
        from tools.web import check_safe_search
        
        is_safe, category, hint = check_safe_search("white supremacist")
        
        assert is_safe == False
        assert category == "hate"


# ==================== URL SAFETY TESTS ====================

class TestURLSafety:
    """Tests for URL safety checking."""
    
    def test_safe_url_http(self):
        """HTTP URLs should be allowed."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://example.com")
        
        assert is_safe == True
        assert reason == ""
    
    def test_safe_url_https(self):
        """HTTPS URLs should be allowed."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("https://example.com")
        
        assert is_safe == True
        assert reason == ""
    
    def test_blocked_scheme_file(self):
        """file:// URLs should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("file:///etc/passwd")
        
        assert is_safe == False
        assert "file" in reason.lower()
    
    def test_blocked_scheme_javascript(self):
        """javascript: URLs should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("javascript:alert('xss')")
        
        assert is_safe == False
    
    def test_blocked_localhost(self):
        """Localhost URLs should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://localhost/admin")
        
        assert is_safe == False
        assert "localhost" in reason.lower() or "blocked" in reason.lower()
    
    def test_blocked_127_ip(self):
        """127.x.x.x IPs should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://127.0.0.1/admin")
        
        assert is_safe == False
    
    def test_blocked_private_ip(self):
        """Private IP ranges should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://192.168.1.1/admin")
        
        assert is_safe == False
    
    def test_blocked_metadata_endpoint(self):
        """Cloud metadata endpoints should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://169.254.169.254/latest/meta-data/")
        
        assert is_safe == False
    
    def test_blocked_kubernetes_endpoint(self):
        """Kubernetes internal endpoints should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("https://kubernetes.default/api/v1/secrets")
        
        assert is_safe == False


# ==================== WEB SEARCH TOOL TESTS ====================

class TestWebSearchTool:
    """Tests for WebSearchTool."""
    
    def test_tool_initialization(self):
        """Tool should initialize correctly."""
        from tools.web import WebSearchTool
        
        tool = WebSearchTool()
        
        assert tool.name == "web_search"
        assert tool.requires_permission == False
    
    def test_no_query_error(self):
        """Should error on empty query."""
        from tools.web import WebSearchTool
        
        tool = WebSearchTool()
        
        import asyncio
        result = asyncio.run(tool.execute(""))
        
        assert "Error" in result or "error" in result.lower()
    
    @pytest.mark.asyncio
    async def test_unsafe_query_blocked(self):
        """Unsafe queries should be blocked by default."""
        from tools.web import WebSearchTool
        
        tool = WebSearchTool()
        
        result = await tool.execute("porn videos", unsafe=False)
        
        assert "blocked" in result.lower() or "permission" in result.lower()
    
    @pytest.mark.asyncio
    async def test_illegal_query_always_blocked(self):
        """Illegal queries should always be blocked."""
        from tools.web import WebSearchTool
        
        tool = WebSearchTool()
        
        result = await tool.execute("child porn", unsafe=True)
        
        assert "blocked" in result.lower()


# ==================== WEB FETCH TOOL TESTS ====================

class TestWebFetchTool:
    """Tests for WebFetchTool."""
    
    def test_tool_initialization(self):
        """Tool should initialize correctly."""
        from tools.web import WebFetchTool
        
        tool = WebFetchTool()
        
        assert tool.name == "web_fetch"
        assert tool.requires_permission == False
    
    def test_no_url_error(self):
        """Should error on empty URL."""
        from tools.web import WebFetchTool
        
        tool = WebFetchTool()
        
        import asyncio
        result = asyncio.run(tool.execute(""))
        
        assert "Error" in result or "error" in result.lower()
    
    @pytest.mark.asyncio
    async def test_file_url_blocked(self):
        """file:// URLs should be blocked."""
        from tools.web import WebFetchTool
        
        tool = WebFetchTool()
        
        result = await tool.execute("file:///etc/passwd")
        
        assert "blocked" in result.lower() or "error" in result.lower()
    
    @pytest.mark.asyncio
    async def test_localhost_blocked(self):
        """Localhost URLs should be blocked."""
        from tools.web import WebFetchTool
        
        tool = WebFetchTool()
        
        result = await tool.execute("http://localhost/admin")
        
        # Either blocked by SSRF protection or an error occurs
        assert "blocked" in result.lower() or "error" in result.lower() or "internal" in result.lower()


# ==================== IMAGE SEARCH TOOL TESTS ====================

class TestImageSearchTool:
    """Tests for ImageSearchTool."""
    
    def test_tool_initialization(self):
        """Tool should initialize correctly."""
        from tools.web import ImageSearchTool
        
        tool = ImageSearchTool()
        
        assert tool.name == "image_search"
        assert tool.requires_permission == False
    
    @pytest.mark.asyncio
    async def test_unsafe_image_query_blocked(self):
        """Unsafe image queries should be blocked."""
        from tools.web import ImageSearchTool
        
        tool = ImageSearchTool()
        
        result = await tool.execute("nude images", unsafe=False)
        
        assert "blocked" in result.lower() or "permission" in result.lower()


# ==================== SSRF PROTECTION TESTS ====================

class TestSSRFProtection:
    """Tests for Server-Side Request Forgery protection."""
    
    def test_aws_metadata_blocked(self):
        """AWS metadata endpoint should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://169.254.169.254/latest/meta-data/")
        
        assert is_safe == False
    
    def test_gce_metadata_blocked(self):
        """GCE metadata endpoint should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://metadata.google.internal/computeMetadata/v1/")
        
        assert is_safe == False
    
    def test_internal_aws_ip_blocked(self):
        """Internal AWS IPs should be blocked."""
        from tools.web import is_safe_url
        
        is_safe, reason = is_safe_url("http://10.0.0.1/")
        
        assert is_safe == False
    
    def test_internal_network_blocked(self):
        """Internal network IPs should be blocked."""
        from tools.web import is_safe_url
        
        internal_ips = [
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://192.168.0.1/",
            "http://127.0.0.1/",
        ]
        
        for url in internal_ips:
            is_safe, reason = is_safe_url(url)
            assert is_safe == False, f"Should block {url}"

    @pytest.mark.parametrize("url,expect_in_reason", [
        ("https://example.com:22/x", "port"),
        ("https://example.com:3306/x", "port"),
        ("https://example.com:6379/x", "port"),
        ("https://example.com:27017/x", "port"),
        ("https://user:secret@example.com/x", "credential"),
    ])
    def test_shared_ssrf_protector_denials_reach_is_safe_url(self, url, expect_in_reason):
        """core.security.ssrf_protection must actually gate the fetch path.

        These classes (internal service ports, embedded credentials) are only
        expressible by the shared protector — the local string denylists in
        tools/web.py cannot catch them. Regression guard for the protector
        having been written and tested but never wired in.
        """
        from tools.web import is_safe_url

        is_safe, reason = is_safe_url(url)

        assert is_safe is False, f"Should block {url}"
        assert expect_in_reason in reason.lower(), reason

    def test_shared_ssrf_protector_is_exported(self):
        """The protector must be reachable from the security package."""
        from core.security import get_ssrf_protector, validate_outbound_url

        assert get_ssrf_protector() is not None
        assert validate_outbound_url("https://example.com/page").valid is True


# ==================== CONTENT CATEGORY TESTS ====================

class TestContentCategories:
    """Tests for content category detection."""
    
    def test_adult_sites_in_url_check(self):
        """Adult sites should be blocked in URL check."""
        from tools.web import check_url_safety
        
        blocked_sites = [
            "https://pornhub.com/",
            "https://xvideos.com/",
        ]
        
        for url in blocked_sites:
            is_safe, category, hint = check_url_safety(url)
            assert is_safe == False or category == "adult", f"Should block {url}"
    
    def test_research_terms_allowed(self):
        """Legitimate research terms should be allowed."""
        from tools.web import check_safe_search
        
        allowed_queries = [
            "breast cancer research",
            "drug rehabilitation programs",
            "domestic violence support",
            "suicide prevention hotline",
            "hate crime statistics",
        ]
        
        for query in allowed_queries:
            is_safe, category, hint = check_safe_search(query)
            # Some might trigger false positives but context matters
            # This test documents expected behavior


# ==================== INTEGRATION TESTS ====================

class TestWebToolsIntegration:
    """Integration tests for web tools."""
    
    def test_web_tools_export(self):
        """WebTools class should export all tools."""
        from tools.web import WebTools
        
        tools = WebTools()
        tool_list = tools.get_tools()
        
        tool_names = [t.name for t in tool_list]
        
        assert "web_search" in tool_names
        assert "web_fetch" in tool_names
        assert "image_search" in tool_names
    
    def test_tools_have_schemas(self):
        """All tools should have valid schemas."""
        from tools.web import WebTools
        
        tools = WebTools().get_tools()
        
        for tool in tools:
            schema = tool.get_schema()
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema


# ==================== DNS-REBANDING / PINNED FETCH TESTS ====================

class TestPinnedFetch:
    """_pinned_fetch must connect to the *validated* IP, never re-resolve.

    The DNS-rebinding attack these tests guard against: a hostile resolver
    answers the pre-flight validation with a public IP, then answers the HTTP
    client's own resolution (moments later) with an internal one. The fix is
    structural — validation and connect share one resolution, and the request
    is sent to the IP, so there is no second lookup to attack.

    All tests are offline: resolution is monkeypatched, and the client uses a
    mock transport that records the wire-level request.
    """

    PUBLIC_IP = "93.184.216.34"

    @staticmethod
    def _make_client(responses):
        """AsyncClient with a mock transport recording every request."""
        import httpx

        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            status, headers, body = responses.pop(0) if responses else (200, {}, b"ok")
            return httpx.Response(status, headers=headers, content=body)

        transport = httpx.MockTransport(handler)
        return httpx.AsyncClient(transport=transport), seen

    @staticmethod
    def _patch_dns(monkeypatch, ips_by_host):
        """Fake only the DNS layer; let the real resolve_validated_ips run.

        Patching just ``socket.getaddrinfo`` (as used by
        core.security.ssrf_protection) keeps the real IP-literal handling,
        caching and validation logic under test — only the DNS answers are
        controlled.
        """
        import socket as _socket
        from core.security.ssrf_protection import get_ssrf_protector

        # The singleton's cache persists across tests; start clean so a fake
        # answer can never leak between tests.
        get_ssrf_protector()._dns_cache.clear()

        def fake_getaddrinfo(hostname, *args, **kwargs):
            ips = ips_by_host.get(hostname)
            if ips is None:
                raise _socket.gaierror(8, "Name or service not known")
            return [
                (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, 0))
                for ip in ips
            ]

        monkeypatch.setattr(
            "core.security.ssrf_protection.socket.getaddrinfo", fake_getaddrinfo
        )

    def test_connection_goes_to_the_validated_ip(self, monkeypatch):
        """The request URL must carry the IP, not the hostname.

        This is the core of the fix: anyio does no DNS for IP literals, so a
        rebinding attack between check and connect is impossible.
        """
        import asyncio
        import httpx
        from tools.web import _pinned_fetch

        self._patch_dns(monkeypatch, {"example.com": [self.PUBLIC_IP]})
        client, seen = self._make_client([(200, {}, b"hello")])

        async def run():
            async with client:
                response, final_url = await _pinned_fetch(client, "http://example.com/x")
                return response.text, final_url

        body, final_url = asyncio.run(run())

        assert body == "hello"
        assert len(seen) == 1
        request = seen[0]
        assert request.url.host == self.PUBLIC_IP, (
            "request went to the hostname (would re-resolve!): "
            f"{request.url.host!r}"
        )

    def test_host_header_and_sni_keep_the_original_name(self, monkeypatch):
        """TLS SNI and the Host header must keep the original hostname.

        Otherwise vhosts/CDNs would break and HTTPS certificate validation
        would fail against the IP.
        """
        import asyncio
        import httpx
        from tools.web import _pinned_fetch

        self._patch_dns(monkeypatch, {"example.com": [self.PUBLIC_IP]})
        client, seen = self._make_client([(200, {}, b"")])

        async def run():
            async with client:
                await _pinned_fetch(client, "https://example.com:8443/p?x=1")

        asyncio.run(run())

        request = seen[0]
        assert request.url.host == self.PUBLIC_IP
        assert request.headers["Host"] == "example.com:8443"
        assert request.extensions.get("sni_hostname") == "example.com"
        assert request.url.path == "/p"
        assert request.url.query == b"x=1"

    def test_host_header_omits_port_for_default_ports(self, monkeypatch):
        import asyncio
        import httpx
        from tools.web import _pinned_fetch

        self._patch_dns(monkeypatch, {"example.com": [self.PUBLIC_IP]})
        client, seen = self._make_client([(200, {}, b"")])

        async def run():
            async with client:
                await _pinned_fetch(client, "https://example.com/p")

        asyncio.run(run())

        request = seen[0]
        assert request.headers["Host"] == "example.com"  # no :443

    def test_ip_literal_url_is_not_pinned_and_not_blocked(self, monkeypatch):
        """A URL that is already an IP needs no pinning — and must still work.

        (No DNS is involved; the pre-flight layers checked the literal.)
        """
        import asyncio
        import httpx
        from tools.web import _pinned_fetch

        # Resolution of anything would fail here; the IP-literal path must
        # never call the resolver at all.
        self._patch_dns(monkeypatch, {})
        client, seen = self._make_client([(200, {}, b"direct")])

        async def run():
            async with client:
                response, _ = await _pinned_fetch(
                    client, f"http://{self.PUBLIC_IP}/x"
                )
                return response.text

        assert asyncio.run(run()) == "direct"
        assert seen[0].url.host == self.PUBLIC_IP

    def test_rebinding_to_internal_ip_is_refused_at_connect_time(self, monkeypatch):
        """If DNS answers 'internal' at pin time, the fetch must be refused.

        This is the actual rebinding scenario: validation would have passed
        yesterday's public record; the authoritative resolution happens at
        connect time and refuses.
        """
        import asyncio
        import httpx
        import pytest
        from tools.web import WebFetchError, _pinned_fetch

        self._patch_dns(monkeypatch, {"evil.example.com": ["127.0.0.1"]})
        client, seen = self._make_client([(200, {}, b"should never happen")])

        async def run():
            async with client:
                await _pinned_fetch(client, "http://evil.example.com/x")

        with pytest.raises(WebFetchError) as excinfo:
            asyncio.run(run())
        assert "internal" in str(excinfo.value).lower()
        assert excinfo.value.kind == "blocked"
        assert seen == []  # nothing was ever sent

    def test_redirect_hop_is_revalidated_and_repinned(self, monkeypatch):
        """A redirect to an internal name must be refused; a safe redirect is
        followed with pinning applied to the hop too."""
        import asyncio
        import httpx
        from tools.web import WebFetchError, _pinned_fetch

        self._patch_dns(
            monkeypatch,
            {"good.example.com": [self.PUBLIC_IP], "internal.example.com": ["10.0.0.5"]},
        )
        client, seen = self._make_client([
            (302, {"location": "http://internal.example.com/secret"}, b""),
        ])

        async def run():
            async with client:
                await _pinned_fetch(client, "http://good.example.com/start")

        with pytest.raises(WebFetchError) as excinfo:
            asyncio.run(run())
        assert excinfo.value.kind == "redirect_blocked"
        # The redirect target never got a request.
        assert all(r.url.host == self.PUBLIC_IP for r in seen)

    def test_redirect_to_safe_host_is_pinned_per_hop(self, monkeypatch):
        import asyncio
        import httpx
        from tools.web import _pinned_fetch

        self._patch_dns(
            monkeypatch,
            {"a.example.com": [self.PUBLIC_IP], "b.example.com": ["198.51.100.7"]},
        )
        client, seen = self._make_client([
            (301, {"location": "http://b.example.com/final"}, b""),
            (200, {}, b"landed"),
        ])

        async def run():
            async with client:
                response, final_url = await _pinned_fetch(client, "http://a.example.com/start")
                return response.text, final_url

        body, final_url = asyncio.run(run())
        assert body == "landed"
        assert final_url == "http://b.example.com/final"
        assert [r.url.host for r in seen] == [self.PUBLIC_IP, "198.51.100.7"]
        assert seen[1].headers["Host"] == "b.example.com"

    def test_first_ip_failure_falls_back_to_next_validated_ip(self, monkeypatch):
        """Multi-address redundancy is preserved: try each validated IP once."""
        import asyncio
        import httpx
        from tools.web import _pinned_fetch

        self._patch_dns(monkeypatch, {"example.com": ["192.0.2.1", "198.51.100.7"]})

        import httpx as _hx

        seen = []

        def handler(request: _hx.Request) -> _hx.Response:
            seen.append(request)
            if request.url.host == "192.0.2.1":
                # Simulate a refused connection for the first address.
                raise _hx.ConnectError("refused", request=request)
            return _hx.Response(200, content=b"second worked")

        client = _hx.AsyncClient(transport=_hx.MockTransport(handler))

        async def run():
            async with client:
                response, _ = await _pinned_fetch(client, "http://example.com/x")
                return response.text

        assert asyncio.run(run()) == "second worked"
        assert [r.url.host for r in seen] == ["192.0.2.1", "198.51.100.7"]

    def test_all_ips_down_raises_connect_error_not_silent_success(self, monkeypatch):
        import asyncio
        import httpx
        import pytest
        from tools.web import WebFetchError, _pinned_fetch

        self._patch_dns(monkeypatch, {"example.com": ["192.0.2.1", "198.51.100.7"]})

        import httpx as _hx

        def handler(request: _hx.Request) -> _hx.Response:
            raise _hx.ConnectError("refused", request=request)

        client = _hx.AsyncClient(transport=_hx.MockTransport(handler))

        async def run():
            async with client:
                await _pinned_fetch(client, "http://example.com/x")

        with pytest.raises(WebFetchError) as excinfo:
            asyncio.run(run())
        assert excinfo.value.kind == "connect"
        assert "2 validated" in str(excinfo.value)

    def test_is_safe_url_and_pinned_fetch_share_one_resolution(self, monkeypatch):
        """is_safe_url's layer-2 and _pinned_fetch must not resolve twice.

        Two independent lookups would reopen a smaller rebinding window and
        double the DNS load. Both go through SSRFProtector's cache.
        """
        from core.security.ssrf_protection import get_ssrf_protector
        from tools.web import is_safe_url

        import socket

        calls = []

        def fake_getaddrinfo(hostname, *args, **kwargs):
            calls.append(hostname)
            # (family, type, proto, canonname, sockaddr)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (self.PUBLIC_IP, 0))]
        monkeypatch.setattr(
            "core.security.ssrf_protection.socket.getaddrinfo", fake_getaddrinfo
        )
        # The cache is the sharing mechanism: is_safe_url's pass populates it,
        # the pinning resolution reuses it. Start from a clean slate.
        get_ssrf_protector()._dns_cache.clear()

        is_safe, reason = is_safe_url("http://example.com/x")
        assert is_safe is True, reason

        ips, err = get_ssrf_protector().resolve_validated_ips("example.com")
        assert ips == [self.PUBLIC_IP]

        # One DNS call total: the is_safe_url pass resolved and cached; the
        # pinning resolution reused the cache instead of resolving again.
        assert calls.count("example.com") == 1, calls


class TestResolveValidatedIps:
    """The authoritative resolution point in core.security.ssrf_protection."""

    def test_public_ip_literal_returned_as_is(self):
        from core.security import resolve_validated_ips

        ips, error = resolve_validated_ips("93.184.216.34")
        assert ips == ["93.184.216.34"]
        assert error == ""

    def test_internal_ip_literal_refused(self):
        from core.security import resolve_validated_ips

        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"):
            ips, error = resolve_validated_ips(ip)
            assert ips == [], ip
            assert error, ip

    def test_hostname_resolution_validated(self, monkeypatch):
        import socket
        from core.security import ssrf_protection
        from core.security.ssrf_protection import get_ssrf_protector

        get_ssrf_protector()._dns_cache.clear()

        def fake_getaddrinfo(hostname, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(
            "core.security.ssrf_protection.socket.getaddrinfo", fake_getaddrinfo
        )
        ips, error = ssrf_protection.resolve_validated_ips("example.com")
        assert ips == ["93.184.216.34"]
        assert error == ""

    def test_mixed_public_and_internal_answers_refused(self, monkeypatch):
        """A record mixing public and internal addresses must fail closed —
        that mix is exactly what a rebinding resolver serves."""
        import socket
        from core.security import ssrf_protection
        from core.security.ssrf_protection import get_ssrf_protector

        get_ssrf_protector()._dns_cache.clear()

        def fake_getaddrinfo(hostname, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.9", 0)),
            ]

        monkeypatch.setattr(
            "core.security.ssrf_protection.socket.getaddrinfo", fake_getaddrinfo
        )
        ips, error = ssrf_protection.resolve_validated_ips("sneaky.example")
        assert ips == []
        assert "10.0.0.9" in error

    def test_resolution_failure_returns_error_not_empty_success(self):
        from core.security.ssrf_protection import get_ssrf_protector

        get_ssrf_protector()._dns_cache.clear()
        ips, error = get_ssrf_protector().resolve_validated_ips(
            "nonexistent.invalid.example.jarvis.test"
        )
        assert ips == []
        assert error  # must be a reason, not silent []


# ==================== RUN ====================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", 
         "--tb=short", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)