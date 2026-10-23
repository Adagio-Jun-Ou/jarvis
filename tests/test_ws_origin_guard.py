"""Regression tests for the WebSocket cross-site hijacking (CSWSH) guard.

The HTTP CSRF middleware only covers POST/PUT/PATCH/DELETE, and a
WebSocket handshake is a GET — so before this fix, none of the three WS
endpoints (/ws, /ws/api-activity, /api/mobile/ws) validated Origin at
all. Browsers always send Origin on a WS handshake and do NOT enforce
same-origin on WebSocket connections, so any website open in the user's
browser could connect to ws://localhost:<port>/ws and chat as the user.

Policy (identical semantics to the HTTP CSRF origin check — one policy,
not two):
  - Origin absent            -> allowed (non-browser client)
  - Origin netloc == Host    -> allowed (same-origin)
  - Origin netloc in JARVIS_CORS_ORIGINS -> allowed (explicit opt-in)
  - anything else            -> refused

The suffix/prefix weakness that once admitted "localhost.evil.com:8080"
against allowed "localhost:8080" is pinned here as a regression case.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ws(origin: str | None, host: str = "localhost:8766"):
    """A minimal WebSocket stand-in carrying only the headers we test."""
    headers = {}
    if origin is not None:
        headers["origin"] = origin
    if host:
        headers["host"] = host
    return SimpleNamespace(headers=headers)


def _interface():
    """WebInterface._ws_origin_allowed without booting the real app."""
    from interfaces.web_react import WebInterface

    iface = WebInterface.__new__(WebInterface)  # skip __init__ (agent, llm...)
    iface._allowed_cors_origins = set()
    return iface


class TestWSOriginGuard:
    def test_browser_same_origin_allowed(self):
        iface = _interface()
        assert iface._ws_origin_allowed(_ws("https://localhost:8766")) is True

    def test_browser_cross_origin_refused(self):
        iface = _interface()
        assert iface._ws_origin_allowed(_ws("https://evil.com")) is False

    def test_no_origin_allowed_nonbrowser(self):
        iface = _interface()
        assert iface._ws_origin_allowed(_ws(None)) is True

    def test_suffix_attack_refused(self):
        """localhost.evil.com must NOT match allowed host localhost —
        the exact netloc equality that the old HTTP CSRF prefix/suffix
        test got wrong."""
        iface = _interface()
        assert iface._ws_origin_allowed(
            _ws("https://localhost.evil.com:8766")) is False

    def test_cors_allowlist_entry_allowed(self):
        iface = _interface()
        iface._allowed_cors_origins = {"https://trusted.example.com"}
        assert iface._ws_origin_allowed(_ws("https://trusted.example.com")) is True

    def test_cors_allowlist_case_insensitive(self):
        iface = _interface()
        iface._allowed_cors_origins = {"https://Trusted.Example.com"}
        assert iface._ws_origin_allowed(_ws("https://trusted.example.com")) is True

    def test_case_insensitive_host_match(self):
        iface = _interface()
        assert iface._ws_origin_allowed(
            _ws("https://LOCALHOST:8766", host="localhost:8766")) is True

    def test_malformed_origin_refused(self):
        iface = _interface()
        # urlparse can fail on control-character headers
        assert iface._ws_origin_allowed(_ws("https://exa mple.com")) is False

    def test_empty_origin_header_allowed(self):
        """A present-but-empty Origin header is treated as absent
        (curl sends nothing; some proxies strip to empty)."""
        iface = _interface()
        assert iface._ws_origin_allowed(_ws("")) is True

    def test_subdomain_of_host_refused(self):
        iface = _interface()
        assert iface._ws_origin_allowed(
            _ws("https://sub.localhost:8766", host="localhost:8766")) is False

    def test_http_vs_https_scheme_irrelevant_netloc_only(self):
        """The same page served over http vs https is still same-origin
        for WS purposes (netloc equality, like the HTTP CSRF check)."""
        iface = _interface()
        assert iface._ws_origin_allowed(_ws("http://localhost:8766")) is True

    def test_port_mismatch_refused(self):
        iface = _interface()
        assert iface._ws_origin_allowed(
            _ws("https://localhost:9999", host="localhost:8766")) is False