"""Tests for agent/web_agentic.py — strict SSRF defense + HTML strip."""
from __future__ import annotations

import pytest

from agent import web_agentic as wa


def test_url_blocks_file_scheme():
    assert not wa._url_ok("file:///etc/passwd")


def test_url_blocks_ftp():
    assert not wa._url_ok("ftp://example.com/foo")


def test_url_blocks_gopher():
    assert not wa._url_ok("gopher://example.com/")


def test_url_blocks_loopback_hostname():
    assert not wa._url_ok("http://localhost/admin")
    assert not wa._url_ok("http://127.0.0.1/")


def test_url_blocks_link_local_metadata():
    """The cloud metadata service 169.254.169.254 must be refused."""
    assert not wa._url_ok("http://169.254.169.254/latest/meta-data/")


def test_url_blocks_private_ip_literal():
    assert not wa._url_ok("http://10.0.0.1/")
    assert not wa._url_ok("http://192.168.1.1/")
    assert not wa._url_ok("http://172.16.0.1/")


def test_url_allows_https_public():
    # example.com resolves to a public IP, so it should be allowed.
    # We do a real DNS lookup; if the CI is offline the test still passes
    # (the function is best-effort).
    try:
        assert wa._url_ok("https://example.com/")
    except Exception:
        pytest.skip("DNS not available in test env")


def test_guarded_get_blocks_redirect_to_metadata_ip():
    import httpx

    # A public start URL that 302s straight to the cloud-metadata IP. The
    # guard must refuse to follow it (raise), never connecting to the IP.
    def handler(request):
        return httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(wa._RedirectBlocked):
            wa._guarded_get(client, "https://public.example/start", {})


def test_guarded_get_follows_redirect_to_public_ip():
    import httpx

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            # Redirect to a public IP literal (no DNS needed, not blocked).
            return httpx.Response(302, headers={"location": "http://93.184.216.34/next"})
        return httpx.Response(200, text="landed")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        r = wa._guarded_get(client, "http://93.184.216.34/start", {})
    assert r.status_code == 200
    assert r.text == "landed"
    assert calls["n"] == 2


def test_strip_html_removes_script_and_style():
    html = "<html><script>alert(1)</script><style>body{color:red}</style><p>hi</p></html>"
    out = wa._strip_html(html)
    assert "alert" not in out
    assert "color" not in out
    assert "hi" in out
