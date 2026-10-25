"""
Web agentic: a small set of tools for the agent to do public-web
research (search, fetch, synthesize). Strict SSRF defense:

  * No calls to private IP ranges (RFC 1918, link-local, multicast,
    loopback, 169.254.0.0/16 metadata, ::1, fc00::/7, fe80::/10).
  * No calls to hostnames that resolve to private IPs (resolved here
    once, and re-resolved after every redirect).
  * All requests go through the corporate egress proxy; the proxy is
    the second line of defense.
  * The HTML strip is conservative (script/style removal + tag strip),
    not a real browser engine.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger("agent.web_agentic")

MAX_HOPS = 12
MAX_REDIRECTS = 5
MAX_PAGE_CHARS = 200_000
PROXY = os.environ.get("ALEX_EGRESS_PROXY", "").strip()  # e.g. http://proxy.corp:3128

# Block by scheme + by IP literal. The hostname check catches everything
# that resolves to one of these.
_BLOCKED_SCHEMES = frozenset({"file", "ftp", "gopher", "dict", "ldap", "ldap+tls"})


# We refuse to connect to anything in these ranges. Includes link-local
# (169.254.0.0/16 — cloud metadata), private (10/8, 172.16/12, 192.168/16),
# loopback, multicast, and the IPv6 equivalents.
#
# CGNAT (100.64.0.0/10) is explicit: none of the stdlib predicates cover
# it (is_private/is_reserved are all False there), and Tailscale and
# several enterprise overlays live in that space — a peer node reachable
# through it is exactly as internal as a 10.x host.
_EXTRA_BLOCKED_NETS = tuple(
    ipaddress.ip_network(n)
    for n in ("100.64.0.0/10", "192.0.0.0/24", "198.18.0.0/15")
)


def _ip_is_blocked(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # unparseable -> treat as blocked
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    return any(ip in net for net in _EXTRA_BLOCKED_NETS)


def _resolve_and_check(host: str) -> str | None:
    """Return the resolved IP for `host` if it's safe; otherwise None."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    for _family, _, _, _, sockaddr in infos:
        addr = sockaddr[0]
        if _ip_is_blocked(addr):
            return None
    return infos[0][4][0] if infos else None


def _url_ok(url: str) -> bool:
    """Validate the URL syntactically and against blocked schemes/hosts."""
    try:
        u = urlparse(url)
    except ValueError:
        return False
    if u.scheme.lower() in _BLOCKED_SCHEMES:
        return False
    if u.scheme not in ("http", "https"):
        return False
    if not u.hostname:
        return False
    if _resolve_and_check(u.hostname) is None:
        return False
    return True


class _RedirectBlocked(Exception):
    """A redirect hop pointed at a disallowed (e.g. private/metadata) URL."""


def _guarded_get(client: httpx.Client, url: str, headers: dict, **kwargs):
    """GET `url`, following redirects MANUALLY so each hop is validated with
    _url_ok BEFORE we connect to it.

    httpx's follow_redirects=True connects to every intermediate URL itself,
    which defeats the SSRF guard: a public URL can 302 to
    http://169.254.169.254/ (cloud metadata) and httpx would fetch it, with
    only the final URL re-checked afterwards. Here nothing internal is ever
    contacted."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        r = client.get(current, headers=headers, follow_redirects=False, **kwargs)
        if not r.is_redirect:
            return r
        location = r.headers.get("location")
        if not location:
            return r
        nxt = urljoin(current, location)
        if not _url_ok(nxt):
            raise _RedirectBlocked(nxt)
        current = nxt
    raise _RedirectBlocked(f"too many redirects (>{MAX_REDIRECTS})")


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script\b.*?</script>", " ", html)
    html = re.sub(r"(?is)<style\b.*?</style>", " ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def web_search(query: str, *, max_results: int = 5) -> str:
    """Search the public web. Returns a JSON string of the top results.

    The search API is the corporate egress proxy's search endpoint, not
    an arbitrary third party. Configure via `ALEX_SEARCH_URL`."""
    base = os.environ.get("ALEX_SEARCH_URL", "").strip()
    if not base:
        return "[error] web search is not configured (ALEX_SEARCH_URL is empty)"
    url = urljoin(base.rstrip("/") + "/", "search")
    if not _url_ok(url):
        return "[error] search endpoint is not allowed"
    headers = {"Accept": "application/json"}
    proxy = PROXY or None
    try:
        with httpx.Client(timeout=15.0, proxy=proxy) as client:
            r = _guarded_get(
                client, url, headers, params={"q": query, "limit": max_results}
            )
    except _RedirectBlocked as e:
        return f"[error] search redirect to a disallowed target: {e}"
    except httpx.HTTPError as e:
        return f"[error] search failed: {e}"
    if r.status_code >= 400:
        return f"[error] search HTTP {r.status_code}"
    return r.text[:20_000]


def web_fetch(url: str, *, hops: dict | None = None) -> str:
    """Fetch a single page and return the visible text.

    `hops` is an internal counter passed by the agent to enforce a
    maximum number of fetches per turn (the agent re-uses the same
    dict across calls)."""
    if hops is not None:
        hops["n"] = hops.get("n", 0) + 1
        if hops["n"] > MAX_HOPS:
            return f"[error] exceeded {MAX_HOPS} web fetches this turn"
    if not _url_ok(url):
        return f"[error] URL {url!r} is not allowed (blocked scheme, host, or IP)"
    headers = {"User-Agent": "Alex-Agent/2.0", "Accept": "text/html,application/json"}
    proxy = PROXY or None
    try:
        with httpx.Client(timeout=15.0, proxy=proxy) as client:
            r = _guarded_get(client, url, headers)
    except _RedirectBlocked as e:
        return f"[error] redirect target {str(e)!r} is not allowed"
    except httpx.HTTPError as e:
        return f"[error] fetch {url} failed: {e}"
    if r.status_code >= 400:
        return f"[error] HTTP {r.status_code} from {url}"
    ctype = r.headers.get("content-type", "")
    if "json" in ctype:
        return r.text[:MAX_PAGE_CHARS]
    text = _strip_html(r.text)
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"\n[truncated to {MAX_PAGE_CHARS} chars]"
    return text

# The LangChain tool factory that used to live here has been removed.
# It wrapped these functions directly, so a LangChain caller got them
# with no ACL check, no governor, no constitution and no audit row —
# and one of them wrote to an internal API. agent/langchain_tools.py
# exports the same capability through VM._invoke_tool instead, which
# is where all of that lives. An ungoverned export sitting beside a
# governed one is a trap, not a convenience.
