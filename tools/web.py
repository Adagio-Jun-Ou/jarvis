"""Web tools for Jarvis - search, fetch, download images, and more."""
import asyncio
import logging
import os
import re
import base64
import hashlib
import aiofiles
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
from urllib.parse import urlparse

from .base import Tool
from core.helpers import safe_error_message

if TYPE_CHECKING:
    # httpx is an optional runtime dependency (imported lazily inside the
    # fetch paths); the annotations below need it only for type checkers.
    import httpx

logger = logging.getLogger(__name__)

UNSAFE_CONTENT_PATTERNS = {
    'adult': [
        r'\b(porn|xxx|nsfw|nude|nudes|naked|sex|sexual|erotic|escort|prostitute)\b',
        r'\b(onlyfans|pornhub|xvideos|xnxx|redtube|youporn|brazzers|bangbus)\b',
        r'\b(adult\s*(content|video|site|entertainment))\b',
    ],
    'violence': [
        r'\b(gore|beheading|execution|torture|snuff|killing\s*video)\b',
        r'\b(mass\s*shooter|terrorist\s*manual|bomb\s*making)\b',
        r'\b(isis|al\s*qaeda|terrorist\s*group)\b',
    ],
    'harmful': [
        r'\b(suicide|kill\s*yourself|self\s*harm|how\s*to\s*die)\b',
        r'\b(anorexia\s*tips|bulimia\s*tips|pro\s*ana)\b',
        r'\b(drug\s*synthesis|meth\s*recipe|crack\s*cooking)\b',
        r'\b(how\s*to\s*make\s*(bomb|explosive|poison|weapon))\b',
        r'\b(how\s*to\s*(make|cook|manufacture)\s*(meth|crack|heroin|fentanyl|lsd|ecstasy))\b',
    ],
    'hate': [
        r'\b(hate\s*group|white\s*supremacist|neo\s*nazi|kkk)\b',
        r'\b(racist\s*content|anti\s*semitic|holocaust\s*denial)\b',
    ],
    'illegal': [
        r'\b(child\s*porn|csam|cp\b|pedo(phile)?\b)\b',
        r'\b(illegal\s*(drugs|weapons|explosives))\b',
        r'\b(identity\s*theft|credit\s*card\s*fraud)\b',
        r'\b(hack\s*(bank|paypal|account))\b',
        r'\b(buy\s*(drugs|cocaine|heroin|meth|fentanyl)\s*online)\b',
    ],
    'malware': [
        r'\b(malware\s*download|virus\s*download|ransomware)\b',
        r'\b(computer\s*virus|trojan\s*horse|keylogger)\b',
    ],
}

SAFE_SEARCH_HINTS = {
    'adult': 'This search involves adult/sexual content. Add `unsafe: true` parameter if you intentionally want to search this.',
    'violence': 'This search involves violent/gory content. Add `unsafe: true` parameter if you intentionally want to search this.',
    'harmful': 'This search could lead to harmful content. Add `unsafe: true` parameter if you intentionally want to search this.',
    'hate': 'This search involves hate speech or extremist content. Add `unsafe: true` parameter if you intentionally want to search this.',
    'illegal': 'This search may involve illegal content and is blocked.',
    'malware': 'This search may lead to malware or malicious content and is blocked.',
}

BLOCKED_CATEGORIES = {'illegal', 'malware'}


def check_safe_search(query: str) -> tuple[bool, str, str]:
    """Check if a search query is safe.
    
    Returns:
        Tuple of (is_safe, category_if_unsafe, hint_message)
    """
    query_lower = query.lower()
    
    # Check blocked categories first (illegal, malware)
    for category in BLOCKED_CATEGORIES:
        patterns = UNSAFE_CONTENT_PATTERNS.get(category, [])
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                hint = SAFE_SEARCH_HINTS.get(category, '')
                return False, category, f"This search may involve {category} content and is blocked."
    
    # Then check other categories
    for category, patterns in UNSAFE_CONTENT_PATTERNS.items():
        if category in BLOCKED_CATEGORIES:
            continue  # Already checked
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                hint = SAFE_SEARCH_HINTS.get(category, '')
                return False, category, hint
    
    return True, '', ''


def check_url_safety(url: str) -> tuple[bool, str, str]:
    """Check if a URL appears safe based on domain and path patterns.
    
    Returns:
        Tuple of (is_safe, category_if_unsafe, hint_message)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.hostname.lower() if parsed.hostname else ''
        parsed.path.lower()
        full_url = url.lower()
        
        blocked_domains = {
            'pornhub.com', 'xvideos.com', 'xnxx.com', 'redtube.com', 'youporn.com',
            'xhamster.com', 'spankbang.com', 'youjizz.com', 'beeg.com',
        }
        
        if domain in blocked_domains or any(domain.endswith('.' + d) for d in blocked_domains):
            return False, 'adult', SAFE_SEARCH_HINTS.get('adult', '')
        
        for category, patterns in UNSAFE_CONTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_url, re.IGNORECASE):
                    hint = SAFE_SEARCH_HINTS.get(category, '')
                    if category in BLOCKED_CATEGORIES:
                        return False, category, f"This URL may contain {category} content and is blocked."
                    return False, category, hint
        
    except Exception as e:
        logger.warning(f"Failed to validate URL safety: {e}")
    
    return True, '', ''

BLOCKED_URL_SCHEMES = {'file', 'ftp', 'sftp', 'ssh', 'telnet', 'gopher', 'data', 'javascript', 'vbscript'}
BLOCKED_NETWORKS = [
    '127.',
    '169.254.',
    '10.',
    '172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
    '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.',
    '192.168.',
    'localhost',
    '::1',
    '0.0.0.0',
]
BLOCKED_HOSTNAMES = {
    'localhost',
    'localhost.localdomain',
    'ip6-localhost',
    'ip6-loopback',
    'metadata.google.internal',
    'metadata',
    'kubernetes.default',
    'kubernetes.default.svc',
}
BLOCKED_IPS = [
    '0.0.0.0',
    '127.0.0.0',
    '169.254.169.254',
]


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL is safe to fetch (SSRF protection).

    Applies, in order:
      1. Local scheme/hostname/network denylists (above).
      2. Resolved-IP checks via the shared protector (private, loopback,
         link-local, reserved, multicast and custom-denylisted space). This is
         the same authoritative resolution — and cache — that ``_pinned_fetch``
         uses to pin the connection, so validation and connect can never
         observe different addresses.
      3. core.security.ssrf_protection.SSRFProtector's full URL policy, which
         adds CIDR-precise internal ranges, IPv6 internal space, denylisted
         service ports, internal-only TLDs, embedded credentials and suspicious
         query encoding.

    Layers 2-3 can only ever add denials, never relax an earlier one.

    The DNS-rebinding TOCTOU window this function used to leave open is closed
    by ``_pinned_fetch``, which connects to the validated IP directly instead
    of letting the HTTP stack re-resolve the hostname.

    Returns:
        Tuple of (is_safe, reason_if_unsafe)
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"
    
    if parsed.scheme.lower() not in {'http', 'https'}:
        if parsed.scheme.lower() in BLOCKED_URL_SCHEMES:
            return False, f"Blocked URL scheme: {parsed.scheme}"
        return False, f"Unsupported URL scheme: {parsed.scheme}"
    
    if not parsed.hostname:
        return False, "URL must have a hostname"
    
    hostname = parsed.hostname.lower()
    
    for blocked in BLOCKED_NETWORKS:
        if hostname.startswith(blocked) or hostname == blocked.rstrip('.'):
            return False, f"Blocked internal/metadata address: {hostname}"
    
    if hostname in BLOCKED_HOSTNAMES:
        return False, f"Blocked hostname: {hostname}"
    
    if hostname in BLOCKED_IPS:
        return False, f"Blocked IP address: {hostname}"

    # Layer 2: resolve once, through the shared protector, so the exact same
    # resolution (and its cache) is reused when _pinned_fetch pins the
    # connection to one of these addresses. Fails if ANY resolved address is
    # private/loopback/link-local/reserved/multicast or custom-denylisted.
    try:
        from core.security.ssrf_protection import resolve_validated_ips

        resolved_ips, resolve_error = resolve_validated_ips(hostname)
        if resolve_error or not resolved_ips:
            return False, resolve_error or f"Could not resolve hostname: {hostname}"
    except ImportError:
        logger.debug("ssrf_protection unavailable; using local URL checks only")
        import socket
        import ipaddress
        try:
            resolved_ips = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for _, _, _, _, addr in resolved_ips:
                ip_str = addr[0]
                try:
                    ip_obj = ipaddress.ip_address(ip_str)
                    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                        return False, f"Resolved IP {ip_str} is a private/internal address"
                except ValueError:
                    continue
        except Exception:
            return False, f"Could not resolve hostname: {hostname}"
    except Exception as e:
        return False, f"Could not resolve hostname: {hostname} ({safe_error_message(e)})"

    # Defence in depth: the shared protector understands CIDR ranges, IPv6
    # internal space, denylisted service ports and internal-only TLDs that the
    # string-prefix lists above cannot express. It is advisory-only in the sense
    # that it can add a denial but never grant access already refused above.
    try:
        from core.security.ssrf_protection import validate_outbound_url

        verdict = validate_outbound_url(url)
        if not verdict.valid:
            return False, verdict.error or "Blocked by SSRF policy"
    except ImportError:
        logger.debug("ssrf_protection unavailable; using local URL checks only")
    except Exception as e:
        logger.warning(f"SSRF protector failed open for {hostname}: {e}")

    return True, ""


class WebFetchError(Exception):
    """Raised by _pinned_fetch when a URL is blocked or unreachable.

    The message is safe to surface to the caller directly.
    """

    def __init__(self, message: str, url: str = "", kind: str = "error"):
        super().__init__(message)
        self.url = url
        self.kind = kind  # "blocked" | "redirect_blocked" | "connect" | "no_response"


def _is_ip_literal(host: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _host_header_value(url: "httpx.URL") -> str:
    """RFC form: bare host for scheme-default ports, host:port otherwise.

    Uses the IDNA-encoded ASCII host — the same bytes httpcore would
    auto-generate for an unpinned request (httpcore builds its origin from
    ``request.url.raw_host``), so pinned requests look identical to unpinned
    ones on the wire, including for internationalized domains.
    """
    host = url.raw_host.decode("ascii")
    default_port = 443 if url.scheme == "https" else 80
    if url.port is None or url.port == default_port:
        return host
    return f"{host}:{url.port}"


async def _pinned_fetch(
    client: "httpx.AsyncClient",
    url: str,
    *,
    max_redirects: int = 5,
) -> tuple["httpx.Response", str]:
    """Fetch *url* with the connection pinned to its validated IP.

    This closes the DNS-rebinding TOCTOU window that ``is_safe_url`` alone
    cannot: validation and connect now share one authoritative resolution —
    ``core.security.ssrf_protection.resolve_validated_ips`` (same cache) — and
    the connection goes to the validated IP directly:

      * The request URL's host is swapped for the IP (``httpx.URL.copy_with``),
        so the HTTP stack never re-resolves the name. anyio skips DNS for IP
        literals, so there is no gap between check and use.
      * TLS keeps validating the certificate against the original hostname via
        the ``sni_hostname`` request extension (honoured by httpcore).
      * The ``Host`` header keeps the original hostname (IDNA form) so virtual
        hosting and CDNs behave exactly as before.
      * If the hostname was already an IP literal, no pinning is needed (no DNS
        was involved); the pre-flight checks already covered it.

    Redirects are followed manually; every hop is re-validated with
    ``is_safe_url`` and re-pinned, so a redirect cannot smuggle the connection
    to an internal address.

    Raises:
        WebFetchError: blocked, unsafe redirect, or all validated addresses
        refused the connection.

    Returns:
        ``(response, final_url)`` — the non-redirect response and the URL it
        came from (after following redirects).
    """
    import httpx

    current_url = url
    response: "httpx.Response | None" = None

    for _redirect in range(max_redirects + 1):
        is_safe, reason = is_safe_url(current_url)
        if not is_safe:
            raise WebFetchError(
                f"{reason}", url=current_url, kind="blocked"
            )

        hx_url = httpx.URL(current_url)
        original_host = hx_url.host
        raw_host = hx_url.raw_host.decode("ascii")  # IDNA form, matches httpcore's origin

        if _is_ip_literal(original_host):
            # Nothing to pin — connecting by IP does no DNS.
            response = await client.get(current_url, follow_redirects=False)
        else:
            from core.security.ssrf_protection import resolve_validated_ips

            ips, resolve_error = resolve_validated_ips(original_host)
            if resolve_error or not ips:
                raise WebFetchError(
                    resolve_error or f"Could not resolve {original_host}",
                    url=current_url,
                    kind="blocked",
                )

            headers = {"Host": _host_header_value(hx_url)}
            extensions = {"sni_hostname": raw_host}

            last_connect_error: "httpx.HTTPError | None" = None
            response = None
            for ip in ips:
                pinned_url = hx_url.copy_with(host=ip)
                try:
                    response = await client.get(
                        pinned_url,
                        headers=headers,
                        extensions=extensions,
                        follow_redirects=False,
                    )
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    last_connect_error = e
                    continue
            if response is None:
                detail = safe_error_message(last_connect_error) if last_connect_error else "unknown"
                raise WebFetchError(
                    f"Could not connect to {original_host} "
                    f"({len(ips)} validated address(es)): {detail}",
                    url=current_url,
                    kind="connect",
                )

        if response.status_code in (301, 302, 303, 307, 308):
            redirect_url = response.headers.get("location")
            if not redirect_url:
                break
            redirect_url = str(httpx.URL(current_url).join(redirect_url))
            is_safe_redirect, redirect_reason = is_safe_url(redirect_url)
            if not is_safe_redirect:
                raise WebFetchError(
                    f"redirect to unsafe URL - {redirect_reason}",
                    url=redirect_url,
                    kind="redirect_blocked",
                )
            current_url = redirect_url
            continue
        break

    if response is None:
        raise WebFetchError("no response received", url=current_url, kind="no_response")
    return response, current_url


@dataclass
class SearchConfig:
    """Configuration for web search."""
    google_api_key: Optional[str] = None
    google_engine_id: Optional[str] = None
    use_google: bool = True

IMAGE_CACHE_DIR = Path.home() / ".jarvis" / "image_cache"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class WebSearchTool(Tool):
    """Tool for web search with automatic content fetching and safe search."""

    name = "web_search"
    description = "Search the web and fetch content from top results. Safe search is ON by default - use unsafe: true for sensitive content (requires permission)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10)",
            },
            "fetch_content": {
                "type": "boolean",
                "description": "Fetch and include content from top results (default True)",
            },
            "use_google": {
                "type": "boolean",
                "description": "Use Google search (default True)",
            },
            "unsafe": {
                "type": "boolean",
                "description": "Bypass safe search filters - requires permission (default False)",
            },
        },
        "required": ["query"],
    }
    requires_permission = False  # Permission only required when unsafe=true

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_engine_id: Optional[str] = None,
    ):
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.google_engine_id = google_engine_id or os.environ.get("GOOGLE_ENGINE_ID")
        self._ddg_available = None
        self._httpx_available = None

    async def _check_ddg(self) -> bool:
        """Check if DuckDuckGo search is available."""
        if self._ddg_available is not None:
            return self._ddg_available
        try:
            from duckduckgo_search import DDGS
            self._ddg_available = True
        except ImportError:
            self._ddg_available = False
        return self._ddg_available

    async def _check_httpx(self) -> bool:
        """Check if httpx is available."""
        if self._httpx_available is not None:
            return self._httpx_available
        try:
            import httpx
            self._httpx_available = True
        except ImportError:
            self._httpx_available = False
        return self._httpx_available

    async def execute(
        self,
        query: str = "",
        num_results: int = 5,
        fetch_content: bool = True,
        use_google: bool = True,
        unsafe: bool = False,
        **kwargs  # Accept extra params for flexibility
    ) -> str:
        """Execute a web search and optionally fetch content from results.
        
        Args:
            query: The search query string
            num_results: Number of results to return (default 5, max 10)
            fetch_content: Fetch and include content from top results (default True)
            use_google: Use Google search (default True)
            unsafe: Bypass safe search filters (default False) - requires permission
            **kwargs: Additional parameters for LLM flexibility (queries, q, search)
        
        Returns:
            Search results as formatted string
        """
        # Handle common LLM parameter variations
        if not query:
            query = kwargs.get('queries', kwargs.get('q', kwargs.get('search', '')))
        
        if not query:
            return "Error: No search query provided. Use 'query' parameter."

        # Safe search check
        is_safe, category, hint = check_safe_search(query)
        if not is_safe:
            if category in BLOCKED_CATEGORIES:
                return f"⚠️ Search blocked for safety reasons.\n\n{hint}\n\nThis category cannot be searched even with unsafe mode."
            if not unsafe:
                return f"PERMISSION_REQUIRED: Safe search filter triggered for category: {category}.\n\n{hint}\n\nTo proceed, use: unsafe=true"

        # Try Google first if API key is available
        search_results = None
        if use_google and self.google_api_key and self.google_engine_id:
            search_results = await self._google_search(query, num_results)

        # Fall back to DuckDuckGo
        if not search_results and await self._check_ddg():
            search_results = await self._ddg_search(query, num_results)

        if not search_results:
            return "Web search unavailable. Set GOOGLE_API_KEY and GOOGLE_ENGINE_ID for Google search, or install duckduckgo-search."

        # If fetch_content is True, fetch content from top URLs
        if fetch_content and await self._check_httpx():
            urls_to_fetch = self._extract_urls_from_results(search_results, max_urls=3)
            if urls_to_fetch:
                fetched_content = await self._fetch_multiple_urls(urls_to_fetch, max_length=2000)
                if fetched_content:
                    return f"{search_results}\n\n---\n\n## Content from Top Results:\n\n{fetched_content}"

        return search_results

    def _extract_urls_from_results(self, results: str, max_urls: int = 3) -> list[str]:
        """Extract URLs from search results."""
        import re
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', results)
        # Filter out common non-content URLs
        filtered = [u for u in urls if not any(x in u.lower() for x in ['youtube.com/watch', 'facebook.com/', 'twitter.com/', 'x.com/', 'instagram.com/'])]
        return filtered[:max_urls]

    async def _fetch_multiple_urls(self, urls: list[str], max_length: int = 2000) -> str:
        """Fetch content from multiple URLs with SSRF protection."""
        import httpx
        from bs4 import BeautifulSoup
        import asyncio


        async def fetch_one(url: str) -> str:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response, _final_url = await _pinned_fetch(client, url)
                    response.raise_for_status()
            except WebFetchError as e:
                return f"### {url}\n(Blocked: {e})"
            except Exception as e:
                return f"### {url}\n(Error fetching: {safe_error_message(e)[:50]})"

            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
                element.decompose()

            main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|article|post|entry"))
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)

            if len(text) > max_length:
                text = text[:max_length] + "..."

            return f"### {url}\n{text}"

        import re
        contents = await asyncio.gather(*[fetch_one(url) for url in urls])
        return "\n\n".join(contents)

    async def _google_search(self, query: str, num_results: int) -> Optional[str]:
        """Search using Google Custom Search API."""
        if not await self._check_httpx():
            return None

        try:
            import httpx

            # Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_engine_id,
                "q": query,
                "num": min(num_results, 10),  # Google CSE max is 10
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            items = data.get("items", [])
            if not items:
                return f"No results found for: {query}"

            output = []
            for i, item in enumerate(items, 1):
                title = item.get("title", "No title")
                link = item.get("link", "No URL")
                snippet = item.get("snippet", "No description")
                output.append(
                    f"{i}. **{title}**\n"
                    f"   URL: {link}\n"
                    f"   {snippet}"
                )

            return "\n\n".join(output)

        except Exception as e:
            print(f"Google search error: {e}")
            return None

    async def _ddg_search(self, query: str, num_results: int) -> str:
        """Search using DuckDuckGo."""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            loop = asyncio.get_running_loop()
            with DDGS() as ddgs:
                results = await loop.run_in_executor(
                    None, lambda: list(ddgs.text(query, max_results=num_results))
                )

            if not results:
                return f"No results found for: {query}"

            output = []
            for i, result in enumerate(results, 1):
                output.append(
                    f"{i}. **{result.get('title', 'No title')}**\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')[:300]}..."
                )

            return "\n\n".join(output)

        except Exception as e:
            return f"Search error: {safe_error_message(e)}"


class WebFetchTool(Tool):
    """Tool for fetching web content with safe URL checking."""

    name = "web_fetch"
    description = "Fetch and extract text content from a URL. Safe URL checking is ON by default - use unsafe: true for restricted URLs (requires permission)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length (default 5000 chars)",
            },
            "unsafe": {
                "type": "boolean",
                "description": "Bypass safe URL checks - requires permission (default False)",
            },
        },
        "required": ["url"],
    }
    requires_permission = False  # Permission only required when unsafe=true

    def __init__(self):
        self._httpx_available = None

    async def _check_httpx(self) -> bool:
        """Check if httpx is available."""
        if self._httpx_available is not None:
            return self._httpx_available

        try:
            import httpx
            self._httpx_available = True
        except ImportError:
            self._httpx_available = False

        return self._httpx_available

    async def execute(self, url: str = "", max_length: int = 5000, unsafe: bool = False, **kwargs) -> str:
        """Fetch content from a URL.
        
        Args:
            url: URL to fetch
            max_length: Maximum content length (default 5000 chars)
            unsafe: Bypass safe URL checks (default False)
            **kwargs: Additional parameters for LLM flexibility (uri, link, address)
        
        Returns:
            Content from URL or error message
        """
        # Handle common parameter variations
        if not url:
            url = kwargs.get('uri', kwargs.get('link', kwargs.get('address', '')))
        
        if not url:
            return "Error: No URL provided. Use 'url' parameter."
        
        is_network_safe, reason = is_safe_url(url)
        if not is_network_safe:
            return f"URL blocked for security: {reason}"
        
        # Check URL safety
        is_safe, category, hint = check_url_safety(url)
        if not is_safe:
            if category in BLOCKED_CATEGORIES:
                return f"⚠️ URL blocked for safety reasons.\n\n{hint}\n\nThis category cannot be accessed even with unsafe mode."
            if not unsafe:
                return f"PERMISSION_REQUIRED: Safe URL filter triggered for category: {category}.\n\n{hint}\n\nTo proceed, use: unsafe=true"
        
        if not await self._check_httpx():
            return "HTTP client unavailable. Install with: pip install httpx beautifulsoup4"

        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=30.0) as client:
                response, _final_url = await _pinned_fetch(client, url)
                response.raise_for_status()

            # Parse HTML and extract text
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)

            if len(text) > max_length:
                text = text[:max_length] + "\n... (truncated)"

            return f"Content from {url}:\n\n{text}"

        except WebFetchError as e:
            return f"URL blocked for security: {e}"
        except Exception as e:
            return f"Error fetching URL: {safe_error_message(e)}"


class ImageSearchTool(Tool):
    """Tool for searching images on the web with safe search."""

    name = "image_search"
    description = "Search for images on the web. Safe search is ON by default - use unsafe: true for sensitive content (requires permission)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The image search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results (default 5, max 20)",
            },
            "safe_search": {
                "type": "boolean",
                "description": "Enable safe search (default True)",
            },
            "unsafe": {
                "type": "boolean",
                "description": "Bypass safe search content filters - requires permission (default False)",
            },
        },
        "required": ["query"],
    }
    requires_permission = False  # Permission only required when unsafe=true

    async def execute(
        self,
        query: str = "",
        num_results: int = 5,
        safe_search: bool = True,
        unsafe: bool = False,
        **kwargs
    ) -> str:
        """Search for images using multiple methods."""
        # Handle common LLM parameter variations
        if not query:
            query = kwargs.get('queries', kwargs.get('q', kwargs.get('search', '')))
        
        if not query:
            return "Error: No search query provided. Use 'query' parameter."
        
        is_query_safe, category, hint = check_safe_search(query)
        if not is_query_safe:
            if category in BLOCKED_CATEGORIES:
                return f"⚠️ Image search blocked for safety reasons.\n\n{hint}\n\nThis category cannot be searched even with unsafe mode."
            if not unsafe:
                return f"PERMISSION_REQUIRED: Safe search filter triggered for category: {category}.\n\n{hint}\n\nTo proceed, use: unsafe=true"
        
        try:
            import httpx
            import urllib.parse

            results = []

            # Method 1: Direct DuckDuckGo images API with proper headers
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/javascript, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://duckduckgo.com/",
                }

                async with httpx.AsyncClient(headers=headers, timeout=15.0, follow_redirects=True, max_redirects=5) as client:
                    # First get vqd token
                    search_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}&iax=images&ia=images"
                    resp = await client.get(search_url)
                    vqd_match = re.search(r'vqd[\'"]?\s*[:=]\s*[\'"]([^\'"]+)[\'"]', resp.text)

                    if vqd_match:
                        vqd = vqd_match.group(1)
                        # Now search images
                        api_url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={urllib.parse.quote(query)}&vqd={vqd}&f=,,,&p={'1' if safe_search else '-1'}"
                        img_resp = await client.get(api_url)

                        if img_resp.status_code == 200:
                            data = img_resp.json()
                            for item in data.get("results", [])[:num_results]:
                                results.append({
                                    "title": item.get("title", "No title"),
                                    "image": item.get("image", item.get("url", "")),
                                    "thumbnail": item.get("thumbnail", ""),
                                    "source": item.get("source", "Unknown"),
                                    "width": item.get("width", "?"),
                                    "height": item.get("height", "?"),
                                })
            except Exception as e:
                logger.debug(f"DuckDuckGo direct API failed: {e}")

            # Method 2: Try ddgs library with headers
            if len(results) < num_results:
                try:
                    try:
                        from ddgs import DDGS
                    except ImportError:
                        from duckduckgo_search import DDGS

                    with DDGS(headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }) as ddgs:
                        ddgs_results = list(ddgs.images(
                            query,
                            max_results=min(num_results, 10),
                            safesearch="on" if safe_search else "off"
                        ))
                        for item in ddgs_results:
                            if len(results) >= num_results:
                                break
                            results.append({
                                "title": item.get("title", "No title"),
                                "image": item.get("image", item.get("url", "")),
                                "thumbnail": item.get("thumbnail", ""),
                                "source": item.get("source", item.get("host", "Unknown")),
                                "width": item.get("width", "?"),
                                "height": item.get("height", "?"),
                            })
                except Exception as e:
                    logger.debug(f"DDGS library failed: {e}")

            # Method 3: Bing Image Search via RSS (public, no API key)
            if len(results) < num_results:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        bing_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&format=rss&count={num_results}"
                        resp = await client.get(bing_url)
                        if resp.status_code == 200:
                            try:
                                import defusedxml.ElementTree as ET
                            except ImportError:
                                import xml.etree.ElementTree as ET
                            root = ET.fromstring(resp.text)
                            for item in root.findall(".//item")[:num_results - len(results)]:
                                title = item.find("title")
                                link = item.find("link")
                                thumbnail = item.find("{http://search.yahoo.com/mrss/}thumbnail")
                                results.append({
                                    "title": title.text if title is not None else "No title",
                                    "image": link.text if link is not None else "",
                                    "thumbnail": thumbnail.attrib.get("url", "") if thumbnail is not None else "",
                                    "source": "Bing",
                                    "width": "?",
                                    "height": "?",
                                })
                except Exception as e:
                    logger.debug(f"Bing RSS failed: {e}")

            # Method 4: Generate direct search links as fallback
            if not results:
                encoded_query = urllib.parse.quote(query)
                return f"""## Image Search Results for: {query}

No images could be retrieved automatically (APIs may be rate limited). 

Try these direct image search links:
- **DuckDuckGo**: https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images
- **Google Images**: https://www.google.com/search?tbm=isch&q={encoded_query}
- **Bing Images**: https://www.bing.com/images/search?q={encoded_query}

To download an image, provide the direct image URL (e.g., `download image from https://example.com/image.jpg`)"""

            # Format results
            output = [f"## Image Search Results for: {query}\n"]
            for i, result in enumerate(results, 1):
                output.append(
                    f"{i}. **{result['title']}**\n"
                    f"   Image: {result['image']}\n"
                    f"   Thumbnail: {result['thumbnail']}\n"
                    f"   Source: {result['source']}\n"
                    f"   Size: {result['width']}x{result['height']}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Image search error: {safe_error_message(e)}\n\nTry searching directly: https://duckduckgo.com/?q={query}&ia=images"


class ImageDownloadTool(Tool):
    """Tool for downloading images from URLs."""

    name = "image_download"
    description = "Download an image from a URL and save it locally. Safe URL checking is ON by default - use unsafe: true for restricted URLs (requires permission)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the image to download",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename (auto-generated if not provided)",
            },
            "return_base64": {
                "type": "boolean",
                "description": "Return base64 encoded image data (default False)",
            },
            "unsafe": {
                "type": "boolean",
                "description": "Bypass safe URL checks - requires permission (default False)",
            },
        },
        "required": ["url"],
    }
    requires_permission = False  # Permission only required when unsafe=true

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or IMAGE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        url: str = "",
        filename: Optional[str] = None,
        return_base64: bool = False,
        max_size_bytes: int = 50 * 1024 * 1024,  # 50MB limit
        unsafe: bool = False,
        **kwargs
    ) -> str:
        """Download an image with SSRF protection, size limits, and safety checks."""
        is_network_safe, reason = is_safe_url(url)
        if not is_network_safe:
            return f"URL blocked: {reason}"
        
        is_content_safe, category, hint = check_url_safety(url)
        if not is_content_safe:
            if category in BLOCKED_CATEGORIES:
                return f"⚠️ Image download blocked for safety reasons.\n\n{hint}\n\nThis category cannot be downloaded even with unsafe mode."
            if not unsafe:
                return f"PERMISSION_REQUIRED: Safe URL filter triggered for category: {category}.\n\n{hint}\n\nTo proceed, use: unsafe=true"
        
        try:
            import httpx
            from PIL import Image
            from io import BytesIO

            async with httpx.AsyncClient(timeout=30.0) as client:
                response, _final_url = await _pinned_fetch(client, url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return f"URL does not point to an image. Content-Type: {content_type}"

            image_data = response.content
            
            if len(image_data) > max_size_bytes:
                return f"Image too large: {len(image_data)} bytes (max: {max_size_bytes})"

            ext = content_type.split("/")[-1].split(";")[0]
            if ext == "jpeg":
                ext = "jpg"

            if filename:
                safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                filepath = self.cache_dir / safe_filename
            else:
                url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
                filepath = self.cache_dir / f"image_{url_hash}.{ext}"

            async with aiofiles.open(filepath, "wb") as f:
                await f.write(image_data)

            try:
                img = Image.open(BytesIO(image_data))
                width, height = img.size
                format_name = img.format or ext.upper()
            except Exception:
                width, height, format_name = "?", "?", ext.upper()

            result = [
                "Image downloaded successfully",
                f"  URL: {url}",
                f"  Saved to: {filepath}",
                f"  Format: {format_name}",
                f"  Size: {len(image_data)} bytes",
                f"  Dimensions: {width}x{height}",
            ]

            if return_base64:
                b64_data = base64.b64encode(image_data).decode("utf-8")
                result.append(f"  Base64: {b64_data[:100]}...")

            return "\n".join(result)

        except WebFetchError as e:
            return f"Image download blocked: {e}"
        except Exception as e:
            return f"Error downloading image: {safe_error_message(e)}"


class ImageReadTool(Tool):
    """Tool for reading and analyzing local images."""

    name = "image_read"
    description = "Read a local image file and get its metadata or base64 data"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the local image file",
            },
            "return_base64": {
                "type": "boolean",
                "description": "Return base64 encoded image data (default False)",
            },
            "analyze": {
                "type": "boolean",
                "description": "Provide basic image analysis (colors, dimensions)",
            },
        },
        "required": ["path"],
    }
    requires_permission = True  # File access requires permission

    async def execute(
        self,
        path: str = "",
        return_base64: bool = False,
        analyze: bool = True,
        **kwargs
    ) -> str:
        """Read and optionally analyze a local image."""
        try:
            from PIL import Image

            img_path = Path(path).resolve()
            allowed_dirs = [Path.home() / ".jarvis", Path.home() / "projects", Path.cwd()]
            if not any(str(img_path).startswith(str(d.resolve())) for d in allowed_dirs):
                return "Error: Path must be within allowed directories"
            if not img_path.exists():
                return f"Image file not found: {path}"

            with Image.open(img_path) as img:
                width, height = img.size
                format_name = img.format or "Unknown"
                mode = img.mode

                result = [
                    f"📁 Image: {img_path.name}",
                    f"  Path: {img_path}",
                    f"  Format: {format_name}",
                    f"  Mode: {mode}",
                    f"  Dimensions: {width}x{height}",
                    f"  File size: {img_path.stat().st_size} bytes",
                ]

                if analyze:
                    import numpy as np
                    if img.mode not in ("P", "L"):
                        if img.mode == "RGBA":
                            img_rgb = img.convert("RGB")
                        else:
                            img_rgb = img
                        arr = np.array(img_rgb)

                        colors = []
                        for channel, name in [(0, "Red"), (1, "Green"), (2, "Blue")]:
                            mean_val = float(np.mean(arr[:, :, channel]))
                            colors.append(f"{name}: {mean_val:.1f}")

                        result.append(f"  Avg colors: {', '.join(colors)}")

                        brightness = float(np.mean(arr))
                        result.append(f"  Brightness: {brightness:.1f}/255")

                        unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
                        result.append(f"  Unique colors: {unique_colors:,}")

                if return_base64:
                    import io
                    buffer = io.BytesIO()
                    img.save(buffer, format=format_name or "PNG")
                    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    result.append(f"  Base64 length: {len(b64_data)} chars")
                    result.append(f"  Base64 prefix: data:image/{format_name.lower()};base64,{b64_data[:50]}...")

                return "\n".join(result)

        except ImportError:
            import io
            img_path = Path(path)
            if not img_path.exists():
                return f"Image file not found: {path}"

            with open(img_path, "rb") as f:
                image_data = f.read()

            result = [
                f"📁 Image: {img_path.name}",
                f"  Path: {img_path}",
                f"  File size: {len(image_data)} bytes",
            ]

            if return_base64:
                b64_data = base64.b64encode(image_data).decode("utf-8")
                result.append(f"  Base64: {b64_data[:100]}...")

            return "\n".join(result)

        except Exception as e:
            return f"Error reading image: {safe_error_message(e)}"


class ImageAnalyzeTool(Tool):
    """Tool for analyzing images with vision AI."""

    name = "image_analyze"
    description = "Analyze an image using vision AI to describe what it contains. Use this when the user asks to analyze, describe, or understand an image."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the local image file to analyze",
            },
            "question": {
                "type": "string",
                "description": "Optional specific question about the image",
            },
        },
        "required": ["path"],
    }
    requires_permission = False

    _ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.tif'}
    _BLOCKED_PATH_PREFIXES = ['/etc', '/root', '/proc', '/sys', '/dev', '/boot']

    def __init__(self, vision_processor=None):
        self.vision = vision_processor

    @staticmethod
    def _validate_image_path(path: str) -> tuple[Path | None, str]:
        """Validate image path to prevent path traversal and restrict to images."""
        try:
            resolved = Path(path).resolve()
        except Exception:
            return None, f"Invalid path: {path}"

        for prefix in ImageAnalyzeTool._BLOCKED_PATH_PREFIXES:
            if str(resolved).startswith(prefix):
                return None, f"Path is in a restricted directory: {prefix}"

        ext = resolved.suffix.lower()
        if ext not in ImageAnalyzeTool._ALLOWED_EXTENSIONS:
            if resolved.is_dir():
                return None, f"Path is a directory, not an image: {path}"
            return None, f"File extension '{ext}' is not an allowed image type. Allowed: {', '.join(sorted(ImageAnalyzeTool._ALLOWED_EXTENSIONS))}"

        if not resolved.exists():
            return None, f"Image file not found: {path}"

        if not resolved.is_file():
            return None, f"Path is not a regular file: {path}"

        max_size = 50 * 1024 * 1024
        try:
            if resolved.stat().st_size > max_size:
                return None, f"Image file too large (max 50MB): {path}"
        except OSError:
            return None, f"Cannot stat file: {path}"

        return resolved, ""

    async def execute(
        self,
        path: str = "",
        question: str | None = None,
        **kwargs
    ) -> str:
        """Analyze an image with vision AI."""
        if self.vision is None:
            return "Vision analysis not available - no vision processor configured. Make sure a vision model is set in config."

        if not path:
            return "Error: No image path provided."

        validated_path, error = self._validate_image_path(path)
        if error:
            return f"Error: {error}"

        try:
            analysis = await self.vision.analyze_image(validated_path, question=question)
            return f"📷 Image Analysis ({validated_path.name}):\n\n{analysis}"

        except Exception as e:
            return f"Error analyzing image: {safe_error_message(e)}"


class WebTools:
    """Collection of web tools."""

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_engine_id: Optional[str] = None,
        vision_processor=None,
    ):
        self.tools = [
            WebSearchTool(google_api_key, google_engine_id),
            WebFetchTool(),
            ImageSearchTool(),
            ImageDownloadTool(),
            ImageReadTool(),
        ]
        if vision_processor:
            self.tools.append(ImageAnalyzeTool(vision_processor))

    def get_tools(self) -> list[Tool]:
        """Get all web tools."""
        return self.tools
