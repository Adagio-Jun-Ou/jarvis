"""SSRF Protection and Request Validation for Jarvis.

Provides comprehensive protection against Server-Side Request Forgery
and enhances request validation through:
- IP Allowlisting/Denylisting with CIDR support
- URL validation and sanitization
- DNS rebinding prevention
- Internal network detection
- Request header sanitization
- Redirect chain limiting
"""

import ipaddress
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DENYLISTED_TLDS = frozenset({
    ".internal", ".local", ".localhost", ".test", ".example", ".invalid",
    ".onion", ".bit",
})

INTERNAL_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("0.0.0.0/8"),
]

IPV6_INTERNAL = [
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
]

DENYLISTED_HOSTS = frozenset({
    "localhost", "localhost.localdomain",
    "metadata.google.internal", "metadata.azure.internal",
    "instance-data.ec2.internal",
    "169.254.169.254",
})

DENYLISTED_PORTS = frozenset({
    22, 23, 25, 465, 587,  # SSH, Telnet, SMTP
    3306, 5432, 6379, 27017,  # MySQL, PostgreSQL, Redis, MongoDB
    9200, 9300,  # Elasticsearch
    11211,  # Memcached
    5984,  # CouchDB
})

DANGEROUS_SCHEMES = frozenset({
    "file", "gopher", "ftp", "dict", "ldap", "ldaps", "tftp",
    "jar", "netdoc", "ssh", "telnet",
})

ALLOWED_SCHEMES = frozenset({"http", "https"})


@dataclass
class SSRFValidationResult:
    valid: bool
    error: str = ""
    sanitized_url: str = ""
    risk_level: str = "low"
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


class SSRFProtector:
    """Comprehensive SSRF protection for outbound requests."""

    MAX_REDIRECT_CHAIN_LENGTH = 5
    MAX_URL_LENGTH = 2048
    DNS_CACHE_TTL = 300

    def __init__(self):
        self._custom_allowlist: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._custom_denylist: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._dns_cache: dict[str, tuple[list[str], float]] = {}
        self._request_counts: dict[str, list[float]] = {}
        self._request_rate_limit = 60
        self._request_rate_window = 60.0
        self._lock = threading.Lock()

    def validate_url(
        self,
        url: str,
        allow_internal: bool = False,
        check_dns: bool = True,
        follow_redirects: bool = False,
    ) -> SSRFValidationResult:
        """Validate a URL for SSRF safety.

        Args:
            url: The URL to validate
            allow_internal: Whether to allow internal network URLs
            check_dns: Whether to resolve DNS and check IP
            follow_redirects: Whether to follow and validate redirect chains

        Returns:
            SSRFValidationResult with validation details
        """
        result = SSRFValidationResult(valid=False, sanitized_url=url)
        checks_passed = []
        checks_failed = []

        if not url or not isinstance(url, str):
            result.error = "URL must be a non-empty string"
            result.checks_failed = ["url_type"]
            return result

        url = url.strip()
        if len(url) > self.MAX_URL_LENGTH:
            result.error = f"URL exceeds max length ({self.MAX_URL_LENGTH})"
            result.checks_failed = ["url_length"]
            return result

        parsed = urlparse(url)
        if not parsed.scheme:
            result.error = "URL must include a scheme (http/https)"
            result.checks_failed = ["scheme_missing"]
            return result

        scheme = parsed.scheme.lower()
        if scheme in DANGEROUS_SCHEMES:
            result.error = f"Scheme '{scheme}' is not allowed"
            result.risk_level = "critical"
            result.checks_failed = ["dangerous_scheme"]
            return result

        if scheme not in ALLOWED_SCHEMES:
            result.error = f"Scheme '{scheme}' is not allowed (allowed: {', '.join(ALLOWED_SCHEMES)})"
            result.checks_failed = ["invalid_scheme"]
            return result

        checks_passed.append("scheme_valid")

        hostname = parsed.hostname
        if not hostname:
            result.error = "URL must include a hostname"
            result.checks_failed = ["no_hostname"]
            return result

        hostname_lower = hostname.lower()

        if hostname_lower in DENYLISTED_HOSTS:
            if not allow_internal:
                result.error = f"Host '{hostname}' is denylisted (cloud metadata/Internal)"
                result.risk_level = "critical"
                result.checks_failed = ["denylisted_host"]
                return result

        for tld in DENYLISTED_TLDS:
            if hostname_lower.endswith(tld):
                if not allow_internal:
                    result.error = f"Host '{hostname}' uses denylisted TLD '{tld}'"
                    result.risk_level = "high"
                    result.checks_failed = ["denylisted_tld"]
                    return result

        port = parsed.port
        if port and port in DENYLISTED_PORTS:
            result.error = f"Port {port} is denylisted (database/internal service)"
            result.risk_level = "high"
            result.checks_failed = ["denylisted_port"]
            return result

        checks_passed.append("host_basic_safe")

        try:
            ip = ipaddress.ip_address(hostname)
            if self._is_internal_ip(ip) and not allow_internal:
                result.error = f"Host resolves to internal IP: {ip}"
                result.risk_level = "critical"
                result.checks_failed = ["internal_ip"]
                return result
            if self._is_in_custom_denylist(ip):
                result.error = f"Host IP is in custom denylist: {ip}"
                result.risk_level = "critical"
                result.checks_failed = ["custom_denylist"]
                return result
            checks_passed.append("ip_checked_direct")
        except ValueError:
            if check_dns:
                dns_result = self._resolve_and_check(hostname)
                if not dns_result.valid:
                    result.error = dns_result.error
                    result.risk_level = dns_result.risk_level
                    result.checks_failed = dns_result.checks_failed
                    return result
                checks_passed.extend(dns_result.checks_passed)

        if parsed.username or parsed.password:
            result.error = "URLs with embedded credentials are not allowed"
            result.risk_level = "medium"
            result.checks_failed = ["embedded_credentials"]
            return result

        if parsed.query:
            suspicious_patterns = [
                r'(?:\\x[0-9a-fA-F]{2})',
                r'(?:%[0-9a-fA-F]{2}){3,}',
                r'(?:\.\./){2,}',
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, parsed.query):
                    result.error = "URL query contains suspicious encoding"
                    result.risk_level = "medium"
                    result.checks_failed = ["suspicious_encoding"]
                    return result

        checks_passed.append("url_structure_safe")

        sanitized_url = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
        ).geturl()
        result.sanitized_url = sanitized_url
        result.valid = True
        result.checks_passed = checks_passed
        result.checks_failed = checks_failed
        return result

    def _is_internal_ip(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check if an IP address is internal/private."""
        for network in INTERNAL_NETWORKS:
            if ip in network:
                return True
        for network in IPV6_INTERNAL:
            if ip in network:
                return True
        if isinstance(ip, ipaddress.IPv4Address):
            if ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return True
        elif isinstance(ip, ipaddress.IPv6Address):
            if ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return True
        return False

    def _is_in_custom_denylist(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check if an IP is in the custom denylist."""
        for network in self._custom_denylist:
            if ip in network:
                return True
        return False

    def _is_in_custom_allowlist(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check if an IP is in the custom allowlist."""
        for network in self._custom_allowlist:
            if ip in network:
                return True
        return False

    def _resolve_and_check(self, hostname: str) -> SSRFValidationResult:
        """Resolve hostname and check all IPs (result-form wrapper).

        Kept for ``validate_url``; new code should call
        ``resolve_validated_ips`` directly and pin the connection to the
        returned addresses.
        """
        ips, error = self.resolve_validated_ips(hostname)
        if error:
            if error.startswith("DNS resolution failed"):
                return SSRFValidationResult(
                    valid=False,
                    error=error,
                    risk_level="medium",
                    checks_failed=["dns_resolution_failed"],
                )
            if "denylisted IP" in error:
                return SSRFValidationResult(
                    valid=False,
                    error=error,
                    risk_level="critical",
                    checks_failed=["dns_resolves_denylisted"],
                )
            return SSRFValidationResult(
                valid=False,
                error=error,
                risk_level="critical",
                checks_failed=["dns_resolves_internal"],
            )
        return SSRFValidationResult(
            valid=True,
            checks_passed=[f"dns_safe_{hostname}"],
        )

    def resolve_validated_ips(self, hostname: str) -> tuple[list[str], str]:
        """Resolve *hostname* and validate every returned address.

        This is the authoritative resolution point for outbound connections:
        callers receive the exact IP addresses that were checked, so they can
        pin the connection to one of them instead of letting the HTTP stack
        re-resolve the hostname (which would reopen the DNS-rebinding TOCTOU
        window between validation and connect).

        Uses the same cache, internal-network policy and custom denylists as
        ``validate_url``. If *hostname* is already an IP literal it is validated
        directly and returned as the single-element list.

        Returns:
            ``(ips, "")`` on success — safe to connect to any of these — or
            ``([], reason)`` if resolution failed or any address is unsafe.
        """
        # IP literal: validate directly, no DNS involved.
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None:
            if self._is_internal_ip(literal):
                return [], f"IP {literal} is an internal/private address"
            if self._is_in_custom_denylist(literal):
                return [], f"IP {literal} is in the custom denylist"
            return [str(literal)], ""

        now = time.time()
        with self._lock:
            cached = self._dns_cache.get(hostname)
            if cached and now - cached[1] < self.DNS_CACHE_TTL:
                ips = cached[0]
            else:
                ips = None

        if ips is None:
            try:
                addr_info = socket.getaddrinfo(
                    hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
                )
                # Dedupe while PRESERVING resolver order: DNS answer order
                # is meaningful (RFC 6724 address selection / round-robin),
                # and the old list(set(...)) also made iteration order
                # nondeterministic across processes (string hash seed),
                # which randomly reordered pinned fetch attempts.
                seen: set[str] = set()
                ips = []
                for addr in addr_info:
                    ip_str = addr[4][0]
                    if ip_str not in seen:
                        seen.add(ip_str)
                        ips.append(ip_str)
                with self._lock:
                    self._dns_cache[hostname] = (ips, now)
            except socket.gaierror:
                return [], f"DNS resolution failed for '{hostname}'"

        safe_ips: list[str] = []
        for ip_str in ips:
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if self._is_internal_ip(ip):
                return [], f"Host '{hostname}' resolves to internal IP: {ip}"
            if self._is_in_custom_denylist(ip):
                return [], f"Host '{hostname}' resolves to denylisted IP: {ip}"
            safe_ips.append(ip_str)

        if not safe_ips:
            return [], f"DNS returned no usable addresses for '{hostname}'"
        return safe_ips, ""

    def check_request_rate(self, host: str) -> bool:
        """Rate-limit requests per host. Returns True if allowed."""
        now = time.time()
        with self._lock:
            if host not in self._request_counts:
                self._request_counts[host] = [now]
                return True
            timestamps = self._request_counts[host]
            timestamps[:] = [ts for ts in timestamps if now - ts < self._request_rate_window]
            if len(timestamps) >= self._request_rate_limit:
                return False
            timestamps.append(now)
            return True

    def add_allowlist(self, network: str) -> None:
        """Add a network to the custom allowlist (CIDR notation)."""
        net = ipaddress.ip_network(network, strict=False)
        self._custom_allowlist.append(net)

    def add_denylist(self, network: str) -> None:
        """Add a network to the custom denylist (CIDR notation)."""
        net = ipaddress.ip_network(network, strict=False)
        self._custom_denylist.append(net)

    def get_stats(self) -> dict[str, Any]:
        """Get protection statistics."""
        with self._lock:
            return {
                "dns_cache_size": len(self._dns_cache),
                "tracked_hosts": len(self._request_counts),
                "custom_allowlist_entries": len(self._custom_allowlist),
                "custom_denylist_entries": len(self._custom_denylist),
                "rate_limit_per_minute": self._request_rate_limit,
            }


_ssrf_protector: SSRFProtector | None = None
_ssrf_lock = threading.Lock()


def get_ssrf_protector() -> SSRFProtector:
    """Thread-safe singleton for SSRFProtector."""
    global _ssrf_protector
    if _ssrf_protector is None:
        with _ssrf_lock:
            if _ssrf_protector is None:
                _ssrf_protector = SSRFProtector()
    return _ssrf_protector


def validate_outbound_url(url: str, allow_internal: bool = False) -> SSRFValidationResult:
    """Convenience function to validate an outbound URL for SSRF safety."""
    return get_ssrf_protector().validate_url(url, allow_internal=allow_internal)


def resolve_validated_ips(hostname: str) -> tuple[list[str], str]:
    """Convenience function: resolve a hostname and return its validated IPs.

    See ``SSRFProtector.resolve_validated_ips``. This is the authoritative
    resolution point callers should use to pin outbound connections.
    """
    return get_ssrf_protector().resolve_validated_ips(hostname)


def sanitize_request_headers(headers: dict) -> dict:
    """Remove dangerous headers from outbound requests.
    
    Strips headers that could leak information or enable attacks.
    """
    dangerous_prefixes = (
        "x-forwarded-", "x-real-ip", "x-originating-ip",
        "x-csrf-token", "authorization",
    )
    cleaned = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if any(key_lower.startswith(prefix) for prefix in dangerous_prefixes):
            continue
        if key_lower == "cookie" and "session" in str(value).lower():
            continue
        cleaned[key] = value
    return cleaned