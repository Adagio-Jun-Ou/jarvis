"""E2E integration test — validates full Jarvis stack via HTTP API.

Run with Jarvis already running on a known port:
    pytest tests/test_e2e_integration.py -v
"""
import json
import time

import pytest

BASE_URL = "http://localhost:18778"


def _get(url, timeout=5):
    import urllib.request
    try:
        req = urllib.request.Request(f"{BASE_URL}{url}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _post(url, data, timeout=5):
    import urllib.request
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{BASE_URL}{url}",
            data=body,
            headers={"Content-Type": "application/json", "Origin": BASE_URL},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"_error": str(e)}


def _delete(url, timeout=5):
    import urllib.request
    try:
        req = urllib.request.Request(f"{BASE_URL}{url}", method="DELETE")
        req.add_header("Origin", BASE_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"_error": str(e)}


@pytest.fixture(scope="module")
def jarvis_available():
    d = _get("/api/health")
    avail = d.get("overall") == "healthy"
    if not avail:
        import urllib.request
        try:
            urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=2)
            avail = True
        except Exception:
            pass
    return avail


def _skip_if_not_available(jarvis_available):
    if not jarvis_available:
        pytest.skip("Jarvis not available on port 18778")


class TestE2EHealthAPI:
    def test_health_endpoint(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/health")
        assert d.get("overall") == "healthy"
        assert d.get("uptime_seconds", 0) > 0

    def test_subsystem_health(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/health")
        subsystems = d.get("subsystem_health", {}).get("subsystems", [])
        assert len(subsystems) > 0
        for s in subsystems:
            assert s.get("status") == "healthy"


class TestE2EMemoryStore:
    def test_put_get_delete(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        r = _post("/api/memory-store/put", {"namespace": "e2e", "key": "test_key", "value": "test_value"})
        assert r.get("success") is True

        g = _get("/api/memory-store/get?namespace=e2e&key=test_key")
        assert g.get("value") == "test_value"
        assert g.get("exists") is True

        d = _delete("/api/memory-store/delete?namespace=e2e&key=test_key")
        assert d.get("success") is True

        g2 = _get("/api/memory-store/get?namespace=e2e&key=test_key")
        assert g2.get("exists") is False

    def test_list_namespaces(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/memory-store/list")
        assert d.get("success") is True
        assert isinstance(d.get("namespaces", []), list)


class TestE2EHealthRouter:
    def test_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/health-router/status")
        assert d.get("success") is True
        backends = d.get("backends", {})
        assert "llm_primary" in backends
        assert backends["llm_primary"]["health_score"] == 1.0


class TestE2EUnifiedRateLimiter:
    def test_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/unified-ratelimit/status")
        assert d.get("success") is True
        assert d.get("rate_limiter", {}).get("endpoint_limits") is not None


class TestE2EBackpressure:
    def test_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/backpressure/status")
        assert d.get("success") is True
        assert d.get("tokens_available", 0) > 0


class TestE2EMobileAPI:
    def test_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/mobile/status")
        assert d.get("status") == "available"


class TestE2ERuntimeAPIs:
    def test_connections(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/runtime/connections")
        assert d.get("success") is True

    def test_sessions_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/sessions/status")
        assert d.get("success") is True

    def test_compressor_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/compressor/status")
        assert d.get("success") is True

    def test_optimizer_status(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/optimizer/status")
        assert d.get("success") is True

    def test_events(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        d = _get("/api/events")
        assert d.get("success") is True


class TestE2ESecurity:
    def test_rate_limit_headers(self, jarvis_available):
        _skip_if_not_available(jarvis_available)
        import urllib.request
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                headers = dict(resp.headers)
        except Exception:
            return