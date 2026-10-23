"""Tests for Connection Pool Manager.

Covers:
- Connection acquisition and release
- Pool size limits
- Connection reuse
- Health checking and reconnection
- Connection age/idle limits
- Thread safety
- Pool shutdown
- Metrics
"""
import time
import threading
from pathlib import Path

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from core.connection_pool import ConnectionPool, PooledConnection, get_connection_pool


class TestPooledConnection:
    def test_connection_defaults(self):
        conn = PooledConnection(host="localhost", port=8080)
        assert conn.host == "localhost"
        assert conn.port == 8080
        assert conn.is_healthy is True
        assert conn.is_busy is False
        assert conn.use_count == 0

    def test_connection_age(self):
        conn = PooledConnection(host="localhost", port=8080)
        conn.created_at = time.time() - 100
        assert conn.age_seconds >= 99

    def test_connection_idle(self):
        conn = PooledConnection(host="localhost", port=8080)
        conn.last_used = time.time() - 50
        assert conn.idle_seconds >= 49

    def test_connection_to_dict(self):
        conn = PooledConnection(host="localhost", port=8080, use_count=5)
        d = conn.to_dict()
        assert d["host"] == "localhost"
        assert d["port"] == 8080
        assert d["use_count"] == 5


class TestConnectionPool:
    def setup_method(self):
        self.pool = ConnectionPool(max_pool_size=5)

    def test_acquire_creates_connection(self):
        conn = self.pool.acquire("localhost", 8080)
        assert conn is not None
        assert conn.host == "localhost"
        assert conn.port == 8080
        assert conn.is_busy is True

    def test_release_marks_idle(self):
        conn = self.pool.acquire("localhost", 8080)
        self.pool.release(conn)
        assert conn.is_busy is False

    def test_connection_reuse(self):
        conn1 = self.pool.acquire("localhost", 8080)
        self.pool.release(conn1)
        conn2 = self.pool.acquire("localhost", 8080)
        assert conn1 is conn2
        assert conn2.use_count == 2
        self.pool.release(conn2)

    def test_multiple_connections(self):
        conn1 = self.pool.acquire("localhost", 8080)
        conn2 = self.pool.acquire("localhost", 8080)
        assert conn1 is not conn2
        self.pool.release(conn1)
        self.pool.release(conn2)

    def test_acquired_count(self):
        self.pool.acquire("localhost", 8080)
        status = self.pool.get_status()
        assert status["total_acquired"] == 1

    def test_released_count(self):
        conn = self.pool.acquire("localhost", 8080)
        self.pool.release(conn)
        status = self.pool.get_status()
        assert status["total_released"] == 1

    def test_created_count(self):
        self.pool.acquire("localhost", 8080)
        status = self.pool.get_status()
        assert status["total_created"] == 1

    def test_pool_status_fields(self):
        status = self.pool.get_status()
        assert "total_connections" in status
        assert "busy_connections" in status
        assert "idle_connections" in status
        assert "pools" in status
        assert "max_pool_size" in status
        assert "total_acquired" in status
        assert "total_released" in status
        assert "uptime_seconds" in status

    def test_get_pool_status_specific(self):
        conn = self.pool.acquire("localhost", 8080)
        self.pool.release(conn)
        status = self.pool.get_pool_status(host="localhost", port=8080)
        assert status["key"] == "localhost:8080"
        assert status["total"] >= 1

    def test_get_pool_status_all(self):
        self.pool.acquire("host1", 8080)
        self.pool.acquire("host2", 9090)
        status = self.pool.get_pool_status()
        assert "pools" in status
        assert status["total_connections"] >= 2


class TestConnectionPoolLimits:
    def test_max_pool_size(self):
        pool = ConnectionPool(max_pool_size=2)
        conn1 = pool.acquire("host1", 80)
        conn2 = pool.acquire("host1", 80)
        with pytest.raises(TimeoutError):
            pool.acquire("host1", 80, timeout=0.5)
        pool.release(conn1)
        pool.release(conn2)

    def test_max_pool_size_clamped(self):
        pool = ConnectionPool(max_pool_size=1000)
        assert pool._max_pool_size == pool.MAX_POOL_SIZE

    def test_min_pool_size(self):
        pool = ConnectionPool(max_pool_size=0)
        assert pool._max_pool_size == 1


class TestConnectionPoolHealth:
    def setup_method(self):
        self.pool = ConnectionPool(max_pool_size=5)

    def test_health_check_returns_count(self):
        conn = self.pool.acquire("localhost", 8080)
        self.pool.release(conn)
        count = self.pool.check_health()
        assert isinstance(count, int)

    def test_healthy_connection_survives(self):
        conn = self.pool.acquire("localhost", 8080)
        self.pool.release(conn)
        self.pool.check_health()
        status = self.pool.get_pool_status(host="localhost", port=8080)
        assert status["healthy"] >= 1

    def test_health_check_count(self):
        self.pool.check_health()
        status = self.pool.get_status()
        assert status["total_health_checks"] >= 1


class TestConnectionPoolShutdown:
    def test_shutdown_clears(self):
        pool = ConnectionPool(max_pool_size=5)
        conn = pool.acquire("localhost", 8080)
        pool.release(conn)
        pool.shutdown()
        status = pool.get_status()
        assert status["total_connections"] == 0

    def test_destroyed_count_after_shutdown(self):
        pool = ConnectionPool(max_pool_size=5)
        conn = pool.acquire("localhost", 8080)
        pool.release(conn)
        pool._destroy_connection(conn)
        status = pool.get_status()
        assert status["total_destroyed"] >= 1


class TestConnectionPoolThreadSafety:
    def test_concurrent_acquire_release(self):
        pool = ConnectionPool(max_pool_size=20)
        errors = []
        acquired_count = []

        def acquire_release(i):
            try:
                conn = pool.acquire("localhost", 8080, timeout=5)
                acquired_count.append(1)
                time.sleep(0.01)
                pool.release(conn)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=acquire_release, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(acquired_count) == 10

    def test_singleton(self):
        p1 = get_connection_pool()
        p2 = get_connection_pool()
        assert p1 is p2