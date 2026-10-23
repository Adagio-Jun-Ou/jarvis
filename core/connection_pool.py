"""Connection Pool Manager — Reusable connection pooling for LLM/API calls.

For a 24/7 agent, creating new HTTP connections for every LLM call is wasteful.
This module provides:
- Connection pooling with configurable pool size
- Health checking (detect dead connections)
- Automatic reconnection on failure
- Per-host pool isolation
- Connection lifecycle metrics
- Thread-safe borrowing/returning
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    host: str
    port: int
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_healthy: bool = True
    is_busy: bool = False
    session: Any = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "age_seconds": round(self.age_seconds, 1),
            "idle_seconds": round(self.idle_seconds, 1),
            "use_count": self.use_count,
            "is_healthy": self.is_healthy,
            "is_busy": self.is_busy,
            "has_session": self.session is not None,
        }


class ConnectionPool:
    """Thread-safe connection pool for LLM and API connections.

    Usage:
        pool = get_connection_pool()
        conn = pool.acquire("localhost", 11434)
        try:
            conn.session.post(url, json=data)
        finally:
            pool.release(conn)
    """

    DEFAULT_POOL_SIZE = 10
    MAX_POOL_SIZE = 50
    MAX_CONNECTION_AGE = 3600.0
    MAX_IDLE_TIME = 300.0
    HEALTH_CHECK_INTERVAL = 60.0

    def __init__(self, max_pool_size: int = DEFAULT_POOL_SIZE):
        self._max_pool_size = max(1, min(max_pool_size, self.MAX_POOL_SIZE))
        self._lock = threading.RLock()
        self._pools: dict[str, list[PooledConnection]] = {}
        self._total_acquired = 0
        self._total_released = 0
        self._total_created = 0
        self._total_destroyed = 0
        self._total_health_checks = 0
        self._total_reconnects = 0
        self._start_time = time.time()

    def _pool_key(self, host: str, port: int) -> str:
        return f"{host}:{port}"

    def _create_connection(self, host: str, port: int) -> PooledConnection:
        conn = PooledConnection(host=host, port=port)
        try:
            import requests
            conn.session = requests.Session()
            conn.session.headers.update({"Connection": "keep-alive"})
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1,
                pool_maxsize=self._max_pool_size,
                max_retries=3,
            )
            conn.session.mount(f"http://{host}", adapter)
            conn.session.mount(f"https://{host}", adapter)
        except ImportError:
            conn.session = None

        with self._lock:
            self._total_created += 1
        return conn

    def acquire(self, host: str, port: int, timeout: float = 10.0) -> PooledConnection:
        key = self._pool_key(host, port)
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                pool = self._pools.get(key, [])
                for conn in pool:
                    if not conn.is_busy and conn.is_healthy and conn.session is not None:
                        conn.is_busy = True
                        conn.last_used = time.time()
                        conn.use_count += 1
                        self._total_acquired += 1
                        return conn

                current_total = sum(len(p) for p in self._pools.values())
                if current_total < self._max_pool_size:
                    conn = self._create_connection(host, port)
                    conn.is_busy = True
                    conn.last_used = time.time()
                    conn.use_count += 1
                    if key not in self._pools:
                        self._pools[key] = []
                    self._pools[key].append(conn)
                    self._total_acquired += 1
                    return conn

            time.sleep(0.05)

        raise TimeoutError(f"Could not acquire connection to {key} within {timeout}s")

    def release(self, conn: PooledConnection) -> None:
        with self._lock:
            conn.is_busy = False
            conn.last_used = time.time()
            self._total_released += 1

            if conn.age_seconds > self.MAX_CONNECTION_AGE or not conn.is_healthy:
                self._destroy_connection(conn)

    def _destroy_connection(self, conn: PooledConnection) -> None:
        if conn.session:
            try:
                conn.session.close()
            except Exception as e:
                logger.debug(f"Connection close failed: {e}")
        conn.is_healthy = False
        conn.session = None

        key = self._pool_key(conn.host, conn.port)
        with self._lock:
            pool = self._pools.get(key, [])
            try:
                pool.remove(conn)
            except ValueError:
                pass
            self._total_destroyed += 1

    def check_health(self) -> int:
        destroyed = 0
        with self._lock:
            self._total_health_checks += 1
            for key, pool in list(self._pools.items()):
                stale = []
                for conn in pool:
                    if not conn.is_busy:
                        if conn.idle_seconds > self.MAX_IDLE_TIME:
                            stale.append(conn)
                        elif conn.session is None:
                            stale.append(conn)
                        elif not conn.is_healthy:
                            stale.append(conn)

                for conn in stale:
                    self._destroy_connection(conn)
                    destroyed += 1

                    new_conn = self._create_connection(conn.host, conn.port)
                    pool.append(new_conn)
                    self._total_reconnects += 1

        return destroyed

    def get_pool_status(self, host: str | None = None, port: int | None = None) -> dict[str, Any]:
        with self._lock:
            if host and port:
                key = self._pool_key(host, port)
                pool = self._pools.get(key, [])
                return {
                    "key": key,
                    "total": len(pool),
                    "busy": sum(1 for c in pool if c.is_busy),
                    "idle": sum(1 for c in pool if not c.is_busy),
                    "healthy": sum(1 for c in pool if c.is_healthy),
                    "connections": [c.to_dict() for c in pool],
                }

            result = {}
            total_connections = 0
            for key, pool in self._pools.items():
                result[key] = {
                    "total": len(pool),
                    "busy": sum(1 for c in pool if c.is_busy),
                    "idle": sum(1 for c in pool if not c.is_busy),
                    "healthy": sum(1 for c in pool if c.is_healthy),
                }
                total_connections += len(pool)

            return {
                "pools": result,
                "total_connections": total_connections,
                "max_pool_size": self._max_pool_size,
            }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            total_conns = sum(len(p) for p in self._pools.values())
            busy_conns = sum(1 for pool in self._pools.values() for c in pool if c.is_busy)
            return {
                "total_connections": total_conns,
                "busy_connections": busy_conns,
                "idle_connections": total_conns - busy_conns,
                "pools": len(self._pools),
                "max_pool_size": self._max_pool_size,
                "total_acquired": self._total_acquired,
                "total_released": self._total_released,
                "total_created": self._total_created,
                "total_destroyed": self._total_destroyed,
                "total_health_checks": self._total_health_checks,
                "total_reconnects": self._total_reconnects,
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }

    def shutdown(self) -> None:
        with self._lock:
            for pool in self._pools.values():
                for conn in pool:
                    if conn.session:
                        try:
                            conn.session.close()
                        except Exception as e:
                            logger.debug(f"Connection close during shutdown failed: {e}")
                    conn.session = None
                    conn.is_healthy = False
            self._pools.clear()


_connection_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_connection_pool(max_pool_size: int = ConnectionPool.DEFAULT_POOL_SIZE) -> ConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = ConnectionPool(max_pool_size=max_pool_size)
    return _connection_pool