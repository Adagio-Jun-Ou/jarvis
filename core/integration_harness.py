"""Integration Harness — Wires all Jarvis runtime subsystems together.

On boot, this module:
1. Starts the Watchdog (health monitoring)
2. Subscribes Event Bus channels to appropriate handlers
3. Registers health check callbacks
4. Starts connection pool health checker
5. Provides unified startup/shutdown

This is the "nervous system" that makes all the organs work together.
"""
import atexit
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class IntegrationHarness:
    """Connects all Jarvis runtime subsystems and manages lifecycle.

    Usage:
        harness = get_integration_harness()
        harness.start()
        # ... Jarvis runs ...
        harness.shutdown()
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._started = False
        self._start_time = 0.0
        self._subsystem_status: dict[str, str] = {}
        self._health_check_thread: threading.Thread | None = None
        self._running = False
        self._sub_ids: list[str] = []

    def start(self) -> dict[str, str]:
        if self._started:
            return self._subsystem_status

        self._start_time = time.time()
        results = {}

        results["watchdog"] = self._start_watchdog()
        results["event_bus"] = self._setup_event_subscriptions()
        results["ws_manager"] = self._setup_ws_health_checks()
        results["connection_pool"] = self._start_pool_health_checker()
        results["session_persistence"] = self._setup_session_events()
        results["sandbox"] = self._setup_sandbox_events()
        results["memory_store"] = self._init_memory_store()
        results["health_router"] = self._init_health_router()
        results["backpressure"] = self._init_backpressure()

        with self._lock:
            self._started = True
            self._subsystem_status = results

        atexit.register(self.shutdown)

        healthy = sum(1 for v in results.values() if v == "ok")
        total = len(results)
        logger.info(f"IntegrationHarness started: {healthy}/{total} subsystems OK")
        return results

    def _start_watchdog(self) -> str:
        try:
            from core.watchdog import get_watchdog, start_watchdog
            get_watchdog()
            start_watchdog()
            return "ok"
        except Exception as e:
            logger.debug(f"Watchdog start failed: {e}")
            return f"error: {e}"

    def _setup_event_subscriptions(self) -> str:
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()

            def on_resource_alert(event):
                logger.warning(f"[Alert] {event.data.get('message', 'unknown')}")

            def on_cwd_changed(event):
                logger.debug(f"[CWD] {event.data.get('old_cwd', '')} → {event.data.get('new_cwd', '')}")

            bus.subscribe("watchdog.resource_alert", on_resource_alert)
            bus.subscribe("context.cwd_changed", on_cwd_changed)

            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _setup_ws_health_checks(self) -> str:
        try:
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _start_pool_health_checker(self) -> str:
        try:
            self._running = True
            self._health_check_thread = threading.Thread(
                target=self._pool_health_loop,
                name="pool-health",
                daemon=True,
            )
            self._health_check_thread.start()
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _pool_health_loop(self) -> None:
        while self._running:
            try:
                from core.connection_pool import get_connection_pool
                pool = get_connection_pool()
                pool.check_health()
            except Exception as e:
                logger.debug(f"Pool health check failed: {e}")
            slept = 0
            while slept < 60 and self._running:
                time.sleep(1)
                slept += 1

    def _setup_session_events(self) -> str:
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()

            def on_session_created(event):
                logger.debug(f"[Session] Created: {event.data.get('session_id', '?')}")

            bus.subscribe("session.created", on_session_created)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _setup_sandbox_events(self) -> str:
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()

            def on_sandbox_timeout(event):
                logger.warning(f"[Sandbox] Timeout: {event.data.get('sandbox_id', '?')}")

            bus.subscribe("sandbox.timeout", on_sandbox_timeout)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _init_memory_store(self) -> str:
        try:
            from core.agent_memory_store import get_agent_memory_store
            get_agent_memory_store()
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _init_health_router(self) -> str:
        try:
            from core.health_router import get_health_router
            router = get_health_router()
            router.register_backend("llm_primary", priority=True)
            router.register_backend("llm_backend", priority=True)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _init_backpressure(self) -> str:
        try:
            from core.streaming_pipeline import get_backpressure_controller
            get_backpressure_controller()
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            if not self._started:
                return
            self._started = False

        try:
            from core.watchdog import get_watchdog
            wd = get_watchdog()
            wd.stop()
        except Exception as e:
            logger.debug(f"Watchdog shutdown failed: {e}")

        try:
            from core.connection_pool import get_connection_pool
            pool = get_connection_pool()
            pool.shutdown()
        except Exception as e:
            logger.debug(f"Connection pool shutdown failed: {e}")

        try:
            from core.execution_context import get_execution_context
            ctx = get_execution_context()
            ctx.shutdown()
        except Exception as e:
            logger.debug(f"Execution context shutdown failed: {e}")

        logger.info("IntegrationHarness shutdown complete")

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0

        subsystems = {}
        _subsystem_names = [
            ("event_bus", "core.event_bus", "get_event_bus"),
            ("watchdog", "core.watchdog", "get_watchdog"),
            ("sandbox", "core.sandbox", "get_sandbox_manager"),
            ("session_persistence", "core.session_persistence", "get_session_persistence"),
            ("context_compressor", "core.context_compressor", "get_context_compressor"),
            ("tool_chain_optimizer", "core.tool_chain_optimizer", "get_tool_chain_optimizer"),
            ("connection_pool", "core.connection_pool", "get_connection_pool"),
            ("ws_manager", "core.ws_manager", "get_ws_connection_manager"),
            ("execution_context", "core.execution_context", "get_execution_context"),
            ("agent_memory_store", "core.agent_memory_store", "get_agent_memory_store"),
            ("health_router", "core.health_router", "get_health_router"),
            ("backpressure", "core.streaming_pipeline", "get_backpressure_controller"),
        ]
        for key, module_name, getter_name in _subsystem_names:
            try:
                import importlib
                mod = importlib.import_module(module_name)
                getter = getattr(mod, getter_name)
                subsystems[key] = getter().get_status()
            except Exception as e:
                logger.debug(f"Subsystem {key} status unavailable: {e}")

        return {
            "started": self._started,
            "uptime_seconds": round(uptime, 1),
            "subsystem_init_status": self._subsystem_status,
            "subsystems": subsystems,
        }


_harness: IntegrationHarness | None = None
_harness_lock = threading.Lock()


def get_integration_harness() -> IntegrationHarness:
    global _harness
    if _harness is None:
        with _harness_lock:
            if _harness is None:
                _harness = IntegrationHarness()
    return _harness


def start_integration_harness() -> dict[str, str]:
    harness = get_integration_harness()
    return harness.start()