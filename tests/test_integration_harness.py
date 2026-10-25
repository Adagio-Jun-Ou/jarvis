"""Tests for Integration Harness.

Covers:
- Harness startup lifecycle
- Subsystem initialization
- Event bus wiring
- Health check thread
- Shutdown
- Status reporting
"""
import time
import threading
from pathlib import Path

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys_path:
    sys.path.insert(0, sys_path)

from core.integration_harness import IntegrationHarness, get_integration_harness


class TestIntegrationHarnessStartup:
    def setup_method(self):
        self.harness = IntegrationHarness()

    def test_start_returns_results(self):
        results = self.harness.start()
        assert isinstance(results, dict)
        assert "watchdog" in results
        assert "event_bus" in results

    def test_start_idempotent(self):
        results1 = self.harness.start()
        results2 = self.harness.start()
        assert results1 is results2

    def test_subsystems_marked_ok(self):
        results = self.harness.start()
        ok_count = sum(1 for v in results.values() if v == "ok")
        assert ok_count >= 3

    def test_started_flag(self):
        self.harness.start()
        assert self.harness._started is True


class TestIntegrationHarnessStatus:
    def setup_method(self):
        self.harness = IntegrationHarness()

    def test_status_before_start(self):
        status = self.harness.get_status()
        assert status["started"] is False

    def test_status_after_start(self):
        self.harness.start()
        status = self.harness.get_status()
        assert status["started"] is True
        assert "uptime_seconds" in status

    def test_status_has_subsystems(self):
        self.harness.start()
        status = self.harness.get_status()
        assert "subsystems" in status
        assert isinstance(status["subsystems"], dict)

    def test_status_has_init_status(self):
        self.harness.start()
        status = self.harness.get_status()
        assert "subsystem_init_status" in status
        assert isinstance(status["subsystem_init_status"], dict)


class TestIntegrationHarnessShutdown:
    def setup_method(self):
        self.harness = IntegrationHarness()

    def test_shutdown_after_start(self):
        self.harness.start()
        self.harness.shutdown()
        assert self.harness._started is False

    def test_shutdown_idempotent(self):
        self.harness.start()
        self.harness.shutdown()
        self.harness.shutdown()

    def test_shutdown_without_start(self):
        self.harness.shutdown()


class TestIntegrationHarnessEvents:
    def setup_method(self):
        self.harness = IntegrationHarness()

    def test_event_bus_receives_watchdog_events(self):
        self.harness.start()
        from core.event_bus import get_event_bus
        from core.watchdog import get_watchdog
        received = []

        bus = get_event_bus()
        bus.subscribe("watchdog.resource_alert", lambda e: received.append(e))

        wd = get_watchdog()
        wd._emit("resource_alert", "Test alert", {"test": True})

        history = bus.get_history(channel="watchdog.resource_alert")
        assert len(history) >= 1

    def test_cwd_change_published_to_bus(self):
        self.harness.start()
        import tempfile
        from pathlib import Path
        from core.execution_context import ExecutionContext
        from core.event_bus import get_event_bus

        bus = get_event_bus()
        received = []
        bus.subscribe("context.cwd_changed", lambda e: received.append(e))

        tmpdir = tempfile.mkdtemp()
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True
        ctx.cd(tmpdir)

        history = bus.get_history(channel="context.cwd_changed")
        assert len(history) >= 1


class TestIntegrationHarnessSingleton:
    def test_singleton(self):
        h1 = get_integration_harness()
        h2 = get_integration_harness()
        assert h1 is h2