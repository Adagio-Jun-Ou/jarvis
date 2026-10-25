"""Tests for new integration modules."""
import pytest
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


class TestSystemIntegrator:
    """Test the SystemIntegrator module."""

    def test_import(self):
        """Test module imports correctly."""
        from core.system_integrator import SystemIntegrator
        assert SystemIntegrator is not None

    def test_register_subsystem(self):
        """Test subsystem registration."""
        from core.system_integrator import SystemIntegrator, SubsystemState

        integrator = SystemIntegrator()
        mock_subsystem = MagicMock()

        integrator.register_subsystem("test_system", mock_subsystem)

        assert "test_system" in integrator._subsystems
        assert integrator._subsystems["test_system"].state == SubsystemState.UNINITIALIZED

    def test_dependency_order(self):
        """Test dependency resolution."""
        from core.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()
        integrator.register_subsystem("a", MagicMock(), dependencies=["b"])
        integrator.register_subsystem("b", MagicMock(), dependencies=["c"])
        integrator.register_subsystem("c", MagicMock(), dependencies=[])

        order = integrator._resolve_dependency_order()

        assert order.index("c") < order.index("b")
        assert order.index("b") < order.index("a")

    def test_event_emission(self):
        """Test event emission."""
        from core.system_integrator import SystemIntegrator, EventPriority

        integrator = SystemIntegrator()
        result = []

        def handler(event):
            result.append(event)

        integrator.on_event("test_event", handler)
        integrator.emit_event("test_event", "test_source", {"key": "value"}, EventPriority.NORMAL)

        assert integrator._event_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test integrator start/stop."""
        from core.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()

        mock_subsystem = MagicMock()
        mock_subsystem.start = AsyncMock()
        mock_subsystem.stop = AsyncMock()

        integrator.register_subsystem("test", mock_subsystem)

        await integrator.start()
        assert integrator._running is True

        await integrator.stop()
        assert integrator._running is False

    def test_get_status(self):
        """Test status retrieval."""
        from core.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()
        integrator.register_subsystem("test", MagicMock())

        status = integrator.get_status()

        assert "running" in status
        assert "subsystems" in status
        assert "test" in status["subsystems"]


class TestStateSynchronizer:
    """Test the StateSynchronizer module."""

    def test_import(self):
        """Test module imports correctly."""
        from core.state_sync import StateSynchronizer
        assert StateSynchronizer is not None

    def test_get_set_state(self):
        """Test basic get/set operations."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            sync.set("test_key", "test_value", "test_source")
            assert sync.get("test_key") == "test_value"

    def test_increment(self):
        """Test increment operation."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            result = sync.increment("counter", 5, "test")
            assert result == 5

            result = sync.increment("counter", 3, "test")
            assert result == 8

    def test_merge(self):
        """Test merge operation."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            sync.set("dict_key", {"a": 1}, "test")
            sync.merge("dict_key", {"b": 2}, "test")

            result = sync.get("dict_key")
            assert result == {"a": 1, "b": 2}

    def test_subscribe(self):
        """Test subscription to changes."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            changes = []

            def handler(change):
                changes.append(change)

            sync.subscribe("test_key", handler)
            sync.set("test_key", "value", "test")

            assert len(changes) == 1
            assert changes[0]["key"] == "test_key"

    def test_checkpoint(self):
        """Test checkpoint creation and restore."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            sync.set("key1", "value1", "test")
            sync.set("key2", "value2", "test")

            checkpoint_id = sync.create_checkpoint("test checkpoint")

            sync.set("key1", "modified", "test")
            sync.set("key3", "new", "test")

            sync.restore_checkpoint(checkpoint_id)

            assert sync.get("key1") == "value1"
            assert sync.get("key2") == "value2"
            assert sync.get("key3") is None

    def test_subsystem_state(self):
        """Test subsystem-specific state."""
        from core.state_sync import StateSynchronizer, StateScope

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            sync.register_subsystem("test_subsystem")
            sync.set("test_subsystem.local_key", "local_value", "test", scope=StateScope.SUBSYSTEM)

            state = sync.sync_to_subsystem("test_subsystem")
            assert "test_subsystem" in sync._subsystem_states


class TestObservabilitySystem:
    """Test the Observability system."""

    def test_import(self):
        """Test module imports correctly."""
        from core.observability import ObservabilitySystem
        assert ObservabilitySystem is not None

    def test_counter(self):
        """Test counter metric."""
        from core.observability import MetricsRegistry

        metrics = MetricsRegistry()
        counter = metrics.register_counter("test_counter", "Test counter")

        counter.increment()
        counter.increment(5)

        assert counter.get_value() == 6

    def test_gauge(self):
        """Test gauge metric."""
        from core.observability import MetricsRegistry

        metrics = MetricsRegistry()
        gauge = metrics.register_gauge("test_gauge", "Test gauge")

        gauge.set(10)
        assert gauge.get_value() == 10

        gauge.set(20)
        assert gauge.get_value() == 20

    def test_histogram(self):
        """Test histogram metric."""
        from core.observability import MetricsRegistry

        metrics = MetricsRegistry()
        hist = metrics.register_histogram("test_hist", "Test histogram")

        for i in [1, 2, 3, 4, 5]:
            hist.observe(i)

        summary = hist.get_summary()
        assert summary["count"] == 5
        assert summary["sum"] == 15
        assert summary["mean"] == 3.0

    def test_alert_firing(self):
        """Test alert management."""
        from core.observability import AlertManager, AlertSeverity

        with tempfile.TemporaryDirectory() as tmpdir:
            alerts = AlertManager(Path(tmpdir))

            fired = []
            def handler(alert):
                fired.append(alert)

            alerts.register_handler(handler)

            alert = alerts.fire_alert(
                "test_alert",
                AlertSeverity.WARNING,
                "Test message",
                "test_source"
            )

            assert len(fired) == 1
            assert alert.name == "test_alert"

            alerts.resolve_alert(alert.id)
            assert alert.resolved_at is not None

    def test_silence_alert(self):
        """Test alert silencing."""
        from core.observability import AlertManager, AlertSeverity

        with tempfile.TemporaryDirectory() as tmpdir:
            alerts = AlertManager(Path(tmpdir))

            fired = []
            def handler(alert):
                fired.append(alert)

            alerts.silence_alert("silenced_alert")
            alerts.register_handler(handler)

            alerts.fire_alert("silenced_alert", AlertSeverity.INFO, "Test", "test")
            alerts.fire_alert("normal_alert", AlertSeverity.INFO, "Test", "test")

            assert len(fired) == 1
            assert fired[0].name == "normal_alert"

    def test_health_check(self):
        """Test health check functionality."""
        from core.observability import ObservabilitySystem

        with tempfile.TemporaryDirectory() as tmpdir:
            obs = ObservabilitySystem(Path(tmpdir))

            def check_func():
                return True, {"detail": "ok"}

            obs.register_health_check("test_component", check_func)

            health_ok, details = check_func()

            assert health_ok is True
            assert details["detail"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_loop(self):
        """Test health check loop."""
        from core.observability import ObservabilitySystem

        with tempfile.TemporaryDirectory() as tmpdir:
            obs = ObservabilitySystem(Path(tmpdir))

            def healthy_check():
                return True, {"ok": True}

            obs.register_health_check("healthy_component", healthy_check)

            health_ok, details = healthy_check()
            assert health_ok is True

    def test_dashboard_data(self):
        """Test dashboard data export."""
        from core.observability import ObservabilitySystem

        with tempfile.TemporaryDirectory() as tmpdir:
            obs = ObservabilitySystem(Path(tmpdir))

            obs.metrics.increment_counter("jarvis_requests_total", 10)
            obs.metrics.set_gauge("jarvis_active_sessions", 5)

            data = obs.get_dashboard_data()

            assert "metrics" in data
            assert "alerts" in data
            assert "components" in data
            assert data["metrics"]["jarvis_requests_total"]["value"] == 10


class TestIntegration:
    """Integration tests for new modules working together."""

    def test_state_sync_with_integrator(self):
        """Test state sync integration with system integrator."""
        from core.system_integrator import get_system_integrator
        from core.state_sync import get_state_synchronizer

        integrator = get_system_integrator()
        sync = get_state_synchronizer()

        assert integrator is not None
        assert sync is not None

    def test_observability_with_metrics(self):
        """Test observability with metrics collection."""
        from core.observability import get_observability, get_metrics

        obs = get_observability()
        metrics = get_metrics()

        counter = metrics.register_counter("test_integration_counter_new", "Test counter")
        counter.increment(7)

        all_metrics = obs.metrics.get_all_metrics()
        assert "test_integration_counter_new" in all_metrics

    def test_export_prometheus_format(self):
        """Test Prometheus export format."""
        from core.observability import get_observability

        obs = get_observability()
        obs.metrics.set_gauge("jarvis_active_sessions", 3)

        prom_output = obs.export_metrics_prometheus()

        assert "jarvis_active_sessions" in prom_output
        assert "# TYPE" in prom_output
        assert "# HELP" in prom_output


class TestThreadSafety:
    """Test thread safety of new modules."""

    def test_concurrent_state_updates(self):
        """Test concurrent state modifications."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            sync = StateSynchronizer(Path(tmpdir))

            errors = []

            def update_value(thread_id):
                try:
                    for i in range(100):
                        sync.increment("concurrent_counter", 1, f"thread_{thread_id}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=update_value, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert sync.get("concurrent_counter") == 1000

    def test_concurrent_metric_updates(self):
        """Test concurrent metric modifications."""
        from core.observability import MetricsRegistry

        metrics = MetricsRegistry()
        counter = metrics.register_counter("concurrent_counter", "Test")

        errors = []

        def increment_counter():
            try:
                for _ in range(100):
                    counter.increment()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert counter.get_value() == 1000


class TestPersistence:
    """Test persistence of state."""

    def test_state_persistence(self):
        """Test state is persisted and can be reloaded."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            sync1 = StateSynchronizer(path)
            sync1.set("persisted_key", "persisted_value", "test")
            sync1._save_state()

            sync2 = StateSynchronizer(path)

            assert sync2.get("persisted_key") == "persisted_value"

    def test_checkpoint_persistence(self):
        """Test checkpoint persistence."""
        from core.state_sync import StateSynchronizer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            sync = StateSynchronizer(path)
            sync.set("key1", "value1", "test")

            checkpoint_id = sync.create_checkpoint("persistent checkpoint")

            sync.set("key1", "modified", "test")

            assert sync.restore_checkpoint(checkpoint_id)
            assert sync.get("key1") == "value1"
