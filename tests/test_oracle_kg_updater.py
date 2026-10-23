"""Tests for Oracle Knowledge Graph Auto-Updater."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.oracle_kg_updater import (
    OracleKGUpdater,
    get_oracle_kg_updater,
    oracle_kg_update_task,
    register_oracle_kg_updater,
    DEFAULT_DOMAINS,
    DEFAULT_INTERVAL_MINUTES,
)


class TestOracleKGUpdaterInit:
    def test_default_initialization(self):
        updater = OracleKGUpdater()
        assert updater.domains == DEFAULT_DOMAINS
        assert updater.interval_minutes == DEFAULT_INTERVAL_MINUTES
        assert updater._total_cycles == 0
        assert updater._total_insights_stored == 0

    def test_custom_domains(self):
        updater = OracleKGUpdater(domains=["economic", "climate"], interval_minutes=15)
        assert updater.domains == ["economic", "climate"]
        assert updater.interval_minutes == 15

    def test_status_initial(self):
        updater = OracleKGUpdater()
        status = updater.get_status()
        assert status["running"] is False
        assert status["total_cycles"] == 0
        assert status["last_run"] is None
        assert status["seen_keys_count"] == 0


class TestOracleKGUpdaterDomainManagement:
    def test_add_domain(self):
        updater = OracleKGUpdater()
        updater.add_domain("technology")
        assert "technology" in updater.domains

    def test_add_duplicate_domain(self):
        updater = OracleKGUpdater()
        updater.add_domain("economic")
        count = updater.domains.count("economic")
        assert count == 1

    def test_remove_domain(self):
        updater = OracleKGUpdater()
        updater.remove_domain("economic")
        assert "economic" not in updater.domains

    def test_remove_nonexistent_domain(self):
        updater = OracleKGUpdater()
        original_count = len(updater.domains)
        updater.remove_domain("nonexistent")
        assert len(updater.domains) == original_count


class TestOracleKGUpdaterCycle:
    @pytest.mark.asyncio
    async def test_cycle_without_oracle(self):
        updater = OracleKGUpdater()
        with patch.dict("sys.modules", {"core.oracle_integration": None}):
            result = await updater.run_cycle()
            assert isinstance(result, dict)
            assert "cycle" in result

    @pytest.mark.asyncio
    async def test_cycle_updates_status(self):
        updater = OracleKGUpdater()
        with patch("core.oracle_integration.oracle_predict_to_knowledge_graph", return_value={"kg_integrated": False}):
            with patch("core.oracle_integration.oracle_diagnose_to_knowledge_graph", return_value={"kg_integrated": False}):
                await updater.run_cycle()
                status = updater.get_status()
                assert status["total_cycles"] == 1
                assert status["last_run"] is not None

    @pytest.mark.asyncio
    async def test_cycle_prevents_concurrent(self):
        updater = OracleKGUpdater()
        updater._running = True
        result = await updater.run_cycle()
        assert result.get("skipped") is True

    @pytest.mark.asyncio
    async def test_cycle_with_kg_integration(self):
        updater = OracleKGUpdater()
        mock_result = {"kg_integrated": True, "kg_nodes_created": 3}
        with patch("core.oracle_integration.oracle_predict_to_knowledge_graph", return_value=mock_result):
            with patch("core.oracle_integration.oracle_diagnose_to_knowledge_graph", return_value={"kg_integrated": True, "kg_edges_created": 2}):
                result = await updater.run_cycle()
                assert result["insights_stored"] >= 0
                assert "diagnose_edges" in result

    @pytest.mark.asyncio
    async def test_cycle_handles_domain_error(self):
        updater = OracleKGUpdater()
        with patch("core.oracle_integration.oracle_predict_to_knowledge_graph", side_effect=Exception("Oracle error")):
            with patch("core.oracle_integration.oracle_diagnose_to_knowledge_graph", return_value={"kg_integrated": False}):
                result = await updater.run_cycle()
                assert result["errors"] > 0

    @pytest.mark.asyncio
    async def test_multiple_cycles(self):
        updater = OracleKGUpdater()
        with patch("core.oracle_integration.oracle_predict_to_knowledge_graph", return_value={"kg_integrated": True, "kg_nodes_created": 1}):
            with patch("core.oracle_integration.oracle_diagnose_to_knowledge_graph", return_value={"kg_integrated": False}):
                await updater.run_cycle()
                await updater.run_cycle()
                status = updater.get_status()
                assert status["total_cycles"] == 2

    @pytest.mark.asyncio
    async def test_cycle_records_history(self):
        updater = OracleKGUpdater()
        with patch("core.oracle_integration.oracle_predict_to_knowledge_graph", return_value={"kg_integrated": False}):
            with patch("core.oracle_integration.oracle_diagnose_to_knowledge_graph", return_value={"kg_integrated": False}):
                await updater.run_cycle()
                history = updater.get_history()
                assert len(history) == 1
                assert "timestamp" in history[0]


class TestOracleKGUpdaterHistory:
    def test_history_bounded(self):
        updater = OracleKGUpdater()
        for i in range(600):
            updater._history.append({"timestamp": time.time(), "cycle": i})
        assert len(updater._history) <= 500

    def test_get_history_limit(self):
        updater = OracleKGUpdater()
        for i in range(30):
            updater._history.append({"timestamp": time.time(), "cycle": i})
        history = updater.get_history(limit=10)
        assert len(history) == 10


class TestOracleKGUpdaterSeenKeys:
    def test_prune_seen_keys(self):
        updater = OracleKGUpdater()
        updater._seen_insight_keys = {f"key_{i}" for i in range(3000)}
        assert len(updater._seen_insight_keys) == 3000
        updater.prune_seen_keys()
        assert len(updater._seen_insight_keys) <= 1000

    def test_prune_not_needed(self):
        updater = OracleKGUpdater()
        updater._seen_insight_keys = {f"key_{i}" for i in range(100)}
        original_len = len(updater._seen_insight_keys)
        updater.prune_seen_keys()
        assert len(updater._seen_insight_keys) == original_len


class TestGetOracleKGUpdater:
    def setup_method(self):
        import core.oracle_kg_updater as mod
        mod._oracle_updater = None

    def test_singleton(self):
        u1 = get_oracle_kg_updater()
        u2 = get_oracle_kg_updater()
        assert u1 is u2

    def test_custom_params(self):
        updater = get_oracle_kg_updater(domains=["climate"], interval_minutes=10)
        assert updater.domains == ["climate"]
        assert updater.interval_minutes == 10


class TestOracleKGUpdateTask:
    @pytest.mark.asyncio
    async def test_update_task_calls_cycle(self):
        import core.oracle_kg_updater as mod
        mod._oracle_updater = None
        with patch("core.oracle_integration.oracle_predict_to_knowledge_graph", return_value={"kg_integrated": False}):
            with patch("core.oracle_integration.oracle_diagnose_to_knowledge_graph", return_value={"kg_integrated": False}):
                result = await oracle_kg_update_task()
                assert isinstance(result, dict)


class TestRegisterOracleKGUpdater:
    def test_register_with_scheduler(self):
        mock_agent = MagicMock()
        mock_scheduler = MagicMock()
        mock_agent.scheduler = mock_scheduler

        import core.oracle_kg_updater as mod
        mod._oracle_updater = None

        result = register_oracle_kg_updater(mock_agent)
        assert result is True
        mock_scheduler.register.assert_called_once()

    def test_register_without_scheduler(self):
        mock_agent = MagicMock()
        mock_agent.scheduler = None
        result = register_oracle_kg_updater(mock_agent)
        assert result is False

    def test_register_verifies_task_name(self):
        mock_agent = MagicMock()
        mock_scheduler = MagicMock()
        mock_agent.scheduler = mock_scheduler

        import core.oracle_kg_updater as mod
        mod._oracle_updater = None

        register_oracle_kg_updater(mock_agent)
        call_kwargs = mock_scheduler.register.call_args
        assert call_kwargs[1].get("name") == "oracle_kg_update" or call_kwargs[0][0] == "oracle_kg_update"