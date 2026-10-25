"""Tests for Oracle Integration Bridge."""

import pytest
import threading
from core.oracle_bridge import (
    OracleBridge,
    OracleForecast,
    BridgeMetrics,
    get_oracle_bridge,
    reset_oracle_bridge,
)


@pytest.fixture(autouse=True)
def cleanup():
    reset_oracle_bridge()
    yield
    reset_oracle_bridge()


class TestOracleBridge:
    def test_init(self):
        ob = OracleBridge()
        assert not ob.is_available
        state = ob.get_state()
        assert state["queries_total"] == 0
        assert state["oracle_available"] is False

    def test_heuristic_forecast(self):
        ob = OracleBridge()
        f = ob.query_forecast(domain="technology", horizon_days=30)
        assert isinstance(f, OracleForecast)
        assert f.domain == "technology"
        assert f.confidence > 0.0
        assert f.uncertainty > 0.0
        assert len(f.key_signals) > 0
        assert f.risk_level in ("low", "moderate", "high")

    def test_heuristic_forecast_domains(self):
        ob = OracleBridge()
        for domain in ["technology", "economics", "geopolitics", "science"]:
            f = ob.query_forecast(domain=domain, horizon_days=60)
            assert f.domain == domain
            assert f.time_horizon_days == 60

    def test_heuristic_uncertainty_increases_with_horizon(self):
        ob = OracleBridge()
        short = ob.query_forecast(domain="technology", horizon_days=7)
        long = ob.query_forecast(domain="technology", horizon_days=365)
        assert long.uncertainty >= short.uncertainty

    def test_caching(self):
        ob = OracleBridge()
        f1 = ob.query_forecast(domain="economics", horizon_days=30)
        f2 = ob.query_forecast(domain="economics", horizon_days=30)
        assert ob.get_state()["cache_hits"] >= 1

    def test_metrics_tracking(self):
        ob = OracleBridge()
        ob.query_forecast(domain="tech")
        ob.query_forecast(domain="economics")
        state = ob.get_state()
        assert state["queries_total"] == 2
        assert state["queries_success"] == 2
        assert state["last_domain"] == "economics"

    def test_get_uncertainty_for_affect(self):
        ob = OracleBridge()
        result = ob.get_uncertainty_for_affect(domain="technology")
        assert "prediction_uncertainty" in result
        assert "confidence" in result
        assert "risk_level" in result
        assert "domain" in result
        assert "signals" in result

    def test_cross_domain_signals(self):
        ob = OracleBridge()
        signals = ob.get_cross_domain_signals(["technology", "economics"])
        assert len(signals) == 2
        for s in signals:
            assert "domain" in s
            assert "confidence" in s
            assert "risk" in s

    def test_recent_forecasts(self):
        ob = OracleBridge()
        ob.query_forecast(domain="tech")
        ob.query_forecast(domain="economics")
        recent = ob.get_recent_forecasts(limit=5)
        assert len(recent) == 2
        for f in recent:
            assert "domain" in f
            assert "confidence" in f

    def test_recent_forecasts_limit(self):
        ob = OracleBridge()
        for i in range(15):
            ob.query_forecast(domain=f"domain_{i % 5}")
        recent = ob.get_recent_forecasts(limit=5)
        assert len(recent) == 5

    def test_get_state(self):
        ob = OracleBridge()
        ob.query_forecast(domain="test")
        state = ob.get_state()
        assert "oracle_available" in state
        assert "queries_total" in state
        assert "avg_confidence" in state
        assert "recent_forecasts" in state

    def test_try_connect_graceful_failure(self):
        ob = OracleBridge()
        result = ob.try_connect()
        assert isinstance(result, bool)

    def test_singleton(self):
        ob1 = get_oracle_bridge()
        ob2 = get_oracle_bridge()
        assert ob1 is ob2

    def test_reset(self):
        ob1 = get_oracle_bridge()
        ob1.query_forecast(domain="test")
        reset_oracle_bridge()
        ob2 = get_oracle_bridge()
        assert ob2.get_state()["queries_total"] == 0

    def test_thread_safety(self):
        ob = OracleBridge()
        errors = []

        def worker(tid):
            try:
                for i in range(20):
                    ob.query_forecast(domain=f"domain_{tid}_{i}")
                    ob.get_state()
                    ob.get_recent_forecasts(limit=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert ob.get_state()["queries_total"] == 100


class TestOracleForecast:
    def test_defaults(self):
        f = OracleForecast()
        assert f.confidence == 0.5
        assert f.uncertainty == 0.5
        assert f.risk_level == "moderate"

    def test_custom(self):
        f = OracleForecast(
            domain="tech",
            confidence=0.8,
            uncertainty=0.2,
            risk_level="low",
        )
        assert f.domain == "tech"
        assert f.risk_level == "low"


class TestBridgeMetrics:
    def test_defaults(self):
        m = BridgeMetrics()
        assert m.queries_total == 0
        assert m.avg_confidence == 0.5