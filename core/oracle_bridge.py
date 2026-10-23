"""Oracle Integration Bridge for Jarvis.

Connects Jarvis's consciousness architecture with the Oracle Prediction Engine.
This allows Jarvis to:

1. **Query Oracle predictions** relevant to current conversation topics
2. **Feed forecast uncertainty** into Jarvis's affective context
   (high uncertainty → increased prediction error → arousal)
3. **Use cross-domain interactions** to enrich context understanding
4. **Apply forecast confidence** to adjust Jarvis's response certainty

The bridge is designed to be optional and gracefully degrades when Oracle
is unavailable. Integration points:
- Oracle prediction errors → Jarvis predictive processing
- Oracle forecast confidence → Jarvis affective context (certainty)
- Oracle cross-domain interactions → Jarvis attention system
- Oracle scenario outcomes → Jarvis proactive suggestions
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_oracle_bridge: Optional["OracleBridge"] = None
_ob_lock = threading.Lock()


@dataclass
class OracleForecast:
    domain: str = ""
    summary: str = ""
    confidence: float = 0.5
    uncertainty: float = 0.5
    time_horizon_days: int = 30
    key_signals: list = field(default_factory=list)
    risk_level: str = "moderate"


@dataclass
class BridgeMetrics:
    queries_total: int = 0
    queries_success: int = 0
    queries_failed: int = 0
    last_query_time: float = 0.0
    last_domain: str = ""
    cache_hits: int = 0
    avg_confidence: float = 0.5


class OracleBridge:
    _CACHE_TTL = 300.0

    def __init__(self):
        self._lock = threading.RLock()
        self._oracle_engine = None
        self._oracle_available = False
        self._cache: dict[str, tuple[OracleForecast, float]] = {}
        self._metrics = BridgeMetrics()
        self._recent_forecasts: list[OracleForecast] = []
        self._last_connect_attempt = 0.0

    def try_connect(self) -> bool:
        with self._lock:
            try:
                from oracle.prediction.engine import OracleEngine
                self._oracle_engine = OracleEngine()
                self._oracle_available = True
                logger.info("OracleBridge: connected to Oracle Prediction Engine")
                return True
            except ImportError:
                logger.debug("OracleBridge: Oracle not available, using heuristic mode")
                self._oracle_available = False
                return False
            except Exception as e:
                logger.debug(f"OracleBridge: connection failed: {e}")
                self._oracle_available = False
                return False

    @property
    def is_available(self) -> bool:
        return self._oracle_available

    def query_forecast(self, domain: str = "", topic: str = "",
                        horizon_days: int = 30) -> OracleForecast:
        with self._lock:
            self._metrics.queries_total += 1
            cache_key = f"{domain}:{topic}:{horizon_days}"
            if cache_key in self._cache:
                cached_forecast, cached_time = self._cache[cache_key]
                if time.time() - cached_time < self._CACHE_TTL:
                    self._metrics.cache_hits += 1
                    return cached_forecast

            if self._oracle_available and self._oracle_engine:
                forecast = self._query_oracle(domain, topic, horizon_days)
            else:
                forecast = self._heuristic_forecast(domain, topic, horizon_days)

            self._cache[cache_key] = (forecast, time.time())
            self._metrics.queries_success += 1
            self._metrics.last_query_time = time.time()
            self._metrics.last_domain = domain
            self._recent_forecasts.append(forecast)
            if len(self._recent_forecasts) > 20:
                self._recent_forecasts = self._recent_forecasts[-20:]
            return forecast

    def _query_oracle(self, domain: str, topic: str,
                       horizon_days: int) -> OracleForecast:
        try:
            from datetime import date, timedelta
            target_date = date.today() + timedelta(days=horizon_days)
            result = self._oracle_engine.predict(target_date)
            if result and hasattr(result, 'predictions'):
                preds = result.predictions if hasattr(result, 'predictions') else []
                if preds:
                    conf = sum(getattr(p, 'confidence', 0.5) for p in preds) / len(preds)
                    key_signals = []
                    for p in preds[:5]:
                        if hasattr(p, 'summary'):
                            key_signals.append(p.summary[:100])
                    return OracleForecast(
                        domain=domain or "general",
                        summary=f"Oracle: {len(preds)} predictions for {domain or 'general'}",
                        confidence=round(conf, 3),
                        uncertainty=round(1.0 - conf, 3),
                        time_horizon_days=horizon_days,
                        key_signals=key_signals[:5],
                        risk_level="high" if conf < 0.3 else "moderate" if conf < 0.6 else "low",
                    )
            return OracleForecast(domain=domain, summary="Oracle: no predictions available")
        except Exception as e:
            logger.debug(f"Oracle query failed: {e}")
            self._metrics.queries_failed += 1
            return self._heuristic_forecast(domain, topic, horizon_days)

    def _heuristic_forecast(self, domain: str, topic: str,
                             horizon_days: int) -> OracleForecast:
        uncertainty = min(0.9, 0.3 + horizon_days * 0.005)
        confidence = 1.0 - uncertainty
        domain_signals = {
            "technology": ["AI acceleration", "compute scaling", "open source momentum"],
            "economics": ["inflation trends", "employment data", "policy shifts"],
            "geopolitics": ["alliance shifts", "trade patterns", "conflict zones"],
            "science": ["research breakthroughs", "funding trends", "replication concerns"],
        }
        signals = domain_signals.get(domain.lower() if domain else "technology",
                                     ["general trends", "cross-domain effects"])
        return OracleForecast(
            domain=domain or "general",
            summary=f"Heuristic: baseline forecast for {domain or 'general'} ({horizon_days}d)",
            confidence=round(confidence, 3),
            uncertainty=round(uncertainty, 3),
            time_horizon_days=horizon_days,
            key_signals=signals[:3],
            risk_level="moderate",
        )

    def get_uncertainty_for_affect(self, domain: str = "") -> dict:
        forecast = self.query_forecast(domain=domain)
        return {
            "prediction_uncertainty": forecast.uncertainty,
            "confidence": forecast.confidence,
            "risk_level": forecast.risk_level,
            "domain": forecast.domain,
            "signals": forecast.key_signals,
        }

    def get_cross_domain_signals(self, domains: list[str] = None) -> list[dict]:
        if domains is None:
            domains = ["technology", "economics", "geopolitics"]
        signals = []
        for d in domains:
            f = self.query_forecast(domain=d, horizon_days=30)
            signals.append({
                "domain": f.domain,
                "confidence": f.confidence,
                "risk": f.risk_level,
                "top_signal": f.key_signals[0] if f.key_signals else "",
            })
        return signals

    def get_recent_forecasts(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [
                {
                    "domain": f.domain,
                    "summary": f.summary,
                    "confidence": f.confidence,
                    "uncertainty": f.uncertainty,
                    "risk_level": f.risk_level,
                    "horizon_days": f.time_horizon_days,
                }
                for f in self._recent_forecasts[-limit:]
            ]

    def get_state(self) -> dict:
        with self._lock:
            avg_conf = self._metrics.avg_confidence
            if self._recent_forecasts:
                avg_conf = sum(f.confidence for f in self._recent_forecasts) / len(self._recent_forecasts)
                self._metrics.avg_confidence = avg_conf
            return {
                "oracle_available": self._oracle_available,
                "queries_total": self._metrics.queries_total,
                "queries_success": self._metrics.queries_success,
                "queries_failed": self._metrics.queries_failed,
                "cache_hits": self._metrics.cache_hits,
                "last_domain": self._metrics.last_domain,
                "avg_confidence": round(avg_conf, 3),
                "recent_forecasts": [
                    {"domain": f.domain, "confidence": f.confidence, "risk": f.risk_level}
                    for f in self._recent_forecasts[-5:]
                ],
            }


def get_oracle_bridge() -> OracleBridge:
    global _oracle_bridge
    if _oracle_bridge is None:
        with _ob_lock:
            if _oracle_bridge is None:
                _oracle_bridge = OracleBridge()
    return _oracle_bridge


def reset_oracle_bridge() -> None:
    global _oracle_bridge
    with _ob_lock:
        _oracle_bridge = None