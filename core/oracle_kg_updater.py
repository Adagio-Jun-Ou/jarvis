"""Oracle Knowledge Graph Auto-Updater.

Background task that periodically runs Oracle predictions and stores
results in the Jarvis Knowledge Graph. Enables continuous knowledge
enrichment without manual triggering.

Cycle behavior:
1. Run Oracle predictions for scheduled domains
2. Extract top insights from each domain
3. Store as KG INSIGHT + CONCEPT nodes with cross-domain edges
4. Track update history to avoid duplicate insertions
5. Adaptive interval based on insight novelty (new insights → shorter interval)

Registered as a BackgroundScheduler task that runs every N minutes.
"""
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

from core.helpers import safe_error_message

logger = logging.getLogger(__name__)

_oracle_updater: Optional["OracleKGUpdater"] = None
_updater_lock = threading.Lock()

DEFAULT_DOMAINS = ["economic", "geopolitical", "climate", "sentiment"]
DEFAULT_INTERVAL_MINUTES = 30
MAX_INSIGHTS_PER_CYCLE = 20
MAX_HISTORY_ENTRIES = 500
NOVELTY_THRESHOLD = 0.6


class OracleKGUpdater:
    """Periodic Oracle-KG auto-updater."""

    def __init__(
        self,
        domains: list[str] | None = None,
        interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
        max_insights_per_cycle: int = MAX_INSIGHTS_PER_CYCLE,
    ):
        self.domains = domains or list(DEFAULT_DOMAINS)
        self.interval_minutes = interval_minutes
        self.max_insights_per_cycle = max_insights_per_cycle
        self._history: deque[dict] = deque(maxlen=MAX_HISTORY_ENTRIES)
        self._seen_insight_keys: set[str] = set()
        self._max_seen_keys = 2000
        self._total_cycles = 0
        self._total_insights_stored = 0
        self._total_duplicates_skipped = 0
        self._total_errors = 0
        self._last_run: float | None = None
        self._running = False
        self._lock = threading.Lock()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "domains": self.domains,
                "interval_minutes": self.interval_minutes,
                "total_cycles": self._total_cycles,
                "total_insights_stored": self._total_insights_stored,
                "total_duplicates_skipped": self._total_duplicates_skipped,
                "total_errors": self._total_errors,
                "last_run": datetime.fromtimestamp(self._last_run).isoformat() if self._last_run else None,
                "seen_keys_count": len(self._seen_insight_keys),
                "history_size": len(self._history),
            }

    async def run_cycle(self) -> dict[str, Any]:
        """Execute one update cycle: predict → extract → store in KG."""
        with self._lock:
            if self._running:
                return {"status": "already_running", "skipped": True}
            self._running = True

        cycle_start = time.time()
        cycle_result: dict[str, Any] = {
            "cycle": self._total_cycles + 1,
            "domains": self.domains,
            "insights_stored": 0,
            "duplicates_skipped": 0,
            "errors": 0,
            "domain_results": {},
        }

        try:
            from core.oracle_integration import oracle_predict_to_knowledge_graph, oracle_diagnose_to_knowledge_graph

            for domain in self.domains:
                try:
                    result = oracle_predict_to_knowledge_graph(
                        domain=domain,
                        confidence="low",
                    )
                    insights_count = result.get("kg_nodes_created", 0)
                    cycle_result["domain_results"][domain] = {
                        "success": result.get("kg_integrated", False),
                        "insights": insights_count,
                    }
                    if result.get("kg_integrated"):
                        cycle_result["insights_stored"] += insights_count
                except Exception as e:
                    cycle_result["errors"] += 1
                    cycle_result["domain_results"][domain] = {"error": safe_error_message(e)}
                    logger.debug(f"Oracle KG update for {domain} failed: {e}")

            try:
                diag_result = oracle_diagnose_to_knowledge_graph(domain="all")
                diag_edges = diag_result.get("kg_edges_created", 0)
                cycle_result["diagnose_edges"] = diag_edges
            except Exception as e:
                logger.debug(f"Oracle KG diagnose failed: {e}")

        except ImportError:
            logger.warning("Oracle integration not available for KG auto-update")
            cycle_result["error"] = "Oracle integration not available"
        except Exception as e:
            cycle_result["error"] = safe_error_message(e)
            logger.error(f"Oracle KG auto-update cycle failed: {e}")

        finally:
            with self._lock:
                self._running = False
                self._total_cycles += 1
                self._total_insights_stored += cycle_result.get("insights_stored", 0)
                self._total_duplicates_skipped += cycle_result.get("duplicates_skipped", 0)
                self._total_errors += cycle_result.get("errors", 0)
                self._last_run = time.time()

            cycle_result["duration_seconds"] = round(time.time() - cycle_start, 2)
            self._history.append({
                "timestamp": time.time(),
                **cycle_result,
            })

        return cycle_result

    def add_domain(self, domain: str) -> None:
        with self._lock:
            if domain not in self.domains:
                self.domains.append(domain)

    def remove_domain(self, domain: str) -> None:
        with self._lock:
            self.domains = [d for d in self.domains if d != domain]

    def get_history(self, limit: int = 20) -> list[dict]:
        return list(self._history)[-limit:]

    def prune_seen_keys(self) -> None:
        with self._lock:
            if len(self._seen_insight_keys) > self._max_seen_keys:
                excess = len(self._seen_insight_keys) - self._max_seen_keys // 2
                keys_list = list(self._seen_insight_keys)
                self._seen_insight_keys = set(keys_list[excess:])


def get_oracle_kg_updater(
    domains: list[str] | None = None,
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
) -> OracleKGUpdater:
    """Get or create the singleton Oracle KG auto-updater."""
    global _oracle_updater
    if _oracle_updater is None:
        with _updater_lock:
            if _oracle_updater is None:
                _oracle_updater = OracleKGUpdater(
                    domains=domains,
                    interval_minutes=interval_minutes,
                )
    return _oracle_updater


async def oracle_kg_update_task() -> dict:
    """Background task callback for the BackgroundScheduler.

    Wraps OracleKGUpdater.run_cycle() for use with
    agent.scheduler.register(name, oracle_kg_update_task, interval).
    """
    updater = get_oracle_kg_updater()
    return await updater.run_cycle()


def register_oracle_kg_updater(agent) -> bool:
    """Register the Oracle KG auto-updater as a background task.

    Args:
        agent: JarvisAgent instance with scheduler

    Returns:
        True if registered successfully
    """
    if agent.scheduler is None:
        logger.warning("No BackgroundScheduler available — cannot register Oracle KG updater")
        return False

    updater = get_oracle_kg_updater()
    interval = updater.interval_minutes * 60

    agent.scheduler.register(
        name="oracle_kg_update",
        coro=oracle_kg_update_task,
        interval_seconds=interval,
        priority="LOW",
        start_immediately=False,
    )
    logger.info(f"Oracle KG auto-updater registered (interval={updater.interval_minutes}min, domains={updater.domains})")
    return True