"""Memory Consolidation Engine - Background memory optimization during idle.

Inspired by hippocampal replay during sleep, this module:
1. Replays and strengthens important memories
2. Forgets/decays weak or redundant memories
3. Creates cross-references between related concepts
4. Compresses episodic memories into semantic knowledge
5. Reorganizes memory indices for faster retrieval

Runs as a background task when the system is idle or during scheduled consolidation windows.
"""
import logging
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConsolidationPhase(Enum):
    REPLAY = auto()
    DECAY = auto()
    CROSS_LINK = auto()
    COMPRESS = auto()
    REINDEX = auto()


@dataclass
class ConsolidationReport:
    phase: ConsolidationPhase
    memories_processed: int = 0
    memories_decayed: int = 0
    memories_strengthened: int = 0
    cross_links_created: int = 0
    memories_compressed: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryConsolidationEngine:
    """Background memory optimization engine."""

    _CONSOLIDATION_INTERVAL = 300
    _MIN_IDLE_SECONDS = 60
    _DECAY_HALF_LIFE_HOURS = 24
    _IMPORTANCE_DECAY_FACTOR = 0.98
    _MIN_IMPORTANCE = 0.05
    _REPLAY_BOOST = 1.15
    _MAX_REPLAYS_PER_CYCLE = 50
    _MAX_CROSS_LINKS_PER_CYCLE = 20
    _MAX_REPORTS = 100
    _CROSS_LINK_THRESHOLD = 0.75
    _MAX_MEMORY_WEIGHTS = 10000

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".jarvis" / "consolidation"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_consolidation: datetime | None = None
        self._last_activity: datetime = datetime.now()
        self._consolidation_count: int = 0

        self._reports: deque[ConsolidationReport] = deque(maxlen=self._MAX_REPORTS)
        self._memory_weights: OrderedDict[str, float] = OrderedDict()
        self._access_counts: dict[str, int] = {}
        self._cross_links: dict[str, set[str]] = defaultdict(set)
        self._tag_index: dict[str, set[str]] = defaultdict(set)
        self._compressed_facts: OrderedDict[str, dict] = OrderedDict()

        self._memory_ref: Any = None
        self._vector_memory_ref: Any = None
        self._pfc_ref: Any = None
        self._kg_ref: Any = None
        self._thought_sense_ref: Any = None

    def _get_memory(self):
        if self._memory_ref is None:
            try:
                from core.memory import get_memory
                self._memory_ref = get_memory()
            except Exception as e:
                logger.debug(f"Memory unavailable: {e}")
        return self._memory_ref

    def _get_vector_memory(self):
        if self._vector_memory_ref is None:
            try:
                from core.vector_memory import get_vector_memory
                self._vector_memory_ref = get_vector_memory()
            except Exception as e:
                logger.debug(f"Vector memory unavailable: {e}")
        return self._vector_memory_ref

    def _get_pfc(self):
        if self._pfc_ref is None:
            try:
                from core.prefrontal_cortex import get_prefrontal_cortex
                self._pfc_ref = get_prefrontal_cortex()
            except Exception as e:
                logger.debug(f"PFC unavailable: {e}")
        return self._pfc_ref

    def _get_kg(self):
        if self._kg_ref is None:
            try:
                from core.knowledge_graph import get_knowledge_graph
                self._kg_ref = get_knowledge_graph()
            except Exception as e:
                logger.debug(f"Knowledge Graph unavailable: {e}")
        return self._kg_ref

    def _get_thought_sense(self):
        if self._thought_sense_ref is None:
            try:
                from core.thought_sense import get_thought_sense
                self._thought_sense_ref = get_thought_sense()
            except Exception as e:
                logger.debug(f"Thought Sense unavailable: {e}")
        return self._thought_sense_ref

    def notify_activity(self):
        self._last_activity = datetime.now()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._consolidation_loop,
            name="memory-consolidation",
            daemon=True,
        )
        self._thread.start()
        logger.info("Memory consolidation engine started")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Memory consolidation engine stopped")

    def _consolidation_loop(self):
        while self._running:
            try:
                time.sleep(self._CONSOLIDATION_INTERVAL)
                if not self._running:
                    break

                idle_seconds = (datetime.now() - self._last_activity).total_seconds()
                if idle_seconds < self._MIN_IDLE_SECONDS:
                    continue

                self._run_consolidation_cycle()

            except Exception as e:
                logger.warning(f"Consolidation cycle error: {e}")
                time.sleep(60)

    def _run_consolidation_cycle(self):
        start_time = time.time()
        self._consolidation_count += 1

        logger.info(f"Starting consolidation cycle #{self._consolidation_count}")

        for phase in ConsolidationPhase:
            if not self._running:
                break
            self._execute_phase(phase)

        total_duration = time.time() - start_time
        self._last_consolidation = datetime.now()

        self._publish_compressed_to_kg()
        self._notify_thought_sense()

        logger.info(
            f"Consolidation cycle #{self._consolidation_count} completed "
            f"in {total_duration:.1f}s"
        )

    def _execute_phase(self, phase: ConsolidationPhase) -> ConsolidationReport:
        start_time = time.time()
        report = ConsolidationReport(phase=phase)

        try:
            if phase == ConsolidationPhase.REPLAY:
                report = self._phase_replay()
            elif phase == ConsolidationPhase.DECAY:
                report = self._phase_decay()
            elif phase == ConsolidationPhase.CROSS_LINK:
                report = self._phase_cross_link()
            elif phase == ConsolidationPhase.COMPRESS:
                report = self._phase_compress()
            elif phase == ConsolidationPhase.REINDEX:
                report = self._phase_reindex()
        except Exception as e:
            logger.warning(f"Consolidation phase {phase.name} failed: {e}")

        report.duration_seconds = time.time() - start_time
        with self._lock:
            self._reports.append(report)
        return report

    def _phase_replay(self) -> ConsolidationReport:
        report = ConsolidationReport(phase=ConsolidationPhase.REPLAY)

        with self._lock:
            weighted = sorted(
                self._memory_weights.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_memories = weighted[:self._MAX_REPLAYS_PER_CYCLE]

        for memory_id, weight in top_memories:
            with self._lock:
                self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
                old_weight = self._memory_weights.get(memory_id, 0.5)
                new_weight = min(1.0, old_weight * self._REPLAY_BOOST)
                self._memory_weights[memory_id] = new_weight
            report.memories_processed += 1
            report.memories_strengthened += 1

        return report

    def _phase_decay(self) -> ConsolidationReport:
        report = ConsolidationReport(phase=ConsolidationPhase.DECAY)

        with self._lock:
            memory_ids = list(self._memory_weights.keys())
            items = [(mid, self._memory_weights.get(mid, 0), self._access_counts.get(mid, 0)) for mid in memory_ids]

        to_remove = []
        for memory_id, weight, access_count in items:
            decayed = weight * (self._IMPORTANCE_DECAY_FACTOR ** (access_count + 1))
            if access_count > 0:
                decayed = min(1.0, decayed * (1 + 0.01 * access_count))

            with self._lock:
                self._memory_weights[memory_id] = decayed

            if decayed < self._MIN_IMPORTANCE:
                to_remove.append(memory_id)
            report.memories_processed += 1

        for memory_id in to_remove:
            with self._lock:
                self._memory_weights.pop(memory_id, None)
                self._access_counts.pop(memory_id, None)
                for linked in self._cross_links.pop(memory_id, set()):
                    self._cross_links[linked].discard(memory_id)
                for tag in list(self._tag_index.keys()):
                    self._tag_index[tag].discard(memory_id)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]
            report.memories_decayed += 1

        return report

    def _phase_cross_link(self) -> ConsolidationReport:
        report = ConsolidationReport(phase=ConsolidationPhase.CROSS_LINK)

        vector_memory = self._get_vector_memory()
        if vector_memory is None:
            return report

        try:
            with self._lock:
                memory_ids = list(self._memory_weights.keys())
                weights_snapshot = dict(self._memory_weights)

            pairs_checked = 0
            links_created = 0

            for i, mem_id_a in enumerate(memory_ids):
                if links_created >= self._MAX_CROSS_LINKS_PER_CYCLE:
                    break
                if not self._running:
                    break

                weight_a = weights_snapshot.get(mem_id_a, 0)
                if weight_a < 0.3:
                    continue

                for mem_id_b in memory_ids[i + 1:]:
                    if links_created >= self._MAX_CROSS_LINKS_PER_CYCLE:
                        break

                    with self._lock:
                        already_linked = mem_id_b in self._cross_links.get(mem_id_a, set())
                    if already_linked:
                        continue

                    similarity = self._compute_similarity(mem_id_a, mem_id_b)
                    pairs_checked += 1

                    if similarity >= self._CROSS_LINK_THRESHOLD:
                        with self._lock:
                            self._cross_links[mem_id_a].add(mem_id_b)
                            self._cross_links[mem_id_b].add(mem_id_a)
                        links_created += 1

            report.memories_processed = pairs_checked
            report.cross_links_created = links_created

        except Exception as e:
            logger.debug(f"Cross-link phase failed: {e}")

        return report

    def _compute_similarity(self, id_a: str, id_b: str) -> float:
        tags_a = self._tag_index.get(id_a, set())
        tags_b = self._tag_index.get(id_b, set())

        if not tags_a or not tags_b:
            return 0.0

        intersection = len(tags_a & tags_b)
        union = len(tags_a | tags_b)
        if union == 0:
            return 0.0
        return intersection / union

    def _phase_compress(self) -> ConsolidationReport:
        report = ConsolidationReport(phase=ConsolidationPhase.COMPRESS)

        with self._lock:
            clusters = self._find_clusters()

        for cluster_id, members in clusters.items():
            if len(members) < 3:
                continue

            with self._lock:
                if cluster_id in self._compressed_facts:
                    continue

                total_weight = sum(
                    self._memory_weights.get(m, 0) for m in members
                )
                avg_weight = total_weight / len(members) if members else 0

                all_tags: set[str] = set()
                for m in members:
                    all_tags.update(self._tag_index.get(m, set()))

                compressed = {
                    "id": cluster_id,
                    "member_count": len(members),
                    "avg_weight": avg_weight,
                    "tags": list(all_tags)[:10],
                    "created_at": datetime.now().isoformat(),
                }
                self._compressed_facts[cluster_id] = compressed
                while len(self._compressed_facts) > 100:
                    self._compressed_facts.popitem(last=False)

            report.memories_compressed += 1

        report.memories_processed = len(clusters)
        return report

    def _find_clusters(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = {}
        visited: set[str] = set()

        for memory_id in list(self._cross_links.keys()):
            if memory_id in visited:
                continue

            cluster_members: list[str] = []
            stack = [memory_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster_members.append(current)
                for neighbor in self._cross_links.get(current, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(cluster_members) >= 3:
                cluster_id = f"cluster_{len(clusters)}_{int(time.time())}"
                clusters[cluster_id] = cluster_members

        return clusters

    def _phase_reindex(self) -> ConsolidationReport:
        report = ConsolidationReport(phase=ConsolidationPhase.REINDEX)

        with self._lock:
            new_tag_index: dict[str, set[str]] = defaultdict(set)
            for memory_id, weight in self._memory_weights.items():
                if weight >= self._MIN_IMPORTANCE:
                    for tag in self._tag_index.get(memory_id, set()):
                        new_tag_index[tag].add(memory_id)
            self._tag_index = dict(new_tag_index)

            stale_ids = set(self._cross_links.keys()) - set(self._memory_weights.keys())
            for stale_id in stale_ids:
                del self._cross_links[stale_id]

            for mid in list(self._cross_links.keys()):
                self._cross_links[mid] -= stale_ids

        report.memories_processed = len(self._memory_weights)
        return report

    def register_memory(self, memory_id: str, weight: float = 0.5, tags: set[str] | None = None):
        with self._lock:
            self._memory_weights[memory_id] = max(weight, self._memory_weights.get(memory_id, 0))
            self._memory_weights.move_to_end(memory_id)
            self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
            if tags:
                for tag in tags:
                    self._tag_index[tag].add(memory_id)
            while len(self._memory_weights) > self._MAX_MEMORY_WEIGHTS:
                oldest_id, _ = self._memory_weights.popitem(last=False)
                self._access_counts.pop(oldest_id, None)
                for linked in self._cross_links.pop(oldest_id, set()):
                    self._cross_links[linked].discard(oldest_id)

    def record_access(self, memory_id: str):
        with self._lock:
            self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
            if memory_id in self._memory_weights:
                self._memory_weights[memory_id] = min(
                    1.0, self._memory_weights[memory_id] * 1.02
                )

    def get_related(self, memory_id: str, max_results: int = 10) -> list[str]:
        with self._lock:
            direct = self._cross_links.get(memory_id, set())
            results = list(direct)[:max_results]
            remaining = max_results - len(results)
            if remaining > 0:
                related_tags: set[str] = set()
                for tag in self._tag_index.keys():
                    if memory_id in self._tag_index.get(tag, set()):
                        related_tags.add(tag)

                for tag in related_tags:
                    for mid in self._tag_index.get(tag, set()):
                        if mid != memory_id and mid not in results:
                            results.append(mid)
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break

            return results

    def get_compressed_knowledge(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {"id": k, **v}
                for k, v in list(self._compressed_facts.items())[-limit:]
            ]

    def get_consolidation_stats(self) -> dict:
        with self._lock:
            weights = list(self._memory_weights.values())
            avg_weight = sum(weights) / len(weights) if weights else 0

            return {
                "running": self._running,
                "consolidation_count": self._consolidation_count,
                "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
                "tracked_memories": len(self._memory_weights),
                "avg_weight": round(avg_weight, 3),
                "cross_links": sum(len(v) for v in self._cross_links.values()) // 2,
                "compressed_facts": len(self._compressed_facts),
                "tag_count": len(self._tag_index),
                "idle_seconds": (datetime.now() - self._last_activity).total_seconds(),
                "recent_reports": [
                    {
                        "phase": r.phase.name,
                        "processed": r.memories_processed,
                        "decayed": r.memories_decayed,
                        "strengthened": r.memories_strengthened,
                        "cross_links": r.cross_links_created,
                        "compressed": r.memories_compressed,
                        "duration": round(r.duration_seconds, 2),
                    }
                    for r in list(self._reports)[-5:]
                ],
            }

    def _publish_compressed_to_kg(self):
        kg = self._get_kg()
        if kg is None:
            return

        with self._lock:
            facts_to_publish = list(self._compressed_facts.values())

        for fact in facts_to_publish[-5:]:
            try:
                node_id = f"consolidated_{fact.get('id', 'unknown')}"
                existing = kg.get_node(node_id)
                if existing is not None:
                    continue

                tags = fact.get("tags", [])
                content = (
                    f"Compressed knowledge cluster ({fact.get('member_count', 0)} memories, "
                    f"avg weight {fact.get('avg_weight', 0):.2f}): "
                    f"tags=[{', '.join(tags[:5])}]"
                )
                kg.add_node(
                    node_id=node_id,
                    node_type="concept",
                    content=content,
                    metadata={
                        "source": "consolidation",
                        "member_count": fact.get("member_count", 0),
                        "avg_weight": fact.get("avg_weight", 0),
                        "tags": tags,
                    },
                )

                for tag in tags[:3]:
                    tag_node = f"tag_{tag}"
                    kg.add_node(tag_node, "concept", f"Tag: {tag}", {})
                    kg.add_edge(node_id, tag_node, "related_to", weight=0.5)

                logger.debug(f"Published compressed fact to KG: {node_id}")
            except Exception as e:
                logger.debug(f"Failed to publish fact to KG: {e}")

    def _notify_thought_sense(self):
        ts = self._get_thought_sense()
        if ts is None:
            return

        try:
            with self._lock:
                report = list(self._reports)[-1] if self._reports else None

            if report and report.memories_strengthened > 0:
                ts.observe(
                    content=f"consolidation_cycle_{self._consolidation_count}",
                    concept_type="system_maintenance",
                    source="memory_consolidation",
                )
        except Exception as e:
            logger.debug(f"Failed to notify Thought Sense: {e}")

    def force_consolidation(self) -> dict:
        if not self._running:
            self._running = True
        self._run_consolidation_cycle()
        return self.get_consolidation_stats()


_consolidation_engine: MemoryConsolidationEngine | None = None
_consolidation_lock = threading.Lock()


def get_consolidation_engine() -> MemoryConsolidationEngine:
    global _consolidation_engine
    if _consolidation_engine is None:
        with _consolidation_lock:
            if _consolidation_engine is None:
                _consolidation_engine = MemoryConsolidationEngine()
    return _consolidation_engine