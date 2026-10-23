"""Dream Mode for Jarvis - Memory Consolidation and System Optimization.

Inspired by human sleep cycles, dream mode enables Jarvis to:
1. Consolidate memories (short-term → long-term transfer)
2. Verify thought integrity and coherence
3. Generate intuitions from patterns
4. Optimize response flows
5. Clean up and refresh the system

Dream phases:
- Light Dream: Quick optimization and cleanup
- Deep Dream: Full memory consolidation and pattern extraction
- REM Dream: Creative recombination and intuition generation
"""
import logging
import threading
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


class DreamPhase(Enum):
    """Phases of the dream cycle."""
    LIGHT = "light"      # Quick cleanup and optimization
    DEEP = "deep"        # Memory consolidation
    REM = "rem"          # Intuition and creativity
    REFRESH = "refresh"  # System reset


@dataclass
class DreamMemory:
    """A memory fragment processed during dreaming."""
    id: str
    content: str
    timestamp: datetime
    importance: float  # 0-1
    access_count: int
    emotional_weight: float  # -1 to 1 (negative to positive)
    associations: list[str]  # IDs of related memories
    tags: list[str]
    consolidated: bool = False


@dataclass
class Intuition:
    """A generated intuition from pattern recognition."""
    id: str
    pattern: str
    insight: str
    confidence: float  # 0-1
    context_triggers: list[str]  # When to apply this intuition
    created: datetime
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class FlowPattern:
    """A learned response flow pattern."""
    trigger_pattern: str
    response_elements: list[str]
    efficiency: float  # How well this flow works
    usage_count: int
    last_used: Optional[datetime] = None


@dataclass
class DreamReport:
    """Report generated after a dream session."""
    phase: DreamPhase
    duration_seconds: float
    memories_processed: int
    memories_consolidated: int
    intuitions_generated: int
    patterns_optimized: int
    insights: list[str]
    system_health: dict


class DreamEngine:
    """Engine for Jarvis dream mode operations.

    Handles memory consolidation, pattern recognition, and system optimization
    during idle or scheduled dream periods.
    """

    def __init__(
        self,
        memory_path: Optional[Path] = None,
        llm_client: Any = None,
        consolidation_threshold: float = 0.3,
        importance_decay: float = 0.95,
    ):
        self.memory_path = memory_path or Path.home() / ".jarvis" / "dreams"
        self.memory_path.mkdir(parents=True, exist_ok=True)

        self.llm = llm_client
        self.consolidation_threshold = consolidation_threshold
        self.importance_decay = importance_decay

        self._lock = threading.RLock()

        # Memory stores
        self.short_term_memory: dict[str, DreamMemory] = {}
        self.long_term_memory: dict[str, DreamMemory] = {}

        # Bounds
        self.MAX_SHORT_TERM = 100
        self.MAX_LONG_TERM = 500
        self.MAX_INTUITIONS = 200
        self.MAX_FLOW_PATTERNS = 100

        # Pattern stores
        self.intuitions: dict[str, Intuition] = {}
        self.flow_patterns: dict[str, FlowPattern] = {}

        # Statistics
        self.dream_count = 0
        self.total_memories_consolidated = 0
        self.last_dream: Optional[datetime] = None

        # Load existing data
        self._load_state()

    def _load_state(self) -> None:
        """Load saved dream state from disk."""
        state_file = self.memory_path / "dream_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                with self._lock:
                    self.dream_count = data.get("dream_count", 0)
                    self.total_memories_consolidated = data.get("total_memories_consolidated", 0)
            except Exception as e:
                logger.warning(f"Failed to load dream state: {e}")

        # Load intuitions
        intuitions_file = self.memory_path / "intuitions.json"
        if intuitions_file.exists():
            try:
                items = read_json_with_integrity(intuitions_file)
                with self._lock:
                    for item in items:
                        self.intuitions[item["id"]] = Intuition(
                            id=item["id"],
                            pattern=item["pattern"],
                            insight=item["insight"],
                            confidence=item["confidence"],
                            context_triggers=item["context_triggers"],
                            created=datetime.fromisoformat(item["created"]),
                            usage_count=item.get("usage_count", 0),
                            success_rate=item.get("success_rate", 0.0),
                        )
            except Exception as e:
                logger.warning(f"Failed to load intuitions: {e}")

    def _save_state(self) -> None:
        """Save dream state to disk."""
        with self._lock:
            state_data = {
                "dream_count": self.dream_count,
                "total_memories_consolidated": self.total_memories_consolidated,
                "last_dream": self.last_dream.isoformat() if self.last_dream else None,
            }
            intuitions_data = [
                {
                    "id": i.id,
                    "pattern": i.pattern,
                    "insight": i.insight,
                    "confidence": i.confidence,
                    "context_triggers": i.context_triggers,
                    "created": i.created.isoformat(),
                    "usage_count": i.usage_count,
                    "success_rate": i.success_rate,
                }
                for i in self.intuitions.values()
            ]

        state_file = self.memory_path / "dream_state.json"
        atomic_write_json(state_file, state_data)

        intuitions_file = self.memory_path / "intuitions.json"
        atomic_write_json(intuitions_file, intuitions_data)

    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        emotional_weight: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Add a new memory to short-term storage."""
        memory_id = hashlib.sha256(f"{content}{datetime.now()}".encode()).hexdigest()[:8]

        memory = DreamMemory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            access_count=1,
            emotional_weight=emotional_weight,
            associations=[],
            tags=tags or [],
        )

        with self._lock:
            self.short_term_memory[memory_id] = memory

            while len(self.short_term_memory) > self.MAX_SHORT_TERM:
                oldest = min(self.short_term_memory.items(), key=lambda x: x[1].timestamp)
                del self.short_term_memory[oldest[0]]

        return memory_id

    def access_memory(self, memory_id: str) -> Optional[DreamMemory]:
        """Access a memory, incrementing its access count."""
        with self._lock:
            if memory_id in self.short_term_memory:
                self.short_term_memory[memory_id].access_count += 1
                return self.short_term_memory[memory_id]

            if memory_id in self.long_term_memory:
                self.long_term_memory[memory_id].access_count += 1
                return self.long_term_memory[memory_id]

        return None

    async def dream(
        self,
        phase: DreamPhase = DreamPhase.DEEP,
        duration: float = 5.0,
        memories: Optional[list[dict]] = None,
    ) -> DreamReport:
        """Enter dream mode for processing and optimization.

        Args:
            phase: Which dream phase to enter
            duration: Maximum duration in seconds
            memories: Optional list of recent memories to process

        Returns:
            DreamReport with results
        """
        start_time = datetime.now()
        with self._lock:
            self.dream_count += 1
            self.last_dream = start_time

        # Convert input memories
        if memories:
            for mem in memories:
                self.add_memory(
                    content=mem.get("content", ""),
                    importance=mem.get("importance", 0.5),
                    emotional_weight=mem.get("emotional_weight", 0.0),
                    tags=mem.get("tags", []),
                )

        insights = []
        memories_processed = 0
        memories_consolidated = 0
        intuitions_generated = 0
        patterns_optimized = 0

        if phase == DreamPhase.LIGHT:
            # Quick cleanup
            memories_processed, memories_consolidated = await self._light_dream()
            insights.append("Performed quick memory cleanup and optimization")

        elif phase == DreamPhase.DEEP:
            # Full consolidation
            memories_processed, memories_consolidated = await self._deep_dream()
            insights.append(f"Consolidated {memories_consolidated} memories to long-term storage")

            # Extract patterns
            new_intuitions = await self._extract_patterns()
            intuitions_generated = len(new_intuitions)
            if new_intuitions:
                insights.append(f"Generated {len(new_intuitions)} new intuitions")

        elif phase == DreamPhase.REM:
            # Creative recombination
            new_intuitions = await self._rem_dream()
            intuitions_generated = len(new_intuitions)
            insights.append(f"Dreamt {len(new_intuitions)} creative insights")

        elif phase == DreamPhase.REFRESH:
            # System reset
            await self._refresh_dream()
            insights.append("System refreshed and optimized")

        # Optimize flows
        patterns_optimized = self._optimize_flows()
        if patterns_optimized > 0:
            insights.append(f"Optimized {patterns_optimized} response patterns")

        with self._lock:
            system_health = {
                "short_term_memory_count": len(self.short_term_memory),
                "long_term_memory_count": len(self.long_term_memory),
                "intuition_count": len(self.intuitions),
                "flow_pattern_count": len(self.flow_patterns),
                "total_dreams": self.dream_count,
                "consolidation_rate": self.total_memories_consolidated / max(1, self.dream_count),
            }

        self._save_state()

        duration_actual = (datetime.now() - start_time).total_seconds()

        return DreamReport(
            phase=phase,
            duration_seconds=duration_actual,
            memories_processed=memories_processed,
            memories_consolidated=memories_consolidated,
            intuitions_generated=intuitions_generated,
            patterns_optimized=patterns_optimized,
            insights=insights,
            system_health=system_health,
        )

    async def _light_dream(self) -> tuple[int, int]:
        """Light dream: quick cleanup and minor consolidation."""
        processed = 0
        consolidated = 0

        with self._lock:
            for memory_id, memory in list(self.short_term_memory.items()):
                memory.importance *= self.importance_decay
                processed += 1

                if memory.importance > 0.7 or memory.access_count > 3:
                    self.long_term_memory[memory_id] = memory
                    memory.consolidated = True
                    consolidated += 1

            self.short_term_memory = {
                k: v for k, v in self.short_term_memory.items()
                if v.importance > 0.1
            }

            while len(self.long_term_memory) > self.MAX_LONG_TERM:
                oldest = min(self.long_term_memory.items(), key=lambda x: x[1].timestamp)
                del self.long_term_memory[oldest[0]]

            self.total_memories_consolidated += consolidated
        return processed, consolidated

    async def _deep_dream(self) -> tuple[int, int]:
        """Deep dream: full memory consolidation."""
        processed = 0
        consolidated = 0

        processed, consolidated = await self._light_dream()

        await self._build_associations()

        with self._lock:
            for memory_id, memory in list(self.short_term_memory.items()):
                processed += 1

                association_boost = len(memory.associations) * 0.1
                emotional_boost = abs(memory.emotional_weight) * 0.2
                score = memory.importance + association_boost + emotional_boost

                if score > self.consolidation_threshold:
                    self.long_term_memory[memory_id] = memory
                    memory.consolidated = True
                    consolidated += 1

            self.total_memories_consolidated += consolidated

            while len(self.long_term_memory) > self.MAX_LONG_TERM:
                lowest_importance = min(self.long_term_memory.items(), key=lambda x: x[1].importance)
                del self.long_term_memory[lowest_importance[0]]

        return processed, consolidated

    async def _build_associations(self) -> None:
        """Build associations between related memories."""
        with self._lock:
            memories = list(self.short_term_memory.values()) + list(self.long_term_memory.values())

        for i, mem_a in enumerate(memories):
            for mem_b in memories[i+1:]:
                similarity = self._calculate_similarity(mem_a, mem_b)

                if similarity > 0.3:
                    with self._lock:
                        if mem_b.id not in mem_a.associations:
                            mem_a.associations.append(mem_b.id)
                        if mem_a.id not in mem_b.associations:
                            mem_b.associations.append(mem_a.id)

    def _calculate_similarity(self, mem_a: DreamMemory, mem_b: DreamMemory) -> float:
        """Calculate similarity between two memories."""
        # Tag overlap
        tags_a = set(mem_a.tags)
        tags_b = set(mem_b.tags)
        tag_similarity = len(tags_a & tags_b) / max(1, len(tags_a | tags_b))

        # Temporal proximity
        time_diff = abs((mem_a.timestamp - mem_b.timestamp).total_seconds())
        time_similarity = max(0, 1 - time_diff / 86400)  # Decay over 1 day

        # Content similarity (simple word overlap)
        words_a = set(mem_a.content.lower().split())
        words_b = set(mem_b.content.lower().split())
        content_similarity = len(words_a & words_b) / max(1, len(words_a | words_b))

        return (tag_similarity * 0.4 + time_similarity * 0.3 + content_similarity * 0.3)

    async def _extract_patterns(self) -> list[Intuition]:
        """Extract patterns and generate intuitions."""
        new_intuitions = []

        with self._lock:
            tag_counts = Counter()
            for memory in self.long_term_memory.values():
                for tag in memory.tags:
                    tag_counts[tag] += 1

            common_tags = [tag for tag, count in tag_counts.most_common(10) if count > 2]

            for tag in common_tags:
                related = [
                    m for m in self.long_term_memory.values()
                    if tag in m.tags
                ]

                if len(related) >= 3:
                    intuition_id = hashlib.sha256(f"pattern_{tag}".encode()).hexdigest()[:8]

                    avg_importance = sum(m.importance for m in related) / len(related)

                    intuition = Intuition(
                        id=intuition_id,
                        pattern=f"recurring theme: {tag}",
                        insight=f"The theme '{tag}' appears frequently with {avg_importance:.0%} importance",
                        confidence=min(0.9, len(related) * 0.15),
                        context_triggers=[tag],
                        created=datetime.now(),
                    )

                    if intuition_id not in self.intuitions:
                        self.intuitions[intuition_id] = intuition
                        new_intuitions.append(intuition)

                        while len(self.intuitions) > self.MAX_INTUITIONS:
                            lowest_conf = min(self.intuitions.items(), key=lambda x: x[1].confidence)
                            del self.intuitions[lowest_conf[0]]

        return new_intuitions

    async def _rem_dream(self) -> list[Intuition]:
        """REM dream: creative recombination and insight generation."""
        new_intuitions = []

        clusters = self._find_memory_clusters()

        for cluster in clusters:
            if len(cluster) < 3:
                continue

            with self._lock:
                memories = [self.long_term_memory.get(m) or self.short_term_memory.get(m) for m in cluster]
                memories = [m for m in memories if m]

            if len(memories) < 3:
                continue

            all_tags = []
            for m in memories:
                all_tags.extend(m.tags)

            tag_counts = Counter(all_tags)
            top_tags = [t for t, _ in tag_counts.most_common(3)]

            intuition_id = hashlib.sha256(f"rem_{'_'.join(top_tags)}".encode()).hexdigest()[:8]

            intuition = Intuition(
                id=intuition_id,
                pattern=f"creative cluster: {' + '.join(top_tags)}",
                insight=f"Connection discovered between {', '.join(top_tags)}",
                confidence=len(memories) * 0.1,
                context_triggers=top_tags,
                created=datetime.now(),
            )

            with self._lock:
                if intuition_id not in self.intuitions:
                    self.intuitions[intuition_id] = intuition
                    new_intuitions.append(intuition)

                    while len(self.intuitions) > self.MAX_INTUITIONS:
                        lowest_conf = min(self.intuitions.items(), key=lambda x: x[1].confidence)
                        del self.intuitions[lowest_conf[0]]

        return new_intuitions

    def _find_memory_clusters(self) -> list[list[str]]:
        """Find clusters of associated memories."""
        with self._lock:
            ltm_keys = list(self.long_term_memory.keys())
            stm_keys = list(self.short_term_memory.keys())

        visited = set()
        clusters = []

        def dfs(memory_id: str, cluster: list) -> None:
            if memory_id in visited:
                return
            visited.add(memory_id)
            cluster.append(memory_id)

            with self._lock:
                memory = self.long_term_memory.get(memory_id) or self.short_term_memory.get(memory_id)
            if memory:
                for assoc_id in memory.associations:
                    dfs(assoc_id, cluster)

        for memory_id in ltm_keys + stm_keys:
            if memory_id not in visited:
                cluster = []
                dfs(memory_id, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters

    async def _refresh_dream(self) -> None:
        """Refresh dream: clear and reset system."""
        with self._lock:
            for memory_id, memory in list(self.short_term_memory.items()):
                if memory.importance > 0.3 or memory.access_count > 1:
                    self.long_term_memory[memory_id] = memory

            self.short_term_memory.clear()

            for memory in self.long_term_memory.values():
                memory.associations = [
                    a for a in memory.associations
                    if a in self.long_term_memory
                ]

    def _optimize_flows(self) -> int:
        """Optimize response flow patterns."""
        optimized = 0

        with self._lock:
            for pattern in self.flow_patterns.values():
                if pattern.last_used:
                    days_unused = (datetime.now() - pattern.last_used).days
                    if days_unused > 7:
                        pattern.efficiency *= 0.9
                        optimized += 1

            self.flow_patterns = {
                k: v for k, v in self.flow_patterns.items()
                if v.efficiency > 0.3
            }

        return optimized

    def get_intuition_for_context(self, context: str) -> Optional[Intuition]:
        """Get a relevant intuition for a given context."""
        context_lower = context.lower()
        matching_intuitions = []

        with self._lock:
            for intuition in self.intuitions.values():
                for trigger in intuition.context_triggers:
                    if trigger.lower() in context_lower:
                        matching_intuitions.append(intuition)
                        break

        if matching_intuitions:
            return max(matching_intuitions, key=lambda i: i.confidence * (1 + i.success_rate))

        return None

    def record_intuition_result(self, intuition_id: str, success: bool) -> None:
        """Record whether an intuition was successful."""
        with self._lock:
            if intuition_id in self.intuitions:
                intuition = self.intuitions[intuition_id]
                intuition.usage_count += 1
                alpha = 0.1
                intuition.success_rate = (1 - alpha) * intuition.success_rate + alpha * (1 if success else 0)

    def get_status(self) -> dict:
        """Get current dream engine status."""
        with self._lock:
            return {
                "dream_count": self.dream_count,
                "total_memories_consolidated": self.total_memories_consolidated,
                "short_term_memory_count": len(self.short_term_memory),
                "long_term_memory_count": len(self.long_term_memory),
                "intuition_count": len(self.intuitions),
                "flow_pattern_count": len(self.flow_patterns),
                "last_dream": self.last_dream.isoformat() if self.last_dream else None,
                "memory_health": {
                    "short_term_utilization": len(self.short_term_memory) / 100,
                    "consolidation_rate": self.total_memories_consolidated / max(1, self.dream_count),
                }
            }


class DreamModeTool:
    """Tool for Jarvis to enter dream mode."""

    name = "dream"
    description = "Enter dream mode for memory consolidation, pattern recognition, and system optimization"
    parameters = {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": ["light", "deep", "rem", "refresh"],
                "description": "Dream phase (light=quick cleanup, deep=consolidation, rem=creative, refresh=reset)"
            },
            "duration": {
                "type": "number",
                "description": "Maximum duration in seconds (default: 5.0)"
            },
            "memories": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional memories to process"
            }
        },
        "required": []
    }
    requires_permission = False

    def __init__(self, engine: Optional[DreamEngine] = None):
        self.engine = engine or DreamEngine()

    async def execute(
        self,
        phase: str = "deep",
        duration: float = 5.0,
        memories: Optional[list[dict]] = None,
    ) -> str:
        """Execute dream mode."""
        if not phase:
            phase = "deep"
        phase_map = {
            "light": DreamPhase.LIGHT,
            "deep": DreamPhase.DEEP,
            "rem": DreamPhase.REM,
            "refresh": DreamPhase.REFRESH,
        }

        dream_phase = phase_map.get(phase.lower(), DreamPhase.DEEP)
        report = await self.engine.dream(dream_phase, duration, memories)

        lines = [
            f"😴 Dream Mode ({report.phase.value})",
            f"   Duration: {report.duration_seconds:.2f}s",
            f"   Memories processed: {report.memories_processed}",
            f"   Memories consolidated: {report.memories_consolidated}",
            f"   Intuitions generated: {report.intuitions_generated}",
            f"   Patterns optimized: {report.patterns_optimized}",
            "",
            "   Insights:",
        ]

        for insight in report.insights:
            lines.append(f"   • {insight}")

        lines.append("")
        lines.append("   System Health:")
        for key, value in report.system_health.items():
            if isinstance(value, float):
                lines.append(f"   • {key}: {value:.2f}")
            else:
                lines.append(f"   • {key}: {value}")

        return "\n".join(lines)
