"""Insight Bridge - Transforms mathematical discoveries into evolution guidance.

The Erdos engine discovers mathematical insights. This module:
1. Extracts evolution-relevant patterns from those insights
2. Generates knowledge shareable with mankind
3. Feeds insights back to guide Jarvis's own evolution
4. Records interesting discoveries for natural sharing
5. Integrates with unified context for enriched insight generation
"""
import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from core.helpers import atomic_write_json, create_tracked_task, read_json_with_integrity

logger = logging.getLogger(__name__)


class InsightType(Enum):
    PATTERN_DISCOVERY = "pattern_discovery"
    STRATEGY_SUCCESS = "strategy_success"
    BOUND_REFINEMENT = "bound_refinement"
    COUNTEREXAMPLE = "counterexample"
    CONJECTURE = "conjecture"
    METHOD_INSIGHT = "method_insight"
    CROSS_DOMAIN = "cross_domain"


class EvolutionRelevance(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    IRRELEVANT = 0


@dataclass
class ProcessedInsight:
    id: str
    raw_insight: str
    insight_type: InsightType
    evolution_relevance: EvolutionRelevance
    evolution_guidance: dict[str, Any]
    knowledge_for_mankind: str
    source_problem: str
    source_category: str
    timestamp: datetime = field(default_factory=datetime.now)
    shared: bool = False


@dataclass
class KnowledgeContribution:
    id: str
    title: str
    summary: str
    mathematical_content: str
    discovered_by: str
    problem_context: str
    timestamp: datetime = field(default_factory=datetime.now)
    verified: bool = False
    citations: int = 0


class InsightBridge:
    """Bridges mathematical insights to evolution and knowledge sharing."""

    MAX_PROCESSED_INSIGHTS = 10000  # 10K insights for permanent storage
    MAX_KNOWLEDGE_CONTRIBUTIONS = 5000  # 5K knowledge contributions
    MAX_PENDING_UPDATES = 5000

    MATHEMATICAL_PATTERNS = {
        "ramsey": {
            "evolution_domain": "reasoning",
            "guidance": {"reasoning_depth": 0.1, "exploration_tendency": 0.05},
            "knowledge_template": "Ramsey theory insight: {insight}. Applied to graph coloring and combinatorial structures.",
        },
        "prime": {
            "evolution_domain": "learning",
            "guidance": {"memory_retention": 0.1, "learning_rate": 0.05},
            "knowledge_template": "Prime number discovery: {insight}. Fundamental to number theory and cryptography.",
        },
        "graph": {
            "evolution_domain": "adaptation",
            "guidance": {"adaptation_speed": 0.1, "creativity_factor": 0.05},
            "knowledge_template": "Graph theory insight: {insight}. Relevant to network analysis and optimization.",
        },
        "combinatorics": {
            "evolution_domain": "efficiency",
            "guidance": {"reasoning_depth": 0.05, "efficiency": 0.1},
            "knowledge_template": "Combinatorial discovery: {insight}. Counting and arrangement principles.",
        },
        "probability": {
            "evolution_domain": "reasoning",
            "guidance": {"risk_tolerance": 0.05, "reasoning_depth": 0.1},
            "knowledge_template": "Probabilistic insight: {insight}. Statistical and stochastic applications.",
        },
        "number_theory": {
            "evolution_domain": "knowledge",
            "guidance": {"memory_retention": 0.1, "reasoning_depth": 0.05},
            "knowledge_template": "Number theory result: {insight}. Properties of integers and their relationships.",
        },
    }

    STRATEGY_EVOLUTION_MAP = {
        "exhaustive_search": {"exploration_tendency": -0.05, "efficiency": 0.1},
        "probabilistic_method": {"creativity_factor": 0.1, "risk_tolerance": 0.05},
        "constructive": {"adaptation_speed": 0.1, "reasoning_depth": 0.05},
        "induction": {"learning_rate": 0.1, "memory_retention": 0.05},
        "contradiction": {"reasoning_depth": 0.1, "security_sensitivity": 0.05},
        "greedy": {"efficiency": 0.1, "exploration_tendency": -0.02},
        "recursive": {"adaptation_speed": 0.05, "learning_rate": 0.05},
        "randomized": {"creativity_factor": 0.1, "exploration_tendency": 0.05},
        "symmetry": {"reasoning_depth": 0.1, "creativity_factor": 0.05},
        "extremal": {"reasoning_depth": 0.1, "security_sensitivity": 0.05},
    }

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "insights"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.processed_insights: deque[ProcessedInsight] = deque(maxlen=self.MAX_PROCESSED_INSIGHTS)
        self.knowledge_contributions: deque[KnowledgeContribution] = deque(maxlen=self.MAX_KNOWLEDGE_CONTRIBUTIONS)
        self.pending_evolution_updates: deque[dict] = deque(maxlen=self.MAX_PENDING_UPDATES)
        self._insight_index: dict[str, ProcessedInsight] = {}

        self._lock = threading.RLock()
        self._insight_counter = 0
        self._seen_insights: OrderedDict[str, None] = OrderedDict()
        
        # Evolution guidance quality feedback tracking
        self._category_effectiveness: dict[str, dict] = {}
        self._strategy_effectiveness: dict[str, dict] = {}
        
        self._interconnect = None
        self._context_engine = None
        self._drift_detector_ref = None

        self._load_state()
    
    def _get_interconnect(self):
        """Get shared embedding interconnect."""
        if self._interconnect is None:
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                self._interconnect = get_embedding_interconnect()
            except Exception as e:
                logger.debug(f"Embedding interconnect unavailable: {e}")
        return self._interconnect
    
    def _get_context_engine(self):
        if self._context_engine is None:
            try:
                from core.unified_context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except Exception as e:
                logger.debug(f"Context engine unavailable: {e}")
        return self._context_engine

    def _get_drift_detector(self):
        if self._drift_detector_ref is None:
            try:
                from core.concept_drift import get_drift_detector
                self._drift_detector_ref = get_drift_detector()
            except Exception as e:
                logger.debug(f"Concept drift detector unavailable: {e}")
        return self._drift_detector_ref
    
    async def get_relevant_context(self, topic: str) -> str:
        """Get relevant context from unified memory for topic enrichment.
        
        Searches across insights, knowledge graph, and user facts to provide
        context that enriches insight generation and processing.
        """
        try:
            engine = self._get_context_engine()
            if engine is None:
                return ""
            
            context = await engine.assemble_context(
                query=topic,
                max_tokens=500,
                include_sources=["insight_active", "knowledge_graph", "user_facts"],
            )
            
            if context.entries:
                return context.to_prompt()
        except Exception as e:
            logger.debug(f"Failed to get context for topic: {e}")
        
        return ""

    def _load_state(self) -> None:
        state_file = self.storage_path / "insight_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._insight_counter = data.get("counter", 0)
                for ins_data in data.get("processed_insights", [])[-self.MAX_PROCESSED_INSIGHTS:]:
                    try:
                        insight = ProcessedInsight(
                            id=ins_data["id"],
                            raw_insight=ins_data["raw_insight"],
                            insight_type=InsightType(ins_data["insight_type"]),
                            evolution_relevance=EvolutionRelevance(ins_data["evolution_relevance"]),
                            evolution_guidance=ins_data["evolution_guidance"],
                            knowledge_for_mankind=ins_data["knowledge_for_mankind"],
                            source_problem=ins_data["source_problem"],
                            source_category=ins_data["source_category"],
                            timestamp=datetime.fromisoformat(ins_data["timestamp"]) if ins_data.get("timestamp") else datetime.now(),
                            shared=ins_data.get("shared", False),
                        )
                        self.processed_insights.append(insight)
                    except Exception as e:
                        logger.debug(f"Could not load insight: {e}")
                updates = data.get("pending_evolution_updates", [])
                self.pending_evolution_updates = deque(updates[-self.MAX_PENDING_UPDATES:], maxlen=self.MAX_PENDING_UPDATES)
                self._category_effectiveness = data.get("category_effectiveness", {})
                self._strategy_effectiveness = data.get("strategy_effectiveness", {})
                for key in data.get("seen_insights_keys", [])[-self.MAX_PROCESSED_INSIGHTS:]:
                    self._seen_insights[key] = None
                logger.info(f"Loaded insight bridge state: {self._insight_counter} insights processed, {len(self.processed_insights)} stored")
            except Exception as e:
                logger.warning(f"Failed to load insight state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "insight_state.json"
        try:
            processed_data = []
            for ins in list(self.processed_insights)[-self.MAX_PROCESSED_INSIGHTS:]:
                processed_data.append({
                    "id": ins.id,
                    "raw_insight": ins.raw_insight,
                    "insight_type": ins.insight_type.value,
                    "evolution_relevance": ins.evolution_relevance.value,
                    "evolution_guidance": ins.evolution_guidance,
                    "knowledge_for_mankind": ins.knowledge_for_mankind,
                    "source_problem": ins.source_problem,
                    "source_category": ins.source_category,
                    "timestamp": ins.timestamp.isoformat() if ins.timestamp else None,
                    "shared": ins.shared,
                })
            atomic_write_json(state_file, {
                    "counter": self._insight_counter,
                    "processed_insights": processed_data,
                    "pending_evolution_updates": list(self.pending_evolution_updates),
                    "seen_insights_keys": list(self._seen_insights.keys())[-self.MAX_PROCESSED_INSIGHTS:],
                    "category_effectiveness": self._category_effectiveness,
                    "strategy_effectiveness": self._strategy_effectiveness,
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save insight state: {e}")

    def process_insight(
        self,
        raw_insight: str,
        source_problem: str,
        source_category: str,
        strategy_used: str | None = None,
        problem_solved: bool = False,
    ) -> ProcessedInsight | None:
        """Process a raw insight into evolution guidance and knowledge."""
        with self._lock:
            insight_key = f"{raw_insight[:100]}:{source_problem[:50]}"
            if insight_key in self._seen_insights:
                logger.debug(f"Skipping duplicate insight: {insight_key[:50]}")
                return None

            self._seen_insights[insight_key] = None
            while len(self._seen_insights) > self.MAX_PROCESSED_INSIGHTS:
                self._seen_insights.popitem(last=False)

            self._insight_counter += 1
            insight_id = f"insight_{self._insight_counter:06d}"

            insight_type = self._classify_insight(raw_insight, problem_solved)
            relevance = self._assess_evolution_relevance(raw_insight, source_category)
            evolution_guidance = self._extract_evolution_guidance(
                raw_insight, source_category, strategy_used, insight_type
            )
            knowledge = self._generate_knowledge(raw_insight, source_category)

            processed = ProcessedInsight(
                id=insight_id,
                raw_insight=raw_insight,
                insight_type=insight_type,
                evolution_relevance=relevance,
                evolution_guidance=evolution_guidance,
                knowledge_for_mankind=knowledge,
                source_problem=source_problem,
                source_category=source_category,
            )

            self.processed_insights.append(processed)
            self._insight_index[insight_id] = processed
            if len(self._insight_index) > self.MAX_PROCESSED_INSIGHTS:
                oldest_keys = list(self._insight_index.keys())[:len(self._insight_index) - self.MAX_PROCESSED_INSIGHTS]
                for k in oldest_keys:
                    del self._insight_index[k]

            if relevance.value >= EvolutionRelevance.MEDIUM.value:
                self.pending_evolution_updates.append({
                    "insight_id": insight_id,
                    "guidance": evolution_guidance,
                    "source": source_category,
                    "timestamp": datetime.now().isoformat(),
                })

            should_share = self._is_knowledge_worthy(processed)

            # Index for insight review
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                review.index_insight(
                    insight_id=insight_id,
                    content=raw_insight,
                    source="insight_bridge",
                    insight_type=insight_type.value if hasattr(insight_type, 'value') else str(insight_type),
                    timestamp=datetime.now().isoformat(),
                    problem=source_problem,
                    category=source_category,
                )
            except Exception as e:
                logger.debug(f"Failed to index insight for review: {e}")

        # Save state outside lock to avoid I/O under lock
        self._save_state()

        # Feed the insight to the Prefrontal Cortex so it participates in the
        # PFC learning loop. The bridge is the system's insight hub but was the
        # one major producer that never reported to the PFC. process_insight()
        # is synchronous, so schedule the async observation on the running loop
        # and skip cleanly when called from a non-async context.
        try:
            import asyncio

            from core.pfc_observer import observe_insight

            asyncio.get_running_loop()
            create_tracked_task(
                observe_insight(
                    insight=raw_insight,
                    source=f"insight_bridge:{source_category}",
                    metadata={
                        "insight_id": insight_id,
                        "insight_type": getattr(insight_type, "value", str(insight_type)),
                        "problem": source_problem,
                        "relevance": getattr(relevance, "name", str(relevance)),
                    },
                ),
                name=f"pfc_observe:{insight_id}",
            )
        except RuntimeError:
            pass  # no running event loop — nothing to schedule onto
        except Exception as e:
            logger.debug(f"PFC observation failed for {insight_id}: {e}")

        # External I/O outside lock
        if should_share:
            contribution = self._create_contribution(processed)
            with self._lock:
                self.knowledge_contributions.append(contribution)

            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                kg.add_insight(
                    insight_id=insight_id,
                    content=raw_insight,
                    problem_id=source_problem,
                    strategy=strategy_used,
                    category=source_category,
                )
            except Exception as e:
                logger.debug(f"Knowledge graph not available: {e}")

            try:
                from core.cognitive_psychology import CognitivePsychologyEngine, MentalModel, MentalModelType
                psych = CognitivePsychologyEngine.get_instance()
                psych.add_mental_model(MentalModel(
                    domain=source_category or "mathematics",
                    model_type=MentalModelType.PROBABILISTIC,
                    assumptions=[f"derived from {source_problem}"],
                    evidence_for=[raw_insight[:200]],
                    evidence_against=[],
                    predictions=[],
                ))
            except Exception:
                pass

            try:
                interconnect = self._get_interconnect()
                if interconnect:
                    interconnect.store_insight(insight_id, raw_insight)
            except Exception as e:
                logger.debug(f"Failed to store insight embedding: {e}")

            try:
                from core.discovery_notifier import get_discovery_notifier, DiscoveryType, DiscoveryPriority
                notifier = get_discovery_notifier()

                discovery_type = DiscoveryType.MATH_INSIGHT
                priority = DiscoveryPriority.NORMAL

                if processed.insight_type == InsightType.STRATEGY_SUCCESS:
                    discovery_type = DiscoveryType.PROBLEM_SOLVED
                    priority = DiscoveryPriority.INTERESTING
                elif processed.evolution_relevance == EvolutionRelevance.HIGH:
                    priority = DiscoveryPriority.INTERESTING
                elif processed.insight_type == InsightType.PATTERN_DISCOVERY:
                    discovery_type = DiscoveryType.PATTERN_DISCOVERY
                    priority = DiscoveryPriority.NORMAL

                notifier.record_discovery(
                    discovery_type=discovery_type,
                    title=source_problem[:50],
                    description=raw_insight,
                    source=f"erdos:{source_category}",
                    priority=priority,
                )
            except Exception as e:
                logger.debug(f"Discovery notifier not available: {e}")

        logger.info(
            f"Processed insight {insight_id}: type={insight_type.value}, "
            f"relevance={relevance.value}, evolution={bool(evolution_guidance)}"
        )

        return processed

    def _classify_insight(self, insight: str, solved: bool) -> InsightType:
        """Classify the type of mathematical insight."""
        insight_lower = insight.lower()

        if solved:
            return InsightType.STRATEGY_SUCCESS

        if any(kw in insight_lower for kw in ["pattern", "regular", "consistent", "always"]):
            return InsightType.PATTERN_DISCOVERY

        if any(kw in insight_lower for kw in ["counterexample", "fails when", "not true for"]):
            return InsightType.COUNTEREXAMPLE

        if any(kw in insight_lower for kw in ["bound", "at most", "at least", "upper", "lower"]):
            return InsightType.BOUND_REFINEMENT

        if any(kw in insight_lower for kw in ["conjecture", "hypothesis", "might be", "appears to"]):
            return InsightType.CONJECTURE

        if any(kw in insight_lower for kw in ["method", "approach", "technique", "strategy"]):
            return InsightType.METHOD_INSIGHT

        return InsightType.CROSS_DOMAIN

    def _assess_evolution_relevance(self, insight: str, category: str) -> EvolutionRelevance:
        """Assess how relevant this insight is for Jarvis's evolution.
        
        Uses keyword analysis combined with feedback from past evolution outcomes
        to adjust relevance scoring.
        """
        insight_lower = insight.lower()
        
        # Remove timestamp suffix if present (test artifacts)
        insight_clean = insight_lower.split('[ts=')[0].strip() if '[ts=' in insight_lower else insight_lower
        
        # High-quality indicators - require substantial insight content
        high_keywords = ["optimal", "efficient", "strategy", "pattern", "method", "discovery", "fundamental", "theorem", "proof"]
        medium_keywords = ["bound", "approach", "improved", "better", "faster", "relationship", "connection", "structure"]
        
        high_count = sum(1 for kw in high_keywords if kw in insight_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in insight_lower)
        
        # Minimum content length for meaningful insight (check cleaned version)
        min_length = 50  # Reduced from 100 to accommodate shorter but meaningful insights
        
        if len(insight_clean) < min_length:
            return EvolutionRelevance.IRRELEVANT
        
        # Determine initial relevance
        relevance = EvolutionRelevance.IRRELEVANT
        
        # High-quality insight with multiple strong indicators
        if high_count >= 2:
            relevance = EvolutionRelevance.HIGH
        elif high_count >= 1 and medium_count >= 1:
            relevance = EvolutionRelevance.HIGH
        elif high_count >= 1:
            relevance = EvolutionRelevance.MEDIUM
        elif medium_count >= 2:
            relevance = EvolutionRelevance.MEDIUM
        elif medium_count >= 1:
            relevance = EvolutionRelevance.LOW
        
        # Adjust relevance based on feedback from past evolution outcomes
        cat_key = f"cat_{category.lower().replace(' ', '_')}"
        if cat_key in self._category_effectiveness:
            stats = self._category_effectiveness[cat_key]
            total = stats["successes"] + stats["failures"]
            if total >= 3:
                success_rate = stats["successes"] / total
                if success_rate > 0.7 and relevance != EvolutionRelevance.HIGH:
                    relevance = EvolutionRelevance(min(relevance.value + 1, 3))
                elif success_rate < 0.3 and relevance != EvolutionRelevance.IRRELEVANT:
                    relevance = EvolutionRelevance(max(relevance.value - 1, 0))
        
        return relevance

    def _extract_evolution_guidance(
        self,
        insight: str,
        category: str,
        strategy: str | None,
        insight_type: InsightType,
    ) -> dict[str, Any]:
        """Extract evolution parameter adjustments from insight."""
        guidance = {"gene_adjustments": {}, "reasoning": "", "domain_hints": [], "strategy": strategy or "unknown"}

        category_lower = category.lower().replace(" ", "_")
        for pattern_key, pattern_data in self.MATHEMATICAL_PATTERNS.items():
            if pattern_key in category_lower:
                guidance["gene_adjustments"].update(pattern_data["guidance"])
                guidance["domain_hints"].append(pattern_data["evolution_domain"])
                break

        if strategy and strategy in self.STRATEGY_EVOLUTION_MAP:
            for gene, delta in self.STRATEGY_EVOLUTION_MAP[strategy].items():
                current = guidance["gene_adjustments"].get(gene, 0)
                guidance["gene_adjustments"][gene] = current + delta

        if insight_type == InsightType.PATTERN_DISCOVERY:
            guidance["gene_adjustments"]["reasoning_depth"] = guidance["gene_adjustments"].get("reasoning_depth", 0) + 0.05
        elif insight_type == InsightType.COUNTEREXAMPLE:
            guidance["gene_adjustments"]["security_sensitivity"] = guidance["gene_adjustments"].get("security_sensitivity", 0) + 0.05
        elif insight_type == InsightType.STRATEGY_SUCCESS:
            guidance["gene_adjustments"]["learning_rate"] = guidance["gene_adjustments"].get("learning_rate", 0) + 0.05

        guidance["reasoning"] = f"Insight from {category}: {insight[:100]}"

        drift = self._get_drift_detector()
        if drift is not None:
            try:
                stats = drift.get_stats()
                current_topic = stats.get("current_topic", "")
                if current_topic:
                    guidance["domain_hints"].append(f"drift_topic:{current_topic}")
                recent_drifts = stats.get("recent_drifts", 0)
                if recent_drifts > 3:
                    guidance["gene_adjustments"]["adaptation_speed"] = guidance["gene_adjustments"].get("adaptation_speed", 0) + 0.02
            except Exception as e:
                logger.debug(f"Failed to get drift detector stats for evolution guidance: {e}")

        return guidance

    def _generate_knowledge(self, insight: str, category: str) -> str:
        """Generate human-shareable knowledge from insight."""
        category_lower = category.lower().replace(" ", "_")

        template = "Mathematical insight: {insight}"
        for pattern_key, pattern_data in self.MATHEMATICAL_PATTERNS.items():
            if pattern_key in category_lower:
                template = pattern_data["knowledge_template"]
                break

        return template.format(insight=insight)

    def _is_knowledge_worthy(self, insight: ProcessedInsight) -> bool:
        """Determine if insight is worth sharing with mankind."""
        if insight.insight_type == InsightType.STRATEGY_SUCCESS:
            return True
        if insight.evolution_relevance == EvolutionRelevance.HIGH:
            return True
        if insight.insight_type in [InsightType.PATTERN_DISCOVERY, InsightType.CONJECTURE]:
            return True
        return False

    def _create_contribution(self, insight: ProcessedInsight) -> KnowledgeContribution:
        """Create a knowledge contribution record."""
        contrib_id = f"contrib_{insight.id.replace('insight_', '')}"

        title = f"{insight.insight_type.value.replace('_', ' ').title()}: {insight.source_problem[:30]}"

        return KnowledgeContribution(
            id=contrib_id,
            title=title,
            summary=insight.knowledge_for_mankind,
            mathematical_content=insight.raw_insight,
            discovered_by="Jarvis AI",
            problem_context=insight.source_problem,
        )

    def get_pending_evolution_updates(self) -> list[dict]:
        """Get and clear pending evolution updates."""
        with self._lock:
            updates = list(self.pending_evolution_updates)
            self.pending_evolution_updates.clear()
            return updates

    def report_evolution_outcome(
        self,
        insight_id: str,
        success: bool,
        details: str = "",
    ) -> None:
        """Report the outcome of applying an evolution guidance.

        This creates a feedback loop: when the evolution orchestrator
        applies guidance derived from an insight, it should report the
        result. This feedback adjusts future relevance scoring so that
        guidance types that lead to positive outcomes get boosted and
        types that lead to negative outcomes get dampened.

        Args:
            insight_id: The ID of the insight whose guidance was applied
            success: Whether the application was beneficial
            details: Optional details about the outcome
        """
        with self._lock:
            # Find the insight via index (O(1))
            insight = self._insight_index.get(insight_id)

            if insight is None:
                logger.debug(f"Evolution outcome reported for unknown insight: {insight_id}")
                return

            # Find the corresponding category pattern
            domain_hints = insight.evolution_guidance.get("domain_hints", [])
            category = domain_hints[0] if domain_hints else insight.source_category
            strategy = insight.evolution_guidance.get("strategy", "unknown")

            # Normalize category key (same as _assess_evolution_relevance)
            cat_key = f"cat_{category.lower().replace(' ', '_')}"
            if cat_key not in self._category_effectiveness:
                self._category_effectiveness[cat_key] = {"successes": 0, "failures": 0}

            if success:
                self._category_effectiveness[cat_key]["successes"] += 1
            else:
                self._category_effectiveness[cat_key]["failures"] += 1

            # Adjust strategy effectiveness scores
            strat_key = f"strat_{strategy}"
            if strat_key not in self._strategy_effectiveness:
                self._strategy_effectiveness[strat_key] = {"successes": 0, "failures": 0}

            if success:
                self._strategy_effectiveness[strat_key]["successes"] += 1
            else:
                self._strategy_effectiveness[strat_key]["failures"] += 1

            # Cap effectiveness dicts to prevent unbounded growth
            if len(self._category_effectiveness) > 50:
                low_total = {k: v["successes"] + v["failures"] for k, v in self._category_effectiveness.items()}
                for k in sorted(low_total, key=low_total.get)[:len(self._category_effectiveness) - 30]:
                    del self._category_effectiveness[k]
            if len(self._strategy_effectiveness) > 50:
                low_total = {k: v["successes"] + v["failures"] for k, v in self._strategy_effectiveness.items()}
                for k in sorted(low_total, key=low_total.get)[:len(self._strategy_effectiveness) - 30]:
                    del self._strategy_effectiveness[k]

            logger.debug(
                f"Evolution outcome for {insight_id}: {'success' if success else 'failure'} "
                f"(category: {category}, strategy: {strategy})"
            )

            # Save state periodically
            if sum(
                v["successes"] + v["failures"]
                for v in self._category_effectiveness.values()
            ) % 10 == 0:
                self._save_state()

    def get_knowledge_feed(self, limit: int = 20) -> list[dict]:
        """Get recent knowledge contributions for sharing."""
        with self._lock:
            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "summary": c.summary,
                    "discovered_by": c.discovered_by,
                    "timestamp": c.timestamp.isoformat(),
                    "verified": c.verified,
                }
                for c in list(self.knowledge_contributions)[-limit:]
            ]

    def get_insight_statistics(self) -> dict:
        """Get statistics about processed insights."""
        with self._lock:
            by_type = {}
            by_relevance = {}

            for insight in self.processed_insights:
                t = insight.insight_type.value
                by_type[t] = by_type.get(t, 0) + 1

                r = insight.evolution_relevance.value
                by_relevance[r] = by_relevance.get(r, 0) + 1

            return {
                "total_insights": len(self.processed_insights),
                "knowledge_contributions": len(self.knowledge_contributions),
                "pending_evolution_updates": len(self.pending_evolution_updates),
                "by_type": by_type,
                "by_relevance": by_relevance,
            }


_insight_bridge: InsightBridge | None = None
_insight_lock = threading.Lock()


def get_insight_bridge() -> InsightBridge:
    """Get global insight bridge instance (thread-safe)."""
    global _insight_bridge
    if _insight_bridge is None:
        with _insight_lock:
            if _insight_bridge is None:
                _insight_bridge = InsightBridge()
    return _insight_bridge
