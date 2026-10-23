"""Intuition Bridge - Connects Learning/Daydream to Auto-Refine System.

This module creates a feedback loop between:
1. Erdős Engine - Mathematical exploration and discovery
2. Daydream Engine - Creative exploration during idle
3. Learning Pipeline - Pattern recognition and improvement
4. Auto-Refine - Code analysis and repair

The intuition bridge transforms insights from learning into actionable
improvements for the codebase.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     INTUITION BRIDGE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  ERDŐS       │     │  DAYDREAM    │     │  LEARNING    │    │
│  │  ENGINE      │────▶│  ENGINE      │────▶│  PIPELINE    │    │
│  │              │     │              │     │              │    │
│  │ Math proofs  │     │ Exploration  │     │ Patterns     │    │
│  │ Strategies   │     │ Creativity   │     │ Strategies   │    │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    INSIGHT BUFFER                        │   │
│  │                                                          │   │
│  │  - Strategy successes/failures                           │   │
│  │  - Discoveries and breakthroughs                         │   │
│  │  - Performance metrics                                   │   │
│  │  - Code patterns (good/bad)                              │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 INTUITION PROCESSOR                      │   │
│  │                                                          │   │
│  │  1. Pattern Extraction - Find recurring patterns         │   │
│  │  2. Insight Scoring - Rate importance/impact            │   │
│  │  3. Action Mapping - Convert to refinement actions      │   │
│  │  4. Priority Queue - Order by potential improvement     │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  AUTO-REFINE SYSTEM                      │   │
│  │                                                          │   │
│  │  - Code improvements                                     │   │
│  │  - Performance optimizations                            │   │
│  │  - Bug fixes                                             │   │
│  │  - Architecture enhancements                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
"""
import asyncio
import json
from core.helpers import atomic_write_json, create_tracked_task, read_json_with_integrity
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class InsightType(Enum):
    STRATEGY_SUCCESS = "strategy_success"       # A strategy worked well
    STRATEGY_FAILURE = "strategy_failure"       # A strategy failed
    DISCOVERY = "discovery"                     # New discovery made
    PATTERN = "pattern"                         # Recurring pattern detected
    PERFORMANCE = "performance"                 # Performance insight
    OPTIMIZATION = "optimization"               # Optimization opportunity
    BUG_HINT = "bug_hint"                       # Potential bug detected
    ARCHITECTURE = "architecture"               # Architecture insight
    LEARNING = "learning"                       # General learning


class ActionType(Enum):
    CODE_SCAN = "code_scan"                     # Scan codebase
    REFACTOR = "refactor"                       # Refactor code
    OPTIMIZE = "optimize"                       # Optimize performance
    FIX_BUG = "fix_bug"                         # Fix a bug
    ADD_TEST = "add_test"                       # Add tests
    IMPROVE_DOCS = "improve_docs"               # Improve documentation
    ARCHITECTURE = "architecture"               # Architecture change
    CONFIGURE = "configure"                     # Configuration change


@dataclass
class IntuitionInsight:
    """An insight from the learning system."""
    id: str
    insight_type: InsightType
    source: str                    # Where it came from (erdos, daydream, learning)
    content: str                   # The insight itself
    confidence: float = 0.5        # How confident we are (0-1)
    impact: float = 0.5            # Potential impact (0-1)
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    actions_generated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "insight_type": self.insight_type.value,
            "source": self.source,
            "content": self.content,
            "confidence": self.confidence,
            "impact": self.impact,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "actions_generated": self.actions_generated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntuitionInsight":
        return cls(
            id=data["id"],
            insight_type=InsightType(data["insight_type"]),
            source=data["source"],
            content=data["content"],
            confidence=data.get("confidence", 0.5),
            impact=data.get("impact", 0.5),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            processed=data.get("processed", False),
            actions_generated=data.get("actions_generated", []),
        )


@dataclass
class RefinementAction:
    """An action derived from an insight."""
    id: str
    action_type: ActionType
    insight_id: str               # Source insight
    target: str                   # File, function, or component
    description: str              # What to do
    priority: float = 0.5         # Priority (0-1, higher = more important)
    estimated_effort: str = "medium"  # low, medium, high
    risk: str = "low"             # low, medium, high
    prerequisites: list[str] = field(default_factory=list)
    status: str = "pending"       # pending, in_progress, completed, failed
    result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "insight_id": self.insight_id,
            "target": self.target,
            "description": self.description,
            "priority": self.priority,
            "estimated_effort": self.estimated_effort,
            "risk": self.risk,
            "prerequisites": self.prerequisites,
            "status": self.status,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
        }


class IntuitionBridge:
    """Bridges learning insights to code refinement actions.
    
    Subscribes to:
    - Erdős engine discoveries
    - Daydream engine explorations
    - Learning pipeline patterns
    
    Produces:
    - Refinement actions for auto-refine system
    - Improvement suggestions
    - Bug reports
    """

    MAX_INSIGHTS = 1000
    MAX_ACTIONS = 100
    INSIGHT_EXPIRY_DAYS = 7

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "intuition"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Insight buffer
        self._insights: dict[str, IntuitionInsight] = {}
        self._insight_counter = 0

        # Action queue
        self._actions: dict[str, RefinementAction] = {}
        self._action_counter = 0

        # Pattern tracking
        self._patterns: dict[str, int] = defaultdict(int)  # pattern -> count
        self._strategies: dict[str, dict] = {}  # strategy_name -> {successes, failures}

        # Connection to subsystems
        self._erdos_engine = None
        self._daydream_engine = None
        self._learning_pipeline = None
        self._refine_manager = None

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Lock
        self._lock = threading.RLock()

        # Load state
        self._load()

        logger.info(f"IntuitionBridge initialized: {len(self._insights)} insights, {len(self._actions)} actions")

    def attach_erdos(self, erdos_engine):
        """Attach Erdős engine for insight collection."""
        self._erdos_engine = erdos_engine
        logger.info("Erdős engine attached to intuition bridge")

    def attach_daydream(self, daydream_engine):
        """Attach daydream engine for insight collection."""
        self._daydream_engine = daydream_engine
        logger.info("Daydream engine attached to intuition bridge")

    def attach_learning(self, learning_pipeline):
        """Attach learning pipeline for insight collection."""
        self._learning_pipeline = learning_pipeline
        logger.info("Learning pipeline attached to intuition bridge")

    def attach_refine_manager(self, refine_manager):
        """Attach auto-refine manager for action execution."""
        self._refine_manager = refine_manager
        logger.info("Auto-refine manager attached to intuition bridge")

    def record_insight(
        self,
        insight_type: InsightType,
        source: str,
        content: str,
        confidence: float = 0.5,
        impact: float = 0.5,
        evidence: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record a new insight from any source.
        
        Args:
            insight_type: Type of insight
            source: Where it came from (erdos, daydream, learning, etc.)
            content: The insight content
            confidence: Confidence level (0-1)
            impact: Potential impact (0-1)
            evidence: Supporting evidence
            metadata: Additional metadata
            
        Returns:
            Insight ID
        """
        with self._lock:
            self._insight_counter += 1
            insight_id = f"insight_{self._insight_counter:06d}"

            insight = IntuitionInsight(
                id=insight_id,
                insight_type=insight_type,
                source=source,
                content=content,
                confidence=confidence,
                impact=impact,
                evidence=evidence or [],
                metadata=metadata or {},
            )

            self._insights[insight_id] = insight

            # Track patterns
            self._track_pattern(insight)

            # Enforce capacity
            if len(self._insights) > self.MAX_INSIGHTS:
                self._evict_old_insights()

            logger.debug(f"Recorded insight {insight_id}: {insight_type.value} from {source}")

            return insight_id

    def _track_pattern(self, insight: IntuitionInsight):
        """Track recurring patterns in insights."""
        # Extract key terms
        terms = set(insight.content.lower().split())
        for term in terms:
            if len(term) > 4:  # Skip short terms
                self._patterns[term] += 1

    def _evict_old_insights(self):
        """Evict old, low-impact insights."""
        now = datetime.now()
        to_remove = []

        for insight_id, insight in self._insights.items():
            # Skip if already processed into actions
            if insight.actions_generated:
                continue

            age_days = (now - insight.timestamp).total_seconds() / 86400
            if age_days > self.INSIGHT_EXPIRY_DAYS and insight.impact < 0.7:
                to_remove.append(insight_id)

        for insight_id in to_remove:
            del self._insights[insight_id]

        if to_remove:
            logger.debug(f"Evicted {len(to_remove)} old insights")

    def process_insights(self) -> list[RefinementAction]:
        """Process pending insights and generate refinement actions.
        
        Returns:
            List of new actions generated
        """
        with self._lock:
            new_actions = []

            for insight_id, insight in self._insights.items():
                if insight.processed:
                    continue

                # Generate actions based on insight type
                actions = self._generate_actions(insight)

                for action in actions:
                    self._action_counter += 1
                    action.id = f"action_{self._action_counter:06d}"
                    self._actions[action.id] = action
                    insight.actions_generated.append(action.id)
                    new_actions.append(action)

                insight.processed = True

            # Enforce action capacity
            if len(self._actions) > self.MAX_ACTIONS:
                self._evict_completed_actions()

            # Save state
            self._save()

            if new_actions:
                logger.info(f"Generated {len(new_actions)} refinement actions from insights")

            return new_actions

    def _generate_actions(self, insight: IntuitionInsight) -> list[RefinementAction]:
        """Generate refinement actions from an insight."""
        actions = []

        if insight.insight_type == InsightType.STRATEGY_SUCCESS:
            # Successful strategy - consider applying to other areas
            if insight.confidence > 0.7 and insight.impact > 0.6:
                actions.append(RefinementAction(
                    id="",  # Will be assigned
                    action_type=ActionType.CODE_SCAN,
                    insight_id=insight.id,
                    target="codebase",
                    description=f"Apply successful pattern from {insight.source}: {insight.content[:100]}",
                    priority=insight.impact * 0.8,
                    estimated_effort="low",
                    risk="low",
                ))

        elif insight.insight_type == InsightType.STRATEGY_FAILURE:
            # Failed strategy - investigate and possibly fix
            if insight.confidence > 0.5:
                actions.append(RefinementAction(
                    id="",
                    action_type=ActionType.CODE_SCAN,
                    insight_id=insight.id,
                    target="relevant_modules",
                    description=f"Investigate failure pattern: {insight.content[:100]}",
                    priority=0.7,
                    estimated_effort="medium",
                    risk="medium",
                ))

        elif insight.insight_type == InsightType.DISCOVERY:
            # New discovery - may lead to improvements
            if insight.impact > 0.5:
                actions.append(RefinementAction(
                    id="",
                    action_type=ActionType.OPTIMIZE,
                    insight_id=insight.id,
                    target="discovery_target",
                    description=f"Leverage discovery: {insight.content[:100]}",
                    priority=insight.impact,
                    estimated_effort="medium",
                    risk="low",
                ))

        elif insight.insight_type == InsightType.PERFORMANCE:
            # Performance insight - optimize
            actions.append(RefinementAction(
                id="",
                action_type=ActionType.OPTIMIZE,
                insight_id=insight.id,
                target="performance_target",
                description=f"Performance improvement: {insight.content[:100]}",
                priority=insight.impact,
                estimated_effort="medium",
                risk="medium",
            ))

        elif insight.insight_type == InsightType.BUG_HINT:
            # Bug hint - investigate
            actions.append(RefinementAction(
                id="",
                action_type=ActionType.FIX_BUG,
                insight_id=insight.id,
                target="bug_location",
                description=f"Potential bug: {insight.content[:100]}",
                priority=0.8,  # Bugs are high priority
                estimated_effort="low",
                risk="low",
                prerequisites=insight.evidence[:3] if insight.evidence else [],
            ))

        elif insight.insight_type == InsightType.ARCHITECTURE:
            # Architecture insight - consider refactoring
            if insight.confidence > 0.6:
                actions.append(RefinementAction(
                    id="",
                    action_type=ActionType.ARCHITECTURE,
                    insight_id=insight.id,
                    target="architecture_target",
                    description=f"Architecture improvement: {insight.content[:100]}",
                    priority=insight.impact * 0.7,
                    estimated_effort="high",
                    risk="medium",
                ))

        elif insight.insight_type == InsightType.PATTERN:
            # Recurring pattern - may indicate good or bad practice
            if insight.impact > 0.6:
                actions.append(RefinementAction(
                    id="",
                    action_type=ActionType.REFACTOR,
                    insight_id=insight.id,
                    target="pattern_location",
                    description=f"Pattern detected: {insight.content[:100]}",
                    priority=insight.impact,
                    estimated_effort="medium",
                    risk="low",
                ))

        return actions

    def _evict_completed_actions(self):
        """Evict completed or old actions."""
        now = datetime.now()
        to_remove = []

        for action_id, action in self._actions.items():
            if action.status in ("completed", "failed"):
                age_days = (now - action.timestamp).total_seconds() / 86400
                if age_days > 3:  # Keep completed/failed for 3 days
                    to_remove.append(action_id)

        for action_id in to_remove:
            del self._actions[action_id]

    def get_pending_actions(self, limit: int = 10) -> list[RefinementAction]:
        """Get pending refinement actions, sorted by priority."""
        with self._lock:
            pending = [a for a in self._actions.values() if a.status == "pending"]
            pending.sort(key=lambda a: a.priority, reverse=True)
            return pending[:limit]

    def mark_action(self, action_id: str, status: str, result: Optional[str] = None):
        """Update action status."""
        with self._lock:
            action = self._actions.get(action_id)
            if action:
                action.status = status
                action.result = result
                self._save()

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        with self._lock:
            return {
                "total_insights": len(self._insights),
                "processed_insights": sum(1 for i in self._insights.values() if i.processed),
                "total_actions": len(self._actions),
                "pending_actions": sum(1 for a in self._actions.values() if a.status == "pending"),
                "completed_actions": sum(1 for a in self._actions.values() if a.status == "completed"),
                "failed_actions": sum(1 for a in self._actions.values() if a.status == "failed"),
                "patterns_tracked": len(self._patterns),
                "strategies_tracked": len(self._strategies),
                "by_source": self._count_by_source(),
                "by_type": self._count_by_type(),
            }

    def _count_by_source(self) -> dict:
        """Count insights by source."""
        counts = defaultdict(int)
        for insight in self._insights.values():
            counts[insight.source] += 1
        return dict(counts)

    def _count_by_type(self) -> dict:
        """Count insights by type."""
        counts = defaultdict(int)
        for insight in self._insights.values():
            counts[insight.insight_type.value] += 1
        return dict(counts)

    def _load(self):
        """Load state from storage."""
        state_file = self.storage_path / "intuition_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)

                for insight_data in data.get("insights", []):
                    try:
                        insight = IntuitionInsight.from_dict(insight_data)
                        self._insights[insight.id] = insight
                        try:
                            counter_part = insight.id.split("_")[1]
                            self._insight_counter = max(self._insight_counter, int(counter_part))
                        except (IndexError, ValueError):
                            pass
                    except Exception as e:
                        logger.debug(f"Failed to load insight: {e}")

                for action_data in data.get("actions", []):
                    try:
                        action = RefinementAction(
                            id=action_data["id"],
                            action_type=ActionType(action_data["action_type"]),
                            insight_id=action_data["insight_id"],
                            target=action_data["target"],
                            description=action_data["description"],
                            priority=action_data.get("priority", 0.5),
                            estimated_effort=action_data.get("estimated_effort", "medium"),
                            risk=action_data.get("risk", "low"),
                            prerequisites=action_data.get("prerequisites", []),
                            status=action_data.get("status", "pending"),
                            result=action_data.get("result"),
                            timestamp=datetime.fromisoformat(action_data["timestamp"]) if action_data.get("timestamp") else datetime.now(),
                        )
                        self._actions[action.id] = action
                        try:
                            counter_part = action.id.split("_")[1]
                            self._action_counter = max(self._action_counter, int(counter_part))
                        except (IndexError, ValueError):
                            pass
                    except Exception as e:
                        logger.debug(f"Failed to load action: {e}")

                self._patterns = defaultdict(int, data.get("patterns", {}))
                self._strategies = data.get("strategies", {})

                logger.info(f"Loaded {len(self._insights)} insights, {len(self._actions)} actions")

            except Exception as e:
                logger.error(f"Failed to load intuition state: {e}")

    def _save(self):
        """Save state to storage."""
        state_file = self.storage_path / "intuition_state.json"
        try:
            data = {
                "insights": [i.to_dict() for i in self._insights.values()],
                "actions": [a.to_dict() for a in self._actions.values()],
                "patterns": dict(self._patterns),
                "strategies": self._strategies,
                "saved_at": datetime.now().isoformat(),
            }
            atomic_write_json(state_file, data)
        except Exception as e:
            logger.error(f"Failed to save intuition state: {e}")

    async def start(self):
        """Start the intuition bridge background task."""
        if self._running:
            return

        self._running = True
        self._task = create_tracked_task(self._run_loop(), name="intuition_bridge_loop")
        logger.info("Intuition bridge started")

    async def stop(self):
        """Stop the intuition bridge."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save()
        logger.info("Intuition bridge stopped")

    async def _run_loop(self):
        """Background loop for processing insights."""
        while self._running:
            try:
                # Collect insights from subsystems
                await self._collect_insights()

                # Process insights into actions
                self.process_insights()

                # Execute high-priority actions if auto-refine is attached
                if self._refine_manager:
                    await self._execute_actions()

                # Save state periodically
                self._save()

                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Intuition bridge error: {e}")
                await asyncio.sleep(10)

    async def _collect_insights(self):
        """Collect insights from attached subsystems."""
        # Collect from Erdős engine
        if self._erdos_engine:
            try:
                state = getattr(self._erdos_engine, '_state', {})
                learning_state = state.get('learning_state', {})

                # Get recent solves
                recent_solves = learning_state.get('recent_solves', [])
                for solve in recent_solves[-5:]:  # Last 5 solves
                    self.record_insight(
                        insight_type=InsightType.STRATEGY_SUCCESS,
                        source="erdos",
                        content=f"Solved {solve.get('category', 'unknown')}: {solve.get('description', '')[:200]}",
                        confidence=0.8,
                        impact=0.6,
                        metadata=solve,
                    )
            except Exception as e:
                logger.debug(f"Failed to collect from Erdős: {e}")

        # Collect from learning pipeline
        if self._learning_pipeline:
            try:
                metrics = getattr(self._learning_pipeline, 'metrics', None)
                if metrics:
                    # Check for learning improvements
                    if hasattr(metrics, 'defense_improvements') and metrics.defense_improvements > 0:
                        self.record_insight(
                            insight_type=InsightType.LEARNING,
                            source="learning",
                            content=f"Defense improved {metrics.defense_improvements} times",
                            confidence=0.7,
                            impact=0.5,
                        )
            except Exception as e:
                logger.debug(f"Failed to collect from learning: {e}")

    async def _execute_actions(self):
        """Execute pending high-priority actions."""
        pending = self.get_pending_actions(limit=5)

        for action in pending:
            if action.priority < 0.6:  # Only auto-execute high-priority
                continue

            if action.status != "pending":
                continue

            try:
                # Mark as in progress
                self.mark_action(action.id, "in_progress")

                # Execute based on type
                if action.action_type == ActionType.CODE_SCAN:
                    result = await self._execute_code_scan(action)
                elif action.action_type == ActionType.OPTIMIZE:
                    result = await self._execute_optimize(action)
                else:
                    result = "Action type not implemented for auto-execution"

                self.mark_action(action.id, "completed", result)

            except Exception as e:
                self.mark_action(action.id, "failed", str(e))
                logger.error(f"Failed to execute action {action.id}: {e}")

    async def _execute_code_scan(self, action: RefinementAction) -> str:
        """Execute a code scan action."""
        if not self._refine_manager:
            return "No refine manager attached"

        try:

            # Run scan
            result = await self._refine_manager.scan_codebase(include_ai=False)

            # Get summary
            summary = {
                "files_scanned": result.files_scanned,
                "issues_found": len(result.issues_found),
                "by_severity": {},
            }

            for issue in result.issues_found:
                sev = issue.severity.value
                summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

            return json.dumps(summary, indent=2)

        except Exception as e:
            return f"Scan failed: {e}"

    async def _execute_optimize(self, action: RefinementAction) -> str:
        """Execute an optimization action."""
        # For now, just log it
        logger.info(f"Optimization action: {action.description}")
        return f"Logged optimization suggestion: {action.description}"


# Singleton
_intuition_bridge: Optional[IntuitionBridge] = None
_bridge_lock = threading.Lock()


def get_intuition_bridge() -> IntuitionBridge:
    """Get the intuition bridge singleton."""
    global _intuition_bridge
    if _intuition_bridge is None:
        with _bridge_lock:
            if _intuition_bridge is None:
                _intuition_bridge = IntuitionBridge()
    return _intuition_bridge
