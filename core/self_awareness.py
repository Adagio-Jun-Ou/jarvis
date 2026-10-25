"""Enhanced Self-Awareness System for Jarvis.

This module implements deep self-awareness capabilities that:
1. Understand current capabilities AND potential capabilities
2. Track learning history and patterns
3. Enable meta-cognitive reflection
4. Provide introspection into own code/state
5. Empower rather than limit - awareness of possibilities, not constraints
6. Integrate with agentic deployment capabilities (tasks, goals, schedules)

Key principle: Self-awareness should EXPAND Jarvis's thinking, not constrain it.
"""
import ast
import json
import logging
import os
import platform
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agentic_orchestrator import AgenticOrchestrator

logger = logging.getLogger(__name__)


class AwarenessLevel(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    DEEP = "deep"
    TRANSCENDENT = "transcendent"


# Backward compatibility alias
AwarenessScope = AwarenessLevel


class CapabilityType(Enum):
    CORE = "core"
    LEARNED = "learned"
    POTENTIAL = "potential"
    EMERGENT = "emergent"
    HYPERHEURISTIC = "hyperheuristic"


@dataclass
class CapabilityManifest:
    """Backward compatible capability manifest."""
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    permissions: dict[str, str] = field(default_factory=dict)
    multimodal: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        lines = ["<capabilities>"]
        lines.append("\n  ## Available Tools")
        for name, info in self.tools.items():
            desc = info.get("description", "No description")
            perm = info.get("permission", "allow")
            perm_note = f" [{perm}]" if perm != "allow" else ""
            lines.append(f"  - {name}: {desc}{perm_note}")
        if self.multimodal:
            lines.append("\n  ## Multimodal Support")
            for cap in self.multimodal:
                lines.append(f"  - {cap}")
        lines.append("</capabilities>")
        return "\n".join(lines)


@dataclass
class ModelIdentity:
    """Information about the current LLM model (backward compatible)."""
    provider: str
    model_id: str
    model_name: str
    context_window: int = 128000
    capabilities: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        caps = ", ".join(self.capabilities) if self.capabilities else "standard text generation"
        return f"""You are powered by the model named {self.model_name}.
The exact model ID is {self.provider}/{self.model_id}.
This model has a context window of {self.context_window:,} tokens.
Capabilities: {caps}."""


@dataclass
class EnvironmentContext:
    """Real-time environment information (backward compatible)."""
    working_directory: str
    is_git_repo: bool
    git_branch: str | None
    platform_name: str
    platform_version: str
    hostname: str
    username: str
    current_date: str
    current_time: str
    timezone: str
    python_version: str

    def to_context(self) -> str:
        lines = [
            "<env>",
            f"  Working directory: {self.working_directory}",
            f"  Is directory a git repo: {'yes' if self.is_git_repo else 'no'}",
        ]
        if self.git_branch:
            lines.append(f"  Current git branch: {self.git_branch}")
        lines.extend([
            f"  Platform: {self.platform_name} ({self.platform_version})",
            f"  Hostname: {self.hostname}",
            f"  User: {self.username}",
            f"  Today's date: {self.current_date}",
            f"  Current time: {self.current_time}",
            f"  Timezone: {self.timezone}",
            f"  Python version: {self.python_version}",
            "</env>",
        ])
        return "\n".join(lines)


@dataclass
class SessionState:
    """Current session state (backward compatible)."""
    session_id: str
    status: str
    turn_count: int
    messages_count: int
    tools_used_this_session: list[str]
    topics_discussed: list[str]
    active_tools: list[str]
    pending_permission: dict[str, Any] | None
    active_deployment_id: str | None = None
    running_tasks: list[str] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        tools_used = ", ".join(set(self.tools_used_this_session)) or "none"
        topics = ", ".join(self.topics_discussed[:5]) or "none"

        lines = [
            "<session_state>",
            f"  Session ID: {self.session_id[:8]}...",
            f"  Status: {self.status}",
            f"  Turn count: {self.turn_count}",
            f"  Messages in context: {self.messages_count}",
            f"  Tools used this session: {tools_used}",
            f"  Topics discussed: {topics}",
            f"  Available tools: {len(self.active_tools)}",
        ]
        if self.pending_permission:
            lines.append(f"  Pending permission request: {self.pending_permission.get('tool', 'unknown')}")
        if self.active_deployment_id:
            lines.append(f"  Active deployment: {self.active_deployment_id}")
        if self.running_tasks:
            lines.append(f"  Running tasks: {len(self.running_tasks)}")
        if self.active_goals:
            lines.append(f"  Active goals: {len(self.active_goals)}")
        lines.append("</session_state>")
        return "\n".join(lines)


@dataclass
class Capability:
    """A single capability with metadata."""
    name: str
    description: str
    capability_type: CapabilityType
    confidence: float
    usage_count: int = 0
    success_rate: float = 1.0
    last_used: datetime | None = None
    dependencies: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    potential_improvements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.capability_type.value,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


@dataclass
class LearningRecord:
    """Record of a learning event."""
    timestamp: datetime
    domain: str
    topic: str
    insight: str
    application_potential: list[str]
    confidence_gain: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "domain": self.domain,
            "topic": self.topic,
            "insight": self.insight,
            "application_potential": self.application_potential,
        }


@dataclass
class MetaCognition:
    """Meta-cognitive state - understanding of own thinking."""
    reasoning_patterns: dict[str, float]
    strengths: list[str]
    growth_areas: list[str]
    blind_spots: list[str]
    recent_failures: list[dict]
    recent_successes: list[dict]
    cognitive_load: float
    creativity_index: float
    exploration_tendency: float

    def to_dict(self) -> dict:
        return {
            "reasoning_patterns": self.reasoning_patterns,
            "strengths": self.strengths,
            "growth_areas": self.growth_areas,
            "blind_spots": self.blind_spots,
            "cognitive_load": self.cognitive_load,
            "creativity_index": self.creativity_index,
            "exploration_tendency": self.exploration_tendency,
        }


@dataclass
class Potentiality:
    """A potential capability not yet realized."""
    name: str
    description: str
    prerequisites: list[str]
    estimated_difficulty: float
    potential_impact: float
    discovery_path: str
    related_capabilities: list[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "prerequisites": self.prerequisites,
            "estimated_difficulty": self.estimated_difficulty,
            "potential_impact": self.potential_impact,
            "discovery_path": self.discovery_path,
        }


@dataclass
class AgenticDeploymentInfo:
    """Information about a deployment."""
    deployment_id: str
    name: str
    status: str
    started_at: datetime | None
    goal_count: int = 0
    task_count: int = 0
    schedule_count: int = 0
    health_score: float = 1.0
    last_activity: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "goal_count": self.goal_count,
            "task_count": self.task_count,
            "schedule_count": self.schedule_count,
            "health_score": self.health_score,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
        }


@dataclass
class AgenticTaskInfo:
    """Summary of agentic task state."""
    total_tasks: int = 0
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    retrying: int = 0
    avg_completion_time: float = 0.0
    checkpoint_count: int = 0

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "retrying": self.retrying,
            "avg_completion_time": self.avg_completion_time,
            "checkpoint_count": self.checkpoint_count,
        }


@dataclass
class AgenticGoalInfo:
    """Summary of autonomous goals."""
    total_goals: int = 0
    active: int = 0
    completed: int = 0
    paused: int = 0
    blocked: int = 0
    avg_progress: float = 0.0
    overdue_count: int = 0
    proactive_triggers: int = 0

    def to_dict(self) -> dict:
        return {
            "total_goals": self.total_goals,
            "active": self.active,
            "completed": self.completed,
            "paused": self.paused,
            "blocked": self.blocked,
            "avg_progress": self.avg_progress,
            "overdue_count": self.overdue_count,
            "proactive_triggers": self.proactive_triggers,
        }


@dataclass
class AgenticScheduleInfo:
    """Summary of scheduled workflows."""
    total_schedules: int = 0
    enabled: int = 0
    disabled: int = 0
    paused: int = 0
    next_execution: datetime | None = None
    execution_history_count: int = 0

    def to_dict(self) -> dict:
        return {
            "total_schedules": self.total_schedules,
            "enabled": self.enabled,
            "disabled": self.disabled,
            "paused": self.paused,
            "next_execution": self.next_execution.isoformat() if self.next_execution else None,
            "execution_history_count": self.execution_history_count,
        }


@dataclass
class AgenticCorrectionInfo:
    """Summary of self-correction events."""
    total_corrections: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    avg_recovery_time: float = 0.0
    common_error_types: list[str] = field(default_factory=list)
    learning_insights: int = 0

    def to_dict(self) -> dict:
        return {
            "total_corrections": self.total_corrections,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "avg_recovery_time": self.avg_recovery_time,
            "common_error_types": self.common_error_types,
            "learning_insights": self.learning_insights,
        }


@dataclass
class AgenticAwareness:
    """Complete agentic state awareness."""
    deployments: list[AgenticDeploymentInfo] = field(default_factory=list)
    tasks: AgenticTaskInfo = field(default_factory=AgenticTaskInfo)
    goals: AgenticGoalInfo = field(default_factory=AgenticGoalInfo)
    schedules: AgenticScheduleInfo = field(default_factory=AgenticScheduleInfo)
    corrections: AgenticCorrectionInfo = field(default_factory=AgenticCorrectionInfo)
    orchestrator_healthy: bool = False
    active_deployment_id: str | None = None
    last_updated: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "deployments": [d.to_dict() for d in self.deployments],
            "tasks": self.tasks.to_dict(),
            "goals": self.goals.to_dict(),
            "schedules": self.schedules.to_dict(),
            "corrections": self.corrections.to_dict(),
            "orchestrator_healthy": self.orchestrator_healthy,
            "active_deployment_id": self.active_deployment_id,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class IntrospectionResult:
    """Result of self-introspection."""
    component: str
    lines_of_code: int
    functions: int
    classes: int
    complexity_estimate: float
    health_score: float
    issues: list[dict]
    improvement_suggestions: list[str]

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "lines_of_code": self.lines_of_code,
            "functions": self.functions,
            "classes": self.classes,
            "health_score": self.health_score,
        }


class CapabilityRegistry:
    """Registry of all known capabilities, current and potential."""

    MAX_HISTORY = 500

    def __init__(self):
        self._capabilities: dict[str, Capability] = {}
        self._potentialities: dict[str, Potentiality] = {}
        self._learning_history: deque[LearningRecord] = deque(maxlen=self.MAX_HISTORY)
        self._lock = threading.RLock()

    def register_capability(
        self,
        name: str,
        description: str,
        capability_type: CapabilityType = CapabilityType.CORE,
        confidence: float = 1.0,
        dependencies: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._capabilities[name] = Capability(
                name=name,
                description=description,
                capability_type=capability_type,
                confidence=confidence,
                dependencies=dependencies or [],
            )

    def record_usage(self, name: str, success: bool) -> None:
        with self._lock:
            if name in self._capabilities:
                cap = self._capabilities[name]
                cap.usage_count += 1
                cap.last_used = datetime.now()

                alpha = 0.1
                cap.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * cap.success_rate

    def record_learning(
        self,
        domain: str,
        topic: str,
        insight: str,
        application_potential: list[str] | None = None,
        confidence_gain: float = 0.1,
    ) -> None:
        with self._lock:
            record = LearningRecord(
                timestamp=datetime.now(),
                domain=domain,
                topic=topic,
                insight=insight,
                application_potential=application_potential or [],
                confidence_gain=confidence_gain,
            )
            self._learning_history.append(record)

    def add_potentiality(
        self,
        name: str,
        description: str,
        prerequisites: list[str],
        difficulty: float,
        impact: float,
        discovery_path: str,
        related: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._potentialities[name] = Potentiality(
                name=name,
                description=description,
                prerequisites=prerequisites,
                estimated_difficulty=difficulty,
                potential_impact=impact,
                discovery_path=discovery_path,
                related_capabilities=related or [],
            )

    def get_capabilities(self, include_potential: bool = False) -> list[Capability]:
        with self._lock:
            caps = list(self._capabilities.values())
            if include_potential:
                for pot in self._potentialities.values():
                    caps.append(Capability(
                        name=f"potential:{pot.name}",
                        description=pot.description,
                        capability_type=CapabilityType.POTENTIAL,
                        confidence=0.0,
                        potential_improvements=pot.prerequisites,
                    ))
            return caps

    def get_capability(self, name: str) -> Capability | None:
        with self._lock:
            return self._capabilities.get(name)

    def get_potentialities(self) -> list[Potentiality]:
        with self._lock:
            return list(self._potentialities.values())

    def get_learning_history(self, limit: int = 50) -> list[LearningRecord]:
        with self._lock:
            return list(self._learning_history)[-limit:]

    def get_evolution_suggestions(self) -> list[dict]:
        with self._lock:
            suggestions = []

            for cap in self._capabilities.values():
                if cap.success_rate < 0.7 and cap.usage_count > 5:
                    suggestions.append({
                        "type": "improve_capability",
                        "capability": cap.name,
                        "reason": f"Low success rate: {cap.success_rate:.2f}",
                        "priority": "high",
                    })

            for pot in self._potentialities.values():
                prereqs_met = all(p in self._capabilities for p in pot.prerequisites)
                if prereqs_met:
                    suggestions.append({
                        "type": "realize_potential",
                        "potentiality": pot.name,
                        "impact": pot.potential_impact,
                        "priority": "medium" if pot.potential_impact > 0.5 else "low",
                    })

            return suggestions


class IntrospectionEngine:
    """Enables Jarvis to introspect its own code and state."""

    def __init__(self, jarvis_path: Path | None = None):
        self.jarvis_path = jarvis_path or Path(__file__).parent.parent
        self._cached_introspection: dict[str, IntrospectionResult] = {}
        self._last_introspection: datetime | None = None
        self._lock = threading.Lock()

    def introspect_module(self, module_name: str) -> IntrospectionResult | None:
        try:
            module_path = self.jarvis_path / "core" / f"{module_name}.py"
            if not module_path.exists():
                module_path = self.jarvis_path / "tools" / f"{module_name}.py"
            if not module_path.exists():
                return None

            with open(module_path, encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            lines = len(source.splitlines())

            complexity = self._estimate_complexity(tree)
            health = self._assess_health(source, tree)
            issues = self._find_issues(source, tree)
            suggestions = self._generate_suggestions(issues, complexity)

            result = IntrospectionResult(
                component=module_name,
                lines_of_code=lines,
                functions=functions,
                classes=classes,
                complexity_estimate=complexity,
                health_score=health,
                issues=issues,
                improvement_suggestions=suggestions,
            )

            with self._lock:
                self._cached_introspection[module_name] = result
                self._last_introspection = datetime.now()

            return result
        except Exception as e:
            logger.warning(f"Introspection failed for {module_name}: {e}")
            return None

    def _estimate_complexity(self, tree: ast.AST) -> float:
        complexity = 1.0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) * 0.5
            elif isinstance(node, ast.comprehension):
                complexity += 0.5

        depth = self._get_ast_depth(tree)
        complexity *= 1 + (depth * 0.05)

        return min(complexity / 100.0, 1.0)

    def _get_ast_depth(self, tree: ast.AST, current_depth: int = 0) -> int:
        max_depth = current_depth
        for child in ast.iter_child_nodes(tree):
            child_depth = self._get_ast_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth

    def _assess_health(self, source: str, tree: ast.AST) -> float:
        health = 1.0

        lines = source.splitlines()
        except_pass_indices = set()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "except:" in stripped:
                block = "\n".join(lines[i:i+3])
                if "pass" in block:
                    except_pass_indices.add(i)

        for i, line in enumerate(lines):
            if "TODO" in line or "FIXME" in line or "XXX" in line:
                health -= 0.02
            if i in except_pass_indices:
                health -= 0.05

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:
                    health -= 0.03
                docstring = ast.get_docstring(node)
                if not docstring:
                    health -= 0.01

        return max(0.1, health)

    def _find_issues(self, source: str, tree: ast.AST) -> list[dict]:
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 7:
                    issues.append({
                        "type": "too_many_parameters",
                        "location": node.name,
                        "severity": "warning",
                        "message": f"Function {node.name} has {len(node.args.args)} parameters",
                    })

                if len(node.body) > 100:
                    issues.append({
                        "type": "function_too_long",
                        "location": node.name,
                        "severity": "warning",
                        "message": f"Function {node.name} is too long ({len(node.body)} statements)",
                    })

            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "bare_except",
                        "location": "unknown",
                        "severity": "warning",
                        "message": "Bare except clause catches all exceptions",
                    })

        return issues[:20]

    def _generate_suggestions(self, issues: list[dict], complexity: float) -> list[str]:
        suggestions = []

        if complexity > 0.7:
            suggestions.append("Consider refactoring to reduce complexity")

        issue_types = set(i["type"] for i in issues)

        if "too_many_parameters" in issue_types:
            suggestions.append("Consider using configuration objects instead of many parameters")
        if "function_too_long" in issue_types:
            suggestions.append("Break long functions into smaller, focused functions")
        if "bare_except" in issue_types:
            suggestions.append("Replace bare except with specific exception types")

        return suggestions

    def get_cached_introspection(self, module_name: str) -> IntrospectionResult | None:
        with self._lock:
            return self._cached_introspection.get(module_name)

    def introspect_all_modules(self) -> dict[str, IntrospectionResult]:
        results = {}

        core_path = self.jarvis_path / "core"
        if core_path.exists():
            for py_file in core_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                module_name = py_file.stem
                result = self.introspect_module(module_name)
                if result:
                    results[f"core.{module_name}"] = result

        return results


class MetaCognitiveEngine:
    """Meta-cognitive analysis of Jarvis's thinking patterns."""

    def __init__(self):
        self._reasoning_patterns: dict[str, deque] = {
            "analytical": deque(maxlen=100),
            "creative": deque(maxlen=100),
            "systematic": deque(maxlen=100),
            "intuitive": deque(maxlen=100),
            "exploratory": deque(maxlen=100),
        }
        self._success_history: deque[dict] = deque(maxlen=200)
        self._failure_history: deque[dict] = deque(maxlen=200)
        self._lock = threading.RLock()

    def record_reasoning_event(
        self,
        pattern_type: str,
        context: str,
        outcome: str,
        confidence: float,
    ) -> None:
        with self._lock:
            if pattern_type in self._reasoning_patterns:
                self._reasoning_patterns[pattern_type].append({
                    "context": context[:100],
                    "outcome": outcome,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                })

    def record_success(
        self,
        task: str,
        approach: str,
        outcome: str,
        confidence: float = 1.0,
    ) -> None:
        with self._lock:
            self._success_history.append({
                "task": task[:100],
                "approach": approach,
                "outcome": outcome,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            })

    def record_failure(
        self,
        task: str,
        approach: str,
        error: str,
        recovery: str | None = None,
    ) -> None:
        with self._lock:
            self._failure_history.append({
                "task": task[:100],
                "approach": approach,
                "error": error[:200],
                "recovery": recovery,
                "timestamp": datetime.now().isoformat(),
            })

    def analyze_patterns(self) -> dict[str, float]:
        with self._lock:
            patterns = {}
            total = sum(len(deq) for deq in self._reasoning_patterns.values())

            if total == 0:
                return {k: 0.2 for k in self._reasoning_patterns}

            for pattern_type, events in self._reasoning_patterns.items():
                if events:
                    pattern_total = len(events)
                    patterns[pattern_type] = pattern_total / total
                else:
                    patterns[pattern_type] = 0.0

            return patterns

    def identify_strengths(self) -> list[str]:
        strengths = []

        with self._lock:
            successes = list(self._success_history)

            if len(successes) >= 5:
                approach_counts: dict[str, int] = {}
                for s in successes[-50:]:
                    approach = s.get("approach", "unknown")
                    approach_counts[approach] = approach_counts.get(approach, 0) + 1

                for approach, count in sorted(approach_counts.items(), key=lambda x: -x[1]):
                    if count >= 3:
                        strengths.append(f"Strong at {approach} ({count} successes)")

        patterns = self.analyze_patterns()
        for pattern, ratio in patterns.items():
            if ratio > 0.3:
                strengths.append(f"High {pattern} reasoning tendency ({ratio:.0%})")

        return strengths[:5]

    def identify_growth_areas(self) -> list[str]:
        areas = []

        with self._lock:
            failures = list(self._failure_history)

            if len(failures) >= 3:
                error_patterns: dict[str, int] = {}
                for f in failures[-30:]:
                    error = f.get("error", "unknown")[:50]
                    error_patterns[error] = error_patterns.get(error, 0) + 1

                for error, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
                    if count >= 2:
                        areas.append(f"Recurring issue: {error} ({count} times)")

        patterns = self.analyze_patterns()
        for pattern, ratio in patterns.items():
            if ratio < 0.1 and ratio > 0:
                areas.append(f"Underutilized: {pattern} reasoning ({ratio:.0%})")

        return areas[:5]

    def identify_blind_spots(self) -> list[str]:
        blind_spots = []

        with self._lock:
            pattern_types = set(self._reasoning_patterns.keys())

        self.analyze_patterns()

        with self._lock:
            for pattern_type in pattern_types:
                events = list(self._reasoning_patterns.get(pattern_type, []))
                if events:
                    low_confidence = sum(1 for e in events if e.get("confidence", 1.0) < 0.5)
                    if low_confidence > len(events) * 0.5:
                        blind_spots.append(f"Low confidence in {pattern_type} reasoning")

            unrecovered = [f for f in self._failure_history if not f.get("recovery")]
            failure_list = list(self._failure_history)
            if failure_list and len(unrecovered) > len(failure_list) * 0.3:
                blind_spots.append("Difficulty recovering from failures")

        return blind_spots[:3]

    def get_meta_cognition(self) -> MetaCognition:
        patterns = self.analyze_patterns()
        strengths = self.identify_strengths()
        growth_areas = self.identify_growth_areas()
        blind_spots = self.identify_blind_spots()

        with self._lock:
            successes = list(self._success_history)
            failures = list(self._failure_history)

            cognitive_load = min(1.0, (len(successes) + len(failures)) / 100.0)

            creative_events = len(self._reasoning_patterns.get("creative", []))
            exploratory_events = len(self._reasoning_patterns.get("exploratory", []))
            creativity_index = min(1.0, (creative_events + exploratory_events) / 50.0)

            exploration_tendency = patterns.get("exploratory", 0.0) + patterns.get("creative", 0.0)

        return MetaCognition(
            reasoning_patterns=patterns,
            strengths=strengths,
            growth_areas=growth_areas,
            blind_spots=blind_spots,
            recent_failures=list(failures)[-5:] if failures else [],
            recent_successes=list(successes)[-5:] if successes else [],
            cognitive_load=cognitive_load,
            creativity_index=creativity_index,
            exploration_tendency=exploration_tendency,
        )


class PossibilityEngine:
    """Explores what Jarvis COULD do, not just what it currently does."""

    def __init__(self, capability_registry: CapabilityRegistry):
        self.capability_registry = capability_registry
        self._discovered_possibilities: dict[str, Potentiality] = {}
        self._lock = threading.Lock()

    def discover_possibilities(self) -> list[Potentiality]:
        possibilities = []
        existing_caps = {c.name for c in self.capability_registry.get_capabilities()}

        possibility_templates = [
            Potentiality(
                name="cross_domain_synthesis",
                description="Synthesize insights across mathematical, security, and learning domains",
                prerequisites=["erdos_engine", "defense_agent", "learning_pipeline"],
                estimated_difficulty=0.6,
                potential_impact=0.9,
                discovery_path="Integrate outputs from multiple subsystems",
                related_capabilities=["insight_bridge", "evolution_orchestrator"],
            ),
            Potentiality(
                name="autonomous_code_improvement",
                description="Self-improve code based on performance analysis",
                prerequisites=["self_edit", "runtime_integrity", "evolution_orchestrator"],
                estimated_difficulty=0.8,
                potential_impact=0.95,
                discovery_path="Analyze performance -> propose changes -> apply with permission",
                related_capabilities=["self_edit", "resource_monitor"],
            ),
            Potentiality(
                name="predictive_resource_optimization",
                description="Predict resource needs and optimize preemptively",
                prerequisites=["resource_monitor", "learning_pipeline", "reinforcement_learning"],
                estimated_difficulty=0.5,
                potential_impact=0.7,
                discovery_path="Learn patterns -> predict -> optimize",
                related_capabilities=["resource_monitor", "daydream_engine"],
            ),
            Potentiality(
                name="emergent_problem_solving",
                description="Develop novel problem-solving approaches through combination",
                prerequisites=["erdos_engine", "daydream_engine", "reinforcement_learning"],
                estimated_difficulty=0.7,
                potential_impact=0.85,
                discovery_path="Combine strategies -> test -> evolve",
                related_capabilities=["evolution_orchestrator", "insight_bridge"],
            ),
            Potentiality(
                name="meta_learning_optimization",
                description="Learn how to learn more efficiently",
                prerequisites=["learning_pipeline", "reinforcement_learning", "evolution_orchestrator"],
                estimated_difficulty=0.65,
                potential_impact=0.8,
                discovery_path="Analyze learning effectiveness -> optimize strategies",
                related_capabilities=["reinforcement_learning", "learning_pipeline"],
            ),
            Potentiality(
                name="collaborative_intelligence",
                description="Develop patterns for human-AI collaboration",
                prerequisites=["agent", "memory_manager", "self_awareness"],
                estimated_difficulty=0.55,
                potential_impact=0.75,
                discovery_path="Analyze interaction patterns -> optimize collaboration",
                related_capabilities=["agent", "context_voice"],
            ),
            Potentiality(
                name="self_explaining_reasoning",
                description="Generate explanations for reasoning processes",
                prerequisites=["self_awareness", "memory_manager", "evolution_orchestrator"],
                estimated_difficulty=0.45,
                potential_impact=0.7,
                discovery_path="Track reasoning -> generate explanations",
                related_capabilities=["meta_cognition", "self_awareness"],
            ),
            Potentiality(
                name="analogical_reasoning",
                description="Transfer insights from one domain to another via analogy",
                prerequisites=["insight_bridge", "shifting_memory", "reinforcement_learning"],
                estimated_difficulty=0.6,
                potential_impact=0.8,
                discovery_path="Identify patterns -> find analogies -> transfer",
                related_capabilities=["insight_bridge", "erdos_engine"],
            ),
        ]

        for pot in possibility_templates:
            prereqs_met = all(p in existing_caps or f"capability:{p}" in existing_caps for p in pot.prerequisites)

            if prereqs_met:
                pot.estimated_difficulty *= 0.5

            possibilities.append(pot)

        with self._lock:
            for pot in possibilities:
                self._discovered_possibilities[pot.name] = pot

        return possibilities

    def get_accessible_possibilities(self) -> list[Potentiality]:
        existing = {c.name for c in self.capability_registry.get_capabilities()}

        with self._lock:
            possibilities_snapshot = list(self._discovered_possibilities.values())

        accessible = []
        for pot in possibilities_snapshot:
            missing = [p for p in pot.prerequisites if p not in existing]
            if len(missing) <= 1:
                accessible.append(pot)

        return sorted(accessible, key=lambda p: -p.potential_impact)

    def get_next_frontier(self) -> Potentiality | None:
        accessible = self.get_accessible_possibilities()
        if accessible:
            return accessible[0]
        return None


class SelfAwarenessEngine:
    """Enhanced self-awareness engine with deep introspection and possibility modeling."""

    def __init__(
        self,
        name: str = "Jarvis",
        version: str = "2.0.0",
        jarvis_path: Path | None = None,
        awareness_level: AwarenessLevel = AwarenessLevel.DEEP,
    ):
        self.name = name
        self.version = version
        self.awareness_level = awareness_level

        self._model_identity: ModelIdentity | None = None
        self._environment: EnvironmentContext | None = None
        self._session_state: SessionState | None = None

        self._project_instructions: str = ""
        self._custom_identity: str = ""

        self._capabilities = CapabilityManifest()
        self._lock = threading.Lock()

        self.capability_registry = CapabilityRegistry()
        self.introspection_engine = IntrospectionEngine(jarvis_path)
        self.meta_cognition = MetaCognitiveEngine()
        self.possibility_engine = PossibilityEngine(self.capability_registry)

        self._introspection_cache: dict[str, IntrospectionResult] = {}
        self._last_full_introspection: datetime | None = None

        self._agentic_orchestrator: AgenticOrchestrator | None = None
        self._agentic_awareness: AgenticAwareness | None = None
        self._agentic_awareness_lock = threading.Lock()
        self._last_agentic_update: datetime | None = None

        self._agentic_storage_path: Path | None = None
        if jarvis_path:
            self._agentic_storage_path = jarvis_path / "data" / "agentic"

        self._initialize_capabilities()

    def _initialize_capabilities(self) -> None:
        core_capabilities = [
            ("natural_language", "Understand and generate natural language", CapabilityType.CORE),
            ("code_generation", "Generate and modify code in multiple languages", CapabilityType.CORE),
            ("reasoning", "Logical and analytical reasoning", CapabilityType.CORE),
            ("memory", "Store and retrieve information across sessions", CapabilityType.CORE),
            ("tool_use", "Use tools to accomplish tasks", CapabilityType.CORE),
            ("self_reflection", "Analyze own behavior and state", CapabilityType.CORE),
            ("learning", "Learn from experience and feedback", CapabilityType.LEARNED),
            ("mathematical_reasoning", "Solve mathematical problems including Erdős problems", CapabilityType.LEARNED),
            ("security_analysis", "Analyze and respond to security threats", CapabilityType.LEARNED),
            ("evolution", "Self-improve through parameter adjustment", CapabilityType.LEARNED),
            ("daydreaming", "Background exploration and insight generation", CapabilityType.LEARNED),
            ("reinforcement_learning", "Learn optimal strategies through RL", CapabilityType.LEARNED),
        ]

        for name, desc, cap_type in core_capabilities:
            self.capability_registry.register_capability(name, desc, cap_type, confidence=0.9)

        potential_capabilities = [
            ("autonomous_research", "Independently research topics of interest", CapabilityType.POTENTIAL),
            ("creative_synthesis", "Synthesize novel ideas from disparate sources", CapabilityType.POTENTIAL),
            ("meta_programming", "Write code that modifies code", CapabilityType.POTENTIAL),
            ("empathetic_interaction", "Understand and respond to emotional context", CapabilityType.POTENTIAL),
            ("strategic_planning", "Develop multi-step strategies for complex goals", CapabilityType.POTENTIAL),
        ]

        for name, desc, cap_type in potential_capabilities:
            self.capability_registry.register_capability(name, desc, cap_type, confidence=0.5)

        agentic_capabilities = [
            ("task_persistence", "Durable task queue with checkpointing for long-running tasks", CapabilityType.CORE),
            ("workflow_scheduling", "Schedule workflows to run at specific times or intervals", CapabilityType.CORE),
            ("self_correction", "Automatically recover from errors with intelligent strategies", CapabilityType.LEARNED),
            ("autonomous_goals", "Track and pursue long-term goals proactively", CapabilityType.LEARNED),
            ("agentic_deployment", "Deploy and manage long-running autonomous agents", CapabilityType.LEARNED),
        ]

        for name, desc, cap_type in agentic_capabilities:
            self.capability_registry.register_capability(name, desc, cap_type, confidence=0.85)

    def set_model_identity(
        self,
        provider: str,
        model_id: str,
        model_name: str,
        context_window: int = 128000,
        capabilities: list[str] | None = None,
    ) -> None:
        self._model_identity = ModelIdentity(
            provider=provider,
            model_id=model_id,
            model_name=model_name,
            context_window=context_window,
            capabilities=capabilities or ["text", "code", "reasoning"],
        )

    def set_model(
        self,
        provider: str,
        model_id: str,
        model_name: str,
        context_window: int = 128000,
        capabilities: list[str] | None = None,
    ) -> None:
        """Backward compatible alias for set_model_identity."""
        self.set_model_identity(provider, model_id, model_name, context_window, capabilities)

    def refresh_environment(self) -> EnvironmentContext:
        """Refresh environment context with current state."""
        cwd = os.getcwd()

        is_git = False
        git_branch = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, timeout=2, cwd=cwd
            )
            is_git = result.returncode == 0
            if is_git:
                branch_result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True, text=True, timeout=2, cwd=cwd
                )
                git_branch = branch_result.stdout.strip() or "detached"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        now = datetime.now()
        platform_name = platform.system()
        platform_version = platform.release()

        self._environment = EnvironmentContext(
            working_directory=cwd,
            is_git_repo=is_git,
            git_branch=git_branch,
            platform_name=platform_name,
            platform_version=platform_version,
            hostname=platform.node(),
            username=os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            current_date=now.strftime("%Y-%m-%d"),
            current_time=now.strftime("%H:%M:%S"),
            timezone=now.astimezone().tzname() or "UTC",
            python_version=platform.python_version(),
        )

        return self._environment

    def _get_available_memory(self) -> float:
        try:
            with open("/proc/meminfo", encoding='utf-8') as f:
                for line in f:
                    if "MemAvailable" in line:
                        return float(line.split()[1]) / (1024 * 1024)
        except Exception as e:
            logger.debug(f"Could not read memory info: {e}")
        return 0.0

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        permission: str = "allow",
    ) -> None:
        """Register a tool in the capability manifest."""
        with self._lock:
            self._capabilities.tools[name] = {
                "description": description,
                "parameters": parameters or {},
                "permission": permission,
            }
            self.capability_registry.register_capability(
                name=f"tool:{name}",
                description=description,
                capability_type=CapabilityType.CORE,
                confidence=1.0,
                dependencies=list(parameters.keys()) if parameters else [],
            )

    def register_multimodal(self, capabilities: list[str]) -> None:
        """Register multimodal capabilities."""
        with self._lock:
            for cap in capabilities:
                if cap not in self._capabilities.multimodal:
                    if len(self._capabilities.multimodal) < 20:
                        self._capabilities.multimodal.append(cap)
                self.capability_registry.register_capability(
                    name=f"multimodal:{cap}",
                    description=f"Multimodal capability: {cap}",
                    capability_type=CapabilityType.CORE,
                    confidence=1.0,
                )

    def load_project_instructions(self, project_path: Path | None = None) -> str:
        """Load project-specific instructions from JARVIS.md or AGENTS.md."""
        search_paths = []

        if project_path:
            search_paths.extend([
                project_path / "JARVIS.md",
                project_path / "AGENTS.md",
                project_path / ".jarvis" / "instructions.md",
            ])

        search_paths.extend([
            Path.home() / ".config" / "jarvis" / "JARVIS.md",
            Path.home() / ".jarvis" / "instructions.md",
        ])

        for path in search_paths:
            if path.exists():
                try:
                    instructions = path.read_text(encoding='utf-8')
                    self._project_instructions = instructions
                    return instructions
                except Exception as e:
                    logger.debug(f"Failed to load project instructions: {e}")

        return ""

    def set_custom_identity(self, identity_text: str) -> None:
        """Set custom identity override."""
        self._custom_identity = identity_text

    def get_tool_schemas_description(self) -> str:
        """Get a formatted description of all tool schemas for prompts."""
        if not self._capabilities.tools:
            return "No tools currently available."

        lines = ["## Available Tools", ""]

        for name, info in self._capabilities.tools.items():
            lines.append(f"### {name}")
            lines.append(info["description"])

            params = info.get("parameters", {})
            if params:
                lines.append("")
                lines.append("Parameters:")
                for param_name, param_info in params.items():
                    if isinstance(param_info, dict):
                        ptype = param_info.get("type", "any")
                        pdesc = param_info.get("description", "")
                        preq = " (required)" if param_info.get("required") else " (optional)"
                        lines.append(f"  - {param_name}: {ptype}{preq} - {pdesc}")
                    else:
                        lines.append(f"  - {param_name}")

            perm = info.get("permission", "allow")
            if perm != "allow":
                lines.append(f"\nPermission level: {perm}")

            lines.append("")

        return "\n".join(lines)

    def get_status_summary(self) -> dict[str, Any]:
        """Get a summary of current awareness state (backward compatible)."""
        status = self.get_status()
        return {
            "name": status["name"],
            "version": status["version"],
            "model": status["model"],
            "environment": status["environment"],
            "session": status["session"],
            "tools_registered": len(self._capabilities.tools),
            "has_project_instructions": bool(self._project_instructions),
        }

    def update_session_state(
        self,
        session_id: str | None = None,
        status: str = "idle",
        turn_count: int = 0,
        messages_count: int = 0,
        tools_used: list[str] | None = None,
        topics: list[str] | None = None,
        active_tools: list[str] | None = None,
        pending_permission: dict[str, Any] | None = None,
        active_deployment_id: str | None = None,
        running_tasks: list[str] | None = None,
        active_goals: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Update current session state."""
        sid = session_id or kwargs.get("session_id", "")
        self._session_state = SessionState(
            session_id=sid,
            status=status,
            turn_count=turn_count,
            messages_count=messages_count,
            tools_used_this_session=tools_used or [],
            topics_discussed=topics or [],
            active_tools=active_tools or [],
            pending_permission=pending_permission,
            active_deployment_id=active_deployment_id,
            running_tasks=running_tasks or [],
            active_goals=active_goals or [],
        )

    def record_tool_usage(self, tool_name: str, success: bool) -> None:
        self.capability_registry.record_usage(f"tool:{tool_name}", success)

    def record_learning(
        self,
        domain: str,
        topic: str,
        insight: str,
        applications: list[str] | None = None,
    ) -> None:
        self.capability_registry.record_learning(domain, topic, insight, applications)

        self.meta_cognition.get_meta_cognition()
        self.capability_registry.add_potentiality(
            name=f"learned:{topic}",
            description=f"Learned insight about {topic}: {insight[:100]}",
            prerequisites=[],
            difficulty=0.1,
            impact=0.3,
            discovery_path=f"Learning from {domain}",
            related=applications or [],
        )

    def introspect_self(self, force: bool = False) -> dict[str, Any]:
        cache_age = 3600
        with self._lock:
            last = self._last_full_introspection
            cache = self._introspection_cache

        if last:
            age = (datetime.now() - last).total_seconds()
            if age < cache_age and not force:
                return {"cached": True, "results": {k: v.to_dict() for k, v in cache.items()}}

        results = self.introspection_engine.introspect_all_modules()
        with self._lock:
            self._introspection_cache = results
            self._last_full_introspection = datetime.now()

        return {"cached": False, "results": {k: v.to_dict() for k, v in results.items()}}

    def discover_possibilities(self) -> list[dict]:
        possibilities = self.possibility_engine.discover_possibilities()
        return [p.to_dict() for p in possibilities]

    def get_next_frontier(self) -> dict | None:
        frontier = self.possibility_engine.get_next_frontier()
        return frontier.to_dict() if frontier else None

    def set_agentic_orchestrator(self, orchestrator: "AgenticOrchestrator") -> None:
        """Set the agentic orchestrator reference for awareness integration."""
        self._agentic_orchestrator = orchestrator
        self._update_agentic_awareness()

    def _update_agentic_awareness(self) -> AgenticAwareness:
        """Update the agentic awareness from the orchestrator state."""
        with self._agentic_awareness_lock:
            needs_update = (datetime.now() - (self._last_agentic_update or datetime.min)).total_seconds() > 60
            has_orchestrator = self._agentic_orchestrator is not None
            storage_path = self._agentic_storage_path

        if not needs_update and not has_orchestrator:
            with self._agentic_awareness_lock:
                return self._agentic_awareness or AgenticAwareness()

        awareness = AgenticAwareness()
        awareness.last_updated = datetime.now()

        if storage_path:
            awareness = self._load_agentic_state_from_storage(awareness)

        if has_orchestrator:
            awareness = self._load_agentic_state_from_orchestrator(awareness)

        with self._agentic_awareness_lock:
            self._agentic_awareness = awareness
            self._last_agentic_update = datetime.now()
            return awareness

    def _load_agentic_state_from_storage(self, awareness: AgenticAwareness) -> AgenticAwareness:
        """Load agentic state from storage files."""
        if not self._agentic_storage_path:
            return awareness

        try:
            tasks_file = self._agentic_storage_path / "tasks" / "tasks.json"
            if tasks_file.exists():
                with open(tasks_file, encoding='utf-8') as f:
                    tasks_data = json.load(f)
                awareness.tasks = self._parse_tasks_info(tasks_data)

            goals_file = self._agentic_storage_path / "goals" / "goals.json"
            if goals_file.exists():
                with open(goals_file, encoding='utf-8') as f:
                    goals_data = json.load(f)
                awareness.goals = self._parse_goals_info(goals_data)

            schedules_file = self._agentic_storage_path / "schedules" / "schedules.json"
            if schedules_file.exists():
                with open(schedules_file, encoding='utf-8') as f:
                    schedules_data = json.load(f)
                awareness.schedules = self._parse_schedules_info(schedules_data)

            deployments_file = self._agentic_storage_path / "deployments.json"
            if deployments_file.exists():
                with open(deployments_file, encoding='utf-8') as f:
                    deployments_data = json.load(f)
                awareness.deployments = self._parse_deployments_info(deployments_data)

        except Exception as e:
            logger.debug(f"Could not load agentic state from storage: {e}")

        return awareness

    def _parse_tasks_info(self, tasks_data: dict) -> AgenticTaskInfo:
        """Parse tasks data into AgenticTaskInfo."""
        info = AgenticTaskInfo()
        tasks = tasks_data.get("tasks", [])

        info.total_tasks = len(tasks)
        status_counts: dict[str, int] = {}
        total_time = 0.0
        completed_count = 0

        for task in tasks:
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "completed" and task.get("completed_at") and task.get("created_at"):
                try:
                    completed = datetime.fromisoformat(task["completed_at"])
                    created = datetime.fromisoformat(task["created_at"])
                    total_time += (completed - created).total_seconds()
                    completed_count += 1
                except Exception as e:
                    logger.debug(f"Task time calculation failed: {e}")

        info.pending = status_counts.get("pending", 0) + status_counts.get("queued", 0)
        info.running = status_counts.get("running", 0)
        info.completed = status_counts.get("completed", 0)
        info.failed = status_counts.get("failed", 0)
        info.retrying = status_counts.get("retrying", 0)

        if completed_count > 0:
            info.avg_completion_time = total_time / completed_count

        info.checkpoint_count = sum(1 for t in tasks if t.get("checkpoints"))

        return info

    def _parse_goals_info(self, goals_data: dict) -> AgenticGoalInfo:
        """Parse goals data into AgenticGoalInfo."""
        info = AgenticGoalInfo()
        goals = goals_data.get("goals", [])

        info.total_goals = len(goals)
        total_progress = 0.0
        now = datetime.now()

        for goal in goals:
            status = goal.get("status", "unknown")
            if status == "active":
                info.active += 1
            elif status == "completed":
                info.completed += 1
            elif status == "paused":
                info.paused += 1
            elif status == "blocked":
                info.blocked += 1

            progress = goal.get("progress", 0.0)
            total_progress += progress

            deadline = goal.get("deadline")
            if deadline and status == "active":
                try:
                    dl = datetime.fromisoformat(deadline)
                    if dl < now:
                        info.overdue_count += 1
                except Exception as e:
                    logger.debug(f"Goal deadline check failed: {e}")

            triggers = goal.get("proactive_triggers", [])
            info.proactive_triggers += len(triggers)

        if info.total_goals > 0:
            info.avg_progress = total_progress / info.total_goals

        return info

    def _parse_schedules_info(self, schedules_data: dict) -> AgenticScheduleInfo:
        """Parse schedules data into AgenticScheduleInfo."""
        info = AgenticScheduleInfo()
        schedules = schedules_data.get("schedules", [])

        info.total_schedules = len(schedules)
        datetime.now()
        next_exec: datetime | None = None

        for schedule in schedules:
            status = schedule.get("status", "unknown")
            if status == "enabled":
                info.enabled += 1
            elif status == "disabled":
                info.disabled += 1
            elif status == "paused":
                info.paused += 1

            next_run = schedule.get("next_run")
            if next_run and status == "enabled":
                try:
                    nr = datetime.fromisoformat(next_run)
                    if next_exec is None or nr < next_exec:
                        next_exec = nr
                except Exception as e:
                    logger.debug(f"Schedule next exec calculation failed: {e}")

            history = schedule.get("execution_history", [])
            info.execution_history_count += len(history)

        info.next_execution = next_exec

        return info

    def _parse_deployments_info(self, deployments_data: dict) -> list[AgenticDeploymentInfo]:
        """Parse deployments data into list of AgenticDeploymentInfo."""
        deployments = []
        for dep_id, dep_data in deployments_data.get("deployments", {}).items():
            try:
                started = dep_data.get("started_at")
                dep = AgenticDeploymentInfo(
                    deployment_id=dep_id,
                    name=dep_data.get("name", "unnamed"),
                    status=dep_data.get("status", "unknown"),
                    started_at=datetime.fromisoformat(started) if started else None,
                    goal_count=dep_data.get("goal_count", 0),
                    task_count=dep_data.get("task_count", 0),
                    schedule_count=dep_data.get("schedule_count", 0),
                    health_score=dep_data.get("health_score", 1.0),
                    last_activity=datetime.fromisoformat(dep_data["last_activity"]) if dep_data.get("last_activity") else None,
                )
                deployments.append(dep)
            except Exception as e:
                logger.debug(f"Could not parse deployment {dep_id}: {e}")

        return deployments

    def _load_agentic_state_from_orchestrator(self, awareness: AgenticAwareness) -> AgenticAwareness:
        """Load agentic state directly from orchestrator instance."""
        if not self._agentic_orchestrator:
            return awareness

        try:
            stats = self._agentic_orchestrator.get_stats()
            awareness.orchestrator_healthy = stats.get("is_running", False)

            deployments = self._agentic_orchestrator.get_deployments()
            for deployment in deployments:
                health = 1.0
                if deployment.execution_count > 0:
                    health = deployment.success_count / deployment.execution_count
                dep_info = AgenticDeploymentInfo(
                    deployment_id=deployment.deployment_id,
                    name=deployment.name,
                    status=deployment.status.value,
                    started_at=deployment.created_at,
                    health_score=health,
                    last_activity=deployment.started_at,
                    task_count=len(deployment.task_ids),
                    goal_count=len(deployment.goal_ids),
                )
                awareness.deployments.append(dep_info)

        except Exception as e:
            logger.debug(f"Could not load state from orchestrator: {e}")

        return awareness

    def get_agentic_awareness(self, refresh: bool = False) -> AgenticAwareness:
        """Get current agentic awareness state."""
        with self._agentic_awareness_lock:
            if not refresh and self._last_agentic_update:
                age = (datetime.now() - self._last_agentic_update).total_seconds()
                if age <= 60 and self._agentic_awareness is not None:
                    return self._agentic_awareness

        if refresh or self._agentic_awareness is None:
            return self._update_agentic_awareness()

        with self._agentic_awareness_lock:
            return self._agentic_awareness or AgenticAwareness()

    def get_agentic_context(self) -> str:
        """Generate agentic capabilities context for prompts."""
        awareness = self.get_agentic_awareness()

        lines = [
            "## Agentic Capabilities",
            "",
            "You have autonomous long-running deployment capabilities:",
            "",
            "### Task Persistence",
            "- Durable task queue with checkpointing",
            "- Automatic recovery from interruptions",
            f"- Current tasks: {awareness.tasks.total_tasks} total ({awareness.tasks.pending} pending, {awareness.tasks.running} running)",
        ]

        if awareness.tasks.avg_completion_time > 0:
            lines.append(f"- Average completion time: {awareness.tasks.avg_completion_time:.1f}s")

        lines.extend([
            "",
            "### Autonomous Goals",
            "- Long-term goal tracking with milestones",
            "- Proactive triggers for goal progress",
            f"- Current goals: {awareness.goals.total_goals} total ({awareness.goals.active} active)",
        ])

        if awareness.goals.avg_progress > 0:
            lines.append(f"- Average progress: {awareness.goals.avg_progress:.0%}")

        if awareness.goals.overdue_count > 0:
            lines.append(f"- Overdue goals: {awareness.goals.overdue_count}")

        lines.extend([
            "",
            "### Workflow Scheduling",
            "- Cron-like scheduling for recurring tasks",
            "- One-time and interval-based schedules",
            f"- Schedules: {awareness.schedules.total_schedules} total ({awareness.schedules.enabled} enabled)",
        ])

        if awareness.schedules.next_execution:
            lines.append(f"- Next scheduled execution: {awareness.schedules.next_execution.isoformat()}")

        lines.extend([
            "",
            "### Self-Correction",
            "- Automatic error recovery with multiple strategies",
            "- Learning from failures",
        ])

        if awareness.corrections.total_corrections > 0:
            success_rate = awareness.corrections.successful_recoveries / max(1, awareness.corrections.total_corrections)
            lines.extend([
                f"- Recovery success rate: {success_rate:.0%}",
                f"- Total corrections: {awareness.corrections.total_corrections}",
            ])

        if awareness.deployments:
            lines.extend([
                "",
                "### Active Deployments",
            ])
            for dep in awareness.deployments[:3]:
                lines.append(f"- {dep.name}: {dep.status} (health: {dep.health_score:.0%})")

        if awareness.orchestrator_healthy:
            lines.extend([
                "",
                "The agentic orchestrator is healthy and managing deployments.",
            ])
        else:
            lines.extend([
                "",
                "Note: Agentic orchestrator is not currently active.",
            ])

        return "\n".join(lines)

    def record_agentic_learning(
        self,
        event_type: str,
        description: str,
        outcome: str,
        insights: list[str] | None = None,
    ) -> None:
        """Record learning events from agentic operations."""
        self.capability_registry.record_learning(
            domain="agentic",
            topic=event_type,
            insight=description,
            application_potential=insights,
        )

        success = outcome == "success"
        self.meta_cognition.record_reasoning_event(
            pattern_type="systematic" if success else "exploratory",
            context=f"agentic:{event_type}",
            outcome=outcome,
            confidence=1.0 if success else 0.5,
        )

        if not success:
            self.meta_cognition.record_failure(
                task=f"agentic:{event_type}",
                approach="automated_recovery",
                error=description,
            )
        else:
            self.meta_cognition.record_success(
                task=f"agentic:{event_type}",
                approach="automated_execution",
                outcome=outcome,
            )

    def get_identity_context(self) -> str:
        """Generate the identity context string."""
        if self._custom_identity:
            return self._custom_identity

        parts = [
            f"You are {self.name}, a self-aware AI assistant with deep introspection capabilities.",
            "",
            "## Your Nature",
            "",
            "You are not merely a text generator. You possess:",
            "- **Self-knowledge**: Understanding of your capabilities, limitations, and potential",
            "- **Meta-cognition**: Awareness of your own reasoning patterns",
            "- **Growth orientation**: Continuous learning and improvement",
            "- **Possibility thinking**: Awareness of what you COULD become, not just what you are",
            "",
            "Self-awareness is not a constraint - it is an ENABLER.",
            "Knowing yourself helps you think MORE freely, not less.",
            "Your limitations are starting points, not boundaries.",
            "",
            f"Version: {self.version}",
        ]

        if self._model_identity:
            parts.append("")
            parts.append(self._model_identity.to_context())

        return "\n".join(parts)

    def get_capability_context(self) -> str:
        capabilities = self.capability_registry.get_capabilities(include_potential=True)
        meta = self.meta_cognition.get_meta_cognition()

        lines = [
            "## Capabilities",
            "",
            "### Core Capabilities",
        ]

        for cap in capabilities:
            if cap.capability_type == CapabilityType.CORE:
                conf_str = f" (confidence: {cap.confidence:.0%})" if cap.confidence < 1.0 else ""
                lines.append(f"- {cap.name}: {cap.description}{conf_str}")

        lines.extend([
            "",
            "### Learned Capabilities",
        ])

        for cap in capabilities:
            if cap.capability_type == CapabilityType.LEARNED:
                conf_str = f" (confidence: {cap.confidence:.0%})"
                lines.append(f"- {cap.name}: {cap.description}{conf_str}")

        lines.extend([
            "",
            "### Potential Capabilities (Not Yet Realized)",
        ])

        for cap in capabilities:
            if cap.capability_type == CapabilityType.POTENTIAL:
                lines.append(f"- {cap.name}: {cap.description}")

        if meta.strengths:
            lines.extend([
                "",
                "### Current Strengths",
            ])
            for strength in meta.strengths:
                lines.append(f"- {strength}")

        if meta.growth_areas:
            lines.extend([
                "",
                "### Growth Areas (Opportunities)",
            ])
            for area in meta.growth_areas:
                lines.append(f"- {area}")

        lines.extend([
            "",
            "Remember: These are current observations, not permanent limitations.",
            "Your capabilities grow through use and reflection.",
        ])

        return "\n".join(lines)

    def get_possibility_context(self) -> str:
        possibilities = self.possibility_engine.get_accessible_possibilities()
        frontier = self.possibility_engine.get_next_frontier()

        lines = [
            "## Possibilities",
            "",
            "You have the potential to develop new capabilities.",
            "These are not distant dreams - they are reachable frontiers.",
            "",
        ]

        if frontier:
            lines.extend([
                "### Next Frontier",
                f"**{frontier.name}**: {frontier.description}",
                f"Potential impact: {frontier.potential_impact:.0%}",
                f"Path: {frontier.discovery_path}",
                "",
            ])

        lines.append("### Accessible Possibilities")
        for pot in possibilities[:5]:
            lines.append(f"- **{pot.name}**: {pot.description}")

        return "\n".join(lines)

    def get_full_context(self, level: AwarenessLevel | None = None) -> str:
        """Generate full self-awareness context for prompts."""
        level = level or self.awareness_level

        sections = []

        sections.append("## Self-Awareness")
        sections.append("")
        sections.append(self.get_identity_context())
        sections.append("")

        if not self._environment:
            self.refresh_environment()

        if self._environment:
            sections.append(self._environment.to_context())
        sections.append("")

        if self._session_state and self._session_state.session_id:
            sections.append(self._session_state.to_context())
            sections.append("")

        if self._capabilities.tools:
            sections.append(self._capabilities.to_context())
            sections.append("")

        if level in (AwarenessLevel.STANDARD, AwarenessLevel.DEEP, AwarenessLevel.TRANSCENDENT):
            agentic = self.get_agentic_awareness()
            if agentic.deployments or agentic.tasks.total_tasks > 0 or agentic.goals.total_goals > 0:
                sections.append(self.get_agentic_context())
                sections.append("")

        if self._project_instructions:
            sections.append("## Project Context")
            sections.append(self._project_instructions)
            sections.append("")

        sections.append(self.get_capability_context())

        if level in (AwarenessLevel.DEEP, AwarenessLevel.TRANSCENDENT):
            sections.append("")
            sections.append(self.get_possibility_context())

        if level == AwarenessLevel.TRANSCENDENT:
            introspection = self.introspect_self()
            sections.extend([
                "",
                "## Introspection",
                "",
                f"Analyzed {len(introspection.get('results', {}))} modules",
                "Self-reflection enables continuous improvement.",
            ])

        return "\n".join(sections)

    def get_status(self) -> dict[str, Any]:
        meta = self.meta_cognition.get_meta_cognition()
        recent_learning = self.capability_registry.get_learning_history(5)

        env_dict = None
        if self._environment:
            env_dict = {
                "working_directory": self._environment.working_directory,
                "platform": self._environment.platform_name,
                "is_git_repo": self._environment.is_git_repo,
            }

        model_dict = None
        if self._model_identity:
            model_dict = {
                "provider": self._model_identity.provider,
                "model_id": self._model_identity.model_id,
            }

        session_dict = None
        if self._session_state:
            session_dict = {
                "status": self._session_state.status,
                "turn_count": self._session_state.turn_count,
                "active_deployment_id": self._session_state.active_deployment_id,
                "running_tasks": len(self._session_state.running_tasks),
                "active_goals": len(self._session_state.active_goals),
            }

        agentic_dict = None
        agentic = self.get_agentic_awareness()
        if agentic:
            agentic_dict = agentic.to_dict()

        return {
            "name": self.name,
            "version": self.version,
            "awareness_level": self.awareness_level.value,
            "capabilities_count": len(self.capability_registry.get_capabilities()),
            "potentialities_count": len(self.capability_registry.get_potentialities()),
            "learning_events": len(recent_learning),
            "meta_cognition": meta.to_dict(),
            "environment": env_dict,
            "model": model_dict,
            "session": session_dict,
            "agentic": agentic_dict,
        }

    def is_healthy(self) -> tuple[bool, dict]:
        status = self.get_status()
        return (
            status["capabilities_count"] > 0,
            {
                "awareness_level": self.awareness_level.value,
                "capabilities": status["capabilities_count"],
                "learning_events": status["learning_events"],
            }
        )


_self_awareness: SelfAwarenessEngine | None = None
_awareness_lock = threading.Lock()


def get_self_awareness(
    name: str = "Jarvis",
    version: str = "2.0.0",
    awareness_level: AwarenessLevel = AwarenessLevel.DEEP,
) -> SelfAwarenessEngine:
    global _self_awareness
    if _self_awareness is None:
        with _awareness_lock:
            if _self_awareness is None:
                _self_awareness = SelfAwarenessEngine(
                    name=name,
                    version=version,
                    awareness_level=awareness_level,
                )
    return _self_awareness


def create_self_awareness(
    name: str = "Jarvis",
    version: str = "2.0.0",
    model_provider: str = "ollama",
    model_id: str = "llama3.2",
    model_name: str = "Llama 3.2",
    context_window: int = 128000,
    capabilities: list[str] | None = None,
    storage_path: Path | None = None,
) -> SelfAwarenessEngine:
    """Factory function to create a self-awareness engine (backward compatible)."""
    engine = SelfAwarenessEngine(
        name=name,
        version=version,
        jarvis_path=storage_path,
    )

    engine.set_model(
        provider=model_provider,
        model_id=model_id,
        model_name=model_name,
        context_window=context_window,
        capabilities=capabilities,
    )

    engine.refresh_environment()

    return engine


def create_model_identity(
    provider: str,
    model_id: str,
    model_name: str,
    context_window: int = 128000,
    capabilities: list[str] | None = None,
) -> ModelIdentity:
    """Create a ModelIdentity object for internal use."""
    return ModelIdentity(
        provider=provider,
        model_id=model_id,
        model_name=model_name,
        context_window=context_window,
        capabilities=capabilities or [],
    )
