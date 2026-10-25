"""Tool learning and improvement system for Jarvis.

Integrates with:
- Daydream engine for offline practice
- Evolution orchestrator for tool-specific genes
- Memory for storing tool competency data
- Insight bridge for learning from mistakes
"""
import asyncio
import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from core.helpers import safe_error_message
from typing import ClassVar
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolCompetencyLevel(Enum):
    NOVICE = "novice"
    LEARNING = "learning"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    EXPERT = "expert"


@dataclass
class ToolUsageRecord:
    """Record of a single tool usage."""
    tool_name: str
    timestamp: datetime
    success: bool
    execution_time: float
    error_type: str | None = None
    error_message: str | None = None
    parameters_used: dict = field(default_factory=dict)
    result_quality: float = 0.5  # 0-1, estimated quality
    user_feedback: str | None = None


@dataclass
class ToolCompetency:
    """Competency tracking for a single tool."""
    tool_name: str
    level: ToolCompetencyLevel = ToolCompetencyLevel.NOVICE
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    avg_execution_time: float = 0.0
    avg_quality: float = 0.0
    common_errors: dict[str, int] = field(default_factory=dict)
    optimal_parameters: dict = field(default_factory=dict)
    learning_insights: list[str] = field(default_factory=list)
    last_used: datetime | None = None
    improvement_trends: list[float] = field(default_factory=list)
    practice_sessions: int = 0
    _MAX_COMMON_ERRORS: ClassVar[int] = 20
    _MAX_OPTIMAL_PARAMS: ClassVar[int] = 20
    _MAX_LEARNING_INSIGHTS: ClassVar[int] = 50
    _MAX_IMPROVEMENT_TRENDS: ClassVar[int] = 100

    def _evict_common_errors(self):
        while len(self.common_errors) > self._MAX_COMMON_ERRORS:
            least_common = min(self.common_errors, key=self.common_errors.get)
            del self.common_errors[least_common]

    def _evict_optimal_parameters(self):
        while len(self.optimal_parameters) > self._MAX_OPTIMAL_PARAMS:
            oldest_key = next(iter(self.optimal_parameters))
            del self.optimal_parameters[oldest_key]


@dataclass
class ToolGene:
    """A gene that affects tool behavior."""
    tool_name: str
    gene_name: str
    current_value: float
    min_value: float = 0.0
    max_value: float = 1.0
    adaptation_rate: float = 0.1
    history: list[tuple[datetime, float]] = field(default_factory=list)


class ToolLearningEngine:
    """Engine for learning and improving tool usage.
    
    Features:
    - Tracks competency per tool
    - Learns optimal parameters
    - Identifies common errors and fixes
    - Practices tools during dreaming
    - Integrates with evolution for tool-specific genes
    - Stores learned knowledge in memory
    """

    MAX_RECORDS_PER_TOOL = 1000
    PRACTICE_INTERVAL = 3600  # 1 hour between practice sessions
    COMPETENCY_THRESHOLDS = {
        ToolCompetencyLevel.NOVICE: 0,
        ToolCompetencyLevel.LEARNING: 10,
        ToolCompetencyLevel.COMPETENT: 50,
        ToolCompetencyLevel.PROFICIENT: 100,
        ToolCompetencyLevel.EXPERT: 200,
    }

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "tool_learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.competencies: dict[str, ToolCompetency] = {}
        self.recent_usage: dict[str, list[ToolUsageRecord]] = defaultdict(list)
        self.tool_genes: dict[str, dict[str, ToolGene]] = defaultdict(dict)
        self._running = False
        self._practice_task: asyncio.Task | None = None

        self._load_state()
        self._init_tool_genes()

    def _init_tool_genes(self) -> None:
        """Initialize tool-specific genes for tools that have them."""
        tool_specific_genes = {
            "web_search": {
                "result_depth": ToolGene("web_search", "result_depth", 0.5, 0.1, 1.0, 0.05),
                "query_refinement": ToolGene("web_search", "query_refinement", 0.5, 0.0, 1.0, 0.1),
            },
            "code_execute": {
                "timeout_sensitivity": ToolGene("code_execute", "timeout_sensitivity", 0.7, 0.3, 1.0, 0.05),
                "error_recovery": ToolGene("code_execute", "error_recovery", 0.5, 0.1, 0.9, 0.1),
            },
            "memory_recall": {
                "context_window": ToolGene("memory_recall", "context_window", 0.6, 0.2, 1.0, 0.1),
                "relevance_threshold": ToolGene("memory_recall", "relevance_threshold", 0.5, 0.1, 0.9, 0.05),
            },
            "erdos_math": {
                "problem_complexity": ToolGene("erdos_math", "problem_complexity", 0.5, 0.1, 1.0, 0.1),
                "strategy_exploration": ToolGene("erdos_math", "strategy_exploration", 0.4, 0.1, 0.8, 0.1),
            },
            "daydream": {
                "depth_level": ToolGene("daydream", "depth_level", 0.5, 0.1, 1.0, 0.1),
                "creativity_factor": ToolGene("daydream", "creativity_factor", 0.4, 0.1, 0.9, 0.1),
            },
            "evolution": {
                "risk_tolerance": ToolGene("evolution", "risk_tolerance", 0.3, 0.1, 0.6, 0.05),
                "exploration_bias": ToolGene("evolution", "exploration_bias", 0.5, 0.1, 0.9, 0.1),
            },
            "notes": {
                "organization_style": ToolGene("notes", "organization_style", 0.5, 0.0, 1.0, 0.1),
            },
            "reason": {
                "depth_preference": ToolGene("reason", "depth_preference", 0.6, 0.2, 1.0, 0.1),
                "verification_level": ToolGene("reason", "verification_level", 0.7, 0.3, 1.0, 0.05),
            },
        }

        for tool_name, genes in tool_specific_genes.items():
            self.tool_genes[tool_name] = genes

    def _load_state(self) -> None:
        """Load tool learning state from disk."""
        state_file = self.storage_path / "tool_learning_state.json"
        if state_file.exists():
            try:
                with open(state_file, encoding='utf-8') as f:
                    data = json.load(f)

                for tool_name, comp_data in data.get("competencies", {}).items():
                    total = comp_data.get("total_uses", 0)
                    successful = min(comp_data.get("successful_uses", 0), total)
                    failed = min(comp_data.get("failed_uses", 0), total - successful)
                    
                    self.competencies[tool_name] = ToolCompetency(
                        tool_name=tool_name,
                        level=ToolCompetencyLevel(comp_data.get("level", "novice")),
                        total_uses=total,
                        successful_uses=successful,
                        failed_uses=failed,
                        avg_execution_time=comp_data.get("avg_execution_time", 0.0),
                        avg_quality=comp_data.get("avg_quality", 0.0),
                        common_errors=comp_data.get("common_errors", {}),
                        optimal_parameters=comp_data.get("optimal_parameters", {}),
                        learning_insights=comp_data.get("learning_insights", []),
                        practice_sessions=comp_data.get("practice_sessions", 0),
                    )

                logger.info(f"Loaded tool learning state: {len(self.competencies)} tools tracked")
            except Exception as e:
                logger.warning(f"Failed to load tool learning state: {e}")

    def _save_state(self) -> None:
        """Save tool learning state to disk."""
        state_file = self.storage_path / "tool_learning_state.json"
        try:
            data = {
                "competencies": {
                    name: {
                        "level": comp.level.value,
                        "total_uses": comp.total_uses,
                        "successful_uses": comp.successful_uses,
                        "failed_uses": comp.failed_uses,
                        "avg_execution_time": comp.avg_execution_time,
                        "avg_quality": comp.avg_quality,
                        "common_errors": comp.common_errors,
                        "optimal_parameters": comp.optimal_parameters,
                        "learning_insights": comp.learning_insights[-50:],
                        "practice_sessions": comp.practice_sessions,
                    }
                    for name, comp in self.competencies.items()
                },
                "tool_genes": {
                    tool: {
                        gene_name: {
                            "current_value": gene.current_value,
                            "history": [(t.isoformat(), v) for t, v in gene.history[-20:]],
                        }
                        for gene_name, gene in genes.items()
                    }
                    for tool, genes in self.tool_genes.items()
                },
                "last_updated": datetime.now().isoformat(),
            }

            temp_file = state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(state_file)
        except Exception as e:
            logger.error(f"Failed to save tool learning state: {e}")

    def record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        error_type: str | None = None,
        error_message: str | None = None,
        parameters: dict | None = None,
        result_quality: float = 0.5,
        user_feedback: str | None = None,
    ) -> None:
        """Record a tool usage for learning."""
        with self._lock:
            if tool_name not in self.competencies:
                self.competencies[tool_name] = ToolCompetency(tool_name=tool_name)

            comp = self.competencies[tool_name]
            comp.total_uses += 1
            comp.last_used = datetime.now()

            if success:
                comp.successful_uses += 1
            else:
                comp.failed_uses += 1
                if error_type:
                    comp.common_errors[error_type] = comp.common_errors.get(error_type, 0) + 1
                    comp._evict_common_errors()

            # Update running averages
            total = comp.total_uses
            comp.avg_execution_time = (
                (comp.avg_execution_time * (total - 1) + execution_time) / total
            )
            comp.avg_quality = (
                (comp.avg_quality * (total - 1) + result_quality) / total
            )

            # Track improvement
            if success and result_quality > 0.5:
                comp.improvement_trends.append(result_quality)
                if len(comp.improvement_trends) > 50:
                    comp.improvement_trends = comp.improvement_trends[-50:]

            # Validate consistency
            if comp.successful_uses > comp.total_uses:
                comp.successful_uses = comp.total_uses
            if comp.failed_uses > comp.total_uses:
                comp.failed_uses = comp.total_uses

            # Update level based on usage
            self._update_competency_level(tool_name)

            # Learn from success
            if success and parameters and result_quality > 0.7:
                self._update_optimal_parameters(tool_name, parameters)

            # Learn from failure
            if not success and error_message:
                self._extract_failure_insight(tool_name, error_message, parameters)

            # Store record
            record = ToolUsageRecord(
                tool_name=tool_name,
                timestamp=datetime.now(),
                success=success,
                execution_time=execution_time,
                error_type=error_type,
                error_message=error_message,
                parameters_used=parameters or {},
                result_quality=result_quality,
                user_feedback=user_feedback,
            )
            self.recent_usage[tool_name].append(record)
            if len(self.recent_usage[tool_name]) > self.MAX_RECORDS_PER_TOOL:
                self.recent_usage[tool_name] = self.recent_usage[tool_name][-self.MAX_RECORDS_PER_TOOL:]

            # Adapt tool genes based on usage
            self._adapt_tool_genes(tool_name, success, result_quality)

            # Save periodically
            if comp.total_uses % 10 == 0:
                self._save_state()

    def _update_competency_level(self, tool_name: str) -> None:
        """Update competency level based on usage and success rate."""
        comp = self.competencies.get(tool_name)
        if not comp:
            return

        success_rate = comp.successful_uses / max(1, comp.total_uses)

        # Determine level based on uses and success rate
        for level, threshold in [
            (ToolCompetencyLevel.EXPERT, self.COMPETENCY_THRESHOLDS[ToolCompetencyLevel.EXPERT]),
            (ToolCompetencyLevel.PROFICIENT, self.COMPETENCY_THRESHOLDS[ToolCompetencyLevel.PROFICIENT]),
            (ToolCompetencyLevel.COMPETENT, self.COMPETENCY_THRESHOLDS[ToolCompetencyLevel.COMPETENT]),
            (ToolCompetencyLevel.LEARNING, self.COMPETENCY_THRESHOLDS[ToolCompetencyLevel.LEARNING]),
        ]:
            if comp.total_uses >= threshold and success_rate >= 0.6:
                comp.level = level
                break
        else:
            comp.level = ToolCompetencyLevel.NOVICE

    def _update_optimal_parameters(self, tool_name: str, parameters: dict) -> None:
        """Update optimal parameters based on successful usage."""
        comp = self.competencies.get(tool_name)
        if not comp:
            return

        for key, value in parameters.items():
            if key not in comp.optimal_parameters:
                comp.optimal_parameters[key] = {"value": value, "count": 1, "success_rate": 1.0}
                comp._evict_optimal_parameters()
            else:
                existing = comp.optimal_parameters[key]
                if isinstance(value, (int, float)) and isinstance(existing["value"], (int, float)):
                    # Weighted average
                    total = existing["count"] + 1
                    existing["value"] = (existing["value"] * existing["count"] + value) / total
                    existing["count"] = total
                elif value == existing["value"]:
                    existing["count"] += 1

    def _extract_failure_insight(self, tool_name: str, error_message: str, parameters: dict | None) -> None:
        """Extract learning insight from a failure."""
        comp = self.competencies.get(tool_name)
        if not comp:
            return

        # Create insight from error
        insight = f"Error: {error_message[:100]}"
        if parameters:
            problematic = [k for k, v in parameters.items() if v is not None]
            if problematic:
                insight += f" (with params: {', '.join(problematic[:3])})"

        if insight not in comp.learning_insights[-10:]:
            comp.learning_insights.append(insight)
            if len(comp.learning_insights) > 50:
                comp.learning_insights = comp.learning_insights[-50:]

    def _adapt_tool_genes(self, tool_name: str, success: bool, quality: float) -> None:
        """Adapt tool-specific genes based on usage outcome."""
        if tool_name not in self.tool_genes:
            return

        genes = self.tool_genes[tool_name]
        for gene_name, gene in genes.items():
            if success:
                # Slightly move toward quality-weighted adaptation
                adjustment = (quality - 0.5) * gene.adaptation_rate * 0.1
            else:
                # Move away from current value on failure
                adjustment = -gene.adaptation_rate * 0.05

            new_value = gene.current_value + adjustment
            gene.current_value = max(gene.min_value, min(gene.max_value, new_value))
            gene.history.append((datetime.now(), gene.current_value))
            if len(gene.history) > 50:
                gene.history = gene.history[-50:]

    def get_tool_competency(self, tool_name: str) -> ToolCompetency | None:
        """Get competency record for a tool."""
        return self.competencies.get(tool_name)

    def get_all_competencies(self) -> dict[str, ToolCompetency]:
        """Get all tool competencies."""
        return self.competencies.copy()

    def get_tool_gene(self, tool_name: str, gene_name: str) -> ToolGene | None:
        """Get a specific tool gene value."""
        return self.tool_genes.get(tool_name, {}).get(gene_name)

    def get_tool_genes(self, tool_name: str) -> dict[str, ToolGene]:
        """Get all genes for a tool."""
        return self.tool_genes.get(tool_name, {})

    def suggest_tool_improvement(self, tool_name: str) -> dict | None:
        """Suggest improvements for a tool based on learning data."""
        comp = self.competencies.get(tool_name)
        if not comp or comp.total_uses < 5:
            return None

        suggestions = {
            "tool": tool_name,
            "level": comp.level.value,
            "success_rate": comp.successful_uses / max(1, comp.total_uses),
            "issues": [],
            "recommendations": [],
            "optimal_parameters": comp.optimal_parameters,
        }

        # Identify issues
        if comp.successful_uses / max(1, comp.total_uses) < 0.5:
            suggestions["issues"].append("Low success rate")
            suggestions["recommendations"].append("Consider different parameter combinations")

        if comp.avg_execution_time > 5.0:
            suggestions["issues"].append("High execution time")
            suggestions["recommendations"].append("Try simplified parameters")

        if comp.common_errors:
            top_error = max(comp.common_errors.items(), key=lambda x: x[1])
            suggestions["issues"].append(f"Common error: {top_error[0]}")
            suggestions["recommendations"].append("Review error handling for this case")

        if comp.level == ToolCompetencyLevel.NOVICE:
            suggestions["recommendations"].append("More practice needed - use during dreaming")

        return suggestions

    def get_practice_priorities(self) -> list[str]:
        """Get tools that need practice, ordered by priority.
        
        Considers:
        - Low competency level (NOVICE/LEARNING get high priority)
        - Low success rate
        - Declining performance trends
        - Time since last use (stale tools need refresh)
        """
        priorities = []
        now = datetime.now()

        for tool_name, comp in self.competencies.items():
            priority_score = 0.0
            
            # Factor 1: Competency level (lower = higher priority)
            level_weights = {
                ToolCompetencyLevel.NOVICE: 40,
                ToolCompetencyLevel.LEARNING: 30,
                ToolCompetencyLevel.COMPETENT: 15,
                ToolCompetencyLevel.PROFICIENT: 5,
                ToolCompetencyLevel.EXPERT: 2,
            }
            priority_score += level_weights.get(comp.level, 10)
            
            # Factor 2: Success rate (lower = higher priority)
            if comp.total_uses > 0:
                success_rate = comp.successful_uses / comp.total_uses
                priority_score += (1 - success_rate) * 20
            else:
                priority_score += 20  # Never used
            
            # Factor 3: Declining trends
            if comp.improvement_trends and len(comp.improvement_trends) >= 3:
                recent_trend = sum(comp.improvement_trends[-3:]) / 3
                if recent_trend < 0.4:
                    priority_score += 15  # Declining performance
            
            # Factor 4: Time since last use (stale tools)
            if comp.last_used:
                days_since_use = (now - comp.last_used).days
                if days_since_use > 7:
                    priority_score += min(days_since_use, 20)
            
            # Factor 5: Low practice sessions relative to level threshold
            expected_sessions = self.COMPETENCY_THRESHOLDS.get(comp.level, 10) // 2
            if comp.practice_sessions < expected_sessions:
                priority_score += 10

            priorities.append((tool_name, priority_score))

        priorities.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in priorities[:10]]

    async def practice_tool(self, tool_name: str, actual_execution: bool = True, tool_registry: dict | None = None) -> dict:
        """Practice a tool to improve competency (called during dreaming).
        
        Args:
            tool_name: Name of tool to practice
            actual_execution: If True (default), attempt real execution with LLM evaluation.
                             If False, skip practice and return empty result.
            tool_registry: Dict of tool name -> tool instance for actual execution (REQUIRED)
            
        Returns:
            Practice result with session info and improvements
            
        Raises:
            ValueError: If actual_execution=True but tool_registry not provided
        """
        if actual_execution and tool_registry is None:
            return {
                "tool": tool_name,
                "session": 0,
                "scenarios_practiced": 0,
                "improvements": [],
                "execution_mode": "error",
                "message": f"tool_registry is REQUIRED for actual execution of '{tool_name}'. Practice skipped.",
            }

        if not actual_execution:
            return {
                "tool": tool_name,
                "session": 0,
                "scenarios_practiced": 0,
                "improvements": [],
                "execution_mode": "skipped",
                "message": "Practice skipped - actual_execution=False. No simulated practice allowed.",
            }

        with self._lock:
            if tool_name not in self.competencies:
                self.competencies[tool_name] = ToolCompetency(tool_name=tool_name)

            comp = self.competencies[tool_name]
            comp.practice_sessions += 1
            comp.total_uses += 1

        practice_result = {
            "tool": tool_name,
            "session": self.competencies[tool_name].practice_sessions,
            "scenarios_practiced": 0,
            "improvements": [],
            "execution_mode": "actual",
        }

        practice_scenarios = self._generate_practice_scenarios(tool_name)

        for scenario in practice_scenarios[:3]:
            practice_result["scenarios_practiced"] += 1
            
            exec_result = await self._execute_practice_scenario(
                tool_name, scenario, tool_registry or {}
            )
            practice_result["improvements"].extend(exec_result.get("improvements", []))

        self._update_competency_level(tool_name)
        self._save_state()

        practice_result["new_level"] = self.competencies[tool_name].level.value
        return practice_result

    async def _execute_practice_scenario(
        self, 
        tool_name: str, 
        scenario: dict,
        tool_registry: dict
    ) -> dict:
        """Execute a practice scenario and evaluate the result.
        
        Uses LLM to evaluate tool output quality.
        """
        result = {"improvements": []}
        
        try:
            tool = tool_registry.get(tool_name)
            if not tool:
                return result
            
            # Extract parameters from scenario
            action = scenario.get("action", scenario.get("type", "default"))
            params = {k: v for k, v in scenario.items() if k not in ["type", "action"]}
            
            # Execute the tool
            output = await tool.execute(action=action, **params)
            
            # Evaluate output quality with LLM
            quality_score = await self._evaluate_output_quality(tool_name, output, scenario)
            
            # Update competency based on quality
            with self._lock:
                comp = self.competencies.get(tool_name)
                if comp:
                    success = quality_score >= 0.5
                    if success:
                        comp.successful_uses += 1
                    else:
                        comp.failed_uses += 1
                    comp.improvement_trends.append(quality_score)
                    if len(comp.improvement_trends) > 50:
                        comp.improvement_trends = comp.improvement_trends[-50:]
                    
                    if quality_score < 0.3:
                        result["improvements"].append(
                            f"Need improvement: {scenario.get('type', 'unknown')} scenario"
                        )
                    elif quality_score > 0.7:
                        result["improvements"].append(
                            f"Good performance: {scenario.get('type', 'unknown')} scenario"
                        )
            
        except Exception as e:
            logger.debug(f"Practice execution error for {tool_name}: {e}")
            with self._lock:
                comp = self.competencies.get(tool_name)
                if comp:
                    comp.failed_uses += 1
                    error_type = type(e).__name__
                    comp.common_errors[error_type] = comp.common_errors.get(error_type, 0) + 1
                    comp._evict_common_errors()

        # Both the success path and the except path used to fall off the end and
        # return None, so the caller's `exec_result.get("improvements", [])`
        # raised AttributeError on every practice run where the tool existed —
        # the only path that returned was the early-out for a missing tool.
        return result

    async def _evaluate_output_quality(
        self, 
        tool_name: str, 
        output: str, 
        scenario: dict
    ) -> float:
        """Evaluate tool output quality using LLM.
        
        Returns quality score from 0.0 to 1.0.
        """
        try:
            from core.dual_helpers import get_integration
            
            integration = get_integration()
            if not integration or not integration.backend_llm:
                return 0.5  # Default score if LLM not available
            
            prompt = f"""Evaluate the quality of this tool execution result.

Tool: {tool_name}
Scenario: {scenario.get('type', 'unknown')}
Expected outcome: {scenario.get('expected', 'successful completion')}

Tool output:
{str(output)[:1000]}

Rate the output quality as a single number from 0 to 1 where:
- 0.0-0.3: Failed or error
- 0.4-0.6: Partially successful  
- 0.7-1.0: Fully successful and high quality

Return ONLY the number, nothing else."""

            response = await integration.backend_llm.generate(prompt)
            if response:
                try:
                    score = float(response.strip())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass
            
        except Exception as e:
            logger.debug(f"Quality evaluation error: {e}")
        
        # Check for error indicators in output
        output_lower = str(output).lower()
        if "error" in output_lower or "failed" in output_lower:
            return 0.2
        elif "success" in output_lower or "created" in output_lower:
            return 0.8
        return 0.5

    def _generate_practice_scenarios(self, tool_name: str) -> list[dict]:
        """Generate practice scenarios for a tool.
        
        Returns realistic, testable scenarios based on tool type.
        """

        # Comprehensive tool-specific scenarios
        TOOL_SCENARIOS = {
            "web_search": [
                {"type": "search", "query": "Python async programming tutorial", "expected": "relevant results"},
                {"type": "search", "query": "machine learning basics", "expected": "educational content"},
            ],
            "web_fetch": [
                {"type": "fetch", "url": "https://example.com", "expected": "page content"},
            ],
            "notes": [
                {"type": "create", "title": "practice_note_test", "content": "Test content for practice", "expected": "note created"},
                {"type": "list", "expected": "notes listed"},
                {"type": "search", "query": "practice", "expected": "matching notes"},
                {"type": "update", "title": "practice_note_test", "content": "Updated test content", "expected": "note updated"},
                {"type": "delete", "title": "practice_note_test", "expected": "note deleted"},
            ],
            "reminders": [
                {"type": "create", "text": "practice reminder", "time": "in 1 hour", "expected": "reminder created"},
                {"type": "list", "expected": "reminders listed"},
            ],
            "shell": [
                {"type": "execute", "command": "echo 'practice'", "expected": "output: practice"},
                {"type": "execute", "command": "ls -la", "expected": "file listing"},
            ],
            "read_file": [
                {"type": "read", "path": "/etc/hostname", "expected": "file content"},
            ],
            "write_file": [
                {"type": "write", "path": "/tmp/jarvis_practice.txt", "content": "test", "expected": "file written"},
            ],
            "terminal_execute": [
                {"type": "execute", "command": "date", "expected": "current date"},
            ],
            "docker": [
                {"type": "list_containers", "expected": "container list"},
                {"type": "list_images", "expected": "image list"},
            ],
            "data_analysis": [
                {"type": "cache_stats", "expected": "cache statistics"},
            ],
            "list_tools": [
                {"type": "list", "expected": "tool list"},
            ],
            "get_tool_info": [
                {"type": "info", "tool_name": "notes", "expected": "tool details"},
            ],
            "spatial_calc": [
                {"type": "area", "shape": "rectangle", "dimensions": {"width": 5, "height": 3}, "expected": "area: 15"},
                {"type": "volume", "shape": "cube", "dimensions": {"side": 2}, "expected": "volume: 8"},
            ],
            "physics_sim": [
                {"type": "projectile", "params": {"velocity": 10, "angle": 45}, "expected": "trajectory data"},
            ],
            "design_calc": [
                {"type": "beam", "requirements": {"load": 1000, "span": 5}, "expected": "beam specs"},
            ],
            "sql_query": [
                {"type": "query", "query": "SELECT 1", "expected": "result: 1"},
            ],
            "statistics": [
                {"type": "describe", "data": [1, 2, 3, 4, 5], "expected": "statistics summary"},
            ],
            "security_scan": [
                {"type": "scan", "data": "test data without secrets", "expected": "clean scan"},
            ],
            "evolution_status": [
                {"type": "status", "expected": "evolution status"},
            ],
            "erdos_math": [
                {"type": "status", "expected": "erdos stats"},
                {"type": "solved", "expected": "solved problems"},
            ],
            "evolution_pathway": [
                {"type": "view", "expected": "pathway info"},
            ],
            "self_edit": [
                {"type": "list", "expected": "pending edits"},
            ],
            "sketch": [
                {"type": "list", "expected": "sketches list"},
                {"type": "stats", "expected": "sketch statistics"},
            ],
            "daydream_status": [
                {"type": "status", "expected": "daydream stats"},
            ],
            "ui_sense": [
                {"type": "sense", "expected": "UI state and visible elements"},
                {"type": "get_context", "expected": "current UI context"},
            ],
            "ui_watch": [
                {"type": "watch", "duration": "2", "expected": "UI changes observed"},
                {"type": "watch", "duration": "5", "expected": "UI activity log"},
            ],
            "ui_visualize": [
                {"type": "visualize", "expected": "UI structure visualization"},
                {"type": "tree", "expected": "UI element tree"},
            ],
            "system_sense": [
                {"focus": "all", "expected": "comprehensive system status"},
                {"focus": "resources", "expected": "CPU, memory, disk usage"},
                {"focus": "subsystems", "expected": "daydream, refine, evolution status"},
                {"focus": "processes", "expected": "running Jarvis processes"},
                {"focus": "models", "expected": "available AI models"},
            ],
            "system_watch": [
                {"thresholds": True, "expected": "resource threshold alerts"},
            ],
            "tool_create": [
                {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}, "category": "meta"},
            ],
            "tool_test": [
                {"name": "test_tool", "expected": "validation results"},
            ],
            "tool_validate": [
                {"name": "test_tool", "expected": "validation report"},
            ],
             "tool_list_sandbox": [
                {"expected": "list of sandbox tools"},
             ],
        }

        # Dynamic scenario generation for tools without predefined scenarios
        # This ensures ALL registered tools can be practiced during daydream
        from core.tool_registry import TOOL_REGISTRY, ToolCategory
        
        # Category-specific generic scenarios
        CATEGORY_SCENARIOS = {
            ToolCategory.WEB: [
                {"type": "search", "query": "test query", "expected": "results"},
                {"type": "fetch", "expected": "content retrieved"},
            ],
            ToolCategory.BROWSER: [
                {"type": "list", "expected": "browser instances"},
                {"type": "navigate", "url": "about:blank", "expected": "page loaded"},
            ],
            ToolCategory.SCREEN: [
                {"type": "capture", "expected": "screen captured"},
                {"type": "find", "text": "test", "expected": "element found"},
            ],
            ToolCategory.TERMINAL: [
                {"type": "execute", "command": "echo test", "expected": "output"},
                {"type": "status", "expected": "system status"},
            ],
            ToolCategory.SYSTEM: [
                {"type": "list", "expected": "files listed"},
                {"type": "read", "expected": "file content"},
            ],
            ToolCategory.DATA: [
                {"type": "analyze", "expected": "analysis results"},
                {"type": "query", "expected": "query results"},
            ],
            ToolCategory.MULTIMODAL: [
                {"type": "process", "expected": "processed output"},
                {"type": "analyze", "expected": "analysis"},
            ],
            ToolCategory.MEMORY: [
                {"type": "search", "expected": "results"},
                {"type": "stats", "expected": "statistics"},
            ],
            ToolCategory.SECURITY: [
                {"type": "scan", "expected": "scan results"},
                {"type": "audit", "expected": "audit report"},
            ],
            ToolCategory.HOME: [
                {"type": "status", "expected": "device status"},
                {"type": "list", "expected": "devices"},
            ],
            ToolCategory.AUTOMATION: [
                {"type": "list", "expected": "workflows"},
                {"type": "status", "expected": "status"},
            ],
            ToolCategory.SPATIAL: [
                {"type": "calculate", "expected": "calculation result"},
                {"type": "simulate", "expected": "simulation data"},
            ],
            ToolCategory.META: [
                {"type": "list", "expected": "items listed"},
                {"type": "info", "expected": "information"},
            ],
            ToolCategory.LEARNING: [
                {"type": "status", "expected": "status"},
                {"type": "list", "expected": "results"},
            ],
            ToolCategory.EVOLUTION: [
                {"type": "status", "expected": "status"},
                {"type": "list", "expected": "genes listed"},
            ],
            ToolCategory.REFINEMENT: [
                {"type": "scan", "expected": "issues found"},
                {"type": "status", "expected": "status"},
            ],
        }

        # Check if tool has predefined scenarios
        if tool_name in TOOL_SCENARIOS:
            return TOOL_SCENARIOS[tool_name]
        
        # Generate scenarios based on tool category
        if tool_name in TOOL_REGISTRY:
            tool_info = TOOL_REGISTRY[tool_name]
            category = tool_info.category
            if category in CATEGORY_SCENARIOS:
                return CATEGORY_SCENARIOS[category]
        
        # Ultimate fallback - introspect tool for common methods
        return [
            {"type": "list", "expected": f"{tool_name} results"},
            {"type": "status", "expected": f"{tool_name} status"},
            {"type": "info", "expected": f"{tool_name} information"},
        ]

    def _record_practice_quality(self, tool_name: str, scenario: dict, quality: float) -> None:
        """Record practice quality based on real execution results.
        
        This replaces the fake _simulate_practice method with actual quality recording
        from real tool execution outcomes.
        
        Args:
            tool_name: Name of tool practiced
            scenario: The scenario that was executed
            quality: Quality score from 0.0 to 1.0 based on actual execution results
        """
        with self._lock:
            comp = self.competencies.get(tool_name)
            if not comp:
                return
            
            scenario_type = scenario.get("type", "generic")
            
            comp.improvement_trends.append(quality)
            if len(comp.improvement_trends) > 50:
                comp.improvement_trends = comp.improvement_trends[-50:]
            
            if quality >= 0.5:
                comp.successful_uses += 1
            else:
                comp.failed_uses += 1
                error_key = f"{scenario_type}_error"
                comp.common_errors[error_key] = comp.common_errors.get(error_key, 0) + 1
                comp._evict_common_errors()


    def apply_competency_decay(self, days_threshold: int = 7) -> list[str]:
        """Apply competency decay for tools not used recently.
        
        Decrements competency level for stale tools to encourage refresh.
        
        Args:
            days_threshold: Days of inactivity before decay applies
            
        Returns:
            List of tools that were decayed
        """
        decayed = []
        now = datetime.now()
        
        with self._lock:
            for tool_name, comp in self.competencies.items():
                if not comp.last_used:
                    continue
                
                days_inactive = (now - comp.last_used).days
                
                if days_inactive > days_threshold:
                    # Calculate decay severity based on inactivity
                    decay_levels = days_inactive // days_threshold
                    decay_levels = min(decay_levels, 2)  # Max 2 level decay
                    
                    level_order = [
                        ToolCompetencyLevel.EXPERT,
                        ToolCompetencyLevel.PROFICIENT,
                        ToolCompetencyLevel.COMPETENT,
                        ToolCompetencyLevel.LEARNING,
                        ToolCompetencyLevel.NOVICE,
                    ]
                    
                    current_idx = level_order.index(comp.level) if comp.level in level_order else 4
                    new_idx = min(current_idx + decay_levels, 4)
                    
                    if new_idx != current_idx:
                        old_level = comp.level
                        comp.level = level_order[new_idx]
                        decayed.append(f"{tool_name}: {old_level.value} -> {comp.level.value}")
                        logger.info(f"Tool competency decayed: {tool_name} from {old_level.value} to {comp.level.value}")
        
        if decayed:
            self._save_state()
        
        return decayed

    def export_to_evolution(self) -> dict:
        """Export tool learning data for evolution system."""
        return {
            "tool_competencies": {
                name: {
                    "level": comp.level.value,
                    "success_rate": comp.successful_uses / max(1, comp.total_uses),
                    "quality": comp.avg_quality,
                    "uses": comp.total_uses,
                }
                for name, comp in self.competencies.items()
            },
            "tool_genes": {
                tool: {
                    gene: g.current_value
                    for gene, g in genes.items()
                }
                for tool, genes in self.tool_genes.items()
            },
            "learning_insights": {
                name: comp.learning_insights[-5:]
                for name, comp in self.competencies.items()
                if comp.learning_insights
            },
            "practice_priorities": self.get_practice_priorities(),
        }

    def export_to_skills_md(self, skills_path: Path | None = None) -> dict:
        """Export learned tool patterns to SKILLS.md for persistent learning.
        
        This allows Jarvis to remember how to use tools better over time by:
        - Recording optimal parameters discovered through usage
        - Documenting common errors and their solutions
        - Tracking successful patterns for each tool
        
        Args:
            skills_path: Path to SKILLS.md (defaults to JARVIS_DIR/SKILLS.md)
        
        Returns:
            Summary of what was updated
        """
        if skills_path is None:
            skills_path = Path(__file__).parent.parent / "SKILLS.md"
        
        if not skills_path.exists():
            logger.warning(f"SKILLS.md not found at {skills_path}")
            return {"error": "SKILLS.md not found"}
        
        try:
            with open(skills_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read SKILLS.md: {e}")
            return {"error": safe_error_message(e)}
        
        updates = {
            "tools_updated": 0,
            "patterns_added": 0,
            "errors_documented": 0,
        }
        
        # Check if LEARNED TOOL PATTERNS section exists
        section_marker = "## Learned Tool Patterns"
        if section_marker not in content:
            # Add the section
            learned_section = f"\n\n{section_marker}\n\n"
            learned_section += " Jarvis learns optimal patterns from successful tool usage.\n"
            learned_section += " This section is automatically updated based on experience.\n\n"
            content += learned_section
            logger.info("Added LEARNED TOOL PATTERNS section to SKILLS.md")
        
        # Find the section
        section_start = content.find(section_marker)
        next_section = content.find("\n## ", section_start + 1)
        if next_section == -1:
            next_section = len(content)
        
        existing_patterns = content[section_start:next_section]
        
        # Build learned patterns
        new_patterns = []
        for tool_name, comp in self.competencies.items():
            if comp.total_uses < 5:
                continue  # Not enough data
            
            if comp.level.value in ["novice", "learning"]:
                continue  # Not proficient enough
            
            success_rate = comp.successful_uses / max(1, comp.total_uses)
            if success_rate > 1.0:
                continue  # Invalid data - success can't exceed total
            if success_rate < 0.5:
                continue  # Poor success rate
            
            # Build pattern entry
            pattern_lines = [f"\n### {tool_name}"]
            pattern_lines.append(f"**Competency:** {comp.level.value}")
            pattern_lines.append(f"**Success Rate:** {success_rate:.1%} ({comp.successful_uses}/{comp.total_uses} uses)")
            
            # Optimal parameters discovered
            if comp.optimal_parameters:
                pattern_lines.append("\n**Optimal Parameters:**")
                for param, data in list(comp.optimal_parameters.items())[:5]:
                    if isinstance(data.get("value"), (int, float)):
                        pattern_lines.append(f"- `{param}`: {data['value']:.2f} (used {data['count']} times)")
                    else:
                        pattern_lines.append(f"- `{param}`: {data['value']} (used {data['count']} times)")
                updates["patterns_added"] += 1
            
            # Common errors and solutions
            if comp.common_errors:
                pattern_lines.append("\n**Common Errors & Solutions:**")
                for error, count in list(sorted(comp.common_errors.items(), key=lambda x: -x[1])[:3]):
                    pattern_lines.append(f"- `{error}` (occurred {count} times)")
                    # Add solution hint if available
                    if comp.learning_insights:
                        for insight in comp.learning_insights[-3:]:
                            if error.split('_')[0] in insight.lower():
                                pattern_lines.append(f"  - Fix: {insight}")
                                break
                updates["errors_documented"] += 1
            
            # Learning insights
            if comp.learning_insights:
                pattern_lines.append("\n**Learning Insights:**")
                for insight in comp.learning_insights[-3:]:
                    pattern_lines.append(f"- {insight}")
            
            new_patterns.append('\n'.join(pattern_lines))
        
        if new_patterns:
            pattern_text = '\n'.join(new_patterns)
            
            tools_added = 0
            for pattern in new_patterns:
                tool_match = pattern.split('\n')[0].replace('### ', '')
                if f"### {tool_match}" not in existing_patterns:
                    tools_added += 1
            
            if tools_added > 0:
                content[section_start:next_section]
                content = content[:next_section] + pattern_text + '\n' + content[next_section:]
                updates["tools_updated"] = tools_added
        
        # Write updated content
        if updates["tools_updated"] > 0 or updates["patterns_added"] > 0:
            try:
                with open(skills_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Updated SKILLS.md: {updates}")
            except Exception as e:
                logger.error(f"Failed to write SKILLS.md: {e}")
                return {"error": safe_error_message(e)}
        
        return updates

    def export_to_memory(self) -> dict:
        """Export tool learning data for memory system."""
        memories = []

        for tool_name, comp in self.competencies.items():
            if comp.total_uses > 0:
                memories.append({
                    "type": "tool_competency",
                    "content": f"Tool {tool_name}: {comp.level.value}, {comp.total_uses} uses, {comp.successful_uses}/{comp.total_uses} success",
                    "importance": min(1.0, comp.total_uses / 100),
                    "metadata": {
                        "tool": tool_name,
                        "level": comp.level.value,
                        "success_rate": comp.successful_uses / max(1, comp.total_uses),
                    },
                })

            if comp.learning_insights:
                for insight in comp.learning_insights[-3:]:
                    memories.append({
                        "type": "tool_insight",
                        "content": f"{tool_name}: {insight}",
                        "importance": 0.6,
                        "metadata": {"tool": tool_name},
                    })

        return {"memories": memories}

    async def start(self) -> None:
        """Start the tool learning engine."""
        self._running = True
        logger.info("Tool learning engine started")

    async def stop(self) -> None:
        """Stop the tool learning engine."""
        self._running = False
        self._save_state()
        logger.info("Tool learning engine stopped")


_tool_learning_engine: ToolLearningEngine | None = None
_tool_learning_lock = threading.Lock()


def get_tool_learning_engine(storage_path: Path | None = None) -> ToolLearningEngine:
    """Get or create the singleton tool learning engine."""
    global _tool_learning_engine
    with _tool_learning_lock:
        if _tool_learning_engine is None:
            _tool_learning_engine = ToolLearningEngine(storage_path)
        return _tool_learning_engine
