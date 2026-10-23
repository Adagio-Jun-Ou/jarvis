"""Composable Workflow Stages for Jarvis Subsystems.

This module provides Stage adapters that wrap background subsystems
(ErdosEngine, DaydreamEngine, MusicCreationEngine, etc.) into the composable
workflow system, enabling pipeline composition with `|`, `||`, `>>` operators.

Key features:
- SubsystemStage base class for wrapping any background engine
- ErdosProblemStage: Math problem exploration
- MusicCompositionStage: Music creation with acoustic tokens
- DaydreamStage: Idle-time learning and creativity
- InsightStage: Knowledge processing and insight generation
- PFCObserveStage: Prefrontal cortex observation
- Pre-built pipelines for common workflows

Example:
    # Single math problem exploration
    result = await ErdosProblemStage().execute(None, context)
    
    # Composed pipeline: math -> insight -> knowledge graph
    pipeline = (
        Pipeline("math_insight")
        | ErdosProblemStage()
        | InsightStage()
        | KnowledgeGraphStage()
    )
    result = await pipeline.execute(None)
    
    # Parallel exploration
    parallel_math = parallel(
        ErdosProblemStage(category="ramsey_theory"),
        ErdosProblemStage(category="number_theory"),
        ErdosProblemStage(category="combinatorics"),
    )
"""
import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Callable

from core.helpers import safe_error_message
from core.workflow_shell import (
    Stage,
    StageResult,
    StageStatus,
    PipelineContext,
    Pipeline,
    ParallelPipeline,
    parallel as create_parallel,
    get_workflow_shell,
)

logger = logging.getLogger(__name__)


class SubsystemStage(Stage[dict, dict]):
    """Base class for subsystem stages.
    
    Handles common patterns:
    - Getting singleton subsystem instances
    - Error handling and status reporting
    - Activity tracking for visualization
    - PFC observation of results
    """
    
    name: str = "subsystem"
    description: str = "Base subsystem stage"
    
    def __init__(
        self,
        observe_with_pfc: bool = True,
        timeout_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.observe_with_pfc = observe_with_pfc
        self.timeout_seconds = timeout_seconds
        self.metadata = metadata or {}
        self._subsystem = None
        self._lock = threading.Lock()
    
    async def execute(self, input_data: dict | None, context: PipelineContext) -> StageResult[dict]:
        """Execute the subsystem operation."""
        start_time = datetime.now()
        
        try:
            self._subsystem = self._get_subsystem()
            
            if self._subsystem is None:
                return StageResult(
                    value={"error": f"{self.name} subsystem not available"},
                    status=StageStatus.FAILED,
                    metadata={"reason": "subsystem_unavailable"},
                )
            
            result = await self._execute_subsystem(input_data, context)
            
            if self.observe_with_pfc and result:
                await self._observe_with_pfc(result, context)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return StageResult(
                value=result,
                status=StageStatus.COMPLETED,
                duration_ms=duration_ms,
                metadata=self.metadata,
            )
            
        except asyncio.TimeoutError:
            return StageResult(
                error=asyncio.TimeoutError(f"{self.name} timed out after {self.timeout_seconds}s"),
                status=StageStatus.FAILED,
                metadata={"timeout": True, "seconds": self.timeout_seconds},
            )
        except Exception as e:
            logger.error(f"{self.name} stage failed: {e}")
            return StageResult(
                error=e,
                status=StageStatus.FAILED,
                metadata={"error": safe_error_message(e)},
            )
    
    def _get_subsystem(self) -> Any:
        """Get the subsystem instance. Override in subclasses."""
        return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        """Execute the subsystem operation. Override in subclasses."""
        return {}
    
    async def _observe_with_pfc(self, result: dict, context: PipelineContext) -> None:
        """Observe result with Prefrontal Cortex and context cascade."""
        try:
            content = result.get("insight") or result.get("result") or str(result)[:500]
            
            # Observe via context cascade for full integration
            from core.context_cascade import observe_activity
            try:
                cascade = observe_activity(
                    trigger_type="subsystem_event",
                    content=content[:500],
                    source=self.name,
                )
                context.shared_state["_cascade"] = cascade
            except Exception as e:
                logger.debug(f"Context cascade observation failed: {e}")
            
            # Also observe directly with PFC for immediate meta-reasoning
            from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType
            
            pfc = get_prefrontal_cortex()
            if pfc and pfc._running:
                await pfc.observe(
                    content=content,
                    concept_type=ConceptType.SUBSYSTEM_EVENT,
                    source=self.name,
                    metadata={"pipeline_id": context.workflow_id, **self.metadata},
                )
        except Exception as e:
            logger.debug(f"PFC observation failed: {e}")


class ErdosProblemStage(SubsystemStage):
    """Stage for Erdős-style math problem exploration.
    
    Generates and attempts to solve a mathematical problem.
    Can specify category and difficulty.
    
    Example:
        stage = ErdosProblemStage(category="ramsey_theory")
        result = await stage.execute(None, context)
    """
    
    name = "erdos_problem"
    description = "Explore and solve an Erdős-style mathematical problem"
    
    def __init__(
        self,
        category: str | None = None,
        difficulty: str | None = None,
        observe_with_pfc: bool = True,
        timeout_seconds: float = 120.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.category = category
        self.difficulty = difficulty
        self.metadata["category"] = category or "random"
        self.metadata["difficulty"] = difficulty or "random"
    
    def _get_subsystem(self):
        try:
            from core.erdos_engine import get_erdos_engine
            return get_erdos_engine()
        except Exception as e:
            logger.debug(f"Failed to get ErdosEngine: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        from core.erdos_engine import ProblemCategory, ProblemDifficulty
        
        if self.category:
            try:
                ProblemCategory(self.category)
            except ValueError:
                pass
        
        if self.difficulty:
            try:
                ProblemDifficulty[self.difficulty.upper()]
            except (KeyError, ValueError):
                pass
        
        if hasattr(self._subsystem, 'daydream_step'):
            result = await self._subsystem.daydream_step()
            return result or {}
        
        return {"status": "error", "error": "daydream_step not available"}


class MusicCompositionStage(SubsystemStage):
    """Stage for autonomous music composition.
    
    Creates compositions with acoustic tokens for PFC learning.
    
    Example:
        stage = MusicCompositionStage(style="baroque", key="C_major")
        result = await stage.execute(None, context)
    """
    
    name = "music_composition"
    description = "Generate classical music composition with acoustic tokens"
    
    def __init__(
        self,
        style: str = "classical",
        key: str = "C_major",
        measures: int = 16,
        tempo: int = 120,
        observe_with_pfc: bool = True,
        timeout_seconds: float = 180.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.style = style
        self.key = key
        self.measures = measures
        self.tempo = tempo
        self.metadata.update({
            "style": style,
            "key": key,
            "measures": measures,
            "tempo": tempo,
        })
    
    def _get_subsystem(self):
        try:
            from core.music_creation_engine import get_music_creation_engine
            return get_music_creation_engine()
        except Exception as e:
            logger.debug(f"Failed to get MusicCreationEngine: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        import random
        
        if hasattr(self._subsystem, 'create_composition'):
            style = self.style
            key = self.key
            
            if style == "random":
                style = random.choice(["baroque", "classical", "romantic"])
            if key == "random":
                key = random.choice(["C_major", "G_major", "D_major", "A_minor", "E_minor"])
            
            result = await self._subsystem.create_composition(
                style=style,
                key=key,
                measures=self.measures,
                tempo=self.tempo,
            )
            
            if result:
                return {
                    "composition_id": result.get("id"),
                    "style": style,
                    "key": key,
                    "has_audio": result.get("has_audio", False),
                    "measures": result.get("measures", []),
                    "tokens": result.get("tokens", []),
                    "insight": f"Composed {style} piece in {key}",
                }
        
        return {"status": "composed", "style": self.style, "key": self.key}


class DaydreamExplorationStage(SubsystemStage):
    """Stage for idle-time learning and creativity.
    
    Runs a daydream cycle for exploration.
    
    Example:
        stage = DaydreamExplorationStage(mode="math")
        result = await stage.execute(None, context)
    """
    
    name = "daydream_exploration"
    description = "Run idle-time learning and creativity exploration"
    
    def __init__(
        self,
        mode: str = "auto",
        observe_with_pfc: bool = True,
        timeout_seconds: float = 60.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.mode = mode
        self.metadata["mode"] = mode
    
    def _get_subsystem(self):
        try:
            from core.daydream_engine import get_daydream_engine
            return get_daydream_engine()
        except Exception as e:
            logger.debug(f"Failed to get DaydreamEngine: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        if hasattr(self._subsystem, 'can_daydream') and not self._subsystem.can_daydream():
            return {"status": "skipped", "reason": "system busy"}
        
        try:
            if self.mode == "math":
                from core.erdos_engine import get_erdos_engine
                erdos = get_erdos_engine()
                if erdos:
                    result = await erdos.daydream_step()
                    return result or {"status": "completed"}
            elif self.mode == "music":
                result = await self._subsystem._music_exploration()
                return {"status": "completed", "mode": "music"}
            else:
                pass
            
            return {"status": "completed", "mode": self.mode}
            
        except Exception as e:
            return {"status": "error", "error": safe_error_message(e)}


class InsightProcessingStage(SubsystemStage):
    """Stage for processing insights and storing in knowledge graph.
    
    Takes input from previous stage and processes as an insight.
    
    Example:
        pipeline = ErdosProblemStage() | InsightProcessingStage()
    """
    
    name = "insight_processing"
    description = "Process insights and store in knowledge graph"
    
    def __init__(
        self,
        source: str = "workflow",
        observe_with_pfc: bool = True,
        timeout_seconds: float = 30.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.source = source
        self.metadata["source"] = source
    
    def _get_subsystem(self):
        try:
            from core.insight_bridge import get_insight_bridge
            return get_insight_bridge()
        except Exception as e:
            logger.debug(f"Failed to get InsightBridge: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        if not input_data:
            return {"status": "skipped", "reason": "no input data"}
        
        insight_text = None
        if isinstance(input_data, dict):
            insight_text = input_data.get("insight") or input_data.get("result", "")
        elif isinstance(input_data, str):
            insight_text = input_data
        else:
            insight_text = str(input_data)
        
        if not insight_text:
            return {"status": "skipped", "reason": "no insight text"}
        
        if hasattr(self._subsystem, 'process_insight'):
            try:
                result = self._subsystem.process_insight(
                    raw_insight=insight_text[:500],
                    source_problem=self.metadata.get("problem", "workflow"),
                    source_category=self.source,
                    strategy_used=input_data.get("strategy") if isinstance(input_data, dict) else None,
                    problem_solved=input_data.get("success", False) if isinstance(input_data, dict) else False,
                )
                
                if result:
                    return {
                        "status": "processed",
                        "insight_id": result.id if hasattr(result, 'id') else None,
                        "insight_type": result.insight_type.value if hasattr(result, 'insight_type') else None,
                    }
            except Exception as e:
                logger.debug(f"Insight processing failed: {e}")
        
        return {"status": "processed", "insight": insight_text[:200]}


class KnowledgeGraphStage(SubsystemStage):
    """Stage for knowledge graph operations.
    
    Can add nodes, query connections, or get PageRank.
    
    Example:
        stage = KnowledgeGraphStage(operation="add_nodes")
        result = await stage.execute({"nodes": [...]}, context)
    """
    
    name = "knowledge_graph"
    description = "Add nodes or query the knowledge graph"
    
    def __init__(
        self,
        operation: str = "add_nodes",
        observe_with_pfc: bool = True,
        timeout_seconds: float = 30.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.operation = operation
        self.metadata["operation"] = operation
    
    def _get_subsystem(self):
        try:
            from core.knowledge_graph import get_knowledge_graph
            return get_knowledge_graph()
        except Exception as e:
            logger.debug(f"Failed to get KnowledgeGraph: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        if self._subsystem is None:
            return {"status": "error", "error": "knowledge graph unavailable"}
        
        try:
            if self.operation == "add_nodes" and isinstance(input_data, dict):
                nodes = input_data.get("nodes", [])
                for node in nodes:
                    if hasattr(self._subsystem, 'add_node'):
                        self._subsystem.add_node(
                            node.get("id"),
                            node.get("type"),
                            node.get("properties", {}),
                        )
                return {"status": "added", "count": len(nodes)}
            
            elif self.operation == "get_pagerank":
                if hasattr(self._subsystem, 'get_pagerank'):
                    pagerank = self._subsystem.get_pagerank()
                    return {"status": "pagerank", "top_nodes": list(pagerank.items())[:10]}
            
            elif self.operation == "get_communities":
                if hasattr(self._subsystem, 'get_community_structure'):
                    communities = self._subsystem.get_community_structure()
                    return {"status": "communities", "count": len(communities)}
            
            else:
                return {"status": "unknown_operation", "operation": self.operation}
                
        except Exception as e:
            return {"status": "error", "error": safe_error_message(e)}
        
        return {"status": "completed"}


class PFCObserveStage(SubsystemStage):
    """Stage for observing content with Prefrontal Cortex.
    
    Observes arbitrary content as a concept type.
    
    Example:
        stage = PFCObserveStage(concept_type="INSIGHT")
        result = await stage.execute({"content": "...", "source": "workflow"}, context)
    """
    
    name = "pfc_observe"
    description = "Observe content with Prefrontal Cortex for LCM learning"
    
    def __init__(
        self,
        concept_type: str = "SUBSYSTEM_EVENT",
        observe_with_pfc: bool = True,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.concept_type = concept_type
        self.metadata["concept_type"] = concept_type
    
    def _get_subsystem(self):
        try:
            from core.prefrontal_cortex import get_prefrontal_cortex
            return get_prefrontal_cortex()
        except Exception as e:
            logger.debug(f"Failed to get PFC: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        if self._subsystem is None or not self._subsystem._running:
            return {"status": "skipped", "reason": "PFC not running"}
        
        content = None
        source = "workflow"
        
        if isinstance(input_data, dict):
            content = input_data.get("content") or input_data.get("insight") or input_data.get("result", "")
            source = input_data.get("source", "workflow")
        elif isinstance(input_data, str):
            content = input_data
        else:
            content = str(input_data)
        
        if not content:
            return {"status": "skipped", "reason": "no content"}
        
        try:
            from core.prefrontal_cortex import ConceptType
            
            concept_map = {
                "INSIGHT": ConceptType.INSIGHT,
                "TOOL_CALL": ConceptType.TOOL_CALL,
                "MEMORY_TOKEN": ConceptType.MEMORY_TOKEN,
                "MUSIC_TOKEN": ConceptType.MUSIC_TOKEN,
                "SUBSYSTEM_EVENT": ConceptType.SUBSYSTEM_EVENT,
                "USER_INPUT": ConceptType.USER_INPUT,
            }
            
            concept_enum = concept_map.get(self.concept_type, ConceptType.SUBSYSTEM_EVENT)
            
            await self._subsystem.observe(
                content=str(content)[:1000],
                concept_type=concept_enum,
                source=source,
                metadata={"pipeline_id": context.workflow_id},
            )
            
            return {"status": "observed", "content_length": len(str(content))}
            
        except Exception as e:
            return {"status": "error", "error": safe_error_message(e)}


class ToolPracticeStage(SubsystemStage):
    """Stage for practicing tools during idle time.
    
    Practicing tool competency improvement.
    
    Example:
        stage = ToolPracticeStage(tools=["web_search", "read_file"])
        result = await stage.execute(None, context)
    """
    
    name = "tool_practice"
    description = "Practice tools to improve competency"
    
    def __init__(
        self,
        tools: list[str] | None = None,
        count: int = 3,
        observe_with_pfc: bool = True,
        timeout_seconds: float = 60.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
        self.tools = tools
        self.count = count
        self.metadata["tool_count"] = count
    
    def _get_subsystem(self):
        try:
            from core.tool_learning import get_tool_learning_engine
            return get_tool_learning_engine()
        except Exception as e:
            logger.debug(f"Failed to get ToolLearningEngine: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        if self._subsystem is None:
            return {"status": "error", "error": "tool learning engine unavailable"}
        
        try:
            
            tools_to_practice = self.tools
            
            if not tools_to_practice:
                if hasattr(self._subsystem, 'get_practice_priorities'):
                    tools_to_practice = self._subsystem.get_practice_priorities()[:self.count]
                else:
                    tools_to_practice = ["web_search", "read_file", "list_files"]
            
            results = []
            for tool_name in tools_to_practice[:self.count]:
                try:
                    if hasattr(self._subsystem, 'practice_tool'):
                        result = await self._subsystem.practice_tool(tool_name)
                        results.append({
                            "tool": tool_name,
                            "status": "practiced",
                            "result": result,
                        })
                    else:
                        results.append({
                            "tool": tool_name,
                            "status": "skipped",
                            "reason": "practice_tool not available",
                        })
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "status": "error",
                        "error": safe_error_message(e),
                    })
            
            return {
                "status": "completed",
                "practiced": len([r for r in results if r.get("status") == "practiced"]),
                "results": results,
            }
            
        except Exception as e:
            return {"status": "error", "error": safe_error_message(e)}


class VerificationStage(SubsystemStage):
    """Stage for verifying solutions during daydream.
    
    Takes a proposed solution from input and verifies it.
    
    Example:
        pipeline = ErdosProblemStage() | VerificationStage()
    """
    
    name = "verification"
    description = "Verify proposed solutions"
    
    def __init__(
        self,
        observe_with_pfc: bool = True,
        timeout_seconds: float = 120.0,
    ):
        super().__init__(observe_with_pfc, timeout_seconds)
    
    def _get_subsystem(self):
        try:
            from core.daydream_verification import get_verifier
            return get_verifier()
        except Exception as e:
            logger.debug(f"Failed to get Verifier: {e}")
            return None
    
    async def _execute_subsystem(self, input_data: dict | None, context: PipelineContext) -> dict:
        if self._subsystem is None:
            return {"status": "skipped", "reason": "verifier unavailable"}
        
        if not input_data or not isinstance(input_data, dict):
            return {"status": "skipped", "reason": "no solution to verify"}
        
        try:
            solution = getattr(self._subsystem, 'submit_solution', None)
            if solution:
                submitted = solution(
                    problem_type=input_data.get("category", "unknown"),
                    problem_statement=input_data.get("statement", ""),
                    proposed_solution=input_data.get("result", ""),
                    insight=input_data.get("insight", ""),
                )
                
                verified = await self._subsystem.verify(submitted)
                
                return {
                    "status": "verified",
                    "verification_status": verified.value if hasattr(verified, 'value') else str(verified),
                    "confidence": submitted.confidence_score if hasattr(submitted, 'confidence_score') else 0.0,
                }
            
            return {"status": "completed"}
            
        except Exception as e:
            return {"status": "error", "error": safe_error_message(e)}


# Pre-built composite pipelines
def create_math_exploration_pipeline() -> Pipeline:
    """Create a pipeline for math exploration with insight processing.
    
    Pipeline: ErdosProblem -> InsightProcessing -> KnowledgeGraph
    
    Returns:
        Composable pipeline ready for execution
    """
    return (
        Pipeline("math_exploration")
        .add_stage(ErdosProblemStage())
        .add_stage(InsightProcessingStage(source="erdos"))
        .add_stage(KnowledgeGraphStage(operation="add_nodes"))
    )


def create_music_creation_pipeline(
    style: str = "classical",
    key: str = "C_major",
) -> Pipeline:
    """Create a pipeline for music composition with PFC observation.
    
    Pipeline: MusicComposition -> PFCObserve
    
    Args:
        style: Musical style (baroque, classical, romantic)
        key: Musical key (C_major, A_minor, etc.)
    
    Returns:
        Composable pipeline ready for execution
    """
    return (
        Pipeline("music_creation")
        .add_stage(MusicCompositionStage(style=style, key=key))
        .add_stage(PFCObserveStage(concept_type="MUSIC_TOKEN"))
    )


def create_tool_practice_pipeline(tools: list[str] | None = None) -> Pipeline:
    """Create a pipeline for tool practice with competency tracking.
    
    Pipeline: ToolPractice -> InsightProcessing
    
    Args:
        tools: List of tools to practice (optional)
    
    Returns:
        Composable pipeline ready for execution
    """
    return (
        Pipeline("tool_practice")
        .add_stage(ToolPracticeStage(tools=tools))
        .add_stage(InsightProcessingStage(source="tool_learning"))
    )


def create_parallel_exploration_pipeline(
    categories: list[str] | None = None,
) -> ParallelPipeline:
    """Create a parallel pipeline for exploring multiple problem categories.
    
    Runs multiple ErdosProblemStage instances in parallel.
    
    Args:
        categories: Problem categories to explore (default: ramsey, number_theory, graph_theory)
    
    Returns:
        Parallel pipeline ready for execution
    """
    if categories is None:
        categories = ["ramsey_theory", "number_theory", "graph_theory"]
    
    stages = [ErdosProblemStage(category=cat) for cat in categories]
    return create_parallel(*stages)


def create_daydream_workflow_pipeline() -> Pipeline:
    """Create a full daydream workflow pipeline.
    
    Pipeline: DaydreamExploration -> InsightProcessing -> KnowledgeGraph -> PFCObserve
    
    Returns:
        Composable pipeline for complete daydream cycle
    """
    return (
        Pipeline("daydream_workflow")
        .add_stage(DaydreamExplorationStage())
        .add_stage(InsightProcessingStage(source="daydream"))
        .add_stage(KnowledgeGraphStage(operation="add_nodes"))
        .add_stage(PFCObserveStage(concept_type="INSIGHT"))
    )


# SubsystemStageRegistry - maps stage names to classes
STAGE_REGISTRY: dict[str, type[SubsystemStage]] = {
    "erdos_problem": ErdosProblemStage,
    "music_composition": MusicCompositionStage,
    "daydream_exploration": DaydreamExplorationStage,
    "insight_processing": InsightProcessingStage,
    "knowledge_graph": KnowledgeGraphStage,
    "pfc_observe": PFCObserveStage,
    "tool_practice": ToolPracticeStage,
    "verification": VerificationStage,
}


PIPELINE_REGISTRY: dict[str, Callable[[], Pipeline]] = {
    "math_exploration": create_math_exploration_pipeline,
    "music_creation": create_music_creation_pipeline,
    "tool_practice": create_tool_practice_pipeline,
    "parallel_exploration": create_parallel_exploration_pipeline,
    "daydream_workflow": create_daydream_workflow_pipeline,
}


def register_subsystem_pipelines() -> None:
    """Register all subsystem pipelines with the WorkflowShell."""
    shell = get_workflow_shell()
    
    for name, pipeline_factory in PIPELINE_REGISTRY.items():
        try:
            pipeline = pipeline_factory()
            if isinstance(pipeline, Pipeline):
                shell.register(name, pipeline)
            logger.info(f"Registered pipeline: {name}")
        except Exception as e:
            logger.warning(f"Failed to register pipeline {name}: {e}")


def create_stage(
    stage_name: str,
    **kwargs,
) -> SubsystemStage | None:
    """Create a stage by name from the registry.
    
    Args:
        stage_name: Name of the stage type
        **kwargs: Arguments to pass to stage constructor
    
    Returns:
        Stage instance or None if not found
    """
    stage_class = STAGE_REGISTRY.get(stage_name)
    if stage_class:
        return stage_class(**kwargs)
    return None


def create_pipeline(
    pipeline_name: str,
    **kwargs,
) -> Pipeline | None:
    """Create a pipeline by name from the registry.

    Counterpart to create_stage(), for the pre-built subsystem pipelines
    registered in PIPELINE_REGISTRY.

    Args:
        pipeline_name: Name of the registered pipeline
        **kwargs: Arguments to pass to the pipeline factory

    Returns:
        Pipeline instance or None if not found
    """
    pipeline_factory = PIPELINE_REGISTRY.get(pipeline_name)
    if pipeline_factory:
        return pipeline_factory(**kwargs)
    return None


def run_pipeline(
    pipeline_name: str,
    input_data: dict | None = None,
    context: PipelineContext | None = None,
) -> "asyncio.Future":
    """Run a registered pipeline by name.
    
    Args:
        pipeline_name: Name of registered pipeline
        input_data: Input data for pipeline
        context: Optional pipeline context
    
    Returns:
        Future that resolves to StageResult
    """
    shell = get_workflow_shell()
    
    async def _run():
        return await shell.execute(pipeline_name, input_data, context)
    
    loop = asyncio.get_running_loop()
    return loop.create_task(_run())


def compose_subsystem_workflow(
    *stages: SubsystemStage,
    name: str = "custom_workflow",
) -> Pipeline:
    """Compose a custom workflow from subsystem stages.
    
    Args:
        *stages: Stage instances to compose
        name: Name for the pipeline
    
    Returns:
        Composed pipeline
    
    Example:
        pipeline = compose_subsystem_workflow(
            ErdosProblemStage(category="ramsey_theory"),
            VerificationStage(),
            InsightProcessingStage(source="verification"),
        )
    """
    pipeline = Pipeline(name)
    for stage in stages:
        pipeline.add_stage(stage)
    return pipeline


# Integration with Agentic Orchestrator
def create_deployment_workflow(
    workflow_type: str,
    config: dict[str, Any] | None = None,
) -> Pipeline:
    """Create a workflow suitable for agentic deployment.
    
    Args:
        workflow_type: Type of workflow (math, music, tools, daydream)
        config: Configuration options
    
    Returns:
        Pipeline ready for deployment
    """
    config = config or {}
    
    workflows = {
        "math": create_math_exploration_pipeline,
        "music": lambda: create_music_creation_pipeline(
            style=config.get("style", "classical"),
            key=config.get("key", "C_major"),
        ),
        "tools": lambda: create_tool_practice_pipeline(
            tools=config.get("tools"),
        ),
        "daydream": create_daydream_workflow_pipeline,
    }
    
    factory = workflows.get(workflow_type)
    if factory:
        return factory()
    
    return create_daydream_workflow_pipeline()


# Module initialization
_subsystem_pipelines_registered = False
_registration_lock = threading.Lock()


def ensure_pipelines_registered() -> None:
    """Ensure pipelines are registered (thread-safe)."""
    global _subsystem_pipelines_registered
    
    if _subsystem_pipelines_registered:
        return
    
    with _registration_lock:
        if _subsystem_pipelines_registered:
            return
        
        try:
            register_subsystem_pipelines()
            _subsystem_pipelines_registered = True
        except Exception as e:
            logger.error(f"Failed to register subsystem pipelines: {e}")


# Auto-register on import in background systems
def init_subsystem_workflows() -> None:
    """Initialize subsystem workflows for background processing.
    
    Called during Jarvis startup to enable composable workflows
    for subsystems like ErdosEngine, MusicCreationEngine, etc.
    """
    ensure_pipelines_registered()
    logger.info("Subsystem workflows initialized")


# Convenience exports
__all__ = [
    "SubsystemStage",
    "ErdosProblemStage",
    "MusicCompositionStage",
    "DaydreamExplorationStage",
    "InsightProcessingStage",
    "KnowledgeGraphStage",
    "PFCObserveStage",
    "ToolPracticeStage",
    "VerificationStage",
    "STAGE_REGISTRY",
    "PIPELINE_REGISTRY",
    "create_math_exploration_pipeline",
    "create_music_creation_pipeline",
    "create_tool_practice_pipeline",
    "create_parallel_exploration_pipeline",
    "create_daydream_workflow_pipeline",
    "create_stage",
    "create_pipeline",
    "run_pipeline",
    "compose_subsystem_workflow",
    "create_deployment_workflow",
    "register_subsystem_pipelines",
    "ensure_pipelines_registered",
    "init_subsystem_workflows",
]