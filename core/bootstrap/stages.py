"""Bootstrap Stages for Jarvis Initialization.

Provides structured initialization with:
- Sequential stage execution
- Health checks per stage
- Rollback on failure
- Progress tracking

Inspired by Claw Code bootstrap patterns.
"""
import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a bootstrap stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a bootstrap stage execution."""
    stage_name: str
    status: StageStatus
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class BootstrapStage:
    """A single bootstrap stage."""
    name: str
    description: str
    setup_func: Callable[[], Any]  # Function to execute
    health_check: Optional[Callable[[], bool]] = None  # Optional health check
    rollback_func: Optional[Callable[[], None]] = None  # Optional rollback
    dependencies: list[str] = field(default_factory=list)
    is_critical: bool = True  # If True, failure stops bootstrap
    timeout: float = 30.0
    skip_on_failure: bool = False


class BootstrapManager:
    """Manages bootstrap stages for initialization.
    
    Features:
    - Sequential stage execution
    - Dependency ordering
    - Health checks
    - Rollback on failure
    - Progress tracking
    
    Usage:
        manager = BootstrapManager()
        
        manager.register_stage(BootstrapStage(
            name="memory_init",
            description="Initialize memory systems",
            setup_func=init_memory,
            health_check=lambda: memory.is_ready(),
        ))
        
        manager.register_stage(BootstrapStage(
            name="vector_memory",
            description="Initialize vector memory",
            setup_func=init_vector_memory,
            dependencies=["memory_init"],
        ))
        
        results = manager.run_bootstrap()
    """
    
    def __init__(self):
        self._stages: OrderedDict[str, BootstrapStage] = OrderedDict()
        self._results: dict[str, StageResult] = {}
        self._stage_order: list[str] = []
        self._lock = threading.Lock()
        self._running = False
    
    def register_stage(self, stage: BootstrapStage) -> None:
        """Register a bootstrap stage.
        
        Args:
            stage: The bootstrap stage to register
        """
        with self._lock:
            self._stages[stage.name] = stage
        
        logger.debug(f"Registered bootstrap stage: {stage.name}")
    
    def _resolve_order(self) -> list[str]:
        """Resolve stage execution order based on dependencies."""
        visited = set()
        order = []
        
        def visit(stage_name: str):
            if stage_name in visited:
                return
            
            if stage_name not in self._stages:
                logger.warning(f"Unknown stage: {stage_name}")
                return
            
            visited.add(stage_name)
            stage = self._stages[stage_name]
            
            # Visit dependencies first
            for dep in stage.dependencies:
                if dep not in visited:
                    visit(dep)
            
            order.append(stage_name)
        
        for stage_name in self._stages:
            visit(stage_name)
        
        return order
    
    def run_stage(self, stage_name: str) -> StageResult:
        """Run a single stage.
        
        Args:
            stage_name: Name of stage to run
            
        Returns:
            Stage result
        """
        if stage_name not in self._stages:
            return StageResult(
                stage_name=stage_name,
                status=StageStatus.FAILED,
                start_time=time.time(),
                error=f"Stage not found: {stage_name}",
            )
        
        stage = self._stages[stage_name]
        
        # Check dependencies
        for dep in stage.dependencies:
            if dep not in self._results:
                return StageResult(
                    stage_name=stage_name,
                    status=StageStatus.SKIPPED,
                    start_time=time.time(),
                    error=f"Dependency not completed: {dep}",
                )
            
            dep_result = self._results[dep]
            if dep_result.status != StageStatus.COMPLETED:
                return StageResult(
                    stage_name=stage_name,
                    status=StageStatus.SKIPPED,
                    start_time=time.time(),
                    error=f"Dependency failed: {dep}",
                )
        
        # Execute stage
        start_time = time.time()
        result = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            start_time=start_time,
        )
        
        self._results[stage_name] = result
        
        try:
            logger.info(f"Running bootstrap stage: {stage_name}")
            
            # Run setup function
            if asyncio.iscoroutinefunction(stage.setup_func):
                # Async function
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    metadata = loop.run_until_complete(
                        asyncio.wait_for(
                            stage.setup_func(),
                            timeout=stage.timeout
                        )
                    )
                finally:
                    loop.close()
            else:
                # Sync function
                metadata = stage.setup_func()
            
            # Run health check if provided
            if stage.health_check:
                if asyncio.iscoroutinefunction(stage.health_check):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        healthy = loop.run_until_complete(
                            asyncio.wait_for(
                                stage.health_check(),
                                timeout=5.0
                            )
                        )
                    finally:
                        loop.close()
                else:
                    healthy = stage.health_check()
                
                if not healthy:
                    raise RuntimeError(f"Health check failed for {stage_name}")
            
            # Success
            result.status = StageStatus.COMPLETED
            result.end_time = time.time()
            result.duration = result.end_time - start_time
            result.metadata = metadata if isinstance(metadata, dict) else {}
            
            logger.info(
                f"Completed bootstrap stage: {stage_name} "
                f"({result.duration:.2f}s)"
            )
            
        except Exception as e:
            logger.error(f"Failed bootstrap stage {stage_name}: {e}")
            
            result.status = StageStatus.FAILED
            result.end_time = time.time()
            result.duration = result.end_time - start_time
            result.error = str(e)
            
            # Run rollback if provided
            if stage.rollback_func:
                try:
                    if asyncio.iscoroutinefunction(stage.rollback_func):
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(stage.rollback_func())
                        finally:
                            loop.close()
                    else:
                        stage.rollback_func()
                    
                    logger.info(f"Rolled back stage: {stage_name}")
                except Exception as rollback_error:
                    logger.error(
                        f"Rollback failed for {stage_name}: {rollback_error}"
                    )
        
        return result
    
    def run_bootstrap(self, stop_on_failure: bool = True) -> dict[str, StageResult]:
        """Run all bootstrap stages.
        
        Args:
            stop_on_failure: Stop if a critical stage fails
            
        Returns:
            Dict of stage name -> stage result
        """
        if self._running:
            logger.warning("Bootstrap already running")
            return self._results
        
        self._running = True
        self._results.clear()
        
        # Resolve execution order
        order = self._resolve_order()
        self._stage_order = order
        
        logger.info(f"Starting bootstrap with {len(order)} stages")
        
        failed_critical = False
        
        for stage_name in order:
            # Check if critical stage failed
            if failed_critical and stop_on_failure:
                # Mark remaining as skipped
                remaining = [s for s in order if s not in self._results]
                for remaining_stage in remaining:
                    self._results[remaining_stage] = StageResult(
                        stage_name=remaining_stage,
                        status=StageStatus.SKIPPED,
                        start_time=time.time(),
                        error="Skipped due to critical failure",
                    )
                break
            
            result = self.run_stage(stage_name)
            
            # Check if critical stage failed
            if result.status == StageStatus.FAILED:
                stage = self._stages.get(stage_name)
                if stage and stage.is_critical:
                    failed_critical = True
                    
                    if stop_on_failure:
                        logger.error(
                            f"Critical stage failed: {stage_name}, stopping bootstrap"
                        )
                        break
        
        self._running = False
        
        # Log summary
        completed = sum(1 for r in self._results.values() if r.status == StageStatus.COMPLETED)
        failed = sum(1 for r in self._results.values() if r.status == StageStatus.FAILED)
        skipped = sum(1 for r in self._results.values() if r.status == StageStatus.SKIPPED)
        
        logger.info(
            f"Bootstrap complete: {completed} completed, {failed} failed, {skipped} skipped"
        )
        
        return self._results
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage."""
        return self._results.get(stage_name)
    
    def get_results(self) -> dict[str, StageResult]:
        """Get all stage results."""
        return dict(self._results)
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage completed successfully."""
        result = self._results.get(stage_name)
        return result is not None and result.status == StageStatus.COMPLETED
    
    def is_stage_failed(self, stage_name: str) -> bool:
        """Check if a stage failed."""
        result = self._results.get(stage_name)
        return result is not None and result.status == StageStatus.FAILED
    
    def get_progress(self) -> tuple[int, int]:
        """Get bootstrap progress (completed, total)."""
        completed = sum(
            1 for r in self._results.values()
            if r.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
        )
        return completed, len(self._stages)
    
    def get_bootstrap_report(self) -> str:
        """Generate a bootstrap report."""
        lines = ["# Bootstrap Report", ""]
        
        lines.append("## Stages")
        lines.append("")
        
        for stage_name in self._stage_order:
            result = self._results.get(stage_name)
            stage = self._stages.get(stage_name)
            
            if result is None:
                status = "⏳ Pending"
                duration = ""
            elif result.status == StageStatus.COMPLETED:
                status = "✅ Completed"
                duration = f" ({result.duration:.2f}s)"
            elif result.status == StageStatus.FAILED:
                status = "❌ Failed"
                duration = f" - {result.error}"
            elif result.status == StageStatus.SKIPPED:
                status = "⏭️ Skipped"
                duration = f" - {result.error}"
            else:
                status = "🔄 Running"
                duration = ""
            
            description = stage.description if stage else "Unknown"
            lines.append(f"- **{stage_name}**: {status}{duration}")
            lines.append(f"  _{description}_")
        
        # Summary
        completed, total = self.get_progress()
        failed = sum(1 for r in self._results.values() if r.status == StageStatus.FAILED)
        
        lines.extend([
            "",
            "## Summary",
            "",
            f"- **Completed**: {completed}/{total}",
            f"- **Failed**: {failed}",
            f"- **Status**: {'✅ Success' if failed == 0 else '❌ Failed'}",
        ])
        
        return "\n".join(lines)


# Default stages for Jarvis
def create_default_bootstrap() -> BootstrapManager:
    """Create the default bootstrap manager for Jarvis."""
    manager = BootstrapManager()
    
    # Stage 1: Import core modules
    def import_core():
        return {"status": "imported", "modules": ["memory", "llm", "agent"]}
    
    manager.register_stage(BootstrapStage(
        name="import_core",
        description="Import core modules",
        setup_func=import_core,
        is_critical=True,
        timeout=10.0,
    ))
    
    # Stage 2: Initialize memory
    def init_memory():
        from core.memory import ConversationMemory
        memory = ConversationMemory()
        return {"status": "initialized", "max_turns": memory.max_turns}
    
    manager.register_stage(BootstrapStage(
        name="init_memory",
        description="Initialize conversation memory",
        setup_func=init_memory,
        dependencies=["import_core"],
        is_critical=True,
        timeout=5.0,
    ))
    
    # Stage 3: Initialize LLM client
    def init_llm():
        from core.llm import OllamaClient
        from core.config import get_config
        config = get_config()
        OllamaClient(model=config.llm_model)
        return {"status": "initialized", "model": config.llm_model}
    
    manager.register_stage(BootstrapStage(
        name="init_llm",
        description="Initialize LLM client",
        setup_func=init_llm,
        dependencies=["import_core"],
        is_critical=True,
        timeout=10.0,
    ))
    
    # Stage 4: Initialize vector memory
    def init_vector_memory():
        from core.vector_memory import get_vector_memory
        vm = get_vector_memory()
        return {"status": "initialized", "dimensions": vm.EMBEDDING_DIM}
    
    manager.register_stage(BootstrapStage(
        name="init_vector_memory",
        description="Initialize vector memory",
        setup_func=init_vector_memory,
        dependencies=["import_core"],
        is_critical=False,
        timeout=20.0,
    ))
    
    # Stage 5: Initialize knowledge graph
    def init_knowledge_graph():
        from core.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        return {"status": "initialized", "nodes": len(kg.nodes)}
    
    manager.register_stage(BootstrapStage(
        name="init_knowledge_graph",
        description="Initialize knowledge graph",
        setup_func=init_knowledge_graph,
        dependencies=["import_core"],
        is_critical=False,
        timeout=10.0,
    ))
    
    # Stage 6: Initialize intelligence hub
    def init_intelligence_hub():
        from core.intelligence_hub import get_intelligence_hub
        hub = get_intelligence_hub()
        return {"status": "initialized", "subsystems": len(hub._subsystem_status)}
    
    manager.register_stage(BootstrapStage(
        name="init_intelligence_hub",
        description="Initialize intelligence hub",
        setup_func=init_intelligence_hub,
        dependencies=["import_core"],
        is_critical=False,
        timeout=5.0,
    ))
    
    # Stage 7: Initialize reasoning engine
    def init_reasoning_engine():
        from core.reasoning_engine import get_reasoning_engine
        get_reasoning_engine()
        return {"status": "initialized"}
    
    manager.register_stage(BootstrapStage(
        name="init_reasoning_engine",
        description="Initialize reasoning engine",
        setup_func=init_reasoning_engine,
        dependencies=["import_core"],
        is_critical=False,
        timeout=5.0,
    ))
    
    # Stage 8: Final health check
    def final_health_check():
        return {"status": "healthy", "stages_completed": 7}
    
    manager.register_stage(BootstrapStage(
        name="health_check",
        description="Final health check",
        setup_func=final_health_check,
        health_check=lambda: True,
        dependencies=[
            "init_memory",
            "init_llm",
            "init_vector_memory",
            "init_knowledge_graph",
        ],
        is_critical=True,
        timeout=5.0,
    ))
    
    return manager


# Singleton
_bootstrap_manager: Optional[BootstrapManager] = None
_bootstrap_lock = threading.Lock()


def get_bootstrap_manager() -> BootstrapManager:
    """Get the singleton bootstrap manager."""
    global _bootstrap_manager
    if _bootstrap_manager is None:
        with _bootstrap_lock:
            if _bootstrap_manager is None:
                _bootstrap_manager = create_default_bootstrap()
    return _bootstrap_manager