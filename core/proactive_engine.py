"""Proactive Intelligence Engine for Jarvis.

Keeps Jarvis "alive" by making API calls during idle time:
- Self-reflection and learning
- Math exploration (Erdős engine)
- Memory consolidation
- Knowledge graph updates
- Model warmup calls

This ensures the dashboard shows activity and Jarvis stays responsive.
"""
import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from core.dual_helpers import proactive_curious, proactive_reflect
from core.helpers import get_response_content, create_tracked_task, safe_error_message

logger = logging.getLogger(__name__)
from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType

_proactive_engine: Optional["ProactiveEngine"] = None
_engine_lock = threading.Lock()


class ProactiveTask(str, Enum):
    """Types of proactive tasks."""
    SELF_REFLECT = "self_reflect"
    MATH_EXPLORE = "math_explore"
    MEMORY_CONSOLIDATE = "memory_consolidate"
    KNOWLEDGE_UPDATE = "knowledge_update"
    MODEL_WARMUP = "model_warmup"
    INSIGHT_GENERATE = "insight_generate"
    CURIOSITY = "curiosity"


@dataclass
class ProactiveResult:
    """Result from a proactive task."""
    task: ProactiveTask
    success: bool
    message: str
    tokens_used: int = 0
    duration_ms: float = 0.0
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "success": self.success,
            "message": self.message,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class ProactiveEngine:
    """Engine for proactive intelligence activities.
    
    Runs background tasks when Jarvis is idle to:
    1. Keep the LLM connection warm
    2. Generate insights autonomously
    3. Explore problems in free time
    4. Consolidate memories
    5. Update knowledge graph
    """
    
    DEFAULT_IDLE_THRESHOLD = 20.0  # Seconds before proactive mode
    MIN_INTERVAL = 30.0  # Minimum seconds between proactive calls
    MAX_INTERVAL = 120.0  # Maximum seconds between proactive calls
    
    CURIOSITY_PROMPTS = [
        "What's an interesting pattern in mathematics that I should explore?",
        "How can I be more helpful to users?",
        "What connections exist between concepts I've learned recently?",
        "What are some open questions in the topics we've discussed?",
        "Reflect on my recent interactions - what could I improve?",
        "What new capabilities would be valuable to develop?",
        "Generate a insight about the relationship between logic and creativity.",
        "What's something counterintuitive about AI assistants?",
    ]
    
    def __init__(
        self,
        llm_client,
        get_idle_status: Callable[[], bool] | None = None,
        idle_threshold: float = DEFAULT_IDLE_THRESHOLD,
    ):
        self.llm = llm_client
        self.get_idle_status = get_idle_status
        self.idle_threshold = idle_threshold
        
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_proactive: datetime | None = None
        self._idle_since: datetime | None = None
        self._total_proactive_calls = 0
        self._successful_calls = 0
        self._results: list[ProactiveResult] = []
        self._lock = threading.RLock()
        
        self._idle_callback: Callable | None = None
        self._result_callback: Callable | None = None
        
    def set_callbacks(
        self,
        on_idle: Callable | None = None,
        on_result: Callable | None = None,
    ) -> None:
        """Set callbacks for proactive events."""
        self._idle_callback = on_idle
        self._result_callback = on_result
    
    def is_proactive_mode(self) -> bool:
        """Check if system is in proactive (idle) mode."""
        if not self.get_idle_status:
            return False
        return self.get_idle_status()
    
    def get_stats(self) -> dict:
        """Get proactive engine statistics."""
        with self._lock:
            return {
                "running": self._running,
                "total_calls": self._total_proactive_calls,
                "successful_calls": self._successful_calls,
                "success_rate": self._successful_calls / max(1, self._total_proactive_calls),
                "last_proactive": self._last_proactive.isoformat() if self._last_proactive else None,
                "idle_since": self._idle_since.isoformat() if self._idle_since else None,
                "is_proactive_mode": self.is_proactive_mode(),
                "recent_results": [
                    {
                        "task": r.task.value,
                        "success": r.success,
                        "message": r.message[:100],
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in self._results[-10:]
                ],
            }
    
    async def start(self) -> None:
        """Start the proactive intelligence engine."""
        if self._running:
            return
        
        self._running = True
        self._task = create_tracked_task(self._proactive_loop(), name="proactive_loop")
        logger.info("Proactive intelligence engine started")
    
    async def stop(self) -> None:
        """Stop the proactive intelligence engine."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Proactive intelligence engine stopped")
    
    async def _proactive_loop(self) -> None:
        """Main proactive intelligence loop."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                if not self.is_proactive_mode():
                    self._idle_since = None
                    continue
                
                if self._idle_since is None:
                    self._idle_since = datetime.now()
                
                idle_time = (datetime.now() - self._idle_since).total_seconds()
                
                if idle_time < self.idle_threshold:
                    continue
                
                if self._last_proactive:
                    since_last = (datetime.now() - self._last_proactive).total_seconds()
                    min_interval = random.uniform(self.MIN_INTERVAL, self.MAX_INTERVAL)
                    if since_last < min_interval:
                        continue
                
                task = self._select_task()
                result = await self._execute_task(task)
                
                with self._lock:
                    self._total_proactive_calls += 1
                    if result.success:
                        self._successful_calls += 1
                    self._results.append(result)
                    self._results = self._results[-50:]
                    self._last_proactive = datetime.now()
                
                if self._result_callback:
                    try:
                        self._result_callback(result)
                    except Exception as e:
                        logger.debug(f"Result callback error: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Proactive loop error: {e}")
                await asyncio.sleep(30.0)
    
    def _select_task(self) -> ProactiveTask:
        """Select which proactive task to run."""
        weights = {
            ProactiveTask.SELF_REFLECT: 0.20,
            ProactiveTask.CURIOSITY: 0.25,
            ProactiveTask.MEMORY_CONSOLIDATE: 0.15,
            ProactiveTask.INSIGHT_GENERATE: 0.20,
            ProactiveTask.MODEL_WARMUP: 0.10,
            ProactiveTask.KNOWLEDGE_UPDATE: 0.10,
        }
        
        tasks = list(weights.keys())
        probs = list(weights.values())
        return random.choices(tasks, weights=probs, k=1)[0]
    
    async def _execute_task(self, task: ProactiveTask) -> ProactiveResult:
        """Execute a proactive task."""
        start_time = time.time()
        
        try:
            if task == ProactiveTask.SELF_REFLECT:
                return await self._self_reflect(start_time)
            elif task == ProactiveTask.CURIOSITY:
                return await self._curiosity(start_time)
            elif task == ProactiveTask.MEMORY_CONSOLIDATE:
                return await self._memory_consolidate(start_time)
            elif task == ProactiveTask.INSIGHT_GENERATE:
                return await self._insight_generate(start_time)
            elif task == ProactiveTask.MODEL_WARMUP:
                return await self._model_warmup(start_time)
            elif task == ProactiveTask.KNOWLEDGE_UPDATE:
                return await self._knowledge_update(start_time)
            else:
                return ProactiveResult(
                    task=task,
                    success=False,
                    message="Unknown task type",
                )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=task,
                success=False,
                message=f"Error: {safe_error_message(e)}",
                duration_ms=duration,
            )
    
    async def _self_reflect(self, start_time: float) -> ProactiveResult:
        """Self-reflection task - analyze recent interactions."""
        prompt = """Briefly reflect on how I can improve my assistance. Consider:
1. What types of questions am I best at answering?
2. Where could I be more helpful?
3. What's one thing I should learn more about?

Respond in 2-3 short sentences."""
        
        try:
            try:
                result = await proactive_reflect(prompt, {"task": "self_reflect"})
                duration = (time.time() - start_time) * 1000
                return ProactiveResult(
                    task=ProactiveTask.SELF_REFLECT,
                    success=True,
                    message=result[:200],
                    tokens_used=len(result.split()),
                    duration_ms=duration,
                    data={"model": "dual-parallel"},
                )
            except Exception:
                response = await self.llm.chat(
                    user_input=prompt,
                    system_prompt="You are Jarvis doing brief self-reflection. Be concise.",
                    conversation_history=[],
                )
                content = get_response_content(response) if isinstance(response, dict) else str(response)
                duration = (time.time() - start_time) * 1000
                return ProactiveResult(
                    task=ProactiveTask.SELF_REFLECT,
                    success=True,
                    message=content[:200],
                    tokens_used=len(content.split()),
                    duration_ms=duration,
                )
        except Exception as e:
            return ProactiveResult(
                task=ProactiveTask.SELF_REFLECT,
                success=False,
                message=f"Self-reflect failed: {e}",
            )
    
    async def _curiosity(self, start_time: float) -> ProactiveResult:
        """Curiosity task - explore a random interesting topic."""
        prompt = random.choice(self.CURIOSITY_PROMPTS)
        
        try:
            try:
                result = await proactive_curious(prompt, {"task": "curiosity"})
                duration = (time.time() - start_time) * 1000
                
                # Observe curiosity with PFC
                try:
                    pfc = get_prefrontal_cortex()
                    if pfc and pfc._running:
                        from core.helpers import create_tracked_task
                        create_tracked_task(
                            pfc.observe(
                                content=f"Curiosity exploration: {prompt[:100]} -> {result[:100]}",
                                concept_type=ConceptType.INSIGHT,
                                source="proactive:curiosity",
                                metadata={"prompt": prompt[:200]},
                            ),
                            name="pfc_observe_proactive_curiosity",
                        )
                except Exception as e:
                    logger.debug(f"PFC observation failed: {e}")
                
                return ProactiveResult(
                    task=ProactiveTask.CURIOSITY,
                    success=True,
                    message=result[:200],
                    tokens_used=len(result.split()),
                    duration_ms=duration,
                    data={"prompt": prompt, "model": "dual-primary"},
                )
            except Exception:
                response = await self.llm.chat(
                    user_input=prompt,
                    system_prompt="You are Jarvis being curious and thoughtful. Respond briefly but insightfully.",
                    conversation_history=[],
                )
                content = get_response_content(response) if isinstance(response, dict) else str(response)
                duration = (time.time() - start_time) * 1000
                return ProactiveResult(
                    task=ProactiveTask.CURIOSITY,
                    success=True,
                    message=content[:200],
                    tokens_used=len(content.split()),
                    duration_ms=duration,
                    data={"prompt": prompt},
                )
        except Exception as e:
            return ProactiveResult(
                task=ProactiveTask.CURIOSITY,
                success=False,
                message=f"Curiosity task failed: {e}",
            )
    
    async def _memory_consolidate(self, start_time: float) -> ProactiveResult:
        """Consolidate recent memories into long-term storage."""
        try:
            from core.unified_memory import get_unified_memory
            memory = get_unified_memory()
            
            result = memory.consolidate_all()
            
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=ProactiveTask.MEMORY_CONSOLIDATE,
                success=True,
                message=f"Consolidated {result['consolidated']} memories (working→persistent: {result['working_to_persistent']}, conv→episodic: {result['conversation_to_episodic']})",
                duration_ms=duration,
                data=result,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=ProactiveTask.MEMORY_CONSOLIDATE,
                success=False,
                message=f"Memory consolidation failed: {e}",
                duration_ms=duration,
            )
    
    async def _insight_generate(self, start_time: float) -> ProactiveResult:
        """Generate insights from recent patterns."""
        prompt = """Based on general patterns in computing and AI, generate one interesting insight.
Keep it to 1-2 sentences. The insight should be novel or non-obvious."""
        
        try:
            try:
                from core.dual_helpers import insight_extract
                content = await insight_extract(prompt, {"task": "insight_generate"})
            except Exception:
                response = await self.llm.chat(
                    user_input=prompt,
                    system_prompt="You are Jarvis generating insights. Be brief and thought-provoking.",
                    conversation_history=[],
                )
                content = get_response_content(response) if isinstance(response, dict) else str(response)
            
            try:
                from core.insight_bridge import get_insight_bridge
                bridge = get_insight_bridge()
                bridge.process_insight(
                    raw_insight=content,
                    source_problem="proactive_curiosity",
                    source_category="general",
                    strategy_used=None,
                    problem_solved=False,
                )
            except Exception as e:
                logger.debug(f"Proactive result handling failed: {e}")
            
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=ProactiveTask.INSIGHT_GENERATE,
                success=True,
                message=content[:200],
                tokens_used=len(content.split()),
                duration_ms=duration,
            )
        except Exception as e:
            return ProactiveResult(
                task=ProactiveTask.INSIGHT_GENERATE,
                success=False,
                message=f"Insight generation failed: {e}",
            )
    
    async def _model_warmup(self, start_time: float) -> ProactiveResult:
        """Warm up the model connection with a simple call."""
        try:
            response = await self.llm.chat(
                user_input="Say 'ready' if you're available.",
                system_prompt="You are Jarvis. Respond with a single word.",
                conversation_history=[],
            )
            content = get_response_content(response) if isinstance(response, dict) else str(response)
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=ProactiveTask.MODEL_WARMUP,
                success=True,
                message="Model connection warmed up",
                tokens_used=len(content.split()),
                duration_ms=duration,
            )
        except Exception as e:
            return ProactiveResult(
                task=ProactiveTask.MODEL_WARMUP,
                success=False,
                message=f"Model warmup failed: {e}",
            )
    
    async def _knowledge_update(self, start_time: float) -> ProactiveResult:
        """Update knowledge graph with recent insights."""
        try:
            from core.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            
            stats = kg.get_graph_statistics()
            nodes = stats.get("total_nodes", 0)
            edges = stats.get("total_edges", 0)
            
            kg.compute_pagerank()
            
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=ProactiveTask.KNOWLEDGE_UPDATE,
                success=True,
                message=f"Knowledge graph updated: {nodes} nodes, {edges} edges",
                duration_ms=duration,
                data=stats,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ProactiveResult(
                task=ProactiveTask.KNOWLEDGE_UPDATE,
                success=False,
                message=f"Knowledge update failed: {e}",
                duration_ms=duration,
            )
    
    async def trigger_task(self, task: ProactiveTask) -> ProactiveResult:
        """Manually trigger a proactive task."""
        return await self._execute_task(task)


def get_proactive_engine(llm_client=None) -> ProactiveEngine:
    """Get or create the global ProactiveEngine instance."""
    global _proactive_engine, _engine_lock
    
    if _proactive_engine is None:
        with _engine_lock:
            if _proactive_engine is None:
                if llm_client is None:
                    raise ValueError("llm_client required for first initialization")
                _proactive_engine = ProactiveEngine(llm_client)
    
    return _proactive_engine