"""Daydream Engine - Runs background learning only when idle.

Jarvis "daydreams" about math problems when:
1. User is not interacting (idle)
2. No tasks in queue
3. No pending permissions/operations

This is distinct from "task" mode - daydreaming is passive imagination
that supports growth, not active work.

Integration with n8n:
- Can trigger n8n workflows for automation
- Math exploration via Erdős engine
- Tool practice and self-improvement
- Knowledge consolidation
- Music exploration and composition
"""
import asyncio
import json
import logging
import random
import threading
from collections import deque
from datetime import datetime
from typing import Any, Callable

from core.prefrontal_cortex import get_prefrontal_cortex
from core.helpers import create_tracked_task, safe_error_message
from core.dual_helpers import daydream_explore, learning_discover

logger = logging.getLogger(__name__)


class DaydreamEngine:
    """Manages idle-time background learning and imagination.
    
    Daydreaming happens automatically when Jarvis is not busy:
    - Math problem exploration (Erdős engine)
    - Tool practice and competency improvement
    - Insight generation for evolution
    - Creative exploration without user interaction
    
    NOT run when:
    - User is actively interacting
    - Tasks are in queue
    - Permissions pending
    """

    DEFAULT_IDLE_THRESHOLD = 10.0  # Reduced from 15s for faster engagement
    DEFAULT_DAYDREAM_INTERVAL = 15.0  # Increased frequency: daydream every 15s when idle
    MAX_CONTINUOUS_DAYDREAM = 900  # Increased from 600s to 15 min
    MAX_EXPLORATION_ACTIONS = 25  # Max actions per daydream session
    API_CALLS_PER_SESSION = 20  # Max API calls per daydream session

    def __init__(
        self,
        get_idle_status: Callable[[], bool] | None = None,
        get_queue_size: Callable[[], int] | None = None,
        get_pending_count: Callable[[], int] | None = None,
        idle_threshold: float = DEFAULT_IDLE_THRESHOLD,
    ):
        self.get_idle_status = get_idle_status
        self.get_queue_size = get_queue_size
        self.get_pending_count = get_pending_count
        self.idle_threshold = idle_threshold
        self._daydream_interval = self.DEFAULT_DAYDREAM_INTERVAL

        self._running = False
        self._autonomous_mode = False
        self._daydream_task: asyncio.Task | None = None
        self._last_daydream: datetime | None = None
        self._daydream_count = 0
        self._total_daydream_time = 0.0
        self._throttled = False
        self._interval_lock = threading.Lock()
        self._exploration_topics: deque[str] = deque(maxlen=20)  # Bounded deque
        self._last_tool_practice: datetime | None = None
        self._tools_practiced_this_session = 0
        self._insights_generated = 0
        self._pending_tasks: set[asyncio.Task] = set()  # Track fire-and-forget tasks
        
        self._context_engine = None

        self._on_daydream_start: Callable | None = None
        self._on_daydream_end: Callable | None = None
        self._daydream_callback: Callable | None = None
        
        # Track current activity for visualization
        self._current_activity: str | None = None
        
        # Tool registry for practice during daydream
        # Set by run.py after agent.start() connects all tools
        self._tool_registry: dict | None = None
        
        # Agent reference for tool usage tracking
        # Set by run.py to enable tracking of tool practice in flow visualization
        self._agent: Any = None
    
    def _get_context_engine(self):
        """Get unified context engine for enriched daydream exploration."""
        if self._context_engine is None:
            try:
                from core.unified_context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except Exception as e:
                logger.debug(f"Context engine unavailable: {e}")
        return self._context_engine
    
    async def get_exploration_context(self, topic: str) -> str:
        """Get relevant context from unified memory for daydream exploration.
        
        Enriches exploration with:
        - Past insights about the topic
        - Knowledge graph connections
        - User preferences and facts
        - Working memory from recent sessions
        """
        try:
            engine = self._get_context_engine()
            if engine is None:
                return ""
            
            context = await engine.assemble_context(
                query=f"daydream exploration: {topic}",
                max_tokens=400,
                include_sources=["insight_active", "knowledge_graph", "user_facts", "working_memory"],
            )
            
            if context.entries:
                return context.to_prompt()
        except Exception as e:
            logger.debug(f"Failed to get exploration context: {e}")
        
        return ""

    def set_callbacks(
        self,
        on_start: Callable | None = None,
        on_end: Callable | None = None,
        daydream_action: Callable | None = None,
    ) -> None:
        self._on_daydream_start = on_start
        self._on_daydream_end = on_end
        self._daydream_callback = daydream_action

    def _create_tracked_task(self, coro, *, name: str = "daydream") -> asyncio.Task:
        from core.helpers import create_tracked_task
        return create_tracked_task(coro, name=f"daydream_{name}")
    
    def _track_pfc_insight(self, pfc, insight: str, source: str, metadata: dict = None) -> None:
        """Safely observe insight with PFC, tracking the task."""
        if pfc and pfc._running:
            task = self._create_tracked_task(pfc.observe_insight(
                insight=insight[:200] if insight else "",
                source=source,
                metadata=metadata or {}
            ))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
    
    async def _cleanup_pending_tasks(self) -> None:
        """Wait for pending tasks to complete or cancel them."""
        if self._pending_tasks:
            for task in list(self._pending_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

    def can_daydream(self) -> bool:
        if not self.get_idle_status:
            return False

        is_idle = self.get_idle_status()
        queue_empty = self.get_queue_size() == 0 if self.get_queue_size else True
        no_pending = self.get_pending_count() == 0 if self.get_pending_count else True

        resource_ok = self._check_resources()

        return is_idle and queue_empty and no_pending and resource_ok

    def _check_resources(self) -> bool:
        """Check if system resources allow daydreaming.
        
        Returns True if resources are available, False if under pressure or unavailable.
        Gracefully degrades if resource monitor cannot be initialized.
        """
        try:
            from core.resource_monitor import get_resource_monitor
            monitor = get_resource_monitor()
            if monitor is None:
                logger.warning("Resource monitor not available, allowing daydream with caution")
                return True
            under_pressure, pressure_details = monitor.is_under_pressure()
            if under_pressure:
                logger.info(f"Resource pressure detected, skipping daydream: {pressure_details}")
                return False
            return True
        except ImportError as e:
            logger.warning(f"Resource monitor not available: {e}, allowing daydream with caution")
            return True
        except Exception as e:
            logger.warning(f"Resource check failed: {e}, allowing daydream with caution")
            return True

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._daydream_task = create_tracked_task(self._daydream_loop(), name="daydream_loop")

        try:
            from core.resource_monitor import get_resource_monitor
            monitor = get_resource_monitor()
            monitor.register_callback(self._on_resource_pressure)
            monitor.start_monitoring()
        except Exception as e:
            logger.debug(f"Resource monitor not available: {e}")

        logger.info("Daydream engine started - will imagine when idle")

    def _on_resource_pressure(self, snapshot, message: str) -> None:
        """Handle resource pressure events."""
        logger.warning(f"Daydream throttled: {message}")
        self._throttled = True

    async def stop(self) -> None:
        self._running = False
        if self._daydream_task:
            self._daydream_task.cancel()
            try:
                await self._daydream_task
            except asyncio.CancelledError:
                pass
        await self._cleanup_pending_tasks()
        logger.info(f"Daydream engine stopped (dreamed {self._daydream_count} times)")

    async def _daydream_loop(self) -> None:
        while self._running:
            try:
                with self._interval_lock:
                    interval = self._daydream_interval
                await asyncio.sleep(interval)

                if self.can_daydream():
                    try:
                        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
                        lb = get_load_balancer()
                        if not lb.should_run_task(SubsystemType.DAYDREAM):
                            continue
                        lb.notify_task_start(SubsystemType.DAYDREAM)
                    except Exception as e:
                        logger.warning(f"Failed to notify load balancer of daydream start: {e}")

                    await self._enter_daydream()

                    try:
                        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
                        lb = get_load_balancer()
                        lb.notify_task_complete(SubsystemType.DAYDREAM)
                    except Exception as e:
                        logger.warning(f"Failed to notify load balancer of daydream completion: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daydream loop error: {e}")
                await asyncio.sleep(60)

    async def _enter_daydream(self) -> None:
        self._daydream_count += 1
        
        # Mark as autonomous operation - shell tools require explicit permission
        self._autonomous_mode = True
        
        if self._on_daydream_start:
            try:
                await self._on_daydream_start()
            except Exception as e:
                logger.debug(f"Daydream start callback error: {e}")

        logger.debug("Entering daydream state...")

        dream_duration = 0.0
        max_iterations = self.MAX_EXPLORATION_ACTIONS  # Configurable max actions
        iteration = 0
        api_calls_made = 0

        for i in range(max_iterations):
            if not self._running or not self.can_daydream():
                break

            if dream_duration > self.MAX_CONTINUOUS_DAYDREAM:
                break
            
            if api_calls_made >= self.API_CALLS_PER_SESSION:
                logger.debug(f"API call limit reached ({api_calls_made}), ending daydream")
                break

            # Perform different types of exploration based on iteration
            # Expanded action types for more diverse exploration
            action_type = i % 10  # 10 different exploration modes (added music exploration)

            try:
                if action_type == 0:
                    # Tool practice
                    await self._practice_tools_dream()
                    api_calls_made += 2  # Tool practice uses a few calls
                elif action_type == 1:
                    # Math exploration with Erdős
                    if self._daydream_callback:
                        await self._daydream_callback()
                    api_calls_made += 3  # Math exploration uses several calls
                elif action_type == 2:
                    # Deep insight generation with dual-model
                    await self._generate_deep_insights()
                    api_calls_made += 4  # Deep insights use both models
                elif action_type == 3:
                    # Knowledge consolidation
                    await self._consolidate_knowledge()
                    api_calls_made += 1  # Mostly local, few API calls
                elif action_type == 4:
                    # Creative exploration (new)
                    await self._creative_exploration()
                    api_calls_made += 3
                elif action_type == 5:
                    # System self-reflection (new)
                    await self._self_reflection_dream()
                    api_calls_made += 2
                elif action_type == 6:
                    # n8n workflow automation (enhanced)
                    await self._run_enhanced_n8n()
                    api_calls_made += 2
                elif action_type == 7:
                    # Tool creation opportunity analysis (new)
                    await self._analyze_tool_creation_opportunities()
                    api_calls_made += 2
                elif action_type == 8:
                    # Music exploration (new)
                    logger.info(f"[Daydream] Starting music exploration (iteration {i})")
                    await self._music_exploration()
                    api_calls_made += 2
                else:
                    # Extended n8n + pattern discovery
                    await self._extended_exploration()
                    api_calls_made += 3
            except Exception as e:
                logger.warning(f"Daydream action error: {e}")

            iteration += 1
            with self._interval_lock:
                dream_duration += self._daydream_interval
                interval = self._daydream_interval
            await asyncio.sleep(interval)

        self._last_daydream = datetime.now()
        self._total_daydream_time += dream_duration

        # Reset autonomous mode - daydream complete
        self._autonomous_mode = False

        if self._on_daydream_end:
            try:
                await self._on_daydream_end()
            except Exception as e:
                logger.debug(f"Daydream end callback error: {e}")

        logger.debug(f"Exited daydream after {dream_duration:.0f}s, {iteration} iterations")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "can_daydream": self.can_daydream() if self.get_idle_status else False,
            "daydream_count": self._daydream_count,
            "total_daydream_time": self._total_daydream_time,
            "last_daydream": self._last_daydream.isoformat() if self._last_daydream else None,
            "throttled": self._throttled,
            "tools_practiced": self._tools_practiced_this_session,
            "insights_generated": self._insights_generated,
            "exploration_topics": list(self._exploration_topics)[-5:],
            "current_activity": self._current_activity,
        }

    async def _practice_tools_dream(self) -> None:
        """Practice tools during daydream to improve competency.
        
        Uses dual-model for learning and actual tool execution when possible.
        Practices tools from ALL categories for comprehensive autonomous ability.
        """
        try:
            from core.tool_learning import get_tool_learning_engine
            from core.tool_registry import TOOL_REGISTRY, ToolCategory
            
            engine = get_tool_learning_engine()
            
            if self._daydream_count % 10 == 0:
                decayed = engine.apply_competency_decay(days_threshold=7)
                if decayed:
                    logger.info(f"Applied competency decay: {', '.join(decayed)}")

            # ALWAYS get tools from all categories for comprehensive practice
            category_priorities = [
                (ToolCategory.WEB, ["web_search", "web_fetch", "image_search"]),
                (ToolCategory.BROWSER, ["browser_launch", "browser_navigate", "browser_click"]),
                (ToolCategory.TERMINAL, ["terminal_execute", "shell"]),
                (ToolCategory.SYSTEM, ["read_file", "write_file", "list_files"]),
                (ToolCategory.DATA, ["data_analyze", "sql_query", "statistics"]),
                (ToolCategory.SCREEN, ["see_screen", "click_element", "type_text"]),
                (ToolCategory.MULTIMODAL, ["vision", "transcribe", "webcam"]),
                (ToolCategory.MEMORY, ["insight_search", "insight_stats"]),
                (ToolCategory.META, ["list_tools", "get_tool_info", "self_reflect"]),
                (ToolCategory.LEARNING, ["erdos_math", "learning", "sandbox_explore"]),
                (ToolCategory.EVOLUTION, ["evolution", "self_edit", "evolution_status"]),
                (ToolCategory.SECURITY, ["security_scan", "defense"]),
                (ToolCategory.REFINEMENT, ["refine_scan", "refine_apply"]),
                (ToolCategory.AUTOMATION, ["docker", "n8n"]),
                (ToolCategory.SPATIAL, ["spatial_calc", "physics_sim"]),
            ]
            
            # Build comprehensive tool list from all categories
            category_tools = []
            for category, tool_names in category_priorities:
                for tool_name in tool_names:
                    if tool_name in TOOL_REGISTRY:
                        category_tools.append(tool_name)
            
            # ALSO get practice priorities (stale/NOVICE tools)
            priority_tools = engine.get_practice_priorities()
            
            # Shuffle category_priorities to get different tools each session
            import random
            shuffled_categories = list(category_priorities)
            random.shuffle(shuffled_categories)
            
            # Pick one tool from each category to ensure variety
            combined_tools = []
            seen = set()
            
            # First, pick one tool from each shuffled category (ensures variety across ALL categories)
            for category, tool_names in shuffled_categories:
                # Shuffle tools in this category too
                shuffled_tools = list(tool_names)
                random.shuffle(shuffled_tools)
                
                for tool_name in shuffled_tools:
                    if tool_name in TOOL_REGISTRY and tool_name not in seen:
                        combined_tools.append(tool_name)
                        seen.add(tool_name)
                        break  # Only one tool per category in first pass
            
            # Then add priority tools that need attention
            for tool_name in priority_tools:
                if tool_name not in seen and tool_name in TOOL_REGISTRY:
                    combined_tools.append(tool_name)
                    seen.add(tool_name)
                    if len(combined_tools) >= 15:  # Allow more than 8 for variety
                        break
            
            if not combined_tools:
                logger.debug("No tools available for practice")
                return
            
            # Log the variety of tools being practiced
            category_count = {}
            for cat, tools_list in category_priorities:
                for t in combined_tools:
                    if t in tools_list:
                        category_count[cat.value] = category_count.get(cat.value, 0) + 1
                        break
            logger.info(f"[Daydream] Practicing {len(combined_tools)} tools across {len(category_count)} categories: {list(category_count.keys())}")

            tool_registry = getattr(self, '_tool_registry', {})
            if not tool_registry:
                logger.warning("Tool registry not available - cannot practice tools. Ensure _tool_registry is set.")
                return

            # Get agent reference for tool tracking (passed from run.py)
            agent = getattr(self, '_agent', None)

            practiced_count = 0
            for tool_name in combined_tools[:8]:  # Practice up to 8 tools per daydream session
                result = None
                try:
                    # Check if tool exists in registry
                    if tool_name not in tool_registry:
                        logger.debug(f"Tool {tool_name} not in registry, skipping")
                        continue
                    
                    # Dual-model learning first
                    dual_learning = await learning_discover(
                        f"Practice and improve {tool_name} tool usage - analyze best practices and common pitfalls",
                        context={"tool": tool_name, "source": "daydream"}
                    )
                    
                    if dual_learning and dual_learning != "Dual model integration not available":
                        self._tools_practiced_this_session += 1
                        self._exploration_topics.append(f"tool:{tool_name}:dual")
                        logger.info(f"✓ Dual-model tool learning: {tool_name}")
                    
                    # Actual execution with real tool
                    result = await engine.practice_tool(
                        tool_name, 
                        actual_execution=True,
                        tool_registry=tool_registry
                    )
                    
                    if result is None:
                        logger.debug(f"Tool practice returned None for {tool_name}")
                        continue

                    if result.get("scenarios_practiced", 0) > 0:
                        practiced_count += 1
                        self._tools_practiced_this_session += 1
                        
                        # Track tool usage for flow visualization
                        if agent and hasattr(agent, '_last_tool_calls'):
                            agent._last_tool_calls.append((tool_name, {"source": "daydream_practice"}))
                        if agent and hasattr(agent, 'tool_usage_count'):
                            agent.tool_usage_count[tool_name] = agent.tool_usage_count.get(tool_name, 0) + 1
                        
                        improvement = result.get("improvements", [])
                        new_level = result.get("new_level", "unknown")
                        scenarios = result.get("scenarios_practiced", 0)
                        
                        self._exploration_topics.append(f"tool:{tool_name}:{new_level}")
                        
                        logger.info(f"✓ Practiced {tool_name}: {scenarios} scenarios → {new_level}")

                        if improvement:
                            from core.memory import get_conversation_memory
                            memory = get_conversation_memory()
                            if memory and hasattr(memory, 'store_insight'):
                                for imp in improvement[:2]:
                                    await memory.store_insight(
                                        f"Tool practice [{tool_name}]: {imp}",
                                        importance=0.4
                                    )
                    
                    # Record tool usage in Prefrontal Cortex for LCM learning
                    try:
                        from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType
                        pfc = get_prefrontal_cortex()
                        if pfc and pfc._running:
                            await pfc.observe(
                                content=f"Practiced {tool_name} tool during daydream",
                                concept_type=ConceptType.SUBSYSTEM_EVENT,
                                source="tool_practice",
                                metadata={"tool": tool_name, "scenarios": result.get("scenarios_practiced", 0)}
                            )
                    except Exception as e:
                        logger.debug(f"PFC observation failed: {e}")
                        
                except Exception as e:
                    logger.warning(f"Tool practice error for {tool_name}: {e}")

            # Log summary
            if practiced_count > 0:
                logger.info(f"[Daydream] Practiced {practiced_count} tools from {len(combined_tools)} candidates")

        except Exception as e:
            logger.warning(f"Tool practice dream error: {e}")

    async def _generate_insights(self) -> None:
        """Generate insights during daydream using dual-model routing."""
        try:
            exploration_topic = "What patterns and insights can we discover?"
            
            dual_result = await daydream_explore(
                exploration_topic,
                context={"source": "daydream", "session": self._daydream_count}
            )
            
            if dual_result and dual_result != "Dual model integration not available":
                self._insights_generated += 1
                
                # Observe with Prefrontal Cortex
                try:
                    pfc = get_prefrontal_cortex()
                    self._track_pfc_insight(pfc, dual_result[:200], "daydream", {"topic": exploration_topic})
                except Exception as e:
                    logger.debug(f"PFC insight tracking failed: {e}")
                topic_preview = dual_result[:50].replace('\n', ' ')
                self._exploration_topics.append(f"dual:{topic_preview}")
                logger.debug(f"Dual-model insights: {topic_preview}")
                return

            from core.erdos_engine import get_erdos_engine

            erdos = get_erdos_engine()
            if erdos:
                result = await erdos.daydream_step()
                if result and result.get("status") != "not_running":
                    self._insights_generated += 1
                    
                    try:
                        pfc = get_prefrontal_cortex()
                        self._track_pfc_insight(pfc, str(result)[:200], "daydream", {"topic": "erdos_explore"})
                    except Exception as e:
                        logger.debug(f"PFC insight tracking failed: {e}")
                    
                    problem_id = result.get("problem_id", "")
                    approach = result.get("approach", "")
                    if problem_id and problem_id != "unknown":
                        title = f"Math problem {problem_id}"
                    elif approach:
                        title = f"Exploring: {approach[:30]}"
                    else:
                        title = "Math exploration"
                    self._exploration_topics.append(title[:50])
                    logger.debug(f"Explored: {title}")

        except Exception as e:
            logger.warning(f"Insight generation error: {e}")

    async def _consolidate_knowledge(self) -> None:
        """Consolidate knowledge during daydream."""
        try:
            # Export tool learning insights to memory
            from core.tool_learning import get_tool_learning_engine
            engine = get_tool_learning_engine()
            memory_data = engine.export_to_memory()

            if memory_data.get("memories"):
                self._insights_generated += len(memory_data["memories"])
                logger.debug(f"Consolidated {len(memory_data['memories'])} tool memories")

        except Exception as e:
            logger.warning(f"Knowledge consolidation error: {e}")

    async def _run_n8n_automation(self) -> None:
        """Run n8n workflow automation during daydream."""
        try:
            from core.daydream_n8n import get_daydream_n8n
            
            integration = get_daydream_n8n()
            
            # Choose an activity based on current needs
            activities = ["math", "tools", "insights", "consolidate"]
            
            # Try to trigger an activity
            for activity in activities:
                result = await integration.trigger_daydream_activity(activity)
                if result.get("success"):
                    self._exploration_topics.append(f"n8n:{activity}")
                    logger.info(f"n8n automation completed: {activity}")
                    return
                    
        except ImportError:
            logger.debug("n8n integration not available")
        except Exception as e:
            logger.warning(f"n8n automation error: {e}")

    async def _generate_deep_insights(self) -> None:
        """Generate deep insights using dual-model collaboration.
        
        Uses both models in a sequential pattern:
        1. Primary (Kimi) generates creative hypotheses
        2. Backend (GLM-5) evaluates and refines them
        """
        try:
            from core.dual_subsystem import SubsystemTask, get_dual_subsystem_integration
            
            integration = get_dual_subsystem_integration()
            if not integration or not integration.primary_llm or not integration.backend_llm:
                await self._generate_insights()
                return
            
            topics = [
                "What hidden patterns exist in the codebase architecture?",
                "How could the dual-model system be improved for better collaboration?",
                "What mathematical insights can enhance the Erdős engine?",
                "What emergent behaviors might arise from the subsystem interactions?",
                "How can daydreaming be more productive for learning?",
            ]
            
            import random
            topic = random.choice(topics)
            
            creative_prompt = f"""Generate novel, creative hypotheses about this topic.
Think outside the box. Consider unconventional approaches. Dream big.

Topic: {topic}

Provide 3-5 creative hypotheses or ideas."""
            
            creative_result = await integration.call(
                subsystem="daydream",
                task=SubsystemTask.CREATIVE_EXPLORATION,
                prompt=creative_prompt,
                context={"source": "deep_insight", "session": self._daydream_count},
                system_prompt="You are a visionary thinker. Be bold and creative.",
            )
            
            if creative_result and creative_result != "Dual model integration not available":
                evaluation_prompt = f"""Evaluate these creative hypotheses critically.
Identify which have the most merit and why. Refine the best ideas.

Hypotheses:
{creative_result}

Provide your analytical evaluation."""
                
                evaluation = await integration.call(
                    subsystem="daydream",
                    task=SubsystemTask.IDEA_EVALUATION,
                    prompt=evaluation_prompt,
                    context={"source": "deep_insight", "session": self._daydream_count},
                    system_prompt="You are a critical thinker. Be thorough and analytical.",
                )
                
                integration.track_sequential(
                    "daydream",
                    SubsystemTask.CREATIVE_EXPLORATION,
                    SubsystemTask.IDEA_EVALUATION
                )
                
                self._insights_generated += 2
                topic_preview = topic[:40].replace('\n', ' ')
                self._exploration_topics.append(f"deep:{topic_preview}")
                logger.info(f"Generated deep insights on: {topic_preview}: {evaluation[:100] if evaluation else 'N/A'}")
            
        except Exception as e:
            logger.warning(f"Deep insight generation error: {e}")
            await self._generate_insights()

    async def _creative_exploration(self) -> None:
        """Creative exploration using primary model's imagination.
        
        Lets Kimi freely associate and generate ideas without constraints.
        """
        try:
            from core.dual_helpers import get_integration
            from core.dual_subsystem import SubsystemTask
            
            integration = get_integration()
            if not integration or not integration.primary_llm:
                logger.debug("No primary LLM for creative exploration")
                return
            
            creative_prompts = [
                "What if Jarvis could rewrite its own code? Describe a safe approach.",
                "Invent a new type of mathematical proof. Describe its structure.",
                "Design a novel pattern for inter-subsystem communication.",
                "Imagine a self-improving memory system. How would it work?",
                "Create a concept for 'meta-daydreaming' - daydreaming about dreaming.",
                "Propose an unconventional approach to token optimization.",
                "Design a creative solution for balancing exploration vs exploitation.",
            ]
            
            import random
            prompt = random.choice(creative_prompts)
            
            result = await integration.call(
                subsystem="daydream",
                task=SubsystemTask.CREATIVE_EXPLORATION,
                prompt=f"Explore creatively:\n\n{prompt}\n\nLet your imagination flow freely. There are no wrong answers.",
                context={"source": "creative_exploration", "session": self._daydream_count},
                system_prompt="You are an imaginative AI exploring possibilities. Be creative, playful, and generative.",
            )
            
            if result:
                self._insights_generated += 1
                
                # Observe with Prefrontal Cortex
                try:
                    pfc = get_prefrontal_cortex()
                    self._track_pfc_insight(pfc, str(result)[:200], "daydream", {"topic": prompt})
                except Exception as e:
                    logger.debug(f"PFC insight tracking failed: {e}")
                prompt_preview = prompt[:40].replace('\n', ' ')
                self._exploration_topics.append(f"creative:{prompt_preview}")
                logger.info(f"Creative exploration: {prompt_preview}")
                
        except Exception as e:
            logger.debug(f"Creative exploration error: {e}")

    async def _self_reflection_dream(self) -> None:
        """Self-reflection on Jarvis's own state and performance.
        
        Uses backend model to analyze system metrics and suggest improvements.
        """
        try:
            status = self.get_status()
            
            from core.dual_helpers import proactive_reflect
            
            context_data = f"""Jarvis Daydream Engine Status:
- Running: {status['running']}
- Can Daydream: {status['can_daydream']}
- Daydream Count: {status['daydream_count']}
- Total Dream Time: {status['total_daydream_time']:.0f}s
- Tools Practiced: {status['tools_practiced']}
- Insights Generated: {status['insights_generated']}
- Throttled: {status['throttled']}
- Recent Topics: {status['exploration_topics']}

What improvements could make daydreaming more effective?"""
            
            reflection = await proactive_reflect(
                context_data,
                context={"source": "self_reflection_dream", "session": self._daydream_count}
            )
            
            if reflection and reflection != "Dual model integration not available":
                self._insights_generated += 1
                
                # Observe with Prefrontal Cortex
                try:
                    pfc = get_prefrontal_cortex()
                    self._track_pfc_insight(pfc, str(reflection)[:200], "daydream", {"topic": "self_reflection"})
                except Exception as e:
                    logger.debug(f"PFC insight tracking failed: {e}")
                self._exploration_topics.append("self:reflection")
                logger.info("Self-reflection completed")
                
        except Exception as e:
            logger.debug(f"Self-reflection dream error: {e}")

    async def _run_enhanced_n8n(self) -> None:
        """Enhanced n8n automation with more activities.
        
        Extends basic n8n automation with additional workflow types.
        """
        try:
            from core.daydream_n8n import get_daydream_n8n
            
            integration = get_daydream_n8n()
            
            enhanced_activities = [
                "math",
                "tools", 
                "insights",
                "consolidate",
                "pattern_discovery",
                "knowledge_sync",
                "health_check",
            ]
            
            import random
            activities_to_try = random.sample(enhanced_activities, min(3, len(enhanced_activities)))
            
            for activity in activities_to_try:
                if not self._running or not self.can_daydream():
                    break
                
                result = await integration.trigger_daydream_activity(activity)
                if result.get("success"):
                    self._exploration_topics.append(f"n8n+:{activity}")
                    logger.info(f"Enhanced n8n: {activity}")
                    return
                    
        except ImportError:
            await self._run_n8n_automation()
        except Exception as e:
            logger.warning(f"Enhanced n8n error: {e}")
            await self._run_n8n_automation()

    async def _analyze_tool_creation_opportunities(self) -> None:
        """Analyze tool usage patterns and identify opportunities for new tools.
        
        During daydream, analyzes:
        - Low-competency tools that may need improvement
        - Missing tool functionality from failed calls
        - Common patterns that could be automated
        - Creates sandbox tools for testing
        """
        try:
            from core.tool_learning import get_tool_learning_engine
            from core.tool_registry import TOOL_TREES
            
            engine = get_tool_learning_engine()
            
            # Get tools with low competency
            priorities = engine.get_practice_priorities()
            low_competency_tools = []
            
            for tool_name in priorities[:5]:
                competency = engine.get_competency(tool_name)
                if competency and competency.level in ['novice', 'beginner']:
                    low_competency_tools.append((tool_name, competency.level))
            
            # Analyze tool usage for gaps
            existing_categories = set()
            for tree_name in TOOL_TREES:
                for tool_name in TOOL_TREES[tree_name]:
                    existing_categories.add(tree_name)
            
            # Identify potential gaps - categories with few tools
            gap_analysis = {}
            for tree_name, tools in TOOL_TREES.items():
                if len(tools) < 3:
                    gap_analysis[tree_name] = f"Only {len(tools)} tool(s) in {tree_name}"
            
            # Log findings
            if low_competency_tools:
                tools_str = ', '.join([f"{n}({l})" for n, l in low_competency_tools])
                self._exploration_topics.append(f"tool_gaps:{tools_str}")
                logger.info(f"Tool creation opportunities - low competency: {tools_str}")
            
            if gap_analysis:
                gap_str = '; '.join([f"{k}: {v}" for k, v in list(gap_analysis.items())[:3]])
                self._exploration_topics.append(f"category_gaps:{gap_str[:50]}")
                logger.debug(f"Category gaps identified: {gap_str}")
            
            # Practice creating hypothetical tool specifications
            # This exercises the tool creation mental model without actual deployment
            if low_competency_tools and len(self._exploration_topics) < 10:
                tool_name = low_competency_tools[0][0]
                competency_level = low_competency_tools[0][1]
                
                # Generate improvement specification (mental exercise)
                analysis = f"Tool '{tool_name}' at {competency_level} level needs improvement"
                
                # Store as insight for potential tool creation
                from core.memory import get_conversation_memory
                memory = get_conversation_memory()
                if memory and hasattr(memory, 'store_insight'):
                    await memory.store_insight(
                        f"Tool creation analysis: {analysis}. "
                        f"Consider enhanced version with better parameter handling or output formatting.",
                        importance=0.3
                    )
                self._insights_generated += 1
                
                # Observe with Prefrontal Cortex
                try:
                    pfc = get_prefrontal_cortex()
                    self._track_pfc_insight(pfc, analysis[:200], "daydream", {"topic": f"tool_creation:{tool_name}"})
                except Exception as e:
                    logger.debug(f"PFC insight tracking failed: {e}")
                
        except Exception as e:
            logger.warning(f"Tool creation analysis error: {e}")

    async def _extended_exploration(self) -> None:
        """Extended exploration combining multiple techniques.
        
        Combines n8n workflows, Erdős math, and knowledge graph exploration.
        """
        try:
            from core.knowledge_graph import get_knowledge_graph
            from core.erdos_engine import get_erdos_engine
            from core.dual_helpers import insight_extract
            
            kg = get_knowledge_graph()
            erdos = get_erdos_engine()
            
            exploration_results = []
            
            if erdos:
                try:
                    result = await erdos.daydream_step()
                    if result and result.get("status") != "not_running":
                        exploration_results.append(f"erdos:{result.get('problem_id', 'unknown')}")
                        self._insights_generated += 1
                        
                        # Observe with Prefrontal Cortex
                        try:
                            pfc = get_prefrontal_cortex()
                            self._track_pfc_insight(pfc, str(result)[:200], "daydream", {"topic": "erdos_exploration"})
                        except Exception as e:
                            logger.debug(f"PFC insight tracking failed: {e}")
                except Exception as e:
                    logger.debug(f"Erdős step error: {e}")
            
            if kg and hasattr(kg, 'get_recent_insights'):
                try:
                    insights = kg.get_recent_insights(limit=3)
                    if insights:
                        for insight in insights:
                            extraction = await insight_extract(
                                str(insight),
                                context={"source": "extended_exploration"}
                            )
                            if extraction:
                                exploration_results.append("insight:extracted")
                                self._insights_generated += 1
                                
                                # Observe with Prefrontal Cortex
                                try:
                                    pfc = get_prefrontal_cortex()
                                    self._track_pfc_insight(pfc, str(extraction)[:200], "daydream", {"topic": "insight_extraction"})
                                except Exception as e:
                                    logger.debug(f"PFC insight tracking failed: {e}")
                except Exception as e:
                    logger.debug(f"Knowledge graph exploration error: {e}")
            
            if exploration_results:
                self._exploration_topics.extend(exploration_results[:3])
                logger.info(f"Extended exploration: {', '.join(exploration_results[:3])}")
                
        except Exception as e:
            logger.warning(f"Extended exploration error: {e}")
    
    async def _music_exploration(self) -> None:
        """Explore music composition during idle time.
        
        Uses the music module to:
        1. Generate compositions in various classical styles
        2. Create variations on themes
        3. Generate acoustic tokens (spectral features)
        4. Observe both symbolic and acoustic tokens with PFC for LCM learning
        """
        try:
            from tools.music import get_music_generator
            
            # Set current activity for visualization
            self._current_activity = "music"
            
            music_gen = get_music_generator()
            pfc = get_prefrontal_cortex()
            
            # Choose exploration style
            styles = ['baroque', 'classical', 'romantic']
            style = styles[self._daydream_count % len(styles)]
            
            # Keys to explore
            keys = ['C_major', 'G_major', 'D_major', 'A_minor', 'E_minor']
            key = keys[self._daydream_count % len(keys)]
            
            # Generate composition with acoustic tokens
            composition = await music_gen.generate_composition_with_acoustics(
                style=style,
                key=key,
                measures=random.choice([8, 16, 24]),
                tempo=random.choice([60, 80, 100, 120])
            )
            
            logger.info(f"[Music] Generated {style} composition in {key}")
            
            # Observe symbolic tokens with PFC
            tokens_observed = 0
            if pfc and pfc._running:
                # Symbolic tokens (style, structure, harmony)
                for token_data in composition.get("tokens", [])[:5]:
                    try:
                        from core.prefrontal_cortex import ConceptType
                        content = token_data.get("content", token_data.get("data", {}))
                        if isinstance(content, dict):
                            content = json.dumps(content)
                        await pfc.observe(
                            content=str(content)[:300],
                            concept_type=ConceptType.MUSIC_TOKEN,
                            source="music_generation",
                            metadata={"style": style, "key": key, "type": token_data.get("type", "symbolic")}
                        )
                        tokens_observed += 1
                    except Exception as e:
                        logger.warning(f"Symbolic token observation error: {e}")
                
                # Acoustic tokens (spectral features for sound learning)
                for acoustic_token in composition.get("spectral_tokens", [])[:10]:  # First 10 spectral frames
                    try:
                        from core.prefrontal_cortex import ConceptType
                        # Spectral tokens contain frequency/energy information
                        content = json.dumps(acoustic_token.get("content", acoustic_token))
                        await pfc.observe(
                            content=str(content)[:400],
                            concept_type=ConceptType.MUSIC_TOKEN,
                            source="acoustic_generation",
                            metadata={
                                "style": style,
                                "key": key,
                                "type": "spectral",
                                "frame": acoustic_token.get("content", {}).get("frame", 0),
                            }
                        )
                        tokens_observed += 1
                    except Exception as e:
                        logger.warning(f"Acoustic token observation error: {e}")
                
                logger.info(f"[Music] Observed {tokens_observed} tokens (symbolic + acoustic) with PFC")
            
            # Generate a variation
            if random.random() < 0.5:
                variation_type = random.choice(['ornamentation', 'inversion', 'sequence', 'diminution'])
                theme = "C4-E4-G4"  # Simple C major triad
                await music_gen.generate_variation(
                    theme=theme,
                    variation_type=variation_type
                )
                logger.info(f"[Music] Created {variation_type} variation")
            
            # Record exploration
            self._exploration_topics.append(f"music:{style}:{key}")
                
        except Exception as e:
            logger.warning(f"Music exploration error: {e}")
        finally:
            self._current_activity = None
    
    async def practice_tools(self) -> dict:
        """Practice tools during idle time to improve competency."""
        try:
            from core.tool_learning import get_tool_learning_engine
            engine = get_tool_learning_engine()

            priorities = engine.get_practice_priorities()
            results = {}

            for tool_name in priorities[:2]:  # Practice top 2 priority tools
                try:
                    tool_registry = getattr(self, '_tool_registry', {})
                    if tool_registry:
                        result = await engine.practice_tool(tool_name, actual_execution=True, tool_registry=tool_registry)
                    else:
                        result = await engine.practice_tool(tool_name, actual_execution=False)
                    results[tool_name] = result
                    logger.info(f"Practiced tool {tool_name}: {result.get('new_level', 'unknown') if result else 'N/A'}")
                except Exception as e:
                    logger.debug(f"Tool practice error for {tool_name}: {e}")

            return {"practiced": len(results), "results": results}
        except Exception as e:
            logger.debug(f"Tool practice error: {e}")
            return {"practiced": 0, "error": safe_error_message(e)}

    def throttle(self, aggressive: bool = False) -> None:
        """Throttle daydream activity for resource conservation."""
        with self._interval_lock:
            self._throttled = True
            if aggressive:
                self._daydream_interval = 90.0
            else:
                self._daydream_interval = max(self._daydream_interval * 2, 120)
        if aggressive:
            self.MAX_CONTINUOUS_DAYDREAM = 120
        logger.info(f"Daydream engine throttled (aggressive={aggressive})")

    def resume(self) -> None:
        """Resume normal operation after throttling."""
        with self._interval_lock:
            self._daydream_interval = max(self._daydream_interval // 2, self.DEFAULT_DAYDREAM_INTERVAL)
            self._throttled = False
        self.MAX_CONTINUOUS_DAYDREAM = 900
        logger.info("Daydream engine resumed")

    def is_healthy(self) -> tuple[bool, dict]:
        """Check daydream engine health."""
        return (
            self._running,
            {
                "daydream_count": self._daydream_count,
                "total_daydream_time": self._total_daydream_time,
                "throttled": self._throttled,
            }
        )


_daydream_engine: DaydreamEngine | None = None
_daydream_lock = threading.Lock()


def get_daydream_engine() -> DaydreamEngine:
    global _daydream_engine
    if _daydream_engine is None:
        with _daydream_lock:
            if _daydream_engine is None:
                _daydream_engine = DaydreamEngine()
    return _daydream_engine
