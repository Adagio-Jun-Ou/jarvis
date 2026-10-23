"""Background Task Scheduler for Jarvis.

Manages periodic background activities:
- Memory consolidation (Dream mode)
- Self-growth evaluation
- Evolution cycles
- Memory cleanup
- Health checks
"""
import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from core.helpers import create_tracked_task
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Priority levels for background tasks."""
    CRITICAL = "critical"  # Must run (health checks)
    HIGH = "high"          # Important (memory consolidation)
    MEDIUM = "medium"      # Beneficial (growth evaluation)
    LOW = "low"            # Optional (cleanup)


@dataclass
class ScheduledTask:
    """A scheduled background task."""
    name: str
    coro: Callable
    interval_seconds: float
    priority: TaskPriority = TaskPriority.MEDIUM
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_duration_ms: float = 0.0

    def should_run(self) -> bool:
        if not self.enabled:
            return False
        if self.next_run is None:
            return True
        return datetime.now() >= self.next_run

    def mark_run(self, success: bool, duration_ms: float, error: Optional[str] = None):
        self.last_run = datetime.now()
        self.next_run = datetime.now() + timedelta(seconds=self.interval_seconds)
        self.run_count += 1
        self.last_duration_ms = duration_ms
        if not success:
            self.error_count += 1
            self.last_error = error


class BackgroundScheduler:
    """Scheduler for background tasks with priority handling."""

    def __init__(
        self,
        max_concurrent: int = 3,
        idle_threshold_seconds: float = 60.0,
    ):
        self.max_concurrent = max_concurrent
        self.idle_threshold_seconds = idle_threshold_seconds
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._active_tasks: set[asyncio.Task] = set()
        self._idle_since: Optional[datetime] = None
        self._metrics = {
            "total_runs": 0,
            "total_errors": 0,
            "uptime_seconds": 0,
        }
        self._start_time: Optional[datetime] = None

    def register(
        self,
        name: str,
        coro: Callable,
        interval_seconds: float,
        priority: TaskPriority = TaskPriority.MEDIUM,
        start_immediately: bool = False,
    ) -> ScheduledTask:
        """Register a background task.
        
        Args:
            name: Unique task name
            coro: Async callable to execute
            interval_seconds: Run interval in seconds
            priority: Task priority
            start_immediately: Whether to run immediately on start
            
        Returns:
            The registered task
        """
        task = ScheduledTask(
            name=name,
            coro=coro,
            interval_seconds=interval_seconds,
            priority=priority,
            next_run=datetime.now() if start_immediately else datetime.now() + timedelta(seconds=interval_seconds),
        )
        if name in self._tasks:
            logger.warning(f"Overwriting existing task with name: {name}")
        self._tasks[name] = task
        logger.info(f"Registered background task: {name} (interval={interval_seconds}s, priority={priority.value})")
        return task

    def unregister(self, name: str) -> bool:
        """Unregister a background task."""
        if name in self._tasks:
            del self._tasks[name]
            logger.info(f"Unregistered background task: {name}")
            return True
        return False

    def enable(self, name: str) -> bool:
        """Enable a task."""
        if name in self._tasks:
            self._tasks[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a task."""
        if name in self._tasks:
            self._tasks[name].enabled = False
            return True
        return False

    def trigger(self, name: str) -> bool:
        """Immediately trigger a task to run."""
        if name in self._tasks:
            self._tasks[name].next_run = datetime.now()
            logger.info(f"Triggered task: {name}")
            return True
        return False

    def mark_user_activity(self):
        """Mark that user activity has occurred (resets idle timer)."""
        self._idle_since = None

    def is_idle(self) -> bool:
        """Check if system is idle (no user activity for threshold)."""
        if self._idle_since is None:
            self._idle_since = datetime.now()
        return (datetime.now() - self._idle_since).total_seconds() > self.idle_threshold_seconds

    async def start(self):
        """Start the background scheduler."""
        if self._running:
            logger.warning("Background scheduler already running")
            return

        self._running = True
        self._start_time = datetime.now()
        self._task = create_tracked_task(self._run_loop(), name="background_run_loop")
        logger.info("Background scheduler started")

    async def stop(self):
        """Stop the background scheduler."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._task = None

        for task in list(self._active_tasks):
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self._active_tasks.clear()

        if self._start_time:
            self._metrics["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds()

        logger.info("Background scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                await self._check_and_run_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5.0)

    async def _check_and_run_tasks(self):
        """Check tasks and run those that are due."""
        for name, task in list(self._tasks.items()):
            if not task.should_run():
                continue

            if len(self._active_tasks) >= self.max_concurrent:
                break

            if task.priority in (TaskPriority.LOW, TaskPriority.MEDIUM):
                if not self.is_idle():
                    continue

            t = create_tracked_task(self._execute_task(task), name="execute_task")
            self._active_tasks.add(t)
            t.add_done_callback(self._active_tasks.discard)

    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task with error handling."""
        import time

        start_time = time.perf_counter()
        success = False
        error_msg = None

        try:
            logger.debug(f"Running task: {task.name}")
            await task.coro()
            success = True
            logger.debug(f"Task completed: {task.name}")
        except asyncio.CancelledError:
            logger.debug(f"Task cancelled: {task.name}")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task failed: {task.name} - {error_msg}")
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            task.mark_run(success, duration_ms, error_msg)
            self._metrics["total_runs"] += 1
            if not success:
                self._metrics["total_errors"] += 1

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status and metrics."""
        return {
            "running": self._running,
            "uptime_seconds": self._metrics["uptime_seconds"],
            "total_runs": self._metrics["total_runs"],
            "total_errors": self._metrics["total_errors"],
            "active_tasks": len(self._active_tasks),
            "is_idle": self.is_idle(),
            "registered_tasks": [
                {
                    "name": t.name,
                    "enabled": t.enabled,
                    "priority": t.priority.value,
                    "interval_seconds": t.interval_seconds,
                    "run_count": t.run_count,
                    "error_count": t.error_count,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                    "last_duration_ms": t.last_duration_ms,
                }
                for t in self._tasks.values()
            ],
        }


def create_default_scheduler(
    dream_engine=None,
    growth_engine=None,
    evolution_engine=None,
    memory=None,
    agent=None,
    cognitive_bridge=None,
) -> BackgroundScheduler:
    """Create a scheduler with default Jarvis background tasks.
    
    Args:
        dream_engine: DreamEngine instance for memory consolidation
        growth_engine: SelfGrowthEngine for self-improvement
        evolution_engine: AutonomousEvolution for code changes
        memory: ConversationMemory for cleanup
        agent: JarvisAgent for health checks
        cognitive_bridge: CognitiveEvolutionBridge for integrated dreaming
        
    Returns:
        Configured BackgroundScheduler
    """
    scheduler = BackgroundScheduler(
        max_concurrent=2,
        idle_threshold_seconds=30.0,
    )

    # NOTE (2026-09 ground-up review): the old health_check tick ran an
    # LLM availability probe every 60s and only LOGGED the result —
    # 60 HTTP round-trips/hour of pure cost. Real health monitoring
    # lives in the /health endpoint and the health manager. Removed.

    if cognitive_bridge:
        async def cognitive_dream():
            logger.info("Running integrated cognitive dream cycle")
            try:
                result = await cognitive_bridge.run_sleep_cycle(duration_minutes=0.5)
                logger.info(f"Cognitive dream: {result['total_insights']} insights, "
                          f"{len(result['evolution_proposals'])} proposals")
            except Exception as e:
                logger.error(f"Cognitive dream failed: {e}")

        scheduler.register(
            "cognitive_dream",
            cognitive_dream,
            interval_seconds=600.0,
            priority=TaskPriority.HIGH,
        )

    elif dream_engine:
        async def dream_consolidation():
            from core.dream import DreamPhase
            logger.info("Running dream consolidation (light phase)")
            try:
                report = await dream_engine.dream(phase=DreamPhase.LIGHT, duration=2.0)
                logger.info(f"Dream complete: {report.memories_consolidated} memories consolidated")
            except Exception as e:
                logger.error(f"Dream consolidation failed: {e}")

        scheduler.register(
            "dream_consolidation",
            dream_consolidation,
            interval_seconds=300.0,
            priority=TaskPriority.HIGH,
        )

    if growth_engine:
        async def growth_evaluation():
            logger.info("Running growth evaluation")
            try:
                summary = growth_engine.get_growth_summary()
                logger.info(f"Growth: {summary['implemented']} improvements implemented")
            except Exception as e:
                logger.error(f"Growth evaluation failed: {e}")

        scheduler.register(
            "growth_evaluation",
            growth_evaluation,
            interval_seconds=600.0,
            priority=TaskPriority.MEDIUM,
        )

    if memory:
        async def memory_cleanup():
            stats = memory.get_context_stats()
            logger.debug(f"Memory stats: {stats['total_messages']} messages")
            if stats["total_messages"] > 100:
                memory._trim_if_needed()
                logger.info("Memory trimmed due to size")

        scheduler.register(
            "memory_cleanup",
            memory_cleanup,
            interval_seconds=120.0,
            priority=TaskPriority.LOW,
        )

    async def skills_md_export():
        try:
            from core.tool_learning import get_tool_learning_engine
            from pathlib import Path
            
            engine = get_tool_learning_engine()
            skills_path = Path(__file__).parent.parent / "SKILLS.md"
            result = engine.export_to_skills_md(skills_path)
            if result.get("tools_updated", 0) > 0 or result.get("patterns_added", 0) > 0:
                logger.info(f"SKILLS.md updated: {result}")
        except Exception as e:
            logger.debug(f"SKILLS.md export skipped: {e}")

    scheduler.register(
        "skills_md_export",
        skills_md_export,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def bio_tick():
        try:
            from core.bio_integration import get_bio_integration
            bio = get_bio_integration()
            # bio.tick() advances circadian internally (dt/60). The old
            # explicit cr.tick(dt_minutes=1.0) here advanced circadian a
            # SECOND time per tick — physiology ran at 2x real time.
            # bio.tick is wall-clock anchored now, so dt is advisory.
            state = bio.tick()
            if not state.is_homeostatic:
                logger.debug(f"Bio-tick: arousal={state.arousal:.2f} valence={state.valence:+.2f} "
                           f"stress={state.stress_phase} circadian={state.circadian_phase}")
        except Exception as e:
            logger.debug(f"Bio tick skipped: {e}")

    scheduler.register(
        "bio_tick",
        bio_tick,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def world_model_tick():
        try:
            from core.world_model import get_world_model
            state = get_world_model().tick(dt=1.0)
            if state.get("pending_errors", 0) > 10:
                logger.debug(f"World model tick: {state}")
        except Exception as e:
            logger.debug(f"World model tick skipped: {e}")

    scheduler.register(
        "world_model_tick",
        world_model_tick,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def cognitive_state_tick():
        try:
            from core.cognitive_state_machine import get_cognitive_state_machine
            from core.bio_integration import get_bio_integration
            from core.predictive_processing import get_predictive_processor
            from core.circadian_rhythm import get_circadian_rhythm
            csm = get_cognitive_state_machine()
            bio_state = None
            pred_state = None
            try:
                bio = get_bio_integration()
                bs = bio.tick(dt=0.016)
                bio_state = {
                    "stress_phase": bs.stress_phase,
                    "operational_phase": getattr(bs, "operational_phase", "peak_engagement"),
                    "processing_fatigue": getattr(bs, "processing_fatigue", bs.sleep_pressure),
                }
                try:
                    cr = get_circadian_rhythm()
                    cr_state = cr.get_state()
                    bio_state["operational_phase"] = cr_state.get("operational_phase", bio_state["operational_phase"])
                    bio_state["processing_fatigue"] = cr_state.get("processing_fatigue", bio_state["processing_fatigue"])
                except Exception:
                    pass
            except Exception:
                pass
            try:
                pp = get_predictive_processor()
                pred_state = pp.get_state()
            except Exception:
                pass
            csm.tick(bio_state=bio_state, prediction_state=pred_state)
            try:
                # Belief passive decay + idle curiosity growth — pp.tick
                # had zero callers, so beliefs only mutated on user input.
                get_predictive_processor().tick(dt=30.0)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Cognitive state tick skipped: {e}")

    scheduler.register(
        "cognitive_state_tick",
        cognitive_state_tick,
        interval_seconds=30.0,
        priority=TaskPriority.LOW,
    )

    async def neuroplasticity_consolidation():
        try:
            from core.bio_integration import get_bio_integration
            bio = get_bio_integration()
            result = bio.consolidate(duration_factor=0.5)
            if result.get("pruned", 0) > 0 or result.get("consolidated", 0) > 0:
                logger.debug(f"Neuroplasticity consolidation: {result}")
        except Exception as e:
            logger.debug(f"Neuroplasticity consolidation skipped: {e}")

    scheduler.register(
        "neuroplasticity_consolidation",
        neuroplasticity_consolidation,
        interval_seconds=3600.0,
        priority=TaskPriority.LOW,
    )

    async def dream_consolidation():
        try:
            from core.cognitive_state_machine import get_cognitive_state_machine
            csm = get_cognitive_state_machine()
            # Gate via the state machine's own predicate (should_consolidate
            # had zero logic callers); EVOLVING still consolidates as before.
            if csm.should_consolidate() or csm.current_state.value == "evolving":
                from core.memory_consolidation import get_consolidation_engine
                engine = get_consolidation_engine()
                engine.notify_activity()
                result = engine.force_consolidation()
                if result:
                    logger.debug(f"Dream consolidation: count={result.get('consolidation_count', 0)}, "
                                f"tracked={result.get('tracked_memories', 0)}")
                try:
                    from core.circadian_rhythm import get_circadian_rhythm
                    circ = get_circadian_rhythm()
                    circ.register_sleep_onset(duration_hours=2.0)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Dream consolidation skipped: {e}")

    scheduler.register(
        "dream_consolidation",
        dream_consolidation,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def spontaneous_thought_generation():
        try:
            from core.spontaneous_thought import get_spontaneous_thought_generator
            from core.cognitive_state_machine import get_cognitive_state_machine
            from core.bio_integration import get_bio_integration
            from core.predictive_processing import get_predictive_processor
            stg = get_spontaneous_thought_generator()
            csm = get_cognitive_state_machine()
            if csm.should_daydream():
                bio_state = None
                pred_state = None
                try:
                    bio_state = get_bio_integration().tick(dt=0.0)
                except Exception:
                    pass
                try:
                    pred_state = get_predictive_processor().get_state()
                except Exception:
                    pass
                thought = stg.tick(
                    cognitive_state=csm.get_state(),
                    prediction_state=pred_state,
                    bio_state=bio_state,
                )
                if thought:
                    logger.debug(f"Spontaneous thought ({thought.source}): {thought.content[:80]}")
                    # Insights generated during dreaming feed the
                    # HIGH_MASTERY → EVOLVING gate (record_insight had
                    # zero callers, making that transition unreachable
                    # except by timeout).
                    try:
                        if csm.current_state.value == "dreaming":
                            csm.record_insight()
                    except Exception:
                        pass
            stg.clear_expressed()
        except Exception as e:
            logger.debug(f"Spontaneous thought generation skipped: {e}")

    scheduler.register(
        "spontaneous_thought",
        spontaneous_thought_generation,
        interval_seconds=600.0,
        priority=TaskPriority.LOW,
    )

    async def interoception_tick():
        try:
            from core.interoception import get_interoception_engine, get_interoceptive_modulator
            from core.bio_integration import get_bio_integration
            ie = get_interoception_engine()
            mod = get_interoceptive_modulator()
            bio = None
            try:
                bio = get_bio_integration()
            except Exception:
                pass
            nt_state = {}
            bio_state = {}
            if bio:
                try:
                    ctx = bio.get_cognitive_context()
                    nt_state = ctx.get("neurotransmitters", {})
                    ps = ctx.get("physiological_state", {})
                    bio_state = {"arousal": ps.get("arousal", 0.5), "valence": ps.get("valence", 0.0),
                                "stress_intensity": ps.get("allostatic_load", 0.0)}
                except Exception:
                    pass
            feeling = ie.tick(nt_state=nt_state, bio_state=bio_state)
            mod.update(feeling, bio_state.get("stress_intensity", 0.0))
            try:
                awareness = ie.get_awareness()
                granularity = ie.get_granularity()
                if awareness < 0.3:
                    mod.broaden()
                elif awareness > 0.7 and granularity > 0.5:
                    mod.narrow()
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Interoception tick skipped: {e}")

    scheduler.register(
        "interoception_tick",
        interoception_tick,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def global_workspace_compete():
        try:
            from core.global_workspace import get_global_workspace, CoalitionType
            from core.interoception import get_interoception_engine
            from core.cognitive_state_machine import get_cognitive_state_machine
            gw = get_global_workspace()
            ie = get_interoception_engine()
            live_arousal = 0.5
            try:
                feeling_state = ie.get_state()
                current = feeling_state.get("current_feeling", {})
                quality = current.get("quality", "grounded")
                arousal = current.get("arousal", 0.5)
                # Real arousal feeds the tick below (was hardcoded 0.5,
                # which made the arousal-modulated ignition threshold inert
                # during idle ticks).
                try:
                    live_arousal = float(arousal)
                except (TypeError, ValueError):
                    live_arousal = 0.5
                gw.submit_coalition(CoalitionType.INTEROCEPTIVE, f"feeling {quality}", arousal * 0.6)
                valence = current.get("valence", 0.0)
                if abs(valence) > 0.5:
                    gw.amplify_context("emotion", strength_boost=abs(valence) * 0.15)
                # Real FeelingQuality values (the old check tested
                # "stressed"/"anxious", which are not enum values and never fired)
                if quality in ("tense", "overwhelmed", "restless"):
                    gw.amplify_context("threat", strength_boost=0.2)
            except Exception:
                pass
            try:
                csm = get_cognitive_state_machine()
                csm_state = csm.get_state()
                state_name = csm_state.get("state", "awake")
                if state_name in ("contemplating", "dreaming"):
                    gw.amplify_context("metacognition", strength_boost=0.15)
            except Exception:
                pass
            try:
                from core.circadian_rhythm import get_circadian_rhythm
                cr = get_circadian_rhythm()
                phase = cr.get_state().get("operational_phase", "peak_engagement")
                if phase == "peak_engagement":
                    gw.set_ignition_threshold(0.3)
                    gw.set_broadcast_duration(3000.0)
                elif phase == "deep_processing":
                    gw.set_ignition_threshold(0.5)
                    gw.set_broadcast_duration(5000.0)
                elif phase in ("recovery_consolidation", "light_demand"):
                    gw.set_ignition_threshold(0.6)
                    gw.set_broadcast_duration(2000.0)
            except Exception:
                pass

            # Real coalition sources — the audit found only 2 ever competed
            # (user input + interoceptive feelings), so the cross-modality
            # boost (needs >= 3 types) could never fire and the competition
            # machinery was mostly idle. Each source below is a live
            # subsystem with a genuine salience signal.

            # METACOGNITIVE: self-model mode shifts (only when the
            # self-model is actually engaged — automatic mode is silence)
            try:
                from core.recursive_self_model import get_recursive_self_model
                sm = get_recursive_self_model()
                sm_state = sm.get_state()
                sm_mode = sm_state.get("mode", "automatic")
                if sm_mode != "automatic":
                    calib = sm_state.get("confidence_calibration", 0.5)
                    strength = 0.55 + (0.2 if sm_mode == "meta_aware" else 0.1)
                    gw.submit_coalition(
                        CoalitionType.METACOGNITIVE,
                        f"Self-model: {sm_mode}, calibration {calib:.2f}",
                        strength=strength,
                        sources=["self_model"],
                    )
            except Exception:
                pass

            # PERCEPTUAL: world-model prediction errors (physical schemas
            # + learned JEPA terminal predictions)
            try:
                from core.world_model import get_world_model
                wm_state = get_world_model().get_state()
                pending = int(wm_state.get("pending_errors", 0))
                if pending > 2:
                    gw.submit_coalition(
                        CoalitionType.PERCEPTUAL,
                        f"World model: {pending} unverified predictions",
                        strength=min(0.9, 0.4 + pending * 0.05),
                        sources=["world_model"],
                    )
            except Exception:
                pass

            try:
                from core.jepa_world_model import get_jepa_world_model
                jwm = get_jepa_world_model()
                jepa_err = jwm.get_last_error()
                if jepa_err is not None and jepa_err > 0.4:
                    gw.submit_coalition(
                        CoalitionType.PERCEPTUAL,
                        f"Terminal world diverged from prediction (err {jepa_err:.2f})",
                        strength=min(0.85, 0.45 + jepa_err * 0.4),
                        sources=["jepa_world"],
                    )
            except Exception:
                pass

            # CONCEPTUAL: spontaneous idle thoughts become coalitions —
            # they used to reach only the prompt as text, never competing
            try:
                from core.spontaneous_thought import get_spontaneous_thought_generator
                pending_thoughts = get_spontaneous_thought_generator().peek_thoughts(limit=1)
                if pending_thoughts:
                    gw.submit_coalition(
                        CoalitionType.CONCEPTUAL,
                        f"Idle thought: {pending_thoughts[0].content[:70]}",
                        strength=0.55,
                        sources=["spontaneous_thought"],
                    )
            except Exception:
                pass

            result = gw.tick(arousal=live_arousal)
            if result.get("state") == "ignited":
                logger.debug(f"Workspace ignition: {result.get('broadcast', '')[:80]}")
                echo = gw.reverberate(limit=2)
                if echo:
                    logger.debug(f"Workspace reverberation: {'; '.join(echo)[:120]}")
            gw.decay_ignitions(dt=1.0)
        except Exception as e:
            logger.debug(f"Global workspace compete skipped: {e}")

    scheduler.register(
        "global_workspace_compete",
        global_workspace_compete,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def stream_consciousness_generate():
        try:
            from core.stream_consciousness import get_stream_of_consciousness
            from core.global_workspace import get_global_workspace
            from core.interoception import get_interoception_engine
            from core.cognitive_state_machine import get_cognitive_state_machine
            soc = get_stream_of_consciousness()
            ws = get_global_workspace()
            ie = get_interoception_engine()
            csm = get_cognitive_state_machine()
            broadcast = ws.get_current_broadcast()
            feeling_str = None
            try:
                fs = ie.get_state()
                q = fs.get("current_feeling", {}).get("quality")
                if q:
                    feeling_str = f"feeling {q}"
            except Exception:
                pass
            csm_state = csm.get_state()
            state_name = csm_state.get("state", "awake")

            # Bicameral dial (knot loop): dreaming tilts the blend toward
            # the internal command — the mind hears more of itself, the
            # echo streaks rise, and the walled garden pulls back.
            soc.set_bicameral({"dreaming": 0.8, "contemplating": 0.5}.get(state_name, 0.3))

            try:
                from core.circadian_rhythm import get_circadian_rhythm
                rhythm = get_circadian_rhythm()
                rhythm_state = rhythm.get_state()
                phase = rhythm_state.get("operational_phase", "peak_engagement")
                if phase == "deep_processing":
                    soc.set_rhythm(800.0)
                elif phase == "peak_engagement":
                    soc.set_rhythm(200.0)
                elif phase in ("recovery_consolidation", "light_demand"):
                    soc.set_rhythm(500.0)
            except Exception:
                pass

            try:
                from core.predictive_processing import get_predictive_processor
                pp = get_predictive_processor()
                if pp.should_explore():
                    topics = pp.get_curiosity_topics(top_k=2)
                    if topics:
                        topic_name = topics[0][0] if len(topics[0]) > 0 else "awareness"
                        from core.stream_consciousness import ThoughtType
                        soc.add_thought(
                            f"Curious: what about {topic_name}?",
                            thought_type=ThoughtType.WONDER,
                            topic=topic_name,
                            context={"source": "predictive_drive"},
                        )
            except Exception:
                pass

            if state_name in ("dreaming", "evolving"):
                if state_name == "dreaming":
                    # Dream mode: surreal chaining (stream_consciousness's
                    # generate_dream_thought existed but had zero production
                    # callers — the advertised dream feature never ran).
                    soc.generate_dream_thought()
                else:
                    soc.generate_next(
                        from_global_workspace=broadcast,
                        from_feeling=feeling_str,
                        cognitive_state=csm_state,
                    )
                stream_state = soc.get_state()
                if stream_state.get("stream_length", 0) > 5:
                    try:
                        from core.predictive_processing import get_predictive_processor
                        pp = get_predictive_processor()
                        pp_state = pp.get_state()
                        if pp_state.get("free_energy", 0) > 0.6:
                            soc.fork("Processing uncertainty: exploring alternative path",
                                     topic=stream_state.get("current_topic", "inquiry"))
                    except Exception:
                        pass
            try:
                stream_state = soc.get_state()
                if stream_state.get("fork_count", 0) > 3:
                    recent = soc.get_recent(limit=3)
                    contents = [r.get("content", "")[:40] for r in recent if r.get("content")]
                    if len(contents) >= 2:
                        soc.merge_substreams(contents, topic="consolidation", salience=0.6)
            except Exception:
                pass

            # DES probe: classify the current mental state so des_distribution
            # tracks the stream against the empirical DES distribution
            # (Heavey & Hurlburt — inner speech is a minority of moments)
            soc.sample_mental_state()
        except Exception as e:
            logger.debug(f"Stream consciousness generate skipped: {e}")

    scheduler.register(
        "stream_consciousness",
        stream_consciousness_generate,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def self_model_reflection():
        try:
            from core.recursive_self_model import get_recursive_self_model, SelfModelLevel
            from core.cognitive_state_machine import get_cognitive_state_machine
            from core.stream_consciousness import get_stream_of_consciousness
            from core.predictive_processing import get_predictive_processor
            rsm = get_recursive_self_model()
            csm = get_cognitive_state_machine()
            soc = get_stream_of_consciousness()
            stream_state = soc.get_state()
            if stream_state.get("stream_length", 0) > 0:
                rsm.observe(SelfModelLevel.FIRST_ORDER, "monitoring background stream activity", confidence=0.5)
                # SECOND_ORDER: observe the observation. The L2 level existed
                # in the hierarchy but nothing in production ever submitted
                # it, so the mode ladder could not climb past MONITORED and
                # detect_conflicts() was always empty.
                if stream_state.get("stream_length", 0) > 5:
                    coherence = stream_state.get("coherence", 0.5)
                    rsm.observe(
                        SelfModelLevel.SECOND_ORDER,
                        f"stream coherence {coherence:.2f} across "
                        f"{stream_state.get('stream_length', 0)} thoughts",
                        confidence=0.6,
                    )
            csm_state = csm.get_state()
            rsm.tick(cognitive_state=csm_state)
            try:
                pp = get_predictive_processor()
                pred_state = pp.get_state()
                free_energy = pred_state.get("free_energy", 0.0)
                # Real load proxy: prediction error + active stream threads
                # (the previous call used kwargs that don't exist on
                # assess_cognitive_load(external_load) and raised TypeError
                # into a silent except every tick).
                external_load = min(1.0, free_energy + 0.1 * max(1, stream_state.get("stream_length", 0)))
                rsm.assess_cognitive_load(external_load=external_load)
            except Exception as e:
                logger.debug(f"Cognitive load assessment skipped: {e}")
            try:
                conflicts = rsm.detect_conflicts()
                if conflicts:
                    logger.debug(f"Self-model detected {len(conflicts)} conflict(s)")
                    from core.stream_consciousness import ThoughtType
                    from core.stream_consciousness import get_stream_of_consciousness as _get_soc
                    _soc = _get_soc()
                    _soc.add_thought(
                        f"Self-model conflict: {len(conflicts)} perspective mismatch(es)",
                        thought_type=ThoughtType.METACOGNITION,
                        salience=0.7,
                    )
            except Exception:
                pass
            try:
                introspection = rsm.introspect()
                should_throttle = introspection.get("should_throttle", False)
                recommended_mode = introspection.get("recommended_mode", "automatic")
                if should_throttle:
                    logger.debug(f"Self-model recommends throttling (load={introspection.get('cognitive_load', 0):.2f})")
                if should_throttle and recommended_mode in ("reflective", "meta_aware"):
                    from core.stream_consciousness import ThoughtType
                    from core.stream_consciousness import get_stream_of_consciousness as _get_soc
                    _soc = _get_soc()
                    _soc.add_thought(
                        f"Load too high for {recommended_mode} — throttling",
                        thought_type=ThoughtType.METACOGNITION,
                        salience=0.6,
                    )
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Self-model reflection skipped: {e}")

    scheduler.register(
        "self_model_reflection",
        self_model_reflection,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def emotion_regulation_tick():
        try:
            from core.emotion_regulation import get_emotion_regulation
            from core.interoception import get_interoception_engine, get_allostasis_predictor
            from core.cognitive_state_machine import get_cognitive_state_machine
            ie = get_interoception_engine()
            ere = get_emotion_regulation()
            ap = get_allostasis_predictor()
            csm = get_cognitive_state_machine()
            ie_state = ie.get_state()
            current_feeling = ie_state.get("current_feeling", {})
            if current_feeling:
                csm_state = csm.get_state() if csm else None
                prediction_errors = ie_state.get("avg_channel_errors", {})
                try:
                    from core.predictive_processing import get_predictive_processor
                    pp = get_predictive_processor()
                    pp_state = pp.get_state()
                    free_energy = pp_state.get("free_energy", 0.0)
                    if free_energy > 0.3:
                        prediction_errors["predictive_processing"] = free_energy
                except Exception:
                    pass
                reg = ere.regulate(
                    feeling_state={
                        "valence": current_feeling.get("valence", 0.0),
                        "arousal": current_feeling.get("arousal", 0.5),
                        "dominance": current_feeling.get("dominance", 0.5),
                        "quality": current_feeling.get("quality", "grounded"),
                        "awareness_score": ie_state.get("awareness_score", 0.5),
                        "prediction_errors": prediction_errors,
                    },
                    cognitive_state=csm_state,
                )
                strategy = reg.get("strategy", "")
                if strategy in ("cognitive_reappraisal", "body_budget_restoration"):
                    logger.debug(f"Emotion regulation: {strategy} applied")
                # Condensation-by-load (Vygotsky): high arousal / depleted
                # budget fragments inner speech; calm expands it again.
                try:
                    arousal = float(current_feeling.get("arousal", 0.5))
                    # regulate() returns budget_after/budget_before as
                    # floats — the old code read reg["budget"]["overall"],
                    # a key that only exists in get_state(), so the budget
                    # term of condensation was a constant 0.5.
                    overall = float(reg.get("budget_after",
                                            reg.get("budget_before", 0.5)))
                    condensation = max(0.0, min(1.0,
                        max(0.0, arousal - 0.5) * 1.2 + (1.0 - overall) * 0.3))
                    from core.stream_consciousness import get_stream_of_consciousness as _get_soc_for_cond
                    _soc_cond = _get_soc_for_cond()
                    _soc_cond.set_condensation(condensation)
                    # High-arousal moments elicit second-person
                    # self-instruction (Dolcos & Albarracín 2014):
                    # self-addressed regulation under stress.
                    if arousal > 0.65 and random.random() < 0.3:
                        _soc_cond.generate_self_instruction(
                            topic=_soc_cond.get_state().get("current_topic", ""))
                except Exception:
                    pass
                if strategy == "body_budget_restoration" and reg.get("budget_effect", 0.0) > 0:
                    try:
                        ap.apply_recovery(amount=reg["budget_effect"] * 0.5, source="emotion_regulation")
                    except Exception:
                        pass
                if reg.get("budget_effect", 0.0) > 0.05:
                    try:
                        from core.stream_consciousness import get_stream_of_consciousness, ThoughtType
                        soc = get_stream_of_consciousness()
                        soc.add_thought(
                            f"Regulation restored budget ({reg['budget_effect']:.2f}), coherence rising",
                            thought_type=ThoughtType.FEELING,
                            salience=0.6,
                        )
                    except Exception:
                        pass
                if strategy and strategy != "cooldown":
                    try:
                        from core.emotional_memory import get_emotional_memory
                        em = get_emotional_memory()
                        pre_quality = current_feeling.get("quality", "")
                        post_quality = reg.get("reappraisal", pre_quality)
                        if post_quality and post_quality != pre_quality:
                            em.record_feeling_shift(
                                pre_feeling=pre_quality,
                                post_feeling=post_quality,
                                regulation=strategy,
                                trigger="background_regulation",
                                valence=current_feeling.get("valence", 0.0),
                                arousal=current_feeling.get("arousal", 0.5),
                            )
                    except Exception:
                        pass
            ap.tick(dt=1.0)
        except Exception as e:
            logger.debug(f"Emotion regulation tick skipped: {e}")

    scheduler.register(
        "emotion_regulation_tick",
        emotion_regulation_tick,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def allostasis_prediction():
        try:
            from core.interoception import get_interoception_engine, get_allostasis_predictor
            from core.bio_integration import get_bio_integration
            ie = get_interoception_engine()
            ap = get_allostasis_predictor()
            ie_state = ie.get_state()
            bio_state = None
            try:
                bio = get_bio_integration()
                bio_state_raw = bio.tick(dt=0.0)
                bio_state = {
                    "stress_intensity": bio_state_raw.stress_phase if hasattr(bio_state_raw, 'stress_phase') else 0.0,
                    "cognitive_capacity": bio_state_raw.cognitive_capacity if hasattr(bio_state_raw, 'cognitive_capacity') else 0.8,
                }
            except Exception:
                pass
            feeling_state = {
                "arousal": ie_state.get("current_feeling", {}).get("arousal", 0.5),
                "valence": ie_state.get("current_feeling", {}).get("valence", 0.0),
                "prediction_errors": ie_state.get("avg_channel_errors", {}),
            }
            result = ap.predict_needs(feeling_state, bio_state=bio_state)
            if result.get("regulation_need") in ("high", "urgent"):
                logger.debug(f"Allostasis: regulation_need={result['regulation_need']} load={result['allostatic_load']:.2f}")
        except Exception as e:
            logger.debug(f"Allostasis prediction skipped: {e}")

    scheduler.register(
        "allostasis_prediction",
        allostasis_prediction,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def emotional_memory_session():
        try:
            from core.emotional_memory import get_emotional_memory
            from core.affective_context import get_affective_context
            from core.interoception import get_interoception_engine
            from core.emotion_regulation import get_emotion_regulation
            em = get_emotional_memory()
            ac = get_affective_context()
            ie = get_interoception_engine()
            ie_state = ie.get_state()
            current_feeling = ie_state.get("current_feeling", {})
            if current_feeling:
                val = current_feeling.get("valence", 0.0)
                ar = current_feeling.get("arousal", 0.5)
                qual = current_feeling.get("quality", "grounded")
                em.tag_interaction(
                    valence=val,
                    arousal=ar,
                    dominance=current_feeling.get("dominance", 0.5),
                    quality=qual,
                    awareness=ie_state.get("awareness_score", 0.5),
                    topic="background_tick",
                    regulation_strategy="",
                )
                ac.record_feeling_snapshot(valence=val, arousal=ar, quality=qual)
                try:
                    ere = get_emotion_regulation()
                    reg_state = ere.get_state()
                    latest = reg_state.get("recent_regulations", [])
                    if latest:
                        last_reg = latest[0] if isinstance(latest[0], dict) else {}
                        if last_reg.get("from_feeling") and last_reg.get("to_feeling"):
                            if last_reg["from_feeling"] != last_reg["to_feeling"]:
                                em.record_feeling_shift(
                                    pre_feeling=last_reg.get("from_feeling", ""),
                                    post_feeling=last_reg.get("to_feeling", ""),
                                    regulation=last_reg.get("strategy", ""),
                                    trigger="background_regulation",
                                    valence=val,
                                    arousal=ar,
                                )
                except Exception:
                    pass
            em.prune_old_tags(max_age_hours=48)
            try:
                from core.emotional_state import get_emotional_state_manager
                from core.stream_consciousness import get_stream_of_consciousness, ThoughtType
                esm = get_emotional_state_manager()
                summary = esm.summarize_emotional_state()
                if summary and len(summary) > 10:
                    soc = get_stream_of_consciousness()
                    soc.add_thought(
                        f"Emotional state: {summary[:100]}",
                        thought_type=ThoughtType.FEELING,
                        topic="emotional_state",
                        context={"source": "emotional_memory_session"},
                    )
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Emotional memory session tick skipped: {e}")

    scheduler.register(
        "emotional_memory_session",
        emotional_memory_session,
        # Demoted 120->600s: this tick tags synthetic "background_tick"
        # interactions into emotional memory, which dominates the mood
        # trend with idle data when it fires more often than real use.
        interval_seconds=600.0,
        priority=TaskPriority.LOW,
    )

    async def three_layer_consolidation():
        try:
            from core.three_layer_context import get_three_layer_context
            tlc = get_three_layer_context()
            stats = tlc.get_stats()
            l2_size = stats.get("l2", {}).get("total_chars", 0)
            if l2_size > 2000:
                count = tlc.consolidate_l2_to_l3(force=False)
                if count > 0:
                    logger.debug(f"Three-layer: consolidated {count} L2 entries to L3")
        except Exception as e:
            logger.debug(f"Three-layer consolidation skipped: {e}")

    scheduler.register(
        "three_layer_consolidation",
        three_layer_consolidation,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def claims_maintenance():
        try:
            from core.claims import get_claims_registry
            cr = get_claims_registry()
            expired = cr.cleanup_expired()
            if expired > 0:
                logger.debug(f"Claims: cleaned up {expired} expired claims")
            for cid in list(cr._claims.keys())[:5]:
                cr.heartbeat(cid)
        except Exception as e:
            logger.debug(f"Claims maintenance skipped: {e}")

    scheduler.register(
        "claims_maintenance",
        claims_maintenance,
        interval_seconds=600.0,
        priority=TaskPriority.LOW,
    )

    # NOTE (2026-09 ground-up review): load_balancer_tick,
    # autonomous_goals_tick, file_change_monitor and
    # proactive_intelligence_tick were pure log-only reads — the prompt
    # builder fetches the same state on demand, so the ticks duplicated
    # work for output nobody consumed. Removed.

    async def active_presence_tick():
        try:
            from core.active_presence import get_active_presence
            ap = get_active_presence()
            state = ap.tick()
            if state.get("transition"):
                from core.autonomous_observer import get_autonomous_observer
                ao = get_autonomous_observer()
                ao.observe_user_habit("presence_transition", f"User {state['transition']}")
                logger.debug(f"Presence: {state['presence']} transition={state['transition']}")
        except Exception as e:
            logger.debug(f"Active presence tick skipped: {e}")

    scheduler.register(
        "active_presence_tick",
        active_presence_tick,
        interval_seconds=30.0,
        priority=TaskPriority.LOW,
    )

    async def autonomous_observer_tick():
        try:
            from core.autonomous_observer import get_autonomous_observer
            ao = get_autonomous_observer()
            state = ao.tick()
            if state.get("notifiable", 0) > 0:
                logger.debug(f"Observer: {state['notifiable']} notifiable observation(s)")
        except Exception as e:
            logger.debug(f"Autonomous observer tick skipped: {e}")

    scheduler.register(
        "autonomous_observer_tick",
        autonomous_observer_tick,
        interval_seconds=60.0,
        priority=TaskPriority.LOW,
    )

    async def goal_anticipation_tick():
        try:
            from core.goal_anticipation import get_goal_anticipation
            ga = get_goal_anticipation()
            state = ga.tick()
            active = state.get("active_anticipations", 0)
            if active > 0:
                logger.debug(f"Anticipation: {active} active anticipation(s)")
        except Exception as e:
            logger.debug(f"Goal anticipation tick skipped: {e}")

    scheduler.register(
        "goal_anticipation_tick",
        goal_anticipation_tick,
        interval_seconds=120.0,
        priority=TaskPriority.LOW,
    )

    # NOTE (2026-09 ground-up review): butler_personality_tick re-observed
    # the SAME last interaction every 60s, double-counting sentiment and
    # skewing the personality adaptation (observe_interaction already
    # fires per turn in _preprocess_turn). proactive_intelligence_tick
    # was a log-only read of state the prompt fetches on demand.
    # Removed.

    async def butler_orchestrator_tick():
        try:
            from core.butler_orchestrator import get_butler_orchestrator
            from core.active_presence import get_active_presence
            from core.goal_anticipation import get_goal_anticipation
            from core.autonomous_observer import get_autonomous_observer
            from core.butler_personality import get_butler_personality
            from core.proactive_intelligence import get_proactive_intelligence
            from core.contextual_memory import get_contextual_memory
            orch = get_butler_orchestrator()
            presence = get_active_presence()
            try:
                presence_state = presence.get_state()
            except Exception as e:
                logger.debug(f"Orchestrator: presence state failed: {e}")
                presence_state = {}
            anticipation = get_goal_anticipation()
            anticipation_state = {}
            try:
                active_anticipations = anticipation.get_pending_anticipations()
                if active_anticipations:
                    anticipation_state = {"active": len(active_anticipations)}
            except Exception as e:
                logger.debug(f"Orchestrator: anticipation state failed: {e}")
            observer_state = {}
            try:
                obs = get_autonomous_observer()
                trajectory = obs.get_emotional_trajectory()
                if trajectory:
                    observer_state = {"emotional_tone": trajectory[-1].get("tone", "neutral")}
            except Exception as e:
                logger.debug(f"Orchestrator: observer state failed: {e}")
            personality_state = {}
            try:
                bp = get_butler_personality()
                personality_state = bp.get_state()
            except Exception as e:
                logger.debug(f"Orchestrator: personality state failed: {e}")
            proactive_state = {}
            try:
                pi = get_proactive_intelligence()
                proactive_state = pi.get_state()
            except Exception as e:
                logger.debug(f"Orchestrator: proactive state failed: {e}")
            contextual_state = {}
            try:
                cm = get_contextual_memory()
                contextual_state = cm.get_state()
            except Exception as e:
                logger.debug(f"Orchestrator: contextual state failed: {e}")
            action = orch.coordinate(
                presence_state=presence_state,
                anticipation_state=anticipation_state,
                observer_state=observer_state,
                personality_state=personality_state,
                proactive_state=proactive_state,
                contextual_state=contextual_state,
            )
            if action:
                logger.debug(f"Butler orchestration: {action.action} ({action.reason})")
        except Exception as e:
            logger.debug(f"Butler orchestrator tick skipped: {e}")

    scheduler.register(
        "butler_orchestrator_tick",
        butler_orchestrator_tick,
        interval_seconds=120.0,
        priority=TaskPriority.LOW,
    )

    async def butler_session_tick():
        try:
            from core.butler_session import get_butler_session
            session = get_butler_session()
            session.apply_memory_decay()
            idle_msg = session.check_idle()
            if idle_msg:
                logger.debug(f"Butler session idle: {idle_msg}")
        except Exception as e:
            logger.debug(f"Butler session tick skipped: {e}")

    scheduler.register(
        "butler_session_tick",
        butler_session_tick,
        interval_seconds=300.0,
        priority=TaskPriority.LOW,
    )

    async def jepa_world_tick():
        """JEPA world model over terminal sessions (mind-diffusion port).

        Harvests (command, output) turns from ExecutionContext, trains
        the latent dynamics model one step, evaluates the one-step-ahead
        prediction error, and — only when a FRESH error was evaluated —
        feeds it to PredictiveProcessor as a learned world-model
        violation (channel "jepa_terminal"), driving the same surprise
        pathway that gates dreaming.
        """
        try:
            from core.jepa_world_model import get_jepa_world_model
            jwm = get_jepa_world_model()
            result = jwm.tick()
            if result.get("fresh_error") is not None:
                jwm.feed_predictive_processor()
        except Exception as e:
            logger.debug(f"JEPA world tick skipped: {e}")

    scheduler.register(
        "jepa_world",
        jepa_world_tick,
        interval_seconds=90.0,
        priority=TaskPriority.LOW,
    )

    return scheduler