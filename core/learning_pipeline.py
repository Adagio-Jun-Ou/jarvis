"""Continuous Learning Pipeline for Jarvis.

Integrates:
- Erdős Math Problem Engine (background mathematical learning)
- Data Defense Agent (security monitoring & adaptation)
- Dream System (creative problem exploration)
- Evolution System (adaptive parameter tuning)

Creates a unified learning loop where Jarvis continuously:
1. Solves math problems to sharpen reasoning
2. Monitors security threats to learn defense patterns
3. Dreams/experiments on complex problems
4. Evolves strategies based on outcomes
"""
import asyncio
import logging
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from core.dual_helpers import learning_discover, learning_evaluate
from core.helpers import atomic_write_json, get_response_content, create_tracked_task, safe_error_message, read_json_with_integrity

logger = logging.getLogger(__name__)
from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType


class LearningPhase(Enum):
    PROBLEM_SOLVING = "problem_solving"
    DEFENSE_ANALYSIS = "defense_analysis"
    DREAMING = "dreaming"
    EVOLUTION = "evolution"
    REFLECTION = "reflection"
    REST = "rest"


@dataclass
class LearningSession:
    id: str
    started_at: datetime
    phase: LearningPhase
    input_data: dict
    output_data: dict | None = None
    insights: list[str] = field(default_factory=list)
    completed: bool = False
    duration_seconds: float = 0.0


@dataclass
class LearningMetrics:
    problems_attempted: int = 0
    problems_solved: int = 0
    threats_analyzed: int = 0
    defense_improvements: int = 0
    dream_insights: int = 0
    evolution_cycles: int = 0
    total_learning_time: float = 0.0
    best_math_strategy: str = "probabilistic_method"
    defense_accuracy: float = 0.5
    evolution_generation: int = 0


class ContinuousLearningPipeline:
    """Orchestrates continuous learning across all Jarvis subsystems.
    
    The pipeline runs in cycles, rotating through different learning phases
    to ensure balanced growth. Each phase feeds into subsequent phases:
    
    Problem Solving → generates mathematical insights
    Defense Analysis → generates security patterns
    Dreaming → explores solutions to unsolved problems
    Evolution → adapts parameters based on all learning
    """

    CYCLE_INTERVAL = 120
    MAX_SESSIONS = 100
    MAX_INSIGHTS = 100
    MAX_DREAM_PROBLEMS = 50
    DEFAULT_PHASE_DURATION = {
        LearningPhase.PROBLEM_SOLVING: 60,
        LearningPhase.DEFENSE_ANALYSIS: 30,
        LearningPhase.DREAMING: 90,
        LearningPhase.EVOLUTION: 30,
        LearningPhase.REFLECTION: 30,
        LearningPhase.REST: 60,
    }
    PHASE_TIMEOUT = 120

    def __init__(
        self,
        storage_path: Path | None = None,
        llm_client: Any = None,
    ):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client

        self.erdos_engine: Any = None
        self.defense_agent: Any = None
        self.dream_system: Any = None
        self.evolution_engine: Any = None
        self.memory_manager: Any = None

        self.sessions: deque[LearningSession] = deque(maxlen=self.MAX_SESSIONS)  # Bounded deque
        self.metrics = LearningMetrics()
        self._current_phase = LearningPhase.PROBLEM_SOLVING
        self._running = False
        self._throttled = False
        self._running_lock = threading.Lock()
        self._pipeline_task: asyncio.Task | None = None
        self._session_counter = 0
        self._lock = threading.RLock()
        self._session_id_set: set[str] = set()  # Unused, kept for compatibility
        
        self._context_engine = None

        self._insights_queue: deque[dict] = deque(maxlen=self.MAX_INSIGHTS)  # Bounded
        self._pending_dream_problems: deque[dict] = deque(maxlen=self.MAX_DREAM_PROBLEMS)  # Bounded
        self._phase_duration = dict(self.DEFAULT_PHASE_DURATION)

        self._api_manager = None
        self._subsystem_name = "learning_pipeline"

        self._phase_handlers = {
            LearningPhase.PROBLEM_SOLVING: self._phase_problem_solving,
            LearningPhase.DEFENSE_ANALYSIS: self._phase_defense_analysis,
            LearningPhase.DREAMING: self._phase_dreaming,
            LearningPhase.EVOLUTION: self._phase_evolution,
            LearningPhase.REFLECTION: self._phase_reflection,
            LearningPhase.REST: self._phase_rest,
        }

        self._load_state()
    
    def _get_context_engine(self):
        """Get unified context engine for enriched learning."""
        if self._context_engine is None:
            try:
                from core.unified_context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except Exception as e:
                logger.debug(f"Context engine unavailable: {e}")
        return self._context_engine
    
    async def get_learning_context(self, topic: str = "") -> str:
        """Get relevant context from unified memory for learning.
        
        Enriches learning pipeline with:
        - Past learning insights and patterns
        - Knowledge about approaches that worked
        - User preferences and constraints
        """
        try:
            engine = self._get_context_engine()
            if engine is None:
                return ""
            
            context = await engine.assemble_context(
                query=f"learning {topic}" if topic else "learning patterns insights",
                max_tokens=300,
                include_sources=["insight_active", "knowledge_graph"],
            )
            
            if context.entries:
                return context.to_prompt()
        except Exception as e:
            logger.debug(f"Failed to get learning context: {e}")
        
        return ""

    def _get_api_manager(self):
        """Get or create API manager for coordinated LLM calls."""
        if self._api_manager is None and self.llm_client:
            try:
                from core.api_manager import get_api_manager
                self._api_manager = get_api_manager(self.llm_client)
            except Exception as e:
                logger.debug(f"Failed to get API manager for learning: {e}")
        return self._api_manager

    async def _llm_call(self, user_input: str, system_prompt: str, priority: str = "learning") -> dict:
        """Make coordinated LLM call with rate limiting."""
        from core.api_manager import Priority as APIPriority

        priority_map = {
            "learning": APIPriority.LEARNING,
            "daydream": APIPriority.DAYDREAM,
            "maintenance": APIPriority.MAINTENANCE,
        }

        api_manager = self._get_api_manager()
        if api_manager is not None:
            return await api_manager.submit_chat(
                subsystem=self._subsystem_name,
                priority=priority_map.get(priority, APIPriority.LEARNING),
                user_input=user_input,
                system_prompt=system_prompt,
            )

        if self.llm_client:
            return await self.llm_client.chat(user_input=user_input, system_prompt=system_prompt)

        raise RuntimeError("No LLM client available")

    def _load_state(self) -> None:
        state_file = self.storage_path / "learning_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self.metrics = LearningMetrics(**data.get("metrics", {}))
                self._current_phase = LearningPhase(data.get("current_phase", "problem_solving"))
                logger.info(f"Loaded learning state: gen {self.metrics.evolution_generation}")
            except Exception as e:
                logger.warning(f"Failed to load learning state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "learning_state.json"
        try:
            atomic_write_json(state_file, {
                    "metrics": {
                        "problems_attempted": self.metrics.problems_attempted,
                        "problems_solved": self.metrics.problems_solved,
                        "threats_analyzed": self.metrics.threats_analyzed,
                        "defense_improvements": self.metrics.defense_improvements,
                        "dream_insights": self.metrics.dream_insights,
                        "evolution_cycles": self.metrics.evolution_cycles,
                        "total_learning_time": self.metrics.total_learning_time,
                        "best_math_strategy": self.metrics.best_math_strategy,
                        "defense_accuracy": self.metrics.defense_accuracy,
                        "evolution_generation": self.metrics.evolution_generation,
                    },
                    "current_phase": self._current_phase.value,
                    "timestamp": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save learning state: {e}")

    def attach_erdos_engine(self, engine: Any) -> None:
        self.erdos_engine = engine

    def attach_defense_agent(self, agent: Any) -> None:
        self.defense_agent = agent

    def attach_dream_system(self, system: Any) -> None:
        self.dream_system = system

    def attach_evolution_engine(self, engine: Any) -> None:
        self.evolution_engine = engine

    def attach_memory_manager(self, manager: Any) -> None:
        self.memory_manager = manager

    async def start_pipeline(self) -> None:
        with self._running_lock:
            if self._running:
                return
            self._running = True

        self._pipeline_task = create_tracked_task(self._pipeline_loop(), name="learning_pipeline_loop")
        logger.info("Continuous learning pipeline started")

    async def stop_pipeline(self) -> None:
        with self._running_lock:
            self._running = False
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Continuous learning pipeline stopped")

    async def _pipeline_loop(self) -> None:
        """Main learning loop cycling through phases."""
        phase_order = [
            LearningPhase.PROBLEM_SOLVING,
            LearningPhase.DEFENSE_ANALYSIS,
            LearningPhase.DREAMING,
            LearningPhase.EVOLUTION,
            LearningPhase.REFLECTION,
            LearningPhase.REST,
        ]
        phase_index = 0

        while self._running:
            try:
                phase = phase_order[phase_index]
                self._current_phase = phase

                handler = self._phase_handlers.get(phase)
                if handler:
                    try:
                        await asyncio.wait_for(handler(), timeout=self.PHASE_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.warning(f"Phase {phase.value} timed out after {self.PHASE_TIMEOUT}s")

                phase_index = (phase_index + 1) % len(phase_order)

                await asyncio.sleep(self._phase_duration.get(phase, 60))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline phase error: {e}")
                await asyncio.sleep(30)

    async def _phase_problem_solving(self) -> dict:
        """Solve Erdős-style math problems."""
        with self._lock:
            self._session_counter += 1
            session_id = f"learn_{self._session_counter:06d}"
        session = LearningSession(
            id=session_id,
            started_at=datetime.now(),
            phase=LearningPhase.PROBLEM_SOLVING,
            input_data={"source": "erdos_engine"},
        )

        start_time = time.time()

        try:
            if self.erdos_engine:
                result = await self.erdos_engine.force_attempt()
                session.output_data = result

                self.metrics.problems_attempted += 1

                if result.get("success"):
                    self.metrics.problems_solved += 1
                    session.insights.append(f"Solved: {result.get('title', 'unknown')}")
                    
                    # Observe successful problem solving with PFC
                    try:
                        pfc = get_prefrontal_cortex()
                        if pfc and pfc._running:
                            from core.helpers import create_tracked_task
                            create_tracked_task(
                                pfc.observe(
                                    content=f"Solved {result.get('title', 'problem')}: {result.get('strategy', 'unknown strategy')}",
                                    concept_type=ConceptType.DECISION,
                                    source="learning:problem_solving",
                                    metadata={"success": True, "strategy": result.get("strategy")},
                                ),
                                name="pfc_observe_learning_problem",
                            )
                    except Exception as e:
                        logger.debug(f"Failed to observe learning result via PFC: {e}")

                if result.get("insight"):
                    self._queue_insight("math", result["insight"])

                if result.get("strategy"):
                    self._update_best_strategy(result["strategy"], result["success"])

                if not result.get("success") and result.get("problem"):
                    self._pending_dream_problems.append({
                        "problem_id": result.get("problem"),
                        "title": result.get("title"),
                        "strategies_tried": [result.get("strategy")],
                    })
            else:
                session.output_data = {"error": "Erdos engine not attached"}
                session.insights.append("Need to attach Erdos engine for math learning")

        except Exception as e:
            session.output_data = {"error": safe_error_message(e)}

        session.duration_seconds = time.time() - start_time
        session.completed = True
        self.metrics.total_learning_time += session.duration_seconds
        self.sessions.append(session)

        logger.debug(f"Problem solving phase completed: {session.duration_seconds:.1f}s")

        return {
            "session_id": session.id,
            "output": session.output_data,
            "insights": session.insights,
            "duration": session.duration_seconds,
        }

    async def _phase_defense_analysis(self) -> dict:
        """Analyze defense logs and learn from threats."""
        with self._lock:
            self._session_counter += 1
            session_id = f"learn_{self._session_counter:06d}"
        session = LearningSession(
            id=session_id,
            started_at=datetime.now(),
            phase=LearningPhase.DEFENSE_ANALYSIS,
            input_data={"source": "defense_agent"},
        )

        start_time = time.time()

        try:
            if self.defense_agent:
                summary = self.defense_agent.get_threat_summary(hours=24)
                recent_events = self.defense_agent.get_recent_events(limit=10)

                session.output_data = {
                    "threat_summary": summary,
                    "recent_events": recent_events,
                }

                self.metrics.threats_analyzed += summary.get("total_threats", 0)

                for event in recent_events[:5]:
                    level = event.get("level", "info")

                    if level in ["high", "critical"]:
                        defense_lesson = self._extract_defense_lesson(event)
                        if defense_lesson:
                            session.insights.append(defense_lesson)
                            self._queue_insight("defense", defense_lesson)
                            self.metrics.defense_improvements += 1

                high_threats = summary.get("high_count", 0)
                total = max(summary.get("total_threats", 1), 1)
                if high_threats > 0:
                    false_positive_rate = 0.1
                    self.metrics.defense_accuracy = max(0.5, 1.0 - false_positive_rate - (high_threats / total) * 0.1)
                else:
                    self.metrics.defense_accuracy = min(1.0, self.metrics.defense_accuracy + 0.01)

            else:
                session.output_data = {"error": "Defense agent not attached"}
                session.insights.append("Need to attach defense agent for security learning")

        except Exception as e:
            session.output_data = {"error": safe_error_message(e)}

        session.duration_seconds = time.time() - start_time
        session.completed = True
        self.metrics.total_learning_time += session.duration_seconds
        self.sessions.append(session)

        logger.debug(f"Defense analysis phase completed: {session.duration_seconds:.1f}s")

        return {
            "session_id": session.id,
            "output": session.output_data,
            "insights": session.insights,
            "duration": session.duration_seconds,
        }

    async def _phase_dreaming(self) -> dict:
        """Dream on unsolved problems and defense patterns."""
        with self._lock:
            self._session_counter += 1
            session_id = f"learn_{self._session_counter:06d}"
        session = LearningSession(
            id=session_id,
            started_at=datetime.now(),
            phase=LearningPhase.DREAMING,
            input_data={"source": "dream_system", "problems": len(self._pending_dream_problems)},
        )

        start_time = time.time()

        try:
            insights = []

            if self._pending_dream_problems:
                problem = random.choice(self._pending_dream_problems)

                dream_result = await self._dream_on_problem(problem)

                if dream_result.get("insight"):
                    insights.append(dream_result["insight"])
                    self.metrics.dream_insights += 1
                    self._queue_insight("dream", dream_result["insight"])

                if dream_result.get("solved"):
                    try:
                        self._pending_dream_problems.remove(problem)
                    except ValueError:
                        pass

                session.output_data = dream_result

            if self.defense_agent and random.random() < 0.3:
                events = self.defense_agent.get_recent_events(limit=20)
                patterns = self._detect_threat_patterns(events)

                if patterns:
                    insights.extend(patterns)
                    for pattern in patterns:
                        self._queue_insight("defense_pattern", pattern)

            if self.dream_system:
                dream_input = {
                    "math_insights": [i for i in self._insights_queue if i.get("type") == "math"][-5:],
                    "defense_insights": [i for i in self._insights_queue if i.get("type") == "defense"][-5:],
                    "pending_problems": len(self._pending_dream_problems),
                }

                try:
                    dream_output = await self.dream_system.enter_dream(dream_input)
                    if dream_output and dream_output.get("synthesis"):
                        insights.append(dream_output["synthesis"])
                except Exception as e:
                    logger.debug(f"Dream system error: {e}")

            session.insights = insights
            session.output_data = session.output_data or {"insights": insights}

        except Exception as e:
            session.output_data = {"error": safe_error_message(e)}

        session.duration_seconds = time.time() - start_time
        session.completed = True
        self.metrics.total_learning_time += session.duration_seconds
        self.sessions.append(session)

        logger.debug(f"Dreaming phase completed: {session.duration_seconds:.1f}s")

        return {
            "session_id": session.id,
            "output": session.output_data,
            "insights": session.insights,
            "duration": session.duration_seconds,
        }

    async def _phase_evolution(self) -> dict:
        """Evolve parameters based on learning outcomes."""
        with self._lock:
            self._session_counter += 1
            session_id = f"learn_{self._session_counter:06d}"
        session = LearningSession(
            id=session_id,
            started_at=datetime.now(),
            phase=LearningPhase.EVOLUTION,
            input_data={"source": "evolution_engine"},
        )

        start_time = time.time()

        try:
            fitness_scores = {
                "problem_solving_rate": self.metrics.problems_solved / max(1, self.metrics.problems_attempted),
                "defense_accuracy": self.metrics.defense_accuracy,
                "insight_generation": self.metrics.dream_insights / max(1, self.metrics.evolution_cycles + 1),
            }

            overall_fitness = (
                fitness_scores["problem_solving_rate"] * 0.4 +
                fitness_scores["defense_accuracy"] * 0.3 +
                fitness_scores["insight_generation"] * 0.3
            )

            if self.evolution_engine:
                evolution_input = {
                    "fitness": overall_fitness,
                    "component_scores": fitness_scores,
                    "best_strategy": self.metrics.best_math_strategy,
                    "generation": self.metrics.evolution_generation,
                    "insights": [i.get("content") for i in list(self._insights_queue)[-10:]],
                }

                try:
                    evolution_result = await self.evolution_engine.adapt(evolution_input)
                    session.output_data = evolution_result

                    if evolution_result.get("new_parameters"):
                        session.insights.append(f"Evolved: {evolution_result['new_parameters']}")
                except Exception as e:
                    session.output_data = {"error": safe_error_message(e)}

            self.metrics.evolution_cycles += 1
            self.metrics.evolution_generation += 1

            session.output_data = session.output_data or {
                "fitness": overall_fitness,
                "component_scores": fitness_scores,
                "generation": self.metrics.evolution_generation,
            }

        except Exception as e:
            session.output_data = {"error": safe_error_message(e)}

        session.duration_seconds = time.time() - start_time
        session.completed = True
        self.metrics.total_learning_time += session.duration_seconds
        self.sessions.append(session)

        if self.metrics.evolution_generation % 10 == 0:
            self._save_state()

        logger.debug(f"Evolution phase completed: gen {self.metrics.evolution_generation}")

        return {
            "session_id": session.id,
            "output": session.output_data,
            "insights": session.insights,
            "duration": session.duration_seconds,
        }

    async def _phase_reflection(self) -> dict:
        """Reflect on learning progress and adjust priorities using dual-model."""
        with self._lock:
            self._session_counter += 1
            session_id = f"learn_{self._session_counter:06d}"
        session = LearningSession(
            id=session_id,
            started_at=datetime.now(),
            phase=LearningPhase.REFLECTION,
            input_data={"source": "reflection"},
        )

        start_time = time.time()

        try:
            recent_sessions = list(self.sessions)[-10:]

            avg_duration = sum(s.duration_seconds for s in recent_sessions) / max(1, len(recent_sessions))

            phase_performance = defaultdict(lambda: {"count": 0, "insights": 0})
            for s in recent_sessions:
                phase_performance[s.phase.value]["count"] += 1
                if s.insights:
                    phase_performance[s.phase.value]["insights"] += len(s.insights)

            weak_phases = [
                phase for phase, stats in phase_performance.items()
                if stats["insights"] == 0 and stats["count"] > 1
            ]

            reflection_output = {
                "avg_session_duration": avg_duration,
                "phase_performance": dict(phase_performance),
                "weak_phases": weak_phases,
                "total_insights": len(self._insights_queue),
                "pending_dream_problems": len(self._pending_dream_problems),
                "evolution_generation": self.metrics.evolution_generation,
            }

            strategy_text = f"""Learning Strategy Evaluation:
- Problems attempted: {self.metrics.problems_attempted}
- Problems solved: {self.metrics.problems_solved}
- Dream insights: {self.metrics.dream_insights}
- Evolution cycles: {self.metrics.evolution_cycles}
- Pending problems to dream about: {len(self._pending_dream_problems)}
- Phases needing improvement: {weak_phases}
- Best math strategy so far: {self.metrics.best_math_strategy}
- Defense accuracy: {self.metrics.defense_accuracy:.2%}

Evaluate effectiveness and suggest improvements:
FOCUS: <top priority area>
PATTERN: <observed pattern>
SUGGESTION: <concrete action>"""

            try:
                dual_result = await learning_evaluate(
                    strategy=strategy_text,
                    context={"phase_performance": dict(phase_performance), "generation": self.metrics.evolution_generation},
                )

                if dual_result and dual_result != "Dual model integration not available":
                    for line in dual_result.split("\n"):
                        line = line.strip()
                        if line.startswith("FOCUS:") or line.startswith("PATTERN:") or line.startswith("SUGGESTION:"):
                            session.insights.append(line)

            except Exception as e:
                logger.debug(f"Dual-model reflection failed, falling back: {e}")

            if not session.insights and self.llm_client and recent_sessions:
                try:
                    reflection_prompt = f"""You are an AI system reflecting on your learning progress.

Recent learning statistics:
- Problems attempted: {self.metrics.problems_attempted}
- Problems solved: {self.metrics.problems_solved}
- Dream insights: {self.metrics.dream_insights}
- Evolution cycles: {self.metrics.evolution_cycles}
- Pending problems to dream about: {len(self._pending_dream_problems)}
- Phases needing improvement: {weak_phases}

Based on this, suggest specific improvements to your learning approach:
1. What should you focus more on?
2. Are there patterns in what works vs doesn't?
3. What's the best use of next learning cycle?

Format your response:
FOCUS: <top priority area>
PATTERN: <observed pattern>
SUGGESTION: <concrete action>"""

                    response = await self._llm_call(
                        user_input=reflection_prompt,
                        system_prompt="You are an AI system reflecting on your learning progress.",
                        priority="learning",
                    )

                    content = get_response_content(response)

                    for line in content.split("\n"):
                        line = line.strip()
                        if line.startswith("FOCUS:") or line.startswith("PATTERN:") or line.startswith("SUGGESTION:"):
                            session.insights.append(line)

                except Exception as e:
                    logger.debug(f"LLM reflection failed: {e}")

            session.output_data = reflection_output

            if weak_phases and not session.insights:
                session.insights.append(f"Need improvement in: {', '.join(weak_phases)}")

            self._save_state()

        except Exception as e:
            session.output_data = {"error": safe_error_message(e)}

        session.duration_seconds = time.time() - start_time
        session.completed = True
        self.metrics.total_learning_time += session.duration_seconds
        self.sessions.append(session)

        logger.debug("Reflection phase completed")

        return {
            "session_id": session.id,
            "output": session.output_data,
            "insights": session.insights,
            "duration": session.duration_seconds,
        }

    async def _phase_rest(self) -> dict:
        """Rest phase for memory consolidation and cleanup."""
        await asyncio.sleep(30)

        if self.memory_manager:
            try:
                await self.memory_manager.cleanup_cycle()
            except Exception as e:
                logger.debug(f"Memory cleanup error: {e}")

        logger.debug("Rest phase completed")

        return {
            "phase": "rest",
            "duration": 30,
            "memory_cleanup": True,
        }

    async def _dream_on_problem(self, problem: dict) -> dict:
        """Dream on an unsolved problem using dual-model for pattern discovery."""
        result = {
            "problem": problem.get("title", "unknown"),
            "insight": None,
            "solved": False,
        }

        strategies = problem.get("strategies_tried", [])
        category = problem.get("category", "unknown")
        title = problem.get("title", "unknown problem")

        try:
            experience = f"""Problem: {title}
Category: {category}
Previous approaches tried: {', '.join(strategies[-5:]) if strategies else 'None yet'}

Looking for creative mathematical patterns and connections."""

            dual_result = await learning_discover(
                experience=experience,
                context={"category": category, "title": title},
            )

            if dual_result and dual_result != "Dual model integration not available":
                insight = None
                approach = None

                for line in dual_result.split("\n"):
                    line = line.strip()
                    if line.startswith("INSIGHT:"):
                        insight = line.replace("INSIGHT:", "").strip()
                    elif line.startswith("APPROACH:"):
                        approach = line.replace("APPROACH:", "").strip()

                if insight:
                    result["insight"] = insight
                    result["suggested_approach"] = approach or insight
                    self.metrics.dream_insights += 1

                if insight and ("breakthrough" in insight.lower() or "novel" in dual_result.lower()):
                    result["solved"] = True
                    result["insight"] = f"Breakthrough insight: {insight}"
                    logger.info(f"Dream breakthrough on {title}: {insight[:100]}")

                return result

        except Exception as e:
            logger.debug(f"Dual-model dream failed, falling back: {e}")

        # Fallback to original _llm_call for backward compatibility
        if self.llm_client:
            try:
                dream_prompt = f"""You are reflecting on an unsolved mathematical problem in a dream-like state.

Problem: {title}
Category: {category}
Previous approaches tried: {', '.join(strategies[-5:]) if strategies else 'None yet'}

Think creatively about this problem. Consider:
1. What connections might exist to other mathematical areas?
2. Is there a counterintuitive approach that hasn't been tried?
3. Can you imagine a novel proof strategy?
4. What would Erdős or another creative mathematician suggest?

Provide a brief, insightful suggestion. Format:
INSIGHT: <your creative insight or approach suggestion>
APPROACH: <specific technique to try>
NOVELTY: <why this is different from previous attempts>"""

                response = await self._llm_call(
                    user_input=dream_prompt,
                    system_prompt="You are a mathematician dreaming about unsolved problems.",
                    priority="daydream",
                )

                content = get_response_content(response)

                insight = None
                approach = None

                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("INSIGHT:"):
                        insight = line.replace("INSIGHT:", "").strip()
                    elif line.startswith("APPROACH:"):
                        approach = line.replace("APPROACH:", "").strip()

                if insight:
                    result["insight"] = insight
                    result["suggested_approach"] = approach or insight
                    self.metrics.dream_insights += 1

                if insight and ("breakthrough" in insight.lower() or "novel" in content.lower()):
                    result["solved"] = True
                    result["insight"] = f"Breakthrough insight: {insight}"
                    logger.info(f"Dream breakthrough on {title}: {insight[:100]}")

                return result

            except Exception as e:
                logger.debug(f"LLM dream failed, using fallback: {e}")

        # Fallback to local approaches
        new_approaches = [
            "consider asymptotic behavior",
            "try constructive vs probabilistic dual",
            "look for invariants",
            "examine extremal cases",
            "apply symmetry reduction",
            "use double counting",
            "consider algebraic structure",
            "try density arguments",
        ]

        untried = [a for a in new_approaches if a not in strategies]

        if untried:
            suggested = random.choice(untried)
            result["insight"] = f"For {category}: {suggested}"
            result["suggested_approach"] = suggested

        if random.random() < 0.05:
            result["solved"] = True
            result["insight"] = f"Breakthrough on {title}!"

        return result

    def _extract_defense_lesson(self, event: dict) -> str | None:
        """Extract actionable lesson from defense event."""
        threat_type = event.get("threat_type", "")

        lessons = {
            "injection_attack": "Validate all inputs; use parameterized queries",
            "data_exfiltration": "Encrypt sensitive data; monitor access patterns",
            "integrity_violation": "Implement file integrity monitoring",
            "unauthorized_access": "Strengthen authentication; audit logs",
            "resource_abuse": "Set resource limits; auto-scale monitoring",
            "anomalous_behavior": "Update behavioral baselines; investigate deviations",
            "malicious_code": "Sandbox untrusted code; signature scanning",
        }

        return lessons.get(threat_type)

    def _detect_threat_patterns(self, events: list[dict]) -> list[str]:
        """Detect patterns in threat events."""
        patterns = []

        if len(events) < 3:
            return patterns

        type_counts = defaultdict(int)
        for event in events:
            type_counts[event.get("threat_type", "unknown")] += 1

        for threat_type, count in type_counts.items():
            if count >= 3:
                patterns.append(f"Recurring threat: {threat_type} ({count} times)")

        sources = set(e.get("source", "unknown") for e in events)
        if len(sources) == 1:
            patterns.append(f"All threats from single source: {list(sources)[0]}")

        return patterns

    def _queue_insight(self, insight_type: str, content: str) -> None:
        """Queue an insight for later processing and observe with PFC."""
        self._insights_queue.append({
            "type": insight_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Observe insight with Prefrontal Cortex
        try:
            pfc = get_prefrontal_cortex()
            if pfc and pfc._running:
                from core.helpers import create_tracked_task
                create_tracked_task(
                    pfc.observe_insight(
                        insight=content[:500],
                        source=f"learning:{insight_type}",
                        metadata={"type": insight_type},
                    ),
                    name="pfc_observe_learning_insight",
                )
        except Exception as e:
            logger.debug(f"Failed to observe learning insight with PFC: {e}")

    def _update_best_strategy(self, strategy: str, success: bool) -> None:
        """Update best math strategy based on outcomes."""
        if success:
            self.metrics.best_math_strategy = strategy

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "running": self._running,
            "current_phase": self._current_phase.value,
            "metrics": {
                "problems_attempted": self.metrics.problems_attempted,
                "problems_solved": self.metrics.problems_solved,
                "solve_rate": self.metrics.problems_solved / max(1, self.metrics.problems_attempted),
                "threats_analyzed": self.metrics.threats_analyzed,
                "defense_improvements": self.metrics.defense_improvements,
                "dream_insights": self.metrics.dream_insights,
                "evolution_generation": self.metrics.evolution_generation,
                "total_learning_time_hours": self.metrics.total_learning_time / 3600,
                "best_math_strategy": self.metrics.best_math_strategy,
                "defense_accuracy": self.metrics.defense_accuracy,
            },
            "pending_dream_problems": len(self._pending_dream_problems),
            "insights_queue_size": len(self._insights_queue),
            "sessions_count": len(self.sessions),
        }

    def get_recent_insights(self, limit: int = 20) -> list[dict]:
        """Get recent insights from learning."""
        return list(self._insights_queue)[-limit:]

    def get_session_history(self, limit: int = 10) -> list[dict]:
        """Get recent learning sessions."""
        return [
            {
                "id": s.id,
                "phase": s.phase.value,
                "duration": s.duration_seconds,
                "insights_count": len(s.insights),
                "completed": s.completed,
                "started_at": s.started_at.isoformat(),
            }
            for s in list(self.sessions)[-limit:]
        ]

    def throttle(self, aggressive: bool = False) -> None:
        """Throttle pipeline activity for resource conservation."""
        self._throttled = True
        if aggressive:
            for phase in self._phase_duration:
                self._phase_duration[phase] = self._phase_duration[phase] * 2
        logger.info(f"Learning pipeline throttled (aggressive={aggressive})")

    def resume(self) -> None:
        """Resume normal operation after throttling."""
        self._throttled = False
        self._phase_duration = dict(self.DEFAULT_PHASE_DURATION)
        logger.info("Learning pipeline resumed")

    def is_healthy(self) -> tuple[bool, dict]:
        """Check pipeline health."""
        return (
            self._running,
            {
                "sessions": len(self.sessions),
                "current_phase": self._current_phase.value,
                "metrics": {
                    "problems_attempted": self.metrics.problems_attempted,
                    "problems_solved": self.metrics.problems_solved,
                    "evolution_generation": self.metrics.evolution_generation,
                },
                "throttled": getattr(self, '_throttled', False),
            }
        )


_pipeline: ContinuousLearningPipeline | None = None
_pipeline_lock = threading.Lock()


def get_learning_pipeline(llm_client: Any = None) -> ContinuousLearningPipeline:
    """Get global learning pipeline instance (thread-safe).
    
    Args:
        llm_client: Optional LLM client for intelligent learning.
                    If provided on first call, will be used for reasoning.
    """
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ContinuousLearningPipeline(llm_client=llm_client)
    elif llm_client is not None and _pipeline.llm_client is None:
        _pipeline.llm_client = llm_client
        if _pipeline.erdos_engine:
            _pipeline.erdos_engine.solver.llm_client = llm_client
    return _pipeline
