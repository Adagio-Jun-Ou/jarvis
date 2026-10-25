"""Unified Execution Router for Jarvis.

Selects the optimal execution strategy based on task characteristics:
1. DIRECT - Simple queries: single LLM call, no tools
2. AGENTIC_LOOP - Medium complexity: continuous tool-calling loop until complete
3. HARNESS - Multi-step structured: Plan-Execute-Verify with retry/backoff
4. ORCHESTRATOR - Cross-model: dual-model decomposition and synthesis
5. COUNCIL - Low-confidence: multi-model council deliberation

Decision criteria:
- Task complexity (heuristics + LLM classification)
- Step count estimate (1 step -> DIRECT, 2-5 -> AGENTIC_LOOP, 6+ -> HARNESS)
- Tool dependency structure (independent -> parallel, sequential -> harness)
- Model routing needs (analysis -> backend, creative -> primary, complex -> orchestrator)
- Confidence level (low -> council fallback)

Integration with existing systems:
- AgentLoop: used for agentic loop strategy (already exists)
- AgenticHarness: used for harness strategy (already exists)
- IntelligentOrchestrator: used for orchestrator strategy (already exists)
- Both AgentLoop and AgenticHarness can be used together (harness delegates to agent loop)
"""
import asyncio
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from core.helpers import safe_error_message

if TYPE_CHECKING:
    from .agent import JarvisAgent

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    DIRECT = "direct"
    AGENTIC_LOOP = "agentic_loop"
    HARNESS = "harness"
    ORCHESTRATOR = "orchestrator"
    COUNCIL = "council"


@dataclass
class RoutingDecision:
    strategy: ExecutionStrategy
    confidence: float
    reasoning: str
    estimated_steps: int
    requires_parallel: bool = False
    requires_verification: bool = False
    task_type: str = "general"
    fallback_strategy: Optional[ExecutionStrategy] = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "estimated_steps": self.estimated_steps,
            "requires_parallel": self.requires_parallel,
            "requires_verification": self.requires_verification,
            "task_type": self.task_type,
            "fallback_strategy": self.fallback_strategy.value if self.fallback_strategy else None,
        }


@dataclass
class ExecutionResult:
    strategy_used: ExecutionStrategy
    success: bool
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    tool_calls: int = 0
    steps_completed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_used": self.strategy_used.value,
            "success": self.success,
            "output": self.output[:500] if self.output else "",
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "tool_calls": self.tool_calls,
            "steps_completed": self.steps_completed,
            "metadata": self.metadata,
        }


class ExecutionRouter:
    _instance: Optional["ExecutionRouter"] = None

    def __init__(self, agent: Optional["JarvisAgent"] = None):
        self.agent = agent
        self._routing_history: list[dict] = []
        self._max_history = 200
        self._strategy_stats: dict[str, dict] = {
            s.value: {"calls": 0, "successes": 0, "failures": 0, "total_ms": 0.0}
            for s in ExecutionStrategy
        }

    @classmethod
    def get_instance(cls, agent: Optional["JarvisAgent"] = None) -> "ExecutionRouter":
        if cls._instance is None:
            cls._instance = ExecutionRouter(agent)
        elif agent is not None:
            cls._instance.agent = agent
        return cls._instance

    MULTI_STEP_PATTERNS = [
        r"\b(create|build|implement|develop)\s+.*\b(and|then|after)\b",
        r"\b(fix|debug|resolve|troubleshoot)\s+.*\b(and|then|verify)\b",
        r"\b(analyze|examine|review)\s+.*\b(and|then|report)\b",
        r"\b(refactor|restructure|reorganize)\b.*\b(tests?|verify)\b",
        r"\b(setup|configure|deploy)\s+.*\b(and|then)\b",
        r"\bfirst\b.*\bthen\b.*\bfinally\b",
        r"\bstep\s+by\s+step\b",
        r"\bone\s+by\s+one\b",
        r"\bplease\s+(also|then|additionally)\b",
    ]

    VERIFICATION_PATTERNS = [
        r"\bverify\b", r"\bconfirm\b", r"\bcheck\b.*\bworks?\b",
        r"\bmake\s+sure\b", r"\bensure\b", r"\bvalidate\b",
        r"\btest\b.*\bpass\b", r"\brun\s+tests?\b",
    ]

    PARALLEL_PATTERNS = [
        r"\bsimultaneously\b", r"\bin\s+parallel\b", r"\bconcurrently\b",
        r"\ball\s+at\s+once\b", r"\bat\s+the\s+same\s+time\b",
        r"\bcompare\s+.*\b(and|vs)\b",
    ]

    ANALYSIS_PATTERNS = [
        r"\bcompare\s+.*\b(analyze|evaluate|assess)\b",
        r"\bpros\s+and\s+cons\b", r"\btrade[- ]?offs?\b",
        r"\bmulti[- ]?perspective\b", r"\bfrom\s+different\s+angles\b",
        r"\bsynthesize\b.*\b(information|data|findings)\b",
    ]

    def route(self, user_input: str, context: dict | None = None) -> RoutingDecision:
        scores: dict[ExecutionStrategy, float] = {s: 0.0 for s in ExecutionStrategy}
        input_lower = user_input.lower()
        word_count = len(user_input.split())
        sentence_count = max(1, len(re.split(r'[.!?]+', user_input)))
        question_count = user_input.count("?")

        if word_count < 8 and question_count <= 1:
            scores[ExecutionStrategy.DIRECT] += 2.0

        simple_starts = ["what is", "tell me about", "explain", "how does", "why is", "who is", "define", "describe"]
        if any(input_lower.startswith(s) for s in simple_starts) and word_count < 30:
            scores[ExecutionStrategy.DIRECT] += 3.0

        multi_step_matches = sum(1 for p in self.MULTI_STEP_PATTERNS if re.search(p, input_lower))
        if multi_step_matches > 0:
            scores[ExecutionStrategy.HARNESS] += multi_step_matches * 2.0

        if word_count > 100:
            scores[ExecutionStrategy.HARNESS] += 1.0
        if sentence_count >= 4:
            scores[ExecutionStrategy.HARNESS] += 1.5

        verification_matches = sum(1 for p in self.VERIFICATION_PATTERNS if re.search(p, input_lower))
        if verification_matches > 0:
            scores[ExecutionStrategy.HARNESS] += verification_matches

        parallel_matches = sum(1 for p in self.PARALLEL_PATTERNS if re.search(p, input_lower))
        if parallel_matches > 0:
            scores[ExecutionStrategy.HARNESS] += parallel_matches * 1.5

        tool_keywords = ["search", "fetch", "read", "write", "run", "execute", "list", "count", "find", "check"]
        tool_count = sum(1 for kw in tool_keywords if kw in input_lower)
        if tool_count >= 2:
            scores[ExecutionStrategy.AGENTIC_LOOP] += 1.5

        if word_count > 50 and word_count <= 150 and tool_count >= 1:
            scores[ExecutionStrategy.AGENTIC_LOOP] += 1.0

        action_words = ["create", "build", "implement", "fix", "debug", "write", "add", "remove", "update", "deploy"]
        action_count = sum(1 for aw in action_words if aw in input_lower)
        if action_count == 1 and word_count < 80:
            scores[ExecutionStrategy.AGENTIC_LOOP] += 1.5
        elif action_count >= 2:
            scores[ExecutionStrategy.HARNESS] += 2.0

        analysis_matches = sum(1 for p in self.ANALYSIS_PATTERNS if re.search(p, input_lower))
        if analysis_matches > 0:
            scores[ExecutionStrategy.ORCHESTRATOR] += analysis_matches * 2.0

        if sentence_count >= 3 and question_count >= 2:
            scores[ExecutionStrategy.ORCHESTRATOR] += 1.0

        compound_indicators = ["and then", "after that", "also need", "while you", "in addition to"]
        if any(ind in input_lower for ind in compound_indicators):
            scores[ExecutionStrategy.HARNESS] += 1.5

        if scores[ExecutionStrategy.HARNESS] > 0:
            scores[ExecutionStrategy.HARNESS] = max(
                scores[ExecutionStrategy.HARNESS],
                scores[ExecutionStrategy.AGENTIC_LOOP] * 0.5,
            )

        if scores[ExecutionStrategy.AGENTIC_LOOP] > 0:
            scores[ExecutionStrategy.AGENTIC_LOOP] = max(
                scores[ExecutionStrategy.AGENTIC_LOOP],
                scores[ExecutionStrategy.HARNESS] * 0.3,
            )

        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]
        total_score = sum(scores.values()) or 1.0
        confidence = min(1.0, best_score / max(total_score, 1.0))

        if confidence < 0.3 or best_score < 1.0:
            best_strategy = ExecutionStrategy.AGENTIC_LOOP
            confidence = max(confidence, 0.3)

        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fallback = sorted_strategies[1][0] if len(sorted_strategies) > 1 else None

        estimated_steps = 1
        if best_strategy == ExecutionStrategy.AGENTIC_LOOP:
            estimated_steps = min(5, max(2, action_count + tool_count))
        elif best_strategy == ExecutionStrategy.HARNESS:
            estimated_steps = min(10, max(3, multi_step_matches * 2 + action_count + verification_matches))
        elif best_strategy == ExecutionStrategy.ORCHESTRATOR:
            estimated_steps = min(8, max(3, analysis_matches * 2 + sentence_count // 2))

        requires_parallel = parallel_matches > 0 or estimated_steps >= 5
        requires_verification = verification_matches > 0

        task_type = "general"
        if any(w in input_lower for w in ["code", "program", "script", "function", "class", "module"]):
            task_type = "code"
        elif any(w in input_lower for w in ["analyze", "examine", "review", "compare", "evaluate"]):
            task_type = "analysis"
        elif any(w in input_lower for w in ["create", "build", "implement", "write", "design"]):
            task_type = "creation"
        elif any(w in input_lower for w in ["fix", "debug", "resolve", "troubleshoot", "repair"]):
            task_type = "debugging"

        decision = RoutingDecision(
            strategy=best_strategy,
            confidence=confidence,
            reasoning=self._generate_reasoning(best_strategy, scores, user_input),
            estimated_steps=estimated_steps,
            requires_parallel=requires_parallel,
            requires_verification=requires_verification,
            task_type=task_type,
            fallback_strategy=fallback,
        )

        self._record_routing(decision)
        return decision

    def _generate_reasoning(self, strategy: ExecutionStrategy, scores: dict[ExecutionStrategy, float], user_input: str) -> str:
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        parts = [f"{s.value}={sc:.1f}" for s, sc in top_scores if sc > 0]
        return f"Selected {strategy.value} (scores: {', '.join(parts)})"

    def _record_routing(self, decision: RoutingDecision) -> None:
        self._routing_history.append({
            "timestamp": time.time(),
            **decision.to_dict(),
        })
        if len(self._routing_history) > self._max_history:
            self._routing_history = self._routing_history[-self._max_history:]

    async def execute(
        self,
        user_input: str,
        context: dict | None = None,
        force_strategy: ExecutionStrategy | None = None,
    ) -> ExecutionResult:
        # perf_counter, not time.time(): the wall clock has ~15.6ms granularity
        # on Windows, which rounds sub-millisecond dispatches down to 0.
        start_time = time.perf_counter()
        decision = self.route(user_input, context) if force_strategy is None else None
        strategy = force_strategy or (decision.strategy if decision else ExecutionStrategy.DIRECT)

        if decision:
            logger.info(f"[ExecutionRouter] Strategy={strategy.value} confidence={decision.confidence:.2f} steps≈{decision.estimated_steps} type={decision.task_type}")

        try:
            result = await self._dispatch(strategy, user_input, context or {}, decision)
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(strategy, result.success, result.duration_ms)
            return result
        except Exception as e:
            error_msg = safe_error_message(e)
            logger.warning(f"[ExecutionRouter] Strategy {strategy.value} failed: {error_msg}")

            fallback = decision.fallback_strategy if decision else None
            if fallback and fallback != strategy:
                logger.info(f"[ExecutionRouter] Falling back to {fallback.value}")
                try:
                    result = await self._dispatch(fallback, user_input, context or {}, decision)
                    result.duration_ms = (time.perf_counter() - start_time) * 1000
                    self._update_stats(fallback, result.success, result.duration_ms)
                    result.metadata["fallback_from"] = strategy.value
                    return result
                except Exception as fallback_err:
                    logger.error(f"[ExecutionRouter] Fallback {fallback.value} also failed: {safe_error_message(fallback_err)}")

            return ExecutionResult(
                strategy_used=strategy,
                success=False,
                error=error_msg,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def _dispatch(
        self,
        strategy: ExecutionStrategy,
        user_input: str,
        context: dict,
        decision: RoutingDecision | None,
    ) -> ExecutionResult:
        if strategy == ExecutionStrategy.DIRECT:
            return await self._execute_direct(user_input, context)
        elif strategy == ExecutionStrategy.AGENTIC_LOOP:
            return await self._execute_agentic_loop(user_input, context)
        elif strategy == ExecutionStrategy.HARNESS:
            return await self._execute_harness(user_input, context, decision)
        elif strategy == ExecutionStrategy.ORCHESTRATOR:
            return await self._execute_orchestrator(user_input, context)
        elif strategy == ExecutionStrategy.COUNCIL:
            return await self._execute_council(user_input, context)
        return ExecutionResult(strategy_used=strategy, success=False, error="Unknown strategy")

    async def _execute_direct(self, user_input: str, context: dict) -> ExecutionResult:
        agent = self.agent
        if agent is None:
            return ExecutionResult(strategy_used=ExecutionStrategy.DIRECT, success=False, error="No agent available")

        system_prompt = agent.get_system_prompt()
        enhanced = context.get("unified_context", "")
        if enhanced:
            system_prompt = f"{system_prompt}\n\n## Relevant Context\n\n{enhanced}"

        try:
            from core.llm import subsystem_context
            async with subsystem_context("ExecRouter"):
                response = await agent.llm.chat(
                    user_input=user_input,
                    system_prompt=system_prompt,
                )
        except Exception as e:
            return ExecutionResult(strategy_used=ExecutionStrategy.DIRECT, success=False, error=safe_error_message(e))

        output = response if isinstance(response, str) else str(response)
        return ExecutionResult(
            strategy_used=ExecutionStrategy.DIRECT,
            success=True,
            output=output,
            steps_completed=1,
        )

    async def _execute_agentic_loop(self, user_input: str, context: dict) -> ExecutionResult:
        agent = self.agent
        if agent is None:
            return ExecutionResult(strategy_used=ExecutionStrategy.AGENTIC_LOOP, success=False, error="No agent available")

        try:
            from core.agent_loop import AgentLoop, FinishReason
            llm = agent.backend_llm if hasattr(agent, 'backend_llm') and agent.backend_llm else agent.llm
            loop = AgentLoop(
                llm_client=llm,
                tool_registry=agent.tools,
                permission_manager=agent.permissions,
                max_steps=10,
                auto_approve_tools=getattr(agent.config.agent, 'auto_approve_tools', False),
            )

            messages = list(agent.memory.get_messages_for_llm())
            messages.append({"role": "user", "content": user_input})
            system_prompt = agent.get_system_prompt()
            enhanced = context.get("unified_context", "")
            if enhanced:
                system_prompt = f"{system_prompt}\n\n## Relevant Context\n\n{enhanced}"

            result = await loop.execute(
                messages=messages,
                system_prompt=system_prompt,
                tools=list(agent._tool_schemas) if agent._tool_schemas else None,
            )

            text_parts = []
            for step in result.steps:
                if step.assistant_message.text:
                    text_parts.append(step.assistant_message.text)
            output = "\n".join(text_parts) if text_parts else ""

            return ExecutionResult(
                strategy_used=ExecutionStrategy.AGENTIC_LOOP,
                success=result.finish_reason != FinishReason.ERROR,
                output=output,
                tool_calls=result.total_tool_calls,
                steps_completed=len(result.steps),
                metadata={"finish_reason": result.finish_reason.value, "duration_ms": result.duration_ms},
            )
        except Exception as e:
            return ExecutionResult(strategy_used=ExecutionStrategy.AGENTIC_LOOP, success=False, error=safe_error_message(e))

    async def _execute_harness(self, user_input: str, context: dict, decision: RoutingDecision | None) -> ExecutionResult:
        agent = self.agent
        if agent is None:
            return ExecutionResult(strategy_used=ExecutionStrategy.HARNESS, success=False, error="No agent available")

        try:
            harness = agent.get_agentic_harness()
            if harness is None:
                return ExecutionResult(strategy_used=ExecutionStrategy.HARNESS, success=False, error="AgenticHarness not available")

            from core.agentic_harness import llm_generate_plan

            available_tools = list(agent.tools.keys())
            llm = agent.backend_llm if hasattr(agent, 'backend_llm') and agent.backend_llm else None

            heuristic_steps = await llm_generate_plan(user_input, llm, available_tools)
            max_iter = (decision.estimated_steps + 2) if decision else 10
            plan = harness.create_plan(user_input, heuristic_steps, max_iterations=max_iter)
            harness.save_checkpoint(plan)

            plan_result = await harness.execute_plan(plan, tools=agent.tools)
            harness.save_checkpoint(plan)

            success = plan.is_complete
            output_parts = []
            for step in plan.steps:
                if step.status.value in ("success", "verified") and step.result is not None:
                    if isinstance(step.result, str):
                        output_parts.append(step.result)
                    elif isinstance(step.result, dict):
                        if "output" in step.result:
                            output_parts.append(str(step.result["output"]))
                        else:
                            output_parts.append(str(step.result))
            output = "\n".join(output_parts) if output_parts else plan_result.get("objective", "")

            return ExecutionResult(
                strategy_used=ExecutionStrategy.HARNESS,
                success=success,
                output=output,
                tool_calls=harness._total_tools_called,
                steps_completed=plan.completed_steps,
                metadata=plan_result,
            )
        except Exception as e:
            return ExecutionResult(strategy_used=ExecutionStrategy.HARNESS, success=False, error=safe_error_message(e))

    async def _execute_orchestrator(self, user_input: str, context: dict) -> ExecutionResult:
        agent = self.agent
        if agent is None or agent._intelligent_orchestrator is None:
            return ExecutionResult(strategy_used=ExecutionStrategy.ORCHESTRATOR, success=False, error="Orchestrator not available")

        try:
            orch_context = {
                "conversation": agent.memory.get_recent_messages(10),
                "system_prompt": agent.get_system_prompt(),
            }
            if context.get("unified_context"):
                orch_context["unified_context"] = context["unified_context"]

            result = await agent._intelligent_orchestrator.plan_and_execute(
                request=user_input,
                context=orch_context,
            )

            return ExecutionResult(
                strategy_used=ExecutionStrategy.ORCHESTRATOR,
                success=True,
                output=result if isinstance(result, str) else str(result),
                steps_completed=1,
            )
        except Exception as e:
            return ExecutionResult(strategy_used=ExecutionStrategy.ORCHESTRATOR, success=False, error=safe_error_message(e))

    async def _execute_council(self, user_input: str, context: dict) -> ExecutionResult:
        agent = self.agent
        if agent is None or not hasattr(agent, '_council') or agent._council is None:
            return ExecutionResult(strategy_used=ExecutionStrategy.COUNCIL, success=False, error="Council not available")

        try:
            council_session = await agent._council.deliberate(
                topic=user_input,
                primary_llm=agent.llm,
                backend_llm=agent.backend_llm if hasattr(agent, 'backend_llm') else None,
            )

            output = council_session.final_response or "No consensus reached"
            return ExecutionResult(
                strategy_used=ExecutionStrategy.COUNCIL,
                success=council_session.consensus,
                output=output,
                metadata={"confidence": council_session.confidence},
            )
        except Exception as e:
            return ExecutionResult(strategy_used=ExecutionStrategy.COUNCIL, success=False, error=safe_error_message(e))

    def _update_stats(self, strategy: ExecutionStrategy, success: bool, duration_ms: float) -> None:
        stats = self._strategy_stats[strategy.value]
        stats["calls"] += 1
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_ms"] += duration_ms

    def get_routing_stats(self) -> dict:
        return {
            "total_routings": len(self._routing_history),
            "strategy_stats": dict(self._strategy_stats),
            "recent_decisions": self._routing_history[-10:],
        }

    def get_strategy_for_task(self, user_input: str) -> ExecutionStrategy:
        decision = self.route(user_input)
        return decision.strategy

    async def route_with_llm(
        self,
        user_input: str,
        context: dict | None = None,
    ) -> RoutingDecision:
        """Route using LLM classification when heuristic confidence is low.

        Falls back to heuristic routing if LLM unavailable.
        """
        heuristic_decision = self.route(user_input, context)

        if heuristic_decision.confidence >= 0.6:
            return heuristic_decision

        agent = self.agent
        if agent is None:
            return heuristic_decision

        try:
            llm = getattr(agent, 'backend_llm', None) or agent.llm
            if not llm:
                return heuristic_decision

            prompt = f"""Classify this query into one execution strategy:

Query: {user_input}

Strategies:
- direct: Simple question, single answer (e.g. "what is X")
- agentic_loop: Needs tool calls, 2-4 steps (e.g. "search for X and summarize")
- harness: Multi-step with dependencies, 5+ steps, needs verification (e.g. "build X then test it")
- orchestrator: Cross-model analysis, compare perspectives (e.g. "analyze pros and cons of X vs Y")
- council: High-stakes, low-confidence, needs consensus

Return JSON: {{"strategy": "direct|agentic_loop|harness|orchestrator|council", "confidence": 0.0-1.0, "reasoning": "brief explanation", "estimated_steps": number}}"""

            if hasattr(llm, 'chat'):
                response = await llm.chat(
                    user_input=prompt,
                )
            else:
                return heuristic_decision

            text = response if isinstance(response, str) else str(response)
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if not json_match:
                return heuristic_decision

            llm_result = json.loads(json_match.group())
            strategy_str = llm_result.get("strategy", "")
            strategy_map = {s.value: s for s in ExecutionStrategy}
            llm_strategy = strategy_map.get(strategy_str)

            if llm_strategy is None:
                return heuristic_decision

            llm_confidence = min(1.0, max(0.0, float(llm_result.get("confidence", 0.5))))
            blended_confidence = (
                heuristic_decision.confidence * 0.4 + llm_confidence * 0.6
            )

            if llm_confidence > heuristic_decision.confidence:
                return RoutingDecision(
                    strategy=llm_strategy,
                    confidence=blended_confidence,
                    reasoning=f"LLM: {llm_result.get('reasoning', '')} | Heuristic: {heuristic_decision.reasoning}",
                    estimated_steps=int(llm_result.get("estimated_steps", heuristic_decision.estimated_steps)),
                    requires_parallel=heuristic_decision.requires_parallel,
                    requires_verification=heuristic_decision.requires_verification,
                    task_type=heuristic_decision.task_type,
                    fallback_strategy=heuristic_decision.strategy,
                )

            return heuristic_decision

        except Exception as e:
            logger.debug(f"LLM routing fallback to heuristic: {safe_error_message(e)}")
            return heuristic_decision


class CircuitBreaker:
    """Circuit breaker for execution strategy failure protection.

    Prevents repeated execution of failing strategies:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Strategy has failed too many times, requests rejected
    - HALF_OPEN: Testing if strategy has recovered
    """

    class State:
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max: int = 1,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._circuits: dict[str, dict] = {}

    def _get_circuit(self, strategy: str) -> dict:
        if strategy not in self._circuits:
            self._circuits[strategy] = {
                "state": self.State.CLOSED,
                "failures": 0,
                "successes": 0,
                # Monotonic stamp drives state transitions; the wall-clock
                # twin exists only for display/reporting.
                "last_failure_mono": 0.0,
                "last_failure": 0.0,
                "half_open_calls": 0,
            }
        return self._circuits[strategy]

    def can_execute(self, strategy: str) -> bool:
        circuit = self._get_circuit(strategy)
        state = circuit["state"]

        if state == self.State.CLOSED:
            return True

        if state == self.State.OPEN:
            # Monotonic clock: a wall-clock adjustment (NTP step, manual
            # change, suspend/resume skew) must not pin the breaker open or
            # trip it early.
            if time.monotonic() - circuit["last_failure_mono"] > self._recovery_timeout:
                circuit["state"] = self.State.HALF_OPEN
                circuit["half_open_calls"] = 0
                return True
            return False

        if state == self.State.HALF_OPEN:
            return circuit["half_open_calls"] < self._half_open_max

        return True

    def record_success(self, strategy: str) -> None:
        circuit = self._get_circuit(strategy)
        circuit["successes"] += 1
        circuit["failures"] = 0
        circuit["state"] = self.State.CLOSED

    def record_failure(self, strategy: str) -> None:
        circuit = self._get_circuit(strategy)
        circuit["failures"] += 1
        circuit["last_failure_mono"] = time.monotonic()
        circuit["last_failure"] = time.time()  # display twin

        if circuit["state"] == self.State.HALF_OPEN:
            circuit["state"] = self.State.OPEN
            circuit["half_open_calls"] = 0
        elif circuit["failures"] >= self._failure_threshold:
            circuit["state"] = self.State.OPEN

    def record_half_open_call(self, strategy: str) -> None:
        circuit = self._get_circuit(strategy)
        circuit["half_open_calls"] += 1

    def get_status(self) -> dict[str, dict]:
        return {
            strategy: {
                "state": circuit["state"],
                "failures": circuit["failures"],
                "successes": circuit["successes"],
                # Wall-clock for humans/dashboards; the breaker itself decides
                # on the monotonic stamp, which is deliberately not exposed.
                "last_failure": circuit["last_failure"],
                "open_for_s": max(
                    0.0,
                    round(
                        time.monotonic() - circuit["last_failure_mono"], 1
                    ),
                ) if circuit["last_failure_mono"] else 0.0,
            }
            for strategy, circuit in self._circuits.items()
        }


class ExecutionEventEmitter:
    """Event emitter for execution streaming via SSE.

    Emits events during execution for real-time monitoring:
    - routing_decision: When a routing decision is made
    - execution_started: When execution begins
    - step_completed: When a step/tool call completes
    - execution_completed: When execution finishes
    - execution_failed: When execution fails
    - fallback_triggered: When a fallback strategy is used
    """

    def __init__(self):
        self._listeners: list = []
        self._event_log: deque = deque(maxlen=500)

    def add_listener(self, callback) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback) -> None:
        self._listeners = [cb for cb in self._listeners if cb != callback]

    async def emit(self, event_type: str, data: dict | None = None) -> None:
        event = {
            "type": event_type,
            "data": data or {},
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        }
        self._event_log.append(event)
        for callback in self._listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.debug(f"Event emitter callback error: {safe_error_message(e)}")

    def get_events(self, limit: int = 50, event_type: str | None = None) -> list[dict]:
        events = list(self._event_log)
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]

    @property
    def event_count(self) -> int:
        return len(self._event_log)


_enhanced_router: Optional["EnhancedExecutionRouter"] = None
_enhanced_router_lock = threading.Lock()


class EnhancedExecutionRouter(ExecutionRouter):
    """Enhanced execution router with LLM routing, circuit breaker, and SSE events."""

    def __init__(self, agent: Optional["JarvisAgent"] = None):
        super().__init__(agent)
        self.circuit_breaker = CircuitBreaker()
        self.event_emitter = ExecutionEventEmitter()
        self._llm_routing_enabled = True
        self._circuit_breaker_enabled = True

    @classmethod
    def get_instance(cls, agent: Optional["JarvisAgent"] = None) -> "EnhancedExecutionRouter":
        global _enhanced_router
        if _enhanced_router is None:
            with _enhanced_router_lock:
                if _enhanced_router is None:
                    _enhanced_router = EnhancedExecutionRouter(agent)
        elif agent is not None:
            _enhanced_router.agent = agent
        return _enhanced_router

    async def execute(
        self,
        user_input: str,
        context: dict | None = None,
        force_strategy: ExecutionStrategy | None = None,
    ) -> ExecutionResult:
        start_time = time.perf_counter()

        if self._llm_routing_enabled and force_strategy is None:
            decision = await self.route_with_llm(user_input, context)
        else:
            decision = self.route(user_input, context) if force_strategy is None else None

        strategy = force_strategy or (decision.strategy if decision else ExecutionStrategy.DIRECT)

        await self.event_emitter.emit("routing_decision", {
            "query": user_input[:100],
            "strategy": strategy.value,
            "confidence": decision.confidence if decision else 1.0,
        })

        if self._circuit_breaker_enabled and not self.circuit_breaker.can_execute(strategy.value):
            await self.event_emitter.emit("fallback_triggered", {
                "from": strategy.value,
                "reason": "circuit_open",
            })
            fallback = decision.fallback_strategy if decision else None
            if fallback:
                strategy = fallback
            else:
                return ExecutionResult(
                    strategy_used=strategy,
                    success=False,
                    error=f"Circuit breaker open for {strategy.value}",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )

        if decision:
            logger.info(f"[EnhancedRouter] Strategy={strategy.value} confidence={decision.confidence:.2f}")

        await self.event_emitter.emit("execution_started", {
            "strategy": strategy.value,
            "estimated_steps": decision.estimated_steps if decision else 1,
        })

        try:
            result = await self._dispatch(strategy, user_input, context or {}, decision)
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(strategy, result.success, result.duration_ms)
            self.circuit_breaker.record_success(strategy.value)

            await self.event_emitter.emit("execution_completed", {
                "strategy": strategy.value,
                "success": result.success,
                "duration_ms": result.duration_ms,
                "steps": result.steps_completed,
            })
            return result
        except Exception as e:
            error_msg = safe_error_message(e)
            self.circuit_breaker.record_failure(strategy.value)
            await self.event_emitter.emit("execution_failed", {
                "strategy": strategy.value,
                "error": error_msg,
            })

            fallback = decision.fallback_strategy if decision else None
            if fallback and fallback != strategy and self.circuit_breaker.can_execute(fallback.value):
                await self.event_emitter.emit("fallback_triggered", {
                    "from": strategy.value,
                    "to": fallback.value,
                    "reason": "primary_failed",
                })
                try:
                    result = await self._dispatch(fallback, user_input, context or {}, decision)
                    result.duration_ms = (time.perf_counter() - start_time) * 1000
                    self._update_stats(fallback, result.success, result.duration_ms)
                    result.metadata["fallback_from"] = strategy.value
                    self.circuit_breaker.record_success(fallback.value)
                    return result
                except Exception as fallback_err:
                    self.circuit_breaker.record_failure(fallback.value)
                    logger.error(f"[EnhancedRouter] Fallback {fallback.value} failed: {safe_error_message(fallback_err)}")

            return ExecutionResult(
                strategy_used=strategy,
                success=False,
                error=error_msg,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

    def get_full_status(self) -> dict:
        return {
            **self.get_routing_stats(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "event_count": self.event_emitter.event_count,
            "recent_events": self.event_emitter.get_events(limit=10),
        }