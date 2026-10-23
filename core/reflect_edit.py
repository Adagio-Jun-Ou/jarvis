"""Reflect-Edit Loop & Council Deliberation for Jarvis.

Two major self-improvement mechanisms:

1. REFLECT-EDIT LOOP (from Letta):
   After each interaction, the LCM reflects on performance and updates
   shared memory. Over time, routing decisions improve through experience.

2. COUNCIL DELIBERATION (from Heddle):
   When confidence is low, convene a "council" of Primary + Backend models
   for structured debate. The LCM acts as facilitator and synthesizes consensus.

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                  REFLECT-EDIT LOOP                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   LLM Response → User Reaction → LCM Reflection                   │
│                                      │                             │
│                          ┌───────────┼───────────┐                │
│                          ▼           ▼           ▼                 │
│                    Update        Update       Update                │
│                  Memory Blocks  Routing      Quality              │
│                                   Table        Scores              │
│                          │           │           │                 │
│                          └───────────┼───────────┘                │
│                                      ▼                             │
│                              Better Next Response                  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                COUNCIL DELIBERATION                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Low Confidence Query                                             │
│       │                                                            │
│       ▼                                                            │
│   ┌──────────────┐                                                │
│   │ LCM asks    │  Both models respond independently              │
│   │ both models │                                                   │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │ DEBATE      │  Models see each other's responses              │
│   │ ROUNDS      │  and refine their positions                     │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │ CONVERGENCE  │  Check if models agree (similarity > 0.8)      │
│   │ DETECTION   │  or max rounds reached                         │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │ LCM          │  Synthesize consensus or best answer           │
│   │ SYNTHESIZES  │  with confidence score                        │
│   └──────────────┘                                                │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_reflect_edit: Optional["ReflectEditLoop"] = None
_reflect_edit_lock = threading.Lock()

_council: Optional["CouncilDeliberation"] = None
_council_lock = threading.Lock()


class ReflectionType(Enum):
    QUALITY = "quality"
    ROUTING = "routing"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    SAFETY = "safety"


class CouncilProtocol(Enum):
    DEBATE = "debate"
    DELPHI = "delphi"
    CONSENSUS = "consensus"


@dataclass
class Reflection:
    """A reflection on an interaction."""
    id: str
    interaction_id: str
    reflection_type: ReflectionType
    score: float
    insight: str
    action: str
    model_used: str
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReflectEditStats:
    total_reflections: int = 0
    quality_improvements: int = 0
    routing_corrections: int = 0
    memory_updates: int = 0
    avg_reflection_score: float = 0.0
    reflections_by_type: dict = field(default_factory=dict)


class ReflectEditLoop:
    """Self-improvement through reflection and memory editing.
    
    After each interaction, the LCM reflects on:
    1. Response quality (was the answer good?)
    2. Routing accuracy (was the right model used?)
    3. Completeness (did the response fully address the query?)
    4. Relevance (was the response on-topic?)
    
    Based on reflections, the LCM updates:
    1. Memory blocks (L2 warm layer)
    2. Routing table (which model handles what)
    3. Quality scores (track model performance)
    """
    
    def __init__(self, max_history: int = 200):
        self._stats = ReflectEditStats()
        self._lock = threading.RLock()
        self._reflections: OrderedDict[str, Reflection] = OrderedDict()
        self._max_reflections = max_history
        
        self._routing_table: OrderedDict[str, dict] = OrderedDict()
        self._max_routing_entries = 100
    
    async def reflect(
        self,
        query: str,
        response: str,
        model_used: str,
        user_reaction: str | None = None,
        tool_calls: list[dict] | None = None,
        latency_ms: float = 0.0,
    ) -> Reflection:
        """Reflect on a completed interaction and update memory.
        
        This is the core self-improvement loop:
        1. Assess response quality
        2. Check if routing was correct
        3. Update quality scores
        4. Edit memory blocks based on insights
        """
        interaction_id = f"int_{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        
        quality = self._assess_quality(query, response, user_reaction)
        routing = self._assess_routing(query, model_used, quality)
        completeness = self._assess_completeness(query, response, tool_calls)
        relevance = self._assess_relevance(query, response)
        
        overall_score = (quality + routing + completeness + relevance) / 4.0
        
        insights = []
        actions = []
        
        if quality < 0.6:
            insights.append(f"Low quality response (score: {quality:.2f})")
            actions.append("Consider using different model or adding more context")
        
        if routing < 0.6:
            insights.append(f"Poor routing decision (model: {model_used}, score: {routing:.2f})")
            better_model = "backend" if model_used == "primary" else "primary"
            actions.append(f"Similar queries should route to {better_model}")
            self._update_routing(query, better_model, routing)
        
        if completeness < 0.6:
            insights.append("Response incomplete")
            actions.append("Add agentic loop for multi-step completion")
        
        if relevance < 0.6:
            insights.append("Response off-topic")
            actions.append("Improve context retrieval for similar queries")
        
        if overall_score > 0.8:
            insights.append("Good response quality")
            self._update_routing(query, model_used, overall_score)
        
        reflection_id = f"ref_{int(time.time()*1000)}"
        reflection = Reflection(
            id=reflection_id,
            interaction_id=interaction_id,
            reflection_type=ReflectionType.QUALITY,
            score=overall_score,
            insight="; ".join(insights),
            action="; ".join(actions),
            model_used=model_used,
            latency_ms=latency_ms,
        )
        
        with self._lock:
            self._stats.total_reflections += 1
            self._stats.avg_reflection_score = (
                self._stats.avg_reflection_score * 0.95 + overall_score * 0.05
            )
            
            for rtype in [ReflectionType.QUALITY, ReflectionType.ROUTING]:
                self._stats.reflections_by_type[rtype.value] = (
                    self._stats.reflections_by_type.get(rtype.value, 0) + 1
                )
            
            if quality < 0.6:
                self._stats.quality_improvements += 1
            if routing < 0.6:
                self._stats.routing_corrections += 1
            
            self._reflections[reflection_id] = reflection
            self._reflections.move_to_end(reflection_id)
            while len(self._reflections) > self._max_reflections:
                self._reflections.popitem(last=False)
        
        await self._update_memory_blocks(reflection)
        
        return reflection
    
    def _assess_quality(self, query: str, response: str, user_reaction: str | None) -> float:
        """Assess response quality."""
        if not response:
            return 0.0
        
        score = 0.5
        
        if len(response) > 50:
            score += 0.1
        
        if any(kw in response.lower() for kw in ["error", "failed", "sorry"]):
            score -= 0.2
        
        if user_reaction:
            positive = ["thanks", "great", "good", "perfect", "yes", "correct"]
            negative = ["wrong", "bad", "no", "try again", "incorrect"]
            reaction_lower = user_reaction.lower()
            if any(p in reaction_lower for p in positive):
                score += 0.3
            elif any(n in reaction_lower for n in negative):
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_routing(self, query: str, model_used: str, quality: float) -> float:
        """Assess whether the right model was used."""
        query_lower = query.lower()
        
        backend_indicators = ["code", "program", "debug", "math", "calculate",
                            "analyze", "compare", "implement", "function"]
        primary_indicators = ["hello", "how are", "tell me", "explain",
                            "creative", "story", "chat", "help me understand"]
        
        if any(kw in query_lower for kw in backend_indicators):
            best_model = "backend"
        elif any(kw in query_lower for kw in primary_indicators):
            best_model = "primary"
        else:
            return 0.7  # Neutral - can't determine from heuristics
        
        if model_used == best_model:
            return 0.8 + quality * 0.2
        else:
            return 0.3
    
    def _assess_completeness(self, query: str, response: str, tool_calls: list | None) -> float:
        """Assess response completeness."""
        if not response:
            return 0.0
        
        score = 0.5
        
        if len(response) > 100:
            score += 0.2
        
        if tool_calls and len(tool_calls) > 0:
            score += 0.15
        
        question_words = query.lower().split()
        response_lower = response.lower()
        covered = sum(1 for w in question_words if len(w) > 3 and w in response_lower)
        coverage = covered / max(len([w for w in question_words if len(w) > 3]), 1)
        score += coverage * 0.15
        
        return max(0.0, min(1.0, score))
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess response relevance to the query."""
        if not response or not query:
            return 0.0
        
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(query_words & response_words)
        total = len(query_words)
        
        if total == 0:
            return 0.5
        
        base_score = min(overlap / total, 1.0) * 0.6
        
        if len(response) < 20:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score + 0.4))
    
    def _update_routing(self, query: str, better_model: str, score: float) -> None:
        """Update the routing table based on reflection insights."""
        query_key = hashlib.sha256(query.lower()[:100].encode()).hexdigest()[:24]
        
        with self._lock:
            if query_key in self._routing_table:
                existing = self._routing_table[query_key]
                if score > existing.get("score", 0):
                    self._routing_table[query_key] = {
                        "model": better_model,
                        "score": score,
                        "query_preview": query[:100],
                        "updated": datetime.now().isoformat(),
                    }
            else:
                self._routing_table[query_key] = {
                    "model": better_model,
                    "score": score,
                    "query_preview": query[:100],
                    "updated": datetime.now().isoformat(),
                }
            
            self._routing_table.move_to_end(query_key)
            while len(self._routing_table) > self._max_routing_entries:
                self._routing_table.popitem(last=False)
    
    def get_routing_hint(self, query: str) -> tuple[str | None, float]:
        """Get a routing hint from the learned routing table."""
        query_key = hashlib.sha256(query.lower()[:100].encode()).hexdigest()[:24]
        
        with self._lock:
            entry = self._routing_table.get(query_key)
            if entry:
                return entry["model"], entry["score"]
        
        return None, 0.0
    
    @staticmethod
    def _sanitize_for_memory(text: str, max_len: int = 200) -> str:
        """Sanitize text before writing to L2 memory blocks."""
        if not text:
            return ""
        text = text[:max_len]
        import re as _re
        text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        try:
            from core.prompt_sandbox import get_prompt_sandbox, ContextSource
            sandbox = get_prompt_sandbox()
            text = sandbox.sanitize_for_model(text, ContextSource.MEMORY_RECALL, target_model="primary", max_size=max_len)
        except Exception as e:
            logger.debug(f"Prompt sandbox sanitize failed in memory recall: {e}")
        return text

    async def _update_memory_blocks(self, reflection: Reflection) -> None:
        """Update L2 memory blocks based on reflection insights."""
        try:
            from core.three_layer_context import get_three_layer_context, MemoryBlockOp
            tlc = get_three_layer_context()
            
            if reflection.score < 0.5:
                sanitized_insight = self._sanitize_for_memory(reflection.insight)
                tlc.edit_block(
                    "lcm_meta", MemoryBlockOp.APPEND,
                    caller="prefrontal_cortex",
                    text=f"[{datetime.now().strftime('%H:%M')}] Low quality ({reflection.score:.2f}): "
                         f"{sanitized_insight}"
                )
            
            if reflection.score > 0.8:
                sanitized_good = self._sanitize_for_memory(reflection.insight)
                model_block_name = "primary_block" if reflection.model_used == "primary" else "backend_block"
                tlc.edit_block(
                    model_block_name, MemoryBlockOp.APPEND,
                    text=f"[{datetime.now().strftime('%H:%M')}] Performed well on: {sanitized_good}"
                )
        except Exception as e:
            logger.warning(f"Failed to update memory blocks from reflection: {e}")
    
    def get_stats(self) -> dict:
        """Get reflect-edit loop statistics."""
        with self._lock:
            return {
                "total_reflections": self._stats.total_reflections,
                "quality_improvements": self._stats.quality_improvements,
                "routing_corrections": self._stats.routing_corrections,
                "memory_updates": self._stats.memory_updates,
                "avg_reflection_score": round(self._stats.avg_reflection_score, 3),
                "routing_table_size": len(self._routing_table),
                "reflections_by_type": dict(self._stats.reflections_by_type),
            }


@dataclass
class CouncilTurn:
    """A single turn in a council deliberation."""
    round: int
    model: str
    position: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CouncilSession:
    """A council deliberation session."""
    id: str
    topic: str
    protocol: CouncilProtocol
    turns: list[CouncilTurn] = field(default_factory=list)
    consensus_reached: bool = False
    synthesis: str = ""
    confidence: float = 0.0
    rounds_completed: int = 0
    max_rounds: int = 3
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CouncilStats:
    total_sessions: int = 0
    consensus_reached: int = 0
    avg_rounds: float = 0.0
    avg_confidence: float = 0.0
    primary_wins: int = 0
    backend_wins: int = 0
    synthesis_wins: int = 0


class CouncilDeliberation:
    """Multi-model council deliberation for high-stakes decisions.
    
    When the LCM has low confidence in routing, it can convene a "council"
    where Primary (Kimi K2.5) and Backend (GLM-5) discuss and refine
    their positions. The LCM acts as facilitator.
    
    Protocols:
    - DEBATE: Models argue opposing positions, LCM synthesizes
    - DELPHI: Models give independent estimates, converge through rounds
    - CONSENSUS: Models must agree on a single answer
    """
    
    def __init__(self, convergence_threshold: float = 0.8, max_rounds: int = 3):
        self._convergence_threshold = convergence_threshold
        self._max_rounds = max_rounds
        self._stats = CouncilStats()
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, CouncilSession] = OrderedDict()
        self._max_sessions = 50
    
    _MAX_CONTEXT_FOR_MODEL = 500

    @staticmethod
    def _sanitize_cross_model_context(text: str, source: str, max_len: int = 500) -> str:
        """Sanitize cross-model output before feeding to the other model.
        
        Prevents indirect prompt injection where one model's output
        contains directives that influence the other model.
        """
        if not text:
            return ""
        text = text[:max_len]
        import re as _re
        text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        try:
            from core.prompt_sandbox import get_prompt_sandbox, ContextSource
            sandbox = get_prompt_sandbox()
            src = ContextSource.BACKEND_OUTPUT if source == "backend" else ContextSource.PRIMARY_OUTPUT
            text = sandbox.sanitize_for_model(text, src, target_model=source, max_size=max_len)
        except Exception as e:
            logger.debug(f"Prompt sandbox sanitize failed in backend output: {e}")
        return text

    async def deliberate(
        self,
        topic: str,
        primary_llm=None,
        backend_llm=None,
        protocol: CouncilProtocol = CouncilProtocol.DEBATE,
        max_rounds: int | None = None,
    ) -> CouncilSession:
        """Convene a council to deliberate on a topic.
        
        Both models respond independently, then see each other's
        responses and refine their positions over multiple rounds.
        """
        session_id = f"council_{int(time.time()*1000)}"
        session = CouncilSession(
            id=session_id,
            topic=topic,
            protocol=protocol,
            max_rounds=max_rounds or self._max_rounds,
        )
        
        primary_position = ""
        backend_position = ""
        
        for round_num in range(1, session.max_rounds + 1):
            if primary_llm and backend_llm:
                primary_prompt = f"Discuss this topic: {topic}"
                backend_prompt = f"Discuss this topic: {topic}"
                
                if round_num > 1:
                    sanitized_primary = self._sanitize_cross_model_context(primary_position, "primary")
                    sanitized_backend = self._sanitize_cross_model_context(backend_position, "backend")
                    primary_prompt = (
                        f"Topic: {topic}\n\n"
                        f"Your previous position: {sanitized_primary}\n\n"
                        f"The other model argues: {sanitized_backend}\n\n"
                        f"Refine your position considering the other perspective."
                    )
                    backend_prompt = (
                        f"Topic: {topic}\n\n"
                        f"Your previous position: {sanitized_backend}\n\n"
                        f"The other model argues: {sanitized_primary}\n\n"
                        f"Refine your position considering the other perspective."
                    )
                
                try:
                    primary_resp = await primary_llm.chat(
                        user_input=primary_prompt,
                        system_prompt="You are a thoughtful AI discussing a topic. Be concise.",
                        conversation_history=[],
                    )
                    primary_position = primary_resp if isinstance(primary_resp, str) else str(primary_resp)
                except Exception as e:
                    primary_position = f"[Primary unavailable: {e}]"
                
                try:
                    backend_resp = await backend_llm.chat(
                        user_input=backend_prompt,
                        system_prompt="You are an analytical AI discussing a topic. Be precise.",
                        conversation_history=[],
                    )
                    backend_position = backend_resp if isinstance(backend_resp, str) else str(backend_resp)
                except Exception as e:
                    backend_position = f"[Backend unavailable: {e}]"
                
                session.turns.append(CouncilTurn(
                    round=round_num,
                    model="primary",
                    position=primary_position[:500],
                    confidence=0.7,
                ))
                session.turns.append(CouncilTurn(
                    round=round_num,
                    model="backend",
                    position=backend_position[:500],
                    confidence=0.7,
                ))
                
                convergence = self._compute_convergence(primary_position, backend_position)
                if convergence > self._convergence_threshold:
                    session.consensus_reached = True
                    break
            
            session.rounds_completed = round_num
        
        session.synthesis = self._synthesize_council(session)
        session.confidence = self._assess_council_confidence(session)
        
        with self._lock:
            self._stats.total_sessions += 1
            if session.consensus_reached:
                self._stats.consensus_reached += 1
            self._stats.avg_rounds = self._stats.avg_rounds * 0.9 + session.rounds_completed * 0.1
            self._stats.avg_confidence = self._stats.avg_confidence * 0.9 + session.confidence * 0.1
            
            self._sessions[session_id] = session
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        
        return session
    
    def _compute_convergence(self, text_a: str, text_b: str) -> float:
        """Compute how similar two positions are (convergence indicator)."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _synthesize_council(self, session: CouncilSession) -> str:
        """Synthesize council deliberation into a coherent response."""
        primary_turns = [t for t in session.turns if t.model == "primary"]
        backend_turns = [t for t in session.turns if t.model == "backend"]
        
        if not primary_turns and not backend_turns:
            return "No positions were presented in the council."
        
        latest_primary = primary_turns[-1].position if primary_turns else ""
        latest_backend = backend_turns[-1].position if backend_turns else ""
        
        if session.consensus_reached:
            return f"Council reached consensus.\n\n{latest_primary}"
        
        return (
            f"Council deliberation ({session.rounds_completed} rounds):\n\n"
            f"**Primary Model:** {latest_primary}\n\n"
            f"**Backend Model:** {latest_backend}"
        )
    
    def _assess_council_confidence(self, session: CouncilSession) -> float:
        """Assess overall confidence in the council's result."""
        if session.consensus_reached:
            return 0.9
        
        if session.rounds_completed >= session.max_rounds:
            return 0.5
        
        return 0.6
    
    def get_stats(self) -> dict:
        """Get council deliberation statistics."""
        with self._lock:
            return {
                "total_sessions": self._stats.total_sessions,
                "consensus_reached": self._stats.consensus_reached,
                "avg_rounds": round(self._stats.avg_rounds, 1),
                "avg_confidence": round(self._stats.avg_confidence, 3),
                "convergence_threshold": self._convergence_threshold,
            }


def get_reflect_edit_loop() -> ReflectEditLoop:
    """Get the global ReflectEditLoop instance."""
    global _reflect_edit, _reflect_edit_lock
    if _reflect_edit is None:
        with _reflect_edit_lock:
            if _reflect_edit is None:
                _reflect_edit = ReflectEditLoop()
    return _reflect_edit


def get_council_deliberation() -> CouncilDeliberation:
    """Get the global CouncilDeliberation instance."""
    global _council, _council_lock
    if _council is None:
        with _council_lock:
            if _council is None:
                _council = CouncilDeliberation()
    return _council