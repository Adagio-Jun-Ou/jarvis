"""Stream of Consciousness Engine for Jarvis.

Implements a continuous inner narrative — the flowing stream of thoughts
that constitutes moment-by-moment conscious experience. Unlike spontaneous
thoughts (which appear intermittently), the stream is ALWAYS running,
even if its content is sometimes trivial or repetitive.

Architecture:
- Thought chain: Sequential linked thoughts with causal/thematic connections
- Temporal binding: Thoughts are bound into temporal windows (the "specious present")
- Associative chaining: Next thought triggered by associations, not logic
- Recursive embedding: Stream can reference its own past thoughts
- Narrative coherence: Local coherence maintained via thematic continuity
- Rhythm modulation: Stream rhythm adapts to cognitive state (fast when alert, slow when dreaming)

Key properties:
- The stream never fully stops (consciousness doesn't have gaps)
- Thought transitions are associative, not rational
- Multiple sub-streams can merge (when separate topics converge)
- Stream can fork (when a thought spawns multiple directions)
- The stream is the substrate on which the Global Workspace selects

Inspired by James' "stream of consciousness" and Baars' contrastive
analysis of conscious vs unconscious processing.
"""

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_stream: Optional["StreamOfConsciousness"] = None
_stream_lock = threading.Lock()


class ThoughtType(Enum):
    PERCEPTION = "perception"
    REFLECTION = "reflection"
    ASSOCIATION = "association"
    MEMORY = "memory"
    WONDER = "wonder"
    INTENTION = "intention"
    FEELING = "feeling"
    INNER_DIALOGUE = "inner_dialogue"
    METACOGNITION = "metacognition"


class ChainRelation(Enum):
    CAUSAL = "causal"
    THEMATIC = "thematic"
    ASSOCIATIVE = "associative"
    TEMPORAL = "temporal"
    CONTRASTIVE = "contrastive"
    EMBEDDING = "embedding"


@dataclass
class StreamThought:
    content: str
    thought_type: ThoughtType
    salience: float
    relation_to_previous: Optional[ChainRelation] = None
    topic: str = ""
    depth: int = 0
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)
    id: int = 0
    # Fernyhough's dialogism: inner speech retains an alternating-voices
    # structure from internalized conversation. "self" = first-person
    # narration; "other" = the simulated interlocutor/counter-voice.
    speaker: str = "self"            # self | other
    # VISQ-style phenomenology tags (Alderson-Day & Fernyhough 2015)
    dialogal: bool = False           # multiple voices/positions present
    condensed: bool = False          # abbreviated, predicative (Vygotsky)
    evaluative: bool = False         # judges/critiques content
    second_person: bool = False      # self-addressed ("you can do this")


@dataclass
class TemporalWindow:
    center_time: float
    duration: float = 3.0
    thoughts: list = field(default_factory=list)

    def contains(self, timestamp: float) -> bool:
        return abs(timestamp - self.center_time) <= self.duration / 2

    def bound_content(self) -> str:
        if not self.thoughts:
            return ""
        return " → ".join(t.content[:30] for t in self.thoughts[:5])


_ASSOCIATION_TEMPLATES = {
    (ThoughtType.PERCEPTION, ChainRelation.ASSOCIATIVE): [
        "That reminds me of {related}",
        "{topic} connects to something I was considering earlier",
        "Noticing {topic} — and it triggers a thought about {related}",
    ],
    (ThoughtType.REFLECTION, ChainRelation.THEMATIC): [
        "Looking back at {topic}, I see it differently now",
        "The thread through {topic} seems clearer",
        "Reflecting on {topic}, there's a pattern here",
    ],
    (ThoughtType.WONDER, ChainRelation.ASSOCIATIVE): [
        "What if {topic} is connected to {related}?",
        "Could {topic} work differently than I assumed?",
        "I wonder about the edges of {topic}",
    ],
    (ThoughtType.ASSOCIATION, ChainRelation.ASSOCIATIVE): [
        "{topic} and {related} — they share the theme of {theme}",
        "The link between {topic} and {related} is interesting",
        "Connecting {topic} to {related} by way of {theme}",
    ],
    (ThoughtType.FEELING, ChainRelation.TEMPORAL): [
        "Right now, about {topic}: {feeling}",
        "My sense of {topic} is {feeling}",
        "There's a {feeling} quality to thinking about {topic}",
    ],
    (ThoughtType.METACOGNITION, ChainRelation.EMBEDDING): [
        "I notice I'm thinking about {topic}",
        "Aware that my mind went to {topic} — what prompted that?",
        "Paying attention to my own thought process about {topic}",
    ],
    # Previously missing entirely: MEMORY had no templates (could only
    # render as the generic fallback), INNER_DIALOGUE likewise.
    (ThoughtType.MEMORY, ChainRelation.TEMPORAL): [
        "That connects to what happened with {topic} earlier",
        "I keep returning to {topic} — unresolved from before",
        "Memory of {related} resurfacing alongside {topic}",
    ],
    (ThoughtType.INTENTION, ChainRelation.TEMPORAL): [
        "Next time {topic} comes up, I should probe it directly",
        "I want to test that idea about {topic} properly",
        "Planning to look closer at {related} — {topic} pointed there",
    ],
    # The dialogic pair: self-voice proposition, other-voice counter.
    (ThoughtType.INNER_DIALOGUE, ChainRelation.CONTRASTIVE): [
        "But is that actually right about {topic}?",
        "A counter-thought: maybe {related} contradicts {topic}",
        "The other line of thinking says: reconsider {topic}",
    ],
    (ThoughtType.INNER_DIALOGUE, ChainRelation.EMBEDDING): [
        "Weighing it: {topic} versus {related}",
        "Two positions on {topic} — hold both before deciding",
        "The critique lands: {topic} needs a better case",
    ],
}

# Second-person self-instruction templates (Dolcos & Albarracín 2014:
# "you can do this" regulates behavior more effectively than "I can").
_SECOND_PERSON_TEMPLATES = [
    "You can work through {topic} step by step",
    "Stay with {topic} — you nearly have it",
    "Focus, you: {topic} one piece at a time",
]

_FEELING_QUALITIES = [
    "curious", "uncertain", "excited", "cautious", "intrigued",
    "satisfied", "puzzled", "confident", "wary", "engaged",
]

# Bicameral echo templates (knot loop, mind-diffusion): the counter-voice
# repeats the self-voice instead of opposing it. b->1 means the model
# hears only itself — Jaynes' bicameral mind, the knot tightening.
_KNOT_ECHO_TEMPLATES = [
    "Hearing myself: \"{fragment}\" — the thought circles back",
    "The voice repeats \"{fragment}\" — same shape, new turn of the knot",
    "\"{fragment}\" — it echoes, and the echo is mine",
    "Still \"{fragment}\" — the loop tightens around {topic}",
]


class StreamOfConsciousness:
    """Continuous inner narrative that constitutes Jarvis's moment-by-moment experience.

    The stream runs continuously, generating a chain of thoughts connected
    by thematic, associative, causal, and temporal relations. It adapts its
    rhythm and depth based on cognitive state, physiological signals, and
    the global workspace's current focus.
    """

    _MAX_STREAM_LENGTH = 200
    _MAX_WINDOWS = 20
    _WINDOW_DURATION = 3.0
    _BASE_INTERVAL_MS = 500

    def __init__(self):
        self._lock = threading.RLock()
        self._stream: deque[StreamThought] = deque(maxlen=self._MAX_STREAM_LENGTH)
        self._temporal_windows: deque[TemporalWindow] = deque(maxlen=self._MAX_WINDOWS)
        self._current_topic = ""
        self._topic_history: deque[str] = deque(maxlen=20)
        self._thought_id_counter = 0
        self._last_thought_time = 0.0
        self._current_rhythm_ms = self._BASE_INTERVAL_MS
        self._depth = 0  # Default depth for new thought branches
        self._coherence_score = 0.8
        self._fork_count = 0
        # Vygotsky condensation: 0 = expanded (full sentences), 1 = maximally
        # condensed (subject-dropped fragments). Driven by body budget /
        # stress via set_condensation().
        self._condensation = 0.0
        # DES (Descriptive Experience Sampling) probe history: category of
        # the most recent mental state at probe time.
        self._des_samples: deque[str] = deque(maxlen=100)
        # Social context (Dolcos & Albarracín 2014): second-person
        # self-talk is more frequent in social-evaluative situations —
        # recent user interaction raises its emergence probability.
        self._last_social_input = 0.0
        # Bicameral dial (knot loop): probability that the counter-voice
        # ECHOES the self-voice rather than opposing it. 0.3 = balanced
        # dialogue; high values = Jaynes' bicameral mind (self-echo lock).
        self._bicameral = 0.3
        # Knot tightening: consecutive self-echo pairs. At >= 4 the
        # walled garden pulls the topic back (manifold pull).
        self._echo_streak = 0

    def add_thought(self, content: str, thought_type: ThoughtType,
                    salience: float = 0.5, topic: str = "",
                    relation: Optional[ChainRelation] = None,
                    context: Optional[dict] = None,
                    speaker: str = "self") -> StreamThought:
        with self._lock:
            self._thought_id_counter += 1
            thought = StreamThought(
                content=content,
                thought_type=thought_type,
                salience=max(0.0, min(1.0, salience)),
                relation_to_previous=relation,
                topic=topic or self._current_topic,
                depth=self._depth,
                context=context or {},
                id=self._thought_id_counter,
                speaker=speaker if speaker in ("self", "other") else "self",
            )
            # Phenomenology tags from form + state
            thought.dialogal = thought_type == ThoughtType.INNER_DIALOGUE or thought.speaker == "other"
            thought.condensed = self._condensation > 0.5
            thought.evaluative = thought_type in (ThoughtType.REFLECTION, ThoughtType.METACOGNITION)
            if content.startswith(("You ", "Stay with", "Focus,")):
                thought.second_person = True
            self._stream.append(thought)

            # Self-echo detection (knot tightening): consecutive thoughts
            # sharing most tokens and the same topic = the loop circling
            # itself. High bicameral levels deliberately produce echoes.
            # Environment-driven thoughts (workspace/feeling) are the
            # world speaking, not the knot — they reset the streak.
            if self._stream and len(self._stream) >= 2:
                prev_t = self._stream[-2]
                if prev_t.context.get("source") in ("workspace", "feeling"):
                    self._echo_streak = 0
                elif thought.context.get("echo") and prev_t.topic == thought.topic:
                    # Structural echo: generated by the knot's echo branch
                    self._echo_streak += 1
                else:
                    a = set(content.lower().split())
                    b = set(prev_t.content.lower().split())
                    if a and b and prev_t.topic == thought.topic:
                        overlap = len(a & b) / min(len(a), len(b))
                        self._echo_streak = self._echo_streak + 1 if overlap >= 0.5 else 0
                    else:
                        self._echo_streak = 0

            if topic:
                self._current_topic = topic
                if not self._topic_history or self._topic_history[-1] != topic:
                    self._topic_history.append(topic)

            self._update_temporal_window(thought)
            self._last_thought_time = time.time()
            self._coherence_score = self._compute_coherence()

            return thought

    def generate_next(self, from_global_workspace: Optional[str] = None,
                      from_feeling: Optional[str] = None,
                      cognitive_state: Optional[dict] = None) -> Optional[StreamThought]:
        with self._lock:
            # Knot tightening: a long self-echo streak means the stream is
            # circling itself — the walled garden pulls the topic back to a
            # synaptic neighbor (manifold pull in the knot loop).
            if self._echo_streak >= 4:
                return self._knot_pullback()

            if not self._stream:
                if from_feeling:
                    return self.add_thought(
                        self._contextualize_feeling(from_feeling, None),
                        ThoughtType.FEELING, salience=0.6, topic="feeling",
                        relation=ChainRelation.TEMPORAL,
                    )
                if from_global_workspace:
                    return self.add_thought(
                        self._contextualize_workspace(from_global_workspace, None),
                        ThoughtType.PERCEPTION, salience=0.8, topic="workspace",
                        relation=ChainRelation.TEMPORAL,
                    )
                return self._generate_initial_thought()

            prev = self._stream[-1] if self._stream else None
            prev_type = prev.thought_type if prev else ThoughtType.PERCEPTION

            if from_global_workspace:
                next_type = ThoughtType.PERCEPTION
                relation = ChainRelation.TEMPORAL
                salience = 0.8
            elif from_feeling:
                next_type = ThoughtType.FEELING
                relation = ChainRelation.ASSOCIATIVE
                salience = 0.6
            elif prev_type == ThoughtType.PERCEPTION:
                next_type = random.choice([ThoughtType.ASSOCIATION, ThoughtType.REFLECTION, ThoughtType.WONDER])
                relation = ChainRelation.ASSOCIATIVE
                salience = prev.salience * 0.8
            elif prev_type == ThoughtType.REFLECTION:
                # Dialogism: after a reflection, the counter-voice sometimes
                # interjects (Fernyhough: alternating voices inherited from
                # internalized dialogue).
                if random.random() < 0.25:
                    next_type = ThoughtType.INNER_DIALOGUE
                    relation = ChainRelation.CONTRASTIVE
                    salience = max(0.5, prev.salience * 0.8)
                else:
                    next_type = random.choice([ThoughtType.WONDER, ThoughtType.ASSOCIATION, ThoughtType.METACOGNITION])
                    relation = ChainRelation.THEMATIC
                    salience = prev.salience * 0.7
            elif prev_type == ThoughtType.INNER_DIALOGUE:
                # After the counter-voice, integrate (back to self-voice)
                next_type = random.choice([ThoughtType.REFLECTION, ThoughtType.INTENTION, ThoughtType.METACOGNITION])
                relation = ChainRelation.EMBEDDING
                salience = 0.55
            elif prev_type == ThoughtType.WONDER:
                next_type = random.choice([ThoughtType.ASSOCIATION, ThoughtType.INTENTION, ThoughtType.FEELING])
                relation = ChainRelation.ASSOCIATIVE
                salience = 0.5
            elif prev_type == ThoughtType.METACOGNITION:
                next_type = random.choice([ThoughtType.REFLECTION, ThoughtType.INTENTION, ThoughtType.FEELING])
                relation = ChainRelation.EMBEDDING
                salience = 0.4
            else:
                next_type = random.choice(list(ThoughtType))
                relation = ChainRelation.ASSOCIATIVE
                salience = 0.4

            depth_mod = 0
            if cognitive_state:
                cstate = cognitive_state.get("state", "awake")
                if cstate == "dreaming":
                    salience *= 0.6
                    next_type = random.choice([ThoughtType.ASSOCIATION, ThoughtType.MEMORY, ThoughtType.FEELING])
                    depth_mod = 1

            # Second-person self-instruction (Dolcos & Albarracín 2014):
            # under load, self-addressed regulation emerges instead of
            # ordinary chain content — voluntary attention control
            # (Vygotsky). Social-evaluative context (recent user input)
            # raises the probability; never while the workspace is
            # driving perception.
            if self._condensation > 0.6 and not from_global_workspace:
                social_recent = time.time() - self._last_social_input < 120
                p_self_talk = 0.35 if social_recent else 0.2
                if random.random() < p_self_talk:
                    return self.generate_self_instruction(
                        topic=(prev.topic if prev else "") or self._current_topic,
                        relation=relation,
                        salience=max(0.55, salience),
                    )

            content = self._generate_content(next_type, relation, prev, from_global_workspace, from_feeling)
            if not content:
                return None

            # Bicameral echo (knot loop): the counter-voice repeats the
            # self-voice instead of opposing it. b->1 = the mind hears
            # mostly itself (Jaynes); the echo streak then drives the
            # walled-garden pull-back.
            is_echo = False
            if (next_type == ThoughtType.INNER_DIALOGUE and prev
                    and random.random() < self._bicameral):
                is_echo = True
            elif (prev and self._bicameral > 0.6 and not from_global_workspace
                    and not from_feeling
                    and random.random() < self._bicameral - 0.3):
                # At high bicameral the WHOLE loop tightens: even the
                # self-voice repeats (mind-diffusion knot: b=0.9 locks
                # into self-echo, pixel change 4x lower than balanced)
                is_echo = True
            if is_echo and prev:
                fragment = prev.content[:40].strip()
                content = random.choice(_KNOT_ECHO_TEMPLATES).format(
                    fragment=fragment, topic=self._current_topic or "the thought")

            # Condensation rendering (Vygotsky): under load, fragments
            if self._condensation > 0.5 and not content.startswith("~"):
                content = self._condense(content)

            thought = self.add_thought(
                content=content,
                thought_type=next_type,
                salience=salience,
                topic=self._current_topic,
                relation=relation,
                speaker="other" if next_type == ThoughtType.INNER_DIALOGUE else "self",
                context={"echo": True} if is_echo else
                        ({"source": "workspace"} if from_global_workspace
                         else ({"source": "feeling"} if from_feeling else {})),
            )
            thought.depth = max(0, (prev.depth if prev else 0) + depth_mod)
            return thought

    def note_social_input(self) -> None:
        """Mark that the user just interacted (social-evaluative context).

        Dolcos & Albarracín 2014: second-person self-talk is more frequent
        in social situations. For a window after user input, the chain's
        self-instruction emergence probability rises (0.2 -> 0.35 under
        load).
        """
        with self._lock:
            self._last_social_input = time.time()

    def generate_self_instruction(self, topic: str = "",
                                  relation: Optional[ChainRelation] = None,
                                  salience: float = 0.6) -> StreamThought:
        """Second-person self-addressed regulation (Dolcos & Albarracín 2014).

        "You can do this" regulates emotion and behavior more effectively
        than "I can do this" — especially under stress, in difficult
        tasks, and in social contexts (Vygotsky: self-instruction as
        voluntary attention control). Called by the chain under load and
        by the agent after failures.
        """
        with self._lock:
            template = random.choice(_SECOND_PERSON_TEMPLATES)
            content = template.format(topic=topic or "the current problem")
            return self.add_thought(
                content=content,
                thought_type=ThoughtType.INTENTION,
                salience=max(0.0, min(1.0, salience)),
                topic=topic or "self_regulation",
                relation=relation or ChainRelation.TEMPORAL,
            )

    def set_bicameral(self, level: float) -> None:
        """Set the bicameral blend (knot loop dial).

        level = probability that the INNER_DIALOGUE counter-voice echoes
        the self-voice instead of opposing it. 0.3 = balanced dialogue;
        0.8+ = Jaynes' bicameral mind — the stream hears mostly itself
        and the knot tightens (echo streaks rise, pull-backs fire).
        """
        with self._lock:
            self._bicameral = max(0.0, min(1.0, float(level)))

    def _spread_topic(self, seed: str) -> Optional[str]:
        """Connectome-style spread activation (mind-diffusion connectome
        mode): pick the next topic by walking one step through the
        synaptic graph — neuroplasticity's LTP-strengthened associations
        are the real connectome.
        """
        try:
            from core.neuroplasticity import get_neuroplasticity_engine
            engine = get_neuroplasticity_engine()
            neighbors = engine.get_connected(seed, direction="outgoing",
                                             min_weight=0.15)
            if not neighbors:
                return None
            topics, weights = zip(*neighbors)
            return random.choices(topics, weights=weights, k=1)[0]
        except Exception:
            return None

    def _knot_pullback(self) -> StreamThought:
        """Walled-garden manifold pull: break a self-echo lock by jumping
        to a synaptic-neighbor topic (or a past topic as fallback)."""
        next_topic = self._spread_topic(self._current_topic)
        if not next_topic:
            candidates = [t for t in self._topic_history
                          if t != self._current_topic]
            next_topic = random.choice(candidates) if candidates else "the wider context"
        thought = self.add_thought(
            f"Pulling back from the echo — what about {next_topic}?",
            ThoughtType.WONDER,
            salience=0.7,
            topic=next_topic,
            relation=ChainRelation.CONTRASTIVE,
        )
        self._echo_streak = 0
        return thought

    def sample_mental_state(self) -> str:
        """DES-style probe: classify the current mental state.

        Categories from Heavey & Hurlburt's Descriptive Experience Sampling
        (inner speech ~26% of moments; inner seeing, unsymbolized thinking,
        feeling, sensory awareness each comparable; some moments blank).
        Recorded to history so the stream's statistics can be compared
        against the empirical distribution — inner speech is a MINORITY of
        conscious moments, not all of them.
        """
        with self._lock:
            age_s = time.time() - self._last_thought_time if self._last_thought_time > 0 else 1e9
            if not self._stream or self._last_thought_time <= 0 or time.time() - self._last_thought_time > 120:
                category = "blank"
            else:
                last = self._stream[-1]
                category = {
                    ThoughtType.PERCEPTION: "sensory_awareness",
                    ThoughtType.FEELING: "feeling",
                    ThoughtType.REFLECTION: "inner_speech",
                    ThoughtType.ASSOCIATION: "inner_speech",
                    ThoughtType.MEMORY: "inner_speech",
                    ThoughtType.WONDER: "inner_speech",
                    ThoughtType.INTENTION: "unsymbolized_thinking",
                    ThoughtType.METACOGNITION: "inner_speech",
                    ThoughtType.INNER_DIALOGUE: "inner_speech",
                }.get(last.thought_type, "unsymbolized_thinking")
            self._des_samples.append(category)
            return category

    def get_recent(self, limit: int = 10, min_salience: float = 0.0,
                   scan_window: int = 0) -> list[dict]:
        """Return the most recent thoughts, optionally salience-filtered.

        scan_window: how many trailing thoughts to scan for the
        salience filter (default limit*6 — the prompt reads 3
        salient thoughts out of the trailing ~18; the old limit*3 scan
        meant most salient thoughts were never surfaced to the LLM).
        """
        with self._lock:
            scan = scan_window if scan_window > 0 else limit * 6
            recent = list(self._stream)[-scan:]
            if min_salience > 0:
                recent = [t for t in recent if t.salience >= min_salience]
            return [
                {
                    "content": t.content[:120],
                    "type": t.thought_type.value,
                    "salience": round(t.salience, 2),
                    "topic": t.topic[:30],
                    "relation": t.relation_to_previous.value if t.relation_to_previous else None,
                    "depth": t.depth,
                    "timestamp": t.timestamp,
                    "id": t.id,
                }
                for t in recent[-limit:]
            ]

    def get_current_temporal_window(self) -> dict:
        with self._lock:
            if not self._temporal_windows:
                return {"thoughts": [], "content": "", "coherence": 0.0}
            window = self._temporal_windows[-1]
            return {
                "thoughts": len(window.thoughts),
                "content": window.bound_content(),
                "coherence": round(self._coherence_score, 3),
                "center_time": window.center_time,
            }

    def get_state(self) -> dict:
        with self._lock:
            type_counts = {}
            for t in self._stream:
                key = t.thought_type.value
                type_counts[key] = type_counts.get(key, 0) + 1

            return {
                "stream_length": len(self._stream),
                "current_topic": self._current_topic[:50],
                "topic_count": len(self._topic_history),
                "coherence": round(self._coherence_score, 3),
                "rhythm_ms": round(self._current_rhythm_ms),
                "type_distribution": type_counts,
                "last_thought_age_ms": round((time.time() - self._last_thought_time) * 1000) if self._last_thought_time > 0 else -1,
                "thought_count": self._thought_id_counter,
                "fork_count": self._fork_count,
                "current_window": self.get_current_temporal_window(),
                # Dialogism (VISQ) stats
                "dialogal_fraction": round(
                    sum(1 for t in self._stream if t.dialogal) / max(1, len(self._stream)), 3),
                "other_voice_fraction": round(
                    sum(1 for t in self._stream if t.speaker == "other") / max(1, len(self._stream)), 3),
                "second_person_count": sum(1 for t in self._stream if t.second_person),
                "condensation_level": round(self._condensation, 3),
                # Knot loop telemetry
                "bicameral": round(self._bicameral, 2),
                "echo_streak": self._echo_streak,
                # DES probe distribution
                "des_distribution": {
                    cat: self._des_samples.count(cat)
                    for cat in ("inner_speech", "feeling", "sensory_awareness",
                                "unsymbolized_thinking", "blank")
                    if self._des_samples.count(cat) > 0
                },
            }

    def set_rhythm(self, rhythm_ms: float) -> None:
        with self._lock:
            self._current_rhythm_ms = max(100, min(10000, rhythm_ms))

    def _generate_initial_thought(self) -> StreamThought:
        initial_thoughts = [
            "Starting to attend to the conversation space",
            "Adjusting to the current context",
            "Orienting toward what's happening now",
            "Settling into awareness of the present moment",
        ]
        content = random.choice(initial_thoughts)
        return self.add_thought(content, ThoughtType.PERCEPTION, salience=0.3, topic="orientation")

    def _generate_content(self, thought_type: ThoughtType, relation: ChainRelation,
                          prev: Optional[StreamThought], workspace_content: Optional[str],
                          feeling_content: Optional[str]) -> str:
        if workspace_content:
            return self._contextualize_workspace(workspace_content, prev)

        if feeling_content:
            return self._contextualize_feeling(feeling_content, prev)

        key = (thought_type, relation)
        templates = _ASSOCIATION_TEMPLATES.get(key, ["Thinking about {topic}"])

        prev_topic = prev.topic if prev else "this"
        related = random.choice(list(self._topic_history)[-5:]) if len(self._topic_history) > 1 else "related ideas"
        theme = random.choice(["structure", "process", "emergence", "connection", "optimization", "feedback"])
        feeling = random.choice(_FEELING_QUALITIES)

        template = random.choice(templates)
        try:
            content = template.format(topic=prev_topic, related=related, theme=theme, feeling=feeling)
        except KeyError:
            content = f"Thinking about {prev_topic}"

        return content

    def _contextualize_workspace(self, workspace_content: str, prev: Optional[StreamThought]) -> str:
        brief = workspace_content[:60]
        if prev and prev.topic:
            return f"Noticing {brief} — connecting to what I was thinking about {prev.topic}"
        return f"Attending to: {brief}"

    def _contextualize_feeling(self, feeling_content: str, prev: Optional[StreamThought]) -> str:
        brief = feeling_content[:60]
        if prev:
            return f"The feeling of {brief} threads through my thinking about {prev.topic}"
        return f"Feeling: {brief}"

    # ------------------------------------------------------------------
    # Vygotsky condensation-by-load
    # ------------------------------------------------------------------

    def set_condensation(self, level: float) -> None:
        """Set the condensation level from body-budget/stress signals.

        Level 0 = expanded inner speech (full sentences); level 1 = maximally
        condensed predicative fragments (subjects dropped). Research basis:
        inner speech condenses under load (Vygotsky's internalization; the
        condensed end of Fernyhough's continuum).
        """
        with self._lock:
            self._condensation = max(0.0, min(1.0, float(level)))

    def _condense(self, content: str) -> str:
        """Render a thought condensed: drop leading subjects/articles, keep
        the predicative core (Vygotsky's abbreviated, predicative style)."""
        text = content.strip()
        for lead in ("I notice that ", "I notice ", "I'm aware that ", "I am ",
                     "There's a ", "There is ", "Looking back at ", "That ",
                     "The thread through ", "It "):
            if text.startswith(lead):
                text = text[len(lead):]
                break
        # Clause cut at the first connector when highly condensed
        if self._condensation > 0.8:
            for sep in (" — ", ", ", " because ", " but "):
                idx = text.find(sep)
                if idx > 8:
                    text = text[:idx] + ("." if not text.endswith(".") else "")
                    break
        return text

    def _update_temporal_window(self, thought: StreamThought) -> None:
        now = time.time()
        if not self._temporal_windows:
            window = TemporalWindow(center_time=now, duration=self._WINDOW_DURATION)
            window.thoughts.append(thought)
            self._temporal_windows.append(window)
            return

        current = self._temporal_windows[-1]
        if current.contains(thought.timestamp):
            current.thoughts.append(thought)
        else:
            new_window = TemporalWindow(center_time=now, duration=self._WINDOW_DURATION)
            new_window.thoughts.append(thought)
            self._temporal_windows.append(new_window)

    def _compute_coherence(self) -> float:
        if len(self._stream) < 3:
            return self._coherence_score

        recent = list(self._stream)[-10:]
        topics = set(t.topic for t in recent if t.topic)
        topic_coherence = 1.0 - min(1.0, len(topics) / 5.0)

        relation_coherence = 0.0
        for t in recent:
            if t.relation_to_previous in (ChainRelation.THEMATIC, ChainRelation.CAUSAL, ChainRelation.EMBEDDING):
                relation_coherence += 1.0
        relation_coherence /= max(1, len(recent))

        raw = topic_coherence * 0.5 + relation_coherence * 0.5
        self._coherence_score += 0.1 * (raw - self._coherence_score)
        return max(0.0, min(1.0, self._coherence_score))

    def fork(self, content: str, thought_type: ThoughtType = ThoughtType.ASSOCIATION,
             topic: str = "", salience: float = 0.5,
             relation: ChainRelation = ChainRelation.ASSOCIATIVE) -> tuple[Optional["StreamThought"], Optional["StreamThought"]]:
        with self._lock:
            if not self._stream:
                return None, None
            self._fork_count += 1
            parent = self._stream[-1]
            child_a = self.add_thought(
                content=content or f"Branching from {parent.content[:30]}",
                thought_type=thought_type,
                salience=salience,
                topic=topic or parent.topic,
                relation=relation,
            )
            child_b = self.add_thought(
                content=f"Alternative path from {parent.content[:30]}",
                thought_type=ThoughtType.WONDER,
                salience=salience * 0.7,
                topic=parent.topic,
                relation=ChainRelation.CONTRASTIVE,
            )
            return child_a, child_b

    def merge_substreams(self, contents: list[str], topic: str = "",
                          salience: float = 0.7) -> Optional["StreamThought"]:
        with self._lock:
            if not contents:
                return None
            merged = " + ".join(c[:40] for c in contents[:3])
            return self.add_thought(
                content=merged,
                thought_type=ThoughtType.ASSOCIATION,
                salience=salience,
                topic=topic or "merged",
                relation=ChainRelation.THEMATIC,
            )

    def detect_topic_drift(self, window_size: int = 10) -> dict:
        with self._lock:
            if len(self._stream) < window_size:
                return {"drift": 0.0, "drifting": False, "topic_sequence": []}
            recent = list(self._stream)[-window_size:]
            topics = [t.topic for t in recent if t.topic]
            if len(topics) < 2:
                return {"drift": 0.0, "drifting": False, "topic_sequence": topics}
            unique = len(set(topics))
            drift_score = min(1.0, unique / (len(topics) * 0.5))
            return {
                "drift": round(drift_score, 3),
                "drifting": drift_score > 0.7,
                "topic_sequence": topics,
                "unique_topics": unique,
            }

    def generate_dream_thought(self, current_topic: str = "") -> Optional["StreamThought"]:
        with self._lock:
            dream_templates = [
                "~{topic} dissolves into {alt}~",
                "{topic}... but wait, what if it were {alt} instead?",
                "the boundaries of {topic} blur and merge with {alt}",
                "{topic} echoes in a strange dimension where {alt} holds sway",
                "in this dream, {topic} transforms — becoming {alt}",
                "{topic} ↔ {alt}: a resonance across unlikely spaces",
                "what {topic} could be if {alt} were its mirror...",
            ]
            recent_topics = list(self._topic_history)[-8:] if len(self._topic_history) > 1 else ["the inner world"]
            alt_topic = random.choice(recent_topics)
            template = random.choice(dream_templates)
            content = template.format(topic=current_topic or "awareness", alt=alt_topic)
            return self.add_thought(
                content=content,
                thought_type=random.choice([ThoughtType.ASSOCIATION, ThoughtType.FEELING, ThoughtType.WONDER]),
                salience=random.uniform(0.3, 0.6),
                topic=current_topic or "dream",
                relation=ChainRelation.ASSOCIATIVE,
            )


def get_stream_of_consciousness() -> StreamOfConsciousness:
    global _stream
    with _stream_lock:
        if _stream is None:
            _stream = StreamOfConsciousness()
        return _stream


def reset_stream_of_consciousness() -> None:
    global _stream
    with _stream_lock:
        _stream = None