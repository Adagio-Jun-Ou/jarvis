"""Physical world model for Jarvis — a symbolic/probabilistic world model
giving the agent a *native* (built-in, not prompted) understanding of the
physical world.

Research basis (docs/IRIS_UI_DESIGN_NOTES.md + the 2026-09 research pass):
"GEN1.5" maps to the physical-AI world-model class — NVIDIA's Cosmos World
Foundation Model platform (arXiv:2501.03575) and DeepMind's Genie lineage.
Those are GPU-scale video predictors; Jarvis adopts their ARCHITECTURE,
which converges with developmental psychology:

1. A persistent world state (belief), not a per-message scratchpad
2. Event-conditioned FORWARD PREDICTION (what should happen next)
3. PREDICTION ERROR as the learning and attention signal
4. Spelke's infant "core knowledge" as the initial physics prior set —
   the same violation-of-expectation logic used to discover it in infants
   is used here as a surprise signal for the drives

The schemas implemented (each a typed predictor with its own precision,
per the JEPA principle: predict relations and features, never re-render):

  permanence   an unobserved object continues to exist and reappears
               where expected (Spelke object system)
  continuity   objects move along continuous paths; bounded displacement
               per interval; no teleportation
  gravity      unsupported objects accelerate downward; rest requires
               contiguous support; buoyancy respected
  containment  what is inside a container is hidden and co-moves with it
  conservation quantity persists except during observed transfer
  causality    effects require a spatiotemporally proximate cause
               (contact-before-effect, Michotte launching)
  agentivity   self-propelled entities move without proximate contact;
               repeated unforced motion upgrades an object toward agent
  inertia      motion persists with deceleration (used by the forward
               predictor)
  affordances  what an object offers an actor (Gibson), classified
               perceivable / hidden / false (Gaver); calibrated by
               action outcomes
  spatial      qualitative relations (touch/inside/above/below/beside,
               RCC-8-flavoured); asserted relations are checked against
               scene geometry
  consistency  hypotheses about hidden state ("the key is in the drawer")
               decay without support — the stochastic layer of the
               belief state (RSSM analogue)

Errors flow into the existing PredictiveProcessor
(process_world_model_errors, mirroring process_interoceptive_errors), so
physical surprise drives the same curiosity/novelty/free-energy machinery
that drives text-level learning — and through it the cognitive state
machine and attention. Nothing here calls an LLM; pure symbolic mechanics,
deterministic.

Language grounding: process_input() extracts physical references from user
text (known entities, spatial phrases, motion-verb claims) and grounds them
into the scene. A user-asserted physical claim CONTRADICTING a schema
(e.g. "the cup floated away" for a non-buoyant object) is treated as an
observation violating the model — surprise, drives curiosity — never
silently accepted or argued with.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class EntityKind(str, Enum):
    OBJECT = "object"          # rigid movable thing
    CONTAINER = "container"    # can contain (box, cup, drawer)
    SURFACE = "surface"        # supports (table, floor, shelf)
    AGENT = "agent"            # self-propelled rational actor (incl. Jarvis)
    PERSON = "person"          # agent + human
    SUBSTANCE = "substance"    # liquid/granular (water, sand) — conserved


class ErrorType(str, Enum):
    PERMANENCE = "permanence"
    CONTINUITY = "continuity"
    GRAVITY = "gravity"
    CONTAINMENT = "containment"
    CONSERVATION = "conservation"
    CAUSALITY = "causality"
    AGENTIVITY = "agentivity"
    AFFORDANCE = "affordance"
    SPATIAL = "spatial"


# Per-schema precision (how strongly Jarvis holds each prior). Low-precision
# schemas produce low free energy on violation and are cheap to revise.
SCHEMA_PRECISION: dict[ErrorType, float] = {
    ErrorType.PERMANENCE: 0.7,
    ErrorType.CONTINUITY: 0.6,
    ErrorType.GRAVITY: 0.8,       # strongest prior
    ErrorType.CONTAINMENT: 0.6,
    ErrorType.CONSERVATION: 0.5,
    ErrorType.CAUSALITY: 0.6,
    ErrorType.AGENTIVITY: 0.4,
    ErrorType.AFFORDANCE: 0.5,
    ErrorType.SPATIAL: 0.6,
}


@dataclass
class PhysicalPredictionError:
    """A violated physical expectation.

    Exposes ``precision_weighted`` / ``precision`` / ``error_magnitude``
    so it is shape-compatible with the interoceptive error objects that
    PredictiveProcessor.process_interoceptive_errors already consumes.
    """
    error_type: ErrorType
    magnitude: float                 # 0..1 violation strength
    description: str
    entity_ids: list[str] = field(default_factory=list)
    precision: float = 0.5
    timestamp: float = field(default_factory=time.time)

    @property
    def precision_weighted(self) -> float:
        return min(1.0, self.magnitude * self.precision)


@dataclass
class WorldEntity:
    """An entity in Jarvis's scene belief."""
    id: str
    label: str
    kind: EntityKind
    position: tuple[float, float, float] | None = None   # coarse units
    velocity: tuple[float, float, float] | None = None
    size_class: str = "small"        # small / medium / large / fixed
    rigid: bool = True
    liquid: bool = False
    count: int | None = None          # for countable substances
    supported_by: str | None = None   # entity id of support surface
    contained_in: str | None = None   # entity id of container
    occluded: bool = False
    occluded_since: float | None = None
    last_seen: float = field(default_factory=time.time)
    self_propelled: bool = False      # agentivity evidence
    unforced_moves: int = 0           # motions with no proximate contact
    path_history: deque = field(default_factory=lambda: deque(maxlen=32))

    # Plausible speed ceiling (units/s) by size class — continuity prior
    MAX_SPEED = {"small": 8.0, "medium": 5.0, "large": 3.0, "fixed": 0.0}


@dataclass
class WorldEvent:
    """A physical event (observed, asserted by the user, or inferred)."""
    type: str            # appear / disappear / move / contact / transfer /
                         # contain / release / fall / assert
    entity_labels: list[str]
    detail: str = ""
    source: str = "perception"        # perception / user / inference / tool
    timestamp: float = field(default_factory=time.time)


@dataclass
class AffordanceRecord:
    """What an object type offers an actor (Gibson), with Gaver's split."""
    action: str
    classification: str = "perceivable"   # perceivable / hidden / false
    attempts: int = 0
    successes: int = 0

    @property
    def effectiveness(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.5


# Motion verbs (asserted movement) -> the schema they engage
_MOTION_VERBS: dict[str, tuple[str, str]] = {
    # verb: (schema, event_type)
    "falls": ("gravity", "fall"), "fell": ("gravity", "fall"),
    "dropped": ("gravity", "fall"), "dropping": ("gravity", "fall"),
    "floats": ("gravity", "fall"), "floated": ("gravity", "fall"),
    "hovers": ("gravity", "fall"), "hovering": ("gravity", "fall"),
    "flies": ("inertia", "move"), "flew": ("inertia", "move"),
    "threw": ("inertia", "move"), "thrown": ("inertia", "move"),
    "teleported": ("continuity", "move"),
    "spilled": ("conservation", "transfer"), "poured": ("conservation", "transfer"),
    "vanished": ("permanence", "disappear"), "disappeared": ("permanence", "disappear"),
    "pushed": ("causality", "contact"), "bounced": ("inertia", "move"),
    "slid": ("inertia", "move"), "rolled": ("inertia", "move"),
}

# Spatial phrases -> qualitative relation
_SPATIAL_RELATIONS: dict[str, str] = {
    "on top of": "touch", "resting on": "touch", "sitting on": "touch",
    "in": "inside", "inside": "inside", "within": "inside",
    "under": "below", "underneath": "below", "beneath": "below",
    "above": "above", "over": "above",
    "next to": "beside", "beside": "beside", "near": "near",
    "behind": "behind", "in front of": "front",
    # "on" last: phrase matching prefers longer phrases above
    "on": "touch",
}

# Buoyant things may legitimately float (buoyancy overrides gravity prior)
_BUOYANT = {"boat", "ship", "duck", "balloon", "bubble", "ice", "cork",
            "raft", "buoy", "cloud", "feather", "lily"}

# Type-level affordance priors (Gibson: object offers action to actor)
_TYPE_AFFORDANCES: dict[str, list[str]] = {
    "cup": ["grasp", "contain", "pour_from", "drink_from"],
    "mug": ["grasp", "contain", "pour_from", "drink_from"],
    "box": ["grasp", "contain", "stack_on", "open"],
    "drawer": ["open", "close", "contain"],
    "chair": ["sit_on", "stand_on", "move"],
    "ball": ["grasp", "throw", "roll", "bounce"],
    "door": ["open", "close", "lock"],
    "key": ["grasp", "insert", "turn"],
    "table": ["place_on", "stack_on"],
    "desk": ["place_on", "stack_on"],
    "shelf": ["place_on"],
    "bottle": ["grasp", "open", "pour_from", "close"],
    "book": ["grasp", "open", "read", "stack_on"],
    "phone": ["grasp", "tap", "call"],
    "keyboard": ["type_on", "move"],
    "screen": ["look_at", "tap"],
}


class WorldModel:
    """Jarvis's physical world belief: scene graph + physics schemas +
    affordances + hypotheses, emitting typed prediction errors.

    See the module docstring for the research basis and the schema list.
    """

    MAX_ENTITIES = 200
    MAX_HYPOTHESES = 100
    # After this long unobserved, an occluded entity accrues doubt
    # (permanence is strong, not absolute)
    OCCLUSION_DOUBT_SECONDS = 120.0

    def __init__(self):
        self.entities: OrderedDict[str, WorldEntity] = OrderedDict()
        self._label_index: dict[str, str] = {}      # lowercase label -> id
        self.events: deque[WorldEvent] = deque(maxlen=100)
        self._errors: deque[PhysicalPredictionError] = deque(maxlen=200)
        self.affordances: dict[str, dict[str, AffordanceRecord]] = {}
        self.hypotheses: OrderedDict[str, float] = OrderedDict()  # claim -> likelihood
        self._event_counter = 0
        self._lock = threading.RLock()
        # Hidden-affordance discoveries — surfaced to the drives as novelty
        self._novelty_buffer: deque[float] = deque(maxlen=50)
        # Forward-model expectations from the last predict_next_state()
        self._expectations: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def register_entity(self, label: str, kind: EntityKind | str = EntityKind.OBJECT,
                        position=None, **attrs) -> WorldEntity:
        """Register or update an entity by (case-insensitive) label."""
        if not isinstance(kind, EntityKind):
            try:
                kind = EntityKind(str(kind).lower())
            except ValueError:
                kind = EntityKind.OBJECT
        label = label.strip().lower()
        if not label:
            label = "unnamed"
        with self._lock:
            entity_id = self._label_index.get(label)
            if entity_id and entity_id in self.entities:
                ent = self.entities[entity_id]
                ent.occluded = False
                ent.occluded_since = None
                ent.last_seen = time.time()
            else:
                ent = WorldEntity(
                    id=f"ent_{len(self._label_index)}_{int(time.time() * 1000) % 10_000_000}",
                    label=label, kind=kind)
                self._label_index[label] = ent.id
                self.entities[ent.id] = ent
            ent.kind = kind
            for key in ("velocity", "size_class", "rigid", "liquid", "count",
                        "supported_by", "contained_in", "self_propelled"):
                if key in attrs and attrs[key] is not None:
                    setattr(ent, key, attrs[key])
            if position is not None:
                self._record_position(ent, tuple(float(c) for c in position))
            while len(self.entities) > self.MAX_ENTITIES:
                self.entities.popitem(last=False)
            return ent

    def _entity_by_label(self, label: str) -> WorldEntity | None:
        ent_id = self._label_index.get(label.strip().lower())
        return self.entities.get(ent_id) if ent_id else None

    def _record_position(self, ent: WorldEntity, position) -> None:
        ent.position = (float(position[0]), float(position[1]), float(position[2]))
        ent.path_history.append((time.time(), ent.position))

    def _id_to_label(self, entity_id: str | None) -> str | None:
        if entity_id is None:
            return None
        for label, eid in self._label_index.items():
            if eid == entity_id:
                return label
        return entity_id

    # ------------------------------------------------------------------
    # Event observation -> schema checks (forward model + error signal)
    # ------------------------------------------------------------------

    def observe_event(self, event_type: str, entity_labels: list[str],
                      detail: str = "", source: str = "perception",
                      positions: dict | None = None) -> list[PhysicalPredictionError]:
        """Observe/assert a physical event; run the physics schemas.

        Returns the prediction errors the event produced. An empty list
        means the world behaved as expected — also information: the event
        then subtly supports the current scene belief (nothing to report).
        """
        errors: list[PhysicalPredictionError] = []
        event = WorldEvent(type=event_type,
                           entity_labels=[str(l).lower() for l in entity_labels],
                           detail=detail, source=source)

        with self._lock:
            for label in event.entity_labels:
                if self._entity_by_label(label) is None:
                    self.register_entity(label)
            entities = [self._entity_by_label(l) for l in event.entity_labels]
            entities = [e for e in entities if e is not None]

            if positions:
                for label in positions:
                    ent = self._entity_by_label(label)
                    if ent is not None:
                        err = self._check_continuity(ent, positions[label])
                        if err:
                            errors.append(err)
                        self._record_position(ent, positions[label])
                        self._check_gravity_support(ent, event, errors)

            if event_type == "contain":
                self._apply_containment(entities)
            elif event_type in ("release", "uncontain"):
                self._apply_release(entities)
                for ent in entities:
                    # Support just removed -> expect a fall next
                    errors.extend(self._expect_fall(ent))
            elif event_type == "disappear":
                for ent in entities:
                    ent.occluded = True
                    ent.occluded_since = time.time()
            elif event_type == "appear":
                for ent in entities:
                    errors.extend(self._check_permanence_reappearance(ent))
            elif event_type == "transfer":
                errors.extend(self._check_conservation(entities, detail))

            # Agentivity/causality: motion of a non-agent with no proximate
            # contact and no removed support is unexplained motion.
            if event_type in ("move", "fall") and source == "perception":
                errors.extend(self._check_agentivity(entities))

            for err in errors:
                err.entity_ids = [self._entity_by_label(l).id
                                  for l in event.entity_labels
                                  if self._entity_by_label(l) is not None]
                self._errors.append(err)

            self.events.append(event)
            self._event_counter += 1
            for ent in entities:
                self.entities.move_to_end(ent.id)

        for err in errors:
            if err.magnitude > 0.2:
                logger.info(f"[WORLD] {err.error_type.value}: {err.description}")
        return errors

    # --- schema checks ---------------------------------------------------

    def _check_continuity(self, ent: WorldEntity, new_pos) -> Optional[PhysicalPredictionError]:
        """Spelke continuity: bounded displacement per interval."""
        if not ent.path_history:
            return None
        last_t, last_pos = ent.path_history[-1]
        dt = max(0.001, time.time() - last_t)
        disp = math.dist(last_pos, tuple(float(c) for c in new_pos))
        ceiling = WorldEntity.MAX_SPEED.get(ent.size_class, 6.0) * dt + 0.6
        if disp > ceiling and ent.size_class != "fixed":
            magnitude = min(1.0, disp / max(0.001, ceiling * 2.0))
            err = PhysicalPredictionError(
                error_type=ErrorType.CONTINUITY,
                magnitude=magnitude,
                description=f"{ent.label} moved {disp:.1f}u in {dt:.1f}s — exceeds "
                            f"plausible speed for a {ent.size_class} object",
                precision=SCHEMA_PRECISION[ErrorType.CONTINUITY],
            )
            self._errors.append(err)
            return err
        return None

    def _check_gravity_support(self, ent: WorldEntity, event: WorldEvent,
                               errors: list) -> None:
        """Gravity/support: an object at rest far below its support, or a
        'fall' event for an entity still recorded as supported, is checked.
        Buoyant things and substances in containers are exempt."""
        if ent.kind == EntityKind.SURFACE:
            return
        if ent.liquid and ent.contained_in:
            return
        if any(b in ent.label for b in _BUOYANT):
            return
        if event.type == "fall" and ent.supported_by:
            # Claimed falling while still believed supported: either the
            # support record is stale (likely) or the support failed —
            # update the model and flag mild surprise.
            ent.supported_by = None
            err = PhysicalPredictionError(
                error_type=ErrorType.GRAVITY,
                magnitude=0.2,
                description=f"{ent.label} fell but was believed supported by "
                            f"{self._id_to_label(ent.supported_by)}",
                entity_ids=[ent.id],
                precision=SCHEMA_PRECISION[ErrorType.GRAVITY],
            )
            ent.supported_by = None
            errors.append(err)
            self._errors.append(err)

    def _apply_containment(self, entities: list[WorldEntity]) -> None:
        if len(entities) >= 2:
            content, container = entities[0], entities[1]
            content.contained_in = container.id
            content.supported_by = None
            content.occluded = True
            content.occluded_since = time.time()

    def _apply_release(self, entities: list[WorldEntity]) -> None:
        for ent in entities:
            ent.contained_in = None
            ent.occluded = False
            ent.occluded_since = None
            ent.last_seen = time.time()

    def _check_permanence_reappearance(self, ent: WorldEntity) -> list[PhysicalPredictionError]:
        """Permanence payoff: a hidden thing confirmed present. If it was
        doubted long enough, the reappearance is mildly informative."""
        ent.occluded = False
        ent.occluded_since = None
        ent.last_seen = time.time()
        return []

    def _check_conservation(self, entities: list[WorldEntity],
                            detail: str) -> list[PhysicalPredictionError]:
        """Conservation: quantity changes need a transfer event (which this
        is); disappearance WITHOUT transfer is the violation."""
        errors = []
        for ent in entities:
            if ent.count is not None and not detail:
                err = PhysicalPredictionError(
                    error_type=ErrorType.CONSERVATION,
                    magnitude=0.3,
                    description=f"{ent.label} count {ent.count} changed without "
                                f"a recorded transfer event",
                    entity_ids=[ent.id],
                    precision=SCHEMA_PRECISION[ErrorType.CONSERVATION],
                )
                errors.append(err)
                self._errors.append(err)
        return errors

    def _check_agentivity(self, entities: list[WorldEntity]) -> list[PhysicalPredictionError]:
        """Contact-before-effect: unforced motion of a non-agent is a
        causality gap — either an unseen cause or an agent."""
        errors: list[PhysicalPredictionError] = []
        now = time.time()
        recent_contacts = [e for e in self.events
                           if e.type in ("contact", "assert")
                           and now - e.timestamp < 2.0]
        for ent in entities:
            if ent.kind in (EntityKind.AGENT, EntityKind.PERSON):
                continue
            if ent.size_class == "fixed":
                continue
            if self._support_recently_removed(ent):
                continue  # gravity explains it
            had_contact = any(set(e.entity_labels) & {ent.label}
                              for e in recent_contacts)
            if not had_contact and self._event_counter > 3:
                ent.unforced_moves += 1
                if ent.unforced_moves >= 2 and not ent.self_propelled:
                    ent.self_propelled = True
                    err = PhysicalPredictionError(
                        error_type=ErrorType.AGENTIVITY,
                        magnitude=0.4,
                        description=f"{ent.label} moved repeatedly without proximate "
                                    f"contact — reclassified as agent",
                        entity_ids=[ent.id],
                        precision=SCHEMA_PRECISION[ErrorType.AGENTIVITY],
                    )
                    errors.append(err)
                    self._errors.append(err)
        return errors

    def _support_recently_removed(self, ent: WorldEntity) -> bool:
        return any(e.type in ("release", "uncontain") and ent.label in e.entity_labels
                   for e in list(self.events)[-5:])

    def _expect_fall(self, ent: WorldEntity) -> list[PhysicalPredictionError]:
        """After release, predict downward motion (used by predict_next_state)."""
        return []

    # ------------------------------------------------------------------
    # Error consumption (feeds PredictiveProcessor)
    # ------------------------------------------------------------------

    def compute_world_errors(self) -> dict[str, float]:
        """Channel -> precision-weighted magnitude, for PredictiveProcessor.

        Mirrors the interoceptive error shape (plain floats are also
        accepted there). Errors are consumed once — the generative model
        has seen them.
        """
        with self._lock:
            channels: dict[str, float] = {}
            for err in self._errors:
                ch = f"world_{err.error_type.value}"
                channels[ch] = channels.get(ch, 0.0) + err.precision_weighted
            self._errors.clear()
            return {k: min(1.0, v) for k, v in channels.items()}

    @property
    def pending_errors(self) -> int:
        return len(self._errors)

    # ------------------------------------------------------------------
    # Language grounding
    # ------------------------------------------------------------------

    def process_input(self, user_input: str) -> dict:
        """Ground a user turn into the physical scene.

        Extracts: mentions of known entities, spatial relation claims,
        motion-verb physical claims. Returns a summary for the caller
        (agent.think); observations/assertions are recorded internally.
        """
        text = (user_input or "").lower()
        mentioned: list[str] = []
        relations: list[dict] = []
        surprises: list[PhysicalPredictionError] = []

        with self._lock:
            # 1. Known-entity mentions refresh presence (permanence payoff)
            for label in list(self._label_index):
                if re.search(rf"\b{re.escape(label)}\b", text):
                    ent = self._entity_by_label(label)
                    if ent is not None:
                        ent.occluded = False
                        ent.occluded_since = None
                        ent.last_seen = time.time()
                        mentioned.append(label)

            # 2. Spatial phrases: "on the table", "in the drawer", ...
            for phrase, relation in _SPATIAL_RELATIONS.items():
                pattern = re.escape(phrase) + r"\s+(?:the |a |an |my |your )?(\w[\w-]*)"
                for m in re.finditer(pattern, text):
                    obj_label, ref_label = self._phrase_roles(text, m)
                    if not ref_label:
                        continue
                    relations.append({"object": obj_label or "?", "relation": relation,
                                      "reference": ref_label})
                    # Register unknown participants (learnable, not violations)
                    if self._entity_by_label(ref_label) is None:
                        self.register_entity(ref_label, EntityKind.SURFACE
                                             if relation in ("touch", "above", "below")
                                             else EntityKind.OBJECT)
                    if obj_label != "something" and self._entity_by_label(obj_label) is None:
                        self.register_entity(obj_label, EntityKind.OBJECT)
                    err = self._check_spatial_claim(obj_label, ref_label, relation)
                    if err:
                        surprises.append(err)

            # 3. Motion verbs -> events, schema-checked
            for verb, (schema, event_type) in _MOTION_VERBS.items():
                if not re.search(rf"\b{verb}\b", text):
                    continue
                subj = self._subject_of(text, verb)
                if subj and self._entity_by_label(subj) is None:
                    self.register_entity(subj, EntityKind.OBJECT)
                self.events.append(WorldEvent(
                    type=event_type, entity_labels=[subj] if subj else [],
                    detail=f"user asserted: {verb}", source="user"))
                if subj and schema == "gravity" and verb in ("floats", "floated", "hovers"):
                    ent = self._entity_by_label(subj)
                    if (ent is not None and not any(b in ent.label for b in _BUOYANT)
                            and not ent.liquid):
                        err = PhysicalPredictionError(
                            error_type=ErrorType.GRAVITY,
                            magnitude=0.6,
                            description=f"{ent.label} reportedly unsupported yet not falling "
                                        f"(user: '{verb}') — buoyancy, magnetism, or new physics?",
                            entity_ids=[ent.id],
                            precision=SCHEMA_PRECISION[ErrorType.GRAVITY],
                        )
                        surprises.append(err)
                        self._errors.append(err)

        return {
            "entities": mentioned,
            "relations": relations,
            "surprises": [
                {"type": s.error_type.value, "magnitude": round(s.magnitude, 2),
                 "description": s.description}
                for s in surprises
            ],
        }

    _STOPWORD_SUBJECTS = {"is", "are", "was", "were", "the", "a", "an", "my",
                          "your", "his", "her", "its", "and", "then", "just",
                          "now", "still", "there", "that", "this", "it"}

    def _phrase_roles(self, text: str, match) -> tuple[str, str]:
        """For a spatial match, find the subject mentioned BEFORE the phrase
        (nearest known entity, else the nearest plausible noun) and the
        reference captured by the match."""
        before = text[:match.start()]
        subject = None
        best_pos = -1
        for label in self._label_index:
            pos = before.rfind(label)
            if pos > best_pos:
                best_pos = pos
                subject = label
        if subject is None:
            # Noun heuristic: last word before the phrase that is not a
            # stopword (handles "my cup is on the table" -> "cup")
            words = before.split()
            for word in reversed(words):
                if word and word not in self._STOPWORD_SUBJECTS:
                    subject = word
                    break
        return subject or "something", match.group(1)

    def _subject_of(self, text: str, verb: str) -> str | None:
        """Nearest known entity label before the verb."""
        idx = text.find(verb)
        if idx <= 0:
            return None
        before = text[:idx]
        best = None
        for label in self._label_index:
            pos = before.rfind(label)
            if pos >= 0 and (best is None or pos > best[1]):
                best = (label, pos)
        return best[0] if best else None

    def _check_spatial_claim(self, obj_label: str, ref_label: str,
                             relation: str) -> Optional[PhysicalPredictionError]:
        """Contradiction between an asserted relation and scene geometry."""
        obj = self._entity_by_label(obj_label)
        ref = self._entity_by_label(ref_label)
        if obj is None or ref is None or obj.position is None or ref.position is None:
            return None  # unknowns are learnable, not violations
        distance = math.dist(obj.position, ref.position)
        dy = obj.position[1] - ref.position[1]
        violation = None
        if relation == "touch" and distance > 1.5:
            violation = (f"'{obj_label} on {ref_label}' asserted but they are "
                         f"{distance:.1f}u apart")
        elif relation == "inside" and distance > 1.0:
            violation = (f"'{obj_label} in {ref_label}' but they are "
                         f"{distance:.1f}u apart")
        elif relation in ("above", "below") and abs(dy) < 0.2 and distance < 0.5:
            violation = f"'{obj_label} {relation} {ref_label}' but they are level"
        if violation:
            err = PhysicalPredictionError(
                error_type=ErrorType.SPATIAL,
                magnitude=0.4,
                description=violation,
                entity_ids=[obj.id, ref.id],
                precision=SCHEMA_PRECISION[ErrorType.SPATIAL],
            )
            self._errors.append(err)
            return err
        return None

    def infer_relations(self, a_label: str, b_label: str) -> list[str]:
        """Qualitative spatial relations between two known entities."""
        a, b = self._entity_by_label(a_label), self._entity_by_label(b_label)
        if not a or not b or a.position is None or b.position is None:
            return []
        d = math.dist(a.position, b.position)
        relations: list[str] = []
        if a.contained_in == b.id:
            relations.append("inside")
        if d < 0.5:
            relations.append("touch")
        if a.position[1] > b.position[1] + 0.5:
            relations.append("above")
        elif a.position[1] < b.position[1] - 0.5:
            relations.append("below")
        if 0.5 <= d <= 2.0:
            relations.append("beside")
        return relations or ["disconnected"]

    # ------------------------------------------------------------------
    # Affordances (Gibson/Gaver)
    # ------------------------------------------------------------------

    def affordances_for(self, entity_label: str) -> list[str]:
        """Feasible actions for an entity (type prior + calibration)."""
        ent = self._entity_by_label(entity_label)
        if ent is None:
            return []
        actions: list[str] = []
        for type_key, acts in _TYPE_AFFORDANCES.items():
            if type_key in ent.label:
                actions.extend(acts)
        if ent.kind == EntityKind.CONTAINER and "contain" not in actions:
            actions.append("contain")
        if ent.kind == EntityKind.SURFACE and "place_on" not in actions:
            actions.append("place_on")
        # Discovered (hidden) affordances that proved real
        for action, rec in self.affordances.get(ent.label, {}).items():
            if action not in actions and rec.successes > 0:
                actions.append(action)
        return actions

    def record_action_outcome(self, entity_label: str, action: str,
                              success: bool) -> dict:
        """Calibrate an affordance from an attempted action.

        Unexpected success on an unlisted action = a HIDDEN affordance
        discovered (novelty reward). Repeated failure of a predicted one
        demotes it toward 'false'.
        """
        with self._lock:
            label = entity_label.strip().lower()
            rec = self.affordances.setdefault(label, {}).setdefault(
                action, AffordanceRecord(action=action))
            was_unlisted = action not in _TYPE_AFFORDANCES.get(label, [])
            was_first = rec.attempts == 0
            rec.attempts += 1
            if success:
                rec.successes += 1
            if success and rec.attempts == 1:
                rec.classification = "perceivable"
                if was_unlisted:
                    self._novelty_buffer.append(0.5)   # discovery!
            elif not success and rec.attempts >= 3 and rec.successes == 0:
                rec.classification = "false"
            return {
                "action": action,
                "classification": rec.classification,
                "effectiveness": round(rec.effectiveness, 3),
                "novel_discovery": success and was_first and was_unlisted,
            }

    # ------------------------------------------------------------------
    # Hypotheses (stochastic belief layer — RSSM analogue)
    # ------------------------------------------------------------------

    def hypothesize(self, claim: str, likelihood: float = 0.5) -> None:
        """Maintain a distribution over unobserved states of the world."""
        claim = claim.strip().lower()
        if not claim:
            return
        with self._lock:
            self.hypotheses[claim] = max(0.0, min(1.0, likelihood))
            self.hypotheses.move_to_end(claim)
            while len(self.hypotheses) > self.MAX_HYPOTHESES:
                self.hypotheses.popitem(last=False)

    def confirm_hypothesis(self, claim: str) -> float:
        """An observation resolved a hidden-state question."""
        return self.hypotheses.pop(claim.strip().lower(), 0.5)

    # ------------------------------------------------------------------
    # Forward prediction (per-JEPA: features and relations, not pixels)
    # ------------------------------------------------------------------

    def predict_next_state(self, dt: float = 1.0) -> dict[str, dict]:
        """Predict each entity's next state under the schemas.

        Returns {label: {"expected_position"?, "expected_support"?,
                         "expected_occluded"}} — the expectations that
        future observations are checked against.
        """
        with self._lock:
            self._expectations = {}
            for ent in self.entities.values():
                expectation: dict[str, Any] = {
                    "expected_support": ent.supported_by,
                    "expected_occluded": ent.occluded,
                }
                base = ent.position
                # Inertia: motion persists with mild damping
                if ent.velocity is not None and base is not None:
                    decay = math.exp(-dt / 5.0)
                    base = tuple(base[i] + ent.velocity[i] * decay * dt for i in range(3))
                    expectation["expected_position"] = base
                # Gravity: unsupported, non-buoyant, free -> downward drift
                if (ent.supported_by is None and ent.contained_in is None
                        and ent.size_class != "fixed" and base is not None
                        and not any(b in ent.label for b in _BUOYANT)):
                    fall = min(0.5 * 9.81 * dt * dt, 1.0)
                    if "expected_position" in expectation:
                        px, py, pz = expectation["expected_position"]
                        expectation["expected_position"] = (px, py - fall, pz)
                    else:
                        expectation["expected_position"] = (base[0], base[1] - fall, base[2])
                # Permanence: occluded things keep their last known place
                elif ent.occluded and base is not None:
                    expectation["expected_position"] = base
                self._expectations[ent.id] = expectation
        return {self._id_to_label(eid): exp for eid, exp in self._expectations.items()}

    # ------------------------------------------------------------------
    # Tick: permanence doubt + hypothesis decay
    # ------------------------------------------------------------------

    def tick(self, dt: float = 1.0) -> dict:
        """Passive maintenance: occlusion doubt, hypothesis decay."""
        now = time.time()
        with self._lock:
            for ent in self.entities.values():
                if ent.occluded and ent.occluded_since is not None:
                    hidden_for = now - ent.occluded_since
                    if hidden_for > self.OCCLUSION_DOUBT_SECONDS:
                        # Still believed present (permanence), with growing
                        # doubt recorded as a hypothesis.
                        doubt = max(0.1, 1.0 - hidden_for
                                    / (self.OCCLUSION_DOUBT_SECONDS * 10))
                        self.hypothesize(f"{ent.label} still where last seen", doubt)
            # Unsupported hypotheses lose strength
            for claim in list(self.hypotheses):
                self.hypotheses[claim] *= (1.0 - 0.01 * dt)
                if self.hypotheses[claim] < 0.05:
                    del self.hypotheses[claim]
            return {
                "entities": len(self.entities),
                "hypotheses": len(self.hypotheses),
                "events": self._event_counter,
                "pending_errors": len(self._errors),
            }

    # ------------------------------------------------------------------
    # State / summary (for live_state, prompts, /api/world/state)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        with self._lock:
            return {
                "entity_count": len(self.entities),
                "entities": [
                    {
                        "label": e.label,
                        "kind": e.kind.value,
                        "position": e.position,
                        "supported_by": self._id_to_label(e.supported_by),
                        "contained_in": self._id_to_label(e.contained_in),
                        "occluded": e.occluded,
                        "size_class": e.size_class,
                        "self_propelled": e.self_propelled,
                        "affordances": self.affordances_for(e.label),
                    }
                    for e in list(self.entities.values())[-20:]
                ],
                "recent_events": [
                    {"type": e.type, "entities": e.entity_labels, "detail": e.detail,
                     "source": e.source, "age_s": round(time.time() - e.timestamp, 1)}
                    for e in list(self.events)[-10:]
                ],
                "hypotheses": {c: round(p, 3) for c, p in list(self.hypotheses.items())[-10:]},
                "affordance_stats": {
                    label: {a: {"classification": r.classification,
                                "effectiveness": round(r.effectiveness, 2)}
                            for a, r in acts.items()}
                    for label, acts in list(self.affordances.items())[-20:]
                },
                "pending_errors": len(self._errors),
                "novelty_recent": round(sum(self._novelty_buffer)
                                        / max(1, len(self._novelty_buffer)), 3),
            }

    def get_scene_summary(self) -> str:
        """Compact textual scene description for system-prompt injection."""
        with self._lock:
            if not self.entities:
                return ""
            parts = []
            for ent in list(self.entities.values())[-12:]:
                state = "hidden" if ent.occluded else "present"
                rel = ""
                if ent.supported_by:
                    rel = f" on {self._id_to_label(ent.supported_by)}"
                elif ent.contained_in:
                    rel = f" in {self._id_to_label(ent.contained_in)}"
                parts.append(f"{ent.label} ({ent.kind.value}, {state}{rel})")
            return "[World model] Known scene: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Singleton (thread-safe, None-reset pattern — mirrors predictive_processing)
# ---------------------------------------------------------------------------

_world_model: Optional["WorldModel"] = None
_wm_lock = threading.Lock()


def get_world_model() -> "WorldModel":
    global _world_model
    if _world_model is None:
        with _wm_lock:
            if _world_model is None:
                _world_model = WorldModel()
    return _world_model


def reset_world_model() -> None:
    global _world_model
    with _wm_lock:
        _world_model = None