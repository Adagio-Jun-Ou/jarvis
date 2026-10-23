"""Regression tests for the physical world model and its integration.

Research basis (2026-09 pass): "GEN1.5" maps to the physical-AI
world-model class (NVIDIA Cosmos WFM, arXiv:2501.03575; DeepMind Genie).
Jarvis adopts the architecture those systems share with developmental
psychology: persistent world belief + forward prediction + prediction
error as the learning signal, seeded with Spelke's core knowledge.

Covered here:
- each Spelke schema's violation detection (permanence, continuity,
  gravity/support incl. buoyancy exemption, containment, conservation,
  causality/agentivity)
- affordance maps (Gibson) with Gaver's perceivable/hidden/false
  classification and hidden-affordance discovery (novelty)
- qualitative spatial relations + contradiction detection
- language grounding (spatial phrases, motion verbs, float-claims)
- hypotheses (stochastic belief layer, RSSM analogue) with decay
- PredictiveProcessor.process_world_model_errors (mirror of the
  interoceptive integration: beliefs + curiosity boost)
- agent wiring (init block, getter, think() slot, turn-complete store)
- background world_model_tick registration
"""
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.world_model import get_world_model, reset_world_model  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_world_model():
    reset_world_model()
    yield
    reset_world_model()


class TestSingleton:
    def test_singleton_identity_and_reset(self):
        from core.world_model import get_world_model, reset_world_model

        wm1 = get_world_model()
        wm2 = get_world_model()
        assert wm1 is wm2
        reset_world_model()
        assert get_world_model() is not wm1


class TestEntityManagement:
    def test_register_and_merge_by_label(self):
        from core.world_model import get_world_model, EntityKind

        wm = get_world_model()
        a = wm.register_entity("cup", EntityKind.OBJECT, position=(0, 0, 0))
        b = wm.register_entity("CUP", EntityKind.OBJECT, position=(1, 1, 1))
        assert a.id == b.id, "case-insensitive labels must merge"
        assert b.position == (1.0, 1.0, 1.0)

    def test_entity_bound_evicts_least_recently_seen(self):
        from core.world_model import get_world_model

        wm = get_world_model()
        wm.MAX_ENTITIES = 5
        for i in range(7):
            wm.register_entity(f"thing{i}", position=(i, 0, 0))
        state = wm.get_state()
        assert state["entity_count"] <= 5

    def test_unknown_kind_falls_back_to_object(self):
        from core.world_model import get_world_model

        wm = get_world_model()
        ent = wm.register_entity("thing", "nonsense-kind")
        assert ent.kind.value == "object"


class TestSchemaErrors:
    def test_continuity_teleport_violation(self):
        wm = get_world_model()
        wm.register_entity("ball", position=(1, 1, 0), size_class="medium")
        wm._event_counter = 10
        time.sleep(0.01)
        errors = wm.observe_event("move", ["ball"], positions={"ball": (80, 80, 0)},
                                  source="perception")
        types = [e.error_type.value for e in errors]
        assert "continuity" in types

    def test_continuity_normal_move_no_error(self):
        wm = get_world_model()
        wm.register_entity("ball", position=(1, 1, 0), size_class="small")
        time.sleep(0.01)
        errors = wm.observe_event("move", ["ball"], positions={"ball": (1.1, 1.0, 0)},
                                  source="perception")
        assert errors == []

    def test_buoyant_float_not_a_violation(self):
        """Boats float — the buoyancy exemption must suppress the gravity
        surprise for buoyant things."""
        wm = get_world_model()
        result = wm.process_input("the boat floated gently on the lake")
        assert result["surprises"] == []

    def test_nonbuoyant_float_is_gravity_surprise(self):
        wm = get_world_model()
        wm.register_entity("anvil", position=(0, 0, 0))
        result = wm.process_input("the anvil floated up into the sky")
        assert any(s["type"] == "gravity" and s["magnitude"] > 0.4
                   for s in result["surprises"])

    def test_agentivity_upgrade_after_unforced_moves(self):
        wm = get_world_model()
        wm.register_entity("box", position=(0, 0, 0), size_class="medium")
        wm._event_counter = 10  # established scene
        agent_errors = []
        for i in range(3):
            errs = wm.observe_event("move", ["box"],
                                    positions={"box": (1 + i * 0.1, 0, 0)},
                                    source="perception")
            agent_errors = [e for e in errs if e.error_type.value == "agentivity"]
            if agent_errors:
                break
        assert wm._entity_by_label("box").self_propelled is True

    def test_containment_hides_and_co_moves(self):
        wm = get_world_model()
        wm.register_entity("drawer", kind="container", position=(0, 0, 0))
        wm.register_entity("key", position=(0.1, 0.1, 0))
        wm.observe_event("contain", ["key", "drawer"])
        key = wm._entity_by_label("key")
        assert key.contained_in == wm._entity_by_label("drawer").id
        assert key.occluded is True
        # release restores visibility
        wm.observe_event("release", ["key"])
        assert key.occluded is False

    def test_disappear_then_mention_confirms_permanence(self):
        wm = get_world_model()
        wm.register_entity("phone")
        wm.observe_event("disappear", ["phone"])
        assert wm._entity_by_label("phone").occluded is True
        # mentioning it confirms presence (permanence payoff, no error)
        result = wm.process_input("where is my phone?")
        assert wm._entity_by_label("phone").occluded is False

    def test_error_consumed_once(self):
        wm = get_world_model()
        wm.register_entity("ball", position=(1, 1, 0))
        time.sleep(0.01)
        wm.observe_event("move", ["ball"], positions={"ball": (99, 99, 0)},
                         source="perception")
        first = wm.compute_world_errors()
        assert first  # something reported
        second = wm.compute_world_errors()
        assert second == {}, "errors must be consumed exactly once"


class TestAffordances:
    def test_type_prior_affordances(self):
        wm = get_world_model()
        wm.register_entity("cup")
        acts = wm.affordances_for("cup")
        assert "grasp" in acts and "contain" in acts and "pour_from" in acts

    def test_unknown_entity_no_affordances(self):
        wm = get_world_model()
        assert wm.affordances_for("quantum-flux-capacitor") == []

    def test_hidden_affordance_discovery_is_novel(self):
        wm = get_world_model()
        # 'prop_open' is not a listed cup affordance
        result = wm.record_action_outcome("cup", "prop_open", True)
        assert result["novel_discovery"] is True
        assert wm.get_state()["novelty_recent"] > 0

    def test_repeated_failure_demotes_to_false(self):
        wm = get_world_model()
        for _ in range(3):
            outcome = wm.record_action_outcome("cup", "pour_from", False)
        assert outcome["classification"] == "false"

    def test_discovered_affordance_becomes_listed(self):
        wm = get_world_model()
        wm.register_entity("cup")
        wm.record_action_outcome("cup", "hammer_with", True)
        assert "hammer_with" in wm.affordances_for("cup")


class TestSpatial:
    def test_grounding_extracts_object_and_reference(self):
        wm = get_world_model()
        result = wm.process_input("my cup is on the table")
        assert result["relations"] == [
            {"object": "cup", "relation": "touch", "reference": "table"}]
        assert "cup" in wm._label_index and "table" in wm._label_index

    def test_relation_contradiction_is_spatial_error(self):
        wm = get_world_model()
        wm.register_entity("cup", position=(0, 0, 0))
        wm.register_entity("table", position=(20, 20, 0), kind="surface")
        result = wm.process_input("the cup is on the table")
        assert any(s["type"] == "spatial" for s in result["surprises"])

    def test_consistent_claim_no_surprise(self):
        wm = get_world_model()
        wm.register_entity("cup", position=(0, 0, 0))
        wm.register_entity("table", position=(0.1, 0.1, 0), kind="surface")
        result = wm.process_input("the cup is on the table")
        assert result["surprises"] == []

    def test_infer_relations(self):
        wm = get_world_model()
        wm.register_entity("cup", position=(0, 0, 0))
        wm.register_entity("shelf", position=(0, 3, 0), kind="surface")
        rels = wm.infer_relations("cup", "shelf")
        assert "below" in rels


class TestHypotheses:
    def test_hypothesize_and_confirm(self):
        wm = get_world_model()
        wm.hypothesize("the key is in the drawer", 0.7)
        assert wm.hypotheses["the key is in the drawer"] == pytest.approx(0.7)
        prior = wm.confirm_hypothesis("the key is in the drawer")
        assert prior == pytest.approx(0.7)
        assert "the key is in the drawer" not in wm.hypotheses

    def test_likelihood_clamped(self):
        wm = get_world_model()
        wm.hypothesize("claim", 5.0)
        assert wm.hypotheses["claim"] == 1.0

    def test_tick_decays_unsupported_hypotheses(self):
        wm = get_world_model()
        wm.hypothesize("ghost in the garage", 0.5)
        for _ in range(20):
            wm.tick(dt=100.0)
        assert "ghost in the garage" not in wm.hypotheses

    def test_max_hypotheses_bounded(self):
        wm = get_world_model()
        for i in range(150):
            wm.hypothesize(f"hypothesis number {i}")
        assert len(wm.hypotheses) <= 100


class TestForwardPrediction:
    def test_unsupported_object_expected_to_fall(self):
        wm = get_world_model()
        wm.register_entity("ball", position=(0, 10, 0), size_class="small")
        wm.observe_event("release", ["ball"], source="perception")
        expectations = wm.predict_next_state(dt=1.0)
        exp = expectations["ball"]
        assert exp["expected_position"][1] < 10.0, "gravity: should expect downward drift"
        assert exp["expected_support"] is None

    def test_supported_object_expects_support(self):
        wm = get_world_model()
        table = wm.register_entity("table", kind="surface", position=(0, 0, 0))
        cup = wm.register_entity("cup", position=(0.1, 0.1, 0))
        cup.supported_by = table.id
        expectations = wm.predict_next_state(dt=1.0)
        exp = expectations["cup"]
        assert exp["expected_support"] == table.id

    def test_buoyant_no_fall_expectation(self):
        wm = get_world_model()
        wm.register_entity("boat", position=(0, 0, 0))
        wm.observe_event("release", ["boat"], source="perception")
        expectations = wm.predict_next_state(dt=1.0)
        exp = expectations["boat"]
        # Buoyant: no downward drift is expected (either no expectation at
        # all, or one that stays level)
        assert 'expected_position' not in exp or exp["expected_position"][1] == 0.0


class TestPredictiveIntegration:
    def test_process_world_model_errors_basic(self):
        from core.predictive_processing import PredictiveProcessor

        pp = PredictiveProcessor()
        result = pp.process_world_model_errors(
            {"world_gravity": 0.6, "world_continuity": 0.1})
        assert result["integrated_errors"] == 2
        assert 0.3 < result["avg_error"] < 0.45
        assert result["world_beliefs_updated"] is True

    def test_physical_surprise_boosts_curiosity(self):
        from core.predictive_processing import PredictiveProcessor

        pp = PredictiveProcessor()
        before = pp.drives.curiosity_level
        pp.process_world_model_errors({"world_gravity": 0.9})
        assert pp.drives.curiosity_level > 0.5

    def test_stable_world_no_boost(self):
        from core.predictive_processing import PredictiveProcessor

        pp = PredictiveProcessor()
        result = pp.process_world_model_errors({"world_gravity": 0.02})
        assert result["avg_error"] < 0.05

    def test_error_object_shape_compatible(self):
        """PhysicalPredictionError exposes the interoceptive shape, so a
        mixed-error feed works."""
        from core.predictive_processing import PredictiveProcessor
        from core.world_model import ErrorType, PhysicalPredictionError

        pp = PredictiveProcessor()
        obj = SimpleNamespace(precision_weighted=0.5, precision=0.8, error_magnitude=0.6)
        result = pp.process_world_model_errors({"world_gravity": obj})
        assert result["integrated_errors"] == 1

    def test_full_pipeline_world_to_processor(self):
        """The agent wiring's exact call sequence: surprises -> errors ->
        predictive processor with scene state."""
        from core.predictive_processing import PredictiveProcessor

        wm = get_world_model()
        wm.register_entity("anvil", position=(0, 0, 0))
        wm.process_input("the anvil floated up into the sky")
        pp = PredictiveProcessor()
        world_errors = wm.compute_world_errors()
        result = pp.process_world_model_errors(world_errors, scene_state=wm.get_state())
        assert result["world_beliefs_updated"] is True
        assert result["avg_error"] > 0.2

    def test_scene_summary_injection_shape(self):
        wm = get_world_model()
        wm.register_entity("cup", position=(0.1, 0.1, 0))
        summary = wm.get_scene_summary()
        assert summary.startswith("[World model]")
        assert "cup" in summary


class TestTickAndState:
    def test_tick_reports_state(self):
        wm = get_world_model()
        wm.register_entity("cup")
        wm.register_entity("box", position=(0, 0, 0))
        state = wm.tick(dt=1.0)
        assert state["entities"] == 2
        assert "hypotheses" in state

    def test_occlusion_doubt_after_long_hide(self):
        wm = get_world_model()
        wm.register_entity("sock")
        ent = wm._entity_by_label("sock")
        ent.occluded = True
        ent.occluded_since = time.time() - 300.0  # 5 min hidden
        wm.tick(dt=1.0)
        assert any("sock" in c for c in wm.hypotheses)

    def test_get_state_shape(self):
        wm = get_world_model()
        wm.register_entity("cup", position=(0, 0, 0))
        state = wm.get_state()
        for key in ("entity_count", "entities", "recent_events",
                    "hypotheses", "affordance_stats", "pending_errors",
                    "novelty_recent"):
            assert key in state
        assert state["entities"][0]["label"] == "cup"

    def test_scene_summary_empty_world(self):
        wm = get_world_model()
        assert wm.get_scene_summary() == ""


class TestAgentWiring:
    def test_agent_init_block_registered(self):
        """Structural pin: the agent wires the world model like the other
        subsystems (attribute + init block + live_state getter)."""
        import inspect

        from core.agent import JarvisAgent

        init_src = inspect.getsource(JarvisAgent._init_live_state)
        assert "get_world_model()" in init_src
        assert 'register_subsystem_getter("world_model"' in init_src
        assert hasattr(JarvisAgent, "_get_world_model_state")

    def test_think_calls_world_model(self):
        import inspect

        from core.agent import JarvisAgent

        think_src = inspect.getsource(JarvisAgent.think)
        assert "self._world_model.process_input(" in think_src
        assert "compute_world_errors" in think_src
        assert "process_world_model_errors" in think_src


class TestBackgroundTick:
    def test_world_model_tick_registered(self):
        """Source pin: the background subsystem registers a world-model
        tick alongside bio_tick."""
        import inspect

        import core.background as bg

        src = inspect.getsource(bg)
        assert "async def world_model_tick():" in src
        assert '"world_model_tick"' in src
        assert "get_world_model().tick(" in src
