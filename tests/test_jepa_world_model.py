"""Tests for the JEPA world model over terminal sessions.

Ported from the mind-diffusion project (mind_jepa_bridge.py): a
latent-only world model (EMA target + GRU context + action-conditioned
predictor, VICReg-lite anti-collapse) trained on Jarvis's own shell
history. Prediction errors feed PredictiveProcessor as the learned
"jepa_terminal" channel.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch required for JEPA tests")

from core.jepa_world_model import (  # noqa: E402
    JepaWorldModel, MindJEPA, hash_embed, session_to_frames,
    get_jepa_world_model, reset_jepa_world_model,
)


class TestHashEmbed:
    def test_deterministic(self):
        a = hash_embed("git status now", 64)
        b = hash_embed("git status now", 64)
        assert np.allclose(a, b)

    def test_normalized_or_zero(self):
        v = hash_embed("ls -la /tmp", 64)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5
        empty = hash_embed("", 64)
        assert np.linalg.norm(empty) == 0.0

    def test_different_texts_differ(self):
        a = hash_embed("git status", 64)
        b = hash_embed("rm -rf /", 64)
        assert not np.allclose(a, b)


class TestFrames:
    def test_shapes(self):
        turns = [("ls -la", "total 4"), ("git status", "On branch main")]
        obs, act = session_to_frames(turns)
        assert obs.shape == (2, 64)
        assert act.shape == (2, 32)
        assert obs.dtype == np.float32 or isinstance(obs, torch.Tensor)

    def test_empty_turns(self):
        obs, act = session_to_frames([])
        assert obs.shape[0] == 0
        assert act.shape[0] == 0

    def test_output_leaks_context(self):
        """obs blends output (1.0) with command (0.2) — the command
        partially conditions the observation, per mind-diffusion."""
        turns = [("unique_command_xyz", "output text")]
        obs, _act = session_to_frames(turns)
        pure_output = hash_embed("output text", 64)
        pure_cmd = hash_embed("unique_command_xyz", 64)
        expected = pure_output + 0.2 * pure_cmd
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(obs[0].numpy(), expected, atol=1e-5)


class TestMindJEPA:
    def test_training_reduces_loss(self):
        """On a synthetic world with learnable structure (output = echo
        of the command tokens), the loss must decrease."""
        torch.manual_seed(0)
        model = MindJEPA()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=3e-3)

        def make_batch():
            cmds = [f"command_{i % 5} alpha beta" for i in range(12)]
            outs = [c.replace("command", "result") for c in cmds]
            obs, act = session_to_frames(list(zip(cmds, outs)))
            return obs.unsqueeze(0), act.unsqueeze(0)

        obs, act = make_batch()
        model.train()
        first_loss = None
        last_loss = None
        for step in range(40):
            opt.zero_grad()
            loss, _, _ = model.training_step(obs, act)
            loss.backward()
            opt.step()
            if step == 0:
                first_loss = float(loss.detach())
            last_loss = float(loss.detach())
        assert last_loss < first_loss, (
            f"loss did not decrease: {first_loss} -> {last_loss}")

    def test_roll_shapes(self):
        torch.manual_seed(0)
        model = MindJEPA()
        model.eval()
        obs, _ = session_to_frames([("a", "b"), ("c", "d"), ("e", "f")])
        h_n, z_last = model.encode_last(obs.unsqueeze(0))
        # z_last is (B, zdim); roll expects z0 (B, zdim), actions (B, S, act)
        _o, acts = session_to_frames([("g", "h")])
        rolled = model.roll(z_last, acts.unsqueeze(0))
        assert rolled.shape == (1, 1, 64)

    def test_vicreg_finite(self):
        model = MindJEPA()
        obs = torch.randn(2, 4, 64)
        act = torch.randn(2, 4, 32)
        loss, pred, reg = model.training_step(obs, act)
        assert torch.isfinite(loss)
        assert pred >= 0.0 and reg >= 0.0


class TestJepaWorldModel:
    def _model(self, tmp_path):
        return JepaWorldModel(persistence_path=tmp_path / "model.pt")

    def test_too_few_turns_returns_none(self, tmp_path):
        jwm = self._model(tmp_path)
        jwm.record_turn("ls", "file.txt")
        assert jwm.train_step() is None
        assert jwm.evaluate_prediction_error() is None

    def test_error_bounded_and_fresh(self, tmp_path):
        jwm = self._model(tmp_path)
        for i in range(10):
            jwm.record_turn(f"cmd_{i} alpha", f"result_{i} alpha beta")
        err = jwm.evaluate_prediction_error()
        assert err is not None
        assert 0.0 <= err <= 1.0
        assert jwm.get_last_error() == err
        # Nothing new -> None (no stale re-evaluation)
        assert jwm.evaluate_prediction_error() is None
        # New turn -> evaluable again
        jwm.record_turn("cmd_new alpha", "result_new alpha beta")
        assert jwm.evaluate_prediction_error() is not None

    def test_ingest_deduplicates(self, tmp_path):
        jwm = self._model(tmp_path)
        records = [
            {"pid": 1, "started_at": 100.0, "command": "ls -la",
             "stdout_tail": "total 4", "stderr_tail": ""},
            {"pid": 1, "started_at": 100.0, "command": "ls -la",
             "stdout_tail": "total 4", "stderr_tail": ""},
            {"pid": 2, "started_at": 200.0, "command": "git status",
             "stdout_tail": "On branch main", "stderr_tail": ""},
            {"pid": 3, "started_at": 300.0, "command": "cd /tmp",
             "stdout_tail": "", "stderr_tail": ""},  # no output: skipped
        ]
        first = jwm.ingest_context_commands(records)
        assert first == 2
        again = jwm.ingest_context_commands(records)
        assert again == 0

    def test_train_step_runs_and_tracks(self, tmp_path):
        jwm = self._model(tmp_path)
        for i in range(20):
            jwm.record_turn(f"command_{i} test", f"result_{i} test data")
        loss = jwm.train_step()
        assert loss is not None and loss >= 0.0
        state = jwm.get_state()
        assert state["train_count"] == 1
        assert len(state["loss_history"]) == 1

    def test_learns_temporal_structure(self, tmp_path):
        """The real behavioral contract: identical models trained on
        data WITH temporal structure (output echoes previous command)
        reach LOWER one-step-ahead error than on shuffled data with the
        same token distribution — the predictor learned dynamics, not
        just the VICReg equilibrium."""
        import random
        import statistics

        random.seed(0)

        def train_and_eval(turns, tmp_path, tag):
            jwm = JepaWorldModel(
                persistence_path=tmp_path / f"{tag}.pt")
            for c, o in turns:
                jwm.record_turn(c, o)
            errs = []
            for _step in range(120):
                jwm.train_step()
                e = jwm.evaluate_prediction_error(force=True)
                if e is not None:
                    errs.append(e)
            return statistics.median(errs[-30:])

        turns_a = [(f"cmd_{i % 6} shared prefix",
                    f"echo of cmd_{(i - 1) % 6} shared prefix")
                   for i in range(60)]
        outs = [o for _c, o in turns_a]
        shuffled = outs[:]
        random.shuffle(shuffled)
        turns_b = [(f"cmd_{i % 6} shared prefix", shuffled[i])
                   for i in range(60)]

        err_a = train_and_eval(turns_a, tmp_path, "learnable")
        err_b = train_and_eval(turns_b, tmp_path, "shuffled")
        assert err_a < err_b, (
            f"structured ({err_a:.3f}) must beat shuffled ({err_b:.3f})")

    def test_eval_matches_training_window(self, tmp_path):
        """Regression: evaluation must use a _TRAIN_WINDOW-length GRU
        context (matching training), not the full history — a long
        context is out-of-distribution and drove cosine error to 1.0."""
        jwm = self._model(tmp_path)
        for i in range(60):
            jwm.record_turn(f"cmd_{i % 6} shared prefix",
                            f"echo of cmd_{(i - 1) % 6} shared prefix")
        errs = []
        for _ in range(150):
            jwm.train_step()
            e = jwm.evaluate_prediction_error(force=True)
            if e is not None:
                errs.append(e)
        import statistics
        # The window-mismatch regression drove error to ~1.0 (orthogonal
        # prediction); a trained model on learnable dynamics sits well
        # below that even when undertrained.
        assert statistics.median(errs) < 0.8, (
            f"error stuck near 1.0 means window mismatch: {statistics.median(errs):.3f}")

    def test_tick_end_to_end(self, tmp_path, monkeypatch):
        """tick() harvests context commands, trains, evaluates, and the
        fresh error is feedable to PredictiveProcessor."""
        reset_jepa_world_model()
        jwm = self._model(tmp_path)

        class _FakeCtx:
            def get_recent_commands(self, limit=20):
                return [
                    {"pid": i, "started_at": float(i),
                     "command": f"command_{i}",
                     "stdout_tail": f"result_{i} output",
                     "stderr_tail": ""}
                    for i in range(12)
                ]

        import core.execution_context as ec
        monkeypatch.setattr(ec, "get_execution_context", lambda: _FakeCtx())
        # tick() imports inside the function from the same module path
        import core.jepa_world_model as jwm_mod
        original_persist = jwm._PERSIST_EVERY

        result = None
        monkeypatch.setattr(jwm_mod, "get_execution_context",
                            lambda: _FakeCtx(), raising=False)
        # tick() does `from core.execution_context import ...` — patch there
        monkeypatch.setattr(ec, "get_execution_context", lambda: _FakeCtx())
        result = jwm.tick()
        assert result["ingested"] == 12
        assert result["fresh_error"] is not None
        assert jwm.get_state()["turn_count"] == 12

        fed = jwm.feed_predictive_processor()
        assert fed is not None

    def test_persistence_roundtrip(self, tmp_path):
        jwm = self._model(tmp_path)
        for i in range(8):
            jwm.record_turn(f"command_{i}", f"result_{i}")
        jwm.train_step()
        assert jwm.save() is True
        assert (tmp_path / "model.pt").exists()
        # A fresh instance loads the same weights
        jwm2 = JepaWorldModel(persistence_path=tmp_path / "model.pt")
        state1 = jwm._model.state_dict()
        state2 = jwm2._model.state_dict()
        for k in state1:
            assert torch.allclose(state1[k], state2[k])

    def test_state_keys(self, tmp_path):
        jwm = self._model(tmp_path)
        state = jwm.get_state()
        for key in ("torch_ok", "turn_count", "train_count", "tick_count",
                    "last_error", "error_history", "loss_history",
                    "persistence", "model_params"):
            assert key in state
        assert state["torch_ok"] is True
        assert state["model_params"] > 0

    def test_singleton(self):
        reset_jepa_world_model()
        a = get_jepa_world_model()
        b = get_jepa_world_model()
        assert a is b
        reset_jepa_world_model()
