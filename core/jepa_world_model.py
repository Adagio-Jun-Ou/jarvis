"""JEPA world model for Jarvis's own terminal sessions.

Ported from the mind-diffusion project (C:\\projects\\mind-diffusion,
src/mind_diffusion/mind_jepa_bridge.py): a latent-only world model —
JEPA, Joint-Embedding Predictive Architecture — that learns the
dynamics of Jarvis's shell sessions entirely in embedding space.
No decoder, no reconstruction: the world is PREDICTED, not redrawn.

Architecture (identical to mind-diffusion's MindJEPA):

    z_t   = frame_enc(obs_t)                 hashed bag-of-words frame
    h_t   = GRU(z_{<=t})                     context
    z_hat = pred(h_t, act_t)                 action OUT: what Jarvis ran
    loss  = MSE(z_hat, sg(frame_tgt(obs_{t+1})))   EMA target (stop-grad)
          + VICReg-lite variance/covariance  anti-collapse

The prediction error — how far the world's answer is from what the
model expected — is a LEARNED prediction-error source for
PredictiveProcessor (channel "jepa_terminal"), feeding the same
surprise pathway that gates dreaming.

Data comes from Jarvis's own history (ExecutionContext process log):
    turns = [("ls -la", "total 4 ..."), ("git status", "On branch main ...")]

Torch is a lazy dependency: without it the module still collects turns
and reports state, it just cannot train or predict.
"""

import hashlib
import logging
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_jepa_world = None
_jepa_world_lock = threading.Lock()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception as _e:  # pragma: no cover - environment dependent
    _TORCH_OK = False
    logger.debug(f"JEPA world model: torch unavailable ({_e})")


# --------------------------------------------------------------------------- #
# Frames: raw terminal turns -> hashed embeddings (no vocab, no deps)
# --------------------------------------------------------------------------- #
def hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (no vocab, no deps)."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in str(text).lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def session_to_frames(turns, obs_dim: int = 64, act_dim: int = 32):
    """turns: list of (command, output), ordered in time.

    obs_t    = hash(output_t) + 0.2 * hash(command_t)   (context leaks in)
    action_t = hash(command_t)                          (what was done)

    Returns (obs, act): (T, obs_dim), (T, act_dim) float32 tensors.
    """
    obs, act = [], []
    for command, output in turns:
        o = hash_embed(output, obs_dim) + 0.2 * hash_embed(command, obs_dim)
        n = float(np.linalg.norm(o))
        if n > 0:
            o = o / n
        obs.append(o)
        act.append(hash_embed(command, act_dim))
    if not obs:
        z_o = np.zeros((0, obs_dim), dtype=np.float32)
        z_a = np.zeros((0, act_dim), dtype=np.float32)
        if _TORCH_OK:
            return (torch.from_numpy(z_o), torch.from_numpy(z_a))
        return (z_o, z_a)
    obs_a = np.stack(obs).astype(np.float32)
    act_a = np.stack(act).astype(np.float32)
    if _TORCH_OK:
        return (torch.from_numpy(obs_a), torch.from_numpy(act_a))
    return (obs_a, act_a)


if _TORCH_OK:

    class MindJEPA(nn.Module):
        """Small CPU-friendly JEPA: frame encoder + GRU context + action
        predictor. No decoder, no reconstruction. The only grounding is
        the EMA target encoder plus VICReg-lite variance/covariance
        terms against collapse. Ported from mind-diffusion."""

        def __init__(self, obs_dim: int = 64, act_dim: int = 32,
                     zdim: int = 64, hidden: int = 128, ema: float = 0.99):
            super().__init__()
            self.zdim = zdim
            self.ema = ema
            self.frame_enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(),
                                           nn.Linear(hidden, zdim))
            self.frame_tgt = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(),
                                           nn.Linear(hidden, zdim))
            for p in self.frame_tgt.parameters():
                p.requires_grad_(False)
            self.ctx = nn.GRU(zdim, hidden, batch_first=True)
            self.act_emb = nn.Linear(act_dim, hidden)
            self.pred = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, zdim))

        @torch.no_grad()
        def _ema_update(self):
            m = self.ema
            for pt, p in zip(self.frame_tgt.parameters(), self.frame_enc.parameters()):
                pt.mul_(m).add_(p.detach(), alpha=1.0 - m)

        def _vicreg(self, z, var_w=1.0, cov_w=0.04):
            std = torch.sqrt(z.var(0) + 1e-4)
            var_loss = torch.relu(1.0 - std).mean()
            zc = z - z.mean(0)
            B = z.shape[0]
            cov = (zc.T @ zc) / max(B - 1, 1)
            cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / z.shape[1]
            return var_w * var_loss + cov_w * cov_loss

        def training_step(self, obs, act):
            """obs: (B, T, obs_dim), act: (B, T, act_dim).

            Predicts frame t+1 from frames <= t and action_t. Returns
            (loss, pred_loss, reg_loss).
            """
            B, T, _ = obs.shape
            if T < 2:
                z0 = torch.zeros((), requires_grad=True)
                return (z0 + 0.0), 0.0, 0.0
            with torch.no_grad():
                z_tgt = self.frame_tgt(obs[:, 1:])           # (B, T-1, zdim)
            z = self.frame_enc(obs)                          # (B, T, zdim)
            h, _ = self.ctx(z)
            h_t = h[:, :-1]                                  # context up to t
            a_t = self.act_emb(act[:, :-1])
            z_hat = self.pred(torch.tanh(h_t + a_t))         # (B, T-1, zdim)

            pred_loss = F.mse_loss(z_hat, z_tgt)
            reg_loss = self._vicreg(z_hat.reshape(-1, self.zdim))
            self._ema_update()
            return pred_loss + reg_loss, float(pred_loss.detach()), float(reg_loss.detach())

        @torch.no_grad()
        def roll(self, z0, actions):
            """Roll the world forward in latent space.

            z0: (B, zdim) initial world latent; actions: (B, S, act_dim).
            Returns (B, S, zdim) predicted world latents.
            """
            h = None
            zs = []
            z = z0
            for s in range(actions.shape[1]):
                out, h = self.ctx(z.unsqueeze(1), h)
                z = self.pred(torch.tanh(out[:, 0] + self.act_emb(actions[:, s])))
                zs.append(z)
            return torch.stack(zs, dim=1)

        @torch.no_grad()
        def encode_last(self, obs):
            """Encode observation history -> (GRU state h_n, last latent)."""
            if obs.shape[1] == 0:
                h0 = torch.zeros(1, obs.shape[0], self.ctx.hidden_size)
                z0 = torch.zeros(obs.shape[0], self.zdim)
                return h0, z0
            z = self.frame_enc(obs)
            _out, h_n = self.ctx(z)
            return h_n, z[:, -1]


class JepaWorldModel:
    """Jarvis-facing JEPA world model over its own terminal sessions.

    Collects (command, output) turns, trains the latent dynamics model
    online in a background tick, evaluates one-step-ahead prediction
    error, and feeds it to PredictiveProcessor as a learned
    prediction-error channel ("jepa_terminal").
    """

    _MAX_TURNS = 500
    _MAX_LOSS_HISTORY = 100
    _MAX_ERROR_HISTORY = 100
    _TRAIN_WINDOW = 8       # frames per training window
    _TRAIN_BATCH = 4        # windows per training step
    _MIN_TURNS = 4          # need >= this many turns to train/evaluate
    _PERSIST_EVERY = 10     # ticks between persistence saves
    _SEEN_KEYS_MAX = 1000

    def __init__(self, persistence_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._turns: deque[tuple[str, str]] = deque(maxlen=self._MAX_TURNS)
        self._seen_keys: deque[str] = deque(maxlen=self._SEEN_KEYS_MAX)
        self._loss_history: deque[float] = deque(maxlen=self._MAX_LOSS_HISTORY)
        self._error_history: deque[float] = deque(maxlen=self._MAX_ERROR_HISTORY)
        self._train_count = 0
        self._tick_count = 0
        self._last_error: Optional[float] = None
        self._last_error_time = 0.0
        self._last_eval_step = -1  # turn index the error was evaluated at
        self._model = None
        self._optimizer = None
        self._torch_ok = _TORCH_OK
        if _TORCH_OK:
            self._model = MindJEPA()
            self._optimizer = torch.optim.Adam(
                [p for p in self._model.parameters() if p.requires_grad],
                lr=3e-3)
        self._persistence_path = Path(persistence_path) if persistence_path \
            else Path.home() / ".jarvis" / "jepa_world" / "model.pt"
        self._load_state()

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #
    def record_turn(self, command: str, output: str) -> None:
        """Record one terminal turn (command, output)."""
        with self._lock:
            self._turns.append((str(command)[:2000], str(output)[:2000]))

    def ingest_context_commands(self, records: list[dict]) -> int:
        """Ingest ExecutionContext.get_recent_commands() dicts.

        Deduplicates by (pid, started_at); returns how many new turns
        were ingested. Records lacking output tails are skipped — a
        command without its world-answer teaches nothing.
        """
        new_turns = 0
        with self._lock:
            for r in records or []:
                try:
                    key = f"{r.get('pid')}:{r.get('started_at')}"
                    if key in self._seen_keys:
                        continue
                    command = r.get("command", "")
                    output = " ".join(filter(None, (
                        r.get("stdout_tail", ""), r.get("stderr_tail", ""))))
                    if not command or not output.strip():
                        # Mark seen even when unusable so it is not retried
                        self._seen_keys.append(key)
                        continue
                    self._seen_keys.append(key)
                    self._turns.append((command[:2000], output[:2000]))
                    new_turns += 1
                except Exception as e:
                    logger.debug(f"JEPA ingest record failed: {e}")
        return new_turns

    # ------------------------------------------------------------------ #
    # Training + evaluation
    # ------------------------------------------------------------------ #
    def train_step(self) -> Optional[float]:
        """One training step on a sampled batch of contiguous windows.

        Returns the total loss, or None when there is not enough data
        or torch is unavailable.
        """
        if not self._torch_ok or self._model is None:
            return None
        with self._lock:
            turns = list(self._turns)
        if len(turns) < self._MIN_TURNS:
            return None

        obs, act = session_to_frames(turns)
        T_total = obs.shape[0]
        W = min(self._TRAIN_WINDOW, T_total)
        if W < 2:
            return None

        losses = []
        with self._lock:
            model = self._model
            optimizer = self._optimizer
        model.train()
        for _ in range(self._TRAIN_BATCH):
            if T_total == W:
                start = 0
            else:
                start = random.randint(0, T_total - W)
            w_obs = obs[start:start + W].unsqueeze(0)
            w_act = act[start:start + W].unsqueeze(0)
            optimizer.zero_grad()
            loss, _pred, _reg = model.training_step(w_obs, w_act)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = sum(losses) / max(1, len(losses))
        with self._lock:
            self._train_count += 1
            self._loss_history.append(round(mean_loss, 5))
        return mean_loss

    def evaluate_prediction_error(self, force: bool = False) -> Optional[float]:
        """One-step-ahead error on the most recent turn.

        Predicts the last frame's latent from the preceding context and
        action (the same math as the training loss, in eval mode) and
        compares it to the actual encoded frame via cosine distance
        (scale-invariant, clamped to [0, 1]). By default returns None
        when there is nothing new to evaluate (no stale re-feeding to
        PredictiveProcessor); force=True re-evaluates regardless —
        useful for measuring learning progress on a static dataset.
        """
        if not self._torch_ok or self._model is None:
            return None
        with self._lock:
            turns = list(self._turns)
            last_eval = self._last_eval_step
        if len(turns) < self._MIN_TURNS:
            return None
        if not force and last_eval == len(turns) - 1:
            return None  # nothing new since the last evaluation

        obs, act = session_to_frames(turns)
        model = self._model
        model.eval()
        with torch.no_grad():
            # Evaluate on the SAME window structure training used —
            # a full-history GRU state is out-of-distribution for a
            # model trained on _TRAIN_WINDOW-length windows.
            W = min(len(turns), self._TRAIN_WINDOW)
            obs_w = obs[-W:]
            act_w = act[-W:]
            z_ctx = model.frame_enc(obs_w[:-1].unsqueeze(0))     # (1, W-1, zdim)
            h, _ = model.ctx(z_ctx)
            a_t = model.act_emb(act_w[:-1][-1]).unsqueeze(0)     # (1, hidden)
            z_hat = model.pred(torch.tanh(h[:, -1] + a_t))       # (1, zdim)
            z_tgt = model.frame_tgt(obs_w[-1].unsqueeze(0))      # (1, zdim)
            cos = F.cosine_similarity(z_hat, z_tgt, dim=1)
            error = float((1.0 - cos).clamp(0.0, 1.0)[0])

        with self._lock:
            self._error_history.append(round(error, 4))
            self._last_error = error
            self._last_error_time = time.time()
            self._last_eval_step = len(turns) - 1
        return error

    def tick(self) -> dict:
        """Background tick: harvest ExecutionContext turns, train one
        step, evaluate prediction error. Returns tick diagnostics."""
        with self._lock:
            self._tick_count += 1
        ingested = 0
        try:
            from core.execution_context import get_execution_context
            ctx = get_execution_context()
            ingested = self.ingest_context_commands(ctx.get_recent_commands(200))
        except Exception as e:
            logger.debug(f"JEPA context harvest skipped: {e}")

        loss = None
        try:
            loss = self.train_step()
        except Exception as e:
            logger.debug(f"JEPA train step failed: {e}")

        error = None
        try:
            error = self.evaluate_prediction_error()
        except Exception as e:
            logger.debug(f"JEPA evaluation failed: {e}")

        if self._tick_count % self._PERSIST_EVERY == 0:
            try:
                self.save()
            except Exception as e:
                logger.debug(f"JEPA persistence failed: {e}")

        return {
            "ingested": ingested,
            "loss": loss,
            "fresh_error": error,
        }

    # ------------------------------------------------------------------ #
    # Consumption + state
    # ------------------------------------------------------------------ #
    def get_last_error(self) -> Optional[float]:
        with self._lock:
            return self._last_error

    def feed_predictive_processor(self) -> Optional[float]:
        """Push the latest prediction error into PredictiveProcessor as
        a learned world-model violation (channel "jepa_terminal")."""
        error = self.get_last_error()
        if error is None:
            return None
        try:
            from core.predictive_processing import get_predictive_processor
            get_predictive_processor().process_world_model_errors(
                {"jepa_terminal": error})
            return error
        except Exception as e:
            logger.debug(f"JEPA error feed failed: {e}")
            return None

    def get_state(self) -> dict:
        with self._lock:
            return {
                "torch_ok": self._torch_ok,
                "turn_count": len(self._turns),
                "train_count": self._train_count,
                "tick_count": self._tick_count,
                "last_error": round(self._last_error, 4) if self._last_error is not None else None,
                "error_history": list(self._error_history)[-10:],
                "loss_history": list(self._loss_history)[-10:],
                "persistence": str(self._persistence_path),
                "model_params": sum(p.numel() for p in self._model.parameters())
                                if self._model is not None else 0,
            }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> bool:
        if not self._torch_ok or self._model is None:
            return False
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persistence_path.with_suffix(".tmp")
            torch.save(self._model.state_dict(), tmp)
            tmp.replace(self._persistence_path)
            return True
        except Exception as e:
            logger.debug(f"JEPA save failed: {e}")
            return False

    def _load_state(self) -> None:
        if not self._torch_ok or self._model is None:
            return
        try:
            if self._persistence_path.exists():
                self._model.load_state_dict(
                    torch.load(self._persistence_path, map_location="cpu"))
                logger.info(f"JEPA world model loaded from {self._persistence_path}")
        except Exception as e:
            logger.debug(f"JEPA load failed (starting fresh): {e}")


def get_jepa_world_model() -> JepaWorldModel:
    global _jepa_world
    with _jepa_world_lock:
        if _jepa_world is None:
            _jepa_world = JepaWorldModel()
        return _jepa_world


def reset_jepa_world_model() -> None:
    global _jepa_world
    with _jepa_world_lock:
        _jepa_world = None
