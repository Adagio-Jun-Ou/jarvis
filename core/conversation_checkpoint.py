"""Conversation Checkpoint - Crash recovery for conversation state.

Saves conversation checkpoints periodically so that if Jarvis crashes
or is restarted, it can resume from the most recent checkpoint rather
than losing the entire conversation.

Inspired by LangGraph's checkpointing pattern for stateful workflows.
"""
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_checkpointer: Optional["ConversationCheckpointer"] = None
_checkpoint_lock = threading.Lock()


class ConversationCheckpointer:
    def __init__(self, checkpoint_dir: Path | None = None, max_checkpoints: int = 5):
        self._checkpoint_dir = checkpoint_dir or Path.home() / ".jarvis" / "checkpoints"
        self._max_checkpoints = max_checkpoints
        self._lock = threading.RLock()
        self._last_checkpoint_time: float = 0
        self._min_checkpoint_interval: float = 30.0  # seconds
        self._stats = {
            "checkpoints_saved": 0,
            "checkpoints_restored": 0,
            "bytes_written": 0,
        }

    def save_checkpoint(self, agent_state: dict, checkpoint_id: str | None = None) -> str | None:
        if time.time() - self._last_checkpoint_time < self._min_checkpoint_interval:
            return None

        checkpoint_id = checkpoint_id or f"checkpoint_{int(time.time()*1000)}"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in checkpoint_id)
        file_path = self._checkpoint_dir / f"{safe_id}.json"

        try:
            agent_state["checkpoint_time"] = datetime.now().isoformat()
            agent_state["checkpoint_id"] = checkpoint_id

            data = json.dumps(agent_state, default=str, ensure_ascii=False)

            resolved = file_path.resolve()
            if not str(resolved).startswith(str(self._checkpoint_dir.resolve())):
                logger.error(f"Path traversal blocked in checkpoint: {checkpoint_id}")
                return None

            temp_path = file_path.with_suffix(".tmp")
            temp_path.write_text(data, encoding="utf-8")
            temp_path.replace(file_path)

            with self._lock:
                self._stats["checkpoints_saved"] += 1
                self._stats["bytes_written"] += len(data)
                self._last_checkpoint_time = time.time()

                self._cleanup_old_checkpoints()
                self._prune_checkpoints()

            logger.debug(f"Checkpoint saved: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_latest_checkpoint(self) -> dict | None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = sorted(
            self._checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for cp_file in checkpoints:
            try:
                data = json.loads(cp_file.read_text(encoding="utf-8"))
                with self._lock:
                    self._stats["checkpoints_restored"] += 1
                logger.info(f"Restored checkpoint: {cp_file.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {cp_file.name}: {e}")
                continue

        return None

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = deque(sorted(
            self._checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
        ))
        while len(checkpoints) > self._max_checkpoints:
            oldest = checkpoints.popleft()
            try:
                oldest.unlink()
            except OSError:
                pass

    def _prune_checkpoints(self) -> None:
        now = time.time()
        max_age = 7 * 24 * 3600  # 7 days
        for cp_file in list(self._checkpoint_dir.glob("checkpoint_*.json")):
            try:
                if now - cp_file.stat().st_mtime > max_age:
                    cp_file.unlink()
            except OSError:
                pass

    def get_stats(self) -> dict:
        with self._lock:
            return {**self._stats, "checkpoint_dir": str(self._checkpoint_dir)}


def get_conversation_checkpointer() -> ConversationCheckpointer:
    global _checkpointer, _checkpoint_lock
    if _checkpointer is None:
        with _checkpoint_lock:
            if _checkpointer is None:
                _checkpointer = ConversationCheckpointer()
    return _checkpointer