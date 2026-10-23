"""Live camera feed for the web UI — Iris-inspired "seeing" capability.

Design idea imported from F:\projects\iris-ui (ambient-intelligence UI):
a shared in-memory latest-frame holder fed by a background capture loop,
polled over HTTP or streamed over WebSocket so the UI can show what
Jarvis sees, with an animated state (idle / seeing / thinking).

Privacy-first lifecycle (differs from Iris on purpose):
- The capture loop ONLY runs while at least one consumer is attached.
  No idle camera: attach() starts it, release() stops it and the camera
  LED goes off the moment the last viewer leaves.
- Every consumer path goes through the same WS origin guard as the other
  WebSocket endpoints (browsers always send Origin on handshakes; a
  cross-site page must not be able to watch the user's camera).
- Frames are held in memory only — never written to disk, never logged.

Uses Jarvis's existing WebcamCapture (asyncio.to_thread-wrapped cv2), so
the blocking device I/O stays off the event loop.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LiveCameraFeed:
    """Shared latest-frame camera feed, running only while consumers exist."""

    def __init__(self, device: int | None = None):
        self.device = int(device if device is not None
                          else os.environ.get("JARVIS_CAMERA_INDEX", 0))
        self._capture = None          # WebcamCapture (lazy)
        self._frame_b64: Optional[str] = None
        self._frame_ts: float = 0.0
        self._consumers = 0
        self._loop_task: Optional[asyncio.Task] = None
        self._fps = 2.0
        self._quality = 60
        self._stop_event = asyncio.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Consumer lifecycle
    # ------------------------------------------------------------------

    def attach(self) -> int:
        """Register a consumer; starts the feed loop on the first one.

        Returns the current consumer count.
        """
        with self._lock:
            self._consumers += 1
            first = self._consumers == 1
        if first:
            self._start_loop()
        return self._consumers

    def release(self) -> int:
        """Deregister a consumer; stops the feed (and the camera) on the last.

        Returns the remaining consumer count.
        """
        with self._lock:
            self._consumers = max(0, self._consumers - 1)
            last = self._consumers == 0
        if last:
            self.stop()
        return self._consumers

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    @property
    def consumer_count(self) -> int:
        return self._consumers

    # ------------------------------------------------------------------
    # Feed loop
    # ------------------------------------------------------------------

    def _start_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("LiveCameraFeed: no running loop; feed not started")
            return
        self._stop_event = asyncio.Event()
        self._loop_task = loop.create_task(self._feed_loop())

    def stop(self) -> None:
        """Stop the loop and release the camera (LED off)."""
        self._stop_event.set()
        self._frame_b64 = None
        capture = self._capture
        self._capture = None
        if capture is not None:
            from core.helpers import create_tracked_task
            create_tracked_task(capture.stop(), name="live_feed_camera_release")
        logger.info("LiveCameraFeed stopped (camera released)")

    async def _feed_loop(self) -> None:
        self._capture = self._create_capture()
        if not await self._capture.start():
            logger.warning(f"LiveCameraFeed: camera {self.device} unavailable")
            self._capture = None
            return

        interval = 1.0 / max(0.5, self._fps)
        logger.info(f"LiveCameraFeed started (device={self.device}, fps={self._fps:.1f})")
        try:
            while not self._stop_event.is_set():
                frame = await self._capture.capture_frame()
                if frame is not None:
                    encoded = await asyncio.to_thread(self._encode_jpeg, frame, self._quality)
                    if encoded:
                        self._frame_b64 = base64.b64encode(encoded).decode()
                        self._frame_ts = time.time()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            if self._capture is not None:
                try:
                    await self._capture.stop()
                except Exception:
                    pass
                self._capture = None
            logger.info("LiveCameraFeed loop ended")

    def _create_capture(self):
        """Capture backend factory (overridable in tests)."""
        from core.vision_tokens import WebcamCapture

        return WebcamCapture(device=self.device, fps=int(self._fps))

    @staticmethod
    def _encode_jpeg(frame, quality: int) -> Optional[bytes]:
        """Encode a BGR/RGB numpy frame to JPEG; None on failure."""
        try:
            from PIL import Image

            image = Image.fromarray(frame)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=max(10, min(quality, 95)))
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"LiveCameraFeed encode failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[dict]:
        """Latest frame as {"image": b64, "timestamp": ts} or None."""
        if not self._frame_b64:
            return None
        return {"image": self._frame_b64, "timestamp": self._frame_ts}

    def set_params(self, fps: float | None = None, quality: int | None = None) -> None:
        """Adjust feed parameters (applied live for quality, next start for fps)."""
        if fps is not None:
            self._fps = max(0.5, min(float(fps), 15.0))
        if quality is not None:
            self._quality = max(10, min(int(quality), 95))


# ---------------------------------------------------------------------------
# Singleton (thread-safe, per repo convention)
# ---------------------------------------------------------------------------

_feed: LiveCameraFeed | None = None
_feed_lock = threading.Lock()


def get_live_camera_feed() -> LiveCameraFeed:
    global _feed
    if _feed is None:
        with _feed_lock:
            if _feed is None:
                _feed = LiveCameraFeed()
    return _feed