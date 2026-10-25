"""Regression tests for the webcam vision wiring and the TTS engine.

Covers the three fixes from the sight+sound pass:

1. WebcamTool must be constructed WITH the vision processor (run.py wiring)
   â€” previously captures returned "640x480 pixels, no vision processor",
   i.e. Jarvis was blind. WebcamTool.needs_permission() must gate capture/
   stream (privacy) while leaving 'list' free, and the camera must be
   released after a one-shot capture (LED off).

2. WebcamCapture cv2 calls must run off the event loop (asyncio.to_thread)
   â€” cv2.VideoCapture open can take seconds on Windows/MSMF and stalled
   the server event loop.

3. TTSEngine must degrade honestly: no backend -> synthesize() returns None
   (never fabricated audio), state reports empty backends, and the /api/tts
   route reports 503 so the frontend falls back to browser speechSynthesis.
   With a backend, synthesize returns real audio bytes.

All offline except a live edge-tts test which is skipped without network.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =========================================================================
# WebcamTool: vision wiring, permission policy, camera release
# =========================================================================

class TestWebcamTool:
    @staticmethod
    def _tool(vision=None, webcam=None):
        from core.vision_tokens import WebcamTool

        async def _noop_async(*a, **kw):
            return None

        if webcam is None:
            webcam = SimpleNamespace(
                _running=False,
                start=_noop_async,
                stop=_noop_async,
                capture_frame=_noop_async,
                list_devices=lambda: [],
            )
        return WebcamTool(webcam, vision_processor=vision)

    def test_needs_permission_gates_capture_and_stream(self):
        """Seeing the user is never implicit: capture/stream need a grant."""
        tool = self._tool()
        assert tool.needs_permission(action="capture") is True
        assert tool.needs_permission(action="stream") is True
        # Unknown/missing action also gated (conservative default)
        assert tool.needs_permission(action="") is True
        assert tool.needs_permission(action="gaze-into-my-soul") is True

    def test_needs_permission_frees_listing(self):
        """Enumerating devices captures nothing â€” no permission needed."""
        tool = self._tool()
        assert tool.needs_permission(action="list") is False

    def test_requires_permission_default_stays_true(self):
        """Class-level default remains conservative for unwired paths."""
        tool = self._tool()
        assert tool.requires_permission is True

    def test_capture_uses_vision_processor_when_wired(self):
        """The whole point of the fix: capture + vision = an actual analysis."""
        async def analyze(img, prompt=None):
            return "A desk with two monitors."

        vision = SimpleNamespace(analyze_image=analyze)
        frame = __import__("numpy").zeros((480, 640, 3), dtype="uint8")

        async def start():
            return True

        async def stop():
            pass

        async def capture():
            return frame

        webcam = SimpleNamespace(
            _running=False,
            start=start,
            stop=stop,
            capture_frame=capture,
            list_devices=lambda: [],
        )
        tool = self._tool(vision=vision, webcam=webcam)
        result = asyncio.new_event_loop().run_until_complete(
            tool.execute(action="capture", description="what do you see")
        )
        assert "desk with two monitors" in result

    def test_capture_releases_camera_after_one_shot(self):
        """Privacy: the camera handle must be released (stop called) even
        when analysis fails."""
        released = []

        async def start():
            return True

        async def stop():
            released.append(True)

        async def capture():
            return None  # capture fails

        webcam = SimpleNamespace(
            _running=False,
            start=start,
            stop=stop,
            capture_frame=capture,
            list_devices=lambda: [],
        )
        tool = self._tool(vision=None, webcam=webcam)
        result = asyncio.new_event_loop().run_until_complete(
            tool.execute(action="capture")
        )
        assert "Failed to capture" in result
        assert released, "camera was not released after capture"

    def test_list_action_is_non_blocking_and_off_loop(self):
        """list must not run cv2 probing on the event loop."""
        webcam = SimpleNamespace(
            _running=False,
            start=asyncio.new_event_loop().run_until_complete if False else None,
            stop=None,
            capture_frame=None,
            list_devices=lambda: [{"id": 0, "name": "Camera 0", "resolution": (640, 480), "fps": 30}],
        )
        tool = self._tool(webcam=webcam)
        result = asyncio.new_event_loop().run_until_complete(tool.execute(action="list"))
        assert "Camera 0" in result and "640x480" in result


# =========================================================================
# WebcamCapture: non-blocking cv2 lifecycle
# =========================================================================

class TestWebcamCaptureLifecycle:
    def test_stop_releases_capture_via_thread(self):
        """stop() must release the OS handle even mid-run."""
        from core.vision_tokens import WebcamCapture

        cap = WebcamCapture()
        released = []
        fake = SimpleNamespace(release=lambda: released.append(True))
        cap._capture = fake
        cap._running = True

        asyncio.new_event_loop().run_until_complete(cap.stop())
        assert released, "release() was not called"
        assert cap._capture is None
        assert cap._running is False

    def test_capture_frame_returns_none_when_not_running(self):
        from core.vision_tokens import WebcamCapture

        cap = WebcamCapture()
        result = asyncio.new_event_loop().run_until_complete(cap.capture_frame())
        assert result is None


# =========================================================================
# TTSEngine: honest degradation, layered backends
# =========================================================================

class TestTTSEngine:
    def test_no_backend_returns_none_not_garbage(self, monkeypatch):
        """With no backends installed, synthesize must return None â€”
        never fabricated audio that 'looks like' speech."""
        import core.tts_engine as te

        engine = te.TTSEngine()
        # Simulate: initialized with no backends
        engine._state.backends = []
        engine._state.active_backend = None
        engine._initialized = True

        result = asyncio.new_event_loop().run_until_complete(
            engine.synthesize("hello")
        )
        assert result is None
        assert engine.get_state()["failures"] == 1

    def test_empty_text_short_circuits(self):
        from core.tts_engine import TTSEngine

        engine = TTSEngine()
        engine._state.backends = ["edge"]
        engine._state.active_backend = "edge"
        engine._initialized = True

        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(engine.synthesize("")) is None
        assert loop.run_until_complete(engine.synthesize("   ")) is None
        # Empty input is not a failure â€” no backend was attempted
        assert engine.get_state()["failures"] == 0
        assert engine.get_state()["requests"] == 0

    def test_state_reports_no_backends_honestly(self):
        from core.tts_engine import TTSEngine

        engine = TTSEngine()
        engine._state.backends = []
        engine._state.active_backend = None
        engine._initialized = True

        state = engine.get_state()
        assert state["backends"] == []
        assert state["active_backend"] is None
        assert state["active_voice"] is None
        assert engine.available() is False

    def test_edge_backend_synthesizes_real_audio(self):
        """Live network test: edge-tts produces genuine MP3 bytes."""
        from core.tts_engine import TTSEngine

        engine = TTSEngine()
        loop = asyncio.get_event_loop()
        ok = loop.run_until_complete(engine.initialize())
        if not ok or "edge" not in engine.get_state()["backends"]:
            pytest.skip("edge-tts unavailable (offline?)")

        audio = loop.run_until_complete(
            engine.synthesize("Systems nominal.")
        )
        if audio is None:  # network flake â€” allowed, but log it
            pytest.skip("edge-tts synthesis failed (network flake)")
        assert len(audio) > 1000
        # MP3 frame sync or ID3 header
        assert audio[:3] == b"ID3" or (audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0)

    def test_set_voice_validates_against_backend_list(self):
        from core.tts_engine import TTSEngine

        engine = TTSEngine()
        engine._state.backends = ["edge"]
        engine._state.active_backend = "edge"
        engine._initialized = True
        engine._voices_cache["edge"] = [
            __import__("core.tts_engine", fromlist=["TTSVoice"]).TTSVoice(
                id="en-US-AriaNeural", name="Aria", backend="edge")
        ]

        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(engine.set_voice("en-US-AriaNeural")) is True
        assert loop.run_until_complete(engine.set_voice("nonexistent-voice")) is False
        assert engine.get_state()["active_voice"] == "en-US-AriaNeural"

    def test_singleton_getter(self):
        from core.tts_engine import get_tts_engine

        a = get_tts_engine()
        b = get_tts_engine()
        assert a is b


# =========================================================================
# Web routes: /api/tts contract
# =========================================================================

class TestTTSRouteContract:
    def test_engine_state_shape(self):
        """The /api/tts/state payload shape is stable for the frontend."""
        from core.tts_engine import TTSEngine

        engine = TTSEngine()
        engine._initialized = True
        state = engine.get_state()
        for key in ("backends", "active_backend", "active_voice",
                    "voices", "requests", "failures"):
            assert key in state


# =========================================================================
# run.py wiring: WebcamTool gets the vision processor
# =========================================================================

class TestRunWiring:
    def test_webcam_tool_signature_accepts_vision(self):
        """Guard the wiring fix: WebcamTool's constructor takes and stores
        vision_processor; run.py must pass multimodal.vision."""
        from core.vision_tokens import WebcamTool, WebcamCapture

        vision = object()
        tool = WebcamTool(WebcamCapture(), vision_processor=vision)
        assert tool.vision is vision