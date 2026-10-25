"""Regression tests for the Iris-inspired integrations.

1. Model-id single source of truth: dual_model + model_coordinator resolve
   their display/routing ids from the LIVE clients (config.yaml's
   llm.model / backend_llm.model) instead of 38 hardcoded "kimi-k2.5"/
   "glm-5" strings that had already diverged from the configured models.

2. LiveCameraFeed (Iris's live-feed design, privacy-first): the capture
   loop runs ONLY while consumers are attached; the last release stops the
   loop and releases the camera; frames live in memory only.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeLLM:
    def __init__(self, model):
        self.model = model


class TestModelIdSingleSource:
    def test_coordinator_resolves_from_clients(self):
        from core.model_coordinator import ModelCoordinator

        coord = ModelCoordinator(
            primary_llm=_FakeLLM("glm-5.3-flash"),
            backend_llm=_FakeLLM("kimi-k2.7-code"),
        )
        assert coord.primary_model_id == "glm-5.3-flash"
        assert coord.backend_model_id == "kimi-k2.7-code"
        # Registry: live ids present, template ids gone
        assert "glm-5.3-flash" in coord.models
        assert "kimi-k2.7-code" in coord.models
        assert "kimi-k2.5" not in coord.models
        assert "glm-5" not in coord.models

    def test_routing_substitutes_live_ids(self):
        from core.model_coordinator import ModelCoordinator

        coord = ModelCoordinator(
            primary_llm=_FakeLLM("glm-5.3-flash"),
            backend_llm=_FakeLLM("kimi-k2.7-code"),
        )
        assert coord.TASK_ROUTING["chat"][0] == "glm-5.3-flash"
        assert coord.TASK_ROUTING["code"][0] == "kimi-k2.7-code"
        assert coord.get_model_for_task("chat") == "glm-5.3-flash"
        assert coord.get_model_for_task("code") == "kimi-k2.7-code"
        # No template ids leak through any route
        for routes in coord.TASK_ROUTING.values():
            assert "kimi-k2.5" not in routes
            assert "glm-5" not in routes

    def test_same_client_collapses_both_roles(self):
        """Backend falls back to primary when they are the same client."""
        from core.model_coordinator import ModelCoordinator

        client = _FakeLLM("solo-model")
        coord = ModelCoordinator(primary_llm=client, backend_llm=client)
        assert coord.primary_model_id == "solo-model"
        assert coord.backend_model_id == "solo-model"

    def test_mock_client_falls_back_to_role_default(self):
        """MagicMock's .model is a Mock, not a str — must not poison the
        registry keys; resolution falls back to the role defaults."""
        from core.model_coordinator import ModelCoordinator

        coord = ModelCoordinator(
            primary_llm=MagicMock(), backend_llm=MagicMock())
        assert coord.primary_model_id == "glm-5.3-flash"
        assert coord.backend_model_id == "kimi-k2.7-code"
        # Registry keys are all strings
        assert all(isinstance(k, str) for k in coord.models)

    def test_dual_model_orchestrator_resolves_ids(self):
        from core.dual_model import DualModelOrchestrator

        orch = DualModelOrchestrator(
            primary_llm=_FakeLLM("glm-5.3-flash"),
            backend_llm=_FakeLLM("kimi-k2.7-code"),
        )
        assert orch.primary_config.model_id == "glm-5.3-flash"
        assert orch.backend_config.model_id == "kimi-k2.7-code"

    def test_status_reports_live_ids(self):
        from core.model_coordinator import ModelCoordinator

        coord = ModelCoordinator(
            primary_llm=_FakeLLM("glm-5.3-flash"),
            backend_llm=_FakeLLM("kimi-k2.7-code"),
        )
        status = coord.get_dual_model_status()
        assert status["primary"]["model"] == "glm-5.3-flash"
        assert status["backend"]["model"] == "kimi-k2.7-code"
        assert coord.get_connected_models() == ["glm-5.3-flash", "kimi-k2.7-code"]


class TestLiveCameraFeed:
    @staticmethod
    def _feed():
        from core.live_feed import LiveCameraFeed

        return LiveCameraFeed(device=99)  # nonexistent device: honest failure

    def test_attach_starts_release_stops(self):
        """Consumer-count lifecycle: first attach starts, last release stops."""
        feed = self._feed()
        assert feed.consumer_count == 0

        feed.attach()
        assert feed.consumer_count == 1
        # No running loop in this test context -> loop start is skipped with
        # a warning, but the consumer accounting must still be correct.
        feed.attach()
        assert feed.consumer_count == 2

        feed.release()
        assert feed.consumer_count == 1
        feed.release()
        assert feed.consumer_count == 0

    def test_frame_holder_honest_none(self):
        """No frames yet -> get_frame() returns None (never fabricated)."""
        feed = self._feed()
        assert feed.get_frame() is None

    def test_set_params_bounded(self):
        feed = self._feed()
        feed.set_params(fps=999, quality=999)
        assert feed._fps <= 15.0
        assert feed._quality <= 95
        feed.set_params(fps=0.01, quality=0)
        assert feed._fps >= 0.5
        assert feed._quality >= 10

    def test_singleton(self):
        from core.live_feed import get_live_camera_feed

        assert get_live_camera_feed() is get_live_camera_feed()

    def test_feed_loop_releases_camera_on_stop(self):
        """The loop must release the camera even if cancelled mid-flight."""
        from core.live_feed import LiveCameraFeed

        released = []

        class FakeCapture:
            async def start(self):
                return True

            async def capture_frame(self):
                return None

            async def stop(self):
                released.append(True)

        feed = LiveCameraFeed(device=0)
        feed._create_capture = lambda: FakeCapture()

        async def run():
            loop = asyncio.get_running_loop()
            feed._stop_event = asyncio.Event()
            task = loop.create_task(feed._feed_loop())
            await asyncio.sleep(0.05)
            feed._stop_event.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.new_event_loop().run_until_complete(run())
        assert released, "camera was not released after the loop ended"

    def test_feed_loop_honest_failure_when_camera_unavailable(self):
        """A camera that fails to start ends the loop with nothing fabricated."""
        from core.live_feed import LiveCameraFeed

        class DeadCapture:
            async def start(self):
                return False

            async def stop(self):
                pass

        feed = LiveCameraFeed(device=0)
        feed._create_capture = lambda: DeadCapture()

        async def run():
            loop = asyncio.get_running_loop()
            feed._stop_event = asyncio.Event()
            task = loop.create_task(feed._feed_loop())
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.new_event_loop().run_until_complete(run())
        assert feed.get_frame() is None
        assert feed._capture is None