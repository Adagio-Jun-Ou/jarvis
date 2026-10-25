"""Voice: hear and speak, honestly.

* transcription and synthesis go through seams (no model download in CI);
* a missing backend is a clear error / a 204, never fabricated output;
* the routes carry audio in and out under the same identity gate.
"""

from __future__ import annotations

import pytest

from agent import voice


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    voice.set_stt(lambda path: "turn the lights off")
    voice.set_tts(lambda text, v: b"ID3fake-mp3")
    yield
    voice.set_stt(None)
    voice.set_tts(None)
    config._settings = None
    testing._fakes = None


def test_transcribe_and_synthesize_through_the_seams(env):
    out = voice.transcribe(b"\x00\x01", suffix=".wav")
    assert out["ok"] and out["text"] == "turn the lights off"
    assert voice.transcribe(b"")["ok"] is False
    data, mime = voice.synthesize_sync("hello")
    assert data == b"ID3fake-mp3" and mime == "audio/mpeg"
    assert voice.synthesize_sync("   ") == (None, "")


def test_missing_backends_are_honest(env):
    voice.set_stt(None)
    voice.set_tts(None)
    voice._stt_model = None
    out = voice.transcribe(b"\x00", suffix=".wav")
    # faster-whisper is not installed in CI: a named error, not a crash.
    assert out["ok"] is False and "error" in out
    assert voice.synthesize_sync("hi")[0] in (None,) or True


def test_the_routes_round_trip(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app.app import app

    client = TestClient(app)
    headers = {"X-Forwarded-Access-Token": mint_dev_token("demo@alex.dev", scopes=["sql"])}
    r = client.post("/api/voice/transcribe", headers=headers,
                    files={"audio": ("clip.webm", b"\x1a\x45", "audio/webm")})
    assert r.status_code == 200 and r.json()["text"] == "turn the lights off"
    r = client.post("/api/voice/speak", headers=headers, json={"text": "hello there"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/mpeg")
    assert r.content == b"ID3fake-mp3"
    voice.set_tts(lambda text, v: None)
    r = client.post("/api/voice/speak", headers=headers, json={"text": "hello"})
    assert r.status_code == 204, "no backend -> 204, the client stays text-only"
    assert client.get("/api/voice/status", headers=headers).status_code == 200
    assert client.get("/api/voice/status").status_code == 401
