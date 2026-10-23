"""Voice — hear and speak, on this machine's hardware.

Ported from Jarvis's voice interface as two pure functions behind
swappable seams:

* :func:`transcribe` — faster-whisper, on **CUDA when the GPU is there**
  (the RTX class runs large-v3 in real time; Jarvis pinned CPU int8 and
  a 16-GB GPU sat idle), CPU int8 otherwise; model size by
  ``ALEX_WHISPER_MODEL`` (default large-v3 on GPU, base on CPU).
* :func:`synthesize` — edge-tts (Microsoft neural voices, MP3), voice by
  ``ALEX_TTS_VOICE``.

Both degrade honestly: a missing package is a clear error string, never
a fabricated transcript or silence pretending to be speech. Wake-word
listening stays out of the server on purpose — an always-on microphone
belongs to the client the person can see.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any

log = logging.getLogger("agent.voice")

_stt_model: Any = None
_stt_override: Any = None   # test seam: callable(path) -> str
_tts_override: Any = None   # test seam: callable(text, voice) -> bytes | None

DEFAULT_VOICE = os.environ.get("ALEX_TTS_VOICE", "en-GB-RyanNeural")
MAX_TTS_CHARS = 3000


def set_stt(fn: Any) -> None:
    global _stt_override
    _stt_override = fn


def set_tts(fn: Any) -> None:
    global _tts_override
    _tts_override = fn


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _load_whisper() -> Any:
    """The STT model, loaded once; GPU first."""
    global _stt_model
    if _stt_model is not None:
        return _stt_model
    from faster_whisper import WhisperModel

    if _cuda_available():
        size = os.environ.get("ALEX_WHISPER_MODEL", "large-v3")
        device, compute = "cuda", "float16"
    else:
        size = os.environ.get("ALEX_WHISPER_MODEL", "base")
        device, compute = "cpu", "int8"
    log.info("loading whisper %s on %s (%s)", size, device, compute)
    _stt_model = WhisperModel(size, device=device, compute_type=compute)
    return _stt_model


def transcribe(audio: bytes, *, suffix: str = ".webm", language: str | None = None) -> dict:
    """Audio bytes (webm/wav/mp3/ogg) -> {"ok", "text", "language", "duration_s"}."""
    if not audio:
        return {"ok": False, "error": "no audio"}
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        if _stt_override is not None:
            return {"ok": True, "text": str(_stt_override(path)).strip(),
                    "language": language or "en", "duration_s": 0.0}
        try:
            model = _load_whisper()
        except ImportError:
            return {"ok": False,
                    "error": "speech recognition needs: pip install faster-whisper"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"whisper failed to load: {exc}"}
        segments, info = model.transcribe(path, language=language, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"ok": True, "text": text,
                "language": getattr(info, "language", language or ""),
                "duration_s": round(float(getattr(info, "duration", 0.0) or 0.0), 1)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def synthesize(text: str, voice: str | None = None) -> tuple[bytes | None, str]:
    """Text -> (audio bytes, mime). None when no backend can speak."""
    text = (text or "").strip()[:MAX_TTS_CHARS]
    if not text:
        return None, ""
    voice = voice or DEFAULT_VOICE
    if _tts_override is not None:
        data = _tts_override(text, voice)
        return (data, "audio/mpeg") if data else (None, "")
    try:
        import edge_tts
    except ImportError:
        log.warning("edge-tts not installed; text-only")
        return None, ""
    try:
        buf = bytearray()
        async for chunk in edge_tts.Communicate(text=text, voice=voice).stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return (bytes(buf), "audio/mpeg") if buf else (None, "")
    except Exception as exc:  # noqa: BLE001
        log.warning("edge-tts failed: %s", exc)
        return None, ""


def synthesize_sync(text: str, voice: str | None = None) -> tuple[bytes | None, str]:
    return asyncio.run(synthesize(text, voice))


def status() -> dict:
    caps = {}
    for mod in ("faster_whisper", "edge_tts", "ctranslate2"):
        try:
            __import__(mod)
            caps[mod] = True
        except ImportError:
            caps[mod] = False
    caps["cuda"] = _cuda_available()
    caps["voice"] = DEFAULT_VOICE
    return caps
