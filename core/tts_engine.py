"""Server-side text-to-speech for Jarvis.

Provides TTSEngine, a single pipeline with layered backends:

1. edge-tts (Microsoft neural voices, online, highest quality, free)
2. Piper (local neural, offline, if installed with a voice model)
3. None — honest failure, callers degrade to text-only

Design notes (same honesty principles as memory_hub embeddings):

- No fabricated audio: if a backend cannot synthesize, it returns None and
  callers must handle it; never substitute garbage that "looks like" speech.
- Blocking work (Piper WAV synthesis) runs via asyncio.to_thread so the
  server event loop is never stalled.
- Voices are enumerated once per backend, lazily, and cached.
- Rate/volume/pitch settings map onto each backend where supported.

Used by the /api/tts route and the voice interface; the web frontend falls
back to browser speechSynthesis when this endpoint reports no backend.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSVoice:
    """A selectable TTS voice."""
    id: str                    # Backend-specific voice id
    name: str                  # Human-readable name
    backend: str               # "edge" | "piper"
    language: str = ""         # e.g. "en-US"
    gender: str = ""           # "male" | "female" | ""
    description: str = ""


@dataclass
class TTSEngineState:
    """Snapshot for /api/tts/state."""
    backends: list[str] = field(default_factory=list)
    active_backend: Optional[str] = None
    active_voice: Optional[str] = None
    voices: list[TTSVoice] = field(default_factory=list)
    requests: int = 0
    failures: int = 0


class TTSEngine:
    """Layered text-to-speech engine.

    Backend order: edge-tts (online neural) -> piper (offline neural).
    Each backend reports availability honestly; if none is available,
    synthesize() returns None and callers must degrade to text.
    """

    DEFAULT_EDGE_VOICE = "en-US-AriaNeural"  # warm, natural female
    EDGE_VOICE_FALLBACKS = [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
    ]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._edge: any = None          # edge_tts.Communicate factory
        self._piper_voice: any = None
        self._piper_dir = Path.home() / ".jarvis" / "voices"
        self._state = TTSEngineState()
        self._voices_cache: dict[str, list[TTSVoice]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        # Optional overrides from config
        self.default_voice = self.config.get("tts_voice", self.DEFAULT_EDGE_VOICE)
        self.default_rate = self.config.get("tts_rate", "+0%")
        self.default_volume = self.config.get("tts_volume", "+0%")
        self.default_pitch = self.config.get("tts_pitch", "+0Hz")

    # ------------------------------------------------------------------
    # Initialization / availability
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Probe backends once. Returns True if at least one is usable."""
        async with self._lock:
            if self._initialized:
                return bool(self._state.backends)

            backends: list[str] = []

            # --- edge-tts (online neural) ---
            try:
                import edge_tts  # noqa: F401
                backends.append("edge")
            except ImportError:
                logger.debug("edge-tts not installed (pip install edge-tts)")

            # --- Piper (offline neural) ---
            self._piper_dir.mkdir(parents=True, exist_ok=True)
            piper_models = list(self._piper_dir.glob("*.onnx"))
            if piper_models:
                try:
                    from piper import PiperVoice  # noqa: F401
                    backends.append("piper")
                except ImportError:
                    logger.debug("Piper installed models found but piper package missing")

            self._state.backends = backends
            self._state.active_backend = backends[0] if backends else None
            if "edge" in backends:
                self._state.active_voice = self.default_voice
            elif "piper" in backends:
                self._state.active_voice = piper_models[0].stem
            self._state.voices = await self.list_voices()
            self._initialized = True

            if backends:
                logger.info(f"TTS ready: backends={backends} voice={self._state.active_voice}")
            else:
                logger.warning("TTS unavailable: no backend (edge-tts/piper). Text-only mode.")
            return bool(backends)

    def available(self) -> bool:
        return bool(self._state.backends)

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    async def list_voices(self) -> list[TTSVoice]:
        """List voices for the active backend (cached)."""
        if not self._state.backends:
            return []
        backend = self._state.active_backend
        if backend in self._voices_cache:
            return self._voices_cache[backend]

        voices: list[TTSVoice] = []
        if backend == "edge":
            try:
                import edge_tts
                result = await edge_tts.list_voices()
                for v in result:
                    # Keep it useful but bounded: English + a few major languages
                    short = v.get("ShortName", "")
                    if short.startswith(("en-", "en_GB", "en_US")) or "-" in short[:3]:
                        voices.append(TTSVoice(
                            id=short,
                            name=v.get("FriendlyName", short),
                            backend="edge",
                            language=v.get("Locale", ""),
                            gender=v.get("Gender", ""),
                        ))
                # Curated subset first, then the long tail
                curated = [v for v in voices if any(c in v.id for c in self.EDGE_VOICE_FALLBACKS)]
                rest = [v for v in voices if v not in curated]
                voices = curated + rest
            except Exception as e:
                logger.warning(f"edge-tts voice listing failed: {e}")
        elif backend == "piper":
            for model in self._piper_dir.glob("*.onnx"):
                voices.append(TTSVoice(
                    id=model.stem,
                    name=model.stem.replace("_", " "),
                    backend="piper",
                ))

        self._voices_cache[backend] = voices
        return voices

    async def set_voice(self, voice_id: str) -> bool:
        """Switch active voice (must exist in the active backend)."""
        await self.initialize()
        voices = await self.list_voices()
        if any(v.id == voice_id for v in voices):
            self._state.active_voice = voice_id
            return True
        return False

    def get_state(self) -> dict:
        return {
            "backends": self._state.backends,
            "active_backend": self._state.active_backend,
            "active_voice": self._state.active_voice,
            "voices": [
                {"id": v.id, "name": v.name, "backend": v.backend,
                 "language": v.language, "gender": v.gender}
                for v in self._state.voices[:200]  # bounded payload
            ],
            "requests": self._state.requests,
            "failures": self._state.failures,
        }

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, voice: str | None = None) -> Optional[bytes]:
        """Synthesize speech as MP3 bytes (edge) or WAV bytes (piper).

        Returns None when no backend can synthesize — callers must degrade
        honestly (text-only / browser speechSynthesis), never fabricate.
        """
        if not text or not text.strip():
            return None
        await self.initialize()
        self._state.requests += 1

        backend = self._state.active_backend
        if backend == "edge":
            result = await self._synthesize_edge(text, voice)
        elif backend == "piper":
            result = await self._synthesize_piper(text, voice)
        else:
            result = None

        if result is None:
            self._state.failures += 1
        return result

    async def _synthesize_edge(self, text: str, voice: str | None) -> Optional[bytes]:
        try:
            import edge_tts
            voice_id = voice or self._state.active_voice or self.DEFAULT_EDGE_VOICE
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=self.default_rate,
                volume=self.default_volume,
                pitch=self.default_pitch,
            )
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            data = buf.getvalue()
            return data if data else None
        except Exception as e:
            logger.warning(f"edge-tts synthesis failed: {e}")
            return None

    async def _synthesize_piper(self, text: str, voice: str | None) -> Optional[bytes]:
        try:
            from piper import PiperVoice

            voice_id = voice or self._state.active_voice
            model_path = self._piper_dir / f"{voice_id}.onnx"
            if not model_path.exists():
                # Any installed voice beats silence
                models = list(self._piper_dir.glob("*.onnx"))
                if not models:
                    return None
                model_path = models[0]

            if self._piper_voice is None or str(getattr(self._piper_voice, "_path", "")) != str(model_path):
                self._piper_voice = await asyncio.to_thread(PiperVoice.load, str(model_path))

            def _synth() -> bytes:
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wav_file:
                    self._piper_voice.synthesize_wav(text, wav_file)
                return wav_buf.getvalue()

            return await asyncio.to_thread(_synth)
        except Exception as e:
            logger.warning(f"Piper synthesis failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Singleton (thread-safe, like all Jarvis modules)
# ---------------------------------------------------------------------------

_engine: TTSEngine | None = None
_lock = threading.Lock()


def get_tts_engine(config: dict | None = None) -> TTSEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = TTSEngine(config)
    return _engine