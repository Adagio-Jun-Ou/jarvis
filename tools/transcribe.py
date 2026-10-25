"""Transcription tool using Whisper for voice input."""

import logging
from pathlib import Path
from typing import Any

from core.helpers import safe_error_message
from .base import Tool

logger = logging.getLogger(__name__)


class TranscribeTool(Tool):
    """Tool for transcribing audio files using Whisper."""

    name = "transcribe"
    description = "Transcribe audio files to text using Whisper speech recognition"
    parameters = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to the audio file to transcribe"
            },
            "language": {
                "type": "string",
                "description": "Language code (e.g., 'en', 'es'). Auto-detected if not specified."
            }
        },
        "required": ["audio_path"]
    }
    requires_permission = False

    def __init__(self):
        self._model = None

    async def _load_model(self):
        """Load Whisper model lazily."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel("base", device="cpu", compute_type="int8")
                logger.info("Whisper model loaded for transcription")
            except ImportError:
                logger.error("faster-whisper not installed")
                return False
        return True

    async def execute(self, audio_path: str, language: str | None = None, **params) -> str | dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Optional language code

        Returns:
            Transcribed text
        """
        # Load model if needed
        if not await self._load_model():
            return "Error: Whisper model not available. Install faster-whisper."

        # Check if file exists
        path = Path(audio_path)
        if not path.exists():
            return f"Error: Audio file not found: {audio_path}"

        try:
            # Transcribe
            segments, info = self._model.transcribe(
                str(path),
                language=language,
                beam_size=5,
            )

            # Collect text
            text = "".join(segment.text for segment in segments).strip()

            # Clean up temp file
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp audio file {path}: {e}")

            return f"Transcription: {text}"

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Error transcribing audio: {safe_error_message(e, 'Transcription failed')}"


def get_transcribe_tool():
    """Get the transcribe tool instance."""
    return TranscribeTool()
