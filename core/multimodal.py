"""Multi-modal integration for Jarvis.

This module enables Jarvis to process multiple input modalities:
- Vision (images) - using vision-capable models
- Audio (speech) - using Whisper for transcription
- Text - standard text processing
"""
import asyncio
import base64
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from core.helpers import get_response_content, safe_error_message

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of input modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class MultimodalInput:
    """Represents a multi-modal input to Jarvis."""
    text: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)  # List of image data
    audio: dict[str, Any] | None = None
    modality_type: ModalityType = ModalityType.TEXT
    timestamp: datetime = field(default_factory=datetime.now)

    def has_images(self) -> bool:
        """Check if input contains images."""
        return len(self.images) > 0

    def has_audio(self) -> bool:
        """Check if input contains audio."""
        return self.audio is not None

    def is_multimodal(self) -> bool:
        """Check if input is multi-modal."""
        return sum([
            bool(self.text),
            len(self.images) > 0,
            self.audio is not None,
        ]) > 1


class VisionProcessor:
    """Process visual inputs for Jarvis.

    Uses vision-capable models to understand images.
    Supports qwen3.5:397b, qwen3-vl:235b, and other vision models.
    """

    VISION_PROMPT = """Analyze this image carefully. Provide:
1. A clear description of what you see
2. Key objects, people, or elements present
3. Any text visible in the image
4. Context or relevant details
5. Any potential actions or implications

Be thorough but concise. If this relates to a user's question or task, consider how it might help."""

    def __init__(self, llm_client: Any, vision_model: str | None = None):
        """Initialize the vision processor.

        Args:
            llm_client: The LLM client (must support vision)
            vision_model: Optional specific vision model (defaults to main model)
        """
        self.llm = llm_client
        # Use main model for vision if not specified (qwen3.5:397b supports vision)
        self.vision_model = vision_model
        self._tested_vision = False
        self._vision_works = None

    async def analyze_image(
        self,
        image_data: str | bytes | Path,
        prompt: str | None = None,
        question: str | None = None,
    ) -> str:
        """Analyze an image and return a description.

        Args:
            image_data: Image as base64 string, bytes, or file path
            prompt: Custom prompt for analysis
            question: Optional specific question about the image

        Returns:
            Description/analysis of the image
        """
        # Prepare image data
        if isinstance(image_data, Path):
            image_bytes = await asyncio.to_thread(lambda: image_data.read_bytes())
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        elif isinstance(image_data, bytes):
            image_base64 = base64.b64encode(image_data).decode("utf-8")
        else:
            image_base64 = image_data

        # Strip data URL prefix if present
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]

        # Build prompt
        analysis_prompt = prompt or self.VISION_PROMPT
        if question:
            analysis_prompt += f"\n\nUser's question: {question}"

        # Create message in Ollama format (images as separate array)
        messages = [
            {
                "role": "user",
                "content": analysis_prompt,
                "images": [image_base64],
            }
        ]

        try:
            import logging
            logger = logging.getLogger(__name__)

            model_to_use = self.vision_model or self.llm.model
            payload = {
                "model": model_to_use,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": getattr(self.llm, 'temperature', 0.7),
                },
            }

            logger.info(f"Vision analysis request: model={model_to_use}")

            response = await self.llm.client.post(
                self.llm._get_chat_url(),
                json=payload,
            )

            if response.status_code == 400:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get("error", "")
                except Exception as e:
                    logger.debug(f"Error response parse failed: {e}")
                logger.warning(f"Vision model '{model_to_use}' doesn't support images: {error_detail}")
                return f"Vision analysis requires a vision-capable model. The model '{model_to_use}' doesn't support image inputs. Try configuring a vision model like 'llava' or 'bakllava' in config.yaml."

            response.raise_for_status()
            result = response.json()

            content = get_response_content(result)
            logger.info(f"Vision analysis complete: {len(content)} chars")
            return content

        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg or "Bad Request" in error_msg:
                return f"Vision analysis not available: The current model '{self.llm.model}' doesn't support image inputs. Please use a vision-capable model like 'llava' or 'bakllava' for image analysis."
            return f"Error analyzing image: {error_msg}"

    async def compare_images(
        self,
        images: list[str | bytes | Path],
        prompt: str | None = None,
    ) -> str:
        """Compare multiple images.

        Args:
            images: List of images to compare
            prompt: Custom prompt for comparison

        Returns:
            Comparison result
        """
        # Prepare all images
        image_urls = []
        for img in images:
            if isinstance(img, Path):
                img_bytes = await asyncio.to_thread(lambda p=img: p.read_bytes())
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            elif isinstance(img, bytes):
                img_base64 = base64.b64encode(img).decode("utf-8")
            else:
                img_base64 = img
            image_urls.append(f"data:image/jpeg;base64,{img_base64}")

        # Build message with all images
        content = [{"type": "text", "text": prompt or "Compare these images and describe the differences and similarities."}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

        messages = [{"role": "user", "content": content}]

        try:
            import logging
            logger = logging.getLogger(__name__)

            payload = {
                "model": self.llm.model,
                "messages": messages,
                "stream": False,
            }

            logger.info(f"Comparing {len(images)} images")

            response = await self.llm.client.post(
                self.llm._get_chat_url(),
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            content = get_response_content(result)
            logger.info(f"Image comparison complete: {len(content)} chars")
            return content

        except Exception as e:
            return f"Error comparing images: {safe_error_message(e)}"


class AudioProcessor:
    """Process audio inputs for Jarvis.

    Uses Faster-Whisper for speech-to-text transcription.
    Uses Voice Token system for semantic/acoustic token processing.
    Uses lazy loading to avoid loading the model until first use.
    """

    def __init__(self, whisper_model: str = "base", use_voice_tokens: bool = True):
        """Initialize the audio processor.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            use_voice_tokens: Whether to use voice token system for enhanced processing
        """
        self.whisper_model = whisper_model
        self.use_voice_tokens = use_voice_tokens
        self._model = None
        self._model_loaded = False
        self._voice_tokenizer = None

    def _load_model(self):
        """Lazy load the Whisper model - only when first needed."""
        if not self._model_loaded:
            import time
            logger.info(f"Loading Faster-Whisper model '{self.whisper_model}'...")
            start = time.perf_counter()
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
                duration = (time.perf_counter() - start) * 1000
                logger.info(f"Faster-Whisper model loaded in {duration:.0f}ms")
            except ImportError:
                raise ImportError("Faster-Whisper not installed. Run: pip install faster-whisper")
            self._model_loaded = True

            # Load voice tokenizer if enabled
            if self.use_voice_tokens:
                try:
                    from core.voice_tokens import SemanticTokenizer, AcousticTokenizer
                    self._voice_tokenizer = {
                        'semantic': SemanticTokenizer(),
                        'acoustic': AcousticTokenizer(),
                    }
                    logger.info("Voice token system initialized")
                except Exception as e:
                    logger.warning(f"Voice tokenizer not available: {e}")

    async def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "es")

        Returns:
            Dictionary with transcription and metadata
        """
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return {"error": f"Audio file not found: {audio_path}"}

        try:
            # Run transcription in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_running_loop()

            def _transcribe():
                segments, info = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=5,
                )
                # Collect segments into text
                text = "".join(segment.text for segment in segments).strip()
                return {
                    "text": text,
                    "language": info.language if hasattr(info, 'language') else language or "en",
                    "segments": [],  # Faster-whisper uses generator
                }

            result = await loop.run_in_executor(None, _transcribe)

            return {
                "text": result.get("text", ""),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "success": True,
            }

        except Exception as e:
            return {
                "error": f"Transcription failed: {safe_error_message(e)}",
                "success": False,
            }

    async def transcribe_audio_data(
        self,
        audio_data: bytes,
        format: str = "wav",
    ) -> dict[str, Any]:
        """Transcribe audio data (bytes) to text.

        Args:
            audio_data: Raw audio bytes
            format: Audio format (wav, webm, mp3, etc.)

        Returns:
            Dictionary with transcription
        """
        import tempfile
        import asyncio
        import os
        import time

        start = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            if format not in ("wav", "wave"):
                wav_path = temp_path + ".wav"

                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", wav_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        raise TimeoutError("ffmpeg timed out")
                    if proc.returncode == 0 and os.path.exists(wav_path):
                        os.unlink(temp_path)
                        temp_path = wav_path
                    else:
                        logger.debug("ffmpeg conversion failed, trying original file")
                except (TimeoutError, FileNotFoundError):
                    logger.debug("ffmpeg not available, trying original file")

                if not temp_path.endswith(".wav"):
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "sox", temp_path, "-r", "16000", "-c", "1", wav_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        try:
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                        except asyncio.TimeoutError:
                            proc.kill()
                            await proc.wait()
                            raise TimeoutError("sox timed out")
                        if proc.returncode == 0 and os.path.exists(wav_path):
                            os.unlink(temp_path)
                            temp_path = wav_path
                    except (TimeoutError, FileNotFoundError):
                        pass

            result = await self.transcribe(temp_path)

            # Track audio call in dual model stats
            latency_ms = (time.perf_counter() - start) * 1000
            try:
                from core.dual_model import get_dual_model_orchestrator
                orchestrator = get_dual_model_orchestrator()
                if orchestrator:
                    orchestrator.record_audio_call(latency_ms)
            except Exception as e:
                logger.debug(f"Failed to record audio call: {e}")

            # Also extract voice tokens for enhanced processing
            if self._voice_tokenizer and result.get("success"):
                try:
                    import numpy as np
                    # Read audio as numpy for tokenization
                    import wave
                    with wave.open(temp_path, 'rb') as wf:
                        audio_array = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                        audio_array = audio_array.astype(np.float32) / 32768.0
                        sample_rate = wf.getframerate()

                    # Extract semantic tokens
                    semantic_tokens = self._voice_tokenizer['semantic'].tokenize(audio_array, sample_rate)
                    result["semantic_tokens"] = semantic_tokens[:50]  # First 50 tokens for context
                    result["semantic_token_count"] = len(semantic_tokens)

                    # Extract acoustic tokens (for speaker identification)
                    acoustic_tokens = self._voice_tokenizer['acoustic'].tokenize(audio_array, sample_rate)
                    result["acoustic_frame_count"] = len(acoustic_tokens)

                    # Extract speaker embedding
                    from core.voice_tokens import VoiceTokenBridge
                    bridge = VoiceTokenBridge()
                    result["speaker_id"] = bridge.extract_speaker_embedding(acoustic_tokens)

                except Exception as e:
                    logger.debug(f"Voice token extraction failed: {e}")

            return result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def synthesize(
        self,
        text: str,
        speaker_id: str | None = None,
        emotion: str | None = None,
    ) -> tuple[bytes, int]:
        """Synthesize speech from text using voice tokens.
        
        Args:
            text: Text to synthesize
            speaker_id: Optional speaker ID for voice cloning
            emotion: Optional emotion (happy, sad, neutral, etc.)
            
        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        if self._voice_tokenizer:
            try:
                from core.voice_tokens import VoiceTokenManager
                import numpy as np
                manager = VoiceTokenManager()

                # Generate audio from text via voice tokens
                audio, sr = manager.generate_audio_output(
                    text=text,
                    speaker=speaker_id,
                    emotion=emotion,
                )

                # Convert to bytes
                import io
                import wave
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sr)
                    wf.writeframes((audio * 32768).astype(np.int16).tobytes())

                return buffer.getvalue(), sr

            except Exception as e:
                logger.warning(f"Voice token synthesis failed, falling back to Piper: {e}")

        # Fallback: return empty bytes to indicate Piper should be used
        return b"", 0


class MultimodalManager:
    """Manages multi-modal inputs for Jarvis.

    Coordinates processing of text, images, and audio.
    """

    def __init__(
        self,
        llm_client: Any,
        whisper_model: str = "base",
        vision_model: str | None = None,
    ):
        """Initialize the multimodal manager.

        Args:
            llm_client: The LLM client for vision processing
            whisper_model: Whisper model for audio transcription
            vision_model: Optional vision-capable model for image analysis
        """
        self.vision = VisionProcessor(llm_client, vision_model=vision_model)
        self.audio = AudioProcessor(whisper_model)
        self.llm = llm_client

    async def process_input(
        self,
        input_data: MultimodalInput,
    ) -> dict[str, Any]:
        """Process a multi-modal input.

        Args:
            input_data: The multi-modal input to process

        Returns:
            Processed results for each modality
        """
        results = {
            "modality_type": input_data.modality_type.value,
            "timestamp": input_data.timestamp.isoformat(),
            "processed": {},
        }

        # Process text
        if input_data.text:
            results["processed"]["text"] = input_data.text

        # Process images
        if input_data.has_images():
            image_results = []
            for i, img in enumerate(input_data.images):
                if isinstance(img, dict):
                    img_data = img.get("data") or img.get("path")
                    question = img.get("question")
                else:
                    img_data = img
                    question = None

                analysis = await self.vision.analyze_image(
                    img_data,
                    question=question,
                )
                image_results.append({
                    "index": i,
                    "analysis": analysis,
                })
            results["processed"]["images"] = image_results

        # Process audio
        if input_data.has_audio():
            audio_info = input_data.audio
            if isinstance(audio_info, dict):
                audio_path = audio_info.get("path")
                audio_data = audio_info.get("data")
            else:
                audio_path = audio_info
                audio_data = None

            if audio_data:
                transcription = await self.audio.transcribe_audio_data(audio_data)
            else:
                transcription = await self.audio.transcribe(audio_path)

            results["processed"]["audio"] = transcription

        return results

    def format_for_context(self, processed: dict[str, Any]) -> str:
        """Format processed multimodal input for LLM context.

        Args:
            processed: Processed multimodal results

        Returns:
            Formatted string for LLM context
        """
        lines = []

        if "text" in processed.get("processed", {}):
            lines.append(f"Text: {processed['processed']['text']}")

        if "images" in processed.get("processed", {}):
            lines.append("\nImage Analysis:")
            for img in processed["processed"]["images"]:
                lines.append(f"  Image {img['index'] + 1}: {img['analysis'][:500]}...")

        if "audio" in processed.get("processed", {}):
            audio = processed["processed"]["audio"]
            if audio.get("success"):
                lines.append(f"\nAudio Transcription: {audio['text']}")

        return "\n".join(lines)


# Tool for multi-modal inputs
class VisionTool:
    """Tool for Jarvis to process images."""

    name = "vision"
    description = "Analyze images and visual content. Use this tool when the user asks to analyze, describe, or understand an image. Provide the image path and specify the action (analyze, describe, compare)."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["analyze", "compare", "describe"],
                "description": "What to do with the image(s): 'analyze' for detailed analysis, 'describe' for description, 'compare' for comparing multiple images",
            },
            "image": {
                "type": "string",
                "description": "Path to image file (e.g., /home/user/.jarvis/image_cache/image_xxx.jpg) or base64 encoded image",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple images for comparison",
            },
            "question": {
                "type": "string",
                "description": "Specific question about the image",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    def __init__(self, vision_processor: VisionProcessor):
        self.vision = vision_processor

    async def execute(
        self,
        action: str,
        image: str | None = None,
        images: list[str] | None = None,
        question: str | None = None,
    ) -> str:
        """Execute vision action."""
        if action == "analyze" and image:
            return await self.vision.analyze_image(image, question=question)
        elif action == "compare" and images:
            return await self.vision.compare_images(images)
        elif action == "describe" and image:
            return await self.vision.analyze_image(image, prompt="Describe this image in detail.")
        else:
            return f"Invalid action or missing image for: {action}"


class AudioTranscriptionTool:
    """Tool for Jarvis to transcribe audio."""

    name = "transcribe"
    description = "Transcribe audio files to text"
    parameters = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to audio file",
            },
            "language": {
                "type": "string",
                "description": "Language code (optional)",
            },
        },
        "required": ["audio_path"],
    }
    requires_permission = False

    def __init__(self, audio_processor: AudioProcessor):
        self.audio = audio_processor

    async def execute(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Execute audio transcription."""
        result = await self.audio.transcribe(audio_path, language=language)

        if result.get("success"):
            return f"Transcription: {result['text']}"
        else:
            return f"Error: {result.get('error', 'Transcription failed')}"


_multimodal_manager: MultimodalManager | None = None
_mm_lock = threading.Lock()

def get_multimodal_manager(
    llm_client: Any = None,
    whisper_model: str = "base",
    vision_model: str | None = None,
) -> MultimodalManager:
    """Get or create the singleton MultimodalManager."""
    global _multimodal_manager

    if _multimodal_manager is None:
        with _mm_lock:
            if _multimodal_manager is None:
                if llm_client is None:
                    raise ValueError("llm_client required for first initialization")
                _multimodal_manager = MultimodalManager(
                    llm_client=llm_client,
                    whisper_model=whisper_model,
                    vision_model=vision_model,
                )
    return _multimodal_manager
