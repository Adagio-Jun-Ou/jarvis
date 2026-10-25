"""Token processing tools for Jarvis.

Exposes semantic, acoustic, and visual token processing as user-facing tools.
"""
from pathlib import Path
from typing import Optional

from core.helpers import safe_error_message
from tools.base import Tool


class SpeechTokenizeTool(Tool):
    """Tokenize speech audio into semantic and acoustic tokens."""

    name = "speech_tokenize"
    description = "Convert speech audio to semantic and acoustic tokens for analysis"
    parameters = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to audio file (wav, mp3, etc.)"
            },
            "analyze": {
                "type": "boolean",
                "description": "Whether to provide detailed analysis of tokens"
            }
        },
        "required": ["audio_path"]
    }
    requires_permission = False

    def __init__(self):
        self.tokenizer = None

    async def _load_tokenizer(self):
        if self.tokenizer is None:
            from speech.tokenizer import SpeechTokenizer
            from speech.codec import CodecConfig
            self.tokenizer = SpeechTokenizer(CodecConfig())
            await self.tokenizer.load()
        return self.tokenizer

    async def execute(self, audio_path: str, analyze: bool = False) -> str:
        """Tokenize speech audio."""
        try:
            tokenizer = await self._load_tokenizer()

            # Tokenize the audio
            tokens = await tokenizer.encode_file(audio_path)

            result = [
                "🎵 Speech Tokenization Results:",
                "",
                f"**Duration:** {tokens.duration:.2f} seconds",
                f"**Sample Rate:** {tokens.sample_rate} Hz",
                "",
                f"**Semantic Tokens:** {len(tokens.semantic)} tokens",
                f"  - Shape: {tokens.semantic.shape}",
                f"  - Range: [{tokens.semantic.min()}, {tokens.semantic.max()}]",
                "",
                f"**Acoustic Tokens:** {tokens.acoustic.shape}",
                f"  - Codebooks: {tokens.acoustic.shape[0]}",
                f"  - Frames: {tokens.acoustic.shape[1]}",
                f"  - Frame Rate: ~{tokens.acoustic.shape[1] / tokens.duration:.1f} Hz",
            ]

            if analyze:
                import numpy as np
                result.extend([
                    "",
                    "**Analysis:**",
                    f"  - Unique semantic tokens: {len(np.unique(tokens.semantic))}",
                    f"  - Most common semantic: {np.bincount(tokens.semantic).argmax()}",
                    f"  - Acoustic variance: {tokens.acoustic.var():.2f}",
                ])

                # Token distribution
                np.bincount(tokens.semantic)
                result.append(f"  - Semantic distribution: {dict(zip(*np.unique(tokens.semantic, return_counts=True)))[:5]}...")

            return "\n".join(result)

        except Exception as e:
            return f"Error tokenizing audio: {safe_error_message(e, 'Token processing failed')}\n\nNote: Ensure audio file exists and speech module is properly installed."


class SpeechSynthesizeTool(Tool):
    """Synthesize speech from tokens."""

    name = "speech_synthesize"
    description = "Synthesize speech audio from semantic and acoustic tokens"
    parameters = {
        "type": "object",
        "properties": {
            "semantic_tokens": {
                "type": "string",
                "description": "Comma-separated semantic token values"
            },
            "duration": {
                "type": "number",
                "description": "Duration in seconds (default: 2.0)"
            },
            "output_path": {
                "type": "string",
                "description": "Path to save output audio file"
            }
        },
        "required": []
    }
    requires_permission = False

    def __init__(self):
        self.tokenizer = None

    async def _load_tokenizer(self):
        if self.tokenizer is None:
            from speech.tokenizer import SpeechTokenizer
            from speech.codec import CodecConfig
            self.tokenizer = SpeechTokenizer(CodecConfig())
            await self.tokenizer.load()
        return self.tokenizer

    async def execute(
        self,
        semantic_tokens: Optional[str] = None,
        duration: float = 2.0,
        output_path: Optional[str] = None
    ) -> str:
        """Synthesize speech from tokens."""
        try:
            import numpy as np
            from speech.tokenizer import SpeechTokens

            tokenizer = await self._load_tokenizer()
            config = tokenizer.config

            # Generate tokens if not provided
            if semantic_tokens:
                semantic = np.array([int(t.strip()) for t in semantic_tokens.split(",")], dtype=np.int32)
            else:
                # Generate random tokens for demonstration
                num_tokens = int(duration * config.semantic_frame_rate)
                semantic = np.random.randint(0, config.semantic_vocab_size, num_tokens, dtype=np.int32)

            # Generate corresponding acoustic tokens
            num_frames = int(duration * config.acoustic_frame_rate)
            acoustic = np.random.randint(0, config.acoustic_vocab_size, (config.num_codebooks, num_frames), dtype=np.int32)

            tokens = SpeechTokens(
                semantic=semantic,
                acoustic=acoustic,
                duration=duration,
                sample_rate=config.sample_rate
            )

            # Decode to audio
            await tokenizer.decode(tokens)

            # Save to file
            if output_path is None:
                output_path = str(Path.home() / ".jarvis" / "synthesized_speech.wav")

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            await tokenizer.decode_to_file(tokens, output_path)

            return f"🔊 Speech Synthesis Complete:\n\n- Duration: {duration:.2f}s\n- Semantic tokens: {len(semantic)}\n- Acoustic frames: {num_frames}\n- Output: {output_path}"

        except Exception as e:
            return f"Error synthesizing speech: {safe_error_message(e, 'Token processing failed')}"


class ImageTokenizeTool(Tool):
    """Tokenize images into visual tokens."""

    name = "image_tokenize"
    description = "Convert an image into visual tokens for analysis"
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to image file or base64 data"
            },
            "analyze": {
                "type": "boolean",
                "description": "Whether to provide detailed token analysis"
            }
        },
        "required": ["image_path"]
    }
    requires_permission = False

    def __init__(self):
        self.tokenizer = None

    async def _load_tokenizer(self):
        if self.tokenizer is None:
            from core.vision_tokens import VisualTokenizer, VisualTokenConfig
            self.tokenizer = VisualTokenizer(VisualTokenConfig())
            await self.tokenizer.load()
        return self.tokenizer

    async def execute(self, image_path: str, analyze: bool = False) -> str:
        """Tokenize an image."""
        try:
            import numpy as np

            tokenizer = await self._load_tokenizer()

            # Handle base64 or file path
            if image_path.startswith("data:"):
                # Base64 encoded
                import base64
                from io import BytesIO
                from PIL import Image
                header, data = image_path.split(",", 1)
                img_data = base64.b64decode(data)
                image = Image.open(BytesIO(img_data))
            elif len(image_path) > 200:
                # Likely base64 without header
                import base64
                from io import BytesIO
                from PIL import Image
                img_data = base64.b64decode(image_path)
                image = Image.open(BytesIO(img_data))
            else:
                image = image_path  # File path

            # Tokenize
            tokens = await tokenizer.encode(image)

            result = [
                "🖼️ Visual Tokenization Results:",
                "",
                f"**Image Size:** {tokens.original_size[0]}×{tokens.original_size[1]}",
                f"**Token Grid:** {tokens.spatial_grid[0]}×{tokens.spatial_grid[1]}",
                "",
                f"**Visual Tokens:** {len(tokens.tokens)} tokens",
                f"  - Shape: {tokens.tokens.shape}",
                f"  - Range: [{tokens.tokens.min()}, {tokens.tokens.max()}]",
                f"  - Hash: {tokens.image_hash}",
                "",
                f"**Patch Embeddings:** {tokens.patch_embeddings.shape}",
            ]

            if analyze:
                unique_tokens = len(np.unique(tokens.tokens))
                token_counts = np.bincount(tokens.tokens.flatten())

                result.extend([
                    "",
                    "**Analysis:**",
                    f"  - Unique tokens: {unique_tokens}",
                    f"  - Most common token: {token_counts.argmax()}",
                    f"  - Token entropy: {-np.sum((token_counts/token_counts.sum()) * np.log2(token_counts/token_counts.sum() + 1e-10)):.2f} bits",
                ])

                # Spatial distribution
                grid = tokens.tokens.reshape(tokens.spatial_grid)
                result.append(f"  - Grid mean: {grid.mean():.2f}")
                result.append(f"  - Grid std: {grid.std():.2f}")

            return "\n".join(result)

        except Exception as e:
            return f"Error tokenizing image: {safe_error_message(e, 'Token processing failed')}"


class ConversationalSpeechTool(Tool):
    """Use the conversational speech model for end-to-end speech processing."""

    name = "conversational_speech"
    description = "Process speech through the conversational model with token-based understanding"
    parameters = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to audio file with speech"
            },
            "mode": {
                "type": "string",
                "enum": ["analyze", "respond"],
                "description": "analyze = tokenize only, respond = generate response"
            }
        },
        "required": ["audio_path"]
    }
    requires_permission = False

    def __init__(self):
        self.model = None

    async def _load_model(self):
        if self.model is None:
            from speech.conversation import ConversationalSpeechModel, ConversationConfig
            self.model = ConversationalSpeechModel(ConversationConfig())
            await self.model.load()
        return self.model

    async def execute(self, audio_path: str, mode: str = "analyze") -> str:
        """Process speech conversationally."""
        try:
            import numpy as np

            # Load audio
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000)
            except ImportError:
                # Fallback
                import wave
                with wave.open(audio_path, 'rb') as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768

            if mode == "analyze":
                # Just tokenize
                model = await self._load_model()
                tokens = await model.tokenizer.encode(audio, sr)

                return f"""🎤 Conversational Speech Analysis:

**Input Audio:**
- Duration: {tokens.duration:.2f}s
- Sample Rate: {tokens.sample_rate} Hz

**Token Representation:**
- Semantic tokens: {len(tokens.semantic)} (linguistic content)
- Acoustic tokens: {tokens.acoustic.shape} (voice characteristics)

**Speaker Profile:**
- Embedding extracted: {model.speaker_manager.extract_speaker_embedding(tokens).shape}

Ready for conversational response with mode='respond'."""

            else:
                # Full conversational processing
                model = await self._load_model()
                text, audio_response = await model.conversation_turn(audio, 16000)

                return f"""🎤 Conversational Speech Response:

**Transcribed:** "{text[:200]}{'...' if len(text) > 200 else ''}"

**Response Generated:**
- Audio shape: {audio_response.shape}
- Duration: {len(audio_response) / 16000:.2f}s

**Conversation History:**
- Turns: {len(model.conversation_history)}"""

        except Exception as e:
            return f"Error processing conversational speech: {safe_error_message(e, 'Token processing failed')}"


def get_token_tools():
    """Get all token processing tools."""
    return [
        SpeechTokenizeTool(),
        SpeechSynthesizeTool(),
        ImageTokenizeTool(),
        ConversationalSpeechTool(),
    ]
