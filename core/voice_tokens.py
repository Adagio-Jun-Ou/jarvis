"""Voice Token Module for Jarvis.

Implements acoustic and semantic token processing for conversational voice AI,
inspired by Sesame AI's CSM (Conversational Speech Model) approach.

Architecture:
- Semantic tokens: Speaker-invariant phonetic/linguistic features (compressed)
- Acoustic tokens: Fine-grained speaker identity, timbre, prosody (RVQ codebooks)
- Token bridge: Converts between semantic and acoustic representations

Token Flow:
1. Audio Input → Semantic Tokens (understanding)
2. Semantic Tokens → LLM → Semantic Tokens (reasoning)  
3. Semantic Tokens → Acoustic Tokens → Audio Output (synthesis)
"""
import base64
import hashlib
import io
import json
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch not available, using fallback tokenizers")

try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.debug("Transformers not available, using fallback tokenizers")


class TokenType(Enum):
    SEMANTIC = "semantic"      # Speaker-invariant linguistic content
    ACOUSTIC = "acoustic"       # Speaker-specific acoustic details
    MIXED = "mixed"            # Combined semantic + acoustic


@dataclass
class VoiceTokens:
    """Container for voice tokens."""
    semantic_tokens: list[int] = field(default_factory=list)
    acoustic_tokens: list[list[int]] = field(default_factory=list)  # RVQ: [frame][codebook]
    sample_rate: int = 24000
    frame_rate: float = 12.5   # Hz, like Mimi
    n_codebooks: int = 8       # Number of RVQ codebooks
    speaker_id: Optional[str] = None
    duration_ms: float = 0.0
    text_transcription: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "semantic_tokens": self.semantic_tokens,
            "acoustic_tokens": self.acoustic_tokens,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "n_codebooks": self.n_codebooks,
            "speaker_id": self.speaker_id,
            "duration_ms": self.duration_ms,
            "text_transcription": self.text_transcription,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceTokens":
        return cls(
            semantic_tokens=data.get("semantic_tokens", []),
            acoustic_tokens=data.get("acoustic_tokens", []),
            sample_rate=data.get("sample_rate", 24000),
            frame_rate=data.get("frame_rate", 12.5),
            n_codebooks=data.get("n_codebooks", 8),
            speaker_id=data.get("speaker_id"),
            duration_ms=data.get("duration_ms", 0.0),
            text_transcription=data.get("text_transcription"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TokenSequence:
    """Interleaved token sequence for the backbone transformer."""
    tokens: list[int] = field(default_factory=list)
    token_types: list[TokenType] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)

    def add_text_tokens(self, tokens: list[int]):
        """Add text tokens to sequence."""
        pos = len(self.tokens)
        self.tokens.extend(tokens)
        self.token_types.extend([TokenType.SEMANTIC] * len(tokens))
        self.positions.extend(range(pos, pos + len(tokens)))

    def add_audio_tokens(self, semantic: list[int], acoustic: list[list[int]]):
        """Add audio tokens (semantic + acoustic) to sequence."""
        pos = len(self.tokens)
        # Add semantic tokens first (zeroth codebook)
        self.tokens.extend(semantic)
        self.token_types.extend([TokenType.SEMANTIC] * len(semantic))
        self.positions.extend(range(pos, pos + len(semantic)))

        # Add acoustic tokens (remaining codebooks)
        for frame_idx, frame_tokens in enumerate(acoustic):
            self.tokens.extend(frame_tokens)
            self.token_types.extend([TokenType.ACOUSTIC] * len(frame_tokens))
            pos = len(self.tokens)
            self.positions.extend(range(pos, pos + len(frame_tokens)))


class SemanticTokenizer:
    """Tokenizes audio into semantic tokens (speaker-invariant)."""

    # Semantic token vocabulary (phoneme-like units)
    SEMANTIC_VOCAB_SIZE = 1024
    PHONEME_TOKENS = 512     # Phoneme-level tokens
    PROSODY_TOKENS = 256     # Prosody patterns
    SENTIMENT_TOKENS = 128   # Emotional/sentiment markers
    SPECIAL_TOKENS = 128     # Special tokens (pause, breath, etc.)

    # Special token IDs
    PAD_TOKEN = 0
    BOS_TOKEN = 1            # Beginning of sequence
    EOS_TOKEN = 2            # End of sequence
    PAUSE_TOKEN = 3          # Pause/breath
    EMPHASIS_TOKEN = 4       # Emphasis marker
    QUESTION_TOKEN = 5       # Rising intonation
    EXCLAMATION_TOKEN = 6    # Exclamatory
    WHISPER_TOKEN = 7        # Whispered speech
    LAUGHTER_TOKEN = 8       # Laughter
    BREATH_TOKEN = 9         # Inhalation

    # Prosody tokens (256-512)
    RISING_TONE = 256
    FALLING_TONE = 257
    FLAT_TONE = 258
    STRESSED = 259

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None
        self._processor = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the semantic tokenization model."""
        if self._loaded:
            return

        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.info("Using fallback semantic tokenizer (no neural models available)")
            self._loaded = True
            return

        try:
            model_name = "facebook/hubert-base-ls960" if self.model_size == "base" else "facebook/hubert-large-ls960"

            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self._model = Wav2Vec2Model.from_pretrained(model_name)
            self._model.eval()
            self._loaded = True
            logger.info(f"Loaded semantic tokenizer: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}, using fallback")
            self._loaded = True

    def tokenize(self, audio_data: np.ndarray, sample_rate: int = 16000) -> list[int]:
        """Convert audio to semantic tokens.
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            List of semantic token IDs
        """
        self._load_model()

        # Resample if needed
        if sample_rate != 16000:
            audio_data = self._resample(audio_data, sample_rate, 16000)

        tokens = [self.BOS_TOKEN]

        if self._model is not None and TORCH_AVAILABLE:
            try:
                features = self._extract_features_neural(audio_data)
                tokens.extend(self._features_to_tokens(features))
            except Exception as e:
                logger.error(f"Semantic tokenization failed: {e}")
                tokens.extend(self._fallback_tokenize(audio_data))
        else:
            tokens.extend(self._fallback_tokenize(audio_data))

        tokens.append(self.EOS_TOKEN)
        return tokens

    def _extract_features_neural(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features using neural model."""
        import torch

        inputs = self._processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).numpy()

        return features

    def _features_to_tokens(self, features: np.ndarray) -> list[int]:
        """Convert continuous features to discrete tokens."""
        tokens = []

        for frame in features:
            # Quantize feature vector to token
            # Use hashing for deterministic quantization
            feature_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            token = int(feature_hash[:8], 16) % (self.PHONEME_TOKENS - 10) + 10

            # Add prosody based on feature statistics
            energy = np.mean(frame ** 2)
            if energy > 0.5:
                tokens.append(self.EMPHASIS_TOKEN)
            elif energy < 0.1:
                tokens.append(self.PAUSE_TOKEN)

            tokens.append(token)

        return tokens

    def _fallback_tokenize(self, audio_data: np.ndarray) -> list[int]:
        """Fallback tokenization using simple audio features."""
        tokens = []

        # Frame-based processing
        frame_size = 320  # ~20ms at 16kHz
        n_frames = len(audio_data) // frame_size

        for i in range(n_frames):
            frame = audio_data[i * frame_size:(i + 1) * frame_size]

            # Compute simple features
            energy = np.mean(frame ** 2)
            zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2

            # Map to token space
            energy_token = min(int(energy * 1000), 255) + 10
            min(int(zcr * 100), 127) + 266

            # Determine prosody
            if energy < 0.001:
                tokens.append(self.PAUSE_TOKEN)
            elif energy > 0.1:
                tokens.append(self.EMPHASIS_TOKEN)

            tokens.append(energy_token)

        return tokens

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            n_samples = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio), n_samples),
                np.arange(len(audio)),
                audio
            )

    def decode(self, tokens: list[int]) -> dict:
        """Decode semantic tokens to metadata (not audio reconstruction)."""
        result = {
            "phoneme_count": 0,
            "pause_count": 0,
            "emphasis_count": 0,
            "special_tokens": [],
        }

        for token in tokens:
            if token == self.PAUSE_TOKEN:
                result["pause_count"] += 1
            elif token == self.EMPHASIS_TOKEN:
                result["emphasis_count"] += 1
            elif token == self.LAUGHTER_TOKEN:
                result["special_tokens"].append("laughter")
            elif token == self.WHISPER_TOKEN:
                result["special_tokens"].append("whisper")
            elif token >= 10 and token < self.PHONEME_TOKENS:
                result["phoneme_count"] += 1

        return result


class AcousticTokenizer:
    """Tokenizes audio into acoustic tokens using Residual Vector Quantization."""

    N_CODEBOOKS = 8           # Number of RVQ codebooks
    CODEBOOK_SIZE = 1024      # Tokens per codebook
    FRAME_RATE = 12.5         # Hz (80ms frames, like Mimi)

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._codec = None
        self._loaded = False

    def _load_codec(self):
        """Load the neural audio codec for RVQ."""
        if self._loaded:
            return

        try:
            # Try to use EnCodec or DAC

            try:
                from encodec import EncodecModel
                self._codec = EncodecModel.encodec_model_24khz()
                self._codec.eval()
                self._codec.set_target_bandwidth(6.0)
                logger.info("Loaded EnCodec for acoustic tokenization")
            except ImportError:
                try:
                    # Try Descript Audio Codec
                    from dac import DAC
                    self._codec = DAC.load(self.model_path or "dac_44khz.pth")
                    self._codec.eval()
                    logger.info("Loaded DAC for acoustic tokenization")
                except ImportError:
                    logger.warning("No neural codec available, using fallback")

        except Exception as e:
            logger.warning(f"Failed to load acoustic codec: {e}")

        self._loaded = True

    def tokenize(self, audio_data: np.ndarray, sample_rate: int = 24000) -> list[list[int]]:
        """Convert audio to acoustic tokens (RVQ codebooks).
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            List of [codebook_0_token, codebook_1_token, ...] for each frame
        """
        self._load_codec()

        if self._codec is not None:
            return self._neural_tokenize(audio_data, sample_rate)
        else:
            return self._fallback_tokenize(audio_data, sample_rate)

    def _neural_tokenize(self, audio_data: np.ndarray, sample_rate: int) -> list[list[int]]:
        """Tokenize using neural audio codec."""
        import torch

        if sample_rate != 24000:
            audio_data = self._resample(audio_data, sample_rate, 24000)

        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            if hasattr(self._codec, 'encode'):
                encoded = self._codec.encode(audio_tensor)
                if isinstance(encoded, (list, tuple)) and len(encoded) > 0:
                    first = encoded[0]
                    if isinstance(first, (list, tuple)) and len(first) > 0 and hasattr(first[0], 'cpu'):
                        codes = first[0]
                    elif hasattr(first, 'cpu'):
                        codes = first
                    else:
                        codes = None
                elif hasattr(encoded, 'codes'):
                    codes = encoded.codes
                elif hasattr(encoded, 'cpu'):
                    codes = encoded
                else:
                    codes = None
            else:
                codes = self._extract_codes(audio_tensor)

        tokens = []
        if codes is not None:
            if hasattr(codes, 'cpu'):
                codes_np = codes.detach().cpu().numpy()
            elif isinstance(codes, (list, tuple)):
                import numpy as np2
                try:
                    if codes and hasattr(codes[0], 'cpu'):
                        codes_np = torch.stack(list(codes)).detach().cpu().numpy()
                    else:
                        codes_np = np2.array(codes, dtype=np.int64)
                except (ValueError, RuntimeError):
                    tokens = []
                    codes = None
            else:
                codes_np = codes

            if codes is None:
                return tokens

            if hasattr(codes_np, 'shape'):
                while codes_np.ndim > 3:
                    codes_np = codes_np[0]
                if codes_np.ndim == 3 and codes_np.shape[0] == 1:
                    codes_np = codes_np[0]

                n_codebooks = min(self.N_CODEBOOKS, codes_np.shape[0] if codes_np.ndim >= 2 else 1)
                for frame_idx in range(codes_np.shape[-1]):
                    frame_tokens = [int(codes_np[cb, frame_idx]) for cb in range(n_codebooks)]
                    tokens.append(frame_tokens)

        return tokens

    def _extract_codes(self, audio_tensor: "torch.Tensor") -> "torch.Tensor":
        """Extract quantized codes from audio using codec internals."""
        import torch

        # Generic code extraction for any RVQ-based codec
        with torch.no_grad():
            # Get encoder output
            if hasattr(self._codec, 'encoder'):
                features = self._codec.encoder(audio_tensor)
            else:
                features = audio_tensor

            # Quantize with RVQ
            if hasattr(self._codec, 'quantizer'):
                quantized, codes, _ = self._codec.quantizer(features)
                return codes[0]  # [n_codebooks, n_frames]
            else:
                # Create pseudo-codes from feature dimensions
                n_frames = features.shape[-1]
                codes = torch.randint(0, self.CODEBOOK_SIZE, (self.N_CODEBOOKS, n_frames))
                return codes

    def _fallback_tokenize(self, audio_data: np.ndarray, sample_rate: int) -> list[list[int]]:
        """Fallback acoustic tokenization using audio features."""
        # Resample for consistent frame rate
        int(sample_rate * self.FRAME_RATE / (sample_rate / 1000 * self.FRAME_RATE))

        # Frame size for target frame rate
        frame_samples = int(sample_rate / self.FRAME_RATE)
        n_frames = len(audio_data) // frame_samples

        tokens = []
        for i in range(n_frames):
            frame = audio_data[i * frame_samples:(i + 1) * frame_samples]
            frame_tokens = []

            # Extract features for each codebook
            for cb in range(self.N_CODEBOOKS):
                # Different feature for each codebook
                if cb == 0:
                    # Energy
                    feature = np.mean(frame ** 2)
                    token = int(feature * 32768) % self.CODEBOOK_SIZE
                elif cb == 1:
                    # Zero crossing rate
                    feature = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
                    token = int(feature * 10000) % self.CODEBOOK_SIZE
                elif cb == 2:
                    # Spectral centroid approximation
                    fft = np.abs(np.fft.rfft(frame))
                    freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)
                    feature = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)
                    token = int(feature) % self.CODEBOOK_SIZE
                elif cb == 3:
                    # Spectral flatness
                    fft = np.abs(np.fft.rfft(frame)) + 1e-8
                    feature = np.exp(np.mean(np.log(fft))) / np.mean(fft)
                    token = int(feature * 1000) % self.CODEBOOK_SIZE
                else:
                    # Higher codebooks: phase and fine details
                    phase = np.angle(np.fft.fft(frame))
                    idx = cb * (len(phase) // self.N_CODEBOOKS)
                    feature = np.mean(phase[max(0,idx-5):idx+5]) if idx < len(phase) else 0
                    token = int((feature + np.pi) / (2 * np.pi) * self.CODEBOOK_SIZE) % self.CODEBOOK_SIZE

                frame_tokens.append(token)

            tokens.append(frame_tokens)

        return tokens

    def decode(self, tokens: list[list[int]], sample_rate: int = 24000) -> np.ndarray:
        """Decode acoustic tokens back to audio."""
        self._load_codec()

        if self._codec is not None and hasattr(self._codec, 'decode'):
            import torch

            codes = torch.tensor([tokens], dtype=torch.long)
            if codes.dim() == 3:
                codes = codes.permute(0, 2, 1)

            with torch.no_grad():
                try:
                    audio = self._codec.decode([(codes, None)])
                except Exception as e:
                    logger.debug(f"Primary codec decode failed: {e}")
                    try:
                        audio = self._codec.decode(codes)
                    except Exception as e2:
                        logger.debug(f"Secondary codec decode failed: {e2}")
                        n_frames = len(tokens)
                        frame_samples = int(sample_rate / self.FRAME_RATE)
                        return np.zeros(n_frames * frame_samples)

            return audio.squeeze().cpu().numpy()
        else:
            # Fallback: generate white noise shaped by tokens
            n_frames = len(tokens)
            frame_samples = int(sample_rate / self.FRAME_RATE)
            audio = np.zeros(n_frames * frame_samples)

            for i, frame_tokens in enumerate(tokens):
                # Use first few tokens to shape noise
                energy = frame_tokens[0] / self.CODEBOOK_SIZE if frame_tokens else 0.5
                noise = np.random.randn(frame_samples) * energy

                start = i * frame_samples
                audio[start:start + frame_samples] = noise

            return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            ratio = target_sr / orig_sr
            n_samples = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio), n_samples),
                np.arange(len(audio)),
                audio
            )


class VoiceTokenBridge:
    """Bridge between semantic and acoustic tokens."""

    def __init__(self):
        self.semantic_tokenizer = SemanticTokenizer()
        self.acoustic_tokenizer = AcousticTokenizer()
        self._speaker_profiles: dict[str, dict] = {}
        self._MAX_SPEAKER_PROFILES = 20

    def encode(self, audio_data: np.ndarray, sample_rate: int = 16000) -> VoiceTokens:
        """Encode audio to both semantic and acoustic tokens."""
        # Get semantic tokens (speaker-invariant)
        semantic_tokens = self.semantic_tokenizer.tokenize(audio_data, sample_rate)

        # Get acoustic tokens (speaker-specific)
        acoustic_tokens = self.acoustic_tokenizer.tokenize(audio_data, sample_rate)

        # Calculate duration
        duration_ms = len(audio_data) / sample_rate * 1000

        return VoiceTokens(
            semantic_tokens=semantic_tokens,
            acoustic_tokens=acoustic_tokens,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )

    def decode(self, voice_tokens: VoiceTokens, use_acoustic: bool = True) -> np.ndarray:
        """Decode voice tokens back to audio."""
        if use_acoustic and voice_tokens.acoustic_tokens:
            return self.acoustic_tokenizer.decode(
                voice_tokens.acoustic_tokens,
                voice_tokens.sample_rate
            )
        else:
            # Generate from semantic tokens only (speaker-invariant)
            return self._synthesize_from_semantic(voice_tokens.semantic_tokens)

    def _synthesize_from_semantic(self, tokens: list[int]) -> np.ndarray:
        """Synthesize audio from semantic tokens (basic)."""
        # Map semantic tokens to simple audio features
        sample_rate = 24000
        frame_duration = 0.08  # 12.5 Hz = 80ms frames
        frame_samples = int(sample_rate * frame_duration)

        audio = np.zeros(len(tokens) * frame_samples, dtype=np.float32)

        for i, token in enumerate(tokens):
            if token == SemanticTokenizer.PAUSE_TOKEN:
                # Silence
                continue
            elif token == SemanticTokenizer.EMPHASIS_TOKEN:
                # Louder frame
                audio[i*frame_samples:(i+1)*frame_samples] = np.random.randn(frame_samples) * 0.3
            elif token >= 10 and token < SemanticTokenizer.PHONEME_TOKENS:
                # Phoneme-like: generate simple tone
                freq = 200 + (token - 10) * 5  # Map token to frequency
                t = np.arange(frame_samples) / sample_rate
                audio[i*frame_samples:(i+1)*frame_samples] = np.sin(2 * np.pi * freq * t) * 0.2

        return audio

    def extract_speaker_embedding(self, acoustic_tokens: list[list[int]]) -> str:
        """Extract speaker identity from acoustic tokens."""
        # Create a hash of acoustic characteristics
        if not acoustic_tokens:
            return "unknown"

        # Use higher codebooks which encode more speaker-specific info
        speaker_features = []
        for frame in acoustic_tokens:
            if len(frame) >= 4:
                # Ensure tokens are in valid byte range
                speaker_features.extend([t % 256 for t in frame[-4:]])  # Last 4 codebooks

        if not speaker_features:
            return "unknown"

        speaker_hash = hashlib.sha256(bytes(speaker_features[:256])).hexdigest()[:16]  # Limit to 256 bytes
        return f"speaker_{speaker_hash}"

    def transfer_voice(
        self,
        source_tokens: VoiceTokens,
        target_acoustic: VoiceTokens,
    ) -> VoiceTokens:
        """Transfer voice from target to source (voice conversion)."""
        # Keep semantic content from source
        # Replace acoustic tokens with target speaker's characteristics

        # Extract target speaker's acoustic patterns
        target_speaker_id = self.extract_speaker_embedding(target_acoustic.acoustic_tokens)

        return VoiceTokens(
            semantic_tokens=source_tokens.semantic_tokens,
            acoustic_tokens=target_acoustic.acoustic_tokens,  # Voice transfer
            sample_rate=source_tokens.sample_rate,
            frame_rate=source_tokens.frame_rate,
            speaker_id=target_speaker_id,
            duration_ms=source_tokens.duration_ms,
        )

    def register_speaker(self, name: str, audio_samples: list[tuple[np.ndarray, int]]):
        """Register a speaker profile from audio samples."""
        all_acoustic_tokens = []

        for audio, sr in audio_samples:
            tokens = self.acoustic_tokenizer.tokenize(audio, sr)
            all_acoustic_tokens.extend(tokens)

        speaker_id = self.extract_speaker_embedding(all_acoustic_tokens)

        if len(self._speaker_profiles) >= self._MAX_SPEAKER_PROFILES:
            oldest_key = next(iter(self._speaker_profiles))
            del self._speaker_profiles[oldest_key]
        self._speaker_profiles[name] = {
            "speaker_id": speaker_id,
            "n_samples": len(audio_samples),
            "acoustic_tokens": all_acoustic_tokens[-100:],
        }

        logger.info(f"Registered speaker: {name} (id: {speaker_id})")

    def get_speaker_acoustic(self, name: str) -> Optional[list[list[int]]]:
        """Get acoustic tokens for a registered speaker."""
        profile = self._speaker_profiles.get(name)
        if profile:
            return profile.get("acoustic_tokens")
        return None


class VoiceTokenManager:
    """Manages voice token processing for Jarvis."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "voice_tokens"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.bridge = VoiceTokenBridge()
        self._token_cache: dict[str, VoiceTokens] = {}
        self._MAX_TOKEN_CACHE = 200
        self._conversation_history: list[TokenSequence] = []
        self._MAX_CONVERSATION_HISTORY = 100
        
        # Stats tracking
        self._input_count = 0
        self._output_count = 0
        self._semantic_tokens_processed = 0
        self._acoustic_tokens_processed = 0

    def get_stats(self) -> dict:
        """Get voice token processing statistics."""
        return {
            "inputs_processed": self._input_count,
            "outputs_generated": self._output_count,
            "semantic_tokens": self._semantic_tokens_processed,
            "acoustic_frames": self._acoustic_tokens_processed,
        }

    def process_audio_input(
        self,
        audio_data: bytes | np.ndarray,
        sample_rate: int = 16000,
        format: str = "wav",
    ) -> VoiceTokens:
        """Process audio input into voice tokens."""
        # Convert bytes to numpy if needed
        if isinstance(audio_data, bytes):
            audio_data = self._bytes_to_numpy(audio_data, format)

        # Encode to tokens
        voice_tokens = self.bridge.encode(audio_data, sample_rate)

        # Add to conversation history
        sequence = TokenSequence()
        sequence.add_audio_tokens(voice_tokens.semantic_tokens, voice_tokens.acoustic_tokens)
        self._conversation_history.append(sequence)
        if len(self._conversation_history) > self._MAX_CONVERSATION_HISTORY:
            self._conversation_history = self._conversation_history[-self._MAX_CONVERSATION_HISTORY:]

        # Update stats
        self._input_count += 1
        self._semantic_tokens_processed += len(voice_tokens.semantic_tokens)
        self._acoustic_tokens_processed += len(voice_tokens.acoustic_tokens)

        return voice_tokens

    def generate_audio_output(
        self,
        text: str,
        speaker: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """Generate audio output from text using voice tokens."""
        # Convert text to semantic tokens
        semantic_tokens = self._text_to_semantic_tokens(text, emotion)

        # Get speaker acoustic tokens if available
        acoustic_tokens = None
        if speaker:
            acoustic_tokens = self.bridge.get_speaker_acoustic(speaker)

        # Generate acoustic tokens from semantic
        if acoustic_tokens is None:
            # Use default synthesis
            acoustic_tokens = self._generate_acoustic_from_semantic(semantic_tokens)

        # Create voice tokens
        voice_tokens = VoiceTokens(
            semantic_tokens=semantic_tokens,
            acoustic_tokens=acoustic_tokens,
            speaker_id=speaker,
        )

        # Decode to audio
        audio = self.bridge.decode(voice_tokens)

        # Update stats
        self._output_count += 1

        return audio, 24000

    def _bytes_to_numpy(self, audio_bytes: bytes, format: str) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            return audio
        except ImportError:
            # Fallback for WAV
            import wave
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
                return audio

    def _text_to_semantic_tokens(self, text: str, emotion: Optional[str] = None) -> list[int]:
        """Convert text to semantic tokens."""
        tokens = [SemanticTokenizer.BOS_TOKEN]

        # Add emotion token if specified
        emotion_tokens = {
            "happy": SemanticTokenizer.LAUGHTER_TOKEN,
            "excited": SemanticTokenizer.EXCLAMATION_TOKEN,
            "question": SemanticTokenizer.QUESTION_TOKEN,
            "whisper": SemanticTokenizer.WHISPER_TOKEN,
        }
        if emotion and emotion in emotion_tokens:
            tokens.append(emotion_tokens[emotion])

        # Convert text characters to tokens (simplified)
        for char in text:
            if char.isspace():
                tokens.append(SemanticTokenizer.PAUSE_TOKEN)
            elif char in "!?.":
                if char == "?":
                    tokens.append(SemanticTokenizer.QUESTION_TOKEN)
                elif char == "!":
                    tokens.append(SemanticTokenizer.EXCLAMATION_TOKEN)
                else:
                    tokens.append(SemanticTokenizer.PAUSE_TOKEN)
            else:
                # Hash character to token
                token = (hash(char) % (SemanticTokenizer.PHONEME_TOKENS - 10)) + 10
                tokens.append(token)

        tokens.append(SemanticTokenizer.EOS_TOKEN)
        return tokens

    def _generate_acoustic_from_semantic(self, semantic_tokens: list[int]) -> list[list[int]]:
        """Generate acoustic tokens conditioned on semantic tokens."""
        acoustic_tokens = []

        for sem_token in semantic_tokens:
            # Base acoustic frame from semantic token
            frame = []

            # Codebook 0: Close mapping to semantic
            frame.append(sem_token % AcousticTokenizer.CODEBOOK_SIZE)

            # Higher codebooks: Generate with increasing variation
            for cb in range(1, AcousticTokenizer.N_CODEBOOKS):
                # Use semantic token and codebook index to generate
                # In a real model, this would come from a learned distribution
                base = (sem_token * (cb + 1) + cb * 100) % AcousticTokenizer.CODEBOOK_SIZE
                variation = np.random.randint(-50, 50)
                frame.append(max(0, min(AcousticTokenizer.CODEBOOK_SIZE - 1, base + variation)))

            acoustic_tokens.append(frame)

        return acoustic_tokens

    def save_tokens(self, tokens: VoiceTokens, name: str):
        """Save voice tokens to storage."""
        path = self.storage_path / f"{name}.json"
        atomic_write_json(path, tokens.to_dict())

    def load_tokens(self, name: str) -> Optional[VoiceTokens]:
        """Load voice tokens from storage."""
        path = self.storage_path / f"{name}.json"
        if path.exists():
            data = read_json_with_integrity(path)
            return VoiceTokens.from_dict(data) if data else None
        return None


# Tool wrapper for integration with Jarvis
class VoiceTokenTool:
    """Tool for Jarvis to process voice tokens."""

    name = "voice_tokens"
    description = "Process and generate voice using semantic and acoustic tokens for natural conversational speech"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["encode", "decode", "register_speaker", "list_speakers", "transfer_voice"],
                "description": "Action to perform",
            },
            "audio_data": {
                "type": "string",
                "description": "Base64 encoded audio data (for encode)",
            },
            "speaker": {
                "type": "string",
                "description": "Speaker name or ID",
            },
            "text": {
                "type": "string",
                "description": "Text to convert to voice tokens (for decode/generate)",
            },
            "emotion": {
                "type": "string",
                "description": "Emotion for speech (happy, question, whisper, etc.)",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    def __init__(self, manager: Optional[VoiceTokenManager] = None):
        self.manager = manager or VoiceTokenManager()

    async def execute(
        self,
        action: str,
        audio_data: Optional[str] = None,
        speaker: Optional[str] = None,
        text: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> str:
        """Execute voice token action."""
        if action == "encode":
            if not audio_data:
                return "Error: audio_data required for encode action"

            audio_bytes = base64.b64decode(audio_data)
            tokens = self.manager.process_audio_input(audio_bytes)

            return json.dumps({
                "semantic_token_count": len(tokens.semantic_tokens),
                "acoustic_frame_count": len(tokens.acoustic_tokens),
                "duration_ms": tokens.duration_ms,
                "speaker_id": tokens.speaker_id,
            }, indent=2)

        elif action == "decode":
            if text:
                audio, sr = self.manager.generate_audio_output(text, speaker, emotion)
                audio_b64 = base64.b64encode(audio.tobytes()).decode()
                return json.dumps({
                    "audio_data": audio_b64,
                    "sample_rate": sr,
                    "duration_ms": len(audio) / sr * 1000,
                })
            else:
                return "Error: text required for decode/generate action"

        elif action == "register_speaker":
            if not speaker:
                return "Error: speaker name required"
            return f"Speaker '{speaker}' registration requires audio samples. Use audio input."

        elif action == "list_speakers":
            speakers = list(self.manager.bridge._speaker_profiles.keys())
            return json.dumps({"speakers": speakers}, indent=2)

        elif action == "transfer_voice":
            return "Voice transfer requires source and target audio. Use audio input."

        else:
            return f"Unknown action: {action}"


# Singleton manager
_voice_token_manager: Optional[VoiceTokenManager] = None
_voice_token_lock = threading.Lock()


def get_voice_token_manager() -> VoiceTokenManager:
    """Get or create the voice token manager singleton."""
    global _voice_token_manager
    if _voice_token_manager is None:
        with _voice_token_lock:
            if _voice_token_manager is None:
                _voice_token_manager = VoiceTokenManager()
    return _voice_token_manager
