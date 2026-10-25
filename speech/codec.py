"""Neural audio codecs for speech tokenization.

Implements two types of codecs:
1. Semantic Codec: Encodes linguistic content (phonemes, words)
2. Acoustic Codec: Encodes speaker characteristics, prosody, emotion
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class CodecConfig:
    """Configuration for audio codecs."""
    sample_rate: int = 16000
    # Semantic codec settings
    semantic_vocab_size: int = 500  # Number of semantic token types
    semantic_frame_rate: int = 50   # Tokens per second for semantic
    # Acoustic codec settings
    acoustic_vocab_size: int = 1024  # Number of acoustic codebook entries
    num_codebooks: int = 8           # Number of parallel codebooks
    acoustic_frame_rate: int = 75    # Tokens per second for acoustic


class SemanticCodec:
    """Encodes speech into semantic tokens capturing linguistic content.

    Uses a pre-trained model (HuBERT/wav2vec2) to extract semantic
    representations, then quantizes them into discrete tokens.
    """

    def __init__(self, config: Optional[CodecConfig] = None, model_path: Optional[str] = None):
        self.config = config or CodecConfig()
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._loaded = False

    async def load(self) -> bool:
        """Load the semantic codec model."""
        try:
            # Try to load transformers-based model
            from transformers import AutoModel, AutoProcessor

            model_name = self.model_path or "facebook/hubert-large-ls960-ft"

            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._loaded = True

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            return True
        except ImportError:
            print("Warning: transformers not installed. Using fallback codec.")
            return self._load_fallback()
        except Exception as e:
            print(f"Error loading semantic codec: {e}")
            return self._load_fallback()

    def _load_fallback(self) -> bool:
        """Load a fallback implementation using lighter dependencies."""
        self._loaded = True
        return True

    async def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio into semantic tokens.

        Args:
            audio: Audio waveform as numpy array (shape: [samples])

        Returns:
            Semantic tokens as numpy array (shape: [num_tokens])
        """
        if not self._loaded:
            await self.load()

        if self._model is not None:
            return await self._encode_with_model(audio)
        else:
            return self._encode_fallback(audio)

    async def _encode_with_model(self, audio: np.ndarray) -> np.ndarray:
        """Encode using the loaded model."""
        import torch

        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state

        # Quantize to discrete tokens
        # Use k-means style quantization on hidden states
        hidden_states = hidden_states.cpu().numpy()[0]

        # Simple quantization: cluster into vocab_size bins
        tokens = self._quantize(hidden_states)

        return tokens

    def _quantize(self, features: np.ndarray) -> np.ndarray:
        """Quantize continuous features into discrete tokens."""
        # Normalize features
        features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

        # Use random projection for simple quantization
        # In production, use learned codebook
        rng = np.random.RandomState(42)  # Local RNG — avoids global state pollution
        codebook = rng.randn(self.config.semantic_vocab_size, features.shape[-1])
        codebook = codebook / (np.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)

        # Find nearest codebook entry for each frame
        similarities = features @ codebook.T
        tokens = np.argmax(similarities, axis=-1)

        return tokens.astype(np.int32)

    def _encode_fallback(self, audio: np.ndarray) -> np.ndarray:
        """Fallback encoding using simple feature extraction."""
        # Compute frame-level features
        frame_size = self.config.sample_rate // self.config.semantic_frame_rate
        num_frames = len(audio) // frame_size

        tokens = []
        for i in range(num_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            # Simple energy and zero-crossing rate as features
            energy = np.sum(frame ** 2)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

            # Hash to token
            feature_hash = hash((int(energy * 1000), int(zcr * 1000)))
            token = abs(feature_hash) % self.config.semantic_vocab_size
            tokens.append(token)

        return np.array(tokens, dtype=np.int32)

    async def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode semantic tokens back to audio features.

        Note: This produces a rough approximation. For proper synthesis,
        use the full pipeline with AcousticCodec.

        Args:
            tokens: Semantic tokens (shape: [num_tokens])

        Returns:
            Audio features (not full audio)
        """
        # Semantic tokens alone cannot reconstruct audio
        # They need to be combined with acoustic tokens
        return tokens.astype(np.float32)


class AcousticCodec:
    """Encodes speech into acoustic tokens capturing speaker characteristics.

    Uses a neural audio codec (EnCodec/DAC) to compress audio into
    discrete tokens that preserve voice quality and prosody.
    """

    def __init__(self, config: Optional[CodecConfig] = None, model_path: Optional[str] = None):
        self.config = config or CodecConfig()
        self.model_path = model_path
        self._model = None
        self._loaded = False

    async def load(self) -> bool:
        """Load the acoustic codec model."""
        try:
            # Try to load encodec
            from encodec import EncodecModel
            from encodec.utils import convert_audio

            self._model = EncodecModel.encodec_model_24khz()
            self._model.set_target_bandwidth(6.0)
            self._loaded = True

            import torch
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            return True
        except ImportError:
            print("Warning: encodec not installed. Using fallback codec.")
            return self._load_fallback()
        except Exception as e:
            print(f"Error loading acoustic codec: {e}")
            return self._load_fallback()

    def _load_fallback(self) -> bool:
        """Load a fallback implementation."""
        self._loaded = True
        return True

    async def encode(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio into acoustic tokens.

        Args:
            audio: Audio waveform as numpy array (shape: [samples])

        Returns:
            Acoustic tokens as numpy array (shape: [num_codebooks, num_frames])
        """
        if not self._loaded:
            await self.load()

        if self._model is not None:
            return await self._encode_with_model(audio)
        else:
            return self._encode_fallback(audio)

    async def _encode_with_model(self, audio: np.ndarray) -> np.ndarray:
        """Encode using EnCodec model."""
        import torch

        # Resample to 24kHz if needed
        if self.config.sample_rate != 24000:
            import librosa
            audio = librosa.resample(audio, orig_sr=self.config.sample_rate, target_sr=24000)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available():
            audio_tensor = audio_tensor.to("cuda")

        # Encode
        with torch.no_grad():
            encoded_frames = self._model.encode(audio_tensor)

        # Extract codes
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.cpu().numpy()[0]

        return codes

    def _encode_fallback(self, audio: np.ndarray) -> np.ndarray:
        """Fallback encoding using simple quantization."""
        frame_size = self.config.sample_rate // self.config.acoustic_frame_rate
        num_frames = len(audio) // frame_size

        # Generate multiple codebook entries per frame
        codes = np.zeros((self.config.num_codebooks, num_frames), dtype=np.int32)

        for i in range(num_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]

            # Compute various features for different codebooks
            for cb in range(self.config.num_codebooks):
                if cb == 0:
                    # Energy
                    feature = np.sum(frame ** 2)
                elif cb == 1:
                    # Zero crossing rate
                    feature = np.sum(np.abs(np.diff(np.sign(frame))))
                elif cb == 2:
                    # Spectral centroid approximation
                    fft = np.abs(np.fft.rfft(frame))
                    freqs = np.fft.rfftfreq(len(frame), 1/self.config.sample_rate)
                    feature = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)
                else:
                    # Local RNG for remaining codebooks — avoids global state pollution
                    rng = np.random.RandomState(cb * 1000 + i)
                    feature = rng.rand()

                # Quantize to codebook size
                codes[cb, i] = int(feature * self.config.acoustic_vocab_size) % self.config.acoustic_vocab_size

        return codes

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode acoustic tokens back to audio.

        Args:
            codes: Acoustic tokens (shape: [num_codebooks, num_frames])

        Returns:
            Audio waveform as numpy array
        """
        if not self._loaded:
            await self.load()

        if self._model is not None:
            return await self._decode_with_model(codes)
        else:
            return self._decode_fallback(codes)

    async def _decode_with_model(self, codes: np.ndarray) -> np.ndarray:
        """Decode using EnCodec model."""
        import torch

        # Convert to tensor
        codes_tensor = torch.from_numpy(codes).long().unsqueeze(0)

        if torch.cuda.is_available():
            codes_tensor = codes_tensor.to("cuda")

        # Decode
        with torch.no_grad():
            audio = self._model.decode([(codes_tensor, None)])

        audio = audio.cpu().numpy()[0, 0]

        # Resample back to original sample rate if needed
        if self.config.sample_rate != 24000:
            import librosa
            audio = librosa.resample(audio, orig_sr=24000, target_sr=self.config.sample_rate)

        return audio

    def _decode_fallback(self, codes: np.ndarray) -> np.ndarray:
        """Fallback decoding - generates simple audio from codes."""
        frame_size = self.config.sample_rate // self.config.acoustic_frame_rate
        num_frames = codes.shape[1]

        audio = np.zeros(num_frames * frame_size)

        for i in range(num_frames):
            # Generate simple tone based on codes
            freq = 200 + codes[0, i] % 500  # Base frequency
            energy = codes[1, i] / self.config.acoustic_vocab_size

            t = np.arange(frame_size) / self.config.sample_rate
            frame = energy * np.sin(2 * np.pi * freq * t)

            # Add harmonics from other codebooks
            for cb in range(2, min(self.config.num_codebooks, 8)):
                harmonic_freq = freq * (1 + cb * 0.1)
                frame += 0.1 * energy * np.sin(2 * np.pi * harmonic_freq * t)

            audio[i * frame_size:(i + 1) * frame_size] = frame

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.5

        return audio
