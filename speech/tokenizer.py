"""Unified speech tokenizer combining semantic and acoustic codecs."""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .codec import SemanticCodec, AcousticCodec, CodecConfig


@dataclass
class SpeechTokens:
    """Container for speech tokens."""
    semantic: np.ndarray      # Shape: [num_semantic_tokens]
    acoustic: np.ndarray      # Shape: [num_codebooks, num_acoustic_tokens]
    duration: float           # Duration in seconds
    sample_rate: int          # Sample rate of original audio


class SpeechTokenizer:
    """Unified speech tokenizer that produces both semantic and acoustic tokens.

    This tokenizer provides a complete representation of speech that:
    1. Semantic tokens: Capture WHAT is being said (linguistic content)
    2. Acoustic tokens: Capture HOW it's being said (speaker, emotion, prosody)

    This dual-token approach enables:
    - Voice cloning: Keep acoustic tokens, modify semantic
    - Style transfer: Keep semantic tokens, modify acoustic
    - Conversational AI: Generate semantic with LLM, synthesize with acoustic
    """

    def __init__(
        self,
        config: Optional[CodecConfig] = None,
        semantic_model_path: Optional[str] = None,
        acoustic_model_path: Optional[str] = None,
    ):
        self.config = config or CodecConfig()
        self.semantic_codec = SemanticCodec(config, semantic_model_path)
        self.acoustic_codec = AcousticCodec(config, acoustic_model_path)
        self._loaded = False

    async def load(self) -> bool:
        """Load both codec models."""
        semantic_loaded = await self.semantic_codec.load()
        acoustic_loaded = await self.acoustic_codec.load()
        self._loaded = semantic_loaded and acoustic_loaded
        return self._loaded

    async def encode(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> SpeechTokens:
        """Encode audio into semantic and acoustic tokens.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of audio (defaults to config)

        Returns:
            SpeechTokens containing both semantic and acoustic representations
        """
        if not self._loaded:
            await self.load()

        # Resample if needed
        if sample_rate and sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)

        # Encode with both codecs in parallel
        semantic_task = asyncio.create_task(self.semantic_codec.encode(audio))
        acoustic_task = asyncio.create_task(self.acoustic_codec.encode(audio))

        semantic_tokens, acoustic_tokens = await asyncio.gather(
            semantic_task, acoustic_task
        )

        duration = len(audio) / self.config.sample_rate

        return SpeechTokens(
            semantic=semantic_tokens,
            acoustic=acoustic_tokens,
            duration=duration,
            sample_rate=self.config.sample_rate
        )

    async def decode(
        self,
        tokens: SpeechTokens,
        speaker_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Decode tokens back to audio.

        Args:
            tokens: SpeechTokens to decode
            speaker_embedding: Optional speaker embedding for voice control

        Returns:
            Audio waveform as numpy array
        """
        if not self._loaded:
            await self.load()

        # Use acoustic codec for decoding
        # The semantic tokens guide the content
        # The acoustic tokens provide the voice characteristics
        audio = await self.acoustic_codec.decode(tokens.acoustic)

        return audio

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )

    async def encode_file(self, audio_path: str) -> SpeechTokens:
        """Encode an audio file to tokens.

        Args:
            audio_path: Path to audio file

        Returns:
            SpeechTokens
        """
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            return await self.encode(audio, sr)
        except ImportError:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
            return await self.encode(audio, sr)

    async def decode_to_file(
        self,
        tokens: SpeechTokens,
        output_path: str,
        format: str = "wav"
    ) -> None:
        """Decode tokens and save to audio file.

        Args:
            tokens: SpeechTokens to decode
            output_path: Output file path
            format: Audio format (wav, mp3, etc.)
        """
        audio = await self.decode(tokens)

        # Normalize
        audio = np.clip(audio, -1.0, 1.0)

        # Save
        import wave
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    def get_token_sequence(self, tokens: SpeechTokens) -> np.ndarray:
        """Get interleaved token sequence for language modeling.

        Returns a flat sequence of tokens suitable for training
        or running through a language model.
        """
        # Interleave semantic and acoustic tokens
        semantic = tokens.semantic
        acoustic = tokens.acoustic

        # Repeat semantic tokens to match acoustic frame rate
        semantic_ratio = self.config.acoustic_frame_rate / self.config.semantic_frame_rate

        # Create interleaved sequence
        # Format: [S, A1, A2, ..., An, S, A1, A2, ..., An, ...]
        # Where S is semantic token, Ai is acoustic codebook i

        sequence = []
        for i, sem_tok in enumerate(semantic):
            # Add semantic token (offset to distinguish from acoustic)
            sequence.append(sem_tok)

            # Get corresponding acoustic frame
            acoustic_idx = int(i * semantic_ratio)
            if acoustic_idx < acoustic.shape[1]:
                for cb in range(acoustic.shape[0]):
                    # Offset acoustic tokens by semantic vocab size
                    sequence.append(
                        acoustic[cb, acoustic_idx] + self.config.semantic_vocab_size
                    )

        return np.array(sequence, dtype=np.int32)

    def parse_token_sequence(self, sequence: np.ndarray) -> SpeechTokens:
        """Parse a token sequence back to SpeechTokens."""
        # Separate semantic and acoustic tokens
        semantic = []
        acoustic = []

        semantic_vocab_size = self.config.semantic_vocab_size

        for tok in sequence:
            if tok < semantic_vocab_size:
                semantic.append(tok)
            else:
                acoustic.append(tok - semantic_vocab_size)

        # Reshape acoustic to [num_codebooks, num_frames]
        acoustic = np.array(acoustic, dtype=np.int32)
        num_codebooks = self.config.num_codebooks
        num_frames = len(acoustic) // num_codebooks

        if num_frames > 0:
            acoustic = acoustic[:num_frames * num_codebooks].reshape(num_codebooks, num_frames)
        else:
            acoustic = np.zeros((num_codebooks, 1), dtype=np.int32)

        return SpeechTokens(
            semantic=np.array(semantic, dtype=np.int32),
            acoustic=acoustic,
            duration=len(semantic) / self.config.semantic_frame_rate,
            sample_rate=self.config.sample_rate
        )
