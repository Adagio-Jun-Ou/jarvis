"""Neural vocoder for high-quality speech synthesis.

Converts acoustic tokens or mel-spectrograms to audio waveforms.
Supports multiple vocoder backends: HiFi-GAN, BigVGAN, or fallback.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class VocoderConfig:
    """Configuration for neural vocoder."""
    sample_rate: int = 16000
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0


class NeuralVocoder:
    """Neural vocoder for high-quality speech synthesis.

    Supports multiple backends:
    - HiFi-GAN: Fast, high-quality neural vocoder
    - BigVGAN: Improved version with better voice quality
    - Fallback: Simple griffin-lim algorithm
    """

    def __init__(
        self,
        config: Optional[VocoderConfig] = None,
        model_path: Optional[str] = None,
        model_type: str = "hifigan"
    ):
        self.config = config or VocoderConfig()
        self.model_path = model_path
        self.model_type = model_type
        self._model = None
        self._loaded = False

    async def load(self) -> bool:
        """Load the vocoder model."""
        try:
            if self.model_type == "hifigan":
                return await self._load_hifigan()
            elif self.model_type == "bigvgan":
                return await self._load_bigvgan()
            else:
                return self._load_fallback()
        except Exception as e:
            print(f"Error loading vocoder: {e}")
            return self._load_fallback()

    async def _load_hifigan(self) -> bool:
        """Load HiFi-GAN model."""
        try:
            import torch

            # Try to load pre-trained model
            if self.model_path and Path(self.model_path).exists():
                self._model = torch.load(self.model_path)
            else:
                # Use a default universal model
                # In production, download from HuggingFace
                print("HiFi-GAN model not found, using fallback")
                return self._load_fallback()

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
            self._loaded = True
            return True

        except ImportError:
            return self._load_fallback()

    async def _load_bigvgan(self) -> bool:
        """Load BigVGAN model."""
        try:
            import torch

            # BigVGAN typically loaded from NVIDIA's repo
            if self.model_path and Path(self.model_path).exists():
                self._model = torch.load(self.model_path)
            else:
                print("BigVGAN model not found, using fallback")
                return self._load_fallback()

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
            self._loaded = True
            return True

        except ImportError:
            return self._load_fallback()

    def _load_fallback(self) -> bool:
        """Load fallback vocoder (Griffin-Lim)."""
        self._loaded = True
        return True

    async def synthesize(
        self,
        mel_spectrogram: np.ndarray,
        speaker_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert mel-spectrogram to audio.

        Args:
            mel_spectrogram: Mel spectrogram [n_mels, time]
            speaker_embedding: Optional speaker embedding for voice control

        Returns:
            Audio waveform
        """
        if not self._loaded:
            await self.load()

        if self._model is not None:
            return await self._synthesize_with_model(mel_spectrogram)
        else:
            return self._synthesize_fallback(mel_spectrogram)

    async def _synthesize_with_model(self, mel: np.ndarray) -> np.ndarray:
        """Synthesize using neural vocoder model."""
        import torch

        # Convert to tensor
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)

        if torch.cuda.is_available():
            mel_tensor = mel_tensor.to("cuda")

        with torch.no_grad():
            audio = self._model(mel_tensor)

        return audio.cpu().numpy()[0, 0]

    def _synthesize_fallback(self, mel: np.ndarray) -> np.ndarray:
        """Fallback synthesis using Griffin-Lim algorithm."""
        # Griffin-Lim phase reconstruction
        return self._griffin_lim(mel)

    def _griffin_lim(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Griffin-Lim algorithm for phase reconstruction."""
        # Invert mel to linear spectrogram
        spec = self._mel_to_linear(mel)

        # Initialize with random phase
        angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))
        complex_spec = spec * angles

        for _ in range(n_iter):
            # Inverse STFT
            audio = self._istft(complex_spec)
            # Forward STFT
            complex_spec = self._stft(audio)
            # Keep magnitude, update phase
            angles = np.exp(1j * np.angle(complex_spec))
            complex_spec = np.abs(complex_spec) * angles

        return self._istft(complex_spec)

    def _mel_to_linear(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to linear spectrogram."""
        # Simple inversion using pseudo-inverse
        mel_basis = self._create_mel_basis()
        mel_basis_pinv = np.linalg.pinv(mel_basis.T).T
        return np.maximum(1e-10, np.dot(mel_basis_pinv, mel))

    def _create_mel_basis(self) -> np.ndarray:
        """Create mel filterbank matrix."""
        n_fft = self.config.n_fft
        n_mels = self.config.n_mels
        sr = self.config.sample_rate
        fmin = self.config.mel_fmin
        fmax = self.config.mel_fmax

        # Create mel scale points
        mel_min = self._hz_to_mel(fmin)
        mel_max = self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Convert to FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Create filterbank
        basis = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if j < basis.shape[1]:
                    basis[i, j] = (j - left) / (center - left + 1e-10)
            for j in range(center, right):
                if j < basis.shape[1]:
                    basis[i, j] = (right - j) / (right - center + 1e-10)

        return basis

    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Short-time Fourier transform."""
        hop = self.config.hop_length
        win = self.config.win_length
        n_fft = self.config.n_fft

        # Pad audio
        audio = np.pad(audio, (win // 2, win // 2), mode='reflect')

        # Compute frames
        n_frames = 1 + (len(audio) - win) // hop
        frames = np.array([audio[i * hop:i * hop + win] for i in range(n_frames)])

        # Apply window and FFT
        window = np.hanning(win)
        spec = np.fft.rfft(frames * window, n=n_fft)

        return spec.T

    def _istft(self, spec: np.ndarray) -> np.ndarray:
        """Inverse short-time Fourier transform."""
        hop = self.config.hop_length
        win = self.config.win_length
        n_fft = self.config.n_fft

        # Inverse FFT
        frames = np.fft.irfft(spec.T, n=n_fft)[:, :win]

        # Apply window
        window = np.hanning(win)
        frames = frames * window

        # Overlap-add
        n_frames = frames.shape[0]
        audio_len = (n_frames - 1) * hop + win
        audio = np.zeros(audio_len)
        window_sum = np.zeros(audio_len)

        for i in range(n_frames):
            start = i * hop
            audio[start:start + win] += frames[i]
            window_sum[start:start + win] += window ** 2

        # Normalize by window sum
        audio = audio / (window_sum + 1e-10)

        # Remove padding
        return audio[win // 2:-win // 2]

    def mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        # Compute linear spectrogram
        spec = np.abs(self._stft(audio))

        # Apply mel filterbank
        mel_basis = self._create_mel_basis()
        mel = np.dot(mel_basis, spec)

        # Convert to log scale
        mel = np.log(np.maximum(1e-10, mel))

        return mel

    async def synthesize_from_tokens(
        self,
        acoustic_tokens: np.ndarray,
        semantic_tokens: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Synthesize audio directly from acoustic tokens.

        Args:
            acoustic_tokens: Acoustic token codes [num_codebooks, time]
            semantic_tokens: Optional semantic tokens for conditioning

        Returns:
            Audio waveform
        """
        if not self._loaded:
            await self.load()

        # For now, use acoustic codec for synthesis
        # In a full system, this would use a dedicated token-to-wave model
        from .codec import AcousticCodec
        codec = AcousticCodec()
        return await codec.decode(acoustic_tokens)
