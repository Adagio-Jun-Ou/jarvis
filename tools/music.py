"""Music tools for Jarvis - Classical music generation and token conversion.

Two main components:
1. MusicGenerator - Generates classical music compositions
2. MusicTokenizer - Converts music files to tokens for LCM observation

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                        MUSIC MODULE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ MusicGenerator   │         │  MusicTokenizer  │             │
│  │                  │         │                  │             │
│  │ - generate_piece │         │ - tokenize_midi  │             │
│  │ - generate_theme │         │ - tokenize_audio │             │
│  │ - harmonize      │         │ - extract_features             │
│  │ - develop_motive │         │ - analyze_harmony              │
│  └────────┬─────────┘         └────────┬─────────┘             │
│           │                            │                        │
│           └────────────┬───────────────┘                        │
│                        │                                        │
│              ┌─────────▼──────────┐                             │
│              │  MusicToken Format  │                             │
│              │                     │                             │
│              │  - Pitch tokens     │                             │
│              │  - Duration tokens  │                             │
│              │  - Harmony tokens   │                             │
│              │  - Structure tokens │                             │
│              │  - Style tokens     │                             │
│              └─────────┬──────────┘                             │
│                        │                                        │
│           ┌────────────┴────────────┐                           │
│           │                         │                           │
│  ┌────────▼─────────┐    ┌──────────▼───────┐                  │
│  │ Daydream Engine  │    │ Prefrontal Cortex │                  │
│  │                  │    │                   │                  │
│  │ - Practice       │    │ - Observe music   │                  │
│  │ - Explore        │    │ - Learn patterns  │                  │
│  │ - Compose        │    │ - Store concepts  │                  │
│  └──────────────────┘    └───────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Token Format for LCM:
{
    "type": "music_token",
    "pitch": ["C4", "E4", "G4"],  # Notes in chord/melody
    "duration": [1.0, 1.0, 0.5],   # Duration in beats
    "harmony": "I-V-vi-IV",        # Chord progression
    "structure": "AABA",           # Musical form
    "style": "classical",          # Genre/period
    "features": {                  # Extracted features
        "key": "C_major",
        "tempo": 120,
        "complexity": 0.7
    }
}
"""
import asyncio
import hashlib
import json
import logging
import random
import struct
import wave
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)

# Music theory constants
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(0, 9))  # C0 to C8

# Classical music periods
MUSIC_PERIODS = [
    'medieval', 'renaissance', 'baroque', 'classical', 
    'romantic', 'modern', 'contemporary'
]

# Common chord progressions by style
CHORD_PROGRESSIONS = {
    'baroque': [
        ['I', 'IV', 'V', 'I'],           # Common cadence
        ['I', 'V', 'vi', 'IV'],          # Circle progression
        ['i', 'iv', 'V', 'i'],           # Minor progression
    ],
    'classical': [
        ['I', 'IV', 'V', 'I'],           # Basic cadence
        ['I', 'vi', 'IV', 'V'],          # Popular classical
        ['I', 'V', 'vi', 'iii'],         # Mozart variation
    ],
    'romantic': [
        ['I', 'vi', 'IV', 'V'],         # Pop progression origin
        ['i', 'VI', 'III', 'VII'],      # Romantic minor
        ['I', 'IV', 'V', 'vi'],         # Emotional progression
    ],
}

# Scale definitions
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
}


class MusicTokenType(Enum):
    """Types of music tokens for LCM observation."""
    NOTE = "note"
    CHORD = "chord"
    MELODY = "melody"
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    STRUCTURE = "structure"
    STYLE = "style"
    
    # Generation events
    GENERATION = "generation"
    ANALYSIS = "analysis"
    COMPOSITION = "composition"
    
    # Acoustic tokens from EnCodec neural codec
    ACOUSTIC_FRAME = "acoustic_frame"
    ACOUSTIC_CODEBOOK = "acoustic_codebook"
    SPECTRAL_FEATURE = "spectral_feature"


@dataclass
class MusicToken:
    """Token representation for LCM observation."""
    token_type: MusicTokenType
    data: dict
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "music_module"
    
    def to_observation(self) -> dict:
        """Convert to observation format for PFC."""
        return {
            "type": self.token_type.value,
            "content": json.dumps(self.data) if isinstance(self.data, dict) else str(self.data),
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


@dataclass
class Composition:
    """A musical composition."""
    title: str
    composer: str = "Jarvis"
    style: str = "classical"
    tempo: int = 120
    key: str = "C_major"
    time_signature: tuple = (4, 4)
    measures: list = field(default_factory=list)
    tokens: list[MusicToken] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_tokens(self) -> list[MusicToken]:
        """Convert composition to token sequence."""
        tokens = []
        
        # Style token
        tokens.append(MusicToken(
            token_type=MusicTokenType.STYLE,
            data={"style": self.style, "period": self.style}
        ))
        
        # Structure token
        structure = self._analyze_structure()
        tokens.append(MusicToken(
            token_type=MusicTokenType.STRUCTURE,
            data={"form": structure, "measures": len(self.measures)}
        ))
        
        # Harmony tokens
        for i, measure in enumerate(self.measures):
            harmony = self._extract_harmony(measure)
            if harmony:
                tokens.append(MusicToken(
                    token_type=MusicTokenType.HARMONY,
                    data={
                        "measure": i + 1,
                        "chords": harmony,
                        "key": self.key
                    }
                ))
        
        return tokens
    
    def _analyze_structure(self) -> str:
        """Analyze musical structure/form."""
        if len(self.measures) < 8:
            return "phrase"
        elif len(self.measures) < 16:
            return "period"
        else:
            # Detect common forms
            return "binary" if len(self.measures) % 2 == 0 else "through_composed"
    
    def _extract_harmony(self, measure: list) -> list[str]:
        """Extract chord progression from measure."""
        # Simplified: would analyze notes to determine chords
        if not measure:
            return []
        return ["I"]  # Placeholder - would be actual chord analysis


class AcousticTokenGenerator:
    """Generates discrete acoustic tokens from audio using EnCodec neural codec.
    
    EnCodec produces discrete audio tokens at ~50Hz that can be observed by the LCM.
    These tokens capture:
    - Timbre (instrument quality)
    - Dynamics (volume envelope)
    - Spectral content (harmonics, formants)
    - Temporal patterns (rhythm, articulation)
    
    Architecture:
    1. Generate symbolic composition (MusicGenerator)
    2. Synthesize to audio (MIDI -> waveform using sine/triangle waves)
    3. Encode with EnCodec -> discrete tokens
    4. Observe tokens in PFC for learning acoustic patterns
    """
    
    def __init__(self):
        self._model = None
        self._sample_rate = 24000  # EnCodec default
        self._bandwidth = 6.0  # kbps (affects codebook size)
        self._initialized = False
        self._device = "cpu"
    
    def _init_model(self):
        """Lazily initialize EnCodec model."""
        if self._initialized:
            return
        
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            
            # Use pretrained EnCodec model
            self._model = EncodecModel.encodec_model_24khz()
            self._model.set_target_bandwidth(self._bandwidth)
            self._model.eval()
            
            # Check for GPU
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                self._device = "cuda"
            
            self._initialized = True
            logger.info("[AcousticTokenGenerator] EnCodec model loaded (device=%s)", self._device)
            
        except ImportError as e:
            logger.warning(f"[AcousticTokenGenerator] EnCodec not available: {e}")
            logger.info("[AcousticTokenGenerator] Using fallback synthetic tokens")
            self._model = None
            self._initialized = True
    
    def _synthesize_classical(
        self,
        composition: dict,
        style: str = "classical",
        tempo: int = 120,
    ) -> tuple:
        """Synthesize audio from symbolic composition with realistic instrument timbres.
        
        Uses multi-oscillator synthesis with proper ADSR envelopes for
        piano, strings, and classical guitar timbres.
        
        Returns:
            tuple: (audio_tensor, sample_rate)
        """
        import numpy as np
        
        sample_rate = self._sample_rate
        measures = composition.get("notes", [])
        
        # Duration per measure in seconds
        beats_per_measure = 4
        measure_duration = (60.0 / tempo) * beats_per_measure
        total_duration = measure_duration * len(measures)
        
        # Audio buffer
        total_samples = int(total_duration * sample_rate)
        audio = np.zeros(total_samples, dtype=np.float32)
        
        # Instrument timbres - much richer harmonic content
        # Based on real acoustic instrument spectra
        instrument_timbres = {
            'piano': {
                'harmonics': [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0156],
                'attack': 0.003,      # Very fast attack
                'decay': 0.8,          # Long natural decay
                'sustain': 0.3,        # Low sustain (piano decays)
                'release': 0.3,
                'brightness': 1.2,
                'vibrato_rate': 0.0,   # No vibrato
                'vibrato_depth': 0.0,
            },
            'strings': {
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
                'attack': 0.08,        # Slower bow attack
                'decay': 0.1,
                'sustain': 0.85,       # High sustain for bowed strings
                'release': 0.2,
                'brightness': 1.0,
                'vibrato_rate': 5.0,   # Natural vibrato
                'vibrato_depth': 0.015,
            },
            'harpsichord': {
                'harmonics': [1.0, 0.95, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1],
                'attack': 0.001,       # Instant pluck
                'decay': 1.5,          # Long decay
                'sustain': 0.0,        # No sustain
                'release': 0.5,
                'brightness': 1.5,
                'vibrato_rate': 0.0,
                'vibrato_depth': 0.0,
            },
            'classical_guitar': {
                'harmonics': [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05],
                'attack': 0.005,
                'decay': 0.35,
                'sustain': 0.4,
                'release': 0.25,
                'brightness': 0.9,
                'vibrato_rate': 0.0,
                'vibrato_depth': 0.0,
            },
        }
        
        # Choose instrument based on style
        style_to_instrument = {
            'baroque': 'harpsichord',
            'classical': 'piano',
            'romantic': 'strings',
            'modern': 'piano',
        }
        instrument = style_to_instrument.get(style, 'piano')
        timbre = instrument_timbres[instrument]
        
        # Note frequency map (equal temperament)
        note_freqs = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
        }
        
        t = np.arange(total_samples) / sample_rate
        
        # Process each measure
        for m_idx, measure in enumerate(measures):
            measure_start = m_idx * measure_duration
            notes = measure.get("notes", [])
            
            for note_data in notes:
                # Get note details
                pitch = note_data.get("pitch", "C4")
                duration_beats = note_data.get("duration", 1.0)
                velocity = note_data.get("velocity", 80) / 127.0
                
                # Parse pitch
                note_name = pitch[:-1] if pitch[-1].isdigit() else pitch
                octave = int(pitch[-1]) if pitch[-1].isdigit() else 4
                base_freq = note_freqs.get(note_name, 261.63)
                freq = base_freq * (2 ** (octave - 4))
                
                # Duration in seconds
                note_duration = (60.0 / tempo) * duration_beats
                start_sample = int(measure_start * sample_rate)
                end_sample = min(start_sample + int(note_duration * sample_rate), total_samples)
                
                note_samples = end_sample - start_sample
                if note_samples <= 0:
                    continue
                
                note_t = t[start_sample:end_sample]
                
                # ADSR envelope with proper attack/decay/sustain/release
                envelope = self._create_adsr_envelope(
                    note_samples, 
                    sample_rate,
                    attack=timbre['attack'],
                    decay=timbre['decay'],
                    sustain=timbre['sustain'],
                    release=timbre['release'],
                    note_duration=note_duration
                )
                
                # Generate note with harmonics
                note_wave = np.zeros(note_samples)
                
                for h_idx, h_amp in enumerate(timbre['harmonics']):
                    harmonic_freq = freq * (h_idx + 1)
                    
                    # Add slight detune for warmth
                    detune = 1.0 + 0.0005 * (h_idx + 1) * np.sin(2 * np.pi * 0.3 * note_t)
                    
                    # Add vibrato for instruments that have it
                    if timbre['vibrato_rate'] > 0:
                        vibrato = 1 + timbre['vibrato_depth'] * np.sin(2 * np.pi * timbre['vibrato_rate'] * note_t)
                        harmonic_freq *= vibrato
                    
                    # Generate harmonic with phase variation
                    phase = np.random.uniform(0, 2 * np.pi)  # Random phase per harmonic
                    harmonic = h_amp * np.sin(2 * np.pi * harmonic_freq * detune * note_t + phase)
                    
                    # Apply brightness filter
                    harmonic *= timbre['brightness'] ** (-h_idx * 0.3)
                    
                    note_wave += harmonic
                
                # Apply envelope and velocity
                note_wave *= envelope * velocity
                
                # Add to audio buffer
                audio[start_sample:end_sample] += note_wave
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7
        
        # Add subtle reverb for space
        audio = self._add_reverb(audio, sample_rate, decay=0.3, delay_ms=50)
        
        return audio, sample_rate
    
    def _create_adsr_envelope(
        self, 
        samples: int, 
        sample_rate: int,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        note_duration: float
    ) -> np.ndarray:
        """Create ADSR envelope for realistic note articulation."""
        import numpy as np
        
        envelope = np.ones(samples)
        
        # Attack
        attack_samples = int(attack * sample_rate)
        if attack_samples > 0 and attack_samples < samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        decay_samples = int(decay * sample_rate)
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, samples)
        if decay_end > decay_start:
            envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_end - decay_start)
        
        # Sustain (automatic - envelope stays at sustain level)
        
        # Release (at end of note)
        release_samples = int(release * sample_rate)
        if release_samples > 0 and release_samples < samples:
            release_start = max(0, samples - release_samples)
            # Get the level at release start
            release_level = envelope[release_start] if release_start < len(envelope) else sustain
            envelope[release_start:] = np.linspace(release_level, 0, samples - release_start)
        
        return envelope
    
    def _add_reverb(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        decay: float = 0.3,
        delay_ms: int = 50
    ) -> np.ndarray:
        """Add simple reverb for spatial depth."""
        import numpy as np
        
        delay_samples = int(delay_ms * sample_rate / 1000)
        
        # Simple delay-based reverb
        reverb = np.zeros_like(audio)
        
        # Multiple delays for richness
        delays = [delay_samples, delay_samples * 2, delay_samples * 3]
        amplitudes = [decay, decay * 0.7, decay * 0.5]
        
        for d, a in zip(delays, amplitudes):
            if d < len(audio):
                reverb[d:] += audio[:-d] * a
        
        # Mix dry and wet
        output = audio + reverb
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 1:
            output = output / max_val * 0.9
        
        return output
    
    async def generate_acoustic_tokens(
        self,
        composition: dict,
        style: str = "classical",
        tempo: int = 120,
        key: str = "C_major",
    ) -> dict:
        """Generate discrete acoustic tokens from composition.
        
        This is the main entry point for acoustic token generation.
        
        Returns:
            dict with:
                - symbolic_tokens: Musical structure tokens
                - acoustic_tokens: EnCodec discrete audio tokens
                - spectral_features: FFT-derived features
        """
        self._init_model()
        
        import torch
        
        # Step 1: Synthesize audio from composition (offload to executor to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        audio_np, sr = await loop.run_in_executor(
            None, self._synthesize_classical, composition, style, tempo
        )
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, samples]
        
        acoustic_tokens = []
        
        if self._model is not None:
            # Use EnCodec for discrete tokens
            try:
                # Add channel dimension: [samples] -> [1, 1, samples] (batch, channels, samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
                elif audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.unsqueeze(1)  # [1, samples] -> [1, 1, samples]
                
                # Pad to handle frame alignment
                # EnCodec expects ~46ms frames with ~23ms hop at 24kHz
                # Minimum length is about 3 seconds for stable encoding
                min_length = 3 * self._sample_rate
                if audio_tensor.shape[-1] < min_length:
                    # Pad by repeating
                    pad_length = min_length - audio_tensor.shape[-1]
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length), mode='reflect')
                
                if self._device == "cuda":
                    audio_tensor = audio_tensor.cuda()
                
                with torch.no_grad():
                    # Encode returns list of tuples: [(codes, scale), ...] per frame
                    encoded = self._model.encode(audio_tensor)
                    
                    # encoded is a list of tuples [(codebook_tensors, scale)]
                    # Each codebook tensor has shape [B, num_codebooks, num_frames]
                    if isinstance(encoded, list) and len(encoded) > 0:
                        codes, scale = encoded[0]
                        # codes shape: [B, num_codebooks, T] or similar
                        if hasattr(codes, 'cpu'):
                            codes_np = codes.squeeze(0).cpu().numpy()  # [num_codebooks, T]
                            num_frames = codes_np.shape[-1]
                            
                            for frame_idx in range(min(num_frames, 100)):  # Limit to 100 frames
                                if codes_np.ndim == 2:
                                    codebook_indices = codes_np[:, frame_idx].tolist()
                                else:
                                    codebook_indices = [int(codes_np[frame_idx])]
                                
                                acoustic_tokens.append(MusicToken(
                                    token_type=MusicTokenType.ACOUSTIC_FRAME,
                                    data={
                                        "frame": frame_idx,
                                        "codebooks": codebook_indices,
                                        "timestamp_ms": int(frame_idx * 23),  # ~23ms per frame
                                    }
                                ))
                            
                            logger.info(f"[AcousticTokenGenerator] Generated {len(acoustic_tokens)} EnCodec frames")
                    
            except Exception as e:
                logger.warning(f"[AcousticTokenGenerator] EnCodec encoding failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # Fall back to spectral features
                acoustic_tokens = []
        
        if not acoustic_tokens:
            # Fallback: Generate spectral features without EnCodec
            acoustic_tokens = self._extract_spectral_features(audio_np, sr, style)
        
        # Step 2: Also extract spectral features (works without EnCodec)
        spectral_tokens = self._extract_spectral_features(audio_np, sr, style)
        
        return {
            "acoustic_tokens": [t.to_observation() for t in acoustic_tokens],
            "spectral_tokens": [t.to_observation() for t in spectral_tokens],
            "sample_rate": sr,
            "duration": len(audio_np) / sr,
            "style": style,
            "key": key,
            "summary": f"Generated {len(acoustic_tokens)} acoustic tokens + {len(spectral_tokens)} spectral features",
        }
    
    def _extract_spectral_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        style: str,
    ) -> list[MusicToken]:
        """Extract spectral features from audio using FFT.
        
        These features capture:
        - Dominant frequencies (pitch content)
        - Spectral centroid (brightness)
        - Spectral flux (change over time)
        - RMS energy
        """
        import numpy as np
        
        tokens = []
        
        # Frame parameters (50ms frames at ~10fps for LCM observation)
        frame_duration = 0.05  # 50ms
        hop_duration = 0.1    # 100ms
        frame_samples = int(frame_duration * sample_rate)
        hop_samples = int(hop_duration * sample_rate)
        
        # FFT size
        n_fft = 2048
        
        num_frames = (len(audio) - frame_samples) // hop_samples + 1
        
        for frame_idx in range(num_frames):
            start = frame_idx * hop_samples
            end = min(start + frame_samples, len(audio))
            
            frame = audio[start:end]
            
            # Apply Hann window
            windowed = frame * np.hanning(len(frame))
            
            # FFT
            fft_result = np.fft.rfft(windowed, n=n_fft)
            magnitude = np.abs(fft_result)
            
            # Frequency bins
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
            
            # Spectral centroid (brightness)
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
            
            # RMS energy
            rms = np.sqrt(np.mean(frame ** 2))
            
            # Dominant frequencies (top 5)
            top_indices = np.argsort(magnitude)[-5:][::-1]
            dominant_freqs = [float(freqs[i]) for i in top_indices if magnitude[i] > 0]
            
            # Band energies (bass, mid, high)
            bass_mask = freqs < 300
            mid_mask = (freqs >= 300) & (freqs < 2000)
            high_mask = freqs >= 2000
            
            bass_energy = float(np.sum(magnitude[bass_mask] ** 2))
            mid_energy = float(np.sum(magnitude[mid_mask] ** 2))
            high_energy = float(np.sum(magnitude[high_mask] ** 2))
            
            # Create spectral token
            tokens.append(MusicToken(
                token_type=MusicTokenType.SPECTRAL_FEATURE,
                data={
                    "frame": frame_idx,
                    "timestamp_ms": int(frame_idx * hop_duration * 1000),
                    "spectral_centroid_hz": float(spectral_centroid),
                    "rms_energy": float(rms),
                    "dominant_frequencies": dominant_freqs[:3],
                    "band_energies": {
                        "bass": bass_energy,
                        "mid": mid_energy,
                        "high": high_energy,
                    },
                    "style": style,
                }
            ))
        
        logger.info(f"[AcousticTokenGenerator] Extracted {len(tokens)} spectral features")
        
        return tokens


class MusicGenerator:
    """Generates classical music compositions.
    
    Uses rule-based music theory combined with statistical patterns
    for generating coherent classical-style compositions.
    
    Features:
    - Voice leading rules (proper resolution of dissonances)
    - Counterpoint principles
    - Harmonic progressions by style period
    - Melodic development and variation
    """
    
    def __init__(self):
        self.style_weights = {
            'baroque': {'ornamentation': 0.8, 'counterpoint': 0.9},
            'classical': {'balance': 0.9, 'clarity': 0.8},
            'romantic': {'expression': 0.9, 'chromaticism': 0.7},
        }
        
    def register_tools(self, registry):
        """Register music generation tools."""
        
        registry.register_tool(
            name="generate_composition",
            description="Generate a classical music composition",
            parameters={
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": MUSIC_PERIODS,
                        "description": "Musical style period"
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key (e.g., 'C_major', 'A_minor')"
                    },
                    "measures": {
                        "type": "integer",
                        "description": "Number of measures to generate",
                        "default": 16
                    },
                    "tempo": {
                        "type": "integer",
                        "description": "Tempo in BPM",
                        "default": 120
                    }
                },
                "required": ["style"]
            },
            function=self.generate_composition,
            category="music",
        )
        
        registry.register_tool(
            name="generate_variation",
            description="Create a variation on a musical theme",
            parameters={
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "Theme notes in format: 'C4-E4-G4'"
                    },
                    "variation_type": {
                        "type": "string",
                        "enum": ["ornamentation", "inversion", "sequence", "diminution", "augmentation"],
                        "description": "Type of variation to create"
                    }
                },
                "required": ["theme", "variation_type"]
            },
            function=self.generate_variation,
            category="music",
        )
        
    async def generate_composition(
        self,
        style: str = "classical",
        key: str = "C_major",
        measures: int = 16,
        tempo: int = 120,
    ) -> dict:
        """Generate a new musical composition.
        
        Returns composition with notes, harmony, structure, and tokens
        for LCM observation.
        """
        logger.info(f"[MusicGenerator] Generating {style} composition in {key}, {measures} measures")
        
        # Parse key
        key_parts = key.split('_')
        root = key_parts[0]
        mode = key_parts[1] if len(key_parts) > 1 else 'major'
        
        # Get scale
        scale_intervals = SCALES.get(mode, SCALES['major'])
        
        # Build scale notes
        root_idx = NOTES.index(root) if root in NOTES else 0
        scale_notes = []
        for interval in scale_intervals:
            note_idx = (root_idx + interval) % 12
            octave = 4  # Middle octave
            scale_notes.append(f"{NOTES[note_idx]}{octave}")
        
        # Get chord progressions for style
        progressions = CHORD_PROGRESSIONS.get(style, CHORD_PROGRESSIONS['classical'])
        
        # Generate composition
        composition_measures = []
        chord_sequence = []
        
        for m in range(measures):
            # Choose progression
            prog_idx = (m // 4) % len(progressions)
            local_prog = progressions[prog_idx]
            chord = local_prog[m % len(local_prog)]
            chord_sequence.append(chord)
            
            # Generate melody for this measure
            melody = self._generate_melody(
                scale_notes=scale_notes,
                chord=chord,
                style=style,
                position_in_phrase=m % 4
            )
            
            composition_measures.append({
                "measure": m + 1,
                "chord": chord,
                "notes": melody,
            })
        
        # Create composition object
        composition = Composition(
            title=f"{style.capitalize()} Composition in {key}",
            style=style,
            tempo=tempo,
            key=key,
            measures=composition_measures,
        )
        
        # Convert to tokens for LCM
        tokens = composition.to_tokens()
        
        # Generate MIDI-like representation
        midi_data = self._to_midi_representation(composition_measures, tempo, key)
        
        # Return structure (acoustic tokens generated separately for efficiency)
        return {
            "composition": {
                "title": composition.title,
                "style": style,
                "key": key,
                "tempo": tempo,
                "measures": measures,
                "notes": composition_measures,
            },
            "midi": midi_data,
            "tokens": [t.to_observation() for t in tokens],
            "summary": f"Generated {measures}-measure {style} composition in {key}",
        }
    
    async def generate_composition_with_acoustics(
        self,
        style: str = "classical",
        key: str = "C_major",
        measures: int = 16,
        tempo: int = 120,
    ) -> dict:
        """Generate composition with both symbolic and acoustic tokens.
        
        This combines:
        1. Symbolic tokens (style, structure, harmony) - for musical concepts
        2. Acoustic tokens (spectral features) - for sound patterns
        
        Both are observed by the PFC for learning.
        """
        # Generate symbolic composition
        result = await self.generate_composition(style, key, measures, tempo)
        
        # Generate acoustic tokens
        acoustic_gen = AcousticTokenGenerator()
        acoustic_result = await acoustic_gen.generate_acoustic_tokens(
            result["composition"],
            style=style,
            tempo=tempo,
            key=key,
        )
        
        # Combine tokens
        return {
            **result,
            "acoustic_tokens": acoustic_result.get("acoustic_tokens", []),
            "spectral_tokens": acoustic_result.get("spectral_tokens", []),
            "audio_duration": acoustic_result.get("duration", 0),
            "sample_rate": acoustic_result.get("sample_rate", 24000),
            "full_summary": f"{result['summary']} + {acoustic_result.get('summary', '')}",
        }
    
    def _generate_melody(
        self,
        scale_notes: list[str],
        chord: str,
        style: str,
        position_in_phrase: int
    ) -> list[dict]:
        """Generate a melody for a given chord and style."""
        melody = []
        
        # Number of notes per measure (style-dependent)
        notes_per_measure = {
            'baroque': random.choice([4, 6, 8]),
            'classical': random.choice([4, 8]),
            'romantic': random.choice([3, 4, 6]),
        }.get(style, 4)
        
        # Generate notes
        for i in range(notes_per_measure):
            # Choose note from scale with bias towards chord tones
            if random.random() < 0.7:  # 70% chord tones
                note = random.choice(scale_notes[:3])  # Tonic triad
            else:
                note = random.choice(scale_notes)
            
            # Determine duration based on position and style
            if position_in_phrase == 3 and i == notes_per_measure - 1:
                # End of phrase - longer note
                duration = 1.0
            else:
                duration = random.choice([0.25, 0.5, 0.5, 1.0])
            
            melody.append({
                "pitch": note,
                "duration": duration,
                "velocity": random.randint(60, 90),
            })
        
        return melody
    
    def _to_midi_representation(
        self,
        measures: list[dict],
        tempo: int,
        key: str
    ) -> dict:
        """Convert composition to MIDI-like representation."""
        return {
            "format": "midi",
            "tempo": tempo,
            "key": key,
            "tracks": [{
                "channel": 0,
                "events": [
                    {
                        "time": i * 480,  # Ticks per beat (480 = 1 beat)
                        "note": note["pitch"],
                        "velocity": note["velocity"],
                        "duration": int(note["duration"] * 480),
                    }
                    for m in measures
                    for i, note in enumerate(m["notes"])
                ]
            }]
        }
    
    async def generate_variation(
        self,
        theme: str,
        variation_type: str = "ornamentation",
    ) -> dict:
        """Create a variation on a musical theme."""
        logger.info(f"[MusicGenerator] Creating {variation_type} variation on theme: {theme}")
        
        # Parse theme notes
        theme_notes = [n.strip() for n in theme.split('-')]
        
        # Apply variation
        if variation_type == "inversion":
            # Invert intervals
            variations = self._invert_theme(theme_notes)
        elif variation_type == "sequence":
            # Transpose and repeat
            variations = self._sequence_theme(theme_notes)
        elif variation_type == "diminution":
            # Halve note values
            variations = self._diminish_theme(theme_notes)
        elif variation_type == "augmentation":
            # Double note values
            variations = self._augment_theme(theme_notes)
        else:
            # Ornamentation (mordents, trills, etc.)
            variations = self._ornament_theme(theme_notes)
        
        return {
            "theme": theme_notes,
            "variation_type": variation_type,
            "variation": variations,
            "tokens": [{
                "type": "composition",
                "variation": variation_type,
                "content": f"{variation_type} of {theme}",
            }]
        }
    
    def _parse_note(self, note: str) -> tuple[str, int]:
        """Parse 'C4' into ('C', 4), 'F#5' into ('F#', 5), etc."""
        note = note.strip()
        if len(note) >= 2 and note[-1].isdigit():
            octave = int(note[-1])
            name = note[:-1]
            return (name, octave)
        return (note, 4)

    def _note_to_midi(self, note: str) -> int:
        """Convert note name like 'C4' or 'F#5' to MIDI number."""
        note_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                     'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                     'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
        name, octave = self._parse_note(note)
        return (octave + 1) * 12 + note_map.get(name, 0)

    def _midi_to_note(self, midi: int) -> str:
        """Convert MIDI number to note name like 'C4'."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        name = note_names[midi % 12]
        return f"{name}{octave}"

    def _invert_theme(self, notes: list[str]) -> list[str]:
        """Invert a theme by mirroring intervals around the first note."""
        if len(notes) < 2:
            return notes[:]
        inverted = [notes[0]]
        pivot = self._note_to_midi(notes[0])
        for note in notes[1:]:
            midi = self._note_to_midi(note)
            mirrored = pivot - (midi - pivot)
            mirrored = max(36, min(96, mirrored))
            inverted.append(self._midi_to_note(mirrored))
        return inverted

    def _sequence_theme(self, notes: list[str], intervals: int = 2) -> list[str]:
        """Create a sequence by transposing the theme up by semitone intervals."""
        if not notes:
            return notes[:]
        result = notes[:]
        for i in range(1, min(intervals, 4)):
            transposed = []
            for note in notes:
                midi = self._note_to_midi(note) + i * 2
                midi = min(96, midi)
                transposed.append(self._midi_to_note(midi))
            result.extend(transposed)
        return result

    def _diminish_theme(self, notes: list[str]) -> list[str]:
        """Diminish theme by contracting intervals toward the first note."""
        if len(notes) < 2:
            return notes[:]
        pivot = self._note_to_midi(notes[0])
        diminished = [notes[0]]
        for note in notes[1:]:
            midi = self._note_to_midi(note)
            interval = midi - pivot
            contracted = pivot + (interval // 2) if abs(interval) > 1 else midi
            contracted = max(36, min(96, contracted))
            diminished.append(self._midi_to_note(contracted))
        return diminished

    def _augment_theme(self, notes: list[str]) -> list[str]:
        """Augment theme by expanding intervals from the first note."""
        if len(notes) < 2:
            return notes[:]
        pivot = self._note_to_midi(notes[0])
        augmented = [notes[0]]
        for note in notes[1:]:
            midi = self._note_to_midi(note)
            interval = midi - pivot
            expanded = pivot + interval * 2 if abs(interval) > 1 else midi + 1
            expanded = max(36, min(96, expanded))
            augmented.append(self._midi_to_note(expanded))
        return augmented
    
    def _ornament_theme(self, notes: list[str]) -> list[str]:
        """Add ornaments to theme."""
        ornamented = []
        for note in notes:
            ornamented.append(note)
            # Occasionally add ornaments
            if random.random() < 0.3:
                # Add passing tone or mordent
                ornamented.append(note)
        return ornamented


class MusicTokenizer:
    """Converts music files to tokens for LCM observation.
    
    Supports:
    - MIDI files (.mid)
    - Audio files (.wav, .mp3) - converts using simple feature extraction
    - MusicXML files (.musicxml, .mxl)
    
    Token representation includes:
    - Pitch classes and absolute pitches
    - Duration ratios
    - Harmonic intervals
    - Rhythmic patterns
    - Structural markers
    """
    
    def __init__(self, music_library_path: Optional[Path] = None):
        self.music_library_path = music_library_path or Path.home() / ".jarvis" / "music_library"
        self.music_library_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for processed files
        self._processed_cache: dict[str, list[MusicToken]] = {}
        
    def register_tools(self, registry):
        """Register music tokenization tools."""
        registry.register_tool(
            name="tokenize_music_file",
            description="Convert a music file to tokens for analysis",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to music file (MIDI, WAV, MP3)"
                    },
                    "analyze_features": {
                        "type": "boolean",
                        "description": "Whether to extract advanced features",
                        "default": True
                    }
                },
                "required": ["file_path"]
            },
            function=self.tokenize_file,
            category="music",
        )
        
        registry.register_tool(
            name="analyze_harmony",
            description="Analyze harmonic progression in music tokens",
            parameters={
                "type": "object",
                "properties": {
                    "tokens": {
                        "type": "string",
                        "description": "JSON string of music tokens to analyze"
                    }
                },
                "required": ["tokens"]
            },
            function=self.analyze_harmony,
            category="music",
        )
        
        registry.register_tool(
            name="ingest_music_library",
            description="Process all music files in library for LCM learning",
            parameters={
                "type": "object",
                "properties": {
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum files to process",
                        "default": 100
                    }
                }
            },
            function=self.ingest_library,
            category="music",
        )
    
    async def tokenize_file(
        self,
        file_path: str,
        analyze_features: bool = True,
    ) -> dict:
        """Tokenize a music file for LCM observation.
        
        Converts audio/MIDI to structured tokens representing:
        - Pitch sequences
        - Rhythmic patterns
        - Harmonic progressions
        - Structural elements
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        logger.info(f"[MusicTokenizer] Tokenizing: {path.name}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.mid':
            tokens = await self._tokenize_midi(path, analyze_features)
        elif suffix in ['.wav', '.mp3', '.flac', '.ogg']:
            tokens = await self._tokenize_audio(path, analyze_features)
        elif suffix in ['.musicxml', '.mxl', '.xml']:
            tokens = await self._tokenize_musicxml(path, analyze_features)
        else:
            return {"error": f"Unsupported format: {suffix}"}
        
        # Cache result
        cache_key = hashlib.md5(str(path).encode()).hexdigest()
        self._processed_cache[cache_key] = tokens
        
        return {
            "file": str(path),
            "token_count": len(tokens),
            "tokens": [t.to_observation() for t in tokens[:50]],  # First 50 tokens
            "all_tokens_available": True,
            "summary": f"Tokenized {path.name} into {len(tokens)} music tokens",
        }
    
    async def _tokenize_midi(
        self,
        path: Path,
        analyze_features: bool
    ) -> list[MusicToken]:
        """Tokenize MIDI file.
        
        Simplified MIDI parsing - extracts notes, timing, and structure.
        """
        tokens = []
        
        try:
            # For now, use basic MIDI file reading
            # Real implementation would use mido or similar library
            with open(path, 'rb') as f:
                # Read header
                header = f.read(4)
                if header != b'MThd':
                    raise ValueError("Invalid MIDI file")
                
                # Read header parameters
                format_type = struct.unpack('>H', f.read(2))[0]
                num_tracks = struct.unpack('>H', f.read(2))[0]
                division = struct.unpack('>H', f.read(2))[0]
                
                # Add header token
                tokens.append(MusicToken(
                    token_type=MusicTokenType.STRUCTURE,
                    data={
                        "format": format_type,
                        "tracks": num_tracks,
                        "division": division,
                        "file": path.name,
                    }
                ))
                
                # Skip track data for now
                # Real implementation would parse each track
                
        except Exception as e:
            logger.warning(f"MIDI parsing error: {e}")
        
        return tokens
    
    async def _tokenize_audio(
        self,
        path: Path,
        analyze_features: bool
    ) -> list[MusicToken]:
        """Tokenize audio file using feature extraction.
        
        Extracts:
        - Spectral features (pitch estimation)
        - Rhythmic features (tempo, beats)
        - Tonal features (key, chord estimates)
        """
        tokens = []
        
        try:
            # Basic WAV file reading for feature extraction
            if path.suffix.lower() == '.wav':
                with wave.open(str(path), 'rb') as wav:
                    # Get audio parameters
                    channels = wav.getnchannels()
                    wav.getsampwidth()
                    framerate = wav.getframerate()
                    num_frames = wav.getnframes()
                    
                    duration = num_frames / framerate
                    
                    # Add audio feature token
                    tokens.append(MusicToken(
                        token_type=MusicTokenType.ANALYSIS,
                        data={
                            "file": path.name,
                            "duration_seconds": duration,
                            "sample_rate": framerate,
                            "channels": channels,
                        }
                    ))
                    
                    # Read a sample of frames for simple pitch estimation
                    # Real implementation would use FFT or librosa
                    frames_to_read = min(num_frames, framerate * 10)  # First 10 seconds
                    wav.readframes(frames_to_read)
                    
                    # Very simplified pitch estimation (would use FFT in production)
                    # This is a placeholder for actual audio analysis
                    estimated_tempo = 120  # Placeholder
                    
                    tokens.append(MusicToken(
                        token_type=MusicTokenType.RHYTHM,
                        data={
                            "tempo": estimated_tempo,
                            "duration": duration,
                            "estimated_tempo_confidence": 0.5,
                        }
                    ))
                    
                    # Estimate key (simplified)
                    tokens.append(MusicToken(
                        token_type=MusicTokenType.HARMONY,
                        data={
                            "estimated_key": "C_major",  # Placeholder
                            "confidence": 0.6,
                            "mode": "major",
                        }
                    ))
                    
            elif path.suffix.lower() in ['.mp3', '.flac', '.ogg']:
                # For compressed formats, would need additional libraries
                tokens.append(MusicToken(
                    token_type=MusicTokenType.ANALYSIS,
                    data={
                        "file": path.name,
                        "format": path.suffix,
                        "note": "Compressed format - would need ffmpeg/audioread",
                    }
                ))
                
        except Exception as e:
            logger.warning(f"Audio tokenization error: {e}")
            tokens.append(MusicToken(
                token_type=MusicTokenType.NOTE,
                data={"error": safe_error_message(e), "file": path.name}
            ))
        
        return tokens
    
    async def _tokenize_musicxml(
        self,
        path: Path,
        analyze_features: bool
    ) -> list[MusicToken]:
        """Tokenize MusicXML file."""
        tokens = []
        
        try:
            content = path.read_text(encoding='utf-8')
            
            # Very simplified MusicXML parsing
            # Would use music21 or similar for full parsing
            if '<score-partwise' in content or '<score-timewise' in content:
                tokens.append(MusicToken(
                    token_type=MusicTokenType.STRUCTURE,
                    data={"format": "musicxml", "file": path.name}
                ))
            
            # Extract basic info
            import re
            titles = re.findall(r'<work-title>([^<]+)</work-title>', content)
            composers = re.findall(r'<composer>([^<]+)</composer>', content)
            
            if titles:
                tokens.append(MusicToken(
                    token_type=MusicTokenType.STYLE,
                    data={"title": titles[0], "source": path.name}
                ))
            
            if composers:
                tokens.append(MusicToken(
                    token_type=MusicTokenType.STYLE,
                    data={"composer": composers[0], "source": path.name}
                ))
                
        except Exception as e:
            logger.warning(f"MusicXML parsing error: {e}")
        
        return tokens
    
    async def analyze_harmony(self, tokens_json: str) -> dict:
        """Analyze harmonic content of music tokens."""
        try:
            tokens_data = json.loads(tokens_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON token data"}
        
        harmony_tokens = [
            t for t in tokens_data
            if t.get("type") in ["harmony", "chord"]
        ]
        
        # Extract chord progressions
        progressions = []
        for token in harmony_tokens:
            if "chords" in token.get("data", {}):
                progressions.append(token["data"]["chords"])
        
        # Analyze patterns
        pattern_counts = defaultdict(int)
        for prog in progressions:
            if isinstance(prog, list):
                prog_str = "-".join(prog)
                pattern_counts[prog_str] += 1
        
        return {
            "total_harmony_tokens": len(harmony_tokens),
            "unique_progressions": len(pattern_counts),
            "most_common": sorted(pattern_counts.items(), key=lambda x: -x[1])[:5],
            "analysis": "Harmonic analysis complete",
        }
    
    async def ingest_library(self, max_files: int = 100) -> dict:
        """Process all music files in the library for LCM learning."""
        processed = 0
        failed = 0
        total_tokens = 0
        
        music_files = list(self.music_library_path.rglob("*.[mM][iI][dD]"))
        music_files.extend(list(self.music_library_path.rglob("*.[wW][aA][vV]")))
        music_files.extend(list(self.music_library_path.rglob("*.[mM][pP]3")))
        music_files.extend(list(self.music_library_path.rglob("*.[xX][mM][lL]")))
        
        for i, file_path in enumerate(music_files[:max_files]):
            try:
                result = await self.tokenize_file(str(file_path), analyze_features=True)
                if "error" not in result:
                    processed += 1
                    total_tokens += result.get("token_count", 0)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed += 1
        
        return {
            "processed": processed,
            "failed": failed,
            "total_tokens_generated": total_tokens,
            "library_path": str(self.music_library_path),
        }


class MusicAnalyzer:
    """Analyzes musical content for patterns and insights.
    
    Provides:
    - Harmonic analysis
    - Structural analysis
    - Style classification
    - Pattern recognition
    """
    
    def __init__(self):
        self.style_features = {
            'baroque': ['counterpoint', 'ornamentation', 'figured_bass'],
            'classical': ['balance', 'symmetry', 'clarity'],
            'romantic': ['chromaticism', 'expression', 'rubato'],
        }
    
    def register_tools(self, registry):
        """Register music analysis tools."""
        registry.register_tool(
            name="analyze_composition",
            description="Analyze a musical composition for patterns and features",
            parameters={
                "type": "object",
                "properties": {
                    "composition_tokens": {
                        "type": "string",
                        "description": "JSON string of composition tokens"
                    }
                },
                "required": ["composition_tokens"]
            },
            function=self.analyze_composition,
            category="music",
        )
        
        registry.register_tool(
            name="classify_style",
            description="Classify the style period of a composition",
            parameters={
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Musical features to analyze"
                    }
                },
                "required": ["features"]
            },
            function=self.classify_style,
            category="music",
        )
    
    async def analyze_composition(self, composition_tokens: str) -> dict:
        """Analyze composition for musical patterns."""
        try:
            tokens = json.loads(composition_tokens)
        except json.JSONDecodeError:
            return {"error": "Invalid token data"}
        
        analysis = {
            "harmony": self._analyze_harmony_structure(tokens),
            "melody": self._analyze_melodic_contour(tokens),
            "rhythm": self._analyze_rhythmic_patterns(tokens),
            "structure": self._analyze_form(tokens),
        }
        
        return {
            "analysis": analysis,
            "summary": "Composition analysis complete",
        }
    
    def _analyze_harmony_structure(self, tokens: list) -> dict:
        """Analyze harmonic structure."""
        chords = [t for t in tokens if t.get("type") == "harmony"]
        return {
            "chord_count": len(chords),
            "tonal_center": "estimated from analysis",
        }
    
    def _analyze_melodic_contour(self, tokens: list) -> dict:
        """Analyze melodic contour."""
        notes = [t for t in tokens if t.get("type") == "note"]
        return {
            "note_count": len(notes),
            "contour": "ascending/descending analysis",
        }
    
    def _analyze_rhythmic_patterns(self, tokens: list) -> dict:
        """Analyze rhythmic patterns."""
        [t for t in tokens if t.get("type") == "rhythm"]
        return {
            "rhythmic_complexity": "analysis placeholder",
        }
    
    def _analyze_form(self, tokens: list) -> dict:
        """Analyze musical form."""
        structures = [t for t in tokens if t.get("type") == "structure"]
        return {
            "form": "binary/ternary/etc",
            "sections": len(structures),
        }
    
    async def classify_style(self, features: dict) -> dict:
        """Classify musical style based on features."""
        # Simplified classification
        # Real implementation would use ML model
        
        style_scores = {}
        for style, style_features in self.style_features.items():
            score = sum(
                1 for feat in style_features
                if feat in str(features).lower()
            )
            style_scores[style] = score / len(style_features)
        
        best_style = max(style_scores, key=style_scores.get)
        
        return {
            "classified_style": best_style,
            "confidence": style_scores[best_style],
            "all_scores": style_scores,
        }


# Singleton instances
_music_generator: Optional[MusicGenerator] = None
_music_tokenizer: Optional[MusicTokenizer] = None
_music_analyzer: Optional[MusicAnalyzer] = None


def get_music_generator() -> MusicGenerator:
    """Get singleton music generator instance."""
    global _music_generator
    if _music_generator is None:
        _music_generator = MusicGenerator()
    return _music_generator


def get_music_tokenizer() -> MusicTokenizer:
    """Get singleton music tokenizer instance."""
    global _music_tokenizer
    if _music_tokenizer is None:
        _music_tokenizer = MusicTokenizer()
    return _music_tokenizer


def get_music_analyzer() -> MusicAnalyzer:
    """Get singleton music analyzer instance."""
    global _music_analyzer
    if _music_analyzer is None:
        _music_analyzer = MusicAnalyzer()
    return _music_analyzer


def register_music_tools(registry):
    """Register all music tools with the tool registry.
    
    This function is called during startup to register music tools.
    Works with both ToolRegistry-style and agent.tools-style registration.
    """
    generator = get_music_generator()
    tokenizer = get_music_tokenizer()
    analyzer = get_music_analyzer()
    
    # Check if this is a ToolRegistry-style or agent-style registration
    if hasattr(registry, 'register_tool'):
        # ToolRegistry-style registration (used by tool_registry.py)
        generator.register_tools(registry)
        tokenizer.register_tools(registry)
        analyzer.register_tools(registry)
    else:
        # Agent-style registration - return list of Tool objects
        # This handles the case where agent.register_tools() is called
        tools = []
        
        # Create Tool objects for music generation
        from tools.base import Tool
        
        class MusicGenerationTool(Tool):
            name = "generate_composition"
            description = "Generate a classical music composition in various styles (baroque, classical, romantic)"
            parameters = {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": MUSIC_PERIODS,
                        "description": "Musical style period"
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key (e.g., 'C_major', 'A_minor')"
                    },
                    "measures": {
                        "type": "integer",
                        "description": "Number of measures to generate",
                        "default": 16
                    },
                    "tempo": {
                        "type": "integer",
                        "description": "Tempo in BPM",
                        "default": 120
                    }
                },
                "required": ["style"]
            }
            
            async def execute(self, style: str = "classical", key: str = "C_major", measures: int = 16, tempo: int = 120):
                gen = get_music_generator()
                return await gen.generate_composition(style, key, measures, tempo)
        
        class MusicVariationTool(Tool):
            name = "generate_variation"
            description = "Create a variation on a musical theme"
            parameters = {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "Theme notes in format: 'C4-E4-G4'"
                    },
                    "variation_type": {
                        "type": "string",
                        "enum": ["ornamentation", "inversion", "sequence", "diminution", "augmentation"],
                        "description": "Type of variation"
                    }
                },
                "required": ["theme", "variation_type"]
            }
            
            async def execute(self, theme: str, variation_type: str):
                gen = get_music_generator()
                return await gen.generate_variation(theme, variation_type)
        
        tools.append(MusicGenerationTool())
        tools.append(MusicVariationTool())
        
        # Add tokenizer and analyzer tools similarly
        class MusicTokenizeTool(Tool):
            name = "tokenize_music_file"
            description = "Convert a music file to tokens for analysis"
            parameters = {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to music file"
                    }
                },
                "required": ["file_path"]
            }
            
            async def execute(self, file_path: str):
                tok = get_music_tokenizer()
                return await tok.tokenize_file(file_path)
        
        class MusicAnalyzeTool(Tool):
            name = "analyze_harmony"
            description = "Analyze harmonic structure of music"
            parameters = {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "list",
                        "description": "List of notes to analyze"
                    }
                },
                "required": ["notes"]
            }
            
            async def execute(self, notes: list):
                ana = get_music_analyzer()
                return await ana.analyze_harmony(notes)
        
        class MusicClassifyTool(Tool):
            name = "classify_style"
            description = "Classify the style of a musical composition"
            parameters = {
                "type": "object",
                "properties": {
                    "composition": {
                        "type": "dict",
                        "description": "Composition data to classify"
                    }
                },
                "required": ["composition"]
            }
            
            async def execute(self, composition: dict):
                ana = get_music_analyzer()
                return await ana.classify_style(composition)
        
        class MusicIngestTool(Tool):
            name = "ingest_music_library"
            description = "Ingest and analyze a library of music files"
            parameters = {
                "type": "object",
                "properties": {
                    "library_path": {
                        "type": "string",
                        "description": "Path to music library"
                    }
                },
                "required": ["library_path"]
            }
            
            async def execute(self, library_path: str):
                ana = get_music_analyzer()
                return await ana.ingest_library(library_path)
        
        tools.append(MusicTokenizeTool())
        tools.append(MusicAnalyzeTool())
        tools.append(MusicClassifyTool())
        tools.append(MusicIngestTool())
        
        # Register with agent
        if hasattr(registry, 'register_tools'):
            registry.register_tools(tools)
        else:
            # Direct dict assignment
            for tool in tools:
                registry[tool.name] = tool
    
    logger.info(f"[Music] Registered {6} music tools")
    return 6