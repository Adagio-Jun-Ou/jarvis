"""Conversational Speech Model (CSM) for Jarvis.

Inspired by Sesame AI's CSM architecture, this module provides:
1. Semantic tokens: Speaker-invariant linguistic/phonetic content
2. Acoustic tokens: Fine-grained speaker identity, timbre, prosody (RVQ)
3. Tone/Intent tokens: Emotional state and conversational intent
4. Context-aware prosody: Using conversation history for natural speech

Key Architecture:
- Backbone transformer: Models zeroth codebook (semantic) with interleaved text/audio
- Audio decoder: Predicts remaining N-1 acoustic codebooks
- Conversation context: Maintains history for prosody conditioning
- Tone intent system: Captures emotional state and conversational dynamics

Token Flow:
1. Audio Input → Semantic + Acoustic + Tone tokens
2. Conversation History → Context embedding
3. LLM processing with multimodal tokens
4. Semantic + Tone → Acoustic tokens (conditioned on context + speaker)
5. Acoustic tokens → Audio output
"""
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass


class ToneIntent(IntEnum):
    """Tone and intent tokens for conversational speech.
    
    These capture the emotional state and conversational intent,
    enabling natural prosody variation beyond basic semantics.
    """
    PAD = 0
    BOS = 1
    EOS = 2

    NEUTRAL = 10
    FRIENDLY = 11
    PROFESSIONAL = 12
    EMPATHETIC = 13
    CURIOUS = 14
    THOUGHTFUL = 15

    EXCITED = 20
    CALM = 21
    CONFIDENT = 22
    UNCERTAIN = 23
    CERTAIN = 24
    APOLOGETIC = 24
    GRATEFUL = 25

    QUESTIONING = 30
    INFORMATIVE = 31
    INSTRUCTIVE = 32
    SUGGESTIVE = 33
    CONFIRMING = 34

    PAUSE_THOUGHT = 40
    PAUSE_BREATH = 41
    PAUSE_HESITATION = 42
    PAUSE_EMPHASIS = 43

    LAUGHTER = 50
    SIGH = 51
    BREATH = 52
    CLEAR_THROAT = 53

    SPEED_SLOW = 60
    SPEED_NORMAL = 61
    SPEED_FAST = 62

    VOLUME_QUIET = 70
    VOLUME_NORMAL = 71
    VOLUME_LOUD = 72

    PITCH_LOW = 80
    PITCH_NORMAL = 81
    PITCH_HIGH = 82

    IRONY = 90
    SARCASM = 91
    HUMOR = 92
    SERIOUS = 93


class ProsodyPattern(IntEnum):
    """Prosody patterns for speech generation."""
    FLAT = 0
    RISING = 1
    FALLING = 2
    RISE_FALL = 3
    FALL_RISE = 4
    LEVEL_EMPHASIS = 5
    CONTRASTIVE = 6


@dataclass
class ConversationContext:
    """Maintains conversation context for prosody conditioning."""
    speaker_history: list[str] = field(default_factory=list)
    tone_history: list[ToneIntent] = field(default_factory=list)
    turn_count: int = 0
    last_speaker: Optional[str] = None
    current_emotion: ToneIntent = ToneIntent.NEUTRAL
    conversation_energy: float = 0.5
    topic_continuity: float = 0.5

    def add_turn(self, speaker: str, tone: ToneIntent, energy: float = 0.5):
        self.speaker_history.append(speaker)
        self.tone_history.append(tone)
        self.turn_count += 1
        self.last_speaker = speaker
        self.conversation_energy = 0.7 * self.conversation_energy + 0.3 * energy
        if len(self.speaker_history) > 50:
            self.speaker_history = self.speaker_history[-50:]
            self.tone_history = self.tone_history[-50:]

    def get_context_embedding(self) -> np.ndarray:
        embed = np.zeros(64, dtype=np.float32)
        embed[0] = self.turn_count / 100.0
        embed[1] = float(self.current_emotion) / 100.0
        embed[2] = self.conversation_energy
        embed[3] = self.topic_continuity
        if len(self.tone_history) > 0:
            tone_counts = {}
            for t in self.tone_history[-10:]:
                tone_counts[t] = tone_counts.get(t, 0) + 1
            most_common = max(tone_counts, key=tone_counts.get, default=ToneIntent.NEUTRAL)
            embed[4] = float(most_common) / 100.0
        if self.last_speaker:
            embed[5] = hash(self.last_speaker) % 100 / 100.0
        return embed


@dataclass
class ConversationToken:
    """Token with conversation context."""
    token_id: int
    token_type: str  # 'semantic', 'acoustic', 'tone', 'text', 'speaker'
    position: int
    frame_idx: Optional[int] = None
    codebook_idx: Optional[int] = None


@dataclass
class ConversationalTokens:
    """Complete token representation for conversational speech."""
    semantic_tokens: list[int] = field(default_factory=list)
    acoustic_tokens: list[list[int]] = field(default_factory=list)
    tone_tokens: list[ToneIntent] = field(default_factory=list)
    prosody_patterns: list[ProsodyPattern] = field(default_factory=list)
    speaker_id: Optional[str] = None
    intent: ToneIntent = ToneIntent.NEUTRAL
    duration_ms: float = 0.0
    text: Optional[str] = None
    context_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "semantic_tokens": self.semantic_tokens,
            "acoustic_tokens": self.acoustic_tokens,
            "tone_tokens": [int(t) for t in self.tone_tokens],
            "prosody_patterns": [int(p) for p in self.prosody_patterns],
            "speaker_id": self.speaker_id,
            "intent": int(self.intent),
            "duration_ms": self.duration_ms,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationalTokens":
        return cls(
            semantic_tokens=data.get("semantic_tokens", []),
            acoustic_tokens=data.get("acoustic_tokens", []),
            tone_tokens=[ToneIntent(t) for t in data.get("tone_tokens", [])],
            prosody_patterns=[ProsodyPattern(p) for p in data.get("prosody_patterns", [])],
            speaker_id=data.get("speaker_id"),
            intent=ToneIntent(data.get("intent", 10)),
            duration_ms=data.get("duration_ms", 0.0),
            text=data.get("text"),
        )


class ToneIntentClassifier:
    """Classifies tone and intent from audio or text."""

    TONE_KEYWORDS = {
        ToneIntent.FRIENDLY: ["thanks", "great", "awesome", "love", "happy", "glad"],
        ToneIntent.QUESTIONING: ["?", "how", "what", "why", "when", "where", "who", "which"],
        ToneIntent.INFORMATIVE: ["the", "is", "are", "was", "were", "been", "being"],
        ToneIntent.INSTRUCTIVE: ["should", "could", "would", "must", "need", "have to", "please"],
        ToneIntent.APOLOGETIC: ["sorry", "apologize", "regret", "mistake", "fault"],
        ToneIntent.GRATEFUL: ["thank", "appreciate", "grateful", "thanks"],
        ToneIntent.UNCERTAIN: ["maybe", "perhaps", "might", "could", "possibly", "not sure"],
        ToneIntent.CONFIRMING: ["yes", "correct", "right", "exactly", "indeed", "absolutely"],
    }

    AUDIO_TONE_SIGNATURES = {
        ToneIntent.EXCITED: {"energy_mean": 0.6, "pitch_range": 0.4, "rate": 1.2},
        ToneIntent.CALM: {"energy_mean": 0.3, "pitch_range": 0.2, "rate": 0.9},
        ToneIntent.QUESTIONING: {"pitch_final": 0.7, "pitch_range": 0.5},
        ToneIntent.CERTAIN: {"energy_mean": 0.5, "pitch_final": 0.3, "rate": 1.0},
    }

    def classify_text(self, text: str) -> ToneIntent:
        text_lower = text.lower()
        scores = {}
        for tone, keywords in self.TONE_KEYWORDS.items():
            scores[tone] = sum(1 for kw in keywords if kw in text_lower)
        if not scores or max(scores.values()) == 0:
            return ToneIntent.NEUTRAL
        return max(scores, key=scores.get)

    def classify_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> ToneIntent:
        frame_size = int(sample_rate * 0.02)
        n_frames = len(audio) // frame_size

        if n_frames == 0:
            return ToneIntent.NEUTRAL

        energies = []
        pitches = []

        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energy = np.mean(frame ** 2)
            energies.append(energy)

            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)
            if len(fft) > 0:
                pitch = freqs[np.argmax(fft[:len(fft)//4])] if len(fft) > 4 else 0
                pitches.append(pitch)

        if len(energies) == 0:
            return ToneIntent.NEUTRAL

        energy_mean = np.mean(energies)
        np.max(pitches) - np.min(pitches) if pitches else 0
        rate = n_frames / (len(audio) / sample_rate) if len(audio) > 0 else 1.0

        if energy_mean > 0.05 and rate > 15:
            return ToneIntent.EXCITED
        if energy_mean < 0.01:
            return ToneIntent.CALM
        if len(pitches) > 2 and pitches[-1] > pitches[0] * 1.2:
            return ToneIntent.QUESTIONING

        return ToneIntent.NEUTRAL


class ProsodyAnalyzer:
    """Analyzes and generates prosody patterns for speech."""

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> list[ProsodyPattern]:
        frame_size = int(sample_rate * 0.04)
        n_frames = max(1, len(audio) // frame_size)

        patterns = []

        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]

            if len(frame) == 0:
                patterns.append(ProsodyPattern.FLAT)
                continue

            energy = np.mean(frame ** 2)
            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)

            if len(fft) < 4:
                patterns.append(ProsodyPattern.FLAT)
                continue

            pitch = freqs[np.argmax(fft[:len(fft)//4])] if len(fft) > 4 else 0

            if i == 0:
                patterns.append(ProsodyPattern.FLAT)
                continue

            patterns[-1]
            prev_frame = audio[(i-1) * frame_size:i * frame_size]
            prev_energy = np.mean(prev_frame ** 2) if len(prev_frame) > 0 else 0
            prev_fft = np.abs(np.fft.rfft(prev_frame)) if len(prev_frame) > 0 else np.array([0])
            prev_pitch = freqs[np.argmax(prev_fft[:len(prev_fft)//4])] if len(prev_fft) > 4 else 0

            if pitch > prev_pitch * 1.15:
                patterns.append(ProsodyPattern.RISING)
            elif pitch < prev_pitch * 0.85:
                patterns.append(ProsodyPattern.FALLING)
            elif energy > prev_energy * 1.3:
                patterns.append(ProsodyPattern.LEVEL_EMPHASIS)
            else:
                patterns.append(ProsodyPattern.FLAT)

        return patterns

    def generate(self, semantic_tokens: list[int], tone: ToneIntent,
                 context: Optional[ConversationContext] = None) -> list[ProsodyPattern]:
        patterns = []

        for i, token in enumerate(semantic_tokens):
            if token <= 9:
                patterns.append(ProsodyPattern.FLAT)
                continue

            position_factor = i / max(1, len(semantic_tokens) - 1)

            if tone == ToneIntent.QUESTIONING:
                if position_factor > 0.7:
                    patterns.append(ProsodyPattern.RISING)
                else:
                    patterns.append(ProsodyPattern.FLAT)
            elif tone == ToneIntent.EXCITED:
                if i % 4 < 2:
                    patterns.append(ProsodyPattern.RISE_FALL)
                else:
                    patterns.append(ProsodyPattern.LEVEL_EMPHASIS)
            elif tone == ToneIntent.CALM:
                patterns.append(ProsodyPattern.FLAT)
            elif tone == ToneIntent.INSTRUCTIVE:
                if position_factor > 0.5:
                    patterns.append(ProsodyPattern.FALLING)
                else:
                    patterns.append(ProsodyPattern.FLAT)
            elif tone == ToneIntent.UNCERTAIN:
                if i % 3 == 0:
                    patterns.append(ProsodyPattern.RISE_FALL)
                else:
                    patterns.append(ProsodyPattern.FLAT)
            else:
                energy_mod = (context.conversation_energy if context else 0.5)
                if position_factor > 0.8 and energy_mod > 0.6:
                    patterns.append(ProsodyPattern.FALLING)
                else:
                    patterns.append(ProsodyPattern.FLAT)

        return patterns


class ConversationalSpeechModel:
    """End-to-end conversational speech model.
    
    Architecture inspired by Sesame AI CSM:
    - Semantic tokenization (speaker-invariant)
    - Acoustic tokenization (RVQ codebooks)
    - Tone/intent classification
    - Conversation context modeling
    - Backbone + decoder architecture for efficient generation
    """

    FRAME_RATE = 12.5
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = int(SAMPLE_RATE / FRAME_RATE)

    SEMANTIC_VOCAB = 1024
    ACOUSTIC_CODEBOOKS = 8
    ACOUSTIC_VOCAB = 1024
    TONE_VOCAB = 128

    def __init__(self, use_npu: bool = True):
        self.use_npu = use_npu
        self._loaded = False
        self._device = "numpy"

        self.context = ConversationContext()
        self.tone_classifier = ToneIntentClassifier()
        self.prosody_analyzer = ProsodyAnalyzer()

        self._backbone = None
        self._decoder = None

    def _load(self):
        if self._loaded:
            return

        if HAS_TORCH and self.use_npu:
            try:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = "mps"

                if self._device != "numpy":
                    logger.info(f"CSM using device: {self._device}")
            except Exception as e:
                logger.debug(f"CSM device detection failed: {e}")

        self._loaded = True

    def tokenize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        speaker_id: Optional[str] = None,
        text: Optional[str] = None,
    ) -> ConversationalTokens:
        self._load()

        if sample_rate != self.SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self.SAMPLE_RATE)

        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        duration_ms = len(audio) / self.SAMPLE_RATE * 1000

        semantic = self._semantic_tokenize(audio)
        acoustic = self._acoustic_tokenize(audio)
        tone = self.tone_classifier.classify_audio(audio)
        if text:
            text_tone = self.tone_classifier.classify_text(text)
            if text_tone != ToneIntent.NEUTRAL:
                tone = text_tone
        prosody = self.prosody_analyzer.analyze(audio, self.SAMPLE_RATE)
        context_embedding = self.context.get_context_embedding()

        if speaker_id:
            self.context.add_turn(speaker_id, tone)

        return ConversationalTokens(
            semantic_tokens=semantic,
            acoustic_tokens=acoustic,
            tone_tokens=[tone] * len(semantic),
            prosody_patterns=prosody,
            speaker_id=speaker_id,
            intent=tone,
            duration_ms=duration_ms,
            text=text,
            context_embedding=context_embedding,
        )

    def _semantic_tokenize(self, audio: np.ndarray) -> list[int]:
        n_frames = max(1, len(audio) // self.FRAME_SAMPLES)
        tokens = [1]

        for i in range(n_frames):
            frame = audio[i * self.FRAME_SAMPLES:(i + 1) * self.FRAME_SAMPLES]

            energy = np.sqrt(np.mean(frame ** 2))
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame) + 1e-10)

            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1 / self.SAMPLE_RATE)

            centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
            flatness = np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10)

            energy_norm = min(1.0, energy * 10)
            zcr_norm = min(1.0, zcr * 5)
            cent_norm = min(1.0, centroid / 8000)
            flat_norm = min(1.0, flatness)

            token = int(
                energy_norm * 255 +
                zcr_norm * 255 +
                cent_norm * 255 +
                flat_norm * 255
            ) % (self.SEMANTIC_VOCAB - 10) + 10

            if energy < 0.01:
                tokens.append(ToneIntent.PAUSE_BREATH)
            tokens.append(token)

        tokens.append(2)
        return tokens

    def _acoustic_tokenize(self, audio: np.ndarray) -> list[list[int]]:
        n_frames = max(1, len(audio) // self.FRAME_SAMPLES)
        tokens = []

        for i in range(n_frames):
            frame = audio[i * self.FRAME_SAMPLES:(i + 1) * self.FRAME_SAMPLES]
            frame_tokens = []

            energy = np.mean(frame ** 2)
            frame_tokens.append(int(energy * 50000) % self.ACOUSTIC_VOCAB)

            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame) + 1e-10)
            frame_tokens.append(int(zcr * 10000) % self.ACOUSTIC_VOCAB)

            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1 / self.SAMPLE_RATE)

            centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
            frame_tokens.append(int(centroid / 20) % self.ACOUSTIC_VOCAB)

            flatness = np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10)
            frame_tokens.append(int(flatness * 1000) % self.ACOUSTIC_VOCAB)

            rolloff = 0.85
            np.sum(fft[:int(rolloff * len(fft))])
            np.sum(fft) + 1e-10
            frame_tokens.append(int(rolloff * 1000) % self.ACOUSTIC_VOCAB)

            flux = np.sum(np.abs(np.diff(fft)))
            frame_tokens.append(int(flux * 20) % self.ACOUSTIC_VOCAB)

            phase = np.angle(np.fft.fft(frame))
            phase_mean = np.mean(phase[:len(phase)//4])
            frame_tokens.append(int((phase_mean + np.pi) / (2 * np.pi) * self.ACOUSTIC_VOCAB) % self.ACOUSTIC_VOCAB)

            frame_hash = int(hashlib.sha256(frame.tobytes()).hexdigest()[:4], 16)
            frame_tokens.append(frame_hash % self.ACOUSTIC_VOCAB)

            tokens.append(frame_tokens)

        return tokens

    def _generate_acoustic(
        self,
        semantic_tokens: list[int],
        tone_tokens: list[ToneIntent],
        prosody_patterns: list[ProsodyPattern],
    ) -> list[list[int]]:
        acoustic = []

        for i, sem in enumerate(semantic_tokens):
            if sem <= 9:
                continue

            idx = min(i - 1, len(tone_tokens) - 1) if i > 0 and tone_tokens else 0
            tone = tone_tokens[idx] if idx >= 0 and idx < len(tone_tokens) else ToneIntent.NEUTRAL
            prosody = prosody_patterns[idx] if idx >= 0 and idx < len(prosody_patterns) else ProsodyPattern.FLAT

            frame_tokens = []

            codebook_0 = (sem + int(tone)) % self.ACOUSTIC_VOCAB
            frame_tokens.append(codebook_0)

            for cb in range(1, self.ACOUSTIC_CODEBOOKS):
                base = (sem * (cb + 1) + int(tone) * cb + int(prosody)) % self.ACOUSTIC_VOCAB
                variation = np.random.randint(-30, 30)
                frame_tokens.append(max(0, min(self.ACOUSTIC_VOCAB - 1, base + variation)))

            acoustic.append(frame_tokens)

        return acoustic

    def _decode_acoustic(self, acoustic_tokens: list[list[int]]) -> np.ndarray:
        n_frames = len(acoustic_tokens)
        frame_samples = self.FRAME_SAMPLES

        audio = np.zeros(n_frames * frame_samples, dtype=np.float32)

        for i, frame_tokens in enumerate(acoustic_tokens):
            if len(frame_tokens) < 4:
                continue

            energy = frame_tokens[0] / 50000.0
            frame_tokens[1] / 10000.0
            centroid = frame_tokens[2] * 20
            flatness = frame_tokens[3] / 1000.0

            t = np.arange(frame_samples) / self.SAMPLE_RATE

            if energy < 0.001:
                noise = np.random.randn(frame_samples).astype(np.float32) * 0.001
            else:
                freq = max(100, min(3000, centroid))
                signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

                harmonics = 3
                for h in range(2, harmonics + 1):
                    signal += np.sin(2 * np.pi * freq * h * t).astype(np.float32) / h

                noise = np.random.randn(frame_samples).astype(np.float32) * flatness * 0.1
                signal = signal / (np.max(np.abs(signal)) + 1e-10)
                noise = signal * np.sqrt(energy) + noise

            start = i * frame_samples
            end = min((i + 1) * frame_samples, len(audio))
            audio[start:end] = noise[:end - start]

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        ratio = target_sr / orig_sr
        n_samples = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio), n_samples),
            np.arange(len(audio)),
            audio,
        )

    def update_context(self, speaker: str, tone: ToneIntent, energy: float = 0.5):
        self.context.add_turn(speaker, tone, energy)

    def get_conversation_embedding(self) -> np.ndarray:
        return self.context.get_context_embedding()

    def classify_tone(self, text: str) -> ToneIntent:
        return self.tone_classifier.classify_text(text)

    def get_prosody_for_tone(self, tone: ToneIntent) -> dict:
        return self.tone_classifier.AUDIO_TONE_SIGNATURES.get(tone, {
            "energy_mean": 0.4, "pitch_range": 0.3, "rate": 1.0
        })

    def process_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        context_text: Optional[str] = None,
    ) -> dict:
        tokens = self.tokenize(audio, sample_rate, text=context_text)

        prosody = self.get_prosody_for_tone(tokens.intent)

        return {
            "semantic_tokens": tokens.semantic_tokens,
            "acoustic_tokens": tokens.acoustic_tokens,
            "tone_tokens": [int(t) for t in tokens.tone_tokens],
            "tone_intent": {
                "primary": tokens.intent.name,
                "confidence": 1.0,
                "prosocial": tokens.intent in (
                    ToneIntent.FRIENDLY,
                    ToneIntent.EMPATHETIC,
                    ToneIntent.GRATEFUL,
                    ToneIntent.HELPFUL if hasattr(ToneIntent, 'HELPFUL') else ToneIntent.FRIENDLY,
                ),
            },
            "speaker": {
                "identity": tokens.speaker_id or "unknown",
                "confidence": 0.9 if tokens.speaker_id else 0.0,
            },
            "prosody": prosody,
            "duration_ms": tokens.duration_ms,
        }

    def synthesize(
        self,
        text_or_tokens,
        tone: Optional[ToneIntent] = None,
        speaker_id: Optional[str] = None,
        context: Optional[ConversationContext] = None,
    ):
        if isinstance(text_or_tokens, str):
            semantic_tokens = self._text_to_semantic(text_or_tokens, tone or ToneIntent.NEUTRAL)
        else:
            semantic_tokens = text_or_tokens

        n_frames = len(semantic_tokens) - 2

        if tone is None:
            tone = ToneIntent.NEUTRAL
        tone_tokens = [tone] * n_frames
        prosody_patterns = self.prosody_analyzer.generate(semantic_tokens, tone, context)

        acoustic_tokens = self._generate_acoustic(semantic_tokens, tone_tokens, prosody_patterns)
        audio = self._decode_acoustic(acoustic_tokens)

        return {
            "audio": audio,
            "sample_rate": self.SAMPLE_RATE,
            "duration_ms": len(audio) / self.SAMPLE_RATE * 1000,
        }

    def _text_to_semantic(self, text: str, tone: ToneIntent) -> list[int]:
        tokens = [self.SEMANTIC_VOCAB - 2]
        tokens.append(int(tone))

        words = text.split()
        for word in words[:100]:
            word_hash = int(hashlib.sha256(word.encode()).hexdigest()[:6], 16)
            token = word_hash % (self.SEMANTIC_VOCAB - 20) + 10
            tokens.append(token)

            if word.endswith('?'):
                tokens.append(int(ToneIntent.QUESTIONING))
            elif word.endswith('!'):
                tokens.append(int(ToneIntent.EXCITED))
            elif word.endswith('...'):
                tokens.append(int(ToneIntent.PAUSE_THOUGHT))

        tokens.append(self.SEMANTIC_VOCAB - 1)
        return tokens

    def get_stats(self) -> dict:
        return {
            "device": self._device,
            "frame_rate": self.FRAME_RATE,
            "sample_rate": self.SAMPLE_RATE,
            "semantic_vocab": self.SEMANTIC_VOCAB,
            "acoustic_codebooks": self.ACOUSTIC_CODEBOOKS,
            "acoustic_vocab": self.ACOUSTIC_VOCAB,
            "tone_vocab": self.TONE_VOCAB,
            "context_turns": self.context.turn_count,
        }


class ConversationManager:
    """Manages conversation state for multi-turn speech interactions."""

    def __init__(self):
        self.csm = ConversationalSpeechModel()
        self._speakers: dict[str, list[list[int]]] = {}
        self._speaker_embeddings: dict[str, np.ndarray] = {}
        self._conversation_history: list[ConversationalTokens] = []

    def process_turn(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        speaker_id: str = "user",
        text: Optional[str] = None,
    ) -> ConversationalTokens:
        tokens = self.csm.tokenize(audio, sample_rate, speaker_id, text)

        if speaker_id and len(tokens.acoustic_tokens) > 0:
            self._register_speaker_tokens(speaker_id, tokens.acoustic_tokens[-50:])

        self._conversation_history.append(tokens)
        if len(self._conversation_history) > 100:
            self._conversation_history = self._conversation_history[-100:]

        return tokens

    def generate_response(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        tone: Optional[ToneIntent] = None,
    ) -> tuple[np.ndarray, int]:
        if tone is None:
            tone = self._infer_tone_from_context()

        self.csm.get_conversation_embedding()

        semantic_tokens = self._text_to_semantic(text, tone)

        acoustic_tokens = None
        if speaker_id and speaker_id in self._speaker_embeddings:
            acoustic_tokens = self._generate_with_speaker(
                semantic_tokens, speaker_id, tone
            )

        if acoustic_tokens is None:
            prosody = self._generate_prosody(semantic_tokens, tone)
            acoustic_tokens = self.csm._generate_acoustic(
                semantic_tokens,
                [tone] * len(semantic_tokens),
                prosody,
            )

        audio = self.csm._decode_acoustic(acoustic_tokens)

        return audio, self.csm.SAMPLE_RATE

    def _infer_tone_from_context(self) -> ToneIntent:
        if len(self._conversation_history) == 0:
            return ToneIntent.NEUTRAL

        recent_tones = [
            t.intent for t in self._conversation_history[-5:]
        ]
        if not recent_tones:
            return ToneIntent.NEUTRAL

        from collections import Counter
        tone_counts = Counter([int(t) for t in recent_tones])
        most_common = tone_counts.most_common(1)
        if most_common:
            return ToneIntent(most_common[0][0])
        return ToneIntent.NEUTRAL

    def _text_to_semantic(self, text: str, tone: ToneIntent) -> list[int]:
        tokens = [ConversationalSpeechModel.SEMANTIC_VOCAB - 2]
        tokens.append(int(tone))

        words = text.split()
        for word in words[:100]:
            word_hash = int(hashlib.sha256(word.encode()).hexdigest()[:6], 16)
            token = word_hash % (ConversationalSpeechModel.SEMANTIC_VOCAB - 20) + 10
            tokens.append(token)

            if word.endswith('?'):
                tokens.append(int(ToneIntent.QUESTIONING))
            elif word.endswith('!'):
                tokens.append(int(ToneIntent.EXCITED))
            elif word.endswith('...'):
                tokens.append(int(ToneIntent.PAUSE_THOUGHT))

        tokens.append(ConversationalSpeechModel.SEMANTIC_VOCAB - 1)
        return tokens

    def _generate_prosody(
        self,
        semantic_tokens: list[int],
        tone: ToneIntent,
    ) -> list[ProsodyPattern]:
        return self.csm.prosody_analyzer.generate(
            semantic_tokens, tone, self.csm.context
        )

    def _register_speaker_tokens(self, speaker_id: str, tokens: list[list[int]]):
        if speaker_id not in self._speakers:
            self._speakers[speaker_id] = []
        self._speakers[speaker_id].extend(tokens)
        self._speakers[speaker_id] = self._speakers[speaker_id][-200:]

        self._speaker_embeddings[speaker_id] = self._compute_embedding(tokens)

    def _compute_embedding(self, tokens: list[list[int]]) -> np.ndarray:
        if not tokens:
            return np.zeros(128, dtype=np.float32)

        arr = np.array(tokens, dtype=np.float32)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        max_vals = np.max(arr, axis=0)

        hist, _ = np.histogram(arr.flatten(), bins=104, range=(0, 1024))

        embedding = np.concatenate([mean, std, max_vals, hist / max(1, len(tokens))])
        embedding = embedding[:128].astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _generate_with_speaker(
        self,
        semantic_tokens: list[int],
        speaker_id: str,
        tone: ToneIntent,
    ) -> list[list[int]]:
        speaker_acoustic = self._speakers.get(speaker_id, [])
        if not speaker_acoustic:
            return None

        embedding = self._speaker_embeddings.get(speaker_id)
        if embedding is None:
            return None

        acoustic = []
        for i, sem in enumerate(semantic_tokens):
            if sem <= 9:
                continue

            idx = i % len(speaker_acoustic) if speaker_acoustic else 0
            frame_base = speaker_acoustic[idx] if idx < len(speaker_acoustic) else [0] * 8

            frame_tokens = []
            for cb in range(8):
                if cb < len(frame_base):
                    base = frame_base[cb]
                else:
                    base = (sem + int(tone) * cb) % 1024

                variation = np.random.randint(-50, 50)
                token = max(0, min(1023, base + variation))
                frame_tokens.append(token)

            acoustic.append(frame_tokens)

        return acoustic

    def get_stats(self) -> dict:
        return {
            "csm": self.csm.get_stats(),
            "speakers_registered": len(self._speakers),
            "conversation_history": len(self._conversation_history),
        }


_csm: Optional[ConversationalSpeechModel] = None
_manager: Optional[ConversationManager] = None
_lock = threading.Lock()


def get_csm(use_npu: bool = True) -> ConversationalSpeechModel:
    global _csm
    if _csm is None:
        with _lock:
            if _csm is None:
                _csm = ConversationalSpeechModel(use_npu=use_npu)
    return _csm


get_conversational_speech_model = get_csm


def get_conversation_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = ConversationManager()
    return _manager


__all__ = [
    "ToneIntent",
    "ProsodyPattern",
    "ConversationContext",
    "ConversationalTokens",
    "ToneIntentClassifier",
    "ProsodyAnalyzer",
    "ConversationalSpeechModel",
    "ConversationManager",
    "get_csm",
    "get_conversation_manager",
]
