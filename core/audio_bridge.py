"""EnCodec Audio Bridge - Unified audio-text representation for Jarvis.

EnCodec (from Meta AudioCraft) tokenizes audio into discrete codes,
enabling the LCM to reason about audio in the same way it reasons
about text. This creates a unified multi-modal representation space
where the LCM can:
1. Understand music creation engine output as concept sequences
2. Bridge Whisper STT output to the dual-model architecture
3. Reason about TTS/Piper output quality
4. Cross-reference audio concepts with text concepts

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                    ENCODEC AUDIO BRIDGE                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Audio Sources                                                    │
│   ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐        │
│   │ Whisper  │ │ Music   │ │ Piper    │ │  Microphone   │        │
│   │ STT      │ │ Engine  │ │ TTS      │ │  Input       │        │
│   └────┬────┘ └────┬────┘ └────┬─────┘ └──────┬───────┘        │
│        │            │           │               │                   │
│        └────────────┴───────────┴───────────────┘                   │
│                          │                                         │
│               ┌──────────▼──────────┐                             │
│               │  AUDIO TOKENIZER     │                             │
│               │  (Discrete Codes)    │                             │
│               │  - 4 codebooks       │                             │
│               │  - 50Hz frame rate    │                             │
│               │  - ~150 tokens/sec   │                             │
│               └──────────┬──────────┘                             │
│                          │                                         │
│               ┌──────────▼──────────┐                             │
│               │  SEMANTIC PROJECTOR  │                             │
│               │  (Audio → Text Emb)  │                             │
│               │  Maps audio codes    │                             │
│               │  to text embedding   │                             │
│               │  space (768-dim)     │                             │
│               └──────────┬──────────┘                             │
│                          │                                         │
│               ┌──────────▼──────────┐                             │
│               │  LCM CONCEPT SPACE   │                             │
│               │  (Unified 768-dim)    │                             │
│               │  Audio + Text +      │                             │
│               │  Code concepts        │                             │
│               │  all in same space    │                             │
│               └──────────┬──────────┘                             │
│                          │                                         │
│           ┌──────────────┼──────────────┐                         │
│           ▼              ▼              ▼                          │
│    Primary LLM      Backend LLM    Music Engine                   │
│    (chat about      (analyze      (generate from                │
│     audio)           audio)        concepts)                    │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

Note: Full EnCodec model is ~300MB. For edge deployment, we use a
lightweight "Semantic Projector" that maps acoustic features to the
existing 768-dim embedding space without loading the full model.
"""
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_audio_bridge: Optional["AudioBridge"] = None
_audio_bridge_lock = threading.Lock()


class AudioModality(Enum):
    SPEECH_INPUT = "speech_input"
    SPEECH_OUTPUT = "speech_output"
    MUSIC = "music"
    AMBIENT = "ambient"
    SYSTEM = "system"


@dataclass
class AudioTokenSequence:
    """A sequence of audio tokens from EnCodec-style encoding."""
    id: str
    modality: AudioModality
    tokens: list[int] = field(default_factory=list)
    frame_rate: int = 50
    duration_seconds: float = 0.0
    codebook_count: int = 4
    semantic_embedding: np.ndarray | None = None
    text_transcript: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_token_density(self) -> float:
        """Get tokens per second."""
        if self.duration_seconds <= 0:
            return 0.0
        return len(self.tokens) / self.duration_seconds
    
    def to_concept_dict(self) -> dict:
        """Convert to a dict suitable for LCM concept encoding."""
        return {
            "id": self.id,
            "type": f"audio_{self.modality.value}",
            "token_count": len(self.tokens),
            "duration": self.duration_seconds,
            "density": self.get_token_density(),
            "transcript": self.text_transcript[:500],
            "codebooks": self.codebook_count,
        }


@dataclass 
class AudioBridgeStats:
    total_sequences: int = 0
    speech_sequences: int = 0
    music_sequences: int = 0
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    avg_projection_ms: float = 0.0
    lcm_integrations: int = 0


class SemanticProjector:
    """Lightweight audio-to-text embedding projector.
    
    Maps acoustic features (MFCC-like from audio tokens) to the
    768-dim text embedding space without loading a full EnCodec model.
    
    Uses a simple linear projection + normalization that can be
    replaced with a trained model later.
    """
    
    def __init__(self, input_dim: int = 256, output_dim: int = 768):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._projection: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._lock = threading.Lock()
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Initialize the projection matrix (lazy init)."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            rng = np.random.default_rng(42)
            scale = np.sqrt(2.0 / self.input_dim)
            self._projection = rng.standard_normal((self.input_dim, self.output_dim)) * scale
            self._bias = np.zeros(self.output_dim)
            self._initialized = True
            
            logger.info(f"SemanticProjector initialized: {self.input_dim}→{self.output_dim}")
    
    def project(self, audio_features: np.ndarray) -> np.ndarray:
        """Project audio features to text embedding space.
        
        Args:
            audio_features: Shape (batch, input_dim) or (input_dim,)
            
        Returns:
            Projected embedding shape (batch, output_dim) or (output_dim,)
        """
        self._ensure_initialized()
        
        if self._projection is None:
            return np.zeros(self.output_dim)
        
        features = np.atleast_2d(audio_features)
        projected = features @ self._projection + self._bias
        
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / (norms + 1e-8)
        
        if audio_features.ndim == 1:
            return projected[0]
        return projected
    
    def tokens_to_features(self, tokens: list[int], sequence_length: int | None = None) -> np.ndarray:
        """Convert discrete audio tokens to a feature vector.
        
        Uses token frequency histogram as the primary features,
        similar to bag-of-words but for audio codes.
        """
        if not tokens:
            return np.zeros(self.input_dim)
        
        hist = np.zeros(self.input_dim)
        
        for token in tokens:
            idx = abs(token) % self.input_dim
            hist[idx] += 1
        
        if len(tokens) > 0:
            hist = hist / len(tokens)
        
        if sequence_length and sequence_length > 0:
            n_chunks = min(len(tokens) // sequence_length, 10)
            if n_chunks > 1:
                chunk_features = []
                for i in range(n_chunks):
                    chunk = tokens[i * sequence_length:(i + 1) * sequence_length]
                    chunk_hist = np.zeros(self.input_dim)
                    for t in chunk:
                        idx = abs(t) % self.input_dim
                        chunk_hist[idx] += 1
                    if chunk:
                        chunk_hist = chunk_hist / len(chunk)
                    chunk_features.append(chunk_hist)
                
                mean_feature = np.mean(chunk_features, axis=0)
                var_feature = np.var(chunk_features, axis=0)
                hist = np.concatenate([mean_feature, var_feature])[:self.input_dim]
        
        return hist


class AudioBridge:
    """Bridge between audio systems and the LCM concept space.
    
    Enables the LCM to reason about audio by:
    1. Tokenizing audio from all sources (STT, TTS, Music Engine)
    2. Projecting audio tokens to the shared 768-dim embedding space
    3. Creating LCM concepts from audio that can be used alongside text
    4. Cross-referencing audio concepts with text concepts
    """
    
    def __init__(self, embedding_dim: int = 768):
        self._projector = SemanticProjector(output_dim=embedding_dim)
        self._stats = AudioBridgeStats()
        self._lock = threading.RLock()
        
        self._sequences: OrderedDict[str, AudioTokenSequence] = OrderedDict()
        self._max_sequences = 100
        
        self._audio_embeddings: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_embeddings = 200
    
    def tokenize_speech_input(self, text: str, duration_seconds: float = 0.0) -> AudioTokenSequence:
        """Create audio token sequence from speech input (Whisper STT output).
        
        Uses text-derived pseudo-tokens since we have the transcript,
        not the raw audio. This is a practical approach for reasoning
        about what the user said.
        """
        tokens = self._text_to_audio_tokens(text, AudioModality.SPEECH_INPUT)
        
        if duration_seconds <= 0:
            duration_seconds = len(text.split()) * 0.15
        
        embedding = self._project_audio_tokens(tokens)
        
        seq = AudioTokenSequence(
            id=f"speech_in_{int(time.time()*1000)}",
            modality=AudioModality.SPEECH_INPUT,
            tokens=tokens,
            frame_rate=50,
            duration_seconds=duration_seconds,
            semantic_embedding=embedding,
            text_transcript=text,
        )
        
        with self._lock:
            self._stats.total_sequences += 1
            self._stats.speech_sequences += 1
            self._stats.total_tokens += len(tokens)
            self._stats.total_duration_seconds += duration_seconds
            
            self._sequences[seq.id] = seq
            while len(self._sequences) > self._max_sequences:
                self._sequences.popitem(last=False)
        
        return seq
    
    def tokenize_music_output(self, composition_data: dict) -> AudioTokenSequence:
        """Create audio token sequence from music engine output.
        
        Converts music generation data (notes, chords, dynamics) into
        audio tokens that can be processed by the LCM.
        """
        notes = composition_data.get("notes", [])
        chords = composition_data.get("chords", [])
        duration = composition_data.get("duration_seconds", 0.0)
        style = composition_data.get("style", "classical")
        title = composition_data.get("title", "Untitled")
        
        tokens = []
        for note in notes:
            if isinstance(note, str):
                tokens.extend(ord(c) % 1024 for c in note)
            elif isinstance(note, (int, float)):
                tokens.append(int(note) % 1024)
        
        for chord in chords:
            if isinstance(chord, str):
                tokens.extend(ord(c) % 512 for c in chord)
            elif isinstance(chord, (int, float)):
                tokens.append(int(chord) % 512)
        
        embedding = self._project_audio_tokens(tokens)
        
        seq = AudioTokenSequence(
            id=f"music_{int(time.time()*1000)}",
            modality=AudioModality.MUSIC,
            tokens=tokens,
            frame_rate=50,
            duration_seconds=duration,
            semantic_embedding=embedding,
            text_transcript=f"[Music: {title}, Style: {style}, Duration: {duration:.1f}s]",
            metadata={
                "style": style,
                "note_count": len(notes),
                "chord_count": len(chords),
            },
        )
        
        with self._lock:
            self._stats.total_sequences += 1
            self._stats.music_sequences += 1
            self._stats.total_tokens += len(tokens)
            self._stats.total_duration_seconds += duration
            
            self._sequences[seq.id] = seq
            while len(self._sequences) > self._max_sequences:
                self._sequences.popitem(last=False)
        
        return seq
    
    async def observe_with_lcm(self, sequence: AudioTokenSequence) -> bool:
        """Feed audio token sequence to the LCM for concept-level processing.
        
        This is the key integration point: audio tokens become LCM concepts
        that the Prefrontal Cortex can reason about alongside text concepts.
        """
        try:
            from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType
            pfc = get_prefrontal_cortex()
            if not pfc:
                return False
            
            concept_type_map = {
                AudioModality.SPEECH_INPUT: ConceptType.USER_QUERY,
                AudioModality.SPEECH_OUTPUT: ConceptType.MODEL_RESPONSE,
                AudioModality.MUSIC: ConceptType.MUSIC_TOKEN,
                AudioModality.AMBIENT: ConceptType.OBSERVATION,
                AudioModality.SYSTEM: ConceptType.SUBSYSTEM_EVENT,
            }
            
            concept_type = concept_type_map.get(sequence.modality, ConceptType.OBSERVATION)
            
            content = sequence.text_transcript or sequence.to_concept_dict()
            if isinstance(content, dict):
                content = str(content)
            
            await pfc.observe(
                content=content,
                concept_type=concept_type,
                source=f"audio_bridge:{sequence.modality.value}",
            )
            
            with self._lock:
                self._stats.lcm_integrations += 1
            
            if sequence.semantic_embedding is not None:
                emb_key = f"audio_{sequence.id}"
                self._audio_embeddings[emb_key] = sequence.semantic_embedding
                while len(self._audio_embeddings) > self._max_embeddings:
                    self._audio_embeddings.popitem(last=False)
            
            return True
        except Exception as e:
            logger.debug(f"LCM audio observation failed: {e}")
            return False
    
    def find_similar_audio(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Find audio sequences similar to a given embedding.
        
        Used for cross-modal retrieval: given a text embedding,
        find relevant audio sequences, or vice versa.
        """
        with self._lock:
            items = list(self._audio_embeddings.items())
        
        results = []
        query = query_embedding.flatten()
        
        for key, emb in items:
            emb_flat = emb.flatten()
            if len(query) != len(emb_flat):
                min_dim = min(len(query), len(emb_flat))
                sim = float(np.dot(query[:min_dim], emb_flat[:min_dim]) / 
                           (np.linalg.norm(query[:min_dim]) * np.linalg.norm(emb_flat[:min_dim]) + 1e-8))
            else:
                sim = float(np.dot(query, emb_flat) / 
                           (np.linalg.norm(query) * np.linalg.norm(emb_flat) + 1e-8))
            results.append((key, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _text_to_audio_tokens(self, text: str, modality: AudioModality) -> list[int]:
        """Convert text to pseudo-audio tokens.
        
        Uses character-level encoding with modality prefix,
        simulating EnCodec's discrete token output.
        """
        modality_offset = {
            AudioModality.SPEECH_INPUT: 0,
            AudioModality.SPEECH_OUTPUT: 1024,
            AudioModality.MUSIC: 2048,
            AudioModality.AMBIENT: 3072,
            AudioModality.SYSTEM: 4096,
        }
        
        offset = modality_offset.get(modality, 0)
        tokens = []
        
        for char in text[:4000]:
            code = ord(char)
            tokens.append(offset + (code % 1024))
        
        return tokens
    
    def _project_audio_tokens(self, tokens: list[int]) -> np.ndarray:
        """Project audio tokens through the semantic projector."""
        start = time.time()
        features = self._projector.tokens_to_features(tokens, sequence_length=100)
        embedding = self._projector.project(features)
        
        projection_ms = (time.time() - start) * 1000
        with self._lock:
            self._stats.avg_projection_ms = (
                self._stats.avg_projection_ms * 0.9 + projection_ms * 0.1
            )
        
        return embedding
    
    def get_stats(self) -> dict:
        """Get audio bridge statistics."""
        with self._lock:
            return {
                "total_sequences": self._stats.total_sequences,
                "speech_sequences": self._stats.speech_sequences,
                "music_sequences": self._stats.music_sequences,
                "total_tokens": self._stats.total_tokens,
                "total_duration_seconds": round(self._stats.total_duration_seconds, 1),
                "avg_projection_ms": round(self._stats.avg_projection_ms, 2),
                "lcm_integrations": self._stats.lcm_integrations,
                "stored_embeddings": len(self._audio_embeddings),
                "stored_sequences": len(self._sequences),
            }


def get_audio_bridge() -> AudioBridge:
    """Get the global AudioBridge instance."""
    global _audio_bridge, _audio_bridge_lock
    if _audio_bridge is None:
        with _audio_bridge_lock:
            if _audio_bridge is None:
                _audio_bridge = AudioBridge()
    return _audio_bridge