"""Prefrontal Cortex - Large Concept Model for Jarvis.

This module implements a tiny LCM-inspired architecture that serves as the
"prefrontal cortex" of Jarvis, operating at the concept level rather than
token level.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JARVIS PREFRONTAL CORTEX                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│   │  Primary LLM    │    │  Backend LLM    │    │   Subsystems    │         │
│   │  (Kimi K2.5)    │    │   (GLM-5)       │    │ (Erdős, etc.)   │         │
│   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│            │                      │                      │                   │
│            └──────────────────────┼──────────────────────┘                   │
│                                   │                                          │
│                          ┌────────▼────────┐                                │
│                          │     CONCEPT     │                                │
│                          │    ENCODER      │                                │
│                          │  (SONAR-style)  │                                │
│                          │    768-dim      │                                │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                          ┌────────▼────────┐                                │
│                          │  CONCEPT CACHE  │                                │
│                          │  (Training Data)│                                │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                          ┌────────▼────────┐                                │
│                          │   LCM CORE      │                                │
│                          │  (Transformer)  │                                │
│                          │  ~100M params   │                                │
│                          └────────┬────────┘                                │
│                                   │                                          │
│            ┌──────────────────────┼──────────────────────┐                   │
│            │                      │                      │                   │
│   ┌────────▼────────┐    ┌────────▼────────┐    ┌──────▼─────────┐         │
│   │  Next Concept   │    │   Meta-Guidance │    │   Anomaly      │         │
│   │  Prediction     │    │   Generation    │    │   Detection    │         │
│   └─────────────────┘    └─────────────────┘    └────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Key Concepts:
1. CONCEPT = A semantic unit (sentence, action, intention, observation)
2. LCM operates on concept embeddings, not tokens
3. Learns to predict next concept based on sequence of past concepts
4. Provides meta-reasoning, guidance, and anomaly detection
5. Continuously trained on all Jaris activity

Based on Meta's LCM paper: https://arxiv.org/abs/2412.08821
GitHub: https://github.com/facebookresearch/large_concept_model
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from core.helpers import create_tracked_task, read_json_with_integrity
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Singleton
_prefrontal_cortex: Optional["PrefrontalCortex"] = None
_pfc_lock = threading.Lock()

# Default paths
DEFAULT_MODEL_PATH = Path.home() / ".jarvis" / "prefrontal_cortex"
DEFAULT_TRAINING_BUFFER = Path.home() / ".jarvis" / "pfc_training_buffer"


class ConceptType(Enum):
    """Types of concepts the cortex processes."""
    USER_QUERY = "user_query"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INSIGHT = "insight"
    MEMORY = "memory"
    SUBSYSTEM_EVENT = "subsystem_event"
    ERROR = "error"
    DECISION = "decision"
    OBSERVATION = "observation"
    INTENTION = "intention"
    MUSIC_TOKEN = "music_token"  # Musical concepts for LCM


@dataclass
class Concept:
    """A semantic unit in the concept space.
    
    Unlike tokens, concepts represent complete thoughts, actions, or observations.
    """
    id: str
    embedding: np.ndarray  # 768-dim concept embedding
    concept_type: ConceptType
    content: str  # Original text content
    source: str  # Where this concept came from
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Relationships
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    
    # Learning signal
    reward: float = 0.0  # How useful was this concept?
    frequency: int = 1  # How often has this pattern appeared?
    _MAX_RELATED = 10
    _MAX_CHILDREN = 50


@dataclass
class ConceptSequence:
    """A sequence of concepts forming a thought process."""
    concepts: list[Concept]
    context_id: str  # Links to conversation/session
    goal: Optional[str] = None
    outcome: Optional[str] = None
    success: bool = False


@dataclass  
class CortexState:
    """Current state of the prefrontal cortex."""
    _MAX_PREDICTIONS = 5
    _MAX_ATTENTION_KEYS = 100

    active_concepts: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_sequences: deque = field(default_factory=lambda: deque(maxlen=20))
    attention_weights: dict = field(default_factory=dict)
    predicted_next: list[Concept] = field(default_factory=list)
    confidence: float = 0.0
    training_steps: int = 0
    total_concepts_processed: int = 0
    total_semantic_tokens: int = 0
    anomaly_count: int = 0


class ConceptEncoder:
    """Encodes inputs into concept embeddings.
    
    Uses existing embedding infrastructure (Jina v5) but with concept-level
    semantics. The key innovation is encoding complete thoughts/actions,
    not just text.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self._embedder = None
        self._lock = threading.Lock()
        
        # Concept type embeddings (learned)
        self.type_embeddings = self._init_type_embeddings()
        
        # Position in sequence (learned)
        self.position_embeddings = self._init_position_embeddings()
    
    def _init_type_embeddings(self) -> dict[ConceptType, np.ndarray]:
        """Initialize learnable concept type embeddings."""
        type_embeddings = {}
        rng = np.random.RandomState(hash("type_embeddings") % (2**31))
        for ct in ConceptType:
            type_embeddings[ct] = rng.randn(self.embedding_dim) * 0.1
        return type_embeddings
    
    def _init_position_embeddings(self, max_positions: int = 100) -> np.ndarray:
        """Initialize learnable position embeddings."""
        rng = np.random.RandomState(42)
        return rng.randn(max_positions, self.embedding_dim) * 0.05
    
    def _get_embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is not None:
            return self._embedder
        
        with self._lock:
            if self._embedder is not None:
                return self._embedder
            
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                ei = get_embedding_interconnect()
                # The embedder attribute has the actual encoding capability
                self._embedder = ei.embedder if hasattr(ei, 'embedder') else ei
                logger.debug("ConceptEncoder using EmbeddingInterconnect")
            except Exception as e:
                logger.warning(f"Could not get embedding interconnect: {e}")
                self._embedder = None
            
            return self._embedder
    
    async def encode(
        self,
        content: str,
        concept_type: ConceptType,
        source: str,
        metadata: Optional[dict] = None,
    ) -> Concept:
        """Encode content into a concept.
        
        The concept embedding combines:
        1. Content embedding (from Jina v5 or hash fallback)
        2. Type embedding (learned)
        3. Source embedding (derived from source string)
        4. Metadata embeddings (if provided)
        """
        embedder = self._get_embedder()
        
        # Get base content embedding
        if embedder:
            try:
                # EmbeddingInterconnect uses 'encode' method
                content_emb = embedder.encode(content)
                if content_emb is not None:
                    content_emb = np.asarray(content_emb).flatten()
                else:
                    content_emb = np.zeros(self.embedding_dim)
                
                # Check if embedding is valid (not all zeros)
                if np.linalg.norm(content_emb) < 0.001:
                    # Fallback to hash encoding
                    content_emb = self._hash_encode(content)
            except Exception as e:
                logger.debug(f"Embedding failed: {e}")
                content_emb = self._hash_encode(content)
        else:
            # Fallback: hash-based embedding
            content_emb = self._hash_encode(content)
        
        # Add type embedding
        type_emb = self.type_embeddings.get(concept_type, np.zeros(self.embedding_dim))
        
        # Add source embedding (hash-based)
        source_emb = self._hash_encode(source) * 0.1
        
        # Combine embeddings
        combined = content_emb + type_emb + source_emb
        
        # Normalize (avoid division by zero or NaN)
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined = combined / norm
        else:
            # If norm is too small, use content embedding with small type/source
            combined = content_emb * 0.7 + type_emb * 0.2 + source_emb * 0.1
            norm = np.linalg.norm(combined)
            if norm > 1e-8:
                combined = combined / norm
        
        # Create concept ID
        concept_id = hashlib.sha256(
            f"{concept_type.value}:{content[:100]}:{source}".encode()
        ).hexdigest()[:16]
        
        return Concept(
            id=concept_id,
            embedding=combined,
            concept_type=concept_type,
            content=content,
            source=source,
            metadata=metadata or {},
        )
    
    def _hash_encode(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text hash."""
        # Use SHA-256 to get deterministic values
        h = hashlib.sha256(text.encode()).digest()
        # Convert to float32 safely (normalize to [-1, 1])
        raw_values = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # Normalize to [-0.1, 0.1] for stable embeddings
        values = (raw_values / 255.0) * 0.2 - 0.1
        # Expand to 768 dimensions
        result = np.zeros(self.embedding_dim, dtype=np.float32)
        while len(values) < self.embedding_dim:
            h = hashlib.sha256(h).digest()
            raw_values = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            more_values = (raw_values / 255.0) * 0.2 - 0.1
            values = np.concatenate([values, more_values])
        result = values[:self.embedding_dim].astype(np.float32)
        return result
    
    def add_position_embedding(self, concept: Concept, position: int) -> np.ndarray:
        """Add position embedding to concept."""
        pos_emb = self.position_embeddings[min(position, len(self.position_embeddings) - 1)]
        return concept.embedding + pos_emb


class LCMCore:
    """The core Large Concept Model transformer.
    
    A tiny transformer that operates on concept sequences:
    - Input: sequence of concept embeddings
    - Output: predicted next concept embedding
    
    Architecture:
    - 4-6 transformer layers
    - ~100M parameters
    - Trained continuously on Jarvis activity
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self._model = None
        self._initialized = False
        self._lock = threading.Lock()
        
        # Training buffer
        self._training_buffer: deque = deque(maxlen=10000)
        self._loss_history: deque = deque(maxlen=1000)
    
    def _init_model(self):
        """Initialize the transformer model."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                import torch
                import torch.nn as nn
                
                class TinyLCM(nn.Module):
                    """Tiny LCM transformer for concept prediction."""
                    
                    def __init__(self, embedding_dim, num_layers, num_heads, hidden_dim, dropout):
                        super().__init__()
                        
                        self.embedding_dim = embedding_dim
                        
                        # Input projection
                        self.input_proj = nn.Linear(embedding_dim, embedding_dim)
                        
                        # Transformer layers
                        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=embedding_dim,
                            nhead=num_heads,
                            dim_feedforward=hidden_dim,
                            dropout=dropout,
                            batch_first=True,
                        )
                        self.transformer = nn.TransformerEncoder(
                            encoder_layer,
                            num_layers=num_layers,
                        )
                        
                        # Output projection
                        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
                        
                        # Prediction head
                        self.prediction_head = nn.Sequential(
                            nn.Linear(embedding_dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, embedding_dim),
                        )
                        
                        # Type prediction head (auxiliary task)
                        self.type_head = nn.Linear(embedding_dim, len(ConceptType))
                        
                        # Value head (for RL-style learning)
                        self.value_head = nn.Linear(embedding_dim, 1)
                    
                    def forward(self, x, mask=None):
                        """Forward pass.
                        
                        Args:
                            x: (batch, seq_len, embedding_dim)
                            mask: Optional attention mask
                        
                        Returns:
                            next_embedding: (batch, embedding_dim)
                            type_logits: (batch, num_types)
                            value: (batch, 1)
                        """
                        # Project input
                        h = self.input_proj(x)
                        
                        # Apply transformer
                        h = self.transformer(h, src_key_padding_mask=mask)
                        
                        # Use last hidden state for prediction
                        last_hidden = h[:, -1, :]  # (batch, embedding_dim)
                        
                        # Predict next concept embedding
                        next_embedding = self.prediction_head(last_hidden)
                        
                        # Auxiliary predictions
                        type_logits = self.type_head(last_hidden)
                        value = self.value_head(last_hidden)
                        
                        return next_embedding, type_logits, value
                    
                    def parameters_count(self):
                        return sum(p.numel() for p in self.parameters())
                
                self._model = TinyLCM(
                    self.embedding_dim,
                    self.num_layers,
                    self.num_heads,
                    self.hidden_dim,
                    self.dropout,
                )
                
                # Try to load saved weights
                self._load_weights()
                
                self._initialized = True
                params = self._model.parameters_count()
                logger.info(f"LCM Core initialized: {params:,} parameters")
                
            except ImportError as e:
                logger.warning(f"PyTorch not available for LCM: {e}")
                self._model = None
                self._initialized = True
    
    def _load_weights(self):
        """Load saved model weights."""
        model_path = DEFAULT_MODEL_PATH / "lcm_weights.pt"
        if model_path.exists():
            try:
                import torch
                state_dict = torch.load(model_path, map_location="cpu")
                if self._model is not None:
                    self._model.load_state_dict(state_dict)
                    logger.info(f"Loaded LCM weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LCM weights: {e}")
    
    def save_weights(self):
        """Save model weights."""
        if self._model is None:
            return
        
        try:
            import torch
            DEFAULT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            model_path = DEFAULT_MODEL_PATH / "lcm_weights.pt"
            torch.save(self._model.state_dict(), model_path)
            logger.info(f"Saved LCM weights to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save LCM weights: {e}")
    
    def predict_next(
        self,
        concept_sequence: list[Concept],
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[tuple[ConceptType, float]], float]:
        """Predict the next concept given a sequence.
        
        Args:
            concept_sequence: List of concepts forming the context
            top_k: Number of top predictions to return
        
        Returns:
            predicted_embedding: The predicted next concept embedding
            predicted_types: List of (concept_type, probability) tuples
            confidence: Model's confidence in the prediction
        """
        self._init_model()
        
        if self._model is None or len(concept_sequence) == 0:
            # Fallback: return average of recent concepts
            if len(concept_sequence) > 0:
                avg_emb = np.mean([c.embedding for c in concept_sequence], axis=0)
                return avg_emb, [(ConceptType.INTENTION, 0.5)], 0.1
            return np.zeros(self.embedding_dim), [(ConceptType.INTENTION, 0.5)], 0.0
        
        try:
            import torch
            
            # Prepare input
            embeddings = [c.embedding for c in concept_sequence[-50:]]  # Last 50 concepts
            # Convert to single numpy array first to avoid warning
            embeddings_array = np.array([embeddings], dtype=np.float32)
            x = torch.from_numpy(embeddings_array)  # (1, seq_len, dim)
            
            # Forward pass
            self._model.eval()
            with torch.no_grad():
                next_emb, type_logits, value = self._model(x)
            
            # Convert to numpy
            predicted_embedding = next_emb.numpy()[0]
            type_probs = torch.softmax(type_logits, dim=-1).numpy()[0]
            confidence = torch.sigmoid(value).item()
            
            # Get top-k type predictions
            type_indices = np.argsort(type_probs)[-top_k:][::-1]
            predicted_types = [
                (list(ConceptType)[i], float(type_probs[i]))
                for i in type_indices
            ]
            
            return predicted_embedding, predicted_types, confidence
            
        except Exception as e:
            logger.warning(f"LCM prediction failed: {e}")
            return np.zeros(self.embedding_dim), [(ConceptType.INTENTION, 0.5)], 0.0
    
    def add_training_example(
        self,
        context: list[Concept],
        target: Concept,
        reward: float = 1.0,
    ):
        """Add a training example to the buffer."""
        example = {
            "context": [c.embedding for c in context[-50:]],
            "context_types": [c.concept_type for c in context[-50:]],
            "target": target.embedding,
            "target_type": target.concept_type,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock:
            self._training_buffer.append(example)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step.
        
        Returns:
            Loss value or None if not enough data
        """
        self._init_model()
        
        if self._model is None:
            return None
        
        # Sample batch under lock, then train outside lock
        with self._lock:
            if len(self._training_buffer) < batch_size:
                return None
            
            import random
            batch = random.sample(list(self._training_buffer), batch_size)
        
        try:
            import torch
            import torch.nn as nn
            
            # Prepare tensors
            max_len = max(len(ex["context"]) for ex in batch)
            
            # Pad sequences
            context_tensors = []
            target_tensors = []
            type_tensors = []
            reward_tensors = []
            
            for ex in batch:
                ctx = ex["context"]
                pad_len = max_len - len(ctx)
                padded = list(ctx) + [np.zeros(self.embedding_dim, dtype=np.float32)] * pad_len
                context_tensors.append(np.array(padded, dtype=np.float32))
                target_tensors.append(np.array(ex["target"], dtype=np.float32))
                type_tensors.append(list(ConceptType).index(ex["target_type"]))
                reward_tensors.append(ex["reward"])
            
            context = torch.from_numpy(np.array(context_tensors, dtype=np.float32))
            targets = torch.from_numpy(np.array(target_tensors, dtype=np.float32))
            type_targets = torch.tensor(type_tensors, dtype=torch.long)
            rewards = torch.tensor(reward_tensors, dtype=torch.float32).unsqueeze(1)
            
            # Forward pass
            self._model.train()
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)
            
            pred_emb, type_logits, pred_value = self._model(context)
            
            # Losses
            mse_loss = nn.MSELoss()(pred_emb, targets)
            ce_loss = nn.CrossEntropyLoss()(type_logits, type_targets)
            value_loss = nn.MSELoss()(pred_value, rewards)
            
            total_loss = mse_loss + 0.5 * ce_loss + 0.1 * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()
            
            loss_val = total_loss.item()
            self._loss_history.append(loss_val)
            
            return loss_val
            
        except Exception as e:
            logger.warning(f"LCM training step failed: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        stats = {
            "initialized": self._initialized,
            "training_buffer_size": len(self._training_buffer),
            "loss_history_len": len(self._loss_history),
        }
        
        if self._loss_history:
            stats["avg_loss"] = sum(self._loss_history) / len(self._loss_history)
            stats["recent_loss"] = list(self._loss_history)[-10:]
        
        if self._model:
            stats["parameters"] = self._model.parameters_count()
        
        return stats


class PrefrontalCortex:
    """The Prefrontal Cortex of Jarvis.
    
    Central hub that:
    1. Observes ALL activity across models and subsystems
    2. Builds concept sequences from observations
    3. Predicts next concepts (intentions, actions)
    4. Provides meta-guidance to other components
    5. Detects anomalies and unexpected patterns
    6. Continuously learns from experience
    
    This replaces/enhances the concept_model.py with LCM-style architecture.
    """
    
    def __init__(
        self,
        auto_train: bool = True,
        train_interval: float = 60.0,  # Train every 60 seconds
    ):
        self.encoder = ConceptEncoder()
        self.lcm = LCMCore()
        self.state = CortexState()
        
        self._auto_train = auto_train
        self._train_interval = train_interval
        self._running = False
        self._lock = threading.RLock()
        
        # Concept storage
        self._concepts: OrderedDict[str, Concept] = OrderedDict()
        self._max_concepts = 10000
        
        # Concept deduplication — embedding similarity threshold
        self._dedup_threshold = 0.92
        self._DEDUP_CHECK_WINDOW = 50  # Only check recent N concepts
        self._dedup_hits = 0
        self._dedup_checks = 0
        
        # Concept decay — time-based forgetting
        self._DECAY_HALF_LIFE = 1.0  # 1 hour half-life for access frequency (in hours)
        self._DECAY_MIN_IMPORTANCE = 0.1  # Below this, concept can be evicted early
        self._decay_last_run = time.time()
        self._DECAY_INTERVAL = 300  # Run decay every 5 minutes
        
        # Sequence tracking - bounded to prevent memory leak
        self._current_sequence: deque[str] = deque(maxlen=1000)  # Was list, now bounded deque
        self._sequence_goal: Optional[str] = None
        
        # Callbacks - protected by lock, bounded to prevent unbounded growth
        self._on_prediction_callbacks: list[Callable] = []
        self._on_anomaly_callbacks: list[Callable] = []
        self._MAX_CALLBACKS = 20
        self._callback_lock = threading.Lock()  # Protect callback lists
        
        # Training
        self._last_train_time = time.time()
        self._train_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the prefrontal cortex."""
        self._running = True
        
        # Load concepts from previous session
        self._load_concepts()
        
        logger.info("Prefrontal Cortex started")
        logger.info("  LCM params: ~100M")
        logger.info(f"  Auto-train: {self._auto_train}")
        logger.info(f"  Train interval: {self._train_interval}s")
        
        if self._auto_train:
            self._train_task = create_tracked_task(self._training_loop(), name="training_loop")
    
    async def stop(self):
        """Stop the prefrontal cortex."""
        self._running = False
        if self._train_task:
            self._train_task.cancel()
        
        # Save weights
        self.lcm.save_weights()
        
        # Save concepts for persistence across restarts
        self._save_concepts()
        
        logger.info("Prefrontal Cortex stopped")
    
    def _save_concepts(self):
        """Save concepts to disk for persistence across restarts."""
        from core.helpers import atomic_write_json
        
        concepts_file = DEFAULT_MODEL_PATH / "concepts.json"
        try:
            DEFAULT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            
            # Snapshot data under lock, write outside lock
            with self._lock:
                concepts_data = []
                for cid, concept in list(self._concepts.items())[-1000:]:
                    concepts_data.append({
                        "id": concept.id,
                        "type": concept.concept_type.value,
                        "content": concept.content[:500],
                        "source": concept.source,
                        "timestamp": (concept.timestamp.isoformat() if isinstance(concept.timestamp, datetime) else datetime.fromtimestamp(concept.timestamp).isoformat()),
                        "parent_id": concept.parent_id,
                    })
                
                state_data = {
                    "total_concepts": self.state.total_concepts_processed,
                    "total_semantic_tokens": self.state.total_semantic_tokens,
                    "training_steps": self.state.training_steps,
                    "confidence": self.state.confidence,
                    "dedup_checks": self._dedup_checks,
                    "dedup_hits": self._dedup_hits,
                    "anomaly_count": self.state.anomaly_count,
                }
                
                save_data = {
                    "concepts": concepts_data,
                    "state": state_data,
                }
            
            atomic_write_json(concepts_file, save_data)
            logger.info(f"Saved {len(concepts_data)} concepts to {concepts_file}")
        except Exception as e:
            logger.warning(f"Failed to save concepts: {e}")
    
    def _load_concepts(self):
        """Load concepts from disk on startup."""
        
        concepts_file = DEFAULT_MODEL_PATH / "concepts.json"
        if not concepts_file.exists():
            return
        
        try:
            save_data = read_json_with_integrity(concepts_file)
            
            # Load state counters for continuity
            state_data = save_data.get("state", {})
            if state_data:
                self.state.total_concepts_processed = state_data.get("total_concepts", 0)
                self.state.total_semantic_tokens = state_data.get("total_semantic_tokens", 0)
                self.state.training_steps = state_data.get("training_steps", 0)
                self.state.confidence = state_data.get("confidence", 0.0)
                self.state.anomaly_count = state_data.get("anomaly_count", 0)
                self._dedup_checks = state_data.get("dedup_checks", 0)
                self._dedup_hits = state_data.get("dedup_hits", 0)
            
            # Load concepts — re-encode embeddings using content text
            concepts_data = save_data.get("concepts", [])
            
            for cdata in concepts_data:
                concept = Concept(
                    id=cdata["id"],
                    embedding=np.zeros(768),
                    concept_type=ConceptType(cdata["type"]),
                    content=cdata["content"],
                    source=cdata["source"],
                    timestamp=datetime.fromisoformat(cdata["timestamp"]),
                    parent_id=cdata.get("parent_id"),
                )
                self._concepts[concept.id] = concept
            
            # Re-encode loaded concepts asynchronously using the encoder
            self._reencode_loaded_concepts(list(self._concepts.values()))
            
            logger.info(f"Loaded {len(concepts_data)} concepts from {concepts_file}")
            logger.info(f"  Total concepts processed: {self.state.total_concepts_processed}")
            logger.info(f"  Semantic tokens: {self.state.total_semantic_tokens}")
            logger.info(f"  Training steps: {self.state.training_steps}")
        except Exception as e:
            logger.warning(f"Failed to load concepts: {e}")
    
    def _reencode_loaded_concepts(self, concepts: list[Concept], batch_size: int = 50):
        """Re-encode loaded concepts using the encoder to restore proper embeddings.

        Uses the hash-based fallback encoder since we're in a synchronous context.
        The LLM encoder will restore embeddings lazily during later observe() calls.
        """
        try:
            for concept in concepts:
                try:
                    encode_text = f"{concept.concept_type.value}:{concept.content}:{concept.source}"
                    embedding = self.encoder._hash_encode(encode_text)
                    concept.embedding = embedding
                except Exception as e:
                    logger.warning(f"Failed to re-encode concept {concept.id}: {e}")
            reencoded = sum(1 for c in concepts if np.any(c.embedding))
            logger.info(f"Re-encoded {reencoded}/{len(concepts)} loaded concepts via hash fallback")
        except Exception as e:
            logger.warning(f"Concept re-encoding failed: {e}")
    
    def _find_similar_concept(self, concept: Concept) -> Optional[str]:
        """Check if a semantically similar concept already exists.
        
        Uses cosine similarity on embeddings to detect near-duplicates.
        Returns the ID of the similar concept if found, None otherwise.
        """
        if concept.embedding is None or not np.any(concept.embedding):
            return None
        
        self._dedup_checks += 1
        best_sim = 0.0
        best_id = None
        
        # Only check recent concepts to keep O(n) bounded
        recent_ids = list(self._concepts.keys())[-self._DEDUP_CHECK_WINDOW:]
        
        for cid in recent_ids:
            if cid == concept.id:
                continue
            existing = self._concepts.get(cid)
            if existing is None or existing.embedding is None or not np.any(existing.embedding):
                continue
            if existing.concept_type != concept.concept_type:
                continue
            
            # Cosine similarity
            dot = np.dot(concept.embedding, existing.embedding)
            norm_new = np.linalg.norm(concept.embedding)
            norm_old = np.linalg.norm(existing.embedding)
            if norm_new == 0 or norm_old == 0:
                continue
            sim = dot / (norm_new * norm_old)
            
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        
        if best_sim >= self._dedup_threshold and best_id is not None:
            self._dedup_hits += 1
            return best_id
        return None
    
    def _apply_concept_decay(self) -> int:
        """Apply exponential decay to concept importance scores and prune low-value concepts.
        
        Implements exponential decay with half-life, boosting concepts
        that are accessed frequently. Returns number of concepts pruned.
        """
        with self._lock:
            now = datetime.now()
            if (now - datetime.fromtimestamp(self._decay_last_run)).total_seconds() < self._DECAY_INTERVAL:
                return 0
            self._decay_last_run = now.timestamp()
            
            pruned = 0
            to_evict = []
            
            for cid, concept in list(self._concepts.items()):
                # Calculate time since last access
                age_hours = (now - (concept.timestamp if isinstance(concept.timestamp, datetime) else datetime.fromtimestamp(concept.timestamp))).total_seconds() / 3600.0
                
                # Exponential decay: importance = frequency * 0.5^(age / half_life)
                decay_factor = 0.5 ** (age_hours / self._DECAY_HALF_LIFE)
                importance = concept.frequency * decay_factor
                
                # Boost concepts with rewards
                importance += concept.reward * 0.1
                
                # Mark very low importance concepts for eviction
                if importance < self._DECAY_MIN_IMPORTANCE and age_hours > 1.0:
                    to_evict.append((cid, importance))
            
            # Evict lowest importance concepts first
            to_evict.sort(key=lambda x: x[1])
            for cid, _ in to_evict:
                if len(self._concepts) > self._max_concepts * 0.8:
                    del self._concepts[cid]
                    pruned += 1
                else:
                    break
            
            if pruned > 0:
                logger.info(f"Concept decay pruned {pruned} low-importance concepts")
            
            return pruned
    
    def get_concept_importance(self, concept_id: str) -> float:
        """Get the current importance score for a concept."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            return 0.0
        ts = concept.timestamp if isinstance(concept.timestamp, datetime) else datetime.fromtimestamp(concept.timestamp)
        age_hours = (datetime.now() - ts).total_seconds() / 3600.0
        decay_factor = 0.5 ** (age_hours / self._DECAY_HALF_LIFE)
        return concept.frequency * decay_factor + concept.reward * 0.1

    def _cross_link_concept(self, concept: Concept, max_links: int = 3):
        """Cross-link a new concept to semantically related existing concepts.
        
        Uses a lower threshold than dedup to find concepts that are related
        but not duplicates, building a semantic network.
        """
        if concept.embedding is None or not np.any(concept.embedding):
            return
        
        link_threshold = self._dedup_threshold * 0.85  # Lower than dedup threshold
        seen = {concept.id}
        if concept.parent_id:
            seen.add(concept.parent_id)
        seen.update(concept.child_ids)
        
        recent_ids = list(self._concepts.keys())[-self._DEDUP_CHECK_WINDOW:]
        
        for cid in recent_ids:
            if cid in seen or cid == concept.id:
                continue
            existing = self._concepts.get(cid)
            if existing is None or existing.embedding is None or not np.any(existing.embedding):
                continue
            
            dot = np.dot(concept.embedding, existing.embedding)
            norm_new = np.linalg.norm(concept.embedding)
            norm_old = np.linalg.norm(existing.embedding)
            if norm_new == 0 or norm_old == 0:
                continue
            
            sim = dot / (norm_new * norm_old)
            if sim >= link_threshold and len(concept.related_ids) < max_links:
                concept.related_ids.append(cid)
                if len(existing.related_ids) < Concept._MAX_RELATED:
                    existing.related_ids.append(concept.id)
                seen.add(cid)
    
    # ==================== OBSERVATION ====================
    
    async def observe(
        self,
        content: str,
        concept_type: ConceptType,
        source: str,
        metadata: Optional[dict] = None,
        parent_id: Optional[str] = None,
    ) -> Concept:
        """Observe an event and create a concept.
        
        This is the main entry point for all cortex observations.
        Every significant event in Jarvis should be observed.
        """
        # Create concept
        concept = await self.encoder.encode(content, concept_type, source, metadata)
        concept.parent_id = parent_id
        
        # Store concept (with dedup check and decay)
        with self._lock:
            # Check for semantic duplicates
            similar_id = self._find_similar_concept(concept)
            if similar_id and similar_id in self._concepts:
                existing = self._concepts[similar_id]
                existing.frequency += 1
                existing.reward = max(existing.reward, concept.reward)
                # Re-insert at end to maintain recency ordering
                self._concepts.move_to_end(similar_id)
                concept = existing
            else:
                self._concepts[concept.id] = concept
                if parent_id and parent_id in self._concepts:
                    self._concepts[parent_id].child_ids.append(concept.id)
                    if len(self._concepts[parent_id].child_ids) > Concept._MAX_CHILDREN:
                        self._concepts[parent_id].child_ids = self._concepts[parent_id].child_ids[-Concept._MAX_CHILDREN:]
                
                # Cross-link semantically related concepts (weaker threshold than dedup)
                self._cross_link_concept(concept)
            
            # Evict old concepts (respecting decay — low importance first)
            while len(self._concepts) > self._max_concepts:
                self._concepts.popitem(last=False)
            
            concept_to_register = None
            tags_to_register = None

            self.state.active_concepts.append(concept.id)
            self._current_sequence.append(concept.id)
            self.state.total_concepts_processed += 1

            semantic_tokens = len(concept.content) // 4
            self.state.total_semantic_tokens += semantic_tokens

            concept_to_register = concept
            tags_to_register = {concept.concept_type.value, source}

        if concept_to_register and tags_to_register:
            try:
                from core.memory_consolidation import get_consolidation_engine
                consolidation = get_consolidation_engine()
                consolidation.register_memory(
                    concept_to_register.id,
                    weight=concept_to_register.reward,
                    tags=tags_to_register,
                )
            except Exception as e:
                logger.debug(f"Failed to register memory consolidation: {e}")
        
        # Apply concept decay periodically
        self._apply_concept_decay()
        
        # Log music token observations for visibility
        if concept_type == ConceptType.MUSIC_TOKEN:
            logger.info(f"[PFC] Observed music token: {content[:60]}... (style={metadata.get('style') if metadata else 'unknown'})")
        
        # Add training example for ALL concept types (not just tools)
        # Build context from recent concepts and use current concept as target
        recent_concepts = list(self._get_recent_concepts(count=10))
        if len(recent_concepts) >= 2:
            # Use previous concepts as context, current concept as target
            context = recent_concepts[:-1]  # All but the last
            target = recent_concepts[-1]    # The most recent (current)
            
            # Reward based on concept type importance
            reward_map = {
                ConceptType.USER_QUERY: 1.0,
                ConceptType.MODEL_RESPONSE: 0.8,
                ConceptType.TOOL_CALL: 1.0,
                ConceptType.TOOL_RESULT: 0.9,
                ConceptType.INSIGHT: 1.2,  # High value
                ConceptType.MEMORY: 0.6,
                ConceptType.SUBSYSTEM_EVENT: 0.7,
                ConceptType.DECISION: 0.9,
                ConceptType.ERROR: 0.5,
                ConceptType.OBSERVATION: 0.7,
                ConceptType.INTENTION: 0.8,
                ConceptType.MUSIC_TOKEN: 1.1,  # Creative/artistic concepts
            }
            reward = reward_map.get(concept_type, 0.7)
            
            self.lcm.add_training_example(context, target, reward)
        
        # Predict next concept
        predictions = await self._predict_sequence()
        
        # Check for anomalies
        anomaly = await self._detect_anomaly(concept, predictions)
        if anomaly:
            with self._callback_lock:
                callbacks = list(self._on_anomaly_callbacks)
            for callback in callbacks:
                try:
                    await callback(concept, anomaly)
                except Exception as e:
                    logger.warning(f"Anomaly callback failed: {e}")
            
            # Feed anomaly signal to Thought Sense (unexpected user behavior)
            try:
                from core.thought_sense import get_thought_sense
                ts = get_thought_sense()
                ts.observe(
                    content=f"[anomaly] unexpected {concept.concept_type.value}: {concept.content[:100]}",
                    concept_type="anomaly",
                    source="pfc",
                )
            except Exception as e:
                logger.debug(f"Failed to send anomaly to thought sense: {e}")
        
        # Notify prediction callbacks
        with self._callback_lock:
            pred_callbacks = list(self._on_prediction_callbacks)
        for callback in pred_callbacks:
            try:
                await callback(concept, predictions)
            except Exception as e:
                logger.warning(f"Prediction callback failed: {e}")
        
        return concept
    
    async def observe_model_call(
        self,
        model: str,
        prompt: str,
        response: str,
        task_type: Optional[str] = None,
    ):
        """Observe a model API call."""
        from core.helpers import get_response_content
        
        # Observe the prompt as user intention
        prompt_concept = await self.observe(
            content=prompt[:500],
            concept_type=ConceptType.INTENTION,
            source=f"model:{model}",
            metadata={"model": model, "task_type": task_type},
        )
        
        # Observe the response
        response_content = get_response_content({"content": response}) if isinstance(response, dict) else response
        await self.observe(
            content=response_content[:500] if isinstance(response_content, str) else str(response_content)[:500],
            concept_type=ConceptType.MODEL_RESPONSE,
            source=f"model:{model}",
            metadata={
                "model": model,
                "task_type": task_type,
                "parent_id": prompt_concept.id,
            },
            parent_id=prompt_concept.id,
        )
    
    async def observe_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        success: bool,
    ):
        """Observe a tool execution."""
        # Observe tool call
        call_concept = await self.observe(
            content=f"{tool_name}({list(args.keys())})",
            concept_type=ConceptType.TOOL_CALL,
            source=f"tool:{tool_name}",
            metadata={"args": str(args)[:200], "success": success},
        )
        
        # Observe result
        await self.observe(
            content=result[:500],
            concept_type=ConceptType.TOOL_RESULT,
            source=f"tool:{tool_name}",
            metadata={"success": success},
            parent_id=call_concept.id,
        )
        
        # Add to training buffer with reward
        if success:
            self.lcm.add_training_example(
                list(self._get_recent_concepts(count=10)),
                call_concept,
                reward=1.0,
            )
    
    async def observe_insight(
        self,
        insight: str,
        source: str,
        metadata: Optional[dict] = None,
    ):
        """Observe an insight from subsystems."""
        await self.observe(
            content=insight[:500],
            concept_type=ConceptType.INSIGHT,
            source=source,
            metadata=metadata,
        )
    
    async def observe_subsystem_event(
        self,
        subsystem: str,
        event: str,
        metadata: Optional[dict] = None,
    ):
        """Observe a subsystem event (Erdős, Daydream, etc.)."""
        await self.observe(
            content=f"{subsystem}: {event[:200]}",
            concept_type=ConceptType.SUBSYSTEM_EVENT,
            source=f"subsystem:{subsystem}",
            metadata=metadata,
        )
    
    # ==================== PREDICTION ====================
    
    async def _predict_sequence(self) -> list[tuple[ConceptType, float]]:
        """Predict what concepts should come next."""
        recent_concepts = list(self._get_recent_concepts(count=50))

        if len(recent_concepts) < 2:
            return [(ConceptType.INTENTION, 0.5)]

        _, predicted_types, confidence = self.lcm.predict_next(recent_concepts)

        predictions = [
            Concept(
                id=f"pred_{i}",
                embedding=np.zeros(768),
                concept_type=ct,
                content="",
                source="prediction",
            )
            for i, (ct, prob) in enumerate(predicted_types[:5])
        ][:5]
        with self._lock:
            self.state.predicted_next = predictions
            self.state.confidence = confidence

        return predicted_types
    
    async def predict_next_action(self) -> dict:
        """Predict the next action Jarvis should take.
        
        Returns guidance for the next action based on learned patterns.
        """
        predicted_types = await self._predict_sequence()
        
        # Map concept types to actions
        action_map = {
            ConceptType.INTENTION: "wait_for_input",
            ConceptType.TOOL_CALL: "execute_tool",
            ConceptType.INSIGHT: "consider_insight",
            ConceptType.MODEL_RESPONSE: "generate_response",
            ConceptType.ERROR: "handle_error",
            ConceptType.DECISION: "make_decision",
            ConceptType.OBSERVATION: "observe_environment",
        }
        
        return {
            "predicted_actions": [
                {
                    "action": action_map.get(ct, "unknown"),
                    "concept_type": ct.value,
                    "probability": prob,
                }
                for ct, prob in predicted_types[:3]
            ],
            "confidence": self.state.confidence,
            "context_size": len(self._current_sequence),
        }
    
    # ==================== ANOMALY DETECTION ====================
    
    async def _detect_anomaly(
        self,
        concept: Concept,
        predictions: list[tuple[ConceptType, float]],
    ) -> Optional[dict]:
        """Detect if observed concept is anomalous.
        
        An anomaly is when the observed concept type was not predicted
        with high probability.
        """
        if not predictions:
            return None
        
        # Check if observed type matches predictions
        predicted_types = {ct for ct, prob in predictions if prob > 0.2}
        
        if concept.concept_type not in predicted_types:
            with self._lock:
                self.state.anomaly_count += 1
            return {
                "type": "unexpected_concept_type",
                "observed": concept.concept_type.value,
                "predicted": [(ct.value, prob) for ct, prob in predictions[:3]],
                "content_preview": concept.content[:100],
            }
        
        return None
    
    # ==================== META-GUIDANCE ====================
    
    async def get_guidance(self, context: str) -> dict:
        """Get meta-guidance for a decision.
        
        Analyzes recent concept patterns and provides guidance.
        """
        recent = list(self._get_recent_concepts(count=20))
        
        if len(recent) < 3:
            return {"guidance": "insufficient_context", "confidence": 0.0}
        
        # Count concept types
        type_counts = {}
        for c in recent:
            t = c.concept_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Predict next
        predictions = await self._predict_sequence()
        
        return {
            "recent_patterns": type_counts,
            "predicted_next": [(ct.value, prob) for ct, prob in predictions[:5]],
            "confidence": self.state.confidence,
            "sequence_length": len(self._current_sequence),
            "guidance": self._generate_guidance_text(type_counts, predictions),
        }
    
    def _generate_guidance_text(
        self,
        type_counts: dict,
        predictions: list[tuple[ConceptType, float]],
    ) -> str:
        """Generate human-readable guidance."""
        if not predictions:
            return "No prediction available"
        
        top_type, top_prob = predictions[0]
        
        guidance_templates = {
            ConceptType.TOOL_CALL: f"Consider executing a tool (confidence: {top_prob:.0%})",
            ConceptType.INTENTION: f"User may have follow-up intent (confidence: {top_prob:.0%})",
            ConceptType.INSIGHT: f"A new insight may emerge (confidence: {top_prob:.0%})",
            ConceptType.MODEL_RESPONSE: f"Prepare to generate response (confidence: {top_prob:.0%})",
            ConceptType.ERROR: f"Watch for potential errors (confidence: {top_prob:.0%})",
            ConceptType.DECISION: f"A decision point approaching (confidence: {top_prob:.0%})",
            ConceptType.OBSERVATION: f"Continue observing (confidence: {top_prob:.0%})",
        }
        
        return guidance_templates.get(top_type, f"Next: {top_type.value} ({top_prob:.0%})")
    
    # ==================== SEQUENCE TRACKING ====================
    
    def start_sequence(self, goal: Optional[str] = None):
        """Start tracking a new concept sequence."""
        with self._lock:
            self._current_sequence = deque(maxlen=1000)
            self._sequence_goal = goal
    
    async def end_sequence(self, outcome: str, success: bool):
        """End the current sequence and record for training."""
        with self._lock:
            if not self._current_sequence:
                return
            concepts = [
                self._concepts[cid]
                for cid in self._current_sequence
                if cid in self._concepts
            ]
            sequence_goal = self._sequence_goal
            self._current_sequence = deque(maxlen=1000)
            self._sequence_goal = None

        if len(concepts) < 2:
            return

        for i in range(1, len(concepts)):
            context = concepts[:i]
            target = concepts[i]
            reward = 1.0 if success else 0.5
            self.lcm.add_training_example(context, target, reward)

        sequence = ConceptSequence(
            concepts=concepts,
            context_id=hashlib.sha256(str(sequence_goal).encode()).hexdigest()[:8],
            goal=sequence_goal,
            outcome=outcome,
            success=success,
        )
        with self._lock:
            self.state.recent_sequences.append(sequence)
    
    def _get_recent_concepts(self, count: int = 10) -> list[Concept]:
        """Get recent concepts from active sequence (thread-safe)."""
        with self._lock:
            concept_ids = list(self.state.active_concepts)[-count:]
            return [
                self._concepts[cid]
                for cid in concept_ids
                if cid in self._concepts
            ]
    
    # ==================== TRAINING ====================
    
    async def _training_loop(self):
        """Background training loop with thread pool offloading."""
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pfc_train")
        
        while self._running:
            try:
                await asyncio.sleep(self._train_interval)
                
                # Lower threshold for faster training start
                min_buffer = 8
                if len(self.lcm._training_buffer) >= min_buffer:
                    batch_size = min(32, len(self.lcm._training_buffer))
                    
                    # Offload CPU-intensive training to thread pool
                    loop = asyncio.get_running_loop()
                    loss = await loop.run_in_executor(
                        executor,
                        self.lcm.train_step,
                        batch_size
                    )
                    
                    if loss is not None:
                        self.state.training_steps += 1
                        if self.state.training_steps % 10 == 0:
                            logger.debug(f"PFC training step {self.state.training_steps}, loss: {loss:.4f}")
                        
                        # Save concepts and weights periodically during training
                        if self.state.training_steps % 50 == 0:
                            self.lcm.save_weights()
                            self._save_concepts()
                            logger.info(f"PFC checkpoint: {self.state.training_steps} steps, {len(self._concepts)} concepts, {self.state.total_semantic_tokens} tokens")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"PFC training error: {e}")
                await asyncio.sleep(10)
        
        executor.shutdown(wait=False)
    
    # ==================== CALLBACKS ====================
    
    def on_prediction(self, callback: Callable):
        """Register a callback for predictions."""
        with self._callback_lock:
            self._on_prediction_callbacks.append(callback)
            if len(self._on_prediction_callbacks) > self._MAX_CALLBACKS:
                self._on_prediction_callbacks = self._on_prediction_callbacks[-self._MAX_CALLBACKS:]
    
    def on_anomaly(self, callback: Callable):
        """Register a callback for anomalies."""
        with self._callback_lock:
            self._on_anomaly_callbacks.append(callback)
            if len(self._on_anomaly_callbacks) > self._MAX_CALLBACKS:
                self._on_anomaly_callbacks = self._on_anomaly_callbacks[-self._MAX_CALLBACKS:]
    
    # ==================== INTEGRATION ====================
    
    async def get_context_for_query(self, query: str) -> dict:
        """Get prefrontal context for a query.
        
        Used by agent.think() to inform decisions.
        """
        await self._predict_sequence()
        guidance = await self.get_guidance(query)
        
        # Get importance-weighted recent concepts
        important_concepts = []
        with self._lock:
            for cid in list(self.state.active_concepts)[-10:]:
                if cid in self._concepts:
                    c = self._concepts[cid]
                    importance = self.get_concept_importance(cid)
                    important_concepts.append({
                        "type": c.concept_type.value,
                        "importance": round(importance, 3),
                        "content_preview": c.content[:50],
                    })
        
        return {
            "recent_patterns": guidance.get("recent_patterns", {}),
            "predicted_next": guidance.get("predicted_next", []),
            "confidence": guidance.get("confidence", 0.0),
            "suggested_action": guidance.get("guidance", ""),
            "sequence_length": len(self._current_sequence),
            "important_concepts": important_concepts[:5],
            "dedup_hit_rate": self._dedup_hits / max(1, self._dedup_checks),
        }
    
    def get_stats(self) -> dict:
        """Get cortex statistics."""
        return {
            "total_concepts": self.state.total_concepts_processed,
            "stored_concepts": len(self._concepts),
            "active_concepts": len(self.state.active_concepts),
            "recent_sequences": len(self.state.recent_sequences),
            "training_steps": self.state.training_steps,
            "current_confidence": self.state.confidence,
            "total_semantic_tokens": self.state.total_semantic_tokens,
            "dedup_checks": self._dedup_checks,
            "dedup_hits": self._dedup_hits,
            "dedup_hit_rate": self._dedup_hits / max(1, self._dedup_checks),
            "anomaly_count": self.state.anomaly_count,
            "lcm_stats": self.lcm.get_stats(),
        }

    def get_related_concepts(self, concept_id: str, depth: int = 1, max_results: int = 10) -> list[dict]:
        """Traverse the concept relationship graph from a given concept.

        Args:
            concept_id: Starting concept ID
            depth: How many relationship hops to follow (1=direct, 2=two hops)
            max_results: Maximum number of results to return

        Returns:
            List of dicts with concept_id, content, type, importance, and distance
        """
        with self._lock:
            if concept_id not in self._concepts:
                return []

            visited = {concept_id}
            result = []
            current_level = [concept_id]

            for d in range(1, depth + 1):
                next_level = []
                for cid in current_level:
                    concept = self._concepts.get(cid)
                    if concept is None:
                        continue
                    for rid in concept.related_ids:
                        if rid in visited or rid not in self._concepts:
                            continue
                        visited.add(rid)
                        next_level.append(rid)
                        related = self._concepts[rid]
                        importance = self.get_concept_importance(rid)
                        result.append({
                            "id": rid,
                            "content": related.content[:100],
                            "type": related.concept_type.value,
                            "importance": round(importance, 3),
                            "distance": d,
                        })
                current_level = next_level

            result.sort(key=lambda x: x["importance"], reverse=True)
            return result[:max_results]

    def get_concept_cluster(self, concept_id: str, similarity_threshold: float = 0.8) -> list[str]:
        """Find a cluster of concepts similar to the given concept.

        Uses embedding similarity to find all concepts within the threshold,
        building a connected component in the similarity graph.

        Returns:
            List of concept IDs in the cluster, sorted by importance
        """
        with self._lock:
            if concept_id not in self._concepts:
                return []

            seed = self._concepts[concept_id]
            if seed.embedding is None or not np.any(seed.embedding):
                return [concept_id]

            cluster = {concept_id}
            frontier = [concept_id]

            while frontier and len(cluster) < 50:
                cid = frontier.pop(0)
                concept = self._concepts.get(cid)
                if concept is None or concept.embedding is None:
                    continue

                for other_id, other in list(self._concepts.items())[-self._DEDUP_CHECK_WINDOW:]:
                    if other_id in cluster or other.embedding is None:
                        continue
                    dot = np.dot(concept.embedding, other.embedding)
                    n1 = np.linalg.norm(concept.embedding)
                    n2 = np.linalg.norm(other.embedding)
                    if n1 == 0 or n2 == 0:
                        continue
                    sim = dot / (n1 * n2)
                    if sim >= similarity_threshold:
                        cluster.add(other_id)
                        frontier.append(other_id)

            sorted_cluster = sorted(cluster, key=lambda x: self.get_concept_importance(x), reverse=True)
            return sorted_cluster
    
    def get_predictions_for_thought_sense(self) -> dict:
        """Get prediction data for Thought Sense integration.
        
        Returns concept predictions and anomaly information that Thought Sense
        can use to anticipate user needs and cognitive states.
        """
        with self._lock:
            recent_concepts = list(self._get_recent_concepts(count=10))
            
            result = {
                "active_concepts": len(self.state.active_concepts),
                "confidence": self.state.confidence,
                "predicted_next": [],
                "concept_types": [c.concept_type.value for c in recent_concepts[-5:]],
                "anomaly_flags": [],
                "topic_drift": 0.0,
            }
            
            if recent_concepts:
                recent_types = [c.concept_type for c in recent_concepts]
                unique_types = len(set(recent_types))
                total = len(recent_types)
                result["topic_drift"] = unique_types / max(total, 1)
            
            if self.state.predicted_next:
                result["predicted_next"] = [
                    {
                        "type": c.concept_type.value,
                        "source": c.source,
                    }
                    for c in self.state.predicted_next[:5]
                ]
            
            if self.state.anomaly_count > 0:
                result["anomaly_flags"] = ["anomaly_detected"]
            
            return result


# Singleton getter
def get_prefrontal_cortex() -> PrefrontalCortex:
    """Get the singleton PrefrontalCortex instance."""
    global _prefrontal_cortex
    if _prefrontal_cortex is None:
        with _pfc_lock:
            if _prefrontal_cortex is None:
                _prefrontal_cortex = PrefrontalCortex()
    return _prefrontal_cortex