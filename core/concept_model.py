"""Concept Model - High-level pattern recognition and abstraction layer.

Provides a third model layer for meta-reasoning, concept synthesis, and
cross-domain pattern recognition.

Architecture (Tri-Model):
┌──────────────────────────────────────────────────────────────────────────┐
│                     TRI-MODEL ARCHITECTURE                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   PRIMARY LLM   │  │   CONCEPT LLM   │  │   BACKEND LLM   │          │
│  │   (Kimi K2.5)   │  │  (GLM-5 fallback)│  │    (GLM-5)      │          │
│  │                 │  │                 │  │                 │          │
│  │  • Chat         │  │  • Meta-reason  │  │  • Code         │          │
│  │  • Creative     │  │  • Abstractions │  │  • Math         │          │
│  │  • User-facing  │  │  • Patterns     │  │  • Reasoning    │          │
│  │                 │  │  • Synthesis    │  │  • Analysis     │          │
│  └────────┬────────┘  └───────┬─────────┘  └───────┬─────────┘          │
│           │                   │                    │                     │
│           └───────────┬───────┴────────────────────┘                     │
│                       │                                                   │
│              ┌────────▼────────┐                                        │
│              │  CONCEPT EMBED  │                                        │
│              │  Jina v5 nano   │                                        │
│              │ 768-dim, multiL │                                        │
│              └────────┬────────┘                                        │
│                       │                                                   │
└──────────────────────────────────────────────────────────────────────────┘

Concept Model Role:
- Extract abstract patterns from specific instances
- Synthesize concepts across multiple domains
- Provide meta-reasoning: reasoning about reasoning
- Identify conceptual bridges between disparate ideas
- Semantic similarity via sentence embeddings

Concept Embedding Layer:
- jinaai/jina-embeddings-v5-text-nano (~239M params) via sentence-transformers
- 768-dim Matryoshka embeddings, 8192-token context, task-conditioned
  ("retrieval" by default; also text-matching / clustering / classification)
- Genuinely multilingual, not just language-agnostic: paraphrases across
  English/Spanish/French/German/Chinese/Japanese/Arabic/Russian/Hindi align at
  ~0.80-0.91 cosine, versus ~0.09 for unrelated text
- Fast similarity search for concept matching, clustering for concept grouping

NOTE: requires `peft` at runtime. Without it the model silently fails to load
and encode() degrades to zero vectors, which makes every pair perfectly similar
and quietly destroys semantic search, vector memory and KG similarity.
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from core.helpers import get_response_content, get_response_thinking

logger = logging.getLogger(__name__)

_concept_orchestrator: Optional["ConceptModelOrchestrator"] = None
_concept_lock = threading.Lock()

EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v5-text-nano"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 8192  # v5-text-nano supports 8K (v5-small/large supports 32K)
EMBEDDING_PROVIDER = "pytorch"
EMBEDDING_TASK = "retrieval"  # v5 requires task: retrieval, text-matching, clustering, or classification


class ConceptTaskType(Enum):
    """Types of tasks for the concept model."""
    ABSTRACTION = "abstraction"
    SYNTHESIS = "synthesis"
    PATTERN_RECOGNITION = "pattern_recognition"
    META_REASONING = "meta_reasoning"
    BRIDGE = "bridge"
    STRATEGIC = "strategic"


class ConceptMode(Enum):
    """Operating modes for concept processing."""
    STANDALONE = "standalone"
    SYNTHESIZE = "synthesize"
    GUIDE = "guide"
    VALIDATE = "validate"
    ENRICH = "enrich"


@dataclass
class ConceptConfig:
    """Configuration for the concept model."""
    model_id: str = "glm-5"
    embedding_model: str = EMBEDDING_MODEL_NAME
    embedding_dim: int = EMBEDDING_DIM
    embedding_provider: str = EMBEDDING_PROVIDER
    max_seq_length: int = MAX_SEQ_LENGTH
    capabilities: list[str] = field(default_factory=lambda: [
        "meta_reasoning",
        "abstraction",
        "pattern_recognition",
        "concept_synthesis",
        "cross_domain_bridging",
    ])
    preferred_tasks: list = field(default_factory=lambda: [
        ConceptTaskType.ABSTRACTION,
        ConceptTaskType.SYNTHESIS,
        ConceptTaskType.PATTERN_RECOGNITION,
        ConceptTaskType.META_REASONING,
    ])


class ConceptEmbedder:
    """Semantic embedding layer for concepts using Jina Embeddings v5.
    
    Provides:
    - 768-dim sentence embeddings (jina-embeddings-v5-text-nano)
    - 8192 token context length (v5-small/large supports 32K)
    - Semantic similarity computation
    - Concept clustering support
    - Task-specific LoRA adapters (retrieval, clustering, classification)
    
    Model: jinaai/jina-embeddings-v5-text-nano
    - 239M parameters (efficient for edge deployment)
    - Matryoshka representations (flexible dimensions)
    - Multilingual support
    - Apache 2.0 license
    
    For longer context (32K), use jina-embeddings-v5-base (677M params)
    """
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str | None = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        provider: str = EMBEDDING_PROVIDER,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.provider = provider
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._device = device
        self._model_loaded = False
        self._zero_vector_warned = False
        self._load_errors: list[str] = []

    def health(self) -> dict:
        """Report whether embeddings are actually usable.

        A failed model load degrades encode() to zero vectors, which makes every
        pair perfectly similar — semantic search, vector memory and the
        knowledge graph then return confident nonsense instead of failing. That
        state was previously invisible from outside the process; this is the
        signal the health manager and /api/status report on.

        Does not trigger a model load: callers poll this.
        """
        attempted = self._model is not None or bool(self._load_errors)
        degraded = attempted and self._model is None
        return {
            "healthy": not degraded,
            "status": "degraded" if degraded else ("ready" if self._model_loaded else "not_loaded"),
            "model": self.model_name,
            "provider": self.provider,
            "dimension": EMBEDDING_DIM,
            "returns_zero_vectors": degraded,
            "load_errors": list(self._load_errors),
            "message": (
                "Embedding model failed to load — encode() returns zero vectors, "
                "so semantic search, vector memory and knowledge-graph similarity "
                "are non-functional."
                if degraded
                else ("Embeddings ready" if self._model_loaded else "Model not loaded yet (lazy)")
            ),
        }

    def _get_device(self) -> str:
        """Determine the best available device."""
        if self._device:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            logger.debug("Torch device detection failed, using CPU")
        return "cpu"
    
    def _ensure_model(self):
        if self._model is not None:
            return
        
        load_errors = []
        
        try:
            if self.provider == "onnx":
                self._load_onnx_model()
            elif self.provider == "ov":
                self._load_openvino_model()
            else:
                self._load_pytorch_model()
            
            if self._model is not None:
                self._model_loaded = True
                logger.info(f"Concept embedder loaded: {self.model_name}")
                logger.info(f"  Provider: {self.provider}")
                logger.info(f"  Task: {EMBEDDING_TASK}")
                logger.info(f"  Max seq length: {self.max_seq_length} tokens")
                logger.info(f"  Embedding dim: {EMBEDDING_DIM} (Matryoshka)")
                logger.info("  Params: ~239M (v5-text-nano)")
                return
            else:
                load_errors.append(f"{self.provider} returned None")
        except Exception as e:
            load_errors.append(f"{self.provider}: {e}")
            logger.warning(f"Could not load embedding model with {self.provider}: {e}")
        
        if self.provider != "pytorch":
            logger.info("Falling back to PyTorch backend...")
            self.provider = "pytorch"
            try:
                self._load_pytorch_model()
                if self._model is not None:
                    self._model_loaded = True
                    logger.info(f"Concept embedder loaded with PyTorch fallback: {self.model_name}")
                    return
            except Exception as e:
                load_errors.append(f"pytorch fallback: {e}")
                logger.warning(f"PyTorch fallback also failed: {e}")
        
        logger.error(f"All embedding model loading attempts failed: {load_errors}")
        # Retained so health() can report *why* embeddings are degraded rather
        # than only that they are — the cause is usually a missing optional
        # dependency (e.g. 'peft' for jina-embeddings-v5).
        self._load_errors = [str(err) for err in load_errors]
        self._model = None
    
    def _load_pytorch_model(self):
        """Load model using sentence-transformers (PyTorch backend).
        
        Jina v5 may have issues with some GPU drivers, so we force CPU
        and rely on the efficient architecture for fast inference.
        """
        import os
        # Disable GPU for Jina v5 (AMD ROCm compatibility issues)
        os.environ['ROCM_VISIBLE_DEVICES'] = ''
        os.environ['HIP_VISIBLE_DEVICES'] = ''
        # Disable GPU - use CPU for stability
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HIP_VISIBLE_DEVICES'] = ''
        os.environ['ROCR_VISIBLE_DEVICES'] = ''
        
        from sentence_transformers import SentenceTransformer
        
        # Force CPU device for compatibility
        device = "cpu"
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            trust_remote_code=True,
        )
        self._model.max_seq_length = self.max_seq_length
        self._device = device
    
    def _load_onnx_model(self):
        """Load model using ONNX Runtime for fast inference.
        
        ONNX Runtime can leverage:
        - CPU with SIMD optimizations
        - CUDA execution provider
        - NPU via OpenVINO or DirectML
        """
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            import onnxruntime as ort
            
            available_providers = ort.get_available_providers()
            logger.debug(f"ONNX Runtime available providers: {available_providers}")
            
            provider = "CPUExecutionProvider"
            if "CUDAExecutionProvider" in available_providers:
                provider = "CUDAExecutionProvider"
            
            self._model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name,
                export=True,
                provider=provider,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"  ONNX provider: {provider}")
            
        except ImportError:
            logger.warning("optimum[onnxruntime] not available, falling back to PyTorch")
            self._load_pytorch_model()
    
    def _load_openvino_model(self):
        """Load model using OpenVINO for NPU acceleration.
        
        OpenVINO supports:
        - Intel NPU (Meteor Lake, Arrow Lake, Lunar Lake)
        - Intel GPU
        - Intel CPU with VNNI
        """
        try:
            from optimum.intel import OVModelForFeatureExtraction
            from transformers import AutoTokenizer
            
            self._model = OVModelForFeatureExtraction.from_pretrained(
                self.model_name,
                export=True,
                device="NPU",  # Will fall back to GPU/CPU if NPU unavailable
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("  OpenVINO device: NPU (or fallback)")
            
        except ImportError:
            logger.warning("optimum[intel] not available for NPU, falling back to ONNX")
            self._load_onnx_model()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        import torch
        
        if isinstance(model_output, dict):
            token_embeddings = model_output.get("last_hidden_state", model_output.get("token_embeddings"))
        else:
            token_embeddings = model_output[0]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def encode(self, texts: list[str] | str, max_length: int | None = None) -> Any:
        """Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            max_length: Override max sequence length (default: 32768)
            
        Returns:
            768-dim embedding(s)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        self._ensure_model()
        
        if self._model is None:
            # Zero vectors make every pair perfectly similar, so semantic search,
            # vector memory and the knowledge graph all silently return garbage
            # rather than failing. Report it once, loudly, instead of emitting a
            # warning on every single call.
            if not self._zero_vector_warned:
                self._zero_vector_warned = True
                logger.error(
                    "Embedding model failed to load — returning ZERO VECTORS. "
                    "Semantic search, vector memory and knowledge-graph similarity "
                    "are non-functional until this is fixed. A common cause is a "
                    "missing model dependency (e.g. 'peft' for jina-embeddings-v5); "
                    "check the load errors logged above."
                )
            result = np.zeros((len(texts), EMBEDDING_DIM))
            # Return 1D array for single input, 2D for batch
            if single_input:
                return result[0]
            return result
        
        max_len = max_length or self.max_seq_length
        
        with self._lock:
            results: list[Any] = [None] * len(texts)
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if uncached_texts:
                embeddings = self._encode_batch(uncached_texts, max_len)
                
                for i, idx in enumerate(uncached_indices):
                    cache_key = hashlib.sha256(uncached_texts[i].encode()).hexdigest()[:16]
                    self._cache[cache_key] = embeddings[i]
                    results[idx] = embeddings[i]
                
                if len(self._cache) > 2000:
                    keys = list(self._cache.keys())
                    for k in keys[:500]:
                        del self._cache[k]
            
            if single_input:
                return results[0]
            return np.array(results)
    
    def _encode_batch(self, texts: list[str], max_length: int) -> np.ndarray:
        """Encode a batch of texts."""
        import torch
        import torch.nn.functional as F
        
        if self.provider in ["onnx", "ov"] and self._tokenizer is not None:
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                outputs = self._model(**encoded)
            
            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.numpy()
        
        else:
            # Jina v5 requires task parameter
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=min(8, len(texts)),
                task=EMBEDDING_TASK,
            )
            return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        emb1 = np.asarray(self.encode(text1)).flatten()
        emb2 = np.asarray(self.encode(text2)).flatten()
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
    
    def find_similar(self, query: str, candidates: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        if not candidates:
            return []
        
        query_emb = np.asarray(self.encode(query)).flatten()
        candidate_embs = np.asarray(self.encode(candidates))
        if candidate_embs.ndim == 2:
            candidate_embs = candidate_embs.reshape(len(candidates), -1)
        
        similarities = np.dot(candidate_embs, query_emb) / (
            np.linalg.norm(candidate_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(candidates[i], float(similarities[i])) for i in top_indices]
    
    def cluster_concepts(self, texts: list[str], n_clusters: int = 5) -> dict[int, list[str]]:
        if len(texts) < n_clusters:
            return {i: [texts[i]] for i in range(len(texts))}
        
        self._ensure_model()
        if self._model is None:
            return {0: texts}
        
        from sklearn.cluster import KMeans
        embeddings = self.encode(texts)
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        clusters: dict[int, list[str]] = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])
        
        return clusters


@dataclass
class ConceptRequest:
    """A request to the concept model."""
    id: str
    task_type: ConceptTaskType
    prompt: str
    context: dict = field(default_factory=dict)
    input_from_primary: Optional[str] = None
    input_from_backend: Optional[str] = None
    mode: ConceptMode = ConceptMode.SYNTHESIZE
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[str] = None
    abstraction: Optional[str] = None
    patterns_found: list[str] = field(default_factory=list)


@dataclass
class ConceptStats:
    """Statistics for concept model operations."""
    total_requests: int = 0
    abstractions: int = 0
    syntheses: int = 0
    pattern_discoveries: int = 0
    meta_reasoning_calls: int = 0
    collaborations_with_primary: int = 0
    collaborations_with_backend: int = 0
    tri_model_collaborations: int = 0
    avg_latency_ms: float = 0.0
    errors: int = 0


CONCEPT_CONFIG = ConceptConfig()


class ConceptModelOrchestrator:
    """Orchestrates concept-level thinking across the tri-model architecture.
    
    The concept model operates at a higher level of abstraction:
    - Pattern recognition across domains
    - Conceptual synthesis from multiple inputs
    - Meta-reasoning about the reasoning process itself
    - Strategic guidance for other models
    - Semantic similarity via sentence embeddings
    """
    
    def __init__(
        self,
        concept_llm=None,
        primary_llm=None,
        backend_llm=None,
        model_coordinator=None,
    ):
        self.concept_llm = concept_llm
        self.primary_llm = primary_llm
        self.backend_llm = backend_llm or concept_llm
        self.model_coordinator = model_coordinator
        
        self.config = CONCEPT_CONFIG
        self.embedder = ConceptEmbedder(self.config.embedding_model)
        
        self._stats = ConceptStats()
        self._request_history: deque[ConceptRequest] = deque(maxlen=100)
        self._concept_cache: OrderedDict[str, str] = OrderedDict()
        self._pattern_library: OrderedDict[str, list[str]] = OrderedDict()
        self._concept_embeddings: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self._max_embeddings = 2000    # More concepts for rich understanding
        self._max_cache = 1000         # Larger concept cache
        self._max_patterns = 1000       # More pattern recognition
        
        self._on_concept_callback: Optional[Callable] = None
        
        self._running = False
    
    def _store_embedding(self, key: str, embedding: np.ndarray) -> None:
        """Store an embedding with LRU eviction and thread safety."""
        with self._lock:
            self._concept_embeddings[key] = embedding
            self._concept_embeddings.move_to_end(key)
            while len(self._concept_embeddings) > self._max_embeddings:
                self._concept_embeddings.popitem(last=False)
    
    def _store_cache(self, key: str, value: str) -> None:
        """Store a cache entry with LRU eviction and thread safety."""
        with self._lock:
            self._concept_cache[key] = value
            self._concept_cache.move_to_end(key)
            while len(self._concept_cache) > self._max_cache:
                self._concept_cache.popitem(last=False)
    
    async def start(self):
        """Start the concept model orchestrator."""
        self._running = True
        logger.info("Concept Model Orchestrator started")
        logger.info(f"  LLM: {self.config.model_id}")
        logger.info(f"  Embedding: {self.config.embedding_model} ({self.config.embedding_dim}d)")
        logger.info(f"  Capabilities: {', '.join(self.config.capabilities)}")
        logger.info(f"  Tasks: {', '.join(t.value for t in self.config.preferred_tasks)}")
    
    async def stop(self):
        """Stop the concept model orchestrator."""
        self._running = False
        logger.info("Concept Model Orchestrator stopped")
    
    def set_callbacks(self, on_concept: Optional[Callable] = None):
        """Set callbacks for concept model events."""
        self._on_concept_callback = on_concept
    
    async def extract_abstract_pattern(
        self,
        instances: list[str],
        domain: str = "general",
    ) -> str:
        """Extract an abstract pattern from specific instances.
        
        Uses embeddings to:
        1. Find semantic similarities between instances
        2. Cluster related instances
        3. Guide the LLM with embedding-informed context
        
        Args:
            instances: List of specific instances or examples
            domain: The domain these instances come from
            
        Returns:
            An abstract pattern that captures the essence
        """
        embeddings_info = ""
        similarity_info = ""
        
        try:
            if len(instances) > 1:
                embeddings = self.embedder.encode(instances)
                
                embeddings_info = f"\n\n[Embedding Analysis]\n- Instance embeddings: {embeddings.shape[0]} x {embeddings.shape[1]} dimensions"
                
                similarities = []
                for i in range(len(instances)):
                    for j in range(i + 1, len(instances)):
                        sim = float(np.dot(embeddings[i], embeddings[j]) / 
                                   (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
                        similarities.append((i, j, sim))
                
                if similarities:
                    avg_sim = sum(s[2] for s in similarities) / len(similarities)
                    similarity_info = f"\n- Average semantic similarity: {avg_sim:.3f}"
                    
                    clustering = self.embedder.cluster_concepts(instances, n_clusters=min(3, len(instances)))
                    clustering_info = f"\n- Semantic clusters: {len(clustering)} groups identified"
                    embeddings_info += similarity_info + clustering_info
        except Exception as e:
            logger.debug(f"Embedding analysis failed: {e}")
        
        request = ConceptRequest(
            id=f"abstract_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.ABSTRACTION,
            prompt=f"Extract the abstract pattern from these {domain} instances:\n\n" +
                   "\n".join(f"- {i}" for i in instances) +
                   embeddings_info + "\n\nIdentify the underlying pattern that connects all instances.",
            mode=ConceptMode.STANDALONE,
        )
        
        with self._lock:
            self._stats.abstractions += 1
        
        result = await self._execute_concept(request)
        
        for inst in instances:
            key = f"instance_{hash(inst) % 10000}"
            self._store_embedding(key, self.embedder.encode(inst))
        
        return result
    
    async def synthesize_concepts(
        self,
        concepts: list[str],
        context: str = "",
    ) -> str:
        """Synthesize multiple concepts into a unified understanding.
        
        Uses embeddings to:
        1. Measure semantic relationships between concepts
        2. Identify concept clusters
        3. Guide synthesis with embedding-informed context
        
        Args:
            concepts: List of concepts from different domains
            context: Additional context for synthesis
            
        Returns:
            A synthesized concept that integrates all inputs
        """
        embeddings_info = ""
        
        try:
            if len(concepts) > 1:
                embeddings = self.embedder.encode(concepts)
                
                for i, concept in enumerate(concepts):
                    key = f"concept_{hash(concept) % 10000}"
                    self._store_embedding(key, embeddings[i])
                
                similarity_matrix = []
                for i, item_concepts in enumerate(concepts):
                    for j in range(i + 1, len(concepts)):
                        sim = float(np.dot(embeddings[i], embeddings[j]) / 
                                   (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
                        similarity_matrix.append((item_concepts[:30], concepts[j][:30], sim))
                
                embeddings_info = "\n\n[Semantic Analysis]\n"
                for c1, c2, sim in similarity_matrix:
                    embeddings_info += f"- '{c1}...' ↔ '{c2}...': similarity={sim:.3f}\n"
                
                if len(concepts) >= 3:
                    clusters = self.embedder.cluster_concepts(concepts, n_clusters=min(3, len(concepts)))
                    embeddings_info += f"\nConcept clusters: {len(clusters)} semantic groups identified"
        except Exception as e:
            logger.debug(f"Embedding synthesis failed: {e}")
        
        prompt = f"""Synthesize these concepts into a unified understanding:

{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(concepts))}

{f'Context: {context}' if context else ''}
{embeddings_info}
Provide:
1. The core unified concept
2. How each input concept contributes
3. Novel insights from the synthesis
4. Practical applications"""

        request = ConceptRequest(
            id=f"synthesize_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.SYNTHESIS,
            prompt=prompt,
            mode=ConceptMode.SYNTHESIZE,
        )
        
        with self._lock:
            self._stats.syntheses += 1
        
        return await self._execute_concept(request)
    
    async def find_patterns(
        self,
        examples: list[dict],
        pattern_type: str = "structural",
    ) -> list[str]:
        """Find patterns across multiple examples using embedding clustering.
        
        Uses embeddings to:
        1. Cluster semantically similar examples
        2. Identify which examples are most representative
        3. Guide LLM pattern analysis with embedding insights
        4. Store pattern embeddings for future retrieval
        """
        import json
        
        embeddings_info = ""
        example_texts = []
        
        try:
            for ex in examples:
                text = json.dumps(ex, sort_keys=True)
                example_texts.append(text[:500])
            
            if len(example_texts) > 1:
                embeddings = self.embedder.encode(example_texts)
                
                for i, text in enumerate(example_texts):
                    key = f"example_{pattern_type}_{hash(text) % 10000}"
                    self._store_embedding(key, embeddings[i])
                
                clusters = self.embedder.cluster_concepts(example_texts, n_clusters=min(5, len(example_texts)))
                
                embeddings_info = "\n\n[Embedding Pattern Analysis]\n"
                embeddings_info += f"- Analyzed {len(example_texts)} examples with {embeddings.shape[1]}-dim embeddings\n"
                embeddings_info += f"- Identified {len(clusters)} semantic clusters\n"
                
                for cluster_id, cluster_items in clusters.items():
                    embeddings_info += f"\nCluster {cluster_id} ({len(cluster_items)} items):\n"
                    for item in cluster_items[:3]:
                        embeddings_info += f"  - {item[:80]}...\n"
            else:
                key = f"example_{pattern_type}_single"
                self._store_embedding(key, self.embedder.encode(example_texts[0]))
        except Exception as e:
            logger.debug(f"Embedding pattern analysis failed: {e}")
        
        prompt = f"""Analyze these examples to find {pattern_type} patterns:

{json.dumps(examples, indent=2)[:4000]}
{embeddings_info}
Identify:
1. Recurring patterns
2. Similarities across examples
3. Underlying principles
4. Potential generalizations

List each pattern found."""

        request = ConceptRequest(
            id=f"pattern_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.PATTERN_RECOGNITION,
            prompt=prompt,
            mode=ConceptMode.STANDALONE,
        )
        
        with self._lock:
            self._stats.pattern_discoveries += 1
        
        result = await self._execute_concept(request)
        patterns = self._extract_patterns(result)
        
        for pattern in patterns:
            key = f"pattern_{pattern_type}_{hash(pattern) % 10000}"
            self._store_embedding(key, self.embedder.encode(pattern))
        
        with self._lock:
            key = f"{pattern_type}_{len(self._pattern_library)}"
            self._pattern_library[key] = patterns
            self._pattern_library.move_to_end(key)
            while len(self._pattern_library) > self._max_patterns:
                self._pattern_library.popitem(last=False)
        
        return patterns
    
    async def meta_reason(
        self,
        reasoning_process: str,
        question: str = "",
    ) -> str:
        """Perform meta-reasoning about a reasoning process using embedding analysis.
        
        Uses embeddings to:
        1. Store the reasoning process for future comparison
        2. Find similar past reasoning for context
        3. Identify patterns in reasoning approaches
        """
        embeddings_info = ""
        
        try:
            reasoning_emb = self.embedder.encode(reasoning_process)
            key = f"reasoning_{hash(reasoning_process) % 10000}"
            self._store_embedding(key, reasoning_emb)
            
            similar_keys = []
            reason_embeddings = [(k, v) for k, v in self._concept_embeddings.items() if k.startswith("reasoning_") and k != key]
            
            if reason_embeddings:
                for rk, rv in reason_embeddings[:5]:
                    sim = float(np.dot(reasoning_emb, rv) / (np.linalg.norm(reasoning_emb) * np.linalg.norm(rv) + 1e-8))
                    if sim > 0.7:
                        similar_keys.append((rk, sim))
                
                if similar_keys:
                    embeddings_info = f"\n\n[Semantic Context]\n- Found {len(similar_keys)} similar past reasoning processes\n- Highest similarity: {max(s for _, s in similar_keys):.3f}"
        except Exception as e:
            logger.debug(f"Embedding meta-reasoning failed: {e}")
        
        prompt = f"""Analyze this reasoning process at a meta-level:

Reasoning Process:
{reasoning_process}
{embeddings_info}
{f'Specific Question: {question}' if question else ''}

Provide meta-level analysis:
1. What reasoning strategies were used?
2. Were there any cognitive biases or shortcuts?
3. How could the reasoning be improved?
4. What assumptions underlie the reasoning?
5. What alternative approaches could work?"""

        request = ConceptRequest(
            id=f"meta_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.META_REASONING,
            prompt=prompt,
            mode=ConceptMode.STANDALONE,
        )
        
        with self._lock:
            self._stats.meta_reasoning_calls += 1
        
        result = await self._execute_concept(request)
        
        result_key = f"meta_result_{hash(result) % 10000}"
        self._store_embedding(result_key, self.embedder.encode(result))
        
        return result
    
    async def bridge_concepts(
        self,
        concept_a: str,
        concept_b: str,
        context: str = "",
    ) -> str:
        """Find conceptual bridges between two concepts using embedding similarity.
        
        Uses embeddings to:
        1. Compute semantic similarity between concepts
        2. Find intermediate concepts that bridge the gap
        3. Store bridge relationships for future retrieval
        """
        embeddings_info = ""
        
        try:
            emb_a = self.embedder.encode(concept_a)
            emb_b = self.embedder.encode(concept_b)
            
            key_a = f"concept_{hash(concept_a) % 10000}"
            key_b = f"concept_{hash(concept_b) % 10000}"
            self._store_embedding(key_a, emb_a)
            self._store_embedding(key_b, emb_b)
            
            similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
            
            bridge_candidates = []
            for k, v in self._concept_embeddings.items():
                if k.startswith("concept_") and k != key_a and k != key_b:
                    sim_a = float(np.dot(emb_a, v) / (np.linalg.norm(emb_a) * np.linalg.norm(v) + 1e-8))
                    sim_b = float(np.dot(emb_b, v) / (np.linalg.norm(emb_b) * np.linalg.norm(v) + 1e-8))
                    if sim_a > 0.5 and sim_b > 0.5:
                        bridge_candidates.append((k, (sim_a + sim_b) / 2))
            
            embeddings_info = "\n\n[Semantic Bridge Analysis]\n"
            embeddings_info += f"- Direct similarity: {similarity:.3f}\n"
            
            if bridge_candidates:
                bridge_candidates.sort(key=lambda x: x[1], reverse=True)
                embeddings_info += f"- Found {len(bridge_candidates)} potential bridge concepts\n"
                embeddings_info += f"- Best bridge similarity: {bridge_candidates[0][1]:.3f}"
        except Exception as e:
            logger.debug(f"Embedding bridge analysis failed: {e}")
        
        prompt = f"""Find conceptual bridges between these two concepts:

Concept A: {concept_a}

Concept B: {concept_b}
{embeddings_info}
{f'Context: {context}' if context else ''}

Analyze:
1. What do these concepts share?
2. What are the key differences?
3. How can they be unified conceptually?
4. What metaphors connect them?
5. What novel synthesis emerges from combining them?"""

        request = ConceptRequest(
            id=f"bridge_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.BRIDGE,
            prompt=prompt,
            mode=ConceptMode.SYNTHESIZE,
        )
        
        result = await self._execute_concept(request)
        
        bridge_key = f"bridge_{hash(concept_a + concept_b) % 10000}"
        self._store_embedding(bridge_key, self.embedder.encode(result))
        
        return result
    
    async def tri_model_think(
        self,
        topic: str,
        primary_input: str = "",
        backend_input: str = "",
        rounds: int = 2,
    ) -> dict:
        """Collaborative thinking across all three models with embedding interconnect.
        
        Uses embeddings to:
        1. Store all model outputs for cross-model retrieval
        2. Compute semantic alignment between models
        3. Find relevant past tri-model sessions
        """
        accumulated = {
            "primary": primary_input,
            "backend": backend_input,
            "concept_contributions": [],
            "embedding_interconnect": {},
        }
        
        try:
            emb_primary = None
            emb_backend = None
            if primary_input:
                emb_primary = self.embedder.encode(primary_input)
                self._store_embedding("tri_primary_last", emb_primary)
                accumulated["embedding_interconnect"]["primary_dim"] = emb_primary.shape[0]
            
            if backend_input:
                emb_backend = self.embedder.encode(backend_input)
                self._store_embedding("tri_backend_last", emb_backend)
                accumulated["embedding_interconnect"]["backend_dim"] = emb_backend.shape[0]
            
            if emb_primary is not None and emb_backend is not None:
                emb_p = np.asarray(emb_primary).flatten()
                emb_b = np.asarray(emb_backend).flatten()
                cross_sim = float(np.dot(emb_p, emb_b) / 
                                 (np.linalg.norm(emb_p) * np.linalg.norm(emb_b) + 1e-8))
                accumulated["embedding_interconnect"]["primary_backend_similarity"] = round(cross_sim, 3)
            
            topic_emb = self.embedder.encode(topic)
            self._store_embedding("tri_topic_last", topic_emb)
            accumulated["embedding_interconnect"]["topic_stored"] = True
        except Exception as e:
            logger.debug(f"Tri-model embedding interconnect failed: {e}")
        
        current_prompt = f"""Topic: {topic}

{'Primary model perspective: ' + primary_input if primary_input else ''}
{'Backend model perspective: ' + backend_input if backend_input else ''}

As the concept model, provide high-level analysis:
1. What abstract patterns do you see?
2. What conceptual synthesis emerges?
3. What meta-insights apply?
4. What should the other models focus on next?"""
        
        for round_num in range(rounds):
            if not self._running:
                break
            
            concept_result = await self._call_concept(current_prompt)
            accumulated["concept_contributions"].append({
                "round": round_num + 1,
                "insight": concept_result[:500],
            })
            
            try:
                concept_emb = self.embedder.encode(concept_result)
                self._store_embedding(f"tri_concept_r{round_num + 1}", concept_emb)
            except Exception as e:
                logger.debug(f"Failed to store tri-model embedding for round {round_num + 1}: {e}")
            
            if round_num < rounds - 1:
                current_prompt = f"""Building on previous analysis:

{concept_result[:1000]}

Consider additional perspectives and deeper conceptual analysis."""
        
        with self._lock:
            self._stats.tri_model_collaborations += 1
        
        return accumulated
    
    async def guide_primary(
        self,
        user_query: str,
        conversation_context: str = "",
    ) -> str:
        """Provide conceptual guidance for primary model using embedding analysis.
        
        Uses embeddings to:
        1. Analyze query semantics
        2. Find similar past queries for context
        3. Store guidance for future retrieval
        """
        embeddings_info = ""
        
        try:
            query_emb = self.embedder.encode(user_query)
            self._store_embedding("guide_query_last", query_emb)
            
            similar_past = []
            for k, v in self._concept_embeddings.items():
                if k.startswith("guide_result_"):
                    sim = float(np.dot(query_emb, v) / (np.linalg.norm(query_emb) * np.linalg.norm(v) + 1e-8))
                    if sim > 0.75:
                        similar_past.append((k, sim))
            
            if similar_past:
                embeddings_info = f"\n\n[Semantic Context]\n- Found {len(similar_past)} similar past queries"
                embeddings_info += f"\n- Highest similarity: {max(s for _, s in similar_past):.3f}"
            
            if conversation_context:
                ctx_emb = self.embedder.encode(conversation_context[:1000])
                ctx_sim = float(np.dot(query_emb, ctx_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ctx_emb) + 1e-8))
                embeddings_info += f"\n- Query-context relevance: {ctx_sim:.3f}"
        except Exception as e:
            logger.debug(f"Embedding guidance analysis failed: {e}")
        
        prompt = f"""Analyze this user query at a conceptual level:

User Query: {user_query}
{embeddings_info}
{f'Recent conversation: {conversation_context[:1000]}' if conversation_context else ''}

Provide:
1. Key concepts involved
2. Potential ambiguities
3. Relevant analogies or metaphors
4. Suggested angles of approach
5. What deeper questions might be at play

Be concise and conceptual, not detailed."""

        request = ConceptRequest(
            id=f"guide_primary_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.STRATEGIC,
            prompt=prompt,
            mode=ConceptMode.GUIDE,
        )
        
        with self._lock:
            self._stats.collaborations_with_primary += 1
        
        result = await self._execute_concept(request)
        
        result_key = f"guide_result_{hash(result) % 10000}"
        self._store_embedding(result_key, self.embedder.encode(result))
        
        return result
    
    async def validate_output(
        self,
        output: str,
        criteria: list[str],
        output_type: str = "response",
    ) -> dict:
        """Validate output against conceptual criteria.
        
        Args:
            output: The output to validate
            criteria: List of conceptual criteria
            output_type: Type of output (response, code, analysis, etc.)
            
        Returns:
            Validation result with scores and suggestions
        """
        prompt = f"""Validate this {output_type} against conceptual criteria:

Output:
{output[:2000]}

Criteria:
{chr(10).join(f'- {c}' for c in criteria)}

Provide:
1. Score (0-10) for each criterion
2. Overall conceptual soundness
3. Specific issues identified
4. Suggestions for improvement
5. What's missing from a conceptual standpoint?"""

        request = ConceptRequest(
            id=f"validate_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.META_REASONING,
            prompt=prompt,
            mode=ConceptMode.VALIDATE,
        )
        
        result = await self._execute_concept(request)
        
        return {
            "validation": result,
            "scores": self._extract_scores(result),
            "suggestions": self._extract_suggestions(result),
        }
    
    async def enrich_insights(
        self,
        insights: list[str],
        domain: str = "",
    ) -> list[str]:
        """Enrich insights with abstract patterns and connections using embedding analysis.
        
        Uses embeddings to:
        1. Cluster similar insights
        2. Find related patterns in memory
        3. Store enriched insights for cross-model retrieval
        """
        embeddings_info = ""
        
        try:
            if len(insights) > 1:
                insight_embs = self.embedder.encode(insights)
                
                for i, insight in enumerate(insights):
                    key = f"insight_{domain}_{hash(insight) % 10000}"
                    self._store_embedding(key, insight_embs[i])
                
                clusters = self.embedder.cluster_concepts(insights, n_clusters=min(3, len(insights)))
                embeddings_info = "\n\n[Insight Embedding Analysis]\n"
                embeddings_info += f"- {len(insights)} insights analyzed\n"
                embeddings_info += f"- {len(clusters)} semantic clusters identified\n"
                
                related_patterns = []
                for k, v in self._concept_embeddings.items():
                    if k.startswith("pattern_"):
                        for i, insight in enumerate(insights):
                            sim = float(np.dot(insight_embs[i], v) / 
                                       (np.linalg.norm(insight_embs[i]) * np.linalg.norm(v) + 1e-8))
                            if sim > 0.7:
                                related_patterns.append((k, sim))
                
                if related_patterns:
                    embeddings_info += f"- {len(related_patterns)} related patterns found in memory"
            else:
                key = f"insight_{domain}_single"
                self._store_embedding(key, self.embedder.encode(insights[0]))
        except Exception as e:
            logger.debug(f"Embedding insight enrichment failed: {e}")
        
        prompt = f"""Enrich these insights with conceptual depth:

{chr(10).join(f'{i+1}. {insight}' for i, insight in enumerate(insights))}
{embeddings_info}
{f'Domain: {domain}' if domain else ''}

For each insight, provide:
1. The underlying abstract pattern
2. Cross-domain connections
3. Broader implications
4. A meta-level enhancement

Format as an enriched list."""

        request = ConceptRequest(
            id=f"enrich_{int(time.time() * 1000)}",
            task_type=ConceptTaskType.SYNTHESIS,
            prompt=prompt,
            mode=ConceptMode.ENRICH,
        )
        
        result = await self._execute_concept(request)
        enriched = self._parse_enriched_insights(result)
        
        for insight in enriched:
            key = f"enriched_{domain}_{hash(insight) % 10000}"
            self._store_embedding(key, self.embedder.encode(insight))
        
        with self._lock:
            self._stats.collaborations_with_backend += 1
        
        return enriched
    
    async def _execute_concept(self, request: ConceptRequest) -> str:
        """Execute a concept request."""
        with self._lock:
            self._stats.total_requests += 1
        
        start_time = time.time()
        
        try:
            result = await self._call_concept(request.prompt)
            request.result = result
            
            latency_ms = (time.time() - start_time) * 1000
            with self._lock:
                self._stats.avg_latency_ms = (
                    self._stats.avg_latency_ms * 0.9 + latency_ms * 0.1
                )
                self._request_history.append(request)
            
            if self._on_concept_callback:
                try:
                    self._on_concept_callback(request)
                except Exception as e:
                    logger.debug(f"Concept callback failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Concept model request failed: {e}")
            with self._lock:
                self._stats.errors += 1
            raise
    
    async def _call_concept(self, prompt: str) -> str:
        """Call the concept model (uses backend_llm as fallback)."""
        llm = self.concept_llm or self.backend_llm
        if not llm:
            raise RuntimeError("No LLM configured for concept model")
        
        try:
            response = await llm.chat(
                user_input=prompt,
                system_prompt="""You are a conceptual thinking specialist. Your role is to:
1. Identify abstract patterns across domains
2. Synthesize concepts into unified frameworks
3. Provide meta-level reasoning about processes
4. Find bridges between seemingly unrelated ideas
5. Think at a higher level of abstraction

Be insightful, connect disparate concepts, and provide strategic guidance.
Focus on patterns, principles, and conceptual frameworks.""",
                conversation_history=[],
            )
            
            if isinstance(response, dict):
                content = get_response_content(response)
                thinking = get_response_thinking(response)
                if thinking:
                    return f"<thinking>\n{thinking}\n</thinking>\n\n{content}"
                return content
            return str(response)
            
        except Exception as e:
            logger.error(f"Concept model call failed: {e}")
            raise
    
    def _extract_patterns(self, text: str) -> list[str]:
        """Extract pattern descriptions from text."""
        patterns = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("*") or 
                        (line[0].isdigit() and "." in line[:3])):
                pattern = line.lstrip("-*0123456789. ").strip()
                if len(pattern) > 10:
                    patterns.append(pattern)
        return patterns[:10]
    
    def _extract_scores(self, text: str) -> dict:
        """Extract scores from validation text."""
        import re
        scores = {}
        score_matches = re.findall(r"(\w+[^:]*):\s*(\d+(?:\.\d+)?)[/10\s]*", text)
        for name, score in score_matches:
            try:
                scores[name.strip().lower()] = float(score)
            except ValueError:
                pass
        return scores
    
    def _extract_suggestions(self, text: str) -> list[str]:
        """Extract suggestions from validation text."""
        suggestions = []
        lines = text.split("\n")
        in_suggestions = False
        for line in lines:
            if "suggestion" in line.lower():
                in_suggestions = True
            if in_suggestions:
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("*")):
                    suggestions.append(line.lstrip("-* ").strip())
        return suggestions[:5]
    
    def _parse_enriched_insights(self, text: str) -> list[str]:
        """Parse enriched insights from text."""
        insights = []
        current_insight = []
        
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("*") or 
                        (line[0].isdigit() and "." in line[:3])):
                if current_insight:
                    insights.append("\n".join(current_insight))
                current_insight = [line.lstrip("-*0123456789. ").strip()]
            elif line and current_insight:
                current_insight.append(line)
        
        if current_insight:
            insights.append("\n".join(current_insight))
        
        return insights if insights else [text]
    
    def get_stats(self) -> dict:
        """Get concept model statistics."""
        with self._lock:
            return {
                "total_requests": self._stats.total_requests,
                "abstractions": self._stats.abstractions,
                "syntheses": self._stats.syntheses,
                "pattern_discoveries": self._stats.pattern_discoveries,
                "meta_reasoning_calls": self._stats.meta_reasoning_calls,
                "collaborations_with_primary": self._stats.collaborations_with_primary,
                "collaborations_with_backend": self._stats.collaborations_with_backend,
                "tri_model_collaborations": self._stats.tri_model_collaborations,
                "avg_latency_ms": round(self._stats.avg_latency_ms, 2),
                "errors": self._stats.errors,
                "concept_model_connected": self.concept_llm is not None,
                "pattern_library_size": len(self._pattern_library),
                "cache_size": len(self._concept_cache),
            }


def get_concept_orchestrator(
    concept_llm=None,
    primary_llm=None,
    backend_llm=None,
    model_coordinator=None,
) -> ConceptModelOrchestrator:
    """Get or create the global ConceptModelOrchestrator."""
    global _concept_orchestrator, _concept_lock
    
    with _concept_lock:
        if _concept_orchestrator is None:
            _concept_orchestrator = ConceptModelOrchestrator(
                concept_llm=concept_llm,
                primary_llm=primary_llm,
                backend_llm=backend_llm,
                model_coordinator=model_coordinator,
            )
        else:
            if concept_llm is not None:
                _concept_orchestrator.concept_llm = concept_llm
            if primary_llm is not None:
                _concept_orchestrator.primary_llm = primary_llm
            if backend_llm is not None:
                _concept_orchestrator.backend_llm = backend_llm
            if model_coordinator is not None:
                _concept_orchestrator.model_coordinator = model_coordinator
    
    return _concept_orchestrator


def init_concept_model(
    concept_llm,
    primary_llm=None,
    backend_llm=None,
    model_coordinator=None,
) -> ConceptModelOrchestrator:
    """Initialize the concept model system."""
    orchestrator = get_concept_orchestrator(
        concept_llm=concept_llm,
        primary_llm=primary_llm,
        backend_llm=backend_llm,
        model_coordinator=model_coordinator,
    )
    return orchestrator