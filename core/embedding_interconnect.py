"""
Embedding Interconnect - Unified embedding space for all systems.

This module provides a SINGLE embedding instance that ALL systems use:
- Dual-model memory sharing
- Vector memory semantic search  
- Knowledge graph node embeddings
- Insight bridge knowledge storage
- Unified memory semantic queries
- Intelligent orchestrator workflow learning

Architecture for parallel agents:
- Uses EmbeddingClient which connects to shared EmbeddingServer via IPC
- Single model instance shared across all workers
- Falls back to local model if server unavailable

The key insight: ONE embedding model shared across ALL of Jarvis ensures:
1. Kimi and GLM-5 see the same semantic space
2. Vector memory queries work with dual-model embeddings
3. Knowledge graph nodes are embedded consistently
4. Insights are stored in the same space as model outputs
5. Workflows can be retrieved based on semantic similarity
"""

from typing import Optional
import threading
import logging
import os
from core.helpers import atomic_write_json, read_json_with_integrity
import numpy as np
from collections import deque
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Singleton pattern
_interconnect: Optional["EmbeddingInterconnect"] = None
_lock = threading.Lock()

# Persistence path
EMBEDDING_CACHE_PATH = Path.home() / ".jarvis" / "embedding_cache"

# Environment variable to disable IPC (useful for tests)
USE_LOCAL_EMBEDDER = os.environ.get("JARVIS_LOCAL_EMBEDDER", "0") == "1"


class EmbeddingInterconnect:
    """
    Central embedding hub for all of Jarvis.
    
    All systems use this ONE embedder:
    
    1. DUAL_MODEL: Stores prompts/responses for cross-model sharing
    2. VECTOR_MEMORY: Uses this for semantic search
    3. KNOWLEDGE_GRAPH: Embeds nodes for semantic queries
    4. INSIGHT_BRIDGE: Stores insights in embedding space
    5. UNIFIED_MEMORY: Semantic queries across all memory types
    
    This ensures when Kimi talks about "neural networks", 
    GLM-5's "analyze neural networks" query finds it.
    
    For parallel agent deployment, uses IPC client to share single model:
    - Connects to EmbeddingServer via Unix socket
    - Falls back to local model if server unavailable
    - Set JARVIS_LOCAL_EMBEDDER=1 to force local model (for tests)
    """
    
    def __init__(self, prefer_server: bool = True):
        """
        Initialize embedding interconnect.
        
        Args:
            prefer_server: If True, use IPC server when available (default)
        """
        if USE_LOCAL_EMBEDDER or not prefer_server:
            from core.concept_model import ConceptEmbedder
            self.embedder = ConceptEmbedder()
            self._using_ipc = False
        else:
            from core.embedding_client import get_embedding_client
            self.embedder = get_embedding_client(prefer_server=True)
            self._using_ipc = True
        
        self._lock = threading.Lock()
        
        # Cross-model knowledge store
        self._model_outputs: dict[str, list[tuple[str, float, "np.ndarray"]]] = {
            "primary": [],   # Kimi's knowledge (prompt, quality, embedding)
            "backend": [],    # GLM-5's knowledge (prompt, quality, embedding)
            "synthesis": [],  # Combined knowledge
        }
        
        # Cache for cross-model context to avoid re-encoding
        self._model_output_embeddings: dict[str, dict[str, "np.ndarray"]] = {
            "primary": {},   # prompt_hash -> embedding
            "backend": {},
            "synthesis": {},
        }
        
        # Lifetime counters (never reset)
        self._total_embeddings = 0
        self._embedding_rate_per_min = 0
        self._last_embedding_time = None
        
        # Last responses for quick context retrieval
        self._last_responses: dict[str, str] = {
            "primary": "",
            "backend": "",
        }
        
        # Conversation context (shared across models)
        self._conversation_embeddings: list[tuple[str, "np.ndarray"]] = []
        self._conversation_texts: list[str] = []
        
        # Memory integration - bounded dicts with LRU eviction
        self._memory_embeddings: dict[str, "np.ndarray"] = {}
        self._memory_order: deque[str] = deque()
        self._memory_order_set: set[str] = set()
        self._insight_embeddings: dict[str, "np.ndarray"] = {}
        self._insight_order: deque[str] = deque()
        self._insight_order_set: set[str] = set()
        self._knowledge_embeddings: dict[str, "np.ndarray"] = {}
        self._knowledge_order: deque[str] = deque()
        self._knowledge_order_set: set[str] = set()
        self._MAX_EMBEDDINGS_PER_TYPE = 5000  # Limit per type (~15MB each)
        
        # Workflow embeddings (for intelligent orchestrator) - bounded
        self._workflow_embeddings: dict[str, "np.ndarray"] = {}
        self._workflow_order: deque[str] = deque()
        self._workflow_order_set: set[str] = set()
        self._workflow_metadata: dict[str, dict] = {}
        self._MAX_WORKFLOW_EMBEDDINGS = 1000  # ~3MB
        
        # Clusters for faster retrieval
        self._clusters: dict[str, list[str]] = {}
        self._cluster_centroids: dict[str, "np.ndarray"] = {}
        self._cluster_order: deque[str] = deque()
        self._cluster_order_set: set[str] = set()
        self.MAX_CLUSTERS = 100
        
        # Persistence
        self._cache_path = Path.home() / ".jarvis" / "embedding_cache"
        self._cache_path.mkdir(parents=True, exist_ok=True)
        self._dirty = False  # Track if changes need saving
        
        logger.info("Embedding Interconnect initialized")
        logger.info("  Dimension: 768 (Jina v5)")
        logger.info(f"  Mode: {'IPC client (shared server)' if self._using_ipc else 'Local model'}")
        logger.info("  Shared across: dual_model, vector_memory, knowledge_graph, insights, orchestrator")
        logger.info(f"  Cache path: {self._cache_path}")
    
    def encode(self, text: str) -> "np.ndarray":
        """Encode text using shared embedder."""
        return self.embedder.encode(text)

    def health(self) -> dict:
        """Report whether embeddings are usable, plus which backend is serving.

        Every semantic subsystem funnels through this object, so a degraded
        embedder here silently corrupts vector memory, the knowledge graph and
        insight retrieval at once. Surfaced so the health manager can report it.
        """
        inner = getattr(self.embedder, "health", None)
        info = inner() if callable(inner) else {
            "healthy": True,
            "status": "unknown",
            "message": "Backend exposes no health signal",
        }
        return {
            **info,
            "mode": "ipc_shared_server" if self._using_ipc else "local_model",
            "total_embeddings": self._total_embeddings,
        }
    
    def store_model_output(self, model: str, prompt: str, response: str, quality: float = 1.0):
        """Store model output in embedding space (called by dual_model).
        
        Pre-computes embedding outside the lock for concurrency.
        """
        import time
        import hashlib
        
        # Pre-compute embedding OUTSIDE the lock (50-200ms)
        prompt_emb = self.encode(prompt[:500])
        prompt_hash = hashlib.sha256(prompt[:500].encode()).hexdigest()[:16]
        
        with self._lock:
            self._model_outputs[model].append((prompt[:500], quality, prompt_emb))
            if len(self._model_outputs[model]) > 100:
                # Prune stale embeddings whose list entries were evicted
                current_hashes = {hashlib.sha256(p[:500].encode()).hexdigest()[:16] for p, _, _ in self._model_outputs[model][-100:]}
                stale_hashes = [k for k in list(self._model_output_embeddings[model].keys()) if k not in current_hashes]
                for k in stale_hashes:
                    del self._model_output_embeddings[model][k]
                self._model_outputs[model] = self._model_outputs[model][-100:]
            
            self._model_output_embeddings[model][prompt_hash] = prompt_emb
            
            for m in list(self._model_output_embeddings.keys()):
                if len(self._model_output_embeddings[m]) > 200:
                    keys = list(self._model_output_embeddings[m].keys())
                    for k in keys[:-200]:
                        del self._model_output_embeddings[m][k]
            
            self._last_responses[model] = response[:500]
            
            # Track lifetime stats
            self._total_embeddings += 1
            now = time.time()
            if self._last_embedding_time is not None:
                time_diff = now - self._last_embedding_time
                if time_diff > 0:
                    self._embedding_rate_per_min = 60.0 / time_diff
            self._last_embedding_time = now
            
            logger.debug(f"Stored {model} output: prompt_len={len(prompt[:500])}, total={self._total_embeddings}")
    
    def store_conversation(self, role: str, content: str):
        """Store conversation turn (shared across ALL models)."""
        emb = self.encode(content[:500])
        with self._lock:
            self._conversation_embeddings.append((f"{role}: {content[:100]}", emb))
            self._conversation_texts.append(content[:500])
            
            if len(self._conversation_embeddings) > 50:
                self._conversation_embeddings = self._conversation_embeddings[-50:]
                self._conversation_texts = self._conversation_texts[-50:]
    
    def get_conversation_context(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Retrieve relevant conversation turns for context."""
        query_emb = self.encode(query)
        with self._lock:
            if not self._conversation_embeddings:
                return []
            
            results = []
            for text, emb in self._conversation_embeddings:
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                if sim > 0.5:
                    results.append((text, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def get_cross_model_context(self, query: str, from_model: str) -> str:
        """Get context from the OTHER model (for dual_model interconnect)."""
        other_model = "primary" if from_model == "backend" else "backend"
        query_emb = self.encode(query)
        
        with self._lock:
            if not self._model_outputs.get(other_model):
                return ""
            
            best_match = None
            best_sim = 0.0
            
            for stored_prompt, quality, stored_emb in self._model_outputs.get(other_model, []):
                sim = float(np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best_match = stored_prompt
            
            if best_match and best_sim > 0.65:
                last_response = self._last_responses.get(other_model, "")
                context = f"[{other_model.capitalize()} context (sim={best_sim:.0%})]"
                if last_response:
                    context += f"\nRelevant: {last_response[:200]}"
                logger.info(f"Cross-model context: {other_model} (sim={best_sim:.0%})")
                return context
            
            return ""
    
    def _evict_embedding(self, embedding_dict: dict, order_list: deque, max_size: int, metadata_dict: dict = None, order_set: set = None) -> None:
        """Evict oldest entries when over limit."""
        while len(embedding_dict) > max_size and order_list:
            oldest = order_list.popleft()
            if order_set is not None:
                order_set.discard(oldest)
            embedding_dict.pop(oldest, None)
            if metadata_dict is not None:
                metadata_dict.pop(oldest, None)
    
    def store_memory(self, memory_id: str, content: str):
        """Store memory in embedding space (for unified_memory)."""
        emb = self.encode(content[:500])
        with self._lock:
            self._memory_embeddings[memory_id] = emb
            if memory_id not in self._memory_order_set:
                self._memory_order.append(memory_id)
                self._memory_order_set.add(memory_id)
            else:
                try:
                    self._memory_order.remove(memory_id)
                    self._memory_order.append(memory_id)
                except ValueError:
                    pass
            self._evict_embedding(self._memory_embeddings, self._memory_order, self._MAX_EMBEDDINGS_PER_TYPE, order_set=self._memory_order_set)
    
    def search_memories(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search memories by semantic similarity."""
        query_emb = self.encode(query)
        with self._lock:
            if not self._memory_embeddings:
                return []
            
            results = []
            for memory_id, emb in self._memory_embeddings.items():
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                if sim > 0.5:
                    results.append((memory_id, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def store_insight(self, insight_id: str, content: str):
        """Store insight in embedding space (for insight_bridge)."""
        emb = self.encode(content[:500])
        with self._lock:
            self._insight_embeddings[insight_id] = emb
            if insight_id not in self._insight_order_set:
                self._insight_order.append(insight_id)
                self._insight_order_set.add(insight_id)
            else:
                try:
                    self._insight_order.remove(insight_id)
                    self._insight_order.append(insight_id)
                except ValueError:
                    pass
            self._evict_embedding(self._insight_embeddings, self._insight_order, self._MAX_EMBEDDINGS_PER_TYPE, order_set=self._insight_order_set)
    
    def search_insights(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search insights by semantic similarity."""
        query_emb = self.encode(query)
        with self._lock:
            if not self._insight_embeddings:
                return []
            
            results = []
            for insight_id, emb in self._insight_embeddings.items():
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                if sim > 0.5:
                    results.append((insight_id, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def store_knowledge_node(self, node_id: str, content: str):
        """Store knowledge graph node in embedding space."""
        emb = self.encode(content[:500])
        with self._lock:
            self._knowledge_embeddings[node_id] = emb
            if node_id not in self._knowledge_order_set:
                self._knowledge_order.append(node_id)
                self._knowledge_order_set.add(node_id)
            else:
                try:
                    self._knowledge_order.remove(node_id)
                    self._knowledge_order.append(node_id)
                except ValueError:
                    pass
            self._evict_embedding(self._knowledge_embeddings, self._knowledge_order, self._MAX_EMBEDDINGS_PER_TYPE, order_set=self._knowledge_order_set)
    
    def search_knowledge(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search knowledge graph by semantic similarity."""
        query_emb = self.encode(query)
        with self._lock:
            if not self._knowledge_embeddings:
                return []
            
            results = []
            for node_id, emb in self._knowledge_embeddings.items():
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                if sim > 0.5:
                    results.append((node_id, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    # ==================== WORKFLOW EMBEDDINGS ====================
    
    def store_workflow(self, workflow_id: str, request: str, decomposition: dict, success: bool = True):
        """Store workflow decomposition for learning and retrieval.
        
        Pre-computes embedding outside lock for concurrency.
        """
        request_emb = self.encode(request[:500])
        
        with self._lock:
            self._workflow_embeddings[workflow_id] = request_emb
            if workflow_id not in self._workflow_order_set:
                self._workflow_order.append(workflow_id)
                self._workflow_order_set.add(workflow_id)
            else:
                try:
                    self._workflow_order.remove(workflow_id)
                    self._workflow_order.append(workflow_id)
                except ValueError:
                    pass
            self._evict_embedding(self._workflow_embeddings, self._workflow_order, self._MAX_WORKFLOW_EMBEDDINGS, self._workflow_metadata, order_set=self._workflow_order_set)
            
            if workflow_id not in self._workflow_metadata:
                self._workflow_metadata[workflow_id] = {
                    "request": request[:500],
                    "decomposition": decomposition,
                    "success_count": 0,
                    "total_count": 0,
                    "created_at": datetime.now().isoformat(),
                }
            
            self._workflow_metadata[workflow_id]["total_count"] += 1
            if success:
                self._workflow_metadata[workflow_id]["success_count"] += 1
            
            self._dirty = True
            logger.debug(f"Stored workflow {workflow_id}: success={success}")
    
    def search_workflows(self, query: str, top_k: int = 5, min_success_rate: float = 0.5) -> list[tuple[str, float, dict]]:
        """Search for similar past workflows."""
        query_emb = self.encode(query)
        with self._lock:
            if not self._workflow_embeddings:
                return []
            
            results = []
            for workflow_id, emb in self._workflow_embeddings.items():
                metadata = self._workflow_metadata.get(workflow_id, {})
                
                total = metadata.get("total_count", 1)
                successes = metadata.get("success_count", total)
                success_rate = successes / total if total > 0 else 1.0
                
                if success_rate < min_success_rate:
                    continue
                
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                
                if sim > 0.5:
                    results.append((workflow_id, sim, metadata))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def get_workflow_patterns(self, query: str, top_k: int = 3) -> list[dict]:
        """Get successful workflow patterns for a query.
        
        Returns decomposition patterns from similar successful workflows.
        """
        workflows = self.search_workflows(query, top_k=top_k, min_success_rate=0.7)
        
        patterns = []
        for workflow_id, sim, metadata in workflows:
            decomposition = metadata.get("decomposition", {})
            patterns.append({
                "workflow_id": workflow_id,
                "similarity": sim,
                "subtasks": decomposition.get("subtasks", []),
                "execution_mode": decomposition.get("execution_mode", "sequential"),
                "success_rate": metadata.get("success_count", 1) / max(metadata.get("total_count", 1), 1),
            })
        
        return patterns
    
    # ==================== CLUSTERING ====================
    
    def cluster_embeddings(self, embeddings: dict[str, np.ndarray] = None, n_clusters: int = 10) -> dict:
        """Cluster embeddings for faster retrieval.
        
        Groups similar content, returns cluster centroids.
        Uses k-means-style clustering for efficiency.
        
        Args:
            embeddings: Optional specific embeddings to cluster (default: all)
            n_clusters: Number of clusters
        
        Returns:
            Dict mapping cluster_id to list of ids
        """
        # Snapshot embeddings outside lock (copy dict+arrays)
        if embeddings is None:
            with self._lock:
                all_embeddings = {}
                all_embeddings.update(self._memory_embeddings)
                all_embeddings.update(self._knowledge_embeddings)
                all_embeddings.update(self._workflow_embeddings)
            embeddings = all_embeddings
        
        if len(embeddings) < n_clusters:
            return {"cluster_0": list(embeddings.keys())}
        
        ids = list(embeddings.keys())
        vectors = np.array([embeddings[id] for id in ids])
        
        # KMeans is expensive — run without holding the lock
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            labels = kmeans.fit_predict(vectors)
            centroids = list(kmeans.cluster_centers_)
            
            # Update cluster state under lock (fast dict mutations only)
            with self._lock:
                self._clusters = {}
                self._cluster_centroids = {}
                self._cluster_order = deque()
                self._cluster_order_set = set()
                
                for i, (id, label) in enumerate(zip(ids, labels)):
                    cluster_id = f"cluster_{label}"
                    if cluster_id not in self._clusters:
                        self._clusters[cluster_id] = []
                    self._clusters[cluster_id].append(id)
                
                for i, centroid in enumerate(centroids):
                    cluster_key = f"cluster_{i}"
                    self._cluster_centroids[cluster_key] = centroid
                    self._cluster_order.append(cluster_key)
                    self._cluster_order_set.add(cluster_key)
                
                # Evict oldest clusters if over limit
                while len(self._clusters) > self.MAX_CLUSTERS and self._cluster_order:
                    oldest = self._cluster_order.popleft()
                    self._cluster_order_set.discard(oldest)
                    self._clusters.pop(oldest, None)
                    self._cluster_centroids.pop(oldest, None)
                
                logger.info(f"Clustered {len(ids)} embeddings into {n_clusters} clusters")
                return dict(self._clusters)

        except ImportError:
            return {"cluster_0": ids}
    
    def search_in_cluster(self, query: str, cluster_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search within a specific cluster for faster retrieval."""
        query_emb = self.encode(query)
        with self._lock:
            cluster_ids = self._clusters.get(cluster_id, [])
            if not cluster_ids:
                return []
            
            results = []
            for emb_dict in [self._memory_embeddings, self._knowledge_embeddings, 
                            self._workflow_embeddings]:
                for id in cluster_ids:
                    if id in emb_dict:
                        emb = emb_dict[id]
                        sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                        if sim > 0.5:
                            results.append((id, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    # ==================== PERSISTENCE ====================
    
    def save_to_disk(self) -> bool:
        """Persist all embeddings to disk.
        
        Uses quantization (FP16) for storage efficiency.
        
        Returns:
            True if successful
        """
        
        with self._lock:
            try:
                serializable_model_outputs = {}
                for model, outputs in self._model_outputs.items():
                    serializable_model_outputs[model] = [
                        (prompt, quality, emb.astype(np.float16).tolist())
                        for prompt, quality, emb in outputs
                    ]
                
                data = {
                    "version": 2,
                    "timestamp": datetime.now().isoformat(),
                    "model_outputs": serializable_model_outputs,
                    "last_responses": dict(self._last_responses),
                    "conversation_texts": self._conversation_texts[-50:],
                    "memory_embeddings": {k: v.astype(np.float16).tolist() for k, v in self._memory_embeddings.items()},
                    "insight_embeddings": {k: v.astype(np.float16).tolist() for k, v in self._insight_embeddings.items()},
                    "knowledge_embeddings": {k: v.astype(np.float16).tolist() for k, v in self._knowledge_embeddings.items()},
                    "workflow_embeddings": {k: v.astype(np.float16).tolist() for k, v in self._workflow_embeddings.items()},
                    "workflow_metadata": dict(self._workflow_metadata),
                }
                
                memory_keys = list(self._memory_embeddings.keys())
                memory_vecs = np.array([v.astype(np.float16) for v in self._memory_embeddings.values()]) if self._memory_embeddings else np.array([])
                insight_keys = list(self._insight_embeddings.keys())
                insight_vecs = np.array([v.astype(np.float16) for v in self._insight_embeddings.values()]) if self._insight_embeddings else np.array([])
                knowledge_keys = list(self._knowledge_embeddings.keys())
                knowledge_vecs = np.array([v.astype(np.float16) for v in self._knowledge_embeddings.values()]) if self._knowledge_embeddings else np.array([])
                workflow_keys = list(self._workflow_embeddings.keys())
                workflow_vecs = np.array([v.astype(np.float16) for v in self._workflow_embeddings.values()]) if self._workflow_embeddings else np.array([])
            except Exception as e:
                logger.error(f"Failed to snapshot embeddings for save: {e}")
                return False
        
        try:
            cache_file = self._cache_path / "embeddings.json"
            atomic_write_json(cache_file, data)
            
            embeddings_file = self._cache_path / "embeddings.npz.gz"
            np.savez_compressed(
                embeddings_file,
                memory_ids=memory_keys,
                memory_vecs=memory_vecs,
                insight_ids=insight_keys,
                insight_vecs=insight_vecs,
                knowledge_ids=knowledge_keys,
                knowledge_vecs=knowledge_vecs,
                workflow_ids=workflow_keys,
                workflow_vecs=workflow_vecs,
            )
            
            with self._lock:
                self._dirty = False
            logger.info(f"Saved embeddings to disk: {cache_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False
    
    def load_from_disk(self) -> bool:
        """Load embeddings from disk.
        
        Fast startup by avoiding re-embedding large corpus.
        
        Returns:
            True if successful
        """
        with self._lock:
            try:
                cache_file = self._cache_path / "embeddings.json"
                
                if not cache_file.exists():
                    logger.info("No cached embeddings found, starting fresh")
                    return False
                
                # Load JSON data
                data = read_json_with_integrity(cache_file)
                
                # Restore model outputs (need to convert lists back to ndarrays)
                raw_model_outputs = data.get("model_outputs", {"primary": [], "backend": [], "synthesis": []})
                self._model_outputs = {}
                for model, outputs in raw_model_outputs.items():
                    self._model_outputs[model] = [
                        (prompt, quality, np.array(emb, dtype=np.float32))
                        for prompt, quality, emb in outputs
                    ]
                self._last_responses = data.get("last_responses", {"primary": "", "backend": ""})
                self._conversation_texts = data.get("conversation_texts", [])
                self._workflow_metadata = data.get("workflow_metadata", {})
                
                # Load compressed embeddings
                embeddings_file = self._cache_path / "embeddings.npz.gz"
                if embeddings_file.exists():
                    loaded = np.load(embeddings_file, allow_pickle=True)
                    
                    # Restore memory embeddings
                    if len(loaded["memory_ids"]) > 0:
                        for id, vec in zip(loaded["memory_ids"], loaded["memory_vecs"]):
                            self._memory_embeddings[id] = vec.astype(np.float32)
                    
                    # Restore insight embeddings
                    if len(loaded["insight_ids"]) > 0:
                        for id, vec in zip(loaded["insight_ids"], loaded["insight_vecs"]):
                            self._insight_embeddings[id] = vec.astype(np.float32)
                    
                    # Restore knowledge embeddings
                    if len(loaded["knowledge_ids"]) > 0:
                        for id, vec in zip(loaded["knowledge_ids"], loaded["knowledge_vecs"]):
                            self._knowledge_embeddings[id] = vec.astype(np.float32)
                    
                    # Restore workflow embeddings
                    if len(loaded["workflow_ids"]) > 0:
                        for id, vec in zip(loaded["workflow_ids"], loaded["workflow_vecs"]):
                            self._workflow_embeddings[id] = vec.astype(np.float32)
                
                # Restore conversation texts (will re-encode outside lock)
                conv_texts = list(self._conversation_texts[:50])
            
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                return False
        
        # Re-encode conversation texts outside lock (slow embedding calls)
        conv_embs = []
        for text in conv_texts:
            try:
                emb = self.encode(text[:500])
                conv_embs.append((text[:100], emb))
            except Exception as e:
                logger.debug(f"Embedding encode failed: {e}")
        
        with self._lock:
            self._conversation_embeddings = conv_embs
            
            # Rebuild order sets from loaded embeddings
            self._memory_order = deque(self._memory_embeddings.keys())
            self._memory_order_set = set(self._memory_embeddings.keys())
            self._insight_order = deque(self._insight_embeddings.keys())
            self._insight_order_set = set(self._insight_embeddings.keys())
            self._knowledge_order = deque(self._knowledge_embeddings.keys())
            self._knowledge_order_set = set(self._knowledge_embeddings.keys())
            self._workflow_order = deque(self._workflow_embeddings.keys())
            self._workflow_order_set = set(self._workflow_embeddings.keys())
            
            total = (len(self._memory_embeddings) + len(self._insight_embeddings) +
                     len(self._knowledge_embeddings) + len(self._workflow_embeddings))
            
            logger.info(f"Loaded {total} embeddings from disk")
            logger.info(f"  Memory: {len(self._memory_embeddings)}")
            logger.info(f"  Insight: {len(self._insight_embeddings)}")
            logger.info(f"  Knowledge: {len(self._knowledge_embeddings)}")
            logger.info(f"  Workflow: {len(self._workflow_embeddings)}")
            
            return True
    
    def get_model_outputs(self, model: str) -> list[tuple[str, float, np.ndarray]]:
        """Get a snapshot of model outputs (thread-safe).
        
        Returns a copy of the model outputs list for the given model role.
        Each entry is (prompt, quality, embedding).
        
        Args:
            model: "primary", "backend", or "synthesis"
        
        Returns:
            List of (prompt, quality, embedding) tuples
        """
        with self._lock:
            return list(self._model_outputs.get(model, []))
    
    def get_last_responses(self) -> dict[str, str]:
        """Get a snapshot of the last responses from each model (thread-safe).
        
        Returns:
            Dict mapping model name to last response string
        """
        with self._lock:
            return dict(self._last_responses)
    
    def get_stats(self) -> dict:
        """Get comprehensive embedding statistics."""
        with self._lock:
            return {
                "memory_embeddings": len(self._memory_embeddings),
                "insight_embeddings": len(self._insight_embeddings),
                "knowledge_embeddings": len(self._knowledge_embeddings),
                "workflow_embeddings": len(self._workflow_embeddings),
                "conversation_turns": len(self._conversation_embeddings),
                "model_outputs": {
                    "primary": len(self._model_outputs.get("primary", [])),
                    "backend": len(self._model_outputs.get("backend", [])),
                    "synthesis": len(self._model_outputs.get("synthesis", [])),
                },
                "clusters": len(self._clusters),
                "embedding_dim": 768,
                "cache_dirty": self._dirty,
            }


def get_embedding_interconnect() -> EmbeddingInterconnect:
    """Get singleton embedding interconnect instance."""
    global _interconnect
    
    if _interconnect is None:
        with _lock:
            if _interconnect is None:
                _interconnect = EmbeddingInterconnect()
    
    return _interconnect


def reset_embedding_interconnect():
    """Reset singleton - use sparingly, mainly for tests."""
    global _interconnect
    with _lock:
        _interconnect = None
