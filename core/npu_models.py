"""NPU-Optimized Models for Jarvis.

This module provides models for fast processing:
1. Text Embeddings - Fast file/document embeddings for retrieval (CPU)
2. Voice Tokenizers - Semantic and acoustic tokenizers for voice processing
3. File Retrieval - Semantic search over indexed documents

Embeddings run on CPU (sentence-transformers), NPU available for future use.
"""
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HAS_TORCH = False
HAS_TRANSFORMERS = False
HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass


MODEL_CACHE = Path.home() / ".jarvis" / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingResult:
    text: str
    embedding: np.ndarray
    model: str
    tokens: int
    inference_ms: float


@dataclass
class VoiceTokenResult:
    semantic_tokens: list[int]
    acoustic_tokens: list[list[int]]
    duration_ms: float
    inference_ms: float


class FastEmbeddingModel:
    """Fast text embedding model for file/document encoding.
    
    Uses sentence-transformers for reliable, fast embeddings on CPU.
    Perfect for document retrieval and semantic search.
    """

    MODELS = {
        "mini": "sentence-transformers/all-MiniLM-L6-v2",
        "small": "sentence-transformers/all-mpnet-base-v2",
        "bge": "BAAI/bge-small-en-v1.5",
    }

    DIMS = {"mini": 384, "small": 768, "bge": 384}
    DEFAULT = "mini"

    def __init__(self, model_type: str = "mini", force_cpu: bool = True):
        self.model_type = model_type
        self.model_name = self.MODELS.get(model_type, self.MODELS[self.DEFAULT])
        self.embedding_dim = self.DIMS.get(model_type, 384)
        self._model = None
        self._loaded = False

        if force_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['ROCR_VISIBLE_DEVICES'] = ''

    def _load(self):
        if self._loaded:
            return

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformers: {e}")

        self._loaded = True

    def encode(self, texts: str | list[str]) -> list[EmbeddingResult]:
        self._load()

        if isinstance(texts, str):
            texts = [texts]

        results = []
        start = time.perf_counter()

        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            inference_ms = (time.perf_counter() - start) * 1000

            for i, text in enumerate(texts):
                emb = embeddings[i] if len(embeddings.shape) > 1 else embeddings
                results.append(EmbeddingResult(
                    text=text,
                    embedding=emb.astype(np.float32),
                    model=self.model_type,
                    tokens=len(text.split()),
                    inference_ms=inference_ms / len(texts) if texts else 0.0,
                ))
        else:
            for text in texts:
                emb = self._hash_embed(text)
                results.append(EmbeddingResult(
                    text=text,
                    embedding=emb,
                    model="fallback",
                    tokens=len(text.split()),
                    inference_ms=0.1,
                ))

        return results

    def _hash_embed(self, text: str) -> np.ndarray:
        emb = np.zeros(self.embedding_dim, dtype=np.float32)
        for word in text.lower().split():
            h = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
            idx = h % self.embedding_dim
            emb[idx] += 1
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class FastVoiceTokenizer:
    """Fast voice tokenization for audio processing."""

    def __init__(self):
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True

    def tokenize(self, audio: np.ndarray, sample_rate: int = 16000) -> VoiceTokenResult:
        start = time.perf_counter()

        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        semantic = self._semantic_tokens(audio)
        acoustic = self._acoustic_tokens(audio, sample_rate)

        inference_ms = (time.perf_counter() - start) * 1000
        duration_ms = len(audio) / sample_rate * 1000

        return VoiceTokenResult(
            semantic_tokens=semantic,
            acoustic_tokens=acoustic,
            duration_ms=duration_ms,
            inference_ms=inference_ms,
        )

    def _semantic_tokens(self, audio: np.ndarray) -> list[int]:
        frame_size = 320
        n_frames = len(audio) // frame_size
        tokens = [1]

        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energy = np.mean(frame ** 2)

            if energy < 0.001:
                tokens.append(3)
            elif energy > 0.1:
                tokens.append(4)
            else:
                token = min(int(energy * 1000), 255) + 10
                tokens.append(token)

        tokens.append(2)
        return tokens

    def _acoustic_tokens(self, audio: np.ndarray, sr: int) -> list[list[int]]:
        frame_samples = int(sr / 12.5)
        n_frames = len(audio) // frame_samples
        tokens = []

        for i in range(n_frames):
            frame = audio[i * frame_samples:(i + 1) * frame_samples]
            frame_tokens = []

            energy = np.mean(frame ** 2)
            frame_tokens.append(int(energy * 32768) % 1024)

            zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            frame_tokens.append(int(zcr * 10000) % 1024)

            for _ in range(6):
                frame_tokens.append(hashlib.sha256(frame.tobytes()).hexdigest()[_ % 64])

            tokens.append(frame_tokens[:8])

        return tokens

    def _resample(self, audio: np.ndarray, orig: int, target: int) -> np.ndarray:
        ratio = target / orig
        n = int(len(audio) * ratio)
        return np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)


class FileRetriever:
    """Semantic file indexing and retrieval system.
    
    Enables Jarvis to:
    - Index codebases and documents
    - Perform semantic search
    - Retrieve context for queries
    """

    def __init__(self, model_type: str = "mini", index_path: Optional[Path] = None):
        self.embedder = FastEmbeddingModel(model_type)
        self.index_path = index_path or MODEL_CACHE / "file_index"
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, dict] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._MAX_INDEX_SIZE = 50000
        self._lock = threading.Lock()

        self._load()

    def _load(self):
        meta = self.index_path / "metadata.json"
        emb = self.index_path / "embeddings.npz"

        if meta.exists():
            try:
                self._index = json.loads(meta.read_text(encoding='utf-8'))
                logger.info(f"Loaded index: {len(self._index)} chunks")
            except Exception as e:
                logger.error(f"Load error: {e}")

        if emb.exists():
            try:
                data = np.load(str(emb), allow_pickle=True)
                self._embeddings = {k: data[k] for k in data.files}
                logger.info(f"Loaded {len(self._embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Embedding load error: {e}")

    def _save(self):
        try:
            meta_path = self.index_path / "metadata.json"
            tmp_meta = meta_path.with_suffix('.tmp')
            tmp_meta.write_text(json.dumps(self._index, indent=2), encoding='utf-8')
            tmp_meta.replace(meta_path)
            if self._embeddings:
                emb_path = self.index_path / "embeddings.npz"
                tmp_emb = emb_path.with_suffix('.tmp.npz')
                np.savez_compressed(str(tmp_emb), **self._embeddings)
                tmp_emb.replace(emb_path)
        except Exception as e:
            logger.error(f"Save error: {e}")

    def index_file(self, path: Path | str, chunk_size: int = 512, overlap: int = 50) -> int:
        path = Path(path)
        if not path.exists():
            return 0

        if path.is_dir():
            return self.index_directory(path, chunk_size, overlap)

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return 0

        chunks = self._chunk(content, chunk_size, overlap)

        with self._lock:
            for i, chunk in enumerate(chunks):
                cid = f"{path}::{i}"
                if cid in self._index:
                    continue

                results = self.embedder.encode(chunk)
                if results:
                    self._index[cid] = {
                        "path": str(path),
                        "chunk": i,
                        "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        "content": chunk,
                    }
                    self._embeddings[cid] = results[0].embedding
                    if len(self._index) > self._MAX_INDEX_SIZE:
                        oldest_key = next(iter(self._index))
                        del self._index[oldest_key]
                        self._embeddings.pop(oldest_key, None)

        self._save()
        return len(chunks)

    def index_directory(
        self,
        directory: Path | str,
        chunk_size: int = 512,
        overlap: int = 50,
        extensions: Optional[list[str]] = None,
    ) -> int:
        directory = Path(directory)
        extensions = extensions or [
            '.py', '.js', '.ts', '.jsx', '.tsx',
            '.md', '.txt', '.json', '.yaml', '.yml',
            '.html', '.css', '.sql', '.sh', '.toml',
        ]

        total = 0
        for ext in extensions:
            for fp in directory.rglob(f"*{ext}"):
                if any(p.startswith('.') for p in fp.parts):
                    continue
                total += self.index_file(fp, chunk_size, overlap)

        return total

    def _chunk(self, text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        if len(words) <= size:
            return [text]

        chunks = []
        for i in range(0, len(words), size - overlap):
            chunks.append(' '.join(words[i:i + size]))
        return chunks

    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
        if not self._embeddings:
            return []

        q_results = self.embedder.encode(query)
        if not q_results:
            return []

        q_emb = q_results[0].embedding

        scores = []
        for cid, emb in self._embeddings.items():
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
            if sim >= threshold:
                scores.append((cid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in scores[:top_k]:
            if cid in self._index:
                r = self._index[cid].copy()
                r["score"] = score
                r["id"] = cid
                results.append(r)

        return results

    def get_context(self, query: str, max_tokens: int = 4000) -> str:
        results = self.search(query, top_k=10, threshold=0.2)

        parts = []
        total = 0
        max_chars = max_tokens * 4

        for r in results:
            text = r.get("content", r.get("preview", ""))
            if total + len(text) > max_chars:
                break
            parts.append(f"**{r['path']}** (score: {r['score']:.2f})\n{text}")
            total += len(text)

        return "\n\n---\n\n".join(parts)

    def clear(self):
        with self._lock:
            self._index.clear()
            self._embeddings.clear()
        self._save()

    def stats(self) -> dict:
        return {
            "files": len(set(v["path"] for v in self._index.values())),
            "chunks": len(self._index),
            "dim": self.embedder.embedding_dim,
            "model": self.embedder.model_type,
            "size_mb": sum(e.nbytes for e in self._embeddings.values()) / (1024 * 1024),
        }


_embedding: Optional[FastEmbeddingModel] = None
_voice: Optional[FastVoiceTokenizer] = None
_retriever: Optional[FileRetriever] = None
_lock = threading.Lock()


def get_embedding(model: str = "mini") -> FastEmbeddingModel:
    global _embedding
    if _embedding is None:
        with _lock:
            if _embedding is None:
                _embedding = FastEmbeddingModel(model)
    return _embedding


def get_voice() -> FastVoiceTokenizer:
    global _voice
    if _voice is None:
        with _lock:
            if _voice is None:
                _voice = FastVoiceTokenizer()
    return _voice


def get_retriever() -> FileRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = FileRetriever()
    return _retriever


__all__ = [
    "FastEmbeddingModel",
    "FastVoiceTokenizer",
    "FileRetriever",
    "EmbeddingResult",
    "VoiceTokenResult",
    "get_embedding",
    "get_voice",
    "get_retriever",
]
