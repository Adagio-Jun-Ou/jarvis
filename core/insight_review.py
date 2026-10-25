"""Insight Review Module - Search, filter, and analyze insights."""
import hashlib
import threading
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Any

from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)

_insight_review: "InsightReview | None" = None
_lock = threading.Lock()


def get_insight_review() -> "InsightReview":
    """Get singleton InsightReview instance."""
    global _insight_review
    if _insight_review is None:
        with _lock:
            if _insight_review is None:
                _insight_review = InsightReview()
    return _insight_review


class InsightReview:
    """Review, search, and analyze insights from insight_bridge and daydream."""
    
    MAX_WORD_FREQUENCY = 500  # Max words to track in heatmap
    MAX_INSIGHTS = 5000  # Max stored insights
    
    def __init__(self):
        self.storage_path = Path.home() / ".jarvis" / "insights"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._index: dict[str, list[str]] = {}  # word -> insight_ids
        self._hash_index: dict[str, str] = {}   # content_hash -> insight_id
        self._insights: dict[str, dict] = {}     # id -> insight_data
        self._word_frequency: Counter = Counter()
        self._source_index: dict[str, list[str]] = {}  # source -> insight_ids
        self._type_index: dict[str, list[str]] = {}   # type -> insight_ids
        
        self._load_state()
    
    def _load_state(self) -> None:
        """Load indices from disk."""
        index_file = self.storage_path / "review_index.json"
        if index_file.exists():
            try:
                data = read_json_with_integrity(index_file)
                self._index = data.get("word_index", {})
                self._hash_index = data.get("hash_index", {})
                self._word_frequency = Counter(data.get("word_frequency", {}))
                self._source_index = data.get("source_index", {})
                self._type_index = data.get("type_index", {})
                self._insights = data.get("insights", {})
                logger.info(f"Loaded insight review index: {len(self._hash_index)} hashes, {len(self._word_frequency)} words, {len(self._insights)} insights")
            except Exception as e:
                logger.warning(f"Failed to load insight review index: {e}")
    
    def _save_state(self) -> None:
        """Save indices to disk."""
        index_file = self.storage_path / "review_index.json"
        try:
            atomic_write_json(index_file, {
                    "word_index": self._index,
                    "hash_index": self._hash_index,
                    "word_frequency": dict(self._word_frequency.most_common(self.MAX_WORD_FREQUENCY)),
                    "source_index": self._source_index,
                    "type_index": self._type_index,
                    "insights": self._insights,
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save insight review index: {e}")
    
    def _content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _extract_words(self, content: str) -> list[str]:
        """Extract significant words from content."""
        # Remove common words and short tokens
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "until", "while", "this", "that",
            "these", "those", "it", "its", "which", "who", "whom", "what",
        }
        
        # Tokenize and filter
        words = []
        for word in content.lower().split():
            # Clean word
            clean = "".join(c for c in word if c.isalnum())
            if len(clean) >= 3 and clean not in stop_words:
                words.append(clean)
        
        return words
    
    def index_insight(self, insight_id: str, content: str, source: str, 
                      insight_type: str, timestamp: str, 
                      problem: str = "", category: str = "") -> None:
        """Index an insight for search and analysis."""
        with self._lock:
            # Generate content hash for deduplication
            content_hash = self._content_hash(content)
            
            # Skip if already indexed with full data
            if content_hash in self._hash_index:
                existing_id = self._hash_index[content_hash]
                if existing_id in self._insights:
                    return
            
            # Always add/update hash and insight
            self._hash_index[content_hash] = insight_id
            
            # Store insight data
            self._insights[insight_id] = {
                "id": insight_id,
                "content": content,
                "content_hash": content_hash,
                "source": source,
                "type": insight_type,
                "category": category,
                "problem": problem,
                "timestamp": timestamp,
                "indexed_at": datetime.now().isoformat(),
            }
            
            # Index by source
            if source not in self._source_index:
                self._source_index[source] = []
            if insight_id not in self._source_index[source]:
                self._source_index[source].append(insight_id)
            
            # Index by type
            if insight_type not in self._type_index:
                self._type_index[insight_type] = []
            if insight_id not in self._type_index[insight_type]:
                self._type_index[insight_type].append(insight_id)
            
            # Index words
            words = self._extract_words(content)
            for word in words:
                if word not in self._index:
                    self._index[word] = []
                if insight_id not in self._index[word]:
                    self._index[word].append(insight_id)
                self._word_frequency[word] += 1
            
            # Evict oldest insights if over limit
            if len(self._insights) > self.MAX_INSIGHTS:
                oldest_ids = list(self._insights.keys())[:len(self._insights) - self.MAX_INSIGHTS]
                for old_id in oldest_ids:
                    old_data = self._insights.pop(old_id, None)
                    if old_data:
                        old_hash = old_data.get("content_hash", "")
                        self._hash_index.pop(old_hash, None)
                    for word_list in self._index.values():
                        if old_id in word_list:
                            word_list.remove(old_id)
                    for src_list in self._source_index.values():
                        if old_id in src_list:
                            src_list.remove(old_id)
                    for type_list in self._type_index.values():
                        if old_id in type_list:
                            type_list.remove(old_id)
                # Clean up empty index entries
                self._index = {k: v for k, v in self._index.items() if v}
                self._source_index = {k: v for k, v in self._source_index.items() if v}
                self._type_index = {k: v for k, v in self._type_index.items() if v}
            
            # Save periodically
            if len(self._hash_index) % 100 == 0:
                self._save_state()
    
    def index_from_bridge(self) -> int:
        """Index all insights from insight_bridge."""
        try:
            from core.insight_bridge import get_insight_bridge
            bridge = get_insight_bridge()
            
            count = 0
            for insight in bridge.processed_insights:
                self.index_insight(
                    insight_id=insight.id,
                    content=insight.raw_insight,
                    source="insight_bridge",
                    insight_type=insight.insight_type.value if hasattr(insight.insight_type, 'value') else str(insight.insight_type),
                    timestamp=insight.timestamp.isoformat() if hasattr(insight.timestamp, 'isoformat') else str(insight.timestamp),
                    problem=insight.source_problem,
                    category=insight.source_category,
                )
                count += 1
            
            # Save after bulk indexing
            self._save_state()
            logger.info(f"Indexed {count} insights from bridge")
            return count
        except Exception as e:
            logger.error(f"Failed to index from bridge: {e}")
            return 0
    
    def search(self, query: str, filters: dict | None = None) -> list[dict]:
        """Search insights by query and filters.
        
        Args:
            query: Search query string
            filters: Optional filters (source, type, category)
        
        Returns:
            List of matching insights
        """
        with self._lock:
            # Find insights matching query
            query_words = self._extract_words(query.lower())
            
            if not query_words:
                # Return all if empty query
                matching_ids = set(self._insights.keys())
            else:
                # Find insights containing any query word
                matching_ids = set()
                for word in query_words:
                    if word in self._index:
                        matching_ids.update(self._index[word])
            
            # Apply filters
            if filters:
                if filters.get("source"):
                    source_ids = set(self._source_index.get(filters["source"], []))
                    matching_ids &= source_ids
                
                if filters.get("type"):
                    type_ids = set(self._type_index.get(filters["type"], []))
                    matching_ids &= type_ids
            
            # Get insight data
            results = []
            for iid in matching_ids:
                if iid in self._insights:
                    results.append(self._insights[iid])
            
            # Sort by timestamp descending
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return results
    
    def get_by_hash(self, content_hash: str) -> dict | None:
        """Get insight by content hash."""
        with self._lock:
            insight_id = self._hash_index.get(content_hash)
            if insight_id and insight_id in self._insights:
                return self._insights[insight_id]
            return None
    
    def get_word_heatmap(self, limit: int = 100) -> list[tuple[str, int]]:
        """Get word frequency heatmap.
        
        Returns:
            List of (word, count) tuples sorted by frequency
        """
        with self._lock:
            return self._word_frequency.most_common(limit)
    
    def get_sources(self) -> list[str]:
        """Get list of all insight sources."""
        with self._lock:
            return list(self._source_index.keys())
    
    def get_types(self) -> list[str]:
        """Get list of all insight types."""
        with self._lock:
            return list(self._type_index.keys())
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about indexed insights."""
        with self._lock:
            return {
                "total_insights": len(self._insights),
                "total_hashes": len(self._hash_index),
                "unique_words": len(self._word_frequency),
                "sources": {k: len(v) for k, v in self._source_index.items()},
                "types": {k: len(v) for k, v in self._type_index.items()},
                "top_words": self._word_frequency.most_common(20),
            }
    
    def get_all(self, limit: int = 0) -> list[dict]:
        """Get all indexed insights.
        
        Args:
            limit: Max results (0 = all)
        
        Returns:
            List of all insights
        """
        with self._lock:
            results = list(self._insights.values())
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            if limit > 0:
                return results[:limit]
            return results
    
    def find_similar_by_hash(self, content_hash: str, limit: int = 10) -> list[dict]:
        """Find similar insights by looking up content hash and searching for related words.
        
        Args:
            content_hash: Hash of content to find similar insights for
            limit: Max results
        
        Returns:
            List of similar insights
        """
        with self._lock:
            # Get the original insight
            original = self._hash_index.get(content_hash)
            if not original or original not in self._insights:
                return []
            
            original_data = self._insights[original]
            original_words = set(self._extract_words(original_data.get("content", "")))
            
            if not original_words:
                return []
            
            # Find insights with overlapping words
            similar_scores: dict[str, int] = {}
            for word in original_words:
                if word in self._index:
                    for iid in self._index[word]:
                        if iid != original:
                            similar_scores[iid] = similar_scores.get(iid, 0) + 1
            
            # Sort by score and return top results
            sorted_ids = sorted(similar_scores.items(), key=lambda x: -x[1])[:limit]
            results = []
            for iid, score in sorted_ids:
                if iid in self._insights:
                    insight = self._insights[iid].copy()
                    insight["similarity_score"] = score
                    results.append(insight)
            
            return results
    
    def find_by_content(self, content: str, threshold: float = 0.5) -> list[dict]:
        """Find insights with similar content using word overlap.
        
        Args:
            content: Content to search for similar insights
            threshold: Minimum similarity ratio (0-1)
        
        Returns:
            List of similar insights with similarity scores
        """
        self._content_hash(content)
        original_words = set(self._extract_words(content))
        
        if not original_words:
            return []
        
        with self._lock:
            # Find insights with overlapping words
            similar_scores: dict[str, int] = {}
            for word in original_words:
                if word in self._index:
                    for iid in self._index[word]:
                        similar_scores[iid] = similar_scores.get(iid, 0) + 1
            
            # Calculate similarity and filter by threshold
            results = []
            for iid, overlap in similar_scores.items():
                if iid in self._insights:
                    insight_data = self._insights[iid]
                    insight_words = set(self._extract_words(insight_data.get("content", "")))
                    union_size = len(original_words | insight_words)
                    if union_size > 0:
                        similarity = overlap / union_size
                        if similarity >= threshold:
                            insight = insight_data.copy()
                            insight["similarity"] = round(similarity, 3)
                            insight["overlap_words"] = overlap
                            results.append(insight)
            
            # Sort by similarity descending
            results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            return results
    
    def check_duplicate(self, content: str) -> dict | None:
        """Check if content is a duplicate of existing insight.
        
        Args:
            content: Content to check
        
        Returns:
            Existing insight if duplicate, None otherwise
        """
        content_hash = self._content_hash(content)
        return self.get_by_hash(content_hash)
    
    def get_hash(self, content: str) -> str:
        """Generate content hash for external use.
        
        Args:
            content: Content to hash
        
        Returns:
            SHA256 hash (first 16 chars)
        """
        return self._content_hash(content)
    
    def get_insights_by_type(self, insight_type: str, limit: int = 100) -> list[dict]:
        """Get all insights of a specific type.
        
        Args:
            insight_type: Type of insight to retrieve
            limit: Maximum results
        
        Returns:
            List of insights of that type
        """
        with self._lock:
            ids = self._type_index.get(insight_type, [])
            results = [self._insights[iid] for iid in ids if iid in self._insights]
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results[:limit]
    
    def get_insights_by_source(self, source: str, limit: int = 100) -> list[dict]:
        """Get all insights from a specific source.
        
        Args:
            source: Source to retrieve from
            limit: Maximum results
        
        Returns:
            List of insights from that source
        """
        with self._lock:
            ids = self._source_index.get(source, [])
            results = [self._insights[iid] for iid in ids if iid in self._insights]
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results[:limit]
    
    def get_word_context(self, word: str, limit: int = 20) -> list[dict]:
        """Get insights containing a specific word with context.
        
        Args:
            word: Word to search for
            limit: Maximum results
        
        Returns:
            List of insights containing the word
        """
        word_lower = word.lower()
        with self._lock:
            ids = self._index.get(word_lower, [])
            results = [self._insights[iid] for iid in ids[:limit] if iid in self._insights]
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results
    
    def get_all_insights(self, limit: int = 1000) -> list[dict]:
        """Get all insights sorted by timestamp.
        
        Args:
            limit: Maximum results (default 1000)
        
        Returns:
            List of all insights sorted by timestamp (newest first)
        """
        with self._lock:
            results = list(self._insights.values())
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results[:limit]