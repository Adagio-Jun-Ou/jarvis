"""Shifting Memory Window for continuous learning without hard stops.

Provides sliding windows for memory collections so Jarvis never stops
accumulating knowledge, just shifts older data to archive while keeping
recent data active.
"""
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar, MutableSequence

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryWindow:
    active_size: int
    archive_size: int
    decay_rate: float
    importance_threshold: float


@dataclass
class ArchivedItem(Generic[T]):
    item: T
    archived_at: datetime
    importance: float
    access_count: int
    original_timestamp: datetime | None = None


class ShiftingMemoryManager:
    """Manages shifting windows for various memory collections.
    
    Instead of hard cutoffs, this system:
    1. Keeps recent items in active memory
    2. Shifts older but important items to archive
    3. Compresses less important items into summaries
    4. Never loses critical knowledge
    """

    DEFAULT_WINDOWS = {
        "problems": MemoryWindow(active_size=100, archive_size=500, decay_rate=0.1, importance_threshold=0.3),
        "history": MemoryWindow(active_size=200, archive_size=1000, decay_rate=0.05, importance_threshold=0.2),
        "sessions": MemoryWindow(active_size=50, archive_size=200, decay_rate=0.1, importance_threshold=0.25),
        "insights": MemoryWindow(active_size=100, archive_size=500, decay_rate=0.1, importance_threshold=0.3),
        "threats": MemoryWindow(active_size=200, archive_size=1000, decay_rate=0.05, importance_threshold=0.4),
        "patterns": MemoryWindow(active_size=100, archive_size=500, decay_rate=0.1, importance_threshold=0.3),
        "evolutions": MemoryWindow(active_size=50, archive_size=200, decay_rate=0.05, importance_threshold=0.5),
    }

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "memory_windows"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.windows: dict[str, MemoryWindow] = dict(self.DEFAULT_WINDOWS)
        self.active: dict[str, deque] = {}
        self.archive: dict[str, list[ArchivedItem]] = {}
        self.summaries: dict[str, dict] = {}
        self.importance_scores: dict[str, dict[str, float]] = {}

        self._lock = threading.RLock()

        self._load_state()

    def _load_state(self) -> None:
        """Load archived data from disk."""
        archive_file = self.storage_path / "archive.json"
        if archive_file.exists():
            try:
                with open(archive_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for name, items in data.get("archive", {}).items():
                        self.archive[name] = [
                            ArchivedItem(
                                item=item["item"],
                                archived_at=datetime.fromisoformat(item["archived_at"]),
                                importance=item["importance"],
                                access_count=item["access_count"],
                                original_timestamp=datetime.fromisoformat(item["original_timestamp"]) if item.get("original_timestamp") else None,
                            )
                            for item in items
                        ]
                    self.summaries = data.get("summaries", {})
                logger.info(f"Loaded memory archives: {sum(len(v) for v in self.archive.values())} items")
            except Exception as e:
                logger.warning(f"Failed to load memory archives: {e}")

    def _save_state(self) -> None:
        """Save archived data to disk. Reads snapshot under lock, writes outside."""
        archive_file = self.storage_path / "archive.json"
        try:
            with self._lock:
                data = {
                    "archive": {
                        name: [
                            {
                                "item": archived.item if isinstance(archived.item, (dict, list, str, int, float, bool, type(None))) else str(archived.item),
                            "archived_at": archived.archived_at.isoformat(),
                            "importance": archived.importance,
                            "access_count": archived.access_count,
                            "original_timestamp": archived.original_timestamp.isoformat() if archived.original_timestamp else None,
                        }
                        for archived in items
                    ]
                    for name, items in self.archive.items()
                },
                "summaries": self.summaries,
                "last_saved": datetime.now().isoformat(),
            }
            tmp = archive_file.with_suffix(".tmp")
            with open(tmp, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            tmp.replace(archive_file)
        except Exception as e:
            logger.warning(f"Failed to save memory archives: {e}")

    def register_collection(self, name: str, items: MutableSequence | deque, window: MemoryWindow | None = None) -> None:
        """Register a collection for shifting window management."""
        with self._lock:
            if window:
                self.windows[name] = window
            if not isinstance(items, deque):
                items_deque = deque(maxlen=self.windows.get(name, MemoryWindow(100, 500, 0.1, 0.3)).active_size)
                items_deque.extend(items)
                self.active[name] = items_deque
            else:
                self.active[name] = items
            if name not in self.archive:
                self.archive[name] = []
            if name not in self.importance_scores:
                self.importance_scores[name] = {}

    def add_item(self, collection: str, item: Any, importance: float = 0.5, timestamp: datetime | None = None) -> None:
        """Add an item with importance tracking."""
        needs_save = False
        with self._lock:
            if collection not in self.active:
                self.active[collection] = deque(maxlen=self.windows.get(collection, MemoryWindow(100, 500, 0.1, 0.3)).active_size)

            self.active[collection].append(item)

            item_id = self._get_item_id(item)
            if item_id:
                self.importance_scores[collection][item_id] = importance

            needs_save = self._check_shift_needed(collection)
        if needs_save:
            self._save_state()

    def _get_item_id(self, item: Any) -> str | None:
        """Get unique identifier for an item."""
        if isinstance(item, dict):
            return item.get("id") or item.get("timestamp") or str(hash(str(item)))
        elif hasattr(item, "id"):
            return str(item.id)
        return None

    def _check_shift_needed(self, collection: str) -> bool:
        """Check if shifting is needed and perform it. Returns True if state was modified."""
        if collection not in self.windows:
            return False

        window = self.windows[collection]
        if collection not in self.active:
            return False

        active = self.active[collection]
        if len(active) >= window.active_size:
            self._shift_window(collection)
            return True
        return False

    def _shift_window(self, collection: str) -> None:
        """Shift older items to archive based on importance."""
        window = self.windows[collection]
        active = self.active[collection]

        if collection not in self.archive:
            self.archive[collection] = []

        shift_count = len(active) - int(window.active_size * 0.8)
        if shift_count <= 0:
            return

        items_to_shift = []
        for _ in range(shift_count):
            if active:
                items_to_shift.append(active.popleft())

        for item in items_to_shift:
            item_id = self._get_item_id(item)
            importance = self.importance_scores.get(collection, {}).get(item_id or "", 0.5)

            if importance >= window.importance_threshold:
                archived = ArchivedItem(
                    item=item,
                    archived_at=datetime.now(),
                    importance=importance,
                    access_count=0,
                    original_timestamp=self._extract_timestamp(item),
                )
                self.archive[collection].append(archived)
            elif collection in self.importance_scores and item_id:
                self.importance_scores[collection].pop(item_id, None)

        if len(self.archive[collection]) > window.archive_size:
            self._compress_archive(collection)

        self._update_summary(collection)

        logger.debug(f"Shifted {len(items_to_shift)} items from {collection}, {len(self.archive[collection])} in archive")

    def _extract_timestamp(self, item: Any) -> datetime | None:
        """Extract timestamp from item if available."""
        if isinstance(item, dict):
            ts = item.get("timestamp") or item.get("created_at") or item.get("started_at")
            if ts:
                if isinstance(ts, datetime):
                    return ts
                try:
                    return datetime.fromisoformat(str(ts))
                except (ValueError, TypeError):
                    pass
        elif hasattr(item, "timestamp"):
            return item.timestamp if isinstance(item.timestamp, datetime) else None
        elif hasattr(item, "created_at"):
            return item.created_at if isinstance(item.created_at, datetime) else None
        return None

    def _compress_archive(self, collection: str) -> None:
        """Compress old archive entries."""
        window = self.windows[collection]
        archive = self.archive[collection]

        target_size = int(window.archive_size * 0.9)

        archive.sort(key=lambda x: (x.importance, x.access_count), reverse=True)

        removed = archive[target_size:]
        self.archive[collection] = archive[:target_size]

        removed_item_ids = {self._get_item_id(r.item) for r in removed if self._get_item_id(r.item)}
        if collection in self.importance_scores and removed_item_ids:
            for rid in removed_item_ids:
                self.importance_scores[collection].pop(rid, None)

        removed_count = len(archive) - target_size
        if removed_count > 0:
            logger.info(f"Compressed {collection} archive, removed {removed_count} low-importance items")

    def _update_summary(self, collection: str) -> None:
        """Update summary of archived items."""
        if collection not in self.archive:
            return

        archive = self.archive[collection]
        if not archive:
            return

        self.summaries[collection] = {
            "total_items": len(archive),
            "avg_importance": sum(a.importance for a in archive) / len(archive),
            "total_access": sum(a.access_count for a in archive),
            "oldest": min(a.archived_at for a in archive).isoformat() if archive else None,
            "newest": max(a.archived_at for a in archive).isoformat() if archive else None,
        }

    def increment_importance(self, collection: str, item_id: str, delta: float = 0.1) -> None:
        """Increment importance score for an item."""
        with self._lock:
            if collection in self.importance_scores:
                current = self.importance_scores[collection].get(item_id, 0.5)
                self.importance_scores[collection][item_id] = min(1.0, current + delta)

    def recall_from_archive(self, collection: str, item_id: str) -> Any | None:
        """Recall an item from archive, promoting it to active memory."""
        with self._lock:
            if collection not in self.archive:
                return None

            for archived in self.archive[collection]:
                aid = self._get_item_id(archived.item)
                if aid == item_id:
                    archived.access_count += 1
                    archived.importance = min(1.0, archived.importance + 0.1)

                    if collection in self.active:
                        self.active[collection].append(archived.item)

                    return archived.item

            return None

    def get_archive_stats(self, collection: str) -> dict:
        """Get statistics for a collection's archive."""
        with self._lock:
            summary = self.summaries.get(collection, {})
            archive_count = len(self.archive.get(collection, []))
            active_count = len(self.active.get(collection, []))

            return {
                "collection": collection,
                "active_count": active_count,
                "archive_count": archive_count,
                "total_count": active_count + archive_count,
                "summary": summary,
                "window_config": {
                    "active_size": self.windows[collection].active_size if collection in self.windows else None,
                    "archive_size": self.windows[collection].archive_size if collection in self.windows else None,
                },
            }

    def get_all_stats(self) -> dict:
        """Get statistics for all collections."""
        with self._lock:
            names = set(list(self.active.keys()) + list(self.archive.keys()))
            windows_snapshot = dict(self.windows)
            active_snapshot = {n: len(self.active.get(n, [])) for n in names}
            archive_snapshot = {n: len(self.archive.get(n, [])) for n in names}
            summaries_snapshot = dict(self.summaries)
        
        result = {}
        for name in names:
            result[name] = {
                "collection": name,
                "active_count": active_snapshot.get(name, 0),
                "archive_count": archive_snapshot.get(name, 0),
                "total_count": active_snapshot.get(name, 0) + archive_snapshot.get(name, 0),
                "summary": summaries_snapshot.get(name, {}),
                "window_config": {
                    "active_size": windows_snapshot[name].active_size if name in windows_snapshot else None,
                    "archive_size": windows_snapshot[name].archive_size if name in windows_snapshot else None,
                },
            }
        return result

    def search_archive(self, collection: str, query: dict, limit: int = 10) -> list[Any]:
        """Search archive for items matching query."""
        with self._lock:
            if collection not in self.archive:
                return []

            results = []
            for archived in self.archive[collection]:
                if self._matches_query(archived.item, query):
                    results.append(archived.item)
                    archived.access_count += 1
                    if len(results) >= limit:
                        break

            return results

    def _matches_query(self, item: Any, query: dict) -> bool:
        """Check if item matches query criteria."""
        if not isinstance(item, dict):
            return False

        for key, value in query.items():
            if key not in item:
                return False
            if isinstance(value, list):
                if item[key] not in value:
                    return False
            elif item[key] != value:
                return False

        return True

    def get_important_items(self, collection: str, threshold: float = 0.7) -> list[Any]:
        """Get items above importance threshold from both active and archive."""
        with self._lock:
            items = []

            for item_id, importance in self.importance_scores.get(collection, {}).items():
                if importance >= threshold:
                    if collection in self.active:
                        for item in self.active[collection]:
                            if self._get_item_id(item) == item_id:
                                items.append(item)
                                break
                    for archived in self.archive.get(collection, []):
                        if self._get_item_id(archived.item) == item_id:
                            items.append(archived.item)
                            break

            return items

    def prune_low_importance(self, collection: str, threshold: float = 0.2) -> int:
        """Remove items below importance threshold."""
        with self._lock:
            if collection not in self.archive:
                return 0

            original_count = len(self.archive[collection])
            self.archive[collection] = [
                a for a in self.archive[collection]
                if a.importance >= threshold
            ]
            removed = original_count - len(self.archive[collection])

            if removed > 0:
                self._update_summary(collection)

        if removed > 0:
            self._save_state()

        return removed

    def get_evolution_context(self) -> dict:
        """Get context for evolution decisions based on memory patterns."""
        with self._lock:
            context = {
                "collections": {},
                "patterns": {},
                "recommendations": [],
            }

            for name in set(list(self.active.keys()) + list(self.archive.keys())):
                stats = self.get_archive_stats(name)
                context["collections"][name] = stats

                if stats["archive_count"] > stats["active_count"] * 2:
                    context["recommendations"].append({
                        "collection": name,
                        "type": "increase_active_window",
                        "reason": f"Archive ({stats['archive_count']}) >> active ({stats['active_count']})",
                    })

                summary = stats.get("summary", {})
                if summary.get("avg_importance", 0) < 0.3:
                    context["recommendations"].append({
                        "collection": name,
                        "type": "lower_importance_threshold",
                        "reason": f"Low avg importance: {summary.get('avg_importance', 0):.2f}",
                    })

            return context


_memory_manager: ShiftingMemoryManager | None = None
_memory_lock = threading.Lock()


def get_memory_manager() -> ShiftingMemoryManager:
    """Get global memory manager instance (thread-safe)."""
    global _memory_manager
    if _memory_manager is None:
        with _memory_lock:
            if _memory_manager is None:
                _memory_manager = ShiftingMemoryManager()
    return _memory_manager
