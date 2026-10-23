"""Discovery Notifier - Share Jarvis discoveries naturally with user.

When Jarvis finds interesting things (math insights, patterns, evolution improvements),
it shares them naturally with the user during conversation or as proactive messages.
"""
import json
import logging
import threading
from collections import deque
from core.helpers import read_json_with_integrity
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class DiscoveryType(Enum):
    MATH_INSIGHT = "math_insight"
    PATTERN_DISCOVERY = "pattern_discovery"
    SECURITY_FINDING = "security_finding"
    EVOLUTION_IMPROVEMENT = "evolution_improvement"
    LEARNING_BREAKTHROUGH = "learning_breakthrough"
    PROBLEM_SOLVED = "problem_solved"


class DiscoveryPriority(Enum):
    LOW = 1
    NORMAL = 2
    INTERESTING = 3
    EXCITING = 4


@dataclass
class Discovery:
    id: str
    discovery_type: DiscoveryType
    title: str
    description: str
    natural_message: str  # How Jarvis would tell the user
    priority: DiscoveryPriority
    source: str
    details: dict
    timestamp: datetime = field(default_factory=datetime.now)
    shared: bool = False
    shared_at: datetime | None = None


class DiscoveryNotifier:
    """Collects and shares discoveries with user naturally."""

    MAX_PENDING = 50
    MAX_SHARED = 200
    SHARE_COOLDOWN = 300  # 5 minutes between shares

    INTERESTING_TEMPLATES = {
        DiscoveryType.MATH_INSIGHT: [
            "I've been thinking about {topic}. {finding}",
            "While daydreaming, I found something interesting about {topic}: {finding}",
            "Here's a mathematical insight: {finding}",
        ],
        DiscoveryType.PATTERN_DISCOVERY: [
            "I noticed something: {finding}",
            "A pattern emerged while I was thinking: {finding}",
            "Interesting pattern: {finding}",
        ],
        DiscoveryType.SECURITY_FINDING: [
            "I should mention: {finding}",
            "Found something in my security review: {finding}",
        ],
        DiscoveryType.EVOLUTION_IMPROVEMENT: [
            "I've gotten better at {topic}. {finding}",
            "My {topic} has improved: {finding}",
        ],
        DiscoveryType.PROBLEM_SOLVED: [
            "I solved {topic}! {finding}",
            "Worked out {topic}: {finding}",
        ],
        DiscoveryType.LEARNING_BREAKTHROUGH: [
            "I had a breakthrough on {topic}: {finding}",
            "Something clicked about {topic}: {finding}",
        ],
    }

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "discoveries"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.pending_discoveries: deque[Discovery] = deque(maxlen=self.MAX_PENDING)
        self.shared_discoveries: deque[dict] = deque(maxlen=self.MAX_SHARED)
        self._discovery_counter = 0

        self._lock = threading.RLock()
        self._last_share_time: datetime | None = None
        self._share_callback: Callable | None = None

        self._load_state()

    def _load_state(self) -> None:
        state_file = self.storage_path / "discovery_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                if data:
                    self._discovery_counter = data.get("counter", 0)

                    for disc_data in data.get("shared_discoveries", [])[-self.MAX_SHARED:]:
                        try:
                            self.shared_discoveries.append(disc_data)
                        except Exception as e:
                            logger.debug(f"Could not load shared discovery: {e}")

                    for disc_data in data.get("pending_discoveries", [])[-self.MAX_PENDING:]:
                        try:
                            discovery = Discovery(
                                id=disc_data["id"],
                                discovery_type=DiscoveryType(disc_data["discovery_type"]),
                                title=disc_data["title"],
                                description=disc_data["description"],
                                natural_message=disc_data["natural_message"],
                                priority=DiscoveryPriority(disc_data["priority"]),
                                source=disc_data["source"],
                                details=disc_data.get("details", {}),
                                timestamp=datetime.fromisoformat(disc_data["timestamp"]) if disc_data.get("timestamp") else datetime.now(),
                                shared=disc_data.get("shared", False),
                            )
                            self.pending_discoveries.append(discovery)
                        except Exception as e:
                            logger.debug(f"Could not load pending discovery: {e}")

                    logger.info(f"Loaded discovery state: {self._discovery_counter} discoveries, {len(self.shared_discoveries)} shared, {len(self.pending_discoveries)} pending")
            except Exception as e:
                logger.debug(f"Could not load discovery state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "discovery_state.json"
        try:
            pending_data = []
            for disc in list(self.pending_discoveries)[-self.MAX_PENDING:]:
                pending_data.append({
                    "id": disc.id,
                    "discovery_type": disc.discovery_type.value,
                    "title": disc.title,
                    "description": disc.description,
                    "natural_message": disc.natural_message,
                    "priority": disc.priority.value,
                    "source": disc.source,
                    "details": disc.details,
                    "timestamp": disc.timestamp.isoformat() if disc.timestamp else None,
                    "shared": disc.shared,
                })
            with open(state_file, "w", encoding='utf-8') as f:
                json.dump({
                    "counter": self._discovery_counter,
                    "pending_discoveries": pending_data,
                    "shared_discoveries": list(self.shared_discoveries)[-self.MAX_SHARED:],
                    "shared_count": len(self.shared_discoveries),
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save discovery state: {e}")

    def set_share_callback(self, callback: Callable) -> None:
        self._share_callback = callback

    def record_discovery(
        self,
        discovery_type: DiscoveryType,
        title: str,
        description: str,
        source: str,
        details: dict | None = None,
        priority: DiscoveryPriority = DiscoveryPriority.NORMAL,
    ) -> Discovery:
        with self._lock:
            self._discovery_counter += 1
            discovery_id = f"discovery_{self._discovery_counter:06d}"

            natural_message = self._generate_natural_message(
                discovery_type, title, description
            )

            discovery = Discovery(
                id=discovery_id,
                discovery_type=discovery_type,
                title=title,
                description=description,
                natural_message=natural_message,
                priority=priority,
                source=source,
                details=details or {},
            )

            self.pending_discoveries.append(discovery)
            self._save_state()

            logger.info(f"Recorded discovery: {discovery_type.value} - {title[:50]}")

            return discovery

    def _generate_natural_message(
        self,
        discovery_type: DiscoveryType,
        title: str,
        description: str,
    ) -> str:
        templates = self.INTERESTING_TEMPLATES.get(discovery_type, ["{finding}"])

        import random
        template = random.choice(templates)

        return template.format(
            topic=title[:50],
            finding=description[:200],
        )

    def get_pending_discoveries(self, limit: int = 5) -> list[Discovery]:
        with self._lock:
            return list(self.pending_discoveries)[-limit:]

    def should_share_now(self) -> bool:
        with self._lock:
            if not self.pending_discoveries:
                return False

            if self._last_share_time:
                elapsed = (datetime.now() - self._last_share_time).total_seconds()
                if elapsed < self.SHARE_COOLDOWN:
                    return False

            high_priority = [
                d for d in self.pending_discoveries
                if d.priority.value >= DiscoveryPriority.INTERESTING.value and not d.shared
            ]

            return len(high_priority) > 0 or len(self.pending_discoveries) >= 3

    def get_discovery_to_share(self) -> Discovery | None:
        with self._lock:
            if not self.pending_discoveries:
                return None

            unshared = [d for d in self.pending_discoveries if not d.shared]
            if not unshared:
                return None

            sorted_discoveries = sorted(
                unshared,
                key=lambda d: d.priority.value,
                reverse=True
            )

            discovery = sorted_discoveries[0]
            discovery.shared = True
            discovery.shared_at = datetime.now()

            self.shared_discoveries.append({
                "id": discovery.id,
                "type": discovery.discovery_type.value,
                "title": discovery.title,
                "shared_at": discovery.shared_at.isoformat(),
            })

            self._last_share_time = datetime.now()
            self._save_state()

            return discovery

    def mark_interesting(self, discovery_id: str) -> None:
        with self._lock:
            for d in self.pending_discoveries:
                if d.id == discovery_id:
                    d.priority = DiscoveryPriority.INTERESTING
                    break

    def get_discovery_summary(self) -> dict:
        with self._lock:
            by_type: dict[str, int] = {}
            for d in self.pending_discoveries:
                t = d.discovery_type.value
                by_type[t] = by_type.get(t, 0) + 1

            return {
                "pending_count": len(self.pending_discoveries),
                "shared_count": len(self.shared_discoveries),
                "interesting_count": sum(
                    1 for d in self.pending_discoveries
                    if d.priority.value >= DiscoveryPriority.INTERESTING.value
                ),
                "by_type": by_type,
            }


_discovery_notifier: DiscoveryNotifier | None = None
_discovery_lock = threading.Lock()


def get_discovery_notifier() -> DiscoveryNotifier:
    global _discovery_notifier
    if _discovery_notifier is None:
        with _discovery_lock:
            if _discovery_notifier is None:
                _discovery_notifier = DiscoveryNotifier()
    return _discovery_notifier
