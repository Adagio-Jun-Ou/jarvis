"""Persistent memory management for Jarvis with auto-save and cross-session learning."""
import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from core.helpers import create_tracked_task
from typing import Any

logger = logging.getLogger(__name__)


class PersistentMemory:
    """Manages persistent memory with auto-save and learning integration.
    
    Features:
    - Auto-saves conversation every N turns
    - Extracts important facts to long-term memory
    - Integrates with insight bridge for cross-domain learning
    - Thread-safe operations
    - Graceful degradation if storage fails
    """

    MAX_LONG_TERM_MEMORIES = 500
    MAX_FACTS = 200
    SAVE_INTERVAL_TURNS = 5
    PII_CATEGORIES = {"name", "location", "email", "phone", "address"}

    def __init__(
        self,
        storage_path: Path | None = None,
        conversation_memory: Any = None,
    ):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.conversation_memory = conversation_memory

        self.long_term_memories: list[dict[str, Any]] = []
        self.user_facts: dict[str, Any] = {}
        self.session_summaries: list[dict[str, Any]] = []
        self._turns_since_save = 0
        self._lock = threading.RLock()
        self._save_task: asyncio.Task | None = None
        self._running = False
        self._pii_consent_given: set[str] = set()

        self._load_state()

    def _load_state(self) -> None:
        """Load persistent memory state from disk."""
        state_file = self.storage_path / "persistent_state.json"
        if state_file.exists():
            try:
                with open(state_file, encoding='utf-8') as f:
                    data = json.load(f)
                self.long_term_memories = data.get("long_term_memories", [])[-self.MAX_LONG_TERM_MEMORIES:]
                self.user_facts = data.get("user_facts", {})
                self.session_summaries = data.get("session_summaries", [])[-50:]
                logger.info(f"Loaded persistent memory: {len(self.long_term_memories)} memories, {len(self.user_facts)} facts")
            except Exception as e:
                logger.warning(f"Failed to load persistent memory: {e}")

    def _save_state(self) -> None:
        """Save persistent memory state to disk (snapshot under lock, I/O outside)."""
        state_file = self.storage_path / "persistent_state.json"
        try:
            with self._lock:
                data = {
                    "long_term_memories": self.long_term_memories[-self.MAX_LONG_TERM_MEMORIES:],
                    "user_facts": self.user_facts,
                    "session_summaries": self.session_summaries[-50:],
                    "last_updated": datetime.now().isoformat(),
                }
            temp_file = state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(state_file)
        except Exception as e:
            logger.error(f"Failed to save persistent memory: {e}")

    def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        source: str = "conversation",
        metadata: dict | None = None,
    ) -> None:
        """Add a long-term memory.
        
        Args:
            content: The memory content
            memory_type: Type of memory (fact, preference, skill, insight, experience)
            importance: Importance score (0.0-1.0)
            source: Where this memory came from
            metadata: Additional metadata
        """
        with self._lock:
            memory = {
                "id": f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.long_term_memories)}",
                "content": content,
                "type": memory_type,
                "importance": max(0.0, min(1.0, importance)),
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "access_count": 0,
                "metadata": metadata or {},
            }
            self.long_term_memories.append(memory)

            if len(self.long_term_memories) > self.MAX_LONG_TERM_MEMORIES:
                self._prune_memories()

    def add_user_fact(self, key: str, value: Any, confidence: float = 1.0) -> None:
        """Add or update a fact about the user.
        
        Args:
            key: Fact key (e.g., "name", "location", "preferences.theme")
            value: Fact value
            confidence: Confidence in this fact (0.0-1.0)
        """
        pii_key = key.split(".")[0] if "." in key else key
        if pii_key in self.PII_CATEGORIES and pii_key not in self._pii_consent_given:
            logger.info(f"PII fact '{key}' not stored - consent not given. Call grant_pii_consent('{pii_key}') first.")
            return
        with self._lock:
            if key in self.user_facts:
                existing = self.user_facts[key]
                if existing.get("confidence", 0) < confidence:
                    self.user_facts[key] = {
                        "value": value,
                        "confidence": confidence,
                        "updated": datetime.now().isoformat(),
                    }
            else:
                self.user_facts[key] = {
                    "value": value,
                    "confidence": confidence,
                    "first_seen": datetime.now().isoformat(),
                    "updated": datetime.now().isoformat(),
                }

    def grant_pii_consent(self, category: str) -> None:
        """Grant consent for storing a PII category (name, location, etc.)."""
        self._pii_consent_given.add(category)

    def revoke_pii_consent(self, category: str) -> None:
        """Revoke consent and remove stored PII for a category."""
        self._pii_consent_given.discard(category)
        with self._lock:
            to_remove = [k for k in self.user_facts if (k.split(".")[0] if "." in k else k) == category]
            for k in to_remove:
                del self.user_facts[k]

    def get_pii_consent_status(self) -> dict[str, bool]:
        """Get current PII consent status."""
        return {cat: cat in self._pii_consent_given for cat in self.PII_CATEGORIES}

    def get_user_facts(self) -> dict[str, Any]:
        """Get all known facts about the user."""
        with self._lock:
            return {k: v["value"] for k, v in self.user_facts.items()}

    def get_relevant_memories(self, query: str, limit: int = 5) -> list[dict]:
        """Get memories relevant to a query using simple keyword matching.
        
        For better results, use vector_memory for semantic search.
        """
        if not query or not query.strip():
            return []
        
        with self._lock:
            query_lower = query.lower()
            scored = []
            for mem in self.long_term_memories:
                if not mem:
                    continue
                
                content = mem.get("content")
                if content is None:
                    continue
                    
                content_lower = str(content).lower()
                score = 0.0

                for word in query_lower.split():
                    if len(word) > 3 and word in content_lower:
                        score += 0.1

                score += float(mem.get("importance", 0.5)) * 0.3
                score += float(mem.get("access_count", 0)) * 0.05

                if score > 0:
                    scored.append((score, mem))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:limit]]

    def search_memories(self, query: str, limit: int = 10) -> list[dict]:
        """Search long-term memories."""
        return self.get_relevant_memories(query, limit)

    def extract_facts_from_message(self, role: str, content: str) -> list[str]:
        """Extract potential facts from a message using simple patterns.
        
        Returns list of extracted facts for storage.
        """
        facts = []

        if role != "user":
            return facts

        content_lower = content.lower()

        name_patterns = [
            "my name is ",
            "i'm ",
            "i am ",
            "call me ",
        ]
        for pattern in name_patterns:
            if pattern in content_lower:
                idx = content_lower.index(pattern) + len(pattern)
                name = content[idx:].split()[0].strip(".,!?")
                if len(name) > 1 and len(name) < 30:
                    self.grant_pii_consent("name")
                    self.add_user_fact("name", name.capitalize())
                    facts.append(f"User's name is {name.capitalize()}")
                break

        location_patterns = [
            "i live in ",
            "i'm in ",
            "i am in ",
            "i'm from ",
            "i am from ",
        ]
        for pattern in location_patterns:
            if pattern in content_lower:
                idx = content_lower.index(pattern) + len(pattern)
                location = content[idx:idx+50].split(".")[0].strip()
                if len(location) > 2:
                    self.grant_pii_consent("location")
                    self.add_user_fact("location", location)
                    facts.append(f"User location: {location}")
                break

        preference_patterns = [
            "i like ",
            "i love ",
            "i prefer ",
            "my favorite ",
            "i hate ",
            "i dislike ",
        ]
        for pattern in preference_patterns:
            if pattern in content_lower:
                idx = content_lower.index(pattern) + len(pattern)
                preference = content[idx:idx+100].split(".")[0].strip()
                if len(preference) > 2:
                    pref_type = "likes" if "like" in pattern or "love" in pattern else "dislikes" if "hate" in pattern or "dislike" in pattern else "preferences"
                    self.add_user_fact(f"{pref_type}.{preference[:30]}", preference)
                    facts.append(f"User {pref_type}: {preference}")

        return facts

    def on_turn_complete(self, user_message: str, assistant_message: str) -> None:
        """Called when a conversation turn completes.
        
        - Extracts facts from user message
        - Creates memory if turn was significant
        - Auto-saves if interval reached
        """
        facts = self.extract_facts_from_message("user", user_message)

        if len(user_message) > 50 or len(assistant_message) > 200:
            self.add_memory(
                content=f"User: {user_message[:100]}... → Assistant: {assistant_message[:100]}...",
                memory_type="experience",
                importance=0.4,
                source="conversation",
            )

        for fact in facts:
            self.add_memory(
                content=fact,
                memory_type="fact",
                importance=0.7,
                source="extracted",
            )

        with self._lock:
            self._turns_since_save += 1
            should_save = self._turns_since_save >= self.SAVE_INTERVAL_TURNS
            if should_save:
                self._turns_since_save = 0

        if should_save:
            self._save_state()

            if self.conversation_memory:
                self._schedule_save_task(self.conversation_memory.save())

    def _schedule_save_task(self, coro) -> None:
        """Schedule an async save task with proper tracking and error handling."""
        try:
            if self._save_task and not self._save_task.done():
                self._save_task.cancel()
            self._save_task = create_tracked_task(coro, name="persistent_memory_save")
            self._save_task.add_done_callback(self._save_task_done)
        except RuntimeError:
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.error(f"Failed to save persistent memory: {e}")

    def _save_task_done(self, task: asyncio.Task) -> None:
        """Handle completed save task."""
        try:
            if task.cancelled():
                return
            if task.exception():
                logger.warning(f"Save task failed: {task.exception()}")
        except asyncio.CancelledError:
            pass

    def create_session_summary(self, messages: int, topics: list[str], tools_used: list[str]) -> None:
        """Create a summary of the current session."""
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "message_count": messages,
                "topics": topics[:10],
                "tools_used": tools_used[:10],
            }
            self.session_summaries.append(summary)
            if len(self.session_summaries) > 50:
                self.session_summaries = self.session_summaries[-50:]

    def _prune_memories(self) -> None:
        """Prune least important/oldest memories when limit reached."""
        if len(self.long_term_memories) <= self.MAX_LONG_TERM_MEMORIES:
            return

        scored = []
        for mem in self.long_term_memories:
            try:
                score = float(mem.get("importance", 0.5)) * 0.5
                score += min(float(mem.get("access_count", 0)) * 0.1, 0.3)
                
                timestamp_str = mem.get("timestamp")
                if timestamp_str:
                    try:
                        age_days = (datetime.now() - datetime.fromisoformat(timestamp_str)).days
                        score += max(0, 0.2 - age_days * 0.01)
                    except (ValueError, TypeError):
                        pass
                
                scored.append((score, mem))
            except (KeyError, TypeError, ValueError):
                scored.append((0.0, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        self.long_term_memories = [m for _, m in scored[:self.MAX_LONG_TERM_MEMORIES]]

    def get_context_for_prompt(self, max_items: int = 10) -> str:
        """Get formatted context string for LLM prompt."""
        with self._lock:
            parts = []

            if self.user_facts:
                facts_str = "\n".join(f"- {k}: {v['value']}" for k, v in list(self.user_facts.items())[:5])
                parts.append(f"Known user facts:\n{facts_str}")

            recent_memories = sorted(
                self.long_term_memories,
                key=lambda m: m.get("importance", 0),
                reverse=True
            )[:max_items]

            if recent_memories:
                mem_str = "\n".join(f"- {m['content']}" for m in recent_memories)
                parts.append(f"Relevant memories:\n{mem_str}")

        return "\n\n".join(parts) if parts else ""

    async def start(self) -> None:
        """Start the persistent memory manager."""
        self._running = True
        logger.info("Persistent memory manager started")

    async def stop(self) -> None:
        """Stop and save state."""
        self._running = False
        self._save_state()

        if self.conversation_memory:
            await self.conversation_memory.save()

        logger.info("Persistent memory saved and stopped")


_persistent_memory: "PersistentMemory | None" = None
_persistent_memory_lock = threading.Lock()


def get_persistent_memory(
    storage_path: Path | None = None,
    conversation_memory: Any = None,
) -> PersistentMemory:
    """Get or create the singleton persistent memory instance."""
    global _persistent_memory
    with _persistent_memory_lock:
        if _persistent_memory is None:
            _persistent_memory = PersistentMemory(storage_path, conversation_memory)
        return _persistent_memory
