"""Memory Protection System for Jarvis.

Protects important memories from:
- Accidental deletion
- Context compression
- Summarization
- Automatic cleanup

Protection levels:
- PERMANENT: Never deleted, never summarized
- HIGH: Never deleted, can be summarized but preserved in full
- MEDIUM: Can be compressed but summary preserved
- NORMAL: Standard memory lifecycle
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_memory_protection: Optional["MemoryProtection"] = None
_protection_lock = threading.Lock()


class ProtectionLevel(str, Enum):
    """Memory protection levels."""
    PERMANENT = "permanent"  # Never modified or deleted
    HIGH = "high"           # Never deleted, preserved in compression
    MEDIUM = "medium"       # Summary preserved during compression
    NORMAL = "normal"       # Standard lifecycle


@dataclass
class ProtectedMemory:
    """A protected memory entry."""
    id: str
    content: str
    memory_type: str
    protection_level: ProtectionLevel
    created_at: datetime
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "protection_level": self.protection_level.value,
            "created_at": self.created_at.isoformat(),
            "reason": self.reason,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProtectedMemory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data["memory_type"],
            protection_level=ProtectionLevel(data["protection_level"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            reason=data["reason"],
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )


@dataclass
class ProtectionRule:
    """A rule for automatic memory protection."""
    id: str
    name: str
    pattern: str  # Regex or keyword pattern
    protection_level: ProtectionLevel
    memory_types: list[str]  # Types to apply rule to
    enabled: bool = True
    priority: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pattern": self.pattern,
            "protection_level": self.protection_level.value,
            "memory_types": self.memory_types,
            "enabled": self.enabled,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProtectionRule":
        return cls(
            id=data["id"],
            name=data["name"],
            pattern=data["pattern"],
            protection_level=ProtectionLevel(data["protection_level"]),
            memory_types=data["memory_types"],
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0),
        )


class MemoryProtection:
    MAX_RULES = 100
    """Manages memory protection rules and protected memories.
    
    Features:
    - Explicit protection by ID
    - Automatic protection based on rules
    - Protection during compression/summarization
    - Integration with ConversationMemory and UnifiedMemory
    """
    
    DEFAULT_RULES = [
        ProtectionRule(
            id="rule_user_preferences",
            name="User Preferences",
            pattern=r"\b(my name is|i am|i'm|i like|i prefer|my favorite|remember that)\b",
            protection_level=ProtectionLevel.HIGH,
            memory_types=["conversation", "preference"],
            priority=100,
        ),
        ProtectionRule(
            id="rule_important_facts",
            name="Important Facts",
            pattern=r"\b(important|don'?t forget|never forget|must remember|critical|essential)\b",
            protection_level=ProtectionLevel.HIGH,
            memory_types=["conversation", "fact"],
            priority=90,
        ),
        ProtectionRule(
            id="rule_credentials",
            name="Credentials",
            pattern=r"\b(password|api key|secret|token|credential|authenticate)\b",
            protection_level=ProtectionLevel.PERMANENT,
            memory_types=["credential", "secret"],
            priority=100,
        ),
        ProtectionRule(
            id="rule_decisions",
            name="Decisions",
            pattern=r"\b(decided|decision|i choose|my choice|we agreed|conclusion)\b",
            protection_level=ProtectionLevel.HIGH,
            memory_types=["decision"],
            priority=85,
        ),
        ProtectionRule(
            id="rule_todos",
            name="Tasks and Todos",
            pattern=r"\b(todo|task|deadline|due|must do|need to|should do)\b",
            protection_level=ProtectionLevel.MEDIUM,
            memory_types=["task", "todo"],
            priority=70,
        ),
    ]
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "memory_protection"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._protected: dict[str, ProtectedMemory] = {}
        self._rules: list[ProtectionRule] = list(self.DEFAULT_RULES)
        self._lock = threading.RLock()
        
        self._load()
        
        logger.info(f"MemoryProtection initialized: {len(self._protected)} protected, {len(self._rules)} rules")
    
    def _load(self) -> None:
        """Load protected memories and rules from disk."""
        protected_file = self.storage_path / "protected.json"
        if protected_file.exists():
            try:
                data = json.loads(protected_file.read_text(encoding='utf-8'))
                self._protected = {
                    p["id"]: ProtectedMemory.from_dict(p)
                    for p in data.get("protected", [])
                }
            except Exception as e:
                logger.warning(f"Failed to load protected memories: {e}")
        
        rules_file = self.storage_path / "rules.json"
        if rules_file.exists():
            try:
                data = json.loads(rules_file.read_text(encoding='utf-8'))
                custom_rules = [ProtectionRule.from_dict(r) for r in data.get("rules", [])]
                self._rules = list(self.DEFAULT_RULES) + custom_rules
            except Exception as e:
                logger.warning(f"Failed to load protection rules: {e}")
    
    def _save(self) -> None:
        """Save protected memories and rules to disk."""
        protected_file = self.storage_path / "protected.json"
        try:
            data = {
                "protected": [p.to_dict() for p in self._protected.values()],
                "updated": datetime.now().isoformat(),
            }
            temp_file = protected_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            temp_file.replace(protected_file)
        except Exception as e:
            logger.error(f"Failed to save protected memories: {e}")
    
    def protect(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        protection_level: ProtectionLevel,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> ProtectedMemory:
        """Protect a memory.
        
        Args:
            memory_id: Unique ID of the memory
            content: The memory content
            memory_type: Type of memory (conversation, fact, preference, etc.)
            protection_level: Level of protection
            reason: Why this memory is being protected
            metadata: Additional metadata
            
        Returns:
            The ProtectedMemory entry
        """
        with self._lock:
            protected = ProtectedMemory(
                id=memory_id,
                content=content[:1000],  # Store first 1000 chars
                memory_type=memory_type,
                protection_level=protection_level,
                created_at=datetime.now(),
                reason=reason,
                metadata=metadata or {},
            )
            
            self._protected[memory_id] = protected
            self._save()
            
            logger.info(f"Protected memory {memory_id}: {protection_level.value} - {reason}")
            return protected
    
    def unprotect(self, memory_id: str) -> bool:
        """Remove protection from a memory."""
        with self._lock:
            if memory_id in self._protected:
                del self._protected[memory_id]
                self._save()
                logger.info(f"Unprotected memory {memory_id}")
                return True
            return False
    
    def is_protected(self, memory_id: str) -> bool:
        """Check if a memory is protected."""
        return memory_id in self._protected
    
    def get_protection_level(self, memory_id: str) -> Optional[ProtectionLevel]:
        """Get protection level for a memory."""
        with self._lock:
            if memory_id in self._protected:
                return self._protected[memory_id].protection_level
            return None
    
    def check_rules(
        self,
        content: str,
        memory_type: str,
    ) -> Optional[ProtectionLevel]:
        """Check content against protection rules.
        
        Args:
            content: Memory content to check
            memory_type: Type of memory
            
        Returns:
            ProtectionLevel if rule matches, None otherwise
        """
        import re
        
        content_lower = content.lower()
        applicable_rules = []
        
        for rule in self._rules:
            if not rule.enabled:
                continue
            if memory_type not in rule.memory_types and "all" not in rule.memory_types:
                continue
            
            try:
                if re.search(rule.pattern, content_lower, re.IGNORECASE):
                    applicable_rules.append(rule)
            except re.error:
                if rule.pattern.lower() in content_lower:
                    applicable_rules.append(rule)
        
        if applicable_rules:
            applicable_rules.sort(key=lambda r: r.priority, reverse=True)
            return applicable_rules[0].protection_level
        
        return None
    
    def auto_protect(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        importance: float = 0.5,
    ) -> Optional[ProtectedMemory]:
        """Automatically protect memory based on rules and importance.
        
        Args:
            memory_id: Unique memory ID
            content: Memory content
            memory_type: Type of memory
            importance: Calculated importance score (0-1)
            
        Returns:
            ProtectedMemory if protection applied, None otherwise
        """
        protection_level = self.check_rules(content, memory_type)
        
        if protection_level is None and importance >= 0.9:
            protection_level = ProtectionLevel.HIGH
        elif protection_level is None and importance >= 0.7:
            protection_level = ProtectionLevel.MEDIUM
        
        if protection_level:
            return self.protect(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                protection_level=protection_level,
                reason="auto_protected",
                metadata={"importance": importance},
            )
        
        return None
    
    def filter_for_compression(
        self,
        memory_ids: list[str],
        contents: list[str],
    ) -> tuple[list[str], list[str]]:
        """Filter out protected memories from compression candidates.
        
        Args:
            memory_ids: List of candidate memory IDs
            contents: Corresponding contents
            
        Returns:
            Tuple of (compressible_ids, compressible_contents)
        """
        with self._lock:
            compressible_ids = []
            compressible_contents = []
            
            for mid, content in zip(memory_ids, contents):
                if mid not in self._protected:
                    compressible_ids.append(mid)
                    compressible_contents.append(content)
                elif self._protected[mid].protection_level == ProtectionLevel.MEDIUM:
                    compressible_ids.append(mid)
                    compressible_contents.append(content)
            
            return compressible_ids, compressible_contents
    
    def filter_for_deletion(
        self,
        memory_ids: list[str],
    ) -> list[str]:
        """Filter out protected memories from deletion candidates.
        
        Args:
            memory_ids: List of candidate memory IDs
            
        Returns:
            List of deletable memory IDs
        """
        with self._lock:
            deletable = []
            for mid in memory_ids:
                if mid not in self._protected:
                    deletable.append(mid)
                elif self._protected[mid].protection_level == ProtectionLevel.NORMAL:
                    deletable.append(mid)
            return deletable
    
    def add_rule(
        self,
        name: str,
        pattern: str,
        protection_level: ProtectionLevel,
        memory_types: list[str],
        priority: int = 50,
    ) -> ProtectionRule:
        """Add a custom protection rule."""
        import uuid
        
        with self._lock:
            rule = ProtectionRule(
                id=f"rule_{uuid.uuid4().hex[:8]}",
                name=name,
                pattern=pattern,
                protection_level=protection_level,
                memory_types=memory_types,
                priority=priority,
            )
            if len(self._rules) >= self.MAX_RULES:
                return None
            self._rules.append(rule)
            self._save()
            return rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a custom protection rule."""
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.id == rule_id and not rule.id.startswith("rule_"):
                    del self._rules[i]
                    self._save()
                    return True
            return False
    
    def get_protected_memories(self, level: Optional[ProtectionLevel] = None) -> list[ProtectedMemory]:
        """Get all protected memories, optionally filtered by level."""
        with self._lock:
            memories = list(self._protected.values())
            if level:
                memories = [m for m in memories if m.protection_level == level]
            return sorted(memories, key=lambda m: m.created_at, reverse=True)
    
    def get_rules(self, include_defaults: bool = True) -> list[ProtectionRule]:
        """Get protection rules."""
        with self._lock:
            if include_defaults:
                return list(self._rules)
            return [r for r in self._rules if not r.id.startswith("rule_user") and not r.id.startswith("rule_important")]
    
    def record_access(self, memory_id: str) -> None:
        """Record access to a protected memory."""
        with self._lock:
            if memory_id in self._protected:
                self._protected[memory_id].access_count += 1
                self._protected[memory_id].last_accessed = datetime.now()
                self._save()
    
    def get_stats(self) -> dict:
        """Get protection statistics."""
        with self._lock:
            by_level = {}
            for level in ProtectionLevel:
                by_level[level.value] = len([m for m in self._protected.values() if m.protection_level == level])
            
            total_access = sum(m.access_count for m in self._protected.values())
            
            return {
                "total_protected": len(self._protected),
                "by_level": by_level,
                "total_rules": len(self._rules),
                "enabled_rules": len([r for r in self._rules if r.enabled]),
                "total_access_count": total_access,
            }


def get_memory_protection() -> MemoryProtection:
    """Get the global MemoryProtection instance."""
    global _memory_protection, _protection_lock
    if _memory_protection is None:
        with _protection_lock:
            if _memory_protection is None:
                _memory_protection = MemoryProtection()
    return _memory_protection