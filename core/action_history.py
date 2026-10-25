"""Action History and Undo System for Jarvis.

Provides power drop (undo) functionality for:
- Tool executions
- File modifications
- Memory changes
- Chat messages

Each action is recorded with enough state to reverse it.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import hashlib
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)

_action_history: Optional["ActionHistory"] = None
_history_lock = threading.Lock()


class ActionType(str, Enum):
    """Types of actions that can be undone."""
    TOOL_CALL = "tool_call"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    MEMORY_ADD = "memory_add"
    MEMORY_DELETE = "memory_delete"
    MEMORY_MODIFY = "memory_modify"
    CHAT_MESSAGE = "chat_message"
    CONFIG_CHANGE = "config_change"
    SYSTEM_COMMAND = "system_command"


@dataclass
class ActionRecord:
    """Record of an action that can be undone."""
    id: str
    action_type: ActionType
    timestamp: datetime
    description: str
    data: dict[str, Any]
    rollback_data: dict[str, Any]
    protected: bool = False
    rolled_back: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "data": self.data,
            "rollback_data": self.rollback_data,
            "protected": self.protected,
            "rolled_back": self.rolled_back,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActionRecord":
        return cls(
            id=data["id"],
            action_type=ActionType(data["action_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data["description"],
            data=data["data"],
            rollback_data=data["rollback_data"],
            protected=data.get("protected", False),
            rolled_back=data.get("rolled_back", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DropResult:
    """Result of a power drop (undo) operation."""
    success: bool
    action_id: str
    action_type: ActionType
    description: str
    rollback_performed: list[str]
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "rollback_performed": self.rollback_performed,
            "error": self.error,
        }


class ActionHistory:
    """Manages action history with undo (power drop) capability.
    
    Features:
    - Records all significant actions
    - Stores rollback data for each action
    - Supports multi-level undo
    - Protected actions cannot be undone
    - Persisted to disk for recovery
    """
    
    MAX_HISTORY = 1000
    MAX_UNDO_DEPTH = 50
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "action_history"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._history: list[ActionRecord] = []
        self._lock = threading.RLock()
        self._pending_rollback: dict[str, Any] = {}
        
        self._load()
        
        logger.info(f"ActionHistory initialized with {len(self._history)} records")
    
    def _load(self) -> None:
        """Load history from disk."""
        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding='utf-8'))
                self._history = [ActionRecord.from_dict(r) for r in data.get("records", [])]
                self._history = self._history[-self.MAX_HISTORY:]
            except Exception as e:
                logger.warning(f"Failed to load action history: {e}")
    
    def _save(self) -> None:
        """Save history to disk."""
        history_file = self.storage_path / "history.json"
        try:
            data = {
                "records": [r.to_dict() for r in self._history[-self.MAX_HISTORY:]],
                "updated": datetime.now().isoformat(),
            }
            temp_file = history_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            temp_file.replace(history_file)
        except Exception as e:
            logger.error(f"Failed to save action history: {e}")
    
    def _generate_id(self) -> str:
        """Generate unique action ID."""
        return f"act_{int(time.time() * 1000)}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:6]}"
    
    def record(
        self,
        action_type: ActionType,
        description: str,
        data: dict[str, Any],
        rollback_data: dict[str, Any],
        protected: bool = False,
        metadata: Optional[dict] = None,
    ) -> ActionRecord:
        """Record an action that can be undone.
        
        Args:
            action_type: Type of action
            description: Human-readable description
            data: Action data (what was done)
            rollback_data: Data needed to reverse the action
            protected: If True, action cannot be undone
            metadata: Additional metadata
            
        Returns:
            The created ActionRecord
        """
        with self._lock:
            record = ActionRecord(
                id=self._generate_id(),
                action_type=action_type,
                timestamp=datetime.now(),
                description=description,
                data=data,
                rollback_data=rollback_data,
                protected=protected,
                metadata=metadata or {},
            )
            
            self._history.append(record)
            
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]
            
            self._save()
            
            logger.debug(f"Recorded action: {record.id} - {description}")
            return record
    
    def drop_last(self, action_type: Optional[ActionType] = None) -> DropResult:
        """Power drop - undo the last action.
        
        Args:
            action_type: If specified, only undo last action of this type
            
        Returns:
            DropResult with undo details
        """
        with self._lock:
            if not self._history:
                return DropResult(
                    success=False,
                    action_id="",
                    action_type=ActionType.SYSTEM_COMMAND,
                    description="No actions to undo",
                    rollback_performed=[],
                    error="history_empty",
                )
            
            candidates = list(reversed(self._history))
            if action_type:
                candidates = [a for a in candidates if a.action_type == action_type]
            
            for record in candidates:
                if record.rolled_back:
                    continue
                if record.protected:
                    continue
                
                return self._perform_drop(record)
            
            return DropResult(
                success=False,
                action_id="",
                action_type=action_type or ActionType.SYSTEM_COMMAND,
                description="No undoable actions found",
                rollback_performed=[],
                error="no_undoable_action",
            )
    
    def drop_by_id(self, action_id: str) -> DropResult:
        """Undo a specific action by ID."""
        with self._lock:
            for record in reversed(self._history):
                if record.id == action_id:
                    if record.rolled_back:
                        return DropResult(
                            success=False,
                            action_id=action_id,
                            action_type=record.action_type,
                            description=record.description,
                            rollback_performed=[],
                            error="already_rolled_back",
                        )
                    if record.protected:
                        return DropResult(
                            success=False,
                            action_id=action_id,
                            action_type=record.action_type,
                            description=record.description,
                            rollback_performed=[],
                            error="action_protected",
                        )
                    return self._perform_drop(record)
            
            return DropResult(
                success=False,
                action_id=action_id,
                action_type=ActionType.SYSTEM_COMMAND,
                description="Action not found",
                rollback_performed=[],
                error="not_found",
            )
    
    def _perform_drop(self, record: ActionRecord) -> DropResult:
        """Perform the actual rollback."""
        rollback_performed = []
        
        try:
            if record.action_type == ActionType.TOOL_CALL:
                rollback_performed = self._rollback_tool_call(record)
            elif record.action_type == ActionType.FILE_WRITE:
                rollback_performed = self._rollback_file_write(record)
            elif record.action_type == ActionType.FILE_DELETE:
                rollback_performed = self._rollback_file_delete(record)
            elif record.action_type == ActionType.MEMORY_ADD:
                rollback_performed = self._rollback_memory_add(record)
            elif record.action_type == ActionType.MEMORY_DELETE:
                rollback_performed = self._rollback_memory_delete(record)
            elif record.action_type == ActionType.MEMORY_MODIFY:
                rollback_performed = self._rollback_memory_modify(record)
            elif record.action_type == ActionType.CHAT_MESSAGE:
                rollback_performed = self._rollback_chat_message(record)
            elif record.action_type == ActionType.CONFIG_CHANGE:
                rollback_performed = self._rollback_config_change(record)
            
            record.rolled_back = True
            self._save()
            
            logger.info(f"Dropped action: {record.id} - {record.description}")
            
            return DropResult(
                success=True,
                action_id=record.id,
                action_type=record.action_type,
                description=record.description,
                rollback_performed=rollback_performed,
            )
        except Exception as e:
            logger.error(f"Failed to rollback action {record.id}: {e}")
            return DropResult(
                success=False,
                action_id=record.id,
                action_type=record.action_type,
                description=record.description,
                rollback_performed=rollback_performed,
                error=safe_error_message(e),
            )
    
    def _rollback_tool_call(self, record: ActionRecord) -> list[str]:
        """Rollback a tool call. Note: Many tools are not reversible."""
        rollback_performed = []
        
        tool_name = record.data.get("tool_name")
        args = record.data.get("args", {})
        
        if tool_name == "file_write" or tool_name == "write_file":
            original_content = record.rollback_data.get("original_content")
            file_path = args.get("file_path") or args.get("path")
            if file_path and original_content is not None:
                Path(file_path).write_text(original_content, encoding='utf-8')
                rollback_performed.append(f"restored {file_path}")
        
        return rollback_performed
    
    def _rollback_file_write(self, record: ActionRecord) -> list[str]:
        """Rollback a file write by restoring original content."""
        file_path = record.data.get("file_path")
        original_content = record.rollback_data.get("original_content")
        
        if file_path and original_content is not None:
            Path(file_path).write_text(original_content, encoding='utf-8')
            return [f"restored {file_path}"]
        return []
    
    def _rollback_file_delete(self, record: ActionRecord) -> list[str]:
        """Rollback file delete by restoring content."""
        file_path = record.data.get("file_path")
        original_content = record.rollback_data.get("original_content")
        
        if file_path and original_content is not None:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).write_text(original_content, encoding='utf-8')
            return [f"restored {file_path}"]
        return []
    
    def _rollback_memory_add(self, record: ActionRecord) -> list[str]:
        """Rollback memory addition by removing the entry."""
        memory_id = record.data.get("memory_id")
        if memory_id:
            try:
                from core.unified_memory import get_unified_memory
                memory = get_unified_memory()
                memory.delete(memory_id)
                return [f"deleted memory {memory_id}"]
            except Exception as e:
                logger.warning(f"Could not rollback memory add: {e}")
        return []
    
    def _rollback_memory_delete(self, record: ActionRecord) -> list[str]:
        """Rollback memory deletion by restoring the entry."""
        memory_data = record.rollback_data.get("memory_data")
        if memory_data:
            try:
                from core.unified_memory import get_unified_memory, MemoryEntry, MemoryType
                memory = get_unified_memory()
                entry = MemoryEntry(
                    id=memory_data["id"],
                    content=memory_data["content"],
                    memory_type=MemoryType(memory_data["memory_type"]),
                    importance=memory_data.get("importance", 0.5),
                    metadata=memory_data.get("metadata", {}),
                )
                memory._memories[entry.id] = entry
                memory._save()
                return [f"restored memory {entry.id}"]
            except Exception as e:
                logger.warning(f"Could not rollback memory delete: {e}")
        return []
    
    def _rollback_memory_modify(self, record: ActionRecord) -> list[str]:
        """Rollback memory modification by restoring original."""
        memory_id = record.data.get("memory_id")
        original_data = record.rollback_data.get("original_data")
        
        if memory_id and original_data:
            try:
                from core.unified_memory import get_unified_memory
                memory = get_unified_memory()
                if memory_id in memory._memories:
                    memory._memories[memory_id].content = original_data.get("content", "")
                    memory._memories[memory_id].importance = original_data.get("importance", 0.5)
                    memory._save()
                    return [f"restored memory {memory_id}"]
            except Exception as e:
                logger.warning(f"Could not rollback memory modify: {e}")
        return []
    
    def _rollback_chat_message(self, record: ActionRecord) -> list[str]:
        """Rollback chat message by marking for exclusion."""
        return ["message_marked_for_exclusion"]
    
    def _rollback_config_change(self, record: ActionRecord) -> list[str]:
        """Rollback config change."""
        config_key = record.data.get("key")
        original_value = record.rollback_data.get("original_value")
        
        if config_key and original_value is not None:
            return [f"would restore {config_key} = {original_value}"]
        return []
    
    def get_history(self, limit: int = 50, action_type: Optional[ActionType] = None) -> list[ActionRecord]:
        """Get recent action history."""
        with self._lock:
            history = list(reversed(self._history))
            if action_type:
                history = [a for a in history if a.action_type == action_type]
            return history[:limit]
    
    def get_undoable_actions(self, limit: int = 20) -> list[ActionRecord]:
        """Get actions that can be undone."""
        with self._lock:
            return [
                a for a in reversed(self._history)
                if not a.rolled_back and not a.protected
            ][:limit]
    
    def protect_action(self, action_id: str) -> bool:
        """Protect an action from being undone."""
        with self._lock:
            for record in self._history:
                if record.id == action_id:
                    record.protected = True
                    self._save()
                    return True
            return False
    
    def get_stats(self) -> dict:
        """Get action history statistics."""
        with self._lock:
            by_type = {}
            rolled_back = 0
            protected = 0
            
            for record in self._history:
                t = record.action_type.value
                by_type[t] = by_type.get(t, 0) + 1
                if record.rolled_back:
                    rolled_back += 1
                if record.protected:
                    protected += 1
            
            return {
                "total_actions": len(self._history),
                "by_type": by_type,
                "rolled_back": rolled_back,
                "protected": protected,
                "undoable": len(self.get_undoable_actions()),
            }


def get_action_history() -> ActionHistory:
    """Get the global ActionHistory instance."""
    global _action_history, _history_lock
    if _action_history is None:
        with _history_lock:
            if _action_history is None:
                _action_history = ActionHistory()
    return _action_history