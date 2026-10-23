"""Unified Permission Coordinator for Jarvis.

Consolidates all permission systems into a single unified interface:
- Tool permissions
- Checkpoint permissions (frontier domains)
- Self-edit permissions (code modifications)
- Evolution permissions (gene changes)
- Refine permissions (code refinements)
- Free-time permissions (sandbox exploration)

Provides:
- Single pending queue for all types
- Unified API endpoint
- WebSocket notifications
- Consistent approval/deny flow
"""
import asyncio
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List
from collections import deque
import uuid

logger = logging.getLogger(__name__)


class PermissionType(Enum):
    TOOL = "tool"
    CHECKPOINT = "checkpoint"
    SELF_EDIT = "self_edit"
    EVOLUTION = "evolution"
    REFINE = "refine"
    FREE_TIME = "free_time"
    TERMINAL = "terminal"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PendingPermission:
    """A pending permission request from any subsystem."""
    id: str
    permission_type: PermissionType
    source: str  # Which subsystem created this
    operation: str  # Human-readable description
    risk_level: RiskLevel
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    impact_preview: str = ""
    recommendations: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.permission_type.value,
            "source": self.source,
            "operation": self.operation,
            "risk_level": self.risk_level.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "context": self.context,
            "impact_preview": self.impact_preview,
            "recommendations": self.recommendations,
        }
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class PermissionResponse:
    """Response to a permission request."""
    permission_id: str
    approved: bool
    reason: str = ""
    responded_at: datetime = field(default_factory=datetime.now)
    responded_by: str = "user"


class PermissionCoordinator:
    """Unified permission coordinator for all subsystems.
    
    Thread-safe singleton that coordinates all permission requests
    from various Jarvis subsystems.
    
    Usage:
        coordinator = get_permission_coordinator()
        
        # Request permission
        pending = coordinator.request(
            type=PermissionType.TERMINAL,
            source="terminal.py",
            operation="Execute: rm -rf /tmp/test",
            risk_level=RiskLevel.HIGH,
            context={"command": "rm -rf /tmp/test"},
        )
        
        # User approves/denies via UI
        coordinator.respond(pending.id, approved=True)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    DEFAULT_EXPIRY_MINUTES = 5
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "permissions_coordinator"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._pending: Dict[str, PendingPermission] = {}
        self._history: deque = deque(maxlen=1000)
        self._waiting: Dict[str, asyncio.Event] = {}
        self._results: Dict[str, PermissionResponse] = {}
        self._callbacks: List[Callable] = []
        self._ws_callbacks: List[Callable] = []
        
        self._stats = {
            "total_requests": 0,
            "approved": 0,
            "denied": 0,
            "expired": 0,
            "by_type": {t.value: 0 for t in PermissionType},
            "by_source": {},
        }
        
        self._load_state()
    
    @classmethod
    def get_instance(cls) -> "PermissionCoordinator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _load_state(self) -> None:
        """Load persisted state."""
        state_file = self.storage_path / "coordinator_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._stats = data.get("stats", self._stats)
            except Exception as e:
                logger.debug(f"Could not load coordinator state: {e}")
    
    def _save_state(self) -> None:
        """Persist state."""
        state_file = self.storage_path / "coordinator_state.json"
        try:
            atomic_write_json(state_file, {"stats": self._stats})
        except Exception as e:
            logger.debug(f"Could not save coordinator state: {e}")
    
    def request(
        self,
        type: PermissionType,
        source: str,
        operation: str,
        risk_level: RiskLevel,
        context: Dict[str, Any] = None,
        impact_preview: str = "",
        recommendations: List[str] = None,
        expiry_minutes: float = None,
        callback: Callable = None,
    ) -> PendingPermission:
        """Request a permission.
        
        Args:
            type: Type of permission being requested
            source: Subsystem making the request
            operation: Human-readable description
            risk_level: Risk assessment
            context: Additional context data
            impact_preview: Preview of what will happen
            recommendations: Recommended actions
            expiry_minutes: Minutes until expiry (default 5)
            callback: Async callback to receive result
        
        Returns:
            PendingPermission with ID
        """
        permission_id = str(uuid.uuid4())[:8]
        
        expires_at = None
        if expiry_minutes or risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            expires_at = datetime.now() + timedelta(
                minutes=expiry_minutes or self.DEFAULT_EXPIRY_MINUTES
            )
        
        pending = PendingPermission(
            id=permission_id,
            permission_type=type,
            source=source,
            operation=operation,
            risk_level=risk_level,
            context=context or {},
            impact_preview=impact_preview,
            recommendations=recommendations or [],
            expires_at=expires_at,
            callback=callback,
        )
        
        with self._lock:
            self._pending[permission_id] = pending
            self._stats["total_requests"] += 1
            self._stats["by_type"][type.value] = self._stats["by_type"].get(type.value, 0) + 1
            self._stats["by_source"][source] = self._stats["by_source"].get(source, 0) + 1
        
        self._notify_callbacks("request", pending)
        self._save_state()
        
        logger.info(f"Permission requested: {permission_id} [{type.value}] {source}: {operation[:50]}")
        
        return pending
    
    def respond(
        self,
        permission_id: str,
        approved: bool,
        reason: str = "",
        responded_by: str = "user",
    ) -> bool:
        """Respond to a pending permission.
        
        Args:
            permission_id: ID of the pending permission
            approved: Whether to approve
            reason: Reason for the decision
            responded_by: Who made the decision
        
        Returns:
            True if response was recorded, False if not found
        """
        with self._lock:
            pending = self._pending.pop(permission_id, None)
            if pending is None:
                logger.warning(f"Permission not found: {permission_id}")
                return False
            
            if pending.is_expired():
                self._stats["expired"] += 1
                logger.info(f"Permission expired: {permission_id}")
                return False
            
            response = PermissionResponse(
                permission_id=permission_id,
                approved=approved,
                reason=reason,
                responded_by=responded_by,
            )
            
            self._history.append({
                "permission": pending.to_dict(),
                "response": asdict(response),
            })
            
            if approved:
                self._stats["approved"] += 1
            else:
                self._stats["denied"] += 1
            
            self._results[permission_id] = response
            
            # Trigger async event if waiting
            if permission_id in self._waiting:
                self._waiting[permission_id].set()
            
            # Call callback
            if pending.callback:
                try:
                    if asyncio.iscoroutinefunction(pending.callback):
                        from core.helpers import create_tracked_task
                        create_tracked_task(pending.callback(approved, reason), name="permission_callback")
                    else:
                        pending.callback(approved, reason)
                except Exception as e:
                    logger.error(f"Callback error for {permission_id}: {e}")
        
        self._notify_callbacks("response", {"id": permission_id, "approved": approved, "reason": reason})
        self._save_state()
        
        logger.info(f"Permission {permission_id} {'approved' if approved else 'denied'}: {reason}")
        
        return True
    
    async def request_and_wait(
        self,
        type: PermissionType,
        source: str,
        operation: str,
        risk_level: RiskLevel,
        context: Dict[str, Any] = None,
        timeout_seconds: float = 300,
        **kwargs,
    ) -> Optional[PermissionResponse]:
        """Request permission and wait for response.
        
        Args:
            timeout_seconds: Max time to wait (default 5 minutes)
        
        Returns:
            PermissionResponse or None if timed out
        """
        pending = self.request(type, source, operation, risk_level, context, **kwargs)
        
        # Create event for this permission
        event = asyncio.Event()
        self._waiting[pending.id] = event
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            return self._results.pop(pending.id, None)
        except asyncio.TimeoutError:
            with self._lock:
                self._pending.pop(pending.id, None)
            self._stats["expired"] += 1
            logger.info(f"Permission timed out: {pending.id}")
            return None
        finally:
            self._waiting.pop(pending.id, None)
    
    def get_pending(self, type_filter: PermissionType = None) -> List[dict]:
        """Get all pending permissions.
        
        Args:
            type_filter: Optional filter by permission type
        
        Returns:
            List of pending permission dicts
        """
        with self._lock:
            pending = []
            for perm_id, perm in list(self._pending.items()):
                if perm.is_expired():
                    self._pending.pop(perm_id)
                    self._stats["expired"] += 1
                    continue
                
                if type_filter and perm.permission_type != type_filter:
                    continue
                
                pending.append(perm.to_dict())
        
        return pending
    
    def get_history(self, limit: int = 100, type_filter: PermissionType = None) -> List[dict]:
        """Get permission history.
        
        Args:
            limit: Max items to return
            type_filter: Optional filter by permission type
        """
        with self._lock:
            history = list(self._history)
        
        if type_filter:
            history = [h for h in history if h["permission"]["type"] == type_filter.value]
        
        return history[-limit:]
    
    def get_stats(self) -> dict:
        """Get permission statistics."""
        with self._lock:
            return {
                **self._stats,
                "pending_count": len(self._pending),
                "history_count": len(self._history),
            }
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback for permission events.
        
        Callback receives (event_type, data) where:
        - event_type: "request" or "response"
        - data: PendingPermission or response dict
        """
        self._callbacks.append(callback)
    
    def register_ws_callback(self, callback: Callable) -> None:
        """Register a WebSocket callback for real-time updates."""
        self._ws_callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.debug(f"Callback error: {e}")
        
        for callback in self._ws_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.debug(f"WS callback error: {e}")
    
    def block_high_risk(self, permission_id: str, reason: str = "Blocked by policy") -> bool:
        """Block a high-risk permission without user interaction.
        
        Used for automatic blocking of dangerous operations.
        """
        return self.respond(permission_id, approved=False, reason=reason, responded_by="system")
    
    def auto_approve_low_risk(self, permission_id: str, reason: str = "Auto-approved: low risk") -> bool:
        """Auto-approve a low-risk permission.
        
        Used when auto_approve_safe is enabled.
        """
        return self.respond(permission_id, approved=True, reason=reason, responded_by="system")


# Singleton getter
def get_permission_coordinator() -> PermissionCoordinator:
    return PermissionCoordinator.get_instance()
