"""Permission management for Jarvis tools."""
import json
import logging
import threading
from collections import OrderedDict
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"


@dataclass
class SelfEditPermissions:
    enabled: bool = True
    require_confirmation: bool = True
    safe_files_only: bool = True
    auto_test: bool = True
    max_lines_per_edit: int = 100


@dataclass
class AutonomousPermissions:
    background_tasks: str = "allow"
    tool_execution: str = "ask"
    web_requests: str = "allow"


@dataclass
class CodeModificationPermissions:
    read_access: str = "allow"
    write_access: str = "ask"
    safe_dirs: list[str] = field(default_factory=list)


class PermissionManager:
    """Manages permissions for sensitive actions."""

    DEFAULT_SAFE_DIRS = [
        "core/erdos_engine.py",
        "core/insight_bridge.py",
        "core/evolution_orchestrator.py",
        "core/daydream_engine.py",
        "tools/evolution_tools.py",
    ]
    _MAX_SESSION_PERMISSIONS = 1000

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.permissions_file = storage_path / "permissions.json"
        self._lock = threading.RLock()
        self.session_permissions: OrderedDict[str, PermissionLevel] = OrderedDict()
        self.stored_permissions: dict[str, str] = self._load_stored_permissions()

        self.permission_settings: dict[str, str] = {}
        self.self_edit = SelfEditPermissions()
        self.autonomous = AutonomousPermissions()
        self.code_modification = CodeModificationPermissions(safe_dirs=list(self.DEFAULT_SAFE_DIRS))

    def _load_stored_permissions(self) -> dict[str, str]:
        """Load stored permissions from disk."""
        loaded = {}
        if self.permissions_file.exists():
            try:
                loaded = read_json_with_integrity(self.permissions_file)
            except Exception as e:
                logger.warning(f"Failed to load stored permissions: {e}")
        with self._lock:
            return loaded

    def _save_stored_permissions(self) -> None:
        """Save stored permissions to disk."""
        with self._lock:
            snapshot = dict(self.stored_permissions)
        atomic_write_json(self.permissions_file, snapshot)

    def set_config(self, config: dict[str, Any]) -> None:
        """Set permission settings from config dict (supports nested structure)."""
        if isinstance(config, dict):
            if "shell_commands" in config:
                self.permission_settings["shell_commands"] = config["shell_commands"]
            if "file_delete" in config:
                self.permission_settings["file_delete"] = config["file_delete"]
            if "file_write" in config:
                self.permission_settings["file_write"] = config["file_write"]
            if "file_read" in config:
                self.permission_settings["file_read"] = config["file_read"]
            if "web_access" in config:
                self.permission_settings["web_access"] = config["web_access"]

            self_edit = config.get("self_edit", {})
            if self_edit:
                self.self_edit = SelfEditPermissions(
                    enabled=self_edit.get("enabled", True),
                    require_confirmation=self_edit.get("require_confirmation", True),
                    safe_files_only=self_edit.get("safe_files_only", True),
                    auto_test=self_edit.get("auto_test", True),
                    max_lines_per_edit=self_edit.get("max_lines_per_edit", 100),
                )

            autonomous = config.get("autonomous", {})
            if autonomous:
                self.autonomous = AutonomousPermissions(
                    background_tasks=autonomous.get("background_tasks", "allow"),
                    tool_execution=autonomous.get("tool_execution", "ask"),
                    web_requests=autonomous.get("web_requests", "allow"),
                )

            code_mod = config.get("code_modification", {})
            if code_mod:
                safe_dirs = code_mod.get("safe_dirs", self.DEFAULT_SAFE_DIRS)
                self.code_modification = CodeModificationPermissions(
                    read_access=code_mod.get("read_access", "allow"),
                    write_access=code_mod.get("write_access", "ask"),
                    safe_dirs=safe_dirs if safe_dirs else list(self.DEFAULT_SAFE_DIRS),
                )
        else:
            self.permission_settings = config

    def check_permission(self, action: str, details: dict[str, Any] | None = None) -> PermissionLevel | None:
        """Check if permission is already granted.

        Returns:
            PermissionLevel if granted, None if permission needs to be requested.
        """
        with self._lock:
            # Check config settings first
            config_setting = self._get_config_permission(action)
            if config_setting == "allow":
                return PermissionLevel.ALLOW_SESSION
            elif config_setting == "deny":
                return PermissionLevel.DENY

            # Check always-allow permissions
            if action in self.stored_permissions:
                stored = self.stored_permissions[action]
                # Handle TTL-based permissions
                if isinstance(stored, dict):
                    import time
                    if stored.get("permission") == "allow":
                        if time.time() < stored.get("expires", 0):
                            return PermissionLevel.ALLOW_SESSION
                        else:
                            # Expired, remove it
                            del self.stored_permissions[action]
                            self._save_stored_permissions()
                            return None
                elif stored == "allow_always":
                    return PermissionLevel.ALLOW_ALWAYS
                elif stored == "allow":
                    return PermissionLevel.ALLOW_SESSION
                elif stored == "deny":
                    # deny_permission(permanently=True) writes this value, but
                    # there was no branch for it, so an explicit permanent deny
                    # fell through to `return None` — i.e. "undecided". Callers
                    # that fail open on None then executed the action anyway.
                    return PermissionLevel.DENY

            # Check session permissions
            if action in self.session_permissions:
                return self.session_permissions[action]

            return None

    def _get_config_permission(self, action: str) -> str | None:
        """Get permission setting from config."""
        action_map = {
            "shell": "shell_commands",
            "file_delete": "file_delete",
            "file_write": "file_write",
            "file_read": "file_read",
            "web": "web_access",
        }

        for key, config_key in action_map.items():
            if action.startswith(key):
                return self.permission_settings.get(config_key)
        return None

    def can_self_edit(self, file_path: str, line_count: int = 0) -> tuple[bool, str]:
        """Check if self-edit is allowed for a file.
        
        Returns:
            (allowed, reason) tuple
        """
        if not self.self_edit.enabled:
            return False, "Self-edit is disabled in config"

        if line_count > self.self_edit.max_lines_per_edit:
            return False, f"Edit exceeds max lines ({line_count} > {self.self_edit.max_lines_per_edit})"

        if self.self_edit.safe_files_only:
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            safe = any(safe_dir.endswith(file_name) or file_path.endswith(safe_dir)
                      for safe_dir in self.code_modification.safe_dirs)
            if not safe:
                return False, f"File not in safe list: {file_path}"

        if self.self_edit.require_confirmation:
            return True, "Requires confirmation"

        return True, "Allowed"

    def can_autonomous_run(self, action_type: str) -> bool:
        """Check if autonomous action is allowed."""
        settings = {
            "background_tasks": self.autonomous.background_tasks,
            "tool_execution": self.autonomous.tool_execution,
            "web_requests": self.autonomous.web_requests,
        }
        setting = settings.get(action_type, "ask")
        return setting == "allow"

    def can_read_code(self) -> bool:
        """Check if code reading is allowed."""
        return self.code_modification.read_access == "allow"

    def can_write_code(self) -> str:
        """Check if code writing is allowed.
        
        Returns:
            "allow", "ask", or "deny"
        """
        return self.code_modification.write_access

    def grant_permission(self, action: str, level: PermissionLevel | None = None, ttl: int | None = None) -> None:
        """Grant permission for an action with optional TTL in seconds."""
        import time
        
        with self._lock:
            # Support both old PermissionLevel and new TTL-based approach
            if level is not None:
                if level == PermissionLevel.ALLOW_ALWAYS:
                    self.stored_permissions[action] = "allow_always"
                    self._save_stored_permissions()
                elif level == PermissionLevel.ALLOW_SESSION:
                    self.session_permissions[action] = PermissionLevel.ALLOW_SESSION
                    while len(self.session_permissions) > self._MAX_SESSION_PERMISSIONS:
                        self.session_permissions.popitem(last=False)
                # ALLOW_ONCE not stored
            elif ttl:
                self.stored_permissions[action] = {
                    "permission": "allow",
                    "expires": time.time() + ttl
                }
                self._save_stored_permissions()
            else:
                self.stored_permissions[action] = "allow"
                self._save_stored_permissions()

    def deny_permission(self, action: str, permanently: bool = False) -> None:
        """Deny permission for an action."""
        with self._lock:
            if permanently:
                self.stored_permissions[action] = "deny"
                self._save_stored_permissions()

    def clear_permission(self, action: str) -> None:
        """Clear permission for an action (reset to ask)."""
        with self._lock:
            if action in self.stored_permissions:
                del self.stored_permissions[action]
                self._save_stored_permissions()
            if action in self.session_permissions:
                del self.session_permissions[action]

    def get_all_permissions(self) -> dict[str, str]:
        """Get all current permission settings."""
        with self._lock:
            result = {}
            import time
            for action, perm in self.stored_permissions.items():
                if isinstance(perm, dict):
                    if perm.get("expires", float('inf')) > time.time():
                        result[action] = perm.get("permission", "allow")
                    else:
                        result[action] = "ask"
                else:
                    result[action] = perm
            for action, level in self.session_permissions.items():
                if action not in result:
                    result[action] = level.value
            result.update(self.permission_settings)
            return result

    def revoke_permission(self, action: str) -> None:
        """Revoke a stored permission."""
        with self._lock:
            if action in self.stored_permissions:
                del self.stored_permissions[action]
                self._save_stored_permissions()
            if action in self.session_permissions:
                del self.session_permissions[action]

    def clear_session_permissions(self) -> None:
        """Clear all session-only permissions."""
        with self._lock:
            self.session_permissions.clear()

    def format_permission_request(
        self,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Format a permission request message for display."""
        msg = f"Permission required: {action}"
        if details:
            msg += f"\nDetails: {json.dumps(details, indent=2)}"
        msg += "\nOptions: (o)nce, (s)ession, (a)lways, (d)eny"
        return msg
