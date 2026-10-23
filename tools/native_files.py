"""Native File Tools for Jarvis - A personal workspace for file operations.

**DEPRECATED (2026-08-31) — RETIRED, NOT REGISTERED.** Kept on disk because
tests/test_native_files.py and tests/test_security_hardening.py pin its
sandbox-escape regression coverage, but no runtime path reaches it and none
should: file operations are owned by tools/system.py (registered file
tools), tools/terminal.py (shell) and the ExecutionContext CWD (Phase 2's
source of truth). This module's own "~/.jarvis/workspace" CWD model is a
*second, conflicting* CWD model — registering it would recreate the
divergent-policy problem the 2026-07 review cycle eliminated elsewhere.
Do not wire it; if workspace-scoped file tools are ever wanted, port them
into tools/system.py rather than reviving this module.

This module provides Jarvis with a native workspace - a folder on the drive
where Jarvis has full read/write permissions by default. This is Jarvis's
"home directory" for creating, modifying, and organizing files.

Architecture:
- Workspace: ~/.jarvis/workspace/ - Jarvis's personal file space
- Sandbox Mode: Restrict all operations to workspace (default for safety)
- Full Access Mode: Allow operations outside workspace (requires permission)
- Native Integration: Works with terminal, editor, and other tools

Security:
- Protected paths are always blocked
- Path traversal attacks are prevented
- File size limits prevent memory issues
- Binary files are detected and handled appropriately
"""

import asyncio
import logging
import mimetypes
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from core.helpers import is_protected_path, safe_error_message

from .base import Tool, async_read_file, async_read_file_bytes, async_write_file

logger = logging.getLogger(__name__)


class AccessMode(str, Enum):
    """File access mode for Jarvis."""
    SANDBOX = "sandbox"      # Workspace only - safest
    FULL = "full"            # Full filesystem access - requires permission


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    absolute_path: str
    size: int
    size_human: str
    modified: str
    created: str
    is_binary: bool
    mime_type: Optional[str]
    is_readonly: bool
    is_workspace: bool
    hash: Optional[str] = None


@dataclass
class WorkspaceConfig:
    """Configuration for Jarvis's workspace."""
    workspace_path: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "workspace")
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_read_size: int = 10 * 1024 * 1024  # 10MB for reading
    allowed_extensions: list[str] = field(default_factory=list)
    blocked_extensions: list[str] = field(default_factory=lambda: [
        ".exe", ".dll", ".so", ".dylib", ".bin",  # Executables
        ".pem", ".key", ".p12", ".pfx",  # Certificates/keys
    ])
    access_mode: AccessMode = AccessMode.SANDBOX
    
    def __post_init__(self):
        self.workspace_path = Path(self.workspace_path).expanduser()
        self.workspace_path.mkdir(parents=True, exist_ok=True)


class NativeFileManager:
    """Manager for Jarvis's native file operations.
    
    This class manages the workspace and provides safe file operations.
    It's designed to be the single source of truth for file access.
    """
    
    _instance: Optional["NativeFileManager"] = None
    _lock = asyncio.Lock()
    
    def __init__(
        self,
        workspace: Optional[Path] = None,
        access_mode: AccessMode = AccessMode.SANDBOX,
    ):
        self.config = WorkspaceConfig(
            workspace_path=workspace or Path.home() / ".jarvis" / "workspace",
            access_mode=access_mode,
        )
        self._initialize_workspace()
    
    @classmethod
    async def get_instance(
        cls,
        workspace: Optional[Path] = None,
        access_mode: AccessMode = AccessMode.SANDBOX,
    ) -> "NativeFileManager":
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(workspace, access_mode)
            return cls._instance
    
    def _initialize_workspace(self) -> None:
        """Set up the workspace directory structure."""
        workspace = self.config.workspace_path
        
        workspace.mkdir(parents=True, exist_ok=True)
        
        subdirs = [
            "documents",
            "projects",
            "temp",
            "backups",
            "exports",
        ]
        
        for subdir in subdirs:
            (workspace / subdir).mkdir(exist_ok=True)
        
        readme_path = workspace / "README.md"
        if not readme_path.exists():
            readme_content = """# Jarvis Workspace

This is Jarvis's personal workspace directory. Files here are managed by Jarvis.

## Directory Structure

- `documents/` - Documents and notes
- `projects/` - Project files and code
- `temp/` - Temporary files (may be cleaned up)
- `backups/` - Backup copies of important files
- `exports/` - Exported data and reports

## Usage

Use the `native_read` and `native_write` tools to work with files.
Paths are relative to this workspace unless absolute paths are used.

Example:
- `read_file("notes.txt")` - Reads `~/.jarvis/workspace/notes.txt`
- `write_file("documents/report.md", content)` - Writes to documents folder
"""
            readme_path.write_text(readme_content, encoding='utf-8')
        
        logger.info(f"Native workspace initialized at {workspace}")
    
    @property
    def workspace(self) -> Path:
        """Get the workspace path."""
        return self.config.workspace_path
    
    def resolve_path(self, path: str, for_write: bool = False) -> tuple[Path, bool, str]:
        """Resolve a path to its absolute location.
        
        Args:
            path: Input path (relative or absolute)
            for_write: Whether this is for a write operation
            
        Returns:
            Tuple of (resolved_path, is_in_workspace, error_message)
        """
        try:
            input_path = Path(path).expanduser()

            # A rooted path with no drive ("/etc/passwd") reports is_absolute()
            # == False on Windows, but joining it onto the workspace re-anchors
            # it at the drive root and escapes. Treat anything rooted as absolute.
            is_rooted = input_path.is_absolute() or input_path.root != ""

            if is_rooted:
                resolved = input_path.resolve()
            else:
                resolved = (self.workspace / path).resolve()
                try:
                    from core.execution_context import get_execution_context
                    ctx = get_execution_context()
                    if ctx.cwd and ctx.cwd != str(self.workspace):
                        cwd_resolved = (Path(ctx.cwd) / path).resolve()
                        if self._is_in_workspace(cwd_resolved) or self.config.access_mode == AccessMode.FULL:
                            resolved = cwd_resolved
                except Exception:
                    pass

            # Always derive containment from the resolved path. The relative
            # branches previously asserted is_in_workspace=True without checking,
            # so "../escape.txt" reported as in-workspace and silently disabled
            # both the traversal guard and the sandbox guard below.
            is_in_workspace = self._is_in_workspace(resolved)

            # Check the path as written as well as the resolved one. A POSIX
            # path like "/etc/passwd" resolves to "<drive>:\etc\passwd" on
            # Windows and would otherwise slip past the blocklist entirely.
            if self._is_protected(resolved) or self._is_protected(input_path):
                return resolved, False, "Access denied: protected system path"

            # Only block if path contains .. AND the resolved path is outside workspace
            # (legitimate uses like workspace/projects/../documents are fine after resolve)
            if ".." in path and not is_in_workspace and self.config.access_mode == AccessMode.SANDBOX:
                return resolved, False, "Path traversal outside workspace not allowed"

            if self.config.access_mode == AccessMode.SANDBOX and not is_in_workspace:
                return resolved, False, f"Sandbox mode: use workspace paths. Workspace: {self.workspace}"

            return resolved, is_in_workspace, ""

        except Exception:
            return Path(path), False, "Invalid path"

    def _is_protected(self, resolved: Path) -> bool:
        """Check whether a resolved path lives inside a protected system location."""
        return is_protected_path(resolved, PROTECTED_PATHS)
    
    def _is_in_workspace(self, path: Path) -> bool:
        """Check if a path is within the workspace.
        
        Uses path resolution to handle symlinks and prevents escape via symlinks
        pointing outside the workspace. Fails closed on any error (safer default).
        """
        try:
            workspace = self.workspace.resolve()
            resolved = path.resolve()
            return str(resolved).startswith(str(workspace) + os.sep) or resolved == workspace
        except Exception:
            return False  # Fail closed: deny access on error
    
    def is_binary(self, path: Path) -> bool:
        """Check if a file is binary."""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(8192)
            return b"\x00" in chunk
        except Exception:
            return True  # Fail closed: treat inaccessible files as binary
    
    def get_file_info(self, path: Path) -> FileInfo:
        """Get detailed information about a file."""
        stat_info = path.stat()
        
        mime_type, _ = mimetypes.guess_type(str(path))
        is_binary_val = self.is_binary(path) if path.is_file() else False
        
        size = stat_info.st_size
        if size < 1024:
            size_human = f"{size} B"
        elif size < 1024 * 1024:
            size_human = f"{size / 1024:.1f} KB"
        else:
            size_human = f"{size / (1024 * 1024):.1f} MB"
        
        return FileInfo(
            path=str(path.relative_to(self.workspace)) if self._is_in_workspace(path) else str(path),
            absolute_path=str(path),
            size=size,
            size_human=size_human,
            modified=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            created=datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            is_binary=is_binary_val,
            mime_type=mime_type,
            is_readonly=not os.access(path, os.W_OK),
            is_workspace=self._is_in_workspace(path),
        )
    
    def list_directory(self, path: Path) -> list[dict]:
        """List contents of a directory."""
        items = []
        for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                info = self.get_file_info(item) if item.is_file() else None
                items.append({
                    "name": item.name,
                    "path": str(item.relative_to(self.workspace)) if self._is_in_workspace(item) else str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "info": info.__dict__ if info else None,
                })
            except Exception as e:
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "error",
                    "error": safe_error_message(e),
                })
        return items
    
    def create_backup(self, path: Path) -> Optional[Path]:
        """Create a backup of a file before modification."""
        if not path.exists():
            return None
        
        backup_dir = self.workspace / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path


PROTECTED_PATHS = [
    "/etc", "/root", "/boot", "/sys", "/proc", "/dev",
    "/var/log", "/var/lib",
    "/usr/bin", "/usr/sbin", "/bin", "/sbin",
]

# Platform system directories (Windows has no equivalent of the POSIX paths
# above) are contributed by core.helpers.is_protected_path.


_file_manager: Optional[NativeFileManager] = None
_file_manager_lock = asyncio.Lock()


async def get_file_manager(
    workspace: Optional[Path] = None,
    access_mode: AccessMode = AccessMode.SANDBOX,
) -> NativeFileManager:
    """Get the singleton NativeFileManager instance."""
    global _file_manager
    async with _file_manager_lock:
        if _file_manager is None:
            _file_manager = NativeFileManager(workspace, access_mode)
        return _file_manager


class NativeReadTool(Tool):
    """Native file reading tool with workspace support.
    
    This tool reads files with proper handling of:
    - Workspace-relative paths (default)
    - Absolute paths (in full access mode)
    - Binary file detection
    - Large file warnings
    - Encoding detection
    """
    
    name = "native_read"
    description = """Read files from Jarvis's workspace or the filesystem.

Features:
- Default paths are relative to workspace (~/.jarvis/workspace)
- Absolute paths work in full access mode
- Automatic encoding detection for text files
- Binary file detection and warnings
- Large file handling with size limits

Examples:
- native_read("notes.txt") - Read from workspace
- native_read("documents/report.md") - Read from workspace subfolder
- native_read("/absolute/path/file.txt") - Read absolute path (full mode)
"""
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (relative to workspace or absolute)",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding (default: auto-detect)",
                "default": "auto",
            },
            "start_line": {
                "type": "integer",
                "description": "Start reading from this line (for large files)",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "End reading at this line (for large files)",
                "default": None,
            },
            "show_line_numbers": {
                "type": "boolean",
                "description": "Show line numbers in output",
                "default": True,
            },
        },
        "required": ["path"],
    }
    requires_permission = False
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(
        self,
        path: str,
        encoding: str = "auto",
        start_line: int = 1,
        end_line: Optional[int] = None,
        show_line_numbers: bool = True,
        **kwargs,
    ) -> str:
        """Read a file's contents."""
        if not path:
            path = kwargs.get("file", kwargs.get("filepath", ""))
        
        if not path:
            return "Error: No file path provided."
        
        try:
            from core.execution_context import get_execution_context
            get_execution_context().log_file_op("read", path, 0, success=True)
        except Exception:
            pass

        manager = await self._get_manager()
        resolved, is_workspace, error = manager.resolve_path(path)
        
        if error:
            return f"Error: {error}\n\nWorkspace: {manager.workspace}"
        
        if not resolved.exists():
            return f"File not found: {path}\n\nWorkspace: {manager.workspace}"
        
        if resolved.is_dir():
            items = manager.list_directory(resolved)
            lines = [f"# Directory: {path}\n"]
            for item in items:
                icon = "📁" if item["type"] == "directory" else "📄"
                size = ""
                if item.get("info"):
                    size = f" ({item['info']['size_human']})"
                lines.append(f"{icon} {item['name']}{size}")
            return "\n".join(lines)
        
        if resolved.stat().st_size > manager.config.max_read_size:
            size = resolved.stat().st_size
            return f"File too large ({size / (1024*1024):.1f} MB). Max: {manager.config.max_read_size / (1024*1024):.1f} MB. Use start_line/end_line to read in chunks."
        
        if manager.is_binary(resolved):
            info = manager.get_file_info(resolved)
            return f"Binary file: {info.mime_type or 'unknown type'}\nSize: {info.size_human}\nPath: {resolved}\n\nUse `native_read_binary` for binary files."
        
        try:
            encodings_to_try = ["utf-8"]
            if encoding != "auto":
                encodings_to_try = [encoding]
            else:
                encodings_to_try = ["utf-8", "latin-1", "cp1252"]
            
            content = None
            for enc in encodings_to_try:
                try:
                    content = await async_read_file(resolved, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return "Could not read file with any encoding. Try specifying encoding."
            
            lines = content.splitlines()
            total_lines = len(lines)
            
            start_idx = max(0, start_line - 1)
            end_idx = end_line if end_line else total_lines
            
            selected_lines = lines[start_idx:end_idx]
            
            output = []
            output.append(f"# File: {path}")
            if start_line > 1 or end_line:
                output.append(f"# Lines {start_line}-{min(end_line or total_lines, total_lines)} of {total_lines}")
            output.append(f"# Size: {resolved.stat().st_size} bytes")
            output.append("")
            
            for i, line in enumerate(selected_lines, start=start_line):
                if show_line_numbers:
                    output.append(f"{i:6d}│ {line}")
                else:
                    output.append(line)
            
            if end_line and end_line < total_lines:
                output.append(f"\n... ({total_lines - end_line} more lines)")
            
            return "\n".join(output)
            
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {safe_error_message(e)}"


class NativeWriteTool(Tool):
    """Native file writing tool with workspace support.
    
    This tool writes files with proper handling of:
    - Workspace-relative paths (default)
    - Automatic backup creation
    - Directory creation
    - Safe writing (write to temp, then move)
    """
    
    name = "native_write"
    description = """Write files to Jarvis's workspace or the filesystem.

Features:
- Default paths are relative to workspace (~/.jarvis/workspace)
- Automatic backup of existing files
- Creates directories as needed
- Safe write operation (atomic move)

Examples:
- native_write("notes.txt", "content") - Write to workspace
- native_write("documents/report.md", content) - Write to subfolder
- native_write("path", content, mode="append") - Append to file
"""
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (relative to workspace or absolute)",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append", "prepend"],
                "description": "Write mode (default: write)",
                "default": "write",
            },
            "create_backup": {
                "type": "boolean",
                "description": "Create backup if file exists (default: true)",
                "default": True,
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding (default: utf-8)",
                "default": "utf-8",
            },
        },
        "required": ["path", "content"],
    }
    requires_permission = True
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(
        self,
        path: str,
        content: str,
        mode: str = "write",
        create_backup: bool = True,
        encoding: str = "utf-8",
        **kwargs,
    ) -> str:
        """Write content to a file."""
        if not path:
            return "Error: No file path provided."
        
        if content is None:
            return "Error: No content provided."
        
        try:
            from core.execution_context import get_execution_context
            get_execution_context().log_file_op("write", path, len(content) if content else 0, success=True)
        except Exception:
            pass

        manager = await self._get_manager()
        manager = await self._get_manager()
        resolved, is_workspace, error = manager.resolve_path(path, for_write=True)
        
        if error:
            return f"Error: {error}\n\nWorkspace: {manager.workspace}"
        
        ext = resolved.suffix.lower()
        if ext in manager.config.blocked_extensions:
            return f"Error: Cannot write files with extension '{ext}' for security reasons."
        
        try:
            backup_path = None
            if resolved.exists() and create_backup and mode == "write":
                backup_path = manager.create_backup(resolved)
            
            content_to_write = content
            
            if mode == "append" and resolved.exists():
                existing = await async_read_file(resolved, encoding=encoding)
                content_to_write = existing + content
            
            elif mode == "prepend" and resolved.exists():
                existing = await async_read_file(resolved, encoding=encoding)
                content_to_write = content + existing
            
            resolved.parent.mkdir(parents=True, exist_ok=True)
            
            temp_path = resolved.with_suffix(resolved.suffix + ".tmp")
            await async_write_file(temp_path, content_to_write, encoding=encoding)
            
            temp_path.replace(resolved)
            
            lines = content_to_write.count("\n") + (1 if content_to_write and not content_to_write.endswith("\n") else 0)
            result = [
                f"✓ Wrote {len(content_to_write)} characters ({lines} lines) to {path}",
                f"  Location: {resolved}",
                f"  Mode: {mode}",
            ]
            
            if backup_path:
                result.append(f"  Backup: {backup_path}")
            
            return "\n".join(result)
            
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {safe_error_message(e)}"


class NativeListTool(Tool):
    """List files in a directory with detailed information."""
    
    name = "native_list"
    description = """List files in a directory with detailed information.

Shows file sizes, types, modification dates, and other metadata.
Default path is the workspace root.
"""
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (default: workspace root)",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "List recursively (default: false)",
                "default": False,
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Show hidden files (default: false)",
                "default": False,
            },
        },
        "required": [],
    }
    requires_permission = False
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(
        self,
        path: str = ".",
        recursive: bool = False,
        show_hidden: bool = False,
        **kwargs,
    ) -> str:
        """List directory contents."""
        manager = await self._get_manager()
        
        if not path or path == ".":
            path = str(manager.workspace)
        
        resolved, is_workspace, error = manager.resolve_path(path)
        
        if error:
            return f"Error: {error}\n\nWorkspace: {manager.workspace}"
        
        if not resolved.exists():
            return f"Directory not found: {path}\n\nWorkspace: {manager.workspace}"
        
        if not resolved.is_dir():
            return f"Not a directory: {path}"
        
        output = []
        output.append(f"# Directory: {path}")
        output.append(f"# Location: {resolved}")
        output.append("")
        
        total_size = 0
        file_count = 0
        dir_count = 0
        
        if recursive:
            try:
                walk_iter = os.walk(resolved, onerror=lambda e: logger.debug(f"os.walk error: {e}"))
                for root, dirs, files in walk_iter:
                    if not show_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                    
                    rel_root = Path(root).relative_to(resolved)
                    
                    if str(rel_root) != ".":
                        indent = "  " * (len(rel_root.parts))
                        output.append(f"{indent}📁 {rel_root}/")
                        dir_count += 1
                    
                    for f in files:
                        if not show_hidden and f.startswith("."):
                            continue
                    
                        file_path = Path(root) / f
                        try:
                            size = file_path.stat().st_size
                            total_size += size
                            file_count += 1
                            
                            indent = "  " * (len(rel_root.parts) + 1)
                            size_str = f"({size / 1024:.1f} KB)" if size > 1024 else f"({size} B)"
                            output.append(f"{indent}📄 {f} {size_str}")
                        except Exception:
                            output.append(f"{indent}📄 {f} (error)")
            except PermissionError:
                return "Access denied: cannot read directory"
        else:
            items = manager.list_directory(resolved)
            
            for item in items:
                if not show_hidden and item["name"].startswith("."):
                    continue
                
                if item["type"] == "directory":
                    output.append(f"📁 {item['name']}/")
                    dir_count += 1
                elif item["type"] == "file":
                    info = item.get("info")
                    if info:
                        total_size += info["size"]
                        file_count += 1
                        output.append(f"📄 {item['name']} ({info['size_human']})")
                    else:
                        output.append(f"📄 {item['name']}")
                        file_count += 1
                else:
                    output.append(f"❓ {item['name']}")
        
        output.append("")
        output.append("--- Summary ---")
        output.append(f"Files: {file_count}")
        output.append(f"Directories: {dir_count}")
        if total_size > 0:
            output.append(f"Total size: {total_size / (1024*1024):.1f} MB")
        
        return "\n".join(output)


class NativeDeleteTool(Tool):
    """Delete files with safety controls."""
    
    name = "native_delete"
    description = """Delete files or directories with safety controls.

Features:
- Confirmation prompt for safety
- Moves to trash instead of permanent delete (when available)
- Cannot delete protected system paths
- Can delete directories recursively
"""
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to delete",
            },
            "recursive": {
                "type": "boolean",
                "description": "Delete directories recursively",
                "default": False,
            },
            "confirm": {
                "type": "boolean",
                "description": "Confirm deletion (required for safety)",
                "default": False,
            },
        },
        "required": ["path"],
    }
    requires_permission = True
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(
        self,
        path: str,
        recursive: bool = False,
        confirm: bool = False,
        **kwargs,
    ) -> str:
        """Delete a file or directory."""
        if not path:
            return "Error: No path provided."
        
        manager = await self._get_manager()
        resolved, is_workspace, error = manager.resolve_path(path, for_write=True)
        
        if error:
            return f"Error: {error}\n\nWorkspace: {manager.workspace}"
        
        if not resolved.exists():
            return f"Path not found: {path}"
        
        if not confirm:
            if resolved.is_file():
                return f"Would delete file: {path}\nSize: {resolved.stat().st_size} bytes\n\nSet confirm=true to proceed."
            else:
                return f"Would delete directory: {path}\n\nSet confirm=true to proceed."
        
        try:
            if resolved.is_file():
                backup = manager.create_backup(resolved)
                resolved.unlink()
                return f"✓ Deleted file: {path}\nBackup saved: {backup}"
            
            elif resolved.is_dir():
                if not recursive:
                    items = list(resolved.iterdir())
                    if items:
                        return f"Directory not empty. Use recursive=true to delete.\nContents: {len(items)} items"
                
                backup_dir = manager.workspace / "backups" / f"deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(resolved), str(backup_dir))
                
                return f"✓ Moved to backup: {path}\nBackup location: {backup_dir}"
            
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error deleting: {safe_error_message(e)}"


class NativeMoveTool(Tool):
    """Move or rename files."""
    
    name = "native_move"
    description = """Move or rename files and directories.

Examples:
- native_move("old.txt", "new.txt") - Rename file
- native_move("file.txt", "documents/file.txt") - Move to subfolder
"""
    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source path",
            },
            "destination": {
                "type": "string",
                "description": "Destination path",
            },
        },
        "required": ["source", "destination"],
    }
    requires_permission = True
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(self, source: str, destination: str, **kwargs) -> str:
        """Move or rename a file."""
        if not source or not destination:
            return "Error: Source and destination paths required."
        
        manager = await self._get_manager()
        
        src_resolved, src_workspace, src_error = manager.resolve_path(source)
        dst_resolved, dst_workspace, dst_error = manager.resolve_path(destination, for_write=True)
        
        if src_error:
            return f"Source error: {src_error}"
        if dst_error:
            return f"Destination error: {dst_error}"
        
        if not src_resolved.exists():
            return f"Source not found: {source}"
        
        try:
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            backup = None
            if dst_resolved.exists():
                backup = manager.create_backup(dst_resolved)
            
            shutil.move(str(src_resolved), str(dst_resolved))
            
            result = f"✓ Moved: {source} → {destination}"
            if backup:
                result += f"\nBackup created: {backup}"
            
            return result
            
        except Exception as e:
            return f"Error moving file: {safe_error_message(e)}"


class NativeCopyTool(Tool):
    """Copy files or directories."""
    
    name = "native_copy"
    description = """Copy files or directories.

Examples:
- native_copy("file.txt", "file_backup.txt") - Create backup copy
- native_copy("documents", "backups/documents") - Copy directory
"""
    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source path",
            },
            "destination": {
                "type": "string",
                "description": "Destination path",
            },
        },
        "required": ["source", "destination"],
    }
    requires_permission = True
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(self, source: str, destination: str, **kwargs) -> str:
        """Copy a file or directory."""
        if not source or not destination:
            return "Error: Source and destination paths required."
        
        manager = await self._get_manager()
        
        src_resolved, _, src_error = manager.resolve_path(source)
        dst_resolved, _, dst_error = manager.resolve_path(destination, for_write=True)
        
        if src_error:
            return f"Source error: {src_error}"
        if dst_error:
            return f"Destination error: {dst_error}"
        
        if not src_resolved.exists():
            return f"Source not found: {source}"
        
        try:
            dst_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            if dst_resolved.exists():
                manager.create_backup(dst_resolved)
            
            if src_resolved.is_file():
                shutil.copy2(str(src_resolved), str(dst_resolved))
            else:
                shutil.copytree(str(src_resolved), str(dst_resolved))
            
            return f"✓ Copied: {source} → {destination}"
            
        except Exception as e:
            return f"Error copying: {safe_error_message(e)}"


class WorkspaceInfoTool(Tool):
    """Get information about the workspace."""
    
    name = "workspace_info"
    description = """Get information about Jarvis's workspace.

Shows workspace location, disk usage, recent files, and directory structure.
"""
    parameters = {
        "type": "object",
        "properties": {
            "show_tree": {
                "type": "boolean",
                "description": "Show directory tree",
                "default": False,
            },
        },
        "required": [],
    }
    requires_permission = False
    
    def __init__(self, file_manager: Optional[NativeFileManager] = None):
        self._file_manager = file_manager
    
    async def _get_manager(self) -> NativeFileManager:
        if self._file_manager:
            return self._file_manager
        return await get_file_manager()
    
    async def execute(self, show_tree: bool = False, **kwargs) -> str:
        """Get workspace information."""
        manager = await self._get_manager()
        workspace = manager.workspace
        
        output = []
        output.append("# Jarvis Workspace")
        output.append("")
        output.append(f"📁 Location: {workspace}")
        output.append(f"🔒 Access Mode: {manager.config.access_mode.value}")
        output.append(f"📏 Max File Size: {manager.config.max_file_size / (1024*1024):.0f} MB")
        output.append("")
        
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for root, dirs, files in os.walk(workspace):
            dir_count += len(dirs)
            for f in files:
                file_path = Path(root) / f
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Failed to stat file {file_path}: {e}")
        
        output.append("## Statistics")
        output.append(f"- Files: {file_count}")
        output.append(f"- Directories: {dir_count}")
        output.append(f"- Total Size: {total_size / (1024*1024):.1f} MB")
        
        disk_usage = shutil.disk_usage(workspace)
        output.append(f"- Disk Free: {disk_usage.free / (1024*1024*1024):.1f} GB")
        output.append("")
        
        if show_tree:
            output.append("## Directory Structure")
            output.append("```")
            for root, dirs, files in os.walk(workspace):
                level = root.replace(str(workspace), "").count(os.sep)
                indent = "  " * level
                output.append(f"{indent}📁 {Path(root).name}/")
                subindent = "  " * (level + 1)
                for f in sorted(files)[:10]:
                    output.append(f"{subindent}📄 {f}")
                if len(files) > 10:
                    output.append(f"{subindent}... and {len(files) - 10} more files")
            output.append("```")
        
        return "\n".join(output)


TOOLS = [
    NativeReadTool(),
    NativeWriteTool(),
    NativeListTool(),
    NativeDeleteTool(),
    NativeMoveTool(),
    NativeCopyTool(),
    WorkspaceInfoTool(),
]


async def initialize_native_files(
    workspace: Optional[Path] = None,
    access_mode: AccessMode = AccessMode.SANDBOX,
) -> NativeFileManager:
    """Initialize the native file system.
    
    Call this during Jarvis startup to set up the workspace.
    """
    manager = await get_file_manager(workspace, access_mode)
    logger.info(f"Native file tools initialized. Workspace: {manager.workspace}")
    return manager


def register_native_tools(agent, access_mode: AccessMode = AccessMode.SANDBOX) -> int:
    """Register native file tools with an agent.
    
    Args:
        agent: The JarvisAgent instance
        access_mode: File access mode (sandbox or full)
        
    Returns:
        Number of tools registered
    """
    import asyncio
    
    manager = asyncio.run(initialize_native_files(access_mode=access_mode))
    
    tools = [
        NativeReadTool(manager),
        NativeWriteTool(manager),
        NativeListTool(manager),
        NativeDeleteTool(manager),
        NativeMoveTool(manager),
        NativeCopyTool(manager),
        WorkspaceInfoTool(manager),
    ]
    
    agent.register_tools(tools)
    logger.info(f"Native file tools registered ({len(tools)} tools)")
    
    return len(tools)