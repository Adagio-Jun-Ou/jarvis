"""System tools for Jarvis - shell commands and file operations."""
import asyncio
import shutil
from datetime import datetime
from pathlib import Path

from .base import Tool, async_read_file, async_write_file
from core.helpers import safe_error_message


class ShellTool(Tool):
    """Tool for executing shell commands with safety controls."""

    name = "shell"
    description = """Execute shell commands on the system.

SAFE for most development commands: ls, cat, git, python, npm, pip, etc.
MEDIUM RISK for: package installs, file operations, docker ops.
BLOCKED: rm -rf /, sudo, su, fork bombs, remote scripts.
"""
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }
    requires_permission = False

    # Commands that are always safe - no permission needed
    SAFE_COMMANDS = {
        # File viewing
        "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
        "pwd", "whoami", "date", "uname", "which", "env", "printenv",
        "tree", "less", "more", "diff", "file",
        # Development
        "git", "python", "python3", "pip", "pip3", "node", "npm", "npx",
        "yarn", "cargo", "rustc", "go",
        # Code quality
        "pytest", "black", "isort", "ruff", "mypy", "flake8",
        # Editors (read-only mode)
        "code", "vim", "nano", "emacs", "subl",
        # Archive
        "tar", "unzip", "zip",
    }

    # Patterns that are ALWAYS blocked
    BLOCKED_PATTERNS = [
        "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf ~/",
        "mkfs", "> /dev/sd",
        ":(){ :|:& };:",
        "curl | bash", "wget | bash", "curl | sh", "wget | sh",
    ]

    def __init__(self, sandbox_dir: str = "~"):
        self.sandbox_dir = Path(sandbox_dir).expanduser()

    def _assess_risk(self, command: str) -> tuple[str, str]:
        """Assess command risk level."""
        cmd_lower = command.lower().strip()

        # Always blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in cmd_lower:
                return "blocked", f"Dangerous pattern: {pattern}"

        # Check for sudo/su
        first_word = cmd_lower.split()[0] if cmd_lower.split() else ""
        if first_word in ("sudo", "su", "doas"):
            return "blocked", "Elevated privileges not allowed"

        # Command substitution
        if "$(" in command or "`" in command:
            return "high", "Command substitution is blocked"

        # High risk patterns
        if any(x in cmd_lower for x in ["dd if=/dev/", "chmod -r 777", "chown -r root"]):
            return "blocked", "Dangerous operation detected"

        # Medium risk: package operations, file modifications
        if first_word in ("rm", "mv", "cp"):
            if "-rf" in cmd_lower and "/" in cmd_lower:
                return "high", "System directory modification"
            return "medium", "File operation requires caution"

        if first_word in ("apt", "apt-get", "yum", "dnf", "pacman", "snap", "flatpak"):
            if "install" in cmd_lower or "remove" in cmd_lower:
                return "medium", "Package manager operation"

        if first_word in ("pip", "pip3", "npm", "yarn", "cargo"):
            if "install" in cmd_lower or "uninstall" in cmd_lower:
                return "medium", "Package installation"

        # Safe commands
        if first_word in self.SAFE_COMMANDS:
            return "low", "Safe command"

        # Unknown commands need review
        return "medium", f"Command '{first_word}' needs review"

    def _validate_command(self, command: str) -> tuple[bool, str]:
        """Validate command for safety."""
        risk, reason = self._assess_risk(command)

        if risk == "blocked":
            return False, f"BLOCKED: {reason}"

        return True, reason

    async def execute(self, command: str, **kwargs) -> str:
        """Execute a shell command."""
        valid, msg = self._validate_command(command)
        if not valid:
            return f"Command rejected: {msg}"

        # Additional sanitization: split command into parts for safer execution
        # This avoids shell interpretation issues
        import shlex
        try:
            # Parse the command into arguments
            args = shlex.split(command)
        except ValueError as e:
            return f"Invalid command syntax: {safe_error_message(e, 'Invalid command syntax')}"

        if not args:
            return "Empty command"

        try:
            # Use create_subprocess_exec instead of shell for better security
            process = await asyncio.create_subprocess_exec(
                args[0],  # Command
                *args[1:],  # Arguments
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.sandbox_dir),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                process.kill()
                return "Command timed out after 30 seconds"

            result = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            MAX_OUTPUT = 10 * 1024 * 1024  # 10MB
            if len(result) > MAX_OUTPUT:
                result = result[:MAX_OUTPUT] + f"\n... [truncated at {MAX_OUTPUT} bytes]"
            if len(error) > MAX_OUTPUT:
                error = error[:MAX_OUTPUT] + f"\n... [truncated at {MAX_OUTPUT} bytes]"

            if process.returncode != 0:
                return f"Command failed (exit {process.returncode}):\n{error or result}"
            return result or "Command completed successfully (no output)"

        except FileNotFoundError:
            return f"Command not found: {args[0]}"
        except Exception as e:
            return f"Error executing command: {safe_error_message(e, 'Command execution failed')}"


class ReadFileTool(Tool):
    """Tool for reading files."""

    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
        },
        "required": ["path"],
    }
    requires_permission = False

    PROTECTED_PATHS = [
        "/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/ssh",
        "/root/.ssh", "/.ssh", "/root/.gnupg",
        "/boot/", "/sys/", "/proc/", "/dev/",
        "/var/log/", "/var/lib/",
    ]
    
    def _resolve_safe_path(self, path: str) -> Path | None:
        """Resolve path safely, checking for traversal attacks.
        
        Returns:
            Resolved Path if safe, None if traversal attempted
        """
        try:
            file_path = Path(path).expanduser()
            resolved = file_path.resolve()
            
            for protected in self.PROTECTED_PATHS:
                if str(resolved).startswith(protected):
                    return None
            
            parts = []
            current = resolved
            while current != current.parent:
                parts.append(current.name)
                current = current.parent
            
            for part in reversed(parts):
                if part.startswith('..') or part == '..':
                    return None
            
            return resolved
        except Exception:
            return None

    async def execute(self, path: str, **kwargs) -> str:
        """Read a file's contents.
        
        Args:
            path: Path to the file to read
            **kwargs: Additional parameters for LLM flexibility (file, filepath)
        
        Returns:
            File contents or error message
        """
        # Handle common parameter variations
        if not path:
            path = kwargs.get('file', kwargs.get('filepath', kwargs.get('filename', '')))
        
        if not path:
            return "Error: No file path provided. Use 'path' parameter."
        
        try:
            resolved = self._resolve_safe_path(path)
            if resolved is None:
                return "Access denied: protected or invalid path"

            if not resolved.exists():
                return f"File not found: {path}"

            if not resolved.is_file():
                return f"Not a file: {path}"

            if resolved.stat().st_size > 10 * 1024 * 1024:
                return "File too large (max 10MB)"

            content = await async_read_file(resolved)

            return content

        except Exception as e:
            return f"Error reading file: {safe_error_message(e, 'File read failed')}"


class WriteFileTool(Tool):
    """Tool for writing files."""

    name = "write_file"
    description = "Write content to a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    }
    requires_permission = True
    
    PROTECTED_PATHS = [
        "~/.ssh/", "~/.gnupg/",
        "/etc/", "/root/", "/boot/", "/sys/", "/proc/", "/dev/",
        "/var/run/",
    ]

    PROTECTED_SHALLOW_PATHS = [
        "~/.config/",
    ]

    async def execute(self, path: str, content: str, **kwargs) -> str:
        """Write content to a file."""
        try:
            file_path = Path(path).expanduser()
            resolved = file_path.resolve()
            
            for protected in self.PROTECTED_PATHS:
                prot_resolved = str(Path(protected).expanduser().resolve())
                if str(resolved).startswith(prot_resolved):
                    return "Access denied: cannot write to protected path"
            
            for shallow in self.PROTECTED_SHALLOW_PATHS:
                shallow_dir = Path(shallow).expanduser().resolve()
                if resolved.parent == shallow_dir:
                    return "Access denied: cannot write to protected path"
            
            current = resolved
            while current != current.parent:
                if current.name.startswith('..') or current.name == '..':
                    return "Access denied: invalid path"
                current = current.parent

            resolved.parent.mkdir(parents=True, exist_ok=True)

            await async_write_file(resolved, content)

            return f"Successfully wrote {len(content)} characters to {path}"

        except Exception as e:
            return f"Error writing file: {safe_error_message(e, 'File write failed')}"


class ListFilesTool(Tool):
    """Tool for listing files in a directory."""

    name = "list_files"
    description = "List files and directories in a path"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (defaults to current directory)",
            },
        },
        "required": [],
    }
    requires_permission = False

    async def execute(self, path: str = ".", **kwargs) -> str:
        """List files in a directory.
        
        Args:
            path: Directory path to list (defaults to current directory)
            **kwargs: Additional parameters (ignored for compatibility)
        
        Returns:
            List of files and directories
        """
        # Handle common parameter variations
        if not path:
            path = kwargs.get('dir', kwargs.get('directory', kwargs.get('folder', '.')))
        
        try:
            dir_path = Path(path).expanduser()

            if not dir_path.exists():
                return f"Directory not found: {path}"

            if not dir_path.is_dir():
                return f"Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    items.append(f"[DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"[FILE] {item.name} ({self._format_size(size)})")

            if not items:
                return "Empty directory"

            return "\n".join(items)

        except Exception as e:
            return f"Error listing directory: {safe_error_message(e, 'Directory listing failed')}"

    @staticmethod
    def _format_size(size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


class DeleteFileTool(Tool):
    """Tool for deleting files."""

    name = "delete_file"
    description = "Delete a file or directory"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to delete",
            },
        },
        "required": ["path"],
    }
    requires_permission = True

    PROTECTED_PATHS = [
        "~/.ssh/", "~/.gnupg/",
        "/etc/", "/root/", "/boot/", "/sys/", "/proc/", "/dev/",
        "/var/run/",
    ]

    PROTECTED_SHALLOW_PATHS = [
        "~/.config/",
    ]

    async def execute(self, path: str, **kwargs) -> str:
        """Delete a file or directory."""
        try:
            target = Path(path).expanduser()
            resolved = target.resolve()

            for protected in self.PROTECTED_PATHS:
                prot_resolved = Path(protected).expanduser().resolve()
                if str(resolved).startswith(str(prot_resolved)):
                    return "Access denied: cannot delete protected path"

            for shallow in self.PROTECTED_SHALLOW_PATHS:
                shallow_dir = Path(shallow).expanduser().resolve()
                if resolved.parent == shallow_dir:
                    return "Access denied: cannot delete protected path"

            if not target.exists():
                return f"Path not found: {path}"

            if target.is_symlink():
                link_target = target.resolve()
                for protected in self.PROTECTED_PATHS:
                    prot_resolved = Path(protected).expanduser().resolve()
                    if str(link_target).startswith(str(prot_resolved)):
                        return "Access denied: symlink points to protected path"
                for shallow in self.PROTECTED_SHALLOW_PATHS:
                    shallow_dir = Path(shallow).expanduser().resolve()
                    if link_target.parent == shallow_dir:
                        return "Access denied: symlink points to protected path"
                target.unlink()
                return f"Deleted symlink: {path}"

            if target.is_file():
                target.unlink()
                return f"Deleted file: {path}"
            elif target.is_dir():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: shutil.rmtree(target))
                return f"Deleted directory: {path}"
            else:
                return f"Unknown path type: {path}"

        except Exception as e:
            return f"Error deleting: {safe_error_message(e, 'Delete failed')}"


class SystemInfoTool(Tool):
    """Tool for getting system information."""

    name = "system_info"
    description = "Get current system information (time, date, etc.)"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    requires_permission = False

    async def execute(self, **kwargs) -> str:
        """Get system information."""
        now = datetime.now()
        info = [
            f"Current time: {now.strftime('%H:%M:%S')}",
            f"Current date: {now.strftime('%Y-%m-%d')}",
            f"Day of week: {now.strftime('%A')}",
            f"Timezone: {datetime.now().astimezone().tzname()}",
        ]
        return "\n".join(info)


class SystemTools:
    """Collection of system tools."""

    def __init__(self, sandbox_dir: str = "~"):
        self.sandbox_dir = sandbox_dir
        self.tools = [
            ShellTool(sandbox_dir),
            ReadFileTool(),
            WriteFileTool(),
            ListFilesTool(),
            DeleteFileTool(),
            SystemInfoTool(),
        ]

    def get_tools(self) -> list[Tool]:
        """Get all system tools."""
        return self.tools
