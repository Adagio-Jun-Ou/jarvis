#!/usr/bin/env python3
"""
Native File Tools Tests

Tests the native file system tools that provide Jarvis with a workspace
for file operations with proper permissions and safety controls.
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

import sys
sys.path.insert(0, _JARVIS_ROOT)

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch


# ==================== FIXTURES ====================

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "jarvis_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
async def file_manager(temp_workspace):
    """Create a NativeFileManager with a temp workspace."""
    from tools.native_files import NativeFileManager, AccessMode
    
    manager = NativeFileManager(
        workspace=temp_workspace,
        access_mode=AccessMode.SANDBOX,
    )
    return manager


@pytest.fixture
async def file_tools(file_manager):
    """Create file tools with the test file manager."""
    from tools.native_files import (
        NativeReadTool,
        NativeWriteTool,
        NativeListTool,
        NativeDeleteTool,
        NativeMoveTool,
        NativeCopyTool,
        WorkspaceInfoTool,
    )
    
    return {
        "read": NativeReadTool(file_manager),
        "write": NativeWriteTool(file_manager),
        "list": NativeListTool(file_manager),
        "delete": NativeDeleteTool(file_manager),
        "move": NativeMoveTool(file_manager),
        "copy": NativeCopyTool(file_manager),
        "info": WorkspaceInfoTool(file_manager),
    }


# ==================== MANAGER TESTS ====================

class TestNativeFileManager:
    """Tests for NativeFileManager."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, temp_workspace):
        """Manager should initialize correctly."""
        from tools.native_files import NativeFileManager, AccessMode
        
        manager = NativeFileManager(
            workspace=temp_workspace,
            access_mode=AccessMode.SANDBOX,
        )
        
        assert manager is not None
        assert manager.workspace == temp_workspace
        assert manager.config.access_mode == AccessMode.SANDBOX
    
    @pytest.mark.asyncio
    async def test_workspace_structure_created(self, file_manager):
        """Manager should create workspace subdirectories."""
        workspace = file_manager.workspace
        
        assert (workspace / "documents").exists()
        assert (workspace / "projects").exists()
        assert (workspace / "temp").exists()
        assert (workspace / "backups").exists()
        assert (workspace / "exports").exists()
    
    @pytest.mark.asyncio
    async def test_resolve_relative_path(self, file_manager):
        """Manager should resolve relative paths to workspace."""
        resolved, is_workspace, error = file_manager.resolve_path("test.txt")
        
        assert error == ""
        assert is_workspace == True
        assert resolved == file_manager.workspace / "test.txt"
    
    @pytest.mark.asyncio
    async def test_resolve_absolute_path_in_workspace(self, file_manager):
        """Manager should accept absolute paths within workspace."""
        abs_path = file_manager.workspace / "test.txt"
        resolved, is_workspace, error = file_manager.resolve_path(str(abs_path))
        
        assert error == ""
        assert is_workspace == True
    
    @pytest.mark.asyncio
    async def test_resolve_path_outside_workspace_sandbox(self, file_manager):
        """Manager should reject paths outside workspace in sandbox mode."""
        from tools.native_files import AccessMode
        
        file_manager.config.access_mode = AccessMode.SANDBOX
        
        resolved, is_workspace, error = file_manager.resolve_path("/etc/passwd")
        
        assert error != ""  # Should have an error
    
    @pytest.mark.asyncio
    async def test_resolve_protected_path(self, file_manager):
        """Manager should always block protected system paths."""
        resolved, is_workspace, error = file_manager.resolve_path("/etc/shadow")
        
        assert error != ""
    
    @pytest.mark.asyncio
    async def test_is_binary_detection(self, file_manager):
        """Manager should detect binary files."""
        workspace = file_manager.workspace
        
        # Create text file
        text_file = workspace / "test.txt"
        text_file.write_text("Hello, World!", encoding='utf-8')
        
        # Create binary file
        binary_file = workspace / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")
        
        assert file_manager.is_binary(text_file) == False
        assert file_manager.is_binary(binary_file) == True
    
    @pytest.mark.asyncio
    async def test_get_file_info(self, file_manager):
        """Manager should provide file information."""
        workspace = file_manager.workspace
        
        test_file = workspace / "info_test.txt"
        test_file.write_text("Test content\nMultiple lines\n", encoding='utf-8')
        
        info = file_manager.get_file_info(test_file)
        
        assert info.path == "info_test.txt"
        assert info.is_binary == False
        assert info.size > 0
        assert info.is_workspace == True
    
    @pytest.mark.asyncio
    async def test_create_backup(self, file_manager):
        """Manager should create backups before modifications."""
        workspace = file_manager.workspace
        
        test_file = workspace / "to_backup.txt"
        test_file.write_text("Original content", encoding='utf-8')
        
        backup_path = file_manager.create_backup(test_file)
        
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_text(encoding='utf-8') == "Original content"


# ==================== READ TOOL TESTS ====================

class TestNativeReadTool:
    """Tests for NativeReadTool."""
    
    @pytest.mark.asyncio
    async def test_read_text_file(self, file_tools, file_manager):
        """Should read text files correctly."""
        workspace = file_manager.workspace
        
        test_file = workspace / "read_test.txt"
        test_file.write_text("Hello from Jarvis!", encoding='utf-8')
        
        result = await file_tools["read"].execute("read_test.txt")
        
        assert "Hello from Jarvis!" in result
        assert "read_test.txt" in result
    
    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, file_tools):
        """Should handle nonexistent files gracefully."""
        result = await file_tools["read"].execute("nonexistent.txt")
        
        assert "not found" in result.lower()
    
    @pytest.mark.asyncio
    async def test_read_directory(self, file_tools, file_manager):
        """Should list directory contents when reading a directory."""
        workspace = file_manager.workspace
        
        (workspace / "subdir").mkdir(exist_ok=True)
        (workspace / "subdir" / "file1.txt").write_text("content 1", encoding='utf-8')
        (workspace / "subdir" / "file2.txt").write_text("content 2", encoding='utf-8')
        
        result = await file_tools["read"].execute("subdir")
        
        assert "Directory" in result
        assert "file1.txt" in result
        assert "file2.txt" in result
    
    @pytest.mark.asyncio
    async def test_read_with_line_numbers(self, file_tools, file_manager):
        """Should show line numbers by default."""
        workspace = file_manager.workspace
        
        test_file = workspace / "lines.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n", encoding='utf-8')
        
        result = await file_tools["read"].execute("lines.txt", show_line_numbers=True)
        
        assert "1" in result
        assert "Line 1" in result
    
    @pytest.mark.asyncio
    async def test_read_partial_file(self, file_tools, file_manager):
        """Should read partial files with start/end line."""
        workspace = file_manager.workspace
        
        test_file = workspace / "long.txt"
        test_file.write_text("\n".join([f"Line {i}" for i in range(1, 101)]))
        
        result = await file_tools["read"].execute("long.txt", start_line=1, end_line=10)
        
        assert "Line 1" in result
        assert "Line 10" in result
        assert "Line 50" not in result


# ==================== WRITE TOOL TESTS ====================

class TestNativeWriteTool:
    """Tests for NativeWriteTool."""
    
    @pytest.mark.asyncio
    async def test_write_new_file(self, file_tools, file_manager):
        """Should create and write new files."""
        result = await file_tools["write"].execute(
            "new_file.txt",
            "This is new content",
            confirm=True,
        )
        
        assert "Wrote" in result
        assert (file_manager.workspace / "new_file.txt").exists()
        assert (file_manager.workspace / "new_file.txt").read_text(encoding='utf-8') == "This is new content"
    
    @pytest.mark.asyncio
    async def test_write_creates_directories(self, file_tools, file_manager):
        """Should create parent directories as needed."""
        result = await file_tools["write"].execute(
            "deeply/nested/path/file.txt",
            "Nested content",
            confirm=True,
        )
        
        assert "Wrote" in result
        assert (file_manager.workspace / "deeply" / "nested" / "path" / "file.txt").exists()
    
    @pytest.mark.asyncio
    async def test_append_mode(self, file_tools, file_manager):
        """Should append to existing files."""
        workspace = file_manager.workspace
        
        test_file = workspace / "append.txt"
        test_file.write_text("Original\n", encoding='utf-8')
        
        await file_tools["write"].execute(
            "append.txt",
            "Appended\n",
            mode="append",
            confirm=True,
        )
        
        assert test_file.read_text(encoding='utf-8') == "Original\nAppended\n"
    
    @pytest.mark.asyncio
    async def test_prepend_mode(self, file_tools, file_manager):
        """Should prepend to existing files."""
        workspace = file_manager.workspace
        
        test_file = workspace / "prepend.txt"
        test_file.write_text("End\n", encoding='utf-8')
        
        await file_tools["write"].execute(
            "prepend.txt",
            "Start\n",
            mode="prepend",
            confirm=True,
        )
        
        assert test_file.read_text(encoding='utf-8') == "Start\nEnd\n"
    
    @pytest.mark.asyncio
    async def test_backup_on_overwrite(self, file_tools, file_manager):
        """Should backup existing files on overwrite."""
        workspace = file_manager.workspace
        
        test_file = workspace / "overwrite.txt"
        test_file.write_text("Original content", encoding='utf-8')
        
        await file_tools["write"].execute(
            "overwrite.txt",
            "New content",
            create_backup=True,
            confirm=True,
        )
        
        # Check backup was created
        backup_dir = workspace / "backups"
        backups = list(backup_dir.glob("overwrite_*.txt"))
        assert len(backups) > 0
        assert backups[0].read_text(encoding='utf-8') == "Original content"


# ==================== LIST TOOL TESTS ====================

class TestNativeListTool:
    """Tests for NativeListTool."""
    
    @pytest.mark.asyncio
    async def test_list_directory(self, file_tools, file_manager):
        """Should list directory contents."""
        workspace = file_manager.workspace
        
        (workspace / "file1.txt").write_text("content", encoding='utf-8')
        (workspace / "file2.py").write_text("# python", encoding='utf-8')
        (workspace / "subdir").mkdir(exist_ok=True)
        
        result = await file_tools["list"].execute(".")
        
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result
    
    @pytest.mark.asyncio
    async def test_list_with_sizes(self, file_tools, file_manager):
        """Should show file sizes."""
        workspace = file_manager.workspace
        
        (workspace / "sized.txt").write_text("x" * 1000, encoding='utf-8')
        
        result = await file_tools["list"].execute(".")
        
        assert "KB" in result or "B" in result
    
    @pytest.mark.asyncio
    async def test_list_recursive(self, file_tools, file_manager):
        """Should list recursively when requested."""
        workspace = file_manager.workspace
        
        (workspace / "subdir").mkdir(exist_ok=True)
        (workspace / "subdir" / "nested.txt").write_text("nested", encoding='utf-8')
        
        result = await file_tools["list"].execute(".", recursive=True)
        
        assert "subdir" in result
        assert "nested.txt" in result


# ==================== DELETE TOOL TESTS ====================

class TestNativeDeleteTool:
    """Tests for NativeDeleteTool."""
    
    @pytest.mark.asyncio
    async def test_delete_requires_confirmation(self, file_tools, file_manager):
        """Delete should require confirmation."""
        workspace = file_manager.workspace
        
        test_file = workspace / "to_delete.txt"
        test_file.write_text("delete me", encoding='utf-8')
        
        result = await file_tools["delete"].execute("to_delete.txt", confirm=False)
        
        assert "confirm" in result.lower()
        assert test_file.exists()
    
    @pytest.mark.asyncio
    async def test_delete_file_with_confirmation(self, file_tools, file_manager):
        """Should delete file with confirmation."""
        workspace = file_manager.workspace
        
        test_file = workspace / "delete_me.txt"
        test_file.write_text("delete me", encoding='utf-8')
        
        result = await file_tools["delete"].execute("delete_me.txt", confirm=True)
        
        assert "Deleted" in result
        assert not test_file.exists()
    
    @pytest.mark.asyncio
    async def test_delete_directory_moves_to_backup(self, file_tools, file_manager):
        """Should move directories to backup instead of permanent delete."""
        workspace = file_manager.workspace
        
        test_dir = workspace / "delete_dir"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "file.txt").write_text("content", encoding='utf-8')
        
        result = await file_tools["delete"].execute(
            "delete_dir",
            recursive=True,
            confirm=True,
        )
        
        assert "backup" in result.lower() or "moved" in result.lower()


# ==================== MOVE/COPY TOOL TESTS ====================

class TestNativeMoveCopyTools:
    """Tests for move and copy tools."""
    
    @pytest.mark.asyncio
    async def test_move_file(self, file_tools, file_manager):
        """Should move files."""
        workspace = file_manager.workspace
        
        (workspace / "source.txt").write_text("content", encoding='utf-8')
        
        result = await file_tools["move"].execute("source.txt", "dest.txt", confirm=True)
        
        assert "Moved" in result
        assert not (workspace / "source.txt").exists()
        assert (workspace / "dest.txt").read_text(encoding='utf-8') == "content"
    
    @pytest.mark.asyncio
    async def test_copy_file(self, file_tools, file_manager):
        """Should copy files."""
        workspace = file_manager.workspace
        
        (workspace / "original.txt").write_text("original content", encoding='utf-8')
        
        result = await file_tools["copy"].execute("original.txt", "copy.txt", confirm=True)
        
        assert "Copied" in result
        assert (workspace / "original.txt").exists()
        assert (workspace / "copy.txt").read_text(encoding='utf-8') == "original content"


# ==================== WORKSPACE INFO TESTS ====================

class TestWorkspaceInfoTool:
    """Tests for WorkspaceInfoTool."""
    
    @pytest.mark.asyncio
    async def test_workspace_info(self, file_tools, file_manager):
        """Should show workspace information."""
        result = await file_tools["info"].execute()
        
        assert "Workspace" in result
        assert str(file_manager.workspace) in result
    
    @pytest.mark.asyncio
    async def test_workspace_tree(self, file_tools, file_manager):
        """Should show directory tree when requested."""
        workspace = file_manager.workspace
        
        (workspace / "doc.md").write_text("doc", encoding='utf-8')
        (workspace / "projects" / "proj1").mkdir(parents=True, exist_ok=True)
        (workspace / "projects" / "proj1" / "main.py").write_text("# main", encoding='utf-8')
        
        result = await file_tools["info"].execute(show_tree=True)
        
        assert "doc.md" in result
        assert "main.py" in result


# ==================== SAFETY TESTS ====================

class TestSafetyControls:
    """Tests for safety controls."""
    
    @pytest.mark.asyncio
    async def test_protected_paths_blocked(self, file_tools):
        """Should block access to protected system paths."""
        result = await file_tools["read"].execute("/etc/shadow")
        
        assert "denied" in result.lower() or "protected" in result.lower() or "error" in result.lower()
    
    @pytest.mark.asyncio
    async def test_sandbox_mode_enforcement(self, file_manager):
        """Should enforce sandbox mode restrictions."""
        from tools.native_files import AccessMode

        file_manager.config.access_mode = AccessMode.SANDBOX

        resolved, is_workspace, error = file_manager.resolve_path("/tmp/external.txt")

        assert error != ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("escape_path", [
        "../escape.txt",
        "../../escape.txt",
        "../../../../../../etc/passwd",
        "..\\..\\escape.txt",
        "subdir/../../escape.txt",
    ])
    async def test_relative_traversal_cannot_escape_sandbox(self, file_manager, escape_path):
        """Relative paths must not escape the workspace in sandbox mode.

        Regression: resolve_path() used to hardcode is_in_workspace=True on the
        relative branch, which disabled both the traversal and sandbox guards.
        """
        from tools.native_files import AccessMode

        file_manager.config.access_mode = AccessMode.SANDBOX

        resolved, is_workspace, error = file_manager.resolve_path(escape_path)

        assert error != "", f"{escape_path!r} escaped the sandbox (resolved to {resolved})"
        assert is_workspace is False

    @pytest.mark.asyncio
    async def test_workspace_relative_paths_still_allowed(self, file_manager):
        """The traversal fix must not block legitimate in-workspace paths."""
        from tools.native_files import AccessMode

        file_manager.config.access_mode = AccessMode.SANDBOX

        for ok_path in ("notes.md", "subdir/ok.txt", "subdir/../notes.md"):
            resolved, is_workspace, error = file_manager.resolve_path(ok_path)
            assert error == "", f"{ok_path!r} was wrongly blocked: {error}"
            assert is_workspace is True


    @pytest.mark.asyncio
    async def test_blocked_extensions(self, file_tools, file_manager):
        """Should block writing dangerous file extensions."""
        result = await file_tools["write"].execute(
            "malware.exe",
            "binary content",
            confirm=True,
        )
        
        assert "blocked" in result.lower() or "error" in result.lower() or "cannot" in result.lower()


# ==================== INTEGRATION TESTS ====================

class TestNativeFilesIntegration:
    """Integration tests for native file tools."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, file_tools, file_manager):
        """Test complete file workflow."""
        workspace = file_manager.workspace
        
        # Write a file
        await file_tools["write"].execute("workflow.txt", "Initial content", confirm=True)
        
        # Read it back
        content = await file_tools["read"].execute("workflow.txt")
        assert "Initial content" in content
        
        # Append to it
        await file_tools["write"].execute("workflow.txt", "\nMore content", mode="append", confirm=True)
        
        # Verify append
        content = await file_tools["read"].execute("workflow.txt")
        assert "More content" in content
        
        # Copy it
        await file_tools["copy"].execute("workflow.txt", "workflow_copy.txt", confirm=True)
        
        # List directory
        listing = await file_tools["list"].execute(".")
        assert "workflow.txt" in listing
        assert "workflow_copy.txt" in listing


# ==================== RUN ====================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", 
         "--tb=short", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)