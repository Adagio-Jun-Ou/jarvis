"""Self-Edit System - Jarvis can propose and execute code modifications.

Like opencode, Jarvis can suggest code changes but must get user permission
before executing them. All modifications are logged and can be rolled back.
"""
import ast
import asyncio
import difflib
import logging
import threading
from collections import deque
from core.helpers import atomic_write_json, read_json_with_integrity
from core.safe_files import (
    CAUTION_PATTERN_SOURCES,
    CRITICAL_FILES as REGISTRY_CRITICAL_FILES,
    NEVER_SAFE_PATTERN_SOURCES,
    SAFE_EDIT_FILES,
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class EditStatus(Enum):
    PROPOSED = "proposed"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class EditType(Enum):
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    FEATURE = "feature"
    REFACTOR = "refactor"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    EVOLUTION = "evolution"
    USER_REQUEST = "user_request"


class TOCTOUError(Exception):
    """Raised when a file changes between read and write (time-of-check/time-of-use)."""
    pass


@dataclass
class CodeEdit:
    id: str
    edit_type: EditType
    file_path: str
    description: str
    rationale: str
    old_code: str
    new_code: str
    status: EditStatus = EditStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: datetime | None = None
    applied_by: str = "jarvis"
    rollback_data: str | None = None
    impact_level: str = "low"  # low, medium, high, critical
    affected_capabilities: list[str] = field(default_factory=list)
    test_required: bool = True
    tests_passed: bool | None = None


@dataclass
class EditBatch:
    id: str
    description: str
    edits: list[CodeEdit]
    status: EditStatus = EditStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    reasoning: str = ""


class SelfEditEngine:
    """Engine for proposing and executing code modifications.
    
    Safety features:
    - Low-risk edits can be auto-approved with user consent
    - Medium/high-risk edits require explicit user permission
    - Original code is backed up
    - Edits can be rolled back
    - Syntax validation before applying
    - Test requirements tracked
    
    Permission Levels:
    - AUTO_APPROVE: Low-impact, safe changes (typo fixes, comments, logging)
    - QUICK_APPROVE: Moderate changes (refactors, optimizations) - one-click approve
    - FULL_REVIEW: Significant changes (features, bug fixes) - detailed review
    - NEVER_AUTO: High-risk changes - always require explicit approval
    """

    # Self-modification policy lives in core/safe_files.py, the single registry
    # shared by self-edit, self-refine and evolution. Referenced here rather
    # than duplicated: the two copies had drifted and disagreed about which
    # files were editable.
    SAFE_FILES = SAFE_EDIT_FILES

    # Files that should NEVER be auto-edited due to security implications
    CRITICAL_FILES = REGISTRY_CRITICAL_FILES

    # Patterns that are NEVER safe - will block any edit containing these
    NEVER_SAFE_PATTERNS = [src for src, _desc in NEVER_SAFE_PATTERN_SOURCES]

    # Patterns that require extra scrutiny but aren't blocked
    CAUTION_PATTERNS = [src for src, _desc in CAUTION_PATTERN_SOURCES]

    # Edit types that can be auto-approved (with user's consent setting)
    AUTO_APPROVABLE_TYPES = {
        EditType.DOCUMENTATION,      # Comments, docstrings
        EditType.REFACTOR,           # Clean, safe refactors
        EditType.OPTIMIZATION,       # Performance improvements
    }

    # Impact levels that can be auto-approved
    AUTO_APPROVABLE_IMPACTS = {"low"}

    MAX_FILE_SIZE = 400000  # accommodate larger files like web_react.py
    MAX_EDITS_PER_DAY = 50   # Increased from 20
    
    # User preference for auto-approval
    AUTO_APPROVE_ENABLED = False  # Can be toggled via API - disabled by default for security
    AUTO_APPROVE_REQUIRE_TEST = True  # If True, tests must pass for auto-approve

    def __init__(self, jarvis_root: Path | None = None):
        self.jarvis_root = jarvis_root or Path.cwd()
        self.edits_dir = self.jarvis_root / ".jarvis" / "edits"
        self.edits_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._edit_counter = 0
        self.MAX_PENDING_EDITS = 100
        self.MAX_APPLIED_EDITS = 100
        self.MAX_EDIT_HISTORY = 200
        self.pending_edits: deque[CodeEdit] = deque(maxlen=100)
        self.applied_edits: deque[CodeEdit] = deque(maxlen=100)
        self.edit_history: deque[dict] = deque(maxlen=200)
        self._state_lock = threading.Lock()

        self._permission_callback: Callable | None = None
        self._apply_callback: Callable | None = None

        self._load_state()

    def _load_state(self) -> None:
        state_file = self.edits_dir / "edit_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._edit_counter = data.get("counter", 0)
            except Exception as e:
                logger.debug(f"Could not load edit state: {e}")

    def _save_state(self) -> None:
        state_file = self.edits_dir / "edit_state.json"
        with self._lock:
            counter = self._edit_counter
            total_applied = len(self.applied_edits)
        try:
            atomic_write_json(state_file, {
                    "counter": counter,
                    "total_applied": total_applied,
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Could not save edit state: {e}")

    def set_callbacks(
        self,
        permission_callback: Callable | None = None,
        apply_callback: Callable | None = None,
    ) -> None:
        self._permission_callback = permission_callback
        self._apply_callback = apply_callback

    def can_edit_file(self, file_path: str, check_critical: bool = True) -> tuple[bool, str]:
        """Check if a file can be safely edited.
        
        Args:
            file_path: Path to the file to edit
            check_critical: If True, reject edits to critical files
            
        Returns:
            Tuple of (can_edit, reason_message)
        """
        rel_path = file_path.replace(str(self.jarvis_root) + "/", "")
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]
        
        while "../" in rel_path or "..\\" in rel_path:
            rel_path = rel_path.replace("../", "").replace("..\\", "")
        
        if "\\" in rel_path:
            rel_path = rel_path.replace("\\", "/")
        
        while "//" in rel_path:
            rel_path = rel_path.replace("//", "/")
        
        resolved_path = (self.jarvis_root / rel_path).resolve()
        try:
            resolved_path.relative_to(self.jarvis_root.resolve())
        except ValueError:
            return False, f"Path traversal attempt blocked: {rel_path}"

        if check_critical and rel_path in self.CRITICAL_FILES:
            return False, f"File is critical and requires manual review: {rel_path}"

        if rel_path not in self.SAFE_FILES:
            full_path = self.jarvis_root / rel_path
            if not full_path.exists():
                parent_dir = str(full_path.parent.relative_to(self.jarvis_root)) if full_path.parent != self.jarvis_root else ""
                new_file_dirs = {"core/evolution_patches", "tools/custom"}
                if parent_dir in new_file_dirs:
                    import re
                    if re.match(r'^[a-z][a-z0-9_]+\.py$', full_path.name):
                        return True, f"New file in sandboxed directory: {rel_path}"
                return False, f"New file creation not allowed: {rel_path} (only files in SAFE_FILES or sandboxed dirs)"
            return False, f"File not in safe list: {rel_path}"

        full_path = self.jarvis_root / rel_path
        if not full_path.exists():
            return False, f"File does not exist: {rel_path}"

        if full_path.stat().st_size > self.MAX_FILE_SIZE:
            return False, f"File too large: {full_path.stat().st_size} bytes"

        return True, "File is editable"

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate code for syntax and safety.
        
        Returns:
            Tuple of (is_valid, reason_message)
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        import re
        
        # Check for never-safe patterns (block completely)
        # Use re.DOTALL so patterns like .* match across newlines
        for pattern in self.NEVER_SAFE_PATTERNS:
            if re.search(pattern, code, re.DOTALL):
                return False, f"Blocked pattern detected: {pattern}"

        # Check for caution patterns (warn but allow)
        caution_found = []
        for pattern in self.CAUTION_PATTERNS:
            if re.search(pattern, code):
                caution_found.append(pattern)

        if caution_found:
            # Log a warning but don't block
            logger.warning(f"Caution patterns found in code: {caution_found}")

        return True, "Code is valid"

    def can_auto_approve(self, edit: CodeEdit) -> tuple[bool, str]:
        """Check if an edit can be auto-approved.
        
        Auto-approval requires:
        1. User has enabled auto-approval
        2. Edit type is auto-approvable (docs, refactor, optimization)
        3. Impact level is low
        4. Code passes validation
        5. File is not critical
        6. No tests required OR tests have passed
        
        Returns:
            Tuple of (can_auto_approve, reason)
        """
        if not self.AUTO_APPROVE_ENABLED:
            return False, "Auto-approval is disabled"
        
        # Check if edit type is auto-approvable
        if edit.edit_type not in self.AUTO_APPROVABLE_TYPES:
            return False, f"Edit type '{edit.edit_type.value}' requires manual review"
        
        # Check impact level
        if edit.impact_level not in self.AUTO_APPROVABLE_IMPACTS:
            return False, f"Impact level '{edit.impact_level}' requires manual review"
        
        # Check if tests are required
        if self.AUTO_APPROVE_REQUIRE_TEST and edit.test_required:
            if edit.tests_passed is not True:
                return False, "Tests must pass for auto-approval"
        
        # Validate the new code
        valid, msg = self.validate_code(edit.new_code)
        if not valid:
            return False, f"Code validation failed: {msg}"
        
        # Check file permissions (allow non-critical files)
        can_edit, msg = self.can_edit_file(edit.file_path, check_critical=True)
        if not can_edit:
            return False, f"File permission denied: {msg}"
        
        # Check if edit contains any never-safe patterns in the diff
        # Use re.DOTALL to catch patterns split across lines
        import re
        for pattern in self.NEVER_SAFE_PATTERNS:
            if re.search(pattern, edit.new_code, re.DOTALL):
                return False, f"Edit contains blocked pattern: {pattern}"
        
        return True, "Edit eligible for auto-approval"

    def get_permission_level(self, edit: CodeEdit) -> str:
        """Get the required permission level for an edit.
        
        Returns:
            One of: 'AUTO_APPROVE', 'QUICK_APPROVE', 'FULL_REVIEW', 'NEVER_AUTO'
        """
        can_auto, _ = self.can_auto_approve(edit)
        if can_auto:
            return "AUTO_APPROVE"
        
        # Critical files always need full review
        rel_path = edit.file_path
        if rel_path in self.CRITICAL_FILES:
            return "NEVER_AUTO"
        
        # High/critical impact needs full review
        if edit.impact_level in ("high", "critical"):
            return "FULL_REVIEW"
        
        # Bug fixes and features need full review
        if edit.edit_type in (EditType.BUG_FIX, EditType.FEATURE, EditType.SECURITY):
            return "FULL_REVIEW"
        
        # Everything else is quick approve
        return "QUICK_APPROVE"

    def propose_edit(
        self,
        file_path: str,
        old_code: str,
        new_code: str,
        edit_type: EditType,
        description: str,
        rationale: str,
        impact_level: str = "low",
    ) -> CodeEdit | None:
        """Propose a code edit. Returns the edit or None if validation fails."""

        can_edit, msg = self.can_edit_file(file_path)
        if not can_edit:
            logger.warning(f"Cannot edit {file_path}: {msg}")
            return None

        valid, msg = self.validate_code(new_code)
        if not valid:
            logger.warning(f"Invalid code: {msg}")
            return None

        rel_path = file_path.replace(str(self.jarvis_root) + "/", "")
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]

        with self._lock:
            self._edit_counter += 1
            edit_id = f"edit_{self._edit_counter:06d}"

        edit = CodeEdit(
            id=edit_id,
            edit_type=edit_type,
            file_path=rel_path,
            description=description,
            rationale=rationale,
            old_code=old_code,
            new_code=new_code,
            impact_level=impact_level,
        )

        with self._lock:
            self.pending_edits.append(edit)

        self._save_state()

        logger.info(f"Proposed edit {edit_id}: {description}")

        return edit

    def propose_batch(
        self,
        edits: list[dict],
        description: str,
        reasoning: str,
    ) -> EditBatch:
        """Propose a batch of related edits."""

        with self._lock:
            self._edit_counter += 1
            batch_id = f"batch_{self._edit_counter:06d}"

        code_edits = []
        for e in edits:
            edit = self.propose_edit(
                file_path=e["file_path"],
                old_code=e["old_code"],
                new_code=e["new_code"],
                edit_type=EditType(e.get("edit_type", "evolution")),
                description=e.get("description", ""),
                rationale=e.get("rationale", ""),
                impact_level=e.get("impact_level", "low"),
            )
            if edit:
                code_edits.append(edit)

        batch = EditBatch(
            id=batch_id,
            description=description,
            edits=code_edits,
            reasoning=reasoning,
        )

        return batch

    def get_pending_edits(self) -> list[dict]:
        """Get all pending edits awaiting approval."""
        with self._lock:
            return [
                {
                    "id": e.id,
                    "type": e.edit_type.value,
                    "file": e.file_path,
                    "description": e.description,
                    "rationale": e.rationale,
                    "impact": e.impact_level,
                    "diff": self._generate_diff(e),
                }
                for e in self.pending_edits
            ]

    def _generate_diff(self, edit: CodeEdit) -> str:
        """Generate a unified diff for the edit."""
        old_lines = edit.old_code.splitlines(keepends=True)
        new_lines = edit.new_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{edit.file_path} (original)",
            tofile=f"{edit.file_path} (modified)",
        )

        return "".join(diff)

    async def request_permission(self, edit_id: str) -> bool:
        """Request user permission for an edit."""
        edit = self._get_edit(edit_id)
        if not edit:
            return False

        if self._permission_callback:
            return await self._permission_callback(edit)

        return False

    def _get_edit(self, edit_id: str) -> CodeEdit | None:
        """Get an edit by ID from pending or applied lists."""
        with self._lock:
            for e in self.pending_edits:
                if e.id == edit_id:
                    return e
            for e in self.applied_edits:
                if e.id == edit_id:
                    return e
        return None

    def approve_edit(self, edit_id: str) -> bool:
        """Mark an edit as approved (user consent)."""
        with self._lock:
            edit = self._get_edit(edit_id)
            if not edit:
                return False
            edit.status = EditStatus.APPROVED
            logger.info(f"Edit {edit_id} approved")
            return True

    def reject_edit(self, edit_id: str, reason: str = "") -> bool:
        """Reject a proposed edit."""
        needs_save = False
        with self._lock:
            for e in self.pending_edits:
                if e.id == edit_id:
                    e.status = EditStatus.REJECTED
                    self.edit_history.append({
                        "id": edit_id,
                        "action": "rejected",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat(),
                    })
                    self.pending_edits.remove(e)
                    needs_save = True
                    logger.info(f"Edit {edit_id} rejected: {reason}")
                    break
        if needs_save:
            self._save_state()
            return True
        return False

    async def apply_edit(self, edit_id: str) -> tuple[bool, str]:
        """Apply an approved edit to the codebase.
        
        Uses file hash verification to detect changes between read and write
        (TOCTOU protection). If hash doesn't match, the edit is rejected.
        """
        edit = self._get_edit(edit_id)
        if not edit:
            return False, f"Edit {edit_id} not found"

        if edit.status != EditStatus.APPROVED:
            return False, f"Edit {edit_id} not approved (status: {edit.status.value})"

        # Re-validate file policy at apply time (TOCTOU protection)
        can_edit, reason = self.can_edit_file(edit.file_path)
        if not can_edit:
            edit.status = EditStatus.FAILED
            return False, f"File policy changed since proposal: {reason}"

        full_path = self.jarvis_root / edit.file_path

        try:
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(self.jarvis_root.resolve())):
                edit.status = EditStatus.FAILED
                return False, "Resolved path escapes project root (possible symlink attack)"
        except Exception as e:
            edit.status = EditStatus.FAILED
            return False, f"Path resolution failed: {e}"

        backup_file = None

        try:
            original_content, original_hash = await asyncio.to_thread(self._read_file_with_hash, full_path)

            with self._lock:
                occurrence_count = original_content.count(edit.old_code)
                if occurrence_count == 0:
                    edit.status = EditStatus.FAILED
                    return False, "Old code not found in file - file may have changed"
                if occurrence_count > 1:
                    edit.status = EditStatus.FAILED
                    return False, f"Old code found {occurrence_count} times in file - ambiguous edit, please specify more context"

            backup_file = self.edits_dir / f"{edit_id}_backup.txt"
            await asyncio.to_thread(backup_file.write_text, original_content)
            with self._lock:
                edit.rollback_data = str(backup_file)

            new_content = original_content.replace(edit.old_code, edit.new_code, 1)

            valid, msg = self.validate_code(new_content)
            if not valid:
                edit.status = EditStatus.FAILED
                if backup_file and backup_file.exists():
                    backup_file.unlink()
                return False, f"Resulting code invalid: {msg}"

            await asyncio.to_thread(self._write_file_with_verify, full_path, new_content, original_hash)

            edit.status = EditStatus.APPLIED
            edit.applied_at = datetime.now()

            with self._lock:
                if edit in self.pending_edits:
                    self.pending_edits.remove(edit)
                self.applied_edits.append(edit)

                self.edit_history.append({
                    "id": edit_id,
                    "action": "applied",
                    "file": edit.file_path,
                    "timestamp": edit.applied_at.isoformat(),
                })

            self._save_state()

            logger.info(f"Applied edit {edit_id} to {edit.file_path}")

            if self._apply_callback:
                await self._apply_callback(edit)

            return True, f"Edit applied successfully to {edit.file_path}"

        except TOCTOUError as e:
            edit.status = EditStatus.FAILED
            logger.warning(f"TOCTOU check failed for edit {edit_id}: {e}")
            return False, f"File changed during edit (TOCTOU): {e}"
        except Exception as e:
            edit.status = EditStatus.FAILED
            logger.error(f"Failed to apply edit {edit_id}: {e}")
            if backup_file and backup_file.exists():
                try:
                    backup_file.unlink()
                except Exception as err:
                    logger.warning(f"Failed to delete backup file {backup_file}: {err}")
            return False, f"Error applying edit: {e}"

    async def rollback_edit(self, edit_id: str) -> tuple[bool, str]:
        """Rollback an applied edit."""
        edit = self._get_edit(edit_id)
        if not edit:
            return False, f"Edit {edit_id} not found"

        if edit.status != EditStatus.APPLIED:
            return False, f"Edit {edit_id} not applied (status: {edit.status.value})"

        if not edit.rollback_data:
            return False, "No rollback data available"

        try:
            backup_path = Path(edit.rollback_data)
            if not backup_path.exists():
                return False, f"Backup file not found: {backup_path}"

            original_content = await asyncio.to_thread(backup_path.read_text)
            full_path = self.jarvis_root / edit.file_path

            try:
                resolved = full_path.resolve()
                if not str(resolved).startswith(str(self.jarvis_root.resolve())):
                    edit.status = EditStatus.FAILED
                    return False, "Resolved path escapes project root (possible symlink attack)"
            except Exception as e:
                edit.status = EditStatus.FAILED
                return False, f"Path resolution failed: {e}"

            await asyncio.to_thread(full_path.write_text, original_content)

            edit.status = EditStatus.ROLLED_BACK

            with self._lock:
                if edit in self.applied_edits:
                    self.applied_edits.remove(edit)

                self.edit_history.append({
                    "id": edit_id,
                    "action": "rolled_back",
                    "timestamp": datetime.now().isoformat(),
                })

            self._save_state()

            logger.info(f"Rolled back edit {edit_id}")
            return True, f"Edit {edit_id} rolled back successfully"

        except Exception as e:
            logger.error(f"Failed to rollback edit {edit_id}: {e}")
            return False, f"Error rolling back: {e}"

    def _read_file_with_hash(self, file_path: Path) -> tuple[str, str]:
        """Read file and compute hash for TOCTOU protection.
        
        Returns (content, sha256_hash).
        """
        import hashlib
        content = file_path.read_text(encoding='utf-8')
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return content, content_hash

    def _write_file_with_verify(self, file_path: Path, content: str, expected_hash: str) -> None:
        """Write file, verifying it hasn't changed since the hash was computed.
        
        Raises TOCTOUError if the file changed between read and write.
        """
        import hashlib
        if file_path.exists():
            current = file_path.read_text(encoding='utf-8')
            current_hash = hashlib.sha256(current.encode()).hexdigest()
            if current_hash != expected_hash:
                raise TOCTOUError(
                    f"File {file_path} changed between read and write. "
                    f"Expected hash {expected_hash[:12]}... got {current_hash[:12]}..."
                )

        import os
        temp_path = file_path.with_suffix('.tmp')
        temp_path.write_text(content, encoding='utf-8')
        os.replace(temp_path, file_path)

    def get_edit_stats(self) -> dict:
        """Get statistics about edits."""
        with self._lock:
            by_type: dict[str, int] = {}
            by_status: dict[str, int] = {}

            for e in self.pending_edits + self.applied_edits:
                t = e.edit_type.value
                by_type[t] = by_type.get(t, 0) + 1
                s = e.status.value
                by_status[s] = by_status.get(s, 0) + 1

            return {
                "pending": len(self.pending_edits),
                "applied": len(self.applied_edits),
                "total_history": len(self.edit_history),
                "by_type": by_type,
                "by_status": by_status,
            }


_self_edit_engine: SelfEditEngine | None = None
_self_edit_lock = threading.Lock()


def get_self_edit_engine() -> SelfEditEngine:
    global _self_edit_engine
    if _self_edit_engine is None:
        with _self_edit_lock:
            if _self_edit_engine is None:
                _self_edit_engine = SelfEditEngine()
    return _self_edit_engine
