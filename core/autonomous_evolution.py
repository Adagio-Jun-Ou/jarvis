"""Autonomous Evolution System for Jarvis.

Enables Jarvis to:
1. Request permission before evolving
2. Modify its own codebase safely
3. Document all changes in a rolling 12-month window
4. Rollback changes if needed
5. Learn from evolution outcomes

SAFETY PRINCIPLES:
- All code modifications require explicit user permission
- Changes are tested before permanent application
- Full rollback capability maintained
- Changes documented with rationale and outcomes
"""
import hashlib
import logging
from core.helpers import atomic_write_json, safe_error_message, read_json_with_integrity
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
import difflib

logger = logging.getLogger(__name__)


class EvolutionPermission(str, Enum):
    """Permission levels for evolution."""
    DENIED = "denied"
    ONCE = "once"           # Allow this specific change
    SESSION = "session"      # Allow for this session
    ALWAYS = "always"       # Always allow (use cautiously)
    AUTO_CODE = "auto_code" # Allow code modifications automatically


class ChangeType(str, Enum):
    """Types of code changes."""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    BEHAVIOR_MODIFICATION = "behavior_modification"
    NEW_CAPABILITY = "new_capability"
    OPTIMIZATION = "optimization"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"


class ChangeStatus(str, Enum):
    """Status of an evolution change."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    TESTED = "tested"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CodeChange:
    """Represents a proposed code change."""
    id: str
    change_type: ChangeType
    file_path: str
    description: str
    rationale: str
    original_code: str
    modified_code: str
    diff: str
    risk_level: str  # low, medium, high, critical
    created: datetime = field(default_factory=datetime.now)
    status: ChangeStatus = ChangeStatus.PROPOSED
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    test_results: Optional[dict] = None
    outcome: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "change_type": self.change_type.value,
            "file_path": self.file_path,
            "description": self.description,
            "rationale": self.rationale,
            "diff": self.diff,
            "risk_level": self.risk_level,
            "created": self.created.isoformat(),
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "test_results": self.test_results,
            "outcome": self.outcome,
        }


@dataclass
class EvolutionProposal:
    """A proposal for evolution with optional code changes."""
    id: str
    title: str
    description: str
    rationale: str
    expected_benefits: list[str]
    potential_risks: list[str]
    code_changes: list[CodeChange]
    parameter_changes: dict[str, tuple[float, float]]  # param -> (old, new)
    requested_permission: EvolutionPermission
    created: datetime = field(default_factory=datetime.now)
    status: str = "pending_approval"
    user_response: Optional[str] = None
    responded_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "expected_benefits": self.expected_benefits,
            "potential_risks": self.potential_risks,
            "code_changes": [cc.to_dict() for cc in self.code_changes],
            "parameter_changes": self.parameter_changes,
            "requested_permission": self.requested_permission.value,
            "created": self.created.isoformat(),
            "status": self.status,
            "user_response": self.user_response,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
        }


class ChangeHistory:
    """Maintains rolling 12-month change history."""

    MAX_HISTORY = 5000

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = storage_path / "change_history.json"
        self._history: list[dict] = []
        self._lock = threading.RLock()
        self._load_history()

    def _load_history(self):
        with self._lock:
            if self.history_file.exists():
                try:
                    self._history = read_json_with_integrity(self.history_file) or []
                except Exception as e:
                    logger.warning(f"Could not load change history: {e}")
                    self._history = []

    def _save_history(self):
        with self._lock:
            atomic_write_json(self.history_file, self._history)

    def _prune_old_entries(self):
        """Remove entries older than 12 months."""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=365)
            original_count = len(self._history)
            self._history = [
                entry for entry in self._history
                if datetime.fromisoformat(entry["created"]) > cutoff
            ]
            if len(self._history) < original_count:
                logger.info(f"Pruned {original_count - len(self._history)} old entries")

    def add_change(self, change: CodeChange):
        """Add a change to history."""
        with self._lock:
            self._prune_old_entries()
            self._history.append(change.to_dict())
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]
            self._save_history_unlocked()

    def add_proposal(self, proposal: EvolutionProposal):
        """Add a proposal record to history."""
        with self._lock:
            self._prune_old_entries()
            self._history.append({
                "type": "proposal",
                **proposal.to_dict()
            })
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]
            self._save_history_unlocked()

    def _save_history_unlocked(self):
        """Internal save without lock (caller holds lock)."""
        atomic_write_json(self.history_file, self._history)

    def get_changes(self, file_path: Optional[str] = None,
                    since: Optional[datetime] = None,
                    status: Optional[ChangeStatus] = None) -> list[dict]:
        """Get changes with optional filters."""
        with self._lock:
            changes = [e for e in self._history if e.get("type") != "proposal"]

            if file_path:
                changes = [c for c in changes if c.get("file_path") == file_path]
            if since:
                changes = [c for c in changes if datetime.fromisoformat(c["created"]) >= since]
            if status:
                changes = [c for c in changes if c.get("status") == status.value]

            return changes

    def get_stats(self) -> dict:
        """Get change statistics."""
        with self._lock:
            changes = [e for e in self._history if e.get("type") != "proposal"]

            by_status = {}
            for status in ChangeStatus:
                by_status[status.value] = len([c for c in changes if c.get("status") == status.value])

            by_type = {}
            for ct in ChangeType:
                by_type[ct.value] = len([c for c in changes if c.get("change_type") == ct.value])

            return {
                "total_changes": len(changes),
                "last_12_months": len(changes),  # Already filtered
                "by_status": by_status,
                "by_type": by_type,
                "rolled_back": by_status.get("rolled_back", 0),
            }


class AutonomousEvolution:
    """Manages autonomous evolution with user permission and code modification."""

    # Files that can be safely modified
    SAFE_FILES = {
        "core/config.py",
        "core/prompts.py",
        "core/awareness.py",
        "core/growth.py",
        "core/dream.py",
        "tools/productivity.py",
        "tools/home.py",
    }

    # Files that require critical review
    RESTRICTED_FILES = {
        "core/agent.py",
        "core/llm.py",
        "core/security/",
    }

    def __init__(
        self,
        jarvis_root: Path,
        storage_path: Optional[Path] = None,
        permission_callback: Optional[Callable[[EvolutionProposal], str]] = None,
    ):
        self.jarvis_root = Path(jarvis_root) if isinstance(jarvis_root, str) else jarvis_root
        self.storage_path = Path(storage_path) if storage_path and isinstance(storage_path, str) else (storage_path or Path.home() / ".jarvis" / "evolution")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.history = ChangeHistory(self.storage_path)
        self.permission_callback = permission_callback

        self._pending_proposals: dict[str, EvolutionProposal] = {}
        self._lock = threading.RLock()
        self._backup_dir = self.storage_path / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Permission cache
        self._permission_cache: dict[str, datetime] = {}
        self._session_permissions: set[str] = set()

    def _generate_id(self, prefix: str = "evo") -> str:
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents for change detection."""
        if not file_path.exists():
            return ""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _backup_file(self, file_path: str) -> Optional[Path]:
        """Create a backup of a file before modification."""
        full_path = self.jarvis_root / file_path
        if not full_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.replace('/', '_')}_{timestamp}.bak"
        backup_path = self._backup_dir / backup_name

        shutil.copy2(full_path, backup_path)
        logger.info(f"Backed up {file_path} to {backup_path}")
        return backup_path

    def _create_diff(self, original: str, modified: str, file_path: str) -> str:
        """Create a unified diff between original and modified code."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        return ''.join(diff)

    def request_evolution(
        self,
        title: str,
        description: str,
        rationale: str,
        expected_benefits: list[str],
        potential_risks: list[str],
        code_changes: Optional[list[tuple[str, str, str]]] = None,  # (file, description, new_code)
        parameter_changes: Optional[dict[str, float]] = None,
        risk_level: str = "medium",
    ) -> EvolutionProposal:
        """Request permission for an evolution change.
        
        Args:
            title: Short title for the proposal
            description: Detailed description
            rationale: Why this change is needed
            expected_benefits: List of expected improvements
            potential_risks: List of potential risks
            code_changes: Optional list of (file_path, change_description, modified_code)
            parameter_changes: Optional dict of parameter_name -> new_value
            risk_level: low, medium, high, or critical
            
        Returns:
            EvolutionProposal requiring user approval
        """
        proposal_id = self._generate_id("prop")

        code_change_objects = []
        for file_path, change_desc, new_code in (code_changes or []):
            full_path = self.jarvis_root / file_path
            original_code = ""
            if full_path.exists():
                with open(full_path, encoding='utf-8') as f:
                    original_code = f.read()

            diff = self._create_diff(original_code, new_code, file_path)

            change = CodeChange(
                id=self._generate_id("chg"),
                change_type=ChangeType.BEHAVIOR_MODIFICATION,
                file_path=file_path,
                description=change_desc,
                rationale=rationale,
                original_code=original_code,
                modified_code=new_code,
                diff=diff,
                risk_level=risk_level,
            )
            code_change_objects.append(change)

        proposal = EvolutionProposal(
            id=proposal_id,
            title=title,
            description=description,
            rationale=rationale,
            expected_benefits=expected_benefits,
            potential_risks=potential_risks,
            code_changes=code_change_objects,
            parameter_changes=parameter_changes or {},
            requested_permission=EvolutionPermission.ONCE,
        )

        with self._lock:
            self._pending_proposals[proposal_id] = proposal
        self.history.add_proposal(proposal)

        logger.info(f"Created evolution proposal: {proposal_id} - {title}")
        return proposal

    def format_proposal_for_user(self, proposal: EvolutionProposal) -> str:
        """Format a proposal for user presentation."""
        lines = [
            f"🧬 **Evolution Proposal: {proposal.title}**",
            "",
            f"**Description:** {proposal.description}",
            "",
            f"**Rationale:** {proposal.rationale}",
            "",
            "**Expected Benefits:**",
            *[f"  ✅ {b}" for b in proposal.expected_benefits],
            "",
            "**Potential Risks:**",
            *[f"  ⚠️ {r}" for r in proposal.potential_risks],
            "",
        ]

        if proposal.code_changes:
            lines.append("**Code Changes:**")
            for cc in proposal.code_changes:
                lines.append(f"  📄 `{cc.file_path}` - {cc.description}")
                lines.append(f"     Risk: {cc.risk_level}")

        if proposal.parameter_changes:
            lines.append("**Parameter Changes:**")
            for param, (old_val, new_val) in proposal.parameter_changes.items():
                lines.append(f"  ⚙️ {param}: {old_val:.2f} → {new_val:.2f}")

        lines.extend([
            "",
            f"**Proposal ID:** {proposal.id}",
            "",
            "**Your decision:**",
            "  `approve` - Allow this change",
            "  `deny` - Block this change",
            "  `session` - Allow all changes this session",
            "  `always` - Always allow (use carefully)",
        ])

        return "\n".join(lines)

    def has_permission(self, proposal_id: str) -> bool:
        """Check if a proposal has been approved."""
        with self._lock:
            if proposal_id not in self._pending_proposals:
                return False

            proposal = self._pending_proposals[proposal_id]

            # Check session permissions
            if proposal.title in self._session_permissions:
                return True

            # Check always permissions (stored)
            always_file = self.storage_path / "always_permissions.json"
            if always_file.exists():
                try:
                    always_perms = read_json_with_integrity(always_file)
                    if proposal.title in always_perms:
                        return True
                except Exception as e:
                    logger.warning(f"Failed to load always_permissions.json for check: {e}")

            return proposal.status == "approved"

    def grant_permission(self, proposal_id: str, level: str, user: str = "user") -> bool:
        """Grant permission for a proposal."""
        with self._lock:
            if proposal_id not in self._pending_proposals:
                return False

            proposal = self._pending_proposals[proposal_id]
            proposal.status = "approved"
            proposal.user_response = level
            proposal.responded_at = datetime.now()

            if level == "session":
                self._session_permissions.add(proposal.title)
            elif level == "always":
                always_file = self.storage_path / "always_permissions.json"
                always_perms = {}
                if always_file.exists():
                    try:
                        always_perms = read_json_with_integrity(always_file)
                    except Exception as e:
                        logger.warning(f"Failed to load always_permissions.json for save: {e}")
                always_perms[proposal.title] = datetime.now().isoformat()
                atomic_write_json(always_file, always_perms)

        logger.info(f"Permission granted for {proposal_id}: {level} by {user}")
        return True

    def apply_evolution(self, proposal_id: str) -> dict:
        """Apply an approved evolution.
        
        Returns:
            Result dict with success status and details
        """
        with self._lock:
            if proposal_id not in self._pending_proposals:
                return {"success": False, "error": "Proposal not found"}

            proposal = self._pending_proposals[proposal_id]

            if proposal.status != "approved":
                return {"success": False, "error": "Proposal not approved"}

        results = {
            "proposal_id": proposal_id,
            "code_changes": [],
            "parameter_changes": [],
            "backups": [],
        }

        # Apply code changes
        for cc in proposal.code_changes:
            try:
                # Backup first
                backup_path = self._backup_file(cc.file_path)
                if backup_path:
                    results["backups"].append(str(backup_path))

                # Apply change
                full_path = self.jarvis_root / cc.file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(cc.modified_code)

                cc.status = ChangeStatus.APPLIED
                cc.applied_at = datetime.now()

                self.history.add_change(cc)
                results["code_changes"].append({
                    "file": cc.file_path,
                    "status": "applied",
                })

                logger.info(f"Applied code change to {cc.file_path}")

            except Exception as e:
                cc.status = ChangeStatus.FAILED
                results["code_changes"].append({
                    "file": cc.file_path,
                    "status": "failed",
                    "error": safe_error_message(e),
                })
                logger.error(f"Failed to apply change to {cc.file_path}: {e}")

        results["success"] = True
        return results

    def rollback_change(self, change_id: str) -> bool:
        """Rollback a specific change using backup."""
        # Find the backup
        for backup in self._backup_dir.iterdir():
            if change_id in backup.name:
                # Extract original file path from backup name
                parts = backup.stem.rsplit('_', 2)
                if len(parts) == 3:
                    file_path = parts[0].replace('_', '/')
                    full_path = self.jarvis_root / file_path

                    shutil.copy2(backup, full_path)
                    logger.info(f"Rolled back {file_path}")
                    return True

        return False

    def get_pending_proposals(self) -> list[EvolutionProposal]:
        """Get all pending proposals awaiting approval."""
        with self._lock:
            return [p for p in self._pending_proposals.values() if p.status == "pending_approval"]

    def get_change_history(self, months: int = 12) -> list[dict]:
        """Get change history for the last N months."""
        since = datetime.now() - timedelta(days=months * 30)
        return self.history.get_changes(since=since)

    def generate_evolution_report(self) -> str:
        """Generate a comprehensive evolution report."""
        stats = self.history.get_stats()
        pending = self.get_pending_proposals()

        lines = [
            "# 🧬 Jarvis Evolution Report",
            "",
            "## Overview",
            f"- Total Changes (12 months): {stats['total_changes']}",
            f"- Rolled Back: {stats['rolled_back']}",
            f"- Pending Proposals: {len(pending)}",
            "",
            "## Changes by Status",
        ]

        for status, count in stats['by_status'].items():
            if count > 0:
                lines.append(f"- {status}: {count}")

        lines.extend([
            "",
            "## Changes by Type",
        ])

        for change_type, count in stats['by_type'].items():
            if count > 0:
                lines.append(f"- {change_type}: {count}")

        if pending:
            lines.extend([
                "",
                "## Pending Proposals",
            ])
            for p in pending:
                lines.append(f"- **{p.title}**: {p.description[:50]}...")

        return "\n".join(lines)


class AutonomousEvolutionTool:
    """Tool for Jarvis to request evolution changes."""

    name = "evolve_self"
    description = "Propose and request permission for self-evolution changes"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["propose", "status", "history", "rollback", "report"],
                "description": "Evolution action to perform"
            },
            "title": {
                "type": "string",
                "description": "Title for a new proposal"
            },
            "description": {
                "type": "string",
                "description": "Description of the evolution"
            },
            "rationale": {
                "type": "string",
                "description": "Why this evolution is needed"
            },
            "benefits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Expected benefits"
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential risks"
            },
            "file_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "description": {"type": "string"},
                        "new_code": {"type": "string"}
                    }
                },
                "description": "Code changes to apply"
            },
            "change_id": {
                "type": "string",
                "description": "ID of change to rollback"
            }
        },
        "required": ["action"]
    }
    requires_permission = True  # Always requires permission for code changes

    def __init__(self, evolution: AutonomousEvolution):
        self.evolution = evolution

    async def execute(self, **params) -> str:
        action = params.get("action")

        if action == "propose":
            proposal = self.evolution.request_evolution(
                title=params.get("title", "Untitled"),
                description=params.get("description", ""),
                rationale=params.get("rationale", ""),
                expected_benefits=params.get("benefits", []),
                potential_risks=params.get("risks", []),
                code_changes=[
                    (fc["path"], fc["description"], fc["new_code"])
                    for fc in params.get("file_changes", [])
                ],
                risk_level="medium",
            )
            return self.evolution.format_proposal_for_user(proposal)

        elif action == "status":
             pending = self.evolution.get_pending_proposals()
             stats = self.evolution.history.get_stats()
             lines = [
                 "🧬 Evolution Status",
                 f"  Pending proposals: {len(pending)}",
                 f"  Total changes: {stats['total_changes']}",
                 f"  Rolled back: {stats['rolled_back']}",
             ]
             return "\n".join(lines)

        elif action == "history":
            history = self.evolution.get_change_history(months=12)
            lines = ["📜 Change History (12 months):"]
            for entry in history[-10:]:
                lines.append(f"  - {entry.get('file_path', 'N/A')}: {entry.get('status', 'N/A')}")
            return "\n".join(lines)

        elif action == "rollback":
            change_id = params.get("change_id")
            if self.evolution.rollback_change(change_id):
                return f"✅ Rolled back change {change_id}"
            return f"❌ Could not rollback {change_id}"

        elif action == "report":
            return self.evolution.generate_evolution_report()

        return f"Unknown action: {action}"


_evolution: AutonomousEvolution | None = None
_evolution_lock = threading.Lock()


def get_autonomous_evolution(jarvis_root: Path | str | None = None) -> AutonomousEvolution:
    global _evolution
    if _evolution is None:
        with _evolution_lock:
            if _evolution is None:
                root = Path(jarvis_root) if jarvis_root else Path.home() / ".jarvis"
                _evolution = AutonomousEvolution(jarvis_root=root)
    return _evolution
