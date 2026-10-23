"""Auto-Refine & Repair Module for Jarvis.

A self-improvement system that:
1. Scans codebase for issues, bugs, and improvement opportunities
2. Suggests fixes with explanations
3. Generates safe code changes
4. Asks for permission before applying changes
5. Tracks all modifications for rollback

Uses AI-assisted code analysis and generation for safe operations.
"""
import ast
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from core.helpers import atomic_write_json, get_response_content, safe_error_message

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    CRITICAL = "critical"      # Bugs, security issues
    HIGH = "high"              # Performance, reliability
    MEDIUM = "medium"          # Code quality, maintainability
    LOW = "low"                # Style, minor improvements
    SUGGESTION = "suggestion"  # Enhancements, optimizations


class IssueType(Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"
    DEPENDENCY = "dependency"
    DOCUMENTATION = "documentation"
    TYPE_ERROR = "type_error"
    SYNTAX = "syntax"
    IMPORT = "import"
    UNUSED = "unused"
    REFACTOR = "refactor"
    ENHANCEMENT = "enhancement"


class ChangeStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CodeIssue:
    """Represents an issue found in the codebase."""
    id: str
    file_path: str
    line_number: int
    issue_type: IssueType
    severity: IssueSeverity
    title: str
    description: str
    code_snippet: str
    suggested_fix: str | None = None
    auto_fixable: bool = False
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ProposedChange:
    """Represents a proposed code change."""
    id: str
    issue_id: str
    file_path: str
    line_number: int = 0
    severity: str = "medium"
    confidence: float = 0.5
    change_type: str = "edit"
    original_code: str = ""
    proposed_code: str = ""
    explanation: str = ""
    risk_level: str = "medium"
    status: ChangeStatus = ChangeStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: datetime | None = None
    rollback_code: str | None = None
    error_message: str | None = None


@dataclass
class ScanResult:
    """Result of a codebase scan."""
    timestamp: datetime
    files_scanned: int
    issues_found: list[CodeIssue]
    summary: dict[str, int]
    duration_seconds: float


class CodeAnalyzer:
    """Analyzes Python code for issues and improvements."""

    CRITICAL_PATTERNS = [
        (r'\bexec\s*\(', 'Potential code injection via exec()'),
        (r'\beval\s*\(', 'Potential code injection via eval()'),
        (r'__import__\s*\(', 'Dynamic import - potential security risk'),
        (r'subprocess\..*shell\s*=\s*True', 'Shell injection risk'),
        (r'pickle\.loads?\s*\(', 'Pickle deserialization risk'),
        (r'yaml\.load\s*\([^)]*\)\s*$', 'YAML unsafe load - use yaml.safe_load()'),
        (r'\.format\s*\([^)]*user', 'Potential format string injection'),
    ]

    PERFORMANCE_PATTERNS = [
        (r'for\s+\w+\s+in\s+range\(len\(', 'Use enumerate() instead of range(len())'),
        (r'\+\s*=\s*.*\s+in\s+.*for', 'String concatenation in loop - use list join'),
        (r'\.append\s*\([^)]*\)\s*$\s*for', 'Consider list comprehension'),
        (r'if\s+\w+\s+in\s+\[', 'Consider using a set for membership tests'),
        (r'except\s*:', 'Bare except - catches KeyboardInterrupt, SystemExit'),
        (r'except\s+Exception\s*:', 'Broad exception catch - be more specific'),
    ]

    CODE_QUALITY_PATTERNS = [
        (r'# TODO', 'TODO comment found'),
        (r'# FIXME', 'FIXME comment found'),
        (r'# HACK', 'HACK comment found'),
        (r'pass\s*$', 'Empty block - consider adding implementation'),
        (r'print\s*\(', 'Debug print statement'),
    ]

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.exclusions = {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules',
            '.mypy_cache', '.pytest_cache', 'build', 'dist', '*.egg-info',
            'auto_refine.py',  # Skip self - contains pattern definitions
            'self_refine.py',  # Skip self - contains pattern definitions
        }

    def should_scan(self, path: Path) -> bool:
        """Check if path should be scanned."""
        if not path.exists():
            return False

        rel_path = str(path.relative_to(self.base_path))
        for exc in self.exclusions:
            if exc in rel_path:
                return False

        if path.is_file():
            return path.suffix == '.py'

        return True

    def scan_file(self, file_path: Path) -> list[CodeIssue]:
        """Scan a single file for issues."""
        issues = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            rel_path = str(file_path.relative_to(self.base_path))

            # Syntax check
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(CodeIssue(
                    id=f"syn_{file_path.stem}_{e.lineno or 0}",
                    file_path=rel_path,
                    line_number=e.lineno or 0,
                    issue_type=IssueType.SYNTAX,
                    severity=IssueSeverity.CRITICAL,
                    title=f"Syntax Error: {e.msg}",
                    description=f"Python syntax error: {e.msg}",
                    code_snippet=lines[e.lineno - 1] if e.lineno and e.lineno <= len(lines) else "",
                    auto_fixable=False,
                    confidence=1.0,
                ))
                return issues  # Can't continue if syntax is broken

            # Check for critical patterns
            for i, line in enumerate(lines, 1):
                # Security patterns
                for pattern, msg in self.CRITICAL_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(CodeIssue(
                            id=f"sec_{file_path.stem}_{i}_{hash(pattern) % 10000}",
                            file_path=rel_path,
                            line_number=i,
                            issue_type=IssueType.SECURITY,
                            severity=IssueSeverity.CRITICAL,
                            title=f"Security: {msg}",
                            description=f"Found potentially unsafe pattern: {msg}",
                            code_snippet=line.strip()[:100],
                            suggested_fix=self._suggest_security_fix(pattern, line),
                            auto_fixable=self._is_auto_fixable(pattern),
                            confidence=0.9,
                            metadata={"pattern": pattern},
                        ))

                # Performance patterns
                for pattern, msg in self.PERFORMANCE_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(CodeIssue(
                            id=f"perf_{file_path.stem}_{i}_{hash(pattern) % 10000}",
                            file_path=rel_path,
                            line_number=i,
                            issue_type=IssueType.PERFORMANCE,
                            severity=IssueSeverity.MEDIUM,
                            title=f"Performance: {msg}",
                            description=f"Performance improvement opportunity: {msg}",
                            code_snippet=line.strip()[:100],
                            suggested_fix=self._suggest_performance_fix(pattern, line, lines, i),
                            auto_fixable=False,
                            confidence=0.7,
                            metadata={"pattern": pattern},
                        ))

                # Code quality
                for pattern, msg in self.CODE_QUALITY_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(CodeIssue(
                            id=f"qual_{file_path.stem}_{i}",
                            file_path=rel_path,
                            line_number=i,
                            issue_type=IssueType.CODE_QUALITY,
                            severity=IssueSeverity.LOW,
                            title=msg,
                            description=f"Code quality note: {msg}",
                            code_snippet=line.strip()[:100],
                            auto_fixable=False,
                            confidence=0.6,
                        ))

            # AST-based checks
            issues.extend(self._analyze_ast(file_path, content, lines))

        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")

        return issues

    def _analyze_ast(self, file_path: Path, content: str, lines: list[str]) -> list[CodeIssue]:
        """Perform AST-based analysis."""
        issues = []
        rel_path = str(file_path.relative_to(self.base_path))

        try:
            tree = ast.parse(content)

            # Track defined vs used names
            defined_names = set()
            used_names = set()
            imported_names = set()

            for node in ast.walk(tree):
                # Track imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)

                # Track function/class definitions
                elif isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                    # Check for missing docstring
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant)):
                        issues.append(CodeIssue(
                            id=f"doc_{file_path.stem}_{node.lineno}",
                            file_path=rel_path,
                            line_number=node.lineno,
                            issue_type=IssueType.DOCUMENTATION,
                            severity=IssueSeverity.LOW,
                            title="Missing docstring",
                            description=f"Function '{node.name}' has no docstring",
                            code_snippet=lines[node.lineno - 1][:100] if node.lineno <= len(lines) else "",
                            auto_fixable=False,
                            confidence=0.8,
                        ))

                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                    # Check for missing docstring
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant)):
                        issues.append(CodeIssue(
                            id=f"doc_{file_path.stem}_{node.lineno}",
                            file_path=rel_path,
                            line_number=node.lineno,
                            issue_type=IssueType.DOCUMENTATION,
                            severity=IssueSeverity.LOW,
                            title="Missing docstring",
                            description=f"Class '{node.name}' has no docstring",
                            code_snippet=lines[node.lineno - 1][:100] if node.lineno <= len(lines) else "",
                            auto_fixable=False,
                            confidence=0.8,
                        ))

                # Track unused imports
                elif isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)

            # Check for potentially unused imports
            potentially_unused = imported_names - used_names
            for name in potentially_unused:
                # Find the import line
                for i, line in enumerate(lines, 1):
                    if f"import {name}" in line or "from" in line and name in line:
                        # Skip if it's a common pattern (type hints, __all__, etc.)
                        if any(p in content for p in [f": {name}", f"-> {name}", f"[{name}", '"TYPE_CHECKING"']):
                            continue
                        issues.append(CodeIssue(
                            id=f"unused_{file_path.stem}_{i}",
                            file_path=rel_path,
                            line_number=i,
                            issue_type=IssueType.UNUSED,
                            severity=IssueSeverity.LOW,
                            title=f"Potentially unused import: {name}",
                            description=f"Import '{name}' may not be used in the file",
                            code_snippet=line.strip()[:100],
                            suggested_fix=f"# Remove or comment: {line.strip()}",
                            auto_fixable=True,
                            confidence=0.5,
                        ))
                        break

        except Exception as e:
            logger.debug(f"AST analysis failed for {file_path}: {e}")

        return issues

    def _is_auto_fixable(self, pattern: str) -> bool:
        """Check if a pattern can be auto-fixed."""
        auto_fixable = [
            r'yaml\.load\s*\([^)]*\)\s*$',
        ]
        return any(re.search(af, pattern) for af in auto_fixable)

    def _suggest_security_fix(self, pattern: str, line: str) -> str | None:
        """Suggest a fix for security issues."""
        if 'yaml.load' in pattern:
            return line.replace('yaml.load(', 'yaml.safe_load(')
        if 'shell=True' in pattern:
            return line.replace('shell=True', 'shell=False')
        return None

    def _suggest_performance_fix(self, pattern: str, line: str, lines: list[str], line_num: int) -> str | None:
        """Suggest a fix for performance issues."""
        if 'range(len(' in pattern:
            # Suggest enumerate
            match = re.search(r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\)', line)
            if match:
                var, seq = match.groups()
                return line.replace(
                    f'for {var} in range(len({seq}))',
                    f'for {var}, item in enumerate({seq})'
                )
        return None

    def scan_directory(self) -> ScanResult:
        """Scan entire directory for issues."""
        start_time = datetime.now()
        all_issues = []
        files_scanned = 0

        for py_file in self.base_path.rglob('*.py'):
            if self.should_scan(py_file):
                issues = self.scan_file(py_file)
                all_issues.extend(issues)
                files_scanned += 1

        # Sort by severity
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
            IssueSeverity.SUGGESTION: 4,
        }
        all_issues.sort(key=lambda i: severity_order.get(i.severity, 5))

        duration = (datetime.now() - start_time).total_seconds()

        summary = {}
        for issue in all_issues:
            key = f"{issue.severity.value}:{issue.issue_type.value}"
            summary[key] = summary.get(key, 0) + 1

        return ScanResult(
            timestamp=start_time,
            files_scanned=files_scanned,
            issues_found=all_issues,
            summary=summary,
            duration_seconds=duration,
        )


class AICodeAssistant:
    """AI-powered code analysis and fix generation using Jarvis's LLM."""

    def __init__(self, llm_client, base_path: Path):
        self.llm = llm_client
        self.base_path = base_path
        self._api_manager = None
        self._subsystem_name = "auto_refine"

    def _get_api_manager(self):
        """Get or create API manager for coordinated LLM calls."""
        if self._api_manager is None:
            try:
                from core.api_manager import get_api_manager
                self._api_manager = get_api_manager(self.llm)
            except Exception as e:
                logger.warning(f"Failed to create API manager: {e}")
        return self._api_manager

    async def _llm_call(self, user_input: str, system_prompt: str) -> dict:
        """Make coordinated LLM call with rate limiting."""
        try:
            from core.api_manager import Priority as APIPriority
            api_manager = self._get_api_manager()
            if api_manager is not None:
                return await api_manager.submit_chat(
                    subsystem=self._subsystem_name,
                    priority=APIPriority.REFINEMENT,
                    user_input=user_input,
                    system_prompt=system_prompt,
                )
        except (ImportError, Exception) as e:
            logger.debug(f"API manager unavailable, using direct LLM call: {e}")
        return await self.llm.chat(user_input=user_input, system_prompt=system_prompt)

    async def analyze_issue(self, issue: CodeIssue, context: str = "") -> str:
        """Get AI analysis of an issue."""
        def _context_block(ctx: str) -> str:
            if not ctx:
                return ""
            return f"Context (surrounding code):\n```\n{ctx}\n```"
        prompt = f"""Analyze this code issue and provide a detailed fix suggestion.

File: {issue.file_path}
Line: {issue.line_number}
Type: {issue.issue_type.value}
Severity: {issue.severity.value}
Issue: {issue.title}

Code:
```
{issue.code_snippet}
```

{_context_block(context)}

Provide:
1. Brief explanation of the issue
2. Why it matters (security, performance, quality)
3. Suggested fix with code
4. Risk level of applying fix (low/medium/high)

Format your response as JSON:
{{
    "explanation": "...",
    "why_matters": "...",
    "suggested_fix": "...",
    "risk_level": "low|medium|high",
    "code_fix": "the actual code to replace or insert"
}}
"""
        try:
            response = await self._llm_call(
                user_input=prompt,
                system_prompt="You are a code analysis expert. Always respond with valid JSON.",
            )
            return get_response_content(response)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return f'{{"error": "{safe_error_message(e)}"}}'

    async def generate_fix(self, issue: CodeIssue, file_content: str) -> ProposedChange | None:
        """Generate a proposed fix for an issue."""
        if not issue.auto_fixable and issue.severity in (IssueSeverity.LOW, IssueSeverity.SUGGESTION):
            # For low severity non-auto-fixable issues, skip AI generation
            return None

        prompt = f"""Generate a safe, minimal fix for this code issue.

File: {issue.file_path}
Line: {issue.line_number}
Issue: {issue.title}
Description: {issue.description}

Current code at line {issue.line_number}:
```
{issue.code_snippet}
```

Full file context (excerpt around line {issue.line_number}):
```
{self._get_context(file_content, issue.line_number, 10)}
```

Requirements:
1. Make minimal changes - only fix the specific issue
2. Preserve all existing functionality
3. Follow Python best practices
4. Include necessary imports if adding new code

Respond with JSON:
{{
    "original_code": "exact lines to replace",
    "proposed_code": "new code",
    "explanation": "why this fix works",
    "risk_level": "low|medium|high",
    "imports_needed": ["any", "new", "imports"]
}}
"""
        try:
            response = await self._llm_call(
                user_input=prompt,
                system_prompt="You are an expert Python developer. Generate minimal, safe fixes. Always respond with valid JSON.",
            )
            content = get_response_content(response)

            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())

                    return ProposedChange(
                        id=f"change_{issue.id}",
                        issue_id=issue.id,
                        file_path=issue.file_path,
                        line_number=issue.line_number,
                        severity=issue.severity.value,
                        confidence=issue.confidence,
                        change_type="edit",
                        original_code=data.get("original_code", issue.code_snippet),
                        proposed_code=data.get("proposed_code", ""),
                        explanation=data.get("explanation", ""),
                        risk_level=data.get("risk_level", "medium"),
                    )
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON")

        except Exception as e:
            logger.error(f"AI fix generation failed: {e}")

        return None

    async def suggest_enhancements(self, file_path: str, content: str) -> list[CodeIssue]:
        """AI-powered enhancement suggestions for a file."""
        prompt = f"""Analyze this Python file and suggest improvements.

File: {file_path}

Code:
```
{content[:5000]}  # Limit for API
```

Look for:
1. Type hints that could be added
2. Error handling improvements
3. Code that could be simplified
4. Potential bugs or edge cases
5. Documentation improvements

Respond with JSON array of suggestions:
[
    {{
        "line": <line_number>,
        "type": "enhancement|bug|quality",
        "title": "...",
        "description": "...",
        "suggestion": "..."
    }}
]
"""
        try:
            response = await self._llm_call(
                user_input=prompt,
                system_prompt="You are a senior Python developer. Provide actionable suggestions. Always respond with valid JSON.",
            )
            content = get_response_content(response)

            suggestions = []
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                data = json.loads(json_match.group())
                for i, item in enumerate(data):
                    suggestions.append(CodeIssue(
                        id=f"ai_{file_path.replace('/', '_')}_{item.get('line', i)}",
                        file_path=file_path,
                        line_number=item.get("line", 0),
                        issue_type=IssueType.ENHANCEMENT if item.get("type") == "enhancement" else IssueType.CODE_QUALITY,
                        severity=IssueSeverity.SUGGESTION,
                        title=item.get("title", "AI suggestion"),
                        description=item.get("description", ""),
                        code_snippet=item.get("suggestion", "")[:100],
                        suggested_fix=item.get("suggestion"),
                        auto_fixable=False,
                        confidence=0.6,
                    ))
            return suggestions

        except Exception as e:
            logger.error(f"AI enhancement scan failed: {e}")
            return []

    def _get_context(self, content: str, line_num: int, context_lines: int = 10) -> str:
        """Get context around a line."""
        lines = content.split('\n')
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return '\n'.join(f"{i+1}: {lines[i]}" for i in range(start, end))


class ChangeManager:
    """Manages proposed changes with permission and rollback support."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.changes_file = storage_path / "proposed_changes.json"
        self.backups_dir = storage_path / "backups"
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self._changes: list[ProposedChange] = []
        self._lock = threading.RLock()
        self.MAX_CHANGES = 1000
        self._load_changes()

    def _load_changes(self):
        data = None
        try:
            if self.changes_file.exists():
                raw = self.changes_file.read_text(encoding='utf-8')
                parsed = json.loads(raw)
                items = []
                for item in parsed:
                    try:
                        item['status'] = ChangeStatus(item['status'])
                        item['created_at'] = datetime.fromisoformat(item['created_at'])
                        if item.get('applied_at'):
                            item['applied_at'] = datetime.fromisoformat(item['applied_at'])
                        items.append(ProposedChange(**item))
                    except Exception as e:
                        logger.warning(f"Skipping malformed change record: {e}")
                data = items
        except Exception as e:
            logger.error(f"Failed to load changes: {e}")
            return

        with self._lock:
            if data is not None:
                self._changes = data

    def _save_changes(self):
        """Save change history. Snapshots data under lock, writes outside lock."""
        with self._lock:
            try:
                data = []
                for change in self._changes:
                    item = {
                        'id': change.id,
                        'issue_id': change.issue_id,
                        'file_path': change.file_path,
                        'line_number': change.line_number,
                        'severity': change.severity if isinstance(change.severity, str) else change.severity,
                        'confidence': change.confidence,
                        'change_type': change.change_type,
                        'original_code': change.original_code,
                        'proposed_code': change.proposed_code,
                        'explanation': change.explanation,
                        'risk_level': change.risk_level,
                        'status': change.status.value,
                        'created_at': change.created_at.isoformat(),
                        'applied_at': change.applied_at.isoformat() if change.applied_at else None,
                        'rollback_code': change.rollback_code,
                        'error_message': change.error_message,
                    }
                    data.append(item)
                snapshot = list(data)
            except Exception as e:
                logger.error(f"Failed to snapshot changes: {e}")
                return
        try:
            atomic_write_json(self.changes_file, snapshot)
        except Exception as e:
            logger.error(f"Failed to save changes: {e}")

    def propose_change(self, change: ProposedChange) -> str:
        """Register a proposed change."""
        with self._lock:
            self._changes.append(change)
            while len(self._changes) > self.MAX_CHANGES:
                del self._changes[0]
        logger.info(f"Proposed change registered: {change.id}")
        self._save_changes()
        return change.id

    def get_pending_changes(self) -> list[ProposedChange]:
        """Get all pending changes."""
        with self._lock:
            return [c for c in self._changes if c.status == ChangeStatus.PENDING]

    def get_change(self, change_id: str) -> ProposedChange | None:
        """Get a specific change."""
        with self._lock:
            for c in self._changes:
                if c.id == change_id:
                    return c
        return None

    def approve_change(self, change_id: str) -> bool:
        """Approve a change for application."""
        with self._lock:
            change = self.get_change(change_id)
            if change:
                change.status = ChangeStatus.APPROVED
        if change:
            self._save_changes()
            return True
        return False

    def apply_change(self, change_id: str, base_path: Path) -> tuple[bool, str]:
        """Apply a change to the codebase.
        
        Uses a two-phase approach: snapshot state under lock, then do I/O outside lock,
        then update state under lock. This avoids holding the lock during file I/O.
        """
        with self._lock:
            change = self.get_change(change_id)
            if not change:
                return False, "Change not found"

            if change.status not in (ChangeStatus.PENDING, ChangeStatus.APPROVED):
                return False, f"Change status is {change.status.value}"
            
            change.status = ChangeStatus.APPLYING
            file_path_str = change.file_path
            original_code = change.original_code
            proposed_code = change.proposed_code
            change_type = change.change_type

        file_path = base_path / file_path_str

        try:
            original = ""
            rollback_code = None
            if file_path.exists():
                original = file_path.read_text(encoding='utf-8')
                backup_file = self.backups_dir / f"{change_id}_{file_path.name}"
                backup_file.write_text(original, encoding='utf-8')
                rollback_code = original

            new_content = original
            if change_type == "edit":
                if original_code:
                    new_content = original.replace(original_code, proposed_code, 1)
                if not original_code or new_content == original:
                    lines = original.split('\n')
                    for i, line in enumerate(lines):
                        if original_code.strip() and original_code.strip() in line:
                            lines[i] = proposed_code
                            break
                    new_content = '\n'.join(lines)

            elif change_type == "create":
                new_content = proposed_code

            elif change_type == "delete":
                new_content = None

            if new_content is not None:
                file_path.write_text(new_content, encoding='utf-8')
            elif change_type == "delete" and file_path.exists():
                file_path.unlink()

            with self._lock:
                if change.status != ChangeStatus.APPLYING:
                    return False, f"Change status changed during apply: {change.status.value}"
                change.status = ChangeStatus.APPLIED
                change.applied_at = datetime.now()
                change.rollback_code = rollback_code

        except Exception as e:
            with self._lock:
                change.status = ChangeStatus.FAILED
                change.error_message = safe_error_message(e)[:200]
            logger.error(f"Failed to apply change {change_id}: {e}")
            return False, f"Failed to apply change: {safe_error_message(e)[:100]}"
        
        self._save_changes()
        logger.info(f"Applied change: {change_id}")
        return True, f"Successfully applied change to {file_path_str}"

    def reject_change(self, change_id: str) -> bool:
        """Reject and remove a pending change."""
        rejected = False
        with self._lock:
            change = self.get_change(change_id)
            if change and change.status == ChangeStatus.PENDING:
                change.status = ChangeStatus.REJECTED
                rejected = True
        if rejected:
            self._save_changes()
            logger.info(f"Rejected change: {change_id}")
            return True
        return False

    def rollback_change(self, change_id: str, base_path: Path) -> tuple[bool, str]:
        """Rollback an applied change. Reads state under lock, writes file outside lock."""
        with self._lock:
            change = self.get_change(change_id)
            if not change:
                return False, "Change not found"

            if change.status != ChangeStatus.APPLIED:
                return False, "Change was not applied"
            
            rollback_code = change.rollback_code
            file_path_str = change.file_path
        
        if not rollback_code:
            return False, "No rollback code available"

        file_path = base_path / file_path_str

        try:
            file_path.write_text(rollback_code, encoding='utf-8')
            with self._lock:
                change.status = ChangeStatus.ROLLED_BACK
            self._save_changes()
            return True, f"Rolled back change to {file_path_str}"

        except Exception as e:
            logger.error(f"Failed to rollback change {change_id}: {e}")
            return False, f"Rollback failed: {e}"

    def get_statistics(self) -> dict:
        """Get change statistics."""
        with self._lock:
            stats = {
                'total': len(self._changes),
                'pending': 0,
                'approved': 0,
                'applied': 0,
                'rejected': 0,
                'failed': 0,
                'rolled_back': 0,
            }
            for change in self._changes:
                stats[change.status.value] = stats.get(change.status.value, 0) + 1
        return stats


class AutoRefineManager:
    """Main manager for auto-refine and repair system.
    
    Permission Levels:
    - AUTO_APPROVE_LOW_RISK: Auto-approve low-risk changes (docs, comments, style)
    - AUTO_APPROVE_SAFE: Auto-approve safe changes (refactors, optimizations)
    - NEVER_AUTO: Always require explicit approval
    
    Risk Levels:
    - low: Documentation, comments, minor style fixes
    - medium: Refactors, optimizations, non-critical changes
    - high: Feature additions, bug fixes
    - critical: Security fixes, breaking changes
    """

    # Permission settings - can be toggled via API or config
    AUTO_APPROVE_LOW_RISK = True    # Auto-approve documentation, style, comments
    AUTO_APPROVE_SAFE = False       # Auto-approve refactor/optimization (caution)
    REQUIRE_TESTS_FOR_AUTO = False  # If True, tests must pass for auto-approve
    MAX_AUTO_APPROVES_PER_HOUR = 10 # Rate limit for auto-approvals
    NOTIFY_ON_AUTO_APPROVE = True   # Log notification when auto-approving

    def __init__(self, llm_client, base_path: Path | None = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.analyzer = CodeAnalyzer(self.base_path)
        self.ai_assistant = AICodeAssistant(llm_client, self.base_path)
        self._auto_approve_count = 0
        self._last_auto_approve_reset = datetime.now()
        self._lock = threading.RLock()
        self.change_manager = ChangeManager(self.base_path / ".jarvis" / "refine")
        self._last_scan: ScanResult | None = None
        self._permission_callback: Callable | None = None
        self._notify_callback: Callable | None = None

    def set_permission_callback(self, callback: Callable):
        """Set callback for permission requests."""
        self._permission_callback = callback

    def set_notify_callback(self, callback: Callable):
        """Set callback for notifications."""
        self._notify_callback = callback

    def set_auto_approve_settings(
        self,
        low_risk: bool | None = None,
        safe: bool | None = None,
        require_tests: bool | None = None,
        notify: bool | None = None,
    ):
        """Configure auto-approve settings.
        
        Args:
            low_risk: Auto-approve low-risk changes (docs, comments)
            safe: Auto-approve safe changes (refactors, optimizations)
            require_tests: Require tests to pass for auto-approve
            notify: Send notification when auto-approving
        """
        if low_risk is not None:
            self.AUTO_APPROVE_LOW_RISK = low_risk
        if safe is not None:
            self.AUTO_APPROVE_SAFE = safe
        if require_tests is not None:
            self.REQUIRE_TESTS_FOR_AUTO = require_tests
        if notify is not None:
            self.NOTIFY_ON_AUTO_APPROVE = notify
        logger.info(f"Auto-approve settings: low_risk={self.AUTO_APPROVE_LOW_RISK}, safe={self.AUTO_APPROVE_SAFE}")

    def get_auto_approve_settings(self) -> dict:
        """Get current auto-approve settings."""
        return {
            "auto_approve_low_risk": self.AUTO_APPROVE_LOW_RISK,
            "auto_approve_safe": self.AUTO_APPROVE_SAFE,
            "require_tests_for_auto": self.REQUIRE_TESTS_FOR_AUTO,
            "max_auto_approves_per_hour": self.MAX_AUTO_APPROVES_PER_HOUR,
            "notify_on_auto_approve": self.NOTIFY_ON_AUTO_APPROVE,
            "current_hour_count": self._auto_approve_count,
        }

    def can_auto_approve_change(self, change: ProposedChange) -> tuple[bool, str]:
        """Check if a change can be auto-approved.
        
        Returns:
            Tuple of (can_auto_approve, reason)
        """
        with self._lock:
            # Reset counter if hour has passed
            now = datetime.now()
            if (now - self._last_auto_approve_reset).total_seconds() > 3600:
                self._auto_approve_count = 0
                self._last_auto_approve_reset = now
            
            # Check rate limit
            if self._auto_approve_count >= self.MAX_AUTO_APPROVES_PER_HOUR:
                return False, f"Rate limit reached ({self.MAX_AUTO_APPROVES_PER_HOUR}/hour)"
            
            # Low-risk changes (docs, style, comments)
            if change.risk_level == "low":
                if self.AUTO_APPROVE_LOW_RISK:
                    return True, "Low-risk change auto-approved"
                return False, "Auto-approve for low-risk changes is disabled"
            
            # Safe changes (refactors, optimizations)  
            if change.risk_level == "medium":
                if self.AUTO_APPROVE_SAFE:
                    # Additional check: never auto-approve changes to critical files
                    critical_files = {"run.py", "config.py", "core/security", "core/auth"}
                    if any(cf in change.file_path for cf in critical_files):
                        return False, "Critical file requires manual approval"
                    return True, "Safe change auto-approved"
                return False, "Auto-approve for safe changes is disabled"
            
            # High/critical risk - never auto-approve
            if change.risk_level in ("high", "critical"):
                return False, f"High-risk change ({change.risk_level}) requires manual approval"
            
            return False, "Change requires manual approval"

    async def scan_codebase(self, include_ai: bool = False) -> ScanResult:
        """Scan the entire codebase for issues."""
        logger.info("Starting codebase scan...")

        # Basic pattern-based scan
        result = self.analyzer.scan_directory()
        self._last_scan = result

        # Optional AI enhancement scan
        if include_ai:
            # Scan a few key files with AI
            key_files = [
                "core/agent.py",
                "core/llm.py",
                "run.py",
            ]
            for file_name in key_files:
                file_path = self.base_path / file_name
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        ai_issues = await self.ai_assistant.suggest_enhancements(file_name, content)
                        result.issues_found.extend(ai_issues)
                    except Exception as e:
                        logger.error(f"AI scan failed for {file_name}: {e}")

        logger.info(f"Scan complete: {len(result.issues_found)} issues in {result.files_scanned} files")
        return result

    async def propose_fixes(self, issues: list[CodeIssue] | None = None, max_fixes: int = 10) -> list[ProposedChange]:
        """Generate proposed fixes for issues."""
        if issues is None:
            if self._last_scan is None:
                await self.scan_codebase()
            issues = self._last_scan.issues_found if self._last_scan else []

        # Filter to fixable issues
        fixable = [i for i in issues if i.auto_fixable or i.severity in
                   (IssueSeverity.CRITICAL, IssueSeverity.HIGH)]
        fixable = fixable[:max_fixes]

        proposed = []
        for issue in fixable:
            try:
                file_path = self.base_path / issue.file_path
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    change = await self.ai_assistant.generate_fix(issue, content)
                    if change:
                        self.change_manager.propose_change(change)
                        proposed.append(change)
                        logger.info(f"Proposed fix for {issue.id}")
            except Exception as e:
                logger.error(f"Failed to propose fix for {issue.id}: {e}")

        return proposed

    async def request_permission(self, change: ProposedChange) -> bool:
        """Request permission to apply a change."""
        if self._permission_callback:
            return await self._permission_callback(change)
        return False  # Default to no permission

    async def apply_approved_changes(self, auto_permission: bool = False) -> list[tuple[str, bool, str]]:
        """Apply all approved changes.
        
        Args:
            auto_permission: If True, use auto-approve logic for eligible changes
            
        Returns:
            List of (change_id, success, message) tuples
        """
        results = []
        pending = self.change_manager.get_pending_changes()

        for change in pending:
            approved = False
            approval_reason = ""
            
            # Check if already approved
            if change.status == ChangeStatus.APPROVED:
                approved = True
                approval_reason = "pre-approved"
            
            # Check if auto-approve is enabled and change is eligible
            elif auto_permission:
                can_auto, reason = self.can_auto_approve_change(change)
                if can_auto:
                    approved = True
                    approval_reason = f"auto-approved: {reason}"
                    with self._lock:
                        self._auto_approve_count += 1
                    
                    # Log auto-approval
                    logger.info(f"Auto-approved change {change.id}: {reason}")
                    
                    # Notify if enabled
                    if self.NOTIFY_ON_AUTO_APPROVE and self._notify_callback:
                        try:
                            await self._notify_callback({
                                "type": "auto_approve",
                                "change_id": change.id,
                                "file": change.file_path,
                                "risk": change.risk_level,
                                "reason": reason,
                            })
                        except Exception as e:
                            logger.debug(f"Notify callback failed: {e}")
            
            # Request manual permission if not auto-approved
            if not approved:
                approved = await self.request_permission(change)
                approval_reason = "manual approval"

            if approved:
                success, message = self.change_manager.apply_change(change.id, self.base_path)
                results.append((change.id, success, f"{message} ({approval_reason})"))
            else:
                self.change_manager.reject_change(change.id)
                results.append((change.id, False, "Permission denied"))

        return results

    def get_pending_changes(self) -> list[ProposedChange]:
        """Get all pending changes."""
        return self.change_manager.get_pending_changes()

    def get_change(self, change_id: str) -> ProposedChange | None:
        """Get a specific change."""
        return self.change_manager.get_change(change_id)

    def approve_change(self, change_id: str) -> bool:
        """Approve a change."""
        return self.change_manager.approve_change(change_id)

    def apply_change(self, change_id: str) -> tuple[bool, str]:
        """Apply a specific change."""
        return self.change_manager.apply_change(change_id, self.base_path)

    def rollback_change(self, change_id: str) -> tuple[bool, str]:
        """Rollback a change."""
        return self.change_manager.rollback_change(change_id, self.base_path)

    def get_statistics(self) -> dict:
        """Get system statistics."""
        stats = self.change_manager.get_statistics()
        if self._last_scan:
            stats['last_scan'] = {
                'timestamp': self._last_scan.timestamp.isoformat(),
                'files_scanned': self._last_scan.files_scanned,
                'issues_found': len(self._last_scan.issues_found),
                'duration': self._last_scan.duration_seconds,
                'summary': self._last_scan.summary,
            }
        return stats

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = ["## 📊 Auto-Refine Status Report\n"]

        stats = self.get_statistics()

        # Change statistics
        lines.append("### Change Pipeline")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Pending | {stats.get('pending', 0)} |")
        lines.append(f"| Approved | {stats.get('approved', 0)} |")
        lines.append(f"| Applied | {stats.get('applied', 0)} |")
        lines.append(f"| Rejected | {stats.get('rejected', 0)} |")
        lines.append(f"| Failed | {stats.get('failed', 0)} |")
        lines.append(f"| Rolled Back | {stats.get('rolled_back', 0)} |")
        lines.append("")

        if self._last_scan:
            lines.append("### Last Scan")
            lines.append(f"- **Files scanned:** {self._last_scan.files_scanned}")
            lines.append(f"- **Issues found:** {len(self._last_scan.issues_found)}")
            lines.append(f"- **Duration:** {self._last_scan.duration_seconds:.1f}s")
            lines.append("")

            # Issue breakdown
            if self._last_scan.summary:
                lines.append("### Issues by Type")
                for key, count in sorted(self._last_scan.summary.items()):
                    lines.append(f"- {key}: {count}")
                lines.append("")

            # Top issues
            lines.append("### Top Issues")
            for issue in self._last_scan.issues_found[:10]:
                severity_emoji = {
                    IssueSeverity.CRITICAL: "🔴",
                    IssueSeverity.HIGH: "🟠",
                    IssueSeverity.MEDIUM: "🟡",
                    IssueSeverity.LOW: "🟢",
                    IssueSeverity.SUGGESTION: "💡",
                }.get(issue.severity, "❓")
                lines.append(f"{severity_emoji} **{issue.file_path}:{issue.line_number}** - {issue.title}")

        return "\n".join(lines)


# Tool wrapper for integration with Jarvis
class AutoRefineTool:
    """Tool for Jarvis to interact with the auto-refine system."""

    name = "auto_refine"
    description = "Scan codebase for issues, propose fixes, and apply safe changes with permission"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "status", "list", "approve", "apply", "rollback", "report"],
                "description": "Action to perform: scan codebase, check status, list pending changes, approve/apply/rollback changes, or get report",
            },
            "change_id": {
                "type": "string",
                "description": "Change ID for approve/apply/rollback actions",
            },
            "include_ai": {
                "type": "boolean",
                "description": "Include AI-powered deep analysis (slower but more thorough)",
            },
            "max_fixes": {
                "type": "integer",
                "description": "Maximum number of fixes to propose (default 10)",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    def __init__(self, manager: AutoRefineManager):
        self.manager = manager

    async def execute(
        self,
        action: str,
        change_id: str | None = None,
        include_ai: bool = False,
        max_fixes: int = 10,
    ) -> str:
        """Execute auto-refine action."""
        if action == "scan":
            result = await self.manager.scan_codebase(include_ai=include_ai)

            # Auto-propose fixes for critical/high issues
            critical = [i for i in result.issues_found if i.severity in
                       (IssueSeverity.CRITICAL, IssueSeverity.HIGH)]
            if critical:
                await self.manager.propose_fixes(critical, max_fixes=max_fixes)

            return f"Scan complete:\n- Files scanned: {result.files_scanned}\n- Issues found: {len(result.issues_found)}\n- Critical/High: {len(critical)}\n\nRun 'list' to see pending changes."

        elif action == "status":
            stats = self.manager.get_statistics()
            return json.dumps(stats, indent=2)

        elif action == "list":
            pending = self.manager.get_pending_changes()
            if not pending:
                return "No pending changes."

            lines = ["## Pending Changes\n"]
            for change in pending:
                lines.append(f"### {change.id}")
                lines.append(f"- **File:** {change.file_path}")
                lines.append(f"- **Type:** {change.change_type}")
                lines.append(f"- **Risk:** {change.risk_level}")
                lines.append(f"- **Explanation:** {change.explanation}")
                lines.append(f"- **Status:** {change.status.value}")
                lines.append(f"- **Original:**\n```\n{change.original_code[:200]}...```")
                lines.append(f"- **Proposed:**\n```\n{change.proposed_code[:200]}...```")
                lines.append("")

            return "\n".join(lines)

        elif action == "approve":
            if not change_id:
                return "Error: change_id required for approve action"

            if self.manager.approve_change(change_id):
                return f"Change {change_id} approved. Use 'apply' to apply it."
            return f"Change {change_id} not found."

        elif action == "apply":
            if change_id:
                success, message = self.manager.apply_change(change_id)
                return f"{'✓' if success else '✗'} {message}"
            else:
                results = await self.manager.apply_approved_changes(auto_permission=False)
                lines = ["Apply Results:"]
                for cid, success, msg in results:
                    lines.append(f"{'✓' if success else '✗'} {cid}: {msg}")
                return "\n".join(lines)

        elif action == "rollback":
            if not change_id:
                return "Error: change_id required for rollback action"

            success, message = self.manager.rollback_change(change_id)
            return f"{'✓' if success else '✗'} {message}"

        elif action == "report":
            return self.manager.get_summary_report()

        else:
            return f"Unknown action: {action}. Use scan, status, list, approve, apply, rollback, or report."


# Singleton manager
_refine_manager: AutoRefineManager | None = None
_refine_lock = threading.Lock()


def get_refine_manager(llm_client=None) -> AutoRefineManager:
    """Get or create the auto-refine manager."""
    global _refine_manager
    if _refine_manager is None:
        with _refine_lock:
            if _refine_manager is None:
                if llm_client is None:
                    raise ValueError("LLM client required for first initialization")
                _refine_manager = AutoRefineManager(llm_client)
    return _refine_manager
