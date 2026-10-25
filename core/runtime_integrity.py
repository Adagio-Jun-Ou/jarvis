"""Runtime Integrity Checker for Jarvis.

Monitors system integrity during runtime:
- File modification detection
- Process health monitoring
- Memory bounds checking
- State consistency validation
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from core.helpers import create_tracked_task
from core.safe_files import CRITICAL_FILES as REGISTRY_CRITICAL_FILES
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheck:
    """Result of an integrity check."""
    name: str
    passed: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


@dataclass
class FileIntegrity:
    """Tracks file integrity via hash."""
    path: Path
    hash: str
    last_checked: datetime
    modified: bool = False


class RuntimeIntegrityChecker:
    """Checks runtime integrity of the Jarvis system.
    
    Features:
    - Critical file hash monitoring
    - Process memory bounds
    - Subsystem health checks
    - Orphan detection
    """
    
    def __init__(self, jarvis_root: Path = None):
        self.jarvis_root = jarvis_root or Path(__file__).parent.parent
        self._file_hashes: Dict[Path, FileIntegrity] = {}
        self._check_interval = 60.0  # seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Critical files to monitor. Sourced from the shared registry rather
        # than a fourth private copy — this list had drifted to 6 entries while
        # safe_files.CRITICAL_FILES carried 16, so the files self-edit refuses
        # to touch were largely unmonitored at runtime. Entries that do not
        # exist on disk are skipped by start().
        self.critical_files = sorted(REGISTRY_CRITICAL_FILES)
        
        # Bounds for memory
        self.memory_bounds = {
            "max_process_memory_mb": 24 * 1024,  # 24GB
            "warning_threshold": 0.80,
            "critical_threshold": 0.90,
        }
        
        # Subsystem health callbacks
        self._health_callbacks: Dict[str, Callable] = {}
        
    def start(self) -> None:
        """Start integrity monitoring."""
        if self._running:
            return
        
        self._running = True

        # Baseline the hashes before starting the monitor — it compares against
        # them, and previously could observe an empty map on its first pass.
        for file_rel in self.critical_files:
            file_path = self.jarvis_root / file_rel
            if file_path.exists():
                self._hash_file(file_path)

        # Periodic monitoring needs a running event loop. Without one the
        # checker is still fully usable via check_all_now(), so degrade instead
        # of raising — and don't construct the coroutine at all, or it would be
        # left un-awaited with _running already set to True.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "No running event loop: integrity baseline captured, "
                "periodic monitoring not started"
            )
        else:
            self._task = create_tracked_task(
                self._monitor_loop(), name="runtime_integrity_monitor"
            )

        logger.info(f"Integrity checker started, monitoring {len(self._file_hashes)} files")
    
    def stop(self) -> None:
        """Stop integrity monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
    
    def register_health_callback(self, name: str, callback: Callable) -> None:
        """Register a subsystem health check callback."""
        self._health_callbacks[name] = callback
    
    def _hash_file(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        try:
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            self._file_hashes[path] = FileIntegrity(
                path=path,
                hash=file_hash,
                last_checked=datetime.now()
            )
            return file_hash
        except Exception as e:
            logger.error(f"Error hashing {path}: {e}")
            return ""
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_integrity()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Integrity check error: {e}")
                await asyncio.sleep(5.0)
    
    def _check_files_and_memory(self) -> List[IntegrityCheck]:
        """File-hash and memory-bound checks. Fully synchronous.

        Shared by the async monitor loop and the synchronous check_all_now(),
        so neither has to duplicate the logic or fake an event loop.
        """
        checks = []

        for path, integrity in list(self._file_hashes.items()):
            check = self._check_file_integrity(path, integrity)
            checks.append(check)
            if not check.passed:
                logger.warning(f"File integrity issue: {check.message}")

        checks.append(self._check_memory_integrity())
        return checks

    @staticmethod
    def _health_check_result(name: str, result: Any) -> IntegrityCheck:
        return IntegrityCheck(
            name=f"subsystem.{name}",
            passed=bool(result.get("healthy", False)),
            message=result.get("message", "OK"),
            details=result,
        )

    async def _check_integrity(self) -> List[IntegrityCheck]:
        """Run all integrity checks, including async subsystem health callbacks."""
        checks = self._check_files_and_memory()

        for name, callback in self._health_callbacks.items():
            try:
                result = await callback() if asyncio.iscoroutinefunction(callback) else callback()
                checks.append(self._health_check_result(name, result))
            except Exception as e:
                checks.append(IntegrityCheck(
                    name=f"subsystem.{name}",
                    passed=False,
                    message=f"Health check failed: {e}"
                ))

        return checks
    
    def _check_file_integrity(self, path: Path, stored: FileIntegrity) -> IntegrityCheck:
        """Check if a file has been modified."""
        if not path.exists():
            return IntegrityCheck(
                name=f"file.{path.name}",
                passed=False,
                message=f"Critical file missing: {path}"
            )
        
        current_hash = self._hash_file(path)
        
        if current_hash != stored.hash:
            stored.modified = True
            return IntegrityCheck(
                name=f"file.{path.name}",
                passed=False,
                message=f"File modified: {path}",
                details={
                    "old_hash": stored.hash[:16],
                    "new_hash": current_hash[:16]
                }
            )
        
        stored.last_checked = datetime.now()
        return IntegrityCheck(
            name=f"file.{path.name}",
            passed=True,
            message="OK"
        )
    
    def _check_memory_integrity(self) -> IntegrityCheck:
        """Check process memory against bounds."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            usage_ratio = memory_mb / self.memory_bounds["max_process_memory_mb"]
            
            if usage_ratio > self.memory_bounds["critical_threshold"]:
                return IntegrityCheck(
                    name="memory.bounds",
                    passed=False,
                    message=f"Critical memory usage: {memory_mb:.0f}MB ({usage_ratio:.1%})",
                    details={"memory_mb": memory_mb, "usage_ratio": usage_ratio}
                )
            elif usage_ratio > self.memory_bounds["warning_threshold"]:
                return IntegrityCheck(
                    name="memory.bounds",
                    passed=True,
                    message=f"High memory usage: {memory_mb:.0f}MB ({usage_ratio:.1%})",
                    details={"memory_mb": memory_mb, "usage_ratio": usage_ratio}
                )
            
            return IntegrityCheck(
                name="memory.bounds",
                passed=True,
                message=f"Memory OK: {memory_mb:.0f}MB ({usage_ratio:.1%})",
                details={"memory_mb": memory_mb, "usage_ratio": usage_ratio}
            )
        except Exception as e:
            return IntegrityCheck(
                name="memory.bounds",
                passed=True,
                message=f"Could not check memory: {e}"
            )
    
    def check_all_now(self) -> Dict:
        """Run all checks immediately and return results.

        Synchronous by design so it can be called from request handlers. It
        previously tried `loop.run_until_complete()` on the already-running
        loop, fell through to `asyncio.run()` which also raised, and only then
        reached a sync fallback — leaving two orphaned coroutines behind each
        call, one of them scheduled and detached.

        Async subsystem health callbacks cannot be awaited from here; they are
        reported as skipped. Use the monitor loop (or await _check_integrity()
        directly) for the full picture.
        """
        checks = self._check_files_and_memory()

        for name, callback in self._health_callbacks.items():
            if asyncio.iscoroutinefunction(callback):
                checks.append(IntegrityCheck(
                    name=f"subsystem.{name}",
                    passed=True,
                    message="Skipped: async health check needs an event loop",
                ))
                continue
            try:
                checks.append(self._health_check_result(name, callback()))
            except Exception as e:
                checks.append(IntegrityCheck(
                    name=f"subsystem.{name}",
                    passed=False,
                    message=f"Health check failed: {e}",
                ))

        return {
            "passed": all(c.passed for c in checks),
            "checks": [{ "name": c.name, "passed": c.passed, "message": c.message } for c in checks],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_critical_file_status(self) -> Dict[str, bool]:
        """Get status of all critical files."""
        status = {}
        for path, integrity in self._file_hashes.items():
            status[str(path.relative_to(self.jarvis_root))] = not integrity.modified
        return status
    
    def reset_file_hash(self, path: str) -> bool:
        """Reset the stored hash for a file (after authorized modification)."""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.jarvis_root / path
        
        if file_path.exists():
            self._hash_file(file_path)
            return True
        return False


# Singleton
_integrity_checker: Optional[RuntimeIntegrityChecker] = None


def get_integrity_checker() -> RuntimeIntegrityChecker:
    """Get the singleton integrity checker."""
    global _integrity_checker
    if _integrity_checker is None:
        _integrity_checker = RuntimeIntegrityChecker()
    return _integrity_checker
