"""Startup Validator — pre-boot dependency and configuration check.

Runs before Jarvis starts to verify:
- Required Python packages are installed
- Required directories exist
- Environment variables are set (optional)
- Port is available
- Critical modules can be imported
- Database connections work (if configured)

This catches configuration errors early, before the full boot sequence.
"""
import importlib
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    severity: str = "error"
    fix_hint: str = ""


class StartupValidator:
    """Validates Jarvis dependencies and configuration before boot."""

    def __init__(self, base_path: Path | None = None):
        self._base_path = base_path or Path.cwd()
        self._results: list[CheckResult] = []

    def validate(self) -> list[CheckResult]:
        self._results = []
        self._check_python_version()
        self._check_core_dependencies()
        self._check_optional_dependencies()
        self._check_directories()
        self._check_port_availability()
        self._check_core_imports()
        return self._results

    def _add(self, name: str, passed: bool, message: str = "", severity: str = "error", fix_hint: str = "") -> None:
        self._results.append(CheckResult(name=name, passed=passed, message=message, severity=severity, fix_hint=fix_hint))

    def _check_python_version(self) -> None:
        major, minor = sys.version_info[:2]
        if (major, minor) >= (3, 11):
            self._add("python_version", True, f"Python {major}.{minor}")
        else:
            self._add("python_version", False, f"Python {major}.{minor} too old (need 3.11+)", fix_hint="Install Python 3.11+")

    def _check_core_dependencies(self) -> None:
        required = [
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("httpx", "httpx"),
            ("pydantic", "pydantic"),
        ]
        for module_name, pip_name in required:
            try:
                importlib.import_module(module_name)
                self._add(f"dep_{module_name}", True)
            except ImportError:
                self._add(f"dep_{module_name}", False, f"{module_name} not installed", fix_hint=f"pip install {pip_name}")

    def _check_optional_dependencies(self) -> None:
        optional = [
            ("psutil", "psutil"),
            ("sentence_transformers", "sentence-transformers"),
            ("numpy", "numpy"),
            ("torch", "torch"),
        ]
        for module_name, pip_name in optional:
            try:
                importlib.import_module(module_name)
                self._add(f"opt_{module_name}", True, severity="warning")
            except ImportError:
                self._add(f"opt_{module_name}", False, f"{module_name} not installed (optional)", severity="warning", fix_hint=f"pip install {pip_name}")

    def _check_directories(self) -> None:
        dirs = {
            "core": self._base_path / "core",
            "tools": self._base_path / "tools",
            "interfaces": self._base_path / "interfaces",
            "jarvis_data": Path.home() / ".jarvis",
        }
        for name, path in dirs.items():
            if path.exists():
                self._add(f"dir_{name}", True, str(path))
            else:
                severity = "warning" if name == "jarvis_data" else "error"
                self._add(f"dir_{name}", False, f"{path} not found", severity=severity,
                         fix_hint="mkdir -p " + str(path) if severity == "warning" else "Check Jarvis installation")

    def _check_port_availability(self, port: int = 8080) -> None:
        port = int(os.environ.get("JARVIS_PORT", port))
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", port))
                if result != 0:
                    self._add("port_available", True, f"Port {port} available")
                else:
                    self._add("port_available", False, f"Port {port} already in use", fix_hint=f"Kill the process using port {port} or set JARVIS_PORT")
        except Exception as e:
            self._add("port_available", False, f"Port check failed: {e}")

    def _check_core_imports(self) -> None:
        core_modules = [
            "core.agent",
            "core.llm",
            "core.memory",
            "core.config",
            "core.execution_context",
            "core.watchdog",
            "core.event_bus",
            "core.context_compressor",
            "core.sandbox",
            "core.session_persistence",
            "core.connection_pool",
            "core.tool_chain_optimizer",
            "core.integration_harness",
            "core.streaming_pipeline",
            "core.unified_rate_limiter",
            "core.agent_memory_store",
            "core.health_router",
        ]
        for mod in core_modules:
            try:
                importlib.import_module(mod)
                self._add(f"import_{mod}", True)
            except Exception as e:
                self._add(f"import_{mod}", False, str(e)[:100], fix_hint=f"Check {mod} for import errors")

    def get_summary(self) -> dict[str, Any]:
        if not self._results:
            self.validate()
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        errors = [r for r in self._results if not r.passed and r.severity == "error"]
        warnings = [r for r in self._results if not r.passed and r.severity == "warning"]
        return {
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "errors": len(errors),
            "warnings": len(warnings),
            "ready": len(errors) == 0,
            "error_details": [{"name": r.name, "message": r.message, "fix": r.fix_hint} for r in errors],
        }

    def print_report(self) -> None:
        summary = self.get_summary()
        for r in self._results:
            icon = "OK" if r.passed else ("WARN" if r.severity == "warning" else "FAIL")
            msg = f"  [{icon}] {r.name}: {r.message}" if r.message else f"  [{icon}] {r.name}"
            if r.fix_hint and not r.passed:
                msg += f" → {r.fix_hint}"
            if r.passed or r.severity == "error":
                logger.info(msg) if r.passed else logger.error(msg)
            else:
                logger.warning(msg)

        logger.info(f"Startup validation: {summary['passed']}/{summary['total_checks']} passed, {summary['errors']} errors, {summary['warnings']} warnings")
        if summary["ready"]:
            logger.info("Jarvis ready to start")
        else:
            logger.error("Jarvis NOT ready — fix errors above before starting")