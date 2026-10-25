"""Tests for core/startup_validator.py — StartupValidator."""
import sys
from pathlib import Path

import pytest

from core.startup_validator import StartupValidator, CheckResult


class TestStartupValidator:
    def test_validate_returns_results(self):
        sv = StartupValidator(base_path=Path(__file__).parent.parent)
        results = sv.validate()
        assert len(results) > 0

    def test_python_version_check(self):
        sv = StartupValidator()
        sv._check_python_version()
        assert any(r.name == "python_version" for r in sv._results)
        pv = next(r for r in sv._results if r.name == "python_version")
        assert pv.passed is True

    def test_core_dependencies(self):
        sv = StartupValidator()
        sv._check_core_dependencies()
        core_deps = [r for r in sv._results if r.name.startswith("dep_")]
        assert len(core_deps) > 0

    def test_optional_dependencies(self):
        sv = StartupValidator()
        sv._check_optional_dependencies()
        opt_deps = [r for r in sv._results if r.name.startswith("opt_")]
        assert len(opt_deps) > 0

    def test_directories_check(self):
        sv = StartupValidator(base_path=Path(__file__).parent.parent)
        sv._check_directories()
        dir_checks = [r for r in sv._results if r.name.startswith("dir_")]
        assert len(dir_checks) > 0
        core_dir = next((r for r in dir_checks if r.name == "dir_core"), None)
        assert core_dir is not None
        assert core_dir.passed is True

    def test_core_imports(self):
        sv = StartupValidator(base_path=Path(__file__).parent.parent)
        sv._check_core_imports()
        import_checks = [r for r in sv._results if r.name.startswith("import_")]
        assert len(import_checks) > 0

    def test_get_summary(self):
        sv = StartupValidator(base_path=Path(__file__).parent.parent)
        sv.validate()
        summary = sv.get_summary()
        assert "total_checks" in summary
        assert "passed" in summary
        assert "errors" in summary
        assert "ready" in summary
        assert isinstance(summary["ready"], bool)

    def test_ready_when_all_pass(self):
        sv = StartupValidator()
        sv._results = [CheckResult("a", True), CheckResult("b", True)]
        summary = sv.get_summary()
        assert summary["ready"] is True

    def test_not_ready_with_error(self):
        sv = StartupValidator()
        sv._results = [CheckResult("a", True), CheckResult("b", False, "fail", "error")]
        summary = sv.get_summary()
        assert summary["ready"] is False
        assert summary["errors"] == 1

    def test_warning_does_not_block(self):
        sv = StartupValidator()
        sv._results = [CheckResult("a", True), CheckResult("b", False, "warn", "warning")]
        summary = sv.get_summary()
        assert summary["ready"] is True
        assert summary["warnings"] == 1

    def test_port_check(self):
        sv = StartupValidator()
        sv._check_port_availability(port=59999)
        port_check = next((r for r in sv._results if r.name == "port_available"), None)
        assert port_check is not None


class TestCheckResult:
    def test_defaults(self):
        cr = CheckResult(name="test", passed=True)
        assert cr.message == ""
        assert cr.severity == "error"
        assert cr.fix_hint == ""

    def test_failed_with_hint(self):
        cr = CheckResult(name="test", passed=False, fix_hint="do something")
        assert cr.fix_hint == "do something"