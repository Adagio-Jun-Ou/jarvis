#!/usr/bin/env python3
"""Jarvis Refinement Script - Automated code quality and test runner.

Run this script to perform quick health checks and apply auto-fixes.

Usage:
    python scripts/refine.py           # Run all checks
    python scripts/refine.py --fix     # Auto-fix what can be fixed
    python scripts/refine.py --tests   # Run test suite
    python scripts/refine.py --full    # Full refinement including coverage
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], description: str, timeout: int = 60) -> tuple[bool, str]:
    """Run a command and return success status + output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✅ Passed ({elapsed:.1f}s)")
            return True, result.stdout
        else:
            print(f"❌ Failed ({elapsed:.1f}s)")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, str(e)


def check_syntax() -> bool:
    """Check Python syntax for all modules."""
    success, _ = run_command(
        ["./venv/bin/python", "-m", "py_compile", "core/", "tools/", "interfaces/", "tests/"],
        "Syntax Check (py_compile)",
        timeout=30,
    )
    return success


def check_lint(fix: bool = False) -> bool:
    """Run ruff linter."""
    cmd = ["./venv/bin/ruff", "check", "core/", "tools/", "interfaces/", "tests/"]
    if fix:
        cmd.append("--fix")
        cmd.append("--unsafe-fixes")

    success, output = run_command(cmd, "Lint Check (ruff)", timeout=30)
    return success


def check_tests(quick: bool = True) -> bool:
    """Run test suite."""
    if quick:
        cmd = ["./venv/bin/python", "-m", "pytest", "tests/", "-q", "--tb=short", "-x"]
    else:
        cmd = ["./venv/bin/python", "-m", "pytest", "tests/", "-v", "--tb=short"]

    success, _ = run_command(cmd, "Test Suite (pytest)", timeout=300)
    return success


def check_coverage() -> bool:
    """Run tests with coverage."""
    cmd = [
        "./venv/bin/python", "-m", "pytest", "tests/",
        "--cov=core", "--cov=tools",
        "--cov-report=term-missing",
        "-q", "--tb=short",
    ]
    success, _ = run_command(cmd, "Coverage Report", timeout=600)
    return success


def check_imports() -> bool:
    """Check for circular imports and slow imports."""
    print("\n" + "="*60)
    print("🔧 Import Time Check")
    print("="*60)

    start = time.time()
    try:
        import core.agent  # type: ignore
        elapsed = time.time() - start

        if elapsed < 3:
            print(f"✅ Fast imports ({elapsed:.2f}s)")
            return True
        elif elapsed < 6:
            print(f"⚠️ Moderate imports ({elapsed:.2f}s) - consider lazy loading")
            return True
        else:
            print(f"❌ Slow imports ({elapsed:.2f}s) - needs optimization")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_dependencies() -> bool:
    """Check for security vulnerabilities in dependencies."""
    try:
        import pip_audit  # type: ignore
        has_audit = True
    except ImportError:
        has_audit = False

    if has_audit:
        success, _ = run_command(
            ["./venv/bin/pip-audit", "--desc"],
            "Dependency Security (pip-audit)",
            timeout=60,
        )
        return success
    else:
        print("\n⚠️ pip-audit not installed, skipping security check")
        print("   Install with: pip install pip-audit")
        return True


def main():
    parser = argparse.ArgumentParser(description="Jarvis Refinement Script")
    parser.add_argument("--fix", action="store_true", help="Auto-fix lint issues")
    parser.add_argument("--tests", action="store_true", help="Run tests only")
    parser.add_argument("--full", action="store_true", help="Full refinement including coverage")
    parser.add_argument("--quick", action="store_true", help="Quick check (default)")
    args = parser.parse_args()

    print("🚀 Jarvis Refinement Script")
    print(f"   Mode: {'full' if args.full else 'quick'}")
    print(f"   Fix: {args.fix}")

    results = {}

    if args.tests:
        results["tests"] = check_tests(quick=False)
    elif args.full:
        results["syntax"] = check_syntax()
        results["lint"] = check_lint(fix=args.fix)
        results["imports"] = check_imports()
        results["tests"] = check_tests(quick=False)
        results["coverage"] = check_coverage()
        results["deps"] = check_dependencies()
    else:
        results["syntax"] = check_syntax()
        results["lint"] = check_lint(fix=args.fix)
        results["imports"] = check_imports()
        results["tests"] = check_tests(quick=True)

    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)

    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✨ All checks passed!")
        return 0
    else:
        print("\n⚠️ Some checks failed - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
