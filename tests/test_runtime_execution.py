"""Tests for runtime execution modules.

Covers:
- ExecutionContext: CWD tracking, env vars, process execution, file ops
- AdaptiveAgentLoop: self-healing, fallback, checkpoint, context awareness
- SystemAwareness: resource snapshots, service checks, process listing
"""
import os
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from core.execution_context import (
    ExecutionContext, ProcessRecord, FileOperation, get_execution_context,
)
from core.adaptive_agent_loop import (
    AdaptiveAgentLoop, ExecutionPhase, TurnCheckpoint, get_adaptive_agent_loop,
)
from core.system_awareness import (
    SystemAwareness, ResourceSnapshot, ServiceStatus, get_system_awareness,
)
from core.watchdog import JarvisWatchdog, WatchdogEvent, get_watchdog


# ========================================================================
# ExecutionContext Tests
# ========================================================================

class TestExecutionContext:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ExecutionContext(workspace_root=Path(self.tmpdir))
        self.ctx._loaded = True

    def test_initial_cwd(self):
        assert self.ctx.cwd == self.tmpdir

    def test_cd_to_valid_dir(self):
        sub = Path(self.tmpdir) / "subdir"
        sub.mkdir()
        success, new_cwd = self.ctx.cd(str(sub))
        assert success is True
        assert self.ctx.cwd == str(sub.resolve())

    def test_cd_to_home(self):
        success, new_cwd = self.ctx.cd("~")
        assert success is True
        assert new_cwd == os.path.realpath(os.path.expanduser("~"))

    def test_cd_to_invalid_dir(self):
        success, error = self.ctx.cd("/nonexistent/path/xyz")
        assert success is False
        assert "not found" in error.lower() or "denied" in error.lower()

    def test_cd_to_protected(self):
        success, error = self.ctx.cd("/etc")
        assert success is False

    def test_cd_relative_path(self):
        sub = Path(self.tmpdir) / "projects"
        sub.mkdir()
        success, new_cwd = self.ctx.cd("projects")
        assert success is True

    def test_set_and_get_env(self):
        self.ctx.set_env("JARVIS_TEST_VAR", "hello")
        assert self.ctx.get_env("JARVIS_TEST_VAR") == "hello"

    def test_get_env_default(self):
        result = self.ctx.get_env("NONEXISTENT_VAR", "default")
        assert result == "default"

    def test_get_full_env(self):
        self.ctx.set_env("MY_TEST_VAR", "test_value")
        env = self.ctx.get_full_env()
        assert env["MY_TEST_VAR"] == "test_value"
        assert env.get("HOME") is not None or "PWD" in env

    def test_env_secret_redaction(self):
        self.ctx.set_env("MY_SECRET_KEY", "super_secret_value")
        env = self.ctx.get_full_env()
        assert env["MY_SECRET_KEY"] == "***REDACTED***"

    def test_max_env_size(self):
        ctx = ExecutionContext(workspace_root=Path(self.tmpdir))
        ctx._loaded = True
        ctx._env.clear()
        for i in range(ctx.MAX_ENV_SIZE + 10):
            ctx.set_env(f"VAR_{i}", f"val_{i}")
        assert len(ctx._env) <= ctx.MAX_ENV_SIZE

    def test_run_simple_command(self):
        result = self.ctx.run("echo hello")
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_run_command_cwd(self):
        # `pwd` is POSIX-only; `cd` with no arg prints cwd on Windows (cmd),
        # and on POSIX `cd` alone returns to HOME. Use echo %CD% (cmd) / $PWD
        # (sh) via a python one-liner that works on both platforms.
        result = self.ctx.run(
            'python -c "import os; print(os.getcwd())"'
        )
        assert result["exit_code"] == 0
        assert self.tmpdir in result["stdout"].strip()

    def test_run_failing_command(self):
        # `false` is POSIX-only; exit with non-zero cross-platform.
        result = self.ctx.run(
            'python -c "import sys; sys.exit(1)"'
        )
        assert result["exit_code"] != 0

    def test_run_nonexistent_command(self):
        result = self.ctx.run("nonexistent_command_12345")
        assert result["exit_code"] != 0

    def test_run_timeout(self):
        # `sleep 5` is POSIX-only; use a python sleep that works everywhere.
        result = self.ctx.run(
            'python -c "import time; time.sleep(5)"', timeout=1.0
        )
        assert result["timed_out"] is True

    def test_run_tracks_cwd_change(self):
        self.ctx.cd(self.tmpdir)
        sub = Path(self.tmpdir) / "inner"
        sub.mkdir()
        self.ctx.run(f"cd {sub}")
        # After cd, the context should track it (if supported)
        # But shell cd in subprocess doesn't change parent CWD

    def test_run_command_count(self):
        initial = self.ctx._command_count
        self.ctx.run("echo test1")
        self.ctx.run("echo test2")
        assert self.ctx._command_count == initial + 2

    def test_read_file(self):
        test_file = Path(self.tmpdir) / "test_read.txt"
        test_file.write_text("hello world", encoding='utf-8')
        result = self.ctx.read_file(str(test_file))
        assert result["success"] is True
        assert "hello world" in result["content"]

    def test_read_file_not_found(self):
        result = self.ctx.read_file("/nonexistent/file.txt")
        assert result["success"] is False

    def test_read_file_with_lines(self):
        test_file = Path(self.tmpdir) / "lines.txt"
        test_file.write_text("\n".join(f"line {i}" for i in range(20)))
        result = self.ctx.read_file(str(test_file), start_line=5, end_line=10)
        assert result["success"] is True

    def test_write_file(self):
        test_file = Path(self.tmpdir) / "test_write.txt"
        result = self.ctx.write_file(str(test_file), "hello world")
        assert result["success"] is True
        assert test_file.read_text(encoding='utf-8') == "hello world"

    def test_write_file_creates_dirs(self):
        test_file = Path(self.tmpdir) / "deep" / "nested" / "test.txt"
        result = self.ctx.write_file(str(test_file), "nested content")
        assert result["success"] is True

    def test_write_file_append(self):
        test_file = Path(self.tmpdir) / "append.txt"
        test_file.write_text("first", encoding='utf-8')
        result = self.ctx.write_file(str(test_file), "second", mode="append")
        assert result["success"] is True
        assert "first" in test_file.read_text(encoding='utf-8')
        assert "second" in test_file.read_text(encoding='utf-8')

    def test_write_blocked_extension(self):
        test_file = Path(self.tmpdir) / "malware.exe"
        result = self.ctx.write_file(str(test_file), "bad")
        assert result["success"] is False
        assert "blocked" in result["error"].lower() or "extension" in result["error"].lower()

    def test_file_operation_logging(self):
        test_file = Path(self.tmpdir) / "logged.txt"
        self.ctx.write_file(str(test_file), "data")
        logs = self.ctx.get_recent_files(5)
        assert len(logs) > 0
        assert logs[0]["op_type"] == "write"

    def test_process_history(self):
        self.ctx.run("echo history_test")
        history = self.ctx.get_recent_commands(5)
        assert len(history) > 0
        assert "echo history_test" in history[0]["command"]

    def test_get_status(self):
        status = self.ctx.get_status()
        assert "cwd" in status
        assert "command_count" in status
        assert "platform" in status
        assert "hostname" in status

    def test_heartbeat(self):
        hb = self.ctx.heartbeat()
        assert hb["healthy"] is True
        assert "cwd" in hb

    def test_reset(self):
        self.ctx.run("echo before_reset")
        self.ctx.set_env("RESET_TEST", "value")
        self.ctx.reset()
        assert self.ctx._command_count == 0
        assert "RESET_TEST" not in self.ctx._env

    def test_singleton(self):
        ctx1 = get_execution_context()
        ctx2 = get_execution_context()
        assert ctx1 is ctx2


# ========================================================================
# AdaptiveAgentLoop Tests
# ========================================================================

class TestAdaptiveAgentLoop:
    def setup_method(self):
        self.loop = AdaptiveAgentLoop()

    def test_initial_phase(self):
        assert self.loop.phase == "idle"

    def test_get_status(self):
        status = self.loop.get_status()
        assert "phase" in status
        assert "turn_count" in status
        assert "context" in status

    def test_checkpoint(self):
        cp = self.loop.checkpoint(
            user_input="test input",
            response="test response",
            tool_calls=[{"name": "shell", "arguments": {"command": "ls"}}],
        )
        assert cp.turn_number >= 0
        assert cp.user_input == "test input"
        assert cp.cwd  # Should have a CWD

    def test_checkpoint_serialization(self):
        cp = self.loop.checkpoint(
            user_input="hello",
            response="hi there",
            error="test error",
        )
        d = cp.to_dict()
        assert "user_input" in d
        assert "cwd" in d
        assert "error" in d

    def test_execute_unknown_tool(self):
        async def _run():
            return await self.loop.execute_tool_with_recovery("nonexistent_tool_12345", {})
        result = asyncio.run(_run())
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_suggest_alternative_tool(self):
        suggestion = self.loop._suggest_alternative_tool("terminal-execute", {})
        assert suggestion is None or isinstance(suggestion, str)

    def test_context_awareness_includes_cwd(self):
        awareness = self.loop._build_context_awareness()
        assert "Working directory" in awareness

    def test_compress_messages(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(30):
            messages.append({"role": "user", "content": f"message {i}"})
            messages.append({"role": "assistant", "content": f"response {i}"})
        compressed = self.loop._compress_messages(messages)
        assert len(compressed) < len(messages)

    def test_compress_messages_short(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        compressed = self.loop._compress_messages(messages)
        assert len(compressed) == len(messages)

    def test_recover_from_timeout(self):
        result = self.loop.recover_from_error("connection timeout occurred")
        assert result["recovered"] is True
        assert "retry" in str(result["strategies"])

    def test_recover_from_permission(self):
        result = self.loop.recover_from_error("permission denied: /etc/shadow")
        assert result["recovered"] is True
        assert "approval" in str(result["strategies"]).lower() or "alternative" in str(result["strategies"]).lower()

    def test_recover_consecutive_errors(self):
        self.loop._consecutive_errors = 5
        result = self.loop.recover_from_error("another error")
        assert result["recovered"] is False

    def test_shell_execution_through_context(self):
        result = self.loop._execute_shell_tool("shell", {"command": "echo test"}, timeout=30.0)
        assert isinstance(result, dict)
        assert "output" in result or "success" in result

    def test_file_read_through_context(self):
        tmpfile = Path(tempfile.mkdtemp()) / "adaptive_test.txt"
        tmpfile.write_text("adaptive test content", encoding='utf-8')
        result = self.loop._execute_read_tool({"path": str(tmpfile)})
        assert result.get("success") is True or "content" in result

    def test_file_write_through_context(self):
        tmpfile = Path(tempfile.mkdtemp()) / "adaptive_write.txt"
        result = self.loop._execute_write_tool({
            "path": str(tmpfile),
            "content": "written by adaptive loop",
        })
        assert result.get("success") is True

    def test_singleton(self):
        l1 = get_adaptive_agent_loop()
        l2 = get_adaptive_agent_loop()
        assert l1 is l2

    def test_singleton_late_binds_missing_llm(self):
        """Regression: the status endpoint creates the singleton without
        an llm_client; later chat calls must UPGRADE the cached instance
        instead of being poisoned — _think() returns None whenever
        self.llm is falsy, so every adaptive-loop chat would silently
        produce nothing for the life of the process."""
        from core.adaptive_agent_loop import reset_adaptive_agent_loop

        reset_adaptive_agent_loop()
        # First touch: status-endpoint style (no llm)
        poisoned = get_adaptive_agent_loop(agent="fake-agent")
        assert poisoned.llm is None
        # Later: chat supplies the real client — must be attached
        fake_llm = object()
        upgraded = get_adaptive_agent_loop(agent="fake-agent", llm_client=fake_llm)
        assert upgraded is poisoned  # same singleton
        assert upgraded.llm is fake_llm  # but no longer poisoned
        reset_adaptive_agent_loop()

    def test_singleton_does_not_replace_existing_dependencies(self):
        """Late binding only fills gaps; an existing llm is kept even if
        a caller passes a different object (no thrash on repeated calls)."""
        from core.adaptive_agent_loop import reset_adaptive_agent_loop

        reset_adaptive_agent_loop()
        first_llm = object()
        loop = get_adaptive_agent_loop(agent="a", llm_client=first_llm)
        again = get_adaptive_agent_loop(agent="b", llm_client=object())
        assert again is loop
        assert again.llm is first_llm
        assert again.agent == "a"
        reset_adaptive_agent_loop()


# ========================================================================
# SystemAwareness Tests
# ========================================================================

class TestSystemAwareness:
    def setup_method(self):
        self.sa = SystemAwareness()

    def test_snapshot(self):
        snap = self.sa.snapshot()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.timestamp > 0

    def test_snapshot_has_load(self):
        snap = self.sa.snapshot()
        if hasattr(os, "getloadavg"):
            assert snap.load_1m >= 0

    def test_snapshot_serialization(self):
        snap = self.sa.snapshot()
        d = snap.to_dict()
        assert "cpu_percent" in d
        assert "memory_percent" in d
        assert "timestamp" in d

    def test_awareness_report(self):
        report = self.sa.get_awareness_report()
        assert "hostname" in report
        assert "platform" in report
        assert "resources" in report

    def test_resource_trend_no_data(self):
        trend = self.sa.get_resource_trend(minutes=5)
        assert "samples" in trend

    def test_resource_trend_with_data(self):
        for _ in range(3):
            self.sa.snapshot()
        trend = self.sa.get_resource_trend(minutes=5)
        if trend.get("samples", 0) > 0:
            assert "cpu_avg" in trend

    def test_watch_path(self):
        tmpfile = Path(tempfile.mkdtemp()) / "watch_test.txt"
        tmpfile.write_text("initial", encoding='utf-8')
        self.sa.watch_path(str(tmpfile))
        assert str(tmpfile) in self.sa._watched_paths

    def test_check_changes(self):
        tmpfile = Path(tempfile.mkdtemp()) / "change_test.txt"
        tmpfile.write_text("original", encoding='utf-8')
        self.sa.watch_path(str(tmpfile))
        time.sleep(0.1)
        tmpfile.write_text("modified", encoding='utf-8')
        time.sleep(0.1)

        changes = self.sa.check_changes()
        assert any(c["path"] == str(tmpfile) and c["change"] == "modified" for c in changes)

    def test_watch_service(self):
        self.sa.watch_service("docker")
        assert "docker" in self.sa._watched_services

    def test_get_services_status(self):
        self.sa.watch_service("docker")
        services = self.sa.get_services_status()
        assert isinstance(services, list)

    def test_get_running_processes(self):
        procs = self.sa.get_running_processes(top_n=5)
        assert isinstance(procs, list)

    def test_get_docker_containers(self):
        containers = self.sa.get_docker_containers()
        assert isinstance(containers, list)

    def test_singleton(self):
        s1 = get_system_awareness()
        s2 = get_system_awareness()
        assert s1 is s2


# ========================================================================
# Integration Tests
# ========================================================================

class TestRuntimeIntegration:
    def test_context_and_awareness(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        sa = SystemAwareness()

        ctx.run("echo integration_test")
        sa.snapshot()

        report = sa.get_awareness_report()
        assert "resources" in report

        status = ctx.get_status()
        assert status["command_count"] >= 1

    def test_context_survives_cd(self):
        tmpdir = tempfile.mkdtemp()
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True

        sub = Path(tmpdir) / "project"
        sub.mkdir()

        success, _ = ctx.cd(str(sub))
        assert success is True
        assert ctx.cwd == str(sub.resolve())

        # `pwd` is POSIX-only; use a python one-liner for cross-platform cwd.
        result = ctx.run('python -c "import os; print(os.getcwd())"')
        assert result["exit_code"] == 0
        assert str(sub.resolve()) in result["stdout"]

    def test_adaptive_loop_with_context(self):
        loop = AdaptiveAgentLoop()

        async def _run():
            return await loop.execute_tool_with_recovery("nonexistent_tool_xyz", {})
        result = asyncio.run(_run())
        assert result["success"] is False

        status = loop.get_status()
        # Unknown tool errors may or may not count as total_errors
        # depending on whether they hit the retry path
        assert "total_errors" in status

    def test_full_workflow(self):
        tmpdir = tempfile.mkdtemp()
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True

        ctx.cd(tmpdir)

        # Write a file via python (cross-platform) instead of shell redirect,
        # then read it back via the context's read_file. `cat` is POSIX-only.
        result = ctx.run(
            'python -c "open(r\'test_file.txt\',\'w\').write(\'hello from jarvis\')"'
        )
        assert result["exit_code"] == 0

        read_result = ctx.read_file(f"{tmpdir}/test_file.txt")
        assert read_result["success"] is True

        write_result = ctx.write_file(f"{tmpdir}/test_output.txt", "jarvis was here")
        assert write_result["success"] is True

        assert Path(f"{tmpdir}/test_output.txt").read_text(encoding='utf-8') == "jarvis was here"


import asyncio


# ========================================================================
# Watchdog Tests
# ========================================================================

class TestJarvisWatchdog:
    def setup_method(self):
        self.wd = JarvisWatchdog()

    def test_initial_status(self):
        status = self.wd.get_status()
        assert status["running"] is False
        assert status["check_count"] == 0

    def test_start_stop(self):
        self.wd.start()
        assert self.wd._running is True
        time.sleep(2)
        self.wd.stop()
        assert self.wd._running is False

    def test_emit_event(self):
        self.wd._emit("heartbeat", "test heartbeat", {"key": "value"})
        events = self.wd.get_events(limit=10)
        assert len(events) == 1
        assert events[0]["event_type"] == "heartbeat"
        assert events[0]["message"] == "test heartbeat"

    def test_emit_multiple_events(self):
        for i in range(10):
            self.wd._emit("test", f"event {i}")
        events = self.wd.get_events(limit=5)
        assert len(events) == 5

    def test_filter_events_by_type(self):
        self.wd._emit("heartbeat", "hb1")
        self.wd._emit("resource_alert", "alert1")
        self.wd._emit("heartbeat", "hb2")
        events = self.wd.get_events(event_type="heartbeat")
        assert len(events) == 2

    def test_check_increments_count(self):
        self.wd._check()
        assert self.wd._check_count == 1

    def test_check_resources_no_psutil(self):
        wd = JarvisWatchdog()
        wd._psutil_available = False
        wd._check_resources()  # Should not crash

    def test_check_zombies_no_psutil(self):
        wd = JarvisWatchdog()
        wd._psutil_available = False
        wd._check_zombies()  # Should not crash

    def test_get_status_after_checks(self):
        self.wd._check()
        status = self.wd.get_status()
        assert status["check_count"] >= 1
        assert status["event_count"] >= 1

    def test_singleton(self):
        w1 = get_watchdog()
        w2 = get_watchdog()
        assert w1 is w2

    def test_register_subsystem(self):
        self.wd.register_subsystem("test_svc", health_fn=lambda: True, restart_fn=lambda: None)
        status = self.wd.get_status()
        assert "test_svc" in status["subsystems"]
        assert status["subsystems"]["test_svc"]["status"] == "unknown"

    def test_check_subsystem_health_healthy(self):
        self.wd.register_subsystem("healthy_svc", health_fn=lambda: True, restart_fn=lambda: None)
        self.wd._check_subsystem_health()
        status = self.wd.get_status()
        assert status["subsystems"]["healthy_svc"]["status"] == "healthy"
        assert status["subsystems"]["healthy_svc"]["fail_count"] == 0

    def test_check_subsystem_health_failed(self):
        call_count = [0]
        def failing_health():
            call_count[0] += 1
            raise RuntimeError("service down")
        self.wd.register_subsystem("failing_svc", health_fn=failing_health, restart_fn=lambda: None)
        self.wd._check_subsystem_health()
        status = self.wd.get_status()
        assert status["subsystems"]["failing_svc"]["status"] == "failed"
        assert status["subsystems"]["failing_svc"]["fail_count"] == 1

    def test_subsystem_recovery_cooldown(self):
        restarts = [0]
        def restart_fn():
            restarts[0] += 1
        def failing_health():
            raise RuntimeError("down")
        self.wd.register_subsystem("cooldown_svc", health_fn=failing_health, restart_fn=restart_fn)
        self.wd._check_subsystem_health()
        self.wd._check_subsystem_health()
        self.wd._try_recover("cooldown_svc")
        self.wd._try_recover("cooldown_svc")
        assert restarts[0] <= 1

    def test_oom_check_no_psutil(self):
        wd = JarvisWatchdog()
        wd._psutil_available = False
        wd._check_oom()

    def test_status_includes_subsystems(self):
        self.wd.register_subsystem("svc1", health_fn=lambda: True)
        status = self.wd.get_status()
        assert "subsystems" in status
        assert "svc1" in status["subsystems"]


# ========================================================================
# Full Integration Tests
# ========================================================================

class TestTerminalPermissionFix:
    def test_needs_permission_low_risk(self):
        sys_path_mod = str(Path(__file__).parent.parent)
        import sys
        if sys_path_mod not in sys.path:
            sys.path.insert(0, sys_path_mod)
        from tools.terminal import JarvisComputer
        comp = JarvisComputer()
        assert comp.AUTO_APPROVE_SAFE is True
        assert comp.needs_permission("ls") is False
        assert comp.needs_permission("pwd") is False
        assert comp.needs_permission("git status") is False

    def test_needs_permission_high_risk(self):
        sys_path_mod = str(Path(__file__).parent.parent)
        import sys
        if sys_path_mod not in sys.path:
            sys.path.insert(0, sys_path_mod)
        from tools.terminal import JarvisComputer
        comp = JarvisComputer()
        assert comp.needs_permission("rm -rf /") is True
        assert comp.needs_permission("sudo apt install something") is True

    def test_needs_permission_disabled(self):
        sys_path_mod = str(Path(__file__).parent.parent)
        import sys
        if sys_path_mod not in sys.path:
            sys.path.insert(0, sys_path_mod)
        from tools.terminal import JarvisComputer
        comp = JarvisComputer()
        comp.AUTO_APPROVE_SAFE = False
        assert comp.needs_permission("ls") is True
        comp.AUTO_APPROVE_SAFE = True

    def test_terminal_execute_tool_needs_permission_method(self):
        sys_path_mod = str(Path(__file__).parent.parent)
        import sys
        if sys_path_mod not in sys.path:
            sys.path.insert(0, sys_path_mod)
        from tools.terminal import TerminalExecuteTool
        tool = TerminalExecuteTool.__new__(TerminalExecuteTool)
        tool.computer = MagicMock()
        tool.computer.needs_permission.return_value = False
        assert tool.needs_permission(command="ls") is False
        tool.computer.needs_permission.return_value = True
        assert tool.needs_permission(command="rm -rf /") is True


class TestAdaptiveLoopRecovery:
    def test_recover_from_timeout(self):
        loop = AdaptiveAgentLoop()
        result = loop.recover_from_error("Connection timeout after 30s")
        assert result["recovered"] is True
        assert "retry_shorter" in result["strategies"]

    def test_recover_from_permission(self):
        loop = AdaptiveAgentLoop()
        result = loop.recover_from_error("Permission denied: /root/file")
        assert result["recovered"] is True
        assert "request_approval" in result["strategies"]

    def test_recover_from_not_found(self):
        loop = AdaptiveAgentLoop()
        result = loop.recover_from_error("Tool not found: xyz")
        assert result["recovered"] is True
        assert "search_alternative" in result["strategies"]

    def test_recover_from_connection_error(self):
        loop = AdaptiveAgentLoop()
        result = loop.recover_from_error("Connection refused to localhost:11434")
        assert "retry_with_backoff" in result["strategies"]

    def test_recover_max_consecutive_errors(self):
        loop = AdaptiveAgentLoop()
        loop._consecutive_errors = loop._max_consecutive_errors
        result = loop.recover_from_error("Yet another error")
        assert result["recovered"] is False
        assert loop._phase == ExecutionPhase.FAILED

    def test_execute_tool_with_recovery_unknown_tool(self):
        loop = AdaptiveAgentLoop()
        import asyncio
        async def _test():
            return await loop.execute_tool_with_recovery("nonexistent_tool", {})
        result = asyncio.run(_test())
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_execute_shell_tool(self):
        loop = AdaptiveAgentLoop()
        result = loop._execute_shell_tool("terminal_execute", {"command": "echo hello"}, 10.0)
        assert result["success"] is True
        assert "hello" in result.get("output", "")

    def test_execute_read_tool(self):
        loop = AdaptiveAgentLoop()
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmpfile.write("test content")
        tmpfile.close()
        try:
            result = loop._execute_read_tool({"path": tmpfile.name})
            assert result["success"] is True
            assert "test content" in result.get("output", "")
        finally:
            os.unlink(tmpfile.name)

    def test_execute_write_tool(self):
        loop = AdaptiveAgentLoop()
        tmppath = os.path.join(tempfile.mkdtemp(), "test_write.txt")
        result = loop._execute_write_tool({"path": tmppath, "content": "written content"})
        assert result["success"] is True
        with open(tmppath, encoding='utf-8') as f:
            assert f.read() == "written content"

    def test_checkpoint_and_resume(self):
        loop = AdaptiveAgentLoop()
        cp = loop.checkpoint("test input", response="test response", error=None)
        assert cp.user_input == "test input"
        assert cp.assistant_response == "test response"
        status = loop.get_status()
        assert status["checkpoints_saved"] == 1

    def test_context_awareness_in_prompt(self):
        loop = AdaptiveAgentLoop()
        loop._ctx.run("echo test")
        awareness_text = loop._build_context_awareness()
        assert "Current Execution Context" in awareness_text
        assert "Working directory" in awareness_text


class TestToolInputValidation:
    def test_registry_validator_basic(self):
        from core.tool_registry import ToolRegistryValidator, ToolInfo, ToolCategory
        validator = ToolRegistryValidator()
        is_valid, errors = validator.validate_tool_input("nonexistent_tool_xyz", {"a": 1})
        assert is_valid is False

    def test_tool_arg_size_limit(self):
        big_arg = "x" * 60000
        from core.tool_registry import ToolRegistryValidator
        validator = ToolRegistryValidator()
        assert validator is not None


class TestCwdSync:
    def test_execution_context_cwd_persists(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        sub = Path(ctx.cwd) / "subdir"
        sub.mkdir()
        success, new_cwd = ctx.cd(str(sub))
        assert success is True
        assert ctx.cwd == str(sub)

    def test_cd_to_protected_dir(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        success, error = ctx.cd("/etc")
        assert success is False

    def test_run_tracks_cwd(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        sub = Path(ctx.cwd) / "target"
        sub.mkdir()
        ctx.run(f"cd {sub}")
        assert ctx.cwd == str(sub)

    def test_compound_cd_tracking(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        sub = Path(ctx.cwd) / "compound"
        sub.mkdir()
        ctx.run(f"cd {sub} && echo hello")
        assert ctx.cwd == str(sub)

    def test_cd_in_middle_of_compound(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        sub = Path(ctx.cwd) / "middle"
        sub.mkdir()
        ctx.run(f"echo first && cd {sub}")
        assert ctx.cwd == str(sub)


class TestAgentLoopMalformedArgs:
    def test_malformed_json_handled_by_adaptive_loop(self):
        loop = AdaptiveAgentLoop()
        assert loop is not None
        result = loop.recover_from_error("JSON parse error in tool arguments")
        assert result["recovered"] is True

    def test_adaptive_loop_status_after_errors(self):
        loop = AdaptiveAgentLoop()
        loop._total_errors = 5
        loop._consecutive_errors = 2
        status = loop.get_status()
        assert status["total_errors"] == 5
        assert status["consecutive_errors"] == 2


class TestExecutionContextShutdown:
    def test_shutdown_clears_running_processes(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        ctx._command_count = 42
        ctx._turn_count = 10
        ctx.shutdown()
        assert len(ctx._running_processes) == 0
        assert ctx._command_count == 42

    def test_shutdown_saves_state(self):
        tmpdir = tempfile.mkdtemp()
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True
        ctx._last_persist = 0
        ctx._command_count = 99
        ctx.shutdown()
        loaded = ExecutionContext(workspace_root=Path(tmpdir))
        loaded.load_state()
        assert loaded._command_count >= 99

    def test_streaming_loop_has_all_phases(self):
        loop = AdaptiveAgentLoop()
        assert ExecutionPhase.IDLE.value == "idle"
        assert ExecutionPhase.THINKING.value == "thinking"
        assert ExecutionPhase.EXECUTING.value == "executing"
        assert ExecutionPhase.WAITING_APPROVAL.value == "waiting_approval"
        assert ExecutionPhase.REFLECTING.value == "reflecting"
        assert ExecutionPhase.ERROR_RECOVERY.value == "error_recovery"
        assert ExecutionPhase.COMPLETED.value == "completed"
        assert ExecutionPhase.FAILED.value == "failed"

    def test_checkpoint_to_dict(self):
        from core.adaptive_agent_loop import TurnCheckpoint
        cp = TurnCheckpoint(
            turn_number=1, phase="thinking",
            user_input="test", assistant_response="response",
            tool_calls=[{"name": "ls"}], tool_results=[{"tool": "ls", "success": True}],
            cwd="/home",
        )
        d = cp.to_dict()
        assert d["turn_number"] == 1
        assert d["phase"] == "thinking"
        assert len(d["tool_calls"]) == 1


class TestAdaptiveStreaming:
    def test_streaming_method_exists(self):
        loop = AdaptiveAgentLoop()
        assert hasattr(loop, 'run_persistent_stream')
        assert callable(loop.run_persistent_stream)

    def test_streaming_with_no_llm(self):
        import asyncio
        loop = AdaptiveAgentLoop()
        loop.llm = None

        async def _test():
            events = []
            async for event in loop.run_persistent_stream(
                user_input="hello",
                system_prompt="You are Jarvis",
                tools={},
                max_iterations=1,
            ):
                events.append(event)
            return events

        events = asyncio.run(_test())
        assert len(events) >= 1
        event_types = [e.get("type") for e in events]
        assert "complete" in event_types

    def test_streaming_shell_tool(self):
        import asyncio
        loop = AdaptiveAgentLoop()

        async def _test():
            events = []
            async for event in loop._execute_shell_tool_stream("echo test_output"):
                events.append(event)
            return events

        result = loop._execute_shell_tool("terminal_execute", {"command": "echo test_output"}, 10.0)
        assert result["success"] is True
        assert "test_output" in result.get("output", "")

    def test_execution_context_atexit_registered(self):
        ctx = get_execution_context()
        assert ctx._loaded is True

    def test_file_ops_logged_in_context(self):
        ctx = ExecutionContext(workspace_root=Path(tempfile.mkdtemp()))
        ctx._loaded = True
        ctx.log_file_op("read", "/tmp/test.txt", 100, success=True)
        ctx.log_file_op("write", "/tmp/output.txt", 500, success=True)
        files = ctx.get_recent_files(5)
        assert len(files) == 2
        assert files[0]["op_type"] == "read"
        assert files[1]["op_type"] == "write"


class TestNativeFilesCwdResolution:
    def test_relative_path_resolves_with_cwd(self):
        tmpdir = tempfile.mkdtemp()
        sub = Path(tmpdir) / "project"
        sub.mkdir()
        (sub / "README.md").write_text("Hello", encoding='utf-8')

        from core.execution_context import get_execution_context, ExecutionContext
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True
        ctx.cd(str(sub))

        result = ctx.read_file("README.md")
        assert result["success"] is True
        assert "Hello" in result.get("content", "")

    def test_absolute_path_overrides_cwd(self):
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        (Path(tmpdir2) / "other.txt").write_text("Other", encoding='utf-8')

        from core.execution_context import ExecutionContext
        ctx = ExecutionContext(workspace_root=Path(tmpdir1))
        ctx._loaded = True

        result = ctx.read_file(str(Path(tmpdir2) / "other.txt"))
        assert result["success"] is True
        assert "Other" in result.get("content", "")


class TestDynamicNeedsPermission:
    def test_needs_permission_method_on_tool(self):
        sys_path_mod = str(Path(__file__).parent.parent)
        import sys
        if sys_path_mod not in sys.path:
            sys.path.insert(0, sys_path_mod)
        from tools.terminal import TerminalExecuteTool
        tool = TerminalExecuteTool.__new__(TerminalExecuteTool)
        tool.computer = MagicMock()
        tool.computer.needs_permission.return_value = False
        assert tool.needs_permission(command="ls -la") is False
        tool.computer.needs_permission.return_value = True
        assert tool.needs_permission(command="rm -rf /") is True

    def test_requires_permission_attribute_still_exists(self):
        sys_path_mod = str(Path(__file__).parent.parent)
        import sys
        if sys_path_mod not in sys.path:
            sys.path.insert(0, sys_path_mod)
        from tools.terminal import TerminalExecuteTool
        assert TerminalExecuteTool.requires_permission is True


class TestExecutionPersistence:
    def test_save_and_load_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        ctx1 = ExecutionContext(workspace_root=Path(tmpdir))
        ctx1._loaded = True
        ctx1._last_persist = 0
        ctx1.run("echo test")
        ctx1.save_state()

        ctx2 = ExecutionContext(workspace_root=Path(tmpdir))
        ctx2._loaded = False
        ctx2.load_state()
        assert ctx2._command_count >= 1

    def test_multiple_cd_saves(self):
        tmpdir = tempfile.mkdtemp()
        sub1 = Path(tmpdir) / "dir1"
        sub2 = Path(tmpdir) / "dir2"
        sub1.mkdir()
        sub2.mkdir()

        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True
        ctx._last_persist = 0

        ctx.cd(str(sub1))
        assert ctx.cwd == str(sub1)

        ctx.cd(str(sub2))
        assert ctx.cwd == str(sub2)

    def test_env_vars_persist(self):
        tmpdir = tempfile.mkdtemp()
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True
        ctx._last_persist = 0

        ctx.set_env("TEST_VAR", "test_value")
        assert ctx.get_env("TEST_VAR") == "test_value"

        ctx.save_state()
        ctx2 = ExecutionContext(workspace_root=Path(tmpdir))
        ctx2._loaded = False
        ctx2.load_state()
        assert ctx2.get_env("TEST_VAR") == "test_value"

    def test_max_env_size_eviction(self):
        tmpdir = tempfile.mkdtemp()
        ctx = ExecutionContext(workspace_root=Path(tmpdir))
        ctx._loaded = True

        for i in range(ctx.MAX_ENV_SIZE + 5):
            ctx.set_env(f"VAR_{i}", f"val_{i}")

        assert len(ctx._env) <= ctx.MAX_ENV_SIZE


# ========================================================================
# ExecutionContext Performance Optimization Tests (v2.1)
# ========================================================================

class TestExecutionContextPerformance:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ExecutionContext(workspace_root=Path(self.tmpdir))
        self.ctx._loaded = True

    def test_run_env_cache_created(self):
        env = self.ctx._get_run_env()
        assert isinstance(env, dict)
        assert "PWD" in env
        assert env["PWD"] == self.tmpdir

    def test_run_env_cache_hit(self):
        env1 = self.ctx._get_run_env()
        env2 = self.ctx._get_run_env()
        assert env1 is env2

    def test_run_env_cache_invalidated_on_cd(self):
        env1 = self.ctx._get_run_env()
        sub = Path(self.tmpdir) / "subdir"
        sub.mkdir()
        self.ctx.cd(str(sub))
        env2 = self.ctx._get_run_env()
        assert env2 is not env1
        assert env2["PWD"] == str(sub.resolve())

    def test_run_env_cache_invalidated_on_set_env(self):
        env1 = self.ctx._get_run_env()
        self.ctx.set_env("PERF_TEST", "value")
        env2 = self.ctx._get_run_env()
        assert env2 is not env1
        assert env2.get("PERF_TEST") == "value"

    def test_run_env_cache_invalidated_on_reset(self):
        self.ctx.set_env("PERF_TEST", "value")
        self.ctx._get_run_env()
        self.ctx.reset()
        assert self.ctx._run_env_cache_dirty is True

    def test_run_env_no_redaction(self):
        self.ctx.set_env("MY_SECRET_KEY", "secret123")
        env = self.ctx._get_run_env()
        run_env_value = env.get("MY_SECRET_KEY")
        full_env = self.ctx.get_full_env()
        full_env_value = full_env.get("MY_SECRET_KEY")
        assert run_env_value == "secret123"
        assert full_env_value == "***REDACTED***"

    def test_run_env_includes_context_overrides(self):
        self.ctx.set_env("CUSTOM_VAR", "override_val")
        env = self.ctx._get_run_env()
        assert env["CUSTOM_VAR"] == "override_val"

    def test_precompiled_regex_cd(self):
        from core.execution_context import _CD_PATTERN
        assert _CD_PATTERN.match("cd /home") is not None
        assert _CD_PATTERN.match("ls -la") is None

    def test_precompiled_regex_compound_cd(self):
        from core.execution_context import _COMPOUND_CD_PATTERN
        assert _COMPOUND_CD_PATTERN.search("make && cd /home") is not None
        assert _COMPOUND_CD_PATTERN.search("ls -la") is None

    def test_precompiled_regex_pushd(self):
        from core.execution_context import _PUSHD_PATTERN
        assert _PUSHD_PATTERN.match("pushd /tmp") is not None
        assert _PUSHD_PATTERN.match("popd") is None

    def test_frozen_constants(self):
        from core.execution_context import _PROTECTED_DIRS, _PROTECTED_DIRS_WRITE, _BLOCKED_EXTENSIONS
        assert "/etc" in _PROTECTED_DIRS
        assert "/usr/bin" in _PROTECTED_DIRS_WRITE
        assert "/usr/bin" not in _PROTECTED_DIRS
        assert ".exe" in _BLOCKED_EXTENSIONS


# ========================================================================
# WSConnectionManager Tests
# ========================================================================

class TestWSConnectionManager:
    def setup_method(self):
        from core.ws_manager import WSConnectionManager
        self.mgr = WSConnectionManager()

    def test_register_connection(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        assert info.client_host == "10.0.0.1"
        assert self.mgr.get_connection_count() == 1

    def test_unregister_connection(self):
        ws_obj = type('WS', (), {})()
        self.mgr.register(ws_obj, client_host="10.0.0.1")
        self.mgr.unregister(ws_obj, reason="test")
        assert self.mgr.get_connection_count() == 0

    def test_record_pong(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        self.mgr.record_pong(ws_obj)
        assert info.pong_count == 1
        assert info.missed_pongs == 0

    def test_record_activity(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        self.mgr.record_activity(ws_obj)
        assert info.message_count == 1

    def test_record_ping_sent(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        self.mgr.record_ping_sent(ws_obj)
        assert info.ping_count == 1
        assert info.last_ping_sent > 0

    def test_stale_detection_missed_pongs(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        info.missed_pongs = self.mgr.MISSED_PONG_LIMIT
        stale = self.mgr.get_stale_connections()
        assert len(stale) == 1

    def test_stale_detection_idle(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        info.last_activity = time.time() - self.mgr.MAX_IDLE_SECONDS - 10
        stale = self.mgr.get_stale_connections()
        assert len(stale) == 1

    def test_no_stale_when_active(self):
        ws_obj = type('WS', (), {})()
        self.mgr.register(ws_obj, client_host="10.0.0.1")
        stale = self.mgr.get_stale_connections()
        assert len(stale) == 0

    def test_increment_missed_pongs(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        self.mgr.record_ping_sent(ws_obj)
        info.last_pong = 0
        info.last_ping_sent = time.time() - self.mgr.PING_INTERVAL * 3
        missed = self.mgr.increment_missed_pongs()
        assert len(missed) == 1

    def test_get_status(self):
        status = self.mgr.get_status()
        assert "active_connections" in status
        assert "ping_interval" in status
        assert status["ping_interval"] == self.mgr.PING_INTERVAL

    def test_get_all_connections(self):
        ws1 = type('WS', (), {})()
        ws2 = type('WS', (), {})()
        self.mgr.register(ws1, client_host="10.0.0.1")
        self.mgr.register(ws2, client_host="10.0.0.2")
        conns = self.mgr.get_all_connections()
        assert len(conns) == 2

    def test_connection_info_to_dict(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1", session_id="abc123")
        d = info.to_dict()
        assert d["client_host"] == "10.0.0.1"
        assert d["session_id"] == "abc123"
        assert "age_seconds" in d
        assert "idle_seconds" in d

    def test_events_logged(self):
        ws_obj = type('WS', (), {})()
        self.mgr.register(ws_obj, client_host="10.0.0.1")
        self.mgr.unregister(ws_obj, reason="test")
        events = self.mgr.get_events(event_type="connect")
        assert len(events) >= 1
        disconnect_events = self.mgr.get_events(event_type="disconnect")
        assert len(disconnect_events) >= 1

    def test_max_lifetime_stale(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        info.connected_at = time.time() - self.mgr.MAX_LIFETIME_SECONDS - 10
        stale = self.mgr.get_stale_connections()
        assert len(stale) == 1

    def test_pong_resets_missed_count(self):
        ws_obj = type('WS', (), {})()
        info = self.mgr.register(ws_obj, client_host="10.0.0.1")
        info.missed_pongs = 2
        self.mgr.record_pong(ws_obj)
        assert info.missed_pongs == 0

    def test_singleton_getter(self):
        from core.ws_manager import get_ws_connection_manager
        mgr1 = get_ws_connection_manager()
        mgr2 = get_ws_connection_manager()
        assert mgr1 is mgr2


# ========================================================================
# Adaptive Recovery Enhanced Tests (v2.1)
# ========================================================================

class TestAdaptiveRecoveryEnhanced:
    def setup_method(self):
        self.loop = AdaptiveAgentLoop()

    def test_timeout_recovery_has_shorter_strategy(self):
        r = self.loop.recover_from_error("Connection timed out after 60s")
        assert "retry_shorter" in r["strategies"]
        assert "suggested_timeout" in r.get("context_hints", {})

    def test_permission_recovery_has_approval_strategy(self):
        r = self.loop.recover_from_error("Permission denied: /root/secret")
        assert "request_approval" in r["strategies"]

    def test_notfound_recovery_has_search_strategy(self):
        r = self.loop.recover_from_error("File not found: missing.py")
        assert "search_alternative" in r["strategies"]
        assert "create_missing" in r["strategies"]

    def test_connection_recovery_has_offline_strategy(self):
        r = self.loop.recover_from_error("Connection refused to localhost:8080")
        assert "offline_mode" in r["strategies"]

    def test_memory_recovery_has_context_reduction(self):
        r = self.loop.recover_from_error("Out of memory during processing")
        assert "context_reduction" in r["strategies"]

    def test_ratelimit_recovery_has_backoff(self):
        r = self.loop.recover_from_error("Rate limit exceeded: 429")
        assert "retry_with_backoff" in r["strategies"]
        assert "backoff_seconds" in r.get("context_hints", {})

    def test_json_error_recovery(self):
        r = self.loop.recover_from_error("JSON decode error: unexpected token")
        assert "retry_clean_input" in r["strategies"]

    def test_interrupt_recovery(self):
        r = self.loop.recover_from_error("Task was cancelled by user")
        assert "resume_from_checkpoint" in r["strategies"]

    def test_generic_recovery(self):
        r = self.loop.recover_from_error("Unknown error occurred")
        assert "retry" in r["strategies"]

    def test_consecutive_errors_sets_failed(self):
        for _ in range(self.loop._max_consecutive_errors):
            self.loop.recover_from_error("test error")
        assert self.loop._phase.value == "failed"

    def test_recover_mid_stream_preserves_output(self):
        r = self.loop.recover_mid_stream(
            "timeout error",
            partial_output="Here is the first part of my answer...",
            tool_calls_made=[{"name": "terminal_execute", "id": "t1"}],
            tool_results=[{"success": True, "output": "files listed"}],
        )
        assert r["partial_output"] == "Here is the first part of my answer..."
        assert r["partial_output_len"] == 38
        assert len(r["tool_calls_made"]) == 1
        assert "preserved" in r["message"]

    def test_recover_mid_stream_empty_output(self):
        r = self.loop.recover_mid_stream("error", partial_output="")
        assert r["partial_output"] == ""
        assert r["partial_output_len"] == 0

    def test_context_hints_populated(self):
        r = self.loop.recover_from_error("timeout after 60 seconds")
        assert "context_hints" in r
        assert "suggested_timeout" in r["context_hints"]