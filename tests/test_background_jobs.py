"""Tests for background jobs (run_in_background semantics).

Verifies the job registry on JarvisComputer, the terminal tool's
background branch, and the check_jobs tool — without waiting long:
the probe command is the interpreter itself printing a value.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.terminal import BackgroundJobsTool, JarvisComputer, TerminalExecuteTool  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_globals():
    """Fresh attention queue + event bus per test (job completion posts
    to both; other suites read the same singletons)."""
    from core.attention_priority import get_attention_queue, reset_attention_queue
    from core.event_bus import get_event_bus

    get_event_bus().clear()
    reset_attention_queue()
    yield
    reset_attention_queue()
    get_event_bus().clear()


def _probe_command(text: str = "42") -> str:
    # No shell metacharacters and no backslashes (shlex treats \ as an
    # escape on Windows paths): forward slashes work for the OS.
    exe = sys.executable.replace("\\", "/")
    return f"{exe} -c print({text})"


async def _wait_for(job_check, timeout: float = 30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        snapshot = job_check()
        if snapshot and snapshot["status"] != "running":
            return snapshot
        await asyncio.sleep(0.2)
    raise AssertionError("job did not finish in time")


class TestBackgroundJobRegistry:
    def test_start_and_complete(self):
        computer = JarvisComputer()

        async def run():
            started, job_id = computer.start_background(_probe_command())
            assert started is True
            assert job_id.startswith("bg-")
            job = await _wait_for(lambda: computer.get_background_job(job_id))
            return job

        job = asyncio.new_event_loop().run_until_complete(run())
        assert job["status"] == "completed"
        assert job["exit_code"] == 0
        assert "42" in job["stdout"]

    def test_failed_job_records_exit_code(self):
        computer = JarvisComputer()
        # print(1/0) raises ZeroDivisionError -> nonzero exit
        command = _probe_command("1/0")

        async def run():
            started, job_id = computer.start_background(command)
            assert started is True
            return await _wait_for(lambda: computer.get_background_job(job_id))

        job = asyncio.new_event_loop().run_until_complete(run())
        assert job["status"] == "failed"
        assert job["exit_code"] != 0
        assert "ZeroDivisionError" in job["stderr"]

    def test_list_snapshots_hide_task_handle(self):
        computer = JarvisComputer()

        async def run():
            computer.start_background(_probe_command())
            jobs = computer.list_background_jobs()
            assert jobs, "expected at least one job"
            assert "task" not in jobs[0]
            await _wait_for(lambda: computer.get_background_job(jobs[0]["id"]))

        asyncio.new_event_loop().run_until_complete(run())

    def test_unknown_job_is_none(self):
        assert JarvisComputer().get_background_job("bg-999") is None


class TestBackgroundTools:
    def test_terminal_tool_background_branch(self):
        tool = TerminalExecuteTool()

        async def run():
            result = await tool.execute(command=_probe_command(), background=True)
            assert "Background job started" in result
            job_id = result.split("Background job started: ")[1].split("\n")[0]
            await _wait_for(lambda: tool.computer.get_background_job(job_id))
            return job_id

        job_id = asyncio.new_event_loop().run_until_complete(run())
        # check_jobs output mode
        checker = BackgroundJobsTool(computer=tool.computer)
        output = asyncio.new_event_loop().run_until_complete(
            checker.execute(job_id=job_id))
        assert "completed" in output
        assert "42" in output

    def test_check_jobs_list_mode(self):
        computer = JarvisComputer()
        checker = BackgroundJobsTool(computer=computer)

        async def run():
            computer.start_background(_probe_command())
            listing = await checker.execute()
            job_id = computer.list_background_jobs()[0]["id"]
            await _wait_for(lambda: computer.get_background_job(job_id))
            return listing

        listing = asyncio.new_event_loop().run_until_complete(run())
        assert "bg-" in listing

    def test_check_jobs_empty(self):
        checker = BackgroundJobsTool(computer=JarvisComputer())
        output = asyncio.new_event_loop().run_until_complete(checker.execute())
        assert output == "No background jobs."

    def test_check_jobs_unknown_id(self):
        checker = BackgroundJobsTool(computer=JarvisComputer())
        output = asyncio.new_event_loop().run_until_complete(
            checker.execute(job_id="bg-424242"))
        assert "No such background job" in output


async def _wait_attention(timeout: float = 15.0):
    from core.attention_priority import get_attention_queue
    queue = get_attention_queue()
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if queue.size() > 0:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("no attention item posted for finished job")


class TestBackgroundCompletionWakeup:
    """Stage E: a finished job must surface without the agent polling."""

    def test_completion_enqueues_attention_item(self):
        from core.attention_priority import get_attention_queue
        computer = JarvisComputer()

        async def run():
            _, job_id = computer.start_background(_probe_command())
            await _wait_for(lambda: computer.get_background_job(job_id))
            await _wait_attention()
            return job_id

        job_id = asyncio.new_event_loop().run_until_complete(run())
        item = get_attention_queue().dequeue()
        assert item is not None
        assert item.source == "background_job"
        assert item.urgency.name == "FOLLOW_UP"
        assert job_id in item.content
        assert "completed" in item.content

    def test_failure_uses_system_alert(self):
        from core.attention_priority import get_attention_queue
        computer = JarvisComputer()

        async def run():
            _, job_id = computer.start_background(_probe_command("1/0"))
            await _wait_for(lambda: computer.get_background_job(job_id))
            await _wait_attention()
            return job_id

        asyncio.new_event_loop().run_until_complete(run())
        item = get_attention_queue().dequeue()
        assert item is not None
        assert item.urgency.name == "SYSTEM_ALERT"
        assert "failed" in item.content

    def test_completion_publishes_event(self):
        from core.event_bus import get_event_bus
        received = []
        get_event_bus().subscribe(
            "job.background.completed", lambda e: received.append(e))
        computer = JarvisComputer()

        async def run():
            _, job_id = computer.start_background(_probe_command())
            await _wait_for(lambda: computer.get_background_job(job_id))
            await _wait_attention()
            return job_id

        job_id = asyncio.new_event_loop().run_until_complete(run())
        assert received, "expected job.background.completed event"
        assert received[0].data["job_id"] == job_id
        assert received[0].data["status"] == "completed"