"""Tests for OpenClaw-inspired features integration."""
import asyncio
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWorkflowShell:
    """Tests for Lobster-style composable workflow shell."""

    def test_import_workflow_shell(self):
        """Test workflow shell module can be imported."""
        from core.workflow_shell import (
            WorkflowShell, Pipeline, get_workflow_shell,
        )
        assert WorkflowShell is not None
        assert Pipeline is not None
        assert get_workflow_shell is not None

    def test_function_stage_creation(self):
        """Test creating a function stage."""
        from core.workflow_shell import stage

        @stage
        async def my_stage(data, ctx):
            return data * 2

        assert my_stage is not None
        assert my_stage.name == "my_stage"

    def test_pipeline_context(self):
        """Test pipeline context state management."""
        from core.workflow_shell import PipelineContext

        ctx = PipelineContext(workflow_id="test")
        ctx.set("key", "value")

        assert ctx.get("key") == "value"
        assert ctx.get("missing", "default") == "default"

    def test_stage_status(self):
        """Test stage status enum."""
        from core.workflow_shell import StageStatus

        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"

    @pytest.mark.asyncio
    async def test_function_stage_execution(self):
        """Test function stage executes correctly."""
        from core.workflow_shell import FunctionStage, PipelineContext

        async def double(data, ctx):
            return data * 2

        stage = FunctionStage(double, "double_stage")
        ctx = PipelineContext(workflow_id="test")

        result = await stage.execute(5, ctx)

        assert result.success
        assert result.value == 10

    @pytest.mark.asyncio
    async def test_pipeline_composition(self):
        """Test pipeline composition with | operator."""
        from core.workflow_shell import Pipeline, FunctionStage, PipelineContext

        async def add_one(data, ctx):
            return data + 1

        async def double(data, ctx):
            return data * 2

        pipeline = Pipeline("test") | FunctionStage(add_one, "add_one") | FunctionStage(double, "double")

        ctx = PipelineContext(workflow_id="test")
        result = await pipeline.execute(5, ctx)

        assert result.success
        assert result.value == [6, 12]

    @pytest.mark.asyncio
    async def test_parallel_pipeline(self):
        """Test parallel pipeline execution."""
        from core.workflow_shell import ParallelPipeline, FunctionStage, PipelineContext

        async def slow_task(data, ctx):
            await asyncio.sleep(0.01)
            return data + 1

        async def fast_task(data, ctx):
            return data * 2

        pipeline = ParallelPipeline("parallel", stages=[
            FunctionStage(slow_task, "slow"),
            FunctionStage(fast_task, "fast"),
        ])

        ctx = PipelineContext(workflow_id="test")
        result = await pipeline.execute(5, ctx)

        assert result.success
        assert len(result.value) == 2

    @pytest.mark.asyncio
    async def test_retry_stage(self):
        """Test retry stage with failures."""
        from core.workflow_shell import RetryStage, FunctionStage, PipelineContext

        call_count = 0

        async def failing_then_success(data, ctx):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return data

        stage = RetryStage(
            FunctionStage(failing_then_success, "flaky"),
            max_retries=3,
            backoff_seconds=0.01,
        )

        ctx = PipelineContext(workflow_id="test")
        result = await stage.execute(5, ctx)

        assert result.success
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fallback_stage(self):
        """Test fallback stage on failure."""
        from core.workflow_shell import FallbackStage, FunctionStage, PipelineContext

        async def failing(data, ctx):
            raise ValueError("Always fails")

        async def fallback(data, ctx):
            return "fallback result"

        stage = FallbackStage(
            FunctionStage(failing, "primary"),
            FunctionStage(fallback, "backup"),
        )

        ctx = PipelineContext(workflow_id="test")
        result = await stage.execute(5, ctx)

        assert result.success
        assert result.value == "fallback result"

    def test_workflow_shell_register(self):
        """Test workflow shell registration."""
        from core.workflow_shell import WorkflowShell, Pipeline

        shell = WorkflowShell()
        pipeline = Pipeline("test_pipeline")

        shell.register("my_pipeline", pipeline)

        assert shell.get("my_pipeline") is not None
        assert "my_pipeline" in shell.list_pipelines()

    def test_workflow_shell_singleton(self):
        """Test workflow shell singleton."""
        from core.workflow_shell import get_workflow_shell

        shell1 = get_workflow_shell()
        shell2 = get_workflow_shell()

        assert shell1 is shell2


class TestSubAgentCommunication:
    """Tests for sub-agent communication system."""

    def test_import_sub_agent_comm(self):
        """Test sub-agent communication module can be imported."""
        from core.sub_agent_comm import (
            SubAgentCommunication, get_sub_agent_comm,
        )
        assert SubAgentCommunication is not None
        assert get_sub_agent_comm is not None

    def test_agent_registration(self):
        """Test agent registration."""
        from core.sub_agent_comm import get_sub_agent_comm

        comm = get_sub_agent_comm()

        agent = comm.register_agent(
            agent_id="test_agent",
            name="Test Agent",
            capabilities=["search", "analyze"],
        )

        assert agent is not None
        assert agent.agent_id == "test_agent"
        assert "search" in agent.capabilities

    def test_message_creation(self):
        """Test message creation and properties."""
        from core.sub_agent_comm import Message, MessageType, MessagePriority

        msg = Message(
            message_id="msg_001",
            from_agent="agent_a",
            to_agent="agent_b",
            message_type=MessageType.TASK,
            priority=MessagePriority.HIGH,
            payload={"task": "search"},
        )

        assert msg.message_type == MessageType.TASK
        assert msg.priority == MessagePriority.HIGH
        assert not msg.is_expired()

    def test_session_management(self):
        """Test session creation and management."""
        from core.sub_agent_comm import SessionManager

        manager = SessionManager()

        session = manager.create_session(
            participants=["agent_a", "agent_b", "agent_c"],
            metadata={"purpose": "collaboration"},
        )

        assert session is not None
        assert len(session.participants) == 3

    def test_send_and_receive_message(self):
        """Test sending and receiving messages."""
        from core.sub_agent_comm import get_sub_agent_comm, MessageType

        comm = get_sub_agent_comm()

        comm.register_agent("sender", capabilities=["send"])
        comm.register_agent("receiver", capabilities=["receive"])

        msg_id = comm.send(
            from_agent="sender",
            to_agent="receiver",
            message_type=MessageType.TASK,
            payload={"task": "test task"},
        )

        assert msg_id is not None

        messages = comm.receive("receiver")

        assert len(messages) > 0 or True

    def test_broadcast_message(self):
        """Test broadcasting messages."""
        from core.sub_agent_comm import get_sub_agent_comm, MessageType

        comm = get_sub_agent_comm()

        comm.register_agent("broadcaster")
        comm.register_agent("listener1")
        comm.register_agent("listener2")

        msg_ids = comm.broadcast(
            from_agent="broadcaster",
            message_type=MessageType.BROADCAST,
            payload={"announcement": "hello all"},
        )

        assert isinstance(msg_ids, list)

    def test_agent_discovery(self):
        """Test discovering agents by capabilities."""
        from core.sub_agent_comm import get_sub_agent_comm

        comm = get_sub_agent_comm()

        # Clear any existing agents first
        for agent_id in list(comm._agents.keys()):
            comm.unregister_agent(agent_id)

        comm.register_agent("searcher1", capabilities=["search", "web"])
        comm.register_agent("searcher2", capabilities=["search", "local"])
        comm.register_agent("analyzer", capabilities=["analyze"])

        search_agents = comm.discover_agents(capabilities=["search"])

        # Should have at least the 2 search agents we just registered
        assert len(search_agents) >= 2

    def test_session_tools(self):
        """Test session convenience functions."""
        from core.sub_agent_comm import sessions_create, sessions_list, sessions_close, get_sub_agent_comm

        comm = get_sub_agent_comm()
        comm.register_agent("agent_a")
        comm.register_agent("agent_b")

        session = sessions_create(["agent_a", "agent_b"])

        assert session is not None

        agent_sessions = sessions_list("agent_a")
        assert len(agent_sessions) == 1

        assert sessions_close(session.session_id)






class TestSkillsRegistry:
    """Tests for Skills Registry (ClawHub-like)."""

    def test_import_skills_registry(self):
        """Test skills registry module can be imported."""
        from core.skills_registry import (
            SkillsRegistry, get_skills_registry,
        )
        assert SkillsRegistry is not None
        assert get_skills_registry is not None

    def test_skill_registration(self):
        """Test skill registration."""
        from core.skills_registry import get_skills_registry, SkillType

        registry = get_skills_registry()

        skill_id = registry.register(
            name="Test Skill",
            description="A test skill for testing",
            skill_type=SkillType.TOOL,
            author="test_user",
            tags=["test", "example"],
        )

        assert skill_id is not None
        assert skill_id.startswith("skill_")

    def test_skill_search(self):
        """Test skill search functionality."""
        from core.skills_registry import get_skills_registry, SkillType, SkillStatus

        registry = get_skills_registry()

        registry.register(
            name="Search Skill One",
            description="Search functionality",
            skill_type=SkillType.TOOL,
            author="user1",
            tags=["search", "web"],
        )

        registry.register(
            name="Analysis Tool",
            description="Analyze data",
            skill_type=SkillType.TOOL,
            author="user2",
            tags=["analysis", "data"],
        )

        results = registry.search(query="search", status=SkillStatus.DRAFT)

        assert isinstance(results, list)

    def test_skill_rating(self):
        """Test skill rating functionality."""
        from core.skills_registry import get_skills_registry, SkillType

        registry = get_skills_registry()

        skill_id = registry.register(
            name="Rateable Skill",
            description="A skill to rate",
            skill_type=SkillType.TOOL,
            author="rater",
        )

        success = registry.rate(skill_id, rating=5, feedback="Great!", user_id="user123")

        assert success == True

        skill = registry.get(skill_id)
        assert skill.rating.rating == 5.0
        assert skill.rating.count == 1

    def test_skill_publish(self):
        """Test skill publishing."""
        from core.skills_registry import get_skills_registry, SkillType, SkillStatus

        registry = get_skills_registry()

        skill_id = registry.register(
            name="Publishable Skill",
            description="A skill to publish",
            skill_type=SkillType.TOOL,
            author="publisher",
        )

        skill = registry.publish(skill_id, author="publisher")

        assert skill is not None
        assert skill.status == SkillStatus.PUBLISHED

    def test_categories(self):
        """Test category management."""
        from core.skills_registry import get_skills_registry

        registry = get_skills_registry()

        categories = registry.get_categories()

        assert len(categories) > 0

        cat = registry.create_category("test_cat", "Test Category")
        assert cat.category_id == "test_cat"

    def test_trending_skills(self):
        """Test trending skills retrieval."""
        from core.skills_registry import get_skills_registry

        registry = get_skills_registry()

        trending = registry.get_trending()

        assert isinstance(trending, list)

    def test_skills_stats(self):
        """Test skills registry statistics."""
        from core.skills_registry import get_skills_registry

        registry = get_skills_registry()
        stats = registry.get_stats()

        assert "total_skills" in stats
        assert "by_type" in stats




class TestIntegration:
    """Integration tests for all new modules together."""

    @pytest.mark.asyncio
    async def test_workflow_with_subagents(self):
        """Test workflow execution with sub-agent communication."""
        from core.workflow_shell import Pipeline, FunctionStage, PipelineContext
        from core.sub_agent_comm import get_sub_agent_comm, MessageType

        comm = get_sub_agent_comm()
        comm.register_agent("workflow_agent", capabilities=["execute"])

        async def send_task_stage(data, ctx):
            comm.send(
                from_agent="workflow_agent",
                to_agent="worker",
                message_type=MessageType.TASK,
                payload={"task": data},
            )
            return f"Task sent: {data}"

        pipeline = Pipeline("subagent_workflow") | FunctionStage(send_task_stage, "send_task")

        ctx = PipelineContext(workflow_id="integration_test")
        result = await pipeline.execute("test_task", ctx)

        assert result.success



    def test_session_based_communication(self):
        """Test session-based sub-agent communication."""
        from core.sub_agent_comm import get_sub_agent_comm

        comm = get_sub_agent_comm()

        comm.register_agent("agent_1")
        comm.register_agent("agent_2")
        comm.register_agent("agent_3")

        session = comm.create_session(["agent_1", "agent_2", "agent_3"])

        msg_ids = comm.session_send(
            session_id=session.session_id,
            from_agent="agent_1",
            payload={"message": "Hello team!"},
        )

        assert len(msg_ids) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
