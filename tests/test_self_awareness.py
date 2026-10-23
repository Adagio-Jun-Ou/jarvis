"""Unit tests for self_awareness.py module."""
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.self_awareness import (
    SelfAwarenessEngine,
    ModelIdentity,
    EnvironmentContext,
    SessionState,
    CapabilityManifest,
    create_self_awareness,
)


class TestModelIdentity:
    """Tests for ModelIdentity dataclass."""

    def test_model_identity_creation(self):
        """Test basic model identity creation."""
        identity = ModelIdentity(
            provider="ollama",
            model_id="llama3.2",
            model_name="Llama 3.2",
            context_window=128000,
            capabilities=["text", "code"],
        )
        assert identity.provider == "ollama"
        assert identity.model_id == "llama3.2"
        assert identity.context_window == 128000

    def test_model_identity_to_context(self):
        """Test model identity context generation."""
        identity = ModelIdentity(
            provider="openai",
            model_id="gpt-4",
            model_name="GPT-4",
            context_window=8192,
            capabilities=["text", "vision"],
        )
        context = identity.to_context()
        assert "gpt-4" in context
        assert "openai/gpt-4" in context
        assert "8,192" in context
        assert "text, vision" in context

    def test_model_identity_no_capabilities(self):
        """Test model identity with no capabilities specified."""
        identity = ModelIdentity(
            provider="anthropic",
            model_id="claude-3",
            model_name="Claude 3",
        )
        context = identity.to_context()
        assert "standard text generation" in context

    def test_model_identity_empty_capabilities(self):
        """Test model identity with empty capabilities list."""
        identity = ModelIdentity(
            provider="test",
            model_id="test-model",
            model_name="Test Model",
            capabilities=[],
        )
        context = identity.to_context()
        assert "standard text generation" in context


class TestEnvironmentContext:
    """Tests for EnvironmentContext dataclass."""

    def test_environment_context_to_context(self):
        """Test environment context generation."""
        env = EnvironmentContext(
            working_directory="/home/user/project",
            is_git_repo=True,
            git_branch="main",
            platform_name="Linux",
            platform_version="6.5.0",
            hostname="dev-machine",
            username="testuser",
            current_date="2026-02-15",
            current_time="12:00:00",
            timezone="UTC",
            python_version="3.11.0",
        )
        context = env.to_context()
        assert "<env>" in context
        assert "</env>" in context
        assert "/home/user/project" in context
        assert "yes" in context  # git repo
        assert "main" in context  # git branch
        assert "Linux" in context

    def test_environment_context_no_git(self):
        """Test environment without git."""
        env = EnvironmentContext(
            working_directory="/home/user/documents",
            is_git_repo=False,
            git_branch=None,
            platform_name="Windows",
            platform_version="10",
            hostname="win-pc",
            username="user",
            current_date="2026-02-15",
            current_time="12:00:00",
            timezone="EST",
            python_version="3.10.0",
        )
        context = env.to_context()
        assert "no" in context  # not a git repo
        assert "Current git branch" not in context


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_session_state_creation(self):
        """Test session state creation."""
        state = SessionState(
            session_id="abc123def456",
            status="thinking",
            turn_count=5,
            messages_count=10,
            tools_used_this_session=["read", "write"],
            topics_discussed=["api", "testing"],
            active_tools=["read", "write", "search"],
            pending_permission=None,
        )
        assert state.session_id == "abc123def456"
        assert state.status == "thinking"
        assert state.turn_count == 5

    def test_session_state_to_context(self):
        """Test session state context generation."""
        state = SessionState(
            session_id="test-session-id-123",
            status="idle",
            turn_count=3,
            messages_count=6,
            tools_used_this_session=["read"],
            topics_discussed=["python", "testing"],
            active_tools=["read", "write"],
            pending_permission={"tool": "write", "args": {"path": "/tmp/test"}},
        )
        context = state.to_context()
        assert "<session_state>" in context
        assert "</session_state>" in context
        assert "test-ses" in context  # truncated session ID (8 chars)
        assert "idle" in context
        assert "Turn count: 3" in context
        assert "read" in context
        assert "Pending permission" in context

    def test_session_state_no_tools_used(self):
        """Test session state with no tools used."""
        state = SessionState(
            session_id="new-session",
            status="idle",
            turn_count=0,
            messages_count=1,
            tools_used_this_session=[],
            topics_discussed=[],
            active_tools=["read"],
            pending_permission=None,
        )
        context = state.to_context()
        assert "none" in context.lower()  # no tools used


class TestCapabilityManifest:
    """Tests for CapabilityManifest."""

    def test_capability_manifest_empty(self):
        """Test empty capability manifest."""
        manifest = CapabilityManifest()
        context = manifest.to_context()
        assert "Available Tools" in context or context.strip() == "<capabilities>\n</capabilities>"

    def test_capability_manifest_with_tools(self):
        """Test capability manifest with tools."""
        manifest = CapabilityManifest()
        manifest.tools["read"] = {
            "description": "Read file contents",
            "parameters": {"path": {"type": "string", "required": True}},
            "permission": "allow",
        }
        manifest.tools["write"] = {
            "description": "Write file contents",
            "parameters": {"path": {"type": "string"}, "content": {"type": "string"}},
            "permission": "ask",
        }
        context = manifest.to_context()
        assert "read" in context
        assert "write" in context
        assert "Read file contents" in context
        assert "[ask]" in context

    def test_capability_manifest_with_multimodal(self):
        """Test capability manifest with multimodal capabilities."""
        manifest = CapabilityManifest()
        manifest.tools["read"] = {"description": "Read files", "parameters": {}, "permission": "allow"}
        manifest.multimodal = ["vision", "audio"]
        context = manifest.to_context()
        assert "vision" in context
        assert "audio" in context


class TestSelfAwarenessEngine:
    """Tests for SelfAwarenessEngine."""

    def test_engine_creation(self):
        """Test basic engine creation."""
        engine = SelfAwarenessEngine(name="TestBot", version="1.0.0")
        assert engine.name == "TestBot"
        assert engine.version == "1.0.0"

    def test_set_model(self):
        """Test setting model identity."""
        engine = SelfAwarenessEngine()
        engine.set_model(
            provider="ollama",
            model_id="llama3.2",
            model_name="Llama 3.2",
            context_window=128000,
            capabilities=["text"],
        )
        assert engine._model_identity is not None
        assert engine._model_identity.provider == "ollama"

    def test_refresh_environment(self):
        """Test environment refresh."""
        engine = SelfAwarenessEngine()
        env = engine.refresh_environment()
        assert env.working_directory == os.getcwd()
        assert env.platform_name in ["Linux", "Windows", "Darwin"]
        assert engine._environment is not None

    def test_update_session_state(self):
        """Test session state update."""
        engine = SelfAwarenessEngine()
        engine.update_session_state(
            session_id="test-123",
            status="thinking",
            turn_count=1,
            messages_count=2,
            tools_used=["read"],
            topics=["test"],
            active_tools=["read"],
        )
        assert engine._session_state is not None
        assert engine._session_state.session_id == "test-123"
        assert engine._session_state.status == "thinking"

    def test_register_tool(self):
        """Test tool registration."""
        engine = SelfAwarenessEngine()
        engine.register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"arg1": {"type": "string"}},
            permission="allow",
        )
        assert "test_tool" in engine._capabilities.tools
        assert engine._capabilities.tools["test_tool"]["description"] == "A test tool"

    def test_register_multimodal(self):
        """Test multimodal registration."""
        engine = SelfAwarenessEngine()
        engine.register_multimodal(["vision", "audio"])
        assert "vision" in engine._capabilities.multimodal
        assert "audio" in engine._capabilities.multimodal

    def test_get_identity_context(self):
        """Test identity context generation."""
        engine = SelfAwarenessEngine(name="Jarvis", version="2.0.0")
        engine.set_model("ollama", "llama3.2", "Llama 3.2")
        context = engine.get_identity_context()
        assert "Jarvis" in context
        assert "self-aware" in context.lower()
        assert "2.0.0" in context

    def test_get_identity_context_custom(self):
        """Test custom identity override."""
        engine = SelfAwarenessEngine()
        engine.set_custom_identity("You are a custom assistant.")
        context = engine.get_identity_context()
        assert context == "You are a custom assistant."

    def test_get_full_context(self):
        """Test full context generation."""
        engine = SelfAwarenessEngine(name="TestBot")
        engine.set_model("test", "test-model", "Test Model")
        engine.refresh_environment()
        engine.update_session_state(
            session_id="test",
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=[],
            topics=[],
            active_tools=[],
        )
        engine.register_tool("test", "Test tool", {}, "allow")

        context = engine.get_full_context()
        assert "## Self-Awareness" in context
        assert "<env>" in context
        assert "<session_state>" in context
        assert "<capabilities>" in context

    def test_get_tool_schemas_description(self):
        """Test tool schemas description."""
        engine = SelfAwarenessEngine()
        engine.register_tool("read", "Read a file", {"path": {"type": "string", "required": True, "description": "File path"}}, "allow")
        engine.register_tool("write", "Write a file", {"path": {"type": "string"}, "content": {"type": "string"}}, "ask")

        desc = engine.get_tool_schemas_description()
        assert "### read" in desc
        assert "### write" in desc
        assert "Read a file" in desc
        assert "Permission level: ask" in desc

    def test_get_status_summary(self):
        """Test status summary."""
        engine = SelfAwarenessEngine(name="TestBot", version="1.0.0")
        engine.set_model("test", "test-model", "Test Model")
        engine.refresh_environment()

        summary = engine.get_status_summary()
        assert summary["name"] == "TestBot"
        assert summary["version"] == "1.0.0"
        assert summary["model"]["provider"] == "test"
        assert summary["environment"]["platform"] is not None

    def test_load_project_instructions_not_found(self):
        """Test loading project instructions when file doesn't exist."""
        engine = SelfAwarenessEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = engine.load_project_instructions(Path(tmpdir))
            assert result == ""
            assert engine._project_instructions == ""


class TestSelfAwarenessEngineEdgeCases:
    """Edge case tests for SelfAwarenessEngine."""

    def test_get_full_context_no_model(self):
        """Test full context without model set."""
        engine = SelfAwarenessEngine()
        engine.refresh_environment()
        context = engine.get_full_context()
        assert "## Self-Awareness" in context

    def test_get_full_context_no_session(self):
        """Test full context without session state."""
        engine = SelfAwarenessEngine()
        engine.set_model("test", "test", "Test")
        context = engine.get_full_context()
        assert "<env>" in context

    def test_get_full_context_no_tools(self):
        """Test full context without registered tools."""
        engine = SelfAwarenessEngine()
        engine.set_model("test", "test", "Test")
        engine.refresh_environment()
        context = engine.get_full_context()
        assert "Available Tools" not in context or "No tools" in context

    def test_tool_schema_description_empty(self):
        """Test tool schema description with no tools."""
        engine = SelfAwarenessEngine()
        desc = engine.get_tool_schemas_description()
        assert "No tools" in desc

    def test_session_state_with_pending_permission(self):
        """Test session state with pending permission."""
        engine = SelfAwarenessEngine()
        engine.update_session_state(
            session_id="test",
            status="waiting_permission",
            turn_count=1,
            messages_count=2,
            tools_used=[],
            topics=[],
            active_tools=["dangerous_tool"],
            pending_permission={"tool": "dangerous_tool", "args": {"action": "delete"}},
        )
        context = engine._session_state.to_context()
        assert "Pending permission" in context
        assert "dangerous_tool" in context

    def test_multiple_tool_registrations(self):
        """Test registering multiple tools."""
        engine = SelfAwarenessEngine()
        for i in range(10):
            engine.register_tool(f"tool_{i}", f"Tool {i}", {}, "allow")
        assert len(engine._capabilities.tools) == 10

    def test_tool_parameters_with_nested_schema(self):
        """Test tool with nested parameter schema."""
        engine = SelfAwarenessEngine()
        engine.register_tool(
            name="complex_tool",
            description="Complex tool with nested params",
            parameters={
                "config": {
                    "type": "object",
                    "properties": {
                        "nested": {"type": "string"},
                    },
                    "required": True,
                },
            },
            permission="ask",
        )
        desc = engine.get_tool_schemas_description()
        assert "complex_tool" in desc
        assert "config" in desc


class TestSelfAwarenessFactory:
    """Tests for the create_self_awareness factory function."""

    def test_factory_basic(self):
        """Test basic factory function."""
        engine = create_self_awareness()
        assert engine is not None
        assert engine.name == "Jarvis"

    def test_factory_with_custom_params(self):
        """Test factory with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_self_awareness(
                name="CustomBot",
                version="3.0.0",
                model_provider="anthropic",
                model_id="claude-3",
                model_name="Claude 3",
                context_window=200000,
                capabilities=["text", "vision"],
                storage_path=Path(tmpdir),
            )
            assert engine.name == "CustomBot"
            assert engine.version == "3.0.0"
            assert engine._model_identity.provider == "anthropic"
            assert engine._environment is not None


class TestSelfAwarenessThreadSafety:
    """Thread safety tests for SelfAwarenessEngine."""

    def test_concurrent_session_updates(self):
        """Test concurrent session state updates."""
        engine = SelfAwarenessEngine()
        engine.set_model("test", "test", "Test")

        errors = []

        def update_session(thread_id):
            try:
                for i in range(100):
                    engine.update_session_state(
                        session_id=f"thread-{thread_id}",
                        status="thinking",
                        turn_count=i,
                        messages_count=i * 2,
                        tools_used=[],
                        topics=[],
                        active_tools=[],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_session, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_tool_registrations(self):
        """Test concurrent tool registrations."""
        engine = SelfAwarenessEngine()

        errors = []

        def register_tools(thread_id):
            try:
                for i in range(50):
                    engine.register_tool(
                        name=f"tool_{thread_id}_{i}",
                        description=f"Tool {thread_id}:{i}",
                        parameters={},
                        permission="allow",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_tools, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSelfAwarenessEnvironmentDetection:
    """Tests for environment detection features."""

    def test_git_detection_in_git_repo(self):
        """Test git detection when in a git repo."""
        engine = SelfAwarenessEngine()
        env = engine.refresh_environment()

        if env.is_git_repo:
            assert env.git_branch is not None

    def test_environment_values_populated(self):
        """Test that all environment values are populated."""
        engine = SelfAwarenessEngine()
        env = engine.refresh_environment()

        assert env.working_directory != ""
        assert env.platform_name != ""
        assert env.current_date != ""
        assert env.current_time != ""
        assert env.timezone != ""
        assert env.python_version != ""


class TestAgenticAwareness:
    """Tests for agentic awareness features."""

    def test_agentic_task_info_creation(self):
        """Test AgenticTaskInfo creation and serialization."""
        from core.self_awareness import AgenticTaskInfo

        info = AgenticTaskInfo(
            total_tasks=10,
            pending=3,
            running=2,
            completed=4,
            failed=1,
            avg_completion_time=45.5,
            checkpoint_count=8,
        )
        assert info.total_tasks == 10
        assert info.pending == 3
        assert info.running == 2
        d = info.to_dict()
        assert d["total_tasks"] == 10
        assert d["avg_completion_time"] == 45.5

    def test_agentic_goal_info_creation(self):
        """Test AgenticGoalInfo creation and serialization."""
        from core.self_awareness import AgenticGoalInfo

        info = AgenticGoalInfo(
            total_goals=5,
            active=3,
            completed=1,
            blocked=1,
            avg_progress=0.65,
            overdue_count=1,
            proactive_triggers=4,
        )
        assert info.total_goals == 5
        assert info.active == 3
        d = info.to_dict()
        assert d["total_goals"] == 5
        assert d["avg_progress"] == 0.65

    def test_agentic_schedule_info_creation(self):
        """Test AgenticScheduleInfo creation."""
        from core.self_awareness import AgenticScheduleInfo
        from datetime import datetime

        next_exec = datetime.now()
        info = AgenticScheduleInfo(
            total_schedules=3,
            enabled=2,
            disabled=1,
            paused=0,
            next_execution=next_exec,
            execution_history_count=15,
        )
        assert info.total_schedules == 3
        assert info.enabled == 2
        d = info.to_dict()
        assert d["total_schedules"] == 3
        assert d["next_execution"] is not None

    def test_agentic_deployment_info_creation(self):
        """Test AgenticDeploymentInfo creation."""
        from core.self_awareness import AgenticDeploymentInfo
        from datetime import datetime

        dep = AgenticDeploymentInfo(
            deployment_id="dep_123",
            name="test_deployment",
            status="running",
            started_at=datetime.now(),
            goal_count=3,
            task_count=10,
            health_score=0.95,
        )
        assert dep.deployment_id == "dep_123"
        assert dep.health_score == 0.95
        d = dep.to_dict()
        assert d["deployment_id"] == "dep_123"
        assert d["goal_count"] == 3

    def test_agentic_awareness_creation(self):
        """Test AgenticAwareness creation."""
        from core.self_awareness import AgenticAwareness, AgenticTaskInfo, AgenticGoalInfo

        awareness = AgenticAwareness(
            orchestrator_healthy=True,
            active_deployment_id="dep_abc",
        )
        assert awareness.orchestrator_healthy is True
        assert awareness.active_deployment_id == "dep_abc"
        assert len(awareness.deployments) == 0
        
        d = awareness.to_dict()
        assert d["orchestrator_healthy"] is True
        assert d["active_deployment_id"] == "dep_abc"

    def test_engine_get_agentic_awareness(self):
        """Test getting agentic awareness from engine."""
        engine = SelfAwarenessEngine()
        awareness = engine.get_agentic_awareness()
        assert awareness is not None
        assert hasattr(awareness, "deployments")
        assert hasattr(awareness, "tasks")
        assert hasattr(awareness, "goals")
        assert hasattr(awareness, "schedules")

    def test_engine_get_agentic_context(self):
        """Test agentic context generation."""
        engine = SelfAwarenessEngine()
        context = engine.get_agentic_context()
        assert "Agentic Capabilities" in context
        assert "Task Persistence" in context
        assert "Autonomous Goals" in context
        assert "Workflow Scheduling" in context
        assert "Self-Correction" in context

    def test_engine_set_agentic_orchestrator(self):
        """Test setting agentic orchestrator."""
        from core.agentic_orchestrator import get_agentic_orchestrator, DeploymentType

        engine = SelfAwarenessEngine()
        orchestrator = get_agentic_orchestrator()
        
        engine.set_agentic_orchestrator(orchestrator)
        
        awareness = engine.get_agentic_awareness(refresh=True)
        assert awareness is not None

    def test_engine_status_includes_agentic(self):
        """Test that get_status includes agentic info."""
        engine = SelfAwarenessEngine()
        status = engine.get_status()
        assert "agentic" in status
        assert status["agentic"] is not None
        assert "deployments" in status["agentic"]
        assert "tasks" in status["agentic"]
        assert "goals" in status["agentic"]

    def test_record_agentic_learning(self):
        """Test recording agentic learning events."""
        engine = SelfAwarenessEngine()
        
        engine.record_agentic_learning(
            event_type="task_completion",
            description="Completed a long-running task",
            outcome="success",
            insights=["checkpointing improved recovery"],
        )
        
        history = engine.capability_registry.get_learning_history(5)
        assert len(history) > 0
        assert any("agentic" in r.domain for r in history)

    def test_session_state_with_agentic_fields(self):
        """Test session state with agentic fields."""
        engine = SelfAwarenessEngine()
        engine.update_session_state(
            session_id="test_session",
            status="active",
            active_deployment_id="dep_123",
            running_tasks=["task_1", "task_2"],
            active_goals=["goal_1"],
        )
        
        status = engine.get_status()
        assert status["session"]["active_deployment_id"] == "dep_123"
        assert status["session"]["running_tasks"] == 2
        assert status["session"]["active_goals"] == 1

    def test_agentic_capabilities_registered(self):
        """Test that agentic capabilities are registered."""
        engine = SelfAwarenessEngine()
        caps = engine.capability_registry.get_capabilities()
        
        cap_names = [c.name for c in caps]
        assert "task_persistence" in cap_names
        assert "workflow_scheduling" in cap_names
        assert "self_correction" in cap_names
        assert "autonomous_goals" in cap_names
        assert "agentic_deployment" in cap_names

    def test_full_context_includes_agentic_at_standard_level(self):
        """Test that full context includes agentic at STANDARD level."""
        from core.self_awareness import AwarenessLevel

        engine = SelfAwarenessEngine(awareness_level=AwarenessLevel.STANDARD)
        engine.update_session_state(
            session_id="test_session",
            active_deployment_id="dep_123",
        )

        awareness = engine.get_agentic_awareness()
        awareness.deployments.append(
            type('AgenticDeploymentInfo', (), {
                'deployment_id': 'dep_123',
                'name': 'test',
                'status': 'running',
                'started_at': None,
                'goal_count': 0,
                'task_count': 1,
                'schedule_count': 0,
                'health_score': 1.0,
                'last_activity': None,
            })()
        )

        context = engine.get_full_context()
        assert "Agentic Capabilities" in context


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
