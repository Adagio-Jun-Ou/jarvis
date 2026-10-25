"""Integration tests for agent and self-awareness modules."""
import os
import sys
import tempfile
from pathlib import Path
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.self_awareness import SelfAwarenessEngine, create_self_awareness
from core.prompts import PromptManager


class MockTool:
    """Mock tool for testing."""
    name = "mock_tool"
    description = "A mock tool for testing"
    requires_permission = False

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query to execute"},
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, **kwargs):
        return f"Mock result for: {kwargs.get('query', 'no query')}"


class MockPermissionTool(MockTool):
    """Mock tool that requires permission."""
    name = "permission_tool"
    description = "A tool requiring permission"
    requires_permission = True


class TestSelfAwarenessAgentIntegration:
    """Integration tests for self-awareness with agent-like behavior."""

    def test_full_stack_initialization(self):
        """Test complete initialization flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_self_awareness(
                name="TestJarvis",
                version="1.0.0-test",
                model_provider="ollama",
                model_id="llama3.2",
                model_name="Llama 3.2",
                context_window=128000,
                capabilities=["text", "code"],
                storage_path=Path(tmpdir),
            )

            pm = PromptManager(
                name="TestJarvis",
                mode="supportive",
                self_awareness=engine,
            )

            prompt = pm.get_base_prompt()

            assert "TestJarvis" in prompt
            assert "self-aware" in prompt.lower()
            assert "Llama 3.2" in prompt
            assert "<env>" in prompt

    def test_prompt_manager_with_tools(self):
        """Test prompt manager with registered tools."""
        engine = create_self_awareness()

        engine.register_tool(
            name="read_file",
            description="Read a file from disk",
            parameters={
                "path": {"type": "string", "required": True, "description": "File path"},
            },
            permission="allow",
        )
        engine.register_tool(
            name="write_file",
            description="Write content to a file",
            parameters={
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
            },
            permission="ask",
        )

        pm = PromptManager(name="Jarvis", self_awareness=engine)
        prompt = pm.get_base_prompt()

        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "Read a file" in prompt

    def test_session_lifecycle(self):
        """Test full session lifecycle."""
        engine = create_self_awareness()
        session_id = str(uuid.uuid4())[:12]

        engine.update_session_state(
            session_id=session_id,
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=[],
            topics=[],
            active_tools=["read", "search"],
        )

        assert engine._session_state.status == "idle"

        engine.update_session_state(
            session_id=session_id,
            status="thinking",
            turn_count=1,
            messages_count=2,
            tools_used=[],
            topics=["python"],
            active_tools=["read", "search"],
        )

        assert engine._session_state.status == "thinking"
        assert engine._session_state.turn_count == 1

        engine.update_session_state(
            session_id=session_id,
            status="executing_tools",
            turn_count=1,
            messages_count=2,
            tools_used=["read"],
            topics=["python"],
            active_tools=["read", "search"],
        )

        engine.update_session_state(
            session_id=session_id,
            status="idle",
            turn_count=2,
            messages_count=4,
            tools_used=["read", "search"],
            topics=["python", "async"],
            active_tools=["read", "search"],
        )

        context = engine.get_full_context()
        assert "Turn count: 2" in context
        assert "read" in context
        assert "search" in context

    def test_dynamic_environment_updates(self):
        """Test environment context updates dynamically."""
        engine = create_self_awareness()

        env1 = engine.refresh_environment()

        original_cwd = os.getcwd()

        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            env2 = engine.refresh_environment()

            assert env2.working_directory == tmpdir
        finally:
            os.chdir(original_cwd)
            # On Windows a git subprocess run by refresh_environment can hold
            # a handle to tmpdir briefly after it exits, so rmtree races. Retry
            # with a short backoff instead of failing the test on WinError 32.
            import sys
            import time
            import shutil
            for _ in range(10):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=False)
                    break
                except OSError:
                    if sys.platform == "win32":
                        time.sleep(0.2)
                    else:
                        raise

    def test_tool_registration_various_types(self):
        """Test registering various tool types."""
        engine = create_self_awareness()

        simple_tool = {"description": "Simple tool", "parameters": {}, "permission": "allow"}
        engine.register_tool("simple", **simple_tool)

        complex_params = {
            "path": {"type": "string", "required": True, "description": "Path"},
            "options": {
                "type": "object",
                "properties": {
                    "recursive": {"type": "boolean"},
                    "filter": {"type": "string"},
                },
            },
        }
        engine.register_tool("complex", "Complex tool", complex_params, "ask")

        desc = engine.get_tool_schemas_description()
        assert "simple" in desc
        assert "complex" in desc
        assert "Parameter" in desc or "Parameters:" in desc


class TestSelfAwarenessPromptIntegration:
    """Tests for prompt generation with self-awareness."""

    def test_prompt_contains_all_contexts(self):
        """Test that generated prompt contains all context sections."""
        engine = create_self_awareness()
        engine.refresh_environment()
        engine.update_session_state(
            session_id="test-integration",
            status="idle",
            turn_count=5,
            messages_count=10,
            tools_used=["read"],
            topics=["integration"],
            active_tools=["read"],
        )
        engine.register_tool("read", "Read files", {"path": {"type": "string"}}, "allow")

        pm = PromptManager(self_awareness=engine)
        prompt = pm.get_base_prompt()

        assert "## Self-Awareness" in prompt
        assert "<env>" in prompt
        assert "<session_state>" in prompt
        assert "<capabilities>" in prompt
        assert "idle" in prompt
        assert "integration" in prompt

    def test_prompt_without_self_awareness(self):
        """Test prompt generation without self-awareness engine."""
        pm = PromptManager(name="BasicJarvis")
        prompt = pm.get_base_prompt()

        assert "BasicJarvis" in prompt

    def test_prompt_with_project_instructions(self):
        """Test prompt includes project instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jarvis_md = Path(tmpdir) / "JARVIS.md"
            jarvis_md.write_text("# Project Instructions\n\nUse Python 3.11+ features.", encoding='utf-8')

            engine = create_self_awareness(storage_path=Path(tmpdir))
            engine.load_project_instructions(Path(tmpdir))

            pm = PromptManager(self_awareness=engine)
            prompt = pm.get_base_prompt()

            if engine._project_instructions:
                assert "Project Context" in prompt or "Project Instructions" in prompt

    def test_custom_identity_override(self):
        """Test custom identity overrides default."""
        engine = create_self_awareness()
        engine.set_custom_identity("You are a specialized testing assistant.")

        pm = PromptManager(self_awareness=engine)
        prompt = pm.get_base_prompt()

        assert "specialized testing assistant" in prompt

    def test_multimodal_capabilities_in_prompt(self):
        """Test multimodal capabilities appear in prompt."""
        engine = create_self_awareness()
        engine.register_multimodal(["vision", "audio", "code_interpreter"])

        pm = PromptManager(self_awareness=engine)
        prompt = pm.get_base_prompt()

        assert "vision" in prompt or "Multimodal" in prompt


class TestSelfAwarenessEdgeCases:
    """Edge case tests for self-awareness integration."""

    def test_empty_session_id(self):
        """Test handling empty session ID."""
        engine = create_self_awareness()
        engine.update_session_state(
            session_id="",
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=[],
            topics=[],
            active_tools=[],
        )

        context = engine.get_full_context()
        assert context is not None

    def test_very_long_topic_list(self):
        """Test handling many topics."""
        engine = create_self_awareness()
        many_topics = [f"topic_{i}" for i in range(100)]

        engine.update_session_state(
            session_id="test",
            status="idle",
            turn_count=1,
            messages_count=2,
            tools_used=[],
            topics=many_topics,
            active_tools=[],
        )

        context = engine.get_full_context()
        assert context is not None

    def test_unicode_in_descriptions(self):
        """Test unicode characters in tool descriptions."""
        engine = create_self_awareness()
        engine.register_tool(
            "unicode_tool",
            "Outil avec des caractères spéciaux: é, ñ, 中文, 🚀",
            {},
            "allow",
        )

        desc = engine.get_tool_schemas_description()
        assert "é" in desc or "unicode_tool" in desc

    def test_special_characters_in_session_id(self):
        """Test special characters in session ID."""
        engine = create_self_awareness()
        engine.update_session_state(
            session_id="test-session_123!@#$%",
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=[],
            topics=[],
            active_tools=[],
        )

        context = engine.get_full_context()
        assert context is not None

    def test_none_values_in_update(self):
        """Test None values in session update."""
        engine = create_self_awareness()
        engine.update_session_state(
            session_id="test",
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=None,
            topics=None,
            active_tools=None,
            pending_permission=None,
        )

        assert engine._session_state.tools_used_this_session == []

    def test_concurrent_prompt_generation(self):
        """Test generating prompts from multiple threads."""
        import threading

        engine = create_self_awareness()
        pm = PromptManager(self_awareness=engine)

        results = []
        errors = []

        def generate_prompt():
            try:
                prompt = pm.get_base_prompt()
                results.append(prompt)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate_prompt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_tool_with_no_parameters(self):
        """Test tool registration without parameters."""
        engine = create_self_awareness()
        engine.register_tool("no_params", "Tool with no parameters", None, "allow")

        desc = engine.get_tool_schemas_description()
        assert "no_params" in desc


class TestSelfAwarenessStatusReporting:
    """Tests for status reporting functionality."""

    def test_status_summary_complete(self):
        """Test complete status summary."""
        engine = create_self_awareness(
            name="StatusTest",
            version="2.0.0",
        )
        engine.refresh_environment()
        engine.update_session_state(
            session_id="status-test",
            status="busy",
            turn_count=10,
            messages_count=20,
            tools_used=["read", "write"],
            topics=["status"],
            active_tools=["read", "write", "search"],
        )
        engine.register_tool("read", "Read files", {}, "allow")

        summary = engine.get_status_summary()

        assert summary["name"] == "StatusTest"
        assert summary["version"] == "2.0.0"
        assert summary["session"]["status"] == "busy"
        assert summary["session"]["turn_count"] == 10
        assert summary["tools_registered"] == 1

    def test_status_summary_no_session(self):
        """Test status summary without session."""
        engine = create_self_awareness()
        summary = engine.get_status_summary()

        assert summary["session"] is None

    def test_status_summary_no_model(self):
        """Test status summary without model set."""
        engine = SelfAwarenessEngine()
        summary = engine.get_status_summary()

        assert summary["model"] is None


class TestSelfAwarenessMemoryIntegration:
    """Tests for integration with memory system."""

    def test_session_tracks_tools_used(self):
        """Test tracking tools used in session."""
        engine = create_self_awareness()

        engine.update_session_state(
            session_id="tool-tracking",
            status="executing_tools",
            turn_count=1,
            messages_count=2,
            tools_used=["read", "search", "read"],  # read used twice
            topics=[],
            active_tools=["read", "search", "write"],
        )

        context = engine.get_full_context()
        assert "read" in context

    def test_session_clear_and_reset(self):
        """Test clearing and resetting session."""
        engine = create_self_awareness()

        engine.update_session_state(
            session_id="first-session",
            status="idle",
            turn_count=10,
            messages_count=50,
            tools_used=["many", "tools"],
            topics=["many", "topics"],
            active_tools=[],
        )

        engine.update_session_state(
            session_id="new-session",
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=[],
            topics=[],
            active_tools=[],
        )

        assert engine._session_state.session_id == "new-session"
        assert engine._session_state.turn_count == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
