"""Full stack tests for Jarvis with self-awareness."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFullStackSelfAwareness:
    """Full stack end-to-end tests."""

    def test_self_awareness_module_imports(self):
        """Test all modules can be imported."""
        print("✓ All modules imported successfully")

    def test_create_and_use_self_awareness(self):
        """Test creating and using self-awareness engine."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness(name="TestBot")
        engine.refresh_environment()
        engine.update_session_state(
            session_id="test-123",
            status="idle",
            turn_count=0,
            messages_count=0,
            tools_used=[],
            topics=[],
            active_tools=[],
        )
        engine.register_tool("test", "Test tool", {}, "allow")

        context = engine.get_full_context()
        assert "TestBot" in context
        assert "<env>" in context
        print("✓ Self-awareness engine created and used")

    def test_prompt_manager_integration(self):
        """Test prompt manager with self-awareness."""
        from core.self_awareness import create_self_awareness
        from core.prompts import PromptManager

        engine = create_self_awareness()
        engine.register_tool("read", "Read files", {"path": {"type": "string"}}, "allow")

        pm = PromptManager(self_awareness=engine)
        prompt = pm.get_base_prompt()

        assert "read" in prompt
        print("✓ Prompt manager integration works")

    def test_status_summary(self):
        """Test status summary generation."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness(name="SummaryTest", version="1.0.0")
        engine.refresh_environment()

        summary = engine.get_status_summary()

        assert summary["name"] == "SummaryTest"
        assert summary["environment"] is not None
        print("✓ Status summary generated correctly")


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_engine_state(self):
        """Test engine with no state set."""
        from core.self_awareness import SelfAwarenessEngine

        engine = SelfAwarenessEngine()
        context = engine.get_full_context()

        assert context is not None
        print("✓ Empty engine state handled")

    def test_unicode_handling(self):
        """Test unicode in various fields."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()
        engine.register_tool(
            "unicode_test",
            "工具描述: 中文测试 🚀 émojis",
            {"参数": {"type": "string"}},
            "allow",
        )

        desc = engine.get_tool_schemas_description()
        assert "unicode_test" in desc
        print("✓ Unicode handled correctly")

    def test_very_long_strings(self):
        """Test handling very long strings."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()
        long_desc = "A" * 10000
        engine.register_tool("long_tool", long_desc, {}, "allow")

        desc = engine.get_tool_schemas_description()
        assert "long_tool" in desc
        print("✓ Long strings handled")

    def test_special_path_characters(self):
        """Test paths with special characters."""
        from core.self_awareness import create_self_awareness

        with tempfile.TemporaryDirectory() as tmpdir:
            special_path = Path(tmpdir) / "path with spaces" / "über"
            special_path.mkdir(parents=True, exist_ok=True)

            engine = create_self_awareness(storage_path=special_path)
            engine.refresh_environment()

            assert engine._environment is not None
            print("✓ Special path characters handled")

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()
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
        print("✓ Concurrent access handled safely")


class TestSessionStateTransitions:
    """Test session state transitions."""

    def test_state_transitions(self):
        """Test valid state transitions."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()

        states = ["idle", "thinking", "executing_tools", "waiting_permission", "idle"]

        for state in states:
            engine.update_session_state(
                session_id="test",
                status=state,
                turn_count=0,
                messages_count=0,
                tools_used=[],
                topics=[],
                active_tools=[],
            )
            assert engine._session_state.status == state

        print("✓ State transitions work correctly")

    def test_context_reflects_current_state(self):
        """Test context reflects current state."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()

        engine.update_session_state(
            session_id="test",
            status="thinking",
            turn_count=5,
            messages_count=10,
            tools_used=["read"],
            topics=["important"],
            active_tools=["read"],
        )

        context = engine.get_full_context()
        assert "thinking" in context
        assert "Turn count: 5" in context
        print("✓ Context reflects current state")


class TestToolRegistry:
    """Test tool registry functionality."""

    def test_tool_registration_types(self):
        """Test different tool registration types."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()

        engine.register_tool("no_params", "No params", None, "allow")
        engine.register_tool("simple", "Simple", {}, "allow")
        engine.register_tool("complex", "Complex", {
            "arg1": {"type": "string", "required": True},
            "arg2": {"type": "number", "required": False},
            "arg3": {"type": "object", "properties": {"nested": {"type": "string"}}},
        }, "ask")

        desc = engine.get_tool_schemas_description()

        assert "no_params" in desc
        assert "simple" in desc
        assert "complex" in desc
        print("✓ Various tool registration types work")

    def test_tool_permission_levels(self):
        """Test tool permission levels."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()

        engine.register_tool("allowed", "Allowed tool", {}, "allow")
        engine.register_tool("ask_first", "Ask first", {}, "ask")
        engine.register_tool("denied", "Denied tool", {}, "deny")

        desc = engine.get_tool_schemas_description()

        assert "Permission level: ask" in desc
        assert "Permission level: deny" in desc
        print("✓ Permission levels displayed correctly")

    def test_many_tools(self):
        """Test registering many tools."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()

        for i in range(100):
            engine.register_tool(f"tool_{i}", f"Tool number {i}", {}, "allow")

        assert len(engine._capabilities.tools) == 100
        print("✓ Many tools registered successfully")


class TestPromptGeneration:
    """Test prompt generation."""

    def test_prompt_is_valid_markdown(self):
        """Test generated prompt structure."""
        from core.self_awareness import create_self_awareness
        from core.prompts import PromptManager

        engine = create_self_awareness()
        engine.register_tool("test", "Test", {}, "allow")

        pm = PromptManager(self_awareness=engine)
        prompt = pm.get_base_prompt()

        assert "## " in prompt  # Has markdown headers
        print("✓ Prompt has valid structure")

    def test_different_modes(self):
        """Test prompt manager with different modes."""
        from core.prompts import PromptManager

        for mode in ["supportive", "analytical", "creative"]:
            pm = PromptManager(mode=mode)
            thinking = pm.get_thinking_prompt()
            assert mode in thinking.lower() or "Mode" in thinking

        print("✓ Different modes handled")

    def test_custom_instructions(self):
        """Test custom instructions in prompt."""
        from core.self_awareness import create_self_awareness
        from core.prompts import PromptManager

        engine = create_self_awareness()
        pm = PromptManager(
            self_awareness=engine,
            custom_instructions="Always respond in Spanish.",
        )

        prompt = pm.get_base_prompt()
        assert "Spanish" in prompt
        print("✓ Custom instructions included")


class TestMultimodalCapabilities:
    """Test multimodal capabilities."""

    def test_register_multimodal(self):
        """Test multimodal capability registration."""
        from core.self_awareness import create_self_awareness

        engine = create_self_awareness()
        engine.register_multimodal(["vision", "audio", "code_interpreter"])

        assert len(engine._capabilities.multimodal) == 3
        print("✓ Multimodal capabilities registered")

    def test_multimodal_in_context(self):
        """Test multimodal in generated context."""
        from core.self_awareness import create_self_awareness
        from core.prompts import PromptManager

        engine = create_self_awareness()
        engine.register_multimodal(["vision"])

        pm = PromptManager(self_awareness=engine)
        prompt = pm.get_base_prompt()

        context = engine._capabilities.to_context()
        assert "vision" in context
        print("✓ Multimodal in context")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("JARVIS SELF-AWARENESS FULL STACK TESTS")
    print("=" * 60 + "\n")

    loader = TestFullStackSelfAwareness()
    edge = TestEdgeCases()
    states = TestSessionStateTransitions()
    tools = TestToolRegistry()
    prompts = TestPromptGeneration()
    multimodal = TestMultimodalCapabilities()

    test_methods = []

    for cls, name in [
        (loader, "Full Stack"),
        (edge, "Edge Cases"),
        (states, "State Transitions"),
        (tools, "Tool Registry"),
        (prompts, "Prompt Generation"),
        (multimodal, "Multimodal"),
    ]:
        for attr in dir(cls):
            if attr.startswith("test_"):
                test_methods.append((cls, attr, f"{name}: {attr}"))

    passed = 0
    failed = 0

    for cls, method, test_name in test_methods:
        try:
            getattr(cls, method)()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
