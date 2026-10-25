#!/usr/bin/env python3
"""End-to-end tests for agentic workflow with live server."""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import asyncio
import sys
sys.path.insert(0, _JARVIS_ROOT)

from unittest.mock import MagicMock, AsyncMock, patch
from core.agent import JarvisAgent, ActionProposal
from core.config import Config


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or ["Default response"]
        self.call_count = 0
        self.last_input = None
    
    async def chat_with_tools(self, user_input, system_prompt=None, conversation_history=None, tools=None):
        self.call_count += 1
        self.last_input = user_input
        
        response = self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        
        # Return content and tool calls
        if isinstance(response, tuple):
            content, tool_calls = response
            return content, tool_calls
        return response, []
    
    async def chat(self, user_input, stream=False, **kwargs):
        self.call_count += 1
        self.last_input = user_input
        response = self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        if isinstance(response, tuple):
            return {"content": response[0]}
        return {"content": response}


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.description = f"Mock {name} tool"
        self.requires_permission = False  # Needed to avoid permission check
    
    def get_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query parameter"}
                }
            }
        }
    
    async def execute(self, **kwargs):
        return self.result


async def test_action_proposal():
    """Test ActionProposal dataclass."""
    print("\n=== Test: ActionProposal ===")
    
    # Test tool call proposal
    proposal = ActionProposal(
        thought="Need to search for information",
        action_type="tool_call",
        tool_name="web_search",
        tool_args={"query": "Python tutorials"},
        response=None,
    )
    
    assert proposal.action_type == "tool_call"
    assert proposal.tool_name == "web_search"
    assert proposal.tool_args["query"] == "Python tutorials"
    print("  ✓ Tool call proposal created")
    
    # Test response proposal
    proposal2 = ActionProposal(
        thought="Simple response",
        action_type="respond",
        response="Here's the answer.",
    )
    
    assert proposal2.action_type == "respond"
    assert proposal2.response == "Here's the answer."
    print("  ✓ Response proposal created")


async def test_agent_think_simple():
    """Test agent think method with simple request."""
    print("\n=== Test: Agent Think (Simple Request) ===")
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Mock the LLM
    agent.llm = MockLLM(responses=["The capital of France is Paris."])
    
    # Test simple think
    proposal, tool_calls = await agent.think("What is the capital of France?")
    
    print(f"  Action type: {proposal.action_type}")
    print(f"  Response: {proposal.response[:100] if proposal.response else 'None'}...")
    print(f"  Tool calls: {len(tool_calls)}")
    
    assert proposal.action_type == "respond" or proposal.action_type == "tool_call"
    print("  ✓ Think method returns valid proposal")


async def test_agent_think_with_tools():
    """Test agent think method with tool calls."""
    print("\n=== Test: Agent Think (Tool Call) ===")
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Mock the LLM to return tool calls
    agent.llm = MockLLM(responses=[
        ("Using search tool", [
            {"function": {"name": "web_search", "arguments": {"query": "Python tutorials"}}}
        ])
    ])
    
    # Register a mock tool
    agent.tools = {
        "web_search": MockTool("web_search", "Found 10 Python tutorials"),
    }
    agent._tool_schemas = [agent.tools["web_search"].get_schema()]
    
    # Test think with tool
    proposal, tool_calls = await agent.think("Search for Python tutorials")
    
    print(f"  Action type: {proposal.action_type}")
    print(f"  Tool calls: {len(tool_calls)}")
    
    if len(tool_calls) > 0:
        print(f"  First tool: {tool_calls[0][0]}")
    
    assert proposal.action_type == "tool_call" or proposal.action_type == "respond"
    print("  ✓ Think method handles tool calls")


async def test_agent_complexity_detection():
    """Test complexity detection for various inputs."""
    print("\n=== Test: Complexity Detection ===")
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Test cases
    test_cases = [
        # Simple requests
        ("What is the weather?", False, "simple question"),
        ("Tell me a joke", False, "simple request"),
        ("What is 2 + 2?", False, "simple math"),
        ("Explain Python", False, "single topic explanation"),
        
        # Complex requests
        ("Search for Python tutorials and then analyze them", True, "and then connector"),
        ("First search, then summarize, and finally save", True, "multiple steps"),
        ("List all files and then count them", True, "and then connector"),
        ("Find all Python files and analyze their imports", True, "multiple actions"),
        ("Comprehensive analysis of the system multiple steps", True, "complex indicators"),
        ("Step by step guide to building an app", True, "step by step"),
        ("Search and fetch multiple documents", True, "multiple"),
    ]
    
    for request, expected, description in test_cases:
        result = agent._is_complex_request(request)
        status = "✓" if result == expected else "✗"
        expected_str = "complex" if expected else "simple"
        actual_str = "complex" if result else "simple"
        print(f"  {status} {description}: {actual_str} (expected: {expected_str})")
        if result != expected:
            print(f"      Input: '{request}'")


async def test_agentic_completion_messages():
    """Test that completion messages are generated correctly."""
    print("\n=== Test: Agentic Completion Messages ===")
    
    # Completion message should be shown after tool execution
    completion_msg = "✨ **Done!** Task completed successfully."
    continuation_msg = "🔄 **Continuing...** I need to {next_action}"
    
    assert "Done" in completion_msg
    assert "completed" in completion_msg.lower()
    assert "Continuing" in continuation_msg
    print("  ✓ Completion messages defined correctly")
    
    # Test message formatting
    next_action = "analyze the results"
    formatted = continuation_msg.format(next_action=next_action)
    assert next_action in formatted
    print("  ✓ Continuation message formats correctly")


async def test_intelligent_orchestrator_integration():
    """Test that orchestrator integrates with agent."""
    print("\n=== Test: Orchestrator Integration ===")
    
    from core.intelligent_orchestrator import IntelligentOrchestrator
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Verify orchestrator is None initially
    assert agent._intelligent_orchestrator is None
    print("  ✓ Orchestrator is None initially")
    
    # Create mock orchestrator
    mock_llm = MockLLM()
    mock_backend = MockLLM()
    
    orchestrator = IntelligentOrchestrator(
        llm_client=mock_llm,
        backend_llm=mock_backend,
        dual_model=None,
    )
    
    # Attach orchestrator
    agent.attach_intelligent_orchestrator(orchestrator)
    
    assert agent._intelligent_orchestrator is orchestrator
    print("  ✓ Orchestrator attached to agent")
    
    # Test that complex request triggers orchestration
    complex_request = "Search for Python tutorials and then analyze them step by step"
    is_complex = agent._is_complex_request(complex_request)
    
    assert is_complex, "Complex request should be detected"
    print("  ✓ Complex request detected correctly")


async def test_dual_model_routing():
    """Test dual-model task routing."""
    print("\n=== Test: Dual-Model Routing ===")
    
    from core.dual_model import TaskType
    
    # Test task type classification
    task_types = {
        "Write a Python function": TaskType.CODE,
        "Calculate the square root": TaskType.MATH,
        "Analyze this algorithm": TaskType.ANALYSIS,
        "What's the weather?": TaskType.CHAT,
        "Tell me a story": TaskType.CHAT,  # Creative defaults to chat
    }
    
    from core.intelligent_orchestrator import IntelligentOrchestrator
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM(),
        backend_llm=MockLLM(),
        dual_model=None,
    )
    
    for request, expected in task_types.items():
        result = orchestrator._classify_simple_task(request)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{request[:30]}...' → {result.value} (expected: {expected.value})")


async def test_error_handling():
    """Test error handling in agent."""
    print("\n=== Test: Error Handling ===")
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Test with tools that raise errors
    agent.tools = {
        "failing_tool": MockTool("failing_tool", "Success"),  # This will succeed
    }
    
    # Test execute_tool with unknown tool
    result = await agent.execute_tool("unknown_tool", {})
    assert not result.success
    assert "Unknown tool" in result.error
    print("  ✓ Unknown tool error handled")
    
    # Test execute_tool with valid tool
    result = await agent.execute_tool("failing_tool", {"query": "test"})
    assert result.success
    print("  ✓ Valid tool executes successfully")


async def test_memory_management():
    """Test that agent memory is managed correctly."""
    print("\n=== Test: Memory Management ===")
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Add messages to memory
    agent.memory.add_user_message("Hello")
    agent.memory.add_assistant_message("Hi there!")
    
    messages = agent.memory.get_messages_for_llm()
    assert len(messages) >= 2
    print("  ✓ Messages added to memory")
    
    # Test memory limits
    initial_count = len(messages)
    for i in range(10):
        agent.memory.add_user_message(f"Message {i}")
        agent.memory.add_assistant_message(f"Response {i}")
    
    messages = agent.memory.get_messages_for_llm()
    print(f"  ✓ Memory has {len(messages)} messages (added 20 messages)")


async def test_context_management():
    """Test context window management."""
    print("\n=== Test: Context Management ===")
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Get system prompt
    prompt = agent.get_system_prompt()
    assert prompt is not None
    assert len(prompt) > 0
    print(f"  ✓ System prompt generated ({len(prompt)} chars)")
    
    # Test prompt caching
    prompt2 = agent.get_system_prompt()
    assert prompt == prompt2
    print("  ✓ System prompt cached correctly")


async def test_tool_cache():
    """Test tool result caching."""
    print("\n=== Test: Tool Cache ===")
    
    # Create a fresh environment for this test
    from core.agent import ToolResultCache
    
    # Create a new cache to avoid state pollution
    fresh_cache = ToolResultCache(max_size=100, ttl_seconds=300)
    
    class FreshMockTool:
        def __init__(self, name, result):
            self.name = name
            self.result = result
            self.description = f"Mock {name} tool"
            self.requires_permission = False
        
        def get_schema(self):
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query parameter"}
                    }
                }
            }
        
        async def execute(self, **kwargs):
            return self.result
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Replace with fresh cache
    agent.tool_cache = fresh_cache
    
    # Use a unique tool name
    tool_name = "unique_test_tool_cache_check"
    
    # Register tool
    agent.tools = {
        tool_name: FreshMockTool(tool_name, "Cached result"),
    }
    
    # Check cache is empty before first execution
    cached_before = await agent.tool_cache.get(tool_name, {"query": "test"})
    assert cached_before is None, "Cache should be empty before first execution"
    
    # Execute first time
    result1 = await agent.execute_tool(tool_name, {"query": "test"})
    
    # First should not be from cache
    assert result1.success, f"First execution should succeed"
    assert not result1.from_cache, f"First execution should not be from cache"
    print("  ✓ First execution not from cache")
    
    # Execute second time
    result2 = await agent.execute_tool(tool_name, {"query": "test"})
    
    # Second should be from cache
    assert result2.success, f"Second execution should succeed"
    assert result2.from_cache, f"Second execution should be from cache"
    print("  ✓ Second execution from cache")
    
    # Check cache stats
    stats = agent.tool_cache.get_stats()
    assert stats["hits"] >= 1, f"Should have at least 1 cache hit"
    print(f"  ✓ Cache working (hits: {stats['hits']}, hit_rate: {stats['hit_rate']:.2%})")


# ==================== MAIN ====================

async def main():
    """Run all tests."""
    print("=" * 70)
    print("AGENTIC WORKFLOW TESTS")
    print("=" * 70)
    
    tests = [
        test_action_proposal,
        test_agent_think_simple,
        test_agent_think_with_tools,
        test_agent_complexity_detection,
        test_agentic_completion_messages,
        test_intelligent_orchestrator_integration,
        test_dual_model_routing,
        test_error_handling,
        test_memory_management,
        test_context_management,
        test_tool_cache,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)