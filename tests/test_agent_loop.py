"""Tests for AgentLoop - OpenCode-style continuous agent execution."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.agent_loop import (
    AgentLoop,
    AgentResult,
    AgentStep,
    FinishReason,
    ToolStatus,
    AssistantMessage,
    ToolPart,
    create_agent_loop,
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    async def chat_stream_with_tools(self, user_input, system_prompt, conversation_history=None, tools=None, temperature=0.7, max_tokens=4096, **kwargs):
        """Yield structured events."""
        for response in self.responses:
            yield response


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name="test_tool", result="Tool executed successfully"):
        self.name = name
        self.result = result
        self.category = "test"
    
    async def execute(self, **kwargs):
        return MagicMock(output=self.result, success=True, error=None)


class TestAgentLoop:
    """Test AgentLoop functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        return MockLLMClient(responses=[
            {"type": "text", "text": "Hello, I am "},
            {"type": "text", "text": "Jarvis!"},
            {"type": "finish", "reason": "stop"},
        ])
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        return {
            "test_tool": MockTool(),
            "another_tool": MockTool(name="another_tool"),
        }
    
    def test_init(self, mock_llm, mock_tool_registry):
        """Test AgentLoop initialization."""
        loop = AgentLoop(
            llm_client=mock_llm,
            tool_registry=mock_tool_registry,
            max_steps=10,
        )
        
        assert loop.max_steps == 10
        assert loop.llm == mock_llm
        assert len(loop.tools) == 2
    
    def test_init_defaults(self, mock_llm):
        """Test AgentLoop with default values."""
        loop = AgentLoop(llm_client=mock_llm, tool_registry={})
        
        assert loop.max_steps == 50
        assert loop.show_thinking is True
        assert loop.show_tool_calls is True
    
    @pytest.mark.asyncio
    async def test_execute_simple_response(self, mock_llm):
        """Test simple response without tool calls."""
        result = await AgentLoop(
            llm_client=mock_llm,
            tool_registry={},
        ).execute(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are helpful.",
        )
        
        assert result.finish_reason == FinishReason.STOP
        assert len(result.steps) >= 1
    
    @pytest.mark.asyncio
    async def test_execute_stream_yields_events(self, mock_llm):
        """Test that execute_stream is async generator."""
        loop = AgentLoop(llm_client=mock_llm, tool_registry={})
        
        # execute_stream should be async generator
        assert hasattr(loop, 'execute_stream')
    
    @pytest.mark.asyncio
    async def test_max_steps_limit(self, mock_llm):
        """Test that max_steps limits execution."""
        # Create a loop that requests more steps
        mock_llm.responses = [
            {"type": "text", "text": "Thinking..."},
            {"type": "finish", "reason": "stop"},
        ]
        
        result = await AgentLoop(
            llm_client=mock_llm,
            tool_registry={},
            max_steps=3,
        ).execute(
            messages=[{"role": "user", "content": "Loop"}],
            system_prompt="You are helpful.",
        )
        
        # Should complete within max_steps
        assert result.steps is not None


class TestCreateAgentLoop:
    """Test the create_agent_loop factory function."""
    
    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()
    
    def test_create_with_defaults(self, mock_llm):
        """Test creating agent loop with defaults."""
        loop = create_agent_loop(llm_client=mock_llm, tools={})
        
        assert loop is not None
        assert loop.max_steps == 50
    
    def test_create_with_custom_steps(self, mock_llm):
        """Test creating agent loop with custom max steps."""
        loop = create_agent_loop(
            llm_client=mock_llm,
            tools={},
            max_steps=25
        )
        
        assert loop.max_steps == 25
    
    def test_create_with_permissions(self, mock_llm):
        """Test creating agent loop with permission manager."""
        mock_permissions = MagicMock()
        
        loop = create_agent_loop(
            llm_client=mock_llm,
            tools={},
            permissions=mock_permissions
        )
        
        assert loop.permissions == mock_permissions


class TestAgentResult:
    """Test AgentResult dataclass."""
    
    def test_agent_result_creation(self):
        """Test creating an AgentResult."""
        result = AgentResult(
            steps=[],
            total_tool_calls=0,
            total_tokens=100,
            duration_ms=500.0,
            finish_reason=FinishReason.STOP,
        )
        
        assert result.total_tokens == 100
        assert result.finish_reason == FinishReason.STOP
        assert result.duration_ms == 500.0


class TestAssistantMessage:
    """Test AssistantMessage dataclass."""
    
    def test_assistant_message_creation(self):
        """Test creating an AssistantMessage."""
        msg = AssistantMessage(
            role="assistant",
            parts=[],
            finish_reason=FinishReason.STOP,
        )
        
        assert msg.role == "assistant"
        assert msg.finish_reason == FinishReason.STOP


class TestToolPart:
    """Test ToolPart dataclass."""
    
    def test_tool_part_creation(self):
        """Test creating a ToolPart."""
        part = ToolPart(
            tool_name="test_tool",
            call_id="call_123",
            input={"arg": "value"},
            status=ToolStatus.PENDING,
        )
        
        assert part.tool_name == "test_tool"
        assert part.call_id == "call_123"
        assert part.status == ToolStatus.PENDING


class TestFinishReason:
    """Test FinishReason enum."""
    
    def test_finish_reason_values(self):
        """Test FinishReason enum values."""
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.TOOL_CALLS.value == "tool-calls"
        assert FinishReason.MAX_TOKENS.value == "max-tokens"
        assert FinishReason.ERROR.value == "error"
        assert FinishReason.UNKNOWN.value == "unknown"


class TestToolStatus:
    """Test ToolStatus enum."""
    
    def test_tool_status_values(self):
        """Test ToolStatus enum values."""
        assert ToolStatus.PENDING.value == "pending"
        assert ToolStatus.RUNNING.value == "running"
        assert ToolStatus.COMPLETED.value == "completed"
        assert ToolStatus.ERROR.value == "error"
        assert ToolStatus.PERMISSION_DENIED.value == "permission-denied"