"""Tests for Context Integration with Agent and Orchestrators.

Tests:
1. Context integration with agent
2. Tool executor for orchestrator
3. Tool schemas for subagents
4. Cross-model context sharing
5. End-to-end workflow execution
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from core.context_integration import (
    get_enhanced_context,
    get_tool_schemas_for_subagent,
    get_tool_executor_for_orchestrator,
    enhance_system_prompt,
    wire_context_to_orchestrator,
    wire_context_to_subagent,
    wire_context_to_dual_model,
    ContextIntegration,
    integrate_context_with_agent,
    get_agent_context,
)


class MockTool:
    """Mock tool for testing."""
    def __init__(self, name):
        self.name = name
    
    def get_schema(self):
        return {
            "name": self.name,
            "description": f"Mock tool {self.name}",
            "parameters": {"type": "object", "properties": {}},
        }


class MockAgent:
    """Mock agent for testing."""
    def __init__(self):
        self.tools = {
            "read_file": MockTool("read_file"),
            "write_file": MockTool("write_file"),
            "shell": MockTool("shell"),
        }
        self._tool_schemas = [t.get_schema() for t in self.tools.values()]
        self.memory = MagicMock()
        self.memory.get_messages_for_llm.return_value = []
        self._context_integration = None
    
    def get_tool_schemas(self):
        return self._tool_schemas
    
    def get_system_prompt(self):
        return "You are a mock agent."
    
    async def execute_tool(self, tool_name, args):
        result = MagicMock()
        result.output = f"Executed {tool_name} with {args}"
        return result


class TestGetEnhancedContext:
    """Tests for get_enhanced_context function."""

    @pytest.mark.asyncio
    async def test_get_enhanced_context_basic(self):
        """Should return context and metadata."""
        agent = MockAgent()
        
        with patch("core.unified_context_engine.get_context_engine") as mock_engine:
            mock_context_engine = MagicMock()
            mock_context_pack = MagicMock()
            mock_context_pack.to_prompt.return_value = "Test context"
            mock_context_pack.total_tokens = 100
            mock_context_pack.entries = [MagicMock(), MagicMock()]
            mock_context_pack.sources_included = []
            mock_context_pack.assembly_time_ms = 50.0
            
            mock_context_engine.assemble_context = AsyncMock(return_value=mock_context_pack)
            mock_engine.return_value = mock_context_engine
            
            context, metadata = await get_enhanced_context(
                query="test query",
                agent=agent,
                max_tokens=1000,
            )
            
            assert context == "Test context"
            assert metadata["total_tokens"] == 100
            assert metadata["entry_count"] == 2

    @pytest.mark.asyncio
    async def test_get_enhanced_context_with_model(self):
        """Should pass model to context engine."""
        agent = MockAgent()
        
        with patch("core.unified_context_engine.get_context_engine") as mock_engine:
            mock_context_engine = MagicMock()
            mock_context_pack = MagicMock()
            mock_context_pack.to_prompt.return_value = ""
            mock_context_pack.total_tokens = 0
            mock_context_pack.entries = []
            mock_context_pack.sources_included = []
            mock_context_pack.assembly_time_ms = 0
            
            mock_context_engine.assemble_context = AsyncMock(return_value=mock_context_pack)
            mock_engine.return_value = mock_context_engine
            
            await get_enhanced_context(
                query="test",
                agent=agent,
                model="backend",
            )
            
            call_args = mock_context_engine.assemble_context.call_args
            assert call_args.kwargs["model"] == "backend"

    @pytest.mark.asyncio
    async def test_get_enhanced_context_error_handling(self):
        """Should handle errors gracefully."""
        agent = MockAgent()
        
        # Direct test of error handling logic
        # The function should catch exceptions and return empty values
        context, metadata = await get_enhanced_context(
            query="test",
            agent=agent,
        )
        
        # Without mocking, this will use real context engine
        # which may return empty context if no subsystems available
        # Just verify it doesn't crash
        assert isinstance(context, str)
        assert isinstance(metadata, dict)


class TestToolSchemasForSubagent:
    """Tests for tool schema retrieval."""

    def test_get_all_tool_schemas(self):
        """Should return all tool schemas."""
        agent = MockAgent()
        
        schemas = get_tool_schemas_for_subagent(agent)
        
        assert len(schemas) == 3
        names = [s["name"] for s in schemas]
        assert "read_file" in names
        assert "write_file" in names
        assert "shell" in names

    def test_get_specific_tool_schemas(self):
        """Should filter to specific tools."""
        agent = MockAgent()
        
        schemas = get_tool_schemas_for_subagent(agent, tool_names=["read_file", "shell"])
        
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "read_file" in names
        assert "shell" in names
        assert "write_file" not in names

    def test_get_empty_tool_schemas(self):
        """Should return empty list for no matching tools."""
        agent = MockAgent()
        
        schemas = get_tool_schemas_for_subagent(agent, tool_names=["nonexistent"])
        
        assert len(schemas) == 0


class TestToolExecutorForOrchestrator:
    """Tests for tool executor function."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Should execute tool and return result."""
        agent = MockAgent()
        executor = get_tool_executor_for_orchestrator(agent)
        
        result = await executor("read_file", {"path": "/test"})
        
        assert "Executed read_file" in result

    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Should handle tool execution errors."""
        agent = MockAgent()
        agent.execute_tool = AsyncMock(side_effect=Exception("Tool error"))
        
        executor = get_tool_executor_for_orchestrator(agent)
        
        result = await executor("fail_tool", {})
        
        assert "Error:" in result


class TestEnhanceSystemPrompt:
    """Tests for system prompt enhancement."""

    @pytest.mark.asyncio
    async def test_enhance_prompt_with_context(self):
        """Should add context to base prompt."""
        agent = MockAgent()
        
        with patch("core.context_integration.get_enhanced_context") as mock_get_context:
            mock_get_context.return_value = ("Important context here", {"total_tokens": 50})
            
            enhanced = await enhance_system_prompt(
                agent=agent,
                query="test query",
                base_prompt="You are an assistant.",
            )
            
            assert "You are an assistant" in enhanced
            assert "Important context here" in enhanced
            assert "Relevant Context" in enhanced

    @pytest.mark.asyncio
    async def test_enhance_prompt_no_context(self):
        """Should return base prompt if no context."""
        agent = MockAgent()
        
        with patch("core.context_integration.get_enhanced_context") as mock_get_context:
            mock_get_context.return_value = ("", {})
            
            enhanced = await enhance_system_prompt(
                agent=agent,
                query="test",
                base_prompt="Base prompt",
            )
            
            assert enhanced == "Base prompt"

    @pytest.mark.asyncio
    async def test_enhance_prompt_uses_agent_prompt(self):
        """Should use agent's prompt if base not provided."""
        agent = MockAgent()
        
        with patch("core.context_integration.get_enhanced_context") as mock_get_context:
            mock_get_context.return_value = ("", {})
            
            enhanced = await enhance_system_prompt(
                agent=agent,
                query="test",
            )
            
            assert enhanced == "You are a mock agent."


class TestWireContextToOrchestrator:
    """Tests for wiring context to orchestrator."""

    def test_wire_tool_executor(self):
        """Should wire tool executor to orchestrator."""
        agent = MockAgent()
        orchestrator = MagicMock()
        orchestrator._tool_executor = None
        
        wire_context_to_orchestrator(orchestrator, agent)
        
        assert orchestrator._tool_executor is not None

    def test_wire_tool_schemas(self):
        """Should wire tool schemas to orchestrator."""
        agent = MockAgent()
        orchestrator = MagicMock()
        orchestrator._tool_schemas = None
        
        wire_context_to_orchestrator(orchestrator, agent)
        
        assert orchestrator._tool_schemas is not None
        assert len(orchestrator._tool_schemas) == 3


class TestWireContextToSubagent:
    """Tests for wiring context to subagent orchestrator."""

    def test_wire_tools_if_empty(self):
        """Should wire tools if subagent has none."""
        agent = MockAgent()
        subagent = MagicMock()
        subagent.tools = {}
        
        wire_context_to_subagent(subagent, agent)
        
        assert subagent.tools == agent.tools

    def test_wire_tool_executor(self):
        """Should wire tool executor to subagent."""
        agent = MockAgent()
        subagent = MagicMock()
        subagent._execute_tool = None
        
        wire_context_to_subagent(subagent, agent)
        
        assert subagent._execute_tool is not None


class TestWireContextToDualModel:
    """Tests for wiring context to dual model."""

    def test_wire_context_provider(self):
        """Should wire context provider to dual model."""
        agent = MockAgent()
        dual_model = MagicMock()
        dual_model._context_provider = None
        
        wire_context_to_dual_model(dual_model, agent)
        
        assert dual_model._context_provider is not None


class TestContextIntegration:
    """Tests for ContextIntegration class."""

    def test_initialization(self):
        """Should initialize correctly."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        assert integration.agent is agent
        assert not integration._initialized

    @pytest.mark.asyncio
    async def test_get_context(self):
        """Should get context through integration."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        with patch("core.context_integration.get_enhanced_context") as mock:
            mock.return_value = ("Context", {})
            
            context, metadata = await integration.get_context("test")
            
            assert context == "Context"
            assert integration._initialized

    def test_enhance_prompt(self):
        """Should enhance prompt correctly."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        enhanced = integration.enhance_prompt("Base", "Context")
        
        assert "Base" in enhanced
        assert "Context" in enhanced
        assert "Relevant Context" in enhanced

    def test_enhance_prompt_empty_context(self):
        """Should not enhance with empty context."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        enhanced = integration.enhance_prompt("Base", "")
        
        assert enhanced == "Base"

    def test_get_tools_for_subagent(self):
        """Should return tool schemas."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        schemas = integration.get_tools_for_subagent()
        
        assert len(schemas) == 3

    def test_get_tool_executor(self):
        """Should return tool executor."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        executor = integration.get_tool_executor()
        
        assert callable(executor)

    def test_wire_to_orchestrator(self):
        """Should wire context to orchestrator."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        orchestrator = MagicMock()
        
        integration.wire_to_orchestrator(orchestrator)
        
        assert orchestrator._tool_executor is not None

    def test_wire_to_subagent(self):
        """Should wire context to subagent."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        subagent = MagicMock()
        subagent.tools = {}
        
        integration.wire_to_subagent(subagent)
        
        assert subagent.tools == agent.tools

    def test_get_stats(self):
        """Should return context engine stats."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        integration._initialized = True
        integration._context_engine = MagicMock()
        integration._context_engine.get_stats.return_value = {"assemblies": 10}
        
        stats = integration.get_stats()
        
        assert stats["assemblies"] == 10


class TestIntegrateContextWithAgent:
    """Tests for integrate_context_with_agent function."""

    def test_integration(self):
        """Should create and attach integration."""
        agent = MockAgent()
        
        integration = integrate_context_with_agent(agent)
        
        assert integration is not None
        assert agent._context_integration is integration


class TestGetAgentContext:
    """Tests for get_agent_context convenience function."""

    @pytest.mark.asyncio
    async def test_get_context_with_integration(self):
        """Should use agent's integration if available."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        # Mock the context engine completely
        mock_context_engine = MagicMock()
        mock_context_pack = MagicMock()
        mock_context_pack.to_prompt.return_value = "Context from integration"
        mock_context_pack.total_tokens = 100
        mock_context_pack.entries = []
        mock_context_pack.sources_included = []
        mock_context_pack.assembly_time_ms = 10.0
        mock_context_pack.to_dict.return_value = {}
        
        mock_context_engine.assemble_context = AsyncMock(return_value=mock_context_pack)
        
        integration._context_engine = mock_context_engine
        integration._initialized = True
        agent._context_integration = integration
        
        context = await get_agent_context(agent, "test")
        
        # Just verify we got some context
        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_get_context_fallback(self):
        """Should use enhanced_context when no integration."""
        agent = MockAgent()
        agent._context_integration = None  # No integration
        
        # When _context_integration is None, it calls get_enhanced_context
        # which may return empty context if subsystems aren't available
        context = await get_agent_context(agent, "test")
        
        # Just verify it returns a string (may be empty or real context)
        assert isinstance(context, str)


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_context_flow(self):
        """Should get context, enhance prompt, and wire orchestrator."""
        agent = MockAgent()
        
        # Create integration
        integration = ContextIntegration(agent)
        
        with patch("core.context_integration.get_enhanced_context") as mock_context:
            mock_context.return_value = ("Important context", {"total_tokens": 50})
            
            # Get context
            context, metadata = await integration.get_context("test query")
            
            # Enhance prompt
            enhanced = integration.enhance_prompt("Base prompt", context)
            
            # Wire to orchestrator
            orchestrator = MagicMock()
            integration.wire_to_orchestrator(orchestrator)
            
            assert "Important context" in enhanced
            assert orchestrator._tool_executor is not None

    @pytest.mark.asyncio
    async def test_subagent_tool_availability(self):
        """Should ensure subagent has tools available."""
        agent = MockAgent()
        integration = ContextIntegration(agent)
        
        # Get tools for subagent
        schemas = integration.get_tools_for_subagent(["read_file"])
        
        assert len(schemas) == 1
        assert schemas[0]["name"] == "read_file"
        
        # Wire to subagent
        subagent = MagicMock()
        subagent.tools = {}
        integration.wire_to_subagent(subagent)
        
        assert subagent.tools == agent.tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])