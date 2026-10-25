"""Integration between Unified Context Engine and Agent.

This module provides the glue between:
1. UnifiedContextEngine - Context assembly from all sources
2. JarvisAgent - Main agent with tool execution
3. IntelligentOrchestrator - Task decomposition
4. SubAgentOrchestrator - Multi-agent coordination
5. DualModelOrchestrator - Cross-model context sharing

Key responsibilities:
- Inject unified context into agent think() 
- Provide tools to subagents
- Ensure cross-model context flows correctly
- Wire tool executor to orchestrator
"""

import logging
from core.helpers import safe_error_message
from typing import Optional

logger = logging.getLogger(__name__)


async def get_enhanced_context(
    query: str,
    agent,
    max_tokens: int = 2000,
    include_insights: bool = True,
    include_cross_model: bool = True,
    model: str = "primary",
) -> tuple[str, dict]:
    """Get enhanced context from unified context engine.
    
    Args:
        query: User query for context relevance
        agent: JarvisAgent instance for memory access
        max_tokens: Maximum tokens for context
        include_insights: Include insights from bridges
        include_cross_model: Include other model's context
        model: Which model will receive this context
    
    Returns:
        Tuple of (context_prompt, context_metadata)
    """
    from core.unified_context_engine import get_context_engine
    
    context_engine = get_context_engine()
    
    try:
        # Assemble context from all sources
        context_pack = await context_engine.assemble_context(
            query=query,
            max_tokens=max_tokens,
            include_insights=include_insights,
            include_cross_model=include_cross_model,
            model=model,
        )
        
        # Convert to prompt format
        context_prompt = context_pack.to_prompt(format="default")
        
        # Extract metadata for logging/debugging
        metadata = {
            "total_tokens": context_pack.total_tokens,
            "entry_count": len(context_pack.entries),
            "sources": [s.value for s in context_pack.sources_included],
            "assembly_time_ms": context_pack.assembly_time_ms,
        }
        
        return context_prompt, metadata
        
    except Exception as e:
        logger.warning(f"Failed to get enhanced context: {e}")
        return "", {}


def get_tool_schemas_for_subagent(
    agent,
    tool_names: Optional[list[str]] = None,
) -> list[dict]:
    """Get tool schemas for subagent use.
    
    Args:
        agent: JarvisAgent instance
        tool_names: Optional list of specific tool names (None = all)
    
    Returns:
        List of tool schemas for subagent
    """
    schemas = agent.get_tool_schemas()
    
    if tool_names is None:
        return schemas
    
    # Filter to specific tools
    tool_set = set(tool_names)
    return [s for s in schemas if s.get("name") in tool_set]


def get_tool_executor_for_orchestrator(agent):
    """Get a tool executor function for orchestrator use.
    
    This bridges the gap between IntelligentOrchestrator's
    placeholder tool execution and actual tool execution.
    
    Args:
        agent: JarvisAgent instance
    
    Returns:
        Async function that executes tools
    """
    async def execute_tool(tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return result."""
        try:
            # Use agent's execute_tool method
            result = await agent.execute_tool(tool_name, tool_args)
            return result.output if hasattr(result, 'output') else str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}: {e}")
            return f"Error: {safe_error_message(e)}"
    
    return execute_tool


async def enhance_system_prompt(
    agent,
    query: str,
    base_prompt: Optional[str] = None,
    model: str = "primary",
) -> str:
    """Enhance system prompt with unified context.
    
    Args:
        agent: JarvisAgent instance
        query: User query for context relevance
        base_prompt: Base prompt to enhance (None = use agent's)
        model: Which model will receive this
    
    Returns:
        Enhanced system prompt with context
    """
    if base_prompt is None:
        base_prompt = agent.get_system_prompt()
    
    try:
        context_prompt, _ = await get_enhanced_context(
            query=query,
            agent=agent,
            max_tokens=2000,
            model=model,
        )
        
        if context_prompt:
            # Insert context after base prompt
            enhanced = f"{base_prompt}\n\n## Relevant Context\n\n{context_prompt}"
            return enhanced
    except Exception as e:
        logger.debug(f"Failed to enhance system prompt: {e}")
    
    return base_prompt


def wire_context_to_orchestrator(orchestrator, agent):
    """Wire the context engine to an orchestrator.
    
    This ensures that IntelligentOrchestrator has access to:
    1. Tool executor for executing tool subtasks
    2. Context assembly for each subtask
    3. Tool schemas for planning
    
    Args:
        orchestrator: IntelligentOrchestrator instance
        agent: JarvisAgent instance
    """
    # Wire tool executor
    if hasattr(orchestrator, '_tool_executor'):
        orchestrator._tool_executor = get_tool_executor_for_orchestrator(agent)
    
    # Wire tool schemas
    if hasattr(orchestrator, '_tool_schemas'):
        orchestrator._tool_schemas = agent.get_tool_schemas()
    
    # Wire context provider (async wrapper)
    if hasattr(orchestrator, '_context_provider'):
        async def _provide_context(query: str):
            return await get_enhanced_context(query, agent)
        orchestrator._context_provider = _provide_context


def wire_context_to_subagent(subagent_orchestrator, agent):
    """Wire the context engine to subagent orchestrator.
    
    This ensures SubAgentOrchestrator has:
    1. Access to all tools
    2. Context provided to each agent
    3. Tool execution capability
    
    Args:
        subagent_orchestrator: SubAgentOrchestrator instance
        agent: JarvisAgent instance
    """
    
    # Ensure tools are available
    if hasattr(subagent_orchestrator, 'tools'):
        if not subagent_orchestrator.tools:
            subagent_orchestrator.tools = agent.tools
    
    # Set up tool executor
    if hasattr(subagent_orchestrator, '_execute_tool'):
        subagent_orchestrator._execute_tool = get_tool_executor_for_orchestrator(agent)


def wire_context_to_dual_model(dual_model, agent):
    """Wire the context engine to dual model orchestrator.
    
    This ensures DualModelOrchestrator:
    1. Gets unified context for each model
    2. Shares context between models
    3. Uses insights for routing decisions
    
    Args:
        dual_model: DualModelOrchestrator instance
        agent: JarvisAgent instance
    """
    # The EmbeddingInterconnect is already wired in dual_model
    # We just need to ensure the unified context engine is used
    
    async def get_context_for_model(query: str, model: str) -> str:
        """Get context for a specific model."""
        context_prompt, _ = await get_enhanced_context(
            query=query,
            agent=agent,
            model=model,
        )
        return context_prompt
    
    # Wire context provider if the dual_model supports it
    if hasattr(dual_model, '_context_provider'):
        dual_model._context_provider = get_context_for_model


class ContextIntegration:
    """Manages integration between UnifiedContextEngine and Agent components.
    
    Usage:
        integration = ContextIntegration(agent)
        
        # In agent.think():
        context = await integration.get_context(user_input)
        enhanced_prompt = integration.enhance_prompt(base_prompt, context)
        
        # For orchestrators:
        integration.wire_to_orchestrator(orchestrator)
        integration.wire_to_subagent(subagent_orchestrator)
    """
    
    def __init__(self, agent):
        self.agent = agent
        self._context_engine = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization."""
        if self._initialized:
            return
        
        from core.unified_context_engine import get_context_engine
        self._context_engine = get_context_engine()
        self._initialized = True
    
    async def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        model: str = "primary",
    ) -> tuple[str, dict]:
        """Get unified context for a query."""
        self._ensure_initialized()
        return await get_enhanced_context(
            query=query,
            agent=self.agent,
            max_tokens=max_tokens,
            model=model,
        )
    
    def enhance_prompt(self, base_prompt: str, context: str) -> str:
        """Enhance a base prompt with context."""
        if not context:
            return base_prompt
        
        return f"{base_prompt}\n\n## Relevant Context\n\n{context}"
    
    def get_tools_for_subagent(self, tool_names: Optional[list[str]] = None) -> list[dict]:
        """Get tool schemas for subagent."""
        return get_tool_schemas_for_subagent(self.agent, tool_names)
    
    def get_tool_executor(self):
        """Get tool executor function."""
        return get_tool_executor_for_orchestrator(self.agent)
    
    def wire_to_orchestrator(self, orchestrator):
        """Wire context to IntelligentOrchestrator."""
        wire_context_to_orchestrator(orchestrator, self.agent)
    
    def wire_to_subagent(self, subagent_orchestrator):
        """Wire context to SubAgentOrchestrator."""
        wire_context_to_subagent(subagent_orchestrator, self.agent)
    
    def wire_to_dual_model(self, dual_model):
        """Wire context to DualModelOrchestrator."""
        wire_context_to_dual_model(dual_model, self.agent)
    
    def get_stats(self) -> dict:
        """Get context engine statistics."""
        self._ensure_initialized()
        return self._context_engine.get_stats()


def integrate_context_with_agent(agent):
    """Integrate unified context with agent on initialization.
    
    Call this from agent.__init__() to set up the integration.
    
    Args:
        agent: JarvisAgent instance
    
    Returns:
        ContextIntegration instance
    """
    integration = ContextIntegration(agent)
    
    # Store reference on agent
    agent._context_integration = integration
    
    return integration


# Convenience function for use in agent.think()
async def get_agent_context(
    agent,
    query: str,
    max_tokens: int = 2000,
    model: str = "primary",
) -> str:
    """Get context string for use in agent.think().
    
    Args:
        agent: JarvisAgent instance
        query: User query
        max_tokens: Max tokens for context
        model: Target model
    
    Returns:
        Context prompt string
    """
    if hasattr(agent, '_context_integration') and agent._context_integration is not None:
        context_prompt, _ = await agent._context_integration.get_context(
            query=query,
            max_tokens=max_tokens,
            model=model,
        )
        return context_prompt
    
    # Fallback: direct call
    context_prompt, _ = await get_enhanced_context(
        query=query,
        agent=agent,
        max_tokens=max_tokens,
        model=model,
    )
    return context_prompt