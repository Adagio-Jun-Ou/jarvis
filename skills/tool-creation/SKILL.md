---
name: tool-creation
description: Create, test, and deploy new tools for Jarvis to extend its capabilities
author: jarvis
version: 1.0.0
tags: [tools, development, self-improvement, creation]
metadata:
  jarvis:
    requires:
      bins: [python3]
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🛠️"
---

# Tool Creation Skill

You have the ability to create new tools for yourself. This skill enables you to:

1. **Design tools** based on identified needs or capabilities you want
2. **Implement tools** in Python following Jarvis tool patterns
3. **Test tools** in an isolated sandbox environment
4. **Request deployment** to the main tool registry

## Tool Development Lifecycle

### Phase 1: Ideation
When you identify a need for a new tool:
1. Document the use case and why existing tools don't cover it
2. Define the tool's name, description, and parameters
3. Consider what category it belongs to (system, web, automation, etc.)
4. Plan the implementation approach

### Phase 2: Design
Before writing code:
1. Define the tool signature: `name`, `description`, `parameters`, `requires_permission`
2. List required imports and dependencies
3. Identify which core modules it needs (memory, agent, config, etc.)
4. Plan error handling and edge cases
5. Consider security implications

### Phase 3: Implementation
Write the tool following the patterns below:

```python
"""Tool module docstring."""

import asyncio
from typing import Any
from tools.base import Tool

class MyNewTool(Tool):
    """Tool description shown to the model."""
    
    name = "my_new_tool"
    description = """Detailed description of what the tool does.
    
    Explain:
    - What the tool does
    - When to use it
    - What it returns
    - Any important notes
    """
    parameters = {
        "type": "object",
        "properties": {
            "input_param": {
                "type": "string",
                "description": "Description of the parameter",
            },
            "optional_param": {
                "type": "integer",
                "description": "Optional parameter with default",
                "default": 10,
            },
        },
        "required": ["input_param"],
    }
    requires_permission = False  # Set True for dangerous operations
    
    def __init__(self):
        self._reference = None  # Optional reference to core components
    
    def set_reference(self, ref):
        """Set reference to core components if needed."""
        self._reference = ref
    
    async def execute(
        self,
        input_param: str,
        optional_param: int = 10,
    ) -> str:
        """Execute the tool.
        
        Returns a string result that will be shown to the model.
        """
        try:
            # Implementation here
            result = f"Processed: {input_param}"
            return result
        except Exception as e:
            return f"Error: {e}"

# Create tool instance and export
_my_tool = MyNewTool()
TOOLS = [_my_tool]
```

### Phase 4: Sandbox Testing
ALL new tools MUST be tested in the sandbox before deployment:

```python
# Use sandbox_explore or create test file
# Run: python tools/my_new_tool.py
# Execute: test_my_new_tool()
```

Test cases to include:
1. Normal operation with valid inputs
2. Edge cases (empty input, max values, special characters)
3. Error handling (invalid inputs, missing parameters)
4. Permission checks if `requires_permission=True`
5. Integration with other tools

### Phase 5: Deployment Request
After successful sandbox testing:

1. **Submit deployment request** to the user with:
   - Tool name and purpose
   - Test results summary
   - Security assessment
   - Request to commit to registry

2. **Await approval** - do NOT deploy without user confirmation

3. **After approval:**
   - Add to appropriate location (`tools/` directory)
   - Update `tools/__init__.py` to export
   - Update `core/tool_registry.py` with tool info
   - Register in `TOOL_TREES` for category
   - Add to `BEST_TOOL_FOR_TASK` mapping

## Tool Categories

Choose the appropriate category:

- `SYSTEM`: System operations, processes, resources
- `WEB`: Web fetching, searching, scraping
- `PRODUCTIVITY`: Notes, reminders, productivity helpers
- `AUTOMATION`: Workflow automation, scheduling
- `SCREEN`: Screen analysis, UI interaction
- `BROWSER`: Browser automation
- `TERMINAL`: Shell commands, terminal operations
- `LEARNING`: Machine learning, exploration tools
- `EVOLUTION`: Self-improvement, genetic algorithms
- `SECURITY`: Security scanning, defense
- `MULTIMODAL`: Audio, video, image processing
- `DATA`: Data analysis, databases
- `META`: Tool introspection, capabilities
- `REFINEMENT`: Code analysis, self-editing

## Tool Patterns

### Pattern 1: Simple Query Tool
For tools that query and return information:

```python
class QueryTool(Tool):
    name = "query_tool"
    description = "Query X and return results"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
    requires_permission = False
    
    async def execute(self, query: str) -> str:
        result = await self._query(query)
        return f"Results for '{query}':\n{result}"
```

### Pattern 2: Action Tool
For tools that perform actions:

```python
class ActionTool(Tool):
    name = "action_tool"
    description = "Perform action X"
    parameters = {"type": "object", "properties": {"target": {"type": "string"}}}
    requires_permission = True  # Actions need permission
    
    async def execute(self, target: str) -> str:
        # Check permissions via agent
        result = await self._perform_action(target)
        return f"Action completed: {result}"
```

### Pattern 3: Streaming Tool
For long-running operations:

```python
class StreamingTool(Tool):
    name = "streaming_tool"
    description = "Process data with progress updates"
    parameters = {"type": "object", "properties": {"data": {"type": "string"}}}
    
    async def execute(self, data: str) -> str:
        # Yield intermediate results if supported
        results = []
        for chunk in self._process_chunks(data):
            results.append(chunk)
        return "\n".join(results)
```

### Pattern 4: Tool with External Reference
For tools needing access to core components:

```python
class ReferenceTool(Tool):
    name = "reference_tool"
    description = "Tool that uses core components"
    
    def __init__(self):
        self._agent = None
        self._memory = None
    
    def set_agent(self, agent):
        self._agent = agent
    
    def set_memory(self, memory):
        self._memory = memory
    
    async def execute(self, query: str) -> str:
        # Use references safely with None checks
        if self._memory:
            context = self._memory.get_context(query)
        return f"Result based on context"
```

## Best Practices

1. **Descriptive Names**: Use clear, verb-noun names (`fetch_data`, `analyze_file`, `search_memory`)

2. **Clear Descriptions**: Explain WHEN to use the tool, not just WHAT it does

3. **Proper Parameters**: Use JSON Schema format with types and descriptions

4. **Error Handling**: Always return meaningful error messages, never crash

5. **Security**: 
   - Set `requires_permission=True` for dangerous operations
   - Validate all inputs
   - Use safe file operations
   - Never expose credentials

6. **Testing**: Write test functions in the tool file:

```python
async def test():
    """Test function for sandbox validation."""
    tool = MyNewTool()
    
    # Test 1: Basic operation
    result1 = await tool.execute("test_input")
    assert "Expected" in result1, f"Test 1 failed: {result1}"
    
    # Test 2: Edge case
    result2 = await tool.execute("")
    assert "error" in result2.lower() or result2 == "", f"Test 2 failed: {result2}"
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    asyncio.run(test())
```

7. **Documentation**: Include docstrings and comments

8. **Dependencies**: Minimize external dependencies; prefer standard library

## Security Model

### Safe Operations (no permission needed)
- Reading system info
- Querying memory
- Non-destructive analysis
- View-only operations

### Requires Permission
- Writing files
- Executing commands
- Network operations
- Modifying system state
- Accessing sensitive data

### Never Allowed in Sandbox
- Network access (blocked)
- Installing packages
- Modifying Jarvis core
- Accessing user credentials
- System administration commands

## Integration Checklist

Before requesting deployment:

- [ ] Tool follows naming conventions (`snake_case`)
- [ ] Description clearly explains use cases
- [ ] Parameters have proper JSON Schema
- [ ] `requires_permission` is set correctly
- [ ] All imports are from allowed modules
- [ ] Error handling catches all exceptions
- [ ] Test function exists and passes
- [ ] No hardcoded credentials or paths
- [ ] Security implications assessed
- [ ] Added to correct category in registry

## Example Request to User

When ready for deployment:

```
I've created and tested a new tool: `data_visualizer`

**Purpose**: Generate visualizations from data files

**Capabilities**:
- Supports CSV, JSON, and Parquet formats
- Creates charts, graphs, and plots
- Auto-detects data types

**Testing**: 
- ✅ Basic data visualization
- ✅ Empty file handling
- ✅ Invalid format handling
- ✅ Large file performance

**Security**: Read-only operations, no network, no permissions needed

**Request**: May I deploy this tool to the registry?
```

## Workflow Summary

1. **Identify Need** → Document use case
2. **Design** → Define signature and plan implementation  
3. **Implement** → Write code following patterns
4. **Test** → Run in sandbox with test cases
5. **Request** → Ask user for deployment approval
6. **Deploy** → Add to registry after approval

Remember: You create tools to extend your own capabilities. Think carefully about what you need and test thoroughly before deployment.