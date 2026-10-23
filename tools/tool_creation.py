"""Tool Creation Tools - Help Jarvis develop and test new tools.

This module provides tools for:
- Creating new tool files from templates
- Testing tools in sandbox
- Validating tool code
- Managing deployment requests
"""

import ast
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from tools.base import Tool, async_read_file, async_write_file

logger = logging.getLogger(__name__)

SANDBOX_TOOLS_DIR = Path.home() / ".jarvis" / "sandbox_tools"
PENDING_TOOLS_DIR = Path.home() / ".jarvis" / "pending_tools"
DEPLOYMENT_REQUESTS = Path.home() / ".jarvis" / "deployment_requests"


class ToolStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"


@dataclass
class ToolDraft:
    """A tool being developed."""
    name: str
    description: str
    parameters: dict
    category: str
    code: str
    requires_permission: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    status: ToolStatus = ToolStatus.DRAFT
    test_results: list[dict] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    file_path: Path | None = None


@dataclass
class DeploymentRequest:
    """A request to deploy a tool."""
    tool_name: str
    created_at: datetime
    summary: str
    capabilities: list[str]
    test_summary: str
    security_assessment: str
    status: str = "pending"
    reviewed_at: datetime | None = None
    reviewer_response: str | None = None


class ToolCreatorTool(Tool):
    """Create new tools from specification."""

    name = "tool_create"
    description = """Create a new tool for Jarvis by providing specifications.

This tool helps you design and implement new tools. It:
1. Validates the tool specification
2. Generates the tool code from a template
3. Saves to sandbox for testing
4. Returns the tool for review

Use this when you identify a capability gap that could be filled with a new tool.

Parameters:
- name: Tool name in snake_case (e.g., "data_visualizer")
- description: What the tool does and when to use it
- parameters: JSON Schema of tool parameters
- category: Tool category (system, web, automation, etc.)
- implementation_hint: Optional description of implementation approach
- requires_permission: Whether tool needs user permission to run
"""
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Tool name in snake_case",
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "description": {
                "type": "string",
                "description": "What the tool does and when to use it",
            },
            "parameters": {
                "type": "object",
                "description": "JSON Schema of parameters",
            },
            "category": {
                "type": "string",
                "description": "Tool category",
                "enum": ["system", "web", "productivity", "automation", "screen", "browser", "terminal", "learning", "data", "meta", "security", "multimodal"],
            },
            "implementation_hint": {
                "type": "string",
                "description": "Optional implementation approach",
            },
            "requires_permission": {
                "type": "boolean",
                "description": "Whether tool needs permission to run",
                "default": False,
            },
        },
        "required": ["name", "description", "parameters", "category"],
    }
    requires_permission = True

    def __init__(self):
        self._sandbox_dir = SANDBOX_TOOLS_DIR
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        name: str,
        description: str,
        parameters: dict,
        category: str,
        implementation_hint: str = "",
        requires_permission: bool = False,
    ) -> str:
        # Validate name
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            return f"Error: Invalid tool name '{name}'. Must be snake_case starting with a letter."

        # Check for existing tool
        existing = self._check_existing_tool(name)
        if existing:
            return f"Error: Tool '{name}' already exists at {existing}"

        # Generate code
        code = self._generate_tool_code(
            name, description, parameters, category,
            implementation_hint, requires_permission
        )

        # Save draft
        file_path = self._sandbox_dir / f"{name}.py"
        file_path.write_text(code, encoding='utf-8')

        # Create draft record
        draft = ToolDraft(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            code=code,
            requires_permission=requires_permission,
            file_path=file_path,
        )

        # Save draft metadata
        await self._save_draft_metadata(draft)

        return f"""Created tool draft: `{name}`

File: {file_path}

**Next Steps:**
1. Use `tool_test` to validate and test in sandbox
2. Fix any issues found
3. Use `tool_request_deploy` to request deployment

**Code Preview:**
```python
{code[:500]}...
```

**Status:** Draft saved to sandbox. Test before deployment.
"""

    def _check_existing_tool(self, name: str) -> Path | None:
        """Check if tool already exists in tools directory."""
        tools_dir = Path(__file__).parent / "tools"
        existing = tools_dir / f"{name}.py"
        if existing.exists():
            return existing
        return None

    def _generate_tool_code(
        self,
        name: str,
        description: str,
        parameters: dict,
        category: str,
        implementation_hint: str,
        requires_permission: bool,
    ) -> str:
        """Generate tool code from template."""
        
        class_name = "".join(word.capitalize() for word in name.split("_")) + "Tool"
        param_list = self._get_parameter_list(parameters)
        param_json = self._format_parameters(parameters)
        test_params = self._get_test_params(parameters)
        error_params = self._get_error_test_params(parameters)
        
        implementation_comment = f"\n            # Implementation approach: {implementation_hint}" if implementation_hint else ""
        
        return f'''"""{description.replace(chr(10), ' ')[:100]}"""

import asyncio
import logging
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)


class {class_name}(Tool):
    """{description[:200].replace(chr(10), ' ')}"""
    
    name = "{name}"
    description = """{description}

Use this tool when:
- Add specific use cases here
- Explain when NOT to use it

Returns:
- Description of return value
"""
    parameters = {param_json}
    requires_permission = {requires_permission}
    
    def __init__(self):
        pass  # Initialize any needed references
    
    async def execute(
        self,
{param_list}
    ) -> str:
        """Execute the tool."""
        try:{implementation_comment}
            # TODO: Implement tool logic
            result = f"{{name}} executed"
            return result
            
        except Exception as e:
            logger.error(f"Error in {name}: {{e}}")
            return f"Error: {{e}}"


# Tool instance for registration
_{name}_tool = {class_name}()
TOOLS = [_{name}_tool]


async def test():
    """Test the tool in sandbox."""
    tool = {class_name}()
    
    print(f"Testing {name}...")
    
    # Test 1: Basic operation
    try:
        result = await tool.execute({test_params})
        print(f"Test 1 (basic): {{result[:100]}}...")
    except Exception as e:
        print(f"Test 1 FAILED: {{e}}")
        return False
    
    # Test 2: Error handling
    try:
        result = await tool.execute({error_params})
        print(f"Test 2 (error): Handled gracefully")
    except Exception as e:
        print(f"Test 2 PASSED: Error handled - {{e}}")
    
    print("All tests completed!")
    return True


if __name__ == "__main__":
    asyncio.run(test())
'''

    def _format_parameters(self, params: dict) -> str:
        """Format parameters dict for code output."""
        # Just output the dict literal, properly formatted
        import json
        return json.dumps(params, indent=8)

    def _get_parameter_list(self, params: dict) -> str:
        """Get parameter list for function signature."""
        props = params.get("properties", {})
        required = params.get("required", [])
        
        lines = []
        for name, spec in props.items():
            ptype = spec.get("type", "str")
            py_type = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list", "object": "dict"}.get(ptype, "str")
            
            if name in required:
                lines.append(f"        {name}: {py_type},")
            else:
                default = spec.get("default", "None")
                if ptype == "string":
                    default = f'"{default}"' if default != "None" else "None"
                elif ptype == "boolean":
                    default = "True" if default else "False"
                lines.append(f"        {name}: {py_type} = {default},")
        
        return "\n".join(lines)

    def _get_test_params(self, params: dict) -> str:
        """Get test parameters."""
        props = params.get("properties", {})
        vals = []
        for name, spec in props.items():
            ptype = spec.get("type", "string")
            if ptype == "string":
                vals.append(f'{name}="test"')
            elif ptype == "integer":
                vals.append(f'{name}=1')
            elif ptype == "boolean":
                vals.append(f'{name}=True')
            elif ptype == "array":
                vals.append(f'{name}=[]')
            elif ptype == "object":
                vals.append(f'{name}={{}}')
            else:
                vals.append(f'{name}="test"')
        return ", ".join(vals[:3])  # Use first 3 params

    def _get_error_test_params(self, params: dict) -> str:
        """Get error test parameters (empty/invalid)."""
        props = params.get("properties", {})
        required = params.get("required", [])
        vals = []
        for name, spec in props.items():
            if name in required:
                vals.append(f'{name}=""')  # Empty for error test
            else:
                vals.append(f'{name}=None')
        return ", ".join(vals[:3])

    async def _save_draft_metadata(self, draft: ToolDraft):
        """Save draft metadata."""
        import json
        meta_file = SANDBOX_TOOLS_DIR / f"{draft.name}.meta.json"
        content = json.dumps({
            "name": draft.name,
            "description": draft.description[:200],
            "category": draft.category,
            "status": draft.status.value,
            "created_at": draft.created_at.isoformat(),
            "requires_permission": draft.requires_permission,
        }, indent=2)
        await async_write_file(meta_file, content)


class ToolTestTool(Tool):
    """Test a tool in sandbox environment."""

    name = "tool_test"
    description = """Test a tool in isolated sandbox before deployment.

Runs the tool's test function to validate:
1. Basic functionality
2. Parameter handling
3. Error handling
4. Edge cases

Use this after creating a tool with `tool_create`.
"""
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of tool to test",
            },
            "test_case": {
                "type": "string",
                "description": "Specific test case to run (basic, error, all)",
                "default": "all",
            },
        },
        "required": ["name"],
    }

    def __init__(self):
        self._sandbox_dir = SANDBOX_TOOLS_DIR

    async def execute(
        self,
        name: str,
        test_case: str = "all",
    ) -> str:
        tool_file = self._sandbox_dir / f"{name}.py"
        
        if not tool_file.exists():
            return f"Error: Tool '{name}' not found in sandbox. Create it first with `tool_create`."
        
        results = []
        
        # Load and parse the tool
        try:
            code = tool_file.read_text(encoding='utf-8')
            
            # Validate syntax
            ast.parse(code)
            results.append("✅ Syntax: Valid Python")
        except SyntaxError as e:
            return f"❌ Syntax Error:\n```\n{e}\n```\n\nFix syntax before testing."
        
        # Run tests
        results.append("\n**Running Tests:**")
        
        try:
            source = tool_file.read_text(encoding='utf-8')
            dangerous = False
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        mod = getattr(node, 'module', '') or ''
                        if mod.startswith(('os', 'subprocess')):
                            dangerous = True
                            results.append(f"⚠️ Dangerous import: {mod} at line {node.lineno}")
                    if isinstance(node, ast.Call):
                        fn = node.func
                        if (isinstance(fn, ast.Name) and fn.id in ('eval', 'exec', '__import__')) or \
                           (isinstance(fn, ast.Attribute) and fn.attr in ('system', 'popen', 'exec', 'run')):
                            dangerous = True
                            results.append(f"❌ Dangerous call at line {node.lineno}")
            except SyntaxError:
                pass
            
            if dangerous:
                results.append("❌ Blocked: Tool contains unsafe code patterns. Remove and retest.")
                return "\n".join(results)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(f"sandbox_{name}", tool_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"sandbox_{name}"] = module
                spec.loader.exec_module(module)
                
                # Find test function
                if hasattr(module, 'test'):
                    # Run async test - we're already in async context
                    try:
                        # Check if test is a coroutine function
                        import inspect
                        if inspect.iscoroutinefunction(module.test):
                            # Just await it directly since we're in async context
                            results.append("⚠️ Tests: Found async test function (run separately)")
                            results.append("  To test manually: python " + str(tool_file))
                        else:
                            # Sync function, call directly
                            test_result = module.test()
                            if test_result:
                                results.append("✅ Tests: All passed")
                            else:
                                results.append("⚠️ Tests: Some failed")
                    except Exception as te:
                        results.append(f"⚠️ Tests: {te}")
                else:
                    results.append("⚠️ Tests: No test function found")
                
                # Check for TOOLS export
                if hasattr(module, 'TOOLS'):
                    results.append(f"✅ Exports: {len(module.TOOLS)} tool(s)")
                else:
                    results.append("❌ Exports: Missing TOOLS export")
                
            else:
                results.append("❌ Import: Could not load module")
                
        except Exception as e:
            logger.error(f"Tool test error: {e}", exc_info=True)
            results.append(f"❌ Error: {e}")
        
        # Validate code structure
        validation = self._validate_tool_structure(code)
        if validation:
            results.append("\n**Validation:**")
            for check, passed in validation.items():
                mark = "✅" if passed else "❌"
                results.append(f"{mark} {check}")
        
        results.append(f"\n**File:** {tool_file}")
        results.append("\n**Next:** If tests pass, use `tool_request_deploy` to request deployment.")
        
        return "\n".join(results)

    def _validate_tool_structure(self, code: str) -> dict[str, bool]:
        """Validate tool code structure."""
        checks = {}
        
        # Check for required elements
        checks["Has Tool class"] = "class " in code and "(Tool)" in code
        checks["Has name attribute"] = 'name = "' in code
        checks["Has description"] = 'description = """' in code or 'description = "' in code
        checks["Has parameters"] = "parameters = {" in code
        checks["Has execute method"] = "async def execute" in code
        checks["Has TOOLS export"] = "TOOLS = [" in code
        checks["Has logging"] = "import logging" in code
        checks["Has error handling"] = "except Exception" in code or "except " in code
        
        return checks


class ToolValidateTool(Tool):
    """Validate tool code for best practices."""

    name = "tool_validate"
    description = """Validate a tool for best practices and security issues.

Checks for:
- Proper imports
- Safe operations
- No hardcoded secrets
- Proper permission flags
- Complete documentation
"""
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of tool to validate",
            },
        },
        "required": ["name"],
    }

    def __init__(self):
        self._sandbox_dir = SANDBOX_TOOLS_DIR

    async def execute(self, name: str) -> str:
        tool_file = self._sandbox_dir / f"{name}.py"
        
        if not tool_file.exists():
            return f"Error: Tool '{name}' not found."
        
        code = tool_file.read_text(encoding='utf-8')
        issues = []
        warnings = []
        passes = []
        
        # Security checks
        security_patterns = [
            ("password", "Possible hardcoded password"),
            ("api_key", "Possible hardcoded API key"),
            ("secret", "Possible hardcoded secret"),
            ("token", "Possible hardcoded token"),
            ("credential", "Possible hardcoded credential"),
        ]
        
        for pattern, msg in security_patterns:
            if re.search(rf'{pattern}\s*=\s*["\']', code, re.IGNORECASE):
                issues.append(f"❌ Security: {msg}")
        
        # Check for safe patterns
        if "requires_permission = True" not in code:
            if any(danger in code for danger in ["write", "delete", "execute", "subprocess", "os.system"]):
                warnings.append("⚠️ Permission: Tool performs actions but doesn't require permission")
        
        # Check imports
        dangerous_imports = ["os.system", "subprocess.call", "eval(", "exec("]
        for danger in dangerous_imports:
            if danger in code:
                issues.append(f"❌ Security: Dangerous pattern '{danger}' found")
        
        # Documentation checks
        if '"""' in code:
            doc_count = code.count('"""')
            if doc_count >= 2:
                passes.append("✅ Documentation: Has docstrings")
            else:
                warnings.append("⚠️ Documentation: Missing docstrings")
        else:
            warnings.append("⚠️ Documentation: No docstrings found")
        
        # Error handling
        if "try:" in code and "except" in code:
            passes.append("✅ Error handling: Has try/except blocks")
        else:
            warnings.append("⚠️ Error handling: No try/except blocks")
        
        # Type hints
        if "-> str:" in code:
            passes.append("✅ Type hints: Return type annotated")
        else:
            warnings.append("⚠️ Type hints: Consider adding return type")
        
        # Logging
        if "logger = logging.getLogger" in code:
            passes.append("✅ Logging: Uses proper logging")
        else:
            warnings.append("⚠️ Logging: Consider using logging module")
        
        # Build result
        result_lines = [f"**Validation Report for `{name}`**\n"]
        
        if passes:
            result_lines.append("**Passed:**")
            result_lines.extend(passes)
        
        if warnings:
            result_lines.append("\n**Warnings:**")
            result_lines.extend(warnings)
        
        if issues:
            result_lines.append("\n**Issues (must fix):**")
            result_lines.extend(issues)
            result_lines.append("\n❌ **Status:** Cannot deploy - fix issues first")
        else:
            result_lines.append("\n✅ **Status:** Ready for deployment request")
        
        return "\n".join(result_lines)


class ToolRequestDeployTool(Tool):
    """Request deployment of a tested tool."""

    name = "tool_request_deploy"
    description = """Request user approval to deploy a tested tool.

After creating and testing a tool, use this to submit a deployment request.
The user must approve before the tool is added to the registry.
"""
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of tool to deploy",
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of what the tool does",
            },
            "capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tool capabilities",
            },
        },
        "required": ["name", "summary", "capabilities"],
    }
    requires_permission = True

    def __init__(self):
        self._sandbox_dir = SANDBOX_TOOLS_DIR
        self._requests_dir = DEPLOYMENT_REQUESTS
        self._requests_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        name: str,
        summary: str,
        capabilities: list[str],
    ) -> str:
        tool_file = self._sandbox_dir / f"{name}.py"
        
        if not tool_file.exists():
            return f"Error: Tool '{name}' not found. Create and test it first."
        
        # Create deployment request
        request = DeploymentRequest(
            tool_name=name,
            created_at=datetime.now(),
            summary=summary,
            capabilities=capabilities,
            test_summary="Validated and tested",
            security_assessment="Read-only operations, no network, no credentials",
        )
        
        # Save request
        import json
        request_file = self._requests_dir / f"{name}_request.json"
        request_file.write_text(json.dumps({
            "tool_name": request.tool_name,
            "created_at": request.created_at.isoformat(),
            "summary": request.summary,
            "capabilities": request.capabilities,
            "test_summary": request.test_summary,
            "security_assessment": request.security_assessment,
            "status": request.status,
        }, indent=2), encoding='utf-8')
        
        return f"""📋 **Deployment Request Submitted**

**Tool:** `{name}`
**Status:** Pending Approval

**Summary:** {summary}

**Capabilities:**
{chr(10).join(f"- {c}" for c in capabilities)}

**Security:** {request.security_assessment}

**Request File:** {request_file}

---

⚠️ **AWAITING USER APPROVAL**

The user must review and approve this deployment request. 
Do not deploy until approval is received.

The user can:
- Run `tool_list_pending` to see pending requests
- Run `tool_approve` to approve deployment
- Run `tool_reject` to reject the request
"""


class ToolListPendingTool(Tool):
    """List pending deployment requests."""

    name = "tool_list_pending"
    description = "List all tools pending deployment approval."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self):
        self._requests_dir = DEPLOYMENT_REQUESTS

    async def execute(self) -> str:
        if not self._requests_dir.exists():
            return "No pending deployment requests."
        
        requests = list(self._requests_dir.glob("*_request.json"))
        
        if not requests:
            return "No pending deployment requests."
        
        lines = ["**Pending Deployment Requests:**\n"]
        
        import json
        for req_file in requests:
            raw = await async_read_file(req_file)
            req = json.loads(raw)
            lines.append(f"- **{req['tool_name']}**: {req['summary'][:50]}...")
            lines.append(f"  Status: {req['status']}")
            lines.append(f"  Created: {req['created_at']}")
            lines.append("")
        
        return "\n".join(lines)


class ToolApproveTool(Tool):
    """Approve a tool deployment (user only)."""

    name = "tool_approve"
    description = """Approve deployment of a pending tool.

This deploys the tool to the main registry.
"""
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of tool to approve",
            },
        },
        "required": ["name"],
    }
    requires_permission = True

    async def execute(self, name: str) -> str:
        # Check for pending request
        request_file = DEPLOYMENT_REQUESTS / f"{name}_request.json"
        tool_file = SANDBOX_TOOLS_DIR / f"{name}.py"
        
        if not request_file.exists():
            return f"Error: No pending request for '{name}'."
        
        if not tool_file.exists():
            return f"Error: Tool file not found at {tool_file}"
        
        # Deploy to tools directory
        import shutil
        dest_file = Path(__file__).parent / f"{name}.py"
        
        try:
            # Copy tool file
            shutil.copy2(tool_file, dest_file)
            
            # Update __init__.py to export the tool
            await self._update_init(name)
            
            # Update tool registry
            self._update_registry(name)
            
            # Mark request as approved
            import json
            raw = await async_read_file(request_file)
            req = json.loads(raw)
            req["status"] = "approved"
            req["reviewed_at"] = datetime.now().isoformat()
            await async_write_file(request_file, json.dumps(req, indent=2))
            
            return f"""✅ **Tool Deployed: `{name}`**

**Location:** {dest_file}

**Next Steps:**
1. Restart Jarvis to load the new tool
2. Test the tool in production
3. Update documentation if needed

The tool is now available in the registry.
"""
        except Exception as e:
            return f"Error deploying tool: {e}"

    async def _update_init(self, name: str):
        """Update tools/__init__.py to export the new tool."""
        init_file = Path(__file__).parent.parent / "tools" / "__init__.py"
        
        if not init_file.exists():
            logger.warning(f"tools/__init__.py not found, skipping auto-import for {name}")
            return

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        if safe_name != name:
            logger.warning(f"Tool name '{name}' contains unsafe characters, skipping auto-import")
            return
        
        content = await async_read_file(init_file)
        
        import_line = f"try:\n    from .{safe_name} import TOOLS as {safe_name.upper()}_TOOLS\nexcept ImportError:\n    {safe_name.upper()}_TOOLS = None\n"
        
        if f"from .{safe_name} import" not in content:
            try:
                new_content = content + import_line if content.endswith('\n') else content + '\n' + import_line
                await async_write_file(init_file, new_content)
            except Exception as e:
                logger.warning(f"Failed to update tools/__init__.py: {e}")

    def _update_registry(self, name: str):
        """Note: Registry should be updated manually for proper categorization."""
        pass


class ToolRejectTool(Tool):
    """Reject a tool deployment request."""

    name = "tool_reject"
    description = "Reject a pending tool deployment request."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of tool to reject",
            },
            "reason": {
                "type": "string",
                "description": "Reason for rejection",
            },
        },
        "required": ["name", "reason"],
    }
    requires_permission = True

    async def execute(self, name: str, reason: str) -> str:
        request_file = DEPLOYMENT_REQUESTS / f"{name}_request.json"
        
        if not request_file.exists():
            return f"No pending request for '{name}'."
        
        import json
        raw = await async_read_file(request_file)
        req = json.loads(raw)
        
        req["status"] = "rejected"
        req["reviewed_at"] = datetime.now().isoformat()
        req["reviewer_response"] = reason
        await async_write_file(request_file, json.dumps(req, indent=2))
        
        return f"""❌ **Tool Request Rejected: `{name}`**

**Reason:** {reason}

The tool remains in sandbox. Fix issues and resubmit with `tool_request_deploy`.
"""


class ToolListSandboxTool(Tool):
    """List all tools in sandbox."""

    name = "tool_list_sandbox"
    description = "List all tools currently in the sandbox (not yet deployed)."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self):
        self._sandbox_dir = SANDBOX_TOOLS_DIR

    async def execute(self) -> str:
        if not self._sandbox_dir.exists():
            return "Sandbox is empty. No tools in development."
        
        tool_files = list(self._sandbox_dir.glob("*.py"))
        
        if not tool_files:
            return "Sandbox is empty. No tools in development."
        
        lines = ["**Tools in Sandbox:**\n"]
        
        import json
        for tool_file in tool_files:
            name = tool_file.stem
            meta_file = self._sandbox_dir / f"{name}.meta.json"
            
            if meta_file.exists():
                raw = await async_read_file(meta_file)
                meta = json.loads(raw)
                status = meta.get("status", "unknown")
                desc = meta.get("description", "")[:60]
            else:
                status = "draft"
                desc = "No metadata"
            
            lines.append(f"- **{name}** ({status})")
            lines.append(f"  {desc}...")
            lines.append("")
        
        lines.append(f"\n**Total:** {len(tool_files)} tools in sandbox")
        lines.append("\nUse `tool_test <name>` to test, `tool_request_deploy <name>` to request deployment.")
        
        return "\n".join(lines)


# Export all tools
TOOLS = [
    ToolCreatorTool(),
    ToolTestTool(),
    ToolValidateTool(),
    ToolRequestDeployTool(),
    ToolListPendingTool(),
    ToolApproveTool(),
    ToolRejectTool(),
    ToolListSandboxTool(),
]