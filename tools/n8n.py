"""n8n workflow automation tools for Jarvis.

This module provides tools for managing n8n workflows programmatically.
It enables Jarvis to design and execute automation workflows through
the n8n API.

Key features:
- List workflows
- Create/modify workflows (JSON-based)
- Activate/deactivate workflows
- Execute workflows on demand
- Monitor workflow status
- Design automation patterns
"""
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from .base import Tool
from core.helpers import create_tracked_task, safe_error_message

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of an n8n workflow."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNKNOWN = "unknown"


class N8NTool(Tool):
    """Tool for n8n workflow automation."""

    name = "n8n"
    description = "Manage n8n workflows - create, execute, and monitor automation"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The n8n action to perform",
                "enum": [
                    "list",
                    "get",
                    "create",
                    "update",
                    "delete",
                    "activate",
                    "deactivate",
                    "execute",
                    "executions",
                    "design",
                ],
            },
            "workflow_id": {
                "type": "string",
                "description": "Workflow ID (for get, update, delete, activate, deactivate, execute)",
            },
            "workflow_data": {
                "type": "object",
                "description": "Workflow JSON data (for create, update)",
            },
            "execution_id": {
                "type": "string",
                "description": "Execution ID (for checking execution status)",
            },
            "design_description": {
                "type": "string",
                "description": "Description for designing a workflow",
            },
        },
        "required": ["action"],
    }
    requires_permission = False  # n8n workflow automation allowed for autonomous use

    def __init__(
        self,
        base_url: str = "http://localhost:5678",
        api_key: str | None = None,
    ):
        """Initialize the n8n tool.

        Args:
            base_url: Base URL for n8n instance
            api_key: API key for authentication (optional if using API key header)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
        if not self.base_url.startswith(("http://localhost", "http://127.0.0.1", "http://host.docker.internal")):
            import ipaddress
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.base_url)
                host = parsed.hostname
                if host:
                    import socket
                    ip = socket.gethostbyname(host)
                    addr = ipaddress.ip_address(ip)
                    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                        pass
                    else:
                        logger.warning(f"n8n base_url points to external address: {ip}")
            except Exception as e:
                logger.debug(f"Failed to validate n8n base_url: {e}")

        headers = {}
        if api_key:
            headers["X-N8N-API-KEY"] = api_key

        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
        )

        self._n8n_available = None
        self._closed = False

    async def __aenter__(self) -> "N8NTool":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        if not self._closed:
            self._closed = True
            await self.client.aclose()

    def __del__(self) -> None:
        """Best-effort cleanup if not properly closed."""
        if not self._closed:
            try:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    # Tracked so the cleanup coroutine survives until it runs.
                    create_tracked_task(self.client.aclose(), name="n8n:aclose")
                    logger.debug("Scheduled n8n client cleanup via __del__")
                except RuntimeError:
                    pass
            except Exception as e:
                logger.warning(f"Failed to close n8n client in __del__: {e}")

    async def _check_n8n(self) -> bool:
        """Check if n8n is available."""
        if self._n8n_available is not None:
            return self._n8n_available

        try:
            response = await self.client.get(f"{self.base_url}/healthz")
            self._n8n_available = response.status_code == 200
            return self._n8n_available
        except Exception:
            self._n8n_available = False
            return False

    async def execute(
        self,
        action: str,
        workflow_id: str | None = None,
        workflow_data: dict[str, Any] | None = None,
        execution_id: str | None = None,
        design_description: str | None = None,
    ) -> str:
        """Execute an n8n action.

        Args:
            action: The n8n action to perform
            workflow_id: Workflow ID
            workflow_data: Workflow JSON data
            execution_id: Execution ID
            design_description: Description for design action

        Returns:
            Result of the n8n action
        """
        if not await self._check_n8n():
            return f"Error: n8n is not available at {self.base_url}. Make sure n8n is running."

        action_map = {
            "list": self._list_workflows,
            "get": self._get_workflow,
            "create": self._create_workflow,
            "update": self._update_workflow,
            "delete": self._delete_workflow,
            "activate": self._activate_workflow,
            "deactivate": self._deactivate_workflow,
            "execute": self._execute_workflow,
            "executions": self._get_executions,
            "design": self._design_workflow,
        }

        handler = action_map.get(action)
        if not handler:
            return f"Unknown n8n action: {action}"

        try:
            return await handler(
                workflow_id=workflow_id,
                workflow_data=workflow_data,
                execution_id=execution_id,
                design_description=design_description,
            )
        except httpx.HTTPStatusError as e:
            return f"n8n API error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Error executing n8n action: {safe_error_message(e)}"

    async def _list_workflows(self, **kwargs) -> str:
        """List all workflows."""
        response = await self.client.get(f"{self.base_url}/api/v1/workflows")
        response.raise_for_status()

        data = response.json()
        workflows = data.get("data", [])

        if not workflows:
            return "No workflows found."

        lines = ["n8n Workflows:"]
        for wf in workflows:
            wf_id = wf.get("id", "unknown")
            name = wf.get("name", "unnamed")
            active = wf.get("active", False)
            status_icon = "🟢" if active else "⚪"
            lines.append(f"  {status_icon} [{wf_id}] {name}")

        return "\n".join(lines)

    async def _get_workflow(
        self,
        workflow_id: str | None = None,
        **kwargs,
    ) -> str:
        """Get a specific workflow."""
        if not workflow_id:
            return "Error: Workflow ID required."

        response = await self.client.get(f"{self.base_url}/api/v1/workflows/{workflow_id}")
        response.raise_for_status()

        data = response.json()
        wf = data.get("data", {})

        lines = [
            f"Workflow: {wf.get('name', 'unnamed')}",
            f"ID: {wf.get('id')}",
            f"Status: {'Active' if wf.get('active') else 'Inactive'}",
            f"Created: {wf.get('createdAt', 'unknown')}",
            f"Updated: {wf.get('updatedAt', 'unknown')}",
            f"Nodes: {len(wf.get('nodes', []))}",
        ]

        # List nodes
        nodes = wf.get("nodes", [])
        if nodes:
            lines.append("\nNodes:")
            for node in nodes[:10]:  # Show first 10
                node_name = node.get("name", "unnamed")
                node_type = node.get("type", "unknown")
                lines.append(f"  • {node_name} ({node_type})")
            if len(nodes) > 10:
                lines.append(f"  ... and {len(nodes) - 10} more")

        return "\n".join(lines)

    async def _create_workflow(
        self,
        workflow_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Create a new workflow."""
        if not workflow_data:
            return "Error: Workflow data required for create action."

        # Ensure required fields
        if "name" not in workflow_data:
            workflow_data["name"] = f"Workflow {datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if "nodes" not in workflow_data:
            workflow_data["nodes"] = []
        if "connections" not in workflow_data:
            workflow_data["connections"] = {}

        response = await self.client.post(
            f"{self.base_url}/api/v1/workflows",
            json=workflow_data,
        )
        response.raise_for_status()

        data = response.json()
        wf = data.get("data", {})

        return f"Created workflow: {wf.get('name')} (ID: {wf.get('id')})"

    async def _update_workflow(
        self,
        workflow_id: str | None = None,
        workflow_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Update an existing workflow."""
        if not workflow_id:
            return "Error: Workflow ID required."
        if not workflow_data:
            return "Error: Workflow data required for update action."

        response = await self.client.put(
            f"{self.base_url}/api/v1/workflows/{workflow_id}",
            json=workflow_data,
        )
        response.raise_for_status()

        data = response.json()
        wf = data.get("data", {})

        return f"Updated workflow: {wf.get('name')} (ID: {wf.get('id')})"

    async def _delete_workflow(
        self,
        workflow_id: str | None = None,
        **kwargs,
    ) -> str:
        """Delete a workflow."""
        if not workflow_id:
            return "Error: Workflow ID required."

        response = await self.client.delete(
            f"{self.base_url}/api/v1/workflows/{workflow_id}",
        )
        response.raise_for_status()

        return f"Deleted workflow {workflow_id}"

    async def _activate_workflow(
        self,
        workflow_id: str | None = None,
        **kwargs,
    ) -> str:
        """Activate a workflow."""
        if not workflow_id:
            return "Error: Workflow ID required."

        response = await self.client.patch(
            f"{self.base_url}/api/v1/workflows/{workflow_id}/activate",
        )
        response.raise_for_status()

        return f"Activated workflow {workflow_id}"

    async def _deactivate_workflow(
        self,
        workflow_id: str | None = None,
        **kwargs,
    ) -> str:
        """Deactivate a workflow."""
        if not workflow_id:
            return "Error: Workflow ID required."

        response = await self.client.patch(
            f"{self.base_url}/api/v1/workflows/{workflow_id}/deactivate",
        )
        response.raise_for_status()

        return f"Deactivated workflow {workflow_id}"

    async def _execute_workflow(
        self,
        workflow_id: str | None = None,
        **kwargs,
    ) -> str:
        """Execute a workflow manually."""
        if not workflow_id:
            return "Error: Workflow ID required."

        response = await self.client.post(
            f"{self.base_url}/api/v1/workflows/{workflow_id}/execute",
        )
        response.raise_for_status()

        data = response.json()
        execution_id = data.get("executionId", "unknown")

        return f"Started workflow execution. Execution ID: {execution_id}"

    async def _get_executions(
        self,
        workflow_id: str | None = None,
        execution_id: str | None = None,
        **kwargs,
    ) -> str:
        """Get workflow executions."""
        if execution_id:
            # Get specific execution
            response = await self.client.get(
                f"{self.base_url}/api/v1/executions/{execution_id}",
            )
            response.raise_for_status()

            data = response.json()
            exec_data = data.get("data", {})

            status = exec_data.get("status", "unknown")
            finished = exec_data.get("finished", False)
            mode = exec_data.get("mode", "unknown")

            lines = [
                f"Execution: {execution_id}",
                f"Status: {status}",
                f"Finished: {'Yes' if finished else 'No'}",
                f"Mode: {mode}",
            ]

            # Show result data if available
            result = exec_data.get("data", {})
            if result:
                lines.append(f"\nResult: {json.dumps(result, indent=2)[:500]}")

            return "\n".join(lines)
        else:
            # List executions (optionally filtered by workflow)
            url = f"{self.base_url}/api/v1/executions"
            if workflow_id:
                url += f"?workflowId={workflow_id}"

            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            executions = data.get("data", [])

            if not executions:
                return "No executions found."

            lines = ["Recent Executions:"]
            for exec_data in executions[:10]:
                exec_id = exec_data.get("id", "unknown")
                status = exec_data.get("status", "unknown")
                wf_id = exec_data.get("workflowId", "unknown")
                status_icon = "✅" if status == "success" else "❌" if status == "error" else "⏳"
                lines.append(f"  {status_icon} [{exec_id}] Workflow {wf_id} - {status}")

            return "\n".join(lines)

    async def _design_workflow(
        self,
        design_description: str | None = None,
        **kwargs,
    ) -> str:
        """Design a workflow based on description.

        This generates an n8n workflow JSON structure based on the description.
        """
        if not design_description:
            return "Error: Description required for design action."

        # Common workflow templates
        templates = {
            "webhook": {
                "name": "Webhook Handler",
                "nodes": [
                    {
                        "name": "Webhook",
                        "type": "n8n-nodes-base.webhook",
                        "position": [250, 300],
                        "parameters": {
                            "httpMethod": "POST",
                            "path": "webhook",
                        },
                    },
                    {
                        "name": "Respond",
                        "type": "n8n-nodes-base.respondToWebhook",
                        "position": [500, 300],
                        "parameters": {
                            "respondWith": "json",
                        },
                    },
                ],
                "connections": {
                    "Webhook": {"main": [[{"node": "Respond", "type": "main", "index": 0}]]},
                },
            },
            "schedule": {
                "name": "Scheduled Task",
                "nodes": [
                    {
                        "name": "Schedule Trigger",
                        "type": "n8n-nodes-base.scheduleTrigger",
                        "position": [250, 300],
                        "parameters": {
                            "rule": {"interval": [{"field": "hours", "hoursInterval": 1}]},
                        },
                    },
                    {
                        "name": "HTTP Request",
                        "type": "n8n-nodes-base.httpRequest",
                        "position": [500, 300],
                        "parameters": {
                            "url": "https://api.example.com/endpoint",
                        },
                    },
                ],
                "connections": {
                    "Schedule Trigger": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]},
                },
            },
            "email": {
                "name": "Email Automation",
                "nodes": [
                    {
                        "name": "Email Trigger",
                        "type": "n8n-nodes-base.emailTrigger",
                        "position": [250, 300],
                    },
                    {
                        "name": "Process",
                        "type": "n8n-nodes-base.code",
                        "position": [500, 300],
                    },
                    {
                        "name": "Send Response",
                        "type": "n8n-nodes-base.emailSend",
                        "position": [750, 300],
                    },
                ],
                "connections": {},
            },
        }

        # Check for keyword matches
        desc_lower = design_description.lower()

        if "webhook" in desc_lower:
            template = templates["webhook"]
            return f"Suggested workflow template:\n```json\n{json.dumps(template, indent=2)}\n```\n\nWould you like me to create this workflow?"
        elif "schedule" in desc_lower or "cron" in desc_lower or "timer" in desc_lower:
            template = templates["schedule"]
            return f"Suggested workflow template:\n```json\n{json.dumps(template, indent=2)}\n```\n\nWould you like me to create this workflow?"
        elif "email" in desc_lower:
            template = templates["email"]
            return f"Suggested workflow template:\n```json\n{json.dumps(template, indent=2)}\n```\n\nWould you like me to create this workflow?"
        else:
            return f"""To design a workflow for: "{design_description}"

I can help you create an n8n workflow. Common workflow patterns:

1. **Webhook Handler** - Receive HTTP requests and respond
2. **Scheduled Task** - Run tasks on a schedule
3. **Email Automation** - Process and respond to emails
4. **API Integration** - Connect multiple APIs

Available node types include:
- Triggers: webhook, schedule, email, manual
- Actions: httpRequest, emailSend, code, set
- Logic: if, switch, merge, split

Tell me more about what the workflow should do, and I'll generate the JSON structure."""

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class N8NWorkflowBuilder:
    """Helper class for building n8n workflows."""

    def __init__(self, name: str = "New Workflow"):
        """Initialize the workflow builder.

        Args:
            name: Name of the workflow
        """
        self.workflow = {
            "name": name,
            "nodes": [],
            "connections": {},
            "active": False,
            "settings": {},
            "versionId": "1",
        }
        self._node_positions = {}

    def add_node(
        self,
        name: str,
        node_type: str,
        parameters: dict[str, Any] | None = None,
        position: tuple[int, int] | None = None,
    ) -> "N8NWorkflowBuilder":
        """Add a node to the workflow.

        Args:
            name: Node name
            node_type: n8n node type (e.g., "n8n-nodes-base.webhook")
            parameters: Node parameters
            position: Optional position tuple (x, y)

        Returns:
            Self for chaining
        """
        # Auto-position if not specified
        if position is None:
            x = 250 + len(self.workflow["nodes"]) * 250
            position = (x, 300)

        node = {
            "name": name,
            "type": node_type,
            "position": list(position),
            "parameters": parameters or {},
        }

        self.workflow["nodes"].append(node)
        self._node_positions[name] = position

        return self

    def connect(
        self,
        source: str,
        target: str,
        source_output: int = 0,
        target_input: int = 0,
    ) -> "N8NWorkflowBuilder":
        """Connect two nodes.

        Args:
            source: Source node name
            target: Target node name
            source_output: Output index from source
            target_input: Input index on target

        Returns:
            Self for chaining
        """
        if source not in self.workflow["connections"]:
            self.workflow["connections"][source] = {"main": [[]]}

        connection = {
            "node": target,
            "type": "main",
            "index": target_input,
        }

        # Ensure we have enough output slots
        while len(self.workflow["connections"][source]["main"]) <= source_output:
            self.workflow["connections"][source]["main"].append([])

        self.workflow["connections"][source]["main"][source_output].append(connection)

        return self

    def add_webhook_trigger(
        self,
        path: str = "webhook",
        method: str = "POST",
        name: str = "Webhook",
    ) -> "N8NWorkflowBuilder":
        """Add a webhook trigger node.

        Args:
            path: Webhook path
            method: HTTP method
            name: Node name

        Returns:
            Self for chaining
        """
        return self.add_node(
            name=name,
            node_type="n8n-nodes-base.webhook",
            parameters={
                "httpMethod": method,
                "path": path,
            },
            position=(250, 300),
        )

    def add_http_request(
        self,
        url: str,
        method: str = "GET",
        name: str = "HTTP Request",
    ) -> "N8NWorkflowBuilder":
        """Add an HTTP request node.

        Args:
            url: Request URL
            method: HTTP method
            name: Node name

        Returns:
            Self for chaining
        """
        return self.add_node(
            name=name,
            node_type="n8n-nodes-base.httpRequest",
            parameters={
                "url": url,
                "method": method,
            },
        )

    def add_code_node(
        self,
        code: str,
        name: str = "Code",
    ) -> "N8NWorkflowBuilder":
        """Add a code node.

        Args:
            code: JavaScript code to execute
            name: Node name

        Returns:
            Self for chaining
        """
        return self.add_node(
            name=name,
            node_type="n8n-nodes-base.code",
            parameters={
                "jsCode": code,
            },
        )

    def add_set_node(
        self,
        values: dict[str, Any],
        name: str = "Set Values",
    ) -> "N8NWorkflowBuilder":
        """Add a set node to transform data.

        Args:
            values: Key-value pairs to set
            name: Node name

        Returns:
            Self for chaining
        """
        return self.add_node(
            name=name,
            node_type="n8n-nodes-base.set",
            parameters={
                "values": {
                    "string": [
                        {"name": k, "value": str(v)}
                        for k, v in values.items()
                    ],
                },
            },
        )

    def add_if_node(
        self,
        condition: dict[str, Any],
        name: str = "If",
    ) -> "N8NWorkflowBuilder":
        """Add an if condition node.

        Args:
            condition: Condition configuration
            name: Node name

        Returns:
            Self for chaining
        """
        return self.add_node(
            name=name,
            node_type="n8n-nodes-base.if",
            parameters={"conditions": condition},
        )

    def build(self) -> dict[str, Any]:
        """Build and return the workflow JSON.

        Returns:
            The workflow dictionary
        """
        return self.workflow.copy()

    def to_json(self) -> str:
        """Build and return the workflow as JSON string.

        Returns:
            JSON string of the workflow
        """
        return json.dumps(self.build(), indent=2)


# Common workflow templates
WORKFLOW_TEMPLATES = {
    "api_poller": {
        "name": "API Poller",
        "description": "Poll an API endpoint on a schedule and process results",
        "nodes": ["ScheduleTrigger", "HTTPRequest", "Code"],
    },
    "webhook_to_slack": {
        "name": "Webhook to Slack",
        "description": "Receive webhook and send notification to Slack",
        "nodes": ["Webhook", "Slack"],
    },
    "email_processor": {
        "name": "Email Processor",
        "description": "Process incoming emails and take actions",
        "nodes": ["EmailTrigger", "Code", "EmailSend"],
    },
}
