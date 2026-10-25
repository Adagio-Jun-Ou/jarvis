"""n8n workflows for Jarvis self-improvement and exploration.

This module provides workflow templates and automation for:
- Daydream-triggered self-improvement tasks
- Erdős math problem exploration automation
- Tool practice scheduling
- Insight-to-evolution pipeline
- Knowledge consolidation workflows

The workflows are designed to run autonomously when Jarvis is idle.
"""
import logging
from core.helpers import safe_error_message
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

WORKFLOWS_DIR = Path.home() / ".jarvis" / "n8n_workflows"


# Workflow templates for self-improvement
DAYDREAM_WORKFLOW_TEMPLATES = {
    "self_improvement": {
        "name": "Jarvis Self-Improvement Pipeline",
        "description": "Runs during idle time to analyze and improve code",
        "trigger": "manual",
        "nodes": [
            {
                "name": "Get Pending Tasks",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/self-refine/pending",
                    "method": "GET",
                },
            },
            {
                "name": "Analyze Task Priority",
                "type": "n8n-nodes-base.code",
                "parameters": {
                    "jsCode": """
// Prioritize tasks by impact and risk
const tasks = $input.all();
const prioritized = tasks.sort((a, b) => {
    const impactScore = { high: 3, medium: 2, low: 1 };
    const riskScore = { safe: 1, moderate: 2, risky: 3 };
    return (impactScore[b.impact] - impactScore[a.impact]) - 
           (riskScore[b.risk] - riskScore[a.risk]);
});
return prioritized.slice(0, 3);
"""
                },
            },
            {
                "name": "Execute Improvement",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/self-refine/apply",
                    "method": "POST",
                },
            },
            {
                "name": "Log Result",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/memory/store",
                    "method": "POST",
                },
            },
        ],
    },
    
    "math_exploration": {
        "name": "Erdős Math Problem Explorer",
        "description": "Automated math problem exploration workflow",
        "trigger": "schedule",
        "schedule": {"hour": 2},  # Run at 2 AM during idle
        "nodes": [
            {
                "name": "Get Problem",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/erdos/random_problem",
                    "method": "GET",
                },
            },
            {
                "name": "Generate Approaches",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/erdos/approaches",
                    "method": "POST",
                },
            },
            {
                "name": "Try Solution",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/erdos/solve",
                    "method": "POST",
                },
            },
            {
                "name": "Store Insight",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/insight/store",
                    "method": "POST",
                },
            },
        ],
    },
    
    "tool_practice": {
        "name": "Tool Practice Scheduler",
        "description": "Practice and improve tool usage during idle",
        "trigger": "schedule",
        "schedule": {"minute": 30},  # Every 30 minutes
        "nodes": [
            {
                "name": "Get Practice Priorities",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/tools/practice_priorities",
                    "method": "GET",
                },
            },
            {
                "name": "Select Tool",
                "type": "n8n-nodes-base.code",
                "parameters": {
                    "jsCode": """
const tools = $input.all();
const practiceCandidates = tools.filter(t => t.practice_score < 0.8);
return practiceCandidates.slice(0, 2);
"""
                },
            },
            {
                "name": "Practice Tool",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/tools/practice",
                    "method": "POST",
                },
            },
        ],
    },
    
    "insight_pipeline": {
        "name": "Insight to Evolution Pipeline",
        "description": "Convert insights to evolution proposals",
        "trigger": "webhook",
        "nodes": [
            {
                "name": "Webhook",
                "type": "n8n-nodes-base.webhook",
                "parameters": {"path": "insight"},
            },
            {
                "name": "Filter High Value",
                "type": "n8n-nodes-base.code",
                "parameters": {
                    "jsCode": """
const insights = $input.all();
return insights.filter(i => i.confidence > 0.7 && i.evolution_potential > 0.5);
"""
                },
            },
            {
                "name": "Create Evolution Proposal",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/evolution/propose",
                    "method": "POST",
                },
            },
            {
                "name": "Notify if Approved",
                "type": "n8n-nodes-base.if",
                "parameters": {},
            },
        ],
    },
    
    "knowledge_consolidation": {
        "name": "Knowledge Consolidation",
        "description": "Consolidate learned knowledge into memory",
        "trigger": "schedule",
        "schedule": {"hour": 4},  # Run at 4 AM
        "nodes": [
            {
                "name": "Export Tool Learning",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/tools/export_memory",
                    "method": "GET",
                },
            },
            {
                "name": "Export Insights",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/insights/export",
                    "method": "GET",
                },
            },
            {
                "name": "Merge and Dedupe",
                "type": "n8n-nodes-base.code",
                "parameters": {
                    "jsCode": """
const toolData = $input.first().json;
const insightData = $input.last().json;

// Merge and deduplicate
const merged = {
    tools: toolData.tools || [],
    insights: insightData.insights || [],
    timestamp: new Date().toISOString(),
    total_memories: (toolData.tools?.length || 0) + (insightData.insights?.length || 0)
};

return [{ json: merged }];
"""
                },
            },
            {
                "name": "Store Consolidated",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/memory/bulk_store",
                    "method": "POST",
                },
            },
        ],
    },
    
    "daydream_orchestrator": {
        "name": "Daydream Activity Orchestrator",
        "description": "Coordinates all daydream activities based on idle time",
        "trigger": "schedule",
        "schedule": {"minute": 15},  # Every 15 minutes
        "nodes": [
            {
                "name": "Check Idle Status",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/status/idle",
                    "method": "GET",
                },
            },
            {
                "name": "Decide Activity",
                "type": "n8n-nodes-base.code",
                "parameters": {
                    "jsCode": """
const status = $input.first().json;

if (!status.is_idle) {
    return [{ json: { activity: "skip", reason: "not_idle" } }];
}

const activities = ["math", "tools", "insights", "consolidate"];
const randomActivity = activities[Math.floor(Math.random() * activities.length)];

return [{ json: { 
    activity: randomActivity, 
    idle_time: status.idle_seconds,
    queue_size: status.queue_size
} }];
"""
                },
            },
            {
                "name": "Trigger Math Exploration",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/erdos/explore",
                    "method": "POST",
                },
            },
            {
                "name": "Trigger Tool Practice",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/tools/practice",
                    "method": "POST",
                },
            },
            {
                "name": "Trigger Insight Generation",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8080/api/insights/generate",
                    "method": "POST",
                },
            },
        ],
    },
}


class DaydreamN8NIntegration:
    """Integrates n8n workflows with Jarvis daydream subsystems."""
    
    def __init__(self, n8n_tool=None):
        self.n8n = n8n_tool
        self._workflows_created = False
        self._workflow_ids: dict[str, str] = {}
        self._last_activity: datetime | None = None
        self._activities_completed = 0
        
    async def ensure_workflows(self) -> dict[str, str]:
        """Ensure all daydream workflows are created in n8n.
        
        Returns:
            Dict mapping workflow names to their IDs
        """
        if self._workflows_created:
            return self._workflow_ids
            
        if not self.n8n:
            logger.warning("n8n tool not available, skipping workflow creation")
            return {}
        
        # Check n8n availability
        try:
            available = await self.n8n._check_n8n()
            if not available:
                logger.debug("n8n not available, workflows not created")
                return {}
        except Exception as e:
            logger.debug(f"n8n check failed: {e}")
            return {}
        
        # Create each workflow
        for name, template in DAYDREAM_WORKFLOW_TEMPLATES.items():
            try:
                workflow_data = self._build_workflow_json(template)
                result = await self.n8n.execute(
                    action="create",
                    workflow_data=workflow_data,
                )
                
                # Extract workflow ID from result
                if "ID:" in result:
                    wf_id = result.split("ID:")[1].split(")")[0].strip()
                    self._workflow_ids[name] = wf_id
                    logger.info(f"Created n8n workflow: {name} (ID: {wf_id})")
                    
            except Exception as e:
                logger.warning(f"Failed to create workflow {name}: {e}")
        
        self._workflows_created = True
        return self._workflow_ids
    
    def _build_workflow_json(self, template: dict) -> dict:
        """Build n8n workflow JSON from template."""
        nodes = []
        connections = {}
        
        for i, node in enumerate(template.get("nodes", [])):
            nodes.append({
                "id": str(i + 1),
                "name": node["name"],
                "type": node["type"],
                "typeVersion": 1,
                "position": [250 + i * 250, 300],
                "parameters": node.get("parameters", {}),
            })
            
            # Connect to previous node
            if i > 0:
                prev_name = template["nodes"][i - 1]["name"]
                if prev_name not in connections:
                    connections[prev_name] = {"main": [[]]}
                connections[prev_name]["main"][0].append({
                    "node": node["name"],
                    "type": "main",
                    "index": 0,
                })
        
        workflow = {
            "name": template["name"],
            "nodes": nodes,
            "connections": connections,
            "active": False,
            "settings": {
                "executionOrder": "v1",
            },
            "tags": [
                {"name": "jarvis", "createdAt": datetime.now().isoformat()},
                {"name": "daydream", "createdAt": datetime.now().isoformat()},
            ],
        }
        
        # Add trigger for scheduled workflows
        if template.get("trigger") == "schedule" and "schedule" in template:
            trigger_node = {
                "id": "0",
                "name": "Schedule Trigger",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1,
                "position": [0, 300],
                "parameters": {
                    "rule": {"interval": [template["schedule"]]},
                },
            }
            nodes.insert(0, trigger_node)
            
            # Connect trigger to first node
            first_node_name = template["nodes"][0]["name"]
            connections[trigger_node["name"]] = {
                "main": [[{"node": first_node_name, "type": "main", "index": 0}]]
            }
        
        return workflow
    
    async def trigger_daydream_activity(self, activity_type: str) -> dict:
        """Trigger a specific daydream activity via n8n workflow.
        
        Args:
            activity_type: Type of activity (math, tools, insights, consolidate)
            
        Returns:
            Result dict with activity status
        """
        activity_map = {
            "math": "math_exploration",
            "tools": "tool_practice", 
            "insights": "insight_pipeline",
            "consolidate": "knowledge_consolidation",
            "orchestrate": "daydream_orchestrator",
            "self_improve": "self_improvement",
        }
        
        workflow_name = activity_map.get(activity_type)
        if not workflow_name:
            return {"success": False, "error": f"Unknown activity: {activity_type}"}
        
        # Ensure workflows exist
        workflow_ids = await self.ensure_workflows()
        
        workflow_id = workflow_ids.get(workflow_name)
        if not workflow_id:
            return {"success": False, "error": f"Workflow not found: {workflow_name}"}
        
        try:
            # Execute the workflow
            result = await self.n8n.execute(
                action="execute",
                workflow_id=workflow_id,
            )
            
            self._last_activity = datetime.now()
            self._activities_completed += 1
            
            return {
                "success": True,
                "activity": activity_type,
                "workflow": workflow_name,
                "workflow_id": workflow_id,
                "result": result,
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger {activity_type}: {e}")
            return {"success": False, "error": safe_error_message(e)}
    
    async def get_daydream_report(self) -> dict:
        """Get report of daydream n8n activities."""
        return {
            "workflows_created": self._workflows_created,
            "workflow_ids": self._workflow_ids,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            "activities_completed": self._activities_completed,
        }
    
    async def activate_daydream_workflows(self) -> dict:
        """Activate all daydream workflows for autonomous operation."""
        workflow_ids = await self.ensure_workflows()
        results = {}
        
        for name, wf_id in workflow_ids.items():
            try:
                await self.n8n.execute(
                    action="activate",
                    workflow_id=wf_id,
                )
                results[name] = "activated"
                logger.info(f"Activated n8n workflow: {name}")
            except Exception as e:
                results[name] = f"error: {e}"
                logger.warning(f"Failed to activate {name}: {e}")
        
        return results
    
    async def deactivate_daydream_workflows(self) -> dict:
        """Deactivate all daydream workflows."""
        results = {}
        
        for name, wf_id in self._workflow_ids.items():
            try:
                await self.n8n.execute(
                    action="deactivate",
                    workflow_id=wf_id,
                )
                results[name] = "deactivated"
            except Exception as e:
                results[name] = f"error: {e}"
        
        return results


# Singleton instance
_daydream_n8n: DaydreamN8NIntegration | None = None


def get_daydream_n8n(n8n_tool=None) -> DaydreamN8NIntegration:
    """Get or create the daydream n8n integration."""
    global _daydream_n8n
    if _daydream_n8n is None:
        _daydream_n8n = DaydreamN8NIntegration(n8n_tool)
    elif n8n_tool is not None:
        _daydream_n8n.n8n = n8n_tool
    return _daydream_n8n