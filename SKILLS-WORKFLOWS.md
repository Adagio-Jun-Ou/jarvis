# Jarvis Composable Workflows - Complete Guide

This document describes Jarvis's native workflow capabilities and how to use them for agentic deployments.

## Available Workflow Systems

Jarvis has **four integrated workflow systems**:

| System | Purpose | Location |
|--------|---------|----------|
| **Workflow Shell** | Composable pipelines (pipe, parallel, sequence) | `core/workflow_shell.py` |
| **Workflow Scheduler** | Cron-like scheduling, recurring tasks | `core/workflow_scheduler.py` |
| **Agentic Orchestrator** | Long-running autonomous deployments | `core/agentic_orchestrator.py` |
| **n8n Integration** | Visual workflow automation (external) | `tools/n8n.py` + `core/n8n_manager.py` |

## 1. Workflow Shell - Composable Pipelines

The workflow shell provides a **Lobster-style composable pipeline** syntax:

```python
from core.workflow_shell import Pipeline, Stage

# Create a pipeline with pipe notation
pipeline = (
    Pipeline("search_and_analyze")
    | WebSearchTool()      # Sequential: feeds output to next
    | ExtractResults()
    || SummarizeTool()     # Parallel: runs alongside previous
    | CombineResults()
    >> SaveToFile()        # After all complete
)

result = await pipeline.execute("quantum computing advances")
```

### Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `\|` | Pipe - sequential execution | `Stage1() \| Stage2()` |
| `\|\|` | Parallel - run simultaneously | `Stage1() \|\| Stage2()` |
| `>>` | After - wait for all previous | `Stages... >> FinalStage()` |

### Creating Custom Stages

```python
class AnalyzeData(Stage[dict, dict]):
    name = "analyze_data"
    description = "Analyze data and extract insights"
    requires = ["data"]
    provides = ["insights", "metrics"]
    
    async def execute(self, input_data: dict, context: PipelineContext) -> StageResult[dict]:
        # Access previous stage results
        previous = context.get_stage_result("previous_stage")
        
        # Process data
        insights = await analyze(input_data)
        
        return StageResult(
            value={"insights": insights},
            status=StageStatus.COMPLETED
        )
```

### Error Handling

```python
pipeline = (
    Pipeline("robust_search")
    | WebSearchTool()
    | ProcessResults()
    .fallback(lambda e: f"Error: {e}")  # Fallback on failure
    .retry(max_attempts=3)                # Retry up to 3 times
    .timeout(seconds=30)                  # Timeout after 30s
)
```

## 2. Workflow Scheduler - Scheduled Tasks

The scheduler provides **cron-like scheduling** for recurring tasks:

```python
from core.workflow_scheduler import get_workflow_scheduler

scheduler = get_workflow_scheduler()

# Schedule daily report at 9 AM
schedule_id = scheduler.schedule(
    name="morning_report",
    workflow="generate_report",
    cron="0 9 * * *",  # 9 AM every day
    params={"report_type": "summary"},
)

# Schedule with interval (every 5 minutes)
scheduler.schedule(
    name="health_check",
    workflow="system_check",
    interval_seconds=300,
)

# One-time delayed execution
scheduler.schedule(
    name="reminder",
    workflow="send_notification",
    run_at=datetime.now() + timedelta(hours=1),
)

# Pause/resume schedules
scheduler.pause(schedule_id)
scheduler.resume(schedule_id)
scheduler.disable(schedule_id)
```

### Cron Expressions

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
* * * * *
```

Examples:
- `0 9 * * *` - Every day at 9 AM
- `*/15 * * * *` - Every 15 minutes
- `0 9 * * 1-5` - Weekdays at 9 AM
- `0 0 1 * *` - First day of month at midnight

## 3. Agentic Orchestrator - Long-Running Deployments

The orchestrator handles **long-running autonomous operations** with self-recovery:

```python
from core.agentic_orchestrator import get_agentic_orchestrator

orchestrator = get_agentic_orchestrator()

# Deploy a long-running workflow
deployment = orchestrator.deploy(
    name="daily_analysis",
    workflow="analyze_repository",
    schedule="0 6 * * *",  # 6 AM daily
    goals=["improve code quality", "reduce technical debt"],
    auto_recover=True,
    max_retries=3,
)

# Monitor deployment
status = orchestrator.get_deployment_status(deployment.deployment_id)

# Get all deployments
deployments = orchestrator.list_deployments()

# Stop/restart
orchestrator.stop_deployment(deployment.deployment_id)
orchestrator.start_deployment(deployment.deployment_id)
```

### Deployment Types

| Type | Description | Use Case |
|------|-------------|----------|
| `WORKFLOW` | Single workflow execution | One-time task |
| `GOAL` | Goal-driven execution | Autonomous improvement |
| `SCHEDULED` | Scheduled recurring task | Daily reports |
| `CONTINUOUS` | Always-running process | Real-time monitoring |

### Health Monitoring

```python
health = orchestrator.get_health()

# Check system health
if not health.healthy:
    for component, status in health.components.items():
        if not status.get('healthy'):
            logger.warning(f"Component {component} unhealthy: {status}")
```

## 4. n8n Integration - Visual Workflow Automation

n8n provides **visual workflow automation** with a web UI:

### Available Actions

```python
# List all workflows
n8n: {"action": "list"}

# Get workflow details
n8n: {"action": "get", "workflow_id": "123"}

# Create workflow
n8n: {"action": "create", "workflow_data": {...}}

# Execute workflow
n8n: {"action": "execute", "workflow_id": "123"}

# Activate/deactivate
n8n: {"action": "activate", "workflow_id": "123"}
n8n: {"action": "deactivate", "workflow_id": "123"}

# Check execution status
n8n: {"action": "executions", "execution_id": "456"}

# Design workflow from description
n8n: {"action": "design", "design_description": "Send email when webhook received"}
```

### n8n Workflow Builder

Create workflows programmatically:

```python
from tools.n8n import N8NWorkflowBuilder

# Build a workflow
workflow = (
    N8NWorkflowBuilder("My Workflow")
    .add_webhook_trigger(path="my-webhook", method="POST")
    .add_http_request(url="https://api.example.com/process")
    .add_code_node(code="return items.map(item => ({...item, processed: true}));")
    .connect("Webhook", "HTTP Request")
    .connect("HTTP Request", "Code")
    .build()
)

# Create the workflow
n8n_tool.execute(action="create", workflow_data=workflow)
```

### Workflow Templates

Available templates:
- **API Poller** - Poll an endpoint on schedule
- **Webhook to Slack** - Receive webhook, send to Slack
- **Email Processor** - Process emails and respond
- **Schedule Trigger** - Run tasks on schedule

### Jarvis + n8n Integration

n8n is **auto-started** with Jarvis (unless `--no-n8n` flag):

```bash
# Start Jarvis with n8n
./venv/bin/python run.py --interface web

# Access n8n UI at: http://localhost:5678
# Jarvis n8n tool connects to: http://localhost:5678/api/v1/
```

## Combining All Workflow Systems

### Example: Multi-Stage Agentic Pipeline

```python
# 1. Define the workflow shell pipeline
from core.workflow_shell import Pipeline, Stage

analysis_pipeline = (
    Pipeline("repository_analysis")
    | ReadRepository()
    | AnalyzeCode(parallel=True) || AnalyzeTests(parallel=True)
    | CombineResults()
    | GenerateReport()
    >> SaveReport()
)

# 2. Create n8n workflow for notifications
n8n_workflow = (
    N8NWorkflowBuilder("Report Notification")
    .add_webhook_trigger(path="report-complete")
    .add_http_request(url="https://slack.com/api/chat.postMessage", method="POST")
    .build()
)

# 3. Deploy as agentic orchestration
orchestrator = get_agentic_orchestrator()
deployment = orchestrator.deploy(
    name="daily_analysis",
    workflow="repository_analysis",
    schedule="0 6 * * *",
    auto_recover=True,
    goals=["identify code quality issues", "generate actionable recommendations"],
    post_execute=lambda: scheduler.schedule(
        name="notify_complete",
        workflow="report_notification",
        run_at=datetime.now() + timedelta(minutes=5)
    )
)

# 4. Start monitoring
orchestrator.start()
```

## Available n8n Templates

| Template | Nodes | Purpose |
|----------|-------|---------|
| `api_poller` | ScheduleTrigger, HTTPRequest, Code | Poll API and process |
| `webhook_to_slack` | Webhook, Slack | Webhook → Slack notification |
| `email_processor` | EmailTrigger, Code, EmailSend | Email automation |

## Common Patterns

### Pattern 1: Scheduled Data Processing

```python
# Schedule daily data processing
scheduler.schedule(
    name="daily_data_process",
    workflow="process_data",
    cron="0 2 * * *",  # 2 AM daily
    params={"source": "/data/incoming", "dest": "/data/processed"},
)
```

### Pattern 2: Event-Driven Workflow

```python
# Webhook triggers processing
pipeline = (
    Pipeline("event_processor")
    | ValidateInput()
    | ProcessEvent()
    | StoreResults()
    | NotifyComplete()
)

# n8n receives webhook, calls pipeline
n8n: {
    "action": "create",
    "workflow_data": {
        "name": "Webhook Handler",
        "nodes": [
            {"type": "webhook", "path": "process"},
            {"type": "httpRequest", "url": "http://localhost:8443/api/invoke"},
        ]
    }
}
```

### Pattern 3: Self-Improving Goal

```python
# Deploy autonomous goal
orchestrator.deploy(
    name="code_quality_improver",
    deployment_type=DeploymentType.GOAL,
    goals=["reduce technical debt", "improve test coverage"],
    config={
        "scan_paths": ["core/", "tools/"],
        "max_changes_per_run": 5,
        "approval_required": True,
    },
)
```

## n8n Tool Registration

n8n is registered as a tool and available in all agentic workflows:

```python
# In run.py
n8n_tool = N8NTool(
    base_url=config.automation.n8n.url,
    api_key=n8n_api_key,
)
agent.register_tools([n8n_tool])
```

## Permissions

n8n tool has `requires_permission = False` for autonomous use. Sensitive operations (create, delete, execute) don't require explicit approval since n8n itself can enforce permissions.

## Summary

| Use Case | Recommended System |
|----------|-------------------|
| Quick in-code pipelines | Workflow Shell |
| Scheduled/recurring tasks | Workflow Scheduler |
| Long-running autonomous operations | Agentic Orchestrator |
| Visual workflow design | n8n |
| Event-driven processing | n8n + Workflow Shell |
| Multi-step with persistence | Agentic Orchestrator |

All systems integrate seamlessly - use them together for complex automation!