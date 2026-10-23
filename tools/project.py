"""Agentic Project Tool for Jarvis.

Provides tools for Jarvis to work agentically on software projects:
- project_init: Initialize a project workspace
- project_status: Get project status and next steps
- project_decompose: Break down a goal into tasks
- project_execute: Execute a task or plan
- project_file_read: Read project files with context
- project_file_write: Write files with validation
- project_test: Run project tests
- project_commit: Commit changes with message

Example usage in Jarvis:
    User: "Work on improving the simulation accuracy in Research Project"
    
    Jarvis:
    1. project_init("/home/user/Research Project")
    2. project_status() → shows epics, stories, tasks
    3. project_decompose("Improve simulation accuracy") → creates tasks
    4. project_execute(task_id) → executes the task
    5. project_test() → runs tests
    6. project_commit("Improved simulation accuracy") → commits changes
"""
import asyncio
import logging
from pathlib import Path
from typing import Any

from core.helpers import safe_error_message
from tools.base import Tool

logger = logging.getLogger(__name__)


class ProjectInitTool(Tool):
    """Initialize or load a project workspace."""
    
    name = "project_init"
    description = "Initialize a project workspace for agentic work. Loads existing project state or creates a new workspace."
    parameters = {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project directory"
            },
            "analyze": {
                "type": "boolean",
                "description": "Whether to analyze project structure",
                "default": True
            }
        },
        "required": ["project_path"]
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace
            
            project_path = Path(params["project_path"]).expanduser().resolve()
            analyze = params.get("analyze", True)
            
            if not project_path.exists():
                return f"Error: Project path does not exist: {project_path}"
            
            workspace = get_workspace(project_path)
            
            if analyze:
                status = workspace.get_status()
                
                return {
                    "project_name": workspace.project_name,
                    "project_path": str(workspace.project_path),
                    "language": workspace._metadata.get("language"),
                    "frameworks": workspace._metadata.get("frameworks", []),
                    "entry_points": workspace._metadata.get("entry_points", []),
                    "total_epics": status["total_epics"],
                    "total_tasks": status["total_tasks"],
                    "completion_percentage": status["completion_percentage"],
                    "architecture_nodes": len(workspace._architecture),
                    "last_activity": status.get("last_activity"),
                }
            
            return f"Initialized project: {workspace.project_name}"
            
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project initialization failed')}"


class ProjectStatusTool(Tool):
    """Get the current project status and suggested next steps."""
    
    name = "project_status"
    description = "Get the current status of the project workspace including epics, stories, tasks, and progress. Shows suggested next steps."
    parameters = {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional if already initialized)"
            },
            "detailed": {
                "type": "boolean",
                "description": "Include detailed task breakdown",
                "default": False
            }
        },
        "required": []
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances, TaskStatus
            
            project_path = params.get("project_path")
            detailed = params.get("detailed", False)
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            status = workspace.get_status()
            suggestions = workspace.suggest_next_steps()
            
            output_lines = [
                f"Project: {workspace.project_name}",
                f"Path: {workspace.project_path}",
                f"Language: {workspace._metadata.get('language', 'Unknown')}",
                f"Frameworks: {', '.join(workspace._metadata.get('frameworks', [])) or 'None'}",
                "",
                f"Progress: {status['completed_tasks']}/{status['total_tasks']} tasks ({status['completion_percentage']:.1f}%)",
                f"Epics: {status['total_epics']}, Goals: {status['total_goals']}",
                f"In Progress: {status['in_progress_tasks']}, Blocked: {status['blocked_tasks']}",
                "",
                "Epics:",
            ]
            
            for epic_id, epic in workspace._epics.items():
                completion = epic.completion_percentage()
                tasks = len(epic.get_all_tasks())
                output_lines.append(f"  - {epic.title}: {completion:.0f}% ({tasks} tasks)")
            
            if suggestions:
                output_lines.append("")
                output_lines.append("Suggested Next Steps:")
                for i, suggestion in enumerate(suggestions[:5], 1):
                    output_lines.append(f"  {i}. [{suggestion['priority']}] {suggestion['description']}")
            
            if detailed:
                output_lines.append("")
                output_lines.append("Tasks by Status:")
                
                for epic in workspace._epics.values():
                    for story in epic.stories:
                        for task in story.tasks:
                            if task.status != TaskStatus.DONE:
                                status_icon = {"todo": "○", "in_progress": "◐", "blocked": "✕", "review": "◉"}.get(task.status.value, "?")
                                output_lines.append(f"  {status_icon} {task.id}: {task.title} [{task.status.value}]")
            
            return {
                "status": status,
                "suggestions": suggestions,
                "summary": workspace.to_summary(),
            }
            
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"


class ProjectDecomposeTool(Tool):
    """Decompose a goal into structured tasks using LLM."""
    
    name = "project_decompose"
    description = "Break down a high-level goal into stories and tasks. Uses LLM to analyze goal and create structured work items with estimates and dependencies."
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "High-level goal to decompose"
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional if already initialized)"
            },
            "depth": {
                "type": "integer",
                "description": "Decomposition depth: 1=epic only, 2=epic+stories, 3=epic+stories+tasks",
                "default": 3
            },
            "context": {
                "type": "object",
                "description": "Additional context for decomposition (files, constraints, preferences)"
            }
        },
        "required": ["goal"]
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances
            from core.task_decomposition import get_task_decomposition
            from core.llm import get_llm_client
            
            goal = params["goal"]
            project_path = params.get("project_path")
            depth = params.get("depth", 3)
            context = params.get("context", {})
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            
            llm = get_llm_client()
            decomposition = get_task_decomposition(llm_client=llm)
            
            from core.task_decomposition import DecompositionContext
            
            decomp_context = DecompositionContext(
                project_name=workspace.project_name,
                project_path=str(workspace.project_path),
                existing_epics=list(workspace._epics.keys()),
                architecture_overview="\n".join(
                    f"- {node.name}: {node.description}"
                    for node in workspace._architecture.values()[:50]
                ),
                relevant_files=list(workspace._architecture.keys())[:30],
                constraints=context.get("constraints", []),
                preferences=context.get("preferences", []),
            )
            
            result = await decomposition.decompose(goal, decomp_context, depth)
            
            if not result.success:
                return f"Error: Decomposition failed: {result.analysis}"
            
            self._store_result(params, result)
            
            output_lines = [
                f"Decomposed: {goal}",
                f"Analysis: {result.analysis}",
                "",
                f"Epic: {result.epic.title if result.epic else 'N/A'}",
            ]
            
            if result.epic:
                for story in result.epic.stories:
                    output_lines.append(f"  Story: {story.title}")
                    for task in story.tasks:
                        output_lines.append(
                            f"    - Task: {task.title} ({task.estimated_minutes}min, {task.priority.name})"
                        )
            
            output_lines.append(f"\nTotal estimated: {result.total_estimated_minutes} minutes")
            
            if result.risks:
                output_lines.append("\nRisks:")
                for risk in result.risks:
                    output_lines.append(f"  - {risk}")
            
            if result.recommendations:
                output_lines.append("\nRecommendations:")
                for rec in result.recommendations:
                    output_lines.append(f"  - {rec}")
            
            return {
                "epic": result.epic.title if result.epic else None,
                "stories": len(result.epic.stories) if result.epic else 0,
                "tasks": sum(len(s.tasks) for s in result.epic.stories) if result.epic else 0,
                "total_minutes": result.total_estimated_minutes,
                "risks": result.risks,
                "recommendations": result.recommendations,
            }
            
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"
    
    def _store_result(self, args: dict, result: Any) -> None:
        """Store decomposition result for later retrieval."""
        pass


class ProjectExecuteTool(Tool):
    """Execute a task or goal in the project."""
    
    name = "project_execute"
    description = "Execute a specific task or work towards a goal. Can execute single tasks or autonomously work on multi-step goals."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Specific task ID to execute (optional)"
            },
            "goal": {
                "type": "string",
                "description": "Goal to work towards (creates tasks if needed)"
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional if already initialized)"
            },
            "auto_commit": {
                "type": "boolean",
                "description": "Automatically commit changes (default: false)",
                "default": False
            },
            "max_iterations": {
                "type": "integer",
                "description": "Maximum execution iterations",
                "default": 10
            }
        },
        "required": []
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances
            from core.execution_planner import get_execution_planner
            from core.llm import get_llm_client
            
            task_id = params.get("task_id")
            goal = params.get("goal")
            project_path = params.get("project_path")
            params.get("auto_commit", False)
            params.get("max_iterations", 10)
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            
            if task_id:
                task = workspace.get_task(task_id)
                if not task:
                    return f"Error: Task not found: {task_id}"
                
                workspace.start_task(task_id)
                
                return f"Started task: {task.title}\nDescription: {task.description}"
            
            if goal:
                llm = get_llm_client()
                planner = get_execution_planner(llm_client=llm)
                
                plan = await planner.create_plan(goal, workspace)
                
                if plan.state.value == "failed":
                    return f"Error: Failed to create plan: {plan.metadata.get('error', 'Unknown error')}"
                
                output_lines = [
                    f"Created execution plan for: {goal}",
                    f"Plan ID: {plan.id}",
                    f"Steps: {len(plan.steps)}",
                    f"Estimated: {plan.total_estimated_minutes} minutes",
                    "",
                    "Execution steps:",
                ]
                
                for step in plan.steps:
                    output_lines.append(f"  {step.order + 1}. {step.task_id}")
                
                return {
                    "plan_id": plan.id,
                    "steps": len(plan.steps),
                    "total_minutes": plan.total_estimated_minutes,
                    "steps_list": [s.to_dict() for s in plan.steps],
                }
            
            next_task = workspace.get_next_task()
            if next_task:
                return f"Next task: {next_task.title}\nDescription: {next_task.description}\nPriority: {next_task.priority.name}\nEstimated: {next_task.estimated_minutes} minutes"
            
            return "No pending tasks. Use project_decompose to create new tasks from a goal."
            
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"


class ProjectFileReadTool(Tool):
    """Read a file from the project with context."""
    
    name = "project_file_read"
    description = "Read a file from the project workspace. Shows file content and relationships to other files."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to project root"
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional)"
            },
            "show_dependencies": {
                "type": "boolean",
                "description": "Show file dependencies",
                "default": True
            },
            "start_line": {
                "type": "integer",
                "description": "Start reading from line",
                "default": 0
            },
            "end_line": {
                "type": "integer",
                "description": "End reading at line (0 for full file)",
                "default": 0
            }
        },
        "required": ["file_path"]
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances
            
            file_path = params["file_path"]
            project_path = params.get("project_path")
            params.get("show_dependencies", True)
            start_line = params.get("start_line", 0)
            end_line = params.get("end_line", 0)
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            full_path = (workspace.project_path / file_path).resolve()
            
            if not str(full_path).startswith(str(workspace.project_path.resolve())):
                return "Error: Path traversal blocked - file must be within project directory"
            
            if not full_path.exists():
                return f"Error: File not found: {file_path}"
            
            content = full_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            
            if start_line > 0 or end_line > 0:
                end_line = end_line if end_line > 0 else len(lines)
                lines = lines[start_line:end_line]
                content = "\n".join(lines)
            
            return {
                "file_path": file_path,
                "content": content,
                "lines": len(lines),
                "dependencies": workspace.get_dependencies(file_path),
                "dependents": workspace.get_dependents(file_path),
            }
            
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"


class ProjectFileWriteTool(Tool):
    """Write a file to the project with validation."""
    
    name = "project_file_write"
    description = "Write or modify a file in the project workspace. Creates directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to project root"
            },
            "content": {
                "type": "string",
                "description": "File content to write"
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional)"
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Create directories if needed",
                "default": True
            },
            "task_id": {
                "type": "string",
                "description": "Task ID this write is for (optional)"
            }
        },
        "required": ["file_path", "content"]
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances
            
            file_path = params["file_path"]
            content = params["content"]
            project_path = params.get("project_path")
            create_dirs = params.get("create_dirs", True)
            task_id = params.get("task_id")
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            full_path = (workspace.project_path / file_path).resolve()
            
            if not str(full_path).startswith(str(workspace.project_path.resolve())):
                return "Error: Path traversal blocked - file must be within project directory"
            
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            
            full_path.write_text(content, encoding='utf-8')
            
            lines = len(content.splitlines())
            
            if task_id:
                task = workspace.get_task(task_id)
                if task:
                    task.artifacts_modified.append(file_path)
                    workspace.save()
            
            return {
                "file_path": file_path,
                "lines": lines,
                "bytes": len(content),
                "task_id": task_id,
            }
            
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"


class ProjectTestTool(Tool):
    """Run tests in the project."""
    
    name = "project_test"
    description = "Run tests in the project workspace. Detects test framework automatically."
    parameters = {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional)"
            },
            "test_filter": {
                "type": "string",
                "description": "Filter for specific tests"
            },
            "coverage": {
                "type": "boolean",
                "description": "Run with coverage",
                "default": False
            }
        },
        "required": []
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances
            import subprocess
            
            project_path = params.get("project_path")
            test_filter = params.get("test_filter", "")
            coverage = params.get("coverage", False)
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            test_framework = workspace._metadata.get("test_framework", "pytest")
            
            if test_framework == "pytest":
                cmd = ["python", "-m", "pytest", "-v"]
                if coverage:
                    cmd.extend(["--cov=.", "--cov-report=term-missing"])
                if test_filter:
                    cmd.append("-k")
                    cmd.append(test_filter)
            else:
                cmd = ["python", "-m", test_framework]
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    cwd=str(workspace.project_path),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            return {
                "returncode": result.returncode,
                "test_framework": test_framework,
                "output_length": len(output),
                "output": output[:5000],
            }
            
        except subprocess.TimeoutExpired:
            return "Error: Test execution timed out"
        except FileNotFoundError as e:
            return f"Error: Test framework not found: {e}"
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"


class ProjectCommitTool(Tool):
    """Commit changes in the project."""
    
    name = "project_commit"
    description = "Commit changes to the project's git repository with an auto-generated or provided message."
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Commit message (optional, auto-generated if not provided)"
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project (optional)"
            },
            "task_id": {
                "type": "string",
                "description": "Task ID this commit completes (optional)"
            },
            "stage_all": {
                "type": "boolean",
                "description": "Stage all changes before commit",
                "default": True
            }
        },
        "required": []
    }
    
    async def execute(self, **params) -> str | dict[str, Any]:
        try:
            from core.project_workspace import get_workspace, _workspace_instances
            import subprocess
            
            message = params.get("message")
            project_path = params.get("project_path")
            task_id = params.get("task_id")
            stage_all = params.get("stage_all", True)
            
            if project_path:
                project_path = Path(project_path).expanduser().resolve()
            elif _workspace_instances:
                project_path = Path(list(_workspace_instances.keys())[0])
            else:
                return "Error: No project initialized. Use project_init first."
            
            workspace = get_workspace(project_path)
            
            git_dir = workspace.project_path / ".git"
            if not git_dir.exists():
                return "Error: Not a git repository"
            
            if not message:
                if task_id:
                    task = workspace.get_task(task_id)
                    if task:
                        message = f"{task.title}\n\n{task.description}"
                    else:
                        message = f"Completed task {task_id}"
                else:
                    loop = asyncio.get_running_loop()
                    status_result = await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(
                            ["git", "status", "--short"],
                            cwd=str(workspace.project_path),
                            capture_output=True,
                            text=True,
                        )
                    )
                    files_changed = len(status_result.stdout.strip().split("\n")) if status_result.stdout.strip() else 0
                    message = f"Auto-commit: {files_changed} files changed"
            
            loop = asyncio.get_running_loop()
            if stage_all:
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "add", "."],
                        cwd=str(workspace.project_path),
                        check=True,
                        capture_output=True,
                    )
                )
            
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=str(workspace.project_path),
                    capture_output=True,
                    text=True,
                )
            )
            
            if result.returncode == 0:
                if task_id:
                    workspace.complete_task(task_id, message)
                
                hash_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "rev-parse", "--short", "HEAD"],
                        cwd=str(workspace.project_path),
                        capture_output=True,
                        text=True,
                    )
                )
                commit_hash = hash_result.stdout.strip()
                
                return {
                    "commit_hash": commit_hash,
                    "message": message,
                }
            else:
                return f"Error: Git commit failed: {result.stderr}"
            
        except subprocess.CalledProcessError as e:
            return f"Error: Git error: {e.stderr.decode() if e.stderr else safe_error_message(e, 'Project operation failed')}"
        except Exception as e:
            return f"Error: {safe_error_message(e, 'Project operation failed')}"


def register_project_tools() -> list[Tool]:
    """Register all project tools."""
    return [
        ProjectInitTool(),
        ProjectStatusTool(),
        ProjectDecomposeTool(),
        ProjectExecuteTool(),
        ProjectFileReadTool(),
        ProjectFileWriteTool(),
        ProjectTestTool(),
        ProjectCommitTool(),
    ]