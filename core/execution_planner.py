"""Agentic Execution Planner for Multi-Step Project Work.

Plans and executes complex multi-step tasks autonomously:
- Execution plan generation with dependency ordering
- Progress tracking with checkpoint/resume
- Error handling and recovery strategies
- Parallel execution where possible
- Adaptive replanning based on results

Example:
    planner = ExecutionPlanner(llm_client)
    
    # Plan execution
    plan = await planner.create_plan(
        goal="Improve cell viability simulation",
        workspace=workspace,
    )
    
    # Execute plan
    result = await planner.execute_plan(plan)
    
    # Or use autonomous mode
    result = await planner.execute_goal(
        goal="Add support for cardiac tissue templates",
        workspace=workspace,
        max_iterations=10,
    )
"""
import asyncio
import json
import logging
import re
import shlex
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.project_workspace import (
    ProjectWorkspace,
)
from core.task_decomposition import TaskDecomposition, DecompositionContext
from core.helpers import get_response_content, safe_error_message

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """A single step in an execution plan."""
    id: str
    task_id: str
    order: int
    action: str
    expected_outcome: str
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.WAITING
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "order": self.order,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionStep":
        return cls(
            id=data["id"],
            task_id=data["task_id"],
            order=data["order"],
            action=data["action"],
            expected_outcome=data["expected_outcome"],
            dependencies=data.get("dependencies", []),
            status=StepStatus(data.get("status", "waiting")),
            result=data.get("result"),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class ExecutionPlan:
    """A complete execution plan for achieving a goal."""
    id: str
    goal: str
    workspace_path: str
    
    steps: list[ExecutionStep] = field(default_factory=list)
    
    state: ExecutionState = ExecutionState.PENDING
    current_step: int = 0
    progress: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    total_estimated_minutes: int = 0
    total_actual_minutes: int = 0
    
    checkpoint_file: str | None = None
    recovery_strategy: str = "retry"
    
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "workspace_path": self.workspace_path,
            "steps": [s.to_dict() for s in self.steps],
            "state": self.state.value,
            "current_step": self.current_step,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_estimated_minutes": self.total_estimated_minutes,
            "total_actual_minutes": self.total_actual_minutes,
            "checkpoint_file": self.checkpoint_file,
            "recovery_strategy": self.recovery_strategy,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        return cls(
            id=data["id"],
            goal=data["goal"],
            workspace_path=data["workspace_path"],
            steps=[ExecutionStep.from_dict(s) for s in data.get("steps", [])],
            state=ExecutionState(data.get("state", "pending")),
            current_step=data.get("current_step", 0),
            progress=data.get("progress", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            total_estimated_minutes=data.get("total_estimated_minutes", 0),
            total_actual_minutes=data.get("total_actual_minutes", 0),
            checkpoint_file=data.get("checkpoint_file"),
            recovery_strategy=data.get("recovery_strategy", "retry"),
            metadata=data.get("metadata", {}),
        )
    
    def get_next_ready_step(self) -> ExecutionStep | None:
        for step in self.steps:
            if step.status == StepStatus.WAITING:
                if all(
                    self._get_step(dep_id).status == StepStatus.SUCCESS
                    for dep_id in step.dependencies
                    if self._get_step(dep_id)
                ):
                    return step
        return None
    
    def _get_step(self, step_id: str) -> ExecutionStep | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def calculate_progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.SUCCESS)
        return completed / len(self.steps) * 100


class ExecutionPlanner:
    """Plans and executes complex multi-step tasks autonomously.
    
    Features:
    - Creates execution plans from goals using LLM decomposition
    - Tracks progress with checkpoint/resume
    - Handles errors with recovery strategies
    - Executes steps in optimal order with parallel execution
    - Adapts plans based on results
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.decomposition = TaskDecomposition(llm_client=llm_client)
        self._lock = threading.Lock()
        self._plans: dict[str, ExecutionPlan] = {}
        self._action_handlers: dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        self._action_handlers = {
            "create_file": self._handle_create_file,
            "modify_file": self._handle_modify_file,
            "run_tests": self._handle_run_tests,
            "run_command": self._handle_run_command,
            "create_directory": self._handle_create_directory,
            "copy_file": self._handle_copy_file,
            "delete_file": self._handle_delete_file,
            "review_code": self._handle_review_code,
            "git_commit": self._handle_git_commit,
        }
    
    def register_handler(self, action: str, handler: Callable) -> None:
        self._action_handlers[action] = handler
    
    async def create_plan(
        self,
        goal: str,
        workspace: ProjectWorkspace,
        max_steps: int = 20,
    ) -> ExecutionPlan:
        """Create an execution plan from a goal.
        
        Args:
            goal: The high-level goal to achieve
            workspace: Project workspace
            max_steps: Maximum number of steps in the plan
        
        Returns:
            ExecutionPlan with ordered steps
        """
        context = DecompositionContext(
            project_name=workspace.project_name,
            project_path=str(workspace.project_path),
            existing_epics=list(workspace._epics.keys()),
            relevant_files=list(workspace._architecture.keys())[:30],
        )
        
        decomposition_result = await self.decomposition.decompose(goal, context)
        
        if not decomposition_result.success:
            return ExecutionPlan(
                id=f"plan-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                goal=goal,
                workspace_path=str(workspace.project_path),
                state=ExecutionState.FAILED,
                metadata={"error": decomposition_result.analysis},
            )
        
        steps = []
        step_order = 0
        
        if decomposition_result.epic:
            epic = decomposition_result.epic
            
            for story in epic.stories:
                for task in story.tasks:
                    if step_order >= max_steps:
                        break
                    
                    action = self._infer_action(task)
                    
                    step = ExecutionStep(
                        id=f"step-{step_order}",
                        task_id=task.title,
                        order=step_order,
                        action=action,
                        expected_outcome=task.acceptance_criteria[0] if task.acceptance_criteria else task.description,
                        dependencies=[],
                    )
                    
                    steps.append(step)
                    step_order += 1
        
        total_minutes = decomposition_result.total_estimated_minutes
        
        plan_id = f"plan-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        plan = ExecutionPlan(
            id=plan_id,
            goal=goal,
            workspace_path=str(workspace.project_path),
            steps=steps,
            total_estimated_minutes=total_minutes,
            checkpoint_file=str(workspace._config_path / f"{plan_id}.json"),
        )
        
        for i, step in enumerate(steps):
            if i > 0:
                for j in range(i):
                    prev_task = steps[j].task_id
                    if prev_task in decomposition_result.execution_order[:i]:
                        step.dependencies.append(steps[j].id)
        
        self._plans[plan_id] = plan
        
        return plan
    
    def _infer_action(self, task: Any) -> str:
        title_lower = task.title.lower()
        desc_lower = task.description.lower() if task.description else ""
        
        if "create" in title_lower or "create" in desc_lower:
            if "file" in title_lower or "file" in desc_lower:
                return "create_file"
            if "directory" in title_lower or "folder" in desc_lower:
                return "create_directory"
        
        if "modify" in title_lower or "update" in title_lower or "edit" in title_lower:
            return "modify_file"
        
        if "test" in title_lower:
            return "run_tests"
        
        if "run" in title_lower or "execute" in title_lower:
            return "run_command"
        
        if "review" in title_lower:
            return "review_code"
        
        if "commit" in title_lower or "git" in title_lower:
            return "git_commit"
        
        if task.files_to_create:
            return "create_file"
        
        if task.files_to_modify:
            return "modify_file"
        
        return "modify_file"
    
    async def execute_step(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict | None = None,
    ) -> tuple[bool, str]:
        """Execute a single step.
        
        Args:
            step: The step to execute
            workspace: Project workspace
            context: Additional context for execution
        
        Returns:
            Tuple of (success, result_message)
        """
        handler = self._action_handlers.get(step.action)
        
        if not handler:
            return False, f"No handler for action: {step.action}"
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        try:
            success, result = await handler(step, workspace, context or {})
            
            step.completed_at = datetime.now()
            
            if success:
                step.status = StepStatus.SUCCESS
                step.result = result
            else:
                step.status = StepStatus.FAILED
                step.error = result
            
            return success, result
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now()
            return False, safe_error_message(e)
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        workspace: ProjectWorkspace,
        context: dict | None = None,
        on_step_complete: Callable | None = None,
    ) -> ExecutionPlan:
        """Execute a plan step by step.
        
        Args:
            plan: The plan to execute
            workspace: Project workspace
            context: Additional context
            on_step_complete: Callback after each step
        
        Returns:
            Updated execution plan
        """
        plan.state = ExecutionState.RUNNING
        plan.started_at = datetime.now()
        
        max_loop_iterations = len(plan.steps) * 3
        loop_count = 0
        
        while True:
            loop_count += 1
            if loop_count > max_loop_iterations:
                logger.error(f"Execution loop exceeded max iterations ({max_loop_iterations}), aborting")
                plan.state = ExecutionState.FAILED
                plan.error = {"error": f"Execution loop exceeded max iterations ({max_loop_iterations})"}
                break
            
            ready_step = plan.get_next_ready_step()
            
            if ready_step is None:
                all_done = all(
                    s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
                    for s in plan.steps
                )
                
                if all_done:
                    plan.state = ExecutionState.COMPLETED
                    plan.completed_at = datetime.now()
                else:
                    plan.state = ExecutionState.FAILED
                
                plan.progress = plan.calculate_progress()
                break
            
            success, result = await self.execute_step(ready_step, workspace, context)
            
            if not success and ready_step.retry_count < ready_step.max_retries:
                ready_step.retry_count += 1
                ready_step.status = StepStatus.WAITING
                logger.warning(f"Step {ready_step.id} failed, retrying ({ready_step.retry_count}/{ready_step.max_retries})")
                continue
            
            if not success:
                if plan.recovery_strategy == "retry":
                    continue
                elif plan.recovery_strategy == "skip":
                    ready_step.status = StepStatus.SKIPPED
                else:
                    plan.state = ExecutionState.FAILED
                    plan.error = result
                    break
            
            plan.progress = plan.calculate_progress()
            
            if on_step_complete:
                try:
                    on_step_complete(plan, ready_step)
                except Exception as e:
                    logger.warning(f"on_step_complete callback error: {e}")
            
            await self._save_plan_checkpoint(plan)
        
        return plan
    
    async def execute_goal(
        self,
        goal: str,
        workspace: ProjectWorkspace,
        context: dict | None = None,
        max_iterations: int = 10,
        on_progress: Callable | None = None,
    ) -> ExecutionPlan:
        """Execute a goal with autonomous planning and execution.
        
        Args:
            goal: The goal to achieve
            workspace: Project workspace
            context: Additional context
            max_iterations: Maximum planning/execution iterations
            on_progress: Progress callback
        
        Returns:
            Final execution plan
        """
        plan = await self.create_plan(goal, workspace)
        
        if plan.state == ExecutionState.FAILED:
            return plan
        
        async def on_step_complete(plan: ExecutionPlan, step: ExecutionStep):
            if on_progress:
                on_progress(
                    plan.progress,
                    f"Completed: {step.task_id}",
                    step.result,
                )
        
        result = await self.execute_plan(plan, workspace, context, on_step_complete)
        
        iteration = 1
        while result.state == ExecutionState.FAILED and iteration < max_iterations:
            if on_progress:
                on_progress(result.progress, f"Iteration {iteration + 1}: Replanning...", None)
            
            remaining_goal = self._generate_remaining_goal(result)
            
            new_plan = await self.create_plan(remaining_goal, workspace)
            
            if new_plan.state == ExecutionState.FAILED:
                break
            
            result = await self.execute_plan(new_plan, workspace, context, on_step_complete)
            iteration += 1
        
        return result
    
    def _generate_remaining_goal(self, failed_plan: ExecutionPlan) -> str:
        failed_steps = [
            s for s in failed_plan.steps
            if s.status == StepStatus.FAILED
        ]
        
        return f"Fix the following failed tasks and continue: {', '.join(s.task_id for s in failed_steps)}"
    
    async def _save_plan_checkpoint(self, plan: ExecutionPlan) -> None:
        if not plan.checkpoint_file:
            return
        
        try:
            path = Path(plan.checkpoint_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(plan.to_dict(), indent=2)
            def _write_file(path, data):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(data)
            await asyncio.to_thread(_write_file, path, data)
        except Exception as e:
            logger.warning(f"Failed to save plan checkpoint: {e}")
    
    async def load_plan_checkpoint(self, checkpoint_file: str) -> ExecutionPlan | None:
        try:
            path = Path(checkpoint_file)
            if not path.exists():
                return None
            
            data_str = await asyncio.to_thread(path.read_text)
            data = json.loads(data_str)
            
            return ExecutionPlan.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load plan checkpoint: {e}")
            return None
    
    async def _handle_create_file(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        return await self._generate_and_create(step, workspace, context, create=True)
    
    async def _handle_modify_file(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        return await self._generate_and_create(step, workspace, context, create=False)
    
    async def _generate_and_create(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
        create: bool,
    ) -> tuple[bool, str]:
        if not self.llm:
            return False, "No LLM client available"
        
        relevant_files = context.get("relevant_files", [])
        file_constraints = context.get("file_constraints", [])
        
        prompt = f"""{'Create' if create else 'Modify'} a file to complete this task:

Task: {step.task_id}
Goal: {step.expected_outcome}

Workspace: {workspace.project_name}
Path: {workspace.project_path}

Relevant files in project:
{chr(10).join(f'- {f}' for f in relevant_files[:15]) if relevant_files else 'Not specified'}

Constraints:
{chr(10).join(f'- {c}' for c in file_constraints) if file_constraints else 'None specified'}

Output the file content. First line should be the file path (relative to project).
Then the complete file content.

Output:
PATH: <relative/path/to/file.py>
```<language>
<file content>
```

Now output the file:"""
        
        try:
            response = await self.llm.chat(
                user_input=prompt,
                system_prompt="You are an expert software developer. Output clean, well-documented code.",
            )
            
            content = get_response_content(response)
            
            path_match = re.search(r'PATH:\s*(.+)', content)
            if not path_match:
                return False, "Could not extract file path from response"
            
            file_path = workspace.project_path / path_match.group(1).strip()
            
            code_match = re.search(r'```(?:\w+)?\s*\n([\s\S]*?)```', content)
            if code_match:
                file_content = code_match.group(1)
            else:
                lines = content.split('\n')
                file_content = '\n'.join(lines[1:]) if lines else content
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(file_path.write_text, file_content)
            
            return True, f"{'Created' if create else 'Modified'} {file_path.relative_to(workspace.project_path)}"
            
        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_run_tests(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        import asyncio

        test_dir = workspace.project_path / "tests"
        if not test_dir.exists():
            test_dir = workspace.project_path

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", "-v",
                cwd=str(workspace.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "Test execution timed out"

            if proc.returncode == 0:
                return True, f"Tests passed:\n{stdout.decode()[:500]}"
            else:
                return False, f"Tests failed:\n{stderr.decode()[:500]}"

        except FileNotFoundError:
            return False, "pytest not installed"
        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_run_command(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        import asyncio

        command = context.get("command", "python --version")

        try:
            cmd_parts = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=str(workspace.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "Command timed out"

            if proc.returncode == 0:
                return True, f"Command succeeded:\n{stdout.decode()[:500]}"
            else:
                return False, f"Command failed:\n{stderr.decode()[:500]}"

        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_create_directory(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        dir_name = context.get("directory", step.task_id.replace(" ", "_").lower())
        dir_path = workspace.project_path / dir_name
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return True, f"Created directory: {dir_path.relative_to(workspace.project_path)}"
        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_copy_file(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        import shutil
        
        source = context.get("source")
        dest = context.get("destination")
        
        if not source or not dest:
            return False, "source and destination required"
        
        try:
            src_path = workspace.project_path / source
            dst_path = workspace.project_path / dest
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dst_path)
            
            return True, f"Copied {source} to {dest}"
        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_delete_file(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        file_path = context.get("file", step.task_id)
        
        try:
            path = workspace.project_path / file_path
            if path.exists():
                path.unlink()
                return True, f"Deleted: {file_path}"
            return False, f"File not found: {file_path}"
        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_review_code(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        if not self.llm:
            return False, "No LLM client available"
        
        file_path = context.get("file")
        if not file_path:
            return False, "No file specified for review"
        
        path = workspace.project_path / file_path
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        content = await asyncio.to_thread(path.read_text)
        
        prompt = f"""Review this code for:
1. Correctness
2. Best practices
3. Potential bugs
4. Performance issues
5. Security concerns

File: {file_path}
Content:
```
{content[:3000]}
```

Provide:
- ISSUES: List any problems found
- SUGGESTIONS: Improvement recommendations
- APPROVED: yes/no"""

        try:
            response = await self.llm.chat(
                user_input=prompt,
                system_prompt="You are a senior software engineer doing code reviews.",
            )
            
            result = get_response_content(response)
            
            approved = "APPROVED: yes" in result or "approved: yes" in result.lower()
            
            return approved, result[:500]
            
        except Exception as e:
            return False, safe_error_message(e)
    
    async def _handle_git_commit(
        self,
        step: ExecutionStep,
        workspace: ProjectWorkspace,
        context: dict,
    ) -> tuple[bool, str]:
        import asyncio

        message = context.get("commit_message", step.task_id)

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "add", ".",
                cwd=str(workspace.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return False, f"Git add failed: {stderr.decode()}"

            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", message,
                cwd=str(workspace.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return False, f"Git commit failed: {stderr.decode()}"

            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--short", "HEAD",
                cwd=str(workspace.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            commit_hash = stdout.decode().strip()

            return True, f"Committed: {commit_hash} - {message}"

        except FileNotFoundError:
            return False, "git not available"
        except Exception as e:
            return False, safe_error_message(e)


_planner_instances: dict[str, ExecutionPlanner] = {}
_planner_lock = threading.Lock()


def get_execution_planner(llm_client=None) -> ExecutionPlanner:
    """Get or create an execution planner instance."""
    with _planner_lock:
        if "default" not in _planner_instances:
            _planner_instances["default"] = ExecutionPlanner(llm_client=llm_client)
        return _planner_instances["default"]