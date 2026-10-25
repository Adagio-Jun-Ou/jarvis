"""Task Decomposition System for Agentic Project Work.

Uses LLM to intelligently break down high-level goals into executable subtasks
with proper dependencies and acceptance criteria.

Key features:
- Goal analysis and decomposition
- Dependency inference
- Acceptance criteria generation
- Effort estimation
- Risk assessment
- Execution ordering

Example:
    decomposition = TaskDecomposition(llm_client)
    
    # Decompose a goal
    result = await decomposition.decompose(
        goal="Improve simulation accuracy in Research Project",
        context={
            "project": "Research Project",
            "files": ["run_simulation.py", "cfd_nozzle_simulation.py"],
        }
    )
    
    # Result contains:
    # - Stories with tasks
    # - Dependencies between tasks
    # - Acceptance criteria
    # - Estimated effort
    # - Risk factors
"""
import json
import logging
import re
import threading
from dataclasses import dataclass, field

from core.project_workspace import (
    TaskType, TaskPriority, TaskStatus,
    Epic, AcceptanceCriterion,
    ProjectWorkspace,
)

from core.helpers import get_response_content, safe_error_message

logger = logging.getLogger(__name__)


@dataclass
class DecompositionContext:
    """Context for decomposing a goal."""
    project_name: str
    project_path: str
    existing_epics: list[str] = field(default_factory=list)
    existing_stories: list[str] = field(default_factory=list)
    existing_tasks: list[str] = field(default_factory=list)
    architecture_overview: str = ""
    recent_changes: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)


@dataclass
class DecomposedTask:
    """A decomposed task with metadata."""
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    estimated_minutes: int
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    files_to_create: list[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    notes: str = ""


@dataclass
class DecomposedStory:
    """A decomposed story with tasks."""
    title: str
    description: str
    tasks: list[DecomposedTask] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class DecomposedEpic:
    """A decomposed epic with stories."""
    title: str
    description: str
    milestone: str | None = None
    stories: list[DecomposedStory] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Result of goal decomposition."""
    success: bool
    epic: DecomposedEpic | None = None
    analysis: str = ""
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    total_estimated_minutes: int = 0
    execution_order: list[str] = field(default_factory=list)


class TaskDecomposition:
    """Decomposes high-level goals into structured tasks using LLM.
    
    Features:
    - Analyzes goal complexity and scope
    - Identifies affected components
    - Creates stories and tasks with dependencies
    - Generates acceptance criteria
    - Estimates effort and risk
    - Produces execution ordering
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._lock = threading.Lock()
        self._decomposition_cache: dict[str, DecompositionResult] = {}
        self._MAX_DECOMPOSITION_CACHE = 100
    
    async def decompose(
        self,
        goal: str,
        context: DecompositionContext | None = None,
        depth: int = 2,
    ) -> DecompositionResult:
        """Decompose a high-level goal into stories and tasks.
        
        Args:
            goal: High-level goal description
            context: Project context (files, architecture, etc.)
            depth: Decomposition depth (1=epic only, 2=epic+stories, 3=epic+stories+tasks)
        
        Returns:
            DecompositionResult with structured breakdown
        """
        if not self.llm:
            return DecompositionResult(
                success=False,
                analysis="No LLM client available for decomposition",
            )
        
        cache_key = f"{goal}:{depth}:{context.project_name if context else ''}"
        if cache_key in self._decomposition_cache:
            return self._decomposition_cache[cache_key]
        
        try:
            prompt = self._build_decomposition_prompt(goal, context, depth)
            
            response = await self.llm.chat(
                user_input=prompt,
                system_prompt=self._get_system_prompt(),
            )
            
            content = get_response_content(response)
            
            result = self._parse_decomposition(content, goal, context)
            
            self._decomposition_cache[cache_key] = result
            if len(self._decomposition_cache) > self._MAX_DECOMPOSITION_CACHE:
                oldest_key = next(iter(self._decomposition_cache))
                del self._decomposition_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return DecompositionResult(
                success=False,
                analysis=f"Decomposition error: {e}",
            )
    
    def _get_system_prompt(self) -> str:
        return """You are an expert project manager and software architect. Your task is to decompose high-level goals into structured, actionable tasks following software engineering best practices.

When decomposing:
1. Consider project context including architecture, constraints, and existing work
2. Follow the user's preferences and priorities
3. Identify all affected files and components
4. Create clear acceptance criteria for each task
5. Estimate realistic effort based on task complexity
6. Identify dependencies between tasks
7. Assess risks and provide mitigation strategies
8. Order tasks for efficient execution

Output structure:
- Analysis section explaining your reasoning
- Epic with milestone
- Stories grouped by capability
- Tasks with:
  - Clear scope
  - Acceptance criteria
  - Dependencies
  - Effort estimate
  - Risk level
- Execution order
- Risk analysis
- Recommendations"""
    
    def _build_decomposition_prompt(
        self,
        goal: str,
        context: DecompositionContext | None,
        depth: int,
    ) -> str:
        context_str = ""
        if context:
            context_str = f"""
PROJECT CONTEXT:
- Project: {context.project_name}
- Path: {context.project_path}
- Existing Epics: {', '.join(context.existing_epics) if context.existing_epics else 'None'}
- Existing Stories: {len(context.existing_stories)} stories
- Existing Tasks: {len(context.existing_tasks)} tasks

ARCHITECTURE OVERVIEW:
{context.architecture_overview[:2000] if context.architecture_overview else 'Not provided'}

RELEVANT FILES:
{chr(10).join(f'- {f}' for f in context.relevant_files[:20]) if context.relevant_files else 'Not specified'}

CONSTRAINTS:
{chr(10).join(f'- {c}' for c in context.constraints) if context.constraints else 'None specified'}

PREFERENCES:
{chr(10).join(f'- {p}' for p in context.preferences) if context.preferences else 'None specified'}
"""
        
        prompt = f"""Decompose the following goal into structured tasks.

GOAL: {goal}
{context_str}
DEPTH: {'Epic only' if depth == 1 else 'Epic + Stories' if depth == 2 else 'Epic + Stories + Tasks'}

Provide your decomposition in the following JSON format:

{{
  "analysis": "Brief analysis of the goal and decomposition approach",
  "epic": {{
    "title": "Epic title",
    "description": "Epic description",
    "milestone": "Optional milestone",
    "stories": [
      {{
        "title": "Story title",
        "description": "Story description",
        "tasks": [
          {{
            "title": "Task title",
            "description": "Task description",
            "task_type": "implementation|refactoring|testing|documentation|research|fix",
            "priority": 1|2|3|4,
            "estimated_minutes": 30,
            "dependencies": ["task-id-1"],
            "acceptance_criteria": ["Criterion 1", "Criterion 2"],
            "files_to_modify": ["file1.py"],
            "files_to_create": ["new_file.py"],
            "risk_level": "low|medium|high",
            "notes": "Implementation notes"
          }}
        ],
        "acceptance_criteria": ["Story criterion 1"]
      }}
    ],
    "acceptance_criteria": ["Epic criterion 1"]
  }},
  "risks": ["Risk 1", "Risk 2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "execution_order": ["task-id-1", "task-id-2"]
}}

Guidelines:
- Create 3-7 stories per epic
- Create 2-5 tasks per story
- Each task should be completable in 15-120 minutes
- Identify realistic file dependencies
- Consider testing and documentation needs
- Include verification steps

Now decompose the goal:"""
        
        return prompt
    
    def _parse_decomposition(
        self,
        content: str,
        goal: str,
        context: DecompositionContext | None,
    ) -> DecompositionResult:
        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return DecompositionResult(
                    success=False,
                    analysis="Could not find JSON in response",
                )
            
            data = json.loads(json_match.group(0))
            
            epic_data = data.get("epic", {})
            stories = []
            total_minutes = 0
            task_id_map = {}
            
            for s_idx, story_data in enumerate(epic_data.get("stories", [])):
                tasks = []
                for t_idx, task_data in enumerate(story_data.get("tasks", [])):
                    task_type_str = task_data.get("task_type", "implementation").lower()
                    task_type = TaskType.IMPLEMENTATION
                    for tt in TaskType:
                        if tt.value == task_type_str:
                            task_type = tt
                            break
                    
                    priority = TaskPriority(task_data.get("priority", 3))
                    
                    estimated = task_data.get("estimated_minutes", 30)
                    total_minutes += estimated
                    
                    dep_titles = task_data.get("dependencies", [])
                    dep_ids = []
                    for dep_title in dep_titles:
                        if dep_title in task_id_map:
                            dep_ids.append(task_id_map[dep_title])
                    
                    acceptance_criteria = [
                        AcceptanceCriterion(
                            description=ac,
                            verification_method="manual" if "review" in ac.lower() else "automated"
                        )
                        for ac in task_data.get("acceptance_criteria", [])
                    ]
                    
                    risk_level = task_data.get("risk_level", "low")
                    
                    task = DecomposedTask(
                        title=task_data.get("title", f"Task {t_idx + 1}"),
                        description=task_data.get("description", ""),
                        task_type=task_type,
                        priority=priority,
                        estimated_minutes=estimated,
                        dependencies=dep_ids,
                        acceptance_criteria=[ac.description for ac in acceptance_criteria],
                        files_to_modify=task_data.get("files_to_modify", []),
                        files_to_create=task_data.get("files_to_create", []),
                        risk_level=risk_level,
                        notes=task_data.get("notes", ""),
                    )
                    tasks.append(task)
                    
                    task_id_map[task.title] = f"task-{s_idx}-{t_idx}"
                
                story = DecomposedStory(
                    title=story_data.get("title", f"Story {s_idx + 1}"),
                    description=story_data.get("description", ""),
                    tasks=tasks,
                    acceptance_criteria=story_data.get("acceptance_criteria", []),
                )
                stories.append(story)
            
            epic = DecomposedEpic(
                title=epic_data.get("title", goal[:50]),
                description=epic_data.get("description", goal),
                milestone=epic_data.get("milestone"),
                stories=stories,
                acceptance_criteria=epic_data.get("acceptance_criteria", []),
            )
            
            return DecompositionResult(
                success=True,
                epic=epic,
                analysis=data.get("analysis", ""),
                risks=data.get("risks", []),
                recommendations=data.get("recommendations", []),
                total_estimated_minutes=total_minutes,
                execution_order=data.get("execution_order", []),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return DecompositionResult(
                success=False,
                analysis=f"JSON parsing failed: {e}",
            )
        except Exception as e:
            logger.error(f"Decomposition parse error: {e}")
            return DecompositionResult(
                success=False,
                analysis=f"Parse error: {e}",
            )
    
    async def analyze_impact(
        self,
        proposed_change: str,
        context: DecompositionContext,
    ) -> dict:
        """Analyze the impact of a proposed change.
        
        Args:
            proposed_change: Description of the proposed change
            context: Project context
        
        Returns:
            Impact analysis with affected files, risk level, etc.
        """
        if not self.llm:
            return {"error": "No LLM client available"}
        
        try:
            prompt = f"""Analyze the impact of this proposed change on the project.

PROPOSED CHANGE: {proposed_change}

PROJECT CONTEXT:
- Project: {context.project_name}
- Architecture Overview: {context.architecture_overview[:1000] if context.architecture_overview else 'Not provided'}
- Relevant Files: {', '.join(context.relevant_files[:10]) if context.relevant_files else 'Not specified'}

Analyze:
1. Which files/components will be affected?
2. What is the risk level? (low/medium/high)
3. What are potential side effects?
4. What tests need to be updated?
5. What documentation needs updating?

Output as JSON:
{{
  "affected_files": ["file1.py", "file2.py"],
  "risk_level": "low|medium|high",
  "side_effects": ["Side effect 1", "Side effect 2"],
  "tests_to_update": ["test_file1.py"],
  "docs_to_update": ["README.md"],
  "estimated_effort_hours": 2,
  "recommendations": ["Recommendation 1"]
}}"""
            
            response = await self.llm.chat(
                user_input=prompt,
                system_prompt="You are a software architect analyzing change impact.",
            )
            
            content = get_response_content(response)
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            
            return {"error": "Could not parse response", "raw": content}
            
        except Exception as e:
            return {"error": safe_error_message(e)}
    
    async def suggest_next_task(
        self,
        workspace: ProjectWorkspace,
        goal: str | None = None,
    ) -> dict:
        """Suggest the next task to work on based on current state.
        
        Args:
            workspace: Project workspace
            goal: Optional focus goal
        
        Returns:
            Task suggestion with reasoning
        """
        status = workspace.get_status()
        next_task = workspace.get_next_task()
        
        if next_task:
            dependencies_met = all(
                workspace._is_dependency_met(dep) for dep in next_task.dependencies
            )
            
            return {
                "suggestion": next_task.to_dict(),
                "reasoning": f"Next task in queue: {next_task.title}",
                "dependencies_met": dependencies_met,
                "blocked": next_task.status == TaskStatus.BLOCKED,
            }
        
        if workspace._epics:
            return {
                "suggestion": None,
                "reasoning": "No pending tasks. Consider adding new stories or tasks.",
                "create_new": True,
                "project_status": status,
            }
        
        if goal:
            context = DecompositionContext(
                project_name=workspace.project_name,
                project_path=str(workspace.project_path),
                relevant_files=list(workspace._architecture.keys())[:20],
            )
            
            result = await self.decompose(goal, context)
            
            if result.success and result.epic:
                return {
                    "suggestion": None,
                    "reasoning": "No tasks found. Decomposing goal...",
                    "decomposition": result,
                    "project_status": status,
                }
        
        return {
            "suggestion": None,
            "reasoning": "No tasks available and no goal specified.",
            "project_status": status,
        }
    
    def to_workspace_tasks(
        self,
        result: DecompositionResult,
        workspace: ProjectWorkspace,
    ) -> Epic | None:
        """Convert decomposition result to workspace tasks.
        
        Args:
            result: Decomposition result from decompose()
            workspace: Target project workspace
        
        Returns:
            Created Epic with all stories and tasks
        """
        if not result.success or not result.epic:
            return None
        
        epic = workspace.add_epic(
            title=result.epic.title,
            description=result.epic.description,
            milestone=result.epic.milestone,
        )
        
        task_id_map = {}
        
        for story_data in result.epic.stories:
            story = workspace.add_story(
                epic_id=epic.id,
                title=story_data.title,
                description=story_data.description,
            )
            
            if story:
                for task_data in story_data.tasks:
                    task = workspace.add_task(
                        story_id=story.id,
                        title=task_data.title,
                        description=task_data.description,
                        task_type=task_data.task_type,
                        priority=task_data.priority,
                        estimated_minutes=task_data.estimated_minutes,
                    )
                    
                    if task:
                        task_id_map[task_data.title] = task.id
                        
                        for ac_text in task_data.acceptance_criteria:
                            task.acceptance_criteria.append(
                                AcceptanceCriterion(description=ac_text)
                            )
                        
                        task.artifacts_created = task_data.files_to_create
                        task.artifacts_modified = task_data.files_to_modify
                        task.metadata["risk_level"] = task_data.risk_level
                        task.metadata["notes"] = task_data.notes
        
        for story_data in result.epic.stories:
            for task_data in story_data.tasks:
                if task_data.dependencies:
                    task = workspace.get_task(task_id_map.get(task_data.title, ""))
                    if task:
                        task.dependencies = [task_id_map.get(dep, dep) for dep in task_data.dependencies]
        
        workspace.save()
        
        return epic


_decomposition_instances: dict[str, TaskDecomposition] = {}
_decomposition_lock = threading.Lock()


def get_task_decomposition(llm_client=None) -> TaskDecomposition:
    """Get or create a task decomposition instance."""
    with _decomposition_lock:
        if "default" not in _decomposition_instances:
            _decomposition_instances["default"] = TaskDecomposition(llm_client=llm_client)
        return _decomposition_instances["default"]