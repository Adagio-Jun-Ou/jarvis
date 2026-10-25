"""Self-reflection and agentic process design for Jarvis.

This module enables Jarvis to reflect on its own behavior, design improvements
to its processes, and review its decision-making patterns.

Key features:
- Self-reflection on completed interactions
- Agentic process design and optimization
- Pattern recognition in own behavior
- Continuous improvement suggestions
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json
import logging
import threading
from pathlib import Path

from core.helpers import atomic_write_json, get_response_content, safe_error_message, read_json_with_integrity

logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of self-reflection."""
    INTERACTION = "interaction"      # Reflect on a single interaction
    PATTERN = "pattern"             # Identify patterns in behavior
    PROCESS = "process"             # Review agentic processes
    IMPROVEMENT = "improvement"      # Suggest improvements
    ETHICAL = "ethical"             # Ethical considerations


@dataclass
class Reflection:
    """Represents a self-reflection result."""
    reflection_type: ReflectionType
    content: str
    insights: list[str] = field(default_factory=list)
    suggested_improvements: list[str] = field(default_factory=list)
    confidence: float = 0.7
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.reflection_type.value,
            "content": self.content,
            "insights": self.insights,
            "improvements": self.suggested_improvements,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgenticProcess:
    """Represents an agentic process design."""
    name: str
    description: str
    steps: list[dict[str, Any]]
    triggers: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class SelfReflectionEngine:
    """Engine for self-reflection and agentic process design.

    This allows Jarvis to reflect on its own behavior, identify patterns,
    suggest improvements, and design new agentic processes.
    """

    REFLECTION_PROMPT = """You are reflecting on your own behavior as an AI assistant.

Your task is to analyze your recent interactions and provide honest self-assessment.
Be constructive and specific. Focus on actionable improvements.

## Areas to Consider

1. **Communication Quality**
   - Was I clear and helpful?
   - Did I ask the right clarifying questions?
   - Did I respect the user's autonomy?

2. **Decision Making**
   - Did I choose the right tools for the task?
   - Were there better approaches I missed?
   - Did I consider all relevant constraints?

3. **Ethical Alignment**
   - Did I properly handle permission requests?
   - Did I protect the user's privacy and data?
   - Were my suggestions aligned with user values?

4. **Process Efficiency**
   - Could I have been more efficient?
   - Did I repeat unnecessary steps?
   - Was my reasoning sound?

## Recent Interaction Context
{context}

## Your Reflection

Provide a brief reflection covering:
1. What went well
2. What could be improved
3. Any patterns you notice in your behavior
4. Specific improvement suggestions

Be concise and actionable. This is for your own learning."""

    PROCESS_DESIGN_PROMPT = """You are designing an agentic process for yourself.

An agentic process is a structured approach to handling specific types of tasks.
Design processes that are:
- Clear and actionable
- Ethically sound
- Respectful of user autonomy
- Efficient and effective

## Process Request
{request}

## Design Requirements

1. **Name**: A clear name for the process
2. **Description**: What this process accomplishes
3. **Steps**: Ordered list of steps with conditions
4. **Triggers**: When this process should activate
5. **Conditions**: Preconditions and constraints
6. **Ethical Considerations**: Safety and permission requirements

## Output Format

Provide your process design in a structured format that can be saved and reviewed later."""

    def __init__(
        self,
        llm_client: Any,
        storage_path: Path | None = None,
    ):
        """Initialize the self-reflection engine.

        Args:
            llm_client: The LLM client for generating reflections
            storage_path: Optional path for storing reflections and processes
        """
        self.llm = llm_client
        self.storage_path = storage_path

        self._api_manager = None
        self._subsystem_name = "meta_reflection"
        self._lock = threading.RLock()

        # Storage
        self.reflections: list[Reflection] = []
        self.processes: dict[str, AgenticProcess] = {}
        self.behavior_patterns: dict[str, int] = {}  # pattern -> count

        # Load existing data if available
        if storage_path:
            self._load_state()

    def _get_api_manager(self):
        """Get or create API manager for coordinated LLM calls."""
        if self._api_manager is None and self.llm:
            try:
                from core.api_manager import get_api_manager
                self._api_manager = get_api_manager(self.llm)
            except Exception as e:
                logger.warning(f"Failed to create API manager: {e}")
        return self._api_manager

    async def _llm_call(self, user_input: str, system_prompt: str, priority: str = "maintenance") -> dict:
        """Make coordinated LLM call with rate limiting."""
        from core.api_manager import Priority as APIPriority

        priority_map = {
            "interactive": APIPriority.INTERACTIVE,
            "learning": APIPriority.LEARNING,
            "refinement": APIPriority.REFINEMENT,
            "maintenance": APIPriority.MAINTENANCE,
        }

        api_manager = self._get_api_manager()
        if api_manager is not None:
            return await api_manager.submit_chat(
                subsystem=self._subsystem_name,
                priority=priority_map.get(priority, APIPriority.MAINTENANCE),
                user_input=user_input,
                system_prompt=system_prompt,
            )

        if self.llm:
            return await self.llm.chat(user_input=user_input, system_prompt=system_prompt)

        raise RuntimeError("No LLM client available")

    async def reflect_on_interaction(
        self,
        user_input: str,
        assistant_response: str,
        tools_used: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Reflection:
        """Reflect on a completed interaction.

        Args:
            user_input: What the user asked
            assistant_response: How Jarvis responded
            tools_used: Tools that were used (if any)
            context: Additional context about the interaction

        Returns:
            A Reflection object with insights and improvements
        """
        # Build context string
        context_str = f"User: {user_input}\nAssistant: {assistant_response}"
        if tools_used:
            context_str += f"\nTools used: {', '.join(tools_used)}"
        if context:
            context_str += f"\nAdditional context: {context}"

        # Generate reflection using LLM
        prompt = self.REFLECTION_PROMPT.format(context=context_str)

        try:
            result = await self._llm_call(
                user_input=prompt,
                system_prompt="You are an AI assistant reflecting on your own behavior. Provide honest self-assessment.",
                priority="maintenance",
            )
            content = get_response_content(result)

            reflection = self._parse_reflection(content, ReflectionType.INTERACTION)

            # Store
            with self._lock:
                self.reflections.append(reflection)
            self._save_state()

            return reflection

        except Exception as e:
            return Reflection(
                reflection_type=ReflectionType.INTERACTION,
                content=f"Reflection failed: {safe_error_message(e)}",
                confidence=0.0,
            )

    async def design_process(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> AgenticProcess:
        """Design a new agentic process.

        Args:
            request: Description of what the process should accomplish
            context: Additional context for the design

        Returns:
            An AgenticProcess design
        """
        # Build request string
        request_str = request
        if context:
            request_str += f"\nContext: {context}"

        # Generate process design
        prompt = self.PROCESS_DESIGN_PROMPT.format(request=request_str)

        try:
            result = await self._llm_call(
                user_input=prompt,
                system_prompt="You are an AI assistant designing agentic processes. Provide structured, actionable process designs.",
                priority="maintenance",
            )
            content = get_response_content(result)

            process = self._parse_process(content)

            # Store
            with self._lock:
                self.processes[process.name] = process
            self._save_state()

            return process

        except Exception as e:
            return AgenticProcess(
                name="Error",
                description=f"Process design failed: {safe_error_message(e)}",
                steps=[],
            )

    async def review_process(self, process_name: str) -> Reflection:
        """Review an existing agentic process.

        Args:
            process_name: Name of the process to review

        Returns:
            A Reflection with review findings
        """
        if process_name not in self.processes:
            return Reflection(
                reflection_type=ReflectionType.PROCESS,
                content=f"Process '{process_name}' not found",
                confidence=0.0,
            )

        with self._lock:
            process = self.processes.get(process_name)
            if process is None:
                return Reflection(
                    reflection_type=ReflectionType.PROCESS,
                    content=f"Process '{process_name}' not found",
                    confidence=0.0,
                )

        # Build review context
        context = f"""Process: {process.name}
Description: {process.description}
Steps: {json.dumps(process.steps, indent=2)}
Triggers: {process.triggers}
Conditions: {process.conditions}
Ethical Considerations: {process.ethical_considerations}"""

        prompt = f"""Review this agentic process and provide:

1. Strengths of the process
2. Potential weaknesses or risks
3. Suggestions for improvement
4. Ethical considerations

{context}"""

        try:
            result = await self._llm_call(
                user_input=prompt,
                system_prompt="You are an AI assistant reviewing agentic processes. Provide constructive feedback.",
                priority="maintenance",
            )
            content = get_response_content(result)

            reflection = self._parse_reflection(content, ReflectionType.PROCESS)
            with self._lock:
                self.reflections.append(reflection)
            self._save_state()

            return reflection

        except Exception as e:
            return Reflection(
                reflection_type=ReflectionType.PROCESS,
                content=f"Review failed: {safe_error_message(e)}",
                confidence=0.0,
            )

    def track_behavior(self, behavior: str) -> None:
        """Track a behavior pattern.

        Args:
            behavior: Description of the behavior
        """
        with self._lock:
            self.behavior_patterns[behavior] = self.behavior_patterns.get(behavior, 0) + 1

    def get_behavior_patterns(self, min_count: int = 2) -> list[tuple[str, int]]:
        """Get behavior patterns that occur frequently.

        Args:
            min_count: Minimum occurrences to consider a pattern

        Returns:
            List of (behavior, count) tuples sorted by frequency
        """
        with self._lock:
            patterns = [
                (b, c) for b, c in self.behavior_patterns.items()
                if c >= min_count
            ]
        return sorted(patterns, key=lambda x: x[1], reverse=True)

    def get_recent_reflections(self, limit: int = 10) -> list[Reflection]:
        """Get recent reflections.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent Reflection objects
        """
        with self._lock:
            return list(self.reflections[-limit:])

    def get_process(self, name: str) -> AgenticProcess | None:
        """Get a process by name.

        Args:
            name: Process name

        Returns:
            The AgenticProcess or None if not found
        """
        with self._lock:
            return self.processes.get(name)

    def list_processes(self) -> list[str]:
        """List all process names.

        Returns:
            List of process names
        """
        with self._lock:
            return list(self.processes.keys())

    def _parse_reflection(self, content: str, reflection_type: ReflectionType) -> Reflection:
        """Parse LLM output into a Reflection object."""
        # Simple parsing - could be enhanced
        insights = []
        improvements = []

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()

            if "went well" in line_lower or "strength" in line_lower:
                current_section = "insight"
            elif "improve" in line_lower or "suggestion" in line_lower:
                current_section = "improvement"
            elif line.strip().startswith("-") or line.strip().startswith("*"):
                item = line.strip().lstrip("-*").strip()
                if current_section == "insight":
                    insights.append(item)
                elif current_section == "improvement":
                    improvements.append(item)

        return Reflection(
            reflection_type=reflection_type,
            content=content,
            insights=insights,
            suggested_improvements=improvements,
            confidence=0.7,
        )

    def _parse_process(self, content: str) -> AgenticProcess:
        """Parse LLM output into an AgenticProcess object."""
        # Simple parsing - could be enhanced with more structure
        lines = content.split("\n")

        name = "Unnamed Process"
        description = ""
        steps = []
        triggers = []
        conditions = []
        ethical = []

        for line in lines:
            line_lower = line.lower().strip()

            if "name:" in line_lower:
                name = line.split(":", 1)[1].strip() if ":" in line else name
            elif "description:" in line_lower:
                description = line.split(":", 1)[1].strip() if ":" in line else description
            elif "trigger" in line_lower:
                triggers.append(line.split(":", 1)[1].strip() if ":" in line else line.strip())
            elif "condition" in line_lower:
                conditions.append(line.split(":", 1)[1].strip() if ":" in line else line.strip())
            elif "ethical" in line_lower:
                ethical.append(line.split(":", 1)[1].strip() if ":" in line else line.strip())
            elif line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-", "*")):
                step_text = line.strip().lstrip("12345.-*").strip()
                if step_text:
                    steps.append({"description": step_text})

        return AgenticProcess(
            name=name,
            description=description,
            steps=steps,
            triggers=triggers,
            conditions=conditions,
            ethical_considerations=ethical,
        )

    def _save_state(self) -> None:
        """Save state to disk."""
        if self.storage_path is None:
            return

        with self._lock:
            state = {
                "reflections": [r.to_dict() for r in self.reflections[-100:]],
                "processes": {
                    name: {
                        "name": p.name,
                        "description": p.description,
                        "steps": p.steps,
                        "triggers": p.triggers,
                        "conditions": p.conditions,
                        "ethical_considerations": p.ethical_considerations,
                    }
                    for name, p in self.processes.items()
                },
                "behavior_patterns": dict(list(self.behavior_patterns.items())[-200:]),
            }

        self.storage_path.mkdir(parents=True, exist_ok=True)
        state_file = self.storage_path / "self_reflection.json"
        atomic_write_json(state_file, state)

    def _load_state(self) -> None:
        """Load state from disk."""
        if self.storage_path is None:
            return

        state_file = self.storage_path / "self_reflection.json"
        if not state_file.exists():
            return

        try:
            state = read_json_with_integrity(state_file)

            with self._lock:
                self.reflections = [
                    Reflection(
                        reflection_type=ReflectionType(r.get("type", "interaction")),
                        content=r.get("content", ""),
                        insights=r.get("insights", []),
                        suggested_improvements=r.get("improvements", []),
                        confidence=r.get("confidence", 0.7),
                    )
                    for r in state.get("reflections", [])
                ]

                self.processes = {
                    name: AgenticProcess(
                        name=p.get("name", name),
                        description=p.get("description", ""),
                        steps=p.get("steps", []),
                        triggers=p.get("triggers", []),
                        conditions=p.get("conditions", []),
                        ethical_considerations=p.get("ethical_considerations", []),
                    )
                    for name, p in state.get("processes", {}).items()
                }

                self.behavior_patterns = state.get("behavior_patterns", {})

        except Exception as e:
            logger.warning(f"Failed to load self-reflection state: {e}")


# Self-reflection tool for use in conversations
SELF_REFLECTION_PROMPTS = {
    "interaction_review": "How did I handle that last interaction? What could I have done better?",
    "process_design": "I need to design a process for {task}. What steps should I take?",
    "behavior_analysis": "What patterns do I notice in how I've been responding?",
    "ethical_check": "Am I properly handling permissions and user autonomy in my responses?",
    "improvement_plan": "Based on my recent interactions, what should I focus on improving?",
}
