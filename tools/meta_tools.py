"""Meta-cognitive tools for Jarvis - self-reflection and process design.

These tools allow Jarvis to reflect on its own behavior, design agentic
processes, and improve over time.
"""
from typing import Any

from .base import Tool


class SelfReflectionTool(Tool):
    """Tool for Jarvis to reflect on its own behavior."""

    name = "self_reflect"
    description = "Reflect on recent interactions, analyze behavior patterns, and suggest improvements"
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "What to focus the reflection on",
                "enum": ["recent", "patterns", "ethical", "improvements", "processes"],
            },
            "context": {
                "type": "string",
                "description": "Optional context for the reflection",
            },
        },
        "required": ["focus"],
    }
    requires_permission = False

    def __init__(self, reflection_engine: Any):
        """Initialize with a SelfReflectionEngine instance.

        Args:
            reflection_engine: The SelfReflectionEngine to use
        """
        self.engine = reflection_engine

    async def execute(self, focus: str, context: str | None = None) -> str:
        """Execute the self-reflection.

        Args:
            focus: What to focus on
            context: Optional additional context

        Returns:
            Reflection results
        """
        if focus == "recent":
            reflections = self.engine.get_recent_reflections(limit=5)
            if not reflections:
                return "No recent reflections available. I haven't analyzed my behavior yet."

            lines = ["Recent Self-Reflections:"]
            for r in reflections:
                lines.append(f"\n[{r.reflection_type.value}]")
                lines.append(r.content[:300] + "..." if len(r.content) > 300 else r.content)
                if r.insights:
                    lines.append(f"Insights: {', '.join(r.insights[:3])}")
            return "\n".join(lines)

        elif focus == "patterns":
            patterns = self.engine.get_behavior_patterns(min_count=2)
            if not patterns:
                return "No significant behavior patterns detected yet."

            lines = ["Detected Behavior Patterns:"]
            for behavior, count in patterns[:10]:
                lines.append(f"  • {behavior} (occurred {count} times)")
            return "\n".join(lines)

        elif focus == "ethical":
            lines = ["Ethical Self-Assessment:"]
            lines.append("\n1. **Permission Handling**: Do I ask before taking actions that affect the user's system?")
            lines.append("2. **Privacy Protection**: Do I protect sensitive data?")
            lines.append("3. **User Autonomy**: Do I respect the user's decisions?")
            lines.append("4. **Transparency**: Can I explain my reasoning?")

            # Add any ethical reflections
            ethical_reflections = [
                r for r in self.engine.reflections
                if r.reflection_type.value == "ethical"
            ]
            if ethical_reflections:
                lines.append("\nRecent Ethical Reflections:")
                for r in ethical_reflections[-3:]:
                    lines.append(f"  • {r.content[:100]}...")

            return "\n".join(lines)

        elif focus == "improvements":
            reflections = self.engine.get_recent_reflections(limit=10)
            improvements = []
            for r in reflections:
                improvements.extend(r.suggested_improvements)

            if not improvements:
                return "No improvement suggestions yet. I'm still learning from interactions."

            lines = ["Improvement Suggestions:"]
            unique_improvements = list(set(improvements))[:10]
            for i, imp in enumerate(unique_improvements, 1):
                lines.append(f"{i}. {imp}")
            return "\n".join(lines)

        elif focus == "processes":
            processes = self.engine.list_processes()
            if not processes:
                return "No agentic processes designed yet. I can design processes for specific tasks."

            lines = ["Designed Agentic Processes:"]
            for name in processes:
                process = self.engine.get_process(name)
                if process:
                    lines.append(f"\n**{name}**")
                    lines.append(f"Description: {process.description}")
                    lines.append(f"Steps: {len(process.steps)}")
                    if process.ethical_considerations:
                        lines.append(f"Ethical considerations: {len(process.ethical_considerations)}")
            return "\n".join(lines)

        return f"Unknown reflection focus: {focus}"


class ProcessDesignTool(Tool):
    """Tool for designing agentic processes."""

    name = "design_process"
    description = "Design a new agentic process for handling specific types of tasks"
    parameters = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "What the process should accomplish",
            },
            "context": {
                "type": "object",
                "description": "Additional context for the design",
            },
        },
        "required": ["request"],
    }
    requires_permission = False

    def __init__(self, reflection_engine: Any):
        """Initialize with a SelfReflectionEngine instance.

        Args:
            reflection_engine: The SelfReflectionEngine to use
        """
        self.engine = reflection_engine

    async def execute(self, request: str, context: dict[str, Any] | None = None) -> str:
        """Design a new process.

        Args:
            request: Description of what the process should accomplish
            context: Additional context

        Returns:
            The designed process
        """
        process = await self.engine.design_process(request, context)

        lines = [
            f"**Designed Process: {process.name}**",
            f"\nDescription: {process.description}",
        ]

        if process.steps:
            lines.append("\nSteps:")
            for i, step in enumerate(process.steps, 1):
                lines.append(f"  {i}. {step.get('description', 'Execute step')}")

        if process.triggers:
            lines.append(f"\nTriggers: {', '.join(process.triggers)}")

        if process.conditions:
            lines.append(f"\nConditions: {', '.join(process.conditions)}")

        if process.ethical_considerations:
            lines.append("\nEthical Considerations:")
            for ec in process.ethical_considerations:
                lines.append(f"  • {ec}")

        lines.append("\n*Process saved and available for review and improvement.*")

        return "\n".join(lines)


class ProcessReviewTool(Tool):
    """Tool for reviewing designed processes."""

    name = "review_process"
    description = "Review an existing agentic process for improvements"
    parameters = {
        "type": "object",
        "properties": {
            "process_name": {
                "type": "string",
                "description": "Name of the process to review",
            },
        },
        "required": ["process_name"],
    }
    requires_permission = False

    def __init__(self, reflection_engine: Any):
        """Initialize with a SelfReflectionEngine instance.

        Args:
            reflection_engine: The SelfReflectionEngine to use
        """
        self.engine = reflection_engine

    async def execute(self, process_name: str) -> str:
        """Review a process.

        Args:
            process_name: Name of the process to review

        Returns:
            Review results
        """
        reflection = await self.engine.review_process(process_name)

        lines = [
            f"**Review of: {process_name}**",
            "",
            reflection.content,
        ]

        if reflection.insights:
            lines.append("\nKey Insights:")
            for insight in reflection.insights:
                lines.append(f"  • {insight}")

        if reflection.suggested_improvements:
            lines.append("\nSuggested Improvements:")
            for imp in reflection.suggested_improvements:
                lines.append(f"  • {imp}")

        return "\n".join(lines)


def get_meta_tools(reflection_engine: Any) -> list[Tool]:
    """Get all meta-cognitive tools.

    Args:
        reflection_engine: The SelfReflectionEngine instance

    Returns:
        List of meta-cognitive tools
    """
    return [
        SelfReflectionTool(reflection_engine),
        ProcessDesignTool(reflection_engine),
        ProcessReviewTool(reflection_engine),
    ]
