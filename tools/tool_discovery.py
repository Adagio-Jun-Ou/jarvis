"""Tool Discovery - Let Jarvis query his own capabilities at runtime.

This provides tools for the agent to introspect and discover available tools,
skills, and capabilities dynamically.
"""
from typing import Any

from tools.base import Tool


class ListToolsTool(Tool):
    """List all available tools organized by category."""

    name = "list_tools"
    description = "List all available tools organized by category"
    parameters = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter (system, web, screen, browser, terminal, learning, evolution, multimodal, data)",
                "required": False,
            }
        },
    }

    async def execute(self, **params) -> str:
        from core.tool_registry import TOOL_REGISTRY

        category = params.get("category")
        tools_by_category: dict[str, list[dict]] = {}

        for tool_name, info in TOOL_REGISTRY.items():
            cat = info.category.value
            if category and cat != category:
                continue
            if cat not in tools_by_category:
                tools_by_category[cat] = []
            tools_by_category[cat].append({
                "name": info.name,
                "description": info.description,
                "requires_permission": info.requires_permission,
                "parameters": info.parameters,
            })

        result = f"# Available Tools ({len(TOOL_REGISTRY)} total)\n\n"
        for cat, tools in sorted(tools_by_category.items()):
            result += f"## {cat.upper()} ({len(tools)} tools)\n"
            for tool in tools:
                perm = "⚠️ " if tool["requires_permission"] else ""
                result += f"- **{tool['name']}**: {perm}{tool['description']}\n"
            result += "\n"

        return result


class GetToolInfoTool(Tool):
    """Get detailed information about a specific tool."""

    name = "get_tool_info"
    description = "Get detailed information about a specific tool"
    parameters = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the tool to get information about",
                "required": True,
            }
        },
        "required": ["tool_name"],
    }

    async def execute(self, **params) -> str:
        from core.tool_registry import get_tool_info

        tool_name = params.get("tool_name", "")
        info = get_tool_info(tool_name)

        if not info:
            return f"Tool '{tool_name}' not found. Use list_tools to see available tools."

        result = f"# {info.name}\n\n"
        result += f"**Category**: {info.category.value}\n"
        result += f"**Description**: {info.description}\n"
        result += f"**Requires Permission**: {'Yes ⚠️' if info.requires_permission else 'No'}\n\n"

        if info.parameters:
            result += "## Parameters\n"
            for param, desc in info.parameters.items():
                result += f"- `{param}`: {desc}\n"

        if info.examples:
            result += "\n## Examples\n"
            for ex in info.examples:
                result += f"- {ex}\n"

        if info.related_tools:
            result += f"\n**Related Tools**: {', '.join(info.related_tools)}\n"

        return result


class ListSkillsTool(Tool):
    """List all loaded skills."""

    name = "list_skills"
    description = "List all loaded skills and their status"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **params) -> str:
        from core.skill_loader import get_skill_loader

        loader = get_skill_loader()
        skills = loader.get_all_skills()
        loaded = loader.get_loaded_skills()

        result = "# Skills Overview\n\n"
        result += f"**Total**: {len(skills)} skills\n"
        result += f"**Loaded**: {len(loaded)} skills\n\n"

        for skill in skills:
            status = "✅" if skill.status.value == "loaded" else "⊘"
            result += f"## {status} {skill.name}\n"
            result += f"{skill.description[:100]}...\n\n"

        return result


class GetSkillInfoTool(Tool):
    """Get detailed information about a specific skill."""

    name = "get_skill_info"
    description = "Get detailed information about a specific skill"
    parameters = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill to get information about",
                "required": True,
            }
        },
        "required": ["skill_name"],
    }

    async def execute(self, **params) -> str:
        from core.skill_loader import get_skill_loader

        loader = get_skill_loader()
        skill_name = params.get("skill_name", "")

        skill_id = f"skill_{skill_name.lower().replace('-', '_').replace(' ', '_')}"
        skill = loader.get_skill(skill_id)

        if not skill:
            for s in loader.get_all_skills():
                if s.name.lower() == skill_name.lower():
                    skill = s
                    break

        if not skill:
            return f"Skill '{skill_name}' not found. Use list_skills to see available skills."

        result = f"# {skill.metadata.emoji} {skill.name}\n\n"
        result += f"**Status**: {skill.status.value}\n"
        result += f"**Description**: {skill.description}\n"
        result += f"**Version**: {skill.metadata.version}\n"
        result += f"**Author**: {skill.metadata.author}\n\n"

        if skill.requirements.bins:
            result += f"**Required Binaries**: {', '.join(skill.requirements.bins)}\n"
        if skill.requirements.env:
            result += f"**Required Env Vars**: {', '.join(skill.requirements.env)}\n"

        result += f"\n## Instructions\n\n{skill.instructions[:500]}...\n"

        return result


class GetCapabilitiesTool(Tool):
    """Get a summary of all capabilities."""

    name = "get_capabilities"
    description = "Get a summary of all Jarvis capabilities (tools, skills, subsystems)"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **params) -> str:
        from core.tool_registry import TOOL_REGISTRY, get_tool_summary
        from core.skill_loader import get_skill_loader

        loader = get_skill_loader()

        result = "# Jarvis Capabilities Summary\n\n"

        result += "## Tools\n\n"
        result += f"**Total Tools**: {len(TOOL_REGISTRY)}\n\n"
        result += get_tool_summary()

        result += "\n## Skills\n\n"
        result += f"**Total Skills**: {len(loader.get_all_skills())}\n"
        result += f"**Loaded**: {len(loader.get_loaded_skills())}\n\n"

        result += "## Subsystems\n\n"
        result += "- **Memory**: Vector search + conversation history\n"
        result += "- **Learning**: ErdosEngine (math), Daydream (exploration), LearningPipeline\n"
        result += "- **Evolution**: Self-improvement through genetic operations\n"
        result += "- **Vision**: Multimodal image analysis\n"
        result += "- **Voice**: Speech recognition and synthesis\n"
        result += "- **Browser**: Playwright automation\n"
        result += "- **Terminal**: Native shell access\n"

        return result


class FindRelatedToolsTool(Tool):
    """Find tools related to a given tool or task."""

    name = "find_related_tools"
    description = "Find tools related to a given tool or task description"
    parameters = {
        "type": "object",
        "properties": {
            "tool_or_task": {
                "type": "string",
                "description": "Name of a tool or description of a task",
                "required": True,
            }
        },
        "required": ["tool_or_task"],
    }

    async def execute(self, **params) -> str:
        from core.tool_registry import TOOL_REGISTRY, get_tool_info

        tool_or_task = params.get("tool_or_task", "")
        related = []

        info = get_tool_info(tool_or_task)
        if info:
            for related_name in info.related_tools:
                rel_info = get_tool_info(related_name)
                if rel_info:
                    related.append({
                        "name": rel_info.name,
                        "category": rel_info.category.value,
                        "description": rel_info.description,
                    })

        task_lower = tool_or_task.lower()
        if not related:
            for tool_name, tool_info in TOOL_REGISTRY.items():
                desc_lower = tool_info.description.lower()
                if (task_lower in desc_lower or
                    tool_name in task_lower or
                    any(word in desc_lower for word in task_lower.split())):
                    related.append({
                        "name": tool_info.name,
                        "category": tool_info.category.value,
                        "description": tool_info.description,
                    })

        if not related:
            return f"No related tools found for '{tool_or_task}'."

        result = f"# Tools Related to '{tool_or_task}'\n\n"
        for tool in related[:10]:
            result += f"- **{tool['name']}** ({tool['category']}): {tool['description']}\n"

        return result


TOOL_DISCOVERY_TOOLS = [
    ListToolsTool(),
    GetToolInfoTool(),
    ListSkillsTool(),
    GetSkillInfoTool(),
    GetCapabilitiesTool(),
    FindRelatedToolsTool(),
]


def register_tool_discovery_tools(agent: Any) -> int:
    """Register tool discovery tools with an agent."""
    agent.begin_batch_registration()
    for tool in TOOL_DISCOVERY_TOOLS:
        agent.register_tool(tool)
    agent.end_batch_registration()
    return len(TOOL_DISCOVERY_TOOLS)
