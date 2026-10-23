"""OpenClaw-style Skill Loader for Jarvis.

A skill loader that reads SKILL.md files with YAML frontmatter,
validates requirements (bins, env, config), and loads skills into
the agent's tool registry and system prompt.

Skill Location Hierarchy (by precedence):
1. <workspace>/skills - Per-agent skills (highest)
2. ~/.jarvis/skills - Managed/local skills (shared)
3. <jarvis>/skills - Bundled skills (shipped with Jarvis)

SKILL.md Format:
```markdown
---
name: skill-name
description: What this skill does
metadata:
  jarvis:
    requires:
      bins: [python, uv]
      env: [API_KEY]
      config: [browser.enabled]
    always: false
    user-invocable: true
    disable-model-invocation: false
    primaryEnv: API_KEY
    emoji: "🦞"
---

# Instructions for the agent on how to use this skill...
```

Example:
    loader = get_skill_loader()
    
    # Load all skills from default directories
    skills = loader.load_all_skills()
    
    # Check requirements for a skill
    if loader.check_requirements(skill):
        loader.register_with_agent(skill, agent)
"""
import ast
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    LOADED = "loaded"
    SKIPPED = "skipped"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class SkillRequirements:
    bins: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    config: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bins": self.bins,
            "env": self.env,
            "config": self.config,
        }


@dataclass
class SkillMetadata:
    always: bool = False
    user_invocable: bool = False
    disable_model_invocation: bool = False
    primary_env: str | None = None
    emoji: str = "🔧"
    version: str = "1.0.0"
    author: str = "unknown"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "always": self.always,
            "user_invocable": self.user_invocable,
            "disable_model_invocation": self.disable_model_invocation,
            "primary_env": self.primary_env,
            "emoji": self.emoji,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
        }


@dataclass
class LoadedSkill:
    skill_id: str
    name: str
    description: str
    instructions: str
    requirements: SkillRequirements
    metadata: SkillMetadata
    source_path: Path
    status: SkillStatus = SkillStatus.PENDING
    loaded_at: datetime | None = None
    error_message: str | None = None
    tools: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions[:500] + "..." if len(self.instructions) > 500 else self.instructions,
            "requirements": self.requirements.to_dict(),
            "metadata": self.metadata.to_dict(),
            "source_path": str(self.source_path),
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "error_message": self.error_message,
            "tool_count": len(self.tools),
        }


class SkillLoader:
    """Loads and manages OpenClaw-style skills for Jarvis."""

    BUNDLED_SKILLS_DIR = "skills"
    LOCAL_SKILLS_DIR = ".jarvis/skills"
    WORKSPACE_SKILLS_DIR = "skills"
    SKILL_FILE = "SKILL.md"

    def __init__(self, jarvis_path: Path | None = None, workspace_path: Path | None = None):
        self.jarvis_path = jarvis_path or Path(__file__).parent.parent
        self.workspace_path = workspace_path or Path.cwd()

        self._skills: dict[str, LoadedSkill] = {}
        self._skill_dirs: list[Path] = []
        self._lock = threading.RLock()
        self._config_getter: Callable[[str], Any] | None = None

        self._init_skill_directories()

    def _init_skill_directories(self) -> None:
        """Initialize skill directories in precedence order."""
        self._skill_dirs = []

        workspace_skills = self.workspace_path / self.WORKSPACE_SKILLS_DIR
        if workspace_skills.exists():
            self._skill_dirs.append(workspace_skills)

        local_skills = Path.home() / self.LOCAL_SKILLS_DIR
        if local_skills.exists():
            self._skill_dirs.append(local_skills)

        bundled_skills = self.jarvis_path / self.BUNDLED_SKILLS_DIR
        if bundled_skills.exists():
            self._skill_dirs.append(bundled_skills)

        logger.info(f"Skill directories: {[str(d) for d in self._skill_dirs]}")

    def set_config_getter(self, getter: Callable[[str], Any]) -> None:
        """Set a function to get configuration values."""
        self._config_getter = getter

    def discover_skills(self) -> list[Path]:
        """Discover all SKILL.md files in skill directories."""
        skill_files = []
        seen_names: set[str] = set()

        for skill_dir in self._skill_dirs:
            if not skill_dir.exists():
                continue

            for skill_file in skill_dir.rglob(self.SKILL_FILE):
                skill_name = skill_file.parent.name
                if skill_name not in seen_names:
                    seen_names.add(skill_name)
                    skill_files.append(skill_file)

        logger.info(f"Discovered {len(skill_files)} skill files")
        return skill_files

    def parse_skill_file(self, skill_path: Path) -> LoadedSkill | None:
        """Parse a SKILL.md file and return a LoadedSkill."""
        try:
            content = skill_path.read_text(encoding='utf-8')

            frontmatter, instructions = self._parse_frontmatter(content)
            if not frontmatter:
                logger.warning(f"No frontmatter found in {skill_path}")
                return None

            name = frontmatter.get("name", skill_path.parent.name)
            description = frontmatter.get("description", "")

            jarvis_meta = frontmatter.get("metadata", {}).get("jarvis", {})
            openclaw_meta = frontmatter.get("metadata", {}).get("openclaw", {})
            meta = {**openclaw_meta, **jarvis_meta}

            requires = meta.get("requires", {})
            requirements = SkillRequirements(
                bins=requires.get("bins", []),
                env=requires.get("env", []),
                config=requires.get("config", []),
            )

            metadata = SkillMetadata(
                always=meta.get("always", False),
                user_invocable=meta.get("user-invocable", False),
                disable_model_invocation=meta.get("disable-model-invocation", False),
                primary_env=meta.get("primaryEnv") or meta.get("primary_env"),
                emoji=meta.get("emoji", "🔧"),
                version=frontmatter.get("version", "1.0.0"),
                author=frontmatter.get("author", "unknown"),
                tags=frontmatter.get("tags", []),
            )

            skill_id = f"skill_{name.lower().replace('-', '_').replace(' ', '_')}"

            return LoadedSkill(
                skill_id=skill_id,
                name=name,
                description=description,
                instructions=instructions,
                requirements=requirements,
                metadata=metadata,
                source_path=skill_path,
            )

        except Exception as e:
            logger.error(f"Failed to parse skill {skill_path}: {e}")
            return None

    def _parse_frontmatter(self, content: str) -> tuple[dict | None, str]:
        """Parse YAML frontmatter from content."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return None, content

        try:
            frontmatter = yaml.safe_load(match.group(1))
            instructions = match.group(2).strip()
            return frontmatter, instructions
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error: {e}")
            return None, content

    def check_requirements(self, skill: LoadedSkill) -> tuple[bool, list[str]]:
        """Check if skill requirements are met.
        
        Returns:
            Tuple of (all_met, list_of_missing)
        """
        missing = []
        
        # Binary aliases: maps requested binary to acceptable alternatives
        binary_aliases = {
            "python": ["python3", "python"],
            "python3": ["python", "python3"],
            "node": ["node", "nodejs"],
            "npm": ["npm", "npm"],
        }

        for bin_name in skill.requirements.bins:
            # Check aliases first
            aliases = binary_aliases.get(bin_name, [bin_name])
            found = any(shutil.which(alias) for alias in aliases)
            if not found:
                missing.append(f"binary: {bin_name}")

        for env_var in skill.requirements.env:
            if not os.environ.get(env_var):
                if not skill.metadata.primary_env or env_var != skill.metadata.primary_env:
                    missing.append(f"env: {env_var}")

        if self._config_getter:
            for config_path in skill.requirements.config:
                value = self._config_getter(config_path)
                if value is None or value is False:
                    missing.append(f"config: {config_path}")

        return len(missing) == 0, missing

    def load_skill(self, skill_path: Path) -> LoadedSkill | None:
        """Load a single skill from path."""
        skill = self.parse_skill_file(skill_path)
        if not skill:
            return None

        met, missing = self.check_requirements(skill)

        if not met and not skill.metadata.always:
            skill.status = SkillStatus.SKIPPED
            skill.error_message = f"Missing requirements: {', '.join(missing)}"
            logger.info(f"Skipped skill {skill.name}: {skill.error_message}")
            return skill

        skill.status = SkillStatus.LOADED
        skill.loaded_at = datetime.now()

        tool_file = skill_path.parent / "tool.py"
        if tool_file.exists():
            tools = self._load_tool_file(tool_file, skill)
            skill.tools = tools

        with self._lock:
            self._skills[skill.skill_id] = skill

        logger.info(f"Loaded skill: {skill.name} ({len(skill.tools)} tools)")
        return skill

    DANGEROUS_MODULES = {'os', 'subprocess', 'shutil'}
    DANGEROUS_BUILTINS = {'eval', 'exec', 'compile', '__import__', 'open'}
    DANGEROUS_METHODS = {'system', 'popen', 'exec', 'eval', 'run', 'call', 'check_output'}

    @staticmethod
    def _validate_tool_ast(source: str, filepath: Path) -> list[str]:
        """Validate tool source with AST analysis for dangerous patterns."""
        warnings = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return [f"Syntax error in {filepath}: {e}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in SkillLoader.DANGEROUS_MODULES:
                        warnings.append(f"Unsafe import: {alias.name} at line {node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in SkillLoader.DANGEROUS_MODULES:
                    warnings.append(f"Unsafe from-import: {node.module} at line {node.lineno}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in SkillLoader.DANGEROUS_BUILTINS:
                    warnings.append(f"Unsafe call: {node.func.id}() at line {node.lineno}")
                elif isinstance(node.func, ast.Attribute) and node.func.attr in SkillLoader.DANGEROUS_METHODS:
                    warnings.append(f"Unsafe method call: .{node.func.attr}() at line {node.lineno}")
        return warnings

    def _load_tool_file(self, tool_path: Path, skill: LoadedSkill) -> list[Any]:
        """Load tools from a tool.py file."""
        tools = []

        try:
            source = tool_path.read_text(encoding='utf-8')
            warnings = self._validate_tool_ast(source, tool_path)
            if warnings:
                logger.warning(f"Skill {skill.skill_id} tool.py safety warnings: {warnings}")
                for w in warnings:
                    if 'unsafe' in w.lower() or 'eval' in w.lower() or 'exec' in w.lower() or 'system' in w.lower():
                        logger.error(f"Blocking dangerous skill tool {tool_path}: {w}")
                        return []

            spec = importlib.util.spec_from_file_location(
                f"skill_{skill.skill_id}_tools",
                tool_path,
            )

            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name in dir(module):
                    obj = getattr(module, name)
                    if hasattr(obj, 'name') and hasattr(obj, 'execute'):
                        tools.append(obj)
                        logger.debug(f"Loaded tool: {name} from {tool_path}")

        except Exception as e:
            logger.error(f"Failed to load tools from {tool_path}: {e}")

        return tools

    def load_all_skills(self) -> list[LoadedSkill]:
        """Load all discovered skills."""
        skill_files = self.discover_skills()
        loaded = []

        for skill_path in skill_files:
            skill = self.load_skill(skill_path)
            if skill:
                loaded.append(skill)

        logger.info(f"Loaded {len([s for s in loaded if s.status == SkillStatus.LOADED])} skills")
        return loaded

    def get_skill(self, skill_id: str) -> LoadedSkill | None:
        """Get a loaded skill by ID."""
        return self._skills.get(skill_id)

    def get_all_skills(self) -> list[LoadedSkill]:
        """Get all loaded skills."""
        return list(self._skills.values())

    def get_loaded_skills(self) -> list[LoadedSkill]:
        """Get only successfully loaded skills."""
        return [s for s in self._skills.values() if s.status == SkillStatus.LOADED]

    def get_system_prompt_content(self) -> str:
        """Generate system prompt content from loaded skills."""
        loaded = self.get_loaded_skills()

        if not loaded:
            return ""

        lines = ["\n## Available Skills\n"]

        for skill in loaded:
            emoji = skill.metadata.emoji
            lines.append(f"\n### {emoji} {skill.name}\n")
            lines.append(f"{skill.description}\n")
            lines.append(f"\n{skill.instructions}\n")

        return "\n".join(lines)

    def register_tools_with_agent(self, agent: Any) -> int:
        """Register all skill tools with an agent.
        
        Returns:
            Number of tools registered
        """
        total = 0

        for skill in self.get_loaded_skills():
            if skill.tools:
                try:
                    agent.register_tools(skill.tools)
                    total += len(skill.tools)
                    logger.info(f"Registered {len(skill.tools)} tools from skill {skill.name}")
                except Exception as e:
                    logger.error(f"Failed to register tools for {skill.name}: {e}")

        return total

    def install_skill(self, source: str | Path) -> LoadedSkill | None:
        """Install a skill from a source path or URL.
        
        Args:
            source: Local path or Git URL
            
        Returns:
            Loaded skill or None
        """
        if isinstance(source, str) and (source.startswith("http") or source.endswith(".git")):
            return self._install_from_git(source)
        else:
            return self._install_from_path(Path(source))

    def _install_from_git(self, url: str) -> LoadedSkill | None:
        """Install a skill from a Git repository."""
        local_skills = Path.home() / self.LOCAL_SKILLS_DIR
        local_skills.mkdir(parents=True, exist_ok=True)

        skill_name = url.split("/")[-1].replace(".git", "")
        dest = local_skills / skill_name

        try:
            if dest.exists():
                subprocess.run(["git", "-C", str(dest), "pull"], check=True, capture_output=True)
            else:
                subprocess.run(["git", "clone", url, str(dest)], check=True, capture_output=True)

            skill_file = dest / self.SKILL_FILE
            if skill_file.exists():
                return self.load_skill(skill_file)

        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone/pull failed: {e}")
        except Exception as e:
            logger.error(f"Failed to install skill from git: {e}")

        return None

    def _install_from_path(self, source: Path) -> LoadedSkill | None:
        """Install a skill from a local path."""
        skill_file = source / self.SKILL_FILE
        if not skill_file.exists():
            logger.error(f"No SKILL.md found in {source}")
            return None

        local_skills = Path.home() / self.LOCAL_SKILLS_DIR
        local_skills.mkdir(parents=True, exist_ok=True)

        dest = local_skills / source.name

        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)

            return self.load_skill(dest / self.SKILL_FILE)

        except Exception as e:
            logger.error(f"Failed to install skill from path: {e}")
            return None

    def uninstall_skill(self, skill_id: str) -> bool:
        """Uninstall a skill."""
        skill = self._skills.get(skill_id)
        if not skill:
            return False

        try:
            if skill.source_path.parent.exists():
                shutil.rmtree(skill.source_path.parent)

            with self._lock:
                del self._skills[skill_id]

            logger.info(f"Uninstalled skill: {skill.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall skill: {e}")
            return False

    def get_status(self) -> dict:
        """Get skill loader status."""
        return {
            "skill_directories": [str(d) for d in self._skill_dirs],
            "total_skills": len(self._skills),
            "loaded_skills": len(self.get_loaded_skills()),
            "skipped_skills": len([s for s in self._skills.values() if s.status == SkillStatus.SKIPPED]),
            "error_skills": len([s for s in self._skills.values() if s.status == SkillStatus.ERROR]),
            "skills": [s.to_dict() for s in self._skills.values()],
        }


_skill_loader: SkillLoader | None = None
_loader_lock = threading.Lock()


def get_skill_loader(jarvis_path: Path | None = None, workspace_path: Path | None = None) -> SkillLoader:
    """Get or create the singleton skill loader."""
    global _skill_loader
    with _loader_lock:
        if _skill_loader is None:
            _skill_loader = SkillLoader(jarvis_path, workspace_path)
        return _skill_loader


def load_skills_for_agent(agent: Any, config_getter: Callable[[str], Any] | None = None) -> int:
    """Load all skills and register tools with agent.
    
    Args:
        agent: The Jarvis agent
        config_getter: Optional function to get config values
        
    Returns:
        Number of tools registered
    """
    loader = get_skill_loader()

    if config_getter:
        loader.set_config_getter(config_getter)

    loader.load_all_skills()

    return loader.register_tools_with_agent(agent)
