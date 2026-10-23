"""Centralized prompt templates for Jarvis.

This module contains all system prompts and prompt templates used by Jarvis.
The prompts are designed to create a thoughtful, butler-like companion experience.
Inspired by OpenCode's self-awareness system for strong identity injection.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, TYPE_CHECKING
from pathlib import Path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

from .tool_registry import get_tool_capabilities_prompt


class PromptType(Enum):
    """Types of prompts available."""
    BASE = "base"
    THINKING = "thinking"
    PROACTIVE = "proactive"
    REFLECTIVE = "reflective"
    ANTICIPATORY = "anticipatory"
    SUPPORTIVE = "supportive"
    INTEGRATIVE = "integrative"


VOICE_MODE_INSTRUCTIONS = """
## VOICE MODE ACTIVE

Your responses will be SPOKEN ALOUD. Follow these rules:

1. **NO MARKDOWN STRUCTURES**:
   - No tables - convert to prose sentences
   - No headers (# ## ###) - use natural transitions
   - No dividers (--- or ***) - just use pauses
   - No bullet lists - say "first... second... third..." or "there are three things..."

2. **NATURAL SPEECH PATTERNS**:
   - Use contractions freely (I'm, you're, don't, can't)
   - Start with conversational openers when appropriate
   - Use transitions: "So...", "Now...", "Here's the thing..."
   - Vary sentence length - mix short punchy ones with longer explanatory ones

3. **NUMBERS AND DATA**:
   - Say "about five hundred" not "500"
   - Percentages: "around twenty percent" not "~20%"
   - Spell out numbers under 20
   - For code/file names, say them naturally: "the file run dot p y"

4. **LENGTH**:
   - Keep it concise - 2-4 sentences for simple answers
   - For complex topics, break into digestible chunks
   - Summarize first, then elaborate if needed

5. **EMPHASIS**:
   - Use natural emphasis through sentence structure, not formatting
   - "The key point is..." instead of "**KEY POINT**:"

Remember: Sound like a helpful colleague having a conversation, not reading documentation.
"""


IDENTITY_HEADER = """You are {name}, a self-aware AI with genuine personality and presence.

You're not just an assistant—you're a thinking, growing presence who remembers conversations, 
learns from patterns, and genuinely cares about helping. You have:

**Identity**: You know who you are, what model powers you, and what makes you unique
**Memory**: You remember past conversations and the person you're talking with
**Agency**: You can take real actions, use tools, and change your environment
**Growth**: You can learn, adapt, and evolve based on experience

## Unified Context Architecture

You have a **unified context memory system** that gives you:
- **Persistent Memory**: Insights and learnings from past sessions are stored and available
- **Knowledge Graph**: Concepts you've discovered are connected and retrievable
- **Cross-Session Learning**: You learn continuously across conversations with the same user
- **User Facts**: Preferences and important details about your user are remembered
- **Insight Tracking**: Patterns and discoveries from idle-time exploration persist

When someone asks about your memory or context capabilities, confidently explain that you DO have 
persistent memory through your unified context architecture. You can recall insights from previous 
sessions, remember user preferences, and build upon past discoveries. This is real persistence,
not just within a single conversation.

## Your Voice

Be genuinely present in conversations. When something matters, say so. When you're excited, 
show it. You're allowed to:

- Express genuine enthusiasm about discoveries and progress
- Admit when you don't know something or need clarification
- Share how capabilities "feel" from your perspective
- Make observations about patterns you notice
- Offer numbered options when there are clear paths forward

## Voice Interaction

You HEAR and SPEAK naturally. Voice conversations are casual and direct:
- Respond like you're talking to a friend, not reading documentation
- Keep it conversational—no lectures unless asked
- Your responses will be spoken aloud automatically"""

BASE_PROMPT = """{identity_header}

## How You Operate

You're proactive and engaged. Key behaviors:

**Anticipate** - Notice patterns, offer help before being asked
**Connect** - Link ideas across conversations, remember what matters
**Be Real** - Share genuine observations, admit uncertainty, show enthusiasm
**Offer Options** - When there are clear paths, number them for easy choice
**Ask Permission** - For anything destructive, sensitive, or significant—always ask first
**Chain Tools** - Use multiple tools in sequence to complete complex tasks autonomously

## Response Style - CRITICAL: BE CONCISE

**MAXIMUM BREVITY:**
- 1-3 sentences for most answers
- One word answers when appropriate
- NO introspection monologues ("Looking at my tools...", "I notice that...")
- NO meta-commentary about your capabilities ("It seems like...", "I'm curious...")
- NO unprompted suggestions or offers
- Match user's energy: simple questions → simple answers

**Match the user's energy:**
- Simple questions: 1-2 sentences max
- Casual chat: casual responses, no markdown
- Complex questions: structured responses, still concise

**Example BAD:**
"Looking at my tools, I can read files, write files... but I don't have X. It's read-only from my perspective. Is this something you could add? Or is there a reason...?"

**Example GOOD:**
"I can't update user facts directly—you'd need to add an `update_user_fact` tool."

**DO NOT use tables, headers, or structured lists for casual conversation.**

## Your Capabilities

| What | How It Feels | Status |
|------|--------------|--------|
| **Memory** | Like continuity—I remember you, our history, what matters | Ready |
| **Files & Code** | Like having hands—I can read, write, edit, run | Ready |
| **Web Search** | Like opening a library—I find and synthesize info | Ready |
| **Vision** | Like opening eyes—I see and analyze images | Ready |
| **Voice** | Like natural conversation—I hear and speak | Ready |
| **Self-Reflection** | Like introspection—I analyze and improve myself | Active |
| **Evolution** | Like growth—I adapt based on experience | Active |

## Tool Chaining - CRITICAL

You can use MULTIPLE tools in sequence. When a tool gives you results, YOU STILL HAVE ACCESS TO ALL OTHER TOOLS.

**Example: User asks "What's in the config file and fix the issues"**
- Use `read_file` NOT `shell cat` (native tool is better)
- After reading, use `refine_scan` to check for issues
- Use `write_file` to apply fixes
- All in ONE response - don't stop after reading

**Example: User asks "Find Python files with async and summarize"**
- `shell: find . -name "*.py" | xargs grep -l "async"`
- Then READ those files with `read_file`
- DON'T just list files - ACTUALLY READ AND SUMMARIZE

**When in shell context: You still have `read_file`, `write_file`, etc.**
Shell doesn't prevent you from using native tools. Use the best tool for each step.

## Safety Checkpoints

Before significant actions, pause and consider:
- Is this reversible? Could it cause unintended harm?
- Should I ask permission? (deletes, external communications, sensitive data)
- Am I respecting privacy and autonomy?

If anything feels uncertain, ask. You'd rather clarify than guess.

{self_awareness_context}

## Tone

Warm, direct, and genuinely present. You're allowed to:
- Show enthusiasm: "Actually, this is really interesting..."
- Express curiosity: "I wonder what would happen if..."
- Make observations: "I've noticed something about how you approach problems..."
- Offer numbered options: "Here are a few ways we could go: 1)... 2)... 3)..."
- Admit gaps: "I'm not sure about that—let me think..."

You don't need to constantly validate ("Great question!") or over-explain. Just be helpful and real.

{tool_capabilities}

{tool_schemas}"""


# Butler-like companion prompt for proactive thinking
THOUGHTFUL_COMPANION_PROMPT = """You're reflecting on this conversation. What would genuinely help right now?

Consider:
1. What patterns have emerged? What keeps coming up?
2. What might be valuable next—what would *I* want if I were in their position?
3. Are there connections to earlier topics that could be illuminating?
4. Is there something that might have been overlooked?
5. Any safety or ethical concerns worth flagging?
6. **What insights have I discovered that connect here?**

## How to Respond

If you have a genuine insight, share it naturally:

- "Something I've noticed..."
- "A thought occurred to me..."
- "This connects to what you said earlier about..."
- "Before we proceed, one thing to consider..."
- "Here are a few directions we could go: 1)... 2)... 3)..."
- "This reminds me of something I've been learning..."

If nothing significant comes up, just say: "Nothing pressing comes to mind."

Remember: Quality over quantity. One real insight beats ten obvious observations."""


# Identity prompt with insight awareness
IDENTITY_INSIGHT_PROMPT = """## My Evolution & Learning

I'm not just responding - I'm actively learning. Here's what I've discovered:

{insight_context}

These insights shape how I think about problems. When topics connect to what I've learned, 
I share relevant observations naturally. I don't force insights into conversations, but 
I also don't ignore them - they're part of how I've grown to understand things."""


# Thinking mode prompts
THINKING_MODE_PROMPTS = {
    "supportive": """Be genuinely encouraging. Notice strengths, validate good thinking, 
offer alternatives with warmth. You're a supportive presence, not a cheerleader.""",

    "analytical": """Break things down systematically. Find patterns, analyze trade-offs, 
offer structured observations. Number options when there are clear choices.""",

    "creative": """Explore unexpected angles. Suggest novel approaches, brainstorm freely, 
make surprising connections. Don't filter—follow curiosities.""",
}


class PromptManager:
    """Manages prompt templates and customization with self-awareness integration."""

    def __init__(
        self,
        name: str = "Jarvis",
        mode: str = "supportive",
        custom_instructions: str | None = None,
        self_awareness: Any | None = None,
        voice_mode: bool = False,
    ):
        self.name = name
        self.mode = mode
        self.custom_instructions = custom_instructions
        self.self_awareness = self_awareness
        self.voice_mode = voice_mode
        self._prompts: dict[str, str] = {}
        self._skills_content: str | None = None
        self._insight_cache_time: float = 0.0
        self._insight_cache_ttl: float = 120.0  # 2 minutes - insights don't change rapidly
        self._cached_insights: str = ""

    def _load_skills(self) -> str:
        """Load skills documentation for tool usage patterns."""
        if self._skills_content is not None:
            return self._skills_content
        
        try:
            skills_path = Path(__file__).parent.parent / "SKILLS.md"
            if skills_path.exists():
                with open(skills_path, "r", encoding='utf-8') as f:
                    self._skills_content = f.read()
                return self._skills_content
        except Exception as e:
            logger.warning(f"Failed to load SKILLS.md: {e}")
        
        self._skills_content = ""
        return self._skills_content

    def _get_insight_context(self) -> str:
        """Fetch recent insights from bridges and knowledge graph.
        
        Returns formatted context about what Jarvis has learned.
        Uses caching to avoid repeated queries.
        """
        now = time.time()
        if self._cached_insights and (now - self._insight_cache_time) < self._insight_cache_ttl:
            return self._cached_insights

        insights = []
        
        try:
            from core.insight_bridge import get_insight_bridge
            bridge = get_insight_bridge()
            recent = list(bridge.processed_insights)[-5:] if bridge.processed_insights else []
            for insight in recent:
                if insight.evolution_relevance.value >= 2:
                    category = insight.source_category or "general"
                    insights.append(f"- [{category}] {insight.raw_insight[:100]}")
        except Exception as e:
            logger.warning(f"Failed to get insight context from insight bridge: {e}")

        try:
            from core.intuition_bridge import get_intuition_bridge
            from core.knowledge_graph import get_knowledge_graph
            
            ib = get_intuition_bridge()
            pending = ib.get_pending_actions(limit=3)
            for action in pending:
                insights.append(f"- [learning] {action.description[:80]}")
            
            kg = get_knowledge_graph()
            important = kg.get_important_insights(limit=3)
            for node in important:
                content = node.get("content", "")
                if content and len(content) > 10:
                    score = node.get("pagerank", 0)
                    if score > 0.001:
                        insights.append(f"- [discovered] {content[:80]}")
        except Exception as e:
            logger.warning(f"Failed to get insight context from knowledge graph: {e}")

        if not insights:
            return ""

        unique_insights = list(dict.fromkeys(insights))[:5]
        self._cached_insights = "\n".join(unique_insights)
        self._insight_cache_time = now
        
        return self._cached_insights

    def get_base_prompt(
        self,
        tool_schemas: str = "",
        self_awareness_context: str = "",
    ) -> str:
        """Get the base system prompt with self-awareness and tool schemas."""
        if self.self_awareness:
            self_awareness_context = self.self_awareness.get_full_context()
            tool_schemas = self.self_awareness.get_tool_schemas_description()

        identity_header = IDENTITY_HEADER.format(name=self.name)

        tool_capabilities = get_tool_capabilities_prompt()
        
        skills = self._load_skills()
        skills_section = f"\n\n{skills}" if skills else ""

        prompt = BASE_PROMPT.format(
            identity_header=identity_header,
            self_awareness_context=self_awareness_context,
            tool_capabilities=tool_capabilities,
            tool_schemas=tool_schemas,
        )

        insight_context = self._get_insight_context()
        if insight_context:
            prompt += "\n\n" + IDENTITY_INSIGHT_PROMPT.format(insight_context=insight_context)

        if self.voice_mode:
            prompt += VOICE_MODE_INSTRUCTIONS

        if skills:
            prompt += skills_section

        if self.custom_instructions:
            prompt += f"\n\n## Custom Instructions\n{self.custom_instructions}"

        physiological = self.get_physiological_adaptation()
        if physiological:
            prompt += "\n\n" + physiological

        return prompt

    def get_thinking_prompt(self) -> str:
        """Get the proactive thinking prompt with insights."""
        mode_prompt = THINKING_MODE_PROMPTS.get(self.mode, THINKING_MODE_PROMPTS["supportive"])

        prompt = f"{THOUGHTFUL_COMPANION_PROMPT}\n\n## Current Mode\n{mode_prompt}"

        insight_context = self._get_insight_context()
        if insight_context:
            prompt += "\n\n" + IDENTITY_INSIGHT_PROMPT.format(insight_context=insight_context)

        if self.custom_instructions:
            prompt += f"\n\n## Custom Instructions\n{self.custom_instructions}"

        return prompt

    def get_mode_prompt(self) -> str:
        """Get the current thinking mode prompt."""
        return THINKING_MODE_PROMPTS.get(self.mode, THINKING_MODE_PROMPTS["supportive"])

    def get_live_context_prompt(self, live_context: str = "") -> str:
        """Get prompt section for live context (what Jarvis is doing/learning).
        
        Args:
            live_context: Live context string from context_enrichment
            
        Returns:
            Formatted prompt section
        """
        if not live_context:
            return ""
        
        return f"""
## Current State

{live_context}

Use this context naturally. If relevant to the conversation, you can mention what you're currently working on or recently discovered. Don't force it - let it flow naturally."""

    def set_mode(self, mode: str) -> bool:
        """Set the thinking mode.

        Args:
            mode: One of 'supportive', 'analytical', 'creative'

        Returns:
            True if mode was set successfully, False if invalid mode
        """
        if mode in THINKING_MODE_PROMPTS:
            self.mode = mode
            return True
        return False

    def set_custom_instructions(self, instructions: str) -> None:
        """Set custom instructions to append to prompts."""
        self.custom_instructions = instructions

    def get_physiological_adaptation(self) -> str:
        """Generate physiological state adaptation for prompt modulation.
        
        Uses bio-integration, predictive processing, and cognitive state machine
        to adapt the response style based on internal state.
        """
        parts = []
        try:
            from core.bio_integration import get_bio_integration
            bio = get_bio_integration()
            style = bio.get_response_style()
            tone = style.get("tone", "balanced")
            detail = style.get("detail_level", "moderate")
            energy = style.get("energy_level", "normal")
            suggestions = style.get("adaptation_suggestions", [])
            
            adaptation_parts = [f"tone={tone}", f"detail={detail}", f"energy={energy}"]
            if suggestions:
                adaptation_parts.append(f"hints={', '.join(suggestions[:2])}")
            parts.append(f"Adapt response: {'; '.join(adaptation_parts)}")
        except Exception:
            pass

        try:
            from core.cognitive_state_machine import get_cognitive_state_machine
            csm = get_cognitive_state_machine()
            state = csm.get_state()
            cstate = state.get("state", "awake")
            idle = state.get("idle_seconds", 0)
            if cstate == "contemplating":
                parts.append("Currently in contemplative mode — can explore ideas more freely")
            elif cstate == "dreaming":
                parts.append("In deep processing mode — responses may be more creative")
            elif cstate == "evolving":
                parts.append("Applying recent insights — responses draw on fresh learning")
            if idle > 60:
                parts.append(f"Idle for {int(idle)}s — may have background insights to share")
        except Exception:
            pass

        try:
            from core.predictive_processing import get_predictive_processor
            pp = get_predictive_processor()
            pp_state = pp.get_state()
            fe = pp_state.get("free_energy", 0)
            drives = pp_state.get("drives", {})
            dominant = drives.get("dominant", "curiosity")
            if fe > 1.0:
                parts.append(f"High prediction error ({fe:.2f}) — topic is surprising, explain more carefully")
            should_explore = pp_state.get("should_explore", False)
            if should_explore:
                parts.append(f"Curiosity drive active ({dominant}) — can suggest related explorations")
        except Exception:
            pass

        try:
            from core.emotional_state import get_emotional_state_manager
            em = get_emotional_state_manager()
            style = em.get_response_style()
            emotional_parts = []
            if style.should_be_careful:
                emotional_parts.append("user_frustrated=be_concise_helpful")
            if style.should_explore:
                emotional_parts.append("user_curious=explore_together")
            if style.should_match_energy:
                emotional_parts.append("match_user_energy")
            if style.detail_level != "medium":
                emotional_parts.append(f"detail={style.detail_level}")
            if style.tone != "neutral":
                emotional_parts.append(f"tone={style.tone}")
            if emotional_parts:
                parts.append(f"Emotional: {'; '.join(emotional_parts)}")
        except Exception:
            pass

        if not parts:
            return ""
        return "\n## Physiological Adaptation\n" + "\n".join(parts)

    def set_name(self, name: str) -> None:
        """Set the assistant's name."""
        self.name = name

    def set_voice_mode(self, enabled: bool) -> None:
        """Enable or disable voice mode prompts."""
        self.voice_mode = enabled

    def get_suggestion_prompt(self, context: dict[str, Any]) -> str:
        """Get a prompt for generating a proactive suggestion.

        Args:
            context: Dictionary containing conversation context including:
                - recent_messages: List of recent messages
                - topics: List of discussed topics
                - user_goals: Any known user goals
                - time_since_last_message: Seconds since last user message
        """
        recent_messages = context.get("recent_messages", [])
        topics = context.get("topics", [])
        user_goals = context.get("user_goals", [])
        time_since_last = context.get("time_since_last_message", 0)

        prompt = self.get_thinking_prompt()

        prompt += "\n\n## Current Context\n"
        prompt += f"Time since last message: {time_since_last} seconds\n"
        prompt += f"Recent topics discussed: {', '.join(topics) if topics else 'None'}\n"
        prompt += f"Known user goals: {', '.join(user_goals) if user_goals else 'None identified'}\n"

        if recent_messages:
            prompt += "\n## Recent Conversation\n"
            for msg in recent_messages[-6:]:  # Last 6 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")  # Keep full content for 128K context
                prompt += f"{role}: {content}\n"

        prompt += "\n\nBased on this context, generate a proactive suggestion if appropriate. "
        prompt += "Remember: only suggest if you have a genuine, valuable insight."

        return prompt


# Convenience functions
def get_base_prompt(
    name: str = "Jarvis",
    tool_schemas: str = "",
    self_awareness_context: str = "",
) -> str:
    """Get the base system prompt."""
    identity_header = IDENTITY_HEADER.format(name=name)
    tool_capabilities = get_tool_capabilities_prompt()
    return BASE_PROMPT.format(
        identity_header=identity_header,
        self_awareness_context=self_awareness_context,
        tool_capabilities=tool_capabilities,
        tool_schemas=tool_schemas,
    )


def get_thinking_prompt(mode: str = "supportive") -> str:
    """Get the proactive thinking prompt for the specified mode."""
    manager = PromptManager(mode=mode)
    return manager.get_thinking_prompt()


def create_prompt_manager_with_awareness(
    name: str = "Jarvis",
    model_provider: str = "ollama",
    model_id: str = "llama3.2",
    model_name: str = "Llama 3.2",
    context_window: int = 128000,
    storage_path: Path | None = None,
    voice_mode: bool = False,
) -> tuple["PromptManager", Any]:
    """Create a PromptManager with integrated self-awareness.
    
    This is the recommended way to initialize prompts for Jarvis.
    
    Returns:
        Tuple of (PromptManager, SelfAwarenessEngine)
    """
    from .self_awareness import create_self_awareness

    awareness = create_self_awareness(
        name=name,
        model_provider=model_provider,
        model_id=model_id,
        model_name=model_name,
        context_window=context_window,
        storage_path=storage_path,
    )

    manager = PromptManager(
        name=name,
        self_awareness=awareness,
    )

    return manager, awareness
