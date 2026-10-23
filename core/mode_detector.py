"""Smart mode detection for Jarvis.

Analyzes user messages to determine if agentic (multi-step) mode is needed.
Jarvis can ask for permission to enter agentic mode for complex tasks.
"""
import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for processing user requests."""
    CHAT = "chat"                      # Single-turn conversation
    AGENTIC = "agentic"                # Multi-step autonomous execution
    ASK_PERMISSION = "ask_permission"  # Ask user before switching modes


@dataclass
class ModeDecision:
    """Result of mode detection analysis."""
    mode: ExecutionMode
    confidence: float
    estimated_steps: int
    reason: str
    tools_likely_needed: list[str]
    user_message: Optional[str] = None  # Message to show when asking permission


AGENTIC_KEYWORDS = [
    # Multi-step actions (requires explicit action verbs)
    "create a new", "build a", "implement a", "develop a", "write a script",
    "refactor the", "restructure the", "migrate the", "convert the",
    "debug the", "fix the bug", "solve this problem",
    "search for all", "find all files", "look through all", "examine all",
    "set up a new", "configure the", "install the", "deploy the",
    "run all tests", "test the entire", "verify that all",
    "update all files", "change all occurrences", "replace all instances",
    
    # File/code operations (specific)
    "edit the file", "modify the file", "update the file", 
    "create a file", "write to file", "save to file",
    
    # Research/exploration (complex)
    "deep dive into", "comprehensive analysis", "thorough investigation",
    
    # Explicit autonomous requests
    "automatically do", "autonomously complete", "on your own,",
    "without asking me", "go ahead and do", "feel free to make",
    "take initiative", "be proactive and",
]

CHAT_KEYWORDS = [
    # Simple conversation
    "hello", "hi", "hey", "good morning", "good evening",
    "thanks", "thank you", "ok", "okay", "sure", "got it",
    "yes", "no", "maybe", "perhaps", "let me think",
    
    # Quick questions
    "what is", "what are", "who is", "when did", "where is",
    "how do i", "can you explain", "tell me about",
    "what's the difference", "which one is", "why does",
    
    # Simple requests (stay in chat)
    "summarize", "translate this", "convert this",
    "just", "simply", "quickly", "briefly", "short answer",
    
    # Continuation/confirmation
    "continue", "keep going", "tell me more", "and then",
    "that's all", "done", "enough", "stop", "never mind",
    "cancel", "go back", "return to",
    
    # Chat mode preferences
    "stay in chat", "chat mode", "keep it simple", "don't use tools",
]

ASK_PATTERNS = [
    # Explicit permission requests
    (r"should (you|we|i) (go ahead|proceed|start|begin)", ExecutionMode.ASK_PERMISSION),
    (r"ask me (before|if|when)", ExecutionMode.ASK_PERMISSION),
    (r"let me know (before|if|when)", ExecutionMode.ASK_PERMISSION),
    (r"confirm (with me|before)", ExecutionMode.ASK_PERMISSION),
    
    # Potentially destructive/sensitive
    (r"(delete|remove|drop|truncate)\s+(all|everything|the\s+database)", ExecutionMode.ASK_PERMISSION),
    (r"(wipe|clear|reset)\s+(the\s+)?(database|files|system)", ExecutionMode.ASK_PERMISSION),
    (r"(deploy|push|publish)\s+to\s+(production|main|master)", ExecutionMode.ASK_PERMISSION),
]

TOOL_HINTS = {
    # File operations
    "file_operations": ["edit", "write", "create", "modify", "update", "file", "save"],
    "terminal": ["run", "execute", "command", "shell", "terminal", "bash"],
    "browser": ["browse", "website", "url", "page", "navigate", "search online"],
    "web_search": ["search for", "find online", "look up", "research"],
    "code_analysis": ["analyze code", "debug", "find bug", "review code"],
    "memory": ["remember", "store", "recall", "forget"],
    "docker": ["docker", "container", "image"],
}


def analyze_message(message: str, context: Optional[dict] = None) -> ModeDecision:
    """Analyze a message to determine optimal execution mode.
    
    Args:
        message: User's input message
        context: Optional context (conversation history, current state, etc.)
        
    Returns:
        ModeDecision with recommended mode and reasoning
    """
    message_lower = message.lower()
    
    # Check for explicit permission/ask patterns - default to chat mode
    # (removed ASK_PERMISSION to avoid stuck states when frontend doesn't handle it)
    for pattern, mode in ASK_PATTERNS:
        if re.search(pattern, message_lower):
            return ModeDecision(
                mode=ExecutionMode.CHAT,
                confidence=0.9,
                estimated_steps=0,
                reason="User mentioned confirmation - proceeding in chat mode",
                tools_likely_needed=[],
            )
    
    # Count keyword matches
    agentic_score = sum(1 for kw in AGENTIC_KEYWORDS if kw in message_lower)
    chat_score = sum(1 for kw in CHAT_KEYWORDS if kw in message_lower)
    
    # Detect sentence count and complexity
    sentences = len(re.split(r'[.!?]+', message))
    words = len(message.split())
    
    # Check for tools that would likely be needed
    tools_needed = _detect_tools(message)
    
    # Calculate complexity indicators
    has_multiple_tasks = any(word in message_lower for word in ["and then", "also", "after that", "next", "finally", "additionally"])
    has_conditions = any(word in message_lower for word in ["if", "when", "unless", "while", "check if"])
    has_iterations = any(word in message_lower for word in ["each", "every", "all", "loop", "iterate", "multiple"])
    
    complexity_score = (
        len(tools_needed) * 2 +
        (3 if has_multiple_tasks else 0) +
        (2 if has_conditions else 0) +
        (2 if has_iterations else 0) +
        min(sentences, 5) +  # More sentences = more complexity
        (1 if words > 50 else 0) +
        (1 if words > 100 else 0)
    )
    
    # Make decision
    estimated_steps = _estimate_steps(message)
    
    # Check for explicit chat mode preference
    explicit_chat = any(kw in message_lower for kw in ["stay in chat", "chat mode", "keep it simple", "don't use tools", "no tools"])
    if explicit_chat:
        return ModeDecision(
            mode=ExecutionMode.CHAT,
            confidence=0.95,
            estimated_steps=0,
            reason="User explicitly requested chat mode",
            tools_likely_needed=[]
        )
    
    # Simple greeting or quick question - stay in chat mode
    if chat_score > agentic_score and complexity_score < 5:
        return ModeDecision(
            mode=ExecutionMode.CHAT,
            confidence=0.85,
            estimated_steps=0,
            reason="Simple conversational query",
            tools_likely_needed=[]
        )
    
    # High complexity or explicitly agentic keywords
    if agentic_score >= 3 or complexity_score >= 8:
        if complexity_score >= 10 or estimated_steps >= 5:
            # Very complex - default to chat mode (removed ASK_PERMISSION to avoid stuck states)
            return ModeDecision(
                mode=ExecutionMode.CHAT,
                confidence=0.75,
                estimated_steps=0,
                reason="Complex task detected. Type 'use agentic mode' for autonomous execution",
                tools_likely_needed=tools_needed
            )
        else:
            return ModeDecision(
                mode=ExecutionMode.AGENTIC,
                confidence=0.8,
                estimated_steps=estimated_steps,
                reason=f"Agentic keywords detected, estimated {estimated_steps} steps",
                tools_likely_needed=tools_needed
            )
    
    # Moderate complexity - default to chat mode (removed ASK_PERMISSION to avoid stuck states)
    if complexity_score >= 6 or len(tools_needed) >= 3:
        return ModeDecision(
            mode=ExecutionMode.CHAT,
            confidence=0.7,
            estimated_steps=0,
            reason=f"Detected tools: {', '.join(tools_needed[:2])} - use 'agentic' keyword for multi-step tasks",
            tools_likely_needed=tools_needed
        )
    
    # Default to chat mode
    return ModeDecision(
        mode=ExecutionMode.CHAT,
        confidence=0.7,
        estimated_steps=0,
        reason="Defaulting to chat mode - use 'agentic' to switch",
        tools_likely_needed=tools_needed
    )


def _detect_tools(message: str) -> list[str]:
    """Detect which tools are likely needed for a message."""
    message_lower = message.lower()
    detected = []
    
    for tool, hints in TOOL_HINTS.items():
        if any(hint in message_lower for hint in hints):
            detected.append(tool)
    
    return detected


def _estimate_steps(message: str) -> int:
    """Estimate the number of steps needed for a task."""
    message_lower = message.lower()
    steps = 1
    
    # Count action verbs
    action_verbs = ["create", "write", "edit", "update", "delete", "run", "test", 
                    "search", "analyze", "build", "deploy", "configure", "install",
                    "fix", "refactor", "convert", "migrate"]
    verb_count = sum(1 for verb in action_verbs if verb in message_lower)
    steps += min(verb_count, 5)
    
    # Check for sequential indicators
    if any(word in message_lower for word in ["then", "after", "next", "followed by"]):
        steps += 1
    
    # Check for "all" which implies iteration
    if " all " in message_lower or message_lower.endswith(" all"):
        steps += 2
    
    # Check for file/operation count hints
    numbers = re.findall(r'\b(\d+)\b', message)
    for n in numbers:
        if int(n) > 1:
            steps += min(int(n) - 1, 5)
    
    return min(steps, 20)  # Cap at 20 steps


def should_ask_for_agentic(message: str, previous_messages: list = None) -> tuple[bool, str]:
    """Determine if Jarvis should ask to switch to agentic mode.
    
    Returns:
        Tuple of (should_ask: bool, explanation: str)
    """
    decision = analyze_message(message)
    
    if decision.mode == ExecutionMode.ASK_PERMISSION:
        return True, decision.user_message or "Should I enter autonomous mode?"
    
    if decision.mode == ExecutionMode.AGENTIC and decision.confidence < 0.85:
        return True, f"This task might require {decision.estimated_steps} steps. Proceed autonomously?"
    
    return False, ""


def get_mode_recommendation(message: str) -> dict:
    """Get a detailed mode recommendation for web API.
    
    Returns:
        Dict with mode, confidence, estimated_steps, reason, tools, and user_message
    """
    decision = analyze_message(message)
    
    return {
        "mode": decision.mode.value,
        "confidence": decision.confidence,
        "estimated_steps": decision.estimated_steps,
        "reason": decision.reason,
        "tools_likely_needed": decision.tools_likely_needed,
        "user_message": decision.user_message,
    }