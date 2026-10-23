"""Tool Registry - Complete inventory of all Jarvis tools with capabilities.

This module provides a complete inventory of all tools available to Jarvis,
organized by category, with descriptions, parameters, and usage patterns.
"""
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    SYSTEM = "system"
    WEB = "web"
    PRODUCTIVITY = "productivity"
    AUTOMATION = "automation"
    SCREEN = "screen"
    BROWSER = "browser"
    TERMINAL = "terminal"
    LEARNING = "learning"
    EVOLUTION = "evolution"
    SECURITY = "security"
    MULTIMODAL = "multimodal"
    DATA = "data"
    META = "meta"
    REFINEMENT = "refinement"
    SPATIAL = "spatial"
    HOME = "home"
    MEMORY = "memory"


@dataclass
class ToolInfo:
    name: str
    category: ToolCategory
    description: str
    parameters: dict[str, str]
    requires_permission: bool = False
    examples: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)
    best_for: list[str] = field(default_factory=list)


TOOL_TREES: dict[str, list[str]] = {
    "file_operations": ["read_file", "write_file", "list_files", "delete_file", "shell", "terminal_execute"],
    "web_research": ["web_search", "web_fetch", "image_search", "image_download"],
    "screen_automation": ["see_screen", "find_element", "click_element", "type_text", "press_key", "hotkey", "scroll"],
    "browser_tasks": ["browser_launch", "browser_navigate", "browser_click", "browser_type", "browser_screenshot", "browser_get_content", "browser_evaluate", "browser_wait", "browser_list"],
    "terminal_tasks": ["terminal_execute", "terminal_cwd", "terminal_history", "computer_status", "install_package", "terminal_session", "terminal_mode"],
    "data_analysis": ["data_analysis", "sql_query", "statistics", "embed", "search_files"],
    "voice_audio": ["transcribe", "speech_tokenize", "process_speech", "speech_synthesize", "analyze_tone", "voice_stats", "identify_speaker", "register_speaker", "conversational_speech"],
    "learning_exploration": ["erdos_math", "daydream", "learning", "sandbox_explore", "sketch", "free_time_stats", "verified_solutions"],
    "self_improvement": ["evolution", "self_edit", "refine_scan", "refine_apply", "propose_gene", "evolution_status", "evolution_pathway", "gene_list", "gene_approval"],
    "knowledge_management": ["notes", "reminders", "index_files", "get_context", "knowledge_feed"],
    "security_defense": ["security_scan", "security_audit", "defense", "dlp_whitelist", "security_status", "security_report"],
    "meta_tools": ["list_tools", "get_tool_info", "get_capabilities", "find_related_tools", "self_reflect", "design_process", "review_process"],
    "home_automation": ["home_control", "home_status"],
    "physics_spatial": ["collision_detect", "particle_sim", "physics_analyze", "spatial_calc", "design_calc"],
    "art_philosophy": ["art_analysis", "philosophy_analysis"],
    "docker_n8n": ["docker", "docker_compose", "n8n"],
    "web_ui": ["web_ui_view", "web_ui_analyze", "web_ui_propose", "web_ui_screenshot", "web_ui_history", "mobile_ui_view"],
    "ui_sense": ["ui_sense", "ui_watch", "ui_visualize"],
    "system_sense": ["system_sense", "system_watch"],
    "tool_creation": ["tool_create", "tool_test", "tool_validate", "tool_request_deploy", "tool_list_pending", "tool_approve", "tool_reject", "tool_list_sandbox"],
    "insight_search": ["insight_search", "insight_heatmap", "insight_stats"],
    "image_vision": ["image_analyze", "image_read", "image_search", "image_download", "vision", "webcam"],
    "permission_management": ["approve_permission", "deny_permission", "view_permission_requests"],
}

BEST_TOOL_FOR_TASK: dict[str, tuple[str, str]] = {
    "search the web": ("web_search", "Returns top results with summaries from DuckDuckGo"),
    "fetch a webpage": ("web_fetch", "Extracts clean text content from any URL"),
    "read a file": ("read_file", "Safe file reading with path validation"),
    "write a file": ("write_file", "Protected writing with permission checks"),
    "execute shell command": ("terminal_execute", "Full terminal access with risk assessment"),
    "run a quick command": ("shell", "Sandboxed quick command execution"),
    "analyze the screen": ("see_screen", "OCR + UI element detection + context analysis"),
    "find a button": ("find_element", "Locates UI elements by text"),
    "click something": ("click_element", "Click by coordinates or element text"),
    "type text": ("type_text", "Types at current cursor position"),
    "press a key": ("press_key", "Single key press"),
    "keyboard shortcut": ("hotkey", "Multi-key combinations like ctrl+c"),
    "browse the web": ("browser_launch", "Start a browser session for automation"),
    "navigate to url": ("browser_navigate", "Go to any URL in the browser"),
    "take screenshot": ("browser_screenshot", "Capture the current page"),
    "analyze data": ("data_analysis", "Statistics, charts, and insights from data files"),
    "query database": ("sql_query", "Safe SELECT queries with connection pooling"),
    "search files semantically": ("search_files", "Vector-based search in indexed files"),
    "transcribe audio": ("transcribe", "Whisper-based transcription with NPU support"),
    "explore math": ("erdos_math", "Graph theory, number theory, Ramsey problems"),
    "self improve": ("evolution", "Genetic operations for capability improvement"),
    "scan code": ("refine_scan", "Find bugs, security issues, code smells"),
    "what is on screen": ("what_am_i_doing", "Natural language description of screen activity"),
    "check system status": ("computer_status", "CPU, memory, disk usage"),
    "sense the system": ("system_sense", "Comprehensive system awareness: resources, subsystems, processes"),
    "watch system health": ("system_watch", "Monitor for resource spikes and system events"),
    "create a tool": ("tool_create", "Design and implement a new tool for Jarvis"),
    "test a tool": ("tool_test", "Validate and test a tool in sandbox"),
    "list available tools": ("list_tools", "Show all tools organized by category"),
    "get tool details": ("get_tool_info", "Detailed info about a specific tool"),
    "create a note": ("notes", "Store thoughts, reminders, information"),
    "security audit": ("security_scan", "Scan for vulnerabilities and sensitive data"),
    "physics simulation": ("collision_detect", "GPU-accelerated collision detection"),
    "particle simulation": ("particle_sim", "GPU-accelerated particle systems"),
    "search insights": ("insight_search", "Search through Jarvis's accumulated knowledge and insights"),
    "find insights": ("insight_search", "Find relevant insights, patterns, or learned knowledge"),
    "insight statistics": ("insight_stats", "Get counts and metadata about accumulated insights"),
    "insight heatmap": ("insight_heatmap", "See most common words in insights"),
}


TOOL_REGISTRY: dict[str, ToolInfo] = {
    # === SYSTEM TOOLS ===
    "shell": ToolInfo(
        name="shell",
        category=ToolCategory.SYSTEM,
        description="Execute shell commands in a sandboxed environment",
        parameters={"command": "The shell command to execute"},
        requires_permission=True,
        examples=[
            "shell: ls -la",
            "shell: find . -name '*.py' | head -20",
        ],
        related_tools=["read_file", "write_file"],
    ),
    "read_file": ToolInfo(
        name="read_file",
        category=ToolCategory.SYSTEM,
        description="Read contents of a file",
        parameters={"path": "Path to the file"},
        requires_permission=False,
        examples=["read_file: /home/user/src/main.py"],
        related_tools=["write_file", "shell"],
    ),
    "write_file": ToolInfo(
        name="write_file",
        category=ToolCategory.SYSTEM,
        description="Write content to a file",
        parameters={"path": "Path to the file", "content": "Content to write"},
        requires_permission=True,
        examples=["write_file: /tmp/test.py with content"],
        related_tools=["read_file", "shell"],
    ),
    "notes": ToolInfo(
        name="notes",
        category=ToolCategory.SYSTEM,
        description="Store and retrieve notes/thoughts",
        parameters={"action": "add/list/search", "content": "Note content"},
        requires_permission=False,
    ),

    # === WEB TOOLS ===
    "web_search": ToolInfo(
        name="web_search",
        category=ToolCategory.WEB,
        description="Search the web for information. Safe search is enabled by default - use unsafe: true for unrestricted searches (requires permission)",
        parameters={"query": "Search query", "unsafe": "Set to true to bypass safe search (default false)"},
        requires_permission=False,
        examples=["web_search: latest developments in Rust", "web_search: adult content (unsafe: true)"],
        related_tools=["web_fetch"],
    ),
    "web_fetch": ToolInfo(
        name="web_fetch",
        category=ToolCategory.WEB,
        description="Fetch content from a URL. Safe URL checking is enabled by default - use unsafe: true for unrestricted URLs (requires permission)",
        parameters={"url": "URL to fetch", "unsafe": "Set to true to bypass safe URL checks (default false)"},
        requires_permission=False,
        examples=["web_fetch: https://example.com"],
        related_tools=["web_search"],
    ),
    "image_search": ToolInfo(
        name="image_search",
        category=ToolCategory.WEB,
        description="Search for images on the web. Safe search is enabled by default",
        parameters={"query": "Image search query", "safe_search": "Enable safe search (default true)"},
        requires_permission=False,
        examples=["image_search: cute cats"],
        related_tools=["image_download"],
    ),
    "image_download": ToolInfo(
        name="image_download",
        category=ToolCategory.WEB,
        description="Download an image from a URL. Safe URL checking is enabled by default",
        parameters={"url": "Image URL", "unsafe": "Set to true to bypass safety checks (default false)"},
        requires_permission=False,
        examples=["image_download: https://example.com/image.jpg"],
        related_tools=["image_search"],
    ),

    # === SCREEN VISION TOOLS ===
    "see_screen": ToolInfo(
        name="see_screen",
        category=ToolCategory.SCREEN,
        description="Capture and analyze the current screen content",
        parameters={"monitor": "Monitor number (default 1)"},
        requires_permission=True,
        examples=["see_screen: capture and describe what's visible"],
        related_tools=["find_element", "click_element"],
    ),
    "find_element": ToolInfo(
        name="find_element",
        category=ToolCategory.SCREEN,
        description="Find a UI element by text on screen",
        parameters={"text": "Text to search for"},
        requires_permission=True,
        examples=["find_element: 'Submit' button"],
        related_tools=["click_element", "see_screen"],
    ),
    "click_element": ToolInfo(
        name="click_element",
        category=ToolCategory.SCREEN,
        description="Click at coordinates or on a UI element",
        parameters={"x": "X coordinate", "y": "Y coordinate", "text": "Element text"},
        requires_permission=True,
        examples=["click_element: 'OK' button", "click_element: at (100, 200)"],
        related_tools=["find_element", "type_text"],
    ),
    "type_text": ToolInfo(
        name="type_text",
        category=ToolCategory.SCREEN,
        description="Type text at current cursor position",
        parameters={"text": "Text to type"},
        requires_permission=True,
        examples=["type_text: Hello World"],
        related_tools=["click_element", "press_key"],
    ),
    "press_key": ToolInfo(
        name="press_key",
        category=ToolCategory.SCREEN,
        description="Press a keyboard key",
        parameters={"key": "Key to press (e.g., 'enter', 'escape')"},
        requires_permission=True,
        examples=["press_key: enter", "press_key: escape"],
        related_tools=["type_text", "hotkey"],
    ),
    "hotkey": ToolInfo(
        name="hotkey",
        category=ToolCategory.SCREEN,
        description="Press a keyboard shortcut",
        parameters={"keys": "Keys to press together (e.g., 'ctrl+c')"},
        requires_permission=True,
        examples=["hotkey: ctrl+c", "hotkey: alt+tab"],
        related_tools=["press_key"],
    ),
    "get_development_context": ToolInfo(
        name="get_development_context",
        category=ToolCategory.SCREEN,
        description="Analyze screen for development context (IDE, terminal, browser)",
        parameters={},
        requires_permission=True,
        examples=["get_development_context: what development tools are visible?"],
        related_tools=["see_screen"],
    ),
    "what_am_i_doing": ToolInfo(
        name="what_am_i_doing",
        category=ToolCategory.SCREEN,
        description="Analyze current screen activity and provide summary",
        parameters={},
        requires_permission=True,
        examples=["what_am_i_doing: summarize current activity"],
        related_tools=["see_screen", "get_development_context"],
    ),

    # === BROWSER AUTOMATION TOOLS ===
    "browser_launch": ToolInfo(
        name="browser_launch",
        category=ToolCategory.BROWSER,
        description="Launch a new browser instance",
        parameters={"headless": "Run without UI (default false)"},
        requires_permission=False,
        examples=["browser_launch: start new browser"],
        related_tools=["browser_navigate", "browser_close"],
    ),
    "browser_navigate": ToolInfo(
        name="browser_navigate",
        category=ToolCategory.BROWSER,
        description="Navigate browser to a URL",
        parameters={"url": "URL to navigate to"},
        requires_permission=False,
        examples=["browser_navigate: https://github.com"],
        related_tools=["browser_launch", "browser_screenshot"],
    ),
    "browser_screenshot": ToolInfo(
        name="browser_screenshot",
        category=ToolCategory.BROWSER,
        description="Take a screenshot of the current page",
        parameters={},
        requires_permission=False,
        examples=["browser_screenshot: capture current page"],
        related_tools=["browser_navigate"],
    ),
    "browser_click": ToolInfo(
        name="browser_click",
        category=ToolCategory.BROWSER,
        description="Click an element on the page",
        parameters={"selector": "CSS selector", "text": "Text to click"},
        requires_permission=False,
        examples=["browser_click: 'button.submit'", "browser_click: text 'Login'"],
        related_tools=["browser_type", "browser_navigate"],
    ),
    "browser_type": ToolInfo(
        name="browser_type",
        category=ToolCategory.BROWSER,
        description="Type text into an input field",
        parameters={"selector": "CSS selector", "text": "Text to type"},
        requires_permission=False,
        examples=["browser_type: '#username' with 'user@example.com'"],
        related_tools=["browser_click"],
    ),
    "browser_get_content": ToolInfo(
        name="browser_get_content",
        category=ToolCategory.BROWSER,
        description="Get page content as text or HTML",
        parameters={"selector": "Optional CSS selector"},
        requires_permission=False,
        examples=["browser_get_content: get full page text"],
        related_tools=["browser_navigate"],
    ),
    "browser_close": ToolInfo(
        name="browser_close",
        category=ToolCategory.BROWSER,
        description="Close the browser instance",
        parameters={},
        requires_permission=False,
        examples=["browser_close: stop browser"],
        related_tools=["browser_launch"],
    ),

    # === TERMINAL TOOLS ===
    "terminal_execute": ToolInfo(
        name="terminal_execute",
        category=ToolCategory.TERMINAL,
        description="Execute a command in Jarvis's native terminal",
        parameters={"command": "Command to execute", "risk_check": "Assess risk first"},
        requires_permission=True,
        examples=["terminal_execute: npm test", "terminal_execute: ls -la"],
        related_tools=["terminal_cwd", "terminal_history"],
    ),
    "terminal_cwd": ToolInfo(
        name="terminal_cwd",
        category=ToolCategory.TERMINAL,
        description="Get or change current working directory",
        parameters={"path": "New directory path (optional)"},
        requires_permission=False,
        examples=["terminal_cwd: show current directory"],
        related_tools=["terminal_execute"],
    ),
    "terminal_history": ToolInfo(
        name="terminal_history",
        category=ToolCategory.TERMINAL,
        description="Get command history",
        parameters={"limit": "Number of entries"},
        requires_permission=False,
        examples=["terminal_history: get last 20 commands"],
        related_tools=["terminal_execute"],
    ),
    "computer_status": ToolInfo(
        name="computer_status",
        category=ToolCategory.TERMINAL,
        description="Get system status (CPU, memory, disk)",
        parameters={},
        requires_permission=False,
        examples=["computer_status: how's the system doing?"],
        related_tools=[],
    ),

    # === LEARNING TOOLS ===
    "erdos_math": ToolInfo(
        name="erdos_math",
        category=ToolCategory.LEARNING,
        description="Explore mathematical problems (graph theory, number theory, etc.)",
        parameters={"topic": "Math topic or problem"},
        requires_permission=False,
        examples=["erdos_math: explore Ramsey numbers"],
        related_tools=["daydream"],
    ),
    # NOTE: "daydream" is defined once, in the DAYDREAM/LEARNING TOOLS section below.

    # === EVOLUTION TOOLS ===
    "evolution": ToolInfo(
        name="evolution",
        category=ToolCategory.EVOLUTION,
        description="Self-improvement through genetic operations",
        parameters={"action": "mutate/crossover/evaluate"},
        requires_permission=True,
        examples=["evolution: suggest improvements to my code"],
        related_tools=["self_edit", "daydream"],
    ),
    "self_edit": ToolInfo(
        name="self_edit",
        category=ToolCategory.EVOLUTION,
        description="Modify Jarvis's own code with permission",
        parameters={"file": "File to edit", "changes": "Changes to make"},
        requires_permission=True,
        examples=["self_edit: optimize the erdos engine"],
        related_tools=["evolution"],
    ),

    # === MULTIMODAL TOOLS ===
    # NOTE: "vision" and "webcam" are defined once, in the MULTIMODAL section below.

    # === SECURITY TOOLS ===
    "security_scan": ToolInfo(
        name="security_scan",
        category=ToolCategory.SECURITY,
        description="Scan for security issues in code",
        parameters={"path": "Path to scan"},
        requires_permission=False,
        examples=["security_scan: check for vulnerabilities"],
        related_tools=[],
    ),

    # === DATA TOOLS ===
    "data_analyze": ToolInfo(
        name="data_analyze",
        category=ToolCategory.DATA,
        description="Analyze data files (CSV, JSON, etc.)",
        parameters={"path": "Path to data file"},
        requires_permission=False,
        examples=["data_analyze: summarize sales.csv"],
        related_tools=[],
    ),
    "data_visualize": ToolInfo(
        name="data_visualize",
        category=ToolCategory.DATA,
        description="Create visualizations from data",
        parameters={"data": "Data to visualize", "type": "Chart type"},
        requires_permission=False,
        examples=["data_visualize: bar chart from data"],
        related_tools=["data_analyze"],
    ),
    
    # === META TOOLS ===
    "list_tools": ToolInfo(
        name="list_tools",
        category=ToolCategory.META,
        description="List all available tools organized by category",
        parameters={},
        requires_permission=False,
        examples=["list_tools: what tools do I have?"],
        related_tools=["get_tool_info", "get_capabilities"],
        best_for=["discovering tools", "understanding capabilities"],
    ),
    "get_tool_info": ToolInfo(
        name="get_tool_info",
        category=ToolCategory.META,
        description="Get detailed information about a specific tool",
        parameters={"tool_name": "Name of the tool"},
        requires_permission=False,
        examples=["get_tool_info: browser_navigate"],
        related_tools=["list_tools", "find_related_tools"],
        best_for=["learning about a tool", "understanding parameters"],
    ),
    "find_related_tools": ToolInfo(
        name="find_related_tools",
        category=ToolCategory.META,
        description="Find tools related to a specific task",
        parameters={"task": "Description of the task"},
        requires_permission=False,
        examples=["find_related_tools: I want to automate a web form"],
        related_tools=["list_tools", "get_tool_info"],
    ),
    "get_capabilities": ToolInfo(
        name="get_capabilities",
        category=ToolCategory.META,
        description="Get a summary of all Jarvis capabilities",
        parameters={},
        requires_permission=False,
        examples=["get_capabilities: what can you do?"],
        related_tools=["list_tools"],
    ),
    "list_skills": ToolInfo(
        name="list_skills",
        category=ToolCategory.META,
        description="List all loaded skills",
        parameters={},
        requires_permission=False,
        examples=["list_skills: what skills are available?"],
        related_tools=["get_skill_info"],
    ),
    
    # === REFINEMENT TOOLS ===
    "refine_scan": ToolInfo(
        name="refine_scan",
        category=ToolCategory.REFINEMENT,
        description="Scan codebase for issues (bugs, security, style)",
        parameters={"path": "Directory to scan", "include_ai": "Use AI for deeper analysis"},
        requires_permission=False,
        examples=["refine_scan: check my code for issues"],
        related_tools=["refine_apply", "refine_status"],
        best_for=["code quality", "finding bugs", "security audit"],
    ),
    "refine_apply": ToolInfo(
        name="refine_apply",
        category=ToolCategory.REFINEMENT,
        description="Apply a proposed code change",
        parameters={"change_id": "ID of the change to apply"},
        requires_permission=True,
        examples=["refine_apply: fix that issue"],
        related_tools=["refine_scan", "refine_rollback"],
    ),
    "refine_status": ToolInfo(
        name="refine_status",
        category=ToolCategory.REFINEMENT,
        description="Check refinement system status",
        parameters={},
        requires_permission=False,
        examples=["refine_status: how many issues pending?"],
        related_tools=["refine_scan"],
    ),
    "refine_rollback": ToolInfo(
        name="refine_rollback",
        category=ToolCategory.REFINEMENT,
        description="Rollback an applied change",
        parameters={"change_id": "ID of the change to rollback"},
        requires_permission=True,
        examples=["refine_rollback: undo that change"],
        related_tools=["refine_apply"],
    ),
    
    # === SPATIAL TOOLS ===
    "collision_detect": ToolInfo(
        name="collision_detect",
        category=ToolCategory.SPATIAL,
        description="Detect collisions between 3D objects using GPU acceleration",
        parameters={"objects": "List of objects to check"},
        requires_permission=False,
        examples=["collision_detect: check these game objects"],
        related_tools=["particle_sim", "physics_analyze"],
        best_for=["game development", "physics simulation"],
    ),
    "particle_sim": ToolInfo(
        name="particle_sim",
        category=ToolCategory.SPATIAL,
        description="Simulate particle systems with GPU acceleration",
        parameters={"particles": "Number of particles", "duration": "Simulation time"},
        requires_permission=False,
        examples=["particle_sim: 1000 particles for 5 seconds"],
        related_tools=["collision_detect", "physics_analyze"],
    ),
    "physics_analyze": ToolInfo(
        name="physics_analyze",
        category=ToolCategory.SPATIAL,
        description="Analyze physics problems and provide GPU-optimized solutions",
        parameters={"problem": "Physics problem description"},
        requires_permission=False,
        examples=["physics_analyze: calculate trajectory"],
        related_tools=["collision_detect", "particle_sim"],
    ),
    "spatial_calc": ToolInfo(
        name="spatial_calc",
        category=ToolCategory.SPATIAL,
        description="Perform spatial and geometric calculations",
        parameters={"operation": "Type of calculation", "params": "Parameters"},
        requires_permission=False,
        examples=["spatial_calc: calculate distance between points"],
        related_tools=["physics_analyze"],
    ),
    
    # === MULTIMODAL TOOLS (extended) ===
    "transcribe": ToolInfo(
        name="transcribe",
        category=ToolCategory.MULTIMODAL,
        description="Transcribe audio to text using Whisper (NPU optimized)",
        parameters={"audio_path": "Path to audio file"},
        requires_permission=False,
        examples=["transcribe: convert this recording to text"],
        related_tools=["tokenize_voice", "process_speech"],
        best_for=["audio transcription", "speech to text"],
    ),
    "tokenize_voice": ToolInfo(
        name="tokenize_voice",
        category=ToolCategory.MULTIMODAL,
        description="Process audio into semantic and acoustic tokens",
        parameters={"audio_path": "Path to audio file"},
        requires_permission=False,
        examples=["tokenize_voice: extract speech tokens"],
        related_tools=["transcribe", "synthesize_speech"],
    ),
    "synthesize_speech": ToolInfo(
        name="synthesize_speech",
        category=ToolCategory.MULTIMODAL,
        description="Synthesize speech from text with controlled tone",
        parameters={"text": "Text to speak", "tone": "Desired tone"},
        requires_permission=False,
        examples=["synthesize_speech: say hello in a friendly tone"],
        related_tools=["tokenize_voice", "analyze_tone"],
    ),
    "analyze_tone": ToolInfo(
        name="analyze_tone",
        category=ToolCategory.MULTIMODAL,
        description="Analyze the tone and intent of text",
        parameters={"text": "Text to analyze"},
        requires_permission=False,
        examples=["analyze_tone: what's the emotion here?"],
        related_tools=["process_speech"],
    ),
    "image_tokenize": ToolInfo(
        name="image_tokenize",
        category=ToolCategory.MULTIMODAL,
        description="Convert an image into visual tokens",
        parameters={"image_path": "Path to image"},
        requires_permission=False,
        examples=["image_tokenize: convert this image"],
        related_tools=["vision"],
    ),
    
    # === DATA TOOLS (extended) ===
    "sql_query": ToolInfo(
        name="sql_query",
        category=ToolCategory.DATA,
        description="Execute SQL queries on databases (SELECT only for safety)",
        parameters={"query": "SQL query", "database": "Database connection"},
        requires_permission=False,
        examples=["sql_query: SELECT * FROM users LIMIT 10"],
        related_tools=["data_analyze"],
        best_for=["database queries", "data retrieval"],
    ),
    "statistics": ToolInfo(
        name="statistics",
        category=ToolCategory.DATA,
        description="Perform statistical analysis on data",
        parameters={"data": "Data to analyze", "operations": "Statistical operations"},
        requires_permission=False,
        examples=["statistics: get mean, median, std dev"],
        related_tools=["data_analyze"],
    ),
    "embed": ToolInfo(
        name="embed",
        category=ToolCategory.DATA,
        description="Generate text embeddings for semantic search",
        parameters={"text": "Text to embed"},
        requires_permission=False,
        examples=["embed: convert this text to vectors"],
        related_tools=["search_files", "index_files"],
    ),
    "search_files": ToolInfo(
        name="search_files",
        category=ToolCategory.DATA,
        description="Search indexed files semantically",
        parameters={"query": "Search query", "k": "Number of results"},
        requires_permission=False,
        examples=["search_files: find documents about authentication"],
        related_tools=["index_files", "get_context"],
        best_for=["semantic search", "finding relevant files"],
    ),
    "index_files": ToolInfo(
        name="index_files",
        category=ToolCategory.DATA,
        description="Index files or directories for semantic search",
        parameters={"path": "Path to index"},
        requires_permission=False,
        examples=["index_files: /home/user/documents"],
        related_tools=["search_files"],
    ),
    
    # === LEARNING TOOLS (extended) ===
    "learning": ToolInfo(
        name="learning",
        category=ToolCategory.LEARNING,
        description="Control and monitor continuous learning pipeline",
        parameters={"action": "status/start/stop"},
        requires_permission=False,
        examples=["learning: what have you learned?"],
        related_tools=["erdos_math", "daydream"],
    ),
    "sandbox_explore": ToolInfo(
        name="sandbox_explore",
        category=ToolCategory.LEARNING,
        description="Trigger exploration in the sandbox",
        parameters={"topic": "Optional topic to explore"},
        requires_permission=False,
        examples=["sandbox_explore: discover something new"],
        related_tools=["free_time_stats"],
    ),
    "sketch": ToolInfo(
        name="sketch",
        category=ToolCategory.LEARNING,
        description="Capture quick insights and test concepts",
        parameters={"idea": "The idea or concept"},
        requires_permission=False,
        examples=["sketch: test this algorithm idea"],
        related_tools=["verified_solutions"],
    ),
    
    # === AUTOMATION TOOLS ===
    "docker": ToolInfo(
        name="docker",
        category=ToolCategory.AUTOMATION,
        description="Manage Docker containers, images, and compose stacks",
        parameters={"action": "list/run/stop/build", "container": "Container name"},
        requires_permission=True,
        examples=["docker: list running containers"],
        related_tools=["n8n", "browser_launch"],
        best_for=["container management", "deployment"],
    ),
    "n8n": ToolInfo(
        name="n8n",
        category=ToolCategory.AUTOMATION,
        description="Manage n8n workflows - create, execute, monitor",
        parameters={"action": "list/run/create", "workflow": "Workflow ID"},
        requires_permission=False,
        examples=["n8n: list my workflows"],
        related_tools=["docker"],
    ),
    
    # === SECURITY TOOLS (extended) ===
    "security_audit": ToolInfo(
        name="security_audit",
        category=ToolCategory.SECURITY,
        description="Query security audit logs",
        parameters={"query": "Filter query"},
        requires_permission=False,
        examples=["security_audit: show recent threats"],
        related_tools=["security_scan"],
    ),
    "defense": ToolInfo(
        name="defense",
        category=ToolCategory.SECURITY,
        description="Monitor and manage data defense operations",
        parameters={"action": "status/enable/disable"},
        requires_permission=False,
        examples=["defense: what threats have been blocked?"],
        related_tools=["security_scan"],
    ),
    "dlp_whitelist": ToolInfo(
        name="dlp_whitelist",
        category=ToolCategory.SECURITY,
        description="Manage data leak protection whitelist",
        parameters={"action": "add/remove/list", "pattern": "Pattern to whitelist"},
        requires_permission=True,
        examples=["dlp_whitelist: add this API endpoint"],
        related_tools=["defense"],
    ),
    
    # === META TOOLS (extended) ===
    "self_reflect": ToolInfo(
        name="self_reflect",
        category=ToolCategory.META,
        description="Reflect on recent interactions and analyze behavior patterns",
        parameters={"focus": "recent/patterns/ethical/improvements/processes"},
        requires_permission=False,
        examples=["self_reflect: what patterns have you noticed?"],
        related_tools=["design_process", "review_process"],
    ),
    "design_process": ToolInfo(
        name="design_process",
        category=ToolCategory.META,
        description="Design a new agentic process for handling specific tasks",
        parameters={"request": "What the process should accomplish"},
        requires_permission=False,
        examples=["design_process: create a process for daily backups"],
        related_tools=["self_reflect", "review_process"],
    ),
    "review_process": ToolInfo(
        name="review_process",
        category=ToolCategory.META,
        description="Review an existing agentic process for improvements",
        parameters={"process_name": "Name of the process to review"},
        requires_permission=False,
        examples=["review_process: backup_process"],
        related_tools=["self_reflect", "design_process"],
    ),
    "get_skill_info": ToolInfo(
        name="get_skill_info",
        category=ToolCategory.META,
        description="Get detailed information about a loaded skill",
        parameters={"skill_name": "Name of the skill"},
        requires_permission=False,
        examples=["get_skill_info: web_research"],
        related_tools=["list_tools", "get_tool_info"],
    ),
    
    # === UI TOOLS ===
    "ui_sense": ToolInfo(
        name="ui_sense",
        category=ToolCategory.SCREEN,
        description="Sense UI state and get context about visible elements",
        parameters={},
        requires_permission=False,
        examples=["ui_sense: what's on screen?"],
        related_tools=["see_screen", "ui_watch"],
    ),
    "ui_watch": ToolInfo(
        name="ui_watch",
        category=ToolCategory.SCREEN,
        description="Watch for UI changes over time",
        parameters={"duration": "Duration to watch in seconds"},
        requires_permission=False,
        examples=["ui_watch: 5 seconds"],
        related_tools=["ui_sense", "ui_visualize"],
    ),
    "ui_visualize": ToolInfo(
        name="ui_visualize",
        category=ToolCategory.SCREEN,
        description="Visualize UI structure and relationships",
        parameters={},
        requires_permission=False,
        examples=["ui_visualize: show UI tree"],
        related_tools=["ui_sense"],
    ),
    
    # === SYSTEM SENSE TOOLS ===
    "system_sense": ToolInfo(
        name="system_sense",
        category=ToolCategory.SYSTEM,
        description="Sense the entire Jarvis system - resources, processes, subsystems, models",
        parameters={
            "focus": "Area to focus on (all, resources, subsystems, processes, network, models)",
            "detailed": "Include detailed metrics (boolean)",
        },
        requires_permission=False,
        examples=["system_sense: everything about the system", "system_sense with focus: resources"],
        related_tools=["system_watch", "ui_sense", "computer_status"],
    ),
    "system_watch": ToolInfo(
        name="system_watch",
        category=ToolCategory.SYSTEM,
        description="Watch for system resource spikes and health changes",
        parameters={
            "thresholds": "Check resource thresholds (boolean)",
            "recent_changes": "Check for recent process changes (boolean)",
        },
        requires_permission=False,
        examples=["system_watch: any resource alerts?", "system_watch with thresholds: true"],
        related_tools=["system_sense", "health_check"],
    ),
    
    # === INSIGHT SEARCH ===
    "insight_search": ToolInfo(
        name="insight_search",
        category=ToolCategory.MEMORY,
        description="Search through Jarvis's accumulated insights and knowledge",
        parameters={
            "query": "Search query to find relevant insights",
            "source": "Filter by source (insight_bridge, erdos, daydream, all)",
            "type": "Filter by type (pattern_discovery, strategy_success, cross_domain)",
            "limit": "Maximum results (default: 10)",
        },
        requires_permission=False,
        examples=["insight_search: pattern discovery", "insight_search: math graph theory", "insight_search: learning optimization"],
        related_tools=["insight_heatmap", "insight_stats", "memory_search"],
    ),
    "insight_heatmap": ToolInfo(
        name="insight_heatmap",
        category=ToolCategory.MEMORY,
        description="Get word frequency heatmap from accumulated insights",
        parameters={
            "limit": "Number of top words to return (default: 50)",
        },
        requires_permission=False,
        examples=["insight_heatmap", "insight_heatmap with limit: 100"],
        related_tools=["insight_search", "insight_stats"],
    ),
    "insight_stats": ToolInfo(
        name="insight_stats",
        category=ToolCategory.MEMORY,
        description="Get statistics about Jarvis's accumulated insights",
        parameters={},
        requires_permission=False,
        examples=["insight_stats"],
        related_tools=["insight_search", "insight_heatmap"],
    ),
    
    # === TOOL CREATION ===
    "tool_create": ToolInfo(
        name="tool_create",
        category=ToolCategory.META,
        description="Create a new tool from specification",
        parameters={
            "name": "Tool name in snake_case",
            "description": "What the tool does",
            "parameters": "JSON Schema of parameters",
            "category": "Tool category",
            "requires_permission": "Whether tool needs permission",
        },
        requires_permission=True,
        examples=["tool_create: create a data formatter tool"],
        related_tools=["tool_test", "tool_validate"],
    ),
    "tool_test": ToolInfo(
        name="tool_test",
        category=ToolCategory.META,
        description="Test a sandbox tool before deployment",
        parameters={"name": "Name of tool to test"},
        requires_permission=False,
        examples=["tool_test hello_world"],
        related_tools=["tool_create", "tool_validate"],
    ),
    "tool_validate": ToolInfo(
        name="tool_validate",
        category=ToolCategory.META,
        description="Validate tool code for best practices",
        parameters={"name": "Name of tool to validate"},
        requires_permission=False,
        examples=["tool_validate hello_world"],
        related_tools=["tool_test", "tool_create"],
    ),
    "tool_request_deploy": ToolInfo(
        name="tool_request_deploy",
        category=ToolCategory.META,
        description="Request user approval to deploy a tested tool",
        parameters={
            "name": "Name of tool to deploy",
            "summary": "Brief description",
            "capabilities": "List of capabilities",
        },
        requires_permission=True,
        examples=["tool_request_deploy hello_world"],
        related_tools=["tool_approve", "tool_reject"],
    ),
    "tool_list_sandbox": ToolInfo(
        name="tool_list_sandbox",
        category=ToolCategory.META,
        description="List all tools in sandbox (not yet deployed)",
        parameters={},
        requires_permission=False,
        examples=["tool_list_sandbox"],
        related_tools=["tool_create"],
    ),
    
    # === HOME AUTOMATION (extended) ===
    "home_status": ToolInfo(
        name="home_status",
        category=ToolCategory.HOME,
        description="Get status of all home devices",
        parameters={},
        requires_permission=False,
        examples=["home_status: what devices are available?"],
        related_tools=["home_control"],
    ),
    
    # === IMAGE/VISION TOOLS (extended) ===
    "image_analyze": ToolInfo(
        name="image_analyze",
        category=ToolCategory.MULTIMODAL,
        description="Analyze image content and structure",
        parameters={"image": "Image path or base64"},
        requires_permission=False,
        examples=["image_analyze: what's in this screenshot?"],
        related_tools=["vision", "webcam"],
    ),
    "image_read": ToolInfo(
        name="image_read",
        category=ToolCategory.MULTIMODAL,
        description="Read text from image (OCR)",
        parameters={"image": "Image path"},
        requires_permission=False,
        examples=["image_read: extract text from screenshot"],
        related_tools=["image_analyze"],
    ),
    "vision": ToolInfo(
        name="vision",
        category=ToolCategory.MULTIMODAL,
        description="General vision model for image understanding",
        parameters={"image": "Image path or base64", "prompt": "What to analyze"},
        requires_permission=False,
        examples=["vision: describe the scene"],
        related_tools=["webcam", "image_analyze"],
    ),
    "webcam": ToolInfo(
        name="webcam",
        category=ToolCategory.MULTIMODAL,
        description="Capture image from webcam",
        parameters={},
        requires_permission=True,
        examples=["webcam: take a photo"],
        related_tools=["vision"],
    ),
    
    # === DAYDREAM/LEARNING TOOLS (extended) ===
    "daydream": ToolInfo(
        name="daydream",
        category=ToolCategory.LEARNING,
        description="Enter autonomous learning/exploration mode",
        parameters={"duration": "Optional duration in seconds", "topic": "Optional topic"},
        requires_permission=False,
        examples=["daydream: explore something new"],
        related_tools=["erdos_math", "sandbox_explore", "evolution"],
    ),
    "verified_solutions": ToolInfo(
        name="verified_solutions",
        category=ToolCategory.LEARNING,
        description="Get verified solutions from daydream exploration",
        parameters={"limit": "Maximum solutions to return"},
        requires_permission=False,
        examples=["verified_solutions: show recent discoveries"],
        related_tools=["daydream"],
    ),
    
    # === EVOLUTION TOOLS (extended) ===
    "evolution_status": ToolInfo(
        name="evolution_status",
        category=ToolCategory.EVOLUTION,
        description="Get current evolution subsystem status",
        parameters={},
        requires_permission=False,
        examples=["evolution_status: what's the current state?"],
        related_tools=["evolution"],
    ),
    "evolution_pathway": ToolInfo(
        name="evolution_pathway",
        category=ToolCategory.EVOLUTION,
        description="Get evolution pathway for genes",
        parameters={"gene": "Optional gene name"},
        requires_permission=False,
        examples=["evolution_pathway: show pathways"],
        related_tools=["evolution"],
    ),
    "gene_list": ToolInfo(
        name="gene_list",
        category=ToolCategory.EVOLUTION,
        description="List all genes in evolution pool",
        parameters={},
        requires_permission=False,
        examples=["gene_list: what genes exist?"],
        related_tools=["propose_gene"],
    ),
    "gene_approval": ToolInfo(
        name="gene_approval",
        category=ToolCategory.EVOLUTION,
        description="Approve or reject a proposed gene",
        parameters={"gene_id": "Gene ID", "approved": "True/False"},
        requires_permission=True,
        examples=["gene_approval: approve gene_123"],
        related_tools=["propose_gene"],
    ),
    
    # === PERMISSION TOOLS ===
    "approve_permission": ToolInfo(
        name="approve_permission",
        category=ToolCategory.META,
        description="Approve a pending permission request",
        parameters={"request_id": "ID of the request"},
        requires_permission=True,
        examples=["approve_permission: allow terminal_execute"],
        related_tools=["deny_permission", "view_permission_requests"],
    ),
    "deny_permission": ToolInfo(
        name="deny_permission",
        category=ToolCategory.META,
        description="Deny a pending permission request",
        parameters={"request_id": "ID of the request"},
        requires_permission=True,
        examples=["deny_permission: reject dangerous_command"],
        related_tools=["approve_permission"],
    ),
    "view_permission_requests": ToolInfo(
        name="view_permission_requests",
        category=ToolCategory.META,
        description="View all pending permission requests",
        parameters={},
        requires_permission=False,
        examples=["view_permission_requests: what's pending?"],
        related_tools=["approve_permission", "deny_permission"],
    ),
}


def get_tool_info(tool_name: str) -> ToolInfo | None:
    """Get information about a specific tool."""
    return TOOL_REGISTRY.get(tool_name)


def get_tools_by_category(category: ToolCategory) -> list[ToolInfo]:
    """Get all tools in a category."""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


def get_all_tool_names() -> list[str]:
    """Get all registered tool names."""
    return list(TOOL_REGISTRY.keys())


def get_tool_summary() -> str:
    """Get a summary of all available tools for the agent."""
    lines = ["# Available Tools Overview\n"]

    by_category: dict[ToolCategory, list[ToolInfo]] = {}
    for tool in TOOL_REGISTRY.values():
        if tool.category not in by_category:
            by_category[tool.category] = []
        by_category[tool.category].append(tool)

    for category in ToolCategory:
        tools = by_category.get(category, [])
        if not tools:
            continue

        lines.append(f"\n## {category.value.upper()} ({len(tools)} tools)\n")
        for tool in sorted(tools, key=lambda t: t.name):
            perm = "⚠️ " if tool.requires_permission else ""
            lines.append(f"- **{tool.name}**: {perm}{tool.description}")

    return "\n".join(lines)


def get_tool_capabilities_prompt() -> str:
    """Generate a prompt explaining tool capabilities to the LLM."""
    return """
## Your Tool Capabilities

You have access to tools organized into functional trees. Choose the BEST tool for each task.

### Tool Trees (Related Tools for Common Tasks)

**File Operations**: `read_file`, `write_file`, `list_files`, `delete_file`, `shell`, `terminal_execute`
**Web Research**: `web_search`, `web_fetch`, `image_search`, `image_download`
**Screen Automation**: `see_screen`, `find_element`, `click_element`, `type_text`, `press_key`, `hotkey`
**Browser Tasks**: `browser_launch`, `browser_navigate`, `browser_click`, `browser_type`, `browser_screenshot`
**Terminal Access**: `terminal_execute`, `terminal_cwd`, `terminal_history`, `computer_status`
**Data Analysis**: `data_analyze`, `sql_query`, `statistics`, `embed`, `search_files`
**Voice/Audio**: `transcribe`, `tokenize_voice`, `process_speech`, `synthesize_speech`, `analyze_tone`
**Learning**: `erdos_math`, `daydream`, `learning`, `sandbox_explore`, `sketch`
**Self-Improvement**: `evolution`, `self_edit`, `refine_scan`, `refine_apply`, `propose_gene`
**Meta/Discovery**: `list_tools`, `get_tool_info`, `get_capabilities`, `find_related_tools`

### Best Tool Selection Guide

When asked to:
- **Search the web** → Use `web_search` (DuckDuckGo with summaries)
- **Fetch a webpage** → Use `web_fetch` (clean text extraction)
- **Read/write files** → Use `read_file` / `write_file` (safe, validated paths)
- **Execute commands** → Use `terminal_execute` (full features) or `shell` (quick commands)
- **See the screen** → Use `see_screen` (OCR + UI detection + context)
- **Click something** → Use `click_element` (by text or coordinates)
- **Automate browser** → Use `browser_launch`, `browser_navigate`, etc.
- **Analyze data** → Use `data_analyze` (statistics + visualizations)
- **Search files** → Use `search_files` (semantic vector search)
- **Transcribe audio** → Use `transcribe` (Whisper, NPU-optimized)
- **Explore math** → Use `erdos_math` (graph theory, number theory)
- **Improve code** → Use `refine_scan` to find issues, `refine_apply` to fix
- **Check system** → Use `computer_status` for CPU/memory/disk

### Category Overview

| Category | Tools | Purpose |
|----------|-------|---------|
| **SYSTEM** | shell, read/write_file, list_files | Core file and command operations |
| **WEB** | web_search, web_fetch, image_* | Internet research and downloads |
| **SCREEN** | see_screen, find/click_element, type_text | Screen vision and UI automation |
| **BROWSER** | browser_launch/navigate/click/type | Headless browser automation |
| **TERMINAL** | terminal_execute, cwd, history | Native shell access |
| **DATA** | data_analyze, sql_query, statistics | Data analysis and search |
| **MULTIMODAL** | transcribe, synthesize_speech, vision | Audio and image processing |
| **LEARNING** | erdos_math, daydream, learning | Autonomous exploration |
| **EVOLUTION** | evolution, self_edit | Self-improvement |
| **REFINEMENT** | refine_scan, apply, rollback | Code quality |
| **SECURITY** | security_scan, defense, audit | Protection |
| **META** | list_tools, get_tool_info, get_capabilities | Self-awareness |

### Important Notes:
1. ⚠️ = Requires permission before execution
2. Use `get_tool_info(name)` for detailed parameter info
3. Use `find_related_tools(task)` to discover relevant tools
4. Tools in the same tree often work together (chain them)
5. Always prefer the MOST SPECIFIC tool for the task
""".strip()


def find_best_tool_for_task(task_description: str) -> list[tuple[str, str, str]]:
    """Find the best tool(s) for a given task.
    
    Returns list of (tool_name, description, reason) tuples.
    """
    task_lower = task_description.lower()
    results = []
    
    for keyword, (tool_name, reason) in BEST_TOOL_FOR_TASK.items():
        if keyword in task_lower:
            tool_info = TOOL_REGISTRY.get(tool_name)
            if tool_info:
                results.append((tool_name, tool_info.description, reason))
    
    if not results:
        for tree_name, tools in TOOL_TREES.items():
            if tree_name.replace("_", " ") in task_lower:
                for tool_name in tools[:3]:
                    tool_info = TOOL_REGISTRY.get(tool_name)
                    if tool_info:
                        results.append((tool_name, tool_info.description, f"Part of {tree_name} tree"))
    
    return results[:5]


def get_tools_for_category_tree(category: str) -> dict[str, list[str]]:
    """Get all tools organized by category tree."""
    result = {}
    for tree_name, tools in TOOL_TREES.items():
        if category.lower() in tree_name.lower() or tree_name.lower() in category.lower():
            result[tree_name] = tools
    return result


class ToolRegistryValidator:
    """Validates and manages tool registrations with health checks,
    conflict detection, and availability tracking."""

    def __init__(self):
        self._dynamic_tools: dict[str, ToolInfo] = {}
        self._tool_health: dict[str, dict] = {}
        self._conflicts: list[dict] = []
        self._lock = threading.RLock()

    def register_dynamic_tool(self, tool_info: ToolInfo) -> list[str]:
        """Register a new dynamic tool with validation.
        
        Returns list of warnings (empty if all ok).
        Raises ValueError for critical validation failures.
        """
        warnings = []
        name = tool_info.name

        if not name or not re.match(r'^[a-z][a-z0-9_]*$', name):
            raise ValueError(f"Invalid tool name '{name}': must be lowercase snake_case, start with letter")

        if len(name) > 50:
            raise ValueError(f"Tool name too long: '{name}' (max 50 chars)")

        if len(name) < 2:
            raise ValueError(f"Tool name too short: '{name}' (min 2 chars)")

        existing = TOOL_REGISTRY.get(name) or self._dynamic_tools.get(name)
        if existing:
            if existing.category == tool_info.category and existing.description == tool_info.description:
                warnings.append(f"Tool '{name}' already registered with identical definition - skipped")
                return warnings
            raise ValueError(f"Tool '{name}' already registered (category={existing.category.value})")

        if not tool_info.description or len(tool_info.description) < 10:
            raise ValueError(f"Tool '{name}' description too short (min 10 chars)")

        for param_name, param_desc in tool_info.parameters.items():
            if not param_name or not re.match(r'^[a-z][a-z0-9_]*$', param_name):
                raise ValueError(f"Tool '{name}' has invalid parameter name '{param_name}'")
            if not param_desc or len(param_desc) < 3:
                warnings.append(f"Tool '{name}' parameter '{param_name}' has short description")

        category_tools = self._get_category_tools(tool_info.category)
        if len(category_tools) >= 30:
            warnings.append(f"Category '{tool_info.category.value}' has {len(category_tools)} tools - consider splitting")

        name_lower = name.lower()
        for existing_name, existing_info in {**TOOL_REGISTRY, **self._dynamic_tools}.items():
            existing_lower = existing_name.lower()
            if existing_lower == name_lower and existing_name != name:
                warnings.append(f"Tool '{name}' differs only by case from existing '{existing_name}'")

            name_words = set(name_lower.replace('_', ' ').split())
            existing_words = set(existing_lower.replace('_', ' ').split())
            overlap = name_words & existing_words
            if len(overlap) >= 2 and existing_name != name:
                common_cats = {ToolCategory.SYSTEM, ToolCategory.TERMINAL, ToolCategory.WEB}
                if tool_info.category in common_cats and existing_info.category in common_cats:
                    if tool_info.category == existing_info.category:
                        self._conflicts.append({
                            "tool1": existing_name,
                            "tool2": name,
                            "reason": f"Same category '{tool_info.category.value}', 2+ word overlap: {overlap}",
                        })

        for related in tool_info.related_tools:
            if related not in TOOL_REGISTRY and related not in self._dynamic_tools:
                warnings.append(f"Tool '{name}' references unknown related tool '{related}'")

        with self._lock:
            self._dynamic_tools[name] = tool_info
            self._tool_health[name] = {
                "registered_at": time.time(),
                "call_count": 0,
                "error_count": 0,
                "last_called": None,
                "last_error": None,
                "healthy": True,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0,
            }

        return warnings

    def unregister_dynamic_tool(self, name: str) -> bool:
        """Unregister a dynamic tool. Cannot unregister built-in tools."""
        if name in TOOL_REGISTRY:
            return False
        with self._lock:
            self._dynamic_tools.pop(name, None)
            self._tool_health.pop(name, None)
        return True

    def record_tool_call(self, name: str, success: bool, latency_ms: float, error: str | None = None) -> None:
        """Record a tool call for health tracking."""
        with self._lock:
            health = self._tool_health.get(name)
            if not health:
                health = {
                    "registered_at": time.time(),
                    "call_count": 0,
                    "error_count": 0,
                    "last_called": None,
                    "last_error": None,
                    "healthy": True,
                    "avg_latency_ms": 0.0,
                    "total_latency_ms": 0.0,
                }
                self._tool_health[name] = health

            health["call_count"] += 1
            health["last_called"] = time.time()
            health["total_latency_ms"] += latency_ms
            health["avg_latency_ms"] = health["total_latency_ms"] / health["call_count"]

            if not success:
                health["error_count"] += 1
                health["last_error"] = error
                recent_error_rate = health["error_count"] / health["call_count"]
                if health["call_count"] >= 5 and recent_error_rate > 0.5:
                    health["healthy"] = False
            else:
                if health["call_count"] >= 5:
                    recent_error_rate = health["error_count"] / health["call_count"]
                    health["healthy"] = recent_error_rate <= 0.5

    def get_tool_health(self, name: str) -> dict | None:
        """Get health metrics for a tool."""
        with self._lock:
            return self._tool_health.get(name)

    def get_all_health(self) -> dict[str, dict]:
        """Get health metrics for all tools."""
        with self._lock:
            return dict(self._tool_health)

    def get_unhealthy_tools(self) -> list[str]:
        """Get list of tools with poor health."""
        with self._lock:
            return [name for name, health in self._tool_health.items() if not health.get("healthy", True)]

    def get_conflicts(self) -> list[dict]:
        """Get detected tool conflicts."""
        return list(self._conflicts)

    def validate_tool_input(self, tool_name: str, params: dict) -> tuple[bool, list[str]]:
        """Validate tool input parameters against registry definition.
        
        Returns (is_valid, list_of_errors).
        """
        tool_info = TOOL_REGISTRY.get(tool_name) or self._dynamic_tools.get(tool_name)
        if not tool_info:
            return False, [f"Unknown tool: {tool_name}"]

        errors = []
        expected_params = set(tool_info.parameters.keys())
        provided_params = set(params.keys())

        unknown = provided_params - expected_params
        if unknown:
            errors.append(f"Unknown parameters: {', '.join(sorted(unknown))}")

        missing = expected_params - provided_params
        for param in missing:
            errors.append(f"Missing required parameter: {param}")

        max_input_length = 10000
        for param_name, value in params.items():
            if isinstance(value, str) and len(value) > max_input_length:
                errors.append(f"Parameter '{param_name}' exceeds max length ({max_input_length})")
            elif isinstance(value, (dict, list)):
                serialized = json.dumps(value)
                if len(serialized) > max_input_length:
                    errors.append(f"Parameter '{param_name}' serialized exceeds max length ({max_input_length})")

        return len(errors) == 0, errors

    def get_dynamic_tools(self) -> dict[str, ToolInfo]:
        """Get all dynamically registered tools."""
        with self._lock:
            return dict(self._dynamic_tools)

    def get_all_tools(self) -> dict[str, ToolInfo]:
        """Get all tools (built-in + dynamic)."""
        all_tools = dict(TOOL_REGISTRY)
        with self._lock:
            all_tools.update(self._dynamic_tools)
        return all_tools

    def _get_category_tools(self, category: ToolCategory) -> list[ToolInfo]:
        builtin = [t for t in TOOL_REGISTRY.values() if t.category == category]
        dynamic = [t for t in self._dynamic_tools.values() if t.category == category]
        return builtin + dynamic

    def get_registry_stats(self) -> dict:
        """Get comprehensive registry statistics."""
        with self._lock:
            all_tools = {**TOOL_REGISTRY, **self._dynamic_tools}
            category_counts: dict[str, int] = {}
            for tool in all_tools.values():
                cat = tool.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

            perm_count = sum(1 for t in all_tools.values() if t.requires_permission)
            total_calls = sum(h.get("call_count", 0) for h in self._tool_health.values())
            total_errors = sum(h.get("error_count", 0) for h in self._tool_health.values())

            return {
                "total_tools": len(all_tools),
                "builtin_tools": len(TOOL_REGISTRY),
                "dynamic_tools": len(self._dynamic_tools),
                "permission_required": perm_count,
                "categories": category_counts,
                "conflicts_detected": len(self._conflicts),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "unhealthy_tools": len(self.get_unhealthy_tools()),
            }


_validator: ToolRegistryValidator | None = None
_validator_lock = threading.Lock()


def get_tool_validator() -> ToolRegistryValidator:
    """Thread-safe singleton for ToolRegistryValidator."""
    global _validator
    if _validator is None:
        with _validator_lock:
            if _validator is None:
                _validator = ToolRegistryValidator()
    return _validator
