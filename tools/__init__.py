"""Jarvis Tools Module"""
from .base import Tool, ToolRegistry

try:
    from .system import SystemTools
except ImportError:
    SystemTools = None

try:
    from .web import WebTools
except ImportError:
    WebTools = None

try:
    from .productivity import ProductivityTools
except ImportError:
    ProductivityTools = None

try:
    from .home import HomeTools
except ImportError:
    HomeTools = None

try:
    from .spatial import SpatialTools
except ImportError:
    SpatialTools = None

try:
    from .art_philosophy import ArtPhilosophyTools
except ImportError:
    ArtPhilosophyTools = None

try:
    from .physics_sim import PhysicsSimTools
except ImportError:
    PhysicsSimTools = None

try:
    from .npu_tools import RetrievalTools
except ImportError:
    RetrievalTools = None

try:
    from .docker import DockerTool, DockerComposeTool
except ImportError:
    DockerTool = None
    DockerComposeTool = None

try:
    from .n8n import N8NTool
except ImportError:
    N8NTool = None

try:
    from .transcribe import TranscribeTool
except ImportError:
    TranscribeTool = None

try:
    from .token_tools import SpeechTokenizeTool, ImageTokenizeTool
except ImportError:
    SpeechTokenizeTool = None
    ImageTokenizeTool = None

try:
    from .screen_tools import SCREEN_TOOLS, register_screen_tools
except ImportError:
    SCREEN_TOOLS = None
    register_screen_tools = None

try:
    from .browser_tools import BROWSER_TOOLS, register_browser_tools
except ImportError:
    BROWSER_TOOLS = None
    register_browser_tools = None

try:
    from .tool_discovery import TOOL_DISCOVERY_TOOLS, register_tool_discovery_tools
except ImportError:
    TOOL_DISCOVERY_TOOLS = None
    register_tool_discovery_tools = None

try:
    from .web_ui_refine import TOOLS as WEB_UI_REFINE_TOOLS
    from .web_ui_refine import WebUIViewTool, WebUIAnalyzeTool, WebUIProposeTool
    from .web_ui_refine import WebUIApproveTool, WebUIRejectTool, WebUIHistoryTool
    from .web_ui_refine import MobileUIViewTool
except ImportError:
    WEB_UI_REFINE_TOOLS = None
    WebUIViewTool = None
    WebUIAnalyzeTool = None
    WebUIProposeTool = None
    WebUIApproveTool = None
    WebUIRejectTool = None
    WebUIHistoryTool = None
    MobileUIViewTool = None

try:
    from .ui_sense import TOOLS as UI_SENSE_TOOLS
    from .ui_sense import UISenseTool, UIWatchTool, UIVisualizeTool
    from .ui_sense import set_web_interface, log_ui_event
except ImportError:
    UI_SENSE_TOOLS = None
    UISenseTool = None
    UIWatchTool = None
    UIVisualizeTool = None
    set_web_interface = None
    log_ui_event = None

try:
    from .system_sense import TOOLS as SYSTEM_SENSE_TOOLS
    from .system_sense import SystemSenseTool, SystemWatchTool
    from .system_sense import set_agent_for_system_sense, set_web_interface_for_system_sense
except ImportError:
    SYSTEM_SENSE_TOOLS = None
    SystemSenseTool = None
    SystemWatchTool = None
    set_agent_for_system_sense = None
    set_web_interface_for_system_sense = None

try:
    from .tool_creation import TOOLS as TOOL_CREATION_TOOLS
    from .tool_creation import ToolCreatorTool, ToolTestTool, ToolValidateTool
    from .tool_creation import ToolRequestDeployTool, ToolListSandboxTool
except ImportError:
    TOOL_CREATION_TOOLS = None
    ToolCreatorTool = None
    ToolTestTool = None
    ToolValidateTool = None
    ToolRequestDeployTool = None
    ToolListSandboxTool = None

try:
    from .insight_search import TOOLS as INSIGHT_SEARCH_TOOLS
    from .insight_search import InsightSearchTool, InsightHeatmapTool, InsightStatsTool
except ImportError:
    INSIGHT_SEARCH_TOOLS = None
    InsightSearchTool = None
    InsightHeatmapTool = None
    InsightStatsTool = None

try:
    from .native_files import (
        TOOLS as NATIVE_FILE_TOOLS,
        NativeFileManager,
        NativeReadTool,
        NativeWriteTool,
        NativeListTool,
        NativeDeleteTool,
        NativeMoveTool,
        NativeCopyTool,
        WorkspaceInfoTool,
        get_file_manager,
        initialize_native_files,
        register_native_tools,
        AccessMode,
    )
except ImportError:
    NATIVE_FILE_TOOLS = None
    NativeFileManager = None
    NativeReadTool = None
    NativeWriteTool = None
    NativeListTool = None
    NativeDeleteTool = None
    NativeMoveTool = None
    NativeCopyTool = None
    WorkspaceInfoTool = None
    get_file_manager = None
    initialize_native_files = None
    register_native_tools = None
    AccessMode = None

try:
    from .project import (
        ProjectInitTool,
        ProjectStatusTool,
        ProjectDecomposeTool,
        ProjectExecuteTool,
        ProjectFileReadTool,
        ProjectFileWriteTool,
        ProjectTestTool,
        ProjectCommitTool,
        register_project_tools,
    )
    PROJECT_TOOLS = [
        ProjectInitTool,
        ProjectStatusTool,
        ProjectDecomposeTool,
        ProjectExecuteTool,
        ProjectFileReadTool,
        ProjectFileWriteTool,
        ProjectTestTool,
        ProjectCommitTool,
    ]
except ImportError:
    ProjectInitTool = None
    ProjectStatusTool = None
    ProjectDecomposeTool = None
    ProjectExecuteTool = None
    ProjectFileReadTool = None
    ProjectFileWriteTool = None
    ProjectTestTool = None
    ProjectCommitTool = None
    register_project_tools = None
    PROJECT_TOOLS = None


__all__ = [
    "Tool", "ToolRegistry",
    "SystemTools", "WebTools", "ProductivityTools",
    "HomeTools", "SpatialTools", "ArtPhilosophyTools", "PhysicsSimTools",
    "RetrievalTools", "DockerTool", "DockerComposeTool", "N8NTool",
    "TranscribeTool", "SpeechTokenizeTool", "ImageTokenizeTool",
    "SCREEN_TOOLS", "register_screen_tools",
    "BROWSER_TOOLS", "register_browser_tools",
    "TOOL_DISCOVERY_TOOLS", "register_tool_discovery_tools",
    "WEB_UI_REFINE_TOOLS", "WebUIViewTool", "WebUIAnalyzeTool",
    "WebUIProposeTool", "WebUIApproveTool", "WebUIRejectTool", "WebUIHistoryTool",
    "MobileUIViewTool",
    "UI_SENSE_TOOLS", "UISenseTool", "UIWatchTool", "UIVisualizeTool",
    "set_web_interface", "log_ui_event",
    "SYSTEM_SENSE_TOOLS", "SystemSenseTool", "SystemWatchTool",
    "set_agent_for_system_sense", "set_web_interface_for_system_sense",
    "TOOL_CREATION_TOOLS", "ToolCreatorTool", "ToolTestTool", "ToolValidateTool",
    "ToolRequestDeployTool", "ToolListSandboxTool",
    "INSIGHT_SEARCH_TOOLS", "InsightSearchTool", "InsightHeatmapTool", "InsightStatsTool",
    "NATIVE_FILE_TOOLS", "NativeFileManager", "NativeReadTool", "NativeWriteTool",
    "NativeListTool", "NativeDeleteTool", "NativeMoveTool", "NativeCopyTool",
    "WorkspaceInfoTool", "get_file_manager", "initialize_native_files",
    "register_native_tools", "AccessMode",
    "PROJECT_TOOLS", "ProjectInitTool", "ProjectStatusTool",
    "ProjectDecomposeTool", "ProjectExecuteTool", "ProjectFileReadTool",
    "ProjectFileWriteTool", "ProjectTestTool", "ProjectCommitTool",
    "register_project_tools",
]

try:
    from .oracle import OraclePredictTool, register_oracle_tools
    ORACLE_TOOLS = [OraclePredictTool]
except ImportError:
    OraclePredictTool = None
    register_oracle_tools = None
    ORACLE_TOOLS = None

try:
    from .cognitive_ai import (
        CognitivePsychologyTool, MachineAlignmentTool, NeuroscienceTool,
        register_cognitive_tools,
    )
    COGNITIVE_AI_TOOLS = [CognitivePsychologyTool, MachineAlignmentTool, NeuroscienceTool]
except ImportError:
    CognitivePsychologyTool = None
    MachineAlignmentTool = None
    NeuroscienceTool = None
    register_cognitive_tools = None
    COGNITIVE_AI_TOOLS = None

__all__.extend([
    "OraclePredictTool", "register_oracle_tools", "ORACLE_TOOLS",
    "CognitivePsychologyTool", "MachineAlignmentTool", "NeuroscienceTool",
    "register_cognitive_tools", "COGNITIVE_AI_TOOLS",
])
