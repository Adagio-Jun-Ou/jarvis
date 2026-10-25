"""UI Sensing Tools - Allow Jarvis to see and know its own UIs in real-time."""

import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from .base import Tool

@dataclass
class UIState:
    """Current state of a UI instance."""
    interface_type: str  # 'web' or 'mobile'
    connected: bool
    active_tab: str = 'chat'
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    thinking: bool = False
    last_user_input: Optional[str] = None
    last_response: Optional[str] = None
    last_activity: Optional[float] = None
    client_count: int = 0


class UISenseTool(Tool):
    """Sense and understand Jarvis's own UIs in real-time."""
    
    name = "ui_sense"
    description = """See and understand Jarvis's own web and mobile UIs in real-time.

This gives Jarvis self-awareness of what users are seeing and doing on its interfaces.

Returns:
- Current state of both web and mobile UIs
- Active conversations and messages
- Tool calls in progress
- Thinking/response states
- Connected clients count

Use this to:
- Understand what the user is currently viewing
- See conversation history on both interfaces
- Know when users are waiting for responses
- Track active tool executions across UIs
"""
    parameters = {
        "type": "object",
        "properties": {
            "interface": {
                "type": "string",
                "enum": ["all", "web", "mobile"],
                "description": "Which interface to sense",
                "default": "all",
            },
            "include_messages": {
                "type": "boolean",
                "description": "Include recent messages (default true)",
                "default": True,
            },
            "message_limit": {
                "type": "integer",
                "description": "Max messages to return (default 10)",
                "default": 10,
            },
        },
    }
    requires_permission = False
    
    def __init__(self):
        self._web_interface = None
    
    def set_web_interface(self, web_interface):
        """Set reference to the web interface for live state access."""
        self._web_interface = web_interface
    
    async def execute(
        self,
        interface: str = "all",
        include_messages: bool = True,
        message_limit: int = 10,
    ) -> str:
        if not self._web_interface:
            return "UI sensing not available - web interface not connected"
        
        results = []
        
        if interface in ["all", "web"]:
            web_state = await self._sense_web_ui(include_messages, message_limit)
            results.append(web_state)
        
        if interface in ["all", "mobile"]:
            mobile_state = await self._sense_mobile_ui(include_messages, message_limit)
            results.append(mobile_state)
        
        return "\n\n".join(results)
    
    async def _sense_web_ui(self, include_messages: bool, message_limit: int) -> str:
        """Sense the web UI state."""
        lines = ["# Web UI State\n"]
        
        try:
            ui_state = getattr(self._web_interface, '_ui_state', {})
            connections = getattr(self._web_interface, 'connections', set())
            messages = getattr(self._web_interface, '_current_messages', [])
            
            lines.append(f"**Connected Clients:** {len(connections)}")
            lines.append(f"**Thinking:** {ui_state.get('thinking', False)}")
            lines.append(f"**Active Tab:** {ui_state.get('active_tab', 'chat')}")
            
            tool_calls = ui_state.get('tool_calls', [])
            if tool_calls:
                lines.append("\n**Active Tool Calls:**")
                for tc in tool_calls[-5:]:
                    status = tc.get('status', 'unknown')
                    name = tc.get('name', 'unknown')
                    lines.append(f"  - {name}: {status}")
            
            if include_messages and messages:
                lines.append(f"\n**Recent Messages ({len(messages)} total):**")
                for msg in messages[-message_limit:]:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]
                    if len(msg.get('content', '')) > 200:
                        content += "..."
                    lines.append(f"  [{role}]: {content}")
            
            last_response = ui_state.get('last_response')
            if last_response:
                lines.append("\n**Last Response Preview:**")
                lines.append(f"  {last_response[:300]}{'...' if len(last_response) > 300 else ''}")
                
        except Exception as e:
            lines.append(f"Error sensing web UI: {e}")
        
        return "\n".join(lines)
    
    async def _sense_mobile_ui(self, include_messages: bool, message_limit: int) -> str:
        """Sense the mobile UI state."""
        lines = ["# Mobile UI State\n"]
        
        try:
            ui_state = getattr(self._web_interface, '_ui_state', {})
            mobile_connections = getattr(self._web_interface, '_mobile_connections', set())
            mobile_http_clients = getattr(self._web_interface, '_mobile_http_clients', 0)
            
            total_mobile = len(mobile_connections) + mobile_http_clients
            lines.append(f"**Total Mobile Clients:** {total_mobile}")
            lines.append(f"  - WebSocket: {len(mobile_connections)}")
            lines.append(f"  - HTTP (POST): {mobile_http_clients}")
            lines.append(f"**Thinking:** {ui_state.get('thinking', False)}")
            
            from core.mobile_auth import get_mobile_auth
            auth = get_mobile_auth()
            sessions = getattr(auth, 'sessions', {})
            credentials = getattr(auth, 'credentials', {})
            
            lines.append(f"\n**Mobile Sessions:** {len(sessions)} active")
            lines.append(f"**WebAuthn Credentials:** {len(credentials)} registered")
            
            if include_messages:
                last_user = ui_state.get('last_user_input')
                last_resp = ui_state.get('last_response')
                
                if last_user:
                    lines.append("\n**Last User Input:**")
                    lines.append(f"  {last_user[:200]}{'...' if len(last_user) > 200 else ''}")
                
                if last_resp:
                    lines.append("\n**Last Response:**")
                    lines.append(f"  {last_resp[:200]}{'...' if len(last_resp) > 200 else ''}")
            
            lines.append("\n**Mobile URL:** https://your-host.your-tailnet.ts.net/mobile")
            
        except Exception as e:
            lines.append(f"Error sensing mobile UI: {e}")
        
        return "\n".join(lines)


class UIWatchTool(Tool):
    """Watch UI for changes and events."""
    
    name = "ui_watch"
    description = """Watch for UI events and changes in real-time.

Returns recent UI events like:
- New user messages
- Response starts/completes
- Tool call start/complete
- Tab changes
- Client connections/disconnections

Use this to stay aware of user activity across interfaces.
"""
    parameters = {
        "type": "object",
        "properties": {
            "event_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Event types to watch (message, tool_call, thinking, connection)",
            },
            "since_seconds": {
                "type": "integer",
                "description": "Get events from last N seconds (default 60)",
                "default": 60,
            },
        },
    }
    requires_permission = False
    
    def __init__(self):
        self._web_interface = None
        self._event_log = []
        self._max_events = 100
    
    def set_web_interface(self, web_interface):
        self._web_interface = web_interface
    
    def log_event(self, event_type: str, data: dict):
        """Log a UI event."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        self._event_log.append(event)
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events:]
    
    async def execute(
        self,
        event_types: list = None,
        since_seconds: int = 60,
    ) -> str:
        if event_types is None:
            event_types = ["message", "tool_call", "thinking", "connection"]
        
        if not self._event_log:
            return "No UI events logged yet. The event tracking will start after first user interaction."
        
        cutoff = time.time() - since_seconds
        recent_events = [
            e for e in self._event_log
            if e["timestamp"] > cutoff and e["type"] in event_types
        ]
        
        if not recent_events:
            return f"No events in the last {since_seconds} seconds."
        
        lines = [f"# UI Events (last {since_seconds}s)\n"]
        
        for event in recent_events[-20:]:
            ts = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
            etype = event["type"]
            data = event["data"]
            
            if etype == "message":
                role = data.get("role", "?")
                content = data.get("content", "")[:50]
                lines.append(f"[{ts}] 📨 {role}: {content}...")
            elif etype == "tool_call":
                name = data.get("name", "?")
                status = data.get("status", "?")
                icon = "🔧" if status == "start" else "✅" if status == "complete" else "❌"
                lines.append(f"[{ts}] {icon} Tool: {name} ({status})")
            elif etype == "thinking":
                status = data.get("status", "?")
                icon = "💭" if status == "start" else "💭"
                lines.append(f"[{ts}] {icon} Thinking: {status}")
            elif etype == "connection":
                client_type = data.get("client_type", "?")
                action = data.get("action", "?")
                lines.append(f"[{ts}] 🔌 {client_type}: {action}")
        
        return "\n".join(lines)


class UIVisualizeTool(Tool):
    """Get a visual representation of the current UI state."""
    
    name = "ui_visualize"
    description = """Get a visual ASCII representation of the current UI state.

Shows a diagram-like view of:
- Web UI layout and current state
- Mobile UI layout and current state
- Active elements and their states

Useful for understanding what users are seeing.
"""
    parameters = {
        "type": "object",
        "properties": {
            "interface": {
                "type": "string",
                "enum": ["all", "web", "mobile"],
                "default": "all",
            },
        },
    }
    requires_permission = False
    
    def __init__(self):
        self._web_interface = None
    
    def set_web_interface(self, web_interface):
        self._web_interface = web_interface
    
    async def execute(self, interface: str = "all") -> str:
        if not self._web_interface:
            return "UI visualization not available"
        
        results = []
        
        if interface in ["all", "web"]:
            results.append(self._visualize_web())
        
        if interface in ["all", "mobile"]:
            results.append(self._visualize_mobile())
        
        return "\n\n".join(results)
    
    def _visualize_web(self) -> str:
        ui_state = getattr(self._web_interface, '_ui_state', {})
        connections = getattr(self._web_interface, 'connections', set())
        messages = getattr(self._web_interface, '_current_messages', [])
        thinking = ui_state.get('thinking', False)
        active_tab = ui_state.get('active_tab', 'chat')
        
        thinking_indicator = "💭 THINKING..." if thinking else ""
        
        lines = [
            "╔════════════════════════════════════════════════════════════╗",
            "║                    JARVIS WEB UI                           ║",
            f"║  Clients: {len(connections):<3}  {thinking_indicator:<20}                ║",
            "╠════════════════════════════════════════════════════════════╣",
            "║  [Chat] [Tools] [Terminal] [Settings] [Dashboard] [Visual] ║",
            f"║     {'▲' if active_tab == 'chat' else ' '}       {'▲' if active_tab == 'tools' else ' '}        {'▲' if active_tab == 'terminal' else ' '}         {'▲' if active_tab == 'settings' else ' '}          {'▲' if active_tab == 'dashboard' else ' '}          {'▲' if active_tab == 'visual' else ' '}       ║",
            "╠════════════════════════════════════════════════════════════╣",
            "║  MESSAGES:                                                  ║",
        ]
        
        if messages:
            for msg in messages[-3:]:
                role = msg.get('role', '?')[:4].upper()
                content = msg.get('content', '')[:40]
                lines.append(f"║    [{role}] {content:<42}║")
        else:
            lines.append("║    (no messages yet)                                        ║")
        
        lines.extend([
            "║                                                             ║",
            "║  ┌─────────────────────────────────────────────────────┐   ║",
            "║  │ Type your message here...                          │   ║",
            "║  │                                                     │   ║",
            "║  └─────────────────────────────────────────────────────┘   ║",
            "╚════════════════════════════════════════════════════════════╝",
        ])
        
        return "\n".join(lines)
    
    def _visualize_mobile(self) -> str:
        ui_state = getattr(self._web_interface, '_ui_state', {})
        mobile_connections = getattr(self._web_interface, '_mobile_connections', set())
        thinking = ui_state.get('thinking', False)
        
        thinking_indicator = "💭" if thinking else "  "
        
        lines = [
            "╔══════════════════════════╗",
            "║     JARVIS MOBILE        ║",
            f"║  Clients: {len(mobile_connections):<2}  {thinking_indicator}          ║",
            "╠══════════════════════════╣",
            "║                          ║",
            "║    📱 Mobile PWA Active   ║",
            "║                          ║",
            "║  ┌────────────────────┐  ║",
            "║  │ [Chat]  [Tools]    │  ║",
            "║  │ [Dashboard][Set]   │  ║",
            "║  └────────────────────┘  ║",
            "║                          ║",
            "║  Messages shown above    ║",
            "║  ┌────────────────────┐  ║",
            "║  │ Ask Jarvis...      │  ║",
            "║  └────────────────────┘  ║",
            "║  [📎] [🗑️] [Send ➤]      ║",
            "╚══════════════════════════╝",
        ]
        
        return "\n".join(lines)


# Tool instances and registration
_ui_sense_tool = UISenseTool()
_ui_watch_tool = UIWatchTool()
_ui_visualize_tool = UIVisualizeTool()

TOOLS = [_ui_sense_tool, _ui_watch_tool, _ui_visualize_tool]


def set_web_interface(web_interface):
    """Set the web interface reference for all UI tools."""
    _ui_sense_tool.set_web_interface(web_interface)
    _ui_watch_tool.set_web_interface(web_interface)
    _ui_visualize_tool.set_web_interface(web_interface)


def log_ui_event(event_type: str, data: dict):
    """Log a UI event for the watch tool."""
    _ui_watch_tool.log_event(event_type, data)
