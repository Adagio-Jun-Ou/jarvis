"""System Sensing Tools - Allow Jarvis to sense its entire runtime environment.

This module provides system monitoring tools that are always available to Jarvis:
- SystemSenseTool: Comprehensive system state monitoring
- SystemWatchTool: Threshold-based alerting and event watching

Integration points:
- Background scheduler: Can be registered for periodic monitoring
- Security system: Alerts on security-relevant events
- Subsystem status: Monitors daydream, evolution, refine, etc.

The system_sense tools are always-available and require no permission.
"""

import asyncio
import os
import time
import psutil
import logging
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field

from .base import Tool
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system metrics."""
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    memory_percent: float = 0.0
    memory_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    disk_gb: float = 0.0
    disk_total_gb: float = 0.0
    network_connections: int = 0
    process_count: int = 0
    load_avg: tuple = (0.0, 0.0, 0.0)
    uptime_seconds: float = 0.0


@dataclass
class SubsystemStatus:
    """Status of a Jarvis subsystem."""
    name: str
    running: bool
    healthy: bool
    last_activity: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


class SystemSenseTool(Tool):
    """Sense the entire Jarvis system state - processes, resources, subsystems."""
    
    name = "system_sense"
    description = """Get comprehensive awareness of the entire Jarvis system runtime.

Returns:
- System resources (CPU, memory, disk, network)
- Jarvis process status and resource usage
- All subsystem states (daydream, evolution, refine, etc.)
- Active connections and clients
- Background task status
- Container/Docker status (if applicable)
- Model status and availability

Use this to:
- Understand current system load and capacity
- Check if subsystems are running properly
- Diagnose performance issues
- Know what's happening across the entire system
"""
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "enum": ["all", "resources", "subsystems", "processes", "network", "models"],
                "description": "Area to focus on (default: all)",
                "default": "all",
            },
            "detailed": {
                "type": "boolean",
                "description": "Include detailed metrics (default: false)",
                "default": False,
            },
        },
    }
    requires_permission = False
    
    def __init__(self):
        self._agent = None
        self._web_interface = None
        self._start_time = time.time()
    
    def set_agent(self, agent):
        """Set reference to Jarvis agent."""
        self._agent = agent
    
    def set_web_interface(self, web_interface):
        """Set reference to web interface."""
        self._web_interface = web_interface
    
    async def execute(
        self,
        focus: str = "all",
        detailed: bool = False,
    ) -> str:
        results = []
        
        if focus in ["all", "resources"]:
            results.append(await self._sense_resources(detailed))
        
        if focus in ["all", "processes"]:
            results.append(await self._sense_processes(detailed))
        
        if focus in ["all", "subsystems"]:
            results.append(await self._sense_subsystems(detailed))
        
        if focus in ["all", "network"]:
            results.append(await self._sense_network(detailed))
        
        if focus in ["all", "models"]:
            results.append(await self._sense_models(detailed))
        
        return "\n\n".join(results)
    
    async def _sense_resources(self, detailed: bool) -> str:
        """Sense system resource usage."""
        lines = ["# System Resources\n"]
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_cores = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            lines.append(f"**CPU:** {cpu_percent:.1f}% ({cpu_cores} cores)")
            if detailed:
                lines.append(f"  Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
                per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
                if len(per_cpu) <= 8:
                    lines.append(f"  Per Core: {', '.join(f'{c:.0f}%' for c in per_cpu)}")
            
            # Memory
            mem = psutil.virtual_memory()
            mem_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            
            lines.append(f"**Memory:** {mem.percent:.1f}% ({mem_gb:.1f}GB / {mem_total_gb:.1f}GB)")
            if detailed:
                lines.append(f"  Available: {mem.available / (1024**3):.1f}GB")
                lines.append(f"  Cached: {mem.cached / (1024**3):.1f}GB")
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            lines.append(f"**Disk:** {disk.percent:.1f}% ({disk_gb:.1f}GB / {disk_total_gb:.1f}GB)")
            
            # Swap
            swap = psutil.swap_memory()
            if swap.total > 0:
                lines.append(f"**Swap:** {swap.percent:.1f}% ({swap.used / (1024**3):.1f}GB / {swap.total / (1024**3):.1f}GB)")
            
            # Uptime
            uptime = time.time() - self._start_time
            hours = int(uptime // 3600)
            mins = int((uptime % 3600) // 60)
            lines.append(f"**Session Uptime:** {hours}h {mins}m")
            
        except Exception as e:
            lines.append(f"Error sensing resources: {e}")
        
        return "\n".join(lines)
    
    async def _sense_processes(self, detailed: bool) -> str:
        """Sense Jarvis-related processes."""
        lines = ["# Jarvis Processes\n"]
        
        try:
            current_pid = os.getpid()
            lines.append(f"**Main Process:** PID {current_pid}")
            
            # Find all Python processes related to jarvis
            jarvis_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    if any('jarvis' in str(c).lower() or 'run.py' in str(c) for c in cmdline):
                        jarvis_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Also find n8n
            n8n_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    if 'n8n' in str(cmdline).lower():
                        n8n_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if jarvis_procs:
                lines.append(f"\n**Jarvis Processes ({len(jarvis_procs)}):**")
                for proc in jarvis_procs[:5]:
                    try:
                        cpu = proc.cpu_percent(interval=0.05)
                        mem = proc.memory_info()
                        mem_mb = mem.rss / (1024**2)
                        lines.append(f"  PID {proc.pid}: {cpu:.1f}% CPU, {mem_mb:.0f}MB RAM")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            
            if n8n_procs:
                lines.append(f"\n**n8n Processes ({len(n8n_procs)}):**")
                for proc in n8n_procs[:3]:
                    try:
                        cpu = proc.cpu_percent(interval=0.05)
                        mem = proc.memory_info()
                        mem_mb = mem.rss / (1024**2)
                        lines.append(f"  PID {proc.pid}: {cpu:.1f}% CPU, {mem_mb:.0f}MB RAM")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            
            # Docker containers
            if detailed:
                docker_status = await self._check_docker()
                if docker_status:
                    lines.append("\n**Docker:**")
                    lines.append(docker_status)
            
            # Total process count
            total_procs = len(psutil.pids())
            lines.append(f"\n**Total System Processes:** {total_procs}")
            
        except Exception as e:
            lines.append(f"Error sensing processes: {e}")
        
        return "\n".join(lines)
    
    async def _check_docker(self) -> Optional[str]:
        """Check Docker container status."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            result = stdout.decode("utf-8", errors="replace")
            if result.strip():
                lines = result.strip().split('\n')
                return '\n'.join(f"  {line}" for line in lines[:5])
        except Exception as e:
            logger.warning(f"Failed to check Docker status: {e}")
        return None
    
    async def _sense_subsystems(self, detailed: bool) -> str:
        """Sense Jarvis subsystem status."""
        lines = ["# Jarvis Subsystems\n"]
        
        try:
            # Daydream Engine
            daydream_status = await self._get_subsystem_status("daydream")
            lines.append(f"**Daydream:** {'✓ Running' if daydream_status.running else '✗ Stopped'}")
            if daydream_status.running and daydream_status.metrics:
                lines.append(f"  Topics explored: {daydream_status.metrics.get('topics', 0)}")
                lines.append(f"  Insights: {daydream_status.metrics.get('insights', 0)}")
            
            # Self-Refine
            refine_status = await self._get_subsystem_status("refine")
            lines.append(f"\n**Self-Refine:** {'✓ Running' if refine_status.running else '✗ Stopped'}")
            if refine_status.running and refine_status.metrics:
                lines.append(f"  Pending tasks: {refine_status.metrics.get('pending', 0)}")
                lines.append(f"  Applied: {refine_status.metrics.get('applied', 0)}")
            
            # Evolution
            evolution_status = await self._get_subsystem_status("evolution")
            lines.append(f"\n**Evolution:** {'✓ Running' if evolution_status.running else '✗ Stopped'}")
            if evolution_status.running and evolution_status.metrics:
                lines.append(f"  Generations: {evolution_status.metrics.get('generations', 0)}")
                lines.append(f"  Fitness: {evolution_status.metrics.get('fitness', 'N/A')}")
            
            # Knowledge Graph
            kg_status = await self._get_subsystem_status("knowledge")
            lines.append(f"\n**Knowledge Graph:** {'✓ Active' if kg_status.running else '✗ Inactive'}")
            if kg_status.running and kg_status.metrics:
                lines.append(f"  Nodes: {kg_status.metrics.get('nodes', 0)}")
                lines.append(f"  Edges: {kg_status.metrics.get('edges', 0)}")
            
            # Background Scheduler
            bg_status = await self._get_subsystem_status("background")
            lines.append(f"\n**Background Scheduler:** {'✓ Active' if bg_status.running else '✗ Inactive'}")
            if bg_status.metrics:
                lines.append(f"  Tasks: {bg_status.metrics.get('task_count', 0)}")
            
            # Memory Systems
            memory_status = await self._get_subsystem_status("memory")
            lines.append(f"\n**Memory Systems:** {'✓ Active' if memory_status.running else '✗ Inactive'}")
            if memory_status.metrics:
                lines.append(f"  Persistent: {memory_status.metrics.get('persistent', 0)}")
                lines.append(f"  Insights: {memory_status.metrics.get('insights', 0)}")
            
            # n8n
            n8n_status = await self._get_subsystem_status("n8n")
            lines.append(f"\n**n8n:** {'✓ Running' if n8n_status.running else '✗ Stopped'}")
            if n8n_status.running and n8n_status.metrics:
                lines.append(f"  Workflows: {n8n_status.metrics.get('workflows', 0)}")
            
        except Exception as e:
            lines.append(f"Error sensing subsystems: {e}")
        
        return "\n".join(lines)
    
    async def _get_subsystem_status(self, name: str) -> SubsystemStatus:
        """Get status for a specific subsystem."""
        status = SubsystemStatus(name=name, running=False, healthy=False)
        
        try:
            if name == "daydream":
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                status.running = engine is not None and getattr(engine, '_running', False)
                if status.running:
                    status.metrics = {
                        'topics': len(getattr(engine, 'exploration_topics', [])),
                        'insights': getattr(engine, 'total_insights', 0),
                    }
            
            elif name == "refine":
                from core.self_refine import get_self_refine
                refine = get_self_refine()
                status.running = refine is not None
                if status.running:
                    stats = getattr(refine, 'get_statistics', lambda: {})()
                    status.metrics = {
                        'pending': stats.get('pending', 0),
                        'applied': stats.get('applied', 0),
                    }
            
            elif name == "evolution":
                from core.evolution_orchestrator import get_evolution_orchestrator
                evo = get_evolution_orchestrator()
                status.running = evo is not None and getattr(evo, '_running', False)
                if status.running:
                    status.metrics = {
                        'generations': len(getattr(evo, 'generations', [])),
                        'fitness': getattr(evo, 'best_fitness', 'N/A'),
                    }
            
            elif name == "knowledge":
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                status.running = kg is not None
                if status.running:
                    status.metrics = {
                        'nodes': len(getattr(kg, 'nodes', [])),
                        'edges': len(getattr(kg, 'edges', [])),
                    }
            
            elif name == "background":
                from core.background import get_background_scheduler
                scheduler = get_background_scheduler()
                status.running = scheduler is not None
                if status.running:
                    status.metrics = {
                        'task_count': len(getattr(scheduler, 'tasks', [])),
                    }
            
            elif name == "memory":
                status.running = True
                try:
                    from core.persistent_memory import get_persistent_memory
                    from core.insight_bridge import get_insight_bridge
                    from core.insight_review import get_insight_review
                    pm = get_persistent_memory()
                    ib = get_insight_bridge()
                    review = get_insight_review()
                    pm_count = len(getattr(pm, 'memories', []))
                    # Try insight_review first (indexed insights)
                    try:
                        stats = review.get_stats()
                        insight_count = stats.get('total_insights', 0)
                    except Exception:
                        # Fallback to insight_bridge
                        insight_count = len(getattr(ib, 'processed_insights', []))
                    status.metrics = {
                        'persistent': pm_count,
                        'insights': insight_count,
                    }
                except Exception as e:
                    logger.warning(f"Failed to get memory subsystem status: {e}")
            
            elif name == "n8n":
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "pgrep", "-f", "n8n",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
                    result = stdout.decode("utf-8", errors="replace").strip()
                    status.running = bool(result)
                    if status.running:
                        status.metrics = {'workflows': 'active'}
                except Exception as e:
                    logger.warning(f"Failed to check n8n subsystem status: {e}")
            
            status.healthy = status.running
            
        except Exception as e:
            status.error = safe_error_message(e, 'Subsystem status check failed')
        
        return status
    
    async def _sense_network(self, detailed: bool) -> str:
        """Sense network connections."""
        lines = ["# Network Status\n"]
        
        try:
            # WebSocket connections
            if self._web_interface:
                ws_connections = getattr(self._web_interface, 'connections', set())
                mobile_connections = getattr(self._web_interface, '_mobile_connections', set())
                
                lines.append("**WebSocket Connections:**")
                lines.append(f"  Web: {len(ws_connections)}")
                lines.append(f"  Mobile: {len(mobile_connections)}")
            
            # Network I/O
            net = psutil.net_io_counters()
            lines.append("\n**Network I/O:**")
            lines.append(f"  Sent: {net.bytes_sent / (1024**2):.1f}MB")
            lines.append(f"  Received: {net.bytes_recv / (1024**2):.1f}MB")
            
            if detailed:
                lines.append(f"  Packets In/Out: {net.packets_recv}/{net.packets_sent}")
            
            # Active connections
            connections = psutil.net_connections(kind='inet')
            established = len([c for c in connections if c.status == 'ESTABLISHED'])
            listen = len([c for c in connections if c.status == 'LISTEN'])
            
            lines.append("\n**Connections:**")
            lines.append(f"  Established: {established}")
            lines.append(f"  Listening: {listen}")
            
            # Ports
            if detailed:
                ports = {}
                for c in connections:
                    if c.status == 'LISTEN' and c.laddr:
                        port = c.laddr.port
                        ports[port] = ports.get(port, 0) + 1
                if ports:
                    lines.append("\n**Listening Ports:**")
                    for port, count in sorted(ports.items())[:10]:
                        lines.append(f"  {port}: {count} listeners")
            
        except Exception as e:
            lines.append(f"Error sensing network: {e}")
        
        return "\n".join(lines)
    
    async def _sense_models(self, detailed: bool) -> str:
        """Sense AI model availability."""
        lines = ["# AI Models\n"]
        
        try:
            # Check model coordinator
            if self._agent:
                coordinator = getattr(self._agent, 'model_coordinator', None)
                if coordinator:
                    models = getattr(coordinator, 'models', {})
                    primary = getattr(coordinator, 'primary_model', None)
                    backend = getattr(coordinator, 'backend_model', None)
                    
                    lines.append("**Available Models:**")
                    for model_name in models:
                        marker = " (primary)" if model_name == primary else ""
                        marker = " (backend)" if model_name == backend else marker
                        lines.append(f"  - {model_name}{marker}")
            
            # Check embedding model
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                interconnect = get_embedding_interconnect()
                if interconnect:
                    lines.append("\n**Embedding Model:**")
                    lines.append(f"  Dimension: {interconnect.dimension}")
            except Exception as e:
                logger.warning(f"Failed to check embedding model: {e}")
            
            # GPU/NPU availability
            if detailed:
                lines.append("\n**Acceleration:**")
                try:
                    import torch
                    if torch.cuda.is_available():
                        lines.append(f"  CUDA: {torch.cuda.get_device_name(0)}")
                    else:
                        lines.append("  CUDA: Not available")
                except ImportError:
                    lines.append("  CUDA: PyTorch not installed")
                
                try:
                    from core.npu import is_npu_available
                    if is_npu_available():
                        lines.append("  NPU: Available")
                    else:
                        lines.append("  NPU: Not available")
                except Exception:
                    lines.append("  NPU: Not detected")
            
        except Exception as e:
            lines.append(f"Error sensing models: {e}")
        
        return "\n".join(lines)


class SystemWatchTool(Tool):
    """Watch for system events and changes."""
    
    name = "system_watch"
    description = """Watch for system events like:
- High resource usage spikes
- Process starts/stops
- Subsystem health changes
- Connection events

Use this to monitor system health over time.
"""
    parameters = {
        "type": "object",
        "properties": {
            "thresholds": {
                "type": "boolean",
                "description": "Check if resources exceed thresholds (default true)",
                "default": True,
            },
            "recent_changes": {
                "type": "boolean",
                "description": "Check for recent process changes (default false)",
                "default": False,
            },
        },
    }
    requires_permission = False
    
    def __init__(self):
        self._thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'mem_warning': 80.0,
            'mem_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
        }
    
    async def execute(
        self,
        thresholds: bool = True,
        recent_changes: bool = False,
    ) -> str:
        lines = ["# System Watch\n"]
        
        alerts = []
        
        if thresholds:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            if cpu > self._thresholds['cpu_critical']:
                alerts.append(f"🔴 CRITICAL: CPU at {cpu:.1f}%")
            elif cpu > self._thresholds['cpu_warning']:
                alerts.append(f"🟡 WARNING: CPU at {cpu:.1f}%")
            
            if mem.percent > self._thresholds['mem_critical']:
                alerts.append(f"🔴 CRITICAL: Memory at {mem.percent:.1f}%")
            elif mem.percent > self._thresholds['mem_warning']:
                alerts.append(f"🟡 WARNING: Memory at {mem.percent:.1f}%")
            
            if disk.percent > self._thresholds['disk_critical']:
                alerts.append(f"🔴 CRITICAL: Disk at {disk.percent:.1f}%")
            elif disk.percent > self._thresholds['disk_warning']:
                alerts.append(f"🟡 WARNING: Disk at {disk.percent:.1f}%")
            
            if alerts:
                lines.append("**Alerts:**")
                for alert in alerts:
                    lines.append(f"  {alert}")
            else:
                lines.append("**Status:** ✓ All resources within normal limits")
        
        if recent_changes:
            lines.append("\n**Recent Process Changes:**")
            lines.append("  (Process watching not yet implemented)")
        
        return "\n".join(lines)


# Tool instances
_system_sense_tool = SystemSenseTool()
_system_watch_tool = SystemWatchTool()

TOOLS = [_system_sense_tool, _system_watch_tool]


def set_agent_for_system_sense(agent):
    """Set the agent reference for system sense tools."""
    _system_sense_tool.set_agent(agent)


def set_web_interface_for_system_sense(web_interface):
    """Set the web interface reference for system sense tools."""
    _system_sense_tool.set_web_interface(web_interface)


def get_system_sense_tool() -> SystemSenseTool:
    """Get the singleton SystemSenseTool instance."""
    return _system_sense_tool


def get_system_watch_tool() -> SystemWatchTool:
    """Get the singleton SystemWatchTool instance."""
    return _system_watch_tool


async def background_health_check() -> dict:
    """Background task for periodic health monitoring.
    
    Can be registered with BackgroundScheduler for continuous monitoring.
    Returns health status and any alerts.
    """
    try:
        tool = _system_sense_tool
        result = await tool.execute(focus="all", detailed=False)
        
        # Parse result for alerts
        alerts = []
        health_status = "healthy"
        
        if isinstance(result, str):
            lines = result.split('\n')
            for line in lines:
                if 'warning' in line.lower() or 'critical' in line.lower():
                    alerts.append(line.strip())
                    health_status = "warning"
                if 'error' in line.lower():
                    alerts.append(line.strip())
                    health_status = "error"
        
        return {
            "status": health_status,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
            "result": result[:500] if isinstance(result, str) else result,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "alerts": [safe_error_message(e, 'Health check failed')],
            "timestamp": datetime.now().isoformat(),
        }


def create_health_monitor_task(interval_seconds: float = 60.0) -> tuple[str, Callable, float]:
    """Create a health monitoring task for BackgroundScheduler.
    
    Args:
        interval_seconds: How often to run health checks
        
    Returns:
        Tuple of (name, coroutine, interval) for scheduler registration
    """
    return ("system_health_check", background_health_check, interval_seconds)


def integrate_with_security(security_tools: dict | None = None) -> dict:
    """Integrate system sensing with security tools.
    
    This function connects system monitoring with security alerts,
    allowing security tools to be notified of system-level events.
    
    Args:
        security_tools: Dict from tools.security_tools.get_security_tools()
        
    Returns:
        Integration status dict
    """
    integration_status = {
        "security_scanner": False,
        "security_status": False,
        "data_flow_monitor": False,
    }
    
    if security_tools is None:
        return integration_status
    
    # Connect system watch alerts to security auditor if available
    try:
        if "auditor" in security_tools:
            auditor = security_tools["auditor"]
            _system_watch_tool._security_auditor = auditor
            integration_status["security_status"] = True
            logger.info("System sense integrated with security auditor")
    except Exception as e:
        logger.warning(f"Failed to integrate with security auditor: {e}")
    
    # Connect DLP for sensitive data monitoring
    try:
        if "dlp" in security_tools:
            dlp = security_tools["dlp"]
            _system_sense_tool._dlp = dlp
            integration_status["security_scanner"] = True
            logger.info("System sense integrated with DLP")
    except Exception as e:
        logger.warning(f"Failed to integrate with DLP: {e}")
    
    return integration_status


# Add integration attributes to tools
setattr(_system_watch_tool, '_security_auditor', None)
setattr(_system_sense_tool, '_dlp', None)
setattr(_system_sense_tool, '_security_integrated', False)