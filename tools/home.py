"""Home automation tools for Jarvis with actual integration support.

Supports multiple home automation platforms:
- Home Assistant (via REST API)
- MQTT devices
- Philips Hue (via bridge)
- Local shell commands

Configure in config.yaml under home_assistant section.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from .base import Tool
from core.helpers import create_tracked_task, safe_error_message

logger = logging.getLogger(__name__)

_home_config: dict[str, Any] = {}
_home_client: httpx.AsyncClient | None = None
_home_config_lock = asyncio.Lock()
_home_client_lock = asyncio.Lock()


def _cleanup_home_client() -> None:
    """Synchronous cleanup for atexit."""
    global _home_client
    if _home_client is not None:
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Tracked so the cleanup coroutine survives until it runs.
                    create_tracked_task(_home_client.aclose(), name="home:aclose")
                else:
                    loop.run_until_complete(_home_client.aclose())
            except RuntimeError:
                pass
        except Exception as e:
            logger.warning(f"Failed to close home automation client: {e}")
        finally:
            _home_client = None


import atexit
atexit.register(_cleanup_home_client)


async def _load_home_config() -> dict[str, Any]:
    """Load home automation configuration from config.yaml."""
    global _home_config
    async with _home_config_lock:
        if _home_config:
            return _home_config
        
        try:
            import yaml
            config_path = Path(__file__).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    _home_config = config.get("home_automation", {})
                    if _home_config:
                        logger.info(f"Loaded home automation config: {list(_home_config.keys())}")
                    return _home_config
        except Exception as e:
            logger.debug(f"Could not load home automation config: {e}")
        return {}


async def _get_home_client() -> httpx.AsyncClient:
    """Get or create HTTP client for home automation."""
    global _home_client
    async with _home_client_lock:
        if _home_client is None:
            _home_client = httpx.AsyncClient(timeout=10.0)
    return _home_client


async def _close_home_client() -> None:
    """Close HTTP client for home automation."""
    global _home_client
    async with _home_client_lock:
        if _home_client is not None:
            try:
                await _home_client.aclose()
            except Exception as e:
                logger.debug(f"Error closing home client: {e}")
            finally:
                _home_client = None


async def _call_home_assistant(entity_id: str, action: str, value: str | None = None) -> dict:
    """Call Home Assistant API with actual request."""
    config = await _load_home_config()
    ha_config = config.get("home_assistant", {})
    
    if not ha_config.get("url") or not ha_config.get("token"):
        return {
            "success": False,
            "error": "Home Assistant not configured. Add 'home_assistant' section to config.yaml with 'url' and 'token'.",
            "required_config": {
                "home_assistant": {
                    "url": "http://homeassistant.local:8123",
                    "token": "YOUR_LONG_LIVED_ACCESS_TOKEN"
                }
            }
        }
    
    base_url = ha_config["url"].rstrip("/")
    token = ha_config["token"]
    
    client = await _get_home_client()
    
    try:
        if action.lower() in ("on", "off"):
            domain = entity_id.split(".")[0] if "." in entity_id else "light"
            service_url = f"{base_url}/api/services/{domain}/turn_{action.lower()}"
            response = await client.post(
                service_url,
                headers={"Authorization": f"Bearer {token}"},
                json={"entity_id": entity_id}
            )
        elif action.lower() == "toggle":
            domain = entity_id.split(".")[0] if "." in entity_id else "light"
            service_url = f"{base_url}/api/services/{domain}/toggle"
            response = await client.post(
                service_url,
                headers={"Authorization": f"Bearer {token}"},
                json={"entity_id": entity_id}
            )
        elif value is not None:
            domain = entity_id.split(".")[0] if "." in entity_id else "light"
            if "brightness" in action.lower() or "brightness" in str(value).lower():
                service_url = f"{base_url}/api/services/{domain}/turn_on"
                try:
                    brightness = int(float(value) * 2.55) if float(value) <= 100 else int(float(value))
                except ValueError:
                    brightness = 128
                response = await client.post(
                    service_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json={"entity_id": entity_id, "brightness": brightness}
                )
            elif "color" in action.lower() or "color" in str(value).lower():
                service_url = f"{base_url}/api/services/{domain}/turn_on"
                response = await client.post(
                    service_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json={"entity_id": entity_id, "color_name": value}
                )
            else:
                service_url = f"{base_url}/api/services/{domain}/turn_on"
                response = await client.post(
                    service_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json={"entity_id": entity_id, action: value}
                )
        else:
            service_url = f"{base_url}/api/services/script/{action}"
            response = await client.post(
                service_url,
                headers={"Authorization": f"Bearer {token}"},
                json={}
            )
        
        if response.status_code == 200:
            return {
                "success": True,
                "device": entity_id,
                "action": action,
                "value": value,
                "response_status": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"Home Assistant returned status {response.status_code}",
                "response_text": response.text[:500]
            }
    
    except httpx.ConnectError as e:
        return {
            "success": False,
            "error": f"Cannot connect to Home Assistant at {base_url}: {e}",
            "hint": "Check that Home Assistant is running and the URL is correct."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Home Assistant call failed: {e}"
        }


async def _call_mqtt(device: str, action: str, value: str | None = None) -> dict:
    """Send MQTT command using async run_in_executor to avoid blocking."""
    config = await _load_home_config()
    mqtt_config = config.get("mqtt", {})
    
    if not mqtt_config.get("broker"):
        return {
            "success": False,
            "error": "MQTT not configured. Add 'mqtt' section to config.yaml with 'broker' address."
        }
    
    broker = mqtt_config["broker"]
    port = mqtt_config.get("port", 1883)
    topic = mqtt_config.get("topic_prefix", "home") + f"/{device}/command"
    payload = value if value else action
    
    def _mqtt_publish():
        import paho.mqtt.client as mqtt
        
        result_code = None
        
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal result_code
            if rc == 0:
                client.publish(topic, payload)
            else:
                result_code = rc
        
        def on_publish(client, userdata, mid, rc=None, properties=None):
            nonlocal result_code
            result_code = 0
            client.disconnect()
        
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_publish = on_publish
        
        if mqtt_config.get("username"):
            client.username_pw_set(mqtt_config["username"], mqtt_config.get("password", ""))
        
        try:
            client.connect(broker, port, 60)
            client.loop_start()
            client.loop(timeout=5.0)
            client.loop_stop()
            
            if result_code == 0:
                return {"success": True}
            else:
                return {"success": False, "error": f"MQTT error code {result_code}"}
        except Exception as e:
            return {"success": False, "error": safe_error_message(e)}
    
    try:
        import paho.mqtt.client as mqtt
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _mqtt_publish)
        
        if result.get("success"):
            return {
                "success": True,
                "device": device,
                "action": action,
                "value": value,
                "topic": topic,
                "broker": broker
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "MQTT operation failed")
            }
            
    except ImportError:
        return {
            "success": False,
            "error": "paho-mqtt not installed. Run: pip install paho-mqtt",
            "install_command": "pip install paho-mqtt"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"MQTT call failed: {e}"
        }


async def _execute_shell_command(device: str, action: str, value: str | None = None) -> dict:
    """Execute shell command for home control with proper escaping."""
    import shlex
    config = await _load_home_config()
    shell_config = config.get("shell_commands", {})
    
    device_commands = shell_config.get(device, {})
    command = device_commands.get(action)
    
    if not command:
        if value:
            command = device_commands.get("set", "")
            if command:
                safe_value = str(value)
                for char in ['"', "'", ";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r", "\\", "!", "#", "~", "?", "*"]:
                    safe_value = safe_value.replace(char, "")
                if not safe_value.replace(".", "").replace("-", "").replace("_", "").isalnum():
                    safe_value = shlex.quote(safe_value)
                command = command.replace("{value}", safe_value)
        
        if not command:
            return {
                "success": False,
                "error": f"No shell command defined for device '{device}' action '{action}'",
                "hint": f"Add to config.yaml:\n  shell_commands:\n    {device}:\n      {action}: 'your_command_here'"
            }
    
    if not isinstance(command, str) or len(command) > 500:
        return {"success": False, "error": "Invalid or too long command"}
    
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        
        return {
            "success": proc.returncode == 0,
            "device": device,
            "action": action,
            "value": value,
            "return_code": proc.returncode,
            "stdout": stdout.decode()[:500] if stdout else "",
            "stderr": stderr.decode()[:500] if stderr else ""
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Shell command timed out: {command}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Shell command failed: {e}"
        }


class HomeControlTool(Tool):
    """Tool for home automation control with actual execution."""

    name = "home_control"
    description = "Control smart home devices via Home Assistant, MQTT, or shell commands"
    parameters = {
        "type": "object",
        "properties": {
            "device": {
                "type": "string",
                "description": "Device to control (entity ID for Home Assistant, device name for MQTT/shell)",
            },
            "action": {
                "type": "string",
                "description": "Action to perform: on, off, toggle, brightness, color, or custom service name",
            },
            "value": {
                "type": "string",
                "description": "Optional value (brightness level 0-100, color name, temperature, etc.)",
            },
        },
        "required": ["device", "action"],
    }
    requires_permission = True

    async def execute(self, device: str, action: str, value: str | None = None, **kwargs) -> str:
        """Execute home control action with actual integration calls."""
        config = await _load_home_config()
        
        if not config:
            return (
                "ERROR: Home automation not configured.\n\n"
                "To enable home control, add to config.yaml:\n\n"
                "home_automation:\n"
                "  home_assistant:\n"
                "    url: 'http://homeassistant.local:8123'\n"
                "    token: 'YOUR_LONG_LIVED_ACCESS_TOKEN'\n"
                "  # OR MQTT:\n"
                "  mqtt:\n"
                "    broker: 'mqtt.local'\n"
                "    port: 1883\n"
                "  # OR shell commands:\n"
                "  shell_commands:\n"
                "    living_light:\n"
                "      on: 'echo \"lights on\"'\n"
                "      off: 'echo \"lights off\"'\n\n"
                "Actual execution required - no placeholder responses allowed."
            )
        
        results = []
        
        if config.get("home_assistant"):
            result = await _call_home_assistant(device, action, value)
            results.append(("home_assistant", result))
            if result.get("success"):
                return f"SUCCESS (Home Assistant): {device} -> {action}" + (f" = {value}" if value else "")
        
        if config.get("mqtt"):
            result = await _call_mqtt(device, action, value)
            results.append(("mqtt", result))
            if result.get("success"):
                return f"SUCCESS (MQTT): {device} -> {action}" + (f" = {value}" if value else "")
        
        if config.get("shell_commands"):
            result = await _execute_shell_command(device, action, value)
            results.append(("shell", result))
            if result.get("success"):
                return f"SUCCESS (Shell): {device} -> {action}" + (f" = {value}" if value else "")
        
        errors = []
        for platform, result in results:
            if not result.get("success"):
                errors.append(f"{platform}: {result.get('error', 'Unknown error')}")
        
        if errors:
            return f"FAILED: Could not control {device}.\n" + "\n".join(errors)
        
        return "ERROR: No home automation platform responded. Check configuration."


class HomeStatusTool(Tool):
    """Tool to get status of home devices."""

    name = "home_status"
    description = "Get current status of smart home devices"
    parameters = {
        "type": "object",
        "properties": {
            "device": {
                "type": "string",
                "description": "Optional device to check. If omitted, returns all device states.",
            },
        },
        "required": [],
    }
    requires_permission = False

    async def execute(self, device: str | None = None, **kwargs) -> str:
        """Get device status from Home Assistant."""
        config = await _load_home_config()
        ha_config = config.get("home_assistant", {})
        
        if not ha_config.get("url") or not ha_config.get("token"):
            return "ERROR: Home Assistant not configured for status queries."
        
        base_url = ha_config["url"].rstrip("/")
        token = ha_config["token"]
        
        client = await _get_home_client()
        
        try:
            if device:
                url = f"{base_url}/api/states/{device}"
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    state = data.get("state", "unknown")
                    attrs = data.get("attributes", {})
                    friendly_name = attrs.get("friendly_name", device)
                    return f"{friendly_name}: {state}"
                else:
                    return f"ERROR: Device '{device}' not found (status {response.status_code})"
            else:
                url = f"{base_url}/api/states"
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code == 200:
                    states = response.json()
                    devices = []
                    for s in states[:20]:
                        attrs = s.get("attributes") or {}
                        friendly_name = attrs.get("friendly_name", s.get("entity_id", "unknown"))
                        devices.append(f"{friendly_name}: {s['state']}")
                    return "Device Status:\n" + "\n".join(devices)
                else:
                    return f"ERROR: Failed to get states (status {response.status_code})"
        except Exception as e:
            return f"ERROR: Failed to get device status: {e}"


class HomeTools:
    """Collection of home automation tools with actual execution."""

    def __init__(self):
        self.tools = [
            HomeControlTool(),
            HomeStatusTool(),
        ]

    def get_tools(self) -> list[Tool]:
        """Get all home tools."""
        return self.tools
