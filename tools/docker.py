"""Docker automation tools for Jarvis.

This module provides tools for managing Docker containers, images,
and compose stacks. It enables Jarvis to design and execute
Docker-based automation workflows.

Key features:
- List containers and images
- Create/run containers with options
- Manage compose stacks
- Monitor container health
- Execute commands in containers
- Design container architectures
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .base import Tool
from core.helpers import is_protected_path, path_is_within, safe_error_message

# Host directories that must never be bind-mounted into a container. The
# filesystem root is handled separately (see _run below) because containment
# matching against "/" would reject every possible mount.
_BLOCKED_MOUNT_ROOTS = (
    "/etc", "/root", "/home", "/var", "/sys", "/proc", "/dev",
    "/boot", "/run", "/srv", "/opt", "/tmp",
)
_DOCKER_SOCKET_PATHS = ("/var/run/docker.sock", "/run/docker.sock")


class DockerTool(Tool):
    """Tool for Docker container and image management."""

    name = "docker"
    description = "Manage Docker containers, images, and compose stacks"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The Docker action to perform",
                "enum": [
                    "list_containers",
                    "list_images",
                    "pull",
                    "run",
                    "stop",
                    "remove",
                    "logs",
                    "exec",
                    "compose_up",
                    "compose_down",
                    "compose_ps",
                    "inspect",
                    "stats",
                    "design",
                ],
            },
            "container": {
                "type": "string",
                "description": "Container name or ID (for run, stop, remove, logs, exec, inspect)",
            },
            "image": {
                "type": "string",
                "description": "Image name (for run)",
            },
            "command": {
                "type": "string",
                "description": "Command to execute (for exec)",
            },
            "options": {
                "type": "object",
                "description": "Additional options for the action",
                "properties": {
                    "detach": {"type": "boolean"},
                    "ports": {"type": "array", "items": {"type": "string"}},
                    "volumes": {"type": "array", "items": {"type": "string"}},
                    "env": {"type": "array", "items": {"type": "string"}},
                    "network": {"type": "string"},
                    "restart": {"type": "string"},
                    "compose_file": {"type": "string"},
                    "project_name": {"type": "string"},
                },
            },
            "description": {
                "type": "string",
                "description": "Description for designing a container architecture",
            },
        },
        "required": ["action"],
    }
    requires_permission = True  # Docker operations require permission - gives host access

    def __init__(self, docker_socket: str = "/var/run/docker.sock"):
        """Initialize the Docker tool.

        Args:
            docker_socket: Path to the Docker socket
        """
        self.docker_socket = docker_socket
        self._docker_available = None

    async def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            # Check if Docker socket exists
            if not os.path.exists(self.docker_socket):
                self._docker_available = False
                return False

            # Try to run a simple Docker command
            proc = await asyncio.create_subprocess_exec(
                "docker", "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            self._docker_available = proc.returncode == 0
            return self._docker_available
        except Exception:
            self._docker_available = False
            return False

    async def execute(
        self,
        action: str,
        container: str | None = None,
        image: str | None = None,
        command: str | None = None,
        options: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> str:
        """Execute a Docker action.

        Args:
            action: The Docker action to perform
            container: Container name or ID
            image: Image name for run
            command: Command for exec
            options: Additional options
            description: Description for design action

        Returns:
            Result of the Docker action
        """
        if not await self._check_docker():
            return "Error: Docker is not available. Make sure Docker is installed and running."

        options = options or {}

        action_map = {
            "list_containers": self._list_containers,
            "list_images": self._list_images,
            "pull": self._pull_image,
            "run": self._run_container,
            "stop": self._stop_container,
            "remove": self._remove_container,
            "logs": self._get_logs,
            "exec": self._exec_in_container,
            "compose_up": self._compose_up,
            "compose_down": self._compose_down,
            "compose_ps": self._compose_ps,
            "inspect": self._inspect_container,
            "stats": self._container_stats,
            "design": self._design_architecture,
        }

        handler = action_map.get(action)
        if not handler:
            return f"Unknown Docker action: {action}"

        try:
            return await handler(
                container=container,
                image=image,
                command=command,
                options=options,
                description=description,
            )
        except Exception as e:
            return f"Error executing Docker action: {safe_error_message(e)}"

    async def _run_command(self, *args: str, timeout: float = 60.0) -> tuple[int, str, str]:
        """Run a Docker command.

        Args:
            *args: Command arguments
            timeout: Command timeout in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
            return (-1, "", "Command timed out")

    async def _list_containers(
        self,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """List all containers."""
        # Include stopped containers
        returncode, stdout, stderr = await self._run_command(
            "ps", "-a", "--format", "json"
        )

        if returncode != 0:
            return f"Error listing containers: {stderr}"

        try:
            # Docker ps --format json returns newline-separated JSON
            containers = []
            for line in stdout.strip().split("\n"):
                if line:
                    containers.append(json.loads(line))

            if not containers:
                return "No containers found."

            lines = ["Containers:"]
            for c in containers:
                status = c.get("State", "unknown")
                name = c.get("Names", c.get("ID", "unknown"))
                image = c.get("Image", "unknown")
                status_icon = "🟢" if status == "running" else "🔴"
                lines.append(f"  {status_icon} {name} - {image} ({status})")

            return "\n".join(lines)
        except json.JSONDecodeError:
            # Fallback to text parsing
            returncode, stdout, stderr = await self._run_command("ps", "-a")
            return stdout if returncode == 0 else f"Error: {stderr}"

    async def _list_images(
        self,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """List all images."""
        returncode, stdout, stderr = await self._run_command(
            "images", "--format", "json"
        )

        if returncode != 0:
            return f"Error listing images: {stderr}"

        try:
            images = []
            for line in stdout.strip().split("\n"):
                if line:
                    images.append(json.loads(line))

            if not images:
                return "No images found."

            lines = ["Images:"]
            for img in images:
                name = img.get("Repository", "unknown")
                tag = img.get("Tag", "latest")
                size = img.get("Size", "unknown")
                lines.append(f"  📦 {name}:{tag} ({size})")

            return "\n".join(lines)
        except json.JSONDecodeError:
            returncode, stdout, stderr = await self._run_command("images")
            return stdout if returncode == 0 else f"Error: {stderr}"

    async def _pull_image(
        self,
        image: str | None = None,
        **kwargs,
    ) -> str:
        """Pull an image from a registry."""
        if not image:
            return "Error: Image name required for pull action."

        returncode, stdout, stderr = await self._run_command(
            "pull", image, timeout=300.0
        )

        if returncode != 0:
            return f"Error pulling image: {stderr}"

        lines = [f"Pulled image: {image}"]
        lines.append(stdout.strip())

        return "\n".join(lines)

    async def _run_container(
        self,
        container: str | None = None,
        image: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Run a container."""
        if not image:
            return "Error: Image name required for run action."

        DANGEROUS_FLAGS = {
            "privileged", "pid", "userns", "uts", "ipc",
        }
        DANGEROUS_CAPS = {
            "SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE", "SYS_MODULE",
            "DAC_READ_SEARCH", "LINUX_IMMUTABLE", "NET_RAW", "NET_BROADCAST",
        }
        for flag in DANGEROUS_FLAGS:
            if options.get(flag) or options.get(f"{flag}-host"):
                return f"Error: Container flag '--{flag}=host' is blocked for security."

        cap_add = options.get("cap_add", [])
        if isinstance(cap_add, str):
            cap_add = [cap_add]
        for cap in cap_add:
            if cap.upper() in DANGEROUS_CAPS:
                return f"Error: '--cap-add {cap}' is blocked for security."

        options = options or {}
        args = ["run"]

        # Add options
        if options.get("detach", True):
            args.append("-d")

        container_name = container or options.get("name")
        if container_name:
            args.extend(["--name", container_name])

        for port in options.get("ports", []):
            args.extend(["-p", port])

        for volume in options.get("volumes", []):
            vol_parts = volume.split(":")
            if vol_parts:
                host_path = vol_parts[0].rstrip("/") or "/"
                resolved_host = Path(host_path).resolve()

                # Filesystem root is matched exactly, never by containment —
                # every absolute path is "within" /, which would reject all mounts.
                if Path(host_path) == Path(host_path).anchor or str(resolved_host) == resolved_host.anchor:
                    return f"Error: Volume mount '{volume}' rejected - mounting system paths is not allowed"

                # Check the path as written as well as the resolved one: a POSIX
                # spec resolves to "<drive>:\..." on Windows and would otherwise
                # slip past the blocklist entirely.
                for candidate in (host_path, resolved_host):
                    if is_protected_path(candidate, _BLOCKED_MOUNT_ROOTS):
                        return f"Error: Volume mount '{volume}' rejected - mounting system paths is not allowed"

                if any(path_is_within(candidate, sock)
                       for sock in _DOCKER_SOCKET_PATHS
                       for candidate in (host_path, resolved_host)):
                    return f"Error: Volume mount '{volume}' rejected - Docker socket mounting is not allowed"
                args.extend(["-v", volume])

        for env in options.get("env", []):
            args.extend(["-e", env])

        if options.get("network"):
            args.extend(["--network", options["network"]])

        if options.get("restart"):
            args.extend(["--restart", options["restart"]])

        args.append(image)

        returncode, stdout, stderr = await self._run_command(*args, timeout=120.0)

        if returncode != 0:
            return f"Error running container: {stderr}"

        container_id = stdout.strip()
        return f"Container started successfully.\nID: {container_id}"

    async def _stop_container(
        self,
        container: str | None = None,
        **kwargs,
    ) -> str:
        """Stop a container."""
        if not container:
            return "Error: Container name or ID required."

        returncode, stdout, stderr = await self._run_command("stop", container)

        if returncode != 0:
            return f"Error stopping container: {stderr}"

        return f"Container {container} stopped."

    async def _remove_container(
        self,
        container: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Remove a container."""
        if not container:
            return "Error: Container name or ID required."

        args = ["rm"]
        if options and options.get("force"):
            args.append("-f")
        args.append(container)

        returncode, stdout, stderr = await self._run_command(*args)

        if returncode != 0:
            return f"Error removing container: {stderr}"

        return f"Container {container} removed."

    async def _get_logs(
        self,
        container: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Get container logs."""
        if not container:
            return "Error: Container name or ID required."

        args = ["logs"]
        options = options or {}

        if options.get("tail"):
            args.extend(["--tail", str(options["tail"])])
        else:
            args.extend(["--tail", "100"])  # Default to last 100 lines

        if options.get("follow"):
            args.append("-f")

        args.append(container)

        returncode, stdout, stderr = await self._run_command(*args, timeout=10.0)

        if returncode != 0:
            return f"Error getting logs: {stderr}"

        # Combine stdout and stderr (Docker logs output)
        logs = (stdout + stderr).strip()
        if not logs:
            return "No logs available."

        return f"Logs for {container}:\n{logs}"

    async def _exec_in_container(
        self,
        container: str | None = None,
        command: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Execute a command in a container."""
        if not container:
            return "Error: Container name or ID required."
        if not command:
            return "Error: Command required for exec."

        # Validate container name/ID (alphanumeric, dash, underscore, colon, dot)
        import re
        if not re.match(r'^[a-zA-Z0-9_.:-]+$', container):
            return f"Error: Invalid container identifier: {container}"

        # Security: Block raw shell metacharacters that enable injection
        import shlex
        import logging
        _logger = logging.getLogger(__name__)

        shell_metachar_patterns = [
            (r'\$\(', 'command substitution $('),
            (r'\$\{', 'variable expansion ${'),
            (r'`', 'backtick command substitution'),
            (r'\n', 'newline injection'),
            (r'\r', 'carriage return injection'),
        ]
        for pattern, desc in shell_metachar_patterns:
            if re.search(pattern, command):
                _logger.warning("Blocked exec_in_container command with %s: %.200s", desc, command)
                return f"Error: Unsafe command rejected (contains {desc})"

        # Use shlex.quote() to safely escape the entire command so nothing
        # inside it can break out of the single-quoted string that sh -c
        # will receive.  This eliminates shell metacharacter injection
        # because shlex.quote wraps the value in single quotes and escapes
        # any embedded single quotes.
        quoted_command = shlex.quote(command)

        args = ["exec"]
        options = options or {}

        if options.get("interactive"):
            args.append("-i")
        args.append("-t")  # Always use TTY

        args.extend([container, "sh", "-c", quoted_command])

        returncode, stdout, stderr = await self._run_command(*args, timeout=30.0)

        if returncode != 0:
            return f"Error executing command: {stderr}"

        output = stdout.strip() or stderr.strip() or "Command executed successfully (no output)"
        return f"Executed in {container}:\n{output}"

    async def _compose_up(
        self,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Start a compose stack."""
        options = options or {}
        args = ["compose"]

        if options.get("compose_file"):
            args.extend(["-f", options["compose_file"]])

        if options.get("project_name"):
            args.extend(["-p", options["project_name"]])

        args.append("up")

        if options.get("detach", True):
            args.append("-d")

        returncode, stdout, stderr = await self._run_command(*args, timeout=180.0)

        if returncode != 0:
            return f"Error starting compose stack: {stderr}"

        return f"Compose stack started.\n{stdout.strip()}"

    async def _compose_down(
        self,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Stop a compose stack."""
        options = options or {}
        args = ["compose"]

        if options.get("compose_file"):
            args.extend(["-f", options["compose_file"]])

        if options.get("project_name"):
            args.extend(["-p", options["project_name"]])

        args.append("down")

        if options.get("volumes"):
            args.append("-v")

        returncode, stdout, stderr = await self._run_command(*args, timeout=60.0)

        if returncode != 0:
            return f"Error stopping compose stack: {stderr}"

        return f"Compose stack stopped.\n{stdout.strip()}"

    async def _compose_ps(
        self,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """List compose services."""
        options = options or {}
        args = ["compose"]

        if options.get("compose_file"):
            args.extend(["-f", options["compose_file"]])

        args.append("ps")

        returncode, stdout, stderr = await self._run_command(*args)

        if returncode != 0:
            return f"Error listing compose services: {stderr}"

        return f"Compose services:\n{stdout.strip()}"

    async def _inspect_container(
        self,
        container: str | None = None,
        **kwargs,
    ) -> str:
        """Inspect a container."""
        if not container:
            return "Error: Container name or ID required."

        returncode, stdout, stderr = await self._run_command("inspect", container)

        if returncode != 0:
            return f"Error inspecting container: {stderr}"

        try:
            data = json.loads(stdout)
            if isinstance(data, list) and data:
                data = data[0]

            # Extract key information
            name = data.get("Name", "unknown").lstrip("/")
            state = data.get("State", {})
            config = data.get("Config", {})

            lines = [
                f"Container: {name}",
                f"Status: {state.get('Status', 'unknown')}",
                f"Image: {config.get('Image', 'unknown')}",
                f"Created: {data.get('Created', 'unknown')}",
            ]

            # Ports
            ports = data.get("NetworkSettings", {}).get("Ports", {})
            if ports:
                lines.append("Ports:")
                for container_port, host_ports in ports.items():
                    if host_ports:
                        for hp in host_ports:
                            lines.append(f"  {hp.get('HostIp', '0.0.0.0')}:{hp.get('HostPort', '?')} -> {container_port}")

            # Environment
            env = config.get("Env", [])
            if env:
                lines.append("Environment:")
                for e in env[:10]:  # Limit to first 10
                    lines.append(f"  {e}")
                if len(env) > 10:
                    lines.append(f"  ... and {len(env) - 10} more")

            return "\n".join(lines)
        except json.JSONDecodeError:
            return stdout

    async def _container_stats(
        self,
        container: str | None = None,
        **kwargs,
    ) -> str:
        """Get container resource stats."""
        args = ["stats", "--no-stream"]

        if container:
            args.append(container)

        returncode, stdout, stderr = await self._run_command(*args)

        if returncode != 0:
            return f"Error getting stats: {stderr}"

        return f"Container stats:\n{stdout.strip()}"

    async def _design_architecture(
        self,
        description: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Design a Docker architecture based on description.

        This generates a docker-compose.yml based on the user's description.
        """
        if not description:
            return "Error: Description required for design action."

        output_dir = Path(options.get("output_dir", ".")) if options else Path(".")
        output_file = output_dir / "docker-compose.yml"


        compose_content = await self._generate_with_templates(description, options)

        if not compose_content:
            return f"Error: Could not generate docker-compose for: {description}"

        output_file.write_text(compose_content, encoding='utf-8')
        return f"Created docker-compose.yml at {output_file}\n\n{compose_content}"

    async def _generate_with_templates(
        self, description: str, options: dict[str, Any] | None
    ) -> str | None:
        """Generate compose file using pattern matching for common architectures."""
        description_lower = description.lower()
        services = []
        volumes = {}

        if any(word in description_lower for word in ["web", "app", "server", "frontend", "nginx", "apache"]):
            web_image = options.get("web_image", "nginx:alpine") if options else "nginx:alpine"
            web_port = options.get("web_port", "80:80") if options else "80:80"
            services.append(f"""  web:
    image: {web_image}
    ports:
      - "{web_port}"
    restart: unless-stopped""")

        if any(word in description_lower for word in ["database", "db", "postgres", "mysql", "mongo", "redis"]):
            if "postgres" in description_lower or "database" in description_lower:
                db_password = options.get("db_password", "changeme") if options else "changeme"
                services.append(f"""  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD={db_password}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5""")
                volumes["postgres_data"] = None
            elif "redis" in description_lower:
                services.append("""  redis:
    image: redis:alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5""")
            elif "mongo" in description_lower:
                services.append("""  mongo:
    image: mongo:6
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped""")
                volumes["mongo_data"] = None
            elif "mysql" in description_lower:
                db_password = options.get("db_password", "changeme") if options else "changeme"
                services.append(f"""  mysql:
    image: mysql:8
    environment:
      - MYSQL_ROOT_PASSWORD={db_password}
    volumes:
      - mysql_data:/var/lib/mysql
    restart: unless-stopped""")
                volumes["mysql_data"] = None

        if any(word in description_lower for word in ["api", "backend", "fastapi", "flask", "express"]):
            api_port = options.get("api_port", "8000:8000") if options else "8000:8000"
            services.append(f"""  api:
    build: ./api
    ports:
      - "{api_port}"
    environment:
      - DATABASE_URL=postgresql://db:5432/app
    depends_on:
      - db
    restart: unless-stopped""")

        if not services:
            return None

        lines = ["version: '3.8'", "services:"]
        lines.extend(services)

        if volumes:
            lines.append("\nvolumes:")
            for vol_name in volumes:
                lines.append(f"  {vol_name}:")

        return "\n".join(lines)


class DockerComposeTool(Tool):
    """Tool for working with docker-compose files."""

    name = "docker_compose"
    description = "Create and manage docker-compose.yml files"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "validate", "show_template"],
                "description": "The compose action to perform",
            },
            "content": {
                "type": "string",
                "description": "The compose file content (for create)",
            },
            "path": {
                "type": "string",
                "description": "Path to save/load the compose file",
            },
            "template": {
                "type": "string",
                "enum": ["webapp", "database", "microservices", "development"],
                "description": "Template type for show_template action",
            },
        },
        "required": ["action"],
    }
    requires_permission = False  # Docker Compose operations allowed for autonomous management

    # Common templates
    TEMPLATES = {
        "webapp": """version: '3.8'
services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./html:/usr/share/nginx/html
    restart: unless-stopped
""",
        "database": """version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
""",
        "microservices": """version: '3.8'
services:
  api:
    image: node:18-alpine
    working_dir: /app
    command: npm start
    volumes:
      - ./api:/app
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://postgres:password@db:5432/app
    depends_on:
      - db

  frontend:
    image: node:18-alpine
    working_dir: /app
    command: npm run dev
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: app
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
""",
        "development": """version: '3.8'
services:
  dev:
    image: python:3.11-slim
    working_dir: /workspace
    command: bash
    stdin_open: true
    tty: true
    volumes:
      - .:/workspace
      - pip_cache:/root/.cache/pip
    environment:
      - PYTHONUNBUFFERED=1

volumes:
  pip_cache:
""",
    }

    async def execute(
        self,
        action: str,
        content: str | None = None,
        path: str | None = None,
        template: str | None = None,
    ) -> str:
        """Execute a compose action."""
        if action == "create":
            if not content:
                return "Error: Content required for create action."
            if not path:
                path = "docker-compose.yml"

            try:
                import yaml
                # Validate YAML
                yaml.safe_load(content)

                file_path = Path(path)
                file_path.write_text(content, encoding='utf-8')
                return f"Created docker-compose.yml at {path}"
            except Exception as e:
                return f"Error creating compose file: {e}"

        elif action == "validate":
            if not path and not content:
                return "Error: Path or content required for validate action."

            try:
                import yaml
                yaml_content = content or Path(path).read_text(encoding='utf-8')
                data = yaml.safe_load(yaml_content)

                # Basic validation
                if "version" not in data:
                    return "Warning: No version specified in compose file."
                if "services" not in data:
                    return "Warning: No services defined in compose file."

                service_count = len(data.get("services", {}))
                return f"Valid compose file with {service_count} services."
            except Exception as e:
                return f"Invalid compose file: {e}"

        elif action == "show_template":
            template_name = template or "webapp"
            if template_name not in self.TEMPLATES:
                return f"Unknown template: {template_name}. Available: {list(self.TEMPLATES.keys())}"

            return f"Template: {template_name}\n```yaml\n{self.TEMPLATES[template_name]}\n```"

        return f"Unknown action: {action}"
