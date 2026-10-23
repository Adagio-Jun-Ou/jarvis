"""n8n lifecycle manager for Jarvis.

Automatically installs and manages n8n as a background service.
"""
import asyncio
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger("jarvis.n8n")

N8N_PORT = 5678
N8N_DATA_DIR = Path.home() / ".jarvis" / "n8n_data"
N8N_PID_FILE = Path.home() / ".jarvis" / "n8n.pid"


class N8NManager:
    """Manages n8n installation and lifecycle."""

    def __init__(self, port: int = N8N_PORT, data_dir: Path | None = None):
        self.port = port
        self.data_dir = data_dir or N8N_DATA_DIR
        self._process: asyncio.subprocess.Process | None = None
        self._installed: bool | None = None
        self._log_file = None

    def is_installed(self) -> bool:
        """Check if n8n is installed globally via npm."""
        if self._installed is not None:
            return self._installed

        try:
            result = subprocess.run(
                ["npm", "list", "-g", "n8n", "--depth=0"],
                capture_output=True,
                text=True,
            )
            self._installed = "n8n@" in result.stdout
            return self._installed
        except Exception as e:
            logger.debug(f"n8n install check failed: {e}")
            self._installed = False
            return False

    async def install(self) -> bool:
        """Install n8n globally via npm."""
        if self.is_installed():
            logger.info("n8n already installed")
            return True

        logger.info("Installing n8n globally...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "npm", "install", "-g", "n8n",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._installed = True
                logger.info("n8n installed successfully")
                return True
            else:
                logger.error(f"n8n installation failed: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Failed to install n8n: {e}")
            return False

    def is_running(self) -> bool:
        """Check if n8n server is already running."""
        if N8N_PID_FILE.exists():
            try:
                pid = int(N8N_PID_FILE.read_text(encoding='utf-8').strip())
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, ValueError, PermissionError):
                N8N_PID_FILE.unlink(missing_ok=True)
        return False

    async def start(self) -> bool:
        """Start n8n server in background."""
        if self.is_running():
            logger.info(f"n8n already running on port {self.port}")
            return True

        if not self.is_installed():
            if not await self.install():
                return False

        self.data_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["N8N_PORT"] = str(self.port)
        env["N8N_DATA_DIR"] = str(self.data_dir)
        env["N8N_BASIC_AUTH_ACTIVE"] = "false"
        env["N8N_HOST"] = "localhost"
        env["WEBHOOK_URL"] = f"http://localhost:{self.port}/"

        try:
            log_path = self.data_dir / "n8n.log"
            self._log_file = await asyncio.to_thread(open, log_path, "a")

            self._process = await asyncio.create_subprocess_exec(
                "n8n", "start",
                env=env,
                stdout=self._log_file,
                stderr=self._log_file,
                start_new_session=True,
            )

            await asyncio.to_thread(N8N_PID_FILE.write_text, str(self._process.pid))
            logger.info(f"n8n started on port {self.port} (PID: {self._process.pid})")

            for i in range(30):
                await asyncio.sleep(1)
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"http://localhost:{self.port}/healthz", timeout=2)
                        if resp.status_code == 200:
                            logger.info(f"n8n is ready after {i+1}s")
                            return True
                except Exception as e:
                    logger.warning(f"n8n health check attempt {i+1} failed: {e}")

            logger.warning("n8n started but health check timed out")
            return True

        except Exception as e:
            logger.error(f"Failed to start n8n: {e}")
            return False

    async def stop(self) -> None:
        """Stop n8n server."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except Exception as e:
                logger.debug(f"n8n terminate failed: {e}")
                try:
                    self._process.kill()
                    await self._process.wait()
                except Exception as e2:
                    logger.debug(f"n8n kill failed: {e2}")
            finally:
                self._process = None

        if self._log_file:
            try:
                self._log_file.close()
            except Exception as e:
                logger.warning(f"Failed to close n8n log file: {e}")
            finally:
                self._log_file = None

        if N8N_PID_FILE.exists():
            try:
                pid_text = await asyncio.to_thread(N8N_PID_FILE.read_text)
                pid = int(pid_text.strip())
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to kill n8n process: {e}")
            finally:
                N8N_PID_FILE.unlink(missing_ok=True)

        logger.info("n8n stopped")

    async def get_status(self) -> dict:
        """Get n8n server status."""
        status = {
            "installed": self.is_installed(),
            "running": self.is_running(),
            "port": self.port,
            "data_dir": str(self.data_dir),
        }

        if status["running"]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"http://localhost:{self.port}/healthz",
                        timeout=2
                    )
                    status["healthy"] = resp.status_code == 200
            except Exception:
                status["healthy"] = False

        return status


_n8n_manager: N8NManager | None = None
_n8n_manager_lock = threading.Lock()


def get_n8n_manager(port: int = N8N_PORT) -> N8NManager:
    """Get or create n8n manager singleton."""
    global _n8n_manager
    if _n8n_manager is None:
        with _n8n_manager_lock:
            if _n8n_manager is None:
                _n8n_manager = N8NManager(port=port)
            return _n8n_manager
    return _n8n_manager