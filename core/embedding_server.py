"""
Embedding IPC Server - Single shared embedding model for parallel agents.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING SERVER (Single Process)                │
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐                       │
│  │ ConceptEmbedder │    │  Unix Socket     │                       │
│  │ (Jina v5)       │◄───│  /tmp/jarvis_emb │                       │
│  │ ~1GB RAM        │    │                  │                       │
│  └─────────────────┘    └────────┬─────────┘                       │
│                                  │                                  │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
    ┌──────────┐            ┌──────────┐            ┌──────────┐
    │ Agent 1  │            │ Agent 2  │            │ Agent N  │
    │ (Worker) │            │ (Worker) │            │ (Worker) │
    └──────────┘            └──────────┘            └──────────┘

Benefits:
- Single model instance (saves ~1GB RAM per worker)
- No GPU memory duplication
- Fast IPC via Unix socket
- Graceful degradation (falls back to local model)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
from core.helpers import safe_error_message
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Socket path for IPC
SOCKET_PATH = Path("/tmp/jarvis_embedding.sock")
if sys.platform == "win32":
    SOCKET_PATH = Path(os.environ.get("TEMP", ".")) / "jarvis_embedding.sock"

# Lock file for server coordination
LOCK_PATH = Path("/tmp/jarvis_embedding.lock")
if sys.platform == "win32":
    LOCK_PATH = Path(os.environ.get("TEMP", ".")) / "jarvis_embedding.lock"


class EmbeddingServer:
    """
    Singleton embedding server that loads model once and handles requests.
    
    Usage:
        # Start server (usually done automatically or as separate process)
        server = EmbeddingServer()
        server.start()
        
        # Or run as standalone process:
        # python -m core.embedding_server
    """
    
    _instance: Optional["EmbeddingServer"] = None
    _lock = threading.Lock()
    
    def __init__(self, socket_path: Path = None):
        self.socket_path = socket_path or SOCKET_PATH
        self.embedder = None
        self.running = False
        self.server = None
        self._connections = set()
        
    @classmethod
    def get_instance(cls) -> "EmbeddingServer":
        """Get singleton server instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _load_embedder(self):
        """Load embedding model (lazy)."""
        if self.embedder is None:
            from core.concept_model import ConceptEmbedder
            logger.info("Loading embedding model for server...")
            self.embedder = ConceptEmbedder()
            logger.info("Embedding model loaded")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle individual client connection."""
        addr = writer.get_extra_info('peername') or 'unknown'
        logger.debug(f"Client connected: {addr}")
        self._connections.add(writer)
        
        try:
            while self.running:
                # Read request length (4 bytes)
                length_data = await reader.read(4)
                if not length_data:
                    break
                    
                length = int.from_bytes(length_data, 'big')
                
                # Read request data
                data = await reader.read(length)
                if not data:
                    break
                
                try:
                    request = json.loads(data.decode('utf-8'))
                    response = await self._process_request(request)
                except Exception as e:
                    response = {"error": safe_error_message(e)}
                
                # Send response
                response_data = json.dumps(response).encode('utf-8')
                writer.write(len(response_data).to_bytes(4, 'big'))
                writer.write(response_data)
                await writer.drain()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Client error: {e}")
        finally:
            self._connections.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            logger.debug(f"Client disconnected: {addr}")
    
    async def _process_request(self, request: dict) -> dict:
        """Process a single request."""
        self._load_embedder()
        
        cmd = request.get("cmd")
        
        if cmd == "encode":
            text = request.get("text", "")
            if not text:
                return {"error": "Missing text"}
            
            embedding = self.embedder.encode(text)
            return {
                "embedding": embedding.tolist(),
                "dim": len(embedding)
            }
        
        elif cmd == "encode_batch":
            texts = request.get("texts", [])
            if not texts:
                return {"error": "Missing texts"}
            
            embeddings = [self.embedder.encode(t).tolist() for t in texts]
            return {
                "embeddings": embeddings,
                "count": len(embeddings)
            }
        
        elif cmd == "ping":
            return {"pong": True, "status": "running"}
        
        elif cmd == "stats":
            return {
                "connections": len(self._connections),
                "model_loaded": self.embedder is not None
            }
        
        else:
            return {"error": f"Unknown command: {cmd}"}
    
    async def start_async(self):
        """Start server asynchronously."""
        self._load_embedder()
        
        # Remove old socket if exists
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        self.running = True
        
        # Create Unix socket server
        if sys.platform == "win32":
            # Windows: use TCP
            self.server = await asyncio.start_server(
                self._handle_client,
                '127.0.0.1',
                9123  # Fixed port for Windows
            )
        else:
            # Unix: use Unix socket (faster)
            self.server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.socket_path)
            )
            # Make socket accessible
            os.chmod(str(self.socket_path), 0o777)
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f"Embedding server listening on {addr}")
        
        # Write lock file
        LOCK_PATH.write_text(str(os.getpid()))
        
        async with self.server:
            await self.server.serve_forever()
    
    def start(self, blocking: bool = True):
        """
        Start the embedding server.
        
        Args:
            blocking: If True, block until server stops. If False, run in background thread.
        """
        if blocking:
            asyncio.run(self.start_async())
        else:
            thread = threading.Thread(target=self._run_in_thread, daemon=True)
            thread.start()
    
    def _run_in_thread(self):
        """Run server in background thread."""
        asyncio.run(self.start_async())
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server:
            self.server.close()
        if self.socket_path.exists():
            self.socket_path.unlink()
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
        logger.info("Embedding server stopped")


def is_server_running() -> bool:
    """Check if embedding server is running."""
    if not LOCK_PATH.exists():
        return False
    
    try:
        pid = int(LOCK_PATH.read_text(encoding='utf-8').strip())
        # Check if process exists
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


async def start_server_if_needed():
    """Start embedding server if not already running."""
    if is_server_running():
        logger.debug("Embedding server already running")
        return
    
    logger.info("Starting embedding server...")
    server = EmbeddingServer.get_instance()
    
    # Run in background thread
    server.start(blocking=False)
    
    # Wait for server to be ready
    for _ in range(50):  # 5 seconds timeout
        if is_server_running():
            logger.info("Embedding server started")
            return
        await asyncio.sleep(0.1)
    
    raise RuntimeError("Failed to start embedding server")


def main():
    """Run embedding server as standalone process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jarvis Embedding Server")
    parser.add_argument("--socket", type=str, default=None, help="Socket path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    socket_path = Path(args.socket) if args.socket else None
    server = EmbeddingServer(socket_path)
    
    # Handle shutdown signals
    def shutdown(sig, frame):
        logger.info("Shutting down...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    logger.info("Starting embedding server...")
    server.start(blocking=True)


if __name__ == "__main__":
    main()