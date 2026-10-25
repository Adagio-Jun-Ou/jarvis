"""
Embedding IPC Client - Connect to shared embedding server.

This client connects to the embedding server via Unix socket (or TCP on Windows).
If the server is not available, it falls back to loading the model locally.
"""

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Same socket path as server
SOCKET_PATH = Path("/tmp/jarvis_embedding.sock")
if sys.platform == "win32":
    SOCKET_PATH = None  # Windows uses TCP


class EmbeddingClient:
    """
    Client for embedding server with automatic fallback.
    
    Usage:
        client = EmbeddingClient()
        embedding = client.encode("some text")  # Uses server if available, else local model
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, prefer_server: bool = True, timeout: float = 5.0):
        """Singleton pattern for thread-safe initialization."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, prefer_server: bool = True, timeout: float = 5.0):
        """
        Initialize embedding client.
        
        Args:
            prefer_server: If True, try server first, fall back to local model
            timeout: Connection timeout in seconds
        """
        if self._initialized:
            return
            
        self.prefer_server = prefer_server
        self.timeout = timeout
        self._local_embedder = None
        self._server_available = None
        self._last_check = 0
        self._check_interval = 30  # Re-check server every 30 seconds
        self._init_lock = threading.Lock()
        self._initialized = True
        
    def _check_server(self) -> bool:
        """Check if server is available (cached)."""
        now = time.time()
        if self._server_available is None or (now - self._last_check) > self._check_interval:
            self._server_available = self._ping_server()
            self._last_check = now
        return self._server_available
    
    def _ping_server(self) -> bool:
        """Ping the server to check if it's running."""
        try:
            # Use synchronous socket check instead of async
            import socket
            if sys.platform == "win32":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(('127.0.0.1', 9123))
                sock.close()
                return result == 0
            else:
                if not SOCKET_PATH or not SOCKET_PATH.exists():
                    return False
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                try:
                    sock.connect(str(SOCKET_PATH))
                    sock.close()
                    return True
                except Exception:
                    return False
        except Exception as e:
            logger.debug(f"Server ping failed: {e}")
            return False
    
    async def _request_async(self, cmd: str, params: dict) -> dict:
        """Send request to server asynchronously."""
        if sys.platform == "win32":
            # Windows: TCP
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', 9123),
                timeout=self.timeout
            )
        else:
            # Unix: Unix socket
            if not SOCKET_PATH or not SOCKET_PATH.exists():
                raise ConnectionError("Socket not found")
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(SOCKET_PATH)),
                timeout=self.timeout
            )
        
        try:
            # Prepare request
            request = {"cmd": cmd, **params}
            data = json.dumps(request).encode('utf-8')
            
            # Send: 4-byte length + data
            writer.write(len(data).to_bytes(4, 'big'))
            writer.write(data)
            await writer.drain()
            
            # Read response: 4-byte length + data
            length_data = await asyncio.wait_for(reader.read(4), timeout=self.timeout)
            if not length_data:
                raise ConnectionError("No response")
            
            length = int.from_bytes(length_data, 'big')
            response_data = await asyncio.wait_for(reader.read(length), timeout=self.timeout)
            
            return json.loads(response_data.decode('utf-8'))
            
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Uses server if available, falls back to local model.
        
        Args:
            text: Text to encode
            
        Returns:
            768-dimensional numpy array
        """
        if self.prefer_server and self._check_server():
            try:
                result = asyncio.run(self._request_async("encode", {"text": text}))
                if "error" in result:
                    raise RuntimeError(result["error"])
                return np.array(result["embedding"], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Server request failed, falling back to local: {e}")
                self._server_available = False
        
        # Fallback to local model
        return self._encode_local(text)
    
    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of 768-dimensional numpy arrays
        """
        if self.prefer_server and self._check_server():
            try:
                result = asyncio.run(self._request_async("encode_batch", {"texts": texts}))
                if "error" in result:
                    raise RuntimeError(result["error"])
                return [np.array(e, dtype=np.float32) for e in result["embeddings"]]
            except Exception as e:
                logger.warning(f"Server batch request failed, falling back to local: {e}")
                self._server_available = False
        
        # Fallback to local
        return [self._encode_local(t) for t in texts]
    
    def _encode_local(self, text: str) -> np.ndarray:
        """Encode using local model (fallback)."""
        with self._init_lock:
            if self._local_embedder is None:
                from core.concept_model import ConceptEmbedder
                logger.info("Loading local embedding model (fallback)...")
                self._local_embedder = ConceptEmbedder()
        
        return self._local_embedder.encode(text)

    def health(self) -> dict:
        """Report embedding availability across both the IPC and local paths.

        When the shared server is unreachable this client transparently falls
        back to a local model, so "server down" is not itself unhealthy — but a
        local model that failed to load is, because encode() then yields zero
        vectors. Delegates to the local embedder when one has been created.
        """
        server_up = bool(self._server_available)
        if self._local_embedder is not None and hasattr(self._local_embedder, "health"):
            info = dict(self._local_embedder.health())
        else:
            info = {
                "healthy": True,
                "status": "ready" if server_up else "not_loaded",
                "message": (
                    "Served by shared embedding server"
                    if server_up
                    else "No local model loaded yet and server not confirmed"
                ),
            }
        info["server_available"] = server_up
        return info


# Singleton client
_client: Optional[EmbeddingClient] = None
_client_lock = threading.Lock()


def get_embedding_client(prefer_server: bool = True) -> EmbeddingClient:
    """Get singleton embedding client (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = EmbeddingClient(prefer_server=prefer_server)
    return _client