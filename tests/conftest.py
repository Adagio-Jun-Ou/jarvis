"""Pytest configuration for Jarvis tests.

Sets environment variables and fixtures common to all tests.
"""
import os
import tempfile
import pytest

# Force local embedder for tests (no IPC server needed)
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

# Disable parallel agent workers for tests
os.environ["JARVIS_NO_PARALLEL"] = "1"

# Reduce log noise during tests
os.environ["JARVIS_LOG_LEVEL"] = "WARNING"

# Keep adaptive-loop checkpoints out of the real ~/.jarvis (tests that
# run the adaptive loop would otherwise pollute the user's home state).
os.environ.setdefault(
    "JARVIS_ADAPTIVE_CHECKPOINT_DIR",
    os.path.join(tempfile.gettempdir(), "jarvis_test_checkpoints"),
)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure environment for all tests."""
    # Already set via os.environ at module level
    yield
    
    # Cleanup handled by pytest automatically