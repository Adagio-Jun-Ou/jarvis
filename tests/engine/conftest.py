"""
Tests for the `alex` shared package: identity, events, warehouse, retry,
testing, config.

Run with: pytest -q
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _hermetic_embeddings():
    """Tests use the offline embedding fallback.

    The dev embedder calls a local model over HTTP. Letting the suite
    reach it makes tests depend on a running service, non-deterministic,
    and an order of magnitude slower (15s to 170s when this was missed).
    Tests that specifically exercise the real embedder opt back in.
    """
    import os

    previous = os.environ.get("ALEX_EMBED_HASH_ONLY")
    os.environ["ALEX_EMBED_HASH_ONLY"] = "1"
    import agent.vector_memory as vm

    vm._FORCE_HASH = True
    yield
    if previous is None:
        os.environ.pop("ALEX_EMBED_HASH_ONLY", None)
    else:
        os.environ["ALEX_EMBED_HASH_ONLY"] = previous


@pytest.fixture(autouse=True)
def _reset_alex_singletons():
    """Reset process-wide singletons so each test starts clean."""
    from alex import config
    config._settings = None
    from alex import testing
    testing._fakes = None
    _clear_caches()
    yield
    config._settings = None
    testing._fakes = None
    _clear_caches()


@pytest.fixture(autouse=True)
def _unregister_adapted_tools():
    """App-lifespan tests boot ``register_jarvis_tools()``; without this
    cleanup the adapted (deferred) tools leak into later tests, which
    then see a registry/doc count they did not set up."""
    yield
    try:
        from agent.jarvis_tools import unregister_jarvis_tools

        unregister_jarvis_tools()
    except Exception:
        pass


def _clear_caches() -> None:
    """Drop the per-process VM/ACL caches so cached state never leaks
    across tests (the fakes singleton they were built against is reset
    by the fixture above)."""
    try:
        from agent import vm as _vm
        _vm._acl_cache_clear()
    except Exception:
        pass
    try:
        import sys
        app_mod = sys.modules.get("app.app")
        if app_mod is not None:
            app_mod._vm_cache_clear()
    except Exception:
        pass
