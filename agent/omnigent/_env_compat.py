"""Environment-variable shim for the Omnigent vendored copy.

The public upstream ``omnigent`` package mirrors a legacy
``OMNIAGENTS_*`` / ``OMNIGENTS_*`` env-var prefix onto the current
``OMNIGENT_*`` prefix for backward compatibility with older deploys.
We do not have any legacy env-var callers in Alex 2.0, so the mirror
is a no-op; we keep the call site at ``agent/omnigent/__init__.py``
and a stub function here so the upstream ``import
omnigent._env_compat`` still works and the package boot order is
preserved.
"""

from __future__ import annotations

import os

_mirrored = False


def mirror_legacy_env() -> None:
    """No-op mirror; kept for upstream import-shape compatibility.

    The Alex 2.0 deployment has no ``OMNIAGENTS_*`` or
    ``OMNIGENTS_*`` env-var callers.  This function is a placeholder
    that exists so the upstream import graph
    (``omnigent.__init__`` → ``omnigent._env_compat`` → mirror)
    can run unchanged against the vendored copy.
    """
    global _mirrored
    if _mirrored:
        return
    for legacy_prefix in ("OMNIAGENTS_", "OMNIGENTS_"):
        for name, value in list(os.environ.items()):
            if name.startswith(legacy_prefix):
                new_name = "OMNIGENT_" + name[len(legacy_prefix) :]
                os.environ.setdefault(new_name, value)
    _mirrored = True
