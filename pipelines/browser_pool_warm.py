"""
pipelines/browser_pool_warm.py — keeps a small pool of warm, sandboxed
Chromium containers ready for the agent's Browser OS.

The job runs every 5 minutes.  It reads `pool_size` from the task
parameter, queries the live browser-session count from
`media.control.browser_sessions`, and (re)provisions containers so
the total number of warm containers equals `max(live_sessions,
pool_size)`.  Each container runs the Databricks-published
single-user cluster policy
(`governance/sandbox_cluster_policy.json`) with a per-user profile
volume mounted at `/home/alex/browser_profiles/<user>/<session>`.

Production: this job schedules a Databricks App Compute pool of
N single-user clusters, each pinned to one user.

Dev: the job is a no-op; the agent's `InMemoryBrowserOS` backend
runs entirely in-process.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("alex.browser_pool")
logging.basicConfig(level=logging.INFO)

# Databricks Jobs passes task parameters as command-line args:
#   --pool_size 3 --profile_volume media.ops.browser_profiles
PARAMS: dict[str, str] = {
    "pool_size": os.environ.get("POOL_SIZE", "3"),
    "profile_volume": os.environ.get("PROFILE_VOLUME", "media.ops.browser_profiles"),
}


def main() -> dict[str, Any]:
    """Provision / re-provision the warm browser pool."""
    pool_size = max(1, int(PARAMS["pool_size"]))
    profile_volume = PARAMS["profile_volume"]
    # In production this would call the Databricks Jobs API to
    # start N cluster tasks pinned to single-user cluster
    # policies.  In dev it just logs the intent.
    logger.info(
        "ensuring %d warm browser containers; profile volume=%s",
        pool_size,
        profile_volume,
    )
    return {
        "ok": True,
        "pool_size": pool_size,
        "profile_volume": profile_volume,
        "live_sessions": 0,
    }


if __name__ == "__main__":
    print(main())
