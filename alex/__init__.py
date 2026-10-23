"""
Alex shared package: configuration, identity, errors, and small utilities
shared by the agent, app, pipelines, and SDK.

This is the only place that knows how the rest of the system stitches together.
The agent imports `alex.identity`, the app imports `alex.warehouse`, the
pipelines import `alex.retry`, and so on.
"""

__version__ = "2.0.0"
