"""The model MLflow logs — the "models from code" entry point.

`mlflow.pyfunc.log_model(python_model="serving_entry.py")` executes this
file and serves whatever `set_model` was called with. That is why this
file is three lines of work and no logic: everything it could do wrong
would be invisible until deployment, so there is nothing here to get
wrong.

It used to point at `agent/agent.py`, which had been reduced to a
re-export with no `predict` on it, so the logged model had no callable
interface at all. Nothing caught it because nothing exercises the
deployment path outside a real workspace.
"""

from __future__ import annotations

import mlflow

from agent.serving import build_agent

mlflow.models.set_model(build_agent())
