"""Tests for the SDK package.

The SDK lives in `sdk/alex_sdk/`. We add the `sdk/` directory to
`sys.path` so `import alex_sdk` works without an install step.
"""
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[2] / "sdk"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))
