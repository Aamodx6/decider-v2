"""Test path setup: make ``src/decider`` and the ``train`` package importable.

Kept minimal (no plugins); each test module chooses its own markers.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for rel in ("src", "."):
    p = str(PROJECT_ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)
