"""Pytest configuration: ensure the repo root is importable.

The project uses a flat module layout (schemas.py, output.py, ...), so tests
need the repo root on sys.path to `import schemas` etc.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
