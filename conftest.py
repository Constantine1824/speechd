"""Pytest configuration: ensure the repo root is importable.

Tests import the package as `speechsep.*`; putting the repo root on sys.path
lets `import speechsep` resolve without requiring an editable install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
