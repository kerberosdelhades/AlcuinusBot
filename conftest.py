"""Pytest root config — make the `alcuinus` package importable.

The project uses a `src/` layout without a build backend installed in the
venv, so tests need `src/` on ``sys.path`` to ``import alcuinus``. This
conftest does that once per session so `pytest tests/` works out of the box
(no PYTHONPATH gymnastics). See also `tests/test_db.py`, which inserts the
same paths defensively for standalone runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
