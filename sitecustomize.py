"""Project-local Python startup tweaks.

This file is auto-imported by Python's `site` module (if it is importable from sys.path).
We use it to ensure the compiled `cpoker*.pyd` inside ./env is always preferred over any
stale copy that might exist in the project root.
"""

from __future__ import annotations

import os
import sys


def _prepend_env_dir() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(root, "env")
    if os.path.isdir(env_dir) and env_dir not in sys.path:
        # Put ./env first so `import cpoker` always resolves to the freshly built extension.
        sys.path.insert(0, env_dir)


_prepend_env_dir()
