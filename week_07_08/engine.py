"""Compatibility import for the shared serving engine."""
from pathlib import Path
import sys

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hpc_core.engine import *  # noqa: F401,F403
