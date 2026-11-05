"""Compatibility import for the shared Qwen3 implementation in ``hpc_core``."""
from pathlib import Path
import sys

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hpc_core.model.qwen3_model import *  # noqa: F401,F403
