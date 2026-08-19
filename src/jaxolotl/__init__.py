"""Public API for the jaxolotl package."""

from pathlib import Path

from jaxolotl.environments.registration import make
from jaxolotl.hydra_utils.utils import register_custom_resolvers

register_custom_resolvers()

_root = Path(__file__).parent.parent.parent
DATA_DIR = _root / "data"
DEPENDENCIES_DIR = _root / "dependencies"
CACHE_DIR = _root / ".cache"
RUN_DIR = _root / "runs"
THREEJS_OUT_DIR = _root / "threejs_output"

__all__ = [
    "make",
    "DATA_DIR",
    "DEPENDENCIES_DIR",
    "CACHE_DIR",
    "RUN_DIR",
    "THREEJS_OUT_DIR",
]
