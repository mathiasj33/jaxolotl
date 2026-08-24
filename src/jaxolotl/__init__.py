"""Public API for the jaxolotl package."""

from jaxolotl.environments.registration import make
from jaxolotl.hydra_utils.utils import register_custom_resolvers
from jaxolotl.paths import (
    CACHE_DIR,
    DATA_DIR,
    DEPENDENCIES_DIR,
    RUN_DIR,
    THREEJS_OUT_DIR,
)

register_custom_resolvers()

__all__ = [
    "make",
    "DATA_DIR",
    "DEPENDENCIES_DIR",
    "CACHE_DIR",
    "RUN_DIR",
    "THREEJS_OUT_DIR",
]
