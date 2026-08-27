"""Vendored simulator assets and package-independent path resolution."""

from importlib import resources
from pathlib import Path

PANDA_SCENE = "franka_emika_panda/mjx_scene.xml"


def asset_path(relative_path: str) -> Path:
    """Return an absolute path to an asset bundled with :mod:`jaxolotl`."""
    path = Path(str(resources.files(__package__))) / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Vendored asset {relative_path!r} is missing from {path.parent}. "
            "Reinstall jaxolotl so package data is restored."
        )
    return path
