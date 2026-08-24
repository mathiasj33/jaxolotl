from pathlib import Path
from typing import Literal

from jaxolotl.paths import DATA_DIR

type ResetSplit = Literal["train", "test"]
type ResetSource = Literal["native", "train", "test"]

RESET_SPLITS = ("train", "test")
RESET_SOURCES = ("native", *RESET_SPLITS)


def precomputed_reset_path(env_name: str, split: ResetSplit) -> Path:
    """Return the canonical path for an environment's reset-state pool."""
    if split not in RESET_SPLITS:
        raise ValueError(
            f"Unknown reset split {split!r}; expected one of {RESET_SPLITS}."
        )

    return DATA_DIR / env_name / f"sampled_resets_{split}.eqx"
