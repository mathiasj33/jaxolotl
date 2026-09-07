"""Utilities for model artifact management."""

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from jaxolotl import eqx_utils

MODEL_DIRECTORY = "models"
MODEL_FILE_TEMPLATE = "model_seed{seed}.eqx"
MODEL_FILE_RE = re.compile(r"^model_seed(\d+)\.eqx$")

_CONFIG_HASH_EXCLUDED_KEYS = frozenset({"start_seed", "num_seeds"})


def model_filename(seed: int) -> str:
    """Return the final-model filename for ``seed``."""
    return MODEL_FILE_TEMPLATE.format(seed=seed)


def model_path(run_dir: Path | str, seed: int) -> Path:
    """Return the final-model path for ``seed`` within a run directory."""
    return Path(run_dir) / MODEL_DIRECTORY / model_filename(seed)


def discover_seed_models(run_dir: Path | str) -> dict[int, Path]:
    """Return a mapping of completed seeds to their final-model paths in a run directory."""
    model_dir = Path(run_dir) / MODEL_DIRECTORY
    models = {}
    if not model_dir.is_dir():
        return models

    for path in model_dir.iterdir():
        match = MODEL_FILE_RE.fullmatch(path.name)
        if match is not None:
            models[int(match.group(1))] = path
    return models


def config_hash(cfg: DictConfig) -> str:
    """Hash experiment configuration while ignoring the selected seed batch."""
    container = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(container, dict):
        container = {
            key: value
            for key, value in container.items()
            if key not in _CONFIG_HASH_EXCLUDED_KEYS
        }
    payload = json.dumps(container, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def verify_run_dir(
    run_dir: Path | str,
    seeds: Iterable[int],
    *,
    expected_config_hash: str,
) -> None:
    """Raise an error if there are conflicts of completed seeds or mixed configurations in one run."""
    existing = discover_seed_models(run_dir)
    collisions = sorted(set(seeds) & existing.keys())
    if collisions:
        paths = ", ".join(str(existing[seed]) for seed in collisions)
        raise FileExistsError(
            f"Refusing to retrain completed seed(s) {collisions}; final model files "
            f"already exist: {paths}"
        )

    for seed, path in sorted(existing.items()):
        metadata = eqx_utils.load_metadata(path)
        found_hash = metadata.get("config_hash")
        if found_hash is not None and found_hash != expected_config_hash:
            raise ValueError(
                f"Seed {seed} in {path} has config hash {found_hash}, but this batch "
                f"has {expected_config_hash}. Seeds in one run directory must use the "
                "same experiment configuration."
            )


def verify_model_metadata(seed_to_path: dict[int, Path]) -> None:
    """Validate metadata shared by per-seed model files before evaluation."""
    hashes: dict[str, list[int]] = {}
    for seed, path in sorted(seed_to_path.items()):
        metadata = eqx_utils.load_metadata(path)
        recorded_seed = metadata.get("seed")
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise ValueError(
                f"Model filename {path.name} denotes seed {seed}, but its metadata "
                f"records seed {recorded_seed}."
            )
        found_hash = metadata.get("config_hash")
        if found_hash is not None:
            hashes.setdefault(found_hash, []).append(seed)

    if len(hashes) > 1:
        groups = "; ".join(
            f"{config_hash}: seeds {seeds}"
            for config_hash, seeds in sorted(hashes.items())
        )
        raise ValueError(
            f"Final models were trained with different configurations ({groups})."
        )
