"""Utilities for replaying WarehouseEnv trajectories with Three.js."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jax

from jaxolotl import THREEJS_OUT_DIR
from jaxolotl.environments.warehouse_env.warehouse_env import EnvState, WarehouseParams

_ASSET_FILES = [
    "styles.css",
    "config.js",
    "utils.js",
    "scene.js",
    "warehouse.js",
    "playback.js",
    "ui.js",
    "main.js",
]
_TEMPLATE_FILE = "index.html"


def replay_trajectories(
    trajs: EnvState,
    lengths: jax.Array,
    env_params: WarehouseParams,
    *,
    frames_per_step: int,
    pause_between_episodes: float,
) -> Path:
    """Write Three.js replay assets and open the visualization in a browser."""
    assets_dir = Path(__file__).resolve().parent
    destination = THREEJS_OUT_DIR
    destination.mkdir(parents=True, exist_ok=True)

    _copy_assets(assets_dir, destination)

    data = _build_replay_data(
        trajs,
        lengths,
        env_params,
        frames_per_step=frames_per_step,
        pause_between_episodes=pause_between_episodes,
    )

    template = (assets_dir / _TEMPLATE_FILE).read_text()
    rendered = template.replace("__WAREHOUSE_DATA__", json.dumps(data))
    output_path = destination / "index.html"
    output_path.write_text(rendered)

    return destination


def _copy_assets(source_dir: Path, destination: Path) -> None:
    for filename in _ASSET_FILES:
        shutil.copy(source_dir / filename, destination / filename)


def _build_replay_data(
    trajs: EnvState,
    lengths: jax.Array,
    env_params: WarehouseParams,
    *,
    frames_per_step: int,
    pause_between_episodes: float,
) -> dict:
    trajs_np = jax.tree.map(jax.device_get, trajs)
    lengths_np = jax.device_get(lengths)

    trajectories = []
    for episode_index in range(int(lengths_np.shape[0])):
        length = int(lengths_np[episode_index])
        states = []
        for step in range(length + 1):
            states.append(
                {
                    "position": trajs_np.position[episode_index, step].tolist(),
                    "angle": float(trajs_np.angle[episode_index, step]),
                    "vase_positions": trajs_np.vase_positions[
                        episode_index, step
                    ].tolist(),
                    "vase_available": trajs_np.vase_available[
                        episode_index, step
                    ].tolist(),
                    "crate_positions": trajs_np.crate_positions[
                        episode_index, step
                    ].tolist(),
                    "crate_available": trajs_np.crate_available[
                        episode_index, step
                    ].tolist(),
                    "carrying_vase_idx": int(
                        trajs_np.carrying_vase_idx[episode_index, step]
                    ),
                    "carrying_crate_idx": int(
                        trajs_np.carrying_crate_idx[episode_index, step]
                    ),
                }
            )
        trajectories.append({"states": states})

    return {
        "env": {
            "world_size": env_params.world_size,
            "pickup_radius": env_params.pickup_radius,
            "num_vases": env_params.num_vases,
            "num_crates": env_params.num_crates,
            "region_a": list(env_params.region_a),
            "region_b": list(env_params.region_b),
            "door_region": list(env_params.door_region),
        },
        "replay": {
            "frames_per_step": frames_per_step,
            "pause_between_episodes": pause_between_episodes,
        },
        "trajectories": trajectories,
    }
