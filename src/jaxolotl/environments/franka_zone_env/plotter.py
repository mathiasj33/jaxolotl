"""Static 3D end-effector trajectories with sparse Panda skeleton ghosts."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from jaxolotl.environments.franka_zone_env.franka_zone_env import (
    ZONE_RGBA,
    EnvParams,
    EnvState,
    build_model,
)
from jaxolotl.environments.franka_zone_env.kinematics import compute_fk

_SKELETON_BODIES = tuple(f"link{i}" for i in range(8)) + ("hand",)


def _draw_sphere(ax, center: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    azimuth, elevation = np.mgrid[0 : 2 * np.pi : 32j, 0 : np.pi : 20j]
    x = center[0] + radius * np.cos(azimuth) * np.sin(elevation)
    y = center[1] + radius * np.sin(azimuth) * np.sin(elevation)
    z = center[2] + radius * np.cos(elevation)
    ax.plot_surface(
        x,
        y,
        z,
        color=rgba[:3],
        alpha=float(rgba[3]),
        linewidth=0,
        shade=True,
    )
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=4,
        cstride=4,
        color=rgba[:3],
        linewidth=0.35,
        alpha=min(float(rgba[3]) + 0.25, 0.65),
    )


def _draw_link(
    ax, start: np.ndarray, end: np.ndarray, radius: float, alpha: float
) -> None:
    """Draw a shaded cylindrical limb so arm width follows perspective."""
    direction = end - start
    length = np.linalg.norm(direction)
    if np.isclose(length, 0.0):
        return
    direction /= length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normal_a = np.cross(direction, reference)
    normal_a /= np.linalg.norm(normal_a)
    normal_b = np.cross(direction, normal_a)

    angles = np.linspace(0.0, 2.0 * np.pi, 12)
    offsets = radius * (
        np.cos(angles)[:, None] * normal_a + np.sin(angles)[:, None] * normal_b
    )
    ends = np.stack((start, end))
    cylinder = ends[None, :, :] + offsets[:, None, :]
    ax.plot_surface(
        cylinder[:, :, 0],
        cylinder[:, :, 1],
        cylinder[:, :, 2],
        color="#4c566a",
        alpha=alpha,
        linewidth=0,
        shade=True,
    )


def _skeleton_points(
    model: mujoco.MjModel,  # type: ignore
    data: mujoco.MjData,  # type: ignore
    qpos: np.ndarray,
) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)  # type: ignore
    points = [np.asarray(data.body(name).xpos).copy() for name in _SKELETON_BODIES]
    points.append(np.asarray(compute_fk(jnp.asarray(qpos[:7])))[:3, 3])
    return np.stack(points)


def _setup_axis(ax) -> None:
    ax.set_xlim(-0.10, 0.85)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0.0, 0.90)
    ax.set_box_aspect((0.95, 0.90, 0.90))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_proj_type("persp", focal_length=0.9)
    ax.view_init(elev=20, azim=135)
    ax.grid(alpha=0.25)


def draw_trajectories(
    trajs: EnvState,
    lengths: jax.Array,
    params: EnvParams,
    *,
    num_cols: int,
    num_rows: int,
    num_ghosts: int = 6,
    save_path: str | None = None,
) -> None:
    """Draw EE paths, episode zones, and evenly spaced robot skeleton poses."""
    lengths_np = np.asarray(jax.device_get(lengths))
    qpos_all = np.asarray(jax.device_get(trajs.data.qpos))
    zones_all = np.asarray(jax.device_get(trajs.zone_centers))
    if num_cols * num_rows < len(lengths_np):
        raise ValueError("The subplot grid is too small for the trajectory batch.")

    model = build_model()
    data = mujoco.MjData(model)  # type: ignore
    figure = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    for episode, length_value in enumerate(lengths_np):
        ax = figure.add_subplot(num_rows, num_cols, episode + 1, projection="3d")
        _setup_axis(ax)
        length = min(int(length_value) + 1, qpos_all.shape[1])
        qpos = qpos_all[episode, :length]
        poses = jax.vmap(compute_fk)(jnp.asarray(qpos[:, :7]))
        path = np.asarray(poses)[:, :3, 3]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#18864b", linewidth=2.5)
        ax.scatter(*path[0], color="#f28e2b", marker="D", s=35, label="start")
        ax.scatter(*path[-1], color="#18864b", marker="o", s=35, label="end")

        for index, center in enumerate(zones_all[episode, 0]):
            _draw_sphere(ax, center, params.zone_radius, ZONE_RGBA[index])

        ghost_indices = np.unique(
            np.linspace(0, max(length - 1, 0), min(num_ghosts, length), dtype=int)
        )
        for ghost_number, frame in enumerate(ghost_indices):
            skeleton = _skeleton_points(model, data, qpos[frame])
            alpha = 0.20 + 0.65 * ghost_number / max(len(ghost_indices) - 1, 1)
            for start, end in zip(skeleton[:-1], skeleton[1:], strict=True):
                _draw_link(ax, start, end, radius=0.025, alpha=alpha)
            ax.scatter(
                skeleton[:, 0],
                skeleton[:, 1],
                skeleton[:, 2],  # type: ignore
                color="#4c566a",
                s=20,
                alpha=alpha,
            )
        ax.set_title(f"Episode {episode + 1}")

    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.show()
