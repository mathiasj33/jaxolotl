"""Panda analytical kinematics agree with the vendored MuJoCo model."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from jaxolotl.environments.assets import PANDA_SCENE, asset_path
from jaxolotl.environments.franka_zone_env import kinematics as kin


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:  # type: ignore
    return mujoco.MjModel.from_xml_path(str(asset_path(PANDA_SCENE)))  # type: ignore


def _random_configurations(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(np.asarray(kin.Q_MIN), np.asarray(kin.Q_MAX), size=(n, 7))


def test_fk_lands_on_grasp_centre(model: mujoco.MjModel):  # type: ignore
    data = mujoco.MjData(model)  # type: ignore
    for qpos in _random_configurations(16, seed=0):
        data.qpos[:7] = qpos
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)  # type: ignore
        pose = np.asarray(kin.compute_fk(jnp.asarray(qpos)))
        pad_midpoint = 0.5 * (
            data.geom("left_finger_pad").xpos + data.geom("right_finger_pad").xpos
        )
        assert np.linalg.norm(pose[:3, 3] - pad_midpoint) < 5e-4
        assert (
            np.abs(pose[:3, :3] - data.site("gripper").xmat.reshape(3, 3)).max() < 1e-5
        )


def test_usable_ik_solutions_round_trip():
    solve = jax.jit(kin.solve_ik)
    usable = 0
    for qpos in _random_configurations(128, seed=1):
        pose = kin.compute_fk(jnp.asarray(qpos))
        solution, ok = solve(pose, jnp.asarray(qpos[6]), jnp.asarray(qpos))
        if not bool(ok):
            continue
        usable += 1
        reached = np.asarray(kin.compute_fk(solution))
        assert np.linalg.norm(reached[:3, 3] - np.asarray(pose)[:3, 3]) < 1e-3
    assert usable > 0.9 * 128


def test_unreachable_pose_returns_reference_without_nan():
    reference = jnp.zeros(7).at[3].set(-1.5)
    pose = jnp.identity(4).at[:3, 3].set(jnp.array([3.0, 0.0, 1.0]))
    solution, ok = kin.solve_ik(pose, jnp.asarray(0.0), reference)
    assert not bool(ok)
    assert jnp.allclose(solution, reference)


def test_redundancy_sweep_expands_reach():
    grid = jnp.linspace(kin.Q_MIN[6], kin.Q_MAX[6], 17)
    frozen = jax.jit(kin.solve_ik)
    swept = jax.jit(jax.vmap(kin.solve_ik, in_axes=(None, 0, None)))
    frozen_hits = 0
    swept_hits = 0
    for target, reference in zip(
        _random_configurations(32, seed=2),
        _random_configurations(32, seed=3),
        strict=True,
    ):
        pose = kin.compute_fk(jnp.asarray(target))
        q_ref = jnp.asarray(reference)
        frozen_hits += int(bool(frozen(pose, q_ref[6], q_ref)[1]))
        swept_hits += int(bool(jnp.any(swept(pose, grid, q_ref)[1])))
    assert swept_hits > 2 * frozen_hits
