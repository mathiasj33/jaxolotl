"""The Cartesian controller moves the commanded Panda grasp centre safely."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.environments.franka_zone_env.control import ArmController
from jaxolotl.environments.franka_zone_env.franka_zone_env import (
    REACH_HIGH,
    REACH_LOW,
)
from jaxolotl.environments.franka_zone_env.kinematics import HOME_QPOS, compute_fk
from jaxolotl.utils.rotation import quat_to_matrix, so3_exp

MAX_JOINT_STEP = 0.5
HOME = jnp.asarray(HOME_QPOS)


def _controller():
    return eqx.filter_jit(ArmController(REACH_LOW, REACH_HIGH, MAX_JOINT_STEP))


def test_translation_increment_accumulates():
    controller = _controller()
    step = jnp.array([-0.005, 0.003, -0.004])
    q_cmd = HOME
    for _ in range(20):
        q_cmd, found = controller(q_cmd, step, jnp.zeros(3))
        assert bool(found)
    displacement = (
        np.asarray(compute_fk(q_cmd))[:3, 3] - np.asarray(compute_fk(HOME))[:3, 3]
    )
    assert np.allclose(displacement, 20 * np.asarray(step), atol=1e-3)


def test_rotation_increment_uses_base_frame():
    rotation = jnp.array([0.0, 0.0, 0.1])
    q_cmd, found = _controller()(HOME, jnp.zeros(3), rotation)
    assert bool(found)
    home_rotation = np.asarray(compute_fk(HOME))[:3, :3]
    increment = np.asarray(quat_to_matrix(so3_exp(rotation)))
    achieved = np.asarray(compute_fk(q_cmd))[:3, :3]
    assert np.allclose(achieved, increment @ home_rotation, atol=1e-3)


def test_failed_increment_holds_command():
    q_cmd, found = _controller()(HOME, jnp.zeros(3), jnp.array([0.0, 0.0, jnp.pi]))
    assert not bool(found)
    assert jnp.allclose(q_cmd, HOME)


def test_joint_step_and_workspace_are_bounded():
    controller = _controller()
    actions = jax.random.uniform(jax.random.key(0), (64, 6), minval=-1.0, maxval=1.0)
    q_cmd = HOME
    for action in actions:
        previous = q_cmd
        q_cmd, _ = controller(q_cmd, action[:3] * 0.02, action[3:] * 0.1)
        assert np.abs(np.asarray(q_cmd - previous)).max() <= MAX_JOINT_STEP + 1e-6
    position = np.asarray(compute_fk(q_cmd))[:3, 3]
    assert np.all(position >= np.asarray(REACH_LOW) - 1e-3)
    assert np.all(position <= np.asarray(REACH_HIGH) + 1e-3)
