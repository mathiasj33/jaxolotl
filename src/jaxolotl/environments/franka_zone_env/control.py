"""Convert Cartesian pose increments into Panda joint-position targets.

Actions are integrated from the previously commanded pose rather than the
measured pose, so their meaning does not depend on servo tracking error.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments.franka_zone_env.kinematics import (
    Q_MAX,
    Q_MIN,
    compute_fk,
    solve_ik,
)
from jaxolotl.utils.rotation import quat_to_matrix, so3_exp

# The IK treats joint 7 as a redundancy parameter. Search nearby values so the
# controller can use more of the workspace without asking for a large joint jump.
_Q7_CANDIDATES = 17


class ArmController(eqx.Module):
    """Track a commanded grasp-centre pose with bounded inverse kinematics."""

    q7_offsets: jax.Array  # (_Q7_CANDIDATES,) offsets from the current joint 7 to try
    pos_low: jax.Array  # (3,)
    pos_high: jax.Array  # (3,)
    max_joint_step: float

    def __init__(
        self,
        pos_low: tuple[float, ...],
        pos_high: tuple[float, ...],
        max_joint_step: float,
    ):
        self.q7_offsets = jnp.linspace(-max_joint_step, max_joint_step, _Q7_CANDIDATES)
        self.pos_low = jnp.asarray(pos_low, dtype=jnp.float32)
        self.pos_high = jnp.asarray(pos_high, dtype=jnp.float32)
        self.max_joint_step = max_joint_step

    def __call__(
        self, q_cmd: jax.Array, translation: jax.Array, rotation: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return the next joint target and whether the requested increment is feasible.

        Translation (metres) and rotation vector (radians) are expressed in the
        robot base frame. If no bounded IK solution exists, retain the command.
        """
        pose = compute_fk(q_cmd)
        position = jnp.clip(pose[:3, 3] + translation, self.pos_low, self.pos_high)
        # Left composition makes the rotation increment relative to base axes.
        orientation = quat_to_matrix(so3_exp(rotation)) @ pose[:3, :3]
        target = jnp.identity(4).at[:3, :3].set(orientation).at[:3, 3].set(position)

        # Search around the current wrist angle; distant solutions would exceed
        # the per-action joint-motion bound below.
        q7 = jnp.clip(q_cmd[6] + self.q7_offsets, Q_MIN[6], Q_MAX[6])
        solutions, reachable = jax.vmap(solve_ik, in_axes=(None, 0, None))(
            target, q7, q_cmd
        )
        # Prefer the feasible solution with the smallest largest joint movement.
        jump = jnp.max(jnp.abs(solutions - q_cmd), axis=-1)
        usable = reachable & (jump <= self.max_joint_step)
        best = jnp.argmin(jnp.where(usable, jump, jnp.inf))
        found = jnp.any(usable)
        return jnp.where(found, solutions[best], q_cmd), found
