"""Closed-form Panda kinematics in the grasp-centre frame.

Adapted from MuJoCo Playground's Apache-2.0 Panda implementation. The seventh
joint is the IK redundancy parameter. Poses refer to the grasp centre between
the fingertips, not the flange.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

_PI_2 = math.pi / 2
_PI_4 = math.pi / 4

# Link geometry in metres. _D7E extends from the flange to the grasp centre.
_D1 = 0.333
_D3 = 0.316
_D5 = 0.384
_D7E = 0.2104
_A4 = 0.0825
_A7 = 0.088

# Fixed geometry for the joint 2--4--6 triangle used by the elbow solution.
_LL24 = 0.10666225
_LL46 = 0.15426225
_L24 = 0.326591870689
_L46 = 0.392762332715
_THETA_H46 = 1.35916951803
_THETA_342 = 1.31542071191
_THETA_46H = 0.211626808766

# Keep constants as NumPy arrays so importing this module does not initialise JAX.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# The scene's home pose; used to seed and fall back from redundancy sweeps.
HOME_QPOS = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853], dtype=np.float32)

# Number of absolute joint-7 values used when testing reachability.
_Q7_CANDIDATES = 17

# Avoid unstable joint-1 estimates near the shoulder singularity.
_SHOULDER_SINGULAR = 0.999

# Tolerance for validating an IK solution with forward kinematics.
_IK_RESIDUAL_TOL = 1e-3


def _dh_revolute(
    theta: jax.Array | float, alpha: float, a: float, d: float
) -> jax.Array:
    """Homogeneous transform of one revolute joint from its modified-DH parameters."""
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    return jnp.asarray(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def compute_fk(joint_pos: jax.Array) -> jax.Array:
    """Grasp-centre pose as a 4x4 homogeneous transform in the base frame, given `q[:7]`."""
    transforms = [
        _dh_revolute(joint_pos[0], 0.0, 0.0, _D1),
        _dh_revolute(joint_pos[1], -_PI_2, 0.0, 0.0),
        _dh_revolute(joint_pos[2], _PI_2, 0.0, _D3),
        _dh_revolute(joint_pos[3], _PI_2, _A4, 0.0),
        _dh_revolute(joint_pos[4], -_PI_2, -_A4, _D5),
        _dh_revolute(joint_pos[5], _PI_2, 0.0, 0.0),
        _dh_revolute(joint_pos[6], _PI_2, _A7, 0.0),
        # The hand is rotated by a quarter turn relative to the flange.
        _dh_revolute(-_PI_4, 0.0, 0.0, _D7E),
    ]
    pose = jnp.identity(4)
    for transform in transforms:
        pose = pose @ transform
    return pose


def _branch_frames(q_ref: jax.Array) -> list[jax.Array]:
    """Cumulative frames of `q_ref` in He's convention, whose geometry picks the IK's branch."""
    c, s = jnp.cos(q_ref), jnp.sin(q_ref)
    links = [
        jnp.array(
            [
                [c[0], -s[0], 0.0, 0.0],
                [s[0], c[0], 0.0, 0.0],
                [0.0, 0.0, 1.0, _D1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        jnp.array(
            [
                [c[1], -s[1], 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-s[1], -c[1], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        jnp.array(
            [
                [c[2], -s[2], 0.0, 0.0],
                [0.0, 0.0, -1.0, -_D3],
                [s[2], c[2], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        jnp.array(
            [
                [c[3], -s[3], 0.0, _A4],
                [0.0, 0.0, -1.0, 0.0],
                [s[3], c[3], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        jnp.array(
            [
                [1.0, 0.0, 0.0, -_A4],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        jnp.array(
            [
                [c[4], -s[4], 0.0, 0.0],
                [0.0, 0.0, 1.0, _D5],
                [-s[4], -c[4], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        jnp.array(
            [
                [c[5], -s[5], 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [s[5], c[5], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ]
    frames = [links[0]]
    for link in links[1:]:
        frames.append(frames[-1] @ link)
    return frames


def compute_ik(  # noqa: PLR0915
    pose: jax.Array, q7: jax.Array, q_ref: jax.Array
) -> jax.Array:
    """The seven joint angles putting the grasp centre at `pose` with joint 7 held at `q7`.

    `q_ref` is a reference configuration, used only to choose among the mirrored solution branches
    and to fill in joint 1 at the shoulder singularity; passing the arm's current angles is what
    makes the solution the continuation of where it already is.

    Returns NaN where `pose` is out of reach, and angles outside `Q_MIN`/`Q_MAX` where it is
    reachable only by a configuration the joints forbid. Callers must check both -- see
    `solve_ik`.
    """
    frames = _branch_frames(q_ref)

    # Resolve the two shoulder and wrist branches using the reference pose.
    v62 = frames[1][:3, 3] - frames[6][:3, 3]
    v6h = frames[4][:3, 3] - frames[6][:3, 3]
    is_case6_0 = jnp.sum(jnp.cross(v6h, v62) * frames[6][:3, 2]) <= 0
    is_case1_1 = q_ref[1] < 0

    q = jnp.zeros(7).at[6].set(q7)

    # Recover joint 6 from the grasp-centre pose.
    rot_ee = pose[:3, :3]
    z_ee = pose[:3, 2]
    p_7 = pose[:3, 3] - _D7E * z_ee
    x_6 = rot_ee @ jnp.array([jnp.cos(q7 - _PI_4), -jnp.sin(q7 - _PI_4), 0.0])
    x_6 = x_6 / jnp.linalg.norm(x_6)
    p_6 = p_7 - _A7 * x_6

    # Solve the elbow angle from the shoulder-to-wrist triangle.
    p_2 = jnp.array([0.0, 0.0, _D1])
    v26 = p_6 - p_2
    ll26 = jnp.sum(v26 * v26)
    l26 = jnp.sqrt(ll26)
    theta246 = jnp.arccos((_LL24 + _LL46 - ll26) / 2.0 / _L24 / _L46)
    q = q.at[3].set(theta246 + _THETA_H46 + _THETA_342 - 2.0 * jnp.pi)

    # Solve joint 6 from the same triangle.
    theta462 = jnp.arccos((ll26 + _LL46 - _LL24) / 2.0 / l26 / _L46)
    theta26h = _THETA_46H + theta462
    d26 = -l26 * jnp.cos(theta26h)

    z_6 = jnp.cross(z_ee, x_6)
    y_6 = jnp.cross(z_6, x_6)
    rot_6 = jnp.column_stack(
        (x_6, y_6 / jnp.linalg.norm(y_6), z_6 / jnp.linalg.norm(z_6))
    )
    v_6_62 = rot_6.T @ -v26

    phi6 = jnp.arctan2(v_6_62[1], v_6_62[0])
    theta6 = jnp.arcsin(d26 / jnp.sqrt(v_6_62[0] ** 2 + v_6_62[1] ** 2))
    q = jnp.where(
        is_case6_0, q.at[5].set(jnp.pi - theta6 - phi6), q.at[5].set(theta6 - phi6)
    )
    # Wrap the equivalent joint-6 angle into its allowed range.
    q = jnp.where(q[5] <= Q_MIN[5], q.at[5].set(q[5] + 2.0 * jnp.pi), q)
    q = jnp.where(q[5] >= Q_MAX[5], q.at[5].set(q[5] - 2.0 * jnp.pi), q)

    # Solve joints 1 and 2 from the elbow position.
    theta_p26 = 3.0 * jnp.pi / 2 - theta462 - theta246 - _THETA_342
    theta_p = jnp.pi - theta_p26 - theta26h
    lp6 = l26 * jnp.sin(theta_p26) / jnp.sin(theta_p)

    z_5 = rot_6 @ jnp.array([jnp.sin(q[5]), jnp.cos(q[5]), 0.0])
    v2p = p_6 - lp6 * z_5 - p_2
    l2p = jnp.linalg.norm(v2p)

    singular = jnp.abs(v2p[2] / l2p) > _SHOULDER_SINGULAR
    q = jnp.where(
        singular, q.at[0].set(q_ref[0]), q.at[0].set(jnp.arctan2(v2p[1], v2p[0]))
    )
    q = jnp.where(singular, q.at[1].set(0.0), q.at[1].set(jnp.arccos(v2p[2] / l2p)))
    # Select the mirrored shoulder branch when required.
    q = jnp.where(
        is_case1_1,
        jnp.where(q[0] < 0.0, q.at[0].set(q[0] + jnp.pi), q.at[0].set(q[0] - jnp.pi)),
        q,
    )
    q = jnp.where(singular, q, jnp.where(is_case1_1, q.at[1].set(-q[1]), q))

    # Recover the remaining upper-arm and forearm roll angles.
    z_3 = v2p / l2p
    y_3 = -jnp.cross(v26, v2p)
    y_3 = y_3 / jnp.linalg.norm(y_3)
    x_3 = jnp.cross(y_3, z_3)
    c1, s1 = jnp.cos(q[0]), jnp.sin(q[0])
    rot_1 = jnp.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
    c2, s2 = jnp.cos(q[1]), jnp.sin(q[1])
    rot_2 = rot_1 @ jnp.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
    x_2_3 = rot_2.T @ x_3
    q = q.at[2].set(jnp.arctan2(x_2_3[2], x_2_3[0]))

    c6, s6 = jnp.cos(q[5]), jnp.sin(q[5])
    rot_5 = rot_6 @ jnp.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]]).T
    v_5_h4 = rot_5.T @ (p_2 + _D3 * z_3 + _A4 * x_3 - p_6 + _D5 * z_5)
    q = q.at[4].set(-jnp.arctan2(v_5_h4[1], v_5_h4[0]))

    return q


def solve_ik(
    pose: jax.Array, q7: jax.Array, q_ref: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """`compute_ik`, with the solution's usability as a boolean beside it.

    Usable means finite, inside the joint limits, and actually reaching `pose`: on a small
    fraction of poses the closed form picks a branch that satisfies its own equations but lands
    elsewhere, and only the residual catches those. The angles returned are `q_ref` wherever the
    solution is not usable, since NaN must not reach `qpos`.
    """
    q = compute_ik(pose, q7, q_ref)
    reached = compute_fk(jnp.where(jnp.isfinite(q), q, 0.0))
    residual = jnp.max(jnp.abs(reached[:3, :] - pose[:3, :]))
    ok = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(q >= Q_MIN)
        & jnp.all(q <= Q_MAX)
        & (residual < _IK_RESIDUAL_TOL)
    )
    return jnp.where(ok, q, q_ref), ok


def pose_matrix(position: jax.Array, rotation: jax.Array) -> jax.Array:
    """The (4, 4) grasp-centre transform these functions take, from a position and a (3, 3)."""
    return jnp.identity(4).at[:3, :3].set(rotation).at[:3, 3].set(position)


def solve_ik_sweep(pose: jax.Array, q_ref: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`solve_ik` over the wrist redundancy: a usable configuration, and whether one exists.

    The seventh joint is an input to the closed form rather than an output, so whether the arm can
    reach a pose at all is a question about the whole sweep -- unreachable means every wrist angle
    failed. The returned configuration is `q_ref` when none did.
    """
    q7_grid = jnp.linspace(Q_MIN[6], Q_MAX[6], _Q7_CANDIDATES)
    solutions, reachable = jax.vmap(solve_ik, in_axes=(None, 0, None))(
        pose, q7_grid, q_ref
    )
    return solutions[jnp.argmax(reachable)], jnp.any(reachable)
