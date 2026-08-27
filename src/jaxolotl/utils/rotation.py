"""JAX-friendly SO(3) and quaternion utilities.

Quaternions use scalar-first ``[w, x, y, z]`` layout. Rotation values occupy
the final axis, so functions support arbitrary leading batch dimensions.
"""

import jax
import jax.numpy as jnp

# Guard the derivative at the zero vector.
_SAFE_NORM_SQ_EPS = 1e-24
# Use the small-angle limit of sin(theta / 2) / theta below this angle.
_SMALL_ANGLE_EPS = 1e-8


def _safe_norm(v: jax.Array, keepdims: bool = False) -> jax.Array:
    """Euclidean norm over the last axis with a well-defined (zero) gradient at
    the origin. Plain ``jnp.linalg.norm`` has a NaN gradient at ``v = 0`` (from
    ``d/dx √x`` at 0), and a ``where`` guard alone does not rescue it because
    both branches are differentiated — so we guard the squared norm instead.
    """
    sq = jnp.sum(v * v, axis=-1, keepdims=keepdims)
    is_zero = sq < _SAFE_NORM_SQ_EPS
    safe_sq = jnp.where(is_zero, 1.0, sq)
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))


def quat_normalize(q: jax.Array) -> jax.Array:
    """Normalize a quaternion to unit norm. Last axis is size 4."""
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_conj(q: jax.Array) -> jax.Array:
    """Quaternion conjugate ``[w, -x, -y, -z]``."""
    w, xyz = q[..., :1], q[..., 1:]
    return jnp.concatenate([w, -xyz], axis=-1)


def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Hamilton product ``q1 ⊗ q2`` for ``[w, x, y, z]`` quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.stack([w, x, y, z], axis=-1)


def rpy_to_quat(rpy: jax.Array) -> jax.Array:
    """Euler ``(roll, pitch, yaw)`` -> unit quaternion.

    Composes ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, i.e.
    ``q = qz(yaw) ⊗ qy(pitch) ⊗ qx(roll)``.
    """
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    hr, hp, hy = 0.5 * roll, 0.5 * pitch, 0.5 * yaw
    zeros = jnp.zeros_like(roll)
    qx = jnp.stack([jnp.cos(hr), jnp.sin(hr), zeros, zeros], axis=-1)
    qy = jnp.stack([jnp.cos(hp), zeros, jnp.sin(hp), zeros], axis=-1)
    qz = jnp.stack([jnp.cos(hy), zeros, zeros, jnp.sin(hy)], axis=-1)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_to_rpy(q: jax.Array) -> jax.Array:
    """Unit quaternion ``[w, x, y, z]`` -> Euler ``(roll, pitch, yaw)``, the inverse of
    :func:`rpy_to_quat` (same ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` ZYX convention).

    At the ``pitch = ±π/2`` gimbal singularity the (roll, yaw) split is ambiguous, but
    the recovered RPY still round-trips through :func:`rpy_to_quat` back to the same
    rotation (geodesic distance 0), which is all the downstream geodesic predicate /
    keepout uses. Supports leading batch dims ``(..., 4) -> (..., 3)``.
    """
    r = quat_to_matrix(q)
    r00, r10, r20 = r[..., 0, 0], r[..., 1, 0], r[..., 2, 0]
    r21, r22 = r[..., 2, 1], r[..., 2, 2]
    pitch = jnp.arctan2(-r20, jnp.sqrt(r00 * r00 + r10 * r10))
    yaw = jnp.arctan2(r10, r00)
    roll = jnp.arctan2(r21, r22)
    return jnp.stack([roll, pitch, yaw], axis=-1)


def quat_to_matrix(q: jax.Array) -> jax.Array:
    """Unit quaternion ``[w, x, y, z]`` -> ``(..., 3, 3)`` rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    row0 = jnp.stack([r00, r01, r02], axis=-1)
    row1 = jnp.stack([r10, r11, r12], axis=-1)
    row2 = jnp.stack([r20, r21, r22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def matrix_to_quat(r: jax.Array) -> jax.Array:
    """``(..., 3, 3)`` proper rotation matrix -> unit quaternion ``[w, x, y, z]``.

    Inverse of :func:`quat_to_matrix` (up to the quaternion double cover ``±q``).
    Uses the four-branch construction (Shepperd's method): each branch divides by
    ``2√tₖ`` where ``tₖ`` is one of the four trace combinations, and the branch
    with the largest ``tₖ`` is selected — that ``tₖ ≥ 1`` for any proper
    rotation, so the active division is well-conditioned. Every branch guards its
    ``√`` with ``maximum(tₖ, ε)`` so the unused branches (which ``where`` still
    evaluates) never produce NaN gradients. Branchless: all four candidates are
    computed and chosen by ``argmax``, so it is ``jit``/``vmap``-friendly.
    """
    r00 = r[..., 0, 0]
    r01 = r[..., 0, 1]
    r02 = r[..., 0, 2]
    r10 = r[..., 1, 0]
    r11 = r[..., 1, 1]
    r12 = r[..., 1, 2]
    r20 = r[..., 2, 0]
    r21 = r[..., 2, 1]
    r22 = r[..., 2, 2]

    t0 = 1.0 + r00 + r11 + r22  # 4 w²
    t1 = 1.0 + r00 - r11 - r22  # 4 x²
    t2 = 1.0 - r00 + r11 - r22  # 4 y²
    t3 = 1.0 - r00 - r11 + r22  # 4 z²

    def _branch(t):
        return 2.0 * jnp.sqrt(jnp.maximum(t, _SMALL_ANGLE_EPS))

    s0 = _branch(t0)
    q0 = jnp.stack(
        [0.25 * s0, (r21 - r12) / s0, (r02 - r20) / s0, (r10 - r01) / s0], axis=-1
    )
    s1 = _branch(t1)
    q1 = jnp.stack(
        [(r21 - r12) / s1, 0.25 * s1, (r01 + r10) / s1, (r02 + r20) / s1], axis=-1
    )
    s2 = _branch(t2)
    q2 = jnp.stack(
        [(r02 - r20) / s2, (r01 + r10) / s2, 0.25 * s2, (r12 + r21) / s2], axis=-1
    )
    s3 = _branch(t3)
    q3 = jnp.stack(
        [(r10 - r01) / s3, (r02 + r20) / s3, (r12 + r21) / s3, 0.25 * s3], axis=-1
    )

    candidates = jnp.stack([q0, q1, q2, q3], axis=-2)  # (..., 4, 4)
    ts = jnp.stack([t0, t1, t2, t3], axis=-1)  # (..., 4)
    pick = jnp.argmax(ts, axis=-1)  # (...,)
    q = jnp.take_along_axis(candidates, pick[..., None, None], axis=-2)[..., 0, :]
    return quat_normalize(q)


def matrix_to_sixd(r: jax.Array) -> jax.Array:
    """``(..., 3, 3)`` rotation matrix -> 6D continuous rotation rep (Zhou et al. 2019).

    Returns the first two columns flattened to ``(..., 6)``. The entries are components of unit
    vectors, hence naturally in ``[-1, 1]``.
    """
    return jnp.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def quat_to_sixd(q: jax.Array) -> jax.Array:
    """Unit quaternion -> 6D continuous rotation rep (Zhou et al. 2019)."""
    return matrix_to_sixd(quat_to_matrix(q))


def so3_exp(omega: jax.Array) -> jax.Array:
    """Rotation-vector (body-frame ``ω·dt``) -> unit quaternion.

    ``omega`` is an axis-angle vector whose magnitude is the rotation angle.
    Uses the closed-form ``exp`` with a small-angle guard so the ``sin(θ/2)/θ``
    coefficient stays finite (and grad-safe) as ``θ -> 0``.
    """
    theta = _safe_norm(omega, keepdims=True)  # (..., 1)
    half = 0.5 * theta
    small = theta < _SMALL_ANGLE_EPS
    safe_theta = jnp.where(small, 1.0, theta)
    # coefficient k = sin(θ/2)/θ, with limit 1/2 as θ -> 0
    k = jnp.where(small, 0.5, jnp.sin(half) / safe_theta)
    w = jnp.cos(half)  # (..., 1)
    vec = omega * k  # (..., 3)
    q = jnp.concatenate([w, vec], axis=-1)
    return quat_normalize(q)


def geodesic_angle(q_a: jax.Array, q_b: jax.Array) -> jax.Array:
    """Geodesic angle on SO(3) between two unit quaternions, in ``[0, π]``.

    Computes ``θ = angle(q_a⁻¹ ⊗ q_b)`` via ``2·atan2(‖vec‖, |w|)`` on the
    relative quaternion. This ``atan2`` form is numerically stable near ``θ = 0``
    (unlike ``2·acos(⟨·⟩)``, whose derivative diverges at 1, producing float
    error and NaN grads). ``|w|`` handles the quaternion double cover (``q`` and
    ``-q`` are the same rotation -> angle 0). Single source of truth for
    rotation keepout and rotation robustness.
    """
    q_rel = quat_mul(quat_conj(q_a), q_b)
    w = jnp.abs(q_rel[..., 0])
    vec_norm = _safe_norm(q_rel[..., 1:])
    return 2.0 * jnp.arctan2(vec_norm, w)


def rpy_to_matrix(rpy: jax.Array) -> jax.Array:
    """Euler ``(roll, pitch, yaw)`` -> ``(..., 3, 3)`` rotation matrix.

    Convenience composition ``quat_to_matrix(rpy_to_quat(rpy))`` — single source
    of truth for the forward ``R_target`` map.
    """
    return quat_to_matrix(rpy_to_quat(rpy))
