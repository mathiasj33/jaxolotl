"""A 3D coloured-zone reaching task for the Franka Emika Panda.

The arm is simulated with MuJoCo MJX and commanded through Cartesian pose
increments. Zones are virtual spheres: they are fully observable, have no
contact geometry, and label the proposition whose sphere contains the grasp
centre.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import lax

from jaxolotl.environments import environment, spaces
from jaxolotl.environments.assets import PANDA_SCENE, asset_path
from jaxolotl.environments.franka_zone_env.control import ArmController
from jaxolotl.environments.franka_zone_env.kinematics import (
    HOME_QPOS,
    Q_MAX,
    Q_MIN,
    compute_fk,
    pose_matrix,
    solve_ik_sweep,
)
from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.utils.mjx_quiet import mjx
from jaxolotl.utils.rotation import matrix_to_sixd

if TYPE_CHECKING:
    from jaxolotl.environments.renderer.renderer import BaseRenderer

_NUM_ARM_JOINTS = 7
_NUM_FINGERS = 2
_ARM_QVEL_SCALE = 4.0
_FINGER_OPEN = 0.04
_ACTION_DIM = 6
_ARM_OBS_DIM = 40
_ZONE_OBS_DIM = 4
_MAX_SAMPLING_ITERS = 5000
_DT_TOL = 1e-9

# The controller and sampler share this conservative reachable box. Zone
# centres are inset by their radius so each sphere stays inside it.
REACH_LOW = (0.25, -0.32, 0.02)
REACH_HIGH = (0.77, 0.32, 0.50)

COLOR_NAMES = (
    "red",
    "green",
    "purple",
    "yellow",
    "blue",
    "orange",
    "cyan",
    "pink",
    "lime",
    "teal",
    "indigo",
    "brown",
    "grey",
    "gold",
    "navy",
    "maroon",
)

# Display colours, aligned with COLOR_NAMES.
ZONE_RGBA = np.asarray(
    [
        (0.84, 0.15, 0.16, 0.28),
        (0.17, 0.63, 0.17, 0.28),
        (0.58, 0.40, 0.74, 0.28),
        (0.95, 0.77, 0.06, 0.28),
        (0.12, 0.47, 0.71, 0.28),
        (1.00, 0.50, 0.05, 0.28),
        (0.09, 0.75, 0.81, 0.28),
        (0.89, 0.47, 0.76, 0.28),
        (0.50, 0.80, 0.10, 0.28),
        (0.00, 0.50, 0.50, 0.28),
        (0.29, 0.00, 0.51, 0.28),
        (0.55, 0.34, 0.29, 0.28),
        (0.50, 0.50, 0.50, 0.28),
        (1.00, 0.65, 0.00, 0.28),
        (0.00, 0.00, 0.50, 0.28),
        (0.50, 0.00, 0.00, 0.28),
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    num_colors: int
    zone_radius: float
    zone_keepout: float
    initial_zone_keepout: float
    spawn_joint_noise: float
    dt: float
    sim_substeps: int
    max_pos_step: float
    max_rot_step: float
    max_joint_step: float
    # Discretize action space for GCRL-LTL
    discretize: bool = False


class ResetDescriptor(eqx.Module):
    """Random reset values, without simulator state."""

    arm_qpos: jax.Array  # (7,)
    zone_centers: jax.Array  # (num_colors, 3)


class EnvState(eqx.Module):
    data: mjx.Data
    q_cmd: jax.Array  # (7,)
    ik_ok: jax.Array  # (), bool
    zone_centers: jax.Array  # (num_colors, 3)

    @property
    def arm_qpos(self) -> jax.Array:
        return self.data.qpos[:_NUM_ARM_JOINTS]

    @property
    def arm_qvel(self) -> jax.Array:
        return self.data.qvel[:_NUM_ARM_JOINTS]

    @property
    def ee_pose(self) -> jax.Array:
        # FK avoids an MJX forward pass when materialising compact resets.
        return compute_fk(self.arm_qpos)


class ObsFeatures(NamedTuple):
    arm: jax.Array  # (40,)
    zones: jax.Array  # (num_colors, 4): relative xyz and distance


class ResetOptions(NamedTuple):
    pass


def build_model() -> mujoco.MjModel:  # type: ignore
    """Compile the vendored collision-minimal Panda and floor scene."""
    return mujoco.MjModel.from_xml_path(str(asset_path(PANDA_SCENE)))  # type: ignore


class FrankaZoneEnv(
    environment.Environment[
        EnvState, ResetDescriptor, EnvParams, ObsFeatures, ResetOptions
    ]
):
    """Panda reaching one non-overlapping virtual sphere per colour."""

    model: mjx.Model
    controller: ArmController
    _data_prototype: mjx.Data

    default_params = EnvParams(
        max_steps_in_episode=200,
        num_colors=4,
        zone_radius=0.07,
        zone_keepout=0.14,
        initial_zone_keepout=0.14,
        spawn_joint_noise=0.15,
        dt=0.05,
        sim_substeps=10,
        max_pos_step=0.02,
        max_rot_step=0.1,
        max_joint_step=0.5,
        discretize=False,
    )

    def __init__(self, num_colors: int = 4, **kwargs):
        if not 4 <= num_colors <= len(COLOR_NAMES):
            raise ValueError(
                f"num_colors must be in [4, {len(COLOR_NAMES)}], got {num_colors}."
            )
        values = dataclasses.asdict(self.default_params) | kwargs
        values["num_colors"] = num_colors
        params = EnvParams(**values)
        if params.zone_keepout < 2.0 * params.zone_radius:
            raise ValueError("zone_keepout must be at least twice zone_radius.")

        super().__init__(params, COLOR_NAMES[:num_colors])
        mj_model = build_model()
        expected_dt = mj_model.opt.timestep * params.sim_substeps
        if abs(expected_dt - params.dt) > _DT_TOL:
            raise ValueError(
                f"dt={params.dt} must equal sim_substeps={params.sim_substeps} "
                f"times the model timestep {mj_model.opt.timestep} ({expected_dt})."
            )
        self.model = mjx.put_model(mj_model)
        self._data_prototype = mjx.make_data(self.model)
        self.controller = ArmController(REACH_LOW, REACH_HIGH, params.max_joint_step)

    @override
    def _observation_spec(self, params: EnvParams) -> ObservationSpec:
        return ObservationSpec(
            arm=ArraySpec((_ARM_OBS_DIM,), jnp.float32),
            zones=ArraySpec((params.num_colors, _ZONE_OBS_DIM), jnp.float32),
        )

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        if params.discretize:
            # Signed full-scale steps along/about each base axis.
            return spaces.Discrete(n=2 * _ACTION_DIM)
        return spaces.Box(shape=(_ACTION_DIM,), low=-1.0, high=1.0, dtype=jnp.float32)

    @override
    def _sample_reset(
        self,
        key: jax.Array,
        descriptor: ResetDescriptor | EnvState | None,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> ResetDescriptor:
        key_arm, key_zones = jax.random.split(key)
        arm_qpos = self._sample_arm_qpos(key_arm, params)
        zone_centers = self._sample_zones(key_zones, arm_qpos, params)
        return ResetDescriptor(arm_qpos=arm_qpos, zone_centers=zone_centers)

    def _sample_arm_qpos(self, key: jax.Array, params: EnvParams) -> jax.Array:
        home = jnp.asarray(HOME_QPOS)
        q_min = jnp.asarray(Q_MIN)
        q_max = jnp.asarray(Q_MAX)
        low = jnp.asarray(REACH_LOW) + params.zone_radius
        high = jnp.asarray(REACH_HIGH) - params.zone_radius

        def propose(subkey):
            noise = jax.random.uniform(
                subkey,
                (_NUM_ARM_JOINTS,),
                minval=-params.spawn_joint_noise,
                maxval=params.spawn_joint_noise,
            )
            qpos = jnp.clip(home + noise, q_min, q_max)
            position = compute_fk(qpos)[:3, 3]
            return qpos, jnp.all((position >= low) & (position <= high))

        return self._draw_until(key, home, propose, "initial arm pose")

    def _sample_zones(
        self, key: jax.Array, arm_qpos: jax.Array, params: EnvParams
    ) -> jax.Array:
        num_colors = params.num_colors
        low = jnp.asarray(REACH_LOW) + params.zone_radius
        high = jnp.asarray(REACH_HIGH) - params.zone_radius
        centers = jnp.zeros((num_colors, 3), dtype=jnp.float32)
        initial_position = compute_fk(arm_qpos)[:3, 3]
        canonical_rotation = compute_fk(jnp.asarray(HOME_QPOS))[:3, :3]
        home = jnp.asarray(HOME_QPOS)

        def condition(carry):
            _key, _centers, count, iterations = carry
            return (count < num_colors) & (iterations < _MAX_SAMPLING_ITERS)

        def body(carry):
            key, centers, count, iterations = carry
            key, subkey = jax.random.split(key)
            proposal = jax.random.uniform(subkey, (3,), minval=low, maxval=high)
            existing = jnp.arange(num_colors) < count
            separated = jnp.all(
                (~existing)
                | (jnp.linalg.norm(centers - proposal, axis=-1) >= params.zone_keepout)
            )
            clear_of_start = (
                jnp.linalg.norm(proposal - initial_position)
                >= params.initial_zone_keepout
            )
            _, reachable = solve_ik_sweep(
                pose_matrix(proposal, canonical_rotation), home
            )
            accepted = separated & clear_of_start & reachable
            centers = lax.cond(
                accepted,
                lambda value: value.at[count].set(proposal),
                lambda value: value,
                centers,
            )
            count = count + accepted.astype(jnp.int32)
            return key, centers, count, iterations + 1

        _, centers, count, _ = lax.while_loop(
            condition,
            body,
            (key, centers, jnp.int32(0), jnp.int32(0)),
        )
        return eqx.error_if(
            centers,
            count < num_colors,
            "Could not sample non-overlapping reachable Franka zones. Reduce the "
            "number/radius/keepout of zones or enlarge the reach box.",
        )

    @staticmethod
    def _draw_until(key, fallback, propose, name):
        def condition(carry):
            _key, _value, accepted, iterations = carry
            return (~accepted) & (iterations < _MAX_SAMPLING_ITERS)

        def body(carry):
            key, _value, _accepted, iterations = carry
            key, subkey = jax.random.split(key)
            value, accepted = propose(subkey)
            return key, value, accepted, iterations + 1

        key, subkey = jax.random.split(key)
        value, accepted = propose(subkey)
        _, value, accepted, _ = lax.while_loop(
            condition, body, (key, value, accepted, jnp.int32(1))
        )
        return eqx.error_if(
            jnp.where(accepted, value, fallback),
            ~accepted,
            f"Could not sample a valid {name}.",
        )

    @override
    def materialize(
        self,
        descriptor: ResetDescriptor,
        params: EnvParams,  # noqa: ARG002
    ) -> EnvState:
        fingers = jnp.full((_NUM_FINGERS,), _FINGER_OPEN, dtype=jnp.float32)
        qpos = jnp.concatenate([descriptor.arm_qpos, fingers])
        data = self._data_prototype.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(self._data_prototype.qvel),
            ctrl=descriptor.arm_qpos,
            time=jnp.zeros_like(self._data_prototype.time),
        )
        return EnvState(
            data=data,
            q_cmd=descriptor.arm_qpos,
            ik_ok=jnp.asarray(True),
            zone_centers=descriptor.zone_centers,
        )

    def _map_discrete_action(self, action: jax.Array) -> jax.Array:
        # Translation along +x, +y, +z, rotation about +x, +y, +z, then the
        # same six steps negated.
        mapping = jnp.concatenate([jnp.eye(_ACTION_DIM), -jnp.eye(_ACTION_DIM)])
        return mapping[action]

    @override
    def _step(
        self,
        key: jax.Array,  # noqa: ARG002
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        if params.discretize:
            action = self._map_discrete_action(action)
        action = jnp.clip(action, -1.0, 1.0)
        q_cmd, ik_ok = self.controller(
            state.q_cmd,
            action[:3] * params.max_pos_step,
            action[3:] * params.max_rot_step,
        )

        def one_physics_step(data, _):
            # MJX does not apply body gravcomp, so compensate the arm only.
            gravcomp = data.qfrc_bias.at[_NUM_ARM_JOINTS:].set(0.0)
            data = data.replace(ctrl=q_cmd, qfrc_applied=gravcomp)
            return mjx.step(self.model, data), None

        data, _ = lax.scan(
            one_physics_step, state.data, None, length=params.sim_substeps
        )
        next_state = EnvState(
            data=data,
            q_cmd=q_cmd,
            ik_ok=ik_ok,
            zone_centers=state.zone_centers,
        )
        return (
            next_state,
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.bool),
            {"ik_success": ik_ok},
        )

    def _normalize_position(self, position: jax.Array) -> jax.Array:
        low = jnp.asarray(REACH_LOW)
        high = jnp.asarray(REACH_HIGH)
        return 2.0 * (position - low) / (high - low) - 1.0

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        actual = state.ee_pose
        commanded = compute_fk(state.q_cmd)
        arm = jnp.concatenate(
            [
                jnp.sin(state.arm_qpos),
                jnp.cos(state.arm_qpos),
                state.arm_qvel / _ARM_QVEL_SCALE,
                self._normalize_position(actual[:3, 3]),
                matrix_to_sixd(actual[:3, :3]),
                self._normalize_position(commanded[:3, 3]),
                matrix_to_sixd(commanded[:3, :3]),
                jnp.atleast_1d(jnp.where(state.ik_ok, 1.0, -1.0)),
            ]
        ).astype(jnp.float32)

        relative = state.zone_centers - actual[:3, 3]
        span = jnp.asarray(REACH_HIGH) - jnp.asarray(REACH_LOW)
        relative_normalized = relative / span
        distance_normalized = jnp.linalg.norm(
            relative, axis=-1, keepdims=True
        ) / jnp.linalg.norm(span)
        zones = jnp.concatenate([relative_normalized, distance_normalized], axis=-1)
        return ObsFeatures(arm=arm, zones=zones.astype(jnp.float32))

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        distance = jnp.linalg.norm(state.zone_centers - state.ee_pose[:3, 3], axis=-1)
        indices = jnp.arange(params.num_colors, dtype=jnp.int32)
        return jnp.where(distance < params.zone_radius, indices, -1)

    def assignments(self) -> list[Assignment]:  # type: ignore[override]
        assignments = [Assignment(color) for color in self.propositions]
        assignments.append(Assignment(frozenset()))  # empty assignment
        return assignments

    @override
    def get_renderer(self, params: EnvParams, **kwargs) -> "BaseRenderer[ObsFeatures]":
        from jaxolotl.environments.franka_zone_env.renderer import (  # noqa: PLC0415
            Renderer,
        )

        return Renderer(self, params, **kwargs)

    @override
    def plot_trajectories(
        self,
        trajs: EnvState,
        lengths: jax.Array,
        params: EnvParams,
        save_path: str | None = None,
        **plotting_kwargs,
    ) -> None:
        from jaxolotl.environments.franka_zone_env.plotter import (  # noqa: PLC0415
            draw_trajectories,
        )

        draw_trajectories(
            trajs,
            lengths,
            params,
            save_path=save_path,
            **plotting_kwargs,
        )
