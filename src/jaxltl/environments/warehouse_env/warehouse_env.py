"""A continuous Warehouse environment.

The environment simulates a point-mass agent moving in a 2D plane. The agent
applies a forward force aligned with its current heading and can control its
angular velocity. The world contains vases and crates that the agent can
pick up and drop. There are designated regions in the environment: Region A,
Region B, and a Door area. The agent can pick up vases and crates when it is
within a certain radius of them and can drop them anywhere in the environment.
The agent is equipped with a lidar sensor that detects the distance to the
nearest object in a set of evenly spaced angular bins.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments import environment, spaces
from jaxltl.ltl.logic.assignment import Assignment

if TYPE_CHECKING:
    from jaxltl.environments.renderer.renderer import BaseRenderer

_EPS = 1e-8
_DO_NOTHING_INDEX = 0
_PICKUP_VASE_INDEX = 1
_PICKUP_CRATE_INDEX = 2
_DROP_VASE_INDEX = 3
_DROP_CRATE_INDEX = 4


@dataclass(frozen=True)
class WarehouseParams(environment.EnvParams):
    # World Dimensions
    world_size: float

    # Object settings
    num_vases: int
    num_crates: int
    pickup_radius: float
    pickup_cooldown_steps: int  # Steps to wait after dropping before pickup allowed

    # Region settings
    region_a: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    region_b: tuple[float, float, float, float]
    door_region: tuple[float, float, float, float]

    # Lidar
    num_lidar_bins: int
    exp_gain: float

    # Physics
    dt: float
    drag: float
    max_speed: float
    max_force: float
    max_angular_velocity: float


class EnvState(eqx.Module):
    # Physics
    position: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angle: jax.Array  # shape: ()
    angular_velocity: jax.Array  # float
    acceleration: jax.Array  # shape: (2,)

    # Objects
    vase_positions: jax.Array  # shape: (num_vases, 2)
    vase_available: jax.Array  # shape: (num_vases,)
    crate_positions: jax.Array  # shape: (num_crates, 2)
    crate_available: jax.Array  # shape: (num_crates,)

    # Inventory State
    carrying_vase_idx: jax.Array  # shape: ()
    carrying_crate_idx: jax.Array  # shape: ()

    # Cooldown State
    vase_pickup_cooldown: jax.Array  # shape: () - steps until vase pickup allowed
    crate_pickup_cooldown: jax.Array  # shape: () - steps until crate pickup allowed


class ObsFeatures(NamedTuple):
    acceleration: jax.Array
    velocity: jax.Array
    angular_velocity: jax.Array
    lidar: jax.Array
    global_position: jax.Array
    region_vectors: jax.Array  # (3, 2) directions to fixed regions
    carrying_status: jax.Array  # (2,) vase and crate carrying status


class ResetOptions(NamedTuple):
    pass


class WarehouseEnv(
    environment.Environment[EnvState, WarehouseParams, ObsFeatures, ResetOptions]
):
    default_params = WarehouseParams(
        max_steps_in_episode=1000,
        world_size=6.6,
        num_vases=4,
        num_crates=4,
        pickup_radius=0.2,
        pickup_cooldown_steps=10,
        region_a=(0, 2.2, 1.5, 3.7),
        region_b=(3.7, 6.6, 0, 1.5),
        door_region=(5, 6.6, 5.8, 6.6),
        num_lidar_bins=16,
        exp_gain=0.5,
        dt=0.05,
        drag=0.08,
        max_speed=3.0,
        max_force=2.0,
        max_angular_velocity=3.0,
    )

    propositions = ("region_a", "region_b", "door", "vase", "crate")
    max_nodes = 29
    max_edges = 26

    def __init__(self, **kwargs):
        params = dataclasses.asdict(self.default_params) | kwargs
        super().__init__(
            default_params=WarehouseParams(**params),
            propositions=self.propositions,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )

    @override
    def _observation_space(self, params: WarehouseParams) -> spaces.Space:
        lidar_size = 2 * params.num_lidar_bins
        # 5 (phys) + 2 (pos) + 2 (carry) + lidar + 6 (3 regions * 2 coords)
        flat_dim = 9 + lidar_size + 6
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(flat_dim,),
            dtype=jnp.float32,
        )

    @override
    def _action_space(self, params: WarehouseParams) -> spaces.Space:
        # Continuous: [force, angular_velocity], Discrete: [do nothing, pick up vase,
        # pick up crate, drop vase, drop crate]
        cont_space = spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )
        disc_space = spaces.Discrete(n=5)
        return spaces.Composite(cont_space, disc_space)

    @override
    def _reset(
        self,
        key: jax.Array,
        state: EnvState | None,
        params: WarehouseParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        angle_key, pos_key = jax.random.split(key, 2)
        angle = jax.random.uniform(angle_key, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        positions = self._sample_objects(pos_key, params)
        agent_pos = positions[0]
        vase_pos = positions[1 : 1 + params.num_vases]
        crate_pos = positions[1 + params.num_vases :]

        return EnvState(
            position=agent_pos,
            velocity=jnp.zeros(2),
            angle=angle,
            angular_velocity=jnp.zeros(()),
            acceleration=jnp.zeros(2),
            vase_positions=vase_pos,
            vase_available=jnp.ones(params.num_vases, dtype=jnp.bool),
            crate_positions=crate_pos,
            crate_available=jnp.ones(params.num_crates, dtype=jnp.bool),
            carrying_vase_idx=jnp.array(-1, dtype=jnp.int32),
            carrying_crate_idx=jnp.array(-1, dtype=jnp.int32),
            vase_pickup_cooldown=jnp.array(0, dtype=jnp.int32),
            crate_pickup_cooldown=jnp.array(0, dtype=jnp.int32),
        )

    def _sample_objects(self, key: jax.Array, params: WarehouseParams) -> jax.Array:
        """Sample non-overlapping objects.

        Returns (centers:(1 + num_vases + num_crates,2))  with + 1 for agent position.
        """
        num_objects = 1 + params.num_vases + params.num_crates
        minval = params.pickup_radius + 0.1
        maxval = params.world_size - (params.pickup_radius + 0.1)

        centers0 = jnp.zeros((num_objects, 2), dtype=jnp.float32)

        def cond_fun(carry):
            key, centers, count = carry
            return count < num_objects

        def body_fun(carry):
            key, centers, count = carry
            key, sub = jax.random.split(key)
            proposal = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
            idxs = jnp.arange(num_objects)
            mask = idxs < count
            dists = jnp.linalg.norm(centers - proposal, axis=1)
            cond_radius = dists >= 2.0 * params.pickup_radius
            cond_radius = jnp.all(jnp.logical_or(~mask, cond_radius))
            radius = jnp.array(params.pickup_radius)
            in_region_a = self._pos_in_region(proposal, params.region_a, radius)  # type: ignore
            in_region_b = self._pos_in_region(proposal, params.region_b, radius)  # type: ignore
            at_door = self._pos_in_region(proposal, params.door_region, radius)  # type: ignore
            cond_regions = jnp.logical_not(
                jnp.logical_or(jnp.logical_or(in_region_a, in_region_b), at_door)
            )
            ok = jnp.logical_and(cond_radius, cond_regions)
            centers = jax.lax.cond(
                ok,
                lambda c: c.at[count].set(proposal),
                lambda c: c,
                centers,
            )
            count = count + jnp.where(ok, 1, 0)
            return key, centers, count

        key, centers, count = jax.lax.while_loop(
            cond_fun, body_fun, (key, centers0, jnp.int32(0))
        )
        return centers

    @override
    def _cheap_reset(
        self,
        key: jax.Array,
        state: EnvState,
        params: WarehouseParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        raise NotImplementedError("Cheap reset is not implemented for WarehouseEnv.")

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: tuple[jax.Array, jax.Array],
        params: WarehouseParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        cont_action, disc_action = action

        # --- Cooldown Updates ---
        vase_pickup_cooldown = jnp.maximum(state.vase_pickup_cooldown - 1, 0)
        crate_pickup_cooldown = jnp.maximum(state.crate_pickup_cooldown - 1, 0)

        # --- Dynamics ---
        force = jnp.clip(
            cont_action[0] * params.max_force, -params.max_force, params.max_force
        )
        angular_velocity = jnp.clip(
            cont_action[1] * params.max_angular_velocity,
            -params.max_angular_velocity,
            params.max_angular_velocity,
        )
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])
        acceleration = heading * force

        velocity = state.velocity + acceleration * params.dt
        velocity *= 1.0 - params.drag

        speed = jnp.linalg.norm(velocity)
        scaling_factor = jnp.clip(params.max_speed / speed, 0.0, 1.0)
        velocity: jax.Array = jnp.where(
            speed > params.max_speed, velocity * scaling_factor, velocity
        )

        position = state.position + velocity * params.dt
        angle = self._wrap_angle(state.angle + angular_velocity * params.dt)

        # -- Pickup Logic --
        carrying_vase_idx, vase_available = jax.lax.cond(
            disc_action == _PICKUP_VASE_INDEX,
            lambda: self._handle_pickup_object(
                position,
                state.vase_positions,
                state.vase_available,
                state.carrying_vase_idx,
                vase_pickup_cooldown,
                params,
            ),
            lambda: (state.carrying_vase_idx, state.vase_available),
        )
        carrying_crate_idx, crate_available = jax.lax.cond(
            disc_action == _PICKUP_CRATE_INDEX,
            lambda: self._handle_pickup_object(
                position,
                state.crate_positions,
                state.crate_available,
                state.carrying_crate_idx,
                crate_pickup_cooldown,
                params,
            ),
            lambda: (state.carrying_crate_idx, state.crate_available),
        )

        # -- Drop Logic --
        carrying_vase_idx, vase_positions, vase_available, vase_pickup_cooldown = (
            jax.lax.cond(
                disc_action == _DROP_VASE_INDEX,
                lambda: self._handle_drop_object(
                    position,
                    state.vase_positions,
                    state.vase_available,
                    carrying_vase_idx,
                    vase_pickup_cooldown,
                    params.pickup_cooldown_steps,
                ),
                lambda: (
                    carrying_vase_idx,
                    state.vase_positions,
                    vase_available,
                    vase_pickup_cooldown,
                ),
            )
        )
        carrying_crate_idx, crate_positions, crate_available, crate_pickup_cooldown = (
            jax.lax.cond(
                disc_action == _DROP_CRATE_INDEX,
                lambda: self._handle_drop_object(
                    position,
                    state.crate_positions,
                    state.crate_available,
                    carrying_crate_idx,
                    crate_pickup_cooldown,
                    params.pickup_cooldown_steps,
                ),
                lambda: (
                    carrying_crate_idx,
                    state.crate_positions,
                    crate_available,
                    crate_pickup_cooldown,
                ),
            )
        )

        terminated = jnp.logical_or(
            jnp.any(position < 0.0), jnp.any(position > params.world_size)
        )

        next_state = EnvState(
            position=position,
            velocity=velocity,
            angle=angle,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
            vase_positions=vase_positions,
            vase_available=vase_available,
            crate_positions=crate_positions,
            crate_available=crate_available,
            carrying_vase_idx=carrying_vase_idx,
            carrying_crate_idx=carrying_crate_idx,
            vase_pickup_cooldown=vase_pickup_cooldown,
            crate_pickup_cooldown=crate_pickup_cooldown,
        )

        return next_state, jnp.array(0.0), jnp.array(terminated), {}

    @staticmethod
    def _wrap_angle(angle: jax.Array) -> jax.Array:
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def _handle_pickup_object(
        self,
        position: jax.Array,
        object_positions: jax.Array,
        object_available: jax.Array,
        carrying_object_idx: jax.Array,
        pickup_cooldown: jax.Array,
        params: WarehouseParams,
    ) -> tuple[jax.Array, jax.Array]:
        """Logic for handling pickup of an object (vase or crate).

        Returns:
            new_carrying_object_idx: jax.Array
            new_object_available: jax.Array
        """
        object_dists = jnp.linalg.norm(object_positions - position, axis=1)
        closest_object_idx = jnp.argmin(object_dists)
        closest_object_dist = object_dists[closest_object_idx]

        is_carrying_object = carrying_object_idx != -1
        is_on_cooldown = pickup_cooldown > 0
        pickup_success = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(
                    closest_object_dist < params.pickup_radius,
                    object_available[closest_object_idx],
                ),
                jnp.logical_not(is_carrying_object),
            ),
            jnp.logical_not(is_on_cooldown),
        )

        new_carrying_object_idx = jnp.where(
            pickup_success, closest_object_idx, carrying_object_idx
        )
        new_object_available = jax.lax.cond(
            pickup_success,
            lambda: object_available.at[closest_object_idx].set(False),
            lambda: object_available,
        )
        return new_carrying_object_idx, new_object_available

    def _handle_drop_object(
        self,
        position: jax.Array,
        object_positions: jax.Array,
        object_available: jax.Array,
        carrying_object_idx: jax.Array,
        current_cooldown: jax.Array,
        cooldown_steps: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Logic for handling drop of an object (vase or crate).

        Returns:
            new_carrying_object_idx: jax.Array
            new_object_positions: jax.Array
            new_object_available: jax.Array
            new_pickup_cooldown: jax.Array
        """
        drop_success = carrying_object_idx != -1
        new_object_positions = jax.lax.cond(
            drop_success,
            lambda: object_positions.at[carrying_object_idx].set(position),
            lambda: object_positions,
        )
        new_object_available = jax.lax.cond(
            drop_success,
            lambda: object_available.at[carrying_object_idx].set(True),
            lambda: object_available,
        )
        new_carrying_object_idx = jnp.where(drop_success, -1, carrying_object_idx)
        new_pickup_cooldown = jnp.where(drop_success, cooldown_steps, current_cooldown)
        return (
            new_carrying_object_idx,
            new_object_positions,
            new_object_available,
            new_pickup_cooldown,
        )

    @override
    def _compute_obs(self, state: EnvState, params: WarehouseParams) -> ObsFeatures:
        compute_lidars = jax.vmap(
            self._compute_lidar_for_objects,
            in_axes=(None, None, 0, 0, 0),
        )
        lidar = compute_lidars(
            state,
            params,
            jnp.stack([state.vase_positions, state.crate_positions]),
            jnp.array([state.carrying_vase_idx, state.carrying_crate_idx]),
            jnp.stack([state.vase_available, state.crate_available]),
        )

        carrying_status = jnp.array(
            [state.carrying_vase_idx != -1, state.carrying_crate_idx != -1],
            dtype=jnp.float32,
        )
        region_vecs = self._compute_region_vectors(state, params)

        c, s = jnp.cos(state.angle), jnp.sin(state.angle)
        rot = jnp.array([[c, s], [-s, c]])

        # Rotate and normalize acceleration and velocity
        acceleration = jnp.dot(rot, state.acceleration) / params.max_force
        velocity = jnp.dot(rot, state.velocity) / self._get_true_max_speed(params)
        angular_velocity = state.angular_velocity / params.max_angular_velocity
        # Rotate region vectors
        region_vecs = jnp.dot(region_vecs, rot.T) / params.world_size

        return ObsFeatures(
            acceleration=acceleration,
            velocity=velocity,
            angular_velocity=angular_velocity.reshape(1),
            global_position=state.position,
            lidar=lidar,
            region_vectors=region_vecs,
            carrying_status=carrying_status,
        )

    def _get_true_max_speed(self, params: WarehouseParams) -> jax.Array:
        """Calculate the true maximum speed considering drag."""
        steady_state_speed = (params.max_force * params.dt * (1.0 - params.drag)) / (
            params.drag + _EPS
        )
        return jnp.minimum(steady_state_speed, params.max_speed)

    def _compute_lidar_for_objects(
        self,
        state: EnvState,
        params: WarehouseParams,
        objects_pos: jax.Array,
        carrying_index: jax.Array,
        availability: jax.Array,
    ) -> jax.Array:
        pos = state.position
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])

        def object_sensor_contribution(obj_pos, is_avail):
            direction = obj_pos - pos
            dist = jnp.linalg.norm(direction)
            sensor = jnp.exp(-params.exp_gain * dist)
            sensor = sensor * is_avail
            direction = direction / (dist + _EPS)
            dotp = jnp.dot(heading, direction)
            cross = jnp.cross(heading, direction)
            angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
            bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)
            bin_angle = bin_size * bin_idx
            bins = jnp.zeros((params.num_lidar_bins,), dtype=jnp.float32)
            alias = (angle - bin_angle) / bin_size
            bins = bins.at[bin_idx].set(sensor)
            bins = bins.at[(bin_idx + 1) % params.num_lidar_bins].set(sensor * alias)
            bins = bins.at[(bin_idx - 1) % params.num_lidar_bins].set(
                sensor * (1.0 - alias)
            )
            return bins

        all_bins = jax.vmap(object_sensor_contribution, in_axes=0)(
            objects_pos, availability
        )
        bins = jnp.max(all_bins, axis=0)
        return jax.lax.cond(  # If carrying an object, lidar returns ones
            carrying_index != -1, lambda: jnp.ones_like(bins), lambda: bins
        )

    def _compute_region_vectors(
        self, state: EnvState, params: WarehouseParams
    ) -> jax.Array:
        regions = jnp.array(
            [
                params.region_a,
                params.region_b,
                params.door_region,
            ],
            dtype=jnp.float32,
        )  # shape: (3, 4)
        centers = regions[:, [0, 2]] + (regions[:, [1, 3]] - regions[:, [0, 2]]) / 2.0
        vectors = centers - state.position  # shape: (3, 2)
        return vectors

    @override
    def compute_propositions(
        self, state: EnvState, params: WarehouseParams
    ) -> jax.Array:
        """
        Returns ids of active propositions.
        Order: region_a, region_b, door, vase, crate
        """
        pos_in_regions = jax.vmap(self._pos_in_region, in_axes=(None, 0, None))
        in_regions = pos_in_regions(
            state.position,
            jnp.stack(
                [
                    jnp.array(params.region_a),
                    jnp.array(params.region_b),
                    jnp.array(params.door_region),
                ],
                dtype=jnp.float32,
            ),
            jnp.array(0.0),
        )
        in_a, in_b, at_door = in_regions[0], in_regions[1], in_regions[2]

        has_vase = state.carrying_vase_idx != -1
        has_crate = state.carrying_crate_idx != -1

        mask = jnp.array([in_a, in_b, at_door, has_vase, has_crate])
        prop_indices = jnp.arange(len(self.propositions))
        active_props = jnp.where(mask, prop_indices, -1)
        return jnp.sort(active_props, descending=True)

    def _pos_in_region(
        self,
        position: jax.Array,
        region: jax.Array,
        radius: jax.Array,
    ) -> jax.Array:
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = region
        return jnp.logical_and(
            jnp.logical_and(x + radius >= x_min, x - radius <= x_max),
            jnp.logical_and(y + radius >= y_min, y - radius <= y_max),
        )

    @staticmethod
    @override
    def assignments() -> list[Assignment]:
        regions = [
            frozenset({"region_a"}),
            frozenset({"region_b"}),
            frozenset({"door"}),
            frozenset(),
        ]

        items = [
            frozenset({"vase"}),
            frozenset({"crate"}),
            frozenset({"vase", "crate"}),
            frozenset(),
        ]

        assignments = []
        for r in regions:
            for i in items:
                assignments.append(Assignment(r | i))

        return assignments

    @override
    def get_renderer(
        self, env_params: WarehouseParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        from jaxltl.environments.warehouse_env.renderer import Renderer

        return Renderer(params=env_params, **kwargs)

    @override
    def plot_trajectories(
        self,
        trajs: EnvState,
        lengths: jax.Array,
        params: WarehouseParams,
        **plotting_kwargs,
    ) -> None:
        """Plots trajectories of environment states.

        Args:
            trajs: Batched EnvStates of shape (num_episodes, max_length, ...)
            lengths: Trajectory lengths (num_episodes,) int32
            params: Environment parameters
            plotting_kwargs: Additional keyword arguments for the plotting function
        """
        from jaxltl.environments.warehouse_env.plotter import draw_trajectories

        num_episodes = lengths.shape[0]

        # Extract positions for each trajectory up to its length
        positions = [
            trajs.position[i, : lengths[i] + 1].tolist() for i in range(num_episodes)
        ]

        # Extract initial object positions (from first timestep)
        initial_vase_positions = [
            trajs.vase_positions[i, 0].tolist() for i in range(num_episodes)
        ]
        initial_crate_positions = [
            trajs.crate_positions[i, 0].tolist() for i in range(num_episodes)
        ]

        # Extract carrying status for each trajectory
        carrying_vase_idx = [
            trajs.carrying_vase_idx[i, : lengths[i] + 1].tolist()
            for i in range(num_episodes)
        ]
        carrying_crate_idx = [
            trajs.carrying_crate_idx[i, : lengths[i] + 1].tolist()
            for i in range(num_episodes)
        ]

        draw_trajectories(
            positions=positions,
            initial_vase_positions=initial_vase_positions,
            initial_crate_positions=initial_crate_positions,
            carrying_vase_idx=carrying_vase_idx,
            carrying_crate_idx=carrying_crate_idx,
            region_a=params.region_a,
            region_b=params.region_b,
            door_region=params.door_region,
            world_size=params.world_size,
            pickup_radius=params.pickup_radius,
            **plotting_kwargs,
        )
