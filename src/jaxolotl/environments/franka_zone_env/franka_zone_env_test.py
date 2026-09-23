"""Reset, labelling, registration, and MJX integration tests."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl import eqx_utils
from jaxolotl.environments import spaces
from jaxolotl.environments.franka_zone_env.franka_zone_env import (
    REACH_HIGH,
    REACH_LOW,
    EnvState,
    FrankaZoneEnv,
    ResetDescriptor,
    build_model,
)
from jaxolotl.environments.franka_zone_env.kinematics import (
    HOME_QPOS,
    compute_fk,
    pose_matrix,
    solve_ik_sweep,
)
from jaxolotl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)


def test_vendored_model_has_only_minimal_collision_geoms():
    model = build_model()
    colliding = {
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    }
    assert colliding == {
        "floor",
        "hand_capsule",
        "left_finger_pad",
        "right_finger_pad",
    }
    assert model.nu == 7
    assert model.nq == 9
    assert model.opt.timestep == 0.005


def test_assignments_are_singletons_in_zone_order_plus_empty():
    # The GenZ-LTL observation reduction indexes zone rows by assignment index,
    # so assignment i must be the singleton of proposition/zone i, empty last.
    env = FrankaZoneEnv(num_colors=5)
    assignments = env.assignments()
    assert [set(a) for a in assignments[:-1]] == [{p} for p in env.propositions]
    assert len(assignments[-1]) == 0


def test_compact_reset_is_reachable_non_overlapping_and_clear():
    env = FrankaZoneEnv(num_colors=16)
    params = env.default_params
    descriptor = env.sample_reset(jax.random.key(0), None, params)
    assert isinstance(descriptor, ResetDescriptor)
    assert len(jax.tree.leaves(descriptor)) == 2

    centers = np.asarray(descriptor.zone_centers)
    low = np.asarray(REACH_LOW) + params.zone_radius
    high = np.asarray(REACH_HIGH) - params.zone_radius
    assert np.all(centers >= low)
    assert np.all(centers <= high)
    pairwise = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    pairwise += np.eye(params.num_colors) * 10.0
    assert pairwise.min() >= params.zone_keepout - 1e-6
    initial = np.asarray(compute_fk(descriptor.arm_qpos))[:3, 3]
    assert np.linalg.norm(centers - initial, axis=-1).min() >= (
        params.initial_zone_keepout - 1e-6
    )

    canonical_rotation = compute_fk(jnp.asarray(HOME_QPOS))[:3, :3]
    for center in descriptor.zone_centers:
        _, reachable = solve_ik_sweep(
            pose_matrix(center, canonical_rotation), jnp.asarray(HOME_QPOS)
        )
        assert bool(reachable)


def test_materialize_reconstructs_data_without_banking_it():
    env = FrankaZoneEnv()
    params = env.default_params
    descriptor = env.sample_reset(jax.random.key(1), None, params)
    state = env.materialize(descriptor, params)
    assert isinstance(state, EnvState)
    assert state.data.qpos.shape == (9,)
    assert state.data.ctrl.shape == (7,)
    assert jnp.allclose(state.data.qpos[:7], descriptor.arm_qpos)
    assert jnp.allclose(state.data.qpos[7:], 0.04)
    assert jnp.allclose(state.zone_centers, descriptor.zone_centers)


def test_observation_and_distance_labelling_agree():
    env = FrankaZoneEnv()
    params = env.default_params
    descriptor = env.sample_reset(jax.random.key(2), None, params)
    grasp = compute_fk(descriptor.arm_qpos)[:3, 3]
    centers = descriptor.zone_centers.at[0].set(grasp)
    state = env.materialize(ResetDescriptor(descriptor.arm_qpos, centers), params)
    obs = env.compute_obs(state, params)
    propositions = env.compute_propositions(state, params)
    assert obs.features.arm.shape == (40,)
    assert obs.features.zones.shape == (4, 4)
    assert jnp.allclose(obs.features.zones[0], 0.0)
    assert int(propositions[0]) == 0
    assert jnp.all(propositions[1:] == -1)


def test_jitted_vmapped_reset_and_step():
    env = FrankaZoneEnv(num_colors=8)
    params = env.default_params
    keys = jax.random.split(jax.random.key(3), 4)
    descriptors = jax.vmap(env.sample_reset, in_axes=(0, None, None, None))(
        keys, None, params, None
    )
    states = jax.vmap(env.materialize, in_axes=(0, None))(descriptors, params)
    transitions = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        keys, states, jnp.zeros((4, 6), dtype=jnp.float32), params
    )
    jax.block_until_ready(transitions)
    assert transitions.state.data.qpos.shape == (4, 9)
    assert transitions.observation.features.zones.shape == (4, 8, 4)
    assert jnp.allclose(transitions.state.data.time, params.dt)
    assert jnp.all(transitions.reward == 0.0)
    assert jnp.all(~transitions.terminated)


def test_precomputed_wrapper_banks_only_descriptors(tmp_path):
    env = FrankaZoneEnv()
    params = env.default_params
    keys = jax.random.split(jax.random.key(4), 3)
    descriptors = jax.vmap(env.sample_reset, in_axes=(0, None, None, None))(
        keys, None, params, None
    )
    path = tmp_path / "franka_resets.eqx"
    eqx_utils.save(path, descriptors, metadata={"batch_dim": 3})

    wrapped = PrecomputedResetWrapper(env, params, path)
    state, obs = wrapped.reset(jax.random.key(5), None, params)

    assert isinstance(wrapped.reset_descriptors, ResetDescriptor)
    assert len(jax.tree.leaves(wrapped.reset_descriptors)) == 2
    assert isinstance(state, EnvState)
    assert state.data.qpos.shape == (9,)
    assert obs.features.zones.shape == (4, 4)


def test_discretized_actions_match_continuous_unit_steps():
    env = FrankaZoneEnv(discretize=True)
    params = env.default_params
    space = env.action_space(params)
    assert isinstance(space, spaces.Discrete)
    assert space.n == 12

    descriptor = env.sample_reset(jax.random.key(6), None, params)
    state = env.materialize(descriptor, params)
    key = jax.random.key(7)
    continuous_params = dataclasses.replace(params, discretize=False)
    # Index 2 = +z translation, index 9 = -x rotation.
    for index, continuous in [(2, jnp.eye(6)[2]), (9, -jnp.eye(6)[3])]:
        discrete = env.step(key, state, jnp.int32(index), params)
        reference = env.step(key, state, continuous, continuous_params)
        assert jnp.allclose(discrete.state.data.qpos, reference.state.data.qpos)
