import random

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxolotl.alg.sem_ltl.curriculum.conveyor_world_curriculum import (
    ConveyorFormulaSampler,
    chain_formula,
)
from jaxolotl.alg.struct_ltl.curriculum.conveyor_world_curriculum import (
    ConveyorSequenceSampler,
)
from jaxolotl.environments import registration
from jaxolotl.environments.conveyor_world.conveyor_world import ConveyorWorld, EnvState

_RIGHT, _DOWN, _LEFT, _UP = 0, 1, 2, 3


def _rollout_propositions(env: ConveyorWorld, actions: list[int]) -> list[str]:
    """Steps the env from reset and returns the propositions hit, in order."""
    params = env.default_params
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key, None, params)
    seen = []
    for action in actions:
        transition = env.step(key, state, jnp.array(action, dtype=jnp.int32), params)
        state = transition.state
        for idx in np.asarray(transition.propositions):
            if idx >= 0:
                seen.append(env.propositions[idx])
    return seen


def test_default_matches_legacy_layout():
    """chain_length=2 must reproduce the original ConveyorWorld layout."""
    env = ConveyorWorld()
    assert env.propositions == ("a", "b", "c")
    assert env._grid_width == 9

    legacy_walls = np.zeros((9, 9), dtype=bool)
    legacy_walls[0:5, 0] = True
    legacy_walls[0:5, 8] = True
    legacy_walls[4:9, 3:6] = True
    legacy_walls[3, 2:7] = True
    legacy_walls[3:6, 2] = True
    legacy_walls[3:6, 6] = True
    npt.assert_array_equal(np.asarray(env._get_wall_mask()), legacy_walls)

    # Legacy item positions: parcel -> a, wrench -> b, hammer -> c
    assert env.item_map() == {
        (7, 1): "a",
        (7, 7): "a",
        (8, 1): "b",
        (8, 7): "c",
    }


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_item_layout(chain_length: int):
    env = ConveyorWorld(chain_length=chain_length)
    assert len(env.propositions) == chain_length + 1
    items = env.item_map()

    width = env._grid_width
    shared = env.propositions[:-2]
    for i, prop in enumerate(shared):
        assert items[(7 + i, 1)] == prop
        assert items[(7 + i, 7)] == prop
    assert items[(width - 1, 1)] == env.propositions[-2]
    assert items[(width - 1, 7)] == env.propositions[-1]
    assert len(items) == 2 * len(shared) + 2

    # No item may sit on a wall or belt.
    walls = np.asarray(env._get_wall_mask())
    belts = np.asarray(env._get_belt_mask())
    for x, y in items:
        assert not walls[x, y]
        assert not belts[x, y]


def test_belt_blocks_leftward_movement():
    env = ConveyorWorld(chain_length=4)
    params = env.default_params
    key = jax.random.PRNGKey(0)

    # On a belt cell, moving left is blocked.
    state = EnvState(position=jnp.array([4, 1], dtype=jnp.int32))
    transition = env.step(key, state, jnp.array(_LEFT, dtype=jnp.int32), params)
    npt.assert_array_equal(np.asarray(transition.state.position), [4, 1])

    # Off the belt, moving left works.
    state = EnvState(position=jnp.array([7, 1], dtype=jnp.int32))
    transition = env.step(key, state, jnp.array(_LEFT, dtype=jnp.int32), params)
    npt.assert_array_equal(np.asarray(transition.state.position), [6, 1])


@pytest.mark.parametrize("chain_length", [1, 2, 3, 6])
def test_corridor_rollouts_hit_chain_in_order(chain_length: int):
    env = ConveyorWorld(chain_length=chain_length)
    width = env._grid_width
    shared = list(env.propositions[:-2])

    # Bottom corridor: down to y=1, then all the way right.
    bottom = _rollout_propositions(env, [_DOWN] * 3 + [_RIGHT] * (width - 1))
    assert bottom == [*shared, env.propositions[-2]]

    # Top corridor: up to y=7, then all the way right.
    top = _rollout_propositions(env, [_UP] * 3 + [_RIGHT] * (width - 1))
    assert top == [*shared, env.propositions[-1]]


def test_simplified_layout():
    env = ConveyorWorld(chain_length=3, simplified=True)
    assert env.propositions == ("a", "b", "c", "d")
    assert env._grid_width == 3
    assert env._grid_height == 9  # 2k + 3

    # Everything except the centre column is a wall.
    expected_free = np.zeros((3, 9), dtype=bool)
    expected_free[1, 1:8] = True
    npt.assert_array_equal(~np.asarray(env._get_wall_mask()), expected_free)

    # Length-k belts directly above and below the start (1, 4).
    expected_belts = np.zeros((3, 9), dtype=bool)
    expected_belts[1, 1:4] = True
    expected_belts[1, 5:8] = True
    npt.assert_array_equal(np.asarray(env._get_belt_mask()), expected_belts)

    # One chain item per belt cell; finals at the corridor ends.
    assert env.item_map() == {
        (1, 3): "a",
        (1, 5): "a",
        (1, 2): "b",
        (1, 6): "b",
        (1, 1): "c",
        (1, 7): "d",
    }


def test_simplified_belt_blocks_backing_out():
    env = ConveyorWorld(chain_length=3, simplified=True)
    params = env.default_params
    key = jax.random.PRNGKey(0)

    # In the top corridor (flow up), moving down is blocked.
    state = EnvState(position=jnp.array([1, 5], dtype=jnp.int32))
    transition = env.step(key, state, jnp.array(_DOWN, dtype=jnp.int32), params)
    npt.assert_array_equal(np.asarray(transition.state.position), [1, 5])

    # In the bottom corridor (flow down), moving up is blocked.
    state = EnvState(position=jnp.array([1, 3], dtype=jnp.int32))
    transition = env.step(key, state, jnp.array(_UP, dtype=jnp.int32), params)
    npt.assert_array_equal(np.asarray(transition.state.position), [1, 3])

    # Moving with the flow works.
    state = EnvState(position=jnp.array([1, 5], dtype=jnp.int32))
    transition = env.step(key, state, jnp.array(_UP, dtype=jnp.int32), params)
    npt.assert_array_equal(np.asarray(transition.state.position), [1, 6])


@pytest.mark.parametrize("chain_length", [1, 2, 3, 6])
def test_simplified_rollouts_hit_chain_in_order(chain_length: int):
    env = ConveyorWorld(chain_length=chain_length, simplified=True)
    shared = list(env.propositions[:-2])

    bottom = _rollout_propositions(env, [_DOWN] * chain_length)
    assert bottom == [*shared, env.propositions[-2]]

    top = _rollout_propositions(env, [_UP] * chain_length)
    assert top == [*shared, env.propositions[-1]]


def test_simplified_time_limit_scaling():
    # k <= 12 keeps the default 50-step limit for comparability.
    env = ConveyorWorld(chain_length=5, simplified=True)
    assert env.default_params.max_steps_in_episode == 50

    # Larger k scales to 4k so a committed random policy still finishes ~50%
    # of episodes; an explicit override wins.
    env = ConveyorWorld(chain_length=32, simplified=True)
    assert env.default_params.max_steps_in_episode == 128
    env = ConveyorWorld(chain_length=32, simplified=True, max_steps_in_episode=99)
    assert env.default_params.max_steps_in_episode == 99

    # The standard layout keeps the default regardless of k.
    env = ConveyorWorld(chain_length=10)
    assert env.default_params.max_steps_in_episode == 50


def test_multi_letter_propositions():
    env = ConveyorWorld(chain_length=32, simplified=True)
    assert len(env.propositions) == 33
    assert env.propositions[:3] == ("a", "b", "c")
    assert env.propositions[25:28] == ("z", "aa", "ab")
    assert env.propositions[-2:] == ("af", "ag")
    assert env._grid_height == 67

    # The corridors still deliver the chain in order at k=32.
    shared = list(env.propositions[:-2])
    bottom = _rollout_propositions(env, [_DOWN] * 32)
    assert bottom == [*shared, env.propositions[-2]]
    top = _rollout_propositions(env, [_UP] * 32)
    assert top == [*shared, env.propositions[-1]]


def test_simplified_registration():
    env, params = registration.make("ConveyorWorldSimple-4")
    assert isinstance(env, ConveyorWorld)
    assert params.chain_length == 4
    assert params.simplified
    npt.assert_array_equal(np.asarray(env._start_pos), [1, 5])


def test_registration():
    env, params = registration.make("ConveyorWorld-5")
    assert isinstance(env, ConveyorWorld)
    assert params.chain_length == 5
    assert env.propositions == ("a", "b", "c", "d", "e", "f")


def test_chain_formula():
    assert chain_formula(["a"]) == "F a"
    assert chain_formula(["a", "b"]) == "F (a & F b)"
    assert chain_formula(["a", "b", "c"]) == "F (a & F (b & F c))"


def test_samplers_split_between_finals():
    env = ConveyorWorld(chain_length=3)
    rng = random.Random(0)

    formula_sampler = ConveyorFormulaSampler(propositions=env.propositions)
    formulas = {formula_sampler.sample(rng) for _ in range(100)}
    assert formulas == {"F (a & F (b & F c))", "F (a & F (b & F d))"}

    sequence_sampler = ConveyorSequenceSampler(
        propositions=env.propositions, assignments=env.assignments()
    )
    chains = set()
    for _ in range(100):
        seq = sequence_sampler.sample(rng)
        reach = tuple(node.name for node, _ in seq.reach_avoid_formulas)
        avoid = tuple(avoid for _, avoid in seq.reach_avoid_formulas)
        assert avoid == (None,) * 3
        chains.add(reach)
    assert chains == {("a", "b", "c"), ("a", "b", "d")}
