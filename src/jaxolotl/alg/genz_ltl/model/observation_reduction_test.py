import jax
import jax.numpy as jnp

from jaxolotl.alg.genz_ltl.model.observation_reduction import (
    LetterWorldObservationReduction,
)
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.environments.letter_world import letter_world


def _subgoal(reach: int, avoid: list[int]) -> JaxReachAvoidSubgoal:
    return JaxReachAvoidSubgoal(
        reach=jnp.array(reach, dtype=jnp.int32),
        avoid=jnp.array(avoid, dtype=jnp.int32),
        reach_one_hot=jnp.zeros(len(letter_world.LetterWorld.propositions)),
        avoid_one_hot=jnp.zeros(len(letter_world.LetterWorld.assignments())),
    )


def test_letter_world_observation_reduction() -> None:
    grid = jnp.zeros((3, 3, 13), dtype=jnp.int32)
    grid = grid.at[0, 0, 0].set(1)  # reach letter a
    grid = grid.at[0, 1, 1].set(1)  # avoid letter b
    grid = grid.at[0, 2, 2].set(1)  # avoid letter c
    grid = grid.at[1, 1, -1].set(1)  # agent

    result = LetterWorldObservationReduction()(
        letter_world.ObsFeatures(grid=grid),
        _subgoal(reach=0, avoid=[1, 2, -1]),
    )

    expected = jnp.array(
        [
            [[1.0], [0.5], [0.5]],
            [[0.0], [0.2], [0.0]],
            [[0.0], [0.0], [0.0]],
        ],
        dtype=jnp.float32,
    )
    assert result.reduced.dtype == jnp.float32
    assert jnp.array_equal(result.reduced, expected)


def test_letter_world_observation_reduction_overwrite_precedence() -> None:
    grid = jnp.zeros((1, 2, 13), dtype=jnp.int32)
    grid = grid.at[0, 0, 0].set(1)
    grid = grid.at[0, 0, 1].set(1)
    grid = grid.at[0, 0, -1].set(1)
    grid = grid.at[0, 1, 0].set(1)
    grid = grid.at[0, 1, 1].set(1)

    result = jax.jit(LetterWorldObservationReduction())(
        letter_world.ObsFeatures(grid=grid),
        _subgoal(reach=0, avoid=[1, -1]),
    )

    # Agent overwrites reach, which overwrites avoid.
    assert jnp.array_equal(
        result.reduced,
        jnp.array([[[0.2], [1.0]]], dtype=jnp.float32),
    )


def test_letter_world_observation_reduction_ignores_non_letter_assignments() -> None:
    grid = jnp.zeros((1, 1, 13), dtype=jnp.int32).at[0, 0, 0].set(1)
    empty_assignment = len(letter_world.LetterWorld.propositions)

    result = LetterWorldObservationReduction()(
        letter_world.ObsFeatures(grid=grid),
        _subgoal(reach=empty_assignment, avoid=[empty_assignment, -1]),
    )

    assert jnp.array_equal(result.reduced, jnp.zeros((1, 1, 1), dtype=jnp.float32))


def test_letter_world_observation_reduction_output_spec() -> None:
    env = letter_world.LetterWorld(grid_size=5)

    spec = LetterWorldObservationReduction().output_spec(
        env.observation_spec(env.default_params),
        env.default_params,
        num_assignments=len(env.assignments()),
        num_propositions=len(env.propositions),
    )

    assert spec["reduced"].shape == (5, 5, 1)
    assert spec["reduced"].dtype == jnp.float32
