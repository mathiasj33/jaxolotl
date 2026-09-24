import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.genz_ltl.model.observation_reduction import (
    FrankaZoneEnvObservationReduction,
)
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.environments.franka_zone_env.franka_zone_env import ObsFeatures
from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec

_NUM_COLORS = 4
_NUM_ASSIGNMENTS = _NUM_COLORS + 1

_ARM = jnp.arange(40, dtype=jnp.float32)
# Rows are [relative xyz, distance]; distances deliberately not in row order.
_ZONES = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.9],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [1.0, 1.0, 0.0, 0.4],
    ],
    dtype=jnp.float32,
)


def _subgoal(reach: int, avoid: list[int]) -> JaxReachAvoidSubgoal:
    avoid_padded = np.full(_NUM_ASSIGNMENTS, -1, dtype=np.int32)
    avoid_padded[: len(avoid)] = avoid
    reach_one_hot = np.zeros(_NUM_COLORS, dtype=np.int32)
    reach_one_hot[reach] = 1
    avoid_one_hot = np.zeros(_NUM_ASSIGNMENTS, dtype=np.int32)
    avoid_one_hot[avoid] = 1
    return JaxReachAvoidSubgoal(
        reach=jnp.asarray(reach, dtype=jnp.int32),
        avoid=jnp.asarray(avoid_padded),
        reach_one_hot=jnp.asarray(reach_one_hot),
        avoid_one_hot=jnp.asarray(avoid_one_hot),
    )


def test_avoid_rows_are_distance_sorted_with_padding_flagged_last() -> None:
    reduction = FrankaZoneEnvObservationReduction()
    features = ObsFeatures(arm=_ARM, zones=_ZONES)
    reduced = reduction(features, _subgoal(reach=2, avoid=[3, 0])).reduced

    expected_avoid = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.4, 1.0],  # zone 3: nearer avoid zone first
            [1.0, 0.0, 0.0, 0.9, 1.0],  # zone 0
            [0.0, 0.0, 0.0, 0.0, -1.0],  # padding
        ],
        dtype=jnp.float32,
    )
    expected = jnp.concatenate([_ARM, _ZONES[2], jnp.ravel(expected_avoid)])
    assert jnp.array_equal(reduced, expected)


def test_empty_avoid_set_reduces_to_all_padding() -> None:
    reduction = FrankaZoneEnvObservationReduction()
    features = ObsFeatures(arm=_ARM, zones=_ZONES)
    reduced = reduction(features, _subgoal(reach=1, avoid=[])).reduced

    avoid_obs = jnp.reshape(reduced[44:], (_NUM_COLORS - 1, 5))
    assert jnp.array_equal(avoid_obs[:, :4], jnp.zeros((_NUM_COLORS - 1, 4)))
    assert jnp.array_equal(avoid_obs[:, 4], -jnp.ones(_NUM_COLORS - 1))
    assert jnp.array_equal(reduced[40:44], _ZONES[1])


def test_output_spec_matches_reduced_shape() -> None:
    reduction = FrankaZoneEnvObservationReduction()
    input_spec = ObservationSpec(
        arm=ArraySpec((40,), jnp.float32),
        zones=ArraySpec((_NUM_COLORS, 4), jnp.float32),
    )
    spec = reduction.output_spec(
        input_spec,
        params=None,  # type: ignore
        num_assignments=_NUM_ASSIGNMENTS,
        num_propositions=_NUM_COLORS,
    )
    features = ObsFeatures(arm=_ARM, zones=_ZONES)
    reduced = reduction(features, _subgoal(reach=0, avoid=[1])).reduced
    assert spec["reduced"].shape == reduced.shape == (40 + 4 + 5 * (_NUM_COLORS - 1),)
