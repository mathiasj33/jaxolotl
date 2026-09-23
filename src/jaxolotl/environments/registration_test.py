import pytest

from jaxolotl.environments import spaces
from jaxolotl.environments.registration import make


@pytest.mark.parametrize("name", ["LetterWorld", "ConveyorWorld-3"])
def test_make_ignores_discretize_for_discrete_environments(name: str) -> None:
    env, params = make(name, discretize=True)

    assert isinstance(env.action_space(params), spaces.Discrete)
    assert not hasattr(params, "discretize")


@pytest.mark.parametrize("name", ["ZoneEnv", "FrankaZoneEnv-4"])
def test_make_passes_discretize_to_supported_environment(name: str) -> None:
    env, params = make(name, discretize=True)

    assert params.discretize  # type: ignore
    assert isinstance(env.action_space(params), spaces.Discrete)


def test_make_rejects_discretize_for_unsupported_environment() -> None:
    with pytest.raises(
        ValueError, match="Environment 'WarehouseEnv' does not support discretization"
    ):
        make("WarehouseEnv", discretize=True)


def test_make_allows_unsupported_environment_without_discretization() -> None:
    make("WarehouseEnv", discretize=False)
