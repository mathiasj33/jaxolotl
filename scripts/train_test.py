from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.mark.parametrize(
    ("env", "alg", "expected_save_freq"),
    (
        ("conveyor_world", "deep_ltl", 1e5),
        ("letter_world", "struct_ltl", 5e5),
        ("zone_env", "struct_ltl", 5e5),
    ),
)
def test_experiment_overrides_default_save_freq(env, alg, expected_save_freq):
    config_dir = Path(__file__).parents[1] / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="train", overrides=[f"env={env}", f"alg={alg}"])

    assert config.save_freq == expected_save_freq
