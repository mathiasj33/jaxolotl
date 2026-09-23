from pathlib import Path

import jax
from hydra import compose, initialize_config_dir

import jaxolotl
from scripts.train import build_model


def test_representative_experiment_models_can_be_constructed():
    configurations = (
        ("deep_ltl", "letter_world"),
        ("deep_ltl", "zone_env"),
        ("struct_ltl", "zone_env"),
        ("gcn_ltl", "warehouse"),
        ("tokenized_ltl", "warehouse"),
        ("ltl2action", "letter_world"),
        ("genz_ltl", "zone_env"),
        ("genz_ltl_flat", "warehouse"),
    )
    config_dir = Path(__file__).parents[3] / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        for alg, env_name in configurations:
            config = compose(
                config_name="train", overrides=[f"alg={alg}", f"env={env_name}"]
            )
            env, env_params = jaxolotl.make(config.env.name)
            model = build_model(
                config.model,
                env.observation_spec(env_params),
                env.action_space(env_params),
                len(env.assignments()),
                len(env.propositions),
                env_params,
                jax.random.key(0),
            )

            assert model.env_net.output_size > 0  # type: ignore
