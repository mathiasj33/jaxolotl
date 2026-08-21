from pathlib import Path

import jax
from hydra import compose, initialize_config_dir

import jaxolotl
from scripts.train import build_model


def test_representative_experiment_models_can_be_constructed():
    experiments = (
        "deep_ltl/letter",
        "deep_ltl/zones",
        "struct_ltl/zones",
        "gcn_ltl/warehouse",
        "tokenized_ltl/warehouse",
        "ltl2action/letter",
        "genz_ltl/zones",
        "genz_ltl/warehouse",
    )
    config_dir = Path(__file__).parents[3] / "conf"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        for experiment in experiments:
            config = compose(
                config_name="train", overrides=[f"experiment={experiment}"]
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
