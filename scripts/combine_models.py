"""Script to combine models from different runs into a single batched model for evaluation."""

import logging
import os
from pathlib import Path

import equinox as eqx
import hydra
import jax

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.eval.utils import load_batched_models

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="combine_models")
def main(cfg):
    env, env_params = jaxolotl.make(cfg.env.name)
    path1 = Path(f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.runs[0]}/models.eqx")
    path2 = Path(f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.runs[1]}/models.eqx")
    model1, num_models1 = load_batched_models(
        cfg, env=env, env_params=env_params, key=jax.random.key(0), path=path1
    )
    logger.info(f"Loaded first model with {num_models1} seeds.")
    model2, num_models2 = load_batched_models(
        cfg, env=env, env_params=env_params, key=jax.random.key(0), path=path2
    )
    logger.info(f"Loaded second model with {num_models2} seeds.")

    params, static = eqx.partition(model1, eqx.is_array)
    params2, _ = eqx.partition(model2, eqx.is_array)
    combined_params = jax.tree.map(
        lambda p1, p2: jax.numpy.concatenate([p1, p2]), params, params2
    )
    combined_model = eqx.combine(static, combined_params)

    # rename path1 model
    bkp_path = path1.with_suffix(".bkp.eqx")
    os.rename(path1, bkp_path)
    logger.info(f"Backed up original model to {bkp_path}.")

    num_total_models = num_models1 + num_models2
    eqx_utils.save(path1, combined_model, metadata={"num_models": num_total_models})
    logger.info(f"Saved combined model with {num_total_models} seeds to {path1}.")


if __name__ == "__main__":
    main()
