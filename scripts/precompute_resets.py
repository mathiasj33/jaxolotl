"""Script to precompute environment reset states and save them to disk. Can be used
with the PrecomputedResetWrapper to speed up training."""

import logging
import math
import time

import hydra
import jax
from omegaconf import DictConfig

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.environments.reset import RESET_SPLITS, precomputed_reset_path

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="precompute")
def main(cfg: DictConfig):
    if cfg.split not in RESET_SPLITS:
        raise ValueError(
            f"Unknown reset split {cfg.split!r}; expected one of {RESET_SPLITS}."
        )

    num_batch_resets = math.ceil(cfg.num_resets / cfg.rl_alg.num_envs)
    env, params = jaxolotl.make(cfg.env.name)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None, None))
    seed = {"train": 0, "test": 42}[cfg.split]
    key = jax.random.key(seed)

    @jax.jit
    def body(key, _):
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, cfg.rl_alg.num_envs)
        states, _ = vmap_reset(subkeys, None, params, None)
        return key, states

    start_time = time.time()
    _, states = jax.lax.scan(body, key, None, length=num_batch_resets)
    jax.block_until_ready(states)
    seconds = time.time() - start_time
    logger.info(f"Performed {cfg.num_resets} resets in {seconds:.2f} seconds")

    # Reshape states to (num_resets, ...)
    states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), states)
    nbytes = sum(x.nbytes for x in jax.tree.leaves(states))
    logger.info(f"Total data size: {nbytes / 2**20:.2f} MB")
    logger.info(f"Shape: {states.position.shape}")

    file = precomputed_reset_path(cfg.env.name, cfg.split)
    file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {file}")
    eqx_utils.save(
        file,
        states,
        metadata={
            "batch_dim": num_batch_resets * cfg.rl_alg.num_envs,
            "env_name": cfg.env.name,
            "split": cfg.split,
        },
    )


if __name__ == "__main__":
    main()
