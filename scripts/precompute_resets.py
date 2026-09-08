"""Precompute environment resets for training and evaluation."""

import logging
import math
import time

import hydra
import jax
from jaxtyping import PyTree
from omegaconf import DictConfig

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.environments.reset import ResetSplit, precomputed_reset_path

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="precompute")
def main(cfg: DictConfig):
    # Train and test splits
    num_batch_resets = 2 * math.ceil(cfg.num_resets / cfg.num_envs)
    env, params = jaxolotl.make(cfg.env.name)
    vmap_sample_reset = jax.vmap(env.sample_reset, in_axes=(0, None, None, None))
    key = jax.random.key(42)

    @jax.jit
    def body(key, _):
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, cfg.num_envs)
        descriptors = vmap_sample_reset(subkeys, None, params, None)
        return key, descriptors

    start_time = time.time()
    _, descriptors = jax.lax.scan(body, key, None, length=num_batch_resets)
    jax.block_until_ready(descriptors)
    seconds = time.time() - start_time
    logger.info(
        f"Performed {num_batch_resets // 2} resets for train and test splits in {seconds:.2f} seconds"
    )

    # Reshape descriptors to (2, num_resets, ...)
    descriptors = jax.tree.map(lambda x: x.reshape(2, -1, *x.shape[2:]), descriptors)
    leaves = jax.tree.leaves(descriptors)
    nbytes = sum(x.nbytes for x in leaves)
    logger.info(f"Total data size: {nbytes / 2**20:.2f} MB")
    logger.info(
        "Descriptor leaves: %d; batch shape: %s", len(leaves), leaves[0].shape[1]
    )

    save_descriptors(cfg, jax.tree.map(lambda x: x[0], descriptors), "train")
    save_descriptors(cfg, jax.tree.map(lambda x: x[1], descriptors), "test")


def save_descriptors(cfg: DictConfig, descriptors: PyTree, split: ResetSplit):
    file = precomputed_reset_path(cfg.env.name, split)
    file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {file}")
    eqx_utils.save(
        file,
        descriptors,
        metadata={
            "batch_dim": jax.tree.leaves(descriptors)[0].shape[0],
            "env_name": cfg.env.name,
            "split": split,
        },
    )


if __name__ == "__main__":
    main()
