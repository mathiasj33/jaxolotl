"""Script to precompute curriculum samples and save them to disk. Can be used
to speed up training startup."""

import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

import jaxolotl
from jaxolotl import DATA_DIR, eqx_utils
from jaxolotl.alg.curriculum import Curriculum

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    logger.info("Instantiating curriculum to generate samples...")
    start_time = time.time()

    env, _ = jaxolotl.make(cfg.env.name)
    curriculum: Curriculum = hydra.utils.call(cfg.curriculum, env, load_path=None)

    end_time = time.time()
    logger.info(f"Sample generation finished in {end_time - start_time:.2f} seconds.")

    # Save the samples
    save_dir = Path(DATA_DIR) / cfg.env.name / cfg.alg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "curriculum.eqx"

    logger.info(f"Saving precomputed samples to {save_path}")
    eqx_utils.save_with_treedef(save_path, curriculum.samples)
    logger.info("Save complete.")


if __name__ == "__main__":
    main()
