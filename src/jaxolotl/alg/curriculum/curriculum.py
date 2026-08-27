"""Generic implementation of a training curriculum as used in e.g. DeepLTL.

Curricula are defined using curriculum stages that sample generic tasks of type TSample.
In order to allow end-to-end compilation in JAX, the SampleBatcher is used to batch a
list of samples into a PyTree of type TJaxSample.

The entire curriculum is precomputed before training for efficiency reasons. For cases
where this pre-computation is expensive, the curriculum can be saved to disk and loaded
later (see scripts/precompute_curriculum.py).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from tqdm.auto import tqdm

from jaxolotl import eqx_utils
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper

logger = logging.getLogger(__name__)


class Sampler[TSample](ABC):
    """Abstract base class for samplers."""

    @abstractmethod
    def sample(self) -> TSample:
        """Return a random sample."""
        pass


class SampleBatcher[TSample, TJaxSample](ABC):
    """Abstract base class for sample batching functions."""

    @staticmethod
    @abstractmethod
    def batch(
        samples: list[TSample], env: Environment | EnvWrapper, num_parallel: int = 1
    ) -> TJaxSample:
        """Batch a list of samples into a JAX-compatible format."""
        pass


class CurriculumStage[TSample](ABC):
    """Abstract base class for curriculum stages."""

    def __init__(self, threshold: float | None):
        if threshold is None:
            threshold = jnp.inf
        self.threshold = threshold

    @abstractmethod
    def sample(self) -> TSample:
        pass


class RandomCurriculumStage[TSample](CurriculumStage[TSample]):
    """A curriculum stage in which tasks are sampled randomly."""

    def __init__(self, sampler: Sampler[TSample], threshold: float | None):
        super().__init__(threshold)
        self.sampler = sampler

    @override
    def sample(self) -> TSample:
        return self.sampler.sample()


class MultiRandomStage[TSample](CurriculumStage[TSample]):
    """A combination of multiple RandomCurriculumStages with associated sampling probabilities."""

    def __init__(
        self,
        stages: list[RandomCurriculumStage[TSample]],
        probs: list[float],
        threshold: float | None,
    ):
        super().__init__(threshold)
        self.stages = stages
        self.probs = np.array(probs, dtype=np.float32) / np.sum(
            np.array(probs, dtype=np.float32)
        )

    @override
    def sample(self) -> TSample:
        stage_idx = np.random.choice(len(self.stages), p=self.probs)
        stage = self.stages[stage_idx]
        return stage.sample()


class Curriculum[TSample, TJaxSample: eqx.Module](eqx.Module):
    """A curriculum consisting of multiple curriculum stages. Precomputes samples
    for each stage, enabling efficient sampling during training."""

    samples: PyTree  # batched samples for each stage
    thresholds: jax.Array
    num_samples: int

    def __init__(
        self,
        stages: list[CurriculumStage[TSample]],
        batcher: SampleBatcher[TSample, TJaxSample],
        env: Environment | EnvWrapper,
        num_samples: int,
        *,
        load_path: Path | None = None,
        num_parallel: int = 1,
    ):
        self.num_samples = num_samples
        self.thresholds = jnp.array([s.threshold for s in stages], dtype=jnp.float32)

        if load_path is None or not load_path.exists():
            samples_list = []
            for i, stage in enumerate(stages):
                samples = [
                    stage.sample()
                    for _ in tqdm(
                        range(num_samples),
                        desc=f"Precomputing curriculum samples for stage {i + 1} / {len(stages)}",
                    )
                ]
                samples_list.extend(samples)
            logger.info(
                f"Batching {len(samples_list)} samples for {len(stages)} stages..."
            )
            samples = batcher.batch(samples_list, env, num_parallel=num_parallel)
            logger.info("Finished batching curriculum samples.")
            self.samples = jax.tree.map(  # shape: (num_stages, num_samples, ...)
                lambda x: x.reshape(len(stages), -1, *x.shape[1:]),
                samples,
            )
        else:
            metadata = eqx_utils.load_metadata(load_path)
            expected_metadata = {
                "num_stages": len(stages),
                "num_samples": num_samples,
            }
            mismatches = [
                f"{key}: file has {metadata.get(key)!r}, expected {expected!r}"
                for key, expected in expected_metadata.items()
                if metadata.get(key) != expected
            ]
            if mismatches:
                raise ValueError(
                    f"Precomputed curriculum at {load_path} is incompatible with "
                    f"the current configuration ({'; '.join(mismatches)})."
                )
            self.samples = eqx_utils.load_from_template(load_path)
            logger.info(f"Loaded precomputed curriculum samples from {load_path}")

    @eqx.filter_jit
    def sample(self, stage: jax.Array, key: jax.Array) -> TJaxSample:
        index = jax.random.randint(key, (), 0, self.num_samples)
        return jax.tree.map(lambda x: x[stage, index], self.samples)

    @eqx.filter_jit
    def threshold(self, stage: jax.Array) -> jax.Array:
        return self.thresholds[stage]
