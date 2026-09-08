import random
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxolotl import eqx_utils
from jaxolotl.alg.curriculum.curriculum import (
    Curriculum,
    CurriculumStage,
    MultiRandomStage,
    RandomCurriculumStage,
    Sampler,
)


class Samples(eqx.Module):
    values: jax.Array


class Stage(CurriculumStage[None]):
    def sample(self, rng: random.Random) -> None:
        del rng


class RecordingBatcher:
    num_parallel: int | None = None

    @staticmethod
    def batch(samples, env, num_parallel=1):
        del env
        RecordingBatcher.num_parallel = num_parallel
        return Samples(jnp.arange(len(samples)))


class RandomSampler(Sampler[int]):
    def sample(self, rng: random.Random) -> int:
        return rng.randrange(1_000_000)


class ConstantSampler(Sampler[int]):
    def __init__(self, value: int):
        self.value = value

    def sample(self, rng: random.Random) -> int:
        del rng
        return self.value


class ValueBatcher:
    @staticmethod
    def batch(samples, env, num_parallel=1):
        del env, num_parallel
        return Samples(jnp.asarray(samples))


def _save_curriculum(path: Path, *, num_stages: int, num_samples: int) -> None:
    samples = Samples(jnp.zeros((num_stages, num_samples), dtype=jnp.int32))
    eqx_utils.save_with_template(
        path,
        samples,
        metadata={"num_stages": num_stages, "num_samples": num_samples},
    )


def test_loads_curriculum_with_matching_metadata(tmp_path):
    path = tmp_path / "curriculum.eqx"
    _save_curriculum(path, num_stages=2, num_samples=3)

    curriculum = Curriculum(
        stages=[Stage(0.5), Stage(None)],
        batcher=None,  # type: ignore
        env=None,  # type: ignore
        num_samples=3,
        load_path=path,
    )

    assert curriculum.samples.values.shape == (2, 3)


def test_forwards_num_parallel_to_batcher():
    Curriculum(
        stages=[Stage(None)],
        batcher=RecordingBatcher(),  # type: ignore
        env=None,  # type: ignore
        num_samples=2,
        num_parallel=3,
    )

    assert RecordingBatcher.num_parallel == 3


def test_nested_stage_selection_is_deterministic():
    stage = MultiRandomStage(
        stages=[
            RandomCurriculumStage(ConstantSampler(0), threshold=None),
            RandomCurriculumStage(ConstantSampler(1), threshold=None),
        ],
        probs=[0.25, 0.75],
        threshold=None,
    )
    first_rng = random.Random(42)
    second_rng = random.Random(42)

    first = [stage.sample(first_rng) for _ in range(20)]
    second = [stage.sample(second_rng) for _ in range(20)]

    assert first == second


@pytest.mark.parametrize(
    ("file_num_stages", "file_num_samples", "message"),
    [(1, 3, "num_stages"), (2, 4, "num_samples")],
)
def test_rejects_curriculum_with_mismatched_metadata(
    tmp_path, file_num_stages, file_num_samples, message
):
    path = tmp_path / "curriculum.eqx"
    _save_curriculum(
        path,
        num_stages=file_num_stages,
        num_samples=file_num_samples,
    )

    with pytest.raises(ValueError, match=message):
        Curriculum(
            stages=[Stage(0.5), Stage(None)],
            batcher=None,  # type: ignore
            env=None,  # type: ignore
            num_samples=3,
            load_path=path,
        )
