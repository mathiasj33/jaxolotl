from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxolotl import eqx_utils
from jaxolotl.alg.curriculum.curriculum import Curriculum, CurriculumStage


class Samples(eqx.Module):
    values: jax.Array


class Stage(CurriculumStage[None]):
    def sample(self) -> None:
        return None


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
