import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from jaxolotl import eqx_utils
from jaxolotl.utils import artifact_utils


def _write_model(run_dir, seed: int, config_hash: str = "abc") -> None:
    path = artifact_utils.model_path(run_dir, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx_utils.save(
        path,
        jnp.asarray(seed),
        metadata={"seed": seed, "config_hash": config_hash},
    )


def test_discover_seed_models_uses_numeric_seed_labels(tmp_path) -> None:
    _write_model(tmp_path, 10)
    _write_model(tmp_path, 2)
    (tmp_path / "models" / "unrelated.eqx").touch()

    models = artifact_utils.discover_seed_models(tmp_path)

    assert sorted(models) == [2, 10]
    assert models[10].name == "model_seed10.eqx"


def test_config_hash_ignores_seed_batch_but_not_experiment_changes() -> None:
    first = OmegaConf.create({"start_seed": 0, "num_seeds": 5, "lr": 1e-3})
    second = OmegaConf.create({"start_seed": 5, "num_seeds": 5, "lr": 1e-3})
    changed = OmegaConf.create({"start_seed": 5, "num_seeds": 5, "lr": 2e-3})

    assert artifact_utils.config_hash(first) == artifact_utils.config_hash(second)
    assert artifact_utils.config_hash(first) != artifact_utils.config_hash(changed)


def test_check_seed_batch_rejects_completed_seed_overlap(tmp_path) -> None:
    _write_model(tmp_path, 4)

    with pytest.raises(FileExistsError, match=r"completed seed\(s\) \[4\]"):
        artifact_utils.verify_run_dir(tmp_path, [4, 5], expected_config_hash="abc")


def test_check_seed_batch_rejects_configuration_drift(tmp_path) -> None:
    _write_model(tmp_path, 0, config_hash="old")

    with pytest.raises(ValueError, match="same experiment configuration"):
        artifact_utils.verify_run_dir(tmp_path, [1], expected_config_hash="new")
