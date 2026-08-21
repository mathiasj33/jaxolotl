from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec
from jaxolotl.networks.observation_encoder import (
    FieldConvEncoder,
    FlattenMLPEncoder,
    flatten_observation,
)


class MixedObservation(NamedTuple):
    vector: jax.Array
    image: jax.Array


def _spec() -> ObservationSpec:
    return ObservationSpec(
        vector=ArraySpec((3,), jnp.float32),
        image=ArraySpec((4, 4, 2), jnp.int32),
    )


def test_flatten_mlp_encoder_accepts_structured_observations():
    encoder = FlattenMLPEncoder(
        _spec(), out_size=5, hidden_sizes=[8], key=jax.random.key(0)
    )
    observation = MixedObservation(
        vector=jnp.arange(3.0),
        image=jnp.ones((4, 4, 2), dtype=jnp.int32),
    )

    assert flatten_observation(observation).shape == (35,)
    assert encoder(observation).shape == (5,)
    assert encoder.output_size == 5


def test_flatten_mlp_encoder_with_field_filtering():
    encoder = FlattenMLPEncoder(
        _spec(), out_size=5, hidden_sizes=[8], key=jax.random.key(0), fields=["vector"]
    )
    observation = MixedObservation(
        vector=jnp.arange(3.0),
        image=jnp.ones((4, 4, 2), dtype=jnp.int32),
    )

    assert flatten_observation(observation, fields=["vector"]).shape == (3,)
    assert encoder(observation).shape == (5,)
    assert encoder.output_size == 5


def test_field_conv_encoder_selects_image_field():
    encoder = FieldConvEncoder(
        _spec(),
        channels=[4],
        kernel_size=(2, 2),
        field="image",
        key=jax.random.key(0),
    )
    observation = MixedObservation(
        vector=jnp.zeros(3),
        image=jnp.ones((4, 4, 2), dtype=jnp.int32),
    )

    assert encoder(observation).shape == (encoder.output_size,)


def test_parallel_encoder_fuses_independent_branches():
    config = OmegaConf.create(
        {
            "_target_": "jaxolotl.networks.observation_encoder.ParallelEncoder",
            "branches": [
                {
                    "_target_": "jaxolotl.networks.observation_encoder.FieldConvEncoder",
                    "_partial_": True,
                    "channels": [4],
                    "kernel_size": [2, 2],
                    "field": "image",
                },
                {
                    "_target_": "jaxolotl.networks.observation_encoder.FlattenMLPEncoder",
                    "_partial_": True,
                    "out_size": 5,
                    "hidden_sizes": [8],
                    "fields": ["vector"],
                },
            ],
            "fusion_hidden_sizes": [4],
            "fusion_out_size": 6,
        }
    )
    encoder = hydra.utils.instantiate(
        config,
        input_spec=_spec(),
        key=jax.random.key(2),
        _convert_="object",
    )
    observation = MixedObservation(
        vector=jnp.ones(3),
        image=jnp.ones((4, 4, 2)),
    )

    assert encoder(observation).shape == (6,)
    assert encoder.output_size == 6
