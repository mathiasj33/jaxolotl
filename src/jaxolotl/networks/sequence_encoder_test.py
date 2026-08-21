import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from jaxolotl.networks.sequence_encoder import SequenceEncoder


def test_gru_sequence_encoder_has_declared_output_size():
    encoder = SequenceEncoder(
        input_size=4,
        hidden_size=6,
        mode="gru",
        attention_config=None,
        key=jax.random.key(0),
    )
    embeddings = jnp.ones((5, 4))

    assert encoder(embeddings, jnp.array(3)).shape == (6,)
    assert encoder.output_size == 6


def test_attention_sequence_encoder_has_input_sized_output():
    config = OmegaConf.create(
        {
            "_target_": "jaxolotl.networks.alibi_attention.ALiBiAttention",
            "hidden_dim": 8,
            "alibi_slope": 0.5,
        }
    )
    encoder = SequenceEncoder(
        input_size=4,
        hidden_size=6,
        mode="attention",
        attention_config=config,
        key=jax.random.key(0),
    )
    embeddings = jnp.ones((5, 4))

    assert encoder(embeddings, jnp.array(3)).shape == (4,)
    assert encoder.output_size == 4
