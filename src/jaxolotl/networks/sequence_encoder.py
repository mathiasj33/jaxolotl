"""Shared sequence aggregation for task-conditioned policy models."""

from typing import Literal, override

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from jaxolotl.networks.alibi_attention import ALiBiAttention
from jaxolotl.networks.callable_module import CallableModule
from jaxolotl.networks.gru_cell import GRUCell
from jaxolotl.networks.positional_attention import PositionalAttention


class SequenceEncoder(CallableModule):
    """Aggregate padded sequence embeddings using a reverse GRU or attention."""

    gru: GRUCell | None
    attention: ALiBiAttention | PositionalAttention | None
    mode: Literal["gru", "attention"]
    output_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: Literal["gru", "attention"],
        attention_config: DictConfig | None,
        *,
        key: jax.Array,
    ):
        self.mode = mode
        if mode == "gru":
            self.gru = GRUCell(input_size, hidden_size, key=key)
            self.attention = None
            self.output_size = hidden_size
        elif mode == "attention":
            if attention_config is None:
                raise ValueError("An attention configuration is required.")
            self.gru = None
            self.attention = hydra.utils.instantiate(
                attention_config, input_dim=input_size, key=key
            )
            self.output_size = input_size
        else:
            raise ValueError(f"Unsupported sequence encoder: {mode}")

    @override
    def __call__(self, embeddings: jax.Array, length: jax.Array) -> jax.Array:
        max_length = embeddings.shape[0]

        if self.mode == "attention":
            assert self.attention is not None
            mask = jnp.arange(max_length) < length
            return self.attention(embeddings[0], embeddings, mask)

        assert self.gru is not None
        gru = self.gru
        h0 = jnp.zeros((gru.hidden_size,))

        def step(carry: tuple[jax.Array, int], inputs: jax.Array):
            hidden, index = carry
            hidden = jax.lax.cond(
                index <= length, lambda: gru(inputs, hidden), lambda: hidden
            )
            return (hidden, index - 1), None

        (result, _), _ = jax.lax.scan(
            step, (h0, max_length), embeddings, reverse=True, unroll=8
        )
        return result
