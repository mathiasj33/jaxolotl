"""Sinusoidal positional encoding attention module for sequence encoding."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.networks.callable_module import CallableModule
from jaxolotl.networks.network_utils import make_linear


class PositionalAttention(CallableModule):
    """Single-head scaled dot-product attention with sinusoidal positional encodings.

    Uses standard transformer sinusoidal positional encodings added to input embeddings
    before projection.

    Reference: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    input_dim: int
    hidden_dim: int
    max_seq_len: int
    base: float

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        max_seq_len: int = 16,
        base: float = 100.0,
        *,
        key: jax.Array,
        **kwargs,
    ):
        """Initialize sinusoidal attention module.

        Args:
            input_dim: Dimension of input embeddings.
            hidden_dim: Dimension of query/key/value projections. Defaults to input_dim.
            max_seq_len: Maximum sequence length. Default 16 (suitable for sequences 1-10).
            base: Base for the sinusoidal frequency calculation. Default 100.0, chosen
                for better discrimination in short sequences (1-10). Smaller values
                create higher frequency variations across positions.
            key: PRNG key for initialization.
        """
        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.base = base

        keys = jax.random.split(key, 5)
        self.query_proj = make_linear(
            input_dim, hidden_dim, jax.nn.initializers.orthogonal(), None, key=keys[0]
        )
        self.key_proj = make_linear(
            input_dim, hidden_dim, jax.nn.initializers.orthogonal(), None, key=keys[1]
        )
        self.value_proj = make_linear(
            input_dim, hidden_dim, jax.nn.initializers.orthogonal(), None, key=keys[2]
        )
        self.output_proj = make_linear(
            hidden_dim, input_dim, jax.nn.initializers.orthogonal(), None, key=keys[3]
        )
        self.layer_norm = eqx.nn.LayerNorm(input_dim)

    def _compute_positional_encoding(self, seq_len: int, dim: int) -> jax.Array:
        """Compute sinusoidal positional encodings.

        Uses the standard transformer formula:
            PE(pos, 2i) = sin(pos / base^(2i/dim))
            PE(pos, 2i+1) = cos(pos / base^(2i/dim))

        Args:
            seq_len: Length of the sequence.
            dim: Dimension of the embeddings.

        Returns:
            Positional encoding matrix of shape (seq_len, dim).
        """
        positions = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
        dims = jnp.arange(dim)[None, :]  # (1, dim)

        # Compute frequencies: 1 / base^(2i/dim) for each dimension pair
        angles = positions / jnp.power(self.base, (2 * (dims // 2)) / dim)

        # Apply sin to even indices, cos to odd indices
        pe = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))

        return pe  # (seq_len, dim)

    def __call__(
        self,
        query: jax.Array,
        keys_values: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Apply attention with sinusoidal positional encodings.

        Args:
            query: Query vector of shape (input_dim,), typically the "current" step.
            keys_values: Key/value sequence of shape (seq_len, input_dim).
            mask: Optional boolean mask of shape (seq_len,). True = valid, False = masked.

        Returns:
            Output vector of shape (input_dim,) after attention and residual + layer norm.
        """
        seq_len = keys_values.shape[0]

        # Compute positional encodings
        pe = self._compute_positional_encoding(
            seq_len, self.input_dim
        )  # (seq_len, input_dim)

        # Add positional encodings to inputs
        keys_values_with_pe = keys_values + pe

        # Project query, keys, and values
        q = self.query_proj(query)  # (hidden_dim,)
        k = jax.vmap(self.key_proj)(keys_values_with_pe)  # (seq_len, hidden_dim)
        v = jax.vmap(self.value_proj)(keys_values_with_pe)  # (seq_len, hidden_dim)

        # Compute scaled dot-product attention scores
        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = (q @ k.T) * scale  # (seq_len,)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        # Softmax
        weights = jax.nn.softmax(scores)  # (seq_len,)

        # Weighted sum of values
        context = weights @ v  # (hidden_dim,)

        # Project output
        output = self.output_proj(context)  # (input_dim,)

        # Residual connection and layer norm (fuse attention output with query)
        return self.layer_norm(query + output)
