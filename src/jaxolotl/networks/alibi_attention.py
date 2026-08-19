"""ALiBi (Attention with Linear Biases) attention module for sequence encoding."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.networks.callable_module import CallableModule
from jaxolotl.networks.network_utils import make_linear


class ALiBiAttention(CallableModule):
    """Single-head scaled dot-product attention with ALiBi positional biases.

    ALiBi adds a linear bias based on distance between query and key positions,
    eliminating the need for explicit positional embeddings.

    Reference: https://arxiv.org/abs/2108.12409
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    input_dim: int
    hidden_dim: int
    alibi_slope: float

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        alibi_slope: float = 1.0,
        *,
        key: jax.Array,
    ):
        """Initialize ALiBi attention module.

        Args:
            input_dim: Dimension of input embeddings.
            hidden_dim: Dimension of query/key/value projections. Defaults to input_dim.
            alibi_slope: Slope for ALiBi linear bias. Controls how strongly distance
                affects attention weights.
            key: PRNG key for initialization.
        """
        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alibi_slope = alibi_slope

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

    def _compute_alibi_bias(self, seq_len: int) -> jax.Array:
        """Compute ALiBi bias matrix for given sequence length.

        Args:
            seq_len: Length of the sequence.

        Returns:
            Bias matrix of shape (seq_len,) representing distances from position 0.
        """
        # For our use case, we always query from position 0 (the "current" step)
        # So we compute distances from position 0 to all positions
        positions = jnp.arange(seq_len)
        # Negative bias for positions further away (positions > 0)
        return -self.alibi_slope * positions

    def __call__(
        self,
        query: jax.Array,
        keys_values: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Apply attention with ALiBi positional bias.

        Args:
            query: Query vector of shape (input_dim,), typically the "current" step.
            keys_values: Key/value sequence of shape (seq_len, input_dim).
            mask: Optional boolean mask of shape (seq_len,). True = valid, False = masked.

        Returns:
            Output vector of shape (input_dim,) after attention and residual + layer norm.
        """
        seq_len = keys_values.shape[0]

        # Project query, keys, and values
        q = self.query_proj(query)  # (hidden_dim,)
        k = jax.vmap(self.key_proj)(keys_values)  # (seq_len, hidden_dim)
        v = jax.vmap(self.value_proj)(keys_values)  # (seq_len, hidden_dim)

        # Compute scaled dot-product attention scores
        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = (q @ k.T) * scale  # (seq_len,)

        # Add ALiBi bias
        alibi_bias = self._compute_alibi_bias(seq_len)
        scores = scores + alibi_bias

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
