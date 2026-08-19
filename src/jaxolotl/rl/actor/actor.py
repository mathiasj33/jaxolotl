from abc import abstractmethod

import distrax
import equinox as eqx
import jax
from jaxtyping import PyTree


class Actor(eqx.Module):
    """Abstract base class for actors."""

    @abstractmethod
    def __call__(self, features: jax.Array, obs: PyTree) -> distrax.Distribution:
        """Input shape: (batch_size, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        pass
