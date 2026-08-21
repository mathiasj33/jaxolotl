from collections.abc import Callable

import distrax
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxolotl.networks.mlp import MLP
from jaxolotl.rl.actor.actor import Actor
from jaxolotl.rl.distributions import EpsilonDistribution


class ContinuousActor(Actor):
    encoder: MLP
    action_mean: MLP
    action_std: MLP | None
    log_std: jax.Array | None
    epsilon_prob: MLP | None
    use_epsilon: bool

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        in_size: int,
        action_dim: int,
        hidden_sizes: list[int],
        use_epsilon: bool,
        state_dependent_std: bool = True,
        hidden_activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        output_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),  # noqa
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        enc_key, mean_key, std_key, eps_key = jax.random.split(key, 4)
        self.encoder = MLP(
            in_size,
            hidden_sizes[-1],
            hidden_sizes[:-1],
            hidden_activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=enc_key,
        )
        self.action_mean = MLP(
            hidden_sizes[-1],
            action_dim,
            [],
            output_activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=mean_key,
        )
        if state_dependent_std:
            self.action_std = MLP(
                hidden_sizes[-1],
                action_dim,
                [],
                output_activation,
                weight_init,
                bias_init,
                final_layer_activation=True,
                key=std_key,
            )
            self.log_std = None
        else:
            self.log_std = jnp.zeros((action_dim,))
            self.action_std = None

        self.use_epsilon = use_epsilon
        if use_epsilon:
            self.epsilon_prob = MLP(
                hidden_sizes[-1],
                1,
                [],
                weight_init=weight_init,
                bias_init=bias_init,
                final_layer_activation=False,
                key=eps_key,
            )
        else:
            self.epsilon_prob = None

    def __call__(
        self, features: jax.Array, epsilon_mask: jax.Array | None
    ) -> distrax.Distribution:
        """Input shape: (batch_size, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        encoded = jax.vmap(self.encoder)(features)
        mean = jax.vmap(self.action_mean)(encoded)
        if self.action_std is not None:
            std = jax.vmap(self.action_std)(encoded)
            # numerical stability for large std
            std = jnp.where(std >= 20, std, jax.nn.softplus(std))
        else:
            std = jnp.exp(self.log_std)[None, :].reshape(mean.shape)  # type: ignore
        std += 1e-3  # numerical stability
        if self.use_epsilon:
            assert epsilon_mask is not None
            log_eps = jax.vmap(self.epsilon_prob)(encoded)  # type: ignore
        action_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        if self.use_epsilon:
            assert epsilon_mask is not None
            return EpsilonDistribution(action_dist, log_eps.squeeze(-1), epsilon_mask)
        else:
            return action_dist
