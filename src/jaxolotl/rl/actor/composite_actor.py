from collections.abc import Callable

import distrax
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxolotl.alg.common.epsilon_distribution import EpsilonDistribution
from jaxolotl.networks.mlp import MLP
from jaxolotl.rl.actor.actor import Actor


class CompositeActor(Actor):
    """
    An actor for mixed discrete and continuous action spaces. Assumes that any discrete
    action can be taken together with any continuous action.
    """

    encoder: MLP
    cont_action_mean: MLP
    cont_action_std: MLP | None
    log_std: jax.Array | None
    disc_action_probs: MLP
    epsilon_prob: MLP | None
    use_epsilon: bool

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        in_size: int,
        continuous_action_dim: int,
        num_discrete_actions: int,
        hidden_sizes: list[int],
        use_epsilon: bool,
        state_dependent_std: bool = True,
        hidden_activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        output_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        enc_key, mean_key, std_key, disc_key, eps_key = jax.random.split(key, 5)
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
        self.cont_action_mean = MLP(
            hidden_sizes[-1],
            continuous_action_dim,
            [],
            output_activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=mean_key,
        )
        if state_dependent_std:
            self.cont_action_std = MLP(
                hidden_sizes[-1],
                continuous_action_dim,
                [],
                output_activation,
                weight_init,
                bias_init,
                final_layer_activation=True,
                key=std_key,
            )
            self.log_std = None
        else:
            self.log_std = jnp.zeros((continuous_action_dim,))
            self.cont_action_std = None
        self.disc_action_probs = MLP(
            hidden_sizes[-1],
            num_discrete_actions,
            [],
            weight_init=weight_init,
            bias_init=bias_init,
            final_layer_activation=False,
            key=disc_key,
        )
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
        mean = jax.vmap(self.cont_action_mean)(encoded)
        if self.cont_action_std is not None:
            std = jax.vmap(self.cont_action_std)(encoded)
            # numerical stability for large std
            std = jnp.where(std >= 20, std, jax.nn.softplus(std))
        else:
            std = jnp.exp(self.log_std)[None, :].reshape(mean.shape)  # type: ignore
        std += 1e-3  # numerical stability
        cont_action_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

        log_probs = jax.vmap(self.disc_action_probs)(encoded)
        disc_action_dist = distrax.Categorical(logits=log_probs)

        joint_dist = distrax.Joint((cont_action_dist, disc_action_dist))

        if self.use_epsilon:
            assert epsilon_mask is not None
            log_eps = jax.vmap(self.epsilon_prob)(encoded)  # type: ignore
            return EpsilonDistribution(joint_dist, log_eps.squeeze(-1), epsilon_mask)
        return joint_dist
