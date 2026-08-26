from collections.abc import Callable

import distrax
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxolotl.alg.sem_ltl.model.multi_epsilon_distribution import (
    MultiEpsilonDistribution,
)
from jaxolotl.rl.actor.discrete_actor import DiscreteActor


class MultiEpsilonDiscreteActor(DiscreteActor):
    def __init__(
        self,
        in_size: int,
        num_actions: int,
        hidden_sizes: list[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        super().__init__(
            in_size,
            num_actions,
            hidden_sizes,
            use_epsilon=True,
            activation=activation,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
        )

    def __call__(
        self, features_tuple: tuple[jax.Array, jax.Array], epsilon_mask: jax.Array
    ) -> distrax.Distribution:
        """Input shape: (batch_size, in_size) and (batch_size, max_eps_transitions, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        features, eps_features = features_tuple
        encoded = jax.vmap(self.encoder)(features)
        action_probs = jax.vmap(self.action_probs)(encoded)
        action_dist = distrax.Categorical(logits=action_probs)
        eps_encoded = jax.vmap(jax.vmap(self.encoder))(eps_features)
        log_eps = jax.vmap(jax.vmap(self.epsilon_prob))(eps_encoded)  # type: ignore
        stay_log_prob = jax.vmap(self.epsilon_prob)(encoded)  # type: ignore
        log_eps = jnp.concatenate([log_eps, stay_log_prob[:, None, :]], axis=1).squeeze(
            -1
        )
        epsilon_mask = jnp.concatenate(
            [
                epsilon_mask,
                jnp.ones((epsilon_mask.shape[0], 1), dtype=epsilon_mask.dtype),
            ],
            axis=1,
        )
        return MultiEpsilonDistribution(
            action_dist,
            log_eps,
            epsilon_mask,
            stay_index=jnp.array(log_eps.shape[1] - 1, dtype=jnp.int32),
        )
