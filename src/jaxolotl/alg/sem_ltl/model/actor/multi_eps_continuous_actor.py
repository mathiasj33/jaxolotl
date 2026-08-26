from collections.abc import Callable

import distrax
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxolotl.alg.sem_ltl.model.multi_epsilon_distribution import (
    MultiEpsilonDistribution,
)
from jaxolotl.rl.actor.continuous_actor import ContinuousActor


class MultiEpsilonContinuousActor(ContinuousActor):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        in_size: int,
        action_dim: int,
        hidden_sizes: list[int],
        state_dependent_std: bool = True,
        hidden_activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        output_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),  # noqa
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        super().__init__(
            in_size,
            action_dim,
            hidden_sizes,
            use_epsilon=True,
            state_dependent_std=state_dependent_std,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
        )

    def __call__(
        self, features_tuple: tuple[jax.Array, jax.Array], epsilon_mask: jax.Array
    ) -> MultiEpsilonDistribution:
        """Input shape: (batch_size, in_size) and (batch_size, max_eps_transitions, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        features, eps_features = features_tuple
        encoded = jax.vmap(self.encoder)(features)
        mean = jax.vmap(self.action_mean)(encoded)
        if self.action_std is not None:
            std = jax.vmap(self.action_std)(encoded)
            # numerical stability for large std
            std = jnp.where(std >= 20, std, jax.nn.softplus(std))
        else:
            std = jnp.exp(self.log_std)[None, :].reshape(mean.shape)  # type: ignore
        std += 1e-3  # numerical stability
        eps_encoded = jax.vmap(jax.vmap(self.encoder))(eps_features)
        log_eps = jax.vmap(jax.vmap(self.epsilon_prob))(eps_encoded)  # type: ignore
        stay_log_prob = jax.vmap(self.epsilon_prob)(encoded)  # type: ignore
        action_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
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
