import distrax
import jax
import jax.numpy as jnp

from jaxolotl.alg.sem_ltl.model.multi_epsilon_distribution import (
    MultiEpsilonDistribution,
)
from jaxolotl.rl.actor.composite_actor import CompositeActor


class MultiEpsilonCompositeActor(CompositeActor):
    """Hybrid continuous/discrete actor with multiple automaton epsilon actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_epsilon=True, **kwargs)

    def __call__(
        self, features_tuple: tuple[jax.Array, jax.Array], epsilon_mask: jax.Array
    ) -> MultiEpsilonDistribution:
        """Build a hybrid environment policy and a categorical epsilon policy."""
        features, eps_features = features_tuple
        encoded = jax.vmap(self.encoder)(features)

        mean = jax.vmap(self.cont_action_mean)(encoded)
        if self.cont_action_std is not None:
            std = jax.vmap(self.cont_action_std)(encoded)
            std = jnp.where(std >= 20, std, jax.nn.softplus(std))
        else:
            std = jnp.exp(self.log_std)[None, :].reshape(mean.shape)  # type: ignore
        std += 1e-3
        continuous_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

        discrete_logits = jax.vmap(self.disc_action_probs)(encoded)
        discrete_dist = distrax.Categorical(logits=discrete_logits)
        action_dist = distrax.Joint((continuous_dist, discrete_dist))

        assert self.epsilon_prob is not None
        eps_encoded = jax.vmap(jax.vmap(self.encoder))(eps_features)
        log_eps = jax.vmap(jax.vmap(self.epsilon_prob))(eps_encoded)
        stay_log_prob = jax.vmap(self.epsilon_prob)(encoded)
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
