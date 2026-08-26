import jax
import jax.numpy as jnp
from distrax import Categorical, Distribution


class MultiEpsilonDistribution(Distribution):
    """Models a policy that first decides whether to execute an epsilon action, and then
    samples from a different action distribution if not.

    In contrast to EpsilonDistribution, this distribution allows for multiple different
    epsilon actions to be taken.

    Note: this is not a valid distribution in the mathematical sense, since the sample
    function has to return a tuple of (action, eps_index). In the case that no epsilon
    action is taken, eps_index=max_num_epsilon_actions. The calling code must handle this
    appropriately.
    """

    def __init__(
        self,
        action_dist: Distribution,
        epsilon_log_probs: jax.Array,
        epsilon_mask: jax.Array,
        stay_index: jax.Array,
    ):
        """
        Args:
            action_dist: The distribution to sample from when not taking an epsilon action.
            epsilon_log_probs: The log probabilities of taking each epsilon action. Shape
                (max_num_epsilon_actions + 1,). An index of max_num_epsilon_actions indicates
                that no epsilon action is taken.
            epsilon_mask: A boolean mask indicating which (if any) epsilon actions are valid.
            stay_index: Index indicating no epsilon action.
        """
        self.action_dist = action_dist
        logits = jnp.where(epsilon_mask, epsilon_log_probs, -jnp.inf)
        self.epsilon_dist = Categorical(logits=logits)
        self.stay_index = stay_index

    def _sample_n(self, key: jax.Array, n: int):
        action_key, eps_key = jax.random.split(key)
        actions = self.action_dist._sample_n(action_key, n)
        take_epsilon = self.epsilon_dist._sample_n(eps_key, n)
        return actions, take_epsilon

    def log_prob(self, value: tuple[jax.Array, jax.Array]):
        actions, take_epsilon = value
        return self.epsilon_dist.log_prob(take_epsilon) + (
            take_epsilon == self.stay_index
        ) * self.action_dist.log_prob(actions)

    def event_shape(self):
        return self.action_dist.event_shape, self.epsilon_dist.event_shape

    def _sample_n_and_log_prob(self, key, n):
        action_key, eps_key = jax.random.split(key)
        env_samples, env_log_prob = self.action_dist._sample_n_and_log_prob(
            action_key, n
        )
        eps_samples, eps_log_prob = self.epsilon_dist._sample_n_and_log_prob(eps_key, n)
        log_prob = eps_log_prob + (eps_samples == self.stay_index) * env_log_prob
        return (env_samples, eps_samples), log_prob

    def entropy(self):
        eps_entropy = self.epsilon_dist.entropy()
        not_eps_prob = self.epsilon_dist.prob(self.stay_index)
        return eps_entropy + not_eps_prob * self.action_dist.entropy()

    def mode(self):
        return self.action_dist.mode(), self.epsilon_dist.mode()
