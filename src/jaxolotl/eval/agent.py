import distrax
import equinox as eqx
import jax

from jaxolotl.environments.environment import EnvObservation
from jaxolotl.environments.wrappers.wrapper import EnvWrapper, WrapperState
from jaxolotl.rl.actor_critic import ActorCritic


class Agent[TEvalState: WrapperState](eqx.Module):
    """Agent interface for evaluation. Implements a default agent that does not keep
    track of any state."""

    model: ActorCritic

    @classmethod
    def instantiate(cls, model: ActorCritic) -> "Agent":
        """Instantiates an agent. Can be overriden by subclasses to take additional arguments.

        Args:
            model: The model to be used by the agent.
        """
        return cls(model=model)

    def get_action(self, obsv: EnvObservation) -> distrax.Distribution:
        return self.model.get_action(obsv)

    def update(
        self, obsv: EnvObservation, state: TEvalState, props: jax.Array, env: EnvWrapper
    ) -> "Agent":
        """Update the agent's state based on the observation and environment propositions."""
        del obsv, state, props, env
        return self  # No state to update
