import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum import (
    SampleBatcher,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


class GoalBatcher(SampleBatcher[str, jax.Array]):
    @staticmethod
    def batch(
        samples: list[str],
        env: Environment | EnvWrapper,
        num_parallel: int = 1,
    ) -> jax.Array:
        del num_parallel
        prop_to_idx = {
            proposition: idx for idx, proposition in enumerate(env.propositions)
        }
        indices = [prop_to_idx[s] for s in samples]
        return jnp.array(indices, dtype=jnp.int32)
