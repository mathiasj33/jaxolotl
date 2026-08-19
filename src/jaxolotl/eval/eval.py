from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from jaxolotl import eqx_utils
from jaxolotl.environments.environment import EnvParams
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.eval.agent import Agent


class EvalState(NamedTuple):
    """State to keep track of during evaluation."""

    returns: jax.Array  # (num_episodes,)
    disc_returns: jax.Array  # (num_episodes,) discounted returns
    lengths: jax.Array  # (num_episodes,) lengths of episodes
    completed: jax.Array  # (num_episodes,) whether episode is completed
    num_violations: jax.Array  # scalar, number of violations across all episodes


class EvalResetOptions(NamedTuple):
    """Options for resetting the EvalWrapper environment."""

    task: PyTree  # task to evaluate


class Evaluator(eqx.Module):
    """Implements model evaluation. Runs num_episodes episodes in parallel and records
    returns, discounted returns, and lengths.

    Note: Assumes that the underlying environment is already vectorized.
    This is for efficiency reasons, since the ActorCritic model already expects batched inputs.
    """

    num_episodes: int  # Number of evaluation episodes. We run each episode in parallel.
    discount: float  # Discount factor for returns.
    return_trajs: bool  # Whether to return trajectories. May use a lot of memory.

    def __init__(
        self,
        num_episodes: int,
        discount: float,
        return_trajs: bool = False,
    ):
        self.num_episodes = num_episodes
        self.discount = discount
        self.return_trajs = return_trajs

    @eqx.filter_jit
    def eval(
        self,
        agent: Agent,
        deterministic: bool,
        env: VectorizeWrapper,
        env_params: EnvParams,
        task: PyTree,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, PyTree | None]:
        """Evaluate the model in parallel on the environment.

        Returns:
            returns: jax.Array of shape (num_episodes,) with returns
            disc_returns: jax.Array of shape (num_episodes,) with discounted returns
            lengths: jax.Array of shape (num_episodes,) with lengths of episodes
            trajs: PyTree of shape (num_episodes, max_length, ...) with env states or
                None if return_trajs is False
        """

        def rollout_cond(
            carry: tuple[
                Agent, WrapperState, PyTree, EvalState, PyTree, jax.Array, jax.Array
            ],
        ) -> jax.Array:
            eval_state = carry[3]
            return jnp.sum(eval_state.completed).astype(jnp.int32) < self.num_episodes

        def rollout_step(
            carry: tuple[
                Agent, WrapperState, PyTree, EvalState, PyTree, jax.Array, jax.Array
            ],
        ) -> tuple[
            Agent, WrapperState, PyTree, EvalState, PyTree, jax.Array, jax.Array
        ]:
            agent, env_state, obsv, eval_state, trajs, index, key = carry

            # select action
            pi = agent.get_action(obsv)
            if deterministic:
                action = pi.mode()
            else:
                key, sample_key = jax.random.split(key)
                action = pi.sample(seed=sample_key)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.num_episodes)
            transition = env.step(step_key, env_state, action, env_params)  # type: ignore

            # update agent
            agent = agent.update(
                transition.observation, transition.state, transition.propositions, env
            )

            # record trajectory
            if self.return_trajs:
                trajs = jax.tree.map(
                    lambda x, y: x.at[:, index + 1].set(y),
                    trajs,
                    transition.state.state,
                )

            # update metrics
            rewards = jax.nn.relu(transition.reward)  # binary success indicator
            rewards = jnp.where(eval_state.completed, 0, rewards)
            returns = eval_state.returns + rewards
            disc_returns = (
                eval_state.disc_returns
                + jnp.power(self.discount, index) * transition.reward
            )
            lengths = eval_state.lengths + jnp.where(eval_state.completed, 0, 1)
            is_sink: jax.Array = transition.info["is_sink"].astype(jnp.int32)
            violations = jnp.where(eval_state.completed, 0, is_sink)
            total_violations = eval_state.num_violations + jnp.sum(violations)

            # update completed
            completed = jnp.logical_or(eval_state.completed, transition.done)

            new_eval_state = EvalState(
                returns=returns,
                disc_returns=disc_returns,
                lengths=lengths,
                completed=completed,
                num_violations=total_violations,
            )
            return (
                agent,
                transition.state,
                transition.observation,
                new_eval_state,
                trajs,
                index + 1,
                key,
            )

        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.num_episodes)
        options = EvalResetOptions(task=task)
        env_state, obsv = env.reset(reset_keys, None, env_params, options)
        props: jax.Array = eqx.filter_vmap(env.compute_propositions)(
            env_state, env_params
        )
        agent = agent.update(obsv, env_state, props, env)
        max_length = env_params.max_steps_in_episode
        if self.return_trajs:
            trajs = jax.tree.map(
                lambda x: jnp.zeros(
                    (self.num_episodes, max_length + 1) + x.shape[1:], dtype=x.dtype
                ),
                env_state.state,
            )
            trajs = jax.tree.map(lambda x, y: x.at[:, 0].set(y), trajs, env_state.state)
        else:
            trajs = None
        index = jnp.zeros((), dtype=jnp.int32)
        state = EvalState(
            returns=jnp.zeros((self.num_episodes,), dtype=jnp.float32),
            disc_returns=jnp.zeros((self.num_episodes,), dtype=jnp.float32),
            lengths=jnp.zeros((self.num_episodes,), dtype=jnp.int32),
            completed=jnp.zeros((self.num_episodes,), dtype=bool),
            num_violations=jnp.zeros((), dtype=jnp.int32),
        )
        final = eqx_utils.filter_while_loop(
            rollout_cond,
            rollout_step,
            (agent, env_state, obsv, state, trajs, index, key),
        )
        _, _, _, state, trajs, _, _ = final
        return (
            state.returns,
            state.disc_returns,
            state.lengths,
            state.num_violations,
            trajs,
        )
