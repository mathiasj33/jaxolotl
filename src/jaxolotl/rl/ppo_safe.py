"""Safety-constrained PPO (based on GenZ-LTL implementation)."""

import math
from collections.abc import Callable
from typing import NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.experimental import io_callback
from jaxtyping import PyTree

from jaxolotl import eqx_utils
from jaxolotl.alg.curriculum.curriculum_manager import (
    CurriculumManager,
    GlobalCurriculumState,
)
from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eqx_utils.training import TrainState
from jaxolotl.rl.actor_critic import ActorCritic
from jaxolotl.rl.algorithm import RLAlgorithm
from jaxolotl.rl.utils import stagger_initial_episodes


class PPOSafeConfig(NamedTuple):
    total_timesteps: int
    num_envs: int
    num_steps: int
    num_minibatches: int
    update_epochs: int
    gamma: float
    cost_gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    cost_vf_coef: float
    lag_coef: float
    lr: float
    max_grad_norm: float
    anneal_lr: bool
    adam_eps: float
    target_cost: float
    min_lag: float
    max_lag: float
    stagger_initial_episodes: bool = True


class PPOSafeTransition(NamedTuple):
    """Transition data collected during rollout."""

    terminated: jax.Array
    truncated: jax.Array
    action: jax.Array
    value: jax.Array
    cost_value: jax.Array
    reward: jax.Array
    cost: jax.Array
    log_prob: jax.Array
    obs: PyTree
    terminal_obs: PyTree
    info: PyTree


class PPOSafe(RLAlgorithm):
    config: PPOSafeConfig
    curriculum_manager: CurriculumManager

    def __init__(self, curriculum_manager: CurriculumManager, **kwargs):
        self.config = PPOSafeConfig(**kwargs)
        if (
            self.config.num_envs * self.config.num_steps
        ) % self.config.num_minibatches != 0:
            raise ValueError(
                "num_envs * num_steps must be divisible by num_minibatches"
            )
        self.curriculum_manager = curriculum_manager

    @override
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def train(
        self,
        model: ActorCritic,
        env: Environment | EnvWrapper,
        env_params: EnvParams,
        key: jax.Array,
        callback: Callable | None = None,
        callback_freq: int | None = None,
        seed: jax.Array | None = None,
    ) -> ActorCritic:
        optim = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(
                learning_rate=(
                    self.linear_schedule if self.config.anneal_lr else self.config.lr
                ),
                eps=self.config.adam_eps,
            ),
        )
        train_state = TrainState.create(model, optim)
        curriculum_state = self.curriculum_manager.init_state()

        key, reset_key, stagger_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, self.config.num_envs)
        env_state, obsv = env.reset(reset_keys, None, env_params, None)
        if self.config.stagger_initial_episodes:
            env_state = stagger_initial_episodes(
                env_state, stagger_key, env_params.max_steps_in_episode
            )

        num_steps_per_update = self.config.num_envs * self.config.num_steps
        num_updates = self.config.total_timesteps // num_steps_per_update
        if callback_freq is not None:
            if callback is None:
                raise ValueError("callback_freq is set but callback is None")
            updates_per_callback = math.ceil(callback_freq / num_steps_per_update)
            num_callbacks = math.ceil(num_updates / updates_per_callback)
        else:
            updates_per_callback = num_updates
            num_callbacks = 1

        def callback_iter(carry, _):
            def step(carry, _):
                (
                    train_state,
                    obsv,
                    env_state,
                    curriculum_state,
                    key,
                    step_count,
                ) = carry
                key, step_key = jax.random.split(key)
                train_state, obsv, env_state, curriculum_state, metric = (
                    self._train_step(
                        train_state,
                        optim,
                        obsv,
                        env,
                        env_state,
                        env_params,
                        curriculum_state,
                        key=step_key,
                    )
                )
                return (
                    train_state,
                    obsv,
                    env_state,
                    curriculum_state,
                    key,
                    step_count + 1,
                ), metric

            carry, metric = eqx_utils.filter_scan(
                step, carry, None, updates_per_callback
            )
            if callback:
                train_state, step_count = carry[0], carry[5]
                total_step = step_count * self.config.num_envs * self.config.num_steps
                params, _ = eqx.partition(train_state.model, eqx.is_array)
                io_callback(callback, None, metric, params, seed, total_step)
            return carry, None

        key, update_key = jax.random.split(key)
        carry = (
            train_state,
            obsv,
            env_state,
            curriculum_state,
            update_key,
            jnp.zeros((), jnp.int32),
        )
        carry, _ = eqx_utils.filter_scan(callback_iter, carry, None, num_callbacks)
        return carry[0].model

    def linear_schedule(self, count):
        frac = (
            1.0
            - (count // (self.config.num_minibatches * self.config.update_epochs))
            / self.config.total_timesteps
        )
        return self.config.lr * frac

    def _train_step(
        self,
        train_state,
        optim,
        obsv,
        env,
        env_state,
        env_params,
        curriculum_state: GlobalCurriculumState,
        *,
        key,
    ):
        key, rollout_key = jax.random.split(key)
        trajs, last_obs, env_state = self._rollout(
            train_state.model,
            obsv,
            env,
            env_state,
            env_params,
            key=rollout_key,
        )

        key, curriculum_key = jax.random.split(key)
        curriculum_state, env_state = self.curriculum_manager.update_progress(
            curriculum_state,
            env_state,
            trajs.terminated | trajs.truncated,
            trajs.info,
            curriculum_key,
        )

        advantages, targets, cost_advantages, cost_targets = self._calculate_gae(
            trajs, last_obs, train_state.model
        )

        def update_epoch(carry, _):
            train_state, key = carry
            key, shuffle_key = jax.random.split(key)
            minibatches = self._get_minibatches(
                trajs,
                advantages,
                targets,
                cost_advantages,
                cost_targets,
                shuffle_key,
            )

            def update_minibatch(train_state, minibatch):
                trajs, advantages, targets, cost_advantages, cost_targets = minibatch
                grad_fn = eqx.filter_value_and_grad(self._loss_fn, has_aux=True)
                losses, grads = grad_fn(
                    train_state.model,
                    trajs,
                    advantages,
                    targets,
                    cost_advantages,
                    cost_targets,
                )
                train_state = train_state.apply_gradients(optim, grads)
                return train_state, losses

            train_state, losses = eqx_utils.filter_scan(
                update_minibatch, train_state, minibatches
            )
            return (train_state, key), losses

        key, update_key = jax.random.split(key)
        (train_state, _), _ = eqx_utils.filter_scan(
            update_epoch, (train_state, update_key), None, self.config.update_epochs
        )
        metric = trajs.info
        return train_state, last_obs, env_state, curriculum_state, metric

    def _rollout(self, model, obsv, env, env_state, env_params, *, key):
        def env_step(carry, _):
            env_state, last_obs, key = carry
            key, sample_key = jax.random.split(key)
            pi, value = model(last_obs)
            cost_value = model.get_cost_value(last_obs)
            action, log_prob = pi.sample_and_log_prob(seed=sample_key)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.config.num_envs)
            transition = env.step(step_key, env_state, action, env_params)
            reward = transition.reward
            cost = transition.info["cost"]
            ppo_transition = PPOSafeTransition(
                transition.terminated,
                transition.truncated,
                action,
                value,
                cost_value,
                reward,
                cost,
                log_prob,
                last_obs,
                transition.terminal_observation,
                transition.info,
            )
            return (transition.state, transition.observation, key), ppo_transition

        carry = (env_state, obsv, key)
        carry, trajs = jax.lax.scan(env_step, carry, None, self.config.num_steps)
        return trajs, carry[1], carry[0]

    def _calculate_gae(self, trajs, last_obs, model):
        def get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            next_value = jax.lax.select(
                transition.truncated,
                model.get_value(transition.terminal_obs),
                next_value,
            )
            terminated, value, reward = (
                transition.terminated,
                transition.value,
                transition.reward,
            )
            not_terminated = 1.0 - terminated.astype(jnp.float32)
            not_done = 1.0 - jnp.logical_or(
                terminated, transition.truncated
            ).astype(jnp.float32)
            delta = reward + self.config.gamma * next_value * not_terminated - value
            gae = delta + self.config.gamma * self.config.gae_lambda * not_done * gae
            return (gae, value), gae

        def get_cost_advantages(gae_and_next_value, transition):
            gae, next_cost_value = gae_and_next_value
            next_cost_value = jax.lax.select(
                transition.truncated,
                model.get_cost_value(transition.terminal_obs),
                next_cost_value,
            )
            not_terminated = 1.0 - transition.terminated.astype(jnp.float32)
            not_done = 1.0 - jnp.logical_or(
                transition.terminated, transition.truncated
            ).astype(jnp.float32)
            cost_delta = (
                (1.0 - self.config.cost_gamma) * transition.cost
                + self.config.cost_gamma
                * jnp.maximum(transition.cost, next_cost_value)
                * not_terminated
                - transition.cost_value
            )
            gae = (
                cost_delta
                + self.config.cost_gamma * self.config.gae_lambda * not_done * gae
            )
            return (gae, transition.cost_value), gae

        last_val = jax.lax.select(
            trajs.truncated[-1],
            model.get_value(jax.tree.map(lambda x: x[-1], trajs.terminal_obs)),
            model.get_value(last_obs),
        )
        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajs,
            reverse=True,
            unroll=16,
        )

        last_cost_val = jax.lax.select(
            trajs.truncated[-1],
            model.get_cost_value(jax.tree.map(lambda x: x[-1], trajs.terminal_obs)),
            model.get_cost_value(last_obs),
        )
        _, cost_advantages = jax.lax.scan(
            get_cost_advantages,
            (jnp.zeros_like(last_cost_val), last_cost_val),
            trajs,
            reverse=True,
            unroll=16,
        )

        # Max-style cost return backup used by GenZ-LTL reference implementation.
        def get_cost_return(next_cost_return, transition):
            not_done = 1.0 - jnp.logical_or(
                transition.terminated, transition.truncated
            ).astype(jnp.float32)
            cost_return = jnp.maximum(
                transition.cost,
                next_cost_return * not_done,
            )
            return cost_return, cost_return

        init_cost_return = jnp.zeros_like(last_cost_val)
        _, cost_returns = jax.lax.scan(
            get_cost_return,
            init_cost_return,
            trajs,
            reverse=True,
            unroll=16,
        )

        return (
            advantages,
            advantages + trajs.value,
            cost_advantages,
            cost_returns,
        )

    def _get_minibatches(
        self, trajs, advantages, targets, cost_advantages, cost_targets, key
    ):
        num_transitions = self.config.num_steps * self.config.num_envs
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_transitions)
        data = (trajs, advantages, targets, cost_advantages, cost_targets)
        data = jax.tree.map(lambda x: x.reshape((num_transitions,) + x.shape[2:]), data)
        shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), data)
        minibatches = jax.tree.map(
            lambda x: x.reshape((self.config.num_minibatches, -1) + x.shape[1:]),
            shuffled,
        )
        return minibatches

    def _loss_fn(self, model, trajs, gae, targets, cost_gae, cost_targets):
        pi, value = model(trajs.obs)
        cost_value = model.get_cost_value(trajs.obs)
        lag = jnp.clip(
            model.get_lagrangian(trajs.obs), self.config.min_lag, self.config.max_lag
        )
        log_prob = pi.log_prob(trajs.action)

        value_pred_clipped = trajs.value + (value - trajs.value).clip(
            -self.config.clip_eps, self.config.clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        cost_value_pred_clipped = trajs.cost_value + (
            cost_value - trajs.cost_value
        ).clip(-self.config.clip_eps, self.config.clip_eps)
        cost_value_losses = jnp.square(cost_value - cost_targets)
        cost_value_losses_clipped = jnp.square(cost_value_pred_clipped - cost_targets)
        cost_value_loss = (
            0.5 * jnp.maximum(cost_value_losses, cost_value_losses_clipped).mean()
        )

        ratio = jnp.exp(log_prob - trajs.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        rew1 = ratio * gae
        rew2 = (
            jnp.clip(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
            * gae
        )
        policy_loss_reward = jnp.minimum(rew1, rew2)

        policy_loss_cost = ratio * cost_gae
        lag_detached = jax.lax.stop_gradient(lag)
        policy_loss = (
            -policy_loss_reward
            + lag_detached
            * (
                policy_loss_cost
                + (1.0 - self.config.cost_gamma) * cost_targets
                - self.config.target_cost
            )
        ).mean()

        lag_loss = -(
            lag
            * (
                jax.lax.stop_gradient(policy_loss_cost)
                + (1.0 - self.config.cost_gamma) * cost_targets
                - self.config.target_cost
            )
        ).mean()

        entropy = pi.entropy().mean()
        total_loss = (
            policy_loss
            + self.config.vf_coef * value_loss
            + self.config.cost_vf_coef * cost_value_loss
            + self.config.lag_coef * lag_loss
            - self.config.ent_coef * entropy
        )
        return total_loss, (value_loss, cost_value_loss, lag_loss, policy_loss, entropy)
