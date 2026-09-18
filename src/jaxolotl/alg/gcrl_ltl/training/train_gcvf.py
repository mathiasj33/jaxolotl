"""Implementation of the GCVF training procedure.

We follow GCRL-LTL's official implementation, which trains the GCVF once for the final
policy. The implementation also differs from the original paper in that the "source"
goal is the goal that was actually reached at the end of the episode, and the "target"
goal is every other goal, for every state along the trajectory.
"""

import logging
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jaxolotl import eqx_utils
from jaxolotl.alg.gcrl_ltl.model.gcvf import GCVF
from jaxolotl.alg.gcrl_ltl.model.model import GCRLModel
from jaxolotl.alg.gcrl_ltl.wrappers.goal_wrapper import GoalObservation
from jaxolotl.environments.environment import Environment, EnvObservation, EnvParams
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eqx_utils.training import TrainState


class GCVFTrainingConfig(NamedTuple):
    # Training parameters
    num_samples: int
    batch_size: int
    lr: float
    epochs: int

    # Data collection parameters
    num_envs: int
    steps_per_env: int  # Number of steps to run per environment at each iteration


class GCVFDataset(NamedTuple):
    obsv: EnvObservation  # includes both features and goal. [num_samples, ...]
    target_goals: jax.Array  # [num_samples] target goal for each sample
    values: jax.Array  # [num_samples] target values for each sample


logger = logging.getLogger(__name__)


class GCVFTrainer(eqx.Module):
    config: GCVFTrainingConfig

    def __init__(self, config: GCVFTrainingConfig):
        self.config = config

    def train(
        self,
        policies: GCRLModel,
        gcvf: GCVF,
        num_seeds: int,
        env: Environment | EnvWrapper,
        env_params: EnvParams,
        key: jax.Array,
    ) -> GCVF:
        """Train GCVF models for each seed. Uses a Python loop over seeds due to
        data-dependent episode collection."""
        trained_gcvfs = []
        (policies_arr, gcvf_arr), (policies_static, gcvf_static) = eqx.partition(
            (policies, gcvf), eqx.is_array
        )
        for seed in range(num_seeds):
            logger.info(f"Training GCVF for seed {seed}...")
            logger.info("Collecting dataset...")
            policy = eqx.combine(
                jax.tree.map(lambda x, seed=seed: x[seed], policies_arr),
                policies_static,
            )
            key, subkey = jax.random.split(key)
            dataset = self.collect_dataset(policy, env, env_params, subkey)
            logger.info(f"Collected dataset with {dataset.values.shape[0]} samples.")
            logger.info("Training...")
            gcvf = eqx.combine(
                jax.tree.map(lambda x, seed=seed: x[seed], gcvf_arr), gcvf_static
            )
            key, subkey = jax.random.split(key)
            gcvf = self.train_gcvf(dataset, gcvf, subkey)
            logger.info("GCVF training complete.")
            trained_gcvfs.append(eqx.partition(gcvf, eqx.is_array)[0])
        trained_gcvfs = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *trained_gcvfs)
        return eqx.combine(trained_gcvfs, gcvf_static)

    def collect_dataset(
        self,
        policy: GCRLModel,
        env: Environment | EnvWrapper,
        env_params: EnvParams,
        key: jax.Array,
    ) -> GCVFDataset:
        """Collect a dataset of source/target goals and corresponding labels for GCVF training.
        Environment and model must be vmapped / batched."""

        @jax.jit
        def iteration(key: jax.Array) -> tuple[dict, jax.Array]:
            def step(carry, _):
                state, obs, step_key = carry
                step_key, action_key, env_key = jax.random.split(step_key, 3)
                distribution = policy.get_action(obs)
                action = distribution.sample(seed=action_key)
                transition = env.step(
                    jax.random.split(env_key, self.config.num_envs),
                    state,
                    action,
                    env_params,
                )
                item = {
                    "success": transition.reward > 0,
                    "obs": transition.terminal_observation,
                    "done": transition.done,
                }
                return (transition.state, transition.observation, step_key), item

            key, reset_key = jax.random.split(key)
            reset_key = jax.random.split(reset_key, self.config.num_envs)
            env_state, obsv = env.reset(reset_key, None, env_params, None)
            carry = (env_state, obsv, key)
            (_, _, key), collected = jax.lax.scan(
                step, carry, None, self.config.steps_per_env
            )
            return collected, key

        collected_obs = []
        collected_targets = []
        collected_values = []
        total_samples = 0

        while total_samples < self.config.num_samples:
            collected, key = iteration(key)  # [steps_per_env, num_envs, ...]
            data = jax.tree.map(lambda x: x.swapaxes(0, 1), collected)
            dones = data["done"]
            successes = data["success"]
            if not bool(jnp.any(successes)):
                continue

            # Compute episode IDs
            shifted_dones = jnp.pad(dones[:, :-1], ((0, 0), (1, 0)), constant_values=0)
            local_episode_ids = jnp.cumsum(shifted_dones, axis=1)
            max_eps = self.config.steps_per_env + 1
            env_offsets = jnp.arange(self.config.num_envs)[:, None] * max_eps
            global_ids = (local_episode_ids + env_offsets).reshape(-1)

            # Flatten successes and observations
            flat_success = successes.reshape(-1)
            flat_obs = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), data["obs"])

            # Get episodes that ended in success
            success_indices = global_ids[flat_success]
            success_mask = jnp.isin(global_ids, success_indices)  # [steps]
            obsv = jax.tree.map(
                lambda x, sm=success_mask: x[sm], flat_obs
            )  # [steps, ...]

            # Construct target goals: goal for each state is every goal that was not
            # the original episode goal
            num_props = len(env.propositions)
            goals = obsv.goal  # [steps]
            target_goals = jnp.arange(num_props - 1, dtype=jnp.int32)
            target_goals = jnp.where(
                target_goals < goals[:, None], target_goals, target_goals + 1
            ).flatten()  # [steps*(num_props-1)]

            # Repeat observations for each target goal
            repeated_obs = jax.tree.map(
                lambda x, num_props=num_props: jnp.repeat(x, num_props - 1, axis=0),
                obsv,
            )  # [steps*(num_props-1), ...]

            # Obtain values from trained value function for target goals
            obs_with_target = GoalObservation.from_obs(repeated_obs, target_goals)
            values = policy.get_value(obs_with_target)  # [steps*(num_props-1)]

            # Append to dataset
            collected_obs.append(repeated_obs)
            collected_targets.append(target_goals)
            collected_values.append(values)
            total_samples += len(values)

        dataset = GCVFDataset(
            obsv=jax.tree.map(
                lambda *xs: jnp.concatenate(xs, axis=0)[: self.config.num_samples],
                *collected_obs,
            ),
            target_goals=jnp.concatenate(collected_targets, axis=0)[
                : self.config.num_samples
            ],
            values=jnp.concatenate(collected_values, axis=0)[: self.config.num_samples],
        )
        return dataset

    def train_gcvf(self, dataset: GCVFDataset, gcvf: GCVF, key: jax.Array) -> GCVF:
        num_samples = dataset.values.shape[0]
        optimizer = optax.adamw(float(self.config.lr))
        state = TrainState.create(gcvf, optimizer)

        @eqx.filter_jit
        def update_epoch(
            state: TrainState, data: GCVFDataset, epoch_key: jax.Array
        ) -> tuple[TrainState, jax.Array]:
            num_batches = num_samples // self.config.batch_size
            num_train = num_batches * self.config.batch_size
            perm = jax.random.permutation(epoch_key, num_samples)
            batches = jax.tree.map(
                lambda x: x[perm[:num_train]].reshape(
                    (-1, self.config.batch_size) + x.shape[1:]
                ),
                data,
            )  # [num_batches, batch_size, ...]

            def loss_fn(model: GCVF, batch: GCVFDataset) -> jax.Array:
                predicted_values = model(batch.obsv, batch.target_goals)
                return jnp.mean((predicted_values - batch.values) ** 2)

            def update_batch(
                state: TrainState, batch: GCVFDataset
            ) -> tuple[TrainState, jax.Array]:
                loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model, batch)
                return state.apply_gradients(optimizer, grads), loss

            state, losses = eqx_utils.filter_scan(update_batch, state, batches)
            return state, jnp.mean(losses)

        for epoch in range(self.config.epochs):
            key, epoch_key = jax.random.split(key)
            state, loss = update_epoch(state, dataset, epoch_key)
            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: loss={loss:.4f}")
        return state.model
