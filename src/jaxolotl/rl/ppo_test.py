from collections.abc import Sequence

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from jaxolotl.alg.curriculum.curriculum_manager import CurriculumManager
from jaxolotl.rl.ppo import PPO, PPOTransition
from jaxolotl.rl.ppo_safe import PPOSafe, PPOSafeTransition

GAMMA = 0.9
GAE_LAMBDA = 0.5


class _ValueIsObservation:
    def get_value(self, observation):
        return observation

    def get_cost_value(self, observation):
        return observation


def _manager(num_envs: int = 1) -> CurriculumManager:
    return CurriculumManager(jnp.ones((1,)), num_envs=num_envs, window=1)


def _ppo(**overrides) -> PPO:
    config = {
        "total_timesteps": 3,
        "num_envs": 1,
        "num_steps": 3,
        "num_minibatches": 1,
        "update_epochs": 1,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "lr": 1e-3,
        "max_grad_norm": 0.5,
        "anneal_lr": False,
        "adam_eps": 1e-5,
    } | overrides
    return PPO(_manager(config["num_envs"]), **config)


def _ppo_safe(**overrides) -> PPOSafe:
    config = {
        "total_timesteps": 3,
        "num_envs": 1,
        "num_steps": 3,
        "num_minibatches": 1,
        "update_epochs": 1,
        "gamma": GAMMA,
        "cost_gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "cost_vf_coef": 0.5,
        "lag_coef": 1.0,
        "lr": 1e-3,
        "max_grad_norm": 0.5,
        "anneal_lr": False,
        "adam_eps": 1e-5,
        "target_cost": 0.0,
        "min_lag": 0.0,
        "max_lag": 1.0,
    } | overrides
    return PPOSafe(_manager(config["num_envs"]), **config)


def _column(values: Sequence[float | bool], dtype) -> jnp.ndarray:
    return jnp.asarray(values, dtype=dtype).reshape(-1, 1)


def test_ppo_truncation_bootstraps_without_crossing_episode_boundary() -> None:
    zeros = jnp.zeros((3, 1), dtype=jnp.float32)
    trajectories = PPOTransition(
        terminated=_column([False, False, False], bool),
        truncated=_column([False, True, False], bool),
        action=zeros,
        value=zeros,
        reward=_column([1.0, 2.0, 1_000.0], jnp.float32),
        log_prob=zeros,
        obs=zeros,
        terminal_obs=_column([0.0, 7.0, 0.0], jnp.float32),
        info={},
    )

    advantages, _ = _ppo()._calculate_gae(
        trajectories, jnp.zeros((1,)), _ValueIsObservation()
    )

    boundary_advantage = 2.0 + GAMMA * 7.0
    npt.assert_allclose(advantages[1], boundary_advantage)
    npt.assert_allclose(advantages[0], 1.0 + GAMMA * GAE_LAMBDA * boundary_advantage)


def test_ppo_safe_cost_backups_do_not_cross_truncation() -> None:
    zeros = jnp.zeros((3, 1), dtype=jnp.float32)
    trajectories = PPOSafeTransition(
        terminated=_column([False, False, False], bool),
        truncated=_column([False, True, False], bool),
        action=zeros,
        value=zeros,
        cost_value=zeros,
        reward=zeros,
        cost=_column([0.0, 0.25, 100.0], jnp.float32),
        log_prob=zeros,
        obs=zeros,
        terminal_obs=_column([0.0, 0.5, 0.0], jnp.float32),
        info={},
    )

    _, _, cost_advantages, cost_returns = _ppo_safe()._calculate_gae(
        trajectories, jnp.zeros((1,)), _ValueIsObservation()
    )

    boundary_delta = (1.0 - GAMMA) * 0.25 + GAMMA * 0.5
    npt.assert_allclose(cost_advantages[1], boundary_delta)
    npt.assert_allclose(cost_returns[1], 0.25)


@pytest.mark.parametrize("make_algorithm", [_ppo, _ppo_safe])
def test_annealing_uses_elapsed_environment_timesteps(make_algorithm) -> None:
    algorithm = make_algorithm(
        total_timesteps=80,
        num_envs=4,
        num_steps=5,
        num_minibatches=2,
        update_epochs=2,
        lr=0.1,
        anneal_lr=True,
        anneal_lr_end=0.02,
        ent_coef=0.04,
        anneal_ent_coef=True,
        anneal_ent_coef_end=0.0,
    )

    optimizer_steps_at_halfway = jnp.asarray(2 * 2 * 2)
    updates_at_halfway = jnp.asarray(2)
    npt.assert_allclose(algorithm.linear_schedule(optimizer_steps_at_halfway), 0.06)
    npt.assert_allclose(algorithm.ent_coef_schedule(updates_at_halfway), 0.02)

    optimizer_steps_after_end = jnp.asarray(8 * 2 * 2)
    updates_after_end = jnp.asarray(8)
    npt.assert_allclose(algorithm.linear_schedule(optimizer_steps_after_end), 0.02)
    npt.assert_allclose(algorithm.ent_coef_schedule(updates_after_end), 0.0)
