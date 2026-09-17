"""GCRL-LTL training pipeline.

Trains primitives, goal-conditioned policy, and goal-conditioned value functions.
"""

import logging
from pathlib import Path

import equinox as eqx
import hydra
import jax
import pandas as pd
from omegaconf import DictConfig

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.alg.gcrl_ltl.model.model import GCRLModel
from jaxolotl.alg.gcrl_ltl.training.capability import (
    CapabilityDataset,
    collect_dataset,
    train_capability,
)
from jaxolotl.alg.gcrl_ltl.training.ppo import PipelinePPO
from jaxolotl.alg.gcrl_ltl.wrappers.goal_wrapper import GoalConditionedWrapper
from jaxolotl.alg.gcrl_ltl.wrappers.primitive_wrapper import PrimitiveRewardWrapper
from jaxolotl.environments.wrappers import AutoResetWrapper, VectorizeWrapper
from jaxolotl.environments.wrappers.auto_reset_wrapper import ResetStrategy
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.utils.artifact_utils import model_path

logger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Run primitives, goal PPO/Q, and capability regression for each seed."""
    run_dir = Path.cwd()
    seeds = list(range(int(cfg.start_seed), int(cfg.start_seed) + int(cfg.num_seeds)))

    for seed in seeds:
        logger.info("Starting GCRL-LTL pipeline for seed %s", seed)
        key = jax.random.key(seed)
        key, model_key = jax.random.split(key)
        env, env_params = _make_base_env(cfg)
        model = _build_model(cfg, env, env_params, model_key)

        primitive_path = _stage_model_path(run_dir, "primitives", seed)
        if _can_resume(primitive_path, run_hash):
            model = _load_module(primitive_path, model)
            logger.info("Resumed primitive stage for seed %s", seed)
        else:
            model, key = _train_primitives(
                cfg, model, env_params, key, run_dir, seed, run_hash
            )
            _save_module(primitive_path, model, seed, run_hash, "primitives")

        goal_path = _stage_model_path(run_dir, "goal_policy", seed)
        if _can_resume(goal_path, run_hash):
            model = _load_module(goal_path, model)
            logger.info("Resumed goal-policy stage for seed %s", seed)
        else:
            model, key = _train_goal_model(
                cfg, model, env_params, key, run_dir, seed, run_hash
            )
            _save_module(goal_path, model, seed, run_hash, "goal_policy")

        dataset_path = (
            run_dir / "stages" / "goal_policy" / "datasets" / f"dataset_seed{seed}.eqx"
        )
        if _can_resume(dataset_path, run_hash):
            dataset = eqx_utils.load_from_template(dataset_path)
        else:
            key, collect_key = jax.random.split(key)
            dataset = _collect_dataset(cfg, model, env_params, collect_key)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            eqx_utils.save_with_template(
                dataset_path,
                dataset,
                metadata={"seed": seed, "config_hash": run_hash, "stage": "dataset"},
            )

        capability_path = _stage_model_path(run_dir, "capability", seed)
        if _can_resume(capability_path, run_hash):
            model = _load_module(capability_path, model)
            logger.info("Resumed capability stage for seed %s", seed)
        else:
            key, capability_key = jax.random.split(key)
            callback = _make_callback(
                run_dir,
                "capability",
                seed,
                run_hash,
                save_freq=int(cfg.capability_training.save_freq),
            )
            model = train_capability(
                model,
                dataset,
                cfg.capability_training,
                key=capability_key,
                callback=callback,
            )
            _save_module(capability_path, model, seed, run_hash, "capability")

        final_path = model_path(run_dir, seed)
        _save_module(final_path, model, seed, run_hash, "final")
        logger.info("Completed GCRL-LTL pipeline for seed %s", seed)


def _make_base_env(cfg):
    env, env_params = jaxolotl.make(
        cfg.env.name, reset_source=cfg.env.get("reset_source", "train")
    )
    return env, env_params


def _build_model(cfg, env, env_params, key) -> GCRLModel:
    factory = hydra.utils.instantiate(
        cfg.model,
        obs_spec=env.observation_spec(env_params),
        num_assignments=len(env.assignments()),
        num_propositions=len(env.propositions),
        env_params=env_params,
        key=key,
        _partial_=True,
    )
    return factory(act_space=env.action_space(env_params))


def _train_primitives(cfg, model, env_params, key, run_dir, seed, run_hash):
    for direction in range(model.learned_primitives):
        key, train_key = jax.random.split(key)
        base_env, _ = _make_base_env(cfg)
        env = TimeLimitWrapper(base_env)
        env = PrimitiveRewardWrapper(env, direction)
        env = AutoResetWrapper(env, reset_strategy=ResetStrategy.FULL)
        env = VectorizeWrapper(env)
        callback = _make_callback(
            run_dir,
            "primitives",
            seed,
            run_hash,
            save_freq=int(cfg.primitives.save_freq),
            substage=f"direction_{direction}",
        )
        trainer = PipelinePPO(cfg.primitives)
        trained = trainer.train(
            model.primitive_models[direction],
            env,
            env_params,
            train_key,
            callback=callback,
        )
        primitive_models = list(model.primitive_models)
        primitive_models[direction] = trained
        model = eqx.tree_at(
            lambda item: item.primitive_models, model, tuple(primitive_models)
        )
        primitive_path = (
            run_dir
            / "stages"
            / "primitives"
            / f"direction_{direction}"
            / "models"
            / f"model_seed{seed}.eqx"
        )
        _save_module(primitive_path, trained, seed, run_hash, f"primitive_{direction}")
    return model, key


def _train_goal_model(cfg, model, env_params, key, run_dir, seed, run_hash):
    key, train_key = jax.random.split(key)
    base_env, _ = _make_base_env(cfg)
    env = TimeLimitWrapper(base_env)
    env = GoalConditionedWrapper(env, model)
    env = AutoResetWrapper(env, reset_strategy=ResetStrategy.FULL)
    env = VectorizeWrapper(env)
    callback = _make_callback(
        run_dir,
        "goal_policy",
        seed,
        run_hash,
        save_freq=int(cfg.goal_training.save_freq),
    )
    trainer = PipelinePPO(
        cfg.goal_training,
        q_coef=float(cfg.goal_training.q_coef),
        terminal_on_truncation=True,
    )
    model = trainer.train(model, env, env_params, train_key, callback=callback)
    return model, key


def _collect_dataset(cfg, model, env_params, key) -> CapabilityDataset:
    base_env, _ = _make_base_env(cfg)
    env = TimeLimitWrapper(base_env)
    env = GoalConditionedWrapper(env, model)
    env = AutoResetWrapper(env, reset_strategy=ResetStrategy.FULL)
    env = VectorizeWrapper(env)
    return collect_dataset(
        model,
        env,
        env_params,
        num_envs=int(cfg.capability_training.num_envs),
        timesteps=int(cfg.capability_training.collection_steps),
        max_successes=int(cfg.capability_training.max_successes),
        key=key,
    )


def _stage_model_path(run_dir: Path, stage: str, seed: int) -> Path:
    return run_dir / "stages" / stage / "models" / f"model_seed{seed}.eqx"


def _can_resume(path: Path, run_hash: str) -> bool:
    if not path.exists():
        return False
    metadata = eqx_utils.load_metadata(path)
    found_hash = metadata.get("config_hash")
    if found_hash != run_hash:
        raise ValueError(
            f"Cannot resume {path}: config hash {found_hash} != {run_hash}."
        )
    return True


def _save_module(path, module, seed, run_hash, stage):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters, _ = eqx.partition(module, eqx.is_array)
    eqx_utils.save(
        path,
        parameters,
        metadata={"seed": seed, "config_hash": run_hash, "stage": stage},
    )


def _load_module(path, template):
    parameters, static = eqx.partition(template, eqx.is_array)
    return eqx.combine(eqx_utils.load(path, parameters), static)


def _make_callback(run_dir, stage, seed, run_hash, save_freq, substage=""):
    next_checkpoint = save_freq
    stage_dir = Path(stage) / substage if substage else Path(stage)

    def callback(step, metrics, model):
        nonlocal next_checkpoint
        row = {
            "seed": seed,
            "stage": stage,
            "substage": substage,
            "step": step,
        } | metrics
        _append_csv(run_dir / "logs.csv", row)
        _append_csv(run_dir / "stages" / stage_dir / f"logs_seed{seed}.csv", row)
        logger.info(
            "seed %s | stage %s%s | step %s | loss %.5f",
            seed,
            stage,
            f"/{substage}" if substage else "",
            step,
            metrics.get("loss", metrics.get("train_loss", float("nan"))),
        )
        if step >= next_checkpoint:
            checkpoint = (
                run_dir
                / "stages"
                / stage_dir
                / "checkpoints"
                / f"model_seed{seed}_step{step}.eqx"
            )
            _save_module(checkpoint, model, seed, run_hash, stage)
            next_checkpoint += save_freq

    return callback


def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        new_row = pd.concat([existing, new_row], ignore_index=True, sort=False)
    new_row.to_csv(path, index=False)
