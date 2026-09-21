"""Evaluation utilities."""

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from jaxolotl import eqx_utils
from jaxolotl.alg.gcrl_ltl.model.gcvf import GCVF
from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eval.eval import Evaluator
from jaxolotl.rl.actor_critic import ActorCritic
from jaxolotl.utils.artifact_utils import (
    GCVF_DIRECTORY,
    MODEL_DIRECTORY,
    discover_seed_models,
    verify_model_metadata,
)


def load_batched_actor_critic(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
) -> tuple[ActorCritic, list[int]]:
    model_fn = hydra.utils.instantiate(
        cfg.model,
        obs_spec=env.observation_spec(env_params),
        num_assignments=len(env.assignments()),
        num_propositions=len(env.propositions),
        env_params=env_params,
        key=key,
        _partial_=True,
    )
    model: ActorCritic = model_fn(act_space=env.action_space(env_params))
    return load_batched_models(cfg, model)  # type: ignore


def load_batched_gcvf(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
) -> tuple[eqx.Module, list[int]]:
    gcvf_model = GCVF(
        env.observation_spec(env_params),
        cfg.model.env_net,
        cfg.model.critic,
        cfg.model.embedding_dim,
        len(env.propositions),
        key,
    )
    return load_batched_models(
        cfg, gcvf_model, model_directory=GCVF_DIRECTORY
    )


def load_batched_models(
    cfg: DictConfig,
    model: eqx.Module,
    *,
    path: Path | None = None,
    model_directory: str = MODEL_DIRECTORY,
) -> tuple[eqx.Module, list[int]]:
    """Load a batched model (over seeds) from disk.

    Returns:
        batched model, seeds
    """
    run_dir = path or Path(f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}")
    seed_to_file = discover_seed_models(run_dir, model_directory)
    if not seed_to_file:
        raise FileNotFoundError(
            f"No final models found in {run_dir / model_directory}."
        )
    verify_model_metadata(seed_to_file)
    seeds = sorted(seed_to_file)
    requested = cfg.eval.get("num_seeds", None)
    if requested is not None:
        if requested > len(seeds):
            raise ValueError(
                f"Requested {requested} seeds, but only {len(seeds)} are available."
            )
        seeds = seeds[: int(requested)]
    params, static = eqx.partition(model, eqx.is_array)
    per_seed = [eqx_utils.load(seed_to_file[seed], params) for seed in seeds]
    batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_seed)
    return eqx.combine(batched, static), seeds


def load_latest_checkpoint_models(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
    max_steps: int | float,
) -> tuple[ActorCritic, list[int], int]:
    """Load every seed from the latest checkpoint at or before ``max_steps``.

    Returns:
        Batched ActorCritic model, seed labels, and the selected checkpoint
        step.

    Raises:
        FileNotFoundError: If no checkpoint at or before ``max_steps`` exists.
        ValueError: If checkpoint files at the selected step do not cover the
            same seed set as the other available checkpoint steps.
    """
    model_fn = hydra.utils.instantiate(
        cfg.model,
        obs_spec=env.observation_spec(env_params),
        num_assignments=len(env.assignments()),
        num_propositions=len(env.propositions),
        env_params=env_params,
        _partial_=True,
    )
    model: ActorCritic = model_fn(act_space=env.action_space(env_params), key=key)
    params, static = eqx.partition(model, eqx.is_array)

    checkpoint_folder = Path(
        f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/checkpoints"
    )
    checkpoint_pattern = re.compile(r"model_seed(\d+)_step(\d+)\.eqx")
    checkpoints: dict[int, dict[int, Path]] = defaultdict(dict)
    for file in checkpoint_folder.iterdir():
        match = checkpoint_pattern.fullmatch(file.name)
        if match is not None:
            seed, step = map(int, match.groups())
            checkpoints[step][seed] = file

    eligible_steps = [step for step in checkpoints if step <= max_steps]
    if not eligible_steps:
        raise FileNotFoundError(
            f"No checkpoint at or before step {max_steps} in {checkpoint_folder}."
        )
    checkpoint_step = max(eligible_steps)
    checkpoint_files = checkpoints[checkpoint_step]

    # A partially written checkpoint should not silently drop seeds from eval.
    seed_sets = [set(files) for files in checkpoints.values()]
    if not all(seeds == set(checkpoint_files) for seeds in seed_sets):
        raise ValueError("Not all checkpoints have the same seeds.")

    seeds = sorted(checkpoint_files)
    models_per_seed = [eqx_utils.load(checkpoint_files[seed], params) for seed in seeds]
    models = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *models_per_seed)
    return eqx.combine(models, static), seeds, checkpoint_step


def load_model_checkpoints(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
) -> tuple[ActorCritic, list[int], list[int]]:
    """Load model checkpoints from disk.

    Returns:
        Batched ActorCritic model with shape (num_checkpoints, num_seeds, ...),
        seed labels,
        list of checkpoint steps.
    """

    model_fn = hydra.utils.instantiate(
        cfg.model,
        obs_spec=env.observation_spec(env_params),
        num_assignments=len(env.assignments()),
        num_propositions=len(env.propositions),
        env_params=env_params,
        _partial_=True,
    )
    make_model = lambda key: model_fn(act_space=env.action_space(env_params), key=key)
    model = make_model(key)
    params, static = eqx.partition(model, eqx.is_array)

    # load checkpoints
    step_to_models = defaultdict(dict)
    checkpoint_folder = Path(
        f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/checkpoints"
    )
    checkpoint_pattern = re.compile(r"model_seed(\d+)_step(\d+)\.eqx")
    for file in checkpoint_folder.iterdir():
        match = checkpoint_pattern.fullmatch(file.name)
        if match is None:
            continue
        seed, step = match.groups()
        checkpoint_params = eqx_utils.load(file, params)
        step_to_models[int(step)][int(seed)] = checkpoint_params

    if not step_to_models:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_folder}.")
    seeds_per_step = [set(seeds.keys()) for seeds in step_to_models.values()]
    if not all(seeds == seeds_per_step[0] for seeds in seeds_per_step):
        raise ValueError("Not all checkpoints have the same seeds.")

    # load initial models
    seeds = sorted(seeds_per_step[0])
    for seed in seeds:
        model_key = jax.random.split(jax.random.key(seed))[1]
        init_params, _ = eqx.partition(make_model(model_key), eqx.is_array)
        step_to_models[0][seed] = init_params

    sorted_steps = sorted(step_to_models)
    models_list = []
    for step in sorted_steps:
        seeds_dict = step_to_models[step]
        models_per_seed = []
        for seed in sorted(seeds_dict):
            models_per_seed.append(seeds_dict[seed])
        models_list.append(
            jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *models_per_seed)
        )
    models = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *models_list)
    return eqx.combine(models, static), seeds, sorted_steps


def make_eval_fn(
    cfg: DictConfig, num_models: int, num_formulas: int, return_trajs: bool
) -> Callable:
    """Creates an eval function for different formulas and seeds. The function
    returns arrays of shape (num_models, num_formulas, num_episodes)."""

    evaluator = Evaluator(
        num_episodes=cfg.eval.num_episodes,
        discount=cfg.eval.discount,
        return_trajs=return_trajs,
    )
    if cfg.eval.models_per_batch == "all":
        model_batch_size = num_models
    else:
        model_batch_size = cfg.eval.models_per_batch

    def eval_fn(agents, env, env_params, formulas, eval_key):
        if cfg.eval.formulas_per_batch == "all":
            formula_batch_size = num_formulas
        else:
            formula_batch_size = cfg.eval.formulas_per_batch

        def eval_seed(x):
            key, agent = x
            formula_keys = jax.random.split(key, num_formulas)

            def eval_formula(x):
                key, formula = x
                return evaluator.eval(
                    agent, cfg.eval.deterministic, env, env_params, formula, key=key
                )

            res = eqx_utils.batch_map(
                eval_formula,
                (formula_keys, formulas),
                batch_size=formula_batch_size,
            )
            return res

        keys = jax.random.split(eval_key, num_models)
        res = eqx_utils.filter_map(
            eval_seed, (keys, agents), batch_size=model_batch_size
        )
        return res

    return eval_fn
