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
from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eval.eval import Evaluator
from jaxolotl.rl.actor_critic import ActorCritic


def load_batched_models(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
    path: Path | None = None,
) -> tuple[ActorCritic, int]:
    """Load a batched model (over seeds) from disk.

    Returns:
        batched model, batch size
    """

    model_path = path or f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/models.eqx"
    metadata = eqx_utils.load_metadata(model_path)
    num_models = metadata["num_models"]
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
    models = eqx_utils.add_batch_dim(model, num_models)
    models = eqx_utils.load(model_path, models)
    num_seeds = cfg.eval.get("num_seeds", None)
    if num_seeds is not None:
        params, static = eqx.partition(models, eqx.is_array)
        models = eqx.combine(jax.tree.map(lambda x: x[:num_seeds], params), static)
        num_models = num_seeds
    return models, num_models


def load_latest_checkpoint_models(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
    max_steps: int | float,
) -> tuple[ActorCritic, int, int]:
    """Load every seed from the latest checkpoint at or before ``max_steps``.

    Returns:
        Batched ActorCritic model, number of seeds, and the selected checkpoint
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

    models_per_seed = [
        eqx_utils.load(checkpoint_files[seed], params)
        for seed in sorted(checkpoint_files)
    ]
    models = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *models_per_seed)
    return eqx.combine(models, static), len(models_per_seed), checkpoint_step


def load_model_checkpoints(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
) -> tuple[ActorCritic, int, list[int]]:
    """Load model checkpoints from disk.

    Returns:
        Batched ActorCritic model with shape (num_checkpoints, num_seeds, ...),
        number of seeds,
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
    for file in checkpoint_folder.iterdir():
        seed = re.search(r"seed(\d+)", file.name).group(1)  # type: ignore
        step = re.search(r"step(\d+)", file.name).group(1)  # type: ignore
        checkpoint_params = eqx_utils.load(file, params)
        step_to_models[int(step)][int(seed)] = checkpoint_params

    seeds_per_step = [set(seeds.keys()) for seeds in step_to_models.values()]
    if not all(seeds == seeds_per_step[0] for seeds in seeds_per_step):
        raise ValueError("Not all checkpoints have the same seeds.")

    # load initial models
    num_seeds = len(seeds_per_step[0])
    for seed in range(num_seeds):
        key, subkey = jax.random.split(key)
        init_params, _ = eqx.partition(make_model(subkey), eqx.is_array)
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
    return eqx.combine(models, static), num_seeds, sorted_steps


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
