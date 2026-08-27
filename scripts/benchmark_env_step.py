"""Measure JAX environment-step throughput for a batch of parallel environments."""

import argparse
import time

import jax
import jax.numpy as jnp

import jaxolotl
from jaxolotl.environments.reset import RESET_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Number of environments to step in parallel.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1_000,
        help="Number of batched steps to time after compilation (default: 1000).",
    )
    parser.add_argument(
        "--env-name",
        default="FrankaZoneEnv",
        help="Registered environment name (default: FrankaZoneEnv).",
    )
    parser.add_argument(
        "--reset-source",
        choices=RESET_SOURCES,
        default="train",
        help="Reset pool to use when creating the environment (default: train).",
    )
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    env, params = jaxolotl.make(args.env_name, reset_source=args.reset_source)
    action_space = env.action_space(params)

    reset_keys, action_keys, step_keys = jax.random.split(jax.random.key(0), 3)
    reset_keys = jax.random.split(reset_keys, args.num_envs)
    action_keys = jax.random.split(action_keys, args.num_envs)
    step_keys = jax.random.split(step_keys, args.num_steps * args.num_envs).reshape(
        args.num_steps, args.num_envs, -1
    )
    actions = jax.vmap(action_space.sample)(action_keys)

    reset_batch = jax.jit(jax.vmap(lambda key: env.reset(key, None, params), in_axes=0))
    states, _ = reset_batch(reset_keys)

    step_batch = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def run_steps(initial_states, keys, batch_actions):
        def step(states, inputs):
            keys, actions = inputs
            transition = step_batch(keys, states, actions, params)
            return transition.state, None

        return jax.lax.scan(step, initial_states, (keys, batch_actions))[0]

    repeated_actions = jax.tree.map(
        lambda action: jnp.broadcast_to(action, (args.num_steps,) + action.shape),
        actions,
    )
    states = run_steps(states, step_keys, repeated_actions)
    jax.block_until_ready(states)  # Compile and warm up outside the measurement.

    start = time.perf_counter()
    states = run_steps(states, step_keys, repeated_actions)
    jax.block_until_ready(states)
    elapsed = time.perf_counter() - start
    total_steps = args.num_envs * args.num_steps
    print(
        f"{args.env_name}: {total_steps / elapsed:,.0f} steps/s "
        f"({total_steps:,} steps across {args.num_envs:,} environments in "
        f"{elapsed:.3f} s)"
    )


if __name__ == "__main__":
    main()
