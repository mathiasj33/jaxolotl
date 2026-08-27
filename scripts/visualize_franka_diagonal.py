"""Roll out a fixed diagonal Panda motion and show its plot and pygame replay.

Run with ``pixi run python scripts/visualize_franka_diagonal.py``. Close the
Matplotlib window to begin the pygame replay; press Q or close its window when
finished. On a headless machine, set ``MUJOCO_GL=egl`` before starting Python
and pass ``--skip-replay``.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

import jaxolotl

# Move upward and diagonally across the Panda base frame. Rotation stays fixed.
_DIAGONAL_ACTION = jnp.array([0.05, 0.05, 0.10, 0.0, 0.0, 0.0], dtype=jnp.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30, help="Number of policy steps.")
    parser.add_argument("--seed", type=int, default=0, help="Reset random seed.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path(".artifacts/franka_diagonal_trajectory.png"),
        help="Static trajectory plot destination.",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=5,
        help="pygame frames shown for each policy step.",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Save and show only the static trajectory plot.",
    )
    return parser.parse_args()


def _rollout(env, params, key: jax.Array, steps: int):
    if steps < 1:
        raise ValueError("steps must be positive.")

    key, reset_key = jax.random.split(key)
    state, _ = env.reset(reset_key, None, params)
    states = [state]
    ik_successes = []
    for _ in range(steps):
        key, step_key = jax.random.split(key)
        transition = env.step(step_key, state, _DIAGONAL_ACTION, params)
        state = transition.state
        states.append(state)
        ik_successes.append(transition.info["ik_success"])

    trajectory = jax.tree.map(lambda *values: jnp.stack(values), *states)
    trajectory = jax.tree.map(lambda values: values[None, ...], trajectory)
    lengths = jnp.asarray([steps], dtype=jnp.int32)
    return trajectory, lengths, jnp.asarray(ik_successes)


def main() -> None:
    args = _parse_args()
    env, params = jaxolotl.make("FrankaZoneEnv", reset_source="native")
    trajs, lengths, ik_successes = _rollout(
        env, params, jax.random.key(args.seed), args.steps
    )

    print(
        f"Diagonal policy accepted on {int(jnp.sum(ik_successes))}/{args.steps} steps. "
        f"Static plot: {args.save_path}"
    )
    env.plot_trajectories(
        trajs,
        lengths,
        params,
        num_rows=1,
        num_cols=1,
        save_path=str(args.save_path),
    )

    if not args.skip_replay:
        renderer = env.get_renderer(params)
        try:
            renderer.replay_trajectories(
                trajs, lengths, frames_per_step=args.frames_per_step
            )
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
