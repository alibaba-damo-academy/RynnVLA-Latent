"""Bounded local LIBERO closed-loop evaluation of RynnVLA (no RPC/Ray)."""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from .predict import add_policy_arguments, load_policy, nonnegative_int, positive_int, seed_policy
from ..inference_wrappers.rynn_brain_vla import libero_observation_to_sample

MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def load_libero():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO evaluation requires the optional upstream LIBERO package, its "
            "robosuite/MuJoCo dependencies, BDDL files and initial states. Install and "
            "configure them separately; offline prediction does not require LIBERO. "
            f"Import failed: {exc}"
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def _success(env, info):
    # Never label arbitrary termination/time limits as task success.
    check = getattr(env, "check_success", None)
    if callable(check):
        return bool(check())
    for key in ("success", "is_success"):
        if key in info:
            return bool(info[key])
    raise RuntimeError("Environment exposes neither check_success() nor explicit success info")


def _step(env, command):
    result = env.step(command)
    if len(result) == 4:
        obs, _, done, info = result
        info = info or {}
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    elif len(result) == 5:
        obs, _, terminated, truncated, info = result
        info = info or {}
    else:
        raise ValueError("Expected a 4- or 5-element environment step result")
    return obs, _success(env, info), bool(terminated), bool(truncated)


def run_episode(
    env_factory, policy, initial_state, instruction, *, seed=7, max_steps=220,
    replan_steps=10, denoising_steps=10, warmup_steps=10, frame_callback=None,
):
    """Own and close one environment, including on warmup/prediction failures.

    CPU tests inject a simulator double here; that is not a real benchmark run.
    max_steps counts learned-policy commands, warmup is separately bounded.
    """
    if min(max_steps, replan_steps, denoising_steps) < 1 or warmup_steps < 0:
        raise ValueError("Step budgets must be positive (warmup may be zero)")
    env = env_factory()
    try:
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(initial_state)
        success = _success(env, {})
        terminated = truncated = False
        warmup_done = steps = 0
        commands = []

        def capture():
            if frame_callback is not None:
                frame_callback(warmup_done + steps, observation)

        capture()
        for _ in range(warmup_steps):
            if success or terminated or truncated:
                break
            observation, success, terminated, truncated = _step(env, LIBERO_DUMMY_ACTION.copy())
            warmup_done += 1
            capture()

        plan = deque()
        while not (success or terminated or truncated) and steps < max_steps:
            if not plan:
                sample = libero_observation_to_sample(observation)
                chunk = np.asarray(policy.predict_libero(
                    text=instruction, **sample, denoising_steps=denoising_steps,
                ), dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk) or not np.isfinite(chunk).all():
                    raise ValueError("Policy must return finite, nonempty raw OSC actions (T, 7)")
                plan.extend(chunk[:replan_steps])
            command = plan.popleft().tolist()
            # Already raw OSC inputs: no current-pose subtraction, controller scaling,
            # gripper inversion, clipping, thresholding, or coordinate transforms.
            observation, success, terminated, truncated = _step(env, command)
            commands.append(command)
            steps += 1
            capture()

        reason = "success" if success else "terminated" if terminated else "truncated" if truncated else "max_steps"
        return {
            "success": success, "terminated": terminated, "truncated": truncated,
            "stop_reason": reason, "steps": steps, "warmup_steps": warmup_done,
            "environment_steps": steps + warmup_done, "actions": commands,
        }
    finally:
        env.close()


def _frame_writer(directory):
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)

    def save(index, observation):
        images = libero_observation_to_sample(observation)["images"]
        for camera, image in images.items():
            Image.fromarray(image).save(directory / f"{index:05d}_{camera}.png")

    return save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_arguments(parser)
    parser.add_argument("--suite", required=True, choices=tuple(MAX_STEPS_BY_SUITE))
    parser.add_argument("--task-id", required=True, type=nonnegative_int)
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replan-steps", type=positive_int, default=10)
    parser.add_argument("--max-steps", type=positive_int, help="Policy-step cap, also capped at the suite budget")
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=10)
    parser.add_argument("--resolution", type=positive_int, default=256)
    parser.add_argument("--save-frames", action="store_true", help="Save both model-oriented RGB views as PNG")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("--output-dir must be an empty directory to avoid mixing evaluation runs")
    try:
        benchmark, get_libero_path, env_class = load_libero()
        suites = benchmark.get_benchmark_dict()
        if args.suite not in suites:
            raise ValueError(f"Installed LIBERO has no suite {args.suite!r}")
        suite = suites[args.suite]()
        if args.task_id >= suite.get_num_tasks():
            raise ValueError("--task-id is outside the installed suite")
        task = suite.get_task(args.task_id)
        initial_states = suite.get_task_init_states(args.task_id)
        if args.episodes > len(initial_states):
            raise ValueError(f"Only {len(initial_states)} distinct initial states are available for this task")
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl_file.is_file():
            raise FileNotFoundError(f"Missing LIBERO task BDDL: {bddl_file}")
        policy = load_policy(args)
        max_steps = min(args.max_steps or MAX_STEPS_BY_SUITE[args.suite], MAX_STEPS_BY_SUITE[args.suite])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run_config.update(effective_max_steps=max_steps, task_name=task.name, language=task.language)
        (args.output_dir / "run.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        successes = 0
        with (args.output_dir / "results.jsonl").open("x", encoding="utf-8") as results:
            for episode in range(args.episodes):
                episode_seed = args.seed + episode
                seed_policy(episode_seed)
                start = time.monotonic()
                record = {
                    "suite": args.suite, "task_id": args.task_id, "task_name": task.name,
                    "episode": episode, "initial_state_index": episode, "seed": episode_seed,
                    "max_steps": max_steps, "replan_steps": args.replan_steps,
                }
                try:
                    frames = _frame_writer(args.output_dir / f"episode_{episode:04d}_frames") if args.save_frames else None
                    outcome = run_episode(
                        lambda: env_class(bddl_file_name=str(bddl_file), camera_heights=args.resolution,
                                          camera_widths=args.resolution),
                        policy, initial_states[episode], task.language, seed=episode_seed,
                        max_steps=max_steps, replan_steps=args.replan_steps,
                        denoising_steps=args.denoising_steps, warmup_steps=args.warmup_steps,
                        frame_callback=frames,
                    )
                    actions = np.asarray(outcome.pop("actions"), dtype=np.float32).reshape(-1, 7)
                    action_name = f"episode_{episode:04d}_actions.npy"
                    np.save(args.output_dir / action_name, actions, allow_pickle=False)
                    record.update(outcome, actions_file=action_name)
                    successes += int(outcome["success"])
                except Exception as exc:
                    # Infrastructure errors are unknown outcomes, never measured failures.
                    record.update(success=None, stop_reason="error", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    record["seconds"] = round(time.monotonic() - start, 3)
                    results.write(json.dumps(record) + "\n")
                    results.flush()
        summary = {"episodes": args.episodes, "successes": successes, "success_rate": successes / args.episodes}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"RynnVLA LIBERO evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
