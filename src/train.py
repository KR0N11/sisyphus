"""Train PPO from scratch on SMB 1-1 with a live wall of parallel Marios
racing the WR ghost.

Hyperparameters follow the community-proven config for SMB (see PLAN.md):
lr 1e-4, gamma 0.9, GAE lambda 1.0, entropy 0.01, clip 0.2, 512-step rollouts,
10 epochs per update.
"""

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from callbacks import GhostRenderCallback
from ghost import Ghost
from mario_env import make_env

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=10,
                   help="parallel emulators (match your CPU core count to train fastest)")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--realtime", action="store_true",
                   help="pace the game to real NES speed for smooth watching "
                        "(training runs ~2x slower)")
    p.add_argument("--no-display", action="store_true",
                   help="headless: still writes runs/latest.png snapshots")
    p.add_argument("--checkpoint-every", type=int, default=50_000,
                   help="save a checkpoint every N total env steps")
    p.add_argument("--resume", type=Path, default=None,
                   help="path to a checkpoint .zip to continue training from")
    p.add_argument("--device", default="auto", help="auto | cpu | mps")
    return p.parse_args()


def _graceful_term(*_):
    raise KeyboardInterrupt


def main() -> None:
    # background shells ignore SIGINT, so make SIGTERM trigger the
    # save-on-exit path in the finally block too
    signal.signal(signal.SIGTERM, _graceful_term)
    args = parse_args()
    ghost = Ghost()
    print(f"ghost to beat: {ghost.finish_time_s:.2f}s ({ghost.source})")

    env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        print(f"resumed from {args.resume} at {model.num_timesteps:,} steps")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=(512 * args.num_envs) // 16,
            n_epochs=10,
            gamma=0.9,
            gae_lambda=1.0,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=str(ROOT / "runs" / "tb"),
            device=args.device,
            verbose=1,
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.num_envs, 1),
            save_path=str(ROOT / "checkpoints"),
            name_prefix="ppo_mario",
        ),
        GhostRenderCallback(ghost, args.num_envs, realtime=args.realtime,
                            display=not args.no_display, verbose=1),
    ])

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    reset_num_timesteps=args.resume is None)
    finally:
        model.save(ROOT / "checkpoints" / "ppo_mario_latest")
        env.close()
        print("saved checkpoints/ppo_mario_latest.zip")


if __name__ == "__main__":
    main()
