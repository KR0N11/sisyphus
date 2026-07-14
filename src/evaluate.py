"""Run a trained checkpoint deterministically and report clear times vs the ghost."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from ghost import Ghost
from mario_env import FRAME_SKIP, make_env

ROOT = Path(__file__).resolve().parents[1]


def newest_checkpoint() -> Path:
    ckpts = sorted((ROOT / "checkpoints").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        sys.exit("no checkpoints found; train first")
    return ckpts[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the argmax")
    args = p.parse_args()

    model_path = args.model or newest_checkpoint()
    ghost = Ghost()
    env = VecFrameStack(DummyVecEnv([make_env()]), n_stack=4)
    model = PPO.load(model_path, device="cpu")
    print(f"model: {model_path.name} ({model.num_timesteps:,} steps trained)  "
          f"ghost: {ghost.finish_time_s:.2f}s")

    obs = env.reset()
    for ep in range(1, args.episodes + 1):
        steps, done, info = 0, False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, dones, infos = env.step(action)
            done, info = dones[0], infos[0]
            steps += 1
        clear_s = steps * FRAME_SKIP / 60.0
        if info.get("flag_get"):
            delta = clear_s - ghost.finish_time_s
            print(f"ep {ep}: CLEARED in {clear_s:.2f}s real time "
                  f"(game timer left: {info.get('time')})  ghost {ghost.finish_time_s:.2f}s  "
                  f"{'+' if delta >= 0 else ''}{delta:.2f}s vs ghost")
        else:
            print(f"ep {ep}: died/timed out at x={info.get('x_pos')} after {clear_s:.2f}s")
    env.close()


if __name__ == "__main__":
    main()
