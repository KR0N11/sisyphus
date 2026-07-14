"""Showcase mode: run a trained checkpoint on a big wall of emulators (e.g. 30)
with the ghost-race HUD, throttled to roughly real-time. No training happens.

Press q in the window to quit.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from evaluate import newest_checkpoint
from ghost import Ghost
from hud import compose
from mario_env import FRAME_SKIP, make_env

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--num-envs", type=int, default=30)
    p.add_argument("--fps", type=float, default=60.0,
                   help="target display frame rate (game frames per second)")
    args = p.parse_args()

    model_path = args.model or newest_checkpoint()
    ghost = Ghost()
    env = VecFrameStack(SubprocVecEnv([make_env(i) for i in range(args.num_envs)]), n_stack=4)
    model = PPO.load(model_path, device="cpu")
    print(f"showcasing {model_path.name} on {args.num_envs} envs; q to quit")

    ep_steps = [0] * args.num_envs
    episodes = flags = 0
    best: float | None = None
    step_period = FRAME_SKIP / args.fps

    obs = env.reset()
    while True:
        t0 = time.monotonic()
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = env.step(action)

        states = []
        for i in range(args.num_envs):
            ep_steps[i] += 1
            if dones[i]:
                episodes += 1
                if infos[i].get("flag_get"):
                    flags += 1
                    clear_s = ep_steps[i] * FRAME_SKIP / 60.0
                    best = clear_s if best is None or clear_s < best else best
                ep_steps[i] = 0
            states.append({"x": int(infos[i].get("x_pos", 0)),
                           "frame": ep_steps[i] * FRAME_SKIP})

        img = compose(env.get_images(), states, ghost,
                      {"steps": 0, "episodes": episodes, "flags": flags, "best": best})
        cv2.imshow("mario-rl showcase: AI vs WR ghost", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(max(0.0, step_period - (time.monotonic() - t0)))

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
