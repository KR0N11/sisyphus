"""Phase-1 gate: boot one emulator, take random actions, verify the info dict
and rendering work before anything else is built on top.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mario_env import make_env  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "sanity.png"


def main() -> None:
    env = make_env()()
    obs, info = env.reset()
    print(f"obs shape after preprocessing: {obs.shape} dtype={obs.dtype}")
    print(f"action space: {env.action_space}")

    frame = None
    for step in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if step == 0:
            print(f"info keys: {sorted(info)}")
        if step % 50 == 0:
            print(f"step {step}: x_pos={info.get('x_pos')} time={info.get('time')} "
                  f"score={info.get('score')} flag={info.get('flag_get')} reward={reward:.3f}")
        frame = env.render()
        if terminated or truncated:
            obs, info = env.reset()

    assert frame is not None and frame.ndim == 3, "render() did not return an RGB frame"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    print(f"render frame {frame.shape} saved to {OUT}")
    env.close()
    print("SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
