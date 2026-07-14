"""Extract Mario's sprite from the emulator and save it as an RGBA cutout
(data/ghost_sprite.png) for the low-opacity ghost overlay.

Mario spawns in front of a hill, so a plain background mask can't isolate him
there. Instead we make him jump straight up and capture the frame at the apex,
where everything that isn't sky inside a window around him is Mario.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mario_env import make_env  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "ghost_sprite.png"
JUMP = 5  # SIMPLE_MOVEMENT index for ['A']


def main() -> None:
    env = make_env()()
    env.reset()

    best_frame, best_y = None, 1_000
    for _ in range(10):
        _, _, term, trunc, info = env.step(JUMP)
        y = int(info["y_pixel"])
        if y < best_y:  # screen y grows downward; apex = smallest y
            best_y, best_frame = y, np.asarray(env.render()).copy()
        if term or trunc:
            break
    x = int(info["x_pos"])
    env.close()

    # window around Mario at the apex: columns near spawn x, rows near apex y
    win_y = slice(max(best_y - 24, 0), min(best_y + 48, 240))
    win_x = slice(max(x - 12, 0), min(x + 28, 256))
    window = best_frame[win_y, win_x]

    sky = np.array([104, 136, 252])  # NES SMB sky blue (RGB)
    mask = np.abs(window.astype(int) - sky).sum(axis=2) > 40
    ys, xs = np.nonzero(mask)
    assert len(ys) > 20, "no sprite found at jump apex"
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    sprite = window[y0:y1, x0:x1]
    alpha = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    rgba = np.dstack([cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR), alpha])
    cv2.imwrite(str(OUT), rgba)
    print(f"sprite {x1 - x0}x{y1 - y0} saved to {OUT}")


if __name__ == "__main__":
    main()
