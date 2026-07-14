"""Generate the WR-pace ghost trajectory for World 1-1.

The ghost is a frame-by-frame x-position trace that models a runner holding
max run speed from the start of the level to the flagpole, with a short
acceleration ramp. This approximates world-record pace (a real WR/TAS input
trace can be dropped into data/ghost_1_1.json in the same format to replace it).

SMB physics constants used:
- Mario's x_pos at spawn in 1-1 is ~40 (as reported by the env's info dict).
- The flagpole grab happens around x_pos 3161.
- Max run speed is 2.5625 px/frame (0x28.90 subpixels in the original engine).
"""

import argparse
import json
from pathlib import Path

START_X = 40.0
FLAG_X = 3161.0
MAX_RUN_SPEED = 2.5625  # px per frame at 60fps
RUN_ACCEL = 0.045       # px per frame^2, ~1s ramp to full speed


def build_trace(pace_scale: float) -> list[float]:
    """Simulate x per frame until the flag. pace_scale < 1.0 slows the ghost."""
    x, v = START_X, 0.0
    top_speed = MAX_RUN_SPEED * pace_scale
    frames = [x]
    while x < FLAG_X:
        v = min(v + RUN_ACCEL, top_speed)
        x += v
        frames.append(round(min(x, FLAG_X), 2))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-scale", type=float, default=1.0,
                        help="ghost speed multiplier, e.g. 0.8 for an easier ghost")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "ghost_1_1.json")
    args = parser.parse_args()

    frames = build_trace(args.pace_scale)
    finish_frame = len(frames) - 1
    ghost = {
        "level": "1-1",
        "source": f"synthetic WR-pace model (max run speed, pace_scale={args.pace_scale})",
        "fps": 60,
        "start_x": START_X,
        "flag_x": FLAG_X,
        "finish_frame": finish_frame,
        "finish_time_s": round(finish_frame / 60.0, 2),
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ghost))
    print(f"ghost written to {args.out}: finishes at frame {finish_frame} "
          f"({ghost['finish_time_s']}s)")


if __name__ == "__main__":
    main()
