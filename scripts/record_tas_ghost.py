"""Replay a WR/TAS input movie (.fm2) through the emulator and record Mario's
frame-by-frame x/y as the ghost trace (data/ghost_1_1.json).

fm2 input lines look like |0|RLDUTSBA|........|| where port0's characters map
directly onto the NES controller bitmask (right=128 ... A=1). The movie starts
at power-on but our env starts in-level, so we search for the input offset
where the replay syncs (detected by Mario reaching the flag at record pace).
"""

import argparse
import json
import sys
from pathlib import Path

import gym_super_mario_bros

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path(__file__).resolve().parents[1] / "data" / "ghost_1_1.json"
BITS = {"R": 128, "L": 64, "D": 32, "U": 16, "T": 8, "S": 4, "B": 2, "A": 1}
GROUND_TOP = 192  # screen y where a grounded small Mario's sprite starts


def parse_fm2(path: Path) -> list[int]:
    actions = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        port0 = line.split("|")[2]
        byte = 0
        for ch, bit in zip(port0, (128, 64, 32, 16, 8, 4, 2, 1)):
            if ch not in ". ":
                byte |= bit
        actions.append(byte)
    return actions


def try_offset(env, actions: list[int], offset: int, max_frames: int = 1500):
    """Feed actions[offset:] frame by frame. Returns (xs, ys, y_ground, time_left)
    if Mario reaches the flag, else None."""
    env.reset()
    xs, ys = [], []
    y_ground = None
    for f in range(min(max_frames, len(actions) - offset)):
        _, _, terminated, truncated, info = env.step(actions[offset + f])
        x, y = int(info["x_pos"]), int(info["y_pixel"])
        if y_ground is None:
            y_ground = y
        xs.append(x)
        ys.append(y)
        if info.get("flag_get"):
            return xs, ys, y_ground, int(info["time"])
        if terminated or truncated:
            return None
        # desync prune: a synced TAS keeps near max run speed throughout
        if f > 150 and x < 40 + 1.7 * f:
            return None
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fm2", type=Path, required=True)
    p.add_argument("--max-offset", type=int, default=400)
    args = p.parse_args()

    actions = parse_fm2(args.fm2)
    print(f"{len(actions)} input frames parsed from {args.fm2.name}")

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    best = None
    for offset in range(args.max_offset):
        result = try_offset(env, actions, offset)
        if result:
            xs, ys, y_ground, time_left = result
            print(f"offset {offset}: SYNCED, flag in {len(xs)} frames "
                  f"({len(xs) / 60:.2f}s), game timer left {time_left}")
            best = (xs, ys, y_ground, time_left, offset)
            break  # deterministic game: first synced offset is the run
    env.close()

    if best is None:
        sys.exit("no offset synced; the movie may target a different ROM revision")

    xs, ys, y_ground, time_left, offset = best
    y_cal = GROUND_TOP - y_ground
    ghost = {
        "level": "1-1",
        "source": f"TAS replay of {args.fm2.name} (offset {offset}), "
                  f"game timer at flag: {time_left}",
        "fps": 60,
        "start_x": float(xs[0]),
        "flag_x": float(xs[-1]),
        "finish_frame": len(xs) - 1,
        "finish_time_s": round((len(xs) - 1) / 60.0, 2),
        "frames": [float(x) for x in xs],
        "y_top": [y + y_cal for y in ys],
    }
    OUT.write_text(json.dumps(ghost))
    print(f"ghost written to {OUT}: {ghost['finish_time_s']}s to flag, "
          f"x {xs[0]}..{xs[-1]}, y_top range {min(ghost['y_top'])}..{max(ghost['y_top'])}")


if __name__ == "__main__":
    main()
