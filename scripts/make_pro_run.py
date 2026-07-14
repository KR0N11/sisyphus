"""Auto-tune a WR-style speedrun of 1-1 inside the emulator and record it as
the ghost trace (data/ghost_1_1.json), including y for jumps.

Strategy: hold right+B (max run speed) the whole level. Whenever progress
stalls (pipe, pit, goomba, stairs), search jump press frames and hold
durations that get past the obstacle, preferring the fastest. If a chosen
jump leads to a dead end later (e.g. a landing that can't avoid the pit),
backtrack and take the next-best jump for the previous obstacle.
"""

import json
import sys
from pathlib import Path

import gym_super_mario_bros

OUT = Path(__file__).resolve().parents[1] / "data" / "ghost_1_1.json"
RUN = 128 | 2  # right + B
JUMP_DURATIONS = (28, 22, 16, 10, 6)
SEARCH_WINDOW = 110       # how many frames before the stall to try presses
CANDIDATES_PER_OBSTACLE = 4
GROUND_TOP = 192


def a_frames_of(jumps):
    """Frames where A is held; a release frame is forced before each press so
    consecutive jumps register as separate presses."""
    held = set()
    for press, dur in jumps:
        held.update(range(press, press + dur))
    for press, _ in jumps:
        held.discard(press - 1)
    return held


def simulate(env, jumps, limit=1500):
    held = a_frames_of(jumps)
    env.reset()
    xs, ys = [], []
    for f in range(limit):
        _, _, term, trunc, info = env.step(RUN | (1 if f in held else 0))
        xs.append(int(info["x_pos"]))
        ys.append(int(info["y_pixel"]))
        if info.get("flag_get"):
            return "flag", xs, ys, int(info["time"])
        if term or trunc:
            return "dead", xs, ys, None
    return "stuck", xs, ys, None


def plateau_frame(xs):
    peak = max(xs)
    return next(f for f, x in enumerate(xs) if x >= peak - 1)


def find_candidates(env, jumps, stall, stall_x):
    """Jumps that make progress past this obstacle, best (fastest) first."""
    scored = []
    for press in range(stall, max(stall - SEARCH_WINDOW, 0) - 1, -2):
        for dur in JUMP_DURATIONS:
            cand = (press, dur)
            st, cxs, _, _ = simulate(env, jumps + [cand],
                                     limit=min(stall + 450, 1500))
            if st == "flag":
                return [cand]  # done, nothing beats finishing
            peak = max(cxs)
            if peak > stall_x + 24:
                scored.append((peak, -plateau_frame(cxs), cand))
    scored.sort(reverse=True)
    return [c for _, _, c in scored[:CANDIDATES_PER_OBSTACLE]]


def main() -> None:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    jumps: list[tuple[int, int]] = []
    stack: list[tuple[list, list]] = []

    for _ in range(200):  # search iterations, includes backtracks
        status, xs, ys, time_left = simulate(env, jumps)
        if status == "flag":
            break
        stall, stall_x = plateau_frame(xs), max(xs)
        print(f"{status} at x={stall_x} (frame {stall}), searching...")
        cands = find_candidates(env, jumps, stall, stall_x)
        while not cands:
            if not stack:
                env.close()
                sys.exit(f"unsolvable at x={stall_x} even after backtracking")
            print("  dead end, backtracking")
            jumps, cands = stack.pop()
        chosen = cands.pop(0)
        stack.append((jumps, cands))
        jumps = jumps + [chosen]
        print(f"  -> jump at frame {chosen[0]} hold {chosen[1]}")
    else:
        env.close()
        sys.exit("search did not converge")

    status, xs, ys, time_left = simulate(env, jumps)
    assert status == "flag"
    env.close()

    y_cal = GROUND_TOP - ys[0]
    ghost = {
        "level": "1-1",
        "source": f"auto-tuned WR-style speedrun ({len(jumps)} jumps, "
                  f"game timer at flag: {time_left})",
        "fps": 60,
        "start_x": float(xs[0]),
        "flag_x": float(xs[-1]),
        "finish_frame": len(xs) - 1,
        "finish_time_s": round((len(xs) - 1) / 60.0, 2),
        "frames": [float(x) for x in xs],
        "y_top": [y + y_cal for y in ys],
        "jumps": sorted(jumps),
    }
    OUT.write_text(json.dumps(ghost))
    print(f"ghost written: {ghost['finish_time_s']}s to flag, {len(jumps)} jumps, "
          f"timer left {time_left}")


if __name__ == "__main__":
    main()
