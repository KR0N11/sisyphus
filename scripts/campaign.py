"""Beat Super Mario Bros the speedrunner way: the warps route, one PPO brain
per level, each level starting from the previous level's brain (transfer
learning).

For each level in the route:
1. skip it if checkpoints/ppo_mario_<level>_latest.zip already exists
2. try to auto-tune a WR-style ghost (fails harmlessly on castle/water levels)
3. train with --stop-on-mastery until >=50% of the last 100 runs clear it,
   or the per-level step budget runs out
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
WARPS_ROUTE = ["1-1", "1-2", "4-1", "4-2", "8-1", "8-2", "8-3", "8-4"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", default=",".join(WARPS_ROUTE),
                   help="comma-separated levels, in order")
    p.add_argument("--steps-per-level", type=int, default=1_500_000)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args()

    prev_model: Path | None = None
    for level in args.route.split(","):
        level = level.strip()
        latest = ROOT / "checkpoints" / f"ppo_mario_{level}_latest.zip"
        if latest.exists():
            print(f"[{level}] already trained, skipping")
            prev_model = latest
            continue

        ghost_file = ROOT / "data" / f"ghost_{level.replace('-', '_')}.json"
        if not ghost_file.exists():
            print(f"[{level}] tuning ghost run...")
            r = subprocess.run([str(PYTHON), str(ROOT / "scripts" / "make_pro_run.py"),
                                "--level", level])
            if r.returncode:
                print(f"[{level}] no ghost possible (expected for castles/water); "
                      f"training without one")

        cmd = [str(PYTHON), str(ROOT / "src" / "train.py"),
               "--level", level, "--stop-on-mastery",
               "--total-steps", str(args.steps_per_level),
               "--num-envs", str(args.num_envs), "--device", args.device]
        if args.realtime:
            cmd.append("--realtime")
        if args.no_display:
            cmd.append("--no-display")
        if prev_model:
            cmd += ["--resume", str(prev_model)]
            print(f"[{level}] training with transfer from {prev_model.name}")
        else:
            print(f"[{level}] training from scratch")

        if subprocess.run(cmd).returncode:
            sys.exit(f"[{level}] training failed, campaign stopped")
        prev_model = latest
        print(f"[{level}] done -> {latest.name}")

    print("CAMPAIGN COMPLETE: every level on the route has a trained brain")


if __name__ == "__main__":
    main()
