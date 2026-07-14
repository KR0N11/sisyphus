# mario-rl: AI vs the World Record Ghost

PPO learns Super Mario Bros World 1-1 **from scratch** (no pretrained weights), with a live wall of parallel emulator instances all racing a world-record-pace ghost, TrackMania-AI style.

A **translucent ghost Mario** runs inside every tile at world-record pace (when he's on camera), so you can see each AI literally racing him. Tile borders show whether that instance is **ahead (green)** or **behind (red)** the ghost at the same point in its run, and a race bar at the bottom tracks the current leader against the ghost all the way to the flag.

## How it works

- **Environment**: `gym-super-mario-bros` (Gymnasium-native, runs a real NES emulator per instance via `nes-py`). N instances run in parallel subprocesses through Stable-Baselines3's `SubprocVecEnv`.
- **The AI**: Stable-Baselines3 PPO with a small CNN. It sees 4 stacked 84x84 grayscale frames (frame skip 4) and picks one of 7 `SIMPLE_MOVEMENT` actions. Community-proven hyperparameters: lr `1e-4`, gamma `0.9`, 512-step rollouts, 10 epochs per update.
- **Reward**: the env's built-in speed-biased reward (rightward velocity + clock penalty + death penalty), shaped with score-delta/40, +50 for the flag, -50 for dying, scaled by 1/10 (`src/mario_env.py`).
- **The ghost** (`data/ghost_1_1.json`): a frame-by-frame x/y trace of a WR-style speedrun. `scripts/make_pro_run.py` generates it by auto-tuning jump timings inside the emulator (hold max run speed, search the fastest jump for every pipe, pit, and staircase, with backtracking when a jump line dead-ends), so the ghost jumps obstacles the way a record run does. `scripts/record_tas_ghost.py` can instead replay a real .fm2 TAS movie, but frame-perfect TAS inputs desync between emulators (nes-py vs FCEUX lag-frame timing), so the auto-tuned run is the default. `scripts/make_ghost.py` makes a simple straight-line pace ghost as a fallback.
- **The display** (`src/hud.py`): full-color frames from every emulator are tiled into one mosaic. Each tile gets the low-opacity ghost Mario sprite drawn at the ghost's world position (camera offset derived from the env's `left_x_pos`), plus per-instance ghost deltas, a stats header (steps, episodes, flags, best clear time), and the leader-vs-ghost race bar. The sprite itself is extracted from the emulator at runtime by `scripts/make_ghost_sprite.py` (Mario jumps, we grab him mid-air against clean sky).

## Pretrained model (skip training)

Trained checkpoints are published on the repo's **Releases** page so nobody has to retrain from scratch: download the `.zip` into `checkpoints/`, then run `src/watch.py` (spectate) or `src/evaluate.py` (measure clear times). Training picks up from it with `--resume checkpoints/<file>.zip`.

Generated artifacts (checkpoints + logs) are hard-capped at **5GB** locally: oldest TensorBoard runs and checkpoints are auto-deleted first, and the newest model is never touched.

## Setup

Requires Python 3.13+.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
python scripts/make_ghost.py
```

## Train

```
.venv/bin/python src/train.py --num-envs 10 --realtime --device mps
```

- `--realtime` paces the game to true NES speed so the wall is watchable (training runs ~2x slower). Omit it for max-speed learning (Marios fast-forward).
- `--device mps` runs the network on Apple GPU, which shrinks the between-rollout "STUDYING..." pauses. PPO must pause play to learn: it collects 512 decisions per instance, studies that batch 10 times, discards it, and resumes, that pause is fundamental to on-policy RL, not a hang.
- Match `--num-envs` to your CPU core count; emulators are CPU-bound and extra instances beyond that just time-slice.
- `--no-display` for headless training (a HUD snapshot is still written to `runs/latest.png` every couple of seconds).
- Checkpoints land in `checkpoints/` every 100k steps; resume anytime with `--resume checkpoints/<file>.zip`.
- TensorBoard logs: `runs/tb`.

## Evaluate

```
.venv/bin/python src/evaluate.py
```

Runs the newest checkpoint deterministically and prints each episode's clear time against the ghost.

## Showcase (the 30-Mario wall)

```
.venv/bin/python src/watch.py --num-envs 30
```

Loads a checkpoint into a big wall of instances at real-time speed, no training, pure spectacle. Press `q` to quit.

## Roadmap

1. Reliable 1-1 completion (current phase)
2. Speed phase: heavier time pressure in the reward, chase the ghost
3. Real WR/TAS input trace as the ghost
