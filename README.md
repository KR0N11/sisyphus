# mario-rl: AI vs the World Record Ghost

PPO learns Super Mario Bros World 1-1 **from scratch** (no pretrained weights), with a live wall of parallel emulator instances all racing a world-record-pace ghost, TrackMania-AI style.

Every Mario tile shows whether that instance is **ahead (green)** or **behind (red)** the ghost at the same point in its run, and a race bar at the bottom tracks the current leader against the ghost all the way to the flag.

## How it works

- **Environment**: `gym-super-mario-bros` (Gymnasium-native, runs a real NES emulator per instance via `nes-py`). N instances run in parallel subprocesses through Stable-Baselines3's `SubprocVecEnv`.
- **The AI**: Stable-Baselines3 PPO with a small CNN. It sees 4 stacked 84x84 grayscale frames (frame skip 4) and picks one of 7 `SIMPLE_MOVEMENT` actions. Community-proven hyperparameters: lr `1e-4`, gamma `0.9`, 512-step rollouts, 10 epochs per update.
- **Reward**: the env's built-in speed-biased reward (rightward velocity + clock penalty + death penalty), shaped with score-delta/40, +50 for the flag, -50 for dying, scaled by 1/10 (`src/mario_env.py`).
- **The ghost** (`data/ghost_1_1.json`): a frame-by-frame x-position trace at world-record pace, finishing 1-1 in **20.77s**. The bundled trace is a synthetic max-run-speed model (`scripts/make_ghost.py`); any real WR/TAS trace in the same JSON format is a drop-in replacement.
- **The display** (`src/hud.py`): full-color frames from every emulator are tiled into one mosaic with per-instance ghost deltas, a stats header (steps, episodes, flags, best clear time), and the leader-vs-ghost race bar.

## Setup

Requires Python 3.13+.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
python scripts/make_ghost.py
```

## Train

```
.venv/bin/python src/train.py --num-envs 10
```

- Match `--num-envs` to your CPU core count; emulators are CPU-bound and extra instances beyond that just time-slice.
- `--no-display` for headless training (a HUD snapshot is still written to `runs/latest.png` continuously).
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
