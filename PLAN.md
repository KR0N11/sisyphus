# Mario 1-1 RL Project: Plan & Research Doc

Goal: train an AI **from scratch** (no imported/pretrained model) to complete Super Mario Bros World 1-1, with a live tiled window showing every parallel Mario instance learning in real time, running on your Mac (Apple M5, 10 cores, Python 3.13).

Research method: deep-research pass over 20 sources (PyPI, GitHub repos, Stable-Baselines3 docs, blogs), 97 claims extracted, top 25 adversarially fact-checked with 3 verifier votes each. 23 confirmed, 2 refuted. Everything below marked "verified" survived that process.

---

## 1. The headline finding (this changes the plan for the better)

The classic Mario RL stack was **dead for 4 years and got revived one month ago**:

- `gym-super-mario-bros` **9.1.0** (released 2026-06-10) and `nes-py` **9.0.1** are now **Gymnasium-native**, actively maintained, and ship **prebuilt Apple Silicon (arm64) wheels**. No compiling, no M1-era build hacks. *(verified 3-0 against PyPI)*
- Catch: they require **Python 3.13+**. You have 3.13.14, so you're in the clear. On older Pythons, pip silently installs the broken 2022 version, a classic trap to know about.
- Old internet advice ("it doesn't install on Apple Silicon", "the package is abandoned") is now **stale**. Ignore any tutorial complaining about `-march=native` build errors.

**Fallback options** (only if the new stack misbehaves):
- `stable-retro` 1.0.1: the maintained Farama fork of gym-retro, NES supported, arm64 wheels, but you must supply the SMB ROM yourself. *(verified 3-0)*
- `tooichitake/gymnasium-mario` forks: for Python 3.10-3.12 environments. *(weaker evidence, 2-1 vote, last updated Nov 2025)*

## 2. The stack

| Piece | Choice | Why |
|---|---|---|
| Emulator + env | `gym-super-mario-bros` 9.1.0 / `nes-py` 9.0.1 | Gymnasium-native, arm64 wheels, `SuperMarioBros-1-1-v0` env built in |
| RL algorithm | PPO via `stable-baselines3` (SB3) | Community standard for SMB, handles parallel envs natively |
| Parallelism | SB3 `SubprocVecEnv` | One emulator per process, all stepped together |
| Live display | SB3 `VecEnv.render()` | See section 5: the tiled wall of Marios is built in |
| Framework | PyTorch (comes with SB3) | Can use Apple's MPS GPU backend; CPU is fine too, the network is tiny |

## 3. How the AI sees and controls the game (preprocessing)

This is the standard recipe confirmed across every serious SMB repo *(verified 3-0)*:

- **Grayscale + downscale to 84x84**: Mario doesn't need color or HD to spot a goomba. Cuts the input ~40x.
- **Frame skip 4 (with max-pooling)**: the AI picks an action every 4th frame and holds it. 4x faster learning, and no human presses buttons 60x/sec either.
- **Frame stack 4**: the AI sees the last 4 frames at once. One frame can't show velocity; four can.
- **Reduced action space via `JoypadSpace`**: instead of all 256 button combos, give it `SIMPLE_MOVEMENT` (7 actions: noop, right, right+A, right+B, right+A+B, A, left). For pure 1-1 speed you can even try `RIGHT_ONLY` (5 actions).

So the network's actual input is a tiny 4x84x84 stack of gray images, and its output is 1 of 7 choices. That's why this trains on a laptop.

## 4. PPO settings (proven, not guessed)

Taken from the ~1.1k-star `uvipen/Super-mario-bros-PPO-pytorch` repo, verified against its actual source code *(3-0)*:

| Hyperparameter | Value | Note |
|---|---|---|
| Learning rate | `1e-4` | The "how big are the updates" knob |
| Gamma (discount) | `0.9` | Notably lower than the Atari default 0.99; short-horizon game |
| GAE lambda | `1.0` | Advantage estimation smoothing |
| Entropy coef | `0.01` | Keeps it exploring instead of settling early |
| Clip epsilon | `0.2` | PPO's "don't change the policy too fast" limit |
| Rollout length | `512` steps per env per update | |
| Epochs per update | `10` | |
| Step budget | `5e6` (5M) | That repo's default total training cap |

We'll use these as the starting config in SB3 rather than tuning from zero.

## 5. The wall of Marios (live display)

Best engineering news of the research: **zero custom display code needed** *(verified 3-0 against SB3 docs and source)*:

- Create every sub-env with `render_mode="rgb_array"`.
- Call `vec_env.render(mode="human")` during training (via a small callback).
- SB3 collects a frame from every subprocess, tiles them into a near-square grid with its built-in `tile_images` utility, and shows the mosaic in an OpenCV window automatically.

We'll add a throttle so it renders the mosaic every Nth update (rendering every single frame slows training for nothing).

## 6. How many Marios? (the 30-instance reality check on your M5)

- SB3's own guidance *(verified 3-0)*: for CPU-bound envs like an emulator, **don't exceed your logical core count**. Your M5 has **10 cores** (4 performance + 6 efficiency).
- 30 instances **will run**, but they time-slice the same 10 cores, so total frames/sec is not better than ~10 instances, and can be worse due to overhead.

**Plan:** make instance count a config value.
- **Training mode: 10 envs** (matches cores, max learning speed).
- **Showcase mode: 20-30 envs** when you want the visual wall for a demo, accepting slower stepping.
- The env count doesn't change the AI itself, so you can train at 10 and demo at 30 with the same saved checkpoint.

## 7. Reward design (what the AI gets paid for)

Two layers:

**Built-in reward** (comes with the env, already speed-biased): `r = v + c + d`, where `v` = rightward velocity, `c` = clock penalty (time passing hurts), `d` = -15 on death, clipped to (-15, 15).

**Shaping layer on top** (verified from the uvipen repo *(3-0)*, we port it to Gymnasium's API):
- `+ (score delta) / 40`: stomps and pickups correlate with safe forward progress
- `+ 50` on flag grab
- `- 50` on dying or timing out without the flag
- divide total by 10 to keep numbers small (PPO trains better on small rewards)

Phase 2 (speed tuning, after it completes reliably): increase the time penalty and/or add a finish bonus scaled by the in-game timer remaining. Note: no verified source covered speedrun-grade shaping, so this part is experimentation, not recipe.

## 8. Honest answer on training time

This was the weakest area of the research. The two concrete "it took X" claims **failed fact-checking** (one repo's "beat 31/32 levels" claim: refuted 1-2; another's budget numbers: refuted 0-3). What survives:

- The proven config caps training at **5M steps** and solves individual levels within it.
- Unverified practitioner datapoints for calibration only: ~10M steps / ~4.5 hours for a 1-1-completing SB3 PPO model; ~20 hours for an older DQN approach on cloud GPU.
- With 10 parallel envs + frame skip on your M5, a reasonable **expectation** (not promise): **first flag within a few hours of wall clock, consistent completion within an overnight run.** "Perfect" (WR-style) is open-ended and not this phase's goal.

We'll log steps/sec in the first 10 minutes of training and extrapolate real numbers from your machine instead of trusting anyone's blog.

## 9. Build plan (phases)

1. **Env sanity check**: install stack, boot one instance of 1-1, random actions, confirm rendering and `info` dict (x_pos, score, flag_get) all work on macOS.
2. **Vectorize**: 10 subprocess envs + preprocessing wrappers + the tiled render callback. Random actions, watch 10 Marios die simultaneously. Milestone: the wall works.
3. **Train**: SB3 PPO with section 4 config + section 7 rewards. Checkpoint every ~100k steps, TensorBoard logging so you can watch reward curves.
4. **Evaluate**: script that loads the latest checkpoint and runs it deterministically, reporting completion rate and in-game time.
5. **(Later) Speed phase**: reward reshaping for time, longer training, chase the clock.

## 10. Risks & open questions (from the research)

- **Untested integration**: nobody has publicly confirmed gym-super-mario-bros 9.1.0 + SB3 SubprocVecEnv end-to-end on Python 3.13. The revival is 1 month old. If APIs clash, fallback is stable-retro. Phase 1 exists to catch this early and cheap.
- The uvipen reward/skip wrappers target the old gym API (4-tuple step). Porting to Gymnasium's 5-tuple (`terminated`/`truncated`) is a known, small task.
- Whether SubprocVecEnv frame-piping becomes a display bottleneck at 30 envs: unknown, hence the render throttle.
- MPS (Apple GPU) vs CPU for the PPO network: untested for this workload; we benchmark both in phase 3.

## 11. Key sources

- PyPI: [nes-py](https://pypi.org/project/nes-py/), [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/) (versions/wheels verified live)
- [uvipen/Super-mario-bros-PPO-pytorch](https://github.com/uvipen/Super-mario-bros-PPO-pytorch) (hyperparameters, reward shaping, preprocessing)
- [SB3 Vectorized Environments docs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) (tiled rendering, core-count guidance)
- [Farama-Foundation/stable-retro](https://github.com/Farama-Foundation/stable-retro) (fallback path)
- [Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) (built-in reward function definition)
