# Session Handoff (updated 2026-07-15, paused mid-campaign)

## 2026-07-16 update (most recent)

- 1-2 BROKE THROUGH after the anti-stuck fixes: best clear 22.40s vs ghost 21.72s when paused (was permanently stuck at x≈978 before). Not yet mastered (no `.mastered` marker); resume continues it automatically.
- Resume command unchanged: `nohup nice -n 10 .venv/bin/python -u scripts/campaign.py --device mps --num-envs 6 > runs/campaign.log 2>&1 &` (add `--no-display` for headless).

## 2026-07-15 update (supersedes "Where things stand" below where they conflict)

- Campaign switched to FAST mode per user (no --realtime) and LOW-HEAT profile: `--num-envs 6 --no-display` (user's laptop was overheating; RAM kept modest).
- **1-2 is NOT cleared yet.** It got stuck at x≈978 (local optimum: standing still). Fixes shipped and active: `StuckTruncate` wrapper (idle >600 frames = truncated with failure penalty) and `--ent-coef 0.03` on transferred brains. 1-2 has ~1.09M total steps in `ppo_mario_1-2_latest.zip` (includes 450k of post-fix training, zero clears so far when paused).
- **Mastery markers**: campaign now skips a level only if `checkpoints/ppo_mario_<level>.mastered` exists (1-1 has one); unfinished levels resume their own `_latest`. The old "skips any level with a checkpoint" gotcha is fixed.
- 4-1 ghost tuning was interrupted; `data/ghost_4_1.json` does NOT exist yet (campaign will re-tune it automatically).
- Resume command (fast, low-heat): `nohup nice -n 10 .venv/bin/python -u scripts/campaign.py --device mps --num-envs 6 --no-display > runs/campaign.log 2>&1 &`
- Plan after all levels clear: real-time relay showcase (user wants runs sped up during training, real time for the final watch).
- If 1-2 still won't clear after the current fixes get a fair shot (~1M more steps), next escalation: exploration bonus for new max-x milestones per episode.

---

# Original handoff (2026-07-14 evening)

Read this to continue the project from exactly where it stopped, in any session or tool. Full background and concepts: `docs/JOURNEY.md`. Original research plan: `PLAN.md`.

## Where things stand

- **1-1 is DONE and beat the WR ghost**: AI 23.44s vs ghost 23.54s (real-world seconds). Model: `checkpoints/ppo_mario_1-1_record.zip` (also `ppo_mario_1-1_latest.zip`, and published as GitHub release `model-v1.0`). Trophy screenshot: `docs/record-19.53s.png` (its header shows the pre-correction time).
- **Whole-game campaign is mid-flight**, paused during level 1-2:
  - 1-2's ghost is tuned and committed (`data/ghost_1_2.json`, 21.7s real, 13 jumps).
  - 1-2 training had only just started. `checkpoints/ppo_mario_1-2_latest.zip` exists but is nearly untrained.
- Everything is committed and pushed to `github.com/KR0N11/sisyphus` (private).

## How to resume

From `~/sisyphus`, the campaign command is:

```
nohup nice -n 10 .venv/bin/python -u scripts/campaign.py --realtime --device mps > runs/campaign.log 2>&1 &
```

**Gotcha first**: the campaign skips any level that has a `ppo_mario_<level>_latest.zip`, and 1-2's exists but is barely trained. Either delete `checkpoints/ppo_mario_1-2_latest.zip` before launching (it restarts 1-2 with transfer from 1-1), or train 1-2 manually first:

```
.venv/bin/python src/train.py --level 1-2 --stop-on-mastery --realtime --device mps --resume checkpoints/ppo_mario_1-2_latest.zip
```

Route: 1-1 (done), 1-2, 4-1, 4-2, 8-1, 8-2, 8-3, 8-4. Each level transfers from the previous level's brain and auto-advances at mastery (50% clear rate over the last 100 runs).

## Critical technical facts (hard-won, do not rediscover)

1. **The bundled ROM has PAL physics** (max run speed 0x30 = 3.0 px/frame, timer ticks every 20 frames) emulated at 60fps. Therefore **real-world seconds = frames / 50**, everywhere (`GAME_FPS = 50` in `src/mario_env.py`). Any time computed with /60 is 20% too flattering. This is also why NTSC TAS input files (`scripts/record_tas_ghost.py`) can never sync: different physics.
2. **SB3 resume gotcha**: `learn(..., reset_num_timesteps=False)` already adds the resumed step count to `total_timesteps` internally. Never pre-add it (doing so once caused a runaway 50k-step "smoke test").
3. **In-game timer is not seconds**: it ticks every 20 frames (2.5x real seconds at 50fps). Do not use it for timing claims.
4. **Python env**: the venv is uv-managed (`~/.local/bin/uv`); Homebrew's python3.13 is broken (pyexpat/libexpat mismatch), do not rebuild the venv with it.
5. **Emulators are CPU-bound**: 10 parallel envs (= core count) is the training sweet spot on this M5; 30 is demo-only.
6. Storage self-manages: newest 5 checkpoints + 1M-step milestones kept, hard 5GB fuse on generated artifacts (`src/callbacks.py`).

## Open design decisions (user input wanted)

- **Warp brains**: a true warps-route continuous run needs 1-2 and 4-2 trained to *enter the warp zones* instead of reaching their flags (reward: reach world 4 / world 8 instead of flag_get). Queued after the normal-exit levels prove out. This is the hardest RL piece remaining.
- **The finale**: a "relay run", one continuous playthrough handing control to each level's brain at level boundaries, producing a start-to-finish "AI beats SMB" video.
- Repo is **private**; user may want it public later for portfolio use.

## Quick commands

- Watch the 1-1 record: `.venv/bin/python src/watch.py --level 1-1 --model checkpoints/ppo_mario_1-1_record.zip`
- Measure a model: `.venv/bin/python src/evaluate.py --level <lvl>`
- Live snapshot during any training: `runs/latest.png` (refreshes every 2s)
- Tune a ghost for a new level: `.venv/bin/python scripts/make_pro_run.py --level <lvl>` (works on run-right levels; castles/water fail harmlessly and train ghost-less)
