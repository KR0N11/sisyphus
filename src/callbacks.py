"""Training callback that tracks per-env episode progress, clear times vs the
ghost, and renders the live HUD mosaic.

Two display paces:
- fast (default): training runs unthrottled (the game runs faster than real
  time); the window refreshes ~10x/s.
- realtime: each vec step is paced to real NES speed (60 game fps) and every
  step renders, so the wall plays like actual gameplay. Training throughput
  drops roughly in half.
"""

import time
from collections import deque
from pathlib import Path

import cv2
from stable_baselines3.common.callbacks import BaseCallback

from ghost import Ghost
from hud import compose, with_banner
from mario_env import FRAME_SKIP

SNAPSHOT = Path(__file__).resolve().parents[1] / "runs" / "latest.png"
CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
FAST_RENDER_PERIOD = 0.1   # seconds between window refreshes in fast mode
SNAPSHOT_PERIOD = 2.0      # seconds between runs/latest.png writes
KEEP_RECENT_CKPTS = 5      # newest step-numbered checkpoints to keep
MILESTONE_STEPS = 1_000_000  # step-numbered checkpoints on these multiples survive


MAX_PROJECT_BYTES = 5 * 1024**3  # hard cap for checkpoints + runs combined
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _step_of(p: Path) -> int:
    return int(p.stem.split("_")[-2])


def prune_checkpoints() -> None:
    """Keep the newest N step checkpoints plus 1M-step milestones."""
    ckpts = sorted(CKPT_DIR.glob("ppo_mario_*_steps.zip"), key=_step_of)
    for p in ckpts[:-KEEP_RECENT_CKPTS]:
        if _step_of(p) % MILESTONE_STEPS:
            p.unlink(missing_ok=True)


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def enforce_storage_budget() -> None:
    """Hard 5GB cap on generated artifacts. Sacrifice order: old TensorBoard
    runs, then oldest step checkpoints (milestones included). The newest
    checkpoint and ppo_mario_latest.zip are never deleted."""
    def used() -> int:
        return sum(_dir_bytes(d) for d in (CKPT_DIR, RUNS_DIR) if d.exists())

    if used() <= MAX_PROJECT_BYTES:
        return
    tb_runs = sorted((RUNS_DIR / "tb").glob("PPO_*"), key=lambda p: p.stat().st_mtime)
    for old_tb in tb_runs[:-1]:
        for f in sorted(old_tb.rglob("*"), reverse=True):
            f.unlink(missing_ok=True) if f.is_file() else f.rmdir()
        old_tb.rmdir()
        if used() <= MAX_PROJECT_BYTES:
            return
    ckpts = sorted(CKPT_DIR.glob("ppo_mario_*_steps.zip"), key=_step_of)
    for p in ckpts[:-1]:
        p.unlink(missing_ok=True)
        if used() <= MAX_PROJECT_BYTES:
            return
    print(f"storage budget warning: still {used() / 1e9:.2f}GB after pruning")


class GhostRenderCallback(BaseCallback):
    def __init__(self, ghost: Ghost | None, num_envs: int, realtime: bool = False,
                 display: bool = True, level: str = "1-1", verbose: int = 0):
        super().__init__(verbose)
        self.ghost = ghost
        self.level = level
        self.num_envs = num_envs
        self.realtime = realtime
        self.display = display
        self._step_period = FRAME_SKIP / 60.0
        self._next_step_t: float | None = None
        self._last_render = 0.0
        self._last_snapshot = 0.0
        self.ep_steps = [0] * num_envs
        self.last_x = [0] * num_envs
        self.last_screen_x = [0] * num_envs
        self.episodes = 0
        self.flags = 0
        self.best_clear_s: float | None = None

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i in range(self.num_envs):
            self.ep_steps[i] += 1
            self.last_x[i] = int(infos[i].get("x_pos", self.last_x[i]))
            self.last_screen_x[i] = int(infos[i].get("left_x_pos", 0))
            if dones[i]:
                self.episodes += 1
                if infos[i].get("flag_get"):
                    self.flags += 1
                    clear_s = self.ep_steps[i] * FRAME_SKIP / 60.0
                    if self.best_clear_s is None or clear_s < self.best_clear_s:
                        self.best_clear_s = clear_s
                        if self.verbose:
                            ref = (f"(ghost: {self.ghost.finish_time_s:.2f}s)"
                                   if self.ghost else "")
                            print(f"[{self.level}] new best clear: {clear_s:.2f}s {ref}")
                self.ep_steps[i] = 0

        now = time.monotonic()
        if self.realtime:
            if self._next_step_t is None:
                self._next_step_t = now
            self._next_step_t += self._step_period
            delay = self._next_step_t - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:
                self._next_step_t = now  # fell behind (e.g. PPO update); resync
            self._render(now)
        elif now - self._last_render >= FAST_RENDER_PERIOD:
            self._render(now)
        return True

    def _on_rollout_start(self) -> None:
        self._rollout_start_eps = self.episodes
        self._next_step_t = None  # resync realtime pacing after the update
        prune_checkpoints()
        enforce_storage_budget()

    def _on_rollout_end(self) -> None:
        """Called right before the PPO update: label the pause on screen."""
        if not self.display:
            return
        runs = self.episodes - getattr(self, "_rollout_start_eps", 0)
        img = with_banner(self._compose(), f"STUDYING LAST {runs} RUNS...")
        cv2.imshow("mario-rl: AI vs WR ghost", img)
        cv2.waitKey(1)

    def _compose(self):
        frames = self.training_env.get_images()
        states = [{"x": self.last_x[i], "screen_x": self.last_screen_x[i],
                   "frame": self.ep_steps[i] * FRAME_SKIP}
                  for i in range(self.num_envs)]
        stats = {"steps": self.num_timesteps, "episodes": self.episodes,
                 "flags": self.flags, "best": self.best_clear_s,
                 "level": self.level}
        return compose(frames, states, self.ghost, stats)

    def _render(self, now: float) -> None:
        self._last_render = now
        img = self._compose()
        if now - self._last_snapshot >= SNAPSHOT_PERIOD:
            self._last_snapshot = now
            SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(SNAPSHOT), img)
        if self.display:
            cv2.imshow("mario-rl: AI vs WR ghost", img)
            cv2.waitKey(1)


class StopOnMastery(BaseCallback):
    """End training once the level is mastered: at least `min_flags` total
    clears and a clear rate of `rate` over the last `window` episodes."""

    def __init__(self, window: int = 100, rate: float = 0.5,
                 min_flags: int = 30, verbose: int = 1):
        super().__init__(verbose)
        self.recent = deque(maxlen=window)
        self.rate = rate
        self.min_flags = min_flags
        self.flags = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                cleared = bool(self.locals["infos"][i].get("flag_get"))
                self.recent.append(cleared)
                self.flags += cleared
        if (self.flags >= self.min_flags
                and len(self.recent) == self.recent.maxlen
                and sum(self.recent) / len(self.recent) >= self.rate):
            if self.verbose:
                print(f"mastery reached: {sum(self.recent)}/{len(self.recent)} "
                      f"recent episodes cleared, stopping level")
            return False
        return True
