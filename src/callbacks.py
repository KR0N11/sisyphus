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
from pathlib import Path

import cv2
from stable_baselines3.common.callbacks import BaseCallback

from ghost import Ghost
from hud import compose, with_banner
from mario_env import FRAME_SKIP

SNAPSHOT = Path(__file__).resolve().parents[1] / "runs" / "latest.png"
FAST_RENDER_PERIOD = 0.1   # seconds between window refreshes in fast mode
SNAPSHOT_PERIOD = 2.0      # seconds between runs/latest.png writes


class GhostRenderCallback(BaseCallback):
    def __init__(self, ghost: Ghost, num_envs: int, realtime: bool = False,
                 display: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.ghost = ghost
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
                            print(f"new best clear: {clear_s:.2f}s "
                                  f"(ghost: {self.ghost.finish_time_s:.2f}s)")
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
                 "flags": self.flags, "best": self.best_clear_s}
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
