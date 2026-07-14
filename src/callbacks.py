"""Training callback that tracks per-env episode progress, clear times vs the
ghost, and renders the live HUD mosaic."""

from pathlib import Path

import cv2
from stable_baselines3.common.callbacks import BaseCallback

from ghost import Ghost
from hud import compose
from mario_env import FRAME_SKIP

SNAPSHOT = Path(__file__).resolve().parents[1] / "runs" / "latest.png"


class GhostRenderCallback(BaseCallback):
    def __init__(self, ghost: Ghost, num_envs: int, render_every: int = 8,
                 display: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.ghost = ghost
        self.num_envs = num_envs
        self.render_every = render_every
        self.display = display
        self.ep_steps = [0] * num_envs
        self.last_x = [0] * num_envs
        self.episodes = 0
        self.flags = 0
        self.best_clear_s: float | None = None

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i in range(self.num_envs):
            self.ep_steps[i] += 1
            self.last_x[i] = int(infos[i].get("x_pos", self.last_x[i]))
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

        if self.n_calls % self.render_every == 0:
            self._render()
        return True

    def _render(self) -> None:
        frames = self.training_env.get_images()
        states = [{"x": self.last_x[i], "frame": self.ep_steps[i] * FRAME_SKIP}
                  for i in range(self.num_envs)]
        stats = {"steps": self.num_timesteps, "episodes": self.episodes,
                 "flags": self.flags, "best": self.best_clear_s}
        img = compose(frames, states, self.ghost, stats)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SNAPSHOT), img)
        if self.display:
            cv2.imshow("mario-rl: AI vs WR ghost", img)
            cv2.waitKey(1)
