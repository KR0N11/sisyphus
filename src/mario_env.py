"""Environment factory: SMB 1-1 with speed-focused reward shaping and
Atari-style preprocessing (frame skip 4, grayscale 84x84).

The policy sees small gray frames; the HUD grabs full-color frames separately
via VecEnv.get_images(), which bypasses the observation wrappers.
"""

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame

FRAME_SKIP = 4

# The bundled ROM uses PAL physics constants (max run speed 0x30, timer tick
# every 20 frames), i.e. it was balanced for 50fps. Pacing and time reporting
# use 50fps so gameplay speed and clear times match the real-world game.
GAME_FPS = 50


def env_id(level: str) -> str:
    return f"SuperMarioBros-{level}-v0"


class SpeedReward(gym.Wrapper):
    """Shaping on top of the env's built-in reward (x-velocity + clock + death).

    Adds score-delta/40 (stomps and pickups correlate with safe forward
    progress), +50 on flag, -50 on dying or timing out, then scales the total
    by 1/10 to keep magnitudes PPO-friendly.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._score = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += (info.get("score", 0) - self._score) / 40.0
        self._score = info.get("score", 0)
        if terminated or truncated:
            reward += 50.0 if info.get("flag_get") else -50.0
        return obs, reward / 10.0, terminated, truncated, info


class StuckTruncate(gym.Wrapper):
    """End the episode when Mario makes no forward progress for a while.
    Kills the 'stand still until the timer runs out' local optimum: idling
    now leads straight to the same -50 as dying, much sooner."""

    def __init__(self, env: gym.Env, max_idle_frames: int = 600):
        super().__init__(env)
        self.max_idle = max_idle_frames
        self._best_x = 0
        self._idle = 0

    def reset(self, **kwargs):
        self._best_x, self._idle = 0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = int(info.get("x_pos", 0))
        if x > self._best_x:
            self._best_x, self._idle = x, 0
        else:
            self._idle += 1
        if self._idle >= self.max_idle and not (terminated or truncated):
            truncated = True
            reward -= 50.0 / 10.0  # same failure signal SpeedReward gives (scaled)
        return obs, reward, terminated, truncated, info


def make_env(rank: int = 0, level: str = "1-1"):
    """Thunk for SubprocVecEnv: each subprocess builds its own emulator."""

    def _init() -> gym.Env:
        env = gym_super_mario_bros.make(env_id(level), render_mode="rgb_array")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SpeedReward(env)
        env = StuckTruncate(env)
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = WarpFrame(env)
        return env

    return _init
