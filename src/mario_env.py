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

LEVEL = "SuperMarioBros-1-1-v0"
FRAME_SKIP = 4


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


def make_env(rank: int = 0):
    """Thunk for SubprocVecEnv: each subprocess builds its own emulator."""

    def _init() -> gym.Env:
        env = gym_super_mario_bros.make(LEVEL, render_mode="rgb_array")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SpeedReward(env)
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = WarpFrame(env)
        return env

    return _init
