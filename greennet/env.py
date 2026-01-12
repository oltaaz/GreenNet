"""Gymnasium environment wrapper for GreenNet."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    """Configuration for the GreenNet environment."""

    max_steps: int = 1000


class GreenNetEnv(gym.Env):
    """Minimal Gymnasium-compatible environment interface.

    This class is intentionally lightweight; wire in the simulator, routing,
    and observation builder once those are implemented.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()

        self.config = config or EnvConfig()
        self._step_count = 0

        # Placeholder spaces; update with real observations/actions.
        self.observation_space = spaces.Dict({
            "time": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(1)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        observation = {"time": np.array([0.0], dtype=np.float32)}
        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        observation = {
            "time": np.array([min(1.0, self._step_count / self.config.max_steps)], dtype=np.float32)
        }
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.config.max_steps
        info: Dict[str, Any] = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


if __name__ == "__main__":
    from greennet.env import GreenNetEnv

    env = GreenNetEnv()
    obs, info = env.reset()
    print("reset OK. obs keys:", list(obs.keys()) if hasattr(obs, "keys") else type(obs), "info:", info)

    for t in range(10):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        print(f"t={t} r={r:.3f} term={terminated} trunc={truncated} info_keys={list(info.keys())}")

    print("Smoke test OK ✅")