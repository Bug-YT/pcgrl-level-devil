"""
player_env.py
==============
Gymnasium-Environment fuer den Player-Agenten.

Aktionsraum: Discrete(6) -> nur Kombinationen aus Space (Jump) und A/D
(Links/Rechts), siehe game_engine.N_ACTIONS. Alles andere (Physik,
Timing, etc.) lernt der Agent selbst.

Ein "Run" (= eine Episode) besteht aus mehreren "Versuchen" (Attempts):
stirbt der Spieler an einer Falle, wird er an den Start zurueckgesetzt
(Death-Counter +1), der Run laeuft weiter bis entweder das Goal erreicht
wird ODER max_attempts_per_run Tode erreicht sind. So bildet avg_deaths
pro Run direkt die Level-Devil-Schwierigkeit ab.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_engine import LevelDevilEngine, N_ACTIONS

OBS_DIM = 8


class PlayerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: dict, max_attempts_per_run: int = 30, max_steps_per_attempt: int = 400):
        super().__init__()
        self.level = level
        self.max_attempts_per_run = max_attempts_per_run
        self.max_steps_per_attempt = max_steps_per_attempt

        self.engine = LevelDevilEngine(level)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.deaths = 0
        self.steps_in_attempt = 0
        self.total_steps_in_run = 0
        self.won = False

    def set_level(self, level: dict) -> None:
        self.level = level
        self.engine = LevelDevilEngine(level)

    def _obs(self) -> np.ndarray:
        e = self.engine
        w, h = e.width, e.height
        dx = (e.goal_x - e.x) / max(w, 1)
        dy = (e.goal_y - e.y) / max(h, 1)
        nearest_trap_dx, nearest_trap_dy = self._nearest_trap_offset()
        obs = np.array(
            [
                (e.x / w) * 2 - 1,
                (e.y / h) * 2 - 1,
                np.clip(e.vx * 4, -1, 1),
                np.clip(e.vy * 4, -1, 1),
                1.0 if e.on_ground else -1.0,
                np.clip(dx, -1, 1),
                np.clip(dy, -1, 1),
                np.clip(nearest_trap_dx, -1, 1),
            ],
            dtype=np.float32,
        )
        return obs

    def _nearest_trap_offset(self) -> tuple[float, float]:
        from level_schema import TILE_TRAP

        e = self.engine
        best_d = None
        best = (1.0, 1.0)
        ys, xs = np.where(e.grid == TILE_TRAP)
        for gx, gy in zip(xs, ys):
            d = (gx - e.x, gy - e.y)
            dist = abs(d[0]) + abs(d[1])
            if best_d is None or dist < best_d:
                best_d = dist
                best = (d[0] / max(e.width, 1), d[1] / max(e.height, 1))
        return best

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.deaths = 0
        self.steps_in_attempt = 0
        self.total_steps_in_run = 0
        self.won = False
        return self._obs(), {}

    def step(self, action: int):
        result = self.engine.step(int(action))
        self.steps_in_attempt += 1
        self.total_steps_in_run += 1

        reward = -0.001  # kleine Zeitstrafe pro Schritt
        prev_dx = abs(self.engine.goal_x - result.x)
        reward += 0.001 * (self.engine.width - prev_dx) / self.engine.width  # Fortschritt Richtung Ziel

        terminated = False
        truncated = False

        if result.won:
            reward += 10.0
            self.won = True
            terminated = True
        elif result.died or self.steps_in_attempt >= self.max_steps_per_attempt:
            reward -= 5.0
            self.deaths += 1
            self.engine.reset()
            self.steps_in_attempt = 0
            if self.deaths >= self.max_attempts_per_run:
                terminated = True  # Run vorbei: nicht geschafft

        info = {}
        if terminated:
            info = {
                "run_won": self.won,
                "run_deaths": self.deaths,
                "run_steps": self.total_steps_in_run,
            }

        return self._obs(), reward, terminated, truncated, info
