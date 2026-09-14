"""
player_env.py
==============
Gymnasium-Environment fuer den Player-Agenten.

Aktionsraum: Discrete(6) -> nur Kombinationen aus Space (Jump) und A/D
(Links/Rechts), siehe game_engine.N_ACTIONS. Alles andere (Physik,
Timing, etc.) lernt der Agent selbst.

Ein "Run" (= eine Episode) besteht aus mehreren "Versuchen" (Attempts):
stirbt der Spieler an einer Falle/Buzzsaw, wird er an den Start
zurueckgesetzt (Death-Counter +1), der Run laeuft weiter bis entweder das
Goal erreicht wird ODER max_attempts_per_run Tode erreicht sind. So bildet
avg_deaths pro Run direkt die Level-Devil-Schwierigkeit ab.

Die Observation enthaelt neben Position/Geschwindigkeit auch die aktuelle
Gravitationsrichtung (wichtig nach einem Gravity-Flip-Trigger, sonst kann
der Agent die invertierte Steuerung nicht lernen) sowie die relative
Position der naechsten Falle UND des naechsten Buzzsaws (bewegliche
Hindernisse lassen sich nicht aus dem statischen Level-Layout allein
vorhersagen).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_engine import LevelDevilEngine, N_ACTIONS
from level_schema import TILE_TRAP

OBS_DIM = 11
_TRAP_TYPES = (TILE_TRAP,)


class PlayerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        level: dict,
        max_attempts_per_run: int = 30,
        max_steps_per_attempt: int = 400,
        gravity: float | None = None,
        move_speed: float | None = None,
        jump_velocity: float | None = None,
        max_fall_speed: float | None = None,
    ):
        super().__init__()
        self.level = level
        self.max_attempts_per_run = max_attempts_per_run
        self.max_steps_per_attempt = max_steps_per_attempt
        # None => game_engine-Standardwerte verwenden (siehe LevelDevilEngine)
        self._physics_kwargs = {
            k: v
            for k, v in dict(
                gravity=gravity, move_speed=move_speed, jump_velocity=jump_velocity, max_fall_speed=max_fall_speed
            ).items()
            if v is not None
        }

        self.engine = LevelDevilEngine(level, **self._physics_kwargs)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.deaths = 0
        self.steps_in_attempt = 0
        self.total_steps_in_run = 0
        self.won = False

    def set_level(self, level: dict) -> None:
        self.level = level
        self.engine = LevelDevilEngine(level, **self._physics_kwargs)

    def _obs(self) -> np.ndarray:
        e = self.engine
        w, h = e.width, e.height
        dx = (e.goal_x - e.x) / max(w, 1)
        dy = (e.goal_y - e.y) / max(h, 1)
        trap_dx, trap_dy = self._nearest_match_offset(_TRAP_TYPES)
        saw_dx = self._nearest_buzzsaw_offset()
        obs = np.array(
            [
                (e.x / w) * 2 - 1,
                (e.y / h) * 2 - 1,
                np.clip(e.vx * 4, -1, 1),
                np.clip(e.vy * 4, -1, 1),
                1.0 if e.on_ground else -1.0,
                np.clip(dx, -1, 1),
                np.clip(dy, -1, 1),
                np.clip(trap_dx, -1, 1),
                np.clip(trap_dy, -1, 1),
                np.clip(saw_dx, -1, 1),
                float(e.gravity_sign),
            ],
            dtype=np.float32,
        )
        return obs

    def _nearest_match_offset(self, tile_types) -> tuple[float, float]:
        e = self.engine
        best_d = None
        best = (1.0, 1.0)
        mask = np.isin(e.grid, list(tile_types))
        ys, xs = np.where(mask)
        for gx, gy in zip(xs, ys):
            d = (gx - e.x, gy - e.y)
            dist = abs(d[0]) + abs(d[1])
            if best_d is None or dist < best_d:
                best_d = dist
                best = (d[0] / max(e.width, 1), d[1] / max(e.height, 1))
        return best

    def _nearest_buzzsaw_offset(self) -> float:
        e = self.engine
        if not e._buzzsaws:
            return 1.0
        best_d = None
        best_dx = 1.0
        for saw in e._buzzsaws:
            d = saw["x"] - e.x
            dist = abs(d)
            if best_d is None or dist < best_d:
                best_d = dist
                best_dx = d / max(e.width, 1)
        return best_dx

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Ein komplett neues Level (Runde) soll auch die Laufzeit-
        # Mutationen des Vorlaufs (kollabierte Fake-Floors, eingesammelte
        # Coins, ausgeloeste Fallen, geflipte Gravitation, gefluechtetes
        # Ziel) zuruecksetzen -- also die Engine ganz neu aufbauen statt
        # nur engine.reset() (das erhaelt Mutationen bewusst INNERHALB
        # eines Runs, siehe game_engine.LevelDevilEngine.reset).
        self.engine = LevelDevilEngine(self.level, **self._physics_kwargs)
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
            # Nur die Spielerposition zuruecksetzen -- Laufzeit-Mutationen
            # (kollabierte Boeden, eingesammelte Coins, ausgeloeste Fallen,
            # geflipte Gravitation, gefluechtetes Ziel) bleiben INNERHALB
            # desselben Runs bestehen, damit wiederholte Versuche auf
            # DEMSELBEN, bereits veraenderten Level stattfinden.
            self.engine.reset()
            self.steps_in_attempt = 0
            if self.deaths >= self.max_attempts_per_run:
                terminated = True  # Run vorbei: nicht geschafft

        # Bei JEDEM Schritt (nicht nur am Episodenende) den Render-State
        # mitschicken, damit eine Live-GUI Fake-Floor-Kollaps, Coin-
        # Sammlung, Buzzsaw-Bewegung, Gravity-Flip und ein gefluechtetes
        # Ziel auch dann sehen kann, wenn die Engine in einem separaten
        # Prozess (SubprocVecEnv) laeuft.
        info = {"engine_diff": self.engine.get_render_state()}
        if terminated:
            info["run_won"] = self.won
            info["run_deaths"] = self.deaths
            info["run_steps"] = self.total_steps_in_run

        return self._obs(), reward, terminated, truncated, info
