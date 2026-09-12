"""
builder_env.py
================
Gymnasium-Environment fuer den Builder-Agenten.

Start- und Ziel-Tile sind (wie im Beispiel-Level) fix am linken/rechten
Rand des Levels platziert -- das ist Level-Devil-Konvention. Der Builder
lernt die interessante Entscheidung: WO Spikes/Traps platziert werden,
um moeglichst nah an die Ziel-Schwierigkeit (TARGET_DIFFICULTY, gemessen
in avg_deaths) heranzukommen.

Eine Episode = EIN generiertes Level ("one-shot"): Der Builder gibt in
einem einzigen step() einen Aktionsvektor aus, der in Spike-Positionen
decodiert wird. step() fuehrt daraufhin selbst den kompletten Rest des
Loops aus (Export -> Lint -> Load -> Play -> Feedback) und liefert erst
danach den Reward zurueck. So bekommt der Builder pro step() ein
vollstaendiges Feedback-Signal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from level_schema import TILE_GOAL, TILE_START, TILE_TRAP

OBS_DIM = 6  # target_norm, last_winrate, last_avg_deaths_norm, last_avg_time_norm, last_score_norm, cycle_norm

PlayAndScoreFn = Callable[[dict], tuple[float, dict]]
ExportFn = Callable[[dict], tuple[bool, list[str], str]]  # -> (lint_ok, errors, filepath)


class BuilderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        width: int,
        height: int,
        max_traps: int,
        target_difficulty: float,
        export_and_lint_fn: ExportFn,
        play_and_score_fn: PlayAndScoreFn,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_traps = max_traps
        self.target_difficulty = target_difficulty
        self.export_and_lint_fn = export_and_lint_fn
        self.play_and_score_fn = play_and_score_fn

        # Pro Trap-Slot: [aktiv?(-1..1), x_position(-1..1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max_traps * 2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.last_feedback = {
            "winrate": 0.5,
            "avg_deaths": target_difficulty,
            "avg_time": 0.0,
            "score": 0.0,
        }
        self.cycle = 0
        self.last_level: dict | None = None
        self.last_level_path: str | None = None
        self.last_lint_ok: bool = True
        self.last_lint_errors: list[str] = []

    def update_context(self, feedback: dict, cycle: int) -> None:
        """Wird von der Pipeline nach jedem Zyklus aufgerufen (read Feedback)."""
        self.last_feedback = feedback
        self.cycle = cycle

    def _obs(self) -> np.ndarray:
        fb = self.last_feedback
        return np.array(
            [
                np.clip(self.target_difficulty / 50.0, 0, 1) * 2 - 1,
                np.clip(fb.get("winrate", 0.5), 0, 1) * 2 - 1,
                np.clip(fb.get("avg_deaths", 0.0) / 50.0, 0, 1) * 2 - 1,
                np.clip(fb.get("avg_time", 0.0) / 400.0, 0, 1) * 2 - 1,
                np.clip(fb.get("score", 0.0) / 100.0, 0, 1) * 2 - 1,
                np.clip(self.cycle / 1000.0, 0, 1) * 2 - 1,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def _decode_level(self, action: np.ndarray) -> dict:
        floor_y = self.height - 2  # eine Reihe ueber dem festen Boden
        start_x = 1
        goal_x = self.width - 2

        tiles = [
            {"x": start_x, "y": floor_y, "type": TILE_START},
            {"x": goal_x, "y": floor_y, "type": TILE_GOAL},
        ]

        occupied = {start_x, goal_x}
        for i in range(self.max_traps):
            active_signal = action[i * 2]
            pos_signal = action[i * 2 + 1]
            if active_signal > 0.0:  # Schwelle: Slot ist aktiv
                x = int(round((pos_signal + 1) / 2 * (self.width - 3))) + 1
                x = max(1, min(self.width - 2, x))
                if x not in occupied:
                    occupied.add(x)
                    tiles.append({"x": x, "y": floor_y, "type": TILE_TRAP})

        level = {
            "version": "1.0",
            "name": f"gen_c{self.cycle}",
            "width": self.width,
            "height": self.height,
            "target_deaths": self.target_difficulty,
            "cycle": self.cycle,
            "tiles": tiles,
        }
        return level

    def step(self, action: np.ndarray):
        level = self._decode_level(np.asarray(action, dtype=np.float32))
        self.last_level = level

        lint_ok, errors, filepath = self.export_and_lint_fn(level)
        self.last_lint_ok = lint_ok
        self.last_lint_errors = errors
        self.last_level_path = filepath

        if not lint_ok:
            # Ungueltiges Level: verworfen + Penalty, sofort zurueck (Cooldown macht die Pipeline)
            reward = -5.0
            info = {"lint_ok": False, "errors": errors, "feedback": None, "level_path": filepath}
            return self._obs(), reward, True, False, info

        score, feedback = self.play_and_score_fn(level)
        reward = float(score) / 10.0  # Skalierung fuer PPO

        info = {"lint_ok": True, "errors": [], "feedback": feedback, "level_path": filepath}
        return self._obs(), reward, True, False, info
