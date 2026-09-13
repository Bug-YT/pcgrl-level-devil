"""
builder_env.py
================
Gymnasium-Environment fuer den Builder-Agenten.

Start- und Ziel-Position sind FREI waehlbar (nicht mehr fix am Rand) --
der Builder lernt sie ueber die ersten 4 Aktionswerte
[start_x, start_y, goal_x, goal_y]. Danach folgen bis zu max_objects
generische Objekt-Slots im Format [aktiv, x, y, type_selector], ueber die
der Builder Traps, Plattformen oder solide Bloecke platzieren kann (siehe
OBJECT_TYPES). Ein einzelner, kompakter [aktiv,x,y,type]-Slot haelt den
Action-Space uebersichtlich, auch wenn spaeter weitere Tile-Typen dazu
kommen -- man muss nur OBJECT_TYPES erweitern, nicht die Vektorstruktur.

Eine Episode = EIN generiertes Level ("one-shot"): Der Builder gibt in
einem einzigen step() einen Aktionsvektor aus, der in ein komplettes
Level decodiert wird. step() fuehrt daraufhin selbst den kompletten Rest
des Loops aus (Export -> Lint [inkl. Reachability-Check] -> Load -> Play
-> Feedback) und liefert erst danach den Reward zurueck. So bekommt der
Builder pro step() ein vollstaendiges Feedback-Signal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from level_schema import TILE_GOAL, TILE_PLATFORM, TILE_SOLID, TILE_START, TILE_TRAP

OBS_DIM = 6  # target_norm, last_winrate, last_avg_deaths_norm, last_avg_time_norm, last_score_norm, cycle_norm

# Reihenfolge = Bucket-Zuordnung bei der Dekodierung des type_selector-Werts.
# Erweiterbar, ohne die Action-Space-Struktur (4 Werte pro Slot) zu aendern.
OBJECT_TYPES = [TILE_TRAP, TILE_PLATFORM, TILE_SOLID]
VALUES_PER_OBJECT = 4  # [aktiv, x, y, type_selector]
N_FIXED_ACTIONS = 4  # [start_x, start_y, goal_x, goal_y]

PlayAndScoreFn = Callable[[dict], tuple[float, dict]]
ExportFn = Callable[[dict], tuple[bool, list[str], str]]  # -> (lint_ok, errors, filepath)


class BuilderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        width: int,
        height: int,
        max_objects: int,
        target_difficulty: float,
        export_and_lint_fn: ExportFn,
        play_and_score_fn: PlayAndScoreFn,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_objects = max_objects
        self.target_difficulty = target_difficulty
        self.export_and_lint_fn = export_and_lint_fn
        self.play_and_score_fn = play_and_score_fn

        action_dim = N_FIXED_ACTIONS + max_objects * VALUES_PER_OBJECT
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
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
        self.last_info: dict | None = None

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

    def _norm_to_x(self, v: float) -> int:
        x = int(round((v + 1) / 2 * (self.width - 3))) + 1
        return max(1, min(self.width - 2, x))

    def _norm_to_y(self, v: float) -> int:
        # y-Bereich 0..height-2: die unterste Zeile (height-1) ist per
        # Konvention der feste Standard-Boden (siehe game_engine.py) und
        # bleibt dem Builder hier bewusst nicht direkt zugaenglich.
        y = int(round((v + 1) / 2 * (self.height - 2)))
        return max(0, min(self.height - 2, y))

    def _decode_level(self, action: np.ndarray) -> dict:
        start_x = self._norm_to_x(action[0])
        start_y = self._norm_to_y(action[1])
        goal_x = self._norm_to_x(action[2])
        goal_y = self._norm_to_y(action[3])

        # Start und Ziel duerfen nicht auf derselben Zelle landen
        if (start_x, start_y) == (goal_x, goal_y):
            if goal_x < self.width - 2:
                goal_x += 1
            else:
                goal_x -= 1

        tiles = [
            {"x": start_x, "y": start_y, "type": TILE_START},
            {"x": goal_x, "y": goal_y, "type": TILE_GOAL},
        ]
        occupied = {(start_x, start_y), (goal_x, goal_y)}

        obj_actions = np.asarray(action[N_FIXED_ACTIONS:], dtype=np.float32).reshape(self.max_objects, VALUES_PER_OBJECT)
        for active, x_sig, y_sig, type_sig in obj_actions:
            if active <= 0.0:  # Schwelle: Slot ist aktiv
                continue
            x = self._norm_to_x(x_sig)
            y = self._norm_to_y(y_sig)
            if (x, y) in occupied:
                continue
            type_idx = int((type_sig + 1) / 2 * len(OBJECT_TYPES))
            type_idx = min(type_idx, len(OBJECT_TYPES) - 1)
            occupied.add((x, y))
            tiles.append({"x": x, "y": y, "type": OBJECT_TYPES[type_idx]})

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
            self.last_info = info
            return self._obs(), reward, True, False, info

        score, feedback = self.play_and_score_fn(level)
        reward = float(score) / 10.0  # Skalierung fuer PPO

        info = {"lint_ok": True, "errors": [], "feedback": feedback, "level_path": filepath}
        self.last_info = info
        return self._obs(), reward, True, False, info
