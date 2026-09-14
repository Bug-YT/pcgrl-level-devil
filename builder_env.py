"""
builder_env.py
================
Gymnasium-Environment fuer den Builder-Agenten.

Start- und Ziel-Position sind FREI waehlbar (nicht mehr fix am Rand) --
der Builder lernt sie ueber die ersten 4 Aktionswerte
[start_x, start_y, goal_x, goal_y]. Zwei weitere feste Aktionswerte
steuern, ob das Ziel "fliehen" kann (siehe game_engine.py) und mit
welcher Wahrscheinlichkeit. Danach folgen bis zu max_objects generische
Objekt-Slots im Format [aktiv, x, y, type_selector, extra], ueber die der
Builder aus der vollen, Level-Devil-inspirierten Tile-Palette waehlen
kann (siehe OBJECT_TYPES): Traps, Plattformen, feste Bloecke, Bounce-
Pads, Fake-Floors, Buzzsaws, Coins (mit versteckter Falle) und
Gravity-Flip-Trigger. Das "extra"-Feld wird je nach Typ unterschiedlich
interpretiert (Buzzsaw-Patrouillenradius, Coin-Fallen-Richtung) und sonst
ignoriert. Ein einzelner, kompakter [aktiv,x,y,type,extra]-Slot haelt den
Action-Space uebersichtlich, auch wenn spaeter weitere Tile-Typen dazu
kommen -- man muss nur OBJECT_TYPES erweitern, nicht die Vektorstruktur.

=== "Inspiriert von echten Level-Devil-Leveln" ===
Die rohe RL-Platzierung allein reproduziert die typischen Troll-Muster
des echten Spiels nicht zuverlaessig genug (dafuer bruachte es sehr viel
Training). Deshalb legt _apply_level_devil_flavor() nach der RL-Dekodierung
eine kleine, KURATIERTE Konstruktionsschicht drueber, die drei konkrete,
oft genannte Muster verstaerkt:
  1. "Bounce-Pad katapultiert in Spikes": hinter einem Bounce-Pad landet
     mit einer gewissen Wahrscheinlichkeit ein Trap in Sprungrichtung.
  2. "Boden bricht kurz vor dem Ziel weg": solide/Plattform-Tiles nahe am
     Ziel werden mit einer gewissen Wahrscheinlichkeit zu Fake-Floors.
  3. "Coins sind Koeder": jeder vom Builder platzierte Coin bekommt
     garantiert eine Fallen-Ziel-Zelle zugewiesen (statt nur manchmal).
Das ist explizit eine HAND-AUTORIERTE Heuristik, kein gelerntes Verhalten
-- ehrlich benannt, damit klar ist, was RL und was Konstruktionsregel ist.

Eine Episode = EIN generiertes Level ("one-shot"): Der Builder gibt in
einem einzigen step() einen Aktionsvektor aus, der in ein komplettes
Level decodiert wird. step() fuehrt daraufhin selbst den kompletten Rest
des Loops aus (Export -> Lint [inkl. Reachability-Check] -> Load -> Play
-> Feedback) und liefert erst danach den Reward zurueck. So bekommt der
Builder pro step() ein vollstaendiges Feedback-Signal.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from level_schema import (
    TILE_BOUNCE,
    TILE_BUZZSAW,
    TILE_COIN,
    TILE_FAKE_FLOOR,
    TILE_GOAL,
    TILE_GRAVITY_FLIP,
    TILE_INVISIBLE_PLATFORM,
    TILE_PLATFORM,
    TILE_SOLID,
    TILE_START,
    TILE_TRAP,
)

OBS_DIM = 6  # target_norm, last_winrate, last_avg_deaths_norm, last_avg_time_norm, last_score_norm, cycle_norm

# Reihenfolge = Bucket-Zuordnung bei der Dekodierung des type_selector-Werts.
# Erweiterbar, ohne die Action-Space-Struktur zu aendern -- einfach hier
# anhaengen.
OBJECT_TYPES = [
    TILE_TRAP,
    TILE_PLATFORM,
    TILE_SOLID,
    TILE_BOUNCE,
    TILE_FAKE_FLOOR,
    TILE_BUZZSAW,
    TILE_COIN,
    TILE_GRAVITY_FLIP,
    TILE_INVISIBLE_PLATFORM,
]
VALUES_PER_OBJECT = 5  # [aktiv, x, y, type_selector, extra]
N_FIXED_ACTIONS = 6  # [start_x, start_y, goal_x, goal_y, goal_flees, goal_flee_chance]

BUZZSAW_RANGE_MIN, BUZZSAW_RANGE_MAX = 1, 5
GOAL_FLEE_CHANCE_MIN, GOAL_FLEE_CHANCE_MAX = 0.05, 0.5

# Konstruktionsschicht (siehe Docstring oben) -- feste Wahrscheinlichkeiten,
# bewusst kein Lernparameter.
BOUNCE_INTO_TRAP_CHANCE = 0.4
FAKE_FLOOR_NEAR_GOAL_CHANCE = 0.35
FAKE_FLOOR_NEAR_GOAL_RADIUS = 4

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
        goal_flees = bool(action[4] > 0.0)
        goal_flee_chance = GOAL_FLEE_CHANCE_MIN + (action[5] + 1) / 2 * (GOAL_FLEE_CHANCE_MAX - GOAL_FLEE_CHANCE_MIN)
        goal_flee_chance = float(goal_flee_chance)  # numpy.float32 -> natives Python float (sonst nicht JSON-serialisierbar)

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

        travel_dir = 1 if goal_x >= start_x else -1  # Richtung Start -> Ziel, fuer die Flavor-Schicht

        obj_actions = np.asarray(action[N_FIXED_ACTIONS:], dtype=np.float32).reshape(self.max_objects, VALUES_PER_OBJECT)
        for active, x_sig, y_sig, type_sig, extra_sig in obj_actions:
            if active <= 0.0:  # Schwelle: Slot ist aktiv
                continue
            x = self._norm_to_x(x_sig)
            y = self._norm_to_y(y_sig)
            if (x, y) in occupied:
                continue
            type_idx = int((type_sig + 1) / 2 * len(OBJECT_TYPES))
            type_idx = min(type_idx, len(OBJECT_TYPES) - 1)
            tile_type = OBJECT_TYPES[type_idx]

            tile: dict = {"x": x, "y": y, "type": tile_type}

            if tile_type == TILE_BUZZSAW:
                rng = BUZZSAW_RANGE_MIN + int(round((extra_sig + 1) / 2 * (BUZZSAW_RANGE_MAX - BUZZSAW_RANGE_MIN)))
                tile["range"] = max(BUZZSAW_RANGE_MIN, min(BUZZSAW_RANGE_MAX, rng))
            elif tile_type == TILE_COIN:
                self._assign_coin_trap(tile, extra_sig, occupied)

            occupied.add((x, y))
            tiles.append(tile)

        self._apply_level_devil_flavor(tiles, occupied, travel_dir)

        level = {
            "version": "1.0",
            "name": f"gen_c{self.cycle}",
            "width": self.width,
            "height": self.height,
            "target_deaths": self.target_difficulty,
            "cycle": self.cycle,
            "goal_flees": goal_flees,
            "goal_flee_chance": round(goal_flee_chance, 3),
            "tiles": tiles,
        }
        return level

    def _assign_coin_trap(self, coin_tile: dict, extra_sig: float, occupied: set[tuple[int, int]]) -> None:
        """Waehlt ueber das 'extra'-Aktionsfeld eine von 4 Richtungen fuer
        die versteckte Falle, die der Coin beim Einsammeln ausloest (siehe
        game_engine._handle_tile_interactions). Passt die Richtung nicht
        ins Level, wird sie einfach weggelassen (kein Trap) -- der Coin
        bleibt dann ein harmloses Sammelobjekt statt Koeder."""
        directions = [(2, 0), (-2, 0), (0, -2), (0, 2)]
        idx = min(len(directions) - 1, int((extra_sig + 1) / 2 * len(directions)))
        dx, dy = directions[idx]
        tx, ty = coin_tile["x"] + dx, coin_tile["y"] + dy
        if 0 <= tx < self.width and 0 <= ty < self.height - 1 and (tx, ty) not in occupied:
            coin_tile["trap_x"] = tx
            coin_tile["trap_y"] = ty

    def _apply_level_devil_flavor(self, tiles: list[dict], occupied: set[tuple[int, int]], travel_dir: int) -> None:
        """Hand-autorierte Konstruktionsschicht, siehe Moduldocstring.
        Mutiert `tiles` und `occupied` in-place."""
        goal_tile = next((t for t in tiles if t["type"] == TILE_GOAL), None)

        for tile in list(tiles):
            # 1) Bounce-Pad -> mit Wahrscheinlichkeit ein Trap dahinter
            if tile["type"] == TILE_BOUNCE and random.random() < BOUNCE_INTO_TRAP_CHANCE:
                tx = tile["x"] + travel_dir * 2
                ty = tile["y"]
                if 0 <= tx < self.width and (tx, ty) not in occupied:
                    occupied.add((tx, ty))
                    tiles.append({"x": tx, "y": ty, "type": TILE_TRAP})

            # 2) Solide/Plattform-Tiles nahe am Ziel -> manchmal Fake-Floor
            elif (
                goal_tile is not None
                and tile["type"] in (TILE_SOLID, TILE_PLATFORM)
                and abs(tile["x"] - goal_tile["x"]) + abs(tile["y"] - goal_tile["y"]) <= FAKE_FLOOR_NEAR_GOAL_RADIUS
                and random.random() < FAKE_FLOOR_NEAR_GOAL_CHANCE
            ):
                tile["type"] = TILE_FAKE_FLOOR

            # 3) Jeder Coin OHNE bereits zugewiesene Falle bekommt eine
            #    garantierte (Coins sind grundsaetzlich Koeder, siehe Docstring)
            elif tile["type"] == TILE_COIN and "trap_x" not in tile:
                for dx, dy in ((2, 0), (-2, 0), (0, -2), (0, 2)):
                    tx, ty = tile["x"] + dx, tile["y"] + dy
                    if 0 <= tx < self.width and 0 <= ty < self.height - 1 and (tx, ty) not in occupied:
                        tile["trap_x"], tile["trap_y"] = tx, ty
                        break

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
