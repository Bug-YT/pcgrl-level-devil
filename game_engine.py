"""
game_engine.py
===============
Eigenstaendige, vereinfachte Physik-Engine, die das Kernspielgefuehl von
"Level Devil" nachbildet: Ein Spieler bewegt sich auf einem Tile-Grid,
kann springen (Space) und sich links/rechts bewegen (A/D). Spikes (type 2)
toeten bei Beruehrung, das Goal-Tile (type 5) gewinnt das Level.

Es handelt sich NICHT um das Original-Spiel (kein Zugriff auf dessen
Code/Assets), sondern um eine kompatible Simulation auf Basis des
vereinbarten Level-JSON-Formats, gegen die die RL-Agenten trainieren.

Boden-Konvention: Jede Zeile, die nicht explizit ein Tile enthaelt, ist
Luft (leer) -- AUSSER der untersten Zeile (y = height-1), die per
Konvention als fester Boden gilt, sofern sie nicht explizit ueberschrieben
wird (z.B. durch eine Luecke/Trap). Das entspricht dem Beispiel-Level aus
der Spezifikation.

Plattformen (type 3) sind EINWEG-begehbar: von unten kann man durch sie
hindurchspringen, von oben kann man auf ihnen landen/stehen. Das wird ueber
die vertikale Bewegungsrichtung entschieden (siehe _collides_vertical).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from level_schema import TILE_EMPTY, TILE_GOAL, TILE_PLATFORM, TILE_SOLID, TILE_START, TILE_TRAP

GRAVITY = 0.045
MOVE_SPEED = 0.22
JUMP_VELOCITY = -0.62
MAX_FALL_SPEED = 0.6
PLAYER_W = 0.6
PLAYER_H = 0.9

# Action-Space (Discrete 6): nur Space (Jump) + A/D (Links/Rechts) wie gefordert
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_LEFT_JUMP = 4
ACTION_RIGHT_JUMP = 5

N_ACTIONS = 6


@dataclass
class StepResult:
    died: bool
    won: bool
    x: float
    y: float
    vx: float
    vy: float
    on_ground: bool


class LevelDevilEngine:
    """Grid-basierte Physik-Simulation fuer ein geladenes Level."""

    def __init__(
        self,
        level: dict,
        gravity: float = GRAVITY,
        move_speed: float = MOVE_SPEED,
        jump_velocity: float = JUMP_VELOCITY,
        max_fall_speed: float = MAX_FALL_SPEED,
    ):
        self.width = level["width"]
        self.height = level["height"]
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)

        # Physik-Parameter sind instanzweise konfigurierbar (siehe .env /
        # --gravity, --move-speed, --jump-velocity, --max-fall-speed),
        # damit man die Spielgefuehl-Schwierigkeit ohne Codeaenderung
        # tunen kann.
        self.gravity = gravity
        self.move_speed = move_speed
        self.jump_velocity = jump_velocity
        self.max_fall_speed = max_fall_speed

        # Standard-Boden in unterster Zeile, kann von expliziten Tiles ueberschrieben werden
        self.grid[self.height - 1, :] = TILE_SOLID

        start = None
        goal = None
        for t in level["tiles"]:
            x, y, typ = t["x"], t["y"], t["type"]
            self.grid[y, x] = typ
            if typ == TILE_START:
                start = (x, y)
            elif typ == TILE_GOAL:
                goal = (x, y)

        if start is None:
            start = (1, self.height - 2)
        if goal is None:
            goal = (self.width - 2, self.height - 2)

        self.start_x, self.start_y = start
        self.goal_x, self.goal_y = goal
        self.reset()

    def reset(self) -> None:
        self.x = float(self.start_x) + 0.5
        self.y = float(self.start_y)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False

    def _is_solid(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True  # Levelgrenzen wirken wie Waende
        return self.grid[gy, gx] == TILE_SOLID

    def _is_platform(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy, gx] == TILE_PLATFORM

    def _is_trap(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy, gx] == TILE_TRAP

    def _is_goal(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy, gx] == TILE_GOAL

    def step(self, action: int) -> StepResult:
        move = 0.0
        jump = False
        if action == ACTION_LEFT:
            move = -1.0
        elif action == ACTION_RIGHT:
            move = 1.0
        elif action == ACTION_JUMP:
            jump = True
        elif action == ACTION_LEFT_JUMP:
            move = -1.0
            jump = True
        elif action == ACTION_RIGHT_JUMP:
            move = 1.0
            jump = True

        self.vx = move * self.move_speed
        if jump and self.on_ground:
            self.vy = self.jump_velocity

        self.vy = min(self.vy + self.gravity, self.max_fall_speed)

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Horizontale Kollision (Plattformen blockieren seitlich nicht,
        # nur solide Bloecke -- Plattformen sind duenn/einweg)
        if self._collides_rect(new_x, self.y):
            new_x = self.x
            self.vx = 0.0

        # Vertikale Kollision. Plattformen wirken nur, wenn der Spieler
        # faellt/steht (vy >= 0) UND vorher komplett oberhalb der Plattform
        # war -- so kann man von unten durchspringen, aber von oben landen.
        self.on_ground = False
        if self._collides_vertical(new_x, new_y, self.vy, self.y):
            if self.vy > 0:
                # Landung
                new_y = float(int(new_y + PLAYER_H)) - PLAYER_H
                self.on_ground = True
            self.vy = 0.0
            new_y = self.y if self._collides_vertical(new_x, new_y, self.vy, self.y) else new_y

        self.x, self.y = new_x, new_y

        died = self._check_trap_collision(self.x, self.y) or self.y > self.height
        won = self._check_goal_collision(self.x, self.y)

        return StepResult(died=died, won=won, x=self.x, y=self.y, vx=self.vx, vy=self.vy, on_ground=self.on_ground)

    def _collides_rect(self, x: float, y: float) -> bool:
        """Nur solide Bloecke -- fuer horizontale Kollision. Plattformen
        blockieren hier bewusst nicht (duenn, nur von oben begehbar)."""
        x0, x1 = x, x + PLAYER_W
        y0, y1 = y, y + PLAYER_H
        for gx in (int(np.floor(x0)), int(np.floor(x1))):
            for gy in (int(np.floor(y0)), int(np.floor(y1))):
                if self._is_solid(gx, gy):
                    return True
        return False

    def _collides_vertical(self, x: float, y: float, vy: float, prev_y: float) -> bool:
        """Vertikale Kollision inkl. Einweg-Plattformen: eine Plattform
        blockiert nur, wenn der Spieler faellt/steht (vy >= 0) UND sein
        Fuss vor diesem Schritt bereits oberhalb der Plattform-Oberkante
        war (verhindert, dass man beim Hochspringen an einer Plattform
        haengen bleibt)."""
        x0, x1 = x, x + PLAYER_W
        y0, y1 = y, y + PLAYER_H
        prev_bottom = prev_y + PLAYER_H
        for gx in (int(np.floor(x0)), int(np.floor(x1))):
            for gy in (int(np.floor(y0)), int(np.floor(y1))):
                if self._is_solid(gx, gy):
                    return True
                if self._is_platform(gx, gy) and vy >= 0 and prev_bottom <= gy + 0.05:
                    return True
        return False

    def _check_trap_collision(self, x: float, y: float) -> bool:
        x0, x1 = x, x + PLAYER_W
        y0, y1 = y, y + PLAYER_H
        for gx in (int(np.floor(x0)), int(np.floor(x1))):
            for gy in (int(np.floor(y0)), int(np.floor(y1))):
                if self._is_trap(gx, gy):
                    return True
        return False

    def _check_goal_collision(self, x: float, y: float) -> bool:
        x0, x1 = x, x + PLAYER_W
        y0, y1 = y, y + PLAYER_H
        for gx in (int(np.floor(x0)), int(np.floor(x1))):
            for gy in (int(np.floor(y0)), int(np.floor(y1))):
                if self._is_goal(gx, gy):
                    return True
        return False
