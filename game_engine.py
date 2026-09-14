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

=== Level-Devil-inspirierte Zusatzmechaniken ===
Bounce-Pad, Fake-Floor, Buzzsaw, Coin (mit versteckter Falle), Gravity-
Flip und Invisible Platform sind bewusst ausgewaehlte, RL-freundliche
Vereinfachungen echter Level-Devil-Mechaniken (siehe level_schema.py fuer
die vollstaendige Tile-Typ-Liste). NICHT umgesetzt (zu komplex fuer den
aktuellen Scope, siehe README): druckbare/zerstoerbare Kisten mit echter
Schiebe-Physik, Hebel/Knopf-Kettenlogik, Projektil-Tuerme, fliehende
Tueren mit exakter Pfad-Planung -- diese wuerden eigene Zustandsautomaten
und deutlich mehr Tile-Metadaten brauchen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from level_schema import (
    TILE_BOUNCE,
    TILE_BUZZSAW,
    TILE_COIN,
    TILE_EMPTY,
    TILE_FAKE_FLOOR,
    TILE_GOAL,
    TILE_GRAVITY_FLIP,
    TILE_INVISIBLE_PLATFORM,
    TILE_PLATFORM,
    TILE_SOLID,
    TILE_START,
    TILE_TRAP,
)

GRAVITY = 0.045
MOVE_SPEED = 0.22
JUMP_VELOCITY = -0.62
MAX_FALL_SPEED = 0.6
PLAYER_W = 0.6
PLAYER_H = 0.9

# Level-Devil-Zusatzmechaniken: Standardwerte (per Level-JSON ueberschreibbar)
DEFAULT_FAKE_FLOOR_COLLAPSE_STEPS = 6   # so viele Schritte STEHEND, bevor der Boden bricht
DEFAULT_BUZZSAW_RANGE = 3               # Patrouillenradius in Tiles um die Spawn-Position
BUZZSAW_SPEED = 0.05                    # Tiles pro Schritt
BOUNCE_VELOCITY_MULTIPLIER = 1.6        # staerker als ein normaler Sprung
DEFAULT_GOAL_FLEE_RADIUS = 3.0          # ab wann das Ziel "nervoes" wird
DEFAULT_GOAL_FLEE_CHANCE = 0.15         # Wahrscheinlichkeit pro Schritt in Reichweite

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


@dataclass
class RenderState:
    """Laufzeit-Abweichungen vom statischen, autorisierten Level -- fuer
    die Live-GUI, damit sie eingestuerzte Boeden, eingesammelte Coins,
    ausgeloeste Fallen, Buzzsaw-Positionen, Gravitationsrichtung und ein
    (moeglicherweise gefluechtetes) Ziel korrekt darstellen kann, obwohl
    sie selbst nur das urspruengliche Level-JSON kennt."""

    collapsed_floors: list[tuple[int, int]] = field(default_factory=list)
    collected_coins: list[tuple[int, int]] = field(default_factory=list)
    spawned_traps: list[tuple[int, int]] = field(default_factory=list)
    revealed_invisible: list[tuple[int, int]] = field(default_factory=list)
    buzzsaw_positions: list[tuple[float, float]] = field(default_factory=list)
    gravity_sign: int = 1
    goal_pos: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "collapsed_floors": self.collapsed_floors,
            "collected_coins": self.collected_coins,
            "spawned_traps": self.spawned_traps,
            "revealed_invisible": self.revealed_invisible,
            "buzzsaw_positions": self.buzzsaw_positions,
            "gravity_sign": self.gravity_sign,
            "goal_pos": self.goal_pos,
        }


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
        self._buzzsaws: list[dict] = []
        self._coin_triggers: dict[tuple[int, int], tuple[int, int]] = {}

        for t in level["tiles"]:
            x, y, typ = t["x"], t["y"], t["type"]
            self.grid[y, x] = typ
            if typ == TILE_START:
                start = (x, y)
            elif typ == TILE_GOAL:
                goal = (x, y)
            elif typ == TILE_BUZZSAW:
                # Buzzsaw bleibt im Grid nur als Spawn-Marker fuer die
                # Anzeige stehen (siehe get_render_state) -- Kollision
                # laeuft ausschliesslich ueber die dynamische Position
                # unten, damit sie sich unabhaengig vom Grid bewegen kann.
                rng = t.get("range", DEFAULT_BUZZSAW_RANGE)
                self._buzzsaws.append({"origin_x": float(x), "y": y, "x": float(x), "range": float(rng), "dir": 1})
            elif typ == TILE_COIN:
                tx, ty = t.get("trap_x"), t.get("trap_y")
                if tx is not None and ty is not None and 0 <= tx < self.width and 0 <= ty < self.height:
                    self._coin_triggers[(x, y)] = (tx, ty)

        if start is None:
            start = (1, self.height - 2)
        if goal is None:
            goal = (self.width - 2, self.height - 2)

        self.start_x, self.start_y = start
        self.goal_x, self.goal_y = goal

        self.goal_flees = bool(level.get("goal_flees", False))
        self.goal_flee_chance = float(level.get("goal_flee_chance", DEFAULT_GOAL_FLEE_CHANCE))
        self.fake_floor_collapse_steps = int(level.get("fake_floor_collapse_steps", DEFAULT_FAKE_FLOOR_COLLAPSE_STEPS))

        self._fake_floor_timers: dict[tuple[int, int], int] = {}
        self._collapsed_floors: list[tuple[int, int]] = []
        self._collected_coins: list[tuple[int, int]] = []
        self._spawned_traps: list[tuple[int, int]] = []
        self._revealed_invisible: set[tuple[int, int]] = set()
        self.gravity_sign = 1

        self.reset()

    def reset(self) -> None:
        self.x = float(self.start_x) + 0.5
        self.y = float(self.start_y)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        # Laufzeit-Mutationen (kollabierte Boeden, eingesammelte Coins, ...)
        # bleiben bewusst ÜBER reset() hinweg bestehen -- ein "Run" innerhalb
        # desselben Levels (siehe player_env.py) soll nicht bei jedem Tod
        # wieder von vorne "frisch" werden, sonst waeren Fake-Floors/Coins
        # nach jedem Tod nutzlos wirkungslos.

    def _is_blocking(self, gx: int, gy: int) -> bool:
        """Solide Bloecke UND (noch nicht kollabierte) Fake-Floors --
        beides wirkt strukturell wie eine Wand/ein Boden."""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True  # Levelgrenzen wirken wie Waende
        return self.grid[gy, gx] in (TILE_SOLID, TILE_FAKE_FLOOR)

    def _is_platform_like(self, gx: int, gy: int) -> bool:
        """Alles, was sich wie eine Einweg-Plattform verhaelt: normale
        Plattform, Bounce-Pad (steht man einfach drauf, ohne zu springen,
        wirkt es wie eine Plattform) und die unsichtbare Plattform."""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy, gx] in (TILE_PLATFORM, TILE_BOUNCE, TILE_INVISIBLE_PLATFORM)

    def _is_trap(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy, gx] == TILE_TRAP

    def _is_goal(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        return self.grid[gy, gx] == TILE_GOAL

    def _occupied_cells(self, x: float, y: float) -> list[tuple[int, int]]:
        """Liefert die (bis zu 4) Grid-Zellen, die das Spieler-Rechteck
        [x, x+PLAYER_W) x [y, y+PLAYER_H) ueberlappt.

        WICHTIG: die "ferne" Kante (x+PLAYER_W, y+PLAYER_H) bekommt ein
        winziges Epsilon abgezogen, bevor sie geflooret wird. Ohne das
        wuerde ein Spieler, der EXAKT auf einer Tile-Grenze ruht (z.B.
        bottom == 14.0 nach einer Landung), faelschlich als "in Zeile 14"
        gezaehlt werden (floor(14.0) == 14), obwohl sein Rechteck als
        halb-offenes Intervall [13.1, 14.0) eigentlich nur Zeile 13
        beruehrt, nicht ueberlappt. Diese Ambiguitaet fuehrte sonst dazu,
        dass ein stehender Spieler sich selbst horizontal blockiert
        (faelschliche Kollision mit dem Boden UNTER ihm)."""
        eps = 1e-6
        x0, x1 = x, x + PLAYER_W - eps
        y0, y1 = y, y + PLAYER_H - eps
        gxs = (int(np.floor(x0)), int(np.floor(x1)))
        gys = (int(np.floor(y0)), int(np.floor(y1)))
        return [(gx, gy) for gx in gxs for gy in gys]

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
            self.vy = self.jump_velocity * self.gravity_sign

        # Gravitation wirkt in Richtung gravity_sign (+1 normal, -1 nach
        # einem Gravity-Flip-Trigger). max_fall_speed begrenzt symmetrisch
        # in beide Richtungen.
        self.vy = self.vy + self.gravity * self.gravity_sign
        self.vy = max(-self.max_fall_speed, min(self.vy, self.max_fall_speed))

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Horizontale Kollision (Plattform-artige Tiles blockieren seitlich
        # nicht, nur solide/Fake-Floor-Bloecke)
        if self._collides_rect(new_x, self.y):
            new_x = self.x
            self.vx = 0.0

        # Vertikale Kollision. Symmetrisch fuer beide Gravitationsrichtungen:
        # vy > 0 = Bewegung "mit" normaler Schwerkraft (Landung von oben),
        # vy < 0 = Bewegung "gegen" normale Schwerkraft (z.B. nach einem
        # Gravity-Flip: Landung von unten an einer "Decke").
        self.on_ground = False
        if self._collides_vertical(new_x, new_y, self.vy, self.y):
            if self.vy > 0:
                new_y = float(int(new_y + PLAYER_H)) - PLAYER_H
                self.on_ground = True
            elif self.vy < 0:
                new_y = float(int(new_y)) + 1.0
                self.on_ground = True
            self.vy = 0.0

        self.x, self.y = new_x, new_y

        # --- Tod-Pruefung (Traps + Buzzsaws) ---
        self._update_buzzsaws()
        died = (
            self._check_trap_collision(self.x, self.y)
            or self._check_buzzsaw_collision(self.x, self.y)
            or self.y > self.height
            or self.y < -1
        )
        won = False
        if not died:
            won = self._check_goal_collision(self.x, self.y)

        if not died and not won:
            self._handle_tile_interactions()

        return StepResult(died=died, won=won, x=self.x, y=self.y, vx=self.vx, vy=self.vy, on_ground=self.on_ground)

    def _collides_rect(self, x: float, y: float) -> bool:
        """Nur blockierende Bloecke -- fuer horizontale Kollision.
        Plattform-artige Tiles blockieren hier bewusst nicht (duenn, nur
        von oben begehbar)."""
        for gx, gy in self._occupied_cells(x, y):
            if self._is_blocking(gx, gy):
                return True
        return False

    def _collides_vertical(self, x: float, y: float, vy: float, prev_y: float) -> bool:
        """Vertikale Kollision inkl. Einweg-Plattform-artiger Tiles: eine
        Plattform blockiert nur, wenn der Spieler in Gravitationsrichtung
        faellt/steht UND sein Fuss vor diesem Schritt bereits auf der
        "Eintritts-Seite" der Plattform war (verhindert, dass man beim
        Hochspringen an einer Plattform haengen bleibt). Bei invertierter
        Gravitation (gravity_sign < 0) ist dieses Verhalten eine bewusst
        vereinfachte Naeherung -- die Kombination aus Einweg-Plattformen
        und Gravity-Flip ist ein bekannter Randfall, siehe README."""
        prev_bottom = prev_y + PLAYER_H
        for gx, gy in self._occupied_cells(x, y):
            if self._is_blocking(gx, gy):
                return True
            if self._is_platform_like(gx, gy) and vy >= 0 and prev_bottom <= gy + 0.05:
                return True
        return False

    def _check_trap_collision(self, x: float, y: float) -> bool:
        return any(self._is_trap(gx, gy) for gx, gy in self._occupied_cells(x, y))

    def _check_goal_collision(self, x: float, y: float) -> bool:
        return any(self._is_goal(gx, gy) for gx, gy in self._occupied_cells(x, y))

    def _check_buzzsaw_collision(self, x: float, y: float) -> bool:
        for saw in self._buzzsaws:
            if x < saw["x"] + 1.0 and x + PLAYER_W > saw["x"] and y < saw["y"] + 1.0 and y + PLAYER_H > saw["y"]:
                return True
        return False

    def _update_buzzsaws(self) -> None:
        for saw in self._buzzsaws:
            saw["x"] += saw["dir"] * BUZZSAW_SPEED
            if saw["x"] >= saw["origin_x"] + saw["range"]:
                saw["x"] = saw["origin_x"] + saw["range"]
                saw["dir"] = -1
            elif saw["x"] <= saw["origin_x"] - saw["range"]:
                saw["x"] = saw["origin_x"] - saw["range"]
                saw["dir"] = 1

    def _handle_tile_interactions(self) -> None:
        """Bounce-Pad, Fake-Floor-Timer, Invisible-Platform-Reveal, Coin-
        Einsammeln (+ Fallen-Ausloesung) und Gravity-Flip. Wird nur
        aufgerufen, wenn der Spieler in diesem Schritt weder gestorben
        noch am Ziel angekommen ist.

        Zwei verschiedene Beruehrungsarten:
        - "Durchlaufen" (Coin, Gravity-Flip, Invisible-Platform von unten):
          der Spieler ueberlappt die Zelle koerperlich -> occupied_cells.
        - "Stehen auf" (Bounce-Pad, Fake-Floor, Invisible-Platform von
          oben): die Landung endet GENAU auf der Tile-Grenze, der Spieler
          ueberlappt die Zelle also NICHT koerperlich -- die relevante
          Zelle liegt direkt unter (bzw. bei geflippter Gravitation ueber)
          den Fuessen. Nur ueber occupied_cells wuerde ein Bounce-Pad nie
          ausgeloest werden, sobald der Spieler sauber darauf steht.
        """
        occupied = self._occupied_cells(self.x, self.y)

        for gx, gy in occupied:
            if not (0 <= gx < self.width and 0 <= gy < self.height):
                continue
            cell = self.grid[gy, gx]

            if cell == TILE_INVISIBLE_PLATFORM:
                self._revealed_invisible.add((gx, gy))

            elif cell == TILE_COIN:
                self.grid[gy, gx] = TILE_EMPTY
                self._collected_coins.append((gx, gy))
                target = self._coin_triggers.pop((gx, gy), None)
                if target is not None:
                    tx, ty = target
                    if self.grid[ty, tx] not in (TILE_START, TILE_GOAL, TILE_SOLID):
                        self.grid[ty, tx] = TILE_TRAP
                        self._spawned_traps.append((tx, ty))

            elif cell == TILE_GRAVITY_FLIP:
                self.gravity_sign *= -1
                self.grid[gy, gx] = TILE_EMPTY

        if self.on_ground:
            foot_y = int(np.floor(self.y + PLAYER_H)) if self.gravity_sign > 0 else int(np.floor(self.y)) - 1
            for gx in (int(np.floor(self.x)), int(np.floor(self.x + PLAYER_W))):
                if not (0 <= gx < self.width and 0 <= foot_y < self.height):
                    continue
                surface_cell = self.grid[foot_y, gx]

                if surface_cell == TILE_BOUNCE:
                    self.vy = self.jump_velocity * self.gravity_sign * BOUNCE_VELOCITY_MULTIPLIER
                    self.on_ground = False

                elif surface_cell == TILE_INVISIBLE_PLATFORM:
                    self._revealed_invisible.add((gx, foot_y))

                elif surface_cell == TILE_FAKE_FLOOR:
                    key = (gx, foot_y)
                    count = self._fake_floor_timers.get(key, 0) + 1
                    if count >= self.fake_floor_collapse_steps:
                        self.grid[foot_y, gx] = TILE_EMPTY
                        self._collapsed_floors.append(key)
                        self._fake_floor_timers.pop(key, None)
                    else:
                        self._fake_floor_timers[key] = count

        # Fliehendes Ziel: nur wenn aktiviert und der Spieler nah genug ist
        if self.goal_flees:
            dist = abs(self.x - self.goal_x) + abs(self.y - self.goal_y)
            if dist <= DEFAULT_GOAL_FLEE_RADIUS and random.random() < self.goal_flee_chance:
                self._flee_goal()

    def _flee_goal(self) -> None:
        """Versucht, das Ziel an eine andere freie, nicht-blockierende
        Stelle im Level zu verschieben ('Level Devil'-Tueren, die
        wegfliegen/wegsinken, wenn man sich naehert). Findet die Suche
        nach ein paar Versuchen keine freie Zelle, bleibt das Ziel einfach
        stehen -- kein hartes Scheitern."""
        for _ in range(20):
            nx = random.randint(1, self.width - 2)
            ny = random.randint(0, self.height - 2)
            if (nx, ny) == (self.goal_x, self.goal_y):
                continue
            if self.grid[ny, nx] not in (TILE_EMPTY, TILE_PLATFORM, TILE_INVISIBLE_PLATFORM, TILE_BOUNCE):
                continue
            self.grid[self.goal_y, self.goal_x] = TILE_EMPTY
            self.goal_x, self.goal_y = nx, ny
            self.grid[ny, nx] = TILE_GOAL
            return

    def get_render_state(self) -> dict:
        """Fuer die Live-GUI: alles, was seit Levelstart von der
        autorisierten JSON-Beschreibung abweicht."""
        return RenderState(
            collapsed_floors=list(self._collapsed_floors),
            collected_coins=list(self._collected_coins),
            spawned_traps=list(self._spawned_traps),
            revealed_invisible=list(self._revealed_invisible),
            buzzsaw_positions=[(s["x"], float(s["y"])) for s in self._buzzsaws],
            gravity_sign=self.gravity_sign,
            goal_pos=(self.goal_x, self.goal_y),
        ).to_dict()
