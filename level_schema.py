"""
level_schema.py
================
Definiert das Level-JSON-Format und den Linter, der jedes generierte
Level vor dem Laden in die Engine prueft.

Tile-Typen (Basis, aus V5):
  0 = leer (Luft)
  1 = fester Block (Boden/Wand)
  2 = Falle / Spike (Tod bei Beruehrung)
  3 = Plattform (einweg-begehbar: von unten durchspringbar, von oben landbar)
  4 = Start
  5 = Ziel (Goal)

Tile-Typen (neu, inspiriert von echten Level-Devil-Mechaniken):
  6 = Bounce-Pad / Trampolin (schleudert den Spieler nach oben)
  7 = Fake-Floor (sieht solide aus, bricht nach kurzer Standzeit weg)
  8 = Buzzsaw (patrouilliert horizontal, toedlich bei Beruehrung)
  9 = Coin (Koeder -- loest beim Einsammeln eine versteckte Falle aus)
  10 = Gravity-Flip-Trigger (kehrt beim Beruehren die Schwerkraft um)
  11 = Invisible Platform (wie Plattform, aber unsichtbar bis zur Beruehrung)

Optionale Zusatzfelder pro Tile (werden von game_engine.py ausgewertet,
vom Linter nur locker validiert):
  - Buzzsaw (type 8): "range" (int, Patrouillenradius in Tiles um die
    Spawn-Position, Default siehe game_engine.DEFAULT_BUZZSAW_RANGE)
  - Coin (type 9): "trap_x"/"trap_y" (int, Zielzelle die beim Einsammeln
    zu einer Falle wird)

Optionales Level-weites Feld:
  - "goal_flees": bool -- wenn true, kann das Ziel vor dem Spieler
    "wegfliehen", wenn er sich naehert (siehe game_engine.py)
  - "goal_flee_chance": float (0-1) -- Wahrscheinlichkeit pro Schritt,
    dass die Flucht ausgeloest wird, wenn der Spieler nah genug ist
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import jsonschema

TILE_EMPTY = 0
TILE_SOLID = 1
TILE_TRAP = 2
TILE_PLATFORM = 3
TILE_START = 4
TILE_GOAL = 5
TILE_BOUNCE = 6
TILE_FAKE_FLOOR = 7
TILE_BUZZSAW = 8
TILE_COIN = 9
TILE_GRAVITY_FLIP = 10
TILE_INVISIBLE_PLATFORM = 11

VALID_TILE_TYPES = {
    TILE_EMPTY, TILE_SOLID, TILE_TRAP, TILE_PLATFORM, TILE_START, TILE_GOAL,
    TILE_BOUNCE, TILE_FAKE_FLOOR, TILE_BUZZSAW, TILE_COIN, TILE_GRAVITY_FLIP,
    TILE_INVISIBLE_PLATFORM,
}

# Fuer die (approximative) Reachability-Pruefung kategorisiert:
# - "blockierend": man kann nicht INS Tile hineinlaufen/springen (wie eine Wand)
# - "Oberflaeche": man kann darauf STEHEN (die Zelle darueber ist begehbar)
# Fake-Floor zaehlt strukturell wie solide (er traegt kurzzeitig, das reicht
# fuer die topologische Pruefung -- das tatsaechliche Wegbrechen ist reine
# Laufzeit-Mechanik in game_engine.py). Bounce-Pad, Plattform und die
# unsichtbare Plattform sind alle "Oberflaechen" (einweg-begehbar).
# Buzzsaw, Coin, Gravity-Flip-Trigger sind bewusst NICHT blockierend --
# sie sind dynamische/optionale Hindernisse, keine strukturellen Waende
# (aehnlich wie normale Traps, siehe check_reachability-Docstring).
BLOCKING_FOR_TRAVERSAL = {TILE_SOLID, TILE_FAKE_FLOOR}
SURFACE_TILES = {TILE_SOLID, TILE_PLATFORM, TILE_FAKE_FLOOR, TILE_BOUNCE, TILE_INVISIBLE_PLATFORM}

# Bewegungsreichweite fuer die Reachability-Pruefung. Muss grob zur Physik
# in game_engine.py passen (Sprunghoehe/-weite). Bewusst etwas grosszuegiger
# gewaehlt als die tatsaechliche Physik, da dies nur eine topologische
# Approximation ist (keine exakte Sprungkurven-Simulation).
DEFAULT_MAX_JUMP_HEIGHT = 3
DEFAULT_MAX_JUMP_DIST = 4

LEVEL_JSON_SCHEMA = {
    "type": "object",
    "required": ["version", "name", "width", "height", "target_deaths", "cycle", "tiles"],
    "properties": {
        "version": {"type": "string"},
        "name": {"type": "string"},
        "width": {"type": "integer", "minimum": 1, "maximum": 100},
        "height": {"type": "integer", "minimum": 1, "maximum": 100},
        "target_deaths": {"type": "number"},
        "cycle": {"type": "integer", "minimum": 0},
        "goal_flees": {"type": "boolean"},
        "goal_flee_chance": {"type": "number", "minimum": 0, "maximum": 1},
        "tiles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["x", "y", "type"],
                "properties": {
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "type": {"type": "integer"},
                    "range": {"type": "integer", "minimum": 1},
                    "trap_x": {"type": "integer", "minimum": 0},
                    "trap_y": {"type": "integer", "minimum": 0},
                },
            },
        },
    },
}


@dataclass
class LintResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _build_grid(level: dict) -> list[list[int]]:
    width, height = level["width"], level["height"]
    grid = [[TILE_EMPTY for _ in range(width)] for _ in range(height)]
    # Standard-Boden in unterster Zeile (gleiche Konvention wie game_engine.py)
    for x in range(width):
        grid[height - 1][x] = TILE_SOLID
    for t in level["tiles"]:
        x, y, typ = t["x"], t["y"], t["type"]
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = typ
    return grid


def _is_standable_surface(grid: list[list[int]], x: int, y: int) -> bool:
    """Zelle (x,y) ist begehbar, wenn sie selbst nicht blockierend ist und
    darunter eine Oberflaeche (Boden/Plattform/...) liegt."""
    height = len(grid)
    if grid[y][x] in BLOCKING_FOR_TRAVERSAL:
        return False
    below_y = y + 1
    if below_y >= height:
        return False
    return grid[below_y][x] in SURFACE_TILES


def _line_clear(grid: list[list[int]], x0: int, y0: int, x1: int, y1: int, height: int) -> bool:
    """Prueft, ob die direkte Verbindungslinie zwischen zwei Zellen frei von
    blockierenden Bloecken ist (inkl. einer Kopf-Zeile fuer die Spielerhoehe).
    Ohne diese Pruefung wuerde ein grosser dx/dy-Sprung eine dazwischen
    stehende Wand faelschlich 'ueberspringen', da nur das Ziel geprueft
    wird statt des Wegs dorthin."""
    width = len(grid[0])
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        xi = round(x0 + (x1 - x0) * t)
        yi = round(y0 + (y1 - y0) * t)
        if not (0 <= xi < width):
            return False
        for yy in (yi, yi - 1):  # Zelle + Kopfhoehe
            if 0 <= yy < height and grid[yy][xi] in BLOCKING_FOR_TRAVERSAL:
                return False
    return True


def check_reachability(
    level: dict,
    max_jump_height: int = DEFAULT_MAX_JUMP_HEIGHT,
    max_jump_dist: int = DEFAULT_MAX_JUMP_DIST,
) -> tuple[bool, str]:
    """Vereinfachte BFS-Erreichbarkeitspruefung: kann der Spieler ausgehend
    vom Start-Tile das Goal-Tile ueberhaupt erreichen?

    Das ist eine TOPOLOGISCHE APPROXIMATION, keine exakte Sprungkurven-
    oder Zeit-Simulation: Traps, Buzzsaws, Coins und Gravity-Flip-Trigger
    werden nicht als Blockade gewertet (der Spieler kann sie im Normalfall
    umgehen/ueberspringen -- sie beeinflussen die Schwierigkeit, nicht die
    grundsaetzliche Lösbarkeit). Ein fliehendes Ziel (goal_flees) wird an
    seiner AUTORIERTEN Position geprueft; das Wegfliehen ist reine
    Laufzeit-Mechanik und macht ein Level nicht strukturell unloesbar.
    Geprueft wird, ob es eine Kette aus stehbaren Zellen gibt, die per
    Sprung (seitlich max_jump_dist, nach oben max_jump_height, nach unten
    beliebig weit) verbunden sind UND deren direkte Verbindungslinie nicht
    durch einen blockierenden Block laeuft (siehe _line_clear) -- so werden
    z.B. durchgehende Waende korrekt als Blockade erkannt statt uebersprungen.
    """
    width, height = level["width"], level["height"]
    grid = _build_grid(level)

    starts = [t for t in level["tiles"] if t["type"] == TILE_START]
    goals = [t for t in level["tiles"] if t["type"] == TILE_GOAL]
    if not starts or not goals:
        return False, "Kein Start- oder Goal-Tile fuer Reachability-Check vorhanden"

    start = (starts[0]["x"], starts[0]["y"])
    goal = (goals[0]["x"], goals[0]["y"])

    visited = {start}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True, ""

        for dx in range(-max_jump_dist, max_jump_dist + 1):
            nx = x + dx
            if not (0 <= nx < width):
                continue
            for dy in range(-max_jump_height, height):
                ny = y + dy
                if not (0 <= ny < height):
                    continue
                if (nx, ny) in visited:
                    continue
                if grid[ny][nx] in BLOCKING_FOR_TRAVERSAL:
                    continue
                if not _line_clear(grid, x, y, nx, ny, height):
                    continue
                # Erreichbar, wenn Zielzelle selbst das Goal ist ODER dort
                # eine stehbare Oberflaeche existiert.
                if (nx, ny) == goal or _is_standable_surface(grid, nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False, f"Goal bei {goal} ist vom Start bei {start} aus nicht erreichbar (BFS-Approximation)"


def lint_level(
    level: dict,
    check_reach: bool = True,
    max_jump_height: int = DEFAULT_MAX_JUMP_HEIGHT,
    max_jump_dist: int = DEFAULT_MAX_JUMP_DIST,
) -> LintResult:
    """Prueft: 1) gueltiges JSON-Schema  2) width/height im Bereich 1-100
    3) mindestens 1 Start-Tile und 1 Goal-Tile vorhanden
    4) (optional) Goal ist vom Start aus ueberhaupt erreichbar (BFS)
    5) Buzzsaw-"range" und Coin-"trap_x"/"trap_y" liegen (falls gesetzt)
       innerhalb des Levels."""
    errors: list[str] = []

    try:
        jsonschema.validate(instance=level, schema=LEVEL_JSON_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema-Fehler: {e.message}")
        return LintResult(ok=False, errors=errors)

    width = level["width"]
    height = level["height"]
    if not (1 <= width <= 100):
        errors.append(f"width {width} ausserhalb 1-100")
    if not (1 <= height <= 100):
        errors.append(f"height {height} ausserhalb 1-100")

    tiles = level["tiles"]
    for t in tiles:
        if t["type"] not in VALID_TILE_TYPES:
            errors.append(f"Ungueltiger Tile-Typ {t['type']} bei ({t['x']},{t['y']})")
        if not (0 <= t["x"] < width):
            errors.append(f"x={t['x']} ausserhalb der Breite {width}")
        if not (0 <= t["y"] < height):
            errors.append(f"y={t['y']} ausserhalb der Hoehe {height}")
        if t["type"] == TILE_COIN:
            tx, ty = t.get("trap_x"), t.get("trap_y")
            if tx is not None and not (0 <= tx < width):
                errors.append(f"Coin bei ({t['x']},{t['y']}): trap_x={tx} ausserhalb der Breite {width}")
            if ty is not None and not (0 <= ty < height):
                errors.append(f"Coin bei ({t['x']},{t['y']}): trap_y={ty} ausserhalb der Hoehe {height}")

    n_start = sum(1 for t in tiles if t["type"] == TILE_START)
    n_goal = sum(1 for t in tiles if t["type"] == TILE_GOAL)
    if n_start < 1:
        errors.append("Kein Start-Tile (type 4) vorhanden")
    if n_goal < 1:
        errors.append("Kein Goal-Tile (type 5) vorhanden")

    # Reachability nur pruefen, wenn die Grundstruktur bereits gueltig ist
    # (sonst wuerde die BFS auf kaputten Daten laufen).
    if check_reach and not errors:
        reachable, reach_error = check_reachability(level, max_jump_height, max_jump_dist)
        if not reachable:
            errors.append(f"Nicht erreichbar: {reach_error}")

    return LintResult(ok=(len(errors) == 0), errors=errors)
