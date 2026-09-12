"""
level_schema.py
================
Definiert das Level-JSON-Format und den Linter, der jedes generierte
Level vor dem Laden in die Engine prueft.

Tile-Typen:
  0 = leer (Luft)
  1 = fester Block (Boden/Wand)
  2 = Falle / Spike (Tod bei Beruehrung)
  4 = Start
  5 = Ziel (Goal)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jsonschema

TILE_EMPTY = 0
TILE_SOLID = 1
TILE_TRAP = 2
TILE_START = 4
TILE_GOAL = 5

VALID_TILE_TYPES = {TILE_EMPTY, TILE_SOLID, TILE_TRAP, TILE_START, TILE_GOAL}

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
        "tiles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["x", "y", "type"],
                "properties": {
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "type": {"type": "integer"},
                },
            },
        },
    },
}


@dataclass
class LintResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def lint_level(level: dict) -> LintResult:
    """Prueft: 1) gueltiges JSON-Schema  2) width/height im Bereich 1-100
    3) mindestens 1 Start-Tile und 1 Goal-Tile vorhanden."""
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

    n_start = sum(1 for t in tiles if t["type"] == TILE_START)
    n_goal = sum(1 for t in tiles if t["type"] == TILE_GOAL)
    if n_start < 1:
        errors.append("Kein Start-Tile (type 4) vorhanden")
    if n_goal < 1:
        errors.append("Kein Goal-Tile (type 5) vorhanden")

    return LintResult(ok=(len(errors) == 0), errors=errors)
