"""
config.py
=========
Liest die .env-Datei (python-dotenv) ein und erlaubt CLI-Overrides.
Alle Pipeline-Parameter leben in der Klasse Config.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv


def _get_bool(val: str, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_optional_float(val: str | None) -> float | None:
    if val is None or val.strip() == "":
        return None
    return float(val)


def _get_optional_int(val: str | None) -> int | None:
    if val is None or val.strip() == "":
        return None
    return int(val)


@dataclass
class Config:
    target_difficulty: float = 12.0
    cooldown_seconds: float = 5.0

    grid_width: int = 20
    grid_height: int = 15

    levels_folder: str = "levels"
    logs_folder: str = "logs"
    models_folder: str = "models"

    n_envs: int = 4
    device: str = "auto"

    n_builder_steps: int = 1
    n_player_steps: int = 2048

    max_objects: int = 6

    eval_runs: int = 4
    max_attempts_per_run: int = 30
    max_steps_per_attempt: int = 400

    ema_alpha: float = 0.2

    gui_enabled: bool = False
    gui_render_every: int = 3

    # --- "Level erst wechseln wenn geschafft/unschaffbar" ---
    # Winrate, ab der ein Level als "geschafft" gilt (1.0 = jeder Run muss gewonnen werden)
    win_threshold: float = 1.0
    # Wie oft (Trainings+Eval-Runden) hoechstens auf DEMSELBEN Level weitertrainiert
    # wird, bevor es -- obwohl laut Linter erreichbar -- als praktisch nicht
    # schaffbar aufgegeben und ein neues Level generiert wird.
    max_rounds_per_level: int = 10
    # Score-Abzug (0-100 Skala), wenn ein Level aufgegeben statt geschafft wurde
    giveup_penalty: float = 20.0

    # --- Reachability-Check (siehe level_schema.check_reachability) ---
    max_jump_height: int = 3
    max_jump_dist: int = 4

    # --- PPO-Lernraten ---
    player_learning_rate: float = 3e-4
    builder_learning_rate: float = 3e-4
    ppo_verbose: int = 0

    # --- Physik (siehe game_engine.py) ---
    gravity: float | None = None
    move_speed: float | None = None
    jump_velocity: float | None = None
    max_fall_speed: float | None = None

    # --- Statistik ---
    stats_window: int = 20
    stats_only: bool = False

    # --- Reproduzierbarkeit ---
    seed: int | None = None

    def ensure_folders(self) -> None:
        for folder in (self.levels_folder, self.logs_folder, self.models_folder):
            Path(folder).mkdir(parents=True, exist_ok=True)


def load_config(env_path: str = ".env", argv: list[str] | None = None) -> Config:
    """Laedt Config zuerst aus .env, danach aus CLI-Argumenten (CLI gewinnt)."""
    load_dotenv(dotenv_path=env_path, override=False)

    cfg = Config(
        target_difficulty=float(os.getenv("TARGET_DIFFICULTY", 12.0)),
        cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", 5.0)),
        grid_width=int(os.getenv("GRID_WIDTH", 20)),
        grid_height=int(os.getenv("GRID_HEIGHT", 15)),
        levels_folder=os.getenv("LEVELS_FOLDER", "levels"),
        logs_folder=os.getenv("LOGS_FOLDER", "logs"),
        models_folder=os.getenv("MODELS_FOLDER", "models"),
        n_envs=int(os.getenv("N_ENVS", 4)),
        device=os.getenv("DEVICE", "auto"),
        n_builder_steps=int(os.getenv("N_BUILDER_STEPS", 1)),
        n_player_steps=int(os.getenv("N_PLAYER_STEPS", 2048)),
        max_objects=int(os.getenv("MAX_OBJECTS", os.getenv("MAX_TRAPS", 6))),
        eval_runs=int(os.getenv("EVAL_RUNS", 4)),
        max_attempts_per_run=int(os.getenv("MAX_ATTEMPTS_PER_RUN", 30)),
        max_steps_per_attempt=int(os.getenv("MAX_STEPS_PER_ATTEMPT", 400)),
        ema_alpha=float(os.getenv("EMA_ALPHA", 0.2)),
        gui_enabled=_get_bool(os.getenv("GUI_ENABLED"), False),
        gui_render_every=int(os.getenv("GUI_RENDER_EVERY", 3)),
        win_threshold=float(os.getenv("WIN_THRESHOLD", 1.0)),
        max_rounds_per_level=int(os.getenv("MAX_ROUNDS_PER_LEVEL", 10)),
        giveup_penalty=float(os.getenv("GIVEUP_PENALTY", 20.0)),
        max_jump_height=int(os.getenv("MAX_JUMP_HEIGHT", 3)),
        max_jump_dist=int(os.getenv("MAX_JUMP_DIST", 4)),
        player_learning_rate=float(os.getenv("PLAYER_LEARNING_RATE", 3e-4)),
        builder_learning_rate=float(os.getenv("BUILDER_LEARNING_RATE", 3e-4)),
        ppo_verbose=int(os.getenv("PPO_VERBOSE", 0)),
        gravity=_get_optional_float(os.getenv("GRAVITY")),
        move_speed=_get_optional_float(os.getenv("MOVE_SPEED")),
        jump_velocity=_get_optional_float(os.getenv("JUMP_VELOCITY")),
        max_fall_speed=_get_optional_float(os.getenv("MAX_FALL_SPEED")),
        stats_window=int(os.getenv("STATS_WINDOW", 20)),
        seed=_get_optional_int(os.getenv("SEED")),
    )

    parser = argparse.ArgumentParser(description="PCGRL Level Devil Pipeline V5")
    parser.add_argument("--target", type=float, dest="target_difficulty")
    parser.add_argument("--cooldown", type=float, dest="cooldown_seconds")
    parser.add_argument("--grid-width", type=int, dest="grid_width")
    parser.add_argument("--grid-height", type=int, dest="grid_height")
    parser.add_argument("--n-envs", type=int, dest="n_envs")
    parser.add_argument("--device", type=str, dest="device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--builder-steps", type=int, dest="n_builder_steps")
    parser.add_argument("--player-steps", type=int, dest="n_player_steps")
    parser.add_argument("--max-objects", "--max-traps", type=int, dest="max_objects")
    parser.add_argument("--eval-runs", type=int, dest="eval_runs")
    parser.add_argument("--max-attempts", type=int, dest="max_attempts_per_run")
    parser.add_argument("--max-steps", type=int, dest="max_steps_per_attempt")
    parser.add_argument(
        "--gui",
        dest="gui_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Live-GUI (pygame) anzeigen, die Builder-Level und Player-Bewegung visualisiert",
    )
    parser.add_argument("--gui-render-every", type=int, dest="gui_render_every")

    parser.add_argument("--win-threshold", type=float, dest="win_threshold", help="Winrate ab der ein Level als geschafft gilt (0-1)")
    parser.add_argument("--max-rounds-per-level", type=int, dest="max_rounds_per_level", help="Max. Trainingsrunden auf demselben Level, bevor aufgegeben wird")
    parser.add_argument("--giveup-penalty", type=float, dest="giveup_penalty", help="Score-Abzug bei aufgegebenem (praktisch unschaffbarem) Level")

    parser.add_argument("--max-jump-height", type=int, dest="max_jump_height", help="Reachability-Check: max. Sprunghoehe in Tiles")
    parser.add_argument("--max-jump-dist", type=int, dest="max_jump_dist", help="Reachability-Check: max. Sprungweite in Tiles")

    parser.add_argument("--player-lr", type=float, dest="player_learning_rate", help="PPO-Lernrate des Player-Agenten")
    parser.add_argument("--builder-lr", type=float, dest="builder_learning_rate", help="PPO-Lernrate des Builder-Agenten")
    parser.add_argument("--ppo-verbose", type=int, dest="ppo_verbose", choices=[0, 1, 2], help="SB3-Verbosity (0/1/2)")

    parser.add_argument("--gravity", type=float, dest="gravity")
    parser.add_argument("--move-speed", type=float, dest="move_speed")
    parser.add_argument("--jump-velocity", type=float, dest="jump_velocity")
    parser.add_argument("--max-fall-speed", type=float, dest="max_fall_speed")

    parser.add_argument("--stats-window", type=int, dest="stats_window", help="Rolling-Average-Fenster fuer die Statistik")
    parser.add_argument("--stats", dest="stats_only", action="store_true", help="Nur gespeicherte Statistik anzeigen und beenden (kein Training)")

    parser.add_argument("--seed", type=int, dest="seed", help="Zufalls-Seed fuer Reproduzierbarkeit")

    args = parser.parse_args(argv)
    for f in fields(cfg):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)

    cfg.ensure_folders()
    return cfg
