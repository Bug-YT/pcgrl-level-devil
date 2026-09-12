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

    args = parser.parse_args(argv)
    for f in fields(cfg):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)

    cfg.ensure_folders()
    return cfg
