"""
stats.py
=========
Aggregierte Statistiken ueber die gesamte Pipeline-Laufzeit. Wird von
pipeline.py nach jedem abgeschlossenen Level (geschafft ODER aufgegeben)
aktualisiert und in logs/stats.json persistiert.

`python pipeline.py --stats` zeigt die zuletzt gespeicherte Statistik an,
ohne zu trainieren.
"""

from __future__ import annotations

import json
import statistics as _st
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class StatsTracker:
    window: int = 20

    levels_generated: int = 0
    levels_lint_failed: int = 0
    levels_beaten: int = 0
    levels_given_up: int = 0

    total_training_rounds: int = 0
    total_player_env_steps: int = 0

    best_score: float = -1.0
    best_cycle: int = 0
    best_level_name: str | None = None

    session_start: float = field(default_factory=time.time)

    recent_scores: deque = field(default_factory=deque)
    recent_avg_deaths: deque = field(default_factory=deque)
    recent_winrates: deque = field(default_factory=deque)
    recent_rounds: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        # deque-maxlen laesst sich nicht direkt im dataclass-Feld setzen,
        # deshalb hier nachtraeglich mit dem gewuenschten window binden.
        self.recent_scores = deque(self.recent_scores, maxlen=self.window)
        self.recent_avg_deaths = deque(self.recent_avg_deaths, maxlen=self.window)
        self.recent_winrates = deque(self.recent_winrates, maxlen=self.window)
        self.recent_rounds = deque(self.recent_rounds, maxlen=self.window)

    def record_lint_fail(self) -> None:
        self.levels_generated += 1
        self.levels_lint_failed += 1

    def record_level_result(self, cycle: int, level_name: str, feedback: dict, rounds_used: int, player_steps_used: int) -> None:
        self.levels_generated += 1
        self.total_training_rounds += rounds_used
        self.total_player_env_steps += player_steps_used

        if feedback.get("beaten"):
            self.levels_beaten += 1
        else:
            self.levels_given_up += 1

        self.recent_scores.append(feedback["score"])
        self.recent_avg_deaths.append(feedback["avg_deaths"])
        self.recent_winrates.append(feedback["winrate"])
        self.recent_rounds.append(rounds_used)

        if feedback["score"] > self.best_score:
            self.best_score = feedback["score"]
            self.best_cycle = cycle
            self.best_level_name = level_name

    def summary_lines(self) -> list[str]:
        runtime = time.time() - self.session_start
        lines = [
            f"Laufzeit: {_fmt_duration(runtime)}",
            f"Level generiert: {self.levels_generated} (Lint-Fails: {self.levels_lint_failed})",
            f"Geschafft: {self.levels_beaten} | Aufgegeben: {self.levels_given_up}",
            f"Trainingsrunden gesamt: {self.total_training_rounds}",
            f"Player-Env-Steps gesamt: {self.total_player_env_steps:,}".replace(",", "."),
            f"Best Score: {self.best_score:.2f} @ Cycle {self.best_cycle}",
        ]
        if self.recent_scores:
            n = len(self.recent_scores)
            lines.append(f"Ø Score (letzte {n}): {_st.mean(self.recent_scores):.2f}")
            lines.append(f"Ø avg_deaths (letzte {n}): {_st.mean(self.recent_avg_deaths):.2f}")
            lines.append(f"Ø winrate (letzte {n}): {_st.mean(self.recent_winrates):.2f}")
            lines.append(f"Ø Runden/Level (letzte {n}): {_st.mean(self.recent_rounds):.1f}")
        return lines

    def to_dict(self) -> dict:
        return {
            "levels_generated": self.levels_generated,
            "levels_lint_failed": self.levels_lint_failed,
            "levels_beaten": self.levels_beaten,
            "levels_given_up": self.levels_given_up,
            "total_training_rounds": self.total_training_rounds,
            "total_player_env_steps": self.total_player_env_steps,
            "best_score": self.best_score,
            "best_cycle": self.best_cycle,
            "best_level_name": self.best_level_name,
            "session_runtime_seconds": time.time() - self.session_start,
            "recent_scores": list(self.recent_scores),
            "recent_avg_deaths": list(self.recent_avg_deaths),
            "recent_winrates": list(self.recent_winrates),
            "recent_rounds": list(self.recent_rounds),
        }

    def save(self, path: Path) -> None:
        try:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass  # Statistik ist best-effort, darf die Pipeline nicht abbrechen


def print_stats(logs_folder: str) -> None:
    """Fuer `python pipeline.py --stats`: zeigt die zuletzt gespeicherte
    Statistik an, ohne zu trainieren."""
    stats_path = Path(logs_folder) / "stats.json"
    print("=== PCGRL Pipeline -- Statistik ===")

    if not stats_path.exists():
        print(f"Noch keine Statistik vorhanden ({stats_path} fehlt). Starte die Pipeline erst einmal ohne --stats.")
        return

    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"{stats_path} ist beschaedigt/kein gueltiges JSON.")
        return

    print(f"Laufzeit der letzten Session: {_fmt_duration(data.get('session_runtime_seconds', 0))}")
    print(f"Level generiert: {data.get('levels_generated', 0)} (Lint-Fails: {data.get('levels_lint_failed', 0)})")
    print(f"Geschafft: {data.get('levels_beaten', 0)} | Aufgegeben: {data.get('levels_given_up', 0)}")
    print(f"Trainingsrunden gesamt: {data.get('total_training_rounds', 0)}")
    print(f"Player-Env-Steps gesamt: {data.get('total_player_env_steps', 0):,}".replace(",", "."))
    print(f"Best Score: {data.get('best_score', 0):.2f} @ Cycle {data.get('best_cycle', 0)} ({data.get('best_level_name')})")

    scores = data.get("recent_scores", [])
    if scores:
        n = len(scores)
        print(f"Ø Score (letzte {n}): {_st.mean(scores):.2f}")
        print(f"Ø avg_deaths (letzte {n}): {_st.mean(data.get('recent_avg_deaths', [0])):.2f}")
        print(f"Ø winrate (letzte {n}): {_st.mean(data.get('recent_winrates', [0])):.2f}")
        print(f"Ø Runden/Level (letzte {n}): {_st.mean(data.get('recent_rounds', [0])):.1f}")
