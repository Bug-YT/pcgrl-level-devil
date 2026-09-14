"""
pipeline.py
============
PCGRL Pipeline V5 fuer Level Devil.

Loop (KEIN Early Stop, laeuft endlos bis STRG+C im Cooldown):

  (read Feedback) -> Gen Level -> Export Level as "<name>.json"
  -> Check Syntax per Linter -> Load Level into engine/game
  -> Play with the RL (bis geschafft oder aufgegeben) -> Give Feedback
  -> Cooldown 5s -> repeat

Ein Level wird so lange weitertrainiert (mehrere "Runden": Training +
Evaluation), bis entweder die Ziel-Winrate erreicht ist ("geschafft")
oder max_rounds_per_level Runden ohne Erfolg vergangen sind ("aufgegeben"
-- praktisch nicht schaffbar, obwohl der Linter es als erreichbar
eingestuft hat). Erst dann generiert der Builder ein neues Level.

Start:
  python pipeline.py
  python pipeline.py --cooldown 10 --target 20
  python pipeline.py --stats   (nur gespeicherte Statistik anzeigen)
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import Config, load_config
from device_utils import detect_device, log_device_info
from level_schema import lint_level
from builder_env import BuilderEnv
from player_env import PlayerEnv
from stats import StatsTracker, print_stats


def _apply_learning_rate(model, learning_rate: float) -> None:
    """Setzt die Lernrate auf einem (ggf. geladenen) SB3-Modell korrekt.
    Nur model.learning_rate = x zu setzen reicht NICHT, da SB3 intern
    ueber model.lr_schedule(progress_remaining) plant -- dieser muss bei
    einem geladenen Modell explizit neu gebaut werden, sonst greift der
    alte (gespeicherte) Schedule weiter."""
    from stable_baselines3.common.utils import get_schedule_fn

    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)


def make_builder_collector(builder_env: BuilderEnv):
    """SB3-Callback, der waehrend builder_model.learn() JEDEN einzelnen
    Env-Schritt mitschneidet (info-Dict + ein Snapshot des zu diesem
    Zeitpunkt generierten Levels). Noetig, weil PPO aus Stable-Baselines3
    batch_size > 1 verlangt (https://github.com/DLR-RM/stable-baselines3/issues/440),
    der Builder-Env aber nur 1 Env hat -- also muss n_steps (und damit
    batch_size) mindestens 2 sein. Das heisst: EIN learn()-Aufruf generiert
    und spielt intern mehrere Level auf einmal ("Batch"), nicht nur eins.
    Ohne diesen Collector wuerde die Pipeline nur das LETZTE Level dieses
    Batches sehen und alle anderen (samt ihrem Feedback) stillschweigend
    verlieren."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _BuilderCollector(BaseCallback):
        def __init__(self):
            super().__init__()
            self.entries: list[tuple[dict, dict]] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            if infos:
                # last_level wurde JETZT (in diesem Schritt) von builder_env.step()
                # gesetzt -- als Snapshot sichern, bevor der naechste Schritt ihn ueberschreibt.
                self.entries.append((infos[0], builder_env.last_level))
            return True

    return _BuilderCollector()


class GuiQuit(Exception):
    """Wird ausgeloest, wenn das GUI-Fenster geschlossen (oder ESC gedrueckt)
    wurde. Wird im Hauptloop wie STRG+C behandelt: sauberer Shutdown mit
    Speichern der Modelle/Historie."""


def set_global_seed(seed: int | None) -> None:
    """Setzt Python-, NumPy- und (falls installiert) Torch-Seeds fuer
    Reproduzierbarkeit. None (Default) = kein fester Seed."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def make_player_vec_env(level: dict, cfg: Config, n_envs: int):
    """Baut ein vektorisiertes Player-Environment. Nutzt SubprocVecEnv fuer
    echten Multi-Core-Support, sobald n_envs > 1 (sonst DummyVecEnv)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    physics_kwargs = dict(
        gravity=cfg.gravity,
        move_speed=cfg.move_speed,
        jump_velocity=cfg.jump_velocity,
        max_fall_speed=cfg.max_fall_speed,
    )

    def _make(level_copy):
        def _init():
            return PlayerEnv(
                level_copy,
                max_attempts_per_run=cfg.max_attempts_per_run,
                max_steps_per_attempt=cfg.max_steps_per_attempt,
                **physics_kwargs,
            )

        return _init

    fns = [_make(level) for _ in range(n_envs)]
    if n_envs > 1:
        try:
            return SubprocVecEnv(fns, start_method="fork")
        except Exception:
            return DummyVecEnv(fns)
    return DummyVecEnv(fns)


class FeedbackTracker:
    """Haelt EMA-geglaettetes Feedback + Best-Level ueber alle Zyklen."""

    def __init__(self, alpha: float, target_difficulty: float):
        self.alpha = alpha
        self.target_difficulty = target_difficulty
        self.ema = {"winrate": 0.5, "avg_deaths": target_difficulty, "avg_time": 0.0, "score": 0.0}
        self.best_score = -1.0
        self.best_cycle = 0
        self.best_level_name = None

    def update(self, raw: dict, cycle: int, level_name: str) -> dict:
        for k in ("winrate", "avg_deaths", "avg_time", "score"):
            self.ema[k] = self.alpha * raw[k] + (1 - self.alpha) * self.ema[k]
        if raw["score"] > self.best_score:
            self.best_score = raw["score"]
            self.best_cycle = cycle
            self.best_level_name = level_name
        return dict(self.ema)


def evaluate_level(
    level: dict,
    player_model,
    cfg: Config,
    target_difficulty: float,
    gui=None,
    cycle: int = 0,
) -> tuple[float, dict]:
    """Trainiert & spielt dasselbe Level in mehreren Runden, bis entweder
    die Ziel-Winrate (cfg.win_threshold) erreicht ist ("geschafft") oder
    cfg.max_rounds_per_level Runden ohne Erfolg vergangen sind
    ("aufgegeben" -- praktisch nicht schaffbar). Zeichnet bei aktivem GUI
    live die Spielerbewegung waehrend Training und Eval.

    Gibt (score, feedback) zurueck; feedback enthaelt zusaetzlich
    "beaten" (bool) und "rounds_used" (int). Bei "aufgegeben" wird der
    Score um cfg.giveup_penalty reduziert, damit der Builder lernt, dass
    dieses Level trotz gueltigem Lint praktisch zu schwer war."""
    vec_env = make_player_vec_env(level, cfg, cfg.n_envs)
    player_model.set_env(vec_env)

    feedback = None
    beaten = False
    rounds_used = 0

    try:
        for round_num in range(1, cfg.max_rounds_per_level + 1):
            rounds_used = round_num

            callback = None
            if gui is not None:
                from gui import make_gui_callback

                def _train_info(round_num=round_num):
                    return [
                        f"Cycle: {cycle}",
                        f"Runde: {round_num}/{cfg.max_rounds_per_level}",
                        f"Level: {level['name']}",
                        f"Target avg_deaths: {target_difficulty}",
                        f"Tiles: {len(level['tiles'])}",
                        "",
                        "Player-PPO trainiert auf diesem Level ...",
                    ]

                callback = make_gui_callback(
                    gui, level, phase=f"PLAYER TRAINING (Runde {round_num})",
                    render_every=cfg.gui_render_every, info_fn=_train_info,
                )

            player_model.learn(total_timesteps=cfg.n_player_steps, reset_num_timesteps=False, progress_bar=False, callback=callback)

            if gui is not None and gui.want_quit:
                raise GuiQuit()

            # Evaluation: spiele bis genug komplette Runs eingesammelt wurden
            obs = vec_env.reset()
            completed = []
            max_eval_steps = cfg.max_attempts_per_run * cfg.max_steps_per_attempt * 4
            steps = 0
            render_counter = 0
            while len(completed) < cfg.eval_runs and steps < max_eval_steps:
                action, _ = player_model.predict(obs, deterministic=False)
                obs, rewards, dones, infos = vec_env.step(action)
                for info_e, done in zip(infos, dones):
                    if done and "run_won" in info_e:
                        completed.append(info_e)
                steps += cfg.n_envs

                if gui is not None:
                    render_counter += 1
                    if render_counter % cfg.gui_render_every == 0:
                        from gui import decode_obs_position

                        x, y = decode_obs_position(obs[0], level)
                        dynamic_state = infos[0].get("engine_diff") if infos else None
                        info_lines = [
                            f"Cycle: {cycle}",
                            f"Runde: {round_num}/{cfg.max_rounds_per_level}",
                            f"Level: {level['name']}",
                            f"Target avg_deaths: {target_difficulty}",
                            "",
                            f"Evaluiert: {len(completed)}/{cfg.eval_runs} Runs abgeschlossen",
                        ]
                        gui.render(level, player_pos=(x, y), info_lines=info_lines, phase="PLAYER EVALUATION", dynamic_state=dynamic_state)
                    if gui.want_quit:
                        raise GuiQuit()

            if not completed:
                completed = [{"run_won": False, "run_deaths": cfg.max_attempts_per_run, "run_steps": cfg.max_steps_per_attempt}]

            winrate = float(np.mean([1.0 if c["run_won"] else 0.0 for c in completed]))
            avg_deaths = float(np.mean([c["run_deaths"] for c in completed]))
            avg_time = float(np.mean([c["run_steps"] for c in completed]))
            score = 100.0 / (1.0 + abs(avg_deaths - target_difficulty))
            feedback = {"winrate": winrate, "avg_deaths": avg_deaths, "avg_time": avg_time, "score": score}

            print(
                f"      [RUNDE {round_num}/{cfg.max_rounds_per_level}] "
                f"winrate={winrate:.2f} avg_deaths={avg_deaths:.1f} score={score:.2f}"
            )

            if winrate >= cfg.win_threshold:
                beaten = True
                break
    finally:
        vec_env.close()

    feedback["beaten"] = beaten
    feedback["rounds_used"] = rounds_used
    if not beaten:
        print(f"      [AUFGEGEBEN] Level nach {rounds_used} Runden nicht geschafft -- gilt als praktisch nicht schaffbar")
        feedback["score"] = max(0.0, feedback["score"] - cfg.giveup_penalty)

    return feedback["score"], feedback


def export_level(level: dict, cfg: Config, cycle: int) -> tuple[bool, list[str], str]:
    timestamp = int(time.time())
    name = f"level_c{cycle}_{timestamp}.json"
    filepath = str(Path(cfg.levels_folder) / name)

    lint_result = lint_level(level, max_jump_height=cfg.max_jump_height, max_jump_dist=cfg.max_jump_dist)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(level, f, indent=2)

    return lint_result.ok, lint_result.errors, filepath


def append_history(cfg: Config, entry: dict) -> None:
    history_path = Path(cfg.logs_folder) / "history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def cooldown(seconds: float, gui=None, level: dict | None = None, info_lines: list[str] | None = None) -> None:
    """Cooldown mit Countdown. STRG+C hier ist sicher (kein Trap, sauberer
    Exit). Bei aktivem GUI wird der Countdown zusaetzlich im Fenster
    angezeigt; Fenster schliessen wirft hier GuiQuit (wird wie STRG+C
    behandelt)."""
    remaining = int(round(seconds))
    parts = [str(n) for n in range(remaining, 0, -1)] + ["GO"]
    print("[COOLDOWN] " + " ".join(parts))

    if gui is None or level is None:
        time.sleep(seconds)
        return

    end_time = time.time() + seconds
    while time.time() < end_time:
        secs_left = max(0, int(round(end_time - time.time())))
        lines = (info_lines or []) + ["", f"Cooldown: {secs_left}s", "(STRG+C oder Fenster schliessen zum Stoppen)"]
        gui.render(level, player_pos=None, info_lines=lines, phase="COOLDOWN")
        if gui.want_quit:
            raise GuiQuit()
        time.sleep(0.1)


def main() -> None:
    cfg = load_config(argv=sys.argv[1:])

    if cfg.stats_only:
        print_stats(cfg.logs_folder)
        return

    set_global_seed(cfg.seed)

    dev_info = detect_device(cfg.device, cfg.n_envs)
    log_device_info(dev_info)

    device_str = dev_info.device

    from stable_baselines3 import PPO

    print("=== PCGRL Pipeline gestartet ===")
    print(
        f"Config: target={cfg.target_difficulty} | cooldown={cfg.cooldown_seconds}s | "
        f"grid={cfg.grid_width}x{cfg.grid_height} | win_threshold={cfg.win_threshold} | "
        f"max_rounds_per_level={cfg.max_rounds_per_level}"
    )
    if cfg.seed is not None:
        print(f"[SEED] Fester Seed: {cfg.seed}")

    models_dir = Path(cfg.models_folder)
    builder_path = models_dir / "builder.zip"
    player_path = models_dir / "player.zip"
    stats_path = Path(cfg.logs_folder) / "stats.json"

    gui = None
    if cfg.gui_enabled:
        try:
            from gui import PipelineGUI

            gui = PipelineGUI()
            print("[GUI] Live-Visualisierung aktiv (Fenster schliessen oder ESC zum Stoppen)")
        except ImportError:
            print("[GUI] pygame nicht installiert -- GUI deaktiviert. Installiere mit: pip install pygame")

    tracker = FeedbackTracker(alpha=cfg.ema_alpha, target_difficulty=cfg.target_difficulty)
    stats = StatsTracker(window=cfg.stats_window)

    def export_and_lint_fn(level: dict):
        return export_level(level, cfg, level["cycle"])

    def play_and_score_fn(level: dict):
        return evaluate_level(level, player_model, cfg, cfg.target_difficulty, gui=gui, cycle=level["cycle"])

    builder_env = BuilderEnv(
        width=cfg.grid_width,
        height=cfg.grid_height,
        max_objects=cfg.max_objects,
        target_difficulty=cfg.target_difficulty,
        export_and_lint_fn=export_and_lint_fn,
        play_and_score_fn=play_and_score_fn,
    )

    # Builder-PPO: Episodenlaenge = 1 (one-shot Levelgenerierung). Da nur
    # 1 Env genutzt wird, verlangt SB3 trotzdem n_steps (= batch_size) >= 2
    # (siehe https://github.com/DLR-RM/stable-baselines3/issues/440) --
    # ein learn()-Aufruf generiert deshalb IMMER mindestens 2 Level auf
    # einmal (siehe make_builder_collector oben), auch wenn N_BUILDER_STEPS=1
    # gesetzt ist.
    builder_n_steps = max(2, cfg.n_builder_steps)

    if builder_path.exists():
        builder_model = PPO.load(str(builder_path), env=builder_env, device=device_str)
        _apply_learning_rate(builder_model, cfg.builder_learning_rate)
        print(f"[MODELS] Builder-Modell geladen von {builder_path}")
    else:
        builder_model = PPO(
            "MlpPolicy",
            builder_env,
            device=device_str,
            n_steps=builder_n_steps,
            batch_size=builder_n_steps,
            n_epochs=4,
            learning_rate=cfg.builder_learning_rate,
            verbose=cfg.ppo_verbose,
            seed=cfg.seed,
        )

    # Player-PPO: persistentes Modell, Env wird pro Zyklus neu gesetzt
    dummy_level = {
        "version": "1.0",
        "name": "bootstrap",
        "width": cfg.grid_width,
        "height": cfg.grid_height,
        "target_deaths": cfg.target_difficulty,
        "cycle": 0,
        "tiles": [
            {"x": 1, "y": cfg.grid_height - 2, "type": 4},
            {"x": cfg.grid_width - 2, "y": cfg.grid_height - 2, "type": 5},
        ],
    }
    bootstrap_vec_env = make_player_vec_env(dummy_level, cfg, cfg.n_envs)
    if player_path.exists():
        player_model = PPO.load(str(player_path), env=bootstrap_vec_env, device=device_str)
        _apply_learning_rate(player_model, cfg.player_learning_rate)
        print(f"[MODELS] Player-Modell geladen von {player_path}")
    else:
        player_model = PPO(
            "MlpPolicy", bootstrap_vec_env, device=device_str,
            learning_rate=cfg.player_learning_rate, verbose=cfg.ppo_verbose, seed=cfg.seed,
        )
    bootstrap_vec_env.close()

    cycle = 0
    try:
        while True:  # kein Early Stop
            last_ema = tracker.ema
            builder_env.update_context(last_ema, cycle + 1)

            print(f"\n[BATCH] Builder generiert {builder_n_steps} Level(e) (SB3 verlangt n_steps >= 2) ...")
            collector = make_builder_collector(builder_env)
            # Ein einziger learn()-Aufruf uebernimmt sowohl das Sampeln der
            # Aktionen (-> Level) als auch das Policy-Update mit den daraus
            # resultierenden Rewards -- kein separater manueller step()-Call
            # mehr noetig (der wuerde sonst zusaetzliche, ungenutzte Level
            # erzeugen und unnoetig oft den Player trainieren). Der Collector
            # sammelt dabei JEDES im Batch generierte Level einzeln ein.
            builder_model.learn(total_timesteps=builder_n_steps, reset_num_timesteps=False, callback=collector)

            for info, level_snapshot in collector.entries:
                cycle += 1
                print(f"\n===== CYCLE {cycle} =====")
                print(f"[1. READ] Target deaths: {cfg.target_difficulty} | Last winrate: {last_ema['winrate']:.2f}")
                print("[2. GEN] Builder PPO hat Level generiert")

                level_path = info.get("level_path")
                print(f"[3. EXPORT] -> {level_path}")

                if not info["lint_ok"]:
                    print(f"[4. LINT FAIL] {info['errors']}")
                    stats.record_lint_fail()
                    stats.save(stats_path)
                    append_history(cfg, {
                        "cycle": cycle,
                        "timestamp": datetime.now().isoformat(),
                        "level_file": level_path,
                        "lint_ok": False,
                        "errors": info["errors"],
                    })
                    if gui is not None:
                        gui.render(
                            level_snapshot,
                            player_pos=None,
                            info_lines=[f"Cycle: {cycle}", "LINT FEHLGESCHLAGEN:"] + [f"- {e}" for e in info["errors"][:6]],
                            phase="BUILDER: LEVEL VERWORFEN",
                        )
                    cooldown(
                        cfg.cooldown_seconds,
                        gui=gui,
                        level=level_snapshot,
                        info_lines=[f"Cycle: {cycle}", "Level verworfen (Lint-Fehler)"],
                    )
                    continue

                print("[4. LINT OK]")
                print("[5. LOAD] Level geladen -- trainiere bis geschafft oder aufgegeben")

                feedback_raw = info["feedback"]
                status = "GESCHAFFT" if feedback_raw["beaten"] else "AUFGEGEBEN"
                print(
                    f"[6. PLAY] {status} nach {feedback_raw['rounds_used']} Runde(n). "
                    f"Winrate: {feedback_raw['winrate']:.2f} | Deaths: {feedback_raw['avg_deaths']:.1f}"
                )

                ema = tracker.update(feedback_raw, cycle, level_path)
                print(
                    f"[7. FEEDBACK] Score: {feedback_raw['score']:.2f} | "
                    f"Best: {tracker.best_score:.2f} @c{tracker.best_cycle} | "
                    f"AvgDeaths: {ema['avg_deaths']:.2f}"
                )

                stats.record_level_result(
                    cycle, level_path, feedback_raw,
                    rounds_used=feedback_raw["rounds_used"],
                    player_steps_used=feedback_raw["rounds_used"] * cfg.n_player_steps,
                )
                stats.save(stats_path)
                print("      --- Statistik ---")
                for line in stats.summary_lines():
                    print(f"      {line}")

                append_history(cfg, {
                    "cycle": cycle,
                    "timestamp": datetime.now().isoformat(),
                    "level_file": level_path,
                    "lint_ok": True,
                    "feedback_raw": feedback_raw,
                    "feedback_ema": ema,
                    "best_score": tracker.best_score,
                    "best_cycle": tracker.best_cycle,
                })

                models_dir.mkdir(parents=True, exist_ok=True)
                builder_model.save(str(builder_path))
                player_model.save(str(player_path))

                cooldown(
                    cfg.cooldown_seconds,
                    gui=gui,
                    level=level_snapshot,
                    info_lines=[
                        f"Cycle: {cycle} ({status})",
                        f"Score: {feedback_raw['score']:.2f} | Best: {tracker.best_score:.2f} @c{tracker.best_cycle}",
                        f"avg_deaths: {feedback_raw['avg_deaths']:.1f} (Ziel: {cfg.target_difficulty})",
                        f"winrate: {feedback_raw['winrate']:.2f}",
                    ],
                )

    except (KeyboardInterrupt, GuiQuit) as e:
        reason = "STRG+C erkannt" if isinstance(e, KeyboardInterrupt) else "GUI-Fenster geschlossen"
        print(f"\n[STOP] {reason} -- speichere und beende sauber ...")
        models_dir.mkdir(parents=True, exist_ok=True)
        builder_model.save(str(builder_path))
        player_model.save(str(player_path))
        stats.save(stats_path)
        if gui is not None:
            gui.close()
        print("Pipeline sauber gestoppt.")
        sys.exit(0)


if __name__ == "__main__":
    main()
