"""
pipeline.py
============
PCGRL Pipeline V5 fuer Level Devil.

Loop (KEIN Early Stop, laeuft endlos bis STRG+C im Cooldown):

  (read Feedback) -> Gen Level -> Export Level as "<name>.json"
  -> Check Syntax per Linter -> Load Level into engine/game
  -> Play with the RL -> Give Feedback -> Cooldown 5s -> repeat

Start:
  python pipeline.py
  python pipeline.py --cooldown 10 --target 20
"""

from __future__ import annotations

import json
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


def make_player_vec_env(level: dict, cfg: Config, n_envs: int):
    """Baut ein vektorisiertes Player-Environment. Nutzt SubprocVecEnv fuer
    echten Multi-Core-Support, sobald n_envs > 1 (sonst DummyVecEnv)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def _make(level_copy):
        def _init():
            return PlayerEnv(
                level_copy,
                max_attempts_per_run=cfg.max_attempts_per_run,
                max_steps_per_attempt=cfg.max_steps_per_attempt,
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


def evaluate_level(level: dict, player_model, cfg: Config, target_difficulty: float) -> tuple[float, dict]:
    """Trainiert den Player kurz auf diesem Level, spielt dann eval_runs
    komplette Runs und berechnet den difficulty_score."""
    vec_env = make_player_vec_env(level, cfg, cfg.n_envs)
    player_model.set_env(vec_env)
    player_model.learn(total_timesteps=cfg.n_player_steps, reset_num_timesteps=False, progress_bar=False)

    # Evaluation: spiele bis genug komplette Runs eingesammelt wurden
    obs = vec_env.reset()
    completed = []
    max_eval_steps = cfg.max_attempts_per_run * cfg.max_steps_per_attempt * 4
    steps = 0
    while len(completed) < cfg.eval_runs and steps < max_eval_steps:
        action, _ = player_model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = vec_env.step(action)
        for info, done in zip(infos, dones):
            if done and "run_won" in info:
                completed.append(info)
        steps += cfg.n_envs

    vec_env.close()

    if not completed:
        completed = [{"run_won": False, "run_deaths": cfg.max_attempts_per_run, "run_steps": cfg.max_steps_per_attempt}]

    winrate = float(np.mean([1.0 if c["run_won"] else 0.0 for c in completed]))
    avg_deaths = float(np.mean([c["run_deaths"] for c in completed]))
    avg_time = float(np.mean([c["run_steps"] for c in completed]))
    score = 100.0 / (1.0 + abs(avg_deaths - target_difficulty))

    feedback = {"winrate": winrate, "avg_deaths": avg_deaths, "avg_time": avg_time, "score": score}
    return score, feedback


def export_level(level: dict, cfg: Config, cycle: int) -> tuple[bool, list[str], str]:
    timestamp = int(time.time())
    name = f"level_c{cycle}_{timestamp}.json"
    filepath = str(Path(cfg.levels_folder) / name)

    lint_result = lint_level(level)

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


def cooldown(seconds: float) -> None:
    """Cooldown mit Countdown. STRG+C hier ist sicher (kein Trap, sauberer Exit)."""
    remaining = int(round(seconds))
    parts = [str(n) for n in range(remaining, 0, -1)] + ["GO"]
    print("[COOLDOWN] " + " ".join(parts))
    time.sleep(seconds)


def main() -> None:
    cfg = load_config(argv=sys.argv[1:])

    dev_info = detect_device(cfg.device, cfg.n_envs)
    log_device_info(dev_info)

    device_str = dev_info.device

    from stable_baselines3 import PPO

    print("=== PCGRL Pipeline gestartet ===")
    print(f"Config: target={cfg.target_difficulty} | cooldown={cfg.cooldown_seconds}s | grid={cfg.grid_width}x{cfg.grid_height}")

    models_dir = Path(cfg.models_folder)
    builder_path = models_dir / "builder.zip"
    player_path = models_dir / "player.zip"

    tracker = FeedbackTracker(alpha=cfg.ema_alpha, target_difficulty=cfg.target_difficulty)

    def export_and_lint_fn(level: dict):
        return export_level(level, cfg, level["cycle"])

    def play_and_score_fn(level: dict):
        return evaluate_level(level, player_model, cfg, cfg.target_difficulty)

    builder_env = BuilderEnv(
        width=cfg.grid_width,
        height=cfg.grid_height,
        max_objects=cfg.max_objects,
        target_difficulty=cfg.target_difficulty,
        export_and_lint_fn=export_and_lint_fn,
        play_and_score_fn=play_and_score_fn,
    )

    # Builder-PPO: Episodenlaenge = 1 (one-shot Levelgenerierung) -> n_steps=1
    if builder_path.exists():
        builder_model = PPO.load(str(builder_path), env=builder_env, device=device_str)
        print(f"[MODELS] Builder-Modell geladen von {builder_path}")
    else:
        builder_model = PPO(
            "MlpPolicy",
            builder_env,
            device=device_str,
            n_steps=max(1, cfg.n_builder_steps),
            batch_size=max(1, cfg.n_builder_steps),
            n_epochs=4,
            verbose=0,
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
        print(f"[MODELS] Player-Modell geladen von {player_path}")
    else:
        player_model = PPO("MlpPolicy", bootstrap_vec_env, device=device_str, verbose=0)
    bootstrap_vec_env.close()

    cycle = 0
    try:
        while True:  # kein Early Stop
            cycle += 1
            print(f"\n===== CYCLE {cycle} =====")

            last_ema = tracker.ema
            print(f"[1. READ] Target deaths: {cfg.target_difficulty} | Last winrate: {last_ema['winrate']:.2f}")
            builder_env.update_context(last_ema, cycle)

            obs = builder_env.reset()[0]
            action, _ = builder_model.predict(obs, deterministic=False)
            print("[2. GEN] Builder PPO sampled new level")

            # step() fuehrt Export -> Lint -> (falls ok) Load+Play+Feedback intern aus
            _, reward, done, _, info = builder_env.step(action)

            level_path = info.get("level_path")
            print(f"[3. EXPORT] -> {level_path}")

            if not info["lint_ok"]:
                print(f"[4. LINT FAIL] {info['errors']}")
                append_history(cfg, {
                    "cycle": cycle,
                    "timestamp": datetime.now().isoformat(),
                    "level_file": level_path,
                    "lint_ok": False,
                    "errors": info["errors"],
                })
                # Builder-Update auch bei Fail (negativer Reward), dann direkt Cooldown
                builder_model.learn(total_timesteps=max(1, cfg.n_builder_steps), reset_num_timesteps=False)
                cooldown(cfg.cooldown_seconds)
                continue

            print("[4. LINT OK]")
            print("[5. LOAD] Level geladen")

            feedback_raw = info["feedback"]
            print(
                f"[6. PLAY] Player PPO trainierte & spielte {cfg.eval_runs} Runs parallel. "
                f"Win: {feedback_raw['winrate'] >= 0.5} | Deaths: {feedback_raw['avg_deaths']:.1f}"
            )

            ema = tracker.update(feedback_raw, cycle, level_path)
            print(
                f"[7. FEEDBACK] Score: {feedback_raw['score']:.2f} | "
                f"Best: {tracker.best_score:.2f} @c{tracker.best_cycle} | "
                f"AvgDeaths: {ema['avg_deaths']:.2f}"
            )

            # Builder-Policy mit dem gerade erhaltenen Reward aktualisieren
            builder_model.learn(total_timesteps=max(1, cfg.n_builder_steps), reset_num_timesteps=False)

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

            cooldown(cfg.cooldown_seconds)

    except KeyboardInterrupt:
        print("\n[STOP] STRG+C erkannt waehrend Cooldown -- speichere und beende sauber ...")
        models_dir.mkdir(parents=True, exist_ok=True)
        builder_model.save(str(builder_path))
        player_model.save(str(player_path))
        print("Pipeline sauber gestoppt.")
        sys.exit(0)


if __name__ == "__main__":
    main()
