# PCGRL Pipeline V5 – Level Devil (Builder + Player RL)

Endloser Loop, in dem ein **Builder-Agent** (PPO) Level generiert und ein
**Player-Agent** (PPO, Aktionen nur Space=Jump und A/D=Links/Rechts) sie
spielt. Aus dem Spiel-Feedback lernt der Builder, Level zu bauen, deren
Schwierigkeit (gemessen in `avg_deaths`) möglichst nah an
`TARGET_DIFFICULTY` liegt. **Kein Early Stop** – die Pipeline läuft, bis
du im Cooldown-Fenster `STRG+C` drückst.

## Wichtige Annahme (bitte lesen)

Es gibt keinen offiziellen API-/Code-Zugriff auf das echte Mobile-Game
"Level Devil". `game_engine.py` ist daher eine **eigene, kompatible
Physik-Simulation** (Grid-Tiles, Schwerkraft, Sprung, Spikes, Ziel), die
gegen das vereinbarte Level-JSON-Format läuft. Die RL-Agenten trainieren
gegen diese Engine. Wenn du später Zugriff auf die echte Spiel-Engine
bekommst (z. B. über einen Screen-Reader/Emulator-Bridge), tauschst du in
`player_env.py` nur `LevelDevilEngine` gegen deinen echten Adapter aus –
Observation-/Action-Space bleiben gleich.

Start- und Ziel-Tile sitzen fix am linken/rechten Rand (wie im
Beispiel-Level). Der Builder lernt die eigentlich interessante
Entscheidung: **wo** er Spikes/Traps platziert.

## Setup

```bash
pip install -r requirements.txt
```

`.env` enthält alle Parameter (Zielschwierigkeit, Cooldown, Grid-Größe,
Ordner, `N_ENVS`, `DEVICE`, PPO-Steps, etc.) und wird automatisch von
`config.py` per `python-dotenv` gelesen. Jeder Wert kann per CLI
überschrieben werden:

```bash
python pipeline.py
python pipeline.py --cooldown 10 --target 20
python pipeline.py --n-envs 8 --device cuda
```

## Loop

```
(read Feedback) -> Gen Level -> Export Level as "level_c<CYCLE>_<TS>.json"
-> Check Syntax per Linter -> Load Level into engine/game
-> Play with the RL -> Give Feedback -> Cooldown 5s -> repeat
```

- **Linter** (`level_schema.py`): JSON-Schema, `width`/`height` in
  1–100, mindestens 1 Start- und 1 Goal-Tile. Bei Fail: Level verworfen,
  Penalty-Reward für den Builder, sofort Cooldown.
- **Feedback**: `winrate`, `avg_deaths`, `avg_time`, `score = 100 /
  (1 + |avg_deaths - target|)`, EMA-geglättet mit `alpha=0.2`. Bestes
  Level wird gemerkt, es gibt aber keinen Abbruch.
- **Cooldown**: Countdown `5 4 3 2 1 GO`. `STRG+C` **hier** ist sicher –
  `history.json` wird gespeichert, Modelle werden gesichert, sauberer
  Exit ohne Traceback.
- **Multi-Core**: `device_utils.py` erkennt GPU (`torch.cuda`) und
  CPU-Kerne automatisch und meldet z. B. `[DEVICE] Using GPU - 1 x
  NVIDIA RTX 4090` bzw. `[DEVICE] Using CPU - 12 Cores`. `N_ENVS`
  parallele `PlayerEnv`-Instanzen laufen über `SubprocVecEnv` (echte
  Multi-Core-Nutzung) für Training und Evaluation.

## Ordnerstruktur

```
pcgrl_level_devil/
  .env
  requirements.txt
  config.py            Config laden (.env + CLI)
  device_utils.py       CPU/GPU-Erkennung
  level_schema.py        Level-JSON-Schema + Linter
  game_engine.py          Physik-Simulation
  player_env.py            Gymnasium-Env fuer den Player (Space, A/D)
  builder_env.py            Gymnasium-Env fuer den Builder (Level-Generierung)
  pipeline.py                Hauptloop
  levels/                     generierte Level (*.json)
  logs/history.json            Verlauf aller Zyklen
  models/builder.zip           Builder-PPO-Gewichte (wird geladen falls vorhanden)
  models/player.zip            Player-PPO-Gewichte (wird geladen falls vorhanden)
```

## Level-JSON-Format

```json
{
  "version": "1.0",
  "name": "gen_c1",
  "width": 20,
  "height": 15,
  "target_deaths": 12,
  "cycle": 1,
  "tiles": [
    {"x": 1, "y": 13, "type": 4},
    {"x": 18, "y": 13, "type": 5},
    {"x": 5, "y": 13, "type": 2}
  ]
}
```
`type`: `0`=leer, `1`=fester Block, `2`=Falle/Spike, `4`=Start, `5`=Ziel.
Die unterste Zeile (`y = height-1`) ist standardmäßig fester Boden.

## Resume / Weiterlernen

Vorhandene `models/builder.zip` / `models/player.zip` werden beim Start
automatisch geladen und nach jedem Zyklus überschrieben – du kannst die
Pipeline also jederzeit stoppen (`STRG+C` im Cooldown) und später
nahtlos weitertrainieren.

## Getestet in dieser Umgebung

`config.py`, `level_schema.py` (Linter) und `game_engine.py` (Physik,
inkl. Landung und Tod an Spikes) sowie `builder_env.py` (Aktions-
Dekodierung, Export/Lint-Verkettung) wurden hier bereits funktional
geprüft. `torch`/`stable-baselines3` sind in dieser Sandbox nicht
installiert – teste den vollen PPO-Trainingsloop (`python pipeline.py`)
bitte einmal lokal nach `pip install -r requirements.txt`.
