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

Start- und Ziel-Position sind **frei wählbar** (Action-Werte 0–3), und der
Builder kann pro Objekt-Slot `[aktiv, x, y, type]` zwischen **Trap**,
**Plattform** (einweg-begehbar) und **festem Block** wählen (`OBJECT_TYPES`
in `builder_env.py`). Bevor ein Level gespielt wird, prüft der Linter per
BFS-Erreichbarkeitscheck (`check_reachability` in `level_schema.py`), ob
das Ziel vom Start aus überhaupt erreichbar ist – unerreichbare Level
werden verworfen (Penalty-Reward), statt fälschlich als "sehr schwer"
gewertet zu werden.

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

- **Linter** (`level_schema.py`): (1) JSON-Schema, (2) `width`/`height` in
  1–100, (3) mindestens 1 Start- und 1 Goal-Tile, (4) BFS-Erreichbarkeit
  (Ziel muss vom Start aus per Sprung/Fall erreichbar sein – berücksichtigt
  `max_jump_height`/`max_jump_dist` und prüft die Sichtlinie zwischen zwei
  Zellen, damit z. B. eine durchgehende Wand korrekt als Blockade erkannt
  wird statt übersprungen zu werden). Das ist eine **topologische
  Approximation**, keine exakte Sprungkurven-Simulation – Traps blockieren
  die Erreichbarkeit bewusst nicht (der Spieler kann normalerweise
  darüberspringen; sie beeinflussen die Schwierigkeit, nicht die
  grundsätzliche Lösbarkeit). Bei Fail: Level verworfen, Penalty-Reward für
  den Builder, sofort Cooldown.
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
`type`: `0`=leer, `1`=fester Block, `2`=Falle/Spike, `3`=Plattform
(einweg-begehbar), `4`=Start, `5`=Ziel. Die unterste Zeile
(`y = height-1`) ist standardmäßig fester Boden.

## Action-Space des Builders (nach der Erweiterung)

```
[start_x, start_y, goal_x, goal_y,
 aktiv_0, x_0, y_0, type_0,
 aktiv_1, x_1, y_1, type_1,
 ... (MAX_OBJECTS Slots)]
```

Alle Werte liegen in `[-1, 1]` (Standard-Box-Action-Space für PPO).
`aktiv_i > 0` schaltet Objekt-Slot `i` frei; `type_i` wird in einen Index
über `OBJECT_TYPES = [TILE_TRAP, TILE_PLATFORM, TILE_SOLID]` gebucketed.
Weitere Tile-Typen lassen sich durch Erweitern dieser Liste hinzufügen,
ohne die Vektor-Struktur zu ändern.

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
