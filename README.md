# PCGRL Pipeline V5 – Level Devil (Builder + Player RL)

Endloser Loop, in dem ein **Builder-Agent** (PPO) Level generiert und ein
**Player-Agent** (PPO, Aktionen nur Space=Jump und A/D=Links/Rechts) sie
spielt. Aus dem Spiel-Feedback lernt der Builder, Level zu bauen, deren
Schwierigkeit (gemessen in `avg_deaths`) möglichst nah an
`TARGET_DIFFICULTY` liegt. Ein Level wird so lange trainiert, bis es
**geschafft** oder als **praktisch nicht schaffbar aufgegeben** wird –
erst dann generiert der Builder ein neues. **Kein Early Stop** – die
Pipeline läuft, bis du im Cooldown-Fenster `STRG+C` drückst.

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
python pipeline.py --gui                     # mit Live-Visualisierung
python pipeline.py --gui --gui-render-every 5 # GUI, aber seltener rendern (schneller)
```

## Live-GUI

Mit `--gui` (oder `GUI_ENABLED=true` in `.env`) öffnet sich ein pygame-
Fenster, das live zeigt, was Builder und Player gerade tun:

- **Level-Grid (links):** das aktuell aktive Level inkl. Start (grün),
  Ziel (gold), Traps (rot), Plattformen (braun) und festen Blöcken (grau).
- **Spielerposition:** ein blauer Punkt bewegt sich während Training und
  Evaluation live durchs Level (Sprünge, Tode, Landungen sichtbar).
- **Info-Panel (rechts):** Zyklus, aktuelle Phase (`PLAYER TRAINING`,
  `PLAYER EVALUATION`, `COOLDOWN`, `BUILDER: LEVEL VERWORFEN`), Score,
  `avg_deaths` vs. Zielschwierigkeit, Best-Score.

**Fenster schließen oder ESC** stoppt die Pipeline genauso sauber wie
`STRG+C` im Cooldown (Modelle und `history.json` werden vorher gespeichert).

`pygame` ist optional: ist es nicht installiert, läuft die Pipeline mit
`--gui` trotzdem weiter (mit einer Warnung), nur eben ohne Fenster. Zum
Nachinstallieren: `pip install pygame`.

`GUI_RENDER_EVERY` (Default `3`) throttelt, wie oft tatsächlich gezeichnet
wird — jeden Simulationsschritt zu rendern würde das Training unnötig
verlangsamen.

## Loop

```
(read Feedback) -> Gen Level -> Export Level as "level_c<CYCLE>_<TS>.json"
-> Check Syntax per Linter -> Load Level into engine/game
-> Play with the RL (bis geschafft oder aufgegeben) -> Give Feedback
-> Cooldown 5s -> repeat
```

- **Linter** (`level_schema.py`): (1) JSON-Schema, (2) `width`/`height` in
  1–100, (3) mindestens 1 Start- und 1 Goal-Tile, (4) BFS-Erreichbarkeit
  (Ziel muss vom Start aus per Sprung/Fall erreichbar sein – berücksichtigt
  `MAX_JUMP_HEIGHT`/`MAX_JUMP_DIST` und prüft die Sichtlinie zwischen zwei
  Zellen, damit z. B. eine durchgehende Wand korrekt als Blockade erkannt
  wird statt übersprungen zu werden). Das ist eine **topologische
  Approximation**, keine exakte Sprungkurven-Simulation – Traps blockieren
  die Erreichbarkeit bewusst nicht (der Spieler kann normalerweise
  darüberspringen; sie beeinflussen die Schwierigkeit, nicht die
  grundsätzliche Lösbarkeit). Bei Fail: Level verworfen, Penalty-Reward für
  den Builder, sofort Cooldown.
- **Play-Phase (Level erst wechseln wenn geschafft/unschaffbar):** Ein
  lint-gültiges Level wird in **Runden** (Training + Evaluation)
  wiederholt gespielt – **nicht** sofort durch ein neues ersetzt. Nach
  jeder Runde prüft die Pipeline die Winrate:
  - `winrate >= WIN_THRESHOLD` (Default `1.0`) → Level gilt als
    **geschafft**. Der Builder generiert danach ein neues Level.
  - Nach `MAX_ROUNDS_PER_LEVEL` (Default `10`) Runden ohne Erfolg gilt
    das Level als **praktisch nicht schaffbar** und wird **aufgegeben**
    (obwohl der Linter es als erreichbar eingestuft hatte – die
    BFS-Prüfung ist nur eine Approximation, siehe oben). Der Score wird
    dabei um `GIVEUP_PENALTY` (Default `20`) reduziert, damit der Builder
    lernt, dass es zu schwer war.

  Jede Runde wird in der Konsole als `[RUNDE n/N] winrate=... avg_deaths=...
  score=...` geloggt; bei aktivem GUI läuft die Live-Visualisierung über
  alle Runden hinweg weiter.
- **Feedback**: `winrate`, `avg_deaths`, `avg_time`, `score = 100 /
  (1 + |avg_deaths - target|)` (nach `GIVEUP_PENALTY`-Abzug bei
  aufgegebenen Leveln), EMA-geglättet mit `EMA_ALPHA`. Bestes Level wird
  gemerkt, es gibt aber keinen Abbruch.
- **Cooldown**: Countdown `5 4 3 2 1 GO`. `STRG+C` **hier** ist sicher –
  `history.json` und `stats.json` werden gespeichert, Modelle werden
  gesichert, sauberer Exit ohne Traceback.
- **Multi-Core**: `device_utils.py` erkennt GPU (`torch.cuda`) und
  CPU-Kerne automatisch und meldet z. B. `[DEVICE] Using GPU - 1 x
  NVIDIA RTX 4090` bzw. `[DEVICE] Using CPU - 12 Cores`. `N_ENVS`
  parallele `PlayerEnv`-Instanzen laufen über `SubprocVecEnv` (echte
  Multi-Core-Nutzung) für Training und Evaluation.

## Statistik

Nach jedem abgeschlossenen Level (geschafft oder aufgegeben) aktualisiert
die Pipeline `logs/stats.json` und druckt eine Zusammenfassung in die
Konsole: Laufzeit, Anzahl generierter/gescheiterter/geschaffter/
aufgegebener Level, Gesamt-Trainingsrunden, Gesamt-Player-Env-Steps,
Best-Score sowie rollierende Durchschnittswerte (`STATS_WINDOW`, Default
`20`) für Score, `avg_deaths`, Winrate und Runden/Level.

Um nur die zuletzt gespeicherte Statistik zu sehen, ohne zu trainieren:

```bash
python pipeline.py --stats
```

## Weitere Einstellungen (Flags & `.env`)

Alle Werte stehen in `.env` und lassen sich per CLI überschreiben (siehe
`config.py` für die vollständige Liste). Auswahl der neueren Optionen:

| `.env`-Variable | CLI-Flag | Bedeutung | Default |
|---|---|---|---|
| `WIN_THRESHOLD` | `--win-threshold` | Winrate ab der ein Level als geschafft gilt | `1.0` |
| `MAX_ROUNDS_PER_LEVEL` | `--max-rounds-per-level` | Max. Runden pro Level vor "aufgegeben" | `10` |
| `GIVEUP_PENALTY` | `--giveup-penalty` | Score-Abzug bei aufgegebenem Level | `20.0` |
| `MAX_JUMP_HEIGHT` / `MAX_JUMP_DIST` | `--max-jump-height` / `--max-jump-dist` | Reachability-Check-Parameter | `3` / `4` |
| `PLAYER_LEARNING_RATE` / `BUILDER_LEARNING_RATE` | `--player-lr` / `--builder-lr` | PPO-Lernraten | `0.0003` |
| `PPO_VERBOSE` | `--ppo-verbose` | SB3-Ausgabelevel (0/1/2) | `0` |
| `GRAVITY` / `MOVE_SPEED` / `JUMP_VELOCITY` / `MAX_FALL_SPEED` | `--gravity` / `--move-speed` / `--jump-velocity` / `--max-fall-speed` | Physik-Konstanten der Engine (leer = Engine-Default) | leer |
| `STATS_WINDOW` | `--stats-window` | Rolling-Average-Fenster für die Statistik | `20` |
| `SEED` | `--seed` | Fixer Zufalls-Seed für reproduzierbare Läufe | leer (zufällig) |
| — | `--stats` | Nur gespeicherte Statistik zeigen, kein Training | — |

Beispiel für einen schnelleren, deterministischen Testlauf mit lockerer
Erfolgsschwelle:

```bash
python pipeline.py --win-threshold 0.6 --max-rounds-per-level 5 --seed 42 --player-lr 0.001
```

## Ordnerstruktur

```
pcgrl_level_devil/
  .env
  requirements.txt
  config.py            Config laden (.env + CLI)
  device_utils.py       CPU/GPU-Erkennung
  level_schema.py        Level-JSON-Schema + Linter
  game_engine.py          Physik-Simulation (konfigurierbare Konstanten)
  player_env.py            Gymnasium-Env fuer den Player (Space, A/D)
  builder_env.py            Gymnasium-Env fuer den Builder (Level-Generierung)
  gui.py                     Optionale Live-Visualisierung (pygame, --gui)
  stats.py                    Statistik-Tracking (logs/stats.json, --stats)
  pipeline.py                   Hauptloop
  levels/                        generierte Level (*.json)
  logs/history.json               Verlauf aller Zyklen/Runden
  logs/stats.json                  Aggregierte Statistik (siehe --stats)
  models/builder.zip                Builder-PPO-Gewichte (wird geladen falls vorhanden)
  models/player.zip                 Player-PPO-Gewichte (wird geladen falls vorhanden)
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
  "goal_flees": true,
  "goal_flee_chance": 0.2,
  "tiles": [
    {"x": 1, "y": 13, "type": 4},
    {"x": 18, "y": 13, "type": 5},
    {"x": 5, "y": 13, "type": 2},
    {"x": 9, "y": 13, "type": 8, "range": 3},
    {"x": 12, "y": 13, "type": 9, "trap_x": 14, "trap_y": 13}
  ]
}
```

`type`: `0`=leer, `1`=fester Block, `2`=Falle/Spike, `3`=Plattform
(einweg-begehbar), `4`=Start, `5`=Ziel, `6`=Bounce-Pad, `7`=Fake-Floor,
`8`=Buzzsaw, `9`=Coin, `10`=Gravity-Flip-Trigger, `11`=Invisible
Platform. Die unterste Zeile (`y = height-1`) ist standardmäßig fester
Boden.

## Level-Devil-inspirierte Mechaniken

Basierend auf den öffentlich beschriebenen Mechaniken des echten Spiels
habe ich eine **kuratierte, RL-taugliche Teilmenge** umgesetzt – bewusst
vereinfacht, damit sie sich sauber in das Grid-Physik-Modell einfügt:

| Tile | Verhalten |
|---|---|
| **Bounce-Pad** (`6`) | Katapultiert beim Betreten nach oben – stärker als ein normaler Sprung (`BOUNCE_VELOCITY_MULTIPLIER` in `game_engine.py`). |
| **Fake-Floor** (`7`) | Sieht wie normaler Boden aus, bricht aber nach `fake_floor_collapse_steps` Schritten Standzeit weg (Default `6`, pro Level überschreibbar). Danach fällt der Spieler durch. |
| **Buzzsaw** (`8`) | Patrouilliert horizontal um seine Spawn-Position (`"range"`-Feld, Default 3 Tiles) und tötet bei Berührung – unabhängig vom statischen Grid, als eigenes bewegliches Objekt. |
| **Coin** (`9`) | Köder: Beim Einsammeln wird optional eine `"trap_x"`/`"trap_y"`-Zielzelle zu einer Falle. Der Builder weist praktisch jedem Coin automatisch eine Falle zu (siehe unten). |
| **Gravity-Flip-Trigger** (`10`) | Kehrt beim Berühren die Schwerkraftrichtung für den Rest der Episode um (einmalig, danach verschwindet der Trigger). |
| **Invisible Platform** (`11`) | Verhält sich wie eine normale Einweg-Plattform, wird im GUI aber erst nach Berührung sichtbar (`get_render_state()["revealed_invisible"]`). |
| **Fliehendes Ziel** (`goal_flees`/`goal_flee_chance`, Level-weit) | Kommt der Spieler dem Ziel nahe, kann es sich mit einer gewissen Wahrscheinlichkeit an eine andere freie Stelle im Level teleportieren ("die Tür fliegt weg"). |

**Bewusst NICHT umgesetzt** (zu komplex für den aktuellen Scope, siehe
Kommentar am Anfang von `game_engine.py`): druckbare/zerstörbare Kisten
mit echter Schiebe-Physik, Hebel/Knopf-Kettenlogik, Projektil-Türme mit
eigener Flugbahn-Simulation, exakte Sprungkurven-Pfadplanung für ein
fliehendes Ziel. Diese bräuchten eigene Zustandsautomaten und deutlich
mehr Tile-Metadaten – ein guter nächster Ausbauschritt, aber ein eigenes
Stück Arbeit für sich.

**Bekannte Einschränkung:** Einweg-Plattformen (Plattform, Bounce-Pad,
Invisible Platform) funktionieren nur für normale Schwerkraft zuverlässig
sauber; die Kombination aus Einweg-Kollision und invertierter
Schwerkraft (nach einem Gravity-Flip) ist eine vereinfachte Näherung
(siehe Kommentar in `_collides_vertical`).

Die Reachability-Prüfung (`level_schema.py`) behandelt Fake-Floors
strukturell wie soliden Boden (kurzzeitig tragfähig genug, um als Brücke
zu zählen) und Traps/Buzzsaws/Coins/Gravity-Flip-Trigger wie bisher als
nicht-blockierend (überspringbar/umgehbar). Ein fliehendes Ziel wird an
seiner **autorisierten** Position geprüft – das Wegfliehen selbst ist
reine Laufzeit-Mechanik und macht ein Level nicht strukturell unlösbar.

### "Inspiriert von echten Level-Devil-Leveln"

Die rohe RL-Platzierung allein reproduziert die typischen Troll-Muster
des echten Spiels nicht zuverlässig (dafür bräuchte es sehr viel
Training). Deshalb legt `_apply_level_devil_flavor()` in `builder_env.py`
nach der RL-Dekodierung eine kleine, **hand-kuratierte** Konstruktionsschicht
darüber – bewusst transparent benannt, kein gelerntes Verhalten:

1. **"Bounce-Pad katapultiert in Spikes"** – hinter einem platzierten
   Bounce-Pad landet mit `BOUNCE_INTO_TRAP_CHANCE` (Default 40 %) eine
   Falle in Sprungrichtung.
2. **"Boden bricht kurz vor dem Ziel weg"** – solide/Plattform-Tiles in
   der Nähe des Ziels (`FAKE_FLOOR_NEAR_GOAL_RADIUS`, Default 4 Tiles)
   werden mit `FAKE_FLOOR_NEAR_GOAL_CHANCE` (Default 35 %) zu Fake-Floors.
3. **"Coins sind Köder"** – jeder Coin ohne bereits zugewiesene Falle
   bekommt garantiert eine (statt nur manchmal).

## Action-Space des Builders

```
[start_x, start_y, goal_x, goal_y, goal_flees, goal_flee_chance,
 aktiv_0, x_0, y_0, type_0, extra_0,
 aktiv_1, x_1, y_1, type_1, extra_1,
 ... (MAX_OBJECTS Slots)]
```

Alle Werte liegen in `[-1, 1]` (Standard-Box-Action-Space für PPO).
`aktiv_i > 0` schaltet Objekt-Slot `i` frei; `type_i` wird in einen Index
über

```python
OBJECT_TYPES = [TILE_TRAP, TILE_PLATFORM, TILE_SOLID, TILE_BOUNCE,
                TILE_FAKE_FLOOR, TILE_BUZZSAW, TILE_COIN,
                TILE_GRAVITY_FLIP, TILE_INVISIBLE_PLATFORM]
```

gebucketed. `extra_i` wird je nach Typ unterschiedlich interpretiert:
Buzzsaw → Patrouillenradius (1–5 Tiles), Coin → eine von 4 Richtungen für
die Fallen-Zielzelle, sonst ignoriert. Weitere Tile-Typen lassen sich
durch Erweitern von `OBJECT_TYPES` hinzufügen, ohne die Vektor-Struktur
zu ändern.

## Resume / Weiterlernen

Vorhandene `models/builder.zip` / `models/player.zip` werden beim Start
automatisch geladen und nach jedem Zyklus überschrieben – du kannst die
Pipeline also jederzeit stoppen (`STRG+C` im Cooldown) und später
nahtlos weitertrainieren.

**Wichtig nach diesem Update:** Der Player-Observation-Space ist von 8
auf 11 Werte gewachsen (Gravitationsrichtung + Buzzsaw-Nähe kamen dazu).
Ein **altes** `models/player.zip` von vor diesem Update lässt sich damit
**nicht** mehr laden (Shape-Mismatch) – lösch es einmalig, dann baut sich
ein neues (leeres) Modell automatisch neu auf.

## Getestet in dieser Umgebung

Diesmal konnte ich `torch` + `stable-baselines3` tatsächlich installieren
und **echtes End-to-End-Training laufen lassen** (nicht nur isolierte
Bausteine): 30+ Zyklen über mehrere Minuten, inklusive echter PPO-Updates
für Builder und Player. Dabei fand und fixte ich zwei echte Bugs, die
sich erst unter echter Last zeigten:

- Ein JSON-Serialisierungsfehler (`goal_flee_chance` blieb ein
  `numpy.float32` statt eines nativen `float`).
- Ein Floating-Point-Grenzfall in der Kollisionserkennung: Landet der
  Spieler exakt auf einer Tile-Grenze (z.B. `bottom == 14.0`), zählte
  `floor()` das faelschlich als "in der Zelle darunter", was sowohl eine
  ueberfluessige Landungs-Verifizierung (jetzt entfernt) als auch die
  horizontale Kollisionspruefung (jetzt mit Epsilon an der fernen Kante)
  durcheinanderbrachte. Betraf VOR dem Fix auch normale Boden-/
  Plattform-Landungen (nicht nur die neuen Mechaniken) – jetzt behoben
  und mit mehreren Regressionstests abgesichert.

Danach im echten Trainingslauf verifiziert: alle sechs neuen Tile-Typen
(Bounce-Pad, Fake-Floor, Buzzsaw mit `range`, Coin mit `trap_x`/`trap_y`,
Gravity-Flip, Invisible Platform) UND `goal_flees` tauchten in real vom
Builder generierten, gelinteten Leveln auf; die "Coins sind Köder"-Regel
griff bei jedem einzelnen generierten Coin; GUI-Rendering (inkl. der
neuen Overlays für Laufzeit-Mutationen) lief headless über mehrere Frames
ohne Absturz, auch mit `--gui` im vollen Pipeline-Lauf. Modelle, History
und Statistik wurden nach jedem Zyklus korrekt gespeichert.

`config.py`, `level_schema.py` (Linter inkl. BFS-Reachability-Check mit
Sichtlinien-Prüfung), `stats.py` und die einzelnen Mechaniken in
`game_engine.py` wurden zusätzlich isoliert getestet (siehe Kommentare in
den jeweiligen Testaufrufen während der Entwicklung).
