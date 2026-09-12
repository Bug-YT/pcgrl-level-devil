"""
gui.py
=======
Optionale Live-Visualisierung der Pipeline via pygame. Aktiviert per
--gui Flag (oder GUI_ENABLED=true in .env). Zeigt:

  - links: das aktuelle Level als Tile-Grid, inkl. Spielerposition
    waehrend der Player-Agent trainiert/spielt
  - rechts: ein Info-Panel mit Zyklus, Phase (GENERIERT / TRAINING /
    EVALUATION / COOLDOWN), Builder-Feedback und Player-Fortschritt

pygame ist eine OPTIONALE Abhaengigkeit: wird --gui nicht genutzt, wird
dieses Modul gar nicht importiert. Wird --gui genutzt und pygame ist
nicht installiert, faengt pipeline.py das ab und laeuft ohne GUI weiter.

Fenster schliessen oder ESC druecken setzt `want_quit = True` -- die
Pipeline prueft dieses Flag an sicheren Stellen (Cooldown, nach jedem
Trainingsblock) und faehrt dann genauso sauber herunter wie bei STRG+C.
"""

from __future__ import annotations

from typing import Callable

from level_schema import TILE_EMPTY, TILE_GOAL, TILE_PLATFORM, TILE_SOLID, TILE_START, TILE_TRAP

CELL = 26
PANEL_W = 340
MIN_HEIGHT = 380

COLORS = {
    TILE_EMPTY: (24, 24, 32),
    TILE_SOLID: (95, 95, 105),
    TILE_TRAP: (205, 45, 45),
    TILE_PLATFORM: (150, 110, 60),
    TILE_START: (60, 180, 95),
    TILE_GOAL: (230, 190, 40),
}
BG = (14, 14, 18)
PANEL_BG = (26, 26, 34)
TEXT_COLOR = (230, 230, 235)
TITLE_COLOR = (140, 200, 255)
PLAYER_COLOR = (90, 160, 240)
GRID_LINE = (40, 40, 50)


class PipelineGUI:
    """Duennes pygame-Fenster fuer Live-Feedback. Ein Objekt lebt fuer die
    gesamte Pipeline-Laufzeit; render() wird pro Frame aufgerufen."""

    def __init__(self):
        import pygame  # kann ImportError werfen -- vom Aufrufer abgefangen

        self._pg = pygame
        pygame.init()
        pygame.display.set_caption("PCGRL Level Devil – Builder & Player")
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 15)
        self.font_big = pygame.font.SysFont("menlo,consolas,monospace", 19, bold=True)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.want_quit = False

    def _ensure_window(self, width_tiles: int, height_tiles: int) -> None:
        pygame = self._pg
        w = width_tiles * CELL + PANEL_W
        h = max(height_tiles * CELL, MIN_HEIGHT)
        if self.screen is None or self.screen.get_size() != (w, h):
            self.screen = pygame.display.set_mode((w, h))

    def handle_events(self) -> bool:
        """Verarbeitet pygame-Events (Fenster schliessen, ESC). Gibt
        want_quit zurueck, damit Aufrufer sofort reagieren koennen."""
        pygame = self._pg
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.want_quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.want_quit = True
        return self.want_quit

    @staticmethod
    def _build_grid(level: dict) -> list[list[int]]:
        width, height = level["width"], level["height"]
        grid = [[TILE_EMPTY for _ in range(width)] for _ in range(height)]
        for x in range(width):
            grid[height - 1][x] = TILE_SOLID
        for t in level["tiles"]:
            x, y, typ = t["x"], t["y"], t["type"]
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = typ
        return grid

    def render(
        self,
        level: dict,
        player_pos: tuple[float, float] | None = None,
        info_lines: list[str] | None = None,
        phase: str = "",
    ) -> None:
        """Zeichnet einen Frame. player_pos ist optional (z.B. waehrend
        COOLDOWN oder direkt nach der Level-Generierung nicht vorhanden)."""
        pygame = self._pg
        width, height = level["width"], level["height"]
        self._ensure_window(width, height)

        grid = self._build_grid(level)

        self.screen.fill(BG)
        for y in range(height):
            for x in range(width):
                color = COLORS.get(grid[y][x], COLORS[TILE_EMPTY])
                rect = (x * CELL, y * CELL, CELL - 1, CELL - 1)
                pygame.draw.rect(self.screen, color, rect)

        if player_pos is not None:
            px, py = player_pos
            cx = int(px * CELL + CELL * 0.5)
            cy = int(py * CELL + CELL * 0.5)
            pygame.draw.circle(self.screen, PLAYER_COLOR, (cx, cy), max(4, CELL // 2 - 2))

        panel_x = width * CELL
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, PANEL_W, self.screen.get_height()))

        y_off = 14
        title_surf = self.font_big.render(phase or "PCGRL Pipeline", True, TITLE_COLOR)
        self.screen.blit(title_surf, (panel_x + 14, y_off))
        y_off += 32
        pygame.draw.line(self.screen, GRID_LINE, (panel_x + 14, y_off), (panel_x + PANEL_W - 14, y_off))
        y_off += 12

        for line in info_lines or []:
            surf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, (panel_x + 14, y_off))
            y_off += 22

        pygame.display.flip()
        self.handle_events()
        self.clock.tick(60)

    def close(self) -> None:
        self._pg.quit()


def make_gui_callback(gui: "PipelineGUI", level: dict, phase: str, render_every: int, info_fn: Callable[[], list[str]] | None = None):
    """Factory statt Klasse auf Modulebene: importiert
    stable_baselines3.common.callbacks.BaseCallback erst bei Bedarf, damit
    gui.py auch importierbar bleibt, wenn (noch) kein SB3 installiert ist
    und man nur die reinen Render-Funktionen von PipelineGUI nutzen will."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _GuiCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.gui = gui
            self.level = level
            self.phase = phase
            self.render_every = max(1, render_every)
            self.info_fn = info_fn
            self._counter = 0

        def _on_step(self) -> bool:
            if self.gui.want_quit:
                return False

            self._counter += 1
            if self._counter % self.render_every != 0:
                return True

            obs = self.locals.get("new_obs")
            if obs is None:
                return True

            width, height = self.level["width"], self.level["height"]
            x = (float(obs[0][0]) + 1) / 2 * width
            y = (float(obs[0][1]) + 1) / 2 * height

            info_lines = self.info_fn() if self.info_fn else []
            self.gui.render(self.level, player_pos=(x, y), info_lines=info_lines, phase=self.phase)
            return not self.gui.want_quit

    return _GuiCallback()


def decode_obs_position(obs_row, level: dict) -> tuple[float, float]:
    """Rechnet die normalisierten Player-Env-Beobachtungswerte (obs[0],
    obs[1] in [-1,1]) zurueck in Grid-Koordinaten, fuer die Anzeige
    ausserhalb des SB3-Callbacks (z.B. in der Eval-Schleife)."""
    width, height = level["width"], level["height"]
    x = (float(obs_row[0]) + 1) / 2 * width
    y = (float(obs_row[1]) + 1) / 2 * height
    return x, y