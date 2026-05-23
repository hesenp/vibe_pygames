"""Top-level pygame menu for selecting and launching a game.

Each entry spawns the chosen game as a subprocess so the launcher stays
responsive while the game runs and can present the menu again on exit.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
WIDTH = 720
HEIGHT = 560
BACKGROUND = (5, 8, 18)
PANEL = (13, 20, 33)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
ACCENT = (96, 165, 250)
HIGHLIGHT = (30, 41, 59)


@dataclass(frozen=True)
class GameEntry:
    title: str
    description: str
    path: Path


GAMES = [
    GameEntry(
        title="Bouncing Ball Sandbox",
        description="10k balls, PyTorch tensor physics, mouse gravity field.",
        path=REPO_ROOT / "games" / "1-10000_balls" / "main.py",
    ),
    GameEntry(
        title="Binary Star Rotation",
        description="Two stars orbit their barycentre with adjustable time scale.",
        path=REPO_ROOT / "games" / "2-stars" / "main.py",
    ),
    GameEntry(
        title="Three-Star Rotation",
        description="Stable equilateral orbit with optional chaos reset.",
        path=REPO_ROOT / "games" / "3-stars" / "main.py",
    ),
    GameEntry(
        title="N-Body Sandbox",
        description="GPU-vectorised pairwise gravity for hundreds of bodies.",
        path=REPO_ROOT / "games" / "4-nbody" / "main.py",
    ),
]


def launch(entry: GameEntry) -> None:
    pygame.display.quit()
    subprocess.run([sys.executable, str(entry.path)], cwd=REPO_ROOT, check=False)
    pygame.display.init()
    pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("vibe_pygames launcher")


def main() -> None:
    pygame.init()
    pygame.display.set_caption("vibe_pygames launcher")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("consolas", 28, bold=True)
    item_font = pygame.font.SysFont("consolas", 22)
    desc_font = pygame.font.SysFont("consolas", 16)
    hint_font = pygame.font.SysFont("consolas", 14)

    selected = 0
    running = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_UP, pygame.K_k):
                    selected = (selected - 1) % len(GAMES)
                elif event.key in (pygame.K_DOWN, pygame.K_j):
                    selected = (selected + 1) % len(GAMES)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    launch(GAMES[selected])
                    screen = pygame.display.get_surface() or screen

        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, PANEL, (0, 0, WIDTH, 74))
        screen.blit(title_font.render("vibe_pygames", True, TEXT), (24, 18))
        screen.blit(hint_font.render("a curated set of pygame physics demos", True, MUTED), (24, 48))

        for idx, entry in enumerate(GAMES):
            y = 110 + idx * 96
            is_selected = idx == selected
            if is_selected:
                pygame.draw.rect(screen, HIGHLIGHT, (24, y - 8, WIDTH - 48, 80), border_radius=8)
                pygame.draw.rect(screen, ACCENT, (24, y - 8, 4, 80))
            screen.blit(item_font.render(entry.title, True, TEXT), (44, y))
            screen.blit(desc_font.render(entry.description, True, MUTED), (44, y + 32))

        hint = hint_font.render(
            "up/down: move   enter/space: launch   esc: quit",
            True,
            MUTED,
        )
        screen.blit(hint, (24, HEIGHT - 30))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
