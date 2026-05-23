"""Shared scaffolding for the orbital-simulation games (2-stars, 3-stars, ...).

Provides the ``Star`` dataclass, a unified theme, common drawing helpers, and
a ``run_simulation`` driver. Each game supplies the bits that actually differ
(initial conditions, physics step, extra hotkeys, extra HUD text) as
callbacks.
"""

import argparse
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

# Shared dark-slate theme used across all star sims. Centralised so palette
# adjustments require touching only this file.
BACKGROUND = (5, 8, 18)
PANEL = (13, 20, 33)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
ORBIT_LINE = (71, 85, 105)
CONNECTOR = (51, 65, 85)
BARYCENTER_COLOR = (248, 250, 252)
GRID_LINE = (15, 23, 42)


@dataclass
class Star:
    name: str
    mass: float
    radius: int
    color: tuple[int, int, int]
    glow: tuple[int, int, int]
    trail_color: tuple[int, int, int]
    pos: pygame.Vector2
    vel: pygame.Vector2
    spin_angle: float = 0.0
    spin_speed: float = 0.0
    trail: list[pygame.Vector2] = field(default_factory=list)


@dataclass
class SimConfig:
    title: str
    caption: str
    hint: str
    width: int = 1280
    height: int = 820
    fps: int = 60
    max_trail_points: int = 1000
    default_time_scale: float = 1.0
    time_scale_min: float = 0.15
    time_scale_max: float = 4.0
    max_dt: float = 0.025
    grid_step: int = 80
    panel_height: int = 74

    @property
    def center(self) -> pygame.Vector2:
        return pygame.Vector2(self.width / 2, self.height / 2 + 30)


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true", help="Run with SDL's dummy video driver.")
    parser.add_argument("--frames", type=int, default=0, help="Exit after this many frames. 0 = unlimited.")
    return parser


def apply_headless_env(headless: bool) -> None:
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"


def reset_trails(stars: list[Star]) -> None:
    for star in stars:
        star.trail.clear()


def barycenter(stars: list[Star]) -> pygame.Vector2:
    total_mass = sum(star.mass for star in stars)
    weighted = pygame.Vector2()
    for star in stars:
        weighted += star.pos * star.mass
    return weighted / total_mass


def append_trail(star: Star, max_points: int) -> None:
    star.trail.append(star.pos.copy())
    if len(star.trail) > max_points:
        del star.trail[0 : len(star.trail) - max_points]


def draw_background(
    screen: pygame.Surface,
    font: pygame.font.Font,
    cfg: SimConfig,
    show_grid: bool,
) -> None:
    screen.fill(BACKGROUND)

    if show_grid:
        for x in range(0, cfg.width, cfg.grid_step):
            pygame.draw.line(screen, GRID_LINE, (x, 0), (x, cfg.height), 1)
        for y in range(0, cfg.height, cfg.grid_step):
            pygame.draw.line(screen, GRID_LINE, (0, y), (cfg.width, y), 1)

    pygame.draw.rect(screen, PANEL, (0, 0, cfg.width, cfg.panel_height))
    title = font.render(cfg.title, True, TEXT)
    hint = font.render(cfg.hint, True, MUTED)
    screen.blit(title, (24, 14))
    screen.blit(hint, (24, 42))


def draw_trail(
    screen: pygame.Surface,
    points: list[pygame.Vector2],
    color: tuple[int, int, int],
    max_points: int,
) -> None:
    if len(points) < 2:
        return

    start = max(1, len(points) - max_points)
    inv_total = 1.0 / len(points)
    for i in range(start, len(points)):
        alpha = i * inv_total
        faded = (
            max(0, min(255, int(color[0] * alpha))),
            max(0, min(255, int(color[1] * alpha))),
            max(0, min(255, int(color[2] * alpha))),
        )
        pygame.draw.line(screen, faded, points[i - 1], points[i], 2)


def draw_star(
    screen: pygame.Surface,
    star: Star,
    spin_offsets: tuple[float, ...] = (-0.45, 0.15, 0.58),
    glow_levels: tuple[tuple[int, float], ...] = ((5, 0.10), (3, 0.14), (2, 0.22)),
) -> None:
    """Draw a single star with a layered glow and three spin tick marks.

    ``glow_levels`` is a tuple of ``(radius_multiplier, alpha_scale)`` pairs.
    """
    for multiplier, alpha_scale in glow_levels:
        radius = star.radius * multiplier
        glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        alpha = int(255 * alpha_scale)
        pygame.draw.circle(glow_surface, (*star.glow, alpha), (radius, radius), radius)
        screen.blit(
            glow_surface,
            star.pos - pygame.Vector2(radius, radius),
            special_flags=pygame.BLEND_PREMULTIPLIED,
        )

    pygame.draw.circle(screen, star.color, star.pos, star.radius)
    for offset in spin_offsets:
        angle = star.spin_angle + offset
        start = star.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * (star.radius * 0.22)
        end = star.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * (star.radius * 0.82)
        pygame.draw.line(screen, (255, 255, 255), start, end, max(2, star.radius // 10))
    pygame.draw.circle(
        screen,
        (255, 255, 255),
        star.pos - pygame.Vector2(star.radius * 0.28),
        max(3, star.radius // 5),
    )


# Callback signatures used by run_simulation.
BuildFn = Callable[[], list[Star]]
StepFn = Callable[[list[Star], float], None]
KeyFn = Callable[[pygame.event.Event, list[Star]], list[Star] | None]
ExtraDrawFn = Callable[[pygame.Surface, list[Star]], None]
StatusFn = Callable[[list[Star]], str]


def _default_status(_stars: list[Star]) -> str:
    return ""


def _default_key(_event: pygame.event.Event, _stars: list[Star]) -> list[Star] | None:
    return None


def _default_extra_draw(_screen: pygame.Surface, _stars: list[Star]) -> None:
    return None


def run_simulation(
    cfg: SimConfig,
    args: argparse.Namespace,
    build_stars: BuildFn,
    step_stars: StepFn,
    extra_draw: ExtraDrawFn = _default_extra_draw,
    on_key: KeyFn = _default_key,
    status_extra: StatusFn = _default_status,
    spin_offsets: tuple[float, ...] = (-0.45, 0.15, 0.58),
) -> None:
    """Run the standard star-sim loop.

    Shared behaviour: pause/reset/trails/grid/time-scale hotkeys, panel HUD,
    trail rendering, glow rendering, headless+frame-cap support.
    """
    apply_headless_env(args.headless)
    pygame.init()
    pygame.display.set_caption(cfg.caption)
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    stars = build_stars()
    paused = False
    trails_on = True
    show_grid = False
    time_scale = cfg.default_time_scale
    running = True
    frames = 0

    while running:
        dt = min(clock.tick(cfg.fps) / 1000.0, cfg.max_dt) * time_scale

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                stars = build_stars()
            elif event.key == pygame.K_t:
                trails_on = not trails_on
                if not trails_on:
                    reset_trails(stars)
            elif event.key == pygame.K_g:
                show_grid = not show_grid
            elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                time_scale = max(cfg.time_scale_min, time_scale / 1.25)
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                time_scale = min(cfg.time_scale_max, time_scale * 1.25)
            else:
                replacement = on_key(event, stars)
                if replacement is not None:
                    stars = replacement

        if not paused:
            step_stars(stars, dt)
            for star in stars:
                append_trail(star, cfg.max_trail_points)

        draw_background(screen, font, cfg, show_grid)
        extra_draw(screen, stars)
        if trails_on:
            for star in stars:
                draw_trail(screen, star.trail, star.trail_color, cfg.max_trail_points)
        for star in stars:
            draw_star(screen, star, spin_offsets=spin_offsets)

        stats_line = (
            f"time scale: {time_scale:0.2f}x   paused: {'yes' if paused else 'no'}   fps: {clock.get_fps():0.1f}"
        )
        extra_status = status_extra(stars)
        if extra_status:
            stats_line = f"{extra_status}   {stats_line}"
        stats = font.render(stats_line, True, TEXT)
        screen.blit(stats, (cfg.width - stats.get_width() - 24, 28))
        pygame.display.flip()

        frames += 1
        if args.frames and frames >= args.frames:
            running = False

    pygame.quit()
