import argparse
import math
import os
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


WIDTH = 1280
HEIGHT = 820
FPS = 60
CENTER = pygame.Vector2(WIDTH / 2, HEIGHT / 2 + 20)

BACKGROUND = (5, 8, 18)
PANEL = (13, 20, 33)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
ORBIT_LINE = (71, 85, 105)
TRAIL_A = (250, 204, 21)
TRAIL_B = (96, 165, 250)

G = 72_000.0
SOFTENING = 12.0
STAR_A_MASS = 640.0
STAR_B_MASS = 430.0
STAR_A_RADIUS = 34
STAR_B_RADIUS = 27
SEPARATION = 390.0
DEFAULT_TIME_SCALE = 1.0
MAX_TRAIL_POINTS = 900


@dataclass
class Star:
    name: str
    mass: float
    radius: int
    color: tuple[int, int, int]
    glow: tuple[int, int, int]
    pos: pygame.Vector2
    vel: pygame.Vector2
    spin_angle: float
    spin_speed: float
    trail: list[pygame.Vector2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary star rotation simulation.")
    parser.add_argument("--headless", action="store_true", help="Run with SDL's dummy video driver.")
    parser.add_argument("--frames", type=int, default=0, help="Exit after this many frames. Useful for smoke tests.")
    return parser.parse_args()


def build_stars() -> list[Star]:
    total_mass = STAR_A_MASS + STAR_B_MASS
    left_distance = SEPARATION * STAR_B_MASS / total_mass
    right_distance = SEPARATION * STAR_A_MASS / total_mass
    angular_speed = math.sqrt(G * total_mass / (SEPARATION**3))

    star_a = Star(
        name="primary",
        mass=STAR_A_MASS,
        radius=STAR_A_RADIUS,
        color=(255, 226, 123),
        glow=(251, 191, 36),
        pos=CENTER + pygame.Vector2(-left_distance, 0),
        vel=pygame.Vector2(0, -angular_speed * left_distance),
        spin_angle=0.0,
        spin_speed=1.65,
        trail=[],
    )
    star_b = Star(
        name="companion",
        mass=STAR_B_MASS,
        radius=STAR_B_RADIUS,
        color=(147, 197, 253),
        glow=(59, 130, 246),
        pos=CENTER + pygame.Vector2(right_distance, 0),
        vel=pygame.Vector2(0, angular_speed * right_distance),
        spin_angle=math.pi * 0.35,
        spin_speed=-2.25,
        trail=[],
    )
    return [star_a, star_b]


def acceleration(target: Star, source: Star) -> pygame.Vector2:
    delta = source.pos - target.pos
    distance_sq = delta.length_squared() + SOFTENING * SOFTENING
    if distance_sq <= 0.0:
        return pygame.Vector2()

    return delta * (G * source.mass / (distance_sq * math.sqrt(distance_sq)))


def step_stars(stars: list[Star], dt: float) -> None:
    a0 = acceleration(stars[0], stars[1])
    a1 = acceleration(stars[1], stars[0])

    stars[0].vel += a0 * dt * 0.5
    stars[1].vel += a1 * dt * 0.5
    stars[0].pos += stars[0].vel * dt
    stars[1].pos += stars[1].vel * dt
    a0 = acceleration(stars[0], stars[1])
    a1 = acceleration(stars[1], stars[0])
    stars[0].vel += a0 * dt * 0.5
    stars[1].vel += a1 * dt * 0.5

    for star in stars:
        star.spin_angle = (star.spin_angle + star.spin_speed * dt) % math.tau
        star.trail.append(star.pos.copy())
        if len(star.trail) > MAX_TRAIL_POINTS:
            del star.trail[0 : len(star.trail) - MAX_TRAIL_POINTS]


def reset_trails(stars: list[Star]) -> None:
    for star in stars:
        star.trail.clear()


def barycenter(stars: list[Star]) -> pygame.Vector2:
    total_mass = sum(star.mass for star in stars)
    weighted = pygame.Vector2()
    for star in stars:
        weighted += star.pos * star.mass
    return weighted / total_mass


def draw_background(screen: pygame.Surface, font: pygame.font.Font, show_grid: bool) -> None:
    screen.fill(BACKGROUND)

    if show_grid:
        for x in range(0, WIDTH, 80):
            pygame.draw.line(screen, (15, 23, 42), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 80):
            pygame.draw.line(screen, (15, 23, 42), (0, y), (WIDTH, y), 1)

    pygame.draw.rect(screen, PANEL, (0, 0, WIDTH, 74))
    title = font.render("Binary Star Rotation", True, TEXT)
    hint = font.render("space pause   r reset   t trails   g grid   -/+ time scale   esc quit", True, MUTED)
    screen.blit(title, (24, 14))
    screen.blit(hint, (24, 42))


def draw_trail(screen: pygame.Surface, points: list[pygame.Vector2], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return5

    start = max(1, len(points) - MAX_TRAIL_POINTS)
    for i in range(start, len(points)):
        alpha = i / len(points)
        faded = tuple(max(0, min(255, int(channel * alpha))) for channel in color)
        pygame.draw.line(screen, faded, points[i - 1], points[i], 2)


def draw_star(screen: pygame.Surface, star: Star) -> None:
    for radius, alpha_scale in (
        (star.radius * 5, 0.10),
        (star.radius * 3, 0.14),
        (star.radius * 2, 0.22),
    ):
        glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        alpha = int(255 * alpha_scale)
        pygame.draw.circle(glow_surface, (*star.glow, alpha), (radius, radius), radius)
        screen.blit(glow_surface, star.pos - pygame.Vector2(radius, radius), special_flags=pygame.BLEND_PREMULTIPLIED)

    pygame.draw.circle(screen, star.color, star.pos, star.radius)
    for offset in (-0.45, 0.15, 0.58):3
        angle = star.spin_angle + offset
        start = star.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * (star.radius * 0.22)
        end = star.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * (star.radius * 0.82)
        pygame.draw.line(screen, (255, 255, 255), start, end, max(2, star.radius // 10))
    pygame.draw.circle(screen, (255, 255, 255), star.pos - pygame.Vector2(star.radius * 0.28), max(3, star.radius // 5))


def draw_scene(
    screen: pygame.Surface,
    font: pygame.font.Font,
    stars: list[Star],
    paused: bool,
    trails_on: bool,
    show_grid: bool,
    time_scale: float,
    fps: float,
) -> None:
    draw_background(screen, font, show_grid)

    center = barycenter(stars)
    pygame.draw.circle(screen, ORBIT_LINE, center, int(SEPARATION * STAR_B_MASS / (STAR_A_MASS + STAR_B_MASS)), 1)
    pygame.draw.circle(screen, ORBIT_LINE, center, int(SEPARATION * STAR_A_MASS / (STAR_A_MASS + STAR_B_MASS)), 1)
    pygame.draw.circle(screen, (248, 250, 252), center, 4)

    if trails_on:
        draw_trail(screen, stars[0].trail, TRAIL_A)
        draw_trail(screen, stars[1].trail, TRAIL_B)

    pygame.draw.line(screen, (51, 65, 85), stars[0].pos, stars[1].pos, 1)
    draw_star(screen, stars[0])
    draw_star(screen, stars[1])

    stats = font.render(
        f"time scale: {time_scale:0.2f}x   paused: {'yes' if paused else 'no'}   fps: {fps:0.1f}",
        True,
        TEXT,
    )
    screen.blit(stats, (WIDTH - stats.get_width() - 24, 28))
    pygame.display.flip()


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    pygame.init()
    pygame.display.set_caption("Binary Star Rotation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    stars = build_stars()
    paused = False
    trails_on = True
    show_grid = False
    time_scale = DEFAULT_TIME_SCALE
    running = True
    frames = 0

    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.025) * time_scale

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
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
                    time_scale = max(0.15, time_scale / 1.25)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    time_scale = min(4.0, time_scale * 1.25)

        if not paused:
            step_stars(stars, dt)

        draw_scene(screen, font, stars, paused, trails_on, show_grid, time_scale, clock.get_fps())

        frames += 1
        if args.frames and frames >= args.frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
