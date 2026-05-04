import argparse
import math
import os
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


WIDTH = 1280
HEIGHT = 820
FPS = 60
CENTER = pygame.Vector2(WIDTH / 2, HEIGHT / 2 + 30)

BACKGROUND = (5, 8, 18)
PANEL = (13, 20, 33)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
ORBIT_LINE = (71, 85, 105)
BARYCENTER_COLOR = (248, 250, 252)
CONNECTOR = (51, 65, 85)

G = 86_000.0
SOFTENING = 16.0
STAR_MASS = 500.0
ORBIT_RADIUS = 245.0
DEFAULT_TIME_SCALE = 1.0
MAX_TRAIL_POINTS = 1000

STAR_COLORS = (
    ((255, 226, 123), (251, 191, 36), (250, 204, 21)),
    ((147, 197, 253), (59, 130, 246), (96, 165, 250)),
    ((252, 165, 165), (239, 68, 68), (248, 113, 113)),
)


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
    spin_angle: float
    spin_speed: float
    trail: list[pygame.Vector2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-star rotation simulation.")
    parser.add_argument("--headless", action="store_true", help="Run with SDL's dummy video driver.")
    parser.add_argument("--frames", type=int, default=0, help="Exit after this many frames. Useful for smoke tests.")
    return parser.parse_args()


def build_stars(chaotic: bool = False) -> list[Star]:
    angular_speed = math.sqrt(G * STAR_MASS / (math.sqrt(3.0) * ORBIT_RADIUS**3))
    stars = []

    for index, (color, glow, trail_color) in enumerate(STAR_COLORS):
        angle = -math.pi / 2 + index * math.tau / 3
        radial = pygame.Vector2(math.cos(angle), math.sin(angle))
        tangent = pygame.Vector2(-radial.y, radial.x)
        radius = 30 if index == 0 else 27
        velocity = tangent * (angular_speed * ORBIT_RADIUS)

        if chaotic:
            velocity += pygame.Vector2((index - 1) * 24.0, (1 - index % 2) * 18.0)

        stars.append(
            Star(
                name=f"star {index + 1}",
                mass=STAR_MASS,
                radius=radius,
                color=color,
                glow=glow,
                trail_color=trail_color,
                pos=CENTER + radial * ORBIT_RADIUS,
                vel=velocity,
                spin_angle=index * math.tau / 3,
                spin_speed=(1.55 + index * 0.32) * (-1 if index == 1 else 1),
                trail=[],
            )
        )

    return stars


def accelerations(stars: list[Star]) -> list[pygame.Vector2]:
    values = [pygame.Vector2() for _ in stars]

    for i, target in enumerate(stars):
        for j, source in enumerate(stars):
            if i == j:
                continue

            delta = source.pos - target.pos
            distance_sq = delta.length_squared() + SOFTENING * SOFTENING
            values[i] += delta * (G * source.mass / (distance_sq * math.sqrt(distance_sq)))

    return values


def step_stars(stars: list[Star], dt: float) -> None:
    first_accelerations = accelerations(stars)

    for star, accel in zip(stars, first_accelerations):
        star.vel += accel * dt * 0.5
        star.pos += star.vel * dt

    second_accelerations = accelerations(stars)
    for star, accel in zip(stars, second_accelerations):
        star.vel += accel * dt * 0.5
        star.spin_angle = (star.spin_angle + star.spin_speed * dt) % math.tau
        star.trail.append(star.pos.copy())
        if len(star.trail) > MAX_TRAIL_POINTS:
            del star.trail[0 : len(star.trail) - MAX_TRAIL_POINTS]


def barycenter(stars: list[Star]) -> pygame.Vector2:
    total_mass = sum(star.mass for star in stars)
    weighted = pygame.Vector2()
    for star in stars:
        weighted += star.pos * star.mass
    return weighted / total_mass


def reset_trails(stars: list[Star]) -> None:
    for star in stars:
        star.trail.clear()


def draw_background(screen: pygame.Surface, font: pygame.font.Font, show_grid: bool) -> None:
    screen.fill(BACKGROUND)

    if show_grid:
        for x in range(0, WIDTH, 80):
            pygame.draw.line(screen, (15, 23, 42), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 80):
            pygame.draw.line(screen, (15, 23, 42), (0, y), (WIDTH, y), 1)

    pygame.draw.rect(screen, PANEL, (0, 0, WIDTH, 74))
    title = font.render("Three Star Rotation", True, TEXT)
    hint = font.render("space pause   r reset stable   c chaos reset   t trails   g grid   -/+ time scale   esc quit", True, MUTED)
    screen.blit(title, (24, 14))
    screen.blit(hint, (24, 42))


def draw_trail(screen: pygame.Surface, points: list[pygame.Vector2], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return

    for i in range(1, len(points)):
        alpha = i / len(points)
        faded = tuple(max(0, min(255, int(channel * alpha))) for channel in color)
        pygame.draw.line(screen, faded, points[i - 1], points[i], 2)


def draw_star(screen: pygame.Surface, star: Star) -> None:
    for radius, alpha_scale in (
        (star.radius * 5, 0.09),
        (star.radius * 3, 0.14),
        (star.radius * 2, 0.22),
    ):
        glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*star.glow, int(255 * alpha_scale)), (radius, radius), radius)
        screen.blit(glow_surface, star.pos - pygame.Vector2(radius, radius), special_flags=pygame.BLEND_PREMULTIPLIED)

    pygame.draw.circle(screen, star.color, star.pos, star.radius)
    for offset in (-0.52, 0.08, 0.64):
        angle = star.spin_angle + offset
        start = star.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * (star.radius * 0.18)
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
    chaotic: bool,
    fps: float,
) -> None:
    draw_background(screen, font, show_grid)

    center = barycenter(stars)
    pygame.draw.circle(screen, ORBIT_LINE, center, int(ORBIT_RADIUS), 1)
    pygame.draw.circle(screen, BARYCENTER_COLOR, center, 4)

    if trails_on:
        for star in stars:
            draw_trail(screen, star.trail, star.trail_color)

    for i, star in enumerate(stars):
        pygame.draw.line(screen, CONNECTOR, star.pos, stars[(i + 1) % len(stars)].pos, 1)

    for star in stars:
        draw_star(screen, star)

    stats = font.render(
        f"mode: {'chaotic' if chaotic else 'stable'}   time scale: {time_scale:0.2f}x   paused: {'yes' if paused else 'no'}   fps: {fps:0.1f}",
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
    pygame.display.set_caption("Three Star Rotation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    chaotic = False
    stars = build_stars(chaotic)
    paused = False
    trails_on = True
    show_grid = False
    time_scale = DEFAULT_TIME_SCALE
    running = True
    frames = 0

    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.02) * time_scale

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    chaotic = False
                    stars = build_stars(chaotic)
                elif event.key == pygame.K_c:
                    chaotic = True
                    stars = build_stars(chaotic)
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

        draw_scene(screen, font, stars, paused, trails_on, show_grid, time_scale, chaotic, clock.get_fps())

        frames += 1
        if args.frames and frames >= args.frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
