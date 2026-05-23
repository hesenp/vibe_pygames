"""Binary star rotation simulation.

Two gravitationally bound stars orbit their common barycentre. Demonstrates a
clean two-body system with visible spin, trails, and adjustable time scale.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame  # noqa: E402

from _shared.star_common import (  # noqa: E402
    BARYCENTER_COLOR,
    CONNECTOR,
    ORBIT_LINE,
    SimConfig,
    Star,
    apply_headless_env,
    barycenter,
    parse_common_args,
    run_simulation,
)

G = 72_000.0
SOFTENING = 12.0
STAR_A_MASS = 640.0
STAR_B_MASS = 430.0
STAR_A_RADIUS = 34
STAR_B_RADIUS = 27
SEPARATION = 390.0

CFG = SimConfig(
    title="Binary Star Rotation",
    caption="Binary Star Rotation",
    hint="space pause   r reset   t trails   g grid   -/+ time scale   esc quit",
    max_trail_points=900,
)
CENTER = pygame.Vector2(CFG.width / 2, CFG.height / 2 + 20)

TRAIL_A = (250, 204, 21)
TRAIL_B = (96, 165, 250)


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
        trail_color=TRAIL_A,
        pos=CENTER + pygame.Vector2(-left_distance, 0),
        vel=pygame.Vector2(0, -angular_speed * left_distance),
        spin_angle=0.0,
        spin_speed=1.65,
    )
    star_b = Star(
        name="companion",
        mass=STAR_B_MASS,
        radius=STAR_B_RADIUS,
        color=(147, 197, 253),
        glow=(59, 130, 246),
        trail_color=TRAIL_B,
        pos=CENTER + pygame.Vector2(right_distance, 0),
        vel=pygame.Vector2(0, angular_speed * right_distance),
        spin_angle=math.pi * 0.35,
        spin_speed=-2.25,
    )
    return [star_a, star_b]


def acceleration(target: Star, source: Star) -> pygame.Vector2:
    """Newtonian gravity with Plummer-style softening to avoid the singularity."""
    delta = source.pos - target.pos
    distance_sq = delta.length_squared() + SOFTENING * SOFTENING
    if distance_sq <= 0.0:
        return pygame.Vector2()
    return delta * (G * source.mass / (distance_sq * math.sqrt(distance_sq)))


def step_stars(stars: list[Star], dt: float) -> None:
    """Velocity-Verlet style integrator (half-kick, drift, half-kick).

    Two acceleration evaluations per step preserves second-order accuracy
    while keeping the two stars symmetric.
    """
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


def draw_extras(screen: pygame.Surface, stars: list[Star]) -> None:
    center = barycenter(stars)
    pygame.draw.circle(
        screen, ORBIT_LINE, center,
        int(SEPARATION * STAR_B_MASS / (STAR_A_MASS + STAR_B_MASS)), 1,
    )
    pygame.draw.circle(
        screen, ORBIT_LINE, center,
        int(SEPARATION * STAR_A_MASS / (STAR_A_MASS + STAR_B_MASS)), 1,
    )
    pygame.draw.circle(screen, BARYCENTER_COLOR, center, 4)
    pygame.draw.line(screen, CONNECTOR, stars[0].pos, stars[1].pos, 1)


def main() -> None:
    args = parse_common_args("Binary star rotation simulation.").parse_args()
    apply_headless_env(args.headless)
    run_simulation(CFG, args, build_stars, step_stars, extra_draw=draw_extras)


if __name__ == "__main__":
    main()
