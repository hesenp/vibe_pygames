"""Three-star rotation simulation.

Three equal-mass stars orbit their common barycentre on an equilateral
triangle. A chaos-mode hotkey nudges the velocities off the stable
configuration to produce a three-body dance.
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

G = 86_000.0
SOFTENING = 16.0
STAR_MASS = 500.0
ORBIT_RADIUS = 245.0

CFG = SimConfig(
    title="Three Star Rotation",
    caption="Three Star Rotation",
    hint="space pause   r reset stable   c chaos reset   t trails   g grid   -/+ time scale   esc quit",
)
CENTER = pygame.Vector2(CFG.width / 2, CFG.height / 2 + 30)

STAR_COLORS = (
    ((255, 226, 123), (251, 191, 36), (250, 204, 21)),
    ((147, 197, 253), (59, 130, 246), (96, 165, 250)),
    ((252, 165, 165), (239, 68, 68), (248, 113, 113)),
)


def _build(chaotic: bool = False) -> list[Star]:
    # Equilateral arrangement: angular speed chosen so net gravitational
    # acceleration is radial and matches centripetal requirement.
    angular_speed = math.sqrt(G * STAR_MASS / (math.sqrt(3.0) * ORBIT_RADIUS**3))
    stars: list[Star] = []

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
            )
        )

    return stars


def build_stable_stars() -> list[Star]:
    return _build(chaotic=False)


def accelerations(stars: list[Star]) -> list[pygame.Vector2]:
    """Pairwise N-body gravitation with softening (O(N^2)).

    For N=3 the loop overhead is negligible. For larger N consider porting
    to a torch-tensor formulation (see games/4-nbody for an example).
    """
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
    first = accelerations(stars)
    for star, accel in zip(stars, first, strict=True):
        star.vel += accel * dt * 0.5
        star.pos += star.vel * dt

    second = accelerations(stars)
    for star, accel in zip(stars, second, strict=True):
        star.vel += accel * dt * 0.5
        star.spin_angle = (star.spin_angle + star.spin_speed * dt) % math.tau


def draw_extras(screen: pygame.Surface, stars: list[Star]) -> None:
    center = barycenter(stars)
    pygame.draw.circle(screen, ORBIT_LINE, center, int(ORBIT_RADIUS), 1)
    pygame.draw.circle(screen, BARYCENTER_COLOR, center, 4)
    for i, star in enumerate(stars):
        pygame.draw.line(screen, CONNECTOR, star.pos, stars[(i + 1) % len(stars)].pos, 1)


def main() -> None:
    args = parse_common_args("Three-star rotation simulation.").parse_args()
    apply_headless_env(args.headless)

    # Track chaos state in a list (so the closure can mutate it).
    mode: list[bool] = [False]

    def status_extra(_stars: list[Star]) -> str:
        return f"mode: {'chaotic' if mode[0] else 'stable'}"

    def on_key(event: pygame.event.Event, _stars: list[Star]) -> list[Star] | None:
        if event.key == pygame.K_c:
            mode[0] = True
            return _build(chaotic=True)
        return None

    def build_with_reset_mode() -> list[Star]:
        mode[0] = False
        return build_stable_stars()

    run_simulation(
        CFG,
        args,
        build_with_reset_mode,
        step_stars,
        extra_draw=draw_extras,
        on_key=on_key,
        status_extra=status_extra,
        spin_offsets=(-0.52, 0.08, 0.64),
    )


if __name__ == "__main__":
    main()
