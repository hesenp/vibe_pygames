"""Unit tests for the shared star-sim helpers."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from _shared.star_common import Star, append_trail, barycenter, reset_trails  # noqa: E402


def _star(mass: float, x: float, y: float) -> Star:
    return Star(
        name="test",
        mass=mass,
        radius=10,
        color=(255, 255, 255),
        glow=(255, 255, 255),
        trail_color=(255, 255, 255),
        pos=pygame.Vector2(x, y),
        vel=pygame.Vector2(0, 0),
    )


def test_barycenter_equal_mass() -> None:
    stars = [_star(1.0, 0.0, 0.0), _star(1.0, 100.0, 0.0)]
    bc = barycenter(stars)
    assert bc.x == 50.0
    assert bc.y == 0.0


def test_barycenter_weighted() -> None:
    # 3:1 mass ratio puts the barycentre 25% of the way toward the heavier star.
    stars = [_star(3.0, 0.0, 0.0), _star(1.0, 400.0, 0.0)]
    bc = barycenter(stars)
    assert bc.x == 100.0


def test_append_trail_caps_length() -> None:
    star = _star(1.0, 0.0, 0.0)
    for i in range(50):
        star.pos.x = float(i)
        append_trail(star, max_points=10)
    assert len(star.trail) == 10
    # The newest points should win; the oldest five iterations were pushed out.
    assert star.trail[-1].x == 49.0
    assert star.trail[0].x == 40.0


def test_reset_trails_clears_all() -> None:
    stars = [_star(1.0, 0.0, 0.0), _star(1.0, 1.0, 0.0)]
    for s in stars:
        for _ in range(3):
            append_trail(s, max_points=10)
    assert all(len(s.trail) == 3 for s in stars)
    reset_trails(stars)
    assert all(s.trail == [] for s in stars)
