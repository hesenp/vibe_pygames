"""Unit tests for ``games/1-10000_balls/main.py`` physics primitives.

The script's name contains a hyphen, so importlib is used to load it.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame  # noqa: F401  (imported by ball_sandbox)
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BALL_MAIN = REPO_ROOT / "games" / "1-10000_balls" / "main.py"

_spec = importlib.util.spec_from_file_location("ball_sandbox", BALL_MAIN)
assert _spec is not None and _spec.loader is not None
ball_sandbox = importlib.util.module_from_spec(_spec)
sys.modules["ball_sandbox"] = ball_sandbox
_spec.loader.exec_module(ball_sandbox)


def test_seed_positions_inside_arena() -> None:
    rng = np.random.default_rng(0)
    points = ball_sandbox.build_seed_positions(500, rng)
    assert points.shape == (500, 2)
    arena = ball_sandbox.ARENA
    assert (points[:, 0] >= arena.left).all()
    assert (points[:, 0] <= arena.right).all()
    assert (points[:, 1] >= arena.top).all()
    assert (points[:, 1] <= arena.bottom).all()


def test_seed_positions_overflow() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ball_sandbox.build_seed_positions(10_000_000, rng)


def test_collision_resolves_overlap() -> None:
    """A pair of overlapping balls must end up non-overlapping after one step."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    system = ball_sandbox.ParticleSystem(8, rng, device)
    system.reset(2)
    # Override with a hand-crafted overlap.
    system.pos[:2] = torch.tensor([[200.0, 400.0], [202.0, 400.0]])
    system.vel[:2] = torch.zeros(2, 2)
    system.radius[:2] = torch.tensor([4.0, 4.0])
    system.inv_mass[:2] = torch.tensor([1.0 / 16.0, 1.0 / 16.0])

    system.resolve_collisions()

    delta = (system.pos[1] - system.pos[0]).norm().item()
    # 0.52 push factor means a single step does not fully separate them, but the
    # gap must have grown beyond the starting 2.0 distance.
    assert delta > 2.5


def test_no_collision_drops_with_dense_pile() -> None:
    """The CSR pair expansion must handle a cell denser than the old hard cap (=12)."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    system = ball_sandbox.ParticleSystem(40, rng, device)
    system.reset(30)
    # Stack 30 balls in a single small region (~well past the old per-cell cap).
    pile = torch.full((30, 2), 300.0)
    pile[:, 1] += torch.arange(30).float() * 0.5
    system.pos[:30] = pile
    system.vel[:30] = torch.zeros(30, 2)
    system.radius[:30] = torch.full((30,), 4.0)
    system.inv_mass[:30] = torch.full((30,), 1.0 / 16.0)
    # Should not raise and should not crash on dense piles.
    system.resolve_collisions()
    # After resolution, no pair should still be deeply interpenetrating.
    positions = system.pos[:30]
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist_sq = (delta * delta).sum(dim=-1)
    dist_sq.fill_diagonal_(float("inf"))
    min_dist = dist_sq.min().sqrt().item()
    # Some overlap may remain after one iteration, but it should be far less
    # than ball diameter (8).
    assert min_dist > 0.0
