"""End-to-end smoke tests: each game must run for a handful of headless frames."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

GAMES = [
    "games/1-10000_balls/main.py",
    "games/2-stars/main.py",
    "games/3-stars/main.py",
    "games/4-nbody/main.py",
]


@pytest.mark.parametrize("game", GAMES)
def test_game_runs_headless(game: str) -> None:
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / game), "--headless", "--frames", "5"],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{game} exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_ball_sandbox_benchmark_output() -> None:
    """--benchmark must print a parseable summary line."""
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "games/1-10000_balls/main.py"),
            "--headless",
            "--frames",
            "3",
            "--benchmark",
            "--count",
            "200",
        ],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "frames=3" in result.stdout
    assert "mean=" in result.stdout
