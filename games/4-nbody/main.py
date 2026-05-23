"""GPU-accelerated N-body gravitational sandbox.

A direct (O(N^2)) pairwise gravity solver written as PyTorch tensor ops, so
the whole compute graph runs on CUDA / MPS / CPU. Up to ~1500 bodies stay
interactive at 60 FPS on a modest GPU.

Two starting configurations:
  - ``solar``   : one heavy central star with lighter bodies on roughly
                  circular orbits (slightly perturbed).
  - ``cluster`` : a Gaussian blob with random velocities and mild rotation.

Run ``python main.py --help`` for CLI options.
"""

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-accelerated N-body sandbox.")
    parser.add_argument("--headless", action="store_true", help="Run with SDL's dummy video driver.")
    parser.add_argument("--frames", type=int, default=0, help="Exit after this many frames. 0 = unlimited.")
    parser.add_argument("--count", type=int, default=400, help="Number of bodies (1..MAX_BODIES).")
    parser.add_argument(
        "--mode",
        choices=["solar", "cluster"],
        default="solar",
        help="Initial configuration.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch device.",
    )
    parser.add_argument("--seed", type=int, default=17, help="RNG seed.")
    parser.add_argument("--benchmark", action="store_true", help="Print frame-time stats on exit.")
    return parser.parse_args()


def _bootstrap_sdl(headless: bool) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"


# Bootstrap SDL before importing pygame so --headless picks up the dummy driver.
_ARGS = parse_args()
_bootstrap_sdl(_ARGS.headless)

import pygame  # noqa: E402
import torch  # noqa: E402

from _shared.star_common import BACKGROUND, MUTED, PANEL, TEXT  # noqa: E402

WIDTH = 1280
HEIGHT = 820
FPS = 60
PANEL_HEIGHT = 74
MAX_BODIES = 2000
G = 8000.0
SOFTENING = 6.0
PHYSICS_SUBSTEPS = 2

BODY_PALETTE = np.array(
    [
        (252, 211, 77),
        (147, 197, 253),
        (252, 165, 165),
        (134, 239, 172),
        (196, 181, 253),
        (244, 114, 182),
        (94, 234, 212),
    ],
    dtype=np.uint8,
)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NBodySystem:
    """Direct-summation pairwise gravity, vectorised on PyTorch.

    The per-step cost is dominated by an ``(N, N, 2)`` pairwise tensor; for
    N up to ~1500 this stays well below 16 ms/frame on a modern GPU.
    """

    def __init__(self, max_bodies: int, device: torch.device, rng: np.random.Generator):
        self.max_bodies = max_bodies
        self.device = device
        self.rng = rng
        self.count = 0
        self.pos = torch.zeros((max_bodies, 2), dtype=torch.float32, device=device)
        self.vel = torch.zeros((max_bodies, 2), dtype=torch.float32, device=device)
        self.mass = torch.zeros(max_bodies, dtype=torch.float32, device=device)
        self.radius_host = np.zeros(max_bodies, dtype=np.int32)
        self.color_idx = np.zeros(max_bodies, dtype=np.uint8)

    def reset(self, count: int, mode: str) -> None:
        count = max(1, min(count, self.max_bodies))
        self.count = count
        if mode == "solar":
            pos, vel, mass, radii, colors = self._solar(count)
        elif mode == "cluster":
            pos, vel, mass, radii, colors = self._cluster(count)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        self.pos[:count] = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        self.vel[:count] = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        self.mass[:count] = torch.as_tensor(mass, dtype=torch.float32, device=self.device)
        self.radius_host[:count] = radii
        self.color_idx[:count] = colors

    def _solar(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center = np.array([WIDTH / 2, HEIGHT / 2 + PANEL_HEIGHT / 2], dtype=np.float32)
        pos = np.zeros((n, 2), dtype=np.float32)
        vel = np.zeros((n, 2), dtype=np.float32)
        mass = np.ones(n, dtype=np.float32)
        radii = np.full(n, 3, dtype=np.int32)
        colors = self.rng.integers(0, len(BODY_PALETTE), size=n, dtype=np.uint8)

        # Central heavy body.
        pos[0] = center
        mass[0] = 4000.0 if n > 1 else 1.0
        radii[0] = 10
        colors[0] = 0

        if n == 1:
            return pos, vel, mass, radii, colors

        radii_orbital = self.rng.uniform(80.0, 320.0, size=n - 1).astype(np.float32)
        angles = self.rng.uniform(0.0, math.tau, size=n - 1).astype(np.float32)
        radial = np.column_stack((np.cos(angles), np.sin(angles)))
        tangent = np.column_stack((-np.sin(angles), np.cos(angles)))
        pos[1:] = center + radial * radii_orbital[:, None]
        # Circular orbital speed: v = sqrt(G * M / r). Add a small perturbation.
        speeds = np.sqrt(G * mass[0] / radii_orbital).astype(np.float32)
        perturbation = self.rng.uniform(0.92, 1.08, size=n - 1).astype(np.float32)
        vel[1:] = tangent * (speeds * perturbation)[:, None]
        return pos, vel, mass, radii, colors

    def _cluster(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center = np.array([WIDTH / 2, HEIGHT / 2 + PANEL_HEIGHT / 2], dtype=np.float32)
        sigma = 140.0
        pos = (center + self.rng.normal(0.0, sigma, size=(n, 2))).astype(np.float32)
        # Mild rigid-body rotation seeds the angular momentum.
        relative = pos - center
        tangent = np.column_stack((-relative[:, 1], relative[:, 0])).astype(np.float32)
        vel = tangent * 0.45 + self.rng.normal(0.0, 12.0, size=(n, 2)).astype(np.float32)
        mass = self.rng.uniform(0.6, 1.4, size=n).astype(np.float32)
        radii = np.full(n, 2, dtype=np.int32)
        # Make a few bodies larger and brighter for visual interest.
        heavy = self.rng.choice(n, size=max(1, n // 80), replace=False)
        mass[heavy] *= 25.0
        radii[heavy] = 6
        colors = self.rng.integers(0, len(BODY_PALETTE), size=n, dtype=np.uint8)
        return pos, vel, mass, radii, colors

    @torch.no_grad()
    def step(self, dt: float) -> None:
        if self.count < 2:
            return
        sub_dt = dt / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            pos = self.pos[: self.count]
            vel = self.vel[: self.count]
            mass = self.mass[: self.count]

            # Pairwise displacement vectors, shape (N, N, 2).
            delta = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist_sq = (delta * delta).sum(dim=-1) + SOFTENING * SOFTENING
            inv_r3 = mass.unsqueeze(0) * dist_sq.pow(-1.5)
            inv_r3.fill_diagonal_(0.0)
            acc = (delta * inv_r3.unsqueeze(-1)).sum(dim=0) * G

            # Semi-implicit Euler is plenty stable at this timestep & softening.
            vel += acc * sub_dt
            pos += vel * sub_dt

    def positions_host(self) -> np.ndarray:
        count = self.count
        if count == 0:
            return np.empty((0, 2), dtype=np.int32)
        return self.pos[:count].to(torch.int32).detach().cpu().numpy()


def build_sprite_atlas() -> dict[tuple[int, int], pygame.Surface]:
    sprites: dict[tuple[int, int], pygame.Surface] = {}
    for color_idx, color_arr in enumerate(BODY_PALETTE):
        color = tuple(int(c) for c in color_arr)
        for radius in (2, 3, 6, 10):
            size = radius * 2 + 2
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, 90), (size // 2, size // 2), radius + 1)
            pygame.draw.circle(surf, color, (size // 2, size // 2), radius)
            sprites[(color_idx, radius)] = surf
    return sprites


def draw_scene(
    screen: pygame.Surface,
    font: pygame.font.Font,
    system: NBodySystem,
    sprites: dict[tuple[int, int], pygame.Surface],
    mode: str,
    paused: bool,
    fps: float,
    backend: str,
) -> None:
    screen.fill(BACKGROUND)
    pygame.draw.rect(screen, PANEL, (0, 0, WIDTH, PANEL_HEIGHT))

    count = system.count
    if count > 0:
        positions = system.positions_host()
        radii = system.radius_host[:count]
        colors = system.color_idx[:count]
        blit_seq = []
        for i in range(count):
            radius = int(radii[i])
            sprite = sprites.get((int(colors[i]), radius))
            if sprite is None:
                continue
            offset = sprite.get_width() // 2
            blit_seq.append((sprite, (int(positions[i, 0]) - offset, int(positions[i, 1]) - offset)))
        screen.blits(blit_seq, doreturn=False)

    title = font.render("N-Body Sandbox", True, TEXT)
    hint = font.render(
        "space pause   r reset   s solar   c cluster   n +50   m -50   esc quit",
        True,
        MUTED,
    )
    screen.blit(title, (24, 14))
    screen.blit(hint, (24, 42))

    stats = font.render(
        f"bodies: {count}   mode: {mode}   paused: {'yes' if paused else 'no'}   fps: {fps:0.1f}   backend: {backend}",
        True,
        TEXT,
    )
    screen.blit(stats, (WIDTH - stats.get_width() - 24, 28))
    pygame.display.flip()


def main() -> None:
    args = _ARGS
    torch.set_grad_enabled(False)

    pygame.init()
    pygame.display.set_caption("N-Body Sandbox")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    system = NBodySystem(MAX_BODIES, device, rng)
    mode = args.mode
    count = args.count
    system.reset(count, mode)
    sprites = build_sprite_atlas()
    backend = f"pytorch {torch.__version__} {device.type}"

    paused = False
    running = True
    frames = 0
    frame_times: list[float] = []

    while running:
        frame_start = time.perf_counter()
        dt = min(clock.tick(FPS) / 1000.0, 0.02)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    system.reset(count, mode)
                elif event.key == pygame.K_s:
                    mode = "solar"
                    system.reset(count, mode)
                elif event.key == pygame.K_c:
                    mode = "cluster"
                    system.reset(count, mode)
                elif event.key == pygame.K_n:
                    count = min(MAX_BODIES, count + 50)
                    system.reset(count, mode)
                elif event.key == pygame.K_m:
                    count = max(1, count - 50)
                    system.reset(count, mode)

        if not paused:
            system.step(dt)

        draw_scene(screen, font, system, sprites, mode, paused, clock.get_fps(), backend)
        frames += 1
        if args.benchmark:
            frame_times.append(time.perf_counter() - frame_start)
        if args.frames and frames >= args.frames:
            running = False

    pygame.quit()

    if args.benchmark and frame_times:
        millis = sorted(t * 1000.0 for t in frame_times)
        p95 = millis[int(0.95 * (len(millis) - 1))]
        print(
            f"frames={len(millis)} "
            f"mean={statistics.mean(millis):0.2f}ms "
            f"p50={statistics.median(millis):0.2f}ms "
            f"p95={p95:0.2f}ms "
            f"max={millis[-1]:0.2f}ms "
            f"bodies={system.count} mode={mode} backend={backend}"
        )


if __name__ == "__main__":
    main()
