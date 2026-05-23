"""High-performance bouncing-ball sandbox.

Physics runs on PyTorch tensors (CUDA/MPS/CPU auto-detected) using a uniform
spatial-hash broad phase. Rendering uses a small pre-rendered sprite atlas
and a single ``screen.blits`` call so the per-frame Python overhead does not
scale with ball count.

Run ``python main.py --help`` for CLI options.
"""

import argparse
import math
import os
import statistics
import time

import numpy as np

WIDTH = 1280
HEIGHT = 820
FPS = 60
BACKGROUND = (17, 24, 39)
START_BALLS = 10_000
MAX_BALLS = 12_000
MIN_RADIUS = 2
MAX_RADIUS = 4
CELL_SIZE = MAX_RADIUS * 2 + 1
GRAVITY = 160.0
AIR_DRAG = 0.999
WALL_BOUNCE = 0.97
FLOOR_BOUNCE = 0.92
PHYSICS_SUBSTEPS = 2
FIELD_RADIUS = 150.0
FIELD_STRENGTH = 48_000.0
FIELD_RING_COLOR = (125, 211, 252)
PALETTE = np.array(
    [
        (96, 165, 250),
        (52, 211, 153),
        (248, 113, 113),
        (251, 191, 36),
        (196, 181, 253),
        (244, 114, 182),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-performance bouncing-ball sandbox.")
    parser.add_argument("--headless", action="store_true", help="Run with SDL's dummy video driver.")
    parser.add_argument("--frames", type=int, default=0, help="Exit after this many frames. 0 = unlimited.")
    parser.add_argument("--count", type=int, default=START_BALLS, help="Initial ball count.")
    parser.add_argument("--no-gravity", action="store_true", help="Start with gravity disabled.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch device (default: auto-detect).",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print frame-time statistics on exit.",
    )
    return parser.parse_args()


def _bootstrap_sdl(headless: bool) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"


# pygame and torch are imported lazily after _bootstrap_sdl so the dummy
# SDL driver picks up correctly when --headless is used.
import pygame  # noqa: E402
import torch  # noqa: E402

ARENA = pygame.Rect(24, 72, WIDTH - 48, HEIGHT - 120)
ARENA_LEFT = ARENA.left
ARENA_RIGHT = ARENA.right
ARENA_TOP = ARENA.top
ARENA_BOTTOM = ARENA.bottom
GRID_WIDTH = (ARENA.width + CELL_SIZE - 1) // CELL_SIZE
GRID_HEIGHT = (ARENA.height + CELL_SIZE - 1) // CELL_SIZE
TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT


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


def build_seed_positions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Jittered grid sampling of ``count`` non-overlapping seed positions."""
    spacing = CELL_SIZE
    padding = MAX_RADIUS + 2
    xs = np.arange(ARENA.left + padding, ARENA.right - padding, spacing, dtype=np.float32)
    ys = np.arange(ARENA.top + padding, ARENA.bottom - padding, spacing, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    if count > len(grid):
        raise ValueError(f"Requested {count} balls, but arena seed grid only fits {len(grid)}.")

    order = rng.permutation(len(grid))[:count]
    points = grid[order].copy()
    jitter = rng.uniform(-0.65, 0.65, size=points.shape).astype(np.float32)
    points += jitter
    return points


class ParticleSystem:
    """GPU-resident particle store with uniform-grid broad-phase collisions.

    Broad phase
    -----------
    Each frame, particles are bucketed into a uniform spatial grid (CSR-style:
    sorted by cell, with per-cell ``starts``/``counts`` offsets). For every
    particle we visit its 9 neighbour cells and expand the full candidate
    list via ``repeat_interleave``. There is no fixed per-cell capacity, so
    arbitrarily dense clumps are resolved correctly.
    """

    def __init__(self, max_particles: int, rng: np.random.Generator, device: torch.device):
        self.max_particles = max_particles
        self.rng = rng
        self.device = device
        self.count = 0
        self.pos = torch.zeros((max_particles, 2), dtype=torch.float32, device=device)
        self.vel = torch.zeros((max_particles, 2), dtype=torch.float32, device=device)
        self.radius = torch.zeros(max_particles, dtype=torch.float32, device=device)
        self.inv_mass = torch.zeros(max_particles, dtype=torch.float32, device=device)
        # Host-side mirrors of constant-per-reset fields (radius, colour).
        # These never need to round-trip through the device during draw.
        self.radius_host = np.zeros(max_particles, dtype=np.int32)
        self.color_idx = np.zeros(max_particles, dtype=np.uint8)
        # Pre-allocated scratch tensors for collision resolution.
        self._position_delta = torch.zeros_like(self.pos)
        self._velocity_delta = torch.zeros_like(self.vel)
        self._neighbor_offsets = torch.tensor(
            [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),  (0, 0),  (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ],
            dtype=torch.long,
            device=device,
        )
        self._slot_arange = torch.arange(9, device=device, dtype=torch.long)

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def reset(self, count: int = START_BALLS) -> None:
        count = max(0, min(count, self.max_particles))
        positions = build_seed_positions(count, self.rng) if count > 0 else np.empty((0, 2), np.float32)
        radii = self.rng.integers(MIN_RADIUS, MAX_RADIUS + 1, size=count, dtype=np.int16).astype(np.float32)
        angles = self.rng.uniform(0.0, math.tau, size=count).astype(np.float32)
        speeds = self.rng.uniform(12.0, 68.0, size=count).astype(np.float32)
        velocity = np.column_stack(
            (
                np.cos(angles) * speeds,
                np.sin(angles) * speeds - 10.0,
            )
        ).astype(np.float32) if count > 0 else np.empty((0, 2), np.float32)
        colors = self.rng.integers(0, len(PALETTE), size=count, dtype=np.uint8)

        self.count = count
        if count == 0:
            return
        self.pos[:count] = self._tensor(positions)
        self.vel[:count] = self._tensor(velocity)
        self.radius[:count] = self._tensor(radii)
        self.inv_mass[:count] = 1.0 / torch.clamp(self.radius[:count] * self.radius[:count], min=1.0)
        # Cache CPU-side mirrors so render_arrays() does not need device→host copies for them.
        self.radius_host[:count] = radii.astype(np.int32)
        self.color_idx[:count] = colors

    def clear(self) -> None:
        self.count = 0

    @torch.no_grad()
    def apply_mouse_field(self, mouse_pos: tuple[int, int], dt: float) -> None:
        if self.count == 0:
            return

        positions = self.pos[: self.count]
        velocity = self.vel[: self.count]
        mouse = torch.tensor(mouse_pos, dtype=torch.float32, device=self.device)
        delta = mouse - positions
        distance_sq = (delta * delta).sum(dim=1) + 36.0
        mask = distance_sq < FIELD_RADIUS * FIELD_RADIUS
        pull = torch.where(mask, (FIELD_STRENGTH * dt) / distance_sq, torch.zeros_like(distance_sq))
        velocity += delta * pull.unsqueeze(1)

    @torch.no_grad()
    def integrate(self, dt: float, gravity_on: bool) -> None:
        if self.count == 0:
            return

        positions = self.pos[: self.count]
        velocity = self.vel[: self.count]
        radii = self.radius[: self.count]

        if gravity_on:
            velocity[:, 1] += GRAVITY * dt

        velocity *= AIR_DRAG
        positions += velocity * dt

        left = ARENA_LEFT + radii
        right = ARENA_RIGHT - radii
        top = ARENA_TOP + radii
        bottom = ARENA_BOTTOM - radii

        hit_left = positions[:, 0] < left
        hit_right = positions[:, 0] > right
        positions[:, 0] = torch.clamp(positions[:, 0], min=left, max=right)
        velocity[:, 0] = torch.where(hit_left | hit_right, -velocity[:, 0] * WALL_BOUNCE, velocity[:, 0])

        hit_top = positions[:, 1] < top
        hit_bottom = positions[:, 1] > bottom
        positions[:, 1] = torch.clamp(positions[:, 1], min=top, max=bottom)
        velocity[:, 1] = torch.where(hit_top, -velocity[:, 1] * WALL_BOUNCE, velocity[:, 1])
        velocity[:, 1] = torch.where(hit_bottom, -velocity[:, 1] * FLOOR_BOUNCE, velocity[:, 1])
        velocity[:, 0] = torch.where(hit_bottom, velocity[:, 0] * 0.998, velocity[:, 0])

    @torch.no_grad()
    def resolve_collisions(self) -> None:
        count = self.count
        if count < 2:
            return

        positions = self.pos[:count]
        cell_x = torch.div(positions[:, 0] - ARENA_LEFT, CELL_SIZE, rounding_mode="floor").to(torch.long)
        cell_y = torch.div(positions[:, 1] - ARENA_TOP, CELL_SIZE, rounding_mode="floor").to(torch.long)
        cell_x.clamp_(0, GRID_WIDTH - 1)
        cell_y.clamp_(0, GRID_HEIGHT - 1)
        cell_ids = cell_y * GRID_WIDTH + cell_x

        # CSR layout: sort particles by cell so [starts[c]:starts[c]+counts[c]]
        # is the contiguous range of particle ids belonging to cell c.
        _, order = torch.sort(cell_ids)
        counts = torch.bincount(cell_ids, minlength=TOTAL_CELLS)
        starts = torch.cumsum(counts, dim=0) - counts

        # For each particle, its 9 neighbour cells (with off-grid masked out).
        neighbor_x = cell_x.unsqueeze(1) + self._neighbor_offsets[:, 0]  # (N, 9)
        neighbor_y = cell_y.unsqueeze(1) + self._neighbor_offsets[:, 1]
        valid_neighbor = (
            (neighbor_x >= 0)
            & (neighbor_x < GRID_WIDTH)
            & (neighbor_y >= 0)
            & (neighbor_y < GRID_HEIGHT)
        )
        neighbor_x = neighbor_x.clamp(0, GRID_WIDTH - 1)
        neighbor_y = neighbor_y.clamp(0, GRID_HEIGHT - 1)
        neighbor_cells = neighbor_y * GRID_WIDTH + neighbor_x  # (N, 9)

        # Per (i, n): how many candidates live in that neighbour cell.
        neighbor_counts = (counts[neighbor_cells] * valid_neighbor.long()).reshape(-1)
        neighbor_starts = starts[neighbor_cells].reshape(-1)

        # Expand to one entry per candidate pair. group_ids[k] identifies the
        # (i*9 + n) slot owning the k-th candidate; pair_j gathers from order.
        group_ids = torch.repeat_interleave(
            torch.arange(neighbor_counts.numel(), device=self.device), neighbor_counts
        )
        if group_ids.numel() == 0:
            return

        cum_offsets = torch.cumsum(neighbor_counts, dim=0) - neighbor_counts
        slots_within = (
            torch.arange(group_ids.numel(), device=self.device, dtype=torch.long)
            - cum_offsets[group_ids]
        )
        pair_j = order[neighbor_starts[group_ids] + slots_within]
        pair_i = group_ids // 9

        # Each unordered pair appears twice (once with i<j, once with j<i); keep one.
        valid = pair_i < pair_j
        if not bool(valid.any()):
            return
        pair_i = pair_i[valid]
        pair_j = pair_j[valid]
        self._resolve_pair_batch(pair_i, pair_j)

    def _resolve_pair_batch(self, pair_i: torch.Tensor, pair_j: torch.Tensor) -> None:
        positions = self.pos[: self.count]
        velocity = self.vel[: self.count]
        radii = self.radius[: self.count]
        inv_mass = self.inv_mass[: self.count]

        delta = positions[pair_j] - positions[pair_i]
        min_distance = radii[pair_i] + radii[pair_j]
        distance_sq = (delta * delta).sum(dim=1)
        overlaps = (distance_sq > 1.0e-8) & (distance_sq < min_distance * min_distance)
        if not bool(overlaps.any()):
            return

        pair_i = pair_i[overlaps]
        pair_j = pair_j[overlaps]
        delta = delta[overlaps]
        min_distance = min_distance[overlaps]
        distance = torch.sqrt(distance_sq[overlaps])
        normal = delta / distance.unsqueeze(1)

        overlap = (min_distance - distance) * 0.52
        correction = normal * overlap.unsqueeze(1)
        position_delta = self._position_delta[: self.count]
        position_delta.zero_()
        position_delta.index_add_(0, pair_i, -correction)
        position_delta.index_add_(0, pair_j, correction)
        positions += position_delta

        relative_velocity = velocity[pair_j] - velocity[pair_i]
        impact_speed = (relative_velocity * normal).sum(dim=1)
        approaching = impact_speed < 0.0
        if not bool(approaching.any()):
            return

        pair_i = pair_i[approaching]
        pair_j = pair_j[approaching]
        normal = normal[approaching]
        impact_speed = impact_speed[approaching]
        inverse_mass_sum = inv_mass[pair_i] + inv_mass[pair_j]
        # 1.96 == (1 + restitution); pick e=0.96 so collisions damp slightly.
        impulse = -1.96 * impact_speed / inverse_mass_sum
        impulse_vector = normal * impulse.unsqueeze(1)

        velocity_delta = self._velocity_delta[: self.count]
        velocity_delta.zero_()
        velocity_delta.index_add_(0, pair_i, -impulse_vector * inv_mass[pair_i].unsqueeze(1))
        velocity_delta.index_add_(0, pair_j, impulse_vector * inv_mass[pair_j].unsqueeze(1))
        velocity += velocity_delta

    def step(self, dt: float, gravity_on: bool, field_active: bool, mouse_pos: tuple[int, int]) -> None:
        sub_dt = dt / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            if field_active:
                self.apply_mouse_field(mouse_pos, sub_dt)
            self.integrate(sub_dt, gravity_on)
            self.resolve_collisions()

    def warmup(self) -> None:
        if self.count == 0:
            return

        self.integrate(0.0, False)
        self.resolve_collisions()

    def positions_host(self) -> np.ndarray:
        """Return integer positions on CPU. The only device→host copy per frame."""
        count = self.count
        if count == 0:
            return np.empty((0, 2), dtype=np.int32)
        return self.pos[:count].to(torch.int32).detach().cpu().numpy()


def build_sprite_atlas() -> list[pygame.Surface]:
    """One sprite per (colour, radius). Flattened so lookup is a single index."""
    radius_count = MAX_RADIUS - MIN_RADIUS + 1
    sprites: list[pygame.Surface] = []
    for color_idx in range(len(PALETTE)):
        color = tuple(int(c) for c in PALETTE[color_idx])
        for radius in range(MIN_RADIUS, MAX_RADIUS + 1):
            size = radius * 2 + 2
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size // 2, size // 2), radius)
            sprites.append(surf)
    assert len(sprites) == len(PALETTE) * radius_count
    return sprites


def sprite_keys(color_idx: np.ndarray, radii: np.ndarray) -> np.ndarray:
    radius_count = MAX_RADIUS - MIN_RADIUS + 1
    return color_idx.astype(np.int32) * radius_count + (radii - MIN_RADIUS)


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    system: ParticleSystem,
    gravity_on: bool,
    paused: bool,
    field_active: bool,
    fps: float,
    backend: str,
) -> None:
    title = font.render("Bouncing Ball Sandbox", True, (241, 245, 249))
    stats = font.render(
        f"balls: {system.count}   gravity: {'on' if gravity_on else 'off'}   "
        f"field: {'active' if field_active else 'idle'}   paused: {'yes' if paused else 'no'}   "
        f"fps: {fps:0.1f}   backend: {backend}",
        True,
        (191, 219, 254),
    )
    help_text = font.render(
        "hold left mouse: drag   space: pause   g: gravity   c: clear   r: reset",
        True,
        (148, 163, 184),
    )
    screen.blit(title, (24, 18))
    screen.blit(stats, (24, 42))
    screen.blit(help_text, (24, HEIGHT - 30))


def draw_scene(
    screen: pygame.Surface,
    font: pygame.font.Font,
    system: ParticleSystem,
    sprites: list[pygame.Surface],
    gravity_on: bool,
    paused: bool,
    field_active: bool,
    mouse_pos: tuple[int, int],
    fps: float,
    backend: str,
) -> None:
    screen.fill(BACKGROUND)
    pygame.draw.rect(screen, (30, 41, 59), ARENA, border_radius=24)
    pygame.draw.rect(screen, (125, 211, 252), ARENA, width=3, border_radius=24)

    count = system.count
    if count > 0:
        positions = system.positions_host()
        keys = sprite_keys(system.color_idx[:count], system.radius_host[:count])
        # Single C-side batched blit; sprites are pre-rendered so per-ball Python cost is minimal.
        offset = MAX_RADIUS + 1
        blit_seq = [
            (sprites[int(keys[i])], (int(positions[i, 0]) - offset, int(positions[i, 1]) - offset))
            for i in range(count)
        ]
        screen.blits(blit_seq, doreturn=False)

    if field_active and ARENA.collidepoint(mouse_pos):
        pygame.draw.circle(screen, FIELD_RING_COLOR, mouse_pos, int(FIELD_RADIUS), 1)
        pygame.draw.circle(screen, FIELD_RING_COLOR, mouse_pos, 8, 2)

    draw_hud(screen, font, system, gravity_on, paused, field_active, fps, backend)
    pygame.display.flip()


def main() -> None:
    args = parse_args()
    _bootstrap_sdl(args.headless)

    torch.set_grad_enabled(False)
    pygame.init()
    pygame.display.set_caption("Bouncing Ball Sandbox")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    system = ParticleSystem(MAX_BALLS, rng, device)
    system.reset(args.count)
    system.warmup()
    sprites = build_sprite_atlas()
    backend = f"pytorch {torch.__version__} {device.type}"

    gravity_on = not args.no_gravity
    paused = False
    running = True
    frames = 0
    frame_times: list[float] = []

    while running:
        frame_start = time.perf_counter()
        dt = min(clock.tick(FPS) / 1000.0, 0.025)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_g:
                    gravity_on = not gravity_on
                elif event.key == pygame.K_c:
                    system.clear()
                elif event.key == pygame.K_r:
                    system.reset(args.count)

        mouse_pos = pygame.mouse.get_pos()
        field_active = pygame.mouse.get_pressed()[0] and ARENA.collidepoint(mouse_pos)

        if not paused:
            system.step(dt, gravity_on, field_active, mouse_pos)

        draw_scene(
            screen,
            font,
            system,
            sprites,
            gravity_on,
            paused,
            field_active,
            mouse_pos,
            clock.get_fps(),
            backend,
        )

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
            f"backend={backend}"
        )


if __name__ == "__main__":
    main()
