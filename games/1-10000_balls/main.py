import math

import numpy as np
import pygame
import torch


WIDTH = 1280
HEIGHT = 820
FPS = 60
BACKGROUND = (17, 24, 39)
ARENA = pygame.Rect(24, 72, WIDTH - 48, HEIGHT - 120)
ARENA_LEFT = ARENA.left
ARENA_RIGHT = ARENA.right
ARENA_TOP = ARENA.top
ARENA_BOTTOM = ARENA.bottom
START_BALLS = 10_000
MAX_BALLS = 12_000
MIN_RADIUS = 2
MAX_RADIUS = 4
CELL_SIZE = MAX_RADIUS * 2 + 1
GRID_WIDTH = (ARENA.width + CELL_SIZE - 1) // CELL_SIZE
GRID_HEIGHT = (ARENA.height + CELL_SIZE - 1) // CELL_SIZE
TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT
COLLISION_CELL_CAPACITY = 12
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


def choose_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_seed_positions(count: int, rng: np.random.Generator) -> np.ndarray:
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
    def __init__(self, max_particles: int, rng: np.random.Generator, device: torch.device):
        self.max_particles = max_particles
        self.rng = rng
        self.device = device
        self.count = 0
        self.overflowed_cells = 0
        self.dropped_collision_slots = 0
        self.pos = torch.zeros((max_particles, 2), dtype=torch.float32, device=device)
        self.vel = torch.zeros((max_particles, 2), dtype=torch.float32, device=device)
        self.radius = torch.zeros(max_particles, dtype=torch.float32, device=device)
        self.inv_mass = torch.zeros(max_particles, dtype=torch.float32, device=device)
        self.color_idx = np.zeros(max_particles, dtype=np.uint8)
        self.cell_particles = torch.empty(
            (TOTAL_CELLS, COLLISION_CELL_CAPACITY),
            dtype=torch.long,
            device=device,
        )
        self.neighbor_offsets = torch.tensor(
            [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ],
            dtype=torch.long,
            device=device,
        )

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def reset(self, count: int = START_BALLS) -> None:
        count = min(count, self.max_particles)
        positions = build_seed_positions(count, self.rng)
        radii = self.rng.integers(MIN_RADIUS, MAX_RADIUS + 1, size=count, dtype=np.int16).astype(np.float32)
        angles = self.rng.uniform(0.0, math.tau, size=count).astype(np.float32)
        speeds = self.rng.uniform(12.0, 68.0, size=count).astype(np.float32)
        velocity = np.column_stack(
            (
                np.cos(angles) * speeds,
                np.sin(angles) * speeds - 10.0,
            )
        ).astype(np.float32)

        self.count = count
        self.pos[:count] = self._tensor(positions)
        self.vel[:count] = self._tensor(velocity)
        self.radius[:count] = self._tensor(radii)
        self.inv_mass[:count] = 1.0 / torch.clamp(self.radius[:count] * self.radius[:count], min=1.0)
        self.color_idx[:count] = self.rng.integers(0, len(PALETTE), size=count, dtype=np.uint8)
        self.overflowed_cells = 0
        self.dropped_collision_slots = 0

    def clear(self) -> None:
        self.count = 0
        self.overflowed_cells = 0
        self.dropped_collision_slots = 0

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
        if not bool(mask.any()):
            return

        pull = (FIELD_STRENGTH * dt) / distance_sq[mask]
        velocity[mask] += delta[mask] * pull.unsqueeze(1)

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
        radii = self.radius[:count]
        cell_x = torch.div(positions[:, 0] - ARENA_LEFT, CELL_SIZE, rounding_mode="floor").to(torch.long)
        cell_y = torch.div(positions[:, 1] - ARENA_TOP, CELL_SIZE, rounding_mode="floor").to(torch.long)
        cell_x.clamp_(0, GRID_WIDTH - 1)
        cell_y.clamp_(0, GRID_HEIGHT - 1)
        cell_ids = cell_y * GRID_WIDTH + cell_x

        sorted_cells, order = torch.sort(cell_ids)
        counts = torch.bincount(sorted_cells, minlength=TOTAL_CELLS)
        starts = torch.cumsum(counts, dim=0) - counts
        sorted_slots = torch.arange(count, device=self.device, dtype=torch.long) - starts[sorted_cells]
        in_capacity = sorted_slots < COLLISION_CELL_CAPACITY

        overflow = counts - COLLISION_CELL_CAPACITY
        overflow = torch.clamp(overflow, min=0)
        self.overflowed_cells = int((overflow > 0).sum().item())
        self.dropped_collision_slots = int(overflow.sum().item())

        self.cell_particles.fill_(-1)
        kept_cells = sorted_cells[in_capacity]
        kept_slots = sorted_slots[in_capacity]
        kept_particles = order[in_capacity]
        self.cell_particles[kept_cells, kept_slots] = kept_particles

        neighbor_x = cell_x.unsqueeze(1) + self.neighbor_offsets[:, 0]
        neighbor_y = cell_y.unsqueeze(1) + self.neighbor_offsets[:, 1]
        valid_neighbor = (
            (neighbor_x >= 0)
            & (neighbor_x < GRID_WIDTH)
            & (neighbor_y >= 0)
            & (neighbor_y < GRID_HEIGHT)
        )
        neighbor_x = neighbor_x.clamp(0, GRID_WIDTH - 1)
        neighbor_y = neighbor_y.clamp(0, GRID_HEIGHT - 1)
        neighbor_cells = neighbor_y * GRID_WIDTH + neighbor_x

        candidates_j = self.cell_particles[neighbor_cells.reshape(-1)]
        candidates_j = candidates_j.reshape(count, 9, COLLISION_CELL_CAPACITY)
        candidates_i = torch.arange(count, device=self.device, dtype=torch.long).view(count, 1, 1)
        valid_pairs = valid_neighbor.unsqueeze(2) & (candidates_j >= 0) & (candidates_i < candidates_j)
        if not bool(valid_pairs.any()):
            return

        pair_i = candidates_i.expand_as(candidates_j)[valid_pairs]
        pair_j = candidates_j[valid_pairs]
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
        position_delta = torch.zeros_like(positions)
        correction = normal * overlap.unsqueeze(1)
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
        impulse = -1.96 * impact_speed / inverse_mass_sum
        impulse_vector = normal * impulse.unsqueeze(1)

        velocity_delta = torch.zeros_like(velocity)
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

        self.apply_mouse_field((int(self.pos[0, 0].item()), int(self.pos[0, 1].item())), 0.0)
        self.integrate(0.0, False)
        self.resolve_collisions()

    def render_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = self.count
        if count == 0:
            empty = np.empty(0, dtype=np.int32)
            return np.empty((0, 2), dtype=np.int32), empty, empty

        positions = self.pos[:count].detach().cpu().numpy().astype(np.int32, copy=False)
        radii = self.radius[:count].detach().cpu().numpy().astype(np.int32, copy=False)
        colors = self.color_idx[:count].astype(np.int32, copy=False)
        return positions, radii, colors


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
        f"balls: {system.count}   gravity: {'on' if gravity_on else 'off'}   field: {'active' if field_active else 'idle'}   paused: {'yes' if paused else 'no'}   fps: {fps:0.1f}   backend: {backend}",
        True,
        (191, 219, 254),
    )
    if system.dropped_collision_slots:
        collision_status = f"collision grid overflow: {system.dropped_collision_slots} balls across {system.overflowed_cells} cells"
    else:
        collision_status = "collision grid: stable"
    help_text = font.render(
        f"hold left mouse: drag   space: pause   g: gravity   c: clear   r: reset 10k   {collision_status}",
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

    positions, radii, color_idx = system.render_arrays()
    for index in range(system.count):
        pygame.draw.circle(
            screen,
            PALETTE[color_idx[index]].tolist(),
            (int(positions[index, 0]), int(positions[index, 1])),
            int(radii[index]),
        )

    if field_active and ARENA.collidepoint(mouse_pos):
        pygame.draw.circle(screen, FIELD_RING_COLOR, mouse_pos, int(FIELD_RADIUS), 1)
        pygame.draw.circle(screen, FIELD_RING_COLOR, mouse_pos, 8, 2)

    draw_hud(screen, font, system, gravity_on, paused, field_active, fps, backend)
    pygame.display.flip()


def main() -> None:
    torch.set_grad_enabled(False)
    pygame.init()
    pygame.display.set_caption("Bouncing Ball Sandbox")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    rng = np.random.default_rng(7)
    device = choose_torch_device()
    system = ParticleSystem(MAX_BALLS, rng, device)
    system.reset()
    system.warmup()
    backend = f"pytorch {torch.__version__} {device.type}"

    gravity_on = True
    paused = False
    running = True

    while running:
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
                    system.reset()

        mouse_pos = pygame.mouse.get_pos()
        field_active = pygame.mouse.get_pressed(3)[0] and ARENA.collidepoint(mouse_pos)

        if not paused:
            system.step(dt, gravity_on, field_active, mouse_pos)

        draw_scene(screen, font, system, gravity_on, paused, field_active, mouse_pos, clock.get_fps(), backend)

    pygame.quit()


if __name__ == "__main__":
    main()
