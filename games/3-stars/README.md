# Three Star Rotation

Three-star orbital simulation built with `pygame`. It starts in a stable equilateral rotation around the shared barycenter, with optional trails and a chaos reset that nudges the velocities into a less predictable three-body dance.

## Controls

- `Space`: pause or resume
- `R`: reset to the stable three-star rotation
- `C`: reset with a small chaotic velocity nudge
- `T`: toggle trails
- `G`: toggle reference grid
- `-` / `+`: adjust time scale
- `Esc`: quit

## Run

From this folder:

```bash
python3 main.py
```

From the repo root:

```bash
python3 games/3-stars/main.py
```

For a quick non-window smoke test:

```bash
python3 games/3-stars/main.py --headless --frames 5
```

## Implementation notes

Shared rendering helpers (background, glow, trails, main loop, hotkeys)
live in `games/_shared/star_common.py`. This file contains only the
three-star specifics: initial conditions, the N-body integrator, the
chaos-mode reset hotkey, and the equilateral-triangle overlay.

For larger N, see `games/4-nbody`, which ports the same gravitational
formulation to vectorised PyTorch tensors on the GPU.
