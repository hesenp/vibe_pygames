# Binary Star Rotation

Two-star orbital simulation built with `pygame`. The stars orbit their shared barycenter, spin visibly, and leave optional trails so the rotation is easy to see.

## Controls

- `Space`: pause or resume
- `R`: reset the orbit
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
python3 games/2-stars/main.py
```

For a quick non-window smoke test:

```bash
python3 games/2-stars/main.py --headless --frames 5
```

## Implementation notes

Shared rendering helpers (background, glow, trails, main loop, hotkeys)
live in `games/_shared/star_common.py` and are reused by the 3-stars
simulation as well. This file contains only the binary-system specific
bits: initial conditions, the two-body integrator, and the orbit-ring
overlay.
