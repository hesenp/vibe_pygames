# Games

A curated set of small `pygame` games and interactive physics simulations.
Physics-heavy demos lean on PyTorch for GPU-accelerated tensor math.

## Structure

```text
.
|-- README.md
|-- requirements.txt
|-- pyproject.toml          # ruff, mypy, pytest config
|-- launcher.py             # top-level menu (optional)
|-- games/
|   |-- _shared/
|   |   `-- star_common.py  # shared scaffolding for star-sim games
|   |-- 1-10000_balls/
|   |   |-- README.md
|   |   `-- main.py
|   |-- 2-stars/
|   |   |-- README.md
|   |   `-- main.py
|   |-- 3-stars/
|   |   |-- README.md
|   |   `-- main.py
|   `-- 4-nbody/
|       |-- README.md
|       `-- main.py
`-- tests/                  # smoke tests + physics unit tests
```

Each game folder is self-contained:

- `main.py`: executable game / simulation entrypoint
- `README.md`: controls, run commands, and game-specific notes

## Included Projects

- `games/1-10000_balls`: a 10k-ball particle sandbox with PyTorch tensor
  physics, broad-phase collision via uniform spatial hash, mouse gravity,
  and a pre-rendered sprite atlas for fast batched rendering.
- `games/2-stars`: a binary star orbital simulation with visible spin,
  trails, reference grid, time scaling, and headless smoke-test mode.
- `games/3-stars`: a three-star orbital simulation with stable equilateral
  rotation, optional chaotic reset (`c`), trails, reference grid, time
  scaling, and headless smoke-test mode.
- `games/4-nbody`: a GPU-accelerated N-body sandbox (direct O(N^2) gravity
  on PyTorch tensors). Two presets: solar system and Gaussian cluster.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Each game can be launched directly:

```bash
python3 games/1-10000_balls/main.py
python3 games/2-stars/main.py
python3 games/3-stars/main.py
python3 games/4-nbody/main.py
```

Or use the top-level launcher menu:

```bash
python3 launcher.py
```

Every game supports `--headless --frames N` for non-window smoke tests:

```bash
python3 games/1-10000_balls/main.py --headless --frames 5 --benchmark
python3 games/4-nbody/main.py --headless --frames 30 --benchmark --count 800
```

See each game folder's `README.md` for controls and game-specific notes.

## Development

```bash
pip install ruff mypy pytest

ruff check .
mypy games/1-10000_balls/main.py
mypy games/2-stars/main.py
mypy games/3-stars/main.py
mypy games/4-nbody/main.py
python3 -m pytest
```

CI runs the same checks on every push and pull request (see
`.github/workflows/ci.yml`).
