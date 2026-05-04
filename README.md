# Games

This repository is a collection of small `pygame` games and interactive simulations.

## Structure

```text
.
|-- README.md
|-- requirements.txt
`-- games/
    |-- 1-10000_balls/
    |   |-- README.md
    |   `-- main.py
    |-- 2-stars/
    |   |-- README.md
    |   `-- main.py
    `-- 3-stars/
        |-- README.md
        `-- main.py
```

Each game folder is self-contained:

- `main.py`: executable game or simulation entrypoint
- `README.md`: controls, run commands, and game-specific notes

## Included Projects

- `games/1-10000_balls`: a `10000`-ball particle sandbox with PyTorch tensor physics, mouse gravity, and broad-phase collision culling.
- `games/2-stars`: a binary star orbital simulation with visible spin, trails, a reference grid toggle, time scaling, and a headless smoke-test mode.
- `games/3-stars`: a three-star orbital simulation with stable equilateral rotation, optional chaotic reset, trails, a reference grid toggle, time scaling, and a headless smoke-test mode.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Each game lives in its own folder and can be launched directly:

```bash
python3 games/1-10000_balls/main.py
python3 games/2-stars/main.py
python3 games/3-stars/main.py
```

The star simulations also support quick non-window smoke tests:

```bash
python3 games/2-stars/main.py --headless --frames 5
python3 games/3-stars/main.py --headless --frames 5
```

See each game folder's `README.md` for controls and game-specific notes.
