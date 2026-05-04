# Games

This repository is a collection of small `pygame` games and interactive simulations.

## Included Projects

- `games/1-10000_balls`: a bouncing ball particle sandbox with mouse gravity and broad-phase collision culling.
- `games/2-stars`: a binary star orbital simulation with visible spin, trails, and a headless smoke-test mode.

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
```

See each game folder's `README.md` for controls and game-specific notes.
