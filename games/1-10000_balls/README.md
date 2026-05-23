# Bouncing Ball Sandbox

A pygame particle sandbox tuned for a 10k-ball start, with PyTorch tensor
physics, uniform spatial-hash broad-phase collision, and a mouse gravity
field.

## Controls

- `Hold left mouse`: pull nearby balls toward the cursor
- `Space`: pause / resume
- `G`: toggle gravity
- `C`: clear all balls
- `R`: reset the layout
- `Esc`: quit

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 games/1-10000_balls/main.py
```

## CLI options

```text
--count N         number of balls at startup (default: 10000)
--no-gravity      start with gravity disabled
--device DEV      auto|cpu|cuda|mps  (default: auto)
--seed S          RNG seed (default: 7)
--headless        run with SDL's dummy video driver
--frames N        exit after N frames (smoke test / benchmark)
--benchmark       print frame-time stats (mean/p50/p95/max) on exit
```

Example: benchmark 30 headless frames on the GPU:

```bash
python3 games/1-10000_balls/main.py --headless --frames 30 --benchmark
```

## Notes for WSL

If the window does not open, make sure your WSL setup has GUI app support
enabled (`WSLg` on Windows 11, or an X server on older setups).

## Implementation notes

- Particle state lives in PyTorch tensors so the per-step compute graph runs
  on CUDA / MPS / CPU.
- Broad phase: particles are bucketed into a uniform spatial grid laid out
  CSR-style (sorted by cell, plus `starts` / `counts` offset tables). For
  each particle the full candidate set across its 9 neighbour cells is
  expanded with `torch.repeat_interleave`, so there is no per-cell capacity
  limit and dense piles are resolved correctly.
- Collision resolution batches overlaps and uses `index_add_` to accumulate
  position / velocity impulses.
- Rendering uses a small pre-rendered sprite atlas (one sprite per
  `(colour, radius)` combination) and a single `screen.blits` call, so
  per-frame Python overhead does not scale with ball count.
- Per-reset state (radius, colour) lives in CPU-side mirrors, so the only
  device → host copy each frame is the position tensor.
