# Bouncing Ball Sandbox

Simple `pygame` particle sandbox tuned for a `10000`-ball start, with PyTorch tensor physics, broad-phase collision culling, and a mouse gravity field.

## Controls

- `Hold left mouse`: pull nearby balls toward the cursor
- `Space`: pause or resume
- `G`: toggle gravity
- `C`: clear all balls
- `R`: reset the `10000`-ball layout
- `Esc`: quit

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Notes for WSL

If the window does not open, make sure your WSL setup has GUI app support enabled (`WSLg` on Windows 11, or an X server on older setups).

## Performance Notes

- The simulation uses PyTorch tensors for particle state and vectorized physics updates.
- Ball-ball checks use a fixed-capacity uniform spatial grid, so each particle only tests neighbors in nearby cells instead of all `n^2` pairs.
- Collision resolution batches overlaps and accumulates position/velocity impulses with tensor `index_add_`.
- The game automatically uses CUDA or MPS when PyTorch exposes it, and falls back to CPU otherwise.
- Rendering uses a simple direct circle draw path, which is faster for these tiny particles than building and blitting `10000` sprite tuples each frame.
