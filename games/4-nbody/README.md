# N-Body Sandbox

A GPU-accelerated N-body gravitational sandbox. Direct O(N^2) pairwise
gravity computed as PyTorch tensor ops — the whole compute graph runs on
CUDA / MPS / CPU. Up to ~1500 bodies stay interactive at 60 FPS on a modest
GPU.

## Controls

- `Space`: pause / resume
- `R`: reset with current settings
- `S`: switch to solar-system preset
- `C`: switch to Gaussian cluster preset
- `N`: add 50 bodies
- `M`: remove 50 bodies
- `Esc`: quit

## Run

```bash
python3 games/4-nbody/main.py
python3 games/4-nbody/main.py --count 800 --mode cluster
```

## CLI options

```text
--count N         initial body count (default: 400, max: 2000)
--mode M          solar | cluster  (default: solar)
--device DEV      auto|cpu|cuda|mps  (default: auto)
--seed S          RNG seed (default: 17)
--headless        run with SDL's dummy video driver
--frames N        exit after N frames (smoke test / benchmark)
--benchmark       print frame-time stats on exit
```

## Implementation notes

- Position, velocity, and mass live in `(N, *)` PyTorch tensors.
- The per-step pairwise displacement tensor has shape `(N, N, 2)`. For
  `N <= 1500` this fits comfortably in GPU memory and avoids the
  bookkeeping of a spatial hash.
- A semi-implicit (Euler) kick-then-drift integrator is used; energy
  conservation is good enough for visual purposes at the chosen timestep
  and softening.
- Rendering uses a small sprite atlas plus `screen.blits` so frame time
  scales with N only on the compute side.
