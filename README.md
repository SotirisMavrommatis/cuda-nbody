# CUDA N-Body: Galaxy Collision

A gravitational N-body simulation written in CUDA C++. Every body attracts every
other body, so one simulation step costs **N² interactions**. That is a workload a GPU
handles very well, and this project measures how well.

![Two galaxies colliding, 8192 bodies](galaxy_collision.gif)

*Two rotating disk galaxies (8,192 bodies) pass through each other. Tidal forces pull out
streams of stars. The full 1,500-step simulation runs in about 0.3 s on a laptop GPU.*

## What's inside

| Version | Idea |
|---|---|
| **CPU** | Plain double loop, single thread. Used as the reference result. |
| **GPU naive** | One thread per body. Each thread reads all N bodies from global memory. |
| **GPU tiled** | Each block loads 256 bodies at a time into **shared memory**, and all 256 threads reuse that tile. Global memory traffic drops by about 256×. |

The program checks itself before it benchmarks anything. It runs the same simulation on
the CPU and on both GPU kernels and compares the positions. Both kernels match the CPU
to within `2.4e-07`.

## Results

NVIDIA GeForce RTX 5070 Laptop GPU (36 SMs), CUDA 13.5, Windows 11. Times are per
simulation step.

| N | CPU (1 thread) | GPU naive | GPU tiled | Tiled GFLOPS* | Tiled vs CPU |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2.88 ms | 0.058 ms | 0.029 ms | 728 | 100× |
| 4,096 | 47.2 ms | 0.224 ms | 0.091 ms | 3,694 | 520× |
| 16,384 | 695 ms | 0.911 ms | 0.602 ms | 8,917 | 1,154× |
| 65,536 | *(too slow)* | 13.3 ms | 10.6 ms | 8,087 | – |

\* Counted as 20 FLOPs per interaction, the usual convention for N-body benchmarks.

**What the numbers show**
- **GPU speedup grows with N.** At small N the GPU is underused: 1,024 threads is far fewer
  than the GPU can run at once. From N ≈ 16K the GPU is saturated at roughly 8–9 TFLOPS.
- **Shared memory helps most at mid-size N** (2.5× at N = 4096). At larger N the GPU's
  L1/L2 caches already absorb much of the reuse in the naive version, so the gap shrinks
  to about 1.25×.
- The CPU baseline is a single unoptimised thread. A multithreaded, SIMD CPU version would
  narrow the gap considerably, but it would still be much slower than the GPU.

## Physics

- Units with `G = 1`. Softening `ε² = 0.0025` keeps forces finite when two bodies get very
  close.
- Semi-implicit (symplectic) Euler integration, `dt = 0.01`.
- Each galaxy is one heavy central mass plus a thin disk of stars on circular orbits. The
  two galaxies spin in opposite directions and approach on an off-centre path.

## Build and run

Requirements: an NVIDIA GPU, the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads),
and, on Windows, Visual Studio with the *Desktop development with C++* workload.
Rendering the GIF needs Python with `numpy`, `matplotlib` and `pillow`.

**Windows (easy way):** double-click `build_and_run.bat`. It compiles, verifies,
benchmarks, simulates and renders the GIF, and saves all output to `run_log.txt`.

**By hand:**
```bash
nvcc -O3 -o nbody nbody.cu
./nbody                                   # verify + benchmark + write frames.bin
python render.py frames.bin galaxy_collision.gif
```

## Files

| File | Purpose |
|---|---|
| `nbody.cu` | Simulation, CUDA kernels, verification and benchmark |
| `render.py` | Turns `frames.bin` into `galaxy_collision.gif` and a still `.png` |
| `build_and_run.bat` | One-click build and run on Windows |

## Ideas for next steps

- Real-time OpenGL rendering through CUDA–OpenGL interop, instead of writing frames to disk
- Leapfrog or Runge-Kutta integration, plus an energy-conservation check
- Barnes–Hut tree, O(N log N), for millions of bodies
- Multithreaded CPU baseline (OpenMP) for a fairer comparison
