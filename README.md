# cupyturbo — 2D Turbulence Simulation (NumPy / CuPy)

`cupyturbo` is a small playground DNS code for **2D homogeneous, incompressible turbulence**, structurally ported from a legacy FORTRAN 77 implementation.

It supports:

- **NumPy** for CPU runs
- **CuPy** (optional) for GPU acceleration on CUDA devices (e.g. RTX 3090)

The solver mirrors the original structure:

- PAO-style random-field initialization
- 3/2 de-aliasing in spectral space
- Crank–Nicolson time integration
- Time loop: STEP2B → STEP3 → STEP2A → NEXTDT


## Installation

### Using uv (recommended)

From the project root:

    uv sync

This creates a virtual environment and installs the project and its dependencies from `pyproject.toml`.

### Using plain pip

From a cloned repo:

    python -m venv .venv
    source .venv/bin/activate      # on Windows: .venv\Scripts\activate
    pip install -e .

(Once published on PyPI, you’ll be able to do `pip install cupyturbo` directly.)


## Running the DNS

![DNS Viewer Window](https://raw.githubusercontent.com/mannetroll/cupyturbo/main/window.png)

### Quick start (CPU)

    uv sync
    uv run python -m cupyturbo.dns_main

This runs with default parameters (e.g. N=256, Re=10000, a default number of steps, CPU backend).

### Full CLI

    python -m cupyturbo.dns_simulator N Re K0 STEPS CFL BACKEND

Where:

- N       — grid size (e.g. 256, 512)
- Re      — Reynolds number (e.g. 10000)
- K0      — peak wavenumber of the energy spectrum
- STEPS   — number of time steps
- CFL     — target CFL number (e.g. 0.75)
- BACKEND — "cpu", "gpu", or "auto"

Examples:

    # CPU run (NumPy)
    python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 cpu

    # Auto-select backend (GPU if CuPy + CUDA are available)
    python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 auto


## Enabling GPU with CuPy (CUDA)

On a CUDA machine (e.g. RTX 3090):

1. Check that the driver/CUDA are available:

       nvidia-smi | head -n 3

2. Install CuPy into the uv environment:

       uv sync
       uv pip install cupy

3. Verify that CuPy sees the GPU:

       uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"

4. Run in GPU mode:

       uv run python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 gpu

Or let the backend auto-detect:

       uv run python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 auto


## Profiling

### cProfile (CPU)

    python -m cProfile -o dns_simulator.prof -m cupyturbo.dns_simulator    

Inspect the results:

    python -m pstats dns_simulator.prof
    # inside pstats:
    sort time
    stats 20


### GUI profiling with SnakeViz

Install SnakeViz:

    uv pip install snakeviz

Visualize the profile:

    snakeviz dns_simulator.prof


### Memory & CPU profiling with Scalene (GUI)

Install Scalene:

    uv pip install scalene

Run with GUI report:

    scalene -m cupyturbo.dns_simulator 256 10000 10 201 0.75 cpu


### Memory & CPU profiling with Scalene (CLI only)

For a terminal-only summary:

    scalene --cli --cpu -m cupyturbo.dns_simulator 256 10000 10 201 0.75 cpu


## Project layout (key modules)

- `cupyturbo/dns_main.py`  
  PyQt6 GUI viewer; displays DNS fields (U, V, ω, kinetic) in real time.

- `cupyturbo/dns_simulator.py`  
  Headless CLI DNS solver:
  - PAO initialization (dns_pao_host_init)
  - FFT helpers (vfft_full_*)
  - STEP2A, STEP2B, STEP3
  - CFL-based time-step control (compute_cflm, next_dt)

- `cupyturbo/dns_wrapper.py`  
  Thin wrapper for programmatic use.

- `cupyturbo/gpu_test.py`  
  Simple CuPy test script to verify GPU availability and basic performance.

## one-liner

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv cache clean mannetroll-cupyturbo
uv run --with mannetroll-cupyturbo==0.1.0 python -m cupyturbo.dns_main
```

## License

Copyright © Mannetroll
See the project repository for license details.
