# cupyturbo — 2D Turbulence Simulation (NumPy / CuPy)

`cupyturbo` is a small playground DNS code for **2D homogeneous, incompressible turbulence**, structurally ported from a legacy Fortran/CUDA implementation.

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

The main entry point is `numpy_dns_main.py`.

![DNS Viewer Window](https://github.com/mannetroll/cupyturbo/blob/main/window.png?raw=true)


### Quick start (CPU)

    uv sync
    uv run python numpy_dns_main.py

This runs with default parameters (e.g. N=256, Re=10000, a default number of steps, CPU backend).

### Full CLI

    python numpy_dns_main.py N Re K0 STEPS CFL BACKEND

Where:

- N       — grid size (e.g. 256, 512)
- Re      — Reynolds number (e.g. 10000)
- K0      — peak wavenumber of the energy spectrum
- STEPS   — number of time steps
- CFL     — target CFL number (e.g. 0.75)
- BACKEND — "cpu", "gpu", or "auto"

Examples:

    # CPU run (NumPy)
    python numpy_dns_main.py 256 10000 10 1001 0.75 cpu

    # Auto-select backend (GPU if CuPy + CUDA are available)
    python numpy_dns_main.py 256 10000 10 1001 0.75 auto


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

       uv run python numpy_dns_main.py 256 10000 10 1001 0.75 gpu

Or let the backend auto-detect:

       uv run python numpy_dns_main.py 256 10000 10 1001 0.75 auto


## Profiling

### cProfile (CPU)

    python -m cProfile -o dns.prof \
        numpy_dns_simulator.py 256 10000 10 301 0.75 cpu

Inspect the results:

    python -m pstats dns.prof
    # inside pstats:
    sort time
    stats 20


### GUI profiling with SnakeViz

Install SnakeViz:

    uv pip install snakeviz

Visualize the profile:

    snakeviz dns.prof


### Memory & CPU profiling with Scalene (GUI)

Install Scalene:

    uv pip install scalene

Run with GUI report:

    scalene numpy_dns_simulator.py 256 10000 10 201 0.75 cpu


### Memory & CPU profiling with Scalene (CLI only)

For a terminal-only summary:

    scalene --cli --cpu numpy_dns_simulator.py -- 256 10000 10 201 0.75 cpu

(Note: the `--` separates Scalene’s own options from the script arguments.)


## Project layout (key modules)

- `numpy_dns_main.py`  
  CLI entry point; sets up the DNS state and runs the time loop.

- `numpy_dns_simulator.py`  
  Core DNS implementation:
  - PAO initialization (dns_pao_host_init)
  - FFT helpers (vfft_full_*)
  - STEP2A, STEP2B, STEP3
  - CFL-based time-step control (compute_cflm, next_dt)

- `numpy_dns_wrapper.py`  
  Thin wrapper for programmatic use.

- `gpu_test.py`  
  Simple CuPy test script to verify GPU availability and basic performance.


## License

Copyright © Mannetroll
See the project repository for license details.
