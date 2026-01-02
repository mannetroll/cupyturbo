# cupyturbo — 2D Turbulence Simulation (NumPy / CuPy)

Direct Numerical Simulation (DNS) code for 
**2D Homogeneous, Incompressible Turbulence**

It supports:

- **NumPy** for CPU runs
- **CuPy** (optional) for GPU acceleration on CUDA devices (e.g. RTX 3090)

The solver mirrors the original structure:

- PAO-style random-field initialization
- 3/2 de-aliasing in spectral space
- Crank–Nicolson time integration


## Installation

### Using uv (recommended)

From the project root:

    $ uv sync
    $ uv run python -m cupyturbo.dns_main

This creates a virtual environment and installs the project and its dependencies from `pyproject.toml`.

### Using plain pip

From a cloned repo:

    $ python -m venv .venv
    $ source .venv/bin/activate      # on Windows: .venv\Scripts\activate
    $ pip install -e .
    $ python -m cupyturbo.dns_main

## The DNS with NumPy (256 x 256)

![DNS NumPy](https://raw.githubusercontent.com/mannetroll/cupyturbo/main/window.png)


### Full CLI

    $ python -m cupyturbo.dns_simulator N Re K0 STEPS CFL BACKEND

Where:

- N       — grid size (e.g. 256, 512)
- Re      — Reynolds number (e.g. 10000)
- K0      — peak wavenumber of the energy spectrum
- STEPS   — number of time steps
- CFL     — target CFL number (e.g. 0.75)
- BACKEND — "cpu", "gpu", or "auto"

Examples:

    # CPU run (NumPy)
    $ python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 cpu

    # Auto-select backend (GPU if CuPy + CUDA are available)
    $ python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 auto


## Enabling GPU with CuPy (CUDA 13.1)

On a CUDA machine (e.g. RTX 3090):

1. Check that the driver/CUDA are available:

       $ nvidia-smi

2. Install CuPy into the uv environment:

       $ uv sync --extra cuda
       $ uv run -- cupyturbo

3. Verify that CuPy sees the GPU:

       $ uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"

4. Run in GPU mode:

       $ uv run python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 gpu

Or let the backend auto-detect:

       $ uv run python -m cupyturbo.dns_simulator 256 10000 10 1001 0.75 auto


## The DNS with CuPy (4096 x 4096)

![DNS CuPy](https://raw.githubusercontent.com/mannetroll/cupyturbo/main/window4096.png)


## Profiling

### cProfile (CPU)

    $ python -m cProfile -o dns_simulator.prof -m cupyturbo.dns_simulator    

Inspect the results:

    $ python -m pstats dns_simulator.prof
    # inside pstats:
    dns_simulator.prof% sort time
    dns_simulator.prof% stats 20


### GUI profiling with SnakeViz

Install SnakeViz:

    $ uv pip install snakeviz

Visualize the profile:

    $ snakeviz dns_simulator.prof


### Memory & CPU profiling with Scalene (GUI)

Install Scalene:

    $ uv pip install "scalene==1.5.55"

Run with GUI report:

    $ scalene -m cupyturbo.dns_simulator 256 10000 10 201 0.75 cpu


### Memory & CPU profiling with Scalene (CLI only)

For a terminal-only summary:

    $ scalene --cli --cpu -m cupyturbo.dns_simulator 256 10000 10 201 0.75 cpu


## one-liner

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv cache clean mannetroll-cupyturbo
$ uv run --python 3.13 --with mannetroll-cupyturbo==0.1.5 python -m cupyturbo.dns_main
```

## License

Copyright © Mannetroll
See the project repository for license details.

