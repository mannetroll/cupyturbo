# 2D Turbulence Simulation (NumPy/CuPy) #

## Run

     uv run python numpy_dns_main.py

## Install CUDA (RTX 3090)

    nvidia-smi | head -n 3
    uv sync
    uv pip install cupy
    uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"

## Profile

    python -m cProfile -o dns.prof numpy_dns_simulator.py 256 10000 10 301 0.75 cpu
    python -m pstats dns.prof
    dns.prof% sort time
    dns.prof% stats 20

## GUI

    uv pip install snakeviz
    snakeviz dns.prof