# 2D Turbulence Simulation (NumPy/CuPy) #

## Run

     uv run python numpy_dns_main.py

## Install CUDA (RTX 3090)

    nvidia-smi | head -n 3
    uv sync
    uv pip install cupy
    uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"

## cProfile

    python -m cProfile -o dns.prof numpy_dns_simulator.py 256 10000 10 301 0.75 cpu
    python -m pstats dns.prof
    dns.prof% sort time
    dns.prof% stats 20

## GUI snakeviz

    uv pip install snakeviz
    snakeviz dns.prof


## GUI Memory & CPU scalene

    uv pip install scalene
    scalene numpy_dns_simulator.py 256 10000 10 201 0.75 cpu

## CLI Memory & CPU scalene

    scalene --cli --cpu numpy_dns_simulator.py --- 256 10000 10 201 0.75 cpu

## Win11 GPU

    PS C:\Windows\system32> tasklist | findstr python
    PS C:\Windows\system32> nvidia-smi -l 1
