# 2D Turbulence Simulation (NumPy/CuPy) #

## Run

     uv run python numpy_dns_main.py

## Install CUDA (RTX 3090)

    nvidia-smi | head -n 3
    uv sync
    uv pip install cupy
    uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"


