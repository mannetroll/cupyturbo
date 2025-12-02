# gpu_test.py
import time
import cupy as cp

print("cp.show_config():")
cp.show_config()

# Big arrays on GPU
x = cp.ones((8000, 8000), dtype=cp.float32)
y = cp.ones((8000, 8000), dtype=cp.float32)

print("Starting heavy matmul on GPU...")
t0 = time.perf_counter()
z = x @ y
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()
print(f"Done matmul, elapsed: {t1 - t0:.3f} s")

print("Holding process for 60s so you can inspect nvidia-smi...")
time.sleep(60)