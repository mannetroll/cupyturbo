#!/usr/bin/env python3
import csv
import os
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt


# Will pick up: fps_sweep_fortran.csv, fps_sweep_scipy.csv, fps_sweep_cuda.csv, etc.
CSV_GLOB = "fps_sweep_*.csv"
OUT_PNG = "fps_compare.png"


def label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    name = name.replace("fps_sweep_", "")
    return name


def load_nf(filename: str) -> Tuple[List[int], List[float]]:
    n_vals: List[int] = []
    fps_vals: List[float] = []

    with open(filename, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n_vals.append(int(row["N"]))
            fps_vals.append(float(row["FPS"]))

    # sort by N
    idx = sorted(range(len(n_vals)), key=lambda i: n_vals[i])
    n_vals = [n_vals[i] for i in idx]
    fps_vals = [fps_vals[i] for i in idx]
    return n_vals, fps_vals


def main() -> None:
    files = sorted(glob(CSV_GLOB))
    if not files:
        raise SystemExit(f"No CSV files found matching {CSV_GLOB}")

    markers = ["o", "s", "^", "D", "v", "x", "*", "+"]
    plt.figure(figsize=(9, 5))

    for i, fn in enumerate(files):
        n, fps = load_nf(fn)
        plt.plot(n, fps, marker=markers[i % len(markers)], label=label_from_filename(fn))

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Grid size N (N = 2^K)")
    plt.ylabel("Frames per second (FPS)")
    plt.title("DNS FPS vs Grid Size (comparison)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()