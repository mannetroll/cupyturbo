#!/usr/bin/env python3
import csv
import os
import math
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt


CSV_GLOB = "fps_sweep_*.csv"
OUT_PNG = "fps_compare.png"


def label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0].replace("fps_sweep_", "")
    key = name.lower()

    label_map = {
        "cuda":    "cuda (.cu) RTX 3090",
        "cupy":    "cupy (.py) RTX 3090",
        "fortran": "fortran (.f77) Apple M1 (OpenMP, 4 threads)",
        "numpy":   "numpy (.py) Apple M1 (single thread)",
        "scipy":   "scipy (.py) Apple M1 (4 workers)",
    }

    # exact match first, then fall back to substring match (for names like "cuda_fast", etc.)
    if key in label_map:
        return label_map[key]
    for k, v in label_map.items():
        if k in key:
            return v

    return name

def load_nf(filename: str) -> Tuple[List[int], List[float]]:
    n_vals: List[int] = []
    fps_vals: List[float] = []

    with open(filename, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n_vals.append(int(row["N"]))
            fps_vals.append(float(row["FPS"]))

    idx = sorted(range(len(n_vals)), key=lambda i: n_vals[i])
    n_vals = [n_vals[i] for i in idx]
    fps_vals = [fps_vals[i] for i in idx]
    return n_vals, fps_vals


def main() -> None:
    files = sorted(glob(CSV_GLOB))
    if not files:
        raise SystemExit(f"No CSV files found matching {CSV_GLOB}")

    markers = ["o", "s", "^", "D", "v", "x", "*", "+"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title("fps_compare")

    all_n_min = None
    all_n_max = None

    for i, fn in enumerate(files):
        n, fps = load_nf(fn)
        lbl = label_from_filename(fn)
        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]

        ax1.plot(n, fps, marker=m, linestyle=ls, label=lbl)  # log-log
        ax2.plot(n, fps, marker=m, linestyle=ls, label=lbl)  # log-lin

        all_n_min = min(n) if all_n_min is None else min(all_n_min, min(n))
        all_n_max = max(n) if all_n_max is None else max(all_n_max, max(n))

    # ---- Left subplot: log-log ----
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Grid size N (N = 2^K)")
    ax1.set_ylabel("Frames per second (FPS)")
    ax1.set_title("log-log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # ---- Right subplot: log-lin ----
    ax2.set_xscale("log", base=2)  # log x
    ax2.set_xlabel("Grid size N (integer)")
    ax2.set_ylabel("Frames per second (FPS)")
    ax2.set_ylim(0, 100)
    ax2.set_title("log-lin")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # Integer tick labels at powers of two (avoid 1eX formatting)
    if all_n_min is not None and all_n_max is not None and all_n_min > 0:
        k_min = int(math.floor(math.log2(all_n_min)))
        k_max = int(math.ceil(math.log2(all_n_max)))
        xticks = [2 ** k for k in range(k_min, k_max + 1)]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([str(t) for t in xticks], rotation=45, ha="right")
        ax2.set_xlim(512, 8192)

    fig.suptitle("DNS FPS vs Grid Size (comparison)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PNG, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()