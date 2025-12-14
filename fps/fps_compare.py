#!/usr/bin/env python3
import csv
import os
import math
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.canvas.manager.set_window_title("fps_compare")

    # Track overall N range for nice integer ticks on the lin-log plot
    all_n_min = None
    all_n_max = None

    for i, fn in enumerate(files):
        n, fps = load_nf(fn)
        lbl = label_from_filename(fn)
        m = markers[i % len(markers)]

        # Plot 1: log-log
        ax1.plot(n, fps, marker=m, label=lbl)

        # Plot 2: lin-log
        ax2.plot(n, fps, marker=m, label=lbl)

        all_n_min = min(n) if all_n_min is None else min(all_n_min, min(n))
        all_n_max = max(n) if all_n_max is None else max(all_n_max, max(n))

    # ---- Left subplot: log-log (x base 2) ----
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Grid size N (N = 2^K)")
    ax1.set_ylabel("Frames per second (FPS)")
    ax1.set_title("log-log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # ---- Right subplot: lin-log (x as integer values) ----
    ax2.set_yscale("log")
    ax2.set_xlabel("Grid size N (integer)")
    ax2.set_title("lin-log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    # Force integer ticks at powers of two across the plotted range (prevents 1e3 formatting)
    if all_n_min is not None and all_n_max is not None and all_n_min > 0:
        k_min = int(math.floor(math.log2(all_n_min)))
        k_max = int(math.ceil(math.log2(all_n_max)))
        xticks = [2 ** k for k in range(k_min, k_max + 1)]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([str(t) for t in xticks], rotation=45, ha="right")
        ax2.set_xlim(all_n_min, all_n_max)

    # Legend (once is enough)
    ax1.legend(loc="best")

    fig.suptitle("DNS FPS vs Grid Size (comparison)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PNG, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()