#!/usr/bin/env python3
"""
plot_results.py

Reads results/benchmark.csv (produced by the C++ `benchmark` executable --
never hand-edited) and plots the speed-vs-accuracy tradeoff: Recall@10 on
the Y axis against Queries Per Second on the X axis, one point per nprobe
setting, each point labeled with its nprobe value.

This script is the ONLY place pandas/matplotlib are used. The vector
database itself (include/, src/) is pure C++17 standard library.

Usage:
    python3 plot_results.py
    python3 plot_results.py --csv results/benchmark.csv --out results/recall_vs_qps.png
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Recall@10 vs QPS from benchmark.csv")
    parser.add_argument("--csv", default="results/benchmark.csv", help="Path to benchmark CSV")
    parser.add_argument("--out", default="results/recall_vs_qps.png", help="Output PNG path")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found. Run the `benchmark` executable first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    required_cols = {"nprobe", "recall_at_10", "qps", "avg_latency_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: {args.csv} is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.sort_values("qps")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["qps"], df["recall_at_10"], marker="o", linestyle="-", color="#2b6cb0")

    for _, row in df.iterrows():
        ax.annotate(
            f"nprobe={int(row['nprobe'])}",
            (row["qps"], row["recall_at_10"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_xlabel("Queries Per Second (higher = faster)")
    ax.set_ylabel("Recall@10 (higher = more accurate)")
    ax.set_title("IVF Speed vs Accuracy Tradeoff (varying nprobe)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
