#!/usr/bin/env python3
"""Plot RuntimeTracer CSVs into summary figures."""

import argparse
import glob
import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def _load_csvs(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {pattern}")

    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_queue_vs_active(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["queue_depth"], label="Queue Depth")
    ax.plot(df["timestamp"], df["active_batch_size"], label="Active Batch Size")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")
    ax.set_title("Queue Depth vs Active Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "queue_vs_active.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_concurrency_vs_itl(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["active_batch_size"], df["itl_s"], alpha=0.6, s=12)
    ax.set_xlabel("Active Batch Size")
    ax.set_ylabel("ITL (s)")
    ax.set_title("Concurrency vs ITL")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "concurrency_vs_itl.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_memory_pressure(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "allocated_kv_slots" in df.columns and "free_kv_slots" in df.columns:
        total_capacity = df["allocated_kv_slots"] + df["free_kv_slots"]
        ax.plot(df["timestamp"], df["allocated_kv_slots"], label="Allocated KV Slots")
        ax.plot(df["timestamp"], total_capacity, label="Total KV Capacity", linestyle="--")
    else:
        total_capacity = df["active_batch_size"] + df["queue_depth"]
        ax.plot(df["timestamp"], df["active_batch_size"], label="Active Batch Size")
        ax.plot(df["timestamp"], total_capacity, label="Queue + Active", linestyle="--")
        ax.set_title("Memory Pressure Proxy Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("KV Slots")
    if "allocated_kv_slots" in df.columns:
        ax.set_title("KV Memory Pressure Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "memory_pressure.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RuntimeTracer CSV metrics.")
    parser.add_argument(
        "--input",
        default="/tmp/engine_metrics_test.csv",
        help="CSV path or glob pattern (default: /tmp/engine_metrics_test.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/runtime_traces",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    df = _load_csvs(args.input)
    _ensure_output_dir(args.out_dir)

    outputs = [
        plot_queue_vs_active(df, args.out_dir),
        plot_concurrency_vs_itl(df, args.out_dir),
        plot_memory_pressure(df, args.out_dir),
    ]

    for path in outputs:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
