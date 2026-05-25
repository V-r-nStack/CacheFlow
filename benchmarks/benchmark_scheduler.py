#!/usr/bin/env python3
"""Benchmark scheduler policies using the synthetic workload generator."""

import argparse
import asyncio
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.engine import SimulatedComputePacer, run_engine
from memory.memory_factory import build_memory_manager
from scheduler.scheduler import Scheduler
from runtime.sequence import Sequence
from tracing.tracer import RuntimeTracer
from workloads.workload import run_synthetic_workload, start_engine_background, stop_engine_background


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    policy: str
    preempt_waiting_threshold: Optional[int] = None
    preempt_long_context_tokens: Optional[int] = None


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, 8)
        self.proj = nn.Linear(8, vocab_size)

    def forward(self, idx, memory_backend=None, sequence_id=None, logical_length=None, runtime_tracer=None, kv_cache=None):
        x = self.embed(idx)
        return self.proj(x)


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _drain_scheduler(scheduler: Scheduler, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if not scheduler.waiting_queue and not scheduler.active_batch:
            break
        time.sleep(0.01)


def _run_one(
    config: SchedulerConfig,
    args: argparse.Namespace,
    profile: str,
) -> Tuple[pd.DataFrame, Dict[str, float], List[float]]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)

    model = DummyModel(vocab_size=args.vocab_size, max_seq_len=args.max_seq_len).to(device)
    page_size = 16
    total_slots = args.max_batch_size * args.max_seq_len
    memory_manager = build_memory_manager(
        args.memory_backend,
        total_slots=total_slots,
        page_size=page_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
    )
    scheduler = Scheduler(max_batch_size=args.max_batch_size)
    runtime_tracer = RuntimeTracer()
    stop_event = threading.Event()

    compute_pacer = SimulatedComputePacer(
        base_delay_s=args.pacer_base_delay_s,
        per_sequence_delay_s=args.pacer_per_sequence_delay_s,
        per_token_delay_s=args.pacer_per_token_delay_s,
        max_delay_s=args.pacer_max_delay_s,
    )

    thread = start_engine_background(
        model,
        scheduler,
        memory_manager,
        stop_event,
        eos_token_id=-1,
        max_seq_len=args.max_seq_len,
        top_k=args.top_k,
        policy=config.policy,
        min_decode_tokens=args.min_decode_tokens,
        preempt_waiting_threshold=config.preempt_waiting_threshold,
        preempt_long_context_tokens=config.preempt_long_context_tokens,
        runtime_tracer=runtime_tracer,
        compute_pacer=compute_pacer,
    )

    async def _run_workload() -> None:
        await run_synthetic_workload(
            scheduler,
            duration_s=args.duration_s,
            base_rate=args.base_rate,
            burst_rate=args.burst_rate,
            burst_prob=args.burst_prob,
            vocab_size=args.vocab_size,
            profile=profile,
            stop_event=stop_event,
        )

    start_time = time.time()
    try:
        asyncio.run(_run_workload())
        _drain_scheduler(scheduler, args.drain_s)
    finally:
        stop_engine_background(stop_event, thread, timeout_s=args.stop_timeout_s)

    end_time = time.time()
    elapsed_s = max(1e-6, end_time - start_time)

    df = runtime_tracer.to_dataframe()
    trace_path = os.path.join(args.out_dir, f"trace_{profile}_{config.name}.csv")
    runtime_tracer.dump_csv(trace_path)

    completed = list(scheduler.completed_sequences)
    total_tokens = sum(len(seq.generated_token_ids) for seq in completed)

    wait_times = [seq.total_wait_time for seq in completed if seq.total_wait_time is not None]
    starvation_times = [seq.starvation_duration for seq in completed if seq.starvation_duration is not None]

    summary = {
        "profile": profile,
        "policy": config.name,
        "throughput_toks_per_s": total_tokens / elapsed_s,
        "p95_itl_s": float(df["itl_s"].quantile(0.95)) if not df.empty else 0.0,
        "avg_queue_wait_s": float(sum(wait_times) / len(wait_times)) if wait_times else 0.0,
        "p95_queue_wait_s": float(pd.Series(wait_times).quantile(0.95)) if wait_times else 0.0,
        "avg_starvation_s": float(sum(starvation_times) / len(starvation_times)) if starvation_times else 0.0,
        "p95_starvation_s": float(pd.Series(starvation_times).quantile(0.95)) if starvation_times else 0.0,
        "avg_batch_utilization": float(df["avg_batch_utilization"].iloc[-1]) if not df.empty else 0.0,
        "p95_batch_utilization": float(df["p95_batch_utilization"].iloc[-1]) if not df.empty else 0.0,
        "trace_path": trace_path,
    }

    return df, summary, starvation_times


def _plot_summary(summary_df: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    summary_df.plot(
        kind="bar",
        x="policy",
        y="throughput_toks_per_s",
        ax=axes[0, 0],
        legend=False,
        title="Throughput (tokens/s)",
    )
    summary_df.plot(
        kind="bar",
        x="policy",
        y="p95_itl_s",
        ax=axes[0, 1],
        legend=False,
        title="P95 ITL (s)",
    )
    summary_df.plot(
        kind="bar",
        x="policy",
        y="p95_queue_wait_s",
        ax=axes[1, 0],
        legend=False,
        title="P95 Queue Wait (s)",
    )
    summary_df.plot(
        kind="bar",
        x="policy",
        y="p95_starvation_s",
        ax=axes[1, 1],
        legend=False,
        title="P95 Starvation (s)",
    )

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "scheduler_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_utilization(traces: Dict[str, pd.DataFrame], out_dir: str, profile: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in traces.items():
        if df.empty:
            continue
        t0 = df["timestamp"].iloc[0]
        ax.plot(df["timestamp"] - t0, df["batch_utilization_ratio"], label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Batch Utilization")
    ax.set_title("Batch Utilization Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, f"batch_utilization_{profile}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _align_time_series(
    traces: Dict[str, pd.DataFrame],
    column: str,
    samples: int = 300,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    series: Dict[str, pd.Series] = {}
    min_end = None
    for name, df in traces.items():
        if df.empty or column not in df.columns:
            continue
        time_rel = df["timestamp"] - df["timestamp"].iloc[0]
        series[name] = pd.Series(df[column].values, index=time_rel.values)
        end = float(time_rel.iloc[-1]) if not time_rel.empty else 0.0
        min_end = end if min_end is None else min(min_end, end)

    if min_end is None or min_end <= 0.0:
        return np.array([]), {}

    grid = np.linspace(0.0, min_end, samples)
    aligned: Dict[str, np.ndarray] = {}
    for name, s in series.items():
        aligned[name] = np.interp(grid, s.index.values, s.values)
    return grid, aligned


def _plot_queue_divergence(traces: Dict[str, pd.DataFrame], out_dir: str, profile: str) -> str:
    grid, aligned = _align_time_series(traces, "queue_depth")
    if grid.size == 0 or "fcfs" not in aligned:
        return ""

    baseline = aligned["fcfs"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, series in aligned.items():
        if name == "fcfs":
            continue
        ax.plot(grid, series - baseline, label=f"{name} - fcfs")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Depth Delta")
    ax.set_title(f"Queue Depth Divergence ({profile})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, f"queue_divergence_{profile}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_starvation_cdf(
    starvation_samples: Dict[str, List[float]],
    out_dir: str,
    profile: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, samples in starvation_samples.items():
        if not samples:
            continue
        data = np.sort(np.asarray(samples, dtype=float))
        y = np.arange(1, len(data) + 1) / float(len(data))
        ax.plot(data, y, label=name)

    ax.set_xlabel("Starvation Duration (s)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Starvation CDF ({profile})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, f"starvation_cdf_{profile}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark scheduler policies.")
    parser.add_argument("--out-dir", default="/tmp/scheduler_bench")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--drain-s", type=float, default=1.0)
    parser.add_argument("--stop-timeout-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--profiles",
        default="sustained_overload,starvation_pressure",
        help="Comma-separated workload profiles to run.",
    )
    parser.add_argument("--base-rate", type=float, default=20.0)
    parser.add_argument("--burst-rate", type=float, default=80.0)
    parser.add_argument("--burst-prob", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--min-decode-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--preempt-waiting-threshold", type=int, default=8)
    parser.add_argument("--preempt-long-context-tokens", type=int, default=128)
    parser.add_argument("--pacer-base-delay-s", type=float, default=0.001)
    parser.add_argument("--pacer-per-sequence-delay-s", type=float, default=0.00005)
    parser.add_argument("--pacer-per-token-delay-s", type=float, default=0.0000002)
    parser.add_argument("--pacer-max-delay-s", type=float, default=0.02)
    parser.add_argument(
        "--memory-backend",
        choices=["contiguous", "paged"],
        default="paged",
        help="Select the KV memory architecture to benchmark.",
    )
    args = parser.parse_args()

    if args.duration_s < 300.0:
        print("[INFO] duration_s raised to 300s for sustained pressure harness")
        args.duration_s = 300.0

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    _ensure_out_dir(args.out_dir)

    configs = [
        SchedulerConfig(name="fcfs", policy="fcfs"),
        SchedulerConfig(name="shortest_prompt", policy="shortest_prompt_first"),
        SchedulerConfig(
            name="fairness_preemptive",
            policy="fcfs",
            preempt_waiting_threshold=args.preempt_waiting_threshold,
            preempt_long_context_tokens=args.preempt_long_context_tokens,
        ),
    ]

    summaries: List[Dict[str, float]] = []
    plot_paths: List[str] = []

    for profile in profiles:
        traces: Dict[str, pd.DataFrame] = {}
        starvation_samples: Dict[str, List[float]] = {}

        for config in configs:
            df, summary, starvation_times = _run_one(config, args, profile)
            traces[config.name] = df
            starvation_samples[config.name] = starvation_times
            summaries.append(summary)

        plot_paths.append(_plot_queue_divergence(traces, args.out_dir, profile))
        plot_paths.append(_plot_starvation_cdf(starvation_samples, args.out_dir, profile))
        plot_paths.append(_plot_utilization(traces, args.out_dir, profile))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "scheduler_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(summary_path)
    if not summary_df.empty:
        print(_plot_summary(summary_df, args.out_dir))
    for path in plot_paths:
        if path:
            print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
