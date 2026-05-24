#!/usr/bin/env python3
"""Phase 3 evaluation harness for sustained runtime pressure."""

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

from runtime.engine import SimulatedComputePacer
from runtime.memory_manager import MemoryManager
from runtime.scheduler import Scheduler
from runtime.page_allocator import PageAllocator
from runtime.tracer import RuntimeTracer
from runtime.workload import run_synthetic_workload, start_engine_background, stop_engine_background


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

    def forward(self, idx, page_allocator=None, slot_mapping=None, memory_manager=None, sequence_id=None, block_table=None, runtime_tracer=None):
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


def _detect_overload_onset(df: pd.DataFrame, min_queue: int = 50, window: int = 10) -> Optional[float]:
    if df.empty or "queue_growth_rate" not in df.columns or "queue_depth" not in df.columns:
        return None

    growth = df["queue_growth_rate"].rolling(window=window, min_periods=window).mean()
    active = (df["queue_depth"] > min_queue) & (growth > 0.0)
    idx = active.idxmax() if active.any() else None
    if idx is None or not bool(active.loc[idx]):
        return None

    t0 = df["timestamp"].iloc[0]
    return float(df["timestamp"].loc[idx] - t0)


def _detect_memory_saturation(df: pd.DataFrame, window: int = 5) -> Optional[float]:
    if df.empty or "free_kv_slots" not in df.columns:
        return None

    saturated = (df["free_kv_slots"] <= 0).rolling(window=window, min_periods=window).mean()
    idx = saturated[saturated >= 1.0].index.min() if not saturated.empty else None
    if idx is None:
        return None

    t0 = df["timestamp"].iloc[0]
    return float(df["timestamp"].loc[idx] - t0)


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
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=args.device,
        dtype=dtype,
    )
    memory_manager = MemoryManager(allocator)
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
        "overload_onset_s": _detect_overload_onset(df),
        "memory_saturation_s": _detect_memory_saturation(df),
    }

    return df, summary, starvation_times


def _plot_throughput_vs_latency(summary_df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile, group in summary_df.groupby("profile"):
        ax.scatter(
            group["throughput_toks_per_s"],
            group["p95_queue_wait_s"],
            label=profile,
        )
        for _, row in group.iterrows():
            ax.annotate(row["policy"], (row["throughput_toks_per_s"], row["p95_queue_wait_s"]))

    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_ylabel("P95 Queue Wait (s)")
    ax.set_title("Throughput vs Latency")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, "throughput_vs_latency.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_queue_timeline(traces: Dict[str, pd.DataFrame], out_dir: str, profile: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in traces.items():
        if df.empty:
            continue
        t0 = df["timestamp"].iloc[0]
        ax.plot(df["timestamp"] - t0, df["queue_depth"], label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Depth")
    ax.set_title(f"Queue Timeline ({profile})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, f"queue_timeline_{profile}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_kv_residency(traces: Dict[str, pd.DataFrame], out_dir: str, profile: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in traces.items():
        if df.empty or "avg_sequence_residency_lifetime" not in df.columns:
            continue
        t0 = df["timestamp"].iloc[0]
        ax.plot(df["timestamp"] - t0, df["avg_sequence_residency_lifetime"], label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Sequence Residency Lifetime (s)")
    ax.set_title(f"KV Residency Curve ({profile})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, f"kv_residency_{profile}.png")
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


def _write_report(
    summary_df: pd.DataFrame,
    out_dir: str,
    plot_paths: List[str],
) -> str:
    report_path = os.path.join(out_dir, "phase3_report.md")

    summary_table = summary_df[
        [
            "profile",
            "policy",
            "throughput_toks_per_s",
            "p95_queue_wait_s",
            "p95_starvation_s",
            "overload_onset_s",
            "memory_saturation_s",
        ]
    ].copy()

    summary_table = summary_table.round(4)

    lines = [
        "# Phase 3: Serving-Runtime Pressure Characterization",
        "",
        "## Overview",
        "This report summarizes policy behavior under sustained overload and starvation pressure.",
        "",
        "## Policy Comparison Summary",
        summary_table.to_markdown(index=False),
        "",
        "## Observations",
        "- Fairness tradeoffs: compare starvation tail vs throughput for each policy.",
        "- Queue poisoning: watch for early overload onset and persistent queue growth.",
        "- Static allocation failure: memory saturation points show when KV slots reach zero.",
        "",
        "## Figures",
    ]

    for path in plot_paths:
        rel_path = os.path.relpath(path, out_dir)
        if rel_path:
            lines.append(f"- {rel_path}")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 runtime pressure evaluation.")
    parser.add_argument("--out-dir", default="/home/earthy-zeus/MyProjects/runtime_benchmarks")
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
    args = parser.parse_args()

    if args.duration_s < 300.0:
        print("[INFO] duration_s raised to 300s for sustained pressure harness")
        args.duration_s = 300.0

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    traces_dir = os.path.join(args.out_dir, "traces")
    plots_dir = os.path.join(args.out_dir, "plots")

    _ensure_out_dir(args.out_dir)
    _ensure_out_dir(traces_dir)
    _ensure_out_dir(plots_dir)

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
            trace_path = os.path.join(traces_dir, f"trace_{profile}_{config.name}.csv")
            df.to_csv(trace_path, index=False)
            traces[config.name] = df
            starvation_samples[config.name] = starvation_times
            summaries.append(summary)

        plot_paths.append(_plot_queue_timeline(traces, plots_dir, profile))
        plot_paths.append(_plot_kv_residency(traces, plots_dir, profile))
        plot_paths.append(_plot_starvation_cdf(starvation_samples, plots_dir, profile))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "phase3_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    plot_paths.append(_plot_throughput_vs_latency(summary_df, plots_dir))
    report_path = _write_report(summary_df, args.out_dir, plot_paths)

    print(summary_path)
    print(report_path)
    for path in plot_paths:
        if path:
            print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
