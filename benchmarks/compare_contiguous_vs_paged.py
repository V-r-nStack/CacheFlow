#!/usr/bin/env python3
"""Compare contiguous KV residency vs paged fragmented KV under identical workloads.

Uses the same runtime engine, scheduler, tracer, and workload profiles. Only
``--memory-backend`` semantics differ between runs (contiguous vs paged).

Outputs:
- CSV traces for each workload/backend pair
- Queue Depth vs Time plot
- Concurrent Residency vs Memory Churn plot
- Markdown report with summary tables and artifact paths
"""

from __future__ import annotations

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
import pandas as pd
import torch
import torch.nn as nn

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.engine import SimulatedComputePacer
from memory.memory_factory import build_memory_manager
from scheduler.scheduler import Scheduler
from tracing.tracer import RuntimeTracer
from workloads.workload import WORKLOAD_PROFILES, WorkloadProfile, run_synthetic_workload, start_engine_background, stop_engine_background


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    memory_backend: str


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, 8)
        self.proj = nn.Linear(8, vocab_size)

    def forward(
        self,
        idx,
        memory_backend=None,
        sequence_id=None,
        logical_length=None,
        runtime_tracer=None,
        kv_cache=None,
    ):
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


def _estimate_total_pages(
    device: torch.device,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    kv_cache_fraction: float,
    requested_total_pages: Optional[int],
) -> int:
    if requested_total_pages is not None:
        return int(requested_total_pages)

    page_bytes = 2 * num_layers * page_size * num_heads * head_dim * torch.tensor([], dtype=dtype).element_size()
    if device.type != "cuda" or not torch.cuda.is_available():
        return max(1024, int((512 * 1024 * 1024) // max(1, page_bytes)))

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes = torch.cuda.get_device_properties(device).total_memory

    usable_bytes = int(free_bytes * float(kv_cache_fraction))
    if usable_bytes <= 0:
        raise RuntimeError("Unable to size KV cache within the available device memory")

    return max(1, usable_bytes // int(page_bytes))


def _run_workload_once(
    profile_name: str,
    workload: WorkloadProfile,
    config: BenchmarkConfig,
    args: argparse.Namespace,
    total_pages: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)

    model = DummyModel(vocab_size=args.vocab_size, max_seq_len=args.max_seq_len).to(device)
    memory_manager = build_memory_manager(
        config.memory_backend,
        total_slots=total_pages * args.page_size,
        page_size=args.page_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        total_num_pages=total_pages,
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
        min_decode_tokens=args.min_decode_tokens,
        runtime_tracer=runtime_tracer,
        compute_pacer=compute_pacer,
        fallback_ttft=args.fallback_ttft,
    )

    async def _run_synthetic_load() -> None:
        await run_synthetic_workload(
            scheduler,
            duration_s=args.duration_s,
            base_rate=workload.base_rate,
            burst_rate=workload.burst_rate,
            burst_prob=workload.burst_prob,
            vocab_size=args.vocab_size,
            profile=profile_name,
            stop_event=stop_event,
        )

    start_time = time.time()
    try:
        asyncio.run(_run_synthetic_load())
        deadline = time.monotonic() + float(args.drain_s)
        while time.monotonic() < deadline:
            if not scheduler.waiting_queue and not scheduler.active_batch:
                break
            time.sleep(0.01)
    finally:
        stop_engine_background(stop_event, thread, timeout_s=args.stop_timeout_s)

    elapsed_s = max(1e-6, time.time() - start_time)
    df = runtime_tracer.to_dataframe()
    df = df.copy()
    if not df.empty:
        df["profile"] = profile_name
        df["memory_backend"] = config.memory_backend

    completed = list(scheduler.completed_sequences)
    total_tokens = sum(len(sequence.generated_token_ids) for sequence in completed)

    summary = {
        "profile": profile_name,
        "memory_backend": config.memory_backend,
        "throughput_toks_per_s": total_tokens / elapsed_s,
        "max_queue_depth": float(df["queue_depth"].max()) if not df.empty else 0.0,
        "p95_queue_depth": float(df["queue_depth"].quantile(0.95)) if not df.empty else 0.0,
        "avg_residency": float(df["active_batch_size"].mean()) if not df.empty else 0.0,
        "max_residency": float(df["active_batch_size"].max()) if not df.empty else 0.0,
        "avg_allocator_churn_rate": float(df["allocator_churn_rate"].mean()) if not df.empty else 0.0,
        "avg_page_gather_latency_s": float(df["page_gather_latency"].mean()) if not df.empty else 0.0,
        "final_internal_fragmentation_ratio": float(df["internal_fragmentation_ratio"].iloc[-1]) if not df.empty else 0.0,
        "final_page_pool_occupancy": float(df["page_pool_occupancy"].iloc[-1]) if not df.empty else 0.0,
    }

    return df, summary


def _plot_queue_depth_vs_time(
    traces: Dict[Tuple[str, str], pd.DataFrame],
    out_dir: str,
    profiles: List[str],
) -> str:
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 4.5 * len(profiles)), sharex=False)
    if len(profiles) == 1:
        axes = [axes]

    colors = {"contiguous": "tab:red", "paged": "tab:blue"}
    labels = {"contiguous": "Contiguous KV", "paged": "Paged fragmented KV"}

    for ax, profile_name in zip(axes, profiles):
        for backend in ("contiguous", "paged"):
            df = traces[(profile_name, backend)]
            if df.empty:
                continue
            t0 = df["timestamp"].iloc[0]
            ax.plot(
                df["timestamp"] - t0,
                df["queue_depth"],
                color=colors[backend],
                linewidth=1.8,
                label=labels[backend],
            )
        ax.set_title(f"Queue Depth vs Time - {profile_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Queue Depth")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Contiguous vs Paged KV: Queue Depth Under Adversarial Load", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(out_dir, "queue_depth_vs_time.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_residency_vs_memory_churn(
    traces: Dict[Tuple[str, str], pd.DataFrame],
    out_dir: str,
    profiles: List[str],
) -> str:
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 4.5 * len(profiles)), sharex=False)
    if len(profiles) == 1:
        axes = [axes]

    colors = {"contiguous": "tab:red", "paged": "tab:blue"}
    labels = {"contiguous": "Contiguous KV", "paged": "Paged fragmented KV"}

    for ax, profile_name in zip(axes, profiles):
        for backend in ("contiguous", "paged"):
            df = traces[(profile_name, backend)]
            if df.empty:
                continue
            ax.scatter(
                df["active_batch_size"],
                df["allocator_churn_rate"],
                s=14,
                alpha=0.6,
                color=colors[backend],
                label=labels[backend],
            )
        ax.set_title(f"Concurrent Residency vs Memory Churn - {profile_name}")
        ax.set_xlabel("Concurrent Residency")
        ax.set_ylabel("Memory Churn (pages/s or slots/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Contiguous vs Paged KV: Scheduler Pressure vs Memory Churn", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(out_dir, "residency_vs_memory_churn.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _write_report(
    summary_df: pd.DataFrame,
    out_dir: str,
    plot_paths: List[str],
) -> str:
    report_path = os.path.join(out_dir, "compare_contiguous_vs_paged_report.md")
    lines = [
        "# Contiguous vs Paged KV Comparative Benchmark",
        "",
        "## Summary",
        summary_df.round(4).to_markdown(index=False),
        "",
        "## Notes",
        "- Same runtime engine, scheduler, workload generator, and KV pool size.",
        "- Only `--memory-backend` differs: contiguous slot ownership vs paged fragmentation.",
        "",
        "## Artifacts",
    ]
    for path in plot_paths:
        lines.append(f"- {os.path.relpath(path, out_dir)}")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def _comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for profile_name in sorted(summary_df["profile"].unique()):
        contiguous = summary_df[
            (summary_df["profile"] == profile_name) & (summary_df["memory_backend"] == "contiguous")
        ].iloc[0]
        paged = summary_df[
            (summary_df["profile"] == profile_name) & (summary_df["memory_backend"] == "paged")
        ].iloc[0]
        rows.append(
            {
                "profile": profile_name,
                "contiguous_max_queue_depth": float(contiguous["max_queue_depth"]),
                "paged_max_queue_depth": float(paged["max_queue_depth"]),
                "queue_depth_ratio_contiguous_over_paged": float(contiguous["max_queue_depth"])
                / max(1e-6, float(paged["max_queue_depth"])),
                "contiguous_avg_churn": float(contiguous["avg_allocator_churn_rate"]),
                "paged_avg_churn": float(paged["avg_allocator_churn_rate"]),
                "contiguous_avg_residency": float(contiguous["avg_residency"]),
                "paged_avg_residency": float(paged["avg_residency"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare contiguous vs paged KV memory using the real runtime engine."
    )
    parser.add_argument("--out-dir", default="/home/earthy-zeus/MyProjects/contiguous_vs_paged_benchmark")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--drain-s", type=float, default=10.0)
    parser.add_argument("--stop-timeout-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--profiles",
        default="sustained_overload,starvation_pressure",
        help="Comma-separated workload profiles to compare.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--total-pages", type=int, default=None)
    parser.add_argument("--kv-cache-fraction", type=float, default=0.85)
    parser.add_argument("--min-decode-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pacer-base-delay-s", type=float, default=0.001)
    parser.add_argument("--pacer-per-sequence-delay-s", type=float, default=0.00005)
    parser.add_argument("--pacer-per-token-delay-s", type=float, default=0.0000002)
    parser.add_argument("--pacer-max-delay-s", type=float, default=0.02)
    parser.add_argument(
        "--fallback-ttft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Populate TTFT on completion if the exact first-token timestamp was not recorded.",
    )
    args = parser.parse_args()

    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    if not profiles:
        raise ValueError("At least one workload profile must be specified")

    _ensure_out_dir(args.out_dir)
    traces_dir = os.path.join(args.out_dir, "traces")
    plots_dir = os.path.join(args.out_dir, "plots")
    _ensure_out_dir(traces_dir)
    _ensure_out_dir(plots_dir)

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    page_bytes = 2 * args.num_layers * args.page_size * args.num_heads * args.head_dim * torch.tensor([], dtype=dtype).element_size()
    total_pages = _estimate_total_pages(
        device=device,
        page_size=args.page_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        kv_cache_fraction=args.kv_cache_fraction,
        requested_total_pages=args.total_pages,
    )

    print(
        "[INFO] compare benchmark using total_pages="
        f"{total_pages} page_size={args.page_size} approx_cache_bytes={total_pages * page_bytes}"
    )

    configs = [
        BenchmarkConfig(name="contiguous", memory_backend="contiguous"),
        BenchmarkConfig(name="paged", memory_backend="paged"),
    ]

    traces: Dict[Tuple[str, str], pd.DataFrame] = {}
    summary_rows: List[Dict[str, float]] = []

    for profile_name in profiles:
        workload = WORKLOAD_PROFILES.get(profile_name)
        if workload is None:
            raise ValueError(f"Unknown workload profile: {profile_name}")

        for config in configs:
            trace_df, summary = _run_workload_once(
                profile_name=profile_name,
                workload=workload,
                config=config,
                args=args,
                total_pages=total_pages,
            )
            traces[(profile_name, config.memory_backend)] = trace_df
            trace_path = os.path.join(traces_dir, f"trace_{profile_name}_{config.memory_backend}.csv")
            trace_df.to_csv(trace_path, index=False)
            summary["trace_path"] = trace_path
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "compare_contiguous_vs_paged_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    comparison_df = _comparison_table(summary_df)
    comparison_path = os.path.join(args.out_dir, "compare_contiguous_vs_paged_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    plot_paths = [
        _plot_queue_depth_vs_time(traces, plots_dir, profiles),
        _plot_residency_vs_memory_churn(traces, plots_dir, profiles),
    ]
    report_path = _write_report(summary_df, args.out_dir, plot_paths)

    print(summary_path)
    print(comparison_path)
    print(report_path)
    for path in plot_paths:
        print(path)

    print(summary_df.round(4).to_string(index=False))
    print(comparison_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
