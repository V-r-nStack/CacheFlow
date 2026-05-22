#!/usr/bin/env python3
"""Stress test scheduler behavior under extreme KV-cache pressure."""

import argparse
import asyncio
import csv
import os
import threading
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.engine import run_engine
from runtime.memory_manager import MemoryManager
from runtime.scheduler import Scheduler
from runtime.sequence import Sequence
from runtime.static_kv_cache import StaticKVCache
from runtime.tracer import RuntimeTracer
from runtime.workload import run_synthetic_workload, start_engine_background, stop_engine_background


@dataclass
class CollapseThresholds:
    min_throughput_tps: float
    queue_explosion_depth: int
    itl_inflation_factor: float


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, 8)
        self.proj = nn.Linear(8, vocab_size)

    def forward(self, idx, static_kv_cache=None, slot_mapping=None, memory_manager=None, sequence_id=None):
        x = self.embed(idx)
        return self.proj(x)


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _count_total_generated(scheduler: Scheduler) -> int:
    total = 0
    for sequence in scheduler.active_batch + scheduler.completed_sequences:
        total += len(sequence.generated_token_ids)
    return total


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress scheduler under constrained KV slots.")
    parser.add_argument("--out-dir", default="/tmp/scheduler_memory_stress")
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--drain-s", type=float, default=1.0)
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--gpu-preset", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--base-rate", type=float, default=150.0)
    parser.add_argument("--burst-rate", type=float, default=400.0)
    parser.add_argument("--burst-prob", type=float, default=0.7)
    parser.add_argument("--min-decode-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-len", type=int, default=384)
    parser.add_argument("--collapse-tps", type=float, default=100.0)
    parser.add_argument("--queue-explosion", type=int, default=200)
    parser.add_argument("--itl-inflation", type=float, default=4.0)
    args = parser.parse_args()

    if args.gpu_preset:
        args.device = "cuda"
        args.dtype = "float16"
        args.max_batch_size = 128
        args.max_seq_len = 2048
        args.num_layers = 8
        args.num_heads = 8
        args.head_dim = 64
        args.base_rate = 300.0
        args.burst_rate = 800.0
        args.burst_prob = 0.7
        args.min_decode_tokens = 256
        args.long_prompt_len = 1536

    _ensure_out_dir(args.out_dir)

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)

    model = DummyModel(vocab_size=args.vocab_size, max_seq_len=args.max_seq_len).to(device)
    cache = StaticKVCache(
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=args.device,
        dtype=dtype,
    )
    memory_manager = MemoryManager(cache)
    scheduler = Scheduler(max_batch_size=args.max_batch_size)
    runtime_tracer = RuntimeTracer()
    stop_event = threading.Event()

    thread = start_engine_background(
        model,
        scheduler,
        memory_manager,
        stop_event,
        eos_token_id=-1,
        max_seq_len=args.max_seq_len,
        top_k=10,
        min_decode_tokens=args.min_decode_tokens,
        runtime_tracer=runtime_tracer,
    )

    async def _run_workload() -> None:
        await run_synthetic_workload(
            scheduler,
            duration_s=args.duration_s,
            base_rate=args.base_rate,
            burst_rate=args.burst_rate,
            burst_prob=args.burst_prob,
            vocab_size=args.vocab_size,
            prompt_lengths=[args.long_prompt_len],
            prompt_weights=[1.0],
            stop_event=stop_event,
        )

    thresholds = CollapseThresholds(
        min_throughput_tps=args.collapse_tps,
        queue_explosion_depth=args.queue_explosion,
        itl_inflation_factor=args.itl_inflation,
    )

    collapse_logged = False
    explosion_logged = False
    itl_logged = False
    admission_fail_samples = 0
    total_samples = 0

    monitor_rows: List[List[float]] = []
    last_sample_time = time.time()
    last_generated = _count_total_generated(scheduler)
    baseline_itl = None

    try:
        asyncio.run(_run_workload())
        end_time = time.time() + args.drain_s

        while time.time() < end_time:
            time.sleep(max(args.sample_ms / 1000.0, 0.001))
            now = time.time()
            queue_depth = len(scheduler.waiting_queue)
            active_batch = len(scheduler.active_batch)
            free_slots = memory_manager.free_slots_count()
            total_generated = _count_total_generated(scheduler)

            dt = max(1e-6, now - last_sample_time)
            d_tokens = total_generated - last_generated
            throughput = d_tokens / dt

            itl = 0.0
            if runtime_tracer._rows:
                itl = float(runtime_tracer._rows[-1].get("itl_s", 0.0))

            if baseline_itl is None and itl > 0.0:
                baseline_itl = itl

            min_required_slots = args.long_prompt_len + args.min_decode_tokens
            admission_failed = int(queue_depth > 0 and free_slots < min_required_slots)
            admission_fail_samples += admission_failed
            total_samples += 1

            if not collapse_logged and throughput < thresholds.min_throughput_tps:
                print(
                    "[COLLAPSE] throughput="
                    f"{throughput:.2f} t/s queue={queue_depth} active={active_batch}"
                )
                collapse_logged = True

            if not explosion_logged and queue_depth >= thresholds.queue_explosion_depth:
                print(
                    "[QUEUE EXPLOSION] queue_depth="
                    f"{queue_depth} active={active_batch}"
                )
                explosion_logged = True

            if (
                not itl_logged
                and baseline_itl is not None
                and itl > baseline_itl * thresholds.itl_inflation_factor
            ):
                print(
                    "[ITL INFLATION] itl="
                    f"{itl:.6f}s baseline={baseline_itl:.6f}s"
                )
                itl_logged = True

            monitor_rows.append(
                [
                    now,
                    queue_depth,
                    active_batch,
                    free_slots,
                    throughput,
                    itl,
                    admission_failed,
                ]
            )

            last_sample_time = now
            last_generated = total_generated
    finally:
        stop_engine_background(stop_event, thread, timeout_s=2.0)

    admission_fail_rate = (
        admission_fail_samples / float(total_samples) if total_samples > 0 else 0.0
    )
    print(f"[SUMMARY] admission_fail_rate={admission_fail_rate:.3f}")

    monitor_path = os.path.join(args.out_dir, "stress_monitor.csv")
    with open(monitor_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "queue_depth",
                "active_batch_size",
                "free_slots",
                "throughput_tps",
                "itl_s",
                "admission_failed",
            ]
        )
        writer.writerows(monitor_rows)

    trace_path = os.path.join(args.out_dir, "trace_stress.csv")
    runtime_tracer.dump_csv(trace_path)

    print(monitor_path)
    print(trace_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
