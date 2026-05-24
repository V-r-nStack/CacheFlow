import asyncio
import csv
import os
import sys
import threading
import math
import time
from typing import Callable, List

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.attention import CausalMultiHeadAttention
from runtime.batching import prepare_continuous_batch
from runtime.engine import run_engine
from runtime.memory_manager import MemoryManager
from runtime.scheduler import Scheduler
from runtime.sequence import Sequence, SequenceStatus
from runtime.page_allocator import PageAllocator
from runtime.tracer import RuntimeTracer
from runtime.workload import (
    WORKLOAD_PROFILES,
    WorkloadProfile,
    run_synthetic_workload,
    start_engine_background,
    stop_engine_background,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _run_test(name: str, func: Callable[[], None]) -> None:
    start = time.perf_counter()
    try:
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log(f"[PASS] {name} | {elapsed_ms:.2f} ms")
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log(f"[FAIL] {name} | {elapsed_ms:.2f} ms | {exc}")
        raise


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 64, max_seq_len: int = 1024):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, 8)
        self.proj = nn.Linear(8, vocab_size)

    def forward(self, idx, page_allocator=None, slot_mapping=None, memory_manager=None, sequence_id=None):
        x = self.embed(idx)
        return self.proj(x)


def test_memory_manager_mapping() -> None:
    page_size = 4
    total_slots = 1 * 8
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    seq = Sequence(seq_id=1, prompt_token_ids=[1, 2, 3])

    assert memory_manager.ensure_mapping_length(seq, 3)
    mapping = memory_manager.get_mapping(seq)
    expected_pages = int(math.ceil(seq.logical_length / float(page_size)))
    assert len(mapping) == expected_pages

    memory_manager.release_sequence(seq)
    assert memory_manager.get_mapping(seq) == []
    assert memory_manager.free_slots_count() == allocator.total_slots

    _log(
        f"[INFO] mapping_pages={len(mapping)} free_slots={memory_manager.free_slots_count()}"
    )


def test_attention_memory_manager_mapping() -> None:
    attn = CausalMultiHeadAttention(dim=8, num_heads=2)
    page_size = 4
    total_slots = 1 * 8
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    seq = Sequence(seq_id=7, prompt_token_ids=[1, 2, 3])
    assert memory_manager.ensure_mapping_length(seq, 3)

    x = torch.randn(1, 3, 8)
    slot_mapping = torch.tensor(
        memory_manager.get_slot_mapping(seq, seq.logical_length), dtype=torch.long
    ).unsqueeze(0)
    out = attn(
        x,
        page_allocator=allocator,
        slot_mapping=slot_mapping,
        layer_idx=0,
    )
    assert out.shape == x.shape
    _log(f"[INFO] attention_out_shape={tuple(out.shape)}")


def test_batch_prep_memory_manager() -> None:
    page_size = 4
    total_slots = 2 * 8
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    seq_a = Sequence(seq_id=1, prompt_token_ids=[10, 11, 12])
    seq_b = Sequence(seq_id=2, prompt_token_ids=[20, 21], generated_token_ids=[30, 31])

    assert memory_manager.ensure_mapping_length(seq_a, seq_a.logical_length)
    assert memory_manager.ensure_mapping_length(seq_b, seq_b.logical_length)

    batch = prepare_continuous_batch([seq_a, seq_b], memory_manager)
    _log(
        "[INFO] batch_prep "
        f"input_len={batch['input_ids'].numel()} slot_len={batch['slot_mapping'].numel()}"
    )


def test_scheduler_eviction_and_preemption() -> None:
    page_size = 8
    total_slots = 1 * 32
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    scheduler = Scheduler(max_batch_size=2)

    long_seq = Sequence(seq_id=1, prompt_token_ids=[1])
    long_seq.generated_token_ids = list(range(24))
    assert memory_manager.ensure_mapping_length(long_seq, long_seq.logical_length)
    long_seq.status = SequenceStatus.RUNNING

    short_seq = Sequence(seq_id=2, prompt_token_ids=[1, 2])
    scheduler.active_batch.append(long_seq)
    scheduler.add_request(short_seq)

    preempted = scheduler.step_preemption(
        memory_manager=memory_manager,
        waiting_threshold=0,
        long_context_tokens=10,
        min_decode_tokens=2,
    )
    assert preempted is not None
    assert preempted.status == SequenceStatus.PREEMPTED
    assert scheduler.active_batch and scheduler.active_batch[0].seq_id == short_seq.seq_id

    scheduler.active_batch[0].status = SequenceStatus.FINISHED
    scheduler.step_eviction(memory_manager)
    _log(
        "[INFO] preemption "
        f"preempted={preempted.seq_id} active={len(scheduler.active_batch)} waiting={len(scheduler.waiting_queue)}"
    )


def test_engine_metrics_and_eviction() -> None:
    model = DummyModel(vocab_size=256, max_seq_len=2048)
    page_size = 16
    total_slots = 32 * 2048
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    scheduler = Scheduler(max_batch_size=32)

    for seq_id in range(1, 33):
        prompt_tokens = [token_id % 256 for token_id in range(1024)]
        scheduler.add_request(Sequence(seq_id=seq_id, prompt_token_ids=prompt_tokens))

    metrics_path = "/tmp/engine_metrics_test.csv"
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    run_engine(
        model,
        scheduler,
        memory_manager,
        eos_token_id=-1,
        max_seq_len=1536,
        top_k=20,
        metrics_path=metrics_path,
    )

    assert os.path.exists(metrics_path)
    with open(metrics_path, "r", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) >= 2
    _log(f"[INFO] metrics_rows={len(rows) - 1} path={metrics_path}")


def test_fairness_tracer_output() -> None:
    model = DummyModel(vocab_size=128, max_seq_len=512)
    page_size = 16
    total_slots = 16 * 512
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    scheduler = Scheduler(max_batch_size=16)

    for seq_id in range(1, 17):
        prompt_tokens = [token_id % 128 for token_id in range(256)]
        scheduler.add_request(Sequence(seq_id=seq_id, prompt_token_ids=prompt_tokens))

    tracer_path = "/tmp/engine_fairness_metrics_test.csv"
    if os.path.exists(tracer_path):
        os.remove(tracer_path)

    runtime_tracer = RuntimeTracer()
    run_engine(
        model,
        scheduler,
        memory_manager,
        eos_token_id=-1,
        max_seq_len=384,
        top_k=10,
        runtime_tracer=runtime_tracer,
        tracer_dump_path=tracer_path,
    )

    assert os.path.exists(tracer_path)
    with open(tracer_path, "r", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) >= 2
    header = {name.strip() for name in rows[0]}
    assert "avg_wait_s" in header
    assert "p95_wait_s" in header
    assert "max_starvation_s" in header
    assert "short_long_ratio" in header
    _log(f"[INFO] fairness_rows={len(rows) - 1} path={tracer_path}")


def test_async_workload_with_pressure() -> None:
    model = DummyModel(vocab_size=256, max_seq_len=4096)
    page_size = 16
    total_slots = 64 * 4096
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    scheduler = Scheduler(max_batch_size=64)
    stop_event = threading.Event()
    stress_profile = WorkloadProfile(
        name="stress_pressure",
        base_rate=80.0,
        burst_rate=200.0,
        burst_prob=0.7,
        prompt_lengths=[128, 256, 512, 1024, 1536, 2048, 3072],
        prompt_weights=[0.12, 0.14, 0.18, 0.2, 0.14, 0.12, 0.1],
        min_decode_tokens=256,
        max_decode_tokens=512,
    )
    WORKLOAD_PROFILES[stress_profile.name] = stress_profile
    thread = start_engine_background(
        model,
        scheduler,
        memory_manager,
        stop_event,
        eos_token_id=-1,
        max_seq_len=4096,
        top_k=20,
    )

    async def _run():
        await run_synthetic_workload(
            scheduler,
            duration_s=1.5,
            base_rate=stress_profile.base_rate,
            burst_rate=stress_profile.burst_rate,
            burst_prob=stress_profile.burst_prob,
            vocab_size=256,
            profile=stress_profile.name,
            stop_event=stop_event,
        )

    try:
        asyncio.run(_run())
    finally:
        stop_engine_background(stop_event, thread, timeout_s=2.0)
    _log(
        "[INFO] workload_pressure "
        f"waiting={len(scheduler.waiting_queue)} active={len(scheduler.active_batch)}"
    )


def benchmark_scheduler(iterations: int = 5000, batch_size: int = 256) -> None:
    scheduler = Scheduler(max_batch_size=batch_size)
    page_size = 16
    total_slots = batch_size * 256
    total_pages = int(math.ceil(total_slots / float(page_size)))
    allocator = PageAllocator(
        total_num_pages=total_pages,
        page_size=page_size,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    memory_manager = MemoryManager(allocator)
    sequences = [Sequence(seq_id=i, prompt_token_ids=list(range(128))) for i in range(batch_size)]

    start = time.perf_counter()
    for _ in range(iterations):
        scheduler.waiting_queue.clear()
        for seq in sequences:
            scheduler.add_request(seq)
        scheduler.active_batch.clear()
        scheduler.schedule_next_iteration(policy="fcfs", memory_manager=memory_manager)
    elapsed = time.perf_counter() - start

    per_iter_ms = (elapsed / iterations) * 1000.0
    throughput = iterations / elapsed if elapsed > 0 else 0.0
    _log(
        f"Scheduler benchmark: {per_iter_ms:.4f} ms/iter | {throughput:.2f} it/s | "
        f"batch_size={batch_size}"
    )


def test_workload_profiles() -> None:
    profile_names = sorted(WORKLOAD_PROFILES.keys())
    _log(f"[INFO] workload_profiles={profile_names}")
    assert "bursty_chat" in WORKLOAD_PROFILES
    assert "heavy_document_qa" in WORKLOAD_PROFILES
    assert "mixed_contention" in WORKLOAD_PROFILES
    assert "persistent_long_context" in WORKLOAD_PROFILES
    assert "starvation_pressure" in WORKLOAD_PROFILES
    assert "sustained_overload" in WORKLOAD_PROFILES
    assert "mixed_decode_tail_latency" in WORKLOAD_PROFILES


def run_all() -> None:
    _run_test("memory_manager_mapping", test_memory_manager_mapping)
    _run_test("attention_memory_manager", test_attention_memory_manager_mapping)
    _run_test("batch_prep_memory_manager", test_batch_prep_memory_manager)
    _run_test("scheduler_eviction_preemption", test_scheduler_eviction_and_preemption)
    _run_test("engine_metrics_eviction", test_engine_metrics_and_eviction)
    _run_test("fairness_tracer_output", test_fairness_tracer_output)
    _run_test("async_workload_pressure", test_async_workload_with_pressure)
    _run_test("workload_profiles", test_workload_profiles)
    _run_test("scheduler_benchmark", benchmark_scheduler)
    _log("All tests completed")


if __name__ == "__main__":
    run_all()
