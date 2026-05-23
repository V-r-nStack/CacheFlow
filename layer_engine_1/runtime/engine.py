"""Continuous batching event loop for token-by-token decoding."""

from __future__ import annotations

import csv
import os
import time
import threading
from dataclasses import dataclass
from typing import List, Optional

import torch

from runtime.batching import prepare_continuous_batch
from runtime.memory_manager import MemoryManager
from runtime.scheduler import Scheduler
from runtime.sequence import Sequence, SequenceStatus
from runtime.tracer import RuntimeTracer


@dataclass
class SimulatedComputePacer:
    base_delay_s: float = 0.0
    per_sequence_delay_s: float = 0.0
    per_token_delay_s: float = 0.0
    max_delay_s: float = 0.1

    def compute_delay_s(self, active_batch_size: int, total_context_tokens: int) -> float:
        delay = (
            self.base_delay_s
            + self.per_sequence_delay_s * float(active_batch_size)
            + self.per_token_delay_s * float(total_context_tokens)
        )
        if self.max_delay_s is not None:
            delay = min(delay, self.max_delay_s)
        return max(delay, 0.0)


def _sample_next_token(
    logits: torch.Tensor,
    prior_token_ids: List[int],
    temperature: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> int:
    temperature = max(float(temperature), 1e-8)
    scaled_logits = logits / temperature

    if repetition_penalty not in (None, 1.0) and prior_token_ids:
        repeated = torch.unique(torch.tensor(prior_token_ids, device=logits.device))
        if repeated.numel() > 0:
            adjusted = scaled_logits.clone()
            repeated_logits = adjusted.index_select(dim=-1, index=repeated)
            repeated_logits = torch.where(
                repeated_logits < 0,
                repeated_logits * repetition_penalty,
                repeated_logits / repetition_penalty,
            )
            adjusted.index_copy_(dim=-1, index=repeated, source=repeated_logits)
            scaled_logits = adjusted

    if top_k is not None and top_k > 0 and top_k < scaled_logits.size(-1):
        values, _ = torch.topk(scaled_logits, top_k, dim=-1)
        cutoff = values[..., -1, None]
        scaled_logits = scaled_logits.masked_fill(
            scaled_logits < cutoff, torch.finfo(scaled_logits.dtype).min
        )

    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def _assign_prompt_slots(
    sequence: Sequence,
    memory_manager: MemoryManager,
    prompt_len: int,
) -> bool:
    return sequence.request_blocks(memory_manager, prompt_len)


def run_engine(
    model,
    scheduler: Scheduler,
    memory_manager: MemoryManager,
    eos_token_id: int = 50256,
    max_seq_len: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    policy: str = "fcfs",
    min_decode_tokens: int = 50,
    preempt_waiting_threshold: Optional[int] = None,
    preempt_long_context_tokens: Optional[int] = None,
    compute_pacer: Optional[SimulatedComputePacer] = None,
    metrics_path: Optional[str] = None,
    runtime_tracer: Optional[RuntimeTracer] = None,
    tracer_dump_path: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the continuous decoding loop until all work is complete."""

    if max_seq_len is None and hasattr(model, "max_seq_len"):
        max_seq_len = int(model.max_seq_len)

    device = next(model.parameters()).device
    static_kv_cache = memory_manager.static_kv_cache

    metrics_writer = None
    metrics_file = None
    if metrics_path is not None:
        metrics_exists = os.path.exists(metrics_path)
        metrics_file = open(metrics_path, "a", newline="")
        metrics_writer = csv.writer(metrics_file)
        if not metrics_exists:
            metrics_writer.writerow(
                [
                    "timestamp",
                    "tokens_per_sec",
                    "queue_depth",
                    "active_utilization",
                    "itl_s",
                    "active_batch_size",
                    "tokens_generated",
                ]
            )

    try:
        while scheduler.waiting_queue or scheduler.active_batch:
            if stop_event is not None and stop_event.is_set():
                break
            scheduler.step_eviction(memory_manager)
            scheduler.schedule_next_iteration(
                policy=policy,
                memory_manager=memory_manager,
                min_decode_tokens=min_decode_tokens,
                preempt_waiting_threshold=preempt_waiting_threshold,
                preempt_long_context_tokens=preempt_long_context_tokens,
            )

            if not scheduler.active_batch:
                if metrics_writer is not None:
                    queue_depth = len(scheduler.waiting_queue)
                    metrics_writer.writerow(
                        [
                            f"{time.time():.6f}",
                            "0.000000",
                            queue_depth,
                            "0.000000",
                            "0.000000",
                            0,
                            0,
                        ]
                    )
                    metrics_file.flush()
                continue

            for sequence in scheduler.active_batch:
                if sequence.status == SequenceStatus.WAITING:
                    sequence.status = SequenceStatus.RUNNING

            tick_start = time.perf_counter()
            tokens_generated = 0
            total_itl_s = 0.0

            for sequence in scheduler.active_batch:
                if stop_event is not None and stop_event.is_set():
                    break
                prompt_len = len(sequence.prompt_token_ids)
                generated_len = len(sequence.generated_token_ids)
                logical_len = prompt_len + generated_len

                if logical_len == 0:
                    continue

                if generated_len == 0:
                    if not _assign_prompt_slots(sequence, memory_manager, prompt_len):
                        sequence.status = SequenceStatus.FINISHED
                        sequence.finish_time = time.time()
                        continue
                    input_ids = torch.tensor(
                        sequence.prompt_token_ids, dtype=torch.long, device=device
                    ).unsqueeze(0)
                else:
                    mapping_list = memory_manager.get_mapping(sequence)
                    if len(mapping_list) < logical_len:
                        raise ValueError("memory manager mapping is behind logical length")

                    input_ids = torch.tensor(
                        [sequence.generated_token_ids[-1]], dtype=torch.long, device=device
                    ).unsqueeze(0)

                if generated_len == 0 and sequence.prefill_start_at is None:
                    sequence.prefill_start_at = time.time()
                elif generated_len == 1 and sequence.first_decode_start_at is None:
                    sequence.first_decode_start_at = time.time()

                start_time = time.perf_counter()
                with torch.inference_mode():
                    _ = prepare_continuous_batch([sequence], memory_manager, device=device)
                    logits = model(
                        input_ids,
                        static_kv_cache=static_kv_cache,
                        memory_manager=memory_manager,
                        sequence_id=sequence.seq_id,
                    )
                elapsed = time.perf_counter() - start_time

                if generated_len == 0 and sequence.prefill_end_at is None:
                    sequence.prefill_end_at = time.time()
                elif generated_len == 1 and sequence.first_decode_end_at is None:
                    sequence.first_decode_end_at = time.time()
                elif generated_len >= 2:
                    sequence._steady_state_itl_sum += float(elapsed)
                    sequence._steady_state_itl_count += 1

                if generated_len == 0 and sequence.ttft_s is None:
                    sequence.ttft_s = elapsed

                last_logits = logits[:, -1, :]
                prior_tokens = sequence.prompt_token_ids + sequence.generated_token_ids
                next_token_id = _sample_next_token(
                    last_logits,
                    prior_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )

                sequence.generated_token_ids.append(next_token_id)
                if not memory_manager.allocate_for_sequence(sequence, 1):
                    sequence.status = SequenceStatus.FINISHED
                    sequence.finish_time = time.time()
                    continue

                tokens_generated += 1
                total_itl_s += elapsed

                if max_seq_len is not None and sequence.logical_length >= max_seq_len:
                    sequence.status = SequenceStatus.FINISHED
                    sequence.finish_time = time.time()
                elif sequence.decode_limit is not None and len(sequence.generated_token_ids) >= sequence.decode_limit:
                    sequence.status = SequenceStatus.FINISHED
                    sequence.finish_time = time.time()
                elif next_token_id == eos_token_id:
                    sequence.status = SequenceStatus.FINISHED
                    sequence.finish_time = time.time()

            tick_elapsed = time.perf_counter() - tick_start
            if tokens_generated > 0 and tick_elapsed > 0:
                tokens_per_sec = tokens_generated / tick_elapsed
            else:
                tokens_per_sec = 0.0

            if tokens_generated > 0:
                itl_s = total_itl_s / tokens_generated
            else:
                itl_s = 0.0

            queue_depth = len(scheduler.waiting_queue)
            active_batch_size = len(scheduler.active_batch)
            active_utilization = (
                active_batch_size / scheduler.max_batch_size
                if scheduler.max_batch_size > 0
                else 0.0
            )
            batch_utilization_ratio = active_utilization
            idle_decode_slots = max(scheduler.max_batch_size - active_batch_size, 0)

            if metrics_writer is not None:
                metrics_writer.writerow(
                    [
                        f"{time.time():.6f}",
                        f"{tokens_per_sec:.6f}",
                        queue_depth,
                        f"{active_utilization:.6f}",
                        f"{itl_s:.6f}",
                        active_batch_size,
                        tokens_generated,
                    ]
                )
                metrics_file.flush()

            if runtime_tracer is not None:
                free_slots = memory_manager.free_slots_count()
                allocated_slots = static_kv_cache.total_slots - free_slots
                fairness = scheduler.aggregate_fairness_metrics()
                latency_breakdown = scheduler.aggregate_latency_metrics()
                runtime_tracer.record_tick(
                    timestamp=time.time(),
                    queue_depth=queue_depth,
                    active_batch_size=active_batch_size,
                    allocated_kv_slots=allocated_slots,
                    free_kv_slots=free_slots,
                    itl_s=itl_s,
                    max_batch_size=scheduler.max_batch_size,
                    active_sequence_ids=[seq.seq_id for seq in scheduler.active_batch],
                    avg_wait_s=fairness["avg_wait_s"],
                    p95_wait_s=fairness["p95_wait_s"],
                    max_starvation_s=fairness["max_starvation_s"],
                    short_long_ratio=fairness["short_long_ratio"],
                    queue_wait_latency=latency_breakdown["queue_wait_latency"],
                    admission_latency=latency_breakdown["admission_latency"],
                    compute_prefill_latency=latency_breakdown["compute_prefill_latency"],
                    first_decode_latency=latency_breakdown["first_decode_latency"],
                    steady_state_itl=latency_breakdown["steady_state_itl"],
                    batch_utilization_ratio=batch_utilization_ratio,
                    idle_decode_slots=idle_decode_slots,
                )

            if compute_pacer is not None:
                total_context_tokens = sum(
                    sequence.logical_length for sequence in scheduler.active_batch
                )
                delay_s = compute_pacer.compute_delay_s(
                    active_batch_size=active_batch_size,
                    total_context_tokens=total_context_tokens,
                )
                if delay_s > 0:
                    time.sleep(delay_s)
    finally:
        if metrics_file is not None:
            metrics_file.close()
        if runtime_tracer is not None and tracer_dump_path is not None:
            runtime_tracer.dump_csv(tracer_dump_path)


__all__ = ["run_engine"]