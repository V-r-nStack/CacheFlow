"""Async synthetic workload generator for stress testing."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Iterable, List, Optional, Sequence as SeqType

from runtime.engine import run_engine
from runtime.memory_manager import MemoryManager
from runtime.scheduler import Scheduler
from runtime.sequence import Sequence


def start_engine_background(
    model,
    scheduler: Scheduler,
    memory_manager: MemoryManager,
    stop_event: threading.Event,
    **engine_kwargs,
) -> threading.Thread:
    """Run the engine loop in a background thread until stop_event is set."""

    def _worker():
        while not stop_event.is_set():
            if scheduler.waiting_queue or scheduler.active_batch:
                run_engine(model, scheduler, memory_manager, **engine_kwargs)
            else:
                time.sleep(0.001)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _sample_prompt_length(prompt_lengths: SeqType[int], weights: SeqType[float]) -> int:
    return int(random.choices(list(prompt_lengths), weights=weights, k=1)[0])


def _build_prompt_tokens(prompt_len: int, vocab_size: int) -> List[int]:
    return [random.randrange(vocab_size) for _ in range(prompt_len)]


async def run_synthetic_workload(
    scheduler: Scheduler,
    duration_s: float,
    base_rate: float,
    burst_rate: float,
    burst_prob: float = 0.2,
    vocab_size: int = 50257,
    prompt_lengths: Optional[Iterable[int]] = None,
    prompt_weights: Optional[Iterable[float]] = None,
) -> None:
    """Generate requests asynchronously using Poisson inter-arrival times."""

    if prompt_lengths is None:
        prompt_lengths = [20, 64, 128, 256, 500]
    if prompt_weights is None:
        prompt_weights = [0.4, 0.25, 0.2, 0.1, 0.05]

    prompt_lengths = list(prompt_lengths)
    prompt_weights = list(prompt_weights)

    end_time = time.monotonic() + float(duration_s)
    seq_id = 0

    while time.monotonic() < end_time:
        rate = burst_rate if random.random() < burst_prob else base_rate
        rate = max(rate, 1e-6)
        await asyncio.sleep(random.expovariate(rate))

        seq_id += 1
        prompt_len = _sample_prompt_length(prompt_lengths, prompt_weights)
        prompt_tokens = _build_prompt_tokens(prompt_len, vocab_size)

        sequence = Sequence(seq_id=seq_id, prompt_token_ids=prompt_tokens)
        scheduler.enqueue_request(sequence)


__all__ = ["run_synthetic_workload", "start_engine_background"]