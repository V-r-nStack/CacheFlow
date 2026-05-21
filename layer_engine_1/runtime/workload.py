"""Async synthetic workload generator for stress testing."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence as SeqType

from runtime.engine import run_engine
from runtime.memory_manager import MemoryManager
from runtime.scheduler import Scheduler
from runtime.sequence import Sequence


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    base_rate: float
    burst_rate: float
    burst_prob: float
    prompt_lengths: List[int]
    prompt_weights: List[float]
    min_decode_tokens: int
    max_decode_tokens: int


WORKLOAD_PROFILES = {
    "bursty_chat": WorkloadProfile(
        name="bursty_chat",
        base_rate=40.0,
        burst_rate=120.0,
        burst_prob=0.6,
        prompt_lengths=[20, 30, 40, 50],
        prompt_weights=[0.25, 0.25, 0.25, 0.25],
        min_decode_tokens=50,
        max_decode_tokens=100,
    ),
    "heavy_document_qa": WorkloadProfile(
        name="heavy_document_qa",
        base_rate=2.0,
        burst_rate=5.0,
        burst_prob=0.2,
        prompt_lengths=[1000, 1500, 2000],
        prompt_weights=[0.4, 0.3, 0.3],
        min_decode_tokens=10,
        max_decode_tokens=30,
    ),
    "mixed_contention": WorkloadProfile(
        name="mixed_contention",
        base_rate=20.0,
        burst_rate=80.0,
        burst_prob=0.5,
        prompt_lengths=[20, 30, 40, 50, 1000, 1500, 2000],
        prompt_weights=[0.125, 0.125, 0.125, 0.125, 0.166, 0.167, 0.167],
        min_decode_tokens=10,
        max_decode_tokens=100,
    ),
}


def start_engine_background(
    model,
    scheduler: Scheduler,
    memory_manager: MemoryManager,
    stop_event: threading.Event,
    **engine_kwargs,
) -> threading.Thread:
    """Run the engine loop in a background thread until stop_event is set."""

    def _worker():
        try:
            while not stop_event.is_set():
                if scheduler.waiting_queue or scheduler.active_batch:
                    run_engine(
                        model,
                        scheduler,
                        memory_manager,
                        stop_event=stop_event,
                        **engine_kwargs,
                    )
                else:
                    time.sleep(0.001)
        except Exception as exc:
            print(f"[WARN] engine background thread exception: {exc}")
        finally:
            print("[INFO] engine background thread stopped")

    thread = threading.Thread(target=_worker, daemon=False, name="engine-background")
    thread.start()
    return thread


def stop_engine_background(
    stop_event: threading.Event,
    thread: threading.Thread,
    timeout_s: float = 2.0,
) -> None:
    """Signal the background thread to stop and join it."""

    stop_event.set()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        print("[WARN] engine background thread did not stop in time")


def _sample_prompt_length(prompt_lengths: SeqType[int], weights: SeqType[float]) -> int:
    return int(random.choices(list(prompt_lengths), weights=weights, k=1)[0])


def _build_prompt_tokens(prompt_len: int, vocab_size: int) -> List[int]:
    return [random.randrange(vocab_size) for _ in range(prompt_len)]


def _sample_decode_limit(min_decode: int, max_decode: int) -> int:
    return int(random.randint(min_decode, max_decode))


async def run_synthetic_workload(
    scheduler: Scheduler,
    duration_s: float,
    base_rate: float,
    burst_rate: float,
    burst_prob: float = 0.2,
    vocab_size: int = 50257,
    prompt_lengths: Optional[Iterable[int]] = None,
    prompt_weights: Optional[Iterable[float]] = None,
    stop_event: Optional[threading.Event] = None,
    profile: Optional[str] = None,
) -> None:
    """Generate requests asynchronously using Poisson inter-arrival times."""

    selected_profile = None
    if profile is not None:
        selected_profile = WORKLOAD_PROFILES.get(profile)
        if selected_profile is None:
            raise ValueError(f"Unknown workload profile: {profile}")
        base_rate = selected_profile.base_rate
        burst_rate = selected_profile.burst_rate
        burst_prob = selected_profile.burst_prob
        prompt_lengths = selected_profile.prompt_lengths
        prompt_weights = selected_profile.prompt_weights

    if prompt_lengths is None:
        prompt_lengths = [20, 64, 128, 256, 500]
    if prompt_weights is None:
        prompt_weights = [0.4, 0.25, 0.2, 0.1, 0.05]

    prompt_lengths = list(prompt_lengths)
    prompt_weights = list(prompt_weights)

    end_time = time.monotonic() + float(duration_s)
    seq_id = 0

    try:
        while time.monotonic() < end_time:
            if stop_event is not None and stop_event.is_set():
                break
            rate = burst_rate if random.random() < burst_prob else base_rate
            rate = max(rate, 1e-6)
            await asyncio.sleep(random.expovariate(rate))

            seq_id += 1
            prompt_len = _sample_prompt_length(prompt_lengths, prompt_weights)
            prompt_tokens = _build_prompt_tokens(prompt_len, vocab_size)

            if selected_profile is not None:
                decode_limit = _sample_decode_limit(
                    selected_profile.min_decode_tokens,
                    selected_profile.max_decode_tokens,
                )
            else:
                decode_limit = 0

            sequence = Sequence(seq_id=seq_id, prompt_token_ids=prompt_tokens)
            if decode_limit > 0:
                sequence.decode_limit = decode_limit
            scheduler.enqueue_request(sequence)
    except asyncio.CancelledError:
        print("[INFO] workload generator cancelled")
        raise


__all__ = [
    "run_synthetic_workload",
    "start_engine_background",
    "stop_engine_background",
    "WorkloadProfile",
    "WORKLOAD_PROFILES",
]