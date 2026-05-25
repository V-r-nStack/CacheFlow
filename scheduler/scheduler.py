"""Queue management for LLM inference requests.

This module stays strictly focused on request placement and batch bookkeeping.
It does not perform model execution, tensor allocation, or KV-cache access.
"""

from typing import Dict, List, Optional
import math
import threading
import time

from memory.memory_manager import MemoryManager
from runtime.sequence import Sequence, SequenceStatus


class Scheduler:
    """Manage waiting and active inference requests.

    The scheduler keeps two queues:
    - `waiting_queue`: requests that have arrived but are not yet active
    - `active_batch`: requests currently assigned to the next decode iteration

    It does not decide how model execution happens. It only tracks which
    sequences are waiting, which are active, and which completed requests can be
    removed to free batch capacity.
    """

    def __init__(self, max_batch_size: int):
        max_batch_size = int(max_batch_size)
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")

        self.max_batch_size = max_batch_size
        self.waiting_queue: List[Sequence] = []
        self.active_batch: List[Sequence] = []
        self.completed_sequences: List[Sequence] = []
        self._waiting_lock = threading.Lock()

    def add_request(self, sequence: Sequence) -> None:
        """Ingest a new Sequence object into the waiting queue."""

        if not isinstance(sequence, Sequence):
            raise TypeError("sequence must be a Sequence instance")

        now = time.time()
        if sequence.queued_at is None:
            sequence.queued_at = now
        self._mark_wait_start(sequence, now)
        with self._waiting_lock:
            self.waiting_queue.append(sequence)

    def enqueue_request(self, sequence: Sequence) -> None:
        """Thread-safe alias for add_request."""

        self.add_request(sequence)

    def _evict_completed(self) -> List[Sequence]:
        """Remove FINISHED sequences from the active batch.

        Returns the list of evicted sequences so the caller can inspect or log
        which requests were released before the next scheduling iteration.
        """

        evicted_sequences: List[Sequence] = []
        remaining_active_batch: List[Sequence] = []

        for sequence in self.active_batch:
            if sequence.status == SequenceStatus.FINISHED:
                evicted_sequences.append(sequence)
                continue
            remaining_active_batch.append(sequence)

        self.active_batch = remaining_active_batch
        return evicted_sequences

    def step_eviction(self, memory_manager: MemoryManager) -> List[Sequence]:
        """Evict FINISHED sequences, return KV slots, and log latency stats."""

        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager instance")

        evicted_sequences: List[Sequence] = []
        remaining_active_batch: List[Sequence] = []

        for sequence in self.active_batch:
            if sequence.status != SequenceStatus.FINISHED:
                remaining_active_batch.append(sequence)
                continue

            sequence.release_blocks(memory_manager)

            ttft_s = sequence.ttft_s
            total_latency_s = self._resolve_total_latency(sequence)
            self._log_sequence_latency(sequence, ttft_s, total_latency_s)

            evicted_sequences.append(sequence)
            self.completed_sequences.append(sequence)

        self.active_batch = remaining_active_batch
        return evicted_sequences

    @staticmethod
    def _mark_wait_start(sequence: Sequence, now: float) -> None:
        if sequence._wait_start_time is None:
            sequence._wait_start_time = now

    @staticmethod
    def _mark_wait_end(sequence: Sequence, now: float) -> None:
        if sequence._wait_start_time is not None:
            waited = max(0.0, now - sequence._wait_start_time)
            sequence.total_wait_time += waited
            sequence.starvation_duration += waited
            sequence._wait_start_time = None
        if sequence.status == SequenceStatus.PREEMPTED:
            sequence.status = SequenceStatus.WAITING

    @staticmethod
    def _current_starvation(sequence: Sequence, now: float) -> float:
        starvation = sequence.starvation_duration
        if sequence._wait_start_time is not None:
            starvation += max(0.0, now - sequence._wait_start_time)
        return float(starvation)

    def _gather_sequences(self) -> List[Sequence]:
        sequences: List[Sequence] = []
        seen_ids = set()
        for sequence in self.waiting_queue + self.active_batch + self.completed_sequences:
            seq_key = id(sequence)
            if seq_key in seen_ids:
                continue
            seen_ids.add(seq_key)
            sequences.append(sequence)
        return sequences

    def aggregate_fairness_metrics(self, short_prompt_threshold: int = 128) -> Dict[str, float]:
        now = time.time()
        wait_samples: List[float] = []
        max_starvation = 0.0
        for sequence in self._gather_sequences():
            wait_time = sequence.total_wait_time
            if sequence._wait_start_time is not None:
                wait_time += max(0.0, now - sequence._wait_start_time)
            if wait_time > 0.0:
                wait_samples.append(wait_time)

            starvation = self._current_starvation(sequence, now)
            if starvation > max_starvation:
                max_starvation = starvation

        avg_wait = sum(wait_samples) / len(wait_samples) if wait_samples else 0.0
        if wait_samples:
            ordered = sorted(wait_samples)
            idx = max(0, int(math.ceil(0.95 * len(ordered))) - 1)
            p95_wait = ordered[idx]
        else:
            p95_wait = 0.0

        short_count = 0
        long_count = 0
        for sequence in self.completed_sequences:
            prompt_len = len(sequence.prompt_token_ids)
            if prompt_len <= short_prompt_threshold:
                short_count += 1
            else:
                long_count += 1
        if long_count > 0:
            short_long_ratio = short_count / float(long_count)
        else:
            short_long_ratio = 0.0

        return {
            "avg_wait_s": float(avg_wait),
            "p95_wait_s": float(p95_wait),
            "max_starvation_s": float(max_starvation),
            "short_long_ratio": float(short_long_ratio),
        }

    def calculate_starvation_distribution(
        self,
        bins: Optional[List[float]] = None,
        short_prompt_threshold: int = 128,
    ) -> Dict[str, object]:
        now = time.time()
        samples: List[float] = []
        short_samples: List[float] = []
        long_samples: List[float] = []

        for sequence in self._gather_sequences():
            starvation = self._current_starvation(sequence, now)
            if starvation <= 0.0:
                continue
            samples.append(starvation)
            if len(sequence.prompt_token_ids) <= short_prompt_threshold:
                short_samples.append(starvation)
            else:
                long_samples.append(starvation)

        if bins is None:
            bins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

        def _percentile(values: List[float], q: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = max(0, int(math.ceil(q * len(ordered))) - 1)
            return float(ordered[idx])

        def _mean(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        hist_counts = [0 for _ in range(max(len(bins) - 1, 0))]
        overflow = 0
        for value in samples:
            placed = False
            for idx in range(len(bins) - 1):
                if bins[idx] <= value < bins[idx + 1]:
                    hist_counts[idx] += 1
                    placed = True
                    break
            if not placed:
                overflow += 1

        p50 = _percentile(samples, 0.5)
        p95 = _percentile(samples, 0.95)
        p99 = _percentile(samples, 0.99)

        short_p95 = _percentile(short_samples, 0.95)
        long_p95 = _percentile(long_samples, 0.95)
        short_avg = _mean(short_samples)
        long_avg = _mean(long_samples)

        long_tail_ratio_p99_p50 = p99 / p50 if p50 > 0.0 else 0.0
        long_tail_ratio_long_short_p95 = long_p95 / short_p95 if short_p95 > 0.0 else 0.0
        long_tail_ratio_long_short_avg = long_avg / short_avg if short_avg > 0.0 else 0.0

        return {
            "histogram_bins": bins,
            "histogram_counts": hist_counts,
            "histogram_overflow": overflow,
            "p95_starvation_s": p95,
            "p99_starvation_s": p99,
            "long_tail_ratio_p99_p50": long_tail_ratio_p99_p50,
            "long_tail_ratio_long_short_p95": long_tail_ratio_long_short_p95,
            "long_tail_ratio_long_short_avg": long_tail_ratio_long_short_avg,
        }

    def aggregate_latency_metrics(self) -> Dict[str, float]:
        now = time.time()
        queue_wait_samples: List[float] = []
        admission_samples: List[float] = []
        prefill_samples: List[float] = []
        first_decode_samples: List[float] = []
        steady_state_samples: List[float] = []

        for sequence in self._gather_sequences():
            if sequence.queued_at is not None:
                if sequence.admitted_at is not None:
                    queue_wait_samples.append(
                        max(0.0, sequence.admitted_at - sequence.queued_at)
                    )
                else:
                    queue_wait_samples.append(max(0.0, now - sequence.queued_at))

            if sequence.admitted_at is not None and sequence.prefill_start_at is not None:
                admission_samples.append(
                    max(0.0, sequence.prefill_start_at - sequence.admitted_at)
                )

            if sequence.prefill_start_at is not None and sequence.prefill_end_at is not None:
                prefill_samples.append(
                    max(0.0, sequence.prefill_end_at - sequence.prefill_start_at)
                )

            if (
                sequence.first_decode_start_at is not None
                and sequence.first_decode_end_at is not None
            ):
                first_decode_samples.append(
                    max(0.0, sequence.first_decode_end_at - sequence.first_decode_start_at)
                )

            if sequence._steady_state_itl_count > 0:
                steady_state_samples.append(
                    sequence._steady_state_itl_sum / sequence._steady_state_itl_count
                )

        return {
            "queue_wait_latency": self._mean(queue_wait_samples),
            "admission_latency": self._mean(admission_samples),
            "compute_prefill_latency": self._mean(prefill_samples),
            "first_decode_latency": self._mean(first_decode_samples),
            "steady_state_itl": self._mean(steady_state_samples),
        }

    @staticmethod
    def _mean(samples: List[float]) -> float:
        return float(sum(samples) / len(samples)) if samples else 0.0

    @staticmethod
    def _resolve_total_latency(sequence: Sequence) -> Optional[float]:
        if sequence.total_latency_s is not None:
            return sequence.total_latency_s
        if sequence.finish_time is None:
            return None
        return max(0.0, sequence.finish_time - sequence.arrival_time)

    @staticmethod
    def _log_sequence_latency(
        sequence: Sequence,
        ttft_s: Optional[float],
        total_latency_s: Optional[float],
    ) -> None:
        ttft_text = "N/A" if ttft_s is None else f"{ttft_s:.6f}s"
        total_text = "N/A" if total_latency_s is None else f"{total_latency_s:.6f}s"
        print(
            "Sequence "
            f"{sequence.seq_id} finished | TTFT: {ttft_text} | Total latency: {total_text}"
        )

    def schedule_next_iteration(
        self,
        policy: str = "fcfs",
        memory_manager: Optional[MemoryManager] = None,
        min_decode_tokens: int = 50,
        preempt_waiting_threshold: Optional[int] = None,
        preempt_long_context_tokens: Optional[int] = None,
    ) -> List[Sequence]:
        """Promote waiting requests into the active batch.

        Supported policies:
        - ``fcfs``: order by ``arrival_time`` from oldest to newest
        - ``shortest_prompt_first``: order by prompt token length, then arrival

        The method first evicts completed active requests, then fills any free
        capacity up to ``max_batch_size``. If a ``memory_manager`` is provided,
        the scheduler enforces memory-aware admission control by requiring enough
        free slots for ``prompt_len + min_decode_tokens``. Optional starvation
        control can preempt long-context sequences when the waiting queue grows
        beyond ``preempt_waiting_threshold``. If the waiting queue is empty, the
        method simply returns the current active batch state after eviction.
        """

        normalized_policy = str(policy).strip().lower()
        self._evict_completed()

        if preempt_waiting_threshold is not None and preempt_long_context_tokens is not None:
            if memory_manager is None:
                raise ValueError("memory_manager is required for preemption")
            self.step_preemption(
                memory_manager=memory_manager,
                waiting_threshold=preempt_waiting_threshold,
                long_context_tokens=preempt_long_context_tokens,
                min_decode_tokens=min_decode_tokens,
            )

        available_capacity = self.max_batch_size - len(self.active_batch)
        with self._waiting_lock:
            if available_capacity <= 0 or not self.waiting_queue:
                return self.active_batch
            waiting_snapshot = list(self.waiting_queue)

        if normalized_policy == "fcfs":
            ordered_waiting_queue = sorted(
                waiting_snapshot,
                key=lambda sequence: (sequence.arrival_time, sequence.seq_id),
            )
        elif normalized_policy == "shortest_prompt_first":
            ordered_waiting_queue = sorted(
                waiting_snapshot,
                key=lambda sequence: (
                    len(sequence.prompt_token_ids),
                    sequence.arrival_time,
                    sequence.seq_id,
                ),
            )
        else:
            raise ValueError(
                "policy must be either 'fcfs' or 'shortest_prompt_first'"
            )

        free_slots = None
        if memory_manager is not None:
            if not isinstance(memory_manager, MemoryManager):
                raise TypeError("memory_manager must be a MemoryManager instance")
            free_slots = memory_manager.free_slots_count()

        promoted_sequences: List[Sequence] = []
        for sequence in ordered_waiting_queue:
            if len(promoted_sequences) >= available_capacity:
                break
            if free_slots is not None:
                required_slots = len(sequence.prompt_token_ids) + int(min_decode_tokens)
                if required_slots > free_slots:
                    continue
                free_slots -= required_slots
            promoted_sequences.append(sequence)

        if not promoted_sequences:
            return self.active_batch

        now = time.time()
        for sequence in promoted_sequences:
            self._mark_wait_end(sequence, now)
            if sequence.admitted_at is None:
                sequence.admitted_at = now

        promoted_ids = {id(sequence) for sequence in promoted_sequences}

        with self._waiting_lock:
            self.waiting_queue = [
                sequence for sequence in self.waiting_queue if id(sequence) not in promoted_ids
            ]
        self.active_batch.extend(promoted_sequences)
        return self.active_batch

    def step_preemption(
        self,
        memory_manager: MemoryManager,
        waiting_threshold: int,
        long_context_tokens: int,
        min_decode_tokens: int = 50,
    ) -> Optional[Sequence]:
        """Preempt a long-running sequence to admit a shorter waiting request.

        Returns the preempted sequence when a preemption occurs, otherwise None.
        """

        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager instance")

        waiting_threshold = int(waiting_threshold)
        long_context_tokens = int(long_context_tokens)
        if waiting_threshold < 0 or long_context_tokens <= 0:
            return None

        with self._waiting_lock:
            waiting_depth = len(self.waiting_queue)
            waiting_snapshot = list(self.waiting_queue)

        if waiting_depth <= waiting_threshold:
            return None

        preempt_candidate = None
        preempt_len = -1
        for sequence in self.active_batch:
            if sequence.status != SequenceStatus.RUNNING:
                continue
            generated_len = len(sequence.generated_token_ids)
            if generated_len >= long_context_tokens and generated_len > preempt_len:
                preempt_candidate = sequence
                preempt_len = generated_len

        if preempt_candidate is None:
            return None

        self.active_batch = [
            sequence for sequence in self.active_batch if sequence is not preempt_candidate
        ]
        preempt_candidate.status = SequenceStatus.PREEMPTED
        preempt_candidate.preemption_count += 1
        preempt_candidate.release_blocks(memory_manager)
        now = time.time()
        preempt_candidate.preempted_at = now
        self._mark_wait_start(preempt_candidate, now)
        with self._waiting_lock:
            self.waiting_queue.append(preempt_candidate)

        free_slots = memory_manager.free_slots_count()
        admissible = []
        for sequence in sorted(waiting_snapshot, key=lambda seq: len(seq.prompt_token_ids)):
            required_slots = len(sequence.prompt_token_ids) + int(min_decode_tokens)
            if required_slots <= free_slots:
                admissible.append(sequence)
                free_slots -= required_slots
                break

        if admissible:
            admitted = admissible[0]
            with self._waiting_lock:
                self.waiting_queue = [
                    sequence for sequence in self.waiting_queue if sequence is not admitted
                ]
            self._mark_wait_end(admitted, now)
            if admitted.admitted_at is None:
                admitted.admitted_at = now
            self.active_batch.append(admitted)

        return preempt_candidate


__all__ = ["Scheduler"]