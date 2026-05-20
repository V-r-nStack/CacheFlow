"""Queue management for LLM inference requests.

This module stays strictly focused on request placement and batch bookkeeping.
It does not perform model execution, tensor allocation, or KV-cache access.
"""

from typing import List, Optional

from runtime.sequence import Sequence, SequenceStatus
from runtime.static_kv_cache import StaticKVCache


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

    def add_request(self, sequence: Sequence) -> None:
        """Ingest a new Sequence object into the waiting queue."""

        if not isinstance(sequence, Sequence):
            raise TypeError("sequence must be a Sequence instance")

        self.waiting_queue.append(sequence)

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

    def step_eviction(self, static_kv_cache: StaticKVCache) -> List[Sequence]:
        """Evict FINISHED sequences, return KV slots, and log latency stats."""

        if not isinstance(static_kv_cache, StaticKVCache):
            raise TypeError("static_kv_cache must be a StaticKVCache instance")

        evicted_sequences: List[Sequence] = []
        remaining_active_batch: List[Sequence] = []

        for sequence in self.active_batch:
            if sequence.status != SequenceStatus.FINISHED:
                remaining_active_batch.append(sequence)
                continue

            static_kv_cache.return_slots(sequence.kv_slot_indices)
            sequence.clear_kv_slot_indices()

            ttft_s = sequence.ttft_s
            total_latency_s = self._resolve_total_latency(sequence)
            self._log_sequence_latency(sequence, ttft_s, total_latency_s)

            evicted_sequences.append(sequence)

        self.active_batch = remaining_active_batch
        return evicted_sequences

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

    def schedule_next_iteration(self, policy: str = "fcfs") -> List[Sequence]:
        """Promote waiting requests into the active batch.

        Supported policies:
        - ``fcfs``: order by ``arrival_time`` from oldest to newest
        - ``shortest_prompt_first``: order by prompt token length, then arrival

        The method first evicts completed active requests, then fills any free
        capacity up to ``max_batch_size``. If the waiting queue is empty, the
        method simply returns the current active batch state after eviction.
        """

        normalized_policy = str(policy).strip().lower()
        self._evict_completed()

        available_capacity = self.max_batch_size - len(self.active_batch)
        if available_capacity <= 0 or not self.waiting_queue:
            return self.active_batch

        if normalized_policy == "fcfs":
            ordered_waiting_queue = sorted(
                self.waiting_queue,
                key=lambda sequence: (sequence.arrival_time, sequence.seq_id),
            )
        elif normalized_policy == "shortest_prompt_first":
            ordered_waiting_queue = sorted(
                self.waiting_queue,
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

        promoted_sequences = ordered_waiting_queue[:available_capacity]
        promoted_ids = {id(sequence) for sequence in promoted_sequences}

        self.waiting_queue = [
            sequence for sequence in self.waiting_queue if id(sequence) not in promoted_ids
        ]
        self.active_batch.extend(promoted_sequences)
        return self.active_batch


__all__ = ["Scheduler"]