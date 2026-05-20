"""Queue management for LLM inference requests.

This module stays strictly focused on request placement and batch bookkeeping.
It does not perform model execution, tensor allocation, or KV-cache access.
"""

from typing import List

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


__all__ = ["Scheduler"]