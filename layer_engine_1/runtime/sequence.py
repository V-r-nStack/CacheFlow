"""Pure request-state tracking for transformer inference.

This module intentionally contains no tensor objects, model calls, or runtime
execution logic. It only tracks per-request metadata and token IDs.
"""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import List, Optional


class SequenceStatus(str, Enum):
    """Lifecycle states for an inference request."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED = "FINISHED"


@dataclass
class Sequence:
    """Pure Python state container for one inference request.

    The class stores prompt and generated token IDs separately so the caller can
    manage prompt replay, decoding, and bookkeeping without coupling this object
    to any model execution code.

    The `kv_slot_indices` list stores the physical row indices assigned from a
    pre-allocated master KV cache tensor. Its order matches token order: entry 0
    maps to the first logical token in the sequence, entry 1 maps to the second
    logical token, and so on. A future PagedAttention runtime can use this as a
    logical-to-physical indirection layer while evicting or remapping rows in the
    shared cache.
    """

    seq_id: int
    prompt_token_ids: List[int] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    kv_slot_indices: List[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time)
    finish_time: Optional[float] = None
    ttft_s: Optional[float] = None
    total_latency_s: Optional[float] = None
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def logical_length(self) -> int:
        """Return the total token count tracked by this request."""

        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def append_kv_slot_index(self, slot_index: int) -> None:
        """Record the physical KV cache row for the next logical token."""

        self.kv_slot_indices.append(int(slot_index))

    def clear_kv_slot_indices(self) -> None:
        """Drop all slot mappings when the request is preempted or finished."""

        self.kv_slot_indices.clear()


InferenceRequest = Sequence


__all__ = ["InferenceRequest", "Sequence", "SequenceStatus"]