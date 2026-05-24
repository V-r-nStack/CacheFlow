"""Pure request-state tracking for transformer inference.

This module intentionally contains no tensor objects, model calls, or runtime
execution logic. It only tracks per-request metadata and token IDs.
"""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.memory_manager import MemoryManager


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

    Logical tokens are mapped to physical KV slots through the MemoryManager,
    keeping raw slot indices out of this state object.
    """

    seq_id: int
    prompt_token_ids: List[int] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time)
    queued_at: Optional[float] = None
    admitted_at: Optional[float] = None
    preempted_at: Optional[float] = None
    prefill_start_at: Optional[float] = None
    prefill_end_at: Optional[float] = None
    first_decode_start_at: Optional[float] = None
    first_decode_end_at: Optional[float] = None
    finish_time: Optional[float] = None
    ttft_s: Optional[float] = None
    total_latency_s: Optional[float] = None
    decode_limit: Optional[int] = None
    status: SequenceStatus = SequenceStatus.WAITING
    total_wait_time: float = 0.0
    starvation_duration: float = 0.0
    preemption_count: int = 0
    _wait_start_time: Optional[float] = None
    _steady_state_itl_sum: float = 0.0
    _steady_state_itl_count: int = 0

    @property
    def logical_length(self) -> int:
        """Return the total token count tracked by this request."""

        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def request_blocks(self, memory_manager: "MemoryManager", target_len: int) -> bool:
        """Ensure the sequence has a logical-to-physical mapping of target_len."""

        return memory_manager.ensure_mapping_length(self, target_len)

    def release_blocks(self, memory_manager: "MemoryManager") -> None:
        """Release all mapped blocks for this sequence via the memory manager."""

        memory_manager.release_sequence(self)


InferenceRequest = Sequence


__all__ = ["InferenceRequest", "Sequence", "SequenceStatus"]