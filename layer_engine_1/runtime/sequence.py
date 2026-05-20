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
    """

    seq_id: int
    prompt_token_ids: List[int] = field(default_factory=list)
    generated_token_ids: List[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time)
    finish_time: Optional[float] = None
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def logical_length(self) -> int:
        """Return the total token count tracked by this request."""

        return len(self.prompt_token_ids) + len(self.generated_token_ids)


InferenceRequest = Sequence


__all__ = ["InferenceRequest", "Sequence", "SequenceStatus"]