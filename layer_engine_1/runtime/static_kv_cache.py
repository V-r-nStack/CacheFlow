"""Static KV cache bookkeeping with a free-slot pool.

This module pre-allocates a single KV tensor at boot to reduce fragmentation and
tracks which slot indices are available for reuse.
"""

from typing import Iterable, List, Optional

import torch


class StaticKVCache:
    """Pre-allocate KV cache storage and manage a free slot pool."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        max_batch_size = int(max_batch_size)
        max_seq_len = int(max_seq_len)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        head_dim = int(head_dim)

        if max_batch_size <= 0 or max_seq_len <= 0:
            raise ValueError("max_batch_size and max_seq_len must be positive")
        if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.total_slots = max_batch_size * max_seq_len
        # Shape: (2, num_layers, total_slots, num_heads, head_dim)
        self.cache = torch.zeros(
            (2, num_layers, self.total_slots, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        self.free_pool: List[int] = list(range(self.total_slots))

    def allocate_slot(self) -> int:
        """Allocate a single slot index from the free pool."""

        if not self.free_pool:
            raise RuntimeError("No KV cache slots available")
        return int(self.free_pool.pop())

    def free_slot(self, slot_index: int) -> None:
        """Return a single slot index back to the free pool."""

        self.free_pool.append(int(slot_index))

    def return_slots(self, slot_indices: Iterable[int]) -> None:
        """Return a collection of slot indices back to the free pool."""

        for slot_index in slot_indices:
            self.free_slot(slot_index)

    def free_slots_count(self) -> int:
        """Return the number of available KV cache slots."""

        return len(self.free_pool)


__all__ = ["StaticKVCache"]