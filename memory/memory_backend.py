"""Pluggable KV memory backends for transformer inference."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import torch

from memory.block_table import BlockTable
from memory.page_allocator import PageAllocator
from tracing.tracer import RuntimeTracer


@dataclass(frozen=True)
class MemoryBackendStats:
    total_slots: int
    free_slots: int
    allocated_slots: int
    total_allocated_slots: int
    total_freed_slots: int


class MemoryBackend(ABC):
    """Minimal KV memory operations required by runtime scheduling and attention.

    Conceptual mapping for experiment tooling:

    - allocate: ``ensure_sequence_capacity``
    - free: ``release_sequence``
    - append decode tokens + gather KV for attention: ``materialize_kv``
    - stats: ``stats``
    """

    backend_kind: str

    @abstractmethod
    def ensure_sequence_capacity(self, sequence_id: int, target_len: int) -> bool:
        """Grow or allocate KV residency for ``sequence_id`` up to ``target_len`` tokens."""
        raise NotImplementedError

    @abstractmethod
    def release_sequence(self, sequence_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_mapping(self, sequence_id: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def get_slot_mapping(self, sequence_id: int, target_len: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def get_token_capacity(self, sequence_id: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def materialize_kv(
        self,
        layer_idx: int,
        sequence_id: int,
        logical_length: int,
        key: torch.Tensor,
        value: torch.Tensor,
        runtime_tracer: Optional[RuntimeTracer] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def free_slots_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def total_slots(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> MemoryBackendStats:
        raise NotImplementedError


class ContiguousMemoryBackend(MemoryBackend):
    """Contiguous KV residency with direct slot ownership."""

    backend_kind = "contiguous"

    def __init__(
        self,
        total_slots: int,
        page_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        total_slots = int(total_slots)
        page_size = int(page_size)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        head_dim = int(head_dim)

        if total_slots <= 0 or page_size <= 0:
            raise ValueError("total_slots and page_size must be positive")
        if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16

        self._total_slots = total_slots
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache = torch.zeros(
            (2, num_layers, total_slots, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._free_ranges: List[Tuple[int, int]] = [(0, total_slots)]
        self._allocations: Dict[int, Tuple[int, int]] = {}
        self._total_allocated_slots = 0
        self._total_freed_slots = 0

    def _merge_free_ranges(self) -> None:
        if not self._free_ranges:
            return
        self._free_ranges.sort(key=lambda span: span[0])
        merged: List[Tuple[int, int]] = [self._free_ranges[0]]
        for start, length in self._free_ranges[1:]:
            last_start, last_length = merged[-1]
            last_end = last_start + last_length
            current_end = start + length
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end) - last_start)
            else:
                merged.append((start, length))
        self._free_ranges = merged

    def _allocate_span(self, target_len: int) -> Optional[Tuple[int, int]]:
        for idx, (start, length) in enumerate(self._free_ranges):
            if length < target_len:
                continue
            allocated = (start, target_len)
            remainder = length - target_len
            if remainder > 0:
                self._free_ranges[idx] = (start + target_len, remainder)
            else:
                self._free_ranges.pop(idx)
            return allocated
        return None

    def ensure_sequence_capacity(self, sequence_id: int, target_len: int) -> bool:
        sequence_id = int(sequence_id)
        target_len = int(target_len)
        if target_len <= 0:
            return True

        current = self._allocations.get(sequence_id)
        if current is not None and current[1] >= target_len:
            return True

        new_span = self._allocate_span(target_len)
        if new_span is None:
            return False

        if current is not None:
            old_start, old_len = current
            new_start, _new_len = new_span
            copy_len = min(old_len, target_len)
            for cache_kind in (0, 1):
                self.cache[cache_kind, :, new_start:new_start + copy_len].copy_(
                    self.cache[cache_kind, :, old_start:old_start + copy_len]
                )
            self._free_ranges.append((old_start, old_len))
            self._merge_free_ranges()
        self._allocations[sequence_id] = new_span
        self._total_allocated_slots += target_len if current is None else max(0, target_len - current[1])
        if current is not None:
            self._total_freed_slots += current[1]
        return True

    def release_sequence(self, sequence_id: int) -> None:
        allocation = self._allocations.pop(int(sequence_id), None)
        if allocation is None:
            return
        start, length = allocation
        self._total_freed_slots += length
        self._free_ranges.append((start, length))
        self._merge_free_ranges()

    def get_mapping(self, sequence_id: int) -> List[int]:
        allocation = self._allocations.get(int(sequence_id))
        if allocation is None:
            return []
        start, length = allocation
        return list(range(start, start + length))

    def get_slot_mapping(self, sequence_id: int, target_len: int) -> List[int]:
        target_len = int(target_len)
        if target_len <= 0:
            return []
        if not self.ensure_sequence_capacity(sequence_id, target_len):
            return []
        start, length = self._allocations[int(sequence_id)]
        return list(range(start, start + min(length, target_len)))

    def get_token_capacity(self, sequence_id: int) -> int:
        allocation = self._allocations.get(int(sequence_id))
        return int(allocation[1]) if allocation is not None else 0

    def materialize_kv(
        self,
        layer_idx: int,
        sequence_id: int,
        logical_length: int,
        key: torch.Tensor,
        value: torch.Tensor,
        runtime_tracer: Optional[RuntimeTracer] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key.dim() != 4 or value.dim() != 4:
            raise ValueError("key and value must have shape [B, H, T, D]")

        batch_size, num_heads, seq_len, head_dim = key.shape
        if batch_size != 1:
            raise ValueError("ContiguousMemoryBackend currently expects batch_size == 1")
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("key/value head shape does not match backend configuration")

        logical_length = int(logical_length)
        if logical_length < seq_len:
            raise ValueError("logical_length must be >= input sequence length")
        if not self.ensure_sequence_capacity(sequence_id, logical_length):
            raise RuntimeError("No contiguous KV capacity available")

        slot_mapping = self.get_slot_mapping(sequence_id, logical_length)
        write_slots = torch.tensor(slot_mapping[-seq_len:], device=key.device, dtype=torch.long)

        key_to_store = key.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
        value_to_store = value.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)

        flat_cache_k = self.cache[0, layer_idx].view(-1, self.num_heads, self.head_dim)
        flat_cache_v = self.cache[1, layer_idx].view(-1, self.num_heads, self.head_dim)

        with torch.no_grad():
            flat_cache_k.index_copy_(0, write_slots, key_to_store)
            flat_cache_v.index_copy_(0, write_slots, value_to_store)

        read_slots = torch.tensor(slot_mapping, device=key.device, dtype=torch.long)
        gathered_key = flat_cache_k.index_select(0, read_slots)
        gathered_value = flat_cache_v.index_select(0, read_slots)
        gathered_key = gathered_key.view(1, logical_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        gathered_value = gathered_value.view(1, logical_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return gathered_key, gathered_value

    def free_slots_count(self) -> int:
        return sum(length for _, length in self._free_ranges)

    def total_slots(self) -> int:
        return self._total_slots

    def stats(self) -> MemoryBackendStats:
        allocated_slots = self._total_slots - self.free_slots_count()
        return MemoryBackendStats(
            total_slots=self._total_slots,
            free_slots=self.free_slots_count(),
            allocated_slots=allocated_slots,
            total_allocated_slots=self._total_allocated_slots,
            total_freed_slots=self._total_freed_slots,
        )


class PagedMemoryBackend(MemoryBackend):
    """Paged fragmented KV residency backed by PageAllocator + BlockTable."""

    backend_kind = "paged"

    def __init__(self, page_allocator: PageAllocator) -> None:
        if not isinstance(page_allocator, PageAllocator):
            raise TypeError("page_allocator must be a PageAllocator instance")
        self.page_allocator = page_allocator
        self.block_table = BlockTable(page_allocator)

    @property
    def page_size(self) -> int:
        return int(self.page_allocator.page_size)

    def ensure_sequence_capacity(self, sequence_id: int, target_len: int) -> bool:
        return self.block_table.ensure_logical_blocks(sequence_id, target_len)

    def release_sequence(self, sequence_id: int) -> None:
        self.block_table.release_sequence(sequence_id)

    def get_mapping(self, sequence_id: int) -> List[int]:
        return self.block_table.get_block_mapping(sequence_id)

    def get_slot_mapping(self, sequence_id: int, target_len: int) -> List[int]:
        return self.block_table.get_slot_mapping(sequence_id, target_len)

    def get_token_capacity(self, sequence_id: int) -> int:
        return self.block_table.get_token_capacity(sequence_id)

    def materialize_kv(
        self,
        layer_idx: int,
        sequence_id: int,
        logical_length: int,
        key: torch.Tensor,
        value: torch.Tensor,
        runtime_tracer: Optional[RuntimeTracer] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key.dim() != 4 or value.dim() != 4:
            raise ValueError("key and value must have shape [B, H, T, D]")

        batch_size, num_heads, seq_len, head_dim = key.shape
        if batch_size != 1:
            raise ValueError("PagedMemoryBackend currently expects batch_size == 1")
        if num_heads != self.page_allocator.num_heads or head_dim != self.page_allocator.head_dim:
            raise ValueError("key/value head shape does not match backend configuration")

        logical_length = int(logical_length)
        if logical_length < seq_len:
            raise ValueError("logical_length must be >= input sequence length")
        if not self.ensure_sequence_capacity(sequence_id, logical_length):
            raise RuntimeError("No paged KV capacity available")

        slot_mapping = self.block_table.get_slot_mapping(sequence_id, logical_length)
        write_slots = torch.tensor(slot_mapping[-seq_len:], device=key.device, dtype=torch.long)

        key_to_store = key.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)
        value_to_store = value.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)

        flat_cache_k = self.page_allocator.cache[0, layer_idx].view(-1, num_heads, head_dim)
        flat_cache_v = self.page_allocator.cache[1, layer_idx].view(-1, num_heads, head_dim)

        with torch.no_grad():
            flat_cache_k.index_copy_(0, write_slots, key_to_store)
            flat_cache_v.index_copy_(0, write_slots, value_to_store)

        gather_start = time.perf_counter()
        page_indices = torch.tensor(
            self.block_table.get_block_mapping(sequence_id),
            device=key.device,
            dtype=torch.long,
        )
        cache_k_pages = self.page_allocator.cache[0, layer_idx].index_select(0, page_indices)
        cache_v_pages = self.page_allocator.cache[1, layer_idx].index_select(0, page_indices)
        cache_k_flat = cache_k_pages.contiguous().view(-1, num_heads, head_dim)
        cache_v_flat = cache_v_pages.contiguous().view(-1, num_heads, head_dim)

        gathered_key = cache_k_flat[:logical_length]
        gathered_value = cache_v_flat[:logical_length]
        gathered_key = gathered_key.view(1, logical_length, num_heads, head_dim).permute(0, 2, 1, 3)
        gathered_value = gathered_value.view(1, logical_length, num_heads, head_dim).permute(0, 2, 1, 3)

        if runtime_tracer is not None:
            runtime_tracer.record_page_gather_latency(time.perf_counter() - gather_start)

        return gathered_key, gathered_value

    def free_slots_count(self) -> int:
        return self.page_allocator.free_pages_count() * self.page_allocator.page_size

    def total_slots(self) -> int:
        return self.page_allocator.total_slots

    def stats(self) -> MemoryBackendStats:
        allocated_slots = self.total_slots() - self.free_slots_count()
        return MemoryBackendStats(
            total_slots=self.total_slots(),
            free_slots=self.free_slots_count(),
            allocated_slots=allocated_slots,
            total_allocated_slots=self.page_allocator.total_pages_allocated * self.page_allocator.page_size,
            total_freed_slots=self.page_allocator.total_pages_freed * self.page_allocator.page_size,
        )


__all__ = [
    "MemoryBackend",
    "MemoryBackendStats",
    "ContiguousMemoryBackend",
    "PagedMemoryBackend",
]