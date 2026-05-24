"""Memory manager for logical-to-physical KV page mapping."""

from __future__ import annotations

import math
from typing import Dict, List

from runtime.sequence import Sequence
from runtime.page_allocator import PageAllocator, PhysicalPage


class MemoryManager:
    """Manage logical blocks per sequence and map them to physical KV pages."""

    def __init__(self, page_allocator: PageAllocator):
        if not isinstance(page_allocator, PageAllocator):
            raise TypeError("page_allocator must be a PageAllocator instance")
        self._allocator = page_allocator
        self._mapping: Dict[int, List[PhysicalPage]] = {}

    @property
    def page_allocator(self) -> PageAllocator:
        return self._allocator

    def free_slots_count(self) -> int:
        return self._allocator.free_pages_count() * self._allocator.page_size

    def get_mapping(self, sequence: Sequence) -> List[PhysicalPage]:
        return list(self._mapping.get(sequence.seq_id, []))

    def get_mapping_by_id(self, seq_id: int) -> List[PhysicalPage]:
        return list(self._mapping.get(int(seq_id), []))

    def get_slot_mapping(self, sequence: Sequence, target_len: int) -> List[int]:
        return self.get_slot_mapping_by_id(sequence.seq_id, target_len)

    def get_slot_mapping_by_id(self, seq_id: int, target_len: int) -> List[int]:
        target_len = int(target_len)
        if target_len <= 0:
            return []

        pages = self._mapping.get(int(seq_id), [])
        if not pages:
            return []

        page_size = self._allocator.page_size
        slot_mapping: List[int] = []
        for page in pages:
            start = int(page.index) * page_size
            slot_mapping.extend(range(start, start + page_size))
            if len(slot_mapping) >= target_len:
                break
        return slot_mapping[:target_len]

    def get_token_capacity(self, sequence: Sequence) -> int:
        pages = self._mapping.get(sequence.seq_id, [])
        return len(pages) * self._allocator.page_size

    def ensure_mapping_length(self, sequence: Sequence, target_len: int) -> bool:
        target_len = int(target_len)
        if target_len <= 0:
            return True

        mapping = self._mapping.setdefault(sequence.seq_id, [])
        target_pages = int(math.ceil(target_len / float(self._allocator.page_size)))
        pages_needed = target_pages - len(mapping)
        if pages_needed <= 0:
            return True
        try:
            allocated = self._allocator.allocate_pages(pages_needed)
        except RuntimeError:
            return False
        mapping.extend(PhysicalPage(page_index) for page_index in allocated)
        return True

    def allocate_for_sequence(self, sequence: Sequence, count: int) -> bool:
        count = int(count)
        if count <= 0:
            return True
        target_len = sequence.logical_length
        if target_len <= 0:
            target_len = count
        return self.ensure_mapping_length(sequence, target_len)

    def release_sequence(self, sequence: Sequence) -> None:
        mapping = self._mapping.pop(sequence.seq_id, [])
        if mapping:
            self._allocator.free_pages([page.index for page in mapping])


__all__ = ["MemoryManager"]