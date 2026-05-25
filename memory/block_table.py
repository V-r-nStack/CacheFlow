"""Logical-to-physical block mapping for paged KV cache."""

from __future__ import annotations

import math
from typing import Dict, List

from memory.page_allocator import PageAllocator


class BlockTable:
    """Track logical blocks per sequence and map them to physical pages."""

    def __init__(self, page_allocator: PageAllocator) -> None:
        if not isinstance(page_allocator, PageAllocator):
            raise TypeError("page_allocator must be a PageAllocator instance")
        self._allocator = page_allocator
        self._mapping: Dict[int, List[int]] = {}

    @property
    def page_allocator(self) -> PageAllocator:
        return self._allocator

    @property
    def page_size(self) -> int:
        return int(self._allocator.page_size)

    def free_pages_count(self) -> int:
        return self._allocator.free_pages_count()

    def free_slots_count(self) -> int:
        return self._allocator.free_pages_count() * self._allocator.page_size

    def get_block_mapping(self, seq_id: int) -> List[int]:
        return list(self._mapping.get(int(seq_id), []))

    def get_slot_mapping(self, seq_id: int, target_len: int) -> List[int]:
        target_len = int(target_len)
        if target_len <= 0:
            return []

        blocks = self._mapping.get(int(seq_id), [])
        if not blocks:
            return []

        page_size = self._allocator.page_size
        slot_mapping: List[int] = []
        for page_index in blocks:
            start = int(page_index) * page_size
            slot_mapping.extend(range(start, start + page_size))
            if len(slot_mapping) >= target_len:
                break
        return slot_mapping[:target_len]

    def get_token_capacity(self, seq_id: int) -> int:
        blocks = self._mapping.get(int(seq_id), [])
        return len(blocks) * self._allocator.page_size

    def ensure_logical_blocks(self, seq_id: int, target_len: int) -> bool:
        target_len = int(target_len)
        if target_len <= 0:
            return True

        mapping = self._mapping.setdefault(int(seq_id), [])
        target_blocks = int(math.ceil(target_len / float(self._allocator.page_size)))
        blocks_needed = target_blocks - len(mapping)
        if blocks_needed <= 0:
            return True
        try:
            allocated = self._allocator.allocate_pages(blocks_needed)
        except RuntimeError:
            return False
        mapping.extend(int(page_index) for page_index in allocated)
        return True

    def release_sequence(self, seq_id: int) -> None:
        mapping = self._mapping.pop(int(seq_id), [])
        if mapping:
            self._allocator.free_pages(mapping)


__all__ = ["BlockTable"]
