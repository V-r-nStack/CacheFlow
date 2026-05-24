"""Memory manager wrapper for paged KV allocation."""

from __future__ import annotations

from runtime.page_allocator import PageAllocator
from runtime.block_table import BlockTable


class MemoryManager:
    """Expose the page allocator for legacy call sites."""

    def __init__(self, page_allocator: PageAllocator):
        if not isinstance(page_allocator, PageAllocator):
            raise TypeError("page_allocator must be a PageAllocator instance")
        self._allocator = page_allocator

    @property
    def page_allocator(self) -> PageAllocator:
        return self._allocator

    def free_slots_count(self) -> int:
        return self._allocator.free_pages_count() * self._allocator.page_size

    def attach_block_table(self, block_table) -> None:
        """Attach a BlockTable to centralize logical-to-physical mapping."""

        # Weak duck-typing: expect block_table to provide the BlockTable API
        setattr(self, "block_table", block_table)

    # Backwards-compatible wrappers for legacy call sites ---------------------------------
    def ensure_mapping_length(self, sequence, target_len: int) -> bool:
        block_table = getattr(self, "block_table", None)
        if block_table is None:
            # lazily create and attach a BlockTable for backwards compatibility
            block_table = BlockTable(self._allocator)
            self.attach_block_table(block_table)
        return block_table.ensure_logical_blocks(sequence.seq_id, target_len)

    def get_mapping(self, sequence):
        block_table = getattr(self, "block_table", None)
        if block_table is None:
            block_table = BlockTable(self._allocator)
            self.attach_block_table(block_table)
        return block_table.get_block_mapping(sequence.seq_id)

    def get_slot_mapping(self, sequence, target_len: int):
        block_table = getattr(self, "block_table", None)
        if block_table is None:
            block_table = BlockTable(self._allocator)
            self.attach_block_table(block_table)
        return block_table.get_slot_mapping(sequence.seq_id, target_len)

    def release_sequence(self, sequence) -> None:
        block_table = getattr(self, "block_table", None)
        if block_table is None:
            block_table = BlockTable(self._allocator)
            self.attach_block_table(block_table)
        return block_table.release_sequence(sequence.seq_id)

    def get_token_capacity(self, sequence) -> int:
        block_table = getattr(self, "block_table", None)
        if block_table is None:
            block_table = BlockTable(self._allocator)
            self.attach_block_table(block_table)
        return block_table.get_token_capacity(sequence.seq_id)

    def allocate_for_sequence(self, sequence, num_tokens: int) -> bool:
        """Ensure space for `num_tokens` additional generated tokens for `sequence`.

        This mirrors the older MemoryManager API used by the engine loop. It will
        prefer the scheduler-managed `BlockTable` when attached, otherwise lazily
        create one for backward compatibility.
        """
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            return True

        block_table = getattr(self, "block_table", None)
        if block_table is None:
            # lazily create a BlockTable to service legacy calls
            block_table = BlockTable(self._allocator)
            self.attach_block_table(block_table)

        target_len = sequence.logical_length + num_tokens
        return block_table.ensure_logical_blocks(sequence.seq_id, target_len)


__all__ = ["MemoryManager"]