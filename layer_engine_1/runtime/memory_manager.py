"""Compatibility wrapper around the pluggable KV memory backend."""

from __future__ import annotations

from runtime.memory_backend import (
    ContiguousMemoryBackend,
    MemoryBackend,
    PagedMemoryBackend,
)
from runtime.page_allocator import PageAllocator


class MemoryManager:
    """Expose the backend through the legacy MemoryManager API."""

    def __init__(self, backend_or_allocator, *, backend_kind: str = "paged", **backend_kwargs):
        if isinstance(backend_or_allocator, MemoryBackend):
            self._backend = backend_or_allocator
        elif isinstance(backend_or_allocator, PageAllocator):
            self._backend = PagedMemoryBackend(backend_or_allocator)
        elif backend_kind == "contiguous":
            self._backend = ContiguousMemoryBackend(**backend_kwargs)
        else:
            raise TypeError(
                "MemoryManager requires a MemoryBackend, a PageAllocator, or backend_kind='contiguous' with backend_kwargs"
            )

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    @property
    def backend_kind(self) -> str:
        return str(self._backend.backend_kind)

    @property
    def page_allocator(self):
        return getattr(self._backend, "page_allocator", None)

    def free_slots_count(self) -> int:
        return self._backend.free_slots_count()

    def ensure_mapping_length(self, sequence, target_len: int) -> bool:
        return self._backend.ensure_sequence_capacity(sequence.seq_id, target_len)

    def get_mapping(self, sequence):
        return self._backend.get_slot_mapping(sequence.seq_id, sequence.logical_length)

    def get_slot_mapping(self, sequence, target_len: int):
        return self._backend.get_slot_mapping(sequence.seq_id, target_len)

    def release_sequence(self, sequence) -> None:
        return self._backend.release_sequence(sequence.seq_id)

    def get_token_capacity(self, sequence) -> int:
        return self._backend.get_token_capacity(sequence.seq_id)

    def allocate_for_sequence(self, sequence, num_tokens: int) -> bool:
        target_len = sequence.logical_length + int(num_tokens)
        return self._backend.ensure_sequence_capacity(sequence.seq_id, target_len)

    def materialize_kv(self, *args, **kwargs):
        return self._backend.materialize_kv(*args, **kwargs)

    def stats(self):
        return self._backend.stats()


__all__ = ["MemoryManager"]