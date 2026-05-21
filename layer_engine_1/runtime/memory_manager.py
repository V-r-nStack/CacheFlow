"""Memory manager for logical-to-physical KV slot mapping."""

from __future__ import annotations

from typing import Dict, List

from runtime.sequence import Sequence
from runtime.static_kv_cache import StaticKVCache


class MemoryManager:
    """Manage logical blocks per sequence and map them to physical KV slots."""

    def __init__(self, static_kv_cache: StaticKVCache):
        if not isinstance(static_kv_cache, StaticKVCache):
            raise TypeError("static_kv_cache must be a StaticKVCache instance")
        self._cache = static_kv_cache
        self._mapping: Dict[int, List[int]] = {}

    @property
    def static_kv_cache(self) -> StaticKVCache:
        return self._cache

    def free_slots_count(self) -> int:
        return self._cache.free_slots_count()

    def get_mapping(self, sequence: Sequence) -> List[int]:
        return list(self._mapping.get(sequence.seq_id, []))

    def get_mapping_by_id(self, seq_id: int) -> List[int]:
        return list(self._mapping.get(int(seq_id), []))

    def ensure_mapping_length(self, sequence: Sequence, target_len: int) -> bool:
        target_len = int(target_len)
        if target_len <= 0:
            return True

        mapping = self._mapping.setdefault(sequence.seq_id, [])
        while len(mapping) < target_len:
            try:
                mapping.append(self._cache.allocate_slot())
            except RuntimeError:
                return False
        return True

    def allocate_for_sequence(self, sequence: Sequence, count: int) -> bool:
        count = int(count)
        if count <= 0:
            return True

        mapping = self._mapping.setdefault(sequence.seq_id, [])
        for _ in range(count):
            try:
                mapping.append(self._cache.allocate_slot())
            except RuntimeError:
                return False
        return True

    def release_sequence(self, sequence: Sequence) -> None:
        mapping = self._mapping.pop(sequence.seq_id, [])
        if mapping:
            self._cache.return_slots(mapping)


__all__ = ["MemoryManager"]