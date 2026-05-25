"""Construct MemoryManager instances for benchmarking and tests."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from memory.memory_backend import ContiguousMemoryBackend, MemoryBackend
from memory.memory_manager import MemoryManager
from memory.page_allocator import PageAllocator


def build_memory_manager(
    backend_kind: str,
    *,
    total_slots: int,
    page_size: int = 16,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    total_num_pages: Optional[int] = None,
) -> MemoryManager:
    """Build a memory manager with identical pool capacity for both backends.

    ``total_slots`` is the logical token capacity shared by contiguous and paged
    runs so benchmarks differ only in memory semantics, not pool size.
    """

    backend_kind = backend_kind.strip().lower()
    total_slots = int(total_slots)
    page_size = int(page_size)
    if total_slots <= 0 or page_size <= 0:
        raise ValueError("total_slots and page_size must be positive")

    if dtype is None:
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    if not isinstance(device, torch.device):
        device = torch.device(device)

    total_pages = total_num_pages
    if total_pages is None:
        total_pages = int(math.ceil(total_slots / float(page_size)))
    pooled_slots = total_pages * page_size

    if backend_kind == "contiguous":
        backend: MemoryBackend = ContiguousMemoryBackend(
            total_slots=pooled_slots,
            page_size=page_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        return MemoryManager(backend)

    if backend_kind == "paged":
        allocator = PageAllocator(
            total_num_pages=total_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        return MemoryManager(allocator)

    raise ValueError("backend_kind must be 'contiguous' or 'paged'")


__all__ = ["build_memory_manager"]
