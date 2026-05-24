"""Paged KV cache allocator with fixed physical pages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

import torch


@dataclass(frozen=True)
class PhysicalPage:
    """Handle for a physical page index in the KV pool."""

    index: int


class PageAllocator:
    """Manage a fixed pool of KV cache pages."""

    def __init__(
        self,
        total_num_pages: int,
        page_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        total_num_pages = int(total_num_pages)
        page_size = int(page_size)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        head_dim = int(head_dim)

        if total_num_pages <= 0 or page_size <= 0:
            raise ValueError("total_num_pages and page_size must be positive")
        if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16

        self.total_num_pages = total_num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.total_slots = total_num_pages * page_size
        self.cache = torch.zeros(
            (2, num_layers, total_num_pages, page_size, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        self._free_pages: Deque[int] = deque(range(total_num_pages))

    def allocate_pages(self, num_pages: int) -> List[int]:
        """Allocate a list of page indices from the free pool."""

        num_pages = int(num_pages)
        if num_pages <= 0:
            return []
        if num_pages > len(self._free_pages):
            raise RuntimeError("No KV pages available")
        return [int(self._free_pages.popleft()) for _ in range(num_pages)]

    def free_pages(self, page_indices: Iterable[int]) -> None:
        """Return page indices back to the free pool."""

        for page_index in page_indices:
            self._free_pages.append(int(page_index))

    def free_pages_count(self) -> int:
        """Return the number of available pages."""

        return len(self._free_pages)


__all__ = ["PageAllocator", "PhysicalPage"]
