"""KV cache runtime abstraction for stateful GPT inference.

Provides a per-layer KV storage container (`KVCache`) and an
orchestrator (`KVCacheManager`) to create, clear and inspect
the memory footprint of cached key/value tensors.

This module intentionally keeps state separated by layer and
does not use any global variables.
"""
from typing import List, Optional, Dict

import torch


class KVCache:
    """Per-layer key/value cache container.

    Attributes
    - n_layers: number of transformer layers
    - layers: list of dicts, each with keys 'k' and 'v' storing tensors
    """

    def __init__(self, n_layers: int):
        self.n_layers = int(n_layers)
        # Each element is a dict: {'k': Optional[Tensor], 'v': Optional[Tensor]}
        self.layers: List[Dict[str, Optional[torch.Tensor]]] = [
            {"k": None, "v": None} for _ in range(self.n_layers)
        ]

    def set_layer(self, layer_idx: int, k: Optional[torch.Tensor], v: Optional[torch.Tensor]):
        """Set the key and value tensors for a specific layer.

        Parameters
        - layer_idx: 0-based layer index
        - k, v: torch.Tensor or None
        """
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError("layer_idx out of range")
        self.layers[layer_idx]["k"] = k
        self.layers[layer_idx]["v"] = v

    def get_layer(self, layer_idx: int):
        """Return a tuple (k, v) for the requested layer."""
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError("layer_idx out of range")
        entry = self.layers[layer_idx]
        return entry.get("k"), entry.get("v")


class KVCacheManager:
    """Orchestrator for KVCache lifecycle and utilities."""

    @staticmethod
    def init_cache(n_layers: int) -> KVCache:
        """Create and return a fresh KVCache instance."""
        return KVCache(n_layers=n_layers)

    @staticmethod
    def clear_cache(cache_obj: KVCache):
        """Clear and delete all stored tensors in `cache_obj`.

        This method explicitly removes references to the tensors and
        attempts to free CUDA memory if applicable.
        """
        if not isinstance(cache_obj, KVCache):
            raise TypeError("cache_obj must be a KVCache instance")

        for i, layer in enumerate(cache_obj.layers):
            for key in ("k", "v"):
                tensor = layer.get(key)
                if tensor is None:
                    continue

                try:
                    # Move to CPU and detach to break autograd graph, then delete
                    if tensor.device.type != "cpu":
                        tensor = tensor.detach().cpu()
                    else:
                        tensor = tensor.detach()
                except Exception:
                    # If any operation fails, still remove the reference
                    pass

                # Remove reference from cache
                layer[key] = None

        # Try to release CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def get_memory_footprint_mb(cache_obj: KVCache) -> float:
        """Calculate total memory used by all cached tensors in megabytes.

        Iterates all per-layer `k` and `v` tensors, sums `numel() * element_size()`
        and returns the value in MiB.
        """
        if not isinstance(cache_obj, KVCache):
            raise TypeError("cache_obj must be a KVCache instance")

        total_bytes = 0
        for layer in cache_obj.layers:
            for key in ("k", "v"):
                tensor = layer.get(key)
                if tensor is None:
                    continue
                # Use tensor.numel() and tensor.element_size() for accuracy
                try:
                    total_bytes += int(tensor.numel()) * int(tensor.element_size())
                except Exception:
                    # Fallback: attempt to estimate via dtype
                    dtype_size = 4  # assume float32
                    total_bytes += int(tensor.numel()) * dtype_size

        mib = total_bytes / (1024.0 * 1024.0)
        return mib


__all__ = ["KVCache", "KVCacheManager"]
