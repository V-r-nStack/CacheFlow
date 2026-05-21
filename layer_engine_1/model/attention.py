import torch
import torch.nn as nn
from typing import Optional

from runtime.memory_manager import MemoryManager
from runtime.static_kv_cache import StaticKVCache


class CausalMultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, dim, num_heads, dropout=0.0):
        """Initialize the attention projection stack."""
        super().__init__()

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection: (B, T, C) -> (B, T, 3C)
        self.linear_qkv = nn.Linear(dim, 3 * dim)

        # Output projection.
        self.linear_out = nn.Linear(dim, dim)

        # Attention weight dropout.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        kv_cache=None,
        layer_idx=None,
        static_kv_cache: Optional[StaticKVCache] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        memory_manager: Optional[MemoryManager] = None,
        sequence_id: Optional[int] = None,
    ):
        """Apply causal attention with optional KV cache support."""

        batch_size, seq_len, dim = x.shape

        qkv = self.linear_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        if static_kv_cache is not None or slot_mapping is not None or memory_manager is not None:
            if static_kv_cache is None:
                raise ValueError("static_kv_cache must be provided when using slot mapping")
            if memory_manager is not None and sequence_id is not None and batch_size == 1:
                mapping_list = memory_manager.get_mapping_by_id(sequence_id)
                slot_mapping = torch.tensor(
                    mapping_list, dtype=torch.long, device=x.device
                ).unsqueeze(0)
            elif slot_mapping is None:
                raise ValueError("slot_mapping must be provided without memory_manager mapping")
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when static_kv_cache is used")

            slot_mapping = slot_mapping.to(device=x.device, dtype=torch.long)
            total_seq_len = int(slot_mapping.size(1))
            if total_seq_len < seq_len:
                raise ValueError("slot_mapping length must be >= input sequence length")

            write_slots = slot_mapping[:, -seq_len:].reshape(-1)
            key_to_store = key.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
            value_to_store = value.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

            cache_k = static_kv_cache.cache[0, layer_idx]
            cache_v = static_kv_cache.cache[1, layer_idx]
            with torch.no_grad():
                cache_k.index_copy_(0, write_slots, key_to_store)
                cache_v.index_copy_(0, write_slots, value_to_store)

            read_slots = slot_mapping.reshape(-1)
            key = cache_k.index_select(0, read_slots)
            value = cache_v.index_select(0, read_slots)
            key = key.reshape(batch_size, total_seq_len, self.num_heads, self.head_dim)
            value = value.reshape(batch_size, total_seq_len, self.num_heads, self.head_dim)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            if key.dtype != query.dtype:
                key = key.to(query.dtype)
            if value.dtype != query.dtype:
                value = value.to(query.dtype)
        elif kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when kv_cache is used")

            cached_key, cached_value = kv_cache.get_layer(layer_idx)

            if cached_key is not None and cached_value is not None:
                key = torch.cat([cached_key, key], dim=2)
                value = torch.cat([cached_value, value], dim=2)

            kv_cache.set_layer(layer_idx, key, value)
        else:
            cached_key = None
            cached_value = None

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        query_len = query.size(2)
        key_len = key.size(2)
        if key_len < query_len:
            raise ValueError("key sequence length cannot be shorter than query length")

        if key_len > 1:
            if key_len == query_len:
                causal_mask = torch.tril(
                    torch.ones(query_len, key_len, device=x.device, dtype=torch.bool)
                )
            else:
                key_positions = torch.arange(key_len, device=x.device)
                query_positions = torch.arange(query_len, device=x.device) + (key_len - query_len)
                causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.linear_out(out)

        return out
