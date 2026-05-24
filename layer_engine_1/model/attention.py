import torch
import torch.nn as nn
import time
from typing import Optional

from runtime.block_table import BlockTable
from runtime.page_allocator import PageAllocator
from runtime.tracer import RuntimeTracer


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
        page_allocator: Optional[PageAllocator] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[BlockTable] = None,
        sequence_id: Optional[int] = None,
        runtime_tracer: Optional[RuntimeTracer] = None,
    ):
        """Apply causal attention with optional KV cache support."""

        batch_size, seq_len, dim = x.shape

        qkv = self.linear_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        if page_allocator is not None or slot_mapping is not None or block_table is not None:
            if page_allocator is None:
                raise ValueError("page_allocator must be provided when using slot mapping")
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when paged KV is used")

            if slot_mapping is None:
                raise ValueError("slot_mapping must be provided when paged KV is used")

            slot_mapping = slot_mapping.to(device=x.device, dtype=torch.long)
            total_seq_len = int(slot_mapping.size(1))
            if total_seq_len < seq_len:
                raise ValueError("slot_mapping length must be >= input sequence length")

            write_slots = slot_mapping[:, -seq_len:].reshape(-1)
            key_to_store = key.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
            value_to_store = value.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

            flat_cache_k = page_allocator.cache[0, layer_idx].view(
                -1, self.num_heads, self.head_dim
            )
            flat_cache_v = page_allocator.cache[1, layer_idx].view(
                -1, self.num_heads, self.head_dim
            )
            with torch.no_grad():
                flat_cache_k.index_copy_(0, write_slots, key_to_store)
                flat_cache_v.index_copy_(0, write_slots, value_to_store)

            if block_table is not None:
                if sequence_id is None:
                    raise ValueError("sequence_id must be provided when block_table is used")

                page_indices = block_table.get_block_mapping(sequence_id)
                if not page_indices:
                    raise ValueError("block_table returned no physical pages for sequence")

                page_indices = torch.tensor(page_indices, device=x.device, dtype=torch.long)
                gather_start = time.perf_counter()
                cache_k_pages = page_allocator.cache[0, layer_idx].index_select(0, page_indices)
                cache_v_pages = page_allocator.cache[1, layer_idx].index_select(0, page_indices)

                # Rebuild a temporary contiguous KV buffer from scattered pages.
                cache_k_flat = cache_k_pages.contiguous().view(-1, self.num_heads, self.head_dim)
                cache_v_flat = cache_v_pages.contiguous().view(-1, self.num_heads, self.head_dim)

                key = cache_k_flat[:total_seq_len]
                value = cache_v_flat[:total_seq_len]
                key = key.reshape(1, total_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                value = value.reshape(1, total_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                if runtime_tracer is not None:
                    runtime_tracer.record_page_gather_latency(time.perf_counter() - gather_start)
            else:
                read_slots = slot_mapping.reshape(-1)
                key = flat_cache_k.index_select(0, read_slots)
                value = flat_cache_v.index_select(0, read_slots)
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
