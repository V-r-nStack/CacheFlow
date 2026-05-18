import torch
import torch.nn as nn


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

    def forward(self, x, kv_cache=None, layer_idx=None):
        """Apply causal attention, with optional KV cache support."""

        batch_size, seq_len, dim = x.shape

        qkv = self.linear_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
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

        if query.size(2) > 1:
            causal_mask = torch.tril(
                torch.ones(query.size(2), key.size(2), device=x.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.linear_out(out)

        return out
