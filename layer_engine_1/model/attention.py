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
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # QKV projection: (B, T, C) -> (B, T, 3C)
        self.linear_qkv = nn.Linear(dim, 3 * dim)
        
        # Output projection.
        self.linear_out = nn.Linear(dim, dim)
        
        # Attention weight dropout.
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Apply masked attention over the full sequence."""
        batch_size, seq_len, dim = x.shape
        
        # (B, T, C) -> (B, T, 3C)
        qkv = self.linear_qkv(x)

        # Split heads: (B, T, 3C) -> (B, T, 3, H, D)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # (B, T, 3, H, D) -> (3, B, H, T, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Each: (B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, H, T, D) x (B, H, D, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Scale by 1/sqrt(d_k).
        scores = scores * self.scale

        # Causal mask prevents future-token leakage.
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # Mask upper triangle before softmax.
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Softmax over keys.
        attn_weights = torch.softmax(scores, dim=-1)

        # Training-time only.
        attn_weights = self.dropout(attn_weights)

        # (B, H, T, T) x (B, H, T, D) -> (B, H, T, D)
        out = torch.matmul(attn_weights, v)

        # (B, H, T, D) -> (B, T, H, D)
        out = out.transpose(1, 2)

        # view() requires contiguous layout.
        out = out.contiguous()

        # (B, T, H, D) -> (B, T, C)
        out = out.view(batch_size, seq_len, dim)

        # Output projection.
        out = self.linear_out(out)
        
        return out
