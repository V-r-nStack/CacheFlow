import torch
import torch.nn as nn
import math
from .attention import CausalMultiHeadAttention


class NewGELUActivation(nn.Module):
    """GPT-style GELU approximation."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cdf = 0.5 * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
        ))
        return input * cdf


class FeedForward(nn.Module):
    """Position-wise MLP used inside each block."""

    def __init__(self, dim, dropout=0.0):
        """Build the 4x expansion MLP."""
        super().__init__()

        self.dim = dim
        # GPT-2 FFN width.
        self.hidden_dim = 4 * dim

        # (B, T, C) -> (B, T, 4C)
        self.linear_expand = nn.Linear(dim, self.hidden_dim)

        # Match GPT-2 activation.
        self.gelu = NewGELUActivation()

        # Training-time dropout only.
        self.dropout = nn.Dropout(dropout)

        # (B, T, 4C) -> (B, T, C)
        self.linear_contract = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        """Apply the MLP branch."""
        x = self.linear_expand(x)

        x = self.gelu(x)

        x = self.dropout(x)

        x = self.linear_contract(x)

        return x


class TransformerBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(self, dim, num_heads, dropout=0.0):
        """Initialize attention and MLP branches."""
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Pre-norm before attention.
        self.norm_attn = nn.LayerNorm(dim)

        self.attention = CausalMultiHeadAttention(dim, num_heads, dropout=dropout)

        # Pre-norm before MLP.
        self.norm_ffn = nn.LayerNorm(dim)

        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x,
        kv_cache=None,
        layer_idx=None,
        page_allocator=None,
        slot_mapping=None,
        block_table=None,
        sequence_id=None,
        runtime_tracer=None,
    ):
        """Run attention and MLP residual branches."""

        # x -> LN -> attention -> residual
        x_norm = self.norm_attn(x)
        attn_out = self.attention(
            x_norm,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            page_allocator=page_allocator,
            slot_mapping=slot_mapping,
            block_table=block_table,
            sequence_id=sequence_id,
            runtime_tracer=runtime_tracer,
        )
        x = x + attn_out

        # x -> LN -> MLP -> residual
        x_norm = self.norm_ffn(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
