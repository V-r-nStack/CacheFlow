import torch
import torch.nn as nn


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    
    Implements scaled dot-product attention with:
    - Single linear layer for Q, K, V projections
    - Manual head splitting using .view() and .transpose()
    - Explicit causal masking using torch.tril
    """
    
    def __init__(self, dim, num_heads, dropout=0.0):
        """
        Initialize the multi-head attention layer.
        
        Args:
            dim: Embedding/model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        # Single linear projection for all Q, K, V
        # Input: dim, Output: 3*dim (concatenated Q, K, V)
        self.linear_qkv = nn.Linear(dim, 3 * dim)
        
        # Output projection after attention
        self.linear_out = nn.Linear(dim, dim)
        
        # Dropout applied to attention weights
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of causal multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # Get input dimensions
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        
        # ===== PROJECT TO Q, K, V =====
        
        # Single linear layer produces concatenated Q, K, V
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 3*dim)
        qkv = self.linear_qkv(x)
        
        # Reshape to separate Q, K, V and split into heads
        # (batch_size, seq_len, 3*dim) -> (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        # Rearrange dimensions for easier head processing
        # (batch_size, seq_len, 3, num_heads, head_dim) -> (3, batch_size, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Unpack Q, K, V (each has shape: batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # ===== COMPUTE ATTENTION SCORES =====
        
        # Matrix multiplication: Q @ K^T
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale scores by sqrt(head_dim)
        # (batch_size, num_heads, seq_len, seq_len) * scale
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = scores * self.scale
        
        # ===== APPLY CAUSAL MASK =====
        
        # Create lower triangular causal mask (allows attending to past and current)
        # torch.tril creates lower triangular matrix with 1s where allowed
        # (seq_len, seq_len) with dtype=bool
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        
        # Apply mask: set scores at masked (future) positions to -inf
        # masked_fill(~causal_mask, ...) means "where mask is False (upper triangle), fill with -inf"
        # (batch_size, num_heads, seq_len, seq_len)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # ===== COMPUTE ATTENTION WEIGHTS =====
        
        # Apply softmax across the key dimension (last dimension)
        # Softmax will convert -inf to 0 automatically
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # ===== APPLY ATTENTION TO VALUES =====
        
        # Weighted sum of values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # ===== CONCATENATE HEADS =====
        
        # Transpose to get heads adjacent for view() operation
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        out = out.transpose(1, 2)
        
        # Ensure tensor is contiguous in memory before view()
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        out = out.contiguous()
        
        # Concatenate all heads by flattening last two dimensions
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, dim)
        out = out.view(batch_size, seq_len, dim)
        
        # ===== OUTPUT PROJECTION =====
        
        # Final linear transformation
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        out = self.linear_out(out)
        
        return out
