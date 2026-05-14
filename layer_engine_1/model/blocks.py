import torch
import torch.nn as nn
from .attention import CausalMultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Implements: Linear (expand 4x) -> GELU -> Linear (project back)
    """
    
    def __init__(self, dim, dropout=0.0):
        """
        Initialize feed-forward network.
        
        Args:
            dim: Model dimension (input/output dimension)
            dropout: Dropout probability after first linear layer
        """
        super().__init__()
        
        self.dim = dim
        # Expand dimension by 4x as per transformer architecture convention
        self.hidden_dim = 4 * dim
        
        # First linear layer: project to expanded dimension
        self.linear_expand = nn.Linear(dim, self.hidden_dim)
        
        # Activation function
        self.gelu = nn.GELU()
        
        # Dropout after activation
        self.dropout = nn.Dropout(dropout)
        
        # Second linear layer: project back to original dimension
        self.linear_contract = nn.Linear(self.hidden_dim, dim)
    
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # First linear: expand to 4x dimension
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        x = self.linear_expand(x)
        
        # Apply GELU activation
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.gelu(x)
        
        # Apply dropout
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        
        # Second linear: contract back to original dimension
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, dim)
        x = self.linear_contract(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-Norm architecture.
    
    Structure:
        - Pre-Norm: LayerNorm applied before each sub-layer
        - Multi-head self-attention with causal masking
        - Position-wise feed-forward network
        - Explicit residual connections around both sub-layers
    """
    
    def __init__(self, dim, num_heads, dropout=0.0):
        """
        Initialize transformer block.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # ===== ATTENTION SUB-LAYER =====
        
        # Pre-norm layer normalization before attention
        self.norm_attn = nn.LayerNorm(dim)
        
        # Causal multi-head self-attention
        self.attention = CausalMultiHeadAttention(dim, num_heads, dropout=dropout)
        
        # ===== FEED-FORWARD SUB-LAYER =====
        
        # Pre-norm layer normalization before feed-forward
        self.norm_ffn = nn.LayerNorm(dim)
        
        # Position-wise feed-forward network
        self.ffn = FeedForward(dim, dropout=dropout)
    
    def forward(self, x):
        """
        Forward pass of transformer block with Pre-Norm architecture.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        
        # ===== ATTENTION SUB-LAYER WITH RESIDUAL CONNECTION =====
        
        # Apply layer norm before attention (Pre-Norm architecture)
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x_norm = self.norm_attn(x)
        
        # Apply causal multi-head attention
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        attn_out = self.attention(x_norm)
        
        # Add residual connection: combine input with attention output
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x = x + attn_out
        
        # ===== FEED-FORWARD SUB-LAYER WITH RESIDUAL CONNECTION =====
        
        # Apply layer norm before feed-forward (Pre-Norm architecture)
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x_norm = self.norm_ffn(x)
        
        # Apply feed-forward network
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        ffn_out = self.ffn(x_norm)
        
        # Add residual connection: combine input with feed-forward output
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x = x + ffn_out
        
        return x
