import torch
import torch.nn as nn
from .blocks import TransformerBlock


class GPT(nn.Module):
    """
    GPT language model with causal self-attention.
    
    Architecture:
        - Token embeddings
        - Position embeddings (absolute positioning)
        - Stack of transformer blocks with Pre-Norm attention
        - Final layer normalization
        - Language model head for next-token prediction
    """
    
    def __init__(self, vocab_size, max_seq_len, dim, num_heads, num_layers, dropout=0.0):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length for position embeddings
            dim: Model dimension (embedding and hidden dimension)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # ===== EMBEDDINGS =====
        
        # Token embeddings: map token IDs to dense vectors
        # Shape: (vocab_size, dim)
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Position embeddings: absolute positional encoding
        # Shape: (max_seq_len, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Dropout applied to embedding sums
        self.emb_dropout = nn.Dropout(dropout)
        
        # ===== TRANSFORMER BLOCKS =====
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # ===== OUTPUT LAYER =====
        
        # Final layer normalization before language model head
        self.final_norm = nn.LayerNorm(dim)
        
        # Language model head: project back to vocabulary size for next-token logits
        # Input: (batch_size, seq_len, dim), Output: (batch_size, seq_len, vocab_size)
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, token_ids):
        """
        Forward pass of GPT model.
        
        Args:
            token_ids: Token IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        
        # Get sequence length from input
        # token_ids: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape
        
        # ===== TOKEN AND POSITION EMBEDDINGS =====
        
        # Look up token embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        token_embeds = self.token_emb(token_ids)
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # (seq_len,)
        pos_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        
        # Look up position embeddings
        # (seq_len,) -> (seq_len, dim)
        pos_embeds = self.pos_emb(pos_ids)
        
        # Add token and position embeddings
        # (batch_size, seq_len, dim) + (seq_len, dim) -> (batch_size, seq_len, dim) [broadcasting]
        x = token_embeds + pos_embeds
        
        # Apply dropout to embeddings
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x = self.emb_dropout(x)
        
        # ===== TRANSFORMER BLOCKS =====
        
        # Pass through stack of transformer blocks
        for block in self.blocks:
            # Each block: (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
            x = block(x)
        
        # ===== OUTPUT PROJECTION =====
        
        # Apply final layer normalization
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x = self.final_norm(x)
        
        # Project to vocabulary size for next-token prediction
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits
    
    @classmethod
    def load_pretrained_weights(cls, model, state_dict_path):
        """
        Load pretrained weights from HuggingFace GPT-2 state dict.
        
        Maps HuggingFace GPT-2 parameter names and shapes to this custom model's architecture.
        Handles Conv1D weight transposition (HF uses Conv1D with (in_features, out_features) layout,
        PyTorch nn.Linear expects (out_features, in_features)).
        
        HuggingFace GPT-2 structure:
            transformer.wte.weight → token embeddings
            transformer.wpe.weight → position embeddings
            transformer.h.{i}.ln_1.weight/bias → attention pre-norm
            transformer.h.{i}.attn.c_attn.weight/bias → attention Q,K,V projection (Conv1D)
            transformer.h.{i}.attn.c_proj.weight/bias → attention output projection (Conv1D)
            transformer.h.{i}.ln_2.weight/bias → FFN pre-norm
            transformer.h.{i}.mlp.c_fc.weight/bias → FFN expand (Conv1D)
            transformer.h.{i}.mlp.c_proj.weight/bias → FFN contract (Conv1D)
            transformer.ln_f.weight/bias → final layer norm
            lm_head.weight → language model head
        
        Custom model structure:
            token_emb.weight → token embeddings
            pos_emb.weight → position embeddings
            blocks.{i}.norm_attn.weight/bias → attention pre-norm
            blocks.{i}.attention.linear_qkv.weight/bias → attention Q,K,V
            blocks.{i}.attention.linear_out.weight/bias → attention output
            blocks.{i}.norm_ffn.weight/bias → FFN pre-norm
            blocks.{i}.ffn.linear_expand.weight/bias → FFN expand
            blocks.{i}.ffn.linear_contract.weight/bias → FFN contract
            final_norm.weight/bias → final layer norm
            lm_head.weight → language model head
        
        Args:
            model: GPT model instance to load weights into
            state_dict_path: Path to HuggingFace GPT-2 state_dict file (.pt)
        
        Returns:
            None (modifies model in-place)
        """
        
        # Load the pretrained state dict
        pretrained_state = torch.load(state_dict_path, map_location='cpu')
        
        # Initialize new state dict for custom model
        model_state = {}
        
        # ===== EMBEDDING WEIGHTS =====
        
        # Token embeddings: direct copy
        if 'transformer.wte.weight' in pretrained_state:
            model_state['token_emb.weight'] = pretrained_state['transformer.wte.weight']
        
        # Position embeddings: direct copy
        if 'transformer.wpe.weight' in pretrained_state:
            model_state['pos_emb.weight'] = pretrained_state['transformer.wpe.weight']
        
        # ===== TRANSFORMER BLOCKS =====
        
        for layer_idx in range(model.num_layers):
            hf_prefix = f'transformer.h.{layer_idx}'
            custom_prefix = f'blocks.{layer_idx}'
            
            # ===== ATTENTION LAYER NORM =====
            
            # Pre-norm before attention: layer norm 1
            # HF: transformer.h.{i}.ln_1.weight -> Custom: blocks.{i}.norm_attn.weight
            if f'{hf_prefix}.ln_1.weight' in pretrained_state:
                model_state[f'{custom_prefix}.norm_attn.weight'] = pretrained_state[f'{hf_prefix}.ln_1.weight']
            if f'{hf_prefix}.ln_1.bias' in pretrained_state:
                model_state[f'{custom_prefix}.norm_attn.bias'] = pretrained_state[f'{hf_prefix}.ln_1.bias']
            
            # ===== ATTENTION PROJECTIONS =====
            
            # Attention Q, K, V projection: combined into single linear
            # HF uses Conv1D: c_attn.weight shape is (in_features, out_features)
            # PyTorch nn.Linear expects: weight shape is (out_features, in_features)
            # Must transpose: (in, out).T -> (out, in)
            if f'{hf_prefix}.attn.c_attn.weight' in pretrained_state:
                # HF c_attn produces [Q, K, V] concatenated
                # Shape: (in_features, 3*out_features) -> need to transpose to (3*out_features, in_features)
                c_attn_weight = pretrained_state[f'{hf_prefix}.attn.c_attn.weight']
                # Transpose Conv1D weight to Linear weight
                model_state[f'{custom_prefix}.attention.linear_qkv.weight'] = c_attn_weight.T
            
            if f'{hf_prefix}.attn.c_attn.bias' in pretrained_state:
                model_state[f'{custom_prefix}.attention.linear_qkv.bias'] = pretrained_state[f'{hf_prefix}.attn.c_attn.bias']
            
            # Attention output projection
            # HF c_proj.weight shape is (in_features, out_features), transpose to (out_features, in_features)
            if f'{hf_prefix}.attn.c_proj.weight' in pretrained_state:
                c_proj_weight = pretrained_state[f'{hf_prefix}.attn.c_proj.weight']
                model_state[f'{custom_prefix}.attention.linear_out.weight'] = c_proj_weight.T
            
            if f'{hf_prefix}.attn.c_proj.bias' in pretrained_state:
                model_state[f'{custom_prefix}.attention.linear_out.bias'] = pretrained_state[f'{hf_prefix}.attn.c_proj.bias']
            
            # ===== FFN LAYER NORM =====
            
            # Pre-norm before feed-forward: layer norm 2
            # HF: transformer.h.{i}.ln_2.weight -> Custom: blocks.{i}.norm_ffn.weight
            if f'{hf_prefix}.ln_2.weight' in pretrained_state:
                model_state[f'{custom_prefix}.norm_ffn.weight'] = pretrained_state[f'{hf_prefix}.ln_2.weight']
            if f'{hf_prefix}.ln_2.bias' in pretrained_state:
                model_state[f'{custom_prefix}.norm_ffn.bias'] = pretrained_state[f'{hf_prefix}.ln_2.bias']
            
            # ===== FFN PROJECTIONS =====
            
            # FFN expand layer (mlp.c_fc): expand to 4x dimension
            # HF c_fc.weight shape is (in_features, out_features), transpose
            if f'{hf_prefix}.mlp.c_fc.weight' in pretrained_state:
                c_fc_weight = pretrained_state[f'{hf_prefix}.mlp.c_fc.weight']
                model_state[f'{custom_prefix}.ffn.linear_expand.weight'] = c_fc_weight.T
            
            if f'{hf_prefix}.mlp.c_fc.bias' in pretrained_state:
                model_state[f'{custom_prefix}.ffn.linear_expand.bias'] = pretrained_state[f'{hf_prefix}.mlp.c_fc.bias']
            
            # FFN contract layer (mlp.c_proj): contract back to original dimension
            # HF c_proj.weight shape is (in_features, out_features), transpose
            if f'{hf_prefix}.mlp.c_proj.weight' in pretrained_state:
                c_proj_weight = pretrained_state[f'{hf_prefix}.mlp.c_proj.weight']
                model_state[f'{custom_prefix}.ffn.linear_contract.weight'] = c_proj_weight.T
            
            if f'{hf_prefix}.mlp.c_proj.bias' in pretrained_state:
                model_state[f'{custom_prefix}.ffn.linear_contract.bias'] = pretrained_state[f'{hf_prefix}.mlp.c_proj.bias']
        
        # ===== FINAL OUTPUT LAYER =====
        
        # Final layer normalization
        if 'transformer.ln_f.weight' in pretrained_state:
            model_state['final_norm.weight'] = pretrained_state['transformer.ln_f.weight']
        if 'transformer.ln_f.bias' in pretrained_state:
            model_state['final_norm.bias'] = pretrained_state['transformer.ln_f.bias']
        
        # Language model head
        if 'lm_head.weight' in pretrained_state:
            model_state['lm_head.weight'] = pretrained_state['lm_head.weight']
        
        # Load the mapped weights into the model
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded pretrained weights from {state_dict_path}")
        print(f"Loaded {len(model_state)} parameters")
