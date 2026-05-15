import torch
import torch.nn as nn

from .blocks import TransformerBlock


class GPT(nn.Module):
    """Decoder-only GPT-2 backbone."""

    def __init__(self, vocab_size, max_seq_len, dim, num_heads, num_layers, dropout=0.0):
        """Build embeddings, blocks, and output head."""
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token embeddings: (V, C)
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Absolute positions: (T, C)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.emb_dropout = nn.Dropout(dropout)

        # Decoder stack.
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # Final norm before logits.
        self.final_norm = nn.LayerNorm(dim)

        # (B, T, C) -> (B, T, V)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids):
        """Return per-token logits for the full context."""
        batch_size, seq_len = token_ids.shape

        # Token lookup.
        token_embeds = self.token_emb(token_ids)

        # Position ids are broadcast across batch.
        pos_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids)

        # (B, T, C) + (T, C) -> (B, T, C)
        x = token_embeds + pos_embeds
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def load_pretrained_weights(cls, model, state_dict_path):
        """Map a GPT-2 state dict into this module layout."""
        pretrained_state = torch.load(state_dict_path, map_location='cpu')
        has_transformer_prefix = any(k.startswith('transformer.') for k in pretrained_state.keys())

        def resolve_key(base_key):
            key = f'transformer.{base_key}' if has_transformer_prefix else base_key
            return key if key in pretrained_state else None

        model_state = {}

        wte_key = resolve_key('wte.weight')
        if wte_key:
            model_state['token_emb.weight'] = pretrained_state[wte_key]

        wpe_key = resolve_key('wpe.weight')
        if wpe_key:
            model_state['pos_emb.weight'] = pretrained_state[wpe_key]

        block_prefix = 'transformer.h.' if has_transformer_prefix else 'h.'

        for layer_idx in range(model.num_layers):
            hf_prefix = f'{block_prefix}{layer_idx}'
            custom_prefix = f'blocks.{layer_idx}'

            ln1_w_key = f'{hf_prefix}.ln_1.weight'
            ln1_b_key = f'{hf_prefix}.ln_1.bias'
            if ln1_w_key in pretrained_state:
                model_state[f'{custom_prefix}.norm_attn.weight'] = pretrained_state[ln1_w_key]
            if ln1_b_key in pretrained_state:
                model_state[f'{custom_prefix}.norm_attn.bias'] = pretrained_state[ln1_b_key]

            c_attn_w_key = f'{hf_prefix}.attn.c_attn.weight'
            c_attn_b_key = f'{hf_prefix}.attn.c_attn.bias'
            if c_attn_w_key in pretrained_state:
                # Conv1D -> Linear: transpose weight layout.
                model_state[f'{custom_prefix}.attention.linear_qkv.weight'] = pretrained_state[c_attn_w_key].T
            if c_attn_b_key in pretrained_state:
                model_state[f'{custom_prefix}.attention.linear_qkv.bias'] = pretrained_state[c_attn_b_key]

            c_proj_w_key = f'{hf_prefix}.attn.c_proj.weight'
            c_proj_b_key = f'{hf_prefix}.attn.c_proj.bias'
            if c_proj_w_key in pretrained_state:
                # Conv1D -> Linear: transpose weight layout.
                model_state[f'{custom_prefix}.attention.linear_out.weight'] = pretrained_state[c_proj_w_key].T
            if c_proj_b_key in pretrained_state:
                model_state[f'{custom_prefix}.attention.linear_out.bias'] = pretrained_state[c_proj_b_key]

            ln2_w_key = f'{hf_prefix}.ln_2.weight'
            ln2_b_key = f'{hf_prefix}.ln_2.bias'
            if ln2_w_key in pretrained_state:
                model_state[f'{custom_prefix}.norm_ffn.weight'] = pretrained_state[ln2_w_key]
            if ln2_b_key in pretrained_state:
                model_state[f'{custom_prefix}.norm_ffn.bias'] = pretrained_state[ln2_b_key]

            c_fc_w_key = f'{hf_prefix}.mlp.c_fc.weight'
            c_fc_b_key = f'{hf_prefix}.mlp.c_fc.bias'
            if c_fc_w_key in pretrained_state:
                # Conv1D -> Linear: transpose weight layout.
                model_state[f'{custom_prefix}.ffn.linear_expand.weight'] = pretrained_state[c_fc_w_key].T
            if c_fc_b_key in pretrained_state:
                model_state[f'{custom_prefix}.ffn.linear_expand.bias'] = pretrained_state[c_fc_b_key]

            mlp_proj_w_key = f'{hf_prefix}.mlp.c_proj.weight'
            mlp_proj_b_key = f'{hf_prefix}.mlp.c_proj.bias'
            if mlp_proj_w_key in pretrained_state:
                # Conv1D -> Linear: transpose weight layout.
                model_state[f'{custom_prefix}.ffn.linear_contract.weight'] = pretrained_state[mlp_proj_w_key].T
            if mlp_proj_b_key in pretrained_state:
                model_state[f'{custom_prefix}.ffn.linear_contract.bias'] = pretrained_state[mlp_proj_b_key]

        ln_f_w_key = resolve_key('ln_f.weight')
        ln_f_b_key = resolve_key('ln_f.bias')
        if ln_f_w_key:
            model_state['final_norm.weight'] = pretrained_state[ln_f_w_key]
        if ln_f_b_key:
            model_state['final_norm.bias'] = pretrained_state[ln_f_b_key]

        lm_head_key = resolve_key('lm_head.weight')
        if lm_head_key:
            model_state['lm_head.weight'] = pretrained_state[lm_head_key]
        elif wte_key:
            model_state['lm_head.weight'] = pretrained_state[wte_key]

        model.load_state_dict(model_state, strict=False)
        model.lm_head.weight = model.token_emb.weight

        print(f"Loaded pretrained weights from {state_dict_path}")
        print(f"Loaded {len(model_state)} parameters")
        print("Tied lm_head.weight to token_emb.weight (weight tying)")
