from __future__ import annotations

from pathlib import Path

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

        # GPT-2 ties the head to the token embeddings and keeps it bias-free.
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        """Keep the token embedding and output head shared."""

        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, kv_cache=None):
        """Return per-token logits for the full context."""

        batch_size, seq_len = idx.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context window {self.max_seq_len}"
            )

        token_embeds = self.token_emb(idx)

        pos_ids = torch.arange(
            seq_len,
            device=idx.device,
            dtype=torch.long,
        )
        pos_embeds = self.pos_emb(pos_ids)
        x = token_embeds + pos_embeds
        x = self.emb_dropout(x)

        for layer_idx, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=layer_idx)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def load_pretrained_weights(cls, model, state_dict_path):
        """Map a GPT-2 checkpoint into this module layout."""

        state_path = Path(state_dict_path)
        pretrained_state = torch.load(state_path, map_location='cpu')
        if not isinstance(pretrained_state, dict):
            raise TypeError(f"Expected a state dict at {state_path}, got {type(pretrained_state)!r}")

        has_transformer_prefix = any(key.startswith('transformer.') for key in pretrained_state)
        source_prefix = 'transformer.' if has_transformer_prefix else ''
        block_prefix = f'{source_prefix}h.'

        expected_state = model.state_dict()
        model_state = {}
        mapped_keys = []

        def checkpoint_key(name):
            key = f'{source_prefix}{name}'
            return key if key in pretrained_state else None

        def add_tensor(model_key, checkpoint_name, transpose=False):
            source_key = checkpoint_key(checkpoint_name)
            if source_key is None:
                return None

            tensor = pretrained_state[source_key]
            if transpose:
                tensor = tensor.T

            expected_tensor = expected_state[model_key]
            if tensor.shape != expected_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {model_key}: expected {tuple(expected_tensor.shape)}, got {tuple(tensor.shape)} from {source_key}"
                )

            model_state[model_key] = tensor
            mapped_keys.append((model_key, source_key, tuple(tensor.shape)))
            return tensor

        add_tensor('token_emb.weight', 'wte.weight')
        add_tensor('pos_emb.weight', 'wpe.weight')

        for layer_idx in range(model.num_layers):
            hf_prefix = f'{block_prefix}{layer_idx}'
            custom_prefix = f'blocks.{layer_idx}'

            add_tensor(f'{custom_prefix}.norm_attn.weight', f'{hf_prefix}.ln_1.weight')
            add_tensor(f'{custom_prefix}.norm_attn.bias', f'{hf_prefix}.ln_1.bias')
            add_tensor(f'{custom_prefix}.attention.linear_qkv.weight', f'{hf_prefix}.attn.c_attn.weight', transpose=True)
            add_tensor(f'{custom_prefix}.attention.linear_qkv.bias', f'{hf_prefix}.attn.c_attn.bias')
            add_tensor(f'{custom_prefix}.attention.linear_out.weight', f'{hf_prefix}.attn.c_proj.weight', transpose=True)
            add_tensor(f'{custom_prefix}.attention.linear_out.bias', f'{hf_prefix}.attn.c_proj.bias')
            add_tensor(f'{custom_prefix}.norm_ffn.weight', f'{hf_prefix}.ln_2.weight')
            add_tensor(f'{custom_prefix}.norm_ffn.bias', f'{hf_prefix}.ln_2.bias')
            add_tensor(f'{custom_prefix}.ffn.linear_expand.weight', f'{hf_prefix}.mlp.c_fc.weight', transpose=True)
            add_tensor(f'{custom_prefix}.ffn.linear_expand.bias', f'{hf_prefix}.mlp.c_fc.bias')
            add_tensor(f'{custom_prefix}.ffn.linear_contract.weight', f'{hf_prefix}.mlp.c_proj.weight', transpose=True)
            add_tensor(f'{custom_prefix}.ffn.linear_contract.bias', f'{hf_prefix}.mlp.c_proj.bias')

        add_tensor('final_norm.weight', 'ln_f.weight')
        add_tensor('final_norm.bias', 'ln_f.bias')

        if checkpoint_key('lm_head.weight') is not None:
            add_tensor('lm_head.weight', 'lm_head.weight')
        else:
            model_state['lm_head.weight'] = model_state['token_emb.weight']

        load_result = model.load_state_dict(model_state, strict=False)
        model.tie_weights()

        loaded_tensors = len(mapped_keys)
        loaded_parameters = sum(tensor.numel() for tensor in model_state.values())

        print(f"Loaded pretrained weights from {state_path}")
        print(f"Loaded tensor count: {loaded_tensors}")
        print(f"Loaded parameter elements: {loaded_parameters}")
        print(f"Missing keys: {load_result.missing_keys if load_result.missing_keys else []}")
        print(f"Unexpected keys: {load_result.unexpected_keys if load_result.unexpected_keys else []}")

        if mapped_keys:
            print("Loaded mapping summary:")
            for model_key, source_key, shape in mapped_keys[:8]:
                print(f"  {source_key} -> {model_key} {shape}")

        print("Tied lm_head.weight to token_emb.weight")
