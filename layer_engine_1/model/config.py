from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GPTConfig:
    vocab_size: int = 50257
    context_len: int = 1024
    embed_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
