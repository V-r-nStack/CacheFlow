"""Batch preparation utilities for continuous decoding."""

from typing import Dict, List, Optional

import torch

from runtime.memory_manager import MemoryManager
from runtime.block_table import BlockTable
from runtime.sequence import Sequence


def prepare_continuous_batch(
    active_sequences: List[Sequence],
    memory_manager: MemoryManager,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Prepare flat inputs and slot mappings for a mixed prefill/decode batch.

    New sequences (generated_token_ids length == 0) are treated as prefill and
    contribute their full prompt. Existing sequences contribute a single decode
    token (the most recent generated token).

    Returns a dict with:
    - input_ids: 1D concatenated token IDs for this iteration
    - position_ids: 1D concatenated position IDs for these tokens
    - slot_mapping: 1D concatenated slot mapping per sequence
    - input_offsets: start offsets into input_ids for each sequence
    - slot_offsets: start offsets into slot_mapping for each sequence
    - slot_lengths: logical sequence lengths (prompt + generated)
    """

    if device is None:
        device = torch.device("cpu")

    input_chunks: List[torch.Tensor] = []
    position_chunks: List[torch.Tensor] = []
    slot_chunks: List[torch.Tensor] = []
    input_offsets: List[int] = []
    slot_offsets: List[int] = []
    slot_lengths: List[int] = []

    input_cursor = 0
    slot_cursor = 0

    for sequence in active_sequences:
        prompt_len = len(sequence.prompt_token_ids)
        generated_len = len(sequence.generated_token_ids)
        logical_len = prompt_len + generated_len

        if logical_len == 0:
            raise ValueError("Sequence has no prompt or generated tokens")
        # Require BlockTable for logical-to-physical mapping
        block_table: Optional[BlockTable] = getattr(memory_manager, "block_table", None)
        if block_table is None:
            raise RuntimeError("BlockTable not attached to memory_manager; scheduler must manage BlockTable")
        slot_mapping_list = block_table.get_slot_mapping(sequence.seq_id, logical_len)
        if len(slot_mapping_list) < logical_len:
            raise ValueError(
                "memory manager slot mapping must cover the logical sequence length"
            )

        if generated_len == 0:
            input_ids = torch.tensor(sequence.prompt_token_ids, dtype=torch.long)
            position_ids = torch.arange(prompt_len, dtype=torch.long)
        else:
            input_ids = torch.tensor(
                [sequence.generated_token_ids[-1]], dtype=torch.long
            )
            position_ids = torch.tensor(
                [prompt_len + generated_len - 1], dtype=torch.long
            )

        slot_mapping = torch.tensor(slot_mapping_list[:logical_len], dtype=torch.long)

        input_offsets.append(input_cursor)
        slot_offsets.append(slot_cursor)
        slot_lengths.append(logical_len)

        input_chunks.append(input_ids)
        position_chunks.append(position_ids)
        slot_chunks.append(slot_mapping)

        input_cursor += input_ids.numel()
        slot_cursor += slot_mapping.numel()

    input_ids_flat = torch.cat(input_chunks).to(device=device)
    position_ids_flat = torch.cat(position_chunks).to(device=device)
    slot_mapping_flat = torch.cat(slot_chunks).to(device=device)

    return {
        "input_ids": input_ids_flat,
        "position_ids": position_ids_flat,
        "slot_mapping": slot_mapping_flat,
        "input_offsets": torch.tensor(input_offsets, dtype=torch.long, device=device),
        "slot_offsets": torch.tensor(slot_offsets, dtype=torch.long, device=device),
        "slot_lengths": torch.tensor(slot_lengths, dtype=torch.long, device=device),
    }


__all__ = ["prepare_continuous_batch"]