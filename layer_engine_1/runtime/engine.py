"""Continuous batching event loop for token-by-token decoding."""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from runtime.batching import prepare_continuous_batch
from runtime.scheduler import Scheduler
from runtime.sequence import Sequence, SequenceStatus
from runtime.static_kv_cache import StaticKVCache


def _sample_next_token(
    logits: torch.Tensor,
    prior_token_ids: List[int],
    temperature: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> int:
    temperature = max(float(temperature), 1e-8)
    scaled_logits = logits / temperature

    if repetition_penalty not in (None, 1.0) and prior_token_ids:
        repeated = torch.unique(torch.tensor(prior_token_ids, device=logits.device))
        if repeated.numel() > 0:
            adjusted = scaled_logits.clone()
            repeated_logits = adjusted.index_select(dim=-1, index=repeated)
            repeated_logits = torch.where(
                repeated_logits < 0,
                repeated_logits * repetition_penalty,
                repeated_logits / repetition_penalty,
            )
            adjusted.index_copy_(dim=-1, index=repeated, source=repeated_logits)
            scaled_logits = adjusted

    if top_k is not None and top_k > 0 and top_k < scaled_logits.size(-1):
        values, _ = torch.topk(scaled_logits, top_k, dim=-1)
        cutoff = values[..., -1, None]
        scaled_logits = scaled_logits.masked_fill(
            scaled_logits < cutoff, torch.finfo(scaled_logits.dtype).min
        )

    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def _assign_prompt_slots(
    sequence: Sequence,
    static_kv_cache: StaticKVCache,
    prompt_len: int,
) -> None:
    while len(sequence.kv_slot_indices) < prompt_len:
        sequence.append_kv_slot_index(static_kv_cache.allocate_slot())


def run_engine(
    model,
    scheduler: Scheduler,
    static_kv_cache: StaticKVCache,
    eos_token_id: int = 50256,
    max_seq_len: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    policy: str = "fcfs",
) -> None:
    """Run the continuous decoding loop until all work is complete."""

    if max_seq_len is None and hasattr(model, "max_seq_len"):
        max_seq_len = int(model.max_seq_len)

    device = next(model.parameters()).device

    while scheduler.waiting_queue or scheduler.active_batch:
        scheduler.step_eviction(static_kv_cache)
        scheduler.schedule_next_iteration(policy=policy)

        if not scheduler.active_batch:
            continue

        for sequence in scheduler.active_batch:
            if sequence.status == SequenceStatus.WAITING:
                sequence.status = SequenceStatus.RUNNING

        for sequence in scheduler.active_batch:
            prompt_len = len(sequence.prompt_token_ids)
            generated_len = len(sequence.generated_token_ids)
            logical_len = prompt_len + generated_len

            if logical_len == 0:
                continue

            if generated_len == 0:
                _assign_prompt_slots(sequence, static_kv_cache, prompt_len)
                input_ids = torch.tensor(
                    sequence.prompt_token_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
                slot_mapping = torch.tensor(
                    sequence.kv_slot_indices[:prompt_len],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
            else:
                if len(sequence.kv_slot_indices) < logical_len:
                    raise ValueError("kv_slot_indices length is behind logical length")

                input_ids = torch.tensor(
                    [sequence.generated_token_ids[-1]], dtype=torch.long, device=device
                ).unsqueeze(0)
                slot_mapping = torch.tensor(
                    sequence.kv_slot_indices[:logical_len],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

            start_time = time.perf_counter()
            with torch.inference_mode():
                _ = prepare_continuous_batch([sequence], device=device)
                logits = model(
                    input_ids,
                    static_kv_cache=static_kv_cache,
                    slot_mapping=slot_mapping,
                )
            elapsed = time.perf_counter() - start_time

            if generated_len == 0 and sequence.ttft_s is None:
                sequence.ttft_s = elapsed

            last_logits = logits[:, -1, :]
            prior_tokens = sequence.prompt_token_ids + sequence.generated_token_ids
            next_token_id = _sample_next_token(
                last_logits,
                prior_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            sequence.generated_token_ids.append(next_token_id)
            sequence.append_kv_slot_index(static_kv_cache.allocate_slot())

            if max_seq_len is not None and sequence.logical_length >= max_seq_len:
                sequence.status = SequenceStatus.FINISHED
                sequence.finish_time = time.time()
            elif next_token_id == eos_token_id:
                sequence.status = SequenceStatus.FINISHED
                sequence.finish_time = time.time()


__all__ = ["run_engine"]