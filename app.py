import torch
import argparse
from pathlib import Path

from runtime.gpt import GPT
from memory.kv_cache import KVCacheManager
from tracing.profiler import ExecutionTimer, decode_throughput_tokens_per_second, format_mb
from runtime.tokenizer import GPT2Tokenizer


def _apply_repetition_penalty(logits, generated_token_ids, penalty):
    if penalty is None or penalty == 1.0:
        return logits

    repeated_token_ids = torch.unique(generated_token_ids.view(-1))
    if repeated_token_ids.numel() == 0:
        return logits

    adjusted_logits = logits.clone()
    repeated_logits = adjusted_logits.index_select(dim=-1, index=repeated_token_ids)
    repeated_logits = torch.where(repeated_logits < 0, repeated_logits * penalty, repeated_logits / penalty)
    adjusted_logits.index_copy_(dim=-1, index=repeated_token_ids, source=repeated_logits)
    return adjusted_logits


def _top_k_filter(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits

    values, _ = torch.topk(logits, top_k, dim=-1)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)


def sample_next_token(
    logits,
    generated_token_ids,
    temperature=1.0,
    top_k=50,
    repetition_penalty=1.0,
):
    temperature = max(float(temperature), 1e-8)
    logits = logits / temperature
    logits = _apply_repetition_penalty(logits, generated_token_ids, repetition_penalty)
    logits = _top_k_filter(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _build_token_tensor(token_ids, device):
    return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)


def _print_decode_step(step_idx, decode_latency_s, cache_footprint_mb=None):
    if cache_footprint_mb is None:
        print(f"Decode step {step_idx:03d} | Decode Latency (ITL): {decode_latency_s:.6f} seconds")
    else:
        print(
            f"Decode step {step_idx:03d} | Decode Latency (ITL): {decode_latency_s:.6f} seconds | "
            f"KV Cache: {format_mb(cache_footprint_mb)}"
        )


def _print_decode_summary(itl_times):
    if itl_times:
        average_itl = sum(itl_times) / len(itl_times)
        throughput = decode_throughput_tokens_per_second(len(itl_times), itl_times)
        print(f"Decode Latency (ITL) Average: {average_itl:.6f} seconds")
        if throughput is not None:
            print(f"Decode Throughput: {throughput:.2f} tokens/second")
        else:
            print("Decode Throughput: N/A")
    else:
        print("Decode Latency (ITL) Average: N/A (no decode tokens generated)")
        print("Decode Throughput: N/A")


def _print_itl_benchmark_table(baseline_itl_times, cached_itl_times):
    max_steps = max(len(baseline_itl_times), len(cached_itl_times))
    print("\n=== Decode ITL Scaling Table (ms) ===")
    print("Step | No Cache | KV Cache | Speedup")
    print("-----|----------|----------|--------")

    for step_idx in range(max_steps):
        no_cache_itl_ms = baseline_itl_times[step_idx] * 1000.0 if step_idx < len(baseline_itl_times) else None
        cache_itl_ms = cached_itl_times[step_idx] * 1000.0 if step_idx < len(cached_itl_times) else None

        if no_cache_itl_ms is None or cache_itl_ms is None or cache_itl_ms <= 0:
            speedup_text = "N/A"
            no_cache_text = "   N/A"
            cache_text = "   N/A"
        else:
            speedup_text = f"{no_cache_itl_ms / cache_itl_ms:.2f}x"
            no_cache_text = f"{no_cache_itl_ms:8.3f}"
            cache_text = f"{cache_itl_ms:8.3f}"

        print(
            f"{step_idx + 1:4d} | "
            f"{no_cache_text} | "
            f"{cache_text} | "
            f"{speedup_text:>7}"
        )


def _format_decode_throughput(itl_times):
    throughput = decode_throughput_tokens_per_second(len(itl_times), itl_times)
    return f"{throughput:.2f} tokens/s" if throughput is not None else "N/A"


def _run_generation_once(
    model,
    tokenizer,
    prompt,
    max_tokens,
    device,
    temperature,
    top_k,
    repetition_penalty,
    use_cache=True,
):
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_token_tensor = _build_token_tensor(prompt_token_ids, device)
    if prompt_token_tensor.size(1) > model.max_seq_len:
        prompt_token_tensor = prompt_token_tensor[:, -model.max_seq_len:]

    sequence_token_ids = prompt_token_tensor[0].tolist()
    kv_cache = KVCacheManager.init_cache(model.num_layers) if use_cache else None

    print(f"Prompt: {prompt}")
    print(f"Encoded to {len(prompt_token_ids)} tokens")
    print(f"Generating up to {max_tokens} tokens...\n")

    model.eval()

    ttft = None
    itl_times = []

    with torch.inference_mode():
        with ExecutionTimer() as timer:
            if use_cache:
                logits = model(prompt_token_tensor, kv_cache=kv_cache)
            else:
                logits = model(prompt_token_tensor)

        ttft = timer.elapsed
        print(f"Prefill Latency (TTFT): {ttft:.6f} seconds")

        last_token_logits = logits[:, -1, :]
        latest_token = sample_next_token(
            last_token_logits,
            prompt_token_tensor,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        sequence_token_ids.append(latest_token.item())

        if latest_token.item() == 50256:
            print("Generated [END] token")
        else:
            for step_idx in range(1, max_tokens):
                if use_cache:
                    decode_input = latest_token
                else:
                    decode_input = _build_token_tensor(sequence_token_ids, device)
                    if decode_input.size(1) > model.max_seq_len:
                        decode_input = decode_input[:, -model.max_seq_len:]

                with ExecutionTimer() as timer:
                    if use_cache:
                        logits = model(decode_input, kv_cache=kv_cache)
                    else:
                        logits = model(decode_input)

                decode_latency_s = timer.elapsed
                itl_times.append(decode_latency_s)

                cache_footprint_mb = None
                if use_cache and kv_cache is not None:
                    cache_footprint_mb = KVCacheManager.get_memory_footprint_mb(kv_cache)

                _print_decode_step(step_idx, decode_latency_s, cache_footprint_mb)

                last_token_logits = logits[:, -1, :]
                generated_token_tensor = _build_token_tensor(sequence_token_ids, device)
                latest_token = sample_next_token(
                    last_token_logits,
                    generated_token_tensor,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
                sequence_token_ids.append(latest_token.item())

                if latest_token.item() == 50256:
                    print("Generated [END] token")
                    break

    generated_text = tokenizer.decode(sequence_token_ids)
    _print_decode_summary(itl_times)

    return generated_text, ttft, itl_times, sequence_token_ids


def generate(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    device='cpu',
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.1,
    use_cache=True,
    use_kv_cache=None,
    return_metrics=False,
):
    """Autoregressive generation with temperature/top-k sampling.

    Parameters
    - model: GPT model instance
    - tokenizer: GPT2Tokenizer instance
    - prompt: initial text prompt
    - max_tokens: maximum number of tokens to generate
    - device: inference device ('cpu' or 'cuda')
    - temperature, top_k, repetition_penalty: sampling controls
    - use_cache: enable stateful KV-cached decoding when True
    - use_kv_cache: legacy alias for use_cache
    - return_metrics: when True, return text plus timing metadata

    Returns
    - Generated text string
    """
    
    
    if use_kv_cache is not None:
        use_cache = use_kv_cache

    generated_text, ttft, itl_times, sequence_token_ids = _run_generation_once(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        use_cache=use_cache,
    )

    if return_metrics:
        return generated_text, ttft, itl_times, sequence_token_ids

    return generated_text


def benchmark_cache_scaling(model, tokenizer, args):
    benchmark_tokens = 256

    print("\n=== Cache Scaling Benchmark ===")
    print(f"Prompt: {args.prompt}")
    print(f"Tokens: {benchmark_tokens}")

    torch.manual_seed(args.seed)
    baseline_text, baseline_ttft, baseline_itl, _ = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=benchmark_tokens,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        use_cache=False,
        return_metrics=True,
    )

    torch.manual_seed(args.seed)
    cached_text, cached_ttft, cached_itl, _ = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=benchmark_tokens,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        use_cache=True,
        return_metrics=True,
    )

    print("\n=== Comparative Benchmark Summary ===")
    print(f"No Cache  - TTFT: {baseline_ttft:.6f}s | Decode throughput: {_format_decode_throughput(baseline_itl)}")
    print(f"KV Cache  - TTFT: {cached_ttft:.6f}s | Decode throughput: {_format_decode_throughput(cached_itl)}")
    print(f"Outputs match: {baseline_text == cached_text}")

    _print_itl_benchmark_table(baseline_itl, cached_itl)


def main():
    """Main execution entrypoint for text generation."""
    
    
    parser = argparse.ArgumentParser(
        description="Generate text using GPT model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of AI is",
        help="Initial text prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to repeated tokens"
    )
    parser.add_argument(
        "--use-cache",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable KV-cached decoding (use --use_cache False for Phase 1 stateless generation)"
    )
    parser.add_argument(
        "--benchmark_cache_scaling",
        action="store_true",
        help="Run a 256-token no-cache vs KV-cache benchmark and print an ITL scaling table"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/gpt2_124m_state_dict.pt",
        help="Path to model weights state dict"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cpu or cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used for benchmark comparisons"
    )
    
    args = parser.parse_args()
    
    # GPT-2 small configuration
    vocab_size = 50257  # Standard GPT-2 vocabulary size
    max_seq_len = 1024  # Maximum sequence length
    dim = 768           # Model/embedding dimension
    num_heads = 12      # Number of attention heads
    num_layers = 12     # Number of transformer blocks
    
    print(f"Initializing GPT model with {num_layers} layers, {num_heads} heads, {dim} dim")
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0
    )
    
    
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"Loading pretrained weights from {args.model_path}")
        try:
            GPT.load_pretrained_weights(model, args.model_path)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"Warning: Model weights file not found at {args.model_path}")
        print("Using randomly initialized weights")
    
    # Move model to device
    model.to(args.device)
    
    
    print("Initializing GPT-2 tokenizer")
    tokenizer = GPT2Tokenizer()

    if args.benchmark_cache_scaling:
        benchmark_cache_scaling(model, tokenizer, args)
        return
    
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        use_cache=args.use_cache,
    )
    
    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)


if __name__ == "__main__":
    main()
