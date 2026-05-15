# CacheFlow

A systems-level, first-principles implementation of a Transformer inference runtime. 

This repository systematically reconstructs the LLM serving stack. The long-term trajectory of CacheFlow spans from a naive autoregressive execution engine to a memory-aware runtime featuring KV-caching, continuous batching, and custom memory management.

## Project Structure

CacheFlow is built in progressive phases to isolate specific systems engineering challenges.

| Phase | Directory | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `layer_engine_1/` | Core decoding, causal attention, autoregressive baseline, instrumentation. | Active |
| **Phase 2** | `kv_engine_2/` | KV-cache mechanics, memory allocation, state management. | Planned |
| **Phase 3** | `batch_engine_3/` | Continuous batching, runtime scheduling, throughput optimization. | Planned |

## Current Phase: `layer_engine_1`

The `layer_engine_1` module provides a minimal, structurally accurate decoder-only Transformer (GPT-2 124M scale) using PyTorch. It intentionally avoids memory optimizations to establish a baseline for computational cost and tensor shape evolution during sequence generation.

### Key Features
* Custom weight loading pipeline from standard state dictionaries.
* Explicit multi-head causal self-attention mechanism.
* Unoptimized, full-sequence $O(N^2)$ autoregressive decoding loop.
* First-class runtime instrumentation tracking Time to First Token (TTFT) and Inter-Token Latency (ITL).

## Installation and Setup

Navigate to the active engine directory and install the exact dependencies to ensure deterministic execution.

# Run Commands

## Create Virtual Environment

```bash
python3 -m venv .venv
```

---

## Activate Environment

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Minimal setup:

```bash
pip install torch requests tiktoken
```

---

# Inference Commands

## Basic Generation

```bash
python3 generate.py --prompt "The future of AI is"
```

---

## Custom Max Tokens

```bash
python3 generate.py \
  --prompt "Artificial intelligence will" \
  --max-tokens 100
```

---

## CPU Inference

```bash
python3 generate.py \
  --prompt "Transformers are" \
  --device cpu
```

---

## CUDA Inference

```bash
python3 generate.py \
  --prompt "Deep learning models" \
  --device cuda
```

---

## Lower Temperature

```bash
python3 generate.py \
  --prompt "The meaning of life is" \
  --temperature 0.7
```

---

## Higher Creativity

```bash
python3 generate.py \
  --prompt "Once upon a time" \
  --temperature 1.0 \
  --top-k 100
```

---

## Repetition Penalty

```bash
python3 generate.py \
  --prompt "Artificial intelligence" \
  --repetition-penalty 1.2
```

---

## Full Example

```bash
python3 generate.py \
  --prompt "The future of autonomous systems is" \
  --max-tokens 120 \
  --temperature 0.8 \
  --top-k 50 \
  --repetition-penalty
```
