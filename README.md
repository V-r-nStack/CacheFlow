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

```bash
git clone [https://github.com/V-r-nStack/CacheFlow.git](https://github.com/V-r-nStack/CacheFlow.git)
cd CacheFlow/layer_engine_1
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt