### `project.md`

```markdown
# Engineering Roadmap: CacheFlow Phase 1

## Core Objective

Phase 1 (`layer_engine_1`) establishes the architectural baseline for transformer inference. The goal is to build a mathematically complete, strictly unoptimized autoregressive forward pass. Understanding this computational graph and its massive memory overhead during generation is the prerequisite for future work on KV-caching and advanced runtime scheduling.

## Technical Specifications

* **Architecture:** Decoder-only Transformer (GPT-2 scale, 124M parameters).
* **Framework:** PyTorch (Tensors and Autograd only; no high-level NLP abstractions).
* **Target Environment:** CPU/CUDA (Standard PyTorch execution, no custom C++ kernels yet).
* **Primary Constraints:** No KV-caching. Recompute the entire sequence at every decoding step to benchmark the compute waste.

## Implementation Sequence

The development lifecycle for `layer_engine_1` is strictly divided into functional milestones.

### 1. Initialization and Data Plane
* Define the `GPTConfig` configuration schema.
* Implement the script to fetch pre-trained weights.
* Establish the base vocabulary and tokenization wrapper.

### 2. Execution Primitives
* **Embeddings:** Map discrete input IDs to dense vector spaces (Token and Positional). Track tensor shape transformation: `(batch, seq) -> (batch, seq, dim)`.
* **Causal Attention:** Implement $Q, K, V$ projections. Apply head splitting, the lower-triangular causal mask, and softmax. Calculate the compute-heavy $Q \times K^T$ attention scores.
* **Feedforward:** Implement the intermediate multi-layer perceptron expansion (4x hidden dimension) and GELU activation.

### 3. Block Assembly and Routing
* Combine Attention and FFN into a `TransformerBlock`.
* Route the residual stream cleanly: $x = x + Attention(LayerNorm(x))$.
* Stack blocks dynamically based on configuration.
* Map downloaded pre-trained weights directly into custom PyTorch parameters.

### 4. Runtime and Instrumentation
* Construct the `while` loop for autoregressive greedy decoding.
* Slice the logits tensor to extract the final timestep probability distribution.
* Implement decorators for precision timing using `time.perf_counter()`.
* Log Time to First Token (TTFT) and Inter-Token Latency (ITL) to the console.

## Git Workflow Standard

All development follows a strict feature-branch workflow to maintain a clean history.

* **Main Branch:** `main` (Stable execution only).
* **Integration Branch:** `dev` (Tested features).
* **Feature Branches:** `feat/feature-name` (e.g., `feat/causal-attention`).

### Commit Convention
Commits must be atomic and categorized.
* `feat:` A new architectural component or script.
* `fix:` Resolving tensor shape errors or broadcasting issues.
* `chore:` Updating requirements or formatting.
* `docs:` Modifying this roadmap or inline tensor shape comments.

### Merge Protocol
1. Rebase `feat` branch against `dev` locally.
2. Resolve any conflicts in standard execution files.
3. Open a Pull Request for self-review.
4. Merge into `dev` using a fast-forward or squash merge.

## Success Criteria

Phase 1 is considered complete when the engine can ingest a prompt, deterministically generate coherent text using pre-trained weights, and output a detailed latency breakdown of the inference cycle demonstrating the $O(N^2)$ slowdown as sequence length increases.