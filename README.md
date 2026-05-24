# CacheFlow

CacheFlow is a systems-level, first-principles implementation of a Transformer inference runtime. The project reconstructs the LLM serving stack end-to-end, with an emphasis on observability, scheduling, and memory pressure under sustained load.

## What This Project Does Today
- Implements a GPT-2 124M-scale decoder-only model in PyTorch with explicit causal attention.
- Runs autoregressive generation and reports Time to First Token (TTFT) and Inter-Token Latency (ITL).
- Manages KV-cache slots with a static logical-to-physical mapping.
- Executes continuous batching with multiple scheduler policies: FCFS, shortest-prompt-first, and fairness-preemptive.
- Generates adversarial workloads that create persistent queue buildup and starvation pressure.
- Streams per-tick runtime metrics (queue depth, batch utilization, KV pressure, residency, starvation) to CSV.
- Provides benchmarking and a Phase 3 evaluation harness with plots and a Markdown report.

## Repository Layout
| Phase | Directory | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `layer_engine_1/` | Baseline decoding, scheduling, instrumentation | Active |
| **Phase 2** | `kv_engine_2/` | KV-cache mechanics, allocator improvements | Planned |
| **Phase 3** | `batch_engine_3/` | Production batching, throughput optimization | Planned |

## Environment Setup

### 1) Create a Virtual Environment
```bash
python3 -m venv .venv
```

### 2) Activate the Environment
```bash
source .venv/bin/activate
```

### 3) Install Dependencies
```bash
pip install -r layer_engine_1/requirements.txt
```

### 4) Download GPT-2 Weights
```bash
python layer_engine_1/utils/weights.py
```
This writes the checkpoint to `weights/gpt2_124m_state_dict.pt` by default.

## Quick Start: Text Generation

### Basic Generation
```bash
python layer_engine_1/generate.py --prompt "The future of AI is"
```

### Custom Max Tokens
```bash
python layer_engine_1/generate.py \
  --prompt "Artificial intelligence will" \
  --max-tokens 100
```

### CPU or CUDA
```bash
python layer_engine_1/generate.py \
  --prompt "Transformers are" \
  --device cpu
```

```bash
python layer_engine_1/generate.py \
  --prompt "Deep learning models" \
  --device cuda
```

### Sampling Controls
```bash
python layer_engine_1/generate.py \
  --prompt "Once upon a time" \
  --temperature 1.0 \
  --top-k 100 \
  --repetition-penalty 1.2
```

## Runtime Benchmarks and Stress Tests

### Scheduler Benchmark (Policies + Adversarial Profiles)
```bash
python layer_engine_1/tools/benchmark_scheduler.py \
  --profiles sustained_overload,starvation_pressure \
  --duration-s 300
```

### Phase 3 Evaluation Harness
```bash
python layer_engine_1/tools/evaluate_phase3.py \
  --out-dir /home/earthy-zeus/MyProjects/runtime_benchmarks \
  --profiles sustained_overload,starvation_pressure \
  --duration-s 300
```

### Stress Memory Under Load
```bash
python layer_engine_1/tools/stress_memory.py --duration-s 2.0 --drain-s 1.0
```

### Plot Runtime Traces
```bash
python layer_engine_1/tools/benchmark_traces.py \
  --input "/tmp/engine_metrics_test.csv" \
  --out-dir /tmp/runtime_traces
```

## Tests
```bash
python layer_engine_1/tests/test_runtime.py
```

## Outputs and Artifacts
- CSV traces are emitted by the runtime tracer and evaluation harnesses.
- Plots and reports are written to output directories and printed to stdout.
- The Phase 3 harness writes `phase3_summary.csv` and `phase3_report.md`.

## Operational Notes
- For smaller GPUs, reduce `--max-batch-size` and `--max-seq-len` on benchmark runs.
- Adversarial profiles are intentionally aggressive and can saturate queues quickly.
- Workload duration defaults to 300 seconds in Phase 3 to force sustained pressure behavior.

## Common Issues
- If weight download fails, rerun `python layer_engine_1/utils/weights.py` to retry.
- If CUDA OOM occurs, rerun with smaller batch size and sequence length.