# CacheFlow: Real Status Notes

This file is the candid, day-to-day truth about the project. It is not polished marketing. It is a working note on what exists, what is fragile, and what the system proves.

## What Exists Right Now
- A GPT-2 124M-scale decoder-only model with explicit causal attention and custom weight loading.
- Autoregressive generation with TTFT and ITL instrumentation.
- Static KV allocation and logical-to-physical slot mapping.
- Continuous batching with FCFS, shortest-prompt-first, and fairness-preemptive scheduling.
- Adversarial workload profiles that force persistent queue buildup and starvation pressure.
- Runtime tracer exporting queue depth, batch utilization, KV pressure, residency, and starvation metrics.
- Benchmark harnesses and a Phase 3 evaluation script that produce plots and a Markdown report.

## What It Proves
- Queue growth under sustained overload is measurable and repeatable.
- Shortest-prompt-first can starve long sequences and surface fairness tradeoffs.
- Static KV allocation hits saturation quickly under long-context pressure.
- Residency metrics show sequence lifespan stretching when overload persists.

## What Is Still Fragile
- GPU memory is the first failure mode. Small GPUs need reduced batch size and seq length.
- High arrival rates can saturate queues so fast that plots flatten unless duration is long.
- Fairness-preemptive behavior depends heavily on preemption thresholds.

## What Is Not Here (Yet)
- No paged memory or adaptive KV allocation.
- No distributed or multi-GPU execution.
- No production serving API.

## Where the Truth Shows Up
- `runtime/tracer.py` for CSV fields and metric definitions.
- `runtime/scheduler.py` for starvation and fairness behavior.
- `runtime/workload.py` for adversarial profile definitions.

## How to Reproduce the Pressure Findings
```bash
python layer_engine_1/tools/benchmark_scheduler.py \
  --profiles sustained_overload,starvation_pressure \
  --duration-s 300
```

```bash
python layer_engine_1/tools/evaluate_phase3.py \
  --out-dir /home/earthy-zeus/MyProjects/runtime_benchmarks \
  --profiles sustained_overload,starvation_pressure \
  --duration-s 300
```

## Known Failure Patterns
- `CUDA out of memory` when batch size or max seq length is too high.
- Queue depth spikes that never recover when base/burst rates exceed drain rate.
- Fairness-preemptive can reduce worst starvation, but may reduce throughput.

## Next Engineering Steps
- Add more realistic arrival processes for traffic bursts.
- Calibrate compute pacer to match a fixed throughput budget.
- Extend evaluation to include utilization vs latency tradeoffs across policies.