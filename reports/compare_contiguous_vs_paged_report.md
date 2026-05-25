# Contiguous vs Paged KV Comparative Benchmark

## Summary
| profile             | memory_backend   |   throughput_toks_per_s |   max_queue_depth |   p95_queue_depth |   avg_residency |   max_residency |   avg_allocator_churn_rate |   avg_page_gather_latency_s |   final_internal_fragmentation_ratio |   final_page_pool_occupancy | trace_path                                                                     |
|:--------------------|:-----------------|------------------------:|------------------:|------------------:|----------------:|----------------:|---------------------------:|----------------------------:|-------------------------------------:|----------------------------:|:-------------------------------------------------------------------------------|
| sustained_overload  | contiguous       |                 502.592 |             92881 |           91704.6 |         63.964  |              64 |                  484104    |                           0 |                               0.1355 |                     50.1561 | /tmp/cacheflow_final_benchmark/traces/trace_sustained_overload_contiguous.csv  |
| sustained_overload  | paged            |                 728.402 |             83826 |           82461.9 |         63.9158 |              64 |                    4783.78 |                           0 |                               1.1282 |                     51.1043 | /tmp/cacheflow_final_benchmark/traces/trace_sustained_overload_paged.csv       |
| starvation_pressure | contiguous       |                1185.91  |             47822 |           46967.8 |         63.7528 |              64 |                  254313    |                           0 |                               0.5223 |                     13.0118 | /tmp/cacheflow_final_benchmark/traces/trace_starvation_pressure_contiguous.csv |
| starvation_pressure | paged            |                1412.86  |             45301 |           44354.2 |         63.7807 |              64 |                    6704.77 |                           0 |                               4.3147 |                     12.6741 | /tmp/cacheflow_final_benchmark/traces/trace_starvation_pressure_paged.csv      |

## Notes
- Same runtime engine, scheduler, workload generator, and KV pool size.
- Only `--memory-backend` differs: contiguous slot ownership vs paged fragmentation.

## Artifacts
- plots/queue_depth_vs_time.png
- plots/residency_vs_memory_churn.png