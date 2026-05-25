# CacheFlow — Runtime Architecture

## Overview

CacheFlow is an experimental transformer inference runtime designed to study how scheduler behavior, KV-cache residency pressure, and memory topology interact under sustained serving load.

The project began as a minimal decoder-only transformer implementation and evolved into a systems-oriented runtime for investigating:

* continuous batching
* scheduler saturation
* KV-cache memory amplification
* fragmented paged KV residency
* allocator churn
* runtime observability
* overload behavior under concurrent decoding

The repository is intentionally focused on runtime systems engineering rather than model quality or application-layer LLM tooling.

The runtime supports interchangeable KV memory architectures through a pluggable memory backend abstraction, allowing direct comparison between:

* contiguous KV residency
* paged fragmented KV residency

under identical workloads and scheduler pressure.

---

# Core Runtime Architecture

The runtime is organized into a small set of systems-oriented subsystems.

---

## runtime/

The `runtime/` subsystem owns execution orchestration.

Responsibilities:

* token-by-token decode loop
* active batch execution
* sequence lifecycle management
* model block execution
* decode iteration pacing
* runtime state evolution

Key files:

* `runtime/engine.py`
* `runtime/batching.py`
* `runtime/sequence.py`
* `runtime/gpt.py`
* `runtime/blocks.py`

The engine continuously executes decode iterations over the scheduler’s active batch while interacting with the memory subsystem to materialize KV state for attention traversal.

This subsystem models the behavior of a serving runtime rather than an offline transformer inference script.

---

## scheduler/

The `scheduler/` subsystem controls admission and execution policy.

Responsibilities:

* waiting queue management
* active batch formation
* fairness control
* starvation mitigation
* preemption
* admission control under memory pressure

Key file:

* `scheduler/scheduler.py`

Implemented policies include:

* FCFS scheduling
* shortest-prompt-first scheduling
* fairness-oriented preemptive scheduling

The scheduler intentionally operates independently from physical memory layout. It reasons only about:

* sequence state
* token counts
* residency estimates
* queue pressure

This separation allows memory semantics to evolve independently from scheduling logic.

---

## memory/

The `memory/` subsystem is the architectural center of CacheFlow.

It implements the runtime’s KV residency semantics.

### Major Concepts

#### MemoryBackend

A pluggable abstraction used by attention layers to retrieve KV tensors.

Two implementations are supported:

* contiguous residency
* paged fragmented residency

This abstraction enables experimentally rigorous comparison under identical runtime conditions.

---

### Contiguous KV Residency

The contiguous backend models traditional KV ownership semantics where sequences reserve physically contiguous regions of memory.

Characteristics:

* simple allocation behavior
* low gather overhead
* allocator instability under saturation
* residency amplification during overload

---

### Paged Fragmented Residency

The paged backend virtualizes KV ownership through:

* PhysicalPage pools
* PageAllocator
* BlockTable logical mappings

Logical token continuity becomes independent from physical memory placement.

Attention traversal dynamically reconstructs logical KV continuity by gathering fragmented pages at runtime.

This models the core memory semantics behind modern serving systems such as vLLM-style paged attention architectures.

---

### MemoryManager

The `MemoryManager` provides a unified logical capacity abstraction across all memory backends.

This ensures:

* equal scheduler pressure
* equal logical residency capacity
* experimentally fair backend comparisons

The runtime therefore compares:

# memory semantics

rather than:

# raw memory capacity.

---

### Key Files

* `memory/memory_backend.py`
* `memory/page_allocator.py`
* `memory/block_table.py`
* `memory/memory_manager.py`
* `memory/memory_factory.py`
* `memory/kv_cache.py`

---

## attention/

The `attention/` subsystem performs causal attention traversal over runtime-managed KV memory.

Key file:

* `attention/attention.py`

The attention layer does not directly own KV storage.

Instead, it requests materialized KV tensors through the `MemoryBackend` interface.

Depending on the active backend:

* contiguous tensors may be returned directly
* fragmented pages may be dynamically gathered and reconstructed into temporary logical tensors

This separation allows the same attention implementation to operate across different memory architectures.

---

## tracing/

The `tracing/` subsystem provides runtime observability.

Responsibilities:

* TTFT instrumentation
* ITL instrumentation
* queue depth tracking
* batch utilization tracking
* allocator churn measurement
* residency lifetime tracking
* internal fragmentation tracking
* page gather latency tracking
* overload onset detection
* memory saturation detection

Key files:

* `tracing/tracer.py`
* `tracing/profiler.py`

The tracer records runtime state continuously during execution and exports structured CSV traces for offline analysis.

---

## workloads/

The `workloads/` subsystem generates synthetic serving pressure.

Responsibilities:

* concurrent request generation
* sustained overload simulation
* starvation pressure generation
* burst traffic simulation
* long-context residency stress
* synthetic decode workloads

Key file:

* `workloads/workload.py`

Workloads are intentionally adversarial and designed to expose:

* scheduler instability
* queue poisoning
* residency amplification
* allocator churn
* fragmentation behavior

---

## benchmarks/

The `benchmarks/` subsystem contains reproducible runtime evaluation harnesses.

Responsibilities:

* scheduler benchmarking
* runtime saturation experiments
* contiguous vs paged comparison
* stress testing
* trace generation
* runtime evaluation

Key scripts:

* `benchmarks/evaluate_runtime.py`
* `benchmarks/compare_contiguous_vs_paged.py`
* `benchmarks/stress_memory.py`
* `benchmarks/benchmark_scheduler.py`

These scripts form the primary experimental interface of CacheFlow.

---

# Runtime Data Flow

The runtime executes through the following pipeline:

1. Workload Generation

   * synthetic workloads generate incoming `Sequence` objects

2. Scheduler Admission

   * sequences enter waiting queues
   * scheduler forms active decode batches

3. Runtime Decode Iteration

   * engine executes one decode step for all active sequences

4. Attention Traversal

   * attention requests KV tensors through `MemoryBackend`

5. Memory Materialization

   * backend reconstructs KV state:

     * directly for contiguous residency
     * dynamically through fragmented page gathering for paged residency

6. Token Generation

   * next-token logits are computed
   * sequence state advances

7. Tracing

   * runtime metrics are emitted continuously

This loop models a serving-oriented autoregressive decode runtime rather than isolated transformer inference.

---

# Experimental Focus

CacheFlow primarily studies how scheduler behavior interacts with memory topology under sustained decode residency.

The project evolved through four major architectural stages:

### Stage 1 — Transformer Runtime

* decoder-only GPT implementation
* explicit causal attention
* autoregressive decoding

### Stage 2 — KV Reuse

* incremental decoding
* KV-cache integration
* TTFT and ITL instrumentation

### Stage 3 — Continuous Batching

* multi-sequence serving
* runtime scheduling
* overload simulation
* fairness and starvation experiments

### Stage 4 — Fragmented Paged Residency

* logical/physical memory separation
* PageAllocator
* BlockTable mappings
* fragmented KV traversal
* allocator churn analysis

---

# Key Systems Insights

The runtime experiments demonstrate several important behaviors:

* scheduler pressure amplifies KV residency under sustained concurrent decoding
* contiguous KV ownership tightly couples scheduler turbulence to allocator instability
* paged fragmented residency stabilizes allocator mutation under saturation
* logical/physical memory separation improves overload resilience
* allocator churn becomes a dominant systems signal under residency-dominated pressure

The project therefore focuses on:

# runtime memory systems

rather than:

# model optimization.

---

# Extending CacheFlow

## Adding a Scheduler Policy

Add a new scheduling strategy inside:

* `scheduler/scheduler.py`

Instrumentation hooks should emit:

* queue statistics
* residency metrics
* starvation measurements

---

## Adding a Memory Backend

Implement the `MemoryBackend` interface inside:

* `memory/`

and register the backend through:

* `memory/memory_factory.py`

---

## Adding New Workloads

Extend:

* `workloads/workload.py`

to generate new traffic distributions or adversarial serving patterns.

---

# Recommended Entry Points

Runtime execution:

* `app.py`

Runtime evaluation:

* `benchmarks/evaluate_runtime.py`

Backend comparison:

* `benchmarks/compare_contiguous_vs_paged.py`

Stress testing:

* `benchmarks/stress_memory.py`

---

# Research Context

CacheFlow is heavily inspired by prior work on:

* continuous batching
* KV-cache optimization
* paged attention
* serving-runtime scheduling
* memory virtualization for autoregressive decoding

Relevant papers and systems:

* Orca: A Distributed Serving System for Transformer-Based Generative Models
* vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
* Efficient Memory Management for Large Language Model Serving with PagedAttention
* Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
* FasterTransformer

CacheFlow does not attempt to reproduce these systems directly. Instead, it isolates and reconstructs the core runtime semantics needed to experimentally study scheduler pressure and KV memory behavior under sustained overload.

---
