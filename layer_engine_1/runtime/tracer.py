"""Runtime tracer for per-tick engine metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import pandas as pd


class ResidencyAnalyzer:
    """Track residency lifetimes and overlap dynamics under sustained load."""

    def __init__(self) -> None:
        self._baseline_ids: Optional[set] = None
        self._baseline_start: Optional[float] = None
        self._last_half_life_s: Optional[float] = None
        self._decode_overlap_s: float = 0.0
        self._last_timestamp: Optional[float] = None
        self._completed_count: int = 0
        self._sum_residency_lifetime_s: float = 0.0
        self._sum_residency_amplification: float = 0.0
        self._last_residency_lifetime_s: Optional[float] = None
        self._last_residency_amplification: Optional[float] = None

    def observe_tick(
        self,
        timestamp: float,
        active_sequence_ids: Optional[List[int]],
        active_batch_size: int,
    ) -> Dict[str, Optional[float]]:
        if self._last_timestamp is None:
            self._last_timestamp = float(timestamp)

        dt_s = max(0.0, float(timestamp) - float(self._last_timestamp))
        if dt_s > 0.0 and int(active_batch_size) > 1:
            self._decode_overlap_s += dt_s

        half_life = self._last_half_life_s
        if active_sequence_ids is not None:
            active_set = set(active_sequence_ids)
            if not active_set:
                if self._baseline_ids and self._baseline_start is not None:
                    self._last_half_life_s = float(timestamp) - float(self._baseline_start)
                    half_life = self._last_half_life_s
                self._baseline_ids = None
                self._baseline_start = None
            else:
                if not self._baseline_ids:
                    self._baseline_ids = set(active_set)
                    self._baseline_start = float(timestamp)
                else:
                    baseline_size = len(self._baseline_ids)
                    remaining = len(self._baseline_ids.intersection(active_set))
                    if baseline_size > 0 and remaining <= baseline_size / 2.0:
                        if self._baseline_start is not None:
                            self._last_half_life_s = float(timestamp) - float(self._baseline_start)
                            half_life = self._last_half_life_s
                        self._baseline_ids = set(active_set)
                        self._baseline_start = float(timestamp)

        self._last_timestamp = float(timestamp)
        return {
            "active_sequence_residency_half_life": half_life,
            "decode_overlap_duration": self._decode_overlap_s,
        }

    def record_completion(self, sequence, fallback_finish_time: float) -> Dict[str, Optional[float]]:
        finish_time = sequence.finish_time or fallback_finish_time
        start_time = (
            sequence.first_decode_start_at
            or sequence.prefill_start_at
            or sequence.arrival_time
        )
        lifetime = max(0.0, float(finish_time) - float(start_time))

        compute_time = 0.0
        if sequence.prefill_start_at is not None and sequence.prefill_end_at is not None:
            compute_time += max(0.0, sequence.prefill_end_at - sequence.prefill_start_at)
        if sequence.first_decode_start_at is not None and sequence.first_decode_end_at is not None:
            compute_time += max(0.0, sequence.first_decode_end_at - sequence.first_decode_start_at)
        if sequence._steady_state_itl_sum > 0.0:
            compute_time += float(sequence._steady_state_itl_sum)
        if compute_time <= 0.0 and sequence.ttft_s is not None:
            compute_time = float(sequence.ttft_s)

        amplification = lifetime / compute_time if compute_time > 0.0 else 0.0

        self._completed_count += 1
        self._sum_residency_lifetime_s += lifetime
        self._sum_residency_amplification += amplification
        self._last_residency_lifetime_s = lifetime
        self._last_residency_amplification = amplification

        return {
            "sequence_residency_lifetime": lifetime,
            "residency_amplification": amplification,
        }

    def summary(self) -> Dict[str, Optional[float]]:
        avg_lifetime = (
            self._sum_residency_lifetime_s / self._completed_count
            if self._completed_count > 0
            else 0.0
        )
        avg_amplification = (
            self._sum_residency_amplification / self._completed_count
            if self._completed_count > 0
            else 0.0
        )
        return {
            "avg_sequence_residency_lifetime": avg_lifetime,
            "avg_residency_amplification": avg_amplification,
            "last_sequence_residency_lifetime": self._last_residency_lifetime_s,
            "last_residency_amplification": self._last_residency_amplification,
        }


@dataclass
class RuntimeTracer:
    """Collect per-tick metrics into an in-memory DataFrame."""

    _rows: List[Dict[str, float]] = field(default_factory=list)
    _batch_util_samples: List[float] = field(default_factory=list)
    _start_time: Optional[float] = None
    _last_timestamp: Optional[float] = None
    _total_uptime_s: float = 0.0
    _queue_over_threshold_s: float = 0.0
    _queue_persistence_threshold: int = 50
    _max_batch_occupancy_start: Optional[float] = None
    _max_batch_occupancy_max_s: float = 0.0
    _last_allocated_kv_slots: Optional[int] = None
    _last_queue_depth: Optional[int] = None
    _residency_analyzer: ResidencyAnalyzer = field(default_factory=ResidencyAnalyzer)

    def _update_batch_util_stats(self, batch_utilization_ratio: Optional[float]) -> Dict[str, float]:
        if batch_utilization_ratio is not None:
            self._batch_util_samples.append(float(batch_utilization_ratio))

        if not self._batch_util_samples:
            return {"avg": 0.0, "p95": 0.0}

        ordered = sorted(self._batch_util_samples)
        avg = sum(ordered) / len(ordered)
        idx = max(0, int(math.ceil(0.95 * len(ordered))) - 1)
        p95 = ordered[idx]
        return {"avg": float(avg), "p95": float(p95)}

    def record_sequence_completions(
        self,
        sequences: List,
        timestamp: Optional[float] = None,
    ) -> None:
        if timestamp is None:
            timestamp = 0.0
        for sequence in sequences:
            self._residency_analyzer.record_completion(sequence, float(timestamp))

    def record_tick(
        self,
        timestamp: float,
        queue_depth: int,
        active_batch_size: int,
        allocated_kv_slots: int,
        free_kv_slots: int,
        itl_s: float,
        max_batch_size: Optional[int] = None,
        active_sequence_ids: Optional[List[int]] = None,
        avg_wait_s: Optional[float] = None,
        p95_wait_s: Optional[float] = None,
        max_starvation_s: Optional[float] = None,
        short_long_ratio: Optional[float] = None,
        queue_wait_latency: Optional[float] = None,
        admission_latency: Optional[float] = None,
        compute_prefill_latency: Optional[float] = None,
        first_decode_latency: Optional[float] = None,
        steady_state_itl: Optional[float] = None,
        batch_utilization_ratio: Optional[float] = None,
        idle_decode_slots: Optional[int] = None,
    ) -> None:
        if self._start_time is None:
            self._start_time = float(timestamp)
            self._last_timestamp = float(timestamp)

        dt_s = 0.0
        if self._last_timestamp is not None:
            dt_s = max(0.0, float(timestamp) - float(self._last_timestamp))

        if dt_s > 0.0:
            self._total_uptime_s += dt_s
            if int(queue_depth) > self._queue_persistence_threshold:
                self._queue_over_threshold_s += dt_s

        queue_persistence_ratio = (
            self._queue_over_threshold_s / self._total_uptime_s
            if self._total_uptime_s > 0.0
            else 0.0
        )

        max_batch_occupancy_duration = self._max_batch_occupancy_max_s
        if max_batch_size is not None:
            max_batch_size = int(max_batch_size)
            if int(active_batch_size) == max_batch_size and max_batch_size > 0:
                if self._max_batch_occupancy_start is None:
                    self._max_batch_occupancy_start = float(timestamp)
                current_duration = float(timestamp) - float(self._max_batch_occupancy_start)
                if current_duration > self._max_batch_occupancy_max_s:
                    self._max_batch_occupancy_max_s = current_duration
                max_batch_occupancy_duration = self._max_batch_occupancy_max_s
            else:
                if self._max_batch_occupancy_start is not None:
                    current_duration = float(timestamp) - float(self._max_batch_occupancy_start)
                    if current_duration > self._max_batch_occupancy_max_s:
                        self._max_batch_occupancy_max_s = current_duration
                self._max_batch_occupancy_start = None
                max_batch_occupancy_duration = self._max_batch_occupancy_max_s

        kv_allocation_churn_rate = 0.0
        kv_allocated_per_step = 0.0
        kv_freed_per_step = 0.0
        if self._last_allocated_kv_slots is not None and dt_s > 0.0:
            delta_slots = int(allocated_kv_slots) - int(self._last_allocated_kv_slots)
            kv_allocation_churn_rate = abs(delta_slots) / dt_s
            if delta_slots > 0:
                kv_allocated_per_step = float(delta_slots)
            elif delta_slots < 0:
                kv_freed_per_step = float(-delta_slots)
        self._last_allocated_kv_slots = int(allocated_kv_slots)

        queue_growth_rate = 0.0
        if self._last_queue_depth is not None and dt_s > 0.0:
            queue_growth_rate = (int(queue_depth) - int(self._last_queue_depth)) / dt_s
        self._last_queue_depth = int(queue_depth)
        residency_metrics = self._residency_analyzer.observe_tick(
            timestamp=float(timestamp),
            active_sequence_ids=active_sequence_ids,
            active_batch_size=int(active_batch_size),
        )
        residency_summary = self._residency_analyzer.summary()
        self._last_timestamp = float(timestamp)
        utilization_stats = self._update_batch_util_stats(batch_utilization_ratio)
        self._rows.append(
            {
                "timestamp": float(timestamp),
                "queue_depth": int(queue_depth),
                "active_batch_size": int(active_batch_size),
                "allocated_kv_slots": int(allocated_kv_slots),
                "free_kv_slots": int(free_kv_slots),
                "itl_s": float(itl_s),
                "avg_wait_s": None if avg_wait_s is None else float(avg_wait_s),
                "p95_wait_s": None if p95_wait_s is None else float(p95_wait_s),
                "max_starvation_s": None if max_starvation_s is None else float(max_starvation_s),
                "short_long_ratio": None if short_long_ratio is None else float(short_long_ratio),
                "queue_wait_latency": None
                if queue_wait_latency is None
                else float(queue_wait_latency),
                "admission_latency": None
                if admission_latency is None
                else float(admission_latency),
                "compute_prefill_latency": None
                if compute_prefill_latency is None
                else float(compute_prefill_latency),
                "first_decode_latency": None
                if first_decode_latency is None
                else float(first_decode_latency),
                "steady_state_itl": None
                if steady_state_itl is None
                else float(steady_state_itl),
                "batch_utilization_ratio": None
                if batch_utilization_ratio is None
                else float(batch_utilization_ratio),
                "idle_decode_slots": None if idle_decode_slots is None else int(idle_decode_slots),
                "queue_persistence_ratio": float(queue_persistence_ratio),
                "max_batch_occupancy_duration": float(max_batch_occupancy_duration),
                "active_sequence_residency_half_life": None
                if residency_metrics["active_sequence_residency_half_life"] is None
                else float(residency_metrics["active_sequence_residency_half_life"]),
                "decode_overlap_duration": float(residency_metrics["decode_overlap_duration"]),
                "sequence_residency_lifetime": None
                if residency_summary["last_sequence_residency_lifetime"] is None
                else float(residency_summary["last_sequence_residency_lifetime"]),
                "residency_amplification": None
                if residency_summary["last_residency_amplification"] is None
                else float(residency_summary["last_residency_amplification"]),
                "avg_sequence_residency_lifetime": float(
                    residency_summary["avg_sequence_residency_lifetime"]
                ),
                "avg_residency_amplification": float(
                    residency_summary["avg_residency_amplification"]
                ),
                "kv_allocated_per_step": float(kv_allocated_per_step),
                "kv_freed_per_step": float(kv_freed_per_step),
                "queue_growth_rate": float(queue_growth_rate),
                "kv_allocation_churn_rate": float(kv_allocation_churn_rate),
                "avg_batch_utilization": utilization_stats["avg"],
                "p95_batch_utilization": utilization_stats["p95"],
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def dump_csv(self, file_path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)


__all__ = ["RuntimeTracer", "ResidencyAnalyzer"]