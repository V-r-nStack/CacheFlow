"""Runtime tracer for per-tick engine metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class RuntimeTracer:
    """Collect per-tick metrics into an in-memory DataFrame."""

    _rows: List[Dict[str, float]] = field(default_factory=list)

    def record_tick(
        self,
        timestamp: float,
        queue_depth: int,
        active_batch_size: int,
        allocated_kv_slots: int,
        free_kv_slots: int,
        itl_s: float,
        avg_wait_s: Optional[float] = None,
        p95_wait_s: Optional[float] = None,
        max_starvation_s: Optional[float] = None,
        short_long_ratio: Optional[float] = None,
        queue_wait_latency: Optional[float] = None,
        admission_latency: Optional[float] = None,
        compute_prefill_latency: Optional[float] = None,
        first_decode_latency: Optional[float] = None,
        steady_state_itl: Optional[float] = None,
    ) -> None:
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
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def dump_csv(self, file_path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)


__all__ = ["RuntimeTracer"]