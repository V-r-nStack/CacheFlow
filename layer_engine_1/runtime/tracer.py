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
    ) -> None:
        self._rows.append(
            {
                "timestamp": float(timestamp),
                "queue_depth": int(queue_depth),
                "active_batch_size": int(active_batch_size),
                "allocated_kv_slots": int(allocated_kv_slots),
                "free_kv_slots": int(free_kv_slots),
                "itl_s": float(itl_s),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def dump_csv(self, file_path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)


__all__ = ["RuntimeTracer"]