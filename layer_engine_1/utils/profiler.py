from contextlib import ContextDecorator
from dataclasses import dataclass
import sys
import time
from typing import Optional

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import resource
except ImportError:  # pragma: no cover - optional dependency
    resource = None


def _synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ExecutionTimer(ContextDecorator):
    def __enter__(self):
        _synchronize_cuda()
        self._start = time.perf_counter()
        self.elapsed = None
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        _synchronize_cuda()
        self.elapsed = time.perf_counter() - self._start
        return False


def profile_execution(fn):
    def wrapper(*args, **kwargs):
        with ExecutionTimer() as timer:
            result = fn(*args, **kwargs)
        return result, timer.elapsed

    return wrapper


@dataclass
class SystemSnapshot:
    wall_time_s: float
    cpu_time_s: float
    rss_mb: Optional[float]
    gpu_allocated_mb: Optional[float]
    gpu_reserved_mb: Optional[float]
    gpu_peak_allocated_mb: Optional[float]
    gpu_peak_reserved_mb: Optional[float]


def is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def synchronize_device(device: str):
    if is_cuda_device(device):
        torch.cuda.synchronize()


def get_process_rss_mb() -> Optional[float]:
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0

    return None


def get_gpu_memory_mb(device: str) -> Optional[dict]:
    if not is_cuda_device(device):
        return None

    device_obj = torch.device(device)
    device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_index) / (1024.0 * 1024.0),
        "reserved_mb": torch.cuda.memory_reserved(device_index) / (1024.0 * 1024.0),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device_index) / (1024.0 * 1024.0),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device_index) / (1024.0 * 1024.0),
    }


def capture_system_snapshot(device: str, start_wall_time: float, start_cpu_time: float) -> SystemSnapshot:
    gpu_stats = get_gpu_memory_mb(device)
    return SystemSnapshot(
        wall_time_s=time.perf_counter() - start_wall_time,
        cpu_time_s=time.process_time() - start_cpu_time,
        rss_mb=get_process_rss_mb(),
        gpu_allocated_mb=None if gpu_stats is None else gpu_stats["allocated_mb"],
        gpu_reserved_mb=None if gpu_stats is None else gpu_stats["reserved_mb"],
        gpu_peak_allocated_mb=None if gpu_stats is None else gpu_stats["peak_allocated_mb"],
        gpu_peak_reserved_mb=None if gpu_stats is None else gpu_stats["peak_reserved_mb"],
    )


def format_mb(value: Optional[float]) -> str:
    return f"{value:.2f} MB" if value is not None else "N/A"


def decode_throughput_tokens_per_second(token_count: int, itl_times_s: list[float]) -> Optional[float]:
    if token_count <= 0 or not itl_times_s:
        return None
    total_decode_time = sum(itl_times_s)
    if total_decode_time <= 0:
        return None
    return token_count / total_decode_time