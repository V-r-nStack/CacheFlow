from contextlib import ContextDecorator
import time

import torch


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