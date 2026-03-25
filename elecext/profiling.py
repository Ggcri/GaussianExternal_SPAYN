# Performance profiling utilities - disabled (no-op stubs)
# All decorators and context managers are pass-throughs with zero overhead.
import functools
from contextlib import contextmanager
from collections import defaultdict


class PerformanceProfiler:
    """No-op profiler. All methods do nothing, no files are created."""

    def __init__(self, log_file=None):
        self.timings = defaultdict(list)

    @contextmanager
    def measure_operation(self, operation_name, track_blocking=True):
        yield

    def track_thread_start(self, thread_name):
        pass

    def track_thread_end(self, thread_name):
        pass

    def track_io_operation(self, operation_type, file_path=None):
        pass

    def detect_thread_contention(self):
        return False

    def print_summary(self):
        pass


# Global profiler instance (no-op)
profiler = PerformanceProfiler()


def profile_function(operation_name=None, track_blocking=True):
    """No-op decorator - returns the function unchanged."""
    def decorator(func):
        return func
    return decorator


def profile_io(operation_type):
    """No-op decorator - returns the function unchanged."""
    def decorator(func):
        return func
    return decorator
