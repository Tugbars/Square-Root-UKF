"""
Benchmark: Python ctypes overhead vs pure C computation

Compares:
1. N individual Python->C calls (includes N ctypes crossings)
2. 1 Python->C call that does N steps (includes 1 ctypes crossing)

The difference isolates the overhead per call.
"""

import numpy as np
import time
import ctypes
from srukf import _lib, c_double_p, c_void_p, create_trend_filter

# Set up batch function signature
_lib.srukf_step_batch.argtypes = [c_void_p, c_double_p, ctypes.c_int]
_lib.srukf_step_batch.restype = None


def benchmark_overhead():
    print("=" * 70)
    print("Python ctypes Overhead Benchmark")
    print("=" * 70)
    
    # Test sizes
    N_VALUES = [1000, 10000, 100000]
    
    for N in N_VALUES:
        print(f"\nN = {N:,} steps")
        print("-" * 40)
        
        # Generate measurements
        np.random.seed(42)
        measurements = np.ascontiguousarray(
            100.0 + np.cumsum(np.random.randn(N) * 0.1),
            dtype=np.float64
        )
        
        # ─────────────────────────────────────────────────────────────
        # Method 1: N individual calls
        # ─────────────────────────────────────────────────────────────
        ukf1 = create_trend_filter(nu=4.0)
        z = np.zeros(1, dtype=np.float64)
        
        start = time.perf_counter()
        for i in range(N):
            z[0] = measurements[i]
            ukf1.step(z)
        time_individual = time.perf_counter() - start
        
        # ─────────────────────────────────────────────────────────────
        # Method 2: 1 batch call
        # ─────────────────────────────────────────────────────────────
        ukf2 = create_trend_filter(nu=4.0)
        
        start = time.perf_counter()
        _lib.srukf_step_batch(
            ukf2._ptr,
            measurements.ctypes.data_as(c_double_p),
            N
        )
        time_batch = time.perf_counter() - start
        
        # ─────────────────────────────────────────────────────────────
        # Analysis
        # ─────────────────────────────────────────────────────────────
        overhead_total = time_individual - time_batch
        overhead_per_call = overhead_total / N * 1e6  # microseconds
        
        time_per_step_individual = time_individual / N * 1e6
        time_per_step_batch = time_batch / N * 1e6
        
        print(f"  Individual calls: {time_individual*1000:8.2f} ms  ({time_per_step_individual:.2f} μs/step)")
        print(f"  Batch call:       {time_batch*1000:8.2f} ms  ({time_per_step_batch:.2f} μs/step)")
        print(f"  ─────────────────────────────────")
        print(f"  Pure C time:      {time_per_step_batch:.2f} μs/step")
        print(f"  Python overhead:  {overhead_per_call:.2f} μs/call ({overhead_per_call/time_per_step_individual*100:.0f}%)")
        print(f"  Speedup (batch):  {time_individual/time_batch:.1f}x")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  - 'Batch call' = pure C computation time (MKL + filter logic)")
    print("  - 'Python overhead' = ctypes marshalling per call")
    print("  - For streaming, use batch processing when possible")
    print("=" * 70)


def benchmark_vs_bench_srukf():
    """Compare with the pure C benchmark results."""
    print("\n" + "=" * 70)
    print("Comparison with Pure C Benchmark (bench_srukf.c)")
    print("=" * 70)
    
    N = 100000
    measurements = np.ascontiguousarray(
        100.0 + np.cumsum(np.random.randn(N) * 0.1),
        dtype=np.float64
    )
    
    ukf = create_trend_filter(nu=4.0)
    
    # Warm up
    _lib.srukf_step_batch(ukf._ptr, measurements.ctypes.data_as(c_double_p), 1000)
    
    # Benchmark
    start = time.perf_counter()
    _lib.srukf_step_batch(ukf._ptr, measurements.ctypes.data_as(c_double_p), N)
    elapsed = time.perf_counter() - start
    
    us_per_step = elapsed / N * 1e6
    steps_per_sec = N / elapsed
    
    print(f"\n  Python batch call: {us_per_step:.2f} μs/step")
    print(f"  Steps per second:  {steps_per_sec/1000:.0f}K")
    print(f"\n  Expected from bench_srukf.c: ~2.0-2.5 μs/step")
    print(f"  Difference is Python->C call overhead (one-time per batch)")


if __name__ == "__main__":
    benchmark_overhead()
    benchmark_vs_bench_srukf()
