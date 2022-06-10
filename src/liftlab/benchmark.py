"""Benchmark the C++ engine against the NumPy reference implementation."""
from __future__ import annotations

import os
import time

import pandas as pd

from . import resampling
from .simulate import ExperimentConfig, simulate_experiment


def _time(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run_benchmark(n_users: int = 20_000, n_resamples: int = 2000, repeats: int = 3,
                  thread_counts: tuple[int, ...] = (1, 0)) -> pd.DataFrame:
    """Best-of-`repeats` wall time per operation/engine. threads=0 means all cores."""
    if not resampling.cpp_available():
        raise RuntimeError("C++ engine not built; run `pip install -e .`")
    df = simulate_experiment(ExperimentConfig(n_users=n_users, relative_lift=0.03))
    c = df.loc[df.arm == "control", "value"].to_numpy()
    t = df.loc[df.arm == "treatment", "value"].to_numpy()
    rows = []
    for op_name, op in (("bootstrap", resampling.bootstrap_mean_diff),
                        ("permutation", resampling.permutation_mean_diff)):
        py_s = _time(lambda: op(c, t, n_resamples, 0, "python"), repeats)
        rows.append((op_name, "numpy", "-", py_s, 1.0))
        for th in thread_counts:
            s = _time(lambda: op(c, t, n_resamples, 0, "cpp", th), repeats)
            label = "all" if th == 0 else str(th)
            rows.append((op_name, "c++", label, s, py_s / s))
    out = pd.DataFrame(rows, columns=["operation", "engine", "threads", "seconds", "speedup_vs_numpy"])
    out.attrs.update(n_users=n_users, n_resamples=n_resamples, cores=os.cpu_count())
    return out
