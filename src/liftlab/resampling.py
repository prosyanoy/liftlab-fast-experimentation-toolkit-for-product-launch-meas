"""Bootstrap / permutation engines: C++ (pybind11, multithreaded) and NumPy reference.

Both return arrays of mean(treatment) - mean(control) replicates. The C++ engine is
deterministic for a given seed regardless of thread count; the NumPy one for a given seed.
"""
from __future__ import annotations

import numpy as np

try:  # compiled extension is optional: everything falls back to NumPy
    from . import _cpp
except ImportError:  # pragma: no cover - exercised only when the extension is not built
    _cpp = None

_CHUNK_ELEMS = 4_000_000  # cap temporary index matrices to ~32 MB


def cpp_available() -> bool:
    return _cpp is not None


def _pick(engine: str) -> str:
    if engine not in {"auto", "cpp", "python"}:
        raise ValueError("engine must be 'auto', 'cpp' or 'python'")
    if engine == "cpp" and _cpp is None:
        raise RuntimeError("C++ engine not built; run `pip install -e .`")
    return "cpp" if engine == "auto" and _cpp is not None else ("python" if engine == "auto" else engine)


def _clean(x) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("each group must be a non-empty 1-D array")
    return a


def bootstrap_mean_diff(control, treatment, n_boot: int = 2000, seed: int = 0,
                        engine: str = "auto", threads: int = 0) -> np.ndarray:
    c, t = _clean(control), _clean(treatment)
    if _pick(engine) == "cpp":
        return _cpp.bootstrap_mean_diff(c, t, n_boot, seed, threads)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    step = max(1, _CHUNK_ELEMS // max(c.size, t.size))
    for s in range(0, n_boot, step):
        b = min(step, n_boot - s)
        mc = c[rng.integers(0, c.size, (b, c.size))].mean(axis=1)
        mt = t[rng.integers(0, t.size, (b, t.size))].mean(axis=1)
        out[s:s + b] = mt - mc
    return out


def permutation_mean_diff(control, treatment, n_perm: int = 2000, seed: int = 0,
                          engine: str = "auto", threads: int = 0) -> np.ndarray:
    c, t = _clean(control), _clean(treatment)
    if _pick(engine) == "cpp":
        return _cpp.permutation_mean_diff(c, t, n_perm, seed, threads)
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([c, t])
    n, nt = pooled.size, t.size
    total = pooled.sum()
    out = np.empty(n_perm)
    step = max(1, _CHUNK_ELEMS // n)
    for s in range(0, n_perm, step):
        b = min(step, n_perm - s)
        # argpartition of random keys == uniform random subset of size nt per row
        idx = np.argpartition(rng.random((b, n)), nt - 1, axis=1)[:, :nt] if nt < n else \
            np.tile(np.arange(n), (b, 1))
        st = pooled[idx].sum(axis=1)
        out[s:s + b] = st / nt - (total - st) / c.size
    return out
