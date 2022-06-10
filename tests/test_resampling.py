import numpy as np
import pytest

from liftlab import resampling

ENGINES = ["python"] + (["cpp"] if resampling.cpp_available() else [])
rng = np.random.default_rng(0)
C = rng.lognormal(3, 0.5, 800)
T = rng.lognormal(3.05, 0.5, 700)


def test_cpp_extension_built():
    assert resampling.cpp_available()


@pytest.mark.parametrize("engine", ENGINES)
def test_bootstrap_distribution(engine):
    d = resampling.bootstrap_mean_diff(C, T, 3000, seed=1, engine=engine)
    se = np.sqrt(C.var(ddof=1) / C.size + T.var(ddof=1) / T.size)
    assert d.mean() == pytest.approx(T.mean() - C.mean(), abs=0.1 * se)
    assert d.std() == pytest.approx(se, rel=0.06)


@pytest.mark.parametrize("engine", ENGINES)
def test_permutation_null(engine):
    d = resampling.permutation_mean_diff(C, T, 3000, seed=1, engine=engine)
    pooled = np.concatenate([C, T])
    se = pooled.std(ddof=1) * np.sqrt(1 / C.size + 1 / T.size)
    assert abs(d.mean()) < 0.1 * se
    assert d.std() == pytest.approx(se, rel=0.06)


@pytest.mark.parametrize("op", ["bootstrap_mean_diff", "permutation_mean_diff"])
@pytest.mark.parametrize("engine", ENGINES)
def test_seed_reproducible(op, engine):
    f = getattr(resampling, op)
    assert np.array_equal(f(C, T, 200, 5, engine), f(C, T, 200, 5, engine))
    assert not np.array_equal(f(C, T, 200, 5, engine), f(C, T, 200, 6, engine))


@pytest.mark.skipif(not resampling.cpp_available(), reason="C++ engine not built")
@pytest.mark.parametrize("op", ["bootstrap_mean_diff", "permutation_mean_diff"])
def test_thread_count_does_not_change_result(op):
    f = getattr(resampling, op)
    assert np.array_equal(f(C, T, 300, 3, "cpp", 1), f(C, T, 300, 3, "cpp", 4))


@pytest.mark.skipif(not resampling.cpp_available(), reason="C++ engine not built")
@pytest.mark.parametrize("op", ["bootstrap_mean_diff", "permutation_mean_diff"])
def test_engines_agree_statistically(op):
    f = getattr(resampling, op)
    a, b = f(C, T, 4000, 1, "cpp"), f(C, T, 4000, 2, "python")
    assert a.mean() == pytest.approx(b.mean(), abs=0.15 * a.std())
    assert a.std() == pytest.approx(b.std(), rel=0.08)


def test_bad_inputs():
    with pytest.raises(ValueError):
        resampling.bootstrap_mean_diff([], T)
    with pytest.raises(ValueError):
        resampling.bootstrap_mean_diff(C, T, engine="gpu")
