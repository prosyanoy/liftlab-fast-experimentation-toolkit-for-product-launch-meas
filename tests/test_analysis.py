import numpy as np
import pytest

from liftlab.analysis import METHODS, analyze, cuped_adjust
from liftlab.simulate import ExperimentConfig, simulate_experiment

LIFT = 0.05


def _cfg(seed, **kw):
    return ExperimentConfig(n_users=6000, relative_lift=kw.pop("lift", LIFT), seed=seed, **kw)


@pytest.mark.parametrize("method", METHODS)
def test_detects_known_effect(method):
    cfg = ExperimentConfig(n_users=60_000, relative_lift=LIFT, seed=11)
    r = analyze(simulate_experiment(cfg), method, n_resamples=2000, seed=1)
    assert r.significant and r.ci_low < cfg.true_effect < r.ci_high
    assert r.relative_lift == pytest.approx(LIFT, abs=0.02)
    assert r.ci_low < r.effect < r.ci_high


@pytest.mark.parametrize("method", ["ttest", "cuped", "bootstrap", "permutation"])
def test_ci_coverage_of_true_effect(method):
    n_sim = 150 if method in ("ttest", "cuped") else 80
    hit = 0
    for i in range(n_sim):
        cfg = _cfg(1000 + i)
        r = analyze(simulate_experiment(cfg), method, n_resamples=600, seed=i)
        hit += r.ci_low <= cfg.true_effect <= r.ci_high
    # nominal 95%; allow Monte-Carlo noise
    assert hit / n_sim > (0.89 if n_sim == 150 else 0.85)


def test_cuped_reduces_variance_as_theory_predicts():
    df = simulate_experiment(ExperimentConfig(n_users=100_000, relative_lift=0.0, seed=5))
    rho = df.value.corr(df.pre_value)
    plain, cup = analyze(df, "ttest"), analyze(df, "cuped")
    assert cup.variance_reduction == pytest.approx(rho**2, abs=0.01)
    assert (cup.ci_high - cup.ci_low) < (plain.ci_high - plain.ci_low)
    width_ratio = (cup.ci_high - cup.ci_low) / (plain.ci_high - plain.ci_low)
    assert width_ratio == pytest.approx(np.sqrt(1 - rho**2), abs=0.03)


def test_cuped_unbiased_across_runs():
    est = [analyze(simulate_experiment(_cfg(i)), "cuped").effect for i in range(100)]
    true = LIFT * 18
    assert np.mean(est) == pytest.approx(true, abs=3 * np.std(est) / 10)


def test_cuped_adjust_zeroes_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 2 * x + rng.normal(size=5000)
    adj, theta = cuped_adjust(y, x)
    assert theta == pytest.approx(2, abs=0.1)
    assert abs(np.corrcoef(adj, x)[0, 1]) < 1e-10
    assert adj.mean() == pytest.approx(y.mean())


def test_ttest_matches_scipy():
    from scipy import stats
    df = simulate_experiment(_cfg(3))
    r = analyze(df, "ttest")
    ref = stats.ttest_ind(df[df.arm == "treatment"].value, df[df.arm == "control"].value, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_methods_agree_on_p_value():
    df = simulate_experiment(_cfg(8, lift=0.04))
    # CUPED is a different (sharper) estimator, so only the unadjusted methods must agree
    ps = {m: analyze(df, m, n_resamples=20_000, seed=2).p_value
          for m in ("ttest", "bootstrap", "permutation")}
    assert max(ps.values()) - min(ps.values()) < 0.02 + 0.2 * np.mean(list(ps.values()))
    assert analyze(df, "cuped").p_value < ps["ttest"]


def test_python_and_cpp_engines_agree():
    df = simulate_experiment(_cfg(4))
    a = analyze(df, "bootstrap", n_resamples=4000, engine="cpp")
    b = analyze(df, "bootstrap", n_resamples=4000, engine="python")
    assert a.engine == "cpp" and b.engine == "python"
    assert a.ci_low == pytest.approx(b.ci_low, abs=0.03) and a.ci_high == pytest.approx(b.ci_high, abs=0.03)


def test_negative_effect_and_errors():
    r = analyze(simulate_experiment(ExperimentConfig(n_users=60_000, relative_lift=-0.05, seed=2)))
    assert r.significant and r.effect < 0
    with pytest.raises(ValueError):
        analyze(simulate_experiment(_cfg(1)), "magic")
    with pytest.raises(ValueError):
        analyze(simulate_experiment(_cfg(1)).drop(columns="pre_value"), "cuped")
