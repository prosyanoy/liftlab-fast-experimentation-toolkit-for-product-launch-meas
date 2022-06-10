import numpy as np
import pytest

from liftlab.power import (minimum_detectable_effect, power_at, sample_size_per_arm,
                           sample_size_proportion)
from liftlab.simulate import ExperimentConfig, simulate_experiment


def test_simulation_is_seeded():
    a = simulate_experiment(ExperimentConfig(seed=3))
    b = simulate_experiment(ExperimentConfig(seed=3))
    c = simulate_experiment(ExperimentConfig(seed=4))
    assert a.equals(b) and not a.equals(c)


def test_simulation_matches_config():
    cfg = ExperimentConfig(n_users=200_000, relative_lift=0.1, baseline_mean=20, cv=0.5, seed=1)
    df = simulate_experiment(cfg)
    ctrl, trt = (df[df.arm == a].value for a in ("control", "treatment"))
    assert ctrl.mean() == pytest.approx(20, rel=0.01)
    assert ctrl.std() / ctrl.mean() == pytest.approx(0.5, rel=0.03)
    assert trt.mean() - ctrl.mean() == pytest.approx(cfg.true_effect, abs=0.1)
    assert (df.value > 0).all() and (df.pre_value > 0).all()
    assert abs(df.value.corr(df.pre_value)) > 0.4


def test_dropout_removes_treated_users():
    df = simulate_experiment(ExperimentConfig(n_users=50_000, srm_dropout=0.2, seed=2))
    assert (df.arm == "treatment").sum() < 0.9 * (df.arm == "control").sum()


def test_invalid_config():
    with pytest.raises(ValueError):
        simulate_experiment(ExperimentConfig(treatment_share=1.2))


def test_sample_size_known_value():
    # classic result: d=0.2, alpha=.05, power=.8 -> ~393 per arm
    assert sample_size_per_arm(0.2, 1.0) == pytest.approx(393, abs=2)


def test_sample_size_monotonic_and_cuped():
    assert sample_size_per_arm(0.5, 10) > sample_size_per_arm(1.0, 10)
    base = sample_size_per_arm(0.5, 10)
    assert sample_size_per_arm(0.5, 10, cuped_corr=0.6) == pytest.approx(base * (1 - 0.36), rel=0.02)


def test_mde_inverts_sample_size_and_power():
    n = sample_size_per_arm(0.5, 10)
    assert minimum_detectable_effect(n, 10) == pytest.approx(0.5, rel=0.01)
    assert power_at(0.5, n, 10) == pytest.approx(0.8, abs=0.01)


def test_proportion_sample_size():
    assert sample_size_proportion(0.10, 0.12) == pytest.approx(3841, rel=0.02)
    with pytest.raises(ValueError):
        sample_size_proportion(0.1, 0.1)


def test_empirical_power_matches_calculator():
    cfg = ExperimentConfig(relative_lift=0.05, n_users=1, seed=0)
    sd = 0.6 * 18  # approximately; verified via a pilot draw below
    pilot = simulate_experiment(ExperimentConfig(n_users=200_000, seed=9)).value.std()
    effect = 0.05 * 18
    n = sample_size_per_arm(effect, pilot)
    from liftlab.analysis import analyze
    hits = 0
    for i in range(200):
        df = simulate_experiment(ExperimentConfig(n_users=2 * n, relative_lift=0.05, seed=100 + i))
        hits += analyze(df, "ttest").significant
    assert 0.68 < hits / 200 < 0.92
