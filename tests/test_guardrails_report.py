import pytest

from liftlab.analysis import analyze
from liftlab.guardrails import aa_simulation, srm_check, srm_check_df
from liftlab.report import DO_NOT_LAUNCH, HOLD, INVALID, LAUNCH, recommend
from liftlab.simulate import ExperimentConfig, simulate_experiment
from liftlab.store import arm_summary, connect, load_experiment, save_experiment


def test_srm_passes_on_balanced_and_fails_on_skew():
    assert not srm_check(5000, 5030).mismatch
    bad = srm_check(5000, 4500)
    assert bad.mismatch and bad.p_value < 1e-3
    assert not srm_check(3000, 7000, expected_share=0.7).mismatch


def test_srm_detects_simulated_logging_bug():
    ok = simulate_experiment(ExperimentConfig(n_users=40_000, seed=1))
    bug = simulate_experiment(ExperimentConfig(n_users=40_000, srm_dropout=0.05, seed=1))
    assert not srm_check_df(ok).mismatch
    assert srm_check_df(bug).mismatch


def test_srm_false_alarm_rate_is_low():
    alarms = sum(srm_check_df(simulate_experiment(ExperimentConfig(n_users=3000, seed=s))).mismatch
                 for s in range(300))
    assert alarms <= 2


@pytest.mark.parametrize("method", ["ttest", "cuped"])
def test_aa_false_positive_rate_near_alpha(method):
    r = aa_simulation(method, n_simulations=400, seed=1)
    assert r.consistent_with_alpha, r
    assert 0.02 < r.false_positive_rate < 0.085


def test_aa_permutation_with_cpp_is_calibrated():
    r = aa_simulation("permutation", n_simulations=300, n_resamples=300, seed=3)
    assert r.consistent_with_alpha, r
    assert 0.02 < r.false_positive_rate < 0.09


def _decide(lift, n=40_000, **kw):
    df = simulate_experiment(ExperimentConfig(n_users=n, relative_lift=lift, seed=3, **kw))
    return recommend(analyze(df, "cuped"), srm_check_df(df), sd=float(df.value.std()))


def test_recommendations():
    assert _decide(0.06).decision == LAUNCH
    assert _decide(-0.06).decision == DO_NOT_LAUNCH
    assert _decide(0.0, n=60_000).decision == DO_NOT_LAUNCH     # tight CI around zero
    hold = _decide(0.02, n=3000)
    assert hold.decision == HOLD and "per group" in hold.markdown()
    broken = _decide(0.06, srm_dropout=0.1)
    assert broken.decision == INVALID and "split" in broken.markdown()


def test_duckdb_roundtrip_and_summary():
    df = simulate_experiment(ExperimentConfig(n_users=5000, seed=1))
    con = connect(":memory:")
    save_experiment(con, "e1", df)
    save_experiment(con, "e1", df)  # idempotent
    back = load_experiment(con, "e1")
    assert len(back) == len(df)
    s = arm_summary(con, "e1").set_index("arm")
    assert s.loc["control", "mean_value"] == pytest.approx(df[df.arm == "control"].value.mean())
    assert s.users.sum() == len(df)
