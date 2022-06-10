"""Guardrails: sample-ratio-mismatch check and A/A false-positive-rate simulation."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
from scipy import stats

from .analysis import analyze
from .simulate import ExperimentConfig, simulate_experiment

SRM_THRESHOLD = 0.001  # conventional, deliberately strict: SRM means the data is broken


@dataclass(frozen=True)
class SRMResult:
    n_control: int
    n_treatment: int
    expected_share: float
    observed_share: float
    chi2: float
    p_value: float
    mismatch: bool


def srm_check(n_control: int, n_treatment: int, expected_share: float = 0.5,
              threshold: float = SRM_THRESHOLD) -> SRMResult:
    """Chi-square goodness-of-fit of arm sizes against the planned treatment share."""
    n = n_control + n_treatment
    if n == 0:
        raise ValueError("no users")
    expected = [n * (1 - expected_share), n * expected_share]
    chi2, p = stats.chisquare([n_control, n_treatment], expected)
    return SRMResult(n_control, n_treatment, expected_share, n_treatment / n, float(chi2),
                     float(p), bool(p < threshold))


def srm_check_df(df: pd.DataFrame, expected_share: float = 0.5, **kw) -> SRMResult:
    counts = df["arm"].value_counts()
    return srm_check(int(counts.get("control", 0)), int(counts.get("treatment", 0)),
                     expected_share, **kw)


@dataclass(frozen=True)
class AAResult:
    method: str
    n_simulations: int
    alpha: float
    false_positive_rate: float
    ci_low: float     # 95% binomial interval on the observed rate
    ci_high: float
    consistent_with_alpha: bool


def aa_simulation(method: str = "ttest", n_simulations: int = 500, alpha: float = 0.05,
                  config: ExperimentConfig | None = None, seed: int = 0,
                  n_resamples: int = 500, engine: str = "auto") -> AAResult:
    """Run many experiments with zero true effect; the rejection rate should be ~alpha."""
    base = config or ExperimentConfig(n_users=4000)
    rejections = 0
    for i in range(n_simulations):
        cfg = replace(base, relative_lift=0.0, srm_dropout=0.0, seed=seed * 1_000_003 + i)
        res = analyze(simulate_experiment(cfg), method=method, alpha=alpha,
                      n_resamples=n_resamples, seed=i, engine=engine)
        rejections += res.significant
    rate = rejections / n_simulations
    ci = stats.binomtest(rejections, n_simulations).proportion_ci(0.95, method="wilson")
    return AAResult(method, n_simulations, alpha, rate, float(ci.low), float(ci.high),
                    bool(ci.low <= alpha <= ci.high))
