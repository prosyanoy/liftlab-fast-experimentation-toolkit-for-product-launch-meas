"""Effect estimation: Welch t-test, CUPED, bootstrap and permutation inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import resampling

METHODS = ("ttest", "cuped", "bootstrap", "permutation")


@dataclass(frozen=True)
class AnalysisResult:
    method: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    effect: float            # absolute difference, treatment - control
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float
    relative_lift: float     # effect / control_mean
    rel_ci_low: float
    rel_ci_high: float
    variance_reduction: float = 0.0   # CUPED only: 1 - Var(adj)/Var(raw)
    engine: str = ""

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def as_dict(self) -> dict:
        return asdict(self)


def split_arms(df: pd.DataFrame, outcome: str = "value", arm: str = "arm"):
    c = df.loc[df[arm] == "control", outcome].to_numpy(dtype=float)
    t = df.loc[df[arm] == "treatment", outcome].to_numpy(dtype=float)
    if c.size < 2 or t.size < 2:
        raise ValueError("need at least 2 users in each arm")
    return c, t


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (y - theta * (x - mean(x)), theta) with theta = cov(y, x) / var(x).

    theta is estimated on the pooled data, which is independent of assignment, so the
    adjusted difference in means stays unbiased.
    """
    xc = x - x.mean()
    theta = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
    return y - theta * xc, theta


def _welch(c: np.ndarray, t: np.ndarray, alpha: float):
    diff = t.mean() - c.mean()
    vc, vt = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    se = np.sqrt(vc + vt)
    df = (vc + vt) ** 2 / (vc**2 / (c.size - 1) + vt**2 / (t.size - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    p = 2 * stats.t.sf(abs(diff) / se, df)
    return diff, diff - crit * se, diff + crit * se, float(p)


def analyze(df: pd.DataFrame, method: str = "ttest", alpha: float = 0.05, outcome: str = "value",
            covariate: str = "pre_value", n_resamples: int = 5000, seed: int = 0,
            engine: str = "auto", threads: int = 0) -> AnalysisResult:
    """Estimate the treatment effect on `outcome` with the chosen method."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    c_raw, t_raw = split_arms(df, outcome)
    c, t = c_raw, t_raw
    vr, used_engine = 0.0, ""

    if method == "cuped":
        if covariate not in df:
            raise ValueError(f"CUPED needs a '{covariate}' column")
        mask_c, mask_t = (df["arm"] == "control").to_numpy(), (df["arm"] == "treatment").to_numpy()
        y, x = df[outcome].to_numpy(float), df[covariate].to_numpy(float)
        adj, _ = cuped_adjust(y[mask_c | mask_t], x[mask_c | mask_t])
        in_c = mask_c[mask_c | mask_t]
        c, t = adj[in_c], adj[~in_c]
        vr = float(1 - adj.var(ddof=1) / y[mask_c | mask_t].var(ddof=1))

    if method in ("ttest", "cuped"):
        effect, lo, hi, p = _welch(c, t, alpha)
    elif method == "bootstrap":
        used_engine = resampling._pick(engine)
        boots = resampling.bootstrap_mean_diff(c, t, n_resamples, seed, engine, threads)
        effect = float(t.mean() - c.mean())
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        # two-sided p-value from the bootstrap distribution's mass on each side of zero
        p = float(min(1.0, 2 * (min((boots <= 0).mean(), (boots >= 0).mean()) + 1 / (n_resamples + 1))))
    else:  # permutation
        used_engine = resampling._pick(engine)
        null = resampling.permutation_mean_diff(c, t, n_resamples, seed, engine, threads)
        effect = float(t.mean() - c.mean())
        p = float((1 + np.sum(np.abs(null) >= abs(effect) - 1e-12)) / (n_resamples + 1))
        # shift-model CI: observed effect minus the null quantiles
        q_lo, q_hi = np.quantile(null, [alpha / 2, 1 - alpha / 2])
        lo, hi = effect - q_hi, effect - q_lo

    base = float(c_raw.mean())
    return AnalysisResult(
        method=method, n_control=c_raw.size, n_treatment=t_raw.size, control_mean=base,
        treatment_mean=float(t_raw.mean()), effect=float(effect), ci_low=float(lo), ci_high=float(hi),
        p_value=float(p), alpha=alpha, relative_lift=float(effect / base),
        rel_ci_low=float(lo / base), rel_ci_high=float(hi / base), variance_reduction=vr,
        engine=used_engine,
    )
