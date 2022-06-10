"""Seeded simulator of restaurant experiments.

Scenario: a checkout-flow change is tested on order value. Each user has a pre-experiment
average order value (the CUPED covariate) and an in-experiment one. Both are log-normal
(right-skewed, like real spend) and correlated through a Gaussian copula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    n_users: int = 20_000
    treatment_share: float = 0.5
    baseline_mean: float = 18.0      # mean order value in dollars
    cv: float = 0.6                  # coefficient of variation of order value
    relative_lift: float = 0.0       # true multiplicative effect on treated users
    pre_post_corr: float = 0.6       # latent correlation between pre and in-experiment spend
    srm_dropout: float = 0.0         # fraction of treated users silently lost (simulates a logging bug)
    seed: int = 42

    @property
    def true_effect(self) -> float:
        """True absolute difference in mean order value (treatment - control)."""
        return self.relative_lift * self.baseline_mean


def simulate_experiment(cfg: ExperimentConfig = ExperimentConfig()) -> pd.DataFrame:
    """Return one row per user: user_id, arm ('control'/'treatment'), pre_value, value."""
    if not 0 < cfg.treatment_share < 1:
        raise ValueError("treatment_share must be in (0, 1)")
    if not -1 < cfg.pre_post_corr < 1:
        raise ValueError("pre_post_corr must be in (-1, 1)")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users
    sigma = np.sqrt(np.log1p(cfg.cv**2))
    mu = np.log(cfg.baseline_mean) - sigma**2 / 2

    z_pre = rng.standard_normal(n)
    z_post = cfg.pre_post_corr * z_pre + np.sqrt(1 - cfg.pre_post_corr**2) * rng.standard_normal(n)
    treated = rng.random(n) < cfg.treatment_share
    pre = np.exp(mu + sigma * z_pre)
    post = np.exp(mu + sigma * z_post) * np.where(treated, 1.0 + cfg.relative_lift, 1.0)

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "arm": np.where(treated, "treatment", "control"),
        "pre_value": pre,
        "value": post,
    })
    if cfg.srm_dropout > 0:
        lost = treated & (rng.random(n) < cfg.srm_dropout)
        df = df[~lost].reset_index(drop=True)
    return df
