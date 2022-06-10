"""Power and sample-size calculations (continuous metrics, with optional CUPED adjustment)."""
from __future__ import annotations

import math

from statsmodels.stats.power import NormalIndPower, TTestIndPower


def _effective_sd(sd: float, cuped_corr: float) -> float:
    """CUPED shrinks variance by (1 - rho^2), so the standard deviation by sqrt(1 - rho^2)."""
    if not -1 < cuped_corr < 1:
        raise ValueError("cuped_corr must be in (-1, 1)")
    return sd * math.sqrt(1.0 - cuped_corr**2)


def sample_size_per_arm(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8,
                        ratio: float = 1.0, cuped_corr: float = 0.0) -> int:
    """Users needed in the control arm to detect an absolute `effect` (two-sided t-test).

    `ratio` = treatment size / control size. `cuped_corr` is the pre/post correlation if CUPED
    will be used. Returns the control-arm size (treatment = ratio * control).
    """
    if effect == 0 or sd <= 0:
        raise ValueError("effect must be non-zero and sd positive")
    d = abs(effect) / _effective_sd(sd, cuped_corr)
    n = TTestIndPower().solve_power(effect_size=d, alpha=alpha, power=power, ratio=ratio,
                                    alternative="two-sided")
    return int(math.ceil(n))


def minimum_detectable_effect(n_per_arm: int, sd: float, alpha: float = 0.05, power: float = 0.8,
                              ratio: float = 1.0, cuped_corr: float = 0.0) -> float:
    """Smallest absolute effect detectable with `n_per_arm` control users."""
    d = TTestIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=ratio,
                                    alternative="two-sided")
    return float(d * _effective_sd(sd, cuped_corr))


def power_at(effect: float, n_per_arm: int, sd: float, alpha: float = 0.05, ratio: float = 1.0,
             cuped_corr: float = 0.0) -> float:
    """Probability of detecting `effect` with `n_per_arm` control users."""
    d = abs(effect) / _effective_sd(sd, cuped_corr)
    return float(TTestIndPower().power(effect_size=d, nobs1=n_per_arm, alpha=alpha, ratio=ratio,
                                       alternative="two-sided"))


def sample_size_proportion(p_control: float, p_treatment: float, alpha: float = 0.05,
                           power: float = 0.8, ratio: float = 1.0) -> int:
    """Control-arm size for a conversion-rate style metric (normal approximation)."""
    from statsmodels.stats.proportion import proportion_effectsize
    h = abs(proportion_effectsize(p_treatment, p_control))
    if h == 0:
        raise ValueError("proportions must differ")
    n = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio,
                                     alternative="two-sided")
    return int(math.ceil(n))
