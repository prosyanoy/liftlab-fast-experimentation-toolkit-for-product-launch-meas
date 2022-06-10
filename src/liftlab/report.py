"""Turn statistical output into a plain-English launch recommendation."""
from __future__ import annotations

from dataclasses import dataclass

from .analysis import AnalysisResult
from .guardrails import SRMResult
from .power import sample_size_per_arm

LAUNCH, HOLD, DO_NOT_LAUNCH, INVALID = "LAUNCH", "KEEP TESTING", "DO NOT LAUNCH", "DO NOT TRUST"


@dataclass(frozen=True)
class Recommendation:
    decision: str
    headline: str
    details: list[str]

    def markdown(self) -> str:
        return f"### {self.decision}: {self.headline}\n\n" + "\n".join(f"- {d}" for d in self.details)


def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def recommend(result: AnalysisResult, srm: SRMResult, min_lift: float = 0.02,
              sd: float | None = None) -> Recommendation:
    """Decide launch / hold / stop.

    `min_lift` is the smallest relative lift the business cares about (practical significance).
    `sd` (outcome standard deviation) lets us estimate how much more data an inconclusive test needs.
    """
    r = result
    facts = [
        f"{r.n_control:,} control and {r.n_treatment:,} treatment customers were measured.",
        f"Average order value: {_money(r.control_mean)} with the current checkout vs "
        f"{_money(r.treatment_mean)} with the new one.",
        f"Estimated change: {_money(r.effect)} per order ({_pct(r.relative_lift)}). "
        f"We are {100 * (1 - r.alpha):.0f}% confident the true change is between "
        f"{_money(r.ci_low)} ({_pct(r.rel_ci_low)}) and {_money(r.ci_high)} ({_pct(r.rel_ci_high)}).",
    ]
    if r.variance_reduction > 0:
        facts.append(f"Using each customer's past spending removed {r.variance_reduction:.0%} of the "
                     "noise, so the result is sharper than a plain comparison.")

    if srm.mismatch:
        return Recommendation(INVALID, "the experiment data looks broken", [
            f"Customers were meant to be split {1 - srm.expected_share:.0%}/{srm.expected_share:.0%}, but "
            f"{srm.n_control:,} vs {srm.n_treatment:,} landed in each group (chance of this by luck: "
            f"p = {srm.p_value:.2g}).",
            "This usually means customers are being lost or mis-assigned (a tracking or "
            "assignment bug), which can make any result misleading.",
            "Ask engineering to fix the split before reading anything into the numbers below.",
            *facts,
        ])

    if r.ci_low > 0 and r.rel_ci_low >= min_lift:
        return Recommendation(LAUNCH, "the new checkout clearly increases order value", [
            *facts, f"Even the cautious end of the range clears the {min_lift:.0%} lift the business needs."])
    if r.significant and r.effect > 0:
        return Recommendation(LAUNCH, "the new checkout increases order value", [
            *facts,
            f"The gain is real, but its low end ({_pct(r.rel_ci_low)}) is under the {min_lift:.0%} target; "
            "launch if the change is cheap to run and expect a modest benefit."])
    if r.significant and r.effect < 0:
        return Recommendation(DO_NOT_LAUNCH, "the new checkout reduces order value", [
            *facts, "The drop is statistically solid. Keep the current checkout."])
    if r.rel_ci_high < min_lift:
        return Recommendation(DO_NOT_LAUNCH, "no meaningful benefit", [
            *facts,
            f"No lift was detected, and even the optimistic end ({_pct(r.rel_ci_high)}) is below the "
            f"{min_lift:.0%} target. The change is not worth shipping for order value."])
    extra = []
    if sd:
        need = sample_size_per_arm(min_lift * r.control_mean, sd, alpha=r.alpha)
        extra.append(f"To reliably detect a {min_lift:.0%} lift we need about {need:,} customers per "
                     f"group; we currently have {min(r.n_control, r.n_treatment):,}.")
    return Recommendation(HOLD, "the result is not conclusive yet", [
        *facts, "The range includes both no change and a worthwhile gain, so we cannot tell yet.",
        *extra, "Keep the test running rather than deciding on this data."])
