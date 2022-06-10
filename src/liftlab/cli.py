"""Command line entry point: `liftlab demo | power | aa | bench | app`."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .analysis import METHODS, analyze
from .config import Settings
from .guardrails import aa_simulation, srm_check_df
from .power import minimum_detectable_effect, sample_size_per_arm
from .report import recommend
from .simulate import ExperimentConfig, simulate_experiment


def main(argv: list[str] | None = None) -> int:
    s = Settings.from_env()
    p = argparse.ArgumentParser(prog="liftlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="simulate one experiment and print the recommendation")
    d.add_argument("--lift", type=float, default=0.04)
    d.add_argument("--users", type=int, default=20_000)
    d.add_argument("--method", choices=METHODS, default="cuped")
    d.add_argument("--seed", type=int, default=s.seed)

    pw = sub.add_parser("power", help="sample size / MDE calculator")
    pw.add_argument("--baseline", type=float, default=18.0, help="mean order value")
    pw.add_argument("--sd", type=float, default=10.8, help="std dev of order value")
    pw.add_argument("--lift", type=float, default=0.03, help="relative lift to detect")
    pw.add_argument("--corr", type=float, default=0.0, help="pre/post correlation for CUPED")
    pw.add_argument("--power", type=float, default=0.8)

    a = sub.add_parser("aa", help="A/A false-positive-rate simulation")
    a.add_argument("--method", choices=METHODS, default="ttest")
    a.add_argument("--sims", type=int, default=500)

    b = sub.add_parser("bench", help="benchmark C++ vs NumPy resampling")
    b.add_argument("--users", type=int, default=20_000)
    b.add_argument("--resamples", type=int, default=2000)

    sub.add_parser("app", help="launch the Streamlit report")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        df = simulate_experiment(ExperimentConfig(n_users=args.users, relative_lift=args.lift, seed=args.seed))
        res = analyze(df, args.method, alpha=s.alpha, seed=args.seed, threads=s.threads)
        print(recommend(res, srm_check_df(df), sd=float(df["value"].std())).markdown())
    elif args.cmd == "power":
        eff = args.lift * args.baseline
        n = sample_size_per_arm(eff, args.sd, power=args.power, alpha=s.alpha, cuped_corr=args.corr)
        mde = minimum_detectable_effect(n, args.sd, power=args.power, alpha=s.alpha, cuped_corr=args.corr)
        print(f"Detect {args.lift:.1%} lift (${eff:.2f}) at power {args.power:.0%}: "
              f"{n:,} users per arm ({2 * n:,} total). MDE at that size: ${mde:.2f}")
    elif args.cmd == "aa":
        r = aa_simulation(args.method, args.sims, alpha=s.alpha)
        print(f"{r.method}: false-positive rate {r.false_positive_rate:.3f} over {r.n_simulations} A/A "
              f"tests (95% CI {r.ci_low:.3f}-{r.ci_high:.3f}, alpha={r.alpha}) -> "
              f"{'OK' if r.consistent_with_alpha else 'MISCALIBRATED'}")
    elif args.cmd == "bench":
        from .benchmark import run_benchmark
        out = run_benchmark(args.users, args.resamples)
        print(f"{out.attrs['n_users']:,} users, {out.attrs['n_resamples']:,} resamples, "
              f"{out.attrs['cores']} cores\n")
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    elif args.cmd == "app":
        app = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
