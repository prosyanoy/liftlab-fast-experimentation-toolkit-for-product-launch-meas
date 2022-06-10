"""Regenerate docs/benchmark.png and print the numbers quoted in the README."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from liftlab.analysis import analyze
from liftlab.benchmark import run_benchmark
from liftlab.guardrails import aa_simulation
from liftlab.simulate import ExperimentConfig, simulate_experiment

out = Path(__file__).resolve().parents[1] / "docs"
out.mkdir(exist_ok=True)

bench = run_benchmark(50_000, 5_000, repeats=3)
boot = bench[bench.operation == "bootstrap"].reset_index(drop=True)
labels = ["NumPy", "C++ (1 thread)", "C++ (all cores)"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
bars = ax.barh(labels[::-1], boot.seconds[::-1], color=["#2a9d8f", "#2a9d8f", "#8d99ae"])
for b, s, sp in zip(bars, boot.seconds[::-1], boot.speedup_vs_numpy[::-1]):
    ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
            f"{s:.2f} s" + ("" if sp == 1 else f"  ({sp:.0f}x)"), va="center")
ax.set_xlabel("seconds (lower is better)")
ax.set_title(f"Bootstrap, 50,000 users x 5,000 resamples ({bench.attrs['cores']} cores)")
ax.set_xlim(0, boot.seconds.max() * 1.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "benchmark.png", dpi=150)
print(bench.to_string(index=False))

df = simulate_experiment(ExperimentConfig(n_users=100_000, relative_lift=0.04, seed=7))
p, c = analyze(df, "ttest"), analyze(df, "cuped")
w = lambda r: r.ci_high - r.ci_low
print(f"\nCUPED: corr={df.value.corr(df.pre_value):.3f} var_reduction={c.variance_reduction:.1%} "
      f"CI width {w(p):.3f} -> {w(c):.3f} ({1 - w(c) / w(p):.1%} narrower)")
for m in ("ttest", "cuped"):
    r = aa_simulation(m, 1000, seed=11)
    print(f"A/A {m}: {r.false_positive_rate:.3f} (95% CI {r.ci_low:.3f}-{r.ci_high:.3f})")
