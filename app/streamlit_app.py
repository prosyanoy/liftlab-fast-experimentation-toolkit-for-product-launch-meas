"""Streamlit launch report: simulate (or upload) an experiment and get a plain-English verdict."""
import pandas as pd
import streamlit as st

from liftlab import resampling
from liftlab.analysis import METHODS, analyze
from liftlab.config import Settings
from liftlab.guardrails import aa_simulation, srm_check_df
from liftlab.power import minimum_detectable_effect, sample_size_per_arm
from liftlab.report import recommend
from liftlab.simulate import ExperimentConfig, simulate_experiment
from liftlab.store import arm_summary, connect, save_experiment

settings = Settings.from_env()
st.set_page_config(page_title="LiftLab launch report", layout="wide")
st.title("LiftLab: checkout-flow launch report")

with st.sidebar:
    st.header("Experiment")
    source = st.radio("Data", ["Simulated demo", "Upload CSV"])
    if source == "Simulated demo":
        n_users = st.slider("Customers", 2_000, 100_000, 20_000, 1_000)
        lift = st.slider("True lift in order value (%)", -10.0, 10.0, 4.0, 0.5) / 100
        corr = st.slider("Past/present spend correlation", 0.0, 0.95, 0.6, 0.05)
        dropout = st.slider("Simulated tracking loss in treatment (%)", 0.0, 10.0, 0.0, 0.5) / 100
        seed = st.number_input("Seed", value=settings.seed, step=1)
        upload = None
    else:
        upload = st.file_uploader("CSV with columns arm (control/treatment), value[, pre_value]")
    st.header("Analysis")
    method = st.selectbox("Method", METHODS, index=1)
    min_lift = st.slider("Smallest lift worth shipping (%)", 0.0, 10.0, 2.0, 0.5) / 100
    expected_share = st.slider("Planned treatment share", 0.1, 0.9, 0.5, 0.05)
    engines = ["auto", "python"] + (["cpp"] if resampling.cpp_available() else [])
    engine = st.selectbox("Resampling engine", engines)

if source == "Upload CSV":
    if upload is None:
        st.info("Upload a CSV to analyse your own experiment, or switch to the simulated demo.")
        st.stop()
    df = pd.read_csv(upload)
    if not {"arm", "value"} <= set(df.columns):
        st.error("CSV needs `arm` and `value` columns.")
        st.stop()
    if "user_id" not in df:
        df["user_id"] = range(len(df))
    if method == "cuped" and "pre_value" not in df:
        st.error("CUPED needs a `pre_value` column; pick another method.")
        st.stop()
else:
    df = simulate_experiment(ExperimentConfig(n_users=n_users, relative_lift=lift, pre_post_corr=corr,
                                              srm_dropout=dropout, treatment_share=expected_share,
                                              seed=int(seed)))

try:
    srm = srm_check_df(df, expected_share)
    result = analyze(df, method, alpha=settings.alpha, engine=engine, threads=settings.threads)
except ValueError as e:
    st.error(str(e))
    st.stop()
rec = recommend(result, srm, min_lift=min_lift, sd=float(df["value"].std()))

banner = {"LAUNCH": st.success, "DO NOT LAUNCH": st.error, "KEEP TESTING": st.warning,
          "DO NOT TRUST": st.error}[rec.decision]
banner(f"**{rec.decision}** — {rec.headline}")
# "$" would start LaTeX math in Streamlit markdown, so escape it
st.markdown("\n".join(f"- {d}" for d in rec.details).replace("$", "\\$"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effect on order value", f"${result.effect:+.2f}", f"{result.relative_lift:+.1%}")
c2.metric(f"{1 - result.alpha:.0%} CI for change (USD)", f"{result.ci_low:.2f}–{result.ci_high:.2f}")
c3.metric("p-value", f"{result.p_value:.4f}")
c4.metric("Sample ratio check", "FAILED" if srm.mismatch else "passed", f"p = {srm.p_value:.3g}",
          delta_color="inverse" if srm.mismatch else "off")

st.subheader("Per-arm summary (DuckDB SQL)")
con = connect(":memory:")
save_experiment(con, "report", df)
st.dataframe(arm_summary(con, "report"), hide_index=True)

with st.expander("Plan a test: sample size and detectable effect"):
    sd = float(df["value"].std())
    base = float(df["value"].mean())
    target = st.slider("Lift to detect (%)", 0.5, 10.0, 3.0, 0.5) / 100
    rho = st.slider("CUPED correlation to assume", 0.0, 0.95, 0.0, 0.05)
    n = sample_size_per_arm(target * base, sd, alpha=settings.alpha, cuped_corr=rho)
    st.write((f"Detecting a **{target:.1%}** lift (${target * base:.2f}) with 80% power needs "
             f"**{n:,}** customers per group. With that sample the smallest detectable change is "
             f"${minimum_detectable_effect(n, sd, alpha=settings.alpha, cuped_corr=rho):.2f}.").replace("$", "\\$"))

with st.expander("Health check: A/A test false-positive rate"):
    sims = st.slider("Simulations", 100, 1000, 300, 100)
    if st.button("Run A/A simulation"):
        aa = aa_simulation(method if method in ("ttest", "cuped") else "ttest", sims,
                           alpha=settings.alpha)
        st.write(f"Rejected {aa.false_positive_rate:.1%} of {aa.n_simulations} experiments with no real "
                 f"effect (expected {aa.alpha:.0%}; 95% interval {aa.ci_low:.1%}–{aa.ci_high:.1%}). "
                 + ("The testing pipeline is well calibrated." if aa.consistent_with_alpha
                    else "The false-positive rate is off: investigate before trusting results."))
