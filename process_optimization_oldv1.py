# app.py
# Process Parameter Optimization (per part_number + machine_number)
# Repo preset: chad-k/Process-Optimization
# - Default: use repo files in /data
# - Optional: upload CSVs instead
# - Auto-suggest bounds from historical data (editable)
# - Choose models (LR / RF / SVR), tune tolerance
# - Download optimized_machine_settings.csv + failures.csv

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.optimize import minimize

st.set_page_config(page_title="Process Optimization Dashboard", layout="wide")

# ------------------------------
# PRESET REPO PATHS
# ------------------------------
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
REPO_DATA_PATH = DATA_DIR / "synthetic_process_data.csv"
REPO_SPECS_PATH = DATA_DIR / "spec_limits.csv"

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def _finite_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s[np.isfinite(s)].dropna()

def suggest_bounds(
    s: pd.Series,
    q_low: float = 0.05,
    q_high: float = 0.95,
    pad_frac: float = 0.10,
    hard_min: float | None = None,
    hard_max: float | None = None,
):
    """
    Suggest bounds from historical data:
      - base window = [q_low, q_high] quantiles
      - pad by pad_frac * (q_high - q_low)
      - optionally clamp to hard_min/hard_max
    Returns: (lo, hi, meta_dict)
    """
    s = _finite_series(s)
    if len(s) < 20:
        lo = float(s.min()) if len(s) else 0.0
        hi = float(s.max()) if len(s) else 1.0
        span = max(hi - lo, 1e-9)
        lo -= pad_frac * span
        hi += pad_frac * span
        src = "fallback(min/max)"
    else:
        lo_q = float(s.quantile(q_low))
        hi_q = float(s.quantile(q_high))
        span = max(hi_q - lo_q, 1e-9)
        lo = lo_q - pad_frac * span
        hi = hi_q + pad_frac * span
        src = f"quantiles({q_low:.0%}-{q_high:.0%})+pad({pad_frac:.0%})"

    if hard_min is not None:
        lo = max(lo, float(hard_min))
    if hard_max is not None:
        hi = min(hi, float(hard_max))

    if lo >= hi:
        hi = lo + 1.0

    meta = {"source": src, "n": int(len(s))}
    return float(lo), float(hi), meta

def optimize_parameters(model, X, y, target, bounds):
    """
    Fit model to (X,y), then find params within bounds that minimize squared error to target.
    Returns (optimized_params, predicted_at_optimized) or (None, None)
    """
    model.fit(X, y)

    def objective(params):
        pred = model.predict(np.array(params, dtype=float).reshape(1, -1))[0]
        return float((pred - target) ** 2)

    initial_guess = [float((b[0] + b[1]) / 2.0) for b in bounds]
    result = minimize(objective, x0=initial_guess, bounds=bounds)

    if result.success:
        optimized = np.array(result.x, dtype=float)
        predicted = float(model.predict(optimized.reshape(1, -1))[0])
        return optimized, predicted

    return None, None

def build_model_dict(
    use_lr: bool,
    use_rf: bool,
    use_svr: bool,
    rf_estimators: int,
    rf_random_state: int,
    svr_kernel: str,
    svr_c: float,
    svr_gamma: str,
    svr_epsilon: float
):
    models = {}
    if use_lr:
        models["LinearRegression"] = LinearRegression()
    if use_rf:
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=int(rf_estimators),
            random_state=int(rf_random_state),
            n_jobs=-1
        )
    if use_svr:
        models["SVR"] = SVR(
            kernel=svr_kernel,
            C=float(svr_c),
            gamma=svr_gamma,
            epsilon=float(svr_epsilon)
        )
    return models

def compute_results(
    data: pd.DataFrame,
    specs: pd.DataFrame,
    bounds,
    soft_tol_percent: float,
    model_choices: dict,
    feature_cols=("temperature", "pressure", "speed"),
    group_cols=("part_number", "machine_number"),
):
    missing = [c for c in list(feature_cols) + ["measurement"] + list(group_cols) if c not in data.columns]
    if missing:
        raise ValueError(f"Data file is missing columns: {missing}")

    if "part_number" not in specs.columns:
        raise ValueError("Specs file must contain column 'part_number'.")

    df = data.merge(specs, on="part_number", how="left")

    for required in ["target", "lower_spec_limit", "upper_spec_limit"]:
        if required not in df.columns:
            raise ValueError(
                f"After merge, missing '{required}'. Specs must contain: "
                f"part_number, target, lower_spec_limit, upper_spec_limit"
            )

    results = []
    failures = []

    for (part, machine), group in df.groupby(list(group_cols)):
        group = group.dropna(subset=list(feature_cols) + ["measurement"])
        if len(group) < 10:
            failures.append((part, machine, "Too few rows after cleaning (<10)"))
            continue

        X = group[list(feature_cols)].values
        y = group["measurement"].values

        target = float(group["target"].iloc[0]) if pd.notna(group["target"].iloc[0]) else np.nan
        lsl = float(group["lower_spec_limit"].iloc[0]) if pd.notna(group["lower_spec_limit"].iloc[0]) else np.nan
        usl = float(group["upper_spec_limit"].iloc[0]) if pd.notna(group["upper_spec_limit"].iloc[0]) else np.nan

        if not np.isfinite(target):
            failures.append((part, machine, "Missing/invalid target"))
            continue

        best_result = None
        lowest_error = float("inf")

        for model_name, model in model_choices.items():
            optimized, predicted = optimize_parameters(model, X, y, target, bounds)
            if optimized is None:
                continue

            error = abs(predicted - target)
            if error < lowest_error:
                lowest_error = error

                soft_lower = target * (1.0 - soft_tol_percent)
                soft_upper = target * (1.0 + soft_tol_percent)
                soft_pass = bool(soft_lower <= predicted <= soft_upper)

                if np.isfinite(lsl) and np.isfinite(usl):
                    in_spec = bool(lsl <= predicted <= usl)
                elif np.isfinite(lsl):
                    in_spec = bool(predicted >= lsl)
                elif np.isfinite(usl):
                    in_spec = bool(predicted <= usl)
                else:
                    in_spec = True

                best_result = {
                    "part_number": part,
                    "machine_number": machine,
                    "model_type": model_name,
                    "optimized_temperature": round(float(optimized[0]), 2),
                    "optimized_pressure": round(float(optimized[1]), 2),
                    "optimized_speed": round(float(optimized[2]), 2),
                    "predicted_measurement": round(float(predicted), 6),
                    "target": float(target),
                    "lower_spec_limit": lsl if np.isfinite(lsl) else np.nan,
                    "upper_spec_limit": usl if np.isfinite(usl) else np.nan,
                    "abs_error": round(float(error), 6),
                    "percent_error": round(float(100.0 * error / target), 4) if target != 0 else np.nan,
                    "in_spec": in_spec,
                    "soft_pass": soft_pass,
                    "rows_used": int(len(group)),
                }

        if best_result:
            results.append(best_result)
        else:
            failures.append((part, machine, "Optimization failed for all selected models"))

    results_df = pd.DataFrame(results).sort_values(["part_number", "machine_number"], kind="stable") if results else pd.DataFrame()
    failures_df = pd.DataFrame(failures, columns=["part_number", "machine_number", "reason"]) if failures else pd.DataFrame()
    return results_df, failures_df


# ------------------------------
# UI
# ------------------------------
st.title("Process Parameter Optimization")
st.caption("Optimize temperature / pressure / speed to hit target measurement (per part + machine).")

with st.sidebar:
    st.header("Input files")

    mode = st.radio(
        "Load from",
        ["Use repo data (default)", "Upload CSVs"],
        index=0
    )

    data_up = specs_up = None
    if mode == "Upload CSVs":
        data_up = st.file_uploader("Upload synthetic_process_data.csv", type=["csv"])
        specs_up = st.file_uploader("Upload spec_limits.csv", type=["csv"])
    else:
        st.success("Using preset repo paths")
        st.code("data/synthetic_process_data.csv")
        st.code("data/spec_limits.csv")

    st.divider()
    st.header("Models")

    use_lr = st.checkbox("LinearRegression", value=True)
    use_rf = st.checkbox("RandomForestRegressor", value=True)
    use_svr = st.checkbox("SVR", value=False)

    if use_rf:
        rf_estimators = st.slider("RF n_estimators", 50, 600, 200, 50)
        rf_random_state = st.number_input("RF random_state", value=42, step=1)
    else:
        rf_estimators = 200
        rf_random_state = 42

    if use_svr:
        svr_kernel = st.selectbox("SVR kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        svr_c = st.number_input("SVR C", value=10.0, step=1.0)
        svr_gamma = st.selectbox("SVR gamma", ["scale", "auto"], index=0)
        svr_epsilon = st.number_input("SVR epsilon", value=0.1, step=0.05)
    else:
        svr_kernel = "rbf"
        svr_c = 10.0
        svr_gamma = "scale"
        svr_epsilon = 0.1

    st.divider()
    st.header("Targets")
    soft_tol_percent_ui = st.slider("Soft tolerance (±%) around target", 0.0, 50.0, 10.0, 0.5)

    st.divider()
    run_btn = st.button("Run optimization", type="primary")

model_choices = build_model_dict(
    use_lr=use_lr,
    use_rf=use_rf,
    use_svr=use_svr,
    rf_estimators=rf_estimators,
    rf_random_state=rf_random_state,
    svr_kernel=svr_kernel,
    svr_c=svr_c,
    svr_gamma=svr_gamma,
    svr_epsilon=svr_epsilon
)

if not model_choices:
    st.warning("Select at least one model.")
    st.stop()

# ------------------------------
# Load data (needed before bounds suggestion)
# ------------------------------
with st.spinner("Loading input data..."):
    try:
        if mode == "Upload CSVs":
            if data_up is None or specs_up is None:
                st.info("Upload both CSV files, then click **Run optimization**.")
                st.stop()
            data_df = read_csv_bytes(data_up.getvalue())
            specs_df = read_csv_bytes(specs_up.getvalue())
        else:
            if (not REPO_DATA_PATH.exists()) or (not REPO_SPECS_PATH.exists()):
                st.error("Repo files not found in the deployed filesystem.")
                st.write("Expected:", str(REPO_DATA_PATH))
                st.write("Expected:", str(REPO_SPECS_PATH))
                st.write("Does /data exist?", DATA_DIR.exists(), "->", str(DATA_DIR))
                if DATA_DIR.exists():
                    st.write("Files in /data:", [p.name for p in DATA_DIR.iterdir()])
                st.stop()
            data_df = pd.read_csv(REPO_DATA_PATH)
            specs_df = pd.read_csv(REPO_SPECS_PATH)

    except Exception as e:
        st.error(f"File load failed: {e}")
        st.stop()

# ------------------------------
# Bounds (auto-suggest from historical data, editable)
# ------------------------------
with st.sidebar:
    st.header("Optimization bounds")

    auto_bounds = st.checkbox("Auto-suggest bounds from historical data", value=True)
    q_low = st.slider("Lower quantile", 0.0, 0.20, 0.05, 0.01)
    q_high = st.slider("Upper quantile", 0.80, 1.0, 0.95, 0.01)
    pad_frac = st.slider("Padding (% of span)", 0.0, 0.50, 0.10, 0.01)

    # Optional: suggest bounds per part/machine
    per_group = st.checkbox("Suggest bounds per selected Part+Machine", value=False)

    if per_group and ("part_number" in data_df.columns) and ("machine_number" in data_df.columns):
        parts = sorted(data_df["part_number"].dropna().astype(str).unique().tolist())
        machines = sorted(data_df["machine_number"].dropna().astype(str).unique().tolist())
        sel_part = st.selectbox("Part for bound suggestion", parts, index=0 if parts else 0)
        sel_machine = st.selectbox("Machine for bound suggestion", machines, index=0 if machines else 0)
        hist_df = data_df[(data_df["part_number"].astype(str) == str(sel_part)) &
                          (data_df["machine_number"].astype(str) == str(sel_machine))]
        if hist_df.empty:
            hist_df = data_df
            st.warning("No rows for selected Part+Machine; using all data for suggestion.")
    else:
        hist_df = data_df

    if auto_bounds:
        t_lo, t_hi, t_meta = suggest_bounds(hist_df.get("temperature", pd.Series(dtype=float)), q_low=q_low, q_high=q_high, pad_frac=pad_frac)
        p_lo, p_hi, p_meta = suggest_bounds(hist_df.get("pressure", pd.Series(dtype=float)), q_low=q_low, q_high=q_high, pad_frac=pad_frac)
        s_lo, s_hi, s_meta = suggest_bounds(hist_df.get("speed", pd.Series(dtype=float)), q_low=q_low, q_high=q_high, pad_frac=pad_frac)
    else:
        t_lo, t_hi = 170.0, 200.0
        p_lo, p_hi = 45.0, 60.0
        s_lo, s_hi = 1100.0, 1300.0
        t_meta = p_meta = s_meta = {"source": "manual defaults", "n": int(len(hist_df))}

    with st.expander("How were these bounds suggested?"):
        st.write(f"Temperature: {t_meta['source']} (n={t_meta['n']})")
        st.write(f"Pressure:    {p_meta['source']} (n={p_meta['n']})")
        st.write(f"Speed:       {s_meta['source']} (n={s_meta['n']})")

    c1, c2 = st.columns(2)
    with c1:
        t_min = st.number_input("Temp min", value=float(t_lo), step=1.0)
        p_min = st.number_input("Pressure min", value=float(p_lo), step=0.5)
        s_min = st.number_input("Speed min", value=float(s_lo), step=10.0)
    with c2:
        t_max = st.number_input("Temp max", value=float(t_hi), step=1.0)
        p_max = st.number_input("Pressure max", value=float(p_hi), step=0.5)
        s_max = st.number_input("Speed max", value=float(s_hi), step=10.0)

# Validate bounds
if t_min >= t_max or p_min >= p_max or s_min >= s_max:
    st.error("Invalid bounds: each min must be < max.")
    st.stop()

bounds = [(float(t_min), float(t_max)), (float(p_min), float(p_max)), (float(s_min), float(s_max))]
soft_tol_percent = float(soft_tol_percent_ui) / 100.0

# Stop until user runs
if not run_btn:
    st.info("Bounds/models loaded. Click **Run optimization** to compute.")
    st.stop()

# ------------------------------
# Compute
# ------------------------------
with st.spinner("Optimizing per part + machine..."):
    try:
        results_df, failures_df = compute_results(
            data=data_df,
            specs=specs_df,
            bounds=bounds,
            soft_tol_percent=soft_tol_percent,
            model_choices=model_choices
        )
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()

# ------------------------------
# Summary
# ------------------------------
st.subheader("Summary")
total = len(results_df)
in_spec_count = int(results_df["in_spec"].sum()) if total else 0
soft_pass_count = int(results_df["soft_pass"].sum()) if total else 0
avg_error = float(results_df["abs_error"].mean()) if total else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimized groups", f"{total:,}")
c2.metric("In spec", f"{in_spec_count:,}")
c3.metric("Soft pass (±%)", f"{soft_pass_count:,}")
c4.metric("Avg abs error", f"{avg_error:.6f}" if np.isfinite(avg_error) else "NA")

tabs = st.tabs(["Results", "Failures", "Input Preview"])

with tabs[0]:
    st.subheader("Optimized Settings")
    if results_df.empty:
        st.warning("No results produced. Check data quality, bounds, and selected models.")
    else:
        st.dataframe(results_df, use_container_width=True, height=520)
        st.download_button(
            "Download optimized_machine_settings.csv",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="optimized_machine_settings.csv",
            mime="text/csv"
        )

with tabs[1]:
    st.subheader("Groups That Did Not Optimize")
    if failures_df.empty:
        st.success("No failures.")
    else:
        st.dataframe(failures_df, use_container_width=True, height=420)
        st.download_button(
            "Download optimization_failures.csv",
            data=failures_df.to_csv(index=False).encode("utf-8"),
            file_name="optimization_failures.csv",
            mime="text/csv"
        )

with tabs[2]:
    st.subheader("Data Preview")
    st.write("**synthetic_process_data.csv (head)**")
    st.dataframe(data_df.head(200), use_container_width=True)
    st.write("**spec_limits.csv (head)**")
    st.dataframe(specs_df.head(200), use_container_width=True)

st.caption(
    "Deploy tip: commit your CSVs into /data for demo mode, or use Upload mode for real/variable data. "
    "Auto-suggested bounds are just defaults—always verify they’re within safe process limits."
)
