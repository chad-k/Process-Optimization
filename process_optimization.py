# app.py
# Flexible Process Optimization (non-standard customer CSVs + subgroup Format C)
# Contact: chad@hertzler.com
#
# This version supports ONLY:
#   1) Single measurement column (one measurement per row)
#   2) Wide subgroup member columns (Format C: Data1..DataN / Sample1..SampleN, etc.)
#   3) Auto (Mixed) that prefers Wide -> Single
#
# (Long format with subgroup has been removed, per request.)
#
# Key features:
# - No hardcoded parameter names (temperature/pressure/speed not assumed)
# - Uploaded/repo files load ONLY when user clicks "Load / Refresh files" (persisted in session_state)
# - Subgroup "Format C" supported via Wide subgroup member columns (Data1..DataN, Sample1..SampleN, etc.)
# - Robust parameter attach (tries: __rid__ join; fallback: group-mean by Part+Machine)
# - Auto-suggest bounds from historical data, but always editable
# - Optimization can auto-run (optional) or run-on-click; results are cached in session_state
# - Help section + contact info

import io
import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.optimize import minimize

st.set_page_config(page_title="Flexible Process Optimization", layout="wide")

# ------------------------------
# PRESET REPO PATHS (optional demo mode)
# ------------------------------
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
REPO_DATA_PATH = DATA_DIR / "synthetic_process_data.csv"
REPO_SPECS_PATH = DATA_DIR / "spec_limits.csv"

CONTACT_EMAIL = "chad@hertzler.com"

# ------------------------------
# Heuristics / synonyms (best-effort)
# ------------------------------
SYNONYMS = {
    "part": ["part", "part_number", "partnumber", "part no", "partno", "pn", "construction", "item", "sku"],
    "machine": ["machine", "machine_number", "machinenumber", "machine no", "workcenter", "line", "cell", "asset"],
    "measurement": ["measurement", "meas", "value", "result", "output", "response", "ctq", "dimension", "y"],
    "target": ["target", "tgt", "nominal", "center", "aim", "target_value"],
    "lsl": ["lsl", "lower_spec_limit", "lower spec limit", "lower", "min", "lo spec", "low spec", "min spec"],
    "usl": ["usl", "upper_spec_limit", "upper spec limit", "upper", "max", "hi spec", "high spec", "max spec"],
}

# Common wide subgroup member patterns (Format C often lands here)
WIDE_PATTERNS_DEFAULT = [
    r"^data[\s\-_]*\d+$",        # Data 1, data_1, data-1, data1
    r"^meas[\s\-_]*\d+$",        # Meas1
    r"^x[\s\-_]*\d+$",           # X1
    r"^sample[\s\-_]*\d+$",      # Sample1
    r"^reading[\s\-_]*\d+$",     # Reading1
    r"^value[\s\-_]*\d+$",       # Value1
]

MEAS_MODES = [
    "Auto (Mixed)",
    "Single measurement column",
    "Wide subgroup member columns (Format C / Data1..DataN)",
]

# ------------------------------
# Caching helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

# ------------------------------
# Utility helpers
# ------------------------------
def norm_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s).strip().lower())

def guess_col(df: pd.DataFrame, key: str) -> Optional[str]:
    cols = list(df.columns)
    cols_norm = {c: norm_name(c) for c in cols}

    keyn = norm_name(key)
    for c in cols:
        if cols_norm[c] == keyn:
            return c

    for syn in SYNONYMS.get(key, []):
        synn = norm_name(syn)
        for c in cols:
            if cols_norm[c] == synn:
                return c
    return None

def detect_wide_measure_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        cn = str(c).strip().lower()
        for pat in patterns:
            if re.match(pat, cn):
                cols.append(c)
                break

    def trailing_num(colname: str) -> int:
        m = re.search(r"(\d+)\s*$", str(colname).strip())
        return int(m.group(1)) if m else 10**9

    return sorted(cols, key=trailing_num)

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _finite_series(s: pd.Series) -> pd.Series:
    s = coerce_numeric(s)
    return s[np.isfinite(s)].dropna()

def suggest_bounds(
    s: pd.Series,
    q_low: float = 0.05,
    q_high: float = 0.95,
    pad_frac: float = 0.10,
    hard_min: Optional[float] = None,
    hard_max: Optional[float] = None,
) -> Tuple[float, float, Dict]:
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

    return float(lo), float(hi), {"source": src, "n": int(len(s))}

def detect_parameter_candidates(df: pd.DataFrame, exclude_cols: set) -> List[str]:
    candidates = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        s = coerce_numeric(df[c])
        if len(s) == 0:
            continue
        if (s.notna().mean() >= 0.70) and (s.notna().sum() >= 10):
            candidates.append(c)
    return candidates

def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"nan": ""}).fillna("").astype(str)

# ------------------------------
# Mapping template support (save/load)
# ------------------------------
def mapping_payload(state: dict) -> Dict:
    return {
        "version": 2,
        "contact": CONTACT_EMAIL,
        "data_mapping": {
            "part_col": state.get("part_col"),
            "machine_col": state.get("machine_col"),
            "measurement_mode": state.get("measurement_mode"),
            "agg_func": state.get("agg_func"),
            "measurement_col": state.get("measurement_col"),
            "wide_cols": state.get("wide_cols", []),
            "wide_patterns": state.get("wide_patterns", WIDE_PATTERNS_DEFAULT),
        },
        "specs_mapping": {
            "specs_part_col": state.get("specs_part_col"),
            "target_col": state.get("target_col"),
            "lsl_col": state.get("lsl_col"),
            "usl_col": state.get("usl_col"),
        },
        "parameters": {
            "param_cols": state.get("param_cols", []),
        },
        "notes": "This template supports Single + Wide subgroup (Format C). Long format removed.",
    }

# ------------------------------
# Build model-ready dataset (Single/Wide + specs)
# ------------------------------
def build_model_df(
    data_df: pd.DataFrame,
    specs_df: pd.DataFrame,
    part_col: str,
    machine_col: str,
    specs_part_col: str,
    target_col: str,
    lsl_col: Optional[str],
    usl_col: Optional[str],
    measurement_mode: str,
    agg_func: str,
    measurement_col: Optional[str],
    wide_cols: Optional[List[str]],
    wide_patterns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    df = data_df.copy()

    # Ensure stable row id exists
    if "__rid__" not in df.columns:
        df["__rid__"] = np.arange(len(df), dtype=int)

    meta = {"rows_before": len(df), "measurement_mode_used": None}

    df["part_key"] = safe_str_series(df[part_col]).str.strip()
    df["machine_key"] = safe_str_series(df[machine_col]).str.strip()

    used_mode = measurement_mode

    # Auto mode: prefer Wide -> Single
    if measurement_mode == "Auto (Mixed)":
        patterns = wide_patterns or WIDE_PATTERNS_DEFAULT
        auto_wide = wide_cols if wide_cols else detect_wide_measure_cols(df, patterns)
        has_wide = len(auto_wide) >= 2
        has_single = measurement_col is not None and measurement_col in df.columns

        if has_wide:
            used_mode = "Wide subgroup member columns (Format C / Data1..DataN)"
            wide_cols = auto_wide
        elif has_single:
            used_mode = "Single measurement column"
        else:
            raise ValueError("Auto mode could not detect Wide subgroup columns or a measurement column. Choose a mode manually.")

    if used_mode == "Single measurement column":
        if not measurement_col or measurement_col not in df.columns:
            raise ValueError("Single measurement mode requires selecting a measurement column.")
        df["y"] = coerce_numeric(df[measurement_col])
        meta["measurement_mode_used"] = used_mode
        meta["measurement_col_used"] = measurement_col

    elif used_mode == "Wide subgroup member columns (Format C / Data1..DataN)":
        if not wide_cols or len(wide_cols) < 2:
            raise ValueError("Wide subgroup mode (Format C) requires selecting at least 2 subgroup member columns.")
        vals = df[wide_cols].apply(pd.to_numeric, errors="coerce")
        df["y"] = vals.median(axis=1) if agg_func == "median" else vals.mean(axis=1)
        meta["measurement_mode_used"] = used_mode
        meta["wide_cols_used"] = list(wide_cols)

    else:
        raise ValueError("Unknown measurement mode.")

    # Merge specs by part_key
    specs = specs_df.copy()
    if specs_part_col not in specs.columns:
        raise ValueError("Specs Part column not found.")
    if target_col not in specs.columns:
        raise ValueError("Specs target column not found.")

    specs["spec_part_key"] = safe_str_series(specs[specs_part_col]).str.strip()

    specs_out = pd.DataFrame({"spec_part_key": specs["spec_part_key"]})
    specs_out["target"] = coerce_numeric(specs[target_col])
    specs_out["lsl"] = coerce_numeric(specs[lsl_col]) if (lsl_col and lsl_col in specs.columns) else np.nan
    specs_out["usl"] = coerce_numeric(specs[usl_col]) if (usl_col and usl_col in specs.columns) else np.nan

    df = df.merge(specs_out, left_on="part_key", right_on="spec_part_key", how="left").drop(columns=["spec_part_key"])

    # Clean
    df = df.dropna(subset=["part_key", "machine_key", "y", "target"])
    meta["rows_after"] = len(df)

    return df, meta

# ------------------------------
# Parameter attach (robust for Single/Wide)
# ------------------------------
def attach_params_to_model_df(
    model_df: pd.DataFrame,
    data_df: pd.DataFrame,
    param_cols: list,
    part_col: str,
    machine_col: str,
) -> pd.DataFrame:
    temp_raw = data_df.copy()
    if "__rid__" not in temp_raw.columns:
        temp_raw["__rid__"] = np.arange(len(temp_raw), dtype=int)

    temp_raw["part_key"] = safe_str_series(temp_raw[part_col]).str.strip()
    temp_raw["machine_key"] = safe_str_series(temp_raw[machine_col]).str.strip()

    for p in param_cols:
        if p in temp_raw.columns:
            temp_raw[p] = coerce_numeric(temp_raw[p])

    # Primary: merge params by stable __rid__
    if "__rid__" in model_df.columns:
        temp = temp_raw[["__rid__"] + [p for p in param_cols if p in temp_raw.columns]].copy()
        model_df = model_df.merge(temp, on="__rid__", how="left")

    # Fallback: group mean by part+machine (ensures columns exist)
    missing_or_allnan = []
    for p in param_cols:
        if p not in model_df.columns:
            missing_or_allnan.append(p)
        else:
            if model_df[p].notna().sum() == 0:
                missing_or_allnan.append(p)

    if missing_or_allnan:
        gp = (
            temp_raw.groupby(["part_key", "machine_key"], dropna=False)[[p for p in param_cols if p in temp_raw.columns]]
            .mean()
            .reset_index()
        )
        model_df = model_df.merge(gp, on=["part_key", "machine_key"], how="left", suffixes=("", ""))

    for p in param_cols:
        if p not in model_df.columns:
            model_df[p] = np.nan

    return model_df

# ------------------------------
# Optimization core (dynamic params)
# ------------------------------
def optimize_parameters(model, X, y, target, bounds):
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

def build_model_dict(use_lr, use_rf, use_svr, rf_estimators, rf_random_state, svr_kernel, svr_c, svr_gamma, svr_epsilon):
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
        models["SVR"] = SVR(kernel=svr_kernel, C=float(svr_c), gamma=svr_gamma, epsilon=float(svr_epsilon))
    return models

def compute_results_dynamic(model_df, param_cols, bounds_by_param, bounds_overrides, soft_tol_percent, model_choices, min_rows_per_group):
    results = []
    failures = []

    missing_params = [p for p in param_cols if p not in model_df.columns]
    if missing_params:
        raise ValueError(f"Parameters missing from model-ready table: {missing_params}")
    # Default bounds (used unless a Part+Machine override exists)
    default_bounds_by_param = {p: (float(bounds_by_param[p][0]), float(bounds_by_param[p][1])) for p in param_cols}

    for (part, machine), group in model_df.groupby(["part_key", "machine_key"]):
        # Support both tuple keys and "part|||machine" string keys (for JSON round-trip safety)
        group_key = (str(part), str(machine))
        group_over = {}
        if isinstance(bounds_overrides, dict):
            group_over = (
                bounds_overrides.get(group_key)
                or bounds_overrides.get(f"{part}|||{machine}")
                or {}
            )

        # Build bounds list in the same order as param_cols
        bounds = []
        for pcol in param_cols:
            if pcol in group_over and isinstance(group_over[pcol], (list, tuple)) and len(group_over[pcol]) == 2:
                lo, hi = group_over[pcol]
            else:
                lo, hi = default_bounds_by_param[pcol]
            bounds.append((float(lo), float(hi)))

        gg = group.copy()
        for p in param_cols:
            gg[p] = coerce_numeric(gg[p])
        gg["y"] = coerce_numeric(gg["y"])

        gg = gg.dropna(subset=param_cols + ["y", "target"])
        if len(gg) < min_rows_per_group:
            failures.append((part, machine, f"Too few rows after cleaning (<{min_rows_per_group})"))
            continue

        X = gg[param_cols].values
        y = gg["y"].values

        target = float(gg["target"].iloc[0]) if pd.notna(gg["target"].iloc[0]) else np.nan
        lsl = float(gg["lsl"].iloc[0]) if ("lsl" in gg.columns and pd.notna(gg["lsl"].iloc[0])) else np.nan
        usl = float(gg["usl"].iloc[0]) if ("usl" in gg.columns and pd.notna(gg["usl"].iloc[0])) else np.nan

        if not np.isfinite(target):
            failures.append((part, machine, "Missing/invalid target after merge"))
            continue

        best = None
        best_err = float("inf")

        for model_name, model in model_choices.items():
            optimized, predicted = optimize_parameters(model, X, y, target, bounds)
            if optimized is None:
                continue

            err = abs(predicted - target)
            if err < best_err:
                best_err = err

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

                row = {
                    "part_number": part,
                    "machine_number": machine,
                    "model_type": model_name,
                    "predicted_measurement": float(predicted),
                    "target": float(target),
                    "lower_spec_limit": lsl if np.isfinite(lsl) else np.nan,
                    "upper_spec_limit": usl if np.isfinite(usl) else np.nan,
                    "abs_error": float(err),
                    "percent_error": float(100.0 * err / target) if target != 0 else np.nan,
                    "in_spec": in_spec,
                    "soft_pass": soft_pass,
                    "rows_used": int(len(gg)),
                }
                for i, p in enumerate(param_cols):
                    row[f"optimized_{p}"] = float(optimized[i])
                best = row

        if best is not None:
            results.append(best)
        else:
            failures.append((part, machine, "Optimization failed for all selected models"))

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["part_number", "machine_number"], kind="stable")

    fail_df = pd.DataFrame(failures, columns=["part_number", "machine_number", "reason"])
    return res_df, fail_df

# ------------------------------
# Session-state file persistence
# ------------------------------
def ensure_loaded_data():
    return st.session_state.get("data_df"), st.session_state.get("specs_df")

def load_files_into_session(mode: str, data_up, specs_up):
    if mode == "Upload CSVs":
        if data_up is None or specs_up is None:
            raise ValueError("Upload both DATA and SPECS CSVs, then click Load / Refresh files.")
        data_df = read_csv_bytes(data_up.getvalue())
        specs_df = read_csv_bytes(specs_up.getvalue())
    else:
        if (not REPO_DATA_PATH.exists()) or (not REPO_SPECS_PATH.exists()):
            raise FileNotFoundError("Repo demo files not found in /data.")
        data_df = pd.read_csv(REPO_DATA_PATH)
        specs_df = pd.read_csv(REPO_SPECS_PATH)

    data_df = data_df.copy()
    if "__rid__" not in data_df.columns:
        data_df["__rid__"] = np.arange(len(data_df), dtype=int)

    st.session_state["data_df"] = data_df
    st.session_state["specs_df"] = specs_df

    # If files changed, clear results
    st.session_state.pop("results_df", None)
    st.session_state.pop("failures_df", None)

# ------------------------------
# UI: Title + Help
# ------------------------------
st.title("Flexible Process Parameter Optimization")
st.caption(f"Supports Single + Wide subgroup (Format C). Questions/issues: {CONTACT_EMAIL}")

with st.expander("Help: How the app works + how to use it", expanded=False):
    st.markdown(
        f"""
## What this app does
Recommends **process parameter settings** (any number of parameters) to make a **measurement** hit a **target**,
**per Part + Machine** group.

## Supported data layouts
- **Single measurement column**: one measurement per row (y column)
- **Wide subgroup member columns (Format C)**: Data1..DataN / Sample1..SampleN etc. are subgroup measurements that get aggregated to y
- **Auto (Mixed)**: tries Wide first (Format C) then falls back to Single measurement

> Long format with subgroup has been intentionally removed.

## How to use it
1. **Load files** (Upload or Repo Demo) using **Load / Refresh files**
2. **Map columns**: Part, Machine, Target, and select your measurement layout
3. **Choose parameters** to optimize (the app suggests numeric columns; you confirm)
4. Review **auto-suggested bounds** (editable)
5. Toggle **Auto-run** or click **Run optimization**
6. Download **optimized_machine_settings.csv** and **optimization_failures.csv**

## Notes
- Bounds are suggested from historical data, but you should keep them within safe operating limits.
- If parameters don't align row-by-row, the app falls back to Part+Machine averages.

Questions/issues: **{CONTACT_EMAIL}**
        """
    )

# ------------------------------
# Sidebar: template + files + models + run settings
# ------------------------------
with st.sidebar:
    st.header("Mode")
    app_mode = st.radio(
        "Interface mode",
        ["Simple", "Developer"],
        index=0,
        horizontal=True,
        key="app_mode",
        help=(
            "Simple: streamlined interface with sensible defaults — just load data, map columns, pick parameters, and run.\n\n"
            "Developer: full control over models, bounds, quantile tuning, per-combo overrides, and diagnostic previews."
        )
    )
    is_dev = (app_mode == "Developer")

    st.divider()
    st.info(
        "Quick Help\n\n"
        "1) Load/Refresh files\n"
        "2) Map Part/Machine/Target\n"
        "3) Choose Single or Wide (Format C)\n"
        "4) Pick parameter columns + bounds\n"
        "5) Run optimization + download\n\n"
        f"Questions/issues: {CONTACT_EMAIL}"
    )

    st.header("Mapping template (JSON)")
    tmpl_file = st.file_uploader("Load mapping template (JSON)", type=["json"], key="tmpl_json")
    template = None
    if tmpl_file is not None:
        try:
            template = json.loads(tmpl_file.getvalue().decode("utf-8"))
            st.success("Template loaded.")
        except Exception as e:
            st.error(f"Template load failed: {e}")

    st.divider()
    st.header("Input files")
    mode = st.radio("Load from", ["Use repo data (default)", "Upload CSVs"], index=1, key="mode_choice")

    data_up = specs_up = None
    if mode == "Upload CSVs":
        data_up = st.file_uploader("Upload DATA CSV", type=["csv"], key="u_data")
        specs_up = st.file_uploader("Upload SPECS CSV", type=["csv"], key="u_specs")
    else:
        st.code("data/synthetic_process_data.csv")
        st.code("data/spec_limits.csv")

    load_btn = st.button("Load / Refresh files", type="primary", key="load_btn")

    st.divider()
    st.header("Models")
    if is_dev:
        use_lr = st.checkbox(
            "LinearRegression",
            value=True,
            help="Fast linear model. Good baseline — works well when parameter–measurement relationships are roughly linear."
        )
        use_rf = st.checkbox(
            "RandomForestRegressor",
            value=True,
            help="Ensemble of decision trees. Handles non-linear relationships and interactions well. Slower than Linear Regression but usually more accurate on real process data."
        )
        use_svr = st.checkbox(
            "SVR (Support Vector Regression)",
            value=False,
            help="Support Vector Regression. Can capture complex non-linear patterns via kernels. Requires more tuning and is the slowest of the three options."
        )

        if use_rf:
            rf_estimators = st.slider(
                "RF n_estimators",
                50, 600, 200, 50,
                help="Number of trees in the Random Forest. More trees = more stable predictions but slower training. 100–300 is usually a good range."
            )
            rf_random_state = st.number_input(
                "RF random_state",
                value=42, step=1,
                help="Seed for the random number generator. Set this to any fixed integer to get reproducible results across runs."
            )
        else:
            rf_estimators = 200
            rf_random_state = 42

        if use_svr:
            svr_kernel = st.selectbox(
                "SVR kernel",
                ["rbf", "linear", "poly", "sigmoid"],
                index=0,
                help="The kernel function used by SVR to map inputs into a higher-dimensional space. 'rbf' (Radial Basis Function) works well for most process data. 'linear' is faster but assumes a linear relationship."
            )
            svr_c = st.number_input(
                "SVR C",
                value=10.0, step=1.0,
                help="Regularization parameter. Higher C = model tries harder to fit training data (risk of overfitting). Lower C = smoother, more generalized fit. Try values between 1 and 100."
            )
            svr_gamma = st.selectbox(
                "SVR gamma",
                ["scale", "auto"],
                index=0,
                help="Kernel coefficient. 'scale' uses 1/(n_features × variance) — recommended default. 'auto' uses 1/n_features. Only applies to rbf, poly, and sigmoid kernels."
            )
            svr_epsilon = st.number_input(
                "SVR epsilon",
                value=0.1, step=0.05,
                help="Width of the insensitive tube around predictions — errors smaller than epsilon are ignored during training. Increase if your measurement has some expected noise or tolerance."
            )
        else:
            svr_kernel = "rbf"
            svr_c = 10.0
            svr_gamma = "scale"
            svr_epsilon = 0.1
    else:
        # Simple mode: Linear Regression only with sensible defaults
        use_lr = True
        use_rf = False
        use_svr = False
        rf_estimators = 200
        rf_random_state = 42
        svr_kernel = "rbf"
        svr_c = 10.0
        svr_gamma = "scale"
        svr_epsilon = 0.1
        st.caption("Using Linear Regression. Switch to Developer mode to change models.")

    st.divider()
    st.header("Run settings")
    if is_dev:
        auto_run = st.checkbox(
            "Auto-run optimization on changes",
            value=True,
            help="When enabled, optimization re-runs automatically whenever any setting changes (column mapping, parameters, bounds, models, etc.). Uncheck to only run when you click the 'Run optimization' button."
        )
        soft_tol_percent_ui = st.slider(
            "Soft tolerance (±%) around target",
            0.0, 50.0, 10.0, 0.5,
            help="A secondary pass/fail band around the target value. A result is marked 'soft pass' if the predicted measurement falls within ±X% of the target — even if it is technically outside the hard spec limits (LSL/USL). Useful for flagging near-misses."
        )
        min_rows_per_group = st.number_input(
            "Min rows per Part+Machine group",
            min_value=5, max_value=500, value=10, step=1,
            help="Minimum number of clean data rows required for a Part+Machine group to be included in optimization. Groups with fewer rows are skipped and listed in the Failures tab. Increase this if you want more statistically reliable fits."
        )
    else:
        auto_run = True
        soft_tol_percent_ui = 10.0
        min_rows_per_group = 10
        st.caption("Auto-run enabled. Switch to Developer mode to adjust run settings.")

model_choices = build_model_dict(use_lr, use_rf, use_svr, rf_estimators, rf_random_state, svr_kernel, svr_c, svr_gamma, svr_epsilon)
if not model_choices:
    st.warning("Select at least one model.")
    st.stop()

# ------------------------------
# Load/refresh files into session_state (only on button click or first run)
# ------------------------------
data_df_ss, specs_df_ss = ensure_loaded_data()

if load_btn or (data_df_ss is None) or (specs_df_ss is None):
    try:
        load_files_into_session(mode, data_up, specs_up)
        st.success("Files loaded into session. You can change widgets without re-uploading.")
    except Exception as e:
        st.error(f"File load failed: {e}")
        st.stop()

data_df, specs_df = ensure_loaded_data()
if data_df is None or specs_df is None:
    st.warning("Load files to continue.")
    st.stop()

data_cols = list(data_df.columns)
specs_cols = list(specs_df.columns)

# ------------------------------
# Step 1 — Column mapping
# ------------------------------
st.subheader("Step 1 — Map your columns")

guess_part = guess_col(data_df, "part")
guess_machine = guess_col(data_df, "machine")
guess_measure = guess_col(data_df, "measurement")

guess_specs_part = guess_col(specs_df, "part")
guess_target = guess_col(specs_df, "target")
guess_lsl = guess_col(specs_df, "lsl")
guess_usl = guess_col(specs_df, "usl")

tpl_dm = (template or {}).get("data_mapping", {})
tpl_sm = (template or {}).get("specs_mapping", {})
tpl_pm = (template or {}).get("parameters", {})

def pick_default(options, prefer):
    if prefer in options:
        return options.index(prefer)
    return 0

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Data mapping")
    part_col = st.selectbox(
        "Part column",
        options=data_cols,
        index=pick_default(data_cols, tpl_dm.get("part_col") or guess_part),
        help="The column in your data CSV that identifies the part number or part type. Optimization runs separately for each unique Part + Machine combination."
    )
    machine_col = st.selectbox(
        "Machine column",
        options=data_cols,
        index=pick_default(data_cols, tpl_dm.get("machine_col") or guess_machine),
        help="The column that identifies the machine, work center, or production line. Optimization runs separately for each unique Part + Machine combination."
    )

    measurement_mode = st.selectbox(
        "Measurement layout",
        options=MEAS_MODES,
        index=pick_default(MEAS_MODES, tpl_dm.get("measurement_mode") or "Auto (Mixed)"),
        help=(
            "How measurements are stored in your data:\n\n"
            "• Auto (Mixed): tries Wide subgroup columns first, then falls back to a single column.\n"
            "• Single measurement column: one measurement value per row (e.g. a 'Value' or 'Result' column).\n"
            "• Wide subgroup (Format C): each row has multiple measurement columns (e.g. Data1, Data2, Data3) that are averaged/medianed into one value."
        )
    )
    agg_func = st.selectbox(
        "Aggregation (for subgroup)",
        options=["mean", "median"],
        index=pick_default(["mean", "median"], tpl_dm.get("agg_func") or "mean"),
        help="How to combine multiple subgroup measurement columns (e.g. Data1..DataN) into a single value per row. Mean is the arithmetic average. Median is more robust to outliers."
    )

    measurement_col = None
    wide_cols = None

    meas_opts = ["(none)"] + data_cols
    if measurement_mode in ["Auto (Mixed)", "Single measurement column"]:
        meas_pick = st.selectbox(
            "Measurement column (optional for Auto)",
            options=meas_opts,
            index=pick_default(meas_opts, tpl_dm.get("measurement_col") or guess_measure or "(none)"),
            help="The column containing the measurement or quality characteristic (CTQ) you want to optimize toward the target. Required for Single mode. Optional for Auto — if left as (none), Auto will only look for Wide subgroup columns."
        )
        measurement_col = None if meas_pick == "(none)" else meas_pick

    if measurement_mode != "Single measurement column":
        st.markdown("#### Format C / Wide subgroup detection")
        st.caption("If subgroup members are columns like Data1..DataN / Sample1..SampleN, use this (or Auto).")
        patterns_text = st.text_area(
            "Wide column regex patterns (one per line)",
            value="\n".join(tpl_dm.get("wide_patterns") or WIDE_PATTERNS_DEFAULT),
            height=120,
            help="Regular expressions used to auto-detect which columns are subgroup members. Each pattern is matched against column names (case-insensitive). Add your own patterns here if your columns follow a different naming convention (e.g. ^obs\\d+$)."
        )
        wide_patterns = [p.strip() for p in patterns_text.splitlines() if p.strip()]

        auto_wide = detect_wide_measure_cols(data_df, wide_patterns)
        default_wide = tpl_dm.get("wide_cols") or auto_wide
        default_wide = [c for c in default_wide if c in data_cols]
        wide_cols = st.multiselect(
            "Wide subgroup member columns (Format C) (optional for Auto)",
            options=data_cols,
            default=default_wide[:30],
            help="Manually select which columns are subgroup measurement members (e.g. Data1, Data2, Data3). These are averaged or medianed into a single 'y' value per row. If using Auto mode, this is pre-filled by the regex patterns above but can be overridden here."
        )
    else:
        wide_patterns = WIDE_PATTERNS_DEFAULT
        wide_cols = []

with c2:
    st.markdown("### Specs mapping")
    specs_part_col = st.selectbox(
        "Specs Part column",
        options=specs_cols,
        index=pick_default(specs_cols, tpl_sm.get("specs_part_col") or guess_specs_part),
        help="The column in your specs CSV that contains part numbers or part identifiers. This is used to join spec limits (target, LSL, USL) onto your process data by matching part keys."
    )
    target_col = st.selectbox(
        "Target column",
        options=specs_cols,
        index=pick_default(specs_cols, tpl_sm.get("target_col") or guess_target),
        help="The column in your specs CSV containing the target (nominal) value for the measurement. The optimizer tries to find parameter settings that make the predicted measurement as close to this value as possible."
    )

    lsl_opts = ["(none)"] + specs_cols
    usl_opts = ["(none)"] + specs_cols

    lsl_pick = st.selectbox(
        "LSL column (optional)",
        lsl_opts,
        index=pick_default(lsl_opts, tpl_sm.get("lsl_col") or guess_lsl or "(none)"),
        help="Lower Spec Limit — the minimum acceptable measurement value. If provided, the results table will show whether each optimized prediction falls within spec. Leave as (none) if you only have a target or USL."
    )
    usl_pick = st.selectbox(
        "USL column (optional)",
        usl_opts,
        index=pick_default(usl_opts, tpl_sm.get("usl_col") or guess_usl or "(none)"),
        help="Upper Spec Limit — the maximum acceptable measurement value. If provided, the results table will show whether each optimized prediction falls within spec. Leave as (none) if you only have a target or LSL."
    )

    lsl_col = None if lsl_pick == "(none)" else lsl_pick
    usl_col = None if usl_pick == "(none)" else usl_pick

# Template download
payload = mapping_payload({
    "part_col": part_col,
    "machine_col": machine_col,
    "measurement_mode": measurement_mode,
    "agg_func": agg_func,
    "measurement_col": measurement_col,
    "wide_cols": wide_cols,
    "wide_patterns": wide_patterns,
    "specs_part_col": specs_part_col,
    "target_col": target_col,
    "lsl_col": lsl_col or "(none)",
    "usl_col": usl_col or "(none)",
    "param_cols": tpl_pm.get("param_cols", []),
})
st.download_button(
    "Download mapping template JSON",
    data=json.dumps(payload, indent=2).encode("utf-8"),
    file_name="process_optimization_mapping_template.json",
    mime="application/json"
)

# ------------------------------
# Build model-ready table (measurement + specs merge)
# ------------------------------
with st.spinner("Building model-ready table (measurement + specs merge)..."):
    try:
        model_df, meta = build_model_df(
            data_df=data_df,
            specs_df=specs_df,
            part_col=part_col,
            machine_col=machine_col,
            specs_part_col=specs_part_col,
            target_col=target_col,
            lsl_col=lsl_col,
            usl_col=usl_col,
            measurement_mode=measurement_mode,
            agg_func=agg_func,
            measurement_col=measurement_col,
            wide_cols=wide_cols,
            wide_patterns=wide_patterns
        )
    except Exception as e:
        st.error(f"Could not build model-ready data: {e}")
        st.stop()

st.success(f"Measurement mode used: **{meta['measurement_mode_used']}** | Rows: {meta['rows_before']} → {meta['rows_after']}")
if meta.get("wide_cols_used"):
    st.caption(f"Wide subgroup cols used (first 15): {meta['wide_cols_used'][:15]}")

# ------------------------------
# Step 2 — Choose parameters (dynamic)
# ------------------------------
st.subheader("Step 2 — Choose parameters to optimize (dynamic)")

exclude = {part_col, machine_col, "__rid__", "part_key", "machine_key", "y", "target", "lsl", "usl"}
if measurement_col:
    exclude.add(measurement_col)
if wide_cols:
    exclude |= set(wide_cols)

param_candidates = detect_parameter_candidates(data_df, exclude_cols=exclude)
if not param_candidates:
    st.error("No numeric parameter candidates detected. Check your data or mappings.")
    st.stop()

default_params = [p for p in (tpl_pm.get("param_cols") or param_candidates[:3]) if p in param_candidates]
param_cols = st.multiselect(
    "Select parameter columns to optimize",
    options=param_candidates,
    default=default_params,
    help="The process input parameters (X variables) you want the optimizer to adjust in order to hit the target measurement. Only numeric columns with sufficient data are shown. Select all parameters that are controllable on the machine."
)

if not param_cols:
    st.warning("Select at least one parameter column.")
    st.stop()

# Attach params robustly
model_df = attach_params_to_model_df(
    model_df=model_df,
    data_df=data_df,
    param_cols=param_cols,
    part_col=part_col,
    machine_col=machine_col,
)

missing_after = [p for p in param_cols if p not in model_df.columns]
if missing_after:
    st.error(f"Parameters missing from model-ready table after merge: {missing_after}")
    st.write("Model-ready columns:", list(model_df.columns))
    st.stop()

# ------------------------------
# Step 3 — Bounds

with st.sidebar:
    if is_dev:
        st.header("Bounds suggestion (defaults)")
        auto_bounds = st.checkbox(
            "Use custom quantiles for bound suggestion",
            value=True,
            help="When checked, the three sliders below control how default bounds are calculated from your historical data. When unchecked, fixed defaults are used (5th–95th percentile, 10% padding) regardless of slider positions."
        )
        if auto_bounds:
            q_low = st.slider(
                "Lower quantile",
                0.0, 0.20, 0.05, 0.01,
                help="The lower percentile of your historical data used as the starting point for the minimum bound. e.g. 0.05 = 5th percentile. A small padding is then subtracted (see Padding below)."
            )
            q_high = st.slider(
                "Upper quantile",
                0.80, 1.0, 0.95, 0.01,
                help="The upper percentile of your historical data used as the starting point for the maximum bound. e.g. 0.95 = 95th percentile. A small padding is then added (see Padding below)."
            )
            pad_frac = st.slider(
                "Padding (% of span)",
                0.0, 0.50, 0.10, 0.01,
                help="Extra buffer added beyond the quantile range on both sides. e.g. 0.10 = 10% of the quantile span is added above and below. Prevents the optimizer from being boxed in right at the data edge."
            )
        else:
            q_low = 0.05
            q_high = 0.95
            pad_frac = 0.10
            st.caption("Using fixed defaults: 5th–95th percentile + 10% padding.")

        per_group_suggest = st.checkbox(
            "Suggest defaults from one Part+Machine subset",
            value=False,
            help="When checked, the default bounds are calculated using only the rows matching the Part and Machine you select below — instead of all rows in the dataset. Useful when different parts or machines operate in very different parameter ranges."
        )
    else:
        # Simple mode: fixed defaults, no tuning
        q_low = 0.05
        q_high = 0.95
        pad_frac = 0.10
        per_group_suggest = False

raw_hist = data_df.copy()
if per_group_suggest:
    parts_raw = sorted(raw_hist[part_col].dropna().astype(str).unique().tolist())
    machines_raw = sorted(raw_hist[machine_col].dropna().astype(str).unique().tolist())
    if parts_raw and machines_raw:
        sel_p_raw = st.sidebar.selectbox("Subset Part (raw)", parts_raw, index=0, key="subset_part_raw")
        sel_m_raw = st.sidebar.selectbox("Subset Machine (raw)", machines_raw, index=0, key="subset_machine_raw")
        raw_hist = raw_hist[(raw_hist[part_col].astype(str) == str(sel_p_raw)) & (raw_hist[machine_col].astype(str) == str(sel_m_raw))]
        if raw_hist.empty:
            st.sidebar.warning("Subset empty; using all rows for default suggestion.")
            raw_hist = data_df

# ------------------------------
# Step 3 — Bounds (developer mode only)
# ------------------------------
if is_dev:
    st.subheader("Step 3 — Bounds (auto-suggested defaults, with per Part–Machine overrides)")
else:
    st.info("⚙️ **Step 3 — Bounds** is only available in Developer mode. Auto-suggested bounds are being used.")

# Session state for bounds always needed (logic runs in both modes)
if "bounds_default" not in st.session_state:
    st.session_state.bounds_default = {}
if "bounds_overrides" not in st.session_state:
    st.session_state.bounds_overrides = {}

reset_defaults = st.sidebar.button(
    "Reset ALL default bounds to suggested",
    use_container_width=True,
    help="Recalculate all default bounds from historical data using the current quantile settings."
) if is_dev else False

bounds_by_param: Dict[str, Tuple[float, float]] = {}
default_meta: Dict[str, Dict] = {}

for pcol in param_cols:
    if pcol not in data_df.columns:
        st.error(f"Parameter '{pcol}' not found in DATA CSV.")
        st.stop()

    lo_s, hi_s, meta_b = suggest_bounds(
        raw_hist[pcol],
        q_low=q_low,
        q_high=q_high,
        pad_frac=pad_frac,
    )

    default_meta[pcol] = meta_b

    if reset_defaults:
        kmin = f"def_min__{pcol}"
        kmax = f"def_max__{pcol}"
        if kmin in st.session_state:
            del st.session_state[kmin]
        if kmax in st.session_state:
            del st.session_state[kmax]
        st.session_state.bounds_default[pcol] = (float(lo_s), float(hi_s))
    elif pcol not in st.session_state.bounds_default:
        st.session_state.bounds_default[pcol] = (float(lo_s), float(hi_s))

    lo0, hi0 = st.session_state.bounds_default[pcol]
    bounds_by_param[pcol] = (float(lo0), float(hi0))

# ---- Default bounds editor (global) — developer only ----
if is_dev:
    with st.expander("Default bounds (apply to ALL Part–Machine combos unless overridden)", expanded=False):
        st.caption("These are the **global defaults**. To set different bounds for a specific Part–Machine, use the override editor below.")
        for pcol in param_cols:
            meta_b = default_meta.get(pcol, {"source": "unknown", "n": 0})
            lo0, hi0 = st.session_state.bounds_default[pcol]
            c1, c2 = st.columns(2)
            with c1:
                lo = st.number_input(
                    f"{pcol} default min",
                    value=float(lo0),
                    key=f"def_min__{pcol}",
                    help=f"Global minimum bound for '{pcol}'. Applied to every Part + Machine group unless overridden below. Suggested from historical data."
                )
            with c2:
                hi = st.number_input(
                    f"{pcol} default max",
                    value=float(hi0),
                    key=f"def_max__{pcol}",
                    help=f"Global maximum bound for '{pcol}'. Applied to every Part + Machine group unless overridden below. Suggested from historical data."
                )
            if lo >= hi:
                st.error(f"Invalid default bounds for {pcol}: min must be < max.")
                st.stop()
            st.session_state.bounds_default[pcol] = (float(lo), float(hi))
            bounds_by_param[pcol] = (float(lo), float(hi))
            st.caption(f"Suggested from: {meta_b.get('source')} (n={meta_b.get('n')})")

# ---- Per Part–Machine override editor — developer only ----
if is_dev:
    st.subheader("Per Part–Machine bounds overrides")
    st.caption("Edit bounds for **one** Part–Machine combo at a time. Changes here do **not** affect other combos.")

combo_df = model_df[["part_key", "machine_key"]].drop_duplicates().astype(str)
parts_k = sorted(combo_df["part_key"].unique().tolist())
machines_k = sorted(combo_df["machine_key"].unique().tolist())

# BUG FIX #5: Initialize machines_for_part to a safe default before any conditional block
machines_for_part = machines_k

if is_dev:
    if not parts_k or not machines_k:
        st.warning("No Part/Machine keys available to build bounds overrides.")
    else:
        colA, colB = st.columns(2)
        with colA:
            sel_part_k = st.selectbox(
                "Part (model key)",
                parts_k,
                index=0,
                key="bounds_sel_part_k",
                help="Select the part you want to edit bounds for. Only the selected Part + Machine combo is affected — other combos keep their own bounds."
            )
        with colB:
            machines_for_part = sorted(combo_df.loc[combo_df["part_key"] == str(sel_part_k), "machine_key"].unique().tolist())
            if not machines_for_part:
                machines_for_part = machines_k
            sel_machine_k = st.selectbox(
                "Machine (model key)",
                machines_for_part,
                index=0,
                key="bounds_sel_machine_k",
                help="Select the machine you want to edit bounds for. Only the selected Part + Machine combo is affected."
            )

        group_key = (str(sel_part_k), str(sel_machine_k))

        b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
        with b1:
            reset_this = st.button(
                "Reset this combo",
                help="Clear all overrides for this Part + Machine and revert its bounds to the global defaults.",
                use_container_width=True,
                key=f"btn_reset_this__{group_key[0]}__{group_key[1]}"
            )
        with b2:
            reset_all = st.button(
                "Reset ALL combos",
                help="Clear every per-combo override across all parts and machines. Everything reverts to the global default bounds.",
                use_container_width=True,
                key="btn_reset_all_combos"
            )
        with b3:
            copy_to_part = st.button(
                "Copy to all machines (this part)",
                help="Copy the overrides you've set for the currently selected combo to every other machine under the same part number.",
                use_container_width=True,
                key=f"btn_copy_to_part__{group_key[0]}"
            )
        with b4:
            show_all_summary = st.checkbox(
                "Show summary for all combos",
                value=False,
                key="chk_show_all_bounds_summary",
                help="When checked, the bounds summary table below shows every Part + Machine combo. When unchecked, only combos with active overrides are shown."
            )

        if reset_this:
            st.session_state.bounds_overrides.pop(group_key, None)
            for pcol in param_cols:
                def_lo, def_hi = st.session_state.bounds_default[pcol]
                kmin = f"ov_min__{group_key[0]}__{group_key[1]}__{pcol}"
                kmax = f"ov_max__{group_key[0]}__{group_key[1]}__{pcol}"
                st.session_state[kmin] = float(def_lo)
                st.session_state[kmax] = float(def_hi)
            st.rerun()

        if reset_all:
            st.session_state.bounds_overrides = {}
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and (k.startswith("ov_min__") or k.startswith("ov_max__")):
                    prefix = "ov_min__" if k.startswith("ov_min__") else "ov_max__"
                    remainder = k[len(prefix):]
                    for pcol in param_cols:
                        suffix = f"__{pcol}"
                        if remainder.endswith(suffix):
                            def_lo, def_hi = st.session_state.bounds_default[pcol]
                            st.session_state[k] = float(def_lo) if prefix == "ov_min__" else float(def_hi)
                            break
            st.rerun()

        current_over = st.session_state.bounds_overrides.get(group_key, {})
        st.markdown(f"**Editing override for:** `{group_key[0]}` / `{group_key[1]}`")

        for pcol in param_cols:
            def_lo, def_hi = st.session_state.bounds_default[pcol]
            ov = current_over.get(pcol, (def_lo, def_hi))
            ov_lo, ov_hi = float(ov[0]), float(ov[1])

            with st.expander(f"{pcol} (default: {def_lo:.6g} … {def_hi:.6g})", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    lo = st.number_input(
                        f"{pcol} min",
                        value=float(ov_lo),
                        key=f"ov_min__{group_key[0]}__{group_key[1]}__{pcol}",
                        help=f"Minimum allowed value for '{pcol}' for this Part + Machine. Default: {def_lo:.6g}"
                    )
                with c2:
                    hi = st.number_input(
                        f"{pcol} max",
                        value=float(ov_hi),
                        key=f"ov_max__{group_key[0]}__{group_key[1]}__{pcol}",
                        help=f"Maximum allowed value for '{pcol}' for this Part + Machine. Default: {def_hi:.6g}"
                    )
                if lo >= hi:
                    st.error(f"Invalid override bounds for {pcol}: min must be < max.")
                    st.stop()

                tol = 1e-12 * max(1.0, abs(def_lo), abs(def_hi))
                is_same = (abs(float(lo) - float(def_lo)) <= tol) and (abs(float(hi) - float(def_hi)) <= tol)

                if is_same:
                    if group_key in st.session_state.bounds_overrides and pcol in st.session_state.bounds_overrides[group_key]:
                        st.session_state.bounds_overrides[group_key].pop(pcol, None)
                        if len(st.session_state.bounds_overrides[group_key]) == 0:
                            st.session_state.bounds_overrides.pop(group_key, None)
                else:
                    if group_key not in st.session_state.bounds_overrides:
                        st.session_state.bounds_overrides[group_key] = {}
                    st.session_state.bounds_overrides[group_key][pcol] = (float(lo), float(hi))

        if copy_to_part:
            src_bounds = st.session_state.bounds_overrides.get(group_key, {})
            if not src_bounds:
                st.info("No overrides set for this combo yet (it is using defaults). Nothing to copy.")
            else:
                for mkey in machines_for_part:
                    gk = (str(sel_part_k), str(mkey))
                    st.session_state.bounds_overrides[gk] = dict(src_bounds)
                st.success(f"Copied bounds to all machines for part {sel_part_k}.")

        # ---- Bounds summary table ----
        st.subheader("Bounds summary")
        rows = []
        overrides = st.session_state.bounds_overrides

        if show_all_summary:
            combos_to_show = [(r["part_key"], r["machine_key"]) for _, r in combo_df.iterrows()]
        else:
            combos_to_show = [k for k, v in overrides.items() if isinstance(v, dict) and len(v) > 0]

        seen = set()
        combos_to_show = [c for c in combos_to_show if not (c in seen or seen.add(c))]

        if not combos_to_show:
            st.info("No per-combo overrides set yet. Everything is using default bounds.")
        else:
            for pk, mk in combos_to_show:
                gk = (str(pk), str(mk))
                ov = overrides.get(gk, {})
                row = {"part_key": gk[0], "machine_key": gk[1]}
                for pcol in param_cols:
                    def_lo, def_hi = st.session_state.bounds_default[pcol]
                    if pcol in ov:
                        lo, hi = ov[pcol]
                        src = "override"
                    else:
                        lo, hi = def_lo, def_hi
                        src = "default"
                    row[f"{pcol}__min"] = float(lo)
                    row[f"{pcol}__max"] = float(hi)
                    row[f"{pcol}__src"] = src
                rows.append(row)

            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True, height=420)
            st.download_button(
                "Download bounds_summary.csv",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="bounds_summary.csv",
                mime="text/csv"
            )


# ------------------------------
# Step 4 — Run optimization (auto-run optional)
# ------------------------------
st.subheader("Step 4 — Optimization")

soft_tol_percent = float(soft_tol_percent_ui) / 100.0

btn_row = st.columns([1, 1, 2])
with btn_row[0]:
    run_btn = st.button("Run optimization", type="primary", key="run_opt")
with btn_row[1]:
    clear_btn = st.button("Clear results", key="clear_results")

if clear_btn:
    st.session_state.pop("results_df", None)
    st.session_state.pop("failures_df", None)
    st.success("Cleared results.")

should_run = run_btn or (auto_run and st.session_state.get("results_df") is None)

# -----------------------------------------------------------------------
# BUG FIX #4: Include bounds_overrides in the signature hash so that
# auto-run triggers whenever a per-combo override is added or changed.
# -----------------------------------------------------------------------
sig = {
    "part_col": part_col,
    "machine_col": machine_col,
    "measurement_mode": meta["measurement_mode_used"],
    "measurement_col": measurement_col,
    "wide_cols": wide_cols,
    "param_cols": param_cols,
    "bounds": bounds_by_param,
    "bounds_overrides": json.dumps(
        {f"{k[0]}|||{k[1]}": v for k, v in st.session_state.get("bounds_overrides", {}).items()},
        sort_keys=True, default=str
    ),
    "soft_tol": soft_tol_percent,
    "min_rows": int(min_rows_per_group),
    "models": list(model_choices.keys()),
}
sig_str = json.dumps(sig, sort_keys=True, default=str)
sig_hash = hash(sig_str)

last_hash = st.session_state.get("last_run_hash")

if auto_run and (last_hash is None or last_hash != sig_hash):
    should_run = True

if should_run:
    with st.spinner("Optimizing per Part + Machine..."):
        try:
            results_df, failures_df = compute_results_dynamic(
                model_df=model_df,
                param_cols=param_cols,
                bounds_by_param=bounds_by_param,
                bounds_overrides=st.session_state.get('bounds_overrides', {}),
                soft_tol_percent=soft_tol_percent,
                model_choices=model_choices,
                min_rows_per_group=int(min_rows_per_group)
            )
            st.session_state["results_df"] = results_df
            st.session_state["failures_df"] = failures_df
            st.session_state["last_run_hash"] = sig_hash
            st.success("Optimization complete.")
        except Exception as e:
            st.error(f"Optimization failed: {e}")

results_df = st.session_state.get("results_df")
failures_df = st.session_state.get("failures_df")

if results_df is None or failures_df is None:
    st.info("Click **Run optimization** (or enable Auto-run) to compute results.")
    st.stop()

# ------------------------------
# Summary + Tabs
# ------------------------------
st.subheader("Summary")

total = len(results_df)
in_spec_count = int(results_df["in_spec"].sum()) if total and "in_spec" in results_df.columns else 0
soft_pass_count = int(results_df["soft_pass"].sum()) if total and "soft_pass" in results_df.columns else 0
avg_error = float(results_df["abs_error"].mean()) if total and "abs_error" in results_df.columns else float("nan")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Optimized groups", f"{total:,}")
m2.metric("In spec", f"{in_spec_count:,}")
m3.metric("Soft pass", f"{soft_pass_count:,}")
m4.metric("Avg abs error", f"{avg_error:.6f}" if np.isfinite(avg_error) else "NA")

if is_dev:
    tabs = st.tabs(["Results", "Failures", "Model-ready preview", "Raw preview"])
else:
    tabs = st.tabs(["Results", "Failures"])

with tabs[0]:
    st.subheader("Optimized Settings")
    if results_df.empty:
        st.warning("No results produced. Check targets/spec match, bounds, parameter columns, and row counts.")
    else:
        st.dataframe(results_df, use_container_width=True, height=520)
        st.download_button(
            "Download optimized_machine_settings.csv",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="optimized_machine_settings.csv",
            mime="text/csv",
        )

with tabs[1]:
    st.subheader("Groups That Did Not Optimize")
    if failures_df is None or failures_df.empty:
        st.success("No failures.")
    else:
        st.dataframe(failures_df, use_container_width=True, height=420)
        st.download_button(
            "Download optimization_failures.csv",
            data=failures_df.to_csv(index=False).encode("utf-8"),
            file_name="optimization_failures.csv",
            mime="text/csv",
        )

if is_dev:
    with tabs[2]:
        st.subheader("Model-ready data (what the optimizer uses)")
        st.dataframe(model_df.head(300), use_container_width=True, height=520)

    with tabs[3]:
        st.subheader("Raw upload preview")
        st.write("**DATA CSV (head)**")
        st.dataframe(data_df.head(200), use_container_width=True)
        st.write("**SPECS CSV (head)**")
        st.dataframe(specs_df.head(200), use_container_width=True)

st.caption(f"Questions/issues: {CONTACT_EMAIL}")
