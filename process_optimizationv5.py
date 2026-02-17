# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:16:45 2026

@author: chad
"""

# app_final_v4.py
# Process Optimization Dashboard (Single + Wide Format C only)
# - Per Part–Machine editable bounds with TRUE overrides (stored only when changed)
# - Bounds Summary table + download
# - Reset this combo, Copy to all machines for part
# - Reset ALL to suggested defaults (clears overrides + widget state)  ✅ fixed

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

CONTACT_EMAIL = "chad@hertzler.com"

# ------------------------------
# Session state init
# ------------------------------
if "bounds_overrides" not in st.session_state:
    # dict: (part, machine) -> {param: (min,max), ...}  (ONLY differs from defaults)
    st.session_state.bounds_overrides = {}

if "file_cache" not in st.session_state:
    st.session_state.file_cache = {"mode": None, "data_bytes": None, "specs_bytes": None, "data_df": None, "specs_df": None}

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def _finite_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s[np.isfinite(s)].dropna()

def suggest_bounds(s: pd.Series, q_low=0.05, q_high=0.95, pad_frac=0.10):
    s = _finite_series(s)
    if len(s) < 20:
        lo = float(s.min()) if len(s) else 0.0
        hi = float(s.max()) if len(s) else 1.0
        span = max(hi - lo, 1e-9)
        lo -= pad_frac * span
        hi += pad_frac * span
        meta = {"source": "fallback(min/max)+pad", "n": int(len(s))}
        return lo, hi, meta
    lo_q = float(s.quantile(q_low))
    hi_q = float(s.quantile(q_high))
    span = max(hi_q - lo_q, 1e-9)
    lo = lo_q - pad_frac * span
    hi = hi_q + pad_frac * span
    if lo >= hi:
        hi = lo + 1.0
    meta = {"source": f"quantiles({q_low:.0%}-{q_high:.0%})+pad({pad_frac:.0%})", "n": int(len(s))}
    return float(lo), float(hi), meta

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

def is_format_c_wide(df: pd.DataFrame) -> bool:
    # Wide Format C: multiple sample columns like Data1..DataN or Sample_1..Sample_N etc.
    cols = [str(c) for c in df.columns]
    # Heuristic: at least 2 columns that match pattern <letters><digits> or <letters>_<digits>
    import re
    matches = 0
    for c in cols:
        if re.match(r"^[A-Za-z]+_?\d+$", c.strip()):
            matches += 1
    return matches >= 2

def find_wide_measurement_cols(df: pd.DataFrame):
    import re
    cols = [str(c) for c in df.columns]
    # Prefer Data1..DataN
    data_cols = [c for c in cols if re.match(r"^Data\d+$", c, flags=re.IGNORECASE)]
    if len(data_cols) >= 2:
        # sort by numeric suffix
        data_cols = sorted(data_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))
        return data_cols
    # Else any prefix_1..N (Sample_1)
    pat_cols = [c for c in cols if re.match(r"^[A-Za-z]+_?\d+$", c.strip())]
    # group by prefix
    def prefix(c):
        m = re.match(r"^([A-Za-z]+)_?(\d+)$", c.strip())
        return m.group(1).lower() if m else ""
    from collections import defaultdict
    byp = defaultdict(list)
    for c in pat_cols:
        byp[prefix(c)].append(c)
    if not byp:
        return []
    # choose the largest group
    best_prefix = max(byp.keys(), key=lambda k: len(byp[k]))
    best = byp[best_prefix]
    best = sorted(best, key=lambda x: int(re.findall(r"\d+", x)[0]))
    return best

def auto_detect_columns(data_df: pd.DataFrame, specs_df: pd.DataFrame):
    """
    Return a dict describing:
      part_col, machine_col, measurement_mode ('single'/'wide'),
      measurement_col (if single) OR wide_cols (if wide),
      param_cols (candidate machine parameters),
      specs mapping (spec_part_col, target_col, lsl_col, usl_col)
    """
    # ---- detect part/machine in DATA
    cols = [c for c in data_df.columns]
    lower = {c: str(c).strip().lower() for c in cols}

    def pick_first(candidates):
        for cand in candidates:
            for c in cols:
                if lower[c] == cand:
                    return c
        # fuzzy contains
        for cand in candidates:
            for c in cols:
                if cand in lower[c]:
                    return c
        return None

    part_col = pick_first(["part_number", "part no", "part", "pn"])
    machine_col = pick_first(["machine_number", "machine", "workcenter", "work center", "line"])
    datetime_col = pick_first(["datetime", "timestamp", "date/time", "date time", "date"])

    # ---- detect measurement
    measurement_col = pick_first(["measurement", "result", "value", "y"])
    wide_cols = []
    measurement_mode = "single"

    if measurement_col is None:
        # Try Wide Format C
        wide_cols = find_wide_measurement_cols(data_df)
        if len(wide_cols) >= 2:
            measurement_mode = "wide"
        else:
            measurement_mode = "single"

    # ---- detect specs columns
    scols = [c for c in specs_df.columns]
    slower = {c: str(c).strip().lower() for c in scols}

    def pick_specs(candidates):
        for cand in candidates:
            for c in scols:
                if slower[c] == cand:
                    return c
        for cand in candidates:
            for c in scols:
                if cand in slower[c]:
                    return c
        return None

    spec_part_col = pick_specs(["part_number", "part no", "part", "pn"])
    target_col = pick_specs(["target", "nominal", "target x"])
    lsl_col = pick_specs(["lower_spec_limit", "lsl", "lo spec"])
    usl_col = pick_specs(["upper_spec_limit", "usl", "hi spec"])

    # ---- param candidates: numeric columns excluding ids + measurement cols
    excluded = set([c for c in [part_col, machine_col, datetime_col, measurement_col] if c])
    excluded |= set(wide_cols)
    numeric_cols = []
    for c in cols:
        if c in excluded:
            continue
        s = pd.to_numeric(data_df[c], errors="coerce")
        if np.isfinite(s).sum() >= max(10, int(0.05 * len(data_df))):
            numeric_cols.append(c)

    return {
        "part_col": part_col,
        "machine_col": machine_col,
        "datetime_col": datetime_col,
        "measurement_mode": measurement_mode,
        "measurement_col": measurement_col,
        "wide_cols": wide_cols,
        "param_cols": numeric_cols,
        "spec_part_col": spec_part_col,
        "target_col": target_col,
        "lsl_col": lsl_col,
        "usl_col": usl_col
    }

def make_model_ready(data_df: pd.DataFrame, cfg: dict):
    """
    Convert DATA into a model-ready table with:
      part_col, machine_col, measurement
      parameter columns (numeric)
    For Wide Format C: measurement = mean across wide cols per row.
    """
    df = data_df.copy()

    if cfg["part_col"] is None or cfg["machine_col"] is None:
        raise ValueError("Could not auto-detect Part and Machine columns. Please rename columns or add mapping UI.")

    if cfg["measurement_mode"] == "single":
        if cfg["measurement_col"] is None:
            raise ValueError("Could not detect measurement column for single format.")
        df["_measurement_"] = pd.to_numeric(df[cfg["measurement_col"]], errors="coerce")
    else:
        wide_cols = cfg["wide_cols"]
        if not wide_cols:
            raise ValueError("Could not detect wide measurement columns (Format C).")
        df["_measurement_"] = df[wide_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # keep only needed base columns + params
    base_cols = [cfg["part_col"], cfg["machine_col"], "_measurement_"]
    keep = base_cols + cfg["param_cols"]
    keep = [c for c in keep if c in df.columns]
    model_df = df[keep].copy()

    # coerce params numeric
    for p in cfg["param_cols"]:
        if p in model_df.columns:
            model_df[p] = pd.to_numeric(model_df[p], errors="coerce")

    # drop rows missing essentials
    model_df = model_df.dropna(subset=[cfg["part_col"], cfg["machine_col"], "_measurement_"])
    return model_df

def merge_specs(model_df: pd.DataFrame, specs_df: pd.DataFrame, cfg: dict):
    if cfg["spec_part_col"] is None:
        raise ValueError("Could not detect Part column in SPECS file.")
    if cfg["target_col"] is None:
        raise ValueError("Could not detect Target/Nominal column in SPECS file.")

    df = model_df.merge(
        specs_df,
        left_on=cfg["part_col"],
        right_on=cfg["spec_part_col"],
        how="left",
        suffixes=("", "_spec")
    )

    # map to standard internal names
    df["_target_"] = pd.to_numeric(df[cfg["target_col"]], errors="coerce")
    df["_lsl_"] = pd.to_numeric(df[cfg["lsl_col"]], errors="coerce") if cfg["lsl_col"] else np.nan
    df["_usl_"] = pd.to_numeric(df[cfg["usl_col"]], errors="coerce") if cfg["usl_col"] else np.nan
    return df

def compute_results(merged_df: pd.DataFrame, cfg: dict, selected_params: list, bounds_by_group: dict,
                    soft_tol_percent: float, model_choices: dict):
    part_col = cfg["part_col"]
    machine_col = cfg["machine_col"]

    results=[]
    failures=[]

    # Validate params exist
    missing_params = [p for p in selected_params if p not in merged_df.columns]
    if missing_params:
        raise ValueError(f"Parameters missing from model-ready table after merge: {missing_params}")

    for (part, machine), g in merged_df.groupby([part_col, machine_col]):
        g = g.dropna(subset=selected_params + ["_measurement_", "_target_"])
        if len(g) < 10:
            failures.append((part, machine, "Too few rows after cleaning (<10)"))
            continue

        X = g[selected_params].values
        y = g["_measurement_"].values
        target = float(g["_target_"].iloc[0]) if pd.notna(g["_target_"].iloc[0]) else np.nan
        lsl = g["_lsl_"].iloc[0]
        usl = g["_usl_"].iloc[0]

        if not np.isfinite(target):
            failures.append((part, machine, "Missing/invalid target"))
            continue

        # bounds in order of selected_params
        key = (str(part), str(machine))
        bdict = bounds_by_group.get(key, None)
        if bdict is None:
            failures.append((part, machine, "Missing bounds for group"))
            continue
        bounds = [tuple(map(float, bdict[p])) for p in selected_params]

        best=None
        best_err=float("inf")

        for model_name, model in model_choices.items():
            opt, pred = optimize_parameters(model, X, y, target, bounds)
            if opt is None:
                continue
            err = abs(pred - target)
            if err < best_err:
                best_err = err
                soft_lo = target*(1-soft_tol_percent)
                soft_hi = target*(1+soft_tol_percent)
                soft_pass = bool(soft_lo <= pred <= soft_hi)

                in_spec = True
                lsl_f = float(lsl) if pd.notna(lsl) else np.nan
                usl_f = float(usl) if pd.notna(usl) else np.nan
                if np.isfinite(lsl_f) and np.isfinite(usl_f):
                    in_spec = bool(lsl_f <= pred <= usl_f)
                elif np.isfinite(lsl_f):
                    in_spec = bool(pred >= lsl_f)
                elif np.isfinite(usl_f):
                    in_spec = bool(pred <= usl_f)

                row = {
                    "part_number": part,
                    "machine_number": machine,
                    "model_type": model_name,
                    "predicted_measurement": float(pred),
                    "target": float(target),
                    "lower_spec_limit": lsl_f,
                    "upper_spec_limit": usl_f,
                    "abs_error": float(err),
                    "percent_error": float(100.0*err/target) if target != 0 else np.nan,
                    "in_spec": in_spec,
                    "soft_pass": soft_pass,
                    "rows_used": int(len(g)),
                }
                for i,p in enumerate(selected_params):
                    row[f"optimized_{p}"] = float(opt[i])
                best = row

        if best:
            results.append(best)
        else:
            failures.append((part, machine, "Optimization failed for all selected models"))

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["part_number","machine_number"], kind="stable").reset_index(drop=True)
    fail_df = pd.DataFrame(failures, columns=["part_number","machine_number","reason"])
    return res_df, fail_df

# ------------------------------
# Title
# ------------------------------
st.title("Process Parameter Optimization")
st.caption("Optimize machine parameters to hit a target measurement (per part + machine).")

# ------------------------------
# Sidebar: Files + Models
# ------------------------------
with st.sidebar:
    st.header("Input files")

    # Default to repo mode on first load (Option B)
    if "mode_radio" not in st.session_state:
        st.session_state["mode_radio"] = "Use repo data (default)"

    mode = st.radio("Load from", ["Use repo data (default)", "Upload CSVs"], index=0, key="mode_radio")

    data_up = specs_up = None
    if mode == "Upload CSVs":
        data_up = st.file_uploader("Upload DATA CSV", type=["csv"], key="uploader_data")
        specs_up = st.file_uploader("Upload SPECS CSV", type=["csv"], key="uploader_specs")
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
        rf_estimators, rf_random_state = 200, 42

    if use_svr:
        svr_kernel = st.selectbox("SVR kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        svr_c = st.number_input("SVR C", value=10.0, step=1.0)
        svr_gamma = st.selectbox("SVR gamma", ["scale", "auto"], index=0)
        svr_epsilon = st.number_input("SVR epsilon", value=0.1, step=0.05)
    else:
        svr_kernel, svr_c, svr_gamma, svr_epsilon = "rbf", 10.0, "scale", 0.1

    st.divider()
    st.header("Targets")
    soft_tol_percent_ui = st.slider("Soft tolerance (±%) around target", 0.0, 50.0, 10.0, 0.5)

model_choices = build_model_dict(use_lr, use_rf, use_svr, rf_estimators, rf_random_state, svr_kernel, svr_c, svr_gamma, svr_epsilon)
if not model_choices:
    st.warning("Select at least one model.")
    st.stop()

# ------------------------------
# Load data with caching so UI changes don't force re-upload
# ------------------------------
def load_data(mode, data_up, specs_up):
    cache = st.session_state.file_cache

    if mode == "Upload CSVs":
        if data_up is None or specs_up is None:
            return None, None
        data_bytes = data_up.getvalue()
        specs_bytes = specs_up.getvalue()

        # if bytes differ, reload
        if cache["mode"] != mode or cache["data_bytes"] != data_bytes or cache["specs_bytes"] != specs_bytes:
            cache["mode"] = mode
            cache["data_bytes"] = data_bytes
            cache["specs_bytes"] = specs_bytes
            cache["data_df"] = read_csv_bytes(data_bytes)
            cache["specs_df"] = read_csv_bytes(specs_bytes)
        return cache["data_df"], cache["specs_df"]

    # repo mode
    if cache["mode"] != mode:
        cache["mode"] = mode
        cache["data_bytes"] = None
        cache["specs_bytes"] = None
        cache["data_df"] = None
        cache["specs_df"] = None

    if cache["data_df"] is None or cache["specs_df"] is None:
        if (not REPO_DATA_PATH.exists()) or (not REPO_SPECS_PATH.exists()):
            st.error("Repo files not found in the deployed filesystem.")
            st.write("Expected:", str(REPO_DATA_PATH))
            st.write("Expected:", str(REPO_SPECS_PATH))
            st.write("Does /data exist?", DATA_DIR.exists(), "->", str(DATA_DIR))
            if DATA_DIR.exists():
                st.write("Files in /data:", [p.name for p in DATA_DIR.iterdir()])
            return None, None
        cache["data_df"] = pd.read_csv(REPO_DATA_PATH)
        cache["specs_df"] = pd.read_csv(REPO_SPECS_PATH)
    return cache["data_df"], cache["specs_df"]

data_df, specs_df = load_data(mode, data_up, specs_up)

if data_df is None or specs_df is None:
    st.info("Provide both DATA and SPECS files to proceed.")
    st.stop()

# ------------------------------
# Auto-detect columns + build model-ready table
# ------------------------------
cfg = auto_detect_columns(data_df, specs_df)

model_df = make_model_ready(data_df, cfg)
merged_df = merge_specs(model_df, specs_df, cfg)

# ------------------------------
# Parameter selection UI (customer columns can vary)
# ------------------------------
st.subheader("Parameter & Measurement Detection")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("**Detected Part column (DATA):**", cfg["part_col"])
    st.write("**Detected Machine column (DATA):**", cfg["machine_col"])
with c2:
    st.write("**Detected Measurement mode:**", "Wide Format C" if cfg["measurement_mode"] == "wide" else "Single")
    if cfg["measurement_mode"] == "wide":
        st.write("**Wide measurement columns:**", ", ".join(cfg["wide_cols"][:8]) + (" ..." if len(cfg["wide_cols"]) > 8 else ""))
    else:
        st.write("**Measurement column:**", cfg["measurement_col"])
with c3:
    st.write("**Detected Part column (SPECS):**", cfg["spec_part_col"])
    st.write("**Detected Target column:**", cfg["target_col"])
    st.write("**Detected LSL/USL:**", cfg["lsl_col"], "/", cfg["usl_col"])

st.caption("Select which numeric columns are *machine parameters* to optimize. (Exclude measurement/sample columns.)")

default_params = cfg["param_cols"][:3] if cfg["param_cols"] else []
selected_params = st.multiselect("Machine parameter columns to optimize", options=cfg["param_cols"], default=default_params)

if not selected_params:
    st.warning("Select at least one machine parameter column to optimize.")
    st.stop()

# ------------------------------
# Bounds: defaults + per-group overrides
# ------------------------------
with st.sidebar:
    st.header("Optimization bounds")

    st.caption("Defaults are suggested from historical values. You can override per Part–Machine.")

    q_low = st.slider("Lower quantile", 0.0, 0.20, 0.05, 0.01)
    q_high = st.slider("Upper quantile", 0.80, 1.0, 0.95, 0.01)
    pad_frac = st.slider("Padding (% of span)", 0.0, 0.50, 0.10, 0.01)

# available groups
parts = sorted(model_df[cfg["part_col"]].astype(str).unique().tolist())
machines = sorted(model_df[cfg["machine_col"]].astype(str).unique().tolist())
if not parts or not machines:
    st.error("No parts/machines found after parsing data.")
    st.stop()

# choose a group for editing bounds
with st.sidebar:
    st.subheader("Edit bounds for a Part–Machine")
    edit_part = st.selectbox("Part", parts, index=0, key="bounds_edit_part")
    edit_machine = st.selectbox("Machine", machines, index=0, key="bounds_edit_machine")
    group_key = (str(edit_part), str(edit_machine))

# build default bounds for this group from history (group-specific suggestion)
hist = model_df[(model_df[cfg["part_col"]].astype(str) == str(edit_part)) & (model_df[cfg["machine_col"]].astype(str) == str(edit_machine))]
if hist.empty:
    hist = model_df

default_bounds = {}
default_meta = {}
for p in selected_params:
    lo, hi, meta = suggest_bounds(hist[p] if p in hist.columns else model_df[p], q_low=q_low, q_high=q_high, pad_frac=pad_frac)
    default_bounds[p] = (float(lo), float(hi))
    default_meta[p] = meta

# fetch current override dict for group
group_over = st.session_state.bounds_overrides.get(group_key, {})

def get_effective_bounds_for_group(gkey, defaults):
    over = st.session_state.bounds_overrides.get(gkey, {})
    eff = {}
    for p,(lo,hi) in defaults.items():
        if p in over:
            eff[p] = tuple(map(float, over[p]))
        else:
            eff[p] = (float(lo), float(hi))
    return eff

effective = get_effective_bounds_for_group(group_key, default_bounds)

# ---- RESET / COPY controls (fixed keys clearing)
def _clear_override_widgets_for_group(gkey):
    """
    Clear ALL widget keys that belong to a specific Part–Machine combo.
    This must be aggressive because Streamlit will otherwise keep the prior number_input values
    even after we remove overrides.
    """
    part, machine = str(gkey[0]), str(gkey[1])
    # Keys are generated like: ov_min__{part}__{machine}__{param} and ov_max__...
    # But param names can contain spaces/symbols, so match by the part+machine segment only.
    seg = f"__{part}__{machine}__"
    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (k.startswith("ov_min__") or k.startswith("ov_max__")) and (seg in k):
            del st.session_state[k]

def _clear_all_override_widgets():
    """Clear ALL override widget keys for ALL combos."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (k.startswith("ov_min__") or k.startswith("ov_max__")):
            del st.session_state[k]

with st.sidebar:

    btn_row1, btn_row2 = st.columns(2)
    with btn_row1:
        if st.button("Reset this combo", key="btn_reset_this"):
            st.session_state.bounds_overrides.pop(group_key, None)
            _clear_override_widgets_for_group(group_key)
            st.rerun()
    with btn_row2:
        if st.button("Copy to all machines for this part", key="btn_copy_part"):
            # copy overrides from current combo to all machines for same part
            src_over = st.session_state.bounds_overrides.get(group_key, {})
            if not src_over:
                st.info("No overrides on this combo to copy.")
            else:
                for m in machines:
                    dest = (str(edit_part), str(m))
                    if dest == group_key:
                        continue
                    # set overrides exactly as source
                    st.session_state.bounds_overrides[dest] = dict(src_over)
                # also clear widgets so they re-render correctly when you switch machines
                _clear_all_override_widgets()
                st.rerun()

    # ✅ Fixed: Reset ALL to suggested defaults
    if st.button("Reset ALL combos to suggested defaults", key="btn_reset_all_defaults"):
        st.session_state.bounds_overrides = {}
        _clear_all_override_widgets()
        st.rerun()

# ---- editable bounds inputs (unique keys per group+param)
with st.sidebar:
    st.markdown("**Editable bounds (store override only if different than suggested)**")
    for p in selected_params:
        d_lo, d_hi = default_bounds[p]
        eff_lo, eff_hi = effective[p]

        min_key = f"ov_min__{group_key[0]}__{group_key[1]}__{p}"
        max_key = f"ov_max__{group_key[0]}__{group_key[1]}__{p}"

        c1, c2 = st.columns(2)
        with c1:
            vmin = st.number_input(f"{p} min", value=float(eff_lo), key=min_key)
        with c2:
            vmax = st.number_input(f"{p} max", value=float(eff_hi), key=max_key)

        # validate per param
        if float(vmin) >= float(vmax):
            st.error(f"Invalid bounds for {p}: min must be < max.")
            st.stop()

        # store override only if different from default
        changed = (abs(float(vmin) - float(d_lo)) > 1e-12) or (abs(float(vmax) - float(d_hi)) > 1e-12)
        if changed:
            group_over[p] = (float(vmin), float(vmax))
        else:
            group_over.pop(p, None)

    # clean group entry if empty
    if group_over:
        st.session_state.bounds_overrides[group_key] = group_over
    else:
        st.session_state.bounds_overrides.pop(group_key, None)

# ------------------------------
# Build bounds_by_group for all groups (effective bounds)
# ------------------------------
bounds_by_group = {}
for p in parts:
    for m in machines:
        gkey = (str(p), str(m))
        # suggest defaults per-group for current selected_params
        ghist = model_df[(model_df[cfg["part_col"]].astype(str) == str(p)) & (model_df[cfg["machine_col"]].astype(str) == str(m))]
        if ghist.empty:
            ghist = model_df
        defaults = {}
        for param in selected_params:
            lo, hi, _ = suggest_bounds(ghist[param] if param in ghist.columns else model_df[param], q_low=q_low, q_high=q_high, pad_frac=pad_frac)
            defaults[param] = (float(lo), float(hi))
        eff = get_effective_bounds_for_group(gkey, defaults)
        bounds_by_group[gkey] = eff

# ------------------------------
# Run optimization (auto-run is fine)
# ------------------------------
soft_tol_percent = float(soft_tol_percent_ui) / 100.0

with st.spinner("Optimizing per part + machine..."):
    results_df, failures_df = compute_results(
        merged_df=merged_df,
        cfg=cfg,
        selected_params=selected_params,
        bounds_by_group=bounds_by_group,
        soft_tol_percent=soft_tol_percent,
        model_choices=model_choices
    )

# ------------------------------
# Summary
# ------------------------------
st.subheader("Summary")
total = len(results_df)
in_spec = int(results_df["in_spec"].sum()) if total else 0
soft_pass = int(results_df["soft_pass"].sum()) if total else 0
avg_err = float(results_df["abs_error"].mean()) if total else np.nan

a,b,c,d = st.columns(4)
a.metric("Optimized groups", f"{total:,}")
b.metric("In spec", f"{in_spec:,}")
c.metric("Soft pass (±%)", f"{soft_pass:,}")
d.metric("Avg abs error", f"{avg_err:.6f}" if np.isfinite(avg_err) else "NA")

tabs = st.tabs(["Results", "Failures", "Bounds Summary", "Input Preview", "Help"])

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
    st.subheader("Bounds Summary (Overrides)")
    over = st.session_state.bounds_overrides
    if not over:
        st.info("No overrides set. (All groups using suggested defaults.)")
    else:
        rows=[]
        for (p,m), dct in over.items():
            for param,(lo,hi) in dct.items():
                rows.append({"part_number": p, "machine_number": m, "parameter": param, "min": lo, "max": hi})
        bdf = pd.DataFrame(rows).sort_values(["part_number","machine_number","parameter"], kind="stable")
        st.dataframe(bdf, use_container_width=True, height=420)
        st.download_button("Download bounds_overrides.csv", data=bdf.to_csv(index=False).encode("utf-8"),
                           file_name="bounds_overrides.csv", mime="text/csv")

with tabs[3]:
    st.subheader("Data Preview")
    st.write("**DATA (head)**")
    st.dataframe(data_df.head(200), use_container_width=True)
    st.write("**SPECS (head)**")
    st.dataframe(specs_df.head(200), use_container_width=True)

with tabs[4]:
    st.subheader("Help / How to use")
    st.markdown(
        f"""
### What this app does
This dashboard trains a regression model (per Part + Machine group) to learn how **machine parameters** influence a **measurement**.
It then uses numerical optimization to find parameter values (within your bounds) that make the predicted measurement as close as possible to the **Target** from the SPECS file.

### Supported input formats
- **Single measurement column** (e.g., `measurement`, `Result`, `Value`)
- **Wide Format C subgroups** where measurements are stored in multiple columns like `Data1..DataN` or `Sample_1..Sample_N`.
  - In Wide Format C, the app uses the **row mean of the Data/Sample columns** as the measurement for modeling/optimization.

### Typical workflow
1. Load DATA + SPECS (repo mode or upload).
2. Pick which numeric columns are **machine parameters** to optimize.
3. Review / edit bounds (suggested defaults appear automatically).
4. Results update automatically (no need to press a Run button).

### Bounds behavior (important)
- Suggested bounds are computed from historical values (quantiles + padding).
- Overrides are stored **per Part–Machine only when you change a bound away from its suggested default**.
- **Reset this combo** clears overrides for the selected Part–Machine (and resets the widgets).
- **Copy to all machines for this part** copies the selected combo’s overrides to all machines for that part.
- **Reset ALL combos to suggested defaults** clears *all* overrides and wipes the bounds summary.

### Questions / Issues
Contact: **{CONTACT_EMAIL}**
"""
    )

st.caption("Tip: Suggested bounds are defaults; always verify they are within safe operating limits.")
