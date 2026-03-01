"""
Specification search for paper 112474-V1
Dinkelman (2011) - "The Effects of Rural Electrification on Employment: New Evidence from South Africa"

Design: Instrumental Variables (2SLS)
  - Treatment: T (community received Eskom electrification project)
  - Instrument: mean_grad_new (community land gradient)
  - Outcomes: d_prop_emp_f (female employment change), d_prop_emp_m (male employment change)
"""

import json
import hashlib
import traceback
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pyfixest as pf
from linearmodels.iv import IV2SLS, IVLIML
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
PAPER_ID = "112474-V1"
BASE_DIR = Path(__file__).resolve().parents[2]
PKG_DIR = BASE_DIR / "data" / "downloads" / "extracted" / PAPER_ID
DATA_DIR = PKG_DIR / "20080791_dataset" / "data"
SURFACE_PATH = PKG_DIR / "SPECIFICATION_SURFACE.json"
OUTPUT_DIR = PKG_DIR  # outputs go to top-level of package dir

# ============================================================
# Load surface + compute hash
# ============================================================
with open(SURFACE_PATH) as f:
    surface_text = f.read()
    surface = json.loads(surface_text)

# Compute canonical hash matching the validator (sorted keys, compact separators)
_canon = json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
surface_hash = "sha256:" + hashlib.sha256(_canon.encode("utf-8")).hexdigest()

# ============================================================
# Software block
# ============================================================
software_block = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pandas": pd.__version__,
        "pyfixest": pf.__version__,
        "numpy": np.__version__,
        "linearmodels": "6.1",
        "scipy": stats.scipy_version.full_version if hasattr(stats, "scipy_version") else "unknown",
    },
}
try:
    import scipy
    software_block["packages"]["scipy"] = scipy.__version__
except:
    pass

# ============================================================
# Load and prepare data
# ============================================================
df_raw = pd.read_stata(str(DATA_DIR / "matched_censusdata.dta"))
df_full = df_raw.copy()

# Main analysis sample: large areas
df = df_raw[df_raw["largeareas"] == 1].copy()
df["sexratio0"] = df["sexratio0_a"]

# Full sample (for rc/sample/restriction/full_sample)
df_full["sexratio0"] = df_full["sexratio0_a"]

# ============================================================
# Variable definitions
# ============================================================
BASELINE_CONTROLS = [
    "kms_to_subs0", "baseline_hhdens0", "base_hhpovrate0",
    "prop_head_f_a0", "sexratio0", "prop_indianwhite0",
    "kms_to_road0", "kms_to_town0", "prop_matric_m0", "prop_matric_f0",
]
SERVICE_CONTROLS = ["d_prop_waterclose", "d_prop_flush"]
ALL_CONTROLS = BASELINE_CONTROLS + SERVICE_CONTROLS

TREATMENT = "T"
INSTRUMENT = "mean_grad_new"
FE_VAR = "dccode0"
CLUSTER_VAR = "placecode0"

# Design audit block (from surface)
DESIGN_AUDIT = {
    "estimator": "2sls",
    "endog_vars": ["T"],
    "instrument_vars": ["mean_grad_new"],
    "n_instruments": 1,
    "overid_df": 0,
    "fe_structure": ["dccode0"],
    "cluster_vars": ["placecode0"],
    "sample_restriction": "largeareas==1",
    "bundle": {
        "bundle_type": "iv",
        "linked_adjustment": True,
        "notes": "Controls and district FE are shared across first and second stages per paper code.",
    },
}

INFERENCE_CANONICAL = {
    "spec_id": "infer/se/cluster/placecode0",
    "params": {"cluster_var": "placecode0"},
}

# ============================================================
# Helper functions
# ============================================================
def run_iv_pyfixest(outcome, controls, fe, data, cluster_var=CLUSTER_VAR):
    """Run 2SLS via pyfixest. Returns model or raises."""
    ctrl_str = " + ".join(controls) if controls else "1"
    if fe:
        formula = f"{outcome} ~ {ctrl_str} | {fe} | {TREATMENT} ~ {INSTRUMENT}"
    else:
        formula = f"{outcome} ~ {ctrl_str} | {TREATMENT} ~ {INSTRUMENT}"
    m = pf.feols(formula, data=data, vcov={"CRV1": cluster_var})
    return m


def run_iv_linearmodels(outcome, controls, fe_var, data, cluster_var=CLUSTER_VAR, method="2sls"):
    """Run IV via linearmodels (supports LIML). Returns model."""
    work = data.dropna(subset=[outcome, TREATMENT, INSTRUMENT] + controls).copy()
    if fe_var:
        fe_dummies = pd.get_dummies(work[fe_var], prefix="fe", drop_first=True).astype(float)
        X_exog = work[controls].join(fe_dummies)
    else:
        X_exog = work[controls].copy()
    X_exog = X_exog.assign(const=1.0)

    if method == "liml":
        m = IVLIML(work[outcome], X_exog, work[[TREATMENT]], work[[INSTRUMENT]])
    else:
        m = IV2SLS(work[outcome], X_exog, work[[TREATMENT]], work[[INSTRUMENT]])

    return m.fit(cov_type="clustered", clusters=work[cluster_var])


def run_ols_pyfixest(outcome, regressors, fe, data, cluster_var=CLUSTER_VAR):
    """Run OLS via pyfixest."""
    reg_str = " + ".join(regressors) if regressors else "1"
    if fe:
        formula = f"{outcome} ~ {reg_str} | {fe}"
    else:
        formula = f"{outcome} ~ {reg_str}"
    m = pf.feols(formula, data=data, vcov={"CRV1": cluster_var})
    return m


def extract_pyfixest(m, focal_var=TREATMENT):
    """Extract results from pyfixest model."""
    coef = float(m.coef()[focal_var])
    se = float(m.se()[focal_var])
    pval = float(m.pvalue()[focal_var])
    ci = m.confint()
    ci_lower = float(ci.loc[focal_var].iloc[0])
    ci_upper = float(ci.loc[focal_var].iloc[1])
    n_obs = int(m._N)
    r2 = float(m._r2) if hasattr(m, "_r2") and m._r2 is not None else np.nan
    all_coefs = {k: float(v) for k, v in m.coef().items()}
    return coef, se, pval, ci_lower, ci_upper, n_obs, r2, all_coefs


def extract_linearmodels(m, focal_var=TREATMENT):
    """Extract results from linearmodels model."""
    coef = float(m.params[focal_var])
    se = float(m.std_errors[focal_var])
    pval = float(m.pvalues[focal_var])
    ci = m.conf_int()
    ci_lower = float(ci.loc[focal_var, "lower"])
    ci_upper = float(ci.loc[focal_var, "upper"])
    n_obs = int(m.nobs)
    r2 = float(m.r2) if hasattr(m, "r2") else np.nan
    all_coefs = {k: float(v) for k, v in m.params.items()}
    return coef, se, pval, ci_lower, ci_upper, n_obs, r2, all_coefs


def make_success_payload(all_coefs, design_audit, extra_blocks=None):
    """Build coefficient_vector_json for successful run."""
    payload = {
        "coefficients": all_coefs,
        "inference": INFERENCE_CANONICAL,
        "software": software_block,
        "surface_hash": surface_hash,
        "design": {"instrumental_variables": design_audit},
    }
    if extra_blocks:
        payload.update(extra_blocks)
    return payload


def make_failure_payload(error_msg, stage="estimation", exc=None):
    """Build coefficient_vector_json for failed run."""
    details = {
        "stage": stage,
        "exception_type": type(exc).__name__ if exc else "Unknown",
        "exception_message": str(exc) if exc else error_msg,
    }
    return {"error": error_msg, "error_details": details}


def first_stage_f(data, controls, fe_var):
    """Compute first-stage F statistic."""
    try:
        m_fs = run_ols_pyfixest(TREATMENT, [INSTRUMENT] + controls, fe_var, data)
        t_stat = float(m_fs.tstat()[INSTRUMENT])
        return t_stat ** 2
    except:
        return np.nan


# ============================================================
# Specification runner
# ============================================================
spec_results = []
inference_results = []
run_counter = 0


def add_result(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
               coef, se, pval, ci_lower, ci_upper, n_obs, r2,
               coef_vec_json, sample_desc, fixed_effects, controls_desc, cluster_var,
               run_success, run_error=""):
    global run_counter
    run_counter += 1
    spec_results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": f"{PAPER_ID}_run{run_counter:03d}",
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(coef_vec_json) if isinstance(coef_vec_json, dict) else coef_vec_json,
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": run_success,
        "run_error": run_error,
    })
    return f"{PAPER_ID}_run{run_counter:03d}"


def add_inference_result(spec_run_id, spec_id, spec_tree_path, baseline_group_id,
                         coef, se, pval, ci_lower, ci_upper, n_obs, r2,
                         coef_vec_json, run_success, run_error="",
                         outcome_var="", treatment_var="", cluster_var=""):
    global run_counter
    run_counter += 1
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": f"{PAPER_ID}_inf{run_counter:03d}",
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(coef_vec_json) if isinstance(coef_vec_json, dict) else coef_vec_json,
        "cluster_var": cluster_var,
        "run_success": run_success,
        "run_error": run_error,
    })


def run_spec_for_group(group_id, outcome_var, spec_id, spec_tree_path,
                       controls, fe_var, data, sample_desc,
                       design_audit_override=None, extra_blocks=None,
                       method="pyfixest", focal_var=TREATMENT):
    """Run a single specification and record result."""
    da = design_audit_override or DESIGN_AUDIT
    fe_desc = fe_var if fe_var else "none"
    ctrl_desc = ", ".join(controls) if controls else "none"

    try:
        if method == "pyfixest":
            m = run_iv_pyfixest(outcome_var, controls, fe_var, data)
            coef, se, pval, ci_lo, ci_hi, n_obs, r2, all_coefs = extract_pyfixest(m, focal_var)
        elif method == "liml":
            m = run_iv_linearmodels(outcome_var, controls, fe_var, data, method="liml")
            coef, se, pval, ci_lo, ci_hi, n_obs, r2, all_coefs = extract_linearmodels(m, focal_var)
        elif method == "ols":
            m = run_ols_pyfixest(outcome_var, [focal_var] + controls, fe_var, data)
            coef, se, pval, ci_lo, ci_hi, n_obs, r2, all_coefs = extract_pyfixest(m, focal_var)
        elif method == "ols_rf":
            # Reduced form: regress outcome on instrument
            m = run_ols_pyfixest(outcome_var, [INSTRUMENT] + controls, fe_var, data)
            coef, se, pval, ci_lo, ci_hi, n_obs, r2, all_coefs = extract_pyfixest(m, INSTRUMENT)
        else:
            raise ValueError(f"Unknown method: {method}")

        payload = make_success_payload(all_coefs, da, extra_blocks)

        # Add bundle info
        payload["bundle"] = {
            "bundle_type": "iv",
            "linked_adjustment": True,
            "components": {
                "second_stage": {"controls": controls, "fixed_effects": [fe_var] if fe_var else []},
                "first_stage": {"controls": controls, "fixed_effects": [fe_var] if fe_var else []},
            },
        }

        spec_run_id = add_result(
            spec_id, spec_tree_path, group_id, outcome_var, TREATMENT,
            coef, se, pval, ci_lo, ci_hi, n_obs, r2,
            payload, sample_desc, fe_desc, ctrl_desc, CLUSTER_VAR,
            1, "",
        )
        return spec_run_id, coef, se, pval, n_obs

    except Exception as e:
        error_msg = str(e)[:200]
        payload = make_failure_payload(error_msg, "estimation", e)
        spec_run_id = add_result(
            spec_id, spec_tree_path, group_id, outcome_var, TREATMENT,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            payload, sample_desc, fe_desc, ctrl_desc, CLUSTER_VAR,
            0, error_msg,
        )
        return spec_run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Run all specs for each baseline group
# ============================================================
GROUPS = [
    ("G1", "d_prop_emp_f"),
    ("G2", "d_prop_emp_m"),
]

baseline_run_ids = {}

for group_id, outcome_var in GROUPS:
    print(f"\n{'='*60}")
    print(f"Running specifications for {group_id}: {outcome_var}")
    print(f"{'='*60}")

    # --------------------------------------------------------
    # BASELINE: Table 4 Col 9 (IV with full controls + FE + services)
    # --------------------------------------------------------
    print("  [1] Baseline (Table 4 Col 9)...")
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "baseline",
        "designs/instrumental_variables.md#baseline",
        ALL_CONTROLS, FE_VAR, df,
        "largeareas==1, N=1816",
    )
    baseline_run_ids[group_id] = rid
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # ADDITIONAL BASELINES
    # --------------------------------------------------------
    # Table 4 Col 8: IV with baseline controls + FE, no services
    print("  [2] Baseline (Table 4 Col 8, no services)...")
    if group_id == "G1":
        sid = "baseline__table4_col8_female"
    else:
        sid = "baseline__table4_col8_male"
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var, sid,
        "designs/instrumental_variables.md#baseline",
        BASELINE_CONTROLS, FE_VAR, df,
        "largeareas==1, N=1816",
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # Table 4 Col 7: IV with baseline controls, no FE
    print("  [3] Baseline (Table 4 Col 7, controls only, no FE)...")
    if group_id == "G1":
        sid = "baseline__table4_col7_female"
    else:
        sid = "baseline__table4_col7_male"
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var, sid,
        "designs/instrumental_variables.md#baseline",
        BASELINE_CONTROLS, None, df,
        "largeareas==1, N=1816",
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # DESIGN VARIANT: LIML
    # --------------------------------------------------------
    print("  [4] Design: LIML...")
    da_liml = DESIGN_AUDIT.copy()
    da_liml["estimator"] = "liml"
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "design/instrumental_variables/estimator/liml",
        "designs/instrumental_variables.md#a-second-stage-estimators-holding-the-instrument-set-fixed",
        ALL_CONTROLS, FE_VAR, df,
        "largeareas==1, N=1816",
        design_audit_override=da_liml,
        method="liml",
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # RC: CONTROLS LOO
    # --------------------------------------------------------
    print("  [5-16] RC: Controls LOO...")
    for ctrl in ALL_CONTROLS:
        loo_controls = [c for c in ALL_CONTROLS if c != ctrl]
        rid, c, s, p, n = run_spec_for_group(
            group_id, outcome_var,
            f"rc/controls/loo/drop_{ctrl}",
            "modules/robustness/controls.md#leave-one-out-controls-loo",
            loo_controls, FE_VAR, df,
            "largeareas==1, N=1816",
            extra_blocks={
                "controls": {
                    "spec_id": f"rc/controls/loo/drop_{ctrl}",
                    "family": "loo",
                    "dropped": [ctrl],
                    "n_controls": len(loo_controls),
                }
            },
        )
        print(f"      LOO drop {ctrl}: coef={c:.4f}, se={s:.4f}, p={p:.4f}")

    # --------------------------------------------------------
    # RC: CONTROLS PROGRESSION
    # --------------------------------------------------------
    print("  [17] RC: Bivariate (no controls, no FE)...")
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/controls/progression/bivariate",
        "modules/robustness/controls.md#progressive-control-sets",
        [], None, df,
        "largeareas==1, N=1816",
        extra_blocks={
            "controls": {
                "spec_id": "rc/controls/progression/bivariate",
                "family": "progression",
                "included": [],
                "n_controls": 0,
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}")

    print("  [18] RC: Geographic controls only...")
    geo_controls = ["kms_to_subs0", "kms_to_road0", "kms_to_town0"]
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/controls/progression/geographic",
        "modules/robustness/controls.md#progressive-control-sets",
        geo_controls, FE_VAR, df,
        "largeareas==1, N=1816",
        extra_blocks={
            "controls": {
                "spec_id": "rc/controls/progression/geographic",
                "family": "progression",
                "included": geo_controls,
                "n_controls": len(geo_controls),
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}")

    print("  [19] RC: Demographic controls only...")
    demo_controls = [
        "baseline_hhdens0", "base_hhpovrate0", "prop_head_f_a0",
        "sexratio0", "prop_indianwhite0", "prop_matric_m0", "prop_matric_f0",
    ]
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/controls/progression/demographic",
        "modules/robustness/controls.md#progressive-control-sets",
        demo_controls, FE_VAR, df,
        "largeareas==1, N=1816",
        extra_blocks={
            "controls": {
                "spec_id": "rc/controls/progression/demographic",
                "family": "progression",
                "included": demo_controls,
                "n_controls": len(demo_controls),
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}")

    print("  [20] RC: Full without services...")
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/controls/progression/full_no_services",
        "modules/robustness/controls.md#progressive-control-sets",
        BASELINE_CONTROLS, FE_VAR, df,
        "largeareas==1, N=1816",
        extra_blocks={
            "controls": {
                "spec_id": "rc/controls/progression/full_no_services",
                "family": "progression",
                "included": BASELINE_CONTROLS,
                "n_controls": len(BASELINE_CONTROLS),
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}")

    # --------------------------------------------------------
    # RC: CONTROLS RANDOM SUBSETS
    # --------------------------------------------------------
    print("  [21-30] RC: Random control subsets...")
    rng = np.random.RandomState(112474)
    for i in range(1, 11):
        # Stratified by size: pick size first, then draw
        size = rng.randint(2, len(BASELINE_CONTROLS) + 1)  # 2 to 10
        indices = sorted(rng.choice(len(BASELINE_CONTROLS), size=size, replace=False))
        subset = [BASELINE_CONTROLS[j] for j in indices]
        # Always include district FE for these
        rid, c, s, p, n = run_spec_for_group(
            group_id, outcome_var,
            f"rc/controls/subset/random_{i:03d}",
            "modules/robustness/controls.md#random-control-subsets",
            subset, FE_VAR, df,
            "largeareas==1, N=1816",
            extra_blocks={
                "controls": {
                    "spec_id": f"rc/controls/subset/random_{i:03d}",
                    "family": "subset",
                    "draw_index": i,
                    "seed": 112474,
                    "included": subset,
                    "n_controls": len(subset),
                }
            },
        )
        print(f"      subset {i}: {len(subset)} controls, coef={c:.4f}, se={s:.4f}, p={p:.4f}")

    # --------------------------------------------------------
    # RC: SAMPLE RESTRICTIONS
    # --------------------------------------------------------
    print("  [31] RC: Full sample (no largeareas restriction)...")
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/sample/restriction/full_sample",
        "modules/robustness/sample.md#sample-restrictions",
        ALL_CONTROLS, FE_VAR, df_full,
        "full sample, N=" + str(len(df_full)),
        extra_blocks={
            "sample": {
                "spec_id": "rc/sample/restriction/full_sample",
                "family": "restriction",
                "description": "Full sample without largeareas>=100 restriction",
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    print("  [32] RC: No roads sample...")
    df_noroads = df[df["count_roads"] != 1].copy()
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/sample/restriction/no_roads",
        "modules/robustness/sample.md#sample-restrictions",
        ALL_CONTROLS, FE_VAR, df_noroads,
        "largeareas==1 & count_roads!=1",
        extra_blocks={
            "sample": {
                "spec_id": "rc/sample/restriction/no_roads",
                "family": "restriction",
                "description": "Exclude communities with major roads (Appendix 3 Table 2)",
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    print("  [33] RC: No nearby treated 1km...")
    df_no1km = df[
        (df["nearby_T_before_1km"] == 0) & (df["nearby_T_during_1km"] == 0) | (df["T"] == 1)
    ].copy()
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/sample/restriction/no_nearby_treated_1km",
        "modules/robustness/sample.md#sample-restrictions",
        ALL_CONTROLS, FE_VAR, df_no1km,
        "no untreated near treated (1km buffer)",
        extra_blocks={
            "sample": {
                "spec_id": "rc/sample/restriction/no_nearby_treated_1km",
                "family": "restriction",
                "description": "Exclude untreated areas within 1km of treated (Table 8 spillover test)",
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    print("  [34] RC: No nearby treated 5km...")
    df_no5km = df[
        (df["nearby_T_before_5km"] == 0) & (df["nearby_T_during_5km"] == 0) | (df["T"] == 1)
    ].copy()
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/sample/restriction/no_nearby_treated_5km",
        "modules/robustness/sample.md#sample-restrictions",
        ALL_CONTROLS, FE_VAR, df_no5km,
        "no untreated near treated (5km buffer)",
        extra_blocks={
            "sample": {
                "spec_id": "rc/sample/restriction/no_nearby_treated_5km",
                "family": "restriction",
                "description": "Exclude untreated areas within 5km of treated (Table 8 spillover test)",
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # RC: TRIMMING
    # --------------------------------------------------------
    print("  [35] RC: Trim outcome 1-99...")
    p1 = df[outcome_var].quantile(0.01)
    p99 = df[outcome_var].quantile(0.99)
    df_trim1 = df[(df[outcome_var] >= p1) & (df[outcome_var] <= p99)].copy()
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/sample/outliers/trim_y_1_99",
        "modules/robustness/sample.md#outlier-trimming",
        ALL_CONTROLS, FE_VAR, df_trim1,
        f"trim {outcome_var} at 1st/99th pctile",
        extra_blocks={
            "sample": {
                "spec_id": "rc/sample/outliers/trim_y_1_99",
                "family": "outliers",
                "trim_lower": 0.01,
                "trim_upper": 0.99,
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    print("  [36] RC: Trim outcome 5-95...")
    p5 = df[outcome_var].quantile(0.05)
    p95 = df[outcome_var].quantile(0.95)
    df_trim5 = df[(df[outcome_var] >= p5) & (df[outcome_var] <= p95)].copy()
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/sample/outliers/trim_y_5_95",
        "modules/robustness/sample.md#outlier-trimming",
        ALL_CONTROLS, FE_VAR, df_trim5,
        f"trim {outcome_var} at 5th/95th pctile",
        extra_blocks={
            "sample": {
                "spec_id": "rc/sample/outliers/trim_y_5_95",
                "family": "outliers",
                "trim_lower": 0.05,
                "trim_upper": 0.95,
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # RC: FIXED EFFECTS
    # --------------------------------------------------------
    print("  [37] RC: Drop district FE...")
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/fe/drop/dccode0",
        "modules/robustness/fixed_effects.md#drop-fixed-effects",
        ALL_CONTROLS, None, df,
        "largeareas==1, no district FE",
        extra_blocks={
            "fixed_effects": {
                "spec_id": "rc/fe/drop/dccode0",
                "family": "drop",
                "dropped": ["dccode0"],
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    print("  [38] RC: Add hetindex control (political)...")
    df_het = df.dropna(subset=["hetindex"]).copy()
    controls_het = ALL_CONTROLS + ["hetindex"]
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/fe/add/hetindex_controls",
        "modules/robustness/fixed_effects.md#add-controls",
        controls_het, FE_VAR, df_het,
        "largeareas==1, add hetindex (Appendix 3 Table 2)",
        extra_blocks={
            "fixed_effects": {
                "spec_id": "rc/fe/add/hetindex_controls",
                "family": "add",
                "added": ["hetindex"],
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # RC: FUNCTIONAL FORM
    # --------------------------------------------------------
    print("  [39] RC: asinh(outcome)...")
    df_asinh = df.copy()
    df_asinh[f"asinh_{outcome_var}"] = np.arcsinh(df_asinh[outcome_var])
    rid, c, s, p, n = run_spec_for_group(
        group_id, f"asinh_{outcome_var}",
        "rc/form/outcome/asinh",
        "modules/robustness/functional_form.md#outcome-transformations",
        ALL_CONTROLS, FE_VAR, df_asinh,
        "largeareas==1, asinh transform of outcome",
        extra_blocks={
            "functional_form": {
                "spec_id": "rc/form/outcome/asinh",
                "family": "outcome_transform",
                "transform": "asinh",
                "interpretation": "Coefficient is approximate percentage effect of T on employment change (asinh approximation for values near zero)",
            }
        },
    )
    print(f"      coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # RC: OLS REDUCED FORM
    # --------------------------------------------------------
    print("  [40] RC: OLS Reduced Form (outcome ~ instrument)...")
    rid, c, s, p, n = run_spec_for_group(
        group_id, outcome_var,
        "rc/estimation/ols_reduced_form",
        "modules/robustness/controls.md#reduced-form",
        ALL_CONTROLS, FE_VAR, df,
        "largeareas==1, OLS reduced form",
        method="ols_rf",
        focal_var=INSTRUMENT,
        extra_blocks={
            "estimation": {
                "spec_id": "rc/estimation/ols_reduced_form",
                "family": "reduced_form",
                "description": "OLS regression of outcome on instrument with controls and FE",
            }
        },
    )
    print(f"      RF coef on {INSTRUMENT}={c:.6f}, se={s:.6f}, p={p:.4f}, N={n}")

    # --------------------------------------------------------
    # INFERENCE VARIANTS (separate table)
    # --------------------------------------------------------
    print("  Running inference variants on baseline...")
    base_rid = baseline_run_ids[group_id]

    # HC1 (no clustering)
    try:
        ctrl_str = " + ".join(ALL_CONTROLS)
        m_hc1 = pf.feols(
            f"{outcome_var} ~ {ctrl_str} | {FE_VAR} | {TREATMENT} ~ {INSTRUMENT}",
            data=df, vcov="hetero",
        )
        coef_hc1 = float(m_hc1.coef()[TREATMENT])
        se_hc1 = float(m_hc1.se()[TREATMENT])
        pval_hc1 = float(m_hc1.pvalue()[TREATMENT])
        ci_hc1 = m_hc1.confint()
        ci_lo_hc1 = float(ci_hc1.loc[TREATMENT].iloc[0])
        ci_hi_hc1 = float(ci_hc1.loc[TREATMENT].iloc[1])
        all_coefs_hc1 = {k: float(v) for k, v in m_hc1.coef().items()}

        inf_payload = {
            "coefficients": all_coefs_hc1,
            "inference": {"spec_id": "infer/se/hc/hc1", "params": {}},
            "software": software_block,
            "surface_hash": surface_hash,
            "design": {"instrumental_variables": DESIGN_AUDIT},
        }
        add_inference_result(
            base_rid, "infer/se/hc/hc1",
            "modules/inference/standard_errors.md#heteroskedasticity-robust",
            group_id, coef_hc1, se_hc1, pval_hc1, ci_lo_hc1, ci_hi_hc1,
            int(m_hc1._N), float(m_hc1._r2) if m_hc1._r2 else np.nan,
            inf_payload, 1,
            outcome_var=outcome_var, treatment_var=TREATMENT, cluster_var="",
        )
        print(f"      HC1: se={se_hc1:.4f}, p={pval_hc1:.4f}")
    except Exception as e:
        add_inference_result(
            base_rid, "infer/se/hc/hc1",
            "modules/inference/standard_errors.md#heteroskedasticity-robust",
            group_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            make_failure_payload(str(e), "inference", e), 0, str(e)[:200],
            outcome_var=outcome_var, treatment_var=TREATMENT, cluster_var="",
        )

    # District-level clustering
    try:
        m_dc = pf.feols(
            f"{outcome_var} ~ {ctrl_str} | {FE_VAR} | {TREATMENT} ~ {INSTRUMENT}",
            data=df, vcov={"CRV1": "dccode0"},
        )
        coef_dc = float(m_dc.coef()[TREATMENT])
        se_dc = float(m_dc.se()[TREATMENT])
        pval_dc = float(m_dc.pvalue()[TREATMENT])
        ci_dc = m_dc.confint()
        ci_lo_dc = float(ci_dc.loc[TREATMENT].iloc[0])
        ci_hi_dc = float(ci_dc.loc[TREATMENT].iloc[1])
        all_coefs_dc = {k: float(v) for k, v in m_dc.coef().items()}

        inf_payload = {
            "coefficients": all_coefs_dc,
            "inference": {"spec_id": "infer/se/cluster/dccode0", "params": {"cluster_var": "dccode0"}},
            "software": software_block,
            "surface_hash": surface_hash,
            "design": {"instrumental_variables": DESIGN_AUDIT},
        }
        add_inference_result(
            base_rid, "infer/se/cluster/dccode0",
            "modules/inference/standard_errors.md#cluster-robust",
            group_id, coef_dc, se_dc, pval_dc, ci_lo_dc, ci_hi_dc,
            int(m_dc._N), float(m_dc._r2) if m_dc._r2 else np.nan,
            inf_payload, 1,
            outcome_var=outcome_var, treatment_var=TREATMENT, cluster_var="dccode0",
        )
        print(f"      CRV1(dccode0): se={se_dc:.4f}, p={pval_dc:.4f}")
    except Exception as e:
        add_inference_result(
            base_rid, "infer/se/cluster/dccode0",
            "modules/inference/standard_errors.md#cluster-robust",
            group_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            make_failure_payload(str(e), "inference", e), 0, str(e)[:200],
            outcome_var=outcome_var, treatment_var=TREATMENT, cluster_var="dccode0",
        )

print(f"\n{'='*60}")
print(f"DONE: {len(spec_results)} estimate rows, {len(inference_results)} inference rows")
print(f"{'='*60}")

# ============================================================
# Write outputs
# ============================================================

# specification_results.csv
df_spec = pd.DataFrame(spec_results)
df_spec.to_csv(OUTPUT_DIR / "specification_results.csv", index=False)
print(f"Wrote specification_results.csv ({len(df_spec)} rows)")

# inference_results.csv
df_inf = pd.DataFrame(inference_results)
df_inf.to_csv(OUTPUT_DIR / "inference_results.csv", index=False)
print(f"Wrote inference_results.csv ({len(df_inf)} rows)")

# Summary stats
n_success = df_spec["run_success"].sum()
n_fail = len(df_spec) - n_success
print(f"\nResults: {n_success} successful, {n_fail} failed out of {len(df_spec)} total")

# Quick summary
for gid, ovar in GROUPS:
    gdf = df_spec[df_spec["baseline_group_id"] == gid]
    gsuc = gdf[gdf["run_success"] == 1]
    if len(gsuc) > 0:
        print(f"\n{gid} ({ovar}):")
        print(f"  Specs: {len(gsuc)} successful / {len(gdf)} total")
        print(f"  Coefficient range: [{gsuc['coefficient'].min():.4f}, {gsuc['coefficient'].max():.4f}]")
        print(f"  Median coefficient: {gsuc['coefficient'].median():.4f}")
        bl = gsuc[gsuc["spec_id"] == "baseline"]
        if len(bl) > 0:
            print(f"  Baseline: coef={bl['coefficient'].values[0]:.4f}, p={bl['p_value'].values[0]:.4f}")
