#!/usr/bin/env python3
"""
Specification search runner for 131981-V1
"Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey"
Altindag, Erten, and Keskin (AEJ: Applied)

Design: Sharp regression discontinuity at age-65 COVID curfew cutoff in Turkey.
Running variable: dif (birth month distance from December 1955 cutoff)
Treatment: before1955 (1 = born before Dec 1955 = subject to curfew)
"""

import json
import sys
import os
import hashlib
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from agent_output_utils import (
    surface_hash,
    software_block,
    make_success_payload,
    make_failure_payload,
    error_details_from_exception,
    safe_single_line,
)

# ── Paths ──────────────────────────────────────────────────────────────
PAPER_ID = "131981-V1"
PKG_DIR = PROJECT_ROOT / "data" / "downloads" / "extracted" / PAPER_ID
DATA_PATH = PKG_DIR / "konda_data_for_analysis.dta"
SURFACE_PATH = PKG_DIR / "SPECIFICATION_SURFACE.json"
OUT_DIR = PKG_DIR  # outputs go to top level of extracted package

# ── Load surface ───────────────────────────────────────────────────────
with open(SURFACE_PATH) as f:
    SURFACE = json.load(f)
SURF_HASH = surface_hash(SURFACE)
SW_BLOCK = software_block()
BG = SURFACE["baseline_groups"][0]
DESIGN_AUDIT = BG["design_audit"]
CANONICAL_INFERENCE = BG["inference_plan"]["canonical"]

# ── Load and prepare data ──────────────────────────────────────────────
print(f"Loading data from {DATA_PATH}")
df_raw = pd.read_stata(str(DATA_PATH))
df = df_raw[df_raw["non_response"] == 0].copy()
print(f"Observations after non_response filter: {len(df)}")

# Construct index outcomes (replicate Stata's weightave2 + normby)
# Anderson (2008) inverse-covariance-weighted index
list1 = ["head_ache", "mal_appetite", "sleeplessness", "scared", "shaking", "nervous",
         "indigestion", "unfocused", "unhappy", "weepy", "unwillingness", "undecisiveness",
         "disrupted", "useless", "uninterest", "worthless", "suicidal", "usually_tired",
         "stomach_discomfort", "quickly_tired"]
list2 = ["head_ache", "shaking", "indigestion", "stomach_discomfort"]
list3 = ["mal_appetite", "sleeplessness", "scared", "nervous", "unfocused",
         "unhappy", "weepy", "unwillingness", "undecisiveness", "disrupted", "useless",
         "uninterest", "worthless", "suicidal", "usually_tired", "quickly_tired"]


def make_anderson_index(data, items, control_mask):
    """
    Construct Anderson (2008) inverse-covariance-weighted index.
    1. Standardize each item to mean=0, sd=1 using full sample
    2. Compute inverse covariance matrix from control group
    3. Weight = row sums of inverse covariance matrix
    4. Index = weighted average of standardized items
    5. Normalize so control group has mean=0, sd=1
    """
    sub = data[items].copy()
    valid = sub.notna().all(axis=1)

    # Standardize each item (full sample)
    means = sub.mean()
    sds = sub.std()
    sds[sds == 0] = 1
    z = (sub - means) / sds

    # Covariance from control group
    ctrl_z = z.loc[control_mask & valid]
    if len(ctrl_z) < 5:
        return pd.Series(np.nan, index=data.index)

    cov_mat = ctrl_z.cov().values
    try:
        inv_cov = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_mat)

    # Weights = column sums of inverse covariance matrix
    weights = inv_cov.sum(axis=0)

    # Weighted index
    idx = z.values @ weights
    idx_series = pd.Series(idx, index=data.index)

    # Set missing if any item is missing
    idx_series[~valid] = np.nan

    # Normalize so control group mean=0, sd=1
    ctrl_vals = idx_series[control_mask & valid]
    ctrl_mean = ctrl_vals.mean()
    ctrl_sd = ctrl_vals.std()
    if ctrl_sd > 0:
        idx_series = (idx_series - ctrl_mean) / ctrl_sd
    else:
        idx_series = idx_series - ctrl_mean

    return idx_series


# Construct outcome indices
control_mask = df["before1955"] == 0
df["z_depression"] = make_anderson_index(df, list1, control_mask)
df["z_somatic"] = make_anderson_index(df, list2, control_mask)
df["z_nonsomatic"] = make_anderson_index(df, list3, control_mask)

# sum_srq: simple row total
df["sum_srq"] = df[list1].sum(axis=1)
for item in list1:
    df.loc[df[item].isna(), "sum_srq"] = np.nan

# Construct RD polynomial interaction terms
df["dif_x_treat"] = df["dif"] * df["before1955"]
df["dif2"] = df["dif"] ** 2
df["dif2_x_treat"] = df["dif2"] * df["before1955"]

# Construct FE dummies for controls
# Sanitize column names to avoid spaces/dots that break pyfixest formulas
import re
def sanitize_colname(name):
    """Replace spaces, dots, and special chars with underscores for formula compatibility."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

edu_dummies = pd.get_dummies(df["education"], prefix="edu", drop_first=False, dtype=float)
edu_dummies.columns = [sanitize_colname(c) for c in edu_dummies.columns]
edu_dummies = edu_dummies.iloc[:, 1:]  # drop first category as baseline
df = pd.concat([df, edu_dummies], axis=1)

eth_dummies = pd.get_dummies(df["ethnicity"], prefix="eth", drop_first=False, dtype=float)
eth_dummies.columns = [sanitize_colname(c) for c in eth_dummies.columns]
eth_dummies = eth_dummies.iloc[:, 1:]
df = pd.concat([df, eth_dummies], axis=1)

st_dummies = pd.get_dummies(df["survey_taker_id"], prefix="st", drop_first=False, dtype=float)
st_dummies.columns = [sanitize_colname(c) for c in st_dummies.columns]
st_dummies = st_dummies.iloc[:, 1:]
df = pd.concat([df, st_dummies], axis=1)

# Collect column name groups
month_cols = [c for c in df.columns if c.startswith("month_")]
province_cols = [c for c in df.columns if c.startswith("province_n_")]
edu_cols = [c for c in df.columns if c.startswith("edu_")]
eth_cols = [c for c in df.columns if c.startswith("eth_")]
st_cols = [c for c in df.columns if c.startswith("st_")]

# Full control set (matching Stata's $contr)
FULL_CONTROLS = month_cols + province_cols + eth_cols + edu_cols + ["female"] + st_cols

print(f"z_depression: mean={df['z_depression'].mean():.4f}, sd={df['z_depression'].std():.4f}, "
      f"ctrl_mean={df.loc[control_mask, 'z_depression'].mean():.4f}")
print(f"Full controls: {len(FULL_CONTROLS)} variables")

# ── Helper functions ───────────────────────────────────────────────────

def run_rd_spec(outcome, bw, poly_order, controls, cluster_var, donut=0, label=""):
    """Run a single RD specification and return results dict."""
    try:
        mask = df["dif"].between(-bw, bw)
        if donut > 0:
            mask = mask & (df["dif"].abs() >= donut)

        sub = df[mask].copy()

        rd_terms = ["dif", "dif_x_treat"]
        if poly_order >= 2:
            rd_terms += ["dif2", "dif2_x_treat"]

        all_rhs = ["before1955"] + rd_terms + controls
        formula = f"{outcome} ~ " + " + ".join(all_rhs)

        needed_cols = [outcome, "before1955", "dif", "dif_x_treat"] + controls
        if poly_order >= 2:
            needed_cols += ["dif2", "dif2_x_treat"]
        if cluster_var:
            needed_cols.append(cluster_var)
        sub = sub.dropna(subset=[c for c in needed_cols if c in sub.columns])

        if len(sub) < 10:
            raise ValueError(f"Too few observations: {len(sub)}")

        if cluster_var:
            model = pf.feols(formula, data=sub, vcov={"CRV1": cluster_var})
        else:
            model = pf.feols(formula, data=sub, vcov="hetero")

        coef = float(model.coef().get("before1955", np.nan))
        se = float(model.se().get("before1955", np.nan))
        pval = float(model.pvalue().get("before1955", np.nan))
        nobs = int(model._N)
        r2 = float(model._r2)

        ci = model.confint()
        if "before1955" in ci.index:
            ci_lower = float(ci.loc["before1955", "2.5%"])
            ci_upper = float(ci.loc["before1955", "97.5%"])
        else:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

        all_coefs = {k: float(v) for k, v in model.coef().items()}

        return {
            "success": True,
            "coefficient": coef,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficients": all_coefs,
            "model": model,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "exception": e,
        }


def make_row(spec_run_id, spec_id, spec_tree_path, baseline_group_id, outcome_var,
             treatment_var, result, controls_desc="", sample_desc="", fixed_effects="",
             cluster_var="modate", extra_payload=None):
    """Create a CSV row dict.

    extra_payload: dict of axis blocks to include as top-level keys in
    coefficient_vector_json (e.g., controls, sample, functional_form, joint).
    """
    if result["success"]:
        payload = make_success_payload(
            coefficients=result["coefficients"],
            inference={"spec_id": CANONICAL_INFERENCE["spec_id"],
                       "params": CANONICAL_INFERENCE["params"]},
            software=SW_BLOCK,
            surface_hash=SURF_HASH,
            design={"regression_discontinuity": DESIGN_AUDIT},
            blocks=extra_payload,
        )
        return {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": result["coefficient"],
            "std_error": result["std_error"],
            "p_value": result["p_value"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "n_obs": result["n_obs"],
            "r_squared": result["r_squared"],
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        }
    else:
        err_str = safe_single_line(result["error"])
        ed = error_details_from_exception(result.get("exception", Exception(result["error"])),
                                          stage="estimation")
        payload = make_failure_payload(error=err_str, error_details=ed,
                                       software=SW_BLOCK, surface_hash=SURF_HASH)
        return {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_str,
        }


# ── Run specifications ─────────────────────────────────────────────────
rows = []
inference_rows = []
run_counter = 0

def next_run_id():
    global run_counter
    run_counter += 1
    return f"{PAPER_ID}_run{run_counter:03d}"

# ────── BASELINE ──────
print("\n=== BASELINE SPECS ===")

# Primary baseline: Table 4, Col 2 (bw=30)
result = run_rd_spec("z_depression", bw=30, poly_order=1, controls=FULL_CONTROLS,
                     cluster_var="modate", label="baseline_bw30")
rows.append(make_row(
    spec_run_id=next_run_id(), spec_id="baseline",
    spec_tree_path="designs/regression_discontinuity.md#baseline",
    baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
    result=result, controls_desc="full (month/province/ethnicity/education/female/surveytaker FE)",
    sample_desc="bw=30, non_response==0", fixed_effects="month, province, ethnicity, education, survey_taker_id",
    cluster_var="modate"
))
if result["success"]:
    print(f"  Baseline (bw=30): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, "
          f"p={result['p_value']:.4f}, N={result['n_obs']}")

# Additional baseline specs from Table 4
for bw, col_label in [(17, "col1_bw17"), (45, "col3_bw45"), (60, "col4_bw60")]:
    result = run_rd_spec("z_depression", bw=bw, poly_order=1, controls=FULL_CONTROLS,
                         cluster_var="modate", label=f"baseline_bw{bw}")
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=f"baseline__table4_{col_label}",
        spec_tree_path="designs/regression_discontinuity.md#baseline",
        baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
        result=result, controls_desc="full (month/province/ethnicity/education/female/surveytaker FE)",
        sample_desc=f"bw={bw}, non_response==0",
        fixed_effects="month, province, ethnicity, education, survey_taker_id",
        cluster_var="modate"
    ))
    if result["success"]:
        print(f"  Baseline (bw={bw}): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, "
              f"p={result['p_value']:.4f}, N={result['n_obs']}")

# ────── DESIGN VARIANTS ──────
print("\n=== DESIGN VARIANTS ===")

# Bandwidth variations (at baseline controls + linear poly)
for bw in [17, 24, 36, 45, 48, 60, 72]:
    result = run_rd_spec("z_depression", bw=bw, poly_order=1, controls=FULL_CONTROLS,
                         cluster_var="modate", label=f"design_bw{bw}")
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=f"design/regression_discontinuity/bandwidth/bw{bw}",
        spec_tree_path="designs/regression_discontinuity.md#b-bandwidth-selection",
        baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
        result=result, controls_desc="full baseline controls",
        sample_desc=f"bw={bw}, non_response==0",
        fixed_effects="month, province, ethnicity, education, survey_taker_id",
        cluster_var="modate"
    ))
    if result["success"]:
        print(f"  Design bw={bw}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, N={result['n_obs']}")

# Polynomial order: quadratic at baseline bw=30
result = run_rd_spec("z_depression", bw=30, poly_order=2, controls=FULL_CONTROLS,
                     cluster_var="modate", label="design_poly2")
rows.append(make_row(
    spec_run_id=next_run_id(), spec_id="design/regression_discontinuity/poly/local_quadratic",
    spec_tree_path="designs/regression_discontinuity.md#c-local-polynomial-order",
    baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
    result=result, controls_desc="full baseline controls",
    sample_desc="bw=30, poly=2, non_response==0",
    fixed_effects="month, province, ethnicity, education, survey_taker_id",
    cluster_var="modate"
))
if result["success"]:
    print(f"  Design poly=2, bw=30: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── ROBUSTNESS CHECKS: CONTROLS LOO ──────
print("\n=== RC: CONTROLS LOO ===")

control_blocks = {
    "month_fe": month_cols,
    "province_fe": province_cols,
    "ethnicity_fe": eth_cols,
    "education_fe": edu_cols,
    "female": ["female"],
    "survey_taker_fe": st_cols,
}

for block_name, block_cols in control_blocks.items():
    remaining = [c for c in FULL_CONTROLS if c not in block_cols]
    result = run_rd_spec("z_depression", bw=30, poly_order=1, controls=remaining,
                         cluster_var="modate", label=f"loo_{block_name}")
    axis_block = {
        "spec_id": f"rc/controls/loo/drop_{block_name}",
        "family": "loo",
        "dropped": [block_name],
        "n_controls": len(remaining),
    }
    payload_extra = {"controls": axis_block}
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=f"rc/controls/loo/drop_{block_name}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
        result=result, controls_desc=f"full minus {block_name}",
        sample_desc="bw=30, non_response==0",
        fixed_effects=f"all except {block_name}",
        cluster_var="modate", extra_payload=payload_extra
    ))
    if result["success"]:
        print(f"  LOO drop {block_name}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── ROBUSTNESS CHECKS: CONTROL SETS ──────
print("\n=== RC: CONTROL SETS ===")

# No controls (RD polynomial only)
result = run_rd_spec("z_depression", bw=30, poly_order=1, controls=[],
                     cluster_var="modate", label="no_controls")
rows.append(make_row(
    spec_run_id=next_run_id(), spec_id="rc/controls/sets/no_controls",
    spec_tree_path="modules/robustness/controls.md#control-sets",
    baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
    result=result, controls_desc="none (RD polynomial only)",
    sample_desc="bw=30, non_response==0", fixed_effects="none", cluster_var="modate",
    extra_payload={"controls": {"spec_id": "rc/controls/sets/no_controls", "family": "sets", "set": "none", "n_controls": 0}}
))
if result["success"]:
    print(f"  No controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Minimal demographics (female + education + ethnicity)
minimal_controls = ["female"] + edu_cols + eth_cols
result = run_rd_spec("z_depression", bw=30, poly_order=1, controls=minimal_controls,
                     cluster_var="modate", label="minimal_demo")
rows.append(make_row(
    spec_run_id=next_run_id(), spec_id="rc/controls/sets/minimal_demo",
    spec_tree_path="modules/robustness/controls.md#control-sets",
    baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
    result=result, controls_desc="minimal (female + education + ethnicity FE)",
    sample_desc="bw=30, non_response==0", fixed_effects="education, ethnicity",
    cluster_var="modate",
    extra_payload={"controls": {"spec_id": "rc/controls/sets/minimal_demo", "family": "sets", "set": "minimal_demo", "n_controls": len(minimal_controls)}}
))
if result["success"]:
    print(f"  Minimal controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Full baseline (explicit RC row)
result = run_rd_spec("z_depression", bw=30, poly_order=1, controls=FULL_CONTROLS,
                     cluster_var="modate", label="full_controls")
rows.append(make_row(
    spec_run_id=next_run_id(), spec_id="rc/controls/sets/full_baseline",
    spec_tree_path="modules/robustness/controls.md#control-sets",
    baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
    result=result, controls_desc="full baseline controls",
    sample_desc="bw=30, non_response==0",
    fixed_effects="month, province, ethnicity, education, survey_taker_id",
    cluster_var="modate",
    extra_payload={"controls": {"spec_id": "rc/controls/sets/full_baseline", "family": "sets", "set": "full_baseline", "n_controls": len(FULL_CONTROLS)}}
))
if result["success"]:
    print(f"  Full controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── ROBUSTNESS CHECKS: DONUT HOLES ──────
print("\n=== RC: DONUT HOLES ===")

for donut in [1, 2, 3]:
    result = run_rd_spec("z_depression", bw=30, poly_order=1, controls=FULL_CONTROLS,
                         cluster_var="modate", donut=donut, label=f"donut_{donut}")
    sample_block = {
        "spec_id": f"rc/sample/donut/exclude_{donut}month",
        "rule": f"exclude |dif| < {donut}",
        "donut_radius": donut,
    }
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=f"rc/sample/donut/exclude_{donut}month",
        spec_tree_path="modules/robustness/sample.md#donut-hole",
        baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
        result=result, controls_desc="full baseline controls",
        sample_desc=f"bw=30, donut={donut}mo, non_response==0",
        fixed_effects="month, province, ethnicity, education, survey_taker_id",
        cluster_var="modate", extra_payload={"sample": sample_block}
    ))
    if result["success"]:
        print(f"  Donut {donut}mo: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, N={result['n_obs']}")

# ────── ROBUSTNESS CHECKS: ALTERNATIVE OUTCOMES ──────
print("\n=== RC: ALTERNATIVE OUTCOMES ===")

for alt_outcome, alt_label in [("sum_srq", "sum_srq"), ("z_somatic", "z_somatic"),
                                ("z_nonsomatic", "z_nonsomatic")]:
    result = run_rd_spec(alt_outcome, bw=30, poly_order=1, controls=FULL_CONTROLS,
                         cluster_var="modate", label=f"outcome_{alt_label}")
    ff_block = {
        "spec_id": f"rc/form/outcome/{alt_label}",
        "transform": "alternative_index",
        "interpretation": f"RD effect on {alt_label} at age-65 curfew cutoff",
    }
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=f"rc/form/outcome/{alt_label}",
        spec_tree_path="modules/robustness/functional_form.md",
        baseline_group_id="G1", outcome_var=alt_outcome, treatment_var="before1955",
        result=result, controls_desc="full baseline controls",
        sample_desc="bw=30, non_response==0",
        fixed_effects="month, province, ethnicity, education, survey_taker_id",
        cluster_var="modate", extra_payload={"functional_form": ff_block}
    ))
    if result["success"]:
        print(f"  Outcome {alt_label}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── JOINT SPECS: BW x POLYNOMIAL ──────
print("\n=== RC: JOINT BW x POLY ===")

for bw in [30, 45, 60]:
    result = run_rd_spec("z_depression", bw=bw, poly_order=2, controls=FULL_CONTROLS,
                         cluster_var="modate", label=f"joint_bw{bw}_poly2")
    _sid = f"rc/joint/bw_poly/bw{bw}_poly2"
    joint_block = {
        "spec_id": _sid,
        "axes_changed": ["design/bandwidth", "design/poly"],
        "details": {"bandwidth": bw, "poly_order": 2},
    }
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=_sid,
        spec_tree_path="modules/robustness/joint.md",
        baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
        result=result, controls_desc="full baseline controls",
        sample_desc=f"bw={bw}, poly=2, non_response==0",
        fixed_effects="month, province, ethnicity, education, survey_taker_id",
        cluster_var="modate", extra_payload={"joint": joint_block}
    ))
    if result["success"]:
        print(f"  Joint bw={bw}, poly=2: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── JOINT SPECS: BW x NO CONTROLS ──────
print("\n=== RC: JOINT BW x NO CONTROLS ===")

for bw in [17, 45, 60]:
    result = run_rd_spec("z_depression", bw=bw, poly_order=1, controls=[],
                         cluster_var="modate", label=f"joint_bw{bw}_nocontrols")
    _sid = f"rc/joint/bw_controls/bw{bw}_nocontrols"
    joint_block = {
        "spec_id": _sid,
        "axes_changed": ["design/bandwidth", "controls"],
        "details": {"bandwidth": bw, "controls": "none"},
    }
    rows.append(make_row(
        spec_run_id=next_run_id(), spec_id=_sid,
        spec_tree_path="modules/robustness/joint.md",
        baseline_group_id="G1", outcome_var="z_depression", treatment_var="before1955",
        result=result, controls_desc="none",
        sample_desc=f"bw={bw}, no controls, non_response==0",
        fixed_effects="none", cluster_var="modate",
        extra_payload={"joint": joint_block}
    ))
    if result["success"]:
        print(f"  Joint bw={bw}, no controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── JOINT SPECS: OUTCOME x BW ──────
print("\n=== RC: JOINT OUTCOME x BW ===")

for alt_outcome in ["sum_srq", "z_somatic", "z_nonsomatic"]:
    for bw in [17, 45, 60]:
        result = run_rd_spec(alt_outcome, bw=bw, poly_order=1, controls=FULL_CONTROLS,
                             cluster_var="modate", label=f"joint_{alt_outcome}_bw{bw}")
        _sid = f"rc/joint/outcome_bw/{alt_outcome}_bw{bw}"
        joint_block = {
            "spec_id": _sid,
            "axes_changed": ["form/outcome", "design/bandwidth"],
            "details": {"outcome": alt_outcome, "bandwidth": bw},
        }
        ff_block = {
            "spec_id": _sid,
            "transform": "alternative_index",
            "interpretation": f"RD effect on {alt_outcome} at age-65 curfew cutoff, bw={bw}",
        }
        rows.append(make_row(
            spec_run_id=next_run_id(),
            spec_id=f"rc/joint/outcome_bw/{alt_outcome}_bw{bw}",
            spec_tree_path="modules/robustness/joint.md",
            baseline_group_id="G1", outcome_var=alt_outcome, treatment_var="before1955",
            result=result, controls_desc="full baseline controls",
            sample_desc=f"bw={bw}, non_response==0",
            fixed_effects="month, province, ethnicity, education, survey_taker_id",
            cluster_var="modate",
            extra_payload={"joint": joint_block, "functional_form": ff_block}
        ))
        if result["success"]:
            print(f"  Joint {alt_outcome}, bw={bw}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ────── JOINT SPECS: OUTCOME x DONUT ──────
print("\n=== RC: JOINT OUTCOME x DONUT ===")

for alt_outcome in ["sum_srq", "z_somatic", "z_nonsomatic"]:
    for donut in [2]:
        result = run_rd_spec(alt_outcome, bw=30, poly_order=1, controls=FULL_CONTROLS,
                             cluster_var="modate", donut=donut,
                             label=f"joint_{alt_outcome}_donut{donut}")
        _sid = f"rc/joint/outcome_donut/{alt_outcome}_donut{donut}"
        joint_block = {
            "spec_id": _sid,
            "axes_changed": ["form/outcome", "sample/donut"],
            "details": {"outcome": alt_outcome, "donut": donut},
        }
        ff_block = {
            "spec_id": _sid,
            "transform": "alternative_index",
            "interpretation": f"RD effect on {alt_outcome} at age-65 curfew cutoff, donut={donut}",
        }
        rows.append(make_row(
            spec_run_id=next_run_id(),
            spec_id=f"rc/joint/outcome_donut/{alt_outcome}_donut{donut}",
            spec_tree_path="modules/robustness/joint.md",
            baseline_group_id="G1", outcome_var=alt_outcome, treatment_var="before1955",
            result=result, controls_desc="full baseline controls",
            sample_desc=f"bw=30, donut={donut}mo, non_response==0",
            fixed_effects="month, province, ethnicity, education, survey_taker_id",
            cluster_var="modate",
            extra_payload={"joint": joint_block, "functional_form": ff_block}
        ))
        if result["success"]:
            print(f"  Joint {alt_outcome}, donut={donut}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")


# ────── INFERENCE VARIANTS ──────
print("\n=== INFERENCE VARIANTS ===")

# Recompute baseline under alternative inference
baseline_specs_for_inference = [
    ("baseline", "z_depression", 30, 1, FULL_CONTROLS, 0),
]

inf_counter = 0

for base_spec_id, outcome, bw, poly, controls, donut in baseline_specs_for_inference:
    base_run_id = None
    for r in rows:
        if r["spec_id"] == base_spec_id:
            base_run_id = r["spec_run_id"]
            break
    if base_run_id is None:
        continue

    # HC1 (no clustering)
    inf_counter += 1
    try:
        mask = df["dif"].between(-bw, bw)
        if donut > 0:
            mask = mask & (df["dif"].abs() >= donut)
        sub = df[mask].copy()
        rd_terms = ["dif", "dif_x_treat"]
        if poly >= 2:
            rd_terms += ["dif2", "dif2_x_treat"]
        all_rhs = ["before1955"] + rd_terms + controls
        formula = f"{outcome} ~ " + " + ".join(all_rhs)
        needed_cols = [outcome, "before1955", "dif", "dif_x_treat"] + controls
        sub = sub.dropna(subset=[c for c in needed_cols if c in sub.columns])

        model_hc1 = pf.feols(formula, data=sub, vcov="hetero")
        coef_hc1 = float(model_hc1.coef().get("before1955", np.nan))
        se_hc1 = float(model_hc1.se().get("before1955", np.nan))
        pval_hc1 = float(model_hc1.pvalue().get("before1955", np.nan))
        ci_hc1 = model_hc1.confint()
        ci_lower_hc1 = float(ci_hc1.loc["before1955", "2.5%"]) if "before1955" in ci_hc1.index else np.nan
        ci_upper_hc1 = float(ci_hc1.loc["before1955", "97.5%"]) if "before1955" in ci_hc1.index else np.nan

        all_coefs_hc1 = {k: float(v) for k, v in model_hc1.coef().items()}
        payload_hc1 = make_success_payload(
            coefficients=all_coefs_hc1,
            inference={"spec_id": "infer/se/hc/hc1", "params": {}},
            software=SW_BLOCK,
            surface_hash=SURF_HASH,
            design={"regression_discontinuity": DESIGN_AUDIT},
        )

        inference_rows.append({
            "paper_id": PAPER_ID,
            "inference_run_id": f"{PAPER_ID}_inf{inf_counter:03d}",
            "spec_run_id": base_run_id,
            "spec_id": "infer/se/hc/hc1",
            "spec_tree_path": "modules/inference/standard_errors.md#heteroskedasticity-robust-hc",
            "baseline_group_id": "G1",
            "outcome_var": outcome,
            "treatment_var": "before1955",
            "cluster_var": "",
            "coefficient": coef_hc1,
            "std_error": se_hc1,
            "p_value": pval_hc1,
            "ci_lower": ci_lower_hc1,
            "ci_upper": ci_upper_hc1,
            "n_obs": int(model_hc1._N),
            "r_squared": float(model_hc1._r2),
            "coefficient_vector_json": json.dumps(payload_hc1),
            "run_success": 1,
            "run_error": "",
        })
        print(f"  HC1: coef={coef_hc1:.4f}, se={se_hc1:.4f}, p={pval_hc1:.4f}")
    except Exception as e:
        ed = error_details_from_exception(e, stage="inference_hc1")
        payload_fail = make_failure_payload(error=str(e), error_details=ed,
                                             software=SW_BLOCK, surface_hash=SURF_HASH)
        inference_rows.append({
            "paper_id": PAPER_ID,
            "inference_run_id": f"{PAPER_ID}_inf{inf_counter:03d}",
            "spec_run_id": base_run_id,
            "spec_id": "infer/se/hc/hc1",
            "spec_tree_path": "modules/inference/standard_errors.md#heteroskedasticity-robust-hc",
            "baseline_group_id": "G1",
            "outcome_var": outcome,
            "treatment_var": "before1955",
            "cluster_var": "",
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload_fail),
            "run_success": 0,
            "run_error": safe_single_line(str(e)),
        })

    # Province clustering
    inf_counter += 1
    try:
        model_prov = pf.feols(formula, data=sub, vcov={"CRV1": "province_n"})
        coef_prov = float(model_prov.coef().get("before1955", np.nan))
        se_prov = float(model_prov.se().get("before1955", np.nan))
        pval_prov = float(model_prov.pvalue().get("before1955", np.nan))
        ci_prov = model_prov.confint()
        ci_lower_prov = float(ci_prov.loc["before1955", "2.5%"]) if "before1955" in ci_prov.index else np.nan
        ci_upper_prov = float(ci_prov.loc["before1955", "97.5%"]) if "before1955" in ci_prov.index else np.nan

        all_coefs_prov = {k: float(v) for k, v in model_prov.coef().items()}
        payload_prov = make_success_payload(
            coefficients=all_coefs_prov,
            inference={"spec_id": "infer/se/cluster/province", "params": {"cluster_var": "province_n"}},
            software=SW_BLOCK,
            surface_hash=SURF_HASH,
            design={"regression_discontinuity": DESIGN_AUDIT},
        )

        inference_rows.append({
            "paper_id": PAPER_ID,
            "inference_run_id": f"{PAPER_ID}_inf{inf_counter:03d}",
            "spec_run_id": base_run_id,
            "spec_id": "infer/se/cluster/province",
            "spec_tree_path": "modules/inference/standard_errors.md#cluster-robust",
            "baseline_group_id": "G1",
            "outcome_var": outcome,
            "treatment_var": "before1955",
            "cluster_var": "province_n",
            "coefficient": coef_prov,
            "std_error": se_prov,
            "p_value": pval_prov,
            "ci_lower": ci_lower_prov,
            "ci_upper": ci_upper_prov,
            "n_obs": int(model_prov._N),
            "r_squared": float(model_prov._r2),
            "coefficient_vector_json": json.dumps(payload_prov),
            "run_success": 1,
            "run_error": "",
        })
        print(f"  Province cluster: coef={coef_prov:.4f}, se={se_prov:.4f}, p={pval_prov:.4f}")
    except Exception as e:
        ed = error_details_from_exception(e, stage="inference_province_cluster")
        payload_fail = make_failure_payload(error=str(e), error_details=ed,
                                             software=SW_BLOCK, surface_hash=SURF_HASH)
        inference_rows.append({
            "paper_id": PAPER_ID,
            "inference_run_id": f"{PAPER_ID}_inf{inf_counter:03d}",
            "spec_run_id": base_run_id,
            "spec_id": "infer/se/cluster/province",
            "spec_tree_path": "modules/inference/standard_errors.md#cluster-robust",
            "baseline_group_id": "G1",
            "outcome_var": outcome,
            "treatment_var": "before1955",
            "cluster_var": "province_n",
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload_fail),
            "run_success": 0,
            "run_error": safe_single_line(str(e)),
        })

# ── Write outputs ──────────────────────────────────────────────────────
print(f"\n=== WRITING OUTPUTS ===")
print(f"Total specification rows: {len(rows)}")
print(f"Total inference rows: {len(inference_rows)}")

# specification_results.csv
spec_df = pd.DataFrame(rows)
spec_df.to_csv(OUT_DIR / "specification_results.csv", index=False)
print(f"Wrote specification_results.csv ({len(spec_df)} rows)")

# inference_results.csv
if inference_rows:
    inf_df = pd.DataFrame(inference_rows)
    inf_df.to_csv(OUT_DIR / "inference_results.csv", index=False)
    print(f"Wrote inference_results.csv ({len(inf_df)} rows)")

# Summary stats
n_success = spec_df["run_success"].sum()
n_fail = len(spec_df) - n_success
print(f"\nSuccess: {n_success}, Failures: {n_fail}")

if n_success > 0:
    successful = spec_df[spec_df["run_success"] == 1]
    print(f"\nCoefficient range: [{successful['coefficient'].min():.4f}, {successful['coefficient'].max():.4f}]")
    print(f"P-value range: [{successful['p_value'].min():.4f}, {successful['p_value'].max():.4f}]")
    sig_05 = (successful["p_value"] < 0.05).sum()
    sig_10 = (successful["p_value"] < 0.10).sum()
    print(f"Significant at 5%: {sig_05}/{n_success} ({100*sig_05/n_success:.1f}%)")
    print(f"Significant at 10%: {sig_10}/{n_success} ({100*sig_10/n_success:.1f}%)")

print("\nDone!")
