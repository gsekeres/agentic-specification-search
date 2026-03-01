#!/usr/bin/env python3
"""
Specification search script for 138922-V1:
"The Long-Run Effects of Sports Club Vouchers for Primary School Children"
Marcus, Siedler & Ziebarth, AEJ: Economic Policy

Surface-driven execution of 54 core specifications for baseline group G1.
Design: Difference-in-Differences (repeated cross-section).
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import hashlib
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/138922-V1"
PAPER_ID = "138922-V1"

# ============================================================
# LOAD SURFACE
# ============================================================
with open(f"{PACKAGE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface = json.load(f)

surface_hash_val = "sha256:" + hashlib.sha256(
    json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()

# ============================================================
# SOFTWARE BLOCK
# ============================================================
software_block = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pyfixest": pf.__version__ if hasattr(pf, '__version__') else "0.40+",
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
}

# ============================================================
# DESIGN AUDIT BLOCK (from surface)
# ============================================================
bg = surface["baseline_groups"][0]
design_audit = bg["design_audit"]
design_block = {"difference_in_differences": design_audit}

# ============================================================
# CANONICAL INFERENCE
# ============================================================
canonical_inference = bg["inference_plan"]["canonical"]

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
df = pd.read_stata(f"{PACKAGE_DIR}/Data/MSZ_main-data.dta")

# Convert categorical variables to numeric
def extract_bula_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    parts = s.split('.')
    try:
        return float(parts[0])
    except:
        return np.nan

def extract_year(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    if '/' in s:
        return float(s.split('/')[0])
    try:
        return float(s)
    except:
        return np.nan

# Apply conversions (must convert to list first to avoid categorical output)
for col in ['bula_3rd', 'bula_1st', 'bula']:
    df[col] = pd.Series([extract_bula_code(x) for x in df[col]], index=df.index, dtype='float64')

for col in ['year_3rd']:
    df[col] = pd.Series([extract_year(x) for x in df[col]], index=df.index, dtype='float64')

# Convert other categoricals to numeric (extract leading number from German labels)
for col in ['female', 'siblings', 'born_germany', 'newspaper', 'art_at_home', 'sport_hrs']:
    if df[col].dtype.name == 'category':
        df[col] = pd.Series([extract_bula_code(x) for x in df[col]], index=df.index, dtype='float64')

# Ensure all numeric types are float64 for pyfixest
for col in df.columns:
    if df[col].dtype in ['float32']:
        df[col] = df[col].astype('float64')

# Create integer versions for FE (pyfixest needs clean integer-like values)
df['year_3rd_fe'] = df['year_3rd'].copy()
df['bula_3rd_fe'] = df['bula_3rd'].copy()
df['cityno_fe'] = df['cityno'].copy()
df['bula_1st_fe'] = df['bula_1st'].copy()
df['bula_fe'] = df['bula'].copy()
df['year_1st_fe'] = df['year_1st'].copy()

# ============================================================
# SAMPLE DEFINITIONS
# ============================================================
def baseline_sample(data):
    """Standard baseline sample: 3 states, target==1, nonmiss==1, 2006-2010"""
    mask = (data['bula_3rd'].isin([4, 13, 16]) &
            (data['target'] == 1) &
            (data['nonmiss'] == 1) &
            data['year_3rd'].between(2006, 2010))
    return data[mask].copy()

# ============================================================
# HELPER: RUN A SINGLE SPEC AND EXTRACT RESULTS
# ============================================================
results = []
inference_results = []

def run_spec(spec_id, spec_run_id, spec_tree_path, outcome_var, treatment_var,
             data, formula, vcov, sample_desc, fixed_effects_desc, controls_desc,
             cluster_var, baseline_group_id="G1", extra_payload=None, weights_col=None):
    """Run a single specification and append to results."""
    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects_desc,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
    }

    try:
        # Drop NaN in regression variables
        reg_data = data.dropna(subset=[outcome_var, treatment_var])

        # Determine FE vars from formula (after |)
        fe_part = formula.split("|")[1].strip() if "|" in formula else ""
        rhs_part = formula.split("~")[1].split("|")[0].strip()
        rhs_vars = [v.strip() for v in rhs_part.split("+") if v.strip()]
        fe_vars = [v.strip() for v in fe_part.split("+") if v.strip()] if fe_part else []

        all_vars = [outcome_var] + rhs_vars + fe_vars
        if cluster_var and cluster_var != "":
            if isinstance(cluster_var, list):
                all_vars.extend(cluster_var)
            else:
                all_vars.append(cluster_var)
        if weights_col:
            all_vars.append(weights_col)

        all_vars = [v for v in all_vars if v in reg_data.columns]
        reg_data = reg_data.dropna(subset=all_vars)

        if weights_col:
            model = pf.feols(formula, data=reg_data, vcov=vcov, weights=weights_col)
        else:
            model = pf.feols(formula, data=reg_data, vcov=vcov)

        coef_val = float(model.coef().get(treatment_var, np.nan))
        se_val = float(model.se().get(treatment_var, np.nan))
        pval = float(model.pvalue().get(treatment_var, np.nan))
        ci = model.confint()
        if treatment_var in ci.index:
            ci_lower = float(ci.loc[treatment_var].iloc[0])
            ci_upper = float(ci.loc[treatment_var].iloc[1])
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        tstat = float(model.tstat().get(treatment_var, np.nan))
        n_obs = int(model._N)
        r2 = float(model._r2)

        # Full coefficient vector
        all_coefs = {k: float(v) for k, v in model.coef().items()}

        payload = {
            "coefficients": all_coefs,
            "inference": {"spec_id": canonical_inference["spec_id"],
                         "params": canonical_inference["params"]},
            "software": software_block,
            "surface_hash": surface_hash_val,
            "design": design_block,
        }
        if extra_payload:
            payload.update(extra_payload)

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = str(e).replace("\n", " ")[:240]
        error_details = {
            "stage": "estimation",
            "exception_type": type(e).__name__,
            "exception_message": str(e)[:500],
            "traceback_tail": "\n".join(tb.splitlines()[-10:]),
        }
        fail_payload = {
            "error": error_msg,
            "error_details": error_details,
        }
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(fail_payload),
            "run_success": 0,
            "run_error": error_msg,
        })

    results.append(row)
    return row


def run_inference_variant(base_row, infer_spec_id, infer_tree_path, data, formula,
                          vcov_new, cluster_var_new, infer_run_id, weights_col=None):
    """Re-estimate with alternative inference and record to inference_results."""
    treatment_var = base_row["treatment_var"]
    outcome_var = base_row["outcome_var"]
    row = {
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": base_row["spec_run_id"],
        "spec_id": infer_spec_id,
        "spec_tree_path": infer_tree_path,
        "baseline_group_id": base_row["baseline_group_id"],
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "cluster_var": str(cluster_var_new) if cluster_var_new else "",
    }

    try:
        reg_data = data.dropna(subset=[outcome_var, treatment_var])

        fe_part = formula.split("|")[1].strip() if "|" in formula else ""
        rhs_part = formula.split("~")[1].split("|")[0].strip()
        rhs_vars = [v.strip() for v in rhs_part.split("+") if v.strip()]
        fe_vars = [v.strip() for v in fe_part.split("+") if v.strip()] if fe_part else []

        all_vars = [outcome_var] + rhs_vars + fe_vars
        if isinstance(cluster_var_new, list):
            all_vars.extend([c for c in cluster_var_new if c in reg_data.columns])
        elif cluster_var_new and cluster_var_new in reg_data.columns:
            all_vars.append(cluster_var_new)
        if weights_col:
            all_vars.append(weights_col)
        all_vars = [v for v in all_vars if v in reg_data.columns]
        reg_data = reg_data.dropna(subset=all_vars)

        if weights_col:
            model = pf.feols(formula, data=reg_data, vcov=vcov_new, weights=weights_col)
        else:
            model = pf.feols(formula, data=reg_data, vcov=vcov_new)

        coef_val = float(model.coef().get(treatment_var, np.nan))
        se_val = float(model.se().get(treatment_var, np.nan))
        pval = float(model.pvalue().get(treatment_var, np.nan))
        ci = model.confint()
        if treatment_var in ci.index:
            ci_lower = float(ci.loc[treatment_var].iloc[0])
            ci_upper = float(ci.loc[treatment_var].iloc[1])
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        n_obs = int(model._N)
        r2 = float(model._r2)

        all_coefs = {k: float(v) for k, v in model.coef().items()}

        payload = {
            "coefficients": all_coefs,
            "inference": {"spec_id": infer_spec_id, "params": {"cluster_var": cluster_var_new}},
            "software": software_block,
            "surface_hash": surface_hash_val,
            "design": design_block,
        }

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = str(e).replace("\n", " ")[:240]
        error_details = {
            "stage": "inference_recomputation",
            "exception_type": type(e).__name__,
            "exception_message": str(e)[:500],
            "traceback_tail": "\n".join(tb.splitlines()[-10:]),
        }
        fail_payload = {
            "error": error_msg,
            "error_details": error_details,
        }
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(fail_payload),
            "run_success": 0,
            "run_error": error_msg,
        })

    inference_results.append(row)
    return row


# ============================================================
# CONTROL VARIABLE DEFINITIONS
# ============================================================
ALL_CONTROLS = ["female", "siblings", "born_germany", "parent_nongermany",
                "newspaper", "art_at_home", "academictrack", "sportsclub_4_7", "music_4_7"]

# ============================================================
# BASELINE FORMULAS
# ============================================================
BASELINE_FE3 = "year_3rd_fe + bula_3rd_fe + cityno_fe"
BASELINE_FE2 = "year_3rd_fe + bula_3rd_fe"

def make_formula(outcome, treatment, controls=None, fe=None):
    """Build a pyfixest formula string."""
    rhs = treatment
    if controls:
        rhs += " + " + " + ".join(controls)
    if fe:
        return f"{outcome} ~ {rhs} | {fe}"
    else:
        return f"{outcome} ~ {rhs}"

# ============================================================
# STEP 1: BASELINE SPECIFICATIONS
# ============================================================
print("=" * 60)
print("STEP 1: Running baseline specifications")
print("=" * 60)

sample_base = baseline_sample(df)
print(f"Baseline sample: N = {len(sample_base)}")

# PRIMARY BASELINE: Table 2, Column 3 - sportsclub
baseline_formula = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
print(f"\nPrimary baseline formula: {baseline_formula}")

base_row = run_spec(
    spec_id="baseline",
    spec_run_id="138922-V1__baseline",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#baseline-required",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base,
    formula=baseline_formula,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states (Saxony, Brandenburg, Thuringia), target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
)
print(f"  baseline: coef={base_row.get('coefficient', 'FAIL'):.4f}, se={base_row.get('std_error', 'FAIL'):.4f}, N={base_row.get('n_obs', 'FAIL')}")

# ADDITIONAL BASELINE: Col1 - minimal FE (group dummies)
# Table 2, Col 1: reg sportsclub treat tbula_3rd tcoh, vce(cluster cityno)
col1_formula = make_formula("sportsclub", "treat + tbula_3rd + tcoh")
run_spec(
    spec_id="baseline__sportsclub_col1",
    spec_run_id="138922-V1__baseline__sportsclub_col1",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#baseline-required",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base,
    formula=col1_formula,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="group dummies: tbula_3rd + tcoh",
    controls_desc="none",
    cluster_var="cityno",
)

# ADDITIONAL BASELINE: Col2 - year + state FE, no city FE
col2_formula = make_formula("sportsclub", "treat", fe=BASELINE_FE2)
run_spec(
    spec_id="baseline__sportsclub_col2",
    spec_run_id="138922-V1__baseline__sportsclub_col2",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#baseline-required",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base,
    formula=col2_formula,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd",
    controls_desc="none",
    cluster_var="cityno",
)

# ADDITIONAL BASELINES: Other outcomes with Col3 spec
for out_var, label in [("sport_hrs", "sport_hrs"), ("oweight", "oweight"),
                        ("kommheard", "kommheard"), ("kommgotten", "kommgotten"),
                        ("kommused", "kommused")]:
    fml = make_formula(out_var, "treat", fe=BASELINE_FE3)
    r = run_spec(
        spec_id=f"baseline__{label}",
        spec_run_id=f"138922-V1__baseline__{label}",
        spec_tree_path="specification_tree/designs/difference_in_differences.md#baseline-required",
        outcome_var=out_var,
        treatment_var="treat",
        data=sample_base,
        formula=fml,
        vcov={"CRV1": "cityno_fe"},
        sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc="none",
        cluster_var="cityno",
    )
    print(f"  baseline__{label}: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

print(f"\nTotal baseline specs: {len(results)}")

# ============================================================
# STEP 2: RC SPECIFICATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Running RC specifications")
print("=" * 60)

# -------------------------------------------------------
# 2A: Controls axis
# -------------------------------------------------------
print("\n--- Controls axis ---")

# rc/controls/sets/full: all 9 individual controls
fml = make_formula("sportsclub", "treat", controls=ALL_CONTROLS, fe=BASELINE_FE3)
r = run_spec(
    spec_id="rc/controls/sets/full",
    spec_run_id="138922-V1__rc_controls_sets_full",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="full (all 9): " + ", ".join(ALL_CONTROLS),
    cluster_var="cityno",
    extra_payload={"controls": {"spec_id": "rc/controls/sets/full", "family": "sets", "n_controls": 9, "controls": ALL_CONTROLS}},
)
print(f"  rc/controls/sets/full: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# rc/controls/loo: leave-one-out from full controls
for ctrl in ALL_CONTROLS:
    loo_controls = [c for c in ALL_CONTROLS if c != ctrl]
    fml = make_formula("sportsclub", "treat", controls=loo_controls, fe=BASELINE_FE3)
    spec_id = f"rc/controls/loo/drop_{ctrl}"
    r = run_spec(
        spec_id=spec_id,
        spec_run_id=f"138922-V1__rc_controls_loo_drop_{ctrl}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
        outcome_var="sportsclub",
        treatment_var="treat",
        data=sample_base,
        formula=fml,
        vcov={"CRV1": "cityno_fe"},
        sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc=f"LOO drop {ctrl}: " + ", ".join(loo_controls),
        cluster_var="cityno",
        extra_payload={"controls": {"spec_id": spec_id, "family": "loo", "dropped": [ctrl], "n_controls": len(loo_controls)}},
    )
    print(f"  {spec_id}: coef={r.get('coefficient', 'FAIL'):.4f}")

# rc/controls/progression: build up controls
progression_sets = {
    "bivariate": [],
    "demographics": ["female", "siblings", "born_germany", "parent_nongermany"],
    "socioeconomic": ["female", "siblings", "born_germany", "parent_nongermany",
                      "newspaper", "art_at_home", "academictrack"],
    "full": ALL_CONTROLS,
}
for prog_name, prog_controls in progression_sets.items():
    fml = make_formula("sportsclub", "treat", controls=prog_controls if prog_controls else None, fe=BASELINE_FE3)
    spec_id = f"rc/controls/progression/{prog_name}"
    r = run_spec(
        spec_id=spec_id,
        spec_run_id=f"138922-V1__rc_controls_progression_{prog_name}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
        outcome_var="sportsclub",
        treatment_var="treat",
        data=sample_base,
        formula=fml,
        vcov={"CRV1": "cityno_fe"},
        sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc=f"progression/{prog_name}: " + (", ".join(prog_controls) if prog_controls else "none"),
        cluster_var="cityno",
        extra_payload={"controls": {"spec_id": spec_id, "family": "progression", "stage": prog_name, "n_controls": len(prog_controls), "controls": prog_controls}},
    )
    print(f"  {spec_id}: coef={r.get('coefficient', 'FAIL'):.4f}")

# rc/controls/subset: random control subsets
rng = np.random.RandomState(138922)
for i in range(1, 11):
    # Draw a random subset size from {1, ..., 9}
    size = rng.randint(1, 10)  # 1 to 9 inclusive
    subset = list(rng.choice(ALL_CONTROLS, size=size, replace=False))
    fml = make_formula("sportsclub", "treat", controls=subset, fe=BASELINE_FE3)
    spec_id = f"rc/controls/subset/random_{i:03d}"
    r = run_spec(
        spec_id=spec_id,
        spec_run_id=f"138922-V1__rc_controls_subset_random_{i:03d}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#subset",
        outcome_var="sportsclub",
        treatment_var="treat",
        data=sample_base,
        formula=fml,
        vcov={"CRV1": "cityno_fe"},
        sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc=f"random subset {i:03d} (n={size}): " + ", ".join(subset),
        cluster_var="cityno",
        extra_payload={"controls": {"spec_id": spec_id, "family": "subset", "draw_index": i,
                                     "n_controls": len(subset), "controls": subset},
                       "sampling": {"seed": 138922, "draw_index": i}},
    )
    print(f"  {spec_id}: coef={r.get('coefficient', 'FAIL'):.4f}, n_controls={size}")

# -------------------------------------------------------
# 2B: Sample axis
# -------------------------------------------------------
print("\n--- Sample axis ---")

# Time window variations
time_specs = [
    ("rc/sample/time/drop_2006", (2007, 2010), "Drop 2006, cohorts 2007-2010"),
    ("rc/sample/time/drop_2007", None, "Drop 2007 only"),  # special handling
    ("rc/sample/time/extend_2000_2010", (2000, 2010), "Extended pre-period: cohorts 2000-2010"),
    ("rc/sample/time/extend_2006_2011", (2006, 2011), "Extended post-period: cohorts 2006-2011"),
    ("rc/sample/time/shorten_2006_2009", (2006, 2009), "Shortened: cohorts 2006-2009"),
]

for spec_id, yr_range, desc in time_specs:
    if spec_id == "rc/sample/time/drop_2007":
        # Drop only 2007
        mask = (df['bula_3rd'].isin([4, 13, 16]) &
                (df['target'] == 1) & (df['nonmiss'] == 1) &
                df['year_3rd'].between(2006, 2010) & (df['year_3rd'] != 2007))
        sub = df[mask].copy()
    else:
        mask = (df['bula_3rd'].isin([4, 13, 16]) &
                (df['target'] == 1) & (df['nonmiss'] == 1) &
                df['year_3rd'].between(yr_range[0], yr_range[1]))
        sub = df[mask].copy()

    fml = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
    r = run_spec(
        spec_id=spec_id,
        spec_run_id=f"138922-V1__{spec_id.replace('/', '_')}",
        spec_tree_path="specification_tree/modules/robustness/sample.md#time",
        outcome_var="sportsclub",
        treatment_var="treat",
        data=sub,
        formula=fml,
        vcov={"CRV1": "cityno_fe"},
        sample_desc=f"3 states, target==1, nonmiss==1; {desc}",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc="none",
        cluster_var="cityno",
        extra_payload={"sample": {"spec_id": spec_id, "family": "time", "description": desc}},
    )
    print(f"  {spec_id}: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# State variations
# Only Thuringia: Saxony + Thuringia (13, 16)
mask = (df['bula_3rd'].isin([13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) & df['year_3rd'].between(2006, 2010))
sub = df[mask].copy()
fml = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
r = run_spec(
    spec_id="rc/sample/states/only_thuringia",
    spec_run_id="138922-V1__rc_sample_states_only_thuringia",
    spec_tree_path="specification_tree/modules/robustness/sample.md#geographic",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="Saxony + Thuringia only (drop Brandenburg), target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/states/only_thuringia", "family": "states", "states": [13, 16]}},
)
print(f"  rc/sample/states/only_thuringia: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# Only Brandenburg: Saxony + Brandenburg (4, 13)
mask = (df['bula_3rd'].isin([4, 13]) & (df['target'] == 1) & (df['nonmiss'] == 1) & df['year_3rd'].between(2006, 2010))
sub = df[mask].copy()
r = run_spec(
    spec_id="rc/sample/states/only_brandenburg",
    spec_run_id="138922-V1__rc_sample_states_only_brandenburg",
    spec_tree_path="specification_tree/modules/robustness/sample.md#geographic",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="Saxony + Brandenburg only (drop Thuringia), target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/states/only_brandenburg", "family": "states", "states": [4, 13]}},
)
print(f"  rc/sample/states/only_brandenburg: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# Quality filters
# no_missing_filter: drop nonmiss==1 requirement
mask = (df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) & df['year_3rd'].between(2006, 2010))
sub = df[mask].copy()
fml = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
r = run_spec(
    spec_id="rc/sample/quality/no_missing_filter",
    spec_run_id="138922-V1__rc_sample_quality_no_missing_filter",
    spec_tree_path="specification_tree/modules/robustness/sample.md#quality",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, NO nonmiss filter, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/quality/no_missing_filter", "family": "quality", "dropped_filter": "nonmiss==1"}},
)
print(f"  rc/sample/quality/no_missing_filter: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# high_duration_quality: duration>10 & female_check==1 & deutsch_check==1 & dob_check==1
mask = (df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_3rd'].between(2006, 2010) &
        (df['duration'] > 10) & (df['female_check'] == 1) &
        (df['deutsch_check'] == 1) & (df['dob_check'] == 1))
sub = df[mask].copy()
r = run_spec(
    spec_id="rc/sample/quality/high_duration_quality",
    spec_run_id="138922-V1__rc_sample_quality_high_duration_quality",
    spec_tree_path="specification_tree/modules/robustness/sample.md#quality",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, high quality (duration>10, female_check, deutsch_check, dob_check), cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/quality/high_duration_quality", "family": "quality",
                               "additional_filters": "duration>10 & female_check==1 & deutsch_check==1 & dob_check==1"}},
)
print(f"  rc/sample/quality/high_duration_quality: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# no_older_siblings_treatment: sib_part==0 | inrange(year_3rd, 2009, 2010)
mask = (df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_3rd'].between(2006, 2010) &
        ((df['sib_part'] == 0) | df['year_3rd'].between(2009, 2010)))
sub = df[mask].copy()
r = run_spec(
    spec_id="rc/sample/quality/no_older_siblings_treatment",
    spec_run_id="138922-V1__rc_sample_quality_no_older_siblings_treatment",
    spec_tree_path="specification_tree/modules/robustness/sample.md#quality",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, exclude siblings treated (sib_part==0|year>=2009), cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/quality/no_older_siblings_treatment", "family": "quality",
                               "additional_filters": "sib_part==0 | inrange(year_3rd, 2009, 2010)"}},
)
print(f"  rc/sample/quality/no_older_siblings_treatment: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# no_older_siblings_any: anz_osiblings==0
mask = (df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_3rd'].between(2006, 2010) &
        (df['anz_osiblings'] == 0))
sub = df[mask].copy()
r = run_spec(
    spec_id="rc/sample/quality/no_older_siblings_any",
    spec_run_id="138922-V1__rc_sample_quality_no_older_siblings_any",
    spec_tree_path="specification_tree/modules/robustness/sample.md#quality",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, only children (anz_osiblings==0), cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/quality/no_older_siblings_any", "family": "quality",
                               "additional_filters": "anz_osiblings==0"}},
)
print(f"  rc/sample/quality/no_older_siblings_any: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# balanced_movers_excluded: bula_3rd==bula (did not move states)
mask = (df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_3rd'].between(2006, 2010) &
        (df['bula_3rd'] == df['bula']))
sub = df[mask].copy()
r = run_spec(
    spec_id="rc/sample/quality/balanced_movers_excluded",
    spec_run_id="138922-V1__rc_sample_quality_balanced_movers_excluded",
    spec_tree_path="specification_tree/modules/robustness/sample.md#quality",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, non-movers only (bula_3rd==bula), cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"sample": {"spec_id": "rc/sample/quality/balanced_movers_excluded", "family": "quality",
                               "additional_filters": "bula_3rd==bula (exclude movers)"}},
)
print(f"  rc/sample/quality/balanced_movers_excluded: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# -------------------------------------------------------
# 2C: Fixed effects axis
# -------------------------------------------------------
print("\n--- Fixed effects axis ---")

# rc/fe/drop/cityno: year + state FE only
fml = make_formula("sportsclub", "treat", fe=BASELINE_FE2)
r = run_spec(
    spec_id="rc/fe/drop/cityno",
    spec_run_id="138922-V1__rc_fe_drop_cityno",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd (dropped cityno)",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"fixed_effects": {"spec_id": "rc/fe/drop/cityno", "family": "drop", "dropped": ["cityno"]}},
)
print(f"  rc/fe/drop/cityno: coef={r.get('coefficient', 'FAIL'):.4f}")

# rc/fe/drop/bula_3rd: year + city FE only (state absorbed by city)
fml = make_formula("sportsclub", "treat", fe="year_3rd_fe + cityno_fe")
r = run_spec(
    spec_id="rc/fe/drop/bula_3rd",
    spec_run_id="138922-V1__rc_fe_drop_bula_3rd",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + cityno (dropped bula_3rd; subsumed by cityno)",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"fixed_effects": {"spec_id": "rc/fe/drop/bula_3rd", "family": "drop", "dropped": ["bula_3rd"]}},
)
print(f"  rc/fe/drop/bula_3rd: coef={r.get('coefficient', 'FAIL'):.4f}")

# rc/fe/add/bula_3rd_x_year_3rd: add state-by-year interaction FE
# Create interaction variable
sample_base_interact = sample_base.copy()
sample_base_interact['bula_year_fe'] = (sample_base_interact['bula_3rd'].astype(str) + "_" +
                                         sample_base_interact['year_3rd'].astype(str))
# Map to integer codes
bula_year_codes = {v: i for i, v in enumerate(sample_base_interact['bula_year_fe'].unique())}
sample_base_interact['bula_year_fe'] = sample_base_interact['bula_year_fe'].map(bula_year_codes).astype(float)

fml = make_formula("sportsclub", "treat", fe="year_3rd_fe + bula_3rd_fe + cityno_fe + bula_year_fe")
r = run_spec(
    spec_id="rc/fe/add/bula_3rd_x_year_3rd",
    spec_run_id="138922-V1__rc_fe_add_bula_3rd_x_year_3rd",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sample_base_interact,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno + bula_3rd_x_year_3rd (may be collinear with treat)",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"fixed_effects": {"spec_id": "rc/fe/add/bula_3rd_x_year_3rd", "family": "add",
                                      "added": ["bula_3rd x year_3rd"],
                                      "note": "State-by-year FE may absorb treatment effect due to collinearity"}},
)
print(f"  rc/fe/add/bula_3rd_x_year_3rd: coef={r.get('coefficient', 'FAIL')}, N={r.get('n_obs', 'FAIL')}")

# -------------------------------------------------------
# 2D: Treatment definition axis
# -------------------------------------------------------
print("\n--- Treatment definition axis ---")

# rc/data/treatment/1st_grade_timing: use t_tcoh_1st with bula_1st
# Sample: inlist(bula_1st, 4, 13, 16) & inrange(year_1st, 2004, 2008) & target==1 & nonmiss==1
mask = (df['bula_1st'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_1st'].between(2004, 2008))
sub = df[mask].copy()
sub['year_1st_fe'] = sub['year_1st'].copy()
sub['bula_1st_fe'] = sub['bula_1st'].copy()
fml = make_formula("sportsclub", "t_tcoh_1st", fe="year_1st_fe + bula_1st_fe + cityno_fe")
r = run_spec(
    spec_id="rc/data/treatment/1st_grade_timing",
    spec_run_id="138922-V1__rc_data_treatment_1st_grade_timing",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#merge--linkage-robustness",
    outcome_var="sportsclub",
    treatment_var="t_tcoh_1st",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states (bula_1st), target==1, nonmiss==1, 1st-grade years 2004-2008",
    fixed_effects_desc="year_1st + bula_1st + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"data_construction": {"spec_id": "rc/data/treatment/1st_grade_timing", "family": "treatment",
                                          "treatment_var": "t_tcoh_1st", "timing": "1st grade"}},
)
print(f"  rc/data/treatment/1st_grade_timing: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# rc/data/treatment/current_state: use t_tcoh_bula with current state bula
mask = (df['bula'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_3rd'].between(2006, 2010))
sub = df[mask].copy()
sub['bula_curr_fe'] = sub['bula'].copy()
fml = make_formula("sportsclub", "t_tcoh_bula", fe="year_3rd_fe + bula_curr_fe + cityno_fe")
r = run_spec(
    spec_id="rc/data/treatment/current_state",
    spec_run_id="138922-V1__rc_data_treatment_current_state",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#merge--linkage-robustness",
    outcome_var="sportsclub",
    treatment_var="t_tcoh_bula",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states (current bula), target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula (current) + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"data_construction": {"spec_id": "rc/data/treatment/current_state", "family": "treatment",
                                          "treatment_var": "t_tcoh_bula", "timing": "current state of residence"}},
)
print(f"  rc/data/treatment/current_state: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# rc/data/treatment/treat_v2: alternative treatment coding
# Use baseline sample but with treat_v2 instead of treat
fml = make_formula("sportsclub", "treat_v2", fe=BASELINE_FE3)
r = run_spec(
    spec_id="rc/data/treatment/treat_v2",
    spec_run_id="138922-V1__rc_data_treatment_treat_v2",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#merge--linkage-robustness",
    outcome_var="sportsclub",
    treatment_var="treat_v2",
    data=sample_base,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, cohorts 2006-2010",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    extra_payload={"data_construction": {"spec_id": "rc/data/treatment/treat_v2", "family": "treatment",
                                          "treatment_var": "treat_v2", "note": "Alternative treatment coding"}},
)
print(f"  rc/data/treatment/treat_v2: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

# -------------------------------------------------------
# 2E: Weights axis
# -------------------------------------------------------
print("\n--- Weights axis ---")

# rc/weights/main/ebalance_weighted: entropy balanced weights
# The original code creates ebw weights by collapsing to city-year level.
# We need to replicate the ebalance procedure. Since ebalance is a Stata ado,
# we'll approximate using the entropy balancing approach in Python.
# First, try to create the weights following the synthetic control do-file approach.
print("  Constructing entropy balance weights (ebw)...")

try:
    # Follow 03_synthetic_control.do:
    # 1. Keep baseline sample but also restrict to non-movers
    ebw_data = df[df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) &
                   (df['nonmiss'] == 1) & df['year_3rd'].between(2006, 2010) &
                   (df['bula_3rd'] == df['bula'])].copy()

    # Drop wrongly-coded municipalities (from the do file)
    ebw_data = ebw_data[~((ebw_data['cityno'] == 1) & (ebw_data['year_3rd'] == 2007) & (ebw_data['bula_3rd'] == 16))]
    ebw_data = ebw_data[~((ebw_data['cityno'] == 34) & (ebw_data['bula_3rd'] == 16))]
    ebw_data = ebw_data[~((ebw_data['cityno'] == 53) & (ebw_data['bula_3rd'] == 4))]
    ebw_data = ebw_data[~((ebw_data['cityno'] == 70) & (ebw_data['bula_3rd'] == 13))]

    # Collapse to city-year level
    collapse_vars = ['sportsclub', 'sport_hrs', 'oweight', 'kommheard', 'kommgotten', 'kommused'] + \
                    ALL_CONTROLS + ['urban', 'treat', 'bula_3rd']
    ebw_collapsed = ebw_data.groupby(['year_3rd', 'cityno'])[collapse_vars].mean().reset_index()
    ebw_collapsed['wgt'] = ebw_data.groupby(['year_3rd', 'cityno']).size().reset_index(name='wgt')['wgt'].values

    # Balanced panel: keep only cities with all 5 years
    city_counts = ebw_collapsed.groupby('cityno').size()
    balanced_cities = city_counts[city_counts == 5].index
    ebw_collapsed = ebw_collapsed[ebw_collapsed['cityno'].isin(balanced_cities)].copy()

    # Reshape wide
    ebw_wide = ebw_collapsed.pivot(index='cityno', columns='year_3rd', values=['sportsclub', 'sport_hrs', 'oweight'] + ALL_CONTROLS + ['urban', 'wgt', 'treat'])
    ebw_wide.columns = [f"{col[0]}_{int(col[1])}" for col in ebw_wide.columns]
    ebw_wide = ebw_wide.reset_index()

    # Create treat at city level
    ebw_wide['treat'] = ebw_wide['treat_2008']
    ebw_wide['wgt'] = ebw_wide[['wgt_2006', 'wgt_2007', 'wgt_2008', 'wgt_2009', 'wgt_2010']].mean(axis=1)

    # Entropy balancing: match on pre-treatment outcome means
    # Treatment indicator: treat==1 (Saxony cities)
    # We'll use a simple exponential tilting approach
    from scipy.optimize import minimize as scipy_minimize

    treat_mask = ebw_wide['treat'] == 1
    control_mask = ebw_wide['treat'] == 0

    # Matching variables (from ebalance command in do-file: ebw1)
    match_vars_1 = ['sportsclub_2007', 'sport_hrs_2007', 'oweight_2007',
                     'sportsclub_2006', 'sport_hrs_2006', 'oweight_2006']

    X_treat = ebw_wide.loc[treat_mask, match_vars_1].values
    X_control = ebw_wide.loc[control_mask, match_vars_1].values
    w_base = ebw_wide.loc[control_mask, 'wgt'].values

    # Target moments: weighted means of treatment group
    treat_wgt = ebw_wide.loc[treat_mask, 'wgt'].values
    target_means = np.average(X_treat, weights=treat_wgt, axis=0)

    # Entropy balancing: find weights for control group to match target means
    def entropy_loss(lam, X, w_base, target):
        exp_term = np.exp(X @ lam)
        w_new = w_base * exp_term
        w_norm = w_new / w_new.sum()
        moments = X.T @ w_norm
        return np.sum((moments - target) ** 2)

    lam0 = np.zeros(len(match_vars_1))
    res = scipy_minimize(entropy_loss, lam0, args=(X_control, w_base, target_means), method='BFGS')

    exp_term = np.exp(X_control @ res.x)
    ebw_weights_control = w_base * exp_term
    ebw_weights_control = ebw_weights_control / ebw_weights_control.sum() * len(ebw_weights_control)

    # Create weight mapping: cityno -> ebw weight
    city_weights = {}
    control_cities = ebw_wide.loc[control_mask, 'cityno'].values
    for c, w in zip(control_cities, ebw_weights_control):
        city_weights[c] = w
    treat_cities = ebw_wide.loc[treat_mask, 'cityno'].values
    for c in treat_cities:
        city_weights[c] = 1.0  # treatment gets uniform weight

    # Merge weights back to individual-level data
    # Use the balanced panel cities and non-movers
    ebw_indiv = ebw_data[ebw_data['cityno'].isin(balanced_cities)].copy()
    ebw_indiv['ebw1'] = ebw_indiv['cityno'].map(city_weights)
    ebw_indiv['ebw1'] = ebw_indiv['ebw1'].fillna(1.0)

    fml = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
    r = run_spec(
        spec_id="rc/weights/main/ebalance_weighted",
        spec_run_id="138922-V1__rc_weights_main_ebalance_weighted",
        spec_tree_path="specification_tree/modules/robustness/weights.md#a-main-weight-choices",
        outcome_var="sportsclub",
        treatment_var="treat",
        data=ebw_indiv,
        formula=fml,
        vcov={"CRV1": "cityno_fe"},
        sample_desc="3 states, non-movers, balanced city panel, entropy-balanced weights",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc="none",
        cluster_var="cityno",
        weights_col="ebw1",
        extra_payload={"weights": {"spec_id": "rc/weights/main/ebalance_weighted", "family": "ebalance",
                                    "match_vars": match_vars_1, "weight_type": "entropy_balance"}},
    )
    print(f"  rc/weights/main/ebalance_weighted: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

    # Joint: ebalance + controls
    fml_joint = make_formula("sportsclub", "treat", controls=ALL_CONTROLS, fe=BASELINE_FE3)
    r = run_spec(
        spec_id="rc/joint/ebalance_plus_controls",
        spec_run_id="138922-V1__rc_joint_ebalance_plus_controls",
        spec_tree_path="specification_tree/modules/robustness/joint.md#purpose",
        outcome_var="sportsclub",
        treatment_var="treat",
        data=ebw_indiv,
        formula=fml_joint,
        vcov={"CRV1": "cityno_fe"},
        sample_desc="3 states, non-movers, balanced city panel, entropy-balanced weights, full controls",
        fixed_effects_desc="year_3rd + bula_3rd + cityno",
        controls_desc="full (all 9): " + ", ".join(ALL_CONTROLS),
        cluster_var="cityno",
        weights_col="ebw1",
        extra_payload={"joint": {"spec_id": "rc/joint/ebalance_plus_controls",
                                  "components": ["rc/weights/main/ebalance_weighted", "rc/controls/sets/full"]},
                       "weights": {"weight_type": "entropy_balance"},
                       "controls": {"n_controls": 9, "controls": ALL_CONTROLS}},
    )
    print(f"  rc/joint/ebalance_plus_controls: coef={r.get('coefficient', 'FAIL'):.4f}")

    ebw_success = True
except Exception as e:
    ebw_success = False
    tb_str = traceback.format_exc()
    error_msg = str(e).replace("\n", " ")[:240]
    error_details = {
        "stage": "ebalance_weight_construction",
        "exception_type": type(e).__name__,
        "exception_message": str(e)[:500],
        "traceback_tail": "\n".join(tb_str.splitlines()[-10:]),
    }
    # Record failures for both ebalance specs
    for sid, srid in [("rc/weights/main/ebalance_weighted", "138922-V1__rc_weights_main_ebalance_weighted"),
                       ("rc/joint/ebalance_plus_controls", "138922-V1__rc_joint_ebalance_plus_controls")]:
        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": srid,
            "spec_id": sid,
            "spec_tree_path": "specification_tree/modules/robustness/weights.md#a-main-weight-choices" if "weights" in sid else "specification_tree/modules/robustness/joint.md#purpose",
            "baseline_group_id": "G1",
            "outcome_var": "sportsclub",
            "treatment_var": "treat",
            "sample_desc": "3 states, non-movers, balanced city panel",
            "fixed_effects": "year_3rd + bula_3rd + cityno",
            "controls_desc": "none" if "weights" in sid else "full (all 9)",
            "cluster_var": "cityno",
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": error_msg, "error_details": error_details}),
            "run_success": 0, "run_error": error_msg,
        })
    print(f"  FAILED to construct ebalance weights: {error_msg[:100]}")

# rc/weights/main/survey_weighted: use weight2 with ins_register==1
mask = (df['bula_3rd'].isin([4, 13, 16]) & (df['target'] == 1) & (df['nonmiss'] == 1) &
        df['year_3rd'].between(2006, 2010) & (df['ins_register'] == 1))
sub = df[mask].copy()
fml = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
r = run_spec(
    spec_id="rc/weights/main/survey_weighted",
    spec_run_id="138922-V1__rc_weights_main_survey_weighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#a-main-weight-choices",
    outcome_var="sportsclub",
    treatment_var="treat",
    data=sub,
    formula=fml,
    vcov={"CRV1": "cityno_fe"},
    sample_desc="3 states, target==1, nonmiss==1, ins_register==1, cohorts 2006-2010, survey weighted",
    fixed_effects_desc="year_3rd + bula_3rd + cityno",
    controls_desc="none",
    cluster_var="cityno",
    weights_col="weight2",
    extra_payload={"weights": {"spec_id": "rc/weights/main/survey_weighted", "family": "survey_weights",
                                "weight_var": "weight2", "sample_restriction": "ins_register==1"}},
)
print(f"  rc/weights/main/survey_weighted: coef={r.get('coefficient', 'FAIL'):.4f}, N={r.get('n_obs', 'FAIL')}")

print(f"\nTotal specs after RC: {len(results)}")

# ============================================================
# STEP 3: INFERENCE VARIANTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Running inference variants on baseline")
print("=" * 60)

baseline_formula = make_formula("sportsclub", "treat", fe=BASELINE_FE3)
infer_counter = 0

for variant in bg["inference_plan"]["variants"]:
    infer_counter += 1
    infer_spec_id = variant["spec_id"]
    params = variant["params"]

    if infer_spec_id == "infer/se/hc/hc1":
        vcov_new = "hetero"
        cluster_new = None
        tree_path = "specification_tree/modules/inference/standard_errors.md#hc"
    elif infer_spec_id == "infer/se/cluster/bula_3rd":
        vcov_new = {"CRV1": "bula_3rd_fe"}
        cluster_new = "bula_3rd"
        tree_path = "specification_tree/modules/inference/standard_errors.md#cluster"
    elif infer_spec_id == "infer/se/cluster/cohort":
        vcov_new = {"CRV1": "cohort"}
        cluster_new = "cohort"
        tree_path = "specification_tree/modules/inference/standard_errors.md#cluster"
    elif infer_spec_id == "infer/se/cluster/twoway_cityno_year":
        vcov_new = {"CRV1": "cityno_fe + year_3rd_fe"}
        cluster_new = ["cityno", "year_3rd"]
        tree_path = "specification_tree/modules/inference/standard_errors.md#twoway"
    elif "wildboot" in infer_spec_id:
        # Wild bootstrap - record as attempted but may fail since wildboottest not installed
        vcov_new = None
        cluster_new = params.get("cluster_var", "cohort")
        tree_path = "specification_tree/modules/inference/resampling.md#wildboot"
    else:
        vcov_new = "hetero"
        cluster_new = None
        tree_path = "specification_tree/modules/inference/standard_errors.md"

    infer_run_id = f"138922-V1__infer_{infer_counter:03d}"

    if "wildboot" in infer_spec_id:
        # Try wildboottest
        try:
            from wildboottest.wildboottest import wildboottest

            model = pf.feols(baseline_formula, data=sample_base, vcov={"CRV1": "cohort"})

            weight_type = params.get("weight_type", "rademacher")
            reps = params.get("reps", 999)

            boot_result = wildboottest(
                model, param="treat", cluster=sample_base["cohort"],
                B=reps, weights_type=weight_type, seed=138922
            )

            coef_val = float(model.coef().get("treat", np.nan))
            pval = float(boot_result.pvalue)
            se_val = float(model.se().get("treat", np.nan))  # original SE

            payload = {
                "coefficients": {k: float(v) for k, v in model.coef().items()},
                "inference": {"spec_id": infer_spec_id, "params": params},
                "software": software_block,
                "surface_hash": surface_hash_val,
                "design": design_block,
            }

            inference_results.append({
                "paper_id": PAPER_ID,
                "inference_run_id": infer_run_id,
                "spec_run_id": "138922-V1__baseline",
                "spec_id": infer_spec_id,
                "spec_tree_path": tree_path,
                "baseline_group_id": "G1",
                "outcome_var": "sportsclub",
                "treatment_var": "treat",
                "cluster_var": str(params.get("cluster_var", "cohort")),
                "coefficient": coef_val,
                "std_error": se_val,
                "p_value": pval,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_obs": int(model._N),
                "r_squared": float(model._r2),
                "coefficient_vector_json": json.dumps(payload),
                "run_success": 1,
                "run_error": "",
            })
            print(f"  {infer_spec_id}: p={pval:.4f}")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = str(e).replace("\n", " ")[:240]
            error_details = {
                "stage": "wild_bootstrap_inference",
                "exception_type": type(e).__name__,
                "exception_message": str(e)[:500],
                "traceback_tail": "\n".join(tb_str.splitlines()[-10:]),
            }
            inference_results.append({
                "paper_id": PAPER_ID,
                "inference_run_id": infer_run_id,
                "spec_run_id": "138922-V1__baseline",
                "spec_id": infer_spec_id,
                "spec_tree_path": tree_path,
                "baseline_group_id": "G1",
                "outcome_var": "sportsclub",
                "treatment_var": "treat",
                "cluster_var": str(params.get("cluster_var", "cohort")),
                "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
                "coefficient_vector_json": json.dumps({"error": error_msg, "error_details": error_details}),
                "run_success": 0,
                "run_error": error_msg,
            })
            print(f"  {infer_spec_id}: FAILED - {error_msg[:80]}")
    elif infer_spec_id == "infer/se/cluster/twoway_cityno_year":
        # Two-way clustering: pyfixest may support this via "cityno_fe + year_3rd_fe" syntax
        # but let's use the proper twoway syntax
        try:
            # pyfixest twoway clustering
            model = pf.feols(baseline_formula, data=sample_base, vcov={"CRV1": "cityno_fe+year_3rd_fe"})
            coef_val = float(model.coef().get("treat", np.nan))
            se_val = float(model.se().get("treat", np.nan))
            pval = float(model.pvalue().get("treat", np.nan))
            ci = model.confint()
            ci_lower = float(ci.loc["treat"].iloc[0]) if "treat" in ci.index else np.nan
            ci_upper = float(ci.loc["treat"].iloc[1]) if "treat" in ci.index else np.nan

            payload = {
                "coefficients": {k: float(v) for k, v in model.coef().items()},
                "inference": {"spec_id": infer_spec_id, "params": params},
                "software": software_block,
                "surface_hash": surface_hash_val,
                "design": design_block,
            }

            inference_results.append({
                "paper_id": PAPER_ID,
                "inference_run_id": infer_run_id,
                "spec_run_id": "138922-V1__baseline",
                "spec_id": infer_spec_id,
                "spec_tree_path": tree_path,
                "baseline_group_id": "G1",
                "outcome_var": "sportsclub",
                "treatment_var": "treat",
                "cluster_var": "cityno, year_3rd",
                "coefficient": coef_val,
                "std_error": se_val,
                "p_value": pval,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_obs": int(model._N),
                "r_squared": float(model._r2),
                "coefficient_vector_json": json.dumps(payload),
                "run_success": 1,
                "run_error": "",
            })
            print(f"  {infer_spec_id}: se={se_val:.4f}, p={pval:.4f}")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = str(e).replace("\n", " ")[:240]
            error_details = {
                "stage": "twoway_cluster_inference",
                "exception_type": type(e).__name__,
                "exception_message": str(e)[:500],
                "traceback_tail": "\n".join(tb_str.splitlines()[-10:]),
            }
            inference_results.append({
                "paper_id": PAPER_ID,
                "inference_run_id": infer_run_id,
                "spec_run_id": "138922-V1__baseline",
                "spec_id": infer_spec_id,
                "spec_tree_path": tree_path,
                "baseline_group_id": "G1",
                "outcome_var": "sportsclub",
                "treatment_var": "treat",
                "cluster_var": "cityno, year_3rd",
                "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
                "coefficient_vector_json": json.dumps({"error": error_msg, "error_details": error_details}),
                "run_success": 0,
                "run_error": error_msg,
            })
            print(f"  {infer_spec_id}: FAILED - {error_msg[:80]}")
    else:
        r = run_inference_variant(
            base_row=results[0],  # baseline row
            infer_spec_id=infer_spec_id,
            infer_tree_path=tree_path,
            data=sample_base,
            formula=baseline_formula,
            vcov_new=vcov_new,
            cluster_var_new=cluster_new,
            infer_run_id=infer_run_id,
        )
        print(f"  {infer_spec_id}: se={r.get('std_error', 'FAIL')}, p={r.get('p_value', 'FAIL')}")


# ============================================================
# STEP 4: WRITE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Writing outputs")
print("=" * 60)

# 4.1 specification_results.csv
spec_df = pd.DataFrame(results)
spec_cols = ["paper_id", "spec_run_id", "spec_id", "spec_tree_path", "baseline_group_id",
             "outcome_var", "treatment_var", "coefficient", "std_error", "p_value",
             "ci_lower", "ci_upper", "n_obs", "r_squared", "coefficient_vector_json",
             "sample_desc", "fixed_effects", "controls_desc", "cluster_var",
             "run_success", "run_error"]
for c in spec_cols:
    if c not in spec_df.columns:
        spec_df[c] = ""
spec_df = spec_df[spec_cols]
spec_df.to_csv(f"{PACKAGE_DIR}/specification_results.csv", index=False)
print(f"  Wrote specification_results.csv: {len(spec_df)} rows")

# 4.2 inference_results.csv
if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_cols = ["paper_id", "inference_run_id", "spec_run_id", "spec_id", "spec_tree_path",
                  "baseline_group_id", "outcome_var", "treatment_var",
                  "coefficient", "std_error", "p_value",
                  "ci_lower", "ci_upper", "n_obs", "r_squared", "coefficient_vector_json",
                  "cluster_var", "run_success", "run_error"]
    for c in infer_cols:
        if c not in infer_df.columns:
            infer_df[c] = ""
    infer_df = infer_df[infer_cols]
    infer_df.to_csv(f"{PACKAGE_DIR}/inference_results.csv", index=False)
    print(f"  Wrote inference_results.csv: {len(infer_df)} rows")

# Summary stats
n_success = spec_df[spec_df['run_success'] == 1].shape[0]
n_fail = spec_df[spec_df['run_success'] == 0].shape[0]
n_infer_success = sum(1 for r in inference_results if r['run_success'] == 1) if inference_results else 0
n_infer_fail = sum(1 for r in inference_results if r['run_success'] == 0) if inference_results else 0

print(f"\n  SUMMARY:")
print(f"    Specification results: {n_success} succeeded, {n_fail} failed, {len(spec_df)} total")
print(f"    Inference results: {n_infer_success} succeeded, {n_infer_fail} failed, {len(inference_results)} total")

# Baseline coefficient
if results and results[0]['run_success'] == 1:
    print(f"\n  PRIMARY BASELINE (sportsclub):")
    print(f"    coefficient = {results[0]['coefficient']:.4f}")
    print(f"    std_error   = {results[0]['std_error']:.4f}")
    print(f"    p_value     = {results[0]['p_value']:.4f}")
    print(f"    n_obs       = {results[0]['n_obs']}")

print("\nDone.")
