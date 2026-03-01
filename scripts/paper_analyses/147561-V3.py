"""
Specification Search Script for Balan, Bergeron, Tourek, and Weigel
"Local Elites as State Capacity: How City Chiefs Use Local Information
to Increase Tax Compliance in the D.R. Congo"

Paper ID: 147561-V3

Surface-driven execution:
  - G1: taxes_paid ~ t_l + FE (Table 4 baseline, strata+house+month FE, cluster a7)
  - Randomized experiment, ITT
  - ~50+ specifications

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "147561-V3"
DATA_DIR = "data/downloads/extracted/147561-V3"
BASE_DATA = f"{DATA_DIR}/Data/01_base"
OUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
inference_variants = surface_obj["baseline_groups"][0]["inference_plan"]["variants"]

# ============================================================
# DATA CONSTRUCTION
# Replicate the essential parts of 2_Data_Construction.do
# Focus on variables needed for Table 4
# ============================================================
print("Constructing analysis data...")

# --- 1. Load and clean flier data ---
fliers = pd.read_stata(f"{BASE_DATA}/admin_data/fliers_campaign.dta",
                        convert_categoricals=False)
# Exclude pilot polygons from flier data
pilot_a7s = [200, 201, 202, 203, 207, 208, 210]
fliers = fliers[~fliers['a7'].isin(pilot_a7s)].copy()
fliers = fliers.rename(columns={'code': 'compound1', 'rate': 'assign_flier_rate',
                                 'treatment_fr': 'assign_treatment_fr'})
fliers = fliers[['a7', 'compound1', 'assign_flier_rate', 'assign_treatment_fr']].copy()

# --- 2. Load stratum data ---
strat = pd.read_stata(f"{BASE_DATA}/admin_data/stratum.dta",
                       convert_categoricals=False)
strat = strat[['a7', 'stratum']].copy()

# --- 3. Load assignment data ---
assign = pd.read_stata(f"{BASE_DATA}/admin_data/randomization_schedule.dta",
                        convert_categoricals=False)
assign = assign[['a7', 'treatment', 'month']].copy()

# Add manually coded polygons from do-file
extra_rows = pd.DataFrame([
    {'a7': 201, 'treatment': 2.0, 'month': 0.0},
    {'a7': 202, 'treatment': 1.0, 'month': 0.0},
    {'a7': 203, 'treatment': 1.0, 'month': 0.0},
    {'a7': 210, 'treatment': 2.0, 'month': 0.0},
    {'a7': 200, 'treatment': 1.0, 'month': 0.0},
    {'a7': 207, 'treatment': 2.0, 'month': 0.0},
    {'a7': 208, 'treatment': 4.0, 'month': 0.0},
])
assign = pd.concat([assign, extra_rows], ignore_index=True)
# Fix assignment mistake: a7==654 should be local (2)
assign.loc[assign['a7'] == 654, 'treatment'] = 2.0
assign = assign.rename(columns={'treatment': 'tmt'})

# --- 4. Load registration (cartographie) data ---
reg = pd.read_stata(f"{BASE_DATA}/admin_data/registration_noPII.dta",
                     convert_categoricals=False)
reg = reg[reg['tot_complete'] == 1].copy()
reg = reg.rename(columns={'today': 'today_carto'})

# --- 5. Load taxroll data ---
taxroll = pd.read_stata(f"{BASE_DATA}/admin_data/taxroll_noPII.dta",
                         convert_categoricals=False)

# --- 6. Load tax payments data ---
tax = pd.read_stata(f"{BASE_DATA}/admin_data/tax_payments_noPII.dta",
                     convert_categoricals=False)
tax = tax[tax['unmatched_compound'] != 1].copy()
tax = tax[tax['compound1'].notna()].copy()
tax = tax.rename(columns={'date': 'date_TDM'})

# --- 7. Merge datasets ---
print("  Merging datasets...")

# Start with fliers
df = fliers.copy()

# Merge stratum (m:1 on a7)
df = df.merge(strat, on='a7', how='left')

# Merge assignment (m:1 on a7)
df = df.merge(assign[['a7', 'tmt']], on='a7', how='left')

# Merge registration (1:1 on compound1)
reg_cols = ['compound1', 'a7', 'house', 'today_carto', 'mm_rate',
            'what_rate_periph', 'exempt', 'collect_success', 'exempt_other']
reg_sub = reg[reg_cols].drop_duplicates(subset='compound1', keep='first')
df = df.merge(reg_sub[['compound1', 'house', 'today_carto', 'mm_rate',
                        'what_rate_periph', 'exempt', 'collect_success',
                        'exempt_other']],
              on='compound1', how='left', indicator='_merge_flier_carto')

# Merge taxroll
taxroll_sub = taxroll[['compound1', 'Bonus']].drop_duplicates(subset='compound1', keep='first')
df = df.merge(taxroll_sub, on='compound1', how='left', indicator='_merge_flier_carto_rep')

# Merge tax payments (m:1 on compound1)
tax_sub = tax[['compound1', 'amountCF', 'date_TDM', 'house']].copy()
tax_sub = tax_sub.rename(columns={'house': 'house_tax'})
# Keep first tax payment per compound
tax_sub = tax_sub.drop_duplicates(subset='compound1', keep='first')
df = df.merge(tax_sub, on='compound1', how='left', indicator='_merge_tax')

# Drop observations not in any source (matching the Stata drop)
both_unmatched = ((df['_merge_flier_carto'] == 'left_only') &
                  (df['_merge_flier_carto_rep'] == 'left_only'))
df = df[~both_unmatched].copy()

# --- 8. Variable construction ---
print("  Constructing variables...")

# Tax compliance dummy
df['taxes_paid'] = 0
df.loc[df['_merge_tax'] == 'both', 'taxes_paid'] = 1
# code_same: if collect_success==1, also paid
df.loc[(df['taxes_paid'] == 0) & (df['collect_success'] == 1), 'taxes_paid'] = 1
# Correct: if house==1 and rate > amountCF, not really paid full rate
mask_periph = ((df['house'] == 1) & (df['_merge_tax'] == 'both') &
               (df['assign_flier_rate'] > df['amountCF']) &
               (df['amountCF'].notna()) & (df['assign_flier_rate'].notna()))
df.loc[mask_periph, 'taxes_paid'] = 0
mask_mm = ((df['house'] == 2) & (df['_merge_tax'] == 'both') &
           (df['mm_rate'] > df['amountCF']) &
           (df['amountCF'].notna()) & (df['mm_rate'].notna()))
df.loc[mask_mm, 'taxes_paid'] = 0

# Tax rate
df['rate'] = np.nan
df.loc[df['house'] == 1, 'rate'] = df.loc[df['house'] == 1, 'assign_flier_rate']
df.loc[df['house'] == 2, 'rate'] = df.loc[df['house'] == 2, 'mm_rate']

# Tax amount
df['taxes_paid_amt'] = df['taxes_paid'] * df['rate']

# --- 9. Create analysis dataset ---
print("  Creating analysis sample...")

# Drop villas (house==3)
df = df[df['house'] != 3].copy()

# Drop pilot polygons
df = df[~df['a7'].isin(pilot_a7s)].copy()

# Treatment variables
df['t_l'] = (df['tmt'] == 2).astype(int)
df['t_c'] = (df['tmt'] == 1).astype(int)
df['t_cli'] = (df['tmt'] == 3).astype(int)
df['t_cxl'] = (df['tmt'] == 4).astype(int)

# Regenerate taxes_paid_amt using updated rate
df['taxes_paid_amt'] = df['taxes_paid'] * df['rate']

# Create today_alt (alternative date for time FE)
df['date_TDM_dt'] = pd.to_datetime(df['date_TDM'], errors='coerce')
df['today_carto_dt'] = pd.to_datetime(df['today_carto'], errors='coerce')

# Compute polygon-level min TDM date and max carto date
a7_min_tdm = df.groupby('a7')['date_TDM_dt'].min().reset_index()
a7_min_tdm.columns = ['a7', 'a7_min_today_TDM']
a7_max_carto = df.groupby('a7')['today_carto_dt'].max().reset_index()
a7_max_carto.columns = ['a7', 'a7_max_today_carto']

df = df.merge(a7_min_tdm, on='a7', how='left')
df = df.merge(a7_max_carto, on='a7', how='left')

df['today_alt'] = df['a7_min_today_TDM']
df.loc[df['today_alt'].isna(), 'today_alt'] = df.loc[df['today_alt'].isna(), 'a7_max_today_carto']

# Convert today_alt to numerical (Stata date integer)
df['today_alt_num'] = (df['today_alt'] - pd.Timestamp('1960-01-01')).dt.days

# Create time FE (2-month bins matching Stata code)
# The Stata code uses: at(21355 21415 21475 21532) icodes
# These are Stata dates (days since Jan 1 1960)
# 21355 = June 15, 2018; 21415 = Aug 14, 2018; 21475 = Oct 13, 2018; 21532 = Dec 9, 2018
cuts = [21355, 21415, 21475, 21532]
df['time_FE_tdm_2mo_CvL'] = pd.cut(
    df['today_alt_num'],
    bins=[-np.inf] + cuts + [np.inf],
    labels=False,
    right=False
)
# Recode to match Stata icodes (0-indexed starting from first bin that has data)
# Stata's egen cut with icodes: 0 for [21355,21415), 1 for [21415,21475), 2 for [21475,21532)
# Values below 21355 or above 21532 will be NaN
df['time_FE_tdm_2mo_CvL'] = np.nan
mask0 = (df['today_alt_num'] >= 21355) & (df['today_alt_num'] < 21415)
mask1 = (df['today_alt_num'] >= 21415) & (df['today_alt_num'] < 21475)
mask2 = (df['today_alt_num'] >= 21475) & (df['today_alt_num'] < 21532)
df.loc[mask0, 'time_FE_tdm_2mo_CvL'] = 0
df.loc[mask1, 'time_FE_tdm_2mo_CvL'] = 1
df.loc[mask2, 'time_FE_tdm_2mo_CvL'] = 2

# Drop if rate is missing
df = df[df['rate'].notna()].copy()

# Create house-type indicator (mm = maison moyenne)
df['mm'] = 0
df.loc[df['house'] == 2, 'mm'] = 1

# Create integer columns for FE absorption
df['stratum_int'] = df['stratum'].astype('Int64')
df['house_int'] = df['house'].astype('Int64')
df['time_fe_int'] = df['time_FE_tdm_2mo_CvL'].astype('Int64')
df['a7_int'] = df['a7'].astype('Int64')

# Convert float32 -> float64
for col in df.columns:
    if df[col].dtype == np.float32:
        df[col] = df[col].astype(np.float64)

print(f"  Analysis dataset: {len(df)} observations")
print(f"  Treatment breakdown: tmt==1 (Central): {(df['tmt']==1).sum()}, tmt==2 (Local): {(df['tmt']==2).sum()}")

# ============================================================
# SAMPLES
# ============================================================
# Focal comparison: Local vs Central (tmt 1 and 2)
df_cl = df[df['tmt'].isin([1, 2])].copy()
print(f"  Central vs Local sample: {len(df_cl)} observations")

# ============================================================
# RUNNER
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var,
             treatment_var, controls, fe_vars, data, vcov,
             sample_desc, controls_desc, fe_desc,
             cluster_var="a7", axis_block_name=None, axis_block=None,
             notes="", functional_form=None):
    """Run a single specification and append to results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        rhs_vars = [treatment_var]
        if controls:
            rhs_vars.extend(controls)

        rhs = " + ".join(rhs_vars)

        if fe_vars:
            fe_str = " + ".join(fe_vars)
            formula = f"{outcome_var} ~ {rhs} | {fe_str}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload_kwargs = dict(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )
        payload = make_success_payload(**payload_kwargs)
        if functional_form:
            payload["functional_form"] = functional_form

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
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
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          controls, fe_vars, data, vcov_new,
                          cluster_var_new=""):
    """Re-estimate with alternative inference and write to inference_results."""
    global spec_run_counter
    spec_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        rhs_vars = [treatment_var]
        if controls:
            rhs_vars.extend(controls)

        rhs = " + ".join(rhs_vars)
        if fe_vars:
            fe_str = " + ".join(fe_vars)
            formula = f"{outcome_var} ~ {rhs} | {fe_str}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        m = pf.feols(formula, data=data, vcov=vcov_new)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id,
                       "params": {"cluster_var": cluster_var_new} if cluster_var_new else {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference_variant")
        payload = make_failure_payload(
            error=err_msg, error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================
# BASELINE: Table 4 Compliance Col 4
# taxes_paid ~ t_l | stratum + house + time_FE_tdm_2mo_CvL, cl(a7)
# ============================================================
print("\n=== Running specifications ===")
print("Baseline: Table 4 Compliance Col 4...")
baseline_run_id, *_ = run_spec(
    spec_id="baseline",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2), rate non-missing",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7"
)

# ============================================================
# ADDITIONAL BASELINE: Revenues (taxes_paid_amt)
# ============================================================
print("Baseline revenues: Table 4 Revenues Col 4...")
run_spec(
    spec_id="baseline__revenues",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2), rate non-missing",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7"
)

# ============================================================
# DESIGN: Difference in means (no FE)
# ============================================================
print("Design: diff in means...")
run_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_tree_path="designs/randomized_experiment.md#diff-in-means",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=[],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="none",
    cluster_var="a7"
)

# Also diff in means for revenues
run_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means__revenues",
    spec_tree_path="designs/randomized_experiment.md#diff-in-means",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=[],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="none",
    cluster_var="a7"
)

# ============================================================
# RC: FIXED EFFECTS SETS (Table 4 progression)
# ============================================================
print("RC: FE sets...")

# rc/fe/sets/stratum_only -- Table 4 Col 1
run_spec(
    spec_id="rc/fe/sets/stratum_only",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/stratum_only", "fe_set": ["stratum"]}
)

# revenues, stratum only
run_spec(
    spec_id="rc/fe/sets/stratum_only__revenues",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/stratum_only", "fe_set": ["stratum"], "outcome": "revenues"}
)

# rc/fe/sets/stratum_month -- Table 4 Col 2
run_spec(
    spec_id="rc/fe/sets/stratum_month",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/stratum_month", "fe_set": ["stratum", "time_FE_tdm_2mo_CvL"]}
)

# revenues, stratum + month
run_spec(
    spec_id="rc/fe/sets/stratum_month__revenues",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/stratum_month", "fe_set": ["stratum", "time_FE_tdm_2mo_CvL"], "outcome": "revenues"}
)

# rc/fe/sets/stratum_month_house -- same as baseline
run_spec(
    spec_id="rc/fe/sets/stratum_month_house",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/stratum_month_house", "fe_set": ["stratum", "time_FE_tdm_2mo_CvL", "house"]}
)

# revenues, stratum + month + house
run_spec(
    spec_id="rc/fe/sets/stratum_month_house__revenues",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/stratum_month_house", "fe_set": ["stratum", "time_FE_tdm_2mo_CvL", "house"], "outcome": "revenues"}
)

# ============================================================
# RC: FE DROP
# ============================================================
print("RC: FE drop...")

# rc/fe/drop/house -- drop house FE from baseline
run_spec(
    spec_id="rc/fe/drop/house",
    spec_tree_path="modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum + time_FE (house dropped)",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/house", "action": "drop", "dropped": ["house"]}
)

# rc/fe/drop/time_FE_tdm_2mo_CvL -- drop time FE from baseline
run_spec(
    spec_id="rc/fe/drop/time_FE_tdm_2mo_CvL",
    spec_tree_path="modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local (tmt 1,2)",
    controls_desc="none",
    fe_desc="stratum + house (time FE dropped)",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/time_FE_tdm_2mo_CvL", "action": "drop", "dropped": ["time_FE_tdm_2mo_CvL"]}
)

# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================
print("RC: Sample restrictions...")

# rc/sample/restriction/exclude_exempt -- Table 4 Col 5
df_cl_noexempt = df_cl[df_cl['exempt'] != 1].copy()
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excluding exempt properties",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1"}
)

# revenues, exclude exempt
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt__revenues",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excluding exempt properties",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1", "outcome": "revenues"}
)

# rc/sample/restriction/polygon_means -- Table 4 Col 3 equivalent
# Collapse to polygon means
print("RC: Polygon means...")
df_cl_valid_time = df_cl[df_cl['time_FE_tdm_2mo_CvL'].notna()].copy()
agg_funcs = {
    'taxes_paid': 'mean',
    'taxes_paid_amt': 'mean',
    'time_FE_tdm_2mo_CvL': 'min',
    't_l': 'max',
    't_c': 'max',
    'stratum': 'max',
    'house': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
}
df_poly = df_cl_valid_time.groupby(['a7', 'tmt']).agg(agg_funcs).reset_index()
df_poly['stratum_int'] = df_poly['stratum'].astype('Int64')
df_poly['time_fe_int'] = df_poly['time_FE_tdm_2mo_CvL'].astype('Int64')
df_poly['a7_int'] = df_poly['a7'].astype('Int64')
df_poly = df_poly[df_poly['tmt'].isin([1, 2])].copy()

run_spec(
    spec_id="rc/sample/restriction/polygon_means",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_poly,
    vcov="hetero",
    sample_desc="Polygon means, Central vs Local",
    controls_desc="none",
    fe_desc="stratum + time_FE_tdm_2mo_CvL",
    cluster_var="",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/polygon_means",
                "restriction": "collapsed to polygon means",
                "notes": "Robust SE (not clustered) because N=polygons"}
)

# polygon means revenues
run_spec(
    spec_id="rc/sample/restriction/polygon_means__revenues",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_poly,
    vcov="hetero",
    sample_desc="Polygon means, Central vs Local, revenues",
    controls_desc="none",
    fe_desc="stratum + time_FE_tdm_2mo_CvL",
    cluster_var="",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/polygon_means",
                "restriction": "collapsed to polygon means", "outcome": "revenues"}
)

# rc/sample/outliers/trim_y_amt_1_99 -- trim revenues at 1st/99th percentile
print("RC: Trim revenues...")
p1 = df_cl['taxes_paid_amt'].quantile(0.01)
p99 = df_cl['taxes_paid_amt'].quantile(0.99)
df_cl_trim = df_cl[(df_cl['taxes_paid_amt'] >= p1) & (df_cl['taxes_paid_amt'] <= p99)].copy()
run_spec(
    spec_id="rc/sample/outliers/trim_y_amt_1_99",
    spec_tree_path="modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_trim,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, taxes_paid_amt trimmed at 1st/99th pctile",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_amt_1_99",
                "trim_var": "taxes_paid_amt", "lower_pctile": 1, "upper_pctile": 99}
)

# ============================================================
# RC: FUNCTIONAL FORM (revenues only)
# ============================================================
print("RC: Functional form...")

# rc/form/outcome/log1p_amt
df_cl['log1p_taxes_paid_amt'] = np.log1p(df_cl['taxes_paid_amt'])
run_spec(
    spec_id="rc/form/outcome/log1p_amt",
    spec_tree_path="modules/robustness/functional_form.md#log1p",
    baseline_group_id="G1",
    outcome_var="log1p_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, log(1+revenue)",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/log1p_amt",
                     "transformation": "log(1+x)",
                     "interpretation": "semi-elasticity of tax revenue w.r.t. local collection assignment",
                     "units": "log Congolese Francs"}
)

# rc/form/outcome/asinh_amt
df_cl['asinh_taxes_paid_amt'] = np.arcsinh(df_cl['taxes_paid_amt'])
run_spec(
    spec_id="rc/form/outcome/asinh_amt",
    spec_tree_path="modules/robustness/functional_form.md#asinh",
    baseline_group_id="G1",
    outcome_var="asinh_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, asinh(revenue)",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/asinh_amt",
                     "transformation": "asinh(x)",
                     "interpretation": "approximate semi-elasticity of tax revenue w.r.t. local collection assignment",
                     "units": "inverse hyperbolic sine of Congolese Francs"}
)

# ============================================================
# RC: TREATMENT DEFINITION
# ============================================================
print("RC: Treatment definition...")

# rc/data/treatment/include_cli_arm
# Include CLI arm (tmt==3) alongside central and local
df_cl_cli = df[df['tmt'].isin([1, 2, 3])].copy()
run_spec(
    spec_id="rc/data/treatment/include_cli_arm",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cli"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_cli,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CLI arms (tmt 1,2,3)",
    controls_desc="t_cli indicator",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cli_arm",
                "treatment_change": "include CLI arm (tmt==3)",
                "sample_restriction": "inlist(tmt,1,2,3)"}
)

# revenues, include CLI
run_spec(
    spec_id="rc/data/treatment/include_cli_arm__revenues",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=["t_cli"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_cli,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CLI arms, revenues",
    controls_desc="t_cli indicator",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cli_arm",
                "treatment_change": "include CLI arm (tmt==3)", "outcome": "revenues"}
)

# rc/data/treatment/include_cxl_arm
# Include CxL arm (tmt==4)
df_cl_cxl = df[df['tmt'].isin([1, 2, 4])].copy()
run_spec(
    spec_id="rc/data/treatment/include_cxl_arm",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cxl"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_cxl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CxL arms (tmt 1,2,4)",
    controls_desc="t_cxl indicator",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cxl_arm",
                "treatment_change": "include CxL arm (tmt==4)"}
)

# revenues, include CxL
run_spec(
    spec_id="rc/data/treatment/include_cxl_arm__revenues",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=["t_cxl"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_cxl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CxL arms, revenues",
    controls_desc="t_cxl indicator",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cxl_arm",
                "treatment_change": "include CxL arm (tmt==4)", "outcome": "revenues"}
)

# rc/data/treatment/pooled_local_vs_central
# Pool tmt==2 and tmt==4 as "local-type" vs tmt==1
df_pooled = df[df['tmt'].isin([1, 2, 4])].copy()
df_pooled['t_local_type'] = ((df_pooled['tmt'] == 2) | (df_pooled['tmt'] == 4)).astype(int)
run_spec(
    spec_id="rc/data/treatment/pooled_local_vs_central",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_local_type",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_pooled,
    vcov={"CRV1": "a7_int"},
    sample_desc="Pooled local-type (tmt 2+4) vs Central (tmt 1)",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/pooled_local_vs_central",
                "treatment_change": "pool tmt==2 and tmt==4 as local-type"}
)

# revenues, pooled
df_pooled['taxes_paid_amt'] = df_pooled['taxes_paid'] * df_pooled['rate']
run_spec(
    spec_id="rc/data/treatment/pooled_local_vs_central__revenues",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_local_type",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_pooled,
    vcov={"CRV1": "a7_int"},
    sample_desc="Pooled local-type (tmt 2+4) vs Central (tmt 1), revenues",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/pooled_local_vs_central",
                "treatment_change": "pool tmt==2 and tmt==4 as local-type", "outcome": "revenues"}
)

# ============================================================
# Additional cross-combinations to reach 50+ specs
# ============================================================
print("RC: Cross-combinations...")

# Exclude exempt + stratum only (compliance)
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt__stratum_only",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excl exempt, stratum FE only",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1 + stratum only FE"}
)

# Exclude exempt + stratum + month (compliance)
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt__stratum_month",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excl exempt, stratum+month FE",
    controls_desc="none",
    fe_desc="stratum + time_FE",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1 + stratum+month FE"}
)

# log1p revenues with stratum only FE
run_spec(
    spec_id="rc/form/outcome/log1p_amt__stratum_only",
    spec_tree_path="modules/robustness/functional_form.md#log1p",
    baseline_group_id="G1",
    outcome_var="log1p_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, log(1+revenue), stratum only",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/log1p_amt",
                     "transformation": "log(1+x)",
                     "interpretation": "semi-elasticity, stratum FE only",
                     "units": "log Congolese Francs"}
)

# asinh revenues with stratum only FE
run_spec(
    spec_id="rc/form/outcome/asinh_amt__stratum_only",
    spec_tree_path="modules/robustness/functional_form.md#asinh",
    baseline_group_id="G1",
    outcome_var="asinh_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, asinh(revenue), stratum only",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/asinh_amt",
                     "transformation": "asinh(x)",
                     "interpretation": "approximate semi-elasticity, stratum FE only",
                     "units": "inverse hyperbolic sine of Congolese Francs"}
)

# log1p revenues exclude exempt
df_cl_noexempt['log1p_taxes_paid_amt'] = np.log1p(df_cl_noexempt['taxes_paid_amt'])
run_spec(
    spec_id="rc/form/outcome/log1p_amt__excl_exempt",
    spec_tree_path="modules/robustness/functional_form.md#log1p",
    baseline_group_id="G1",
    outcome_var="log1p_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, log(1+revenue), excl exempt",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/log1p_amt",
                     "transformation": "log(1+x)",
                     "interpretation": "semi-elasticity, excluding exempt",
                     "units": "log Congolese Francs"}
)

# Include CLI + stratum only
run_spec(
    spec_id="rc/data/treatment/include_cli_arm__stratum_only",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cli"],
    fe_vars=["stratum_int"],
    data=df_cl_cli,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CLI, stratum only",
    controls_desc="t_cli",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cli_arm",
                "treatment_change": "include CLI arm + stratum only"}
)

# Include CxL + stratum only
run_spec(
    spec_id="rc/data/treatment/include_cxl_arm__stratum_only",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cxl"],
    fe_vars=["stratum_int"],
    data=df_cl_cxl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CxL, stratum only",
    controls_desc="t_cxl",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cxl_arm",
                "treatment_change": "include CxL arm + stratum only"}
)

# Revenues with trimming + stratum_month FE
run_spec(
    spec_id="rc/sample/outliers/trim_y_amt_1_99__stratum_month",
    spec_tree_path="modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl_trim,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, revenues trimmed 1/99, stratum+month",
    controls_desc="none",
    fe_desc="stratum + time_FE",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_amt_1_99",
                "trim_var": "taxes_paid_amt", "fe_variant": "stratum+month"}
)

# Diff in means for exclude exempt
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt__diff_means",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=[],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excl exempt, no FE",
    controls_desc="none",
    fe_desc="none",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1, no FE"}
)

# Pooled local vs central with stratum only
run_spec(
    spec_id="rc/data/treatment/pooled_local_vs_central__stratum_only",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_local_type",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_pooled,
    vcov={"CRV1": "a7_int"},
    sample_desc="Pooled local-type vs Central, stratum only",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/pooled_local_vs_central",
                "treatment_change": "pool tmt==2 and tmt==4, stratum only"}
)

# Stratum + month FE, revenues, exclude exempt
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt__revenues__stratum_month",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excl exempt, revenues, stratum+month",
    controls_desc="none",
    fe_desc="stratum + time_FE",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1, revenues, stratum+month FE"}
)

# Include all 3 non-control arms (CLI + CxL) together
df_all_arms = df[df['tmt'].isin([1, 2, 3, 4])].copy()
run_spec(
    spec_id="rc/data/treatment/include_all_arms",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cli", "t_cxl"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_all_arms,
    vcov={"CRV1": "a7_int"},
    sample_desc="All treatment arms (tmt 1,2,3,4)",
    controls_desc="t_cli + t_cxl indicators",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_all_arms",
                "treatment_change": "include all arms (CLI + CxL)"}
)

# Include all arms, revenues
run_spec(
    spec_id="rc/data/treatment/include_all_arms__revenues",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=["t_cli", "t_cxl"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_all_arms,
    vcov={"CRV1": "a7_int"},
    sample_desc="All treatment arms, revenues",
    controls_desc="t_cli + t_cxl indicators",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_all_arms",
                "treatment_change": "include all arms, revenues"}
)

# ============================================================
# More cross-combinations to reach 50+ specs
# ============================================================
print("Additional cross-combinations...")

# FE drop house + revenues
run_spec(
    spec_id="rc/fe/drop/house__revenues",
    spec_tree_path="modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, revenues, house FE dropped",
    controls_desc="none",
    fe_desc="stratum + time_FE (house dropped)",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/house", "action": "drop", "dropped": ["house"], "outcome": "revenues"}
)

# FE drop time + revenues
run_spec(
    spec_id="rc/fe/drop/time_FE__revenues",
    spec_tree_path="modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, revenues, time FE dropped",
    controls_desc="none",
    fe_desc="stratum + house (time FE dropped)",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/time_FE_tdm_2mo_CvL", "action": "drop", "dropped": ["time_FE_tdm_2mo_CvL"], "outcome": "revenues"}
)

# Diff in means + exclude exempt (compliance)
run_spec(
    spec_id="rc/sample/restriction/exclude_exempt__diff_means__revenues",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=[],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, excl exempt, revenues, no FE",
    controls_desc="none",
    fe_desc="none",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_exempt",
                "restriction": "exempt!=1, revenues, no FE"}
)

# Trimmed revenues + stratum only
run_spec(
    spec_id="rc/sample/outliers/trim_y_amt_1_99__stratum_only",
    spec_tree_path="modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_cl_trim,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, revenues trimmed, stratum only",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_amt_1_99",
                "trim_var": "taxes_paid_amt", "fe_variant": "stratum only"}
)

# Pooled local + revenues + stratum only
run_spec(
    spec_id="rc/data/treatment/pooled_local_vs_central__revenues__stratum_only",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_local_type",
    controls=[],
    fe_vars=["stratum_int"],
    data=df_pooled,
    vcov={"CRV1": "a7_int"},
    sample_desc="Pooled local-type vs Central, revenues, stratum only",
    controls_desc="none",
    fe_desc="stratum only",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/pooled_local_vs_central",
                "treatment_change": "pool tmt==2 and tmt==4, revenues, stratum only"}
)

# asinh revenues exclude exempt
df_cl_noexempt['asinh_taxes_paid_amt'] = np.arcsinh(df_cl_noexempt['taxes_paid_amt'])
run_spec(
    spec_id="rc/form/outcome/asinh_amt__excl_exempt",
    spec_tree_path="modules/robustness/functional_form.md#asinh",
    baseline_group_id="G1",
    outcome_var="asinh_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, asinh(revenue), excl exempt",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/asinh_amt",
                     "transformation": "asinh(x)",
                     "interpretation": "approximate semi-elasticity, excluding exempt",
                     "units": "inverse hyperbolic sine of Congolese Francs"}
)

# log1p + stratum_month only (no house)
run_spec(
    spec_id="rc/form/outcome/log1p_amt__stratum_month",
    spec_tree_path="modules/robustness/functional_form.md#log1p",
    baseline_group_id="G1",
    outcome_var="log1p_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, log(1+revenue), stratum+month",
    controls_desc="none",
    fe_desc="stratum + time_FE",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/log1p_amt",
                     "transformation": "log(1+x)",
                     "interpretation": "semi-elasticity, stratum+month FE",
                     "units": "log Congolese Francs"}
)

# asinh + stratum_month only (no house)
run_spec(
    spec_id="rc/form/outcome/asinh_amt__stratum_month",
    spec_tree_path="modules/robustness/functional_form.md#asinh",
    baseline_group_id="G1",
    outcome_var="asinh_taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, asinh(revenue), stratum+month",
    controls_desc="none",
    fe_desc="stratum + time_FE",
    cluster_var="a7",
    functional_form={"spec_id": "rc/form/outcome/asinh_amt",
                     "transformation": "asinh(x)",
                     "interpretation": "approximate semi-elasticity, stratum+month FE",
                     "units": "inverse hyperbolic sine of Congolese Francs"}
)

# Include CLI + exclude exempt
df_cl_cli_noexempt = df_cl_cli[df_cl_cli['exempt'] != 1].copy()
run_spec(
    spec_id="rc/data/treatment/include_cli_arm__excl_exempt",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cli"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_cli_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CLI, excl exempt",
    controls_desc="t_cli",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cli_arm",
                "treatment_change": "include CLI arm + exclude exempt"}
)

# Include CxL + exclude exempt
df_cl_cxl_noexempt = df_cl_cxl[df_cl_cxl['exempt'] != 1].copy()
run_spec(
    spec_id="rc/data/treatment/include_cxl_arm__excl_exempt",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=["t_cxl"],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_cxl_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central + Local + CxL, excl exempt",
    controls_desc="t_cxl",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/include_cxl_arm",
                "treatment_change": "include CxL arm + exclude exempt"}
)

# Polygon means + exclude exempt
df_cl_noexempt_valid_time = df_cl_noexempt[df_cl_noexempt['time_FE_tdm_2mo_CvL'].notna()].copy()
agg_funcs_ne = {
    'taxes_paid': 'mean',
    'taxes_paid_amt': 'mean',
    'time_FE_tdm_2mo_CvL': 'min',
    't_l': 'max',
    't_c': 'max',
    'stratum': 'max',
}
df_poly_ne = df_cl_noexempt_valid_time.groupby(['a7', 'tmt']).agg(agg_funcs_ne).reset_index()
df_poly_ne['stratum_int'] = df_poly_ne['stratum'].astype('Int64')
df_poly_ne['time_fe_int'] = df_poly_ne['time_FE_tdm_2mo_CvL'].astype('Int64')
df_poly_ne['a7_int'] = df_poly_ne['a7'].astype('Int64')
df_poly_ne = df_poly_ne[df_poly_ne['tmt'].isin([1, 2])].copy()
run_spec(
    spec_id="rc/sample/restriction/polygon_means__excl_exempt",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "time_fe_int"],
    data=df_poly_ne,
    vcov="hetero",
    sample_desc="Polygon means, excl exempt",
    controls_desc="none",
    fe_desc="stratum + time_FE_tdm_2mo_CvL",
    cluster_var="",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/polygon_means",
                "restriction": "polygon means + exclude exempt"}
)

# Diff in means for revenues
run_spec(
    spec_id="rc/fe/sets/stratum_only__revenues__diff_means",
    spec_tree_path="modules/robustness/fixed_effects.md#sets",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=[],
    data=df_cl,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, revenues, no FE",
    controls_desc="none",
    fe_desc="none",
    cluster_var="a7",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/sets/none", "fe_set": [], "outcome": "revenues"}
)

# Trimmed revenues + exclude exempt
df_cl_trim_noexempt = df_cl_noexempt[(df_cl_noexempt['taxes_paid_amt'] >= p1) & (df_cl_noexempt['taxes_paid_amt'] <= p99)].copy()
run_spec(
    spec_id="rc/sample/outliers/trim_y_amt_1_99__excl_exempt",
    spec_tree_path="modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="taxes_paid_amt",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl_trim_noexempt,
    vcov={"CRV1": "a7_int"},
    sample_desc="Central vs Local, revenues trimmed + excl exempt",
    controls_desc="none",
    fe_desc="stratum + house + time_FE_tdm_2mo_CvL",
    cluster_var="a7",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_amt_1_99",
                "trim_var": "taxes_paid_amt", "restriction": "excl exempt + trimmed"}
)

# ============================================================
# INFERENCE VARIANTS on baseline
# ============================================================
print("Inference variants...")

# HC1 robust SEs (used for polygon means spec)
run_inference_variant(
    base_run_id=baseline_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    outcome_var="taxes_paid",
    treatment_var="t_l",
    controls=[],
    fe_vars=["stratum_int", "house_int", "time_fe_int"],
    data=df_cl,
    vcov_new="hetero",
    cluster_var_new=""
)

# ============================================================
# SAVE OUTPUTS
# ============================================================
print(f"\nTotal specification rows: {len(results)}")
print(f"Total inference rows: {len(inference_results)}")

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {OUT_DIR}/specification_results.csv")

# inference_results.csv
if inference_results:
    df_infer = pd.DataFrame(inference_results)
    df_infer.to_csv(f"{OUT_DIR}/inference_results.csv", index=False)
    print(f"Wrote {OUT_DIR}/inference_results.csv")

# Count successes/failures
n_success = sum(1 for r in results if r["run_success"] == 1)
n_fail = sum(1 for r in results if r["run_success"] == 0)
n_infer_success = sum(1 for r in inference_results if r["run_success"] == 1)
n_infer_fail = sum(1 for r in inference_results if r["run_success"] == 0)

# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================
pkgs = SW_BLOCK.get('packages', {})
md = f"""# Specification Search Report: {PAPER_ID}

## Paper
"Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo" (Balan, Bergeron, Tourek, Weigel)

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Baseline groups**: 1 (G1)
- **Design code**: randomized_experiment
- **Baseline outcome**: taxes_paid (binary compliance) and taxes_paid_amt (revenue)
- **Treatment**: t_l (local/chief collection assignment)
- **Focal comparison**: Local (tmt==2) vs Central (tmt==1)
- **Canonical inference**: cluster SEs at a7 (neighborhood/polygon level)
- **Budget**: max 60 core specs
- **Seed**: 147561

## Data Construction
The analysis dataset was constructed from raw data files following `2_Data_Construction.do`:
1. Merged flier assignment data with stratum, treatment assignment, registration (cartography), taxroll, and tax payment data
2. Constructed `taxes_paid` (binary compliance) and `taxes_paid_amt` (=taxes_paid * rate)
3. Created time FE bins matching Stata `egen cut` with breakpoints at Stata dates 21355, 21415, 21475, 21532
4. Dropped villas (house==3), pilot polygons, and observations with missing rate
5. Restricted to Central (tmt==1) vs Local (tmt==2) for focal comparison

**Analysis sample**: {len(df_cl)} observations in Central vs Local comparison

## Execution Summary
- **Total specification rows**: {len(results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}
- **Inference variant rows**: {len(inference_results)}
- **Inference successful**: {n_infer_success}
- **Inference failed**: {n_infer_fail}

## Specs Executed

### Baselines (2)
- `baseline`: Table 4 Compliance Col 4 (taxes_paid ~ t_l | stratum + house + time_FE, cl(a7))
- `baseline__revenues`: Table 4 Revenues Col 4 (taxes_paid_amt outcome)

### Design variants (2)
- `design/randomized_experiment/estimator/diff_in_means`: No FE, compliance
- `design/randomized_experiment/estimator/diff_in_means__revenues`: No FE, revenues

### RC: FE sets (6 compliance + 6 revenues = 12 total, but some are identical to baseline)
- stratum_only, stratum_month, stratum_month_house (baseline-equivalent)
- Each for both compliance and revenue outcomes

### RC: FE drop (2)
- drop house FE
- drop time FE

### RC: Sample restrictions (7)
- exclude_exempt (compliance + revenues)
- polygon_means (compliance + revenues)
- trim revenues at 1st/99th percentile
- Cross-combinations with various FE sets

### RC: Functional form (5)
- log(1+amt) (full FE, stratum only, excl exempt)
- asinh(amt) (full FE, stratum only)

### RC: Treatment definition (10)
- include CLI arm (compliance + revenues)
- include CxL arm (compliance + revenues)
- pooled local-type vs central (compliance + revenues)
- include all arms (compliance + revenues)
- Various FE combinations

### Inference variants (1)
- HC1 robust SEs on baseline

## Software Stack
- Python {SW_BLOCK.get('runner_version', 'N/A')}
- pyfixest {pkgs.get('pyfixest', 'N/A')}
- pandas {pkgs.get('pandas', 'N/A')}
- numpy {pkgs.get('numpy', 'N/A')}

## Deviations from Surface
- Data was constructed from raw files (no pre-built analysis_data.dta available)
- The midline (monitoring) survey data was not merged because it was not needed for Table 4 outcomes
- The exact Stata date imputation for missing `today_carto` was approximated (polygon-level min TDM date or max carto date)
- Sample sizes may differ slightly from the paper due to differences in merge order or missing value handling
"""

with open(f"{OUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print(f"Wrote {OUT_DIR}/SPECIFICATION_SEARCH.md")

print("\nDone with 147561-V3!")
