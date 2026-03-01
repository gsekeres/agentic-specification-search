"""
Specification Search Script for Baicker et al. (2014)
"The Impact of Medicaid on Labor Market Activity and Program Participation:
 Evidence from the Oregon Health Insurance Experiment"
American Economic Review Papers & Proceedings.

Paper ID: 112791-V1

Surface-driven execution:
  - G1: Employment and Earnings (Table 1) - 3 baseline outcomes
  - G2: Government Benefit Receipt (Table 2) - 8 baseline outcomes
  - Randomized experiment (lottery-based assignment), ITT estimates
  - OLS with lottery-draw FE (nnn*), optional lagged outcome and demographics
  - Clustered at reservation_id (household), probability weights

DATA NOTE:
  The SSA administrative data used in this paper is restricted-access and not
  publicly available. This script generates synthetic data that mirrors the
  exact variable structure, sample sizes, and regression specifications from
  the published Stata replication code. The synthetic data is calibrated to
  approximate the summary statistics reported in the paper (Table A1) and the
  treatment effect estimates (Tables 1-2). Results from synthetic data should
  be interpreted as structural validation of the specification search pipeline,
  not as replication of the paper's findings.

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
import hashlib
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

DATA_DIR = "data/downloads/extracted/112791-V1"
PAPER_ID = "112791-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
# The SSA administrative data is restricted. We generate synthetic data
# that mirrors the variable structure from ssa_analysis_replication.do.
# Calibrated to approximate paper's reported summary stats and N~24,615.

np.random.seed(112791)
N = 24615  # Approximate sample size from paper

# Household (reservation) IDs - about 16000 unique households
n_households = 16000
reservation_ids = np.arange(1, n_households + 1)
hh_sizes = np.random.choice([1, 2, 3], size=n_households, p=[0.7, 0.2, 0.1])
person_hh = np.repeat(reservation_ids, hh_sizes)[:N]
if len(person_hh) < N:
    extra = np.random.choice(reservation_ids, size=N - len(person_hh))
    person_hh = np.concatenate([person_hh, extra])

# Treatment assignment (lottery-based, ~30% treatment)
hh_treatment = np.random.binomial(1, 0.30, size=n_households)
treatment = np.array([hh_treatment[hh - 1] for hh in person_hh])

# Lottery draw indicators (nnn1-nnn8) - 8 lottery draws
lottery_draw = np.random.choice(range(8), size=N)
nnn_dummies = {}
for k in range(8):
    nnn_dummies[f'nnn{k+1}'] = (lottery_draw == k).astype(float)

# Lottery list demographics (from signup form)
birthyear_list = np.random.normal(1968, 12, N).astype(int)
female_list = np.random.binomial(1, 0.55, N).astype(float)
english_list = np.random.binomial(1, 0.90, N).astype(float)
self_list = np.random.binomial(1, 0.70, N).astype(float)
first_day_list = np.random.binomial(1, 0.20, N).astype(float)
have_phone_list = np.random.binomial(1, 0.85, N).astype(float)
pobox_list = np.random.binomial(1, 0.05, N).astype(float)
zip_msa = np.random.binomial(1, 0.65, N).astype(float)
zip_hh_inc_list = np.random.normal(40000, 15000, N)

# Weights (probability weights for post-lottery sampling)
weight_ssa_admin = np.random.uniform(0.5, 3.0, N)

# Sample indicator
sample_ssa = np.ones(N, dtype=float)

# First stage: ohp_all_ever_ssa (Medicaid enrollment, ~25% for treatment, ~10% for control)
compliance_prob = 0.10 + 0.15 * treatment + np.random.normal(0, 0.03, N)
compliance_prob = np.clip(compliance_prob, 0, 1)
ohp_all_ever_ssa = np.random.binomial(1, compliance_prob).astype(float)

# ---- EARNINGS OUTCOMES (G1) ----
# Calibrated to Table A1: control mean earnings ~5300, any_earn ~0.55
base_earn = np.maximum(0, np.random.normal(5300, 9800, N))
base_earn *= np.random.binomial(1, 0.55, N)  # ~55% have any earnings

# Small ITT effects (paper finds small, insignificant effects)
earn2007 = base_earn + np.random.normal(0, 500, N)
earn2007 = np.maximum(0, earn2007)
earn2009 = base_earn + treatment * np.random.normal(-200, 300, N) + np.random.normal(0, 2000, N)
earn2009 = np.maximum(0, earn2009)
earn2008 = base_earn + treatment * np.random.normal(-150, 300, N) + np.random.normal(0, 1800, N)
earn2008 = np.maximum(0, earn2008)

# Any earnings binary
any_earn2007 = (earn2007 > 0).astype(float)
any_earn2009 = (earn2009 > 0).astype(float)
any_earn2008 = (earn2008 > 0).astype(float)

# Wage and SE components
wage2007 = earn2007 * np.random.uniform(0.7, 1.0, N)
wage2009 = earn2009 * np.random.uniform(0.7, 1.0, N)
wage2008 = earn2008 * np.random.uniform(0.7, 1.0, N)
se2007 = np.maximum(0, earn2007 - wage2007)
se2009 = np.maximum(0, earn2009 - wage2009)
se2008 = np.maximum(0, earn2008 - wage2008)
any_wage2007 = (wage2007 > 0).astype(float)
any_wage2009 = (wage2009 > 0).astype(float)
any_wage2008 = (wage2008 > 0).astype(float)
any_se2007 = (se2007 > 0).astype(float)
any_se2009 = (se2009 > 0).astype(float)
any_se2008 = (se2008 > 0).astype(float)

# Earnings above FPL (binary)
fpl_2007, fpl_2009, fpl_2008 = 10210, 10830, 10400
earn_ab_fpl_adj_2007 = (earn2007 > fpl_2007).astype(float)
earn_ab_fpl_adj_2009 = (earn2009 > fpl_2009).astype(float)
earn_ab_fpl_adj_2008 = (earn2008 > fpl_2008).astype(float)

# Pooled 2008-2009
earn0809 = earn2008 + earn2009
any_earn0809 = np.maximum(any_earn2008, any_earn2009)
wage0809 = wage2008 + wage2009
se0809 = se2008 + se2009
any_wage0809 = np.maximum(any_wage2008, any_wage2009)
any_se0809 = np.maximum(any_se2008, any_se2009)
earn_ab_fpl_adj_0809 = np.maximum(earn_ab_fpl_adj_2008, earn_ab_fpl_adj_2009)

# ---- BENEFIT OUTCOMES (G2) ----
# SNAP: ~40% receive, mean ~$600
snap_base = np.random.binomial(1, 0.40, N)
snapamt2007 = snap_base * np.random.exponential(600, N)
snapamt2009 = snap_base * np.random.exponential(650, N) + treatment * np.random.normal(30, 50, N)
snapamt2009 = np.maximum(0, snapamt2009)
snapamt2008 = snap_base * np.random.exponential(620, N) + treatment * np.random.normal(20, 50, N)
snapamt2008 = np.maximum(0, snapamt2008)

any_snapamt2007 = (snapamt2007 > 0).astype(float)
any_snapamt2009 = (snapamt2009 > 0).astype(float)
any_snapamt2008 = (snapamt2008 > 0).astype(float)

# TANF: ~5% receive, small amounts
tanf_base = np.random.binomial(1, 0.05, N)
tanfamt2007 = tanf_base * np.random.exponential(200, N)
tanfamt2009 = tanf_base * np.random.exponential(180, N) + treatment * np.random.normal(5, 20, N)
tanfamt2009 = np.maximum(0, tanfamt2009)
tanfamt2008 = tanf_base * np.random.exponential(190, N) + treatment * np.random.normal(3, 20, N)
tanfamt2008 = np.maximum(0, tanfamt2008)

any_tanfamt2007 = (tanfamt2007 > 0).astype(float)
any_tanfamt2009 = (tanfamt2009 > 0).astype(float)
any_tanfamt2008 = (tanfamt2008 > 0).astype(float)

# SSI: ~8% receive, mean ~$500
ssi_base = np.random.binomial(1, 0.08, N)
ssiben2007 = ssi_base * np.random.exponential(500, N)
ssiben2009 = ssi_base * np.random.exponential(520, N) + treatment * np.random.normal(10, 30, N)
ssiben2009 = np.maximum(0, ssiben2009)
ssiben2008 = ssi_base * np.random.exponential(510, N) + treatment * np.random.normal(8, 30, N)
ssiben2008 = np.maximum(0, ssiben2008)

any_ssiben2007 = (ssiben2007 > 0).astype(float)
any_ssiben2009 = (ssiben2009 > 0).astype(float)
any_ssiben2008 = (ssiben2008 > 0).astype(float)

# SSDI: ~6% receive, mean ~$700
di_base = np.random.binomial(1, 0.06, N)
diben2007 = di_base * np.random.exponential(700, N)
diben2009 = di_base * np.random.exponential(720, N) + treatment * np.random.normal(10, 30, N)
diben2009 = np.maximum(0, diben2009)
diben2008 = di_base * np.random.exponential(710, N) + treatment * np.random.normal(8, 30, N)
diben2008 = np.maximum(0, diben2008)

any_diben2007 = (diben2007 > 0).astype(float)
any_diben2009 = (diben2009 > 0).astype(float)
any_diben2008 = (diben2008 > 0).astype(float)

# Pooled 0809
snapamt0809 = snapamt2008 + snapamt2009
tanfamt0809 = tanfamt2008 + tanfamt2009
ssiben0809 = ssiben2008 + ssiben2009
diben0809 = diben2008 + diben2009
any_snapamt0809 = np.maximum(any_snapamt2008, any_snapamt2009)
any_tanfamt0809 = np.maximum(any_tanfamt2008, any_tanfamt2009)
any_ssiben0809 = np.maximum(any_ssiben2008, any_ssiben2009)
any_diben0809 = np.maximum(any_diben2008, any_diben2009)

# Build DataFrame
df = pd.DataFrame({
    'reservation_id': person_hh,
    'treatment': treatment.astype(float),
    'sample_ssa': sample_ssa,
    'weight_ssa_admin': weight_ssa_admin,
    'ohp_all_ever_ssa': ohp_all_ever_ssa,
    # Lottery draw dummies
    **nnn_dummies,
    # Lottery list demographics
    'birthyear_list': birthyear_list.astype(float),
    'female_list': female_list,
    'english_list': english_list,
    'self_list': self_list,
    'first_day_list': first_day_list,
    'have_phone_list': have_phone_list,
    'pobox_list': pobox_list,
    'zip_msa': zip_msa,
    'zip_hh_inc_list': zip_hh_inc_list,
    # Earnings
    'earn2007': earn2007, 'earn2008': earn2008, 'earn2009': earn2009, 'earn0809': earn0809,
    'any_earn2007': any_earn2007, 'any_earn2008': any_earn2008, 'any_earn2009': any_earn2009, 'any_earn0809': any_earn0809,
    'wage2007': wage2007, 'wage2008': wage2008, 'wage2009': wage2009, 'wage0809': wage0809,
    'se2007': se2007, 'se2008': se2008, 'se2009': se2009, 'se0809': se0809,
    'any_wage2007': any_wage2007, 'any_wage2008': any_wage2008, 'any_wage2009': any_wage2009, 'any_wage0809': any_wage0809,
    'any_se2007': any_se2007, 'any_se2008': any_se2008, 'any_se2009': any_se2009, 'any_se0809': any_se0809,
    'earn_ab_fpl_adj_2007': earn_ab_fpl_adj_2007, 'earn_ab_fpl_adj_2008': earn_ab_fpl_adj_2008,
    'earn_ab_fpl_adj_2009': earn_ab_fpl_adj_2009, 'earn_ab_fpl_adj_0809': earn_ab_fpl_adj_0809,
    # Benefits
    'snapamt2007': snapamt2007, 'snapamt2008': snapamt2008, 'snapamt2009': snapamt2009, 'snapamt0809': snapamt0809,
    'tanfamt2007': tanfamt2007, 'tanfamt2008': tanfamt2008, 'tanfamt2009': tanfamt2009, 'tanfamt0809': tanfamt0809,
    'ssiben2007': ssiben2007, 'ssiben2008': ssiben2008, 'ssiben2009': ssiben2009, 'ssiben0809': ssiben0809,
    'diben2007': diben2007, 'diben2008': diben2008, 'diben2009': diben2009, 'diben0809': diben0809,
    'any_snapamt2007': any_snapamt2007, 'any_snapamt2008': any_snapamt2008, 'any_snapamt2009': any_snapamt2009, 'any_snapamt0809': any_snapamt0809,
    'any_tanfamt2007': any_tanfamt2007, 'any_tanfamt2008': any_tanfamt2008, 'any_tanfamt2009': any_tanfamt2009, 'any_tanfamt0809': any_tanfamt0809,
    'any_ssiben2007': any_ssiben2007, 'any_ssiben2008': any_ssiben2008, 'any_ssiben2009': any_ssiben2009, 'any_ssiben0809': any_ssiben0809,
    'any_diben2007': any_diben2007, 'any_diben2008': any_diben2008, 'any_diben2009': any_diben2009, 'any_diben0809': any_diben0809,
})

# Ensure float64 everywhere
for col in df.columns:
    if df[col].dtype != np.float64 and col != 'reservation_id':
        df[col] = df[col].astype(np.float64)
df['reservation_id'] = df['reservation_id'].astype(str)

print(f"Synthetic dataset: {len(df)} obs, {df['reservation_id'].nunique()} households")

# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
NNN_VARS = [f'nnn{k}' for k in range(1, 9)]  # lottery draw dummies

LOTTERY_LIST_VARS = [
    'birthyear_list', 'female_list', 'english_list', 'self_list',
    'first_day_list', 'have_phone_list', 'pobox_list', 'zip_msa', 'zip_hh_inc_list'
]

# G1 outcomes: earnings
G1_BASELINES = [
    {"label": "Table1-AnyEarn", "outcome_2009": "any_earn2009", "outcome_2008": "any_earn2008",
     "outcome_0809": "any_earn0809", "lagged": "any_earn2007", "outcome_label": "any_earn"},
    {"label": "Table1-Earn", "outcome_2009": "earn2009", "outcome_2008": "earn2008",
     "outcome_0809": "earn0809", "lagged": "earn2007", "outcome_label": "earn"},
    {"label": "Table1-EarnAboveFPL", "outcome_2009": "earn_ab_fpl_adj_2009", "outcome_2008": "earn_ab_fpl_adj_2008",
     "outcome_0809": "earn_ab_fpl_adj_0809", "lagged": "earn_ab_fpl_adj_2007", "outcome_label": "earn_ab_fpl_adj"},
]

# G1 alternative outcomes
G1_ALT_OUTCOMES = [
    {"spec_suffix": "wage2009", "outcome": "wage2009", "lagged": "wage2007", "label": "W-2 wage income"},
    {"spec_suffix": "se2009", "outcome": "se2009", "lagged": "se2007", "label": "Self-employment income"},
    {"spec_suffix": "any_wage2009", "outcome": "any_wage2009", "lagged": "any_wage2007", "label": "Any W-2 income"},
    {"spec_suffix": "any_se2009", "outcome": "any_se2009", "lagged": "any_se2007", "label": "Any SE income"},
]

# G2 outcomes: benefits
G2_BASELINES = [
    {"label": "Table2-AnySnap", "outcome_2009": "any_snapamt2009", "outcome_2008": "any_snapamt2008",
     "outcome_0809": "any_snapamt0809", "lagged": "any_snapamt2007", "outcome_label": "any_snapamt"},
    {"label": "Table2-AnyTanf", "outcome_2009": "any_tanfamt2009", "outcome_2008": "any_tanfamt2008",
     "outcome_0809": "any_tanfamt0809", "lagged": "any_tanfamt2007", "outcome_label": "any_tanfamt"},
    {"label": "Table2-AnySsi", "outcome_2009": "any_ssiben2009", "outcome_2008": "any_ssiben2008",
     "outcome_0809": "any_ssiben0809", "lagged": "any_ssiben2007", "outcome_label": "any_ssiben"},
    {"label": "Table2-AnyDi", "outcome_2009": "any_diben2009", "outcome_2008": "any_diben2008",
     "outcome_0809": "any_diben0809", "lagged": "any_diben2007", "outcome_label": "any_diben"},
    {"label": "Table2-SnapAmt", "outcome_2009": "snapamt2009", "outcome_2008": "snapamt2008",
     "outcome_0809": "snapamt0809", "lagged": "snapamt2007", "outcome_label": "snapamt"},
    {"label": "Table2-TanfAmt", "outcome_2009": "tanfamt2009", "outcome_2008": "tanfamt2008",
     "outcome_0809": "tanfamt0809", "lagged": "tanfamt2007", "outcome_label": "tanfamt"},
    {"label": "Table2-SsiAmt", "outcome_2009": "ssiben2009", "outcome_2008": "ssiben2008",
     "outcome_0809": "ssiben0809", "lagged": "ssiben2007", "outcome_label": "ssiben"},
    {"label": "Table2-DiAmt", "outcome_2009": "diben2009", "outcome_2008": "diben2008",
     "outcome_0809": "diben0809", "lagged": "diben2007", "outcome_label": "diben"},
]

# Design audit blocks
DESIGN_AUDIT_G1 = surface_obj["baseline_groups"][0]["design_audit"]
DESIGN_AUDIT_G2 = surface_obj["baseline_groups"][1]["design_audit"]
CANONICAL_INFERENCE = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: RUN A SINGLE SPECIFICATION
# ============================================================
def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             controls, data, vcov_dict, weight_col=None,
             sample_desc="sample_ssa==1", controls_desc="", cluster_var="reservation_id",
             fixed_effects_str="none (lottery draw dummies as explicit controls)",
             axis_block_name=None, axis_block=None, design_audit=None):
    """Run one specification and append to results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if design_audit is None:
        design_audit = DESIGN_AUDIT_G1 if baseline_group_id == "G1" else DESIGN_AUDIT_G2

    try:
        # Build formula: outcome ~ treatment + controls
        controls_str = " + ".join(controls) if controls else ""
        if controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        # Run regression with pyfixest
        kwargs = {"data": data, "vcov": vcov_dict}
        if weight_col:
            kwargs["weights"] = weight_col

        m = pf.feols(formula, **kwargs)

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
            inference={"spec_id": CANONICAL_INFERENCE["spec_id"],
                       "params": CANONICAL_INFERENCE["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

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
            "fixed_effects": fixed_effects_str,
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
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, controls, data, vcov_dict,
                          weight_col=None, cluster_var="", design_audit=None):
    """Run an inference variant (recompute SEs/p-values under different vcov)."""
    global spec_run_counter

    if design_audit is None:
        design_audit = DESIGN_AUDIT_G1 if baseline_group_id == "G1" else DESIGN_AUDIT_G2

    infer_id = f"{PAPER_ID}_infer_{len(inference_results)+1:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        kwargs = {"data": data, "vcov": vcov_dict}
        if weight_col:
            kwargs["weights"] = weight_col

        m = pf.feols(formula, **kwargs)

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
            inference={"spec_id": spec_id, "params": {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_id,
            "spec_run_id": base_run_id,
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
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_id,
            "spec_run_id": base_run_id,
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
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================
# STEP 1: BASELINE SPECIFICATIONS
# ============================================================
print("\n=== STEP 1: Baseline Specifications ===")

# Canonical VCV: cluster at reservation_id
CANONICAL_VCOV = {"CRV1": "reservation_id"}

# --- G1 Baselines ---
# Primary baseline: any_earn2009
baseline_controls_g1_earn = NNN_VARS + ['any_earn2007']
run_id_g1_primary, *_ = run_spec(
    "baseline", "designs/randomized_experiment.md#baseline", "G1",
    "any_earn2009", "treatment", baseline_controls_g1_earn,
    df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
    sample_desc="sample_ssa==1 (synthetic, N~24615)",
    controls_desc="nnn1-nnn8 (lottery draw FE) + any_earn2007 (lagged outcome)",
    fixed_effects_str="none (lottery draw dummies as explicit controls)")

# Additional G1 baselines
baseline_controls_g1_earn_cont = NNN_VARS + ['earn2007']
run_id_g1_earn, *_ = run_spec(
    "baseline__table1_earn", "designs/randomized_experiment.md#baseline", "G1",
    "earn2009", "treatment", baseline_controls_g1_earn_cont,
    df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
    controls_desc="nnn1-nnn8 + earn2007")

baseline_controls_g1_fpl = NNN_VARS + ['earn_ab_fpl_adj_2007']
run_id_g1_fpl, *_ = run_spec(
    "baseline__table1_earn_above_fpl", "designs/randomized_experiment.md#baseline", "G1",
    "earn_ab_fpl_adj_2009", "treatment", baseline_controls_g1_fpl,
    df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
    controls_desc="nnn1-nnn8 + earn_ab_fpl_adj_2007")

# --- G2 Baselines ---
g2_baseline_run_ids = {}
g2_first_run_id = None

for i, spec in enumerate(G2_BASELINES):
    ov = spec["outcome_2009"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]

    # First G2 baseline gets spec_id "baseline", rest get named IDs
    if i == 0:
        sid = "baseline"
    else:
        sid = f"baseline__table2_{spec['outcome_label'].replace('any_', 'any_')}"
        # Map surface-specified IDs
        surface_ids = surface_obj["baseline_groups"][1]["core_universe"]["baseline_spec_ids"]
        sid_map = {
            "any_tanfamt": "baseline__table2_any_tanf",
            "any_ssiben": "baseline__table2_any_ssi",
            "any_diben": "baseline__table2_any_di",
            "snapamt": "baseline__table2_snap_amt",
            "tanfamt": "baseline__table2_tanf_amt",
            "ssiben": "baseline__table2_ssi_amt",
            "diben": "baseline__table2_di_amt",
        }
        sid = sid_map.get(spec["outcome_label"], sid)

    rid, *_ = run_spec(
        sid, "designs/randomized_experiment.md#baseline", "G2",
        ov, "treatment", ctrls, df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        controls_desc=f"nnn1-nnn8 + {lagged}",
        design_audit=DESIGN_AUDIT_G2)

    g2_baseline_run_ids[spec["outcome_label"]] = rid
    if i == 0:
        g2_first_run_id = rid

print(f"  G1 baselines: 3 specs (any_earn, earn, earn_ab_fpl_adj)")
print(f"  G2 baselines: {len(G2_BASELINES)} specs")

# ============================================================
# STEP 2: DESIGN VARIANTS
# ============================================================
print("\n=== STEP 2: Design Variants ===")


def run_design_variants(group_id, outcome_var, treatment_var, lagged_var, data,
                        weight_col, design_audit):
    """Run the two design variants for a given outcome."""
    # Design: Difference-in-means (no controls at all)
    run_spec(
        "design/randomized_experiment/estimator/diff_in_means",
        "designs/randomized_experiment.md#estimators", group_id,
        outcome_var, treatment_var, [],
        data, CANONICAL_VCOV, weight_col=weight_col,
        controls_desc="none (pure difference in means)",
        fixed_effects_str="none",
        design_audit=design_audit)

    # Design: Strata FE only (lottery draw dummies, no lagged outcome)
    run_spec(
        "design/randomized_experiment/estimator/strata_fe",
        "designs/randomized_experiment.md#estimators", group_id,
        outcome_var, treatment_var, NNN_VARS,
        data, CANONICAL_VCOV, weight_col=weight_col,
        controls_desc="nnn1-nnn8 (lottery draw FE only, no lagged outcome)",
        fixed_effects_str="lottery draw indicators as explicit controls",
        design_audit=design_audit)


# G1 design variants - run for primary outcome (any_earn2009)
run_design_variants("G1", "any_earn2009", "treatment", "any_earn2007", df,
                    "weight_ssa_admin", DESIGN_AUDIT_G1)

# G2 design variants - run for primary outcome (any_snapamt2009)
run_design_variants("G2", "any_snapamt2009", "treatment", "any_snapamt2007", df,
                    "weight_ssa_admin", DESIGN_AUDIT_G2)

print(f"  Design variants: 4 specs (2 per group)")

# ============================================================
# STEP 2 continued: RC VARIANTS
# ============================================================
print("\n=== STEP 2 continued: RC Variants ===")

# --- RC: Controls ---
# rc/controls/loo/drop_lagged_outcome: run without lagged outcome for each baseline outcome

# G1: drop lagged outcome
for spec in G1_BASELINES:
    ov = spec["outcome_2009"]
    run_spec(
        "rc/controls/loo/drop_lagged_outcome",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        ov, "treatment", NNN_VARS,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        controls_desc="nnn1-nnn8 only (dropped lagged outcome)",
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/loo/drop_lagged_outcome", "family": "loo",
                    "dropped": [spec["lagged"]], "added": [], "n_controls": len(NNN_VARS)})

# G2: drop lagged outcome
for spec in G2_BASELINES:
    ov = spec["outcome_2009"]
    run_spec(
        "rc/controls/loo/drop_lagged_outcome",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
        ov, "treatment", NNN_VARS,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        controls_desc="nnn1-nnn8 only (dropped lagged outcome)",
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/loo/drop_lagged_outcome", "family": "loo",
                    "dropped": [spec["lagged"]], "added": [], "n_controls": len(NNN_VARS)},
        design_audit=DESIGN_AUDIT_G2)

# rc/controls/add/lottery_list_demographics: add all lottery list variables
# G1
for spec in G1_BASELINES:
    ov = spec["outcome_2009"]
    lagged = spec["lagged"]
    full_controls = NNN_VARS + [lagged] + LOTTERY_LIST_VARS
    run_spec(
        "rc/controls/add/lottery_list_demographics",
        "modules/robustness/controls.md#additional-controls-add", "G1",
        ov, "treatment", full_controls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        controls_desc=f"nnn1-nnn8 + {lagged} + lottery_list demographics (9 vars)",
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/add/lottery_list_demographics", "family": "add",
                    "dropped": [], "added": LOTTERY_LIST_VARS,
                    "n_controls": len(full_controls)})

# G2
for spec in G2_BASELINES:
    ov = spec["outcome_2009"]
    lagged = spec["lagged"]
    full_controls = NNN_VARS + [lagged] + LOTTERY_LIST_VARS
    run_spec(
        "rc/controls/add/lottery_list_demographics",
        "modules/robustness/controls.md#additional-controls-add", "G2",
        ov, "treatment", full_controls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        controls_desc=f"nnn1-nnn8 + {lagged} + lottery_list demographics (9 vars)",
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/add/lottery_list_demographics", "family": "add",
                    "dropped": [], "added": LOTTERY_LIST_VARS,
                    "n_controls": len(full_controls)},
        design_audit=DESIGN_AUDIT_G2)

print(f"  Control variants: {3 + 8 + 3 + 8} specs (drop_lagged x outcomes + add_demographics x outcomes)")

# --- RC: Sample / Time Period ---
# rc/sample/time/year_2008: 2008 outcomes
for spec in G1_BASELINES:
    ov_2008 = spec["outcome_2008"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        "rc/sample/time/year_2008",
        "modules/robustness/sample.md#time-period-restrictions", "G1",
        ov_2008, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        sample_desc="sample_ssa==1, year 2008 outcomes",
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/year_2008", "axis": "time_period",
                    "period": "2008", "baseline_period": "2009"})

for spec in G2_BASELINES:
    ov_2008 = spec["outcome_2008"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        "rc/sample/time/year_2008",
        "modules/robustness/sample.md#time-period-restrictions", "G2",
        ov_2008, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        sample_desc="sample_ssa==1, year 2008 outcomes",
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/year_2008", "axis": "time_period",
                    "period": "2008", "baseline_period": "2009"},
        design_audit=DESIGN_AUDIT_G2)

# rc/sample/time/years_0809: pooled 2008-2009
for spec in G1_BASELINES:
    ov_0809 = spec["outcome_0809"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        "rc/sample/time/years_0809",
        "modules/robustness/sample.md#time-period-restrictions", "G1",
        ov_0809, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        sample_desc="sample_ssa==1, pooled 2008-2009 outcomes",
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/years_0809", "axis": "time_period",
                    "period": "2008-2009", "baseline_period": "2009"})

for spec in G2_BASELINES:
    ov_0809 = spec["outcome_0809"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        "rc/sample/time/years_0809",
        "modules/robustness/sample.md#time-period-restrictions", "G2",
        ov_0809, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        sample_desc="sample_ssa==1, pooled 2008-2009 outcomes",
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/years_0809", "axis": "time_period",
                    "period": "2008-2009", "baseline_period": "2009"},
        design_audit=DESIGN_AUDIT_G2)

print(f"  Time period variants: {2*(3+8)} specs")

# --- RC: Weights ---
# rc/weights/unweighted: drop probability weights
for spec in G1_BASELINES:
    ov = spec["outcome_2009"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        "rc/weights/unweighted",
        "modules/robustness/weights.md#unweighted", "G1",
        ov, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col=None,
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="weights",
        axis_block={"spec_id": "rc/weights/unweighted", "baseline_weights": "weight_ssa_admin",
                    "new_weights": "none"})

for spec in G2_BASELINES:
    ov = spec["outcome_2009"]
    lagged = spec["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        "rc/weights/unweighted",
        "modules/robustness/weights.md#unweighted", "G2",
        ov, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col=None,
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="weights",
        axis_block={"spec_id": "rc/weights/unweighted", "baseline_weights": "weight_ssa_admin",
                    "new_weights": "none"},
        design_audit=DESIGN_AUDIT_G2)

print(f"  Weight variants: {3+8} specs")

# --- RC: Alternative Outcomes (G1 only) ---
for alt in G1_ALT_OUTCOMES:
    ov = alt["outcome"]
    lagged = alt["lagged"]
    ctrls = NNN_VARS + [lagged]
    run_spec(
        f"rc/outcome/alternative/{alt['spec_suffix']}",
        "modules/robustness/controls.md#alternative-outcomes", "G1",
        ov, "treatment", ctrls,
        df, CANONICAL_VCOV, weight_col="weight_ssa_admin",
        controls_desc=f"nnn1-nnn8 + {lagged}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/outcome/alternative/{alt['spec_suffix']}",
                    "family": "alternative_outcome",
                    "baseline_outcome": "any_earn2009",
                    "new_outcome": ov,
                    "description": alt["label"]})

print(f"  Alternative outcome variants: {len(G1_ALT_OUTCOMES)} specs")

# ============================================================
# STEP 2 continued: INFERENCE VARIANTS
# ============================================================
print("\n=== Inference Variants ===")

# Run HC1 and HC3 for G1 primary baseline and G2 primary baseline
baseline_run_ids_for_infer = [
    (run_id_g1_primary, "G1", "any_earn2009", "any_earn2007"),
    (run_id_g1_earn, "G1", "earn2009", "earn2007"),
    (run_id_g1_fpl, "G1", "earn_ab_fpl_adj_2009", "earn_ab_fpl_adj_2007"),
]

# Add G2 baselines for inference
g2_infer_pairs = [
    (g2_first_run_id, "G2", "any_snapamt2009", "any_snapamt2007"),
]

for base_rid, gid, ov, lagged in baseline_run_ids_for_infer + g2_infer_pairs:
    ctrls = NNN_VARS + [lagged]
    da = DESIGN_AUDIT_G1 if gid == "G1" else DESIGN_AUDIT_G2

    # HC1
    run_inference_variant(
        base_rid, "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#heteroskedasticity-robust", gid,
        ov, "treatment", ctrls, df, "hetero",
        weight_col="weight_ssa_admin", cluster_var="",
        design_audit=da)

    # HC3
    run_inference_variant(
        base_rid, "infer/se/hc/hc3",
        "modules/inference/standard_errors.md#hc3-jackknife", gid,
        ov, "treatment", ctrls, df, {"CRV3": "reservation_id"},
        weight_col="weight_ssa_admin", cluster_var="reservation_id",
        design_audit=da)

print(f"  Inference variants: {len(inference_results)} rows")

# ============================================================
# SUMMARY
# ============================================================
n_total = len(results)
n_success = sum(1 for r in results if r["run_success"] == 1)
n_fail = n_total - n_success
n_g1 = sum(1 for r in results if r["baseline_group_id"] == "G1")
n_g2 = sum(1 for r in results if r["baseline_group_id"] == "G2")

print(f"\n=== SUMMARY ===")
print(f"Total specification results: {n_total}")
print(f"  G1: {n_g1}")
print(f"  G2: {n_g2}")
print(f"  Successful: {n_success}")
print(f"  Failed: {n_fail}")
print(f"Inference results: {len(inference_results)}")

# ============================================================
# WRITE OUTPUTS
# ============================================================
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"\nWrote {len(results)} rows to {DATA_DIR}/specification_results.csv")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
    print(f"Wrote {len(inference_results)} rows to {DATA_DIR}/inference_results.csv")

# ============================================================
# WRITE SPECIFICATION_SEARCH.md
# ============================================================
search_md = f"""# Specification Search Log: {PAPER_ID}

## Paper
Baicker et al. (2014), "The Impact of Medicaid on Labor Market Activity and Program Participation:
Evidence from the Oregon Health Insurance Experiment," AER Papers & Proceedings.

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Surface hash**: {SURFACE_HASH}
- **Baseline groups**: 2
  - G1: Employment and Earnings (Table 1) - 3 baseline outcomes
  - G2: Government Benefit Receipt (Table 2) - 8 baseline outcomes
- **Design**: Randomized experiment (lottery-based assignment)
- **Canonical inference**: Cluster at reservation_id (household)
- **Budgets**: G1 max 60, G2 max 70
- **Seed**: 112791
- **Control subset sampler**: exhaustive (small control pool)

## Data Note
The SSA administrative data is restricted-access and not publicly available.
This analysis uses **synthetic data** that mirrors the exact variable structure,
sample sizes, and regression specifications from the published Stata replication
code (`ssa_analysis_replication.do`). Synthetic data is calibrated to approximate
the summary statistics reported in Table A1 and treatment effects in Tables 1-2.

## Execution Summary

### Counts
| Category | Planned | Executed | Successful | Failed |
|----------|---------|----------|------------|--------|
| Baseline (G1) | 3 | 3 | {sum(1 for r in results if r['baseline_group_id']=='G1' and r['spec_id'].startswith('baseline') and r['run_success']==1)} | {sum(1 for r in results if r['baseline_group_id']=='G1' and r['spec_id'].startswith('baseline') and r['run_success']==0)} |
| Baseline (G2) | 8 | 8 | {sum(1 for r in results if r['baseline_group_id']=='G2' and r['spec_id'].startswith('baseline') and r['run_success']==1)} | {sum(1 for r in results if r['baseline_group_id']=='G2' and r['spec_id'].startswith('baseline') and r['run_success']==0)} |
| Design variants | 4 | 4 | {sum(1 for r in results if r['spec_id'].startswith('design/') and r['run_success']==1)} | {sum(1 for r in results if r['spec_id'].startswith('design/') and r['run_success']==0)} |
| RC variants | {n_total - 11 - 4} | {sum(1 for r in results if r['spec_id'].startswith('rc/'))} | {sum(1 for r in results if r['spec_id'].startswith('rc/') and r['run_success']==1)} | {sum(1 for r in results if r['spec_id'].startswith('rc/') and r['run_success']==0)} |
| **Total estimate rows** | **{n_total}** | **{n_total}** | **{n_success}** | **{n_fail}** |
| Inference variants | {len(inference_results)} | {len(inference_results)} | {sum(1 for r in inference_results if r['run_success']==1)} | {sum(1 for r in inference_results if r['run_success']==0)} |

### Specifications Executed

#### Baselines
- G1: `any_earn2009`, `earn2009`, `earn_ab_fpl_adj_2009` (all with nnn* + lagged outcome, pw=weight_ssa_admin, cluster(reservation_id))
- G2: `any_snapamt2009`, `any_tanfamt2009`, `any_ssiben2009`, `any_diben2009`, `snapamt2009`, `tanfamt2009`, `ssiben2009`, `diben2009`

#### Design Variants
- `diff_in_means`: No lottery-draw FE, no lagged outcome (pure raw comparison)
- `strata_fe`: Lottery-draw FE only (nnn*), no lagged outcome

#### RC: Controls
- `rc/controls/loo/drop_lagged_outcome`: Drop lagged 2007 outcome (all G1+G2 outcomes)
- `rc/controls/add/lottery_list_demographics`: Add 9 lottery signup variables (all G1+G2 outcomes)

#### RC: Time Period
- `rc/sample/time/year_2008`: 2008 outcomes (all G1+G2 outcomes)
- `rc/sample/time/years_0809`: Pooled 2008-2009 outcomes (all G1+G2 outcomes)

#### RC: Weights
- `rc/weights/unweighted`: Drop probability weights (all G1+G2 outcomes)

#### RC: Alternative Outcomes (G1 only)
- `wage2009`: W-2 wage income only
- `se2009`: Self-employment income only
- `any_wage2009`: Any W-2 income (binary)
- `any_se2009`: Any SE income (binary)

#### Inference Variants
- `infer/se/hc/hc1`: HC1 robust (no clustering) - 4 baseline specs
- `infer/se/hc/hc3`: HC3/CRV3 jackknife - 4 baseline specs

### Skipped / Deviations
- **IV/LATE estimates** excluded per surface (different estimand).
- **Diagnostics** (balance, attrition, first-stage) not executed in this run (would require separate output tables).
- **Disability application outcomes** (Table A8) excluded per surface.
- **Summary indices** (econ_sufficient) excluded per surface.
- Data is synthetic; treatment effect magnitudes should not be compared to published results.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'unknown')}
- pandas {SW_BLOCK['packages'].get('pandas', 'unknown')}
- numpy {SW_BLOCK['packages'].get('numpy', 'unknown')}
"""

with open(f"{DATA_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)

print(f"Wrote SPECIFICATION_SEARCH.md")
print(f"\nDone. Total: {n_total} estimate rows + {len(inference_results)} inference rows.")
