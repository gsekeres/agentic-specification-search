"""
Specification Search Script for Kumhof, Ranciere & Winant (2015)
"Inequality, Leverage, and Crises"
American Economic Review, 105(3), 1217-1245.

Paper ID: 112840-V1

Surface-driven execution:
  - G1: changerealdebt ~ lag_d_ave_finindex1 + controls | country FE
  - Panel FE (within estimator), HC1 robust SE (Stata xtreg, fe vce(robust))
  - 22 OECD countries, 1975-2005
  - Baselines: Table O1 columns 1, 5, 6
  - Design alternative: first difference estimator
  - RC axes: controls, sample, functional form, FE, preprocessing, joint

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
import itertools
import hashlib
from scipy import stats

warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash,
    software_block
)

PAPER_ID = "112840-V1"
DATA_DIR = "data/downloads/extracted/112840-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = compute_surface_hash(surface_obj)
SW_BLOCK = software_block()

G1 = surface_obj["baseline_groups"][0]
G1_DESIGN_AUDIT = G1["design_audit"]
G1_INFERENCE_CANONICAL = G1["inference_plan"]["canonical"]

TREATMENT_VAR = "lag_d_ave_finindex1"
OUTCOME_VAR = "changerealdebt"
MANDATORY_CONTROLS = ["lag_debtgdp", "lagchangerealgdp"]
OPTIONAL_CONTROLS_TOP1 = ["lag_emu_dum", "size_product1", "d_dep_ratio_old", "changetop1incomeshare"]
OPTIONAL_CONTROLS_GINI = ["lag_emu_dum", "size_product1", "d_dep_ratio_old", "lag_changeave_gini_gross"]

# ============================================================
# DATA CONSTRUCTION (replicating the Stata do file exactly)
# ============================================================

def build_dataset():
    """Build analysis dataset from source files, replicating the Stata do file."""
    # Read source data
    gini_raw = pd.read_csv(f"{DATA_DIR}/data/20120268_dataset/gini_gross.csv")
    gini_raw = gini_raw[['country', 'year', 'gini_gross']].dropna(subset=['gini_gross'])

    country_ccode = {
        'Australia': 'AUS', 'Austria': 'AUT', 'Belgium': 'BEL', 'Canada': 'CAN',
        'Denmark': 'DNK', 'Finland': 'FIN', 'France': 'FRA', 'Germany': 'DEU',
        'Greece': 'GRC', 'Ireland': 'IRL', 'Italy': 'ITA', 'Japan': 'JPN',
        'Korea': 'KOR', 'Netherlands': 'NLD', 'New Zealand': 'NZL', 'Norway': 'NOR',
        'Portugal': 'PRT', 'Spain': 'ESP', 'Sweden': 'SWE', 'Switzerland': 'CHE',
        'United Kingdom': 'GBR', 'United States': 'USA'
    }
    gini_raw['ccode'] = gini_raw['country'].map(country_ccode)
    gini = gini_raw.dropna(subset=['ccode'])[['ccode', 'year', 'gini_gross']]

    subset = pd.read_stata(f"{DATA_DIR}/data/20120268_dataset/subset.dta")
    depratio = pd.read_stata(f"{DATA_DIR}/data/20120268_dataset/DepRatioOld.dta")
    income = pd.read_stata(f"{DATA_DIR}/data/20120268_dataset/incomeinequality.dta")

    # Merge gini -> subset -> depratio
    df = gini.merge(subset, on=['ccode', 'year'], how='outer')
    df = df.merge(depratio, on=['ccode', 'year'], how='outer')

    # EMU dummy
    emu_countries = ['AUT','BEL','FIN','FRA','DEU','IRL','ITA','GRC','NLD','PRT','ESP']
    df['emu_dum'] = ((df['year'].isin([1998, 1999, 2000])) &
                     (df['ccode'].isin(emu_countries))).astype(float)

    # Rescale debt/GDP from percentage to ratio
    df['debtgdp'] = df['debtgdp'] / 100

    # GDP weights
    df['WorldGdp'] = df.groupby('year')['gdptotal'].transform('sum')
    df['weights'] = df['gdptotal'] / df['WorldGdp']
    df['w_finindex1'] = df['finindex1'] * df['weights']
    df['w_finindex2'] = df['finindex2'] * df['weights']
    df['w_gini_gross'] = df['gini_gross'] * df['weights']

    # Real debt and logs
    df['realdebt'] = df['debtgdp'] * df['realgdp']
    df['logrealdebt'] = np.log(df['realdebt'])
    df['logrealgdp'] = np.log(df['realgdp'])

    # Weighted average indexes (across countries within year)
    df['ave_finindex1'] = df.groupby('year')['w_finindex1'].transform('sum')
    df['ave_finindex2'] = df.groupby('year')['w_finindex2'].transform('sum')
    df['ave_gini_gross'] = df.groupby('year')['w_gini_gross'].transform('sum')

    # Sort by country-year for lag generation
    df = df.sort_values(['ccode', 'year']).reset_index(drop=True)

    # Lag and change variables (within country)
    df['lag_logrealdebt'] = df.groupby('ccode')['logrealdebt'].shift(1)
    df['changerealdebt'] = df['logrealdebt'] - df['lag_logrealdebt']
    df['lag_logrealgdp'] = df.groupby('ccode')['logrealgdp'].shift(1)
    df['changerealgdp'] = df['logrealgdp'] - df['lag_logrealgdp']
    df['lagchangerealgdp'] = df.groupby('ccode')['changerealgdp'].shift(1)

    df['lag_ave_gini_gross'] = df.groupby('ccode')['ave_gini_gross'].shift(1)
    df['changeave_gini_gross'] = df['ave_gini_gross'] - df['lag_ave_gini_gross']
    df['lag_changeave_gini_gross'] = df.groupby('ccode')['changeave_gini_gross'].shift(1)

    df['lag_debtgdp'] = df.groupby('ccode')['debtgdp'].shift(1)
    df['lag_ave_finindex1'] = df.groupby('ccode')['ave_finindex1'].shift(1)
    df['lag_ave_finindex2'] = df.groupby('ccode')['ave_finindex2'].shift(1)
    df['d_ave_finindex1'] = df['ave_finindex1'] - df['lag_ave_finindex1']
    df['d_ave_finindex2'] = df['ave_finindex2'] - df['lag_ave_finindex2']
    df['d_dep_ratio_old'] = df['dep_ratio_old'] - df.groupby('ccode')['dep_ratio_old'].shift(1)
    df['lag_d_ave_finindex1'] = df.groupby('ccode')['d_ave_finindex1'].shift(1)
    df['lag_emu_dum'] = df.groupby('ccode')['emu_dum'].shift(1)
    df['lag_weights'] = df.groupby('ccode')['weights'].shift(1)
    df['size_product1'] = df['lag_weights'] * df['lag_d_ave_finindex1']
    df['sizefin2_product1'] = df['lag_weights'] * df['d_ave_finindex2']

    # Sample restrictions
    df = df[(df['year'] > 1974) & (df['year'] <= 2005)]
    df = df[~df['ccode'].isin(['MEX', 'TUR', 'HUN', 'CZE', 'ISL', 'POL'])]

    # Merge income inequality data
    df = df.merge(income[['ccode', 'year', 'changetop1incomeshare']], on=['ccode', 'year'], how='left')

    # Ensure cn is integer
    df['cn'] = df['cn'].astype(int)

    # Region mapping for region FE
    region_map = {
        'AUS': 'Oceania', 'NZL': 'Oceania',
        'JPN': 'Asia', 'KOR': 'Asia',
        'CAN': 'NorthAm', 'USA': 'NorthAm',
        'GBR': 'NorthEurope', 'IRL': 'NorthEurope', 'DNK': 'NorthEurope',
        'NOR': 'NorthEurope', 'SWE': 'NorthEurope', 'FIN': 'NorthEurope',
        'NLD': 'WestEurope', 'BEL': 'WestEurope', 'DEU': 'WestEurope',
        'AUT': 'WestEurope', 'CHE': 'WestEurope', 'FRA': 'WestEurope',
        'ESP': 'SouthEurope', 'PRT': 'SouthEurope', 'ITA': 'SouthEurope',
        'GRC': 'SouthEurope',
    }
    df['region'] = df['ccode'].map(region_map).fillna('Other')

    # Create year_fe column for year fixed effects
    df['year_int'] = df['year'].astype(int)

    return df


# Build main dataset
df_full = build_dataset()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

spec_results = []
inference_results = []
spec_counter = 0
infer_counter = 0


def next_spec_run_id():
    global spec_counter
    spec_counter += 1
    return f"{PAPER_ID}_spec_{spec_counter:03d}"


def next_infer_run_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}_infer_{infer_counter:03d}"


def make_design_block(overrides=None):
    """Create a design audit block for panel_fixed_effects."""
    block = dict(G1_DESIGN_AUDIT)
    if overrides:
        block.update(overrides)
    return {"panel_fixed_effects": block}


def run_panel_fe(data, outcome, treatment, controls, fe_var="cn", vcov="hetero",
                 spec_id="baseline", spec_run_id=None, baseline_group_id="G1",
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 sample_desc="22 OECD countries, 1975-2005",
                 fixed_effects_desc="country FE", controls_desc="",
                 cluster_var="", treatment_var_name=None, outcome_var_name=None,
                 axis_block_name=None, axis_block=None, design_overrides=None,
                 extra=None):
    """Run a single panel FE regression and record the result."""
    if spec_run_id is None:
        spec_run_id = next_spec_run_id()
    if treatment_var_name is None:
        treatment_var_name = treatment
    if outcome_var_name is None:
        outcome_var_name = outcome

    all_vars = [outcome, treatment] + controls
    rhs = " + ".join([treatment] + controls)
    formula = f"{outcome} ~ {rhs} | {fe_var}"

    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var_name,
        "treatment_var": treatment_var_name,
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects_desc,
        "controls_desc": ", ".join(controls),
        "cluster_var": cluster_var,
    }

    try:
        dfreg = data.dropna(subset=all_vars + [fe_var]).copy()
        if vcov == "hetero":
            m = pf.feols(formula, data=dfreg, vcov="hetero")
        elif isinstance(vcov, dict):
            m = pf.feols(formula, data=dfreg, vcov=vcov)
        else:
            m = pf.feols(formula, data=dfreg, vcov=vcov)

        coef_val = float(m.coef()[treatment])
        se_val = float(m.se()[treatment])
        pval = float(m.pvalue()[treatment])
        ci = m.confint()
        ci_lo = float(ci.loc[treatment, ci.columns[0]])
        ci_hi = float(ci.loc[treatment, ci.columns[1]])
        nobs = int(m._N)
        r2 = float(m._r2)

        coefficients = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=coefficients,
            inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"],
                       "params": G1_INFERENCE_CANONICAL.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design=make_design_block(design_overrides),
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            extra=extra,
        )

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })

    spec_results.append(row)
    return row


def run_first_difference(data, outcome, treatment, controls, entity_var="cn", time_var="year",
                         spec_id="design/panel_fixed_effects/estimator/first_difference",
                         spec_run_id=None, baseline_group_id="G1",
                         spec_tree_path="specification_tree/designs/panel_fixed_effects.md#first_difference",
                         sample_desc="22 OECD countries, 1975-2005",
                         controls_desc="", treatment_var_name=None, outcome_var_name=None,
                         extra=None):
    """Run first-difference estimator (pooled OLS on first-differenced data)."""
    if spec_run_id is None:
        spec_run_id = next_spec_run_id()
    if treatment_var_name is None:
        treatment_var_name = treatment
    if outcome_var_name is None:
        outcome_var_name = outcome

    all_vars = [outcome, treatment] + controls

    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var_name,
        "treatment_var": treatment_var_name,
        "sample_desc": sample_desc,
        "fixed_effects": "first-differenced (no explicit FE)",
        "controls_desc": controls_desc or ", ".join(controls),
        "cluster_var": "",
    }

    try:
        dfreg = data.dropna(subset=all_vars + [entity_var, time_var]).copy()
        dfreg = dfreg.sort_values([entity_var, time_var])

        # First-difference all variables within entity
        fd_vars = {}
        for v in [outcome, treatment] + controls:
            fd_vars[f"fd_{v}"] = dfreg.groupby(entity_var)[v].diff()
        fd_df = pd.DataFrame(fd_vars)
        fd_df = fd_df.dropna()

        # Pooled OLS on first-differenced data (with intercept for potential drift)
        rhs = " + ".join([f"fd_{treatment}"] + [f"fd_{c}" for c in controls])
        formula = f"fd_{outcome} ~ {rhs}"
        m = pf.feols(formula, data=fd_df, vcov="hetero")

        focal = f"fd_{treatment}"
        coef_val = float(m.coef()[focal])
        se_val = float(m.se()[focal])
        pval = float(m.pvalue()[focal])
        ci = m.confint()
        ci_lo = float(ci.loc[focal, ci.columns[0]])
        ci_hi = float(ci.loc[focal, ci.columns[1]])
        nobs = int(m._N)
        r2 = float(m._r2)

        coefficients = {k: float(v) for k, v in m.coef().items()}

        design_block = make_design_block({"estimator": "first_difference"})
        payload = make_success_payload(
            coefficients=coefficients,
            inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"],
                       "params": G1_INFERENCE_CANONICAL.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design=design_block,
            extra=extra,
        )

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="first_difference_estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })

    spec_results.append(row)
    return row


def run_inference_variant(base_row, data, outcome, treatment, controls, fe_var,
                          infer_spec_id, infer_vcov, infer_params=None,
                          spec_tree_path="specification_tree/modules/inference/standard_errors.md"):
    """Recompute SE/p-value under an alternative inference choice."""
    infer_run_id = next_infer_run_id()
    all_vars = [outcome, treatment] + controls
    rhs = " + ".join([treatment] + controls)
    formula = f"{outcome} ~ {rhs} | {fe_var}"

    row = {
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": base_row["spec_run_id"],
        "spec_id": infer_spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": base_row["baseline_group_id"],
        "outcome_var": base_row["outcome_var"],
        "treatment_var": base_row["treatment_var"],
    }

    try:
        dfreg = data.dropna(subset=all_vars + [fe_var]).copy()
        m = pf.feols(formula, data=dfreg, vcov=infer_vcov)

        coef_val = float(m.coef()[treatment])
        se_val = float(m.se()[treatment])
        pval = float(m.pvalue()[treatment])
        ci = m.confint()
        ci_lo = float(ci.loc[treatment, ci.columns[0]])
        ci_hi = float(ci.loc[treatment, ci.columns[1]])
        nobs = int(m._N)
        r2 = float(m._r2)

        payload = make_success_payload(
            coefficients={k: float(v) for k, v in m.coef().items()},
            inference={"spec_id": infer_spec_id, "params": infer_params or {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design=make_design_block(),
        )

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })

    inference_results.append(row)
    return row


# ============================================================
# STEP 1: BASELINE SPECIFICATIONS
# ============================================================
print("=== Step 1: Baseline Specifications ===")

# Baseline: Table O1, Col 1 (minimal controls)
b_col1 = run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS,
    spec_id="baseline",
    controls_desc="lag_debtgdp, lagchangerealgdp (Table O1 Col 1 -- minimal)",
    extra={"table": "O1", "column": 1},
)
print(f"  Baseline (Col1): coef={b_col1['coefficient']:.4f}, p={b_col1['p_value']:.4f}, N={b_col1['n_obs']}")

# Baseline: Table O1, Col 5 (extended, top 1% income share)
b_col5 = run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS + OPTIONAL_CONTROLS_TOP1,
    spec_id="baseline__tableO1_col5",
    controls_desc="lag_debtgdp, lagchangerealgdp, lag_emu_dum, size_product1, d_dep_ratio_old, changetop1incomeshare (Table O1 Col 5)",
    extra={"table": "O1", "column": 5},
)
print(f"  Baseline (Col5): coef={b_col5['coefficient']:.4f}, p={b_col5['p_value']:.4f}, N={b_col5['n_obs']}")

# Baseline: Table O1, Col 6 (Gini variant, drop Korea)
df_nokor = df_full[df_full['ccode'] != 'KOR'].copy()
b_col6 = run_panel_fe(
    df_nokor, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS + OPTIONAL_CONTROLS_GINI,
    spec_id="baseline__tableO1_col6",
    controls_desc="lag_debtgdp, lagchangerealgdp, lag_emu_dum, size_product1, d_dep_ratio_old, lag_changeave_gini_gross (Table O1 Col 6, no Korea)",
    sample_desc="21 OECD countries (excl Korea), 1975-2005",
    extra={"table": "O1", "column": 6, "dropped_country": "KOR"},
)
print(f"  Baseline (Col6): coef={b_col6['coefficient']:.4f}, p={b_col6['p_value']:.4f}, N={b_col6['n_obs']}")


# ============================================================
# STEP 2: DESIGN VARIANT -- First Difference
# ============================================================
print("\n=== Step 2: Design Variants ===")

# First difference on Col5 controls
fd_col5 = run_first_difference(
    df_full, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS + OPTIONAL_CONTROLS_TOP1,
    controls_desc="lag_debtgdp, lagchangerealgdp, lag_emu_dum, size_product1, d_dep_ratio_old, changetop1incomeshare",
    extra={"table": "O1", "column": 5, "note": "first-difference estimator on Col5 controls"},
)
print(f"  FD (Col5 controls): coef={fd_col5['coefficient']:.4f}, p={fd_col5['p_value']:.4f}, N={fd_col5['n_obs']}")


# ============================================================
# STEP 3: RC -- CONTROLS
# ============================================================
print("\n=== Step 3: RC Controls ===")

# rc/controls/progression/col1_to_col5: progressive build-up from col1 to col5
# Col2: add lag_emu_dum
run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR,
    MANDATORY_CONTROLS + ["lag_emu_dum"],
    spec_id="rc/controls/progression/col1_to_col5",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    controls_desc="lag_debtgdp, lagchangerealgdp, lag_emu_dum (Col 2 progression)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/col1_to_col5", "family": "progression",
                "step": 2, "added": ["lag_emu_dum"]},
    extra={"table": "O1", "column": 2},
)

# Col3: add lag_emu_dum + size_product1
run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR,
    MANDATORY_CONTROLS + ["lag_emu_dum", "size_product1"],
    spec_id="rc/controls/progression/col1_to_col5",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    controls_desc="lag_debtgdp, lagchangerealgdp, lag_emu_dum, size_product1 (Col 3 progression)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/col1_to_col5", "family": "progression",
                "step": 3, "added": ["lag_emu_dum", "size_product1"]},
    extra={"table": "O1", "column": 3},
)

# Col4: add lag_emu_dum + size_product1 + d_dep_ratio_old
run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR,
    MANDATORY_CONTROLS + ["lag_emu_dum", "size_product1", "d_dep_ratio_old"],
    spec_id="rc/controls/progression/col1_to_col5",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    controls_desc="lag_debtgdp, lagchangerealgdp, lag_emu_dum, size_product1, d_dep_ratio_old (Col 4 progression)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/col1_to_col5", "family": "progression",
                "step": 4, "added": ["lag_emu_dum", "size_product1", "d_dep_ratio_old"]},
    extra={"table": "O1", "column": 4},
)
print(f"  Progression: 3 steps (col2-col4)")

# LOO from Col5 baseline
for ctrl in OPTIONAL_CONTROLS_TOP1:
    remaining = [c for c in (MANDATORY_CONTROLS + OPTIONAL_CONTROLS_TOP1) if c != ctrl]
    slug = ctrl.replace("changetop1incomeshare", "changetop1incomeshare").lower()
    run_panel_fe(
        df_full, OUTCOME_VAR, TREATMENT_VAR, remaining,
        spec_id=f"rc/controls/loo/drop_{ctrl}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
        controls_desc=f"Col5 minus {ctrl}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{ctrl}", "family": "loo",
                    "dropped": [ctrl], "n_controls": len(remaining)},
    )
print(f"  LOO: 4 specs (drop each optional from Col5)")

# Swap inequality measure: replace changetop1incomeshare with lag_changeave_gini_gross
run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR,
    MANDATORY_CONTROLS + ["lag_emu_dum", "size_product1", "d_dep_ratio_old", "lag_changeave_gini_gross"],
    spec_id="rc/controls/single/swap_inequality_measure",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    controls_desc="Col5 with lag_changeave_gini_gross instead of changetop1incomeshare (full sample incl Korea)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/swap_inequality_measure", "family": "single",
                "swapped_in": "lag_changeave_gini_gross", "swapped_out": "changetop1incomeshare"},
)
print(f"  Swap inequality measure: 1 spec")

# Random control subsets (budget: 25 specs)
rng = np.random.RandomState(112840)
optional_pool = ["lag_emu_dum", "size_product1", "d_dep_ratio_old", "changetop1incomeshare"]
# Note: lag_changeave_gini_gross is mutually exclusive with changetop1incomeshare.
# We enumerate subsets of the optional pool (excluding gini since it's a swap).
# Pool size 4 -> 2^4 - 1 = 15 non-empty subsets. Some are already covered by progression/LOO.
# Generate all subsets, then add some with gini swapped in.

all_subsets = []
for size in range(1, len(optional_pool) + 1):
    for combo in itertools.combinations(optional_pool, size):
        all_subsets.append(list(combo))

# Add gini variants (replace changetop1incomeshare with lag_changeave_gini_gross where applicable)
gini_subsets = []
for sub in all_subsets:
    if "changetop1incomeshare" in sub:
        new_sub = [c if c != "changetop1incomeshare" else "lag_changeave_gini_gross" for c in sub]
        gini_subsets.append(new_sub)

all_subsets_combined = all_subsets + gini_subsets
# Remove duplicates of already-run specs (progression cols and LOO variants)
# Shuffle and pick up to 25
rng.shuffle(all_subsets_combined)

random_subset_count = 0
for i, sub in enumerate(all_subsets_combined):
    if random_subset_count >= 25:
        break
    ctrls = MANDATORY_CONTROLS + sub
    run_panel_fe(
        df_full, OUTCOME_VAR, TREATMENT_VAR, ctrls,
        spec_id=f"rc/controls/subset/random_{random_subset_count + 1:02d}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#subset",
        controls_desc=f"mandatory + {sub}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/random_{random_subset_count + 1:02d}",
                    "family": "subset", "included_optional": sub,
                    "n_controls": len(ctrls)},
        extra={"draw_index": random_subset_count + 1, "seed": 112840},
    )
    random_subset_count += 1
print(f"  Random subsets: {random_subset_count} specs")


# ============================================================
# STEP 4: RC -- SAMPLE
# ============================================================
print("\n=== Step 4: RC Sample ===")

# Helper for Col5 controls (our main baseline for most RC specs)
COL5_CTRLS = MANDATORY_CONTROLS + OPTIONAL_CONTROLS_TOP1

# Trim y at 1/99 percentiles
p1, p99 = df_full[OUTCOME_VAR].quantile([0.01, 0.99])
df_trim199 = df_full[(df_full[OUTCOME_VAR] >= p1) & (df_full[OUTCOME_VAR] <= p99)].copy()
run_panel_fe(
    df_trim199, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/outliers/trim_y_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    sample_desc="Trimmed y at [1,99] pctiles",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "family": "outliers",
                "trim_pctiles": [1, 99], "var_trimmed": OUTCOME_VAR},
)
print(f"  Trim y [1,99]: 1 spec")

# Trim y at 5/95 percentiles
p5, p95 = df_full[OUTCOME_VAR].quantile([0.05, 0.95])
df_trim595 = df_full[(df_full[OUTCOME_VAR] >= p5) & (df_full[OUTCOME_VAR] <= p95)].copy()
run_panel_fe(
    df_trim595, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    sample_desc="Trimmed y at [5,95] pctiles",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "family": "outliers",
                "trim_pctiles": [5, 95], "var_trimmed": OUTCOME_VAR},
)
print(f"  Trim y [5,95]: 1 spec")

# Cook's D outlier removal
try:
    import statsmodels.api as sm
    all_vars_col5 = [OUTCOME_VAR, TREATMENT_VAR] + COL5_CTRLS + ["cn"]
    dfreg_cd = df_full.dropna(subset=all_vars_col5).copy()
    # Create entity dummies for Cook's D calculation
    dummies = pd.get_dummies(dfreg_cd['cn'], prefix='fe', drop_first=True).astype(float)
    X_cd = dfreg_cd[[TREATMENT_VAR] + COL5_CTRLS].join(dummies)
    X_cd = sm.add_constant(X_cd)
    y_cd = dfreg_cd[OUTCOME_VAR]
    ols_cd = sm.OLS(y_cd, X_cd).fit()
    influence = ols_cd.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4.0 / len(y_cd)
    keep_mask = cooks_d < threshold
    df_cooksd = dfreg_cd.loc[keep_mask].copy()
    n_dropped = (~keep_mask).sum()

    run_panel_fe(
        df_cooksd, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
        spec_id="rc/sample/outliers/cooksd_4_over_n",
        spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
        sample_desc=f"Cook's D < 4/N (dropped {n_dropped} obs)",
        controls_desc="Col5 controls",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/outliers/cooksd_4_over_n", "family": "outliers",
                    "method": "cooksd", "threshold": "4/N", "n_dropped": int(n_dropped)},
    )
    print(f"  Cook's D: 1 spec (dropped {n_dropped} obs)")
except Exception as e:
    print(f"  Cook's D failed: {e}")
    # Record failure
    srid = next_spec_run_id()
    payload = make_failure_payload(
        error=str(e),
        error_details=error_details_from_exception(e, stage="cooksd_outlier"),
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
    )
    spec_results.append({
        "paper_id": PAPER_ID, "spec_run_id": srid,
        "spec_id": "rc/sample/outliers/cooksd_4_over_n",
        "spec_tree_path": "specification_tree/modules/robustness/sample.md#outliers",
        "baseline_group_id": "G1",
        "outcome_var": OUTCOME_VAR, "treatment_var": TREATMENT_VAR,
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 0, "run_error": str(e)[:240],
        "sample_desc": "", "fixed_effects": "country FE", "controls_desc": "", "cluster_var": "",
    })

# Drop Korea
run_panel_fe(
    df_nokor, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/subset/drop_korea",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    sample_desc="21 OECD countries (excl Korea)",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_korea", "family": "subset",
                "dropped_entities": ["KOR"]},
)
print(f"  Drop Korea: 1 spec")

# Drop Greece
df_nogrc = df_full[df_full['ccode'] != 'GRC'].copy()
run_panel_fe(
    df_nogrc, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/subset/drop_greece",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    sample_desc="21 OECD countries (excl Greece)",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_greece", "family": "subset",
                "dropped_entities": ["GRC"]},
)
print(f"  Drop Greece: 1 spec")

# Drop small countries (bottom quartile GDP)
avg_gdp = df_full.groupby('ccode')['gdptotal'].mean()
q25 = avg_gdp.quantile(0.25)
small_countries = avg_gdp[avg_gdp < q25].index.tolist()
df_nosmall = df_full[~df_full['ccode'].isin(small_countries)].copy()
run_panel_fe(
    df_nosmall, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/subset/drop_small_countries",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    sample_desc=f"Excl small countries (GDP < Q25): {small_countries}",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_small_countries", "family": "subset",
                "dropped_entities": small_countries, "criterion": "avg GDP < Q25"},
)
print(f"  Drop small countries: 1 spec (dropped {small_countries})")

# Period: 1980-2005
df_1980 = df_full[(df_full['year'] >= 1980)].copy()
run_panel_fe(
    df_1980, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/period/1980_2005",
    spec_tree_path="specification_tree/modules/robustness/sample.md#period",
    sample_desc="22 OECD countries, 1980-2005",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/1980_2005", "family": "period",
                "start_year": 1980, "end_year": 2005},
)
print(f"  Period 1980-2005: 1 spec")

# Period: 1975-2000
df_2000 = df_full[(df_full['year'] <= 2000)].copy()
run_panel_fe(
    df_2000, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/sample/period/1975_2000",
    spec_tree_path="specification_tree/modules/robustness/sample.md#period",
    sample_desc="22 OECD countries, 1975-2000",
    controls_desc="Col5 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/1975_2000", "family": "period",
                "start_year": 1975, "end_year": 2000},
)
print(f"  Period 1975-2000: 1 spec")


# ============================================================
# STEP 5: RC -- FUNCTIONAL FORM / TREATMENT
# ============================================================
print("\n=== Step 5: RC Functional Form ===")

# Use Index 2 instead of Index 1
# Table O2: treatment is d_ave_finindex2 (not lagged)
# With Col5-equivalent controls: use sizefin2_product1 instead of size_product1
idx2_controls = ["lag_debtgdp", "lagchangerealgdp", "lag_emu_dum", "sizefin2_product1",
                 "d_dep_ratio_old", "changetop1incomeshare"]
run_panel_fe(
    df_full, OUTCOME_VAR, "d_ave_finindex2", idx2_controls,
    spec_id="rc/form/treatment/index2_instead",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    controls_desc="Col5 controls adapted for Index 2 (sizefin2_product1 replaces size_product1)",
    treatment_var_name="d_ave_finindex2",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/index2_instead",
                "interpretation": "Chinn-Ito (2008) index replaces Abiad et al. (2008) index",
                "family": "treatment_definition"},
)
print(f"  Index 2 instead: 1 spec")

# Also run Index 2 with minimal controls (Table O2 Col 1)
run_panel_fe(
    df_full, OUTCOME_VAR, "d_ave_finindex2", MANDATORY_CONTROLS,
    spec_id="rc/form/treatment/index2_instead",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    controls_desc="lag_debtgdp, lagchangerealgdp (Index 2, minimal)",
    treatment_var_name="d_ave_finindex2",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/index2_instead",
                "interpretation": "Chinn-Ito index, minimal controls",
                "family": "treatment_definition"},
    extra={"table": "O2", "column": 1},
)

# Level change in debt instead of log change
# Construct level change: realdebt - lag(realdebt)
df_level = df_full.copy()
df_level['level_change_debt'] = df_level.groupby('ccode')['realdebt'].diff()
run_panel_fe(
    df_level, "level_change_debt", TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/form/outcome/level_change_debt",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    controls_desc="Col5 controls",
    outcome_var_name="level_change_debt",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/level_change_debt",
                "interpretation": "Level change in real debt instead of log change",
                "family": "outcome_definition"},
)
print(f"  Level change debt: 1 spec")


# ============================================================
# STEP 6: RC -- FIXED EFFECTS
# ============================================================
print("\n=== Step 6: RC Fixed Effects ===")

# Add year FE (country + year two-way FE)
run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    fe_var="cn + year_int",
    spec_id="rc/fe/add_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md",
    fixed_effects_desc="country FE + year FE",
    controls_desc="Col5 controls",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_year", "added": ["year"], "total_fe": ["country", "year"]},
)
print(f"  Add year FE: 1 spec")

# Add region FE (instead of country FE -- less restrictive)
# Use region as FE; also encode as numeric
df_full['region_code'] = df_full['region'].astype('category').cat.codes
run_panel_fe(
    df_full, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    fe_var="region_code",
    spec_id="rc/fe/add_region",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md",
    fixed_effects_desc="region FE (instead of country FE)",
    controls_desc="Col5 controls",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_region", "replaced": ["country"], "with": ["region"],
                "note": "5 regions: Oceania, Asia, NorthAm, NorthEurope, WestEurope, SouthEurope"},
)
print(f"  Region FE: 1 spec")


# ============================================================
# STEP 7: RC -- PREPROCESSING / DATA CONSTRUCTION
# ============================================================
print("\n=== Step 7: RC Preprocessing ===")

# Unweighted average index (instead of GDP-weighted)
# Recompute ave_finindex1 as simple mean across countries within year
df_unwt = df_full.copy()
df_unwt['ave_finindex1_unwt'] = df_unwt.groupby('year')['finindex1'].transform('mean')
df_unwt['lag_ave_finindex1_unwt'] = df_unwt.groupby('ccode')['ave_finindex1_unwt'].shift(1)
df_unwt['d_ave_finindex1_unwt'] = df_unwt['ave_finindex1_unwt'] - df_unwt['lag_ave_finindex1_unwt']
df_unwt['lag_d_ave_finindex1_unwt'] = df_unwt.groupby('ccode')['d_ave_finindex1_unwt'].shift(1)

run_panel_fe(
    df_unwt, OUTCOME_VAR, "lag_d_ave_finindex1_unwt", COL5_CTRLS,
    spec_id="rc/preprocess/treatment/unweighted_index",
    spec_tree_path="specification_tree/modules/robustness/preprocessing.md",
    controls_desc="Col5 controls",
    treatment_var_name="lag_d_ave_finindex1_unwt",
    axis_block_name="preprocess",
    axis_block={"spec_id": "rc/preprocess/treatment/unweighted_index",
                "description": "Unweighted average of finindex1 across countries instead of GDP-weighted"},
)
print(f"  Unweighted index: 1 spec")

# Alternative debt measure: use debtgdp directly (ratio) instead of log(debtgdp * realgdp)
df_altdebt = df_full.copy()
df_altdebt['change_debtgdp'] = df_altdebt.groupby('ccode')['debtgdp'].diff()
run_panel_fe(
    df_altdebt, "change_debtgdp", TREATMENT_VAR, COL5_CTRLS,
    spec_id="rc/data/construction/alternative_debt_measure",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    controls_desc="Col5 controls",
    outcome_var_name="change_debtgdp",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/construction/alternative_debt_measure",
                "description": "Change in debt/GDP ratio instead of change in log real debt"},
)
print(f"  Alternative debt measure: 1 spec")


# ============================================================
# STEP 8: RC -- JOINT VARIATIONS
# ============================================================
print("\n=== Step 8: RC Joint ===")

# Joint: Index 2 x control subsets
joint_count = 0

# Index 2 + Col2 controls
run_panel_fe(
    df_full, OUTCOME_VAR, "d_ave_finindex2",
    MANDATORY_CONTROLS + ["lag_emu_dum"],
    spec_id="rc/joint/index_controls/idx2_col2",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    controls_desc="mandatory + lag_emu_dum (Index 2)",
    treatment_var_name="d_ave_finindex2",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/index_controls/idx2_col2",
                "axes": ["treatment_index", "controls"],
                "treatment": "d_ave_finindex2", "controls_step": 2},
)
joint_count += 1

# Index 2 + Col3 controls
run_panel_fe(
    df_full, OUTCOME_VAR, "d_ave_finindex2",
    MANDATORY_CONTROLS + ["lag_emu_dum", "sizefin2_product1"],
    spec_id="rc/joint/index_controls/idx2_col3",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    controls_desc="mandatory + lag_emu_dum + sizefin2_product1 (Index 2)",
    treatment_var_name="d_ave_finindex2",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/index_controls/idx2_col3",
                "axes": ["treatment_index", "controls"],
                "treatment": "d_ave_finindex2", "controls_step": 3},
)
joint_count += 1

# Index 2 + Col4 controls
run_panel_fe(
    df_full, OUTCOME_VAR, "d_ave_finindex2",
    MANDATORY_CONTROLS + ["lag_emu_dum", "sizefin2_product1", "d_dep_ratio_old"],
    spec_id="rc/joint/index_controls/idx2_col4",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    controls_desc="mandatory + lag_emu_dum + sizefin2_product1 + d_dep_ratio_old (Index 2)",
    treatment_var_name="d_ave_finindex2",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/index_controls/idx2_col4",
                "axes": ["treatment_index", "controls"],
                "treatment": "d_ave_finindex2", "controls_step": 4},
)
joint_count += 1

# Index 2 + full Col5 (already done as rc/form/treatment/index2_instead)
# Covered above; skip.

# Joint: Sample period x controls
# 1980-2005 with minimal controls
run_panel_fe(
    df_1980, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS,
    spec_id="rc/joint/sample_controls/1980_2005_col1",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    sample_desc="22 OECD countries, 1980-2005",
    controls_desc="mandatory only (1980-2005)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/1980_2005_col1",
                "axes": ["sample_period", "controls"],
                "period": "1980-2005", "controls": "minimal"},
)
joint_count += 1

# 1975-2000 with minimal controls
run_panel_fe(
    df_2000, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS,
    spec_id="rc/joint/sample_controls/1975_2000_col1",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    sample_desc="22 OECD countries, 1975-2000",
    controls_desc="mandatory only (1975-2000)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/1975_2000_col1",
                "axes": ["sample_period", "controls"],
                "period": "1975-2000", "controls": "minimal"},
)
joint_count += 1

# 1980-2005 with Gini controls
run_panel_fe(
    df_1980, OUTCOME_VAR, TREATMENT_VAR,
    MANDATORY_CONTROLS + OPTIONAL_CONTROLS_GINI,
    spec_id="rc/joint/sample_controls/1980_2005_gini",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    sample_desc="22 OECD countries, 1980-2005",
    controls_desc="Col6-type controls (Gini, 1980-2005)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/1980_2005_gini",
                "axes": ["sample_period", "controls"],
                "period": "1980-2005", "controls": "gini_variant"},
)
joint_count += 1

# 1975-2000 with Gini controls
run_panel_fe(
    df_2000, OUTCOME_VAR, TREATMENT_VAR,
    MANDATORY_CONTROLS + OPTIONAL_CONTROLS_GINI,
    spec_id="rc/joint/sample_controls/1975_2000_gini",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    sample_desc="22 OECD countries, 1975-2000",
    controls_desc="Col6-type controls (Gini, 1975-2000)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/1975_2000_gini",
                "axes": ["sample_period", "controls"],
                "period": "1975-2000", "controls": "gini_variant"},
)
joint_count += 1

# Drop Greece + minimal
run_panel_fe(
    df_nogrc, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS,
    spec_id="rc/joint/sample_controls/nogrc_col1",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    sample_desc="21 OECD countries (excl Greece)",
    controls_desc="mandatory only (no Greece)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/nogrc_col1",
                "axes": ["sample_subset", "controls"],
                "dropped": ["GRC"], "controls": "minimal"},
)
joint_count += 1

# Drop Korea + year FE
run_panel_fe(
    df_nokor, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS,
    fe_var="cn + year_int",
    spec_id="rc/joint/sample_controls/nokor_yearfe",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    sample_desc="21 OECD countries (excl Korea)",
    fixed_effects_desc="country FE + year FE",
    controls_desc="Col5 controls (no Korea, year FE)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/nokor_yearfe",
                "axes": ["sample_subset", "fixed_effects"],
                "dropped": ["KOR"], "fe": ["country", "year"]},
)
joint_count += 1

print(f"  Joint variations: {joint_count} specs")


# ============================================================
# STEP 9: INFERENCE VARIANTS
# ============================================================
print("\n=== Step 9: Inference Variants ===")

# Recompute inference on baseline (Col5) under three alternative SE choices
baseline_for_infer = b_col5

# Cluster by country
run_inference_variant(
    baseline_for_infer, df_full, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS, "cn",
    infer_spec_id="infer/se/cluster/country",
    infer_vcov={"CRV1": "cn"},
    infer_params={"cluster_var": "cn"},
)
print(f"  Cluster by country: 1 infer spec")

# HC3
run_inference_variant(
    baseline_for_infer, df_full, OUTCOME_VAR, TREATMENT_VAR, COL5_CTRLS, "cn",
    infer_spec_id="infer/se/hc/hc3",
    infer_vcov="HC3",
    infer_params={},
)
print(f"  HC3: 1 infer spec")

# Driscoll-Kraay: use Newey-West-type HAC as approximation
# pyfixest doesn't have DK directly; use statsmodels HAC on demeaned data as approximation
try:
    import statsmodels.api as sm
    all_vars_col5 = [OUTCOME_VAR, TREATMENT_VAR] + COL5_CTRLS + ["cn"]
    dfreg_dk = df_full.dropna(subset=all_vars_col5).copy()
    # Entity-demean for within estimator
    for v in [OUTCOME_VAR, TREATMENT_VAR] + COL5_CTRLS:
        dfreg_dk[f"{v}_dm"] = dfreg_dk[v] - dfreg_dk.groupby('cn')[v].transform('mean')

    dm_rhs = [f"{TREATMENT_VAR}_dm"] + [f"{c}_dm" for c in COL5_CTRLS]
    X_dk = sm.add_constant(dfreg_dk[dm_rhs])
    y_dk = dfreg_dk[f"{OUTCOME_VAR}_dm"]
    m_dk = sm.OLS(y_dk, X_dk).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    focal_dk = f"{TREATMENT_VAR}_dm"
    coef_dk = float(m_dk.params[focal_dk])
    se_dk = float(m_dk.bse[focal_dk])
    pval_dk = float(m_dk.pvalues[focal_dk])
    ci_dk = m_dk.conf_int()
    ci_lo_dk = float(ci_dk.loc[focal_dk, 0])
    ci_hi_dk = float(ci_dk.loc[focal_dk, 1])

    payload_dk = make_success_payload(
        coefficients={k.replace("_dm", ""): float(v) for k, v in m_dk.params.items() if k != "const"},
        inference={"spec_id": "infer/se/panel/driscoll_kraay", "params": {"lags": 3},
                   "notes": "Approximated via HAC on entity-demeaned data (Newey-West, 3 lags)"},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design=make_design_block(),
    )

    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": next_infer_run_id(),
        "spec_run_id": baseline_for_infer["spec_run_id"],
        "spec_id": "infer/se/panel/driscoll_kraay",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md",
        "baseline_group_id": "G1",
        "outcome_var": OUTCOME_VAR,
        "treatment_var": TREATMENT_VAR,
        "coefficient": coef_dk,
        "std_error": se_dk,
        "p_value": pval_dk,
        "ci_lower": ci_lo_dk,
        "ci_upper": ci_hi_dk,
        "n_obs": int(m_dk.nobs),
        "r_squared": float(m_dk.rsquared),
        "coefficient_vector_json": json.dumps(payload_dk),
        "run_success": 1,
        "run_error": "",
    })
    print(f"  Driscoll-Kraay (HAC approx): 1 infer spec")
except Exception as e:
    print(f"  Driscoll-Kraay failed: {e}")
    payload_dk = make_failure_payload(
        error=str(e),
        error_details=error_details_from_exception(e, stage="driscoll_kraay"),
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
    )
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": next_infer_run_id(),
        "spec_run_id": baseline_for_infer["spec_run_id"],
        "spec_id": "infer/se/panel/driscoll_kraay",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md",
        "baseline_group_id": "G1",
        "outcome_var": OUTCOME_VAR, "treatment_var": TREATMENT_VAR,
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload_dk),
        "run_success": 0, "run_error": str(e)[:240],
    })

# Also run inference variants on Col1 baseline
# Cluster by country on Col1
run_inference_variant(
    b_col1, df_full, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS, "cn",
    infer_spec_id="infer/se/cluster/country",
    infer_vcov={"CRV1": "cn"},
    infer_params={"cluster_var": "cn"},
)

# HC3 on Col1
run_inference_variant(
    b_col1, df_full, OUTCOME_VAR, TREATMENT_VAR, MANDATORY_CONTROLS, "cn",
    infer_spec_id="infer/se/hc/hc3",
    infer_vcov="HC3",
    infer_params={},
)
print(f"  Additional Col1 inference: 2 infer specs")


# ============================================================
# STEP 10: WRITE OUTPUTS
# ============================================================
print(f"\n=== Writing Outputs ===")

# Specification results
df_specs = pd.DataFrame(spec_results)
print(f"  Total specification rows: {len(df_specs)}")
print(f"  Successful: {df_specs['run_success'].sum()}")
print(f"  Failed: {(df_specs['run_success'] == 0).sum()}")

df_specs.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  Written: {OUTPUT_DIR}/specification_results.csv")

# Inference results
df_infer = pd.DataFrame(inference_results)
print(f"  Total inference rows: {len(df_infer)}")
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  Written: {OUTPUT_DIR}/inference_results.csv")

# Summary statistics
n_baseline = df_specs[df_specs['spec_id'].str.startswith('baseline')].shape[0]
n_design = df_specs[df_specs['spec_id'].str.startswith('design/')].shape[0]
n_rc = df_specs[df_specs['spec_id'].str.startswith('rc/')].shape[0]
n_total = len(df_specs)

print(f"\n  Summary: {n_baseline} baseline + {n_design} design + {n_rc} rc = {n_total} total specs")
print(f"  Inference variants: {len(df_infer)}")

# Coefficient summary for baseline
for _, r in df_specs[df_specs['spec_id'].str.startswith('baseline')].iterrows():
    if r['run_success'] == 1:
        print(f"  {r['spec_id']}: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

# ============================================================
# WRITE SPECIFICATION_SEARCH.md
# ============================================================

search_md = f"""# Specification Search: 112840-V1

## Paper
Kumhof, Ranciere & Winant (2015), "Inequality, Leverage, and Crises", AER 105(3).

## Surface Summary
- **Paper ID**: 112840-V1
- **Baseline groups**: 1 (G1: Financial liberalization and public debt growth)
- **Design**: Panel fixed effects (within estimator), country FE, HC1 robust SE
- **Data**: 22 OECD countries, 1975-2005 annual panel
- **Surface hash**: {SURFACE_HASH}
- **Sampling seed**: 112840

### Baseline Specifications
- Table O1, Col 1 (minimal): changerealdebt ~ lag_d_ave_finindex1 + lag_debtgdp + lagchangerealgdp | cn
- Table O1, Col 5 (extended): adds lag_emu_dum, size_product1, d_dep_ratio_old, changetop1incomeshare
- Table O1, Col 6 (Gini variant): replaces changetop1incomeshare with lag_changeave_gini_gross, drops Korea

### Budgets
- Max core specs: 70
- Max control subsets: 25
- Seed: 112840

## Execution Summary

### Counts
- **Total specification rows**: {n_total}
  - Baseline: {n_baseline}
  - Design variants: {n_design}
  - RC variants: {n_rc}
- **Inference variant rows**: {len(df_infer)}
- **Total (specs + inference)**: {n_total + len(df_infer)}
- **Successful**: {int(df_specs['run_success'].sum())} / {n_total} specs, {int(df_infer['run_success'].sum())} / {len(df_infer)} inference
- **Failed**: {int((df_specs['run_success'] == 0).sum())} specs, {int((df_infer['run_success'] == 0).sum())} inference

### RC Axes Executed
1. **Controls progression** (Col 2-4): 3 specs
2. **Controls LOO** (drop each optional from Col5): 4 specs
3. **Controls swap** (inequality measure): 1 spec
4. **Controls random subsets**: {random_subset_count} specs (seed=112840)
5. **Sample outliers** (trim y [1,99], [5,95], Cook's D): 3 specs
6. **Sample country exclusions** (Korea, Greece, small countries): 3 specs
7. **Sample period** (1980-2005, 1975-2000): 2 specs
8. **Functional form / treatment** (Index 2, level debt change): 3 specs
9. **Fixed effects** (add year FE, region FE): 2 specs
10. **Preprocessing** (unweighted index, alternative debt): 2 specs
11. **Joint variations** (index x controls, sample x controls): {joint_count} specs

### Inference Variants
1. **Cluster by country** (CRV1, cn): on Col5 and Col1 baselines
2. **HC3**: on Col5 and Col1 baselines
3. **Driscoll-Kraay** (HAC approximation, 3 lags): on Col5 baseline

### Deviations / Notes
- Driscoll-Kraay SE approximated via HAC (Newey-West, 3 lags) on entity-demeaned data, since pyfixest does not natively support DK SE.
- The paper's structural Fortran model (calibration) is excluded per the surface.
- Table O3 (interest rate elasticity) is excluded per the surface.
- The size interaction variable (size_product1) uses the GDP-weighted index; for Index 2, the equivalent sizefin2_product1 is used.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest (panel FE estimation)
- pandas, numpy (data manipulation)
- statsmodels (Cook's D diagnostics, HAC SE)
- scipy (statistical tests)

## Data Construction
All variables reconstructed from source data files following the Stata do file exactly:
- gini_gross.csv + subset.dta + DepRatioOld.dta + incomeinequality.dta
- GDP-weighted average financial liberalization indexes
- First differences of logs for debt and GDP
- EMU dummy, size-finlib interaction, dependency ratio change
- Sample restricted to 22 OECD countries, 1975-2005

## Key Results
"""

# Add baseline results summary
for _, r in df_specs[df_specs['spec_id'].str.startswith('baseline')].iterrows():
    if r['run_success'] == 1:
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        search_md += f"- **{r['spec_id']}**: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}, p={r['p_value']:.4f}{sig}, N={int(r['n_obs'])}\n"

# Support assessment
successful = df_specs[df_specs['run_success'] == 1]
if len(successful) > 0:
    sig_005 = (successful['p_value'] < 0.05).sum()
    sig_010 = (successful['p_value'] < 0.10).sum()
    same_sign = (successful['coefficient'] > 0).sum() if successful['coefficient'].median() > 0 else (successful['coefficient'] < 0).sum()
    search_md += f"""
## Robustness Assessment
- Specifications with p < 0.05: {sig_005}/{len(successful)} ({100*sig_005/len(successful):.0f}%)
- Specifications with p < 0.10: {sig_010}/{len(successful)} ({100*sig_010/len(successful):.0f}%)
- Specifications with same sign as baseline: {same_sign}/{len(successful)} ({100*same_sign/len(successful):.0f}%)
- Median coefficient: {successful['coefficient'].median():.4f}
- Mean coefficient: {successful['coefficient'].mean():.4f}
- Coefficient range: [{successful['coefficient'].min():.4f}, {successful['coefficient'].max():.4f}]
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)
print(f"  Written: {OUTPUT_DIR}/SPECIFICATION_SEARCH.md")

print("\n=== Done ===")
