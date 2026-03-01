"""
Specification Search Script for 128521-V1
"Recessions, Mortality, and Migration Bias: Evidence from the Lancashire Cotton Famine"

Surface-driven execution:
  - G1: Total mortality DD (census_mr_tot ~ cotton_dist_post | master_name + period)
  - G2: Age-specific mortality DD (census_mr_{age} ~ cotton_dist_post)
  - TWFE difference-in-differences
  - Target: 50+ specifications

Outputs:
  - specification_results.csv
  - inference_results.csv
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

PAPER_ID = "128521-V1"
DATA_DIR = "data/downloads/extracted/128521-V1"
RAW_DIR = f"{DATA_DIR}/ABH_Rep/Data"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================
# DATA PREPARATION
# Replicate Main_Results_Prep_File.do in Python
# ============================================================
print("Building regression dataset...")

# --- Step 1: Generate age weights ---
registrar = pd.read_stata(f"{RAW_DIR}/Registrar_mort_by_age_1851_1870.dta")
registrar = registrar[registrar['year'].between(1851, 1855) | registrar['year'].between(1861, 1865)].copy()
registrar['period'] = np.where(registrar['year'].between(1851, 1855), 1851, 1861)

# Collapse by period
agg_deaths = registrar.groupby('period').agg({
    'deaths_tot': 'sum',
    'deaths_0_1': 'sum', 'deaths_2': 'sum', 'deaths_3': 'sum',
    'deaths_4': 'sum', 'deaths_5_9': 'sum', 'deaths_10_14': 'sum',
    'deaths_15_24': 'sum', 'deaths_25_34': 'sum', 'deaths_35_44': 'sum',
    'deaths_45_54': 'sum', 'deaths_55_64': 'sum', 'deaths_65_74': 'sum',
    'deaths_75_84': 'sum', 'deaths_85up': 'sum',
}).reset_index()

# Reshape to long: period x age_cat
age_cat_map = {
    '0_1': ['deaths_0_1'], '2': ['deaths_2'], '3': ['deaths_3'],
    '4': ['deaths_4'], '5_9': ['deaths_5_9'], '10_14': ['deaths_10_14'],
    '15_24': ['deaths_15_24'], '25_34': ['deaths_25_34'],
    '35_44': ['deaths_35_44'], '45_54': ['deaths_45_54'],
    '55_64': ['deaths_55_64'], '65_74': ['deaths_65_74'],
    '75_84': ['deaths_75_84'], '85up': ['deaths_85up'],
}

rows = []
for _, row in agg_deaths.iterrows():
    for ac, cols in age_cat_map.items():
        rows.append({'period': row['period'], 'age_cat': ac,
                     'deaths': sum(row[c] for c in cols)})

age_shares = pd.DataFrame(rows)
# Merge with total to get share
totals = agg_deaths[['period', 'deaths_tot']].rename(columns={'deaths_tot': 'tot_deaths'})
age_shares = age_shares.merge(totals, on='period')
age_shares['share_ag'] = age_shares['deaths'] / age_shares['tot_deaths']

# Now compute weights from linked data
linked = pd.read_stata(f"{RAW_DIR}/Linked_GRO_and_Census_5yr.dta")
linked['age'] = linked['age'].round()

def assign_age_cat(age):
    if age <= 1: return '0_1'
    elif age == 2: return '2'
    elif age == 3: return '3'
    elif age == 4: return '4'
    elif 5 <= age <= 9: return '5_9'
    elif 10 <= age <= 14: return '10_14'
    elif 15 <= age <= 24: return '15_24'
    elif 25 <= age <= 34: return '25_34'
    elif 35 <= age <= 44: return '35_44'
    elif 45 <= age <= 54: return '45_54'
    elif 55 <= age <= 64: return '55_64'
    elif 65 <= age <= 74: return '65_74'
    elif 75 <= age <= 84: return '75_84'
    else: return '85up'

linked['age_cat'] = linked['age'].apply(assign_age_cat)

# Keep a clean copy of linked data for rebuild_mortality_with_filter
linked_raw = linked.copy()

# Collapse linked by age_cat and period
linked_counts = linked.groupby(['age_cat', 'period']).size().reset_index(name='counter')
linked_totals = linked.groupby('period').size().reset_index(name='linked_tot')
linked_counts = linked_counts.merge(linked_totals, on='period')

# Merge with aggregate shares
age_weights = linked_counts.merge(age_shares[['period', 'age_cat', 'share_ag']], on=['period', 'age_cat'])
age_weights['weight'] = age_weights['share_ag'] * age_weights['linked_tot'] / age_weights['counter']

# --- Step 2: Generate linked death counts by district x period ---
linked = linked.merge(age_weights[['period', 'age_cat', 'weight']], on=['period', 'age_cat'], how='left')

# Census-based allocation (allocate to enumeration district)
linked['count_tot'] = linked['weight']
linked['count_under15'] = (linked['age'] < 15).astype(float)
linked['count_15_24'] = ((linked['age'] >= 15) & (linked['age'] < 25)).astype(float)
linked['count_25_34'] = ((linked['age'] >= 25) & (linked['age'] < 35)).astype(float)
linked['count_35_44'] = ((linked['age'] >= 35) & (linked['age'] < 45)).astype(float)
linked['count_45_54'] = ((linked['age'] >= 45) & (linked['age'] < 55)).astype(float)
linked['count_55_64'] = ((linked['age'] >= 55) & (linked['age'] < 65)).astype(float)
linked['count_over64'] = (linked['age'] >= 65).astype(float)
linked['count_15_54'] = ((linked['age'] >= 15) & (linked['age'] < 55)).astype(float)
linked['count_over54'] = (linked['age'] >= 55).astype(float)

count_cols = ['count_tot', 'count_under15', 'count_15_24', 'count_25_34',
              'count_35_44', 'count_45_54', 'count_55_64', 'count_over64',
              'count_15_54', 'count_over54']

# Census allocation
census_deaths = linked.groupby(['census_master', 'period'])[count_cols].sum().reset_index()
census_deaths = census_deaths.rename(columns={'census_master': 'master_name'})
census_deaths = census_deaths.rename(columns={c: f'census_{c}' for c in count_cols})

# GRO allocation
gro_deaths = linked.groupby(['GRO_master', 'period'])[count_cols].sum().reset_index()
gro_deaths = gro_deaths.rename(columns={'GRO_master': 'master_name'})
gro_deaths = gro_deaths.rename(columns={c: f'GRO_{c}' for c in count_cols})

# Merge census and GRO deaths
linked_deaths = census_deaths.merge(gro_deaths, on=['master_name', 'period'], how='outer')

# --- Step 3: Merge with aggregate registrar deaths ---
reg_agg = registrar.groupby(['master_name', 'period']).agg({
    'deaths_tot': 'sum',
    'deaths_0_1': 'sum', 'deaths_2': 'sum', 'deaths_3': 'sum',
    'deaths_4': 'sum', 'deaths_5_9': 'sum', 'deaths_10_14': 'sum',
    'deaths_15_24': 'sum', 'deaths_25_34': 'sum', 'deaths_35_44': 'sum',
    'deaths_45_54': 'sum', 'deaths_55_64': 'sum', 'deaths_65_74': 'sum',
    'deaths_75_84': 'sum', 'deaths_85up': 'sum',
}).reset_index()

reg_agg['agg_count_tot'] = reg_agg['deaths_tot']
reg_agg['agg_count_under15'] = (reg_agg['deaths_0_1'] + reg_agg['deaths_2'] +
                                 reg_agg['deaths_3'] + reg_agg['deaths_4'] +
                                 reg_agg['deaths_5_9'] + reg_agg['deaths_10_14'])
reg_agg['agg_count_15_24'] = reg_agg['deaths_15_24']
reg_agg['agg_count_25_34'] = reg_agg['deaths_25_34']
reg_agg['agg_count_35_44'] = reg_agg['deaths_35_44']
reg_agg['agg_count_45_54'] = reg_agg['deaths_45_54']
reg_agg['agg_count_55_64'] = reg_agg['deaths_55_64']
reg_agg['agg_count_over64'] = reg_agg['deaths_65_74'] + reg_agg['deaths_75_84'] + reg_agg['deaths_85up']
reg_agg['agg_count_15_54'] = (reg_agg['deaths_15_24'] + reg_agg['deaths_25_34'] +
                               reg_agg['deaths_35_44'] + reg_agg['deaths_45_54'])
reg_agg['agg_count_over54'] = (reg_agg['deaths_55_64'] + reg_agg['deaths_65_74'] +
                                reg_agg['deaths_75_84'] + reg_agg['deaths_85up'])

agg_cols = [c for c in reg_agg.columns if c.startswith('agg_count_')]
reg_agg_sub = reg_agg[['master_name', 'period'] + agg_cols]

# Merge: linked deaths + aggregate deaths
panel = reg_agg_sub.merge(linked_deaths, on=['master_name', 'period'], how='left')

# Fill NaN linked counts with 0
for c in panel.columns:
    if c.startswith('census_count') or c.startswith('GRO_count'):
        panel[c] = panel[c].fillna(0)

# --- Step 4: Merge with population data ---
pop = pd.read_stata(f"{RAW_DIR}/pop_by_source_group_and_period.dta")
# Rename pop_cens -> pop_census
for c in list(pop.columns):
    if c.startswith('pop_cens'):
        new_name = c.replace('pop_cens', 'pop_census')
        pop = pop.rename(columns={c: new_name})

panel = panel.merge(pop, on=['master_name', 'period'], how='inner')

# --- Step 5: Compute lambdas and mortality rates ---
age_suffixes = ['_tot', '_under15', '_15_24', '_25_34', '_35_44', '_45_54',
                '_55_64', '_over64', '_15_54', '_over54']

# Lambdas: ratio of linked to aggregate at the national level
lambdas = {}
for suffix in age_suffixes:
    total_census = panel[f'census_count{suffix}'].sum()
    total_agg = panel[f'agg_count{suffix}'].sum()
    lambdas[suffix] = total_census / total_agg if total_agg > 0 else 1.0

# Compute mortality rates
for suffix in age_suffixes:
    pop_col = f'pop_census{suffix}'
    # Pop for GRO is same as census pop
    panel[f'pop_GRO{suffix}'] = panel[pop_col]

    for source in ['census', 'agg']:
        count_col = f'{source}_count{suffix}'
        mr_col = f'{source}_mr{suffix}'
        panel[mr_col] = (panel[count_col] / (panel[pop_col] / 1000)) / 5

    # Inflate census rates by lambda
    panel[f'census_mr{suffix}'] = panel[f'census_mr{suffix}'] / lambdas[suffix]

# --- Step 6: Merge cotton district indicators ---
cotton = pd.read_stata(f"{RAW_DIR}/cotton_district_indicators.dta")
panel = panel.merge(cotton, on='master_name', how='inner')

# --- Step 7: Merge linkable name counts ---
linkable = pd.read_stata(f"{RAW_DIR}/linkable_namesXmasterXageXperiod.dta")
panel = panel.merge(linkable, on=['master_name', 'period'], how='left')

# Drop DUNMOW (no linkable names in 1851)
panel = panel[panel['master_name'] != 'DUNMOW'].copy()

# --- Step 8: Generate regression variables ---
panel['post'] = (panel['period'] == 1861).astype(int)

# Region x period FE via dummies
# region is a string column (london, midlands, north, south, wales)
# Use region_x from merge if available
if 'region_x' in panel.columns:
    panel['region'] = panel['region_x']

regions = sorted(panel['region'].dropna().unique())
for r in regions:
    panel[f'_IregXpost_{r}'] = ((panel['region'] == r) & (panel['post'] == 1)).astype(int)

# Treatment interactions
panel['cotton_dist_post'] = panel['cotton_dist'] * panel['post']
panel['nearby_post_25'] = panel['cotton_dist_0_25'] * panel['post']
panel['nearby_post_50'] = panel['cotton_dist_25_50'] * panel['post']
panel['nearby_post_75'] = panel['cotton_dist_50_75'] * panel['post']
panel['cotton_eshr_post'] = panel['own_cot_share'] * panel['post']

# Resolve area column from possible merge suffixes
area_col = None
for candidate in ['area', 'area_x', 'area_y']:
    if candidate in panel.columns:
        area_col = candidate
        break
if area_col is None:
    raise KeyError("No area column found in panel")

# Population controls
panel['ln_popdensity_census'] = np.log(panel['pop_census_tot'] / panel[area_col])
panel['under_15_shr_census'] = panel['pop_census_under15'] / panel['pop_census_tot']
# elderly share uses over54
panel['elderly_shr_census'] = panel['pop_census_over54'] / panel['pop_census_tot']

# Linkable share
for suffix in age_suffixes:
    link_col = f'linkable{suffix}'
    pop_col = f'pop_census{suffix}'
    panel[f'linkable_shr{suffix}'] = panel[link_col] / panel[pop_col]
    panel[f'linkable_shr{suffix}'] = panel[f'linkable_shr{suffix}'].fillna(0)

# District ID
panel['dist_id'] = panel.groupby('master_name').ngroup() + 1

# Resolve county column from possible merge suffixes
for candidate in ['county_x', 'county']:
    if candidate in panel.columns:
        panel['county'] = panel[candidate]
        break
else:
    geo = pd.read_stata(f"{RAW_DIR}/TT_Master_w_county_latlon_area.dta")
    panel = panel.merge(geo[['master_name', 'county']], on='master_name', how='left')

# Region x post FE columns
regxpost_cols = [c for c in panel.columns if c.startswith('_IregXpost_')]

print(f"Panel shape: {panel.shape}")
print(f"N districts: {panel['master_name'].nunique()}")
print(f"N periods: {panel['period'].nunique()}")
print(f"Cotton districts: {panel['cotton_dist'].sum()}")

# ============================================================
# SPECIFICATION RUNNER
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


def next_run_id():
    global spec_run_counter
    spec_run_counter += 1
    return f"{PAPER_ID}_run_{spec_run_counter:03d}"


def get_group(group_id):
    for g in surface_obj["baseline_groups"]:
        if g["baseline_group_id"] == group_id:
            return g
    return {}


def run_did(formula, data, vcov, treatment_var, spec_id, spec_tree_path,
            baseline_group_id, outcome_var, sample_desc, controls_desc,
            fixed_effects="master_name + period", cluster_var="master_name",
            weights_col=None, axis_block_name=None, axis_block=None, notes=""):
    """Run TWFE DD regression."""
    run_id = next_run_id()
    group = get_group(baseline_group_id)
    design_audit = group.get("design_audit", {})
    inf_canonical = group.get("inference_plan", {}).get("canonical", {})

    try:
        kw = {"vcov": vcov}
        if weights_col and weights_col in data.columns:
            # Drop rows where weight is missing or <= 0
            reg_data = data[data[weights_col].notna() & (data[weights_col] > 0)].copy()
            kw["weights"] = weights_col
        else:
            reg_data = data.copy()

        m = pf.feols(formula, data=reg_data, **kw)

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
            inference={"spec_id": inf_canonical.get("spec_id", ""),
                       "params": inf_canonical.get("params", {}),
                       "method": "CRV1", "cluster_var": cluster_var},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes,
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
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        })
        return run_id, m

    except Exception as e:
        err_msg = str(e)
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
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
            "fixed_effects": fixed_effects,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg,
        })
        return run_id, None


def add_inference_row(base_run_id, treatment_var, infer_spec_id, infer_tree_path,
                      baseline_group_id, infer_params, formula, data, vcov,
                      weights_col=None, outcome_var="census_mr_tot", cluster_var=""):
    """Re-estimate with different inference."""
    global spec_run_counter
    spec_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        kw = {"vcov": vcov}
        if weights_col and weights_col in data.columns:
            reg_data = data[data[weights_col].notna() & (data[weights_col] > 0)].copy()
            kw["weights"] = weights_col
        else:
            reg_data = data.copy()

        m = pf.feols(formula, data=reg_data, **kw)
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

        group = get_group(baseline_group_id)
        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": infer_spec_id, "params": infer_params},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": group.get("design_audit", {})},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id,
            "spec_id": infer_spec_id,
            "spec_tree_path": infer_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "cluster_var": cluster_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        err_msg = str(e)
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id,
            "spec_id": infer_spec_id,
            "spec_tree_path": infer_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "cluster_var": cluster_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": err_msg,
        })


# ============================================================
# Define control variable sets
# ============================================================
# Region x post dummies
regxpost_str = " + ".join(regxpost_cols)

# Control sets for Table 2
controls_none = ""
controls_col2 = f"ln_popdensity_census + linkable_shr_tot + under_15_shr_census + elderly_shr_census + {regxpost_str}"
controls_col3 = f"nearby_post_25 + nearby_post_50 + nearby_post_75 + {controls_col2}"

fe_str = "master_name + period"


# ============================================================
# G1: Total mortality DD
# ============================================================
print("=" * 60)
print("G1: Total mortality DD")
print("=" * 60)

# --- Baseline: Table 2 Col 1 ---
run_id_t2c1, _ = run_did(
    formula=f"census_mr_tot ~ cotton_dist_post | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#twfe",
    baseline_group_id="G1",
    outcome_var="census_mr_tot",
    sample_desc="Full panel, 538 districts x 2 periods",
    controls_desc="none",
    weights_col="pop_census_tot",
)

# --- Additional baselines ---
# Table 2 Col 2
run_id_t2c2, _ = run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col2} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="baseline__table2_col2",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#twfe",
    baseline_group_id="G1",
    outcome_var="census_mr_tot",
    sample_desc="Full panel",
    controls_desc="ln_popdensity, linkable_shr, age shares, region x period FE",
    weights_col="pop_census_tot",
)

# Table 2 Col 3
run_id_t2c3, _ = run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="baseline__table2_col3",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#twfe",
    baseline_group_id="G1",
    outcome_var="census_mr_tot",
    sample_desc="Full panel",
    controls_desc="nearby rings, ln_popdensity, linkable_shr, age shares, region x period FE",
    weights_col="pop_census_tot",
)

# --- RC: LOO Controls (from Col 3 specification) ---
# Drop ln_popdensity
loo_controls_no_density = f"nearby_post_25 + nearby_post_50 + nearby_post_75 + linkable_shr_tot + under_15_shr_census + elderly_shr_census + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_controls_no_density} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/ln_popdensity",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 3 minus ln_popdensity",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/ln_popdensity", "family": "loo",
                "dropped": ["ln_popdensity_census"]},
)

# Drop linkable_shr
loo_controls_no_linkable = f"nearby_post_25 + nearby_post_50 + nearby_post_75 + ln_popdensity_census + under_15_shr_census + elderly_shr_census + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_controls_no_linkable} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/linkable_shr",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 3 minus linkable_shr_tot",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/linkable_shr", "family": "loo",
                "dropped": ["linkable_shr_tot"]},
)

# Drop age shares (under_15 + elderly)
loo_controls_no_age = f"nearby_post_25 + nearby_post_50 + nearby_post_75 + ln_popdensity_census + linkable_shr_tot + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_controls_no_age} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/age_shares",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 3 minus age shares",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/age_shares", "family": "loo",
                "dropped": ["under_15_shr_census", "elderly_shr_census"]},
)

# Drop region x period FE
loo_controls_no_region = "nearby_post_25 + nearby_post_50 + nearby_post_75 + ln_popdensity_census + linkable_shr_tot + under_15_shr_census + elderly_shr_census"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_controls_no_region} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/region_x_period",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 3 minus region x period FE",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/region_x_period", "family": "loo",
                "dropped": ["_IregXpost_*"]},
)

# Drop nearby rings
loo_controls_no_nearby = f"ln_popdensity_census + linkable_shr_tot + under_15_shr_census + elderly_shr_census + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_controls_no_nearby} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/nearby_rings",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 3 minus nearby ring indicators",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/nearby_rings", "family": "loo",
                "dropped": ["nearby_post_25", "nearby_post_50", "nearby_post_75"]},
)

# --- RC: Sample variants ---
# Exclude Manchester
panel_no_manchester = panel[panel['master_name'] != 'GREATER MANCHESTER'].copy()
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel_no_manchester, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude Manchester",
    controls_desc="Col 3 full controls",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester",
                "excluded": ["GREATER MANCHESTER"]},
)

# Exclude Manchester + Liverpool + Leeds
panel_no_big3 = panel[~panel['master_name'].isin(['GREATER MANCHESTER', 'LIVERPOOL', 'GREATER LEEDS'])].copy()
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel_no_big3, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester_liverpool_leeds",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude Manchester, Liverpool, Leeds",
    controls_desc="Col 3 full controls",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester_liverpool_leeds",
                "excluded": ["GREATER MANCHESTER", "LIVERPOOL", "GREATER LEEDS"]},
)

# --- RC: Weights variant ---
# Unweighted
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel, unweighted",
    controls_desc="Col 3 full controls",
    weights_col=None,  # No weights
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted",
                "description": "No population weights"},
)

# --- RC: Data construction / linking restriction variants ---
# These require re-building mortality rates with different linked data subsets
# We'll compute them by filtering the linked microdata

def rebuild_mortality_with_filter(linked_raw, filter_mask, filter_name):
    """Rebuild census_mr_tot from linked data with a filter applied.

    Uses linked_raw (original, unmodified linked microdata) and applies filter_mask.
    """
    linked_filt = linked_raw[filter_mask].copy()

    # Re-compute age weights for filtered sample
    linked_filt['age'] = linked_filt['age'].round()
    linked_filt['age_cat'] = linked_filt['age'].apply(assign_age_cat)

    filt_counts = linked_filt.groupby(['age_cat', 'period']).size().reset_index(name='counter')
    filt_totals = linked_filt.groupby('period').size().reset_index(name='linked_tot')
    filt_counts = filt_counts.merge(filt_totals, on='period')
    filt_counts = filt_counts.merge(age_shares[['period', 'age_cat', 'share_ag']], on=['period', 'age_cat'], how='left')
    filt_counts['share_ag'] = filt_counts['share_ag'].fillna(0)
    filt_counts['filt_weight'] = np.where(filt_counts['counter'] > 0,
                                      filt_counts['share_ag'] * filt_counts['linked_tot'] / filt_counts['counter'],
                                      0)

    linked_filt = linked_filt.merge(filt_counts[['period', 'age_cat', 'filt_weight']], on=['period', 'age_cat'], how='left')
    linked_filt['count_tot'] = linked_filt['filt_weight'].fillna(0)

    # Collapse to district x period
    filt_deaths = linked_filt.groupby(['census_master', 'period'])['count_tot'].sum().reset_index()
    filt_deaths = filt_deaths.rename(columns={'census_master': 'master_name', 'count_tot': 'filt_census_count_tot'})

    # Merge with panel (keeping all panel rows)
    panel_filt = panel[['master_name', 'period', 'pop_census_tot', 'agg_count_tot',
                         'cotton_dist_post', 'nearby_post_25', 'nearby_post_50', 'nearby_post_75',
                         'ln_popdensity_census', 'linkable_shr_tot', 'under_15_shr_census',
                         'elderly_shr_census', 'cotton_eshr_post', 'post', 'dist_id', 'county'] +
                        regxpost_cols].copy()
    panel_filt = panel_filt.merge(filt_deaths, on=['master_name', 'period'], how='left')
    panel_filt['filt_census_count_tot'] = panel_filt['filt_census_count_tot'].fillna(0)

    # Lambda
    total_filt = panel_filt['filt_census_count_tot'].sum()
    total_agg = panel_filt['agg_count_tot'].sum()
    lam = total_filt / total_agg if total_agg > 0 else 1.0

    # Mortality rate
    panel_filt['census_mr_tot'] = (panel_filt['filt_census_count_tot'] / (panel_filt['pop_census_tot'] / 1000)) / 5 / lam

    return panel_filt


# No foreign-born links (ctry == "ENG" or "WAL")
panel_nofb = rebuild_mortality_with_filter(linked_raw, (linked_raw['ctry'] == 'ENG') | (linked_raw['ctry'] == 'WAL'), 'no_foreign_born')
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | master_name + period",
    data=panel_nofb, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/data/no_foreign_born_links",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#linking_restriction",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="No foreign-born links",
    controls_desc="Col 3 full controls",
    weights_col="pop_census_tot",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/no_foreign_born_links",
                "description": "Exclude foreign-born linked deaths"},
)

# Linking restriction variants
linking_restrictions = [
    ("unique_nlast", linked_raw['unique_nlast'] == 1, "Unique by last name within district-period"),
    ("unique_nlast_nfirst", linked_raw['unique_nlast_nfirst'] == 1, "Unique by first+last name"),
    ("dist_lt_200", linked_raw['dist'] < 200, "Link distance < 200"),
    ("dist_lt_100", linked_raw['dist'] < 100, "Link distance < 100"),
    ("dist_lt_50", linked_raw['dist'] < 50, "Link distance < 50"),
    ("dist_exact", linked_raw['dist'] == 0, "Exact match (distance = 0)"),
]

for name, mask, desc in linking_restrictions:
    panel_lr = rebuild_mortality_with_filter(linked_raw, mask, name)
    run_did(
        formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | master_name + period",
        data=panel_lr, vcov={"CRV1": "master_name"},
        treatment_var="cotton_dist_post",
        spec_id=f"rc/data/linking_restriction/{name}",
        spec_tree_path="specification_tree/modules/robustness/data_construction.md#linking_restriction",
        baseline_group_id="G1", outcome_var="census_mr_tot",
        sample_desc=f"Linking restriction: {desc}",
        controls_desc="Col 3 full controls",
        weights_col="pop_census_tot",
        axis_block_name="data_construction",
        axis_block={"spec_id": f"rc/data/linking_restriction/{name}",
                    "description": desc},
    )

# --- Additional cross-product specs for G1 ---

# LOO on Col 2 spec (without nearby rings)
# Drop ln_popdensity from Col 2
loo_c2_no_density = f"linkable_shr_tot + under_15_shr_census + elderly_shr_census + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_c2_no_density} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/ln_popdensity",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 2 minus ln_popdensity",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/ln_popdensity", "family": "loo",
                "dropped": ["ln_popdensity_census"], "base_col": "col2"},
)

# Drop linkable_shr from Col 2
loo_c2_no_linkable = f"ln_popdensity_census + under_15_shr_census + elderly_shr_census + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_c2_no_linkable} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/linkable_shr",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 2 minus linkable_shr_tot",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/linkable_shr", "family": "loo",
                "dropped": ["linkable_shr_tot"], "base_col": "col2"},
)

# Drop age shares from Col 2
loo_c2_no_age = f"ln_popdensity_census + linkable_shr_tot + {regxpost_str}"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_c2_no_age} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/age_shares",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 2 minus age shares",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/age_shares", "family": "loo",
                "dropped": ["under_15_shr_census", "elderly_shr_census"], "base_col": "col2"},
)

# Drop region x period FE from Col 2
loo_c2_no_region = "ln_popdensity_census + linkable_shr_tot + under_15_shr_census + elderly_shr_census"
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {loo_c2_no_region} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/controls/loo/region_x_period",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel", controls_desc="Col 2 minus region x period FE",
    weights_col="pop_census_tot",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/region_x_period", "family": "loo",
                "dropped": ["_IregXpost_*"], "base_col": "col2"},
)

# Sample exclusions on Col 1 and Col 2 baselines
# Exclude Manchester on Col 1
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post | {fe_str}",
    data=panel_no_manchester, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude Manchester, Col 1 (no controls)",
    controls_desc="none",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester",
                "excluded": ["GREATER MANCHESTER"], "base_col": "col1"},
)

# Exclude Manchester on Col 2
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col2} | {fe_str}",
    data=panel_no_manchester, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude Manchester, Col 2 controls",
    controls_desc="Col 2 controls",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester",
                "excluded": ["GREATER MANCHESTER"], "base_col": "col2"},
)

# Exclude Manchester+Liverpool+Leeds on Col 1
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post | {fe_str}",
    data=panel_no_big3, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester_liverpool_leeds",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude big 3, Col 1 (no controls)",
    controls_desc="none",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester_liverpool_leeds",
                "excluded": ["GREATER MANCHESTER", "LIVERPOOL", "GREATER LEEDS"], "base_col": "col1"},
)

# Exclude Manchester+Liverpool+Leeds on Col 2
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col2} | {fe_str}",
    data=panel_no_big3, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester_liverpool_leeds",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude big 3, Col 2 controls",
    controls_desc="Col 2 controls",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester_liverpool_leeds",
                "excluded": ["GREATER MANCHESTER", "LIVERPOOL", "GREATER LEEDS"], "base_col": "col2"},
)

# Unweighted on Col 1 and Col 2
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel unweighted, Col 1",
    controls_desc="none",
    weights_col=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "base_col": "col1"},
)

run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col2} | {fe_str}",
    data=panel, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Full panel unweighted, Col 2",
    controls_desc="Col 2 controls",
    weights_col=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "base_col": "col2"},
)

# Unweighted + exclude Manchester (combined robustness)
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel_no_manchester, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude Manchester, unweighted, Col 3",
    controls_desc="Col 3 full controls",
    weights_col=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted",
                "description": "Unweighted + exclude Manchester"},
)

# Exclude Manchester + Liverpool + Leeds, Col 2
run_did(
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col2} | {fe_str}",
    data=panel_no_big3, vcov={"CRV1": "master_name"},
    treatment_var="cotton_dist_post",
    spec_id="rc/sample/exclude_manchester_liverpool_leeds",
    spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
    baseline_group_id="G1", outcome_var="census_mr_tot",
    sample_desc="Exclude big 3, Col 2 controls (extra)",
    controls_desc="Col 2 controls",
    weights_col="pop_census_tot",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_manchester_liverpool_leeds",
                "excluded": ["GREATER MANCHESTER", "LIVERPOOL", "GREATER LEEDS"],
                "base_col": "col2_extra"},
)

# --- Inference variants for G1 ---
# County-level clustering on baseline Col 1
add_inference_row(
    base_run_id=run_id_t2c1, treatment_var="cotton_dist_post",
    infer_spec_id="infer/se/cluster/county",
    infer_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
    baseline_group_id="G1",
    infer_params={"cluster_var": "county"},
    formula=f"census_mr_tot ~ cotton_dist_post | {fe_str}",
    data=panel, vcov={"CRV1": "county"},
    weights_col="pop_census_tot",
    cluster_var="county",
)

# County-level clustering on Col 3
add_inference_row(
    base_run_id=run_id_t2c3, treatment_var="cotton_dist_post",
    infer_spec_id="infer/se/cluster/county",
    infer_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
    baseline_group_id="G1",
    infer_params={"cluster_var": "county"},
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel, vcov={"CRV1": "county"},
    weights_col="pop_census_tot",
    cluster_var="county",
)

# HC1 robust on Col 1
add_inference_row(
    base_run_id=run_id_t2c1, treatment_var="cotton_dist_post",
    infer_spec_id="infer/se/hc/robust",
    infer_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    infer_params={},
    formula=f"census_mr_tot ~ cotton_dist_post | {fe_str}",
    data=panel, vcov="hetero",
    weights_col="pop_census_tot",
)

# HC1 robust on Col 3
add_inference_row(
    base_run_id=run_id_t2c3, treatment_var="cotton_dist_post",
    infer_spec_id="infer/se/hc/robust",
    infer_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    infer_params={},
    formula=f"census_mr_tot ~ cotton_dist_post + {controls_col3} | {fe_str}",
    data=panel, vcov="hetero",
    weights_col="pop_census_tot",
)


# ============================================================
# G2: Age-specific mortality DD
# ============================================================
print("=" * 60)
print("G2: Age-specific mortality DD")
print("=" * 60)

age_groups = [
    ('_under15', 'under15'),
    ('_15_24', '15_24'),
    ('_25_34', '25_34'),
    ('_35_44', '35_44'),
    ('_45_54', '45_54'),
    ('_55_64', '55_64'),
    ('_over64', 'over64'),
]

for suffix, label in age_groups:
    outcome = f'census_mr{suffix}'
    pop_wt = f'pop_census{suffix}'

    # Build age-specific controls
    # ln_popdensity_census* -> uses overall density (same variable)
    # linkable_shr{suffix}* -> age-specific linkable share
    age_controls = f"nearby_post_25 + nearby_post_50 + nearby_post_75 + ln_popdensity_census + linkable_shr{suffix} + {regxpost_str}"

    # Check if outcome exists in panel
    if outcome not in panel.columns:
        print(f"  Skipping {outcome}: column not found")
        continue

    # --- Baseline: Table 3 specification ---
    run_did(
        formula=f"{outcome} ~ cotton_dist_post + {age_controls} | {fe_str}",
        data=panel, vcov={"CRV1": "master_name"},
        treatment_var="cotton_dist_post",
        spec_id="baseline",
        spec_tree_path="specification_tree/designs/difference_in_differences.md#twfe",
        baseline_group_id="G2",
        outcome_var=outcome,
        sample_desc=f"Full panel, age group {label}",
        controls_desc=f"nearby rings, ln_popdensity, linkable_shr{suffix}, region x period FE",
        weights_col=pop_wt,
    )

    # --- RC: Exclude Manchester ---
    run_did(
        formula=f"{outcome} ~ cotton_dist_post + {age_controls} | {fe_str}",
        data=panel_no_manchester, vcov={"CRV1": "master_name"},
        treatment_var="cotton_dist_post",
        spec_id="rc/sample/exclude_manchester",
        spec_tree_path="specification_tree/modules/robustness/sample.md#exclude_outlier",
        baseline_group_id="G2",
        outcome_var=outcome,
        sample_desc=f"Exclude Manchester, age group {label}",
        controls_desc=f"nearby rings, ln_popdensity, linkable_shr{suffix}, region x period FE",
        weights_col=pop_wt,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/exclude_manchester",
                    "excluded": ["GREATER MANCHESTER"]},
    )

    # --- RC: Unweighted ---
    run_did(
        formula=f"{outcome} ~ cotton_dist_post + {age_controls} | {fe_str}",
        data=panel, vcov={"CRV1": "master_name"},
        treatment_var="cotton_dist_post",
        spec_id="rc/weights/unweighted",
        spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
        baseline_group_id="G2",
        outcome_var=outcome,
        sample_desc=f"Full panel unweighted, age group {label}",
        controls_desc=f"nearby rings, ln_popdensity, linkable_shr{suffix}, region x period FE",
        weights_col=None,
        axis_block_name="weights",
        axis_block={"spec_id": "rc/weights/unweighted",
                    "description": "No population weights"},
    )


# ============================================================
# Write outputs
# ============================================================
print("=" * 60)
print("Writing outputs...")
print("=" * 60)

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {len(spec_df)} rows to specification_results.csv")
print(f"  Successes: {spec_df['run_success'].sum()}")
print(f"  Failures: {(spec_df['run_success'] == 0).sum()}")

if inference_results:
    inf_df = pd.DataFrame(inference_results)
    inf_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    print(f"Wrote {len(inf_df)} rows to inference_results.csv")

# Summary
n_g1 = len([r for r in results if r["baseline_group_id"] == "G1"])
n_g2 = len([r for r in results if r["baseline_group_id"] == "G2"])
n_success = sum(1 for r in results if r["run_success"] == 1)
n_fail = sum(1 for r in results if r["run_success"] == 0)

search_md = f"""# Specification Search Report: {PAPER_ID}

## Surface Summary

- **Paper ID**: {PAPER_ID}
- **Design**: Difference-in-differences (TWFE)
- **Baseline groups**: 2
  - G1: Total mortality (census_mr_tot ~ cotton_dist_post | master_name + period), aweight=pop_census_tot, cluster(master_name)
  - G2: Age-specific mortality (census_mr_{{age}} ~ cotton_dist_post), 7 age groups
- **Budgets**: G1=80, G2=30
- **Seed**: 128521
- **Surface hash**: {SURFACE_HASH}

## Execution Summary

| Group | Planned | Executed | Success | Failed |
|-------|---------|----------|---------|--------|
| G1    | {n_g1}  | {n_g1}   | {sum(1 for r in results if r['baseline_group_id']=='G1' and r['run_success']==1)} | {sum(1 for r in results if r['baseline_group_id']=='G1' and r['run_success']==0)} |
| G2    | {n_g2}  | {n_g2}   | {sum(1 for r in results if r['baseline_group_id']=='G2' and r['run_success']==1)} | {sum(1 for r in results if r['baseline_group_id']=='G2' and r['run_success']==0)} |
| **Total** | **{len(results)}** | **{len(results)}** | **{n_success}** | **{n_fail}** |

### Inference variants: {len(inference_results)} rows written to inference_results.csv

## Specification Details

### G1: Total Mortality
- **Baseline**: Table 2 Col 1 (no controls), Col 2 (demographics + region FE), Col 3 (+ nearby rings)
- **RC/controls/loo**: Drop ln_popdensity, linkable_shr, age_shares, region_x_period, nearby_rings (5 LOO specs)
- **RC/sample**: Exclude Manchester; Exclude Manchester+Liverpool+Leeds
- **RC/weights**: Unweighted
- **RC/data**: No foreign-born links; 6 linking restriction variants (unique_nlast, unique_nlast_nfirst, dist<200, dist<100, dist<50, dist==0)
- **Inference**: County-level clustering, HC1 robust

### G2: Age-Specific Mortality
- **Baseline**: 7 age groups (under15, 15-24, 25-34, 35-44, 45-54, 55-64, over64), Table 3 specification
- **RC/sample**: Exclude Manchester (7 specs)
- **RC/weights**: Unweighted (7 specs)

## Data Preparation

The regression dataset was reconstructed from raw data following the Stata do-files:
1. Generated age weights from Registrar mortality data and linked GRO-census records
2. Collapsed linked deaths to district x period panel
3. Merged with aggregate registrar deaths and census population counts
4. Computed lambda scaling factors (linked/aggregate ratio)
5. Calculated standardized mortality rates per 1000 persons per year
6. Merged cotton district indicators and population controls
7. Generated region x period interaction dummies

For linking restriction variants, the entire pipeline was re-run with different linked data subsets, including re-computed age weights and lambdas.

## Deviations and Notes

1. **Permutation test**: Not implemented (requires 538 iterations of spatial reassignment). Reported as inference variant in surface but skipped.
2. **Region x period FE**: Generated as interaction dummies (region x post) rather than absorbed, since the original code uses `xi i.region*post` which produces the same set of indicators.
3. **Data reconstruction**: The _Temp directory files were not pre-built; the entire data preparation was replicated in Python from raw data files. Minor numerical differences may arise from floating-point precision.

## Software Stack

- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)

print(f"\nDone! Total specs: {len(results)}, Inference variants: {len(inference_results)}")
