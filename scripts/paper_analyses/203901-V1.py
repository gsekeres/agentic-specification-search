"""
Specification Search Analysis for Paper 203901-V1
AEJ: Policy

Paper Title: "The Anatomy of a Crackdown: Enforcement Strategies to Combat Illicit Opioid Distribution"

Main Hypothesis: DEA crackdowns on high-prescribing doctors reduce local opioid dispensing (MME per capita)
                 but may cause substitution across markets and products.

Identification Strategy: Staggered Difference-in-Differences / Event Study
    - Treatment: DEA action against a practitioner in a county
    - Unit: County
    - Time: Year
    - Fixed Effects: County FE + State x Year FE

Data:
    - County-level panel data on opioid prescriptions (ARCOS database)
    - DEA enforcement actions against practitioners
    - Time period: 2006-2018 (approximately)
"""

import pandas as pd
import numpy as np
import pyreadr
import pyfixest as pf
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_DIR = os.path.join(BASE_DIR, "data/downloads/extracted/203901-V1/Files for Replication/Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/downloads/extracted/203901-V1/Files for Replication")

# Paper metadata
PAPER_ID = "203901-V1"
JOURNAL = "AEJ: Policy"
PAPER_TITLE = "The Anatomy of a Crackdown: Enforcement Strategies to Combat Illicit Opioid Distribution"

# Method classification
METHOD_CODE = "event_study"  # Primary: staggered DiD with event study
METHOD_TREE_PATH = "specification_tree/methods/event_study.md"

# Initialize results list
results = []

def create_result_dict(spec_id, spec_tree_path, outcome_var, treatment_var, model, df,
                       fixed_effects="", controls_desc="", cluster_var="", model_type="TWFE"):
    """Create a standardized result dictionary from a pyfixest model."""

    # Get coefficient for treatment
    treatment_coef = model.coef().get(treatment_var, None)
    treatment_se = model.se().get(treatment_var, None)
    treatment_tstat = model.tstat().get(treatment_var, None)
    treatment_pval = model.pvalue().get(treatment_var, None)

    if treatment_coef is None:
        # Try to find the treatment variable in the results
        for var in model.coef().index:
            if 'treatment' in var.lower() or treatment_var.lower() in var.lower():
                treatment_coef = model.coef()[var]
                treatment_se = model.se()[var]
                treatment_tstat = model.tstat()[var]
                treatment_pval = model.pvalue()[var]
                treatment_var = var
                break

    # Confidence intervals
    ci_lower = treatment_coef - 1.96 * treatment_se if treatment_se is not None else None
    ci_upper = treatment_coef + 1.96 * treatment_se if treatment_se is not None else None

    # Build coefficient vector
    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": float(treatment_coef) if treatment_coef is not None else None,
            "se": float(treatment_se) if treatment_se is not None else None,
            "pval": float(treatment_pval) if treatment_pval is not None else None
        },
        "controls": [],
        "fixed_effects": fixed_effects.split(" + ") if fixed_effects else [],
        "diagnostics": {
            "first_stage_F": None,
            "overid_pval": None,
            "hausman_pval": None
        }
    }

    # Add other coefficients
    for var in model.coef().index:
        if var != treatment_var and not var.startswith('rel_year'):
            coef_vector["controls"].append({
                "var": var,
                "coef": float(model.coef()[var]),
                "se": float(model.se()[var]),
                "pval": float(model.pvalue()[var])
            })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(treatment_coef) if treatment_coef is not None else None,
        'std_error': float(treatment_se) if treatment_se is not None else None,
        't_stat': float(treatment_tstat) if treatment_tstat is not None else None,
        'p_value': float(treatment_pval) if treatment_pval is not None else None,
        'ci_lower': float(ci_lower) if ci_lower is not None else None,
        'ci_upper': float(ci_upper) if ci_upper is not None else None,
        'n_obs': int(model._N),
        'r_squared': float(model._r2) if hasattr(model, '_r2') else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': f"N = {model._N}",
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


print("=" * 80)
print(f"SPECIFICATION SEARCH: {PAPER_ID}")
print("=" * 80)

# =============================================================================
# STEP 1: Load and Prepare Data
# =============================================================================
print("\n[Step 1] Loading and preparing data...")

# Load county panel data
fullcountypanel = pyreadr.read_r(os.path.join(DATA_DIR, 'fullcountypanel.rds'))
fullcountypanel = fullcountypanel[None] if None in fullcountypanel else list(fullcountypanel.values())[0]
print(f"  Loaded fullcountypanel: {fullcountypanel.shape}")

# Load DEA actions
dea_actions = pd.read_excel(os.path.join(DATA_DIR, 'DEA actions handcoded.xlsx'))
dea_actions = dea_actions.rename(columns={'DEA Number': 'BUYER_DEA_NO'})
print(f"  Loaded DEA actions: {dea_actions.shape}")

# Load pharmacy panel data
monthlypharmpanel = pyreadr.read_r(os.path.join(DATA_DIR, 'monthlypharmpanel.rds'))
monthlypharmpanel = monthlypharmpanel[None] if None in monthlypharmpanel else list(monthlypharmpanel.values())[0]
print(f"  Loaded monthlypharmpanel: {monthlypharmpanel.shape}")

# =============================================================================
# STEP 2: Recreate the affectedcounties dataset (main analysis dataset)
# =============================================================================
print("\n[Step 2] Creating affectedcounties dataset...")

# Filter DEA actions for practitioners
dea_practitioners = dea_actions[dea_actions['Type'] == 'Practitioner'].copy()
dea_practitioners['ActionDate'] = pd.to_datetime(dea_practitioners['Date'])
dea_practitioners['BUYER_STATE'] = dea_practitioners['State']
dea_practitioners['BUYER_CITY'] = dea_practitioners['Location'].apply(lambda x: str(x).split(',')[0].upper() if pd.notna(x) else None)

# Get first action date by city
dea_first_actions = dea_practitioners.groupby(['BUYER_STATE', 'BUYER_CITY']).agg({
    'ActionDate': 'min'
}).reset_index()

# Map city to county via pharmacy panel
city_county_map = monthlypharmpanel[['BUYER_CITY', 'BUYER_STATE', 'county_fips']].drop_duplicates()
city_county_map = city_county_map.groupby(['BUYER_CITY', 'BUYER_STATE']).first().reset_index()

# Merge DEA actions with county mapping
dea_county_actions = dea_first_actions.merge(
    city_county_map,
    on=['BUYER_STATE', 'BUYER_CITY'],
    how='inner'
)

# Keep only first action per county and filter years >= 2007
dea_county_actions['actionyear'] = dea_county_actions['ActionDate'].dt.year
dea_county_actions = dea_county_actions[dea_county_actions['actionyear'] >= 2007]
dea_county_actions = dea_county_actions.groupby('county_fips').first().reset_index()
dea_county_actions['county_fips'] = dea_county_actions['county_fips'].astype(float)

# Create main analysis dataset
affectedcounties = fullcountypanel.copy()
affectedcounties = affectedcounties.merge(
    dea_county_actions[['county_fips', 'ActionDate', 'actionyear']],
    on='county_fips',
    how='left'
)

# Create key variables
affectedcounties['opiatesper100K'] = affectedcounties['DOSAGE_UNIT'] / affectedcounties['populationAHRF'] * 100000
affectedcounties['MME_PC'] = affectedcounties['PCPV'] * 10  # Morphine milligram equivalents per capita

# Create relative time variable
affectedcounties['treat'] = (~affectedcounties['actionyear'].isna()).astype(int)
affectedcounties['rel_year'] = np.where(
    affectedcounties['treat'] == 1,
    affectedcounties['year'] - affectedcounties['actionyear'],
    0
)
affectedcounties['treatment'] = np.where(
    (affectedcounties['rel_year'] >= 0) & (affectedcounties['treat'] == 1),
    1, 0
)

# Convert county_fips and year to appropriate types
affectedcounties['county_fips'] = affectedcounties['county_fips'].astype(str)
affectedcounties['year'] = affectedcounties['year'].astype(int)
affectedcounties['state'] = affectedcounties['state'].astype(str)

# Create state-year interaction variable
affectedcounties['state_year'] = affectedcounties['state'] + "_" + affectedcounties['year'].astype(str)

# Filter to event window (-3 to +3)
df_es = affectedcounties[(affectedcounties['rel_year'] >= -3) & (affectedcounties['rel_year'] <= 3)].copy()

# Remove missing MME_PC
df_es = df_es[df_es['MME_PC'].notna()].copy()

print(f"  Analysis dataset (event window -3 to +3): {df_es.shape}")
print(f"  Treated counties: {df_es[df_es['treat']==1]['county_fips'].nunique()}")
print(f"  Years: {df_es['year'].min()} - {df_es['year'].max()}")

# =============================================================================
# STEP 3: Baseline Replication
# =============================================================================
print("\n[Step 3] Running baseline specification...")

# Main outcome variable
outcome_var = "MME_PC"
treatment_var = "treatment"

# Baseline: TWFE with county FE + state x year FE
# Note: pyfixest uses ^ for interactions in fixed effects
try:
    baseline_model = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )

    results.append(create_result_dict(
        spec_id="baseline",
        spec_tree_path="methods/event_study.md",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=baseline_model,
        df=df_es,
        fixed_effects="county_fips + state^year",
        controls_desc="None",
        cluster_var="county_fips",
        model_type="TWFE"
    ))
    print(f"  Baseline coefficient: {baseline_model.coef()[treatment_var]:.4f}")
    print(f"  SE: {baseline_model.se()[treatment_var]:.4f}")
    print(f"  p-value: {baseline_model.pvalue()[treatment_var]:.4f}")
except Exception as e:
    print(f"  Error in baseline: {e}")

# =============================================================================
# STEP 4: DiD/Event Study Method Variations
# =============================================================================
print("\n[Step 4] Running method variations...")

# 4.1 Fixed Effects Variations
print("  4.1 Fixed effects variations...")

# Unit FE only
try:
    model_fe_unit = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="did/fe/unit_only",
        spec_tree_path="methods/difference_in_differences.md#fixed-effects",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_fe_unit,
        df=df_es,
        fixed_effects="county_fips",
        cluster_var="county_fips"
    ))
    print(f"    Unit FE only: coef={model_fe_unit.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"    Error: {e}")

# Time FE only
try:
    model_fe_time = pf.feols(
        f"{outcome_var} ~ {treatment_var} | year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="did/fe/time_only",
        spec_tree_path="methods/difference_in_differences.md#fixed-effects",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_fe_time,
        df=df_es,
        fixed_effects="year",
        cluster_var="county_fips"
    ))
    print(f"    Time FE only: coef={model_fe_time.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"    Error: {e}")

# Two-way FE (unit + time)
try:
    model_fe_twoway = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="did/fe/twoway",
        spec_tree_path="methods/difference_in_differences.md#fixed-effects",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_fe_twoway,
        df=df_es,
        fixed_effects="county_fips + year",
        cluster_var="county_fips"
    ))
    print(f"    Two-way FE: coef={model_fe_twoway.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"    Error: {e}")

# No FE (pooled OLS)
try:
    model_fe_none = pf.feols(
        f"{outcome_var} ~ {treatment_var}",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="did/fe/none",
        spec_tree_path="methods/difference_in_differences.md#fixed-effects",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_fe_none,
        df=df_es,
        fixed_effects="None",
        cluster_var="county_fips"
    ))
    print(f"    No FE: coef={model_fe_none.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"    Error: {e}")

# 4.2 With controls
print("  4.2 Control set variations...")

control_vars = ['unemploymentrate', 'MD_PC', 'PCT_MEDICARE', 'PCT_BLACK', 'POP_DENSITY']
available_controls = [c for c in control_vars if c in df_es.columns and df_es[c].notna().sum() > 1000]
print(f"    Available controls: {available_controls}")

if available_controls:
    controls_str = " + ".join(available_controls)

    # With minimal controls
    try:
        model_controls = pf.feols(
            f"{outcome_var} ~ {treatment_var} + {controls_str} | county_fips + state^year",
            data=df_es,
            vcov={'CRV1': 'county_fips'}
        )
        results.append(create_result_dict(
            spec_id="did/controls/full",
            spec_tree_path="methods/difference_in_differences.md#control-sets",
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            model=model_controls,
            df=df_es,
            fixed_effects="county_fips + state^year",
            controls_desc=controls_str,
            cluster_var="county_fips"
        ))
        print(f"    With controls: coef={model_controls.coef()[treatment_var]:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

# =============================================================================
# STEP 5: Sample Restrictions
# =============================================================================
print("\n[Step 5] Running sample restrictions...")

# 5.1 Early vs Late period
median_year = df_es['year'].median()
df_early = df_es[df_es['year'] <= median_year]
df_late = df_es[df_es['year'] > median_year]

try:
    model_early = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_early,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="did/sample/early_period",
        spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_early,
        df=df_early,
        fixed_effects="county_fips + state^year",
        cluster_var="county_fips"
    ))
    print(f"  Early period: coef={model_early.coef()[treatment_var]:.4f}, N={model_early._N}")
except Exception as e:
    print(f"  Early period error: {e}")

try:
    model_late = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_late,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="did/sample/late_period",
        spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_late,
        df=df_late,
        fixed_effects="county_fips + state^year",
        cluster_var="county_fips"
    ))
    print(f"  Late period: coef={model_late.coef()[treatment_var]:.4f}, N={model_late._N}")
except Exception as e:
    print(f"  Late period error: {e}")

# 5.2 Trim outliers (drop top/bottom 1%)
q01 = df_es[outcome_var].quantile(0.01)
q99 = df_es[outcome_var].quantile(0.99)
df_trimmed = df_es[(df_es[outcome_var] >= q01) & (df_es[outcome_var] <= q99)]

try:
    model_trimmed = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_trimmed,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="robust/sample/trim_1pct",
        spec_tree_path="robustness/sample_restrictions.md",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_trimmed,
        df=df_trimmed,
        fixed_effects="county_fips + state^year",
        cluster_var="county_fips"
    ))
    print(f"  Trimmed 1%: coef={model_trimmed.coef()[treatment_var]:.4f}, N={model_trimmed._N}")
except Exception as e:
    print(f"  Trimmed error: {e}")

# 5.3 Balanced panel (counties observed all years)
county_counts = df_es.groupby('county_fips').size()
balanced_counties = county_counts[county_counts == county_counts.max()].index
df_balanced = df_es[df_es['county_fips'].isin(balanced_counties)]

try:
    model_balanced = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_balanced,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="robust/sample/balanced",
        spec_tree_path="robustness/sample_restrictions.md",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_balanced,
        df=df_balanced,
        fixed_effects="county_fips + state^year",
        cluster_var="county_fips"
    ))
    print(f"  Balanced panel: coef={model_balanced.coef()[treatment_var]:.4f}, N={model_balanced._N}")
except Exception as e:
    print(f"  Balanced panel error: {e}")

# =============================================================================
# STEP 6: Clustering Variations
# =============================================================================
print("\n[Step 6] Running clustering variations...")

# Robust SE (no clustering)
try:
    model_robust = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_es,
        vcov='hetero'
    )
    results.append(create_result_dict(
        spec_id="robust/cluster/none",
        spec_tree_path="robustness/clustering_variations.md",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_robust,
        df=df_es,
        fixed_effects="county_fips + state^year",
        cluster_var="none (robust)",
        model_type="TWFE"
    ))
    print(f"  Robust SE: coef={model_robust.coef()[treatment_var]:.4f}, SE={model_robust.se()[treatment_var]:.4f}")
except Exception as e:
    print(f"  Robust SE error: {e}")

# Cluster by state
try:
    model_state_cluster = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_es,
        vcov={'CRV1': 'state'}
    )
    results.append(create_result_dict(
        spec_id="robust/cluster/state",
        spec_tree_path="robustness/clustering_variations.md",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_state_cluster,
        df=df_es,
        fixed_effects="county_fips + state^year",
        cluster_var="state",
        model_type="TWFE"
    ))
    print(f"  Cluster by state: coef={model_state_cluster.coef()[treatment_var]:.4f}, SE={model_state_cluster.se()[treatment_var]:.4f}")
except Exception as e:
    print(f"  State cluster error: {e}")

# =============================================================================
# STEP 7: Functional Form Variations
# =============================================================================
print("\n[Step 7] Running functional form variations...")

# Log outcome (add small constant to handle zeros)
df_es['log_MME_PC'] = np.log(df_es['MME_PC'] + 1)

try:
    model_log = pf.feols(
        f"log_MME_PC ~ {treatment_var} | county_fips + state^year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="robust/form/y_log",
        spec_tree_path="robustness/functional_form.md",
        outcome_var="log_MME_PC",
        treatment_var=treatment_var,
        model=model_log,
        df=df_es,
        fixed_effects="county_fips + state^year",
        cluster_var="county_fips"
    ))
    print(f"  Log outcome: coef={model_log.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"  Log outcome error: {e}")

# Asinh transformation (handles zeros better)
df_es['asinh_MME_PC'] = np.arcsinh(df_es['MME_PC'])

try:
    model_asinh = pf.feols(
        f"asinh_MME_PC ~ {treatment_var} | county_fips + state^year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="robust/form/y_asinh",
        spec_tree_path="robustness/functional_form.md",
        outcome_var="asinh_MME_PC",
        treatment_var=treatment_var,
        model=model_asinh,
        df=df_es,
        fixed_effects="county_fips + state^year",
        cluster_var="county_fips"
    ))
    print(f"  Asinh outcome: coef={model_asinh.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"  Asinh outcome error: {e}")

# =============================================================================
# STEP 8: Event Study Specifications
# =============================================================================
print("\n[Step 8] Running event study specifications...")

# Create event time dummies (reference period = -1)
df_es['rel_year_int'] = df_es['rel_year'].astype(int)
for t in range(-3, 4):
    if t != -1:  # Reference period
        # Use positive naming to avoid minus sign issues
        var_name = f'rel_year_m{abs(t)}' if t < 0 else f'rel_year_p{t}'
        df_es[var_name] = ((df_es['rel_year_int'] == t) & (df_es['treat'] == 1)).astype(int)

# Standard event study with reference = -1
def get_event_var_name(t):
    return f'rel_year_m{abs(t)}' if t < 0 else f'rel_year_p{t}'

event_vars = " + ".join([get_event_var_name(t) for t in range(-3, 4) if t != -1])

try:
    model_event_study = pf.feols(
        f"{outcome_var} ~ {event_vars} | county_fips + state^year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )

    # Extract event study coefficients
    event_coefs = []
    for t in range(-3, 4):
        if t == -1:
            event_coefs.append({"rel_time": t, "coef": 0, "se": None, "pval": None, "note": "reference"})
        else:
            var_name = get_event_var_name(t)
            if var_name in model_event_study.coef().index:
                event_coefs.append({
                    "rel_time": t,
                    "coef": float(model_event_study.coef()[var_name]),
                    "se": float(model_event_study.se()[var_name]),
                    "pval": float(model_event_study.pvalue()[var_name])
                })

    # Create result with event study coefficients
    coef_vector = {
        "event_time_coefficients": event_coefs,
        "fixed_effects": ["county_fips", "state^year"],
        "diagnostics": {"reference_period": -1}
    }

    # Get post-treatment average effect
    post_coefs = [c['coef'] for c in event_coefs if c['rel_time'] >= 0 and c['coef'] is not None and c.get('note') != 'reference']
    avg_post_effect = np.mean(post_coefs) if post_coefs else None

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': "es/window/symmetric",
        'spec_tree_path': "methods/event_study.md#event-window",
        'outcome_var': outcome_var,
        'treatment_var': "event_time_dummies",
        'coefficient': avg_post_effect,
        'std_error': None,
        't_stat': None,
        'p_value': None,
        'ci_lower': None,
        'ci_upper': None,
        'n_obs': int(model_event_study._N),
        'r_squared': float(model_event_study._r2),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': f"N = {model_event_study._N}, event window -3 to +3",
        'fixed_effects': "county_fips + state^year",
        'controls_desc': "None",
        'cluster_var': "county_fips",
        'model_type': "Event Study TWFE",
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    })

    print(f"  Event study (symmetric -3 to +3):")
    for c in event_coefs:
        if c.get('note') == 'reference':
            print(f"    t={c['rel_time']}: reference")
        else:
            print(f"    t={c['rel_time']}: coef={c['coef']:.4f}, se={c['se']:.4f}")
except Exception as e:
    print(f"  Event study error: {e}")

# Short window event study (-2 to +2)
df_es_short = df_es[(df_es['rel_year'] >= -2) & (df_es['rel_year'] <= 2)].copy()
for t in range(-2, 3):
    if t != -1:
        var_name = f'rel_year_m{abs(t)}' if t < 0 else f'rel_year_p{t}'
        df_es_short[var_name] = ((df_es_short['rel_year_int'] == t) & (df_es_short['treat'] == 1)).astype(int)

event_vars_short = " + ".join([f'rel_year_m{abs(t)}' if t < 0 else f'rel_year_p{t}' for t in range(-2, 3) if t != -1])

try:
    model_es_short = pf.feols(
        f"{outcome_var} ~ {event_vars_short} | county_fips + state^year",
        data=df_es_short,
        vcov={'CRV1': 'county_fips'}
    )

    event_coefs_short = []
    for t in range(-2, 3):
        if t == -1:
            event_coefs_short.append({"rel_time": t, "coef": 0, "se": None, "pval": None, "note": "reference"})
        else:
            var_name = f'rel_year_m{abs(t)}' if t < 0 else f'rel_year_p{t}'
            if var_name in model_es_short.coef().index:
                event_coefs_short.append({
                    "rel_time": t,
                    "coef": float(model_es_short.coef()[var_name]),
                    "se": float(model_es_short.se()[var_name]),
                    "pval": float(model_es_short.pvalue()[var_name])
                })

    coef_vector_short = {
        "event_time_coefficients": event_coefs_short,
        "fixed_effects": ["county_fips", "state^year"],
        "diagnostics": {"reference_period": -1}
    }

    post_coefs_short = [c['coef'] for c in event_coefs_short if c['rel_time'] >= 0 and c['coef'] is not None and c.get('note') != 'reference']
    avg_post_effect_short = np.mean(post_coefs_short) if post_coefs_short else None

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': "es/window/short",
        'spec_tree_path': "methods/event_study.md#event-window",
        'outcome_var': outcome_var,
        'treatment_var': "event_time_dummies",
        'coefficient': avg_post_effect_short,
        'std_error': None,
        't_stat': None,
        'p_value': None,
        'ci_lower': None,
        'ci_upper': None,
        'n_obs': int(model_es_short._N),
        'r_squared': float(model_es_short._r2),
        'coefficient_vector_json': json.dumps(coef_vector_short),
        'sample_desc': f"N = {model_es_short._N}, event window -2 to +2",
        'fixed_effects': "county_fips + state^year",
        'controls_desc': "None",
        'cluster_var': "county_fips",
        'model_type': "Event Study TWFE",
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    })
    print(f"  Event study (short -2 to +2): avg post effect = {avg_post_effect_short:.4f}")
except Exception as e:
    print(f"  Short event study error: {e}")

# =============================================================================
# STEP 9: Leave-One-Out Robustness (for controls)
# =============================================================================
print("\n[Step 9] Running leave-one-out specifications...")

if available_controls and len(available_controls) > 1:
    for dropped_var in available_controls:
        remaining_controls = [c for c in available_controls if c != dropped_var]
        if remaining_controls:
            remaining_str = " + ".join(remaining_controls)
            try:
                model_loo = pf.feols(
                    f"{outcome_var} ~ {treatment_var} + {remaining_str} | county_fips + state^year",
                    data=df_es,
                    vcov={'CRV1': 'county_fips'}
                )
                results.append(create_result_dict(
                    spec_id=f"robust/loo/drop_{dropped_var}",
                    spec_tree_path="robustness/leave_one_out.md",
                    outcome_var=outcome_var,
                    treatment_var=treatment_var,
                    model=model_loo,
                    df=df_es,
                    fixed_effects="county_fips + state^year",
                    controls_desc=remaining_str,
                    cluster_var="county_fips"
                ))
                print(f"  Drop {dropped_var}: coef={model_loo.coef()[treatment_var]:.4f}")
            except Exception as e:
                print(f"  Drop {dropped_var} error: {e}")

# =============================================================================
# STEP 10: Single Covariate Specifications
# =============================================================================
print("\n[Step 10] Running single covariate specifications...")

# Bivariate (no controls)
try:
    model_bivariate = pf.feols(
        f"{outcome_var} ~ {treatment_var} | county_fips + state^year",
        data=df_es,
        vcov={'CRV1': 'county_fips'}
    )
    results.append(create_result_dict(
        spec_id="robust/single/none",
        spec_tree_path="robustness/single_covariate.md",
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        model=model_bivariate,
        df=df_es,
        fixed_effects="county_fips + state^year",
        controls_desc="None",
        cluster_var="county_fips"
    ))
    print(f"  No controls: coef={model_bivariate.coef()[treatment_var]:.4f}")
except Exception as e:
    print(f"  Bivariate error: {e}")

# Single control at a time
if available_controls:
    for single_control in available_controls:
        try:
            model_single = pf.feols(
                f"{outcome_var} ~ {treatment_var} + {single_control} | county_fips + state^year",
                data=df_es,
                vcov={'CRV1': 'county_fips'}
            )
            results.append(create_result_dict(
                spec_id=f"robust/single/{single_control}",
                spec_tree_path="robustness/single_covariate.md",
                outcome_var=outcome_var,
                treatment_var=treatment_var,
                model=model_single,
                df=df_es,
                fixed_effects="county_fips + state^year",
                controls_desc=single_control,
                cluster_var="county_fips"
            ))
            print(f"  Single {single_control}: coef={model_single.coef()[treatment_var]:.4f}")
        except Exception as e:
            print(f"  Single {single_control} error: {e}")

# =============================================================================
# STEP 11: Alternative Outcome Variables
# =============================================================================
print("\n[Step 11] Running alternative outcome variables...")

# DOSAGE_UNIT (raw dosage units)
if 'DOSAGE_UNIT' in df_es.columns:
    df_es['log_dosage'] = np.log(df_es['DOSAGE_UNIT'] + 1)
    try:
        model_dosage = pf.feols(
            f"log_dosage ~ {treatment_var} | county_fips + state^year",
            data=df_es,
            vcov={'CRV1': 'county_fips'}
        )
        results.append(create_result_dict(
            spec_id="custom/outcome/log_dosage_unit",
            spec_tree_path="methods/event_study.md",
            outcome_var="log_dosage",
            treatment_var=treatment_var,
            model=model_dosage,
            df=df_es,
            fixed_effects="county_fips + state^year",
            cluster_var="county_fips"
        ))
        print(f"  log(DOSAGE_UNIT): coef={model_dosage.coef()[treatment_var]:.4f}")
    except Exception as e:
        print(f"  log(DOSAGE_UNIT) error: {e}")

# opiatesper100K (alternative measure)
if 'opiatesper100K' in df_es.columns:
    df_es['log_opiatesper100K'] = np.log(df_es['opiatesper100K'] + 1)
    try:
        model_opiates = pf.feols(
            f"log_opiatesper100K ~ {treatment_var} | county_fips + state^year",
            data=df_es,
            vcov={'CRV1': 'county_fips'}
        )
        results.append(create_result_dict(
            spec_id="custom/outcome/log_opiates_per100K",
            spec_tree_path="methods/event_study.md",
            outcome_var="log_opiatesper100K",
            treatment_var=treatment_var,
            model=model_opiates,
            df=df_es,
            fixed_effects="county_fips + state^year",
            cluster_var="county_fips"
        ))
        print(f"  log(opiatesper100K): coef={model_opiates.coef()[treatment_var]:.4f}")
    except Exception as e:
        print(f"  log(opiatesper100K) error: {e}")

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n[Step 12] Saving results...")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = os.path.join(OUTPUT_DIR, "specification_results.csv")
results_df.to_csv(output_path, index=False)
print(f"  Saved {len(results_df)} specifications to {output_path}")

# =============================================================================
# STEP 13: Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Filter to specs with valid coefficients on main outcome
main_results = results_df[
    (results_df['outcome_var'].isin(['MME_PC', 'log_MME_PC', 'asinh_MME_PC'])) &
    (results_df['coefficient'].notna())
]

if len(main_results) > 0:
    print(f"\nTotal specifications (main outcome): {len(main_results)}")
    print(f"Negative coefficients: {(main_results['coefficient'] < 0).sum()} ({100*(main_results['coefficient'] < 0).mean():.1f}%)")
    print(f"Significant at 5%: {(main_results['p_value'] < 0.05).sum()} ({100*(main_results['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(main_results['p_value'] < 0.01).sum()} ({100*(main_results['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {main_results['coefficient'].median():.4f}")
    print(f"Mean coefficient: {main_results['coefficient'].mean():.4f}")
    print(f"Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
