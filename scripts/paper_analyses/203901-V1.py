"""
Specification Search: 203901-V1
Paper: "Disrupting Drug Markets: The Effects of Crackdowns on Rogue Pain Clinics"
Journal: AEJ: Economic Policy

This paper studies the effects of DEA crackdowns on doctors (practitioners) involved in
illegal opioid distribution on county-level opioid dispensing and related outcomes.

Method: Staggered Difference-in-Differences (Event Study)
Main Outcome: MME_PC (Morphine Milligram Equivalents per Capita)
Treatment: DEA crackdown on practitioners (doctors) in a county
Fixed Effects: County + State x Year
Clustering: County level
"""

import pandas as pd
import numpy as np
import pyreadr
import pyfixest as pf
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/203901-V1/Files for Replication/Data"
OUTPUT_PATH = BASE_PATH / "data/downloads/extracted/203901-V1"

# Paper metadata
PAPER_ID = "203901-V1"
JOURNAL = "AEJ: Economic Policy"
PAPER_TITLE = "Disrupting Drug Markets: The Effects of Crackdowns on Rogue Pain Clinics"

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

# Load county panel
fullcountypanel = pyreadr.read_r(DATA_PATH / "fullcountypanel.RData")['fullcountypanel']

# Load monthly pharmacy panel
monthlypharmpanel = pyreadr.read_r(DATA_PATH / "monthlypharmpanel.RData")['monthlypharmpanel']

# Load DEA actions
deahandcoded = pd.read_excel(DATA_PATH / "DEA actions handcoded.xlsx")

print(f"County panel shape: {fullcountypanel.shape}")
print(f"Monthly panel shape: {monthlypharmpanel.shape}")
print(f"DEA actions shape: {deahandcoded.shape}")

# ============================================================================
# DATA PREPARATION (Replicating R code logic)
# ============================================================================

print("\nPreparing data...")

# Rename column
deahandcoded = deahandcoded.rename(columns={'DEA Number': 'BUYER_DEA_NO'})

# Filter non-null DEA numbers
deahandcoded = deahandcoded[deahandcoded['BUYER_DEA_NO'].notna()].copy()

# Extract year and process
deahandcoded['year'] = pd.to_datetime(deahandcoded['Date']).dt.year
deahandcoded['City'] = deahandcoded['Location'].str.replace(r',.*', '', regex=True).str.upper()

# Filter for practitioners only (doctors)
practitioners = deahandcoded[deahandcoded['Type'] == 'Practitioner'].copy()

# Get city mapping from monthlypharmpanel
city_county_mapping = monthlypharmpanel[['county_fips', 'BUYER_CITY', 'BUYER_COUNTY', 'BUYER_STATE']].drop_duplicates()
city_county_mapping = city_county_mapping.rename(columns={'BUYER_CITY': 'City', 'BUYER_STATE': 'State'})

# Process practitioners - get first action per city/state
practitioners_actions = practitioners.groupby(['City', 'State']).agg({
    'Date': 'first'
}).reset_index()
practitioners_actions.columns = ['BUYER_CITY', 'BUYER_STATE', 'ActionDate']

# Merge to get county mapping
practitioners_actions = practitioners_actions.merge(
    city_county_mapping.rename(columns={'City': 'BUYER_CITY', 'State': 'BUYER_STATE'}),
    on=['BUYER_CITY', 'BUYER_STATE'],
    how='inner'
)

# Filter for actions from 2007 onwards
practitioners_actions['ActionDate'] = pd.to_datetime(practitioners_actions['ActionDate'])
practitioners_actions = practitioners_actions[practitioners_actions['ActionDate'].dt.year >= 2007].copy()

# Get first action per county
county_actions = practitioners_actions.groupby('county_fips').agg({
    'ActionDate': 'first'
}).reset_index()
county_actions['actionyear'] = pd.to_datetime(county_actions['ActionDate']).dt.year

# Convert county_fips to numeric for merging
fullcountypanel['county_fips'] = pd.to_numeric(fullcountypanel['county_fips'], errors='coerce')
county_actions['county_fips'] = pd.to_numeric(county_actions['county_fips'], errors='coerce')

# Merge with county panel
affectedcounties = fullcountypanel.merge(county_actions, on='county_fips', how='left')

# Calculate suspicious pharmacy flags from monthly panel
monthlypharmpanel['county_fips_num'] = pd.to_numeric(monthlypharmpanel['county_fips'], errors='coerce')
monthly_sorted = monthlypharmpanel.sort_values(['BUYER_DEA_NO', 'date'])

# Rolling mean for suspicious flags
monthly_sorted['avg_pills'] = monthly_sorted.groupby('BUYER_DEA_NO')['monthlyopiates'].transform(
    lambda x: x.rolling(window=12, min_periods=12).mean()
)
monthly_sorted['avg_pills_2x'] = monthly_sorted['avg_pills'] * 2
monthly_sorted['roll_2x_flag'] = (monthly_sorted['monthlyopiates'] > monthly_sorted['avg_pills_2x']).astype(int)
monthly_sorted['roll_2x_flag'] = monthly_sorted['roll_2x_flag'].fillna(0)

# Aggregate by county and year
county_suspicious = monthly_sorted.groupby(['county_fips_num', 'year']).agg({
    'roll_2x_flag': 'sum',
    'BUYER_DEA_NO': 'nunique'
}).reset_index()
county_suspicious.columns = ['county_fips', 'year', 'flags', 'pharmacies']
county_suspicious['sharesuspicious'] = county_suspicious['flags'] / county_suspicious['pharmacies'] * 100
county_suspicious['eversuspicious'] = (county_suspicious['sharesuspicious'] > 0).astype(int)

# Get ever suspicious by county
county_ever_susp = county_suspicious.groupby('county_fips')['eversuspicious'].max().reset_index()

# Merge suspicious data
affectedcounties = affectedcounties.merge(county_suspicious[['county_fips', 'year', 'sharesuspicious']],
                                          on=['county_fips', 'year'], how='left')
affectedcounties = affectedcounties.merge(county_ever_susp, on='county_fips', how='left', suffixes=('', '_ever'))

# Create outcome variables
affectedcounties['MME_PC'] = affectedcounties['PCPV'] * 10
affectedcounties['opiatesper100K'] = affectedcounties['DOSAGE_UNIT'] / affectedcounties['populationAHRF'] * 100000

# Create treatment variables
affectedcounties['actionyear'] = affectedcounties['actionyear'].fillna(0).astype(int)
affectedcounties['rel_year'] = np.where(
    affectedcounties['actionyear'] > 0,
    affectedcounties['year'] - affectedcounties['actionyear'],
    0
)
affectedcounties['treat'] = np.where(
    (affectedcounties['actionyear'].isna()) | (affectedcounties['actionyear'] == 0),
    0, 1
)
affectedcounties['treatment'] = np.where(
    (affectedcounties['rel_year'] >= 0) & (affectedcounties['treat'] == 1),
    1, 0
)

# Define short run and long run windows
affectedcounties['shortrun'] = np.where(
    (affectedcounties['rel_year'] >= -3) & (affectedcounties['rel_year'] <= 1),
    1, 0
)
affectedcounties['longrun'] = np.where(
    ((affectedcounties['rel_year'] >= -3) & (affectedcounties['rel_year'] > 1) & (affectedcounties['rel_year'] <= 3)) |
    (affectedcounties['rel_year'] <= 0),
    1, 0
)

# Remove counties with missing mortality data (as in original code)
excluded_counties = [8053, 30025, 31143, 38047, 48417]
affectedcounties = affectedcounties[~affectedcounties['county_fips'].isin(excluded_counties)].copy()

# Create state_year for FE
affectedcounties['state_year'] = affectedcounties['state'].astype(str) + "_" + affectedcounties['year'].astype(str)

# Convert county_fips to string for FE
affectedcounties['county_fips_str'] = affectedcounties['county_fips'].astype(str)

print(f"\nFinal dataset shape: {affectedcounties.shape}")
print(f"Number of treated counties: {affectedcounties[affectedcounties['treat']==1]['county_fips'].nunique()}")
print(f"Number of control counties: {affectedcounties[affectedcounties['treat']==0]['county_fips'].nunique()}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_results(model, spec_id, spec_tree_path, treatment_var='treatment',
                    outcome_var='MME_PC', sample_desc='Full sample',
                    fixed_effects='county_fips + state_year', controls_desc='None',
                    cluster_var='county_fips', model_type='TWFE DiD'):
    """Extract results from pyfixest model."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%'] if treatment_var in ci.index else coef - 1.96*se
        ci_upper = ci.loc[treatment_var, '97.5%'] if treatment_var in ci.index else coef + 1.96*se
        nobs = model._N
        r2 = model._r2

        # Get all coefficients as JSON
        coef_dict = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "fixed_effects_absorbed": fixed_effects.split(' + ') if fixed_effects else [],
            "n_obs": int(nobs),
            "r_squared": float(r2) if r2 is not None else None
        }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(nobs),
            'r_squared': float(r2) if r2 is not None else None,
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

results = []

# Main sample: within +/- 3 years of treatment
main_sample = affectedcounties[(affectedcounties['rel_year'] >= -3) &
                               (affectedcounties['rel_year'] <= 3)].copy()

print(f"\nMain sample (rel_year -3 to 3): {main_sample.shape[0]} obs")

# ============================================================================
# 1. BASELINE SPECIFICATION (Table 1 replication)
# ============================================================================

print("\n1. Running baseline specification...")

try:
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'baseline', 'methods/difference_in_differences.md',
                            sample_desc='Counties within +/- 3 years of treatment')
    if result:
        results.append(result)
        print(f"  Baseline: coef={result['coefficient']:.3f}, se={result['std_error']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in baseline: {e}")

# ============================================================================
# 2. FIXED EFFECTS VARIATIONS
# ============================================================================

print("\n2. Running fixed effects variations...")

# 2a. County FE only (no state x year)
try:
    model = pf.feols("MME_PC ~ treatment | county_fips_str + year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'did/fe/county_year', 'methods/difference_in_differences.md#fixed-effects',
                            fixed_effects='county_fips + year')
    if result:
        results.append(result)
        print(f"  County + Year FE: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 2b. County FE only (unit only)
try:
    model = pf.feols("MME_PC ~ treatment | county_fips_str",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'did/fe/unit_only', 'methods/difference_in_differences.md#fixed-effects',
                            fixed_effects='county_fips')
    if result:
        results.append(result)
        print(f"  County FE only: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 2c. Year FE only
try:
    model = pf.feols("MME_PC ~ treatment | year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'did/fe/time_only', 'methods/difference_in_differences.md#fixed-effects',
                            fixed_effects='year')
    if result:
        results.append(result)
        print(f"  Year FE only: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 2d. No FE (pooled OLS)
try:
    model = pf.feols("MME_PC ~ treatment",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'did/fe/none', 'methods/difference_in_differences.md#fixed-effects',
                            fixed_effects='none')
    if result:
        results.append(result)
        print(f"  No FE: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 2e. State FE only
try:
    model = pf.feols("MME_PC ~ treatment | state + year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'did/fe/state_year', 'methods/difference_in_differences.md#fixed-effects',
                            fixed_effects='state + year')
    if result:
        results.append(result)
        print(f"  State + Year FE: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 3. CLUSTERING VARIATIONS
# ============================================================================

print("\n3. Running clustering variations...")

# 3a. Robust SE (no clustering)
try:
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov='hetero')
    result = extract_results(model, 'robust/cluster/none', 'robustness/clustering_variations.md',
                            cluster_var='robust_se')
    if result:
        results.append(result)
        print(f"  Robust SE: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 3b. State-level clustering
try:
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'state'})
    result = extract_results(model, 'robust/cluster/state', 'robustness/clustering_variations.md',
                            cluster_var='state')
    if result:
        results.append(result)
        print(f"  State clustering: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 3c. Two-way clustering (county + year)
try:
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': ['county_fips_str', 'year']})
    result = extract_results(model, 'robust/cluster/unit_time', 'robustness/clustering_variations.md',
                            cluster_var='county_fips + year')
    if result:
        results.append(result)
        print(f"  Two-way clustering: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 4. SAMPLE RESTRICTIONS
# ============================================================================

print("\n4. Running sample restrictions...")

# 4a. Exclude Florida (as in paper appendix)
try:
    sample_noFL = main_sample[main_sample['state'] != 'FL'].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_noFL, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/exclude_FL', 'robustness/sample_restrictions.md',
                            sample_desc='Excluding Florida')
    if result:
        results.append(result)
        print(f"  Exclude FL: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"  Error: {e}")

# 4b. Short run sample (rel_year -3 to 1)
try:
    sample_short = affectedcounties[affectedcounties['shortrun'] == 1].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_short, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/short_run', 'robustness/sample_restrictions.md',
                            sample_desc='Short run (rel_year -3 to 1)')
    if result:
        results.append(result)
        print(f"  Short run: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4c. Long run sample (rel_year -3 to 3, but > 1 for treated)
try:
    sample_long = affectedcounties[affectedcounties['longrun'] == 1].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_long, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/long_run', 'robustness/sample_restrictions.md',
                            sample_desc='Long run sample')
    if result:
        results.append(result)
        print(f"  Long run: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4d. Ever suspicious control group
try:
    sample_susp = main_sample[(main_sample['eversuspicious'] == 1) | (main_sample['treatment'] == 1)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_susp, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/eversuspicious_control', 'robustness/sample_restrictions.md',
                            sample_desc='Ever suspicious or treated counties')
    if result:
        results.append(result)
        print(f"  Ever suspicious control: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4e. Above median suspicious control group
try:
    sample_aboveavg = main_sample[(main_sample['sharesuspicious'] > 0) | (main_sample['treatment'] == 1)].copy()
    if len(sample_aboveavg) > 100:
        model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                         data=sample_aboveavg, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, 'robust/sample/aboveavg_suspicious_control', 'robustness/sample_restrictions.md',
                                sample_desc='Above average suspicious or treated')
        if result:
            results.append(result)
            print(f"  Above avg suspicious: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4f. Early treatment years (2007-2010)
try:
    sample_early_treat = main_sample[(main_sample['actionyear'] <= 2010) | (main_sample['actionyear'] == 0)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_early_treat, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/early_treatment', 'robustness/sample_restrictions.md',
                            sample_desc='Early treatment cohorts (2007-2010)')
    if result:
        results.append(result)
        print(f"  Early treatment: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4g. Late treatment years (2011-2014)
try:
    sample_late_treat = main_sample[(main_sample['actionyear'] >= 2011) | (main_sample['actionyear'] == 0)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_late_treat, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/late_treatment', 'robustness/sample_restrictions.md',
                            sample_desc='Late treatment cohorts (2011-2014)')
    if result:
        results.append(result)
        print(f"  Late treatment: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4h. Balanced panel
try:
    # Keep counties observed in all years
    county_obs = main_sample.groupby('county_fips_str')['year'].nunique()
    max_years = county_obs.max()
    balanced_counties = county_obs[county_obs == max_years].index
    sample_balanced = main_sample[main_sample['county_fips_str'].isin(balanced_counties)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_balanced, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/balanced', 'robustness/sample_restrictions.md',
                            sample_desc='Balanced panel')
    if result:
        results.append(result)
        print(f"  Balanced panel: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4i. Trim outliers (1% and 99%)
try:
    q01 = main_sample['MME_PC'].quantile(0.01)
    q99 = main_sample['MME_PC'].quantile(0.99)
    sample_trimmed = main_sample[(main_sample['MME_PC'] > q01) & (main_sample['MME_PC'] < q99)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_trimmed, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/trim_1pct', 'robustness/sample_restrictions.md',
                            sample_desc='Trimmed top/bottom 1%')
    if result:
        results.append(result)
        print(f"  Trimmed 1%: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4j. Winsorize 5%
try:
    sample_wins = main_sample.copy()
    q05 = sample_wins['MME_PC'].quantile(0.05)
    q95 = sample_wins['MME_PC'].quantile(0.95)
    sample_wins['MME_PC_wins'] = sample_wins['MME_PC'].clip(lower=q05, upper=q95)
    model = pf.feols("MME_PC_wins ~ treatment | county_fips_str + state_year",
                     data=sample_wins, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/winsor_5pct', 'robustness/sample_restrictions.md',
                            sample_desc='Winsorized 5%/95%', outcome_var='MME_PC_wins')
    if result:
        results.append(result)
        print(f"  Winsorized 5%: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 4k - 4q. Drop each year one at a time
for drop_year in main_sample['year'].unique():
    try:
        sample_drop = main_sample[main_sample['year'] != drop_year].copy()
        model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                         data=sample_drop, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, f'robust/sample/drop_year_{int(drop_year)}',
                                'robustness/sample_restrictions.md',
                                sample_desc=f'Excluding year {int(drop_year)}')
        if result:
            results.append(result)
            print(f"  Drop year {int(drop_year)}: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error dropping year {drop_year}: {e}")

# 4r-4w. Drop each state with treatment one at a time (top 6 states by treatment)
treated_counties = main_sample[main_sample['treat'] == 1]
top_states = treated_counties.groupby('state')['county_fips'].nunique().nlargest(6).index.tolist()

for drop_state in top_states:
    try:
        sample_drop = main_sample[main_sample['state'] != drop_state].copy()
        model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                         data=sample_drop, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, f'robust/sample/drop_state_{drop_state}',
                                'robustness/sample_restrictions.md',
                                sample_desc=f'Excluding state {drop_state}')
        if result:
            results.append(result)
            print(f"  Drop state {drop_state}: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error dropping state {drop_state}: {e}")

# ============================================================================
# 5. FUNCTIONAL FORM VARIATIONS
# ============================================================================

print("\n5. Running functional form variations...")

# 5a. Log outcome
try:
    sample_log = main_sample[main_sample['MME_PC'] > 0].copy()
    sample_log['log_MME_PC'] = np.log(sample_log['MME_PC'])
    model = pf.feols("log_MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_log, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/funcform/log_outcome', 'robustness/functional_form.md',
                            outcome_var='log(MME_PC)')
    if result:
        results.append(result)
        print(f"  Log outcome: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 5b. IHS transformation
try:
    main_sample['ihs_MME_PC'] = np.arcsinh(main_sample['MME_PC'])
    model = pf.feols("ihs_MME_PC ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/funcform/ihs_outcome', 'robustness/functional_form.md',
                            outcome_var='asinh(MME_PC)')
    if result:
        results.append(result)
        print(f"  IHS outcome: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 5c. Alternative outcome: DOSAGE_UNIT per 100K
try:
    model = pf.feols("opiatesper100K ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/outcome/opiates_per100K', 'robustness/functional_form.md',
                            outcome_var='opiatesper100K')
    if result:
        results.append(result)
        print(f"  Opiates per 100K: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 5d. Log opioid outcome
try:
    sample_log2 = main_sample[main_sample['opiatesper100K'] > 0].copy()
    sample_log2['log_opiates'] = np.log(sample_log2['opiatesper100K'])
    model = pf.feols("log_opiates ~ treatment | county_fips_str + state_year",
                     data=sample_log2, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/funcform/log_opiates', 'robustness/functional_form.md',
                            outcome_var='log(opiatesper100K)')
    if result:
        results.append(result)
        print(f"  Log opiates per 100K: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 5e. Raw PCPV (not multiplied by 10)
try:
    model = pf.feols("PCPV ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/outcome/PCPV', 'robustness/functional_form.md',
                            outcome_var='PCPV')
    if result:
        results.append(result)
        print(f"  Raw PCPV: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 6. HETEROGENEITY ANALYSES
# ============================================================================

print("\n6. Running heterogeneity analyses...")

# 6a. Rural vs Urban
try:
    main_sample['rural_treat'] = main_sample['treatment'] * main_sample['RURAL']
    model = pf.feols("MME_PC ~ treatment + rural_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/rural', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with RURAL indicator')
    if result:
        results.append(result)
        print(f"  Rural interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6b. High Medicare share
try:
    main_sample['high_medicare'] = (main_sample['PCT_MEDICARE'] > main_sample['PCT_MEDICARE'].median()).astype(int)
    main_sample['medicare_treat'] = main_sample['treatment'] * main_sample['high_medicare']
    model = pf.feols("MME_PC ~ treatment + medicare_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/high_medicare', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with high Medicare share')
    if result:
        results.append(result)
        print(f"  Medicare interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6c. High poverty (using unemployment rate as proxy)
try:
    main_sample['high_unemp'] = (main_sample['unemploymentrate'] > main_sample['unemploymentrate'].median()).astype(int)
    main_sample['unemp_treat'] = main_sample['treatment'] * main_sample['high_unemp']
    model = pf.feols("MME_PC ~ treatment + unemp_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/high_unemployment', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with high unemployment')
    if result:
        results.append(result)
        print(f"  Unemployment interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6d. High Black population share
try:
    main_sample['high_black'] = (main_sample['PCT_BLACK'] > main_sample['PCT_BLACK'].median()).astype(int)
    main_sample['black_treat'] = main_sample['treatment'] * main_sample['high_black']
    model = pf.feols("MME_PC ~ treatment + black_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/high_pct_black', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with high Black population share')
    if result:
        results.append(result)
        print(f"  Black pop interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6e. High population density
try:
    main_sample['high_density'] = (main_sample['POP_DENSITY'] > main_sample['POP_DENSITY'].median()).astype(int)
    main_sample['density_treat'] = main_sample['treatment'] * main_sample['high_density']
    model = pf.feols("MME_PC ~ treatment + density_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/high_density', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with high population density')
    if result:
        results.append(result)
        print(f"  Density interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6f. High MD per capita
try:
    main_sample['high_md'] = (main_sample['MD_PC'] > main_sample['MD_PC'].median()).astype(int)
    main_sample['md_treat'] = main_sample['treatment'] * main_sample['high_md']
    model = pf.feols("MME_PC ~ treatment + md_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/high_md_pc', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with high MD per capita')
    if result:
        results.append(result)
        print(f"  MD/capita interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6g. Manufacturing share
try:
    main_sample['high_manuf'] = (main_sample['manufacture_per'] > main_sample['manufacture_per'].median()).astype(int)
    main_sample['manuf_treat'] = main_sample['treatment'] * main_sample['high_manuf']
    model = pf.feols("MME_PC ~ treatment + manuf_treat | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/heterogeneity/high_manufacturing', 'robustness/heterogeneity.md',
                            sample_desc='Interaction with high manufacturing share')
    if result:
        results.append(result)
        print(f"  Manufacturing interaction: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 6h-6k. Subgroup analyses by region
regions_map = {
    'South': ['FL', 'GA', 'AL', 'MS', 'LA', 'TX', 'AR', 'OK', 'TN', 'KY', 'WV', 'VA', 'NC', 'SC'],
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'West': ['MT', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'ID', 'WA', 'OR', 'CA']
}

for region, states in regions_map.items():
    try:
        sample_region = main_sample[main_sample['state'].isin(states)].copy()
        if len(sample_region) > 100:
            model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                             data=sample_region, vcov={'CRV1': 'county_fips_str'})
            result = extract_results(model, f'robust/heterogeneity/region_{region.lower()}',
                                    'robustness/heterogeneity.md',
                                    sample_desc=f'{region} region only')
            if result:
                results.append(result)
                print(f"  {region} only: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for {region}: {e}")

# ============================================================================
# 7. CONTROL VARIABLE VARIATIONS
# ============================================================================

print("\n7. Running control variable variations...")

# Available controls
controls = ['RURAL', 'PCT_MEDICARE', 'PCT_BLACK', 'unemploymentrate',
            'MD_PC', 'POP_DENSITY', 'manufacture_per']

# 7a-g. Add each control individually
for control in controls:
    try:
        sample_ctrl = main_sample.dropna(subset=[control]).copy()
        model = pf.feols(f"MME_PC ~ treatment + {control} | county_fips_str + state_year",
                         data=sample_ctrl, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, f'robust/control/add_{control}',
                                'robustness/leave_one_out.md',
                                controls_desc=control)
        if result:
            results.append(result)
            print(f"  Add {control}: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error adding {control}: {e}")

# 7h. All controls together
try:
    sample_all_ctrl = main_sample.dropna(subset=controls).copy()
    ctrl_formula = " + ".join(controls)
    model = pf.feols(f"MME_PC ~ treatment + {ctrl_formula} | county_fips_str + state_year",
                     data=sample_all_ctrl, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/control/all_controls',
                            'robustness/leave_one_out.md',
                            controls_desc='All available controls')
    if result:
        results.append(result)
        print(f"  All controls: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error with all controls: {e}")

# 7i-o. Leave one out (from all controls)
for drop_ctrl in controls:
    try:
        remaining_controls = [c for c in controls if c != drop_ctrl]
        sample_loo = main_sample.dropna(subset=remaining_controls).copy()
        ctrl_formula = " + ".join(remaining_controls)
        model = pf.feols(f"MME_PC ~ treatment + {ctrl_formula} | county_fips_str + state_year",
                         data=sample_loo, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, f'robust/control/drop_{drop_ctrl}',
                                'robustness/leave_one_out.md',
                                controls_desc=f'All controls except {drop_ctrl}')
        if result:
            results.append(result)
            print(f"  Drop {drop_ctrl}: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error dropping {drop_ctrl}: {e}")

# ============================================================================
# 8. PLACEBO AND VALIDATION TESTS
# ============================================================================

print("\n8. Running placebo and validation tests...")

# 8a. Pre-treatment trends test (only pre-treatment periods)
try:
    sample_pre = affectedcounties[(affectedcounties['rel_year'] >= -3) &
                                  (affectedcounties['rel_year'] < 0)].copy()
    sample_pre['fake_treat'] = (sample_pre['rel_year'] >= -1).astype(int)
    model = pf.feols("MME_PC ~ fake_treat | county_fips_str + state_year",
                     data=sample_pre, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/placebo/pre_treatment', 'robustness/placebo_tests.md',
                            sample_desc='Pre-treatment periods only (-3 to -1)',
                            treatment_var='fake_treat')
    if result:
        results.append(result)
        print(f"  Pre-treatment placebo: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 8b. Fake treatment timing (shift treatment 2 years earlier)
try:
    sample_fake = affectedcounties.copy()
    sample_fake['fake_actionyear'] = np.where(
        sample_fake['actionyear'] > 0,
        sample_fake['actionyear'] - 2,
        0
    )
    sample_fake['fake_rel_year'] = np.where(
        sample_fake['fake_actionyear'] > 0,
        sample_fake['year'] - sample_fake['fake_actionyear'],
        0
    )
    sample_fake['fake_treatment'] = np.where(
        (sample_fake['fake_rel_year'] >= 0) & (sample_fake['treat'] == 1),
        1, 0
    )
    sample_fake = sample_fake[(sample_fake['fake_rel_year'] >= -3) &
                              (sample_fake['fake_rel_year'] <= 3)].copy()
    model = pf.feols("MME_PC ~ fake_treatment | county_fips_str + state_year",
                     data=sample_fake, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/placebo/fake_timing_2yr_early', 'robustness/placebo_tests.md',
                            sample_desc='Fake treatment timing (2 years early)',
                            treatment_var='fake_treatment')
    if result:
        results.append(result)
        print(f"  Fake timing (2yr early): coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 8c. Unaffected outcome test (using HPSA as placebo - should not be affected)
try:
    model = pf.feols("HPSA_WHOLE ~ treatment | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/placebo/unaffected_outcome_HPSA', 'robustness/placebo_tests.md',
                            outcome_var='HPSA_WHOLE',
                            sample_desc='Placebo: HPSA designation')
    if result:
        results.append(result)
        print(f"  Placebo outcome (HPSA): coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 9. ADDITIONAL ROBUSTNESS CHECKS
# ============================================================================

print("\n9. Running additional robustness checks...")

# 9a. Population weighted
try:
    sample_pop = main_sample.dropna(subset=['populationAHRF']).copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_pop, weights=sample_pop['populationAHRF'],
                     vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/weights/population', 'robustness/sample_restrictions.md',
                            sample_desc='Population weighted')
    if result:
        results.append(result)
        print(f"  Pop weighted: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9b. Large counties only (above median population)
try:
    median_pop = main_sample['populationAHRF'].median()
    sample_large = main_sample[main_sample['populationAHRF'] > median_pop].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_large, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/large_counties', 'robustness/sample_restrictions.md',
                            sample_desc='Above median population counties')
    if result:
        results.append(result)
        print(f"  Large counties: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9c. Small counties only (below median population)
try:
    sample_small = main_sample[main_sample['populationAHRF'] <= median_pop].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_small, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/small_counties', 'robustness/sample_restrictions.md',
                            sample_desc='Below median population counties')
    if result:
        results.append(result)
        print(f"  Small counties: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9d. High baseline opioid counties
try:
    baseline_opioids = main_sample.groupby('county_fips')['MME_PC'].first()
    high_opioid_counties = baseline_opioids[baseline_opioids > baseline_opioids.median()].index
    sample_high = main_sample[main_sample['county_fips'].isin(high_opioid_counties)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_high, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/high_baseline_opioids', 'robustness/sample_restrictions.md',
                            sample_desc='High baseline opioid counties')
    if result:
        results.append(result)
        print(f"  High baseline opioids: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9e. Low baseline opioid counties
try:
    low_opioid_counties = baseline_opioids[baseline_opioids <= baseline_opioids.median()].index
    sample_low = main_sample[main_sample['county_fips'].isin(low_opioid_counties)].copy()
    model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                     data=sample_low, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/sample/low_baseline_opioids', 'robustness/sample_restrictions.md',
                            sample_desc='Low baseline opioid counties')
    if result:
        results.append(result)
        print(f"  Low baseline opioids: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9f. PDMP states only
try:
    sample_pdmp = main_sample[main_sample['PDMPeffective'] == 1].copy()
    if len(sample_pdmp) > 100:
        model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                         data=sample_pdmp, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, 'robust/sample/pdmp_states', 'robustness/sample_restrictions.md',
                                sample_desc='PDMP states only')
        if result:
            results.append(result)
            print(f"  PDMP states: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9g. Non-PDMP states only
try:
    sample_no_pdmp = main_sample[main_sample['PDMPeffective'] == 0].copy()
    if len(sample_no_pdmp) > 100:
        model = pf.feols("MME_PC ~ treatment | county_fips_str + state_year",
                         data=sample_no_pdmp, vcov={'CRV1': 'county_fips_str'})
        result = extract_results(model, 'robust/sample/non_pdmp_states', 'robustness/sample_restrictions.md',
                                sample_desc='Non-PDMP states only')
        if result:
            results.append(result)
            print(f"  Non-PDMP states: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# 9h. Control for PDMP
try:
    model = pf.feols("MME_PC ~ treatment + PDMPeffective | county_fips_str + state_year",
                     data=main_sample, vcov={'CRV1': 'county_fips_str'})
    result = extract_results(model, 'robust/control/add_PDMP', 'robustness/leave_one_out.md',
                            controls_desc='PDMPeffective')
    if result:
        results.append(result)
        print(f"  Control for PDMP: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n{'='*60}")
print(f"TOTAL SPECIFICATIONS RUN: {len(results)}")
print(f"{'='*60}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Add estimation script path
results_df['estimation_script'] = f'scripts/paper_analyses/{PAPER_ID}.py'

# Save to CSV
output_file = OUTPUT_PATH / 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({(results_df['coefficient'] < 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"\nCoefficient range: [{results_df['coefficient'].min():.3f}, {results_df['coefficient'].max():.3f}]")
print(f"Median coefficient: {results_df['coefficient'].median():.3f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.3f}")

# Breakdown by category
print("\n" + "-"*60)
print("BREAKDOWN BY CATEGORY")
print("-"*60)

def categorize_spec(spec_id):
    if spec_id == 'baseline':
        return 'Baseline'
    elif spec_id.startswith('did/fe'):
        return 'FE variations'
    elif spec_id.startswith('robust/cluster'):
        return 'Clustering'
    elif spec_id.startswith('robust/sample'):
        return 'Sample restrictions'
    elif spec_id.startswith('robust/funcform') or spec_id.startswith('robust/outcome'):
        return 'Functional form'
    elif spec_id.startswith('robust/heterogeneity'):
        return 'Heterogeneity'
    elif spec_id.startswith('robust/control'):
        return 'Control variations'
    elif spec_id.startswith('robust/placebo'):
        return 'Placebo tests'
    elif spec_id.startswith('robust/weights'):
        return 'Weights'
    else:
        return 'Other'

results_df['category'] = results_df['spec_id'].apply(categorize_spec)

category_summary = results_df.groupby('category').agg({
    'spec_id': 'count',
    'coefficient': ['mean', 'std'],
    'p_value': lambda x: (x < 0.05).mean() * 100
}).round(3)
category_summary.columns = ['N', 'Mean Coef', 'Std Coef', '% Sig 5%']
print(category_summary)

print("\nSpecification search complete!")
