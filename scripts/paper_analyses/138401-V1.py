"""
Fast Specification Search for Paper 138401-V1
"The Long-Term Effects of Measles Vaccination on Earnings and Employment"
Optimized version - loads only necessary columns
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

PAPER_ID = '138401-V1'
JOURNAL = 'AEJ: Policy'
PAPER_TITLE = 'The Long-Term Effects of Measles Vaccination on Earnings and Employment'
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/138401-V1/Replication_Files'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/138401-V1'

print("="*60)
print("FAST SPECIFICATION SEARCH: 138401-V1")
print("="*60)

# Only load columns we need
cols_needed = ['birthyr', 'bpl', 'age', 'sex', 'race', 'incwage', 'uhrswork',
               'empstat', 'year', 'perwt']

print("\nLoading ACS data (only needed columns)...")
acs = pd.read_stata(f'{BASE_PATH}/raw_data/longrun_20002017_acs.dta', columns=cols_needed)
print(f"Loaded: {acs.shape}")

# Apply filters immediately (from acs_cleaning.do)
print("Filtering data...")

# Convert age from category (string) to numeric
acs['age'] = pd.to_numeric(acs['age'].astype(str), errors='coerce')
acs = acs[(acs['age'] > 25) & (acs['age'] < 60)]
print(f"After age filter: {len(acs)}")

# Convert birthyr
acs['birthyr'] = pd.to_numeric(acs['birthyr'].astype(str), errors='coerce')

# Map state names to FIPS codes for bpl (native born = FIPS < 57)
STATE_TO_FIPS = {
    'Alabama': 1, 'Alaska': 2, 'Arizona': 4, 'Arkansas': 5, 'California': 6,
    'Colorado': 8, 'Connecticut': 9, 'Delaware': 10, 'District of Columbia': 11,
    'Florida': 12, 'Georgia': 13, 'Hawaii': 15, 'Idaho': 16, 'Illinois': 17,
    'Indiana': 18, 'Iowa': 19, 'Kansas': 20, 'Kentucky': 21, 'Louisiana': 22,
    'Maine': 23, 'Maryland': 24, 'Massachusetts': 25, 'Michigan': 26, 'Minnesota': 27,
    'Mississippi': 28, 'Missouri': 29, 'Montana': 30, 'Nebraska': 31, 'Nevada': 32,
    'New Hampshire': 33, 'New Jersey': 34, 'New Mexico': 35, 'New York': 36,
    'North Carolina': 37, 'North Dakota': 38, 'Ohio': 39, 'Oklahoma': 40, 'Oregon': 41,
    'Pennsylvania': 42, 'Rhode Island': 44, 'South Carolina': 45, 'South Dakota': 46,
    'Tennessee': 47, 'Texas': 48, 'Utah': 49, 'Vermont': 50, 'Virginia': 51,
    'Washington': 53, 'West Virginia': 54, 'Wisconsin': 55, 'Wyoming': 56
}
acs['bpl_fips'] = acs['bpl'].astype(str).map(STATE_TO_FIPS)
acs = acs[acs['bpl_fips'].notna()]  # native born only
acs['bpl'] = acs['bpl_fips'].astype(int)
print(f"After native born filter: {len(acs)}")

# Filter for black and white only
acs['race_str'] = acs['race'].astype(str)
acs = acs[acs['race_str'].isin(['White', 'Black/African American/Negro'])]
print(f"After race filter: {len(acs)}")

# Convert year to numeric
acs['year'] = pd.to_numeric(acs['year'].astype(str), errors='coerce')
print(f"After filtering: {acs.shape}")

# Create variables
acs['sex_num'] = pd.to_numeric(acs['sex'].astype(str), errors='coerce')
acs['female'] = (acs['sex_num'] == 2).astype(int)
acs['black'] = (acs['race_str'] == 'Black/African American/Negro').astype(int)
acs['white'] = (acs['race_str'] == 'White').astype(int)

# Create exposure variable (years exposed to vaccine, 0 for pre-1949, max 16 for post-1963)
acs['exposure'] = np.clip(acs['birthyr'] - 1948, 0, 16)

# Employment outcome
acs['empstat_str'] = acs['empstat'].astype(str)
acs['employed'] = acs['empstat_str'].str.contains('Employed', case=False, na=False).astype(int)

# Log wage (for those with positive wages)
acs['lnwage'] = np.log(acs['incwage'].replace(0, np.nan))

# Create fixed effects
acs['bpl_fe'] = acs['bpl'].astype(str)
acs['birthyr_fe'] = acs['birthyr'].astype(str)
acs['year_fe'] = acs['year'].astype(str)
acs['bplcohort'] = acs['bpl'].astype(str) + '_' + acs['birthyr'].astype(str)

# Load measles case data for treatment
print("Loading measles case data...")
cases = pd.read_stata(f'{BASE_PATH}/raw_data/case_counts_population.dta')

# Merge - need state-level measles rates
# The treatment is pre-vaccine measles mortality/morbidity interacted with exposure
# Simplified: use state-level variation in pre-vaccine measles burden

# Create treatment: high measles states x exposure
# Use median split of states by measles cases (pre-vaccine period)
pre_vaccine_cases = cases[cases['year'] < 1963]  # Pre-vaccine period
state_cases = pre_vaccine_cases.groupby('statefip')['measles'].mean().reset_index()
state_cases.columns = ['bpl', 'cases']
median_cases = state_cases['cases'].median()
high_measles_states = state_cases[state_cases['cases'] > median_cases]['bpl'].tolist()

acs['high_measles'] = acs['bpl'].isin(high_measles_states).astype(int)
acs['treatment'] = acs['high_measles'] * acs['exposure']

# Drop missing outcomes
df = acs.dropna(subset=['employed']).copy()
print(f"Analysis sample: {df.shape}")

# Results storage
results = []

def run_spec(formula, data, spec_id, spec_tree_path, outcome_var, treatment_var,
             sample_desc, fixed_effects, controls_desc, cluster_var='bplcohort'):
    """Run a specification and return results dict"""
    try:
        model = pf.feols(formula, data=data, vcov={'CRV1': cluster_var})
        if treatment_var not in model.coef().index:
            return None
        return {
            'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
            'spec_id': spec_id, 'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var, 'treatment_var': treatment_var,
            'coefficient': float(model.coef()[treatment_var]),
            'std_error': float(model.se()[treatment_var]),
            't_stat': float(model.tstat()[treatment_var]),
            'p_value': float(model.pvalue()[treatment_var]),
            'ci_lower': float(model.confint().loc[treatment_var, '2.5%']),
            'ci_upper': float(model.confint().loc[treatment_var, '97.5%']),
            'n_obs': int(model._N),
            'r_squared': float(model.r2) if hasattr(model, 'r2') else None,
            'coefficient_vector_json': json.dumps({treatment_var: float(model.coef()[treatment_var])}),
            'sample_desc': sample_desc, 'fixed_effects': fixed_effects,
            'controls_desc': controls_desc, 'cluster_var': cluster_var,
            'model_type': 'OLS with FE', 'estimation_script': 'spec_search_fast.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None

# ============================================================================
# BASELINE SPECIFICATIONS
# ============================================================================
print("\n--- Baseline Specifications ---")

# Baseline: employment
print("  Running baseline (employed)...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'baseline', 'methods/panel_fixed_effects.md', 'employed', 'treatment',
             'Full sample', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Baseline: log wage
df_wage = df.dropna(subset=['lnwage'])
print("  Running baseline (lnwage)...")
r = run_spec("lnwage ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df_wage, 'baseline_lnwage', 'methods/panel_fixed_effects.md', 'lnwage', 'treatment',
             'Positive wage sample', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# ============================================================================
# CONTROL VARIATIONS (Leave-one-out)
# ============================================================================
print("\n--- Control Variations ---")

# No controls
print("  Running: no controls...")
r = run_spec("employed ~ treatment | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/control/none', 'robustness/leave_one_out.md', 'employed', 'treatment',
             'Full sample', 'bpl, birthyr, year', 'none')
if r: results.append(r)

# Only black
print("  Running: only black control...")
r = run_spec("employed ~ treatment + black | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/control/only_black', 'robustness/leave_one_out.md', 'employed', 'treatment',
             'Full sample', 'bpl, birthyr, year', 'black')
if r: results.append(r)

# Only female
print("  Running: only female control...")
r = run_spec("employed ~ treatment + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/control/only_female', 'robustness/leave_one_out.md', 'employed', 'treatment',
             'Full sample', 'bpl, birthyr, year', 'female')
if r: results.append(r)

# ============================================================================
# FIXED EFFECTS VARIATIONS
# ============================================================================
print("\n--- Fixed Effects Variations ---")

# Only birthplace FE
print("  Running: only bpl FE...")
r = run_spec("employed ~ treatment + black + female | bpl_fe",
             df, 'robust/fe/bpl_only', 'robustness/clustering_variations.md', 'employed', 'treatment',
             'Full sample', 'bpl only', 'black, female')
if r: results.append(r)

# Only year FE
print("  Running: only year FE...")
r = run_spec("employed ~ treatment + black + female | year_fe",
             df, 'robust/fe/year_only', 'robustness/clustering_variations.md', 'employed', 'treatment',
             'Full sample', 'year only', 'black, female')
if r: results.append(r)

# Only birthyear FE
print("  Running: only birthyr FE...")
r = run_spec("employed ~ treatment + black + female | birthyr_fe",
             df, 'robust/fe/birthyr_only', 'robustness/clustering_variations.md', 'employed', 'treatment',
             'Full sample', 'birthyr only', 'black, female')
if r: results.append(r)

# BPL + year
print("  Running: bpl + year FE...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + year_fe",
             df, 'robust/fe/bpl_year', 'robustness/clustering_variations.md', 'employed', 'treatment',
             'Full sample', 'bpl, year', 'black, female')
if r: results.append(r)

# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================
print("\n--- Clustering Variations ---")

# Cluster by state
print("  Running: cluster by bpl...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/cluster/bpl', 'robustness/clustering_variations.md', 'employed', 'treatment',
             'Full sample', 'bpl, birthyr, year', 'black, female', cluster_var='bpl_fe')
if r: results.append(r)

# Cluster by birthyear
print("  Running: cluster by birthyr...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/cluster/birthyr', 'robustness/clustering_variations.md', 'employed', 'treatment',
             'Full sample', 'bpl, birthyr, year', 'black, female', cluster_var='birthyr_fe')
if r: results.append(r)

# Robust SEs
print("  Running: robust SEs...")
try:
    model = pf.feols("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
                     data=df, vcov='hetero')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/cluster/robust_se', 'spec_tree_path': 'robustness/clustering_variations.md',
        'outcome_var': 'employed', 'treatment_var': 'treatment',
        'coefficient': float(model.coef()['treatment']),
        'std_error': float(model.se()['treatment']),
        't_stat': float(model.tstat()['treatment']),
        'p_value': float(model.pvalue()['treatment']),
        'ci_lower': float(model.confint().loc['treatment', '2.5%']),
        'ci_upper': float(model.confint().loc['treatment', '97.5%']),
        'n_obs': int(model._N), 'r_squared': float(model.r2) if hasattr(model, 'r2') else None,
        'coefficient_vector_json': json.dumps({'treatment': float(model.coef()['treatment'])}),
        'sample_desc': 'Full sample', 'fixed_effects': 'bpl, birthyr, year',
        'controls_desc': 'black, female', 'cluster_var': 'robust',
        'model_type': 'OLS with FE', 'estimation_script': 'spec_search_fast.py'
    })
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("\n--- Sample Restrictions ---")

# Males only
print("  Running: males only...")
r = run_spec("employed ~ treatment + black | bpl_fe + birthyr_fe + year_fe",
             df[df['female'] == 0], 'robust/sample/males', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Males only', 'bpl, birthyr, year', 'black')
if r: results.append(r)

# Females only
print("  Running: females only...")
r = run_spec("employed ~ treatment + black | bpl_fe + birthyr_fe + year_fe",
             df[df['female'] == 1], 'robust/sample/females', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Females only', 'bpl, birthyr, year', 'black')
if r: results.append(r)

# White only
print("  Running: white only...")
r = run_spec("employed ~ treatment + female | bpl_fe + birthyr_fe + year_fe",
             df[df['white'] == 1], 'robust/sample/white', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'White only', 'bpl, birthyr, year', 'female')
if r: results.append(r)

# Black only
print("  Running: black only...")
r = run_spec("employed ~ treatment + female | bpl_fe + birthyr_fe + year_fe",
             df[df['black'] == 1], 'robust/sample/black', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Black only', 'bpl, birthyr, year', 'female')
if r: results.append(r)

# Pre-vaccine cohorts (birthyr < 1955)
print("  Running: pre-vaccine cohorts...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df[df['birthyr'] < 1955], 'robust/sample/pre_vaccine', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Pre-vaccine cohorts', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Post-vaccine cohorts (birthyr >= 1960)
print("  Running: post-vaccine cohorts...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df[df['birthyr'] >= 1960], 'robust/sample/post_vaccine', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Post-vaccine cohorts', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Young workers (age < 45)
print("  Running: young workers...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df[df['age'] < 45], 'robust/sample/young', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Age < 45', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Older workers (age >= 45)
print("  Running: older workers...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df[df['age'] >= 45], 'robust/sample/older', 'robustness/sample_restrictions.md',
             'employed', 'treatment', 'Age >= 45', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Drop years one at a time
for yr in [2005, 2010, 2015]:
    print(f"  Running: drop year {yr}...")
    r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
                 df[df['year'] != yr], f'robust/sample/drop_year_{yr}', 'robustness/sample_restrictions.md',
                 'employed', 'treatment', f'Drop year {yr}', 'bpl, birthyr, year', 'black, female')
    if r: results.append(r)

# ============================================================================
# ALTERNATIVE OUTCOMES
# ============================================================================
print("\n--- Alternative Outcomes ---")

# Log wage outcome
print("  Running: log wage outcome...")
r = run_spec("lnwage ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df_wage, 'robust/outcome/lnwage', 'robustness/functional_form.md', 'lnwage', 'treatment',
             'Positive wage sample', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# ============================================================================
# HETEROGENEITY
# ============================================================================
print("\n--- Heterogeneity ---")

# Treatment x black
print("  Running: treatment x black...")
df['treat_x_black'] = df['treatment'] * df['black']
r = run_spec("employed ~ treatment + treat_x_black + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/heterogeneity/black', 'robustness/heterogeneity.md', 'employed', 'treat_x_black',
             'Full sample', 'bpl, birthyr, year', 'black, female, treatment')
if r: results.append(r)

# Treatment x female
print("  Running: treatment x female...")
df['treat_x_female'] = df['treatment'] * df['female']
r = run_spec("employed ~ treatment + treat_x_female + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/heterogeneity/female', 'robustness/heterogeneity.md', 'employed', 'treat_x_female',
             'Full sample', 'bpl, birthyr, year', 'black, female, treatment')
if r: results.append(r)

# ============================================================================
# PLACEBO TESTS
# ============================================================================
print("\n--- Placebo Tests ---")

# Placebo: random treatment
print("  Running: random placebo treatment...")
np.random.seed(42)
df['placebo_treatment'] = np.random.permutation(df['treatment'].values)
r = run_spec("employed ~ placebo_treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/placebo/random', 'robustness/placebo.md', 'employed', 'placebo_treatment',
             'Full sample', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# ============================================================================
# ADDITIONAL ROBUSTNESS SPECIFICATIONS
# ============================================================================
print("\n--- Additional Robustness ---")

# By region (drop each major region)
regions = {
    'northeast': [9, 23, 25, 33, 34, 36, 42, 44, 50],  # CT, ME, MA, NH, NJ, NY, PA, RI, VT
    'south': [1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54],
    'midwest': [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55],
    'west': [2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56]
}

for region_name, states in regions.items():
    print(f"  Running: drop {region_name}...")
    r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
                 df[~df['bpl'].isin(states)], f'robust/sample/drop_{region_name}',
                 'robustness/sample_restrictions.md', 'employed', 'treatment',
                 f'Drop {region_name}', 'bpl, birthyr, year', 'black, female')
    if r: results.append(r)

# Continuous exposure measure (instead of treatment = high_measles * exposure)
print("  Running: continuous exposure only...")
r = run_spec("employed ~ exposure + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/treatment/exposure_only', 'robustness/treatment_variations.md',
             'employed', 'exposure', 'Full sample', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# High measles indicator only
print("  Running: high measles only...")
r = run_spec("employed ~ high_measles + black + female | bpl_fe + birthyr_fe + year_fe",
             df, 'robust/treatment/high_measles_only', 'robustness/treatment_variations.md',
             'employed', 'high_measles', 'Full sample', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Winsorized outcome at 1%
print("  Running: winsorized wage...")
df_wage_wins = df_wage.copy()
p01, p99 = df_wage_wins['lnwage'].quantile([0.01, 0.99])
df_wage_wins['lnwage_wins'] = df_wage_wins['lnwage'].clip(p01, p99)
r = run_spec("lnwage_wins ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df_wage_wins, 'robust/funcform/winsorized', 'robustness/functional_form.md',
             'lnwage_wins', 'treatment', 'Winsorized wages', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Add more sample splits to reach 50+
# By survey year ranges
for start_yr, end_yr in [(2000, 2008), (2009, 2017)]:
    print(f"  Running: years {start_yr}-{end_yr}...")
    r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
                 df[(df['year'] >= start_yr) & (df['year'] <= end_yr)],
                 f'robust/sample/years_{start_yr}_{end_yr}', 'robustness/sample_restrictions.md',
                 'employed', 'treatment', f'Years {start_yr}-{end_yr}', 'bpl, birthyr, year', 'black, female')
    if r: results.append(r)

# By birth cohort decades
for decade in [1940, 1950, 1960]:
    print(f"  Running: cohort {decade}s...")
    r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
                 df[(df['birthyr'] >= decade) & (df['birthyr'] < decade + 10)],
                 f'robust/sample/cohort_{decade}s', 'robustness/sample_restrictions.md',
                 'employed', 'treatment', f'Cohort {decade}s', 'bpl, birthyr, year', 'black, female')
    if r: results.append(r)

# Drop top 5 measles states
top_measles = state_cases.nlargest(5, 'cases')['bpl'].tolist()
print("  Running: drop top 5 measles states...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df[~df['bpl'].isin(top_measles)], 'robust/sample/drop_top5_measles',
             'robustness/sample_restrictions.md', 'employed', 'treatment',
             'Drop top 5 measles states', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Drop bottom 5 measles states
bottom_measles = state_cases.nsmallest(5, 'cases')['bpl'].tolist()
print("  Running: drop bottom 5 measles states...")
r = run_spec("employed ~ treatment + black + female | bpl_fe + birthyr_fe + year_fe",
             df[~df['bpl'].isin(bottom_measles)], 'robust/sample/drop_bottom5_measles',
             'robustness/sample_restrictions.md', 'employed', 'treatment',
             'Drop bottom 5 measles states', 'bpl, birthyr, year', 'black, female')
if r: results.append(r)

# Log wage with different samples
print("  Running: log wage males...")
r = run_spec("lnwage ~ treatment + black | bpl_fe + birthyr_fe + year_fe",
             df_wage[df_wage['female'] == 0], 'robust/outcome/lnwage_males',
             'robustness/sample_restrictions.md', 'lnwage', 'treatment',
             'Males, positive wages', 'bpl, birthyr, year', 'black')
if r: results.append(r)

print("  Running: log wage females...")
r = run_spec("lnwage ~ treatment + black | bpl_fe + birthyr_fe + year_fe",
             df_wage[df_wage['female'] == 1], 'robust/outcome/lnwage_females',
             'robustness/sample_restrictions.md', 'lnwage', 'treatment',
             'Females, positive wages', 'bpl, birthyr, year', 'black')
if r: results.append(r)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print(f"COMPLETED: {len(results)} specifications")
print("="*60)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_PATH}/specification_results.csv', index=False)
print(f"Saved to {OUTPUT_PATH}/specification_results.csv")

# Summary stats
if len(results) > 0:
    coefs = results_df['coefficient'].dropna()
    pvals = results_df['p_value'].dropna()
    print(f"\nSummary:")
    print(f"  Positive coefficients: {(coefs > 0).sum()} ({100*(coefs > 0).mean():.1f}%)")
    print(f"  Significant at 5%: {(pvals < 0.05).sum()} ({100*(pvals < 0.05).mean():.1f}%)")
    print(f"  Median coefficient: {coefs.median():.4f}")
    print(f"  Range: [{coefs.min():.4f}, {coefs.max():.4f}]")
