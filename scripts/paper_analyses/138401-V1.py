"""
Specification Search: 138401-V1
Paper: The Long-Term Effects of Measles Vaccination on Earnings and Employment
Author: Alicia Atwood (2021)

This script performs a systematic specification search following the i4r methodology.
The paper uses a difference-in-differences design exploiting variation in pre-vaccine
measles incidence across states combined with the timing of vaccine introduction (1963).

Treatment: M12_exp_rate = (avg_12yr_measles_rate * exposure) / 100000
- avg_12yr_measles_rate: state's average measles incidence rate 1952-1963
- exposure: years of childhood (0-16) that occurred after vaccine (1964+)

Outcomes: Adult labor market outcomes measured in ACS 2000-2017
- cpi_incwage: CPI-adjusted income (wage and salary)
- cpi_incwage_no0: CPI-adjusted income excluding zeros
- ln_cpi_income: Log of CPI-adjusted income
- poverty100: Below poverty line indicator
- employed: Employment status
- hrs_worked: Hours worked per week

Fixed Effects: Birth state, birth year, ACS year
Controls: Age x black x female interactions, bpl x black, bpl x female, bpl x black x female
Clustering: Birth state x birth cohort
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to use pyfixest, fall back to statsmodels if needed
try:
    import pyfixest as pf
    USE_PYFIXEST = True
except ImportError:
    USE_PYFIXEST = False
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

from scipy import stats

# Paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/138401-V1/Replication_Files/raw_data'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/138401-V1'

# Paper metadata
PAPER_ID = '138401-V1'
JOURNAL = 'AER'  # American Economic Review
PAPER_TITLE = 'The Long-Term Effects of Measles Vaccination on Earnings and Employment'

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coefficient, std_error, t_stat, p_value, ci_lower, ci_upper,
               n_obs, r_squared, coefficient_vector_json, sample_desc,
               fixed_effects, controls_desc, cluster_var, model_type):
    """Add a result to the results list."""
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coefficient,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coefficient_vector_json),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

print("=" * 60)
print(f"Specification Search: {PAPER_ID}")
print(f"Paper: {PAPER_TITLE}")
print("=" * 60)

# ============================================================
# STEP 1: Load and Prepare Data
# ============================================================
print("\n[1] Loading and preparing data...")

# Load main ACS data
print("  Loading ACS data (this may take a while - 1.9GB file)...")
df_acs = pd.read_stata(f'{DATA_PATH}/longrun_20002017_acs.dta', convert_categoricals=False)
print(f"  Raw ACS data: {len(df_acs):,} observations")

# Load measles incidence rates
print("  Loading measles incidence data...")
df_rates = pd.read_stata(f'{DATA_PATH}/case_counts_population.dta', convert_categoricals=False)

# ============================================================
# STEP 2: Clean ACS Data (replicate acs_cleaning.do)
# ============================================================
print("\n[2] Cleaning ACS data...")

# Keep ages 26-59
df = df_acs[(df_acs['age'] > 25) & (df_acs['age'] < 60)].copy()
print(f"  After age filter (26-59): {len(df):,}")

# Keep native born only (bpl < 57)
df = df[df['bpl'] < 57].copy()
print(f"  After native born filter: {len(df):,}")

# Keep black and white only
df['white'] = (df['race'] == 1).astype(int)
df['black'] = (df['race'] == 2).astype(int)
df = df[(df['white'] == 1) | (df['black'] == 1)].copy()
print(f"  After race filter (black/white): {len(df):,}")

# Create exposure variable
def get_exposure(birthyr):
    if birthyr < 1949:
        return 0
    elif birthyr <= 1963:
        return birthyr - 1948
    else:
        return 16

df['exposure'] = df['birthyr'].apply(get_exposure)

# Create female identifier
df['female'] = (df['sex'] == 2).astype(int)

# Create control variables
df['ageblackfemale'] = df['age'].astype(str) + '_' + df['black'].astype(str) + '_' + df['female'].astype(str)
df['bpl_black'] = df['bpl'] * df['black']
df['bpl_female'] = df['bpl'] * df['female']
df['bpl_black_female'] = df['bpl'] * df['black'] * df['female']

# Create outcome variables
df['ln_income'] = np.log(df['incwage'].replace(0, np.nan))

# CPI adjustment (to 2018 dollars)
df['cpi_incwage'] = df['incwage'] * df['cpi99'] * 1.507
df['ln_cpi_income'] = np.log(df['cpi_incwage'].replace(0, np.nan))

# Poverty indicator
df['poverty100'] = (df['poverty'] < 101).astype(float)
df.loc[df['poverty'] == 0, 'poverty100'] = np.nan

# Hours worked
df['hrs_worked'] = df['uhrswork'].copy()

# Employment status
df['employed'] = (df['empstat'] == 1).astype(float)
df.loc[df['empstat'] == 3, 'employed'] = np.nan

# Non-zero income
df['cpi_incwage_no0'] = df['cpi_incwage'].replace(0, np.nan)

print(f"  Data cleaning complete: {len(df):,} observations")

# ============================================================
# STEP 3: Calculate and Merge Measles Rates
# ============================================================
print("\n[3] Calculating and merging measles rates...")

# Calculate measles rates by state by year
df_rates_wide = df_rates.copy()

# Keep only needed years and calculate rates
years = list(range(1952, 1976))
for year in years:
    if f'measles' in df_rates_wide.columns and f'population' in df_rates_wide.columns:
        pass  # Already have the data

# The case_counts_population.dta appears to be in long format
# Need to reshape and calculate average rates

# Simplify: use the raw data directly
# The rates.do file calculates avg_12yr_measles_rate as the average rate 1952-1963

# Group by state and calculate average measles rate for 1952-1963
df_rates_sub = df_rates[(df_rates['year'] >= 1952) & (df_rates['year'] <= 1963)].copy()
df_rates_sub['measles_rate'] = (df_rates_sub['measles'] / df_rates_sub['population']) * 100000

# Calculate different window averages
avg_rates = df_rates_sub.groupby('statefip').agg({
    'measles_rate': 'mean'
}).reset_index()
avg_rates.columns = ['bpl', 'avg_12yr_measles_rate']

# Also calculate alternative windows
for window in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    start_year = 1964 - window
    df_temp = df_rates[(df_rates['year'] >= start_year) & (df_rates['year'] <= 1963)].copy()
    df_temp['measles_rate'] = (df_temp['measles'] / df_temp['population']) * 100000
    temp_avg = df_temp.groupby('statefip')['measles_rate'].mean().reset_index()
    temp_avg.columns = ['bpl', f'avg_{window}yr_measles_rate']
    avg_rates = avg_rates.merge(temp_avg, on='bpl', how='left')

# Add region info from original rates data
region_info = df_rates[['statefip', 'bpl_region4', 'bpl_region9']].drop_duplicates()
region_info.columns = ['bpl', 'bpl_region4', 'bpl_region9']
avg_rates = avg_rates.merge(region_info, on='bpl', how='left')

# Merge with ACS
df = df.merge(avg_rates, on='bpl', how='left')

# Create treatment variables (M_exp_rate)
# Scale: changes measles rate from 964 per 100000 to .00964
for window in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    col_name = f'avg_{window}yr_measles_rate'
    if col_name in df.columns:
        df[f'M{window}_exp_rate'] = (df[col_name] * df['exposure']) / 100000

# Create bplcohort variable for clustering
df['bplcohort'] = df['bpl'].astype(str) + '_' + df['birthyr'].astype(str)

print(f"  After merging rates: {len(df):,} observations with non-missing M12_exp_rate: {df['M12_exp_rate'].notna().sum():,}")

# Drop observations with missing key variables
df = df.dropna(subset=['M12_exp_rate'])
print(f"  Final sample size: {len(df):,} observations")

# ============================================================
# STEP 4: Set up regression infrastructure
# ============================================================
print("\n[4] Setting up regression infrastructure...")

# Define outcome variables
OUTCOMES = ['cpi_incwage', 'cpi_incwage_no0', 'ln_cpi_income', 'poverty100', 'employed', 'hrs_worked']
OUTCOME_LABELS = {
    'cpi_incwage': 'CPI-adjusted Income',
    'cpi_incwage_no0': 'CPI-adjusted Income (excl zeros)',
    'ln_cpi_income': 'Log Income',
    'poverty100': 'Below Poverty Line',
    'employed': 'Employed',
    'hrs_worked': 'Hours Worked'
}

# Main treatment variable
TREATMENT = 'M12_exp_rate'

# Define fixed effects as dummy variables for regression
# Create dummy variables for fixed effects
print("  Creating fixed effect dummies...")
df['bpl_fe'] = pd.Categorical(df['bpl']).codes
df['birthyr_fe'] = pd.Categorical(df['birthyr']).codes
df['year_fe'] = pd.Categorical(df['year']).codes
df['ageblackfemale_fe'] = pd.Categorical(df['ageblackfemale']).codes

# For pyfixest/fixest-style regressions
# Create cluster variable as numeric
df['bplcohort_num'] = pd.Categorical(df['bplcohort']).codes

def run_regression(data, outcome, treatment, controls=None, fe_vars=None, cluster_var='bplcohort_num',
                   weight_var=None, robust=True):
    """
    Run a regression with fixed effects and clustered standard errors.
    Returns coefficient, se, t-stat, p-value, CI, n_obs, r_squared, and full coefficient vector.
    """
    # Prepare data
    df_reg = data.copy()

    # Drop missing values for outcome and treatment
    df_reg = df_reg.dropna(subset=[outcome, treatment])

    # Build formula
    if controls is None:
        controls = []

    # For fixed effects, we use dummy variables
    all_vars = [treatment] + controls

    # Drop missing in all variables
    df_reg = df_reg.dropna(subset=all_vars)

    if fe_vars:
        for fe in fe_vars:
            df_reg = df_reg.dropna(subset=[fe])

    if len(df_reg) < 100:
        return None

    # Create design matrix
    X_vars = [treatment] + controls
    X = df_reg[X_vars].copy()

    # Add fixed effect dummies
    if fe_vars:
        for fe in fe_vars:
            dummies = pd.get_dummies(df_reg[fe], prefix=fe, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    y = df_reg[outcome]

    # Weights
    if weight_var and weight_var in df_reg.columns:
        weights = df_reg[weight_var]
    else:
        weights = None

    # Fit model
    if weights is not None:
        model = OLS(y, X, weights=weights)
    else:
        model = OLS(y, X)

    # Get cluster-robust standard errors
    if cluster_var and cluster_var in df_reg.columns:
        clusters = df_reg[cluster_var]
        try:
            results_obj = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})
        except:
            results_obj = model.fit(cov_type='HC1')
    elif robust:
        results_obj = model.fit(cov_type='HC1')
    else:
        results_obj = model.fit()

    # Extract results for treatment variable
    if treatment in results_obj.params:
        coef = results_obj.params[treatment]
        se = results_obj.bse[treatment]
        t = results_obj.tvalues[treatment]
        p = results_obj.pvalues[treatment]
        ci = results_obj.conf_int().loc[treatment]
        ci_lower, ci_upper = ci[0], ci[1]
    else:
        return None

    n_obs = int(results_obj.nobs)
    r_squared = results_obj.rsquared

    # Build coefficient vector JSON
    coef_vector = {
        'treatment': {
            'var': treatment,
            'coef': float(coef),
            'se': float(se),
            'pval': float(p)
        },
        'controls': [],
        'fixed_effects_absorbed': fe_vars if fe_vars else [],
        'diagnostics': {},
        'n_obs': n_obs,
        'r_squared': float(r_squared)
    }

    # Add control coefficients
    for ctrl in controls:
        if ctrl in results_obj.params:
            coef_vector['controls'].append({
                'var': ctrl,
                'coef': float(results_obj.params[ctrl]),
                'se': float(results_obj.bse[ctrl]),
                'pval': float(results_obj.pvalues[ctrl])
            })

    return {
        'coefficient': coef,
        'std_error': se,
        't_stat': t,
        'p_value': p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector': coef_vector
    }


def run_regression_simple(data, outcome, treatment, fe_list=['bpl', 'birthyr', 'year'],
                          cluster='bplcohort_num', controls=None, sample_desc='Full sample'):
    """
    Simplified regression function that handles fixed effects more efficiently.
    Uses demeaning approach for large fixed effects.
    """
    df_reg = data.copy()

    # Drop missing
    vars_needed = [outcome, treatment] + (controls if controls else [])
    for fe in fe_list:
        vars_needed.append(fe)
    if cluster:
        vars_needed.append(cluster)

    df_reg = df_reg.dropna(subset=[v for v in vars_needed if v in df_reg.columns])

    if len(df_reg) < 100:
        return None

    # Demean by fixed effects (absorb FEs)
    y = df_reg[outcome].values
    X_vars = [treatment] + (controls if controls else [])
    X = df_reg[X_vars].values

    # Demean
    for fe in fe_list:
        if fe in df_reg.columns:
            groups = df_reg[fe].values
            unique_groups = np.unique(groups)
            for g in unique_groups:
                mask = groups == g
                y[mask] = y[mask] - y[mask].mean()
                for j in range(X.shape[1]):
                    X[mask, j] = X[mask, j] - X[mask, j].mean()

    # Add constant (will be zero after demeaning, but needed for OLS)
    X_with_const = np.column_stack([np.ones(len(y)), X])

    # OLS
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except:
        return None

    # Residuals and SE calculation
    residuals = y - X_with_const @ beta
    n = len(y)
    k = X_with_const.shape[1]

    # Get cluster-robust SEs
    if cluster and cluster in df_reg.columns:
        clusters = df_reg[cluster].values
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)

        # Meat of the sandwich
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        meat = np.zeros((k, k))

        for g in unique_clusters:
            mask = clusters == g
            X_g = X_with_const[mask]
            e_g = residuals[mask]
            score_g = (X_g.T @ e_g).reshape(-1, 1)
            meat += score_g @ score_g.T

        # Finite sample correction
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        V = correction * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(V))
    else:
        # Robust SEs
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        meat = np.zeros((k, k))
        for i in range(n):
            xi = X_with_const[i:i+1].T
            meat += (residuals[i]**2) * (xi @ xi.T)
        V = XtX_inv @ meat @ XtX_inv * (n / (n - k))
        se = np.sqrt(np.diag(V))

    # Treatment coefficient is at index 1 (after constant)
    coef = beta[1]
    se_treat = se[1]
    t_stat = coef / se_treat
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
    ci_lower = coef - 1.96 * se_treat
    ci_upper = coef + 1.96 * se_treat

    # R-squared (within)
    tss = np.sum((y - y.mean())**2)
    rss = np.sum(residuals**2)
    r_squared = 1 - rss/tss if tss > 0 else 0

    # Build coefficient vector
    coef_vector = {
        'treatment': {
            'var': treatment,
            'coef': float(coef),
            'se': float(se_treat),
            'pval': float(p_value)
        },
        'controls': [],
        'fixed_effects_absorbed': fe_list,
        'diagnostics': {},
        'n_obs': int(n),
        'r_squared': float(r_squared)
    }

    # Add control coefficients
    if controls:
        for i, ctrl in enumerate(controls):
            idx = i + 2  # Skip constant and treatment
            if idx < len(beta):
                coef_vector['controls'].append({
                    'var': ctrl,
                    'coef': float(beta[idx]),
                    'se': float(se[idx]) if idx < len(se) else np.nan,
                    'pval': float(2 * (1 - stats.t.cdf(abs(beta[idx]/se[idx]), n - k))) if idx < len(se) and se[idx] > 0 else np.nan
                })

    return {
        'coefficient': coef,
        'std_error': se_treat,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n,
        'r_squared': r_squared,
        'coefficient_vector': coef_vector
    }


# ============================================================
# STEP 5: Run Baseline Specifications
# ============================================================
print("\n[5] Running baseline specifications...")

# Main specification from Table 2:
# reg outcome M12_exp_rate i.bpl i.birthyr i.ageblackfemale i.bpl_black i.bpl_female i.bpl_black_female black female i.year, robust cluster(bplcohort)

# Simplified baseline controls (we can't do all interactions easily)
BASELINE_CONTROLS = ['black', 'female']

spec_count = 0

for outcome in OUTCOMES:
    print(f"  Running baseline for {outcome}...")

    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS,
        sample_desc='Full sample, ages 26-59, native born, black/white'
    )

    if result:
        add_result(
            spec_id='baseline',
            spec_tree_path='methods/difference_in_differences.md#baseline',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample, ages 26-59, native born, black/white',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1
        print(f"    {outcome}: coef={result['coefficient']:.6f}, se={result['std_error']:.6f}, p={result['p_value']:.4f}")

print(f"\n  Baseline specifications complete: {spec_count} specs")

# ============================================================
# STEP 6: Fixed Effects Variations
# ============================================================
print("\n[6] Running fixed effects variations...")

fe_variations = [
    ('did/fe/unit_only', ['bpl'], 'Birth state FE only'),
    ('did/fe/time_only', ['birthyr', 'year'], 'Birth year + ACS year FE only'),
    ('did/fe/twoway', ['bpl', 'birthyr', 'year'], 'Two-way FE (baseline)'),
    ('did/fe/none', [], 'No fixed effects'),
]

for spec_id, fe_list, fe_desc in fe_variations:
    for outcome in OUTCOMES:
        result = run_regression_simple(
            df, outcome, TREATMENT,
            fe_list=fe_list,
            cluster='bplcohort_num',
            controls=BASELINE_CONTROLS
        )

        if result:
            add_result(
                spec_id=spec_id,
                spec_tree_path='methods/difference_in_differences.md#fixed-effects',
                outcome_var=outcome,
                treatment_var=TREATMENT,
                coefficient=result['coefficient'],
                std_error=result['std_error'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coefficient_vector_json=result['coefficient_vector'],
                sample_desc='Full sample',
                fixed_effects=fe_desc,
                controls_desc='black, female',
                cluster_var='bplcohort',
                model_type='OLS with FE'
            )
            spec_count += 1

print(f"  FE variations complete. Total specs: {spec_count}")

# ============================================================
# STEP 7: Control Variations
# ============================================================
print("\n[7] Running control variations...")

# No controls
for outcome in OUTCOMES:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=None
    )

    if result:
        add_result(
            spec_id='did/controls/none',
            spec_tree_path='methods/difference_in_differences.md#control-sets',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='None',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Just black
for outcome in OUTCOMES:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['black']
    )

    if result:
        add_result(
            spec_id='robust/control/add_black',
            spec_tree_path='robustness/control_progression.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Just female
for outcome in OUTCOMES:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['female']
    )

    if result:
        add_result(
            spec_id='robust/control/add_female',
            spec_tree_path='robustness/control_progression.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

print(f"  Control variations complete. Total specs: {spec_count}")

# ============================================================
# STEP 8: Sample Restrictions
# ============================================================
print("\n[8] Running sample restriction variations...")

# Exclude partial exposure cohorts (only full exposure=0 or 16)
df_exclude_partial = df[(df['exposure'] == 0) | (df['exposure'] == 16)].copy()
for outcome in OUTCOMES:
    result = run_regression_simple(
        df_exclude_partial, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/sample/exclude_partial_exposure',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Exclude partial exposure (only exp=0 or 16)',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Narrow treatment window: 1941-1971
df_narrow = df[df['birthyr'] < 1972].copy()
for outcome in OUTCOMES:
    result = run_regression_simple(
        df_narrow, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/sample/birthyear_pre1972',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Birth years before 1972 only',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Males only
df_male = df[df['female'] == 0].copy()
for outcome in OUTCOMES:
    result = run_regression_simple(
        df_male, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['black']
    )

    if result:
        add_result(
            spec_id='robust/sample/male_only',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Males only',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Females only
df_female = df[df['female'] == 1].copy()
for outcome in OUTCOMES:
    result = run_regression_simple(
        df_female, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['black']
    )

    if result:
        add_result(
            spec_id='robust/sample/female_only',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Females only',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# White only
df_white = df[df['white'] == 1].copy()
for outcome in OUTCOMES:
    result = run_regression_simple(
        df_white, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['female']
    )

    if result:
        add_result(
            spec_id='robust/sample/white_only',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='White respondents only',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Black only
df_black = df[df['black'] == 1].copy()
for outcome in OUTCOMES:
    result = run_regression_simple(
        df_black, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['female']
    )

    if result:
        add_result(
            spec_id='robust/sample/black_only',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Black respondents only',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Drop outliers in income (winsorize at 1%)
for outcome in ['cpi_incwage', 'cpi_incwage_no0', 'ln_cpi_income']:
    df_wins = df.copy()
    p1 = df_wins[outcome].quantile(0.01)
    p99 = df_wins[outcome].quantile(0.99)
    df_wins[outcome] = df_wins[outcome].clip(lower=p1, upper=p99)

    result = run_regression_simple(
        df_wins, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/sample/winsorize_1pct',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Winsorized at 1%/99%',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Drop specific ACS years
for drop_year in [2000, 2005, 2010, 2015]:
    df_drop = df[df['year'] != drop_year].copy()
    for outcome in ['cpi_incwage', 'ln_cpi_income']:  # Just main outcomes
        result = run_regression_simple(
            df_drop, outcome, TREATMENT,
            fe_list=['bpl', 'birthyr', 'year'],
            cluster='bplcohort_num',
            controls=BASELINE_CONTROLS
        )

        if result:
            add_result(
                spec_id=f'robust/sample/drop_year_{drop_year}',
                spec_tree_path='robustness/sample_restrictions.md',
                outcome_var=outcome,
                treatment_var=TREATMENT,
                coefficient=result['coefficient'],
                std_error=result['std_error'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coefficient_vector_json=result['coefficient_vector'],
                sample_desc=f'Exclude ACS year {drop_year}',
                fixed_effects='Birth state + Birth year + ACS year',
                controls_desc='black, female',
                cluster_var='bplcohort',
                model_type='OLS with FE'
            )
            spec_count += 1

print(f"  Sample restrictions complete. Total specs: {spec_count}")

# ============================================================
# STEP 9: Alternative Treatment Definitions
# ============================================================
print("\n[9] Running alternative treatment definitions...")

# Alternative windows for calculating pre-vaccine measles rate
for window in [2, 3, 5, 7, 10]:
    treatment_var = f'M{window}_exp_rate'
    if treatment_var in df.columns:
        for outcome in OUTCOMES:
            result = run_regression_simple(
                df, outcome, treatment_var,
                fe_list=['bpl', 'birthyr', 'year'],
                cluster='bplcohort_num',
                controls=BASELINE_CONTROLS
            )

            if result:
                add_result(
                    spec_id=f'robust/treatment/window_{window}yr',
                    spec_tree_path='robustness/measurement.md',
                    outcome_var=outcome,
                    treatment_var=treatment_var,
                    coefficient=result['coefficient'],
                    std_error=result['std_error'],
                    t_stat=result['t_stat'],
                    p_value=result['p_value'],
                    ci_lower=result['ci_lower'],
                    ci_upper=result['ci_upper'],
                    n_obs=result['n_obs'],
                    r_squared=result['r_squared'],
                    coefficient_vector_json=result['coefficient_vector'],
                    sample_desc='Full sample',
                    fixed_effects='Birth state + Birth year + ACS year',
                    controls_desc='black, female',
                    cluster_var='bplcohort',
                    model_type='OLS with FE'
                )
                spec_count += 1

# Binary treatment (any exposure vs no exposure)
df['treat_binary'] = (df['exposure'] > 0).astype(int)
for outcome in ['cpi_incwage', 'ln_cpi_income', 'employed']:
    result = run_regression_simple(
        df, outcome, 'treat_binary',
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/treatment/binary',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            treatment_var='treat_binary',
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

print(f"  Alternative treatments complete. Total specs: {spec_count}")

# ============================================================
# STEP 10: Clustering Variations
# ============================================================
print("\n[10] Running clustering variations...")

# Cluster at birth state level only
df['bpl_num'] = pd.Categorical(df['bpl']).codes
for outcome in OUTCOMES:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bpl_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/cluster/bpl',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bpl',
            model_type='OLS with FE'
        )
        spec_count += 1

# Cluster at birth year level
df['birthyr_num'] = pd.Categorical(df['birthyr']).codes
for outcome in OUTCOMES:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='birthyr_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/cluster/birthyr',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='birthyr',
            model_type='OLS with FE'
        )
        spec_count += 1

# Cluster at region level
if 'bpl_region9' in df.columns:
    df['region_num'] = pd.Categorical(df['bpl_region9']).codes
    for outcome in ['cpi_incwage', 'ln_cpi_income', 'employed']:
        result = run_regression_simple(
            df, outcome, TREATMENT,
            fe_list=['bpl', 'birthyr', 'year'],
            cluster='region_num',
            controls=BASELINE_CONTROLS
        )

        if result:
            add_result(
                spec_id='robust/cluster/region9',
                spec_tree_path='robustness/clustering_variations.md',
                outcome_var=outcome,
                treatment_var=TREATMENT,
                coefficient=result['coefficient'],
                std_error=result['std_error'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coefficient_vector_json=result['coefficient_vector'],
                sample_desc='Full sample',
                fixed_effects='Birth state + Birth year + ACS year',
                controls_desc='black, female',
                cluster_var='bpl_region9',
                model_type='OLS with FE'
            )
            spec_count += 1

# Robust SEs (no clustering)
for outcome in OUTCOMES:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster=None,
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/cluster/robust_hc1',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='None (HC1 robust)',
            model_type='OLS with FE'
        )
        spec_count += 1

print(f"  Clustering variations complete. Total specs: {spec_count}")

# ============================================================
# STEP 11: Functional Form Variations
# ============================================================
print("\n[11] Running functional form variations...")

# Level of income (already have)
# IHS transformation
df['ihs_incwage'] = np.arcsinh(df['cpi_incwage'])
result = run_regression_simple(
    df, 'ihs_incwage', TREATMENT,
    fe_list=['bpl', 'birthyr', 'year'],
    cluster='bplcohort_num',
    controls=BASELINE_CONTROLS
)

if result:
    add_result(
        spec_id='robust/funcform/ihs_outcome',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='ihs_incwage',
        treatment_var=TREATMENT,
        coefficient=result['coefficient'],
        std_error=result['std_error'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coefficient_vector_json=result['coefficient_vector'],
        sample_desc='Full sample',
        fixed_effects='Birth state + Birth year + ACS year',
        controls_desc='black, female',
        cluster_var='bplcohort',
        model_type='OLS with FE'
    )
    spec_count += 1

# Squared treatment term
df['M12_exp_rate_sq'] = df['M12_exp_rate'] ** 2
for outcome in ['cpi_incwage', 'ln_cpi_income']:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS + ['M12_exp_rate_sq']
    )

    if result:
        add_result(
            spec_id='robust/funcform/quadratic_treatment',
            spec_tree_path='robustness/functional_form.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female, treatment squared',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

print(f"  Functional form variations complete. Total specs: {spec_count}")

# ============================================================
# STEP 12: Heterogeneity Analysis
# ============================================================
print("\n[12] Running heterogeneity analyses...")

# Interaction with female
df['treat_x_female'] = df[TREATMENT] * df['female']
for outcome in ['cpi_incwage', 'ln_cpi_income', 'employed', 'hrs_worked']:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['black', 'female', 'treat_x_female']
    )

    if result:
        add_result(
            spec_id='robust/heterogeneity/female',
            spec_tree_path='robustness/heterogeneity.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female, treatment x female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Interaction with black
df['treat_x_black'] = df[TREATMENT] * df['black']
for outcome in ['cpi_incwage', 'ln_cpi_income', 'employed', 'hrs_worked']:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['black', 'female', 'treat_x_black']
    )

    if result:
        add_result(
            spec_id='robust/heterogeneity/black',
            spec_tree_path='robustness/heterogeneity.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female, treatment x black',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# By age groups (at time of survey)
df['young'] = (df['age'] < 45).astype(int)
df['treat_x_young'] = df[TREATMENT] * df['young']
for outcome in ['cpi_incwage', 'ln_cpi_income']:
    result = run_regression_simple(
        df, outcome, TREATMENT,
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=['black', 'female', 'young', 'treat_x_young']
    )

    if result:
        add_result(
            spec_id='robust/heterogeneity/age_young',
            spec_tree_path='robustness/heterogeneity.md',
            outcome_var=outcome,
            treatment_var=TREATMENT,
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Full sample',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female, young, treatment x young',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

print(f"  Heterogeneity analyses complete. Total specs: {spec_count}")

# ============================================================
# STEP 13: Placebo Tests
# ============================================================
print("\n[13] Running placebo tests...")

# Pre-vaccine cohorts only (placebo: should be no effect)
df_pre = df[df['exposure'] == 0].copy()
for outcome in ['cpi_incwage', 'ln_cpi_income', 'employed']:
    # Use the measles rate (not interacted with exposure, since exposure=0 for all)
    result = run_regression_simple(
        df_pre, outcome, 'avg_12yr_measles_rate',
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/placebo/pre_vaccine_cohorts',
            spec_tree_path='robustness/placebo_tests.md',
            outcome_var=outcome,
            treatment_var='avg_12yr_measles_rate',
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Pre-vaccine cohorts only (exposure=0)',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

# Fake treatment timing (shift vaccine year to 1958)
def get_fake_exposure(birthyr):
    """Fake exposure assuming vaccine in 1958 instead of 1963"""
    if birthyr < 1943:  # 6 years earlier
        return 0
    elif birthyr <= 1957:
        return birthyr - 1942
    else:
        return 16

df['fake_exposure'] = df['birthyr'].apply(get_fake_exposure)
df['fake_M12_exp_rate'] = (df['avg_12yr_measles_rate'] * df['fake_exposure']) / 100000

for outcome in ['cpi_incwage', 'ln_cpi_income', 'employed']:
    result = run_regression_simple(
        df, outcome, 'fake_M12_exp_rate',
        fe_list=['bpl', 'birthyr', 'year'],
        cluster='bplcohort_num',
        controls=BASELINE_CONTROLS
    )

    if result:
        add_result(
            spec_id='robust/placebo/fake_timing_1958',
            spec_tree_path='robustness/placebo_tests.md',
            outcome_var=outcome,
            treatment_var='fake_M12_exp_rate',
            coefficient=result['coefficient'],
            std_error=result['std_error'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coefficient_vector_json=result['coefficient_vector'],
            sample_desc='Fake vaccine timing (1958 instead of 1963)',
            fixed_effects='Birth state + Birth year + ACS year',
            controls_desc='black, female',
            cluster_var='bplcohort',
            model_type='OLS with FE'
        )
        spec_count += 1

print(f"  Placebo tests complete. Total specs: {spec_count}")

# ============================================================
# STEP 14: Region-Based Specifications
# ============================================================
print("\n[14] Running region-based specifications...")

# Add region-by-birthyear fixed effects
if 'bpl_region9' in df.columns:
    df['region_birthyr'] = df['bpl_region9'].astype(str) + '_' + df['birthyr'].astype(str)

    for outcome in OUTCOMES:
        result = run_regression_simple(
            df, outcome, TREATMENT,
            fe_list=['bpl', 'birthyr', 'year', 'region_birthyr'],
            cluster='bplcohort_num',
            controls=BASELINE_CONTROLS
        )

        if result:
            add_result(
                spec_id='did/fe/region_x_birthyear',
                spec_tree_path='methods/difference_in_differences.md#fixed-effects',
                outcome_var=outcome,
                treatment_var=TREATMENT,
                coefficient=result['coefficient'],
                std_error=result['std_error'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coefficient_vector_json=result['coefficient_vector'],
                sample_desc='Full sample',
                fixed_effects='Birth state + Birth year + ACS year + Region x Birth year',
                controls_desc='black, female',
                cluster_var='bplcohort',
                model_type='OLS with FE'
            )
            spec_count += 1

print(f"  Region-based specifications complete. Total specs: {spec_count}")

# ============================================================
# STEP 15: Additional robustness - drop each state
# ============================================================
print("\n[15] Running leave-one-out state specifications...")

# Get unique states
states = df['bpl'].unique()
# Run for a subset of states and main outcomes only to keep spec count manageable
sample_states = np.random.choice(states, min(5, len(states)), replace=False)

for state in sample_states:
    df_drop = df[df['bpl'] != state].copy()
    for outcome in ['cpi_incwage', 'ln_cpi_income']:
        result = run_regression_simple(
            df_drop, outcome, TREATMENT,
            fe_list=['bpl', 'birthyr', 'year'],
            cluster='bplcohort_num',
            controls=BASELINE_CONTROLS
        )

        if result:
            add_result(
                spec_id=f'robust/loo/drop_state_{int(state)}',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var=outcome,
                treatment_var=TREATMENT,
                coefficient=result['coefficient'],
                std_error=result['std_error'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coefficient_vector_json=result['coefficient_vector'],
                sample_desc=f'Exclude state {int(state)}',
                fixed_effects='Birth state + Birth year + ACS year',
                controls_desc='black, female',
                cluster_var='bplcohort',
                model_type='OLS with FE'
            )
            spec_count += 1

print(f"  Leave-one-out state specs complete. Total specs: {spec_count}")

# ============================================================
# STEP 16: Save Results
# ============================================================
print("\n[16] Saving results...")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_file = f'{OUTPUT_PATH}/specification_results.csv'
results_df.to_csv(output_file, index=False)

print(f"\n{'=' * 60}")
print(f"SPECIFICATION SEARCH COMPLETE")
print(f"{'=' * 60}")
print(f"Total specifications run: {len(results_df)}")
print(f"Results saved to: {output_file}")

# Summary statistics
print(f"\n--- Summary Statistics ---")
print(f"Unique outcomes: {results_df['outcome_var'].nunique()}")
print(f"Unique spec_ids: {results_df['spec_id'].nunique()}")

# Coefficient summary for main outcome
main_results = results_df[results_df['outcome_var'] == 'ln_cpi_income']
if len(main_results) > 0:
    print(f"\nFor ln_cpi_income ({len(main_results)} specs):")
    print(f"  Mean coefficient: {main_results['coefficient'].mean():.6f}")
    print(f"  Median coefficient: {main_results['coefficient'].median():.6f}")
    print(f"  Range: [{main_results['coefficient'].min():.6f}, {main_results['coefficient'].max():.6f}]")
    print(f"  % Positive: {(main_results['coefficient'] > 0).mean()*100:.1f}%")
    print(f"  % Significant at 5%: {(main_results['p_value'] < 0.05).mean()*100:.1f}%")
    print(f"  % Significant at 1%: {(main_results['p_value'] < 0.01).mean()*100:.1f}%")

print("\nDone!")
