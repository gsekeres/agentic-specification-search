"""
Specification Search: Paper 136741-V1
Williams - Historical Lynchings and Black Voter Registration

Paper Overview:
- Examines the persistent effects of historical black lynching rates on contemporary
  black voter registration rates in southern US counties
- Cross-sectional analysis at county level
- Main hypothesis: Higher historical lynching rates lead to lower black voter registration
- Data: 267 counties across 6 southern states (AL, FL, GA, LA, NC, SC)

Method: Cross-sectional OLS with state fixed effects
Treatment: lynchcapitamob (black lynching rate per 10k black population, using 1900 population)
Outcome: Blackrate_regvoters (black voter registration rate, %)
Historical Controls: Black_share_illiterate, initial (county formation), newscapita,
                     farmvalue, sfarmprop1860, landineq1860, fbprop1860
Contemporary Controls: Black_beyondhs, Black_avgage, Black_Earnings, share_maritalblacks
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SETUP
# =============================================================================

BASE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_DIR = f'{BASE_DIR}/data/downloads/extracted/136741-V1/Williams_files/Analysis_data'

# Load data
df = pd.read_stata(f'{DATA_DIR}/maindata.dta')

# Paper information
PAPER_ID = '136741-V1'
JOURNAL = 'AEJ-Applied'  # Based on replication package format
PAPER_TITLE = 'Historical Lynchings and Black Voter Registration'

# Variable definitions from the paper
OUTCOME_VAR = 'Blackrate_regvoters'
TREATMENT_VAR = 'lynchcapitamob'

# Historical controls (from the paper's global definition)
HISTORICAL_CONTROLS = ['Black_share_illiterate', 'initial', 'newscapita', 'farmvalue',
                       'sfarmprop1860', 'landineq1860', 'fbprop1860']

# Contemporary controls
CONTEMPORARY_CONTROLS = ['Black_beyondhs', 'Black_avgage', 'Black_Earnings', 'share_maritalblacks']

# Additional controls that appear in the paper
ADDITIONAL_CONTROLS = ['incarceration_2010', 'pollscapita', 'share_slaves']

# All available controls
ALL_CONTROLS = HISTORICAL_CONTROLS + CONTEMPORARY_CONTROLS + ADDITIONAL_CONTROLS

# State FE variable
FE_VAR = 'State_FIPS'

# Results storage
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    sample_desc, controls_desc, fixed_effects, cluster_var, model_type='OLS',
                    coefficient_vector=None):
    """Extract results from a pyfixest or statsmodels model."""

    # Check model type
    is_pyfixest = hasattr(model, 'coef') and callable(model.coef)
    is_statsmodels = hasattr(model, 'params') and not callable(model.params)

    if is_pyfixest:
        # pyfixest format
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        t_stat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        n_obs = model._N  # Number of observations
        r_sq = model._r2  # R-squared

        # Confidence interval
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Build coefficient vector JSON
        if coefficient_vector is None:
            coef_vector = {
                "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
                "controls": [],
                "fixed_effects": [fixed_effects] if fixed_effects else [],
                "diagnostics": {}
            }
            # Add control coefficients
            for var in model.coef().index:
                if var != treatment_var and var != 'Intercept':
                    coef_vector["controls"].append({
                        "var": var,
                        "coef": float(model.coef()[var]),
                        "se": float(model.se()[var]),
                        "pval": float(model.pvalue()[var])
                    })
        else:
            coef_vector = coefficient_vector

    elif is_statsmodels:
        # Statsmodels format
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        n_obs = int(model.nobs)
        r_sq = model.rsquared if hasattr(model, 'rsquared') else model.prsquared if hasattr(model, 'prsquared') else np.nan

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "fixed_effects": [fixed_effects] if fixed_effects else [],
            "diagnostics": {}
        }
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

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
        't_stat': float(t_stat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r_sq),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

# =============================================================================
# BASELINE SPECIFICATION (Paper's Table 3, Column 1)
# =============================================================================
print("Running baseline specification...")

# Build formula for baseline - state FE with historical controls
historical_formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
baseline = pf.feols(historical_formula, data=df, vcov='hetero')

results.append(extract_results(
    baseline,
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md#baseline',
    outcome_var=OUTCOME_VAR,
    treatment_var=TREATMENT_VAR,
    sample_desc='Full sample of 267 Southern US counties',
    controls_desc='Historical controls: illiteracy, county formation year, newspapers, farm value, small farms, land inequality, free blacks',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))

print(f"Baseline coefficient: {baseline.coef()[TREATMENT_VAR]:.4f} (SE: {baseline.se()[TREATMENT_VAR]:.4f})")

# =============================================================================
# 1. CONTROL VARIATIONS - Leave-One-Out (7 specs)
# =============================================================================
print("\n1. Running leave-one-out specifications...")

for control in HISTORICAL_CONTROLS:
    remaining = [c for c in HISTORICAL_CONTROLS if c != control]
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(remaining) + f" | {FE_VAR}"
    model = pf.feols(formula, data=df, vcov='hetero')

    results.append(extract_results(
        model,
        spec_id=f'robust/loo/drop_{control}',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var=OUTCOME_VAR,
        treatment_var=TREATMENT_VAR,
        sample_desc='Full sample',
        controls_desc=f'Historical controls minus {control}',
        fixed_effects='State FE',
        cluster_var='Robust (heteroskedasticity-consistent)',
        model_type='OLS with FE'
    ))
    print(f"  Drop {control}: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# =============================================================================
# 2. CONTROL PROGRESSION - Build Up (10 specs)
# =============================================================================
print("\n2. Running control progression specifications...")

# Bivariate (no controls except FE)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/build/bivariate',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='No controls (bivariate)',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Bivariate: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Add historical controls incrementally
for i, control in enumerate(HISTORICAL_CONTROLS):
    controls_so_far = HISTORICAL_CONTROLS[:i+1]
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(controls_so_far) + f" | {FE_VAR}"
    model = pf.feols(formula, data=df, vcov='hetero')

    results.append(extract_results(
        model,
        spec_id=f'robust/build/add_{control}',
        spec_tree_path='robustness/control_progression.md',
        outcome_var=OUTCOME_VAR,
        treatment_var=TREATMENT_VAR,
        sample_desc='Full sample',
        controls_desc=f'Historical controls through {control}',
        fixed_effects='State FE',
        cluster_var='Robust (heteroskedasticity-consistent)',
        model_type='OLS with FE'
    ))
    print(f"  Add {control}: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Contemporary controls only
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(CONTEMPORARY_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/build/contemporary_only',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Contemporary controls only',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Contemporary only: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Full model - historical + contemporary
full_controls = HISTORICAL_CONTROLS + CONTEMPORARY_CONTROLS
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(full_controls) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/build/full',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical + contemporary controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Full model: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Kitchen sink - all available controls
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(ALL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/build/kitchen_sink',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='All available controls (kitchen sink)',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Kitchen sink: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# =============================================================================
# 3. INFERENCE VARIATIONS / CLUSTERING (6 specs)
# =============================================================================
print("\n3. Running inference/clustering variations...")

# Cluster by state (few clusters - only 6 states)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov={'CRV1': 'State_FIPS'})
results.append(extract_results(
    model, spec_id='robust/cluster/state',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='State (6 clusters)',
    model_type='OLS with FE'
))
print(f"  Cluster by state: SE = {model.se()[TREATMENT_VAR]:.4f}")

# Cluster by county (fips)
model = pf.feols(formula, data=df, vcov={'CRV1': 'fips'})
results.append(extract_results(
    model, spec_id='robust/cluster/county',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='County FIPS',
    model_type='OLS with FE'
))
print(f"  Cluster by county: SE = {model.se()[TREATMENT_VAR]:.4f}")

# HC1 robust
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/se/hc1',
    spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='HC1 robust',
    model_type='OLS with FE'
))
print(f"  HC1 robust: SE = {model.se()[TREATMENT_VAR]:.4f}")

# HC2 robust (using statsmodels - no FE)
df_model = df.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR] + HISTORICAL_CONTROLS)
X = df_model[[TREATMENT_VAR] + HISTORICAL_CONTROLS].copy()
# Add state dummies
state_dummies = pd.get_dummies(df_model['State_FIPS'], prefix='state', drop_first=True)
X = pd.concat([X.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
X = sm.add_constant(X)
# Ensure numeric
X = X.astype(float)
y = df_model[OUTCOME_VAR].astype(float)
model = sm.OLS(y, X).fit(cov_type='HC2')
results.append(extract_results(
    model, spec_id='robust/se/hc2',
    spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State dummies',
    cluster_var='HC2 robust',
    model_type='OLS with state dummies'
))
print(f"  HC2 robust: SE = {model.bse[TREATMENT_VAR]:.4f}")

# HC3 robust (small sample)
model = sm.OLS(y, X).fit(cov_type='HC3')
results.append(extract_results(
    model, spec_id='robust/se/hc3',
    spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State dummies',
    cluster_var='HC3 robust (small sample)',
    model_type='OLS with state dummies'
))
print(f"  HC3 robust: SE = {model.bse[TREATMENT_VAR]:.4f}")

# Classical SE (no clustering/robustness)
model = pf.feols(formula, data=df, vcov='iid')
results.append(extract_results(
    model, spec_id='robust/se/classical',
    spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Classical (homoskedastic)',
    model_type='OLS with FE'
))
print(f"  Classical SE: SE = {model.se()[TREATMENT_VAR]:.4f}")

# =============================================================================
# 4. SAMPLE RESTRICTIONS (12 specs)
# =============================================================================
print("\n4. Running sample restriction specifications...")

# Drop each state one at a time
for state_fips in df['State_FIPS'].unique():
    state_name = df[df['State_FIPS'] == state_fips]['State'].iloc[0]
    df_sub = df[df['State_FIPS'] != state_fips]

    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
    try:
        model = pf.feols(formula, data=df_sub, vcov='hetero')
        results.append(extract_results(
            model, spec_id=f'robust/sample/drop_{state_name.replace(" ", "_")}',
            spec_tree_path='robustness/sample_restrictions.md#geographic-restrictions',
            outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
            sample_desc=f'Drop {state_name}',
            controls_desc='Historical controls',
            fixed_effects='State FE',
            cluster_var='Robust (heteroskedasticity-consistent)',
            model_type='OLS with FE'
        ))
        print(f"  Drop {state_name}: coef = {model.coef()[TREATMENT_VAR]:.4f}")
    except:
        print(f"  Drop {state_name}: Error (skipped)")

# Trim top/bottom 1% of outcome
lower = df[OUTCOME_VAR].quantile(0.01)
upper = df[OUTCOME_VAR].quantile(0.99)
df_trim = df[(df[OUTCOME_VAR] >= lower) & (df[OUTCOME_VAR] <= upper)]
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df_trim, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Trim top/bottom 1% of outcome',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Trim 1%: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Trim top/bottom 5%
lower = df[OUTCOME_VAR].quantile(0.05)
upper = df[OUTCOME_VAR].quantile(0.95)
df_trim = df[(df[OUTCOME_VAR] >= lower) & (df[OUTCOME_VAR] <= upper)]
model = pf.feols(formula, data=df_trim, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/sample/trim_5pct',
    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Trim top/bottom 5% of outcome',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Trim 5%: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Winsorize at 1%
df_wins = df.copy()
df_wins[OUTCOME_VAR] = df_wins[OUTCOME_VAR].clip(
    lower=df_wins[OUTCOME_VAR].quantile(0.01),
    upper=df_wins[OUTCOME_VAR].quantile(0.99)
)
model = pf.feols(formula, data=df_wins, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/sample/winsor_1pct',
    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Winsorize outcome at 1%/99%',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Winsorize 1%: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Drop counties with voter registration > 100% (data quality issue from paper)
df_quality = df[df[OUTCOME_VAR] <= 100]
model = pf.feols(formula, data=df_quality, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/sample/reg_under_100',
    spec_tree_path='robustness/sample_restrictions.md#data-quality',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Exclude registration rates > 100%',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Reg under 100%: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Convert registration > 100 to 100 (as in paper's Table B3)
df_capped = df.copy()
df_capped.loc[df_capped[OUTCOME_VAR] > 100, OUTCOME_VAR] = 100
model = pf.feols(formula, data=df_capped, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/sample/reg_capped_100',
    spec_tree_path='robustness/sample_restrictions.md#data-quality',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Cap registration rates at 100%',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Reg capped at 100%: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# =============================================================================
# 5. ALTERNATIVE TREATMENT DEFINITIONS (4 specs)
# =============================================================================
print("\n5. Running alternative treatment specifications...")

# Using 1910 population as denominator
formula = f"{OUTCOME_VAR} ~ lynchcapitamob1910 + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/treatment/lynch_1910_pop',
    spec_tree_path='robustness/measurement.md',
    outcome_var=OUTCOME_VAR, treatment_var='lynchcapitamob1910',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  1910 pop: coef = {model.coef()['lynchcapitamob1910']:.4f}")

# Using 1920 population as denominator
formula = f"{OUTCOME_VAR} ~ lynchcapitamob1920 + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/treatment/lynch_1920_pop',
    spec_tree_path='robustness/measurement.md',
    outcome_var=OUTCOME_VAR, treatment_var='lynchcapitamob1920',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  1920 pop: coef = {model.coef()['lynchcapitamob1920']:.4f}")

# Using 1930 population as denominator
formula = f"{OUTCOME_VAR} ~ lynchcapitamob1930 + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/treatment/lynch_1930_pop',
    spec_tree_path='robustness/measurement.md',
    outcome_var=OUTCOME_VAR, treatment_var='lynchcapitamob1930',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  1930 pop: coef = {model.coef()['lynchcapitamob1930']:.4f}")

# Using Stevenson/EJI data
formula = f"{OUTCOME_VAR} ~ lynchcapitasteve + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/treatment/lynch_stevenson',
    spec_tree_path='robustness/measurement.md',
    outcome_var=OUTCOME_VAR, treatment_var='lynchcapitasteve',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Stevenson data: coef = {model.coef()['lynchcapitasteve']:.4f}")

# =============================================================================
# 6. ALTERNATIVE OUTCOMES (4 specs)
# =============================================================================
print("\n6. Running alternative outcome specifications...")

# White voter registration (falsification - should be no effect)
formula = f"Whiterate_regvoters ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/placebo/white_registration',
    spec_tree_path='robustness/placebo_tests.md#outcome-placebos',
    outcome_var='Whiterate_regvoters', treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  White registration: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Using raw registration count
formula = f"register_black ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/outcome/register_count',
    spec_tree_path='robustness/measurement.md',
    outcome_var='register_black', treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Registration count: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Log registration rate
df['log_blackrate'] = np.log(df[OUTCOME_VAR] + 1)
formula = f"log_blackrate ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/y_log',
    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
    outcome_var='log_blackrate', treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Log registration: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# IHS transformation
df['ihs_blackrate'] = np.arcsinh(df[OUTCOME_VAR])
formula = f"ihs_blackrate ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/y_ihs',
    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
    outcome_var='ihs_blackrate', treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  IHS registration: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# =============================================================================
# 7. PLACEBO TESTS (3 specs)
# =============================================================================
print("\n7. Running placebo tests...")

# White lynching rate on black registration (should be no effect)
formula = f"{OUTCOME_VAR} ~ lynchcapitawhite + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/placebo/white_lynching',
    spec_tree_path='robustness/placebo_tests.md#treatment-assignment-placebos',
    outcome_var=OUTCOME_VAR, treatment_var='lynchcapitawhite',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  White lynching rate: coef = {model.coef()['lynchcapitawhite']:.4f}")

# White lynching on white registration
formula = f"Whiterate_regvoters ~ lynchcapitawhite + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/placebo/white_lynch_white_reg',
    spec_tree_path='robustness/placebo_tests.md#treatment-assignment-placebos',
    outcome_var='Whiterate_regvoters', treatment_var='lynchcapitawhite',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  White lynch on white reg: coef = {model.coef()['lynchcapitawhite']:.4f}")

# Black lynching on white registration (double check placebo)
formula = f"Whiterate_regvoters ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
# Already recorded above, skip
print(f"  Black lynch on white reg: Already recorded")

# =============================================================================
# 8. HETEROGENEITY ANALYSIS (10 specs)
# =============================================================================
print("\n8. Running heterogeneity specifications...")

# By education level (median split)
median_edu = df['Black_beyondhs'].median()
df['high_edu'] = (df['Black_beyondhs'] >= median_edu).astype(int)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR}*high_edu + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/interaction_education',
    spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + education interaction',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Education interaction: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# By earnings level
median_earn = df['Black_Earnings'].median()
df['high_earnings'] = (df['Black_Earnings'] >= median_earn).astype(int)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR}*high_earnings + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/interaction_earnings',
    spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + earnings interaction',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Earnings interaction: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# By church membership (from paper's Table 8)
median_church = df['blackmemrate'].median()
df['high_church'] = (df['blackmemrate'] >= median_church).astype(int)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR}*high_church + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/interaction_church',
    spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + church membership interaction',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Church interaction: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# By incarceration rate
median_incarc = df['incarceration_2010'].median()
df['high_incarceration'] = (df['incarceration_2010'] >= median_incarc).astype(int)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR}*high_incarceration + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/interaction_incarceration',
    spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + incarceration interaction',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Incarceration interaction: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# By historical slave share
median_slaves = df['share_slaves'].median()
df['high_slaves'] = (df['share_slaves'] >= median_slaves).astype(int)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR}*high_slaves + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/interaction_slavery',
    spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + slavery share interaction',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Slavery interaction: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Sample splits by education
df_low_edu = df[df['Black_beyondhs'] < median_edu]
df_high_edu = df[df['Black_beyondhs'] >= median_edu]

formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df_low_edu, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/by_education_low',
    spec_tree_path='robustness/heterogeneity.md#socioeconomic-subgroups',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Low education counties',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Low education sample: coef = {model.coef()[TREATMENT_VAR]:.4f}")

model = pf.feols(formula, data=df_high_edu, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/by_education_high',
    spec_tree_path='robustness/heterogeneity.md#socioeconomic-subgroups',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='High education counties',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  High education sample: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# By black population share
median_blackshare = df['blackshare_current'].median()
df_low_black = df[df['blackshare_current'] < median_blackshare]
df_high_black = df[df['blackshare_current'] >= median_blackshare]

model = pf.feols(formula, data=df_low_black, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/by_blackshare_low',
    spec_tree_path='robustness/heterogeneity.md#demographic-subgroups',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Low black population share counties',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Low black share: coef = {model.coef()[TREATMENT_VAR]:.4f}")

model = pf.feols(formula, data=df_high_black, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/het/by_blackshare_high',
    spec_tree_path='robustness/heterogeneity.md#demographic-subgroups',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='High black population share counties',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  High black share: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# =============================================================================
# 9. ESTIMATION METHOD VARIATIONS (5 specs)
# =============================================================================
print("\n9. Running estimation method variations...")

# No fixed effects (pooled OLS)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS)
df_model = df.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR] + HISTORICAL_CONTROLS)
X = sm.add_constant(df_model[[TREATMENT_VAR] + HISTORICAL_CONTROLS])
y = df_model[OUTCOME_VAR]
model = sm.OLS(y, X).fit(cov_type='HC1')
results.append(extract_results(
    model, spec_id='robust/estimation/no_fe',
    spec_tree_path='robustness/model_specification.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='None',
    cluster_var='HC1 robust',
    model_type='Pooled OLS'
))
print(f"  No FE: coef = {model.params[TREATMENT_VAR]:.4f}")

# State dummies explicitly (same as FE but explicit)
formula_dummies = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + " + C(State_FIPS)"
model = pf.feols(formula_dummies, data=df, vcov='hetero')
# Coefficient should match baseline
print(f"  Explicit state dummies: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# WLS weighted by population
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero', weights='pop')
results.append(extract_results(
    model, spec_id='robust/weights/population',
    spec_tree_path='robustness/model_specification.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='WLS (population weighted)'
))
print(f"  Population weighted: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# WLS weighted by black population
model = pf.feols(formula, data=df, vcov='hetero', weights='Black_POP_1900')
results.append(extract_results(
    model, spec_id='robust/weights/black_population',
    spec_tree_path='robustness/model_specification.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='WLS (black population weighted)'
))
print(f"  Black population weighted: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Quantile regression (median)
from statsmodels.regression.quantile_regression import QuantReg
df_model = df.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR] + HISTORICAL_CONTROLS)
# Add state dummies
state_dummies = pd.get_dummies(df_model['State_FIPS'], prefix='state', drop_first=True)
X = pd.concat([df_model[[TREATMENT_VAR] + HISTORICAL_CONTROLS].reset_index(drop=True),
               state_dummies.reset_index(drop=True)], axis=1)
X = sm.add_constant(X)
X = X.astype(float)
y = df_model[OUTCOME_VAR].reset_index(drop=True).astype(float)
model = QuantReg(y, X).fit(q=0.5)
results.append(extract_results(
    model, spec_id='robust/form/quantile_50',
    spec_tree_path='robustness/functional_form.md#alternative-estimators',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State dummies',
    cluster_var='Quantile SE',
    model_type='Median regression (LAD)'
))
print(f"  Median regression: coef = {model.params[TREATMENT_VAR]:.4f}")

# 25th percentile
model = QuantReg(y, X).fit(q=0.25)
results.append(extract_results(
    model, spec_id='robust/form/quantile_25',
    spec_tree_path='robustness/functional_form.md#alternative-estimators',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State dummies',
    cluster_var='Quantile SE',
    model_type='25th percentile regression'
))
print(f"  25th percentile: coef = {model.params[TREATMENT_VAR]:.4f}")

# 75th percentile
model = QuantReg(y, X).fit(q=0.75)
results.append(extract_results(
    model, spec_id='robust/form/quantile_75',
    spec_tree_path='robustness/functional_form.md#alternative-estimators',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State dummies',
    cluster_var='Quantile SE',
    model_type='75th percentile regression'
))
print(f"  75th percentile: coef = {model.params[TREATMENT_VAR]:.4f}")

# =============================================================================
# 10. FUNCTIONAL FORM VARIATIONS (5 specs)
# =============================================================================
print("\n10. Running functional form variations...")

# Quadratic in treatment
df['lynchcapitamob_sq'] = df[TREATMENT_VAR] ** 2
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + lynchcapitamob_sq + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/quadratic',
    spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + treatment squared',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE (quadratic)'
))
print(f"  Quadratic: linear coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Log treatment (add 1 to handle zeros)
df['log_lynch'] = np.log(df[TREATMENT_VAR] + 1)
formula = f"{OUTCOME_VAR} ~ log_lynch + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/x_log',
    spec_tree_path='robustness/functional_form.md#treatment-variable-transformations',
    outcome_var=OUTCOME_VAR, treatment_var='log_lynch',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Log treatment: coef = {model.coef()['log_lynch']:.4f}")

# IHS treatment
df['ihs_lynch'] = np.arcsinh(df[TREATMENT_VAR])
formula = f"{OUTCOME_VAR} ~ ihs_lynch + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/x_ihs',
    spec_tree_path='robustness/functional_form.md#treatment-variable-transformations',
    outcome_var=OUTCOME_VAR, treatment_var='ihs_lynch',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  IHS treatment: coef = {model.coef()['ihs_lynch']:.4f}")

# Binary treatment (any lynching vs none)
df['lynch_binary'] = (df[TREATMENT_VAR] > 0).astype(int)
formula = f"{OUTCOME_VAR} ~ lynch_binary + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/x_binary',
    spec_tree_path='robustness/functional_form.md#treatment-variable-transformations',
    outcome_var=OUTCOME_VAR, treatment_var='lynch_binary',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Binary treatment: coef = {model.coef()['lynch_binary']:.4f}")

# Treatment terciles
df['lynch_tercile'] = pd.qcut(df[TREATMENT_VAR], 3, labels=['low', 'mid', 'high'])
df_tercile = pd.get_dummies(df['lynch_tercile'], prefix='lynch', drop_first=True)
df = pd.concat([df, df_tercile], axis=1)
formula = f"{OUTCOME_VAR} ~ lynch_mid + lynch_high + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/x_terciles',
    spec_tree_path='robustness/functional_form.md#treatment-variable-transformations',
    outcome_var=OUTCOME_VAR, treatment_var='lynch_high',
    sample_desc='Full sample',
    controls_desc='Historical controls',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Terciles (high vs low): coef = {model.coef()['lynch_high']:.4f}")

# =============================================================================
# 11. ADDITIONAL ROBUSTNESS (4 specs)
# =============================================================================
print("\n11. Running additional specifications...")

# Including slavery control (Table 3, Column 2 in paper)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + share_slaves + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/control/add_slavery',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + slave share',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  With slavery control: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Table 4 specification (with shareblack)
formula = f"register_black ~ {TREATMENT_VAR} + share_slaves + shareblack + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/control/with_shareblack',
    spec_tree_path='robustness/control_progression.md',
    outcome_var='register_black', treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Historical controls + slave share + black share',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  With black share control: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Table 5 specification 5 (full set of contemporary controls)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + incarceration_2010 + pollscapita + " + \
          " + ".join(HISTORICAL_CONTROLS) + " + " + " + ".join(CONTEMPORARY_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/control/table5_col5',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=OUTCOME_VAR, treatment_var=TREATMENT_VAR,
    sample_desc='Full sample',
    controls_desc='Full controls including incarceration and polling places',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Table 5 col 5 spec: coef = {model.coef()[TREATMENT_VAR]:.4f}")

# Standardized coefficient
df['lynch_std'] = (df[TREATMENT_VAR] - df[TREATMENT_VAR].mean()) / df[TREATMENT_VAR].std()
df['blackrate_std'] = (df[OUTCOME_VAR] - df[OUTCOME_VAR].mean()) / df[OUTCOME_VAR].std()
formula = f"blackrate_std ~ lynch_std + " + " + ".join(HISTORICAL_CONTROLS) + f" | {FE_VAR}"
model = pf.feols(formula, data=df, vcov='hetero')
results.append(extract_results(
    model, spec_id='robust/form/standardized',
    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
    outcome_var='blackrate_std', treatment_var='lynch_std',
    sample_desc='Full sample',
    controls_desc='Historical controls (standardized Y and X)',
    fixed_effects='State FE',
    cluster_var='Robust (heteroskedasticity-consistent)',
    model_type='OLS with FE'
))
print(f"  Standardized: coef = {model.coef()['lynch_std']:.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print(f"\n{'='*60}")
print(f"TOTAL SPECIFICATIONS RUN: {len(results)}")
print(f"{'='*60}")

# Create DataFrame
results_df = pd.DataFrame(results)

# Save to package directory
output_path = f'{DATA_DIR}/../specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Summary statistics
print(f"\nSummary Statistics:")
print(f"  Total specifications: {len(results_df)}")
print(f"  Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"  Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"  Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

# For main treatment variable only
main_results = results_df[results_df['treatment_var'] == TREATMENT_VAR]
print(f"\nFor main treatment ({TREATMENT_VAR}):")
print(f"  Specifications: {len(main_results)}")
print(f"  Median coefficient: {main_results['coefficient'].median():.4f}")
print(f"  Mean coefficient: {main_results['coefficient'].mean():.4f}")
print(f"  Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")
print(f"  Significant at 5%: {(main_results['p_value'] < 0.05).sum()} ({100*(main_results['p_value'] < 0.05).mean():.1f}%)")

print("\nSpecification search complete!")
