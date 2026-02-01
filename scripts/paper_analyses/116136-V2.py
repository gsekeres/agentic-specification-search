"""
Specification Search Analysis for Paper 116136-V2
"Yours, Mine and Ours: Do Divorce Laws Affect the Intertemporal Behavior of Married Couples?"
Alessandra Voena

Method: Panel Fixed Effects / Difference-in-Differences
Treatment: Unilateral divorce law adoption interacted with property division regimes
Outcome: Household assets (NLSW data)
Identification: Staggered adoption of unilateral divorce across US states
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to linearmodels if needed
try:
    import pyfixest as pf
    USE_PYFIXEST = True
except ImportError:
    USE_PYFIXEST = False

try:
    from linearmodels.panel import PanelOLS, RandomEffects
    USE_LINEARMODELS = True
except ImportError:
    USE_LINEARMODELS = False

import statsmodels.api as sm
from scipy import stats

# Paths
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116136-V2/replicate_empirics/NLSW_women.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116136-V2/specification_results.csv"

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Data shape: {df.shape}")

# Define key variables based on Stata code
# Treatment variables: uni_comprop, uni_title, uni_eqdistr (unilateral divorce interacted with property regime)
# Property regime indicators: comprop, eqdistr
# Outcome: assets
# Individual FE: id
# Clustering: state

# Create control variable lists
# d_age* are age dummies
# yrd* are year dummies
# chd* are children-related controls
# std* are state dummies (but we'll use FE instead)
# since1marr* are years since first marriage

age_dummies = [c for c in df.columns if c.startswith('d_age')]
year_dummies = [c for c in df.columns if c.startswith('yrd')]
child_dummies = [c for c in df.columns if c.startswith('chd')]
since_marr_vars = ['since1marr'] + [c for c in df.columns if c.startswith('since1marr_')]

print(f"Age dummies: {len(age_dummies)}")
print(f"Year dummies: {len(year_dummies)}")
print(f"Child dummies: {len(child_dummies)}")
print(f"Marriage duration vars: {len(since_marr_vars)}")

# Define treatment and outcome
TREATMENT_VARS = ['uni_comprop', 'uni_title', 'uni_eqdistr']
PROPERTY_REGIME_VARS = ['comprop', 'eqdistr']
OUTCOME = 'assets'
UNIT_ID = 'id'
STATE_VAR = 'state'
YEAR_VAR = 'year'

# Results storage
results = []

def extract_results(model_result, spec_id, spec_tree_path, notes="", **kwargs):
    """Extract results from regression model into standard format"""
    result_dict = {
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'notes': notes,
    }

    # Add any extra kwargs
    result_dict.update(kwargs)

    # Try to extract key treatment coefficients
    for tvar in TREATMENT_VARS:
        try:
            if hasattr(model_result, 'params'):
                if tvar in model_result.params.index:
                    result_dict[f'{tvar}_coef'] = model_result.params[tvar]
                    if hasattr(model_result, 'std_errors'):
                        result_dict[f'{tvar}_se'] = model_result.std_errors[tvar]
                    elif hasattr(model_result, 'bse'):
                        result_dict[f'{tvar}_se'] = model_result.bse[tvar]
                    if hasattr(model_result, 'pvalues'):
                        result_dict[f'{tvar}_pval'] = model_result.pvalues[tvar]
        except:
            pass

    # Extract model fit stats
    try:
        if hasattr(model_result, 'nobs'):
            result_dict['n_obs'] = int(model_result.nobs)
        elif hasattr(model_result, 'df_resid'):
            result_dict['n_obs'] = int(model_result.df_resid + model_result.df_model + 1)
    except:
        pass

    try:
        if hasattr(model_result, 'rsquared'):
            result_dict['r_squared'] = model_result.rsquared
        elif hasattr(model_result, 'r2'):
            result_dict['r_squared'] = model_result.r2
    except:
        pass

    try:
        if hasattr(model_result, 'rsquared_within'):
            result_dict['r_squared_within'] = model_result.rsquared_within
    except:
        pass

    return result_dict


def run_pyfixest_regression(formula, data, vcov=None, spec_id="", spec_tree_path="", notes=""):
    """Run regression using pyfixest"""
    try:
        if vcov is None:
            result = pf.feols(formula, data=data)
        else:
            result = pf.feols(formula, data=data, vcov=vcov)

        # Extract results
        res_dict = {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'notes': notes,
        }

        params = result.coef()
        se = result.se()
        pvals = result.pvalue()

        for tvar in TREATMENT_VARS:
            if tvar in params.index:
                res_dict[f'{tvar}_coef'] = float(params[tvar])
                res_dict[f'{tvar}_se'] = float(se[tvar])
                res_dict[f'{tvar}_pval'] = float(pvals[tvar])

        for pvar in PROPERTY_REGIME_VARS:
            if pvar in params.index:
                res_dict[f'{pvar}_coef'] = float(params[pvar])
                res_dict[f'{pvar}_se'] = float(se[pvar])
                res_dict[f'{pvar}_pval'] = float(pvals[pvar])

        # Get number of observations
        res_dict['n_obs'] = int(result._N)

        # Get R-squared from tidy() or summary
        try:
            tidy_df = result.tidy()
            # R2 is in the summary output
            res_dict['r_squared'] = None  # pyfixest doesn't easily expose R2
        except:
            res_dict['r_squared'] = None

        # Store full coefficient vector as JSON
        coef_dict = {
            'treatment': {k: {'coef': float(params[k]), 'se': float(se[k]), 'pval': float(pvals[k])}
                         for k in TREATMENT_VARS if k in params.index},
            'controls': {k: {'coef': float(params[k]), 'se': float(se[k]), 'pval': float(pvals[k])}
                        for k in params.index if k not in TREATMENT_VARS},
        }
        res_dict['coefficient_vector_json'] = json.dumps(coef_dict)

        return res_dict
    except Exception as e:
        return {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'notes': f"ERROR: {str(e)}",
            'error': str(e)
        }


def run_linearmodels_regression(y, X, data, entity_effects=True, time_effects=False,
                                 cluster_entity=False, cluster_var=None,
                                 spec_id="", spec_tree_path="", notes=""):
    """Run regression using linearmodels"""
    try:
        # Prepare panel data
        panel_df = data.copy()
        panel_df = panel_df.set_index([UNIT_ID, YEAR_VAR])

        # Build formula
        if entity_effects and time_effects:
            formula = f"{y} ~ 1 + {' + '.join(X)} + EntityEffects + TimeEffects"
        elif entity_effects:
            formula = f"{y} ~ 1 + {' + '.join(X)} + EntityEffects"
        elif time_effects:
            formula = f"{y} ~ 1 + {' + '.join(X)} + TimeEffects"
        else:
            formula = f"{y} ~ 1 + {' + '.join(X)}"

        model = PanelOLS.from_formula(formula, data=panel_df, drop_absorbed=True)

        if cluster_var is not None:
            # Add cluster variable to index
            cluster_data = data[[cluster_var]].values
            result = model.fit(cov_type='clustered', cluster_entity=True)
        elif cluster_entity:
            result = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            result = model.fit(cov_type='robust')

        # Extract results
        res_dict = {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'notes': notes,
        }

        params = result.params
        se = result.std_errors
        pvals = result.pvalues

        for tvar in TREATMENT_VARS:
            if tvar in params.index:
                res_dict[f'{tvar}_coef'] = params[tvar]
                res_dict[f'{tvar}_se'] = se[tvar]
                res_dict[f'{tvar}_pval'] = pvals[tvar]

        for pvar in PROPERTY_REGIME_VARS:
            if pvar in params.index:
                res_dict[f'{pvar}_coef'] = params[pvar]
                res_dict[f'{pvar}_se'] = se[pvar]
                res_dict[f'{pvar}_pval'] = pvals[pvar]

        res_dict['n_obs'] = int(result.nobs)
        res_dict['r_squared'] = result.rsquared
        res_dict['r_squared_within'] = result.rsquared_within if hasattr(result, 'rsquared_within') else None

        return res_dict
    except Exception as e:
        return {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'notes': f"ERROR: {str(e)}",
            'error': str(e)
        }


# Prepare analysis dataset
print("\nPreparing analysis dataset...")
# Drop missing outcomes
analysis_df = df.dropna(subset=[OUTCOME]).copy()
print(f"Observations with non-missing assets: {len(analysis_df)}")

# Ensure treatment and key variables are present
for var in TREATMENT_VARS + PROPERTY_REGIME_VARS:
    analysis_df[var] = analysis_df[var].fillna(0)

# Create numeric state variable for clustering
analysis_df['state_num'] = analysis_df[STATE_VAR].astype(int)

# Get year dummy columns that exist and are non-zero
year_dummy_cols = [c for c in year_dummies if c in analysis_df.columns and analysis_df[c].sum() > 0]
# Keep only a subset to avoid collinearity
year_dummy_cols = year_dummy_cols[1:]  # Drop first to avoid collinearity

# Get age dummy columns
age_dummy_cols = [c for c in age_dummies if c in analysis_df.columns and analysis_df[c].sum() > 0]
age_dummy_cols = age_dummy_cols[1:]  # Drop first to avoid collinearity

# Get child dummy columns
child_dummy_cols = [c for c in child_dummies if c in analysis_df.columns and analysis_df[c].sum() > 0]

print(f"Using {len(year_dummy_cols)} year dummies")
print(f"Using {len(age_dummy_cols)} age dummies")
print(f"Using {len(child_dummy_cols)} child dummies")

# ============================================
# BASELINE SPECIFICATION (Replication)
# ============================================
print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

# Based on Stata code:
# xtreg assets uni_comprop uni_title uni_eqdistr comprop eqdistr d_age* yrd*, fe i(id) cluster(state)

baseline_controls = TREATMENT_VARS + PROPERTY_REGIME_VARS + age_dummy_cols + year_dummy_cols

if USE_PYFIXEST:
    # Build formula for pyfixest
    rhs = ' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)
    age_controls = ' + '.join(age_dummy_cols[:10])  # Limit to avoid issues
    year_controls = ' + '.join(year_dummy_cols[:10])

    # Formula with individual FE
    formula = f"{OUTCOME} ~ {rhs} + {age_controls} + {year_controls} | {UNIT_ID}"

    print(f"Running baseline with pyfixest...")
    baseline_result = run_pyfixest_regression(
        formula,
        analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="baseline",
        spec_tree_path="methods/panel_fixed_effects.md#baseline",
        notes="Exact replication of Table 1 Column 1: assets ~ uni_comprop + uni_title + uni_eqdistr + comprop + eqdistr + age_dummies + year_dummies, FE=id, cluster=state"
    )
    results.append(baseline_result)
    print(f"Baseline results: uni_comprop coef = {baseline_result.get('uni_comprop_coef', 'N/A')}")

elif USE_LINEARMODELS:
    print(f"Running baseline with linearmodels...")
    X_vars = TREATMENT_VARS + PROPERTY_REGIME_VARS + age_dummy_cols[:20] + year_dummy_cols[:20]
    baseline_result = run_linearmodels_regression(
        OUTCOME, X_vars, analysis_df,
        entity_effects=True, time_effects=False,
        cluster_entity=True,
        spec_id="baseline",
        spec_tree_path="methods/panel_fixed_effects.md#baseline",
        notes="Exact replication of Table 1 Column 1"
    )
    results.append(baseline_result)
    print(f"Baseline results: uni_comprop coef = {baseline_result.get('uni_comprop_coef', 'N/A')}")

else:
    print("ERROR: Neither pyfixest nor linearmodels available")


# ============================================
# PANEL FE VARIATIONS (from specification tree)
# ============================================
print("\n" + "="*60)
print("PANEL FE VARIATIONS")
print("="*60)

# Spec: panel/fe/none - Pooled OLS (no FE)
print("\nRunning panel/fe/none (Pooled OLS)...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="panel/fe/none",
        spec_tree_path="methods/panel_fixed_effects.md#fixed-effects-structure",
        notes="Pooled OLS with no fixed effects"
    )
    results.append(result)

# Spec: panel/fe/unit - Unit FE only (this is the baseline)
print("\nRunning panel/fe/unit (Unit FE only)...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="panel/fe/unit",
        spec_tree_path="methods/panel_fixed_effects.md#fixed-effects-structure",
        notes="Unit (individual) FE only"
    )
    results.append(result)

# Spec: panel/fe/time - Time FE only
print("\nRunning panel/fe/time (Time FE only)...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {YEAR_VAR}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="panel/fe/time",
        spec_tree_path="methods/panel_fixed_effects.md#fixed-effects-structure",
        notes="Time (year) FE only"
    )
    results.append(result)

# Spec: panel/fe/twoway - Two-way FE
print("\nRunning panel/fe/twoway (Two-way FE)...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID} + {YEAR_VAR}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="panel/fe/twoway",
        spec_tree_path="methods/panel_fixed_effects.md#fixed-effects-structure",
        notes="Two-way FE: individual + year"
    )
    results.append(result)


# ============================================
# CONTROL SET VARIATIONS
# ============================================
print("\n" + "="*60)
print("CONTROL SET VARIATIONS")
print("="*60)

# Spec: panel/controls/none - No controls
print("\nRunning panel/controls/none...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="panel/controls/none",
        spec_tree_path="methods/panel_fixed_effects.md#control-sets",
        notes="No controls beyond treatment and FE"
    )
    results.append(result)

# Spec: did/controls/baseline - With age dummies and year dummies (from Stata Column 1)
print("\nRunning did/controls/baseline (age + year dummies)...")
if USE_PYFIXEST:
    age_str = ' + '.join(age_dummy_cols[:15])
    year_str = ' + '.join(year_dummy_cols[:15])
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} + {age_str} + {year_str} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="did/controls/baseline",
        spec_tree_path="methods/difference_in_differences.md#control-sets",
        notes="Baseline controls: age dummies + year dummies"
    )
    results.append(result)

# Spec: did/controls/full - With age, year, children dummies (Stata Column 2)
print("\nRunning did/controls/full (age + year + children)...")
if USE_PYFIXEST:
    child_str = ' + '.join(child_dummy_cols[:10])
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} + {age_str} + {year_str} + {child_str} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="did/controls/full",
        spec_tree_path="methods/difference_in_differences.md#control-sets",
        notes="Full controls: age + year + children dummies"
    )
    results.append(result)


# ============================================
# CLUSTERING VARIATIONS
# ============================================
print("\n" + "="*60)
print("CLUSTERING VARIATIONS")
print("="*60)

# Spec: robust/cluster/none - No clustering (robust SE)
print("\nRunning robust/cluster/none (robust SE)...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov='hetero',
        spec_id="robust/cluster/none",
        spec_tree_path="robustness/clustering_variations.md#single-level-clustering",
        notes="Heteroskedasticity-robust SE (no clustering)"
    )
    results.append(result)

# Spec: robust/cluster/unit - Cluster by individual
print("\nRunning robust/cluster/unit...")
if USE_PYFIXEST:
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'id'},
        spec_id="robust/cluster/unit",
        spec_tree_path="robustness/clustering_variations.md#single-level-clustering",
        notes="Clustered by individual"
    )
    results.append(result)

# Spec: robust/cluster/state - Cluster by state (baseline)
print("\nRunning robust/cluster/state (baseline clustering)...")
if USE_PYFIXEST:
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="robust/cluster/state",
        spec_tree_path="robustness/clustering_variations.md#single-level-clustering",
        notes="Clustered by state (baseline)"
    )
    results.append(result)


# ============================================
# SAMPLE RESTRICTIONS
# ============================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# Get time periods
median_year = analysis_df[YEAR_VAR].median()
print(f"Median year: {median_year}")

# Spec: did/sample/early_period - First half of sample
print("\nRunning did/sample/early_period...")
early_df = analysis_df[analysis_df[YEAR_VAR] <= median_year]
if USE_PYFIXEST and len(early_df) > 100:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, early_df,
        vcov={'CRV1': 'state_num'},
        spec_id="did/sample/early_period",
        spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
        notes=f"Early period only (year <= {median_year})"
    )
    results.append(result)

# Spec: did/sample/late_period - Second half of sample
print("\nRunning did/sample/late_period...")
late_df = analysis_df[analysis_df[YEAR_VAR] > median_year].copy()
late_df['state_num'] = late_df[STATE_VAR].astype(int)
if USE_PYFIXEST and len(late_df) > 100:
    late_formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        late_formula, late_df,
        vcov={'CRV1': 'state_num'},
        spec_id="did/sample/late_period",
        spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
        notes=f"Late period only (year > {median_year})"
    )
    results.append(result)

# Spec: panel/sample/balanced - Balanced panel only
print("\nRunning panel/sample/balanced...")
obs_counts = analysis_df.groupby(UNIT_ID).size()
max_periods = obs_counts.max()
balanced_ids = obs_counts[obs_counts == max_periods].index
balanced_df = analysis_df[analysis_df[UNIT_ID].isin(balanced_ids)]
print(f"Balanced panel: {len(balanced_df)} obs, {len(balanced_ids)} individuals")
if USE_PYFIXEST and len(balanced_df) > 100:
    result = run_pyfixest_regression(
        formula, balanced_df,
        vcov={'CRV1': 'state_num'},
        spec_id="panel/sample/balanced",
        spec_tree_path="methods/panel_fixed_effects.md#sample-restrictions",
        notes=f"Balanced panel only ({len(balanced_ids)} individuals with {max_periods} periods)"
    )
    results.append(result)

# Spec: did/sample/comprop_only - Community property states only (as in Stata code)
print("\nRunning did/sample/comprop_only...")
comprop_df = analysis_df[analysis_df['comprop'] == 1]
print(f"Community property states: {len(comprop_df)} obs")
if USE_PYFIXEST and len(comprop_df) > 100:
    # For comprop states, the relevant treatment is just uni_comprop
    formula = f"{OUTCOME} ~ uni_comprop | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, comprop_df,
        vcov={'CRV1': 'state_num'},
        spec_id="did/sample/comprop_only",
        spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
        notes="Community property states only"
    )
    results.append(result)


# ============================================
# LEAVE-ONE-OUT ROBUSTNESS
# ============================================
print("\n" + "="*60)
print("LEAVE-ONE-OUT ROBUSTNESS")
print("="*60)

# Drop each treatment variable one at a time
for drop_var in TREATMENT_VARS:
    remaining_treat = [v for v in TREATMENT_VARS if v != drop_var]
    print(f"\nRunning robust/loo/drop_{drop_var}...")
    if USE_PYFIXEST:
        formula = f"{OUTCOME} ~ {' + '.join(remaining_treat + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
        result = run_pyfixest_regression(
            formula, analysis_df,
            vcov={'CRV1': 'state_num'},
            spec_id=f"robust/loo/drop_{drop_var}",
            spec_tree_path="robustness/leave_one_out.md",
            notes=f"Drop {drop_var} from treatment variables"
        )
        results.append(result)

# Drop property regime controls
for drop_var in PROPERTY_REGIME_VARS:
    remaining_prop = [v for v in PROPERTY_REGIME_VARS if v != drop_var]
    print(f"\nRunning robust/loo/drop_{drop_var}...")
    if USE_PYFIXEST:
        formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS + remaining_prop)} | {UNIT_ID}"
        result = run_pyfixest_regression(
            formula, analysis_df,
            vcov={'CRV1': 'state_num'},
            spec_id=f"robust/loo/drop_{drop_var}",
            spec_tree_path="robustness/leave_one_out.md",
            notes=f"Drop {drop_var} from controls"
        )
        results.append(result)


# ============================================
# SINGLE COVARIATE ROBUSTNESS
# ============================================
print("\n" + "="*60)
print("SINGLE COVARIATE ROBUSTNESS")
print("="*60)

# Treatment only (bivariate with FE)
print("\nRunning robust/single/none...")
if USE_PYFIXEST:
    formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="robust/single/none",
        spec_tree_path="robustness/single_covariate.md",
        notes="Treatment variables only (no property regime controls)"
    )
    results.append(result)

# Treatment + each property regime
for pvar in PROPERTY_REGIME_VARS:
    print(f"\nRunning robust/single/{pvar}...")
    if USE_PYFIXEST:
        formula = f"{OUTCOME} ~ {' + '.join(TREATMENT_VARS)} + {pvar} | {UNIT_ID}"
        result = run_pyfixest_regression(
            formula, analysis_df,
            vcov={'CRV1': 'state_num'},
            spec_id=f"robust/single/{pvar}",
            spec_tree_path="robustness/single_covariate.md",
            notes=f"Treatment + {pvar} only"
        )
        results.append(result)


# ============================================
# FUNCTIONAL FORM VARIATIONS
# ============================================
print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# Spec: robust/form/y_log - Log outcome
print("\nRunning robust/form/y_log...")
# Assets can be negative, so use asinh or shift
analysis_df['log_assets'] = np.log(analysis_df[OUTCOME] + 1 - analysis_df[OUTCOME].min())
if USE_PYFIXEST:
    formula = f"log_assets ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="robust/form/y_log",
        spec_tree_path="robustness/functional_form.md#outcome-variable-transformations",
        notes="Log(assets + shift) transformation"
    )
    results.append(result)

# Spec: robust/form/y_asinh - Inverse hyperbolic sine (handles zeros and negatives)
print("\nRunning robust/form/y_asinh...")
analysis_df['asinh_assets'] = np.arcsinh(analysis_df[OUTCOME])
if USE_PYFIXEST:
    formula = f"asinh_assets ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="robust/form/y_asinh",
        spec_tree_path="robustness/functional_form.md#outcome-variable-transformations",
        notes="Inverse hyperbolic sine transformation"
    )
    results.append(result)

# Spec: robust/form/y_standardized - Standardized outcome
print("\nRunning robust/form/y_standardized...")
analysis_df['std_assets'] = (analysis_df[OUTCOME] - analysis_df[OUTCOME].mean()) / analysis_df[OUTCOME].std()
if USE_PYFIXEST:
    formula = f"std_assets ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="robust/form/y_standardized",
        spec_tree_path="robustness/functional_form.md#outcome-variable-transformations",
        notes="Standardized (z-score) outcome"
    )
    results.append(result)

# Spec: robust/form/y_winsorized - Winsorized outcome (1st/99th percentile)
print("\nRunning robust/form/y_winsorized...")
p01 = analysis_df[OUTCOME].quantile(0.01)
p99 = analysis_df[OUTCOME].quantile(0.99)
analysis_df['wins_assets'] = analysis_df[OUTCOME].clip(lower=p01, upper=p99)
if USE_PYFIXEST:
    formula = f"wins_assets ~ {' + '.join(TREATMENT_VARS + PROPERTY_REGIME_VARS)} | {UNIT_ID}"
    result = run_pyfixest_regression(
        formula, analysis_df,
        vcov={'CRV1': 'state_num'},
        spec_id="robust/form/y_winsorized",
        spec_tree_path="robustness/functional_form.md#outcome-variable-transformations",
        notes="Winsorized at 1st/99th percentile"
    )
    results.append(result)


# ============================================
# DiD-SPECIFIC VARIATIONS
# ============================================
print("\n" + "="*60)
print("DiD-SPECIFIC VARIATIONS")
print("="*60)

# Spec: did/method/twfe - Standard TWFE (already done as panel/fe/twoway)
# Spec: did/treatment/binary - Already using binary treatment

# Check for event study variables
if 'time_unilateral' in analysis_df.columns:
    print("\nRunning event study analysis...")
    # Create event time dummies
    analysis_df['event_time'] = analysis_df['time_unilateral']

    # Create binned event time dummies (following Stata code)
    for x in range(1, 14, 3):
        varname = f'tduni{x}'
        analysis_df[varname] = ((analysis_df['time_unilateral'] >= x) &
                                (analysis_df['time_unilateral'] <= x + 2)).astype(int)
    analysis_df['tduni13'] = (analysis_df['time_unilateral'] > 13).astype(int) | analysis_df['tduni13']

    # Pre-treatment indicator
    analysis_df['totduni1'] = ((analysis_df['totime_unilateral'] == 1) |
                               (analysis_df['totime_unilateral'] == 2)).astype(int)

    # Run event study on comprop states
    comprop_df = analysis_df[analysis_df['comprop'] == 1].copy()
    event_vars = ['totduni1', 'tduni1', 'tduni4', 'tduni7', 'tduni10', 'tduni13']
    event_vars_exist = [v for v in event_vars if v in comprop_df.columns]

    if len(event_vars_exist) > 0 and USE_PYFIXEST:
        formula = f"{OUTCOME} ~ {' + '.join(event_vars_exist)} | {UNIT_ID}"
        result = run_pyfixest_regression(
            formula, comprop_df,
            vcov={'CRV1': 'state_num'},
            spec_id="did/dynamic/leads_lags",
            spec_tree_path="methods/difference_in_differences.md#dynamic-effects-event-study",
            notes="Event study with binned post-treatment periods (comprop states only)"
        )
        results.append(result)


# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")
print(f"Total specifications run: {len(results)}")

# Print summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for r in results:
    coef = r.get('uni_comprop_coef', 'N/A')
    se = r.get('uni_comprop_se', 'N/A')
    pval = r.get('uni_comprop_pval', 'N/A')
    if coef != 'N/A':
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"{r['spec_id']:40s}: uni_comprop = {coef:10.2f} ({se:.2f}){sig}")
    else:
        print(f"{r['spec_id']:40s}: {r.get('notes', 'No results')[:50]}")

print("\nDone!")
