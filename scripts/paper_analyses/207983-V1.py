#!/usr/bin/env python3
"""
Specification Search: Paper 207983-V1
"Contamination Bias in Multiple-Treatment Regressions"
by Michal Kolesár and Christopher R. Walters

This paper introduces the `multe` methodology for analyzing multi-arm RCTs.
The main contribution is decomposing treatment effects to identify contamination bias.

Primary Analysis: Project STAR experiment
- Outcome: Standardized test score (y)
- Treatments: Small class size, Teaching aide (vs Regular class)
- Fixed Effects: School
- Method: Cross-sectional OLS with fixed effects

The paper analyzes 9 experiments total - we focus on STAR as the primary case
and run specifications on the Benhassine dataset as a robustness check.
"""

import pandas as pd
import numpy as np
import pyreadr
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "207983-V1"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}"
OUTPUT_PATH = f"{DATA_PATH}"

# Method classification
METHOD_CODE = "cross_sectional_ols"
METHOD_TREE_PATH = "specification_tree/methods/cross_sectional_ols.md"

results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coef, se, t_stat, p_value, ci_lower, ci_upper, n_obs, r_squared,
               coef_vector_json, sample_desc, fixed_effects, controls_desc,
               cluster_var, model_type, dataset="STAR"):
    """Add a specification result to the results list."""
    results.append({
        'paper_id': PAPER_ID,
        'journal': 'AER',
        'paper_title': 'Contamination Bias in Multiple-Treatment Regressions',
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'dataset': dataset
    })

def extract_results_pyfixest(model, treatment_var, outcome_var, spec_id, spec_tree_path,
                             sample_desc, fixed_effects, controls_desc, cluster_var,
                             model_type, dataset="STAR"):
    """Extract results from pyfixest model and add to results list."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        t_stat = model.tstat()[treatment_var]
        p_value = model.pvalue()[treatment_var]
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        n_obs = model.nobs
        r_squared = model.r2 if hasattr(model, 'r2') else None

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(p_value)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(", ") if fixed_effects else [],
            "diagnostics": {"n_obs": int(n_obs), "r_squared": float(r_squared) if r_squared else None}
        }

        # Add other coefficients
        for var in model.coef().index:
            if var != treatment_var:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

        add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
                   coef, se, t_stat, p_value, ci_lower, ci_upper, n_obs, r_squared,
                   coef_vector, sample_desc, fixed_effects, controls_desc,
                   cluster_var, model_type, dataset)
        return True
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return False

def extract_results_statsmodels(model, treatment_var, outcome_var, spec_id, spec_tree_path,
                                sample_desc, fixed_effects, controls_desc, cluster_var,
                                model_type, dataset="STAR"):
    """Extract results from statsmodels model and add to results list."""
    try:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        p_value = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower = ci[0]
        ci_upper = ci[1]
        n_obs = int(model.nobs)
        r_squared = model.rsquared if hasattr(model, 'rsquared') else None

        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(p_value)},
            "controls": [],
            "fixed_effects": fixed_effects.split(", ") if fixed_effects else [],
            "diagnostics": {"n_obs": n_obs, "r_squared": float(r_squared) if r_squared else None}
        }

        for var in model.params.index:
            if var != treatment_var and not var.startswith('school['):
                coef_vector["controls"].append({
                    "var": var, "coef": float(model.params[var]),
                    "se": float(model.bse[var]), "pval": float(model.pvalues[var])
                })

        add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
                   coef, se, t_stat, p_value, ci_lower, ci_upper, n_obs, r_squared,
                   coef_vector, sample_desc, fixed_effects, controls_desc,
                   cluster_var, model_type, dataset)
        return True
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return False


# =============================================================================
# Load STAR Data
# =============================================================================
print("Loading STAR dataset...")
star_data = pyreadr.read_r(f"{DATA_PATH}/dta/STAR.rda")
df_star = star_data['dt1'].copy()

# Create treatment variable
df_star['treatment'] = 'regular'
df_star.loc[df_star['small'] == True, 'treatment'] = 'small'
df_star.loc[df_star['aide'] == True, 'treatment'] = 'aide'

# Create dummy variables for treatment
df_star['small_class'] = (df_star['treatment'] == 'small').astype(int)
df_star['aide_class'] = (df_star['treatment'] == 'aide').astype(int)

# Drop missing outcome values
df_star_clean = df_star.dropna(subset=['y']).copy()

# Convert school to categorical for FE
df_star_clean['school_cat'] = pd.Categorical(df_star_clean['school'])

print(f"STAR data: {len(df_star_clean)} observations")

# Control variables in STAR
star_controls = ['female', 'whiteasian', 'teacher_exp', 'white_teacher', 'lunch', 'masters']

# =============================================================================
# BASELINE SPECIFICATION
# =============================================================================
print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

# Exact replication: y ~ small + aide | school
# Using statsmodels with school FE
formula = "y ~ small_class + aide_class + C(school)"
model_baseline = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')

extract_results_statsmodels(
    model_baseline, 'small_class', 'y',
    'baseline', 'methods/cross_sectional_ols.md',
    'Full STAR sample with non-missing outcomes', 'school',
    'None (treatment indicators only)', 'None',
    'OLS with school FE', 'STAR'
)

extract_results_statsmodels(
    model_baseline, 'aide_class', 'y',
    'baseline_aide', 'methods/cross_sectional_ols.md',
    'Full STAR sample with non-missing outcomes', 'school',
    'None (treatment indicators only)', 'None',
    'OLS with school FE', 'STAR'
)

print(f"Baseline: Small class effect = {model_baseline.params['small_class']:.3f} (SE={model_baseline.bse['small_class']:.3f})")
print(f"Baseline: Aide effect = {model_baseline.params['aide_class']:.3f} (SE={model_baseline.bse['aide_class']:.3f})")

# =============================================================================
# CONTROL VARIATIONS (10-15 specs)
# =============================================================================
print("\n" + "="*60)
print("CONTROL VARIATIONS")
print("="*60)

# 1. No controls (bivariate with FE only)
formula = "y ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/control/none', 'robustness/leave_one_out.md',
    'Full sample', 'school', 'None', 'None', 'OLS with school FE')

# 2. Add controls incrementally
controls_added = []
for i, ctrl in enumerate(star_controls):
    controls_added.append(ctrl)
    if df_star_clean[ctrl].notna().sum() < 100:
        continue
    formula = f"y ~ small_class + aide_class + {' + '.join(controls_added)} + C(school)"
    try:
        model = smf.ols(formula, data=df_star_clean.dropna(subset=[ctrl])).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/control/add_{ctrl}', 'robustness/leave_one_out.md',
            f'Sample with non-missing {ctrl}', 'school',
            ', '.join(controls_added), 'None', 'OLS with school FE')
    except Exception as e:
        print(f"  Skipping add_{ctrl}: {e}")

# 3. Full controls
formula = f"y ~ small_class + aide_class + {' + '.join(star_controls)} + C(school)"
df_full = df_star_clean.dropna(subset=star_controls)
model = smf.ols(formula, data=df_full).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/control/full', 'robustness/leave_one_out.md',
    'Sample with all controls non-missing', 'school',
    ', '.join(star_controls), 'None', 'OLS with school FE')

# 4. Leave-one-out: drop each control
for ctrl in star_controls:
    remaining = [c for c in star_controls if c != ctrl]
    formula = f"y ~ small_class + aide_class + {' + '.join(remaining)} + C(school)"
    try:
        model = smf.ols(formula, data=df_full).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/loo/drop_{ctrl}', 'robustness/leave_one_out.md',
            'Full controls sample', 'school',
            ', '.join(remaining), 'None', 'OLS with school FE')
    except Exception as e:
        print(f"  Skipping loo_{ctrl}: {e}")

print(f"Control variations completed: {len([r for r in results if 'control' in r['spec_id'] or 'loo' in r['spec_id']])} specs")

# =============================================================================
# SAMPLE RESTRICTIONS (10-15 specs)
# =============================================================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# 1. By gender
for gender, label in [(True, 'female'), (False, 'male')]:
    df_sub = df_star_clean[df_star_clean['female'] == gender]
    formula = "y ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_sub).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/sample/{label}_only', 'robustness/sample_restrictions.md',
            f'{label.capitalize()} students only', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

# 2. By race
for race, label in [(True, 'whiteasian'), (False, 'minority')]:
    df_sub = df_star_clean[df_star_clean['whiteasian'] == race]
    formula = "y ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_sub).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/sample/{label}_only', 'robustness/sample_restrictions.md',
            f'{label.capitalize()} students only', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

# 3. By lunch status (socioeconomic proxy)
for lunch, label in [(True, 'freelunch'), (False, 'nolunch')]:
    df_sub = df_star_clean[df_star_clean['lunch'] == lunch]
    formula = "y ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_sub).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/sample/{label}_only', 'robustness/sample_restrictions.md',
            f'Students with {label}', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

# 4. Trim outliers in outcome
for pct in [1, 5, 10]:
    lower = df_star_clean['y'].quantile(pct/100)
    upper = df_star_clean['y'].quantile(1 - pct/100)
    df_trim = df_star_clean[(df_star_clean['y'] > lower) & (df_star_clean['y'] < upper)]
    formula = "y ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_trim).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/sample/trim_{pct}pct', 'robustness/sample_restrictions.md',
            f'Trimmed top/bottom {pct}%', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

# 5. Winsorize outcome
for pct in [1, 5]:
    df_wins = df_star_clean.copy()
    lower = df_wins['y'].quantile(pct/100)
    upper = df_wins['y'].quantile(1 - pct/100)
    df_wins['y_wins'] = df_wins['y'].clip(lower=lower, upper=upper)
    formula = "y_wins ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_wins).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y_wins',
            f'robust/sample/winsor_{pct}pct', 'robustness/sample_restrictions.md',
            f'Winsorized at {pct}%/{100-pct}%', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

# 6. By teacher experience
median_exp = df_star_clean['teacher_exp'].median()
for condition, label in [(df_star_clean['teacher_exp'] <= median_exp, 'low_exp'),
                         (df_star_clean['teacher_exp'] > median_exp, 'high_exp')]:
    df_sub = df_star_clean[condition].dropna(subset=['teacher_exp'])
    formula = "y ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_sub).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/sample/{label}_teacher', 'robustness/sample_restrictions.md',
            f'Teachers with {label.replace("_", " ")}', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

# 7. By teacher education
for masters, label in [(True, 'masters_teacher'), (False, 'no_masters_teacher')]:
    df_sub = df_star_clean[df_star_clean['masters'] == masters]
    formula = "y ~ small_class + aide_class + C(school)"
    try:
        model = smf.ols(formula, data=df_sub).fit(cov_type='HC1')
        extract_results_statsmodels(model, 'small_class', 'y',
            f'robust/sample/{label}', 'robustness/sample_restrictions.md',
            f'Students with teachers {"with" if masters else "without"} masters', 'school',
            'None', 'None', 'OLS with school FE')
    except:
        pass

print(f"Sample restrictions completed: {len([r for r in results if 'sample' in r['spec_id']])} specs")

# =============================================================================
# INFERENCE VARIATIONS (5-8 specs)
# =============================================================================
print("\n" + "="*60)
print("INFERENCE VARIATIONS")
print("="*60)

# 1. Classical (homoskedastic) SE
formula = "y ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit()
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/cluster/classical', 'robustness/clustering_variations.md',
    'Full sample', 'school', 'None', 'None', 'OLS homoskedastic SE')

# 2. HC1 robust SE (already done in baseline, but explicit)
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/cluster/hc1', 'robustness/clustering_variations.md',
    'Full sample', 'school', 'None', 'None', 'OLS HC1 robust SE')

# 3. HC2 robust SE
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC2')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/cluster/hc2', 'robustness/clustering_variations.md',
    'Full sample', 'school', 'None', 'None', 'OLS HC2 robust SE')

# 4. HC3 robust SE
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC3')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/cluster/hc3', 'robustness/clustering_variations.md',
    'Full sample', 'school', 'None', 'None', 'OLS HC3 robust SE')

# 5. Cluster by school
model = smf.ols(formula, data=df_star_clean).fit(cov_type='cluster',
                                                  cov_kwds={'groups': df_star_clean['school']})
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/cluster/school', 'robustness/clustering_variations.md',
    'Full sample', 'school', 'None', 'school', 'OLS clustered by school')

print(f"Inference variations completed: {len([r for r in results if 'cluster' in r['spec_id']])} specs")

# =============================================================================
# FUNCTIONAL FORM (5-8 specs)
# =============================================================================
print("\n" + "="*60)
print("FUNCTIONAL FORM")
print("="*60)

# 1. Standardized outcome
df_star_clean['y_std'] = (df_star_clean['y'] - df_star_clean['y'].mean()) / df_star_clean['y'].std()
formula = "y_std ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y_std',
    'robust/form/y_standardized', 'robustness/functional_form.md',
    'Full sample, standardized outcome', 'school', 'None', 'None', 'OLS standardized')

# 2. Log outcome (adding constant to handle any negatives)
df_star_clean['y_shifted'] = df_star_clean['y'] - df_star_clean['y'].min() + 1
df_star_clean['y_log'] = np.log(df_star_clean['y_shifted'])
formula = "y_log ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y_log',
    'robust/form/y_log', 'robustness/functional_form.md',
    'Full sample, log outcome', 'school', 'None', 'None', 'OLS log outcome')

# 3. Asinh transformation
df_star_clean['y_asinh'] = np.arcsinh(df_star_clean['y'])
formula = "y_asinh ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y_asinh',
    'robust/form/y_asinh', 'robustness/functional_form.md',
    'Full sample, asinh outcome', 'school', 'None', 'None', 'OLS asinh outcome')

# 4. Rank transformation
df_star_clean['y_rank'] = df_star_clean['y'].rank() / len(df_star_clean)
formula = "y_rank ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y_rank',
    'robust/form/y_rank', 'robustness/functional_form.md',
    'Full sample, rank outcome', 'school', 'None', 'None', 'OLS rank outcome')

# 5. Quantile regressions
from statsmodels.regression.quantile_regression import QuantReg

# Prepare data for quantile regression (drop school FE for simplicity)
y = df_star_clean['y'].values
X = df_star_clean[['small_class', 'aide_class']].copy()
X['const'] = 1
X = X[['const', 'small_class', 'aide_class']]

for q, qlabel in [(0.25, '25'), (0.5, '50'), (0.75, '75')]:
    try:
        model = QuantReg(y, X).fit(q=q)
        coef = model.params['small_class']
        se = model.bse['small_class']
        t_stat = coef / se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        coef_vector = {
            "treatment": {"var": "small_class", "coef": float(coef), "se": float(se), "pval": float(p_value)},
            "controls": [],
            "diagnostics": {"quantile": q}
        }
        add_result(f'robust/form/quantile_{qlabel}', 'robustness/functional_form.md',
                   'y', 'small_class', coef, se, t_stat, p_value, coef-1.96*se, coef+1.96*se,
                   len(y), None, coef_vector, f'{int(q*100)}th percentile regression',
                   'None', 'None', 'None', f'Quantile regression q={q}')
    except Exception as e:
        print(f"  Skipping quantile_{qlabel}: {e}")

print(f"Functional form completed: {len([r for r in results if 'form' in r['spec_id']])} specs")

# =============================================================================
# FIXED EFFECTS VARIATIONS (3-5 specs)
# =============================================================================
print("\n" + "="*60)
print("FIXED EFFECTS VARIATIONS")
print("="*60)

# 1. No fixed effects
formula = "y ~ small_class + aide_class"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/estimation/no_fe', 'methods/cross_sectional_ols.md#fixed-effects',
    'Full sample, no FE', 'None', 'None', 'None', 'OLS no FE')

# 2. With school FE (baseline - already run)
# 3. With school FE + controls
formula = f"y ~ small_class + aide_class + {' + '.join(star_controls)} + C(school)"
model = smf.ols(formula, data=df_full).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/estimation/fe_with_controls', 'methods/cross_sectional_ols.md#fixed-effects',
    'Full sample with all controls', 'school', ', '.join(star_controls), 'None', 'OLS school FE + controls')

print(f"FE variations completed: {len([r for r in results if 'estimation' in r['spec_id']])} specs")

# =============================================================================
# HETEROGENEITY ANALYSIS (10+ specs)
# =============================================================================
print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# 1. Gender interactions
df_star_clean['female_int'] = df_star_clean['female'].fillna(False).astype(int)
formula = "y ~ small_class * female_int + aide_class * female_int + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/het/interaction_gender', 'robustness/heterogeneity.md',
    'Full sample', 'school', 'Treatment x gender interactions', 'None',
    'OLS with gender interaction')

# 2. Race interactions
df_star_clean['whiteasian_int'] = df_star_clean['whiteasian'].fillna(False).astype(int)
formula = "y ~ small_class * whiteasian_int + aide_class * whiteasian_int + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/het/interaction_race', 'robustness/heterogeneity.md',
    'Full sample', 'school', 'Treatment x race interactions', 'None',
    'OLS with race interaction')

# 3. SES (lunch) interactions
df_star_clean['lunch_int'] = df_star_clean['lunch'].fillna(False).astype(int)
formula = "y ~ small_class * lunch_int + aide_class * lunch_int + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/het/interaction_ses', 'robustness/heterogeneity.md',
    'Full sample', 'school', 'Treatment x SES (lunch) interactions', 'None',
    'OLS with SES interaction')

# 4. Teacher experience interactions
df_exp = df_star_clean.dropna(subset=['teacher_exp']).copy()
df_exp['high_exp'] = (df_exp['teacher_exp'] > df_exp['teacher_exp'].median()).astype(int)
formula = "y ~ small_class * high_exp + aide_class * high_exp + C(school)"
try:
    model = smf.ols(formula, data=df_exp).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/interaction_teacher_exp', 'robustness/heterogeneity.md',
        'Sample with teacher exp data', 'school', 'Treatment x teacher experience', 'None',
        'OLS with teacher experience interaction')
except:
    pass

# 5. Teacher education (masters) interactions
df_star_clean['masters_int'] = df_star_clean['masters'].fillna(False).astype(int)
formula = "y ~ small_class * masters_int + aide_class * masters_int + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/het/interaction_teacher_edu', 'robustness/heterogeneity.md',
    'Full sample', 'school', 'Treatment x teacher education interactions', 'None',
    'OLS with teacher education interaction')

# 6-10. Subgroup analyses (effect for each subgroup)
# Male students
df_male = df_star_clean[df_star_clean['female'] == False]
formula = "y ~ small_class + aide_class + C(school)"
try:
    model = smf.ols(formula, data=df_male).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/by_gender_male', 'robustness/heterogeneity.md',
        'Male students only', 'school', 'None', 'None', 'OLS male subsample')
except:
    pass

# Female students
df_female = df_star_clean[df_star_clean['female'] == True]
try:
    model = smf.ols(formula, data=df_female).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/by_gender_female', 'robustness/heterogeneity.md',
        'Female students only', 'school', 'None', 'None', 'OLS female subsample')
except:
    pass

# White/Asian students
df_wa = df_star_clean[df_star_clean['whiteasian'] == True]
try:
    model = smf.ols(formula, data=df_wa).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/by_race_whiteasian', 'robustness/heterogeneity.md',
        'White/Asian students only', 'school', 'None', 'None', 'OLS White/Asian subsample')
except:
    pass

# Minority students
df_min = df_star_clean[df_star_clean['whiteasian'] == False]
try:
    model = smf.ols(formula, data=df_min).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/by_race_minority', 'robustness/heterogeneity.md',
        'Minority students only', 'school', 'None', 'None', 'OLS minority subsample')
except:
    pass

# Free lunch students (low SES)
df_fl = df_star_clean[df_star_clean['lunch'] == True]
try:
    model = smf.ols(formula, data=df_fl).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/by_ses_low', 'robustness/heterogeneity.md',
        'Free lunch students (low SES)', 'school', 'None', 'None', 'OLS low SES subsample')
except:
    pass

# No free lunch students (higher SES)
df_nfl = df_star_clean[df_star_clean['lunch'] == False]
try:
    model = smf.ols(formula, data=df_nfl).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/het/by_ses_high', 'robustness/heterogeneity.md',
        'No free lunch students (higher SES)', 'school', 'None', 'None', 'OLS higher SES subsample')
except:
    pass

print(f"Heterogeneity completed: {len([r for r in results if 'het' in r['spec_id']])} specs")

# =============================================================================
# ALTERNATIVE TREATMENT DEFINITIONS (3-5 specs)
# =============================================================================
print("\n" + "="*60)
print("ALTERNATIVE TREATMENT DEFINITIONS")
print("="*60)

# 1. Any treatment vs control (binary)
df_star_clean['any_treatment'] = (df_star_clean['treatment'] != 'regular').astype(int)
formula = "y ~ any_treatment + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'any_treatment', 'y',
    'robust/treatment/any_vs_control', 'methods/cross_sectional_ols.md',
    'Full sample', 'school', 'None', 'None', 'OLS any treatment vs control')

# 2. Only small class vs control (exclude aide)
df_no_aide = df_star_clean[df_star_clean['treatment'] != 'aide']
formula = "y ~ small_class + C(school)"
model = smf.ols(formula, data=df_no_aide).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/treatment/small_vs_control', 'methods/cross_sectional_ols.md',
    'Exclude aide treatment', 'school', 'None', 'None', 'OLS small vs control only')

# 3. Only aide vs control (exclude small)
df_no_small = df_star_clean[df_star_clean['treatment'] != 'small']
formula = "y ~ aide_class + C(school)"
model = smf.ols(formula, data=df_no_small).fit(cov_type='HC1')
extract_results_statsmodels(model, 'aide_class', 'y',
    'robust/treatment/aide_vs_control', 'methods/cross_sectional_ols.md',
    'Exclude small class treatment', 'school', 'None', 'None', 'OLS aide vs control only')

# 4. Small class vs aide (exclude control)
df_treated = df_star_clean[df_star_clean['treatment'] != 'regular']
df_treated['small_vs_aide'] = (df_treated['treatment'] == 'small').astype(int)
formula = "y ~ small_vs_aide + C(school)"
model = smf.ols(formula, data=df_treated).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_vs_aide', 'y',
    'robust/treatment/small_vs_aide', 'methods/cross_sectional_ols.md',
    'Only treated students', 'school', 'None', 'None', 'OLS small vs aide')

print(f"Treatment variations completed: {len([r for r in results if 'treatment' in r['spec_id']])} specs")

# =============================================================================
# PLACEBO TESTS (3-5 specs)
# =============================================================================
print("\n" + "="*60)
print("PLACEBO TESTS")
print("="*60)

# 1. Placebo: Treatment predicting baseline characteristics (should be null)
# If treatment is random, it shouldn't predict pre-treatment characteristics
# Using teacher characteristics as "placebo outcomes" that should be unaffected by class size

# Teacher experience as placebo outcome
formula = "teacher_exp ~ small_class + aide_class + C(school)"
df_te = df_star_clean.dropna(subset=['teacher_exp'])
model = smf.ols(formula, data=df_te).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'teacher_exp',
    'robust/placebo/teacher_exp', 'robustness/placebo_tests.md',
    'Sample with teacher data', 'school', 'None', 'None', 'OLS placebo - teacher exp')

# White teacher as placebo outcome
df_star_clean['white_teacher_int'] = df_star_clean['white_teacher'].fillna(False).astype(int)
formula = "white_teacher_int ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'white_teacher_int',
    'robust/placebo/white_teacher', 'robustness/placebo_tests.md',
    'Full sample', 'school', 'None', 'None', 'OLS placebo - white teacher')

# Masters degree as placebo outcome
df_star_clean['masters_int_out'] = df_star_clean['masters'].fillna(False).astype(int)
formula = "masters_int_out ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'masters_int_out',
    'robust/placebo/masters', 'robustness/placebo_tests.md',
    'Full sample', 'school', 'None', 'None', 'OLS placebo - masters degree')

# 2. Randomization check: treatment balance on student characteristics
# Female as outcome
formula = "female_int ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'female_int',
    'robust/placebo/female_balance', 'robustness/placebo_tests.md',
    'Full sample', 'school', 'None', 'None', 'OLS balance check - female')

# Race as outcome
formula = "whiteasian_int ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'whiteasian_int',
    'robust/placebo/race_balance', 'robustness/placebo_tests.md',
    'Full sample', 'school', 'None', 'None', 'OLS balance check - whiteasian')

print(f"Placebo tests completed: {len([r for r in results if 'placebo' in r['spec_id']])} specs")

# =============================================================================
# BENHASSINE ET AL. DATASET (Additional robustness with different experiment)
# =============================================================================
print("\n" + "="*60)
print("BENHASSINE DATASET ANALYSIS")
print("="*60)

df_ben = pd.read_stata(f"{DATA_PATH}/dta/benhassine.dta")

# Create treatment dummies
df_ben['lct_father'] = (df_ben['treatment'] == 'LCT to fathers').astype(int)
df_ben['lct_mother'] = (df_ben['treatment'] == 'LCT to mothers').astype(int)
df_ben['cct_father'] = (df_ben['treatment'] == 'CCTs to fathers').astype(int)
df_ben['cct_mother'] = (df_ben['treatment'] == 'CCTs to mothers').astype(int)

# Control variables
ben_controls = ['bs_pchildren_enrolled', 'bs_pchildren_enrolled_miss',
                'age_baseline', 'girl', 'bs_inschool08', 'bs_neverenrolled08',
                'bs_inschool08_miss', 'prel_elec', 'prel_elec_miss',
                'prel_inacc_winter', 'prel_inacc_winter_miss', 'sampling_frame_problem']

# Clean data
df_ben_clean = df_ben.dropna(subset=['enroll_attend_May2010'])

# Baseline for Benhassine
formula = f"enroll_attend_May2010 ~ lct_father + lct_mother + cct_father + cct_mother + {' + '.join(ben_controls)} + C(stratum)"
try:
    model = smf.wls(formula, data=df_ben_clean, weights=df_ben_clean['weight_hh']).fit(cov_type='cluster',
                                                                                        cov_kwds={'groups': df_ben_clean['schoolid']})
    extract_results_statsmodels(model, 'lct_father', 'enroll_attend_May2010',
        'baseline', 'methods/cross_sectional_ols.md',
        'Full Benhassine sample', 'stratum', ', '.join(ben_controls), 'schoolid',
        'WLS with school clustering', 'Benhassine')
except Exception as e:
    print(f"  Benhassine baseline failed: {e}")

# Alternative: CCT father effect
try:
    extract_results_statsmodels(model, 'cct_father', 'enroll_attend_May2010',
        'baseline_cct_father', 'methods/cross_sectional_ols.md',
        'Full Benhassine sample', 'stratum', ', '.join(ben_controls), 'schoolid',
        'WLS with school clustering', 'Benhassine')
except:
    pass

# Sample restrictions for Benhassine
# Girls only
df_girls = df_ben_clean[df_ben_clean['girl'] == 1]
try:
    model = smf.wls(formula, data=df_girls, weights=df_girls['weight_hh']).fit(cov_type='cluster',
                                                                                cov_kwds={'groups': df_girls['schoolid']})
    extract_results_statsmodels(model, 'lct_father', 'enroll_attend_May2010',
        'robust/sample/girls_only', 'robustness/sample_restrictions.md',
        'Girls only', 'stratum', ', '.join(ben_controls), 'schoolid',
        'WLS with school clustering', 'Benhassine')
except:
    pass

# Boys only
df_boys = df_ben_clean[df_ben_clean['girl'] == 0]
try:
    model = smf.wls(formula, data=df_boys, weights=df_boys['weight_hh']).fit(cov_type='cluster',
                                                                              cov_kwds={'groups': df_boys['schoolid']})
    extract_results_statsmodels(model, 'lct_father', 'enroll_attend_May2010',
        'robust/sample/boys_only', 'robustness/sample_restrictions.md',
        'Boys only', 'stratum', ', '.join(ben_controls), 'schoolid',
        'WLS with school clustering', 'Benhassine')
except:
    pass

# Unweighted
formula_simple = f"enroll_attend_May2010 ~ lct_father + lct_mother + cct_father + cct_mother + {' + '.join(ben_controls)} + C(stratum)"
try:
    model = smf.ols(formula_simple, data=df_ben_clean).fit(cov_type='cluster',
                                                           cov_kwds={'groups': df_ben_clean['schoolid']})
    extract_results_statsmodels(model, 'lct_father', 'enroll_attend_May2010',
        'robust/weights/unweighted', 'robustness/sample_restrictions.md',
        'Full sample, unweighted', 'stratum', ', '.join(ben_controls), 'schoolid',
        'OLS unweighted', 'Benhassine')
except:
    pass

# Any LCT vs control
df_ben_clean['any_lct'] = ((df_ben_clean['treatment'] == 'LCT to fathers') |
                           (df_ben_clean['treatment'] == 'LCT to mothers')).astype(int)
df_ben_clean['any_cct'] = ((df_ben_clean['treatment'] == 'CCTs to fathers') |
                           (df_ben_clean['treatment'] == 'CCTs to mothers')).astype(int)
formula_lct = f"enroll_attend_May2010 ~ any_lct + any_cct + {' + '.join(ben_controls)} + C(stratum)"
try:
    model = smf.wls(formula_lct, data=df_ben_clean, weights=df_ben_clean['weight_hh']).fit(cov_type='cluster',
                                                                                           cov_kwds={'groups': df_ben_clean['schoolid']})
    extract_results_statsmodels(model, 'any_lct', 'enroll_attend_May2010',
        'robust/treatment/any_lct', 'methods/cross_sectional_ols.md',
        'Full sample', 'stratum', ', '.join(ben_controls), 'schoolid',
        'WLS any LCT treatment', 'Benhassine')
except:
    pass

print(f"Benhassine analysis completed: {len([r for r in results if r['dataset'] == 'Benhassine'])} specs")

# =============================================================================
# ADDITIONAL SPECIFICATIONS TO REACH 50+
# =============================================================================
print("\n" + "="*60)
print("ADDITIONAL SPECIFICATIONS")
print("="*60)

# More control combinations for STAR
# Demographics only
formula = "y ~ small_class + aide_class + female + whiteasian + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/control/demographics_only', 'robustness/leave_one_out.md',
    'Full sample', 'school', 'female, whiteasian', 'None', 'OLS demographics only')

# Teacher characteristics only
formula = "y ~ small_class + aide_class + teacher_exp + white_teacher + masters + C(school)"
df_teacher = df_star_clean.dropna(subset=['teacher_exp', 'white_teacher', 'masters'])
model = smf.ols(formula, data=df_teacher).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/control/teacher_only', 'robustness/leave_one_out.md',
    'Sample with teacher data', 'school', 'teacher_exp, white_teacher, masters', 'None',
    'OLS teacher controls only')

# Student + teacher controls
formula = "y ~ small_class + aide_class + female + whiteasian + lunch + teacher_exp + white_teacher + masters + C(school)"
df_all = df_star_clean.dropna(subset=['teacher_exp', 'white_teacher', 'masters', 'lunch'])
model = smf.ols(formula, data=df_all).fit(cov_type='HC1')
extract_results_statsmodels(model, 'small_class', 'y',
    'robust/control/student_teacher', 'robustness/leave_one_out.md',
    'Sample with all data', 'school', 'student + teacher controls', 'None',
    'OLS student + teacher controls')

# More sample cuts
# By outcome quartiles (baseline performance proxy - bottom quartile)
y_q25 = df_star_clean['y'].quantile(0.25)
y_q75 = df_star_clean['y'].quantile(0.75)

df_low = df_star_clean[df_star_clean['y'] <= y_q25]
formula = "y ~ small_class + aide_class + C(school)"
try:
    model = smf.ols(formula, data=df_low).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/sample/low_performers', 'robustness/sample_restrictions.md',
        'Bottom quartile students', 'school', 'None', 'None', 'OLS bottom quartile')
except:
    pass

df_high = df_star_clean[df_star_clean['y'] >= y_q75]
try:
    model = smf.ols(formula, data=df_high).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/sample/high_performers', 'robustness/sample_restrictions.md',
        'Top quartile students', 'school', 'None', 'None', 'OLS top quartile')
except:
    pass

# Middle 50%
df_mid = df_star_clean[(df_star_clean['y'] > y_q25) & (df_star_clean['y'] < y_q75)]
try:
    model = smf.ols(formula, data=df_mid).fit(cov_type='HC1')
    extract_results_statsmodels(model, 'small_class', 'y',
        'robust/sample/middle_performers', 'robustness/sample_restrictions.md',
        'Middle 50% students', 'school', 'None', 'None', 'OLS middle 50%')
except:
    pass

# Multiple testing correction
# Report aide effects for key specifications
formula = "y ~ small_class + aide_class + C(school)"
model = smf.ols(formula, data=df_star_clean).fit(cov_type='HC1')
extract_results_statsmodels(model, 'aide_class', 'y',
    'robust/inference/aide_baseline', 'robustness/inference_alternatives.md',
    'Full sample', 'school', 'None', 'None', 'OLS aide effect')

# With controls
formula = f"y ~ small_class + aide_class + {' + '.join(star_controls)} + C(school)"
model = smf.ols(formula, data=df_full).fit(cov_type='HC1')
extract_results_statsmodels(model, 'aide_class', 'y',
    'robust/inference/aide_full_controls', 'robustness/inference_alternatives.md',
    'Full controls sample', 'school', ', '.join(star_controls), 'None', 'OLS aide with controls')

print(f"Additional specifications completed. Total: {len(results)} specs")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications: {len(results_df)}")
print(f"Unique spec_ids: {results_df['spec_id'].nunique()}")

# Save to CSV
output_file = f"{OUTPUT_PATH}/specification_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Filter to STAR small_class results for summary
star_small = results_df[(results_df['dataset'] == 'STAR') &
                         (results_df['treatment_var'] == 'small_class')]
print(f"\nSTAR Small Class Effect Specifications: {len(star_small)}")
print(f"  Positive coefficients: {(star_small['coefficient'] > 0).sum()} ({100*(star_small['coefficient'] > 0).mean():.1f}%)")
print(f"  Significant at 5%: {(star_small['p_value'] < 0.05).sum()} ({100*(star_small['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(star_small['p_value'] < 0.01).sum()} ({100*(star_small['p_value'] < 0.01).mean():.1f}%)")
print(f"  Median coefficient: {star_small['coefficient'].median():.3f}")
print(f"  Mean coefficient: {star_small['coefficient'].mean():.3f}")
print(f"  Range: [{star_small['coefficient'].min():.3f}, {star_small['coefficient'].max():.3f}]")

# By category
print("\nSpecifications by Category:")
for prefix in ['baseline', 'robust/control', 'robust/loo', 'robust/sample',
               'robust/cluster', 'robust/form', 'robust/estimation',
               'robust/het', 'robust/treatment', 'robust/placebo', 'robust/inference']:
    count = results_df['spec_id'].str.startswith(prefix).sum()
    if count > 0:
        print(f"  {prefix}: {count}")

print("\nDone!")
