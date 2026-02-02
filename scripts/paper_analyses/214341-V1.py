"""
Specification Search: 214341-V1
Paper: "Who Benefits from the Online Gig Economy?" by Stanton and Thomas (AER)

This paper studies worker surplus on an online labor platform. The main analysis
uses survey data to estimate markups (surplus) relative to workers' willingness
to accept (WTA), ex-post WTA, and outside wages.

Method Classification: Cross-sectional OLS / Survey Analysis
- Primary outcomes: Worker surplus/markup measures
- The paper focuses on means and distributions of surplus measures
- Key analyses include balance tests and weighted/unweighted comparisons
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import statistical packages
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats
except ImportError as e:
    print(f"Warning: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "214341-V1"
JOURNAL = "AER"
PAPER_TITLE = "Who Benefits from the Online Gig Economy?"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}/StantonThomas_AER_Replication"
DATA_PATH = f"{PACKAGE_PATH}/PublicData/SurveyData_PublicArchive_DeIdentified.dta"

# Method classification
METHOD_CODE = "cross_sectional_ols"
METHOD_TREE_PATH = "specification_tree/methods/cross_sectional_ols.md"

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Raw data shape: {df.shape}")

# Drop observations with zero jobs (as in original code)
df = df[df['numjobs'] > 0].copy()
print(f"After dropping numjobs=0: {df.shape}")

# Create key variables
df['Ones'] = 1
df['log_numjobs'] = np.log(df['numjobs'])
df['log_profile_rate'] = np.log(df['scraped_profile_rate'].replace(0, np.nan))

# Identify survey respondents
df_survey = df[df['InSurvey'] == 1].copy()
print(f"Survey respondents: {len(df_survey)}")

# ============================================================================
# OUTCOME VARIABLES (Main surplus measures from the paper)
# ============================================================================

OUTCOME_VARS = {
    'hrly_surp_rel_min': 'Markup Relative to WTA',
    'hrly_surp_rel_expost': 'Markup Relative to Ex-Post WTA',
    'hrly_surp_rel_outside': 'Markup Relative to Outside Wage',
    'fixed_surp_rel_expost': 'Markup on Fixed-Price Contracts'
}

# Primary treatment for balance tests
TREATMENT_VAR = 'InSurvey'

# Control variables available
CONTROLS = ['numjobs', 'scraped_profile_rate', 'c_US']

# ============================================================================
# RESULTS STORAGE
# ============================================================================

results = []

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coef, se, t_stat, p_value, ci_lower, ci_upper, n_obs,
               r_squared=None, coef_vector=None, sample_desc="",
               fixed_effects="None", controls_desc="", cluster_var="None",
               model_type="OLS"):
    """Add a result to the results list."""
    # Convert coef_vector to serializable format
    if coef_vector:
        coef_vector = convert_to_serializable(coef_vector)

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(coef) if coef is not None and not pd.isna(coef) else None,
        'std_error': float(se) if se is not None and not pd.isna(se) else None,
        't_stat': float(t_stat) if t_stat is not None and not pd.isna(t_stat) else None,
        'p_value': float(p_value) if p_value is not None and not pd.isna(p_value) else None,
        'ci_lower': float(ci_lower) if ci_lower is not None and not pd.isna(ci_lower) else None,
        'ci_upper': float(ci_upper) if ci_upper is not None and not pd.isna(ci_upper) else None,
        'n_obs': int(n_obs) if n_obs is not None else None,
        'r_squared': float(r_squared) if r_squared is not None and not pd.isna(r_squared) else None,
        'coefficient_vector_json': json.dumps(coef_vector) if coef_vector else "",
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    })

# ============================================================================
# BASELINE SPECIFICATIONS
# Main surplus estimates from Table 7
# ============================================================================

print("\n=== RUNNING BASELINE SPECIFICATIONS ===")

for outcome, outcome_label in OUTCOME_VARS.items():
    # Skip if no valid data
    valid_data = df_survey[df_survey[outcome].notna()]
    if len(valid_data) < 5:
        continue

    # --- Baseline 1: Unweighted mean ---
    y = valid_data[outcome].dropna()
    n = len(y)
    mean_val = y.mean()
    se = y.std() / np.sqrt(n)
    t_stat = mean_val / se if se > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
    ci_lower = mean_val - 1.96 * se
    ci_upper = mean_val + 1.96 * se

    coef_vector = {
        'treatment': {'var': outcome, 'coef': mean_val, 'se': se, 'pval': p_val},
        'diagnostics': {'mean': mean_val, 'std': y.std(), 'median': y.median()}
    }

    add_result(
        spec_id='baseline',
        spec_tree_path=f'{METHOD_TREE_PATH}#baseline',
        outcome_var=outcome,
        treatment_var='mean_surplus',
        coef=mean_val,
        se=se,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=n,
        r_squared=None,
        coef_vector=coef_vector,
        sample_desc=f'Survey respondents, unweighted, {outcome_label}',
        model_type='Mean'
    )
    print(f"  Baseline ({outcome}): mean={mean_val:.4f}, se={se:.4f}, n={n}")

    # --- Baseline 2: Weighted mean ---
    valid_weighted = valid_data[valid_data['weight'].notna() & (valid_data['weight'] > 0)]
    if len(valid_weighted) > 5:
        weights = valid_weighted['weight']
        y_w = valid_weighted[outcome]
        weighted_mean = np.average(y_w, weights=weights)
        # Weighted SE (approximate)
        weighted_var = np.average((y_w - weighted_mean)**2, weights=weights)
        weighted_se = np.sqrt(weighted_var / len(y_w))

        t_stat_w = weighted_mean / weighted_se if weighted_se > 0 else np.nan
        p_val_w = 2 * (1 - stats.t.cdf(abs(t_stat_w), df=len(y_w)-1)) if not np.isnan(t_stat_w) else np.nan
        ci_lower_w = weighted_mean - 1.96 * weighted_se
        ci_upper_w = weighted_mean + 1.96 * weighted_se

        coef_vector_w = {
            'treatment': {'var': outcome, 'coef': weighted_mean, 'se': weighted_se, 'pval': p_val_w},
            'diagnostics': {'weighted_mean': weighted_mean}
        }

        add_result(
            spec_id='baseline_weighted',
            spec_tree_path=f'{METHOD_TREE_PATH}#baseline',
            outcome_var=outcome,
            treatment_var='mean_surplus',
            coef=weighted_mean,
            se=weighted_se,
            t_stat=t_stat_w,
            p_value=p_val_w,
            ci_lower=ci_lower_w,
            ci_upper=ci_upper_w,
            n_obs=len(valid_weighted),
            r_squared=None,
            coef_vector=coef_vector_w,
            sample_desc=f'Survey respondents, IPW weighted, {outcome_label}',
            model_type='Weighted Mean'
        )
        print(f"  Baseline weighted ({outcome}): mean={weighted_mean:.4f}, se={weighted_se:.4f}")

# ============================================================================
# BALANCE TESTS (Table A1)
# Testing survey participation selection
# ============================================================================

print("\n=== RUNNING BALANCE TESTS ===")

balance_vars = ['numjobs', 'scraped_profile_rate', 'c_US']
df_full = df[df['numjobs'] > 0].copy()

for var in balance_vars:
    valid = df_full[df_full[var].notna()].copy()
    if len(valid) < 10:
        continue

    try:
        # Run balance test regression: var ~ InSurvey
        X = sm.add_constant(valid['InSurvey'])
        y = valid[var]
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

        coef = model.params['InSurvey']
        se = model.bse['InSurvey']
        t_stat = model.tvalues['InSurvey']
        p_val = model.pvalues['InSurvey']
        ci = model.conf_int().loc['InSurvey']

        coef_vector = {
            'treatment': {'var': 'InSurvey', 'coef': coef, 'se': se, 'pval': p_val},
            'constant': {'var': 'const', 'coef': model.params['const'],
                        'se': model.bse['const'], 'pval': model.pvalues['const']},
            'diagnostics': {'r2': model.rsquared, 'f_stat': model.fvalue}
        }

        add_result(
            spec_id=f'balance_test/{var}',
            spec_tree_path='robustness/sample_restrictions.md#balance',
            outcome_var=var,
            treatment_var='InSurvey',
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector=coef_vector,
            sample_desc='Full sample balance test',
            model_type='OLS'
        )
        print(f"  Balance test ({var}): coef={coef:.4f}, se={se:.4f}, p={p_val:.4f}")
    except Exception as e:
        print(f"  Balance test ({var}) failed: {e}")

# ============================================================================
# CONTROL VARIATIONS
# ============================================================================

print("\n=== RUNNING CONTROL VARIATIONS ===")

# Use hrly_surp_rel_min as primary outcome for control variations
primary_outcome = 'hrly_surp_rel_min'
df_analysis = df_survey[df_survey[primary_outcome].notna()].copy()

# Create additional controls
df_analysis['log_numjobs'] = np.log(df_analysis['numjobs'])
df_analysis['log_profile_rate'] = np.log(df_analysis['scraped_profile_rate'].replace(0, np.nan))

# Bivariate - just the mean (already done in baseline)

# Add progressively more controls
control_sets = [
    ('no_controls', []),
    ('add_log_numjobs', ['log_numjobs']),
    ('add_log_rate', ['log_numjobs', 'log_profile_rate']),
    ('add_c_US', ['log_numjobs', 'log_profile_rate', 'c_US']),
]

for control_name, control_list in control_sets:
    if len(control_list) == 0:
        continue

    # Filter to non-missing for all controls
    valid = df_analysis.dropna(subset=[primary_outcome] + control_list)
    if len(valid) < 10:
        continue

    try:
        # Regress surplus on controls to understand predictors
        formula = f'{primary_outcome} ~ ' + ' + '.join(control_list)
        model = smf.ols(formula, data=valid).fit(cov_type='HC1')

        # Get first control coefficient as main result
        main_control = control_list[0]
        coef = model.params[main_control]
        se = model.bse[main_control]
        t_stat = model.tvalues[main_control]
        p_val = model.pvalues[main_control]
        ci = model.conf_int().loc[main_control]

        coef_vector = {
            'treatment': {'var': main_control, 'coef': coef, 'se': se, 'pval': p_val},
            'controls': [{'var': c, 'coef': model.params[c], 'se': model.bse[c],
                         'pval': model.pvalues[c]} for c in control_list if c != main_control],
            'diagnostics': {'r2': model.rsquared, 'f_stat': model.fvalue}
        }

        add_result(
            spec_id=f'robust/control/{control_name}',
            spec_tree_path='robustness/control_progression.md',
            outcome_var=primary_outcome,
            treatment_var=main_control,
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector=coef_vector,
            sample_desc=f'Survey respondents, controls: {", ".join(control_list)}',
            controls_desc=', '.join(control_list),
            model_type='OLS'
        )
        print(f"  Control variation ({control_name}): R2={model.rsquared:.4f}, n={model.nobs}")
    except Exception as e:
        print(f"  Control variation ({control_name}) failed: {e}")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("\n=== RUNNING SAMPLE RESTRICTIONS ===")

# Restrictions to test
sample_restrictions = [
    ('us_only', df_survey['c_US'] == 1, 'US workers only'),
    ('non_us', df_survey['c_US'] == 0, 'Non-US workers'),
    ('high_experience', df_survey['numjobs'] >= df_survey['numjobs'].median(), 'High experience (>= median jobs)'),
    ('low_experience', df_survey['numjobs'] < df_survey['numjobs'].median(), 'Low experience (< median jobs)'),
    ('high_rate', df_survey['scraped_profile_rate'] >= df_survey['scraped_profile_rate'].median(), 'High hourly rate'),
    ('low_rate', df_survey['scraped_profile_rate'] < df_survey['scraped_profile_rate'].median(), 'Low hourly rate'),
]

# Add tercile splits for numjobs
try:
    terciles = pd.qcut(df_survey['numjobs'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
    sample_restrictions.extend([
        ('numjobs_tercile_low', terciles == 'low', 'Low experience tercile'),
        ('numjobs_tercile_mid', terciles == 'mid', 'Mid experience tercile'),
        ('numjobs_tercile_high', terciles == 'high', 'High experience tercile'),
    ])
except:
    pass

# Add tercile splits for profile rate
try:
    rate_terciles = pd.qcut(df_survey['scraped_profile_rate'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
    sample_restrictions.extend([
        ('rate_tercile_low', rate_terciles == 'low', 'Low rate tercile'),
        ('rate_tercile_mid', rate_terciles == 'mid', 'Mid rate tercile'),
        ('rate_tercile_high', rate_terciles == 'high', 'High rate tercile'),
    ])
except:
    pass

for outcome, outcome_label in OUTCOME_VARS.items():
    for restrict_name, condition, restrict_desc in sample_restrictions:
        try:
            subset = df_survey[condition & df_survey[outcome].notna()]
            if len(subset) < 5:
                continue

            y = subset[outcome].dropna()
            n = len(y)
            mean_val = y.mean()
            se = y.std() / np.sqrt(n)
            t_stat = mean_val / se if se > 0 else np.nan
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
            ci_lower = mean_val - 1.96 * se
            ci_upper = mean_val + 1.96 * se

            coef_vector = {
                'treatment': {'var': outcome, 'coef': mean_val, 'se': se, 'pval': p_val},
                'diagnostics': {'mean': mean_val, 'std': y.std(), 'n': n}
            }

            add_result(
                spec_id=f'robust/sample/{restrict_name}_{outcome}',
                spec_tree_path='robustness/sample_restrictions.md',
                outcome_var=outcome,
                treatment_var='mean_surplus',
                coef=mean_val,
                se=se,
                t_stat=t_stat,
                p_value=p_val,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_obs=n,
                r_squared=None,
                coef_vector=coef_vector,
                sample_desc=f'{restrict_desc}, {outcome_label}',
                model_type='Mean'
            )
        except Exception as e:
            print(f"  Sample restriction ({restrict_name}, {outcome}) failed: {e}")

print(f"  Completed sample restrictions")

# ============================================================================
# OUTLIER HANDLING
# ============================================================================

print("\n=== RUNNING OUTLIER HANDLING ===")

for outcome, outcome_label in OUTCOME_VARS.items():
    valid_data = df_survey[df_survey[outcome].notna()].copy()
    if len(valid_data) < 10:
        continue

    y_full = valid_data[outcome]

    # Winsorize at different levels
    for pct in [1, 5, 10]:
        lower = y_full.quantile(pct/100)
        upper = y_full.quantile(1 - pct/100)
        y_wins = y_full.clip(lower=lower, upper=upper)

        n = len(y_wins)
        mean_val = y_wins.mean()
        se = y_wins.std() / np.sqrt(n)
        t_stat = mean_val / se if se > 0 else np.nan
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

        add_result(
            spec_id=f'robust/sample/winsorize_{pct}pct_{outcome}',
            spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
            outcome_var=outcome,
            treatment_var='mean_surplus',
            coef=mean_val,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=n,
            coef_vector={'treatment': {'var': outcome, 'coef': mean_val, 'se': se, 'pval': p_val}},
            sample_desc=f'Winsorized at {pct}%, {outcome_label}',
            model_type='Mean (winsorized)'
        )

    # Trim at different levels
    for pct in [1, 5, 10]:
        lower = y_full.quantile(pct/100)
        upper = y_full.quantile(1 - pct/100)
        y_trim = y_full[(y_full >= lower) & (y_full <= upper)]

        if len(y_trim) < 5:
            continue

        n = len(y_trim)
        mean_val = y_trim.mean()
        se = y_trim.std() / np.sqrt(n)
        t_stat = mean_val / se if se > 0 else np.nan
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

        add_result(
            spec_id=f'robust/sample/trim_{pct}pct_{outcome}',
            spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
            outcome_var=outcome,
            treatment_var='mean_surplus',
            coef=mean_val,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=n,
            coef_vector={'treatment': {'var': outcome, 'coef': mean_val, 'se': se, 'pval': p_val}},
            sample_desc=f'Trimmed at {pct}%, {outcome_label}',
            model_type='Mean (trimmed)'
        )

print(f"  Completed outlier handling")

# ============================================================================
# ALTERNATIVE OUTCOMES
# ============================================================================

print("\n=== RUNNING ALTERNATIVE OUTCOMES ===")

# Create log-transformed outcomes
for outcome in OUTCOME_VARS.keys():
    valid = df_survey[df_survey[outcome].notna() & (df_survey[outcome] > 0)].copy()
    if len(valid) < 5:
        continue

    y_log = np.log(valid[outcome])
    n = len(y_log)
    mean_val = y_log.mean()
    se = y_log.std() / np.sqrt(n)
    t_stat = mean_val / se if se > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
    ci_lower = mean_val - 1.96 * se
    ci_upper = mean_val + 1.96 * se

    add_result(
        spec_id=f'robust/funcform/log_{outcome}',
        spec_tree_path='robustness/functional_form.md',
        outcome_var=f'log_{outcome}',
        treatment_var='mean_surplus',
        coef=mean_val,
        se=se,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=n,
        coef_vector={'treatment': {'var': f'log_{outcome}', 'coef': mean_val, 'se': se, 'pval': p_val}},
        sample_desc=f'Log-transformed {outcome}',
        model_type='Mean (log)'
    )

# IHS transformation
for outcome in OUTCOME_VARS.keys():
    valid = df_survey[df_survey[outcome].notna()].copy()
    if len(valid) < 5:
        continue

    y_ihs = np.arcsinh(valid[outcome])
    n = len(y_ihs)
    mean_val = y_ihs.mean()
    se = y_ihs.std() / np.sqrt(n)
    t_stat = mean_val / se if se > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
    ci_lower = mean_val - 1.96 * se
    ci_upper = mean_val + 1.96 * se

    add_result(
        spec_id=f'robust/funcform/ihs_{outcome}',
        spec_tree_path='robustness/functional_form.md',
        outcome_var=f'ihs_{outcome}',
        treatment_var='mean_surplus',
        coef=mean_val,
        se=se,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=n,
        coef_vector={'treatment': {'var': f'ihs_{outcome}', 'coef': mean_val, 'se': se, 'pval': p_val}},
        sample_desc=f'IHS-transformed {outcome}',
        model_type='Mean (IHS)'
    )

print(f"  Completed alternative outcomes")

# ============================================================================
# QUANTILE ANALYSIS
# ============================================================================

print("\n=== RUNNING QUANTILE ANALYSIS ===")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna()]
    if len(valid) < 10:
        continue

    y = valid[outcome]

    for q in quantiles:
        q_val = y.quantile(q)
        # Bootstrap SE for quantile
        n_boot = 500
        boot_quantiles = []
        for _ in range(n_boot):
            boot_sample = y.sample(n=len(y), replace=True)
            boot_quantiles.append(boot_sample.quantile(q))
        se = np.std(boot_quantiles)

        t_stat = q_val / se if se > 0 else np.nan
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y)-1)) if not np.isnan(t_stat) else np.nan
        ci_lower = q_val - 1.96 * se
        ci_upper = q_val + 1.96 * se

        add_result(
            spec_id=f'robust/quantile/q{int(q*100)}_{outcome}',
            spec_tree_path='methods/cross_sectional_ols.md#quantile',
            outcome_var=outcome,
            treatment_var=f'quantile_{int(q*100)}',
            coef=q_val,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=len(y),
            coef_vector={'treatment': {'var': f'q{int(q*100)}', 'coef': q_val, 'se': se, 'pval': p_val}},
            sample_desc=f'{int(q*100)}th percentile, {outcome_label}',
            model_type='Quantile'
        )

print(f"  Completed quantile analysis")

# ============================================================================
# HETEROGENEITY ANALYSIS - By Country/US Status
# ============================================================================

print("\n=== RUNNING HETEROGENEITY ANALYSIS ===")

for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna() & df_survey['c_US'].notna()].copy()
    if len(valid) < 10:
        continue

    # Test for heterogeneity by US status using regression
    try:
        X = sm.add_constant(valid['c_US'])
        y = valid[outcome]
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

        coef = model.params['c_US']
        se = model.bse['c_US']
        t_stat = model.tvalues['c_US']
        p_val = model.pvalues['c_US']
        ci = model.conf_int().loc['c_US']

        coef_vector = {
            'treatment': {'var': 'c_US', 'coef': coef, 'se': se, 'pval': p_val},
            'constant': {'var': 'const', 'coef': model.params['const'],
                        'se': model.bse['const'], 'pval': model.pvalues['const']},
            'diagnostics': {'r2': model.rsquared}
        }

        add_result(
            spec_id=f'robust/het/by_us_{outcome}',
            spec_tree_path='robustness/heterogeneity.md#by_country',
            outcome_var=outcome,
            treatment_var='c_US',
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector=coef_vector,
            sample_desc=f'Heterogeneity by US status, {outcome_label}',
            model_type='OLS'
        )
    except Exception as e:
        print(f"  Het by US ({outcome}) failed: {e}")

# Heterogeneity by experience level
for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna() & df_survey['numjobs'].notna()].copy()
    if len(valid) < 10:
        continue

    try:
        valid['high_experience'] = (valid['numjobs'] >= valid['numjobs'].median()).astype(int)
        X = sm.add_constant(valid['high_experience'])
        y = valid[outcome]
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

        coef = model.params['high_experience']
        se = model.bse['high_experience']
        t_stat = model.tvalues['high_experience']
        p_val = model.pvalues['high_experience']
        ci = model.conf_int().loc['high_experience']

        add_result(
            spec_id=f'robust/het/by_experience_{outcome}',
            spec_tree_path='robustness/heterogeneity.md#by_baseline_outcome',
            outcome_var=outcome,
            treatment_var='high_experience',
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector={'treatment': {'var': 'high_experience', 'coef': coef, 'se': se, 'pval': p_val}},
            sample_desc=f'Heterogeneity by experience level, {outcome_label}',
            model_type='OLS'
        )
    except Exception as e:
        print(f"  Het by experience ({outcome}) failed: {e}")

# Heterogeneity by rate level
for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna() & df_survey['scraped_profile_rate'].notna()].copy()
    if len(valid) < 10:
        continue

    try:
        valid['high_rate'] = (valid['scraped_profile_rate'] >= valid['scraped_profile_rate'].median()).astype(int)
        X = sm.add_constant(valid['high_rate'])
        y = valid[outcome]
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

        coef = model.params['high_rate']
        se = model.bse['high_rate']
        t_stat = model.tvalues['high_rate']
        p_val = model.pvalues['high_rate']
        ci = model.conf_int().loc['high_rate']

        add_result(
            spec_id=f'robust/het/by_rate_{outcome}',
            spec_tree_path='robustness/heterogeneity.md#by_income',
            outcome_var=outcome,
            treatment_var='high_rate',
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector={'treatment': {'var': 'high_rate', 'coef': coef, 'se': se, 'pval': p_val}},
            sample_desc=f'Heterogeneity by profile rate, {outcome_label}',
            model_type='OLS'
        )
    except Exception as e:
        print(f"  Het by rate ({outcome}) failed: {e}")

# Interaction: US x Experience
for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna() & df_survey['c_US'].notna() & df_survey['numjobs'].notna()].copy()
    if len(valid) < 15:
        continue

    try:
        valid['high_experience'] = (valid['numjobs'] >= valid['numjobs'].median()).astype(int)
        valid['us_x_exp'] = valid['c_US'] * valid['high_experience']
        formula = f'{outcome} ~ c_US + high_experience + us_x_exp'
        model = smf.ols(formula, data=valid).fit(cov_type='HC1')

        coef = model.params['us_x_exp']
        se = model.bse['us_x_exp']
        t_stat = model.tvalues['us_x_exp']
        p_val = model.pvalues['us_x_exp']
        ci = model.conf_int().loc['us_x_exp']

        add_result(
            spec_id=f'robust/het/interaction_us_exp_{outcome}',
            spec_tree_path='robustness/heterogeneity.md#interaction',
            outcome_var=outcome,
            treatment_var='us_x_experience',
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector={'treatment': {'var': 'us_x_exp', 'coef': coef, 'se': se, 'pval': p_val}},
            sample_desc=f'US x Experience interaction, {outcome_label}',
            controls_desc='c_US, high_experience',
            model_type='OLS'
        )
    except Exception as e:
        print(f"  Interaction ({outcome}) failed: {e}")

print(f"  Completed heterogeneity analysis")

# ============================================================================
# INFERENCE VARIATIONS - Bootstrap
# ============================================================================

print("\n=== RUNNING INFERENCE VARIATIONS ===")

n_bootstrap = 1000

for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna()]
    if len(valid) < 10:
        continue

    y = valid[outcome]

    # Bootstrap the mean
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = y.sample(n=len(y), replace=True)
        boot_means.append(boot_sample.mean())

    mean_val = y.mean()
    boot_se = np.std(boot_means)
    boot_ci_lower = np.percentile(boot_means, 2.5)
    boot_ci_upper = np.percentile(boot_means, 97.5)
    t_stat = mean_val / boot_se if boot_se > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y)-1)) if not np.isnan(t_stat) else np.nan

    add_result(
        spec_id=f'robust/inference/bootstrap_{outcome}',
        spec_tree_path='robustness/inference_alternatives.md',
        outcome_var=outcome,
        treatment_var='mean_surplus',
        coef=mean_val,
        se=boot_se,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=boot_ci_lower,
        ci_upper=boot_ci_upper,
        n_obs=len(y),
        coef_vector={'treatment': {'var': outcome, 'coef': mean_val, 'se': boot_se, 'pval': p_val},
                    'bootstrap': {'n_reps': n_bootstrap, 'ci_method': 'percentile'}},
        sample_desc=f'Bootstrap inference (n={n_bootstrap}), {outcome_label}',
        model_type='Mean (bootstrap)'
    )

print(f"  Completed inference variations")

# ============================================================================
# WEIGHTING VARIATIONS
# ============================================================================

print("\n=== RUNNING WEIGHTING VARIATIONS ===")

# Different weighting schemes
for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna() & df_survey['weight'].notna()].copy()
    if len(valid) < 10:
        continue

    y = valid[outcome]
    weights = valid['weight']

    # Unweighted (baseline - already done)

    # Weighted (already done in baseline_weighted)

    # Trimmed weights (cap at 90th percentile)
    weight_cap = weights.quantile(0.90)
    weights_trimmed = weights.clip(upper=weight_cap)
    weighted_mean = np.average(y, weights=weights_trimmed)
    weighted_var = np.average((y - weighted_mean)**2, weights=weights_trimmed)
    weighted_se = np.sqrt(weighted_var / len(y))

    t_stat = weighted_mean / weighted_se if weighted_se > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y)-1)) if not np.isnan(t_stat) else np.nan
    ci_lower = weighted_mean - 1.96 * weighted_se
    ci_upper = weighted_mean + 1.96 * weighted_se

    add_result(
        spec_id=f'robust/weights/trimmed_{outcome}',
        spec_tree_path='robustness/measurement.md#weights',
        outcome_var=outcome,
        treatment_var='mean_surplus',
        coef=weighted_mean,
        se=weighted_se,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_obs=len(valid),
        coef_vector={'treatment': {'var': outcome, 'coef': weighted_mean, 'se': weighted_se, 'pval': p_val}},
        sample_desc=f'Trimmed weights (cap at 90th pct), {outcome_label}',
        model_type='Weighted Mean (trimmed weights)'
    )

    # Normalized weights
    weights_norm = weights / weights.sum() * len(weights)
    weighted_mean_norm = np.average(y, weights=weights_norm)
    weighted_var_norm = np.average((y - weighted_mean_norm)**2, weights=weights_norm)
    weighted_se_norm = np.sqrt(weighted_var_norm / len(y))

    t_stat_norm = weighted_mean_norm / weighted_se_norm if weighted_se_norm > 0 else np.nan
    p_val_norm = 2 * (1 - stats.t.cdf(abs(t_stat_norm), df=len(y)-1)) if not np.isnan(t_stat_norm) else np.nan
    ci_lower_norm = weighted_mean_norm - 1.96 * weighted_se_norm
    ci_upper_norm = weighted_mean_norm + 1.96 * weighted_se_norm

    add_result(
        spec_id=f'robust/weights/normalized_{outcome}',
        spec_tree_path='robustness/measurement.md#weights',
        outcome_var=outcome,
        treatment_var='mean_surplus',
        coef=weighted_mean_norm,
        se=weighted_se_norm,
        t_stat=t_stat_norm,
        p_value=p_val_norm,
        ci_lower=ci_lower_norm,
        ci_upper=ci_upper_norm,
        n_obs=len(valid),
        coef_vector={'treatment': {'var': outcome, 'coef': weighted_mean_norm, 'se': weighted_se_norm, 'pval': p_val_norm}},
        sample_desc=f'Normalized weights, {outcome_label}',
        model_type='Weighted Mean (normalized weights)'
    )

print(f"  Completed weighting variations")

# ============================================================================
# PREDICTORS OF SURPLUS
# ============================================================================

print("\n=== RUNNING PREDICTOR REGRESSIONS ===")

for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna()].copy()
    valid['log_numjobs'] = np.log(valid['numjobs'])
    valid['log_profile_rate'] = np.log(valid['scraped_profile_rate'].replace(0, np.nan))
    valid = valid.dropna(subset=['log_numjobs', 'log_profile_rate', 'c_US'])

    if len(valid) < 15:
        continue

    try:
        formula = f'{outcome} ~ log_numjobs + log_profile_rate + c_US'
        model = smf.ols(formula, data=valid).fit(cov_type='HC1')

        for var in ['log_numjobs', 'log_profile_rate', 'c_US']:
            coef = model.params[var]
            se = model.bse[var]
            t_stat = model.tvalues[var]
            p_val = model.pvalues[var]
            ci = model.conf_int().loc[var]

            coef_vector = {
                'treatment': {'var': var, 'coef': coef, 'se': se, 'pval': p_val},
                'all_coefs': {v: {'coef': model.params[v], 'se': model.bse[v], 'pval': model.pvalues[v]}
                             for v in model.params.index},
                'diagnostics': {'r2': model.rsquared, 'f_stat': model.fvalue}
            }

            add_result(
                spec_id=f'ols/predictors/{var}_{outcome}',
                spec_tree_path='methods/cross_sectional_ols.md',
                outcome_var=outcome,
                treatment_var=var,
                coef=coef,
                se=se,
                t_stat=t_stat,
                p_value=p_val,
                ci_lower=ci[0],
                ci_upper=ci[1],
                n_obs=model.nobs,
                r_squared=model.rsquared,
                coef_vector=coef_vector,
                sample_desc=f'Predictors of surplus, {outcome_label}',
                controls_desc='log_numjobs, log_profile_rate, c_US',
                model_type='OLS'
            )
    except Exception as e:
        print(f"  Predictor regression ({outcome}) failed: {e}")

print(f"  Completed predictor regressions")

# ============================================================================
# PLACEBO/FALSIFICATION TESTS
# ============================================================================

print("\n=== RUNNING PLACEBO TESTS ===")

# Test if profile rate (which shouldn't depend on survey framing) differs by survey participation
# This is already captured in balance tests, but we'll add explicit placebo specifications

for var in ['scraped_profile_rate', 'numjobs']:
    valid = df_full[df_full[var].notna()].copy()
    if len(valid) < 20:
        continue

    try:
        X = sm.add_constant(valid['InSurvey'])
        y = valid[var]
        model = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

        coef = model.params['InSurvey']
        se = model.bse['InSurvey']
        t_stat = model.tvalues['InSurvey']
        p_val = model.pvalues['InSurvey']
        ci = model.conf_int().loc['InSurvey']

        add_result(
            spec_id=f'robust/placebo/{var}_on_survey',
            spec_tree_path='robustness/placebo_tests.md',
            outcome_var=var,
            treatment_var='InSurvey',
            coef=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci[0],
            ci_upper=ci[1],
            n_obs=model.nobs,
            r_squared=model.rsquared,
            coef_vector={'treatment': {'var': 'InSurvey', 'coef': coef, 'se': se, 'pval': p_val}},
            sample_desc=f'Placebo: {var} should not differ by survey status',
            model_type='OLS'
        )
    except Exception as e:
        print(f"  Placebo ({var}) failed: {e}")

print(f"  Completed placebo tests")

# ============================================================================
# ADDITIONAL SAMPLE SPLITS
# ============================================================================

print("\n=== RUNNING ADDITIONAL SAMPLE SPLITS ===")

# Split by higher_bid_framing (survey question framing)
for outcome, outcome_label in OUTCOME_VARS.items():
    valid = df_survey[df_survey[outcome].notna() & df_survey['higher_bid_framing'].notna()].copy()

    for framing_val in [0, 1]:
        subset = valid[valid['higher_bid_framing'] == framing_val]
        if len(subset) < 5:
            continue

        y = subset[outcome]
        n = len(y)
        mean_val = y.mean()
        se = y.std() / np.sqrt(n)
        t_stat = mean_val / se if se > 0 else np.nan
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if not np.isnan(t_stat) else np.nan
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

        framing_label = 'higher_bid_framing' if framing_val == 1 else 'lower_bid_framing'

        add_result(
            spec_id=f'robust/sample/{framing_label}_{outcome}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            treatment_var='mean_surplus',
            coef=mean_val,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=n,
            coef_vector={'treatment': {'var': outcome, 'coef': mean_val, 'se': se, 'pval': p_val}},
            sample_desc=f'{framing_label}, {outcome_label}',
            model_type='Mean'
        )

print(f"  Completed additional sample splits")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n=== SAVING RESULTS ===")

results_df = pd.DataFrame(results)
print(f"Total specifications: {len(results_df)}")

# Save to package directory
output_path = f"{PACKAGE_PATH}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n=== SUMMARY STATISTICS ===")
print(f"Total specifications run: {len(results_df)}")
print(f"\nBy spec category:")
results_df['spec_category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else x)
print(results_df['spec_category'].value_counts())

# Coefficient summary
coefs = results_df['coefficient'].dropna()
print(f"\nCoefficient summary:")
print(f"  Mean: {coefs.mean():.4f}")
print(f"  Median: {coefs.median():.4f}")
print(f"  Std: {coefs.std():.4f}")
print(f"  Min: {coefs.min():.4f}")
print(f"  Max: {coefs.max():.4f}")

# Significance summary
sig_05 = (results_df['p_value'] < 0.05).sum()
sig_01 = (results_df['p_value'] < 0.01).sum()
print(f"\nSignificance:")
print(f"  Significant at 5%: {sig_05} ({100*sig_05/len(results_df):.1f}%)")
print(f"  Significant at 1%: {sig_01} ({100*sig_01/len(results_df):.1f}%)")

# Positive coefficients
pos_coef = (results_df['coefficient'] > 0).sum()
print(f"  Positive coefficients: {pos_coef} ({100*pos_coef/len(results_df):.1f}%)")

print("\n=== SPECIFICATION SEARCH COMPLETE ===")
