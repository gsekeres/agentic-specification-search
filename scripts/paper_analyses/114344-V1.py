"""
Specification Search Script for Paper 114344-V1
"Incomplete Disclosure: Evidence of Signaling and Countersignaling"
by Bederson, Jin, Leslie, Quinn, Zou (AER 2018)

This script replicates and extends the main analysis from Table 4 (Disclosure by Imputed Grade).

Paper Overview:
- Studies restaurant hygiene grade disclosure in Maricopa County, Arizona
- Restaurants can choose whether to display their letter grade (A, B, C, D)
- Main hypothesis: Disclosure decision depends on grade quality within grade category
  - Signaling: Better restaurants within a grade category more likely to disclose
  - Countersignaling: Very good restaurants (A+) may not need to disclose

Treatment: Half-grade dummies (A+, A-, B+, B-, CD+, CD-)
Outcome: Disclosure decision (binary: 0/1)
Method: Cross-sectional OLS with binary outcome (Linear Probability Model)

Key test: Whether A+ restaurants are more likely to disclose than A- restaurants
(tests signaling vs countersignaling hypothesis)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
PAPER_ID = "114344-V1"
PAPER_TITLE = "Incomplete Disclosure: Evidence of Signaling and Countersignaling"
JOURNAL = "AER"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114344-V1/20150178_Replica/"
OUTPUT_PATH = DATA_PATH

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
def load_and_prepare_data():
    """Load and prepare data following Tbl4TblA.do specifications"""
    df = pd.read_stata(DATA_PATH + "Sample.dta")

    # Apply sample selection (from Tbl4TblA.do)
    # Drop where grade is empty
    df = df[df['grade'] != ''].copy()

    # Drop where vios != vios2
    df = df[df['vios'] == df['vios2']].copy()

    # Keep last inspection per restaurant
    df = df.sort_values(['restaurant_id', 'ymd'], ascending=[True, False])
    df = df.groupby('restaurant_id').first().reset_index()

    # Drop A grades with violations (inconsistent A grades)
    df = df[~((df['grade'] == 'A') & ((df['vio_p'] > 0) | (df['vio_pf'] > 0) | (df['vio_core'] >= 4)))]

    # Create half-grade definitions (A+/A-, B+/B-, CD+/CD-)
    # Following the Stata code: split by median num_grading among disclosed restaurants

    # For A grade: median among disclosed A-predicted restaurants
    median_A = df.loc[(df['isA_pred1'] == 1) & (df['disclosure'] == 1), 'num_grading'].median()
    df['AP_def1'] = ((df['num_grading'] <= median_A) & (df['isA_pred1'] == 1)).astype(int)
    df['AM_def1'] = ((df['num_grading'] > median_A) & (df['isA_pred1'] == 1)).astype(int)

    # For B grade
    median_B = df.loc[(df['isB_pred1'] == 1) & (df['disclosure'] == 1), 'num_grading'].median()
    df['BP_def1'] = ((df['num_grading'] <= median_B) & (df['isB_pred1'] == 1)).astype(int)
    df['BM_def1'] = ((df['num_grading'] > median_B) & (df['isB_pred1'] == 1)).astype(int)

    # For CD grade
    median_CD = df.loc[(df['isCD_pred1'] == 1) & (df['disclosure'] == 1), 'num_grading'].median()
    df['CDP_def1'] = ((df['num_grading'] <= median_CD) & (df['isCD_pred1'] == 1)).astype(int)
    df['CDM_def1'] = ((df['num_grading'] > median_CD) & (df['isCD_pred1'] == 1)).astype(int)

    # Create yrseason dummies
    df['yrseason'] = df['yrseason'].astype(int)
    yrseason_dummies = pd.get_dummies(df['yrseason'], prefix='yrseason', drop_first=True)
    df = pd.concat([df, yrseason_dummies], axis=1)

    # Create ZIP-level controls
    df['ZIP_firstbatch_frac'] = df.groupby('zip_num')['firstbatch'].transform('mean')
    df['ZIP_restaurants'] = df.groupby('zip_num')['restaurant_id'].transform('count')
    df['ZIP_restaurants_1000'] = df['ZIP_restaurants'] / 1000
    df['zip_mean_num_grading'] = df.groupby('zip_num')['num_grading'].transform('mean')

    return df

# ============================================================================
# RESULTS STORAGE
# ============================================================================
results = []

def extract_results(model, spec_id, spec_tree_path, df, outcome_var, treatment_var,
                   controls_desc, fixed_effects, cluster_var, model_type="LPM"):
    """Extract and format regression results"""

    # Get coefficients for all variables
    coef_dict = {}
    if hasattr(model, 'params'):
        params = model.params
        se = model.bse
        pvals = model.pvalues

        for var in params.index:
            if var != 'Intercept':
                coef_dict[var] = {
                    "var": var,
                    "coef": float(params[var]),
                    "se": float(se[var]),
                    "pval": float(pvals[var])
                }

    # Get treatment coefficient (use A+ as primary treatment variable for comparison)
    treatment_coef = float(model.params.get(treatment_var, np.nan))
    treatment_se = float(model.bse.get(treatment_var, np.nan))
    treatment_pval = float(model.pvalues.get(treatment_var, np.nan))
    treatment_tstat = treatment_coef / treatment_se if treatment_se > 0 else np.nan
    ci_lower = treatment_coef - 1.96 * treatment_se
    ci_upper = treatment_coef + 1.96 * treatment_se

    # Build coefficient vector JSON
    coefficient_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": treatment_coef,
            "se": treatment_se,
            "pval": treatment_pval
        },
        "controls": [v for k, v in coef_dict.items() if k != treatment_var],
        "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
        "diagnostics": {
            "r_squared": float(model.rsquared) if hasattr(model, 'rsquared') else None,
            "f_stat": float(model.fvalue) if hasattr(model, 'fvalue') else None,
            "f_pval": float(model.f_pvalue) if hasattr(model, 'f_pvalue') else None
        }
    }

    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': treatment_coef,
        'std_error': treatment_se,
        't_stat': treatment_tstat,
        'p_value': treatment_pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared) if hasattr(model, 'rsquared') else None,
        'coefficient_vector_json': json.dumps(coefficient_vector),
        'sample_desc': 'Last inspection per restaurant, post-disclosure period',
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

    return result

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def run_specifications():
    """Run all specification searches"""
    global results

    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Sample size: {len(df)}")

    # Get yrseason dummy columns
    yrseason_cols = [c for c in df.columns if c.startswith('yrseason_')]
    yrseason_formula = ' + '.join(yrseason_cols)

    # Base controls
    base_controls = ['firstbatch', 'ischain', 'inyelp']
    extended_controls = base_controls + ['mean_num_grading', 'sd_num_grading']
    zip_controls = extended_controls + ['zip_mean_num_grading', 'ZIP_firstbatch_frac', 'ZIP_restaurants_1000']
    lag_controls = extended_controls + ['isA_pred1_lag1', 'isB_pred1_lag1']

    # Treatment dummies (half-grades, with CD- as omitted)
    half_grade_dummies = ['AP_def1', 'AM_def1', 'BP_def1', 'BM_def1', 'CDP_def1']

    # ========================================================================
    # BASELINE SPECIFICATION (Table 4, Column 2)
    # ========================================================================
    print("\n--- Running Baseline Specification ---")

    # Prepare data (drop missing)
    analysis_vars = ['disclosure'] + half_grade_dummies + extended_controls + yrseason_cols
    df_reg = df.dropna(subset=analysis_vars).copy()

    formula = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(extended_controls)} + {yrseason_formula}"
    model = smf.ols(formula, data=df_reg).fit(cov_type='HC1')

    print(f"Baseline N = {int(model.nobs)}")
    print(f"A+ coefficient: {model.params['AP_def1']:.4f} (SE: {model.bse['AP_def1']:.4f})")
    print(f"A- coefficient: {model.params['AM_def1']:.4f} (SE: {model.bse['AM_def1']:.4f})")

    # Test A+ > A- (main hypothesis test)
    t_test_AP_AM = model.t_test('AP_def1 - AM_def1 = 0')
    t_val = float(t_test_AP_AM.tvalue.flatten()[0]) if hasattr(t_test_AP_AM.tvalue, 'flatten') else float(t_test_AP_AM.tvalue)
    p_val = float(t_test_AP_AM.pvalue.flatten()[0]) if hasattr(t_test_AP_AM.pvalue, 'flatten') else float(t_test_AP_AM.pvalue)
    print(f"Test A+ = A-: t = {t_val:.3f}, p = {p_val:.4f}")

    results.append(extract_results(
        model, 'baseline', 'methods/discrete_choice.md#baseline',
        df_reg, 'disclosure', 'AP_def1',
        'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    # ========================================================================
    # METHOD-SPECIFIC SPECIFICATIONS
    # ========================================================================

    # --- Discrete Choice: Logit ---
    print("\n--- Running Logit Specification ---")
    try:
        logit_model = smf.logit(formula, data=df_reg).fit(disp=0, maxiter=200)
        mfx = logit_model.get_margeff(at='overall')
        mfx_df = mfx.summary_frame()

        # Get AP_def1 marginal effect by variable name
        ap_mfx = float(mfx_df.loc['AP_def1', 'dy/dx'])
        ap_mfx_se = float(mfx_df.loc['AP_def1', 'Std. Err.'])
        ap_mfx_pval = float(mfx_df.loc['AP_def1', 'Pr(>|z|)'])

        # For logit, report average marginal effects
        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'discrete/binary/logit',
            'spec_tree_path': 'methods/discrete_choice.md#model-type-binary-outcome',
            'outcome_var': 'disclosure',
            'treatment_var': 'AP_def1',
            'coefficient': ap_mfx,
            'std_error': ap_mfx_se,
            't_stat': ap_mfx / ap_mfx_se if ap_mfx_se > 0 else np.nan,
            'p_value': ap_mfx_pval,
            'ci_lower': ap_mfx - 1.96 * ap_mfx_se,
            'ci_upper': ap_mfx + 1.96 * ap_mfx_se,
            'n_obs': int(logit_model.nobs),
            'r_squared': float(logit_model.prsquared),
            'coefficient_vector_json': json.dumps({
                "treatment": {"var": "AP_def1", "coef": float(logit_model.params['AP_def1']),
                              "se": float(logit_model.bse['AP_def1']), "marginal_effect": ap_mfx},
                "model_info": {"pseudo_r2": float(logit_model.prsquared), "ll": float(logit_model.llf)}
            }),
            'sample_desc': 'Last inspection per restaurant, post-disclosure period',
            'fixed_effects': 'yrseason',
            'controls_desc': 'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'cluster_var': 'robust',
            'model_type': 'Logit (AME)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"Logit A+ AME: {ap_mfx:.4f}")
    except Exception as e:
        print(f"Logit failed: {e}")

    # --- Discrete Choice: Probit ---
    print("\n--- Running Probit Specification ---")
    try:
        probit_model = smf.probit(formula, data=df_reg).fit(disp=0, maxiter=200)
        mfx_probit = probit_model.get_margeff(at='overall')
        mfx_probit_df = mfx_probit.summary_frame()

        # Get AP_def1 marginal effect by variable name
        ap_mfx_probit = float(mfx_probit_df.loc['AP_def1', 'dy/dx'])
        ap_mfx_probit_se = float(mfx_probit_df.loc['AP_def1', 'Std. Err.'])
        ap_mfx_probit_pval = float(mfx_probit_df.loc['AP_def1', 'Pr(>|z|)'])

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'discrete/binary/probit',
            'spec_tree_path': 'methods/discrete_choice.md#model-type-binary-outcome',
            'outcome_var': 'disclosure',
            'treatment_var': 'AP_def1',
            'coefficient': ap_mfx_probit,
            'std_error': ap_mfx_probit_se,
            't_stat': ap_mfx_probit / ap_mfx_probit_se if ap_mfx_probit_se > 0 else np.nan,
            'p_value': ap_mfx_probit_pval,
            'ci_lower': ap_mfx_probit - 1.96 * ap_mfx_probit_se,
            'ci_upper': ap_mfx_probit + 1.96 * ap_mfx_probit_se,
            'n_obs': int(probit_model.nobs),
            'r_squared': float(probit_model.prsquared),
            'coefficient_vector_json': json.dumps({
                "treatment": {"var": "AP_def1", "coef": float(probit_model.params['AP_def1']),
                              "se": float(probit_model.bse['AP_def1']), "marginal_effect": ap_mfx_probit},
                "model_info": {"pseudo_r2": float(probit_model.prsquared), "ll": float(probit_model.llf)}
            }),
            'sample_desc': 'Last inspection per restaurant, post-disclosure period',
            'fixed_effects': 'yrseason',
            'controls_desc': 'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'cluster_var': 'robust',
            'model_type': 'Probit (AME)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"Probit A+ AME: {ap_mfx_probit:.4f}")
    except Exception as e:
        print(f"Probit failed: {e}")

    # ========================================================================
    # CONTROL VARIATIONS (from Table 4 columns)
    # ========================================================================

    # --- Column 1: Basic controls (no mean/sd num_grading) ---
    print("\n--- Running Control Variation: Basic Controls ---")
    formula_basic = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(base_controls)} + {yrseason_formula}"
    df_basic = df.dropna(subset=['disclosure'] + half_grade_dummies + base_controls + yrseason_cols).copy()
    model_basic = smf.ols(formula_basic, data=df_basic).fit(cov_type='HC1')

    results.append(extract_results(
        model_basic, 'ols/controls/baseline', 'methods/cross_sectional_ols.md#control-sets',
        df_basic, 'disclosure', 'AP_def1',
        'firstbatch, ischain, inyelp, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    # --- Column 3: ZIP-level controls ---
    print("\n--- Running Control Variation: ZIP Controls ---")
    formula_zip = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(zip_controls)} + {yrseason_formula}"
    df_zip = df.dropna(subset=['disclosure'] + half_grade_dummies + zip_controls + yrseason_cols).copy()
    model_zip = smf.ols(formula_zip, data=df_zip).fit(cov_type='HC1')

    results.append(extract_results(
        model_zip, 'ols/controls/full', 'methods/cross_sectional_ols.md#control-sets',
        df_zip, 'disclosure', 'AP_def1',
        'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, zip_mean_num_grading, ZIP_firstbatch_frac, ZIP_restaurants_1000, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    # --- Column 4: Lagged grade controls ---
    print("\n--- Running Control Variation: Lag Controls ---")
    formula_lag = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(lag_controls)} + {yrseason_formula}"
    df_lag = df.dropna(subset=['disclosure'] + half_grade_dummies + lag_controls + yrseason_cols).copy()
    model_lag = smf.ols(formula_lag, data=df_lag).fit(cov_type='HC1')

    results.append(extract_results(
        model_lag, 'ols/controls/kitchen_sink', 'methods/cross_sectional_ols.md#control-sets',
        df_lag, 'disclosure', 'AP_def1',
        'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, isA_pred1_lag1, isB_pred1_lag1, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    # --- No controls (bivariate) ---
    print("\n--- Running Control Variation: No Controls ---")
    formula_none = f"disclosure ~ {' + '.join(half_grade_dummies)} + {yrseason_formula}"
    df_none = df.dropna(subset=['disclosure'] + half_grade_dummies + yrseason_cols).copy()
    model_none = smf.ols(formula_none, data=df_none).fit(cov_type='HC1')

    results.append(extract_results(
        model_none, 'ols/controls/none', 'methods/cross_sectional_ols.md#control-sets',
        df_none, 'disclosure', 'AP_def1',
        'yrseason FE only',
        'yrseason', 'robust', 'LPM'
    ))

    # ========================================================================
    # ROBUSTNESS: LEAVE-ONE-OUT
    # ========================================================================
    print("\n--- Running Leave-One-Out Robustness ---")

    for control in extended_controls:
        remaining_controls = [c for c in extended_controls if c != control]
        formula_loo = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(remaining_controls)} + {yrseason_formula}"
        df_loo = df.dropna(subset=['disclosure'] + half_grade_dummies + remaining_controls + yrseason_cols).copy()

        try:
            model_loo = smf.ols(formula_loo, data=df_loo).fit(cov_type='HC1')
            results.append(extract_results(
                model_loo, f'robust/loo/drop_{control}', 'robustness/leave_one_out.md',
                df_loo, 'disclosure', 'AP_def1',
                f"Baseline minus {control}",
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  Dropped {control}: A+ coef = {model_loo.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Failed for {control}: {e}")

    # ========================================================================
    # ROBUSTNESS: SINGLE COVARIATE
    # ========================================================================
    print("\n--- Running Single Covariate Robustness ---")

    # Bivariate (only treatment dummies + FE)
    results.append(extract_results(
        model_none, 'robust/single/none', 'robustness/single_covariate.md',
        df_none, 'disclosure', 'AP_def1',
        'None (bivariate with FE)',
        'yrseason', 'robust', 'LPM'
    ))

    for control in extended_controls:
        formula_single = f"disclosure ~ {' + '.join(half_grade_dummies)} + {control} + {yrseason_formula}"
        df_single = df.dropna(subset=['disclosure'] + half_grade_dummies + [control] + yrseason_cols).copy()

        try:
            model_single = smf.ols(formula_single, data=df_single).fit(cov_type='HC1')
            results.append(extract_results(
                model_single, f'robust/single/{control}', 'robustness/single_covariate.md',
                df_single, 'disclosure', 'AP_def1',
                f"{control} only",
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  With {control} only: A+ coef = {model_single.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Failed for {control}: {e}")

    # ========================================================================
    # ROBUSTNESS: CLUSTERING VARIATIONS
    # ========================================================================
    print("\n--- Running Clustering Robustness ---")

    # No clustering (already have HC1)
    # Cluster by ZIP code
    try:
        model_zip_cluster = smf.ols(formula, data=df_reg).fit(
            cov_type='cluster', cov_kwds={'groups': df_reg['zip_num']}
        )
        results.append(extract_results(
            model_zip_cluster, 'robust/cluster/region', 'robustness/clustering_variations.md#single-level-clustering',
            df_reg, 'disclosure', 'AP_def1',
            'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'yrseason', 'zip_num', 'LPM'
        ))
        print(f"  Clustered by ZIP: A+ SE = {model_zip_cluster.bse['AP_def1']:.4f}")
    except Exception as e:
        print(f"  ZIP clustering failed: {e}")

    # Cluster by yrseason
    try:
        model_time_cluster = smf.ols(formula, data=df_reg).fit(
            cov_type='cluster', cov_kwds={'groups': df_reg['yrseason']}
        )
        results.append(extract_results(
            model_time_cluster, 'robust/cluster/time', 'robustness/clustering_variations.md#single-level-clustering',
            df_reg, 'disclosure', 'AP_def1',
            'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'yrseason', 'yrseason', 'LPM'
        ))
        print(f"  Clustered by yrseason: A+ SE = {model_time_cluster.bse['AP_def1']:.4f}")
    except Exception as e:
        print(f"  Time clustering failed: {e}")

    # HC2 and HC3
    try:
        model_hc2 = smf.ols(formula, data=df_reg).fit(cov_type='HC2')
        results.append(extract_results(
            model_hc2, 'robust/se/hc2', 'robustness/clustering_variations.md#alternative-se-methods',
            df_reg, 'disclosure', 'AP_def1',
            'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'yrseason', 'HC2', 'LPM'
        ))
        print(f"  HC2: A+ SE = {model_hc2.bse['AP_def1']:.4f}")
    except Exception as e:
        print(f"  HC2 failed: {e}")

    try:
        model_hc3 = smf.ols(formula, data=df_reg).fit(cov_type='HC3')
        results.append(extract_results(
            model_hc3, 'robust/se/hc3', 'robustness/clustering_variations.md#alternative-se-methods',
            df_reg, 'disclosure', 'AP_def1',
            'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'yrseason', 'HC3', 'LPM'
        ))
        print(f"  HC3: A+ SE = {model_hc3.bse['AP_def1']:.4f}")
    except Exception as e:
        print(f"  HC3 failed: {e}")

    # ========================================================================
    # ROBUSTNESS: SAMPLE RESTRICTIONS
    # ========================================================================
    print("\n--- Running Sample Restriction Robustness ---")

    # A-grade restaurants only
    df_A_only = df_reg[df_reg['isA_pred1'] == 1].copy()
    if len(df_A_only) > 100:
        formula_A = f"disclosure ~ AP_def1 + AM_def1 + {' + '.join(extended_controls)} + {yrseason_formula}"
        # Need to handle case where AM_def1 might be collinear
        formula_A_simple = f"disclosure ~ AP_def1 + {' + '.join(extended_controls)} + {yrseason_formula}"
        try:
            model_A = smf.ols(formula_A_simple, data=df_A_only).fit(cov_type='HC1')
            results.append(extract_results(
                model_A, 'robust/sample/subsample_A_grade', 'robustness/sample_restrictions.md#demographic-subgroups',
                df_A_only, 'disclosure', 'AP_def1',
                'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  A-grade only (N={len(df_A_only)}): A+ coef = {model_A.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  A-grade subsample failed: {e}")

    # B-grade restaurants only
    df_B_only = df_reg[df_reg['isB_pred1'] == 1].copy()
    if len(df_B_only) > 100:
        formula_B_simple = f"disclosure ~ BP_def1 + {' + '.join(extended_controls)} + {yrseason_formula}"
        try:
            model_B = smf.ols(formula_B_simple, data=df_B_only).fit(cov_type='HC1')
            results.append(extract_results(
                model_B, 'robust/sample/subsample_B_grade', 'robustness/sample_restrictions.md#demographic-subgroups',
                df_B_only, 'disclosure', 'BP_def1',
                'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  B-grade only (N={len(df_B_only)}): B+ coef = {model_B.params['BP_def1']:.4f}")
        except Exception as e:
            print(f"  B-grade subsample failed: {e}")

    # Yelp restaurants only
    df_yelp_only = df_reg[df_reg['inyelp'] == 1].copy()
    if len(df_yelp_only) > 100:
        controls_no_inyelp = [c for c in extended_controls if c != 'inyelp']
        formula_yelp = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(controls_no_inyelp)} + {yrseason_formula}"
        try:
            model_yelp = smf.ols(formula_yelp, data=df_yelp_only).fit(cov_type='HC1')
            results.append(extract_results(
                model_yelp, 'robust/sample/yelp_only', 'robustness/sample_restrictions.md#demographic-subgroups',
                df_yelp_only, 'disclosure', 'AP_def1',
                'firstbatch, ischain, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  Yelp restaurants only (N={len(df_yelp_only)}): A+ coef = {model_yelp.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Yelp subsample failed: {e}")

    # Non-Yelp restaurants only
    df_no_yelp = df_reg[df_reg['inyelp'] == 0].copy()
    if len(df_no_yelp) > 100:
        try:
            model_no_yelp = smf.ols(formula_yelp, data=df_no_yelp).fit(cov_type='HC1')
            results.append(extract_results(
                model_no_yelp, 'robust/sample/non_yelp', 'robustness/sample_restrictions.md#demographic-subgroups',
                df_no_yelp, 'disclosure', 'AP_def1',
                'firstbatch, ischain, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  Non-Yelp restaurants only (N={len(df_no_yelp)}): A+ coef = {model_no_yelp.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Non-Yelp subsample failed: {e}")

    # Chain restaurants only
    df_chain = df_reg[df_reg['ischain'] == 1].copy()
    if len(df_chain) > 100:
        controls_no_chain = [c for c in extended_controls if c != 'ischain']
        formula_chain = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(controls_no_chain)} + {yrseason_formula}"
        try:
            model_chain = smf.ols(formula_chain, data=df_chain).fit(cov_type='HC1')
            results.append(extract_results(
                model_chain, 'robust/sample/chain_only', 'robustness/sample_restrictions.md#demographic-subgroups',
                df_chain, 'disclosure', 'AP_def1',
                'firstbatch, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  Chain restaurants only (N={len(df_chain)}): A+ coef = {model_chain.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Chain subsample failed: {e}")

    # Non-chain restaurants
    df_non_chain = df_reg[df_reg['ischain'] == 0].copy()
    if len(df_non_chain) > 100:
        try:
            model_non_chain = smf.ols(formula_chain, data=df_non_chain).fit(cov_type='HC1')
            results.append(extract_results(
                model_non_chain, 'robust/sample/non_chain', 'robustness/sample_restrictions.md#demographic-subgroups',
                df_non_chain, 'disclosure', 'AP_def1',
                'firstbatch, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  Non-chain restaurants only (N={len(df_non_chain)}): A+ coef = {model_non_chain.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Non-chain subsample failed: {e}")

    # First batch only
    df_firstbatch = df_reg[df_reg['firstbatch'] == 1].copy()
    if len(df_firstbatch) > 100:
        controls_no_fb = [c for c in extended_controls if c != 'firstbatch']
        formula_fb = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(controls_no_fb)} + {yrseason_formula}"
        try:
            model_fb = smf.ols(formula_fb, data=df_firstbatch).fit(cov_type='HC1')
            results.append(extract_results(
                model_fb, 'robust/sample/first_batch', 'robustness/sample_restrictions.md#time-based-restrictions',
                df_firstbatch, 'disclosure', 'AP_def1',
                'ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  First batch only (N={len(df_firstbatch)}): A+ coef = {model_fb.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  First batch subsample failed: {e}")

    # Later batches only
    df_later = df_reg[df_reg['firstbatch'] == 0].copy()
    if len(df_later) > 100:
        try:
            model_later = smf.ols(formula_fb, data=df_later).fit(cov_type='HC1')
            results.append(extract_results(
                model_later, 'robust/sample/later_batch', 'robustness/sample_restrictions.md#time-based-restrictions',
                df_later, 'disclosure', 'AP_def1',
                'ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
                'yrseason', 'robust', 'LPM'
            ))
            print(f"  Later batches only (N={len(df_later)}): A+ coef = {model_later.params['AP_def1']:.4f}")
        except Exception as e:
            print(f"  Later batch subsample failed: {e}")

    # ========================================================================
    # ROBUSTNESS: FUNCTIONAL FORM
    # ========================================================================
    print("\n--- Running Functional Form Robustness ---")

    # Include quadratic terms for continuous controls
    df_reg['mean_num_grading_sq'] = df_reg['mean_num_grading'] ** 2
    df_reg['sd_num_grading_sq'] = df_reg['sd_num_grading'] ** 2

    formula_quad = f"disclosure ~ {' + '.join(half_grade_dummies)} + {' + '.join(extended_controls)} + mean_num_grading_sq + sd_num_grading_sq + {yrseason_formula}"
    try:
        model_quad = smf.ols(formula_quad, data=df_reg).fit(cov_type='HC1')
        results.append(extract_results(
            model_quad, 'robust/form/quadratic', 'robustness/functional_form.md#nonlinear-specifications',
            df_reg, 'disclosure', 'AP_def1',
            'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, quadratic terms, yrseason FE',
            'yrseason', 'robust', 'LPM'
        ))
        print(f"  Quadratic controls: A+ coef = {model_quad.params['AP_def1']:.4f}")
    except Exception as e:
        print(f"  Quadratic failed: {e}")

    # Use num_grading directly instead of half-grade dummies
    formula_continuous = f"disclosure ~ num_grading + isA_pred1 + isB_pred1 + {' + '.join(extended_controls)} + {yrseason_formula}"
    try:
        model_cont = smf.ols(formula_continuous, data=df_reg).fit(cov_type='HC1')
        result_cont = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/continuous_treatment',
            'spec_tree_path': 'robustness/functional_form.md#treatment-variable-transformations',
            'outcome_var': 'disclosure',
            'treatment_var': 'num_grading',
            'coefficient': float(model_cont.params['num_grading']),
            'std_error': float(model_cont.bse['num_grading']),
            't_stat': float(model_cont.tvalues['num_grading']),
            'p_value': float(model_cont.pvalues['num_grading']),
            'ci_lower': float(model_cont.params['num_grading'] - 1.96 * model_cont.bse['num_grading']),
            'ci_upper': float(model_cont.params['num_grading'] + 1.96 * model_cont.bse['num_grading']),
            'n_obs': int(model_cont.nobs),
            'r_squared': float(model_cont.rsquared),
            'coefficient_vector_json': json.dumps({"treatment": {"var": "num_grading", "coef": float(model_cont.params['num_grading'])}}),
            'sample_desc': 'Last inspection per restaurant, post-disclosure period',
            'fixed_effects': 'yrseason',
            'controls_desc': 'isA_pred1, isB_pred1, firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
            'cluster_var': 'robust',
            'model_type': 'LPM',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result_cont)
        print(f"  Continuous num_grading: coef = {model_cont.params['num_grading']:.4f}")
    except Exception as e:
        print(f"  Continuous treatment failed: {e}")

    # ========================================================================
    # ADDITIONAL ANALYSIS: AM_def1 as treatment (for A- restaurants)
    # ========================================================================
    print("\n--- Running Additional Specifications with A- as treatment ---")

    results.append(extract_results(
        model, 'discrete/binary/lpm_AM', 'methods/discrete_choice.md#model-type-binary-outcome',
        df_reg, 'disclosure', 'AM_def1',
        'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    # B+ as treatment
    results.append(extract_results(
        model, 'discrete/binary/lpm_BP', 'methods/discrete_choice.md#model-type-binary-outcome',
        df_reg, 'disclosure', 'BP_def1',
        'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    # B- as treatment
    results.append(extract_results(
        model, 'discrete/binary/lpm_BM', 'methods/discrete_choice.md#model-type-binary-outcome',
        df_reg, 'disclosure', 'BM_def1',
        'firstbatch, ischain, inyelp, mean_num_grading, sd_num_grading, yrseason FE',
        'yrseason', 'robust', 'LPM'
    ))

    print("\n" + "="*60)
    print(f"Total specifications run: {len(results)}")
    print("="*60)

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    results = run_specifications()

    # Save results to CSV
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_PATH + "specification_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total specifications: {len(df_results)}")
    print(f"Positive coefficients: {(df_results['coefficient'] > 0).sum()} ({100*(df_results['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(df_results['p_value'] < 0.05).sum()} ({100*(df_results['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(df_results['p_value'] < 0.01).sum()} ({100*(df_results['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {df_results['coefficient'].median():.4f}")
    print(f"Mean coefficient: {df_results['coefficient'].mean():.4f}")
    print(f"Coefficient range: [{df_results['coefficient'].min():.4f}, {df_results['coefficient'].max():.4f}]")
