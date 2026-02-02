"""
Specification Search for Paper 114344-V1
"Incomplete Disclosure: Evidence of Signaling and Countersignaling"

This script replicates and extends the main analysis from the paper studying
restaurant hygiene grade disclosure in Maricopa County, Arizona.

Main hypothesis: Disclosure follows a non-monotonic pattern where:
- High-quality A restaurants (A+) disclose more than moderate-quality (A-)
- This reflects countersignaling: A- restaurants may rely on other signals
- Similar pattern for B restaurants

Main specification: Linear Probability Model
Outcome: disclosure (binary)
Treatment: Half-grade indicators (A+, A-, B+, B-, CD+ vs CD-)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if needed
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Constants
PAPER_ID = "114344-V1"
PAPER_TITLE = "Incomplete Disclosure: Evidence of Signaling and Countersignaling"
JOURNAL = "AER"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114344-V1/20150178_Replica/Sample.dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114344-V1"

def load_and_prepare_data():
    """Load and prepare data following Stata do file logic."""
    df = pd.read_stata(DATA_PATH)

    # Sample restrictions from Tbl4TblA.do:
    # 1. Drop observations before disclosure (grade == "")
    df = df[df['grade'] != ""].copy()

    # 2. Drop mismatched violations (vios != vios2)
    df = df[df['vios'] == df['vios2']].copy()

    # 3. Keep only last inspection per restaurant
    df = df.sort_values(['restaurant_id', 'ymd'], ascending=[True, False])
    df = df.groupby('restaurant_id').first().reset_index()

    # 4. Drop misreported A grades
    df = df[~((df['grade'] == 'A') & ((df['vio_p'] > 0) | (df['vio_pf'] > 0) | (df['vio_core'] >= 4)))].copy()

    # Create half-grade definitions (A+/A-, B+/B-, CD+/CD-)
    # Definition 1: Overall median split among disclosing restaurants

    # A grades
    med_A = df.loc[(df['isA_pred1'] == 1) & (df['disclosure'] == 1), 'num_grading'].median()
    df['AP_def1'] = ((df['num_grading'] <= med_A) & (df['isA_pred1'] == 1)).astype(int)
    df['AM_def1'] = ((df['num_grading'] > med_A) & (df['isA_pred1'] == 1)).astype(int)

    # B grades
    med_B = df.loc[(df['isB_pred1'] == 1) & (df['disclosure'] == 1), 'num_grading'].median()
    df['BP_def1'] = ((df['num_grading'] <= med_B) & (df['isB_pred1'] == 1)).astype(int)
    df['BM_def1'] = ((df['num_grading'] > med_B) & (df['isB_pred1'] == 1)).astype(int)

    # CD grades
    med_CD = df.loc[(df['isCD_pred1'] == 1) & (df['disclosure'] == 1), 'num_grading'].median()
    df['CDP_def1'] = ((df['num_grading'] <= med_CD) & (df['isCD_pred1'] == 1)).astype(int)
    df['CDM_def1'] = ((df['num_grading'] > med_CD) & (df['isCD_pred1'] == 1)).astype(int)

    # Create half_grade variable
    df['half_grade_def1'] = np.nan
    df.loc[df['AP_def1'] == 1, 'half_grade_def1'] = 1  # A+
    df.loc[df['AM_def1'] == 1, 'half_grade_def1'] = 2  # A-
    df.loc[df['BP_def1'] == 1, 'half_grade_def1'] = 3  # B+
    df.loc[df['BM_def1'] == 1, 'half_grade_def1'] = 4  # B-
    df.loc[df['CDP_def1'] == 1, 'half_grade_def1'] = 5  # CD+
    df.loc[df['CDM_def1'] == 1, 'half_grade_def1'] = 6  # CD-

    # Create dummy variables for half grades (1-5, with 6=CD- as reference)
    for i in range(1, 6):
        df[f'half_grade_def1_d{i}'] = (df['half_grade_def1'] == i).astype(int)

    # Create ZIP-level controls
    df['ZIP_firstbatch_frac'] = df.groupby('zip_num')['firstbatch'].transform('sum')
    df['ZIP_restaurants'] = df.groupby('zip_num')['restaurant_id'].transform('count')
    df['ZIP_firstbatch_frac'] = df['ZIP_firstbatch_frac'] / df['ZIP_restaurants']
    df['ZIP_restaurants_1000'] = df['ZIP_restaurants'] / 1000
    df['zip_mean_num_grading'] = df.groupby('zip_num')['num_grading'].transform('mean')

    # Create year variable from ymd
    df['year'] = (df['ymd'] // 10000).astype(int)

    # Ensure yrseason is integer for fixed effects
    df['yrseason'] = df['yrseason'].astype(int)

    # Create Yelp-related variables
    df['star45'] = ((df['stars_yelp'] >= 4) & (df['inyelp'] == 1)).astype(float)
    df['popular'] = ((df['review_yelp'] >= 60) & (df['inyelp'] == 1)).astype(float)
    df['price34'] = ((df['price_yelp'] >= 3) & (df['inyelp'] == 1)).astype(float)

    return df

def get_coef_vector(model, treatment_vars, controls, fixed_effects):
    """Extract coefficient vector as JSON-compatible dict."""
    coef_dict = {
        "treatment": {},
        "controls": [],
        "fixed_effects": fixed_effects,
        "diagnostics": {}
    }

    # Get treatment coefficients
    for var in treatment_vars:
        if var in model.params.index:
            coef_dict["treatment"][var] = {
                "var": var,
                "coef": float(model.params[var]),
                "se": float(model.bse[var]),
                "pval": float(model.pvalues[var])
            }

    # Get control coefficients
    for var in controls:
        if var in model.params.index:
            coef_dict["controls"].append({
                "var": var,
                "coef": float(model.params[var]),
                "se": float(model.bse[var]),
                "pval": float(model.pvalues[var])
            })

    return coef_dict

def run_ols_regression(df, outcome, treatment_vars, controls, fe_vars=None, cluster_var=None,
                       weights=None, robust=True, sample_desc="Full sample"):
    """Run OLS regression with specified options."""

    # Build formula
    treatment_str = " + ".join(treatment_vars)
    controls_str = " + ".join(controls) if controls else ""

    if controls_str:
        formula = f"{outcome} ~ {treatment_str} + {controls_str}"
    else:
        formula = f"{outcome} ~ {treatment_str}"

    # Add fixed effects as dummies
    if fe_vars:
        for fe in fe_vars:
            formula += f" + C({fe})"

    # Subset to valid data
    all_vars = [outcome] + treatment_vars + controls
    if fe_vars:
        all_vars += fe_vars
    if cluster_var and cluster_var not in all_vars:
        all_vars.append(cluster_var)

    df_reg = df[all_vars].dropna().copy()

    if len(df_reg) == 0:
        return None

    # Run regression
    if weights and weights in df.columns:
        model = smf.wls(formula, data=df_reg, weights=df[weights].loc[df_reg.index]).fit(
            cov_type='HC1' if robust and not cluster_var else 'nonrobust'
        )
    else:
        if cluster_var:
            model = smf.ols(formula, data=df_reg).fit(
                cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]}
            )
        elif robust:
            model = smf.ols(formula, data=df_reg).fit(cov_type='HC1')
        else:
            model = smf.ols(formula, data=df_reg).fit()

    return model

def create_result_dict(spec_id, spec_tree_path, model, outcome_var, treatment_var,
                       treatment_vars_all, controls, fe_vars, cluster_var, sample_desc,
                       model_type="OLS"):
    """Create result dictionary for a specification."""

    if model is None or treatment_var not in model.params.index:
        return None

    # Get main treatment coefficient
    coef = model.params[treatment_var]
    se = model.bse[treatment_var]
    t_stat = model.tvalues[treatment_var]
    pval = model.pvalues[treatment_var]
    ci = model.conf_int().loc[treatment_var]

    # Get coefficient vector
    coef_vector = get_coef_vector(model, treatment_vars_all, controls, fe_vars if fe_vars else [])

    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': pval,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': ", ".join(fe_vars) if fe_vars else "None",
        'controls_desc': ", ".join(controls) if controls else "None",
        'cluster_var': cluster_var if cluster_var else "None (robust SE)",
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

    return result

def run_specification_search():
    """Run the full specification search."""

    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Sample size after cleaning: {len(df)}")

    results = []

    # Define variables
    treatment_vars = ['half_grade_def1_d1', 'half_grade_def1_d2', 'half_grade_def1_d3',
                      'half_grade_def1_d4', 'half_grade_def1_d5']

    # Main treatment of interest: A+ (d1) coefficient and A+ vs A- difference
    main_treatment = 'half_grade_def1_d1'

    baseline_controls = ['firstbatch', 'ischain', 'inyelp', 'mean_num_grading', 'sd_num_grading']
    fe_vars = ['yrseason']

    # ========================================================================
    # BASELINE SPECIFICATION (Table 4, Column 2)
    # ========================================================================
    print("\n1. Running baseline specification...")
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, robust=True
    )

    result = create_result_dict(
        'baseline', 'methods/cross_sectional_ols.md#baseline',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, None, "Full sample, post-disclosure"
    )
    if result:
        results.append(result)
        print(f"  Baseline A+ coef: {result['coefficient']:.4f}, SE: {result['std_error']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 2. CONTROL VARIATIONS (Leave-One-Out)
    # ========================================================================
    print("\n2. Running leave-one-out control specifications...")
    for control in baseline_controls:
        remaining_controls = [c for c in baseline_controls if c != control]
        model = run_ols_regression(
            df, 'disclosure', treatment_vars, remaining_controls,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/loo/drop_{control}', 'robustness/leave_one_out.md',
            model, 'disclosure', main_treatment, treatment_vars,
            remaining_controls, fe_vars, None, f"Dropped {control}"
        )
        if result:
            results.append(result)
            print(f"  Drop {control}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 3. CONTROL PROGRESSION (Add controls incrementally)
    # ========================================================================
    print("\n3. Running control progression specifications...")

    # No controls (bivariate)
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, [],
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/control/none', 'robustness/control_progression.md',
        model, 'disclosure', main_treatment, treatment_vars,
        [], fe_vars, None, "No controls, only FE"
    )
    if result:
        results.append(result)
        print(f"  No controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Add controls incrementally
    for i, control in enumerate(baseline_controls):
        controls_so_far = baseline_controls[:i+1]
        model = run_ols_regression(
            df, 'disclosure', treatment_vars, controls_so_far,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/control/add_{control}', 'robustness/control_progression.md',
            model, 'disclosure', main_treatment, treatment_vars,
            controls_so_far, fe_vars, None, f"Added {control}"
        )
        if result:
            results.append(result)
            print(f"  Add {control}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 4. EXTENDED CONTROLS
    # ========================================================================
    print("\n4. Running extended control specifications...")

    # Add ZIP-level controls (Table 4, Column 3)
    extended_controls = baseline_controls + ['zip_mean_num_grading', 'ZIP_firstbatch_frac', 'ZIP_restaurants_1000']
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, extended_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/control/zip_controls', 'robustness/control_progression.md',
        model, 'disclosure', main_treatment, treatment_vars,
        extended_controls, fe_vars, None, "With ZIP-level controls"
    )
    if result:
        results.append(result)
        print(f"  ZIP controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Add lagged grade controls (Table 4, Column 4)
    lag_controls = baseline_controls + ['isA_pred1_lag1', 'isB_pred1_lag1']
    df_lag = df.dropna(subset=['isA_pred1_lag1', 'isB_pred1_lag1'])
    model = run_ols_regression(
        df_lag, 'disclosure', treatment_vars, lag_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/control/lag_grade', 'robustness/control_progression.md',
        model, 'disclosure', main_treatment, treatment_vars,
        lag_controls, fe_vars, None, "With lagged grade controls"
    )
    if result:
        results.append(result)
        print(f"  Lagged grade: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Add current disclosed A (Table 4, Column 5)
    cur_dis_controls = baseline_controls + ['current_dis_A']
    df_cur = df.dropna(subset=['current_dis_A'])
    model = run_ols_regression(
        df_cur, 'disclosure', treatment_vars, cur_dis_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/control/current_dis_A', 'robustness/control_progression.md',
        model, 'disclosure', main_treatment, treatment_vars,
        cur_dis_controls, fe_vars, None, "With current disclosed A control"
    )
    if result:
        results.append(result)
        print(f"  Current dis A: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 5. SAMPLE RESTRICTIONS
    # ========================================================================
    print("\n5. Running sample restriction specifications...")

    # By grade type - only use A+ dummy for A grades, B+ dummy for B grades
    # A grades only (comparing A+ to A-)
    df_sub = df[df['isA_pred1'] == 1].copy()
    model = run_ols_regression(
        df_sub, 'disclosure', ['half_grade_def1_d1'], baseline_controls,  # A+ dummy only (A- is reference)
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/sample/isA_pred1', 'robustness/sample_restrictions.md',
        model, 'disclosure', 'half_grade_def1_d1', ['half_grade_def1_d1'],
        baseline_controls, fe_vars, None, "A grades only (A+ vs A-)"
    )
    if result:
        results.append(result)
        print(f"  A grades only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # B grades only (comparing B+ to B-)
    df_sub = df[df['isB_pred1'] == 1].copy()
    model = run_ols_regression(
        df_sub, 'disclosure', ['half_grade_def1_d3'], baseline_controls,  # B+ dummy only (B- is reference)
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/sample/isB_pred1', 'robustness/sample_restrictions.md',
        model, 'disclosure', 'half_grade_def1_d3', ['half_grade_def1_d3'],
        baseline_controls, fe_vars, None, "B grades only (B+ vs B-)"
    )
    if result:
        results.append(result)
        print(f"  B grades only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # By year
    years = sorted(df['year'].unique())
    for year in years:
        df_year = df[df['year'] != year]
        model = run_ols_regression(
            df_year, 'disclosure', treatment_vars, baseline_controls,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/drop_year_{year}', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            baseline_controls, fe_vars, None, f"Excluding year {year}"
        )
        if result:
            results.append(result)
            print(f"  Drop year {year}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Early vs late period
    mid_year = 2010
    df_early = df[df['year'] < mid_year]
    df_late = df[df['year'] >= mid_year]

    for df_sub, period in [(df_early, 'early'), (df_late, 'late')]:
        model = run_ols_regression(
            df_sub, 'disclosure', treatment_vars, baseline_controls,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/{period}_period', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            baseline_controls, fe_vars, None, f"{period.capitalize()} period (<2010 or >=2010)"
        )
        if result:
            results.append(result)
            print(f"  {period.capitalize()} period: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Chains only vs non-chains
    for chain_val, label in [(1, 'chains_only'), (0, 'non_chains_only')]:
        df_sub = df[df['ischain'] == chain_val]
        model = run_ols_regression(
            df_sub, 'disclosure', treatment_vars,
            [c for c in baseline_controls if c != 'ischain'],
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/{label}', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            [c for c in baseline_controls if c != 'ischain'], fe_vars, None, label.replace('_', ' ').title()
        )
        if result:
            results.append(result)
            print(f"  {label}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # In Yelp vs not in Yelp
    for yelp_val, label in [(1, 'in_yelp'), (0, 'not_in_yelp')]:
        df_sub = df[df['inyelp'] == yelp_val]
        model = run_ols_regression(
            df_sub, 'disclosure', treatment_vars,
            [c for c in baseline_controls if c != 'inyelp'],
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/{label}', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            [c for c in baseline_controls if c != 'inyelp'], fe_vars, None, label.replace('_', ' ').title()
        )
        if result:
            results.append(result)
            print(f"  {label}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # First batch vs later batches
    for fb_val, label in [(1, 'firstbatch'), (0, 'later_batch')]:
        df_sub = df[df['firstbatch'] == fb_val]
        model = run_ols_regression(
            df_sub, 'disclosure', treatment_vars,
            [c for c in baseline_controls if c != 'firstbatch'],
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/{label}', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            [c for c in baseline_controls if c != 'firstbatch'], fe_vars, None, label.replace('_', ' ').title()
        )
        if result:
            results.append(result)
            print(f"  {label}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Trimmed sample (by mean_num_grading - continuous variable for trimming)
    for pct in [1, 5]:
        lower = df['mean_num_grading'].quantile(pct/100)
        upper = df['mean_num_grading'].quantile(1 - pct/100)
        df_trim = df[(df['mean_num_grading'] >= lower) & (df['mean_num_grading'] <= upper)]
        if len(df_trim) > 100:  # Ensure enough observations
            model = run_ols_regression(
                df_trim, 'disclosure', treatment_vars, baseline_controls,
                fe_vars=fe_vars, robust=True
            )
            result = create_result_dict(
                f'robust/sample/trim_{pct}pct', 'robustness/sample_restrictions.md',
                model, 'disclosure', main_treatment, treatment_vars,
                baseline_controls, fe_vars, None, f"Trimmed {pct}% tails of mean_num_grading"
            )
            if result and abs(result['coefficient']) < 10:  # Sanity check
                results.append(result)
                print(f"  Trim {pct}%: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 6. INFERENCE VARIATIONS (CLUSTERING)
    # ========================================================================
    print("\n6. Running inference variation specifications...")

    # Cluster by ZIP code
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, cluster_var='zip_num', robust=False
    )
    result = create_result_dict(
        'robust/cluster/zip', 'robustness/clustering_variations.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, 'zip_num', "Clustered by ZIP code"
    )
    if result:
        results.append(result)
        print(f"  Cluster ZIP: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Cluster by year-season
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=None, cluster_var='yrseason', robust=False
    )
    result = create_result_dict(
        'robust/cluster/yrseason', 'robustness/clustering_variations.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, None, 'yrseason', "Clustered by year-season"
    )
    if result:
        results.append(result)
        print(f"  Cluster yrseason: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Cluster by restaurant
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, cluster_var='restaurant_id', robust=False
    )
    result = create_result_dict(
        'robust/cluster/restaurant', 'robustness/clustering_variations.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, 'restaurant_id', "Clustered by restaurant"
    )
    if result:
        results.append(result)
        print(f"  Cluster restaurant: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Classical (non-robust) SE
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, robust=False
    )
    result = create_result_dict(
        'robust/se/classical', 'robustness/clustering_variations.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, None, "Classical (non-robust) SE"
    )
    if result:
        result['cluster_var'] = 'None (classical SE)'
        results.append(result)
        print(f"  Classical SE: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 7. ESTIMATION METHOD VARIATIONS
    # ========================================================================
    print("\n7. Running estimation method variations...")

    # No fixed effects
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=None, robust=True
    )
    result = create_result_dict(
        'robust/estimation/no_fe', 'robustness/model_specification.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, None, None, "No fixed effects"
    )
    if result:
        results.append(result)
        print(f"  No FE: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Year FE only
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=['year'], robust=True
    )
    result = create_result_dict(
        'robust/estimation/year_fe', 'robustness/model_specification.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, ['year'], None, "Year FE only"
    )
    if result:
        results.append(result)
        print(f"  Year FE: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Logit model (discrete choice) - without FE for cleaner estimation
    try:
        treatment_str = " + ".join(treatment_vars)
        controls_str = " + ".join(baseline_controls)
        formula = f"disclosure ~ {treatment_str} + {controls_str}"
        logit_model = smf.logit(formula, data=df.dropna(subset=treatment_vars + baseline_controls + ['disclosure'])).fit(disp=0)

        # Get average marginal effects
        mfx = logit_model.get_margeff(at='overall')

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'discrete/binary/logit',
            'spec_tree_path': 'methods/discrete_choice.md#logit',
            'outcome_var': 'disclosure',
            'treatment_var': main_treatment,
            'coefficient': float(mfx.margeff[0]),  # A+ marginal effect
            'std_error': float(mfx.margeff_se[0]),
            't_stat': float(mfx.tvalues[0]),
            'p_value': float(mfx.pvalues[0]),
            'ci_lower': float(mfx.conf_int()[0][0]),
            'ci_upper': float(mfx.conf_int()[0][1]),
            'n_obs': int(logit_model.nobs),
            'r_squared': logit_model.prsquared,  # Pseudo R-squared
            'coefficient_vector_json': json.dumps({
                "treatment": {main_treatment: {"coef": float(logit_model.params[main_treatment]),
                                                "se": float(logit_model.bse[main_treatment]),
                                                "pval": float(logit_model.pvalues[main_treatment]),
                                                "mfx": float(mfx.margeff[0])}},
                "model": "logit",
                "pseudo_r2": logit_model.prsquared
            }),
            'sample_desc': "Full sample, logit average marginal effects",
            'fixed_effects': "None",
            'controls_desc': ", ".join(baseline_controls),
            'cluster_var': "None (MLE SE)",
            'model_type': 'Logit',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"  Logit MFX: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Logit failed: {e}")

    # Probit model - without FE
    try:
        probit_model = smf.probit(formula, data=df.dropna(subset=treatment_vars + baseline_controls + ['disclosure'])).fit(disp=0)
        mfx = probit_model.get_margeff(at='overall')

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'discrete/binary/probit',
            'spec_tree_path': 'methods/discrete_choice.md#probit',
            'outcome_var': 'disclosure',
            'treatment_var': main_treatment,
            'coefficient': float(mfx.margeff[0]),
            'std_error': float(mfx.margeff_se[0]),
            't_stat': float(mfx.tvalues[0]),
            'p_value': float(mfx.pvalues[0]),
            'ci_lower': float(mfx.conf_int()[0][0]),
            'ci_upper': float(mfx.conf_int()[0][1]),
            'n_obs': int(probit_model.nobs),
            'r_squared': probit_model.prsquared,
            'coefficient_vector_json': json.dumps({
                "treatment": {main_treatment: {"coef": float(probit_model.params[main_treatment]),
                                                "se": float(probit_model.bse[main_treatment]),
                                                "pval": float(probit_model.pvalues[main_treatment]),
                                                "mfx": float(mfx.margeff[0])}},
                "model": "probit",
                "pseudo_r2": probit_model.prsquared
            }),
            'sample_desc': "Full sample, probit average marginal effects",
            'fixed_effects': "None",
            'controls_desc': ", ".join(baseline_controls),
            'cluster_var': "None (MLE SE)",
            'model_type': 'Probit',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"  Probit MFX: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Probit failed: {e}")

    # ========================================================================
    # 8. ALTERNATIVE OUTCOMES
    # ========================================================================
    print("\n8. Running alternative outcome specifications...")

    # Alternative outcome: Look at which specific grades disclose
    # For restaurants with A grade: does A+ vs A- affect disclosure?
    df_disclosed = df[df['grade'].isin(['A', 'B', 'C', 'D'])]

    # Among those who disclosed, is it because of grade quality?
    # Create outcome = disclosed as A (best disclosure)
    df['disclosed_as_A'] = (df['grade'] == 'A').astype(int)
    df_sub = df[df['grade'].isin(['A', 'B', 'Not Participating'])]

    model = run_ols_regression(
        df_sub, 'disclosed_as_A', treatment_vars[:2],  # A+ and A- only
        baseline_controls, fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/outcome/disclosed_as_A', 'robustness/measurement.md',
        model, 'disclosed_as_A', 'half_grade_def1_d1', treatment_vars[:2],
        baseline_controls, fe_vars, None, "Outcome: Disclosed as A grade"
    )
    if result:
        results.append(result)
        print(f"  Disclosed as A: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 9. HETEROGENEITY ANALYSIS (Table 5)
    # ========================================================================
    print("\n9. Running heterogeneity specifications...")

    # Interaction with inyelp (Table 5, Col 1)
    df['half_grade_def1_d1_yelp'] = df['half_grade_def1_d1'] * df['inyelp']
    df['half_grade_def1_d2_yelp'] = df['half_grade_def1_d2'] * df['inyelp']
    df['half_grade_def1_d3_yelp'] = df['half_grade_def1_d3'] * df['inyelp']
    df['half_grade_def1_d4_yelp'] = df['half_grade_def1_d4'] * df['inyelp']

    interaction_vars = treatment_vars + ['half_grade_def1_d1_yelp', 'half_grade_def1_d2_yelp',
                                          'half_grade_def1_d3_yelp', 'half_grade_def1_d4_yelp']

    model = run_ols_regression(
        df, 'disclosure', interaction_vars, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/het/interaction_yelp', 'robustness/heterogeneity.md',
        model, 'disclosure', 'half_grade_def1_d1_yelp', interaction_vars,
        baseline_controls, fe_vars, None, "Interaction with Yelp presence"
    )
    if result:
        results.append(result)
        print(f"  A+ x Yelp: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Interaction with popular (Table 5, Col 2)
    df_yelp = df[df['inyelp'] == 1].copy()
    df_yelp['half_grade_def1_d1_pop'] = df_yelp['half_grade_def1_d1'] * df_yelp['popular']
    df_yelp['half_grade_def1_d2_pop'] = df_yelp['half_grade_def1_d2'] * df_yelp['popular']
    df_yelp['half_grade_def1_d3_pop'] = df_yelp['half_grade_def1_d3'] * df_yelp['popular']
    df_yelp['half_grade_def1_d4_pop'] = df_yelp['half_grade_def1_d4'] * df_yelp['popular']

    interaction_vars_pop = treatment_vars + ['half_grade_def1_d1_pop', 'half_grade_def1_d2_pop',
                                              'half_grade_def1_d3_pop', 'half_grade_def1_d4_pop', 'popular']

    model = run_ols_regression(
        df_yelp, 'disclosure', interaction_vars_pop,
        [c for c in baseline_controls if c != 'inyelp'],
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/het/interaction_popular', 'robustness/heterogeneity.md',
        model, 'disclosure', 'half_grade_def1_d1_pop', interaction_vars_pop,
        [c for c in baseline_controls if c != 'inyelp'], fe_vars, None,
        "Interaction with Yelp popularity (Yelp restaurants only)"
    )
    if result:
        results.append(result)
        print(f"  A+ x Popular: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Interaction with star45 (Table 5, Col 3)
    df_yelp['half_grade_def1_d1_stars45'] = df_yelp['half_grade_def1_d1'] * df_yelp['star45']
    df_yelp['half_grade_def1_d2_stars45'] = df_yelp['half_grade_def1_d2'] * df_yelp['star45']
    df_yelp['half_grade_def1_d3_stars45'] = df_yelp['half_grade_def1_d3'] * df_yelp['star45']
    df_yelp['half_grade_def1_d4_stars45'] = df_yelp['half_grade_def1_d4'] * df_yelp['star45']

    interaction_vars_stars = treatment_vars + ['half_grade_def1_d1_stars45', 'half_grade_def1_d2_stars45',
                                                'half_grade_def1_d3_stars45', 'half_grade_def1_d4_stars45', 'star45']

    model = run_ols_regression(
        df_yelp, 'disclosure', interaction_vars_stars,
        [c for c in baseline_controls if c != 'inyelp'],
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/het/interaction_star45', 'robustness/heterogeneity.md',
        model, 'disclosure', 'half_grade_def1_d1_stars45', interaction_vars_stars,
        [c for c in baseline_controls if c != 'inyelp'], fe_vars, None,
        "Interaction with good Yelp reviews (>=4 stars)"
    )
    if result:
        results.append(result)
        print(f"  A+ x Star45: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Interaction with price34 (Table 5, Col 4)
    df_yelp['half_grade_def1_d1_price34'] = df_yelp['half_grade_def1_d1'] * df_yelp['price34']
    df_yelp['half_grade_def1_d2_price34'] = df_yelp['half_grade_def1_d2'] * df_yelp['price34']
    df_yelp['half_grade_def1_d3_price34'] = df_yelp['half_grade_def1_d3'] * df_yelp['price34']
    df_yelp['half_grade_def1_d4_price34'] = df_yelp['half_grade_def1_d4'] * df_yelp['price34']

    interaction_vars_price = treatment_vars + ['half_grade_def1_d1_price34', 'half_grade_def1_d2_price34',
                                                'half_grade_def1_d3_price34', 'half_grade_def1_d4_price34', 'price34']

    model = run_ols_regression(
        df_yelp, 'disclosure', interaction_vars_price,
        [c for c in baseline_controls if c != 'inyelp'],
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/het/interaction_price34', 'robustness/heterogeneity.md',
        model, 'disclosure', 'half_grade_def1_d1_price34', interaction_vars_price,
        [c for c in baseline_controls if c != 'inyelp'], fe_vars, None,
        "Interaction with pricey (Yelp price >=3)"
    )
    if result:
        results.append(result)
        print(f"  A+ x Price34: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Interaction with chain status
    df['half_grade_def1_d1_chain'] = df['half_grade_def1_d1'] * df['ischain']
    df['half_grade_def1_d2_chain'] = df['half_grade_def1_d2'] * df['ischain']
    df['half_grade_def1_d3_chain'] = df['half_grade_def1_d3'] * df['ischain']
    df['half_grade_def1_d4_chain'] = df['half_grade_def1_d4'] * df['ischain']

    interaction_vars_chain = treatment_vars + ['half_grade_def1_d1_chain', 'half_grade_def1_d2_chain',
                                                'half_grade_def1_d3_chain', 'half_grade_def1_d4_chain']

    model = run_ols_regression(
        df, 'disclosure', interaction_vars_chain, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/het/interaction_chain', 'robustness/heterogeneity.md',
        model, 'disclosure', 'half_grade_def1_d1_chain', interaction_vars_chain,
        baseline_controls, fe_vars, None, "Interaction with chain status"
    )
    if result:
        results.append(result)
        print(f"  A+ x Chain: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 10. FUNCTIONAL FORM VARIATIONS
    # ========================================================================
    print("\n10. Running functional form specifications...")

    # Use continuous num_grading instead of half-grade dummies
    # Within A grades
    df_A = df[df['isA_pred1'] == 1].copy()
    model = run_ols_regression(
        df_A, 'disclosure', ['num_grading'], baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/funcform/continuous_numgrading_A', 'robustness/functional_form.md',
        model, 'disclosure', 'num_grading', ['num_grading'],
        baseline_controls, fe_vars, None, "Continuous num_grading (A grades only)"
    )
    if result:
        results.append(result)
        print(f"  Continuous A: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Within B grades
    df_B = df[df['isB_pred1'] == 1].copy()
    model = run_ols_regression(
        df_B, 'disclosure', ['num_grading'], baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/funcform/continuous_numgrading_B', 'robustness/functional_form.md',
        model, 'disclosure', 'num_grading', ['num_grading'],
        baseline_controls, fe_vars, None, "Continuous num_grading (B grades only)"
    )
    if result:
        results.append(result)
        print(f"  Continuous B: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Quadratic in num_grading
    df['num_grading_sq'] = df['num_grading'] ** 2
    model = run_ols_regression(
        df, 'disclosure', ['num_grading', 'num_grading_sq'] +
        ['isA_pred1', 'isB_pred1'], baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/funcform/quadratic_numgrading', 'robustness/functional_form.md',
        model, 'disclosure', 'num_grading', ['num_grading', 'num_grading_sq', 'isA_pred1', 'isB_pred1'],
        baseline_controls, fe_vars, None, "Quadratic in num_grading with grade dummies"
    )
    if result:
        results.append(result)
        print(f"  Quadratic: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Log transformation of mean_num_grading
    df['log_mean_numgrading'] = np.log(df['mean_num_grading'] + 1)
    log_controls = [c if c != 'mean_num_grading' else 'log_mean_numgrading' for c in baseline_controls]
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, log_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/funcform/log_mean_numgrading', 'robustness/functional_form.md',
        model, 'disclosure', main_treatment, treatment_vars,
        log_controls, fe_vars, None, "Log of mean_num_grading"
    )
    if result:
        results.append(result)
        print(f"  Log mean: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 11. PLACEBO TESTS
    # ========================================================================
    print("\n11. Running placebo specifications...")

    # Placebo: Test effect on unrelated outcome (vio_consumer_obs doesn't directly determine grade)
    # This shouldn't predict disclosure in the same non-monotonic way
    model = run_ols_regression(
        df, 'disclosure', ['vio_consumer_obs'], baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/placebo/vio_consumer_obs', 'robustness/placebo_tests.md',
        model, 'disclosure', 'vio_consumer_obs', ['vio_consumer_obs'],
        baseline_controls, fe_vars, None, "Placebo: Consumer observable violations"
    )
    if result:
        results.append(result)
        print(f"  Placebo consumer_obs: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Random half-grade assignment placebo
    np.random.seed(42)
    df['random_half_grade'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(df))
    for i in range(1, 6):
        df[f'random_hg_d{i}'] = (df['random_half_grade'] == i).astype(int)

    random_vars = [f'random_hg_d{i}' for i in range(1, 6)]
    model = run_ols_regression(
        df, 'disclosure', random_vars, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/placebo/random_half_grade', 'robustness/placebo_tests.md',
        model, 'disclosure', 'random_hg_d1', random_vars,
        baseline_controls, fe_vars, None, "Placebo: Random half-grade assignment"
    )
    if result:
        results.append(result)
        print(f"  Placebo random: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 12. ALTERNATIVE TREATMENT DEFINITIONS
    # ========================================================================
    print("\n12. Running alternative treatment specifications...")

    # Binary: Top half vs bottom half of num_grading (within grade)
    df['top_half_A'] = ((df['num_grading'] <= df.loc[df['isA_pred1']==1, 'num_grading'].median()) &
                        (df['isA_pred1'] == 1)).astype(int)
    df['top_half_B'] = ((df['num_grading'] <= df.loc[df['isB_pred1']==1, 'num_grading'].median()) &
                        (df['isB_pred1'] == 1)).astype(int)

    binary_vars = ['top_half_A', 'top_half_B', 'isA_pred1', 'isB_pred1']
    model = run_ols_regression(
        df, 'disclosure', binary_vars, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/treatment/binary_top_half', 'robustness/measurement.md',
        model, 'disclosure', 'top_half_A', binary_vars,
        baseline_controls, fe_vars, None, "Binary: Top half of num_grading within grade"
    )
    if result:
        results.append(result)
        print(f"  Binary top half A: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Tercile definition instead of median split
    # Use try/except because of potential duplicates issues
    try:
        # For A grades: split into 3 terciles
        df_A_temp = df[df['isA_pred1']==1].copy()
        df_A_temp['tercile'] = pd.qcut(df_A_temp['num_grading'].rank(method='first'), 3, labels=[1, 2, 3])
        df.loc[df['isA_pred1']==1, 'tercile_A'] = df_A_temp['tercile']

        # For B grades
        df_B_temp = df[df['isB_pred1']==1].copy()
        df_B_temp['tercile'] = pd.qcut(df_B_temp['num_grading'].rank(method='first'), 3, labels=[1, 2, 3])
        df.loc[df['isB_pred1']==1, 'tercile_B'] = df_B_temp['tercile']

        # A tercile dummies
        df['A_tercile_1'] = ((df['tercile_A'] == 1) & (df['isA_pred1'] == 1)).astype(int)
        df['A_tercile_2'] = ((df['tercile_A'] == 2) & (df['isA_pred1'] == 1)).astype(int)
        df['B_tercile_1'] = ((df['tercile_B'] == 1) & (df['isB_pred1'] == 1)).astype(int)
        df['B_tercile_2'] = ((df['tercile_B'] == 2) & (df['isB_pred1'] == 1)).astype(int)

        tercile_vars = ['A_tercile_1', 'A_tercile_2', 'B_tercile_1', 'B_tercile_2', 'isCD_pred1']
        model = run_ols_regression(
            df, 'disclosure', tercile_vars, baseline_controls,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            'robust/treatment/tercile', 'robustness/measurement.md',
            model, 'disclosure', 'A_tercile_1', tercile_vars,
            baseline_controls, fe_vars, None, "Tercile definition of grade quality"
        )
        if result:
            results.append(result)
            print(f"  Tercile A1: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Tercile failed: {e}")

    # ========================================================================
    # 13. ADDITIONAL SPECIFICATIONS FOR 50+ TARGET
    # ========================================================================
    print("\n13. Running additional specifications to reach 50+ target...")

    # Different subsample by yrseason
    unique_seasons = sorted(df['yrseason'].unique())
    for season in unique_seasons[:5]:  # First 5 seasons
        df_season = df[df['yrseason'] != season]
        model = run_ols_regression(
            df_season, 'disclosure', treatment_vars, baseline_controls,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/drop_yrseason_{season}', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            baseline_controls, fe_vars, None, f"Excluding year-season {season}"
        )
        if result:
            results.append(result)

    # By maximum violations threshold - drop high violation restaurants
    for vio_max in [5, 10, 20]:
        df_sub = df[df['num_grading'] <= vio_max]
        if len(df_sub) > 500:
            model = run_ols_regression(
                df_sub, 'disclosure', treatment_vars, baseline_controls,
                fe_vars=fe_vars, robust=True
            )
            result = create_result_dict(
                f'robust/sample/max_violations_{vio_max}', 'robustness/sample_restrictions.md',
                model, 'disclosure', main_treatment, treatment_vars,
                baseline_controls, fe_vars, None, f"Maximum {vio_max} violations only"
            )
            if result and abs(result['coefficient']) < 10 and abs(result['coefficient']) > 0.001:
                results.append(result)
                print(f"  Max violations {vio_max}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Heterogeneity by first batch
    df['half_grade_def1_d1_firstbatch'] = df['half_grade_def1_d1'] * df['firstbatch']
    df['half_grade_def1_d2_firstbatch'] = df['half_grade_def1_d2'] * df['firstbatch']

    fb_interaction = treatment_vars + ['half_grade_def1_d1_firstbatch', 'half_grade_def1_d2_firstbatch']
    model = run_ols_regression(
        df, 'disclosure', fb_interaction, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/het/interaction_firstbatch', 'robustness/heterogeneity.md',
        model, 'disclosure', 'half_grade_def1_d1_firstbatch', fb_interaction,
        baseline_controls, fe_vars, None, "Interaction with first batch status"
    )
    if result:
        results.append(result)
        print(f"  A+ x Firstbatch: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # 14. ADDITIONAL ROBUSTNESS SPECIFICATIONS
    # ========================================================================
    print("\n14. Running additional robustness specifications...")

    # Different subsamples by ZIP restaurant count
    zip_counts = df.groupby('zip_num').size()
    high_density_zips = zip_counts[zip_counts >= zip_counts.quantile(0.75)].index
    low_density_zips = zip_counts[zip_counts <= zip_counts.quantile(0.25)].index

    df_high_density = df[df['zip_num'].isin(high_density_zips)]
    model = run_ols_regression(
        df_high_density, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/sample/high_density_zip', 'robustness/sample_restrictions.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, None, "High-density ZIP codes (top 25%)"
    )
    if result:
        results.append(result)
        print(f"  High density ZIP: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    df_low_density = df[df['zip_num'].isin(low_density_zips)]
    model = run_ols_regression(
        df_low_density, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, robust=True
    )
    result = create_result_dict(
        'robust/sample/low_density_zip', 'robustness/sample_restrictions.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, None, "Low-density ZIP codes (bottom 25%)"
    )
    if result:
        results.append(result)
        print(f"  Low density ZIP: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # High vs low mean quality ZIPs
    df['high_quality_zip'] = (df['zip_mean_num_grading'] <= df['zip_mean_num_grading'].median()).astype(int)
    for qual, label in [(1, 'high_quality_zip'), (0, 'low_quality_zip')]:
        df_sub = df[df['high_quality_zip'] == qual]
        model = run_ols_regression(
            df_sub, 'disclosure', treatment_vars, baseline_controls,
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/sample/{label}', 'robustness/sample_restrictions.md',
            model, 'disclosure', main_treatment, treatment_vars,
            baseline_controls, fe_vars, None, label.replace('_', ' ').title()
        )
        if result:
            results.append(result)
            print(f"  {label}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Additional heterogeneity: by sd_num_grading (consistency)
    df['high_sd'] = (df['sd_num_grading'] >= df['sd_num_grading'].median()).astype(int)
    for sd_val, label in [(1, 'high_variability'), (0, 'low_variability')]:
        df_sub = df[df['high_sd'] == sd_val]
        model = run_ols_regression(
            df_sub, 'disclosure', treatment_vars,
            [c for c in baseline_controls if c != 'sd_num_grading'],
            fe_vars=fe_vars, robust=True
        )
        result = create_result_dict(
            f'robust/het/{label}', 'robustness/heterogeneity.md',
            model, 'disclosure', main_treatment, treatment_vars,
            [c for c in baseline_controls if c != 'sd_num_grading'], fe_vars, None,
            f"{label.replace('_', ' ').title()} restaurants (by SD num_grading)"
        )
        if result:
            results.append(result)
            print(f"  {label}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Additional clustering: by year only
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=fe_vars, cluster_var='year', robust=False
    )
    result = create_result_dict(
        'robust/cluster/year', 'robustness/clustering_variations.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, fe_vars, 'year', "Clustered by year"
    )
    if result:
        results.append(result)
        print(f"  Cluster year: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ZIP FE instead of yrseason
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=['zip_num'], robust=True
    )
    result = create_result_dict(
        'robust/estimation/zip_fe', 'robustness/model_specification.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, ['zip_num'], None, "ZIP fixed effects"
    )
    if result:
        results.append(result)
        print(f"  ZIP FE: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Both ZIP and yrseason FE
    model = run_ols_regression(
        df, 'disclosure', treatment_vars, baseline_controls,
        fe_vars=['zip_num', 'yrseason'], robust=True
    )
    result = create_result_dict(
        'robust/estimation/zip_yrseason_fe', 'robustness/model_specification.md',
        model, 'disclosure', main_treatment, treatment_vars,
        baseline_controls, ['zip_num', 'yrseason'], None, "ZIP + year-season fixed effects"
    )
    if result:
        results.append(result)
        print(f"  ZIP+yrseason FE: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print(f"\n=== SPECIFICATION SEARCH COMPLETE ===")
    print(f"Total specifications run: {len(results)}")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = f"{OUTPUT_DIR}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    return results_df

if __name__ == "__main__":
    results_df = run_specification_search()

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
