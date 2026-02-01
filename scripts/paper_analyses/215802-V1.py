"""
Specification Search: Paper 215802-V1
"Long-Run Impacts of Childhood Access to the Safety Net"
Hoynes, Schanzenbach, Almond (AER 2016)

This script performs a systematic specification search on the paper's analysis
of the long-term effects of childhood Food Stamp Program (FSP) exposure on
adult health outcomes using PSID data.

Method: Difference-in-Differences (staggered rollout across counties/cohorts)
Method Tree Path: specification_tree/methods/difference_in_differences.md
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "215802-V1"
JOURNAL = "AER"
PAPER_TITLE = "Long-Run Impacts of Childhood Access to the Safety Net"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/215802-V1/psidAdultHealth.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/215802-V1/"

# Treatment variable
TREATMENT_VAR = "group_fspart_1"  # Share of childhood with FSP access

# Health outcome variables (coded as 1=condition, 5=no condition in raw data)
HEALTH_OUTCOMES = {
    'diabe': 'diabetes',
    'blood': 'high_blood_pressure',
    'heatt': 'heart_attack',
    'hedis': 'heart_disease',
    'strok': 'stroke',
    'arthr': 'arthritis',
    'asthm': 'asthma',
    'cance': 'cancer',
    'lungd': 'lung_disease'
}

# Controls
BASIC_CONTROLS = ['age', 'age_sq']
FULL_CONTROLS = ['age', 'age_sq', 'educ', 'famsize']

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load PSID data and prepare variables for analysis."""
    df = pd.read_stata(DATA_PATH)

    # Create person identifier (as integer for proper grouping)
    df['person_id'] = df['inum1968'] * 1000 + df['person1968']

    # Create binary health outcomes (1 = has condition, 0 = no condition)
    for raw_var, clean_name in HEALTH_OUTCOMES.items():
        df[clean_name] = np.where(df[raw_var] == 1, 1,
                                  np.where(df[raw_var] == 5, 0, np.nan))

    # Create metabolic syndrome composite (obesity proxy via high BP, diabetes)
    df['metabolic_syndrome'] = ((df['diabetes'] == 1) | (df['high_blood_pressure'] == 1)).astype(float)
    df.loc[(df['diabetes'].isna()) & (df['high_blood_pressure'].isna()), 'metabolic_syndrome'] = np.nan

    # Create cardiovascular composite
    df['cardiovascular'] = ((df['heart_attack'] == 1) | (df['heart_disease'] == 1) | (df['stroke'] == 1)).astype(float)
    df.loc[(df['heart_attack'].isna()) & (df['heart_disease'].isna()) & (df['stroke'].isna()), 'cardiovascular'] = np.nan

    # Create age squared
    df['age_sq'] = df['age'].astype(float) ** 2

    # Create numeric sex variable (Female=1, Male=0)
    df['female'] = (df['sex'] == 'Female').astype(int)

    # Create race dummies
    df['black'] = (df['race'] == 'Black').astype(int)
    df['white'] = (df['race'] == 'White').astype(int)

    # Convert numeric columns to float to avoid dtype issues
    for col in ['age', 'educ', 'famsize', 'yob', 'Datayear']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    # Convert treatment to float
    df[TREATMENT_VAR] = pd.to_numeric(df[TREATMENT_VAR], errors='coerce').astype(float)

    return df

# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_ols_with_fe(df, outcome_var, treatment_var, controls=None, fe_vars=None, cluster_var=None, use_robust=True):
    """
    Run OLS regression with optional fixed effects and clustering.
    Returns dictionary with results.
    """
    # Prepare data - start fresh copy
    analysis_df = df.copy()

    # Collect required columns
    required_cols = [outcome_var, treatment_var]
    if controls:
        required_cols.extend(controls)
    if fe_vars:
        required_cols.extend(fe_vars)
    if cluster_var:
        required_cols.append(cluster_var)

    # Drop rows with missing values in required columns
    for col in required_cols:
        if col in analysis_df.columns:
            analysis_df = analysis_df[analysis_df[col].notna()]

    if len(analysis_df) < 100:
        return {
            "coefficient": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": len(analysis_df),
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": "Insufficient observations"})
        }

    # Build X matrix
    X_vars = [treatment_var]
    if controls:
        X_vars.extend(controls)

    # Add fixed effects as dummies
    fe_absorbed = []
    if fe_vars:
        for fe_var in fe_vars:
            if fe_var in analysis_df.columns:
                # Use pd.get_dummies for cleaner FE handling
                unique_vals = analysis_df[fe_var].nunique()
                if unique_vals < 500:  # Only add FE for reasonable number of groups
                    fe_dummies = pd.get_dummies(analysis_df[fe_var], prefix=fe_var, drop_first=True, dtype=float)
                    # Add to analysis_df
                    for col in fe_dummies.columns:
                        analysis_df[col] = fe_dummies[col].values
                        X_vars.append(col)
                    fe_absorbed.append(fe_var)

    # Create X matrix as numpy array (ensures proper dtype)
    X = analysis_df[X_vars].values.astype(float)
    X = np.column_stack([np.ones(len(analysis_df)), X])  # Add constant
    y = analysis_df[outcome_var].values.astype(float)

    # Variable names including constant
    var_names = ['const'] + X_vars

    # Fit model
    try:
        model = sm.OLS(y, X).fit()

        # Apply robust/clustered standard errors
        if cluster_var and cluster_var in analysis_df.columns:
            try:
                groups = analysis_df[cluster_var].values
                model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})
            except Exception:
                # Fall back to robust SE if clustering fails
                model = sm.OLS(y, X).fit(cov_type='HC1')
        elif use_robust:
            model = sm.OLS(y, X).fit(cov_type='HC1')

        # Extract treatment coefficient (index 1 after constant)
        treat_idx = 1  # Treatment is first variable after constant
        coef = model.params[treat_idx]
        se = model.bse[treat_idx]
        tstat = model.tvalues[treat_idx]
        pval = model.pvalues[treat_idx]
        ci = model.conf_int()[treat_idx]

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": []
        }

        if controls:
            for i, c in enumerate(controls):
                idx = 2 + i  # After constant and treatment
                if idx < len(model.params):
                    coef_vector["controls"].append({
                        "var": c,
                        "coef": float(model.params[idx]),
                        "se": float(model.bse[idx]),
                        "pval": float(model.pvalues[idx])
                    })

        coef_vector["fixed_effects_absorbed"] = fe_absorbed
        coef_vector["diagnostics"] = {
            "first_stage_F": None,
            "pretrend_pval": None
        }
        coef_vector["n_obs"] = int(model.nobs)
        coef_vector["r_squared"] = float(model.rsquared)

        return {
            "coefficient": float(coef),
            "std_error": float(se),
            "t_stat": float(tstat),
            "p_value": float(pval),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "coefficient_vector_json": json.dumps(coef_vector)
        }

    except Exception as e:
        return {
            "coefficient": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": 0,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": str(e)})
        }

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

def run_specification_search(df):
    """Run all specifications according to the specification tree."""
    results = []

    # Primary outcome (metabolic syndrome as in paper)
    primary_outcome = 'metabolic_syndrome'

    # All health outcomes for robustness
    all_outcomes = ['metabolic_syndrome', 'cardiovascular', 'diabetes', 'high_blood_pressure',
                    'heart_attack', 'heart_disease', 'stroke', 'arthritis', 'asthma', 'cancer', 'lung_disease']

    # ========================================================================
    # BASELINE SPECIFICATION
    # ========================================================================
    print("Running baseline specification...")

    baseline_result = run_ols_with_fe(
        df,
        outcome_var=primary_outcome,
        treatment_var=TREATMENT_VAR,
        controls=FULL_CONTROLS,
        fe_vars=['yob', 'Datayear'],
        cluster_var='person_id'
    )

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'baseline',
        'spec_tree_path': 'methods/difference_in_differences.md#baseline',
        'outcome_var': primary_outcome,
        'treatment_var': TREATMENT_VAR,
        'sample_desc': 'Full sample with health outcomes',
        'fixed_effects': 'Birth year + Survey year',
        'controls_desc': ', '.join(FULL_CONTROLS),
        'cluster_var': 'person_id',
        'model_type': 'OLS with FE',
        'estimation_script': 'scripts/paper_analyses/215802-V1.py',
        **baseline_result
    })

    # ========================================================================
    # FIXED EFFECTS VARIATIONS
    # ========================================================================
    print("Running fixed effects variations...")

    fe_specs = [
        ('did/fe/none', None, 'No fixed effects'),
        ('did/fe/yob_only', ['yob'], 'Birth year only'),
        ('did/fe/year_only', ['Datayear'], 'Survey year only'),
        ('did/fe/twoway', ['yob', 'Datayear'], 'Birth year + Survey year'),
    ]

    for spec_id, fe_vars, fe_desc in fe_specs:
        result = run_ols_with_fe(
            df,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=fe_vars,
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': 'methods/difference_in_differences.md#fixed-effects',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': 'Full sample with health outcomes',
            'fixed_effects': fe_desc,
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # ========================================================================
    # CONTROL SET VARIATIONS
    # ========================================================================
    print("Running control set variations...")

    control_specs = [
        ('did/controls/none', [], 'No controls'),
        ('did/controls/minimal', ['age', 'age_sq'], 'Age only'),
        ('did/controls/baseline', FULL_CONTROLS, 'Baseline controls'),
        ('did/controls/plus_female', FULL_CONTROLS + ['female'], 'Baseline + female'),
        ('did/controls/plus_race', FULL_CONTROLS + ['black'], 'Baseline + black'),
        ('did/controls/full', FULL_CONTROLS + ['female', 'black'], 'All controls'),
    ]

    for spec_id, controls, controls_desc in control_specs:
        result = run_ols_with_fe(
            df,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=controls if controls else None,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': 'methods/difference_in_differences.md#control-sets',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': 'Full sample with health outcomes',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': controls_desc,
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # ========================================================================
    # LEAVE-ONE-OUT ROBUSTNESS
    # ========================================================================
    print("Running leave-one-out robustness checks...")

    for dropped_var in FULL_CONTROLS:
        remaining = [c for c in FULL_CONTROLS if c != dropped_var]

        result = run_ols_with_fe(
            df,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=remaining,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/loo/drop_{dropped_var}',
            'spec_tree_path': 'robustness/leave_one_out.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': 'Full sample with health outcomes',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': f'Dropped: {dropped_var}',
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # ========================================================================
    # SINGLE COVARIATE ROBUSTNESS
    # ========================================================================
    print("Running single covariate robustness checks...")

    # Bivariate (no controls)
    result = run_ols_with_fe(
        df,
        outcome_var=primary_outcome,
        treatment_var=TREATMENT_VAR,
        controls=None,
        fe_vars=['yob', 'Datayear'],
        cluster_var='person_id'
    )

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/single/none',
        'spec_tree_path': 'robustness/single_covariate.md',
        'outcome_var': primary_outcome,
        'treatment_var': TREATMENT_VAR,
        'sample_desc': 'Full sample with health outcomes',
        'fixed_effects': 'Birth year + Survey year',
        'controls_desc': 'None',
        'cluster_var': 'person_id',
        'model_type': 'OLS with FE',
        'estimation_script': 'scripts/paper_analyses/215802-V1.py',
        **result
    })

    # Single covariate at a time
    for control in FULL_CONTROLS:
        result = run_ols_with_fe(
            df,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=[control],
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/single/{control}',
            'spec_tree_path': 'robustness/single_covariate.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': 'Full sample with health outcomes',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': control,
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # ========================================================================
    # CLUSTERING VARIATIONS
    # ========================================================================
    print("Running clustering variations...")

    cluster_specs = [
        ('robust/cluster/none', None, 'Robust (no clustering)'),
        ('robust/cluster/person', 'person_id', 'Person'),
        ('robust/cluster/yob', 'yob', 'Birth year'),
        ('robust/cluster/year', 'Datayear', 'Survey year'),
    ]

    for spec_id, cluster_var, cluster_desc in cluster_specs:
        result = run_ols_with_fe(
            df,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=['yob', 'Datayear'],
            cluster_var=cluster_var
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': 'robustness/clustering_variations.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': 'Full sample with health outcomes',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': cluster_desc,
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # ========================================================================
    # SAMPLE RESTRICTIONS
    # ========================================================================
    print("Running sample restriction specifications...")

    # By gender
    for gender, gender_label in [('Male', 'male'), ('Female', 'female')]:
        df_sub = df[df['sex'] == gender].copy()

        result = run_ols_with_fe(
            df_sub,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/sample/{gender_label}_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': f'{gender} only',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # By race
    for race in ['White', 'Black']:
        df_sub = df[df['race'] == race].copy()

        result = run_ols_with_fe(
            df_sub,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/sample/{race.lower()}_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': f'{race} only',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # Early vs late birth cohorts
    median_yob = df['yob'].median()

    for period, condition, label in [
        ('early_cohort', df['yob'] < median_yob, f'Birth year < {int(median_yob)}'),
        ('late_cohort', df['yob'] >= median_yob, f'Birth year >= {int(median_yob)}')
    ]:
        df_sub = df[condition].copy()

        result = run_ols_with_fe(
            df_sub,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/sample/{period}',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': label,
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # Trim outliers
    for trim_pct in [0.01, 0.05]:
        lower = df[TREATMENT_VAR].quantile(trim_pct)
        upper = df[TREATMENT_VAR].quantile(1 - trim_pct)
        df_sub = df[(df[TREATMENT_VAR] >= lower) & (df[TREATMENT_VAR] <= upper)].copy()

        result = run_ols_with_fe(
            df_sub,
            outcome_var=primary_outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/sample/trim_{int(trim_pct*100)}pct',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': primary_outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': f'Trim {int(trim_pct*100)}% tails of treatment',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    # ========================================================================
    # ALTERNATIVE OUTCOMES
    # ========================================================================
    print("Running alternative outcome specifications...")

    for outcome in all_outcomes:
        if outcome == primary_outcome:
            continue

        result = run_ols_with_fe(
            df,
            outcome_var=outcome,
            treatment_var=TREATMENT_VAR,
            controls=FULL_CONTROLS,
            fe_vars=['yob', 'Datayear'],
            cluster_var='person_id'
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'did/outcome/{outcome}',
            'spec_tree_path': 'methods/difference_in_differences.md#outcomes',
            'outcome_var': outcome,
            'treatment_var': TREATMENT_VAR,
            'sample_desc': 'Full sample with health outcomes',
            'fixed_effects': 'Birth year + Survey year',
            'controls_desc': ', '.join(FULL_CONTROLS),
            'cluster_var': 'person_id',
            'model_type': 'OLS with FE',
            'estimation_script': 'scripts/paper_analyses/215802-V1.py',
            **result
        })

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 70)

    # Load data
    print("\nLoading and preparing data...")
    df = load_and_prepare_data()
    print(f"Data loaded: {len(df)} observations, {df['person_id'].nunique()} individuals")

    # Run specification search
    print("\nRunning specification search...")
    results = run_specification_search(df)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = f"{OUTPUT_PATH}specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    valid_results = results_df[results_df['coefficient'].notna()]

    print(f"\nTotal specifications run: {len(results_df)}")
    print(f"Valid results: {len(valid_results)}")

    if len(valid_results) > 0:
        print(f"\nCoefficient distribution:")
        print(f"  Mean: {valid_results['coefficient'].mean():.4f}")
        print(f"  Median: {valid_results['coefficient'].median():.4f}")
        print(f"  Std Dev: {valid_results['coefficient'].std():.4f}")
        print(f"  Min: {valid_results['coefficient'].min():.4f}")
        print(f"  Max: {valid_results['coefficient'].max():.4f}")

        n_positive = (valid_results['coefficient'] > 0).sum()
        n_sig_05 = (valid_results['p_value'] < 0.05).sum()
        n_sig_01 = (valid_results['p_value'] < 0.01).sum()

        print(f"\nSignificance:")
        print(f"  Positive coefficients: {n_positive} ({100*n_positive/len(valid_results):.1f}%)")
        print(f"  Significant at 5%: {n_sig_05} ({100*n_sig_05/len(valid_results):.1f}%)")
        print(f"  Significant at 1%: {n_sig_01} ({100*n_sig_01/len(valid_results):.1f}%)")

    print("\n" + "=" * 70)
    print("Specification search complete.")
    print("=" * 70)
