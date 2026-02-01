"""
Specification Search for Paper 193625-V1: Great Recession and Income Gap
=========================================================================

This script performs a systematic specification search following the specification tree.

Paper Summary:
- Title: Great Recession and Income Gap (inferred from analysis content)
- Main Hypothesis: The Great Recession differentially affected income outcomes for
  graduates from different tiers of universities, with elite universities providing
  more insulation against recession-induced income declines.

Identification Strategy:
- Difference-in-Differences / Event Study with Triple Difference
- Treatment: Cohorts graduating during/after the Great Recession (post-2008)
- Comparison: Geographic variation in recession severity (badreccz - severely affected CZ)
- Additional dimension: University tier (tierbarrU)

Main Outcome: Log median income (positive earners) - lnk_medpos

Key Variables:
- gradrtierbarrU{i}badreccz: Triple interaction (post-recession x tier i x severe CZ)
- gradrtierbarrU{i}: Interaction (post-recession x tier i)
- gradr: Post-recession indicator
- badreccz: Binary for severely affected commuting zone

Fixed Effects:
- super_opeid: University fixed effects
- cohort#cz: Cohort x Commuting Zone fixed effects
- gradr#cz: Post-recession x Commuting Zone fixed effects

Controls:
- par_q1-par_q5: Parental income quintile shares
- lncount: Log enrollment count
- par_top10pc, par_top5pc, par_top1pc, par_toppt1pc: Top parental income shares
- female: Share female

Clustering: super_opeid (university level)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Try to import pyfixest for high-dimensional FE
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("Warning: pyfixest not available. Using statsmodels instead.")

# Fallback to statsmodels
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/193625-V1/ReplicationPackage_FinalAccept/ReplicationPackage"
OUTPUT_PATH = DATA_PATH

# Output files
RESULTS_CSV = OUTPUT_PATH / "specification_results.csv"
SUMMARY_MD = OUTPUT_PATH / "SPECIFICATION_SEARCH.md"

# Paper metadata
PAPER_ID = "193625-V1"
PAPER_TITLE = "Great Recession and Income Gap"

# =============================================================================
# Load Data
# =============================================================================

def load_data():
    """Load and prepare the main dataset."""
    print("Loading data...")
    df = pd.read_stata(DATA_PATH / "Mobility_byTier20Aug21.dta", convert_categoricals=False)

    # Apply sample restrictions from the Stata code
    # super_opeid ~= -1 & multi == 0 & tagunivyr == 1 & tierbarrU <= 6
    df = df[
        (df['super_opeid'] != -1) &
        (df['multi'] == 0) &
        (df['tagunivyr'] == 1) &
        (df['tierbarrU'].notna()) &
        (df['tierbarrU'] <= 6)
    ].copy()

    # Create the gradr variable (post-recession indicator)
    # gradr = cohort >= 1986 if tierbarrU <= 4
    # gradr = cohort >= 1983 if tierbarrU == 5
    # gradr = cohort >= 1985 if tierbarrU == 6
    df['gradr'] = 0
    df.loc[(df['tierbarrU'] <= 4) & (df['cohort'] >= 1986), 'gradr'] = 1
    df.loc[(df['tierbarrU'] == 5) & (df['cohort'] >= 1983), 'gradr'] = 1
    df.loc[(df['tierbarrU'] == 6) & (df['cohort'] >= 1985), 'gradr'] = 1

    # Create interaction variables
    # gradrbadreccz = gradr * badreccz
    df['gradrbadreccz'] = df['gradr'] * df['badreccz']

    # Create tier-specific interactions
    for i in [1, 2, 3, 5, 6]:  # Tier 4 is reference/excluded
        df[f'gradrtierbarrU{i}'] = df['gradr'] * (df['tierbarrU'] == i).astype(int)
        df[f'gradrtierbarrU{i}badreccz'] = df['gradr'] * (df['tierbarrU'] == i).astype(int) * df['badreccz']

    # Create log count
    df['lncount'] = np.log(df['count'].clip(lower=1))

    # Create cohort x cz interaction for FE
    df['cohort_cz'] = df['cohort'].astype(str) + '_' + df['cz'].astype(str)
    df['gradr_cz'] = df['gradr'].astype(str) + '_' + df['cz'].astype(str)

    # Drop rows with missing outcome
    df = df[df['lnk_medpos'].notna()].copy()

    print(f"Data loaded: {len(df)} observations")
    return df


# =============================================================================
# Define Specifications
# =============================================================================

# Control variable sets
CONTROLS_MINIMAL = ['female', 'lncount']
CONTROLS_PARENTAL = ['par_q1', 'par_q2', 'par_q3', 'par_q4', 'par_q5']
CONTROLS_TOP_INCOME = ['par_top10pc', 'par_top5pc', 'par_top1pc', 'par_toppt1pc']
CONTROLS_BASELINE = CONTROLS_PARENTAL + CONTROLS_TOP_INCOME + CONTROLS_MINIMAL

# Treatment variables for the triple-difference
TREATMENT_VARS = [
    'gradrtierbarrU1badreccz', 'gradrtierbarrU2badreccz',
    'gradrtierbarrU3badreccz', 'gradrtierbarrU5badreccz', 'gradrtierbarrU6badreccz',
]

TREATMENT_TIER_MAIN = [
    'gradrtierbarrU1', 'gradrtierbarrU2',
    'gradrtierbarrU3', 'gradrtierbarrU5', 'gradrtierbarrU6',
]

# Parental income interactions for robustness
PARENTAL_INTERACTIONS = []
for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
    PARENTAL_INTERACTIONS.extend([f'gradrpar_{q}badreccz', f'gradrpar_{q}'])
PARENTAL_INTERACTIONS.extend(['gradrpar_top10pcbadreccz', 'gradrpar_top10pc'])
PARENTAL_INTERACTIONS.extend(['par_top10pcbadreccz', 'par_q5badreccz', 'par_q4badreccz', 'par_q3badreccz', 'par_q2badreccz'])


# =============================================================================
# Regression Functions
# =============================================================================

def run_regression_statsmodels(df, y_var, x_vars, fe_vars=None, cluster_var=None):
    """Run OLS regression with statsmodels (demeaning for FE)."""
    # Make a copy and drop missing values
    vars_needed = [y_var] + x_vars
    if fe_vars:
        vars_needed.extend(fe_vars)
    if cluster_var:
        vars_needed.append(cluster_var)

    data = df[vars_needed].dropna().copy()

    if len(data) == 0:
        return None

    # Demean by FE groups if specified
    y = data[y_var].values
    X = data[x_vars].values

    if fe_vars:
        for fe_var in fe_vars:
            groups = data[fe_var].values
            unique_groups = np.unique(groups)

            # Demean y
            y_mean = np.array([y[groups == g].mean() for g in unique_groups])
            y_group_means = y_mean[np.searchsorted(unique_groups, groups)]
            y = y - y_group_means + y.mean()  # Within transformation

            # Demean X
            for j in range(X.shape[1]):
                x_col = X[:, j]
                x_mean = np.array([x_col[groups == g].mean() for g in unique_groups])
                x_group_means = x_mean[np.searchsorted(unique_groups, groups)]
                X[:, j] = x_col - x_group_means + x_col.mean()

    # Add constant for pooled OLS
    if not fe_vars:
        X = sm.add_constant(X)
        col_names = ['const'] + x_vars
    else:
        col_names = x_vars

    # Run OLS
    try:
        model = OLS(y, X)

        if cluster_var and cluster_var in data.columns:
            result = model.fit(cov_type='cluster',
                             cov_kwds={'groups': data[cluster_var].values})
        else:
            result = model.fit(cov_type='HC1')

        # Extract results
        results_dict = {
            'n_obs': len(data),
            'r_squared': result.rsquared,
            'coefficients': {}
        }

        for i, var in enumerate(col_names):
            results_dict['coefficients'][var] = {
                'coef': result.params[i],
                'se': result.bse[i],
                'pval': result.pvalues[i],
                'tstat': result.tvalues[i]
            }

        if cluster_var:
            results_dict['n_clusters'] = data[cluster_var].nunique()

        return results_dict

    except Exception as e:
        print(f"Regression failed: {e}")
        return None


def run_regression_pyfixest(df, formula, cluster_var=None):
    """Run regression using pyfixest."""
    try:
        if cluster_var:
            result = pf.feols(formula, data=df, vcov={'CRV1': cluster_var})
        else:
            result = pf.feols(formula, data=df, vcov='hetero')

        # Extract results using pyfixest 0.40 API
        coef_df = result.coef()
        se_df = result.se()
        pval_df = result.pvalue()

        # Get N and R2 using the internal attributes
        n_obs = result._N
        r_squared = result._r2

        results_dict = {
            'n_obs': n_obs,
            'r_squared': r_squared,
            'coefficients': {}
        }

        for var in coef_df.index:
            results_dict['coefficients'][var] = {
                'coef': float(coef_df[var]),
                'se': float(se_df[var]),
                'pval': float(pval_df[var]) if var in pval_df.index else np.nan
            }

        if cluster_var and cluster_var in df.columns:
            results_dict['n_clusters'] = df[cluster_var].nunique()

        return results_dict

    except Exception as e:
        print(f"Pyfixest regression failed: {e}")
        return None


def run_specification(df, spec_id, spec_tree_path, y_var, treatment_vars,
                     control_vars, fe_vars, cluster_var, description):
    """Run a single specification and return formatted results."""

    x_vars = treatment_vars + control_vars

    # Filter out any variables not in the dataframe
    x_vars = [v for v in x_vars if v in df.columns]

    if HAS_PYFIXEST and fe_vars:
        # Build pyfixest formula
        fe_formula = ' + '.join(fe_vars)
        x_formula = ' + '.join(x_vars)
        formula = f"{y_var} ~ {x_formula} | {fe_formula}"
        result = run_regression_pyfixest(df, formula, cluster_var)
    else:
        result = run_regression_statsmodels(df, y_var, x_vars, fe_vars, cluster_var)

    if result is None:
        return None

    # Format output
    output = {
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'description': description,
        'outcome_var': y_var,
        'n_obs': result['n_obs'],
        'r_squared': result['r_squared'],
        'n_clusters': result.get('n_clusters', np.nan),
        'cluster_var': cluster_var,
        'fe_vars': json.dumps(fe_vars) if fe_vars else '',
        'control_vars': json.dumps(control_vars) if control_vars else '',
    }

    # Extract treatment coefficients
    for i, tvar in enumerate(treatment_vars):
        if tvar in result['coefficients']:
            coef_info = result['coefficients'][tvar]
            output[f'treatment_{i+1}_var'] = tvar
            output[f'treatment_{i+1}_coef'] = coef_info['coef']
            output[f'treatment_{i+1}_se'] = coef_info['se']
            output[f'treatment_{i+1}_pval'] = coef_info['pval']

    # Store full coefficient vector as JSON
    output['coefficient_vector_json'] = json.dumps(result['coefficients'], default=str)

    return output


# =============================================================================
# Run Specification Search
# =============================================================================

def run_specification_search(df):
    """Run all specifications from the specification tree."""
    results = []

    # Create additional interaction variables needed for some specifications
    for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
        if f'gradrpar_{q}' not in df.columns:
            df[f'gradrpar_{q}'] = df['gradr'] * df[f'par_{q}']
        if f'gradrpar_{q}badreccz' not in df.columns:
            df[f'gradrpar_{q}badreccz'] = df['gradr'] * df[f'par_{q}'] * df['badreccz']

    if 'gradrpar_top10pc' not in df.columns:
        df['gradrpar_top10pc'] = df['gradr'] * df['par_top10pc']
    if 'gradrpar_top10pcbadreccz' not in df.columns:
        df['gradrpar_top10pcbadreccz'] = df['gradr'] * df['par_top10pc'] * df['badreccz']

    # Create parental income x badreccz interactions
    for v in ['par_top10pc', 'par_q5', 'par_q4', 'par_q3', 'par_q2']:
        if f'{v}badreccz' not in df.columns:
            df[f'{v}badreccz'] = df[v] * df['badreccz']

    print("\n" + "="*70)
    print("RUNNING SPECIFICATION SEARCH")
    print("="*70)

    # =========================================================================
    # BASELINE SPECIFICATION (Table 2, Column 4 - Full specification)
    # =========================================================================
    print("\n1. BASELINE SPECIFICATION")

    spec = run_specification(
        df=df,
        spec_id='baseline',
        spec_tree_path='methods/difference_in_differences.md#baseline',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var='super_opeid',
        description='Baseline: Full model with all FE and controls (Table 2, Col 4)'
    )
    if spec:
        results.append(spec)
        print(f"  - N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # =========================================================================
    # FIXED EFFECTS VARIATIONS (DiD Method Specs)
    # =========================================================================
    print("\n2. FIXED EFFECTS VARIATIONS")

    # 2a. University FE only (no CZ interactions)
    spec = run_specification(
        df=df,
        spec_id='did/fe/unit_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr', 'gradrbadreccz'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid'],
        cluster_var='super_opeid',
        description='University FE only (no CZ interactions)'
    )
    if spec:
        results.append(spec)
        print(f"  - did/fe/unit_only: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 2b. University + Cohort FE
    spec = run_specification(
        df=df,
        spec_id='did/fe/twoway',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr', 'gradrbadreccz'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort'],
        cluster_var='super_opeid',
        description='Two-way FE: University + Cohort'
    )
    if spec:
        results.append(spec)
        print(f"  - did/fe/twoway: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 2c. University + gradr x CZ FE (Table A2, Column 2)
    spec = run_specification(
        df=df,
        spec_id='did/fe/region_x_time',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=[],
        fe_vars=['super_opeid', 'gradr_cz'],
        cluster_var='super_opeid',
        description='University + Post x CZ FE (Table A2, Col 2)'
    )
    if spec:
        results.append(spec)
        print(f"  - did/fe/region_x_time: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 2d. No fixed effects (pooled OLS)
    spec = run_specification(
        df=df,
        spec_id='did/fe/none',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr', 'gradrbadreccz'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=None,
        cluster_var='super_opeid',
        description='No fixed effects (pooled OLS)'
    )
    if spec:
        results.append(spec)
        print(f"  - did/fe/none: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # =========================================================================
    # CONTROL SET VARIATIONS
    # =========================================================================
    print("\n3. CONTROL SET VARIATIONS")

    # 3a. No controls (Table 2, Column 1)
    spec = run_specification(
        df=df,
        spec_id='did/controls/none',
        spec_tree_path='methods/difference_in_differences.md#control-sets',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr', 'gradrbadreccz'],
        control_vars=[],
        fe_vars=['super_opeid'],
        cluster_var='super_opeid',
        description='No controls (Table 2, Col 1)'
    )
    if spec:
        results.append(spec)
        print(f"  - did/controls/none: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 3b. Minimal controls
    spec = run_specification(
        df=df,
        spec_id='did/controls/minimal',
        spec_tree_path='methods/difference_in_differences.md#control-sets',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_MINIMAL,
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var='super_opeid',
        description='Minimal controls (female, lncount)'
    )
    if spec:
        results.append(spec)
        print(f"  - did/controls/minimal: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 3c. Full controls with parental income interactions (Table 2, Column 4)
    parental_int_vars = [v for v in PARENTAL_INTERACTIONS if v in df.columns]
    spec = run_specification(
        df=df,
        spec_id='did/controls/full',
        spec_tree_path='methods/difference_in_differences.md#control-sets',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE + parental_int_vars,
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var='super_opeid',
        description='Full controls with parental income interactions'
    )
    if spec:
        results.append(spec)
        print(f"  - did/controls/full: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n4. SAMPLE RESTRICTIONS")

    # 4a. Elite tiers only (tiers 1-4)
    df_elite = df[df['tierbarrU'] <= 4].copy()
    spec = run_specification(
        df=df_elite,
        spec_id='did/sample/elite_only',
        spec_tree_path='methods/difference_in_differences.md#sample-restrictions',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var='super_opeid',
        description='Elite tiers only (Tier 1-4)'
    )
    if spec:
        results.append(spec)
        print(f"  - did/sample/elite_only: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 4b. Pre-treatment period only (placebo)
    df_pre = df[df['gradr'] == 0].copy()
    # Create fake post variable for half of pre-period
    df_pre['fake_post'] = (df_pre['cohort'] >= df_pre['cohort'].median()).astype(int)
    for i in [1, 2, 3, 5, 6]:
        df_pre[f'fakepost_tier{i}'] = df_pre['fake_post'] * (df_pre['tierbarrU'] == i).astype(int)
        df_pre[f'fakepost_tier{i}bad'] = df_pre['fake_post'] * (df_pre['tierbarrU'] == i).astype(int) * df_pre['badreccz']

    placebo_treat = [f'fakepost_tier{i}bad' for i in [1, 2, 3, 5, 6]]
    placebo_tier = [f'fakepost_tier{i}' for i in [1, 2, 3, 5, 6]]

    spec = run_specification(
        df=df_pre,
        spec_id='did/sample/pre_treatment',
        spec_tree_path='methods/difference_in_differences.md#sample-restrictions',
        y_var='lnk_medpos',
        treatment_vars=placebo_treat + placebo_tier + ['fake_post'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid'],
        cluster_var='super_opeid',
        description='Pre-treatment placebo test'
    )
    if spec:
        results.append(spec)
        print(f"  - did/sample/pre_treatment: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 4c. Severely affected CZs only
    df_bad = df[df['badreccz'] == 1].copy()
    spec = run_specification(
        df=df_bad,
        spec_id='did/sample/severe_cz_only',
        spec_tree_path='methods/difference_in_differences.md#sample-restrictions',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort_cz'],
        cluster_var='super_opeid',
        description='Severely affected CZs only'
    )
    if spec:
        results.append(spec)
        print(f"  - did/sample/severe_cz_only: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # 4d. Mildly affected CZs only
    df_mild = df[df['badreccz'] == 0].copy()
    spec = run_specification(
        df=df_mild,
        spec_id='did/sample/mild_cz_only',
        spec_tree_path='methods/difference_in_differences.md#sample-restrictions',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort_cz'],
        cluster_var='super_opeid',
        description='Mildly affected CZs only'
    )
    if spec:
        results.append(spec)
        print(f"  - did/sample/mild_cz_only: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # =========================================================================
    # ALTERNATIVE OUTCOME VARIABLES
    # =========================================================================
    print("\n5. ALTERNATIVE OUTCOMES")

    alt_outcomes = [
        ('lnk_median', 'Log median income (all)'),
        ('k_q5', 'Fraction in top income quintile'),
        ('k_top10pc', 'Fraction in top 10% income'),
        ('k_top1pc', 'Fraction in top 1% income'),
        ('k_0inc', 'Fraction with zero income')
    ]

    for y_var, desc in alt_outcomes:
        if y_var not in df.columns:
            # Create if needed
            if y_var == 'lnk_median' and 'k_median' in df.columns:
                df['lnk_median'] = np.log(df['k_median'].clip(lower=1))

        if y_var in df.columns:
            spec = run_specification(
                df=df,
                spec_id=f'did/outcome/{y_var}',
                spec_tree_path='methods/difference_in_differences.md#alternative-outcomes',
                y_var=y_var,
                treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
                control_vars=CONTROLS_BASELINE,
                fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
                cluster_var='super_opeid',
                description=f'Alternative outcome: {desc}'
            )
            if spec:
                results.append(spec)
                print(f"  - did/outcome/{y_var}: N={spec['n_obs']}, R2={spec['r_squared']:.4f}")

    # =========================================================================
    # EVENT STUDY (Cohort-by-Cohort Effects)
    # =========================================================================
    print("\n6. EVENT STUDY SPECIFICATIONS")

    # Create cohort x badreccz interaction variables
    for cohort in range(1980, 1992):
        varname = f'cohort{cohort}badreccz'
        if varname not in df.columns:
            df[varname] = ((df['cohort'] == cohort) * df['badreccz']).astype(float)

    # Event study within tier 4 (selective tier, most observations)
    cohort_vars = [f'cohort{c}badreccz' for c in range(1980, 1992) if c != 1983]  # 1983 is reference
    df_tier4 = df[df['tierbarrU'] == 4].copy()

    spec = run_specification(
        df=df_tier4,
        spec_id='es/window/symmetric',
        spec_tree_path='methods/event_study.md#event-window',
        y_var='lnk_medpos',
        treatment_vars=cohort_vars,
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort'],
        cluster_var='super_opeid',
        description='Event study: Tier 4, cohort x badreccz interactions (ref=1983)'
    )
    if spec:
        results.append(spec)
        print(f"  - es/window/symmetric (tier 4): N={spec['n_obs']}")

    # Event study for all tiers pooled
    spec = run_specification(
        df=df,
        spec_id='es/method/twfe',
        spec_tree_path='methods/event_study.md#estimation-method',
        y_var='lnk_medpos',
        treatment_vars=cohort_vars,
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort'],
        cluster_var='super_opeid',
        description='Event study: All tiers, TWFE (ref=1983)'
    )
    if spec:
        results.append(spec)
        print(f"  - es/method/twfe (all tiers): N={spec['n_obs']}")

    # =========================================================================
    # ROBUSTNESS: LEAVE-ONE-OUT
    # =========================================================================
    print("\n7. LEAVE-ONE-OUT ROBUSTNESS")

    for control in CONTROLS_BASELINE:
        loo_controls = [c for c in CONTROLS_BASELINE if c != control]
        spec = run_specification(
            df=df,
            spec_id=f'robust/loo/drop_{control}',
            spec_tree_path='robustness/leave_one_out.md',
            y_var='lnk_medpos',
            treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
            control_vars=loo_controls,
            fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
            cluster_var='super_opeid',
            description=f'Leave-one-out: drop {control}'
        )
        if spec:
            results.append(spec)
            print(f"  - robust/loo/drop_{control}: coef1={spec.get('treatment_1_coef', 'NA'):.4f}")

    # =========================================================================
    # ROBUSTNESS: SINGLE COVARIATE
    # =========================================================================
    print("\n8. SINGLE COVARIATE ROBUSTNESS")

    # No controls
    spec = run_specification(
        df=df,
        spec_id='robust/single/none',
        spec_tree_path='robustness/single_covariate.md',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=[],
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var='super_opeid',
        description='Bivariate: no controls'
    )
    if spec:
        results.append(spec)
        print(f"  - robust/single/none: coef1={spec.get('treatment_1_coef', 'NA'):.4f}")

    for control in CONTROLS_BASELINE:
        spec = run_specification(
            df=df,
            spec_id=f'robust/single/{control}',
            spec_tree_path='robustness/single_covariate.md',
            y_var='lnk_medpos',
            treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
            control_vars=[control],
            fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
            cluster_var='super_opeid',
            description=f'Single covariate: {control} only'
        )
        if spec:
            results.append(spec)
            print(f"  - robust/single/{control}: coef1={spec.get('treatment_1_coef', 'NA'):.4f}")

    # =========================================================================
    # ROBUSTNESS: CLUSTERING VARIATIONS
    # =========================================================================
    print("\n9. CLUSTERING VARIATIONS")

    # No clustering (robust SE)
    spec = run_specification(
        df=df,
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var=None,
        description='Robust SE (no clustering)'
    )
    if spec:
        results.append(spec)
        print(f"  - robust/cluster/none: SE1={spec.get('treatment_1_se', 'NA'):.4f}")

    # Cluster by CZ
    spec = run_specification(
        df=df,
        spec_id='robust/cluster/region',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        y_var='lnk_medpos',
        treatment_vars=TREATMENT_VARS + TREATMENT_TIER_MAIN + ['gradr'],
        control_vars=CONTROLS_BASELINE,
        fe_vars=['super_opeid', 'cohort_cz', 'gradr_cz'],
        cluster_var='cz',
        description='Cluster by CZ'
    )
    if spec:
        results.append(spec)
        print(f"  - robust/cluster/region: SE1={spec.get('treatment_1_se', 'NA'):.4f}")

    # =========================================================================
    # BY TIER ANALYSIS (Event Study Style)
    # =========================================================================
    print("\n10. BY-TIER HETEROGENEITY")

    for tier in [1, 2, 3, 4, 5, 6]:
        df_tier = df[df['tierbarrU'] == tier].copy()
        if len(df_tier) > 100:
            spec = run_specification(
                df=df_tier,
                spec_id=f'did/heterogeneity/tier_{tier}',
                spec_tree_path='methods/difference_in_differences.md#heterogeneity',
                y_var='lnk_medpos',
                treatment_vars=['gradrbadreccz', 'gradr'],
                control_vars=CONTROLS_BASELINE,
                fe_vars=['super_opeid', 'cohort'],
                cluster_var='super_opeid',
                description=f'Tier {tier} only'
            )
            if spec:
                results.append(spec)
                tier_coef = spec.get('treatment_1_coef', 'NA')
                if isinstance(tier_coef, float):
                    print(f"  - did/heterogeneity/tier_{tier}: N={spec['n_obs']}, coef={tier_coef:.4f}")
                else:
                    print(f"  - did/heterogeneity/tier_{tier}: N={spec['n_obs']}")

    return results


# =============================================================================
# Generate Summary Report
# =============================================================================

def generate_summary(results, df):
    """Generate the SPECIFICATION_SEARCH.md summary report."""

    # Find baseline
    baseline = next((r for r in results if r['spec_id'] == 'baseline'), None)

    summary = f"""# Specification Search Report: {PAPER_ID}

## Paper Overview

**Title**: {PAPER_TITLE}

**Main Hypothesis**: The Great Recession differentially affected income outcomes for
graduates from different tiers of universities. Elite universities provided more
insulation against recession-induced income declines compared to less selective
institutions.

**Identification Strategy**: Triple Difference-in-Differences
- Time dimension: Pre vs. Post Great Recession (cohort-based)
- Geographic dimension: Severely vs. Mildly affected commuting zones (badreccz)
- Institution dimension: University tiers (Ivy Plus, Elite, Selective, etc.)

**Key Outcomes**:
- Primary: Log median income (positive earners) - `lnk_medpos`
- Secondary: Fraction in top quintile, top 10%, top 1%, zero income

**Fixed Effects**:
- University (super_opeid)
- Cohort x Commuting Zone
- Post-recession x Commuting Zone

**Clustering**: University level (super_opeid)

---

## Baseline Specification

"""

    if baseline:
        summary += f"""
- **Observations**: {baseline['n_obs']:,}
- **R-squared**: {baseline['r_squared']:.4f}
- **Clusters**: {baseline.get('n_clusters', 'N/A')}

### Main Treatment Effects (Triple-Difference)

| Tier | Coefficient | SE | p-value |
|------|-------------|-----|---------|
"""
        tier_names = {1: 'Ivy Plus', 2: 'Elite', 3: 'Selective', 5: 'Less Selective', 6: 'Community'}
        for i, tvar in enumerate(TREATMENT_VARS):
            tier_num = int(tvar.split('tierbarrU')[1].split('badreccz')[0])
            tier_name = tier_names.get(tier_num, f'Tier {tier_num}')
            coef = baseline.get(f'treatment_{i+1}_coef', 'NA')
            se = baseline.get(f'treatment_{i+1}_se', 'NA')
            pval = baseline.get(f'treatment_{i+1}_pval', 'NA')
            if isinstance(coef, float):
                summary += f"| {tier_name} | {coef:.4f} | {se:.4f} | {pval:.4f} |\n"

    # Count specifications by category
    spec_counts = {}
    for r in results:
        cat = r['spec_id'].split('/')[0]
        spec_counts[cat] = spec_counts.get(cat, 0) + 1

    summary += f"""

---

## Specification Summary

**Total Specifications Run**: {len(results)}

| Category | Count |
|----------|-------|
"""
    for cat, count in sorted(spec_counts.items()):
        summary += f"| {cat} | {count} |\n"

    # Robustness summary
    summary += """

---

## Robustness Analysis

### Leave-One-Out Analysis

"""
    loo_specs = [r for r in results if r['spec_id'].startswith('robust/loo/')]
    if baseline and loo_specs:
        base_coef = baseline.get('treatment_1_coef', 0)
        summary += "| Dropped Variable | Coefficient | Change (%) |\n|------------------|-------------|------------|\n"
        for r in loo_specs:
            dropped = r['spec_id'].split('drop_')[1]
            coef = r.get('treatment_1_coef', 0)
            if isinstance(coef, float) and isinstance(base_coef, float) and base_coef != 0:
                pct_change = ((coef - base_coef) / abs(base_coef)) * 100
                summary += f"| {dropped} | {coef:.4f} | {pct_change:+.1f}% |\n"

    summary += """

### Clustering Variations

"""
    cluster_specs = [r for r in results if r['spec_id'].startswith('robust/cluster/')]
    summary += "| Clustering | SE | p-value |\n|------------|-----|--------|\n"
    for r in cluster_specs:
        cluster_type = r['spec_id'].split('/')[-1]
        se = r.get('treatment_1_se', 'NA')
        pval = r.get('treatment_1_pval', 'NA')
        if isinstance(se, float):
            summary += f"| {cluster_type} | {se:.4f} | {pval:.4f} |\n"

    # Tier heterogeneity
    summary += """

### Tier Heterogeneity

"""
    tier_specs = [r for r in results if r['spec_id'].startswith('did/heterogeneity/tier_')]
    summary += "| Tier | N | Coefficient (gradrbadreccz) | SE | p-value |\n|------|---|----------------------------|-----|--------|\n"
    for r in sorted(tier_specs, key=lambda x: int(x['spec_id'].split('_')[-1])):
        tier = r['spec_id'].split('_')[-1]
        n = r['n_obs']
        coef = r.get('treatment_1_coef', 'NA')
        se = r.get('treatment_1_se', 'NA')
        pval = r.get('treatment_1_pval', 'NA')
        if isinstance(coef, float):
            summary += f"| {tier} | {n:,} | {coef:.4f} | {se:.4f} | {pval:.4f} |\n"

    baseline_n = baseline['n_obs'] if baseline else 0
    summary += f"""

---

## Data Notes

- **Data Source**: Mobility Report Cards (Chetty et al.) merged with Great Recession shock data
- **Sample Period**: Birth cohorts 1980-1991
- **Unit of Observation**: University x Cohort x Parental Income Quintile
- **Sample Size (baseline)**: {baseline_n:,}

---

## Files

- Analysis script: `scripts/paper_analyses/193625-V1.py`
- Results CSV: `specification_results.csv`
- Data: `Mobility_byTier20Aug21.dta`

---

*Generated by specification search script*
"""

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print(f"SPECIFICATION SEARCH: {PAPER_ID}")
    print("="*70)

    # Load data
    df = load_data()

    # Run specification search
    results = run_specification_search(df)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to: {RESULTS_CSV}")

    # Generate summary report
    summary = generate_summary(results, df)
    with open(SUMMARY_MD, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {SUMMARY_MD}")

    print("\n" + "="*70)
    print("SPECIFICATION SEARCH COMPLETE")
    print(f"Total specifications: {len(results)}")
    print("="*70)

    return results_df


if __name__ == "__main__":
    results_df = main()
