"""
Specification Search: 205581-V1 - Global Universalism Survey
============================================================

Paper: "Universalism: Global Evidence" (Cappelen, Enke, and Tungodden)
AER 2024

Main Hypothesis: Higher universalism (domestic and foreign) is associated with
more egalitarian, globalist, and pro-immigrant political attitudes.

Method: Cross-sectional OLS with country fixed effects and demographic controls,
        clustered standard errors at sampling stratum level.

Primary Analysis: Table 1 - Regress political attitudes on universalism measures.
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

PAPER_ID = "205581-V1"
JOURNAL = "AER"
PAPER_TITLE = "Universalism: Global Evidence"

# Paths
BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/205581-V1/GUS_Package_AER"
OUTPUT_PATH = DATA_PATH

# ============================================================================
# Data Loading and Cleaning (Replicates variable-maker.do)
# ============================================================================

def load_and_clean_data():
    """
    Load the Gallup data and construct universalism measures.
    This replicates the key variable transformations from variable-maker.do
    """
    # Load raw data
    df = pd.read_stata(DATA_PATH / "SubData/Original/Gallup/WP_universalism_pub.dta")

    # Convert categorical columns to numeric where needed
    # Many Gallup variables are stored as categorical but need to be numeric for calculations
    alloc_cols = ['WP21567', 'WP21568', 'WP21570', 'WP21571', 'WP21573', 'WP21574',
                  'WP21577', 'WP21578', 'WP21580', 'WP21581', 'WP21583', 'WP21584',
                  'WP21587', 'WP21588', 'WP21590', 'WP21591', 'WP21593', 'WP21594',
                  'WP21596', 'WP21597', 'WP21599', 'WP21600', 'WP21602', 'WP21603',
                  'WP21611', 'WP21606', 'WP21612', 'WP21609', 'WP21613', 'WP21614',
                  'WP21616', 'WP21617', 'WP21619', 'WP21620', 'WP21622', 'WP21623']

    for col in alloc_cols:
        if col in df.columns:
            if df[col].dtype.name == 'category':
                # Convert category to numeric, handling non-numeric categories
                df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle demographic variables that may be categorical
    # WP1219: Gender (1=Male, 2=Female)
    if df['WP1219'].dtype.name == 'category':
        # Map string categories to numeric codes
        gender_map = {cat: 1 if 'male' in cat.lower() and 'female' not in cat.lower() else 2
                     for cat in df['WP1219'].cat.categories}
        df['WP1219'] = df['WP1219'].map(gender_map).astype(float)

    # WP1220: Age
    df['WP1220'] = pd.to_numeric(df['WP1220'].astype(str), errors='coerce')

    # WP3117: Education - convert categorical to numeric codes
    if df['WP3117'].dtype.name == 'category':
        # Get the category codes (1-indexed typically in Stata)
        df['WP3117'] = df['WP3117'].cat.codes + 1  # codes are 0-indexed, add 1 for Stata style
        df.loc[df['WP3117'] <= 0, 'WP3117'] = np.nan  # -1 codes (missing) become NaN

    # WP14: Urban/Rural
    if df['WP14'].dtype.name == 'category':
        df['WP14'] = df['WP14'].cat.codes + 1
        df.loc[df['WP14'] <= 0, 'WP14'] = np.nan

    # INCOME_5: Income quintile
    if df['INCOME_5'].dtype.name == 'category':
        df['INCOME_5'] = df['INCOME_5'].cat.codes + 1
        df.loc[df['INCOME_5'] <= 0, 'INCOME_5'] = np.nan

    # NHH_RANDOM_ENTRANCE: Treatment (1=Baseline/Module A, 2=Moral/Module B, 3=Deserving/Module C)
    if df['NHH_RANDOM_ENTRANCE'].dtype.name == 'category':
        treatment_map = {
            'Module I Section A': 1, 'Baseline': 1,
            'Module I Section B': 2, 'Moral': 2,
            'Module I Section C': 3, 'Deserving': 3
        }
        df['NHH_RANDOM_ENTRANCE'] = df['NHH_RANDOM_ENTRANCE'].map(treatment_map).astype(float)

    # Attitude variables - coded as Likert scale (1=Strongly Agree, 4=Strongly Disagree)
    likert_map = {
        'Strongly agree': 1,
        'Somewhat agree': 2,
        'Somewhat disagree': 3,
        'Strongly disagree': 4,
        '(DK)': np.nan,
        '(Refused)': np.nan
    }

    attitude_cols = ['WP21625', 'WP21626', 'WP21627', 'WP21628', 'WP21629', 'WP21630']
    for col in attitude_cols:
        if col in df.columns and df[col].dtype.name == 'category':
            df[col] = df[col].map(likert_map)

    # Load country-level data files for merging
    mmisocodes = pd.read_stata(DATA_PATH / "SubData/utils/mmisocodes.dta")
    gdp_data = pd.read_stata(DATA_PATH / "SubData/utils/gdp-pp-ppc.dta")
    inc_classif = pd.read_stata(DATA_PATH / "SubData/utils/cty-inc-classif.dta")
    continent = pd.read_stata(DATA_PATH / "SubData/utils/cty-continent.dta")
    maddison = pd.read_stata(DATA_PATH / "SubData/utils/cty-maddison.dta")
    reg_classif = pd.read_stata(DATA_PATH / "SubData/utils/cty-reg-classif.dta")

    # Merge ISO codes
    df = df.merge(mmisocodes, on='countrynew', how='left')

    # Create country numeric encoding
    df['cty'] = pd.factorize(df['countrynew'])[0]

    # Treatment variable
    df['treatment'] = df['NHH_RANDOM_ENTRANCE']

    # ========================================================================
    # Compute allocation percentages (replicate from variable-maker.do)
    # ========================================================================

    # Missing value codes in Gallup
    MISSING_CODES = [999999998, 999999999]

    # Baseline treatment allocations
    df['A_fam'] = df['WP21567'] / (df['WP21567'] + df['WP21568'])
    df['A_friend'] = df['WP21570'] / (df['WP21570'] + df['WP21571'])
    df['A_neigh'] = df['WP21573'] / (df['WP21573'] + df['WP21574'])
    df['A_rel'] = df['WP21577'] / (df['WP21577'] + df['WP21578'])
    df['A_eth'] = df['WP21580'] / (df['WP21580'] + df['WP21581'])
    df['A_str'] = df['WP21583'] / (df['WP21583'] + df['WP21584'])

    # Set missing where DK/refused
    for var, alloc_var in [('A_fam', 'WP21567'), ('A_friend', 'WP21570'),
                           ('A_neigh', 'WP21573'), ('A_rel', 'WP21577'),
                           ('A_eth', 'WP21580'), ('A_str', 'WP21583')]:
        df.loc[df[alloc_var].isin(MISSING_CODES), var] = np.nan

    # Moral treatment allocations
    df['B_fam'] = df['WP21587'] / (df['WP21587'] + df['WP21588'])
    df['B_friend'] = df['WP21590'] / (df['WP21590'] + df['WP21591'])
    df['B_neigh'] = df['WP21593'] / (df['WP21593'] + df['WP21594'])
    df['B_rel'] = df['WP21596'] / (df['WP21597'] + df['WP21596'])
    df['B_eth'] = df['WP21599'] / (df['WP21600'] + df['WP21599'])
    df['B_str'] = df['WP21602'] / (df['WP21602'] + df['WP21603'])

    for var, alloc_var in [('B_fam', 'WP21587'), ('B_friend', 'WP21590'),
                           ('B_neigh', 'WP21593'), ('B_rel', 'WP21596'),
                           ('B_eth', 'WP21599'), ('B_str', 'WP21602')]:
        df.loc[df[alloc_var].isin(MISSING_CODES), var] = np.nan

    # Deserving treatment allocations
    df['C_fam'] = df['WP21611'] / (df['WP21611'] + df['WP21606'])
    df['C_friend'] = df['WP21612'] / (df['WP21609'] + df['WP21612'])
    df['C_neigh'] = df['WP21613'] / (df['WP21614'] + df['WP21613'])
    df['C_rel'] = df['WP21616'] / (df['WP21616'] + df['WP21617'])
    df['C_eth'] = df['WP21619'] / (df['WP21619'] + df['WP21620'])
    df['C_str'] = df['WP21622'] / (df['WP21622'] + df['WP21623'])

    for var, alloc_var in [('C_fam', 'WP21611'), ('C_friend', 'WP21612'),
                           ('C_neigh', 'WP21613'), ('C_rel', 'WP21616'),
                           ('C_eth', 'WP21619'), ('C_str', 'WP21622')]:
        df.loc[df[alloc_var].isin(MISSING_CODES), var] = np.nan

    # Aggregate across treatments (use max to capture whichever treatment was assigned)
    df['alloc_family'] = df[['A_fam', 'B_fam', 'C_fam']].max(axis=1) * 100
    df['alloc_friends'] = df[['A_friend', 'B_friend', 'C_friend']].max(axis=1) * 100
    df['alloc_neighbor'] = df[['A_neigh', 'B_neigh', 'C_neigh']].max(axis=1) * 100
    df['alloc_religion'] = df[['A_rel', 'B_rel', 'C_rel']].max(axis=1) * 100
    df['alloc_ethnicity'] = df[['A_eth', 'B_eth', 'C_eth']].max(axis=1) * 100
    df['alloc_foreign'] = df[['A_str', 'B_str', 'C_str']].max(axis=1) * 100

    # ========================================================================
    # Recoding procedure (simplified - without full wrong_coding adjustment)
    # ========================================================================

    # Count allocations below 50 and total allocations
    alloc_vars = ['alloc_family', 'alloc_friends', 'alloc_neighbor',
                  'alloc_religion', 'alloc_ethnicity', 'alloc_foreign']

    df['sub50_allocs'] = 0
    df['eqlt50_allocs'] = 0
    df['tot_allocs'] = 0

    for var in alloc_vars:
        df['sub50_allocs'] += (df[var] < 50).astype(int)
        df['eqlt50_allocs'] += (df[var] <= 50).astype(int)
        df['tot_allocs'] += df[var].notna().astype(int)

    df['perlteq50'] = df['eqlt50_allocs'] / df['tot_allocs']
    df['perlt50'] = df['sub50_allocs'] / df['tot_allocs']

    # Recode allocations where needed
    for var in alloc_vars:
        df[f'recode_{var}'] = df[var].copy()
        # Recode if all allocs are weakly below 50 and at least 50% strictly below 50
        mask = (df['perlteq50'] == 1) & (df['perlt50'] >= 0.5) & (df[var] < 50) & (df['tot_allocs'] >= 1)
        df.loc[mask, f'recode_{var}'] = 100 - df.loc[mask, var]

    # ========================================================================
    # Compute universalism measures
    # ========================================================================

    # Domestic universalism (based on recoded allocations)
    domestic_vars = ['recode_alloc_family', 'recode_alloc_friends', 'recode_alloc_neighbor',
                     'recode_alloc_religion', 'recode_alloc_ethnicity']
    df['univ_domestic'] = 100 - df[domestic_vars].mean(axis=1)

    # Foreign universalism
    df['univ_foreign'] = 100 - df['recode_alloc_foreign']

    # Composite universalism
    df['univ_overall'] = df[['univ_domestic', 'univ_foreign']].mean(axis=1)

    # ========================================================================
    # Demographic variables
    # ========================================================================

    df['age'] = df['WP1220']
    df['agesq'] = df['age'] ** 2
    df['female'] = df['WP1219'] - 1
    df['male'] = 1 - df['female']

    # Education level
    df['educlevel'] = df['WP3117'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 3})
    df.loc[df['educlevel'] == 3, 'educlevel'] = np.nan

    # Urban/city
    df['urban'] = df['WP14'].replace({4: 5, 5: 5, 3: 4, 6: 3})
    df.loc[df['urban'] > 4, 'urban'] = np.nan
    df['city'] = (df['urban'] == 4).astype(float)
    df.loc[df['urban'].isna(), 'city'] = np.nan

    # College
    df['college'] = (df['educlevel'] == 2).astype(float)
    df.loc[df['educlevel'].isna(), 'college'] = np.nan

    # Income quintile
    df['income'] = df['INCOME_5']

    # ========================================================================
    # Political attitudes variables
    # ========================================================================

    # There are too many immigrants in the area you live in (inverted = pro-immigrant)
    df['immig_area'] = df['WP21625']
    df.loc[df['immig_area'] > 4, 'immig_area'] = np.nan

    # There are too many immigrants in your country (inverted = pro-immigrant)
    df['immig_cty'] = df['WP21626']
    df.loc[df['immig_cty'] > 4, 'immig_cty'] = np.nan

    # Government should focus on helping poor elsewhere vs locally (higher = global focus)
    df['focus_global_poor'] = df['WP21627']
    df.loc[df['focus_global_poor'] > 4, 'focus_global_poor'] = np.nan

    # Government should focus on global vs local environment
    df['focus_global_env'] = df['WP21628']
    df.loc[df['focus_global_env'] > 4, 'focus_global_env'] = np.nan

    # Strong military
    df['focus_military'] = df['WP21629']
    df.loc[df['focus_military'] > 4, 'focus_military'] = np.nan

    # Reduce inequality
    df['focus_ineq'] = df['WP21630']
    df.loc[df['focus_ineq'] > 4, 'focus_ineq'] = np.nan
    df['focus_ineq_inverted'] = 5 - df['focus_ineq']

    # ========================================================================
    # Merge country-level data
    # ========================================================================

    df = df.merge(gdp_data, on='iso', how='left')
    df['loggdp'] = np.log(df['gdp_pp_ppp_2019'])

    df = df.merge(inc_classif, on='iso', how='left')
    df = df.merge(continent, on='iso', how='left')
    df = df.merge(maddison, on='iso', how='left')
    df = df.merge(reg_classif, on='iso', how='left')

    # WEIRD indicator - maddison_region can be categorical strings
    df['weird'] = 0
    # Check if maddison_region contains numeric codes or string labels
    if df['maddison_region'].dtype.name == 'category':
        # String labels
        df.loc[df['maddison_region'].isin(['Western Europe', 'Western Offshoots']), 'weird'] = 1
    else:
        # Numeric codes (6=Western Europe, 7=Western Offshoots in original Stata code)
        df.loc[df['maddison_region'].isin([6, 7]), 'weird'] = 1

    # High income indicator - wb_incomegroup can be categorical
    if df['wb_incomegroup'].dtype.name == 'category':
        df['highincome'] = (df['wb_incomegroup'] == 'High income').astype(float)
        df.loc[~df['wb_incomegroup'].isin(['High income', 'Middle income', 'Low income']), 'highincome'] = np.nan
    else:
        df['highincome'] = (df['wb_incomegroup'] == 12).astype(float)
        df.loc[~df['wb_incomegroup'].isin([12, 22, 27]), 'highincome'] = np.nan

    # Pool moral and deserving treatments
    df['treatment_pooled'] = df['treatment']
    df.loc[df['treatment'] == 3, 'treatment_pooled'] = 2

    # Strata variable for clustering
    df['strata'] = df['WP12258A']

    # Drop observations with missing universalism
    df = df.dropna(subset=['univ_overall'])

    return df


# ============================================================================
# Regression Helper Functions
# ============================================================================

def run_ols_with_fe(df, formula, cluster_var=None, fe_vars=None):
    """
    Run OLS regression with fixed effects and optional clustering.

    Parameters:
    -----------
    df : DataFrame
        Data
    formula : str
        Regression formula (outcome ~ treatment + controls)
    cluster_var : str
        Variable to cluster standard errors on
    fe_vars : list
        List of fixed effect variables to absorb

    Returns:
    --------
    dict with regression results
    """
    # Parse the formula
    parts = formula.split('~')
    dep_var = parts[0].strip()
    indep_part = parts[1].strip()

    # Create working dataframe
    work_df = df.copy()

    # Handle fixed effects by adding dummies
    fe_cols = []
    if fe_vars:
        for fe_var in fe_vars:
            # Ensure the FE variable is numeric or string, not category
            if work_df[fe_var].dtype.name == 'category':
                work_df[fe_var] = work_df[fe_var].astype(str)
            dummies = pd.get_dummies(work_df[fe_var], prefix=fe_var, drop_first=True, dtype=float)
            fe_cols.extend(dummies.columns.tolist())
            work_df = pd.concat([work_df, dummies], axis=1)

    # Parse independent variables
    indep_vars = [v.strip() for v in indep_part.split('+')]

    # Build design matrices
    all_vars = indep_vars + fe_cols

    # Drop rows with missing values
    vars_to_check = [dep_var] + indep_vars
    if cluster_var:
        vars_to_check.append(cluster_var)
    work_df = work_df.dropna(subset=vars_to_check)

    # Build X matrix
    X = work_df[all_vars].copy()
    X = sm.add_constant(X)
    y = work_df[dep_var]

    # Fit model
    model = sm.OLS(y, X, missing='drop')

    if cluster_var:
        # Clustered standard errors
        result = model.fit(cov_type='cluster', cov_kwds={'groups': work_df[cluster_var]})
    else:
        # Robust standard errors
        result = model.fit(cov_type='HC1')

    return result, indep_vars


def extract_results(result, treatment_var, indep_vars, spec_id, spec_tree_path,
                    outcome_var, df, fe_vars=None, cluster_var=None, controls_desc=""):
    """
    Extract results from statsmodels regression into our standard format.
    """

    # Get treatment coefficient info
    coef = result.params.get(treatment_var, np.nan)
    se = result.bse.get(treatment_var, np.nan)
    tstat = result.tvalues.get(treatment_var, np.nan)
    pval = result.pvalues.get(treatment_var, np.nan)

    # Confidence intervals
    ci = result.conf_int()
    ci_lower = ci.loc[treatment_var, 0] if treatment_var in ci.index else np.nan
    ci_upper = ci.loc[treatment_var, 1] if treatment_var in ci.index else np.nan

    # Build coefficient vector JSON
    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": float(coef),
            "se": float(se),
            "pval": float(pval)
        },
        "controls": [],
        "fixed_effects": fe_vars if fe_vars else [],
        "diagnostics": {
            "r_squared": float(result.rsquared),
            "adj_r_squared": float(result.rsquared_adj) if hasattr(result, 'rsquared_adj') else None,
            "f_stat": float(result.fvalue) if hasattr(result, 'fvalue') else None,
            "f_pval": float(result.f_pvalue) if hasattr(result, 'f_pvalue') else None
        }
    }

    # Add control coefficients
    for var in indep_vars:
        if var != treatment_var and var in result.params.index:
            coef_vector["controls"].append({
                "var": var,
                "coef": float(result.params[var]),
                "se": float(result.bse[var]),
                "pval": float(result.pvalues[var])
            })

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
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(result.nobs),
        'r_squared': float(result.rsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': f"N={int(result.nobs)}",
        'fixed_effects': ", ".join(fe_vars) if fe_vars else "none",
        'controls_desc': controls_desc,
        'cluster_var': cluster_var if cluster_var else "none",
        'model_type': 'OLS with FE' if fe_vars else 'OLS',
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


# ============================================================================
# Main Specification Search
# ============================================================================

def run_specification_search():
    """
    Run systematic specification search for the Global Universalism paper.
    """

    print("Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"Data loaded: {len(df)} observations")

    results = []

    # Scale universalism variables to 0-1 for regression (as done in paper)
    df['univ_domestic_scaled'] = df['univ_domestic'] / 100
    df['univ_foreign_scaled'] = df['univ_foreign'] / 100
    df['univ_overall_scaled'] = df['univ_overall'] / 100

    # Define outcome variables (political attitudes)
    outcome_vars = [
        ('focus_ineq_inverted', 'Reduce Inequality'),
        ('focus_global_poor', 'Focus Global Poor'),
        ('focus_global_env', 'Focus Global Environment'),
        ('immig_area', 'Pro Immigrants in Area'),
        ('immig_cty', 'Pro Immigrants in Country'),
        ('focus_military', 'Oppose Strong Military')
    ]

    # Treatment variables - both domestic and foreign universalism
    treatment_vars = ['univ_domestic_scaled', 'univ_foreign_scaled']

    # Control variables
    baseline_controls = ['age', 'male', 'college', 'city', 'income']

    # ========================================================================
    # BASELINE SPECIFICATIONS (Table 1 replication)
    # ========================================================================
    print("\n--- Running Baseline Specifications (Table 1) ---")

    for outcome, outcome_label in outcome_vars:
        # Prepare data
        work_df = df.dropna(subset=[outcome, 'univ_domestic_scaled', 'univ_foreign_scaled',
                                    'cty', 'treatment_pooled', 'strata'] + baseline_controls)

        # Build formula with both universalism measures
        control_str = ' + '.join(baseline_controls)
        formula = f"{outcome} ~ univ_domestic_scaled + univ_foreign_scaled + {control_str}"

        try:
            result, indep_vars = run_ols_with_fe(
                work_df,
                formula,
                cluster_var='strata',
                fe_vars=['cty', 'treatment_pooled']
            )

            # Extract results for domestic universalism
            res = extract_results(
                result, 'univ_domestic_scaled', indep_vars,
                'baseline',
                'methods/cross_sectional_ols.md#baseline',
                outcome, df,
                fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc=f"Demographic controls: {control_str}"
            )
            res['outcome_label'] = outcome_label
            res['treatment_var'] = 'univ_domestic_scaled'
            results.append(res)

            # Also extract for foreign universalism
            res_for = extract_results(
                result, 'univ_foreign_scaled', indep_vars,
                'baseline',
                'methods/cross_sectional_ols.md#baseline',
                outcome, df,
                fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc=f"Demographic controls: {control_str}"
            )
            res_for['outcome_label'] = outcome_label
            res_for['treatment_var'] = 'univ_foreign_scaled'
            results.append(res_for)

            print(f"  {outcome_label}: dom_univ={res['coefficient']:.3f} (p={res['p_value']:.3f}), "
                  f"for_univ={res_for['coefficient']:.3f} (p={res_for['p_value']:.3f})")

        except Exception as e:
            print(f"  Error for {outcome}: {e}")

    # ========================================================================
    # OLS METHOD VARIATIONS
    # ========================================================================
    print("\n--- Running Method Variations ---")

    # For simplicity, focus on first outcome (reduce inequality)
    main_outcome = 'focus_ineq_inverted'
    main_treatment = 'univ_domestic_scaled'

    work_df = df.dropna(subset=[main_outcome, main_treatment, 'univ_foreign_scaled',
                                'cty', 'treatment_pooled', 'strata'] + baseline_controls)

    control_str = ' + '.join(baseline_controls)

    # 1. Standard errors variations
    se_variations = [
        ('robust/cluster/strata', 'strata', 'robustness/clustering_variations.md'),
        ('robust/cluster/none', None, 'robustness/clustering_variations.md'),
    ]

    # Cluster by country
    work_df['cty_cluster'] = work_df['cty']
    se_variations.append(('robust/cluster/country', 'cty_cluster', 'robustness/clustering_variations.md'))

    for spec_id, cluster, spec_path in se_variations:
        try:
            formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled + {control_str}"
            result, indep_vars = run_ols_with_fe(
                work_df, formula, cluster_var=cluster, fe_vars=['cty', 'treatment_pooled']
            )
            res = extract_results(
                result, main_treatment, indep_vars, spec_id, spec_path,
                main_outcome, work_df, fe_vars=['cty', 'treatment_pooled'],
                cluster_var=cluster if cluster else 'robust',
                controls_desc=f"Demographic controls: {control_str}"
            )
            results.append(res)
            print(f"  {spec_id}: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")
        except Exception as e:
            print(f"  Error for {spec_id}: {e}")

    # ========================================================================
    # CONTROL SET VARIATIONS
    # ========================================================================
    print("\n--- Running Control Set Variations ---")

    control_sets = [
        ('ols/controls/none', []),
        ('ols/controls/age_only', ['age']),
        ('ols/controls/demographics', ['age', 'male']),
        ('ols/controls/baseline', baseline_controls),
        ('ols/controls/full', baseline_controls + ['agesq']),
    ]

    for spec_id, controls in control_sets:
        try:
            if controls:
                control_str = ' + '.join(controls)
                formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled + {control_str}"
            else:
                formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled"

            result, indep_vars = run_ols_with_fe(
                work_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
            )
            res = extract_results(
                result, main_treatment, indep_vars, spec_id,
                'methods/cross_sectional_ols.md#control-sets',
                main_outcome, work_df, fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc=f"Controls: {controls if controls else 'none'}"
            )
            results.append(res)
            print(f"  {spec_id}: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
        except Exception as e:
            print(f"  Error for {spec_id}: {e}")

    # ========================================================================
    # LEAVE-ONE-OUT ROBUSTNESS
    # ========================================================================
    print("\n--- Running Leave-One-Out Robustness ---")

    for dropped_var in baseline_controls:
        try:
            remaining = [c for c in baseline_controls if c != dropped_var]
            control_str = ' + '.join(remaining) if remaining else ''

            if control_str:
                formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled + {control_str}"
            else:
                formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled"

            result, indep_vars = run_ols_with_fe(
                work_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
            )
            res = extract_results(
                result, main_treatment, indep_vars,
                f'robust/loo/drop_{dropped_var}',
                'robustness/leave_one_out.md',
                main_outcome, work_df, fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc=f"Dropped: {dropped_var}"
            )
            results.append(res)
            print(f"  Drop {dropped_var}: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
        except Exception as e:
            print(f"  Error dropping {dropped_var}: {e}")

    # ========================================================================
    # SINGLE COVARIATE ANALYSIS
    # ========================================================================
    print("\n--- Running Single Covariate Analysis ---")

    # Bivariate (no controls)
    try:
        formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled"
        result, indep_vars = run_ols_with_fe(
            work_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'robust/single/none',
            'robustness/single_covariate.md',
            main_outcome, work_df, fe_vars=['cty', 'treatment_pooled'],
            cluster_var='strata',
            controls_desc="No controls"
        )
        results.append(res)
        print(f"  Bivariate: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
    except Exception as e:
        print(f"  Error for bivariate: {e}")

    # Single covariate
    for control in baseline_controls:
        try:
            formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled + {control}"
            result, indep_vars = run_ols_with_fe(
                work_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
            )
            res = extract_results(
                result, main_treatment, indep_vars,
                f'robust/single/{control}',
                'robustness/single_covariate.md',
                main_outcome, work_df, fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc=f"Single control: {control}"
            )
            results.append(res)
            print(f"  + {control}: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
        except Exception as e:
            print(f"  Error for {control}: {e}")

    # ========================================================================
    # SAMPLE RESTRICTIONS
    # ========================================================================
    print("\n--- Running Sample Restrictions ---")

    # Full sample (already done in baseline)

    # WEIRD countries only
    weird_df = work_df[work_df['weird'] == 1]
    try:
        control_str = ' + '.join(baseline_controls)
        formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled + {control_str}"
        result, indep_vars = run_ols_with_fe(
            weird_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/sample/weird_only',
            'methods/cross_sectional_ols.md#sample-restrictions',
            main_outcome, weird_df, fe_vars=['cty', 'treatment_pooled'],
            cluster_var='strata',
            controls_desc="WEIRD countries only"
        )
        results.append(res)
        print(f"  WEIRD only: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}, n={res['n_obs']}")
    except Exception as e:
        print(f"  Error for WEIRD: {e}")

    # Non-WEIRD countries
    nonweird_df = work_df[work_df['weird'] == 0]
    try:
        result, indep_vars = run_ols_with_fe(
            nonweird_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/sample/nonweird_only',
            'methods/cross_sectional_ols.md#sample-restrictions',
            main_outcome, nonweird_df, fe_vars=['cty', 'treatment_pooled'],
            cluster_var='strata',
            controls_desc="Non-WEIRD countries only"
        )
        results.append(res)
        print(f"  Non-WEIRD: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}, n={res['n_obs']}")
    except Exception as e:
        print(f"  Error for non-WEIRD: {e}")

    # High income countries
    if 'highincome' in work_df.columns:
        hi_df = work_df[work_df['highincome'] == 1]
        try:
            result, indep_vars = run_ols_with_fe(
                hi_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
            )
            res = extract_results(
                result, main_treatment, indep_vars,
                'ols/sample/high_income',
                'methods/cross_sectional_ols.md#sample-restrictions',
                main_outcome, hi_df, fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc="High income countries only"
            )
            results.append(res)
            print(f"  High income: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}, n={res['n_obs']}")
        except Exception as e:
            print(f"  Error for high income: {e}")

    # Low/middle income countries
    if 'highincome' in work_df.columns:
        lmic_df = work_df[work_df['highincome'] == 0]
        try:
            result, indep_vars = run_ols_with_fe(
                lmic_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
            )
            res = extract_results(
                result, main_treatment, indep_vars,
                'ols/sample/lmic',
                'methods/cross_sectional_ols.md#sample-restrictions',
                main_outcome, lmic_df, fe_vars=['cty', 'treatment_pooled'],
                cluster_var='strata',
                controls_desc="LMIC countries only"
            )
            results.append(res)
            print(f"  LMIC: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}, n={res['n_obs']}")
        except Exception as e:
            print(f"  Error for LMIC: {e}")

    # Male subsample
    male_df = work_df[work_df['male'] == 1]
    try:
        result, indep_vars = run_ols_with_fe(
            male_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/sample/subgroup_male',
            'methods/cross_sectional_ols.md#sample-restrictions',
            main_outcome, male_df, fe_vars=['cty', 'treatment_pooled'],
            cluster_var='strata',
            controls_desc="Male subsample"
        )
        results.append(res)
        print(f"  Male only: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}, n={res['n_obs']}")
    except Exception as e:
        print(f"  Error for male: {e}")

    # Female subsample
    female_df = work_df[work_df['female'] == 1]
    try:
        result, indep_vars = run_ols_with_fe(
            female_df, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/sample/subgroup_female',
            'methods/cross_sectional_ols.md#sample-restrictions',
            main_outcome, female_df, fe_vars=['cty', 'treatment_pooled'],
            cluster_var='strata',
            controls_desc="Female subsample"
        )
        results.append(res)
        print(f"  Female only: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}, n={res['n_obs']}")
    except Exception as e:
        print(f"  Error for female: {e}")

    # ========================================================================
    # FIXED EFFECTS VARIATIONS
    # ========================================================================
    print("\n--- Running Fixed Effects Variations ---")

    # No fixed effects
    try:
        control_str = ' + '.join(baseline_controls)
        formula = f"{main_outcome} ~ {main_treatment} + univ_foreign_scaled + {control_str}"
        result, indep_vars = run_ols_with_fe(
            work_df, formula, cluster_var='strata', fe_vars=None
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/fe/none',
            'methods/cross_sectional_ols.md#fixed-effects',
            main_outcome, work_df, fe_vars=None,
            cluster_var='strata',
            controls_desc="No fixed effects"
        )
        results.append(res)
        print(f"  No FE: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
    except Exception as e:
        print(f"  Error for no FE: {e}")

    # Country FE only (no treatment FE)
    try:
        result, indep_vars = run_ols_with_fe(
            work_df, formula, cluster_var='strata', fe_vars=['cty']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/fe/country_only',
            'methods/cross_sectional_ols.md#fixed-effects',
            main_outcome, work_df, fe_vars=['cty'],
            cluster_var='strata',
            controls_desc="Country FE only"
        )
        results.append(res)
        print(f"  Country FE only: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
    except Exception as e:
        print(f"  Error for country FE only: {e}")

    # Treatment FE only
    try:
        result, indep_vars = run_ols_with_fe(
            work_df, formula, cluster_var='strata', fe_vars=['treatment_pooled']
        )
        res = extract_results(
            result, main_treatment, indep_vars,
            'ols/fe/treatment_only',
            'methods/cross_sectional_ols.md#fixed-effects',
            main_outcome, work_df, fe_vars=['treatment_pooled'],
            cluster_var='strata',
            controls_desc="Treatment FE only"
        )
        results.append(res)
        print(f"  Treatment FE only: coef={res['coefficient']:.3f}, p={res['p_value']:.3f}")
    except Exception as e:
        print(f"  Error for treatment FE only: {e}")

    # ========================================================================
    # ALL OUTCOMES WITH BASELINE SPEC
    # ========================================================================
    print("\n--- Running All Outcomes ---")

    for outcome, outcome_label in outcome_vars:
        for treat_var, treat_label in [('univ_domestic_scaled', 'domestic'),
                                        ('univ_foreign_scaled', 'foreign'),
                                        ('univ_overall_scaled', 'overall')]:

            work_df2 = df.dropna(subset=[outcome, 'univ_domestic_scaled', 'univ_foreign_scaled',
                                         'univ_overall_scaled', 'cty', 'treatment_pooled',
                                         'strata'] + baseline_controls)

            try:
                control_str = ' + '.join(baseline_controls)
                formula = f"{outcome} ~ {treat_var} + {control_str}"
                result, indep_vars = run_ols_with_fe(
                    work_df2, formula, cluster_var='strata', fe_vars=['cty', 'treatment_pooled']
                )
                res = extract_results(
                    result, treat_var, indep_vars,
                    f'ols/outcome/{outcome}/{treat_label}',
                    'methods/cross_sectional_ols.md#baseline',
                    outcome, work_df2, fe_vars=['cty', 'treatment_pooled'],
                    cluster_var='strata',
                    controls_desc=f"Full controls, {treat_label} universalism"
                )
                res['outcome_label'] = outcome_label
                results.append(res)
            except Exception as e:
                print(f"  Error for {outcome}/{treat_label}: {e}")

    return results


# ============================================================================
# Summary Statistics
# ============================================================================

def compute_summary_stats(results_df):
    """Compute summary statistics for the specification search."""

    # Filter to unique specifications
    total_specs = len(results_df)

    # Significance counts
    sig_05 = (results_df['p_value'] < 0.05).sum()
    sig_01 = (results_df['p_value'] < 0.01).sum()

    # Positive coefficients
    positive = (results_df['coefficient'] > 0).sum()

    # Coefficient statistics
    coef_median = results_df['coefficient'].median()
    coef_mean = results_df['coefficient'].mean()
    coef_min = results_df['coefficient'].min()
    coef_max = results_df['coefficient'].max()

    return {
        'total_specifications': total_specs,
        'positive_coefficients': positive,
        'positive_pct': 100 * positive / total_specs,
        'significant_05': sig_05,
        'significant_05_pct': 100 * sig_05 / total_specs,
        'significant_01': sig_01,
        'significant_01_pct': 100 * sig_01 / total_specs,
        'median_coefficient': coef_median,
        'mean_coefficient': coef_mean,
        'min_coefficient': coef_min,
        'max_coefficient': coef_max
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 70)

    # Run specifications
    results = run_specification_search()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = OUTPUT_PATH / "specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    stats = compute_summary_stats(results_df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Breakdown by spec type
    print("\n--- Breakdown by Specification Type ---")
    results_df['spec_type'] = results_df['spec_id'].apply(
        lambda x: x.split('/')[0] if '/' in x else x
    )

    for spec_type, group in results_df.groupby('spec_type'):
        n = len(group)
        sig = (group['p_value'] < 0.05).sum()
        print(f"  {spec_type}: {n} specs, {sig} significant ({100*sig/n:.1f}%)")

    print(f"\n\nTotal specifications: {len(results_df)}")
