"""
Specification Search: The Price of Experience (235621-V1)
=========================================================

Paper: "The Price of Experience" - AER
Method: Panel Fixed Effects with Mincerian Wage Equations
Main Treatment: Experience (e) and Education (s)
Outcome: Log Wages (lw)

This paper studies how experience affects wages and analyzes the changing
returns to experience over time using PSID data from 1968-2007.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# SETUP AND DATA LOADING
# =====================================================================

PAPER_ID = "235621-V1"
PAPER_TITLE = "The Price of Experience"
JOURNAL = "AER"
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/235621-V1"

# Load main data
df = pd.read_stata(f"{DATA_DIR}/basedata.dta")
print(f"Loaded basedata.dta: {df.shape}")

# Basic data cleaning
df = df.dropna(subset=['lw', 'e', 's', 'year'])
df = df[df['lw'] > -5]  # Remove extreme negative log wages

# Create derived variables
df['e2'] = df['e'] ** 2  # Experience squared (standard Mincer)
df['e3'] = df['e'] ** 3  # Experience cubed
df['s2'] = df['s'] ** 2  # Education squared
df['exp_edu'] = df['e'] * df['s']  # Experience-education interaction

# Create individual ID based on cohort (pseudo-individual since no true ID)
# Use cohort + sex + race combination as pseudo-individual
df['pseudo_id'] = df['cohort'].astype(str) + '_' + df['sex'].astype(str) + '_' + df['race'].fillna(0).astype(str)

# Year dummies are already in the data (y68, y69, etc.)
year_dummies = [c for c in df.columns if c.startswith('y') and len(c) in [3,4] and c[1:].isdigit()]

# Cohort dummies
cohort_dummies = [c for c in df.columns if c.startswith('c') and len(c) == 3 and c[1:].isdigit()]

# Region variables
region_vars = ['neast', 'ncenter', 'south', 'west']

print(f"Final sample size: {len(df)}")
print(f"Year dummies: {year_dummies[:5]}...")
print(f"Cohort dummies: {cohort_dummies[:5]}...")

# =====================================================================
# RESULTS CONTAINER
# =====================================================================

results = []

def add_result(spec_id, spec_tree_path, model, outcome_var='lw', treatment_var='e',
               sample_desc='Full sample', fixed_effects='', controls_desc='',
               cluster_var='year', model_type='OLS'):
    """Extract results from pyfixest model and add to results list."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]

        # Get all coefficients for coefficient vector
        coef_dict = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(' + ') if fixed_effects else [],
            "diagnostics": {}
        }

        # Add other coefficients
        for var in model.coef().index:
            if var != treatment_var:
                coef_dict["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

        result = {
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
            'ci_lower': float(coef - 1.96 * se),
            'ci_upper': float(coef + 1.96 * se),
            'n_obs': int(model._N),
            'r_squared': float(model._r2) if hasattr(model, '_r2') else np.nan,
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, n={model._N}")
        return True
    except Exception as ex:
        print(f"  ERROR in {spec_id}: {ex}")
        return False

# =====================================================================
# BASELINE SPECIFICATIONS
# =====================================================================
print("\n" + "="*60)
print("BASELINE SPECIFICATIONS")
print("="*60)

# 1. Baseline: Standard Mincer equation with experience and experience squared
print("\n1. Baseline Mincer equation...")
try:
    baseline = pf.feols("lw ~ e + e2 + s", data=df, vcov='hetero')
    add_result('baseline', 'methods/panel_fixed_effects.md#baseline', baseline,
               treatment_var='e', controls_desc='e, e2, s', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 2. Baseline with year FE
print("\n2. Baseline with year FE...")
try:
    baseline_year = pf.feols("lw ~ e + e2 + s | year", data=df, vcov='hetero')
    add_result('panel/fe/time', 'methods/panel_fixed_effects.md#fixed-effects-structure', baseline_year,
               treatment_var='e', controls_desc='e, e2, s', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 3. Baseline with cohort FE
print("\n3. Baseline with cohort FE...")
try:
    baseline_cohort = pf.feols("lw ~ e + e2 + s | cohort_5yr", data=df, vcov='hetero')
    add_result('panel/fe/cohort', 'methods/panel_fixed_effects.md#fixed-effects-structure', baseline_cohort,
               treatment_var='e', controls_desc='e, e2, s', fixed_effects='cohort_5yr', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 4. Two-way FE: year + cohort
print("\n4. Two-way FE (year + cohort)...")
try:
    twoway = pf.feols("lw ~ e + e2 + s | year + cohort_5yr", data=df, vcov='hetero')
    add_result('panel/fe/twoway', 'methods/panel_fixed_effects.md#fixed-effects-structure', twoway,
               treatment_var='e', controls_desc='e, e2, s', fixed_effects='year + cohort_5yr', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# CONTROL VARIABLE PROGRESSION (Build-up)
# =====================================================================
print("\n" + "="*60)
print("CONTROL VARIABLE PROGRESSION")
print("="*60)

# 5. Bivariate: just experience
print("\n5. Bivariate (experience only)...")
try:
    bivariate = pf.feols("lw ~ e", data=df, vcov='hetero')
    add_result('robust/build/bivariate', 'robustness/control_progression.md', bivariate,
               treatment_var='e', controls_desc='none', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 6. Add experience squared
print("\n6. Add experience squared...")
try:
    add_e2 = pf.feols("lw ~ e + e2", data=df, vcov='hetero')
    add_result('robust/build/add_exp_sq', 'robustness/control_progression.md', add_e2,
               treatment_var='e', controls_desc='e2', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 7. Add education
print("\n7. Add education...")
try:
    add_s = pf.feols("lw ~ e + e2 + s", data=df, vcov='hetero')
    add_result('robust/build/add_education', 'robustness/control_progression.md', add_s,
               treatment_var='e', controls_desc='e2, s', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 8. Add demographic controls (gender, race)
print("\n8. Add demographics...")
try:
    add_demo = pf.feols("lw ~ e + e2 + s + male + black", data=df, vcov='hetero')
    add_result('robust/build/demographics', 'robustness/control_progression.md', add_demo,
               treatment_var='e', controls_desc='e2, s, male, black', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 9. Add region controls
print("\n9. Add geographic controls...")
try:
    add_geo = pf.feols("lw ~ e + e2 + s + male + black + neast + ncenter + south", data=df, vcov='hetero')
    add_result('robust/build/geographic', 'robustness/control_progression.md', add_geo,
               treatment_var='e', controls_desc='e2, s, male, black, region', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 10. Add college indicator
print("\n10. Add college indicator...")
try:
    add_college = pf.feols("lw ~ e + e2 + s + male + black + college", data=df, vcov='hetero')
    add_result('robust/build/add_college', 'robustness/control_progression.md', add_college,
               treatment_var='e', controls_desc='e2, s, male, black, college', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 11. Full controls
print("\n11. Full controls...")
try:
    full_controls = pf.feols("lw ~ e + e2 + s + male + black + college + neast + ncenter + south", data=df, vcov='hetero')
    add_result('robust/build/full', 'robustness/control_progression.md', full_controls,
               treatment_var='e', controls_desc='e2, s, male, black, college, region', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 12. Kitchen sink with year FE
print("\n12. Kitchen sink with year FE...")
try:
    kitchen_sink = pf.feols("lw ~ e + e2 + s + male + black + college + neast + ncenter + south | year", data=df, vcov='hetero')
    add_result('robust/build/kitchen_sink', 'robustness/control_progression.md', kitchen_sink,
               treatment_var='e', controls_desc='e2, s, male, black, college, region',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# LEAVE-ONE-OUT CONTROL VARIATIONS
# =====================================================================
print("\n" + "="*60)
print("LEAVE-ONE-OUT CONTROL VARIATIONS")
print("="*60)

baseline_controls = ['e2', 's', 'male', 'black', 'college']

for i, ctrl in enumerate(baseline_controls):
    print(f"\n{13+i}. Drop {ctrl}...")
    remaining = [c for c in baseline_controls if c != ctrl]
    formula = f"lw ~ e + {' + '.join(remaining)} | year"
    try:
        model = pf.feols(formula, data=df, vcov='hetero')
        add_result(f'robust/control/drop_{ctrl}', 'robustness/leave_one_out.md', model,
                   treatment_var='e', controls_desc=f'baseline minus {ctrl}',
                   fixed_effects='year', cluster_var='robust')
    except Exception as ex:
        print(f"  ERROR: {ex}")

# =====================================================================
# CLUSTERING VARIATIONS
# =====================================================================
print("\n" + "="*60)
print("CLUSTERING VARIATIONS")
print("="*60)

# 18. Cluster by year
print("\n18. Cluster by year...")
try:
    cluster_year = pf.feols("lw ~ e + e2 + s + male + black | year", data=df, vcov={'CRV1': 'year'})
    add_result('robust/cluster/year', 'robustness/clustering_variations.md', cluster_year,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='year')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 19. Cluster by cohort
print("\n19. Cluster by cohort...")
try:
    cluster_cohort = pf.feols("lw ~ e + e2 + s + male + black | year", data=df, vcov={'CRV1': 'cohort_5yr'})
    add_result('robust/cluster/cohort', 'robustness/clustering_variations.md', cluster_cohort,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='cohort_5yr')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 20. Robust SE (HC1)
print("\n20. Robust SE (HC1)...")
try:
    robust_hc1 = pf.feols("lw ~ e + e2 + s + male + black | year", data=df, vcov='hetero')
    add_result('robust/se/hc1', 'robustness/clustering_variations.md', robust_hc1,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='robust_hc1')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 21. IID SE (no clustering, no robust)
print("\n21. IID standard errors...")
try:
    iid_se = pf.feols("lw ~ e + e2 + s + male + black | year", data=df, vcov='iid')
    add_result('robust/cluster/none', 'robustness/clustering_variations.md', iid_se,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='iid')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# SAMPLE RESTRICTIONS
# =====================================================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# 22. Early period (1968-1987)
print("\n22. Early period (1968-1987)...")
try:
    df_early = df[df['year'] <= 1987]
    early = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_early, vcov='hetero')
    add_result('robust/sample/early_period', 'robustness/sample_restrictions.md', early,
               treatment_var='e', sample_desc='1968-1987',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 23. Late period (1988-2007)
print("\n23. Late period (1988-2007)...")
try:
    df_late = df[df['year'] > 1987]
    late = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_late, vcov='hetero')
    add_result('robust/sample/late_period', 'robustness/sample_restrictions.md', late,
               treatment_var='e', sample_desc='1988-2007',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 24. Male only
print("\n24. Male only...")
try:
    df_male = df[df['male'] == 1]
    male_only = pf.feols("lw ~ e + e2 + s + black | year", data=df_male, vcov='hetero')
    add_result('robust/sample/male_only', 'robustness/sample_restrictions.md', male_only,
               treatment_var='e', sample_desc='Male only',
               controls_desc='e2, s, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 25. Female only
print("\n25. Female only...")
try:
    df_female = df[df['male'] == 0]
    female_only = pf.feols("lw ~ e + e2 + s + black | year", data=df_female, vcov='hetero')
    add_result('robust/sample/female_only', 'robustness/sample_restrictions.md', female_only,
               treatment_var='e', sample_desc='Female only',
               controls_desc='e2, s, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 26. College educated only
print("\n26. College educated only...")
try:
    df_college = df[df['college'] == 1]
    college_only = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_college, vcov='hetero')
    add_result('robust/sample/college_only', 'robustness/sample_restrictions.md', college_only,
               treatment_var='e', sample_desc='College educated only',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 27. Non-college only
print("\n27. Non-college only...")
try:
    df_noncollege = df[df['college'] == 0]
    noncollege = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_noncollege, vcov='hetero')
    add_result('robust/sample/noncollege_only', 'robustness/sample_restrictions.md', noncollege,
               treatment_var='e', sample_desc='Non-college only',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 28. Young workers (age <= 35)
print("\n28. Young workers (age <= 35)...")
try:
    df_young = df[df['age'] <= 35]
    young = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_young, vcov='hetero')
    add_result('robust/sample/young', 'robustness/sample_restrictions.md', young,
               treatment_var='e', sample_desc='Age <= 35',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 29. Older workers (age > 35)
print("\n29. Older workers (age > 35)...")
try:
    df_old = df[df['age'] > 35]
    old = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_old, vcov='hetero')
    add_result('robust/sample/old', 'robustness/sample_restrictions.md', old,
               treatment_var='e', sample_desc='Age > 35',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 30-33. Drop each region
print("\n30-33. Drop each region...")
for region, region_name in [('neast', 'Northeast'), ('ncenter', 'North Central'),
                             ('south', 'South'), ('west', 'West')]:
    try:
        df_excl = df[df[region] == 0]
        model = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_excl, vcov='hetero')
        add_result(f'robust/sample/exclude_{region}', 'robustness/sample_restrictions.md', model,
                   treatment_var='e', sample_desc=f'Exclude {region_name}',
                   controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
    except Exception as ex:
        print(f"  ERROR: {ex}")

# 34. Trim 1% outliers on log wages
print("\n34. Trim 1% outliers...")
try:
    lw_low = df['lw'].quantile(0.01)
    lw_high = df['lw'].quantile(0.99)
    df_trim = df[(df['lw'] >= lw_low) & (df['lw'] <= lw_high)]
    trim1 = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_trim, vcov='hetero')
    add_result('robust/sample/trim_1pct', 'robustness/sample_restrictions.md', trim1,
               treatment_var='e', sample_desc='Trim 1% outliers',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 35. Trim 5% outliers on log wages
print("\n35. Trim 5% outliers...")
try:
    lw_low = df['lw'].quantile(0.05)
    lw_high = df['lw'].quantile(0.95)
    df_trim5 = df[(df['lw'] >= lw_low) & (df['lw'] <= lw_high)]
    trim5 = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_trim5, vcov='hetero')
    add_result('robust/sample/trim_5pct', 'robustness/sample_restrictions.md', trim5,
               treatment_var='e', sample_desc='Trim 5% outliers',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 36. Winsorize 1%
print("\n36. Winsorize 1%...")
try:
    df_wins = df.copy()
    lw_low = df['lw'].quantile(0.01)
    lw_high = df['lw'].quantile(0.99)
    df_wins['lw_wins'] = df_wins['lw'].clip(lower=lw_low, upper=lw_high)
    wins1 = pf.feols("lw_wins ~ e + e2 + s + male + black | year", data=df_wins, vcov='hetero')
    add_result('robust/sample/winsor_1pct', 'robustness/sample_restrictions.md', wins1,
               treatment_var='e', outcome_var='lw_wins', sample_desc='Winsorize 1%',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 37. Black workers only
print("\n37. Black workers only...")
try:
    df_black = df[df['black'] == 1]
    black_only = pf.feols("lw ~ e + e2 + s + male | year", data=df_black, vcov='hetero')
    add_result('robust/sample/black_only', 'robustness/sample_restrictions.md', black_only,
               treatment_var='e', sample_desc='Black workers only',
               controls_desc='e2, s, male', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 38. Non-black workers only
print("\n38. Non-black workers only...")
try:
    df_nonblack = df[df['black'] == 0]
    nonblack = pf.feols("lw ~ e + e2 + s + male | year", data=df_nonblack, vcov='hetero')
    add_result('robust/sample/nonblack_only', 'robustness/sample_restrictions.md', nonblack,
               treatment_var='e', sample_desc='Non-black workers only',
               controls_desc='e2, s, male', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# FUNCTIONAL FORM VARIATIONS
# =====================================================================
print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# 39. Linear experience only (no quadratic)
print("\n39. Linear experience only...")
try:
    linear_exp = pf.feols("lw ~ e + s + male + black | year", data=df, vcov='hetero')
    add_result('robust/form/e_linear', 'robustness/functional_form.md', linear_exp,
               treatment_var='e', controls_desc='s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 40. Cubic experience
print("\n40. Cubic experience...")
try:
    cubic_exp = pf.feols("lw ~ e + e2 + e3 + s + male + black | year", data=df, vcov='hetero')
    add_result('robust/form/e_cubic', 'robustness/functional_form.md', cubic_exp,
               treatment_var='e', controls_desc='e2, e3, s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 41. Log experience
print("\n41. Log experience...")
try:
    df_pos_e = df[df['e'] > 0].copy()
    df_pos_e['log_e'] = np.log(df_pos_e['e'])
    log_exp = pf.feols("lw ~ log_e + s + male + black | year", data=df_pos_e, vcov='hetero')
    add_result('robust/form/e_log', 'robustness/functional_form.md', log_exp,
               treatment_var='log_e', controls_desc='s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 42. Education squared
print("\n42. Education squared...")
try:
    educ_sq = pf.feols("lw ~ e + e2 + s + s2 + male + black | year", data=df, vcov='hetero')
    add_result('robust/form/s_quadratic', 'robustness/functional_form.md', educ_sq,
               treatment_var='e', controls_desc='e2, s, s2, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 43. Experience-Education interaction
print("\n43. Experience-Education interaction...")
try:
    exp_edu = pf.feols("lw ~ e + e2 + s + exp_edu + male + black | year", data=df, vcov='hetero')
    add_result('robust/form/e_s_interact', 'robustness/functional_form.md', exp_edu,
               treatment_var='e', controls_desc='e2, s, e*s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 44. Levels outcome (wages, not log wages)
print("\n44. Levels outcome (w instead of lw)...")
try:
    levels = pf.feols("w ~ e + e2 + s + male + black | year", data=df, vcov='hetero')
    add_result('robust/form/y_level', 'robustness/functional_form.md', levels,
               treatment_var='e', outcome_var='w', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 45. Standardized outcome
print("\n45. Standardized outcome...")
try:
    df_std = df.copy()
    df_std['lw_std'] = (df_std['lw'] - df_std['lw'].mean()) / df_std['lw'].std()
    std_out = pf.feols("lw_std ~ e + e2 + s + male + black | year", data=df_std, vcov='hetero')
    add_result('robust/form/y_standardized', 'robustness/functional_form.md', std_out,
               treatment_var='e', outcome_var='lw_std', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# HETEROGENEITY ANALYSIS
# =====================================================================
print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# 46. By gender interaction
print("\n46. Gender interaction...")
try:
    het_gender = pf.feols("lw ~ e + e2 + s + male + male:e + black | year", data=df, vcov='hetero')
    add_result('robust/het/interaction_gender', 'robustness/heterogeneity.md', het_gender,
               treatment_var='e', controls_desc='e2, s, male, male*e, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 47. By race interaction
print("\n47. Race interaction...")
try:
    het_race = pf.feols("lw ~ e + e2 + s + male + black + black:e | year", data=df, vcov='hetero')
    add_result('robust/het/interaction_race', 'robustness/heterogeneity.md', het_race,
               treatment_var='e', controls_desc='e2, s, male, black, black*e',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 48. By education interaction
print("\n48. Education interaction...")
try:
    het_edu = pf.feols("lw ~ e + e2 + s + male + black + college:e | year", data=df, vcov='hetero')
    add_result('robust/het/interaction_education', 'robustness/heterogeneity.md', het_edu,
               treatment_var='e', controls_desc='e2, s, male, black, college*e',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 49. By cohort (early vs late cohorts)
print("\n49. Cohort heterogeneity...")
try:
    df_het = df.copy()
    df_het['late_cohort'] = (df_het['cohort'] >= 1950).astype(int)
    het_cohort = pf.feols("lw ~ e + e2 + s + male + black + late_cohort:e | year", data=df_het, vcov='hetero')
    add_result('robust/het/by_cohort', 'robustness/heterogeneity.md', het_cohort,
               treatment_var='e', controls_desc='e2, s, male, black, late_cohort*e',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 50. By period (early vs late period)
print("\n50. Period heterogeneity...")
try:
    df_het2 = df.copy()
    df_het2['late_period'] = (df_het2['year'] >= 1990).astype(int)
    het_period = pf.feols("lw ~ e + e2 + s + male + black + late_period:e | year", data=df_het2, vcov='hetero')
    add_result('robust/het/by_period', 'robustness/heterogeneity.md', het_period,
               treatment_var='e', controls_desc='e2, s, male, black, late_period*e',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# ADDITIONAL SPECIFICATIONS TO REACH 50+
# =====================================================================
print("\n" + "="*60)
print("ADDITIONAL SPECIFICATIONS")
print("="*60)

# 51. Drop first year (1968)
print("\n51. Drop first year...")
try:
    df_no1968 = df[df['year'] > 1968]
    no1968 = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_no1968, vcov='hetero')
    add_result('robust/sample/exclude_first_year', 'robustness/sample_restrictions.md', no1968,
               treatment_var='e', sample_desc='Exclude 1968',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 52. Drop last year (2007)
print("\n52. Drop last year...")
try:
    df_no2007 = df[df['year'] < 2007]
    no2007 = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_no2007, vcov='hetero')
    add_result('robust/sample/exclude_last_year', 'robustness/sample_restrictions.md', no2007,
               treatment_var='e', sample_desc='Exclude 2007',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 53. Experienced workers only (e >= 5)
print("\n53. Experienced workers (e >= 5)...")
try:
    df_exp5 = df[df['e'] >= 5]
    exp5 = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_exp5, vcov='hetero')
    add_result('robust/sample/min_exp_5', 'robustness/sample_restrictions.md', exp5,
               treatment_var='e', sample_desc='Experience >= 5',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 54. Less experienced workers (e < 20)
print("\n54. Less experienced workers (e < 20)...")
try:
    df_exp_low = df[df['e'] < 20]
    exp_low = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_exp_low, vcov='hetero')
    add_result('robust/sample/max_exp_20', 'robustness/sample_restrictions.md', exp_low,
               treatment_var='e', sample_desc='Experience < 20',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 55. High education (s >= 12)
print("\n55. High education (s >= 12)...")
try:
    df_high_s = df[df['s'] >= 12]
    high_s = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_high_s, vcov='hetero')
    add_result('robust/sample/high_education', 'robustness/sample_restrictions.md', high_s,
               treatment_var='e', sample_desc='Education >= 12',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 56. Low education (s < 12)
print("\n56. Low education (s < 12)...")
try:
    df_low_s = df[df['s'] < 12]
    low_s = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_low_s, vcov='hetero')
    add_result('robust/sample/low_education', 'robustness/sample_restrictions.md', low_s,
               treatment_var='e', sample_desc='Education < 12',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 57. Only unit FE (cohort)
print("\n57. Only unit FE (cohort)...")
try:
    unit_only = pf.feols("lw ~ e + e2 + s + male + black | cohort_5yr", data=df, vcov='hetero')
    add_result('robust/estimation/unit_fe_only', 'methods/panel_fixed_effects.md#estimation-method', unit_only,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='cohort_5yr', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 58. Pooled OLS (no FE)
print("\n58. Pooled OLS (no FE)...")
try:
    pooled = pf.feols("lw ~ e + e2 + s + male + black", data=df, vcov='hetero')
    add_result('panel/fe/none', 'methods/panel_fixed_effects.md#fixed-effects-structure', pooled,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='none', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 59. With age control
print("\n59. With age control...")
try:
    with_age = pf.feols("lw ~ e + e2 + s + male + black + age | year", data=df, vcov='hetero')
    add_result('robust/build/add_age', 'robustness/control_progression.md', with_age,
               treatment_var='e', controls_desc='e2, s, male, black, age',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 60. Middle experience (5 <= e <= 30)
print("\n60. Middle experience (5 <= e <= 30)...")
try:
    df_mid_exp = df[(df['e'] >= 5) & (df['e'] <= 30)]
    mid_exp = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_mid_exp, vcov='hetero')
    add_result('robust/sample/mid_experience', 'robustness/sample_restrictions.md', mid_exp,
               treatment_var='e', sample_desc='5 <= Experience <= 30',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 61-63. Drop each decade
print("\n61-63. Drop each decade...")
decades = [(1970, 1979), (1980, 1989), (1990, 1999)]
for dec_start, dec_end in decades:
    try:
        df_no_dec = df[(df['year'] < dec_start) | (df['year'] > dec_end)]
        model = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_no_dec, vcov='hetero')
        add_result(f'robust/sample/exclude_{dec_start}s', 'robustness/sample_restrictions.md', model,
                   treatment_var='e', sample_desc=f'Exclude {dec_start}s',
                   controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
    except Exception as ex:
        print(f"  ERROR: {ex}")

# 64. Weighted regression (by weight)
print("\n64. Weighted regression...")
try:
    df_weights = df.dropna(subset=['weight'])
    weighted = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_weights,
                        weights='weight', vcov='hetero')
    add_result('robust/weights/survey_weights', 'robustness/measurement.md', weighted,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 65. Unweighted (baseline is already unweighted, but let's confirm)
print("\n65. Unweighted (explicit)...")
try:
    unweighted = pf.feols("lw ~ e + e2 + s + male + black | year", data=df, vcov='hetero')
    add_result('robust/weights/unweighted', 'robustness/measurement.md', unweighted,
               treatment_var='e', controls_desc='e2, s, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 66. Prime age workers (25-54)
print("\n66. Prime age workers (25-54)...")
try:
    df_prime = df[(df['age'] >= 25) & (df['age'] <= 54)]
    prime = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_prime, vcov='hetero')
    add_result('robust/sample/prime_age', 'robustness/sample_restrictions.md', prime,
               treatment_var='e', sample_desc='Age 25-54',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 67. Triple interaction (gender x education x experience)
print("\n67. Triple interaction...")
try:
    triple = pf.feols("lw ~ e + e2 + s + male + black + male:college:e | year", data=df, vcov='hetero')
    add_result('robust/het/triple_diff', 'robustness/heterogeneity.md', triple,
               treatment_var='e', controls_desc='e2, s, male, black, male*college*e',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 68. South region only
print("\n68. South region only...")
try:
    df_south = df[df['south'] == 1]
    south_only = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_south, vcov='hetero')
    add_result('robust/het/by_region_south', 'robustness/heterogeneity.md', south_only,
               treatment_var='e', sample_desc='South region only',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 69. Non-South regions
print("\n69. Non-South regions...")
try:
    df_nonsouth = df[df['south'] == 0]
    nonsouth = pf.feols("lw ~ e + e2 + s + male + black | year", data=df_nonsouth, vcov='hetero')
    add_result('robust/het/by_region_nonsouth', 'robustness/heterogeneity.md', nonsouth,
               treatment_var='e', sample_desc='Non-South regions',
               controls_desc='e2, s, male, black', fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# 70. Returns to education (s as treatment)
print("\n70. Returns to education (s as treatment)...")
try:
    returns_s = pf.feols("lw ~ s + e + e2 + male + black | year", data=df, vcov='hetero')
    add_result('robust/outcome/returns_education', 'robustness/measurement.md', returns_s,
               treatment_var='s', controls_desc='e, e2, male, black',
               fixed_effects='year', cluster_var='robust')
except Exception as ex:
    print(f"  ERROR: {ex}")

# =====================================================================
# SAVE RESULTS
# =====================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"Total specifications: {len(results_df)}")

# Save to CSV
output_path = f"{DATA_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

if len(results_df) > 0:
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # Breakdown by category
    print("\nBreakdown by spec category:")
    results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0])
    for cat in results_df['category'].unique():
        cat_df = results_df[results_df['category'] == cat]
        pct_pos = 100 * (cat_df['coefficient'] > 0).mean()
        pct_sig = 100 * (cat_df['p_value'] < 0.05).mean()
        print(f"  {cat}: n={len(cat_df)}, {pct_pos:.0f}% positive, {pct_sig:.0f}% sig at 5%")

print("\nDone!")
