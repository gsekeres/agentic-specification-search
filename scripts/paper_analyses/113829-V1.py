"""
Specification Search for Paper 113829-V1
=========================================
Title: Political Connections and Sugar Mills in India
Journal: AEJ-Applied

Main hypothesis: Politically connected sugar mills pay higher prices to farmers for sugarcane.

Primary specification:
- Outcome: rprice (real price paid to farmers)
- Treatment: interall (political connection * control period interaction)
- Fixed Effects: mill (tabfinal) + year
- Clustering: Two-way (mill, zone_year)
- Controls: capacity, rainfall variables

Method: Panel Fixed Effects
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113829-V1/AEJ-App-2011-0020_data-and-replication-info"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113829-V1"

# Paper metadata
PAPER_ID = "113829-V1"
JOURNAL = "AEJ-Applied"
PAPER_TITLE = "Political Connections and Sugar Mills in India"

# ============================================================
# Load and Prepare Data
# ============================================================

df = pd.read_stata(f"{DATA_DIR}/entirepanelrec.dta")

# Apply sample restrictions from original do file
# - drops trial years
df = df[df['checktrial1'] != 1]
# - drops mill with no data on political connections
df = df[df['evercheck'].notna()]

# Convert year to int for pyfixest
df['year'] = df['year'].astype(int)
df['tabfinal'] = df['tabfinal'].astype(int)

# Create string versions for clustering (pyfixest requires string for cluster var)
df['tabfinal_str'] = df['tabfinal'].astype(str)
df['zone_year_str'] = df['zone_year'].astype(str)
df['year_str'] = df['year'].astype(str)

# Define control variable sets
rain_vars = ['r1_', 'r2_', 'r3_', 'r4_', 'r5_', 'r6_', 'r7_', 'r8_', 'r9_', 'r10_', 'r11_', 'r12_',
             'r1_dev', 'r2_dev', 'r3_dev', 'r4_dev', 'r5_dev', 'r6_dev', 'r7_dev', 'r8_dev',
             'r9_dev', 'r10_dev', 'r11_dev', 'r12_dev']

# Key controls from paper
key_controls = ['capacity'] + rain_vars


# ============================================================
# Helper Functions
# ============================================================

def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                    controls_desc, fixed_effects, cluster_var, model_type, sample_desc="Full sample"):
    """Extract results from pyfixest model into standardized format"""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%']
        ci_upper = ci.loc[treatment_var, '97.5%']
        nobs = model._N
        r2 = model._r2
    except Exception as e:
        print(f"Error extracting {treatment_var} from {spec_id}: {e}")
        return None

    # Build coefficient vector JSON
    coef_vector = {
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

    # Add all coefficients
    for var in model.coef().index:
        if var != treatment_var:
            coef_vector["controls"].append({
                "var": var,
                "coef": float(model.coef()[var]),
                "se": float(model.se()[var]),
                "pval": float(model.pvalue()[var])
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
        'n_obs': int(nobs),
        'r_squared': float(r2) if r2 is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


# ============================================================
# Run Specifications
# ============================================================

results = []

# ----------------------------------------------------------
# BASELINE: Replicate paper's main specification (Table 2, Column 2)
# ----------------------------------------------------------
print("Running baseline specification...")

# Table 2 Col 2: rprice on polcon, interall with capacity and rain controls, mill and year FE
# Note: pyfixest doesn't support two-way clustering directly, use single cluster as approximation
rain_formula = " + ".join(rain_vars)
baseline_formula = f"rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year"

try:
    # Use single clustering by tabfinal (mill) as primary cluster
    baseline = pf.feols(baseline_formula, data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(
        baseline, 'interall', 'baseline',
        'methods/panel_fixed_effects.md#baseline',
        'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
        'tabfinal', 'Panel FE'
    )
    if result:
        results.append(result)
        print(f"  Baseline: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"  Error in baseline: {e}")


# ----------------------------------------------------------
# METHOD-SPECIFIC: Panel FE Variations
# ----------------------------------------------------------
print("\nRunning panel FE variations...")

# 1. No fixed effects (pooled OLS)
try:
    model = pf.feols(f"rprice ~ polcon + interall + capacity + {rain_formula}", data=df, vcov='hetero')
    result = extract_results(model, 'interall', 'panel/fe/none',
                            'methods/panel_fixed_effects.md#fixed-effects-structure',
                            'rprice', 'polcon + capacity + rainfall', 'None',
                            'robust', 'Pooled OLS')
    if result:
        results.append(result)
        print(f"  panel/fe/none: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in panel/fe/none: {e}")

# 2. Unit FE only
try:
    model = pf.feols(f"rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'panel/fe/unit',
                            'methods/panel_fixed_effects.md#fixed-effects-structure',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal',
                            'tabfinal', 'Panel FE (unit)')
    if result:
        results.append(result)
        print(f"  panel/fe/unit: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in panel/fe/unit: {e}")

# 3. Year FE only
try:
    model = pf.feols(f"rprice ~ polcon + interall + capacity + {rain_formula} | year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'panel/fe/time',
                            'methods/panel_fixed_effects.md#fixed-effects-structure',
                            'rprice', 'polcon + capacity + rainfall', 'year',
                            'tabfinal', 'Panel FE (year)')
    if result:
        results.append(result)
        print(f"  panel/fe/time: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in panel/fe/time: {e}")

# 4. Two-way FE (same as baseline but without controls)
try:
    model = pf.feols("rprice ~ polcon + interall | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'panel/fe/twoway',
                            'methods/panel_fixed_effects.md#fixed-effects-structure',
                            'rprice', 'polcon only', 'tabfinal + year',
                            'tabfinal', 'Panel FE (two-way)')
    if result:
        results.append(result)
        print(f"  panel/fe/twoway (no controls): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in panel/fe/twoway: {e}")


# ----------------------------------------------------------
# ALTERNATIVE OUTCOMES (from Table 4)
# ----------------------------------------------------------
print("\nRunning alternative outcomes...")

alt_outcomes = ['recovery', 'actualhours', 'propbreakdown', 'propshortage',
                'canecrushed', 'lime', 'sulphur', 'caneplant', 'notinoperation']

for outcome in alt_outcomes:
    try:
        formula = f"{outcome} ~ polcon + interall + capacity + {rain_formula} | tabfinal + year"
        model = pf.feols(formula, data=df, vcov={"CRV1": "tabfinal_str"})
        result = extract_results(model, 'interall', f'panel/outcome/{outcome}',
                                'methods/panel_fixed_effects.md#baseline',
                                outcome, 'polcon + capacity + rainfall', 'tabfinal + year',
                                'tabfinal', 'Panel FE')
        if result:
            results.append(result)
            print(f"  panel/outcome/{outcome}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"  Error in panel/outcome/{outcome}: {e}")


# ----------------------------------------------------------
# ALTERNATIVE TREATMENTS
# ----------------------------------------------------------
print("\nRunning alternative treatment definitions...")

# pcconnected - politician connected
try:
    model = pf.feols(f"rprice ~ polcon + pcconnected + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'pcconnected', 'panel/treatment/pcconnected',
                            'methods/panel_fixed_effects.md#baseline',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  panel/treatment/pcconnected: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in pcconnected: {e}")

# acconnected - administrator connected
try:
    model = pf.feols(f"rprice ~ polcon + acconnected + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'acconnected', 'panel/treatment/acconnected',
                            'methods/panel_fixed_effects.md#baseline',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  panel/treatment/acconnected: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in acconnected: {e}")

# pcinter - politician connection interaction
try:
    model = pf.feols(f"rprice ~ polcon + pcinter + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'pcinter', 'panel/treatment/pcinter',
                            'methods/panel_fixed_effects.md#baseline',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  panel/treatment/pcinter: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in pcinter: {e}")

# acinter - administrator connection interaction
try:
    model = pf.feols(f"rprice ~ polcon + acinter + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'acinter', 'panel/treatment/acinter',
                            'methods/panel_fixed_effects.md#baseline',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  panel/treatment/acinter: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in acinter: {e}")


# ----------------------------------------------------------
# ROBUSTNESS: Clustering Variations
# ----------------------------------------------------------
print("\nRunning clustering variations...")

# No clustering (robust SE only)
try:
    model = pf.feols(baseline_formula, data=df, vcov='hetero')
    result = extract_results(model, 'interall', 'robust/cluster/none',
                            'robustness/clustering_variations.md#single-level-clustering',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'robust (no clustering)', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/cluster/none: se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in cluster/none: {e}")

# Cluster by mill only (same as baseline)
try:
    model = pf.feols(baseline_formula, data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/cluster/unit',
                            'robustness/clustering_variations.md#single-level-clustering',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/cluster/unit: se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in cluster/unit: {e}")

# Cluster by zone_year only
try:
    model = pf.feols(baseline_formula, data=df, vcov={"CRV1": "zone_year_str"})
    result = extract_results(model, 'interall', 'robust/cluster/zone_year',
                            'robustness/clustering_variations.md#single-level-clustering',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'zone_year', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/cluster/zone_year: se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in cluster/zone_year: {e}")

# Cluster by year only
try:
    model = pf.feols(baseline_formula, data=df, vcov={"CRV1": "year_str"})
    result = extract_results(model, 'interall', 'robust/cluster/time',
                            'robustness/clustering_variations.md#single-level-clustering',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'year', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/cluster/time: se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in cluster/time: {e}")


# ----------------------------------------------------------
# ROBUSTNESS: Leave-One-Out (key controls)
# ----------------------------------------------------------
print("\nRunning leave-one-out specifications...")

# Key controls for LOO: capacity, rain variables (we'll group rain for simplicity)
loo_controls = ['capacity']

for drop_var in loo_controls:
    try:
        remaining = [v for v in key_controls if v != drop_var]
        remaining_formula = " + ".join(remaining)
        formula = f"rprice ~ polcon + interall + {remaining_formula} | tabfinal + year"
        model = pf.feols(formula, data=df, vcov={"CRV1": "tabfinal_str"})
        result = extract_results(model, 'interall', f'robust/loo/drop_{drop_var}',
                                'robustness/leave_one_out.md',
                                'rprice', f'polcon + rainfall (no {drop_var})', 'tabfinal + year',
                                'tabfinal', 'Panel FE')
        if result:
            results.append(result)
            print(f"  robust/loo/drop_{drop_var}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error in loo/drop_{drop_var}: {e}")

# Drop all rainfall variables
try:
    formula = "rprice ~ polcon + interall + capacity | tabfinal + year"
    model = pf.feols(formula, data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/loo/drop_rainfall',
                            'robustness/leave_one_out.md',
                            'rprice', 'polcon + capacity (no rainfall)', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/loo/drop_rainfall: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in loo/drop_rainfall: {e}")


# ----------------------------------------------------------
# ROBUSTNESS: Single Covariate
# ----------------------------------------------------------
print("\nRunning single covariate specifications...")

# Bivariate (no controls)
try:
    model = pf.feols("rprice ~ polcon + interall | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/single/none',
                            'robustness/single_covariate.md',
                            'rprice', 'polcon only', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/single/none: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in single/none: {e}")

# Single control: capacity
try:
    model = pf.feols("rprice ~ polcon + interall + capacity | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/single/capacity',
                            'robustness/single_covariate.md',
                            'rprice', 'polcon + capacity', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/single/capacity: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in single/capacity: {e}")


# ----------------------------------------------------------
# ROBUSTNESS: Functional Form
# ----------------------------------------------------------
print("\nRunning functional form variations...")

# Log outcome
try:
    model = pf.feols(f"lnrprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/form/y_log',
                            'robustness/functional_form.md#outcome-variable-transformations',
                            'lnrprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/form/y_log: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in form/y_log: {e}")

# Add recovery and mill operation as additional controls (Table 2 Col 3)
try:
    formula = f"rprice ~ polcon + interall + capacity + recovery + propbreakdown + propshortage + {rain_formula} | tabfinal + year"
    model = pf.feols(formula, data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/form/extra_controls',
                            'robustness/functional_form.md',
                            'rprice', 'polcon + capacity + rainfall + recovery + mill ops', 'tabfinal + year',
                            'tabfinal', 'Panel FE')
    if result:
        results.append(result)
        print(f"  robust/form/extra_controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in form/extra_controls: {e}")


# ----------------------------------------------------------
# ROBUSTNESS: Sample Restrictions
# ----------------------------------------------------------
print("\nRunning sample restriction specifications...")

# Early period (1993-1998)
try:
    df_early = df[df['year'] <= 1998]
    model = pf.feols(baseline_formula, data=df_early, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/sample/early_period',
                            'robustness/sample_restrictions.md#time-based-restrictions',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Years 1993-1998')
    if result:
        results.append(result)
        print(f"  robust/sample/early_period: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"  Error in sample/early_period: {e}")

# Late period (1999-2005)
try:
    df_late = df[df['year'] >= 1999]
    model = pf.feols(baseline_formula, data=df_late, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/sample/late_period',
                            'robustness/sample_restrictions.md#time-based-restrictions',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Years 1999-2005')
    if result:
        results.append(result)
        print(f"  robust/sample/late_period: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"  Error in sample/late_period: {e}")

# Exclude first year
try:
    df_no_first = df[df['year'] > 1993]
    model = pf.feols(baseline_formula, data=df_no_first, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/sample/exclude_first_year',
                            'robustness/sample_restrictions.md#time-based-restrictions',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Excluding 1993')
    if result:
        results.append(result)
        print(f"  robust/sample/exclude_first_year: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in sample/exclude_first_year: {e}")

# Exclude last year
try:
    df_no_last = df[df['year'] < 2005]
    model = pf.feols(baseline_formula, data=df_no_last, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/sample/exclude_last_year',
                            'robustness/sample_restrictions.md#time-based-restrictions',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Excluding 2005')
    if result:
        results.append(result)
        print(f"  robust/sample/exclude_last_year: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in sample/exclude_last_year: {e}")

# Trim outliers (1st and 99th percentile of rprice)
try:
    df_trimmed = df[df['rprice'].notna()].copy()
    p01 = df_trimmed['rprice'].quantile(0.01)
    p99 = df_trimmed['rprice'].quantile(0.99)
    df_trimmed = df_trimmed[(df_trimmed['rprice'] >= p01) & (df_trimmed['rprice'] <= p99)]
    model = pf.feols(baseline_formula, data=df_trimmed, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/sample/trim_1pct',
                            'robustness/sample_restrictions.md#outlier-handling',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Trimmed 1/99 pct')
    if result:
        results.append(result)
        print(f"  robust/sample/trim_1pct: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in sample/trim_1pct: {e}")

# Mills with at least 5 observations
try:
    obs_counts = df.groupby('tabfinal').size()
    mills_5plus = obs_counts[obs_counts >= 5].index
    df_5plus = df[df['tabfinal'].isin(mills_5plus)]
    model = pf.feols(baseline_formula, data=df_5plus, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall', 'robust/sample/min_obs_5',
                            'robustness/sample_restrictions.md#panel-specific',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Mills with 5+ obs')
    if result:
        results.append(result)
        print(f"  robust/sample/min_obs_5: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in sample/min_obs_5: {e}")


# ----------------------------------------------------------
# ADDITIONAL SPECIFICATIONS FROM PAPER
# ----------------------------------------------------------
print("\nRunning additional paper specifications...")

# Table 3 Col 7: interall_9497 (placebo test for 1994-1997 pre-deregulation)
try:
    model = pf.feols(f"rprice ~ polcon + interall_9497 + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'interall_9497', 'custom/placebo_pre_deregulation',
                            'custom',
                            'rprice', 'polcon + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Placebo: pre-deregulation period')
    if result:
        results.append(result)
        print(f"  custom/placebo_pre_deregulation: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in placebo_pre_deregulation: {e}")

# Table 5 specifications: before/after analysis
try:
    model = pf.feols(f"rprice ~ pcinter + pcinterafter + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'pcinter', 'custom/pc_during_control',
                            'custom',
                            'rprice', 'pcinterafter + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'PC connection during control period')
    if result:
        results.append(result)
        print(f"  custom/pc_during_control: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in pc_during_control: {e}")

# Table 6: Party in state/center analysis
try:
    model = pf.feols(f"rprice ~ partyinstate + partyincenter + capacity + {rain_formula} | tabfinal + year",
                     data=df, vcov={"CRV1": "tabfinal_str"})
    result = extract_results(model, 'partyinstate', 'custom/party_in_state',
                            'custom',
                            'rprice', 'partyincenter + capacity + rainfall', 'tabfinal + year',
                            'tabfinal', 'Panel FE', 'Party alignment with state government')
    if result:
        results.append(result)
        print(f"  custom/party_in_state: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error in party_in_state: {e}")


# ============================================================
# Save Results
# ============================================================

print("\n" + "="*60)
print("Saving results...")

results_df = pd.DataFrame(results)
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved {len(results)} specifications to {output_path}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total specifications: {len(results)}")

if len(results) > 0:
    # Focus on main treatment (interall)
    interall_results = results_df[results_df['treatment_var'] == 'interall']
    print(f"\nFor main treatment (interall): {len(interall_results)} specs")
    if len(interall_results) > 0:
        print(f"  Positive coefficients: {(interall_results['coefficient'] > 0).sum()} ({(interall_results['coefficient'] > 0).mean()*100:.1f}%)")
        print(f"  Significant at 5%: {(interall_results['p_value'] < 0.05).sum()} ({(interall_results['p_value'] < 0.05).mean()*100:.1f}%)")
        print(f"  Significant at 1%: {(interall_results['p_value'] < 0.01).sum()} ({(interall_results['p_value'] < 0.01).mean()*100:.1f}%)")
        print(f"  Median coefficient: {interall_results['coefficient'].median():.4f}")
        print(f"  Mean coefficient: {interall_results['coefficient'].mean():.4f}")
        print(f"  Range: [{interall_results['coefficient'].min():.4f}, {interall_results['coefficient'].max():.4f}]")

print("\nDone!")
