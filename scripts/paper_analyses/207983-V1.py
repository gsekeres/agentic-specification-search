"""
Specification Search: Contamination Bias in Linear Regressions (207983-V1)
Authors: Goldsmith-Pinkham, Hull, Kolesar (2024)

This paper is a methodology paper that reanalyzes 9 RCT studies using the multe package.
For specification search purposes, we focus on one of the applications:
Benhassine et al. (2015) - "Turning a Shove into a Nudge? A Labeled Cash Transfer for Education"

The paper demonstrates contamination bias in regressions with multi-valued discrete treatments.
The main specification uses OLS with saturated treatment indicators and stratum fixed effects.
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import pyfixest as pf
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "207983-V1"
JOURNAL = "AER"
PAPER_TITLE = "Contamination Bias in Linear Regressions"
BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_DIR = BASE_DIR / "data/downloads/extracted/207983-V1"
OUTPUT_DIR = DATA_DIR

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, coef, se, tstat, pval,
               n_obs, r2, sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
               coef_vector=None):
    """Add a result to the results list"""
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    if coef_vector is None:
        coef_vector = {
            "treatment": {"var": treatment_var, "coef": coef, "se": se, "pval": pval},
            "controls": [],
            "fixed_effects": fixed_effects.split(", ") if fixed_effects else [],
            "diagnostics": {}
        }

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    })

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 60)
print(f"Specification Search: {PAPER_ID}")
print("=" * 60)
print()

# Load Benhassine data
print("Loading Benhassine et al. data...")
df = pd.read_stata(DATA_DIR / "dta/benhassine.dta", convert_categoricals=False)

# Create treatment dummies
df['treatment'] = df['treatment'].astype(int)
for i in range(5):
    df[f'treat_{i}'] = (df['treatment'] == i).astype(int)

# Also create aggregated treatments
df['treat_lct'] = ((df['treatment'] == 1) | (df['treatment'] == 2)).astype(int)
df['treat_cct'] = ((df['treatment'] == 3) | (df['treatment'] == 4)).astype(int)
df['any_treatment'] = (df['treatment'] > 0).astype(int)

# Define control variables
individual_controls = ['age_baseline', 'girl', 'bs_inschool08', 'bs_neverenrolled08',
                       'bs_inschool08_miss', 'bs_neverenrolled08_miss']
household_controls = ['bs_pchildren_enrolled', 'bs_pchildren_enrolled_miss']
school_controls = ['prel_elec', 'prel_elec_miss', 'prel_inacc_winter',
                   'prel_inacc_winter_miss']
all_controls = individual_controls + household_controls + school_controls + ['sampling_frame_problem']

print(f"  Sample size: {len(df)}")
print(f"  Treatment groups: {df['treatment'].value_counts().sort_index().to_dict()}")
print()

# ============================================================================
# BASELINE SPECIFICATION (Weighted, with stratum FE, clustered by school)
# ============================================================================

print("Running baseline specification...")

# Paper's main specification: outcome ~ treatment + controls | stratum (clustered by school)
baseline_formula = (f"enroll_attend_May2010 ~ C(treatment) + " +
                   " + ".join(all_controls) + " | stratum")

try:
    # pyfixest expects weights to be a column name string
    baseline_model = pf.feols(baseline_formula, data=df, vcov={'CRV1': 'schoolid'},
                              weights='weight_hh')

    # Extract results for each treatment arm vs control
    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        coef = float(baseline_model.coef()[coef_name])
        se = float(baseline_model.se()[coef_name])
        tstat = float(baseline_model.tstat()[coef_name])
        pval = float(baseline_model.pvalue()[coef_name])
        n_obs = int(baseline_model._N)
        r2 = float(baseline_model._r2)

        add_result(
            spec_id=f'baseline_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#baseline',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=coef, se=se, tstat=tstat, pval=pval,
            n_obs=n_obs, r2=r2,
            sample_desc='Full sample, Benhassine et al replication',
            fixed_effects='stratum',
            controls_desc='Individual, household, and school controls',
            cluster_var='schoolid',
            model_type='WLS (survey weights)'
        )
        print(f"  Treatment {treat_level}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")

except Exception as e:
    print(f"  Error in baseline: {e}")

print()

# ============================================================================
# METHOD VARIATIONS - STANDARD ERRORS
# ============================================================================

print("Running standard error variations...")

# Robust SE (no clustering)
try:
    model_robust = pf.feols(baseline_formula, data=df, vcov='hetero', weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/se/robust_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_robust.coef()[coef_name]),
            se=float(model_robust.se()[coef_name]),
            tstat=float(model_robust.tstat()[coef_name]),
            pval=float(model_robust.pvalue()[coef_name]),
            n_obs=int(model_robust._N), r2=float(model_robust._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='None (robust)',
            model_type='WLS'
        )
    print("  Robust SE: done")
except Exception as e:
    print(f"  Error with robust SE: {e}")

# Cluster by stratum instead of school
try:
    model_stratum_cluster = pf.feols(baseline_formula, data=df, vcov={'CRV1': 'stratum'},
                                     weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'robust/cluster/stratum_treat{treat_level}',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_stratum_cluster.coef()[coef_name]),
            se=float(model_stratum_cluster.se()[coef_name]),
            tstat=float(model_stratum_cluster.tstat()[coef_name]),
            pval=float(model_stratum_cluster.pvalue()[coef_name]),
            n_obs=int(model_stratum_cluster._N), r2=float(model_stratum_cluster._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='stratum',
            model_type='WLS'
        )
    print("  Cluster by stratum: done")
except Exception as e:
    print(f"  Error with stratum clustering: {e}")

print()

# ============================================================================
# METHOD VARIATIONS - CONTROL SETS
# ============================================================================

print("Running control set variations...")

# No controls
try:
    formula_no_controls = "enroll_attend_May2010 ~ C(treatment) | stratum"
    model_no_controls = pf.feols(formula_no_controls, data=df, vcov={'CRV1': 'schoolid'},
                                  weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/controls/none_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_no_controls.coef()[coef_name]),
            se=float(model_no_controls.se()[coef_name]),
            tstat=float(model_no_controls.tstat()[coef_name]),
            pval=float(model_no_controls.pvalue()[coef_name]),
            n_obs=int(model_no_controls._N), r2=float(model_no_controls._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='None (only FE)',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  No controls: done")
except Exception as e:
    print(f"  Error with no controls: {e}")

# Individual controls only
try:
    formula_indiv = (f"enroll_attend_May2010 ~ C(treatment) + " +
                    " + ".join(individual_controls) + " | stratum")
    model_indiv = pf.feols(formula_indiv, data=df, vcov={'CRV1': 'schoolid'},
                           weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/controls/individual_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_indiv.coef()[coef_name]),
            se=float(model_indiv.se()[coef_name]),
            tstat=float(model_indiv.tstat()[coef_name]),
            pval=float(model_indiv.pvalue()[coef_name]),
            n_obs=int(model_indiv._N), r2=float(model_indiv._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='Individual controls only',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  Individual controls only: done")
except Exception as e:
    print(f"  Error with individual controls: {e}")

# Household controls only
try:
    formula_hh = (f"enroll_attend_May2010 ~ C(treatment) + " +
                 " + ".join(household_controls) + " | stratum")
    model_hh = pf.feols(formula_hh, data=df, vcov={'CRV1': 'schoolid'},
                        weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/controls/household_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_hh.coef()[coef_name]),
            se=float(model_hh.se()[coef_name]),
            tstat=float(model_hh.tstat()[coef_name]),
            pval=float(model_hh.pvalue()[coef_name]),
            n_obs=int(model_hh._N), r2=float(model_hh._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='Household controls only',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  Household controls only: done")
except Exception as e:
    print(f"  Error with household controls: {e}")

print()

# ============================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================

print("Running leave-one-out robustness checks...")

for control in all_controls:
    try:
        remaining = [c for c in all_controls if c != control]
        formula_loo = (f"enroll_attend_May2010 ~ C(treatment) + " +
                      " + ".join(remaining) + " | stratum")
        model_loo = pf.feols(formula_loo, data=df, vcov={'CRV1': 'schoolid'},
                             weights='weight_hh')

        for treat_level in [1, 2, 3, 4]:
            coef_name = f'C(treatment)[T.{treat_level}]'
            add_result(
                spec_id=f'robust/loo/drop_{control}_treat{treat_level}',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var='enroll_attend_May2010',
                treatment_var=f'treatment_{treat_level}',
                coef=float(model_loo.coef()[coef_name]),
                se=float(model_loo.se()[coef_name]),
                tstat=float(model_loo.tstat()[coef_name]),
                pval=float(model_loo.pvalue()[coef_name]),
                n_obs=int(model_loo._N), r2=float(model_loo._r2),
                sample_desc='Full sample',
                fixed_effects='stratum',
                controls_desc=f'Drop {control}',
                cluster_var='schoolid',
                model_type='WLS'
            )
        print(f"  Drop {control}: done")
    except Exception as e:
        print(f"  Error dropping {control}: {e}")

print()

# ============================================================================
# SINGLE COVARIATE ROBUSTNESS
# ============================================================================

print("Running single covariate specifications...")

for control in all_controls:
    try:
        formula_single = f"enroll_attend_May2010 ~ C(treatment) + {control} | stratum"
        model_single = pf.feols(formula_single, data=df, vcov={'CRV1': 'schoolid'},
                                weights='weight_hh')

        for treat_level in [1, 2, 3, 4]:
            coef_name = f'C(treatment)[T.{treat_level}]'
            add_result(
                spec_id=f'robust/single/{control}_treat{treat_level}',
                spec_tree_path='robustness/single_covariate.md',
                outcome_var='enroll_attend_May2010',
                treatment_var=f'treatment_{treat_level}',
                coef=float(model_single.coef()[coef_name]),
                se=float(model_single.se()[coef_name]),
                tstat=float(model_single.tstat()[coef_name]),
                pval=float(model_single.pvalue()[coef_name]),
                n_obs=int(model_single._N), r2=float(model_single._r2),
                sample_desc='Full sample',
                fixed_effects='stratum',
                controls_desc=f'Only {control}',
                cluster_var='schoolid',
                model_type='WLS'
            )
    except Exception as e:
        print(f"  Error with single covariate {control}: {e}")

print("  Single covariate specifications: done")
print()

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("Running sample restriction specifications...")

# Boys only
try:
    df_boys = df[df['girl'] == 0].copy()
    model_boys = pf.feols(baseline_formula, data=df_boys, vcov={'CRV1': 'schoolid'},
                          weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/sample/boys_treat{treat_level}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_boys.coef()[coef_name]),
            se=float(model_boys.se()[coef_name]),
            tstat=float(model_boys.tstat()[coef_name]),
            pval=float(model_boys.pvalue()[coef_name]),
            n_obs=int(model_boys._N), r2=float(model_boys._r2),
            sample_desc='Boys only',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  Boys only: done")
except Exception as e:
    print(f"  Error with boys sample: {e}")

# Girls only
try:
    df_girls = df[df['girl'] == 1].copy()
    model_girls = pf.feols(baseline_formula, data=df_girls, vcov={'CRV1': 'schoolid'},
                           weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/sample/girls_treat{treat_level}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_girls.coef()[coef_name]),
            se=float(model_girls.se()[coef_name]),
            tstat=float(model_girls.tstat()[coef_name]),
            pval=float(model_girls.pvalue()[coef_name]),
            n_obs=int(model_girls._N), r2=float(model_girls._r2),
            sample_desc='Girls only',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  Girls only: done")
except Exception as e:
    print(f"  Error with girls sample: {e}")

# Previously enrolled only
try:
    df_enrolled = df[df['bs_inschool08'] == 1].copy()
    model_enrolled = pf.feols(baseline_formula, data=df_enrolled, vcov={'CRV1': 'schoolid'},
                              weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/sample/prev_enrolled_treat{treat_level}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_enrolled.coef()[coef_name]),
            se=float(model_enrolled.se()[coef_name]),
            tstat=float(model_enrolled.tstat()[coef_name]),
            pval=float(model_enrolled.pvalue()[coef_name]),
            n_obs=int(model_enrolled._N), r2=float(model_enrolled._r2),
            sample_desc='Previously enrolled (2008)',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  Previously enrolled: done")
except Exception as e:
    print(f"  Error with enrolled sample: {e}")

print()

# ============================================================================
# FIXED EFFECTS VARIATIONS
# ============================================================================

print("Running fixed effects variations...")

# No fixed effects
try:
    formula_no_fe = (f"enroll_attend_May2010 ~ C(treatment) + " +
                    " + ".join(all_controls))
    model_no_fe = pf.feols(formula_no_fe, data=df, vcov={'CRV1': 'schoolid'},
                           weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/fe/none_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_no_fe.coef()[coef_name]),
            se=float(model_no_fe.se()[coef_name]),
            tstat=float(model_no_fe.tstat()[coef_name]),
            pval=float(model_no_fe.pvalue()[coef_name]),
            n_obs=int(model_no_fe._N), r2=float(model_no_fe._r2),
            sample_desc='Full sample',
            fixed_effects='None',
            controls_desc='All controls',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  No FE: done")
except Exception as e:
    print(f"  Error with no FE: {e}")

# School fixed effects (treatment varies within schools in this RCT)
try:
    formula_school_fe = (f"enroll_attend_May2010 ~ C(treatment) + " +
                        " + ".join(all_controls) + " | schoolid")
    model_school_fe = pf.feols(formula_school_fe, data=df, vcov={'CRV1': 'stratum'},
                               weights='weight_hh')

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/fe/school_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_school_fe.coef()[coef_name]),
            se=float(model_school_fe.se()[coef_name]),
            tstat=float(model_school_fe.tstat()[coef_name]),
            pval=float(model_school_fe.pvalue()[coef_name]),
            n_obs=int(model_school_fe._N), r2=float(model_school_fe._r2),
            sample_desc='Full sample',
            fixed_effects='schoolid',
            controls_desc='All controls',
            cluster_var='stratum',
            model_type='WLS'
        )
    print("  School FE: done")
except Exception as e:
    print(f"  Error with school FE: {e}")

print()

# ============================================================================
# UNWEIGHTED SPECIFICATION
# ============================================================================

print("Running unweighted specification...")

try:
    model_unweighted = pf.feols(baseline_formula, data=df, vcov={'CRV1': 'schoolid'})

    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/method/unweighted_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#estimation-method',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(model_unweighted.coef()[coef_name]),
            se=float(model_unweighted.se()[coef_name]),
            tstat=float(model_unweighted.tstat()[coef_name]),
            pval=float(model_unweighted.pvalue()[coef_name]),
            n_obs=int(model_unweighted._N), r2=float(model_unweighted._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='schoolid',
            model_type='OLS (unweighted)'
        )
    print("  Unweighted: done")
except Exception as e:
    print(f"  Error with unweighted: {e}")

print()

# ============================================================================
# TREATMENT AGGREGATION (LCT vs CCT)
# ============================================================================

print("Running aggregated treatment specifications...")

# LCT vs CCT aggregation
try:
    formula_agg = (f"enroll_attend_May2010 ~ treat_lct + treat_cct + " +
                  " + ".join(all_controls) + " | stratum")
    model_agg = pf.feols(formula_agg, data=df, vcov={'CRV1': 'schoolid'},
                         weights='weight_hh')

    for treat_type in ['treat_lct', 'treat_cct']:
        add_result(
            spec_id=f'custom/aggregated_{treat_type}',
            spec_tree_path='methods/cross_sectional_ols.md#custom',
            outcome_var='enroll_attend_May2010',
            treatment_var=treat_type,
            coef=float(model_agg.coef()[treat_type]),
            se=float(model_agg.se()[treat_type]),
            tstat=float(model_agg.tstat()[treat_type]),
            pval=float(model_agg.pvalue()[treat_type]),
            n_obs=int(model_agg._N), r2=float(model_agg._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='All controls',
            cluster_var='schoolid',
            model_type='WLS'
        )
    print("  Aggregated treatments (LCT vs CCT): done")
except Exception as e:
    print(f"  Error with aggregated treatments: {e}")

# Any treatment vs control
try:
    formula_any = (f"enroll_attend_May2010 ~ any_treatment + " +
                  " + ".join(all_controls) + " | stratum")
    model_any = pf.feols(formula_any, data=df, vcov={'CRV1': 'schoolid'},
                         weights='weight_hh')

    add_result(
        spec_id='custom/any_treatment',
        spec_tree_path='methods/cross_sectional_ols.md#custom',
        outcome_var='enroll_attend_May2010',
        treatment_var='any_treatment',
        coef=float(model_any.coef()['any_treatment']),
        se=float(model_any.se()['any_treatment']),
        tstat=float(model_any.tstat()['any_treatment']),
        pval=float(model_any.pvalue()['any_treatment']),
        n_obs=int(model_any._N), r2=float(model_any._r2),
        sample_desc='Full sample',
        fixed_effects='stratum',
        controls_desc='All controls',
        cluster_var='schoolid',
        model_type='WLS'
    )
    print("  Any treatment: done")
except Exception as e:
    print(f"  Error with any treatment: {e}")

print()

# ============================================================================
# DISCRETE OUTCOME MODELS (LPM vs Logit/Probit)
# ============================================================================

print("Running discrete outcome variations...")

# Logit model (without stratum FE for convergence)
try:
    # Prepare data
    df_logit = df.dropna(subset=['enroll_attend_May2010', 'treatment'] + all_controls).copy()

    controls_str = " + ".join(all_controls)
    treat_vars = " + ".join([f'treat_{i}' for i in [1, 2, 3, 4]])
    logit_formula = f"enroll_attend_May2010 ~ {treat_vars} + {controls_str}"

    logit_model = smf.logit(logit_formula, data=df_logit).fit(disp=0, maxiter=100)

    for treat_level in [1, 2, 3, 4]:
        var_name = f'treat_{treat_level}'
        add_result(
            spec_id=f'discrete/logit_treat{treat_level}',
            spec_tree_path='methods/discrete_choice.md',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(logit_model.params[var_name]),
            se=float(logit_model.bse[var_name]),
            tstat=float(logit_model.tvalues[var_name]),
            pval=float(logit_model.pvalues[var_name]),
            n_obs=int(logit_model.nobs), r2=float(logit_model.prsquared),
            sample_desc='Full sample',
            fixed_effects='None',
            controls_desc='All controls (no FE in logit)',
            cluster_var='None',
            model_type='Logit'
        )
    print("  Logit model: done")
except Exception as e:
    print(f"  Error with logit: {e}")

# Probit model
try:
    probit_model = smf.probit(logit_formula, data=df_logit).fit(disp=0, maxiter=100)

    for treat_level in [1, 2, 3, 4]:
        var_name = f'treat_{treat_level}'
        add_result(
            spec_id=f'discrete/probit_treat{treat_level}',
            spec_tree_path='methods/discrete_choice.md',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}',
            coef=float(probit_model.params[var_name]),
            se=float(probit_model.bse[var_name]),
            tstat=float(probit_model.tvalues[var_name]),
            pval=float(probit_model.pvalues[var_name]),
            n_obs=int(probit_model.nobs), r2=float(probit_model.prsquared),
            sample_desc='Full sample',
            fixed_effects='None',
            controls_desc='All controls (no FE in probit)',
            cluster_var='None',
            model_type='Probit'
        )
    print("  Probit model: done")
except Exception as e:
    print(f"  Error with probit: {e}")

print()

# ============================================================================
# INTERACTION EFFECTS
# ============================================================================

print("Running interaction specifications...")

# Gender x Treatment interaction
try:
    # Create interaction terms manually
    for treat_level in [1, 2, 3, 4]:
        df[f'treat_{treat_level}_x_girl'] = df[f'treat_{treat_level}'] * df['girl']

    interact_vars = " + ".join([f'treat_{i}_x_girl' for i in [1, 2, 3, 4]])
    formula_interact = (f"enroll_attend_May2010 ~ C(treatment) + girl + {interact_vars} + " +
                       " + ".join([c for c in all_controls if c != 'girl']) + " | stratum")

    model_interact = pf.feols(formula_interact, data=df, vcov={'CRV1': 'schoolid'},
                              weights='weight_hh')

    # Main effects
    for treat_level in [1, 2, 3, 4]:
        coef_name = f'C(treatment)[T.{treat_level}]'
        add_result(
            spec_id=f'ols/interact/gender_main_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}_main',
            coef=float(model_interact.coef()[coef_name]),
            se=float(model_interact.se()[coef_name]),
            tstat=float(model_interact.tstat()[coef_name]),
            pval=float(model_interact.pvalue()[coef_name]),
            n_obs=int(model_interact._N), r2=float(model_interact._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='All controls + gender interactions',
            cluster_var='schoolid',
            model_type='WLS'
        )

    # Interaction effects
    for treat_level in [1, 2, 3, 4]:
        interact_name = f'treat_{treat_level}_x_girl'
        add_result(
            spec_id=f'ols/interact/gender_interact_treat{treat_level}',
            spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
            outcome_var='enroll_attend_May2010',
            treatment_var=f'treatment_{treat_level}_x_girl',
            coef=float(model_interact.coef()[interact_name]),
            se=float(model_interact.se()[interact_name]),
            tstat=float(model_interact.tstat()[interact_name]),
            pval=float(model_interact.pvalue()[interact_name]),
            n_obs=int(model_interact._N), r2=float(model_interact._r2),
            sample_desc='Full sample',
            fixed_effects='stratum',
            controls_desc='All controls + gender interactions',
            cluster_var='schoolid',
            model_type='WLS'
        )

    print("  Gender x Treatment interaction: done")
except Exception as e:
    print(f"  Error with gender interaction: {e}")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("=" * 60)
print("Saving results...")
print("=" * 60)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(OUTPUT_DIR / "specification_results.csv", index=False)
print(f"Saved {len(results_df)} specifications to {OUTPUT_DIR / 'specification_results.csv'}")

# Summary statistics
if len(results_df) > 0:
    print()
    print("Summary Statistics:")
    print(f"  Total specifications: {len(results_df)}")
    print(f"  Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"  Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"  Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"  Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"  Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"  Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
else:
    print("  No results generated!")

print()
print("Done!")
