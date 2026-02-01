"""
Specification Search: 114707-V1
Hospital Mergers and Hospital Prices

This script runs a systematic specification search on the hospital merger study,
testing robustness of the main finding that hospital mergers increase prices.

Method: Difference-in-Differences with hospital and year fixed effects
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_ID = "114707-V1"
PAPER_TITLE = "Do Hospital Mergers Reduce Costs? Evidence on Prices and Hospital Market Consolidation"
JOURNAL = "AEA"

# File paths - using relative path from project root
DATA_PATH = 'data/downloads/extracted/114707-V1/data/HospitalMMC_Data.dta'
OUTPUT_PATH = 'data/downloads/extracted/114707-V1/specification_results.csv'

# Key variables
OUTCOME_VAR = 'lnprnonmed'
TREATMENT_VAR = 'post'
CONTROLS = ['lncmi', 'pctmcaid', 'lnbeds', 'fp', 'hhi', 'sysoth']
UNIT_FE = 'h'
TIME_FE = 'year'
CLUSTER_VAR = 'h'
WEIGHT_VAR = 'dis_tot'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_results(model, spec_id, spec_tree_path, treatment_var, outcome_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
                   n_obs_override=None):
    """Extract standardized results from pyfixest model."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        pval = model.pvalue()[treatment_var]
        tstat = model.tstat()[treatment_var]
        conf_int = model.confint()
        ci_lower = conf_int.loc[treatment_var, '2.5%']
        ci_upper = conf_int.loc[treatment_var, '97.5%']
        n_obs = n_obs_override if n_obs_override else model._N
        r2 = model._r2 if hasattr(model, '_r2') else None

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
            "diagnostics": {}
        }

        # Add control coefficients
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
            'n_obs': int(n_obs),
            'r_squared': float(r2) if r2 else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    # Load data
    print("Loading data...")
    df = pd.read_stata(DATA_PATH)

    # Convert year and h to int for pyfixest
    df['year'] = df['year'].astype(int)
    df['h'] = df['h'].astype(int)

    # Results storage
    results = []

    # =============================================================================
    # BASELINE SPECIFICATION (Table 2, Column 2 - all controls)
    # =============================================================================
    print("Running baseline specification...")

    # Prepare baseline sample (drop indirect==1)
    df_baseline = df[df['indirect'] != 1].copy()
    df_baseline = df_baseline.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR] + CONTROLS)

    # Baseline formula
    controls_str = ' + '.join(CONTROLS)
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"

    model_baseline = pf.feols(formula, data=df_baseline, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    print(model_baseline.summary())

    result = extract_results(
        model_baseline,
        'baseline',
        'methods/difference_in_differences.md#baseline',
        TREATMENT_VAR, OUTCOME_VAR,
        'All hospitals except indirect mergers',
        'Hospital, Year',
        f'Controls: {", ".join(CONTROLS)}',
        CLUSTER_VAR,
        'OLS with Hospital and Year FE'
    )
    if result:
        results.append(result)

    # =============================================================================
    # DiD FIXED EFFECTS VARIATIONS
    # =============================================================================
    print("\nRunning FE variations...")

    # did/fe/unit_only
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE}"
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/fe/unit_only', 'methods/difference_in_differences.md#fixed-effects',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital only',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Hospital FE')
    if result: results.append(result)

    # did/fe/time_only
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {TIME_FE}"
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/fe/time_only', 'methods/difference_in_differences.md#fixed-effects',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Year only',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Year FE')
    if result: results.append(result)

    # did/fe/twoway (same as baseline)
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/fe/twoway', 'methods/difference_in_differences.md#fixed-effects',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # did/fe/none (pooled OLS)
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str}"
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/fe/none', 'methods/difference_in_differences.md#fixed-effects',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'None',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'Pooled OLS')
    if result: results.append(result)

    # =============================================================================
    # DiD CONTROL SET VARIATIONS
    # =============================================================================
    print("\nRunning control set variations...")

    # did/controls/none
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} | {UNIT_FE} + {TIME_FE}"
    df_nocontrols = df[df['indirect'] != 1].dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR])
    model = pf.feols(formula, data=df_nocontrols, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/controls/none', 'methods/difference_in_differences.md#control-sets',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            'No controls', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # did/controls/minimal (just lnbeds and lncmi)
    minimal_controls = ['lnbeds', 'lncmi']
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {' + '.join(minimal_controls)} | {UNIT_FE} + {TIME_FE}"
    df_minimal = df[df['indirect'] != 1].dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR] + minimal_controls)
    model = pf.feols(formula, data=df_minimal, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/controls/minimal', 'methods/difference_in_differences.md#control-sets',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            f'Minimal: {", ".join(minimal_controls)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # did/controls/full (same as baseline for this paper)
    result = extract_results(model_baseline, 'did/controls/full', 'methods/difference_in_differences.md#control-sets',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            f'Full: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # =============================================================================
    # DiD SAMPLE RESTRICTIONS
    # =============================================================================
    print("\nRunning sample restrictions...")

    # did/sample/matched (optmatch==1) - Table 2 matched controls
    df_matched = df[(df['indirect'] != 1) & (df['optmatch'] == 1)].copy()
    df_matched = df_matched.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR] + CONTROLS)
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_matched, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/sample/matched', 'methods/difference_in_differences.md#sample-restrictions',
                            TREATMENT_VAR, OUTCOME_VAR, 'Matched controls only', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # did/sample/same_system (samesysctrl==1) - Table 4
    df_samesys = df[(df['indirect'] != 1) & (df['samesysctrl'] == 1)].copy()
    df_samesys = df_samesys.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR] + CONTROLS)
    model = pf.feols(formula, data=df_samesys, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/sample/same_system', 'methods/difference_in_differences.md#sample-restrictions',
                            TREATMENT_VAR, OUTCOME_VAR, 'Same-system controls', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # did/sample/early_period (first half 1996-2005)
    df_early = df_baseline[df_baseline['year'] <= 2005].copy()
    model = pf.feols(formula, data=df_early, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/sample/early_period', 'robustness/sample_restrictions.md#time-based',
                            TREATMENT_VAR, OUTCOME_VAR, '1996-2005', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # did/sample/late_period (second half 2006-2014)
    df_late = df_baseline[df_baseline['year'] >= 2006].copy()
    model = pf.feols(formula, data=df_late, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/sample/late_period', 'robustness/sample_restrictions.md#time-based',
                            TREATMENT_VAR, OUTCOME_VAR, '2006-2014', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # =============================================================================
    # DiD DYNAMIC EFFECTS (EVENT STUDY)
    # =============================================================================
    print("\nRunning event study specifications...")

    # Event study with leads and lags
    event_vars = ['_m4', '_m3', '_m2', '_p0', '_p1', '_p2', '_p3', '_p4']
    df_event = df[df['indirect'] != 1].copy()
    df_event = df_event.dropna(subset=[OUTCOME_VAR, WEIGHT_VAR] + event_vars + CONTROLS)

    event_str = ' + '.join(event_vars)
    formula = f"{OUTCOME_VAR} ~ {event_str} + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_event, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)

    # Record each event coefficient
    for ev in event_vars:
        try:
            coef = model.coef()[ev]
            se = model.se()[ev]
            pval = model.pvalue()[ev]
            tstat = model.tstat()[ev]
            conf_int = model.confint()
            ci_lower = conf_int.loc[ev, '2.5%']
            ci_upper = conf_int.loc[ev, '97.5%']

            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'did/dynamic/event_{ev}',
                'spec_tree_path': 'methods/difference_in_differences.md#dynamic-effects',
                'outcome_var': OUTCOME_VAR,
                'treatment_var': ev,
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_obs': int(model._N),
                'r_squared': float(model._r2) if model._r2 else None,
                'coefficient_vector_json': json.dumps({"event_var": ev, "coef": float(coef), "se": float(se)}),
                'sample_desc': 'All except indirect',
                'fixed_effects': 'Hospital, Year',
                'controls_desc': f'Controls: {", ".join(CONTROLS)}',
                'cluster_var': CLUSTER_VAR,
                'model_type': 'Event Study',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
        except Exception as e:
            print(f"Error with event var {ev}: {e}")

    # =============================================================================
    # ALTERNATIVE OUTCOME: lnprmed (Medical Prices)
    # =============================================================================
    print("\nRunning alternative outcome (medical prices)...")

    df_med = df[df['indirect'] != 1].copy()
    df_med = df_med.dropna(subset=['lnprmed', TREATMENT_VAR, WEIGHT_VAR] + CONTROLS)
    formula = f"lnprmed ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_med, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'did/outcome/lnprmed', 'methods/difference_in_differences.md',
                            TREATMENT_VAR, 'lnprmed', 'All except indirect', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # =============================================================================
    # HETEROGENEITY: Active vs Passive Mergers
    # =============================================================================
    print("\nRunning heterogeneity by merger type...")

    # Active vs Passive
    df_hetero = df[df['indirect'] != 1].copy()
    df_hetero = df_hetero.dropna(subset=[OUTCOME_VAR, 'post_active', 'post_passive', WEIGHT_VAR] + CONTROLS)
    formula = f"{OUTCOME_VAR} ~ post_active + post_passive + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_hetero, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)

    for treat_var in ['post_active', 'post_passive']:
        try:
            result = extract_results(model, f'did/hetero/{treat_var}', 'methods/difference_in_differences.md',
                                    treat_var, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                                    f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
            if result: results.append(result)
        except Exception as e:
            print(f"Error with {treat_var}: {e}")

    # In-state vs Out-of-state
    df_hetero2 = df[df['indirect'] != 1].copy()
    df_hetero2 = df_hetero2.dropna(subset=[OUTCOME_VAR, 'post_instate', 'post_outstate', WEIGHT_VAR] + CONTROLS)
    formula = f"{OUTCOME_VAR} ~ post_instate + post_outstate + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_hetero2, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)

    for treat_var in ['post_instate', 'post_outstate']:
        try:
            result = extract_results(model, f'did/hetero/{treat_var}', 'methods/difference_in_differences.md',
                                    treat_var, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                                    f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
            if result: results.append(result)
        except Exception as e:
            print(f"Error with {treat_var}: {e}")

    # =============================================================================
    # ROBUSTNESS: LEAVE-ONE-OUT
    # =============================================================================
    print("\nRunning leave-one-out robustness...")

    for control in CONTROLS:
        remaining = [c for c in CONTROLS if c != control]
        df_loo = df[df['indirect'] != 1].copy()
        df_loo = df_loo.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR] + remaining)

        formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {' + '.join(remaining)} | {UNIT_FE} + {TIME_FE}"
        model = pf.feols(formula, data=df_loo, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
        result = extract_results(model, f'robust/loo/drop_{control}', 'robustness/leave_one_out.md',
                                TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                                f'Controls: {", ".join(remaining)}', CLUSTER_VAR, 'OLS with Two-way FE')
        if result: results.append(result)

    # =============================================================================
    # ROBUSTNESS: SINGLE COVARIATE
    # =============================================================================
    print("\nRunning single covariate robustness...")

    # Bivariate (no controls)
    df_biv = df[df['indirect'] != 1].dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR])
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_biv, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'robust/single/none', 'robustness/single_covariate.md',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            'No controls', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # Single covariate
    for control in CONTROLS:
        df_single = df[df['indirect'] != 1].copy()
        df_single = df_single.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, WEIGHT_VAR, control])

        formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {control} | {UNIT_FE} + {TIME_FE}"
        model = pf.feols(formula, data=df_single, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
        result = extract_results(model, f'robust/single/{control}', 'robustness/single_covariate.md',
                                TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                                f'Control: {control}', CLUSTER_VAR, 'OLS with Two-way FE')
        if result: results.append(result)

    # =============================================================================
    # ROBUSTNESS: CLUSTERING VARIATIONS
    # =============================================================================
    print("\nRunning clustering variations...")

    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"

    # Robust (no clustering)
    model = pf.feols(formula, data=df_baseline, vcov='hetero', weights=WEIGHT_VAR)
    result = extract_results(model, 'robust/cluster/none', 'robustness/clustering_variations.md#single-level',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', 'none (robust)', 'OLS with Two-way FE')
    if result: results.append(result)

    # Cluster by year
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': 'year'}, weights=WEIGHT_VAR)
    result = extract_results(model, 'robust/cluster/year', 'robustness/clustering_variations.md#single-level',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', 'year', 'OLS with Two-way FE')
    if result: results.append(result)

    # =============================================================================
    # ROBUSTNESS: UNWEIGHTED
    # =============================================================================
    print("\nRunning unweighted specification...")

    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': CLUSTER_VAR})
    result = extract_results(model, 'robust/form/unweighted', 'robustness/functional_form.md',
                            TREATMENT_VAR, OUTCOME_VAR, 'All except indirect (unweighted)', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE (unweighted)')
    if result: results.append(result)

    # =============================================================================
    # ROBUSTNESS: SAMPLE TRIMMING
    # =============================================================================
    print("\nRunning sample trimming...")

    # Trim outcome at 1%/99%
    q01 = df_baseline[OUTCOME_VAR].quantile(0.01)
    q99 = df_baseline[OUTCOME_VAR].quantile(0.99)
    df_trim = df_baseline[(df_baseline[OUTCOME_VAR] >= q01) & (df_baseline[OUTCOME_VAR] <= q99)].copy()

    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | {UNIT_FE} + {TIME_FE}"
    model = pf.feols(formula, data=df_trim, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
    result = extract_results(model, 'robust/sample/trim_1pct', 'robustness/sample_restrictions.md#outliers',
                            TREATMENT_VAR, OUTCOME_VAR, 'Trimmed 1%/99%', 'Hospital, Year',
                            f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
    if result: results.append(result)

    # =============================================================================
    # ROBUSTNESS: BALANCED PANEL
    # =============================================================================
    print("\nRunning balanced panel...")

    # Find hospitals observed in all years
    obs_per_hosp = df_baseline.groupby('h').size()
    max_obs = obs_per_hosp.max()
    balanced_hosps = obs_per_hosp[obs_per_hosp == max_obs].index
    df_balanced = df_baseline[df_baseline['h'].isin(balanced_hosps)].copy()

    if len(df_balanced) > 100:
        model = pf.feols(formula, data=df_balanced, vcov={'CRV1': CLUSTER_VAR}, weights=WEIGHT_VAR)
        result = extract_results(model, 'robust/sample/balanced', 'robustness/sample_restrictions.md#panel',
                                TREATMENT_VAR, OUTCOME_VAR, 'Balanced panel only', 'Hospital, Year',
                                f'Controls: {", ".join(CONTROLS)}', CLUSTER_VAR, 'OLS with Two-way FE')
        if result: results.append(result)

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================
    print(f"\nTotal specifications run: {len(results)}")

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to: {OUTPUT_PATH}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Filter to main treatment coefficient specs (exclude event study individual coefficients)
    main_specs = results_df[results_df['treatment_var'] == 'post'].copy()

    print(f"Total specifications (main treatment): {len(main_specs)}")
    print(f"Positive coefficients: {(main_specs['coefficient'] > 0).sum()} ({100*(main_specs['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(main_specs['p_value'] < 0.05).sum()} ({100*(main_specs['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(main_specs['p_value'] < 0.01).sum()} ({100*(main_specs['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {main_specs['coefficient'].median():.4f}")
    print(f"Mean coefficient: {main_specs['coefficient'].mean():.4f}")
    print(f"Range: [{main_specs['coefficient'].min():.4f}, {main_specs['coefficient'].max():.4f}]")

    # Print baseline result
    baseline_row = results_df[results_df['spec_id'] == 'baseline'].iloc[0]
    print(f"\nBaseline result:")
    print(f"  Coefficient: {baseline_row['coefficient']:.4f}")
    print(f"  Std Error: {baseline_row['std_error']:.4f}")
    print(f"  P-value: {baseline_row['p_value']:.4f}")
    print(f"  N: {baseline_row['n_obs']}")

    return results_df

if __name__ == "__main__":
    main()
