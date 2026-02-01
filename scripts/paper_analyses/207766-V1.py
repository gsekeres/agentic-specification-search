"""
Specification Search for Paper 207766-V1
"Organized Voters: Elections and Public Funding of Nonprofits" by Camille Urvoy

Method: Regression Discontinuity Design (RD)
Running variable: alignm (alignment margin - win margin of ruling party candidate)
Cutoff: 0
Outcome: amount_pcap, amount_congr1_3_pcap, amount_congr2_3_pcap, amount_congr3_3_pcap
         (governmental transfers to organizations per capita)

Hypothesis: Political alignment affects government funding to nonprofits
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import RD packages
from rdrobust import rdrobust, rdbwselect
from rddensity import rddensity

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/207766-V1/urvoy_organized_replication"
DATA_PATH = f"{BASE_PATH}/processed_data/data_ready.dta"
OUTPUT_PATH = f"{BASE_PATH}/../specification_results.csv"

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH, convert_categoricals=False)
print(f"Total observations: {len(df)}")

# Filter to analysis sample
df_sample = df[df['insample'] == 1].copy()
print(f"Analysis sample (insample==1): {len(df_sample)}")

# Define control variable sets (from original Stata code)
CONTROLS_DEMO = ['sh0014_all', 'sh1529_all', 'sh60p_all', 'shpcs34_all', 'shpcs56_all',
                 'shpcs7_all', 'sh_heduc_all', 'sh_chom_all', 'rp_prop_all', 'rp_lochlmv_all',
                 'ln_revmoy', 'ln_median_income_all']

CONTROLS_ELEC = ['nb_candi_1rd_all', 'nb_candi_fleft_1rd_all', 'nb_candi_green_1rd_all',
                 'nb_candi_left_1rd_all', 'nb_candi_center_1rd_all', 'nb_candi_right_1rd_all',
                 'nb_candi_fright_1rd_all', 'nb_candi_other_1rd_all', 'sh_abs_1rd_all',
                 'tot_left_share_1rd_all', 'tot_right_share_1rd_all']

CONTROLS_ASSO = ['ln_nb_all', 'shnb_employer_all', 'shnb_unique_com_all', 'shnb_unique_dep_all',
                 'shnb_crea1900_all', 'shnb_crea1910_all', 'shnb_crea1920_all', 'shnb_crea1930_all',
                 'shnb_crea1940_all', 'shnb_crea1950_all', 'shnb_crea1960_all', 'shnb_crea1970_all',
                 'shnb_crea1980_all', 'shnb_crea1990_all', 'shnb_crea2000_all', 'shnb_crea2010_all']

ALL_CONTROLS = CONTROLS_DEMO + CONTROLS_ELEC + CONTROLS_ASSO

# Outcome variables
OUTCOMES = {
    'amount_congr1_3_pcap': 'Congruent organizations',
    'amount_congr2_3_pcap': 'Moderate organizations',
    'amount_congr3_3_pcap': 'Non-congruent organizations',
    'amount_pcap': 'All organizations'
}

# Running variable
RUNNING_VAR = 'alignm'
CUTOFF = 0
CLUSTER_VAR = 'comn'  # municipality identifier

# Results storage
results = []

def run_rd_specification(y, x, c=0, covs=None, p=1, kernel='triangular',
                         cluster=None, h=None, spec_id='', spec_tree_path='',
                         outcome_var='', additional_info=None):
    """
    Run a single RD specification and record results.
    """
    try:
        # Clean data
        mask = ~(np.isnan(y) | np.isnan(x))
        y_clean = y[mask].values
        x_clean = x[mask].values

        if cluster is not None:
            cluster_clean = cluster[mask].values
        else:
            cluster_clean = None

        if covs is not None:
            covs_clean = covs[mask].values
        else:
            covs_clean = None

        # Run rdrobust
        result = rdrobust(y_clean, x_clean, c=c, p=p, kernel=kernel,
                         cluster=cluster_clean, covs=covs_clean, h=h)

        # Extract results - rdrobust returns DataFrames with index labels
        # Convention: row 0 = Conventional, row 1 = Bias-Corrected, row 2 = Robust
        coef = float(result.coef.iloc[0])  # Conventional coefficient
        coef_bc = float(result.coef.iloc[1])  # Bias-corrected coefficient
        se = float(result.se.iloc[0])  # Conventional SE
        se_robust = float(result.se.iloc[2])  # Robust SE
        pval = float(result.pv.iloc[0])  # Conventional p-value
        pval_robust = float(result.pv.iloc[2])  # Robust p-value

        # Bandwidth
        bw_left = float(result.bws.iloc[0, 0])
        bw_right = float(result.bws.iloc[0, 1])

        # Sample sizes
        n_left = int(result.N_h[0])
        n_right = int(result.N_h[1])
        n_total = n_left + n_right

        # Compute mean left of cutoff (for comparison)
        mean_left = y_clean[(x_clean >= -2) & (x_clean < 0)].mean() if len(y_clean[(x_clean >= -2) & (x_clean < 0)]) > 0 else np.nan

        # Build coefficient vector JSON
        coef_vector = {
            'treatment': {
                'var': 'above_cutoff',
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval),
                'pval_robust': float(pval_robust),
                'ci_lower': float(coef - 1.96 * se),
                'ci_upper': float(coef + 1.96 * se),
                'ci_type': 'conventional'
            },
            'running_variable': {
                'var': RUNNING_VAR,
                'cutoff': c,
                'bandwidth_left': float(bw_left),
                'bandwidth_right': float(bw_right)
            },
            'diagnostics': {
                'polynomial_order': p,
                'kernel': kernel,
                'n_left': int(n_left),
                'n_right': int(n_right),
                'mean_left_of_cutoff': float(mean_left) if not np.isnan(mean_left) else None
            },
            'n_obs': int(n_total),
            'n_effective': int(n_total)
        }

        if additional_info:
            coef_vector['additional_info'] = additional_info

        return {
            'paper_id': '207766-V1',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': 'above_cutoff',
            'running_var': RUNNING_VAR,
            'coef': float(coef),
            'se': float(se),
            'pval': float(pval),
            'pval_robust': float(pval_robust),
            'n_obs': int(n_total),
            'n_left': int(n_left),
            'n_right': int(n_right),
            'bandwidth': float(bw_left),
            'polynomial_order': p,
            'kernel': kernel,
            'mean_left': float(mean_left) if not np.isnan(mean_left) else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'success': True,
            'error_message': None
        }

    except Exception as e:
        return {
            'paper_id': '207766-V1',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': 'above_cutoff',
            'running_var': RUNNING_VAR,
            'coef': None,
            'se': None,
            'pval': None,
            'pval_robust': None,
            'n_obs': None,
            'n_left': None,
            'n_right': None,
            'bandwidth': None,
            'polynomial_order': p,
            'kernel': kernel,
            'mean_left': None,
            'coefficient_vector_json': None,
            'success': False,
            'error_message': str(e)
        }

# ============================================================================
# 1. BASELINE REPLICATION
# ============================================================================
print("\n" + "="*70)
print("1. BASELINE REPLICATION")
print("="*70)

for outcome, label in OUTCOMES.items():
    print(f"\n  Running baseline for: {label}")
    y = df_sample[outcome]
    x = df_sample[RUNNING_VAR]
    cluster = df_sample[CLUSTER_VAR]

    result = run_rd_specification(
        y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
        cluster=cluster, covs=None,
        spec_id='baseline',
        spec_tree_path='methods/regression_discontinuity.md#baseline',
        outcome_var=outcome,
        additional_info={'label': label, 'controls': 'none'}
    )
    results.append(result)
    if result['success']:
        print(f"    Coef: {result['coef']:.4f}, SE: {result['se']:.4f}, p-value (robust): {result['pval_robust']:.4f}, N: {result['n_obs']}, BW: {result['bandwidth']:.2f}")
    else:
        print(f"    FAILED: {result['error_message']}")

# ============================================================================
# 2. RD DESIGN SPECIFICATIONS
# ============================================================================
print("\n" + "="*70)
print("2. RD DESIGN SPECIFICATIONS")
print("="*70)

# 2.1 Bandwidth variations
print("\n2.1 Bandwidth Variations")
for outcome, label in OUTCOMES.items():
    y = df_sample[outcome]
    x = df_sample[RUNNING_VAR]
    cluster = df_sample[CLUSTER_VAR]

    # Get optimal bandwidth first
    try:
        bw_result = rdbwselect(y.dropna().values, x.dropna().values, c=CUTOFF)
        opt_bw = bw_result.bws.iloc[0, 0] if hasattr(bw_result.bws, 'iloc') else bw_result.bws[0, 0]
    except:
        opt_bw = 15  # fallback

    bw_specs = [
        ('rd/bandwidth/half_optimal', opt_bw / 2, 'Half optimal bandwidth'),
        ('rd/bandwidth/double_optimal', opt_bw * 2, 'Double optimal bandwidth'),
        ('rd/bandwidth/fixed_10', 10, 'Fixed 10pp bandwidth'),
        ('rd/bandwidth/fixed_15', 15, 'Fixed 15pp bandwidth'),
        ('rd/bandwidth/fixed_20', 20, 'Fixed 20pp bandwidth'),
    ]

    for spec_id, bw, desc in bw_specs:
        result = run_rd_specification(
            y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
            cluster=cluster, covs=None, h=bw,
            spec_id=spec_id,
            spec_tree_path='methods/regression_discontinuity.md#bandwidth-selection',
            outcome_var=outcome,
            additional_info={'label': label, 'bandwidth_type': desc}
        )
        results.append(result)

# 2.2 Polynomial order variations
print("\n2.2 Polynomial Order Variations")
for outcome, label in OUTCOMES.items():
    y = df_sample[outcome]
    x = df_sample[RUNNING_VAR]
    cluster = df_sample[CLUSTER_VAR]

    poly_specs = [
        ('rd/poly/local_quadratic', 2, 'Local quadratic'),
        ('rd/poly/local_cubic', 3, 'Local cubic'),
    ]

    for spec_id, p_order, desc in poly_specs:
        result = run_rd_specification(
            y=y, x=x, c=CUTOFF, p=p_order, kernel='triangular',
            cluster=cluster, covs=None,
            spec_id=spec_id,
            spec_tree_path='methods/regression_discontinuity.md#polynomial-order',
            outcome_var=outcome,
            additional_info={'label': label, 'poly_type': desc}
        )
        results.append(result)

# 2.3 Kernel variations
print("\n2.3 Kernel Variations")
for outcome, label in OUTCOMES.items():
    y = df_sample[outcome]
    x = df_sample[RUNNING_VAR]
    cluster = df_sample[CLUSTER_VAR]

    kernel_specs = [
        ('rd/kernel/uniform', 'uniform', 'Uniform kernel'),
        ('rd/kernel/epanechnikov', 'epanechnikov', 'Epanechnikov kernel'),
    ]

    for spec_id, kern, desc in kernel_specs:
        result = run_rd_specification(
            y=y, x=x, c=CUTOFF, p=1, kernel=kern,
            cluster=cluster, covs=None,
            spec_id=spec_id,
            spec_tree_path='methods/regression_discontinuity.md#kernel-function',
            outcome_var=outcome,
            additional_info={'label': label, 'kernel_type': desc}
        )
        results.append(result)

# 2.4 With controls
print("\n2.4 With Controls")
# Filter to sample with controls available
df_controls = df_sample[df_sample['insample_controls'] == 1].copy()
print(f"  Sample with controls: {len(df_controls)}")

for outcome, label in OUTCOMES.items():
    y = df_controls[outcome]
    x = df_controls[RUNNING_VAR]
    cluster = df_controls[CLUSTER_VAR]

    # Prepare covariates - drop any with missing values
    available_controls = [c for c in ALL_CONTROLS if c in df_controls.columns]
    covs_df = df_controls[available_controls].copy()

    # Drop columns with too many missing values
    covs_df = covs_df.dropna(axis=1, thresh=int(0.5*len(covs_df)))

    result = run_rd_specification(
        y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
        cluster=cluster, covs=covs_df,
        spec_id='rd/controls/full',
        spec_tree_path='methods/regression_discontinuity.md#control-sets',
        outcome_var=outcome,
        additional_info={'label': label, 'controls': 'all_available'}
    )
    results.append(result)

# ============================================================================
# 3. SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*70)
print("3. SAMPLE RESTRICTIONS")
print("="*70)

# 3.1 Left-right races only
print("\n3.1 Left-Right Races Only")
if 'insample_leftright' in df_sample.columns:
    df_lr = df_sample[df_sample['insample_leftright'] == 1].copy()
    print(f"  Left-right races sample: {len(df_lr)}")

    for outcome, label in OUTCOMES.items():
        y = df_lr[outcome]
        x = df_lr[RUNNING_VAR]
        cluster = df_lr[CLUSTER_VAR]

        result = run_rd_specification(
            y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
            cluster=cluster, covs=None,
            spec_id='rd/sample/left_right_only',
            spec_tree_path='methods/regression_discontinuity.md#sample-restrictions',
            outcome_var=outcome,
            additional_info={'label': label, 'sample': 'left_right_races'}
        )
        results.append(result)

# 3.2 Donut hole (exclude observations very close to cutoff)
print("\n3.2 Donut Hole")
donut_sizes = [0.5, 1, 2]
for donut in donut_sizes:
    df_donut = df_sample[(abs(df_sample[RUNNING_VAR]) > donut) | (df_sample[RUNNING_VAR].isna())].copy()
    print(f"  Donut (exclude |margin| < {donut}): {len(df_donut)}")

    for outcome, label in OUTCOMES.items():
        y = df_donut[outcome]
        x = df_donut[RUNNING_VAR]
        cluster = df_donut[CLUSTER_VAR]

        result = run_rd_specification(
            y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
            cluster=cluster, covs=None,
            spec_id=f'rd/sample/donut_{donut}',
            spec_tree_path='methods/regression_discontinuity.md#sample-restrictions',
            outcome_var=outcome,
            additional_info={'label': label, 'donut_size': donut}
        )
        results.append(result)

# ============================================================================
# 4. VALIDATION TESTS
# ============================================================================
print("\n" + "="*70)
print("4. VALIDATION TESTS")
print("="*70)

# 4.1 McCrary Density Test
print("\n4.1 McCrary Density Test")
try:
    x_clean = df_sample[RUNNING_VAR].dropna().values
    density_result = rddensity(x_clean, c=CUTOFF)

    # Extract values based on rddensity output structure
    # rddensity returns test statistics as attributes
    test_pval = float(density_result.test.iloc[0])  # p-value

    mccrary_result = {
        'paper_id': '207766-V1',
        'spec_id': 'rd/validity/density',
        'spec_tree_path': 'methods/regression_discontinuity.md#validation-tests',
        'outcome_var': 'density_test',
        'treatment_var': None,
        'running_var': RUNNING_VAR,
        'coef': None,
        'se': None,
        'pval': test_pval,
        'pval_robust': test_pval,
        'n_obs': len(x_clean),
        'n_left': int(density_result.N.iloc[0]),
        'n_right': int(density_result.N.iloc[1]),
        'bandwidth': float(density_result.h.iloc[0]),
        'polynomial_order': None,
        'kernel': None,
        'mean_left': None,
        'coefficient_vector_json': json.dumps({
            'test_type': 'mccrary_density',
            'p_value': test_pval,
            'n_left': int(density_result.N.iloc[0]),
            'n_right': int(density_result.N.iloc[1])
        }),
        'success': True,
        'error_message': None
    }
    results.append(mccrary_result)
    print(f"  Density test p-value: {test_pval:.4f}")
except Exception as e:
    print(f"  McCrary test failed: {e}")

# 4.2 Placebo cutoffs
print("\n4.2 Placebo Cutoffs")
placebo_cutoffs = [-10, -5, 5, 10]
for pc in placebo_cutoffs:
    for outcome, label in OUTCOMES.items():
        y = df_sample[outcome]
        x = df_sample[RUNNING_VAR]
        cluster = df_sample[CLUSTER_VAR]

        result = run_rd_specification(
            y=y, x=x, c=pc, p=1, kernel='triangular',
            cluster=cluster, covs=None,
            spec_id=f'rd/validity/placebo_cutoff_{pc}',
            spec_tree_path='methods/regression_discontinuity.md#validation-tests',
            outcome_var=outcome,
            additional_info={'label': label, 'placebo_cutoff': pc}
        )
        results.append(result)

# ============================================================================
# 5. CLUSTERING VARIATIONS
# ============================================================================
print("\n" + "="*70)
print("5. CLUSTERING VARIATIONS")
print("="*70)

print("\n5.1 No Clustering (Robust SE)")
for outcome, label in OUTCOMES.items():
    y = df_sample[outcome]
    x = df_sample[RUNNING_VAR]

    result = run_rd_specification(
        y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
        cluster=None, covs=None,
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var=outcome,
        additional_info={'label': label, 'clustering': 'none'}
    )
    results.append(result)

# Note: The original paper clusters by municipality (comn), which is already done in baseline

# ============================================================================
# 6. LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================
print("\n" + "="*70)
print("6. LEAVE-ONE-OUT ROBUSTNESS (Controls)")
print("="*70)

# Use controls sample
df_controls = df_sample[df_sample['insample_controls'] == 1].copy()

# Select subset of most important controls for LOO
LOO_CONTROLS = ['sh0014_all', 'sh1529_all', 'sh60p_all', 'sh_heduc_all', 'sh_chom_all',
                'ln_revmoy', 'tot_left_share_1rd_all', 'ln_nb_all']

for outcome in ['amount_congr1_3_pcap', 'amount_pcap']:  # Run for two key outcomes
    label = OUTCOMES[outcome]
    y = df_controls[outcome]
    x = df_controls[RUNNING_VAR]
    cluster = df_controls[CLUSTER_VAR]

    available_controls = [c for c in LOO_CONTROLS if c in df_controls.columns and df_controls[c].notna().sum() > len(df_controls) * 0.5]

    for drop_var in available_controls:
        # Create covariates excluding dropped variable
        loo_controls = [c for c in available_controls if c != drop_var]
        covs_df = df_controls[loo_controls].copy()

        result = run_rd_specification(
            y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
            cluster=cluster, covs=covs_df,
            spec_id=f'robust/loo/drop_{drop_var}',
            spec_tree_path='robustness/leave_one_out.md',
            outcome_var=outcome,
            additional_info={'label': label, 'dropped_variable': drop_var}
        )
        results.append(result)

# ============================================================================
# 7. SINGLE COVARIATE SPECIFICATIONS
# ============================================================================
print("\n" + "="*70)
print("7. SINGLE COVARIATE SPECIFICATIONS")
print("="*70)

SINGLE_COVS = ['sh_heduc_all', 'sh_chom_all', 'ln_revmoy', 'tot_left_share_1rd_all', 'ln_nb_all']

for outcome in ['amount_congr1_3_pcap', 'amount_pcap']:
    label = OUTCOMES[outcome]
    y = df_controls[outcome]
    x = df_controls[RUNNING_VAR]
    cluster = df_controls[CLUSTER_VAR]

    for cov_var in SINGLE_COVS:
        if cov_var in df_controls.columns:
            covs_df = df_controls[[cov_var]].copy()

            result = run_rd_specification(
                y=y, x=x, c=CUTOFF, p=1, kernel='triangular',
                cluster=cluster, covs=covs_df,
                spec_id=f'robust/single_cov/{cov_var}',
                spec_tree_path='robustness/single_covariate.md',
                outcome_var=outcome,
                additional_info={'label': label, 'single_covariate': cov_var}
            )
            results.append(result)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df['timestamp'] = datetime.now().isoformat()

# Save to CSV
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print(f"Total specifications run: {len(results_df)}")
print(f"Successful specifications: {results_df['success'].sum()}")
print(f"Failed specifications: {(~results_df['success']).sum()}")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
success_df = results_df[results_df['success'] == True]
if len(success_df) > 0:
    print("\nBaseline Results (by outcome):")
    baseline = success_df[success_df['spec_id'] == 'baseline']
    for _, row in baseline.iterrows():
        sig = '*' if row['pval_robust'] < 0.05 else ''
        print(f"  {row['outcome_var']}: coef={row['coef']:.4f}{sig}, se={row['se']:.4f}, p={row['pval_robust']:.4f}")

    print("\nCoefficient Range by Outcome (all specs):")
    for outcome in OUTCOMES.keys():
        outcome_df = success_df[success_df['outcome_var'] == outcome]
        if len(outcome_df) > 0:
            print(f"  {outcome}:")
            print(f"    Range: [{outcome_df['coef'].min():.4f}, {outcome_df['coef'].max():.4f}]")
            print(f"    Mean: {outcome_df['coef'].mean():.4f}, Median: {outcome_df['coef'].median():.4f}")

print("\n" + "="*70)
print("SPECIFICATION SEARCH COMPLETE")
print("="*70)
