#!/usr/bin/env python3
"""
Specification Search for Paper 207766-V1
"Organized Voters: Evidence from Governmental Transfers" by Camille Urvoy (AER 2025)

This paper uses a regression discontinuity design to study how government transfers
to nonprofit organizations differ based on political alignment between local and
national governments in France.

Main hypothesis: Municipalities where the ruling party narrowly won receive more
transfers to politically "congruent" (aligned) organizations compared to municipalities
where the ruling party narrowly lost.

Method: Sharp Regression Discontinuity
- Running variable: alignm (win margin of ruling party candidate)
- Cutoff: 0 (alignment = ruling party wins)
- Outcomes: amount transferred per capita, by organization political leaning
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Install rdrobust if needed
try:
    from rdrobust import rdrobust, rdbwselect
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'rdrobust'])
    from rdrobust import rdrobust, rdbwselect

# Paths
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/207766-V1/urvoy_organized_replication/processed_data/data_ready.dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/207766-V1/urvoy_organized_replication"

# Paper metadata
PAPER_ID = "207766-V1"
JOURNAL = "AER"
PAPER_TITLE = "Organized Voters: Evidence from Governmental Transfers"

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Loaded {len(df)} observations")

# Filter to analysis sample
df_analysis = df[df['insample'] == 1].copy()
print(f"Analysis sample: {len(df_analysis)} observations")

# Key variables
RUNNING_VAR = 'alignm'
CUTOFF = 0
CLUSTER_VAR = 'comn'

# Main outcomes (per capita amounts)
MAIN_OUTCOMES = ['amount_congr1_3_pcap', 'amount_congr2_3_pcap', 'amount_congr3_3_pcap', 'amount_pcap']
OUTCOME_LABELS = {
    'amount_congr1_3_pcap': 'Congruent organizations',
    'amount_congr2_3_pcap': 'Moderate organizations',
    'amount_congr3_3_pcap': 'Non-congruent organizations',
    'amount_pcap': 'All organizations'
}

# Control variables from the do file
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

# Results container
results = []

def run_rd_spec(y_data, x_data, cluster_data=None, covs=None, p=1, kernel='triangular',
                bw=None, spec_id='', spec_tree_path='', outcome_var='', sample_desc='',
                controls_desc='', fixed_effects='None'):
    """Run RD specification and record results."""

    # Remove NaN
    mask = ~(y_data.isna() | x_data.isna())
    if covs is not None:
        for cov in covs:
            mask = mask & ~cov.isna()

    y = y_data[mask].values
    x = x_data[mask].values
    cluster = cluster_data[mask].values if cluster_data is not None else None

    if len(y) < 50:
        print(f"  Skipping {spec_id}: too few observations ({len(y)})")
        return None

    try:
        # Prepare covariates
        cov_arr = None
        if covs is not None and len(covs) > 0:
            cov_df = pd.DataFrame({f'c{i}': c[mask].values for i, c in enumerate(covs)})
            cov_arr = cov_df.values

        # Run rdrobust
        if bw is not None:
            rd = rdrobust(y, x, c=CUTOFF, p=p, kernel=kernel, h=bw,
                         covs=cov_arr, cluster=cluster)
        else:
            rd = rdrobust(y, x, c=CUTOFF, p=p, kernel=kernel,
                         covs=cov_arr, cluster=cluster)

        # Extract results
        coef = rd.coef.iloc[0] if hasattr(rd.coef, 'iloc') else rd.coef[0]
        se = rd.se.iloc[0] if hasattr(rd.se, 'iloc') else rd.se[0]
        pval = rd.pv.iloc[0] if hasattr(rd.pv, 'iloc') else rd.pv[0]
        ci_l = rd.ci.iloc[0, 0] if hasattr(rd.ci, 'iloc') else rd.ci[0, 0]
        ci_u = rd.ci.iloc[0, 1] if hasattr(rd.ci, 'iloc') else rd.ci[0, 1]
        bw_l = rd.bws.iloc[0, 0] if hasattr(rd.bws, 'iloc') else rd.bws[0, 0]
        bw_r = rd.bws.iloc[0, 1] if hasattr(rd.bws, 'iloc') else rd.bws[0, 1]
        n_l = int(rd.N_h[0]) if hasattr(rd.N_h, '__len__') else int(rd.N_h)
        n_r = int(rd.N_h[1]) if hasattr(rd.N_h, '__len__') and len(rd.N_h) > 1 else n_l
        n_obs = n_l + n_r

        # Robust bias-corrected results
        coef_rb = rd.coef.iloc[1] if hasattr(rd.coef, 'iloc') and len(rd.coef) > 1 else coef
        pval_rb = rd.pv.iloc[2] if hasattr(rd.pv, 'iloc') and len(rd.pv) > 2 else pval

        # Create coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": "above_cutoff",
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval),
                "ci_lower": float(ci_l),
                "ci_upper": float(ci_u),
                "robust_bias_corrected_coef": float(coef_rb),
                "robust_bias_corrected_pval": float(pval_rb)
            },
            "running_variable": {
                "var": RUNNING_VAR,
                "cutoff": CUTOFF,
                "bandwidth_left": float(bw_l),
                "bandwidth_right": float(bw_r)
            },
            "controls": controls_desc,
            "diagnostics": {
                "polynomial_order": p,
                "kernel": kernel,
                "n_left": n_l,
                "n_right": n_r
            },
            "n_obs": n_obs
        }

        t_stat = coef / se if se > 0 else np.nan

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': f'{RUNNING_VAR} > {CUTOFF}',
            'coefficient': coef,
            'std_error': se,
            't_stat': t_stat,
            'p_value': pval,
            'p_value_robust': pval_rb,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'n_obs': n_obs,
            'r_squared': np.nan,  # RD doesn't report R2
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': CLUSTER_VAR if cluster is not None else 'None',
            'model_type': f'RD_p{p}_{kernel}',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            'bandwidth': float(bw_l)
        }

        results.append(result)
        print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, n={n_obs}")
        return result

    except Exception as e:
        print(f"  Error in {spec_id}: {str(e)}")
        return None


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

print("\n" + "="*80)
print("SPECIFICATION SEARCH FOR PAPER 207766-V1")
print("="*80)

# -----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATIONS
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("1. BASELINE SPECIFICATIONS (Exact replication of Table 2)")
print("-"*80)

for outcome in MAIN_OUTCOMES:
    run_rd_spec(
        y_data=df_analysis[outcome],
        x_data=df_analysis[RUNNING_VAR],
        cluster_data=df_analysis[CLUSTER_VAR],
        spec_id='baseline',
        spec_tree_path='methods/regression_discontinuity.md#baseline',
        outcome_var=outcome,
        sample_desc='Full analysis sample (insample==1)',
        controls_desc='None'
    )

# -----------------------------------------------------------------------------
# 2. BANDWIDTH VARIATIONS (RD-specific robustness)
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("2. BANDWIDTH VARIATIONS")
print("-"*80)

# Get optimal bandwidth for reference
PRIMARY_OUTCOME = 'amount_congr1_3_pcap'
df_bw = df_analysis[[PRIMARY_OUTCOME, RUNNING_VAR, CLUSTER_VAR]].dropna()
bw_obj = rdbwselect(df_bw[PRIMARY_OUTCOME].values, df_bw[RUNNING_VAR].values)
opt_bw = bw_obj.bws.iloc[0, 0] if hasattr(bw_obj.bws, 'iloc') else bw_obj.bws[0, 0]
print(f"Optimal bandwidth: {opt_bw:.2f}")

# Bandwidth multipliers (as in the paper's bandwidth robustness)
bw_multipliers = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5]

for mult in bw_multipliers:
    bw = opt_bw * mult
    for outcome in [PRIMARY_OUTCOME]:  # Focus on main outcome for bandwidth robustness
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            bw=bw,
            spec_id=f'rd/bandwidth/mult_{mult}',
            spec_tree_path='methods/regression_discontinuity.md#bandwidth-selection',
            outcome_var=outcome,
            sample_desc=f'Bandwidth = {mult}x optimal ({bw:.2f})',
            controls_desc='None'
        )

# Fixed bandwidths
for bw_fixed in [5, 7, 10, 15, 20]:
    run_rd_spec(
        y_data=df_analysis[PRIMARY_OUTCOME],
        x_data=df_analysis[RUNNING_VAR],
        cluster_data=df_analysis[CLUSTER_VAR],
        bw=bw_fixed,
        spec_id=f'rd/bandwidth/fixed_{bw_fixed}',
        spec_tree_path='methods/regression_discontinuity.md#bandwidth-selection',
        outcome_var=PRIMARY_OUTCOME,
        sample_desc=f'Fixed bandwidth = {bw_fixed}',
        controls_desc='None'
    )

# -----------------------------------------------------------------------------
# 3. POLYNOMIAL ORDER VARIATIONS
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("3. POLYNOMIAL ORDER VARIATIONS")
print("-"*80)

for p_order in [1, 2, 3]:
    for outcome in MAIN_OUTCOMES:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            p=p_order,
            spec_id=f'rd/poly/order_{p_order}',
            spec_tree_path='methods/regression_discontinuity.md#polynomial-order',
            outcome_var=outcome,
            sample_desc='Full analysis sample',
            controls_desc='None'
        )

# -----------------------------------------------------------------------------
# 4. KERNEL VARIATIONS
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("4. KERNEL VARIATIONS")
print("-"*80)

for kernel in ['triangular', 'uniform', 'epanechnikov']:
    for outcome in MAIN_OUTCOMES:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            kernel=kernel,
            spec_id=f'rd/kernel/{kernel}',
            spec_tree_path='methods/regression_discontinuity.md#kernel-function',
            outcome_var=outcome,
            sample_desc='Full analysis sample',
            controls_desc='None'
        )

# -----------------------------------------------------------------------------
# 5. CONTROL VARIATIONS (with covariates)
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("5. CONTROL VARIATIONS")
print("-"*80)

# Sample with controls
df_controls = df_analysis[df_analysis['insample_controls'] == 1].copy()

# Prepare control columns - handle missing values
for col in ALL_CONTROLS:
    if col in df_controls.columns:
        df_controls[col] = pd.to_numeric(df_controls[col], errors='coerce')

# Full controls
valid_controls = [c for c in ALL_CONTROLS if c in df_controls.columns and df_controls[c].notna().sum() > 100]

if len(valid_controls) > 0:
    for outcome in MAIN_OUTCOMES:
        control_data = [df_controls[c] for c in valid_controls if c in df_controls.columns]
        run_rd_spec(
            y_data=df_controls[outcome],
            x_data=df_controls[RUNNING_VAR],
            cluster_data=df_controls[CLUSTER_VAR],
            covs=control_data,
            spec_id='rd/controls/full',
            spec_tree_path='methods/regression_discontinuity.md#control-sets',
            outcome_var=outcome,
            sample_desc='Sample with controls (insample_controls==1)',
            controls_desc='Full demographic, electoral, and organization controls'
        )

# Demographic controls only
valid_demo = [c for c in CONTROLS_DEMO if c in df_controls.columns and df_controls[c].notna().sum() > 100]
if len(valid_demo) > 0:
    for outcome in MAIN_OUTCOMES:
        control_data = [df_controls[c] for c in valid_demo]
        run_rd_spec(
            y_data=df_controls[outcome],
            x_data=df_controls[RUNNING_VAR],
            cluster_data=df_controls[CLUSTER_VAR],
            covs=control_data,
            spec_id='rd/controls/demographic',
            spec_tree_path='methods/regression_discontinuity.md#control-sets',
            outcome_var=outcome,
            sample_desc='Sample with controls',
            controls_desc='Demographic controls only'
        )

# -----------------------------------------------------------------------------
# 6. SAMPLE RESTRICTIONS
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("6. SAMPLE RESTRICTIONS")
print("-"*80)

# Left-right races only
df_lr = df_analysis[df_analysis['insample_leftright'] == 1].copy()
for outcome in MAIN_OUTCOMES:
    run_rd_spec(
        y_data=df_lr[outcome],
        x_data=df_lr[RUNNING_VAR],
        cluster_data=df_lr[CLUSTER_VAR],
        spec_id='rd/sample/leftright_only',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=outcome,
        sample_desc='Left-right races only (insample_leftright==1)',
        controls_desc='None'
    )

# By year
for year in df_analysis['year'].dropna().unique():
    df_year = df_analysis[df_analysis['year'] == year]
    if len(df_year) > 100:
        run_rd_spec(
            y_data=df_year[PRIMARY_OUTCOME],
            x_data=df_year[RUNNING_VAR],
            cluster_data=df_year[CLUSTER_VAR],
            spec_id=f'rd/sample/year_{int(year)}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=PRIMARY_OUTCOME,
            sample_desc=f'Year = {int(year)}',
            controls_desc='None'
        )

# Drop each year
for year in df_analysis['year'].dropna().unique():
    df_noyear = df_analysis[df_analysis['year'] != year]
    run_rd_spec(
        y_data=df_noyear[PRIMARY_OUTCOME],
        x_data=df_noyear[RUNNING_VAR],
        cluster_data=df_noyear[CLUSTER_VAR],
        spec_id=f'rd/sample/drop_year_{int(year)}',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=PRIMARY_OUTCOME,
        sample_desc=f'Excluding year {int(year)}',
        controls_desc='None'
    )

# Pre-election years (as in heterogeneity analysis)
pre_elec = df_analysis[df_analysis['year'].isin([2006, 2007, 2012, 2013])]
if len(pre_elec) > 100:
    run_rd_spec(
        y_data=pre_elec[PRIMARY_OUTCOME],
        x_data=pre_elec[RUNNING_VAR],
        cluster_data=pre_elec[CLUSTER_VAR],
        spec_id='rd/sample/pre_election_years',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=PRIMARY_OUTCOME,
        sample_desc='Pre-election years only (2006-2007, 2012-2013)',
        controls_desc='None'
    )

# Non-pre-election years
non_pre_elec = df_analysis[~df_analysis['year'].isin([2006, 2007, 2012, 2013])]
if len(non_pre_elec) > 100:
    run_rd_spec(
        y_data=non_pre_elec[PRIMARY_OUTCOME],
        x_data=non_pre_elec[RUNNING_VAR],
        cluster_data=non_pre_elec[CLUSTER_VAR],
        spec_id='rd/sample/non_pre_election_years',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=PRIMARY_OUTCOME,
        sample_desc='Non-pre-election years',
        controls_desc='None'
    )

# Only two-candidate races
if 'nb_candi_1rd_all' in df_analysis.columns:
    df_two = df_analysis[df_analysis['nb_candi_1rd_all'] == 2]
    if len(df_two) > 100:
        for outcome in MAIN_OUTCOMES[:2]:  # Subset of outcomes
            run_rd_spec(
                y_data=df_two[outcome],
                x_data=df_two[RUNNING_VAR],
                cluster_data=df_two[CLUSTER_VAR],
                spec_id='rd/sample/two_candidates',
                spec_tree_path='robustness/sample_restrictions.md',
                outcome_var=outcome,
                sample_desc='Two-candidate races only',
                controls_desc='None'
            )

# By population size (large municipalities)
if 'pop_all' in df_analysis.columns:
    pop_median = df_analysis['pop_all'].median()
    df_large = df_analysis[df_analysis['pop_all'] >= pop_median]
    df_small = df_analysis[df_analysis['pop_all'] < pop_median]

    for outcome in [PRIMARY_OUTCOME]:
        run_rd_spec(
            y_data=df_large[outcome],
            x_data=df_large[RUNNING_VAR],
            cluster_data=df_large[CLUSTER_VAR],
            spec_id='rd/sample/large_municipalities',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            sample_desc='Above-median population',
            controls_desc='None'
        )
        run_rd_spec(
            y_data=df_small[outcome],
            x_data=df_small[RUNNING_VAR],
            cluster_data=df_small[CLUSTER_VAR],
            spec_id='rd/sample/small_municipalities',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=outcome,
            sample_desc='Below-median population',
            controls_desc='None'
        )

# Municipalities above 9000 pop (campaign spending capped)
if 'pop_all' in df_analysis.columns:
    df_capped = df_analysis[df_analysis['pop_all'] >= 9000]
    df_uncapped = df_analysis[df_analysis['pop_all'] < 9000]
    if len(df_capped) > 100:
        run_rd_spec(
            y_data=df_capped[PRIMARY_OUTCOME],
            x_data=df_capped[RUNNING_VAR],
            cluster_data=df_capped[CLUSTER_VAR],
            spec_id='rd/sample/campaign_spending_capped',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=PRIMARY_OUTCOME,
            sample_desc='Pop >= 9000 (campaign spending capped)',
            controls_desc='None'
        )
    if len(df_uncapped) > 100:
        run_rd_spec(
            y_data=df_uncapped[PRIMARY_OUTCOME],
            x_data=df_uncapped[RUNNING_VAR],
            cluster_data=df_uncapped[CLUSTER_VAR],
            spec_id='rd/sample/campaign_spending_uncapped',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var=PRIMARY_OUTCOME,
            sample_desc='Pop < 9000 (campaign spending uncapped)',
            controls_desc='None'
        )

# -----------------------------------------------------------------------------
# 7. DONUT HOLE SPECIFICATIONS
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("7. DONUT HOLE SPECIFICATIONS")
print("-"*80)

for donut in [0.5, 1, 2, 3, 5]:
    df_donut = df_analysis[abs(df_analysis[RUNNING_VAR]) > donut]
    if len(df_donut) > 100:
        run_rd_spec(
            y_data=df_donut[PRIMARY_OUTCOME],
            x_data=df_donut[RUNNING_VAR],
            cluster_data=df_donut[CLUSTER_VAR],
            spec_id=f'rd/donut/exclude_{donut}pp',
            spec_tree_path='methods/regression_discontinuity.md#donut-hole-specifications',
            outcome_var=PRIMARY_OUTCOME,
            sample_desc=f'Donut hole: exclude |alignm| <= {donut}',
            controls_desc='None'
        )

# -----------------------------------------------------------------------------
# 8. ALTERNATIVE OUTCOMES
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("8. ALTERNATIVE OUTCOMES")
print("-"*80)

# Alternative congruence splits (2 groups, 4 groups)
alt_outcomes_2 = ['amount_congr1_2_pcap', 'amount_congr2_2_pcap']
alt_outcomes_4 = ['amount_congr1_4_pcap', 'amount_congr2_4_pcap', 'amount_congr3_4_pcap', 'amount_congr4_4_pcap']

for outcome in alt_outcomes_2:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/split_2groups',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Organizations split into 2 groups',
            controls_desc='None'
        )

for outcome in alt_outcomes_4:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/split_4groups',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Organizations split into 4 groups',
            controls_desc='None'
        )

# Extensive margin (number of transfers)
extensive_outcomes = ['ntr_congr1_3_pcap', 'ntr_congr2_3_pcap', 'ntr_congr3_3_pcap', 'ntr_pcap']
for outcome in extensive_outcomes:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/extensive_margin',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Extensive margin (number of organizations receiving transfers)',
            controls_desc='None'
        )

# By organization age (young vs old)
young_outcomes = ['amount_congr1_3_yng_pcap', 'amount_congr2_3_yng_pcap', 'amount_congr3_3_yng_pcap', 'amount_yng_pcap']
old_outcomes = ['amount_congr1_3_old_pcap', 'amount_congr2_3_old_pcap', 'amount_congr3_3_old_pcap', 'amount_old_pcap']

for outcome in young_outcomes:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/young_orgs',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Young organizations (<= 6 years old)',
            controls_desc='None'
        )

for outcome in old_outcomes:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/old_orgs',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Old organizations (> 6 years old)',
            controls_desc='None'
        )

# Winsorized outcomes
winsorized_95 = ['amount_w95_congr1_3_pcap', 'amount_w95_congr2_3_pcap', 'amount_w95_congr3_3_pcap', 'amount_w95_pcap']
winsorized_99 = ['amount_w99_congr1_3_pcap', 'amount_w99_congr2_3_pcap', 'amount_w99_congr3_3_pcap', 'amount_w99_pcap']

for outcome in winsorized_95:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/winsorized_95',
            spec_tree_path='robustness/sample_restrictions.md#outlier-treatment',
            outcome_var=outcome,
            sample_desc='Winsorized at 95th percentile',
            controls_desc='None'
        )

for outcome in winsorized_99:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/winsorized_99',
            spec_tree_path='robustness/sample_restrictions.md#outlier-treatment',
            outcome_var=outcome,
            sample_desc='Winsorized at 99th percentile',
            controls_desc='None'
        )

# By number of ministries providing funds
multimin_outcomes = ['amount_congr1_3_multimin_pcap', 'amount_congr2_3_multimin_pcap', 'amount_congr3_3_multimin_pcap', 'amount_multimin_pcap']
onemin_outcomes = ['amount_congr1_3_onemin_pcap', 'amount_congr2_3_onemin_pcap', 'amount_congr3_3_onemin_pcap', 'amount_onemin_pcap']

for outcome in multimin_outcomes:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/multiple_ministries',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Organizations receiving from multiple ministries',
            controls_desc='None'
        )

for outcome in onemin_outcomes:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/single_ministry',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Organizations receiving from single ministry',
            controls_desc='None'
        )

# Non-local organizations (HQ not in municipality)
nonlocal_outcomes = ['amount_hq_congr1_3_noloc_pcap', 'amount_hq_congr2_3_noloc_pcap', 'amount_hq_congr3_3_noloc_pcap', 'amount_hq_noloc_pcap']
for outcome in nonlocal_outcomes:
    if outcome in df_analysis.columns:
        run_rd_spec(
            y_data=df_analysis[outcome],
            x_data=df_analysis[RUNNING_VAR],
            cluster_data=df_analysis[CLUSTER_VAR],
            spec_id='rd/outcome/nonlocal_orgs',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            sample_desc='Non-local organizations (HQ outside municipality)',
            controls_desc='None'
        )

# -----------------------------------------------------------------------------
# 9. HETEROGENEITY ANALYSES
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("9. HETEROGENEITY ANALYSES")
print("-"*80)

# By campaign spending levels
if 'campexp_aligned_cand_past_high' in df_analysis.columns:
    for level, label in [(1, 'high'), (0, 'low')]:
        df_het = df_analysis[df_analysis['campexp_aligned_cand_past_high'] == level]
        if len(df_het) > 100:
            run_rd_spec(
                y_data=df_het[PRIMARY_OUTCOME],
                x_data=df_het[RUNNING_VAR],
                cluster_data=df_het[CLUSTER_VAR],
                spec_id=f'rd/het/campaign_spending_{label}',
                spec_tree_path='robustness/heterogeneity.md',
                outcome_var=PRIMARY_OUTCOME,
                sample_desc=f'{label.capitalize()} campaign spending municipalities',
                controls_desc='None'
            )

# By government party popularity
if 'pres_score_high' in df_analysis.columns:
    for level, label in [(1, 'popular'), (0, 'unpopular')]:
        df_het = df_analysis[df_analysis['pres_score_high'] == level]
        if len(df_het) > 100:
            for outcome in MAIN_OUTCOMES[:2]:
                run_rd_spec(
                    y_data=df_het[outcome],
                    x_data=df_het[RUNNING_VAR],
                    cluster_data=df_het[CLUSTER_VAR],
                    spec_id=f'rd/het/govt_party_{label}',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome,
                    sample_desc=f'Govt party {label} locally',
                    controls_desc='None'
                )

# By incumbent status
if 'wasinpower' in df_analysis.columns:
    for level, label in [(1, 'incumbent'), (0, 'new')]:
        df_het = df_analysis[df_analysis['wasinpower'] == level]
        if len(df_het) > 100:
            for outcome in MAIN_OUTCOMES[:2]:
                run_rd_spec(
                    y_data=df_het[outcome],
                    x_data=df_het[RUNNING_VAR],
                    cluster_data=df_het[CLUSTER_VAR],
                    spec_id=f'rd/het/incumbent_{label}',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome,
                    sample_desc=f'Govt party {"was" if level==1 else "was not"} in power last term',
                    controls_desc='None'
                )

# By turnout
if 'low_turnout' in df_analysis.columns:
    for level, label in [(1, 'low'), (0, 'high')]:
        df_het = df_analysis[df_analysis['low_turnout'] == level]
        if len(df_het) > 100:
            for outcome in MAIN_OUTCOMES[:2]:
                run_rd_spec(
                    y_data=df_het[outcome],
                    x_data=df_het[RUNNING_VAR],
                    cluster_data=df_het[CLUSTER_VAR],
                    spec_id=f'rd/het/turnout_{label}',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome,
                    sample_desc=f'{label.capitalize()} turnout municipalities',
                    controls_desc='None'
                )

# -----------------------------------------------------------------------------
# 10. PLACEBO TESTS
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("10. PLACEBO TESTS")
print("-"*80)

# Placebo cutoffs
placebo_cutoffs = [-10, -5, 5, 10, 15, -15]
for pc in placebo_cutoffs:
    # Filter to appropriate side of true cutoff
    if pc < 0:
        df_placebo = df_analysis[df_analysis[RUNNING_VAR] < 0]
    else:
        df_placebo = df_analysis[df_analysis[RUNNING_VAR] > 0]

    if len(df_placebo) > 100:
        try:
            y = df_placebo[PRIMARY_OUTCOME].dropna().values
            x = df_placebo[RUNNING_VAR].dropna().values
            cluster = df_placebo[CLUSTER_VAR].dropna().values

            # Shift running variable
            x_shifted = x - pc

            mask = ~(np.isnan(y) | np.isnan(x_shifted))
            y = y[mask]
            x_shifted = x_shifted[mask]
            cluster = cluster[mask]

            if len(y) > 50:
                rd = rdrobust(y, x_shifted, c=0, cluster=cluster)
                coef = rd.coef.iloc[0] if hasattr(rd.coef, 'iloc') else rd.coef[0]
                se = rd.se.iloc[0] if hasattr(rd.se, 'iloc') else rd.se[0]
                pval = rd.pv.iloc[0] if hasattr(rd.pv, 'iloc') else rd.pv[0]
                bw_l = rd.bws.iloc[0, 0] if hasattr(rd.bws, 'iloc') else rd.bws[0, 0]
                n_l = int(rd.N_h[0]) if hasattr(rd.N_h, '__len__') else int(rd.N_h)
                n_r = int(rd.N_h[1]) if hasattr(rd.N_h, '__len__') and len(rd.N_h) > 1 else n_l

                coef_vector = {
                    "treatment": {"var": f"placebo_cutoff_{pc}", "coef": float(coef), "se": float(se), "pval": float(pval)},
                    "running_variable": {"var": RUNNING_VAR, "placebo_cutoff": pc},
                    "n_obs": n_l + n_r
                }

                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/placebo/cutoff_{pc}',
                    'spec_tree_path': 'methods/regression_discontinuity.md#placebo-cutoff-tests',
                    'outcome_var': PRIMARY_OUTCOME,
                    'treatment_var': f'{RUNNING_VAR} > {pc}',
                    'coefficient': coef,
                    'std_error': se,
                    't_stat': coef/se if se > 0 else np.nan,
                    'p_value': pval,
                    'p_value_robust': pval,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_obs': n_l + n_r,
                    'r_squared': np.nan,
                    'coefficient_vector_json': json.dumps(coef_vector),
                    'sample_desc': f'Placebo cutoff at {pc}',
                    'fixed_effects': 'None',
                    'controls_desc': 'None',
                    'cluster_var': CLUSTER_VAR,
                    'model_type': 'RD_placebo',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    'bandwidth': float(bw_l)
                })
                print(f"  rd/placebo/cutoff_{pc}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
        except Exception as e:
            print(f"  Error in placebo cutoff {pc}: {str(e)}")

# -----------------------------------------------------------------------------
# 11. INFERENCE VARIATIONS (without clustering)
# -----------------------------------------------------------------------------
print("\n" + "-"*80)
print("11. INFERENCE VARIATIONS (no clustering)")
print("-"*80)

for outcome in MAIN_OUTCOMES:
    run_rd_spec(
        y_data=df_analysis[outcome],
        x_data=df_analysis[RUNNING_VAR],
        cluster_data=None,  # No clustering
        spec_id='rd/inference/no_cluster',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var=outcome,
        sample_desc='Full analysis sample',
        controls_desc='None'
    )

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create DataFrame
results_df = pd.DataFrame(results)

# Summary statistics
print(f"\nTotal specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")

# For main outcome only
main_results = results_df[results_df['outcome_var'] == PRIMARY_OUTCOME]
print(f"\nFor main outcome ({PRIMARY_OUTCOME}):")
print(f"  Specifications: {len(main_results)}")
print(f"  Positive: {(main_results['coefficient'] > 0).sum()} ({(main_results['coefficient'] > 0).mean()*100:.1f}%)")
print(f"  Sig at 5%: {(main_results['p_value'] < 0.05).sum()} ({(main_results['p_value'] < 0.05).mean()*100:.1f}%)")

# Save
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

print("\nDone!")
