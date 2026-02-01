"""
Specification Search: Paper 195301-V1
Toews & Vezina - Enemies of the People: The Long-Term Effects of Soviet Forced Labor Camps

Journal: AEJ: Macroeconomics
DOI: https://doi.org/10.1257/mac.20200002

Main Hypothesis: Locations near Soviet GULAG camps with higher shares of
political prisoners ("enemies of the people") have higher wages today due to
human capital transmission from educated political prisoners.

Treatment Variable: share_enemies_1952_100 - Share of political prisoners
                   in camps within 100km of firm location
Primary Outcome: lnwage - Log average wage at firm level

Method: Cross-sectional OLS with region fixed effects
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "195301-V1"
JOURNAL = "AEJ: Macroeconomics"
PAPER_TITLE = "Enemies of the People: The Long-Term Effects of Soviet Forced Labor Camps"

DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/195301-V1/toews_vezina_replication/CodeandData/data/"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/195301-V1/toews_vezina_replication/CodeandData/"

# Method classification
METHOD_CODE = "cross_sectional_ols"
METHOD_TREE_PATH = "specification_tree/methods/cross_sectional_ols.md"

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

print("Loading data...")

# Load main firm-level data with GULAG treatment
df = pd.read_stata(DATA_PATH + "spark_enemies_temp.dta")

# Load region fixed effects data
df_region = pd.read_stata(DATA_PATH + "firms_region_all.dta")

# Merge
df = df.merge(df_region, on='sparkid', how='inner')

# Clean data
df = df.dropna(subset=['sparkid'])

print(f"Initial observations: {len(df)}")

# ============================================================================
# VARIABLE DEFINITIONS
# ============================================================================

# Primary treatment and outcome
TREATMENT_VAR = "share_enemies_1952_100"
OUTCOME_VAR = "lnwage"

# Alternative outcomes
ALT_OUTCOMES = ["lnnetprofitpc_raw", "lnvalueaddedpc_rawemp"]

# Control variables (baseline) - Note: Stata code uses lat/lon but data has latitude/longitude
BASELINE_CONTROLS = ["ln100people1952", "latitude", "longitude", "lnpop_within_100km_1926", "lnkm_to_1937_tracks"]

# Fixed effects
FE_VAR = "Oblast"

# Clustering variable
CLUSTER_VAR = "gulag_cluster"

# Weight variable
WEIGHT_VAR = "employees2018"

# Sample restriction for Moscow exclusion
df['exclude_moscow'] = df['dist_to_Moscow'] > 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_se_stats(result, treatment_var):
    """Extract coefficient, SE, t-stat, p-value, and CI from pyfixest result."""
    try:
        coef = result.coef()[treatment_var]
        se = result.se()[treatment_var]
        tstat = result.tstat()[treatment_var]
        pval = result.pvalue()[treatment_var]
        ci = result.confint().loc[treatment_var]
        ci_lower = ci.iloc[0]
        ci_upper = ci.iloc[1]
        return coef, se, tstat, pval, ci_lower, ci_upper
    except Exception as e:
        print(f"    Error extracting stats: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def get_nobs(result):
    """Get number of observations from pyfixest result."""
    try:
        if hasattr(result, 'nobs'):
            return result.nobs()
        elif hasattr(result, '_N'):
            return result._N
        elif hasattr(result, 'N'):
            return result.N
        else:
            return len(result._data)
    except:
        return np.nan

def get_r2(result):
    """Get R-squared from pyfixest result."""
    try:
        if hasattr(result, 'r2'):
            return result.r2()
        elif hasattr(result, '_r2'):
            return result._r2
        else:
            return np.nan
    except:
        return np.nan

def get_coefficient_vector(result, treatment_var, controls, fixed_effects):
    """Create coefficient vector JSON from result."""
    try:
        coef_dict = {
            "treatment": {
                "var": treatment_var,
                "coef": float(result.coef()[treatment_var]) if treatment_var in result.coef() else None,
                "se": float(result.se()[treatment_var]) if treatment_var in result.se() else None,
                "pval": float(result.pvalue()[treatment_var]) if treatment_var in result.pvalue() else None
            },
            "controls": [],
            "fixed_effects": fixed_effects,
            "diagnostics": {
                "n_obs": int(get_nobs(result)),
                "r_squared": float(get_r2(result)) if get_r2(result) is not None else None
            }
        }

        for ctrl in controls:
            if ctrl in result.coef():
                coef_dict["controls"].append({
                    "var": ctrl,
                    "coef": float(result.coef()[ctrl]),
                    "se": float(result.se()[ctrl]),
                    "pval": float(result.pvalue()[ctrl])
                })

        return json.dumps(coef_dict)
    except Exception as e:
        return json.dumps({"error": str(e)})

def run_specification(df_sub, outcome, treatment, controls, fe, cluster, weight_col,
                      spec_id, spec_tree_path, sample_desc):
    """Run a single specification and return results dict."""

    # Build formula
    if controls:
        ctrl_str = " + ".join(controls)
        formula = f"{outcome} ~ {treatment} + {ctrl_str}"
    else:
        formula = f"{outcome} ~ {treatment}"

    # Add fixed effects
    if fe:
        formula += f" | {fe}"

    # Prepare weights
    if weight_col and weight_col in df_sub.columns:
        df_sub = df_sub.copy()
        df_sub['_weights'] = df_sub[weight_col].fillna(0)
        df_sub = df_sub[df_sub['_weights'] > 0]

    # Run regression
    try:
        if cluster and cluster in df_sub.columns:
            # Need to handle cluster variable
            df_sub = df_sub.dropna(subset=[cluster])
            if weight_col:
                result = pf.feols(formula, data=df_sub, vcov={'CRV1': cluster}, weights='_weights')
            else:
                result = pf.feols(formula, data=df_sub, vcov={'CRV1': cluster})
        else:
            if weight_col:
                result = pf.feols(formula, data=df_sub, vcov='hetero', weights='_weights')
            else:
                result = pf.feols(formula, data=df_sub, vcov='hetero')

        coef, se, tstat, pval, ci_lower, ci_upper = calculate_se_stats(result, treatment)
        n_obs = get_nobs(result)
        r2 = get_r2(result)

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r2,
            'coefficient_vector_json': get_coefficient_vector(result, treatment, controls, [fe] if fe else []),
            'sample_desc': sample_desc,
            'fixed_effects': fe if fe else "none",
            'controls_desc': ", ".join(controls) if controls else "none",
            'cluster_var': cluster if cluster else "none",
            'model_type': 'OLS with FE' if fe else 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None

# ============================================================================
# RUN SPECIFICATIONS
# ============================================================================

results = []

# Prepare main sample
print("\nPreparing main sample...")
df_main = df.copy()
df_main = df_main.dropna(subset=[OUTCOME_VAR, TREATMENT_VAR, FE_VAR])
for ctrl in BASELINE_CONTROLS:
    if ctrl in df_main.columns:
        df_main = df_main.dropna(subset=[ctrl])

print(f"Main sample size: {len(df_main)}")

# ----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATION (Table 5, Column 3 - with controls)
# ----------------------------------------------------------------------------
print("\n=== Running Baseline Specifications ===")

# Baseline with all controls (Table 5, Col 3)
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "baseline", METHOD_TREE_PATH, "All firms, weighted by employees"
)
if spec:
    results.append(spec)
    print(f"  Baseline: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}, p={spec['p_value']:.4f}")

# Baseline without Moscow (Table 5, Col 6)
df_no_moscow = df_main[df_main['exclude_moscow']].copy()
spec = run_specification(
    df_no_moscow, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "baseline_no_moscow", METHOD_TREE_PATH + "#sample-restrictions", "Excluding Moscow (dist>100km)"
)
if spec:
    results.append(spec)
    print(f"  No Moscow: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}, p={spec['p_value']:.4f}")

# ----------------------------------------------------------------------------
# 2. CONTROL VARIATIONS (Single Covariate / Leave-One-Out)
# ----------------------------------------------------------------------------
print("\n=== Running Control Variations ===")

# No controls
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, [], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "robust/single/none", "robustness/single_covariate.md", "No controls"
)
if spec:
    results.append(spec)
    print(f"  No controls: coef={spec['coefficient']:.4f}")

# Single control - treatment only (Table 5, Col 1)
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, [], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "ols/controls/none", METHOD_TREE_PATH + "#control-sets", "Treatment + FE only"
)
if spec:
    results.append(spec)

# Single control variations
for ctrl in BASELINE_CONTROLS:
    spec = run_specification(
        df_main, OUTCOME_VAR, TREATMENT_VAR, [ctrl], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
        f"robust/single/{ctrl}", "robustness/single_covariate.md", f"Single control: {ctrl}"
    )
    if spec:
        results.append(spec)
        print(f"  Single {ctrl}: coef={spec['coefficient']:.4f}")

# Leave-one-out variations
for ctrl in BASELINE_CONTROLS:
    remaining = [c for c in BASELINE_CONTROLS if c != ctrl]
    spec = run_specification(
        df_main, OUTCOME_VAR, TREATMENT_VAR, remaining, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
        f"robust/loo/drop_{ctrl}", "robustness/leave_one_out.md", f"Drop {ctrl}"
    )
    if spec:
        results.append(spec)
        print(f"  LOO drop {ctrl}: coef={spec['coefficient']:.4f}")

# Minimal controls (just prisoners - Table 5, Col 2)
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, ["ln100people1952", "latitude", "longitude"], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "ols/controls/minimal", METHOD_TREE_PATH + "#control-sets", "Minimal controls: prisoners + geography"
)
if spec:
    results.append(spec)

# ----------------------------------------------------------------------------
# 3. ALTERNATIVE OUTCOMES
# ----------------------------------------------------------------------------
print("\n=== Running Alternative Outcomes ===")

for alt_outcome in ALT_OUTCOMES:
    if alt_outcome in df_main.columns:
        df_sub = df_main.dropna(subset=[alt_outcome])
        spec = run_specification(
            df_sub, alt_outcome, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
            f"ols/outcome/{alt_outcome}", METHOD_TREE_PATH, f"Alternative outcome: {alt_outcome}"
        )
        if spec:
            results.append(spec)
            print(f"  {alt_outcome}: coef={spec['coefficient']:.4f}, p={spec['p_value']:.4f}")

# ----------------------------------------------------------------------------
# 4. FIXED EFFECTS VARIATIONS
# ----------------------------------------------------------------------------
print("\n=== Running Fixed Effects Variations ===")

# No fixed effects
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, None, CLUSTER_VAR, WEIGHT_VAR,
    "ols/fe/none", METHOD_TREE_PATH + "#fixed-effects", "No fixed effects"
)
if spec:
    results.append(spec)
    print(f"  No FE: coef={spec['coefficient']:.4f}")

# With sector FE (using Description_Parent if available)
if 'Description_Parent' in df_main.columns:
    df_sector = df_main.dropna(subset=['Description_Parent'])
    spec = run_specification(
        df_sector, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, "Description_Parent + Oblast",
        CLUSTER_VAR, WEIGHT_VAR,
        "ols/fe/sector", METHOD_TREE_PATH + "#fixed-effects", "Region + Sector FE"
    )
    if spec:
        results.append(spec)
        print(f"  Sector FE: coef={spec['coefficient']:.4f}")

# ----------------------------------------------------------------------------
# 5. CLUSTERING VARIATIONS
# ----------------------------------------------------------------------------
print("\n=== Running Clustering Variations ===")

# Robust SE (no clustering)
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, None, WEIGHT_VAR,
    "robust/cluster/none", "robustness/clustering_variations.md", "Robust SE, no clustering"
)
if spec:
    results.append(spec)
    print(f"  No cluster: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}")

# Cluster by region (region_camp is the camp region variable)
if 'region_camp' in df_main.columns:
    spec = run_specification(
        df_main, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, "region_camp", WEIGHT_VAR,
        "robust/cluster/region", "robustness/clustering_variations.md", "Clustered by region"
    )
    if spec:
        results.append(spec)
        print(f"  Cluster region: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}")

# Cluster by Oblast
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, FE_VAR, WEIGHT_VAR,
    "robust/cluster/oblast", "robustness/clustering_variations.md", "Clustered by Oblast"
)
if spec:
    results.append(spec)
    print(f"  Cluster Oblast: coef={spec['coefficient']:.4f}, se={spec['std_error']:.4f}")

# ----------------------------------------------------------------------------
# 6. SAMPLE RESTRICTIONS
# ----------------------------------------------------------------------------
print("\n=== Running Sample Restrictions ===")

# Exclude Moscow (already done above, but add explicitly)
spec = run_specification(
    df_no_moscow, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "robust/sample/exclude_moscow", "robustness/sample_restrictions.md", "Exclude Moscow (dist>100km)"
)
if spec:
    results.append(spec)

# Minimum employee thresholds
for min_emp in [10, 50, 100]:
    df_sub = df_main[df_main['employees2018'] >= min_emp].copy()
    spec = run_specification(
        df_sub, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
        f"robust/sample/min_emp_{min_emp}", "robustness/sample_restrictions.md",
        f"Firms with >= {min_emp} employees"
    )
    if spec:
        results.append(spec)
        print(f"  Min {min_emp} employees: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Exclude mining sector
if 'Description_Parent' in df_main.columns:
    df_no_mining = df_main[df_main['Description_Parent'] != 'MINING AND QUARRYING'].copy()
    spec = run_specification(
        df_no_mining, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
        "robust/sample/exclude_mining", "robustness/sample_restrictions.md", "Exclude mining firms"
    )
    if spec:
        results.append(spec)
        print(f"  No mining: coef={spec['coefficient']:.4f}, n={spec['n_obs']}")

# Exclude extreme enemy share (>90%)
df_sub = df_main[df_main['enemy_share_1952'] < 0.9].copy()
spec = run_specification(
    df_sub, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "robust/sample/exclude_extreme_treatment", "robustness/sample_restrictions.md",
    "Exclude camps >90% enemies"
)
if spec:
    results.append(spec)
    print(f"  No extreme: coef={spec['coefficient']:.4f}")

# Exclude zero enemy share
df_sub = df_main[(df_main['enemy_share_1952'] > 0) & (df_main['enemy_share_1952'].notna())].copy()
spec = run_specification(
    df_sub, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
    "robust/sample/exclude_zero_treatment", "robustness/sample_restrictions.md",
    "Exclude camps with 0% enemies"
)
if spec:
    results.append(spec)
    print(f"  No zero: coef={spec['coefficient']:.4f}")

# ----------------------------------------------------------------------------
# 7. ALTERNATIVE TREATMENT DEFINITIONS (Distance Variations - "Berliner" plots)
# ----------------------------------------------------------------------------
print("\n=== Running Distance Radius Variations ===")

for radius in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    treat_var = f"share_enemies_1952_{radius}"
    if treat_var in df_main.columns:
        df_sub = df_main.dropna(subset=[treat_var])
        spec = run_specification(
            df_sub, OUTCOME_VAR, treat_var, [], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
            f"custom/radius_{radius}km", METHOD_TREE_PATH, f"Treatment radius: {radius}km"
        )
        if spec:
            results.append(spec)
            print(f"  Radius {radius}km: coef={spec['coefficient']:.4f}")

# ----------------------------------------------------------------------------
# 8. ALTERNATIVE TREATMENT MEASURES
# ----------------------------------------------------------------------------
print("\n=== Running Alternative Treatment Measures ===")

# Log enemies (Table A8)
if 'lnenemies100Gulag1952' not in df_main.columns:
    df_main['lnenemies100Gulag1952'] = np.log1p(df_main['enemies100Gulag1952']) if 'enemies100Gulag1952' in df_main.columns else np.nan

if 'lnenemies100Gulag1952' in df_main.columns and df_main['lnenemies100Gulag1952'].notna().sum() > 100:
    df_sub = df_main.dropna(subset=['lnenemies100Gulag1952'])
    spec = run_specification(
        df_sub, OUTCOME_VAR, 'lnenemies100Gulag1952', ["ln100people1952"], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
        "custom/treatment_log_enemies", METHOD_TREE_PATH, "Log enemies (alternative treatment)"
    )
    if spec:
        results.append(spec)
        print(f"  Log enemies: coef={spec['coefficient']:.4f}")

# 1939 camps instead of 1952
if 'share_enemies_1939_100' not in df_main.columns:
    if 'enemies100Gulag1939' in df_main.columns and 'people_100_1939' in df_main.columns:
        df_main['share_enemies_1939_100'] = df_main['enemies100Gulag1939'] / df_main['people_100_1939']

if 'share_enemies_1939_100' in df_main.columns:
    df_sub = df_main.dropna(subset=['share_enemies_1939_100'])
    spec = run_specification(
        df_sub, OUTCOME_VAR, 'share_enemies_1939_100', ["ln100people1952"], FE_VAR, CLUSTER_VAR, WEIGHT_VAR,
        "custom/treatment_1939_camps", METHOD_TREE_PATH, "1939 camps instead of 1952"
    )
    if spec:
        results.append(spec)
        print(f"  1939 camps: coef={spec['coefficient']:.4f}")

# ----------------------------------------------------------------------------
# 9. FUNCTIONAL FORM VARIATIONS
# ----------------------------------------------------------------------------
print("\n=== Running Functional Form Variations ===")

# Unweighted regression
spec = run_specification(
    df_main, OUTCOME_VAR, TREATMENT_VAR, BASELINE_CONTROLS, FE_VAR, CLUSTER_VAR, None,
    "ols/method/unweighted", METHOD_TREE_PATH + "#estimation-method", "Unweighted OLS"
)
if spec:
    results.append(spec)
    print(f"  Unweighted: coef={spec['coefficient']:.4f}")

# ----------------------------------------------------------------------------
# 10. GULAG DUMMY ANALYSIS (Table 6 style)
# ----------------------------------------------------------------------------
print("\n=== Running GULAG Dummy Analysis ===")

# Create GULAG dummies
df_main['gulag_low'] = ((df_main['Gulag1952'] == 1) & (df_main['share_enemies_1952_100'] < 0.01)).astype(int)
df_main['gulag_med'] = ((df_main['Gulag1952'] == 1) & (df_main['share_enemies_1952_100'] >= 0.01) & (df_main['share_enemies_1952_100'] < 0.2)).astype(int)
df_main['gulag_high'] = ((df_main['Gulag1952'] == 1) & (df_main['share_enemies_1952_100'] >= 0.2)).astype(int)

# Run with dummy treatment
for dummy in ['gulag_low', 'gulag_med', 'gulag_high']:
    spec = run_specification(
        df_main, OUTCOME_VAR, dummy, [], FE_VAR, FE_VAR, WEIGHT_VAR,
        f"custom/gulag_dummy_{dummy}", METHOD_TREE_PATH, f"GULAG dummy: {dummy}"
    )
    if spec:
        results.append(spec)
        print(f"  {dummy}: coef={spec['coefficient']:.4f}")

# ============================================================================
# COMPILE RESULTS
# ============================================================================

print("\n=== Compiling Results ===")

# Create DataFrame
results_df = pd.DataFrame([r for r in results if r is not None])

# Save to CSV
output_file = OUTPUT_PATH + "specification_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Saved {len(results_df)} specifications to {output_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("SPECIFICATION SEARCH SUMMARY")
print("="*60)

if len(results_df) > 0:
    print(f"\nTotal specifications run: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

    print(f"\nCoefficient statistics (treatment effect on log wages):")
    print(f"  Median: {results_df['coefficient'].median():.4f}")
    print(f"  Mean: {results_df['coefficient'].mean():.4f}")
    print(f"  Std Dev: {results_df['coefficient'].std():.4f}")
    print(f"  Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # Breakdown by category
    print("\nBreakdown by specification category:")
    results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
    for cat, group in results_df.groupby('category'):
        sig_rate = 100 * (group['p_value'] < 0.05).mean()
        print(f"  {cat}: N={len(group)}, {sig_rate:.1f}% significant at 5%")
else:
    print("\nNo specifications completed successfully. Check errors above.")

print("\n" + "="*60)
print("Analysis complete!")
