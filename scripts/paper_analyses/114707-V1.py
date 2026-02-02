"""
Specification Search: 114707-V1
Do Hospital Mergers Reduce Costs? Evidence on Prices and Hospital Market Consolidation

Paper: Schmitt (2017)
Method: Difference-in-Differences with hospital and year fixed effects
Main outcome: Log non-medical prices (lnprnonmed)
Treatment: Post-merger indicator (post)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "114707-V1"
PAPER_TITLE = "Do Hospital Mergers Reduce Costs? Evidence on Prices and Hospital Market Consolidation"
JOURNAL = "AEA"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114707-V1/data/HospitalMMC_Data.dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114707-V1/"

# Main variables
OUTCOME_VAR = "lnprnonmed"
TREATMENT_VAR = "post"
BASELINE_CONTROLS = ["lncmi", "pctmcaid", "lnbeds", "fp", "hhi", "sysoth"]
CLUSTER_VAR = "h"
WEIGHT_VAR = "dis_tot"

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
df_full = pd.read_stata(DATA_PATH)

# Main sample: drop indirect hospitals
df = df_full[df_full['indirect'] != 1].copy()
print(f"Full dataset: {len(df_full)} obs, Main sample (excl indirect): {len(df)} obs")

# Convert year to int for fixed effects
df['year'] = df['year'].astype(int)
df['h'] = df['h'].astype(int)

# Create additional sample restrictions
df_matched = df[df['optmatch'] == 1].copy()
df_samesys = df[df['samesysctrl'] == 1].copy()

print(f"Matched sample: {len(df_matched)} obs")
print(f"Same-system control sample: {len(df_samesys)} obs")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_print(result, label, include_n=False):
    """Safely print result without crashing on None values."""
    coef = result.get('coefficient')
    pval = result.get('p_value')
    n = result.get('n_obs')

    coef_str = f"{coef:.4f}" if coef is not None else "N/A"
    pval_str = f"{pval:.4f}" if pval is not None else "N/A"
    n_str = f", n={n}" if (include_n and n is not None) else ""

    print(f"  {label}: coef={coef_str}, p={pval_str}{n_str}")


def run_spec(data, formula, spec_id, spec_tree_path, treatment_var=TREATMENT_VAR,
             outcome_var=OUTCOME_VAR, cluster_var=CLUSTER_VAR, weight_var=None,
             sample_desc="Main sample", fixed_effects="Hospital + Year",
             controls_desc="", model_type="TWFE"):
    """Run a single specification and return results dict."""

    try:
        # Handle weights
        if weight_var and weight_var in data.columns:
            # Drop rows with missing weights or zero/negative weights
            data_clean = data[data[weight_var].notna() & (data[weight_var] > 0)].copy()
            model = pf.feols(formula, data=data_clean,
                           vcov={'CRV1': cluster_var},
                           weights=weight_var)
        else:
            data_clean = data.copy()
            model = pf.feols(formula, data=data_clean,
                           vcov={'CRV1': cluster_var})

        # Extract results
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        # Get treatment coefficient
        if treatment_var in coefs.index:
            treat_coef = coefs[treatment_var]
            treat_se = ses[treatment_var]
            treat_pval = pvals[treatment_var]
            treat_tstat = treat_coef / treat_se if treat_se > 0 else np.nan
            ci_lower = treat_coef - 1.96 * treat_se
            ci_upper = treat_coef + 1.96 * treat_se
        else:
            # Treatment var might be named differently
            treat_coef = np.nan
            treat_se = np.nan
            treat_pval = np.nan
            treat_tstat = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        # Get N and R2
        n_obs = model._N
        r_squared = model._r2 if hasattr(model, '_r2') else None

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(treat_coef) if not np.isnan(treat_coef) else None,
                "se": float(treat_se) if not np.isnan(treat_se) else None,
                "pval": float(treat_pval) if not np.isnan(treat_pval) else None
            },
            "controls": [],
            "fixed_effects_absorbed": fixed_effects.split(" + ") if fixed_effects else [],
            "n_obs": int(n_obs) if n_obs else None,
            "n_clusters": int(len(data_clean[cluster_var].unique())) if cluster_var in data_clean.columns else None,
            "r_squared": float(r_squared) if r_squared else None
        }

        # Add control coefficients
        for var in coefs.index:
            if var != treatment_var:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(coefs[var]),
                    "se": float(ses[var]),
                    "pval": float(pvals[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(treat_coef) if not np.isnan(treat_coef) else None,
            'std_error': float(treat_se) if not np.isnan(treat_se) else None,
            't_stat': float(treat_tstat) if not np.isnan(treat_tstat) else None,
            'p_value': float(treat_pval) if not np.isnan(treat_pval) else None,
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r_squared) if r_squared else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  ERROR in {spec_id}: {str(e)}")
        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': None,
            'std_error': None,
            't_stat': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'n_obs': None,
            'r_squared': None,
            'coefficient_vector_json': json.dumps({"error": str(e)}),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

# ============================================================================
# RUN SPECIFICATIONS
# ============================================================================

results = []

# ----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATION (Table 2, Column 2)
# ----------------------------------------------------------------------------
print("\n=== 1. BASELINE SPECIFICATION ===")

controls_str = " + ".join(BASELINE_CONTROLS)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | h + year"

result = run_spec(df, formula,
                  spec_id="baseline",
                  spec_tree_path="methods/difference_in_differences.md#baseline",
                  weight_var=WEIGHT_VAR,
                  controls_desc="Full controls: lncmi, pctmcaid, lnbeds, fp, hhi, sysoth")
results.append(result)
safe_print(result, "Baseline", include_n=True)

# ----------------------------------------------------------------------------
# 2. FIXED EFFECTS VARIATIONS
# ----------------------------------------------------------------------------
print("\n=== 2. FIXED EFFECTS VARIATIONS ===")

# 2.1 Hospital FE only
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | h"
result = run_spec(df, formula, spec_id="did/fe/unit_only",
                  spec_tree_path="methods/difference_in_differences.md#fixed-effects",
                  weight_var=WEIGHT_VAR, fixed_effects="Hospital only", controls_desc="Full controls")
results.append(result)
safe_print(result, "Hospital FE only")

# 2.2 Year FE only
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | year"
result = run_spec(df, formula, spec_id="did/fe/time_only",
                  spec_tree_path="methods/difference_in_differences.md#fixed-effects",
                  weight_var=WEIGHT_VAR, fixed_effects="Year only", controls_desc="Full controls")
results.append(result)
safe_print(result, "Year FE only")

# 2.3 No fixed effects
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str}"
result = run_spec(df, formula, spec_id="did/fe/none",
                  spec_tree_path="methods/difference_in_differences.md#fixed-effects",
                  weight_var=WEIGHT_VAR, fixed_effects="None", controls_desc="Full controls", model_type="Pooled OLS")
results.append(result)
safe_print(result, "No FE")

# ----------------------------------------------------------------------------
# 3. CONTROL VARIATIONS - Leave One Out
# ----------------------------------------------------------------------------
print("\n=== 3. LEAVE-ONE-OUT CONTROLS ===")

for control in BASELINE_CONTROLS:
    remaining = [c for c in BASELINE_CONTROLS if c != control]
    controls_str_loo = " + ".join(remaining)
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str_loo} | h + year"
    result = run_spec(df, formula, spec_id=f"robust/loo/drop_{control}",
                      spec_tree_path="robustness/leave_one_out.md",
                      weight_var=WEIGHT_VAR, controls_desc=f"Drop {control}")
    results.append(result)
    safe_print(result, f"Drop {control}")

# ----------------------------------------------------------------------------
# 4. CONTROL VARIATIONS - Single Covariate
# ----------------------------------------------------------------------------
print("\n=== 4. SINGLE COVARIATE ===")

# No controls
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} | h + year"
result = run_spec(df, formula, spec_id="did/controls/none",
                  spec_tree_path="methods/difference_in_differences.md#control-sets",
                  weight_var=WEIGHT_VAR, controls_desc="No controls")
results.append(result)
safe_print(result, "No controls")

# Add controls incrementally
for i, control in enumerate(BASELINE_CONTROLS):
    controls_so_far = BASELINE_CONTROLS[:i+1]
    controls_str_add = " + ".join(controls_so_far)
    formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str_add} | h + year"
    result = run_spec(df, formula, spec_id=f"robust/control/add_{control}",
                      spec_tree_path="robustness/control_progression.md",
                      weight_var=WEIGHT_VAR, controls_desc=f"Controls: {', '.join(controls_so_far)}")
    results.append(result)
    safe_print(result, f"Add {control}")

# ----------------------------------------------------------------------------
# 5. SAMPLE RESTRICTIONS
# ----------------------------------------------------------------------------
print("\n=== 5. SAMPLE RESTRICTIONS ===")

controls_str = " + ".join(BASELINE_CONTROLS)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | h + year"

# 5.1 Matched sample
result = run_spec(df_matched, formula, spec_id="did/sample/matched",
                  spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
                  weight_var=WEIGHT_VAR, sample_desc="Matched controls only", controls_desc="Full controls")
results.append(result)
safe_print(result, "Matched sample", include_n=True)

# 5.2 Same-system controls
result = run_spec(df_samesys, formula, spec_id="did/sample/same_system",
                  spec_tree_path="methods/difference_in_differences.md#sample-restrictions",
                  weight_var=WEIGHT_VAR, sample_desc="Same-system controls", controls_desc="Full controls")
results.append(result)
safe_print(result, "Same-system", include_n=True)

# 5.3 Time period restrictions
years = sorted(df['year'].unique())
mid_year = years[len(years)//2]

df_early = df[df['year'] <= mid_year].copy()
result = run_spec(df_early, formula, spec_id="robust/sample/early_period",
                  spec_tree_path="robustness/sample_restrictions.md",
                  weight_var=WEIGHT_VAR, sample_desc=f"Early period (year <= {mid_year})", controls_desc="Full controls")
results.append(result)
safe_print(result, "Early period", include_n=True)

df_late = df[df['year'] > mid_year].copy()
result = run_spec(df_late, formula, spec_id="robust/sample/late_period",
                  spec_tree_path="robustness/sample_restrictions.md",
                  weight_var=WEIGHT_VAR, sample_desc=f"Late period (year > {mid_year})", controls_desc="Full controls")
results.append(result)
safe_print(result, "Late period", include_n=True)

# Drop first/last year
first_year, last_year = min(years), max(years)
df_no_first = df[df['year'] != first_year].copy()
result = run_spec(df_no_first, formula, spec_id="robust/sample/exclude_first_year",
                  spec_tree_path="robustness/sample_restrictions.md",
                  weight_var=WEIGHT_VAR, sample_desc=f"Exclude first year ({first_year})", controls_desc="Full controls")
results.append(result)
safe_print(result, "Exclude first year")

df_no_last = df[df['year'] != last_year].copy()
result = run_spec(df_no_last, formula, spec_id="robust/sample/exclude_last_year",
                  spec_tree_path="robustness/sample_restrictions.md",
                  weight_var=WEIGHT_VAR, sample_desc=f"Exclude last year ({last_year})", controls_desc="Full controls")
results.append(result)
safe_print(result, "Exclude last year")

# Drop each year
for year in years[1:-1]:
    df_drop_year = df[df['year'] != year].copy()
    result = run_spec(df_drop_year, formula, spec_id=f"robust/sample/drop_year_{year}",
                      spec_tree_path="robustness/sample_restrictions.md",
                      weight_var=WEIGHT_VAR, sample_desc=f"Drop year {year}", controls_desc="Full controls")
    results.append(result)
print(f"  Drop years: {len(years)-2} specs added")

# Winsorize/Trim
for pct in [1, 5]:
    df_wins = df.copy()
    lower, upper = df_wins[OUTCOME_VAR].quantile(pct/100), df_wins[OUTCOME_VAR].quantile(1 - pct/100)
    df_wins[OUTCOME_VAR] = df_wins[OUTCOME_VAR].clip(lower=lower, upper=upper)
    result = run_spec(df_wins, formula, spec_id=f"robust/sample/winsor_{pct}pct",
                      spec_tree_path="robustness/sample_restrictions.md",
                      weight_var=WEIGHT_VAR, sample_desc=f"Winsorize {pct}%", controls_desc="Full controls")
    results.append(result)
    safe_print(result, f"Winsorize {pct}%")

for pct in [1, 5]:
    df_trim = df[(df[OUTCOME_VAR] >= df[OUTCOME_VAR].quantile(pct/100)) &
                 (df[OUTCOME_VAR] <= df[OUTCOME_VAR].quantile(1 - pct/100))].copy()
    result = run_spec(df_trim, formula, spec_id=f"robust/sample/trim_{pct}pct",
                      spec_tree_path="robustness/sample_restrictions.md",
                      weight_var=WEIGHT_VAR, sample_desc=f"Trim {pct}%", controls_desc="Full controls")
    results.append(result)
    safe_print(result, f"Trim {pct}%", include_n=True)

# Balanced panel
obs_per_hospital = df.groupby('h').size()
balanced_hospitals = obs_per_hospital[obs_per_hospital == obs_per_hospital.max()].index
df_balanced = df[df['h'].isin(balanced_hospitals)].copy()
result = run_spec(df_balanced, formula, spec_id="robust/sample/balanced",
                  spec_tree_path="robustness/sample_restrictions.md",
                  weight_var=WEIGHT_VAR, sample_desc="Balanced panel", controls_desc="Full controls")
results.append(result)
safe_print(result, "Balanced panel", include_n=True)

# ----------------------------------------------------------------------------
# 6. ALTERNATIVE OUTCOMES
# ----------------------------------------------------------------------------
print("\n=== 6. ALTERNATIVE OUTCOMES ===")

formula_med = f"lnprmed ~ {TREATMENT_VAR} + {controls_str} | h + year"
result = run_spec(df, formula_med, spec_id="robust/outcome/lnprmed",
                  spec_tree_path="robustness/measurement.md",
                  outcome_var="lnprmed", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Medical prices")

df_std = df.copy()
df_std['lnprnonmed_std'] = (df_std['lnprnonmed'] - df_std['lnprnonmed'].mean()) / df_std['lnprnonmed'].std()
formula_std = f"lnprnonmed_std ~ {TREATMENT_VAR} + {controls_str} | h + year"
result = run_spec(df_std, formula_std, spec_id="robust/form/y_standardized",
                  spec_tree_path="robustness/functional_form.md",
                  outcome_var="lnprnonmed_std", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Standardized outcome")

# ----------------------------------------------------------------------------
# 7. ALTERNATIVE TREATMENT DEFINITIONS
# ----------------------------------------------------------------------------
print("\n=== 7. ALTERNATIVE TREATMENT DEFINITIONS ===")

formula_active = f"{OUTCOME_VAR} ~ post_active + post_passive + {controls_str} | h + year"
result = run_spec(df, formula_active, spec_id="did/treatment/active_passive",
                  spec_tree_path="methods/difference_in_differences.md#treatment-definition",
                  treatment_var="post_active", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Active merger")

formula_state = f"{OUTCOME_VAR} ~ post_instate + post_outstate + {controls_str} | h + year"
result = run_spec(df, formula_state, spec_id="did/treatment/instate_outstate",
                  spec_tree_path="methods/difference_in_differences.md#treatment-definition",
                  treatment_var="post_instate", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "In-state merger")

formula_hhi = f"{OUTCOME_VAR} ~ post_hhi1 + post_hhi2 + post_hhi3 + {controls_str} | h + year"
result = run_spec(df, formula_hhi, spec_id="did/treatment/hhi_terciles",
                  spec_tree_path="methods/difference_in_differences.md#treatment-definition",
                  treatment_var="post_hhi1", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "HHI tercile 1")

formula_bedshr = f"{OUTCOME_VAR} ~ post_bedshr1 + post_bedshr2 + post_bedshr3 + {controls_str} | h + year"
result = run_spec(df, formula_bedshr, spec_id="did/treatment/bedshr_terciles",
                  spec_tree_path="methods/difference_in_differences.md#treatment-definition",
                  treatment_var="post_bedshr1", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Bed share tercile 1")

formula_sizediff = f"{OUTCOME_VAR} ~ post_sizediff1 + post_sizediff2 + {controls_str} | h + year"
result = run_spec(df, formula_sizediff, spec_id="did/treatment/sizediff",
                  spec_tree_path="methods/difference_in_differences.md#treatment-definition",
                  treatment_var="post_sizediff1", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Size diff 1")

# ----------------------------------------------------------------------------
# 8. INFERENCE VARIATIONS
# ----------------------------------------------------------------------------
print("\n=== 8. INFERENCE VARIATIONS ===")

controls_str = " + ".join(BASELINE_CONTROLS)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | h + year"

# Robust SEs
try:
    model = pf.feols(formula, data=df, vcov='hetero', weights=WEIGHT_VAR)
    coefs, ses, pvals = model.coef(), model.se(), model.pvalue()
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': "robust/cluster/none", 'spec_tree_path': "robustness/clustering_variations.md",
        'outcome_var': OUTCOME_VAR, 'treatment_var': TREATMENT_VAR,
        'coefficient': float(coefs[TREATMENT_VAR]), 'std_error': float(ses[TREATMENT_VAR]),
        't_stat': float(coefs[TREATMENT_VAR] / ses[TREATMENT_VAR]),
        'p_value': float(pvals[TREATMENT_VAR]),
        'ci_lower': float(coefs[TREATMENT_VAR] - 1.96 * ses[TREATMENT_VAR]),
        'ci_upper': float(coefs[TREATMENT_VAR] + 1.96 * ses[TREATMENT_VAR]),
        'n_obs': int(model._N), 'r_squared': float(model._r2),
        'coefficient_vector_json': json.dumps({"note": "Robust SEs"}),
        'sample_desc': "Main sample", 'fixed_effects': "Hospital + Year",
        'controls_desc': "Full controls", 'cluster_var': "None (robust)",
        'model_type': "TWFE", 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    safe_print(result, "Robust SEs")
except Exception as e:
    print(f"  Robust SEs failed: {e}")

# Cluster by year
result = run_spec(df, formula, spec_id="robust/cluster/time",
                  spec_tree_path="robustness/clustering_variations.md",
                  cluster_var="year", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Cluster by year")

# ----------------------------------------------------------------------------
# 9. FUNCTIONAL FORM
# ----------------------------------------------------------------------------
print("\n=== 9. FUNCTIONAL FORM ===")

df_levels = df.copy()
df_levels['prnonmed'] = np.exp(df_levels['lnprnonmed'])
formula_levels = f"prnonmed ~ {TREATMENT_VAR} + {controls_str} | h + year"
result = run_spec(df_levels, formula_levels, spec_id="robust/form/y_level",
                  spec_tree_path="robustness/functional_form.md",
                  outcome_var="prnonmed", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Levels outcome")

df_ihs = df.copy()
df_ihs['prnonmed_ihs'] = np.arcsinh(np.exp(df_ihs['lnprnonmed']))
formula_ihs = f"prnonmed_ihs ~ {TREATMENT_VAR} + {controls_str} | h + year"
result = run_spec(df_ihs, formula_ihs, spec_id="robust/form/y_asinh",
                  spec_tree_path="robustness/functional_form.md",
                  outcome_var="prnonmed_ihs", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "IHS outcome")

df_quad = df.copy()
df_quad['hhi_sq'] = df_quad['hhi'] ** 2
controls_quad = " + ".join(BASELINE_CONTROLS + ['hhi_sq'])
formula_quad = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_quad} | h + year"
result = run_spec(df_quad, formula_quad, spec_id="robust/form/quadratic_hhi",
                  spec_tree_path="robustness/functional_form.md",
                  weight_var=WEIGHT_VAR, controls_desc="Full controls + HHI squared")
results.append(result)
safe_print(result, "Quadratic HHI")

df_quad2 = df.copy()
df_quad2['lnbeds_sq'] = df_quad2['lnbeds'] ** 2
controls_quad2 = " + ".join(BASELINE_CONTROLS + ['lnbeds_sq'])
formula_quad2 = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_quad2} | h + year"
result = run_spec(df_quad2, formula_quad2, spec_id="robust/form/quadratic_beds",
                  spec_tree_path="robustness/functional_form.md",
                  weight_var=WEIGHT_VAR, controls_desc="Full controls + beds squared")
results.append(result)
safe_print(result, "Quadratic beds")

# ----------------------------------------------------------------------------
# 10. WEIGHTS VARIATIONS
# ----------------------------------------------------------------------------
print("\n=== 10. WEIGHTS VARIATIONS ===")

controls_str = " + ".join(BASELINE_CONTROLS)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | h + year"

result = run_spec(df, formula, spec_id="robust/weights/unweighted",
                  spec_tree_path="robustness/measurement.md",
                  weight_var=None, controls_desc="Full controls", sample_desc="Main sample (unweighted)")
results.append(result)
safe_print(result, "Unweighted")

# ----------------------------------------------------------------------------
# 11. EVENT STUDY
# ----------------------------------------------------------------------------
print("\n=== 11. EVENT STUDY ===")

event_vars = ['_m4', '_m3', '_m2', '_p0', '_p1', '_p2', '_p3', '_p4']
event_str = " + ".join(event_vars)
formula_es = f"{OUTCOME_VAR} ~ {event_str} + {controls_str} | h + year"
result = run_spec(df, formula_es, spec_id="did/dynamic/leads_lags",
                  spec_tree_path="methods/difference_in_differences.md#dynamic-effects",
                  treatment_var="_p0", weight_var=WEIGHT_VAR, controls_desc="Full controls + event dummies")
results.append(result)
safe_print(result, "Event study (_p0)")

leads_str = " + ".join(['_m4', '_m3', '_m2'])
formula_leads = f"{OUTCOME_VAR} ~ {leads_str} + {controls_str} | h + year"
result = run_spec(df, formula_leads, spec_id="did/dynamic/leads_only",
                  spec_tree_path="methods/difference_in_differences.md#dynamic-effects",
                  treatment_var="_m2", weight_var=WEIGHT_VAR, controls_desc="Pre-treatment dummies")
results.append(result)
safe_print(result, "Pre-trends (_m2)")

lags_str = " + ".join(['_p0', '_p1', '_p2', '_p3', '_p4'])
formula_lags = f"{OUTCOME_VAR} ~ {lags_str} + {controls_str} | h + year"
result = run_spec(df, formula_lags, spec_id="did/dynamic/lags_only",
                  spec_tree_path="methods/difference_in_differences.md#dynamic-effects",
                  treatment_var="_p4", weight_var=WEIGHT_VAR, controls_desc="Post-treatment dummies")
results.append(result)
safe_print(result, "Post-treatment (_p4)")

# ----------------------------------------------------------------------------
# 12. PLACEBO TESTS
# ----------------------------------------------------------------------------
print("\n=== 12. PLACEBO TESTS ===")

df_pre = df[df['post'] == 0].copy()
formula_pre = f"{OUTCOME_VAR} ~ _m4 + _m3 + _m2 + {controls_str} | h + year"
result = run_spec(df_pre, formula_pre, spec_id="robust/placebo/pre_period_effect",
                  spec_tree_path="robustness/placebo_tests.md",
                  treatment_var="_m2", weight_var=WEIGHT_VAR, sample_desc="Pre-treatment only")
results.append(result)
safe_print(result, "Pre-period placebo")

df_fake = df.copy()
df_fake['post_fake_lead2'] = df_fake.groupby('h')['post'].shift(-2).fillna(0)
formula_fake = f"{OUTCOME_VAR} ~ post_fake_lead2 + {controls_str} | h + year"
result = run_spec(df_fake, formula_fake, spec_id="robust/placebo/fake_treatment_lead2",
                  spec_tree_path="robustness/placebo_tests.md",
                  treatment_var="post_fake_lead2", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Fake treatment (lead 2)")

result = run_spec(df, formula_med, spec_id="robust/placebo/outcome_lnprmed",
                  spec_tree_path="robustness/placebo_tests.md",
                  outcome_var="lnprmed", weight_var=WEIGHT_VAR, controls_desc="Full controls")
results.append(result)
safe_print(result, "Medical prices placebo")

# ----------------------------------------------------------------------------
# 13. HETEROGENEITY ANALYSIS
# ----------------------------------------------------------------------------
print("\n=== 13. HETEROGENEITY ANALYSIS ===")

controls_str = " + ".join(BASELINE_CONTROLS)
formula = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} + {controls_str} | h + year"

# By for-profit status
df_fp = df[df['fp'] == 1].copy()
df_nfp = df[df['fp'] == 0].copy()

result = run_spec(df_fp, formula, spec_id="robust/het/by_fp_forprofit",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, sample_desc="For-profit hospitals only")
results.append(result)
safe_print(result, "For-profit only", include_n=True)

result = run_spec(df_nfp, formula, spec_id="robust/het/by_fp_nonprofit",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, sample_desc="Non-profit hospitals only")
results.append(result)
safe_print(result, "Non-profit only", include_n=True)

# By hospital size
df['beds_tercile'] = pd.qcut(df['lnbeds'], 3, labels=['small', 'medium', 'large'])
for tercile in ['small', 'medium', 'large']:
    df_tercile = df[df['beds_tercile'] == tercile].copy()
    result = run_spec(df_tercile, formula, spec_id=f"robust/het/by_size_{tercile}",
                      spec_tree_path="robustness/heterogeneity.md",
                      weight_var=WEIGHT_VAR, sample_desc=f"Hospital size: {tercile}")
    results.append(result)
    safe_print(result, f"Size {tercile}", include_n=True)

# By HHI
df['hhi_tercile'] = pd.qcut(df['hhi'], 3, labels=['low', 'medium', 'high'])
for tercile in ['low', 'medium', 'high']:
    df_tercile = df[df['hhi_tercile'] == tercile].copy()
    result = run_spec(df_tercile, formula, spec_id=f"robust/het/by_hhi_{tercile}",
                      spec_tree_path="robustness/heterogeneity.md",
                      weight_var=WEIGHT_VAR, sample_desc=f"Market HHI: {tercile}")
    results.append(result)
    safe_print(result, f"HHI {tercile}", include_n=True)

# By system membership
df_sys = df[df['sysoth'] == 1].copy()
df_nosys = df[df['sysoth'] == 0].copy()

result = run_spec(df_sys, formula, spec_id="robust/het/by_system_member",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, sample_desc="System members only")
results.append(result)
safe_print(result, "System member", include_n=True)

result = run_spec(df_nosys, formula, spec_id="robust/het/by_system_nonmember",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, sample_desc="Non-system hospitals only")
results.append(result)
safe_print(result, "Non-system", include_n=True)

# Interactions
formula_int_fp = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} * fp + lncmi + pctmcaid + lnbeds + hhi + sysoth | h + year"
result = run_spec(df, formula_int_fp, spec_id="robust/het/interaction_fp",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, controls_desc="Full controls + treatment x fp")
results.append(result)
safe_print(result, "Interaction (fp)")

formula_int_hhi = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} * hhi + lncmi + pctmcaid + lnbeds + fp + sysoth | h + year"
result = run_spec(df, formula_int_hhi, spec_id="robust/het/interaction_hhi",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, controls_desc="Full controls + treatment x hhi")
results.append(result)
safe_print(result, "Interaction (hhi)")

formula_int_beds = f"{OUTCOME_VAR} ~ {TREATMENT_VAR} * lnbeds + lncmi + pctmcaid + fp + hhi + sysoth | h + year"
result = run_spec(df, formula_int_beds, spec_id="robust/het/interaction_beds",
                  spec_tree_path="robustness/heterogeneity.md",
                  weight_var=WEIGHT_VAR, controls_desc="Full controls + treatment x beds")
results.append(result)
safe_print(result, "Interaction (beds)")

# ----------------------------------------------------------------------------
# 14. ADDITIONAL SPECIFICATIONS
# ----------------------------------------------------------------------------
print("\n=== 14. ADDITIONAL SPECIFICATIONS ===")

# Matched sample combinations
formula_active = f"{OUTCOME_VAR} ~ post_active + post_passive + {controls_str} | h + year"
result = run_spec(df_matched, formula_active, spec_id="did/sample/matched_active_passive",
                  spec_tree_path="methods/difference_in_differences.md",
                  treatment_var="post_active", weight_var=WEIGHT_VAR, sample_desc="Matched + active/passive")
results.append(result)
safe_print(result, "Matched + active")

formula_state = f"{OUTCOME_VAR} ~ post_instate + post_outstate + {controls_str} | h + year"
result = run_spec(df_matched, formula_state, spec_id="did/sample/matched_instate_outstate",
                  spec_tree_path="methods/difference_in_differences.md",
                  treatment_var="post_instate", weight_var=WEIGHT_VAR, sample_desc="Matched + in/out state")
results.append(result)
safe_print(result, "Matched + instate")

# Same-system combinations
result = run_spec(df_samesys, formula_active, spec_id="did/sample/samesys_active_passive",
                  spec_tree_path="methods/difference_in_differences.md",
                  treatment_var="post_active", weight_var=WEIGHT_VAR, sample_desc="Same-system + active/passive")
results.append(result)
safe_print(result, "Same-sys + active")

result = run_spec(df_samesys, formula_state, spec_id="did/sample/samesys_instate_outstate",
                  spec_tree_path="methods/difference_in_differences.md",
                  treatment_var="post_instate", weight_var=WEIGHT_VAR, sample_desc="Same-system + in/out state")
results.append(result)
safe_print(result, "Same-sys + instate")

# Indirect hospitals
df_with_indirect = df_full.copy()
df_with_indirect['year'] = df_with_indirect['year'].astype(int)
df_with_indirect['h'] = df_with_indirect['h'].astype(int)
formula_indirect = f"{OUTCOME_VAR} ~ post + indpost + {controls_str} | h + year"
result = run_spec(df_with_indirect, formula_indirect, spec_id="did/treatment/indirect",
                  spec_tree_path="methods/difference_in_differences.md",
                  treatment_var="indpost", weight_var=WEIGHT_VAR, sample_desc="Full sample including indirect")
results.append(result)
safe_print(result, "Indirect effect")

# Medical prices on subsamples
result = run_spec(df_matched, formula_med, spec_id="robust/outcome/lnprmed_matched",
                  spec_tree_path="robustness/measurement.md",
                  outcome_var="lnprmed", weight_var=WEIGHT_VAR, sample_desc="Matched sample")
results.append(result)
safe_print(result, "Medical prices (matched)")

result = run_spec(df_samesys, formula_med, spec_id="robust/outcome/lnprmed_samesys",
                  spec_tree_path="robustness/measurement.md",
                  outcome_var="lnprmed", weight_var=WEIGHT_VAR, sample_desc="Same-system controls")
results.append(result)
safe_print(result, "Medical prices (same-sys)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n=== SAVING RESULTS ===")

results_df = pd.DataFrame(results)
output_path = OUTPUT_DIR + "specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved {len(results_df)} specifications to {output_path}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

valid = results_df[results_df['coefficient'].notna()]

print(f"\nTotal specifications: {len(results_df)}")
print(f"Valid specifications: {len(valid)}")

if len(valid) > 0:
    print(f"\nCoefficient Statistics:")
    print(f"  Mean: {valid['coefficient'].mean():.4f}")
    print(f"  Median: {valid['coefficient'].median():.4f}")
    print(f"  Std Dev: {valid['coefficient'].std():.4f}")
    print(f"  Min: {valid['coefficient'].min():.4f}")
    print(f"  Max: {valid['coefficient'].max():.4f}")

    sig_05 = (valid['p_value'] < 0.05).sum()
    sig_01 = (valid['p_value'] < 0.01).sum()
    positive = (valid['coefficient'] > 0).sum()

    print(f"\nSignificance:")
    print(f"  Significant at 5%: {sig_05} ({100*sig_05/len(valid):.1f}%)")
    print(f"  Significant at 1%: {sig_01} ({100*sig_01/len(valid):.1f}%)")
    print(f"  Positive coefficients: {positive} ({100*positive/len(valid):.1f}%)")

print("\n" + "="*60)
print("SPECIFICATION SEARCH COMPLETE")
print("="*60)
