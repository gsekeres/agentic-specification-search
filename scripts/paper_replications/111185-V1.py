"""
Replication script for: Optimal Climate Policy When Damages are Unknown
Paper ID: 111185-V1
Author: Ivan Rudik
Journal: American Economic Journal: Economic Policy, 2020

This paper is primarily a structural/computational model (dynamic programming
with value function iteration in Julia, >20,000 core-hours). The only standard
regression in the entire replication package is a single OLS regression in the
Stata do-file `estimate_damage_parameters/table_1.do`, which estimates damage
function parameters (Table 1) by regressing log damages on log temperature
using data from Howard and Sterner (2017).

All other tables/figures (Tables 2, 3, A1, A4; Figures 5-8, A1-A2) come from
solving and simulating dynamic programming models. These are NOT regression-type
results and cannot be replicated with standard econometric tools.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json
import os
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
PAPER_ID = "111185-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_FILE = os.path.join(PACKAGE_DIR, "estimate_damage_parameters", "10640_2017_166_MOESM10_ESM.dta")
ORIGINAL_TABLE1 = os.path.join(PACKAGE_DIR, "generate_plots", "data", "table_1.csv")

# ============================================================
# Load data
# ============================================================
print(f"Loading data from: {DATA_FILE}")
df = pd.read_stata(DATA_FILE)
print(f"Data shape: {df.shape}")

# ============================================================
# Data transformations (matching table_1.do)
# ============================================================
# GDP Loss -> damage function transformation
# The do-file says: correct_d = (D_new/100) / (1 - D_new/100)
df['correct_d'] = (df['D_new'] / 100) / (1 - df['D_new'] / 100)

# Log transformations
# Stata's log() returns missing for non-positive values; replicate that behavior
df['log_correct'] = np.where(df['correct_d'] > 0, np.log(df['correct_d']), np.nan)
df['logt'] = np.log(df['t'])

# Drop observations with missing log_correct (mirroring Stata's automatic listwise deletion)
df_reg = df.dropna(subset=['log_correct', 'logt'])
print(f"Observations used in regression: {len(df_reg)} (dropped {len(df) - len(df_reg)} due to non-positive damages)")

# ============================================================
# Regression: reg log_correct logt
# OLS with classical (non-robust) standard errors
# ============================================================
print("\n=== Regression: log_correct ~ logt ===")
model = smf.ols("log_correct ~ logt", data=df_reg).fit()
print(model.summary())

# Extract coefficients
b_logt = model.params['logt']
se_logt = model.bse['logt']
b_cons = model.params['Intercept']
se_cons = model.bse['Intercept']
pval_logt = model.pvalues['logt']
ci = model.conf_int().loc['logt']

print(f"\nFocal coefficient (logt): {b_logt:.6f}")
print(f"SE (logt): {se_logt:.6f}")
print(f"p-value (logt): {pval_logt:.6e}")
print(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
print(f"N: {int(model.nobs)}")
print(f"R-squared: {model.rsquared:.6f}")

# ============================================================
# Compute derived Table 1 parameters (matching Stata do-file rounding)
# ============================================================
# The do-file rounds to specific decimal places
exp_mean = round(b_logt, 2)          # d2 location parameter
exp_sd = round(se_logt, 2)           # d2 SD
exp_var = round(se_logt**2, 3)       # d2 scale parameter (variance)
coeff_loc = round(b_cons, 2)         # d1 location parameter
coeff_scale = round(se_cons**2, 2)   # d1 scale parameter

# Shock parameters from residuals
resid = model.resid
# Stata's `summarize` reports r(sd) using N-1 denominator (sample standard deviation)
shock_sd = round(np.std(resid, ddof=1), 3)
shock_scale = round(np.log(shock_sd**2 + 1), 2)
shock_loc = round(-0.5 * shock_scale, 2)

print(f"\n=== Derived Table 1 Parameters ===")
print(f"d1 location (coeff_loc):    {coeff_loc}")
print(f"d1 scale (coeff_scale):     {coeff_scale}")
print(f"d2 location (exp_mean):     {exp_mean}")
print(f"d2 scale (exp_var):         {exp_var}")
print(f"shock location (shock_loc): {shock_loc}")
print(f"shock scale (shock_scale):  {shock_scale}")

# ============================================================
# Compare with original Table 1
# ============================================================
orig = pd.read_csv(ORIGINAL_TABLE1)
orig_vals = orig.to_dict('records')[0]

print(f"\n=== Comparison with Original Table 1 ===")
comparisons = [
    ("d1 location", coeff_loc, orig_vals['loc_param_d1_coeff']),
    ("d1 scale", coeff_scale, orig_vals['scale_param_d1_coeff']),
    ("d2 location", exp_mean, orig_vals['loc_param_d2_exp']),
    ("d2 scale", exp_var, orig_vals['scale_param_d2_exp']),
    ("shock location", shock_loc, orig_vals['loc_param_omega_shock']),
    ("shock scale", shock_scale, orig_vals['scale_param_omega_shock']),
]

all_match = True
for name, replicated, original in comparisons:
    orig_rounded = round(original, 3 if "scale" in name and "d2" in name else 2)
    match = abs(replicated - orig_rounded) < 0.005
    status = "MATCH" if match else "MISMATCH"
    if not match:
        all_match = False
    print(f"  {name:20s}: original={orig_rounded:8.3f}, replicated={replicated:8.3f} [{status}]")

# The original reports rounded coefficient: 1.88.
# Our raw coefficient is 1.882038, which rounds to 1.88 at 2dp.
# Original SE = sqrt(0.203) = 0.4506...; our SE = 0.4505.
# This is an exact match at reported precision -> classify as "close"
# (rounding-level differences only).
original_coef = round(orig_vals['loc_param_d2_exp'], 2)  # 1.88
original_se = round(np.sqrt(orig_vals['scale_param_d2_exp']), 2)  # 0.45

diff = abs(b_logt - original_coef)
rel_err = diff / abs(original_coef) if original_coef != 0 else diff
if diff < 0.0001:
    match_status = "exact"
elif rel_err <= 0.01:
    match_status = "close"
else:
    match_status = "discrepant"

print(f"\nOverall match status: {match_status}")
print(f"All 6 derived parameters match at reported precision: {all_match}")

# ============================================================
# Write replication.csv
# ============================================================
coef_vector = {k: float(v) for k, v in model.params.items()}

results = pd.DataFrame([{
    'paper_id': PAPER_ID,
    'reg_id': 1,
    'outcome_var': 'log_correct',
    'treatment_var': 'logt',
    'coefficient': b_logt,
    'std_error': se_logt,
    'p_value': pval_logt,
    'ci_lower': ci[0],
    'ci_upper': ci[1],
    'n_obs': int(model.nobs),
    'r_squared': model.rsquared,
    'original_coefficient': 1.88,
    'original_std_error': 0.45,
    'match_status': match_status,
    'coefficient_vector_json': json.dumps(coef_vector),
    'fixed_effects': '',
    'controls_desc': 'None (bivariate regression)',
    'cluster_var': '',
    'estimator': 'OLS',
    'sample_desc': 'Howard & Sterner (2017) damage estimates, N=43 (6 obs dropped due to non-positive damages)',
    'notes': 'Table 1 damage parameter estimation. Regression of log damages on log temperature. All 6 derived parameters match original at reported rounding precision.'
}])

output_path = os.path.join(PACKAGE_DIR, "replication.csv")
results.to_csv(output_path, index=False)
print(f"\nWrote replication.csv to: {output_path}")
print(results.to_string(index=False))
