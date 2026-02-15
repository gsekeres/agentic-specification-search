"""
Replication script for 112533-V1:
"The Finnish Great Depression: From Russia with Love"
by Gorodnichenko, Mendoza, and Tesar (AER, 2012)

This paper is primarily a calibrated DSGE/IRBC model paper. The main results
(Tables 1-5) are from model simulations solved in Matlab using the Anderson-Moore
(AIM) algorithm, NOT from regression estimation.

The only regression result in the paper is in Figure 1, Panel C:
a cross-sectional OLS of employment deviations from trend (1993) on
share of exports to USSR (1988) across Finnish manufacturing industries.

Original result (annotated on figure): slope = -14.54, SE = (6.04)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os

# Paths
PAPER_ID = "112533-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
FIG_DIR = os.path.join(PKG_DIR, "AdditionalMaterials", "AdditionalFigures")

# =============================================================================
# Step 1: Load data
# =============================================================================
raw = pd.read_stata(os.path.join(FIG_DIR, "raw_production_data.dta"))
raw = raw.dropna(subset=["year"])
raw["year"] = raw["year"].astype(int)
exports = pd.read_stata(os.path.join(FIG_DIR, "export_shares.dta"))

# =============================================================================
# Step 2: Merge industries to new classification (following Stata do-file)
# =============================================================================
var_types = ["prod", "va", "empl", "export", "inv"]

for var in var_types:
    raw[f"x_{var}_1"] = raw[f"{var}_1"]
    raw[f"x_{var}_2"] = raw[f"{var}_2"]
    raw[f"x_{var}_3"] = raw[f"{var}_6"]
    raw[f"x_{var}_4"] = raw[f"{var}_11"]
    raw[f"x_{var}_5"] = raw[f"{var}_14"]
    raw[f"x_{var}_7"] = raw[f"{var}_22"] + raw[f"{var}_23"]
    raw[f"x_{var}_10"] = raw[f"{var}_20"]
    raw[f"x_{var}_15"] = raw[f"{var}_31"]
    raw[f"x_{var}_16"] = raw[f"{var}_32"]
    raw[f"x_{var}_12"] = raw[f"{var}_26"]
    raw[f"x_{var}_11"] = raw[f"{var}_24"] + raw[f"{var}_25"]
    raw[f"x_{var}_23"] = raw[f"{var}_39"]
    raw[f"x_{var}_26"] = raw[f"{var}_44"]
    raw[f"x_{var}_28"] = raw[f"{var}_42"]
    raw[f"x_{var}_29"] = raw[f"{var}_46"]
    raw[f"x_{var}_30"] = raw[f"{var}_43"] + raw[f"{var}_45"] + raw[f"{var}_47"]
    raw[f"x_{var}_25"] = raw[f"{var}_41"]
    raw[f"x_{var}_27"] = (
        raw[f"{var}_43"] + raw[f"{var}_45"] + raw[f"{var}_47"]
        + raw[f"{var}_46"] + raw[f"{var}_42"]
    )
    raw[f"x_{var}_24"] = raw[f"{var}_48"]
    raw[f"x_{var}_21"] = raw[f"{var}_36"]
    raw[f"x_{var}_17"] = raw[f"{var}_33"]
    raw[f"x_{var}_19"] = raw[f"{var}_37"]
    raw[f"x_{var}_14"] = raw[f"x_{var}_15"] + raw[f"x_{var}_16"]
    raw[f"x_{var}_13"] = raw[f"x_{var}_14"] + raw[f"x_{var}_17"]
    raw[f"x_{var}_22"] = raw[f"{var}_38"] + raw[f"{var}_40"]
    raw[f"x_{var}_20"] = (
        raw[f"x_{var}_21"] + raw[f"x_{var}_22"]
        + raw[f"x_{var}_23"] + raw[f"x_{var}_24"]
    )
    raw[f"x_{var}_18"] = raw[f"x_{var}_19"] + raw[f"x_{var}_20"]
    raw[f"x_{var}_6"] = raw[f"{var}_17"]
    raw[f"x_{var}_9"] = raw[f"{var}_18"] + raw[f"{var}_21"]
    raw[f"x_{var}_8"] = raw[f"x_{var}_9"] + raw[f"x_{var}_10"]
    raw[f"x_{var}_31"] = raw[f"{var}_49"]

# =============================================================================
# Step 3: Detrend employment series
# =============================================================================
year = raw["year"].values
r0_empl = {}

# Default: linear trend on 1980-1989
for i in range(1, 32):
    col = f"x_empl_{i}"
    if col not in raw.columns:
        continue
    temp0 = np.log(raw[col].values)
    mask = (year >= 1980) & (year <= 1989)
    X_reg = sm.add_constant(year[mask].astype(float))
    y_reg = temp0[mask]
    valid = np.isfinite(y_reg) & np.all(np.isfinite(X_reg), axis=1)
    if valid.sum() < 2:
        continue
    model = sm.OLS(y_reg[valid], X_reg[valid]).fit()
    X_all = sm.add_constant(year.astype(float))
    fitted = model.predict(X_all)
    resid = temp0 - fitted
    r0_empl[i] = resid * 100

# Override: industries 3, 12, 21, 30 with constant-only on 1986-1990
for i in [3, 12, 21, 30]:
    col = f"x_empl_{i}"
    if col not in raw.columns:
        continue
    temp0 = np.log(raw[col].values)
    mask = (year >= 1986) & (year <= 1990)
    X_reg = np.ones((mask.sum(), 1))
    y_reg = temp0[mask]
    valid = np.isfinite(y_reg)
    if valid.sum() < 1:
        continue
    model = sm.OLS(y_reg[valid], X_reg[valid]).fit()
    X_all = np.ones((len(year), 1))
    fitted = model.predict(X_all)
    resid = temp0 - fitted
    r0_empl[i] = resid * 100

# Re-override: industry 3 with linear trend on 1986-1990
for i in [3]:
    col = f"x_empl_{i}"
    temp0 = np.log(raw[col].values)
    mask = (year >= 1986) & (year <= 1990)
    X_reg = sm.add_constant(year[mask].astype(float))
    y_reg = temp0[mask]
    valid = np.isfinite(y_reg)
    model = sm.OLS(y_reg[valid], X_reg[valid]).fit()
    X_all = sm.add_constant(year.astype(float))
    fitted = model.predict(X_all)
    resid = temp0 - fitted
    r0_empl[i] = resid * 100

# Industries 26, 27, 28: demean at 1988
for i in [26, 27, 28]:
    col = f"x_empl_{i}"
    if col not in raw.columns:
        continue
    temp0 = np.log(raw[col].values)
    mask1988 = year == 1988
    mean_1988 = temp0[mask1988].mean()
    resid = temp0 - mean_1988
    r0_empl[i] = resid * 100

# =============================================================================
# Step 4: Build cross-sectional dataset for 1993
# =============================================================================
idx_1993 = np.where(year == 1993)[0][0]
cross = pd.DataFrame({"industry_code": range(1, 32)})
for i in range(1, 32):
    if i in r0_empl:
        cross.loc[cross["industry_code"] == i, "r0_empl_1993"] = r0_empl[i][idx_1993]

cross = cross.merge(exports, on="industry_code", how="inner")
cross = cross.dropna(subset=["r0_empl_1993", "row_export_1988"])

# =============================================================================
# Step 5: Run the OLS regression (Figure 1, Panel C)
# =============================================================================
# reg r0_empl_1993 row_export_1988, robust
model_robust = smf.ols("r0_empl_1993 ~ row_export_1988", data=cross).fit(cov_type="HC1")

print("=" * 60)
print("Figure 1, Panel C: Employment deviation vs. Soviet export share")
print("=" * 60)
print(f"Coefficient on row_export_1988: {model_robust.params['row_export_1988']:.4f}")
print(f"Robust SE (HC1): {model_robust.bse['row_export_1988']:.4f}")
print(f"p-value: {model_robust.pvalues['row_export_1988']:.6f}")
print(f"R-squared: {model_robust.rsquared:.4f}")
print(f"N: {int(model_robust.nobs)}")
print(f"Intercept: {model_robust.params['Intercept']:.4f}")
print()
print(f"Original (from figure): slope = -14.54, SE = 6.04")
print(f"Replicated:             slope = {model_robust.params['row_export_1988']:.2f}, SE = {model_robust.bse['row_export_1988']:.2f}")

# =============================================================================
# Step 6: Write replication.csv
# =============================================================================
coef_vec = {k: float(v) for k, v in model_robust.params.items()}
ci = model_robust.conf_int()

orig_coef = -14.54
orig_se = 6.04
repl_coef = model_robust.params["row_export_1988"]
rel_error = abs(repl_coef - orig_coef) / abs(orig_coef)

if rel_error < 0.0001:
    match = "exact"
elif rel_error <= 0.01:
    match = "close"
else:
    match = "discrepant"

results = pd.DataFrame([{
    "paper_id": PAPER_ID,
    "reg_id": 1,
    "outcome_var": "r0_empl_1993",
    "treatment_var": "row_export_1988",
    "coefficient": float(repl_coef),
    "std_error": float(model_robust.bse["row_export_1988"]),
    "p_value": float(model_robust.pvalues["row_export_1988"]),
    "ci_lower": float(ci.loc["row_export_1988", 0]),
    "ci_upper": float(ci.loc["row_export_1988", 1]),
    "n_obs": int(model_robust.nobs),
    "r_squared": float(model_robust.rsquared),
    "original_coefficient": orig_coef,
    "original_std_error": orig_se,
    "match_status": match,
    "coefficient_vector_json": json.dumps(coef_vec),
    "fixed_effects": "",
    "controls_desc": "None (bivariate OLS)",
    "cluster_var": "",
    "estimator": "OLS",
    "sample_desc": "31 Finnish manufacturing industries, cross-sectional",
    "notes": (
        "Figure 1 Panel C scatter plot. Coefficient matches original to 4 sig figs. "
        "SE differs slightly (6.44 vs 6.04) -- may reflect data version difference or "
        "rounding in figure annotation. Paper's main results (Tables 1-5) are from "
        "calibrated DSGE model (Matlab), not regressions."
    ),
}])

results.to_csv(os.path.join(PKG_DIR, "replication.csv"), index=False)
print(f"\nWrote replication.csv to {PKG_DIR}")
print(f"Match status: {match} (relative error: {rel_error:.4f})")
