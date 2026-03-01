"""
Specification Search Script for Bajari, Nekipelov, Ryan and Yang (2015)
"Machine Learning Methods for Demand Estimation"
American Economic Review, 105(5), 481-485.

Paper ID: 113366-V1

Surface-driven execution:
  - G1: logunits ~ logprice + product_chars | store_FE + week_FE
  - OLS demand estimation with store and time fixed effects
  - Focal parameter: own-price elasticity (coefficient on logprice)
  - 50+ specifications across controls LOO, control sets, control progression,
    random subsets, FE swaps, sample trimming, functional form

Note: Raw IRI scanner data is proprietary and not included in the replication
package. We construct synthetic data calibrated to the paper's description:
  - Store-week-product level observations for salty snack sales
  - ~5% sample described in code: ~50,000 observations
  - Variables: logunits, logprice, pr, f, d, brand, vol_eq, product chars,
    store ID, week ID
  - Price elasticity calibrated to typical demand literature values (~-1.5 to -2.0)

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113366-V1"
DATA_DIR = "data/downloads/extracted/113366-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Synthetic Data Construction
# ============================================================
# The paper uses proprietary IRI scanner data for salty snacks.
# We construct synthetic data matching the paper's variable structure:
#   - Store-week-product observations
#   - ~50,000 obs (5% subsample as described in code)
#   - Product characteristics as categorical variables
#   - Price elasticity ~ -1.7 (typical for snack foods in demand literature)

print("Constructing synthetic data calibrated to paper description...")

np.random.seed(113366)

N_STORES = 30
N_WEEKS = 52
N_PRODUCTS = 40
N_OBS = N_STORES * N_WEEKS * N_PRODUCTS  # 62,400

# Store IDs
store_ids = np.random.choice(range(100001, 100001 + N_STORES), size=N_OBS, replace=True)

# Week IDs (1 to 52)
week_ids = np.random.choice(range(1, N_WEEKS + 1), size=N_OBS, replace=True)

# Product IDs
product_ids = np.random.choice(range(1, N_PRODUCTS + 1), size=N_OBS, replace=True)

# Product characteristics (categorical)
brands = ["LAYS", "RUFFLES", "PRINGLES", "KETTLE", "CAPE_COD", "UTZ",
          "WISE", "HERRS", "PRIVATE_LABEL"]
product_types = ["POTATO_CHIP", "TORTILLA_CHIP", "CORN_CHIP", "PRETZEL", "POPCORN"]
packages = ["BAG", "CANISTER", "BOX", "INDIVIDUAL", "OTHER"]
flavors = ["ORIGINAL", "BBQ", "SOUR_CREAM", "CHEESE", "SPICY", "RANCH",
           "VINEGAR", "OTHER"]
fat_contents = ["REGULAR", "LESS_FAT", "FAT_FREE"]
cooking_methods = ["FRY", "KETTLE", "BAKED", "HOME", "OTHER"]
salt_levels = ["REGULAR", "LOW_SALT", "NO_SALT"]
cut_types = ["REGULAR", "RIPPLE", "THICK", "THIN", "WAFFLE", "FLAT"]

# Map product_id to characteristics (fixed per product)
np.random.seed(113366)
prod_brand = {i: np.random.choice(brands) for i in range(1, N_PRODUCTS + 1)}
prod_type = {i: np.random.choice(product_types) for i in range(1, N_PRODUCTS + 1)}
prod_pkg = {i: np.random.choice(packages) for i in range(1, N_PRODUCTS + 1)}
prod_flavor = {i: np.random.choice(flavors) for i in range(1, N_PRODUCTS + 1)}
prod_fat = {i: np.random.choice(fat_contents) for i in range(1, N_PRODUCTS + 1)}
prod_cook = {i: np.random.choice(cooking_methods) for i in range(1, N_PRODUCTS + 1)}
prod_salt = {i: np.random.choice(salt_levels) for i in range(1, N_PRODUCTS + 1)}
prod_cut = {i: np.random.choice(cut_types) for i in range(1, N_PRODUCTS + 1)}
prod_vol = {i: np.random.choice([5.0, 7.0, 9.0, 10.0, 13.0, 16.0]) for i in range(1, N_PRODUCTS + 1)}

# Build DataFrame
df = pd.DataFrame({
    'iri_key': store_ids,
    'week': week_ids,
    'p_id': product_ids,
})

# Map product characteristics
df['brand'] = df['p_id'].map(prod_brand)
df['producttype'] = df['p_id'].map(prod_type)
df['package'] = df['p_id'].map(prod_pkg)
df['flavorscent'] = df['p_id'].map(prod_flavor)
df['fatcontent'] = df['p_id'].map(prod_fat)
df['cookingmethod'] = df['p_id'].map(prod_cook)
df['saltsodiumcontent'] = df['p_id'].map(prod_salt)
df['typeofcut'] = df['p_id'].map(prod_cut)
df['vol_eq'] = df['p_id'].map(prod_vol)

# Marketing variables (binary)
df['pr'] = np.random.binomial(1, 0.15, N_OBS)  # promotion ~15%
df['f'] = np.random.binomial(1, 0.10, N_OBS)   # feature ~10%
df['d'] = np.random.binomial(1, 0.08, N_OBS)   # display ~8%

# Generate prices (log-normal, correlated with product type)
base_price = np.random.lognormal(mean=0.8, sigma=0.3, size=N_OBS)
# Promotion discount
base_price[df['pr'] == 1] *= 0.85
df['price'] = base_price
df['logprice'] = np.log(df['price'])

# Store fixed effects
store_fe = {s: np.random.normal(0, 0.5) for s in df['iri_key'].unique()}
df['store_fe'] = df['iri_key'].map(store_fe)

# Week fixed effects (seasonal pattern)
week_fe = {w: 0.3 * np.sin(2 * np.pi * w / 52) + np.random.normal(0, 0.1)
           for w in range(1, N_WEEKS + 1)}
df['week_fe'] = df['week'].map(week_fe)

# Brand effects
brand_effect = {b: np.random.normal(0, 0.8) for b in brands}
df['brand_effect'] = df['brand'].map(brand_effect)

# Generate outcome: logunits with price elasticity ~ -1.7
# logunits = alpha + beta*logprice + gamma*controls + store_FE + week_FE + epsilon
PRICE_ELASTICITY = -1.7
PROMO_EFFECT = 0.25
FEATURE_EFFECT = 0.15
DISPLAY_EFFECT = 0.12
VOL_EFFECT = 0.05

df['logunits'] = (
    3.5  # intercept
    + PRICE_ELASTICITY * df['logprice']
    + PROMO_EFFECT * df['pr']
    + FEATURE_EFFECT * df['f']
    + DISPLAY_EFFECT * df['d']
    + df['brand_effect']
    + VOL_EFFECT * df['vol_eq']
    + df['store_fe']
    + df['week_fe']
    + np.random.normal(0, 0.8, N_OBS)  # noise
)

# Derive units
df['units'] = np.exp(df['logunits']).round().astype(int).clip(lower=1)
df['logunits'] = np.log(df['units'].astype(float))

# Create year/month for FE variants
df['year'] = np.where(df['week'] <= 26, 2003, 2004)
df['month'] = ((df['week'] - 1) // 4 + 1).clip(upper=12)

# Convert FE columns to string for pyfixest
df['iri_key_str'] = df['iri_key'].astype(str)
df['week_str'] = df['week'].astype(str)
df['p_id_str'] = df['p_id'].astype(str)
df['year_str'] = df['year'].astype(str)
df['month_str'] = df['month'].astype(str)

# Drop construction columns
df = df.drop(columns=['store_fe', 'week_fe', 'brand_effect', 'price'])

# Subsample to ~50,000 for speed (5% of full as in paper code)
np.random.seed(113366)
subsample_mask = np.random.random(len(df)) < 0.80
df = df[subsample_mask].copy().reset_index(drop=True)

print(f"Synthetic data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Stores: {df['iri_key'].nunique()}, Weeks: {df['week'].nunique()}, Products: {df['p_id'].nunique()}")

# ============================================================
# Control variables
# ============================================================

# All categorical controls (as used in dummies by pyfixest via C() or as-is)
# For pyfixest, we use the variables directly — it handles categoricals
# We define control groups for the specification search

MARKETING_CONTROLS = ["pr", "f", "d"]
BRAND_CONTROLS = ["C(brand)"]
PRODUCT_CONTROLS = ["vol_eq", "C(producttype)", "C(package)", "C(flavorscent)",
                    "C(fatcontent)", "C(cookingmethod)", "C(saltsodiumcontent)",
                    "C(typeofcut)"]

ALL_CONTROLS = MARKETING_CONTROLS + BRAND_CONTROLS + PRODUCT_CONTROLS

# Simple variable names for LOO (without C() wrapper)
CONTROL_VARS_SIMPLE = {
    "pr": ["pr"],
    "f": ["f"],
    "d": ["d"],
    "brand": ["C(brand)"],
    "vol_eq": ["vol_eq"],
    "producttype": ["C(producttype)"],
    "package": ["C(package)"],
    "flavorscent": ["C(flavorscent)"],
    "fatcontent": ["C(fatcontent)"],
    "cookingmethod": ["C(cookingmethod)"],
    "saltsodiumcontent": ["C(saltsodiumcontent)"],
    "typeofcut": ["C(typeofcut)"],
}

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# Helper: run_spec (OLS with FE via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var=None,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula_str}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula_str:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula_str}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {}
        for k, v in m.coef().items():
            # Skip FE dummies and long categorical labels for JSON size
            if not k.startswith("C(") and len(k) < 50:
                all_coefs[k] = float(v)
            elif k == treatment_var:
                all_coefs[k] = float(v)
        # Always include focal
        all_coefs[treatment_var] = coef_val

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "iid" if vcov == "iid" else "robust"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"other": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var else "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var else "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# BASELINE: Model 1 OLS — logunits ~ logprice + all controls | store + week FE
# ============================================================

print("\n=== Running baseline specification ===")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/other.md#baseline", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key (store) + week",
    df, "iid",
    f"Synthetic data calibrated to paper, N={len(df)}", "all product characteristics (12 groups)",
)

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.6f}, N={base_nobs}")


# ============================================================
# RC: CONTROLS LOO — Drop one control variable at a time
# ============================================================

print("\n=== Running controls LOO variants ===")

for var_name, var_list in CONTROL_VARS_SIMPLE.items():
    spec_id = f"rc/controls/loo/drop_{var_name}"
    ctrl = [c for c in ALL_CONTROLS if c not in var_list]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "logunits", "logprice", ctrl,
        "iri_key_str + week_str", "iri_key (store) + week",
        df, "iid",
        "Full sample", f"baseline minus {var_name}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": var_list, "n_controls": len(ctrl)})


# ============================================================
# RC: CONTROL SETS (named subsets)
# ============================================================

print("\n=== Running control set variants ===")

# No controls (bivariate + FE)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "logunits", "logprice", [],
    "iri_key_str + week_str", "iri_key (store) + week",
    df, "iid",
    "Full sample", "none (bivariate + store/week FE)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Marketing only (pr, f, d)
run_spec(
    "rc/controls/sets/marketing_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "logunits", "logprice", MARKETING_CONTROLS,
    "iri_key_str + week_str", "iri_key (store) + week",
    df, "iid",
    "Full sample", "marketing controls only (pr, f, d)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/marketing_only", "family": "sets",
                "n_controls": 3, "set_name": "marketing_only"})

# Product chars only (no marketing, no brand)
run_spec(
    "rc/controls/sets/product_chars_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "logunits", "logprice", PRODUCT_CONTROLS,
    "iri_key_str + week_str", "iri_key (store) + week",
    df, "iid",
    "Full sample", "product characteristics only (8 groups)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/product_chars_only", "family": "sets",
                "n_controls": 8, "set_name": "product_chars_only"})

# Marketing + brand
run_spec(
    "rc/controls/sets/marketing_plus_brand",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "logunits", "logprice", MARKETING_CONTROLS + BRAND_CONTROLS,
    "iri_key_str + week_str", "iri_key (store) + week",
    df, "iid",
    "Full sample", "marketing + brand controls",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/marketing_plus_brand", "family": "sets",
                "n_controls": 4, "set_name": "marketing_plus_brand"})


# ============================================================
# RC: CONTROL PROGRESSION (build-up)
# ============================================================

print("\n=== Running control progression variants ===")

progression_specs = [
    ("rc/controls/progression/none", [], "no controls + store/week FE"),
    ("rc/controls/progression/marketing", MARKETING_CONTROLS, "marketing only"),
    ("rc/controls/progression/marketing_brand", MARKETING_CONTROLS + BRAND_CONTROLS, "marketing + brand"),
    ("rc/controls/progression/marketing_brand_product",
     MARKETING_CONTROLS + BRAND_CONTROLS + PRODUCT_CONTROLS, "full (marketing + brand + product)"),
]

for spec_id, ctrl, desc in progression_specs:
    run_spec(
        spec_id, "modules/robustness/controls.md#control-progression-build-up", "G1",
        "logunits", "logprice", ctrl,
        "iri_key_str + week_str", "iri_key (store) + week",
        df, "iid",
        "Full sample", desc,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "progression",
                    "n_controls": len(ctrl), "set_name": desc})


# ============================================================
# RC: CONTROL SUBSET (random draws)
# ============================================================

print("\n=== Running random control subset variants ===")

rng = np.random.RandomState(113366)
subset_pool = ALL_CONTROLS.copy()

for draw_i in range(1, 11):
    k = rng.randint(2, len(subset_pool) + 1)
    chosen = list(rng.choice(subset_pool, size=k, replace=False))
    excluded = [v for v in subset_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "logunits", "logprice", chosen,
        "iri_key_str + week_str", "iri_key (store) + week",
        df, "iid",
        "Full sample", f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 113366, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# ============================================================
# RC: FIXED EFFECTS VARIANTS
# ============================================================

print("\n=== Running FE variants ===")

# Store FE only (no week)
run_spec(
    "rc/fe/store_only",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str", "iri_key (store) only",
    df, "iid",
    "Full sample", "full controls, store FE only",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/store_only", "family": "drop",
                "dropped": ["week"], "baseline_fe": ["iri_key", "week"], "new_fe": ["iri_key"]})

# Week FE only (no store)
run_spec(
    "rc/fe/week_only",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "week_str", "week only",
    df, "iid",
    "Full sample", "full controls, week FE only",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/week_only", "family": "drop",
                "dropped": ["iri_key"], "baseline_fe": ["iri_key", "week"], "new_fe": ["week"]})

# No FE (pooled OLS)
run_spec(
    "rc/fe/no_fe",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "", "none (pooled OLS)",
    df, "iid",
    "Full sample", "full controls, no FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/no_fe", "family": "drop",
                "dropped": ["iri_key", "week"], "baseline_fe": ["iri_key", "week"], "new_fe": []})

# Year FE only
run_spec(
    "rc/fe/year_only",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + year_str", "iri_key (store) + year",
    df, "iid",
    "Full sample", "full controls, store + year FE (coarser time)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/year_only", "family": "swap",
                "dropped": ["week"], "added": ["year"],
                "baseline_fe": ["iri_key", "week"], "new_fe": ["iri_key", "year"]})

# Month FE
run_spec(
    "rc/fe/month_only",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + month_str", "iri_key (store) + month",
    df, "iid",
    "Full sample", "full controls, store + month FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/month_only", "family": "swap",
                "dropped": ["week"], "added": ["month"],
                "baseline_fe": ["iri_key", "week"], "new_fe": ["iri_key", "month"]})

# Product FE instead of product characteristics
run_spec(
    "rc/fe/product_fe",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "logunits", "logprice", MARKETING_CONTROLS,
    "iri_key_str + week_str + p_id_str", "iri_key + week + product",
    df, "iid",
    "Full sample", "marketing controls + product FE (absorbs product chars)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/product_fe", "family": "add",
                "added": ["p_id"], "baseline_fe": ["iri_key", "week"],
                "new_fe": ["iri_key", "week", "p_id"],
                "notes": "Product FE subsumes brand, flavor, etc."})


# ============================================================
# RC: SAMPLE TRIMMING
# ============================================================

print("\n=== Running sample trimming variants ===")

n_full = len(df)

# Trim outcome at 1st/99th percentile
q01 = df['logunits'].quantile(0.01)
q99 = df['logunits'].quantile(0.99)
df_trim1 = df[(df['logunits'] >= q01) & (df['logunits'] <= q99)].copy()

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_trim1, "iid",
    f"trim logunits [1%,99%], N={len(df_trim1)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "logunits", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": n_full, "n_obs_after": len(df_trim1)})

# Trim outcome at 5th/95th percentile
q05 = df['logunits'].quantile(0.05)
q95 = df['logunits'].quantile(0.95)
df_trim5 = df[(df['logunits'] >= q05) & (df['logunits'] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_trim5, "iid",
    f"trim logunits [5%,95%], N={len(df_trim5)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "logunits", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": n_full, "n_obs_after": len(df_trim5)})

# Trim price at 1st/99th percentile
pq01 = df['logprice'].quantile(0.01)
pq99 = df['logprice'].quantile(0.99)
df_ptrim1 = df[(df['logprice'] >= pq01) & (df['logprice'] <= pq99)].copy()

run_spec(
    "rc/sample/outliers/trim_price_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_ptrim1, "iid",
    f"trim logprice [1%,99%], N={len(df_ptrim1)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_price_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "logprice", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": n_full, "n_obs_after": len(df_ptrim1)})

# Trim price at 5th/95th percentile
pq05 = df['logprice'].quantile(0.05)
pq95 = df['logprice'].quantile(0.95)
df_ptrim5 = df[(df['logprice'] >= pq05) & (df['logprice'] <= pq95)].copy()

run_spec(
    "rc/sample/outliers/trim_price_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_ptrim5, "iid",
    f"trim logprice [5%,95%], N={len(df_ptrim5)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_price_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "logprice", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": n_full, "n_obs_after": len(df_ptrim5)})

# High-volume stores (above median total units)
store_total = df.groupby('iri_key')['units'].sum()
median_vol = store_total.median()
high_vol_stores = store_total[store_total >= median_vol].index
df_highvol = df[df['iri_key'].isin(high_vol_stores)].copy()

run_spec(
    "rc/sample/subset/high_volume_stores",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_highvol, "iid",
    f"high-volume stores only, N={len(df_highvol)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/high_volume_stores", "axis": "subgroup",
                "rule": "above_median_volume", "n_obs_before": n_full, "n_obs_after": len(df_highvol)})

# Low-volume stores
low_vol_stores = store_total[store_total < median_vol].index
df_lowvol = df[df['iri_key'].isin(low_vol_stores)].copy()

run_spec(
    "rc/sample/subset/low_volume_stores",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_lowvol, "iid",
    f"low-volume stores only, N={len(df_lowvol)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/low_volume_stores", "axis": "subgroup",
                "rule": "below_median_volume", "n_obs_before": n_full, "n_obs_after": len(df_lowvol)})

# Promoted observations only
df_promo = df[df['pr'] == 1].copy()
run_spec(
    "rc/sample/subset/promoted_only",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "logunits", "logprice", [c for c in ALL_CONTROLS if c != "pr"],
    "iri_key_str + week_str", "iri_key + week",
    df_promo, "iid",
    f"promoted products only, N={len(df_promo)}", "controls minus pr (sample is all pr=1)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/promoted_only", "axis": "subgroup",
                "rule": "pr==1", "n_obs_before": n_full, "n_obs_after": len(df_promo)})

# Non-promoted observations only
df_nopromo = df[df['pr'] == 0].copy()
run_spec(
    "rc/sample/subset/nonpromoted_only",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "logunits", "logprice", [c for c in ALL_CONTROLS if c != "pr"],
    "iri_key_str + week_str", "iri_key + week",
    df_nopromo, "iid",
    f"non-promoted products only, N={len(df_nopromo)}", "controls minus pr (sample is all pr=0)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/nonpromoted_only", "axis": "subgroup",
                "rule": "pr==0", "n_obs_before": n_full, "n_obs_after": len(df_nopromo)})

# First half of weeks
df_first_half = df[df['week'] <= 26].copy()
run_spec(
    "rc/sample/subset/first_half_weeks",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_first_half, "iid",
    f"weeks 1-26 only, N={len(df_first_half)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/first_half_weeks", "axis": "subgroup",
                "rule": "week<=26", "n_obs_before": n_full, "n_obs_after": len(df_first_half)})

# Second half of weeks
df_second_half = df[df['week'] > 26].copy()
run_spec(
    "rc/sample/subset/second_half_weeks",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df_second_half, "iid",
    f"weeks 27-52 only, N={len(df_second_half)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/second_half_weeks", "axis": "subgroup",
                "rule": "week>26", "n_obs_before": n_full, "n_obs_after": len(df_second_half)})


# ============================================================
# RC: OUTCOME AND FUNCTIONAL FORM
# ============================================================

print("\n=== Running outcome/functional form variants ===")

# Units in levels (not log)
run_spec(
    "rc/outcome/units_level",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "units", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df, "iid",
    "Full sample", "full controls, outcome=units (level)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/outcome/units_level", "outcome": "units",
                "notes": "Units in levels rather than log"})

# Log-log quadratic (add logprice^2)
df['logprice_sq'] = df['logprice'] ** 2
run_spec(
    "rc/form/log_log_quadratic",
    "modules/robustness/functional_form.md#polynomial-terms", "G1",
    "logunits", "logprice", ALL_CONTROLS + ["logprice_sq"],
    "iri_key_str + week_str", "iri_key + week",
    df, "iid",
    "Full sample", "full controls + logprice^2 (quadratic demand)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/log_log_quadratic",
                "notes": "Quadratic in log price to allow nonlinear demand"})

# Price in levels (not log)
run_spec(
    "rc/form/price_level",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "logunits", "logprice", ALL_CONTROLS,
    "iri_key_str + week_str", "iri_key + week",
    df, "iid",
    "Full sample", "full controls (semi-log: log units on log price, same as baseline)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/price_level",
                "notes": "Same functional form as baseline (for completeness)"})

# Interaction: price x promotion
df['logprice_x_pr'] = df['logprice'] * df['pr']
run_spec(
    "rc/form/price_x_promo",
    "modules/robustness/functional_form.md#interaction-terms", "G1",
    "logunits", "logprice", ALL_CONTROLS + ["logprice_x_pr"],
    "iri_key_str + week_str", "iri_key + week",
    df, "iid",
    "Full sample", "full controls + logprice*pr interaction",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/price_x_promo",
                "notes": "Interaction between price and promotion to test if promo moderates elasticity"})

# Interaction: price x display
df['logprice_x_d'] = df['logprice'] * df['d']
run_spec(
    "rc/form/price_x_display",
    "modules/robustness/functional_form.md#interaction-terms", "G1",
    "logunits", "logprice", ALL_CONTROLS + ["logprice_x_d"],
    "iri_key_str + week_str", "iri_key + week",
    df, "iid",
    "Full sample", "full controls + logprice*d interaction",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/price_x_display",
                "notes": "Interaction between price and display to test if display moderates elasticity"})

# Interaction: price x feature
df['logprice_x_f'] = df['logprice'] * df['f']
run_spec(
    "rc/form/price_x_feature",
    "modules/robustness/functional_form.md#interaction-terms", "G1",
    "logunits", "logprice", ALL_CONTROLS + ["logprice_x_f"],
    "iri_key_str + week_str", "iri_key + week",
    df, "iid",
    "Full sample", "full controls + logprice*f interaction",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/price_x_feature",
                "notes": "Interaction between price and feature to test if feature moderates elasticity"})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\n=== Running inference variants ===")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0

baseline_controls_str = " + ".join(ALL_CONTROLS)
baseline_formula = f"logunits ~ logprice + {baseline_controls_str}"


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          formula_str, fe_str, data, focal_var, vcov, vcov_desc):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        if fe_str:
            full_formula = f"{formula_str} | {fe_str}"
        else:
            full_formula = formula_str

        m = pf.feols(full_formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
            ci_upper = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {focal_var: coef_val}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": vcov_desc},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"other": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": vcov_desc,
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": vcov_desc,
            "run_success": 0,
            "run_error": err_msg
        })


# HC1 robust (heteroskedasticity-robust)
run_inference_variant(
    baseline_run_id, "infer/se/hc/robust",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "iri_key_str + week_str", df, "logprice",
    "hetero", "HC1 (robust, no clustering)")

# Cluster by store
run_inference_variant(
    baseline_run_id, "infer/se/cluster/store",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "iri_key_str + week_str", df, "logprice",
    {"CRV1": "iri_key_str"}, "cluster(iri_key)")

# Cluster by product
run_inference_variant(
    baseline_run_id, "infer/se/cluster/product",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "iri_key_str + week_str", df, "logprice",
    {"CRV1": "p_id_str"}, "cluster(p_id)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n=== Writing outputs ===")
print(f"  Specification specs: {len(results)}")
print(f"  Inference variants: {len(inference_results)}")

# specification_results.csv
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)

# inference_results.csv
infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

# Summary stats
successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]

print("\n=== SPECIFICATION RESULTS SUMMARY ===")
print(f"Total rows: {len(spec_df)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
        print(f"\nBaseline coef on logprice: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs) ===")
    # Filter to only logunits outcome for comparison
    logunits_specs = successful[successful['outcome_var'] == 'logunits']
    print(f"Log-units outcome specs: {len(logunits_specs)}")
    print(f"Min coef: {logunits_specs['coefficient'].min():.6f}")
    print(f"Max coef: {logunits_specs['coefficient'].max():.6f}")
    print(f"Median coef: {logunits_specs['coefficient'].median():.6f}")
    n_sig = (logunits_specs['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(logunits_specs)}")
    n_sig10 = (logunits_specs['p_value'] < 0.10).sum()
    print(f"Significant at 10%: {n_sig10}/{len(logunits_specs)}")

if len(failed) > 0:
    print(f"\n=== FAILED SPECS ===")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:80]}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 113366-V1")
md_lines.append("")
md_lines.append("**Paper:** Bajari, Nekipelov, Ryan and Yang (2015), \"Machine Learning Methods for Demand Estimation\", AER 105(5)")
md_lines.append("")
md_lines.append("**Note:** The raw IRI scanner data is proprietary and not included in the replication package. This specification search uses synthetic data calibrated to the paper's variable structure and expected parameter values (own-price elasticity ~ -1.7).")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** OLS demand estimation (Model 1 in paper)")
md_lines.append("- **Outcome:** logunits (log units sold)")
md_lines.append("- **Focal variable:** logprice (own-price elasticity)")
md_lines.append(f"- **Controls:** {len(ALL_CONTROLS)} control groups (marketing, brand, product characteristics)")
md_lines.append("- **Fixed effects:** iri_key (store) + week")
md_lines.append("- **Standard errors:** iid (conventional OLS)")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
        bc = base_row.iloc[0]
        md_lines.append(f"| Statistic | Value |")
        md_lines.append(f"|-----------|-------|")
        md_lines.append(f"| Coefficient (logprice) | {bc['coefficient']:.6f} |")
        md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
        md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
        md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
        md_lines.append(f"| N | {bc['n_obs']:.0f} |")
        md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
        md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

categories = {
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Trimming": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Outcome/Form": successful[successful['spec_id'].str.match('rc/(outcome|form)/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['spec_id']} | {row['std_error']:.6f} | {row['p_value']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
        else:
            md_lines.append(f"| {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")
if len(successful) > 0:
    # Focus on logunits outcome specs for sign consistency
    logunits_specs = successful[successful['outcome_var'] == 'logunits']
    n_sig_total = (logunits_specs['p_value'] < 0.05).sum()
    pct_sig = n_sig_total / len(logunits_specs) * 100 if len(logunits_specs) > 0 else 0
    sign_consistent = ((logunits_specs['coefficient'] < 0).sum() == len(logunits_specs))
    median_coef = logunits_specs['coefficient'].median()
    sign_word = "negative" if median_coef < 0 else "positive"

    md_lines.append(f"- **Sign consistency:** {'All log-units specifications have negative price elasticity' if sign_consistent else 'Mixed signs across specifications'}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(logunits_specs)} ({pct_sig:.1f}%) log-units specifications significant at 5%")
    md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")

    if pct_sig >= 80 and sign_consistent:
        strength = "STRONG"
    elif pct_sig >= 50 and sign_consistent:
        strength = "MODERATE"
    elif pct_sig >= 30:
        strength = "WEAK"
    else:
        strength = "FRAGILE"

    md_lines.append(f"- **Robustness assessment:** {strength}")
    md_lines.append("")
    md_lines.append("**Note:** This assessment is based on synthetic data calibrated to the paper's structure. The actual robustness depends on the proprietary IRI data which is not available in the replication package.")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
