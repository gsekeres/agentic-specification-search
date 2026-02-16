"""
Specification Search Script for Ferraz & Finan (2011)
"Electoral Accountability and Corruption: Evidence from the Audits of Local Governments"
American Economic Review, 101(4), 1274-1311.

Paper ID: 112431-V1

Surface-driven execution:
  - G1: pcorrupt ~ first (Table 4 Col 6 baseline)
  - Cross-sectional OLS with state FE, HC1 robust SE
  - Target: 50+ specifications

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import json
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "112431-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
SEED = 112431

# =============================================================================
# Load data
# =============================================================================
df = pd.read_stata(os.path.join(PACKAGE_DIR, "corruptiondata_aer.dta"))
dfs = df[df['esample2'] == 1].copy()
print(f"Loaded data: {len(df)} total rows, {len(dfs)} in estimation sample (esample2==1)")

# =============================================================================
# Variable group definitions (matching surface control blocks)
# =============================================================================
mayor_demographics = ["pref_masc", "pref_idade_tse", "pref_escola"]
party_dummies = ["party_d1", "party_d3", "party_d4", "party_d5", "party_d6",
                 "party_d7", "party_d8", "party_d9", "party_d10", "party_d11",
                 "party_d12", "party_d13", "party_d14", "party_d15", "party_d16",
                 "party_d17", "party_d18"]
municipality_chars = ["lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea"]
fiscal = ["lrec_trans"]
political = ["p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]
audit_controls = [f"sorteio{i}" for i in range(1, 11)]
additional_fiscal = ["lfunc_ativ", "lrec_fisc"]

# Baseline control set (Table 4 Col 6)
BASELINE_CONTROLS = (mayor_demographics + party_dummies + municipality_chars +
                     fiscal + political + audit_controls)

# Block definitions for LOO and enumeration
CONTROL_BLOCKS = {
    "mayor_demographics": mayor_demographics,
    "party_dummies": party_dummies,
    "municipality_chars": municipality_chars,
    "fiscal": fiscal,
    "political": political,
    "audit_controls": audit_controls,
}
BLOCK_NAMES = list(CONTROL_BLOCKS.keys())

# Prepare running variable for RDD-style specs (Table 6)
dfs['wm'] = np.nan
dfs.loc[dfs['reeleito'] == 1, 'wm'] = dfs.loc[dfs['reeleito'] == 1, 'winmargin2000']
dfs.loc[dfs['incumbent'] == 1, 'wm'] = dfs.loc[dfs['incumbent'] == 1, 'winmargin2000_inclost']
dfs['running'] = dfs['wm'].copy()
dfs.loc[dfs['incumbent'] == 1, 'running'] = -dfs.loc[dfs['incumbent'] == 1, 'wm']
dfs['running2'] = dfs['running'] ** 2
dfs['running3'] = dfs['running'] ** 3

# Create region variable for FE variant
state_to_region = {
    'AC': 'N', 'AM': 'N', 'AP': 'N', 'PA': 'N', 'RO': 'N', 'RR': 'N', 'TO': 'N',
    'AL': 'NE', 'BA': 'NE', 'CE': 'NE', 'MA': 'NE', 'PB': 'NE', 'PE': 'NE',
    'PI': 'NE', 'RN': 'NE', 'SE': 'NE',
    'ES': 'SE', 'MG': 'SE', 'RJ': 'SE', 'SP': 'SE',
    'PR': 'S', 'RS': 'S', 'SC': 'S',
    'DF': 'CO', 'GO': 'CO', 'MS': 'CO', 'MT': 'CO'
}
dfs['region'] = dfs['uf'].map(state_to_region)

# asinh transform
dfs['asinh_pcorrupt'] = np.arcsinh(dfs['pcorrupt'])
dfs['log1p_pcorrupt'] = np.log1p(dfs['pcorrupt'])

# =============================================================================
# Helper functions
# =============================================================================
spec_results = []
infer_results = []
run_counter = 0
infer_counter = 0


def fml(y, rhs, absorb=None):
    """Build pyfixest formula."""
    if not rhs:
        rhs_str = "1"
    else:
        rhs_str = " + ".join(rhs)
    if absorb:
        return f"{y} ~ {rhs_str} | {absorb}"
    else:
        return f"{y} ~ {rhs_str}"


def run_spec(outcome, treatment, controls, data, fe=None, vcov="hetero",
             spec_id=None, spec_tree_path=None, baseline_group_id="G1",
             controls_desc="", sample_desc="esample2==1", fe_desc="",
             extra_json=None):
    """Run OLS spec and append to spec_results. Returns the row dict."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}_run{run_counter:04d}"

    rhs = [treatment] + controls

    try:
        formula = fml(outcome, rhs, absorb=fe)
        m = pf.feols(formula, data=data, vcov=vcov)

        coef = float(m.coef()[treatment])
        se_val = float(m.se()[treatment])
        pval = float(m.pvalue()[treatment])
        nobs = int(m._N)
        r2 = float(m._r2)
        ci_lower = coef - 1.96 * se_val
        ci_upper = coef + 1.96 * se_val

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        if extra_json:
            coef_dict.update(extra_json)

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": round(coef, 8),
            "std_error": round(se_val, 8),
            "p_value": round(pval, 8),
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": nobs,
            "r_squared": round(r2, 6),
            "coefficient_vector_json": json.dumps(coef_dict),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc if fe_desc else (fe if fe else ""),
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 1,
            "run_error": "",
        }
        spec_results.append(row)
        print(f"  {spec_run_id}: {spec_id} | coef={coef:.6f} se={se_val:.6f} p={pval:.4f} N={nobs}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc if fe_desc else (fe if fe else ""),
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": str(e),
        }
        spec_results.append(row)
        print(f"  {spec_run_id}: {spec_id} | FAILED: {e}")
        return row


def run_infer(base_spec_run_id, outcome, treatment, controls, data, fe=None,
              vcov=None, cluster_var=None, infer_spec_id=None,
              spec_tree_path=None, baseline_group_id="G1",
              controls_desc="", sample_desc="esample2==1", fe_desc=""):
    """Run inference variant and append to infer_results."""
    global infer_counter
    infer_counter += 1
    inference_run_id = f"{PAPER_ID}_infer{infer_counter:04d}"

    rhs = [treatment] + controls

    try:
        formula = fml(outcome, rhs, absorb=fe)
        if cluster_var:
            try:
                m = pf.feols(formula, data=data, vcov={"CRV1": cluster_var})
            except Exception:
                # Fall back to CRV1 without dimension issues
                m = pf.feols(formula, data=data, vcov={"CRV1": cluster_var})
        elif vcov:
            m = pf.feols(formula, data=data, vcov=vcov)
        else:
            m = pf.feols(formula, data=data, vcov="hetero")

        coef = float(m.coef()[treatment])
        se_val = float(m.se()[treatment])
        pval = float(m.pvalue()[treatment])
        nobs = int(m._N)
        r2 = float(m._r2)
        ci_lower = coef - 1.96 * se_val
        ci_upper = coef + 1.96 * se_val

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        coef_dict["inference"] = {
            "spec_id": infer_spec_id,
            "method": "cluster" if cluster_var else "hc",
            "cluster_var": cluster_var if cluster_var else "",
        }

        row = {
            "paper_id": PAPER_ID,
            "inference_run_id": inference_run_id,
            "spec_run_id": base_spec_run_id,
            "spec_id": infer_spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": round(coef, 8),
            "std_error": round(se_val, 8),
            "p_value": round(pval, 8),
            "ci_lower": round(ci_lower, 8),
            "ci_upper": round(ci_upper, 8),
            "n_obs": nobs,
            "r_squared": round(r2, 6),
            "coefficient_vector_json": json.dumps(coef_dict),
            "cluster_var": cluster_var if cluster_var else "",
            "run_success": 1,
            "run_error": "",
        }
        infer_results.append(row)
        print(f"  {inference_run_id}: {infer_spec_id} (base={base_spec_run_id}) | se={se_val:.6f} p={pval:.4f}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "inference_run_id": inference_run_id,
            "spec_run_id": base_spec_run_id,
            "spec_id": infer_spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome,
            "treatment_var": treatment,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "cluster_var": cluster_var if cluster_var else "",
            "run_success": 0,
            "run_error": str(e),
        }
        infer_results.append(row)
        print(f"  {inference_run_id}: {infer_spec_id} | FAILED: {e}")
        return row


# =============================================================================
# BASELINE GROUP G1: pcorrupt ~ first
# =============================================================================
print("\n" + "=" * 70)
print("BASELINE GROUP G1: pcorrupt ~ first")
print("=" * 70)

# ---- 1. BASELINE ----
print("\n--- Baseline (Table 4 Col 6) ---")
bl_row = run_spec("pcorrupt", "first", BASELINE_CONTROLS, dfs, fe="uf",
                  spec_id="baseline",
                  spec_tree_path="baseline",
                  fe_desc="uf",
                  controls_desc="prefchar2 + munichar2 + fiscal + political + sorteio")
baseline_run_id = bl_row["spec_run_id"]

# ---- 2. CONTROL PROGRESSION (rc/controls/progression/*) ----
print("\n--- Control Progression ---")

# Bivariate (no controls, just state FE)
run_spec("pcorrupt", "first", [], dfs, fe="uf",
         spec_id="rc/controls/progression/bivariate",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="none (bivariate + state FE)")

# Mayor demographics only
run_spec("pcorrupt", "first", mayor_demographics, dfs, fe="uf",
         spec_id="rc/controls/progression/mayor_demographics",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="mayor_demographics (3 vars)")

# Mayor demographics + party dummies
run_spec("pcorrupt", "first", mayor_demographics + party_dummies, dfs, fe="uf",
         spec_id="rc/controls/progression/mayor_demographics_party",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="mayor_demographics + party_dummies (20 vars)")

# + municipality characteristics
run_spec("pcorrupt", "first", mayor_demographics + party_dummies + municipality_chars, dfs, fe="uf",
         spec_id="rc/controls/progression/plus_municipality",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="prefchar2 + munichar2 (26 vars)")

# + fiscal
run_spec("pcorrupt", "first", mayor_demographics + party_dummies + municipality_chars + fiscal, dfs, fe="uf",
         spec_id="rc/controls/progression/plus_fiscal",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="prefchar2 + munichar2 + fiscal (27 vars)")

# + political
run_spec("pcorrupt", "first", mayor_demographics + party_dummies + municipality_chars + fiscal + political,
         dfs, fe="uf",
         spec_id="rc/controls/progression/plus_political",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="prefchar2 + munichar2 + fiscal + political (31 vars)")

# + sorteio = baseline (skip, already baseline)

# Full with lfunc_ativ
run_spec("pcorrupt", "first", BASELINE_CONTROLS + ["lfunc_ativ"], dfs, fe="uf",
         spec_id="rc/controls/progression/full_with_lfunc_ativ",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="baseline + lfunc_ativ (42 vars)")

# Full with lfunc_ativ + lrec_fisc
run_spec("pcorrupt", "first", BASELINE_CONTROLS + ["lfunc_ativ", "lrec_fisc"], dfs, fe="uf",
         spec_id="rc/controls/progression/full_with_lrec_fisc",
         spec_tree_path="modules/robustness/controls.md#control-progression",
         fe_desc="uf", controls_desc="baseline + lfunc_ativ + lrec_fisc (43 vars)")

# ---- 3. CONTROL SET VARIANTS (rc/controls/sets/*) ----
print("\n--- Control Sets ---")

# Minimal = audit controls only (sorteio dummies)
run_spec("pcorrupt", "first", audit_controls, dfs, fe="uf",
         spec_id="rc/controls/sets/minimal",
         spec_tree_path="modules/robustness/controls.md#control-sets",
         fe_desc="uf", controls_desc="audit_controls only (sorteio, 10 vars)")

# Extended = baseline + lfunc_ativ + lrec_fisc
run_spec("pcorrupt", "first", BASELINE_CONTROLS + additional_fiscal, dfs, fe="uf",
         spec_id="rc/controls/sets/extended",
         spec_tree_path="modules/robustness/controls.md#control-sets",
         fe_desc="uf", controls_desc="baseline + lfunc_ativ + lrec_fisc (43 vars)")

# ---- 4. LOO CONTROLS (rc/controls/loo/*) ----
print("\n--- LOO Controls ---")

# LOO individual variables (non-block vars from baseline)
loo_individual_vars = ["pref_masc", "pref_idade_tse", "pref_escola",
                       "lpop", "purb", "p_secundario", "mun_novo", "lpib02", "gini_ipea",
                       "lrec_trans", "p_cad_pref", "vereador_eleit", "ENLP2000", "comarca"]

for var in loo_individual_vars:
    remaining = [v for v in BASELINE_CONTROLS if v != var]
    extra = {"controls": {"spec_id": f"rc/controls/loo/drop_{var}", "family": "loo",
                          "dropped": [var], "added": [], "n_controls": len(remaining)}}
    run_spec("pcorrupt", "first", remaining, dfs, fe="uf",
             spec_id=f"rc/controls/loo/drop_{var}",
             spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
             fe_desc="uf", controls_desc=f"baseline minus {var}",
             extra_json=extra)

# LOO block: drop party dummies
remaining_no_party = [v for v in BASELINE_CONTROLS if v not in party_dummies]
run_spec("pcorrupt", "first", remaining_no_party, dfs, fe="uf",
         spec_id="rc/controls/loo/drop_party_block",
         spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
         fe_desc="uf", controls_desc="baseline minus party_dummies (17 vars dropped)")

# LOO block: drop sorteio dummies
remaining_no_sorteio = [v for v in BASELINE_CONTROLS if v not in audit_controls]
run_spec("pcorrupt", "first", remaining_no_sorteio, dfs, fe="uf",
         spec_id="rc/controls/loo/drop_sorteio_block",
         spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
         fe_desc="uf", controls_desc="baseline minus audit_controls (10 vars dropped)")

# ---- 5. RANDOM CONTROL SUBSETS (rc/controls/subset/*) ----
print("\n--- Random Control Subsets ---")

# Define the pool (non-block-level individual controls + blocks as atomic units)
# We sample which blocks to include, with some additional individual-var variation
np.random.seed(SEED)
n_random_specs = 20

for draw_idx in range(n_random_specs):
    # Each draw: randomly include/exclude each of the 6 blocks
    block_bits = np.random.randint(0, 2, size=len(BLOCK_NAMES))
    # Ensure at least one block is included
    if block_bits.sum() == 0:
        block_bits[np.random.randint(0, len(BLOCK_NAMES))] = 1
    # Ensure not all blocks are included (that's the baseline)
    if block_bits.sum() == len(BLOCK_NAMES):
        block_bits[np.random.randint(0, len(BLOCK_NAMES))] = 0

    controls = []
    included_blocks = []
    excluded_blocks = []
    for i, bn in enumerate(BLOCK_NAMES):
        if block_bits[i]:
            controls.extend(CONTROL_BLOCKS[bn])
            included_blocks.append(bn)
        else:
            excluded_blocks.append(bn)

    extra = {
        "controls": {
            "spec_id": f"rc/controls/subset/random_{draw_idx+1:03d}",
            "family": "subset",
            "method": "random_block",
            "seed": SEED,
            "draw_index": draw_idx + 1,
            "included_blocks": included_blocks,
            "excluded_blocks": excluded_blocks,
            "n_controls": len(controls),
        }
    }

    run_spec("pcorrupt", "first", controls, dfs, fe="uf",
             spec_id=f"rc/controls/subset/random_{draw_idx+1:03d}",
             spec_tree_path="modules/robustness/controls.md#control-subset-enumeration",
             fe_desc="uf",
             controls_desc=f"random subset draw {draw_idx+1}: {' + '.join(included_blocks)}",
             extra_json=extra)

# ---- 6. FE VARIANTS (rc/fe/*) ----
print("\n--- FE Variants ---")

# Drop state FE (pooled OLS)
run_spec("pcorrupt", "first", BASELINE_CONTROLS, dfs, fe=None,
         spec_id="rc/fe/drop/uf",
         spec_tree_path="modules/robustness/fixed_effects.md#drop-fe",
         fe_desc="none", controls_desc="baseline controls, no state FE")

# Region FE instead of state FE
run_spec("pcorrupt", "first", BASELINE_CONTROLS, dfs, fe="region",
         spec_id="rc/fe/add/region",
         spec_tree_path="modules/robustness/fixed_effects.md#fe-variants",
         fe_desc="region (5 Brazilian regions)",
         controls_desc="baseline controls, region FE instead of state FE")

# ---- 7. SAMPLE VARIANTS (rc/sample/*) ----
print("\n--- Sample Variants ---")

# Trim pcorrupt at 1-99 percentile
lo1 = dfs['pcorrupt'].quantile(0.01)
hi99 = dfs['pcorrupt'].quantile(0.99)
trimmed_1_99 = dfs[(dfs['pcorrupt'] >= lo1) & (dfs['pcorrupt'] <= hi99)].copy()
extra_trim1 = {"sample": {"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                           "rule": "trim", "params": {"var": "pcorrupt", "lower_q": 0.01, "upper_q": 0.99},
                           "n_obs_before": len(dfs), "n_obs_after": len(trimmed_1_99)}}
run_spec("pcorrupt", "first", BASELINE_CONTROLS, trimmed_1_99, fe="uf",
         spec_id="rc/sample/outliers/trim_y_1_99",
         spec_tree_path="modules/robustness/sample.md#trimming",
         fe_desc="uf", controls_desc="baseline controls",
         sample_desc=f"esample2==1, pcorrupt trimmed [1%,99%], N={len(trimmed_1_99)}",
         extra_json=extra_trim1)

# Trim pcorrupt at 5-95 percentile
lo5 = dfs['pcorrupt'].quantile(0.05)
hi95 = dfs['pcorrupt'].quantile(0.95)
trimmed_5_95 = dfs[(dfs['pcorrupt'] >= lo5) & (dfs['pcorrupt'] <= hi95)].copy()
extra_trim5 = {"sample": {"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                           "rule": "trim", "params": {"var": "pcorrupt", "lower_q": 0.05, "upper_q": 0.95},
                           "n_obs_before": len(dfs), "n_obs_after": len(trimmed_5_95)}}
run_spec("pcorrupt", "first", BASELINE_CONTROLS, trimmed_5_95, fe="uf",
         spec_id="rc/sample/outliers/trim_y_5_95",
         spec_tree_path="modules/robustness/sample.md#trimming",
         fe_desc="uf", controls_desc="baseline controls",
         sample_desc=f"esample2==1, pcorrupt trimmed [5%,95%], N={len(trimmed_5_95)}",
         extra_json=extra_trim5)

# ---- 8. FUNCTIONAL FORM VARIANTS (rc/form/*) ----
print("\n--- Functional Form Variants ---")

# log(1+pcorrupt)
run_spec("log1p_pcorrupt", "first", BASELINE_CONTROLS, dfs, fe="uf",
         spec_id="rc/form/outcome/log1p",
         spec_tree_path="modules/robustness/functional_form.md#outcome-transform",
         fe_desc="uf", controls_desc="baseline controls, log(1+pcorrupt) outcome",
         extra_json={"functional_form": {"outcome_transform": "log1p",
                     "interpretation": "Semi-elasticity; preserves direction of effect."}})

# asinh(pcorrupt)
run_spec("asinh_pcorrupt", "first", BASELINE_CONTROLS, dfs, fe="uf",
         spec_id="rc/form/outcome/asinh",
         spec_tree_path="modules/robustness/functional_form.md#outcome-transform",
         fe_desc="uf", controls_desc="baseline controls, asinh(pcorrupt) outcome",
         extra_json={"functional_form": {"outcome_transform": "asinh",
                     "interpretation": "Approx semi-elasticity for large y; handles zeros."}})


# =============================================================================
# INFERENCE VARIANTS (written to inference_results.csv)
# =============================================================================
print("\n" + "=" * 70)
print("INFERENCE VARIANTS")
print("=" * 70)

# Cluster SE at state level (for baseline spec)
run_infer(baseline_run_id, "pcorrupt", "first", BASELINE_CONTROLS, dfs, fe="uf",
          cluster_var="uf",
          infer_spec_id="infer/se/cluster/uf",
          spec_tree_path="modules/inference/standard_errors.md#cluster",
          fe_desc="uf", controls_desc="baseline controls, cluster SE at uf")

# HC3 SE (for baseline spec)
run_infer(baseline_run_id, "pcorrupt", "first", BASELINE_CONTROLS, dfs, fe="uf",
          vcov={"HC3": True},
          infer_spec_id="infer/se/hc/hc3",
          spec_tree_path="modules/inference/standard_errors.md#hc3",
          fe_desc="uf", controls_desc="baseline controls, HC3 SE")


# =============================================================================
# WRITE OUTPUTS
# =============================================================================
print("\n" + "=" * 70)
print("WRITING OUTPUTS")
print("=" * 70)

# 1. specification_results.csv
spec_df = pd.DataFrame(spec_results)
csv_path = os.path.join(PACKAGE_DIR, "specification_results.csv")
spec_df.to_csv(csv_path, index=False)
print(f"Wrote {len(spec_df)} rows to specification_results.csv")

# Verify no infer/* rows in specification_results
n_infer_in_spec = spec_df['spec_id'].str.startswith('infer/').sum()
assert n_infer_in_spec == 0, f"Found {n_infer_in_spec} infer/* rows in specification_results.csv!"

# 2. inference_results.csv
infer_df = pd.DataFrame(infer_results)
infer_csv_path = os.path.join(PACKAGE_DIR, "inference_results.csv")
infer_df.to_csv(infer_csv_path, index=False)
print(f"Wrote {len(infer_df)} rows to inference_results.csv")

# 3. Quality checks
assert spec_df['spec_run_id'].nunique() == len(spec_df), "spec_run_id not unique!"
print("All spec_run_ids are unique.")

# Count by spec type
for prefix in ["baseline", "rc/controls/", "rc/fe/", "rc/sample/", "rc/form/"]:
    n = spec_df['spec_id'].str.startswith(prefix).sum()
    print(f"  {prefix}: {n} specs")

n_success = (spec_df['run_success'] == 1).sum()
n_fail = (spec_df['run_success'] == 0).sum()
print(f"\nExecution summary: {n_success} succeeded, {n_fail} failed, {len(spec_df)} total")

# Baseline summary
bl = spec_df[spec_df['spec_id'] == 'baseline']
if len(bl) > 0:
    print(f"Baseline: coef={bl.iloc[0]['coefficient']}, se={bl.iloc[0]['std_error']}, "
          f"p={bl.iloc[0]['p_value']}, N={bl.iloc[0]['n_obs']}")
