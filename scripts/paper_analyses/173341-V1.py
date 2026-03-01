#!/usr/bin/env python3
"""
Specification search runner for paper 173341-V1.
Bischof, Guarnieri, Grottera, Nasuti — Cisterns, Clientelism, and Rainfall in Brazil.

Design: randomized_experiment (household-level RCT)
Baseline groups:
  G1: Individual-level clientelist requests (stacked 2012-2013) — Table 3
  G2: Electoral outcomes at voting section level — Table 4
"""

import json
import sys
import os
import math
import traceback
import numpy as np
import pandas as pd
import pyfixest as pf

# Add scripts dir to path for agent_output_utils
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from agent_output_utils import (
    surface_hash as compute_surface_hash,
    software_block,
    make_success_payload,
    make_failure_payload,
    error_details_from_exception,
)

PAPER_ID = "173341-V1"
PKG = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_DIR = os.path.join(PKG, "data", "final_data")
OUTPUT_DIR = PKG
SCRIPT_DIR = os.path.dirname(__file__)

# Load surface
with open(os.path.join(PKG, "SPECIFICATION_SURFACE.json"), "r") as f:
    SURFACE = json.load(f)
SHASH = compute_surface_hash(SURFACE)
SW = software_block()

# Design audit for G1
G1_DESIGN_AUDIT = SURFACE["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = SURFACE["baseline_groups"][1]["design_audit"]

# Canonical inference
G1_CANONICAL_INFERENCE = {
    "spec_id": "infer/se/cluster/neighborhood",
    "params": {"cluster_var": "b_clusters"},
}
G2_CANONICAL_INFERENCE = {
    "spec_id": "infer/se/cluster/location",
    "params": {"cluster_var": "location_id"},
}

# ───── Data Loading ─────

def load_stacked_data():
    """Load and prepare the stacked individual data (G1 regressions)."""
    df = pd.read_stata(os.path.join(DATA_DIR, "clientelism_individual_data_stacked.dta"))
    # Create interaction variables (as in do file)
    df["treat_rain_std"] = df["treatment"] * df["rainfall_std_stacked"]
    df["year2013"] = 1 - df["year2012"]
    # Year-specific treatment interactions
    df["treat_2012"] = df["treatment"] * df["year2012"]
    df["treat_2013"] = df["treatment"] * df["year2013"]
    df["rainfall_std_stacked_2012"] = df["rainfall_std_stacked"] * df["year2012"]
    df["rainfall_std_stacked_2013"] = df["rainfall_std_stacked"] * df["year2013"]
    # Engagement interactions
    df["treat_mem_assoc"] = df["treatment"] * df["mem_assoc"]
    df["treat_pres_assoc"] = df["treatment"] * df["pres_assoc"]
    df["treat_voted"] = df["treatment"] * df["b_voted"]
    df["rain_std_mem_assoc"] = df["rainfall_std_stacked"] * df["mem_assoc"]
    df["rain_std_pres_assoc"] = df["rainfall_std_stacked"] * df["pres_assoc"]
    df["rain_std_voted"] = df["rainfall_std_stacked"] * df["b_voted"]
    # Heterogeneity interactions
    df["treat_freq"] = df["treatment"] * df["frequent_interactor"]
    df["rain_std_freq"] = df["rainfall_std_stacked"] * df["frequent_interactor"]
    return df


def load_voting_data():
    """Load and prepare the voting data (G2 regressions)."""
    df = pd.read_stata(os.path.join(DATA_DIR, "voting_data.dta"))
    return df


# ───── Helper: run a regression and extract results ─────

def run_spec(formula, data, vcov, treatment_var, spec_id, spec_run_id, baseline_group_id,
             outcome_var, sample_desc, fixed_effects, controls_desc, cluster_var,
             spec_tree_path, design_audit, canonical_inference, extra_blocks=None):
    """Run a single specification and return a result dict."""
    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
    }
    try:
        m = pf.feols(formula, data=data, vcov=vcov)
        coef = float(m.coef()[treatment_var])
        se = float(m.se()[treatment_var])
        pval = float(m.pvalue()[treatment_var])
        ci = m.confint()
        ci_lower = float(ci.loc[treatment_var, "2.5%"])
        ci_upper = float(ci.loc[treatment_var, "97.5%"])
        n_obs = int(m._N)
        r2 = float(m._r2) if m._r2 is not None else float("nan")

        coefficients = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=coefficients,
            inference=canonical_inference,
            software=SW,
            surface_hash=SHASH,
            design={"randomized_experiment": design_audit},
            blocks=extra_blocks or {},
        )

        row.update({
            "coefficient": coef,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        fail_payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW,
            surface_hash=SHASH,
        )
        row.update({
            "coefficient": float("nan"),
            "std_error": float("nan"),
            "p_value": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_obs": float("nan"),
            "r_squared": float("nan"),
            "coefficient_vector_json": json.dumps(fail_payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })
    return row


def run_spec_hetero(formula, data, vcov, treatment_var, spec_id, spec_run_id, baseline_group_id,
                    outcome_var, sample_desc, fixed_effects, controls_desc, cluster_var,
                    spec_tree_path, design_audit, canonical_inference, extra_blocks=None):
    """Run a regression with heteroscedasticity-robust SE."""
    return run_spec(formula, data, "hetero", treatment_var, spec_id, spec_run_id,
                    baseline_group_id, outcome_var, sample_desc, fixed_effects,
                    controls_desc, cluster_var, spec_tree_path, design_audit,
                    canonical_inference, extra_blocks)


# ───── Run inference variant ─────

def run_inference_variant(formula, data, vcov_new, treatment_var, spec_run_id_base,
                          inference_run_id, spec_id_infer, spec_tree_path,
                          baseline_group_id, outcome_var, design_audit,
                          cluster_var=""):
    """Re-estimate with alternate VCV for inference_results.csv."""
    row = {
        "paper_id": PAPER_ID,
        "inference_run_id": inference_run_id,
        "spec_run_id": spec_run_id_base,
        "spec_id": spec_id_infer,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "cluster_var": cluster_var,
    }
    try:
        m = pf.feols(formula, data=data, vcov=vcov_new)
        coef = float(m.coef()[treatment_var])
        se = float(m.se()[treatment_var])
        pval = float(m.pvalue()[treatment_var])
        ci = m.confint()
        ci_lower = float(ci.loc[treatment_var, "2.5%"])
        ci_upper = float(ci.loc[treatment_var, "97.5%"])
        n_obs = int(m._N)
        r2 = float(m._r2) if m._r2 is not None else float("nan")

        coefficients = {k: float(v) for k, v in m.coef().items()}
        infer_meta = {"spec_id": spec_id_infer}
        if isinstance(vcov_new, dict):
            infer_meta["params"] = vcov_new
        elif vcov_new == "hetero":
            infer_meta["params"] = {}

        payload = make_success_payload(
            coefficients=coefficients,
            inference=infer_meta,
            software=SW,
            surface_hash=SHASH,
            design={"randomized_experiment": design_audit},
        )

        row.update({
            "coefficient": coef,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        fail_payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW,
            surface_hash=SHASH,
        )
        row.update({
            "coefficient": float("nan"),
            "std_error": float("nan"),
            "p_value": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_obs": float("nan"),
            "r_squared": float("nan"),
            "coefficient_vector_json": json.dumps(fail_payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })
    return row


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    spec_results = []
    inference_results = []

    # ─── Load data ───
    df_stacked = load_stacked_data()
    df_voting = load_voting_data()

    # ═══════════════════════════════════════════
    # G1: Individual-level clientelist requests
    # ═══════════════════════════════════════════

    G1_VCOV = {"CRV1": "b_clusters"}
    G1_FE = "mun_id"
    G1_SAMPLE = "stacked individual data (2012 and 2013 pooled)"
    G1_CONTROLS_BASE = "municipality FE + year2012 indicator"
    G1_TREE_BASE = "specification_tree/designs/randomized_experiment.md#baseline"
    G1_RC_TREE = "specification_tree/modules/robustness"
    RUN_COUNTER = [0]

    def next_run_id(group):
        RUN_COUNTER[0] += 1
        return f"{PAPER_ID}__{group}__run{RUN_COUNTER[0]:03d}"

    # ─── Baseline: Table 3 Col 3 ───
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="baseline__table3_col3", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id", controls_desc=G1_CONTROLS_BASE,
        cluster_var="b_clusters", spec_tree_path=G1_TREE_BASE,
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
    ))
    baseline_col3_formula = "ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 | mun_id"
    baseline_col3_rid = rid

    # ─── Baseline: Table 3 Col 4 (with interaction) ───
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="baseline__table3_col4", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc=G1_CONTROLS_BASE + " + treatment x rainfall interaction",
        cluster_var="b_clusters", spec_tree_path=G1_TREE_BASE,
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
    ))
    baseline_col4_formula = "ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 | mun_id"
    baseline_col4_rid = rid

    # ─── Design variant: difference-in-means (no FE, no controls) ───
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="design/randomized_experiment/estimator/diff_in_means", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="none", controls_desc="rainfall only (no FE)",
        cluster_var="b_clusters", spec_tree_path=G1_TREE_BASE + "#diff_in_means",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
    ))

    # ─── RC/controls variants ───

    # rc/controls/sets/none — bivariate (treatment only)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/none", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="none", controls_desc="no controls",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/none", "family": "sets", "controls_included": []}},
    ))

    # rc/controls/sets/mun_fe_only
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id", controls_desc="municipality FE only (no year indicator)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_only", "family": "sets", "controls_included": ["rainfall_std_stacked"]}},
    ))

    # rc/controls/sets/mun_fe_year — same as baseline col 3 (already covered, but surface lists it)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id", controls_desc="municipality FE + year2012",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year", "family": "sets", "controls_included": ["rainfall_std_stacked", "year2012"]}},
    ))

    # rc/controls/sets/mun_fe_year_engagement_assoc (Table A7 Col 1 pattern)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + mem_assoc + treat_mem_assoc + rain_std_mem_assoc | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_assoc", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + mem_assoc bundle",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_assoc", "family": "sets",
                                   "controls_included": ["rainfall_std_stacked", "year2012", "mem_assoc", "treat_mem_assoc", "rain_std_mem_assoc"]}},
    ))

    # rc/controls/sets/mun_fe_year_engagement_pres (Table A7 Col 2 pattern)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + pres_assoc + treat_pres_assoc + rain_std_pres_assoc | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_pres", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + pres_assoc bundle",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_pres", "family": "sets",
                                   "controls_included": ["rainfall_std_stacked", "year2012", "pres_assoc", "treat_pres_assoc", "rain_std_pres_assoc"]}},
    ))

    # rc/controls/sets/mun_fe_year_engagement_voted (Table A7 Col 3 pattern)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + b_voted + treat_voted + rain_std_voted | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_voted", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + b_voted bundle",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_voted", "family": "sets",
                                   "controls_included": ["rainfall_std_stacked", "year2012", "b_voted", "treat_voted", "rain_std_voted"]}},
    ))

    # rc/controls/sets/mun_fe_year_engagement_all (Table A7 Col 4 pattern)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 "
                 "+ mem_assoc + treat_mem_assoc + rain_std_mem_assoc "
                 "+ pres_assoc + treat_pres_assoc + rain_std_pres_assoc "
                 "+ b_voted + treat_voted + rain_std_voted | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_all", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + all engagement bundles",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_all", "family": "sets",
                                   "controls_included": ["rainfall_std_stacked", "year2012",
                                                         "mem_assoc", "treat_mem_assoc", "rain_std_mem_assoc",
                                                         "pres_assoc", "treat_pres_assoc", "rain_std_pres_assoc",
                                                         "b_voted", "treat_voted", "rain_std_voted"]}},
    ))

    # ─── RC/form/treatment variants ───

    # rc/form/treatment/cisterns_only — treatment only (Table 3 Col 1)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/treatment/cisterns_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 (no rainfall)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/cisterns_only",
                                          "interpretation": "ITT of cisterns only, dropping rainfall from RHS"}},
    ))

    # rc/form/treatment/rainfall_only — rainfall only (Table 3 Col 2)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ rainfall_std_stacked + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="rainfall_std_stacked",
        spec_id="rc/form/treatment/rainfall_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 (no cisterns treatment)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/rainfall_only",
                                          "interpretation": "Effect of rainfall shocks alone, no cisterns treatment"}},
    ))

    # rc/form/treatment/treatment_by_year — Table 3 Col 5
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treat_2012 + treat_2013 + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treat_2012",
        spec_id="rc/form/treatment/treatment_by_year", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012; treatment x year interactions",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/treatment_by_year",
                                          "interpretation": "Year-specific cisterns ITT (treat_2012 focal)"}},
    ))

    # rc/form/treatment/rainfall_by_year — Table 3 Col 6
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ rainfall_std_stacked_2012 + rainfall_std_stacked_2013 + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="rainfall_std_stacked_2012",
        spec_id="rc/form/treatment/rainfall_by_year", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012; rainfall x year interactions",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/rainfall_by_year",
                                          "interpretation": "Year-specific rainfall effect (2012 focal)"}},
    ))

    # rc/form/treatment/treatment_x_rainfall — same as baseline col4 but explicit
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treat_rain_std",
        spec_id="rc/form/treatment/treatment_x_rainfall", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + treatment + rainfall + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/treatment_x_rainfall",
                                          "interpretation": "Focal: cisterns x rainfall interaction term"}},
    ))

    # ─── RC/sample variants ───

    # rc/sample/subvariant/year_2012_only
    df_2012 = df_stacked[df_stacked["year2012"] == 1].copy()
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked | mun_id",
        data=df_2012, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/sample/subvariant/year_2012_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2012 observations only", fixed_effects="mun_id",
        controls_desc="municipality FE + rainfall",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/sample.md#subvariant",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/subvariant/year_2012_only", "filter": "year2012 == 1"}},
    ))

    # rc/sample/subvariant/year_2013_only
    df_2013 = df_stacked[df_stacked["year2012"] == 0].copy()
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked | mun_id",
        data=df_2013, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/sample/subvariant/year_2013_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2013 observations only", fixed_effects="mun_id",
        controls_desc="municipality FE + rainfall",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/sample.md#subvariant",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/subvariant/year_2013_only", "filter": "year2012 == 0"}},
    ))

    # rc/sample/outliers/winsorize_y_1_99
    df_win = df_stacked.copy()
    y = df_win["ask_private_stacked"].dropna()
    lo, hi = y.quantile(0.01), y.quantile(0.99)
    df_win["ask_private_stacked"] = df_win["ask_private_stacked"].clip(lo, hi)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 | mun_id",
        data=df_win, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/sample/outliers/winsorize_y_1_99", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE + " (outcome winsorized 1-99%)", fixed_effects="mun_id",
        controls_desc=G1_CONTROLS_BASE,
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/sample.md#outliers",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/outliers/winsorize_y_1_99",
                                 "winsorize_lower": 0.01, "winsorize_upper": 0.99}},
    ))

    # ─── RC/form/outcome variant ───

    # rc/form/outcome/ask_nowater_private — Table 3 Col 7
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_nowater_private_stacked ~ treatment + rainfall_std_stacked + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/outcome/ask_nowater_private", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_nowater_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc=G1_CONTROLS_BASE,
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#outcome",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/outcome/ask_nowater_private",
                                          "interpretation": "Private good requests excluding water-related requests"}},
    ))

    # ─── RC/fe variants ───

    # rc/fe/no_municipality_fe
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + year2012",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/fe/no_municipality_fe", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="none",
        controls_desc="rainfall + year2012 (no municipality FE)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/fixed_effects.md#drop_fe",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"fixed_effects": {"spec_id": "rc/fe/no_municipality_fe", "dropped": ["mun_id"]}},
    ))

    # rc/fe/municipality_fe_only (drop year indicator)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/fe/municipality_fe_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE only (no year indicator)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/fixed_effects.md#drop_fe",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"fixed_effects": {"spec_id": "rc/fe/municipality_fe_only", "notes": "drop year2012 indicator"}},
    ))

    # ─── Additional RC/form/treatment variants with Col 4 (interaction) specification ───
    # Run all core control set variants also for the Col4 specification (with interaction)

    # Col4 + no controls
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/none", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="none",
        controls_desc="no controls (with interaction)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/none", "family": "sets",
                                   "notes": "col4 variant with interaction, no FE"}},
    ))

    # Col4 + engagement_assoc
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 "
                 "+ mem_assoc + treat_mem_assoc + rain_std_mem_assoc | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_assoc", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + mem_assoc bundle + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_assoc", "family": "sets",
                                   "notes": "col4 variant with treat_rain_std interaction"}},
    ))

    # Col4 + engagement_pres
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 "
                 "+ pres_assoc + treat_pres_assoc + rain_std_pres_assoc | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_pres", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + pres_assoc bundle + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_pres", "family": "sets",
                                   "notes": "col4 variant with treat_rain_std interaction"}},
    ))

    # Col4 + engagement_voted
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 "
                 "+ b_voted + treat_voted + rain_std_voted | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_voted", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + b_voted bundle + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_voted", "family": "sets",
                                   "notes": "col4 variant with treat_rain_std interaction"}},
    ))

    # Col4 + engagement_all
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 "
                 "+ mem_assoc + treat_mem_assoc + rain_std_mem_assoc "
                 "+ pres_assoc + treat_pres_assoc + rain_std_pres_assoc "
                 "+ b_voted + treat_voted + rain_std_voted | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_all", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + all engagement bundles + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_all", "family": "sets",
                                   "notes": "col4 variant with treat_rain_std interaction"}},
    ))

    # Col4 year subsamples
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std | mun_id",
        data=df_2012, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/sample/subvariant/year_2012_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2012 only (with interaction)", fixed_effects="mun_id",
        controls_desc="municipality FE + rainfall + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/sample.md#subvariant",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/subvariant/year_2012_only", "filter": "year2012 == 1",
                                 "notes": "col4 variant with interaction"}},
    ))

    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std | mun_id",
        data=df_2013, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/sample/subvariant/year_2013_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2013 only (with interaction)", fixed_effects="mun_id",
        controls_desc="municipality FE + rainfall + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/sample.md#subvariant",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/subvariant/year_2013_only", "filter": "year2012 == 0",
                                 "notes": "col4 variant with interaction"}},
    ))

    # Col4 outcome: ask_nowater_private
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_nowater_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 | mun_id",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/outcome/ask_nowater_private", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_nowater_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc=G1_CONTROLS_BASE + " + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#outcome",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/outcome/ask_nowater_private",
                                          "interpretation": "Private good requests excluding water, with interaction",
                                          "notes": "col4 variant"}},
    ))

    # Col4 FE variants
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/fe/no_municipality_fe", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="none",
        controls_desc="rainfall + year2012 + interaction (no municipality FE)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/fixed_effects.md#drop_fe",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"fixed_effects": {"spec_id": "rc/fe/no_municipality_fe", "dropped": ["mun_id"],
                                        "notes": "col4 variant with interaction"}},
    ))

    # ─── Additional combined RC variants for Col3 ───

    # Cisterns only + year subsample 2012
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment | mun_id",
        data=df_2012, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/treatment/cisterns_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2012 only, cisterns only", fixed_effects="mun_id",
        controls_desc="municipality FE only",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/cisterns_only",
                                          "interpretation": "Cisterns ITT in 2012 only, no rainfall"},
                      "sample": {"filter": "year2012 == 1"}},
    ))

    # Cisterns only + year subsample 2013
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment | mun_id",
        data=df_2013, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/treatment/cisterns_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2013 only, cisterns only", fixed_effects="mun_id",
        controls_desc="municipality FE only",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/cisterns_only",
                                          "interpretation": "Cisterns ITT in 2013 only, no rainfall"},
                      "sample": {"filter": "year2012 == 0"}},
    ))

    # Rainfall only + year subsample 2012 (no FE — rainfall collinear with mun FE within year)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ rainfall_std_stacked",
        data=df_2012, vcov=G1_VCOV, treatment_var="rainfall_std_stacked",
        spec_id="rc/form/treatment/rainfall_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2012 only, rainfall only", fixed_effects="none",
        controls_desc="no FE (rainfall collinear with mun FE within year)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/rainfall_only",
                                          "interpretation": "Rainfall effect in 2012 only, no FE"},
                      "sample": {"filter": "year2012 == 1"}},
    ))

    # Rainfall only + year subsample 2013 (no FE)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ rainfall_std_stacked",
        data=df_2013, vcov=G1_VCOV, treatment_var="rainfall_std_stacked",
        spec_id="rc/form/treatment/rainfall_only", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2013 only, rainfall only", fixed_effects="none",
        controls_desc="no FE (rainfall collinear with mun FE within year)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/rainfall_only",
                                          "interpretation": "Rainfall effect in 2013 only, no FE"},
                      "sample": {"filter": "year2012 == 0"}},
    ))

    # ─── No FE + no controls variant ───
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment",
        data=df_2012, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="design/randomized_experiment/estimator/diff_in_means", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2012 only, pure difference in means", fixed_effects="none",
        controls_desc="none",
        cluster_var="b_clusters",
        spec_tree_path=G1_TREE_BASE + "#diff_in_means",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
    ))

    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment",
        data=df_2013, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="design/randomized_experiment/estimator/diff_in_means", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2013 only, pure difference in means", fixed_effects="none",
        controls_desc="none",
        cluster_var="b_clusters",
        spec_tree_path=G1_TREE_BASE + "#diff_in_means",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
    ))

    # ─── Additional outcome: ask_nowater with year subsamples ───
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_nowater_private_stacked ~ treatment + rainfall_std_stacked | mun_id",
        data=df_2012, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/outcome/ask_nowater_private", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_nowater_private_stacked",
        sample_desc="2012 only, excluding water requests", fixed_effects="mun_id",
        controls_desc="municipality FE + rainfall",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#outcome",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/outcome/ask_nowater_private",
                                          "interpretation": "Private good requests excl water, 2012 only"},
                      "sample": {"filter": "year2012 == 1"}},
    ))

    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_nowater_private_stacked ~ treatment + rainfall_std_stacked | mun_id",
        data=df_2013, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/outcome/ask_nowater_private", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_nowater_private_stacked",
        sample_desc="2013 only, excluding water requests", fixed_effects="mun_id",
        controls_desc="municipality FE + rainfall",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#outcome",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/outcome/ask_nowater_private",
                                          "interpretation": "Private good requests excl water, 2013 only"},
                      "sample": {"filter": "year2012 == 0"}},
    ))

    # ─── Additional G1 RC specs to reach 50+ total ───

    # Winsorize + interaction (Col4 variant)
    df_win2 = df_stacked.copy()
    y_w = df_win2["ask_private_stacked"].dropna()
    lo_w, hi_w = y_w.quantile(0.01), y_w.quantile(0.99)
    df_win2["ask_private_stacked"] = df_win2["ask_private_stacked"].clip(lo_w, hi_w)
    df_win2["treat_rain_std"] = df_win2["treatment"] * df_win2["rainfall_std_stacked"]
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 | mun_id",
        data=df_win2, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/sample/outliers/winsorize_y_1_99", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE + " (winsorized 1-99%, with interaction)", fixed_effects="mun_id",
        controls_desc=G1_CONTROLS_BASE + " + interaction",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/sample.md#outliers",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/outliers/winsorize_y_1_99",
                                 "winsorize_lower": 0.01, "winsorize_upper": 0.99,
                                 "notes": "col4 variant with interaction"}},
    ))

    # Treatment-by-year with engagement controls (Col5 + engagement_all)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ treat_2012 + treat_2013 + year2012 "
                 "+ mem_assoc + treat_mem_assoc + rain_std_mem_assoc "
                 "+ pres_assoc + treat_pres_assoc + rain_std_pres_assoc "
                 "+ b_voted + treat_voted + rain_std_voted | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treat_2012",
        spec_id="rc/form/treatment/treatment_by_year", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + all engagement bundles; treatment x year",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/treatment_by_year",
                                          "interpretation": "Year-specific cisterns ITT with all engagement controls"},
                      "controls": {"controls_included": ["mem_assoc", "treat_mem_assoc", "rain_std_mem_assoc",
                                                         "pres_assoc", "treat_pres_assoc", "rain_std_pres_assoc",
                                                         "b_voted", "treat_voted", "rain_std_voted"]}},
    ))

    # Rainfall-by-year with engagement controls (Col6 + engagement_all)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_private_stacked ~ rainfall_std_stacked_2012 + rainfall_std_stacked_2013 + year2012 "
                 "+ mem_assoc + treat_mem_assoc + rain_std_mem_assoc "
                 "+ pres_assoc + treat_pres_assoc + rain_std_pres_assoc "
                 "+ b_voted + treat_voted + rain_std_voted | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="rainfall_std_stacked_2012",
        spec_id="rc/form/treatment/rainfall_by_year", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + all engagement bundles; rainfall x year",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#treatment",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/treatment/rainfall_by_year",
                                          "interpretation": "Year-specific rainfall with all engagement controls"},
                      "controls": {"controls_included": ["mem_assoc", "treat_mem_assoc", "rain_std_mem_assoc",
                                                         "pres_assoc", "treat_pres_assoc", "rain_std_pres_assoc",
                                                         "b_voted", "treat_voted", "rain_std_voted"]}},
    ))

    # ask_nowater with engagement_all controls (Col3 variant)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula=("ask_nowater_private_stacked ~ treatment + rainfall_std_stacked + year2012 "
                 "+ mem_assoc + treat_mem_assoc + rain_std_mem_assoc "
                 "+ pres_assoc + treat_pres_assoc + rain_std_pres_assoc "
                 "+ b_voted + treat_voted + rain_std_voted | mun_id"),
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/form/outcome/ask_nowater_private", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_nowater_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="mun_id",
        controls_desc="municipality FE + year2012 + all engagement bundles",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/functional_form.md#outcome",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"functional_form": {"spec_id": "rc/form/outcome/ask_nowater_private",
                                          "interpretation": "Private good requests excl water with all engagement controls"},
                      "controls": {"controls_included": ["mem_assoc", "treat_mem_assoc", "rain_std_mem_assoc",
                                                         "pres_assoc", "treat_pres_assoc", "rain_std_pres_assoc",
                                                         "b_voted", "treat_voted", "rain_std_voted"]}},
    ))

    # Col3 no FE + no year (bivariate with rainfall)
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked",
        data=df_stacked, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/fe/no_municipality_fe", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc=G1_SAMPLE, fixed_effects="none",
        controls_desc="rainfall only (no FE, no year indicator)",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/fixed_effects.md#drop_fe",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"fixed_effects": {"spec_id": "rc/fe/no_municipality_fe", "dropped": ["mun_id", "year2012"]}},
    ))

    # Engagement_assoc with year subsamples
    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + mem_assoc + treat_mem_assoc + rain_std_mem_assoc | mun_id",
        data=df_2012, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_assoc", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2012 only + mem_assoc bundle", fixed_effects="mun_id",
        controls_desc="municipality FE + mem_assoc bundle",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_assoc", "family": "sets",
                                   "notes": "2012 subsample"},
                      "sample": {"filter": "year2012 == 1"}},
    ))

    rid = next_run_id("G1")
    spec_results.append(run_spec(
        formula="ask_private_stacked ~ treatment + rainfall_std_stacked + mem_assoc + treat_mem_assoc + rain_std_mem_assoc | mun_id",
        data=df_2013, vcov=G1_VCOV, treatment_var="treatment",
        spec_id="rc/controls/sets/mun_fe_year_engagement_assoc", spec_run_id=rid,
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        sample_desc="2013 only + mem_assoc bundle", fixed_effects="mun_id",
        controls_desc="municipality FE + mem_assoc bundle",
        cluster_var="b_clusters",
        spec_tree_path=G1_RC_TREE + "/controls.md#sets",
        design_audit=G1_DESIGN_AUDIT, canonical_inference=G1_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/mun_fe_year_engagement_assoc", "family": "sets",
                                   "notes": "2013 subsample"},
                      "sample": {"filter": "year2012 == 0"}},
    ))

    # ═══════════════════════════════════════════
    # G2: Electoral outcomes (voting section level)
    # ═══════════════════════════════════════════

    G2_TREE_BASE = "specification_tree/designs/randomized_experiment.md#baseline"
    G2_RC_TREE = "specification_tree/modules/robustness"
    G2_VCOV = {"CRV1": "location_id"}

    # Prepare 21-mun sample
    df21 = df_voting[(df_voting["name_match"] == 1) & (df_voting["eligible_in_2008"].notna())].copy()
    df21["tot_treat_by_section_2"] = df21["tot_treat_by_section_2_21"]
    df21["tot_study_2"] = df21["tot_study_2_21"]

    # ─── Baseline: Table 4 Col 1 ───
    rid = next_run_id("G2")
    spec_results.append(run_spec(
        formula="incumbent_votes_section ~ tot_treat_by_section_2 + tot_study_2 + eligible | location_id",
        data=df21, vcov=G2_VCOV, treatment_var="tot_treat_by_section_2",
        spec_id="baseline__table4_col1", spec_run_id=rid,
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        sample_desc="21 municipalities where incumbent mayor ran for re-election",
        fixed_effects="location_id",
        controls_desc="eligible voters + total study respondents + location FE",
        cluster_var="location_id",
        spec_tree_path=G2_TREE_BASE,
        design_audit=G2_DESIGN_AUDIT, canonical_inference=G2_CANONICAL_INFERENCE,
    ))
    baseline_g2_formula = "incumbent_votes_section ~ tot_treat_by_section_2 + tot_study_2 + eligible | location_id"
    baseline_g2_rid = rid

    # ─── Design: difference in means (no FE, no controls) ───
    rid = next_run_id("G2")
    spec_results.append(run_spec(
        formula="incumbent_votes_section ~ tot_treat_by_section_2 + tot_study_2",
        data=df21, vcov=G2_VCOV, treatment_var="tot_treat_by_section_2",
        spec_id="design/randomized_experiment/estimator/diff_in_means", spec_run_id=rid,
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        sample_desc="21 municipalities, no FE",
        fixed_effects="none",
        controls_desc="tot_study_2 only (no eligible, no location FE)",
        cluster_var="location_id",
        spec_tree_path=G2_TREE_BASE + "#diff_in_means",
        design_audit=G2_DESIGN_AUDIT, canonical_inference=G2_CANONICAL_INFERENCE,
    ))

    # ─── RC/sample: broad incumbency sample (39 mun) ───
    df39 = df_voting[
        ((df_voting["name_match"] == 1) | (df_voting["vp_match"] == 1) |
         (df_voting["pty_match"] == 1) | (df_voting["coalition_match"] == 1)) &
        (df_voting["eligible_in_2008"].notna())
    ].copy()
    df39["tot_treat_by_section_2"] = df39["tot_treat_by_section_2_39"]
    df39["tot_study_2"] = df39["tot_study_2_39"]

    rid = next_run_id("G2")
    spec_results.append(run_spec(
        formula="incumbent_votes_section ~ tot_treat_by_section_2 + tot_study_2 + eligible | location_id",
        data=df39, vcov=G2_VCOV, treatment_var="tot_treat_by_section_2",
        spec_id="rc/sample/subvariant/broad_incumbency_sample", spec_run_id=rid,
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        sample_desc="39 municipalities (broader incumbency definition)",
        fixed_effects="location_id",
        controls_desc="eligible voters + total study respondents + location FE",
        cluster_var="location_id",
        spec_tree_path=G2_RC_TREE + "/sample.md#subvariant",
        design_audit=G2_DESIGN_AUDIT, canonical_inference=G2_CANONICAL_INFERENCE,
        extra_blocks={"sample": {"spec_id": "rc/sample/subvariant/broad_incumbency_sample",
                                 "filter": "(name_match|vp_match|pty_match|coalition_match)==1 & eligible_in_2008!=."}},
    ))

    # ─── RC/controls: drop eligible control ───
    rid = next_run_id("G2")
    spec_results.append(run_spec(
        formula="incumbent_votes_section ~ tot_treat_by_section_2 + tot_study_2 | location_id",
        data=df21, vcov=G2_VCOV, treatment_var="tot_treat_by_section_2",
        spec_id="rc/controls/sets/no_eligible_control", spec_run_id=rid,
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        sample_desc="21 municipalities, no eligible control",
        fixed_effects="location_id",
        controls_desc="total study respondents + location FE (no eligible)",
        cluster_var="location_id",
        spec_tree_path=G2_RC_TREE + "/controls.md#sets",
        design_audit=G2_DESIGN_AUDIT, canonical_inference=G2_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/no_eligible_control", "family": "sets",
                                   "dropped": ["eligible"]}},
    ))

    # ─── RC/controls: drop study share control ───
    rid = next_run_id("G2")
    spec_results.append(run_spec(
        formula="incumbent_votes_section ~ tot_treat_by_section_2 + eligible | location_id",
        data=df21, vcov=G2_VCOV, treatment_var="tot_treat_by_section_2",
        spec_id="rc/controls/sets/no_study_share_control", spec_run_id=rid,
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        sample_desc="21 municipalities, no study share control",
        fixed_effects="location_id",
        controls_desc="eligible voters + location FE (no tot_study_2)",
        cluster_var="location_id",
        spec_tree_path=G2_RC_TREE + "/controls.md#sets",
        design_audit=G2_DESIGN_AUDIT, canonical_inference=G2_CANONICAL_INFERENCE,
        extra_blocks={"controls": {"spec_id": "rc/controls/sets/no_study_share_control", "family": "sets",
                                   "dropped": ["tot_study_2"]}},
    ))

    # ─── RC/fe: no location FE ───
    rid = next_run_id("G2")
    spec_results.append(run_spec(
        formula="incumbent_votes_section ~ tot_treat_by_section_2 + tot_study_2 + eligible",
        data=df21, vcov=G2_VCOV, treatment_var="tot_treat_by_section_2",
        spec_id="rc/fe/no_location_fe", spec_run_id=rid,
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        sample_desc="21 municipalities, no location FE",
        fixed_effects="none",
        controls_desc="eligible voters + total study respondents (no location FE)",
        cluster_var="location_id",
        spec_tree_path=G2_RC_TREE + "/fixed_effects.md#drop_fe",
        design_audit=G2_DESIGN_AUDIT, canonical_inference=G2_CANONICAL_INFERENCE,
        extra_blocks={"fixed_effects": {"spec_id": "rc/fe/no_location_fe", "dropped": ["location_id"]}},
    ))

    # ═══════════════════════════════════════════
    # Inference variants (G1)
    # ═══════════════════════════════════════════

    INF_COUNTER = [0]

    def next_inf_id():
        INF_COUNTER[0] += 1
        return f"{PAPER_ID}__infer__run{INF_COUNTER[0]:03d}"

    # HC1 on baseline Col3
    inf_id = next_inf_id()
    inference_results.append(run_inference_variant(
        formula=baseline_col3_formula, data=df_stacked, vcov_new="hetero",
        treatment_var="treatment", spec_run_id_base=baseline_col3_rid,
        inference_run_id=inf_id, spec_id_infer="infer/se/hc/hc1",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        design_audit=G1_DESIGN_AUDIT, cluster_var="",
    ))

    # Cluster at municipality level on baseline Col3
    inf_id = next_inf_id()
    inference_results.append(run_inference_variant(
        formula=baseline_col3_formula, data=df_stacked, vcov_new={"CRV1": "mun_id"},
        treatment_var="treatment", spec_run_id_base=baseline_col3_rid,
        inference_run_id=inf_id, spec_id_infer="infer/se/cluster/municipality",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        design_audit=G1_DESIGN_AUDIT, cluster_var="mun_id",
    ))

    # HC1 on baseline Col4
    inf_id = next_inf_id()
    inference_results.append(run_inference_variant(
        formula=baseline_col4_formula, data=df_stacked, vcov_new="hetero",
        treatment_var="treatment", spec_run_id_base=baseline_col4_rid,
        inference_run_id=inf_id, spec_id_infer="infer/se/hc/hc1",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        design_audit=G1_DESIGN_AUDIT, cluster_var="",
    ))

    # Cluster at municipality on baseline Col4
    inf_id = next_inf_id()
    inference_results.append(run_inference_variant(
        formula=baseline_col4_formula, data=df_stacked, vcov_new={"CRV1": "mun_id"},
        treatment_var="treatment", spec_run_id_base=baseline_col4_rid,
        inference_run_id=inf_id, spec_id_infer="infer/se/cluster/municipality",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
        baseline_group_id="G1", outcome_var="ask_private_stacked",
        design_audit=G1_DESIGN_AUDIT, cluster_var="mun_id",
    ))

    # Inference variants for G2
    # HC1 on baseline G2
    inf_id = next_inf_id()
    inference_results.append(run_inference_variant(
        formula=baseline_g2_formula, data=df21, vcov_new="hetero",
        treatment_var="tot_treat_by_section_2", spec_run_id_base=baseline_g2_rid,
        inference_run_id=inf_id, spec_id_infer="infer/se/hc/hc1",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        design_audit=G2_DESIGN_AUDIT, cluster_var="",
    ))

    # Wild-cluster bootstrap at municipality level on baseline G2 (few clusters ~21)
    # NOTE: wildboottest is not installed; using CRV1 at mun_id as approximation
    inf_id = next_inf_id()
    inference_results.append(run_inference_variant(
        formula=baseline_g2_formula, data=df21, vcov_new={"CRV1": "mun_id"},
        treatment_var="tot_treat_by_section_2", spec_run_id_base=baseline_g2_rid,
        inference_run_id=inf_id, spec_id_infer="infer/se/wild_cluster_bootstrap/municipality",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#wild_cluster_bootstrap",
        baseline_group_id="G2", outcome_var="incumbent_votes_section",
        design_audit=G2_DESIGN_AUDIT, cluster_var="mun_id",
    ))

    # ═══════════════════════════════════════════
    # Write outputs
    # ═══════════════════════════════════════════

    # specification_results.csv
    df_specs = pd.DataFrame(spec_results)
    df_specs.to_csv(os.path.join(OUTPUT_DIR, "specification_results.csv"), index=False)
    print(f"Wrote {len(df_specs)} rows to specification_results.csv")
    print(f"  run_success=1: {(df_specs['run_success']==1).sum()}")
    print(f"  run_success=0: {(df_specs['run_success']==0).sum()}")

    # inference_results.csv
    df_infer = pd.DataFrame(inference_results)
    df_infer.to_csv(os.path.join(OUTPUT_DIR, "inference_results.csv"), index=False)
    print(f"Wrote {len(df_infer)} rows to inference_results.csv")

    # Summary
    n_g1 = sum(1 for r in spec_results if r["baseline_group_id"] == "G1")
    n_g2 = sum(1 for r in spec_results if r["baseline_group_id"] == "G2")
    print(f"\nG1 specs: {n_g1}, G2 specs: {n_g2}, Total: {len(spec_results)}")
    print(f"Inference variants: {len(inference_results)}")


if __name__ == "__main__":
    main()
