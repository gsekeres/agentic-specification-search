#!/usr/bin/env python3
"""Generate verification output files for 10 papers."""

import csv
import json
import os

BASE = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"

def read_specs(paper_id):
    path = f"{BASE}/data/downloads/extracted/{paper_id}/specification_results.csv"
    with open(path) as f:
        return list(csv.DictReader(f))

def write_outputs(paper_id, baselines_json, spec_rows, report_md):
    outdir = f"{BASE}/data/verification/{paper_id}"
    os.makedirs(outdir, exist_ok=True)

    with open(f"{outdir}/verification_baselines.json", "w") as f:
        json.dump(baselines_json, f, indent=2)
        f.write("\n")

    with open(f"{outdir}/verification_spec_map.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id","spec_id","spec_tree_path","outcome_var","treatment_var",
                         "baseline_group_id","closest_baseline_spec_id","is_baseline","is_core_test",
                         "category","why","confidence"])
        for row in spec_rows:
            writer.writerow(row)

    with open(f"{outdir}/VERIFICATION_REPORT.md", "w") as f:
        f.write(report_md)

# ============================================================
# PAPER 1: 125821-V1 (Wisconsin School Referendum RDD)
# ============================================================
def do_125821():
    pid = "125821-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Passing an operating referendum (vote share >= 50%) increases school district total expenditures per member by approximately $641.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["tot_exp_mem"],
                "baseline_treatment_vars": ["treatment"],
                "notes": "RDD with 10pp bandwidth, local linear polynomial, baseline controls (membership, econ_disadv_percent, urban_centric_locale, above_median), clustered by district_code. Coefficient 641.24 (SE 344.34, p=0.063). N=4,240. Marginally significant at 10% level."
            }
        ],
        "global_notes": "Paper studies Wisconsin school referendum elections using RDD. The baseline is only marginally significant (p=0.063). The running variable is vote percentage centered at 50%. 79% of specs show positive coefficients. The spec rd/bandwidth/bw10 is an exact duplicate of baseline. Polynomial linear specs at BW=30,45,60 duplicate bandwidth specs. Several kernel specs (triangular, uniform, epanechnikov) produce identical results to baseline."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        # Determine classification
        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: RDD with 10pp BW, local linear, baseline controls, clustered by district", 1.0])
        elif sid == "rd/bandwidth/bw10":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        "Duplicate of baseline (same 10pp bandwidth)", 1.0])
        elif sid.startswith("rd/bandwidth/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Bandwidth variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/poly/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Polynomial order variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/kernel/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_method",
                        f"Kernel function variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/controls/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Control set variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/drop_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Leave-one-out control: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("rd/donut/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Donut hole specification: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/cluster/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Clustering variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo cutoff test: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/heterogeneity/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity test: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Functional form: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 125821-V1

## Paper Information
- **Topic**: Effects of school operating referendum elections on district expenditures in Wisconsin
- **Method**: Regression Discontinuity Design
- **Total Specifications**: 62

## Baseline Groups

### G1: Total Expenditures per Member
- **Claim**: Passing an operating referendum increases school district expenditures per member.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 641.24 (SE: 344.34, p = 0.063)
- **Outcome**: `tot_exp_mem`
- **Treatment**: `treatment` (referendum passage indicator)
- **N**: 4,240

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **40** | |
| core_controls | 9 | Baseline + control set variations + leave-one-out controls |
| core_sample | 17 | Bandwidth variations, donut holes, sample restrictions |
| core_inference | 2 | Clustering variations |
| core_funcform | 5 | Polynomial order + functional form (log, IHS) |
| core_method | 3 | Kernel function variations |
| **Non-core tests** | **22** | |
| noncore_alt_outcome | 8 | Alternative expenditure outcomes (instructional, compensation, dropout, salary, support services, debt, student-teacher ratio) |
| noncore_placebo | 4 | Placebo cutoffs at 40%, 45%, 55%, 60% |
| noncore_heterogeneity | 3 | Urban, large district, high poverty interactions |
| **Total** | **62** | |

## Robustness Assessment

The main finding has **MODERATE** support. The baseline result is only marginally significant (p=0.063). Key observations:
- 79% of specs show positive coefficients
- Only 29% achieve significance at 5%
- Donut hole specs strengthen results (3/4 significant)
- Placebo cutoff tests generally pass (no significant discontinuities at false cutoffs)
- Bandwidth sensitivity: smaller bandwidths yield insignificant results; medium bandwidths (7-15pp) often significant
- Alternative outcomes show mixed results

## Duplicates
- `rd/bandwidth/bw10` is identical to `baseline`
- Kernel specs (triangular, uniform, epanechnikov) produce identical results at same bandwidth
- Some polynomial specs duplicate bandwidth specs
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)

# ============================================================
# PAPER 2: 126722-V1 (Malaria Vouchers RCT)
# ============================================================
def do_126722():
    pid = "126722-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Patient vouchers for antimalarial drugs increase the probability of purchasing malaria treatment by approximately 15.3 percentage points.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["treat_sev_simple_mal"],
                "baseline_treatment_vars": ["patient_voucher"],
                "notes": "Cross-sectional OLS with date FE, clustered by clinic-day. Coefficient 0.153 (p<0.001). N=2,055. Randomized experiment."
            }
        ],
        "global_notes": "RCT of malaria treatment vouchers in Mali. Three arms: patient voucher, doctor voucher, control. Patient voucher effect is highly robust (100% positive, 94.5% significant at 5%). Doctor voucher shows minimal effect. Alternative outcomes (severe malaria, prescribed treatment) show weaker effects. Treatment-illness match quality decreases slightly, suggesting some overtreatment."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: patient voucher effect on malaria treatment purchase with date FE", 1.0])
        elif sid.startswith("robust/control/none"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "No controls (bivariate)", 0.95])
        elif sid.startswith("robust/control/date_fe"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "Date FE only", 0.95])
        elif sid.startswith("robust/control/basic"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "Basic patient controls", 0.95])
        elif sid.startswith("robust/control/full"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "Full control set", 0.95])
        elif sid.startswith("robust/control/include_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Additional control: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/drop_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Leave-one-out: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/add_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Incremental control addition: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/treatment/"):
            if tv == "doctor_voucher":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_treatment",
                            "Doctor voucher treatment instead of patient voucher", 0.95])
            elif tv == "any_voucher":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_treatment",
                            "Any voucher (pooled treatment)", 0.95])
            else:
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_treatment",
                            f"Alternative treatment specification: {tv}", 0.95])
        elif sid.startswith("robust/cluster/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Clustering variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/het/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Functional form / continuous outcome: {s.get('sample_desc','')}", 0.9])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo test: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 126722-V1

## Paper Information
- **Title**: Allocating Health Care Resources Efficiently (Lopez, Sautmann, Schaner 2020)
- **Journal**: AEJ-Applied
- **Method**: Cross-sectional OLS (Randomized Experiment)
- **Total Specifications**: 92

## Baseline Groups

### G1: Patient Voucher Effect on Malaria Treatment Purchase
- **Claim**: Patient vouchers increase the probability of purchasing malaria treatment.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.153 (p < 0.001)
- **Outcome**: `treat_sev_simple_mal`
- **Treatment**: `patient_voucher`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests (incl. baseline)** | **55** | |
| core_controls | 28 | Baseline + no controls + date FE + basic + full + leave-one-out (11) + incremental addition (11) + include doc info |
| core_sample | 24 | Age, gender, malaria risk, symptoms, language, education, ethnicity, illness duration, etc. |
| core_inference | 3 | Robust, clinic-day cluster, date cluster |
| **Non-core tests** | **37** | |
| noncore_alt_outcome | 14 | Prescribed treatment, severe malaria, voucher usage, expected match, match components |
| noncore_alt_treatment | 3 | Doctor voucher, any voucher, patient voucher only |
| noncore_placebo | 4 | Predicted malaria, age, days ill, symptoms |
| noncore_heterogeneity | 13 | Interactions with risk, age, gender, symptoms, info, education, days ill, ethnicity, French, literacy |
| **Total** | **92** | |

## Robustness Assessment

**STRONG** support. 100% of primary outcome specs show positive effects; 94.5% significant at 5%. Effect magnitude is stable (median 0.145). Patient vouchers consistently outperform doctor vouchers. Placebo tests pass. Subgroup effects are consistently positive.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 3: 128143-V1 (Yellow Vests Carbon Tax)
# ============================================================
def do_128143():
    pid = "128143-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Believing one does not lose from a carbon tax with dividend increases tax acceptance by ~36 percentage points.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["tax_acceptance"],
                "baseline_treatment_vars": ["believes_not_lose"],
                "notes": "Cross-sectional OLS with demographic, income, political, and urbanicity controls. Coefficient 0.363 (p<0.001). Survey-weighted. N~3,000 French respondents."
            },
            {
                "baseline_group_id": "G2",
                "claim_summary": "Believing the carbon tax is environmentally effective increases tax approval by ~40 percentage points.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline_ee"],
                "baseline_outcome_vars": ["tax_approval"],
                "baseline_treatment_vars": ["believes_effective"],
                "notes": "Environmental effectiveness channel. Coefficient 0.405 (p<0.001). Separate baseline group because it tests a distinct channel (environmental beliefs vs. self-interest beliefs)."
            }
        ],
        "global_notes": "Survey experiment during Yellow Vests movement in France. Two channels tested: self-interest beliefs (G1) and environmental effectiveness beliefs (G2). IV specifications using randomized information treatments as instruments. 97.4% of all specs positive, 93.5% significant at 5%."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        # Determine group
        if sid.startswith("ee/") or sid == "baseline_ee":
            grp = "G2"
            bsid = "baseline_ee"
        elif sid.startswith("iv/ee/"):
            grp = "G2"
            bsid = "baseline_ee"
        elif sid.startswith("iv/si/"):
            grp = "G1"
            bsid = "baseline"
        elif sid.startswith("custom/"):
            grp = "G1"
            bsid = "baseline"
        else:
            grp = "G1"
            bsid = "baseline"

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: believes_not_lose on tax_acceptance with full controls, weighted", 1.0])
        elif sid == "baseline_ee":
            rows.append([pid, sid, stp, ov, tv, "G2", "baseline_ee", 1, 1, "core_controls",
                        "Environmental effectiveness baseline: believes_effective on tax_approval", 1.0])
        elif sid.startswith("robust/control/none"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        "No controls (bivariate)", 0.95])
        elif sid.startswith("robust/control/basic"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        "Basic demographics only", 0.95])
        elif sid.startswith("robust/control/demo_"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        "Demographics + income controls", 0.95])
        elif sid.startswith("robust/control/drop_"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        f"Leave-one-out: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/add_"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        f"Additional control: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/demographics_") or sid.startswith("robust/control/political_"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        f"Control block: {sid.split('/')[-1]}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/treatment/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_alt_treatment",
                        f"Alternative treatment: {tv}", 0.95])
        elif sid.startswith("robust/inference/") or sid.startswith("robust/fe/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_inference",
                        f"Inference/FE variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_funcform",
                        f"Functional form: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/heterogeneity/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_placebo",
                        f"Placebo/validity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("ee/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G2", "baseline_ee", 0, 0, "noncore_alt_outcome",
                        f"EE channel alternative outcome: {ov}", 0.95])
        elif sid.startswith("ee/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G2", "baseline_ee", 0, 1, "core_sample",
                        f"EE channel sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("iv/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_method",
                        f"IV specification: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("custom/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_diagnostic",
                        f"Cross-outcome/diagnostic: {s.get('sample_desc','')}", 0.9])
        else:
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 128143-V1

## Paper Information
- **Title**: Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion
- **Authors**: Thomas Douenne & Adrien Fabre
- **Journal**: AEJ: Economic Policy
- **Total Specifications**: 77

## Baseline Groups

### G1: Self-Interest Beliefs Channel
- **Claim**: Believing one does not lose from a carbon tax with dividend increases tax acceptance by ~36pp.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.363 (p < 0.001)
- **Outcome**: `tax_acceptance`
- **Treatment**: `believes_not_lose`

### G2: Environmental Effectiveness Channel
- **Claim**: Believing the carbon tax is environmentally effective increases tax approval by ~40pp.
- **Baseline spec**: `baseline_ee`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.405 (p < 0.001)
- **Outcome**: `tax_approval`
- **Treatment**: `believes_effective`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **48** | |
| core_controls | 25 | Two baselines + control variations + leave-one-out + control blocks |
| core_sample | 17 | Gender, age, urban/rural, YV support, income, ecologist + EE subsamples + winsorize |
| core_inference | 4 | Classical SE, unweighted, region cluster, region FE |
| core_funcform | 3 | Quadratic gain, YV interaction, income interaction |
| core_method | 6 | IV/2SLS specifications (EE and SI channels) |
| **Non-core tests** | **29** | |
| noncore_alt_outcome | 5 | Tax approval, targeted acceptance/approval, feedback acceptance, EE acceptance |
| noncore_alt_treatment | 3 | Believes wins, simulated winner, continuous gain |
| noncore_placebo | 3 | First-stage check, reduced form info, ecologist placebo |
| noncore_heterogeneity | 6 | Interactions with female, urban, education, ecologist, age |
| noncore_diagnostic | 2 | Cross-outcome checks |
| **Total** | **77** | |

## Robustness Assessment

**STRONG** support. 97.4% positive, 93.5% significant at 5%. IV estimates (0.40-0.60) are larger than OLS (0.36), consistent with attenuation bias. Results stable across all subgroups.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 4: 128521-V1 (Lancashire Cotton Famine)
# ============================================================
def do_128521():
    pid = "128521-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Aggregate mortality rates appeared to DECREASE in cotton districts relative to non-cotton districts during the famine, consistent with migration bias masking true mortality effects.",
                "expected_sign": "-",
                "baseline_spec_ids": ["baseline", "baseline_controls", "baseline_nearby", "baseline_continuous"],
                "baseline_outcome_vars": ["agg_mr_tot"],
                "baseline_treatment_vars": ["cotton_dist_post", "cotton_eshr_post"],
                "notes": "DiD with district + period FE. Multiple baselines: unweighted (-5.51, p=0.19), with controls (-4.92, p=0.002), nearby districts (-4.76, p=0.005), continuous treatment (-6.62, p=0.003). The paper's main contribution is that LINKED data shows opposite results (positive mortality effect). These aggregate results demonstrate the migration bias."
            }
        ],
        "global_notes": "This specification search uses aggregate registrar data. The paper's key insight is that migration bias causes aggregate mortality statistics to understate true health effects. The negative DiD coefficient in aggregate data supports this -- healthy individuals migrated out of cotton districts. Two-period design (1851-1855 vs 1861-1865). 84.5% of specs show negative coefficients."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if sid in ["baseline", "baseline_controls", "baseline_nearby", "baseline_continuous"]:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        f"Baseline specification: {s.get('sample_desc','')}", 1.0])
        elif sid.startswith("did/fe/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_fe",
                        f"Fixed effects variation: {s.get('fixed_effects','')}", 0.95])
        elif sid.startswith("did/controls/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Control set: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/drop_") or sid.startswith("robust/control/add_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Control variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/cluster/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Clustering: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/treatment/"):
            if tv == "cotton_eshr_post":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_treatment",
                            "Continuous treatment (cotton employment share)", 0.95])
            else:
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_treatment",
                            f"Alternative treatment: {tv} -- nearby district spillover test", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Functional form: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/weights/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Weighting: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/heterogeneity/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 128521-V1

## Paper Information
- **Title**: Recessions, Mortality, and Migration Bias: Evidence from the Lancashire Cotton Famine
- **Method**: Difference-in-Differences
- **Total Specifications**: 58

## Baseline Groups

### G1: Aggregate Mortality Rate
- **Claim**: Aggregate mortality rates appear to decrease in cotton districts during the famine due to migration bias.
- **Baseline specs**: `baseline`, `baseline_controls`, `baseline_nearby`, `baseline_continuous`
- **Expected sign**: Negative (demonstrating migration bias)
- **Baseline coefficient**: -5.51 (p=0.19, unweighted); -4.92 (p=0.002, with controls)
- **Outcome**: `agg_mr_tot`
- **Treatment**: `cotton_dist_post`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **31** | |
| core_controls | 13 | 4 baselines + leave-one-out + incremental controls |
| core_fe | 5 | No FE, unit only, time only, region x time, county x time |
| core_sample | 12 | Drop regions, trim, winsorize, population subgroups, weights |
| core_inference | 3 | No cluster, county cluster, region cluster |
| core_funcform | 2 | IHS outcome, density squared |
| **Non-core tests** | **27** | |
| noncore_alt_outcome | 7 | Age-specific mortality rates |
| noncore_alt_treatment | 4 | Continuous treatment, nearby district spillovers |
| noncore_heterogeneity | 5 | Density, elderly share, region, cotton share interactions |
| noncore_placebo | 2 | Nearby effect, population growth |
| **Total** | **58** | |

## Robustness Assessment

**STRONG** support for the migration bias finding. 84.5% negative, 62.1% significant at 5%. Robust to controls, clustering, and sample restrictions. Age-specific outcomes show largest effects for under-15s.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 5: 130141-V1 (News Shocks SVAR)
# ============================================================
def do_130141():
    pid = "130141-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "TFP news shocks explain approximately 22% of GDP forecast error variance at 20-quarter horizon, indicating they are an important driver of business cycles.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["GDP (FEVD share)"],
                "baseline_treatment_vars": ["TFP_shock"],
                "notes": "Structural VAR with Cholesky identification, 7-variable system (TFP, GDP, Consumption, Hours, GZ spread, SP500, Inflation), 5 lags, 1984Q1-2017Q1. FEVD = 22.3%. Note: paper uses medium-run identification, this replication uses Cholesky."
            }
        ],
        "global_notes": "Paper studies news shocks under financial frictions using SVAR and DSGE. The specification search covers only the VAR component. The key metric is FEVD (share of GDP variance explained by TFP shocks). Results are sensitive to sample period (post-2000 much stronger) and lag selection (AIC/BIC select p=1 with 7.5% FEVD vs paper's p=5 with 22.3%). Excluding 2008-2009 drops FEVD to 2.4%. Rolling windows show instability."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: 7-variable VAR, 5 lags, Cholesky, 1984-2017", 1.0])
        elif sid.startswith("svar/lags/p") and not sid.startswith("svar/lags/aic") and not sid.startswith("svar/lags/bic") and not sid.startswith("svar/lags/hq"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Lag length variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("svar/lags/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Information criterion optimal lag: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("svar/vars/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Variable set variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("svar/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample period: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("svar/order/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_method",
                        f"Cholesky ordering: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("svar/control/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Variable progression: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/loo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Leave-one-out variable: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/heterogeneity/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Subsample heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/rolling/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Rolling window: {s.get('sample_desc','')}", 0.9])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome variable in FEVD: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 130141-V1

## Paper Information
- **Title**: News Shocks under Financial Frictions
- **Authors**: Gortz, Tsoukalas, and Zanetti
- **Journal**: AEJ: Macroeconomics
- **Method**: Structural VAR with Cholesky identification
- **Total Specifications**: 58

## Baseline Groups

### G1: TFP News Shock FEVD Share
- **Claim**: TFP news shocks explain ~22% of GDP forecast error variance at 20-quarter horizon.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive (FEVD is always non-negative)
- **Baseline FEVD**: 22.3%
- **Outcome**: GDP FEVD share
- **Treatment**: TFP shock

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **40** | |
| core_controls | 14 | Baseline + variable sets + leave-one-out + variable progression |
| core_sample | 8 | Sample period variations + exclusions |
| core_funcform | 10 | Lag length variations + IC-optimal lags |
| core_method | 6 | Cholesky ordering variations |
| **Non-core tests** | **18** | |
| noncore_alt_outcome | 4 | Investment, hours, consumption, SP500 as main FEVD outcome |
| noncore_heterogeneity | 6 | First/second half + rolling windows |
| noncore_placebo | 2 | Shuffled TFP, lagged TFP |
| **Total** | **58** | |

## Robustness Assessment

**MODERATE** support. 100% positive FEVD (mechanical), 69% above 10%. But highly sensitive to: sample period (excluding 2008-2009 drops to 2.4%), lag selection (7.5% at p=1 vs 42.6% at p=8), and Cholesky ordering (6-22%). Rolling windows show substantial instability.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 6: 130784-V1 (Child Marriage Bans)
# ============================================================
def do_130784():
    pid = "130784-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Child marriage bans reduce early marriage rates (marriage before 18). A one-unit increase in treatment intensity is associated with a 5.3pp reduction in child marriage.",
                "expected_sign": "-",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["childmarriage"],
                "baseline_treatment_vars": ["bancohort_pcdist"],
                "notes": "DiD with country-age and country-region-urban FE. Coefficient -0.053 (p<0.001). Treatment is interaction of post-ban cohort indicator and regional child marriage intensity. Data from 17 countries. Clustered by country-region-urban. Generated with simulated data matching original structure."
            }
        ],
        "global_notes": "Uses DHS data from 17 low/middle-income countries merged with child marriage legislation data. Results generated with simulated data. All 53 childmarriage-outcome specs are negative, 98% significant. Stronger in rural areas. 17 leave-one-country-out specs all remain significant. Multiple alternative outcomes (education, employment, marriage age) show consistent effects."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: childmarriage on bancohort_pcdist with two-way FE", 1.0])
        elif sid.startswith("robust/outcome/") and ov.startswith("childmarriage"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative marriage age threshold: {ov}", 0.95])
        elif sid.startswith("robust/outcome/marriage_age"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        "Alternative outcome: marriage age (continuous)", 0.95])
        elif sid.startswith("robust/outcome/educ"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        "Alternative outcome: years of education", 0.95])
        elif sid.startswith("robust/outcome/employed"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        "Alternative outcome: employment", 0.95])
        elif sid.startswith("robust/treatment/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Alternative treatment intensity measure: {tv}", 0.95])
        elif sid.startswith("did/fe/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_fe",
                        f"FE variation: {s.get('fixed_effects','')}", 0.95])
        elif sid.startswith("robust/cluster/") or sid.startswith("robust/se/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Inference variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/loo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Leave-one-country-out: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/heterogeneity/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Functional form: {ov}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("did/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative marriage threshold: {ov}", 0.95])
        elif sid.startswith("did/sample_full/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Full sample alternative outcome: {ov}", 0.95])
        elif sid.startswith("did/sample_urban/"):
            if ov == "childmarriage":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                            "Urban subsample: childmarriage", 0.95])
            else:
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                            f"Urban subsample alternative outcome: {ov}", 0.95])
        elif sid.startswith("did/sample_rural/"):
            if ov == "childmarriage":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                            "Rural subsample: childmarriage", 0.95])
            else:
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                            f"Rural subsample alternative outcome: {ov}", 0.95])
        elif sid.startswith("did/treatment/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Binary treatment specification: {tv}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 130784-V1

## Paper Information
- **Title**: Child Marriage Bans and Female Schooling and Labor Market Outcomes
- **Author**: Wilson (2020)
- **Journal**: AEA Papers and Proceedings
- **Method**: Difference-in-Differences with intensity-weighted treatment
- **Total Specifications**: 77
- **Note**: Results generated with simulated data matching original variable structure

## Baseline Groups

### G1: Child Marriage Rate
- **Claim**: Child marriage bans reduce early marriage rates.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.053 (p < 0.001)
- **Outcome**: `childmarriage`
- **Treatment**: `bancohort_pcdist`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **38** | |
| core_controls | 8 | Baseline + treatment definitions (intensity, binary) |
| core_fe | 3 | Country-age only, country-region-urban only, two-way |
| core_sample | 23 | Age restrictions, urban/rural, country groups, 17 LOO countries |
| core_inference | 3 | Country-region-urban, country, robust SE |
| core_funcform | 4 | Log/IHS transformations of education and marriage age |
| **Non-core tests** | **39** | |
| noncore_alt_outcome | 27 | Marriage age thresholds (17,16,15,14), education, employment, marriage age, cross-sample outcomes |
| noncore_heterogeneity | 7 | Urban/rural, age groups, intensity |
| noncore_placebo | 1 | Pre-ban cohort trend |
| **Total** | **77** | |

## Robustness Assessment

**STRONG** support. All 53 childmarriage specs are negative; 98% significant at 5%. Coefficients stable (-0.049 to -0.092). 17 LOO tests all remain significant. Effect 4x larger in rural areas. No pre-trend evidence.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 7: 131981-V1 (Mental Health Costs of Lockdowns)
# ============================================================
def do_131981():
    pid = "131981-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Age-specific COVID curfews on individuals over 65 in Turkey increased mental distress (z-scored depression index) by approximately 0.14 SD, though this is not statistically significant at conventional levels.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["z_depression"],
                "baseline_treatment_vars": ["before1955"],
                "notes": "Sharp RDD. Running variable: birth month relative to Dec 1955 (dif). Treatment: born before Dec 1955 (subject to curfew). BW=45 months. Coefficient 0.140 (SE 0.104, p=0.18). N=1,250. Controls: female, education, ethnicity. Clustered by birth month-year."
            }
        ],
        "global_notes": "RDD with birth month as running variable. Baseline is NOT significant (p=0.18). First-stage (mobility) effects are highly significant. Donut hole specs strengthen results (significant when excluding near-cutoff obs). Strong gender heterogeneity -- effect driven by males. 8 covariate balance tests all pass. 4 placebo cutoffs show no significant effects."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: z_depression on before1955, BW=45, linear, controls", 1.0])
        elif sid.startswith("rd/bandwidth/bw_"):
            if sid == "rd/bandwidth/bw_45":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                            "Duplicate of baseline (BW=45)", 1.0])
            else:
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                            f"Bandwidth variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/poly/linear_"):
            bw = sid.split("_")[-1]
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Linear polynomial at BW={bw}", 0.95])
        elif sid.startswith("rd/poly/quadratic_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Quadratic polynomial: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/z_depression"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "Duplicate of baseline (same outcome z_depression)", 1.0])
        elif sid.startswith("robust/outcome/z_somatic") or sid.startswith("robust/outcome/z_nonsomatic") or sid.startswith("robust/outcome/sum_srq"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative mental health outcome: {ov}", 0.95])
        elif sid.startswith("robust/outcome/outside_week") or sid.startswith("robust/outcome/under_curfew") or sid.startswith("robust/outcome/never_out"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Mobility/first-stage outcome: {ov}", 0.95])
        elif sid.startswith("robust/outcome/channel_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Channel outcome: {ov}", 0.95])
        elif sid.startswith("robust/outcome/symptom_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Individual symptom: {ov}", 0.95])
        elif sid.startswith("rd/controls/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Control variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/loo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Leave-one-out control: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/cluster/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Clustering: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/donut/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Donut hole: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/male_only") or sid.startswith("robust/sample/female_only"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Gender subsample: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/het/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity interaction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo cutoff: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("rd/validity/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_diagnostic",
                        f"Covariate balance test: {ov}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 131981-V1

## Paper Information
- **Title**: Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey
- **Journal**: AEJ: Applied Economics
- **Method**: Sharp Regression Discontinuity Design
- **Total Specifications**: 87

## Baseline Groups

### G1: Depression Index (z-scored)
- **Claim**: Age-specific curfews increased mental distress in those over 65.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive
- **Baseline coefficient**: 0.140 (SE: 0.104, p = 0.18) -- NOT significant
- **Outcome**: `z_depression`
- **Treatment**: `before1955`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **37** | |
| core_controls | 7 | Baseline + no controls + female only + LOO (ethnicity, education, female) + duplicate |
| core_sample | 19 | Bandwidths (7), donut holes (4), gender, marital status, education, chronic disease, psych, winsorize/trim (5), symmetric |
| core_inference | 3 | Robust, modate, province clustering |
| core_funcform | 6 | Linear at 3 BWs + quadratic at 3 BWs |
| **Non-core tests** | **50** | |
| noncore_alt_outcome | 30 | Mental health indices (3), mobility (3), channel outcomes (9), individual symptoms (12), z_depression duplicate |
| noncore_heterogeneity | 4 | Gender, married, chronic, psych support interactions |
| noncore_placebo | 4 | Placebo cutoffs at +/-12 and +/-24 months |
| noncore_diagnostic | 8 | Covariate balance tests |
| **Total** | **87** | |

## Robustness Assessment

**WEAK** support. Baseline is NOT significant (p=0.18). Only 12.2% of main outcome specs significant at 5%. Donut holes make results significant (concerning). Strong gender heterogeneity (males only). First-stage mobility effects are highly significant, confirming treatment works.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 8: 134041-V1 (Gender Wage Gap Beliefs)
# ============================================================
def do_134041():
    pid = "134041-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Information about the gender wage gap (T1=74% relative wage) increases demand for public policies addressing gender inequality, as measured by a z-scored policy index.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["z_lmpolicy_index"],
                "baseline_treatment_vars": ["T1"],
                "notes": "Cross-sectional OLS with survey weights and full demographic/political controls. Coefficient 0.062 (SE 0.025, p=0.014). N~4,000. Multiple baseline outcomes: z_lmpolicy_index is significant; quotaanchor (p=0.12), transparencyanchor (p=0.49), childcare (p=0.94) are not."
            }
        ],
        "global_notes": "Survey experiment with ~4,000 respondents (Waves A+B). T1 provides information about the gender wage gap. Baseline z_lmpolicy_index is significant at 5% but individual policy components show mixed significance. 97.8% of specs positive. Leave-one-out analysis very stable. Heterogeneity reveals differential effects by partisanship (Democrats respond more). Placebo tests on predetermined characteristics pass."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        # The baselines include multiple outcome versions
        if sid == "baseline" and ov == "z_lmpolicy_index":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: T1 on z_lmpolicy_index with full controls, weighted", 1.0])
        elif sid == "baseline" and ov != "z_lmpolicy_index":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Baseline for alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/control/none_") and ov == "z_lmpolicy_index":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "No controls for z_lmpolicy_index", 0.95])
        elif sid.startswith("robust/control/none_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"No controls for alt outcome: {ov}", 0.95])
        elif sid.startswith("robust/loo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Leave-one-out: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/add_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Incremental control addition: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/demographics_") or sid.startswith("robust/control/political_"):
            if ov == "z_lmpolicy_index":
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                            f"Control block: {sid.split('/')[-1]}", 0.95])
            else:
                rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                            f"Control block for alt outcome: {sid.split('/')[-1]}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/inference/") or sid.startswith("robust/cluster/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Inference variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/het/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Functional form: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 134041-V1

## Paper Information
- **Title**: How Do Beliefs about the Gender Wage Gap Affect the Demand for Public Policy?
- **Journal**: AEJ-Policy
- **Method**: Cross-sectional OLS (survey experiment)
- **Total Specifications**: 93

## Baseline Groups

### G1: Policy Demand Index
- **Claim**: Information about the gender wage gap increases demand for public policies addressing gender inequality.
- **Baseline spec**: `baseline` (z_lmpolicy_index)
- **Expected sign**: Positive
- **Baseline coefficient**: 0.062 (SE: 0.025, p = 0.014)
- **Outcome**: `z_lmpolicy_index`
- **Treatment**: `T1` (information treatment)

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **54** | |
| core_controls | 33 | Baseline + no controls + LOO (20) + incremental addition (10) + control blocks (2) |
| core_sample | 19 | Wave, gender, party, region, age, trimming, employment |
| core_inference | 4 | Unweighted, HC2, HC3, wave clustering |
| core_funcform | 2 | Standardized, quadratic prior |
| **Non-core tests** | **39** | |
| noncore_alt_outcome | 20 | Individual policy components (5 baselines), manipulation indices, no-controls variants for alt outcomes |
| noncore_placebo | 5 | Predetermined characteristics (gender, democrat, age, midwest, republican) |
| noncore_heterogeneity | 6 | Gender, democrat, independent, education, employee, multiple interactions |
| **Total** | **93** | |

## Robustness Assessment

**MODERATE** support. z_lmpolicy_index is significant (p=0.014) and 97.8% of specs positive. However, individual policy components show mixed significance. LOO analysis very stable. Subgroup effects vary (Democrats vs. Republicans).

## Notes on Baseline Multiplicity

The CSV contains 6 rows with spec_id="baseline" for different outcomes. The primary baseline is z_lmpolicy_index. The other 5 (quotaanchor, AAanchor, legislationanchor, transparencyanchor, childcare) are individual policy components that form the index. Only 3 of 6 baseline outcomes are significant at 5%.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 9: 136741-V1 (Historical Lynchings)
# ============================================================
def do_136741():
    pid = "136741-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Higher historical black lynching rates are associated with lower contemporary black voter registration rates. A one-unit increase in lynching rate per 10,000 black population (1900 denominator) is associated with a 0.47 percentage point decrease in black voter registration.",
                "expected_sign": "-",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["Blackrate_regvoters"],
                "baseline_treatment_vars": ["lynchcapitamob"],
                "notes": "Cross-sectional OLS with state FE and historical controls. Coefficient -0.469 (SE 0.154, p=0.003). N=267 Southern US counties in 6 states. Controls: black illiteracy (1910), county formation year, newspaper rate (1840), farm value (1860), small farm proportion, land inequality, free black proportion."
            }
        ],
        "global_notes": "Cross-sectional study of 267 counties in 6 Southern states. Identification relies on within-state variation controlling for historical observables. Main result is highly robust across control variations, sample restrictions, and estimation methods. Placebo tests pass: white lynching has no effect on black registration; black lynching has no effect on white registration. Sensitivity to population denominator: using 1920 or 1930 population attenuates effects substantially."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: Blackrate_regvoters on lynchcapitamob with state FE and historical controls", 1.0])
        elif sid.startswith("robust/loo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Leave-one-out control: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/build/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Control progression: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/control/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        f"Control variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/cluster/") or sid.startswith("robust/se/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_inference",
                        f"Inference: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/drop_") and "state" not in sid.lower():
            state = sid.split("drop_")[-1]
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Leave-one-state-out: Drop {state}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/treatment/lynch_stevenson"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_controls",
                        "Alternative data source: EJI/Stevenson lynching data", 0.95])
        elif sid.startswith("robust/treatment/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_treatment",
                        f"Alternative treatment denominator: {tv}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_placebo",
                        f"Placebo: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/form/y_") or sid.startswith("robust/form/standardized"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Functional form (outcome): {ov}", 0.95])
        elif sid.startswith("robust/form/quantile_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_method",
                        f"Quantile regression: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/form/quadratic"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        "Quadratic specification", 0.95])
        elif sid.startswith("robust/form/x_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_funcform",
                        f"Treatment functional form: {tv}", 0.95])
        elif sid.startswith("robust/het/interaction_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity interaction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/het/by_"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "noncore_heterogeneity",
                        f"Subgroup analysis: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/estimation/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_fe",
                        f"Estimation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/weights/"):
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 1, "core_sample",
                        f"Weighting: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 136741-V1

## Paper Information
- **Title**: Historical Lynchings and Black Voter Registration
- **Author**: Williams
- **Journal**: AEJ-Applied
- **Method**: Cross-sectional OLS with state FE
- **Total Specifications**: 70

## Baseline Groups

### G1: Black Voter Registration Rate
- **Claim**: Higher historical black lynching rates are associated with lower contemporary black voter registration.
- **Baseline spec**: `baseline`
- **Expected sign**: Negative
- **Baseline coefficient**: -0.469 (SE: 0.154, p = 0.003)
- **Outcome**: `Blackrate_regvoters`
- **Treatment**: `lynchcapitamob`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **46** | |
| core_controls | 19 | Baseline + LOO (7) + control progression (8) + additional controls (3) + Stevenson data |
| core_sample | 13 | State LOO (6) + trim + winsor + reg caps + weights |
| core_inference | 6 | State cluster, county cluster, HC1/HC2/HC3, classical |
| core_funcform | 7 | Log/IHS outcome, quadratic, log/IHS/binary/tercile treatment, standardized |
| core_fe | 1 | No FE estimation |
| core_method | 3 | Quantile regressions (25th, 50th, 75th) |
| **Non-core tests** | **24** | |
| noncore_alt_outcome | 2 | Register count, white voter registration (in placebo) |
| noncore_alt_treatment | 3 | Lynch rate with 1910/1920/1930 population denominators |
| noncore_placebo | 3 | White lynching on black reg, white on white, black lynching on white reg |
| noncore_heterogeneity | 9 | Interactions (education, earnings, church, incarceration, slavery) + subgroups |
| **Total** | **70** | |

## Robustness Assessment

**STRONG** support. 98.6% negative, 78.6% significant at 5%. Stable coefficient (-0.39 to -0.50) across LOO states. Placebo tests pass perfectly. Sensitive to population denominator choice (1920/1930 attenuates substantially). Binary treatment (any lynching) shows opposite sign (artifact of high-registration counties having some lynchings).
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# PAPER 10: 138401-V1 (Measles Vaccination)
# ============================================================
def do_138401():
    pid = "138401-V1"
    specs = read_specs(pid)

    baselines = {
        "paper_id": pid,
        "package_path": f"data/downloads/extracted/{pid}",
        "verified_at": "2026-02-09",
        "verifier": "verification_agent",
        "baseline_groups": [
            {
                "baseline_group_id": "G1",
                "claim_summary": "Exposure to measles vaccination in childhood affects long-run employment. The simplified treatment (high measles state x post-vaccine cohort) shows a negative employment effect of -0.07 percentage points.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline"],
                "baseline_outcome_vars": ["employed"],
                "baseline_treatment_vars": ["treatment"],
                "notes": "Panel FE with birth state, birth year, and survey year FE. Coefficient -0.00074 (p<0.001). N=16 million (ACS 2000-2017). NOTE: The treatment is simplified from the original paper (median split vs. continuous measure), and the negative sign is unexpected given the hypothesis. The original paper's continuous treatment may show different results."
            },
            {
                "baseline_group_id": "G2",
                "claim_summary": "Measles vaccination exposure affects long-run log wages.",
                "expected_sign": "+",
                "baseline_spec_ids": ["baseline_lnwage"],
                "baseline_outcome_vars": ["lnwage"],
                "baseline_treatment_vars": ["treatment"],
                "notes": "Log wage outcome, positive wage subsample. Coefficient -0.0015 (p<0.001). Also shows negative sign."
            }
        ],
        "global_notes": "Large-sample study using ACS 2000-2017 (16M observations). Treatment is simplified from original paper's approach (median split of states by pre-vaccine measles rates vs. continuous intensity). Negative coefficients are unexpected given the hypothesis that vaccination improves outcomes. This may reflect measurement issues with the simplified treatment. 84% of specs significant due to massive sample size. Placebo test with random treatment appropriately shows no effect."
    }

    rows = []
    for s in specs:
        sid = s['spec_id']
        ov = s['outcome_var']
        tv = s['treatment_var']
        stp = s.get('spec_tree_path', sid)

        if ov == "lnwage" or ov == "lnwage_wins":
            grp = "G2"
            bsid = "baseline_lnwage"
        else:
            grp = "G1"
            bsid = "baseline"

        if sid == "baseline":
            rows.append([pid, sid, stp, ov, tv, "G1", "baseline", 1, 1, "core_controls",
                        "Primary baseline: employed on treatment with bpl+birthyr+year FE", 1.0])
        elif sid == "baseline_lnwage":
            rows.append([pid, sid, stp, ov, tv, "G2", "baseline_lnwage", 1, 1, "core_controls",
                        "Wage baseline: lnwage on treatment with bpl+birthyr+year FE", 1.0])
        elif sid.startswith("robust/control/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_controls",
                        f"Control variation: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/fe/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_fe",
                        f"FE variation: {s.get('fixed_effects','')}", 0.95])
        elif sid.startswith("robust/cluster/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_inference",
                        f"Clustering: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/sample/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_sample",
                        f"Sample restriction: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/outcome/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_alt_outcome",
                        f"Alternative outcome: {ov}", 0.95])
        elif sid.startswith("robust/heterogeneity/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_heterogeneity",
                        f"Heterogeneity: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/placebo/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "noncore_placebo",
                        f"Placebo: {s.get('sample_desc','')}", 0.95])
        elif sid.startswith("robust/funcform/"):
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 1, "core_funcform",
                        f"Functional form: {s.get('sample_desc','')}", 0.95])
        else:
            rows.append([pid, sid, stp, ov, tv, grp, bsid, 0, 0, "unclear",
                        f"Unclassified: {sid}", 0.5])

    report = """# Verification Report: 138401-V1

## Paper Information
- **Title**: The Long-Term Effects of Measles Vaccination on Earnings and Employment
- **Method**: Panel fixed effects with state x cohort variation
- **Data**: ACS 2000-2017 (16 million observations)
- **Total Specifications**: 38

## Baseline Groups

### G1: Employment
- **Claim**: Measles vaccination exposure affects long-run employment.
- **Baseline spec**: `baseline`
- **Expected sign**: Positive (though observed sign is negative)
- **Baseline coefficient**: -0.00074 (p < 0.001)
- **Outcome**: `employed`
- **Treatment**: `treatment` (high measles state x post-vaccine cohort)

### G2: Log Wages
- **Claim**: Measles vaccination exposure affects long-run wages.
- **Baseline spec**: `baseline_lnwage`
- **Expected sign**: Positive (though observed sign is negative)
- **Baseline coefficient**: -0.0015 (p < 0.001)
- **Outcome**: `lnwage`
- **Treatment**: `treatment`

## Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Core tests** | **31** | |
| core_controls | 5 | 2 baselines + no controls + black only + female only |
| core_fe | 4 | bpl only, year only, birthyr only, bpl+year |
| core_sample | 17 | Gender, race, cohort, year, region, measles intensity |
| core_inference | 3 | bpl cluster, birthyr cluster, robust SE |
| core_funcform | 1 | Winsorized wages |
| **Non-core tests** | **7** | |
| noncore_alt_outcome | 2 | lnwage (in robust/outcome), lnwage males |
| noncore_heterogeneity | 2 | Treatment x black, treatment x female |
| noncore_placebo | 1 | Random treatment |
| **Total** | **38** | |

## Robustness Assessment

**MODERATE** support (with caveats). 84.2% of specs significant, but the coefficient sign is negative, which is unexpected. The simplified treatment construction (median split) differs from the original paper's continuous measure. The massive sample size (16M) means even tiny effects are statistically significant. Placebo test with random treatment shows no effect (as expected). Some subsamples (females only) had insufficient variation.
"""

    write_outputs(pid, baselines, rows, report)
    return len(rows)


# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    results = {}
    for func_name, func in [
        ("125821-V1", do_125821),
        ("126722-V1", do_126722),
        ("128143-V1", do_128143),
        ("128521-V1", do_128521),
        ("130141-V1", do_130141),
        ("130784-V1", do_130784),
        ("131981-V1", do_131981),
        ("134041-V1", do_134041),
        ("136741-V1", do_136741),
        ("138401-V1", do_138401),
    ]:
        try:
            n = func()
            results[func_name] = f"OK ({n} specs)"
        except Exception as e:
            results[func_name] = f"FAILED: {e}"

    print("\n=== RESULTS ===")
    for pid, status in results.items():
        print(f"  {pid}: {status}")
