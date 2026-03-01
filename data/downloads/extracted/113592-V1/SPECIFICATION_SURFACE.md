# Specification Surface: 113592-V1

## Paper Overview
- **Title**: Need-Based Grants and College Enrollment (France, BCS grants)
- **Design**: Sharp regression discontinuity using income-based grant eligibility cutoffs
- **Key claim**: Receipt of need-based grants (bourses sur criteres sociaux) at the income eligibility threshold increases college enrollment. Uses local linear regression with optimal bandwidth.

## Baseline Groups

### G1: Grant Eligibility on College Enrollment

**Claim object**: Sharp RD estimate of the discontinuity in college enrollment probability at the BCS income eligibility cutoff.

**Baseline specifications**:
- **Table3-0_X**: Any grant vs no grant cutoff. LLR with optimal bandwidth (~0.20). Pooled 2008-2010.
- **Table3-1_0**: L1 vs L0 grant level cutoff. Optimal bandwidth (~0.16).
- **Table3-6_1**: Higher grant level cutoffs (L6 through L2 vs L1). Optimal bandwidth (~0.06).

Running variable is `inc_distance` (income distance from cutoff, negated so higher values = below cutoff = eligible). The `rd_llr` program calls `rdob_m.ado` which implements IK-optimal bandwidth local linear regression.

## RC Axes Included

### Design variants
- **Bandwidth**: Half and double the baseline optimal bandwidth, plus fixed full bandwidth
- **Polynomial**: Local quadratic (order 2)
- **Kernel**: Uniform (rectangular) and Epanechnikov alternatives to triangular default
- **Procedure**: Robust bias-corrected inference (CCFT style)

### Sample restrictions (from Table 4 subgroup analysis)
- **Year**: 2008, 2009, 2010 separately
- **Gender**: Females only, males only
- **Level of study**: Level 1-3 students (undergrad levels)
- **Academic performance**: Baccalaureat percentile rank quartiles 1-4
- **Donut**: Exclude observations very close to the cutoff

### No controls
The baseline RD specifications use pure local linear regression with no covariates. The paper does not add controls to the RD.

## What Is Excluded and Why

- **Table 4 (subgroups)**: Year, gender, study level, and baccalaureat quartile subgroups are included as `rc/sample/restriction/*` since they all estimate the same estimand for subpopulations.
- **Table 5 (persistence outcomes)**: College enrollment in year t+1, year t+2, degree completion -- these are different outcome concepts and belong in `explore/*`.
- **Table 6 (parametric RD)**: Polynomial RD specifications with global polynomial fit -- these are alternative design implementations but are less standard than LLR. Could be added as `design/*` variants.
- **Appendix tables**: Various robustness checks and extensions.

## Budgets and Sampling

- **Max core specs**: 70
- **Full enumeration**: The specification space is naturally small and fully enumerable
- **No control subset sampling**: No covariates in the RD

## Inference Plan

- **Canonical**: Conventional LLR standard errors (from rdob)
- **Variants**: Robust bias-corrected (CCFT), clustering at discrete income levels
