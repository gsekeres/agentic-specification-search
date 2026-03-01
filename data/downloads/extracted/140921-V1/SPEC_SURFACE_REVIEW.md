# Specification Surface Review: 140921-V1

**Paper**: "Assortative Matching at the Top of the Distribution: Evidence from the World's Most Exclusive Marriage Market" by Marc Goni

**Reviewed**: 2026-02-24

---

## Summary of Baseline Groups

### G1: Effect on probability of marrying a commoner (cOut)
- **Design**: cross_sectional_ols (probit marginal effects in paper; OLS/LPM as design alternative)
- **Claim object**: Well-defined. Binary outcome (cOut = married a commoner) regressed on continuous treatment intensity (syntheticT), with small fixed control set, clustered by birth year.
- **Verified against code**: `master-analysis.do` lines 377-438 confirm `probit cOut syntheticT $controls [distlondon], cluster(byear)` with `global controls pr4 biorder hengpee`. Sample: `base_sample==1` (N=644 in Panel A, N=484 in Panel B due to missing distlondon).

### G2: Effect on wealth sorting (fmissmatch)
- **Design**: cross_sectional_ols (OLS)
- **Claim object**: Well-defined. Continuous outcome (absolute mismatch in landholding percentile ranks) on same treatment, restricted to couples with non-missing landholding data.
- **Verified against code**: `reg fmissmatch syntheticT $controls [distlondon], cluster(byear)`. Sample: subset of base_sample with fmissmatch != . (N=324 in Panel A, N=260 in Panel B).

---

## Changes Made to SPECIFICATION_SURFACE.json

### 1. Removed `baseline__fmissmatch2` from G2 baseline_spec_ids
**Rationale**: `fmissmatch2` is the *signed* mismatch (husband - wife percentile rank), which is a different outcome concept from `fmissmatch` (absolute mismatch). The surface already includes `rc/form/outcome/fmissmatch2_signed` as an RC variant, which is the correct classification. Including it as a baseline spec would double-count it and conflate two distinct outcome measures within the same baseline group. If fmissmatch2 is desired as a standalone claim, it would need its own baseline group.

### 2. Removed `rc/form/outcome/linear_ols` from G1 rc_spec_ids
**Rationale**: This was duplicative with `design/cross_sectional_ols/estimator/ols` already listed in G1's design_spec_ids. Both describe the same specification (OLS/LPM instead of probit marginal effects). Having it in both namespaces would produce duplicate rows. The design namespace is the correct one for estimator alternatives.

### 3. Added `n_clusters: 22` and notes to both design_audit blocks
**Rationale**: The small-cluster log confirms 22 birth-year clusters. This is a critical design parameter for this paper: with only 22 clusters, cluster-robust standard errors are unreliable, which is why the paper supplements with bootstrap-t. Recording this in the design_audit makes results interpretable out of context.

---

## Key Constraints and Linkage Rules

- **Control-count envelope**: [3, 4] -- verified. The paper uses exactly 3 controls in Panel A (`pr4, biorder, hengpee`) and 4 in Panel B (adds `distlondon`). No other controls appear in the main Table 2 specifications.
- **No linked adjustment**: Correct. Simple single-equation OLS/probit with no bundled components.
- **Sample restriction between panels**: Adding `distlondon` drops observations (644 -> 484 for G1; 324 -> 260 for G2). The surface correctly treats Panel B (with distlondon) as the primary baseline spec and Panel A as an RC variant (drop_distlondon).
- **Cluster structure**: 22 birth-year clusters confirmed from small-cluster log. This is extremely few clusters; bootstrap-t is appropriately included as an inference variant.

---

## Budget and Sampling Assessment

- **G1**: 60 specs total. Given 1 baseline + 1 design variant + 12 RC axes, and that these can be combined (LOO controls x sample restrictions x FE, etc.), 60 is a reasonable upper bound for the non-combinatorial axes. Full enumeration is feasible and appropriate.
- **G2**: 50 specs total. Same logic; slightly fewer axes (11 RC specs). Adequate.
- No random subset sampling needed -- the control pool is too small for combinatorial explosion.

---

## What's Missing (minor)

1. **Appendix robustness (Tables B1-B2)**: The paper's appendix constructs alternative treatment variables using different benchmark cohorts (alternative syntheticT_alt based on marriage hazard rates from different historical periods). These are `rc/data/treatment_construction/*` variants that test sensitivity to the treatment variable construction. Not included in the surface, which is defensible (they are data-construction variants that would need the data-prep pipeline), but worth noting.

2. **G2 bootstrap-t inference variant**: G1 includes bootstrap-t as an inference variant, but G2 does not. The small-cluster log shows bootstrap-t was also computed for fmissmatch and fmissmatch2. Consider adding `infer/se/cluster/byear_bootstrap_t` to G2's inference variants for consistency.

3. **Other Table 2 outcomes**: Table 2 also reports mheir (married an heir), fdown (married down), and celibacy. These are correctly excluded from the core specification surface -- they are secondary outcomes, not headline claims. The surface appropriately focuses on cOut (the signature result) and fmissmatch (the quantitative sorting measure).

---

## Approved to Run

**Status**: APPROVED with minor revisions applied.

The specification surface is well-constructed and faithful to the paper's revealed analysis. The two baseline groups are conceptually distinct (binary out-marriage vs. continuous wealth sorting), the control-count envelope is correctly derived, the inference plan accounts for the small-cluster problem, and the design alternative (OLS vs. probit) is appropriate. The three changes above (removing duplicative specs and enriching design_audit) improve auditability without changing the substantive scope.
