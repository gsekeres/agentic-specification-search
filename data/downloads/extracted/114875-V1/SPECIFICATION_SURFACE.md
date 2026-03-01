# Specification Surface: 114875-V1

## Paper: Bronzini & Iachini -- "Are Incentives for R&D Effective? Evidence from a Regression Discontinuity Approach" (AEJ: Economic Policy)

This paper evaluates the effectiveness of an Italian regional R&D subsidy program using a sharp RD design at the eligibility score threshold of 75.

---

## Baseline Groups

### G1: Effect of R&D subsidy on firm investment

- **Design**: Regression discontinuity (sharp)
- **Running variable**: Project evaluation score (centered at 75: s = score - 75)
- **Cutoff**: 75 (firms scoring >= 75 receive the subsidy)
- **Outcome**: INVSALES (investment-to-sales ratio)
- **Treatment**: Receipt of R&D subsidy (treat = 1 if score >= 75)
- **Estimand**: Discontinuity in investment at the subsidy eligibility cutoff
- **Target population**: Firms applying for R&D subsidies from an Italian regional program

The paper's Table 3 presents the core results with varying polynomial orders (0-3) and bandwidth choices (full sample, 50% window, 35% window). The estimation uses separate polynomials on each side of the cutoff (streat, snotreat interactions). SEs are clustered at the score level.

---

## Baseline Specs

- **Table3-FullSample-Poly1**: Local linear with separate slopes, full sample, cluster(score)
- **Table3-FullSample-Poly0**: Simple difference in means (no polynomial), full sample
- Additional baselines: Poly2 (quadratic) and Poly3 (cubic) variants

---

## Core Universe

### Design variants
- Polynomial order: 0, 1, 2, 3 (the paper's main Table 3 variation)
- Bandwidth: multiple windows (25%, 35%, 50%, 75% of score range)
- Kernel: triangular, uniform
- Procedure: conventional, robust bias-corrected

### RC axes
- **Sample/bandwidth**: Various window sizes around the cutoff
- **Sample/donut**: Exclude observations within 1, 2, or 3 score points of cutoff
- **Sample/restrict**: Subgroup analyses from Tables 5-6 (small vs large firms, high vs low coverage ratio, young vs old firms) -- these are the paper's own revealed heterogeneity checks
- **Placebo cutoffs**: Test at non-threshold scores (65, 70, 80, 85)
- **Functional form**: Log and asinh transformations of INVSALES

### Excluded from core
- Tables 5-6 subgroup interactions (treatsmall, treatlarge, etc.) are treated as RC sample restrictions rather than separate baseline groups, since the claim object remains the same (subsidy effect on investment)

---

## Constraints

- Controls are polynomial terms only (no separate covariates)
- Control-count envelope: 0 (difference in means) to 6 (cubic on each side)
- The paper uses parametric separate-polynomial approach rather than modern local polynomial RD

---

## Inference Plan

- **Canonical**: Cluster at score level (matching `cluster(score)` in code)
- **Variant**: HC1 robust (no clustering)

---

## Budget

- Total core specs: up to 70
- No controls-subset sampling needed
- Full enumeration is feasible
