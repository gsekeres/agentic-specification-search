# Methodology

This document describes the methodological approach behind the Agentic Specification Search project.

## Motivation

Published research findings in economics often reflect only a small subset of the specifications that researchers explored during analysis. This "researcher degrees of freedom" problem makes it difficult to assess the robustness of findings without access to the full specification space.

Our goal is to systematically explore the specification space for published papers by:

1. Defining **standardized specification trees** for each empirical method
2. Running **all specifications** from these trees for each paper
3. Recording **complete coefficient vectors** (not just treatment effects)
4. Enabling **meta-analysis** of robustness across papers and methods

## Key Design Decisions

### 1. Coefficient Vector Format

**Decision**: Store all model coefficients in a single JSON column (`coefficient_vector_json`).

**Rationale**:
- Different methods have different output structures (OLS coefficients differ from IV first stages, which differ from event study lead/lag coefficients)
- A flexible JSON format accommodates all methods without requiring method-specific columns
- Downstream analysis can parse the JSON for specific use cases
- Full model output is preserved for reproducibility

### 2. Paper Selection

**Decision**: Random sample from ALL AEA papers with data in openICPSR, stratified by journal.

**Rationale**:
- No restrictions on method type, code quality, or observational vs. experimental
- Only requirement is that data files are present
- This avoids selection bias toward papers with "good" code
- Method type is determined during analysis, not during selection
- Journal stratification ensures representation across AEA outlets

### 3. Specification Tree Structure

**Decision**: Method-specific trees plus universal robustness checks.

**Rationale**:
- Different methods have fundamentally different specification choices
- DiD papers need modern estimators (Sun-Abraham, Callaway-Sant'Anna)
- IV papers need first-stage diagnostics and weak IV tests
- RD papers need bandwidth and polynomial variations
- But ALL papers benefit from:
  - Leave-one-out covariate checks
  - Single covariate analysis
  - Sample restrictions
  - Clustering variations
  - Functional form tests

### 4. Agent-Driven Analysis

**Decision**: Use AI agents to run specifications rather than fully automated scripts.

**Rationale**:
- Each paper has unique data structures and variable names
- Automated scripts cannot handle the heterogeneity across packages
- Agents can read documentation, understand context, and adapt
- Agents can identify appropriate specifications for each paper
- Human review still needed for quality assurance

### 5. Specification Tree Editing

**Decision**: Agents can directly edit the specification tree markdown files.

**Rationale**:
- The tree should evolve as we encounter new specification types
- Initial tree is based on common practices but not exhaustive
- As agents analyze papers, they may identify missing specifications
- Direct editing allows organic tree growth
- Future workflow may transition to flag-for-review

## Specification Categories

### Method-Specific Specifications

Each method has specifications across these categories:

| Category | Purpose |
|----------|---------|
| **Fixed Effects** | How unobserved heterogeneity is controlled |
| **Controls** | Which covariates are included |
| **Sample** | Who is included in the analysis |
| **Estimation** | What estimator is used |
| **Standard Errors** | How inference is conducted |
| **Diagnostics** | Method-specific validity tests |

### Universal Robustness Checks

Applied to all papers regardless of method:

| Check | Purpose |
|-------|---------|
| **Leave-One-Out** | Identify influential covariates |
| **Single Covariate** | Understand covariate contributions |
| **Sample Restrictions** | Test sensitivity to sample composition |
| **Clustering** | Test sensitivity to inference assumptions |
| **Functional Form** | Test sensitivity to specification |

## Output Schema

### Required Fields

Every specification result must include:

```
paper_id          # Unique identifier
spec_id           # Hierarchical specification ID
spec_tree_path    # Path to defining markdown
outcome_var       # Dependent variable
treatment_var     # Main treatment/exposure
coefficient       # Point estimate
std_error         # Standard error
p_value           # p-value
n_obs             # Sample size
```

### Coefficient Vector Schema

The `coefficient_vector_json` field contains:

```json
{
  "treatment": {
    "var": "string",
    "coef": "number",
    "se": "number",
    "pval": "number",
    "ci_lower": "number",
    "ci_upper": "number"
  },
  "controls": [
    {"var": "string", "coef": "number", "se": "number", "pval": "number"}
  ],
  "fixed_effects": ["string"],
  "diagnostics": {
    "first_stage_F": "number|null",
    "overid_pval": "number|null",
    "pretrend_pval": "number|null"
  },
  "n_obs": "number",
  "r_squared": "number|null"
}
```

## Quality Assurance

### Validation Steps

1. **Baseline Replication**: First specification must match the paper's published results
2. **Coefficient Sign Consistency**: Check for unexpected sign reversals
3. **Sample Size Tracking**: Verify N is reasonable across specifications
4. **Standard Error Reasonableness**: Flag unusually small or large SEs
5. **Specification Count**: Ensure sufficient coverage of the tree

### Red Flags

Automated checks flag specifications where:
- Treatment coefficient flips sign from baseline
- p-value jumps from <0.01 to >0.10
- Sample size drops by >50%
- R-squared is suspiciously high (>0.99) or low (<0.01)

## Limitations

1. **Software Translation**: Original Stata code must be translated to Python/R, potentially introducing errors
2. **Variable Naming**: Agent must correctly identify variables across heterogeneous naming conventions
3. **Missing Specifications**: The tree may not capture all reasonable specifications
4. **Data Availability**: Some papers may have restricted data not in the openICPSR package
5. **Computational Constraints**: Some specifications may be computationally infeasible

## References

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208-1214.

Athey, S., & Imbens, G. W. (2022). Design-based analysis in difference-in-differences settings with staggered adoption. *Journal of Econometrics*, 226(1), 62-79.

Young, A. (2022). Consistency without inference: Instrumental variables in practical application. *European Economic Review*, 147, 104112.

Brodeur, A., Cook, N., & Heyes, A. (2020). Methods matter: p-hacking and publication bias in causal analysis in economics. *American Economic Review*, 110(11), 3634-3660.
