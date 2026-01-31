# Specification Tree Index

This directory contains structured templates for systematic specification searches across different empirical methods in economics research.

## Method Types

Each paper is classified by primary empirical method. Select the appropriate method file and run ALL specifications listed therein.

| Method | File | Description |
|--------|------|-------------|
| DiD | [methods/difference_in_differences.md](methods/difference_in_differences.md) | Two-way FE, staggered adoption |
| Event Study | [methods/event_study.md](methods/event_study.md) | Dynamic treatment effects |
| RD | [methods/regression_discontinuity.md](methods/regression_discontinuity.md) | Sharp/fuzzy discontinuity |
| IV | [methods/instrumental_variables.md](methods/instrumental_variables.md) | 2SLS, control function |
| Panel FE | [methods/panel_fixed_effects.md](methods/panel_fixed_effects.md) | Within estimator |
| Cross-sectional | [methods/cross_sectional_ols.md](methods/cross_sectional_ols.md) | Single time period |
| Discrete Choice | [methods/discrete_choice.md](methods/discrete_choice.md) | Logit/probit/multinomial |
| Dynamic Panel | [methods/dynamic_panel.md](methods/dynamic_panel.md) | Arellano-Bond, PMG |

## Robustness Checks (Apply to All Methods)

These checks apply to every paper regardless of primary method:

| Check | File | Description |
|-------|------|-------------|
| Leave-One-Out | [robustness/leave_one_out.md](robustness/leave_one_out.md) | Drop each covariate |
| Single Covariate | [robustness/single_covariate.md](robustness/single_covariate.md) | Treatment + 1 control |
| Sample Restrictions | [robustness/sample_restrictions.md](robustness/sample_restrictions.md) | Subsamples, time windows |
| Clustering Variations | [robustness/clustering_variations.md](robustness/clustering_variations.md) | Different SE levels |
| Functional Form | [robustness/functional_form.md](robustness/functional_form.md) | Logs, polynomials |

## Specification ID Format

All specifications use a hierarchical ID format:
```
{method}/{category}/{variation}
```

Examples:
- `did/fe/twoway` - DiD with two-way fixed effects
- `iv/first_stage/weak_iv_test` - IV weak instrument test
- `robust/leave_one_out/drop_age` - Robustness check dropping age covariate

## Explicit Method Tree Referencing (Required)

Every specification result must include:

1. `spec_id` (hierarchical ID)
2. `spec_tree_path` pointing to the defining markdown file and section header

Example:

```
spec_id: did/fe/twoway
spec_tree_path: methods/difference_in_differences.md#fixed-effects
```

Robustness example:

```
spec_id: robust/cluster/unit
spec_tree_path: robustness/clustering_variations.md#single-level-clustering
```

If a spec is custom and not in the tree, use:

```
spec_id: custom/{description}
spec_tree_path: custom
```

## Workflow

1. **Classify** the paper's primary method
2. **Read** the corresponding method file
3. **Run baseline** specification (exact replication)
4. **Run all method specs** listed in the method file
5. **Run all robustness specs** from the robustness files
6. **Record results** with full coefficient vectors

## Adding New Specifications

If you identify a reasonable specification NOT in the tree:
1. Run it anyway with `spec_id = "custom/{description}"`
2. Document in your output that this should be added to the tree
3. Consider editing the appropriate markdown file directly to add it

## Output Format

Every specification must record:
- `spec_id`: Hierarchical identifier
- `spec_tree_path`: Path to defining markdown file
- `coefficient_vector_json`: Full model output as JSON
- All standard regression outputs (coef, se, pval, n, R2)

See the main `unified_results.csv` schema for complete column list.
