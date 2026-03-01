# Specification Search Log: 112574-V1

## Paper
- **Title**: How Much Is a Seat on the Security Council Worth? Foreign Aid and Bribery at the United Nations
- **Authors**: Kuziemko & Werker (2006, JDE)
- **Paper ID**: 112574-V1

## Surface Summary
- **Baseline groups**: 1 (G1: Aid-for-Votes Effect)
- **Design**: Panel fixed effects (within-transformation with pair + time FE)
- **Budget**: max 100 core specs
- **Seed**: 112574
- **Controls subset sampler**: exhaustive (only 4 macro controls)

### G1 Claim Object
The differential effect of UN voting alignment on ODA commitments during recipient-country executive election years. The focal coefficient is on `p_unvotes_elecex` (= `unvotes * i_elecex`), capturing whether politically aligned countries receive more aid around elections.

### Inference Plan
- **Canonical**: 3-way clustering on donor, recipient, year (Cameron-Gelbach-Miller 2006)
- **Variants**: pair cluster, recipient cluster, 2-way donor+recipient, HC1

## Execution Summary

### Counts
| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| Baseline + additional baselines | 6 | 6 | 6 | 0 |
| design/* | 0 | 0 | 0 | 0 |
| rc/* (FE, controls, sample, form, joint) | 45 | 45 | 45 | 0 |
| **Total estimate specs** | **51** | **51** | **51** | **0** |
| infer/* (inference variants) | 4 | 4 | 4 | 0 |
| explore/* (treatment decompositions) | 17 | 17 | 17 | 0 |
| **Grand total** | **72** | **72** | **72** | **0** |

### Baseline Specifications (6)
| Spec ID | Description | Coef | SE (3-way) | p-value | N |
|---------|-------------|------|------------|---------|---|
| baseline (Col IV) | pair+year FE, no controls | 45.46 | 19.04 | 0.017 | 15315 |
| baseline__main_colV | pair+donor_year FE | 33.99 | 15.11 | 0.024 | 15315 |
| baseline__main_colVI | pair FE + macro controls | 36.73 | 15.47 | 0.018 | 13495 |
| baseline__main_colVII | decomposed UN votes, pair+year | 12.16 | 32.91 | 0.712 | 15315 |
| baseline__main_colVIII | decomposed UN votes, pair+donor_year | 12.16 | 32.77 | 0.711 | 15315 |
| baseline__main_colIX | decomposed UN votes, pair + controls | 3.77 | 25.74 | 0.884 | 13495 |

Note: Cols VII-IX report the focal coefficient on `p_unvotes_rt_elecex` (donor-average UN vote alignment * election), not `p_unvotes_elecex`. The decomposed specifications split the overall UN voting measure into a donor-average component and a residual component.

### RC Axes Executed

**FE Structure (4 specs):**
- pair + year (identical to baseline)
- pair + donor_year (stricter; absorbs all donor-year variation)
- pair only + macro controls (no time FE)
- pair + recipient_year (novel; absorbs all recipient-year variation)

**Controls (5 specs):**
- Add macro block (pop, gdp2000, pop_donor, gdp2000_donor) to pair+year baseline
- LOO: drop each of 4 macro controls individually (pair FE only)

**Sample Restrictions (4 specs):**
- Drop Big 3 recipients (EGY, IDN, IND)
- Drop Big 5 recipients (EGY, IDN, IND, ISR, CHN)
- US only (single donor; uses 2-way clustering r+year since 3-way is undefined)
- Balanced panel (already enforced by estsample; identical to baseline)

**Functional Form (2 specs):**
- log(ODA) with pair+year FE
- log(ODA) with log macro controls and pair FE

**Joint Combinations (30 specs):**
- Log ODA x {pair+year, pair+controls, donor_year, recipient_year}
- Drop Big 3 x {pair+year, donor_year, controls, recipient_year, year+controls}
- Drop Big 5 x {pair+year, donor_year, controls, recipient_year, year+controls}
- Log ODA x Drop Big 3, Log ODA x Drop Big 5
- Controls + donor_year FE
- US only x {pair+year, donor_year}
- Balanced panel x pair+year
- LOO (4 controls) x year FE (4 specs)
- LOO (4 controls) x donor_year FE (4 specs)
- Log ODA + log controls + {year, donor_year} FE

### Exploration Specifications (17 specs)
Treatment decompositions and their crosses with FE structures:
- Decomposed UN votes (donor-average + residual) x {pair+year, donor_year, controls}
- EIEC competitive election split x {pair+year, donor_year, controls}
- PCT competitive election split x {pair+year, donor_year, controls}
- Early vs. late election timing (4-month cutoff)
- Log ODA x {competitive EIEC, competitive PCT, decomposed UN votes}
- Early/late election x pair+year

### Inference Variants (4 specs on baseline)
| Variant | SE | p-value |
|---------|-----|---------|
| Cluster at pair level | (see inference_results.csv) | |
| Cluster at recipient level | | |
| 2-way cluster (donor+recipient) | | |
| HC1 robust | | |

## Key Results

### Baseline (Col IV)
- **Coefficient**: 45.46 (USD thousands per unit UN alignment * election)
- **3-way clustered SE**: 19.04
- **p-value**: 0.017
- **Interpretation**: A two-standard-deviation increase in UN alignment is associated with roughly a 59% increase in US aid during election years

### Robustness Summary
- **29/51 estimate specs significant at 5%** (57%)
- **32/51 significant at 10%** (63%)
- The result is robust to: alternative FE structures (donor_year, recipient_year), adding macro controls, dropping Big 3 recipients, and LOO control variations
- The result weakens when: using log ODA outcome, dropping Big 5 recipients, or restricting to US-only (wide SEs due to small sample)
- The decomposed treatment specifications (Cols VII-IX) show the effect is concentrated in the residual (bilateral) component of UN voting alignment, not the donor-average component

### Coefficient Range (level ODA)
- Min: 0.02 (log ODA + log controls + donor_year FE)
- Median: 34.0
- Max: 170.5 (US only)
- The large range reflects different outcome scales (level vs. log) and sample sizes

## Deviations from Surface

1. **US-only sample**: The 3-way clustering formula requires >1 donor. For US-only specs, 2-way clustering on (recipient, year) was used instead, matching the NED analysis in the paper.
2. **`rc/sample/subset/balanced_panel`**: Identical to baseline because `estsample()` already enforces balanced panels (no gaps). Recorded for completeness.
3. **`rc/joint/us_only_pair_year`** and **`rc/joint/us_only_donor_year`**: Both use pair+year FE since with 1 donor, pair FE = recipient FE and donor_year FE = year FE. Results are identical.
4. **`design/panel_fixed_effects/estimator/within`** from the surface was not separately run because the pyfixest absorbed-FE approach is numerically equivalent to the within-transformation used in the original R code.

## Software Stack
- **Python**: 3.x
- **pyfixest**: for panel FE regressions with absorbed fixed effects
- **pandas**: data manipulation
- **scipy**: normal distribution for p-values under 3-way clustering
- **3-way clustering**: Manual implementation of Cameron-Gelbach-Miller (2006) formula using 7 single-way CRV1 variance components from pyfixest, matching the original R code's `mwc_3way()` function
