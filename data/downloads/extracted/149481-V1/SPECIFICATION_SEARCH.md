# Specification Search Report: 149481-V1

**Paper:** Samek & Longfield, "Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence", AER

## Design

- **Design:** Randomized experiment (ITT)
- **G1 (Experiment 1):** Public TV stations; randomization within station x quarter strata
- **G2 (Experiment 2):** National non-profit; no stratification
- **Treatment:** Thank-you phone call (treat=1 vs control=0)
- **Focal outcomes:** renewing (binary), payment_amount3 (continuous)

## Baseline Specifications

### G1 Baselines (Experiment 1)

| Spec ID | Outcome | Coef | SE | p-value | N |
|---------|---------|------|-----|---------|---|
| baseline__t2_exp1_renewing | renewing | -0.0009 | 0.0021 | 0.6681 | 485767 |
| baseline__t2_exp1_payment_amount3 | payment_amount3 | 0.0034 | 0.4120 | 0.8534 | 485767 |
| baseline__t2_exp1_var13 | var13 | -0.0133 | 0.0099 | 0.5682 | 485767 |
| baseline__t2_exp1_gift_cond | gift_cond | 0.4126 | 1.1298 | 0.2009 | 136950 |
| baseline__t2_exp1_retention | retention | -0.0020 | 0.0038 | 0.7361 | 485767 |
| baseline__tA1_exp1_donated_ols | donated | 0.0008 | 0.0019 | 0.6679 | 485766 |
| baseline__tA1_exp1_gift_cond_ols | gift_cond | 0.6707 | 1.0814 | 0.5351 | 136947 |

### G2 Baselines (Experiment 2)

| Spec ID | Outcome | Coef | SE | p-value | N |
|---------|---------|------|-----|---------|---|
| baseline__t2_exp2_renewing | renewing | 0.0005 | 0.0039 | 0.9057 | 57631 |
| baseline__t2_exp2_payment_amount3 | payment_amount3 | -0.3005 | 0.9166 | 0.8174 | 57631 |
| baseline__t2_exp2_var13 | var13 | -0.0171 | 0.0175 | 0.9430 | 57631 |
| baseline__t2_exp2_gift_cond | gift_cond | -1.1524 | 2.5861 | 0.6120 | 17862 |
| baseline__t2_exp2_retention | retention | 0.0017 | 0.0092 | 0.8275 | 57631 |
| baseline__tA1_exp2_donated_ols | donated | 0.0005 | 0.0037 | 0.8958 | 57631 |
| baseline__tA1_exp2_gift_cond_ols | gift_cond | -0.2314 | 1.8699 | 0.9015 | 17862 |

## Specification Counts

- Total specifications: 87
- Successful: 87
- Failed: 0
- Inference variants: 16
- G1 specs: 58
- G2 specs: 29

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline (G1) | 7 | 0/7 | [-0.0133, 0.6707] |
| Baseline (G2) | 7 | 0/7 | [-1.1524, 0.0017] |
| Design Variants (G1) | 9 | 0/9 | [-0.0009, 0.2737] |
| Design Variants (G2) | 6 | 0/6 | [-0.3005, 0.0005] |
| Controls LOO (G1) | 18 | 0/18 | [0.0007, 0.6776] |
| Controls Sets (G1) | 8 | 0/8 | [0.0007, 0.6707] |
| Controls LOO (G2) | 4 | 0/4 | [-1.1620, 0.0009] |
| Controls Sets (G2) | 4 | 0/4 | [-1.1524, 0.0005] |
| Sample (G1) | 8 | 0/8 | [0.0006, 0.7043] |
| Sample (G2) | 4 | 0/4 | [-0.8746, 0.2598] |
| Functional Form (G1) | 4 | 0/4 | [0.0060, 0.0088] |
| Functional Form (G2) | 4 | 0/4 | [0.0045, 0.0087] |
| FE Variants (G1) | 4 | 0/4 | [-0.0006, 0.4059] |

## Inference Variants

| Group | Spec ID | SE | p-value | 95% CI |
|-------|---------|-----|---------|--------|
| G1 | infer/se/hc/hc1 | 0.0019 | 0.6682 | [-0.0029, 0.0045] |
| G1 | infer/se/hc/hc1 | 1.0431 | 0.5202 | [-1.3737, 2.7152] |
| G1 | infer/se/cluster/station | 0.0013 | 0.5456 | [-0.0018, 0.0034] |
| G1 | infer/se/cluster/station | 1.0528 | 0.5263 | [-1.4320, 2.7734] |
| G1 | infer/test/ttest | 0.0021 | 0.6646 | [-0.0049, 0.0031] |
| G1 | infer/test/ttest | 0.4120 | 0.9933 | [-0.8041, 0.8110] |
| G1 | infer/test/ttest | 0.0099 | 0.1768 | [-0.0327, 0.0060] |
| G1 | infer/test/ttest | 1.1298 | 0.7149 | [-1.8018, 2.6271] |
| G1 | infer/test/ttest | 0.0038 | 0.6062 | [-0.0095, 0.0055] |
| G2 | infer/se/hc/hc1 | 0.0037 | 0.8957 | [-0.0068, 0.0077] |
| G2 | infer/se/hc/hc1 | 1.8614 | 0.9011 | [-3.8800, 3.4172] |
| G2 | infer/test/ttest | 0.0039 | 0.8986 | [-0.0071, 0.0080] |
| G2 | infer/test/ttest | 0.9166 | 0.7430 | [-2.0970, 1.4960] |
| G2 | infer/test/ttest | 0.0175 | 0.3281 | [-0.0515, 0.0172] |
| G2 | infer/test/ttest | 2.5861 | 0.6559 | [-6.2211, 3.9164] |
| G2 | infer/test/ttest | 0.0092 | 0.8539 | [-0.0164, 0.0198] |

## Overall Assessment

### Experiment 1 (G1)

- **Sign consistency:** 52 positive, 6 negative
- **Significance stability:** 0/58 (0.0%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.0047)
- **Robustness assessment:** FRAGILE

### Experiment 2 (G2)

- **Sign consistency:** 16 positive, 13 negative
- **Significance stability:** 0/29 (0.0%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.0005)
- **Robustness assessment:** FRAGILE

Surface hash: `sha256:ce011d674a7943ee59e47850672f6189b7a5e1cf06059c51e540683af2787be0`
