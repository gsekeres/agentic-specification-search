# Specification Search Report: 112498-V1

**Paper:** Ogaki & Zhang (2001), "Decreasing Relative Risk Aversion and Tests of Risk Sharing", Econometrica 69(2)

## Baseline Specification

- **Design:** Panel OLS (standard risk-sharing test)
- **Outcome:** ch_r_totexp (change in real per-capita total consumption)
- **Treatment:** ch_r_nonlabinc (change in real per-capita non-labor income)
- **Controls:** ch_r_totexp_vill (village avg consumption change), l_r_wage, wt_nhh, intfreq
- **Fixed effects:** none (baseline)
- **Sample:** Village E (Kanzara), households with 80+ monthly observations, 1975-1984
- **Hypothesis:** Under full risk sharing, coef on ch_r_nonlabinc = 0; positive coef rejects full insurance

| Statistic | Value |
|-----------|-------|
| Coefficient | 0.009316 |
| Std. Error | 0.004300 |
| p-value | 0.030338 |
| 95% CI | [0.000885, 0.017747] |
| N | 3675 |
| R-squared | 0.2588 |

## Specification Counts

- Total specifications: 64
- Successful: 64
- Failed: 0
- Inference variants: 3

## Category Breakdown

| Category | Count | Sig. (p<0.05) | Coef Range |
|----------|-------|---------------|------------|
| Baseline (OLS) | 1 | 1/1 | [0.0093, 0.0093] |
| Baseline (IV) | 1 | 1/1 | [0.5469, 0.5469] |
| Village variants | 10 | 6/10 | [0.0042, 0.8835] |
| By-caste | 10 | 2/10 | [-0.0008, 0.5322] |
| Controls LOO | 4 | 3/4 | [0.0086, 0.0132] |
| Controls Sets | 3 | 2/3 | [0.0086, 0.0128] |
| Controls Progression | 5 | 4/5 | [0.0086, 0.0128] |
| Controls Add | 7 | 7/7 | [0.0082, 0.0093] |
| Controls Subset | 10 | 7/10 | [0.0079, 0.0128] |
| Controls Alt | 1 | 0/1 | [0.0063, 0.0063] |
| Sample Trimming | 2 | 2/2 | [0.0080, 0.0112] |
| Sample Village | 3 | 2/3 | [0.0042, 0.0153] |
| Sample Min-obs | 2 | 2/2 | [0.0093, 0.0130] |
| Fixed Effects | 4 | 4/4 | [0.0065, 0.0098] |
| Alt Outcomes | 4 | 2/4 | [-0.0001, 0.0045] |
| Alt Treatments | 2 | 2/2 | [0.1569, 7.8849] |
| IV variants | 3 | 3/3 | [0.0411, 0.8835] |

## Inference Variants

| Spec ID | SE | p-value | 95% CI |
|---------|-----|---------|--------|
| infer/se/hc/hc1 | 0.004764 | 0.050575 | [-0.000023, 0.018655] |
| infer/se/cluster/hh | 0.003099 | 0.004952 | [0.003017, 0.015615] |
| infer/se/cluster/yearmonth | 0.004627 | 0.046606 | [0.000143, 0.018490] |

## Overall Assessment

- **Sign consistency:** Mixed signs across specifications
- **Significance stability:** 45/64 (70.3%) specifications significant at 5%
- **Direction:** Median coefficient is positive (0.009306)
- **Robustness assessment:** WEAK

Surface hash: `sha256:094ad910fb92ab35a26473cbf2c3755e0127e19c973de346d3e629b1530031b9`
