# Specification Search Report: 112749-V1

## Surface Summary
- **Paper**: "When the Levee Breaks: Black Migration and Economic Development in the American South"
- **Authors**: Hornbeck & Naidu, AER 2014
- **Design**: panel_fixed_effects
- **Baseline groups**: 2 (G1: labor outcomes, G2: capital outcomes)
- **Surface hash**: sha256:8df75d4a2fc2e1e8

## Execution Summary
- **Total specifications planned**: 56
- **Successfully executed**: 31
- **Failed**: 25
- **Inference variants computed**: 62

## Specifications by Type
- Baseline specs: 28
- RC specs: 28

## Results Summary
### G1 (Labor/Population)
The baseline specifications show that flood intensity significantly reduced Black population share (lnfrac_black) and Black population levels after the flood, consistent with the paper's main finding. The coefficients on f_int_1930 are negative and significant, indicating that more-flooded counties experienced larger declines in Black population share.

### G2 (Capital/Techniques)
The capital outcome results show mixed evidence for the farm mechanization channel. Equipment value and tractor adoption show positive effects of flood intensity, while mules/horses show mixed results depending on controls.

## Software Stack
- Python 3.12
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3

## Notes
- Many specifications fail with "Matrix is singular" due to collinearity among the large number of time-interacted control variables. This is inherent to the research design which includes hundreds of year-specific controls.
- The paper uses Stata's areg which handles absorbed FE differently from pyfixest in some edge cases.
- Conley spatial standard errors (used in the paper's appendix) are not implemented here.
