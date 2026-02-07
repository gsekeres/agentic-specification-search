import csv, os

outdir = '/Users/gabesekeres/Dropbox/papers/competition_science/agentic_specification_search/data/verification/174501-V1'
infile = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/174501-V1/specification_results.csv'

with open(infile) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

results = []

for r in rows:
    sid = r['spec_id']
    stp = r['spec_tree_path']
    ov = r['outcome_var']
    tv = r['treatment_var']
    sd = r['sample_desc']

    bg = ''
    closest = ''
    is_base = 0
    is_core = 0
    cat = 'unclear'
    why = ''
    conf = 0.5

    is_white = 'white' in sid.lower() or 'White' in sd
    is_black = 'black' in sid.lower() or 'Black' in sd
    is_full = 'full' in sid.lower() or 'Full' in sd

    if sid.startswith('baseline/'):
        is_base = 1
        is_core = 1
        conf = 1.0
        if sid == 'baseline/table3/race_iat_white':
            bg = 'G1'; closest = sid; cat = 'core_method'; why = 'Baseline: Race IAT white students (Table 3)'
        elif sid == 'baseline/table3/race_iat_black':
            bg = 'G3'; closest = sid; cat = 'core_method'; why = 'Baseline: Race IAT black students (Table 3)'
        elif sid == 'baseline/table3/academic_iat_white':
            bg = 'G4'; closest = sid; cat = 'core_method'; why = 'Baseline: Academic IAT white (Table 3)'
        elif sid == 'baseline/table3/academic_iat_black':
            bg = 'G4'; closest = sid; cat = 'core_method'; why = 'Baseline: Academic IAT black (Table 3)'
        elif sid == 'baseline/table4/gpa_black':
            bg = 'G2'; closest = sid; cat = 'core_method'; why = 'Baseline: GPA black students (Table 4)'
        elif sid == 'baseline/table4/gpa_white':
            bg = 'G5'; closest = sid; cat = 'core_method'; why = 'Baseline: GPA white students (Table 4)'
        elif sid == 'baseline/table4/gpa_full':
            bg = 'G8'; closest = sid; cat = 'core_method'; why = 'Baseline: GPA full sample (Table 4)'
        elif sid in ['baseline/table4/examspassed_black', 'baseline/table4/continue_black', 'baseline/table4/pcaperf_black']:
            bg = 'G6'; closest = sid; cat = 'core_method'; why = 'Baseline: academic outcome black (Table 4)'
        elif sid in ['baseline/table4/examspassed_white', 'baseline/table4/continue_white', 'baseline/table4/pcaperf_white', 'baseline/table4/examspassed_full', 'baseline/table4/continue_full', 'baseline/table4/pcaperf_full']:
            bg = 'G8'; closest = sid; cat = 'core_method'; why = 'Baseline: academic outcome white/full (Table 4)'
        elif 'table5' in sid:
            bg = 'G7'; closest = sid; cat = 'core_method'; why = 'Baseline: social outcome (Table 5)'
        else:
            cat = 'unclear'; why = 'Unrecognized baseline spec'

    elif sid.startswith('robust/control/'):
        is_core = 1; cat = 'core_controls'; conf = 0.9
        if ov == 'DscoreraceIAT' and is_white:
            bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'Control variation on Race IAT white'
        elif ov == 'DscoreraceIAT' and is_black:
            bg = 'G3'; closest = 'baseline/table3/race_iat_black'; why = 'Control variation on Race IAT black'
        elif ov == 'DscoreacaIAT' and is_white:
            bg = 'G4'; closest = 'baseline/table3/academic_iat_white'; why = 'Control variation on Academic IAT white'
        elif ov == 'DscoreacaIAT' and is_black:
            bg = 'G4'; closest = 'baseline/table3/academic_iat_black'; why = 'Control variation on Academic IAT black'
        elif ov == 'GPA' and is_black:
            bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'Control variation on GPA black'
        elif ov == 'GPA' and is_white:
            bg = 'G5'; closest = 'baseline/table4/gpa_white'; why = 'Control variation on GPA white'
        elif ov == 'examspassed' and is_black:
            bg = 'G6'; closest = 'baseline/table4/examspassed_black'; why = 'Control variation on exams black'
        elif ov == 'examspassed' and is_white:
            bg = 'G8'; closest = 'baseline/table4/examspassed_white'; why = 'Control variation on exams white'
        elif ov == 'continue' and is_black:
            bg = 'G6'; closest = 'baseline/table4/continue_black'; why = 'Control variation on continue black'
        elif ov == 'continue' and is_white:
            bg = 'G8'; closest = 'baseline/table4/continue_white'; why = 'Control variation on continue white'
        elif ov == 'PCAperf' and is_black:
            bg = 'G6'; closest = 'baseline/table4/pcaperf_black'; why = 'Control variation on PCAperf black'
        elif ov == 'PCAperf' and is_white:
            bg = 'G8'; closest = 'baseline/table4/pcaperf_white'; why = 'Control variation on PCAperf white'
        else:
            cat = 'unclear'; why = 'Unmatched control variation'; conf = 0.4

    elif sid.startswith('robust/inference/'):
        is_core = 1; cat = 'core_inference'; conf = 0.9
        if ov == 'DscoreraceIAT' and is_white:
            bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'SE type variation on Race IAT white'
        elif ov == 'GPA' and is_black:
            bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'SE type variation on GPA black'
        else:
            cat = 'unclear'; why = 'Unmatched inference variation'; conf = 0.4

    elif sid.startswith('robust/sample/'):
        conf = 0.85
        if 'female_' in sid or 'male_' in sid:
            is_core = 1; cat = 'core_sample'
            if ov == 'DscoreraceIAT' and is_white:
                bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'Gender subsample on Race IAT white'
            elif ov == 'DscoreraceIAT' and is_black:
                bg = 'G3'; closest = 'baseline/table3/race_iat_black'; why = 'Gender subsample on Race IAT black'
            elif ov == 'GPA' and is_black:
                bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'Gender subsample on GPA black'
            elif ov == 'GPA' and is_white:
                bg = 'G5'; closest = 'baseline/table4/gpa_white'; why = 'Gender subsample on GPA white'
            else:
                cat = 'unclear'; why = 'Unmatched gender subsample'; conf = 0.4
        elif 'drop_res' in sid:
            is_core = 1; cat = 'core_sample'
            if ov == 'GPA' and is_black:
                bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'Leave-one-residence-out on GPA black'
            elif ov == 'DscoreraceIAT' and is_white:
                bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'Leave-one-residence-out on IAT white'
            else:
                cat = 'unclear'; why = 'Unmatched drop-residence'; conf = 0.4
        else:
            cat = 'unclear'; why = 'Unrecognized sample restriction'; conf = 0.3

    elif sid.startswith('robust/placebo/'):
        is_core = 0; cat = 'noncore_placebo'; conf = 0.95
        if 'DscoreraceIATbas' in ov and is_white:
            bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'Placebo: baseline Race IAT (pre-treatment)'
        elif 'DscoreraceIATbas' in ov and is_black:
            bg = 'G3'; closest = 'baseline/table3/race_iat_black'; why = 'Placebo: baseline Race IAT (pre-treatment)'
        elif 'DscoreacaIATbas' in ov and is_white:
            bg = 'G4'; closest = 'baseline/table3/academic_iat_white'; why = 'Placebo: baseline Academic IAT (pre-treat)'
        elif 'DscoreacaIATbas' in ov and is_black:
            bg = 'G4'; closest = 'baseline/table3/academic_iat_black'; why = 'Placebo: baseline Academic IAT (pre-treat)'
        elif 'L_PCAattitude' in ov and is_white:
            bg = 'G7'; closest = 'baseline/table5/pcaattitude_white'; why = 'Placebo: baseline attitude (pre-treatment)'
        elif 'L_PCAattitude' in ov and is_black:
            bg = 'G7'; closest = 'baseline/table5/pcaattitude_black'; why = 'Placebo: baseline attitude (pre-treatment)'
        else:
            why = 'Unmatched placebo'; conf = 0.5

    elif sid.startswith('robust/heterogeneity/'):
        is_core = 0; cat = 'noncore_heterogeneity'; conf = 0.9
        if ov == 'DscoreraceIAT' and is_white:
            bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'Heterogeneity interaction on IAT white'
        elif ov == 'DscoreraceIAT' and is_black:
            bg = 'G3'; closest = 'baseline/table3/race_iat_black'; why = 'Heterogeneity interaction on IAT black'
        elif ov == 'GPA' and is_black:
            bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'Heterogeneity interaction on GPA black'
        elif ov == 'GPA' and is_white:
            bg = 'G5'; closest = 'baseline/table4/gpa_white'; why = 'Heterogeneity interaction on GPA white'
        else:
            why = 'Unmatched heterogeneity'; conf = 0.4

    elif sid.startswith('robust/estimation/'):
        is_core = 1; cat = 'core_fe'; conf = 0.85
        if ov == 'DscoreraceIAT' and is_white:
            bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'No FE variation on Race IAT white'
        elif ov == 'GPA' and is_black:
            bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'No FE variation on GPA black'
        elif ov == 'DscoreraceIAT' and is_black:
            bg = 'G3'; closest = 'baseline/table3/race_iat_black'; why = 'No FE variation on Race IAT black'
        elif ov == 'GPA' and is_white:
            bg = 'G5'; closest = 'baseline/table4/gpa_white'; why = 'No FE variation on GPA white'
        else:
            cat = 'unclear'; why = 'Unmatched estimation variation'; conf = 0.4

    elif sid.startswith('robust/outcome/'):
        conf = 0.8
        if 'year2_' in sid:
            is_core = 0; cat = 'noncore_alt_outcome'
            if is_black:
                bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'Year 2 outcome; different time period'
            elif is_white:
                bg = 'G5'; closest = 'baseline/table4/gpa_white'; why = 'Year 2 outcome; different time period'
            else:
                why = 'Year 2 outcome unmatched'
        elif ov in ['stillinres', 'Inresmix_yr2', 'Same_ro_yr2']:
            is_core = 0; cat = 'noncore_alt_outcome'; conf = 0.85
            if is_black:
                bg = 'G2'; closest = 'baseline/table4/gpa_black'
            elif is_white:
                bg = 'G5'; closest = 'baseline/table4/gpa_white'
            why = 'Residential sorting; different from baseline claims'
        elif ov in ['pctFriendsdiff_net', 'pctStudydiffer_net', 'LeisuOth', 'StudyOth']:
            is_core = 0; cat = 'noncore_alt_outcome'
            if is_white:
                bg = 'G7'; closest = 'baseline/table5/pcafriend_white'
            elif is_black:
                bg = 'G7'; closest = 'baseline/table5/pcafriend_black'
            why = 'Component social outcome; different var than PCA'
        elif ov in ['Cooperate', 'Pris_coopbelief']:
            is_core = 0; cat = 'noncore_alt_outcome'
            if is_white:
                bg = 'G7'; closest = 'baseline/table5/pcasocial_white'
            elif is_black:
                bg = 'G7'; closest = 'baseline/table5/pcasocial_black'
            elif is_full:
                bg = 'G7'; closest = 'baseline/table5/pcasocial_full'
            why = 'Pro-social behavior; not in baseline tables'
        else:
            is_core = 0; cat = 'noncore_alt_outcome'; why = 'Unmatched alt outcome'; conf = 0.5

    elif sid.startswith('robust/funcform/'):
        is_core = 1; cat = 'core_funcform'; conf = 0.9
        if 'DscoreraceIAT_std' in ov and is_white:
            bg = 'G1'; closest = 'baseline/table3/race_iat_white'; why = 'Standardized Race IAT; same estimand'
        elif 'DscoreraceIAT_std' in ov and is_black:
            bg = 'G3'; closest = 'baseline/table3/race_iat_black'; why = 'Standardized Race IAT; same estimand'
        elif 'GPA_std' in ov and is_white:
            bg = 'G5'; closest = 'baseline/table4/gpa_white'; why = 'Standardized GPA; same estimand'
        elif 'GPA_std' in ov and is_black:
            bg = 'G2'; closest = 'baseline/table4/gpa_black'; why = 'Standardized GPA; same estimand'
        else:
            cat = 'unclear'; why = 'Unmatched functional form'; conf = 0.4
    else:
        cat = 'unclear'; why = 'Unrecognized spec_id prefix'; conf = 0.3

    results.append({
        'paper_id': '174501-V1', 'spec_id': sid, 'spec_tree_path': stp,
        'outcome_var': ov, 'treatment_var': tv, 'baseline_group_id': bg,
        'closest_baseline_spec_id': closest, 'is_baseline': is_base,
        'is_core_test': is_core, 'category': cat, 'why': why, 'confidence': conf
    })

fieldnames = ['paper_id', 'spec_id', 'spec_tree_path', 'outcome_var', 'treatment_var',
              'baseline_group_id', 'closest_baseline_spec_id', 'is_baseline', 'is_core_test',
              'category', 'why', 'confidence']

with open(os.path.join(outdir, 'verification_spec_map.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

from collections import Counter
cats = Counter(r['category'] for r in results)
print(f'Total: {len(results)}')
for c, n in sorted(cats.items()):
    print(f'  {c}: {n}')
core_count = sum(1 for r in results if r['is_core_test'] == 1)
noncore_count = sum(1 for r in results if r['is_core_test'] == 0)
baseline_count = sum(1 for r in results if r['is_baseline'] == 1)
print(f'Baselines: {baseline_count}, Core: {core_count}, Non-core: {noncore_count}')
