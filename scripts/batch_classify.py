#!/usr/bin/env python3
"""
Batch classifier for AEA replication packages.
Uses pattern matching for initial classification, then can be refined.
"""

import json
import re
from pathlib import Path

def classify_package(pkg: dict) -> dict:
    """Classify a single package based on title and description."""
    title = (pkg.get('title') or '').lower()
    desc = (pkg.get('description') or '').lower()
    text = title + ' ' + desc

    # Experimental indicators (NOT observational)
    experimental_patterns = [
        r'\bexperiment\b', r'\brandomized\b', r'\brct\b', r'\blab\b',
        r'\bfield experiment\b', r'\btreatment group\b', r'\bcontrol group\b',
        r'\brandomly assign', r'\bexperimental\b', r'\bsurvey experiment\b',
        r'\blaboratory\b', r'\bsubjects\b.*\btreat', r'\bauction experiment\b',
    ]

    # Simulation/Theory indicators (NOT observational)
    theory_patterns = [
        r'\bsimulation\b', r'\bcalibrat', r'\btheoretical model\b',
        r'\bcomputable general equilibrium\b', r'\bcge\b', r'\bdsge\b',
        r'\bforecasting model\b', r'\bmonte carlo\b', r'\bnumerical\b',
    ]

    # Observational indicators
    observational_patterns = [
        r'\bpanel data\b', r'\bdiff-?in-?diff', r'\bregression discontinuity\b',
        r'\binstrumental variable\b', r'\bevent stud', r'\bfixed effect',
        r'\bsynthetic control\b', r'\bdifference-?in-?difference',
        r'\bcross-?section', r'\btime series\b', r'\biv\b',
    ]

    # Restricted data indicators
    restricted_patterns = [
        r'\bproprietary\b', r'\bconfidential\b', r'\brestricted\b',
        r'\bnda\b', r'\brdc\b', r'\blicensed data\b', r'\bprivate data\b',
        r'\badministrative data\b(?!.*public)', r'\bstatistics canada\b',
    ]

    # Public data indicators
    public_patterns = [
        r'\bcensus\b', r'\bacs\b', r'\bipums\b', r'\bcps\b',
        r'\bpublicly available\b', r'\bpublic data\b', r'\bopen data\b',
        r'\bbls\b', r'\bfred\b', r'\bworld bank\b', r'\boecd\b',
        r'\bcompustat\b', r'\bcrsp\b',
    ]

    def has_pattern(patterns):
        return any(re.search(p, text) for p in patterns)

    # Classification logic
    is_experimental = has_pattern(experimental_patterns)
    is_theory = has_pattern(theory_patterns)
    has_observational_method = has_pattern(observational_patterns)
    has_restricted = has_pattern(restricted_patterns)
    has_public = has_pattern(public_patterns)

    # Determine is_observational
    if is_experimental or is_theory:
        is_observational = False
    elif has_observational_method:
        is_observational = True
    else:
        # Default to True for economics papers (most are observational)
        # unless clearly experimental/theoretical
        is_observational = True

    # Determine has_public_data
    if has_restricted and not has_public:
        has_public_data = False
    elif has_public and not has_restricted:
        has_public_data = True
    else:
        has_public_data = None  # Unknown

    # Try to detect software from description
    software_patterns = {
        'stata': r'\bstata\b|\.do\b|\.dta\b',
        'r': r'\br code\b|\.rdata\b|\.rds\b|\br scripts?\b',
        'python': r'\bpython\b|\.py\b',
        'matlab': r'\bmatlab\b|\.m\b',
        'julia': r'\bjulia\b|\.jl\b',
    }

    primary_software = 'unknown'
    for sw, pattern in software_patterns.items():
        if re.search(pattern, text):
            primary_software = sw
            break

    return {
        'is_observational': is_observational,
        'has_public_data': has_public_data,
        'primary_software': primary_software,
        'journal': None,
    }


def main():
    input_path = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/aea/data/metadata/all_replication_2022_classified.jsonl')

    # Load all packages
    packages = []
    with open(input_path) as f:
        for line in f:
            packages.append(json.loads(line))

    # Count stats before
    unclassified_before = sum(1 for p in packages if p.get('is_observational') is None)
    print(f"Packages before: {len(packages)}, unclassified: {unclassified_before}")

    # Classify unclassified packages
    classified_count = 0
    for pkg in packages:
        if pkg.get('is_observational') is None:
            classification = classify_package(pkg)
            pkg['is_observational'] = classification['is_observational']
            pkg['has_public_data'] = classification['has_public_data']
            if pkg.get('primary_software') is None:
                pkg['primary_software'] = classification['primary_software']
            classified_count += 1

    print(f"Classified {classified_count} packages")

    # Summary stats
    obs_count = sum(1 for p in packages if p.get('is_observational') == True)
    exp_count = sum(1 for p in packages if p.get('is_observational') == False)
    obs_pub = sum(1 for p in packages if p.get('is_observational') == True and p.get('has_public_data') == True)
    obs_rest = sum(1 for p in packages if p.get('is_observational') == True and p.get('has_public_data') == False)
    obs_unk = sum(1 for p in packages if p.get('is_observational') == True and p.get('has_public_data') is None)

    print(f"\nClassification summary:")
    print(f"  Observational: {obs_count}")
    print(f"    - Public data: {obs_pub}")
    print(f"    - Restricted data: {obs_rest}")
    print(f"    - Unknown data access: {obs_unk}")
    print(f"  Experimental/Theory: {exp_count}")

    # By year
    print(f"\nBy year:")
    from collections import Counter
    year_obs = Counter()
    year_exp = Counter()
    for p in packages:
        y = p.get('publication_year')
        if p.get('is_observational') == True and p.get('has_public_data') == True:
            year_obs[y] += 1
        elif p.get('is_observational') == False:
            year_exp[y] += 1

    for y in sorted(set(list(year_obs.keys()) + list(year_exp.keys()))):
        print(f"  {y}: {year_obs.get(y, 0)} obs+public, {year_exp.get(y, 0)} exp/theory")

    # Save
    with open(input_path, 'w') as f:
        for pkg in packages:
            f.write(json.dumps(pkg) + '\n')

    print(f"\nSaved to {input_path}")


if __name__ == '__main__':
    main()
