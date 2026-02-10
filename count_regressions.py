#!/usr/bin/env python3
"""
Count the number of regressions in each paper's replication code.

Simple approach: count lines containing regression commands after stripping comments.
No loop inflation - just raw command counts.
"""

import os
import re
import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DUMMY_DIR = BASE_DIR / "dummy_analyses"
STATUS_FILE = BASE_DIR / "data" / "tracking" / "spec_search_status.json"
OUTPUT_CSV = BASE_DIR / "timing_analyses.csv"

# Stata regression commands - match as whole words anywhere in the line
STATA_REG_PATTERN = re.compile(
    r'\b(?:reg|regress|reghdfe|areg|xtreg|ivregress|ivreg2|ivreg|'
    r'logit|logistic|probit|clogit|mlogit|ologit|oprobit|'
    r'xtlogit|xtprobit|xtpoisson|xtnbreg|xtabond2|xtabond|xtdpd|xtdpdsys|'
    r'tobit|truncreg|heckman|heckprobit|'
    r'poisson|nbreg|ppmlhdfe|zip|zinb|'
    r'qreg|sqreg|iqreg|bsqreg|'
    r'rreg|glm|nl|cnsreg|sureg|gmm|'
    r'stcox|streg|'
    r'sem|gsem|'
    r'didregress|rdrobust|'
    r'newey|newey2|prais|'
    r'asclogit|asroprobit|biprobit|mvprobit)\b',
    re.IGNORECASE
)

# Patterns that indicate a line is NOT a regression (post-estimation, etc.)
STATA_NOT_REG = re.compile(
    r'\b(?:est(?:imates)?\s+(?:store|save|restore|replay|table|stats)|'
    r'predict\b|margins\b|marginsplot\b|'
    r'test\b|testparm\b|lincom\b|nlcom\b|contrast\b|'
    r'estat\b|fitstat\b|hausman\b|'
    r'outreg2?\b|esttab\b|estout\b|'
    r'coefplot\b|'
    r'set\s+seed\b)',
    re.IGNORECASE
)

# R regression functions
R_REG_PATTERN = re.compile(
    r'\b(?:lm|glm|felm|feols|fepois|plm|ivreg|iv_robust|'
    r'lm_robust|lm_lin|'
    r'coxph|survreg|'
    r'nls|arima|VAR|'
    r'rq|crq|'
    r'rdrobust|rdestimate|'
    r'att_gt|did_multiplegt|'
    r'fixest::feols|fixest::fepois)\s*\('
)

# Python regression patterns
PY_REG_PATTERN = re.compile(
    r'\b(?:OLS|WLS|GLS|Logit|Probit|Poisson|MNLogit|'
    r'PanelOLS|RandomEffects|BetweenOLS|FirstDifferenceOLS|'
    r'IV2SLS|IVGMM|IVLIML|'
    r'feols|fepois|'
    r'LinearRegression|LogisticRegression|'
    r'smf\.ols|smf\.logit|smf\.probit|smf\.glm)\s*\('
)


def read_file(filepath: str) -> str:
    """Read file with multiple encoding attempts."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def strip_stata_comments(text: str) -> str:
    """Remove Stata comments."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'(?m)^\s*\*.*$', '', text)
    text = re.sub(r'(?m)//.*$', '', text)
    text = re.sub(r'(?m)///.*$', '', text)
    return text


def strip_r_comments(text: str) -> str:
    """Remove R comments."""
    return re.sub(r'#.*$', '', text, flags=re.MULTILINE)


def strip_py_comments(text: str) -> str:
    """Remove Python comments and docstrings."""
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    return text


def count_stata_file(filepath: str) -> int:
    """Count regression commands in a Stata .do file."""
    text = read_file(filepath)
    if not text:
        return 0

    text = strip_stata_comments(text)
    count = 0

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if line contains a regression command
        if STATA_REG_PATTERN.search(line):
            # Make sure it's not a post-estimation or output command
            if not STATA_NOT_REG.search(line):
                # Skip lines that are just variable names containing "reg" etc.
                # A regression line should have the command near the start (after prefixes)
                # Allow prefixes: quietly, xi:, eststo:, capture, noisily, by:
                clean = re.sub(
                    r'^(?:qui(?:etly)?\s+|xi(?:\s*,\s*[^:]+)?:\s*|cap(?:ture)?\s+(?:noisily\s+)?|'
                    r'noisily\s+|by\s+[^:]+:\s*|bysort\s+[^:]+:\s*|'
                    r'eststo\s*[^:]*:\s*|estadd\s*[^:]*:\s*)*',
                    '', line, flags=re.IGNORECASE
                )
                # Now check if the cleaned line starts with a regression command
                if STATA_REG_PATTERN.match(clean):
                    count += 1

    return count


def count_r_file(filepath: str) -> int:
    """Count regression calls in an R file."""
    text = read_file(filepath)
    if not text:
        return 0
    text = strip_r_comments(text)
    return len(R_REG_PATTERN.findall(text))


def count_py_file(filepath: str) -> int:
    """Count regression calls in a Python file."""
    text = read_file(filepath)
    if not text:
        return 0
    text = strip_py_comments(text)
    return len(PY_REG_PATTERN.findall(text))


def count_matlab_regressions(filepath: str) -> int:
    """Count regression-like calls in MATLAB files."""
    text = read_file(filepath)
    if not text:
        return 0
    # Remove MATLAB comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    # MATLAB regression patterns
    pattern = re.compile(
        r'\b(?:fitlm|fitglm|regress|robustfit|mnrfit|lasso|ridge|'
        r'nlinfit|fitnlm|arima|varm|estimate|'
        r'mvregress|stepwiselm|fitcecoc)\s*\('
    )
    return len(pattern.findall(text))


def count_package(package_dir: str) -> dict:
    """Count regressions in all code files in a package."""
    counts = {'stata': 0, 'r': 0, 'python': 0, 'matlab': 0,
              'do_files': 0, 'r_files': 0, 'py_files': 0, 'm_files': 0}

    for root, dirs, files in os.walk(package_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            lower = fname.lower()

            if lower.endswith('.do'):
                counts['stata'] += count_stata_file(fpath)
                counts['do_files'] += 1
            elif lower.endswith('.r') or lower.endswith('.rmd'):
                counts['r'] += count_r_file(fpath)
                counts['r_files'] += 1
            elif lower.endswith('.py'):
                counts['python'] += count_py_file(fpath)
                counts['py_files'] += 1
            elif lower.endswith('.m'):
                counts['matlab'] += count_matlab_regressions(fpath)
                counts['m_files'] += 1

    counts['total'] = counts['stata'] + counts['r'] + counts['python'] + counts['matlab']
    return counts


def get_papers_with_data() -> dict:
    """Return dict of paper_id -> paper info for papers with data."""
    with open(STATUS_FILE) as f:
        status = json.load(f)
    return {p['id']: p for p in status['packages_with_data']}


def main():
    papers = get_papers_with_data()
    all_packages = sorted(d.name for d in DUMMY_DIR.iterdir() if d.is_dir())

    print(f"Found {len(all_packages)} extracted packages")
    print(f"Papers with data in status: {len(papers)}")
    print()

    results = []
    for pkg_name in all_packages:
        pkg_dir = DUMMY_DIR / pkg_name
        counts = count_package(str(pkg_dir))

        paper_info = papers.get(pkg_name, {})
        has_data = pkg_name in papers

        results.append({
            'paper_id': pkg_name,
            'n_regressions_original': counts['total'],
            'stata': counts['stata'],
            'r': counts['r'],
            'python': counts['python'],
            'matlab': counts['matlab'],
            'do_files': counts['do_files'],
            'r_files': counts['r_files'],
            'py_files': counts['py_files'],
            'm_files': counts['m_files'],
            'has_data': has_data,
            'title': paper_info.get('title', ''),
        })

        tag = "DATA" if has_data else "NO_DATA"
        lang_parts = []
        if counts['stata']: lang_parts.append(f"{counts['stata']} Stata")
        if counts['r']: lang_parts.append(f"{counts['r']} R")
        if counts['python']: lang_parts.append(f"{counts['python']} Py")
        if counts['matlab']: lang_parts.append(f"{counts['matlab']} M")
        lang_str = ', '.join(lang_parts) if lang_parts else 'none'
        files_str = f"{counts['do_files']}do {counts['r_files']}R {counts['py_files']}py {counts['m_files']}m"
        print(f"  {pkg_name}: {counts['total']:4d} regs ({lang_str}) [{tag}] files: {files_str}")

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'paper_id', 'n_regressions_original', 'spec_search_time_s',
            'verification_time_s', 'total_agent_time_s',
            'spec_search_success', 'verification_success',
        ])
        for r in results:
            if r['has_data']:
                writer.writerow([
                    r['paper_id'],
                    r['n_regressions_original'],
                    '',  # to be filled by timing batch
                    '', '', '', '',
                ])

    print(f"\nWrote {sum(1 for r in results if r['has_data'])} rows to {OUTPUT_CSV}")

    # Summary stats
    data_results = [r for r in results if r['has_data']]
    totals = [r['n_regressions_original'] for r in data_results]
    zeros = [r['paper_id'] for r in data_results if r['n_regressions_original'] == 0]
    print(f"\nSummary for papers with data:")
    print(f"  Mean regressions: {sum(totals)/len(totals):.1f}")
    print(f"  Median: {sorted(totals)[len(totals)//2]}")
    print(f"  Min: {min(totals)}, Max: {max(totals)}")
    print(f"  Papers with 0 regressions: {len(zeros)}")
    if zeros:
        print(f"    {', '.join(zeros)}")


if __name__ == '__main__':
    main()
