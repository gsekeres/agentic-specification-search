#!/usr/bin/env python3
"""
Write v9's Sample B citation table and replicated-paper bibliography.

The authoritative metadata source is data/tracking/AEA_universe.jsonl, which
contains the AEA DOI, journal, volume, issue, pages, and openICPSR identifiers
for every paper in the 103-paper Sample B catalog.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BASE_DIR.parent
CATALOG_PATH = BASE_DIR / "estimation" / "results" / "paper_catalog.json"
AEA_UNIVERSE_PATH = BASE_DIR / "data" / "tracking" / "AEA_universe.jsonl"
OUT_TABLE = ROOT_DIR / "overleaf" / "v9" / "tables" / "tab_sample_b_papers.tex"
OUT_BIB = ROOT_DIR / "overleaf" / "v9" / "replicated_papers.bib"

JOURNAL_NAMES = {
    "AEJ: Applied": "American Economic Journal: Applied Economics",
    "AEJ: Applied Economics": "American Economic Journal: Applied Economics",
    "AEJ: Policy": "American Economic Journal: Economic Policy",
    "AEJ: Macro": "American Economic Journal: Macroeconomics",
    "AEJ:MACRO": "American Economic Journal: Macroeconomics",
    "AER": "American Economic Review",
    "AER: Insights": "American Economic Review: Insights",
}

TABLE_JOURNAL_NAMES = {
    "American Economic Review": "AER",
    "American Economic Review: Insights": "AER: Insights",
    "American Economic Journal: Applied Economics": "AEJ: Applied",
    "American Economic Journal: Economic Policy": "AEJ: Policy",
    "American Economic Journal: Macroeconomics": "AEJ: Macro",
    "American Economic Journal: Microeconomics": "AEJ: Micro",
    "AEA Papers and Proceedings": "AEA P&P",
}

# The AEA universe file is complete on paper coverage, but a small set of 2022/2023
# records lacks published citation metadata. These corrections use the AEA article
# records corresponding to the replicated papers' published DOIs.
MANUAL_OVERRIDES = {
    "120483-V1": {
        "title": "The Side Effects of Immunity: Malaria and African Slavery in the United States",
        "journal": "American Economic Journal: Applied Economics",
        "year": 2022,
        "volume": "14",
        "issue": "3",
        "pages": "290-328",
        "paper_doi": "10.1257/app.20190372",
    },
    "125321-V1": {
        "journal": "American Economic Review: Insights",
        "year": 2022,
        "volume": "4",
        "issue": "1",
        "pages": "54-70",
        "paper_doi": "10.1257/aeri.20200373",
    },
    "126722-V1": {
        "journal": "American Economic Journal: Applied Economics",
        "year": 2022,
        "volume": "14",
        "issue": "1",
        "pages": "225-260",
        "paper_doi": "10.1257/app.20190722",
    },
    "128143-V1": {
        "journal": "American Economic Journal: Economic Policy",
        "year": 2022,
        "volume": "14",
        "issue": "1",
        "pages": "81-110",
        "paper_doi": "10.1257/pol.20200092",
    },
    "128521-V1": {
        "journal": "American Economic Journal: Applied Economics",
        "year": 2022,
        "volume": "14",
        "issue": "2",
        "pages": "228-255",
        "paper_doi": "10.1257/app.20190131",
    },
    "130141-V1": {
        "authors": ["G{\\\"o}rtz, Christoph", "Tsoukalas, John D.", "Zanetti, Francesco"],
        "title": "News Shocks under Financial Frictions",
        "journal": "American Economic Journal: Macroeconomics",
        "year": 2022,
        "volume": "14",
        "issue": "4",
        "pages": "210-243",
        "paper_doi": "10.1257/mac.20170066",
    },
    "138922-V1": {
        "authors": ["Marcus, Jan", "Siedler, Thomas", "Ziebarth, Nicolas R."],
        "journal": "American Economic Journal: Economic Policy",
        "year": 2022,
        "volume": "14",
        "issue": "3",
        "pages": "128-165",
        "paper_doi": "10.1257/pol.20200431",
    },
    "139262-V1": {
        "journal": "American Economic Review: Insights",
        "year": 2022,
        "volume": "4",
        "issue": "1",
        "pages": "89-105",
        "paper_doi": "10.1257/aeri.20200829",
    },
    "140921-V1": {
        "authors": ["Go{\\~n}i, Marc"],
        "journal": "American Economic Journal: Applied Economics",
        "year": 2022,
        "volume": "14",
        "issue": "3",
        "pages": "445-487",
        "paper_doi": "10.1257/app.20180463",
    },
    "149481-V1": {
        "authors": ["Samek, Anya", "Longfield, Chuck"],
        "title": "Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence",
        "journal": "American Economic Journal: Applied Economics",
        "year": 2023,
        "volume": "15",
        "issue": "2",
        "pages": "103-124",
        "paper_doi": "10.1257/app.20210068",
    },
    "150581-V1": {
        "title": "Wage Cyclicality and Labor Market Sorting",
        "journal": "American Economic Review: Insights",
        "year": 2022,
        "volume": "4",
        "issue": "4",
        "pages": "425-442",
        "paper_doi": "10.1257/aeri.20210161",
    },
    "151841-V1": {
        "authors": ["Hussam, Reshmaan", "Rigol, Natalia", "Roth, Benjamin N."],
        "title": "Targeting High Ability Entrepreneurs Using Community Information: Mechanism Design in the Field",
        "journal": "American Economic Review",
        "year": 2022,
        "volume": "112",
        "issue": "3",
        "pages": "861-898",
        "paper_doi": "10.1257/aer.20200751",
    },
    "171681-V1": {
        "journal": "American Economic Review",
        "year": 2022,
        "volume": "112",
        "issue": "11",
        "pages": "3584-3626",
        "paper_doi": "10.1257/aer.20210290",
    },
    "173341-V1": {
        "authors": ["Bobonis, Gustavo J.", "Gertler, Paul J.", "Gonzalez-Navarro, Marco", "Nichter, Simeon"],
        "journal": "American Economic Review",
        "year": 2022,
        "volume": "112",
        "issue": "11",
        "pages": "3627-3659",
        "paper_doi": "10.1257/aer.20190565",
    },
    "174501-V1": {
        "title": "Interaction, Stereotypes, and Performance: Evidence from South Africa",
        "journal": "American Economic Review",
        "year": 2022,
        "volume": "112",
        "issue": "12",
        "pages": "3848-3875",
        "paper_doi": "10.1257/aer.20181805",
    },
    "181581-V1": {
        "authors": ["Okeke, Edward N."],
        "journal": "American Economic Review",
        "year": 2023,
        "volume": "113",
        "issue": "3",
        "pages": "585-627",
        "paper_doi": "10.1257/aer.20210701",
    },
    "184041-V1": {
        "authors": ["Ngangou{\\'e}, M. Kathleen", "Schotter, Andrew"],
        "journal": "American Economic Review",
        "year": 2023,
        "volume": "113",
        "issue": "6",
        "pages": "1572-1599",
        "paper_doi": "10.1257/aer.20191927",
    },
}


def sample_key(paper_id: str) -> str:
    return "sampleb" + re.sub(r"[^0-9A-Za-z]+", "", paper_id).lower()


def clean_text(s: object) -> str:
    text = "" if s is None else str(s)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def latex_accents(s: str) -> str:
    replacements = {
        "á": r"{\'a}",
        "à": r"{\`a}",
        "ä": r"{\"a}",
        "ç": r"{\c{c}}",
        "é": r"{\'e}",
        "è": r"{\`e}",
        "ê": r"{\^e}",
        "í": r"{\'i}",
        "ñ": r"{\~n}",
        "ó": r"{\'o}",
        "ö": r"{\"o}",
        "ú": r"{\'u}",
        "ü": r"{\"u}",
        "Á": r"{\'A}",
        "Ç": r"{\c{C}}",
        "É": r"{\'E}",
        "Ñ": r"{\~N}",
        "Ö": r"{\"O}",
        "Ü": r"{\"U}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def tex_escape(s: object) -> str:
    text = clean_text(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def bib_escape(s: object) -> str:
    text = latex_accents(clean_text(s))
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def load_universe() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with AEA_UNIVERSE_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            pid = f"{row.get('icpsr_project_id')}-{row.get('icpsr_version', 'V1')}"
            out[pid] = row
    return out


def parse_catalog_authors(authors_bibtex: object) -> list[str]:
    text = clean_text(authors_bibtex)
    if not text:
        return []
    return [author.strip() for author in text.split(" and ") if author.strip()]


def is_blank(value: object) -> bool:
    return value in ("", None) or clean_text(value).lower() == "none"


def normalized_journal(value: object) -> str:
    journal = clean_text(value)
    return JOURNAL_NAMES.get(journal, journal)


def table_journal(value: object) -> str:
    journal = clean_text(value)
    return TABLE_JOURNAL_NAMES.get(journal, journal)


def merge_metadata(pid: str, universe_row: dict, catalog_row: dict | None = None) -> dict:
    row = {
        "authors": list(universe_row.get("authors") or []),
        "title": universe_row.get("title", ""),
        "journal": normalized_journal(universe_row.get("journal", "")),
        "year": universe_row.get("year", ""),
        "volume": universe_row.get("volume", ""),
        "issue": universe_row.get("issue", ""),
        "pages": universe_row.get("pages", ""),
        "paper_doi": universe_row.get("paper_doi", ""),
    }

    if catalog_row:
        fallback = {
            "authors": parse_catalog_authors(catalog_row.get("authors_bibtex")),
            "title": catalog_row.get("title", ""),
            "journal": normalized_journal(catalog_row.get("journal", "")),
            "year": catalog_row.get("year", ""),
            "volume": catalog_row.get("cr_volume", ""),
            "issue": catalog_row.get("cr_issue", ""),
            "pages": catalog_row.get("cr_pages", ""),
            "paper_doi": catalog_row.get("pub_doi", ""),
        }
        for key, value in fallback.items():
            if is_blank(row.get(key)) and not is_blank(value):
                row[key] = value

    row.update(MANUAL_OVERRIDES.get(pid, {}))
    row["journal"] = normalized_journal(row.get("journal", ""))
    row["authors"] = [clean_text(a) for a in row.get("authors", [])]
    row["title"] = clean_text(row.get("title", ""))
    row["pages"] = clean_text(row.get("pages", ""))
    return row


def authors_to_bib(authors: list[str]) -> str:
    return " and ".join(bib_escape(a) for a in authors)


def bib_entry(key: str, row: dict) -> str:
    fields = [
        ("author", authors_to_bib(row.get("authors", []))),
        ("title", bib_escape(row.get("title", ""))),
        ("journal", bib_escape(row.get("journal", ""))),
        ("year", str(row.get("year", ""))),
        ("volume", row.get("volume", "")),
        ("number", row.get("issue", "")),
        ("pages", row.get("pages", "")),
        ("doi", row.get("paper_doi", "")),
        ("url", f"https://doi.org/{row.get('paper_doi', '')}" if row.get("paper_doi") else ""),
    ]
    body = []
    for name, value in fields:
        if value not in ("", None):
            body.append(f"  {name} = {{{value}}},")
    return "@article{" + key + ",\n" + "\n".join(body) + "\n}\n"


def cohen_entry(universe: dict[str, dict]) -> str:
    for row in universe.values():
        if row.get("paper_doi") == "10.1257/pol.20180594":
            return bib_entry("cohen2022mortality", merge_metadata("cohen2022mortality", row))
    return ""


def write_table(catalog: dict, universe: dict[str, dict]) -> None:
    rows = []
    for pid in sorted(catalog):
        row = merge_metadata(pid, universe[pid], catalog[pid])
        key = sample_key(pid)
        citation = rf"\citet{{{key}}}"
        title = tex_escape(row.get("title", ""))
        journal = tex_escape(table_journal(row.get("journal", "")))
        year = tex_escape(row.get("year", ""))
        rows.append(
            rf"{tex_escape(pid)} & {citation}, \emph{{{title}}} & {journal} & {year} \\"
        )

    content = r"""\begin{longtable}{@{}p{0.12\textwidth}p{0.60\textwidth}p{0.14\textwidth}r@{}}
\caption{Sample B papers with bibliography entries. The table lists the 103 AEA-journal papers for which the pipeline constructs a standardized specification surface. Each citation appears in the bibliography through the table entry.}
\label{tab:sample_b_papers}\\
\toprule
Paper ID & Citation and title & Journal & Year \\
\midrule
\endfirsthead
\toprule
Paper ID & Citation and title & Journal & Year \\
\midrule
\endhead
\midrule
\multicolumn{4}{r}{\emph{continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + "\n\\end{longtable}\n"

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(content)


def write_bib(catalog: dict, universe: dict[str, dict]) -> None:
    entries = [
        bib_entry(sample_key(pid), merge_metadata(pid, universe[pid], catalog[pid]))
        for pid in sorted(catalog)
    ]
    extra = cohen_entry(universe)
    if extra:
        entries.append(extra)
    OUT_BIB.write_text("\n".join(entries))


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    universe = load_universe()
    missing = sorted(set(catalog) - set(universe))
    if missing:
        raise RuntimeError(f"AEA universe is missing Sample B paper IDs: {missing}")
    write_table(catalog, universe)
    write_bib(catalog, universe)
    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_BIB} with {len(catalog)} Sample B entries.")


if __name__ == "__main__":
    main()
