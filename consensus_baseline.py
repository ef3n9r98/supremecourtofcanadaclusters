#!/usr/bin/env python3
"""
consensus_baseline.py
=====================

Baseline consensus analysis for SCC opinion-group extraction outputs.

Reads the CSVs produced by extract_opinion_groups_v6.py:
  - case_level.csv
  - judge_case_opinions.csv
  - judge_pair_case.csv
  - judge_pair_summary.csv

Produces:
  - consensus_baseline_report.txt
  - consensus_case_metrics.csv
  - consensus_by_year.csv
  - consensus_by_subject.csv
  - judge_baseline.csv
  - pairwise_baseline.csv
  - several PNG charts

Usage:
  python consensus_baseline.py --input-dir opinion_group_outputs_v6 \
                               --outdir consensus_baseline_outputs
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype("boolean")
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map({
             "true": True,
             "false": False,
             "1": True,
             "0": False,
             "yes": True,
             "no": False,
             "nan": pd.NA,
             "": pd.NA,
         })
         .astype("boolean")
    )


def safe_rate(numer: float, denom: float) -> float:
    if denom == 0 or pd.isna(denom):
        return np.nan
    return numer / denom


def pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.1%}"


def last_name(full: str) -> str:
    return str(full).split(",", 1)[0]


def split_subjects(subject_string: str) -> list[str]:
    if pd.isna(subject_string) or not str(subject_string).strip():
        return []
    return [s.strip() for s in str(subject_string).split("|") if s.strip()]


def write_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                    ylabel: str, out_path: Path, rotate: int = 0) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, float(df[y_col].max()) * 1.15))

    if rotate:
        ax.tick_params(axis="x", rotation=rotate)

    if df[y_col].max() <= 1.0:
        for i, val in enumerate(df[y_col]):
            if pd.notna(val):
                ax.text(i, val, f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_case_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    df = cases.copy()

    for col in ["year", "panel_size", "num_opinion_groups", "assigned_judge_rows", "unassigned_judge_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["has_dissent", "has_concurrence", "is_unanimous_reasons", "is_split_reasons"]:
        if col in df.columns:
            df[col] = parse_bool_series(df[col])

    if "num_opinion_groups" in df.columns:
        df["has_multiple_opinion_groups"] = df["num_opinion_groups"] > 1
        df["one_opinion_group"] = df["num_opinion_groups"] == 1

    if "case_outcome" in df.columns:
        df["known_case_outcome"] = ~df["case_outcome"].isin(["unknown", ""])

    return df


def summarize_overall_cases(cases: pd.DataFrame) -> dict:
    n = len(cases)
    out = {"cases": n}

    def count_rate(col: str) -> tuple[int, float]:
        if col not in cases.columns:
            return 0, np.nan
        count = int(cases[col].fillna(False).sum())
        return count, safe_rate(count, n)

    for col in [
        "is_unanimous_reasons",
        "is_split_reasons",
        "has_dissent",
        "has_concurrence",
        "one_opinion_group",
        "has_multiple_opinion_groups",
    ]:
        count, rate = count_rate(col)
        out[f"{col}_count"] = count
        out[f"{col}_rate"] = rate

    if "num_opinion_groups" in cases.columns:
        out["mean_opinion_groups_per_case"] = cases["num_opinion_groups"].mean()
        out["median_opinion_groups_per_case"] = cases["num_opinion_groups"].median()
        out["max_opinion_groups_in_case"] = cases["num_opinion_groups"].max()

    if "panel_size" in cases.columns:
        out["mean_panel_size"] = cases["panel_size"].mean()
        out["median_panel_size"] = cases["panel_size"].median()

    return out


def summarize_by_year(cases: pd.DataFrame) -> pd.DataFrame:
    if "year" not in cases.columns:
        return pd.DataFrame()

    rows = []
    for year, g in cases.dropna(subset=["year"]).groupby("year"):
        base = summarize_overall_cases(g)
        base["year"] = int(year)
        rows.append(base)

    df = pd.DataFrame(rows).sort_values("year")
    cols_first = [
        "year", "cases", "is_unanimous_reasons_rate", "is_split_reasons_rate",
        "has_dissent_rate", "has_concurrence_rate", "mean_opinion_groups_per_case"
    ]
    cols = [c for c in cols_first if c in df.columns] + [c for c in df.columns if c not in cols_first]
    return df[cols]


def summarize_by_subject(cases: pd.DataFrame, min_cases: int = 5) -> pd.DataFrame:
    if "subjects" not in cases.columns:
        return pd.DataFrame()

    records = []
    for _, row in cases.iterrows():
        for subj in split_subjects(row.get("subjects", "")):
            r = row.to_dict()
            r["subject"] = subj
            records.append(r)

    if not records:
        return pd.DataFrame()

    expanded = pd.DataFrame(records)
    rows = []
    for subject, g in expanded.groupby("subject"):
        if len(g) < min_cases:
            continue
        base = summarize_overall_cases(g)
        base["subject"] = subject
        rows.append(base)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["is_split_reasons_rate", "cases"], ascending=[False, False])
    cols_first = [
        "subject", "cases", "is_unanimous_reasons_rate", "is_split_reasons_rate",
        "has_dissent_rate", "has_concurrence_rate", "mean_opinion_groups_per_case"
    ]
    cols = [c for c in cols_first if c in df.columns] + [c for c in df.columns if c not in cols_first]
    return df[cols]


def summarize_judges(judge_cases: pd.DataFrame) -> pd.DataFrame:
    df = judge_cases.copy()

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    rows = []
    for judge, g in df.groupby("judge"):
        n = len(g)

        def rate_for_types(types: Iterable[str]) -> float:
            return safe_rate(g["opinion_type"].isin(types).sum(), n)

        rows.append({
            "judge": judge,
            "judge_short": last_name(judge),
            "judge_case_rows": n,
            "main_or_unanimous_rate": rate_for_types(["main", "main_inferred", "unanimous"]),
            "concurrence_rate": rate_for_types(["concurrence", "concurrence_in_part"]),
            "dissent_or_partial_rate": rate_for_types(["dissent", "dissent_in_part", "mixed_partial"]),
            "unassigned_rate": rate_for_types(["unknown"]),
            "first_year": int(g["year"].min()) if "year" in g.columns and pd.notna(g["year"].min()) else "",
            "last_year": int(g["year"].max()) if "year" in g.columns and pd.notna(g["year"].max()) else "",
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["dissent_or_partial_rate", "concurrence_rate"], ascending=[False, False])


def summarize_pairwise(pair_summary: pd.DataFrame, min_pair_cases: int) -> tuple[pd.DataFrame, dict]:
    df = pair_summary.copy()

    numeric_cols = [
        "cases_together",
        "outcome_known_cases",
        "outcome_agreement_rate",
        "disposition_known_cases",
        "disposition_agreement_rate",
        "opinion_group_known_cases",
        "opinion_group_agreement_rate",
        "review_rows",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "opinion_group_known_cases" in df.columns:
        filtered = df[df["opinion_group_known_cases"] >= min_pair_cases].copy()
    else:
        filtered = df.copy()

    metrics = {
        "pair_rows_all": len(df),
        "pair_rows_min_cases": len(filtered),
        "min_pair_cases": min_pair_cases,
    }

    for rate_col in ["outcome_agreement_rate", "disposition_agreement_rate", "opinion_group_agreement_rate"]:
        if rate_col in filtered.columns:
            metrics[f"mean_{rate_col}"] = filtered[rate_col].mean()
            metrics[f"median_{rate_col}"] = filtered[rate_col].median()
            metrics[f"min_{rate_col}"] = filtered[rate_col].min()
            metrics[f"max_{rate_col}"] = filtered[rate_col].max()
            metrics[f"std_{rate_col}"] = filtered[rate_col].std()

    return filtered, metrics


def write_report(out_path: Path, overall: dict, pair_metrics: dict,
                 confidence_counts: pd.Series, outcome_counts: pd.Series) -> None:
    lines = []

    lines.append("SCC CONSENSUS BASELINE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Case-level consensus")
    lines.append("-" * 80)
    lines.append(f"Cases: {overall.get('cases', 'n/a')}")
    lines.append(f"Unanimous reasons: {overall.get('is_unanimous_reasons_count', 'n/a')} ({pct(overall.get('is_unanimous_reasons_rate', np.nan))})")
    lines.append(f"Split reasons: {overall.get('is_split_reasons_count', 'n/a')} ({pct(overall.get('is_split_reasons_rate', np.nan))})")
    lines.append(f"Cases with dissents / partial dissents: {overall.get('has_dissent_count', 'n/a')} ({pct(overall.get('has_dissent_rate', np.nan))})")
    lines.append(f"Cases with concurrences: {overall.get('has_concurrence_count', 'n/a')} ({pct(overall.get('has_concurrence_rate', np.nan))})")
    lines.append(f"Mean opinion groups per case: {overall.get('mean_opinion_groups_per_case', np.nan):.2f}")
    lines.append(f"Median opinion groups per case: {overall.get('median_opinion_groups_per_case', np.nan):.2f}")
    lines.append(f"Max opinion groups in a case: {overall.get('max_opinion_groups_in_case', 'n/a')}")
    lines.append(f"Mean panel size: {overall.get('mean_panel_size', np.nan):.2f}")
    lines.append("")

    lines.append("Pairwise consensus")
    lines.append("-" * 80)
    lines.append(f"Judge-pair rows used after minimum-case filter: {pair_metrics.get('pair_rows_min_cases', 'n/a')} / {pair_metrics.get('pair_rows_all', 'n/a')}")
    lines.append(f"Minimum known opinion-group cases per pair: {pair_metrics.get('min_pair_cases', 'n/a')}")
    lines.append(f"Mean outcome agreement: {pct(pair_metrics.get('mean_outcome_agreement_rate', np.nan))}")
    lines.append(f"Mean disposition-side agreement: {pct(pair_metrics.get('mean_disposition_agreement_rate', np.nan))}")
    lines.append(f"Mean opinion-group agreement: {pct(pair_metrics.get('mean_opinion_group_agreement_rate', np.nan))}")
    lines.append(f"Opinion-group agreement spread: {pct(pair_metrics.get('min_opinion_group_agreement_rate', np.nan))} to {pct(pair_metrics.get('max_opinion_group_agreement_rate', np.nan))}")
    lines.append("")

    lines.append("Extraction confidence")
    lines.append("-" * 80)
    if len(confidence_counts):
        for k, v in confidence_counts.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("No extraction confidence column found.")
    lines.append("")

    lines.append("Case outcomes")
    lines.append("-" * 80)
    if len(outcome_counts):
        for k, v in outcome_counts.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("No case_outcome column found.")
    lines.append("")

    lines.append("Interpretation prompt")
    lines.append("-" * 80)
    lines.append("Use this report to establish the baseline: how often the Court produces one set of reasons,")
    lines.append("how often separate reasons appear, and how high pairwise agreement is before testing whether")
    lines.append("the observed pattern is stronger than random assignment of judges to opinion groups.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline consensus analysis on SCC opinion-group outputs.")
    parser.add_argument("--input-dir", default="opinion_group_outputs_v6", help="Folder containing extraction CSV outputs")
    parser.add_argument("--outdir", default="consensus_baseline_outputs", help="Output folder for baseline files")
    parser.add_argument("--min-pair-cases", type=int, default=10, help="Minimum known cases for pairwise baseline stats")
    parser.add_argument("--min-subject-cases", type=int, default=5, help="Minimum cases per subject for subject table")
    parser.add_argument(
        "--confidence",
        nargs="*",
        default=None,
        help="Optional extraction confidence filter, e.g. --confidence high medium",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = read_csv(input_dir / "case_level.csv")
    judge_cases = read_csv(input_dir / "judge_case_opinions.csv")
    pair_summary = read_csv(input_dir / "judge_pair_summary.csv")

    if args.confidence:
        allowed = set(args.confidence)
        if "extraction_confidence" in cases.columns:
            cases = cases[cases["extraction_confidence"].isin(allowed)].copy()
        if "extraction_confidence" in judge_cases.columns:
            judge_cases = judge_cases[judge_cases["extraction_confidence"].isin(allowed)].copy()

    cases = build_case_metrics(cases)

    overall = summarize_overall_cases(cases)
    by_year = summarize_by_year(cases)
    by_subject = summarize_by_subject(cases, min_cases=args.min_subject_cases)
    judge_baseline = summarize_judges(judge_cases)
    pairwise_baseline, pair_metrics = summarize_pairwise(pair_summary, min_pair_cases=args.min_pair_cases)

    pd.DataFrame([overall]).to_csv(outdir / "consensus_case_metrics.csv", index=False)
    by_year.to_csv(outdir / "consensus_by_year.csv", index=False)
    by_subject.to_csv(outdir / "consensus_by_subject.csv", index=False)
    judge_baseline.to_csv(outdir / "judge_baseline.csv", index=False)
    pairwise_baseline.to_csv(outdir / "pairwise_baseline.csv", index=False)

    confidence_counts = cases["extraction_confidence"].value_counts() if "extraction_confidence" in cases.columns else pd.Series(dtype=int)
    outcome_counts = cases["case_outcome"].value_counts() if "case_outcome" in cases.columns else pd.Series(dtype=int)

    write_report(
        outdir / "consensus_baseline_report.txt",
        overall,
        pair_metrics,
        confidence_counts,
        outcome_counts,
    )

    if not by_year.empty:
        write_bar_chart(
            by_year,
            "year",
            "is_unanimous_reasons_rate",
            "SCC unanimity in reasons by year",
            "Share of cases with one opinion group",
            outdir / "unanimous_reasons_by_year.png",
        )
        write_bar_chart(
            by_year,
            "year",
            "is_split_reasons_rate",
            "SCC split reasons by year",
            "Share of cases with multiple opinion groups",
            outdir / "split_reasons_by_year.png",
        )

    if not by_subject.empty:
        top_subjects = by_subject.head(15).copy()
        write_bar_chart(
            top_subjects,
            "subject",
            "is_split_reasons_rate",
            "Most split-prone SCC subject areas",
            "Share of cases with multiple opinion groups",
            outdir / "split_reasons_by_subject.png",
            rotate=75,
        )

    if not judge_baseline.empty:
        write_bar_chart(
            judge_baseline.sort_values("dissent_or_partial_rate", ascending=False),
            "judge_short",
            "dissent_or_partial_rate",
            "Judge dissent / partial dissent rate",
            "Share of judge-case rows",
            outdir / "judge_dissent_partial_rate.png",
            rotate=45,
        )
        write_bar_chart(
            judge_baseline.sort_values("concurrence_rate", ascending=False),
            "judge_short",
            "concurrence_rate",
            "Judge concurrence rate",
            "Share of judge-case rows",
            outdir / "judge_concurrence_rate.png",
            rotate=45,
        )

    print(f"Saved consensus baseline outputs to: {outdir}")
    print("")
    print("Key files:")
    print(f"  {outdir / 'consensus_baseline_report.txt'}")
    print(f"  {outdir / 'consensus_by_year.csv'}")
    print(f"  {outdir / 'consensus_by_subject.csv'}")
    print(f"  {outdir / 'judge_baseline.csv'}")
    print(f"  {outdir / 'pairwise_baseline.csv'}")


if __name__ == "__main__":
    main()
