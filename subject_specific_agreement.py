#!/usr/bin/env python3
"""
subject_specific_agreement.py
=============================

Subject-specific judge-pair agreement tests for SCC opinion-group data.

Input:
  judge_pair_case.csv

This script assumes judge_pair_case.csv was produced by extract_opinion_groups_v6.py
and lives either:
  - in the current folder, or
  - in opinion_group_outputs_v6/judge_pair_case.csv

It expands multi-subject cases, then calculates judge-pair agreement rates by
subject area.

Outputs:
  subject_agreement_outputs/
    subject_pair_agreement.csv
    subject_pair_deviations.csv
    subject_summary.csv
    judge_subject_summary.csv
    pair_subject_matrix_opinion_group.csv
    subject_split_heatmap.png
    most_subject_sensitive_pairs.csv
    subject_agreement_report.txt

Usage:
  python3.11 subject_specific_agreement.py \
    --input judge_pair_case.csv \
    --outdir subject_agreement_outputs

Or:
  python3.11 subject_specific_agreement.py \
    --input opinion_group_outputs_v6/judge_pair_case.csv \
    --outdir subject_agreement_outputs

Recommended first run:
  python3.11 subject_specific_agreement.py \
    --input opinion_group_outputs_v6/judge_pair_case.csv \
    --outdir subject_agreement_outputs \
    --min-pair-subject-cases 5 \
    --min-subject-cases 10 \
    --confidence high medium

Interpretation:
  subject_pair_agreement.csv:
    For each judge pair + subject, their outcome/disposition/opinion-group agreement.

  subject_pair_deviations.csv:
    For each judge pair + subject, compares subject-specific agreement to that
    pair's overall agreement. This asks:
      "Does this pair agree unusually more/less in this subject than they do overall?"

  subject_summary.csv:
    Subject-level consensus: which subjects have lower/higher average agreement.

  most_subject_sensitive_pairs.csv:
    Judge pairs whose agreement varies most across subjects.

Important caveat:
  Cases can have multiple subject tags. This script expands them, meaning the
  same case can count under multiple subjects. That is usually what you want
  for legal subject exploration, but it means subject rows are not mutually
  exclusive.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bool(x) -> Optional[bool]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def bool_mean(series: pd.Series) -> float:
    vals = series.map(parse_bool).dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())


def bool_count(series: pd.Series) -> int:
    return int(series.map(parse_bool).notna().sum())


def bool_sum(series: pd.Series) -> int:
    vals = series.map(parse_bool).dropna()
    if len(vals) == 0:
        return 0
    return int(vals.sum())


def last_name(full: str) -> str:
    return str(full).split(",", 1)[0]


def split_subjects(s: str) -> list[str]:
    if pd.isna(s) or not str(s).strip():
        return ["[No subject]"]
    parts = [p.strip() for p in str(s).split("|") if p.strip()]
    return parts if parts else ["[No subject]"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_default_input() -> Path:
    candidates = [
        Path("judge_pair_case.csv"),
        Path("opinion_group_outputs_v6/judge_pair_case.csv"),
        Path("opinion_group_outputs_v5/judge_pair_case.csv"),
        Path("opinion_group_outputs_v4/judge_pair_case.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("judge_pair_case.csv")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_pair_case(path: Path, confidence: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    df = pd.read_csv(path)

    required = {
        "case_id",
        "case_name",
        "year",
        "subjects",
        "judge_a",
        "judge_b",
        "same_outcome_vote",
        "same_disposition_side",
        "same_opinion_group",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    if confidence and "extraction_confidence" in df.columns:
        df = df[df["extraction_confidence"].isin(confidence)].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["judge_a_short"] = df["judge_a"].map(last_name)
    df["judge_b_short"] = df["judge_b"].map(last_name)
    df["pair"] = df["judge_a_short"] + " - " + df["judge_b_short"]

    return df


def expand_subjects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        subjects = split_subjects(row.get("subjects", ""))
        for subject in subjects:
            r = row.to_dict()
            r["subject"] = subject
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def summarize_agreement(grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    rows = []

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}

        # Add grouping keys by position after function caller renames them.
        row["_keys"] = keys

        row["rows"] = len(g)
        row["unique_cases"] = g["case_id"].nunique()

        row["outcome_known_cases"] = bool_count(g["same_outcome_vote"])
        row["outcome_same_cases"] = bool_sum(g["same_outcome_vote"])
        row["outcome_agreement_rate"] = bool_mean(g["same_outcome_vote"])

        row["disposition_known_cases"] = bool_count(g["same_disposition_side"])
        row["disposition_same_cases"] = bool_sum(g["same_disposition_side"])
        row["disposition_agreement_rate"] = bool_mean(g["same_disposition_side"])

        row["opinion_group_known_cases"] = bool_count(g["same_opinion_group"])
        row["opinion_group_same_cases"] = bool_sum(g["same_opinion_group"])
        row["opinion_group_agreement_rate"] = bool_mean(g["same_opinion_group"])

        row["first_year"] = int(g["year"].min()) if pd.notna(g["year"].min()) else ""
        row["last_year"] = int(g["year"].max()) if pd.notna(g["year"].max()) else ""

        rows.append(row)

    return pd.DataFrame(rows)


def subject_pair_agreement(expanded: pd.DataFrame, min_pair_subject_cases: int) -> pd.DataFrame:
    grouped = expanded.groupby(["subject", "judge_a", "judge_b", "judge_a_short", "judge_b_short", "pair"], dropna=False)
    out = summarize_agreement(grouped)

    key_cols = ["subject", "judge_a", "judge_b", "judge_a_short", "judge_b_short", "pair"]
    out[key_cols] = pd.DataFrame(out["_keys"].tolist(), index=out.index)
    out = out.drop(columns=["_keys"])

    out = out[out["opinion_group_known_cases"] >= min_pair_subject_cases].copy()
    out = out.sort_values(["subject", "opinion_group_agreement_rate", "opinion_group_known_cases"],
                          ascending=[True, True, False])
    return out


def overall_pair_agreement(df: pd.DataFrame, min_pair_cases: int) -> pd.DataFrame:
    grouped = df.groupby(["judge_a", "judge_b", "judge_a_short", "judge_b_short", "pair"], dropna=False)
    out = summarize_agreement(grouped)

    key_cols = ["judge_a", "judge_b", "judge_a_short", "judge_b_short", "pair"]
    out[key_cols] = pd.DataFrame(out["_keys"].tolist(), index=out.index)
    out = out.drop(columns=["_keys"])

    out = out[out["opinion_group_known_cases"] >= min_pair_cases].copy()
    return out


def add_deviations(subject_pair: pd.DataFrame, pair_overall: pd.DataFrame) -> pd.DataFrame:
    overall_cols = [
        "judge_a", "judge_b",
        "opinion_group_known_cases", "opinion_group_agreement_rate",
        "outcome_agreement_rate", "disposition_agreement_rate",
    ]

    overall = pair_overall[overall_cols].rename(columns={
        "opinion_group_known_cases": "overall_opinion_group_known_cases",
        "opinion_group_agreement_rate": "overall_opinion_group_agreement_rate",
        "outcome_agreement_rate": "overall_outcome_agreement_rate",
        "disposition_agreement_rate": "overall_disposition_agreement_rate",
    })

    out = subject_pair.merge(overall, on=["judge_a", "judge_b"], how="left")

    out["subject_minus_overall_opinion_group"] = (
        out["opinion_group_agreement_rate"] - out["overall_opinion_group_agreement_rate"]
    )
    out["abs_subject_minus_overall_opinion_group"] = out["subject_minus_overall_opinion_group"].abs()

    out["subject_minus_overall_outcome"] = (
        out["outcome_agreement_rate"] - out["overall_outcome_agreement_rate"]
    )

    out = out.sort_values("abs_subject_minus_overall_opinion_group", ascending=False)
    return out


def subject_summary(expanded: pd.DataFrame, min_subject_cases: int) -> pd.DataFrame:
    grouped = expanded.groupby("subject", dropna=False)
    out = summarize_agreement(grouped)

    out["subject"] = out["_keys"].apply(lambda x: x[0])
    out = out.drop(columns=["_keys"])

    out = out[out["unique_cases"] >= min_subject_cases].copy()
    out = out.sort_values(["opinion_group_agreement_rate", "unique_cases"], ascending=[True, False])
    return out


def judge_subject_summary(expanded: pd.DataFrame, min_judge_subject_cases: int) -> pd.DataFrame:
    """
    Judge-level subject tendencies from pair data.

    This is not as clean as judge_case_opinions.csv, but it answers:
      "When this judge appears in this subject, how often do their pairwise
       relationships show opinion-group agreement?"
    """
    rows = []
    for (subject, judge), g in pd.concat([
        expanded.rename(columns={"judge_a": "judge"})[["subject", "judge", "case_id", "same_opinion_group", "same_outcome_vote"]],
        expanded.rename(columns={"judge_b": "judge"})[["subject", "judge", "case_id", "same_opinion_group", "same_outcome_vote"]],
    ]).groupby(["subject", "judge"]):
        known = bool_count(g["same_opinion_group"])
        if known < min_judge_subject_cases:
            continue
        rows.append({
            "subject": subject,
            "judge": judge,
            "judge_short": last_name(judge),
            "pair_rows": len(g),
            "unique_cases": g["case_id"].nunique(),
            "opinion_group_known_pair_rows": known,
            "mean_pairwise_opinion_group_agreement": bool_mean(g["same_opinion_group"]),
            "mean_pairwise_outcome_agreement": bool_mean(g["same_outcome_vote"]),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["subject", "mean_pairwise_opinion_group_agreement"], ascending=[True, True])
    return out


def subject_sensitive_pairs(deviations: pd.DataFrame, min_subjects_per_pair: int) -> pd.DataFrame:
    rows = []
    for pair, g in deviations.groupby("pair"):
        if g["subject"].nunique() < min_subjects_per_pair:
            continue

        min_row = g.loc[g["opinion_group_agreement_rate"].idxmin()]
        max_row = g.loc[g["opinion_group_agreement_rate"].idxmax()]

        rows.append({
            "pair": pair,
            "judge_a": min_row["judge_a"],
            "judge_b": min_row["judge_b"],
            "subjects_observed": g["subject"].nunique(),
            "min_subject": min_row["subject"],
            "min_subject_rate": min_row["opinion_group_agreement_rate"],
            "min_subject_cases": min_row["opinion_group_known_cases"],
            "max_subject": max_row["subject"],
            "max_subject_rate": max_row["opinion_group_agreement_rate"],
            "max_subject_cases": max_row["opinion_group_known_cases"],
            "range_across_subjects": max_row["opinion_group_agreement_rate"] - min_row["opinion_group_agreement_rate"],
            "mean_abs_subject_deviation": g["subject_minus_overall_opinion_group"].abs().mean(),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["range_across_subjects", "mean_abs_subject_deviation"], ascending=[False, False])
    return out


# ---------------------------------------------------------------------------
# Matrix/chart outputs
# ---------------------------------------------------------------------------

def make_pair_subject_matrix(subject_pair: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mat = subject_pair.pivot_table(
        index="pair",
        columns="subject",
        values=value_col,
        aggfunc="mean",
    )

    # Order pairs by average agreement, low to high.
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=True).index]

    # Order subjects by average agreement, low to high.
    mat = mat[mat.mean(axis=0).sort_values(ascending=True).index]

    return mat


def plot_subject_heatmap(mat: pd.DataFrame, out_path: Path, title: str) -> None:
    if mat.empty:
        return

    # Keep chart readable.
    max_pairs = 35
    max_subjects = 20
    mat_plot = mat.iloc[:max_pairs, :max_subjects].copy()

    fig_w = max(12, 0.55 * len(mat_plot.columns) + 5)
    fig_h = max(9, 0.32 * len(mat_plot.index) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat_plot.values, vmin=0, vmax=1, cmap="YlGn", aspect="auto")

    ax.set_xticks(range(len(mat_plot.columns)))
    ax.set_yticks(range(len(mat_plot.index)))
    ax.set_xticklabels(mat_plot.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(mat_plot.index, fontsize=8)

    for i in range(len(mat_plot.index)):
        for j in range(len(mat_plot.columns)):
            val = mat_plot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=6, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Opinion-group agreement rate")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_subject_bars(subject_summary_df: pd.DataFrame, outdir: Path) -> None:
    if subject_summary_df.empty:
        return

    top = subject_summary_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(top["subject"], top["opinion_group_agreement_rate"])
    ax.set_title("Lowest opinion-group agreement by subject")
    ax.set_ylabel("Mean pairwise opinion-group agreement")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=70)

    for i, val in enumerate(top["opinion_group_agreement_rate"]):
        if pd.notna(val):
            ax.text(i, val, f"{val:.0%}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(outdir / "lowest_agreement_subjects.png", dpi=200)
    plt.close()


def write_report(out_path: Path, subject_summary_df: pd.DataFrame,
                 deviations: pd.DataFrame, sensitive_pairs: pd.DataFrame) -> None:
    lines = []
    lines.append("SUBJECT-SPECIFIC AGREEMENT TESTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Question:")
    lines.append("  Do judge pairs agree differently depending on subject area?")
    lines.append("")
    lines.append("Caveat:")
    lines.append("  Cases with multiple subject tags are counted under each tag.")
    lines.append("  Subject rows are therefore overlapping, not mutually exclusive.")
    lines.append("")

    if not subject_summary_df.empty:
        lines.append("Lowest-agreement subject areas:")
        lines.append("-" * 80)
        for _, r in subject_summary_df.head(15).iterrows():
            lines.append(
                f"  {r['subject']}: "
                f"opinion-group agreement={r['opinion_group_agreement_rate']:.1%}, "
                f"cases={int(r['unique_cases'])}"
            )
        lines.append("")

    if not deviations.empty:
        lines.append("Largest pair-specific subject deviations:")
        lines.append("-" * 80)
        for _, r in deviations.head(15).iterrows():
            direction = "higher" if r["subject_minus_overall_opinion_group"] > 0 else "lower"
            lines.append(
                f"  {r['pair']} in {r['subject']}: "
                f"{r['opinion_group_agreement_rate']:.1%} subject agreement vs "
                f"{r['overall_opinion_group_agreement_rate']:.1%} overall "
                f"({abs(r['subject_minus_overall_opinion_group']):.1%} {direction}); "
                f"n={int(r['opinion_group_known_cases'])}"
            )
        lines.append("")

    if not sensitive_pairs.empty:
        lines.append("Most subject-sensitive judge pairs:")
        lines.append("-" * 80)
        for _, r in sensitive_pairs.head(15).iterrows():
            lines.append(
                f"  {r['pair']}: range={r['range_across_subjects']:.1%}; "
                f"lowest={r['min_subject']} ({r['min_subject_rate']:.1%}, n={int(r['min_subject_cases'])}); "
                f"highest={r['max_subject']} ({r['max_subject_rate']:.1%}, n={int(r['max_subject_cases'])})"
            )
        lines.append("")

    lines.append("Recommended next use:")
    lines.append("  Use the largest deviations to pick exemplar cases for legal interpretation.")
    lines.append("  For example: if Côté-Karakatsanis agreement collapses in Criminal law,")
    lines.append("  inspect the cases in that subject where their same_opinion_group=false.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-specific SCC judge-pair agreement tests.")
    parser.add_argument("--input", default=None, help="Path to judge_pair_case.csv")
    parser.add_argument("--outdir", default="subject_agreement_outputs", help="Output directory")
    parser.add_argument("--confidence", nargs="*", default=None, help="Optional confidence filter, e.g. --confidence high medium")
    parser.add_argument("--min-pair-cases", type=int, default=10, help="Minimum overall pair cases for deviation baseline")
    parser.add_argument("--min-pair-subject-cases", type=int, default=5, help="Minimum pair-subject cases")
    parser.add_argument("--min-subject-cases", type=int, default=10, help="Minimum unique cases for subject summary")
    parser.add_argument("--min-judge-subject-cases", type=int, default=10, help="Minimum judge-subject pair rows")
    parser.add_argument("--min-subjects-per-pair", type=int, default=3, help="Minimum subjects needed for subject-sensitive pair table")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_default_input()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"Reading: {input_path}")
    pair_case = load_pair_case(input_path, args.confidence)

    print("Expanding subject tags...")
    expanded = expand_subjects(pair_case)

    print("Calculating subject-pair agreement...")
    sp = subject_pair_agreement(expanded, min_pair_subject_cases=args.min_pair_subject_cases)

    print("Calculating overall pair baselines...")
    overall = overall_pair_agreement(pair_case, min_pair_cases=args.min_pair_cases)

    print("Calculating deviations from each pair's overall agreement...")
    deviations = add_deviations(sp, overall)

    print("Calculating subject summaries...")
    subj = subject_summary(expanded, min_subject_cases=args.min_subject_cases)

    print("Calculating judge-subject summaries...")
    judge_subj = judge_subject_summary(expanded, min_judge_subject_cases=args.min_judge_subject_cases)

    print("Calculating subject-sensitive pairs...")
    sensitive = subject_sensitive_pairs(deviations, min_subjects_per_pair=args.min_subjects_per_pair)

    print("Writing CSV outputs...")
    expanded.to_csv(outdir / "judge_pair_case_subject_expanded.csv", index=False)
    sp.to_csv(outdir / "subject_pair_agreement.csv", index=False)
    overall.to_csv(outdir / "overall_pair_agreement.csv", index=False)
    deviations.to_csv(outdir / "subject_pair_deviations.csv", index=False)
    subj.to_csv(outdir / "subject_summary.csv", index=False)
    judge_subj.to_csv(outdir / "judge_subject_summary.csv", index=False)
    sensitive.to_csv(outdir / "most_subject_sensitive_pairs.csv", index=False)

    print("Writing matrix and charts...")
    mat = make_pair_subject_matrix(sp, "opinion_group_agreement_rate")
    mat.to_csv(outdir / "pair_subject_matrix_opinion_group.csv")
    plot_subject_heatmap(
        mat,
        outdir / "subject_pair_agreement_heatmap.png",
        "Judge-pair opinion-group agreement by subject",
    )
    write_subject_bars(subj, outdir)

    write_report(outdir / "subject_agreement_report.txt", subj, deviations, sensitive)

    print("")
    print(f"Saved subject agreement outputs to: {outdir}")
    print("Key files:")
    print(f"  {outdir / 'subject_agreement_report.txt'}")
    print(f"  {outdir / 'subject_pair_agreement.csv'}")
    print(f"  {outdir / 'subject_pair_deviations.csv'}")
    print(f"  {outdir / 'most_subject_sensitive_pairs.csv'}")
    print(f"  {outdir / 'subject_pair_agreement_heatmap.png'}")


if __name__ == "__main__":
    main()
