#!/usr/bin/env python3
"""
extract_mds_exemplar_cases.py
=============================

Extract exemplar SCC cases that help interpret MDS dimensions.

Purpose
-------
Your 3D MDS map tells you which judges are far apart in a hidden agreement space.
This script asks:

  For each MDS dimension, which cases best show the judges at opposite ends
  splitting into different opinion groups?

It uses:
  - mds_3d_outputs/mds_3d_coordinates.csv
  - opinion_group_outputs_v6/judge_case_opinions.csv
  - opinion_group_outputs_v6/judge_pair_case.csv
  - opinion_group_outputs_v6/case_level.csv
  - scc_cases_2016-2026.jsonl

Main idea
---------
For each dimension, the script:
  1. Finds judges at the high end and low end of that dimension.
  2. Finds cases where high-end judges and low-end judges both sat.
  3. Scores cases where they split into different opinion groups.
  4. Pulls metadata and a readable excerpt from the JSONL file.
  5. Writes CSVs and text reports for human inspection.

Recommended run
---------------
  python extract_mds_exemplar_cases.py \
    --coords mds_3d_outputs/mds_3d_coordinates.csv \
    --input-dir opinion_group_outputs_v6 \
    --jsonl scc_cases_2016-2026.jsonl \
    --outdir mds_exemplar_outputs \
    --top-n-judges 3 \
    --top-n-cases 20

Outputs
-------
  mds_exemplar_outputs/
    dimension_judge_extremes.csv
    dimension_exemplar_cases.csv
    dimension_pair_exemplar_cases.csv
    mds_dimension_exemplar_report.txt
    dimension_x_exemplars.txt
    dimension_y_exemplars.txt
    dimension_z_exemplars.txt

Interpretation
--------------
This script does not prove what a dimension "means." It gives you the cases to read.
A dimension gets legal meaning only after you inspect the cases at its extremes.
"""

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def last_name(full: str) -> str:
    return str(full).split(",", 1)[0].strip()


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
        .replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
    )


def compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(s)).strip()


def split_subjects(s: Any) -> list[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    return [p.strip() for p in str(s).split("|") if p.strip()]


def pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.1%}"
    except Exception:
        return "n/a"


def find_default(path_options: list[str]) -> Optional[Path]:
    for p in path_options:
        path = Path(p)
        if path.exists():
            return path
    return None


def parse_year_from_date(date_s: Any) -> Any:
    s = str(date_s or "")
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path)


def load_coords(path: Path) -> pd.DataFrame:
    df = read_csv_required(path, "MDS coordinates")
    required = {"judge", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    for col in ["x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["judge_short"] = df["judge"].map(last_name)
    return df


def load_case_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """
    Load SCC JSONL records and key them by likely case IDs.

    The extraction script normally uses citation_en as case_id.
    We also index by citation2_en and url_en where available.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    out: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: could not parse JSONL line {line_no}")
                continue

            keys = [
                obj.get("citation_en"),
                obj.get("citation2_en"),
                obj.get("url_en"),
                obj.get("case_id"),
            ]

            for key in keys:
                if key:
                    out[str(key)] = obj

    return out


def build_case_metadata(case_level: pd.DataFrame,
                        json_cases: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}

    if not case_level.empty and "case_id" in case_level.columns:
        for _, row in case_level.iterrows():
            case_id = str(row.get("case_id", ""))
            if not case_id:
                continue
            meta[case_id] = {
                "case_id": case_id,
                "case_name": row.get("case_name", ""),
                "year": row.get("year", ""),
                "subjects": row.get("subjects", ""),
                "appeal_from": row.get("appeal_from", ""),
                "case_outcome": row.get("case_outcome", ""),
                "num_opinion_groups": row.get("num_opinion_groups", ""),
                "panel_size": row.get("panel_size", ""),
                "is_unanimous_reasons": row.get("is_unanimous_reasons", ""),
                "is_split_reasons": row.get("is_split_reasons", ""),
                "has_dissent": row.get("has_dissent", ""),
                "has_concurrence": row.get("has_concurrence", ""),
            }

    # Add records that may only exist in the JSONL.
    for key, obj in json_cases.items():
        citation = obj.get("citation_en") or obj.get("citation2_en") or obj.get("url_en") or key
        citation = str(citation)
        if citation not in meta:
            meta[citation] = {
                "case_id": citation,
                "case_name": obj.get("name_en", ""),
                "year": parse_year_from_date(obj.get("document_date_en")),
                "subjects": "",
                "appeal_from": "",
                "case_outcome": "",
                "num_opinion_groups": "",
                "panel_size": "",
                "is_unanimous_reasons": "",
                "is_split_reasons": "",
                "has_dissent": "",
                "has_concurrence": "",
            }

    return meta


# ---------------------------------------------------------------------------
# Excerpt extraction
# ---------------------------------------------------------------------------

def get_json_record(case_id: str, json_cases: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    if case_id in json_cases:
        return json_cases[case_id]

    # Defensive fuzzy fallback for cases where case_id spacing differs.
    cid_norm = compact_ws(case_id).lower()
    for key, obj in json_cases.items():
        if compact_ws(key).lower() == cid_norm:
            return obj
        citation = compact_ws(obj.get("citation_en", "")).lower()
        if citation == cid_norm:
            return obj

    return None


def extract_header_excerpt(text: str, max_chars: int = 2400) -> str:
    """
    Pull a useful early excerpt: Coram, Held, and opinion headings when possible.
    """
    text = normalize_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    useful = []
    triggers = (
        "Coram:",
        "Held:",
        "Reasons for Judgment",
        "Reasons for the Judgment",
        "Joint Reasons",
        "Majority Reasons",
        "Concurring Reasons",
        "Dissenting Reasons",
        "Reasons Concurring",
        "Reasons Dissenting",
        "Unanimous Reasons",
        "Unanimous Judgment",
        "Per ",
    )

    for line in lines[:260]:
        if line.startswith(triggers):
            useful.append(line)

    if not useful:
        return compact_ws("\n".join(lines[:80]))[:max_chars]

    return "\n".join(useful)[:max_chars]


def extract_text_snippet_around_judges(text: str,
                                       judge_names: list[str],
                                       max_chars: int = 1800) -> str:
    """
    Try to pull a snippet near the first mention of the judges' last names.
    This is only a convenience snippet, not a substitute for reading the case.
    """
    text_norm = normalize_text(text)
    lowered = text_norm.lower()

    last_names = [last_name(j).lower() for j in judge_names if j]
    positions = []
    for ln in last_names:
        pos = lowered.find(ln.lower())
        if pos >= 0:
            positions.append(pos)

    if not positions:
        return ""

    center = min(positions)
    start = max(0, center - max_chars // 2)
    end = min(len(text_norm), center + max_chars // 2)
    return compact_ws(text_norm[start:end])


# ---------------------------------------------------------------------------
# Judge extremes and case scoring
# ---------------------------------------------------------------------------

def dimension_extremes(coords: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []

    for dim in ["x", "y", "z"]:
        high = coords.sort_values(dim, ascending=False).head(top_n).copy()
        low = coords.sort_values(dim, ascending=True).head(top_n).copy()

        for side, df_side in [("high", high), ("low", low)]:
            for rank, (_, r) in enumerate(df_side.iterrows(), start=1):
                rows.append({
                    "dimension": dim,
                    "side": side,
                    "rank": rank,
                    "judge": r["judge"],
                    "judge_short": r["judge_short"],
                    "coordinate": r[dim],
                    "x": r["x"],
                    "y": r["y"],
                    "z": r["z"],
                    "mean_observed_agreement_rate": r.get("mean_observed_agreement_rate", np.nan),
                    "mean_z_score": r.get("mean_z_score", np.nan),
                })

    return pd.DataFrame(rows)


def prepare_judge_cases(judge_cases: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "judge", "opinion_group"}
    missing = required - set(judge_cases.columns)
    if missing:
        raise ValueError(f"judge_case_opinions.csv is missing required columns: {sorted(missing)}")

    df = judge_cases.copy()
    df["judge_short"] = df["judge"].map(last_name)
    df["opinion_group"] = df["opinion_group"].astype(str)
    df = df[df["opinion_group"].notna()].copy()
    df = df[df["opinion_group"].astype(str).str.strip() != ""].copy()
    df = df[df["opinion_group"].astype(str) != "UNASSIGNED"].copy()
    return df


def prepare_pair_cases(pair_cases: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "judge_a", "judge_b", "same_opinion_group"}
    missing = required - set(pair_cases.columns)
    if missing:
        raise ValueError(f"judge_pair_case.csv is missing required columns: {sorted(missing)}")

    df = pair_cases.copy()
    df["judge_a_short"] = df["judge_a"].map(last_name)
    df["judge_b_short"] = df["judge_b"].map(last_name)

    # Normalize pair key alphabetically by short name.
    a = df["judge_a_short"].astype(str)
    b = df["judge_b_short"].astype(str)
    df["pair_short"] = np.where(a < b, a + " - " + b, b + " - " + a)

    df["same_opinion_group_bool"] = (
        df["same_opinion_group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )

    return df


def build_pair_strength_lookup(pair_rows_used: Optional[pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """
    Optional lookup from mds_3d_pair_rows_used.csv or permutation_pair_results.csv.
    """
    lookup: dict[str, dict[str, Any]] = {}

    if pair_rows_used is None or pair_rows_used.empty:
        return lookup

    df = pair_rows_used.copy()
    if "judge_a_short" not in df.columns or "judge_b_short" not in df.columns:
        return lookup

    for _, r in df.iterrows():
        a = str(r["judge_a_short"])
        b = str(r["judge_b_short"])
        key = " - ".join(sorted([a, b]))
        lookup[key] = {
            "pair_cases_together": r.get("cases_together", np.nan),
            "pair_observed_agreement_rate": r.get("observed_agreement_rate", np.nan),
            "pair_expected_agreement_rate": r.get("expected_agreement_rate", np.nan),
            "pair_observed_minus_expected": r.get("observed_minus_expected", np.nan),
            "pair_z_score": r.get("z_score", np.nan),
            "pair_p_two_sided": r.get("p_two_sided", np.nan),
        }

    return lookup


def score_dimension_cases(coords: pd.DataFrame,
                          extremes: pd.DataFrame,
                          judge_cases: pd.DataFrame,
                          pair_cases: pd.DataFrame,
                          case_meta: dict[str, dict[str, Any]],
                          json_cases: dict[str, dict[str, Any]],
                          pair_strength: dict[str, dict[str, Any]],
                          top_n_cases: int,
                          require_split: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      dimension_exemplar_cases.csv
      dimension_pair_exemplar_cases.csv

    Case-level scoring:
      For every case containing at least one high-end and one low-end judge,
      calculate how often cross-side judge pairs split into different opinion groups.

    Pair-level scoring:
      For every high-end/low-end judge pair, list the strongest individual split cases.
    """
    coord_lookup = coords.set_index("judge_short")[["x", "y", "z"]].to_dict("index")

    all_case_rows = []
    all_pair_rows = []

    # Pre-index judge_cases by case.
    case_to_judges = {
        str(case_id): g.copy()
        for case_id, g in judge_cases.groupby("case_id")
    }

    pair_cases_by_case = {
        str(case_id): g.copy()
        for case_id, g in pair_cases.groupby("case_id")
    }

    for dim in ["x", "y", "z"]:
        high_judges = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "high")]["judge_short"].tolist()
        low_judges = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "low")]["judge_short"].tolist()

        high_set = set(high_judges)
        low_set = set(low_judges)

        for case_id, g in case_to_judges.items():
            present = set(g["judge_short"])

            high_present = sorted(present & high_set)
            low_present = sorted(present & low_set)

            if not high_present or not low_present:
                continue

            cross_pairs = list(product(high_present, low_present))
            pair_rows_for_case = pair_cases_by_case.get(case_id, pd.DataFrame())

            cross_known = 0
            cross_split = 0
            cross_same = 0
            cross_pair_details = []

            for a, b in cross_pairs:
                pair_key = " - ".join(sorted([a, b]))

                # Pull pair-case row, if it exists.
                if not pair_rows_for_case.empty:
                    mask = pair_rows_for_case["pair_short"].eq(pair_key)
                    matched = pair_rows_for_case[mask]
                else:
                    matched = pd.DataFrame()

                if matched.empty:
                    continue

                row = matched.iloc[0]
                same = row.get("same_opinion_group_bool", np.nan)

                if pd.isna(same):
                    continue

                cross_known += 1
                if bool(same):
                    cross_same += 1
                else:
                    cross_split += 1

                # Pair strength from overall permutation/MDS pair results.
                strength = pair_strength.get(pair_key, {})

                # Dimension gap in coordinate space along this one axis.
                dim_gap = abs(coord_lookup.get(a, {}).get(dim, np.nan) - coord_lookup.get(b, {}).get(dim, np.nan))

                pair_detail = {
                    "dimension": dim,
                    "case_id": case_id,
                    "high_judge": a if a in high_set else b,
                    "low_judge": b if a in high_set else a,
                    "pair": pair_key,
                    "same_opinion_group": bool(same),
                    "dimension_coordinate_gap": dim_gap,
                    "pair_z_score": strength.get("pair_z_score", np.nan),
                    "pair_observed_agreement_rate": strength.get("pair_observed_agreement_rate", np.nan),
                    "pair_expected_agreement_rate": strength.get("pair_expected_agreement_rate", np.nan),
                    "pair_cases_together": strength.get("pair_cases_together", np.nan),
                    "pair_p_two_sided": strength.get("pair_p_two_sided", np.nan),
                }
                cross_pair_details.append(pair_detail)

            if cross_known == 0:
                continue

            if require_split and cross_split == 0:
                continue

            meta = case_meta.get(case_id, {"case_id": case_id})
            json_obj = get_json_record(case_id, json_cases)
            full_text = ""
            if json_obj:
                full_text = json_obj.get("unofficial_text_en") or json_obj.get("text_en") or ""

            # Summary of opinion groups for present high/low judges.
            relevant = g[g["judge_short"].isin(high_present + low_present)].copy()
            relevant["side"] = np.where(relevant["judge_short"].isin(high_set), "high", "low")
            relevant_summary = []
            for _, rr in relevant.sort_values(["side", "judge_short"]).iterrows():
                relevant_summary.append(
                    f"{rr['judge_short']}[{rr['side']}]: "
                    f"group={rr.get('opinion_group', '')}; "
                    f"type={rr.get('opinion_type', '')}; "
                    f"outcome={rr.get('outcome_vote', '')}; "
                    f"side={rr.get('disposition_side', '')}"
                )

            # Scoring:
            # - Prefer cases where more cross-side pairs split.
            # - Prefer cases with higher split rate.
            # - Prefer cases where many extreme judges are present.
            # - Prefer cases where pair z-scores are strongly negative, if available.
            split_rate = cross_split / cross_known if cross_known else np.nan
            present_extreme_count = len(high_present) + len(low_present)

            pair_z_values = [d.get("pair_z_score", np.nan) for d in cross_pair_details]
            pair_z_values = [float(z) for z in pair_z_values if pd.notna(z)]
            mean_pair_z = float(np.mean(pair_z_values)) if pair_z_values else np.nan
            min_pair_z = float(np.min(pair_z_values)) if pair_z_values else np.nan

            case_score = (
                3.0 * cross_split
                + 2.0 * split_rate
                + 0.5 * present_extreme_count
                + (abs(min_pair_z) if pd.notna(min_pair_z) and min_pair_z < 0 else 0.0)
            )

            header_excerpt = extract_header_excerpt(full_text)
            judge_snippet = extract_text_snippet_around_judges(full_text, high_present + low_present)

            all_case_rows.append({
                "dimension": dim,
                "case_id": case_id,
                "case_name": meta.get("case_name", ""),
                "year": meta.get("year", ""),
                "subjects": meta.get("subjects", ""),
                "appeal_from": meta.get("appeal_from", ""),
                "case_outcome": meta.get("case_outcome", ""),
                "panel_size": meta.get("panel_size", ""),
                "num_opinion_groups": meta.get("num_opinion_groups", ""),
                "has_dissent": meta.get("has_dissent", ""),
                "has_concurrence": meta.get("has_concurrence", ""),
                "high_extreme_judges_present": "; ".join(high_present),
                "low_extreme_judges_present": "; ".join(low_present),
                "cross_side_known_pairs": cross_known,
                "cross_side_split_pairs": cross_split,
                "cross_side_same_pairs": cross_same,
                "cross_side_split_rate": split_rate,
                "present_extreme_count": present_extreme_count,
                "mean_cross_pair_z_score": mean_pair_z,
                "min_cross_pair_z_score": min_pair_z,
                "case_score": case_score,
                "relevant_judge_opinion_groups": " | ".join(relevant_summary),
                "header_excerpt": header_excerpt,
                "judge_snippet": judge_snippet,
            })

            for d in cross_pair_details:
                if require_split and d["same_opinion_group"]:
                    continue
                meta = case_meta.get(case_id, {"case_id": case_id})
                d.update({
                    "case_name": meta.get("case_name", ""),
                    "year": meta.get("year", ""),
                    "subjects": meta.get("subjects", ""),
                    "appeal_from": meta.get("appeal_from", ""),
                    "case_outcome": meta.get("case_outcome", ""),
                    "num_opinion_groups": meta.get("num_opinion_groups", ""),
                })
                all_pair_rows.append(d)

    case_df = pd.DataFrame(all_case_rows)
    pair_df = pd.DataFrame(all_pair_rows)

    if not case_df.empty:
        case_df = case_df.sort_values(["dimension", "case_score", "cross_side_split_pairs"], ascending=[True, False, False])
        case_df = case_df.groupby("dimension", group_keys=False).head(top_n_cases).copy()

    if not pair_df.empty:
        # Sort strongest split examples first:
        # high coordinate gap, negative z-score, lots of pair cases together.
        pair_df["sort_pair_z"] = pd.to_numeric(pair_df["pair_z_score"], errors="coerce")
        pair_df["sort_pair_cases"] = pd.to_numeric(pair_df["pair_cases_together"], errors="coerce")
        pair_df = pair_df.sort_values(
            ["dimension", "same_opinion_group", "dimension_coordinate_gap", "sort_pair_z", "sort_pair_cases"],
            ascending=[True, True, False, True, False],
        )
        pair_df = pair_df.drop(columns=["sort_pair_z", "sort_pair_cases"])

    return case_df, pair_df


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_dimension_text_reports(outdir: Path,
                                 extremes: pd.DataFrame,
                                 case_df: pd.DataFrame) -> None:
    for dim in ["x", "y", "z"]:
        lines = []
        lines.append(f"MDS DIMENSION {dim.upper()} EXEMPLAR CASES")
        lines.append("=" * 90)
        lines.append("")

        high = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "high")].sort_values("rank")
        low = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "low")].sort_values("rank")

        lines.append("High-end judges:")
        for _, r in high.iterrows():
            lines.append(f"  {r['rank']}. {r['judge_short']} ({dim}={r['coordinate']:.4f})")
        lines.append("")

        lines.append("Low-end judges:")
        for _, r in low.iterrows():
            lines.append(f"  {r['rank']}. {r['judge_short']} ({dim}={r['coordinate']:.4f})")
        lines.append("")

        subset = case_df[case_df["dimension"] == dim].copy() if not case_df.empty else pd.DataFrame()

        if subset.empty:
            lines.append("No exemplar cases found for this dimension under the current filters.")
        else:
            lines.append("Top exemplar cases:")
            lines.append("")
            for rank, (_, r) in enumerate(subset.iterrows(), start=1):
                lines.append("-" * 90)
                lines.append(f"{rank}. {r.get('case_name', '')} ({r.get('year', '')})")
                lines.append(f"   Case ID: {r.get('case_id', '')}")
                lines.append(f"   Subjects: {r.get('subjects', '')}")
                lines.append(f"   Appeal from: {r.get('appeal_from', '')}")
                lines.append(f"   Outcome: {r.get('case_outcome', '')}")
                lines.append(
                    f"   Cross-side split pairs: {r.get('cross_side_split_pairs', '')}/"
                    f"{r.get('cross_side_known_pairs', '')} "
                    f"({pct(r.get('cross_side_split_rate', np.nan))})"
                )
                lines.append(f"   High-end judges present: {r.get('high_extreme_judges_present', '')}")
                lines.append(f"   Low-end judges present: {r.get('low_extreme_judges_present', '')}")
                lines.append(f"   Relevant opinion groups: {r.get('relevant_judge_opinion_groups', '')}")
                lines.append("")
                lines.append("   Header / opinion excerpt:")
                excerpt = str(r.get("header_excerpt", "") or "")
                if excerpt:
                    for line in excerpt.splitlines()[:20]:
                        lines.append(f"     {line}")
                else:
                    lines.append("     n/a")
                lines.append("")

        (outdir / f"dimension_{dim}_exemplars.txt").write_text("\n".join(lines), encoding="utf-8")


def write_main_report(out_path: Path,
                      coords: pd.DataFrame,
                      extremes: pd.DataFrame,
                      case_df: pd.DataFrame,
                      pair_df: pd.DataFrame,
                      missing_df: Optional[pd.DataFrame],
                      args: argparse.Namespace) -> None:
    lines = []

    lines.append("MDS DIMENSION EXEMPLAR CASE EXTRACTION")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Coordinates: {args.coords}")
    lines.append(f"Input directory: {args.input_dir}")
    lines.append(f"JSONL: {args.jsonl}")
    lines.append(f"Top/bottom judges per dimension: {args.top_n_judges}")
    lines.append(f"Top cases per dimension: {args.top_n_cases}")
    lines.append(f"Require cross-side split: {args.require_split}")
    lines.append("")

    if missing_df is not None and not missing_df.empty:
        lines.append("Caution:")
        lines.append(f"  The MDS run had {len(missing_df)} missing judge-pair distances filled.")
        lines.append("  Treat dimensions involving those judge-pair relationships cautiously.")
        lines.append("")

    lines.append("Dimension extremes:")
    for dim in ["x", "y", "z"]:
        lines.append("")
        lines.append(f"Dimension {dim.upper()}:")
        high = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "high")].sort_values("rank")
        low = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "low")].sort_values("rank")
        lines.append("  High end: " + ", ".join([f"{r['judge_short']} ({r['coordinate']:.2f})" for _, r in high.iterrows()]))
        lines.append("  Low end: " + ", ".join([f"{r['judge_short']} ({r['coordinate']:.2f})" for _, r in low.iterrows()]))

        subset = case_df[case_df["dimension"] == dim].head(5) if not case_df.empty else pd.DataFrame()
        if not subset.empty:
            lines.append("  Top case leads:")
            for _, r in subset.iterrows():
                lines.append(
                    f"    - {r.get('case_name', '')} ({r.get('year', '')}): "
                    f"{r.get('cross_side_split_pairs', '')}/{r.get('cross_side_known_pairs', '')} "
                    f"cross-side pairs split; subjects={r.get('subjects', '')}"
                )

    lines.append("")
    lines.append("Closest/farthest pair-level exemplar rows:")
    if pair_df.empty:
        lines.append("  No pair-level exemplar rows found.")
    else:
        for dim in ["x", "y", "z"]:
            subset = pair_df[pair_df["dimension"] == dim].head(10)
            lines.append("")
            lines.append(f"Dimension {dim.upper()} pair examples:")
            for _, r in subset.iterrows():
                lines.append(
                    f"  {r.get('pair', '')} in {r.get('case_name', '')} ({r.get('year', '')}): "
                    f"same_opinion_group={r.get('same_opinion_group', '')}; "
                    f"dim_gap={r.get('dimension_coordinate_gap', np.nan):.2f}; "
                    f"pair_z={r.get('pair_z_score', np.nan):.2f}; "
                    f"subjects={r.get('subjects', '')}"
                )

    lines.append("")
    lines.append("How to use this:")
    lines.append("  Read the top cases for each dimension. Ask what legal issue, remedy,")
    lines.append("  interpretive method, or institutional posture separates the high-end judges")
    lines.append("  from the low-end judges. Only then name the dimension.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract exemplar SCC cases for interpreting MDS dimensions.")
    parser.add_argument("--coords", default=None,
                        help="Path to mds_3d_coordinates.csv")
    parser.add_argument("--input-dir", default="opinion_group_outputs_v6",
                        help="Folder containing judge_case_opinions.csv, judge_pair_case.csv, case_level.csv")
    parser.add_argument("--jsonl", default="scc_cases_2016-2026.jsonl",
                        help="Path to raw SCC JSONL file")
    parser.add_argument("--pair-rows-used", default=None,
                        help="Optional path to mds_3d_pair_rows_used.csv or permutation_pair_results.csv")
    parser.add_argument("--missing-distances", default=None,
                        help="Optional path to mds_3d_missing_distances.csv")
    parser.add_argument("--outdir", default="mds_exemplar_outputs",
                        help="Output directory")
    parser.add_argument("--top-n-judges", type=int, default=3,
                        help="Number of high-end and low-end judges to use per dimension")
    parser.add_argument("--top-n-cases", type=int, default=20,
                        help="Number of exemplar cases to keep per dimension")
    parser.add_argument("--require-split", action="store_true", default=True,
                        help="Only keep cases where at least one high-vs-low pair split")
    parser.add_argument("--include-non-splits", action="store_true",
                        help="Include cases where high-vs-low judges sat together even if they did not split")
    args = parser.parse_args()

    if args.include_non_splits:
        args.require_split = False

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    coords_path = Path(args.coords) if args.coords else find_default([
        "mds_3d_outputs/mds_3d_coordinates.csv",
        "mds_outputs/mds_coordinates.csv",
    ])
    if coords_path is None:
        raise FileNotFoundError("Could not find MDS coordinates. Provide --coords.")

    input_dir = Path(args.input_dir)
    judge_cases_path = input_dir / "judge_case_opinions.csv"
    pair_cases_path = input_dir / "judge_pair_case.csv"
    case_level_path = input_dir / "case_level.csv"
    jsonl_path = Path(args.jsonl)

    pair_rows_path = Path(args.pair_rows_used) if args.pair_rows_used else find_default([
        "mds_3d_outputs/mds_3d_pair_rows_used.csv",
        "mds_outputs/mds_pair_rows_used.csv",
        "permutation_outputs/permutation_pair_results.csv",
    ])

    missing_path = Path(args.missing_distances) if args.missing_distances else find_default([
        "mds_3d_outputs/mds_3d_missing_distances.csv",
        "mds_outputs/mds_missing_distances.csv",
    ])

    print("Loading MDS coordinates...")
    coords = load_coords(coords_path)

    print("Loading opinion-group outputs...")
    judge_cases_raw = read_csv_required(judge_cases_path, "judge_case_opinions.csv")
    pair_cases_raw = read_csv_required(pair_cases_path, "judge_pair_case.csv")
    case_level = read_csv_required(case_level_path, "case_level.csv")

    judge_cases = prepare_judge_cases(judge_cases_raw)
    pair_cases = prepare_pair_cases(pair_cases_raw)

    print("Loading raw JSONL cases...")
    json_cases = load_case_jsonl(jsonl_path)

    print("Building metadata...")
    case_meta = build_case_metadata(case_level, json_cases)

    print("Loading optional pair-strength rows...")
    pair_strength_df = pd.read_csv(pair_rows_path) if pair_rows_path and pair_rows_path.exists() else pd.DataFrame()
    pair_strength = build_pair_strength_lookup(pair_strength_df)

    missing_df = pd.read_csv(missing_path) if missing_path and missing_path.exists() else pd.DataFrame()

    print("Finding dimension extremes...")
    extremes = dimension_extremes(coords, args.top_n_judges)

    print("Scoring exemplar cases...")
    case_df, pair_df = score_dimension_cases(
        coords=coords,
        extremes=extremes,
        judge_cases=judge_cases,
        pair_cases=pair_cases,
        case_meta=case_meta,
        json_cases=json_cases,
        pair_strength=pair_strength,
        top_n_cases=args.top_n_cases,
        require_split=args.require_split,
    )

    print("Writing outputs...")
    extremes.to_csv(outdir / "dimension_judge_extremes.csv", index=False)
    case_df.to_csv(outdir / "dimension_exemplar_cases.csv", index=False)
    pair_df.to_csv(outdir / "dimension_pair_exemplar_cases.csv", index=False)

    write_dimension_text_reports(outdir, extremes, case_df)
    write_main_report(
        out_path=outdir / "mds_dimension_exemplar_report.txt",
        coords=coords,
        extremes=extremes,
        case_df=case_df,
        pair_df=pair_df,
        missing_df=missing_df,
        args=args,
    )

    print("")
    print(f"Saved MDS exemplar outputs to: {outdir}")
    print("Key files:")
    print(f"  {outdir / 'mds_dimension_exemplar_report.txt'}")
    print(f"  {outdir / 'dimension_judge_extremes.csv'}")
    print(f"  {outdir / 'dimension_exemplar_cases.csv'}")
    print(f"  {outdir / 'dimension_pair_exemplar_cases.csv'}")
    print(f"  {outdir / 'dimension_x_exemplars.txt'}")
    print(f"  {outdir / 'dimension_y_exemplars.txt'}")
    print(f"  {outdir / 'dimension_z_exemplars.txt'}")


if __name__ == "__main__":
    main()
