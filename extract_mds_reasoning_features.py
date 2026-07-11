#!/usr/bin/env python3
"""
extract_mds_reasoning_features.py
=================================

Extract judge-specific reasoning evidence for interpreting MDS dimensions.

Core question
-------------
Instead of asking:

    "What does Dimension X represent?"

this script asks:

    "What reasoning features consistently predict movement along Dimension X?"

It keeps the useful part of the earlier exemplar-case workflow:
  1. Find judges at the high and low ends of each MDS dimension.
  2. Find cases where high-end and low-end judges sat together.
  3. Score cases where high-end and low-end judges split into different opinion groups.

But it changes the evidence collected for LLM/human review:
  4. Extract judge/opinion-specific reasons from the full case text when possible.
  5. Build side-by-side "reasoning packets" for each dimension.
  6. Write LLM-ready prompts asking for recurring reasoning features, confidence,
     counterexamples, and passages supporting each proposed feature.

Recommended run
---------------
python extract_mds_reasoning_features.py \
  --coords mds_3d_outputs/mds_3d_coordinates.csv \
  --input-dir opinion_group_outputs_v6 \
  --jsonl scc_cases_2016-2026.jsonl \
  --outdir mds_reasoning_feature_outputs \
  --top-n-judges 3 \
  --top-n-cases 30 \
  --max-reasons-chars 9000

Expected inputs
---------------
- mds_3d_outputs/mds_3d_coordinates.csv
    columns: judge, x, y, z
- opinion_group_outputs_v6/judge_case_opinions.csv
    columns: case_id, judge, opinion_group, plus optional opinion_type/outcome_vote/etc.
- opinion_group_outputs_v6/judge_pair_case.csv
    columns: case_id, judge_a, judge_b, same_opinion_group
- opinion_group_outputs_v6/case_level.csv
    columns: case_id, case_name, year, subjects, etc.
- scc_cases_2016-2026.jsonl
    should contain citation_en/citation2_en/case_id and full-text fields such as
    unofficial_text_en or text_en.

Outputs
-------
mds_reasoning_feature_outputs/
  dimension_judge_extremes.csv
  dimension_reasoning_cases.csv
  dimension_pair_reasoning_cases.csv
  reasoning_extract_quality.csv
  dimension_x_reasoning_packet.txt
  dimension_y_reasoning_packet.txt
  dimension_z_reasoning_packet.txt
  dimension_x_llm_prompt.txt
  dimension_y_llm_prompt.txt
  dimension_z_llm_prompt.txt
  mds_reasoning_feature_report.txt

Important interpretation note
-----------------------------
This script does not prove that a dimension "means" a single thing. It gives you
case-level reasoning evidence. The right methodological claim is:

    MDS identified latent dimensions in judge agreement patterns.
    This script identifies cases and judge-specific reasons that most strongly
    distinguish the high and low ends of each dimension.
    Human/LLM qualitative coding then asks which reasoning features consistently
    predict movement along that dimension.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(s: Any) -> str:
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


def compact_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(s)).strip()


def last_name(full: Any) -> str:
    return str(full).split(",", 1)[0].strip()


def pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.1%}"
    except Exception:
        return "n/a"


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_year_from_date(date_s: Any) -> Any:
    s = str(date_s or "")
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return ""


def find_default(path_options: list[str]) -> Optional[Path]:
    for p in path_options:
        path = Path(p)
        if path.exists():
            return path
    return None


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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
                obj.get("case_id"),
                obj.get("url_en"),
            ]
            for key in keys:
                if key:
                    out[str(key)] = obj
    return out


def get_json_record(case_id: str, json_cases: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    if case_id in json_cases:
        return json_cases[case_id]

    cid_norm = compact_ws(case_id).lower()
    for key, obj in json_cases.items():
        possible = [
            key,
            obj.get("citation_en", ""),
            obj.get("citation2_en", ""),
            obj.get("case_id", ""),
            obj.get("url_en", ""),
        ]
        if any(compact_ws(p).lower() == cid_norm for p in possible if p):
            return obj
    return None


def full_text_from_json(obj: Optional[dict[str, Any]]) -> str:
    if not obj:
        return ""
    for field in [
        "unofficial_text_en",
        "text_en",
        "document_text_en",
        "full_text_en",
        "body_en",
        "html_en",
        "text",
    ]:
        val = obj.get(field)
        if val and len(str(val)) > 500:
            return normalize_text(val)
    return ""


def build_case_metadata(case_level: pd.DataFrame, json_cases: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
                "panel_size": row.get("panel_size", ""),
                "num_opinion_groups": row.get("num_opinion_groups", ""),
                "has_dissent": row.get("has_dissent", ""),
                "has_concurrence": row.get("has_concurrence", ""),
            }

    for key, obj in json_cases.items():
        citation = obj.get("citation_en") or obj.get("citation2_en") or obj.get("case_id") or key
        citation = str(citation)
        if citation not in meta:
            meta[citation] = {
                "case_id": citation,
                "case_name": obj.get("name_en", ""),
                "year": parse_year_from_date(obj.get("document_date_en")),
                "subjects": "",
                "appeal_from": "",
                "case_outcome": "",
                "panel_size": "",
                "num_opinion_groups": "",
                "has_dissent": "",
                "has_concurrence": "",
            }
    return meta


# ---------------------------------------------------------------------------
# Reason extraction
# ---------------------------------------------------------------------------

@dataclass
class ReasonBlock:
    heading: str
    text: str
    start: int
    end: int
    judges_detected: list[str]


def judge_regex_name(judge_short: str) -> str:
    # SCC text may contain O'Bonsawin, O’Bonsawin, OBonsawin, etc.
    name = re.escape(judge_short)
    name = name.replace("\\'", "['’ʼ]")
    return name


def extract_intro_summary(text: str, max_chars: int = 2500) -> str:
    """Return a short neutral case setup, mostly Coram/Held/headings."""
    text = normalize_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    useful = []
    triggers = (
        "Coram:", "Held:", "Reasons for Judgment", "Reasons for the Judgment",
        "Joint Reasons", "Majority Reasons", "Concurring Reasons", "Dissenting Reasons",
        "Reasons Concurring", "Reasons Dissenting", "Unanimous Reasons", "Unanimous Judgment",
        "Per ",
    )
    for line in lines[:300]:
        if line.startswith(triggers):
            useful.append(line)
    if useful:
        return "\n".join(useful)[:max_chars]
    return compact_ws("\n".join(lines[:80]))[:max_chars]


def detect_judges_in_heading(heading: str, judge_shorts: list[str]) -> list[str]:
    h = normalize_text(heading)
    found = []
    for j in judge_shorts:
        pattern = rf"\b{judge_regex_name(j)}\b"
        if re.search(pattern, h, flags=re.IGNORECASE):
            found.append(j)
    return sorted(set(found))


def find_reason_blocks(text: str, judge_shorts: list[str]) -> list[ReasonBlock]:
    """
    Find likely SCC reasons blocks.

    This deliberately uses broad patterns because CanLII/SCC formatting varies.
    It catches headings such as:
      Per Karakatsanis, Martin and Moreau JJ.:
      Per Rowe J.:
      The reasons of Brown and Rowe JJ. were delivered by
      Reasons for Judgment of Wagner C.J. and Karakatsanis J.
    """
    text = normalize_text(text)
    if not text.strip():
        return []

    escaped_names = "|".join(judge_regex_name(j) for j in sorted(judge_shorts, key=len, reverse=True))
    if not escaped_names:
        return []

    # Headings usually appear at line beginnings. Keep regex multiline.
    heading_patterns = [
        rf"^\s*Per\s+[^\n:]*?(?:{escaped_names})[^\n:]*?:",
        rf"^\s*(?:The\s+)?Reasons?\s+(?:for\s+Judgment\s+)?(?:of|by)\s+[^\n]*?(?:{escaped_names})[^\n]*",
        rf"^\s*(?:Joint\s+)?(?:Concurring|Dissenting|Majority|Minority)\s+Reasons?\s+(?:of|by)?\s*[^\n]*?(?:{escaped_names})[^\n]*",
    ]
    combined = re.compile("|".join(f"({p})" for p in heading_patterns), re.IGNORECASE | re.MULTILINE)

    matches = list(combined.finditer(text))
    if not matches:
        return []

    blocks: list[ReasonBlock] = []
    for i, m in enumerate(matches):
        start = m.start()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading_end = text.find("\n", start, min(next_start, start + 500))
        if heading_end == -1:
            heading_end = min(next_start, start + 500)
        heading = compact_ws(text[start:heading_end])
        block_text = text[start:next_start].strip()
        judges = detect_judges_in_heading(heading, judge_shorts)
        blocks.append(ReasonBlock(heading=heading, text=block_text, start=start, end=next_start, judges_detected=judges))
    return blocks


def fallback_extract_near_judge_mentions(text: str, judge_short: str, max_chars: int) -> tuple[str, str]:
    """Fallback if no structured heading was found."""
    text = normalize_text(text)
    pattern = re.compile(rf"\b{judge_regex_name(judge_short)}\b", flags=re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return "not_found", ""
    start = max(0, m.start() - max_chars // 3)
    end = min(len(text), m.start() + max_chars)
    return "fallback_near_judge_mention", compact_ws(text[start:end])


def extract_reasoning_for_judge(text: str,
                                judge_short: str,
                                all_judge_shorts: list[str],
                                max_chars: int) -> tuple[str, str, str]:
    """
    Return extraction_quality, heading, extracted_text.

    Priority:
      1. A reason block whose heading names the judge.
      2. A fallback snippet near a judge-name mention.
    """
    blocks = find_reason_blocks(text, all_judge_shorts)
    candidates = [b for b in blocks if judge_short in b.judges_detected]
    if candidates:
        # Prefer the longest block because it usually contains the real reasons, not only a summary line.
        best = max(candidates, key=lambda b: len(b.text))
        return "reason_block", best.heading, best.text[:max_chars]

    quality, snippet = fallback_extract_near_judge_mentions(text, judge_short, max_chars)
    return quality, "", snippet[:max_chars]


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
    a = df["judge_a_short"].astype(str)
    b = df["judge_b_short"].astype(str)
    df["pair_short"] = np.where(a < b, a + " - " + b, b + " - " + a)
    df["same_opinion_group_bool"] = (
        df["same_opinion_group"]
        .astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    return df


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


def build_pair_strength_lookup(pair_rows_used: Optional[pd.DataFrame]) -> dict[str, dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Case scoring and reasoning extraction
# ---------------------------------------------------------------------------

def score_and_extract_reasoning(coords: pd.DataFrame,
                                extremes: pd.DataFrame,
                                judge_cases: pd.DataFrame,
                                pair_cases: pd.DataFrame,
                                case_meta: dict[str, dict[str, Any]],
                                json_cases: dict[str, dict[str, Any]],
                                pair_strength: dict[str, dict[str, Any]],
                                top_n_cases: int,
                                require_split: bool,
                                max_reasons_chars: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coord_lookup = coords.set_index("judge_short")[["x", "y", "z"]].to_dict("index")
    all_judge_shorts = sorted(judge_cases["judge_short"].dropna().astype(str).unique().tolist())

    case_to_judges = {str(case_id): g.copy() for case_id, g in judge_cases.groupby("case_id")}
    pair_cases_by_case = {str(case_id): g.copy() for case_id, g in pair_cases.groupby("case_id")}

    case_rows = []
    pair_rows = []
    reason_rows = []

    # First pass: score all candidate cases.
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

            cross_known = 0
            cross_split = 0
            cross_same = 0
            cross_pair_details = []
            pair_rows_for_case = pair_cases_by_case.get(case_id, pd.DataFrame())

            for a, b in product(high_present, low_present):
                pair_key = " - ".join(sorted([a, b]))
                matched = pd.DataFrame()
                if not pair_rows_for_case.empty:
                    matched = pair_rows_for_case[pair_rows_for_case["pair_short"].eq(pair_key)]
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
                strength = pair_strength.get(pair_key, {})
                dim_gap = abs(safe_float(coord_lookup.get(a, {}).get(dim)) - safe_float(coord_lookup.get(b, {}).get(dim)))
                cross_pair_details.append({
                    "dimension": dim,
                    "case_id": case_id,
                    "high_judge": a,
                    "low_judge": b,
                    "pair": pair_key,
                    "same_opinion_group": bool(same),
                    "dimension_coordinate_gap": dim_gap,
                    "pair_z_score": strength.get("pair_z_score", np.nan),
                    "pair_observed_agreement_rate": strength.get("pair_observed_agreement_rate", np.nan),
                    "pair_expected_agreement_rate": strength.get("pair_expected_agreement_rate", np.nan),
                    "pair_cases_together": strength.get("pair_cases_together", np.nan),
                    "pair_p_two_sided": strength.get("pair_p_two_sided", np.nan),
                })

            if cross_known == 0:
                continue
            if require_split and cross_split == 0:
                continue

            relevant = g[g["judge_short"].isin(high_present + low_present)].copy()
            relevant["dimension_side"] = np.where(relevant["judge_short"].isin(high_set), "high", "low")
            relevant_summary = []
            for _, rr in relevant.sort_values(["dimension_side", "judge_short"]).iterrows():
                relevant_summary.append(
                    f"{rr['judge_short']}[{rr['dimension_side']}]: "
                    f"group={rr.get('opinion_group', '')}; "
                    f"type={rr.get('opinion_type', '')}; "
                    f"outcome={rr.get('outcome_vote', '')}; "
                    f"side={rr.get('disposition_side', '')}"
                )

            split_rate = cross_split / cross_known if cross_known else np.nan
            present_extreme_count = len(high_present) + len(low_present)
            pair_z_values = [safe_float(d.get("pair_z_score")) for d in cross_pair_details]
            pair_z_values = [z for z in pair_z_values if pd.notna(z)]
            mean_pair_z = float(np.mean(pair_z_values)) if pair_z_values else np.nan
            min_pair_z = float(np.min(pair_z_values)) if pair_z_values else np.nan
            case_score = (
                3.0 * cross_split
                + 2.0 * split_rate
                + 0.5 * present_extreme_count
                + (abs(min_pair_z) if pd.notna(min_pair_z) and min_pair_z < 0 else 0.0)
            )

            meta = case_meta.get(case_id, {"case_id": case_id})
            case_rows.append({
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
            })

            for d in cross_pair_details:
                if require_split and d["same_opinion_group"]:
                    continue
                d.update({
                    "case_name": meta.get("case_name", ""),
                    "year": meta.get("year", ""),
                    "subjects": meta.get("subjects", ""),
                    "appeal_from": meta.get("appeal_from", ""),
                    "case_outcome": meta.get("case_outcome", ""),
                    "num_opinion_groups": meta.get("num_opinion_groups", ""),
                })
                pair_rows.append(d)

    case_df = pd.DataFrame(case_rows)
    pair_df = pd.DataFrame(pair_rows)

    if not case_df.empty:
        case_df = case_df.sort_values(["dimension", "case_score", "cross_side_split_pairs"], ascending=[True, False, False])
        case_df = case_df.groupby("dimension", group_keys=False).head(top_n_cases).copy()

    if not pair_df.empty:
        pair_df["sort_pair_z"] = pd.to_numeric(pair_df["pair_z_score"], errors="coerce")
        pair_df["sort_pair_cases"] = pd.to_numeric(pair_df["pair_cases_together"], errors="coerce")
        pair_df = pair_df.sort_values(
            ["dimension", "same_opinion_group", "dimension_coordinate_gap", "sort_pair_z", "sort_pair_cases"],
            ascending=[True, True, False, True, False],
        ).drop(columns=["sort_pair_z", "sort_pair_cases"])

    # Second pass: extract judge-specific reasoning only for retained top cases.
    if case_df.empty:
        return case_df, pair_df, pd.DataFrame(reason_rows)

    retained_case_ids = set(case_df["case_id"].astype(str))
    for _, case_row in case_df.iterrows():
        dim = case_row["dimension"]
        case_id = str(case_row["case_id"])
        if case_id not in retained_case_ids:
            continue

        json_obj = get_json_record(case_id, json_cases)
        text = full_text_from_json(json_obj)
        intro = extract_intro_summary(text)
        g = case_to_judges.get(case_id, pd.DataFrame())
        if g.empty:
            continue

        high_present = [j.strip() for j in str(case_row.get("high_extreme_judges_present", "")).split(";") if j.strip()]
        low_present = [j.strip() for j in str(case_row.get("low_extreme_judges_present", "")).split(";") if j.strip()]

        for side, judges in [("high", high_present), ("low", low_present)]:
            for judge_short in judges:
                rr = g[g["judge_short"].eq(judge_short)]
                rr0 = rr.iloc[0] if not rr.empty else {}
                quality, heading, reasons = extract_reasoning_for_judge(
                    text=text,
                    judge_short=judge_short,
                    all_judge_shorts=all_judge_shorts,
                    max_chars=max_reasons_chars,
                )
                reason_rows.append({
                    "dimension": dim,
                    "side": side,
                    "judge_short": judge_short,
                    "judge_full": rr0.get("judge", "") if len(rr) else "",
                    "coordinate": coord_lookup.get(judge_short, {}).get(dim, np.nan),
                    "case_id": case_id,
                    "case_name": case_row.get("case_name", ""),
                    "year": case_row.get("year", ""),
                    "subjects": case_row.get("subjects", ""),
                    "appeal_from": case_row.get("appeal_from", ""),
                    "case_outcome": case_row.get("case_outcome", ""),
                    "opinion_group": rr0.get("opinion_group", "") if len(rr) else "",
                    "opinion_type": rr0.get("opinion_type", "") if len(rr) else "",
                    "outcome_vote": rr0.get("outcome_vote", "") if len(rr) else "",
                    "disposition_side": rr0.get("disposition_side", "") if len(rr) else "",
                    "extraction_quality": quality,
                    "reason_heading": heading,
                    "case_intro": intro,
                    "judge_reasons_excerpt": reasons,
                    "judge_reasons_chars": len(reasons or ""),
                    "has_full_text": bool(text.strip()),
                })

    reason_df = pd.DataFrame(reason_rows)
    return case_df, pair_df, reason_df


# ---------------------------------------------------------------------------
# Text reports and LLM prompts
# ---------------------------------------------------------------------------

def write_reasoning_packets(outdir: Path,
                            extremes: pd.DataFrame,
                            case_df: pd.DataFrame,
                            reason_df: pd.DataFrame,
                            max_cases_in_prompt: int) -> None:
    for dim in ["x", "y", "z"]:
        high = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "high")].sort_values("rank")
        low = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "low")].sort_values("rank")
        cases = case_df[case_df["dimension"] == dim].copy() if not case_df.empty else pd.DataFrame()
        reasons = reason_df[reason_df["dimension"] == dim].copy() if not reason_df.empty else pd.DataFrame()

        lines = []
        lines.append(f"MDS DIMENSION {dim.upper()} REASONING PACKET")
        lines.append("=" * 100)
        lines.append("")
        lines.append("Research question:")
        lines.append(f"  What reasoning features consistently predict movement along Dimension {dim.upper()}?")
        lines.append("")
        lines.append("High-end judges:")
        for _, r in high.iterrows():
            lines.append(f"  {r['rank']}. {r['judge_short']} ({dim}={r['coordinate']:.4f})")
        lines.append("")
        lines.append("Low-end judges:")
        for _, r in low.iterrows():
            lines.append(f"  {r['rank']}. {r['judge_short']} ({dim}={r['coordinate']:.4f})")
        lines.append("")
        lines.append("How to read this packet:")
        lines.append("  Compare high-end reasoning against low-end reasoning case by case.")
        lines.append("  Ignore mere outcomes unless the outcome reflects a recurring method of reasoning.")
        lines.append("  Look for repeated features: text/purpose, remedial posture, institutional role,")
        lines.append("  precedent use, administrability, deference, rule/standard preference, policy reasoning,")
        lines.append("  fact sensitivity, Charter methodology, statutory context, etc.")
        lines.append("")

        if cases.empty:
            lines.append("No retained cases for this dimension.")
        else:
            for rank, (_, c) in enumerate(cases.iterrows(), start=1):
                case_id = str(c["case_id"])
                case_reasons = reasons[reasons["case_id"].astype(str).eq(case_id)].copy()
                lines.append("-" * 100)
                lines.append(f"CASE {rank}: {c.get('case_name', '')} ({c.get('year', '')})")
                lines.append(f"Case ID: {case_id}")
                lines.append(f"Subjects: {c.get('subjects', '')}")
                lines.append(f"Appeal from: {c.get('appeal_from', '')}")
                lines.append(f"Outcome: {c.get('case_outcome', '')}")
                lines.append(
                    f"Cross-side split pairs: {c.get('cross_side_split_pairs', '')}/"
                    f"{c.get('cross_side_known_pairs', '')} ({pct(c.get('cross_side_split_rate', np.nan))})"
                )
                lines.append(f"Opinion groups: {c.get('relevant_judge_opinion_groups', '')}")
                lines.append("")
                if not case_reasons.empty:
                    intro = str(case_reasons.iloc[0].get("case_intro", "") or "")
                    if intro:
                        lines.append("Case setup / official summary excerpt:")
                        lines.append(indent_block(intro, "  "))
                        lines.append("")

                    for side in ["high", "low"]:
                        lines.append(f"{side.upper()}-END REASONING:")
                        side_reasons = case_reasons[case_reasons["side"].eq(side)].sort_values("judge_short")
                        if side_reasons.empty:
                            lines.append("  n/a")
                        else:
                            for _, rr in side_reasons.iterrows():
                                lines.append(
                                    f"  Judge {rr.get('judge_short', '')} "
                                    f"({dim}={safe_float(rr.get('coordinate')):.4f}; "
                                    f"group={rr.get('opinion_group', '')}; "
                                    f"type={rr.get('opinion_type', '')}; "
                                    f"outcome={rr.get('outcome_vote', '')}; "
                                    f"extraction={rr.get('extraction_quality', '')})"
                                )
                                heading = str(rr.get("reason_heading", "") or "")
                                if heading:
                                    lines.append(f"  Heading: {heading}")
                                excerpt = str(rr.get("judge_reasons_excerpt", "") or "")
                                if excerpt:
                                    lines.append(indent_block(excerpt, "    "))
                                else:
                                    lines.append("    [No judge-specific reasoning text found.]")
                                lines.append("")
                        lines.append("")
                else:
                    lines.append("No judge-specific reasoning extraction rows found for this case.")
                    lines.append("")

        (outdir / f"dimension_{dim}_reasoning_packet.txt").write_text("\n".join(lines), encoding="utf-8")

        prompt_lines = build_llm_prompt_for_dimension(dim, high, low, cases.head(max_cases_in_prompt), reasons)
        (outdir / f"dimension_{dim}_llm_prompt.txt").write_text("\n".join(prompt_lines), encoding="utf-8")


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def build_llm_prompt_for_dimension(dim: str,
                                   high: pd.DataFrame,
                                   low: pd.DataFrame,
                                   cases: pd.DataFrame,
                                   reasons: pd.DataFrame) -> list[str]:
    lines = []
    lines.append(f"You are helping interpret Dimension {dim.upper()} from an MDS analysis of Supreme Court of Canada judge agreement patterns.")
    lines.append("")
    lines.append("Do NOT answer: 'What does this dimension represent?' as if it has one essence.")
    lines.append(f"Answer instead: 'What reasoning features consistently predict movement along Dimension {dim.upper()}?' ")
    lines.append("")
    lines.append("High-end judges:")
    lines.append("  " + ", ".join([f"{r['judge_short']} ({dim}={r['coordinate']:.3f})" for _, r in high.iterrows()]))
    lines.append("Low-end judges:")
    lines.append("  " + ", ".join([f"{r['judge_short']} ({dim}={r['coordinate']:.3f})" for _, r in low.iterrows()]))
    lines.append("")
    lines.append("Instructions:")
    lines.append("1. Compare high-end and low-end reasoning within each case before generalizing across cases.")
    lines.append("2. Focus on legal reasoning methods, not just winners/losers or subject areas.")
    lines.append("3. Identify 3-8 recurring reasoning features that appear to move a judge high or low on this dimension.")
    lines.append("4. For each feature, give: direction, confidence, supporting cases, counterexamples, and short textual support.")
    lines.append("5. Separate strong recurring features from weak/speculative ones.")
    lines.append("6. Watch for topic confounding: a dimension may look like 'criminal law' or 'Charter' merely because many exemplar cases are in that area.")
    lines.append("7. End with a cautious label for the dimension only after the feature analysis.")
    lines.append("")
    lines.append("Desired output format:")
    lines.append("- Candidate feature table: feature | high-end tendency | low-end tendency | confidence | cases supporting | counterexamples")
    lines.append("- Narrative synthesis, 500-1000 words")
    lines.append("- Best cautious label for the dimension")
    lines.append("- What evidence would change your mind")
    lines.append("")
    lines.append("Evidence:")
    lines.append("=" * 100)

    for rank, (_, c) in enumerate(cases.iterrows(), start=1):
        case_id = str(c["case_id"])
        case_reasons = reasons[reasons["case_id"].astype(str).eq(case_id)].copy()
        lines.append("")
        lines.append(f"CASE {rank}: {c.get('case_name', '')} ({c.get('year', '')}) [{case_id}]")
        lines.append(f"Subjects: {c.get('subjects', '')}")
        lines.append(f"Split: {c.get('cross_side_split_pairs', '')}/{c.get('cross_side_known_pairs', '')} cross-side pairs")
        lines.append(f"Opinion groups: {c.get('relevant_judge_opinion_groups', '')}")
        lines.append("")
        for side in ["high", "low"]:
            lines.append(f"{side.upper()} END:")
            side_reasons = case_reasons[case_reasons["side"].eq(side)].sort_values("judge_short")
            if side_reasons.empty:
                lines.append("  [No extracted reasons]")
            for _, rr in side_reasons.iterrows():
                lines.append(f"  Judge {rr.get('judge_short', '')}; extraction={rr.get('extraction_quality', '')}; heading={rr.get('reason_heading', '')}")
                excerpt = compact_ws(rr.get("judge_reasons_excerpt", ""))
                lines.append(f"  EXCERPT: {excerpt[:4500]}")
                lines.append("")
    return lines


def write_main_report(out_path: Path,
                      args: argparse.Namespace,
                      extremes: pd.DataFrame,
                      case_df: pd.DataFrame,
                      reason_df: pd.DataFrame,
                      pair_df: pd.DataFrame,
                      missing_df: Optional[pd.DataFrame]) -> None:
    lines = []
    lines.append("MDS REASONING FEATURE EXTRACTION REPORT")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Coordinates: {args.coords}")
    lines.append(f"Input directory: {args.input_dir}")
    lines.append(f"JSONL: {args.jsonl}")
    lines.append(f"Top/bottom judges per dimension: {args.top_n_judges}")
    lines.append(f"Top cases per dimension: {args.top_n_cases}")
    lines.append(f"Max chars per judge reasons excerpt: {args.max_reasons_chars}")
    lines.append(f"Require cross-side split: {args.require_split}")
    lines.append("")

    if missing_df is not None and not missing_df.empty:
        lines.append("Caution:")
        lines.append(f"  The MDS run had {len(missing_df)} missing judge-pair distances filled.")
        lines.append("  Treat dimensions involving those relationships cautiously.")
        lines.append("")

    lines.append("Dimension extremes and top case leads:")
    for dim in ["x", "y", "z"]:
        lines.append("")
        lines.append(f"Dimension {dim.upper()}:")
        high = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "high")].sort_values("rank")
        low = extremes[(extremes["dimension"] == dim) & (extremes["side"] == "low")].sort_values("rank")
        lines.append("  High end: " + ", ".join([f"{r['judge_short']} ({r['coordinate']:.2f})" for _, r in high.iterrows()]))
        lines.append("  Low end: " + ", ".join([f"{r['judge_short']} ({r['coordinate']:.2f})" for _, r in low.iterrows()]))
        subset = case_df[case_df["dimension"].eq(dim)].head(8) if not case_df.empty else pd.DataFrame()
        if subset.empty:
            lines.append("  No top cases retained.")
        else:
            for _, r in subset.iterrows():
                lines.append(
                    f"  - {r.get('case_name', '')} ({r.get('year', '')}): "
                    f"{r.get('cross_side_split_pairs', '')}/{r.get('cross_side_known_pairs', '')} split; "
                    f"subjects={r.get('subjects', '')}"
                )

    lines.append("")
    lines.append("Reason extraction quality:")
    if reason_df.empty:
        lines.append("  No reasoning rows extracted.")
    else:
        q = reason_df.groupby(["dimension", "extraction_quality"]).size().reset_index(name="rows")
        for _, r in q.iterrows():
            lines.append(f"  Dimension {str(r['dimension']).upper()}: {r['extraction_quality']} = {r['rows']}")

    lines.append("")
    lines.append("Use the *_llm_prompt.txt files to ask an LLM for feature coding.")
    lines.append("The methodological framing should be: recurring reasoning features associated with dimension movement, not causal proof.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract judge-specific reasoning evidence for interpreting MDS dimensions.")
    parser.add_argument("--coords", default=None, help="Path to mds_3d_coordinates.csv")
    parser.add_argument("--input-dir", default="opinion_group_outputs_v6",
                        help="Folder containing judge_case_opinions.csv, judge_pair_case.csv, case_level.csv")
    parser.add_argument("--jsonl", default="scc_cases_2016-2026.jsonl", help="Path to raw SCC JSONL file")
    parser.add_argument("--pair-rows-used", default=None,
                        help="Optional path to mds_3d_pair_rows_used.csv or permutation_pair_results.csv")
    parser.add_argument("--missing-distances", default=None, help="Optional path to mds_3d_missing_distances.csv")
    parser.add_argument("--outdir", default="mds_reasoning_feature_outputs", help="Output directory")
    parser.add_argument("--top-n-judges", type=int, default=3,
                        help="Number of high-end and low-end judges to use per dimension")
    parser.add_argument("--top-n-cases", type=int, default=30,
                        help="Number of exemplar cases to keep per dimension")
    parser.add_argument("--max-reasons-chars", type=int, default=9000,
                        help="Max characters of judge-specific reasons to include per judge/case")
    parser.add_argument("--max-cases-in-prompt", type=int, default=12,
                        help="Number of cases to include inside each LLM prompt file. Full packets still include all top cases.")
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
    args.coords = str(coords_path)

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

    print("Building case metadata...")
    case_meta = build_case_metadata(case_level, json_cases)

    print("Loading optional pair-strength rows...")
    pair_strength_df = pd.read_csv(pair_rows_path) if pair_rows_path and pair_rows_path.exists() else pd.DataFrame()
    pair_strength = build_pair_strength_lookup(pair_strength_df)
    missing_df = pd.read_csv(missing_path) if missing_path and missing_path.exists() else pd.DataFrame()

    print("Finding dimension extremes...")
    extremes = dimension_extremes(coords, args.top_n_judges)

    print("Scoring cases and extracting judge-specific reasoning...")
    case_df, pair_df, reason_df = score_and_extract_reasoning(
        coords=coords,
        extremes=extremes,
        judge_cases=judge_cases,
        pair_cases=pair_cases,
        case_meta=case_meta,
        json_cases=json_cases,
        pair_strength=pair_strength,
        top_n_cases=args.top_n_cases,
        require_split=args.require_split,
        max_reasons_chars=args.max_reasons_chars,
    )

    print("Writing outputs...")
    extremes.to_csv(outdir / "dimension_judge_extremes.csv", index=False)
    case_df.to_csv(outdir / "dimension_reasoning_cases.csv", index=False)
    pair_df.to_csv(outdir / "dimension_pair_reasoning_cases.csv", index=False)
    reason_df.to_csv(outdir / "dimension_judge_reasoning_extracts.csv", index=False)

    if not reason_df.empty:
        quality = reason_df.groupby(["dimension", "extraction_quality"]).size().reset_index(name="rows")
        quality.to_csv(outdir / "reasoning_extract_quality.csv", index=False)
    else:
        pd.DataFrame(columns=["dimension", "extraction_quality", "rows"]).to_csv(outdir / "reasoning_extract_quality.csv", index=False)

    write_reasoning_packets(
        outdir=outdir,
        extremes=extremes,
        case_df=case_df,
        reason_df=reason_df,
        max_cases_in_prompt=args.max_cases_in_prompt,
    )

    write_main_report(
        out_path=outdir / "mds_reasoning_feature_report.txt",
        args=args,
        extremes=extremes,
        case_df=case_df,
        reason_df=reason_df,
        pair_df=pair_df,
        missing_df=missing_df,
    )

    print("")
    print(f"Saved MDS reasoning-feature outputs to: {outdir}")
    print("Key files:")
    print(f"  {outdir / 'mds_reasoning_feature_report.txt'}")
    print(f"  {outdir / 'dimension_reasoning_cases.csv'}")
    print(f"  {outdir / 'dimension_judge_reasoning_extracts.csv'}")
    print(f"  {outdir / 'dimension_x_reasoning_packet.txt'}")
    print(f"  {outdir / 'dimension_y_reasoning_packet.txt'}")
    print(f"  {outdir / 'dimension_z_reasoning_packet.txt'}")
    print(f"  {outdir / 'dimension_x_llm_prompt.txt'}")
    print(f"  {outdir / 'dimension_y_llm_prompt.txt'}")
    print(f"  {outdir / 'dimension_z_llm_prompt.txt'}")


if __name__ == "__main__":
    main()
