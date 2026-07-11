#!/usr/bin/env python3
"""
pairwise_predictor_analysis.py
==============================

Pairwise predictor analysis for Supreme Court of Canada agreement.

Question answered
-----------------
Do pairs of judges who share a characteristic agree more often than pairs of
judges who do not share that characteristic?

This is deliberately separate from the judge-level MDS predictor script. Here,
the unit of analysis is the judge pair.

Inputs
------
1. Metadata CSV, e.g. scc_justices_data.csv
   Required first column: Name
   Other columns are treated as predictors unless excluded.

2. Pair summary CSV, e.g. opinion_group_outputs_v6/judge_pair_summary.csv
   Required columns:
     judge_a, judge_b, outcome_agreement_rate, disposition_agreement_rate,
     opinion_group_agreement_rate, and their *_known_cases weight columns.

3. Optional pair-case CSV, e.g. opinion_group_outputs_v6/judge_pair_case.csv
   Used only for exemplar-case output.

Outputs
-------
- pairwise_joined_features.csv
- pairwise_predictor_effects.csv
- pairwise_negative_control_comparison.csv
- pairwise_exemplar_cases.csv
- pairwise_analysis_report.txt

Recommended run
---------------
python pairwise_predictor_analysis.py \
  --metadata scc_justices_data.csv \
  --pair-summary opinion_group_outputs_v6/judge_pair_summary.csv \
  --pair-case opinion_group_outputs_v6/judge_pair_case.csv \
  --outdir pairwise_predictor_analysis \
  --n-permutations 10000

Interpretation
--------------
For binary/categorical metadata predictors, the feature is:
  Same predictor value? 1/0

Example:
  Same Sex, Same Province, Same Law School

The main effect is:
  weighted mean agreement among same-value pairs
  minus
  weighted mean agreement among different-value pairs

For continuous metadata predictors, the feature is:
  absolute difference between the two judges

Example:
  abs(Years on Bench Before SCC A - Years on Bench Before SCC B)

The main effect is a weighted regression slope:
  agreement ~ absolute difference

A negative slope means judges closer on that continuous variable agree more.

Permutation design
------------------
For each predictor, the script shuffles judge-level labels/values across judges,
rebuilds pairwise features, and recalculates the effect. This preserves:
  - the actual agreement data
  - the actual pair structure
  - the predictor's distribution
but breaks the link between judges and that predictor.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


AGREEMENT_OUTCOMES = [
    ("opinion_group_agreement_rate", "opinion_group_known_cases", "Opinion Group Agreement"),
    ("disposition_agreement_rate", "disposition_known_cases", "Disposition-Side Agreement"),
    ("outcome_agreement_rate", "outcome_known_cases", "Outcome Agreement"),
]

DEFAULT_EXCLUDE_METADATA_COLUMNS = {
    "Name",
    "Judge",
    "judge",
    "judge_short",
}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonical_last_name(x: str) -> str:
    """Return a normalized last name usable for joining metadata to SCC names."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Pair files use "Kasirer, Nicholas". Metadata often uses "Nicholas Kasirer".
    if "," in s:
        last = s.split(",", 1)[0]
    else:
        parts = s.split()
        last = parts[-1] if parts else s
    last = (last.replace("’", "'")
                .replace("ʼ", "'")
                .replace("`", "'")
                .strip())
    return last.lower()


def short_name_from_pair_name(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if "," in s:
        return s.split(",", 1)[0].strip()
    return s.split()[-1] if s.split() else s


def clean_cell(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "unknown", "n/a", "na"}:
        return np.nan
    return s


def parse_bool_like(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}))


def is_numeric_predictor(s: pd.Series) -> bool:
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return False
    converted = pd.to_numeric(nonnull, errors="coerce")
    return converted.notna().mean() >= 0.95


def infer_predictor_type(s: pd.Series) -> str:
    cleaned = s.map(clean_cell)
    if is_numeric_predictor(cleaned):
        return "continuous"
    n_unique = cleaned.dropna().nunique()
    if n_unique == 2:
        return "binary"
    return "categorical"


def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(y[mask], weights=w[mask]))


def weighted_r2_binary_or_continuous(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    """Weighted R² from a one-predictor linear model y ~ 1 + x."""
    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() < 3 or len(np.unique(x[mask])) < 2:
        return np.nan
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    w = w[mask].astype(float)
    X = np.column_stack([np.ones(len(x)), x])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    yhat = X @ beta
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def weighted_slope(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() < 3 or len(np.unique(x[mask])) < 2:
        return np.nan
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    w = w[mask].astype(float)
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    denom = np.sum(w * (x - xbar) ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(w * (x - xbar) * (y - ybar)) / denom)


def empirical_p_two_sided(samples: np.ndarray, observed: float) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0 or not np.isfinite(observed):
        return np.nan
    center = np.mean(samples)
    return float((np.sum(np.abs(samples - center) >= abs(observed - center)) + 1) / (len(samples) + 1))


def empirical_p_directional(samples: np.ndarray, observed: float, expected_direction: str) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0 or not np.isfinite(observed):
        return np.nan
    if expected_direction == "high":
        return float((np.sum(samples >= observed) + 1) / (len(samples) + 1))
    if expected_direction == "low":
        return float((np.sum(samples <= observed) + 1) / (len(samples) + 1))
    return np.nan


# ---------------------------------------------------------------------------
# Metadata loading and negative controls
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "Name" not in meta.columns:
        raise ValueError("Metadata CSV must contain a 'Name' column.")

    meta = meta.copy()
    meta["judge_key"] = meta["Name"].map(canonical_last_name)
    meta["judge_short"] = meta["Name"].map(short_name_from_pair_name)

    if meta["judge_key"].duplicated().any():
        dupes = meta.loc[meta["judge_key"].duplicated(), "Name"].tolist()
        raise ValueError(f"Duplicate judge last-name keys in metadata: {dupes}")

    # Derived continuous variables if the raw years are available.
    for col in ["Year Appointed to SCC", "Year First Appointed to the Bench", "Year Called to the Bar", "Year of Birth"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    if {"Year Appointed to SCC", "Year of Birth"}.issubset(meta.columns):
        meta["Age at SCC Appointment"] = meta["Year Appointed to SCC"] - meta["Year of Birth"]

    if {"Year Appointed to SCC", "Year First Appointed to the Bench"}.issubset(meta.columns):
        meta["Years on Bench Before SCC"] = meta["Year Appointed to SCC"] - meta["Year First Appointed to the Bench"]

    if {"Year Appointed to SCC", "Year Called to the Bar"}.issubset(meta.columns):
        meta["Years from Bar Call to SCC"] = meta["Year Appointed to SCC"] - meta["Year Called to the Bar"]

    if {"Year First Appointed to the Bench", "Year Called to the Bar"}.issubset(meta.columns):
        meta["Years Practice Before First Bench Appointment"] = meta["Year First Appointed to the Bench"] - meta["Year Called to the Bar"]

    # Negative controls. These are judge-level variables that should not plausibly
    # explain legal agreement but match common data types.
    surname = meta["judge_short"].astype(str)
    meta["NEG_BINARY_Surname_A_to_M"] = surname.str[0].str.upper().between("A", "M").map({True: "A-M", False: "N-Z"})
    if "Year Appointed to SCC" in meta.columns:
        meta["NEG_BINARY_Even_SCC_Appointment_Year"] = (meta["Year Appointed to SCC"] % 2 == 0).map({True: "Even", False: "Odd"})
    meta["NEG_CATEGORICAL_Last_Name_Initial"] = surname.str[0].str.upper()
    if "Year of Birth" in meta.columns:
        meta["NEG_CATEGORICAL_Birth_Year_Mod_4"] = (meta["Year of Birth"] % 4).astype("Int64").astype(str).replace("<NA>", np.nan)
    meta["NEG_CONTINUOUS_Surname_Length"] = surname.str.replace(r"[^A-Za-zÀ-ÿ]", "", regex=True).str.len()
    meta["NEG_CONTINUOUS_Alphabetical_Surname_Rank"] = surname.rank(method="dense").astype(float)

    return meta


def predictor_columns(meta: pd.DataFrame, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        missing = [c for c in explicit if c not in meta.columns]
        if missing:
            raise ValueError(f"Requested predictors not in metadata: {missing}")
        return explicit
    return [c for c in meta.columns if c not in DEFAULT_EXCLUDE_METADATA_COLUMNS and c != "judge_key"]


# ---------------------------------------------------------------------------
# Pair feature construction
# ---------------------------------------------------------------------------

def load_pair_summary(path: Path, meta: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    required = {"judge_a", "judge_b"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pair summary missing required columns: {sorted(missing)}")

    pairs = pairs.copy()
    pairs["judge_a_key"] = pairs["judge_a"].map(canonical_last_name)
    pairs["judge_b_key"] = pairs["judge_b"].map(canonical_last_name)
    pairs["judge_a_short"] = pairs["judge_a"].map(short_name_from_pair_name)
    pairs["judge_b_short"] = pairs["judge_b"].map(short_name_from_pair_name)

    meta_keys = set(meta["judge_key"])
    missing_a = sorted(set(pairs["judge_a_key"]) - meta_keys)
    missing_b = sorted(set(pairs["judge_b_key"]) - meta_keys)
    missing_all = sorted(set(missing_a + missing_b))
    if missing_all:
        print("Warning: these pair-summary judges are missing from metadata and will produce missing features:")
        print(", ".join(missing_all))

    return pairs


def values_by_judge(meta: pd.DataFrame, predictor: str, ptype: str) -> Dict[str, object]:
    vals = meta.set_index("judge_key")[predictor]
    if ptype == "continuous":
        vals = pd.to_numeric(vals, errors="coerce")
    else:
        vals = vals.map(clean_cell)
    return vals.to_dict()


def pair_feature_from_values(pairs: pd.DataFrame, valmap: Dict[str, object], ptype: str) -> pd.Series:
    a = pairs["judge_a_key"].map(valmap)
    b = pairs["judge_b_key"].map(valmap)
    if ptype == "continuous":
        aa = pd.to_numeric(a, errors="coerce")
        bb = pd.to_numeric(b, errors="coerce")
        return (aa - bb).abs()
    else:
        return ((a.notna()) & (b.notna()) & (a.astype(str) == b.astype(str))).astype(float).where(a.notna() & b.notna(), np.nan)


def add_all_pair_features(pairs: pd.DataFrame, meta: pd.DataFrame, predictors: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = pairs.copy()
    types = {}
    for pred in predictors:
        ptype = infer_predictor_type(meta[pred])
        types[pred] = ptype
        valmap = values_by_judge(meta, pred, ptype)
        prefix = "DIFF" if ptype == "continuous" else "SAME"
        out[f"{prefix}_{pred}"] = pair_feature_from_values(out, valmap, ptype)
    return out, types


# ---------------------------------------------------------------------------
# Effect calculation and permutations
# ---------------------------------------------------------------------------

def calculate_effect_for_feature(df: pd.DataFrame, feature_col: str, ptype: str, y_col: str, w_col: str) -> Dict[str, float]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float) if w_col in df.columns else np.ones(len(df))
    x = pd.to_numeric(df[feature_col], errors="coerce").to_numpy(dtype=float)

    if ptype == "continuous":
        slope = weighted_slope(y, x, w)
        r2 = weighted_r2_binary_or_continuous(y, x, w)
        return {
            "effect": slope,
            "effect_kind": "weighted_slope_agreement_per_unit_difference",
            "weighted_r2": r2,
            "mean_same_or_lowdiff": np.nan,
            "mean_different_or_highdiff": np.nan,
            "n_same_or_lowdiff_pairs": np.nan,
            "n_different_or_highdiff_pairs": np.nan,
        }

    mask_same = np.isfinite(x) & (x == 1) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    mask_diff = np.isfinite(x) & (x == 0) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    mean_same = weighted_mean(y[mask_same], w[mask_same])
    mean_diff = weighted_mean(y[mask_diff], w[mask_diff])
    effect = mean_same - mean_diff if np.isfinite(mean_same) and np.isfinite(mean_diff) else np.nan
    r2 = weighted_r2_binary_or_continuous(y, x, w)
    return {
        "effect": effect,
        "effect_kind": "weighted_mean_same_minus_different",
        "weighted_r2": r2,
        "mean_same_or_lowdiff": mean_same,
        "mean_different_or_highdiff": mean_diff,
        "n_same_or_lowdiff_pairs": int(mask_same.sum()),
        "n_different_or_highdiff_pairs": int(mask_diff.sum()),
    }


def permutation_effects(
    pairs: pd.DataFrame,
    meta: pd.DataFrame,
    predictor: str,
    ptype: str,
    y_col: str,
    w_col: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of permuted effects and permuted weighted R² values."""
    judge_keys = meta["judge_key"].tolist()
    original = meta[predictor].copy()
    if ptype == "continuous":
        original = pd.to_numeric(original, errors="coerce")
    else:
        original = original.map(clean_cell)

    vals = original.to_numpy(dtype=object)
    effects = np.full(n_permutations, np.nan, dtype=float)
    r2s = np.full(n_permutations, np.nan, dtype=float)

    temp = pairs[["judge_a_key", "judge_b_key", y_col, w_col]].copy() if w_col in pairs.columns else pairs[["judge_a_key", "judge_b_key", y_col]].copy()

    for i in range(n_permutations):
        shuffled = rng.permutation(vals)
        valmap = dict(zip(judge_keys, shuffled))
        temp_feature = pair_feature_from_values(temp, valmap, ptype)
        temp["__feature__"] = temp_feature
        res = calculate_effect_for_feature(temp, "__feature__", ptype, y_col, w_col)
        effects[i] = res["effect"]
        r2s[i] = res["weighted_r2"]

    return effects, r2s


def run_effects(
    pairs: pd.DataFrame,
    meta: pd.DataFrame,
    predictors: List[str],
    ptypes: Dict[str, str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for y_col, w_col, outcome_label in AGREEMENT_OUTCOMES:
        if y_col not in pairs.columns:
            continue
        if w_col not in pairs.columns:
            pairs[w_col] = 1.0

        for pred in predictors:
            ptype = ptypes[pred]
            prefix = "DIFF" if ptype == "continuous" else "SAME"
            feature_col = f"{prefix}_{pred}"
            if feature_col not in pairs.columns:
                continue

            obs = calculate_effect_for_feature(pairs, feature_col, ptype, y_col, w_col)
            perm_effects, perm_r2s = permutation_effects(
                pairs, meta, pred, ptype, y_col, w_col, n_permutations, rng
            )

            if ptype == "continuous":
                # Theory: smaller differences should imply higher agreement, so the slope should be negative.
                directional = empirical_p_directional(perm_effects, obs["effect"], "low")
                expected_direction = "negative_slope_closer_judges_agree_more"
            else:
                # Theory: shared category should imply higher agreement.
                directional = empirical_p_directional(perm_effects, obs["effect"], "high")
                expected_direction = "positive_same_category_agrees_more"

            rows.append({
                "outcome": outcome_label,
                "outcome_column": y_col,
                "weight_column": w_col,
                "predictor": pred,
                "predictor_type": ptype,
                "is_negative_control": pred.startswith("NEG_"),
                "feature_column": feature_col,
                "effect": obs["effect"],
                "effect_kind": obs["effect_kind"],
                "expected_direction": expected_direction,
                "weighted_r2": obs["weighted_r2"],
                "mean_same_or_lowdiff": obs["mean_same_or_lowdiff"],
                "mean_different_or_highdiff": obs["mean_different_or_highdiff"],
                "n_same_or_lowdiff_pairs": obs["n_same_or_lowdiff_pairs"],
                "n_different_or_highdiff_pairs": obs["n_different_or_highdiff_pairs"],
                "permutation_mean_effect": float(np.nanmean(perm_effects)),
                "permutation_sd_effect": float(np.nanstd(perm_effects, ddof=1)),
                "permutation_95pct_abs_effect": float(np.nanpercentile(np.abs(perm_effects), 95)),
                "empirical_p_directional": directional,
                "empirical_p_two_sided_effect": empirical_p_two_sided(perm_effects, obs["effect"]),
                "permutation_mean_r2": float(np.nanmean(perm_r2s)),
                "permutation_sd_r2": float(np.nanstd(perm_r2s, ddof=1)),
                "permutation_95pct_r2": float(np.nanpercentile(perm_r2s, 95)),
                "empirical_p_high_r2": empirical_p_directional(perm_r2s, obs["weighted_r2"], "high"),
                "n_permutations": n_permutations,
                "n_pairs_with_outcome": int(pd.to_numeric(pairs[y_col], errors="coerce").notna().sum()),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["outcome", "empirical_p_directional", "empirical_p_high_r2", "predictor"])
    return out


# ---------------------------------------------------------------------------
# Negative-control comparison
# ---------------------------------------------------------------------------

def build_negative_control_comparison(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if effects.empty:
        return pd.DataFrame()
    for (outcome, ptype), g in effects.groupby(["outcome", "predictor_type"]):
        neg = g[g["is_negative_control"]].copy()
        real = g[~g["is_negative_control"]].copy()
        if neg.empty or real.empty:
            continue
        neg_abs_effect = neg["effect"].abs().dropna()
        neg_r2 = neg["weighted_r2"].dropna()
        for _, row in real.iterrows():
            rows.append({
                "outcome": outcome,
                "predictor": row["predictor"],
                "predictor_type": ptype,
                "effect": row["effect"],
                "abs_effect": abs(row["effect"]) if pd.notna(row["effect"]) else np.nan,
                "weighted_r2": row["weighted_r2"],
                "empirical_p_directional": row["empirical_p_directional"],
                "empirical_p_high_r2": row["empirical_p_high_r2"],
                "negative_controls_available": len(neg),
                "negative_control_max_abs_effect": neg_abs_effect.max() if len(neg_abs_effect) else np.nan,
                "negative_control_95pct_abs_effect": neg_abs_effect.quantile(0.95) if len(neg_abs_effect) else np.nan,
                "negative_control_max_r2": neg_r2.max() if len(neg_r2) else np.nan,
                "negative_control_95pct_r2": neg_r2.quantile(0.95) if len(neg_r2) else np.nan,
                "beats_all_same_type_negative_controls_by_abs_effect": bool(abs(row["effect"]) > neg_abs_effect.max()) if len(neg_abs_effect) and pd.notna(row["effect"]) else np.nan,
                "beats_all_same_type_negative_controls_by_r2": bool(row["weighted_r2"] > neg_r2.max()) if len(neg_r2) and pd.notna(row["weighted_r2"]) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["outcome", "empirical_p_directional", "predictor"])
    return out


# ---------------------------------------------------------------------------
# Exemplar cases
# ---------------------------------------------------------------------------

def load_pair_case(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not path.exists():
        print(f"Warning: pair-case file not found: {path}. Skipping exemplars.")
        return None
    df = pd.read_csv(path)
    required = {"case_id", "case_name", "year", "judge_a", "judge_b", "same_opinion_group"}
    missing = required - set(df.columns)
    if missing:
        print(f"Warning: pair-case file missing exemplar columns {sorted(missing)}. Skipping exemplars.")
        return None
    df = df.copy()
    df["judge_a_key"] = df["judge_a"].map(canonical_last_name)
    df["judge_b_key"] = df["judge_b"].map(canonical_last_name)
    df["same_opinion_group_bool"] = parse_bool_like(df["same_opinion_group"])
    return df


def exemplar_cases_for_predictor(pair_case: pd.DataFrame, meta: pd.DataFrame, predictor: str, ptype: str, top_n: int) -> pd.DataFrame:
    valmap = values_by_judge(meta, predictor, ptype)
    df = pair_case.copy()
    df["feature"] = pair_feature_from_values(df, valmap, ptype)
    df["same_opinion"] = df["same_opinion_group_bool"].astype(float)
    rows = []

    for case_id, g in df.groupby("case_id"):
        y = g["same_opinion"].to_numpy(dtype=float)
        x = pd.to_numeric(g["feature"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 4:
            continue
        y = y[mask]
        x = x[mask]

        if ptype == "continuous":
            if len(np.unique(x)) < 2:
                continue
            score = weighted_slope(y, x, np.ones(len(x)))
            # More negative = stronger "closer judges agree more" exemplar.
            sort_score = -score if np.isfinite(score) else np.nan
            same_mean = np.nan
            diff_mean = np.nan
            n_same = np.nan
            n_diff = np.nan
        else:
            same_mask = x == 1
            diff_mask = x == 0
            if same_mask.sum() == 0 or diff_mask.sum() == 0:
                continue
            same_mean = float(np.mean(y[same_mask]))
            diff_mean = float(np.mean(y[diff_mask]))
            score = same_mean - diff_mean
            sort_score = score
            n_same = int(same_mask.sum())
            n_diff = int(diff_mask.sum())

        first = g.iloc[0]
        rows.append({
            "predictor": predictor,
            "predictor_type": ptype,
            "case_id": case_id,
            "citation": first.get("citation", ""),
            "case_name": first.get("case_name", ""),
            "year": first.get("year", ""),
            "subjects": first.get("subjects", ""),
            "exemplar_score": score,
            "sort_score": sort_score,
            "same_group_rate_when_shared_or_slope": same_mean,
            "same_group_rate_when_different": diff_mean,
            "n_shared_pairs": n_same,
            "n_different_pairs": n_diff,
            "panel_pair_rows_used": int(mask.sum()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("sort_score", ascending=False).head(top_n)
    return out.drop(columns=["sort_score"])


def build_exemplars(pair_case: Optional[pd.DataFrame], meta: pd.DataFrame, predictors: List[str], ptypes: Dict[str, str], top_n: int) -> pd.DataFrame:
    if pair_case is None:
        return pd.DataFrame()
    frames = []
    for pred in predictors:
        # Usually exemplars are most interpretable for real predictors, but include
        # negative controls too because they are useful diagnostics.
        ex = exemplar_cases_for_predictor(pair_case, meta, pred, ptypes[pred], top_n)
        if not ex.empty:
            frames.append(ex)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct_points(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:+.1f} pp"


def write_report(path: Path, effects: pd.DataFrame, neg: pd.DataFrame, predictors: List[str], ptypes: Dict[str, str], n_pairs: int, n_perm: int) -> None:
    lines = []
    lines.append("Pairwise judicial predictor analysis report")
    lines.append("============================================")
    lines.append("")
    lines.append(f"Judge pairs included: {n_pairs}")
    lines.append(f"Predictors included: {len(predictors)}")
    lines.append(f"Permutations per predictor/outcome: {n_perm}")
    lines.append("")
    lines.append("Predictor types")
    lines.append("---------------")
    for p in predictors:
        suffix = " [negative control]" if p.startswith("NEG_") else ""
        lines.append(f"- {p}: {ptypes[p]}{suffix}")
    lines.append("")

    if not effects.empty:
        for outcome in effects["outcome"].drop_duplicates():
            # Include ALL predictors, real and negative-control, so negative
            # controls are visible in the same ranked list for direct comparison.
            g = effects[effects["outcome"] == outcome].copy()
            g = g.sort_values(["empirical_p_directional", "empirical_p_high_r2", "weighted_r2"], ascending=[True, True, False])
            lines.append(f"All predictors (including negative controls): {outcome}")
            lines.append("-" * (37 + len(outcome)))
            for _, r in g.iterrows():
                tag = " [NEG CONTROL]" if r["is_negative_control"] else ""
                if r["predictor_type"] == "continuous":
                    eff = f"slope={r['effect']:.4f} per unit difference"
                else:
                    eff = f"same-vs-different={pct_points(r['effect'])}"
                lines.append(
                    f"- {r['predictor']}{tag}: {eff}; R²={r['weighted_r2']:.3f}; "
                    f"p_directional={r['empirical_p_directional']:.4f}; p_R²={r['empirical_p_high_r2']:.4f}"
                )
            lines.append("")

    if not neg.empty:
        lines.append("Negative-control interpretation")
        lines.append("-------------------------------")
        lines.append("A real predictor is more credible when it beats same-type negative controls and has a small permutation p-value.")
        beaters = neg[(neg["beats_all_same_type_negative_controls_by_abs_effect"] == True) & (neg["empirical_p_directional"] <= 0.05)]
        if beaters.empty:
            lines.append("No real predictor both beat all same-type negative controls by absolute effect and reached p <= .05.")
        else:
            for _, r in beaters.sort_values(["outcome", "empirical_p_directional"]).iterrows():
                lines.append(f"- {r['outcome']}: {r['predictor']} beat same-type negative controls; p={r['empirical_p_directional']:.4f}")
        lines.append("")

    lines.append("Notes")
    lines.append("-----")
    lines.append("For categorical/binary variables, effect is weighted agreement among same-category pairs minus weighted agreement among different-category pairs.")
    lines.append("For continuous variables, effect is the weighted slope of agreement on absolute pairwise difference; negative slopes mean more similar judges agree more.")
    lines.append("Weights use the relevant known-case count for each agreement outcome.")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise predictor analysis for SCC judge agreement.")
    parser.add_argument("--metadata", required=True, type=Path, help="Judge metadata CSV, e.g. scc_justices_data.csv")
    parser.add_argument("--pair-summary", required=True, type=Path, help="judge_pair_summary.csv from opinion extraction outputs")
    parser.add_argument("--pair-case", type=Path, default=None, help="Optional judge_pair_case.csv for exemplar cases")
    parser.add_argument("--outdir", type=Path, default=Path("pairwise_predictor_analysis"))
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--predictors", nargs="*", default=None, help="Optional explicit predictor columns to include")
    parser.add_argument("--top-exemplars", type=int, default=10)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    meta = load_metadata(args.metadata)
    predictors = predictor_columns(meta, args.predictors)
    pairs = load_pair_summary(args.pair_summary, meta)
    pairs_with_features, ptypes = add_all_pair_features(pairs, meta, predictors)

    effects = run_effects(
        pairs_with_features,
        meta,
        predictors,
        ptypes,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    neg = build_negative_control_comparison(effects)
    pair_case = load_pair_case(args.pair_case)
    exemplars = build_exemplars(pair_case, meta, predictors, ptypes, args.top_exemplars)

    pairs_with_features.to_csv(args.outdir / "pairwise_joined_features.csv", index=False)
    effects.to_csv(args.outdir / "pairwise_predictor_effects.csv", index=False)
    neg.to_csv(args.outdir / "pairwise_negative_control_comparison.csv", index=False)
    exemplars.to_csv(args.outdir / "pairwise_exemplar_cases.csv", index=False)
    write_report(args.outdir / "pairwise_analysis_report.txt", effects, neg, predictors, ptypes, len(pairs_with_features), args.n_permutations)

    print(f"Done. Wrote outputs to: {args.outdir}")
    if not effects.empty:
        print("\nTop opinion-group predictors:")
        top = effects[(effects["outcome"] == "Opinion Group Agreement") & (~effects["is_negative_control"])].head(10)
        for _, r in top.iterrows():
            if r["predictor_type"] == "continuous":
                eff = f"slope={r['effect']:.4f}"
            else:
                eff = f"same-vs-different={r['effect']*100:+.1f} pp"
            print(f"- {r['predictor']}: {eff}; R²={r['weighted_r2']:.3f}; p={r['empirical_p_directional']:.4f}")


if __name__ == "__main__":
    main()
