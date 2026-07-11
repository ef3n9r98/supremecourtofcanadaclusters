#!/usr/bin/env python3
"""
analyze_judicial_predictors.py
==============================

Purpose
-------
Measure how much commonly discussed SCC appointment characteristics predict
where justices sit on the MDS dimensions of opinion-group agreement.

This script is designed to run after your existing pipeline:

1. extract_opinion_groups_v6.py
   -> opinion_group_outputs_v6/judge_case_opinions.csv

2. permutation_random_group_assignment.py
   -> permutation_outputs/permutation_pair_results.csv

3. mds_agreement_map_3d.py
   -> mds_3d_outputs/mds_3d_coordinates.csv

Main outputs
------------
outputs/judicial_predictor_analysis/
  joined_judge_metadata_and_mds.csv
  marginal_r2_by_predictor.csv
  marginal_r2_summary_by_predictor.csv
  unique_r2_by_predictor.csv
  permutation_r2_by_predictor.csv
  negative_control_comparison.csv
  exemplar_cases_by_predictor.csv
  analysis_report.txt

Recommended run
---------------
python analyze_judicial_predictors.py \
  --metadata scc_justices_data.csv \
  --mds-coordinates mds_3d_outputs/mds_3d_coordinates.csv \
  --judge-case-opinions opinion_group_outputs_v6/judge_case_opinions.csv \
  --outdir judicial_predictor_analysis \
  --n-permutations 10000

Notes
-----
- Marginal R² asks: how much does one predictor explain by itself?
- Unique R² asks: how much does one predictor add after the other selected
  predictors are already in the model?
- With only ~12 justices, a full model can easily saturate. This script detects
  that and either skips unsafe unique-R² models or lets you specify a smaller
  set of predictors with --full-model-predictors.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

ALIASES = {
    "cote": "Côté",
    "côté": "Côté",
    "o'bonsawin": "O'Bonsawin",
    "o’bonsawin": "O'Bonsawin",
    "obonsawin": "O'Bonsawin",
    "karakatsanis": "Karakatsanis",
    "kasirer": "Kasirer",
    "wagner": "Wagner",
    "jamal": "Jamal",
    "moldaver": "Moldaver",
    "martin": "Martin",
    "moreau": "Moreau",
    "abella": "Abella",
    "brown": "Brown",
    "rowe": "Rowe",
}


def normalize_apostrophes(s: str) -> str:
    return str(s).replace("\u2019", "'").replace("\u02bc", "'").strip()


def judge_short_name(name: str) -> str:
    """Return the short judge name used by your MDS outputs."""
    raw = normalize_apostrophes(name)
    if not raw or raw.lower() == "nan":
        return ""

    # Handles "Kasirer, Nicholas" from opinion files.
    if "," in raw:
        last = raw.split(",", 1)[0].strip()
    else:
        parts = raw.split()
        last = parts[-1].strip() if parts else raw

    key = re.sub(r"[^a-zà-ÿ']", "", last.lower())
    return ALIASES.get(key, last)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_metadata(path: Path) -> pd.DataFrame:
    df = read_csv_required(path)
    if "Name" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'Name' column.")

    df = df.copy()
    df["judge"] = df["Name"].map(judge_short_name)

    # Derived predictors. These are often more useful than raw years.
    year_cols = [
        "Year of Birth",
        "Year First Appointed to the Bench",
        "Year Called to the Bar",
        "Year Appointed to SCC",
    ]
    for col in year_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"Year Appointed to SCC", "Year of Birth"}.issubset(df.columns):
        df["Age at SCC Appointment"] = df["Year Appointed to SCC"] - df["Year of Birth"]

    if {"Year Appointed to SCC", "Year First Appointed to the Bench"}.issubset(df.columns):
        df["Years on Bench Before SCC"] = (
            df["Year Appointed to SCC"] - df["Year First Appointed to the Bench"]
        )

    if {"Year Appointed to SCC", "Year Called to the Bar"}.issubset(df.columns):
        df["Years from Bar Call to SCC"] = df["Year Appointed to SCC"] - df["Year Called to the Bar"]

    if {"Year First Appointed to the Bench", "Year Called to the Bar"}.issubset(df.columns):
        df["Years Practice Before First Bench Appointment"] = (
            df["Year First Appointed to the Bench"] - df["Year Called to the Bar"]
        )

    add_negative_controls(df)
    return df


def load_mds_coordinates(path: Path) -> pd.DataFrame:
    df = read_csv_required(path).copy()

    # Your 3D script writes judge,x,y,z. This also supports Dim1/Dimension1 names.
    lower = {c.lower(): c for c in df.columns}

    judge_col = lower.get("judge") or lower.get("name")
    if not judge_col:
        raise ValueError("MDS coordinate CSV must contain a judge/name column.")

    coord_candidates = [
        ("x", "y", "z"),
        ("dim1", "dim2", "dim3"),
        ("dimension1", "dimension2", "dimension3"),
        ("dimension_1", "dimension_2", "dimension_3"),
    ]

    coord_cols = None
    for trio in coord_candidates:
        if all(c in lower for c in trio):
            coord_cols = tuple(lower[c] for c in trio)
            break

    if coord_cols is None:
        raise ValueError(
            "Could not find 3 coordinate columns. Expected x/y/z, dim1/dim2/dim3, "
            "or dimension1/dimension2/dimension3."
        )

    out = df[[judge_col, *coord_cols]].copy()
    out.columns = ["judge", "Legal Category", "Workability", "Remedial Force"]
    out["judge"] = out["judge"].map(judge_short_name)

    for col in ["Legal Category", "Workability", "Remedial Force"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.dropna(subset=["judge"])


# ---------------------------------------------------------------------------
# Negative controls and predictor typing
# ---------------------------------------------------------------------------


def add_negative_controls(df: pd.DataFrame) -> None:
    """Add two controls for binary, categorical, and continuous styles."""
    surnames = df["judge"].astype(str)

    df["NEG_BINARY_Surname_A_to_M"] = surnames.str[0].str.upper().between("A", "M")

    if "Year Appointed to SCC" in df.columns:
        df["NEG_BINARY_Even_SCC_Appointment_Year"] = (
            pd.to_numeric(df["Year Appointed to SCC"], errors="coerce") % 2 == 0
        )
    else:
        df["NEG_BINARY_Even_SCC_Appointment_Year"] = np.arange(len(df)) % 2 == 0

    df["NEG_CATEGORICAL_Last_Name_Initial"] = surnames.str[0].str.upper()

    if "Year of Birth" in df.columns:
        # Fake four-bucket cohort from birth-year modulo 4. It is arbitrary but multi-category.
        df["NEG_CATEGORICAL_Birth_Year_Mod_4"] = (
            pd.to_numeric(df["Year of Birth"], errors="coerce") % 4
        ).map(lambda x: f"mod_{int(x)}" if pd.notna(x) else np.nan)
    else:
        df["NEG_CATEGORICAL_Row_Mod_4"] = [f"mod_{i % 4}" for i in range(len(df))]

    df["NEG_CONTINUOUS_Surname_Length"] = surnames.str.replace(r"[^A-Za-zÀ-ÿ]", "", regex=True).str.len()
    df["NEG_CONTINUOUS_Alphabetical_Surname_Rank"] = surnames.rank(method="dense").astype(float)


def is_negative_control(col: str) -> bool:
    return col.startswith("NEG_")


def infer_predictor_type(s: pd.Series) -> str:
    cleaned = s.dropna()
    unique = cleaned.nunique(dropna=True)

    if unique <= 2:
        return "binary"

    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().all():
        return "continuous"

    return "categorical"


def eligible_predictors(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    exclude_set = set(exclude)
    out = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if df[col].nunique(dropna=True) < 2:
            continue
        out.append(col)
    return out


# ---------------------------------------------------------------------------
# Linear model helpers
# ---------------------------------------------------------------------------


def design_matrix(df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build an intercept + encoded design matrix, plus map predictor -> encoded cols."""
    pieces = []
    encoded_map: dict[str, list[str]] = {}

    for pred in predictors:
        ptype = infer_predictor_type(df[pred])
        if ptype == "continuous":
            colname = pred
            x = pd.to_numeric(df[pred], errors="coerce").rename(colname).to_frame()
            pieces.append(x)
            encoded_map[pred] = [colname]
        else:
            dummies = pd.get_dummies(df[pred].astype("category"), prefix=pred, drop_first=True, dtype=float)
            if dummies.shape[1] == 0:
                continue
            pieces.append(dummies)
            encoded_map[pred] = list(dummies.columns)

    if pieces:
        X = pd.concat(pieces, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    X.insert(0, "Intercept", 1.0)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, encoded_map


def ols_r2(y: pd.Series, X: pd.DataFrame) -> tuple[float, int, int]:
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < 3:
        return np.nan, len(data), X.shape[1]

    yv = data["y"].astype(float).to_numpy()
    Xv = data.drop(columns=["y"]).astype(float).to_numpy()

    # Need variation in y.
    tss = float(np.sum((yv - yv.mean()) ** 2))
    if tss == 0:
        return np.nan, len(data), Xv.shape[1]

    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    pred = Xv @ beta
    rss = float(np.sum((yv - pred) ** 2))
    r2 = 1.0 - rss / tss
    # Numerical cleanup.
    r2 = max(0.0, min(1.0, r2))
    return r2, len(data), Xv.shape[1]


def model_safe_for_unique(n: int, p: int, max_df_ratio: float) -> bool:
    # p includes intercept. Require residual df and avoid near-saturated models.
    if n <= p + 1:
        return False
    return (p - 1) <= max(1, math.floor(max_df_ratio * n))


# ---------------------------------------------------------------------------
# R² analyses
# ---------------------------------------------------------------------------


def marginal_r2(joined: pd.DataFrame, predictors: list[str], dimensions: list[str]) -> pd.DataFrame:
    rows = []
    for dim in dimensions:
        for pred in predictors:
            X, enc = design_matrix(joined, [pred])
            r2, n, p = ols_r2(joined[dim], X)
            rows.append({
                "dimension": dim,
                "predictor": pred,
                "predictor_type": infer_predictor_type(joined[pred]),
                "is_negative_control": is_negative_control(pred),
                "r2": r2,
                "n_judges": n,
                "model_columns_including_intercept": p,
                "encoded_degrees": max(0, p - 1),
            })
    return pd.DataFrame(rows).sort_values(["dimension", "r2"], ascending=[True, False])


def unique_r2(
    joined: pd.DataFrame,
    predictors: list[str],
    dimensions: list[str],
    max_df_ratio: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop-one-predictor unique R². Skips dimensions if model is unsafe."""
    rows = []
    warnings = []

    real_predictors = [p for p in predictors if not is_negative_control(p)]
    X_full, enc_map = design_matrix(joined, real_predictors)

    for dim in dimensions:
        y = joined[dim]
        full_r2, n, p_full = ols_r2(y, X_full)

        if not model_safe_for_unique(n, p_full, max_df_ratio):
            warnings.append(
                f"Unique R² skipped for {dim}: full model has n={n}, columns={p_full} "
                f"including intercept. Too close to saturated. Use --full-model-predictors "
                f"with a smaller theory-driven subset, or increase --max-full-df-ratio knowingly."
            )
            for pred in real_predictors:
                rows.append({
                    "dimension": dim,
                    "predictor": pred,
                    "predictor_type": infer_predictor_type(joined[pred]),
                    "full_model_r2": np.nan,
                    "reduced_model_r2": np.nan,
                    "unique_r2_drop": np.nan,
                    "n_judges": n,
                    "full_model_columns_including_intercept": p_full,
                    "status": "skipped_full_model_too_large",
                })
            continue

        for pred in real_predictors:
            cols_to_drop = enc_map.get(pred, [])
            if not cols_to_drop:
                continue
            X_reduced = X_full.drop(columns=cols_to_drop)
            reduced_r2, _, p_reduced = ols_r2(y, X_reduced)
            rows.append({
                "dimension": dim,
                "predictor": pred,
                "predictor_type": infer_predictor_type(joined[pred]),
                "full_model_r2": full_r2,
                "reduced_model_r2": reduced_r2,
                "unique_r2_drop": full_r2 - reduced_r2 if pd.notna(full_r2) and pd.notna(reduced_r2) else np.nan,
                "n_judges": n,
                "full_model_columns_including_intercept": p_full,
                "reduced_model_columns_including_intercept": p_reduced,
                "status": "ok",
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dimension", "unique_r2_drop"], ascending=[True, False])
    return out, warnings


def permutation_marginal_r2(
    joined: pd.DataFrame,
    predictors: list[str],
    dimensions: list[str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for dim in dimensions:
        y = joined[dim]
        for pred in predictors:
            X_obs, _ = design_matrix(joined, [pred])
            obs_r2, n, p = ols_r2(y, X_obs)
            if pd.isna(obs_r2):
                continue

            samples = np.empty(n_permutations, dtype=float)
            values = joined[pred].to_numpy(copy=True)

            for i in range(n_permutations):
                shuffled = values.copy()
                rng.shuffle(shuffled)
                temp = joined.copy()
                temp[pred] = shuffled
                X_perm, _ = design_matrix(temp, [pred])
                samples[i], _, _ = ols_r2(y, X_perm)

            rows.append({
                "dimension": dim,
                "predictor": pred,
                "predictor_type": infer_predictor_type(joined[pred]),
                "is_negative_control": is_negative_control(pred),
                "observed_r2": obs_r2,
                "permutation_mean_r2": float(np.nanmean(samples)),
                "permutation_sd_r2": float(np.nanstd(samples, ddof=1)),
                "empirical_p_high": float((np.sum(samples >= obs_r2) + 1) / (len(samples) + 1)),
                "permutation_95pct": float(np.nanpercentile(samples, 95)),
                "n_permutations": n_permutations,
                "n_judges": n,
                "model_columns_including_intercept": p,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dimension", "observed_r2"], ascending=[True, False])
    return out


# ---------------------------------------------------------------------------
# Exemplar case analysis
# ---------------------------------------------------------------------------


def categorical_group_separation(panel: pd.DataFrame, predictor: str) -> float:
    """
    Score whether opinion groups separate a categorical/binary predictor.
    0 means no improvement over baseline category concentration.
    Higher means opinion groups are purer than the case panel overall.
    """
    if panel[predictor].nunique(dropna=True) < 2:
        return np.nan

    base = panel[predictor].value_counts(normalize=True, dropna=True).max()
    weighted_purity = 0.0
    total = 0
    for _, g in panel.groupby("opinion_group"):
        n = len(g)
        if n == 0:
            continue
        purity = g[predictor].value_counts(normalize=True, dropna=True).max()
        weighted_purity += n * purity
        total += n
    if total == 0:
        return np.nan
    return float(weighted_purity / total - base)


def continuous_group_separation(panel: pd.DataFrame, predictor: str) -> float:
    """Eta-squared: proportion of predictor variance explained by opinion group."""
    x = pd.to_numeric(panel[predictor], errors="coerce")
    tmp = panel.assign(_x=x).dropna(subset=["_x"])
    if len(tmp) < 3 or tmp["opinion_group"].nunique() < 2 or tmp["_x"].nunique() < 2:
        return np.nan

    grand = tmp["_x"].mean()
    ss_total = float(((tmp["_x"] - grand) ** 2).sum())
    if ss_total == 0:
        return np.nan

    ss_between = 0.0
    for _, g in tmp.groupby("opinion_group"):
        ss_between += len(g) * float((g["_x"].mean() - grand) ** 2)
    return float(ss_between / ss_total)


def group_breakdown(panel: pd.DataFrame, predictor: str) -> str:
    parts = []
    for group, g in panel.groupby("opinion_group"):
        judges = ", ".join(g["judge"].tolist())
        vals = "; ".join(f"{k}: {v}" for k, v in g[predictor].value_counts(dropna=False).items())
        parts.append(f"{group} [{judges}] -> {vals}")
    return " | ".join(parts)


def exemplar_cases(
    judge_case_path: Optional[Path],
    metadata: pd.DataFrame,
    predictors: list[str],
    top_n: int,
) -> pd.DataFrame:
    if judge_case_path is None or not judge_case_path.exists():
        return pd.DataFrame()

    jc = pd.read_csv(judge_case_path).copy()
    required = {"case_id", "case_name", "year", "judge", "opinion_group"}
    missing = required - set(jc.columns)
    if missing:
        raise ValueError(f"judge-case opinions file missing columns: {sorted(missing)}")

    jc = jc.dropna(subset=["opinion_group"]).copy()
    jc = jc[jc["opinion_group"].astype(str).str.strip().ne("")]
    jc = jc[jc["opinion_group"].astype(str).ne("UNASSIGNED")]
    jc["judge"] = jc["judge"].map(judge_short_name)

    meta_cols = ["judge"] + predictors
    data = jc.merge(metadata[meta_cols], on="judge", how="left")

    rows = []
    for pred in predictors:
        ptype = infer_predictor_type(metadata[pred])
        for case_id, panel in data.groupby("case_id"):
            if panel["opinion_group"].nunique() < 2:
                continue
            if panel[pred].nunique(dropna=True) < 2:
                continue

            if ptype == "continuous":
                score = continuous_group_separation(panel, pred)
            else:
                score = categorical_group_separation(panel, pred)

            if pd.isna(score):
                continue

            rows.append({
                "predictor": pred,
                "predictor_type": ptype,
                "case_id": case_id,
                "case_name": panel["case_name"].iloc[0],
                "year": panel["year"].iloc[0],
                "panel_size": panel["judge"].nunique(),
                "num_opinion_groups": panel["opinion_group"].nunique(),
                "separation_score": score,
                "group_breakdown": group_breakdown(panel, pred),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["predictor", "separation_score"], ascending=[True, False])
    return out.groupby("predictor", group_keys=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def negative_control_comparison(perm: pd.DataFrame) -> pd.DataFrame:
    if perm.empty:
        return pd.DataFrame()

    rows = []
    for (dim, ptype), g in perm.groupby(["dimension", "predictor_type"]):
        real = g[~g["is_negative_control"]]
        neg = g[g["is_negative_control"]]
        if real.empty or neg.empty:
            continue
        threshold = neg["observed_r2"].max()
        for _, row in real.iterrows():
            rows.append({
                "dimension": dim,
                "predictor_type": ptype,
                "predictor": row["predictor"],
                "observed_r2": row["observed_r2"],
                "max_negative_control_r2_same_type": threshold,
                "beats_negative_controls_same_type": bool(row["observed_r2"] > threshold),
                "empirical_p_high": row["empirical_p_high"],
            })
    return pd.DataFrame(rows).sort_values(["dimension", "observed_r2"], ascending=[True, False])


def write_report(
    outdir: Path,
    joined: pd.DataFrame,
    marginal: pd.DataFrame,
    unique: pd.DataFrame,
    perm: pd.DataFrame,
    warnings: list[str],
    predictor_types: dict[str, str],
) -> None:
    lines = []
    lines.append("Judicial predictor analysis report")
    lines.append("==================================")
    lines.append("")
    lines.append(f"Judges included: {len(joined)}")
    lines.append("Dimensions: Legal Category, Workability, Remedial Force")
    lines.append("")

    lines.append("Predictor types")
    lines.append("---------------")
    for pred, typ in predictor_types.items():
        lines.append(f"- {pred}: {typ}{' [negative control]' if is_negative_control(pred) else ''}")
    lines.append("")

    lines.append("Top marginal R² by dimension")
    lines.append("----------------------------")
    if not marginal.empty:
        for dim, g in marginal[~marginal["is_negative_control"]].groupby("dimension"):
            lines.append(f"\n{dim}")
            for _, row in g.sort_values("r2", ascending=False).head(8).iterrows():
                lines.append(f"  {row['predictor']}: R²={row['r2']:.3f}")
    lines.append("")

    if warnings:
        lines.append("Unique R² warnings")
        lines.append("------------------")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")
    elif not unique.empty:
        lines.append("Top unique R² drops by dimension")
        lines.append("--------------------------------")
        for dim, g in unique.groupby("dimension"):
            lines.append(f"\n{dim}")
            for _, row in g.sort_values("unique_r2_drop", ascending=False).head(8).iterrows():
                lines.append(f"  {row['predictor']}: unique ΔR²={row['unique_r2_drop']:.3f}")
        lines.append("")

    lines.append("Permutation interpretation")
    lines.append("--------------------------")
    lines.append("empirical_p_high is the share of random label shuffles that achieved at least the observed R².")
    lines.append("Small values mean the predictor did better than random relabelling.")

    (outdir / "analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SCC justice metadata as predictors of MDS dimensions.")
    parser.add_argument("--metadata", type=Path, default=Path("scc_justices_data.csv"))
    parser.add_argument("--mds-coordinates", type=Path, default=Path("mds_3d_outputs/mds_3d_coordinates.csv"))
    parser.add_argument("--judge-case-opinions", type=Path, default=Path("opinion_group_outputs_v6/judge_case_opinions.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("judicial_predictor_analysis"))
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--top-exemplar-cases", type=int, default=10)
    parser.add_argument(
        "--full-model-predictors",
        nargs="*",
        default=None,
        help="Optional smaller subset for unique R². Use exact metadata column names.",
    )
    parser.add_argument(
        "--max-full-df-ratio",
        type=float,
        default=0.50,
        help="Safety limit for full model degrees of freedom relative to n. Default 0.50.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    mds = load_mds_coordinates(args.mds_coordinates)
    joined = metadata.merge(mds, on="judge", how="inner")

    if joined.empty:
        raise ValueError("No judges matched between metadata and MDS coordinates. Check name formats.")

    dimensions = ["Legal Category", "Workability", "Remedial Force"]

    base_exclude = {"Name", "judge", *dimensions}
    all_predictors = eligible_predictors(joined, base_exclude)

    # If the user supplies a smaller full-model list, still compute marginal for all,
    # but unique only for that subset plus no negative controls.
    if args.full_model_predictors:
        missing = [p for p in args.full_model_predictors if p not in joined.columns]
        if missing:
            raise ValueError(f"--full-model-predictors not found in joined data: {missing}")
        unique_predictors = args.full_model_predictors
    else:
        unique_predictors = all_predictors

    predictor_types = {p: infer_predictor_type(joined[p]) for p in all_predictors}

    joined.to_csv(args.outdir / "joined_judge_metadata_and_mds.csv", index=False)

    marginal = marginal_r2(joined, all_predictors, dimensions)
    marginal.to_csv(args.outdir / "marginal_r2_by_predictor.csv", index=False)

    summary = (
        marginal.groupby(["predictor", "predictor_type", "is_negative_control"], as_index=False)
        .agg(mean_r2=("r2", "mean"), max_r2=("r2", "max"))
        .sort_values("max_r2", ascending=False)
    )
    summary.to_csv(args.outdir / "marginal_r2_summary_by_predictor.csv", index=False)

    unique, warnings = unique_r2(joined, unique_predictors, dimensions, args.max_full_df_ratio)
    unique.to_csv(args.outdir / "unique_r2_by_predictor.csv", index=False)

    perm = permutation_marginal_r2(joined, all_predictors, dimensions, args.n_permutations, args.seed)
    perm.to_csv(args.outdir / "permutation_r2_by_predictor.csv", index=False)

    neg = negative_control_comparison(perm)
    neg.to_csv(args.outdir / "negative_control_comparison.csv", index=False)

    exemplars = exemplar_cases(args.judge_case_opinions, metadata, all_predictors, args.top_exemplar_cases)
    exemplars.to_csv(args.outdir / "exemplar_cases_by_predictor.csv", index=False)

    write_report(args.outdir, joined, marginal, unique, perm, warnings, predictor_types)

    print(f"Done. Wrote outputs to: {args.outdir}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
