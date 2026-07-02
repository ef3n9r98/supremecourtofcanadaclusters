#!/usr/bin/env python3
"""
mds_agreement_map_3d.py
=======================

3D multidimensional scaling (MDS) map for SCC judge opinion-group agreement.

This is the 3D version of mds_agreement_map.py.

Best input:
  permutation_outputs/permutation_pair_results.csv

Recommended run:
  python mds_agreement_map_3d.py \
    --pair-results permutation_outputs/permutation_pair_results.csv \
    --outdir mds_3d_outputs \
    --mode z \
    --min-pair-cases 10

If you are using python3.11 instead:
  python3.11 mds_agreement_map_3d.py \
    --pair-results permutation_outputs/permutation_pair_results.csv \
    --outdir mds_3d_outputs \
    --mode z \
    --min-pair-cases 10

Outputs:
  mds_3d_outputs/
    mds_3d_coordinates.csv
    mds_3d_distance_matrix.csv
    mds_3d_similarity_matrix.csv
    mds_3d_stress_by_dimension.csv
    mds_3d_map.png
    mds_3d_pairwise_distances.csv
    mds_3d_report.txt

Optional:
  If plotly is installed, the script also writes:
    mds_3d_interactive.html

Interpretation:
  MDS dimensions do not have automatic legal meaning.
  Use this map to find structure, then inspect exemplar cases to interpret it.
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

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def last_name(full: str) -> str:
    return str(full).split(",", 1)[0]


def pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.1%}"


def find_existing(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_pair_data(pair_results: Optional[Path],
                   pair_summary: Optional[Path],
                   min_pair_cases: int) -> tuple[pd.DataFrame, str]:
    """
    Return standardized pair-level data with columns:
      judge_a_short
      judge_b_short
      cases_together
      observed_agreement_rate
      expected_agreement_rate
      observed_minus_expected
      z_score
    """
    if pair_results and pair_results.exists():
        df = pd.read_csv(pair_results)

        required = {
            "judge_a_short",
            "judge_b_short",
            "cases_together",
            "observed_agreement_rate",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{pair_results} is missing required columns: {sorted(missing)}")

        for col in [
            "cases_together",
            "observed_agreement_rate",
            "expected_agreement_rate",
            "observed_minus_expected",
            "z_score",
            "p_two_sided",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["cases_together"] >= min_pair_cases].copy()
        return df, f"permutation pair results: {pair_results}"

    if pair_summary and pair_summary.exists():
        df = pd.read_csv(pair_summary)

        required = {"judge_a", "judge_b", "opinion_group_known_cases", "opinion_group_agreement_rate"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{pair_summary} is missing required columns: {sorted(missing)}")

        df["judge_a_short"] = df["judge_a"].map(last_name)
        df["judge_b_short"] = df["judge_b"].map(last_name)
        df["cases_together"] = pd.to_numeric(df["opinion_group_known_cases"], errors="coerce")
        df["observed_agreement_rate"] = pd.to_numeric(df["opinion_group_agreement_rate"], errors="coerce")

        df["expected_agreement_rate"] = np.nan
        df["observed_minus_expected"] = np.nan
        df["z_score"] = np.nan
        df["p_two_sided"] = np.nan

        df = df[df["cases_together"] >= min_pair_cases].copy()
        return df, f"judge pair summary: {pair_summary}"

    raise FileNotFoundError(
        "Could not find usable input. Provide --pair-results or --pair-summary."
    )


# ---------------------------------------------------------------------------
# Similarity and distance matrices
# ---------------------------------------------------------------------------

def make_similarity_matrix(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    judges = sorted(set(df["judge_a_short"]) | set(df["judge_b_short"]))
    sim = pd.DataFrame(np.nan, index=judges, columns=judges, dtype=float)

    if mode == "observed":
        value_col = "observed_agreement_rate"
        diagonal_value = 1.0
    elif mode == "observed_minus_expected":
        value_col = "observed_minus_expected"
        diagonal_value = 0.0
    elif mode == "z":
        value_col = "z_score"
        diagonal_value = 0.0
    else:
        raise ValueError("mode must be one of: observed, observed_minus_expected, z")

    if value_col not in df.columns:
        raise ValueError(f"Input does not contain required column for mode={mode}: {value_col}")

    for j in judges:
        sim.loc[j, j] = diagonal_value

    for _, row in df.iterrows():
        a = row["judge_a_short"]
        b = row["judge_b_short"]
        val = row[value_col]
        if pd.notna(val):
            sim.loc[a, b] = float(val)
            sim.loc[b, a] = float(val)

    return sim


def similarity_to_distance(sim: pd.DataFrame,
                           mode: str,
                           missing_distance: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_records = []
    judges = list(sim.index)

    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            if pd.isna(sim.loc[a, b]):
                missing_records.append({"judge_a": a, "judge_b": b})

    if mode == "observed":
        dist = 1.0 - sim
    elif mode in {"z", "observed_minus_expected"}:
        vals = sim.values.astype(float)
        off_diag = vals[~np.eye(vals.shape[0], dtype=bool)]
        finite = off_diag[np.isfinite(off_diag)]
        if len(finite) == 0:
            raise ValueError(f"No finite off-diagonal values available for mode={mode}")
        max_sim = np.nanmax(finite)
        dist = max_sim - sim
    else:
        raise ValueError("mode must be one of: observed, observed_minus_expected, z")

    for j in judges:
        dist.loc[j, j] = 0.0

    off_diag_values = dist.values[~np.eye(len(judges), dtype=bool)]
    finite_distances = off_diag_values[np.isfinite(off_diag_values)]
    if len(finite_distances) == 0:
        raise ValueError("No finite distances available after conversion.")

    if missing_distance == "median":
        fill = float(np.nanmedian(finite_distances))
    elif missing_distance == "mean":
        fill = float(np.nanmean(finite_distances))
    elif missing_distance == "max":
        fill = float(np.nanmax(finite_distances))
    else:
        raise ValueError("missing_distance must be one of: median, mean, max")

    dist = dist.fillna(fill)

    # MDS expects symmetric, nonnegative distances.
    dist = (dist + dist.T) / 2.0
    dist[dist < 0] = 0.0
    for j in judges:
        dist.loc[j, j] = 0.0

    missing_df = pd.DataFrame(missing_records)
    if not missing_df.empty:
        missing_df["filled_distance"] = fill
        missing_df["missing_distance_strategy"] = missing_distance

    return dist, missing_df


# ---------------------------------------------------------------------------
# MDS
# ---------------------------------------------------------------------------

def run_mds(distance: pd.DataFrame,
            n_components: int,
            random_state: int,
            metric: bool = True):
    try:
        from sklearn.manifold import MDS
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for MDS. Install it into the Python you are using:\n"
            "  python -m pip install scikit-learn\n"
            "or:\n"
            "  python3.11 -m pip install scikit-learn"
        ) from e

    # sklearn changed normalized_stress support across versions.
    try:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=50,
            max_iter=2000,
            metric=metric,
            normalized_stress="auto",
        )
    except TypeError:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=50,
            max_iter=2000,
            metric=metric,
        )

    coords = mds.fit_transform(distance.values)
    stress = float(mds.stress_)
    return coords, stress


def stress_by_dimension(distance: pd.DataFrame,
                        max_dimensions: int,
                        random_state: int,
                        metric: bool) -> pd.DataFrame:
    rows = []
    n_judges = len(distance.index)
    max_d = max(1, min(max_dimensions, n_judges - 1))

    for d in range(1, max_d + 1):
        _, stress = run_mds(distance, d, random_state, metric=metric)
        rows.append({"dimensions": d, "stress": stress})

    out = pd.DataFrame(rows)
    out["stress_drop_from_previous"] = out["stress"].shift(1) - out["stress"]
    out["pct_drop_from_previous"] = out["stress_drop_from_previous"] / out["stress"].shift(1)
    return out


# ---------------------------------------------------------------------------
# Summaries and orientation
# ---------------------------------------------------------------------------

def judge_summary_from_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    judges = sorted(set(df["judge_a_short"]) | set(df["judge_b_short"]))

    for judge in judges:
        g = df[(df["judge_a_short"] == judge) | (df["judge_b_short"] == judge)].copy()
        rows.append({
            "judge": judge,
            "pairs_used": len(g),
            "mean_cases_together": g["cases_together"].mean() if "cases_together" in g else np.nan,
            "mean_observed_agreement_rate": g["observed_agreement_rate"].mean() if "observed_agreement_rate" in g else np.nan,
            "mean_observed_minus_expected": g["observed_minus_expected"].mean() if "observed_minus_expected" in g else np.nan,
            "mean_z_score": g["z_score"].mean() if "z_score" in g else np.nan,
            "min_z_score": g["z_score"].min() if "z_score" in g else np.nan,
            "max_z_score": g["z_score"].max() if "z_score" in g else np.nan,
        })

    return pd.DataFrame(rows)


def orient_coordinates_3d(coords_df: pd.DataFrame,
                          judge_summary: pd.DataFrame,
                          orient_by: str) -> pd.DataFrame:
    """
    MDS axes can be arbitrarily mirrored. Flipping an axis does not change the model.
    This just makes maps easier to compare across runs.
    """
    out = coords_df.copy()

    if orient_by == "none":
        return out

    if orient_by == "cote_positive_x":
        mask = out["judge"].str.lower().eq("côté") | out["judge"].str.lower().eq("cote")
        if mask.any() and float(out.loc[mask, "x"].iloc[0]) < out["x"].mean():
            out["x"] = -out["x"]
        return out

    if orient_by == "low_consensus_positive_x":
        merged = out.merge(
            judge_summary[["judge", "mean_observed_agreement_rate"]],
            on="judge",
            how="left",
        )
        corr = merged["x"].corr(merged["mean_observed_agreement_rate"])
        if pd.notna(corr) and corr > 0:
            out["x"] = -out["x"]
        return out

    raise ValueError("orient_by must be one of: none, cote_positive_x, low_consensus_positive_x")


def pairwise_coordinate_distances(coords_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coords = coords_df.set_index("judge")[["x", "y", "z"]]
    judges = list(coords.index)

    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            va = coords.loc[a].values.astype(float)
            vb = coords.loc[b].values.astype(float)
            rows.append({
                "judge_a": a,
                "judge_b": b,
                "mds_3d_distance": float(np.linalg.norm(va - vb)),
            })

    return pd.DataFrame(rows).sort_values("mds_3d_distance", ascending=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_3d_static(coords: pd.DataFrame,
                   out_path: Path,
                   title: str,
                   subtitle: str) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(coords["x"], coords["y"], coords["z"], s=80)

    for _, row in coords.iterrows():
        ax.text(
            row["x"],
            row["y"],
            row["z"],
            "  " + row["judge"],
            fontsize=9,
        )

    ax.set_title(title + "\n" + subtitle)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_zlabel("MDS dimension 3")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_2d_projection(coords: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       out_path: Path,
                       title: str,
                       subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[x_col], coords[y_col], s=90)

    for _, row in coords.iterrows():
        ax.annotate(
            row["judge"],
            (row[x_col], row[y_col]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_title(title + "\n" + subtitle)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_3d_interactive_if_possible(coords: pd.DataFrame,
                                    out_path: Path,
                                    title: str,
                                    subtitle: str) -> bool:
    try:
        import plotly.express as px
    except ImportError:
        return False

    hover_cols = [
        c for c in [
            "judge",
            "mean_observed_agreement_rate",
            "mean_z_score",
            "pairs_used",
            "mean_cases_together",
        ]
        if c in coords.columns
    ]

    fig = px.scatter_3d(
        coords,
        x="x",
        y="y",
        z="z",
        text="judge",
        hover_data=hover_cols,
        title=title + "<br>" + subtitle,
    )

    fig.update_traces(textposition="top center", marker=dict(size=6))
    fig.update_layout(
        scene=dict(
            xaxis_title="MDS dimension 1",
            yaxis_title="MDS dimension 2",
            zaxis_title="MDS dimension 3",
        )
    )

    fig.write_html(out_path)
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(out_path: Path,
                 source_label: str,
                 mode: str,
                 min_pair_cases: int,
                 metric: bool,
                 missing_df: pd.DataFrame,
                 stress_df: pd.DataFrame,
                 coords: pd.DataFrame,
                 pair_distances: pd.DataFrame,
                 judge_summary: pd.DataFrame) -> None:
    lines = []

    lines.append("3D MDS AGREEMENT MAP")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Input source: {source_label}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Minimum pair cases: {min_pair_cases}")
    lines.append(f"MDS type: {'metric' if metric else 'non-metric'}")
    lines.append("Dimensions plotted: 3")
    lines.append("")

    lines.append("How distance was calculated:")
    if mode == "observed":
        lines.append("  distance = 1 - observed_agreement_rate")
    elif mode == "z":
        lines.append("  similarity = z_score")
        lines.append("  distance = max(z_score) - z_score")
    elif mode == "observed_minus_expected":
        lines.append("  similarity = observed_minus_expected")
        lines.append("  distance = max(observed_minus_expected) - observed_minus_expected")
    lines.append("")

    lines.append("Important caution:")
    lines.append("  MDS dimensions do not have automatic legal meaning.")
    lines.append("  Interpret dimensions by extracting cases that separate the judges or clusters.")
    lines.append("")

    if not missing_df.empty:
        lines.append("Missing judge-pair distances:")
        lines.append(f"  Missing pairs filled: {len(missing_df)}")
        lines.append(f"  Fill strategy: {missing_df['missing_distance_strategy'].iloc[0]}")
        lines.append(f"  Filled distance: {missing_df['filled_distance'].iloc[0]:.4f}")
        lines.append("  These judge-pair relationships should not be interpreted strongly.")
        lines.append("")
    else:
        lines.append("Missing judge-pair distances: none")
        lines.append("")

    if not stress_df.empty:
        lines.append("Stress by dimension:")
        for _, r in stress_df.iterrows():
            drop = r.get("stress_drop_from_previous", np.nan)
            if pd.notna(drop):
                lines.append(
                    f"  {int(r['dimensions'])}D: stress={r['stress']:.4f}; "
                    f"drop={drop:.4f}; pct_drop={r['pct_drop_from_previous']:.1%}"
                )
            else:
                lines.append(f"  {int(r['dimensions'])}D: stress={r['stress']:.4f}")
        lines.append("")

    lines.append("3D coordinates:")
    for _, r in coords.sort_values("x").iterrows():
        lines.append(f"  {r['judge']}: x={r['x']:.4f}, y={r['y']:.4f}, z={r['z']:.4f}")
    lines.append("")

    lines.append("Most distant pairs in the 3D MDS coordinate space:")
    for _, r in pair_distances.head(10).iterrows():
        lines.append(
            f"  {r['judge_a']} - {r['judge_b']}: "
            f"distance={r['mds_3d_distance']:.4f}"
        )
    lines.append("")

    lines.append("Closest pairs in the 3D MDS coordinate space:")
    for _, r in pair_distances.sort_values("mds_3d_distance").head(10).iterrows():
        lines.append(
            f"  {r['judge_a']} - {r['judge_b']}: "
            f"distance={r['mds_3d_distance']:.4f}"
        )
    lines.append("")

    lines.append("Judge-level summary from included pair rows:")
    merged = judge_summary.sort_values("mean_observed_agreement_rate", ascending=False)
    for _, r in merged.iterrows():
        if pd.notna(r.get("mean_z_score", np.nan)):
            lines.append(
                f"  {r['judge']}: "
                f"mean observed agreement={pct(r['mean_observed_agreement_rate'])}, "
                f"mean z={r['mean_z_score']:.2f}"
            )
        else:
            lines.append(
                f"  {r['judge']}: "
                f"mean observed agreement={pct(r['mean_observed_agreement_rate'])}"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 3D MDS map of SCC judge agreement.")
    parser.add_argument("--pair-results", default=None,
                        help="Path to permutation_pair_results.csv")
    parser.add_argument("--pair-summary", default=None,
                        help="Path to judge_pair_summary.csv fallback")
    parser.add_argument("--outdir", default="mds_3d_outputs",
                        help="Output directory")
    parser.add_argument("--mode", choices=["z", "observed", "observed_minus_expected"],
                        default="z",
                        help="Similarity metric to map")
    parser.add_argument("--min-pair-cases", type=int, default=10,
                        help="Minimum cases together for a pair to be included")
    parser.add_argument("--missing-distance", choices=["median", "mean", "max"],
                        default="median",
                        help="How to fill missing judge-pair distances")
    parser.add_argument("--nonmetric", action="store_true",
                        help="Use non-metric MDS instead of metric MDS")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-stress-dimensions", type=int, default=6,
                        help="Calculate MDS stress for 1..N dimensions")
    parser.add_argument("--orient-by", choices=["none", "cote_positive_x", "low_consensus_positive_x"],
                        default="cote_positive_x",
                        help="Optionally flip x-axis for easier comparison across runs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    pair_results_path = Path(args.pair_results) if args.pair_results else find_existing([
        Path("permutation_outputs/permutation_pair_results.csv"),
        Path("permutation_outputs_split_only/permutation_pair_results.csv"),
    ])
    pair_summary_path = Path(args.pair_summary) if args.pair_summary else find_existing([
        Path("opinion_group_outputs_v6/judge_pair_summary.csv"),
        Path("judge_pair_summary.csv"),
    ])

    print("Loading pair data...")
    pair_df, source_label = load_pair_data(
        pair_results=pair_results_path,
        pair_summary=pair_summary_path,
        min_pair_cases=args.min_pair_cases,
    )

    if pair_df.empty:
        raise ValueError("No pair rows remain after filtering. Try lowering --min-pair-cases.")

    if args.mode in {"z", "observed_minus_expected"}:
        needed = "z_score" if args.mode == "z" else "observed_minus_expected"
        if needed not in pair_df.columns or pair_df[needed].dropna().empty:
            raise ValueError(
                f"Mode '{args.mode}' requires {needed}, which is only available from "
                "permutation_pair_results.csv. Use --mode observed or provide --pair-results."
            )

    print(f"Using: {source_label}")
    print(f"Pairs used: {len(pair_df)}")
    print(f"Mode: {args.mode}")

    print("Building similarity and distance matrices...")
    similarity = make_similarity_matrix(pair_df, args.mode)
    distance, missing_df = similarity_to_distance(
        similarity,
        mode=args.mode,
        missing_distance=args.missing_distance,
    )

    print("Running 3D MDS...")
    metric = not args.nonmetric
    coords, stress = run_mds(distance, n_components=3, random_state=args.seed, metric=metric)

    coords_df = pd.DataFrame({
        "judge": distance.index,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })

    judge_summary = judge_summary_from_pairs(pair_df)
    coords_df = orient_coordinates_3d(coords_df, judge_summary, args.orient_by)
    coords_df = coords_df.merge(judge_summary, on="judge", how="left")

    print("Calculating stress by dimension...")
    stress_df = stress_by_dimension(
        distance,
        max_dimensions=args.max_stress_dimensions,
        random_state=args.seed,
        metric=metric,
    )

    # Overwrite 3D stress with the actual plotted run.
    if 3 in set(stress_df["dimensions"]):
        stress_df.loc[stress_df["dimensions"] == 3, "stress"] = stress
        stress_df["stress_drop_from_previous"] = stress_df["stress"].shift(1) - stress_df["stress"]
        stress_df["pct_drop_from_previous"] = stress_df["stress_drop_from_previous"] / stress_df["stress"].shift(1)

    pair_distances = pairwise_coordinate_distances(coords_df)

    print("Writing outputs...")
    pair_df.to_csv(outdir / "mds_3d_pair_rows_used.csv", index=False)
    similarity.to_csv(outdir / "mds_3d_similarity_matrix.csv")
    distance.to_csv(outdir / "mds_3d_distance_matrix.csv")
    coords_df.to_csv(outdir / "mds_3d_coordinates.csv", index=False)
    stress_df.to_csv(outdir / "mds_3d_stress_by_dimension.csv", index=False)
    pair_distances.to_csv(outdir / "mds_3d_pairwise_distances.csv", index=False)

    if not missing_df.empty:
        missing_df.to_csv(outdir / "mds_3d_missing_distances.csv", index=False)

    subtitle = f"mode={args.mode}; min pair cases={args.min_pair_cases}; 3D stress={stress:.2f}"

    plot_3d_static(
        coords_df,
        outdir / "mds_3d_map.png",
        "SCC judge agreement 3D MDS map",
        subtitle,
    )

    plot_2d_projection(
        coords_df,
        "x",
        "y",
        outdir / "mds_3d_projection_xy.png",
        "3D MDS projection: dimensions 1 and 2",
        subtitle,
    )

    plot_2d_projection(
        coords_df,
        "x",
        "z",
        outdir / "mds_3d_projection_xz.png",
        "3D MDS projection: dimensions 1 and 3",
        subtitle,
    )

    plot_2d_projection(
        coords_df,
        "y",
        "z",
        outdir / "mds_3d_projection_yz.png",
        "3D MDS projection: dimensions 2 and 3",
        subtitle,
    )

    made_html = plot_3d_interactive_if_possible(
        coords_df,
        outdir / "mds_3d_interactive.html",
        "SCC judge agreement 3D MDS map",
        subtitle,
    )

    write_report(
        outdir / "mds_3d_report.txt",
        source_label=source_label,
        mode=args.mode,
        min_pair_cases=args.min_pair_cases,
        metric=metric,
        missing_df=missing_df,
        stress_df=stress_df,
        coords=coords_df,
        pair_distances=pair_distances,
        judge_summary=judge_summary,
    )

    print("")
    print(f"Saved 3D MDS outputs to: {outdir}")
    print("Key files:")
    print(f"  {outdir / 'mds_3d_map.png'}")
    print(f"  {outdir / 'mds_3d_projection_xy.png'}")
    print(f"  {outdir / 'mds_3d_projection_xz.png'}")
    print(f"  {outdir / 'mds_3d_projection_yz.png'}")
    print(f"  {outdir / 'mds_3d_coordinates.csv'}")
    print(f"  {outdir / 'mds_3d_pairwise_distances.csv'}")
    print(f"  {outdir / 'mds_3d_report.txt'}")
    if made_html:
        print(f"  {outdir / 'mds_3d_interactive.html'}")
    else:
        print("")
        print("Optional interactive 3D map was not created because plotly is not installed.")
        print("Install it with: python -m pip install plotly")


if __name__ == "__main__":
    main()
