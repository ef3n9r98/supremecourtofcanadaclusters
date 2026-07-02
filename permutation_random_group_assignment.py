#!/usr/bin/env python3
"""
permutation_random_group_assignment.py
======================================

Permutation test for SCC opinion-group agreement.

Question:
  If judges were randomly assigned to opinion groups within each case,
  while preserving each case's actual panel and opinion-group sizes,
  would we see pairwise agreement patterns as strong as the real ones?

Input:
  opinion_group_outputs_v6/judge_case_opinions.csv

Outputs:
  permutation_outputs/
    permutation_pair_results.csv
    permutation_global_results.csv
    observed_agreement_heatmap.png
    expected_agreement_heatmap.png
    z_score_heatmap.png
    p_value_heatmap.png
    permutation_report.txt

Recommended first run:
  python3.11 permutation_random_group_assignment.py \
    --input-dir opinion_group_outputs_v6 \
    --outdir permutation_outputs \
    --n-permutations 5000 \
    --confidence high medium

Optional split-only run:
  python3.11 permutation_random_group_assignment.py \
    --input-dir opinion_group_outputs_v6 \
    --outdir permutation_outputs_split_only \
    --n-permutations 5000 \
    --confidence high medium \
    --split-only

Interpretation:
  observed_agreement_rate:
    actual pairwise opinion-group agreement.

  expected_agreement_rate:
    average pairwise agreement under random assignment within each case.

  z_score:
    how far the observed pair agreement is from the random expectation.
    positive = pair agrees more than random.
    negative = pair agrees less than random.

  p_high:
    fraction of random simulations where agreement was >= observed.

  p_low:
    fraction of random simulations where agreement was <= observed.

  p_two_sided:
    two-sided empirical p-value using distance from the null mean.
"""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def last_name(full: str) -> str:
    return str(full).split(",", 1)[0]


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def empirical_p_two_sided(samples: np.ndarray, observed: float, expected: float) -> float:
    """
    Two-sided empirical p-value:
    proportion of permuted values at least as far from the null mean as observed.
    """
    dist_obs = abs(observed - expected)
    dist_samples = np.abs(samples - expected)
    return (np.sum(dist_samples >= dist_obs) + 1) / (len(samples) + 1)


def empirical_p_high(samples: np.ndarray, observed: float) -> float:
    """Right-tail p-value: observed unusually high agreement."""
    return (np.sum(samples >= observed) + 1) / (len(samples) + 1)


def empirical_p_low(samples: np.ndarray, observed: float) -> float:
    """Left-tail p-value: observed unusually low agreement."""
    return (np.sum(samples <= observed) + 1) / (len(samples) + 1)


# ---------------------------------------------------------------------------
# Loading and case construction
# ---------------------------------------------------------------------------

def load_judge_case_rows(input_dir: Path, confidence: list[str] | None) -> pd.DataFrame:
    path = input_dir / "judge_case_opinions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)

    required = {"case_id", "case_name", "year", "judge", "opinion_group", "extraction_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"judge_case_opinions.csv is missing columns: {sorted(missing)}")

    if confidence:
        df = df[df["extraction_confidence"].isin(confidence)].copy()

    # Remove rows where the parser could not assign the judge to an opinion group.
    df = df[df["opinion_group"].notna()].copy()
    df = df[df["opinion_group"].astype(str) != "UNASSIGNED"].copy()
    df = df[df["opinion_group"].astype(str).str.strip() != ""].copy()

    # Drop duplicate judge-case rows defensively.
    df = df.drop_duplicates(subset=["case_id", "judge"], keep="first").copy()

    return df


def build_cases(df: pd.DataFrame, split_only: bool) -> list[dict]:
    """
    Build a list of case dicts.

    Each case contains:
      case_id
      judges: list of judge names
      groups: list of group labels, same length as judges

    The null model preserves:
      - the exact judges in each case
      - the exact number and sizes of opinion groups in each case

    It randomizes:
      - which judge receives which group label within the case
    """
    cases = []

    for case_id, g in df.groupby("case_id"):
        judges = g["judge"].astype(str).tolist()
        groups = g["opinion_group"].astype(str).tolist()

        if len(judges) < 2:
            continue

        n_groups = len(set(groups))
        if split_only and n_groups <= 1:
            continue

        case_name = g["case_name"].iloc[0] if "case_name" in g.columns else ""
        year = g["year"].iloc[0] if "year" in g.columns else ""

        cases.append({
            "case_id": case_id,
            "case_name": case_name,
            "year": year,
            "judges": judges,
            "groups": groups,
            "n_judges": len(judges),
            "n_groups": n_groups,
            "group_sizes": tuple(sorted(pd.Series(groups).value_counts().tolist(), reverse=True)),
        })

    return cases


def build_pair_index(cases: list[dict]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
    pairs = set()
    for c in cases:
        for a, b in combinations(sorted(c["judges"]), 2):
            pairs.add((a, b))

    pair_list = sorted(pairs)
    pair_to_idx = {pair: i for i, pair in enumerate(pair_list)}
    return pair_list, pair_to_idx


# ---------------------------------------------------------------------------
# Observed and permuted agreement
# ---------------------------------------------------------------------------

def observed_counts(cases: list[dict], pair_to_idx: dict[tuple[str, str], int], n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      together_counts[pair] = cases where pair appeared together
      same_counts[pair] = cases where pair had same opinion group
    """
    together = np.zeros(n_pairs, dtype=int)
    same = np.zeros(n_pairs, dtype=int)

    for c in cases:
        judge_group = dict(zip(c["judges"], c["groups"]))
        for a, b in combinations(sorted(c["judges"]), 2):
            idx = pair_to_idx[(a, b)]
            together[idx] += 1
            if judge_group[a] == judge_group[b]:
                same[idx] += 1

    return together, same


def one_permutation_counts(cases: list[dict], pair_to_idx: dict[tuple[str, str], int], n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """
    One randomization of group assignments within each case.
    Returns same-counts for every judge pair.
    """
    same = np.zeros(n_pairs, dtype=int)

    for c in cases:
        judges = sorted(c["judges"])

        # The multiset of actual group labels preserves actual group sizes.
        original_groups_by_judge = dict(zip(c["judges"], c["groups"]))
        group_labels = np.array([original_groups_by_judge[j] for j in judges], dtype=object)

        # Randomly reassign those group labels to the same judges.
        shuffled_groups = rng.permutation(group_labels)
        perm_group = dict(zip(judges, shuffled_groups))

        for a, b in combinations(judges, 2):
            idx = pair_to_idx[(a, b)]
            if perm_group[a] == perm_group[b]:
                same[idx] += 1

    return same


def run_permutations(
    cases: list[dict],
    pair_to_idx: dict[tuple[str, str], int],
    together_counts: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """
    Returns:
      perm_rates: shape (n_permutations, n_pairs)
    """
    rng = np.random.default_rng(seed)
    n_pairs = len(pair_to_idx)

    perm_rates = np.full((n_permutations, n_pairs), np.nan, dtype=float)

    for p in range(n_permutations):
        perm_same = one_permutation_counts(cases, pair_to_idx, n_pairs, rng)
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_rates[p, :] = perm_same / together_counts

        if (p + 1) % max(1, n_permutations // 10) == 0:
            print(f"  completed {p + 1:,}/{n_permutations:,} permutations")

    return perm_rates


# ---------------------------------------------------------------------------
# Results and visualizations
# ---------------------------------------------------------------------------

def make_pair_results(
    pair_list: list[tuple[str, str]],
    together_counts: np.ndarray,
    observed_same: np.ndarray,
    perm_rates: np.ndarray,
    min_pair_cases: int,
) -> pd.DataFrame:
    observed_rate = observed_same / together_counts
    expected_rate = np.nanmean(perm_rates, axis=0)
    null_sd = np.nanstd(perm_rates, axis=0, ddof=1)

    rows = []

    for idx, (a, b) in enumerate(pair_list):
        if together_counts[idx] < min_pair_cases:
            continue

        obs = observed_rate[idx]
        exp = expected_rate[idx]
        sd = null_sd[idx]
        samples = perm_rates[:, idx]

        if pd.isna(obs) or pd.isna(exp):
            continue

        z = np.nan if sd == 0 or pd.isna(sd) else (obs - exp) / sd

        rows.append({
            "judge_a": a,
            "judge_b": b,
            "judge_a_short": last_name(a),
            "judge_b_short": last_name(b),
            "cases_together": int(together_counts[idx]),
            "observed_same_count": int(observed_same[idx]),
            "observed_agreement_rate": obs,
            "expected_agreement_rate": exp,
            "observed_minus_expected": obs - exp,
            "null_sd": sd,
            "z_score": z,
            "p_low": empirical_p_low(samples, obs),
            "p_high": empirical_p_high(samples, obs),
            "p_two_sided": empirical_p_two_sided(samples, obs, exp),
            "more_aligned_than_random": bool(obs > exp),
            "less_aligned_than_random": bool(obs < exp),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("observed_minus_expected", ascending=False)
    return out


def global_results(pair_results: pd.DataFrame, cases: list[dict], n_permutations: int, split_only: bool) -> pd.DataFrame:
    if pair_results.empty:
        return pd.DataFrame([{
            "cases_used": len(cases),
            "n_permutations": n_permutations,
            "split_only": split_only,
        }])

    return pd.DataFrame([{
        "cases_used": len(cases),
        "n_permutations": n_permutations,
        "split_only": split_only,
        "pair_rows_used": len(pair_results),
        "mean_observed_agreement": pair_results["observed_agreement_rate"].mean(),
        "mean_expected_agreement": pair_results["expected_agreement_rate"].mean(),
        "mean_observed_minus_expected": pair_results["observed_minus_expected"].mean(),
        "mean_abs_observed_minus_expected": pair_results["observed_minus_expected"].abs().mean(),
        "mean_z_score": pair_results["z_score"].mean(),
        "max_positive_z_score": pair_results["z_score"].max(),
        "max_negative_z_score": pair_results["z_score"].min(),
        "pairs_p_two_sided_below_0_10": int((pair_results["p_two_sided"] < 0.10).sum()),
        "pairs_p_two_sided_below_0_05": int((pair_results["p_two_sided"] < 0.05).sum()),
        "pairs_p_two_sided_below_0_01": int((pair_results["p_two_sided"] < 0.01).sum()),
    }])


def matrix_from_pair_results(pair_results: pd.DataFrame, value_col: str) -> pd.DataFrame:
    judges = sorted(set(pair_results["judge_a_short"]) | set(pair_results["judge_b_short"]))
    mat = pd.DataFrame(np.nan, index=judges, columns=judges)

    for j in judges:
        if value_col in {"observed_agreement_rate", "expected_agreement_rate"}:
            mat.loc[j, j] = 1.0
        elif value_col == "z_score":
            mat.loc[j, j] = 0.0

    for _, row in pair_results.iterrows():
        a = row["judge_a_short"]
        b = row["judge_b_short"]
        val = row[value_col]
        mat.loc[a, b] = val
        mat.loc[b, a] = val

    return mat


def order_by_mean_agreement(pair_results: pd.DataFrame) -> list[str]:
    mat = matrix_from_pair_results(pair_results, "observed_agreement_rate")
    return mat.mean(axis=1).sort_values(ascending=False).index.tolist()


def plot_matrix(
    mat: pd.DataFrame,
    out_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    percent_labels: bool = False,
    order: list[str] | None = None,
) -> None:
    if mat.empty:
        return

    if order:
        present = [j for j in order if j in mat.index]
        mat = mat.loc[present, present]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)

    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.iloc[i, j]
            if pd.notna(val):
                label = f"{val:.0%}" if percent_labels else f"{val:.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(out_path: Path, pair_results: pd.DataFrame, global_df: pd.DataFrame) -> None:
    lines = []

    lines.append("PERMUTATION TEST AGAINST RANDOM OPINION-GROUP ASSIGNMENT")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Null model:")
    lines.append("  Within each case, preserve the actual judges, number of opinion groups, and group sizes.")
    lines.append("  Randomly shuffle which judge is assigned to which opinion group.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Positive z-score = pair agrees more than random.")
    lines.append("  Negative z-score = pair agrees less than random.")
    lines.append("  p_low tests unusually low agreement.")
    lines.append("  p_high tests unusually high agreement.")
    lines.append("  p_two_sided tests unusual distance from random expectation in either direction.")
    lines.append("")

    if not global_df.empty:
        g = global_df.iloc[0]
        lines.append("Global summary:")
        lines.append(f"  Cases used: {g.get('cases_used', 'n/a')}")
        lines.append(f"  Permutations: {g.get('n_permutations', 'n/a')}")
        lines.append(f"  Split-only: {g.get('split_only', 'n/a')}")
        lines.append(f"  Pair rows used: {g.get('pair_rows_used', 'n/a')}")
        if "mean_observed_agreement" in g:
            lines.append(f"  Mean observed agreement: {g['mean_observed_agreement']:.1%}")
            lines.append(f"  Mean expected agreement: {g['mean_expected_agreement']:.1%}")
            lines.append(f"  Mean observed - expected: {g['mean_observed_minus_expected']:.1%}")
            lines.append(f"  Mean absolute observed - expected: {g['mean_abs_observed_minus_expected']:.1%}")
            lines.append(f"  Pairs with p_two_sided < 0.05: {g['pairs_p_two_sided_below_0_05']}")
        lines.append("")

    if not pair_results.empty:
        lines.append("Most more-aligned-than-random pairs:")
        top = pair_results.sort_values("z_score", ascending=False).head(10)
        for _, r in top.iterrows():
            lines.append(
                f"  {r['judge_a_short']} - {r['judge_b_short']}: "
                f"obs={r['observed_agreement_rate']:.1%}, "
                f"exp={r['expected_agreement_rate']:.1%}, "
                f"z={r['z_score']:.2f}, p_high={r['p_high']:.4f}"
            )
        lines.append("")

        lines.append("Most less-aligned-than-random pairs:")
        bottom = pair_results.sort_values("z_score", ascending=True).head(10)
        for _, r in bottom.iterrows():
            lines.append(
                f"  {r['judge_a_short']} - {r['judge_b_short']}: "
                f"obs={r['observed_agreement_rate']:.1%}, "
                f"exp={r['expected_agreement_rate']:.1%}, "
                f"z={r['z_score']:.2f}, p_low={r['p_low']:.4f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation test for SCC opinion-group agreement.")
    parser.add_argument("--input-dir", default="opinion_group_outputs_v6", help="Folder containing judge_case_opinions.csv")
    parser.add_argument("--outdir", default="permutation_outputs", help="Output folder")
    parser.add_argument("--n-permutations", type=int, default=5000, help="Number of random shuffles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-pair-cases", type=int, default=10, help="Minimum cases together for pair results")
    parser.add_argument("--confidence", nargs="*", default=None, help="Optional confidence filter, e.g. --confidence high medium")
    parser.add_argument("--split-only", action="store_true", help="Use only cases with more than one opinion group")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print("Loading judge-case opinion rows...")
    df = load_judge_case_rows(input_dir, args.confidence)

    print("Building case structures...")
    cases = build_cases(df, split_only=args.split_only)
    if not cases:
        raise ValueError("No usable cases after filtering.")

    pair_list, pair_to_idx = build_pair_index(cases)
    n_pairs = len(pair_list)

    print(f"Cases used: {len(cases):,}")
    print(f"Judge pairs observed: {n_pairs:,}")

    together, observed_same = observed_counts(cases, pair_to_idx, n_pairs)

    print(f"Running {args.n_permutations:,} permutations...")
    perm_rates = run_permutations(cases, pair_to_idx, together, args.n_permutations, args.seed)

    print("Building results...")
    pair_results = make_pair_results(
        pair_list=pair_list,
        together_counts=together,
        observed_same=observed_same,
        perm_rates=perm_rates,
        min_pair_cases=args.min_pair_cases,
    )
    global_df = global_results(pair_results, cases, args.n_permutations, args.split_only)

    pair_results.to_csv(outdir / "permutation_pair_results.csv", index=False)
    global_df.to_csv(outdir / "permutation_global_results.csv", index=False)

    order = order_by_mean_agreement(pair_results)

    observed_mat = matrix_from_pair_results(pair_results, "observed_agreement_rate")
    expected_mat = matrix_from_pair_results(pair_results, "expected_agreement_rate")
    z_mat = matrix_from_pair_results(pair_results, "z_score")
    p_mat = matrix_from_pair_results(pair_results, "p_two_sided")

    plot_matrix(
        observed_mat,
        outdir / "observed_agreement_heatmap.png",
        "Observed opinion-group agreement",
        "Observed agreement rate",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        percent_labels=True,
        order=order,
    )

    plot_matrix(
        expected_mat,
        outdir / "expected_agreement_heatmap.png",
        "Expected agreement under random group assignment",
        "Expected agreement rate",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        percent_labels=True,
        order=order,
    )

    # Symmetric color scale for z-scores.
    if not z_mat.empty:
        z_abs = np.nanmax(np.abs(z_mat.values))
        z_lim = max(1.0, min(5.0, float(z_abs)))
    else:
        z_lim = 3.0

    plot_matrix(
        z_mat,
        outdir / "z_score_heatmap.png",
        "Observed vs random z-scores",
        "Z-score",
        cmap="RdBu_r",
        vmin=-z_lim,
        vmax=z_lim,
        percent_labels=False,
        order=order,
    )

    plot_matrix(
        p_mat,
        outdir / "p_value_heatmap.png",
        "Two-sided empirical p-values",
        "p-value",
        cmap="YlGn_r",
        vmin=0,
        vmax=1,
        percent_labels=False,
        order=order,
    )

    write_report(outdir / "permutation_report.txt", pair_results, global_df)

    print("")
    print(f"Saved permutation outputs to: {outdir}")
    print("Key files:")
    print(f"  {outdir / 'permutation_report.txt'}")
    print(f"  {outdir / 'permutation_pair_results.csv'}")
    print(f"  {outdir / 'z_score_heatmap.png'}")
    print(f"  {outdir / 'observed_agreement_heatmap.png'}")
    print(f"  {outdir / 'expected_agreement_heatmap.png'}")


if __name__ == "__main__":
    main()
