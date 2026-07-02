import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT = "opinion_group_outputs_v6/judge_pair_summary.csv"
MIN_CASES = 10

df = pd.read_csv(INPUT)

# Shorten judge names to last names
def last_name(full):
    return str(full).split(",")[0]

df["judge_a_short"] = df["judge_a"].apply(last_name)
df["judge_b_short"] = df["judge_b"].apply(last_name)

# Use one common judge ordering for all heatmaps so they are comparable.
# We'll base that ordering on opinion-group agreement.
df_order = df[df["opinion_group_known_cases"] >= MIN_CASES].copy()

judges = sorted(set(df["judge_a_short"]) | set(df["judge_b_short"]))

order_mat = pd.DataFrame(np.nan, index=judges, columns=judges)
for j in judges:
    order_mat.loc[j, j] = 1.0

for _, row in df_order.iterrows():
    a = row["judge_a_short"]
    b = row["judge_b_short"]
    val = row["opinion_group_agreement_rate"]
    if pd.notna(val):
        order_mat.loc[a, b] = val
        order_mat.loc[b, a] = val

common_order = order_mat.mean(axis=1).sort_values(ascending=False).index.tolist()


def build_matrix(dataframe, rate_col):
    judges_local = sorted(set(dataframe["judge_a_short"]) | set(dataframe["judge_b_short"]))
    mat = pd.DataFrame(np.nan, index=judges_local, columns=judges_local)

    for j in judges_local:
        mat.loc[j, j] = 1.0

    for _, row in dataframe.iterrows():
        a = row["judge_a_short"]
        b = row["judge_b_short"]
        val = row[rate_col]

        if pd.notna(val):
            mat.loc[a, b] = val
            mat.loc[b, a] = val

    # Reindex to the common order, but only keep judges present in this matrix
    present = [j for j in common_order if j in mat.index]
    mat = mat.loc[present, present]
    return mat


def plot_heatmap(rate_col, n_col, title, colorbar_label, output_file):
    subset = df[df[n_col] >= MIN_CASES].copy()
    mat = build_matrix(subset, rate_col)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat.values, vmin=0, vmax=1, aspect="auto", cmap="YlGn")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)

    # Add percentage labels
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j, i, f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(f"{title}\n(only pairs with at least {MIN_CASES} known cases)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved heatmap to {output_file}")


# 1. Outcome agreement
plot_heatmap(
    rate_col="outcome_agreement_rate",
    n_col="outcome_known_cases",
    title="SCC Judge Pair Outcome Agreement",
    colorbar_label="Outcome agreement rate",
    output_file="outcome_agreement_heatmap.png"
)

# 2. Disposition-side agreement
plot_heatmap(
    rate_col="disposition_agreement_rate",
    n_col="disposition_known_cases",
    title="SCC Judge Pair Disposition-Side Agreement",
    colorbar_label="Disposition-side agreement rate",
    output_file="disposition_agreement_heatmap.png"
)

# 3. Opinion-group agreement
plot_heatmap(
    rate_col="opinion_group_agreement_rate",
    n_col="opinion_group_known_cases",
    title="SCC Judge Pair Opinion-Group Agreement",
    colorbar_label="Opinion-group agreement rate",
    output_file="opinion_group_agreement_heatmap.png"
)