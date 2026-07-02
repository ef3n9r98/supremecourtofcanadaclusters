# Replicating “Canada’s Supreme Court Has Hidden Voting Blocs”

This README explains how to reproduce the Supreme Court of Canada voting-bloc analysis described in the Substack essay **“Canada’s Supreme Court Has Hidden Voting Blocs.”**

https://open.substack.com/pub/eugenefds/p/uncovering-judge-clusters-on-canadas

The analysis asks a narrow question:

> When Supreme Court of Canada judges sit on the same case, how often do they join the same set of reasons?

The key unit is **opinion-group agreement**, not simple win/loss agreement. If five judges join one set of reasons, three judges join a partial dissent, and one judge writes separately, the case has three opinion groups. Two judges “agree” for this project when they join the same opinion group.

## What this reproduces

The replication pipeline produces:

1. Structured opinion-group data from SCC judgments.
2. Pairwise judge-agreement rates.
3. Heatmaps for outcome, disposition-side, and opinion-group agreement.
4. A consensus baseline showing how often cases are unanimous or split.
5. A permutation test comparing observed agreement to random assignment within each case.
6. 2D and 3D MDS maps of judge agreement.
7. Exemplar cases used to interpret the MDS dimensions.
8. Subject-specific agreement tables.

The essay’s major interpretive findings were:

- **Dimension X** appears to capture disagreements about legal thresholds, remedies, and institutional boundaries.
- **Dimension Y** is the strongest dimension and appears to separate more formal-restraint reasoning from more purposive-remedial reasoning.
- **Dimension Z** is weaker and appears to capture something like Brown-style doctrinal architecture, but it is more vulnerable to timing and limited judge overlap.
- The Canadian Court is not well described by a simple one-dimensional “liberal/conservative” axis. Its disagreement structure is better understood as a legal space.

## Repository layout

Put the scripts and source data in one project folder:

```text
scc-voting-blocs/
  README.md
  scc_cases_2016-2026.jsonl
  extract_opinion_groups_v6.py
  make_agreement_heatmap.py
  consensus_baseline.py
  permutation_random_group_assignment.py
  mds_agreement_map.py
  mds_agreement_map_3d.py
  extract_mds_exemplar_cases.py
  subject_specific_agreement.py
```

The uploaded script filenames may include copy suffixes such as `(1)`, `(2)`, or `(3)`. For a clean replication, rename them to the simpler names shown above.

## Required input data

The main input file is:

```text
scc_cases_2016-2026.jsonl
```

Each line should be one SCC case as a JSON object. The extraction script expects at least these fields:

```text
citation_en
name_en
document_date_en
unofficial_text_en
```

The script also benefits from fields such as `citation2_en` or `url_en` if they exist, but those are not required.

### Important data note

The replication scripts do **not** include the raw SCC dataset. To reproduce the essay exactly, use the same JSONL dataset of SCC decisions from 2016–2026 and then filter the analysis to 2020–2026 during extraction.

## Python environment

Use Python 3.11 if possible.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn plotly
```

`plotly` is optional. If installed, the 3D MDS script can also write an interactive HTML map.

## Step 1: Extract SCC opinion groups

Run:

```bash
python extract_opinion_groups_v6.py scc_cases_2016-2026.jsonl \
  --outdir opinion_group_outputs_v6 \
  --start-year 2020 \
  --end-year 2026
```

This creates:

```text
opinion_group_outputs_v6/
  case_level.csv
  judge_case_opinions.csv
  judge_pair_case.csv
  judge_pair_summary.csv
  extraction_warnings.csv
```

The most important output files are:

- `judge_case_opinions.csv`: one row per judge per case.
- `judge_pair_case.csv`: one row per judge-pair per case.
- `judge_pair_summary.csv`: pairwise agreement rates across the full dataset.

The most important field is:

```text
same_opinion_group
```

This indicates whether two judges joined the same set of reasons in a case.

## Step 2: Make basic agreement heatmaps

Run:

```bash
python make_agreement_heatmap.py
```

This reads:

```text
opinion_group_outputs_v6/judge_pair_summary.csv
```

and writes:

```text
outcome_agreement_heatmap.png
disposition_agreement_heatmap.png
opinion_group_agreement_heatmap.png
```

These heatmaps compare three different kinds of agreement:

1. **Outcome agreement**: did the judges agree about who won?
2. **Disposition-side agreement**: did the judges take the same side in the disposition?
3. **Opinion-group agreement**: did the judges join the same written reasons?

For the essay, the opinion-group heatmap is the most important one.

## Step 3: Run the consensus baseline

Run:

```bash
python consensus_baseline.py \
  --input-dir opinion_group_outputs_v6 \
  --outdir consensus_baseline_outputs \
  --min-pair-cases 10 \
  --min-subject-cases 5
```

This creates:

```text
consensus_baseline_outputs/
  consensus_baseline_report.txt
  consensus_case_metrics.csv
  consensus_by_year.csv
  consensus_by_subject.csv
  judge_baseline.csv
  pairwise_baseline.csv
  several PNG charts
```

Use this step to check broad patterns such as:

- how often the Court is unanimous in reasons;
- how often it splits into multiple opinion groups;
- which judges most often join another judge’s reasons;
- which subjects have higher or lower consensus.

## Step 4: Run the permutation test

Raw pairwise agreement can be misleading because some cases are unanimous and others split sharply. The permutation test creates a fairer baseline.

Run:

```bash
python permutation_random_group_assignment.py \
  --input-dir opinion_group_outputs_v6 \
  --outdir permutation_outputs \
  --n-permutations 5000 \
  --confidence high medium \
  --min-pair-cases 10 \
  --seed 42
```

This creates:

```text
permutation_outputs/
  permutation_pair_results.csv
  permutation_global_results.csv
  observed_agreement_heatmap.png
  expected_agreement_heatmap.png
  z_score_heatmap.png
  p_value_heatmap.png
  permutation_report.txt
```

The null model preserves:

- the actual judges who sat on each case;
- the actual number of opinion groups in each case;
- the actual sizes of those opinion groups.

It randomizes:

- which judge is assigned to which opinion group within the case.

The key columns in `permutation_pair_results.csv` are:

```text
observed_agreement_rate
expected_agreement_rate
observed_minus_expected
z_score
p_high
p_low
p_two_sided
```

Interpretation:

- Positive `z_score`: the pair agreed more often than expected by random assignment.
- Negative `z_score`: the pair agreed less often than expected by random assignment.
- Very low `p_two_sided`: the pair’s observed relationship is unusual under the random baseline.

### Optional: split-only permutation test

To focus only on cases where the Court actually divided into more than one opinion group, run:

```bash
python permutation_random_group_assignment.py \
  --input-dir opinion_group_outputs_v6 \
  --outdir permutation_outputs_split_only \
  --n-permutations 5000 \
  --confidence high medium \
  --split-only \
  --min-pair-cases 10 \
  --seed 42
```

This is useful as a robustness check because unanimous cases otherwise contribute automatic agreement for every judge pair.

## Step 5: Create a 2D MDS map

Run:

```bash
python mds_agreement_map.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_outputs \
  --mode z \
  --min-pair-cases 10 \
  --seed 42
```

This creates:

```text
mds_outputs/
  mds_coordinates.csv
  mds_distance_matrix.csv
  mds_similarity_matrix.csv
  mds_stress_by_dimension.csv
  mds_agreement_map.png
  mds_report.txt
```

The recommended mode is:

```text
--mode z
```

That maps judges using how unexpectedly often or unexpectedly rarely they joined the same opinion groups, after accounting for case structure.

Other useful modes:

```bash
# Observed agreement only
python mds_agreement_map.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_outputs_observed \
  --mode observed \
  --min-pair-cases 10

# Observed agreement minus expected agreement
python mds_agreement_map.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_outputs_adjusted \
  --mode observed_minus_expected \
  --min-pair-cases 10
```

## Step 6: Create a 3D MDS map

Run:

```bash
python mds_agreement_map_3d.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_3d_outputs \
  --mode z \
  --min-pair-cases 10 \
  --seed 42
```

This creates:

```text
mds_3d_outputs/
  mds_3d_coordinates.csv
  mds_3d_distance_matrix.csv
  mds_3d_similarity_matrix.csv
  mds_3d_stress_by_dimension.csv
  mds_3d_map.png
  mds_3d_pairwise_distances.csv
  mds_3d_report.txt
  mds_3d_interactive.html   # if plotly is installed
```

The essay’s three dimensions come from this step.

Important: MDS does **not** name the dimensions. It only produces axes of separation. The legal interpretation comes later, by inspecting cases where judges at opposite ends of a dimension split into different opinion groups.

## Step 7: Extract exemplar cases for interpreting dimensions

Run:

```bash
python extract_mds_exemplar_cases.py \
  --coords mds_3d_outputs/mds_3d_coordinates.csv \
  --input-dir opinion_group_outputs_v6 \
  --jsonl scc_cases_2016-2026.jsonl \
  --outdir mds_exemplar_outputs \
  --top-n-judges 3 \
  --top-n-cases 20
```

This creates:

```text
mds_exemplar_outputs/
  dimension_judge_extremes.csv
  dimension_exemplar_cases.csv
  dimension_pair_exemplar_cases.csv
  mds_dimension_exemplar_report.txt
  dimension_x_exemplars.txt
  dimension_y_exemplars.txt
  dimension_z_exemplars.txt
```

This is the interpretive step. For each MDS dimension, the script:

1. finds judges at the high and low ends of the dimension;
2. finds cases where those judges sat together;
3. prioritizes cases where they split into different opinion groups;
4. pulls case metadata and readable excerpts;
5. writes reports for human review.

Use these outputs to support or revise the dimension labels.

## Step 8: Run subject-specific agreement analysis

Run:

```bash
python subject_specific_agreement.py \
  --input opinion_group_outputs_v6/judge_pair_case.csv \
  --outdir subject_agreement_outputs \
  --min-pair-subject-cases 5 \
  --min-subject-cases 10 \
  --confidence high medium
```

This creates:

```text
subject_agreement_outputs/
  subject_pair_agreement.csv
  subject_pair_deviations.csv
  subject_summary.csv
  judge_subject_summary.csv
  pair_subject_matrix_opinion_group.csv
  subject_split_heatmap.png
  most_subject_sensitive_pairs.csv
  subject_agreement_report.txt
```

This analysis asks whether a judge pair agrees unusually more or less in a subject area than they do overall.

Important caveat: cases can have multiple subject tags. The script expands multi-subject cases, so the same case can count under more than one subject. That is useful for exploration, but the subject rows are not mutually exclusive.

## How to verify the main essay results

After running the pipeline, check these files:

```text
opinion_group_outputs_v6/judge_pair_summary.csv
permutation_outputs/permutation_pair_results.csv
mds_3d_outputs/mds_3d_coordinates.csv
mds_3d_outputs/mds_3d_stress_by_dimension.csv
mds_exemplar_outputs/dimension_exemplar_cases.csv
mds_exemplar_outputs/mds_dimension_exemplar_report.txt
```

### To verify the agreement structure

Open:

```text
permutation_outputs/permutation_pair_results.csv
```

Sort by:

```text
z_score
observed_minus_expected
p_two_sided
```

This shows which judge pairs agreed more or less than expected under the random baseline.

### To verify the MDS dimensions

Open:

```text
mds_3d_outputs/mds_3d_coordinates.csv
```

For each axis `x`, `y`, and `z`, sort judges from low to high. Then compare those extremes to:

```text
mds_exemplar_outputs/dimension_exemplar_cases.csv
```

This lets you see which cases caused separation between the judges at opposite ends of each dimension.

### To verify the essay’s interpretive claims

Read the exemplar outputs and inspect the underlying cases.

The essay’s interpretation should be treated as a human reading of the generated exemplars, not as a label produced directly by the model.

Expected pattern:

- Dimension X should surface cases involving thresholds, remedies, Charter consequences, criminal procedure, and institutional boundaries.
- Dimension Y should repeatedly surface disagreements about formal legal categories versus broader statutory or constitutional purposes.
- Dimension Z should surface cases where Brown’s route through doctrine differs from other judges, even where outcomes sometimes overlap.

## Methodological caveats

This analysis is exploratory. Do not overclaim it.

Key limitations:

1. **Automated extraction can be wrong.** The opinion groups are extracted by regex and document structure, not by manual coding.
2. **Opinion-group agreement is not outcome agreement.** Two judges can agree on the result and disagree on the reasons.
3. **Some judges overlap less than others.** Appointment timing creates missing judge-pair relationships.
4. **MDS fills missing distances.** Filled distances are necessary for the map but should not be overinterpreted.
5. **MDS axes are not self-interpreting.** The labels “Dimension X,” “Dimension Y,” and “Dimension Z” come from reading exemplar cases.
6. **Subject tags are not mutually exclusive.** A case can appear under multiple subjects.
7. **The dataset window matters.** The essay focuses on 2020–2026. Changing the date range may change the map.

## Suggested robustness checks

Run these before publishing strong claims:

```bash
# 1. Re-run extraction and inspect warnings
open opinion_group_outputs_v6/extraction_warnings.csv

# 2. Run split-only permutation test
python permutation_random_group_assignment.py \
  --input-dir opinion_group_outputs_v6 \
  --outdir permutation_outputs_split_only \
  --n-permutations 5000 \
  --confidence high medium \
  --split-only

# 3. Run MDS on split-only results
python mds_agreement_map_3d.py \
  --pair-results permutation_outputs_split_only/permutation_pair_results.csv \
  --outdir mds_3d_outputs_split_only \
  --mode z \
  --min-pair-cases 10

# 4. Compare observed-only, adjusted, and z-score MDS maps
python mds_agreement_map_3d.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_3d_outputs_observed \
  --mode observed \
  --min-pair-cases 10

python mds_agreement_map_3d.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_3d_outputs_adjusted \
  --mode observed_minus_expected \
  --min-pair-cases 10
```

If the same broad judge separations and exemplar patterns appear across these checks, the findings are more credible.

## Minimal one-command pipeline

After the raw JSONL file and scripts are in place, this sequence reproduces the main analysis:

```bash
python extract_opinion_groups_v6.py scc_cases_2016-2026.jsonl \
  --outdir opinion_group_outputs_v6 \
  --start-year 2020 \
  --end-year 2026

python make_agreement_heatmap.py

python consensus_baseline.py \
  --input-dir opinion_group_outputs_v6 \
  --outdir consensus_baseline_outputs

python permutation_random_group_assignment.py \
  --input-dir opinion_group_outputs_v6 \
  --outdir permutation_outputs \
  --n-permutations 5000 \
  --confidence high medium

python mds_agreement_map.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_outputs \
  --mode z \
  --min-pair-cases 10

python mds_agreement_map_3d.py \
  --pair-results permutation_outputs/permutation_pair_results.csv \
  --outdir mds_3d_outputs \
  --mode z \
  --min-pair-cases 10

python extract_mds_exemplar_cases.py \
  --coords mds_3d_outputs/mds_3d_coordinates.csv \
  --input-dir opinion_group_outputs_v6 \
  --jsonl scc_cases_2016-2026.jsonl \
  --outdir mds_exemplar_outputs \
  --top-n-judges 3 \
  --top-n-cases 20

python subject_specific_agreement.py \
  --input opinion_group_outputs_v6/judge_pair_case.csv \
  --outdir subject_agreement_outputs \
  --min-pair-subject-cases 5 \
  --min-subject-cases 10 \
  --confidence high medium
```

## Plain-English summary

The pipeline works like this:

1. Turn SCC judgments into structured judge/opinion-group rows.
2. Count how often each pair of judges joins the same reasons.
3. Compare that observed agreement to a random baseline that preserves each case’s real split structure.
4. Turn unexpected agreement and disagreement into distances between judges.
5. Use MDS to place judges in a 2D or 3D space.
6. Read exemplar cases at the extremes of each dimension to figure out what the axes might mean.

The statistics create the map. The law gives the map meaning.
