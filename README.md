# Status Age Prediction

This repository contains the standalone status-age analysis code split out from
`usage_predict_feature_engineering`.

The goal is not to find the best chronological-age predictor. The goal is to
define interpretable feature-derived `state_age`, then test whether ML-predicted
age is closer to that state age than to chronological `true_age`.

## Contents

- `ei_state_age_agegap/`
  - Direct EI-to-state-age mapping and `pred_age - state_age` gap MAE.
- `state_age_feature_benchmark/run_state_age_feature_benchmark.py`
  - Fits multiple interpretable feature sets to `state_age` under subject-level
    CV and evaluates `pred_age` vs `state_age`.
- `state_age_feature_benchmark/compare_ml_runs_to_state_age.py`
  - Compares many ML prediction runs against the same set of fitted state-age
    outputs.
- `dataio/` and `preprocessing/`
  - Minimal copied helper code needed by the benchmark script.
- `results/`
  - Small summary outputs from the latest local runs. Raw images, full feature
    tables, and per-sample prediction files are not included.

## Core Metrics

For each ML run and state-age feature set:

```text
ml_subject_mae = MAE(pred_age, true_age)
subject_gap_mae = MAE(pred_age, state_age)
subject_gain = ml_subject_mae - subject_gap_mae
subject_closer_to_state_rate =
    fraction of subjects where |pred_age - state_age| < |pred_age - true_age|
```

Interpretation:

- `subject_gain > 0` means the ML prediction is closer to interpretable
  `state_age` than chronological age on average.
- `subject_closer_to_state_rate` checks whether this is true for many subjects,
  not only due to a few large errors.
- `state_age_subject_mae` and `state_age_vs_true_corr` are sanity checks for
  whether the fitted state age is reasonable; they are not the main ranking
  target.

## Run A State-Age Feature Benchmark

Provide the prediction file and feature table explicitly. Example for the local
LNX workspace:

```bash
python state_age_feature_benchmark/run_state_age_feature_benchmark.py \
  --pred-file /home/szdx/LNX/usage_predict_feature_engineering/outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv \
  --feature-table /home/szdx/LNX/usage_predict_feature_engineering/outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv \
  --output-root outputs/state_age_feature_benchmark
```

The script also reads image/mask paths from the feature table when deriving
extra interpretable features. If those paths are not available, feature sets
that depend on them may be skipped or partially missing.

## Compare Many ML Runs Against State Age

After running the benchmark above:

```bash
python state_age_feature_benchmark/compare_ml_runs_to_state_age.py \
  --state-age-run outputs/state_age_feature_benchmark/<run_name> \
  --pred-root /home/szdx/LNX/usage_predict_feature_engineering/outputs \
  --output-dir outputs/state_age_model_comparison/<comparison_name>
```

This produces `ml_vs_state_age_leaderboard.csv` and `summary.md`.

## EI-Only State Age Gap

If a table already contains `EI` and predicted age:

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input path/to/predictions_with_ei.csv \
  --ei-column EI \
  --output-dir outputs/ei_state_age_agegap
```

If EI is in a separate table:

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input path/to/predictions.csv \
  --ei-source path/to/ei_table.xlsx \
  --ei-sheet 左连接 \
  --ei-column "股直肌亮度（静息）" \
  --merge-key subject_id:ID \
  --output-dir outputs/ei_state_age_agegap
```

## Validation

The current code was syntax-checked with:

```bash
python -m py_compile \
  ei_state_age_agegap/compute_agegap.py \
  state_age_feature_benchmark/run_state_age_feature_benchmark.py \
  state_age_feature_benchmark/compare_ml_runs_to_state_age.py
```

