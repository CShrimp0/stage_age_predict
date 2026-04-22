# 超声年龄模型的 Bio Age 参考轴分析

This repository builds interpretable ultrasound-derived `bio_age` reference
axes and tests whether ML-predicted age (`pred_age`) is closer to those
reference axes than to chronological label age (`true_age`).

The goal is not to find a single globally best chronological-age predictor.
The goal is to create several explainable biological/status-age proxies and
use them to understand what image signal the ML model learned.

## 核心术语

- `true_age`: chronological / label age.
- `pred_age`: machine-learning predicted age.
- `bio_age`: interpretable image-derived biological/status age proxy.

Backward compatibility: some loaders still accept old input columns such as
`age` or `state_age`, but outputs are normalized to `true_age`, `pred_age`, and
`bio_age`.

## 主 Bio Age 参考轴

The benchmark formally keeps three primary axes:

- `bio_age_ei`
  - EI / first-order axis.
  - Represents overall echogenicity, brightness, and coarse muscle-quality
    signal.
  - Uses ROI first-order statistics such as mean, median, std, iqr, percentiles,
    skewness, and kurtosis.

- `bio_age_texture`
  - Texture-only axis.
  - Represents tissue heterogeneity and local texture-pattern signal.
  - Uses lightweight interpretable texture descriptors such as GLCM and LBP
    features when available.

- `bio_age_ei_texture`
  - Pure-image combined axis.
  - Represents the recommended image-only status-age proxy combining EI /
    first-order features and texture features.

Supplemental axes such as `bio_age_morphology`, `bio_age_texture_metadata`, and
`bio_age_full_image_upper_bound` are retained for interpretation and practical
upper-bound checks. `texture_metadata` and full feature sets should not be used
as the sole scientific definition of `bio_age`.

## 为什么不能只按最优 MAE 排名？

Lowest global MAE can be misleading. It does not prove that every subject or
sample is close to a bio-age axis, and it may reward complex feature sets with
weaker interpretability.

The central questions are:

1. Is `pred_age` closer to `bio_age_ei`, `bio_age_texture`, or
   `bio_age_ei_texture`?
2. Does that closeness hold for most subjects/samples, or only because a few
   large errors dominate the mean?
3. If `pred_age` aligns with `bio_age_ei`, the ML model may rely more on
   global echogenicity. If it aligns with `bio_age_texture`, it may rely more
   on tissue heterogeneity. If it aligns with `bio_age_ei_texture`, it may
   capture a combined pure-image status-age signal.

## 指标体系

For each ML run and `bio_age` reference axis:

```text
sample_ml_true_mae = MAE(pred_age, true_age)
subject_ml_true_mae = MAE(pred_age, true_age)
sample_gap_mae = MAE(pred_age, bio_age)
subject_gap_mae = MAE(pred_age, bio_age)
sample_gain = sample_ml_true_mae - sample_gap_mae
subject_gain = subject_ml_true_mae - subject_gap_mae
```

Coverage metrics:

```text
sample_closer_to_bio_rate =
  fraction of samples where |pred_age - bio_age| < |pred_age - true_age|

subject_closer_to_bio_rate =
  fraction of subjects where |pred_age - bio_age| < |pred_age - true_age|

sample_within_2/5/8_rate =
  fraction of samples where |pred_age - bio_age| <= 2/5/8

subject_within_2/5/8_rate =
  fraction of subjects where |pred_age - bio_age| <= 2/5/8
```

Bio-age sanity checks:

```text
bio_age_vs_true_mae
bio_age_vs_true_corr
bio_age_std
bio_age_min
bio_age_max
bio_age_bias_slope
```

These sanity checks assess whether the proxy is reasonable. They are not the
sole ranking target.

## 仓库结构

- `ei_state_age_agegap/`
  - EI-only `bio_age` baseline tool.
- `state_age_feature_benchmark/run_state_age_feature_benchmark.py`
  - Fits multiple interpretable `bio_age` reference axes under subject-level CV.
  - The directory name is kept for backward compatibility with earlier local
    runs; current outputs use `bio_age` terminology.
- `state_age_feature_benchmark/build_single_ml_report.py`
  - 从一个 benchmark run 生成一份清晰的“单个 ML 实验 vs 多条 bio_age 轴”报告。
  - 这是当前最推荐直接阅读的结果入口。
- `state_age_feature_benchmark/compare_ml_runs_to_bio_age.py`
  - Compares many ML prediction runs against fitted `bio_age` axes.
  - 这是后续进阶分析入口，不是当前首读路径。
- `state_age_feature_benchmark/compare_ml_runs_to_state_age.py`
  - Backward-compatible wrapper around `compare_ml_runs_to_bio_age.py`.
- `results/`
  - Summary-level example outputs. Raw images, full feature tables, and
    per-sample private prediction files are not included.

## 推荐工作流：先看单个 ML 实验

第一步，拟合多条 `bio_age` 参考轴，并自动生成一份清晰单报告：

```bash
python state_age_feature_benchmark/run_state_age_feature_benchmark.py \
  --pred-file /home/szdx/LNX/usage_predict_feature_engineering/outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv \
  --feature-table /home/szdx/LNX/usage_predict_feature_engineering/outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv \
  --output-root results/bio_age_feature_benchmark
```

For a faster smoke run that skips per-image extra feature extraction and still
keeps the main `bio_age_ei / bio_age_texture / bio_age_ei_texture` axes:

```bash
python state_age_feature_benchmark/run_state_age_feature_benchmark.py \
  --pred-file /home/szdx/LNX/usage_predict_feature_engineering/outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv \
  --feature-table /home/szdx/LNX/usage_predict_feature_engineering/outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv \
  --output-root results/bio_age_feature_benchmark \
  --run-name run_quick \
  --skip-extra-image-features
```

Primary outputs:

```text
results/bio_age_feature_benchmark/<run_name>/
  bio_age_reference_leaderboard.csv
  bio_age_reference_summary.md
  feature_sets/
    bio_age_ei/
    bio_age_texture/
    bio_age_ei_texture/
    ...
  figures/
```

同时会自动产出一份更适合阅读的单实验报告：

```bash
results/reports/single_ml/<ml_run_name>/
  summary.md
  tables/
    main_axes_overview.csv
    upper_bound_axes_overview.csv
    subject_error_matrix_main_axes.csv
    worst_subjects_main_axes.csv
  figures/
```

如果你已经有一个 benchmark run，也可以单独重新渲染报告：

```bash
python state_age_feature_benchmark/build_single_ml_report.py \
  --bio-age-run results/bio_age_feature_benchmark/<run_name> \
  --output-root results/reports/single_ml
```

## 进阶：多个 ML runs 对比

等单实验报告看清楚之后，再做多 run 对比：

```bash
python state_age_feature_benchmark/compare_ml_runs_to_bio_age.py \
  --bio-age-run results/bio_age_feature_benchmark/<run_name> \
  --pred-root /home/szdx/LNX/usage_predict_feature_engineering/outputs \
  --output-dir results/ml_vs_bio_age/<comparison_name>
```

## EI-only Bio Age 基线

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input path/to/predictions_with_ei.csv \
  --ei-column EI \
  --output-dir results/ei_bio_age_agegap
```

## 验证

```bash
python -m py_compile \
  ei_state_age_agegap/compute_agegap.py \
  state_age_feature_benchmark/run_state_age_feature_benchmark.py \
  state_age_feature_benchmark/compare_ml_runs_to_bio_age.py \
  state_age_feature_benchmark/compare_ml_runs_to_state_age.py
```
