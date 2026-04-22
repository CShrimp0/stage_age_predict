# ML runs vs bio_age comparison

Purpose: test whether ML-predicted `pred_age` is closer to interpretable feature-derived `bio_age` reference axes than to `true_age`.

- bio_age_run: results/bio_age_feature_benchmark/run_20260422_real_bio_age_benchmark_quick
- pred_root: /home/szdx/LNX/usage_predict_feature_engineering/outputs
- prediction runs compared: 15
- bio_age feature sets compared: 11

## Main-axis top 10 by subject_gap_mae

| rank | ml_run | bio_age_axis | n_features | subject_ml_true_mae | subject_gap_mae | subject_gain | subject_closer_to_bio_rate | subject_within_2_rate | subject_within_5_rate | subject_within_8_rate | bio_age_subject_mae |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | bio_age_ei_texture | 168 | 9.2689 | 2.6807 | 6.5882 | 0.8522 | 0.4868 | 0.8809 | 0.9771 | 9.4334 |
| 2 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | bio_age_ei_texture | 168 | 9.2084 | 2.7121 | 6.4964 | 0.8442 | 0.4662 | 0.8774 | 0.9714 | 9.4334 |
| 3 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | bio_age_ei_texture | 168 | 9.2324 | 2.8827 | 6.3497 | 0.8385 | 0.4158 | 0.8408 | 0.9668 | 9.4334 |
| 4 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | bio_age_texture | 144 | 9.2689 | 2.8936 | 6.3753 | 0.8477 | 0.4765 | 0.8740 | 0.9656 | 9.4297 |
| 5 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | bio_age_texture | 144 | 9.2084 | 2.9172 | 6.2912 | 0.8385 | 0.4765 | 0.8534 | 0.9611 | 9.4297 |
| 6 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | bio_age_texture | 144 | 9.2324 | 3.0416 | 6.1908 | 0.8351 | 0.4238 | 0.8385 | 0.9553 | 9.4297 |
| 7 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_huber | bio_age_ei_texture | 168 | 9.2759 | 3.4714 | 5.8045 | 0.7950 | 0.3780 | 0.7938 | 0.9427 | 9.4334 |
| 8 | run_20260402_151438_ta_healthy_nested_cv_elasticnet | bio_age_texture | 144 | 9.4683 | 3.6078 | 5.8605 | 0.7995 | 0.3700 | 0.7583 | 0.9221 | 9.4297 |
| 9 | run_20260402_151438_ta_healthy_nested_cv_elasticnet | bio_age_ei_texture | 168 | 9.4683 | 3.6858 | 5.7826 | 0.7869 | 0.3608 | 0.7583 | 0.9290 | 9.4334 |
| 10 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_huber | bio_age_texture | 144 | 9.2759 | 3.6961 | 5.5799 | 0.7789 | 0.3757 | 0.7812 | 0.9255 | 9.4297 |

## Main-axis top 10 by subject_closer_to_bio_rate

| rank | ml_run | bio_age_axis | n_features | subject_ml_true_mae | subject_gap_mae | subject_gain | subject_closer_to_bio_rate | bio_age_subject_mae |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | bio_age_ei_texture | 168 | 9.2689 | 2.6807 | 6.5882 | 0.8522 | 9.4334 |
| 2 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | bio_age_texture | 144 | 9.2689 | 2.8936 | 6.3753 | 0.8477 | 9.4297 |
| 3 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | bio_age_ei_texture | 168 | 9.2084 | 2.7121 | 6.4964 | 0.8442 | 9.4334 |
| 4 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | bio_age_ei_texture | 168 | 9.2324 | 2.8827 | 6.3497 | 0.8385 | 9.4334 |
| 5 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | bio_age_texture | 144 | 9.2084 | 2.9172 | 6.2912 | 0.8385 | 9.4297 |
| 6 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | bio_age_texture | 144 | 9.2324 | 3.0416 | 6.1908 | 0.8351 | 9.4297 |
| 7 | run_20260402_151438_ta_healthy_nested_cv_elasticnet | bio_age_texture | 144 | 9.4683 | 3.6078 | 5.8605 | 0.7995 | 9.4297 |
| 8 | run_20260401_173642_ta_healthy_nested_cv_ridge | bio_age_texture | 144 | 9.7073 | 4.1952 | 5.5120 | 0.7984 | 9.4297 |
| 9 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_huber | bio_age_ei_texture | 168 | 9.2759 | 3.4714 | 5.8045 | 0.7950 | 9.4334 |
| 10 | run_20260401_173843_ta_healthy_nested_cv_elasticnet | bio_age_texture | 144 | 9.6905 | 4.1880 | 5.5025 | 0.7938 | 9.4297 |

Interpretation:
- `subject_gain = subject_ml_true_mae - subject_gap_mae`。
- subject_gain > 0 means ML pred_age is closer to bio_age than to true_age at subject level.
- subject_closer_to_bio_rate reports the fraction of subjects where this is true, not just the average.
- `bio_age_texture_metadata` and full feature sets are practical upper bounds, not the sole scientific definition.
