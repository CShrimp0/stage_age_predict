# ML Runs vs State Age Comparison

Purpose: test whether ML-predicted age is closer to interpretable feature-derived state_age than to chronological true_age.

- state_age_run: outputs/state_age_feature_benchmark/run_20260422_141500_state_age_feature_benchmark_state_alignment
- pred_root: outputs
- prediction runs compared: 15
- state_age feature sets compared: 23

## Top 10 by subject_gap_mae

| rank | ml_run | feature_set | n_features | ml_subject_mae | subject_gap_mae | subject_gain | subject_closer_to_state_rate | state_age_subject_mae |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | E3_texture_plus_metadata | 148 | 9.2084 | 2.2668 | 6.9416 | 0.8568 | 9.3376 |
| 2 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | E3_texture_plus_metadata | 148 | 9.2689 | 2.2706 | 6.9984 | 0.8717 | 9.3376 |
| 3 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | E3_texture_plus_metadata | 148 | 9.2324 | 2.3760 | 6.8564 | 0.8454 | 9.3376 |
| 4 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | B5_first_order_plus_texture | 168 | 9.2689 | 2.6807 | 6.5882 | 0.8522 | 9.4334 |
| 5 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | B5_first_order_plus_texture | 168 | 9.2084 | 2.7121 | 6.4964 | 0.8442 | 9.4334 |
| 6 | run_20260402_151438_ta_healthy_nested_cv_elasticnet | E3_texture_plus_metadata | 148 | 9.4683 | 2.7663 | 6.7020 | 0.8373 | 9.3376 |
| 7 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | B5_first_order_plus_texture | 168 | 9.2324 | 2.8827 | 6.3497 | 0.8385 | 9.4334 |
| 8 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | B4_texture_only | 144 | 9.2689 | 2.8936 | 6.3753 | 0.8477 | 9.4297 |
| 9 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | C5_first_order_texture_morphology | 193 | 9.2084 | 2.9046 | 6.3038 | 0.8225 | 9.1127 |
| 10 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | B4_texture_only | 144 | 9.2084 | 2.9172 | 6.2912 | 0.8385 | 9.4297 |

## Top 10 by subject_closer_to_state_rate

| rank | ml_run | feature_set | n_features | ml_subject_mae | subject_gap_mae | subject_gain | subject_closer_to_state_rate | state_age_subject_mae |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | E3_texture_plus_metadata | 148 | 9.2689 | 2.2706 | 6.9984 | 0.8717 | 9.3376 |
| 2 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | E3_texture_plus_metadata | 148 | 9.2084 | 2.2668 | 6.9416 | 0.8568 | 9.3376 |
| 3 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | B5_first_order_plus_texture | 168 | 9.2689 | 2.6807 | 6.5882 | 0.8522 | 9.4334 |
| 4 | run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | B4_texture_only | 144 | 9.2689 | 2.8936 | 6.3753 | 0.8477 | 9.4297 |
| 5 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | E3_texture_plus_metadata | 148 | 9.2324 | 2.3760 | 6.8564 | 0.8454 | 9.3376 |
| 6 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | B5_first_order_plus_texture | 168 | 9.2084 | 2.7121 | 6.4964 | 0.8442 | 9.4334 |
| 7 | run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | B5_first_order_plus_texture | 168 | 9.2324 | 2.8827 | 6.3497 | 0.8385 | 9.4334 |
| 8 | run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | B4_texture_only | 144 | 9.2084 | 2.9172 | 6.2912 | 0.8385 | 9.4297 |
| 9 | run_20260402_151438_ta_healthy_nested_cv_elasticnet | E3_texture_plus_metadata | 148 | 9.4683 | 2.7663 | 6.7020 | 0.8373 | 9.3376 |
| 10 | run_20260410_173622_ta_healthy_nested_cv_roi_only_masked_elasticnet | C4_first_order_plus_morphology | 37 | 10.4199 | 3.4724 | 6.9475 | 0.8362 | 10.2372 |

Interpretation:
- subject_gain = ml_subject_mae - subject_gap_mae.
- subject_gain > 0 means ML pred_age is closer to state_age than to true_age at subject level.
- subject_closer_to_state_rate reports the fraction of subjects where this is true, not just the average.
