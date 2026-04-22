# Migrated ML vs bio_age comparison summary

This is a summary-level migration of previous comparison outputs. Re-run compare_ml_runs_to_bio_age.py against a current bio_age run to produce per-subject diagnostics and figures.

## Main axes by subject_gap_mae

| ml_run | bio_age_axis | subject_gap_mae | subject_gain | subject_closer_to_bio_rate |
| --- | --- | ---: | ---: | ---: |
| run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | bio_age_ei_texture | 2.6807 | 6.5882 | 0.8522 |
| run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | bio_age_ei_texture | 2.7121 | 6.4964 | 0.8442 |
| run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | bio_age_ei_texture | 2.8827 | 6.3497 | 0.8385 |
| run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge | bio_age_texture | 2.8936 | 6.3753 | 0.8477 |
| run_20260410_175641_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_elasticnet | bio_age_texture | 2.9172 | 6.2912 | 0.8385 |
| run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_pls_regression | bio_age_texture | 3.0416 | 6.1908 | 0.8351 |
| run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_huber | bio_age_ei_texture | 3.4714 | 5.8045 | 0.7950 |
| run_20260402_151438_ta_healthy_nested_cv_elasticnet | bio_age_texture | 3.6078 | 5.8605 | 0.7995 |
| run_20260402_151438_ta_healthy_nested_cv_elasticnet | bio_age_ei_texture | 3.6858 | 5.7826 | 0.7869 |
| run_20260411_005829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_huber | bio_age_texture | 3.6961 | 5.5799 | 0.7789 |
| run_20260411_012913_ta_healthy_nested_cv_fusion_whole_roi_benchmark_xgboost | bio_age_texture | 3.9827 | 5.2122 | 0.7652 |
| run_20260411_010829_ta_healthy_nested_cv_fusion_whole_roi_benchmark_compact_hist_gradient_boosting | bio_age_texture | 4.0162 | 5.2880 | 0.7812 |
