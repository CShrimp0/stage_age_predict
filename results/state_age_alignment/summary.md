# state_age 特征基准实验总结

## 输入
- pred_age 输入: /home/szdx/LNX/usage_predict_feature_engineering/outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv
- 缓存特征表: /home/szdx/LNX/usage_predict_feature_engineering/outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv
- 图像根目录: /home/szdx/LNX/data/TA/Healthy/Images
- mask 根目录: /home/szdx/LNX/data/TA/Healthy/Masks

## 主结论
- Ridge 主榜第一: E3_texture_plus_metadata | pred_age vs state_age subject_gap_mae=2.2706, gain=6.9984
- 机器学习原始 MAE: sample=9.4431, subject=9.2689
- 最优方案 pred_age vs state_age gap MAE: sample=2.6805, subject=2.2706
- gap MAE 相比机器学习原始 MAE 的缩小量: sample=6.7626, subject=6.9984
- 最优 state_age 自身 vs true_age MAE 仅作 sanity check: sample=9.5711, subject=9.3376

## Top 5 (Ridge, 按 pred_age vs state_age 的 subject_gap_mae 升序)

| rank | feature_set | n_features | ml_subject_mae | subject_gap_mae | gain | sample_gap_mae | state_age_subject_mae | state_age_vs_true_corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | E3_texture_plus_metadata | 148 | 9.2689 | 2.2706 | 6.9984 | 2.6805 | 9.3376 | 0.5641 |
| 2 | B5_first_order_plus_texture | 168 | 9.2689 | 2.6807 | 6.5882 | 2.9686 | 9.4334 | 0.5444 |
| 3 | B4_texture_only | 144 | 9.2689 | 2.8936 | 6.3753 | 3.2406 | 9.4297 | 0.5430 |
| 4 | C5_first_order_texture_morphology | 193 | 9.2689 | 2.9328 | 6.3362 | 3.2771 | 9.1127 | 0.6528 |
| 5 | E4_full_plus_metadata | 317 | 9.2689 | 3.2162 | 6.0527 | 3.8041 | 8.8313 | 0.6863 |

## sanity check
- pred_age vs true_age MAE: sample=9.4431, subject=9.2689
- 最优方案 state_age vs true_age MAE: sample=9.5711, subject=9.3376
- 最优方案 state_age_vs_true_corr=0.5641
- 最优方案 state_age_std=13.8425
- 最优方案 state_age 范围=[18.8754, 446.7739]
- 未发现明显退化迹象。

## 特征构建备注
- roi:p10-> p5
- roi:p90-> p95
- whole_image:p10-> p5
- whole_image:p90-> p95

## 输出清单
- leaderboard.csv
- feature_sets/<feature_set>/results.csv
- figures/*.png
- inputs_used.json
