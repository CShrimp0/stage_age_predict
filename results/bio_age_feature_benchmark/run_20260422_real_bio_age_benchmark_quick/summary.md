# bio_age 参考轴拟合结果汇总

## 输入
- pred_age 输入: /home/szdx/LNX/usage_predict_feature_engineering/outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv
- 缓存特征表: /home/szdx/LNX/usage_predict_feature_engineering/outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv
- 图像根目录: /home/szdx/LNX/data/TA/Healthy/Images
- mask 根目录: /home/szdx/LNX/data/TA/Healthy/Masks

## 目的
- 本实验不以找到单一最低 MAE 年龄预测器为目标。
- `bio_age` 是 interpretable image-derived biological/status age proxy，用于建立多条可解释参考轴。
- 主问题是 `pred_age` 更接近哪条 `bio_age` 轴，而不是哪组特征最会预测 `true_age`。

## 主参考轴
- `bio_age_ei`: EI / 一阶统计轴：整体回声强度、亮度与粗粒度肌肉质量信号。
- `bio_age_texture`: 纹理轴：组织异质性与局部纹理模式信号。
- `bio_age_ei_texture`: 纯图像综合轴：一阶统计 + 纹理信号。

## 主参考轴（Ridge）

| bio_age_axis | n_features | sample_gap_mae | subject_gap_mae | sample_gain | subject_gain | sample_closer_to_bio_rate | subject_closer_to_bio_rate | bio_age_vs_true_mae | bio_age_vs_true_corr | bio_age_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bio_age_ei | 12 | 5.9384 | 5.5691 | 3.5047 | 3.6998 | 0.6562 | 0.6735 | 10.9201 | 0.5522 | 9.0302 |
| bio_age_texture | 144 | 3.2406 | 2.8936 | 6.2024 | 6.3753 | 0.8211 | 0.8477 | 9.6791 | 0.5430 | 13.8336 |
| bio_age_ei_texture | 168 | 2.9686 | 2.6807 | 6.4745 | 6.5882 | 0.8275 | 0.8522 | 9.6498 | 0.5444 | 13.8469 |

## 补充轴 / upper bound

| bio_age_axis | role | n_features | subject_gap_mae | subject_gain | subject_closer_to_bio_rate | bio_age_subject_mae |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bio_age_full_image_upper_bound | 实践上限轴：更宽的图像、形态、分区和元数据组合。 | 185 | 1.8564 | 7.4125 | 0.8958 | 9.0478 |
| bio_age_texture_metadata | 实践上限轴：纹理 + 简单元数据。 | 148 | 2.2706 | 6.9984 | 0.8717 | 9.3376 |
| bio_age_morphology | 补充形态轴：ROI 形状与空间分布描述。 | 13 | 9.0727 | 0.1962 | 0.4971 | 13.0580 |

## 合理性检查 / 中文说明
- pred_age 与 true_age 的 MAE: sample=9.4431, subject=9.2689
- `bio_age_vs_true_mae` / `bio_age_vs_true_corr` 只用于检查参考轴是否合理，不作为唯一主结论。
- `texture_metadata` 和 full feature set 只作为 practical upper bound，不应作为唯一科学定义。

## 跳过项
- supplement_depthnorm_ei: no available columns
- supplement_partition_depth: no available columns
- supplement_partition_width: no available columns
- supplement_partition_texture: no available columns

## 特征构建备注
- roi:p10-> p5
- roi:p90-> p95
- whole_image:p10-> p5
- whole_image:p90-> p95

## 图像+mask 现算备注
- Skipped per-image extra feature extraction via --skip-extra-image-features.

## 输出清单
- bio_age_reference_leaderboard.csv
- bio_age_reference_summary.md
- bio_age_reference_subject_diagnostics.csv
- bio_age_reference_subject_error_matrix.csv
- feature_sets/<feature_set>/results.csv
- figures/*.png
- inputs_used.json
