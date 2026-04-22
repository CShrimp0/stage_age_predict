# 单个 ML 实验的 bio_age 对比报告

## 基本信息
- ML 实验名: `run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge`
- 原始 pred 文件: `/home/szdx/LNX/usage_predict_feature_engineering/outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv`
- bio_age 拟合结果目录: `/home/szdx/LNX/statusage/results/bio_age_feature_benchmark/run_20260422_real_bio_age_benchmark_quick`
- 报告模型: `ridge`

## 先看结论
- 在三条主参考轴里，按 `subject_pred_vs_bio_mae` 最接近的是 `bio_age_ei_texture`，数值为 `2.6807`。
- 按 `subject_closer_to_bio_rate` 看，最能覆盖多数 subject 的是 `bio_age_ei_texture`，比例为 `0.8522`。
- 当前 ML 自身的 `subject_pred_vs_true_mae` 为 `9.2689`。
- 作为 upper bound，`bio_age_full_image_upper_bound` 的 `subject_pred_vs_bio_mae` 可到 `1.8564`，但它不作为主科学定义。

## 三条主参考轴总览
| 轴                              | 解释                                       |   特征数 |   sample_pred_vs_true_mae |   subject_pred_vs_true_mae |   sample_pred_vs_bio_mae |   subject_pred_vs_bio_mae |   sample_gain |   subject_gain |   sample更接近bio比例 |   subject更接近bio比例 |   subject_2年内比例 |   subject_5年内比例 |   subject_8年内比例 |   bio_age_vs_true_mae |   bio_age_vs_true_corr |   bio_age_std |
|:--------------------------------|:-------------------------------------------|---------:|--------------------------:|---------------------------:|-------------------------:|--------------------------:|--------------:|---------------:|----------------------:|-----------------------:|--------------------:|--------------------:|--------------------:|----------------------:|-----------------------:|--------------:|
| bio_age_ei（EI / 一阶统计）     | 整体亮度 / 回声强度 / 粗粒度肌肉质量信号。 |       12 |                   9.44309 |                    9.26891 |                  5.93841 |                   5.56908 |       3.50469 |        3.69983 |              0.656179 |               0.67354  |            0.249714 |            0.534937 |            0.760596 |              10.9201  |               0.552184 |        9.0302 |
| bio_age_texture（纹理）         | 组织异质性 / 纹理模式信号。                |      144 |                   9.44309 |                    9.26891 |                  3.24064 |                   2.89364 |       6.20245 |        6.37527 |              0.821077 |               0.847652 |            0.476518 |            0.873998 |            0.965636 |               9.67909 |               0.543026 |       13.8336 |
| bio_age_ei_texture（EI + 纹理） | 更综合的纯图像 bio_age 信号。              |      168 |                   9.44309 |                    9.26891 |                  2.96862 |                   2.68067 |       6.47447 |        6.58824 |              0.827521 |               0.852234 |            0.486827 |            0.880871 |            0.97709  |               9.64975 |               0.544435 |       13.8469 |

## upper bound / 补充轴
| 轴                                                 | 解释                               |   特征数 |   subject_pred_vs_bio_mae |   subject_gain |   subject更接近bio比例 |   bio_age_vs_true_mae |   bio_age_vs_true_corr |
|:---------------------------------------------------|:-----------------------------------|---------:|--------------------------:|---------------:|-----------------------:|----------------------:|-----------------------:|
| bio_age_full_image_upper_bound（full upper bound） | 实用上限参考，不作为主科学定义。   |      185 |                   1.85642 |       7.41249  |               0.895762 |               9.29312 |              0.640607  |
| bio_age_texture_metadata（纹理 + metadata）        | 实用上限参考，不作为主科学定义。   |      148 |                   2.27056 |       6.99835  |               0.871707 |               9.57115 |              0.564119  |
| bio_age_morphology（形态）                         | ROI 形态与空间分布信号，作为补充。 |       13 |                   9.07269 |       0.196224 |               0.497136 |              13.128   |              0.0793478 |

## 逐 subject 诊断
- `tables/subject_error_matrix_main_axes.csv`：每个 subject 在三条主轴下的误差并列表。
- `tables/worst_subjects_main_axes.csv`：主轴平均误差最大的 subjects。
- `tables/age_band_overview_best_main_axis.csv`：最佳主参考轴下按年龄段分层的结果。

## 图表
- `figures/01_main_axes_subject_gap_mae.png`：三条主轴的 gap MAE。
- `figures/02_main_axes_subject_closer_rate.png`：多数 subject 是否更接近 bio_age。
- `figures/03_main_axes_subject_within_rates.png`：2/5/8 年内覆盖率。
- `figures/04_worst_subjects_main_axes.png`：最难对齐的 subjects。
- `figures/05_pred_age_vs_true_age.png` 与 `06-08`：散点关系图。
- `figures/09_true_bio_pred_curve_best_main_axis.png`：最佳主参考轴下 true_age、pred_age、bio_age 三条曲线图。
- `figures/10_age_band_curve_best_main_axis.png`：按年龄段聚合后的 true_age、pred_age、bio_age 曲线图。
- `figures/11_age_band_summary_best_main_axis.png`：按年龄段聚合后的 MAE 与 closer-to-bio 摘要图。
- `figures/10_age_band_summary_best_main_axis.png`：最佳主参考轴下按年龄段分层的结果图。

## 建议阅读顺序
1. 先看 `summary.md` 的主参考轴总览。
2. 再看 `tables/worst_subjects_main_axes.csv`，判断问题是少数异常 subject 还是整体现象。
3. 最后再参考 upper bound，避免把复杂组合误当成主科学结论。
