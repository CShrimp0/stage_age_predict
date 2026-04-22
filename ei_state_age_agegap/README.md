# EI Bio Age Baseline

This directory provides the EI-only `bio_age` baseline. It is intentionally
simple and interpretable: EI is mapped onto an 18-100 age-like scale and then
compared with ML-predicted `pred_age`.

This is a baseline reference axis, not the final scientific conclusion.

```text
bio_age = 18 + (EI - EI_min) / (EI_max - EI_min) * (100 - 18)
agegap = pred_age - bio_age
agegap_mae = mean(abs(agegap))
```

By default, higher EI means older `bio_age`. Use `--reverse` if your EI
definition has the opposite direction.

## Input Columns

The tool auto-detects common names:

- `true_age`: `true_age`, `age`, `Age`, `年龄`
- `pred_age`: `pred_age`, `prediction`, `predicted_age`, `预测年龄`
- `EI`: `EI`, `echo_intensity`, or a single column containing `亮度` / `灰度`

Use explicit `--*-column` options if multiple candidates exist.

## Direct EI Table

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input path/to/predictions_with_ei.csv \
  --ei-column EI \
  --output-dir results/ei_bio_age_agegap
```

## Separate EI Source

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input path/to/predictions.csv \
  --ei-source path/to/ei_table.xlsx \
  --ei-sheet 左连接 \
  --ei-column "股直肌亮度（静息）" \
  --merge-key subject_id:ID \
  --output-dir results/ei_bio_age_agegap/run_xxx
```

`--merge-key left:right` maps prediction-table columns to EI-source columns.

## Outputs

- `agegap_results.csv`: row-level results with `bio_age`, `agegap`, `agegap_abs`
- `agegap_subject_results.csv`: subject-level aggregation when a subject column exists
- `metrics.json`: row-level and subject-level gap metrics
- `summary.md`: short readable summary

