#!/usr/bin/env python3
"""Build a clear single-ML-run bio_age report from one benchmark run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_REFERENCE_AXES = ["bio_age_ei", "bio_age_texture", "bio_age_ei_texture"]
SUPPLEMENTAL_AXES = ["bio_age_morphology", "bio_age_texture_metadata", "bio_age_full_image_upper_bound"]
AXIS_LABELS = {
    "bio_age_ei": "bio_age_ei（EI / 一阶统计）",
    "bio_age_texture": "bio_age_texture（纹理）",
    "bio_age_ei_texture": "bio_age_ei_texture（EI + 纹理）",
    "bio_age_morphology": "bio_age_morphology（形态）",
    "bio_age_texture_metadata": "bio_age_texture_metadata（纹理 + metadata）",
    "bio_age_full_image_upper_bound": "bio_age_full_image_upper_bound（full upper bound）",
}
AXIS_DESCRIPTIONS = {
    "bio_age_ei": "整体亮度 / 回声强度 / 粗粒度肌肉质量信号。",
    "bio_age_texture": "组织异质性 / 纹理模式信号。",
    "bio_age_ei_texture": "更综合的纯图像 bio_age 信号。",
    "bio_age_morphology": "ROI 形态与空间分布信号，作为补充。",
    "bio_age_texture_metadata": "实用上限参考，不作为主科学定义。",
    "bio_age_full_image_upper_bound": "实用上限参考，不作为主科学定义。",
}


def infer_ml_run_name(pred_path: str) -> str:
    path = Path(pred_path)
    if path.name == "predictions_readable.csv":
        return path.parent.name
    if path.name == "predictions.csv" and path.parent.name == "tables":
        return path.parent.parent.name
    return path.stem


def _axis_label(axis: str) -> str:
    return AXIS_LABELS.get(axis, axis)


def save_bar(frame: pd.DataFrame, value_col: str, path: Path, title: str, xlabel: str) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plt.figure(figsize=(10, max(4.5, 0.55 * len(plot_frame))))
    plt.barh(plot_frame["轴"], plot_frame[value_col], color="#2563eb")
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_grouped_within_rate_plot(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    cols = ["subject_2年内比例", "subject_5年内比例", "subject_8年内比例"]
    x = np.arange(len(frame))
    width = 0.25
    plt.figure(figsize=(10, 5.5))
    for idx, col in enumerate(cols):
        plt.bar(x + (idx - 1) * width, frame[col], width=width, label=col)
    plt.xticks(x, frame["轴"], rotation=15, ha="right")
    plt.ylabel("比例")
    plt.title("主参考轴的 subject 覆盖率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_scatter(x: np.ndarray, y: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, s=10, alpha=0.55)
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_age_curve_plot(frame: pd.DataFrame, path: Path, title: str) -> None:
    if frame.empty:
        return
    plot_frame = frame.sort_values(["true_age", "subject_id"]).reset_index(drop=True).copy()
    x = np.arange(len(plot_frame))
    plt.figure(figsize=(12, 5.8))
    plt.plot(x, plot_frame["true_age"], label="true_age", linewidth=2.0, color="#111827")
    plt.plot(x, plot_frame["pred_age"], label="pred_age", linewidth=1.8, color="#2563eb")
    plt.plot(x, plot_frame["bio_age"], label="bio_age", linewidth=1.8, color="#dc2626")
    plt.xlabel("subjects（按 true_age 排序）")
    plt.ylabel("age")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_worst_subjects_plot(worst_df: pd.DataFrame, path: Path) -> None:
    axes = [col for col in ["bio_age_ei误差", "bio_age_texture误差", "bio_age_ei_texture误差"] if col in worst_df.columns]
    if worst_df.empty or not axes:
        return
    plt.figure(figsize=(10, max(5, 0.45 * len(worst_df))))
    y = np.arange(len(worst_df))
    width = 0.8 / len(axes)
    for idx, axis in enumerate(axes):
        plt.barh(y + idx * width, worst_df[axis], height=width, label=axis)
    labels = [str(v) for v in worst_df["subject_id"]]
    plt.yticks(y + width * (len(axes) - 1) / 2, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("|pred_age - bio_age|")
    plt.title("主参考轴下误差最大的 subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_single_ml_report(bio_age_run: Path, output_root: Path | None = None, model: str = "ridge") -> Path:
    bio_age_run = bio_age_run.resolve()
    leaderboard_path = bio_age_run / "bio_age_reference_leaderboard.csv"
    if not leaderboard_path.exists():
        leaderboard_path = bio_age_run / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Cannot find leaderboard in {bio_age_run}")

    inputs_path = bio_age_run / "inputs_used.json"
    subject_diag_path = bio_age_run / "bio_age_reference_subject_diagnostics.csv"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing inputs_used.json in {bio_age_run}")
    if not subject_diag_path.exists():
        raise FileNotFoundError(f"Missing bio_age_reference_subject_diagnostics.csv in {bio_age_run}")

    inputs_used = json.loads(inputs_path.read_text(encoding="utf-8"))
    pred_path = inputs_used.get("pred", "")
    ml_run_name = infer_ml_run_name(pred_path)

    if output_root is None:
        output_root = PROJECT_ROOT / "results" / "reports" / "single_ml"
    report_dir = (output_root / ml_run_name).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tables").mkdir(exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)

    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard = leaderboard[leaderboard["model"] == model].copy()
    if leaderboard.empty:
        raise ValueError(f"No rows found for model={model} in {leaderboard_path}")

    subject_diag = pd.read_csv(subject_diag_path)
    subject_diag = subject_diag[subject_diag["model"] == model].copy()

    main = (
        leaderboard[leaderboard["feature_set"].isin(MAIN_REFERENCE_AXES)]
        .set_index("feature_set")
        .reindex(MAIN_REFERENCE_AXES)
        .reset_index()
    )
    supplemental = leaderboard[leaderboard["feature_set"].isin(SUPPLEMENTAL_AXES)].copy()
    supplemental = supplemental.sort_values("subject_gap_mae").reset_index(drop=True)

    main_table = pd.DataFrame(
        {
            "轴": main["feature_set"].map(_axis_label),
            "解释": main["feature_set"].map(lambda x: AXIS_DESCRIPTIONS.get(x, "")),
            "特征数": main["n_features"],
            "sample_pred_vs_true_mae": main["sample_ml_true_mae"],
            "subject_pred_vs_true_mae": main["subject_ml_true_mae"],
            "sample_pred_vs_bio_mae": main["sample_gap_mae"],
            "subject_pred_vs_bio_mae": main["subject_gap_mae"],
            "sample_gain": main["sample_gain"],
            "subject_gain": main["subject_gain"],
            "sample更接近bio比例": main["sample_closer_to_bio_rate"],
            "subject更接近bio比例": main["subject_closer_to_bio_rate"],
            "subject_2年内比例": main["subject_within_2_rate"],
            "subject_5年内比例": main["subject_within_5_rate"],
            "subject_8年内比例": main["subject_within_8_rate"],
            "bio_age_vs_true_mae": main["bio_age_vs_true_mae"],
            "bio_age_vs_true_corr": main["bio_age_vs_true_corr"],
            "bio_age_std": main["bio_age_std"],
        }
    )
    main_table.to_csv(report_dir / "tables" / "main_axes_overview.csv", index=False)

    upper_table = pd.DataFrame(
        {
            "轴": supplemental["feature_set"].map(_axis_label),
            "解释": supplemental["feature_set"].map(lambda x: AXIS_DESCRIPTIONS.get(x, "")),
            "特征数": supplemental["n_features"],
            "subject_pred_vs_bio_mae": supplemental["subject_gap_mae"],
            "subject_gain": supplemental["subject_gain"],
            "subject更接近bio比例": supplemental["subject_closer_to_bio_rate"],
            "bio_age_vs_true_mae": supplemental["bio_age_vs_true_mae"],
            "bio_age_vs_true_corr": supplemental["bio_age_vs_true_corr"],
        }
    )
    upper_table.to_csv(report_dir / "tables" / "upper_bound_axes_overview.csv", index=False)

    main_subject = subject_diag[subject_diag["feature_set"].isin(MAIN_REFERENCE_AXES)].copy()
    base_subject = (
        main_subject.sort_values(["feature_set", "subject_id"])
        .drop_duplicates(subset=["subject_id"], keep="first")[["subject_id", "true_age", "pred_age", "subject_abs_pred_true"]]
        .rename(columns={"subject_abs_pred_true": "pred_vs_true误差"})
    )
    matrix = (
        main_subject.pivot_table(
            index="subject_id",
            columns="feature_set",
            values="subject_abs_pred_bio",
            aggfunc="mean",
        )
        .rename(
            columns={
                "bio_age_ei": "bio_age_ei误差",
                "bio_age_texture": "bio_age_texture误差",
                "bio_age_ei_texture": "bio_age_ei_texture误差",
            }
        )
        .reset_index()
    )
    subject_matrix = base_subject.merge(matrix, on="subject_id", how="left")
    subject_matrix["主轴平均误差"] = subject_matrix[
        [c for c in ["bio_age_ei误差", "bio_age_texture误差", "bio_age_ei_texture误差"] if c in subject_matrix.columns]
    ].mean(axis=1)
    subject_matrix = subject_matrix.sort_values("主轴平均误差", ascending=False).reset_index(drop=True)
    subject_matrix.to_csv(report_dir / "tables" / "subject_error_matrix_main_axes.csv", index=False)

    worst_subjects = subject_matrix.head(20).copy()
    worst_subjects.to_csv(report_dir / "tables" / "worst_subjects_main_axes.csv", index=False)

    best_gap = main.loc[main["subject_gap_mae"].idxmin()]
    best_rate = main.loc[main["subject_closer_to_bio_rate"].idxmax()]
    upper_best = None
    if not supplemental.empty:
        upper_best = supplemental.loc[supplemental["subject_gap_mae"].idxmin()]

    figures_dir = report_dir / "figures"
    save_bar(
        main_table,
        "subject_pred_vs_bio_mae",
        figures_dir / "01_main_axes_subject_gap_mae.png",
        "三条主参考轴的 subject_pred_vs_bio_mae",
        "subject_pred_vs_bio_mae",
    )
    save_bar(
        main_table,
        "subject更接近bio比例",
        figures_dir / "02_main_axes_subject_closer_rate.png",
        "三条主参考轴的 subject 更接近 bio_age 比例",
        "subject_closer_to_bio_rate",
    )
    save_grouped_within_rate_plot(main_table, figures_dir / "03_main_axes_subject_within_rates.png")
    save_worst_subjects_plot(worst_subjects, figures_dir / "04_worst_subjects_main_axes.png")

    ref_results = pd.read_csv(bio_age_run / "feature_sets" / "bio_age_ei" / "results.csv")
    save_scatter(
        ref_results["true_age"].to_numpy(),
        ref_results["pred_age"].to_numpy(),
        figures_dir / "05_pred_age_vs_true_age.png",
        "pred_age 与 true_age",
        "true_age",
        "pred_age",
    )
    for idx, axis in enumerate(MAIN_REFERENCE_AXES, start=6):
        axis_results = pd.read_csv(bio_age_run / "feature_sets" / axis / "results.csv")
        save_scatter(
            axis_results["bio_age"].to_numpy(),
            axis_results["pred_age"].to_numpy(),
            figures_dir / f"{idx:02d}_pred_age_vs_{axis}.png",
            f"pred_age 与 {axis}",
            axis,
            "pred_age",
        )

    best_main_axis = str(best_gap["feature_set"])
    best_main_subject = pd.read_csv(bio_age_run / "feature_sets" / best_main_axis / "subject_results.csv")
    save_age_curve_plot(
        best_main_subject,
        figures_dir / "09_true_bio_pred_curve_best_main_axis.png",
        f"true_age / pred_age / bio_age 曲线图（{best_main_axis}）",
    )

    metadata = {
        "report_generated_at": datetime.now().isoformat(timespec="seconds"),
        "bio_age_run": str(bio_age_run),
        "ml_run_name": ml_run_name,
        "pred_path": pred_path,
        "model": model,
        "main_axes": MAIN_REFERENCE_AXES,
        "supplemental_axes": SUPPLEMENTAL_AXES,
    }
    (report_dir / "report_inputs.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# 单个 ML 实验的 bio_age 对比报告")
    lines.append("")
    lines.append("## 基本信息")
    lines.append(f"- ML 实验名: `{ml_run_name}`")
    lines.append(f"- 原始 pred 文件: `{pred_path}`")
    lines.append(f"- bio_age 拟合结果目录: `{bio_age_run}`")
    lines.append(f"- 报告模型: `{model}`")
    lines.append("")
    lines.append("## 先看结论")
    lines.append(
        f"- 在三条主参考轴里，按 `subject_pred_vs_bio_mae` 最接近的是 `{best_gap['feature_set']}`，"
        f"数值为 `{best_gap['subject_gap_mae']:.4f}`。"
    )
    lines.append(
        f"- 按 `subject_closer_to_bio_rate` 看，最能覆盖多数 subject 的是 `{best_rate['feature_set']}`，"
        f"比例为 `{best_rate['subject_closer_to_bio_rate']:.4f}`。"
    )
    lines.append(
        f"- 当前 ML 自身的 `subject_pred_vs_true_mae` 为 `{float(main.iloc[0]['subject_ml_true_mae']):.4f}`。"
    )
    if upper_best is not None:
        lines.append(
            f"- 作为 upper bound，`{upper_best['feature_set']}` 的 `subject_pred_vs_bio_mae` 可到 "
            f"`{upper_best['subject_gap_mae']:.4f}`，但它不作为主科学定义。"
        )
    lines.append("")
    lines.append("## 三条主参考轴总览")
    lines.append(main_table.to_markdown(index=False))
    lines.append("")
    lines.append("## upper bound / 补充轴")
    if upper_table.empty:
        lines.append("- 本次没有可用的补充轴结果。")
    else:
        lines.append(upper_table.to_markdown(index=False))
    lines.append("")
    lines.append("## 逐 subject 诊断")
    lines.append("- `tables/subject_error_matrix_main_axes.csv`：每个 subject 在三条主轴下的误差并列表。")
    lines.append("- `tables/worst_subjects_main_axes.csv`：主轴平均误差最大的 subjects。")
    lines.append("")
    lines.append("## 图表")
    lines.append("- `figures/01_main_axes_subject_gap_mae.png`：三条主轴的 gap MAE。")
    lines.append("- `figures/02_main_axes_subject_closer_rate.png`：多数 subject 是否更接近 bio_age。")
    lines.append("- `figures/03_main_axes_subject_within_rates.png`：2/5/8 年内覆盖率。")
    lines.append("- `figures/04_worst_subjects_main_axes.png`：最难对齐的 subjects。")
    lines.append("- `figures/05_pred_age_vs_true_age.png` 与 `06-08`：散点关系图。")
    lines.append("- `figures/09_true_bio_pred_curve_best_main_axis.png`：最佳主参考轴下 true_age、pred_age、bio_age 三条曲线图。")
    lines.append("")
    lines.append("## 建议阅读顺序")
    lines.append("1. 先看 `summary.md` 的主参考轴总览。")
    lines.append("2. 再看 `tables/worst_subjects_main_axes.csv`，判断问题是少数异常 subject 还是整体现象。")
    lines.append("3. 最后再参考 upper bound，避免把复杂组合误当成主科学结论。")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a clear single-ML bio_age report from one benchmark run.")
    parser.add_argument("--bio-age-run", required=True, help="Path to one bio_age benchmark run directory.")
    parser.add_argument("--output-root", default="results/reports/single_ml", help="Root directory for single-run reports.")
    parser.add_argument("--model", default="ridge", help="Model name to render, default ridge.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_dir = build_single_ml_report(
        bio_age_run=(PROJECT_ROOT / args.bio_age_run).resolve(),
        output_root=(PROJECT_ROOT / args.output_root).resolve(),
        model=args.model,
    )
    print(f"Wrote single-ML report to {report_dir}")


if __name__ == "__main__":
    main()
