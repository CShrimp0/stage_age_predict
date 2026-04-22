#!/usr/bin/env python3
"""Compare many ML age-prediction runs against precomputed bio-age outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRED_COLUMN_CANDIDATES = ["pred_age", "prediction", "predicted_age", "y_pred"]
TRUE_AGE_CANDIDATES = ["true_age", "age", "y_true"]
SUBJECT_CANDIDATES = ["subject_id", "ID", "id"]
SAMPLE_CANDIDATES = ["sample_id", "image_id", "instance_id"]
MAIN_REFERENCE_AXES = ["bio_age_ei", "bio_age_texture", "bio_age_ei_texture"]


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_column(columns: Iterable[str], candidates: list[str], label: str) -> str:
    column_list = list(columns)
    normalized = {normalize_name(c): c for c in column_list}
    for candidate in candidates:
        if candidate in column_list:
            return candidate
        normalized_candidate = normalize_name(candidate)
        if normalized_candidate in normalized:
            return normalized[normalized_candidate]
    raise KeyError(f"Could not infer {label} column. Available: {column_list}")


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    if len(x_arr) < 2 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def mae(left: pd.Series, right: pd.Series) -> float:
    return float(np.mean(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))))


def run_name_from_prediction_path(path: Path) -> str:
    if path.parent.name == "tables":
        return path.parent.parent.name
    return path.parent.name


def discover_prediction_files(pred_root: Path) -> list[Path]:
    candidates: dict[str, Path] = {}
    for path in sorted(pred_root.rglob("predictions_readable.csv")):
        candidates[run_name_from_prediction_path(path)] = path
    for path in sorted(pred_root.rglob("tables/predictions.csv")):
        run_name = run_name_from_prediction_path(path)
        candidates.setdefault(run_name, path)
    return [candidates[key] for key in sorted(candidates)]


def load_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    sample_col = resolve_column(frame.columns, SAMPLE_CANDIDATES, "sample")
    subject_col = resolve_column(frame.columns, SUBJECT_CANDIDATES, "subject")
    true_col = resolve_column(frame.columns, TRUE_AGE_CANDIDATES, "true_age")
    pred_col = resolve_column(frame.columns, PRED_COLUMN_CANDIDATES, "pred_age")
    out = frame[[sample_col, subject_col, true_col, pred_col]].copy()
    out.columns = ["sample_id", "subject_id", "true_age", "pred_age"]
    out["true_age"] = pd.to_numeric(out["true_age"], errors="coerce")
    out["pred_age"] = pd.to_numeric(out["pred_age"], errors="coerce")
    out = out.dropna(subset=["sample_id", "subject_id", "true_age", "pred_age"])
    out = out.drop_duplicates(subset=["sample_id"], keep="first")
    return out


def discover_bio_age_files(bio_age_run: Path, include_elasticnet: bool) -> list[Path]:
    patterns = ["results.csv"]
    if include_elasticnet:
        patterns.append("results_elasticnet.csv")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted((bio_age_run / "feature_sets").glob(f"*/{pattern}")))
    return files


def bio_age_labels(path: Path) -> tuple[str, str]:
    feature_set = path.parent.name
    bio_age_model = "ridge" if path.name == "results.csv" else path.stem.replace("results_", "")
    return feature_set, bio_age_model


def load_bio_age_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "bio_age" not in frame.columns and "state_age" in frame.columns:
        frame = frame.rename(columns={"state_age": "bio_age"})
    required = ["sample_id", "subject_id", "true_age", "bio_age"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    out = frame[required].copy()
    out["true_age"] = pd.to_numeric(out["true_age"], errors="coerce")
    out["bio_age"] = pd.to_numeric(out["bio_age"], errors="coerce")
    out = out.dropna(subset=required)
    out = out.drop_duplicates(subset=["sample_id"], keep="first")
    return out


def summarize_alignment(merged: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    sample_abs_pred_true = (merged["pred_age"] - merged["true_age"]).abs()
    sample_abs_pred_bio = (merged["pred_age"] - merged["bio_age"]).abs()
    sample_abs_bio_true = (merged["bio_age"] - merged["true_age"]).abs()

    subject = (
        merged.groupby("subject_id", as_index=False)[["true_age", "pred_age", "bio_age"]]
        .mean()
        .dropna()
    )
    subject_abs_pred_true = (subject["pred_age"] - subject["true_age"]).abs()
    subject_abs_pred_bio = (subject["pred_age"] - subject["bio_age"]).abs()
    subject_abs_bio_true = (subject["bio_age"] - subject["true_age"]).abs()

    subject_diag = subject[["subject_id", "true_age", "pred_age", "bio_age"]].copy()
    subject_diag["subject_abs_pred_true"] = subject_abs_pred_true
    subject_diag["subject_abs_pred_bio"] = subject_abs_pred_bio
    subject_diag["subject_abs_bio_true"] = subject_abs_bio_true
    subject_diag["subject_gain"] = subject_diag["subject_abs_pred_true"] - subject_diag["subject_abs_pred_bio"]
    subject_diag["subject_closer_to_bio"] = subject_diag["subject_abs_pred_bio"] < subject_diag["subject_abs_pred_true"]

    metrics = {
        "n_samples": int(len(merged)),
        "n_subjects": int(subject["subject_id"].nunique()),
        "sample_ml_true_mae": float(sample_abs_pred_true.mean()),
        "subject_ml_true_mae": float(subject_abs_pred_true.mean()),
        "ml_sample_mae": float(sample_abs_pred_true.mean()),
        "ml_subject_mae": float(subject_abs_pred_true.mean()),
        "bio_age_sample_mae": float(sample_abs_bio_true.mean()),
        "bio_age_subject_mae": float(subject_abs_bio_true.mean()),
        "sample_gap_mae": float(sample_abs_pred_bio.mean()),
        "subject_gap_mae": float(subject_abs_pred_bio.mean()),
        "sample_gain": float(sample_abs_pred_true.mean() - sample_abs_pred_bio.mean()),
        "subject_gain": float(subject_abs_pred_true.mean() - subject_abs_pred_bio.mean()),
        "sample_closer_to_bio_rate": float((sample_abs_pred_bio < sample_abs_pred_true).mean()),
        "subject_closer_to_bio_rate": float((subject_abs_pred_bio < subject_abs_pred_true).mean()),
        "sample_within_2_rate": float((sample_abs_pred_bio <= 2.0).mean()),
        "sample_within_5_rate": float((sample_abs_pred_bio <= 5.0).mean()),
        "sample_within_8_rate": float((sample_abs_pred_bio <= 8.0).mean()),
        "subject_within_2_rate": float((subject_abs_pred_bio <= 2.0).mean()),
        "subject_within_5_rate": float((subject_abs_pred_bio <= 5.0).mean()),
        "subject_within_8_rate": float((subject_abs_pred_bio <= 8.0).mean()),
        "sample_tie_rate": float(np.isclose(sample_abs_pred_bio, sample_abs_pred_true).mean()),
        "subject_tie_rate": float(np.isclose(subject_abs_pred_bio, subject_abs_pred_true).mean()),
        "sample_gap_median": float(sample_abs_pred_bio.median()),
        "subject_gap_median": float(subject_abs_pred_bio.median()),
        "sample_gap_p90": float(sample_abs_pred_bio.quantile(0.90)),
        "subject_gap_p90": float(subject_abs_pred_bio.quantile(0.90)),
        "pred_bio_corr_sample": safe_corr(merged["pred_age"], merged["bio_age"]),
        "bio_age_vs_true_corr_sample": safe_corr(merged["true_age"], merged["bio_age"]),
    }
    return metrics, subject_diag


def save_bar(frame: pd.DataFrame, value_col: str, path: Path, title: str, ylabel: str) -> None:
    plot_frame = frame.copy()
    plot_frame["label"] = plot_frame["ml_run"].astype(str).str.replace("run_", "", regex=False) + "\n" + plot_frame["feature_set"].astype(str)
    plt.figure(figsize=(11, max(4.5, 0.45 * len(plot_frame))))
    plt.barh(plot_frame["label"], plot_frame[value_col], color="#3b82f6")
    plt.gca().invert_yaxis()
    plt.xlabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_within_rate_plot(frame: pd.DataFrame, path: Path) -> None:
    cols = ["subject_within_2_rate", "subject_within_5_rate", "subject_within_8_rate"]
    plot_frame = frame.head(12).copy()
    labels = plot_frame["feature_set"].astype(str) + "\n" + plot_frame["ml_run"].astype(str).str.replace("run_", "", regex=False)
    x = np.arange(len(plot_frame))
    width = 0.25
    plt.figure(figsize=(12, 5.5))
    for i, col in enumerate(cols):
        plt.bar(x + (i - 1) * width, plot_frame[col], width=width, label=col)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("coverage rate")
    plt.title("Subject within 2/5/8 years of bio_age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_worst_subjects_plot(subject_diagnostics: pd.DataFrame, output_dir: Path) -> None:
    main = subject_diagnostics[subject_diagnostics["feature_set"].isin(MAIN_REFERENCE_AXES)]
    if main.empty:
        return
    pivot = main.pivot_table(
        index=["ml_run", "subject_id"],
        columns="feature_set",
        values="subject_abs_pred_bio",
        aggfunc="mean",
    )
    available = [axis for axis in MAIN_REFERENCE_AXES if axis in pivot.columns]
    if not available:
        return
    pivot["mean_abs_pred_bio"] = pivot[available].mean(axis=1)
    worst = pivot.sort_values("mean_abs_pred_bio", ascending=False).head(15)[available]
    plt.figure(figsize=(10, max(5, 0.4 * len(worst))))
    y = np.arange(len(worst))
    width = 0.8 / len(available)
    for i, axis in enumerate(available):
        plt.barh(y + i * width, worst[axis], height=width, label=axis)
    labels = [f"{idx[0]}\nsubject={idx[1]}" for idx in worst.index]
    plt.yticks(y + width * (len(available) - 1) / 2, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("|pred_age - bio_age|")
    plt.title("Worst subjects across main bio_age axes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "worst_subjects_by_bio_axis.png", dpi=180)
    plt.close()


def write_summary(output_dir: Path, leaderboard: pd.DataFrame, args: argparse.Namespace) -> None:
    ridge = leaderboard[leaderboard["bio_age_model"] == "ridge"].copy()
    main = ridge[ridge["feature_set"].isin(MAIN_REFERENCE_AXES)].copy()
    if main.empty:
        main = ridge.copy()
    top_gap = main.sort_values(["subject_gap_mae", "sample_gap_mae"]).head(10)
    top_rate = main.sort_values(["subject_closer_to_bio_rate", "subject_gain"], ascending=[False, False]).head(10)

    lines = [
        "# ML runs vs bio_age comparison",
        "",
        "Purpose: test whether ML-predicted `pred_age` is closer to interpretable feature-derived `bio_age` reference axes than to `true_age`.",
        "",
        f"- bio_age_run: {args.bio_age_run}",
        f"- pred_root: {args.pred_root}",
        f"- prediction runs compared: {leaderboard['ml_run'].nunique()}",
        f"- bio_age feature sets compared: {leaderboard['feature_set'].nunique()}",
        "",
        "## Main-axis top 10 by subject_gap_mae",
        "",
        "| rank | ml_run | bio_age_axis | n_features | subject_ml_true_mae | subject_gap_mae | subject_gain | subject_closer_to_bio_rate | subject_within_2_rate | subject_within_5_rate | subject_within_8_rate | bio_age_subject_mae |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, (_, row) in enumerate(top_gap.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['ml_run']} | {row['feature_set']} | {int(row.get('n_features', -1))} | "
            f"{row['subject_ml_true_mae']:.4f} | {row['subject_gap_mae']:.4f} | {row['subject_gain']:.4f} | "
            f"{row['subject_closer_to_bio_rate']:.4f} | {row['subject_within_2_rate']:.4f} | "
            f"{row['subject_within_5_rate']:.4f} | {row['subject_within_8_rate']:.4f} | {row['bio_age_subject_mae']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Main-axis top 10 by subject_closer_to_bio_rate",
            "",
            "| rank | ml_run | bio_age_axis | n_features | subject_ml_true_mae | subject_gap_mae | subject_gain | subject_closer_to_bio_rate | bio_age_subject_mae |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, (_, row) in enumerate(top_rate.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['ml_run']} | {row['feature_set']} | {int(row.get('n_features', -1))} | "
            f"{row['subject_ml_true_mae']:.4f} | {row['subject_gap_mae']:.4f} | {row['subject_gain']:.4f} | "
            f"{row['subject_closer_to_bio_rate']:.4f} | {row['bio_age_subject_mae']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- `subject_gain = subject_ml_true_mae - subject_gap_mae`。",
            "- subject_gain > 0 means ML pred_age is closer to bio_age than to true_age at subject level.",
            "- subject_closer_to_bio_rate reports the fraction of subjects where this is true, not just the average.",
            "- `bio_age_texture_metadata` and full feature sets are practical upper bounds, not the sole scientific definition.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare many ML runs against one bio-age benchmark run.")
    parser.add_argument(
        "--bio-age-run",
        "--state-age-run",
        dest="bio_age_run",
        required=True,
        help="Path to a bio_age_feature_benchmark run containing feature_sets/*/results.csv.",
    )
    parser.add_argument("--pred-root", default="outputs", help="Root directory to scan for prediction files.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults under results/ml_vs_bio_age.")
    parser.add_argument(
        "--include-bio-age-elasticnet",
        "--include-state-age-elasticnet",
        dest="include_bio_age_elasticnet",
        action="store_true",
        help="Also compare results_elasticnet.csv bio-age outputs.",
    )
    parser.add_argument("--min-overlap", type=int, default=50, help="Minimum overlapping samples required for one comparison row.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bio_age_run = (PROJECT_ROOT / args.bio_age_run).resolve()
    pred_root = (PROJECT_ROOT / args.pred_root).resolve()
    output_dir = (
        (PROJECT_ROOT / args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "results"
        / "ml_vs_bio_age"
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ml_vs_bio_age"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    pred_files = discover_prediction_files(pred_root)
    bio_age_files = discover_bio_age_files(bio_age_run, include_elasticnet=args.include_bio_age_elasticnet)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found under {pred_root}")
    if not bio_age_files:
        raise FileNotFoundError(f"No bio-age results found under {bio_age_run / 'feature_sets'}")

    bio_age_frames = []
    for path in bio_age_files:
        feature_set, bio_age_model = bio_age_labels(path)
        frame = load_bio_age_file(path)
        n_features = None
        leaderboard_path = bio_age_run / "bio_age_reference_leaderboard.csv"
        if not leaderboard_path.exists():
            leaderboard_path = bio_age_run / "leaderboard.csv"
        if leaderboard_path.exists():
            board = pd.read_csv(leaderboard_path)
            match = board[(board["feature_set"] == feature_set) & (board["model"] == bio_age_model)]
            if not match.empty and "n_features" in match.columns:
                n_features = int(match.iloc[0]["n_features"])
        bio_age_frames.append((path, feature_set, bio_age_model, n_features, frame))

    rows: list[dict[str, object]] = []
    subject_rows: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for pred_path in pred_files:
        ml_run = run_name_from_prediction_path(pred_path)
        try:
            pred = load_prediction_file(pred_path)
        except Exception as exc:
            skipped.append({"ml_run": ml_run, "path": str(pred_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        for bio_path, feature_set, bio_age_model, n_features, bio_age in bio_age_frames:
            merged = pred.merge(
                bio_age,
                on=["sample_id", "subject_id"],
                how="inner",
                suffixes=("_ml", "_bio"),
            )
            if "true_age_bio" in merged.columns:
                merged["true_age"] = merged["true_age_bio"]
            elif "true_age_ml" in merged.columns:
                merged["true_age"] = merged["true_age_ml"]
            if len(merged) < args.min_overlap:
                skipped.append(
                    {
                        "ml_run": ml_run,
                        "feature_set": feature_set,
                        "reason": f"overlap {len(merged)} < {args.min_overlap}",
                    }
                )
                continue
            metrics, subject_diag = summarize_alignment(merged)
            subject_diag.insert(0, "bio_age_model", bio_age_model)
            subject_diag.insert(0, "feature_set", feature_set)
            subject_diag.insert(0, "ml_run", ml_run)
            subject_rows.append(subject_diag)
            rows.append(
                {
                    "ml_run": ml_run,
                    "pred_file": str(pred_path),
                    "feature_set": feature_set,
                    "bio_age_model": bio_age_model,
                    "bio_age_file": str(bio_path),
                    "n_features": n_features if n_features is not None else np.nan,
                    **metrics,
                }
            )

    if not rows:
        raise RuntimeError("No comparison rows were produced.")

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values(
        ["bio_age_model", "subject_gap_mae", "sample_gap_mae", "subject_closer_to_bio_rate"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(output_dir / "ml_vs_bio_age_leaderboard.csv", index=False)
    subject_diagnostics = pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame()
    if not subject_diagnostics.empty:
        subject_diagnostics.to_csv(output_dir / "ml_vs_bio_age_subject_diagnostics.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped.csv", index=False)

    metadata = {
        "bio_age_run": str(bio_age_run),
        "pred_root": str(pred_root),
        "n_prediction_files": len(pred_files),
        "n_bio_age_files": len(bio_age_files),
        "n_rows": int(len(leaderboard)),
    }
    (output_dir / "inputs_used.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(output_dir, leaderboard, args)
    ridge_main = leaderboard[
        (leaderboard["bio_age_model"] == "ridge") & (leaderboard["feature_set"].isin(MAIN_REFERENCE_AXES))
    ].sort_values(["subject_gap_mae", "sample_gap_mae"])
    if not ridge_main.empty:
        save_bar(
            ridge_main.head(12),
            "subject_gain",
            output_dir / "figures" / "subject_gain_by_bio_axis.png",
            "Subject gain by main bio_age axis",
            "subject_gain",
        )
        save_bar(
            ridge_main.head(12),
            "subject_closer_to_bio_rate",
            output_dir / "figures" / "subject_closer_to_bio_rate_by_axis.png",
            "Subject closer-to-bio rate by main bio_age axis",
            "subject_closer_to_bio_rate",
        )
        save_within_rate_plot(ridge_main, output_dir / "figures" / "within_2_5_8_coverage_by_axis.png")
    if not subject_diagnostics.empty:
        save_worst_subjects_plot(subject_diagnostics, output_dir)

    print(f"Wrote {len(leaderboard)} comparison rows to {output_dir}")
    show_cols = [
        "ml_run",
        "feature_set",
        "bio_age_model",
        "n_features",
        "subject_ml_true_mae",
        "subject_gap_mae",
        "subject_gain",
        "subject_closer_to_bio_rate",
        "bio_age_subject_mae",
    ]
    print(leaderboard[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
