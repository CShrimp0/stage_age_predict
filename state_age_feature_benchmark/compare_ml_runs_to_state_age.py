#!/usr/bin/env python3
"""Compare many ML age-prediction runs against precomputed state-age outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRED_COLUMN_CANDIDATES = ["pred_age", "prediction", "predicted_age", "y_pred"]
TRUE_AGE_CANDIDATES = ["true_age", "age", "y_true"]
SUBJECT_CANDIDATES = ["subject_id", "ID", "id"]
SAMPLE_CANDIDATES = ["sample_id", "image_id", "instance_id"]


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


def discover_state_age_files(state_age_run: Path, include_elasticnet: bool) -> list[Path]:
    patterns = ["results.csv"]
    if include_elasticnet:
        patterns.append("results_elasticnet.csv")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted((state_age_run / "feature_sets").glob(f"*/{pattern}")))
    return files


def state_age_labels(path: Path) -> tuple[str, str]:
    feature_set = path.parent.name
    state_age_model = "ridge" if path.name == "results.csv" else path.stem.replace("results_", "")
    return feature_set, state_age_model


def load_state_age_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["sample_id", "subject_id", "true_age", "state_age"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    out = frame[required].copy()
    out["true_age"] = pd.to_numeric(out["true_age"], errors="coerce")
    out["state_age"] = pd.to_numeric(out["state_age"], errors="coerce")
    out = out.dropna(subset=required)
    out = out.drop_duplicates(subset=["sample_id"], keep="first")
    return out


def summarize_alignment(merged: pd.DataFrame) -> dict[str, float | int]:
    sample_abs_pred_true = (merged["pred_age"] - merged["true_age"]).abs()
    sample_abs_pred_state = (merged["pred_age"] - merged["state_age"]).abs()
    sample_abs_state_true = (merged["state_age"] - merged["true_age"]).abs()

    subject = (
        merged.groupby("subject_id", as_index=False)[["true_age", "pred_age", "state_age"]]
        .mean()
        .dropna()
    )
    subject_abs_pred_true = (subject["pred_age"] - subject["true_age"]).abs()
    subject_abs_pred_state = (subject["pred_age"] - subject["state_age"]).abs()
    subject_abs_state_true = (subject["state_age"] - subject["true_age"]).abs()

    return {
        "n_samples": int(len(merged)),
        "n_subjects": int(subject["subject_id"].nunique()),
        "ml_sample_mae": float(sample_abs_pred_true.mean()),
        "ml_subject_mae": float(subject_abs_pred_true.mean()),
        "state_age_sample_mae": float(sample_abs_state_true.mean()),
        "state_age_subject_mae": float(subject_abs_state_true.mean()),
        "sample_gap_mae": float(sample_abs_pred_state.mean()),
        "subject_gap_mae": float(subject_abs_pred_state.mean()),
        "sample_gain": float(sample_abs_pred_true.mean() - sample_abs_pred_state.mean()),
        "subject_gain": float(subject_abs_pred_true.mean() - subject_abs_pred_state.mean()),
        "sample_closer_to_state_rate": float((sample_abs_pred_state < sample_abs_pred_true).mean()),
        "subject_closer_to_state_rate": float((subject_abs_pred_state < subject_abs_pred_true).mean()),
        "sample_tie_rate": float(np.isclose(sample_abs_pred_state, sample_abs_pred_true).mean()),
        "subject_tie_rate": float(np.isclose(subject_abs_pred_state, subject_abs_pred_true).mean()),
        "sample_gap_median": float(sample_abs_pred_state.median()),
        "subject_gap_median": float(subject_abs_pred_state.median()),
        "sample_gap_p90": float(sample_abs_pred_state.quantile(0.90)),
        "subject_gap_p90": float(subject_abs_pred_state.quantile(0.90)),
        "pred_state_corr_sample": safe_corr(merged["pred_age"], merged["state_age"]),
        "state_age_vs_true_corr_sample": safe_corr(merged["true_age"], merged["state_age"]),
    }


def write_summary(output_dir: Path, leaderboard: pd.DataFrame, args: argparse.Namespace) -> None:
    ridge = leaderboard[leaderboard["state_age_model"] == "ridge"].copy()
    top_gap = ridge.sort_values(["subject_gap_mae", "sample_gap_mae"]).head(10)
    top_rate = ridge.sort_values(["subject_closer_to_state_rate", "subject_gain"], ascending=[False, False]).head(10)

    lines = [
        "# ML Runs vs State Age Comparison",
        "",
        "Purpose: test whether ML-predicted age is closer to interpretable feature-derived state_age than to chronological true_age.",
        "",
        f"- state_age_run: {args.state_age_run}",
        f"- pred_root: {args.pred_root}",
        f"- prediction runs compared: {leaderboard['ml_run'].nunique()}",
        f"- state_age feature sets compared: {leaderboard['feature_set'].nunique()}",
        "",
        "## Top 10 by subject_gap_mae",
        "",
        "| rank | ml_run | feature_set | n_features | ml_subject_mae | subject_gap_mae | subject_gain | subject_closer_to_state_rate | state_age_subject_mae |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, (_, row) in enumerate(top_gap.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['ml_run']} | {row['feature_set']} | {int(row.get('n_features', -1))} | "
            f"{row['ml_subject_mae']:.4f} | {row['subject_gap_mae']:.4f} | {row['subject_gain']:.4f} | "
            f"{row['subject_closer_to_state_rate']:.4f} | {row['state_age_subject_mae']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 by subject_closer_to_state_rate",
            "",
            "| rank | ml_run | feature_set | n_features | ml_subject_mae | subject_gap_mae | subject_gain | subject_closer_to_state_rate | state_age_subject_mae |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, (_, row) in enumerate(top_rate.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['ml_run']} | {row['feature_set']} | {int(row.get('n_features', -1))} | "
            f"{row['ml_subject_mae']:.4f} | {row['subject_gap_mae']:.4f} | {row['subject_gain']:.4f} | "
            f"{row['subject_closer_to_state_rate']:.4f} | {row['state_age_subject_mae']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- subject_gain = ml_subject_mae - subject_gap_mae.",
            "- subject_gain > 0 means ML pred_age is closer to state_age than to true_age at subject level.",
            "- subject_closer_to_state_rate reports the fraction of subjects where this is true, not just the average.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare many ML runs against one state-age benchmark run.")
    parser.add_argument(
        "--state-age-run",
        required=True,
        help="Path to a state_age_feature_benchmark run containing feature_sets/*/results.csv.",
    )
    parser.add_argument("--pred-root", default="outputs", help="Root directory to scan for prediction files.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults under outputs/state_age_model_comparison.")
    parser.add_argument("--include-state-age-elasticnet", action="store_true", help="Also compare results_elasticnet.csv state-age outputs.")
    parser.add_argument("--min-overlap", type=int, default=50, help="Minimum overlapping samples required for one comparison row.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state_age_run = (PROJECT_ROOT / args.state_age_run).resolve()
    pred_root = (PROJECT_ROOT / args.pred_root).resolve()
    output_dir = (
        (PROJECT_ROOT / args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "outputs"
        / "state_age_model_comparison"
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ml_vs_state_age"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = discover_prediction_files(pred_root)
    state_age_files = discover_state_age_files(state_age_run, include_elasticnet=args.include_state_age_elasticnet)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found under {pred_root}")
    if not state_age_files:
        raise FileNotFoundError(f"No state-age results found under {state_age_run / 'feature_sets'}")

    state_age_frames = []
    for path in state_age_files:
        feature_set, state_age_model = state_age_labels(path)
        frame = load_state_age_file(path)
        n_features = None
        leaderboard_path = state_age_run / "leaderboard.csv"
        if leaderboard_path.exists():
            board = pd.read_csv(leaderboard_path)
            match = board[(board["feature_set"] == feature_set) & (board["model"] == state_age_model)]
            if not match.empty and "n_features" in match.columns:
                n_features = int(match.iloc[0]["n_features"])
        state_age_frames.append((path, feature_set, state_age_model, n_features, frame))

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for pred_path in pred_files:
        ml_run = run_name_from_prediction_path(pred_path)
        try:
            pred = load_prediction_file(pred_path)
        except Exception as exc:
            skipped.append({"ml_run": ml_run, "path": str(pred_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        for state_path, feature_set, state_age_model, n_features, state_age in state_age_frames:
            merged = pred.merge(
                state_age,
                on=["sample_id", "subject_id"],
                how="inner",
                suffixes=("_ml", "_state"),
            )
            if "true_age_state" in merged.columns:
                merged["true_age"] = merged["true_age_state"]
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
            metrics = summarize_alignment(merged)
            rows.append(
                {
                    "ml_run": ml_run,
                    "pred_file": str(pred_path),
                    "feature_set": feature_set,
                    "state_age_model": state_age_model,
                    "state_age_file": str(state_path),
                    "n_features": n_features if n_features is not None else np.nan,
                    **metrics,
                }
            )

    if not rows:
        raise RuntimeError("No comparison rows were produced.")

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values(
        ["state_age_model", "subject_gap_mae", "sample_gap_mae", "subject_closer_to_state_rate"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(output_dir / "ml_vs_state_age_leaderboard.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped.csv", index=False)

    metadata = {
        "state_age_run": str(state_age_run),
        "pred_root": str(pred_root),
        "n_prediction_files": len(pred_files),
        "n_state_age_files": len(state_age_files),
        "n_rows": int(len(leaderboard)),
    }
    (output_dir / "inputs_used.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(output_dir, leaderboard, args)

    print(f"Wrote {len(leaderboard)} comparison rows to {output_dir}")
    show_cols = [
        "ml_run",
        "feature_set",
        "state_age_model",
        "n_features",
        "ml_subject_mae",
        "subject_gap_mae",
        "subject_gain",
        "subject_closer_to_state_rate",
        "state_age_subject_mae",
    ]
    print(leaderboard[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
