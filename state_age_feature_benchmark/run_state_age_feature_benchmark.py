#!/usr/bin/env python3
"""Fit interpretable image-derived bio_age reference axes under group CV.

In this repository, bio_age is an interpretable image-derived
biological/status age proxy. It is used as a reference axis for understanding
ML-predicted age, not as a competition for best chronological-age prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.load_images import load_grayscale_image, load_mask
from preprocessing.split import RegressionStratifiedGroupKFold
from build_single_ml_report import build_single_ml_report

try:
    from skimage.feature import local_binary_pattern

    HAS_LBP = True
except Exception:
    HAS_LBP = False


PRED_COLUMN_CANDIDATES = ["pred_age", "prediction", "predicted_age", "y_pred"]
TRUE_AGE_CANDIDATES = ["true_age", "age", "y_true"]
SUBJECT_CANDIDATES = ["subject_id", "ID", "id"]
SAMPLE_CANDIDATES = ["sample_id", "image_id", "instance_id"]
MAIN_REFERENCE_AXES = ["bio_age_ei", "bio_age_texture", "bio_age_ei_texture"]
SUPPLEMENTAL_AXES = ["bio_age_morphology", "bio_age_texture_metadata", "bio_age_full_image_upper_bound"]

REFERENCE_AXIS_DESCRIPTIONS = {
    "bio_age_ei": "EI / 一阶统计轴：整体回声强度、亮度与粗粒度肌肉质量信号。",
    "bio_age_texture": "纹理轴：组织异质性与局部纹理模式信号。",
    "bio_age_ei_texture": "纯图像综合轴：一阶统计 + 纹理信号。",
    "bio_age_morphology": "补充形态轴：ROI 形状与空间分布描述。",
    "bio_age_texture_metadata": "实践上限轴：纹理 + 简单元数据。",
    "bio_age_full_image_upper_bound": "实践上限轴：更宽的图像、形态、分区和元数据组合。",
}


@dataclass
class FeatureRunResult:
    feature_set: str
    model: str
    n_features: int
    ml_sample_mae: float
    ml_subject_mae: float
    sample_gap_mae: float
    subject_gap_mae: float
    sample_gap_rmse: float
    subject_gap_rmse: float
    sample_gain: float
    subject_gain: float
    sample_closer_to_bio_rate: float
    subject_closer_to_bio_rate: float
    sample_within_2_rate: float
    sample_within_5_rate: float
    sample_within_8_rate: float
    subject_within_2_rate: float
    subject_within_5_rate: float
    subject_within_8_rate: float
    bio_age_sample_mae: float
    bio_age_subject_mae: float
    bio_age_vs_true_corr: float
    bio_age_bias_slope: float
    bio_age_std: float
    bio_age_min: float
    bio_age_max: float
    status: str
    note: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_column(columns: list[str], candidates: list[str], explicit: str | None, label: str) -> str:
    if explicit:
        if explicit in columns:
            return explicit
        raise KeyError(f"{label} column {explicit!r} is missing. Available: {columns}")
    normalized = {normalize_name(c): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        n_cand = normalize_name(cand)
        if n_cand in normalized:
            return normalized[n_cand]
    raise KeyError(f"Could not infer {label} column. Available: {columns}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def infer_mask_path(row: pd.Series) -> Path | None:
    for key in ("roi_path", "mask_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if candidate.exists():
                return candidate
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        return None
    candidate = Path(image_path.replace("/Images/", "/Masks/"))
    return candidate if candidate.exists() else None


def extract_basic_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}iqr": np.nan,
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}p10": np.nan,
            f"{prefix}p90": np.nan,
        }
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}std": float(np.std(values)),
        f"{prefix}iqr": float(p75 - p25),
        f"{prefix}min": float(np.min(values)),
        f"{prefix}max": float(np.max(values)),
        f"{prefix}p10": p10,
        f"{prefix}p90": p90,
    }


def extract_partition_first_order(
    image: np.ndarray,
    mask: np.ndarray,
    axis: str,
    n_bins: int,
    prefix: str,
) -> dict[str, float]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return {}
    if axis == "depth":
        coords = rows.astype(float)
    elif axis == "width":
        coords = cols.astype(float)
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    c_min = float(coords.min())
    c_max = float(coords.max())
    span = max(c_max - c_min, 1e-6)
    norm = (coords - c_min) / span
    bin_idx = np.floor(norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    out: dict[str, float] = {}
    for b in range(n_bins):
        select = bin_idx == b
        values = image[rows[select], cols[select]] if np.any(select) else np.array([], dtype=float)
        out.update(extract_basic_stats(values, f"{prefix}bin{b}__"))
    return out


def extract_partition_texture(
    image: np.ndarray,
    mask: np.ndarray,
    axis: str,
    n_bins: int,
    prefix: str,
) -> dict[str, float]:
    if not HAS_LBP:
        return {}
    rows, cols = np.where(mask)
    if rows.size == 0:
        return {}
    if axis == "depth":
        coords = rows.astype(float)
    elif axis == "width":
        coords = cols.astype(float)
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    c_min = float(coords.min())
    c_max = float(coords.max())
    span = max(c_max - c_min, 1e-6)
    norm = (coords - c_min) / span
    bin_idx = np.floor(norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    quant = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(quant, P=8, R=1, method="uniform")
    out: dict[str, float] = {}
    for b in range(n_bins):
        select = bin_idx == b
        if not np.any(select):
            out[f"{prefix}bin{b}__lbp_entropy"] = np.nan
            out[f"{prefix}bin{b}__lbp_mean"] = np.nan
            continue
        vals = lbp[rows[select], cols[select]]
        hist, _ = np.histogram(vals, bins=10, range=(0, 10), density=True)
        hist = hist[hist > 0]
        entropy = float(-(hist * np.log2(hist)).sum()) if hist.size else np.nan
        out[f"{prefix}bin{b}__lbp_entropy"] = entropy
        out[f"{prefix}bin{b}__lbp_mean"] = float(np.mean(vals))
    return out


def extract_extra_features_one_row(row: pd.Series) -> tuple[dict[str, float], str | None]:
    sample_id = row["sample_id"]
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        return {"sample_id": sample_id}, "missing_image_path"
    image_file = Path(image_path)
    if not image_file.exists():
        return {"sample_id": sample_id}, "image_not_found"
    mask_file = infer_mask_path(row)
    if mask_file is None:
        return {"sample_id": sample_id}, "mask_not_found"

    try:
        image = load_grayscale_image(image_file)
        mask = load_mask(mask_file)
        if image.shape != mask.shape:
            mask = load_mask(mask_file, resize=(image.shape[1], image.shape[0]))
    except Exception as exc:
        return {"sample_id": sample_id}, f"load_error:{type(exc).__name__}"

    if not np.any(mask):
        return {"sample_id": sample_id}, "empty_mask"

    out: dict[str, float] = {"sample_id": sample_id}
    roi_values = image[mask]

    # A5: depth-normalized first-order in ROI
    row_mean = image.mean(axis=1, keepdims=True)
    row_std = image.std(axis=1, keepdims=True)
    row_std[row_std < 1e-6] = 1.0
    depth_norm = (image - row_mean) / row_std
    depth_values = depth_norm[mask]
    out.update(extract_basic_stats(depth_values, "depthnorm__roi__"))

    # C2: mask depth-position descriptors
    rows, cols = np.where(mask)
    h, w = mask.shape
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    box_h = float(rmax - rmin + 1)
    box_w = float(cmax - cmin + 1)
    area = float(mask.sum())
    eroded = ndimage.binary_erosion(mask)
    border = mask & (~eroded)
    peri = float(border.sum())
    circularity = float(4.0 * np.pi * area / max(peri * peri, 1e-6))

    out.update(
        {
            "mask_depth__centroid": float(rows.mean() / max(h, 1)),
            "mask_depth__min": float(rmin / max(h, 1)),
            "mask_depth__max": float(rmax / max(h, 1)),
            "mask_depth__span": float(box_h / max(h, 1)),
            "mask_depth__superficial_frac": float(np.mean(rows < (h * 0.5))),
            "mask_depth__deep_frac": float(np.mean(rows >= (h * 0.5))),
            "mask_width__centroid": float(cols.mean() / max(w, 1)),
            "mask_width__span": float(box_w / max(w, 1)),
            "mask_shape__bbox_width": box_w,
            "mask_shape__bbox_height": box_h,
            "mask_shape__aspect_ratio": float(box_w / max(box_h, 1e-6)),
            "mask_shape__circularity": circularity,
        }
    )

    # D1/D2: partition first-order
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=2,
            prefix="part_depth2__",
        )
    )
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=4,
            prefix="part_depth4__",
        )
    )
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="width",
            n_bins=2,
            prefix="part_width2__",
        )
    )
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="width",
            n_bins=4,
            prefix="part_width4__",
        )
    )

    # D3: partition texture (LBP-only lightweight)
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=2,
            prefix="part_tex_depth2__",
        )
    )
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=4,
            prefix="part_tex_depth4__",
        )
    )
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="width",
            n_bins=2,
            prefix="part_tex_width2__",
        )
    )
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="width",
            n_bins=4,
            prefix="part_tex_width4__",
        )
    )

    return out, None


def build_or_load_extra_features(df: pd.DataFrame, cache_path: Path) -> tuple[pd.DataFrame, list[str]]:
    if cache_path.exists():
        loaded = pd.read_csv(cache_path)
        return loaded, []

    errors: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        out, err = extract_extra_features_one_row(row)
        rows.append(out)
        if err:
            errors[err] = errors.get(err, 0) + 1
        if i % 300 == 0 or i == total:
            print(f"[extra_features] processed {i}/{total}")

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)

    notes = [f"{k}: {v}" for k, v in sorted(errors.items(), key=lambda x: x[0])]
    return frame, notes


def choose_first_order_columns(columns: list[str], scope: str) -> tuple[list[str], list[str]]:
    wanted = ["mean", "median", "std", "iqr", "min", "max", "p10", "p25", "p75", "p90", "skewness", "kurtosis"]
    fallback = {"p10": ["p10", "p5"], "p90": ["p90", "p95"]}
    selected: list[str] = []
    notes: list[str] = []

    for base in wanted:
        candidates = fallback.get(base, [base])
        picked = None
        for cand in candidates:
            col = f"{scope}__intensity__{cand}"
            if col in columns:
                picked = col
                if cand != base:
                    notes.append(f"{scope}:{base}-> {cand}")
                break
        if picked is not None:
            selected.append(picked)
        else:
            notes.append(f"{scope}:{base} missing")
    return selected, notes


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    columns = list(df.columns)
    notes: list[str] = []

    roi_first_order, notes_roi = choose_first_order_columns(columns, "roi")
    whole_first_order, notes_whole = choose_first_order_columns(columns, "whole_image")
    notes.extend(notes_roi)
    notes.extend(notes_whole)

    roi_mean = "roi__intensity__mean"
    texture_cols = [c for c in columns if "__texture__" in c]
    glcm_cols = [c for c in texture_cols if "__glcm__" in c]
    lbp_cols = [c for c in texture_cols if "__lbp__" in c]
    glrlm_glszm_cols = [c for c in texture_cols if ("__glrlm__" in c or "__glszm__" in c)]

    morphology_cols = [c for c in columns if c.startswith("roi__morphology__")]
    morphology_extra_basic = [
        c
        for c in columns
        if c.startswith("mask_shape__")
    ]
    morphology_depth_cols = [
        c
        for c in columns
        if c.startswith("mask_depth__") or c.startswith("mask_width__")
    ]

    depthnorm_cols = [c for c in columns if c.startswith("depthnorm__roi__")]

    partition_depth_cols = [c for c in columns if c.startswith("part_depth2__") or c.startswith("part_depth4__")]
    partition_width_cols = [c for c in columns if c.startswith("part_width2__") or c.startswith("part_width4__")]
    partition_texture_cols = [c for c in columns if c.startswith("part_tex_")]

    metadata_cols = [c for c in ["meta__sex_male", "height_cm", "weight_kg", "bmi"] if c in columns]

    # Main reference axes. These are intentionally named by interpretation,
    # not by rank, because the goal is to test how pred_age aligns with
    # interpretable bio_age axes.
    feature_sets: dict[str, list[str]] = {
        "bio_age_ei": roi_first_order or ([roi_mean] if roi_mean in columns else []),
        "bio_age_texture": texture_cols,
        "bio_age_ei_texture": sorted(set(roi_first_order + whole_first_order + texture_cols)),
        # Supplemental axes and practical upper bounds.
        "bio_age_morphology": sorted(set(morphology_cols + morphology_extra_basic + morphology_depth_cols)),
        "bio_age_texture_metadata": sorted(set(texture_cols + metadata_cols)),
        "bio_age_ei_metadata": sorted(set(roi_first_order + metadata_cols)),
        "bio_age_full_image_upper_bound": sorted(
            set(
                roi_first_order
                + whole_first_order
                + texture_cols
                + morphology_cols
                + morphology_extra_basic
                + morphology_depth_cols
                + partition_depth_cols
                + partition_width_cols
                + partition_texture_cols
                + metadata_cols
            )
        ),
        "supplement_glcm_only": glcm_cols,
        "supplement_lbp_only": lbp_cols,
        "supplement_glrlm_glszm_only": glrlm_glszm_cols,
        "supplement_depthnorm_ei": depthnorm_cols,
        "supplement_partition_depth": partition_depth_cols,
        "supplement_partition_width": partition_width_cols,
        "supplement_partition_texture": sorted(set(partition_depth_cols + partition_width_cols + partition_texture_cols)),
        "supplement_ei_partition": sorted(set(roi_first_order + partition_depth_cols + partition_width_cols + partition_texture_cols)),
    }
    return feature_sets, notes


def build_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 31))
    elif model_name == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-3, 1, 25),
            cv=5,
            max_iter=8000,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_group_folds(df: pd.DataFrame, n_splits: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    splitter = RegressionStratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
        bin_strategy="quantile",
        n_bins=5,
    )
    x_dummy = np.zeros((len(df), 1), dtype=float)
    y = df["true_age"].to_numpy(dtype=float)
    groups = df["subject_id"].to_numpy()

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    fold_id = np.zeros(len(df), dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_dummy, y=y, groups=groups), start=1):
        folds.append((train_idx, test_idx))
        fold_id[test_idx] = fold_idx
    return folds, fold_id


def run_cv_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    seed: int,
) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=float)
    x_all = df[feature_cols]
    y_all = df["true_age"].to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in folds:
            model = build_model(model_name=model_name, seed=seed)
            model.fit(x_all.iloc[train_idx], y_all[train_idx])
            pred[test_idx] = model.predict(x_all.iloc[test_idx])
    return pred


def evaluate_predictions(df: pd.DataFrame, bio_age: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    out = df[["sample_id", "subject_id", "true_age", "pred_age", "fold"]].copy()
    out["bio_age"] = bio_age
    out["gap_pred_true"] = out["pred_age"] - out["true_age"]
    out["gap_pred_bio"] = out["pred_age"] - out["bio_age"]
    out["abs_pred_true"] = out["gap_pred_true"].abs()
    out["abs_pred_bio"] = out["gap_pred_bio"].abs()

    sample_gap_mae = float(mean_absolute_error(out["pred_age"], out["bio_age"]))
    sample_gap_rmse = rmse(out["pred_age"].to_numpy(), out["bio_age"].to_numpy())
    sample_ml_true_mae = float(mean_absolute_error(out["true_age"], out["pred_age"]))
    sample_closer_to_bio_rate = float((out["abs_pred_bio"] < out["abs_pred_true"]).mean())
    sample_within_2_rate = float((out["abs_pred_bio"] <= 2.0).mean())
    sample_within_5_rate = float((out["abs_pred_bio"] <= 5.0).mean())
    sample_within_8_rate = float((out["abs_pred_bio"] <= 8.0).mean())

    subject_out = (
        out.groupby("subject_id", as_index=False)[["true_age", "pred_age", "bio_age", "gap_pred_true", "gap_pred_bio"]]
        .mean()
    )
    subject_out["abs_pred_true"] = (subject_out["pred_age"] - subject_out["true_age"]).abs()
    subject_out["abs_pred_bio"] = (subject_out["pred_age"] - subject_out["bio_age"]).abs()
    subject_gap_mae = float(mean_absolute_error(subject_out["pred_age"], subject_out["bio_age"]))
    subject_gap_rmse = rmse(subject_out["pred_age"].to_numpy(), subject_out["bio_age"].to_numpy())
    subject_ml_true_mae = float(mean_absolute_error(subject_out["true_age"], subject_out["pred_age"]))
    subject_closer_to_bio_rate = float((subject_out["abs_pred_bio"] < subject_out["abs_pred_true"]).mean())
    subject_within_2_rate = float((subject_out["abs_pred_bio"] <= 2.0).mean())
    subject_within_5_rate = float((subject_out["abs_pred_bio"] <= 5.0).mean())
    subject_within_8_rate = float((subject_out["abs_pred_bio"] <= 8.0).mean())

    bio_age_sample_mae = float(mean_absolute_error(out["true_age"], out["bio_age"]))
    bio_age_subject_mae = float(mean_absolute_error(subject_out["true_age"], subject_out["bio_age"]))
    bio_age_vs_true_corr = safe_corr(out["true_age"].to_numpy(), out["bio_age"].to_numpy())
    if len(out) > 1 and float(out["true_age"].std()) > 1e-12:
        bio_age_bias_slope = float(np.polyfit(out["true_age"], out["bio_age"] - out["true_age"], deg=1)[0])
    else:
        bio_age_bias_slope = float("nan")
    bio_age_std = float(out["bio_age"].std(ddof=1))
    bio_age_min = float(out["bio_age"].min())
    bio_age_max = float(out["bio_age"].max())

    metrics = {
        "sample_ml_true_mae": sample_ml_true_mae,
        "subject_ml_true_mae": subject_ml_true_mae,
        "ml_sample_mae": sample_ml_true_mae,
        "ml_subject_mae": subject_ml_true_mae,
        "sample_gap_mae": sample_gap_mae,
        "subject_gap_mae": subject_gap_mae,
        "sample_gap_rmse": sample_gap_rmse,
        "subject_gap_rmse": subject_gap_rmse,
        "sample_gain": float(sample_ml_true_mae - sample_gap_mae),
        "subject_gain": float(subject_ml_true_mae - subject_gap_mae),
        "sample_closer_to_bio_rate": sample_closer_to_bio_rate,
        "subject_closer_to_bio_rate": subject_closer_to_bio_rate,
        "sample_within_2_rate": sample_within_2_rate,
        "sample_within_5_rate": sample_within_5_rate,
        "sample_within_8_rate": sample_within_8_rate,
        "subject_within_2_rate": subject_within_2_rate,
        "subject_within_5_rate": subject_within_5_rate,
        "subject_within_8_rate": subject_within_8_rate,
        "bio_age_sample_mae": bio_age_sample_mae,
        "bio_age_subject_mae": bio_age_subject_mae,
        "bio_age_vs_true_mae": bio_age_sample_mae,
        "bio_age_vs_true_corr": bio_age_vs_true_corr,
        "bio_age_bias_slope": bio_age_bias_slope,
        "bio_age_std": bio_age_std,
        "bio_age_min": bio_age_min,
        "bio_age_max": bio_age_max,
        "n_samples": int(len(out)),
        "n_subjects": int(subject_out["subject_id"].nunique()),
    }
    return out, subject_out, metrics


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


def save_bar(values: pd.DataFrame, path: Path, title: str) -> None:
    fig_h = max(5.0, 0.3 * len(values))
    plt.figure(figsize=(10, fig_h))
    plt.barh(values["feature_set"], values["subject_gap_mae"], color="#3b82f6")
    plt.gca().invert_yaxis()
    plt.xlabel("subject_gap_mae")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_metric_bar(frame: pd.DataFrame, value_col: str, path: Path, title: str, xlabel: str) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plt.figure(figsize=(10, max(4.5, 0.45 * len(plot_frame))))
    plt.barh(plot_frame["feature_set"], plot_frame[value_col], color="#3b82f6")
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_within_rate_plot(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    cols = ["subject_within_2_rate", "subject_within_5_rate", "subject_within_8_rate"]
    plot_frame = frame.copy()
    x = np.arange(len(plot_frame))
    width = 0.25
    plt.figure(figsize=(10, 5.5))
    for idx, col in enumerate(cols):
        plt.bar(x + (idx - 1) * width, plot_frame[col], width=width, label=col)
    plt.xticks(x, plot_frame["feature_set"], rotation=20, ha="right")
    plt.ylabel("覆盖率")
    plt.title("各主轴 subject 落在 bio_age 2/5/8 岁以内的比例")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_worst_subjects_plot(subject_diagnostics: pd.DataFrame, output_dir: Path) -> None:
    main = subject_diagnostics[subject_diagnostics["feature_set"].isin(MAIN_REFERENCE_AXES)].copy()
    if main.empty:
        return
    pivot = main.pivot_table(
        index="subject_id",
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
    for idx, axis in enumerate(available):
        plt.barh(y + idx * width, worst[axis], height=width, label=axis)
    plt.yticks(y + width * (len(available) - 1) / 2, [str(idx) for idx in worst.index])
    plt.gca().invert_yaxis()
    plt.xlabel("|pred_age - bio_age|")
    plt.title("主 bio_age 轴下误差最大的 subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "worst_subjects_by_bio_axis.png", dpi=180)
    plt.close()


def write_summary(
    run_dir: Path,
    leaderboard: pd.DataFrame,
    reference_rows: pd.DataFrame,
    skipped: list[str],
    feature_notes: list[str],
    extra_notes: list[str],
    input_paths: dict[str, str],
    pred_true_sample_mae: float,
    pred_true_subject_mae: float,
) -> None:
    ridge_board = leaderboard[leaderboard["model"] == "ridge"].sort_values("subject_gap_mae").reset_index(drop=True)
    main_rows = (
        ridge_board[ridge_board["feature_set"].isin(MAIN_REFERENCE_AXES)]
        .set_index("feature_set")
        .reindex(MAIN_REFERENCE_AXES)
        .dropna(subset=["n_features"])
        .reset_index()
    )
    supplemental_rows = ridge_board[ridge_board["feature_set"].isin(SUPPLEMENTAL_AXES)].copy()

    lines: list[str] = []
    lines.append("# bio_age 参考轴拟合结果汇总")
    lines.append("")
    lines.append("## 输入")
    lines.append(f"- pred_age 输入: {input_paths['pred']}")
    lines.append(f"- 缓存特征表: {input_paths['feature_table']}")
    lines.append(f"- 图像根目录: {input_paths['images']}")
    lines.append(f"- mask 根目录: {input_paths['masks']}")
    lines.append("")
    lines.append("## 目的")
    lines.append("- 本实验不以找到单一最低 MAE 年龄预测器为目标。")
    lines.append("- `bio_age` 是 interpretable image-derived biological/status age proxy，用于建立多条可解释参考轴。")
    lines.append("- 主问题是 `pred_age` 更接近哪条 `bio_age` 轴，而不是哪组特征最会预测 `true_age`。")
    lines.append("")
    lines.append("## 主参考轴")
    for axis in MAIN_REFERENCE_AXES:
        lines.append(f"- `{axis}`: {REFERENCE_AXIS_DESCRIPTIONS[axis]}")
    lines.append("")
    lines.append("## 主参考轴（Ridge）")
    lines.append("")
    lines.append("| bio_age_axis | n_features | sample_gap_mae | subject_gap_mae | sample_gain | subject_gain | sample_closer_to_bio_rate | subject_closer_to_bio_rate | bio_age_vs_true_mae | bio_age_vs_true_corr | bio_age_std |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in main_rows.iterrows():
        lines.append(
            f"| {row['feature_set']} | {int(row['n_features'])} | {row['sample_gap_mae']:.4f} | {row['subject_gap_mae']:.4f} | "
            f"{row['sample_gain']:.4f} | {row['subject_gain']:.4f} | {row['sample_closer_to_bio_rate']:.4f} | "
            f"{row['subject_closer_to_bio_rate']:.4f} | {row['bio_age_vs_true_mae']:.4f} | {row['bio_age_vs_true_corr']:.4f} | {row['bio_age_std']:.4f} |"
        )
    lines.append("")
    lines.append("## 补充轴 / upper bound")
    lines.append("")
    lines.append("| bio_age_axis | role | n_features | subject_gap_mae | subject_gain | subject_closer_to_bio_rate | bio_age_subject_mae |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in supplemental_rows.iterrows():
        role = REFERENCE_AXIS_DESCRIPTIONS.get(row["feature_set"], "补充参考轴。")
        lines.append(
            f"| {row['feature_set']} | {role} | {int(row['n_features'])} | {row['subject_gap_mae']:.4f} | "
            f"{row['subject_gain']:.4f} | {row['subject_closer_to_bio_rate']:.4f} | {row['bio_age_subject_mae']:.4f} |"
        )
    lines.append("")
    lines.append("## 合理性检查 / 中文说明")
    lines.append(f"- pred_age 与 true_age 的 MAE: sample={pred_true_sample_mae:.4f}, subject={pred_true_subject_mae:.4f}")
    lines.append("- `bio_age_vs_true_mae` / `bio_age_vs_true_corr` 只用于检查参考轴是否合理，不作为唯一主结论。")
    lines.append("- `texture_metadata` 和 full feature set 只作为 practical upper bound，不应作为唯一科学定义。")

    if skipped:
        lines.append("")
        lines.append("## 跳过项")
        for item in skipped:
            lines.append(f"- {item}")

    if feature_notes:
        lines.append("")
        lines.append("## 特征构建备注")
        for item in sorted(set(feature_notes)):
            lines.append(f"- {item}")

    if extra_notes:
        lines.append("")
        lines.append("## 图像+mask 现算备注")
        for item in extra_notes:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("## 输出清单")
    lines.append("- bio_age_reference_leaderboard.csv")
    lines.append("- bio_age_reference_summary.md")
    lines.append("- bio_age_reference_subject_diagnostics.csv")
    lines.append("- bio_age_reference_subject_error_matrix.csv")
    lines.append("- feature_sets/<feature_set>/results.csv")
    lines.append("- figures/*.png")
    lines.append("- inputs_used.json")

    (run_dir / "bio_age_reference_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark feature sets for bio_age fitting under a unified subject-level CV protocol."
    )
    parser.add_argument(
        "--pred-file",
        default="outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv",
        help="Prediction file containing pred_age.",
    )
    parser.add_argument(
        "--feature-table",
        default="outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv",
        help="Cached feature table path.",
    )
    parser.add_argument(
        "--output-root",
        default="results/bio_age_feature_benchmark",
        help="Root directory for benchmark outputs.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run name.")
    parser.add_argument("--min-age", type=float, default=18.0)
    parser.add_argument("--max-age", type=float, default=100.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-extra-image-features",
        action="store_true",
        help="Skip per-image extra feature extraction from image/mask files. Useful for faster smoke runs.",
    )
    parser.add_argument(
        "--elasticnet-sets",
        default="bio_age_ei,bio_age_texture,bio_age_ei_texture,bio_age_morphology,bio_age_texture_metadata,bio_age_full_image_upper_bound",
        help="Comma-separated feature_set names for ElasticNet robustness runs.",
    )
    parser.add_argument(
        "--single-report-root",
        default="results/reports/single_ml",
        help="Directory root for the clear single-ML report rendered after benchmark.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_bio_age_feature_benchmark"
    run_dir = (PROJECT_ROOT / args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_sets").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)

    pred_path = (PROJECT_ROOT / args.pred_file).resolve()
    feature_path = (PROJECT_ROOT / args.feature_table).resolve()

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")

    pred_df = pd.read_csv(pred_path)
    feat_df = pd.read_csv(feature_path)

    pred_columns = pred_df.columns.tolist()
    feat_columns = feat_df.columns.tolist()

    sample_col = resolve_column(pred_columns, SAMPLE_CANDIDATES, "sample_id" if "sample_id" in pred_columns else None, "sample")
    subject_col = resolve_column(pred_columns, SUBJECT_CANDIDATES, None, "subject")
    true_col = resolve_column(pred_columns, TRUE_AGE_CANDIDATES, None, "true_age")
    pred_col = resolve_column(pred_columns, PRED_COLUMN_CANDIDATES, None, "pred_age")

    if "sample_id" not in feat_df.columns:
        raise KeyError("Feature table must contain sample_id.")
    if "subject_id" not in feat_df.columns:
        raise KeyError("Feature table must contain subject_id.")

    pred_core = pred_df[[sample_col, subject_col, true_col, pred_col]].copy()
    pred_core = pred_core.rename(
        columns={sample_col: "sample_id", subject_col: "subject_id", true_col: "true_age", pred_col: "pred_age"}
    )
    pred_core = pred_core.drop_duplicates(subset=["sample_id"], keep="first")

    feat_df = feat_df.drop_duplicates(subset=["sample_id"], keep="first")

    merged = pred_core.merge(feat_df, on=["sample_id", "subject_id"], how="inner", suffixes=("", "_feat"))

    if merged.empty:
        raise ValueError("No overlap after merging prediction file and feature table.")

    # Metadata cleanup
    if "sex" in merged.columns:
        merged["meta__sex_male"] = (
            merged["sex"].astype(str).str.strip().str.upper().map({"M": 1.0, "F": 0.0})
        )

    merged["true_age"] = pd.to_numeric(merged["true_age"], errors="coerce")
    merged["pred_age"] = pd.to_numeric(merged["pred_age"], errors="coerce")
    for col in ["height_cm", "weight_kg", "bmi"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["sample_id", "subject_id", "true_age", "pred_age"]).copy()
    merged = merged[(merged["true_age"] >= args.min_age) & (merged["true_age"] <= args.max_age)].copy()

    if len(merged) < 50:
        raise ValueError(f"Too few usable rows after filtering: {len(merged)}")

    # Compute optional image+mask extra features
    if args.skip_extra_image_features:
        extra_features = merged[["sample_id"]].drop_duplicates().copy()
        extra_notes = ["Skipped per-image extra feature extraction via --skip-extra-image-features."]
    else:
        extra_cache = run_dir / "tables" / "extra_features.csv"
        extra_features, extra_notes = build_or_load_extra_features(merged, extra_cache)
    merged = merged.merge(extra_features, on="sample_id", how="left")

    feature_sets, feature_notes = build_feature_sets(merged)

    folds, fold_id = build_group_folds(merged, n_splits=args.n_splits, seed=args.seed)
    merged["fold"] = fold_id

    pred_true_sample_mae = float(mean_absolute_error(merged["pred_age"], merged["true_age"]))
    subject_base = merged.groupby("subject_id", as_index=False)[["true_age", "pred_age"]].mean()
    pred_true_subject_mae = float(mean_absolute_error(subject_base["pred_age"], subject_base["true_age"]))

    leaderboard_rows: list[dict[str, Any]] = []
    subject_diagnostics_rows: list[pd.DataFrame] = []
    skipped: list[str] = []

    elasticnet_set_names = {x.strip() for x in args.elasticnet_sets.split(",") if x.strip()}

    for feature_set_name, raw_cols in feature_sets.items():
        cols = [c for c in dict.fromkeys(raw_cols) if c in merged.columns]
        if not cols:
            skipped.append(f"{feature_set_name}: no available columns")
            continue

        numeric_frame = merged[cols].apply(pd.to_numeric, errors="coerce")
        valid_ratio = 1.0 - float(numeric_frame.isna().all(axis=1).mean())
        if valid_ratio < 0.5:
            skipped.append(f"{feature_set_name}: too many fully-missing rows (valid_ratio={valid_ratio:.3f})")
            continue

        for model_name in ["ridge", "elasticnet"]:
            if model_name == "elasticnet" and feature_set_name not in elasticnet_set_names:
                continue

            try:
                bio_age = run_cv_predict(
                    df=merged,
                    feature_cols=cols,
                    folds=folds,
                    model_name=model_name,
                    seed=args.seed,
                )
                result_df, subject_df, metrics = evaluate_predictions(merged, bio_age)
                fs_dir = run_dir / "feature_sets" / feature_set_name
                fs_dir.mkdir(exist_ok=True)
                if model_name == "ridge":
                    result_path = fs_dir / "results.csv"
                    subject_path = fs_dir / "subject_results.csv"
                else:
                    result_path = fs_dir / f"results_{model_name}.csv"
                    subject_path = fs_dir / f"subject_results_{model_name}.csv"

                result_df.to_csv(result_path, index=False)
                subject_df.to_csv(subject_path, index=False)
                subject_diag = subject_df.copy()
                subject_diag.insert(0, "model", model_name)
                subject_diag.insert(0, "feature_set", feature_set_name)
                subject_diag["subject_gain"] = subject_diag["abs_pred_true"] - subject_diag["abs_pred_bio"]
                subject_diag["subject_closer_to_bio"] = subject_diag["abs_pred_bio"] < subject_diag["abs_pred_true"]
                subject_diag["subject_within_2"] = subject_diag["abs_pred_bio"] <= 2.0
                subject_diag["subject_within_5"] = subject_diag["abs_pred_bio"] <= 5.0
                subject_diag["subject_within_8"] = subject_diag["abs_pred_bio"] <= 8.0
                subject_diag = subject_diag.rename(
                    columns={
                        "abs_pred_true": "subject_abs_pred_true",
                        "abs_pred_bio": "subject_abs_pred_bio",
                    }
                )
                subject_diagnostics_rows.append(subject_diag)

                leaderboard_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "n_features": int(len(cols)),
                        "sample_ml_true_mae": metrics["sample_ml_true_mae"],
                        "subject_ml_true_mae": metrics["subject_ml_true_mae"],
                        "ml_sample_mae": metrics["sample_ml_true_mae"],
                        "ml_subject_mae": metrics["subject_ml_true_mae"],
                        "bio_age_sample_mae": metrics["bio_age_sample_mae"],
                        "bio_age_subject_mae": metrics["bio_age_subject_mae"],
                        "sample_gain": metrics["sample_gain"],
                        "subject_gain": metrics["subject_gain"],
                        "gain_sample": metrics["sample_gain"],
                        "gain": metrics["subject_gain"],
                        "sample_closer_to_bio_rate": metrics["sample_closer_to_bio_rate"],
                        "subject_closer_to_bio_rate": metrics["subject_closer_to_bio_rate"],
                        "sample_within_2_rate": metrics["sample_within_2_rate"],
                        "sample_within_5_rate": metrics["sample_within_5_rate"],
                        "sample_within_8_rate": metrics["sample_within_8_rate"],
                        "subject_within_2_rate": metrics["subject_within_2_rate"],
                        "subject_within_5_rate": metrics["subject_within_5_rate"],
                        "subject_within_8_rate": metrics["subject_within_8_rate"],
                        "sample_gap_mae": metrics["sample_gap_mae"],
                        "subject_gap_mae": metrics["subject_gap_mae"],
                        "sample_gap_rmse": metrics["sample_gap_rmse"],
                        "subject_gap_rmse": metrics["subject_gap_rmse"],
                        "bio_age_vs_true_mae": metrics["bio_age_vs_true_mae"],
                        "bio_age_vs_true_corr": metrics["bio_age_vs_true_corr"],
                        "bio_age_bias_slope": metrics["bio_age_bias_slope"],
                        "bio_age_std": metrics["bio_age_std"],
                        "bio_age_min": metrics["bio_age_min"],
                        "bio_age_max": metrics["bio_age_max"],
                        "status": "ok",
                        "note": "",
                    }
                )
            except Exception as exc:
                skipped.append(f"{feature_set_name}/{model_name}: {type(exc).__name__}: {exc}")

    if not leaderboard_rows:
        raise RuntimeError("No feature set finished successfully.")

    leaderboard = pd.DataFrame(leaderboard_rows)

    leaderboard = leaderboard.sort_values(["model", "subject_gap_mae", "sample_gap_mae"]).reset_index(drop=True)
    leaderboard.to_csv(run_dir / "bio_age_reference_leaderboard.csv", index=False)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    subject_diagnostics = pd.concat(subject_diagnostics_rows, ignore_index=True) if subject_diagnostics_rows else pd.DataFrame()
    if not subject_diagnostics.empty:
        subject_diagnostics.to_csv(run_dir / "bio_age_reference_subject_diagnostics.csv", index=False)
        main_subject_matrix = (
            subject_diagnostics[subject_diagnostics["model"] == "ridge"]
            .pivot_table(
                index="subject_id",
                columns="feature_set",
                values="subject_abs_pred_bio",
                aggfunc="mean",
            )
            .reset_index()
        )
        if not main_subject_matrix.empty:
            main_subject_matrix.to_csv(run_dir / "bio_age_reference_subject_error_matrix.csv", index=False)

    ridge_board = leaderboard[leaderboard["model"] == "ridge"].sort_values("subject_gap_mae").reset_index(drop=True)
    if ridge_board.empty:
        raise RuntimeError("Ridge results are empty. Cannot build main summary.")

    reference_rows = ridge_board[ridge_board["feature_set"].isin(MAIN_REFERENCE_AXES)].copy()
    if set(MAIN_REFERENCE_AXES) - set(reference_rows["feature_set"]):
        missing_axes = sorted(set(MAIN_REFERENCE_AXES) - set(reference_rows["feature_set"]))
        raise RuntimeError(f"Missing main bio_age reference axes: {missing_axes}")

    # Figures
    best_set = "bio_age_ei_texture"
    best_results = pd.read_csv(run_dir / "feature_sets" / best_set / "results.csv")
    baseline_results = pd.read_csv(run_dir / "feature_sets" / "bio_age_ei" / "results.csv")

    save_scatter(
        x=baseline_results["true_age"].to_numpy(),
        y=baseline_results["pred_age"].to_numpy(),
        path=run_dir / "figures" / "pred_age_vs_true_age.png",
        title="pred_age 与 true_age",
        xlabel="true_age",
        ylabel="pred_age",
    )
    for axis in MAIN_REFERENCE_AXES:
        axis_results = pd.read_csv(run_dir / "feature_sets" / axis / "results.csv")
        save_scatter(
            x=axis_results["bio_age"].to_numpy(),
            y=axis_results["pred_age"].to_numpy(),
            path=run_dir / "figures" / f"pred_age_vs_{axis}.png",
            title=f"pred_age 与 {axis}",
            xlabel=axis,
            ylabel="pred_age",
        )
        save_scatter(
            x=axis_results["true_age"].to_numpy(),
            y=axis_results["bio_age"].to_numpy(),
            path=run_dir / "figures" / f"{axis}_vs_true_age.png",
            title=f"{axis} 与 true_age",
            xlabel="true_age",
            ylabel=axis,
        )

    bar_df = ridge_board[["feature_set", "subject_gap_mae"]].copy()
    save_bar(bar_df, run_dir / "figures" / "subject_gap_mae_by_feature_set.png", "各特征轴 subject-level gap_mae（Ridge）")
    ridge_main = ridge_board[ridge_board["feature_set"].isin(MAIN_REFERENCE_AXES)].copy()
    save_metric_bar(
        ridge_main,
        "subject_gain",
        run_dir / "figures" / "subject_gain_by_bio_axis.png",
        "主 bio_age 轴的 subject_gain",
        "subject_gain",
    )
    save_metric_bar(
        ridge_main,
        "subject_closer_to_bio_rate",
        run_dir / "figures" / "subject_closer_to_bio_rate_by_axis.png",
        "主 bio_age 轴的 subject_closer_to_bio_rate",
        "subject_closer_to_bio_rate",
    )
    save_within_rate_plot(ridge_main, run_dir / "figures" / "within_2_5_8_coverage_by_axis.png")
    if not subject_diagnostics.empty:
        save_worst_subjects_plot(
            subject_diagnostics[subject_diagnostics["model"] == "ridge"].copy(),
            run_dir,
        )

    plt.figure(figsize=(7, 5.5))
    residual = best_results["bio_age"].to_numpy() - best_results["true_age"].to_numpy()
    plt.scatter(best_results["true_age"], residual, s=10, alpha=0.55)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("true_age")
    plt.ylabel("bio_age - true_age")
    plt.title(f"残差图：bio_age - true_age（{best_set}）")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "best_bio_age_residual_vs_age.png", dpi=180)
    plt.close()

    inputs_used = {
        "pred": str(pred_path),
        "feature_table": str(feature_path),
        "images": "/home/szdx/LNX/data/TA/Healthy/Images",
        "masks": "/home/szdx/LNX/data/TA/Healthy/Masks",
        "n_rows_after_merge_age_filter": int(len(merged)),
        "n_subjects": int(merged["subject_id"].nunique()),
        "pred_true_sample_mae": pred_true_sample_mae,
        "pred_true_subject_mae": pred_true_subject_mae,
        "lbp_available": HAS_LBP,
        "skip_extra_image_features": bool(args.skip_extra_image_features),
    }
    (run_dir / "inputs_used.json").write_text(json.dumps(inputs_used, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_summary(
        run_dir=run_dir,
        leaderboard=leaderboard,
        reference_rows=reference_rows,
        skipped=skipped,
        feature_notes=feature_notes,
        extra_notes=extra_notes,
        input_paths=inputs_used,
        pred_true_sample_mae=pred_true_sample_mae,
        pred_true_subject_mae=pred_true_subject_mae,
    )
    report_dir = build_single_ml_report(
        bio_age_run=run_dir,
        output_root=(PROJECT_ROOT / args.single_report_root).resolve(),
        model="ridge",
    )

    print("=== bio_age 参考轴拟合完成 ===")
    print(f"run_dir: {run_dir}")
    print(f"single_report_dir: {report_dir}")
    print("按 pred_age 与 bio_age 的 subject_gap_mae 排序的 Top-5 Ridge 结果：")
    show_cols = [
        "feature_set",
        "n_features",
        "subject_ml_true_mae",
        "sample_gap_mae",
        "subject_gap_mae",
        "subject_gain",
        "subject_closer_to_bio_rate",
        "bio_age_subject_mae",
        "bio_age_vs_true_corr",
        "bio_age_std",
    ]
    print(ridge_board[show_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
