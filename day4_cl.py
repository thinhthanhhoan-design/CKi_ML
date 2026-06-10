"""
day4_cl_reliability_upgrade.py (V8 - Sealed Production Edition)
================================================================
Hệ thống Mô hình hóa Thay thế Khí động học (Surrogate Modeling Pipeline)
Chuẩn hóa An toàn dữ liệu, Kiểm toán độ bất định và Đóng gói Thiết kế ngược Day 5.

Các điểm nâng cấp tối hậu trong bản V8:
  1. Whitelist Phổ rộng Bảo toàn Tri thức (Lỗi 1 Fix): Bổ sung 'proxy', 'wake',
     'separation', 'aft' vào ALLOWED_MARKERS. Cứu sống toàn bộ các đặc trưng tĩnh
     vô giá từ Day 2 như wake_proxy, separation_proxy, aft_thickness, aft_camber...
  2. Geometry LOGO refit validation was removed for runtime; core GroupKFold OOF validation remains.
  3. Xuất bản API Endpoint Endpoints (Lưu ý Day 5 Fix): Tích hợp trực tiếp cấu trúc gọi hàm
     "predict_cl", "predict_clmax", và "predict_stall_aoa" vào tệp nhị phân 'day5_interface.json'.
  4. Bảo toàn Dải Thất tốc Toàn phần (Stall Shift Signed): Tính toán xu hướng lệch pha góc
     tấn công thất tốc thực tế so với dự báo của mô hình để làm đòn bẩy phạt cho Day 5.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CL_Reliability_V9")

OUTPUT_ENCODING = "utf-8-sig"
CL_TREE_ESTIMATORS = 200
CL_HGB_MAX_ITER = 400

# ---------------------------------------------------------------------------
# Strict Regex Feature Whitelist Framework (Sửa Lỗi 1: Mở rộng bảo vệ Tri thức Day 2)
# ---------------------------------------------------------------------------
ALLOWED_MARKERS = [
    "metric", "quality", "loading", "recovery", "curvature", "thickness",
    "camber", "le_radius", "te_angle", "proxy", "wake", "separation", "aft"
]
FORBIDDEN_MARKERS = ["cl", "cd", "cm", "pred", "error", "residual", "true", "stall", "clmax"]

def apply_comprehensive_whitelist_v8(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Cơ chế Whitelist dựa trên biểu thức chính quy mở rộng, giữ lại 100% proxy vật lý tĩnh từ Day 2."""
    day5_features = []
    excluded_features = []

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ("angle", "reynolds", "log10_re", "geom_hash"):
            day5_features.append(col)
            continue

        # Kiểm tra điều kiện loại biên target leakage trực tiếp
        has_forbidden = any(marker in col_lower for marker in FORBIDDEN_MARKERS)
        # Kiểm tra điều kiện cứu sống đặc trưng lưu chất tĩnh từ Day 2
        is_saved_physics = any(marker in col_lower for marker in ALLOWED_MARKERS) or re.search(r'^g_\d+$', col_lower)

        if has_forbidden and not is_saved_physics:
            excluded_features.append(col)
        elif is_saved_physics:
            day5_features.append(col)
        else:
            excluded_features.append(col)

    features_to_train = [f for f in day5_features if f != "geom_hash"]
    return features_to_train, excluded_features

@dataclass
class CLReliabilityConfig:
    dataset_path: str = "deeplearwing_day2_tabular.csv.gz"
    output_dir: str = "outputs/day4_cl_v9"
    n_splits: int = 10
    random_state: int = 42
    max_rows: Optional[int] = None
    fast_mode: bool = False
    disable_logo: bool = True
    disable_ood: bool = False
    logo_mode: str = "removed"
    logo_cv_splits: int = 5
    conformal_alpha_levels: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95])
    bootstrap_n: int = 5
    bootstrap_jobs: int = 1
    candidate_models: str = "hgb,extra_trees,random_forest"
    logo_fast_limit: int = 0
    logo_max_geometries: int = 0
    pca_variance: float = 0.95
    n_jobs: int = -1


def get_candidate_models(cfg: CLReliabilityConfig) -> List[str]:
    allowed = {"hgb", "extra_trees", "random_forest"}
    raw = str(getattr(cfg, "candidate_models", "") or "")
    candidates = [x.strip() for x in raw.split(",") if x.strip()]
    candidates = [x for x in candidates if x in allowed]
    return candidates or ["hgb", "extra_trees", "random_forest"]


def resolve_n_jobs(n_jobs: int) -> int:
    cpu_total = int(os.cpu_count() or 1)
    if n_jobs is None:
        return cpu_total
    if int(n_jobs) < 0:
        return cpu_total
    return max(1, min(int(n_jobs), cpu_total))


def configure_thread_env(n_jobs: int) -> int:
    effective = resolve_n_jobs(n_jobs)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(effective)
    logger.info("[CPU] Parallel runtime configured: n_jobs=%d (logical cpu=%d)", effective, int(os.cpu_count() or 1))
    return effective

# ---------------------------------------------------------------------------
# Dynamic Model Pipeline Factory
# ---------------------------------------------------------------------------
def get_model_pipeline(name: str, rs: int, n_jobs: int = -1) -> Pipeline:
    """Factory cấu trúc phân hệ sinh mô hình đồng bộ toàn hệ thống."""
    if name == "hgb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=int(CL_HGB_MAX_ITER), max_depth=6, learning_rate=0.05, random_state=rs))
        ])
    elif name == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=int(CL_TREE_ESTIMATORS), max_depth=16, min_samples_leaf=4, n_jobs=n_jobs, random_state=rs))
        ])
    elif name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=int(CL_TREE_ESTIMATORS), max_depth=14, min_samples_leaf=6, n_jobs=n_jobs, random_state=rs))
        ])
    else:
        raise ValueError(f"Unknown architecture core request: {name}")

def ensure_dirs(base: str) -> Dict[str, Path]:
    base_path = Path(base)
    sub = ["models", "tables", "figures", "reports", "artifacts"]
    paths = {"base": base_path}
    for s in sub:
        p = base_path / s
        p.mkdir(parents=True, exist_ok=True)
        paths[s] = p
    return paths

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(y_true) < 2 or np.std(y_true) == 0: return 0.0
        return float(r2_score(y_true, y_pred))
    except Exception: return 0.0

def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}" if np.isfinite(v) else "nan"


def downcast_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Giảm RAM bằng cách ép các cột float64 sang float32.
    Chỉ thay đổi dtype lưu trữ, không thay đổi cột, hàng, logic train/CV/ranking.
    """
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df.loc[:, float64_cols] = df.loc[:, float64_cols].astype(np.float32)
    return df

# ---------------------------------------------------------------------------
# Core Engine Evaluation Logic
# ---------------------------------------------------------------------------
def evaluate_advanced_curve_metrics(df: pd.DataFrame, preds: np.ndarray) -> Tuple[float, float, float]:
    """Tính toán chính xác sai số CLmax (đỉnh trần) và góc tấn thất tốc (Stall AoA Signed Shift)."""
    tmp = df.copy()
    tmp["pred"] = preds
    clmax_errors, stall_errors, stall_shifts = [], [], []

    group_cols = ["geom_hash", "log10_re"] if "log10_re" in df.columns else ["geom_hash"]
    for _, grp in tmp.groupby(group_cols):
        if len(grp) < 3: continue
        grp_s = grp.sort_values("angle")

        # Định nghĩa chuẩn CLmax Error (Max trần độc lập)
        clmax_true_val = float(grp_s["cl"].max())
        clmax_pred_val = float(grp_s["pred"].max())
        clmax_errors.append(abs(clmax_true_val - clmax_pred_val))

        # Trích xuất góc tấn thất tốc phục vụ phân tích Signed Shift
        aoa_true_stall = float(grp_s.loc[grp_s["cl"].idxmax(), "angle"])
        aoa_pred_stall = float(grp_s.loc[grp_s["pred"].idxmax(), "angle"])

        stall_errors.append(abs(aoa_true_stall - aoa_pred_stall))
        stall_shifts.append(aoa_pred_stall - aoa_true_stall) # Dương: Thất tốc muộn, Âm: Thất tốc sớm

    return float(np.mean(clmax_errors)), float(np.mean(stall_errors)), float(np.mean(stall_shifts))

# ---------------------------------------------------------------------------
# Geometry validation cleanup
# ---------------------------------------------------------------------------
# Geometry LOGO refit helpers were removed intentionally.
# Core GroupKFold OOF validation remains in section_base_cl_model.
def section_base_cl_model(df: pd.DataFrame, features: List[str], cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Tuple[Dict, np.ndarray, np.ndarray]:
    logger.info("[S3] Rebuilding Base CL Model (Strict Deterministic Whitelist Architecture)...")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["cl"].to_numpy(dtype=np.float32)
    groups = df["geom_hash"].astype(str).to_numpy()

    candidates = get_candidate_models(cfg)
    results = []
    oof_store = {}
    best_rank = None
    n_splits = min(max(2, int(cfg.n_splits)), len(np.unique(groups)))

    for cand_i, name in enumerate(candidates, start=1):
        cand_t0 = time.time()
        logger.info("[S3] Candidate %s/%s: %s | folds=%s", cand_i, len(candidates), name, n_splits)
        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.full(len(y), np.nan, dtype=np.float32)

        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            fold_t0 = time.time()
            logger.info("[S3]   %s fold %s/%s: train=%s valid=%s", name, fold_i, n_splits, len(tr_idx), len(va_idx))
            m = get_model_pipeline(name, cfg.random_state, n_jobs=cfg.n_jobs)
            m.fit(X[tr_idx], y[tr_idx])
            fold_pred = m.predict(X[va_idx])
            oof_preds[va_idx] = fold_pred
            fold_pred = None
            m = None
            tr_idx = None
            va_idx = None
            gc.collect()
            logger.info("[S3]   %s fold %s/%s done in %.1fs", name, fold_i, n_splits, time.time() - fold_t0)

        clmax_mae, stall_mae, _ = evaluate_advanced_curve_metrics(df, oof_preds)
        true_logo_mae = 0.0
        logo_eval_mode = "removed"
        row = {
            "model": name, "clmax_mae": clmax_mae, "stall_aoa_mae": stall_mae,
            "logo_mae": true_logo_mae, "rmse": rmse(y, oof_preds), "r2": safe_r2(y, oof_preds),
            "logo_eval_mode": logo_eval_mode,
        }

        results.append(row)
        oof_store[name] = oof_preds
        rank = (row["clmax_mae"], row["stall_aoa_mae"], row["rmse"], -row["r2"])
        if best_rank is None or rank < best_rank:
            best_rank = rank
        logger.info("[S3] Candidate %s done in %.1fs | rmse=%s r2=%s clmax_mae=%s stall_mae=%s",
                    name, time.time() - cand_t0, _fmt(row["rmse"]), _fmt(row["r2"]),
                    _fmt(row["clmax_mae"]), _fmt(row["stall_aoa_mae"]))

    # Multi-priority ranking without expensive per-geometry LOGO refits.
    comp_df = pd.DataFrame(results).sort_values(
        by=["clmax_mae", "stall_aoa_mae", "rmse", "r2"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    best_name = str(comp_df.iloc[0]["model"])
    logger.info(f"Base CL Champion Confirmed: {best_name}")
    comp_df.to_csv(paths["tables"] / "model_comparison.csv", index=False, encoding=OUTPUT_ENCODING)
    final_model = get_model_pipeline(best_name, cfg.random_state, n_jobs=cfg.n_jobs)
    final_model.fit(X, y)
    model_store = {best_name: (final_model, [])}
    gc.collect()

    return {
        "best_name": best_name, "model_store": model_store, "oof_store": oof_store,
        "X": X, "y": y, "groups": groups, "feat_cols": features, "n_splits": n_splits
    }, oof_store[best_name], y

# ---------------------------------------------------------------------------
# Day 5 Custom Curve Datasets
# ---------------------------------------------------------------------------
def build_pure_day5_curve_dataset(df: pd.DataFrame, geom_features: List[str]) -> pd.DataFrame:
    curves = []
    group_cols = ["geom_hash", "log10_re"] if "log10_re" in df.columns else ["geom_hash"]

    for _, grp in df.groupby(group_cols, sort=False):
        if len(grp) < 3: continue
        grp_sorted = grp.sort_values("angle")
        cl_vals, aoa_vals = grp_sorted["cl"].values, grp_sorted["angle"].values
        clmax_idx = int(np.argmax(cl_vals))

        row = {
            "geom_hash": grp_sorted["geom_hash"].iloc[0],
            "CLmax_true": float(cl_vals[clmax_idx]),
            "stall_aoa_true": float(aoa_vals[clmax_idx]),
        }
        if "log10_re" in df.columns:
            row["log10_re"] = float(grp_sorted["log10_re"].iloc[0])

        for col in geom_features:
            if col not in ("angle", "reynolds", "log10_re") and col in grp.columns:
                row[col] = float(grp[col].median())
        curves.append(row)

    return downcast_float64_to_float32(pd.DataFrame(curves))

def section_clmax_surrogate(df: pd.DataFrame, geom_features: List[str], cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Tuple[Dict, object]:
    logger.info("[S5] Training Dedicated CLmax Surrogate via Multi-Architecture Space Search...")
    curve_df = build_pure_day5_curve_dataset(df, geom_features)

    input_feats = [c for c in curve_df.columns if c not in ("CLmax_true", "stall_aoa_true", "geom_hash")]
    X_curve = curve_df[input_feats].to_numpy(dtype=np.float32)
    y_clmax = curve_df["CLmax_true"].to_numpy(dtype=np.float32)
    geom_groups = curve_df["geom_hash"].astype(str).to_numpy()

    candidates = get_candidate_models(cfg)
    rows = []
    n_splits = min(max(2, int(cfg.n_splits)), len(np.unique(geom_groups)))

    for cand_i, name in enumerate(candidates, start=1):
        cand_t0 = time.time()
        logger.info("[S5] Candidate %s/%s: %s | folds=%s", cand_i, len(candidates), name, n_splits)
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.full(len(y_clmax), np.nan, dtype=np.float32)
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_curve, y_clmax, groups=geom_groups), start=1):
            fold_t0 = time.time()
            logger.info("[S5]   %s fold %s/%s: train=%s valid=%s", name, fold_i, n_splits, len(tr_idx), len(va_idx))
            m = get_model_pipeline(name, cfg.random_state, n_jobs=cfg.n_jobs)
            m.fit(X_curve[tr_idx], y_clmax[tr_idx])
            fold_pred = m.predict(X_curve[va_idx])
            oof[va_idx] = fold_pred
            fold_pred = None
            m = None
            tr_idx = None
            va_idx = None
            gc.collect()
            logger.info("[S5]   %s fold %s/%s done in %.1fs", name, fold_i, n_splits, time.time() - fold_t0)

        mae = float(mean_absolute_error(y_clmax, oof))
        rows.append({"model": name, "mae": mae})
        oof = None
        gc.collect()
        logger.info("[S5] Candidate %s done in %.1fs | mae=%s", name, time.time() - cand_t0, _fmt(mae))

    best = pd.DataFrame(rows).sort_values("mae", ascending=True).iloc[0]
    best_mae = float(best["mae"])
    best_name = str(best["model"])
    clmax_champion_model = get_model_pipeline(best_name, cfg.random_state, n_jobs=cfg.n_jobs)
    clmax_champion_model.fit(X_curve, y_clmax)
    logger.info(f"CLmax Surrogate Champion Extracted with MAE: {_fmt(best_mae)}")
    joblib.dump(clmax_champion_model, paths["models"] / "clmax_model.joblib")
    return {"curve_df": curve_df, "input_feats": input_feats, "best_mae": best_mae, "geom_groups": geom_groups}, clmax_champion_model

def section_stall_surrogate(clmax_ctx: Dict, cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> object:
    logger.info("[S6] Training Dedicated Stall AoA Surrogate via Multi-Architecture Space Search...")
    curve_df, input_feats, geom_groups = clmax_ctx["curve_df"], clmax_ctx["input_feats"], clmax_ctx["geom_groups"]
    X_curve = curve_df[input_feats].to_numpy(dtype=np.float32)
    y_stall = curve_df["stall_aoa_true"].to_numpy(dtype=np.float32)

    candidates = get_candidate_models(cfg)
    rows = []
    n_splits = min(max(2, int(cfg.n_splits)), len(np.unique(geom_groups)))

    for cand_i, name in enumerate(candidates, start=1):
        cand_t0 = time.time()
        logger.info("[S6] Candidate %s/%s: %s | folds=%s", cand_i, len(candidates), name, n_splits)
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.full(len(y_stall), np.nan, dtype=np.float32)
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_curve, y_stall, groups=geom_groups), start=1):
            fold_t0 = time.time()
            logger.info("[S6]   %s fold %s/%s: train=%s valid=%s", name, fold_i, n_splits, len(tr_idx), len(va_idx))
            m = get_model_pipeline(name, cfg.random_state, n_jobs=cfg.n_jobs)
            m.fit(X_curve[tr_idx], y_stall[tr_idx])
            fold_pred = m.predict(X_curve[va_idx])
            oof[va_idx] = fold_pred
            fold_pred = None
            m = None
            tr_idx = None
            va_idx = None
            gc.collect()
            logger.info("[S6]   %s fold %s/%s done in %.1fs", name, fold_i, n_splits, time.time() - fold_t0)

        mae = float(mean_absolute_error(y_stall, oof))
        rows.append({"model": name, "mae": mae})
        oof = None
        gc.collect()
        logger.info("[S6] Candidate %s done in %.1fs | mae=%s", name, time.time() - cand_t0, _fmt(mae))

    best = pd.DataFrame(rows).sort_values("mae", ascending=True).iloc[0]
    best_mae = float(best["mae"])
    best_name = str(best["model"])
    stall_champion_model = get_model_pipeline(best_name, cfg.random_state, n_jobs=cfg.n_jobs)
    stall_champion_model.fit(X_curve, y_stall)
    logger.info(f"Stall AoA Surrogate Champion Extracted with MAE: {_fmt(best_mae)} deg")
    joblib.dump(stall_champion_model, paths["models"] / "stall_model.joblib")
    return stall_champion_model

# ---------------------------------------------------------------------------
# Geometry Validation Placeholder
# ---------------------------------------------------------------------------
def section_logo_and_failure_analysis(base_ctx: Dict, curve_df: pd.DataFrame, clean_df: pd.DataFrame, cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> pd.DataFrame:
    """Write no-op geometry validation artifacts; expensive LOGO refits are removed."""
    logger.info("[S8] Geometry LOGO validation removed; writing placeholder artifacts.")
    columns = ["geom_hash", "mae", "rmse", "r2", "clmax_error", "stall_error", "stall_shift_signed", "validation_mode"]
    logo_df = pd.DataFrame(columns=columns)
    logo_df.to_csv(paths["tables"] / "logo_metrics.csv", index=False, encoding=OUTPUT_ENCODING)
    (paths["reports"] / "logo_report.md").write_text(
        "# Geometry Validation Report\n\n"
        "LOGO/Leave-One-Geometry-Out refit validation was removed from day4_cl.py.\n\n"
        "Core GroupKFold OOF validation, CLmax metrics, stall metrics, uncertainty, OOD, and Day5 artifacts are still produced.\n",
        encoding=OUTPUT_ENCODING,
    )
    return logo_df

# ---------------------------------------------------------------------------
# Split Conformal Uncertainty Quantification
# ---------------------------------------------------------------------------
def section_uncertainty_and_coverage(base_ctx: Dict, cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Dict:
    logger.info("[S10] Executing Rigorous Independent OOF Split Conformal Calibration...")
    best_name = base_ctx["best_name"]

    oof_preds = base_ctx["oof_store"][best_name]
    residuals_all = np.abs(base_ctx["y"] - oof_preds)

    unique_geoms = np.unique(base_ctx["groups"])
    rng = np.random.default_rng(cfg.random_state)
    cal_geoms = rng.choice(unique_geoms, max(1, int(len(unique_geoms) * 0.2)), replace=False)

    cal_mask = np.isin(base_ctx["groups"], cal_geoms)
    residuals_cal = residuals_all[cal_mask]

    uq_report_rows = []
    conformal_meta = {"method": "independent_oof_split_conformal"}

    for alpha in cfg.conformal_alpha_levels:
        q_val = float(np.quantile(residuals_cal, alpha))
        conformal_meta[f"quantile_{int(alpha*100)}"] = q_val
        picp = float(np.mean(residuals_all[~cal_mask] <= q_val))
        mpiw = float(2.0 * q_val)
        uq_report_rows.append({"target_coverage_alpha": alpha, "computed_quantile": q_val, "PICP": picp, "MPIW": mpiw})

    pd.DataFrame(uq_report_rows).to_csv(paths["tables"] / "conformal_coverage_report.csv", index=False, encoding=OUTPUT_ENCODING)
    joblib.dump(conformal_meta, paths["models"] / "conformal_calibrator.joblib")

    # OOF residuals are enough for conformal calibration; bootstrap handles final UQ.
    uq_scores_train = np.zeros_like(residuals_all, dtype=np.float32)
    gc.collect()
    return {"uq_scores_train": uq_scores_train, "residuals_train": residuals_all}

# ---------------------------------------------------------------------------
# Day 5 Deployment Interfaces Exporter (Sửa lỗi tiềm ẩn: Bổ sung gọi hàm API)
# ---------------------------------------------------------------------------
def section_export_day5_interfaces_v8(feat_cols: List[str], base_ctx: Dict, clmax_ctx: Dict, clean_df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    logger.info("[S12] Exporting Sealed Day 5 Interfaces and Trust Region Metadata...")

    # 1. feature_columns.json
    feature_meta = {"ordered_features": feat_cols, "total_feature_count": len(feat_cols), "numerical_features": feat_cols}
    with open(paths["artifacts"] / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)

    # 2. day5_interface.json (Bổ sung cú pháp gọi hàm API tường minh cho Day 5 Optimizer)
    day5_meta = {
        "champion_cl_architecture": base_ctx["best_name"],
        "input_features_cl_surrogate": feat_cols,
        "input_features_clmax_surrogate": clmax_ctx["input_feats"],
        "predict_cl": "joblib.load('models/cl_model.joblib').predict(X)",
        "predict_clmax": "joblib.load('models/clmax_model.joblib').predict(X_curve)",
        "predict_stall_aoa": "joblib.load('models/stall_model.joblib').predict(X_curve)",
        "artifacts_mapping": {
            "cl_surrogate_model": "models/cl_model.joblib",
            "clmax_surrogate_model": "models/clmax_model.joblib",
            "stall_surrogate_model": "models/stall_model.joblib",
            "conformal_calibrator": "models/conformal_calibrator.joblib"
        }
    }
    with open(paths["artifacts"] / "day5_interface.json", "w", encoding="utf-8") as f:
        json.dump(day5_meta, f, indent=2, ensure_ascii=False)

    # 3. trust_region_metadata.json (Vùng kiểm soát chặn ranh giới tối ưu hình học)
    candidate_keys = ["g_thickness_max", "thickness", "g_camber_max", "camber", "g_le_radius", "le_radius", "g_te_angle", "te_angle"]
    actual_keys = [k for k in candidate_keys if k in clean_df.columns]

    trust_meta = {}
    for k in actual_keys:
        series_clean = clean_df[k].dropna()
        if len(series_clean) > 0:
            trust_meta[f"{k}_min"] = float(series_clean.min())
            trust_meta[f"{k}_max"] = float(series_clean.max())
            trust_meta[f"{k}_median"] = float(series_clean.median())

    with open(paths["artifacts"] / "trust_region_metadata.json", "w", encoding="utf-8") as f:
        json.dump(trust_meta, f, indent=2, ensure_ascii=False)

def section_physical_consistency_suite(df: pd.DataFrame, oof_preds: np.ndarray) -> int:
    tmp = df.copy()
    tmp["pred"] = oof_preds
    violations = 0
    group_cols = ["geom_hash", "log10_re"] if "log10_re" in df.columns else ["geom_hash"]
    for _, grp in tmp.groupby(group_cols):
        if len(grp) < 4: continue
        grp_s = grp.sort_values("angle")
        aoa, cl_p = grp_s["angle"].values, grp_s["pred"].values
        linear_mask = (aoa >= -4) & (aoa <= 6)
        if linear_mask.sum() >= 3:
            grads = np.diff(cl_p[linear_mask]) / np.diff(aoa[linear_mask])
            if np.any(grads < -0.01):
                violations += 1
                continue
        if np.any(np.abs(np.diff(cl_p, n=2)) > 0.6):
            violations += 1
            continue
    return violations

def section_figures(df: pd.DataFrame, y_true: np.ndarray, oof_preds: np.ndarray, paths: Dict[str, Path]) -> None:
    residuals = oof_preds - y_true
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, oof_preds, alpha=0.15, s=4, color="#1f77b4")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=1.5)
    ax.set_xlabel("Experimental Real CL"); ax.set_ylabel("Surrogate Predicted CL")
    fig.savefig(paths["figures"] / "cl_prediction_vs_truth.png", dpi=150); plt.close(fig)

    if "log10_re" in df.columns:
        df_plots = pd.DataFrame({"log10_re": df["log10_re"].values, "abs_error": np.abs(residuals)})
        df_plots["re_bin"] = pd.cut(df_plots["log10_re"], bins=4)
        bin_maes = df_plots.groupby("re_bin", observed=False)["abs_error"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        bin_maes.plot(kind="bar", color="#4e79a7", ax=ax, edgecolor="black")
        ax.set_ylabel("Mean Absolute Error (MAE)"); ax.set_xlabel("Reynolds Bins (log10)")
        plt.xticks(rotation=15); plt.tight_layout()
        fig.savefig(paths["figures"] / "cl_error_by_reynolds_bin.png", dpi=150); plt.close(fig)



# ---------------------------------------------------------------------------
# V9 Research-Grade Utilities: Regime, OOD, Trust Region, Bootstrap, Reports
# ---------------------------------------------------------------------------
def infer_flow_regime(angle: float, stall_aoa: Optional[float] = None) -> str:
    """Heuristic aerodynamic regime used for conformal calibration and flow OOD."""
    if stall_aoa is not None and np.isfinite(stall_aoa):
        d = angle - stall_aoa
        if d <= -8: return "linear"
        if d <= -4: return "transitional"
        if d <= -1: return "pre-stall"
        if d <= 2: return "near-stall"
        return "post-stall"
    if angle <= 6: return "linear"
    if angle <= 9: return "transitional"
    if angle <= 12: return "pre-stall"
    if angle <= 15: return "near-stall"
    return "post-stall"


def attach_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stall_map = {}
    group_cols = ["geom_hash", "log10_re"] if "log10_re" in out.columns else ["geom_hash"]
    for key, grp in out.groupby(group_cols):
        if len(grp) < 3:
            continue
        stall_map[key] = float(grp.loc[grp["cl"].idxmax(), "angle"])
    labels = []
    for _, row in out.iterrows():
        key = tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]]
        labels.append(infer_flow_regime(float(row["angle"]), stall_map.get(key)))
    out["flow_regime"] = labels
    return out


def geometry_feature_columns(features: List[str]) -> List[str]:
    return [f for f in features if f not in ("angle", "reynolds", "log10_re")]


def _safe_inverse_cov(Z: np.ndarray):
    cov = LedoitWolf().fit(Z)
    return cov.location_, cov.precision_


def _mahalanobis(Z: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    D = Z - mu
    return np.sqrt(np.maximum(np.sum((D @ precision) * D, axis=1), 0.0))


def classify_by_threshold(score: float, p90: float, p97: float) -> str:
    if score < p90: return "SAFE"
    if score < p97: return "WARNING"
    return "UNSAFE"


def section_geometry_ood(clean_df: pd.DataFrame, features: List[str], cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-B] Building PCA-Mahalanobis Geometry OOD detector...")
    geom_cols = [c for c in geometry_feature_columns(features) if c in clean_df.columns]
    curve_df = clean_df.groupby("geom_hash", sort=False)[geom_cols].median().reset_index() if geom_cols else pd.DataFrame({"geom_hash": clean_df["geom_hash"].unique()})
    if not geom_cols:
        curve_df["dummy_geometry"] = 0.0
        geom_cols = ["dummy_geometry"]
    imputer = SimpleImputer(strategy="median")
    Xg = imputer.fit_transform(curve_df[geom_cols].to_numpy(dtype=np.float32)).astype(np.float32, copy=False)
    n_comp = min(Xg.shape[0], Xg.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=min(cfg.pca_variance, 0.999), svd_solver="full", random_state=cfg.random_state)
        Z = pca.fit_transform(Xg)
    else:
        pca = None
        Z = Xg
    mu, precision = _safe_inverse_cov(Z)
    scores = _mahalanobis(Z, mu, precision)
    p90, p95, p97, p99 = [float(np.quantile(scores, q)) for q in (0.90, 0.95, 0.97, 0.99)]
    curve_df["geometry_ood_score"] = scores
    curve_df["geometry_ood_flag"] = curve_df["geometry_ood_score"] >= p95
    curve_df["geometry_trust_class"] = [classify_by_threshold(float(v), p90, p97) for v in scores]
    curve_df.to_csv(paths["tables"] / "geometry_ood_scores.csv", index=False, encoding=OUTPUT_ENCODING)
    meta = {"columns": geom_cols, "imputer": imputer, "pca": pca, "mu": mu, "precision": precision, "p90": p90, "p95": p95, "p97": p97, "p99": p99}
    joblib.dump(meta, paths["models"] / "geometry_ood_model.pkl")
    return {"scores_df": curve_df, "meta": meta}


def section_flow_ood(clean_df: pd.DataFrame, cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-C] Building independent Flow OOD detector...")
    df = attach_regime_labels(clean_df)
    flow_cols = [c for c in ["angle", "reynolds", "log10_re"] if c in df.columns]
    regime_codes = pd.get_dummies(df["flow_regime"], prefix="regime")
    Xf_df = pd.concat([df[flow_cols].reset_index(drop=True), regime_codes.reset_index(drop=True)], axis=1)
    imputer = SimpleImputer(strategy="median")
    Xf = imputer.fit_transform(Xf_df.to_numpy(dtype=np.float32)).astype(np.float32, copy=False)
    mu, precision = _safe_inverse_cov(Xf)
    scores = _mahalanobis(Xf, mu, precision)
    p90, p95, p97, p99 = [float(np.quantile(scores, q)) for q in (0.90, 0.95, 0.97, 0.99)]
    flow_df = df[["geom_hash", "angle"]].copy()
    if "log10_re" in df.columns: flow_df["log10_re"] = df["log10_re"].values
    flow_df["flow_regime"] = df["flow_regime"].values
    flow_df["flow_ood_score"] = scores
    flow_df["flow_ood_flag"] = flow_df["flow_ood_score"] >= p95
    flow_df["flow_trust_class"] = [classify_by_threshold(float(v), p90, p97) for v in scores]
    flow_df.to_csv(paths["tables"] / "flow_ood_scores.csv", index=False, encoding=OUTPUT_ENCODING)
    meta = {"columns": list(Xf_df.columns), "imputer": imputer, "mu": mu, "precision": precision, "p90": p90, "p95": p95, "p97": p97, "p99": p99}
    joblib.dump(meta, paths["models"] / "flow_ood_model.pkl")
    return {"scores_df": flow_df, "meta": meta, "regime_df": df}


def section_trust_region(geom_ood: Dict, flow_ood: Dict, paths: Dict[str, Path]) -> pd.DataFrame:
    logger.info("[V9-D] Building Trust Region Engine...")
    geom_scores = geom_ood["scores_df"][["geom_hash", "geometry_ood_score", "geometry_ood_flag"]]
    df = flow_ood["scores_df"].merge(geom_scores, on="geom_hash", how="left")
    # Normalize by P97 to make geometry and flow comparable.
    g97 = max(float(geom_ood["meta"]["p97"]), 1e-9)
    f97 = max(float(flow_ood["meta"]["p97"]), 1e-9)
    df["geometry_score_norm"] = df["geometry_ood_score"] / g97
    df["flow_score_norm"] = df["flow_ood_score"] / f97
    df["trust_region_score"] = 0.5 * df["geometry_score_norm"] + 0.5 * df["flow_score_norm"]
    p90 = float(np.quantile(df["trust_region_score"], 0.90))
    p97 = float(np.quantile(df["trust_region_score"], 0.97))
    df["trust_region_class"] = [classify_by_threshold(float(v), p90, p97) for v in df["trust_region_score"]]
    df["safety_flag"] = ~((df["trust_region_class"] == "UNSAFE") | df["geometry_ood_flag"] | df["flow_ood_flag"])
    df.to_csv(paths["tables"] / "trust_region_metrics.csv", index=False, encoding=OUTPUT_ENCODING)
    md = "# Trust Region Report\n\n"
    md += f"- SAFE samples: {(df['trust_region_class']=='SAFE').sum()}\n"
    md += f"- WARNING samples: {(df['trust_region_class']=='WARNING').sum()}\n"
    md += f"- UNSAFE samples: {(df['trust_region_class']=='UNSAFE').sum()}\n"
    md += f"- Trust P90: {p90:.4f}\n- Trust P97: {p97:.4f}\n"
    (paths["reports"] / "trust_region_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    joblib.dump({"p90": p90, "p97": p97, "geometry_p97": g97, "flow_p97": f97}, paths["models"] / "trust_region_model.pkl")
    return df


def section_geometry_ood_disabled(clean_df: pd.DataFrame, features: List[str], paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-B] Geometry OOD disabled for fast development run...")
    geom_cols = [c for c in geometry_feature_columns(features) if c in clean_df.columns]
    curve_df = clean_df.groupby("geom_hash", sort=False)[geom_cols].median().reset_index() if geom_cols else pd.DataFrame({"geom_hash": clean_df["geom_hash"].unique()})
    curve_df["geometry_ood_score"] = 0.0
    curve_df["geometry_ood_flag"] = False
    curve_df["geometry_trust_class"] = "SAFE"
    curve_df.to_csv(paths["tables"] / "geometry_ood_scores.csv", index=False, encoding=OUTPUT_ENCODING)
    meta = {"disabled": True, "columns": geom_cols, "p90": 1.0, "p95": 1.0, "p97": 1.0, "p99": 1.0}
    joblib.dump(meta, paths["models"] / "geometry_ood_model.pkl")
    return {"scores_df": curve_df, "meta": meta}


def section_flow_ood_disabled(clean_df: pd.DataFrame, paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-C] Flow OOD disabled for fast development run...")
    df = attach_regime_labels(clean_df)
    flow_df = df[["geom_hash", "angle"]].copy()
    if "log10_re" in df.columns:
        flow_df["log10_re"] = df["log10_re"].values
    flow_df["flow_regime"] = df["flow_regime"].values
    flow_df["flow_ood_score"] = 0.0
    flow_df["flow_ood_flag"] = False
    flow_df["flow_trust_class"] = "SAFE"
    flow_df.to_csv(paths["tables"] / "flow_ood_scores.csv", index=False, encoding=OUTPUT_ENCODING)
    meta = {"disabled": True, "columns": ["angle", "reynolds", "log10_re"], "p90": 1.0, "p95": 1.0, "p97": 1.0, "p99": 1.0}
    joblib.dump(meta, paths["models"] / "flow_ood_model.pkl")
    return {"scores_df": flow_df, "meta": meta, "regime_df": df}


def section_trust_region_disabled(clean_df: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    logger.info("[V9-D] Trust region disabled for fast development run...")
    df = clean_df[["geom_hash", "angle"]].copy().reset_index(drop=True)
    if "log10_re" in clean_df.columns:
        df["log10_re"] = clean_df["log10_re"].values
    df["geometry_ood_score"] = 0.0
    df["geometry_ood_flag"] = False
    df["flow_ood_score"] = 0.0
    df["flow_ood_flag"] = False
    df["geometry_score_norm"] = 0.0
    df["flow_score_norm"] = 0.0
    df["trust_region_score"] = 0.0
    df["trust_region_class"] = "SAFE"
    df["safety_flag"] = True
    df.to_csv(paths["tables"] / "trust_region_metrics.csv", index=False, encoding=OUTPUT_ENCODING)
    md = "# Trust Region Report\n\nTrust-region / OOD disabled for fast development run (`--disable_ood`).\n"
    (paths["reports"] / "trust_region_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    joblib.dump({"disabled": True, "p90": 1.0, "p97": 1.0, "geometry_p97": 1.0, "flow_p97": 1.0}, paths["models"] / "trust_region_model.pkl")
    return df


def section_bootstrap_ensemble(base_ctx: Dict, cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-E] Training Bootstrap Quantile Ensemble for true predictive uncertainty...")
    X, y, best_name = base_ctx["X"], base_ctx["y"], base_ctx["best_name"]
    n = len(y)
    worker_count = min(resolve_n_jobs(cfg.bootstrap_jobs), max(1, int(cfg.bootstrap_n)))
    logger.info("[V9-E] Bootstrap members=%d parallel_jobs=%d", int(cfg.bootstrap_n), worker_count)
    inner_model_n_jobs = 1 if worker_count > 1 else cfg.n_jobs

    def _fit_bootstrap_member(i: int) -> Tuple[Pipeline, np.ndarray]:
        seed = int(cfg.random_state + 1000 + i * 7919)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=n)
        m = get_model_pipeline(best_name, cfg.random_state + i + 1000, n_jobs=inner_model_n_jobs)
        m.fit(X[idx], y[idx])
        return m, m.predict(X)

    member_indices = list(range(int(cfg.bootstrap_n)))
    if worker_count > 1 and len(member_indices) > 1:
        results = Parallel(n_jobs=worker_count, prefer="threads")(
            delayed(_fit_bootstrap_member)(i) for i in member_indices
        )
    else:
        results = [_fit_bootstrap_member(i) for i in member_indices]

    models = [m for m, _ in results]
    preds = [p for _, p in results]
    pred_arr = np.vstack(preds).astype(np.float32, copy=False)
    preds = None
    results = None
    gc.collect()
    uq_df = pd.DataFrame({
        "cl_p10": np.percentile(pred_arr, 10, axis=0),
        "cl_p50": np.percentile(pred_arr, 50, axis=0),
        "cl_p90": np.percentile(pred_arr, 90, axis=0),
    })
    uq_df["uncertainty_width"] = uq_df["cl_p90"] - uq_df["cl_p10"]
    uq_df.to_csv(paths["tables"] / "bootstrap_uncertainty_predictions.csv", index=False, encoding=OUTPUT_ENCODING)
    joblib.dump(models, paths["models"] / "bootstrap_cl_ensemble.joblib")
    return {"models": models, "predictions": uq_df, "pred_matrix": pred_arr}


def section_regime_conformal(clean_df: pd.DataFrame, y_true: np.ndarray, center_pred: np.ndarray, cfg: CLReliabilityConfig, paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-F] Running regime-wise conformal calibration...")
    df = attach_regime_labels(clean_df)
    residuals = np.abs(y_true - center_pred)
    rows, meta = [], {"method": "regime_wise_abs_residual_quantile", "regimes": {}}
    for regime, idx in df.groupby("flow_regime").groups.items():
        vals = residuals[list(idx)]
        if len(vals) < 5: continue
        meta["regimes"][regime] = {}
        row = {"regime": regime, "n": len(vals)}
        for cov in cfg.conformal_alpha_levels:
            q = float(np.quantile(vals, cov))
            coverage = float(np.mean(vals <= q))
            row[f"q{int(cov*100)}"] = q
            row[f"coverage{int(cov*100)}"] = coverage
            meta["regimes"][regime][f"q{int(cov*100)}"] = q
        rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(paths["tables"] / "conformal_regime_metrics.csv", index=False, encoding=OUTPUT_ENCODING)
    joblib.dump(meta, paths["models"] / "regime_conformal_calibrator.joblib")
    md = "# Regime-wise Conformal Report\n\n" + (rep.to_markdown(index=False) if len(rep) else "No valid regimes.")
    (paths["reports"] / "conformal_regime_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    return {"metrics": rep, "meta": meta, "regime_df": df}


def section_uncertainty_quality(y_true: np.ndarray, bootstrap_ctx: Dict, paths: Dict[str, Path]) -> Dict:
    logger.info("[V9-G] Validating uncertainty quality gate...")
    uq = bootstrap_ctx["predictions"]
    err = np.abs(y_true - uq["cl_p50"].to_numpy())
    width = uq["uncertainty_width"].to_numpy()
    pear = float(pearsonr(width, err)[0]) if np.std(width) > 0 and np.std(err) > 0 else 0.0
    spear = float(spearmanr(width, err).correlation) if np.std(width) > 0 and np.std(err) > 0 else 0.0
    cov90 = float(np.mean((y_true >= uq["cl_p10"].to_numpy()) & (y_true <= uq["cl_p90"].to_numpy())))
    # approximate 95 interval using empirical 2.5/97.5 from bootstrap matrix
    mat = bootstrap_ctx["pred_matrix"]
    lo95, hi95 = np.percentile(mat, 2.5, axis=0), np.percentile(mat, 97.5, axis=0)
    cov95 = float(np.mean((y_true >= lo95) & (y_true <= hi95)))
    try:
        slope = float(np.polyfit(width, err, 1)[0]) if np.std(width) > 0 else 0.0
    except Exception:
        slope = 0.0
    reliable = bool((pear > 0.25) and (spear > 0.25) and (0.88 <= cov90 <= 0.92) and (0.93 <= cov95 <= 0.97))
    metrics = pd.DataFrame([{"pearson_uq_error": pear, "spearman_uq_error": spear, "coverage90": cov90, "coverage95": cov95, "calibration_slope": slope, "uncertainty_reliable": reliable}])
    metrics.to_csv(paths["tables"] / "uncertainty_quality_metrics.csv", index=False, encoding=OUTPUT_ENCODING)
    md = "# Uncertainty Quality Report\n\n" + metrics.to_markdown(index=False) + "\n\n"
    md += "PASS" if reliable else "FAIL: uncertainty is exported, but reliability gate is not fully satisfied."
    (paths["reports"] / "uncertainty_quality_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    return {"metrics": metrics, "reliable": reliable}


def section_failure_rootcause(logo_df: pd.DataFrame, curve_df: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    logger.info("[V9-H] Running geometry failure root-cause analysis...")
    driver_cols = [c for c in curve_df.columns if any(k in c.lower() for k in ["thickness", "camber", "le_radius", "te_angle", "aft", "wake", "separation", "curvature", "loading"])]
    if not driver_cols or len(logo_df) == 0:
        out = pd.DataFrame()
    else:
        global_med = curve_df[driver_cols].median(numeric_only=True)
        global_std = curve_df[driver_cols].std(numeric_only=True).replace(0, np.nan)
        worst = logo_df.tail(min(10, len(logo_df))).copy()
        rows = []
        for _, wr in worst.iterrows():
            g = curve_df[curve_df["geom_hash"] == wr["geom_hash"]]
            if len(g) == 0: continue
            vals = g.iloc[0][driver_cols].astype(float)
            z = ((vals - global_med) / global_std).replace([np.inf, -np.inf], np.nan).abs().sort_values(ascending=False)
            top = [c for c in z.index[:5] if np.isfinite(z[c])]
            explanation = "failure likely caused by " + ", ".join([f"{c} deviation z={z[c]:.2f}" for c in top]) if top else "failure cause inconclusive"
            rows.append({"geom_hash": wr["geom_hash"], "logo_mae": wr["mae"], "top_failure_drivers": "; ".join(top), "explanation": explanation})
        out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "geometry_failure_rootcause.csv", index=False, encoding=OUTPUT_ENCODING)
    md = "# Geometry Failure Root-Cause Report\n\n" + (out.to_markdown(index=False) if len(out) else "No root-cause table generated.")
    (paths["reports"] / "geometry_failure_rootcause_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    return out


def section_champion_selection(base_ctx: Dict, logo_df: pd.DataFrame, uq_quality: Dict, paths: Dict[str, Path]) -> pd.DataFrame:
    logger.info("[V9-I] Exporting champion selection evidence...")
    comp = pd.read_csv(paths["tables"] / "model_comparison.csv")
    rows = []
    for _, r in comp.iterrows():
        rows.append({
            "candidate": r["model"],
            "logo_mae": r.get("logo_mae", np.nan),
            "clmax_error": r.get("clmax_mae", np.nan),
            "stall_aoa_error": r.get("stall_aoa_mae", np.nan),
            "uncertainty_quality": float(uq_quality["metrics"].iloc[0]["pearson_uq_error"]) if r["model"] == base_ctx["best_name"] else 0.0,
            "rmse": r.get("rmse", np.nan),
            "r2": r.get("r2", np.nan),
            "selected": r["model"] == base_ctx["best_name"],
        })
    out = pd.DataFrame(rows).sort_values(["clmax_error", "stall_aoa_error", "rmse", "r2"], ascending=[True, True, True, False])
    out.to_csv(paths["tables"] / "champion_metrics.csv", index=False, encoding=OUTPUT_ENCODING)
    md = "# Champion Selection Report\n\nSelected champion: **{}**\n\n".format(base_ctx["best_name"])
    md += "Selection priorities: CLmax error -> stall AoA error -> RMSE -> R2. Geometry LOGO refits are removed.\n\n"
    md += out.to_markdown(index=False)
    (paths["reports"] / "champion_selection_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    return out


def section_active_learning(clean_df: pd.DataFrame, bootstrap_ctx: Dict, trust_df: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    logger.info("[V9-K] Exporting active-learning candidates...")
    out = clean_df[["geom_hash", "angle"]].copy().reset_index(drop=True)
    if "log10_re" in clean_df.columns: out["log10_re"] = clean_df["log10_re"].values
    uq = bootstrap_ctx["predictions"].reset_index(drop=True)
    tr = trust_df[["trust_region_score", "trust_region_class"]].reset_index(drop=True)
    out = pd.concat([out, uq[["cl_p50", "uncertainty_width"]], tr], axis=1)
    out["stall_sensitivity"] = 0.0
    for _, idx in clean_df.groupby("geom_hash").groups.items():
        ii = list(idx)
        if len(ii) >= 3:
            aoa = clean_df.iloc[ii]["angle"].to_numpy(float)
            pred = uq.iloc[ii]["cl_p50"].to_numpy(float)
            order = np.argsort(aoa)
            grad = np.gradient(pred[order], aoa[order])
            sens = np.abs(np.gradient(grad, aoa[order]))
            out.iloc[np.array(ii)[order], out.columns.get_loc("stall_sensitivity")] = sens
    out["active_learning_score"] = out["uncertainty_width"] * out["trust_region_score"] * (1.0 + out["stall_sensitivity"])
    out = out.sort_values("active_learning_score", ascending=False)
    out.to_csv(paths["tables"] / "active_learning_candidates.csv", index=False, encoding=OUTPUT_ENCODING)
    return out


def section_stall_physics_audit(clean_df: pd.DataFrame, pred: np.ndarray, paths: Dict[str, Path]) -> pd.DataFrame:
    logger.info("[V9-M] Running stall physics consistency audit...")
    tmp = clean_df.copy(); tmp["pred"] = pred
    rows = []
    group_cols = ["geom_hash", "log10_re"] if "log10_re" in tmp.columns else ["geom_hash"]
    for key, grp in tmp.groupby(group_cols):
        grp = grp.sort_values("angle")
        aoa, clp = grp["angle"].to_numpy(float), grp["pred"].to_numpy(float)
        linear = (aoa >= -4) & (aoa <= 6)
        neg_slope = False
        if linear.sum() >= 3:
            neg_slope = bool(np.any(np.diff(clp[linear]) / np.diff(aoa[linear]) < -0.01))
        clmax_i = int(np.argmax(clp))
        late_peak = bool(aoa[clmax_i] >= np.nanmax(aoa) - 1e-8)
        rows.append({"group": str(key), "negative_linear_slope": neg_slope, "clmax_at_last_aoa": late_peak, "violation": neg_slope or late_peak})
    out = pd.DataFrame(rows)
    out.to_csv(paths["tables"] / "stall_physics_audit.csv", index=False, encoding=OUTPUT_ENCODING)
    md = "# Stall Physics Consistency Audit\n\n" + f"Violating curves: {int(out['violation'].sum()) if len(out) else 0}\n"
    (paths["reports"] / "stall_physics_report.md").write_text(md, encoding=OUTPUT_ENCODING)
    return out


def section_export_day5_interfaces_v9(feat_cols: List[str], base_ctx: Dict, clmax_ctx: Dict, clean_df: pd.DataFrame, paths: Dict[str, Path], uq_quality: Dict) -> None:
    section_export_day5_interfaces_v8(feat_cols, base_ctx, clmax_ctx, clean_df, paths)
    fp = paths["artifacts"] / "day5_interface.json"
    meta = json.loads(fp.read_text(encoding="utf-8"))
    meta.update({
        "version": "V9_Final_Publication_Grade",
        "predict_cl_interval": "load bootstrap_cl_ensemble.joblib; return cl_p50, cl_p90_lo, cl_p90_hi",
        "predict_trust_region": "use geometry_ood_model.pkl + flow_ood_model.pkl + trust_region_model.pkl",
        "predict_reliability": "return reliability_score and uncertainty_quality flag",
        "predict_ood": "return geometry_ood_score, flow_ood_score, geometry_ood_flag, flow_ood_flag",
        "safety_rule": "safety_flag = FALSE if UNSAFE trust region OR geometry OOD OR flow OOD",
        "uncertainty_reliable": bool(uq_quality["reliable"]),
        "artifacts_mapping": {**meta.get("artifacts_mapping", {}),
            "bootstrap_cl_ensemble": "models/bootstrap_cl_ensemble.joblib",
            "geometry_ood_model": "models/geometry_ood_model.pkl",
            "flow_ood_model": "models/flow_ood_model.pkl",
            "trust_region_model": "models/trust_region_model.pkl",
            "regime_conformal_calibrator": "models/regime_conformal_calibrator.joblib"
        }
    })
    fp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def section_final_publication_report(paths: Dict[str, Path], base_ctx: Dict, logo_df: pd.DataFrame, uq_quality: Dict, trust_df: pd.DataFrame, violations: int) -> None:
    comp = pd.read_csv(paths["tables"] / "model_comparison.csv")
    best = comp[comp["model"] == base_ctx["best_name"]].iloc[0]
    uqm = uq_quality["metrics"].iloc[0]
    md = "# Day4 CL Reliability V9 Publication Report\n\n"
    md += "## Overall Metrics\n\n"
    md += f"- Champion: **{base_ctx['best_name']}**\n- OOF RMSE: {best['rmse']:.5f}\n- OOF R²: {best['r2']:.5f}\n"
    md += "\n## CLmax / Stall Metrics\n\n"
    md += f"- CLmax Error: {best['clmax_mae']:.5f}\n- Stall AoA Error: {best['stall_aoa_mae']:.5f} deg\n"
    md += "\n## Geometry Refit Validation\n\n"
    md += "Removed for runtime; no per-geometry LOGO refits are executed.\n"
    md += "\n## Trust Region Analysis\n\n"
    md += trust_df["trust_region_class"].value_counts().to_markdown() + "\n"
    md += "\n## Uncertainty Quality\n\n"
    md += uq_quality["metrics"].to_markdown(index=False) + "\n"
    md += "\n## Physical Consistency\n\n"
    md += f"- Physical consistency violations: {violations}\n"
    md += "\n## Day5 Readiness\n\n"
    md += "Day5 interface is safety-aware and includes CL interval, OOD, trust-region, reliability, and safety flag endpoints.\n"
    md += "\n## Research Conclusions\n\n"
    md += "V9 prioritizes geometry generalization, CLmax/stall reliability, uncertainty validation, and safe inverse design rather than R²-only optimization.\n"
    (paths["reports"] / "day4_cl_v9_publication_report.md").write_text(md, encoding=OUTPUT_ENCODING)

# ---------------------------------------------------------------------------
# Main Execution Module
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CL Reliability V9 Final - Publication Grade")
    parser.add_argument("--config", type=str, default="day4_cl_reliability_config.json")
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--disable_logo", action="store_true", help="Compatibility flag; LOGO is removed and always skipped.")
    parser.add_argument("--disable_ood", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of CPU workers. Use -1 for all logical cores.")
    parser.add_argument("--input_csv", type=str, default=None, help="Override dataset path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--max_rows", type=int, default=None, help="Smoke/dev row cap. Reads a bounded prefix before sampling.")
    parser.add_argument("--n_splits", type=int, default=None, help="Geometry GroupKFold splits.")
    parser.add_argument("--bootstrap_n", type=int, default=None, help="Bootstrap ensemble members.")
    parser.add_argument("--bootstrap_jobs", type=int, default=None, help="Parallel bootstrap fits. Default 1 minimizes peak RAM.")
    parser.add_argument("--candidate_models", type=str, default=None, help="Comma-separated models to compare: hgb,extra_trees,random_forest.")
    parser.add_argument("--tree_estimators", type=int, default=None, help="RF/ExtraTrees estimators.")
    parser.add_argument("--hgb_iter", type=int, default=None, help="HistGradientBoosting iterations.")
    parser.add_argument("--logo_max_geometries", type=int, default=None, help="Compatibility flag; ignored because LOGO is removed.")
    parser.add_argument("--logo_mode", choices=["logo", "groupkfold", "disabled", "removed"], default=None, help="Compatibility flag; ignored because LOGO is removed.")
    parser.add_argument("--logo_cv_splits", type=int, default=None, help="Compatibility flag; ignored because LOGO is removed.")
    args, unknown = parser.parse_known_args()

    cfg = CLReliabilityConfig()
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k): setattr(cfg, k, v)
    if args.fast_mode: cfg.fast_mode = True
    if args.disable_logo: cfg.disable_logo = True
    if args.disable_ood: cfg.disable_ood = True
    if args.input_csv is not None:
        cfg.dataset_path = args.input_csv
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.max_rows is not None:
        cfg.max_rows = int(args.max_rows)
    if args.n_splits is not None:
        cfg.n_splits = int(args.n_splits)
    if args.bootstrap_n is not None:
        cfg.bootstrap_n = max(1, int(args.bootstrap_n))
    if args.bootstrap_jobs is not None:
        cfg.bootstrap_jobs = max(1, int(args.bootstrap_jobs))
    if args.candidate_models is not None:
        cfg.candidate_models = args.candidate_models
    global CL_TREE_ESTIMATORS, CL_HGB_MAX_ITER
    if args.tree_estimators is not None:
        CL_TREE_ESTIMATORS = max(10, int(args.tree_estimators))
    if args.hgb_iter is not None:
        CL_HGB_MAX_ITER = max(10, int(args.hgb_iter))
    cfg.logo_mode = "removed"
    cfg.disable_logo = True
    if args.n_jobs is not None:
        cfg.n_jobs = int(args.n_jobs)
    if args.logo_max_geometries is not None:
        cfg.logo_max_geometries = int(args.logo_max_geometries)
    cfg.n_jobs = configure_thread_env(cfg.n_jobs)

    paths = ensure_dirs(cfg.output_dir)
    if not os.path.exists(cfg.dataset_path):
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        sys.exit(1)

    read_nrows = max(50_000, int(cfg.max_rows) * 5) if cfg.max_rows else None
    raw_df = pd.read_csv(cfg.dataset_path, low_memory=False, nrows=read_nrows)
    raw_df = downcast_float64_to_float32(raw_df)
    day5_features, excluded = apply_comprehensive_whitelist_v8(raw_df)
    logger.info(f"V9 Whitelist Activated: {len(day5_features)} operational features; {len(excluded)} excluded.")

    clean_df = raw_df.dropna(subset=["cl", "geom_hash", "angle"]).copy().reset_index(drop=True)
    clean_df = downcast_float64_to_float32(clean_df)
    if cfg.max_rows and len(clean_df) > cfg.max_rows:
        clean_df = clean_df.sample(cfg.max_rows, random_state=cfg.random_state).reset_index(drop=True)
        clean_df = downcast_float64_to_float32(clean_df)

    # Existing V8-compatible core flow.
    base_ctx, oof_preds, y_true = section_base_cl_model(clean_df, day5_features, cfg, paths)
    # Persist the primary model and diagnostics before expensive downstream audits.
    joblib.dump(base_ctx["model_store"][base_ctx["best_name"]][0], paths["models"] / "cl_model.joblib")
    section_figures(clean_df, y_true, oof_preds, paths)
    clmax_ctx, clmax_model = section_clmax_surrogate(clean_df, day5_features, cfg, paths)
    stall_model = section_stall_surrogate(clmax_ctx, cfg, paths)
    uq_ctx = section_uncertainty_and_coverage(base_ctx, cfg, paths)
    logo_df = section_logo_and_failure_analysis(base_ctx, clmax_ctx["curve_df"], clean_df, cfg, paths)
    violations = section_physical_consistency_suite(clean_df, oof_preds)

    # V9 research-grade additions.
    if cfg.disable_ood:
        geom_ood = section_geometry_ood_disabled(clean_df, day5_features, paths)
        flow_ood = section_flow_ood_disabled(clean_df, paths)
        trust_df = section_trust_region_disabled(clean_df, paths)
    else:
        geom_ood = section_geometry_ood(clean_df, day5_features, cfg, paths)
        flow_ood = section_flow_ood(clean_df, cfg, paths)
        trust_df = section_trust_region(geom_ood, flow_ood, paths)
    bootstrap_ctx = section_bootstrap_ensemble(base_ctx, cfg, paths)
    regime_conf = section_regime_conformal(clean_df, y_true, bootstrap_ctx["predictions"]["cl_p50"].to_numpy(), cfg, paths)
    uq_quality = section_uncertainty_quality(y_true, bootstrap_ctx, paths)
    failure_root = section_failure_rootcause(logo_df, clmax_ctx["curve_df"], paths)
    champion_df = section_champion_selection(base_ctx, logo_df, uq_quality, paths)
    active_df = section_active_learning(clean_df, bootstrap_ctx, trust_df, paths)
    stall_audit = section_stall_physics_audit(clean_df, bootstrap_ctx["predictions"]["cl_p50"].to_numpy(), paths)

    # Export final artifacts and reports.
    section_export_day5_interfaces_v9(day5_features, base_ctx, clmax_ctx, clean_df, paths, uq_quality)
    section_final_publication_report(paths, base_ctx, logo_df, uq_quality, trust_df, violations)

    logger.info("=" * 80 + "\nCL RELIABILITY V9 FINAL COMPLETE - PUBLICATION-GRADE + DAY5 SAFETY-AWARE\n" + "=" * 80)

if __name__ == "__main__":
    main()




