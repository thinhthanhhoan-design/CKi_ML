"""
day4c_cd_v3_1_fixed.py
================================================================================
Publication-grade V2 research upgrade layer for Day4C Cd surrogate.

This script keeps the original Day4C architecture concept intact:
- PCA latent geometry
- geometry descriptors
- regime experts
- base/residual-compatible Cd modeling
- stall-aware regime structure
- conformal safety layer

It adds the missing research-grade pieces:
A. Practical + exact Leave-One-Geometry-Out validation
B. Geometry OOD and Flow OOD detectors
C. Bootstrap Quantile Regime Ensemble uncertainty
D. Regime-wise split conformal calibration
E. Per-regime micro calibration
F. Strict feature validation, no silent zero-fill
G. Geometry failure analysis
H. Day5 interface artifacts
I. Research acceptance gate: baseline vs upgraded when baseline metrics are supplied

Smoke test:
    python day4c_cd_v3_1_fixed.py \
        --input_csv deeplearwing_day2_tabular.csv \
        --output_dir outputs/day4c_research_upgrade_v2 \
        --max_rows 50000 --bootstrap_models 3

Full research run:
    python day4c_cd_v3_1_fixed.py \
        --input_csv deeplearwing_day2_tabular.csv \
        --output_dir outputs/day4c_research_upgrade_v2 \
        --bootstrap_models 5
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False

try:
    import psutil
except Exception:
    psutil = None

OUTPUT_TEXT_ENCODING = "utf-8-sig"
CD_FLOOR = 1e-7
REGIME_IDS = [1, 2, 3, 4, 5, 6]
REGIME_NAMES = {
    1: "R1_linear",
    2: "R2_transitional",
    3: "R3_pre_stall",
    4: "R4_stall_onset",
    5: "R5_near_stall",
    6: "R6_post_stall",
}
FORBIDDEN_FEATURE_TOKENS = [
    "cl", "cd", "cm", "target", "true", "pred", "prediction", "error", "residual",
    "uncert", "coverage", "conformal", "confidence", "quality", "flag", "label",
    "name", "geom_hash", "geometry_id", "airfoil", "stall_onset", "sample_weight",
]
GEOMETRY_FAILURE_FEATURES = [
    "t_max", "x_tmax", "camber_max", "x_cmax", "te_gap", "le_radius_proxy",
    "trailing_edge_angle", "te_wedge_angle", "aft_thickness", "aft_camber",
    "curvature_energy_aft", "upper_curvature_energy_aft_06",
    "upper_aft_curvature_concentration", "slope_variance_aft", "thickness_gradient_aft",
    "thickness_gradient_abs_aft", "wake_proxy", "hysteresis_proxy",
    "aft_to_mid_thickness_ratio", "le_curvature_energy", "separation_proxy",
    "pressure_recovery_proxy", "aft_pressure_recovery_proxy", "aft_curvature_gradient",
    "aft_loading_metric", "camber_area", "camber_centroid_x", "thickness_centroid_x",
    "thickness_moment_arm", "pressure_center_proxy", "camber_pressure_proxy",
    "aft_loading_proxy", "moment_distribution_proxy",
]


@dataclass
class ResearchConfig:
    input_csv: str = "deeplearwing_day2_tabular.csv"
    output_dir: str = "outputs/day4c_research_upgrade_v2"
    random_state: int = 42
    test_size: float = 0.20
    calibration_size: float = 0.20
    max_rows: Optional[int] = None
    max_features: int = 260
    pca_variance: float = 0.995
    pca_max_components: int = 32
    geometry_ood_percentile: float = 95.0
    flow_ood_percentile: float = 95.0
    quantiles: Tuple[float, float, float] = (0.10, 0.50, 0.90)
    bootstrap_models: int = 5
    bootstrap_fraction: float = 0.90
    n_estimators: int = 650
    learning_rate: float = 0.035
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    logo_max_geometries: int = 0
    logo_min_rows_per_geometry: int = 5
    logo_sample_train_rows: int = 0
    logo_refit_pca_each_fold: bool = True
    logo_mode: str = "removed"
    logo_cv_splits: int = 5
    geometry_cv_enabled: bool = True
    geometry_cv_splits: int = 10
    geometry_cv_p50_only: bool = True
    geometry_cv_bootstrap_models: int = 1
    geometry_cv_global_only: bool = True
    geometry_cv_n_estimators: int = 180
    regime_models_enabled: bool = True
    use_isotonic_calibration: bool = False
    baseline_metrics_csv: Optional[str] = None
    target_rmse_tolerance: float = 0.0
    coverage90_min: float = 0.88
    coverage90_max: float = 0.92
    disable_logo: bool = True
    disable_ood: bool = False
    n_jobs: int = -1
    memory_guard_enabled: bool = True
    memory_safety_gb: float = 2.0
    training_mem_multiplier: float = 2.5
    outer_parallel_mem_multiplier: float = 4.0


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
    return effective


def get_available_memory_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def dataframe_nbytes(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(index=True, deep=False).sum())
    except Exception:
        return 0


def downcast_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce RAM by converting only float64 columns to float32.

    This helper intentionally does not alter object, integer, boolean, category,
    or datetime columns, so the pipeline logic and grouping behavior remain unchanged.
    """
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols):
        df.loc[:, float64_cols] = df.loc[:, float64_cols].astype(np.float32)
    return df


def limit_parallel_jobs_for_memory(
    requested_jobs: int,
    data_nbytes: int,
    copies_per_worker: float,
    cfg: ResearchConfig,
    stage: str,
) -> int:
    effective_jobs = resolve_n_jobs(requested_jobs)
    if not cfg.memory_guard_enabled:
        return effective_jobs
    available = get_available_memory_bytes()
    if available is None or data_nbytes <= 0:
        return effective_jobs
    reserve_bytes = int(float(cfg.memory_safety_gb) * (1024 ** 3))
    usable = max(0, available - reserve_bytes)
    if usable <= 0:
        print(f"[RAM_GUARD] stage={stage} forced_n_jobs=1 because free RAM is below reserve.", flush=True)
        return 1
    per_worker_bytes = max(1, int(float(data_nbytes) * float(copies_per_worker)))
    max_jobs = max(1, min(effective_jobs, usable // per_worker_bytes))
    if max_jobs < effective_jobs:
        print(
            f"[RAM_GUARD] stage={stage} requested_n_jobs={effective_jobs} capped_n_jobs={max_jobs} "
            f"free_ram_gb={available / (1024 ** 3):.2f} est_per_worker_gb={per_worker_bytes / (1024 ** 3):.2f}",
            flush=True,
        )
    return max_jobs


def ensure_dirs(base: str) -> Dict[str, Path]:
    root = Path(base)
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
        "reports": root / "reports",
        "logo": root / "logo",
        "day5": root / "day5_interface",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, scope: str) -> Dict[str, Any]:
    return {
        "scope": scope,
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def regime_id_from_angle(angle: Any) -> np.ndarray:
    a = np.abs(np.asarray(angle, dtype=float))
    out = np.full(a.shape, 6, dtype=int)
    out[a < 5.0] = 1
    out[(a >= 5.0) & (a < 8.0)] = 2
    out[(a >= 8.0) & (a < 10.0)] = 3
    out[(a >= 10.0) & (a < 12.0)] = 4
    out[(a >= 12.0) & (a < 16.0)] = 5
    out[a >= 16.0] = 6
    return out


def discover_geometry_group_col(df: pd.DataFrame) -> str:
    for c in ["geometry_id", "geom_hash", "airfoil_id", "name"]:
        if c in df.columns:
            return c
    raise ValueError("Missing geometry group column: expected geometry_id, geom_hash, airfoil_id, or name")


def sorted_g_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("g_")]
    def key(c: str) -> Tuple[int, str]:
        try:
            return int(c.split("_")[1]), c
        except Exception:
            return 10**9, c
    return sorted(cols, key=key)


def is_safe_feature(col: str) -> bool:
    lc = col.lower()
    if col in {"angle", "reynolds", "log10_re"}:
        return True
    if col.startswith("z_"):
        return True
    if col.startswith("g_"):
        return False
    for token in FORBIDDEN_FEATURE_TOKENS:
        if token in lc:
            return False
    return True


def load_dataset(cfg: ResearchConfig) -> pd.DataFrame:
    path = Path(cfg.input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    # --max_rows is a smoke/debug mode: avoid materializing the full wide CSV first.
    read_nrows = max(50_000, int(cfg.max_rows) * 5) if cfg.max_rows else None
    df = pd.read_csv(path, low_memory=False, nrows=read_nrows)
    df = downcast_float64_to_float32(df)
    group_col = discover_geometry_group_col(df)
    required = ["angle", "cd", group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=["angle", "cd", group_col]).copy()
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    df["cd"] = pd.to_numeric(df["cd"], errors="coerce")
    df = df.dropna(subset=["angle", "cd"])
    df = downcast_float64_to_float32(df)
    if "reynolds" in df.columns:
        df["reynolds"] = pd.to_numeric(df["reynolds"], errors="coerce").fillna(df["reynolds"].median())
        if "log10_re" not in df.columns:
            df["log10_re"] = np.log10(np.maximum(df["reynolds"].to_numpy(dtype=float), 1.0))
    elif "log10_re" not in df.columns:
        df["log10_re"] = 6.0
    angle = df["angle"].to_numpy(dtype=np.float32, copy=False)
    abs_angle = np.abs(angle).astype(np.float32, copy=False)
    log10_re = df["log10_re"].to_numpy(dtype=np.float32, copy=False)
    df["abs_angle"] = abs_angle
    df["angle_sq"] = np.square(angle).astype(np.float32, copy=False)
    df["signed_angle_sq"] = (np.sign(angle) * np.square(angle)).astype(np.float32, copy=False)
    df["angle_cu"] = np.power(angle, 3, dtype=np.float32)
    df["angle_x_log10_re"] = (angle * log10_re).astype(np.float32, copy=False)
    df["abs_angle_x_log10_re"] = (abs_angle * log10_re).astype(np.float32, copy=False)
    df["r1_edge_proximity"] = np.clip(abs_angle / 5.0, 0.0, 1.0).astype(np.float32, copy=False)
    df["regime_id"] = regime_id_from_angle(df["angle"].to_numpy(dtype=np.float32))
    df = downcast_float64_to_float32(df)
    if cfg.max_rows and len(df) > cfg.max_rows:
        df = df.sample(n=int(cfg.max_rows), random_state=cfg.random_state).reset_index(drop=True)
        df = downcast_float64_to_float32(df)
    return downcast_float64_to_float32(df.reset_index(drop=True))


def train_cal_test_split(df: pd.DataFrame, group_col: str, cfg: ResearchConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two geometry groups to create train/test split.")
    n_splits = max(2, min(int(cfg.geometry_cv_splits), len(unique_groups)))
    folds = list(GroupKFold(n_splits=n_splits).split(df, groups=groups))
    fold_id = int(abs(cfg.random_state)) % len(folds)
    train_cal_idx, test_idx = folds[fold_id]
    train_cal = df.iloc[train_cal_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    rel_cal = cfg.calibration_size / max(1e-9, 1.0 - cfg.test_size)
    train, cal = train_cal_split_by_geometry(train_cal, group_col, cfg, cfg.random_state + 17, rel_cal)
    return train, cal, test


def train_cal_split_by_geometry(df: pd.DataFrame, group_col: str, cfg: ResearchConfig, seed: int, cal_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two geometry groups to create train/cal split.")
    cal_fraction = float(np.clip(cal_fraction, 0.05, 0.50))
    n_splits = max(2, min(int(round(1.0 / cal_fraction)), len(unique_groups)))
    folds = list(GroupKFold(n_splits=n_splits).split(df, groups=groups))
    fold_id = int(abs(seed)) % len(folds)
    train_idx, cal_idx = folds[fold_id]
    train = df.iloc[train_idx].reset_index(drop=True)
    cal = df.iloc[cal_idx].reset_index(drop=True)
    return train, cal


def build_pca_latents(df_train: pd.DataFrame, df_apply: pd.DataFrame, cfg: ResearchConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    g_cols = sorted_g_cols(df_train)
    if len(g_cols) < 10:
        raise ValueError("Need at least 10 g_* geometry columns for PCA latent geometry.")
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = df_train[g_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xap = df_apply[g_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xtr = imp.fit_transform(Xtr).astype(np.float32, copy=False)
    Xap = imp.transform(Xap).astype(np.float32, copy=False)
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32, copy=False)
    Xap_s = scaler.transform(Xap).astype(np.float32, copy=False)
    n_comp = min(cfg.pca_max_components, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    n_comp = max(2, int(n_comp))
    pca = PCA(n_components=n_comp, random_state=cfg.random_state)
    Ztr_full = pca.fit_transform(Xtr_s).astype(np.float32, copy=False)
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(cum, cfg.pca_variance) + 1)
    keep = min(max(2, keep), Ztr_full.shape[1])
    Zap = pca.transform(Xap_s)[:, :keep].astype(np.float32, copy=False)
    Ztr = Ztr_full[:, :keep].astype(np.float32, copy=False)
    z_cols = [f"z_{i:03d}" for i in range(keep)]
    tr = df_train.copy()
    ap = df_apply.copy()
    for i, c in enumerate(z_cols):
        tr[c] = Ztr[:, i].astype(np.float32, copy=False)
        ap[c] = Zap[:, i].astype(np.float32, copy=False)
    tr = downcast_float64_to_float32(tr)
    ap = downcast_float64_to_float32(ap)
    del Xtr, Xap, Xtr_s, Xap_s, Ztr_full, Zap, Ztr
    gc.collect()
    meta = {"g_cols": g_cols, "z_cols": z_cols, "imputer": imp, "scaler": scaler, "pca": pca, "keep": keep}
    return tr, ap, meta


class StrictFeatureValidator:
    def __init__(self, feature_cols: List[str], report_path: Path):
        self.feature_cols = list(feature_cols)
        self.report_path = Path(report_path)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(
                "Missing required feature columns\n" + "\n".join(missing),
                encoding=OUTPUT_TEXT_ENCODING,
            )
            raise ValueError(f"Missing required feature columns: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        return df[self.feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)


def select_feature_columns(df: pd.DataFrame, z_cols: List[str], cfg: ResearchConfig) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    forced = [c for c in [
        "angle", "abs_angle", "angle_sq", "signed_angle_sq", "angle_cu",
        "log10_re", "angle_x_log10_re", "abs_angle_x_log10_re", "r1_edge_proximity",
    ] if c in df.columns] + list(z_cols)
    candidates = []
    for c in numeric_cols:
        if c in forced or c == "regime_id":
            continue
        if is_safe_feature(c):
            candidates.append(c)
    preferred = ["t_max", "camber", "te_", "trailing", "aft", "curvature", "slope", "thickness", "pressure", "wake", "hysteresis", "separation", "recovery", "radius"]
    candidates = sorted(candidates, key=lambda c: (0 if any(t in c.lower() for t in preferred) else 1, c))
    out = forced + candidates[:max(0, cfg.max_features - len(forced))]
    seen, clean = set(), []
    for c in out:
        if c not in seen:
            clean.append(c); seen.add(c)
    return clean


class MahalanobisDetector:
    def __init__(self, percentile: float = 95.0, ridge: float = 1e-6):
        self.percentile = float(percentile)
        self.ridge = float(ridge)
        self.feature_cols: List[str] = []
        self.mu: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold: float = float("nan")
        self.train_distances: Optional[np.ndarray] = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    def fit_matrix(self, X: np.ndarray) -> "MahalanobisDetector":
        X = np.asarray(X, dtype=float)
        self.mu = np.mean(X, axis=0)
        cov = np.cov(X - self.mu, rowvar=False)
        cov = np.atleast_2d(cov) + self.ridge * np.eye(X.shape[1])
        self.inv_cov = np.linalg.pinv(cov)
        d = self.score_matrix(X)
        self.train_distances = d
        self.threshold = float(np.nanpercentile(d, self.percentile))
        return self

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.inv_cov is None:
            raise RuntimeError("Detector has not been fitted")
        D = np.asarray(X, dtype=float) - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, self.inv_cov, D), 0.0))

    def score(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = self._matrix(df)
        d = self.score_matrix(X)
        return d, d > self.threshold


class GeometryOODDetector(MahalanobisDetector):
    def fit(self, df: pd.DataFrame, z_cols: List[str], group_col: str) -> "GeometryOODDetector":
        self.feature_cols = list(z_cols)
        Z = df.groupby(group_col)[self.feature_cols].mean().to_numpy(dtype=np.float32)
        return self.fit_matrix(Z)


class FlowOODDetector(MahalanobisDetector):
    def fit(self, df: pd.DataFrame) -> "FlowOODDetector":
        self.feature_cols = ["angle", "abs_angle", "log10_re", "regime_id"]
        tmp = df.copy()
        tmp["abs_angle"] = np.abs(tmp["angle"].to_numpy(dtype=np.float32))
        if "log10_re" not in tmp.columns:
            tmp["log10_re"] = 6.0
        X = tmp[self.feature_cols].to_numpy(dtype=np.float32)
        return self.fit_matrix(X)

    def score(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        tmp = df.copy()
        tmp["abs_angle"] = np.abs(tmp["angle"].to_numpy(dtype=np.float32))
        if "log10_re" not in tmp.columns:
            tmp["log10_re"] = 6.0
        d = self.score_matrix(tmp[self.feature_cols].to_numpy(dtype=np.float32))
        return d, d > self.threshold


class DisabledOODDetector:
    """No-op detector for fast runs that skip OOD diagnostics."""
    def __init__(self):
        self.feature_cols: List[str] = []
        self.threshold: float = 1.0
        self.train_distances: Optional[np.ndarray] = None

    def score(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(df)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=bool)


def make_quantile_model(alpha: float, cfg: ResearchConfig, seed: int):
    if HAS_LGB:
        return lgb.LGBMRegressor(
            objective="quantile", alpha=float(alpha), n_estimators=int(cfg.n_estimators),
            learning_rate=float(cfg.learning_rate), num_leaves=int(cfg.max_leaf_nodes),
            min_child_samples=int(cfg.min_samples_leaf), subsample=0.85, colsample_bytree=0.85,
            reg_lambda=8.0, random_state=int(seed), n_jobs=int(cfg.n_jobs), verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=float(alpha), learning_rate=float(cfg.learning_rate),
        max_iter=int(cfg.n_estimators), max_leaf_nodes=int(cfg.max_leaf_nodes),
        min_samples_leaf=int(cfg.min_samples_leaf), l2_regularization=0.05,
        random_state=int(seed),
    )


def make_median_model(cfg: ResearchConfig, seed: int):
    if HAS_LGB:
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=int(cfg.n_estimators), learning_rate=float(cfg.learning_rate),
            num_leaves=int(cfg.max_leaf_nodes), min_child_samples=int(cfg.min_samples_leaf),
            subsample=0.85, colsample_bytree=0.85, reg_lambda=8.0,
            random_state=int(seed), n_jobs=int(cfg.n_jobs), verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=float(cfg.learning_rate), max_iter=int(cfg.n_estimators),
        max_leaf_nodes=int(cfg.max_leaf_nodes), min_samples_leaf=int(cfg.min_samples_leaf),
        l2_regularization=0.05, random_state=int(seed),
    )


class BootstrapRegimeQuantileEnsemble:
    def __init__(self, feature_cols: List[str], cfg: ResearchConfig):
        self.feature_cols = list(feature_cols)
        self.cfg = cfg
        self.models: Dict[int, List[Dict[str, Any]]] = {}
        self.global_models: List[Dict[str, Any]] = []
        self.feature_validator: Optional[StrictFeatureValidator] = None

    def fit(self, df: pd.DataFrame, report_path: Path) -> "BootstrapRegimeQuantileEnsemble":
        self.feature_validator = StrictFeatureValidator(self.feature_cols, report_path)
        X_all = self.feature_validator.validate(df)
        y_log = np.log(np.maximum(df["cd"].to_numpy(dtype=np.float32), CD_FLOOR)).astype(np.float32, copy=False)
        fit_n_jobs = limit_parallel_jobs_for_memory(
            self.cfg.n_jobs,
            dataframe_nbytes(X_all) + int(y_log.nbytes),
            self.cfg.training_mem_multiplier,
            self.cfg,
            "CD:bootstrap_fit",
        )
        fit_cfg = replace(self.cfg, n_jobs=fit_n_jobs)
        self.global_models = self._fit_bootstrap_triplets(X_all, y_log, self.cfg.random_state + 77, fit_cfg)
        rid_arr = df["regime_id"].to_numpy(dtype=int)
        if self.cfg.regime_models_enabled:
            for rid in REGIME_IDS:
                mask = rid_arr == rid
                if int(mask.sum()) < 40:
                    continue
                self.models[rid] = self._fit_bootstrap_triplets(X_all.loc[mask], y_log[mask], self.cfg.random_state + 101 * rid, fit_cfg)
        X_all = None
        y_log = None
        rid_arr = None
        fit_cfg = None
        gc.collect()
        return self

    def _fit_bootstrap_triplets(self, X: pd.DataFrame, y_log: np.ndarray, seed: int, fit_cfg: ResearchConfig) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        n = len(X)
        k = max(1, int(round(n * fit_cfg.bootstrap_fraction)))
        bag_count = max(1, int(fit_cfg.bootstrap_models))
        triplets = []
        for b in range(bag_count):
            if bag_count == 1:
                idx = np.arange(n)
            else:
                idx = rng.choice(n, size=k, replace=True)
            Xb = X.iloc[idx]
            yb = y_log[idx]
            triplet = {}
            for q in fit_cfg.quantiles:
                key = f"q{int(round(q * 100)):02d}"
                model = make_median_model(fit_cfg, seed + b * 1009) if abs(q - 0.5) < 1e-9 else make_quantile_model(q, fit_cfg, seed + b * 1009)
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
                pipe.fit(Xb, yb)
                triplet[key] = pipe
            triplets.append(triplet)
            idx = None
            Xb = None
            yb = None
            gc.collect()
        return triplets

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_validator is None:
            raise RuntimeError("Model not fitted")
        X = self.feature_validator.validate(df)
        rid_arr = df["regime_id"].to_numpy(dtype=int) if "regime_id" in df.columns else regime_id_from_angle(df["angle"])
        out = pd.DataFrame(index=df.index)
        for name in ["cd_p10", "cd_p50", "cd_p90"]:
            out[name] = np.nan
        for rid in np.unique(rid_arr):
            bags = self.models.get(int(rid), self.global_models)
            mask = rid_arr == int(rid)
            Xs = X.loc[mask]
            available_keys = sorted({key for triplet in bags for key in triplet})
            if not available_keys:
                raise RuntimeError(f"No fitted quantile models for regime {int(rid)}")
            bag_preds = {key: [] for key in available_keys}
            for triplet in bags:
                for key in available_keys:
                    if key not in triplet:
                        continue
                    pred_log = np.asarray(triplet[key].predict(Xs), dtype=float)
                    bag_preds[key].append(np.maximum(np.exp(pred_log), CD_FLOOR))
            median_key = "q50" if bag_preds.get("q50") else available_keys[0]
            p50 = np.percentile(np.vstack(bag_preds[median_key]), 50, axis=0)
            # Geometry CV may deliberately fit only q50. Full training still uses q10/q50/q90.
            p10 = np.percentile(np.vstack(bag_preds["q10"]), 10, axis=0) if bag_preds.get("q10") else p50.copy()
            p90 = np.percentile(np.vstack(bag_preds["q90"]), 90, axis=0) if bag_preds.get("q90") else p50.copy()
            arr = np.sort(np.vstack([p10, p50, p90]).T, axis=1)
            out.loc[mask, "cd_p10"] = arr[:, 0]
            out.loc[mask, "cd_p50"] = arr[:, 1]
            out.loc[mask, "cd_p90"] = arr[:, 2]
        out["uncertainty_width"] = out["cd_p90"] - out["cd_p10"]
        return downcast_float64_to_float32(out)


class PerRegimeCalibration:
    def __init__(self, method: str = "linear"):
        self.method = method
        self.models: Dict[int, Any] = {}
        self.global_model: Any = None

    def _fit_one(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        x = np.asarray(y_pred, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=float)
        if self.method == "isotonic" and len(y) >= 30:
            mdl = IsotonicRegression(out_of_bounds="clip")
            mdl.fit(x.ravel(), y)
            return mdl
        mdl = LinearRegression()
        mdl.fit(x, y)
        return mdl

    def _apply_one(self, mdl: Any, y_pred: np.ndarray) -> np.ndarray:
        if isinstance(mdl, IsotonicRegression):
            return np.asarray(mdl.predict(y_pred), dtype=float)
        return np.asarray(mdl.predict(np.asarray(y_pred).reshape(-1, 1)), dtype=float)

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, regime_id: np.ndarray) -> "PerRegimeCalibration":
        self.global_model = self._fit_one(y_true, y_pred)
        for rid in REGIME_IDS:
            mask = np.asarray(regime_id, dtype=int) == rid
            if int(mask.sum()) >= 30:
                self.models[rid] = self._fit_one(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
        return self

    def apply(self, y_pred: np.ndarray, regime_id: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred, dtype=float)
        out = np.empty_like(y_pred)
        rid_arr = np.asarray(regime_id, dtype=int)
        for rid in np.unique(rid_arr):
            mask = rid_arr == int(rid)
            mdl = self.models.get(int(rid), self.global_model)
            out[mask] = self._apply_one(mdl, y_pred[mask])
        return np.maximum(out, CD_FLOOR)


class R1ResidualCorrector:
    """Targeted residual model for attached-flow Cd."""

    def __init__(self, feature_cols: List[str], random_state: int = 42):
        preferred = [
            "angle", "abs_angle", "angle_sq", "signed_angle_sq", "angle_cu",
            "log10_re", "angle_x_log10_re", "abs_angle_x_log10_re", "r1_edge_proximity",
            "t_max", "camber_max", "x_tmax", "x_cmax", "le_radius_proxy",
            "aft_thickness", "aft_camber", "curvature_energy_aft",
            "upper_curvature_energy_aft_06", "slope_variance_aft",
            "thickness_gradient_aft", "wake_proxy", "pressure_recovery_proxy",
            "aft_pressure_recovery_proxy",
        ]
        z_cols = [c for c in feature_cols if c.startswith("z_")][:8]
        self.feature_cols = [c for c in preferred if c in feature_cols] + z_cols
        self.random_state = int(random_state)
        self.imputer: Optional[SimpleImputer] = None
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.clip_value: float = 0.0
        self.fitted: bool = False
        self.training_rows: int = 0

    def fit(self, df: pd.DataFrame, pred: pd.DataFrame) -> "R1ResidualCorrector":
        if not self.feature_cols or "regime_id" not in df.columns:
            return self
        mask = df["regime_id"].to_numpy(dtype=int) == 1
        self.training_rows = int(mask.sum())
        if self.training_rows < 120:
            return self
        resid = df.loc[mask, "cd"].to_numpy(dtype=float) - pred.loc[mask, "cd_p50"].to_numpy(dtype=float)
        if len(resid) < 120 or not np.isfinite(resid).any() or float(np.nanstd(resid)) < 1e-6:
            return self
        X = df.loc[mask, self.feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.imputer = SimpleImputer(strategy="median")
        X = self.imputer.fit_transform(X).astype(np.float32, copy=False)
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=63,
            min_samples_leaf=18,
            l2_regularization=0.02,
            random_state=self.random_state,
        )
        self.model.fit(X, resid)
        self.clip_value = float(max(0.001, 1.25 * np.nanquantile(np.abs(resid), 0.98)))
        self.fitted = True
        return self

    def apply(self, df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.model is None or self.imputer is None or not self.feature_cols:
            return pred
        mask = df["regime_id"].to_numpy(dtype=int) == 1
        if int(mask.sum()) == 0:
            return pred
        out = pred.copy()
        X = df.loc[mask, self.feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        X = self.imputer.transform(X).astype(np.float32, copy=False)
        corr = np.asarray(self.model.predict(X), dtype=float)
        corr = np.clip(corr, -self.clip_value, self.clip_value)
        cols = ["cd_p10", "cd_p50", "cd_p90"]
        arr = out.loc[mask, cols].to_numpy(dtype=float)
        arr = np.maximum(arr + corr[:, None], CD_FLOOR)
        arr = np.sort(arr, axis=1)
        out.loc[mask, "cd_p10"] = arr[:, 0]
        out.loc[mask, "cd_p50"] = arr[:, 1]
        out.loc[mask, "cd_p90"] = arr[:, 2]
        out["uncertainty_width"] = out["cd_p90"] - out["cd_p10"]
        return downcast_float64_to_float32(out)

    def summary_row(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.fitted),
            "training_rows": int(self.training_rows),
            "n_features": int(len(self.feature_cols)),
            "clip_value": float(self.clip_value),
            "feature_cols": ",".join(self.feature_cols),
        }


class TargetedResidualCorrector:
    """Targeted residual model for specific regimes / tail slices."""

    def __init__(
        self,
        name: str,
        feature_cols: List[str],
        random_state: int = 42,
        target_regimes: Optional[Iterable[int]] = None,
        angle_min: Optional[float] = None,
        angle_max: Optional[float] = None,
        true_cd_min: Optional[float] = None,
        pred_cd_min: Optional[float] = None,
        min_rows: int = 120,
        learning_rate: float = 0.05,
        max_iter: int = 220,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 18,
        l2_regularization: float = 0.02,
        clip_floor: float = 0.001,
        clip_quantile: float = 0.98,
    ):
        preferred = [
            "angle", "abs_angle", "angle_sq", "signed_angle_sq", "angle_cu",
            "log10_re", "angle_x_log10_re", "abs_angle_x_log10_re", "r1_edge_proximity",
            "t_max", "camber_max", "x_tmax", "x_cmax", "le_radius_proxy",
            "aft_thickness", "aft_camber", "curvature_energy_aft",
            "upper_curvature_energy_aft_06", "slope_variance_aft",
            "thickness_gradient_aft", "wake_proxy", "pressure_recovery_proxy",
            "aft_pressure_recovery_proxy",
        ]
        z_cols = [c for c in feature_cols if c.startswith("z_")][:8]
        self.base_feature_cols = [c for c in preferred if c in feature_cols] + z_cols
        self.pred_feature_cols = ["cd_p10", "cd_p50", "cd_p90", "uncertainty_width"]
        self.name = str(name)
        self.random_state = int(random_state)
        self.target_regimes = {int(r) for r in (target_regimes or [])}
        self.angle_min = None if angle_min is None else float(angle_min)
        self.angle_max = None if angle_max is None else float(angle_max)
        self.true_cd_min = None if true_cd_min is None else float(true_cd_min)
        self.pred_cd_min = None if pred_cd_min is None else float(pred_cd_min)
        self.min_rows = int(min_rows)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.clip_floor = float(clip_floor)
        self.clip_quantile = float(clip_quantile)
        self.imputer: Optional[SimpleImputer] = None
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.clip_value: float = 0.0
        self.fitted: bool = False
        self.training_rows: int = 0

    def _mask(self, df: pd.DataFrame, pred: pd.DataFrame, fit_phase: bool) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        if self.target_regimes:
            mask &= np.isin(df["regime_id"].to_numpy(dtype=int), list(self.target_regimes))
        if self.angle_min is not None:
            mask &= df["angle"].to_numpy(dtype=float) >= self.angle_min
        if self.angle_max is not None:
            mask &= df["angle"].to_numpy(dtype=float) <= self.angle_max
        if fit_phase and self.true_cd_min is not None and "cd" in df.columns:
            mask &= df["cd"].to_numpy(dtype=float) >= self.true_cd_min
        if (not fit_phase) and self.pred_cd_min is not None:
            mask &= pred["cd_p50"].to_numpy(dtype=float) >= self.pred_cd_min
        return mask

    def _design_matrix(self, df: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
        feat_df = df.loc[mask, self.base_feature_cols].copy()
        feat_df["pred_cd_p10"] = pred.loc[mask, "cd_p10"].to_numpy(dtype=float)
        feat_df["pred_cd_p50"] = pred.loc[mask, "cd_p50"].to_numpy(dtype=float)
        feat_df["pred_cd_p90"] = pred.loc[mask, "cd_p90"].to_numpy(dtype=float)
        feat_df["pred_width"] = pred.loc[mask, "uncertainty_width"].to_numpy(dtype=float)
        feat_df["pred_log_cd"] = np.log(np.maximum(pred.loc[mask, "cd_p50"].to_numpy(dtype=float), CD_FLOOR))
        return feat_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    def fit(self, df: pd.DataFrame, pred: pd.DataFrame) -> "TargetedResidualCorrector":
        if not self.base_feature_cols or "regime_id" not in df.columns:
            return self
        mask = self._mask(df, pred, fit_phase=True)
        self.training_rows = int(mask.sum())
        if self.training_rows < self.min_rows:
            return self
        resid = df.loc[mask, "cd"].to_numpy(dtype=float) - pred.loc[mask, "cd_p50"].to_numpy(dtype=float)
        if len(resid) < self.min_rows or not np.isfinite(resid).any() or float(np.nanstd(resid)) < 1e-6:
            return self
        X = self._design_matrix(df, pred, mask)
        self.imputer = SimpleImputer(strategy="median")
        X = self.imputer.fit_transform(X).astype(np.float32, copy=False)
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.model.fit(X, resid)
        self.clip_value = float(max(self.clip_floor, 1.25 * np.nanquantile(np.abs(resid), self.clip_quantile)))
        self.fitted = True
        return self

    def apply(self, df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.model is None or self.imputer is None or not self.base_feature_cols:
            return pred
        mask = self._mask(df, pred, fit_phase=False)
        if int(mask.sum()) == 0:
            return pred
        out = pred.copy()
        X = self._design_matrix(df, out, mask)
        X = self.imputer.transform(X).astype(np.float32, copy=False)
        corr = np.asarray(self.model.predict(X), dtype=float)
        corr = np.clip(corr, -self.clip_value, self.clip_value)
        cols = ["cd_p10", "cd_p50", "cd_p90"]
        arr = out.loc[mask, cols].to_numpy(dtype=float)
        arr = np.maximum(arr + corr[:, None], CD_FLOOR)
        arr = np.sort(arr, axis=1)
        out.loc[mask, "cd_p10"] = arr[:, 0]
        out.loc[mask, "cd_p50"] = arr[:, 1]
        out.loc[mask, "cd_p90"] = arr[:, 2]
        out["uncertainty_width"] = out["cd_p90"] - out["cd_p10"]
        return downcast_float64_to_float32(out)

    def summary_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": bool(self.fitted),
            "training_rows": int(self.training_rows),
            "n_features": int(len(self.base_feature_cols) + 5),
            "clip_value": float(self.clip_value),
            "target_regimes": ",".join(str(r) for r in sorted(self.target_regimes)),
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
            "true_cd_min": self.true_cd_min,
            "pred_cd_min": self.pred_cd_min,
            "feature_cols": ",".join(self.base_feature_cols + ["pred_cd_p10", "pred_cd_p50", "pred_cd_p90", "pred_width", "pred_log_cd"]),
        }


class MonotoneSliceShiftCorrector:
    """Stable one-dimensional shift corrector for a narrow prediction slice."""

    def __init__(
        self,
        name: str,
        random_state: int = 42,
        target_regimes: Optional[Iterable[int]] = None,
        angle_min: Optional[float] = None,
        angle_max: Optional[float] = None,
        true_cd_min: Optional[float] = None,
        pred_cd_min: Optional[float] = None,
        min_rows: int = 60,
        positive_only_shift: bool = True,
        min_shift_quantile: float = 0.60,
    ):
        self.name = str(name)
        self.random_state = int(random_state)
        self.target_regimes = {int(r) for r in (target_regimes or [])}
        self.angle_min = None if angle_min is None else float(angle_min)
        self.angle_max = None if angle_max is None else float(angle_max)
        self.true_cd_min = None if true_cd_min is None else float(true_cd_min)
        self.pred_cd_min = None if pred_cd_min is None else float(pred_cd_min)
        self.min_rows = int(min_rows)
        self.positive_only_shift = bool(positive_only_shift)
        self.min_shift_quantile = float(min_shift_quantile)
        self.model: Optional[Any] = None
        self.fitted: bool = False
        self.training_rows: int = 0
        self.max_shift: float = 0.0

    def _mask(self, df: pd.DataFrame, pred: pd.DataFrame, fit_phase: bool) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        if self.target_regimes:
            mask &= np.isin(df["regime_id"].to_numpy(dtype=int), list(self.target_regimes))
        if self.angle_min is not None:
            mask &= df["angle"].to_numpy(dtype=float) >= self.angle_min
        if self.angle_max is not None:
            mask &= df["angle"].to_numpy(dtype=float) <= self.angle_max
        if fit_phase and self.true_cd_min is not None and "cd" in df.columns:
            mask &= df["cd"].to_numpy(dtype=float) >= self.true_cd_min
        if (not fit_phase) and self.pred_cd_min is not None:
            mask &= pred["cd_p50"].to_numpy(dtype=float) >= self.pred_cd_min
        return mask

    def fit(self, df: pd.DataFrame, pred: pd.DataFrame) -> "MonotoneSliceShiftCorrector":
        mask = self._mask(df, pred, fit_phase=True)
        self.training_rows = int(mask.sum())
        if self.training_rows < self.min_rows:
            return self
        x = pred.loc[mask, "cd_p50"].to_numpy(dtype=float)
        y = df.loc[mask, "cd"].to_numpy(dtype=float)
        shift = y - x
        if not np.isfinite(shift).any():
            return self
        if self.positive_only_shift and float(np.nanquantile(shift, self.min_shift_quantile)) <= 0.0:
            return self
        if len(x) >= 40:
            mdl = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mdl.fit(x, y)
        else:
            mdl = LinearRegression()
            mdl.fit(x.reshape(-1, 1), y)
        self.model = mdl
        self.max_shift = float(max(0.0015, 1.20 * np.nanquantile(np.abs(shift), 0.99)))
        self.fitted = True
        return self

    def apply(self, df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            return pred
        mask = self._mask(df, pred, fit_phase=False)
        if int(mask.sum()) == 0:
            return pred
        out = pred.copy()
        old_center = out.loc[mask, "cd_p50"].to_numpy(dtype=float)
        if isinstance(self.model, IsotonicRegression):
            new_center = np.asarray(self.model.predict(old_center), dtype=float)
        else:
            new_center = np.asarray(self.model.predict(old_center.reshape(-1, 1)), dtype=float)
        shift = new_center - old_center
        if self.positive_only_shift:
            shift = np.maximum(shift, 0.0)
        shift = np.clip(shift, -self.max_shift, self.max_shift)
        cols = ["cd_p10", "cd_p50", "cd_p90"]
        arr = out.loc[mask, cols].to_numpy(dtype=float)
        arr = np.maximum(arr + shift[:, None], CD_FLOOR)
        arr = np.sort(arr, axis=1)
        out.loc[mask, "cd_p10"] = arr[:, 0]
        out.loc[mask, "cd_p50"] = arr[:, 1]
        out.loc[mask, "cd_p90"] = arr[:, 2]
        out["uncertainty_width"] = out["cd_p90"] - out["cd_p10"]
        return downcast_float64_to_float32(out)

    def summary_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": bool(self.fitted),
            "training_rows": int(self.training_rows),
            "n_features": 1,
            "clip_value": float(self.max_shift),
            "target_regimes": ",".join(str(r) for r in sorted(self.target_regimes)),
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
            "true_cd_min": self.true_cd_min,
            "pred_cd_min": self.pred_cd_min,
            "feature_cols": "pred_cd_p50",
        }


def apply_residual_corrector_stack(correctors: Sequence[Any], df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = pred
    for corr in correctors:
        if corr is None:
            continue
        out = corr.apply(df, out)
    return out


def split_calibration_for_r1_corrector(df: pd.DataFrame, group_col: str, random_state: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(df) < 400:
        return None, None
    try:
        if group_col in df.columns:
            groups = df[group_col].astype(str).to_numpy()
            unique_groups = np.unique(groups)
            if len(unique_groups) < 2:
                return None, None
            folds = list(GroupKFold(n_splits=2).split(df, groups=groups))
            fold_id = int(abs(random_state)) % len(folds)
            fit_idx, conf_idx = folds[fold_id]
        else:
            perm = np.random.default_rng(random_state).permutation(len(df))
            cut = len(perm) // 2
            fit_idx, conf_idx = perm[:cut], perm[cut:]
        if len(fit_idx) < 200 or len(conf_idx) < 200:
            return None, None
        return np.asarray(fit_idx, dtype=int), np.asarray(conf_idx, dtype=int)
    except Exception:
        return None, None


class RegimeWiseConformalCalibrator:
    def __init__(self, levels: Tuple[float, float, float] = (0.80, 0.90, 0.95)):
        self.levels = tuple(levels)
        self.qhat: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _qhat(scores: np.ndarray, level: float) -> float:
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return float("nan")
        n = len(scores)
        q = min(1.0, math.ceil((n + 1) * level) / n)
        return float(np.quantile(scores, q, method="higher"))

    def fit(self, df_cal: pd.DataFrame, pred_cal: pd.DataFrame) -> "RegimeWiseConformalCalibrator":
        y = df_cal["cd"].to_numpy(dtype=float)
        p50 = pred_cal["cd_p50"].to_numpy(dtype=float)
        abs_scores = np.abs(y - p50)
        # interval nonconformity: positive amount by which true value lies outside p10/p90.
        interval_scores = np.maximum.reduce([
            pred_cal["cd_p10"].to_numpy(dtype=float) - y,
            y - pred_cal["cd_p90"].to_numpy(dtype=float),
            np.zeros_like(y),
        ])
        rid = df_cal["regime_id"].to_numpy(dtype=int)
        self.qhat = {}
        self.qhat["global"] = {}
        for level in self.levels:
            key = f"qhat_{int(level * 100)}"
            self.qhat["global"][key] = self._qhat(abs_scores, level)
            self.qhat["global"][f"interval_{int(level * 100)}"] = self._qhat(interval_scores, level)
        for r in REGIME_IDS:
            mask = rid == r
            if int(mask.sum()) < 30:
                continue
            scope = REGIME_NAMES[r]
            self.qhat[scope] = {}
            for level in self.levels:
                self.qhat[scope][f"qhat_{int(level * 100)}"] = self._qhat(abs_scores[mask], level)
                self.qhat[scope][f"interval_{int(level * 100)}"] = self._qhat(interval_scores[mask], level)
        return self

    def _row_qhat(self, rid: int, level: float, interval: bool = False) -> float:
        scope = REGIME_NAMES.get(int(rid), "global")
        key = ("interval_" if interval else "qhat_") + str(int(level * 100))
        if scope in self.qhat and np.isfinite(self.qhat[scope].get(key, np.nan)):
            return float(self.qhat[scope][key])
        return float(self.qhat["global"].get(key, 0.0))

    def apply(self, df: pd.DataFrame, pred: pd.DataFrame, level: float = 0.90) -> pd.DataFrame:
        rid = df["regime_id"].to_numpy(dtype=int)
        abs_q = np.array([self._row_qhat(r, level, interval=False) for r in rid], dtype=float)
        int_q = np.array([self._row_qhat(r, level, interval=True) for r in rid], dtype=float)
        out = pred.copy()
        pct = int(level * 100)
        out[f"cd_c{pct}_lo"] = np.maximum(out["cd_p50"].to_numpy(dtype=float) - abs_q, CD_FLOOR)
        out[f"cd_c{pct}_hi"] = np.maximum(out["cd_p50"].to_numpy(dtype=float) + abs_q, CD_FLOOR)
        out[f"cd_qc{pct}_lo"] = np.maximum(out["cd_p10"].to_numpy(dtype=float) - int_q, CD_FLOOR)
        out[f"cd_qc{pct}_hi"] = np.maximum(out["cd_p90"].to_numpy(dtype=float) + int_q, CD_FLOOR)
        return downcast_float64_to_float32(out)

    def evaluate(self, df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        rows = []
        y = df["cd"].to_numpy(dtype=float)
        rid = df["regime_id"].to_numpy(dtype=int)
        for level in self.levels:
            pp = self.apply(df, pred, level)
            pct = int(level * 100)
            lo = pp[f"cd_c{pct}_lo"].to_numpy(dtype=float)
            hi = pp[f"cd_c{pct}_hi"].to_numpy(dtype=float)
            covered = (y >= lo) & (y <= hi)
            rows.append({"scope": "global", "level": pct, "nominal": level, "coverage": float(np.mean(covered)), "n": int(len(y))})
            for r in REGIME_IDS:
                m = rid == r
                if int(m.sum()) == 0:
                    continue
                rows.append({"scope": REGIME_NAMES[r], "level": pct, "nominal": level, "coverage": float(np.mean(covered[m])), "n": int(m.sum())})
        return pd.DataFrame(rows)


def fit_predict_pipeline(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, group_col: str, cfg: ResearchConfig, paths: Optional[Dict[str, Path]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    all_apply = pd.concat([train, cal, test], ignore_index=True)
    train_z, all_z, pca_meta = build_pca_latents(train, all_apply, cfg)
    train2 = all_z.iloc[:len(train)].reset_index(drop=True)
    cal2 = all_z.iloc[len(train):len(train) + len(cal)].reset_index(drop=True)
    test2 = all_z.iloc[len(train) + len(cal):].reset_index(drop=True)
    z_cols = pca_meta["z_cols"]
    feature_cols = select_feature_columns(train2, z_cols, cfg)
    report_path = (paths["models"] / "missing_feature_report.txt") if paths else Path("missing_feature_report.txt")
    model = BootstrapRegimeQuantileEnsemble(feature_cols, cfg).fit(train2, report_path)
    pred_cal = model.predict(cal2)
    cal_method = "isotonic" if cfg.use_isotonic_calibration else "linear"
    micro = PerRegimeCalibration(method=cal_method).fit(cal2["cd"].to_numpy(float), pred_cal["cd_p50"].to_numpy(float), cal2["regime_id"].to_numpy(int))
    pred_cal["cd_p50"] = micro.apply(pred_cal["cd_p50"].to_numpy(float), cal2["regime_id"].to_numpy(int))
    pred_cal["cd_p10"] = np.minimum(pred_cal["cd_p10"], pred_cal["cd_p50"])
    pred_cal["cd_p90"] = np.maximum(pred_cal["cd_p90"], pred_cal["cd_p50"])
    pred_cal["uncertainty_width"] = pred_cal["cd_p90"] - pred_cal["cd_p10"]
    conformal = RegimeWiseConformalCalibrator().fit(cal2, pred_cal)
    pred_test = model.predict(test2)
    pred_test["cd_p50"] = micro.apply(pred_test["cd_p50"].to_numpy(float), test2["regime_id"].to_numpy(int))
    pred_test["cd_p10"] = np.minimum(pred_test["cd_p10"], pred_test["cd_p50"])
    pred_test["cd_p90"] = np.maximum(pred_test["cd_p90"], pred_test["cd_p50"])
    pred_test["uncertainty_width"] = pred_test["cd_p90"] - pred_test["cd_p10"]
    pred_cal = downcast_float64_to_float32(pred_cal)
    pred_test = downcast_float64_to_float32(pred_test)
    del all_apply, all_z
    gc.collect()
    meta = {"pca_meta": pca_meta, "z_cols": z_cols, "feature_cols": feature_cols, "model": model, "micro": micro, "conformal": conformal, "train_z": train2, "cal_z": cal2, "test_z": test2}
    return pred_test, meta


def write_geometry_validation_removed_artifacts(paths: Dict[str, Path], group_col: str, tag: str = "baseline") -> pd.DataFrame:
    """Write compatibility artifacts without running LOGO/geometry CV refits."""
    columns = ["scope", group_col, "n", "mae", "rmse", "r2", "bias", "validation_mode"]
    out = pd.DataFrame(columns=columns)
    suffix = "" if tag == "baseline" else f"_{tag}"
    out.to_csv(paths["logo"] / f"logo_metrics{suffix}.csv", index=False)
    pd.DataFrame([{
        "validation_mode": "removed",
        "n_logo_geometries": 0,
        "mae_mean": np.nan,
        "rmse_mean": np.nan,
    }]).to_csv(paths["logo"] / f"logo_summary{suffix}.csv", index=False)
    (paths["logo"] / f"logo_report{suffix}.md").write_text(
        f"# Geometry Validation Report - {tag}\n\n"
        "LOGO refit validation was removed from day4_cd.py.\n\n"
        "Geometry GroupKFold CV is produced separately in `tables/groupkfold_*` and `reports/geometry_groupkfold_cv_report.md`.\n",
        encoding=OUTPUT_TEXT_ENCODING,
    )
    return out


def run_geometry_groupkfold_cv(df: pd.DataFrame, group_col: str, cfg: ResearchConfig, paths: Dict[str, Path], weighted: bool = False, tag: str = "baseline") -> Dict[str, pd.DataFrame]:
    """Evaluate Cd generalization with GroupKFold by geometry, without LOGO refits."""
    suffix = "" if tag == "baseline" else f"_{tag}"
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        empty = pd.DataFrame()
        (paths["reports"] / f"geometry_groupkfold_cv_report{suffix}.md").write_text(
            "# CD Geometry GroupKFold CV\n\nSkipped: fewer than two geometry groups.",
            encoding=OUTPUT_TEXT_ENCODING,
        )
        return {"predictions": empty, "overall": empty, "regime": empty, "geometry": empty, "folds": empty}

    n_splits = max(2, min(int(cfg.geometry_cv_splits), len(unique_groups)))
    cv_cfg = replace(
        cfg,
        quantiles=(0.50,),
        bootstrap_models=max(1, int(cfg.geometry_cv_bootstrap_models)),
        n_estimators=max(25, min(int(cfg.n_estimators), int(cfg.geometry_cv_n_estimators))),
        regime_models_enabled=(not cfg.geometry_cv_global_only),
    ) if cfg.geometry_cv_p50_only else cfg
    cv_mode = (
        f"p50_{'global' if cfg.geometry_cv_global_only else 'regime'}"
        f"_iter_{cv_cfg.n_estimators}_bootstrap_{cv_cfg.bootstrap_models}"
        if cfg.geometry_cv_p50_only
        else f"full_quantile_bootstrap_{cv_cfg.bootstrap_models}"
    )
    print(f"[Day4C-CD-V3] Geometry GroupKFold CV: folds={n_splits} tag={tag} mode={cv_mode}", flush=True)
    gkf = GroupKFold(n_splits=n_splits)

    prediction_file = paths["tables"] / f"groupkfold_predictions{suffix}.csv"
    if prediction_file.exists():
        prediction_file.unlink()
    coverage_chunks: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_pool_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        print(f"  [GeometryCV] fold {fold_idx}/{n_splits} start", flush=True)
        train_pool = test = train = cal = pred = meta = test2 = conformal = cov = None
        y = p = row = pred_fold = None
        train_pool = df.iloc[train_pool_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        try:
            train, cal = train_cal_split_by_geometry(
                train_pool,
                group_col,
                cfg,
                cfg.random_state + 1009 * fold_idx + (50000 if weighted else 0),
                cfg.calibration_size,
            )
            pred, meta = fit_predict_pipeline_v3(train, cal, test, group_col, cv_cfg, paths=None, weighted=weighted)
            test2 = meta["test_z"]
            conformal = meta["conformal"]
            for lvl in (0.80, 0.90, 0.95):
                pred = conformal.apply(test2, pred, lvl)
            cov = conformal.evaluate(test2, pred)
            cov["fold"] = int(fold_idx)
            cov["tag"] = tag
            coverage_chunks.append(cov)

            y = test2["cd"].to_numpy(float)
            p = pred["cd_p50"].to_numpy(float)
            row = metric_row(y, p, f"groupkfold_fold_{fold_idx}")
            row.update({
                "fold": int(fold_idx),
                "tag": tag,
                "weighted": bool(weighted),
                "n_test_geometries": int(test2[group_col].astype(str).nunique()),
                "validation_mode": "geometry_groupkfold",
                "cv_mode": cv_mode,
            })
            fold_rows.append(row)

            base_cols = [group_col, "angle", "cd", "regime_id"]
            if "log10_re" in test2.columns:
                base_cols.append("log10_re")
            pred_cols = [c for c in pred.columns if c.startswith("cd_") or c in {"uncertainty_width"}]
            pred_fold = pd.concat(
                [test2[base_cols].reset_index(drop=True), pred[pred_cols].reset_index(drop=True)],
                axis=1,
            )
            pred_fold["fold"] = int(fold_idx)
            pred_fold["tag"] = tag
            pred_fold = downcast_float64_to_float32(pred_fold)
            pred_fold.to_csv(prediction_file, mode="a", header=(not prediction_file.exists()), index=False)
            pd.DataFrame(fold_rows).to_csv(paths["tables"] / f"groupkfold_fold_metrics{suffix}.csv", index=False)
            print(f"  [GeometryCV] fold {fold_idx}/{n_splits} done: rmse={row['rmse']:.6f} mae={row['mae']:.6f}", flush=True)
        except Exception as exc:
            fold_rows.append({
                "scope": f"groupkfold_fold_{fold_idx}",
                "fold": int(fold_idx),
                "tag": tag,
                "weighted": bool(weighted),
                "n": int(len(test)),
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "bias": np.nan,
                "n_test_geometries": int(test[group_col].astype(str).nunique()) if group_col in test.columns else 0,
                "validation_mode": "geometry_groupkfold",
                "cv_mode": cv_mode,
                "error": repr(exc),
            })
            pd.DataFrame(fold_rows).to_csv(paths["tables"] / f"groupkfold_fold_metrics{suffix}.csv", index=False)
            print(f"  [GeometryCV] fold {fold_idx}/{n_splits} failed: {exc!r}", flush=True)
        finally:
            train_pool = None
            test = None
            train = None
            cal = None
            pred = None
            meta = None
            test2 = None
            conformal = None
            cov = None
            y = None
            p = None
            row = None
            pred_fold = None
            gc.collect()

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(paths["tables"] / f"groupkfold_fold_metrics{suffix}.csv", index=False)
    coverage = pd.concat(coverage_chunks, ignore_index=True) if coverage_chunks else pd.DataFrame()
    coverage.to_csv(paths["tables"] / f"groupkfold_conformal_coverage{suffix}.csv", index=False)

    if prediction_file.exists() and prediction_file.stat().st_size > 0:
        pred_all = pd.read_csv(prediction_file, low_memory=False)
    else:
        pred_all = pd.DataFrame(columns=[group_col, "angle", "cd", "regime_id", "cd_p50", "fold", "tag"])

    if len(pred_all) and "cd_p50" in pred_all.columns:
        overall = pd.DataFrame([metric_row(pred_all["cd"].to_numpy(float), pred_all["cd_p50"].to_numpy(float), "geometry_groupkfold_overall")])
        overall["tag"] = tag
        regime_rows: List[Dict[str, Any]] = []
        for rid in REGIME_IDS:
            m = pred_all["regime_id"].to_numpy(int) == rid
            if int(m.sum()):
                rr = metric_row(pred_all.loc[m, "cd"].to_numpy(float), pred_all.loc[m, "cd_p50"].to_numpy(float), REGIME_NAMES[rid])
                rr["regime_id"] = int(rid)
                rr["tag"] = tag
                regime_rows.append(rr)
        regime = pd.DataFrame(regime_rows)
        geom_rows: List[Dict[str, Any]] = []
        for geom, grp in pred_all.groupby(group_col, sort=False):
            if len(grp) < 2:
                continue
            gr = metric_row(grp["cd"].to_numpy(float), grp["cd_p50"].to_numpy(float), str(geom))
            gr[group_col] = str(geom)
            gr["tag"] = tag
            geom_rows.append(gr)
        geometry = pd.DataFrame(geom_rows).sort_values("mae", ascending=False).reset_index(drop=True) if geom_rows else pd.DataFrame()
    else:
        overall = pd.DataFrame()
        regime = pd.DataFrame()
        geometry = pd.DataFrame()

    overall.to_csv(paths["tables"] / f"groupkfold_overall_metrics{suffix}.csv", index=False)
    regime.to_csv(paths["tables"] / f"groupkfold_regime_metrics{suffix}.csv", index=False)
    geometry.to_csv(paths["tables"] / f"groupkfold_geometry_metrics{suffix}.csv", index=False)

    lines = [
        f"# CD Geometry GroupKFold CV - {tag}",
        "",
        f"- Folds: {n_splits}",
        f"- Geometry groups: {len(unique_groups)}",
        f"- Weighted model: {bool(weighted)}",
        f"- CV model mode: {cv_mode}",
        "",
        "## Overall",
        _to_markdown(overall.round(6)) if len(overall) else "No successful folds.",
        "",
        "## Fold Metrics",
        _to_markdown(folds.round(6)) if len(folds) else "No fold metrics.",
        "",
        "## Regime Metrics",
        _to_markdown(regime.round(6)) if len(regime) else "No regime metrics.",
        "",
        "## Worst Geometry Groups",
        _to_markdown(geometry.head(20).round(6)) if len(geometry) else "No geometry metrics.",
    ]
    (paths["reports"] / f"geometry_groupkfold_cv_report{suffix}.md").write_text("\n".join(lines), encoding=OUTPUT_TEXT_ENCODING)
    return {"predictions": pred_all, "overall": overall, "regime": regime, "geometry": geometry, "folds": folds, "coverage": coverage}
def geometry_failure_analysis(df: pd.DataFrame, logo: pd.DataFrame, group_col: str, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, str]:
    if logo is None or logo.empty or "mae" not in logo.columns:
        empty = pd.DataFrame()
        return empty, "Geometry refit validation removed; geometry failure analysis skipped."
    feats = [c for c in GEOMETRY_FAILURE_FEATURES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        return pd.DataFrame(), "No geometry descriptor columns found for failure analysis."
    geom = df.groupby(group_col)[feats].mean().reset_index()
    merged = logo.merge(geom, on=group_col, how="left")
    corr_rows = []
    valid = merged.dropna(subset=["mae"])
    for f in feats:
        if valid[f].notna().sum() >= 5:
            corr_rows.append({"feature": f, "corr_with_logo_mae": float(valid[f].corr(valid["mae"], method="spearman"))})
    corr = pd.DataFrame(corr_rows).sort_values("corr_with_logo_mae", key=lambda s: s.abs(), ascending=False)
    merged.to_csv(paths["tables"] / "geometry_failure_analysis.csv", index=False)
    corr.to_csv(paths["tables"] / "geometry_failure_correlations.csv", index=False)
    txt = ["# Geometry Failure Analysis", ""]
    if not valid.empty:
        best = valid.sort_values("mae").head(1)
        worst = valid.sort_values("mae").tail(1)
        median = valid.iloc[(valid["mae"] - valid["mae"].median()).abs().argsort().iloc[0:1]]
        txt += ["## Representative geometries", "", "### Best", best[[group_col, "mae", "rmse", "r2"] + feats[:8]].round(6).to_markdown(index=False), ""]
        txt += ["### Median", median[[group_col, "mae", "rmse", "r2"] + feats[:8]].round(6).to_markdown(index=False), ""]
        txt += ["### Worst", worst[[group_col, "mae", "rmse", "r2"] + feats[:8]].round(6).to_markdown(index=False), ""]
    txt += ["## Descriptor correlations with geometry validation MAE", "", corr.head(15).round(6).to_markdown(index=False) if not corr.empty else "No stable correlations.", ""]
    report = "\n".join(txt)
    (paths["reports"] / "geometry_failure_report.md").write_text(report, encoding=OUTPUT_TEXT_ENCODING)
    return merged, report


def save_diagnostics(test: pd.DataFrame, pred: pd.DataFrame, logo: pd.DataFrame, geom_ood: GeometryOODDetector, flow_ood: FlowOODDetector, coverage: pd.DataFrame, paths: Dict[str, Path]) -> None:
    figdir = paths["figures"]
    y = test["cd"].to_numpy(float)
    p = pred["cd_p50"].to_numpy(float)
    residual = p - y
    abs_err = np.abs(residual)
    def _save(name: str):
        plt.tight_layout(); plt.savefig(figdir / name, dpi=160, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8, 6))
    plt.scatter(y, p, s=8, alpha=0.40, label="samples")
    lo = float(min(np.min(y), np.min(p)))
    hi = float(max(np.max(y), np.max(p)))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="ideal y=x")
    if len(y) >= 2:
        try:
            a, b = np.polyfit(y, p, 1)
            xs = np.linspace(lo, hi, 200)
            plt.plot(xs, a * xs + b, color="tab:red", lw=1.5, label=f"fit y={a:.3f}x{b:+.3e}")
        except Exception:
            pass
    plt.xlabel("True Cd")
    plt.ylabel("Predicted Cd")
    plt.title("Cd True vs Predicted")
    plt.legend()
    _save("cd_true_vs_pred.png")
    plt.figure(figsize=(8, 5)); plt.scatter(test["angle"], residual, s=8, alpha=0.45); plt.axhline(0, lw=1); plt.xlabel("AoA [deg]"); plt.ylabel("Residual Cd_pred - Cd_true"); plt.title("Residual vs AoA"); _save("residual_vs_aoa.png")
    plt.figure(figsize=(8, 5)); plt.scatter(y, residual, s=8, alpha=0.45); plt.axhline(0, lw=1); plt.xlabel("True Cd"); plt.ylabel("Residual"); plt.title("Residual vs Cd"); _save("residual_vs_cd.png")
    plt.figure(figsize=(8, 5)); plt.scatter(pred["uncertainty_width"], abs_err, s=8, alpha=0.45); plt.xlabel("Uncertainty width p90-p10"); plt.ylabel("Absolute error"); plt.title("Uncertainty vs Error"); _save("uncertainty_vs_error.png")
    if logo is not None and not logo.empty and "mae" in logo.columns:
        plt.figure(figsize=(8, 5)); plt.hist(logo["mae"].dropna(), bins=30); plt.xlabel("Geometry validation MAE"); plt.ylabel("Count"); plt.title("Geometry Validation Distribution"); _save("geometry_validation_distribution.png")
    if geom_ood.train_distances is not None:
        plt.figure(figsize=(8, 5)); plt.hist(geom_ood.train_distances, bins=30, alpha=0.55, label="train geometry centroids"); plt.hist(pred["geometry_ood_score"].dropna(), bins=30, alpha=0.45, label="test rows"); plt.axvline(geom_ood.threshold, ls="--", lw=2); plt.xlabel("Geometry Mahalanobis distance"); plt.ylabel("Count"); plt.title("Geometry OOD Score Distribution"); plt.legend(); _save("ood_score_distribution.png")
    if flow_ood.train_distances is not None:
        plt.figure(figsize=(8, 5)); plt.hist(flow_ood.train_distances, bins=30, alpha=0.55, label="train flow states"); plt.hist(pred["flow_ood_score"].dropna(), bins=30, alpha=0.45, label="test rows"); plt.axvline(flow_ood.threshold, ls="--", lw=2); plt.xlabel("Flow Mahalanobis distance"); plt.ylabel("Count"); plt.title("Flow OOD Score Distribution"); plt.legend(); _save("flow_ood_score_distribution.png")
    glob = coverage[coverage["scope"] == "global"]
    plt.figure(figsize=(8, 5)); plt.plot(glob["level"], glob["coverage"], marker="o", label="actual"); plt.plot(glob["level"], glob["level"] / 100.0, marker="x", ls="--", label="nominal"); plt.ylim(0, 1.02); plt.xlabel("Nominal coverage [%]"); plt.ylabel("Actual coverage"); plt.title("Conformal Coverage Plot"); plt.legend(); _save("conformal_coverage_plot.png")


def export_r1_diagnostics(test_df: pd.DataFrame, pred: pd.DataFrame, group_col: str, paths: Dict[str, Path], tag: str = "baseline") -> None:
    if "regime_id" not in test_df.columns or "angle" not in test_df.columns:
        return
    tmp = test_df[[group_col, "angle", "cd", "regime_id"]].copy().reset_index(drop=True)
    tmp["cd_pred"] = pred["cd_p50"].to_numpy(float)
    tmp["uncertainty_width"] = pred["uncertainty_width"].to_numpy(float)
    r1 = tmp[tmp["regime_id"].to_numpy(dtype=int) == 1].copy()
    if r1.empty:
        return
    r1["resid"] = r1["cd_pred"] - r1["cd"]
    r1["abs_err"] = np.abs(r1["resid"])
    angle_diag = (
        r1.groupby("angle", sort=True)
        .agg(
            n=("abs_err", "size"),
            mae=("abs_err", "mean"),
            rmse=("resid", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("resid", "mean"),
            cd_true=("cd", "mean"),
            cd_pred=("cd_pred", "mean"),
            uncertainty_width=("uncertainty_width", "mean"),
        )
        .reset_index()
    )
    geom_diag = (
        r1.groupby(group_col, sort=False)
        .agg(
            n=("abs_err", "size"),
            mae=("abs_err", "mean"),
            rmse=("resid", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("resid", "mean"),
            cd_true=("cd", "mean"),
            cd_pred=("cd_pred", "mean"),
            uncertainty_width=("uncertainty_width", "mean"),
        )
        .reset_index()
        .sort_values(["mae", "rmse"], ascending=[False, False])
    )
    summary = pd.DataFrame([{
        "scope": "R1_linear",
        "tag": tag,
        "n_rows": int(len(r1)),
        "mae": float(r1["abs_err"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(r1["resid"])))),
        "bias": float(r1["resid"].mean()),
        "mean_cd_true": float(r1["cd"].mean()),
        "mean_cd_pred": float(r1["cd_pred"].mean()),
        "uncertainty_spearman_abs_err": float(r1["abs_err"].corr(r1["uncertainty_width"], method="spearman")) if len(r1) > 2 else float("nan"),
    }])

    angle_path = paths["tables"] / f"r1_angle_diagnostics_{tag}.csv"
    geom_path = paths["tables"] / f"r1_geometry_diagnostics_{tag}.csv"
    summary_path = paths["tables"] / f"r1_summary_{tag}.csv"
    report_path = paths["reports"] / f"r1_diagnostics_{tag}.md"

    angle_diag.to_csv(angle_path, index=False)
    geom_diag.to_csv(geom_path, index=False)
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(angle_diag["angle"], angle_diag["mae"], marker="o", lw=1.5)
    plt.xlabel("AoA [deg]")
    plt.ylabel("R1 MAE")
    plt.title(f"R1 MAE by AoA — {tag}")
    plt.tight_layout()
    plt.savefig(paths["figures"] / f"r1_mae_by_angle_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(angle_diag["angle"], angle_diag["bias"], marker="o", lw=1.5)
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel("AoA [deg]")
    plt.ylabel("R1 bias (pred - true)")
    plt.title(f"R1 Bias by AoA — {tag}")
    plt.tight_layout()
    plt.savefig(paths["figures"] / f"r1_bias_by_angle_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close()

    lines = [
        f"# R1 Diagnostics — {tag}",
        "",
        "## Summary",
        _to_markdown(summary.round(6)),
        "",
        "## Worst geometries",
        _to_markdown(geom_diag.head(20).round(6)),
        "",
        "## Angle diagnostics",
        _to_markdown(angle_diag.round(6)),
    ]
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding=OUTPUT_TEXT_ENCODING)

    if tag == "baseline":
        angle_diag.to_csv(paths["tables"] / "r1_angle_diagnostics.csv", index=False)
        geom_diag.to_csv(paths["tables"] / "r1_geometry_diagnostics.csv", index=False)
        summary.to_csv(paths["tables"] / "r1_summary.csv", index=False)
        (paths["reports"] / "r1_diagnostics.md").write_text(report_text, encoding=OUTPUT_TEXT_ENCODING)


def make_acceptance_report(overall: pd.DataFrame, coverage: pd.DataFrame, logo: pd.DataFrame, cfg: ResearchConfig, paths: Dict[str, Path]) -> Dict[str, Any]:
    upgraded_rmse = float(overall.loc[0, "rmse"])
    upgraded_mae = float(overall.loc[0, "mae"])
    cov90 = coverage[(coverage["scope"] == "global") & (coverage["level"] == 90)]["coverage"]
    cov90v = float(cov90.iloc[0]) if len(cov90) else float("nan")
    logo_mean = float(logo["mae"].dropna().mean()) if logo is not None and not logo.empty and "mae" in logo.columns else float("nan")
    baseline_rmse = baseline_mae = baseline_logo = float("nan")
    baseline_source = None
    if cfg.baseline_metrics_csv and Path(cfg.baseline_metrics_csv).exists():
        b = pd.read_csv(cfg.baseline_metrics_csv)
        baseline_source = cfg.baseline_metrics_csv
        # supports either columns metric/value or direct mae/rmse row
        if {"metric", "value"}.issubset(b.columns):
            vals = dict(zip(b["metric"].astype(str).str.lower(), b["value"]))
            baseline_rmse = float(vals.get("rmse", np.nan)); baseline_mae = float(vals.get("mae", np.nan)); baseline_logo = float(vals.get("logo_mae", np.nan))
        else:
            baseline_rmse = float(b["rmse"].iloc[0]) if "rmse" in b.columns else np.nan
            baseline_mae = float(b["mae"].iloc[0]) if "mae" in b.columns else np.nan
            baseline_logo = float(b["logo_mae"].iloc[0]) if "logo_mae" in b.columns else np.nan
    rmse_pass = True if not np.isfinite(baseline_rmse) else upgraded_rmse <= baseline_rmse + cfg.target_rmse_tolerance
    mae_pass = True if not np.isfinite(baseline_mae) else upgraded_mae <= baseline_mae + cfg.target_rmse_tolerance
    coverage_pass = cfg.coverage90_min <= cov90v <= cfg.coverage90_max
    logo_pass = True if not np.isfinite(baseline_logo) or not np.isfinite(logo_mean) else logo_mean <= baseline_logo + cfg.target_rmse_tolerance
    accepted = bool(rmse_pass and mae_pass and coverage_pass and logo_pass)
    report = {
        "accepted": accepted,
        "baseline_source": baseline_source,
        "baseline_rmse": baseline_rmse,
        "upgraded_rmse": upgraded_rmse,
        "rmse_pass": bool(rmse_pass),
        "baseline_mae": baseline_mae,
        "upgraded_mae": upgraded_mae,
        "mae_pass": bool(mae_pass),
        "coverage90": cov90v,
        "coverage90_pass": bool(coverage_pass),
        "baseline_logo_mae": baseline_logo,
        "upgraded_logo_mae": logo_mean,
        "logo_pass": bool(logo_pass),
    }
    (paths["reports"] / "research_acceptance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Research Upgrade Acceptance Report", "", pd.DataFrame([report]).to_markdown(index=False), ""]
    (paths["reports"] / "research_acceptance_report.md").write_text("\n".join(lines), encoding=OUTPUT_TEXT_ENCODING)
    return report


class Day5CdResearchInterface:
    """Serializable inference facade for Day5 inverse design.

    Expected input is a DataFrame already containing Day2 geometry descriptors and g_* geometry vector.
    The interface applies saved PCA, strict feature validation, Cd quantile prediction,
    per-regime calibration, conformal intervals, and OOD flags.
    """
    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle

    def _add_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = self.bundle["pca_meta"]
        g_cols = meta["g_cols"]
        missing = [c for c in g_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required geometry grid columns for PCA: {missing[:10]}")
        Xg = df[g_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        Xg = meta["imputer"].transform(Xg).astype(np.float32, copy=False)
        Xs = meta["scaler"].transform(Xg).astype(np.float32, copy=False)
        Z = meta["pca"].transform(Xs)[:, :meta["keep"]].astype(np.float32, copy=False)
        out = df.copy()
        for i, c in enumerate(meta["z_cols"]):
            out[c] = Z[:, i].astype(np.float32, copy=False)
        if "regime_id" not in out.columns:
            out["regime_id"] = regime_id_from_angle(out["angle"].to_numpy(float))
        if "log10_re" not in out.columns:
            if "reynolds" in out.columns:
                out["log10_re"] = np.log10(np.maximum(out["reynolds"].to_numpy(float), 1.0))
            else:
                out["log10_re"] = 6.0
        return out

    def _predict_physics_flags(self, x: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
        tmp = x.copy().reset_index(drop=True)
        tmp["cd_pred"] = pred["cd_p50"].to_numpy(float)
        group_col = self.bundle.get("group_col", None)
        if group_col not in tmp.columns:
            tmp["__day5_geom_group__"] = "single_geometry"
            group_col = "__day5_geom_group__"
        if "log10_re" not in tmp.columns:
            tmp["log10_re"] = 6.0
        settings = self.bundle.get("cd_physics_settings", CD_PHYSICS_SETTINGS)
        flags = np.zeros(len(tmp), dtype=bool)
        for _, idx in tmp.groupby([group_col, "log10_re"], sort=False).groups.items():
            loc = np.asarray(list(idx), dtype=int)
            grp = tmp.iloc[loc].copy()
            metrics = evaluate_cd_physics_curve(grp["angle"].to_numpy(float), grp["cd_pred"].to_numpy(float), settings)
            flags[loc] = bool(metrics["cd_physics_flag"])
        return flags

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._add_pca(df)
        model: BootstrapRegimeQuantileEnsemble = self.bundle["model"]
        micro: PerRegimeCalibration = self.bundle["micro"]
        conformal: RegimeWiseConformalCalibrator = self.bundle["conformal"]
        residual_correctors = self.bundle.get("residual_correctors")
        if residual_correctors is None:
            legacy = self.bundle.get("r1_corrector")
            residual_correctors = [legacy] if legacy is not None else []
        geom_ood: GeometryOODDetector = self.bundle["geometry_ood"]
        flow_ood: FlowOODDetector = self.bundle["flow_ood"]
        pred = model.predict(x)
        pred["cd_p50"] = micro.apply(pred["cd_p50"].to_numpy(float), x["regime_id"].to_numpy(int))
        pred["cd_p10"] = np.minimum(pred["cd_p10"], pred["cd_p50"])
        pred["cd_p90"] = np.maximum(pred["cd_p90"], pred["cd_p50"])
        pred["uncertainty_width"] = pred["cd_p90"] - pred["cd_p10"]
        pred = apply_residual_corrector_stack(residual_correctors, x, pred)
        pred = conformal.apply(x, pred, 0.90)
        gd, gf = geom_ood.score(x); fd, ff = flow_ood.score(x)
        pred["geometry_ood_score"] = gd; pred["geometry_ood_flag"] = gf
        pred["flow_ood_score"] = fd; pred["flow_ood_flag"] = ff
        pred["ood_flag"] = gf | ff
        pred["reliability_score"] = np.clip(1.0 - 0.5 * (gd / max(geom_ood.threshold, 1e-9)) - 0.5 * (fd / max(flow_ood.threshold, 1e-9)), 0.0, 1.0)
        return pred

    def predict_cd(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_full(df)["cd_p50"].to_numpy(float)

    def predict_cd_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_full(df)
        return p[["cd_p10", "cd_p50", "cd_p90", "cd_c90_lo", "cd_c90_hi", "uncertainty_width"]]

    def predict_ood(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_full(df)
        return p[["geometry_ood_score", "geometry_ood_flag", "flow_ood_score", "flow_ood_flag", "ood_flag"]]

    def predict_reliability(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_full(df)["reliability_score"].to_numpy(float)


def export_day5_artifacts(bundle: Dict[str, Any], paths: Dict[str, Path], cfg: ResearchConfig) -> None:
    iface = Day5CdResearchInterface(bundle)
    bundle_path = paths["day5"] / "day5_cd_research_interface.pkl"
    joblib.dump(iface, bundle_path)
    spec = {
        "name": "Day4C Cd Research V2 Interface",
        "bundle_path": str(bundle_path),
        "methods": ["predict_cd", "predict_cd_interval", "predict_ood", "predict_reliability", "predict_full"],
        "prediction_columns": ["cd_p10", "cd_p50", "cd_p90", "cd_c90_lo", "cd_c90_hi", "uncertainty_width", "geometry_ood_score", "geometry_ood_flag", "flow_ood_score", "flow_ood_flag", "ood_flag", "reliability_score"],
        "required_input": "Day2-compatible DataFrame with g_* geometry grid, angle, reynolds/log10_re, and geometry descriptors.",
        "strict_feature_validation": True,
        "config": asdict(cfg),
    }
    (paths["day5"] / "day5_interface.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def write_final_report(paths: Dict[str, Path], overall: pd.DataFrame, regime: pd.DataFrame, logo: pd.DataFrame, coverage: pd.DataFrame, uq_corr: Dict[str, float], geom_ood_rate: float, flow_ood_rate: float, failure: pd.DataFrame, acceptance: Dict[str, Any], cfg: ResearchConfig) -> None:
    cov90 = coverage[(coverage["scope"] == "global") & (coverage["level"] == 90)]["coverage"]
    cov90v = float(cov90.iloc[0]) if len(cov90) else float("nan")
    lines = [
        "# Day4C Cd Surrogate — Publication Research Upgrade V2 Final", "",
        "## 1. Overall metrics", "", overall.round(6).to_markdown(index=False), "",
        "## 2. Regime metrics", "", regime.round(6).to_markdown(index=False), "",
        "## 3. Geometry refit validation", "",
        logo.sort_values("mae").head(10).round(6).to_markdown(index=False) if logo is not None and not logo.empty and "mae" in logo.columns else "Removed for runtime; no per-geometry refits are executed.", "",
        "## 4. Geometry extrapolation analysis", "",
        "Per-geometry refit validation was removed for runtime. Use OOD, physics, uncertainty, and regime metrics for fast iteration.", "",
        "## 5. OOD analysis", "",
        f"- Geometry OOD: PCA-latent Mahalanobis, flag rate = {geom_ood_rate:.2%}",
        f"- Flow OOD: AoA/Re/regime Mahalanobis, flag rate = {flow_ood_rate:.2%}", "",
        "## 6. Calibration analysis", "",
        "Per-regime micro calibration is fitted on the calibration split only. No final-test fitting is used.", "",
        "## 7. Conformal coverage", "", coverage.round(4).to_markdown(index=False), "",
        f"- 90% global coverage: {cov90v:.3f}", "",
        "## 8. Uncertainty quality", "",
        f"- Pearson(width, abs_error): {uq_corr.get('pearson', float('nan')):.4f}",
        f"- Spearman(width, abs_error): {uq_corr.get('spearman', float('nan')):.4f}", "",
        "## 9. Failure cases", "", failure.head(20).round(6).to_markdown(index=False), "",
        "## 10. Research acceptance", "", pd.DataFrame([acceptance]).to_markdown(index=False), "",
        "## 11. Day5 readiness", "",
        "Exports `day5_interface/day5_interface.json` and `day5_interface/day5_cd_research_interface.pkl` with `predict_cd`, `predict_cd_interval`, `predict_ood`, and `predict_reliability`.", "",
        "## 12. Recommendations", "",
        "1. Keep geometry refit validation disabled during deadline-sensitive training.",
        "2. Reject or downweight Day5 candidates with either geometry or flow OOD flags.",
        "3. Use conformal intervals, not p10/p90 alone, for safety constraints.",
        "4. Inspect OOD/physics/regime metrics before adding new airfoil families.",
    ]
    txt = "\n".join(lines)
    (paths["root"] / "day4c_publication_report.md").write_text(txt, encoding=OUTPUT_TEXT_ENCODING)
    (paths["reports"] / "day4c_publication_report.md").write_text(txt, encoding=OUTPUT_TEXT_ENCODING)



# =============================================================================
# Day4C CD V3 Final additions: Trust Region, Physics, UQ Gate, Weighted Candidate,
# Champion Selection, Day5 Safety Interface
# =============================================================================

STALL_WEIGHTS = {1: 1.0, 2: 1.5, 3: 2.0, 4: 4.0, 5: 4.0, 6: 3.0}
CD_PHYSICS_SETTINGS = {
    "cd_floor": CD_FLOOR,
    "high_aoa_start_deg": 12.0,
    "attached_aoa_max_deg": 5.0,
    "post_stall_start_deg": 12.0,
    "min_curve_points": 4,
    "min_high_points": 3,
    "high_drop_fraction": 0.25,
    "stall_slope_threshold": -0.01,
    "branch_drop_fraction": 0.20,
    "branch_slope_threshold": -0.006,
    "collapse_ratio": 0.50,
    "oscillation_threshold": 1.50,
    "curve_violation_rate_threshold": 0.40,
}


def compute_cd_training_weights(df: pd.DataFrame) -> np.ndarray:
    """Hybrid weighting: keep stall emphasis, but rescue attached low-drag fidelity.

    The previous branch split did not improve R1. This weighting keeps the model
    structure stable and instead nudges the weighted candidate toward:
    - low-drag attached flow in R1
    - the edge of attached-flow buckets where residuals were largest
    - still preserving stronger weights in stall / post-stall regimes
    """
    rid = df["regime_id"].to_numpy(dtype=int)
    cd = df["cd"].to_numpy(dtype=np.float32, copy=False)
    abs_angle = np.abs(df["angle"].to_numpy(dtype=np.float32, copy=False))

    w = np.array([STALL_WEIGHTS.get(int(r), 1.0) for r in rid], dtype=np.float32)

    r1 = rid == 1
    if np.any(r1):
        low_drag_scale = np.ones(len(df), dtype=np.float32)
        low_drag_scale[cd <= 0.010] = 2.80
        low_drag_scale[(cd > 0.010) & (cd <= 0.020)] = 2.20
        low_drag_scale[(cd > 0.020) & (cd <= 0.030)] = 1.70
        low_drag_scale[(cd > 0.030) & (cd <= 0.050)] = 1.30
        edge_scale = 1.0 + 0.35 * np.clip(abs_angle / 5.0, 0.0, 1.0)
        w[r1] = w[r1] * low_drag_scale[r1] * edge_scale[r1]

    return w.astype(np.float32, copy=False)


def _to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def _resolve_cd_physics_settings(settings: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    out = dict(CD_PHYSICS_SETTINGS)
    if settings:
        out.update(settings)
    return out


def _evaluate_cd_branch(angle: np.ndarray, cd: np.ndarray, settings: Dict[str, float], positive: bool) -> Dict[str, float]:
    if positive:
        mask = angle >= 0.0
        branch_aoa = angle[mask]
    else:
        mask = angle <= 0.0
        branch_aoa = np.abs(angle[mask])
    branch_cd = cd[mask]
    finite = np.isfinite(branch_aoa) & np.isfinite(branch_cd)
    branch_aoa = branch_aoa[finite]
    branch_cd = branch_cd[finite]
    out = {
        "high_drop_count": 0.0,
        "high_drop_rate": 0.0,
        "stall_monotonicity_violation_count": 0.0,
        "stall_monotonicity_violation_rate": 0.0,
        "post_stall_cd_collapse_count": 0.0,
        "cd_curve_oscillation_score": 0.0,
        "violation_rate": 0.0,
    }
    if len(branch_aoa) < int(settings["min_curve_points"]):
        return out

    order = np.argsort(branch_aoa)
    branch_aoa = branch_aoa[order]
    branch_cd = branch_cd[order]
    high = branch_aoa >= float(settings["high_aoa_start_deg"])
    if int(high.sum()) >= int(settings["min_high_points"]):
        cd_high = branch_cd[high]
        aoa_high = branch_aoa[high]
        dcd = np.diff(cd_high)
        if len(dcd):
            daoa = np.maximum(np.diff(aoa_high), 1e-9)
            slope = dcd / daoa
            local_scale = max(float(np.nanmedian(np.abs(cd_high))), float(settings["cd_floor"]))
            out["high_drop_count"] = float(np.sum(dcd < -float(settings["branch_drop_fraction"]) * local_scale))
            out["stall_monotonicity_violation_count"] = float(np.sum(slope < float(settings["branch_slope_threshold"])))
            denom = max(1, len(dcd))
            out["high_drop_rate"] = float(out["high_drop_count"] / denom)
            out["stall_monotonicity_violation_rate"] = float(out["stall_monotonicity_violation_count"] / denom)

    attached = branch_cd[branch_aoa <= float(settings["attached_aoa_max_deg"])]
    post = branch_cd[branch_aoa >= float(settings["post_stall_start_deg"])]
    if len(attached) and len(post):
        out["post_stall_cd_collapse_count"] = float(np.nanmin(post) < float(settings["collapse_ratio"]) * np.nanmedian(attached))

    if len(branch_cd) >= 5:
        d2 = np.diff(branch_cd, n=2)
        out["cd_curve_oscillation_score"] = float(np.nanmean(np.abs(d2)) / max(np.nanmedian(np.abs(branch_cd)), float(settings["cd_floor"])))

    oscillation_rate = min(1.0, float(out["cd_curve_oscillation_score"]) / max(float(settings["oscillation_threshold"]), 1e-9))
    out["violation_rate"] = float(max(
        out["high_drop_rate"],
        out["stall_monotonicity_violation_rate"],
        1.0 if out["post_stall_cd_collapse_count"] > 0 else 0.0,
        oscillation_rate,
    ))
    return out


def evaluate_cd_physics_curve(angle: Any, cd: Any, settings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    cfg = _resolve_cd_physics_settings(settings)
    aoa = np.asarray(angle, dtype=float)
    cdv = np.asarray(cd, dtype=float)
    finite = np.isfinite(aoa) & np.isfinite(cdv)
    aoa = aoa[finite]
    cdv = cdv[finite]

    result = {
        "negative_cd_count": 0,
        "high_aoa_cd_drop_count": 0,
        "stall_monotonicity_violation_count": 0,
        "high_drop_rate": 0.0,
        "stall_monotonicity_violation_rate": 0.0,
        "positive_branch_violation_rate": 0.0,
        "negative_branch_violation_rate": 0.0,
        "post_stall_cd_collapse_count": 0,
        "cd_curve_oscillation_score": 0.0,
        "physics_violation_count": 0,
        "violation_rate": 0.0,
        "cd_physics_flag": False,
    }
    if len(aoa) == 0:
        return result

    result["negative_cd_count"] = int(np.sum(cdv <= 0.0))
    pos = _evaluate_cd_branch(aoa, cdv, cfg, positive=True)
    neg = _evaluate_cd_branch(aoa, cdv, cfg, positive=False)
    result["high_aoa_cd_drop_count"] = int(pos["high_drop_count"] + neg["high_drop_count"])
    result["stall_monotonicity_violation_count"] = int(pos["stall_monotonicity_violation_count"] + neg["stall_monotonicity_violation_count"])
    result["high_drop_rate"] = float(max(pos["high_drop_rate"], neg["high_drop_rate"]))
    result["stall_monotonicity_violation_rate"] = float(max(pos["stall_monotonicity_violation_rate"], neg["stall_monotonicity_violation_rate"]))
    result["positive_branch_violation_rate"] = float(pos["violation_rate"])
    result["negative_branch_violation_rate"] = float(neg["violation_rate"])
    result["post_stall_cd_collapse_count"] = int(pos["post_stall_cd_collapse_count"] + neg["post_stall_cd_collapse_count"])
    result["cd_curve_oscillation_score"] = float(max(pos["cd_curve_oscillation_score"], neg["cd_curve_oscillation_score"]))

    oscillation_rate = min(1.0, float(result["cd_curve_oscillation_score"]) / max(float(cfg["oscillation_threshold"]), 1e-9))
    result["physics_violation_count"] = int(
        result["negative_cd_count"]
        + result["high_aoa_cd_drop_count"]
        + result["stall_monotonicity_violation_count"]
        + result["post_stall_cd_collapse_count"]
        + int(result["cd_curve_oscillation_score"] > float(cfg["oscillation_threshold"]))
        + int(result["positive_branch_violation_rate"] > float(cfg["curve_violation_rate_threshold"]))
        + int(result["negative_branch_violation_rate"] > float(cfg["curve_violation_rate_threshold"]))
    )
    result["violation_rate"] = float(max(
        1.0 if result["negative_cd_count"] > 0 else 0.0,
        result["positive_branch_violation_rate"],
        result["negative_branch_violation_rate"],
        1.0 if result["post_stall_cd_collapse_count"] > 0 else 0.0,
        oscillation_rate,
    ))
    result["cd_physics_flag"] = bool(
        result["negative_cd_count"] > 0
        or result["post_stall_cd_collapse_count"] > 0
        or result["violation_rate"] > float(cfg["curve_violation_rate_threshold"])
    )
    return result


class TrustRegionManager:
    def __init__(self, geometry_threshold: float, flow_threshold: float):
        self.geometry_threshold = float(max(geometry_threshold, 1e-12))
        self.flow_threshold = float(max(flow_threshold, 1e-12))

    def evaluate(self, geometry_score: np.ndarray, flow_score: np.ndarray) -> pd.DataFrame:
        g = np.asarray(geometry_score, dtype=float)
        f = np.asarray(flow_score, dtype=float)
        gn = g / self.geometry_threshold
        fn = f / self.flow_threshold
        trust = 0.5 * gn + 0.5 * fn
        mx = np.maximum(gn, fn)
        klass = np.where((gn >= 1.0) | (fn >= 1.0), "UNSAFE", np.where(mx >= 0.90, "WARNING", "SAFE"))
        return pd.DataFrame({
            "geometry_norm": gn,
            "flow_norm": fn,
            "trust_region_score": trust,
            "trust_region_class": klass,
        })

    def report(self, pred: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
        counts = pred["trust_region_class"].value_counts(dropna=False).rename_axis("trust_region_class").reset_index(name="n")
        counts["rate"] = counts["n"] / max(1, len(pred))
        rows = [{
            "geometry_threshold": self.geometry_threshold,
            "flow_threshold": self.flow_threshold,
            "mean_trust_region_score": float(pred["trust_region_score"].mean()),
            "safe_rate": float((pred["trust_region_class"] == "SAFE").mean()),
            "warning_rate": float((pred["trust_region_class"] == "WARNING").mean()),
            "unsafe_rate": float((pred["trust_region_class"] == "UNSAFE").mean()),
        }]
        metrics = pd.DataFrame(rows)
        metrics.to_csv(paths["tables"] / "trust_region_metrics.csv", index=False)
        lines = ["# Trust Region Report", "", "## Summary", _to_markdown(metrics), "", "## Class distribution", _to_markdown(counts)]
        (paths["reports"] / "trust_region_report.md").write_text("\n".join(lines), encoding=OUTPUT_TEXT_ENCODING)
        return metrics


def cd_physics_consistency(test_df: pd.DataFrame, pred: pd.DataFrame, group_col: str, paths: Dict[str, Path], tag: str = "baseline") -> pd.DataFrame:
    tmp = test_df[[group_col, "angle", "cd", "regime_id"]].copy().reset_index(drop=True)
    if "log10_re" in test_df.columns:
        tmp["log10_re"] = test_df["log10_re"].to_numpy(float)
    else:
        tmp["log10_re"] = 6.0
    tmp["cd_pred"] = pred["cd_p50"].to_numpy(float)
    rows = []
    group_cols = [group_col, "log10_re"] if "log10_re" in tmp.columns else [group_col]
    for keys, grp in tmp.groupby(group_cols, sort=False):
        gs = grp.assign(abs_angle=np.abs(grp["angle"].to_numpy(float))).sort_values("abs_angle")
        true_metrics = evaluate_cd_physics_curve(gs["angle"].to_numpy(float), gs["cd"].to_numpy(float), CD_PHYSICS_SETTINGS)
        pred_metrics = evaluate_cd_physics_curve(gs["angle"].to_numpy(float), gs["cd_pred"].to_numpy(float), CD_PHYSICS_SETTINGS)
        if isinstance(keys, tuple):
            geom_key = keys[0]; re_key = keys[1]
        else:
            geom_key = keys; re_key = float("nan")
        row = {
            group_col: geom_key,
            "log10_re": re_key,
            "n_points": int(len(gs)),
        }
        for prefix, metrics in (("true", true_metrics), ("pred", pred_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        row.update({
            "negative_cd_count": int(pred_metrics["negative_cd_count"]),
            "high_aoa_cd_drop_count": int(pred_metrics["high_aoa_cd_drop_count"]),
            "stall_monotonicity_violation_count": int(pred_metrics["stall_monotonicity_violation_count"]),
            "high_drop_rate": float(pred_metrics["high_drop_rate"]),
            "stall_monotonicity_violation_rate": float(pred_metrics["stall_monotonicity_violation_rate"]),
            "positive_branch_violation_rate": float(pred_metrics["positive_branch_violation_rate"]),
            "negative_branch_violation_rate": float(pred_metrics["negative_branch_violation_rate"]),
            "post_stall_cd_collapse_count": int(pred_metrics["post_stall_cd_collapse_count"]),
            "cd_curve_oscillation_score": float(pred_metrics["cd_curve_oscillation_score"]),
            "physics_violation_count": int(pred_metrics["physics_violation_count"]),
            "violation_rate": float(pred_metrics["violation_rate"]),
            "true_cd_physics_flag": bool(true_metrics["cd_physics_flag"]),
            "pred_cd_physics_flag": bool(pred_metrics["cd_physics_flag"]),
            "true_violation_rate": float(true_metrics["violation_rate"]),
            "pred_violation_rate": float(pred_metrics["violation_rate"]),
            "cd_physics_flag": bool(pred_metrics["cd_physics_flag"]),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            group_col, "log10_re", "n_points",
            "true_cd_physics_flag", "pred_cd_physics_flag",
            "true_violation_rate", "pred_violation_rate",
            "negative_cd_count", "high_aoa_cd_drop_count", "stall_monotonicity_violation_count",
            "high_drop_rate", "stall_monotonicity_violation_rate",
            "positive_branch_violation_rate", "negative_branch_violation_rate",
            "post_stall_cd_collapse_count", "cd_curve_oscillation_score",
            "physics_violation_count", "violation_rate", "cd_physics_flag",
        ])
    metrics_path = paths["tables"] / f"cd_physics_metrics_{tag}.csv"
    summary_path = paths["tables"] / f"cd_physics_summary_{tag}.csv"
    report_path = paths["reports"] / f"cd_physics_report_{tag}.md"
    out.to_csv(metrics_path, index=False)
    pred_only = int((~out["true_cd_physics_flag"] & out["pred_cd_physics_flag"]).sum()) if len(out) else 0
    true_only = int((out["true_cd_physics_flag"] & ~out["pred_cd_physics_flag"]).sum()) if len(out) else 0
    both_violate = int((out["true_cd_physics_flag"] & out["pred_cd_physics_flag"]).sum()) if len(out) else 0
    both_clean = int((~out["true_cd_physics_flag"] & ~out["pred_cd_physics_flag"]).sum()) if len(out) else 0
    summary = pd.DataFrame([{
        "n_curves": int(len(out)),
        "true_violating_curves": int(out["true_cd_physics_flag"].sum()) if len(out) else 0,
        "pred_violating_curves": int(out["pred_cd_physics_flag"].sum()) if len(out) else 0,
        "violating_curves": int(out["pred_cd_physics_flag"].sum()) if len(out) else 0,
        "true_flag_rate": float(out["true_cd_physics_flag"].mean()) if len(out) else 0.0,
        "pred_flag_rate": float(out["pred_cd_physics_flag"].mean()) if len(out) else 0.0,
        "true_violation_rate": float(out["true_violation_rate"].mean()) if len(out) else 0.0,
        "pred_violation_rate": float(out["pred_violation_rate"].mean()) if len(out) else 0.0,
        "violation_rate": float(out["pred_violation_rate"].mean()) if len(out) else 0.0,
        "true_negative_cd_total": int(out["true_negative_cd_count"].sum()) if len(out) and "true_negative_cd_count" in out.columns else 0,
        "pred_negative_cd_total": int(out["pred_negative_cd_count"].sum()) if len(out) and "pred_negative_cd_count" in out.columns else 0,
        "negative_cd_total": int(out["pred_negative_cd_count"].sum()) if len(out) and "pred_negative_cd_count" in out.columns else 0,
        "true_high_aoa_cd_drop_total": int(out["true_high_aoa_cd_drop_count"].sum()) if len(out) and "true_high_aoa_cd_drop_count" in out.columns else 0,
        "pred_high_aoa_cd_drop_total": int(out["pred_high_aoa_cd_drop_count"].sum()) if len(out) and "pred_high_aoa_cd_drop_count" in out.columns else 0,
        "true_stall_monotonicity_violation_total": int(out["true_stall_monotonicity_violation_count"].sum()) if len(out) and "true_stall_monotonicity_violation_count" in out.columns else 0,
        "pred_stall_monotonicity_violation_total": int(out["pred_stall_monotonicity_violation_count"].sum()) if len(out) and "pred_stall_monotonicity_violation_count" in out.columns else 0,
        "true_post_stall_collapse_total": int(out["true_post_stall_cd_collapse_count"].sum()) if len(out) and "true_post_stall_cd_collapse_count" in out.columns else 0,
        "pred_post_stall_collapse_total": int(out["pred_post_stall_cd_collapse_count"].sum()) if len(out) and "pred_post_stall_cd_collapse_count" in out.columns else 0,
        "post_stall_collapse_total": int(out["pred_post_stall_cd_collapse_count"].sum()) if len(out) and "pred_post_stall_cd_collapse_count" in out.columns else 0,
    }])
    summary.to_csv(summary_path, index=False)
    confusion = pd.DataFrame([{
        "both_clean": both_clean,
        "true_only_violations": true_only,
        "pred_only_violations": pred_only,
        "both_violate": both_violate,
    }])
    lines = ["# Cd Physics Consistency Report", "", "## Summary", _to_markdown(summary), "", "## True vs Pred flag breakdown", _to_markdown(confusion), ""]
    pred_violation_rate_mean = float(summary.loc[0, "pred_violation_rate"]) if len(summary) else 0.0
    if pred_violation_rate_mean > 0.30:
        if pred_only > both_violate and pred_only >= true_only:
            cause = "Predicted Cd curves violate physics more often on otherwise clean ground-truth curves, so the main issue is model-induced curve shape drift."
        elif both_violate >= pred_only:
            cause = "A large share of predicted violations overlaps with already non-physical true Cd curves, so source-data physics issues are a major contributor."
        else:
            cause = "Predicted violations are mixed: part comes from non-physical source curves and part from model-induced degradation."
        detail = pd.DataFrame([{
            "pred_only_rate": float(pred_only / max(1, len(out))),
            "true_only_rate": float(true_only / max(1, len(out))),
            "both_violate_rate": float(both_violate / max(1, len(out))),
            "both_clean_rate": float(both_clean / max(1, len(out))),
        }])
        lines.extend(["## Diagnostic explanation", cause, "", _to_markdown(detail), ""])
    lines.extend(["## Worst curves", _to_markdown(out.sort_values(["pred_violation_rate", "physics_violation_count"], ascending=False).head(20)) if len(out) else "No curves evaluated."])
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding=OUTPUT_TEXT_ENCODING)
    if tag == "baseline":
        out.to_csv(paths["tables"] / "cd_physics_metrics.csv", index=False)
        summary.to_csv(paths["tables"] / "cd_physics_summary.csv", index=False)
        (paths["reports"] / "cd_physics_report.md").write_text(report_text, encoding=OUTPUT_TEXT_ENCODING)
    return out


def attach_physics_flags_to_rows(test_df: pd.DataFrame, pred: pd.DataFrame, physics: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = pred.copy()
    if physics.empty:
        out["cd_physics_flag"] = False
        return out
    key_cols = [group_col]
    if "log10_re" in test_df.columns and "log10_re" in physics.columns:
        key_cols = [group_col, "log10_re"]
    tmp = test_df[key_cols].copy().reset_index(drop=True)
    flag_col = "pred_cd_physics_flag" if "pred_cd_physics_flag" in physics.columns else "cd_physics_flag"
    merged = tmp.merge(physics[key_cols + [flag_col]], on=key_cols, how="left")
    out["cd_physics_flag"] = merged[flag_col].fillna(False).astype(bool).to_numpy()
    return out


def uncertainty_quality_gate(test_df: pd.DataFrame, pred: pd.DataFrame, coverage: pd.DataFrame, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, str]:
    y = test_df["cd"].to_numpy(float)
    p = pred["cd_p50"].to_numpy(float)
    err = np.abs(y - p)
    width = pred["uncertainty_width"].to_numpy(float)
    pear = float(pd.Series(width).corr(pd.Series(err), method="pearson")) if len(err) > 2 else float("nan")
    spear = float(pd.Series(width).corr(pd.Series(err), method="spearman")) if len(err) > 2 else float("nan")
    try:
        slope = float(LinearRegression().fit(width.reshape(-1, 1), err).coef_[0])
    except Exception:
        slope = float("nan")
    def cov(level: int) -> float:
        row = coverage[(coverage["scope"] == "global") & (coverage["level"] == level)]
        return float(row["coverage"].iloc[0]) if len(row) else float("nan")
    c80, c90, c95 = cov(80), cov(90), cov(95)
    mpiw90 = float(np.mean(pred["cd_c90_hi"].to_numpy(float) - pred["cd_c90_lo"].to_numpy(float))) if {"cd_c90_hi", "cd_c90_lo"}.issubset(pred.columns) else float("nan")
    passed = bool(np.isfinite(pear) and pear > 0.20 and 0.88 <= c90 <= 0.92 and 0.93 <= c95 <= 0.97)
    status = "PASS" if passed else "UNCERTAINTY_NOT_RELIABLE"
    metrics = pd.DataFrame([{
        "pearson_uq_error": pear,
        "spearman_uq_error": spear,
        "calibration_slope": slope,
        "coverage80": c80,
        "coverage90": c90,
        "coverage95": c95,
        "mpiw90": mpiw90,
        "uncertainty_quality": status,
    }])
    metrics.to_csv(paths["tables"] / "uncertainty_quality_metrics.csv", index=False)
    lines = ["# Uncertainty Quality Report", "", _to_markdown(metrics), "", "Acceptance: PASS requires Pearson > 0.20, Coverage90 in [0.88, 0.92], and Coverage95 in [0.93, 0.97]."]
    (paths["reports"] / "uncertainty_quality_report.md").write_text("\n".join(lines), encoding=OUTPUT_TEXT_ENCODING)
    return metrics, status


class StallWeightedBootstrapRegimeQuantileEnsemble(BootstrapRegimeQuantileEnsemble):
    def fit(self, df: pd.DataFrame, report_path: Path) -> "StallWeightedBootstrapRegimeQuantileEnsemble":
        self.feature_validator = StrictFeatureValidator(self.feature_cols, report_path)
        X_all = self.feature_validator.validate(df)
        y_log = np.log(np.maximum(df["cd"].to_numpy(dtype=np.float32), CD_FLOOR)).astype(np.float32, copy=False)
        rid_arr = df["regime_id"].to_numpy(dtype=int)
        w_all = compute_cd_training_weights(df)
        fit_n_jobs = limit_parallel_jobs_for_memory(
            self.cfg.n_jobs,
            dataframe_nbytes(X_all) + int(y_log.nbytes) + int(w_all.nbytes),
            self.cfg.training_mem_multiplier,
            self.cfg,
            "CD:weighted_bootstrap_fit",
        )
        fit_cfg = replace(self.cfg, n_jobs=fit_n_jobs)
        self.global_models = self._fit_weighted_bootstrap_triplets(X_all, y_log, w_all, self.cfg.random_state + 77, fit_cfg)
        if self.cfg.regime_models_enabled:
            for rid in REGIME_IDS:
                mask = rid_arr == rid
                if int(mask.sum()) < 40:
                    continue
                self.models[rid] = self._fit_weighted_bootstrap_triplets(X_all.loc[mask], y_log[mask], w_all[mask], self.cfg.random_state + 101 * rid, fit_cfg)
        X_all = None
        y_log = None
        rid_arr = None
        w_all = None
        fit_cfg = None
        gc.collect()
        return self

    def _fit_weighted_bootstrap_triplets(self, X: pd.DataFrame, y_log: np.ndarray, w: np.ndarray, seed: int, fit_cfg: ResearchConfig) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        n = len(X)
        k = max(1, int(round(n * fit_cfg.bootstrap_fraction)))
        bag_count = max(1, int(fit_cfg.bootstrap_models))
        prob = np.asarray(w, dtype=float)
        prob = prob / max(prob.sum(), 1e-12)
        triplets = []
        for b in range(bag_count):
            idx = rng.choice(n, size=k, replace=True, p=prob) if bag_count > 1 else np.arange(n)
            Xb = X.iloc[idx]
            yb = y_log[idx]
            wb = np.asarray(w, dtype=float)[idx]
            triplet = {}
            for q in fit_cfg.quantiles:
                key = f"q{int(round(q * 100)):02d}"
                model = make_median_model(fit_cfg, seed + b * 1009) if abs(q - 0.5) < 1e-9 else make_quantile_model(q, fit_cfg, seed + b * 1009)
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
                try:
                    pipe.fit(Xb, yb, model__sample_weight=wb)
                except Exception:
                    pipe.fit(Xb, yb)
                triplet[key] = pipe
            triplets.append(triplet)
            idx = None
            Xb = None
            yb = None
            wb = None
            gc.collect()
        return triplets


def fit_predict_pipeline_v3(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, group_col: str, cfg: ResearchConfig, paths: Optional[Dict[str, Path]] = None, weighted: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    all_apply = pd.concat([train, cal, test], ignore_index=True)
    _, all_z, pca_meta = build_pca_latents(train, all_apply, cfg)
    train2 = all_z.iloc[:len(train)].reset_index(drop=True)
    cal2 = all_z.iloc[len(train):len(train) + len(cal)].reset_index(drop=True)
    test2 = all_z.iloc[len(train) + len(cal):].reset_index(drop=True)
    z_cols = pca_meta["z_cols"]
    feature_cols = select_feature_columns(train2, z_cols, cfg)
    report_path = (paths["models"] / "missing_feature_report.txt") if paths else Path("missing_feature_report.txt")
    klass = StallWeightedBootstrapRegimeQuantileEnsemble if weighted else BootstrapRegimeQuantileEnsemble
    model = klass(feature_cols, cfg).fit(train2, report_path)
    pred_cal = model.predict(cal2)
    cal_method = "isotonic" if cfg.use_isotonic_calibration else "linear"
    micro = PerRegimeCalibration(method=cal_method).fit(cal2["cd"].to_numpy(float), pred_cal["cd_p50"].to_numpy(float), cal2["regime_id"].to_numpy(int))
    pred_cal["cd_p50"] = micro.apply(pred_cal["cd_p50"].to_numpy(float), cal2["regime_id"].to_numpy(int))
    pred_cal["cd_p10"] = np.minimum(pred_cal["cd_p10"], pred_cal["cd_p50"])
    pred_cal["cd_p90"] = np.maximum(pred_cal["cd_p90"], pred_cal["cd_p50"])
    pred_cal["uncertainty_width"] = pred_cal["cd_p90"] - pred_cal["cd_p10"]
    cal_fit_idx, cal_conf_idx = split_calibration_for_r1_corrector(cal2, group_col, cfg.random_state + (17 if weighted else 0))
    residual_correctors: List[Any] = [
        TargetedResidualCorrector(
            name="attached_r12",
            feature_cols=feature_cols,
            random_state=cfg.random_state + (17 if weighted else 0),
            target_regimes={1, 2},
            min_rows=220,
            learning_rate=0.045,
            max_iter=260,
            max_leaf_nodes=63,
            min_samples_leaf=16,
            l2_regularization=0.015,
            clip_floor=0.001,
            clip_quantile=0.985,
        ),
        MonotoneSliceShiftCorrector(
            name="high_cd_tail",
            random_state=cfg.random_state + 101 + (17 if weighted else 0),
            target_regimes={6},
            angle_min=18.0,
            true_cd_min=0.24,
            pred_cd_min=0.22,
            min_rows=60,
            positive_only_shift=True,
            min_shift_quantile=0.55,
        ),
    ]
    if cal_fit_idx is not None:
        cal_fit_df = cal2.iloc[cal_fit_idx].reset_index(drop=True)
        pred_fit_df = pred_cal.iloc[cal_fit_idx].reset_index(drop=True)
    else:
        cal_fit_df = cal2
        pred_fit_df = pred_cal
    for corr in residual_correctors:
        corr.fit(cal_fit_df, pred_fit_df)
        pred_fit_df = corr.apply(cal_fit_df, pred_fit_df)
    pred_cal = apply_residual_corrector_stack(residual_correctors, cal2, pred_cal)
    if cal_conf_idx is not None:
        conformal = RegimeWiseConformalCalibrator().fit(
            cal2.iloc[cal_conf_idx].reset_index(drop=True),
            pred_cal.iloc[cal_conf_idx].reset_index(drop=True),
        )
    else:
        conformal = RegimeWiseConformalCalibrator().fit(cal2, pred_cal)
    pred_test = model.predict(test2)
    pred_test["cd_p50"] = micro.apply(pred_test["cd_p50"].to_numpy(float), test2["regime_id"].to_numpy(int))
    pred_test["cd_p10"] = np.minimum(pred_test["cd_p10"], pred_test["cd_p50"])
    pred_test["cd_p90"] = np.maximum(pred_test["cd_p90"], pred_test["cd_p50"])
    pred_test["uncertainty_width"] = pred_test["cd_p90"] - pred_test["cd_p10"]
    pred_test = apply_residual_corrector_stack(residual_correctors, test2, pred_test)
    pred_cal = downcast_float64_to_float32(pred_cal)
    pred_test = downcast_float64_to_float32(pred_test)
    if paths is not None:
        tag = "stall_weighted" if weighted else "baseline"
        corr_summary = pd.DataFrame([corr.summary_row() for corr in residual_correctors])
        corr_summary.to_csv(paths["tables"] / f"residual_corrector_summary_{tag}.csv", index=False)
        corr_summary.head(1).to_csv(paths["tables"] / f"r1_corrector_summary_{tag}.csv", index=False)
        corr_summary = None
    all_apply = None
    all_z = None
    pred_cal = None
    cal_fit_df = None
    pred_fit_df = None
    cal_fit_idx = None
    cal_conf_idx = None
    gc.collect()
    meta = {"pca_meta": pca_meta, "z_cols": z_cols, "feature_cols": feature_cols, "model": model, "micro": micro, "conformal": conformal, "residual_correctors": residual_correctors, "r1_corrector": residual_correctors[0], "train_z": train2, "cal_z": cal2, "test_z": test2, "weighted": weighted}
    return pred_test, meta


def summarize_candidate(name: str, test2: pd.DataFrame, pred: pd.DataFrame, coverage: pd.DataFrame, logo_mae: float, physics_violation_count: int, uncertainty_status: str) -> Dict[str, Any]:
    y = test2["cd"].to_numpy(float); p = pred["cd_p50"].to_numpy(float)
    rid = test2["regime_id"].to_numpy(int)
    r1_mask = rid == 1
    r2_mask = rid == 2
    stall_mask = rid >= 4
    near_mask = rid == 5
    post_mask = rid == 6
    r1_low_drag_mask = r1_mask & (y <= 0.03)
    high_cd_tail_mask = post_mask & (y >= 0.27)
    abs_err = np.abs(y - p)
    resid = p - y
    def _mae(mask): return float(np.mean(abs_err[mask])) if int(mask.sum()) else float("nan")
    def _rmse(mask): return float(np.sqrt(np.mean(np.square(resid[mask])))) if int(mask.sum()) else float("nan")
    def _bias(mask): return float(np.mean(resid[mask])) if int(mask.sum()) else float("nan")
    width = pred["uncertainty_width"].to_numpy(float)
    uq_pear = float(pd.Series(width).corr(pd.Series(abs_err), method="pearson")) if len(abs_err) > 2 else float("nan")
    row90 = coverage[(coverage["scope"] == "global") & (coverage["level"] == 90)]
    cov90 = float(row90["coverage"].iloc[0]) if len(row90) else float("nan")
    return {
        "candidate": name,
        "overall_mae": float(mean_absolute_error(y, p)),
        "overall_rmse": rmse(y, p),
        "r2": safe_r2(y, p),
        "logo_mae": float(logo_mae) if np.isfinite(logo_mae) else float("nan"),
        "r1_mae": _mae(r1_mask),
        "r1_rmse": _rmse(r1_mask),
        "r1_bias": _bias(r1_mask),
        "r1_low_drag_mae": _mae(r1_low_drag_mask),
        "r2_mae": _mae(r2_mask),
        "r2_rmse": _rmse(r2_mask),
        "r2_bias": _bias(r2_mask),
        "stall_region_mae": _mae(stall_mask),
        "near_stall_mae": _mae(near_mask),
        "post_stall_mae": _mae(post_mask),
        "high_cd_tail_mae": _mae(high_cd_tail_mask),
        "high_cd_tail_rmse": _rmse(high_cd_tail_mask),
        "high_cd_tail_bias": _bias(high_cd_tail_mask),
        "coverage90": cov90,
        "uq_pearson": uq_pear,
        "uncertainty_quality": uncertainty_status,
        "physics_violation_count": int(physics_violation_count),
    }


def summarize_geometry_cv_candidate(name: str, cv_result: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Build selection metrics from out-of-fold geometry predictions."""
    pred = cv_result.get("predictions", pd.DataFrame())
    if pred.empty or "cd_p50" not in pred.columns:
        return {
            "candidate": name,
            "overall_mae": np.inf,
            "overall_rmse": np.inf,
            "r2": -np.inf,
            "logo_mae": np.nan,
            "r1_mae": np.inf,
            "r1_rmse": np.inf,
            "r1_bias": np.nan,
            "r1_low_drag_mae": np.inf,
            "r2_mae": np.inf,
            "r2_rmse": np.inf,
            "r2_bias": np.nan,
            "stall_region_mae": np.inf,
            "near_stall_mae": np.inf,
            "post_stall_mae": np.inf,
            "high_cd_tail_mae": np.inf,
            "high_cd_tail_rmse": np.inf,
            "high_cd_tail_bias": np.nan,
            "coverage90": np.nan,
            "uq_pearson": np.nan,
            "uncertainty_quality": "CV_FAILED",
            "physics_violation_count": 10**9,
            "selection_source": "geometry_groupkfold_oof",
        }

    y = pred["cd"].to_numpy(float)
    p = pred["cd_p50"].to_numpy(float)
    rid = pred["regime_id"].to_numpy(int)
    abs_err = np.abs(y - p)
    resid = p - y

    def _mae(mask: np.ndarray) -> float:
        return float(np.mean(abs_err[mask])) if int(mask.sum()) else float("nan")

    def _rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(resid[mask])))) if int(mask.sum()) else float("nan")

    def _bias(mask: np.ndarray) -> float:
        return float(np.mean(resid[mask])) if int(mask.sum()) else float("nan")

    r1_mask = rid == 1
    r2_mask = rid == 2
    stall_mask = rid >= 4
    near_mask = rid == 5
    post_mask = rid == 6
    r1_low_drag_mask = r1_mask & (y <= 0.03)
    high_cd_tail_mask = post_mask & (y >= 0.27)
    coverage = cv_result.get("coverage", pd.DataFrame())
    if len(coverage) and {"scope", "level", "coverage"}.issubset(coverage.columns):
        row90 = coverage[(coverage["scope"] == "global") & (coverage["level"] == 90)]
        cov90 = float(row90["coverage"].mean()) if len(row90) else float("nan")
    else:
        cov90 = float("nan")
    return {
        "candidate": name,
        "overall_mae": float(mean_absolute_error(y, p)),
        "overall_rmse": rmse(y, p),
        "r2": safe_r2(y, p),
        "logo_mae": np.nan,
        "r1_mae": _mae(r1_mask),
        "r1_rmse": _rmse(r1_mask),
        "r1_bias": _bias(r1_mask),
        "r1_low_drag_mae": _mae(r1_low_drag_mask),
        "r2_mae": _mae(r2_mask),
        "r2_rmse": _rmse(r2_mask),
        "r2_bias": _bias(r2_mask),
        "stall_region_mae": _mae(stall_mask),
        "near_stall_mae": _mae(near_mask),
        "post_stall_mae": _mae(post_mask),
        "high_cd_tail_mae": _mae(high_cd_tail_mask),
        "high_cd_tail_rmse": _rmse(high_cd_tail_mask),
        "high_cd_tail_bias": _bias(high_cd_tail_mask),
        "coverage90": cov90,
        "uq_pearson": np.nan,
        "uncertainty_quality": "CV_P50_ONLY",
        "physics_violation_count": 0,
        "selection_source": "geometry_groupkfold_oof",
    }


def geometry_failure_rootcause_v3(df: pd.DataFrame, logo: pd.DataFrame, group_col: str, paths: Dict[str, Path]) -> pd.DataFrame:
    feats = [c for c in GEOMETRY_FAILURE_FEATURES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if logo is None or logo.empty or not feats or "mae" not in logo.columns:
        out = pd.DataFrame()
        out.to_csv(paths["tables"] / "geometry_failure_rootcause.csv", index=False)
        (paths["reports"] / "geometry_failure_rootcause_report.md").write_text("# Geometry Failure Root Cause\n\nGeometry refit validation was removed; no per-geometry MAE table is available.", encoding=OUTPUT_TEXT_ENCODING)
        return out
    geom = df.groupby(group_col)[feats].median().reset_index()
    valid = logo.dropna(subset=["mae"]).merge(geom, on=group_col, how="left")
    med = df[feats].median(numeric_only=True)
    q1 = df[feats].quantile(0.25, numeric_only=True)
    q3 = df[feats].quantile(0.75, numeric_only=True)
    iqr = (q3 - q1).replace(0, np.nan)
    rows = []
    for f in feats:
        z = (valid[f] - med[f]) / (iqr[f] if np.isfinite(iqr[f]) and abs(iqr[f]) > 1e-12 else 1.0)
        corr = float(valid[f].corr(valid["mae"], method="spearman")) if valid[f].notna().sum() >= 5 else float("nan")
        worst_z = float(np.nanmean(np.abs(z.loc[valid["mae"] >= valid["mae"].quantile(0.80)]))) if len(valid) else float("nan")
        rows.append({"feature": f, "median": float(med[f]), "iqr": float(iqr[f]) if np.isfinite(iqr[f]) else 0.0, "spearman_with_logo_mae": corr, "worst_geometry_abs_z_iqr": worst_z, "driver_score": abs(corr) * (worst_z if np.isfinite(worst_z) else 0.0)})
    out = pd.DataFrame(rows).sort_values("driver_score", ascending=False)
    out.to_csv(paths["tables"] / "geometry_failure_rootcause.csv", index=False)
    top = out.head(8)
    explanation = "; ".join([f"{r.feature} (rho={r.spearman_with_logo_mae:.3f}, zIQR={r.worst_geometry_abs_z_iqr:.2f})" for r in top.itertuples()])
    lines = ["# Geometry Failure Root-Cause Report", "", "## Likely drivers", "", f"Failure likely due to descriptor shifts in: {explanation}.", "", "## Ranked drivers", _to_markdown(top.round(6))]
    (paths["reports"] / "geometry_failure_rootcause_report.md").write_text("\n".join(lines), encoding=OUTPUT_TEXT_ENCODING)
    return out


def champion_selection(candidates: pd.DataFrame, paths: Dict[str, Path]) -> Tuple[str, pd.DataFrame]:
    cand = candidates.copy()
    cand["uq_pass_rank"] = (cand["uncertainty_quality"] == "PASS").astype(int)
    cand = cand.sort_values(
        by=["stall_region_mae", "uq_pass_rank", "overall_rmse", "r2", "physics_violation_count"],
        ascending=[True, False, True, False, True],
    ).reset_index(drop=True)
    champion = str(cand.iloc[0]["candidate"])
    cand.to_csv(paths["tables"] / "champion_metrics.csv", index=False)
    lines = ["# Champion Selection Report", "", f"Selected champion: **{champion}**", "", "Selection priorities: stall-region MAE -> uncertainty quality -> RMSE -> R2 -> physics violations. Geometry refit validation is removed.", "", _to_markdown(cand)]
    (paths["reports"] / "champion_selection_report.md").write_text("\n".join(lines), encoding=OUTPUT_TEXT_ENCODING)
    return champion, cand


class Day5CdResearchInterface:
    """Day5-safe Cd V3 inference facade."""
    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle

    def _add_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = self.bundle["pca_meta"]
        g_cols = meta["g_cols"]
        missing = [c for c in g_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required geometry grid columns for PCA: {missing[:10]}")
        Xg = df[g_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        Xg = meta["imputer"].transform(Xg).astype(np.float32, copy=False)
        Xs = meta["scaler"].transform(Xg).astype(np.float32, copy=False)
        Z = meta["pca"].transform(Xs)[:, :meta["keep"]].astype(np.float32, copy=False)
        out = df.copy()
        for i, c in enumerate(meta["z_cols"]):
            out[c] = Z[:, i].astype(np.float32, copy=False)
        if "regime_id" not in out.columns:
            out["regime_id"] = regime_id_from_angle(out["angle"].to_numpy(float))
        if "log10_re" not in out.columns:
            out["log10_re"] = np.log10(np.maximum(out["reynolds"].to_numpy(float), 1.0)) if "reynolds" in out.columns else 6.0
        return out

    def _predict_physics_flags(self, x: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
        tmp = x.copy().reset_index(drop=True)
        tmp["cd_pred"] = pred["cd_p50"].to_numpy(float)
        group_col = self.bundle.get("group_col", None)
        if group_col not in tmp.columns:
            tmp["__day5_geom_group__"] = "single_geometry"
            group_col = "__day5_geom_group__"
        if "log10_re" not in tmp.columns:
            tmp["log10_re"] = 6.0
        settings = self.bundle.get("cd_physics_settings", CD_PHYSICS_SETTINGS)
        flags = np.zeros(len(tmp), dtype=bool)
        for _, idx in tmp.groupby([group_col, "log10_re"], sort=False).groups.items():
            loc = np.asarray(list(idx), dtype=int)
            grp = tmp.iloc[loc].copy()
            metrics = evaluate_cd_physics_curve(grp["angle"].to_numpy(float), grp["cd_pred"].to_numpy(float), settings)
            flags[loc] = bool(metrics["cd_physics_flag"])
        return flags

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._add_pca(df)
        model = self.bundle["model"]
        micro = self.bundle["micro"]
        conformal = self.bundle["conformal"]
        residual_correctors = self.bundle.get("residual_correctors")
        if residual_correctors is None:
            legacy = self.bundle.get("r1_corrector")
            residual_correctors = [legacy] if legacy is not None else []
        geom_ood = self.bundle["geometry_ood"]
        flow_ood = self.bundle["flow_ood"]
        trust = self.bundle["trust_region"]
        uncertainty_quality = self.bundle.get("uncertainty_quality", "UNKNOWN")
        pred = model.predict(x)
        pred["cd_p50"] = micro.apply(pred["cd_p50"].to_numpy(float), x["regime_id"].to_numpy(int))
        pred["cd_p10"] = np.minimum(pred["cd_p10"], pred["cd_p50"])
        pred["cd_p90"] = np.maximum(pred["cd_p90"], pred["cd_p50"])
        pred["uncertainty_width"] = pred["cd_p90"] - pred["cd_p10"]
        pred = apply_residual_corrector_stack(residual_correctors, x, pred)
        pred = conformal.apply(x, pred, 0.90)
        gd, gf = geom_ood.score(x); fd, ff = flow_ood.score(x)
        pred["geometry_ood_score"] = gd; pred["geometry_ood_flag"] = gf
        pred["flow_ood_score"] = fd; pred["flow_ood_flag"] = ff
        tr = trust.evaluate(gd, fd)
        pred = pd.concat([pred.reset_index(drop=True), tr.reset_index(drop=True)], axis=1)
        # V3.1 FIX: Day5 physics safety is evaluated at inference time, not hardcoded to False.
        pred["cd_physics_flag"] = self._predict_physics_flags(x, pred)
        pred["reliability_score"] = np.clip(1.0 - pred["trust_region_score"].to_numpy(float), 0.0, 1.0)
        pred["safety_flag"] = ~((pred["trust_region_class"] == "UNSAFE") | pred["geometry_ood_flag"] | pred["flow_ood_flag"] | pred["cd_physics_flag"] | (uncertainty_quality == "UNCERTAINTY_NOT_RELIABLE"))
        return pred

    def predict_cd(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_full(df)["cd_p50"].to_numpy(float)

    def predict_cd_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_full(df)
        return p[["cd_p10", "cd_p50", "cd_p90", "cd_c90_lo", "cd_c90_hi", "uncertainty_width"]]

    def predict_ood(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_full(df)
        return p[["geometry_ood_score", "geometry_ood_flag", "flow_ood_score", "flow_ood_flag"]]

    def predict_trust_region(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_full(df)
        return p[["trust_region_score", "trust_region_class"]]

    def predict_reliability(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_full(df)["reliability_score"].to_numpy(float)

    def predict_safety(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_full(df)
        return p[["safety_flag", "trust_region_class", "geometry_ood_flag", "flow_ood_flag", "cd_physics_flag", "reliability_score"]]


def export_day5_artifacts(bundle: Dict[str, Any], paths: Dict[str, Path], cfg: ResearchConfig) -> None:
    iface = Day5CdResearchInterface(bundle)
    bundle_path = paths["day5"] / "day5_cd_v3_interface.pkl"
    joblib.dump(iface, bundle_path)
    spec = {
        "name": "Day4C_CD_V3_1_Fixed Safety Interface",
        "bundle_path": str(bundle_path),
        "methods": ["predict_cd", "predict_cd_interval", "predict_ood", "predict_trust_region", "predict_reliability", "predict_safety", "predict_full"],
        "prediction_columns": ["cd_p10", "cd_p50", "cd_p90", "cd_c90_lo", "cd_c90_hi", "uncertainty_width", "geometry_ood_score", "geometry_ood_flag", "flow_ood_score", "flow_ood_flag", "trust_region_score", "trust_region_class", "cd_physics_flag", "reliability_score", "safety_flag"],
        "safety_rule": "safety_flag=False if UNSAFE trust region OR geometry/flow OOD OR Cd physics flag OR unreliable uncertainty.",
        "required_input": "Day2-compatible DataFrame with g_* geometry grid, angle, reynolds/log10_re, and geometry descriptors.",
        "strict_feature_validation": True,
        "config": asdict(cfg),
    }
    (paths["day5"] / "day5_interface.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def write_final_report_v3(paths: Dict[str, Path], overall: pd.DataFrame, regime: pd.DataFrame, logo: pd.DataFrame, trust_metrics: pd.DataFrame, physics: pd.DataFrame, uq_metrics: pd.DataFrame, coverage: pd.DataFrame, rootcause: pd.DataFrame, champion: str, champion_metrics: pd.DataFrame) -> None:
    physics_summary = pd.DataFrame([{
        "n_curves": int(len(physics)),
        "true_violating_curves": int(physics["true_cd_physics_flag"].sum()) if len(physics) and "true_cd_physics_flag" in physics.columns else 0,
        "pred_violating_curves": int(physics["pred_cd_physics_flag"].sum()) if len(physics) and "pred_cd_physics_flag" in physics.columns else int(physics["cd_physics_flag"].sum()) if len(physics) and "cd_physics_flag" in physics.columns else 0,
        "true_violation_rate": float(physics["true_violation_rate"].mean()) if len(physics) and "true_violation_rate" in physics.columns else 0.0,
        "pred_violation_rate": float(physics["pred_violation_rate"].mean()) if len(physics) and "pred_violation_rate" in physics.columns else float(physics["violation_rate"].mean()) if len(physics) and "violation_rate" in physics.columns else 0.0,
        "negative_cd_total": int(physics["negative_cd_count"].sum()) if len(physics) and "negative_cd_count" in physics.columns else 0,
        "post_stall_collapse_total": int(physics["post_stall_cd_collapse_count"].sum()) if len(physics) and "post_stall_cd_collapse_count" in physics.columns else 0,
    }])
    lines = [
        "# Day4C_CD_V3_1_Fixed Publication Report", "",
        "## Overall metrics", _to_markdown(overall.round(6)), "",
        "## Regime metrics", _to_markdown(regime.round(6)) if len(regime) else "No regime metrics.", "",
        "## Geometry refit validation", _to_markdown(logo.sort_values("mae").head(15).round(6)) if logo is not None and len(logo) and "mae" in logo else "Removed for runtime; no per-geometry refits are executed.", "",
        "## Trust-region analysis", _to_markdown(trust_metrics.round(6)), "",
        "## Physics consistency analysis", _to_markdown(physics_summary.round(6)), "",
        "## Uncertainty quality", _to_markdown(uq_metrics.round(6)), "",
        "## Conformal coverage", _to_markdown(coverage.round(6)), "",
        "## Geometry GroupKFold CV", "See `reports/geometry_groupkfold_cv_report.md` and `tables/groupkfold_*` files. This validation holds out geometry groups by fold and replaces expensive LOGO refits.", "",
        "## Geometry failure root cause", _to_markdown(rootcause.head(12).round(6)) if len(rootcause) else "No root-cause table.", "",
        "## Champion selection", f"Champion: **{champion}**", "", _to_markdown(champion_metrics.round(6)), "",
        "## Day5 readiness", "Exports `day5_interface/day5_cd_v3_interface.pkl` and `day5_interface/day5_interface.json` with safety-aware prediction methods.", "",
        "## Research conclusions", "This V3 system prioritizes geometry generalization, stall-region error, uncertainty validity, trust-region safety, and physics consistency over raw R².",
    ]
    txt = "\n".join(lines)
    (paths["reports"] / "day4c_cd_v3_publication_report.md").write_text(txt, encoding=OUTPUT_TEXT_ENCODING)
    (paths["root"] / "day4c_cd_v3_publication_report.md").write_text(txt, encoding=OUTPUT_TEXT_ENCODING)


def register_pickle_module_alias() -> None:
    """Ensure artifacts created by script execution can be loaded after importing day4_cd."""
    if __name__ != "__main__":
        return
    module_name = Path(__file__).stem
    sys.modules.setdefault(module_name, sys.modules[__name__])
    for obj in list(globals().values()):
        if isinstance(obj, type) and obj.__module__ == "__main__":
            obj.__module__ = module_name



def main() -> None:
    register_pickle_module_alias()
    parser = argparse.ArgumentParser(description="Day4C_CD_V3_1_Fixed — Cd surrogate reliability and Day5 safety pipeline")
    parser.add_argument("--input_csv", default=ResearchConfig.input_csv)
    parser.add_argument("--output_dir", default="outputs/day4c_cd_v3_final")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=None, help="CPU workers (-1 = all cores)")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--bootstrap_models", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=ResearchConfig.n_estimators, help="Boosting iterations for the final full ensemble.")
    parser.add_argument("--logo_max_geometries", type=int, default=0, help="Compatibility flag; ignored because LOGO is removed.")
    parser.add_argument("--logo_sample_train_rows", type=int, default=0, help="Compatibility flag; ignored because LOGO is removed.")
    parser.add_argument("--logo_mode", choices=["logo", "groupkfold", "disabled", "removed"], default="removed", help="Compatibility flag; ignored because LOGO is removed.")
    parser.add_argument("--logo_cv_splits", type=int, default=5, help="Compatibility flag; ignored because LOGO is removed.")
    parser.add_argument("--disable_logo", action="store_true", help="Compatibility flag; LOGO is removed and always skipped.")
    parser.add_argument("--geometry_cv_splits", type=int, default=ResearchConfig.geometry_cv_splits, help="Geometry GroupKFold split count for Cd validation.")
    parser.add_argument("--geometry_cv_bootstrap_models", type=int, default=ResearchConfig.geometry_cv_bootstrap_models, help="Bootstrap bags used per geometry CV fold; default 1 to control runtime.")
    parser.add_argument("--geometry_cv_n_estimators", type=int, default=ResearchConfig.geometry_cv_n_estimators, help="Boosting iterations used only for lightweight architecture-selection CV.")
    parser.add_argument("--geometry_cv_regime_models", action="store_true", help="Also fit per-regime models in CV; slower. Final training always keeps regime models.")
    parser.add_argument("--full_geometry_cv", action="store_true", help="Use full p10/p50/p90 ensemble in every CV fold; much slower.")
    parser.add_argument("--disable_geometry_cv", action="store_true", help="Skip Cd Geometry GroupKFold validation.")
    parser.add_argument("--disable_ood", action="store_true")
    parser.add_argument("--use_isotonic_calibration", action="store_true")
    parser.add_argument("--baseline_metrics_csv", default=None)
    args = parser.parse_args()
    cfg = ResearchConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_jobs=(args.n_jobs if args.n_jobs is not None else ResearchConfig.n_jobs),
        max_rows=args.max_rows,
        bootstrap_models=args.bootstrap_models,
        n_estimators=max(25, args.n_estimators),
        logo_max_geometries=args.logo_max_geometries,
        logo_sample_train_rows=args.logo_sample_train_rows,
        logo_mode="removed",
        logo_cv_splits=args.logo_cv_splits,
        geometry_cv_enabled=(not args.disable_geometry_cv),
        geometry_cv_splits=args.geometry_cv_splits,
        geometry_cv_p50_only=(not args.full_geometry_cv),
        geometry_cv_bootstrap_models=max(1, args.geometry_cv_bootstrap_models),
        geometry_cv_global_only=(not args.geometry_cv_regime_models),
        geometry_cv_n_estimators=max(25, args.geometry_cv_n_estimators),
        disable_logo=True,
        disable_ood=args.disable_ood,
        use_isotonic_calibration=args.use_isotonic_calibration,
        baseline_metrics_csv=args.baseline_metrics_csv,
    )
    cfg.logo_mode = "removed"
    cfg.disable_logo = True
    cfg.n_jobs = configure_thread_env(cfg.n_jobs)
    paths = ensure_dirs(cfg.output_dir)
    t0 = time.time()
    print(f"[Day4C-CD-V3] n_jobs={cfg.n_jobs}", flush=True)
    print("[Day4C-CD-V3] Loading dataset", flush=True)
    df = load_dataset(cfg)
    group_col = discover_geometry_group_col(df)
    print("[Day4C-CD-V3] Architecture selection by Geometry GroupKFold OOF", flush=True)
    selected_source = "baseline"
    if cfg.geometry_cv_enabled:
        cv_base = run_geometry_groupkfold_cv(df, group_col, cfg, paths, weighted=False, tag="baseline")
        cv_weighted = run_geometry_groupkfold_cv(df, group_col, cfg, paths, weighted=True, tag="stall_weighted")
        cv_comp = pd.DataFrame([
            summarize_geometry_cv_candidate("baseline", cv_base),
            summarize_geometry_cv_candidate("stall_weighted", cv_weighted),
        ])
        selected_source, champion_metrics = champion_selection(cv_comp, paths)
        selected_source = "stall_weighted" if selected_source == "stall_weighted" else "baseline"
        cv_comp.to_csv(paths["tables"] / "weighted_vs_baseline.csv", index=False)
        model_selection_lines = [
            "# Model Selection Report",
            "",
            "Architecture selection uses out-of-fold predictions from identical held-out geometry folds.",
            f"Selected full-training architecture: **{selected_source}**",
            "",
            _to_markdown(cv_comp.round(6)),
        ]
        (paths["reports"] / "model_selection_report.md").write_text(
            "\n".join(model_selection_lines),
            encoding=OUTPUT_TEXT_ENCODING,
        )
    else:
        champion_metrics = pd.DataFrame([{
            "candidate": "baseline",
            "selected": True,
            "selection_source": "geometry_cv_disabled",
        }])
        champion_metrics.to_csv(paths["tables"] / "champion_metrics.csv", index=False)
        (paths["reports"] / "geometry_groupkfold_cv_report.md").write_text(
            "# CD Geometry GroupKFold CV\n\nSkipped by --disable_geometry_cv.",
            encoding=OUTPUT_TEXT_ENCODING,
        )
        (paths["reports"] / "model_selection_report.md").write_text(
            "# Model Selection Report\n\nGeometry CV disabled; baseline architecture selected.",
            encoding=OUTPUT_TEXT_ENCODING,
        )

    print("[Day4C-CD-V3] Geometry GroupKFold train/cal/test for final evaluation", flush=True)
    train, cal, test = train_cal_test_split(df, group_col, cfg)
    gc.collect()
    use_weighted = selected_source == "stall_weighted"
    print(
        f"[Day4C-CD-V3] Training one final full ensemble: source={selected_source} "
        f"quantiles={cfg.quantiles} bootstrap={cfg.bootstrap_models}",
        flush=True,
    )
    try:
        pred_final, final_meta = fit_predict_pipeline_v3(
            train, cal, test, group_col, cfg, paths, weighted=use_weighted
        )
    except Exception as exc:
        if not use_weighted:
            raise
        print(
            f"[Day4C-CD-V3] Weighted final training failed; falling back to baseline: {exc!r}",
            flush=True,
        )
        selected_source = "baseline"
        use_weighted = False
        pred_final, final_meta = fit_predict_pipeline_v3(
            train, cal, test, group_col, cfg, paths, weighted=False
        )

    test2 = final_meta["test_z"]
    train2 = final_meta["train_z"]
    conformal_final = final_meta["conformal"]
    for lvl in (0.80, 0.90, 0.95):
        pred_final = conformal_final.apply(test2, pred_final, lvl)
    coverage_final = conformal_final.evaluate(test2, pred_final)

    if cfg.disable_ood:
        print("[Day4C-CD-V3] OOD + Trust Region skipped (--disable_ood)", flush=True)
        geom_ood = DisabledOODDetector()
        flow_ood = DisabledOODDetector()
        gd = np.zeros(len(test2), dtype=float)
        fd = np.zeros(len(test2), dtype=float)
        gf = np.zeros(len(test2), dtype=bool)
        ff = np.zeros(len(test2), dtype=bool)
        trust = TrustRegionManager(1.0, 1.0)
        tr_df = trust.evaluate(gd, fd)
        pred_final["geometry_ood_score"] = gd
        pred_final["geometry_ood_flag"] = gf
        pred_final["flow_ood_score"] = fd
        pred_final["flow_ood_flag"] = ff
        pred_final = pd.concat([pred_final.reset_index(drop=True), tr_df.reset_index(drop=True)], axis=1)
        pred_final["reliability_score"] = 1.0
        trust_metrics = trust.report(pred_final, paths)
    else:
        print("[Day4C-CD-V3] OOD + Trust Region", flush=True)
        geom_ood = GeometryOODDetector(cfg.geometry_ood_percentile).fit(
            train2, final_meta["z_cols"], group_col
        )
        flow_ood = FlowOODDetector(cfg.flow_ood_percentile).fit(train2)
        gd, gf = geom_ood.score(test2)
        fd, ff = flow_ood.score(test2)
        trust = TrustRegionManager(geom_ood.threshold, flow_ood.threshold)
        tr_df = trust.evaluate(gd, fd)
        pred_final["geometry_ood_score"] = gd
        pred_final["geometry_ood_flag"] = gf
        pred_final["flow_ood_score"] = fd
        pred_final["flow_ood_flag"] = ff
        pred_final = pd.concat([pred_final.reset_index(drop=True), tr_df.reset_index(drop=True)], axis=1)
        pred_final["reliability_score"] = np.clip(
            1.0 - pred_final["trust_region_score"].to_numpy(float), 0.0, 1.0
        )
        trust_metrics = trust.report(pred_final, paths)

    print("[Day4C-CD-V3] Cd physics consistency", flush=True)
    physics = cd_physics_consistency(
        test2, pred_final, group_col, paths, tag=selected_source
    )
    pred_final = attach_physics_flags_to_rows(
        test2, pred_final, physics, group_col
    )

    print("[Day4C-CD-V3] Uncertainty quality gate", flush=True)
    uq_metrics, final_uq_status = uncertainty_quality_gate(
        test2, pred_final, coverage_final, paths
    )
    pred_final["safety_flag"] = ~(
        (pred_final["trust_region_class"] == "UNSAFE")
        | pred_final["geometry_ood_flag"]
        | pred_final["flow_ood_flag"]
        | pred_final["cd_physics_flag"]
        | (final_uq_status == "UNCERTAINTY_NOT_RELIABLE")
    )

    y = test2["cd"].to_numpy(float)
    p = pred_final["cd_p50"].to_numpy(float)
    overall = pd.DataFrame([metric_row(y, p, "overall")])
    overall.to_csv(paths["tables"] / "overall_metrics.csv", index=False)
    reg_rows = []
    for r in REGIME_IDS:
        m = test2["regime_id"].to_numpy(int) == r
        if int(m.sum()):
            reg_rows.append(metric_row(y[m], p[m], REGIME_NAMES[r]))
    regime = pd.DataFrame(reg_rows)
    regime.to_csv(paths["tables"] / "regime_metrics.csv", index=False)
    coverage_final.to_csv(paths["tables"] / "conformal_coverage.csv", index=False)
    result = pd.concat(
        [
            test2[[group_col, "angle", "cd", "regime_id"]].reset_index(drop=True),
            pred_final.reset_index(drop=True),
        ],
        axis=1,
    )
    result.to_csv(paths["tables"] / "test_predictions_research_v3.csv", index=False)
    export_r1_diagnostics(test2, pred_final, group_col, paths, tag=selected_source)

    print("[Day4C-CD-V3] Geometry LOGO validation removed; writing placeholder artifacts", flush=True)
    logo = write_geometry_validation_removed_artifacts(paths, group_col, tag=selected_source)
    final_summary = summarize_candidate(
        "research_v3",
        test2,
        pred_final,
        coverage_final,
        float("nan"),
        int(physics["physics_violation_count"].sum()) if len(physics) else 0,
        final_uq_status,
    )
    final_summary["source_candidate"] = selected_source
    final_summary["selection_source"] = "final_holdout_after_geometry_cv"
    final_summary["safety_gated"] = True
    existing_comp = pd.read_csv(paths["tables"] / "weighted_vs_baseline.csv") if (
        paths["tables"] / "weighted_vs_baseline.csv"
    ).exists() else pd.DataFrame()
    pd.concat([existing_comp, pd.DataFrame([final_summary])], ignore_index=True).to_csv(
        paths["tables"] / "weighted_vs_baseline.csv", index=False
    )

    print("[Day4C-CD-V3] Geometry failure root-cause", flush=True)
    geometry_failure_analysis(df, logo, group_col, paths)
    rootcause = geometry_failure_rootcause_v3(df, logo, group_col, paths)
    failure = result.assign(
        abs_error=np.abs(result["cd_p50"] - result["cd"])
    ).sort_values("abs_error", ascending=False).head(50)
    failure.to_csv(paths["tables"] / "failure_cases.csv", index=False)
    save_diagnostics(
        test2, pred_final, logo, geom_ood, flow_ood, coverage_final, paths
    )

    champion = selected_source
    print("[Day4C-CD-V3] Export models and Day5 interface", flush=True)
    bundle = {
        "pca_meta": final_meta["pca_meta"],
        "feature_cols": final_meta["feature_cols"],
        "model": final_meta["model"],
        "micro": final_meta["micro"],
        "conformal": final_meta["conformal"],
        "residual_correctors": final_meta.get("residual_correctors", []),
        "r1_corrector": final_meta.get("r1_corrector"),
        "geometry_ood": geom_ood,
        "flow_ood": flow_ood,
        "trust_region": trust,
        "group_col": group_col,
        "config": cfg,
        "uncertainty_quality": final_uq_status,
        "champion": champion,
        "cd_physics_settings": dict(CD_PHYSICS_SETTINGS),
    }
    joblib.dump(bundle, paths["models"] / "research_v3_model_bundle.pkl")
    joblib.dump(final_meta["pca_meta"], paths["models"] / "pca_latent_model.pkl")
    joblib.dump(final_meta["model"], paths["models"] / "cd_bootstrap_quantile_regime_ensemble.pkl")
    joblib.dump(final_meta["micro"], paths["models"] / "regime_calibration.pkl")
    joblib.dump(final_meta["conformal"], paths["models"] / "regimewise_conformal_calibration.pkl")
    joblib.dump(geom_ood, paths["models"] / "geometry_ood_model.pkl")
    joblib.dump(flow_ood, paths["models"] / "flow_ood_model.pkl")
    joblib.dump(trust, paths["models"] / "trust_region_manager.pkl")
    (paths["tables"] / "feature_columns.json").write_text(
        json.dumps(final_meta["feature_cols"], indent=2), encoding="utf-8"
    )
    export_day5_artifacts(bundle, paths, cfg)

    write_final_report_v3(paths, overall, regime, logo, trust_metrics, physics, uq_metrics, coverage_final, rootcause, champion, champion_metrics)
    (paths["reports"] / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"[Day4C-CD-V3] Completed in {time.time() - t0:.1f}s")
    print(f"[Day4C-CD-V3] Report: {paths['root'] / 'day4c_cd_v3_publication_report.md'}")


if __name__ == "__main__":
    main()

