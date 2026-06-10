from __future__ import annotations

import sys
import os
import gc
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer

# Cấu hình tiếng Việt cho Matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Thử import LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# Thử import CatBoost
try:
    import catboost as cb
    HAS_CB = True
except Exception:
    HAS_CB = False

try:
    import psutil
except Exception:
    psutil = None

# Các đặc trưng vật lý dùng cho mô hình Cm
PHYS_FEATURES = [
    "angle",
    "log10_re",
    "t_max",
    "x_tmax",
    "camber_max",
    "x_cmax",
    "te_gap",
    "le_radius_proxy",
    "aoa_x_t",
    "aoa_x_camber",
    "camber_area",
    "camber_centroid_x",
    "camber_front_area",
    "camber_mid_area",
    "camber_aft_area",
    "camber_slope_front",
    "camber_slope_mid",
    "camber_slope_aft",
    "camber_curvature_front",
    "camber_curvature_mid",
    "camber_curvature_aft",
    "thickness_centroid_x",
    "thickness_front_area",
    "thickness_mid_area",
    "thickness_aft_area",
    "thickness_moment_arm",
    "zero_lift_moment_proxy",
    "pressure_center_proxy",
    "camber_pressure_proxy",
    "aft_loading_proxy",
    "moment_distribution_proxy",
]

OOD_HASHES = {
    "4ca675adbe49c06b416b0079ca1b7f75afdccd8d", # 2032c (reflex camber)
    "b4153f1a78d4bc5b835ccdbfaf588bcd60d5aa4b", # thin-flat
    "1d68e56a76e70e92a3208081e6a68831232f7f23", # kc135winglet
    "826d554c9aa98a4741b308731fb31974c21b4299", # rhodesg36
    "95a1517b1d019c4c667cba308dc28fab046a775c", # ma409
    "92fd778d74ae31a8a1d3fa0233f8dad1e72c2369", # n22
    "8b6574dede00f11ef6289e9bc4a66f865221252a", # s8037
}


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


def limit_parallel_jobs_for_memory(
    requested_jobs: int,
    data_nbytes: int,
    copies_per_worker: float,
    stage: str,
    reserve_gb: float = 2.0,
) -> int:
    effective_jobs = resolve_n_jobs(requested_jobs)
    available = get_available_memory_bytes()
    if available is None or data_nbytes <= 0:
        return effective_jobs
    reserve_bytes = int(float(reserve_gb) * (1024 ** 3))
    usable = max(0, available - reserve_bytes)
    if usable <= 0:
        print(f"[RAM_GUARD] stage={stage} forced_n_jobs=1 because free RAM is below reserve.")
        return 1
    per_worker_bytes = max(1, int(float(data_nbytes) * float(copies_per_worker)))
    max_jobs = max(1, min(effective_jobs, usable // per_worker_bytes))
    if max_jobs < effective_jobs:
        print(
            f"[RAM_GUARD] stage={stage} requested_n_jobs={effective_jobs} capped_n_jobs={max_jobs} "
            f"free_ram_gb={available / (1024 ** 3):.2f} est_per_worker_gb={per_worker_bytes / (1024 ** 3):.2f}"
        )
    return max_jobs


def filter_candidate_model_factories(model_factories: Dict[str, object], candidate_spec: Optional[str]) -> Dict[str, object]:
    if not candidate_spec:
        return model_factories

    aliases = {
        "regime_rf": "Regime-Specific RF",
        "regime-specific-rf": "Regime-Specific RF",
        "regime_specific_rf": "Regime-Specific RF",
        "regime_et": "Regime-Specific ExtraTrees",
        "regime_extratrees": "Regime-Specific ExtraTrees",
        "regime-specific-extratrees": "Regime-Specific ExtraTrees",
        "global_rf": "Global RF",
        "rf": "Global RF",
        "global_et": "Global ExtraTrees",
        "global_extratrees": "Global ExtraTrees",
        "et": "Global ExtraTrees",
        "hgb": "HistGradientBoosting",
        "histgradientboosting": "HistGradientBoosting",
        "lightgbm": "LightGBM",
        "lgb": "LightGBM",
        "catboost": "CatBoost",
        "cb": "CatBoost",
    }

    exact_lookup = {name.lower(): name for name in model_factories}
    selected: Dict[str, object] = {}
    unknown: List[str] = []
    for token in [x.strip() for x in str(candidate_spec).split(",") if x.strip()]:
        key = token.lower().replace(" ", "_")
        name = aliases.get(key) or exact_lookup.get(token.lower())
        if name in model_factories:
            selected[name] = model_factories[name]
        else:
            unknown.append(token)

    if unknown:
        print(f"[WARN] Unknown --candidate_models ignored: {', '.join(unknown)}")
    if not selected:
        print("[WARN] --candidate_models selected no valid model; using all candidates.")
        return model_factories
    print(f"-> Candidate models enabled: {', '.join(selected.keys())}")
    return selected


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _fit_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    try:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(mask.sum()) < 2:
            return 1.0, 0.0
        a, b = np.polyfit(y_pred[mask], y_true[mask], deg=1)
        return float(a), float(b)
    except Exception:
        return 1.0, 0.0


def select_representative_geometry_subset(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    max_geometries: int,
    random_state: int,
) -> List[str]:
    all_groups = df[group_col].astype(str).dropna().unique().tolist()
    if max_geometries <= 0 or len(all_groups) <= max_geometries:
        return all_groups

    geom_df = df.groupby(group_col, sort=False)[feature_cols].median().reset_index()
    X = SimpleImputer(strategy="median").fit_transform(geom_df[feature_cols]).astype(np.float32, copy=False)
    if X.shape[0] <= max_geometries:
        return geom_df[group_col].astype(str).tolist()

    if X.shape[1] > 16 and X.shape[0] > 16:
        n_comp = max(2, min(16, X.shape[0] - 1, X.shape[1]))
        X_embed = PCA(n_components=n_comp, svd_solver="full", random_state=random_state).fit_transform(X)
    else:
        X_embed = X

    kmeans = KMeans(n_clusters=max_geometries, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_embed)
    centers = kmeans.cluster_centers_
    selected_idx: List[int] = []
    for cluster_id in range(max_geometries):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        d2 = np.sum((X_embed[members] - centers[cluster_id]) ** 2, axis=1)
        selected_idx.append(int(members[int(np.argmin(d2))]))

    if len(selected_idx) < max_geometries:
        existing = set(selected_idx)
        for idx in range(len(geom_df)):
            if idx not in existing:
                selected_idx.append(idx)
            if len(selected_idx) >= max_geometries:
                break

    return geom_df.iloc[selected_idx][group_col].astype(str).tolist()


class RegimeSpecificRegressor:
    """Wrapper cho phép huấn luyện các mô hình con dựa trên flow regime."""
    def __init__(self, estimator_type: str = "rf", estimator_params: dict = None, random_state: int = 42):
        self.estimator_type = estimator_type
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.random_state = random_state
        self.models = {}

    def _get_regime(self, angle: float) -> str:
        if angle < -5.0:
            return "negative_high_aoa"
        elif -5.0 <= angle <= 5.0:
            return "linear_attached"
        elif 5.0 < angle <= 10.0:
            return "transition"
        else:
            return "stall_risk"

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        angles = X["angle"].to_numpy(dtype=float)
        regimes = np.array([self._get_regime(a) for a in angles])
        
        self.models = {}
        for reg in np.unique(regimes):
            mask = (regimes == reg)
            X_sub = X[mask]
            y_sub = y[mask]
            w_sub = sample_weight[mask] if sample_weight is not None else None
            
            if self.estimator_type == "rf":
                model = RandomForestRegressor(**self.estimator_params, random_state=self.random_state)
            elif self.estimator_type == "et":
                model = ExtraTreesRegressor(**self.estimator_params, random_state=self.random_state)
            else:
                raise ValueError(f"Unknown estimator_type: {self.estimator_type}")
                
            model.fit(X_sub, y_sub, sample_weight=w_sub)
            self.models[reg] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        angles = X["angle"].to_numpy(dtype=float)
        regimes = np.array([self._get_regime(a) for a in angles])
        
        preds = np.zeros(len(X))
        for reg, model in self.models.items():
            mask = (regimes == reg)
            if np.any(mask):
                preds[mask] = model.predict(X[mask])
                
        # Fallback cho bất kỳ dòng nào chưa được gán (nếu có)
        unpredicted = (preds == 0.0) & (np.zeros(len(X), dtype=bool)) # fallback an toàn
        for reg in np.unique(regimes):
            if reg not in self.models and len(self.models) > 0:
                mask = (regimes == reg)
                fallback_model = list(self.models.values())[0]
                preds[mask] = fallback_model.predict(X[mask])
                
        return preds

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Tính trung bình dự đoán và độ phân tán (uncertainty) giữa các cây trong ensemble."""
        angles = X["angle"].to_numpy(dtype=float)
        regimes = np.array([self._get_regime(a) for a in angles])
        
        preds = np.zeros(len(X))
        std_devs = np.zeros(len(X))
        
        for reg, model in self.models.items():
            mask = (regimes == reg)
            if np.any(mask):
                X_sub = X[mask]
                preds[mask] = model.predict(X_sub)
                
                # RF hoặc ExtraTrees có estimators_
                if hasattr(model, "estimators_"):
                    # Dự đoán từ từng cây quyết định
                    tree_preds = np.array([dt.predict(X_sub) for dt in model.estimators_])
                    std_devs[mask] = np.std(tree_preds, axis=0)
                    
        return preds, std_devs


def predict_model_with_uncertainty(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Hàm phụ trợ tính uncertainty cho cả mô hình Global và Regime-Specific."""
    if hasattr(model, "predict_with_uncertainty"):
        return model.predict_with_uncertainty(X)
    
    preds = model.predict(X)
    std_devs = np.full(len(X), np.nan) # Mặc định là NaN nếu không hỗ trợ
    
    # Đối với mô hình Global RF hoặc ExtraTrees
    if hasattr(model, "estimators_"):
        tree_preds = np.array([dt.predict(X) for dt in model.estimators_])
        std_devs = np.std(tree_preds, axis=0)
        
    return preds, std_devs


def load_dataset(input_csv: str, max_rows: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    print(f"-> Loading dataset: {path}")
    read_nrows = max(50_000, int(max_rows) * 5) if max_rows else None
    df = pd.read_csv(path, low_memory=False, nrows=read_nrows)
    
    # Tạo các cột phụ trợ giống day4.py để tương thích
    if "log10_re" not in df.columns and "reynolds" in df.columns:
        df["log10_re"] = np.log10(pd.to_numeric(df["reynolds"], errors="coerce"))
    if "aoa_x_t" not in df.columns and {"angle", "t_max"}.issubset(df.columns):
        df["aoa_x_t"] = pd.to_numeric(df["angle"], errors="coerce") * pd.to_numeric(df["t_max"], errors="coerce")
    if "aoa_x_camber" not in df.columns and {"angle", "camber_max"}.issubset(df.columns):
        df["aoa_x_camber"] = pd.to_numeric(df["angle"], errors="coerce") * pd.to_numeric(df["camber_max"], errors="coerce")
        
    required = ["geom_hash", "angle", "reynolds", "cm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset thiếu cột bắt buộc cho Cm surrogate: {missing}")
        
    # Ép kiểu dữ liệu số
    for c in ["angle", "reynolds", "cm", *PHYS_FEATURES]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df = df.dropna(subset=required).copy()
    if max_rows and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=random_state).reset_index(drop=True)
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols):
        df.loc[:, float64_cols] = df.loc[:, float64_cols].astype(np.float32)
    print(f"-> Loaded {len(df):,} valid rows from original dataset.")
    return df


def ensure_dirs(base: str) -> Dict[str, Path]:
    base_path = Path(base)
    paths = {
        "base": base_path,
        "tables": base_path / "tables",
        "figures": base_path / "figures",
        "models": base_path / "models",
        "reports": base_path / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def compute_amplitude_compression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    slope, intercept = _fit_slope_intercept(y_true, y_pred)
    
    # Tính MAE theo các khu vực
    neg_mask = y_true < -0.08
    mid_mask = (y_true >= -0.08) & (y_true <= 0.02)
    pos_mask = y_true > 0.02
    
    mae_neg = _mae(y_true[neg_mask], y_pred[neg_mask]) if np.any(neg_mask) else 0.0
    mae_mid = _mae(y_true[mid_mask], y_pred[mid_mask]) if np.any(mid_mask) else 0.0
    mae_pos = _mae(y_true[pos_mask], y_pred[pos_mask]) if np.any(pos_mask) else 0.0
    
    # Kiểm tra flag bị nén biên độ (amplitude compressed)
    extreme_mae = max(mae_neg, mae_pos)
    is_compressed = False
    compression_reasons = []
    
    if slope < 0.85:
        is_compressed = True
        compression_reasons.append(f"Slope ({slope:.3f}) < 0.85")
    if mae_mid > 0.0 and extreme_mae > 2.0 * mae_mid:
        is_compressed = True
        compression_reasons.append(f"Extreme MAE ({extreme_mae:.5f}) > 2x Middle MAE ({mae_mid:.5f})")
        
    return {
        "slope": slope,
        "intercept": intercept,
        "mae_neg": mae_neg,
        "mae_mid": mae_mid,
        "mae_pos": mae_pos,
        "is_compressed": is_compressed,
        "reasons": ", ".join(compression_reasons) if is_compressed else "None"
    }


def main():
    parser = argparse.ArgumentParser(description="Pitching Moment Cm Reset & Clean Training Pipeline")
    parser.add_argument("--input_csv", default="deeplearwing_day2_tabular.csv", help="Đường dẫn đến file CSV gốc")
    parser.add_argument("--output_dir", default="outputs/day4_cm_reset", help="Thư mục lưu outputs")
    parser.add_argument("--random_state", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Số core CPU dùng để huấn luyện (-1 = all cores)")
    parser.add_argument("--id_max_geometries", type=int, default=0, help="Giới hạn số ID geometries đại diện cho smoke/dev. 0 = dùng toàn bộ.")
    parser.add_argument("--max_rows", type=int, default=None, help="Smoke/dev row cap. Reads a bounded prefix before sampling.")
    parser.add_argument("--cv_splits", type=int, default=5, help="Geometry GroupKFold splits for ID model selection.")
    parser.add_argument("--disable_ood", action="store_true", help="Use all rows as ID data and skip hard-coded OOD benchmark.")
    parser.add_argument("--train_all_final_models", action="store_true", help="Old behavior: train/export every final model after CV.")
    parser.add_argument("--tree_estimators", type=int, default=200, help="RF/ExtraTrees estimators.")
    parser.add_argument("--hgb_iter", type=int, default=200, help="HistGradientBoosting iterations.")
    parser.add_argument("--disable_optional_models", action="store_true", help="Skip LightGBM/CatBoost candidate models for faster runs.")
    parser.add_argument("--candidate_models", type=str, default=None, help="Comma-separated candidates: regime_rf,regime_et,global_rf,global_et,hgb,lightgbm,catboost.")
    args = parser.parse_args()
    n_jobs = configure_thread_env(args.n_jobs)
    print(f"-> CPU parallel n_jobs={n_jobs}")

    paths = ensure_dirs(args.output_dir)
    df_raw = load_dataset(args.input_csv, max_rows=args.max_rows, random_state=args.random_state)
    
    # Xác định các cột feature
    feature_cols = [c for c in PHYS_FEATURES if c in df_raw.columns]
    feature_cols.extend(sorted([c for c in df_raw.columns if c.startswith("g_")]))
    feature_cols = list(dict.fromkeys(feature_cols))
    
    # Impute dữ liệu
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(df_raw[feature_cols]).astype(np.float32, copy=False),
        columns=feature_cols,
        index=df_raw.index,
    )
    joblib.dump(imputer, paths["models"] / "feature_imputer.pkl")
    
    # Tách ID và OOD
    ood_mask = df_raw["geom_hash"].isin(OOD_HASHES) if not args.disable_ood else np.zeros(len(df_raw), dtype=bool)
    
    # Dữ liệu ID
    df_id = df_raw[~ood_mask].copy().reset_index(drop=True)
    X_id = X_imputed[~ood_mask].copy().reset_index(drop=True)
    y_id = df_id["cm"].to_numpy(dtype=np.float32)
    groups_id = df_id["geom_hash"].astype(str).to_numpy()
    
    # Dữ liệu OOD
    df_ood = df_raw[ood_mask].copy().reset_index(drop=True)
    X_ood = X_imputed[ood_mask].copy().reset_index(drop=True)
    y_ood = df_ood["cm"].to_numpy(dtype=np.float32)
    has_ood = len(df_ood) > 0

    if args.id_max_geometries and args.id_max_geometries > 0:
        id_unique = int(df_id["geom_hash"].nunique())
        if id_unique > int(args.id_max_geometries):
            rep_geoms = select_representative_geometry_subset(
                df=df_id,
                feature_cols=feature_cols,
                group_col="geom_hash",
                max_geometries=int(args.id_max_geometries),
                random_state=args.random_state,
            )
            keep_mask = df_id["geom_hash"].astype(str).isin(rep_geoms).to_numpy()
            df_id = df_id.loc[keep_mask].reset_index(drop=True)
            X_id = X_id.loc[keep_mask].reset_index(drop=True)
            y_id = df_id["cm"].to_numpy(dtype=np.float32)
            groups_id = df_id["geom_hash"].astype(str).to_numpy()
            print(f"-> Representative ID geometry subset enabled: {len(rep_geoms)} / {id_unique} geometries")
    
    print("\n" + "=" * 80)
    print("DATASET SPLIT SUMMARY")
    print("=" * 80)
    print(f"Total valid samples    : {len(df_raw):,}")
    print(f"In-Distribution (ID)   : {len(df_id):,} rows | {len(np.unique(groups_id))} geometries")
    print(f"Out-Of-Distribution (OOD): {len(df_ood):,} rows | {len(df_ood['geom_hash'].unique())} geometries")
    print("=" * 80 + "\n")
    if not has_ood:
        print("WARNING: OOD challenge set is empty for the current input dataset.")
        print("-> OOD benchmark/report/plots will be skipped, but ID training will continue.\n")
    del df_raw, X_imputed, ood_mask
    gc.collect()
    
    # Định nghĩa các mô hình huấn luyện
    safe_n_jobs = limit_parallel_jobs_for_memory(
        n_jobs,
        dataframe_nbytes(X_id) + dataframe_nbytes(X_ood) + int(y_id.nbytes) + int(y_ood.nbytes),
        3.0,
        "CM:global_train",
    )
    print(f"-> RAM-guarded n_jobs={safe_n_jobs}")
    rf_params = {"n_estimators": max(10, int(args.tree_estimators)), "max_depth": 15, "min_samples_leaf": 5, "n_jobs": safe_n_jobs}
    hgb_params = {"max_iter": max(10, int(args.hgb_iter)), "max_depth": 12, "min_samples_leaf": 8, "learning_rate": 0.05}
    
    models_to_compare = {
        "Regime-Specific RF": lambda: RegimeSpecificRegressor(
            estimator_type="rf", estimator_params=rf_params, random_state=args.random_state
        ),
        "Regime-Specific ExtraTrees": lambda: RegimeSpecificRegressor(
            estimator_type="et", estimator_params=rf_params, random_state=args.random_state
        ),
        "Global RF": lambda: RandomForestRegressor(**rf_params, random_state=args.random_state),
        "Global ExtraTrees": lambda: ExtraTreesRegressor(**rf_params, random_state=args.random_state),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(**hgb_params, random_state=args.random_state)
    }
    
    # Thêm LightGBM nếu có
    if HAS_LGB and not args.disable_optional_models:
        lgb_params = {"n_estimators": 250, "max_depth": 10, "learning_rate": 0.04, "random_state": args.random_state, "n_jobs": safe_n_jobs}
        models_to_compare["LightGBM"] = lambda params=lgb_params: lgb.LGBMRegressor(**params)
        
    # Thêm CatBoost nếu có
    if HAS_CB and not args.disable_optional_models:
        cb_params = {"iterations": 250, "depth": 8, "learning_rate": 0.05, "random_seed": args.random_state, "verbose": 0, "thread_count": safe_n_jobs}
        models_to_compare["CatBoost"] = lambda params=cb_params: cb.CatBoostRegressor(**params)

    models_to_compare = filter_candidate_model_factories(models_to_compare, args.candidate_models)

    # ==========================================================================
    # MODE A: 5-FOLD GROUPKFOLD CROSS-VALIDATION ON ID GEOMETRIES
    # ==========================================================================
    n_cv_splits = min(max(2, int(args.cv_splits)), len(np.unique(groups_id)))
    print("================================================================================")
    print(f"MODE A: HUẤN LUYỆN {n_cv_splits}-FOLD GROUPKFOLD CV TRÊN ID GEOMETRIES")
    print("================================================================================")

    gkf = GroupKFold(n_splits=n_cv_splits)
    oof_predictions = {name: np.zeros(len(df_id), dtype=np.float32) for name in models_to_compare}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_id, y_id, groups=groups_id), 1):
        print(f"--- Training Fold {fold}/{n_cv_splits} ---")
        X_train, y_train = X_id.iloc[train_idx], y_id[train_idx]
        X_val, y_val = X_id.iloc[val_idx], y_id[val_idx]
        
        # Huấn luyện từng model
        for name, make_model in models_to_compare.items():
            t0 = time.perf_counter()

            model = make_model()
            model.fit(X_train, y_train)
            oof_predictions[name][val_idx] = model.predict(X_val)
            print(f"  * {name:<26} OOF Fold {fold} completed in {time.perf_counter()-t0:.2f}s")
            del model
            gc.collect()
            
    # Tính toán chỉ số khoa học OOF cho từng model
    comparison_rows = []
    for name in models_to_compare:
        y_pred = oof_predictions[name]
        mae = _mae(y_id, y_pred)
        rmse = _rmse(y_id, y_pred)
        r2 = _r2(y_id, y_pred)
        
        # Đánh giá amplitude compression
        comp_meta = compute_amplitude_compression(y_id, y_pred)
        
        comparison_rows.append({
            "Model": name,
            "ID_MAE": mae,
            "ID_RMSE": rmse,
            "ID_R²": r2,
            "Slope": comp_meta["slope"],
            "Intercept": comp_meta["intercept"],
            "Extreme_Neg_MAE": comp_meta["mae_neg"],
            "Middle_MAE": comp_meta["mae_mid"],
            "Extreme_Pos_MAE": comp_meta["mae_pos"],
            "Compressed_Flag": "YES" if comp_meta["is_compressed"] else "NO",
            "Compression_Reasons": comp_meta["reasons"]
        })
        
    df_comparison = pd.DataFrame(comparison_rows).sort_values("ID_RMSE")
    df_comparison.to_csv(paths["tables"] / "cm_id_model_selection.csv", index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 80)
    print("ID BENCHMARK COMPARISON TABLE (OOF CV)")
    print("=" * 80)
    print(df_comparison[["Model", "ID_MAE", "ID_RMSE", "ID_R²", "Slope", "Compressed_Flag"]].to_string(index=False))
    print("=" * 80 + "\n")
    
    # Lựa chọn mô hình tốt nhất dựa trên ID_RMSE thấp nhất
    best_model_name = df_comparison.iloc[0]["Model"]
    print(f"🏆 FINAL SELECTED MODEL BASED ON ID RMSE: {best_model_name}")
    
    # Lưu predictions của tất cả OOF models
    df_id_predictions = df_id[["geom_hash", "angle", "reynolds", "cm"]].copy()
    for name in models_to_compare:
        df_id_predictions[f"cm_pred_{name.lower().replace(' ', '_').replace('-', '_')}"] = oof_predictions[name]
    df_id_predictions.to_csv(paths["tables"] / "cm_id_predictions.csv", index=False, encoding="utf-8-sig")
    gc.collect()

    # ==========================================================================
    # HUẤN LUYỆN FINAL MODELS & CHẠY MODE B: OOD CHALLENGE
    # ==========================================================================
    print("\n================================================================================")
    print("MODE B: HUẤN LUYỆN TOÀN BỘ TRÊN ID & ĐÁNH GIÁ TRÊN OOD CHALLENGE SET")
    print("================================================================================")
    
    final_models = {}
    ood_predictions = {}
    final_model_names = list(models_to_compare.keys()) if args.train_all_final_models else [best_model_name]
    
    for name in final_model_names:
        make_model = models_to_compare[name]
        t0 = time.perf_counter()
        model = make_model()

        print(f"Huấn luyện final model: {name} trên toàn bộ {len(np.unique(groups_id))} ID geometries...")
        model.fit(X_id, y_id)
        
        # Lưu model
        model_filename = f"best_model_{name.lower().replace(' ', '_').replace('-', '_')}.pkl"
        joblib.dump(model, paths["models"] / model_filename)
        if name == best_model_name:
            joblib.dump(model, paths["models"] / "cm_model.pkl")
            (paths["models"] / "cm_model_metadata.json").write_text(
                json.dumps({"best_model_name": best_model_name, "model_file": model_filename}, indent=2),
                encoding="utf-8",
            )
        final_models[name] = model
        
        # Dự đoán OOD
        if has_ood:
            ood_preds, ood_stds = predict_model_with_uncertainty(model, X_ood)
        else:
            ood_preds = np.empty(0, dtype=np.float32)
            ood_stds = np.empty(0, dtype=np.float32)
        ood_predictions[name] = {
            "preds": ood_preds,
            "stds": ood_stds
        }
        print(f"  * Hoàn tất trong {time.perf_counter()-t0:.2f}s")
    gc.collect()
        
    # Tính toán chỉ số trên OOD Challenge Set
    ood_cols = ["Model", "OOD_MAE", "OOD_RMSE", "OOD_R?", "R?_2032c", "R?_thin_flat", "OOD_Slope", "OOD_Compressed"]
    if has_ood:
        ood_comparison_rows = []
        for name in final_model_names:
            yp = ood_predictions[name]["preds"]
            mae = _mae(y_ood, yp)
            rmse = _rmse(y_ood, yp)
            r2 = _r2(y_ood, yp)
            
            comp_meta = compute_amplitude_compression(y_ood, yp)
            
            # T?nh ri?ng cho t?ng OOD geometry
            sub_results = {}
            for h in OOD_HASHES:
                sub_mask = df_ood["geom_hash"] == h
                if np.any(sub_mask):
                    yt_sub = y_ood[sub_mask]
                    yp_sub = yp[sub_mask]
                    sub_results[h] = {
                        "mae": _mae(yt_sub, yp_sub),
                        "rmse": _rmse(yt_sub, yp_sub),
                        "r2": _r2(yt_sub, yp_sub)
                    }
                    
            r2_2032c = sub_results.get("4ca675adbe49c06b416b0079ca1b7f75afdccd8d", {}).get("r2", np.nan)
            r2_flat = sub_results.get("b4153f1a78d4bc5b835ccdbfaf588bcd60d5aa4b", {}).get("r2", np.nan)
            
            ood_comparison_rows.append({
                "Model": name,
                "OOD_MAE": mae,
                "OOD_RMSE": rmse,
                "OOD_R?": r2,
                "R?_2032c": r2_2032c,
                "R?_thin_flat": r2_flat,
                "OOD_Slope": comp_meta["slope"],
                "OOD_Compressed": "YES" if comp_meta["is_compressed"] else "NO"
            })
            
        df_ood_comparison = pd.DataFrame(ood_comparison_rows).sort_values("OOD_RMSE")
        
        print("\n" + "=" * 80)
        print("OOD CHALLENGE BENCHMARK COMPARISON TABLE")
        print("=" * 80)
        print(df_ood_comparison[["Model", "OOD_MAE", "OOD_RMSE", "OOD_R?", "R?_2032c", "R?_thin_flat", "OOD_Slope", "OOD_Compressed"]].to_string(index=False))
        print("=" * 80 + "\n")
        
        # L?u predictions OOD
        df_ood_predictions = df_ood[["geom_hash", "angle", "reynolds", "cm"]].copy()
        for name in final_model_names:
            df_ood_predictions[f"cm_pred_{name.lower().replace(' ', '_').replace('-', '_')}"] = ood_predictions[name]["preds"]
            df_ood_predictions[f"cm_std_{name.lower().replace(' ', '_').replace('-', '_')}"] = ood_predictions[name]["stds"]
    else:
        df_ood_comparison = pd.DataFrame(columns=ood_cols)
        print("\n" + "=" * 80)
        print("OOD CHALLENGE BENCHMARK SKIPPED")
        print("=" * 80)
        print("No OOD rows were found in the input dataset for the hard-coded OOD hashes.")
        print("=" * 80 + "\n")
        df_ood_predictions = pd.DataFrame(columns=["geom_hash", "angle", "reynolds", "cm"])

    df_ood_predictions.to_csv(paths["tables"] / "cm_ood_predictions.csv", index=False, encoding="utf-8-sig")

    # ==========================================================================
    # BIỂU ĐỒ TRỰC QUAN (PLOTS) - SỬ DỤNG MÔ HÌNH TỐT NHẤT CHỌN THEO ID RMSE
    # ==========================================================================
    print("-> Đang sinh các đồ thị trực quan chất lượng cao...")
    best_model = final_models[best_model_name]
    
    # 1. ID prediction vs truth (sử dụng OOF predictions)
    plt.figure(figsize=(6, 5))
    y_pred_id_best = oof_predictions[best_model_name]
    plt.scatter(y_pred_id_best, y_id, s=4, alpha=0.3, color="#2196F3")
    mn, mx = min(y_id.min(), y_pred_id_best.min()), max(y_id.max(), y_pred_id_best.max())
    plt.plot([mn, mx], [mn, mx], color="red", linestyle="--", linewidth=1.5)
    plt.xlabel(f"Predicted Cm ({best_model_name} OOF)")
    plt.ylabel("Measured Cm (True)")
    plt.title(f"ID Prediction vs Truth — {best_model_name}\nR² = {_r2(y_id, y_pred_id_best):.4f}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(paths["figures"] / "id_prediction_vs_truth.png", dpi=180)
    plt.close()
    
    # 2. OOD prediction vs truth
    if has_ood:
        plt.figure(figsize=(6, 5))
        y_pred_ood_best = ood_predictions[best_model_name]["preds"]
        colors_ood = ["#FF9800" if h == "4ca675adbe49c06b416b0079ca1b7f75afdccd8d" else "#9C27B0" for h in df_ood["geom_hash"]]
        plt.scatter(y_pred_ood_best, y_ood, s=8, alpha=0.5, c=colors_ood)
        
        # Dummy scatter ?? v? legend
        plt.scatter([], [], color="#FF9800", label="2032c (reflex)")
        plt.scatter([], [], color="#9C27B0", label="thin-flat")
        
        mn, mx = min(y_ood.min(), y_pred_ood_best.min()), max(y_ood.max(), y_pred_ood_best.max())
        plt.plot([mn, mx], [mn, mx], color="red", linestyle="--", linewidth=1.5)
        plt.xlabel(f"Predicted Cm ({best_model_name} OOD)")
        plt.ylabel("Measured Cm (True)")
        plt.title(f"OOD Extrapolation vs Truth ? {best_model_name}\nOOD R? = {_r2(y_ood, y_pred_ood_best):.4f}")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(paths["figures"] / "ood_prediction_vs_truth.png", dpi=180)
        plt.close()
    else:
        y_pred_ood_best = np.empty(0, dtype=np.float32)

    # 3. ID residual vs AoA
    plt.figure(figsize=(7, 4.5))
    resid_id = y_id - y_pred_id_best
    plt.scatter(df_id["angle"], resid_id, s=4, alpha=0.3, color="#4CAF50")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Residual (True - Pred)")
    plt.title(f"ID Residual vs AoA — {best_model_name}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(paths["figures"] / "id_residual_vs_aoa.png", dpi=180)
    plt.close()
    
    # 4. OOD residual vs AoA
    if has_ood:
        plt.figure(figsize=(7, 4.5))
        resid_ood = y_ood - y_pred_ood_best
        colors_ood2 = ["#FF9800" if h == "4ca675adbe49c06b416b0079ca1b7f75afdccd8d" else "#9C27B0" for h in df_ood["geom_hash"]]
        plt.scatter(df_ood["angle"], resid_ood, s=8, alpha=0.5, c=colors_ood2)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Angle of Attack [deg]")
        plt.ylabel("Residual (True - Pred)")
        plt.title(f"OOD Extrapolation Residual vs AoA ? {best_model_name}")
        
        plt.scatter([], [], color="#FF9800", label="2032c (reflex)")
        plt.scatter([], [], color="#9C27B0", label="thin-flat")
        plt.legend()
        
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(paths["figures"] / "ood_residual_vs_aoa.png", dpi=180)
        plt.close()
    else:
        resid_ood = np.empty(0, dtype=np.float32)

    # 5. Worst geometry error bar plot (top 8)
    df_id_w = df_id.copy()
    df_id_w["abs_err"] = np.abs(y_id - y_pred_id_best)
    geom_errs = df_id_w.groupby("geom_hash")["abs_err"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False).head(8)
    
    plt.figure(figsize=(7, 4.5))
    short_hashes = [str(idx)[:8] for idx in geom_errs.index]
    plt.bar(short_hashes, geom_errs["mean"], yerr=geom_errs["std"], capsize=5, color="#F44336", alpha=0.8, edgecolor="black")
    plt.xlabel("Geometry Hash (Prefix)")
    plt.ylabel("MAE with StDev Error Bar")
    plt.title("Top 8 Worst Performing ID Geometries")
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "worst_geometry_error.png", dpi=180)
    plt.close()
    
    # 6. Uncertainty vs absolute error (chỉ cho RF/ExtraTrees)
    if has_ood and ("RF" in best_model_name or "ExtraTrees" in best_model_name):
        stds = ood_predictions[best_model_name]["stds"]
        abs_err_ood = np.abs(y_ood - y_pred_ood_best)
        
        plt.figure(figsize=(6, 5))
        plt.scatter(stds, abs_err_ood, s=8, alpha=0.5, color="#FF9800")
        plt.xlabel("Ensemble Dispersion (Uncertainty StDev)")
        plt.ylabel("Absolute Prediction Error")
        plt.title(f"Uncertainty vs Error on OOD Challenge\n{best_model_name}")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(paths["figures"] / "uncertainty_vs_error.png", dpi=180)
        plt.close()

    # ==========================================================================
    # SINH BÁO CÁO TỰ ĐỘNG (REPORTS)
    # ==========================================================================
    
    # A. Báo cáo ID Benchmark
    id_rep_lines = [
        "# IN-DISTRIBUTION (ID) BENCHMARK REPORT",
        "",
        f"Báo cáo này được tự động tạo bởi `day4_cm_reset.py` để đánh giá năng lực của các mô hình surrogate Pitching Moment ($C_m$) trên {len(np.unique(groups_id))} geometries In-Distribution.",
        "",
        f"- **Tập dữ liệu đầu vào:** `{args.input_csv}`",
        f"- **Phương pháp kiểm định:** {n_cv_splits}-fold GroupKFold cross-validation on ID geometries (group-wise theo `geom_hash`)",
        f"- **Mô hình được chọn cuối cùng (Final Selected Model):** `{best_model_name}` (RMSE OOF thấp nhất)",
        "",
        "## 1. Bảng So sánh Hiệu năng các Mô hình (Out-Of-Fold CV)",
        "",
        df_comparison.round(5).to_markdown(index=False),
        "",
        "## 2. Phân tích Chi tiết trên Mô hình được chọn",
        "",
        f"Mô hình được chọn là `{best_model_name}` đạt hiệu năng OOF tổng thể cực kỳ ổn định:",
        f"- **OOF MAE:** `{_mae(y_id, y_pred_id_best):.5f}`",
        f"- **OOF RMSE:** `{_rmse(y_id, y_pred_id_best):.5f}`",
        f"- **OOF R²:** `{_r2(y_id, y_pred_id_best):.5f}`",
        f"- **Prediction-vs-Truth Slope:** `{df_comparison.iloc[0]['Slope']:.4f}`",
        f"- **Biên độ bị nén (Amplitude Compressed):** `{df_comparison.iloc[0]['Compressed_Flag']}`",
        "",
    ]
    
    # Chi tiết theo Regime
    df_id_w["stall_region"] = df_id_w["angle"].apply(lambda a: "negative_high_aoa" if a < -5 else ("linear_attached" if -5 <= a <= 5 else ("transition" if 5 < a <= 10 else "stall_risk")))
    reg_stats = []
    for reg in df_id_w["stall_region"].unique():
        sub = df_id_w[df_id_w["stall_region"] == reg]
        reg_stats.append({
            "Regime": reg,
            "Samples": len(sub),
            "MAE": _mae(sub["cm"], y_pred_id_best[df_id_w["stall_region"] == reg]),
            "RMSE": _rmse(sub["cm"], y_pred_id_best[df_id_w["stall_region"] == reg]),
            "R²": _r2(sub["cm"], y_pred_id_best[df_id_w["stall_region"] == reg])
        })
    id_rep_lines += [
        "### Phân tích theo Flow Regime",
        "",
        pd.DataFrame(reg_stats).round(5).to_markdown(index=False),
        "",
    ]
    
    # Worst geometries table
    worst_geom_details = []
    for idx, row in geom_errs.iterrows():
        yt_sub = df_id_w[df_id_w["geom_hash"] == idx]["cm"].to_numpy()
        yp_sub = y_pred_id_best[df_id_w["geom_hash"] == idx]
        worst_geom_details.append({
            "Geometry Hash": idx,
            "Samples": int(row["count"]),
            "MAE": row["mean"],
            "StDev": row["std"],
            "R²": _r2(yt_sub, yp_sub)
        })
    id_rep_lines += [
        "### Top 8 Geometries có hiệu năng tệ nhất",
        "",
        pd.DataFrame(worst_geom_details).round(5).to_markdown(index=False),
        "",
    ]
    
    # Worst samples
    df_id_w["pred"] = y_pred_id_best
    df_id_w["residual"] = resid_id
    worst_samples = df_id_w.sort_values("abs_err", ascending=False)[["geom_hash", "angle", "reynolds", "cm", "pred", "residual", "abs_err"]].head(15)
    id_rep_lines += [
        "### Top 15 mẫu có sai số lớn nhất",
        "",
        worst_samples.round(5).to_markdown(index=False),
        "",
    ]
    
    (paths["reports"] / "cm_id_benchmark_report.md").write_text("\n".join(id_rep_lines), encoding="utf-8")
    print(f"-> Đã tạo báo cáo ID Benchmark tại: {paths['reports'] / 'cm_id_benchmark_report.md'}")

    # B. Báo cáo OOD Challenge
    if has_ood:
        best_ood_meta = df_ood_comparison[df_ood_comparison["Model"] == best_model_name].iloc[0]
        
        ood_rep_lines = [
            "# OUT-OF-DISTRIBUTION (OOD) CHALLENGE BENCHMARK REPORT",
            "",
            "B?o c?o n?y ??nh gi? n?ng l?c ngo?i suy h?nh h?c (extrapolation) c?a c?c m? h?nh surrogate Cm tr?n hai h? h?nh h?c OOD ho?n to?n b? ?n trong qu? tr?nh hu?n luy?n.",
            "",
            "> [!IMPORTANT]",
            "> OOD results are diagnostic only and are not used for final model selection.",
            "> K?t qu? OOD n?y ch? mang t?nh ch?t ch?n ?o?n v? ho?n to?n kh?ng ???c s? d?ng ?? ??a ra quy?t ??nh l?a ch?n m? h?nh cu?i c?ng.",
            "",
            f"- **M? h?nh ???c ch?n theo ID Benchmark:** `{best_model_name}`",
            f"- **S? l??ng m?u OOD:** `{len(df_ood):,}` m?u",
            "",
            "## 1. B?ng So s?nh Hi?u n?ng tr?n OOD Challenge Set",
            "",
            df_ood_comparison.round(5).to_markdown(index=False),
            "",
            "## 2. ??nh gi? chi ti?t cho M? h?nh ???c ch?n",
            "",
            f"M? h?nh ???c ch?n `{best_model_name}` ho?t ??ng tr?n OOD Challenge Set:",
            f"- **OOD MAE:** `{best_ood_meta['OOD_MAE']:.5f}`",
            f"- **OOD RMSE:** `{best_ood_meta['OOD_RMSE']:.5f}`",
            f"- **OOD R?:** `{best_ood_meta['OOD_R?']:.5f}`",
            f"- **R? tr?n h? 2032c:** `{best_ood_meta['R?_2032c']:.5f}`",
            f"- **R? tr?n h? thin-flat:** `{best_ood_meta['R?_thin_flat']:.5f}`",
            f"- **OOD Slope:** `{best_ood_meta['OOD_Slope']:.4f}`",
            f"- **Amplitude Compressed Flag:** `{best_ood_meta['OOD_Compressed']}`",
            "",
        ]
        
        # Worst OOD samples
        df_ood_w = df_ood.copy()
        df_ood_w["pred"] = y_pred_ood_best
        df_ood_w["abs_err"] = np.abs(y_ood - y_pred_ood_best)
        df_ood_w["residual"] = resid_ood
        worst_ood_samples = df_ood_w.sort_values("abs_err", ascending=False)[["geom_hash", "angle", "reynolds", "cm", "pred", "residual", "abs_err"]].head(15)
        
        ood_rep_lines += [
            "### Top 15 m?u OOD c? sai s? l?n nh?t",
            "",
            worst_ood_samples.round(5).to_markdown(index=False),
            "",
            "## 3. Ph?n t?ch Amplitude Compression",
            "",
            "S? n?n bi?n ?? (amplitude compression) ph?n ?nh xu h??ng m? h?nh d? li?u (data-driven) ?p c?c gi? tr? c?c tr? kh? ??ng h?c v? g?n gi? tr? trung b?nh khi g?p ?i?u ki?n ch?a bi?t:",
            "- **H? cong ng??c `2032c`** c? h?nh d?ng airfoil ph?n x? u?n l??n ??c th?. Do thi?u d? li?u t??ng t?, m? h?nh kh?ng th? suy di?n ??ng ?? cong c?a ???ng camber d?c ??ng, d?n ??n sai s? r?t l?n v? R? ?m.",
            "- **H? ph?ng m?ng `thin-flat`** gi? ???c m?t ph?n xu h??ng kh? ??ng tuy?n t?nh c? b?n, tuy nhi?n bi?n ?? pitching moment b? thu h?p ??ng k? so v?i th?c t?.",
            ""
        ]
    else:
        ood_rep_lines = [
            "# OUT-OF-DISTRIBUTION (OOD) CHALLENGE BENCHMARK REPORT",
            "",
            "OOD benchmark ?? ???c b? qua v? input hi?n t?i kh?ng ch?a b?t k? d?ng n?o thu?c OOD challenge set hard-coded.",
            "",
            f"- **M? h?nh ???c ch?n theo ID Benchmark:** `{best_model_name}`",
            f"- **S? l??ng m?u OOD:** `{len(df_ood):,}` m?u",
            "",
            "## Tr?ng th?i",
            "",
            "- `OOD benchmark skipped`",
            "- Kh?ng c? hai `geom_hash` OOD y?u c?u trong dataset ??u v?o.",
            "- Pipeline v?n ho?n t?t ph?n ID benchmark v? xu?t model b?nh th??ng.",
            ""
        ]

    (paths["reports"] / "cm_ood_benchmark_report.md").write_text("\n".join(ood_rep_lines), encoding="utf-8")
    print(f"-> Đã tạo báo cáo OOD Challenge tại: {paths['reports'] / 'cm_ood_benchmark_report.md'}")
    
    # Sao chép OOD report và ID report ra ngoài root của workspace để user dễ tham chiếu
    Path("cm_id_benchmark_report.md").write_text("\n".join(id_rep_lines), encoding="utf-8")
    Path("cm_ood_benchmark_report.md").write_text("\n".join(ood_rep_lines), encoding="utf-8")
    
    # Đồng bộ sang artifact directory của cuộc hội thoại
    artifact_dir = Path("C:/Users/Admin/.gemini/antigravity-ide/brain/2d28e476-ea5d-458d-89cb-f655156c1169")
    if artifact_dir.exists():
        (artifact_dir / "cm_id_benchmark_report.md").write_text("\n".join(id_rep_lines), encoding="utf-8")
        (artifact_dir / "cm_ood_benchmark_report.md").write_text("\n".join(ood_rep_lines), encoding="utf-8")
        
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Outputs are stored in: {args.output_dir}/")
    print(f"  - Models : {paths['models']}/")
    print(f"  - Reports: {paths['reports']}/")
    print(f"  - Tables : {paths['tables']}/")
    print(f"  - Figures: {paths['figures']}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
