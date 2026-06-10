from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CD_PHYSICS_SETTINGS = {
    "cd_floor": 1e-7,
    "high_aoa_start_deg": 12.0,
    "attached_aoa_max_deg": 5.0,
    "post_stall_start_deg": 12.0,
    "min_curve_points": 4,
    "min_high_points": 3,
    "branch_drop_fraction": 0.20,
    "branch_slope_threshold": -0.006,
    "collapse_ratio": 0.50,
    "oscillation_threshold": 1.50,
    "curve_violation_rate_threshold": 0.40,
}

GEOMETRY_BASE_COLS = [
    "t_max", "x_tmax", "camber_max", "x_cmax", "te_gap", "le_radius_proxy",
    "trailing_edge_angle", "aft_thickness", "aft_camber", "curvature_energy_aft",
    "upper_curvature_energy_aft_06", "upper_aft_curvature_concentration",
    "slope_variance_aft", "thickness_gradient_aft", "thickness_gradient_abs_aft",
    "upper_aft_slope_change", "lower_aft_slope_change", "wake_proxy", "hysteresis_proxy",
    "aft_to_mid_thickness_ratio", "le_curvature_energy", "separation_proxy",
    "pressure_recovery_proxy", "aft_pressure_recovery_proxy", "aft_curvature_gradient",
    "aft_loading_metric", "te_camber_slope", "te_camber_curvature", "upper_curvature_aft",
    "curvature_aft", "leading_edge_radius", "slope_variance", "le_surface_oscillation",
    "te_surface_oscillation", "local_curvature_variance", "trailing_edge_quality_score",
    "camber_area", "camber_centroid_x", "camber_front_area", "camber_mid_area",
    "camber_aft_area", "camber_slope_front", "camber_slope_mid", "camber_slope_aft",
    "camber_curvature_front", "camber_curvature_mid", "camber_curvature_aft",
    "thickness_centroid_x", "thickness_front_area", "thickness_mid_area", "thickness_aft_area",
    "thickness_moment_arm", "zero_lift_moment_proxy", "pressure_center_proxy",
    "camber_pressure_proxy", "aft_loading_proxy", "moment_distribution_proxy",
]

NEW_SCORE_COLS = [
    "geometry_mahalanobis_score",
    "geometry_iforest_score",
    "geometry_outlier_score",
    "curve_flag_rate",
    "curve_mean_violation_rate",
    "curve_max_violation_rate",
    "curve_oscillation_score",
    "curve_outlier_score",
    "model_difficulty_score",
    "ensemble_outlier_score",
    "outlier_flag_soft",
    "outlier_flag_hard",
    "outlier_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process Day2 data with geometry/curve outlier scores.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--geometry_summary_csv", default=None)
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--name_col", default="name")
    parser.add_argument("--model_error_csv", default=None)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--max_rows", type=int, default=0, help="Optional smoke limit; 0 = full file")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def resolve_compression(path: Path) -> Optional[str]:
    return "gzip" if path.suffix.lower() == ".gz" else None


def discover_group_col(columns: Iterable[str], requested: Optional[str]) -> str:
    cols = set(columns)
    if requested:
        if requested not in cols:
            raise ValueError(f"Requested group_col '{requested}' not found")
        return requested
    for cand in ("geom_hash", "geometry_id", "name"):
        if cand in cols:
            return cand
    raise ValueError("Could not discover geometry group column. Expected one of: geom_hash, geometry_id, name")


def default_sidecar_path(output_csv: Path, suffix: str) -> Path:
    if output_csv.suffix.lower() == ".gz":
        stem = output_csv.name[:-3]
        return output_csv.with_name(stem.replace(".csv", suffix))
    return output_csv.with_name(output_csv.stem + suffix)


def find_default_model_error_csv() -> Optional[Path]:
    candidates = [
        Path(r"e:\Project2\outputs\day4c_cd_v3_final\tables\noisy_geometry_ranking.csv"),
        Path(r"e:\GKI\outputs\day4c_cd_v3_final\tables\noisy_geometry_ranking.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def percentile_rank01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    vals = arr[finite]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=float)
    if len(vals) == 1:
        ranks[order] = 0.5
    else:
        ranks[order] = np.linspace(0.0, 1.0, num=len(vals), endpoint=True)
    out[finite] = ranks
    return out


def ensure_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def choose_geometry_columns(all_columns: List[str]) -> List[str]:
    g_cols = sorted([c for c in all_columns if c.startswith("g_")])
    return [c for c in GEOMETRY_BASE_COLS if c in all_columns] + g_cols


def load_geometry_rows(input_path: Path, group_col: str, name_col: str, geometry_cols: List[str], chunksize: int, max_rows: int) -> pd.DataFrame:
    usecols = [group_col] + ([name_col] if name_col in geometry_cols or name_col != group_col else []) + geometry_cols
    usecols = list(dict.fromkeys([c for c in usecols if c]))
    seen: set[str] = set()
    rows: List[pd.DataFrame] = []
    total_rows = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize, low_memory=False, compression=resolve_compression(input_path)):
        if max_rows > 0:
            remaining = max_rows - total_rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        total_rows += len(chunk)
        keys = chunk[group_col].astype(str)
        keep_mask = ~keys.isin(seen)
        if keep_mask.any():
            kept = chunk.loc[keep_mask].copy()
            seen.update(kept[group_col].astype(str).tolist())
            rows.append(kept)
        if max_rows > 0 and total_rows >= max_rows:
            break
    if not rows:
        raise RuntimeError("No geometry rows were collected from input file")
    geom = pd.concat(rows, ignore_index=True)
    geom[group_col] = geom[group_col].astype(str)
    if name_col in geom.columns:
        geom[name_col] = geom[name_col].astype(str)
    return geom.drop_duplicates(subset=[group_col], keep="first").reset_index(drop=True)


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
    cfg = dict(CD_PHYSICS_SETTINGS)
    if settings:
        cfg.update(settings)
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


def compute_curve_summary(input_path: Path, group_col: str, name_col: str, chunksize: int, max_rows: int) -> pd.DataFrame:
    usecols = [c for c in [group_col, name_col, 'reynolds', 'angle', 'cd'] if c]
    frames: List[pd.DataFrame] = []
    total_rows = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize, low_memory=False, compression=resolve_compression(input_path)):
        if max_rows > 0:
            remaining = max_rows - total_rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        total_rows += len(chunk)
        frames.append(chunk)
        if max_rows > 0 and total_rows >= max_rows:
            break
    if not frames:
        raise RuntimeError('No curve rows loaded from input file')
    df = pd.concat(frames, ignore_index=True)
    df[group_col] = df[group_col].astype(str)
    if name_col in df.columns:
        df[name_col] = df[name_col].astype(str)
    df['angle'] = pd.to_numeric(df['angle'], errors='coerce')
    df['cd'] = pd.to_numeric(df['cd'], errors='coerce')
    if 'reynolds' in df.columns:
        df['reynolds'] = pd.to_numeric(df['reynolds'], errors='coerce')
    else:
        df['reynolds'] = np.nan
    rows: List[Dict[str, Any]] = []
    for (geom_key, reynolds), grp in df.groupby([group_col, 'reynolds'], sort=False):
        metrics = evaluate_cd_physics_curve(grp['angle'].to_numpy(float), grp['cd'].to_numpy(float), CD_PHYSICS_SETTINGS)
        rows.append({
            group_col: str(geom_key),
            'reynolds': reynolds,
            'n_points': int(len(grp)),
            **metrics,
        })
    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        return pd.DataFrame(columns=[group_col, 'curve_flag_rate', 'curve_mean_violation_rate', 'curve_max_violation_rate', 'curve_oscillation_score', 'curve_outlier_score'])
    agg = curve_df.groupby(group_col, sort=False).agg(
        n_curves=('reynolds', 'size'),
        curve_flag_rate=('cd_physics_flag', 'mean'),
        curve_mean_violation_rate=('violation_rate', 'mean'),
        curve_max_violation_rate=('violation_rate', 'max'),
        curve_oscillation_score=('cd_curve_oscillation_score', 'max'),
        negative_cd_curves=('negative_cd_count', lambda s: int(np.sum(np.asarray(s) > 0))),
        collapse_curves=('post_stall_cd_collapse_count', lambda s: int(np.sum(np.asarray(s) > 0))),
    ).reset_index()
    flag_rank = percentile_rank01(agg['curve_flag_rate'].to_numpy(float))
    mean_rank = percentile_rank01(agg['curve_mean_violation_rate'].to_numpy(float))
    max_rank = percentile_rank01(agg['curve_max_violation_rate'].to_numpy(float))
    osc_rank = percentile_rank01(agg['curve_oscillation_score'].to_numpy(float))
    score = 0.35 * np.nan_to_num(flag_rank, nan=0.0) + 0.25 * np.nan_to_num(mean_rank, nan=0.0) + 0.20 * np.nan_to_num(max_rank, nan=0.0) + 0.20 * np.nan_to_num(osc_rank, nan=0.0)
    hard_mask = (agg['negative_cd_curves'].to_numpy(int) > 0) | (agg['collapse_curves'].to_numpy(int) > 0)
    score = np.where(hard_mask, np.maximum(score, 0.98), score)
    agg['curve_outlier_score'] = score
    return agg


def compute_geometry_scores(geom_df: pd.DataFrame, group_col: str, geometry_cols: List[str], random_state: int) -> pd.DataFrame:
    available = [c for c in geometry_cols if c in geom_df.columns]
    X_df = ensure_numeric_frame(geom_df, available)
    keep_cols = [c for c in X_df.columns if X_df[c].notna().any()]
    X_df = X_df[keep_cols]
    if X_df.empty:
        raise RuntimeError('No usable geometry numeric columns were found')
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_df).astype(np.float32, copy=False)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32, copy=False)
    if Xs.shape[0] > 3 and Xs.shape[1] > 3:
        max_comp = min(32, Xs.shape[0] - 1, Xs.shape[1])
        pca = PCA(n_components=max_comp, svd_solver='full', random_state=random_state)
        Z_full = pca.fit_transform(Xs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cum, 0.99) + 1)
        keep = max(2, min(keep, Z_full.shape[1]))
        Z = Z_full[:, :keep].astype(np.float32, copy=False)
    else:
        Z = Xs
    mu = np.nanmean(Z, axis=0)
    cov = np.cov(Z, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float) + np.eye(cov.shape[0], dtype=float) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    D = Z.astype(float) - mu
    mahal = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', D, inv_cov, D), 0.0))
    iso = IsolationForest(n_estimators=300, contamination='auto', random_state=random_state, n_jobs=-1)
    iso.fit(Z)
    iso_raw = -iso.score_samples(Z)
    out = geom_df[[group_col]].copy()
    out['geometry_mahalanobis_raw'] = mahal
    out['geometry_iforest_raw'] = iso_raw
    out['geometry_mahalanobis_score'] = percentile_rank01(mahal)
    out['geometry_iforest_score'] = percentile_rank01(iso_raw)
    out['geometry_outlier_score'] = 0.5 * out['geometry_mahalanobis_score'].to_numpy(float) + 0.5 * out['geometry_iforest_score'].to_numpy(float)
    return out


def merge_model_difficulty(score_df: pd.DataFrame, group_col: str, model_error_csv: Optional[Path]) -> Tuple[pd.DataFrame, str]:
    out = score_df.copy()
    if model_error_csv is None or not model_error_csv.exists():
        out['model_difficulty_score'] = np.nan
        return out, 'missing'
    src = pd.read_csv(model_error_csv, low_memory=False)
    if group_col not in src.columns and 'geom_hash' in src.columns and group_col == 'geom_hash':
        pass
    elif group_col not in src.columns:
        out['model_difficulty_score'] = np.nan
        return out, f'not_found_in_{model_error_csv.name}'
    src[group_col] = src[group_col].astype(str)
    model_score = None
    if 'noise_score' in src.columns:
        model_score = pd.to_numeric(src['noise_score'], errors='coerce').to_numpy(float)
    else:
        parts = []
        for col in ('logo_mae', 'mae_pred', 'rmse_pred', 'pred_violation_rate', 'mean_uncertainty'):
            if col in src.columns:
                parts.append(percentile_rank01(pd.to_numeric(src[col], errors='coerce').to_numpy(float)))
        if parts:
            model_score = np.nanmean(np.vstack(parts), axis=0)
    if model_score is None:
        out['model_difficulty_score'] = np.nan
        return out, f'no_supported_score_in_{model_error_csv.name}'
    model_df = src[[group_col]].copy()
    model_df['model_difficulty_score'] = model_score
    out = out.merge(model_df.drop_duplicates(subset=[group_col]), on=group_col, how='left')
    if 'model_difficulty_score' not in out.columns:
        out['model_difficulty_score'] = np.nan
    return out, str(model_error_csv)


def add_flags(score_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = score_df.copy()
    geom = out['geometry_outlier_score'].to_numpy(float)
    curve = out['curve_outlier_score'].to_numpy(float)
    model = out['model_difficulty_score'].to_numpy(float)
    if np.isfinite(model).any():
        ensemble = 0.40 * np.nan_to_num(geom, nan=0.0) + 0.30 * np.nan_to_num(curve, nan=0.0) + 0.30 * np.nan_to_num(model, nan=0.0)
    else:
        ensemble = 0.60 * np.nan_to_num(geom, nan=0.0) + 0.40 * np.nan_to_num(curve, nan=0.0)
    out['ensemble_outlier_score'] = ensemble
    soft_thr = float(np.nanquantile(ensemble, 0.9)) if len(out) else 0.9
    hard_thr = float(np.nanquantile(ensemble, 0.98)) if len(out) else 0.98
    out['outlier_flag_soft'] = (
        (out['ensemble_outlier_score'] >= soft_thr) |
        (out['geometry_outlier_score'] >= 0.98) |
        (out['curve_outlier_score'] >= 0.98) |
        (out.get('curve_flag_rate', pd.Series(np.zeros(len(out)))) >= 0.50)
    )
    out['outlier_flag_hard'] = (
        (out['ensemble_outlier_score'] >= hard_thr) |
        (out.get('negative_cd_curves', pd.Series(np.zeros(len(out), dtype=int))) > 0) |
        (out.get('collapse_curves', pd.Series(np.zeros(len(out), dtype=int))) > 0)
    )
    reasons = []
    for row in out.itertuples(index=False):
        parts: List[str] = []
        if getattr(row, 'geometry_outlier_score', np.nan) >= 0.98:
            parts.append('geometry_extreme')
        if getattr(row, 'curve_outlier_score', np.nan) >= 0.98:
            parts.append('curve_extreme')
        if getattr(row, 'negative_cd_curves', 0) > 0:
            parts.append('negative_cd_curve')
        if getattr(row, 'collapse_curves', 0) > 0:
            parts.append('post_stall_collapse')
        if np.isfinite(getattr(row, 'model_difficulty_score', np.nan)) and getattr(row, 'model_difficulty_score', np.nan) >= 0.95:
            parts.append('model_difficulty_high')
        reasons.append(';'.join(parts) if parts else 'none')
    out['outlier_reason'] = reasons
    return out, {'soft_threshold': soft_thr, 'hard_threshold': hard_thr}


def write_augmented_dataset(input_path: Path, output_path: Path, group_col: str, score_df: pd.DataFrame, chunksize: int, max_rows: int) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output paths must be different')
    score_cols = [group_col] + [c for c in NEW_SCORE_COLS if c in score_df.columns]
    score_map = score_df[score_cols].copy()
    score_map[group_col] = score_map[group_col].astype(str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    total_rows = 0
    header = True
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False, compression=resolve_compression(input_path)):
        if max_rows > 0:
            remaining = max_rows - total_rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        total_rows += len(chunk)
        chunk[group_col] = chunk[group_col].astype(str)
        merged = chunk.merge(score_map, on=group_col, how='left')
        merged.to_csv(output_path, mode='a', index=False, header=header, compression=resolve_compression(output_path))
        header = False
        if max_rows > 0 and total_rows >= max_rows:
            break
    return total_rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')
    probe = pd.read_csv(input_path, nrows=2, low_memory=False, compression=resolve_compression(input_path))
    columns = probe.columns.tolist()
    group_col = discover_group_col(columns, args.group_col)
    name_col = args.name_col if args.name_col in columns else group_col
    geometry_cols = choose_geometry_columns(columns)
    if not geometry_cols:
        raise RuntimeError('No geometry descriptor columns found in input file')
    print(f'[OUTLIER] group_col={group_col} name_col={name_col} geometry_cols={len(geometry_cols)}')

    geom_rows = load_geometry_rows(input_path, group_col, name_col, geometry_cols, args.chunksize, args.max_rows)
    geom_scores = compute_geometry_scores(geom_rows, group_col, geometry_cols, args.random_state)
    print(f'[OUTLIER] geometry groups={len(geom_scores)}')

    curve_summary = compute_curve_summary(input_path, group_col, name_col, args.chunksize, args.max_rows)
    print(f'[OUTLIER] curve groups={len(curve_summary)}')

    score_df = geom_rows[[group_col] + ([name_col] if name_col in geom_rows.columns else [])].copy()
    score_df[group_col] = score_df[group_col].astype(str)
    score_df = score_df.merge(geom_scores, on=group_col, how='left')
    score_df = score_df.merge(curve_summary, on=group_col, how='left')

    model_error_csv = Path(args.model_error_csv) if args.model_error_csv else find_default_model_error_csv()
    score_df, model_source = merge_model_difficulty(score_df, group_col, model_error_csv)
    score_df, thresholds = add_flags(score_df)

    geometry_summary_csv = Path(args.geometry_summary_csv) if args.geometry_summary_csv else default_sidecar_path(output_path, '_geometry_outlier_summary.csv')
    report_json = Path(args.report_json) if args.report_json else default_sidecar_path(output_path, '_outlier_report.json')
    geometry_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    geometry_summary_cols = [c for c in score_df.columns if c not in geometry_cols]
    summary_export = score_df[geometry_summary_cols].copy()
    summary_export.to_csv(geometry_summary_csv, index=False)

    written_rows = write_augmented_dataset(input_path, output_path, group_col, score_df, args.chunksize, args.max_rows)

    report = {
        'input_csv': str(input_path),
        'output_csv': str(output_path),
        'geometry_summary_csv': str(geometry_summary_csv),
        'group_col': group_col,
        'name_col': name_col,
        'n_geometry_groups': int(len(score_df)),
        'written_rows': int(written_rows),
        'model_error_source': model_source,
        'soft_threshold': thresholds['soft_threshold'],
        'hard_threshold': thresholds['hard_threshold'],
        'soft_flag_count': int(summary_export['outlier_flag_soft'].fillna(False).sum()),
        'hard_flag_count': int(summary_export['outlier_flag_hard'].fillna(False).sum()),
        'top10_hard': summary_export.sort_values('ensemble_outlier_score', ascending=False).head(10)[[c for c in [group_col, name_col, 'ensemble_outlier_score', 'geometry_outlier_score', 'curve_outlier_score', 'model_difficulty_score', 'outlier_reason'] if c in summary_export.columns]].to_dict(orient='records'),
    }
    report_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(f'[OUTLIER] wrote row-level output: {output_path}')
    print(f'[OUTLIER] wrote geometry summary: {geometry_summary_csv}')
    print(f'[OUTLIER] wrote report: {report_json}')
    print(f'[OUTLIER] soft_flag_count={report["soft_flag_count"]} hard_flag_count={report["hard_flag_count"]}')


if __name__ == '__main__':
    main()
