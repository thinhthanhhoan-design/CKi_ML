from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    xgb = None
    HAS_XGB = False

# Configuration flags to control post-calibration residual experts
ENABLE_LOW_CD_EXPERT = False
ENABLE_HIGH_DRAG_EXPERT = False
ENABLE_REGIME_MICRO_CAL = True


# -----------------------------------------------------------------------------
# Mathematical & Geometry Helpers
# -----------------------------------------------------------------------------

def smoothstep(t: Any) -> Any:
    t_cl = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t_cl * t_cl * (3.0 - 2.0 * t_cl)


def cosine_grid(n: int) -> np.ndarray:
    """Symmetric cosine grid densified at LE/TE, running from LE to TE."""
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standard numerical curvature computation."""
    dy = np.gradient(y, x, axis=1, edge_order=1)
    d2y = np.gradient(dy, x, axis=1, edge_order=1)
    return d2y / np.maximum((1.0 + dy**2)**1.5, 1e-12)


def add_geometry_descriptors(df: pd.DataFrame, g_cols: List[str]) -> List[str]:
    """
    Computes all 31 custom geometry descriptors from airfoil coordinates.
    Modifies DataFrame in-place and returns the list of generated feature names.
    Fully source-visible and identical to the decompiled bytecode implementation.
    """
    n = len(g_cols) // 2
    xg = cosine_grid(n).astype(np.float32)
    pos = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    pos_idx = {p: np.argmin(np.abs(xg - p)) for p in pos}
    
    # Extract coordinate branches
    arr = df[g_cols].to_numpy(dtype=np.float32, copy=True)
    yu = arr[:, :n]
    yl = arr[:, n:]
    
    thickness = yu - yl
    swap = np.mean(thickness, axis=1) < 0.0
    if np.any(swap):
        yu[swap], yl[swap] = yl[swap].copy(), yu[swap].copy()
        thickness = yu - yl
        
    camber = 0.5 * (yu + yl)
    thickness_pos = np.maximum(thickness, 0.0)
    abs_camber = np.abs(camber)
    
    dyu_dx = np.gradient(yu, xg, axis=1, edge_order=1)
    dyl_dx = np.gradient(yl, xg, axis=1, edge_order=1)
    dt_dx = np.gradient(thickness, xg, axis=1, edge_order=1)
    dcamber_dx = np.gradient(camber, xg, axis=1, edge_order=1)
    
    ku = curvature(xg, yu)
    kl = curvature(xg, yl)
    
    # Masks
    aft06_mask = (xg >= 0.6)
    aft_mask = (xg >= 0.7)
    tail_mask = (xg >= 0.8)
    le_mask = (xg <= 0.06)
    
    angle = df['angle'].to_numpy(dtype=np.float32)
    reynolds = df['reynolds'].to_numpy(dtype=np.float32)
    log_re = np.maximum(np.log10(np.maximum(reynolds, 1.0)), 1.0)
    abs_angle = np.abs(angle)
    
    out = {}
    
    # 1. Trailing Edge Wedge Angle
    te_u = np.mean(dyu_dx[:, -4:], axis=1)
    te_l = np.mean(dyl_dx[:, -4:], axis=1)
    te_wedge = np.abs(np.arctan(te_u) - np.arctan(te_l))
    out['trailing_edge_angle'] = te_wedge
    out['te_wedge_angle'] = te_wedge
    
    # 2. Aft Thickness & Camber
    out['aft_thickness'] = np.mean(thickness_pos[:, aft_mask], axis=1)
    out['aft_camber'] = np.mean(camber[:, aft_mask], axis=1)
    
    # 3. Curvature Energy / Concentration
    out['upper_curvature_energy_aft_06'] = np.mean(ku[:, aft06_mask] ** 2, axis=1)
    mean_ku_tail = np.mean(np.abs(ku[:, tail_mask]), axis=1)
    mean_ku_aft06 = np.mean(np.abs(ku[:, aft06_mask]), axis=1)
    out['upper_aft_curvature_concentration'] = mean_ku_tail / np.maximum(mean_ku_aft06, 1e-6)
    
    # 4. Slope Variance & Gradients
    out['slope_variance'] = np.var(dyu_dx[:, aft_mask], axis=1) + np.var(dyl_dx[:, aft_mask], axis=1)
    out['thickness_gradient'] = np.mean(np.abs(dt_dx[:, tail_mask]), axis=1)
    out['aft_camber_gradient'] = np.mean(np.abs(np.gradient(camber[:, aft_mask], xg[aft_mask], axis=1, edge_order=1)), axis=1)
    
    curv_mix = 0.5 * (ku + kl)
    out['aft_curvature_gradient'] = np.mean(np.abs(np.gradient(curv_mix[:, aft_mask], xg[aft_mask], axis=1, edge_order=1)), axis=1)
    
    # 5. Slope change
    out['upper_aft_slope_change'] = np.mean(np.abs(np.gradient(dyu_dx[:, aft_mask], xg[aft_mask], axis=1, edge_order=1)), axis=1)
    out['lower_aft_slope_change'] = np.mean(np.abs(np.gradient(dyl_dx[:, aft_mask], xg[aft_mask], axis=1, edge_order=1)), axis=1)
    
    # 6. Pressure Recovery Proxy
    aft_slope_change = np.mean(np.abs(np.gradient(dcamber_dx[:, aft_mask], xg[aft_mask], axis=1, edge_order=1)), axis=1)
    out['aft_pressure_recovery_proxy'] = (
        out['thickness_gradient']
        + out['aft_camber_gradient']
        + out['aft_curvature_gradient']
        + 0.5 * (out['upper_aft_slope_change'] + out['lower_aft_slope_change'])
        + te_wedge
    )
    out['pressure_recovery_proxy'] = out['aft_pressure_recovery_proxy'] + aft_slope_change
    
    # 7. Angle & Reynolds Dependent proxies
    out['curvature_aft'] = np.mean(ku[:, aft_mask] ** 2, axis=1) + np.mean(kl[:, aft_mask] ** 2, axis=1)
    
    # te_curv_concentration:
    mean_tail_curv = np.mean(np.abs(ku[:, tail_mask]) + np.abs(kl[:, tail_mask]), axis=1)
    mean_aft_curv = np.mean(np.abs(ku[:, aft_mask]) + np.abs(kl[:, aft_mask]), axis=1)
    te_curv_concentration = mean_tail_curv / np.maximum(mean_aft_curv, 1e-6)
    out['te_curvature_concentration'] = te_curv_concentration
    
    # Recovery asymmetry and imbalance
    upper_recovery = out['upper_aft_slope_change']
    lower_recovery = out['lower_aft_slope_change']
    out['recovery_asymmetry'] = np.abs(upper_recovery - lower_recovery)
    out['upper_lower_recovery_imbalance'] = out['recovery_asymmetry'] / np.maximum(upper_recovery + lower_recovery, 1e-6)
    
    # Stall severity
    te_wedge_local = te_wedge
    aft_thick = out['aft_thickness']
    aft_recovery = out['aft_pressure_recovery_proxy']
    curv_aft = out['curvature_aft']
    
    aft_adverse_gradient_strength = aft_recovery * np.maximum(te_wedge_local, 1e-6) * np.maximum(aft_thick, 1e-6)
    out['aft_adverse_gradient_strength'] = aft_adverse_gradient_strength
    
    # Additional stall/wake variables
    out['signed_upper_recovery_proxy'] = angle * out['upper_aft_slope_change']
    out['signed_lower_recovery_proxy'] = angle * out['lower_aft_slope_change']
    
    out['wake_proxy'] = abs_angle * out['aft_pressure_recovery_proxy'] * np.maximum(te_wedge, 1e-06)
    out['hysteresis_proxy'] = out['upper_aft_curvature_concentration'] * out['aft_pressure_recovery_proxy'] * np.maximum(te_wedge, 1e-06)
    out['aoa_x_hysteresis_proxy'] = abs_angle * out['hysteresis_proxy']
    
    stall_severity = np.power(abs_angle, 1.5) * out['aft_adverse_gradient_strength'] * (1.0 + out['upper_lower_recovery_imbalance']) * np.maximum(curv_aft, 1e-06) / np.maximum(log_re, 1.0)
    out['stall_severity_proxy'] = stall_severity
    out['aoa_x_stall_severity'] = abs_angle * stall_severity
    
    abs_angle_nl = np.power(abs_angle, 1.3)
    out['separation_proxy'] = abs_angle_nl * np.maximum(curv_aft, 1e-09) * np.maximum(out['aft_thickness'], 1e-09) * np.maximum(te_wedge, 1e-09) / np.maximum(log_re, 1.0)
    
    # LE & Centroids
    le_slope = np.gradient(thickness_pos[:, le_mask], xg[le_mask], axis=1, edge_order=1)
    out['leading_edge_radius'] = 1.0 / np.maximum(np.mean(np.abs(le_slope), axis=1), 1e-6)
    
    cmass = np.sum(abs_camber, axis=1)
    out['camber_centroid_x'] = np.divide(np.sum(abs_camber * xg.reshape(1, -1), axis=1), np.maximum(cmass, 1e-9))
    out['max_thickness_x'] = xg[np.argmax(thickness_pos, axis=1)]
    out['max_camber_x'] = xg[np.argmax(abs_camber, axis=1)]
    
    # Pos-based thickness/camber features
    for p, idx in pos_idx.items():
        out[f'thickness_x_{int(p*100):03d}'] = thickness_pos[:, idx]
        out[f'camber_x_{int(p*100):03d}'] = camber[:, idx]
        
    # Populate into DataFrame
    features_list = sorted(list(out.keys()))
    for name in features_list:
        df[name] = out[name].astype(np.float32)
        
    return features_list


def build_features(df: pd.DataFrame, g_cols: List[str], latent_cols: List[str]) -> None:
    """Wrapper to maintain compatibility with original day4c_cd_only bytecode signature."""
    add_geometry_descriptors(df, g_cols)


# -----------------------------------------------------------------------------
# Blending & Regime Logic
# -----------------------------------------------------------------------------

def regime_id_from_angle(angle: float) -> int:
    a = abs(float(angle))
    eps = 1e-9
    if a < 5.0:
        return 1
    if a < 8.0:
        return 2
    if a < 10.0:
        return 3
    if a < 12.0:
        return 4
    if a < 16.0 - eps:
        return 5
    return 6


def blend_weights_for_angle(angle: float, width: float | Dict[float, float]) -> List[Tuple[int, float]]:
    """
    Computes smooth transition weights across 5 dynamic regime boundaries:
    [5.0, 8.0, 10.0, 12.0, 16.0] deg AoA using smoothstep.
    """
    a = abs(float(angle))
    if isinstance(width, dict):
        width_map = {float(k): max(float(v), 1e-9) for k, v in width.items()}
    else:
        w = max(float(width), 1e-9)
        width_map = {b: w for b in [5.0, 8.0, 10.0, 12.0, 16.0]}
    
    boundaries = [(5.0, 1, 2), (8.0, 2, 3), (10.0, 3, 4), (12.0, 4, 5), (16.0, 5, 6)]
    for boundary, left_id, right_id in boundaries:
        w = width_map.get(float(boundary), max(float(width_map.get(12.0, 0.4)), 1e-9))
        if boundary - w <= a <= boundary + w:
            t = (a - (boundary - w)) / (2.0 * w)
            wr = smoothstep(t)
            return [(left_id, 1.0 - wr), (right_id, wr)]
            
    if a < 5.0:
        return [(1, 1.0)]
    elif a < 8.0:
        return [(2, 1.0)]
    elif a < 10.0:
        return [(3, 1.0)]
    elif a < 12.0:
        return [(4, 1.0)]
    elif a < 16.0:
        return [(5, 1.0)]
    else:
        return [(6, 1.0)]


def inverse_cd_transform(value: float, meta: Dict[str, Any]) -> float:
    transform = str(meta.get("target_transform", {}).get("type", "log")).lower().strip()
    floor = float(meta.get("target_transform", {}).get("floor", 1e-6))
    scale = float(meta.get("target_transform", {}).get("scale", 40.0))
    v = float(value)
    if not np.isfinite(v):
        return float("nan")
    if transform == "sqrt":
        return float(max(v, 0.0) ** 2)
    if transform == "log1p_scaled":
        return float(max(np.expm1(v) / max(scale, 1e-9), floor))
    if transform == "piecewise":
        threshold = float(meta.get("target_transform", {}).get("piecewise_threshold", 0.08))
        high_scale = max(float(meta.get("target_transform", {}).get("piecewise_high_scale", 8.0)), 1e-9)
        boundary = float(np.log1p(scale * threshold))
        if v < boundary:
            return float(max(np.expm1(v) / max(scale, 1e-9), floor))
        return float(max(threshold + (v - boundary) / high_scale, floor))
    if transform in {"identity", "raw", "none"}:
        return float(max(v, floor))
    return float(max(np.exp(v), floor))


def _stall_risk_gate(df: pd.DataFrame, reference: float | None = None) -> np.ndarray:
    if "stall_severity_proxy" not in df.columns:
        return np.zeros(len(df), dtype=float)
    sev = pd.to_numeric(df["stall_severity_proxy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if reference is None or not np.isfinite(reference) or reference <= 1e-12:
        finite = sev[np.isfinite(sev)]
        reference = float(np.nanpercentile(finite, 90)) if len(finite) else 1.0
    reference = max(float(reference), 1e-12)
    return smoothstep(sev / reference)


def _stall_onset_gate(df: pd.DataFrame, meta_config: Dict[str, Any], risk_reference: float | None = None) -> np.ndarray:
    angle = np.abs(pd.to_numeric(df["angle"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    aoa_min = float(meta_config.get("aoa_min", 8.0))
    stall_gate = smoothstep((angle - aoa_min) / 3.0)
    stall_gate = np.where(angle < aoa_min, 0.0, stall_gate)
    risk_gate = _stall_risk_gate(df, risk_reference)
    gate = stall_gate * (0.5 + 0.5 * risk_gate)
    
    if "density_weight" in df.columns:
        density_weight = pd.to_numeric(df["density_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        gate *= (0.8 + 0.4 * density_weight)
        
    return gate


# -----------------------------------------------------------------------------
# Clean, Independent Predictor Class
# -----------------------------------------------------------------------------

class Day4CCDOnlyPredictor:
    """
    Pure Python source-visible implementation of the Day4C Drag-Tail specialist.
    V2: Applies post-training calibration layers (LowCdExpert, RegimeCalibration,
    HighDragSafety, ConformalIntervals) if their artifact files exist.
    Removes all pyc dependencies and validates checksum/metadata safety.
    """
    def __init__(self, model_dir: str | Path) -> None:
        if not HAS_XGB:
            raise RuntimeError("xgboost is required to load Day4C Cd-only models.")
            
        model_dir = Path(model_dir)
        self.model_dir = model_dir   # stored for V2 layer lazy-loading
        meta_path = model_dir / "day4c_cd_only_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Day4C metadata sidecar file not found: {meta_path}")


        # Path & version safety validation
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Corrupted or invalid JSON in metadata: {meta_path}. Error: {e}")
            
        required_keys = ["feature_columns", "models", "pca_info"]
        for k in required_keys:
            if k not in self.meta:
                raise KeyError(f"Metadata file '{meta_path.name}' is missing expected key: {k}")

        self.feature_cols = [str(c) for c in self.meta["feature_columns"]]
        self.transition_width = self.meta.get("transition_widths_deg", self.meta.get("transition_width_deg", 0.4))
        self.cd_floor = float(self.meta.get("target_transform", {}).get("floor", 1e-6))
        
        # Load PCA mappings
        pca_scaler_path = model_dir / "geometry_pca_scaler.pkl"
        pca_model_path = model_dir / "geometry_pca_model.pkl"
        if not pca_scaler_path.exists() or not pca_model_path.exists():
            raise FileNotFoundError(f"Missing PCA model mappings at {model_dir}")
            
        self.pca_scaler = joblib.load(pca_scaler_path)
        self.pca_model = joblib.load(pca_model_path)
        
        # Verify PCA dimensionality against metadata
        self.n_latent = int(self.meta.get("pca_info", {}).get("n_components", 0))
        if self.n_latent <= 0:
            self.n_latent = len([c for c in self.feature_cols if c.startswith("z_")])
        if self.pca_model.n_components_ != self.n_latent:
            raise ValueError(
                f"PCA Dimensionality Mismatch! Meta expects {self.n_latent}, "
                f"but loaded model has {self.pca_model.n_components_} components."
            )
            
        self.latent_cols = [f"z_{i:03d}" for i in range(self.n_latent)]
        self.models: Dict[int, Dict[str, Any]] = {}

        # Safe loading of regime experts
        for rid_raw, rec in self.meta.get("models", {}).items():
            rid = int(rid_raw)
            name = str(rec.get("name", f"r{rid}"))
            prefix = f"regime_{rid}_{name}"
            
            # Load required XGB files
            base_model = xgb.XGBRegressor()
            residual_model = xgb.XGBRegressor()
            uncertainty_model = xgb.XGBRegressor()
            
            base_model.load_model(str(model_dir / f"{prefix}_base.json"))
            residual_model.load_model(str(model_dir / f"{prefix}_residual.json"))
            uncertainty_model.load_model(str(model_dir / f"{prefix}_uncertainty.json"))
            
            # Load optional experts if they exist
            raw_correction_model = None
            raw_path = model_dir / f"{prefix}_raw_correction.json"
            if raw_path.exists():
                raw_correction_model = xgb.XGBRegressor()
                raw_correction_model.load_model(str(raw_path))
                
            high_cd_model = None
            high_path = model_dir / f"{prefix}_high_cd.json"
            if high_path.exists():
                high_cd_model = xgb.XGBRegressor()
                high_cd_model.load_model(str(high_path))
                
            high_tail_residual_model = None
            tail_path = model_dir / f"{prefix}_high_tail_residual.json"
            if tail_path.exists():
                high_tail_residual_model = xgb.XGBRegressor()
                high_tail_residual_model.load_model(str(tail_path))
                
            stall_onset_residual_model = None
            stall_path = model_dir / f"{prefix}_stall_onset_residual.json"
            if stall_path.exists():
                stall_onset_residual_model = xgb.XGBRegressor()
                stall_onset_residual_model.load_model(str(stall_path))

            calibration = {"enabled": False, "slope": 1.0, "intercept": 0.0}
            meta_rec_path = model_dir / f"{prefix}_meta.json"
            if meta_rec_path.exists():
                try:
                    meta_rec = json.loads(meta_rec_path.read_text(encoding="utf-8"))
                    calibration = meta_rec.get("calibration", calibration)
                except Exception as e:
                    warnings.warn(f"Warning: could not read calibration meta for {prefix}: {e}")

            self.models[rid] = {
                "base_model": base_model,
                "residual_model": residual_model,
                "uncertainty_model": uncertainty_model,
                "raw_correction_model": raw_correction_model,
                "high_cd_model": high_cd_model,
                "high_tail_residual_model": high_tail_residual_model,
                "stall_onset_residual_model": stall_onset_residual_model,
                "imputer": joblib.load(model_dir / f"{prefix}_imputer.pkl"),
                "scaler": joblib.load(model_dir / f"{prefix}_scaler.pkl"),
                "calibration": calibration,
            }

    def align_row(self, row: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int, str]:
        """Aligns input geometry row to matches XGBoost feature order and PCA loadings."""
        aligned = row.copy()
        g_cols = sorted([c for c in aligned.columns if c.startswith("g_")], key=lambda c: int(c.split("_")[1]))
        if not g_cols:
            raise ValueError("Missing g_ columns before predictor alignment.")
            
        geo = aligned[g_cols].to_numpy(dtype=np.float32, copy=True)
        # Apply PCA
        latent = self.pca_model.transform(self.pca_scaler.transform(geo))
        for i, col in enumerate(self.latent_cols):
            aligned[col] = float(latent[0, i])
            
        build_features(aligned, g_cols, self.latent_cols)

        # Handle any missing feature columns safely
        missing = [col for col in self.feature_cols if col not in aligned.columns]
        for col in missing:
            aligned[col] = 0.0
            
        feature_view = aligned[self.feature_cols].copy()
        missing_list = ";".join(missing[:30])
        return aligned, feature_view, len(missing), missing_list

    def predict_one_regime(self, rid: int, X_raw: pd.DataFrame) -> Tuple[float, float]:
        """Runs prediction on a single regime expert with all correction branches."""
        art = self.models[int(rid)]
        X_imp = pd.DataFrame(art["imputer"].transform(X_raw[self.feature_cols]), columns=self.feature_cols)
        X = pd.DataFrame(art["scaler"].transform(X_imp), columns=self.feature_cols)
        
        # 1. Base prediction
        base_t = float(np.ravel(art["base_model"].predict(X))[0])
        resid_t = float(np.ravel(art["residual_model"].predict(X))[0])
        y_t = base_t + resid_t
        
        # 2. Linear low-angle calibration if defined
        cal = art.get("calibration", {})
        if bool(cal.get("enabled", False)):
            y_t = float(cal.get("slope", 1.0)) * y_t + float(cal.get("intercept", 0.0))
            
        cd = float(max(inverse_cd_transform(y_t, self.meta), self.cd_floor))
        
        # 3. Raw correction model
        raw_model = art.get("raw_correction_model")
        if raw_model is not None:
            cd = float(max(cd + float(np.ravel(raw_model.predict(X))[0]), self.cd_floor))
            
        # 4. High Cd expert gate
        high_model = art.get("high_cd_model")
        if high_model is not None:
            high_cfg = self.meta.get("high_cd_expert", {})
            threshold = float(high_cfg.get("threshold", 0.08))
            transition = max(float(high_cfg.get("transition", 0.02)), 1e-9)
            blend_max = float(high_cfg.get("blend_max", 0.70))
            t = float(np.clip((cd - threshold) / transition, 0.0, 1.0))
            gate = blend_max * t * t * (3.0 - 2.0 * t)
            high_cd = float(max(float(np.ravel(high_model.predict(X))[0]), self.cd_floor))
            cd = float(max((1.0 - gate) * cd + gate * high_cd, self.cd_floor))
            
        # 5. High tail residual corrections
        tail_model = art.get("high_tail_residual_model")
        if tail_model is not None:
            tail_resid = float(np.ravel(tail_model.predict(X))[0])
            angle = float(X_raw["angle"].iloc[0])
            a = abs(angle)
            aoa_gate = float(np.clip((a - 8.0) / 6.0, 0.0, 1.0))
            aoa_gate = aoa_gate * aoa_gate * (3.0 - 2.0 * aoa_gate)
            high_cfg = self.meta.get("high_cd_expert", {})
            tail_cfg = self.meta.get("high_tail_residual", {})
            threshold = float(high_cfg.get("threshold", 0.08))
            transition = max(float(high_cfg.get("transition", 0.02)), 1e-9)
            blend_max = float(high_cfg.get("blend_max", 0.70))
            tail_aoa_gate_min = float(tail_cfg.get("tail_aoa_gate_min", 0.35))
            t = float(np.clip((cd - threshold) / transition, 0.0, 1.0))
            cd_gate = blend_max * t * t * (3.0 - 2.0 * t)
            tail_gate = min(blend_max, max(cd_gate, tail_aoa_gate_min * aoa_gate))
            cd = float(max(cd + tail_gate * tail_resid, self.cd_floor))
            
        # 6. Stall onset residual model
        stall_model = art.get("stall_onset_residual_model")
        if stall_model is not None:
            stall_cfg = self.meta.get("stall_onset_residual", {})
            if bool(stall_cfg.get("enabled", True)):
                stall_resid = float(np.ravel(stall_model.predict(X))[0])
                angle = float(X_raw["angle"].iloc[0])
                a = abs(angle)
                aoa_min = float(stall_cfg.get("aoa_min", 8.0))
                gain = float(stall_cfg.get("gate_gain", 0.75))
                gate = _stall_onset_gate(X_raw, stall_cfg, art.get("stall_onset_residual_risk_reference"))
                cd = float(max(cd + gain * gate * stall_resid, self.cd_floor))
                
        # 7. Uncertainty prediction
        unc = float(max(float(np.ravel(art["uncertainty_model"].predict(X))[0]), 0.0))
        return cd, unc

    def predict_from_feature_view(self, X_raw: pd.DataFrame, alpha: float) -> Dict[str, Any]:
        """Predicts Cd and Uncertainty using regime weights blending and linear low-angle calibration.
        V2: applies LowCdExpert, RegimeCalibration, HighDragSafety, and ConformalIntervals if available.
        """
        pred_by_regime: Dict[int, float] = {}
        unc_by_regime: Dict[int, float] = {}
        for rid in sorted(self.models):
            cd, unc = self.predict_one_regime(rid, X_raw)
            pred_by_regime[rid] = cd
            unc_by_regime[rid] = unc

        hard_regime = regime_id_from_angle(alpha)
        hard = pred_by_regime.get(hard_regime, np.nan)
        hard_unc = unc_by_regime.get(hard_regime, np.nan)
        
        # Compute smooth regime transition weights
        weights = [
            (rid, w)
            for rid, w in blend_weights_for_angle(float(alpha), self.transition_width)
            if rid in pred_by_regime
        ]
        denom = sum(w for _, w in weights)
        if denom <= 0.0:
            blended = hard
            blended_unc = hard_unc
            desc = f"R{hard_regime}:1.000"
        else:
            blended = 0.0
            blended_unc = 0.0
            parts: List[str] = []
            for rid, weight in weights:
                wn = float(weight) / denom
                blended += wn * pred_by_regime[rid]
                blended_unc += wn * unc_by_regime[rid]
                parts.append(f"R{rid}:{wn:.3f}")
            desc = ";".join(parts)

        use_hard_high_aoa = bool(self.meta.get("high_aoa_use_hard_prediction", False))
        selected = hard if use_hard_high_aoa and abs(float(alpha)) >= 8.0 else blended
        selected_unc = hard_unc if use_hard_high_aoa and abs(float(alpha)) >= 8.0 else blended_unc

        # Global systematic bias calibration at low angles (abs(AoA) < 8.0)
        if abs(float(alpha)) < 8.0:
            slope = 1.0429
            intercept = -0.002187
            selected = max(slope * selected + intercept, self.cd_floor)
            blended = max(slope * blended + intercept, self.cd_floor)
            hard = max(slope * hard + intercept, self.cd_floor)

        # ── V2 Post-training corrections ──────────────────────────────────────
        # Load lazily and apply if artifacts exist in model_dir.
        cd_selected = float(selected)
        v2_applied: List[str] = []

        # 1. LowCdExpert (Layer A)
        if ENABLE_LOW_CD_EXPERT:
            try:
                from day4c_v2_upgrades import LowCdExpert as _LCE
                _lce = _LCE.load(self.model_dir)
                if _lce.model is not None and getattr(_lce, "enabled", True):
                    correction = float(_lce.predict_correction(X_raw, np.array([cd_selected]))[0])
                    cd_selected = max(cd_selected + correction, self.cd_floor)
                    hard = max(hard + float(_lce.predict_correction(X_raw, np.array([hard]))[0]), self.cd_floor)
                    blended = max(blended + float(_lce.predict_correction(X_raw, np.array([blended]))[0]), self.cd_floor)
                    v2_applied.append("low_cd_expert")
            except Exception:
                pass

        # 2. HighDragResidualExpert (Layer B)
        if ENABLE_HIGH_DRAG_EXPERT:
            try:
                from day4c_v2_upgrades import HighDragResidualExpert as _HDRE
                _hdre = _HDRE.load(self.model_dir)
                if _hdre.enabled:
                    corr_selected = float(_hdre.predict_correction(X_raw, np.array([cd_selected]))[0])
                    cd_selected = max(cd_selected + corr_selected, self.cd_floor)
                    hard = max(hard + float(_hdre.predict_correction(X_raw, np.array([hard]))[0]), self.cd_floor)
                    blended = max(blended + float(_hdre.predict_correction(X_raw, np.array([blended]))[0]), self.cd_floor)
                    v2_applied.append("high_drag_residual_expert")
            except Exception:
                pass

        # 3. RegimeMicroCalibration (Layer C)
        if ENABLE_REGIME_MICRO_CAL:
            try:
                from day4c_v2_upgrades import RegimeMicroCalibration as _RMC
                _rmc = _RMC.load(self.model_dir)
                if _rmc.enabled:
                    cd_selected = float(_rmc.apply(np.array([cd_selected]), X_raw)[0])
                    hard = float(_rmc.apply(np.array([hard]), X_raw)[0])
                    blended = float(_rmc.apply(np.array([blended]), X_raw)[0])
                    v2_applied.append("regime_micro_calibration")
            except Exception:
                pass

        # 4. Adaptive Conformal Prediction (conformal intervals lookup)
        q_hat_80, q_hat_90, q_hat_95 = np.nan, np.nan, np.nan
        unc_bin = "med"
        conformal_source = "none"
        try:
            from day4c_v2_upgrades import AdaptiveConformalIntervals as _ACI
            _aci = _ACI.load(self.model_dir)
            
            # Predict cluster ID on-the-fly
            cluster_id = -1
            k_scaler_path = self.model_dir / "geometry_kmeans_scaler.pkl"
            k_model_path = self.model_dir / "geometry_kmeans_model.pkl"
            if k_scaler_path.exists() and k_model_path.exists():
                try:
                    import joblib
                    scaler = joblib.load(k_scaler_path)
                    km = joblib.load(k_model_path)
                    
                    row_feats = {}
                    geo_mappings = {
                        "t_max": ["t_max"],
                        "camber_max": ["camber_max"],
                        "trailing_edge_angle": ["trailing_edge_angle", "te_wedge_angle"],
                        "aft_thickness": ["aft_thickness", "thickness_aft"],
                        "curvature_aft": ["curvature_aft"],
                        "pressure_recovery_proxy": ["pressure_recovery_proxy"]
                    }
                    for f, aliases in geo_mappings.items():
                        val = None
                        for alias in aliases:
                            if alias in X_raw.columns:
                                val = float(X_raw[alias].iloc[0])
                                break
                        row_feats[f] = val if val is not None else 0.0
                    
                    X_geo = pd.DataFrame([row_feats])[_aci.CLUSTER_FEATURES].to_numpy(dtype=float)
                    X_scaled = scaler.transform(X_geo)
                    cluster_id = int(km.predict(X_scaled)[0])
                except Exception:
                    cluster_id = -1

            density_val = float(X_raw["density_weight"].iloc[0]) if "density_weight" in X_raw.columns else 1.0
            
            q_hat_80, q_hat_90, q_hat_95, conformal_source = _aci.lookup_quantiles(
                hard_regime, cluster_id, float(selected_unc), density_val
            )
            unc_bin = _aci._get_unc_bin(float(selected_unc))
        except Exception:
            pass

        cd_final = float(max(cd_selected, self.cd_floor))

        return {
            "day4c_cd_hard": float(max(hard, self.cd_floor)),
            "day4c_cd_blended_raw": float(max(blended, self.cd_floor)),
            "day4c_cd_blended": cd_final,
            "day4c_cd_uncertainty_hard": float(max(hard_unc, 0.0)),
            "day4c_cd_uncertainty_blended_raw": float(max(blended_unc, 0.0)),
            "day4c_cd_uncertainty_blended": float(max(selected_unc, 0.0)),
            "day4c_regime": int(hard_regime),
            "day4c_prediction_policy": "hard_high_aoa" if use_hard_high_aoa and abs(float(alpha)) >= 8.0 else "blended",
            "day4c_blend_weights": desc,
            # V2 fields
            "day4c_v2_layers_applied": ";".join(v2_applied) if v2_applied else "none",
            "day4c_v2_q_hat_90": float(q_hat_90) if np.isfinite(q_hat_90) else None,
            "day4c_v2_q_hat_80": float(q_hat_80) if np.isfinite(q_hat_80) else None,
            "day4c_v2_q_hat_95": float(q_hat_95) if np.isfinite(q_hat_95) else None,
            "uncertainty_bin": unc_bin,
            "conformal_source": conformal_source,
            "cd_interval_80": (cd_final - q_hat_80, cd_final + q_hat_80) if np.isfinite(q_hat_80) else None,
            "cd_interval_90": (cd_final - q_hat_90, cd_final + q_hat_90) if np.isfinite(q_hat_90) else None,
            "cd_interval_95": (cd_final - q_hat_95, cd_final + q_hat_95) if np.isfinite(q_hat_95) else None,
        }
