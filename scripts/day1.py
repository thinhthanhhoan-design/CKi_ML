"""
day1.py v2

Script xu ly / kham pha du lieu khi dong hoc (airfoil) tu file CSV:
- Parse toa do bien dang (x_coords/y_coords) tu chuoi sang mang so.
- Chuan hoa hinh hoc (chord=1), noi suy len luoi x chuan, tao hash dai dien hinh hoc.
- Ap dung Savitzky-Golay va Cd-dominant Shock Score de phat hien di thuong so hoc.
- Flow regime labeling, stall onset detection, confidence decomposition.
- Target-specific confidence (Cl, Cd, Cm) va sample weights.
- Tong hop thong ke chat luong du lieu va xuat bang + hinh do thi.

Day1 scope: physics-aware quality auditing + regime labeling + target-specific confidence/weighting.
Day2 scope: geometry canonicalization + geometry feature engineering (giu nguyen).

Triet ly:
- 'stall/post-stall physics naturally contains structured noise' -- khong xoa sample stall.
- Pipeline la multi-target: ho tro X -> (Cl, Cd, Cm).
- Cd-centric auditing chi ap dung cho weighting & quality scoring.
- 'nonlinear aerodynamic transition is valuable learning signal'.
"""

import hashlib
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURABLE THRESHOLDS
# Override DAY1_CONFIG truoc khi goi main() neu can.
# Tat ca nguong la conservative mac dinh de tranh over-flagging.
# ===========================================================================
DAY1_CONFIG: dict = {
    # --- Geometry audit thresholds ---
    "curvature_spike_global": 50.0,   # Max |kappa| tren toan bo be mat
    "curvature_spike_te":     30.0,   # Max |kappa| vung TE (x > 0.8)
    "te_osc_threshold":        5.0,   # Std curvature vung TE => TE_OSCILLATION
    "le_osc_threshold":        5.0,   # Std curvature vung LE => LE_OSCILLATION
    "te_thickness_max":        0.03,  # Max TE thickness (normalized chord)
    "wedge_angle_max_deg":    30.0,   # Max TE wedge angle (degrees)

    # --- Aerodynamic physics thresholds ---
    "low_aoa_high_cd_angle":   6.0,   # |AoA| < X => Cd should be low
    "low_aoa_high_cd_val":     0.08,  # Cd > X at low AoA => LOW_AOA_HIGH_CD
    "drag_ratio_max":          2.5,   # cd / (cl**2 + 0.05) > X => DRAG_RATIO_OUTLIER
    "cd_curvature_max":        0.05,  # d2cd/daoa2 > X => CD_CONVEXITY_BREAK

    # --- Shock score / jump detection ---
    "shock_score_base_threshold": 8.0,
    "re_reference":            1e6,
    "re_threshold_exponent":   0.3,

    # --- Stall onset ---
    "stall_onset_threshold":   3.0,

    # --- Sample weight clipping ---
    "weight_min":     0.1,
    "weight_max_cd": 10.0,
    "weight_max_cl":  5.0,
    "weight_max_cm":  5.0,
}

# Regime importance factors per target
# Cd: aggressive near stall (hardest to learn)
# Cl: smooth behavior, moderate amplification
# Cm: moderate, TE/pressure sensitive
REGIME_IMPORTANCE_CD = {
    "linear": 1.0, "transitional": 1.3, "pre_stall": 1.8,
    "near_stall": 2.5, "post_stall": 3.0, "unknown": 1.0,
}
REGIME_IMPORTANCE_CL = {
    "linear": 1.0, "transitional": 1.1, "pre_stall": 1.3,
    "near_stall": 1.5, "post_stall": 1.8, "unknown": 1.0,
}
REGIME_IMPORTANCE_CM = {
    "linear": 1.0, "transitional": 1.1, "pre_stall": 1.4,
    "near_stall": 1.6, "post_stall": 1.8, "unknown": 1.0,
}

# AoA bin definitions
AOA_BINS_1D   = [0,   4,    8,    10,    12,    16,    20,    25]
AOA_LABELS_1D = ["0-4", "4-8", "8-10", "10-12", "12-16", "16-20", "20-25"]
AOA_BINS_2D   = [-25, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 25]
RE_BINS_2D    = [1e4, 5e4, 1e5, 3e5, 5e5, 1e6, 2e6, 5e6]

REGIME_COLORS = {
    "linear": "#4CAF50", "transitional": "#2196F3",
    "pre_stall": "#FF9800", "near_stall": "#F44336", "post_stall": "#9C27B0",
    "unknown": "#9E9E9E",
}
QUALITY_COLORS = {"valid": "#4CAF50", "suspect": "#FF9800", "invalid": "#F44336"}


# ===========================================================================
# CORE UTILITY FUNCTIONS (giu nguyen tu v1)
# ===========================================================================

def parse_coord_string(s: str) -> np.ndarray:
    """Chuyen chuoi '0.0 0.1 0.2 ...' (tu CSV) thanh np.ndarray float."""
    if pd.isna(s) or str(s).strip() == "":
        return np.array([], dtype=float)
    try:
        return np.array([float(v) for v in str(s).split()], dtype=float)
    except Exception:
        return np.array([np.nan], dtype=float)


def dedup_monotonic_xy(x, y):
    """Lam sach cap (x, y) de phuc vu noi suy (x tang dan, khong trung)."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if tmp.empty:
        return np.array([]), np.array([])
    out = tmp.groupby("x", as_index=False)["y"].median()
    return out["x"].to_numpy(), out["y"].to_numpy()


def canonicalize_airfoil(x_arr, y_arr, n_grid=200, tol=1e-8, config=None):
    """Chuan hoa bien dang airfoil va tinh geometry audit flags.

    Day1 audit scope: tra ve boolean flags va float oscillation metrics.
    Feature engineering geometry (PCA, camber...) thuoc Day2.
    Tat ca nguong lay tu config (khong hard-code).
    """
    if config is None:
        config = DAY1_CONFIG

    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)

    if len(x) != len(y) or len(x) < 20 or np.isnan(x).any() or np.isnan(y).any():
        return None

    x_min, x_max = np.min(x), np.max(x)
    chord = x_max - x_min
    if chord <= tol:
        return None

    x = (x - x_min) / chord
    y = y / chord

    i_le = int(np.argmin(x))
    upper = np.column_stack([x[: i_le + 1], y[: i_le + 1]])[::-1]
    lower = np.column_stack([x[i_le:], y[i_le:]])

    xu, yu = dedup_monotonic_xy(upper[:, 0], upper[:, 1])
    xl, yl = dedup_monotonic_xy(lower[:, 0], lower[:, 1])

    if len(xu) < 5 or len(xl) < 5:
        return None

    x_grid = np.linspace(0.0, 1.0, n_grid)
    yu_i = np.interp(x_grid, xu, yu)
    yl_i = np.interp(x_grid, xl, yl)

    thickness     = yu_i - yl_i
    min_thickness = float(np.min(thickness))

    payload   = np.round(np.column_stack([x_grid, yu_i, yl_i]), 6)
    geom_hash = hashlib.sha1(payload.tobytes()).hexdigest()

    # --- Geometry audit computations (internal intermediates) ---
    kappa_u = np.gradient(np.gradient(yu_i, x_grid), x_grid)
    kappa_l = np.gradient(np.gradient(yl_i, x_grid), x_grid)
    te_mask = x_grid > 0.8
    le_mask = x_grid < 0.15

    curvature_spike = bool(
        np.max(np.abs(kappa_u)) > config["curvature_spike_global"] or
        np.max(np.abs(kappa_l)) > config["curvature_spike_global"]
    )
    curvature_spike_te = bool(
        te_mask.any() and
        np.max(np.abs(kappa_u[te_mask])) > config["curvature_spike_te"]
    )

    slope_u    = np.gradient(yu_i, x_grid)
    slope_l    = np.gradient(yl_i, x_grid)
    te_mask_97 = x_grid > 0.97
    te_thick   = float(
        abs(np.mean(yu_i[te_mask_97]) - np.mean(yl_i[te_mask_97]))
        if te_mask_97.any() else 0.0
    )
    wedge = float(np.degrees(np.arctan(abs(slope_u[-1] - slope_l[-1]))))
    bad_trailing_edge = bool(
        te_thick > config["te_thickness_max"] or
        wedge    > config["wedge_angle_max_deg"]
    )

    le_osc = float(np.std(kappa_u[le_mask])) if le_mask.sum() > 3 else 0.0
    te_osc = float(np.std(kappa_u[te_mask])) if te_mask.sum() > 3 else 0.0

    # Internal proxy for Cm weighting (prefix _ => NOT exported to CSV)
    _wedge_proxy = float(min(wedge / max(config["wedge_angle_max_deg"], 1.0), 1.0))

    return {
        "x_grid":       x_grid,
        "y_upper":      yu_i,
        "y_lower":      yl_i,
        "thickness":    thickness,
        "min_thickness": min_thickness,
        "geom_hash":    geom_hash,
        "geom_ok":      min_thickness >= -1e-4,
        # Geometry audit flags (exported)
        "curvature_spike":    curvature_spike,
        "curvature_spike_te": curvature_spike_te,
        "bad_trailing_edge":  bad_trailing_edge,
        "le_osc":             le_osc,
        "te_osc":             te_osc,
        # Internal only (NOT exported)
        "_wedge_proxy":       _wedge_proxy,
    }


def _ensure_dirs():
    """Tao cac thu muc output (neu chua co)."""
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs(os.path.join("figures", "curves"), exist_ok=True)


def _update_minmax(agg, v):
    """Cap nhat min/max trong dict agg neu v khong phai NaN."""
    if np.isnan(v):
        return
    if v < agg["min"]:
        agg["min"] = v
    if v > agg["max"]:
        agg["max"] = v


def robust_mad(x):
    """Tinh MAD (median absolute deviation) ben vung; cong epsilon de tranh chia 0."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) + 1e-12


# ===========================================================================
# UPGRADED CORE FUNCTIONS
# ===========================================================================

def diagnose_curve(g, config=None):
    """Chan doan mot curve theo angle va gan co jump_flag cho diem bat thuong.
    Bao toan cau truc du lieu goc de khong lam mat cac ban ghi khuyet thieu (NaN).

    Upgrade v2:
    - Shock score Cd-dominant (1.4x dcd, 1.0x d2cd, 0.5x dcl).
    - Reynolds-aware adaptive threshold thay the hard threshold 8.0.
    - Them output column adaptive_threshold.
    """
    if config is None:
        config = DAY1_CONFIG

    g = g.sort_values("angle").copy()

    # Khoi tao cot ket qua mac dinh cho toan bo dong
    g["cl_smooth"]          = g["cl"]
    g["cd_smooth"]          = g["cd"]
    g["cm_smooth"]          = g["cm"]
    g["shock_score"]        = np.nan
    g["adaptive_threshold"] = np.nan
    g["cd_curvature"]       = np.nan
    g["relative_cd_jump"]   = np.nan
    g["cd_smooth_residual"] = np.nan
    g["jump_flag"]          = False
    g["cd_drop_flag"]       = False
    g["cd_smooth_outlier"]  = False

    g_clean = g.dropna(subset=["cl", "cd", "cm"])
    n = len(g_clean)

    if n < 7:
        return g

    win = min(9, n if n % 2 == 1 else n - 1)
    if win < 5:
        win = 5

    cl_smooth = savgol_filter(g_clean["cl"].to_numpy(), window_length=win, polyorder=2, mode="interp")
    cd_smooth = savgol_filter(g_clean["cd"].to_numpy(), window_length=win, polyorder=2, mode="interp")
    cm_smooth = savgol_filter(g_clean["cm"].to_numpy(), window_length=win, polyorder=2, mode="interp")

    dalpha          = g_clean["angle"].diff()
    dcl             = g_clean["cl"].diff()
    dcd             = g_clean["cd"].diff()
    d2cd            = dcd.diff()
    dcm             = g_clean["cm"].diff()
    relative_cd_jump = dcd.abs() / (g_clean["cd"].abs() + 1e-4)

    mad_cl   = max(robust_mad(dcl.dropna()),  0.01)
    mad_cd   = max(robust_mad(dcd.dropna()),  0.001)
    mad_d2cd = max(robust_mad(d2cd.dropna()), 0.001)
    mad_cm   = max(robust_mad(dcm.dropna()),  0.005)

    # Cd-dominant shock score -- Cd learning near stall is the priority
    shock_score = (
        1.4 * dcd.abs()  / mad_cd          # Cd discontinuity -- dominant
        + 1.0 * d2cd.abs() / mad_d2cd      # Cd curvature -- detect collapse
        + 0.5 * dcl.abs()  / mad_cl        # Cl -- supporting signal
        + 0.4 * relative_cd_jump           # Relative Cd jump ratio
        + 0.2 * dcm.abs()  / mad_cm        # Cm -- minor signal
    )

    # Reynolds-aware adaptive threshold
    re_val = float(g_clean["reynolds"].iloc[0]) if "reynolds" in g_clean.columns else config["re_reference"]
    re_val = max(re_val, 1e4)
    re_factor    = float(np.clip(
        (config["re_reference"] / re_val) ** config["re_threshold_exponent"],
        1.0, 2.0
    ))
    adaptive_thr = config["shock_score_base_threshold"] * re_factor

    sign_flip_far = (
        (g_clean["cl"] * g_clean["cl"].shift(1) < 0)
        & (~g_clean["angle"].between(-2, 2))
    )

    cd_drop_flag = (
        (g_clean["cd"] < 0.7 * g_clean["cd"].shift(1))
        & (g_clean["angle"].abs() > 6)
    ).fillna(False)

    cd_smooth_residual = pd.Series(
        np.abs(g_clean["cd"].to_numpy(dtype=float) - cd_smooth),
        index=g_clean.index,
    )
    mad_cd_smooth_res  = max(robust_mad(cd_smooth_residual.dropna()), 5e-4)
    cd_smooth_outlier  = (cd_smooth_residual > 3.0 * mad_cd_smooth_res).fillna(False)

    jump_flag = (
        ((dalpha.abs() <= 0.5) & (shock_score > adaptive_thr))
        | (g_clean["cd"] < -1e-6)
        | sign_flip_far
        | cd_drop_flag
    )

    # Gan nguoc ket qua ve cac dong tuong ung dua tren index
    g.loc[g_clean.index, "cl_smooth"]          = cl_smooth
    g.loc[g_clean.index, "cd_smooth"]          = cd_smooth
    g.loc[g_clean.index, "cm_smooth"]          = cm_smooth
    g.loc[g_clean.index, "shock_score"]        = shock_score
    g.loc[g_clean.index, "adaptive_threshold"] = adaptive_thr
    g.loc[g_clean.index, "cd_curvature"]       = d2cd.abs()
    g.loc[g_clean.index, "relative_cd_jump"]   = relative_cd_jump
    g.loc[g_clean.index, "cd_smooth_residual"] = cd_smooth_residual
    g.loc[g_clean.index, "jump_flag"]          = jump_flag
    g.loc[g_clean.index, "cd_drop_flag"]       = cd_drop_flag
    g.loc[g_clean.index, "cd_smooth_outlier"]  = cd_smooth_outlier

    return g


def assign_quality(row, config=None):
    """Assign row quality and return (label, reasons, confidence).

    Day1 scope: chi tieu thu audit flags (bool/float) -- khong tieu thu raw geometry floats.
    Su dung DAY1_CONFIG cho tat ca nguong (khong hard-code).
    """
    if config is None:
        config = DAY1_CONFIG

    reasons    = []
    confidence = 1.0

    if (
        pd.isna(row["angle"])
        or pd.isna(row["reynolds"])
        or pd.isna(row["cl"])
        or pd.isna(row["cd"])
        or pd.isna(row["cm"])
    ):
        return "invalid", ["MISSING_CORE_FIELDS"], 0.0

    if row["geom_hash"] == "INVALID_GEOM" or not row["geom_ok"]:
        # Permissive mode: keep rows with recoverable geometry issues as suspect.
        reasons.append("GEOMETRY_INVALID")
        confidence *= 0.2

    if row["n_x"] != row["n_y"]:
        return "invalid", ["COORD_COUNT_MISMATCH"], 0.0

    if row["n_x"] < 20:
        reasons.append("TOO_FEW_POINTS")
        confidence *= 0.2

    if row["cd"] < -1e-6:
        reasons.append("NEGATIVE_CD")
        confidence *= 0.15

    if abs(row["cl"]) > 4 or row["cd"] > 2 or abs(row["cm"]) > 1:
        reasons.append("HARD_BOUND_VIOLATION")
        confidence *= 0.15

    # --- Soft flags (suspect) ---
    if 20 <= row["n_x"] < 50:
        reasons.append("LOW_POINT_COUNT")
        confidence *= 0.7

    if -1e-6 <= row["cd"] < 0:
        reasons.append("CD_NEAR_ZERO_NEGATIVE")
        confidence *= 0.5

    if bool(row.get("jump_flag", False)):
        reasons.append("CURVE_JUMP")
        confidence *= 0.4

    if bool(row.get("cd_drop_flag", False)):
        reasons.append("CD_COLLAPSE")
        confidence *= 0.35

    if bool(row.get("cd_smooth_outlier", False)):
        reasons.append("CD_SMOOTH_RESIDUAL_OUTLIER")
        confidence *= 0.5

    # --- Cd physics checks (configurable thresholds) ---
    aoa_val = float(row["angle"]) if pd.notna(row.get("angle")) else 0.0
    cd_val  = float(row["cd"])    if pd.notna(row.get("cd"))    else 0.0
    cl_val  = float(row["cl"])    if pd.notna(row.get("cl"))    else 0.0

    if abs(aoa_val) < config["low_aoa_high_cd_angle"] and cd_val > config["low_aoa_high_cd_val"]:
        reasons.append("LOW_AOA_HIGH_CD")
        confidence *= 0.5

    drag_ratio = cd_val / (cl_val ** 2 + 0.05)
    if np.isfinite(drag_ratio) and drag_ratio > config["drag_ratio_max"]:
        reasons.append("DRAG_RATIO_OUTLIER")
        confidence *= 0.6

    cd_curv = row.get("cd_curvature", np.nan)
    if pd.notna(cd_curv) and float(cd_curv) > config["cd_curvature_max"]:
        reasons.append("CD_CONVEXITY_BREAK")
        confidence *= 0.65

    # --- Geometry audit flags (boolean/float only, NOT raw geometry values) ---
    if bool(row.get("curvature_spike_te", False)):
        reasons.append("GEOM_CURVATURE_SPIKE")
        confidence *= 0.7

    if bool(row.get("bad_trailing_edge", False)):
        reasons.append("BAD_TRAILING_EDGE")
        confidence *= 0.75

    if float(row.get("le_osc", 0.0) or 0.0) > config["le_osc_threshold"]:
        reasons.append("LE_OSCILLATION")
        confidence *= 0.8

    if float(row.get("te_osc", 0.0) or 0.0) > config["te_osc_threshold"]:
        reasons.append("TE_OSCILLATION")
        confidence *= 0.8

    # --- Stall onset (nhẹ -- không remove) ---
    if bool(row.get("stall_onset_candidate", False)):
        reasons.append("STALL_ONSET_CANDIDATE")
        confidence *= 0.85

    if reasons:
        return "suspect", reasons, float(np.clip(confidence, 0.05, 1.0))

    return "valid", [], 1.0


# ===========================================================================
# NEW FUNCTIONS -- Physics-Aware Regime & Confidence Analysis
# ===========================================================================

def label_flow_regime(df: pd.DataFrame) -> pd.Series:
    """Gan nhan flow regime dua tren |AoA|. Vectorized. Configurable bins.

    NaN angle rows are labeled 'unknown'.
    pd.cut returns NaN for out-of-range/missing values; after astype(str)
    those become the literal string 'nan' -- replace explicitly.
    """
    abs_aoa = df["angle"].abs()
    regime  = pd.cut(
        abs_aoa,
        bins=[0, 4, 8, 12, 16, 90],
        labels=["linear", "transitional", "pre_stall", "near_stall", "post_stall"],
        right=False,
    )
    result = regime.astype(str)
    result = result.replace("nan", "unknown")   # pd.cut NaN -> str 'nan' -> 'unknown'
    return result.fillna("unknown")


def compute_stall_onset_score(g: pd.DataFrame, config=None) -> pd.DataFrame:
    """Tinh stall_onset_score va danh dau stall_onset_candidate theo group (name, reynolds).

    Score = 0.4 * neg_lift_slope + 0.4 * drag_growth + 0.2 * curvature_change
    Khong remove sample -- chi mark va giam confidence nhe (x0.85).
    """
    if config is None:
        config = DAY1_CONFIG

    g = g.sort_values("angle").copy()
    g["stall_onset_score"]     = np.nan
    g["stall_onset_candidate"] = False

    g_c = g.dropna(subset=["cl", "cd", "angle"])
    if len(g_c) < 5:
        return g

    dcl  = g_c["cl"].diff()
    dcd  = g_c["cd"].diff()
    daoa = g_c["angle"].diff().replace(0, np.nan)

    dcl_daoa = dcl / daoa
    dcd_daoa = dcd / daoa
    d2cl     = dcl_daoa.diff()

    neg_lift_slope = (-dcl_daoa).clip(lower=0)   # Cl slope am -> stall
    drag_growth    = dcd_daoa.clip(lower=0)        # Cd tang -> post-stall
    curv_change    = d2cl.abs()                    # Curvature Cl doi dau

    mad_nls = max(robust_mad(neg_lift_slope.dropna()), 1e-4)
    mad_dg  = max(robust_mad(drag_growth.dropna()),    1e-4)
    mad_cc  = max(robust_mad(curv_change.dropna()),    1e-4)

    score = (
        0.4 * neg_lift_slope / mad_nls
        + 0.4 * drag_growth  / mad_dg
        + 0.2 * curv_change  / mad_cc
    ).fillna(0.0)

    candidate = score > config["stall_onset_threshold"]

    g.loc[g_c.index, "stall_onset_score"]     = score
    g.loc[g_c.index, "stall_onset_candidate"] = candidate
    return g


def _compute_confidence_vectorized(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """Tinh 4 confidence components (vectorized -- khong dung apply(axis=1)).

    geometry_confidence: curvature smoothness + TE quality + oscillation
    physics_confidence:  drag consistency + impossible Cd + CL/CD ratio
    stall_confidence:    derivative stability + stall onset
    solver_confidence:   shock_score + cd_smooth_outlier + cd_drop_flag

    final = 0.3*geometry + 0.3*physics + 0.2*stall + 0.2*solver
    """
    if config is None:
        config = DAY1_CONFIG

    n  = len(df)
    gc = np.ones(n, dtype=float)
    pc = np.ones(n, dtype=float)
    sc = np.ones(n, dtype=float)
    vc = np.ones(n, dtype=float)

    def _col(name, default):
        if name in df.columns:
            return df[name].fillna(default).values
        return np.full(n, default)

    # geometry_confidence
    gc = np.where(_col("curvature_spike_te", False).astype(bool), gc * 0.6, gc)
    gc = np.where(_col("bad_trailing_edge",  False).astype(bool), gc * 0.7, gc)
    gc = np.where(_col("le_osc", 0.0) > config["le_osc_threshold"], gc * 0.8, gc)
    gc = np.where(_col("te_osc", 0.0) > config["te_osc_threshold"], gc * 0.8, gc)
    gc = np.where(~_col("geom_ok", True).astype(bool), 0.0, gc)

    # physics_confidence
    aoa_abs    = np.abs(_col("angle", 0.0))
    cd_vals    = _col("cd", 0.0)
    cl_vals    = _col("cl", 0.0)
    drag_ratio = cd_vals / (cl_vals ** 2 + 0.05)

    pc = np.where(
        (aoa_abs < config["low_aoa_high_cd_angle"]) & (cd_vals > config["low_aoa_high_cd_val"]),
        pc * 0.5, pc
    )
    pc = np.where(
        np.isfinite(drag_ratio) & (drag_ratio > config["drag_ratio_max"]),
        pc * 0.6, pc
    )
    pc = np.where(cd_vals < -1e-6, 0.0, pc)

    # stall_confidence
    onset = _col("stall_onset_score", 0.0)
    sc    = sc * np.clip(1.0 - 0.3 * (onset / 5.0), 0.4, 1.0)
    sc    = np.where(_col("stall_onset_candidate", False).astype(bool), sc * 0.85, sc)

    # solver_confidence
    ss = _col("shock_score", 0.0)
    vc = vc * np.clip(1.0 - 0.05 * np.maximum(ss - 5.0, 0.0), 0.2, 1.0)
    vc = np.where(_col("cd_smooth_outlier", False).astype(bool), vc * 0.6, vc)
    vc = np.where(_col("cd_drop_flag",      False).astype(bool), vc * 0.5, vc)

    gc    = np.clip(gc, 0.0, 1.0)
    pc    = np.clip(pc, 0.0, 1.0)
    sc    = np.clip(sc, 0.0, 1.0)
    vc    = np.clip(vc, 0.0, 1.0)
    final = np.clip(0.3 * gc + 0.3 * pc + 0.2 * sc + 0.2 * vc, 0.0, 1.0)

    result = df.copy()
    result["geometry_confidence"] = gc
    result["physics_confidence"]  = pc
    result["stall_confidence"]    = sc
    result["solver_confidence"]   = vc
    result["sample_confidence"]   = final
    return result


def compute_target_confidence(df: pd.DataFrame) -> tuple:
    """Tinh target-specific confidence (vectorized).

    cl_confidence: geometry + stall (Cl bi anh huong stall & geometry)
    cd_confidence: geometry + physics + solver (Cd noisy nhat)
    cm_confidence: geometry + physics (Cm nhay TE quality)
    """
    gc = df["geometry_confidence"].values
    pc = df["physics_confidence"].values
    sc = df["stall_confidence"].values
    vc = df["solver_confidence"].values

    cl_conf = np.clip(0.35 * gc + 0.45 * sc + 0.20 * vc, 0.0, 1.0)
    cd_conf = np.clip(0.25 * gc + 0.35 * pc + 0.20 * sc + 0.20 * vc, 0.0, 1.0)
    cm_conf = np.clip(0.40 * gc + 0.35 * pc + 0.25 * sc, 0.0, 1.0)

    return cl_conf, cd_conf, cm_conf


def compute_stall_risk_score(df: pd.DataFrame) -> pd.Series:
    """Tinh stall_risk_score [0,1] tu AoA, onset score, va Cd residual."""
    abs_aoa  = df["angle"].abs().clip(0, 25)
    norm_aoa = abs_aoa / 25.0

    onset = df.get("stall_onset_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    q95_onset = float(onset.quantile(0.95)) + 1e-8
    onset_norm = onset.clip(0, q95_onset) / q95_onset

    cd_res = df.get("cd_smooth_residual", pd.Series(0.0, index=df.index)).fillna(0.0)
    q95_res = float(cd_res.quantile(0.95)) + 1e-8
    cd_res_norm = cd_res.clip(0, q95_res) / q95_res

    score = 0.45 * norm_aoa + 0.35 * onset_norm + 0.20 * cd_res_norm
    return score.clip(0.0, 1.0)


def compute_aoa_density(df: pd.DataFrame) -> pd.DataFrame:
    """Thong ke density theo AoA bins vat ly (1D)."""
    abs_aoa = df["angle"].abs()
    df_tmp  = df.copy()
    df_tmp["_aoa_bin"] = pd.cut(abs_aoa, bins=AOA_BINS_1D, labels=AOA_LABELS_1D, right=True)
    density = df_tmp.groupby("_aoa_bin", observed=True).agg(
        n_samples=("angle",         "count"),
        n_valid=("quality_status",  lambda x: (x == "valid").sum()),
        n_suspect=("quality_status", lambda x: (x == "suspect").sum()),
        mean_cd=("cd", "mean"),
        mean_cl=("cl", "mean"),
        mean_shock_score=("shock_score", "mean"),
    ).reset_index()
    density.columns.name = None
    median_n = float(density["n_samples"].median()) if len(density) > 0 else 1.0
    density["sparse_flag"] = density["n_samples"] < 0.5 * max(median_n, 1.0)
    return density


def compute_density_matrix(df: pd.DataFrame):
    """Tinh 2D AoA x Reynolds density matrix."""
    df_tmp   = df.copy()
    re_valid = df_tmp["reynolds"].dropna()

    re_bins = [b for b in RE_BINS_2D if b <= re_valid.max() * 1.01]
    if len(re_bins) < 2:
        re_bins = RE_BINS_2D[:]
    if re_bins[-1] < re_valid.max():
        re_bins = re_bins + [re_valid.max() * 1.1]

    df_tmp["_re_bin"]  = pd.cut(df_tmp["reynolds"], bins=re_bins, right=False)
    df_tmp["_aoa_bin"] = pd.cut(df_tmp["angle"], bins=AOA_BINS_2D, right=False)

    density = (
        df_tmp.groupby(["_re_bin", "_aoa_bin"], observed=True)
        .size()
        .reset_index(name="n_samples")
    )
    matrix = density.pivot(
        index="_re_bin", columns="_aoa_bin", values="n_samples"
    ).fillna(0)
    return density, matrix


def _assign_density_weight(df: pd.DataFrame, density_long: pd.DataFrame) -> pd.DataFrame:
    """Gan density_weight cho tung row dua tren 2D bin membership."""
    df             = df.copy()
    density_long   = density_long.copy()
    density_long["raw_w"] = 1.0 / np.sqrt(density_long["n_samples"].clip(lower=1) + 1)
    w_mean = float(density_long["raw_w"].mean())
    density_long["density_weight"] = (
        density_long["raw_w"] / max(w_mean, 1e-8)
    ).clip(0.1, 5.0)

    re_valid = df["reynolds"].dropna()
    re_bins  = [b for b in RE_BINS_2D if b <= re_valid.max() * 1.01]
    if len(re_bins) < 2:
        re_bins = RE_BINS_2D[:]
    if re_bins[-1] < re_valid.max():
        re_bins = re_bins + [re_valid.max() * 1.1]

    df["_re_bin"]  = pd.cut(df["reynolds"], bins=re_bins, right=False)
    df["_aoa_bin"] = pd.cut(df["angle"],    bins=AOA_BINS_2D, right=False)

    dw_map = density_long.set_index(["_re_bin", "_aoa_bin"])["density_weight"].to_dict()

    def _lookup(row):
        return dw_map.get((row["_re_bin"], row["_aoa_bin"]), 1.0)

    df["density_weight"] = df.apply(_lookup, axis=1)
    df = df.drop(columns=["_re_bin", "_aoa_bin"], errors="ignore")
    return df


def compute_sample_weights(df: pd.DataFrame, config=None) -> tuple:
    """Tinh 3 target-specific sample weights (vectorized).

    sample_weight_cd: aggressive near stall + density emphasis + drag_transition
    sample_weight_cl: smooth, khong amplify drag spikes (no density_weight)
    sample_weight_cm: moderate, nhay voi TE quality (_wedge_proxy)

    Khong dung uniform weighting. Moi target mot triet ly rieng.
    """
    if config is None:
        config = DAY1_CONFIG

    conf   = df["sample_confidence"].clip(0.01, 1.0)
    dw     = df.get("density_weight", pd.Series(1.0, index=df.index)).clip(0.1, 5.0)
    regime = df.get("flow_regime", pd.Series("unknown", index=df.index))

    ri_cd = regime.map(REGIME_IMPORTANCE_CD).fillna(1.0)
    ri_cl = regime.map(REGIME_IMPORTANCE_CL).fillna(1.0)
    ri_cm = regime.map(REGIME_IMPORTANCE_CM).fillna(1.0)

    onset = df.get("stall_onset_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    drag_trans = (1.0 + 0.5 * onset.clip(0, 5) / 5.0).clip(1.0, 1.5)

    # _wedge_proxy (internal, not exported) for Cm TE quality
    wp   = df.get("_wedge_proxy", pd.Series(0.0, index=df.index)).fillna(0.0)
    te_w = (1.0 - 0.3 * wp).clip(0.7, 1.0)

    raw_cd = conf * ri_cd * dw * drag_trans
    raw_cl = conf * ri_cl                      # no density to avoid noise overfocus
    raw_cm = conf * ri_cm * te_w

    def _norm(s, w_max):
        mean_val = float(s.mean())
        if mean_val < 1e-8:
            return s.clip(config["weight_min"], w_max)
        return (s / mean_val).clip(config["weight_min"], w_max)

    return (
        _norm(raw_cd, config["weight_max_cd"]),
        _norm(raw_cl, config["weight_max_cl"]),
        _norm(raw_cm, config["weight_max_cm"]),
    )


def compute_regime_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-target regime statistics (Cl, Cd, Cm) per flow_regime."""
    return df.groupby("flow_regime", observed=True).agg(
        n=("cd", "count"),
        # Cd
        mean_cd=("cd", "mean"),
        std_cd=("cd", "std"),
        mean_weight_cd=("sample_weight_cd", "mean"),
        # Cl
        mean_cl=("cl", "mean"),
        std_cl=("cl", "std"),
        mean_weight_cl=("sample_weight_cl", "mean"),
        # Cm
        mean_cm=("cm", "mean"),
        std_cm=("cm", "std"),
        mean_weight_cm=("sample_weight_cm", "mean"),
        # Quality
        mean_shock_score=("shock_score",      lambda x: x.mean()),
        mean_stall_onset=("stall_onset_score", lambda x: x.mean()),
        mean_confidence=("sample_confidence",  "mean"),
        valid_rate=("quality_status", lambda x: (x == "valid").sum() / max(len(x), 1)),
    ).reset_index()


# ===========================================================================
# VISUALIZATION HELPERS
# ===========================================================================

def plot_aoa_cl_curve(g, name, re_value, out_path):
    """Ve duong cong Cl-AoA cho mot airfoil tai mot Reynolds va luu ra file anh."""
    if "cl_smooth" not in g.columns:
        g = diagnose_curve(g)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g["angle"], g["cl"], "o", alpha=0.65, label="Du lieu tho")
    ax.plot(g["angle"], g["cl_smooth"], "-", linewidth=2, label="Duong lam muot")

    if "jump_flag" in g.columns and g["jump_flag"].any():
        ax.scatter(
            g.loc[g["jump_flag"], "angle"],
            g.loc[g["jump_flag"], "cl"],
            marker="x", color="red", s=70, label="Diem nghi loi",
        )
    ax.set_xlabel("Goc tan [do]")
    ax.set_ylabel("Cl")
    ax.set_title(f"{name} @ Re={re_value}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_fig(path, dpi=160):
    """Luu figure hien tai, bo qua neu loi."""
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
    except Exception as e:
        print(f"[day1] Warning: could not save figure {path}: {e}")
    finally:
        plt.close()


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main(
    csv_path="DeepLearWing.csv",
    chunksize=50_000,
    n_grid=200,
    config=None,
    geometry_csv_path="neuralfoil_geometries.csv.gz",
):
    """Doc CSV theo chunks, tong hop thong ke va xuat ket qua ra tables/ va figures/."""
    if config is None:
        config = DAY1_CONFIG

    _ensure_dirs()

    if not os.path.exists(csv_path):
        fallback_csv = "neuralfoil_deeplearwing_augmented.csv.gz"
        if os.path.exists(fallback_csv):
            print(f"Warning: missing {csv_path}. Falling back to {fallback_csv}.")
            csv_path = fallback_csv
        else:
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    numeric_cols = ["angle", "reynolds", "cl", "cd", "cm"]
    core_cols = ["name", "x_coords", "y_coords", "geom_hash"] + numeric_cols

    n_rows_raw  = 0
    t0          = time.time()
    last_log_t  = t0
    chunk_i     = 0

    unique_names      = set()
    unique_geometries = set()
    unique_curves     = set()

    missing_counts = defaultdict(int)
    numeric_values = {c: [] for c in numeric_cols}
    alpha_by_re    = defaultdict(set)
    curve_agg      = {}
    dup_agg        = {}

    high_aoa_n        = 0
    stall_default_n   = 0
    any_stall_by_name = defaultdict(bool)
    cov_counts        = defaultdict(int)

    hist_cols = ["angle", "log10_re", "cl", "cd", "cm", "n_x"]
    hist_bins = 60
    hist_limits = {
        "angle":    (-20.0, 21.0),
        "log10_re": (4.5,   6.5),
        "cl":       (-3.0,  3.0),
        "cd":       (0.0,   0.5),
        "cm":       (-0.5,  0.5),
        "n_x":      (0.0,   400.0),
    }
    hist_counts = {c: np.zeros(hist_bins, dtype=np.int64) for c in hist_cols}

    all_processed_chunks = []

    # Detect actual column names in CSV
    first_row   = pd.read_csv(csv_path, nrows=1)
    actual_cols = first_row.columns.tolist()

    col_mapping = {}
    if "airfoil_name" in actual_cols and "name" not in actual_cols:
        col_mapping["airfoil_name"] = "name"
    if "alpha" in actual_cols and "angle" not in actual_cols:
        col_mapping["alpha"] = "angle"

    csv_usecols = []
    for col in core_cols:
        mapped_src = None
        for src, dest in col_mapping.items():
            if dest == col:
                mapped_src = src
                break
        if mapped_src is not None and mapped_src in actual_cols:
            csv_usecols.append(mapped_src)
        elif col in actual_cols:
            csv_usecols.append(col)

    has_inline_coords = ("x_coords" in actual_cols and "y_coords" in actual_cols)
    geometry_map_df = None
    if not has_inline_coords:
        if not os.path.exists(geometry_csv_path):
            raise FileNotFoundError(
                "Input CSV does not contain x_coords/y_coords and geometry mapping file "
                f"not found: {geometry_csv_path}"
            )
        geometry_map_df = pd.read_csv(
            geometry_csv_path,
            usecols=["geom_hash", "x_coords", "y_coords"],
        ).drop_duplicates(subset=["geom_hash"])
        print(
            f"Info: using geometry mapping from {geometry_csv_path} "
            f"(rows={len(geometry_map_df)})."
        )

    # -----------------------------------------------------------------------
    # Phase 0: Chunk processing (geometry audit + basic stats)
    # -----------------------------------------------------------------------
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=csv_usecols):
        chunk   = chunk.rename(columns=col_mapping)
        if not has_inline_coords:
            if "geom_hash" not in chunk.columns:
                raise KeyError("geom_hash missing in input CSV; cannot recover coordinates.")
            chunk = chunk.merge(geometry_map_df, on="geom_hash", how="left")
            if "name_x" in chunk.columns and "name" not in chunk.columns:
                chunk = chunk.rename(columns={"name_x": "name"})
            if "name" not in chunk.columns and "name_y" in chunk.columns:
                chunk = chunk.rename(columns={"name_y": "name"})
        chunk_i    += 1
        n_rows_raw += len(chunk)

        for c in core_cols:
            missing_counts[c] += int(chunk[c].isna().sum())

        for c in numeric_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        unique_names.update(chunk["name"].dropna().astype(str).unique().tolist())

        x_arr    = chunk["x_coords"].map(parse_coord_string)
        y_arr    = chunk["y_coords"].map(parse_coord_string)
        n_x      = x_arr.map(len)
        n_y      = y_arr.map(len)
        log10_re = np.log10(chunk["reynolds"].astype(float))

        for c in numeric_cols:
            numeric_values[c].extend(chunk[c].to_numpy(dtype=float, na_value=np.nan).tolist())

        # Geometry audit per row
        geom_hashes_list        = []
        geom_ok_list            = []
        curvature_spike_list    = []
        curvature_spike_te_list = []
        bad_trailing_edge_list  = []
        le_osc_list             = []
        te_osc_list             = []
        wedge_proxy_list        = []

        for name, angle, reynolds, cl, cd, cm, xa, ya in zip(
            chunk["name"], chunk["angle"], chunk["reynolds"],
            chunk["cl"], chunk["cd"], chunk["cm"],
            x_arr, y_arr
        ):
            name_str  = str(name)     if pd.notna(name)    else None
            angle_val = float(angle)   if pd.notna(angle)   else np.nan
            re_val    = float(reynolds) if pd.notna(reynolds) else np.nan
            cl_val    = float(cl)       if pd.notna(cl)      else np.nan
            cd_val    = float(cd)       if pd.notna(cd)      else np.nan
            cm_val    = float(cm)       if pd.notna(cm)      else np.nan

            if pd.notna(re_val) and pd.notna(angle_val):
                alpha_by_re[re_val].add(angle_val)
                cov_counts[(re_val, angle_val)] += 1

            if pd.notna(angle_val):
                if angle_val >= 8.0:
                    high_aoa_n += 1
                if angle_val >= 12.0:
                    stall_default_n += 1
                    if name_str is not None:
                        any_stall_by_name[name_str] = True

            canon = (
                canonicalize_airfoil(xa, ya, n_grid=n_grid, config=config)
                if len(xa) == len(ya) and len(xa) > 0
                else None
            )

            if isinstance(canon, dict):
                geom_hash       = canon["geom_hash"]
                geom_ok         = canon["geom_ok"]
                curv_spike      = canon["curvature_spike"]
                curv_spike_te   = canon["curvature_spike_te"]
                bad_te          = canon["bad_trailing_edge"]
                le_osc_val      = canon["le_osc"]
                te_osc_val      = canon["te_osc"]
                wedge_proxy_val = canon["_wedge_proxy"]
            else:
                geom_hash       = "INVALID_GEOM"
                geom_ok         = False
                curv_spike      = False
                curv_spike_te   = False
                bad_te          = False
                le_osc_val      = 0.0
                te_osc_val      = 0.0
                wedge_proxy_val = 0.0

            geom_hashes_list.append(geom_hash)
            geom_ok_list.append(geom_ok)
            curvature_spike_list.append(curv_spike)
            curvature_spike_te_list.append(curv_spike_te)
            bad_trailing_edge_list.append(bad_te)
            le_osc_list.append(le_osc_val)
            te_osc_list.append(te_osc_val)
            wedge_proxy_list.append(wedge_proxy_val)

            if geom_hash != "INVALID_GEOM":
                unique_geometries.add(geom_hash)

            if name_str is not None and pd.notna(re_val):
                curve_key = (name_str, re_val)
                unique_curves.add(curve_key)

                agg = curve_agg.get(curve_key)
                if agg is None:
                    agg = {
                        "n_rows": 0, "angles": set(),
                        "alpha_min": np.inf, "alpha_max": -np.inf,
                        "geom_hashes": set(),
                    }
                    curve_agg[curve_key] = agg

                agg["n_rows"] += 1
                if pd.notna(angle_val):
                    agg["angles"].add(angle_val)
                    if angle_val < agg["alpha_min"]:
                        agg["alpha_min"] = angle_val
                    if angle_val > agg["alpha_max"]:
                        agg["alpha_max"] = angle_val
                agg["geom_hashes"].add(geom_hash)

            if pd.notna(angle_val) and pd.notna(re_val):
                dup_key = (geom_hash, angle_val, re_val)
                da = dup_agg.get(dup_key)
                if da is None:
                    da = {
                        "n": 0,
                        "cl": {"min": np.inf, "max": -np.inf},
                        "cd": {"min": np.inf, "max": -np.inf},
                        "cm": {"min": np.inf, "max": -np.inf},
                    }
                    dup_agg[dup_key] = da
                da["n"] += 1
                _update_minmax(da["cl"], cl_val)
                _update_minmax(da["cd"], cd_val)
                _update_minmax(da["cm"], cm_val)

        # Histograms
        for c in hist_cols:
            if c == "log10_re":
                vals = log10_re.to_numpy(dtype=float, na_value=np.nan)
            elif c == "n_x":
                vals = n_x.to_numpy(dtype=float)
            else:
                vals = chunk[c].to_numpy(dtype=float, na_value=np.nan)
            lo, hi = hist_limits[c]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            counts, _ = np.histogram(vals, bins=hist_bins, range=(lo, hi))
            hist_counts[c] += counts.astype(np.int64)

        # Attach geometry audit flags to chunk
        chunk["n_x"]              = n_x
        chunk["n_y"]              = n_y
        chunk["geom_hash"]        = geom_hashes_list
        chunk["geom_ok"]          = geom_ok_list
        chunk["curvature_spike"]    = curvature_spike_list
        chunk["curvature_spike_te"] = curvature_spike_te_list
        chunk["bad_trailing_edge"]  = bad_trailing_edge_list
        chunk["le_osc"]             = le_osc_list
        chunk["te_osc"]             = te_osc_list
        chunk["_wedge_proxy"]       = wedge_proxy_list

        all_processed_chunks.append(chunk)

        now = time.time()
        if now - last_log_t >= 10:
            elapsed = max(1e-9, now - t0)
            print(
                f"[day1] chunks={chunk_i} rows={n_rows_raw} "
                f"elapsed_s={elapsed:.1f} rows_per_s={n_rows_raw/elapsed:.0f}"
            )
            last_log_t = now

    elapsed = max(1e-9, time.time() - t0)
    print(f"[day1] done rows={n_rows_raw} elapsed_s={elapsed:.1f} rows_per_s={n_rows_raw/elapsed:.0f}")

    # -----------------------------------------------------------------------
    # Basic statistics tables
    # -----------------------------------------------------------------------
    pd.DataFrame([{
        "n_rows_raw":          int(n_rows_raw),
        "n_unique_names":      int(len(unique_names)),
        "n_unique_geometries": int(len(unique_geometries)),
        "n_unique_curves":     int(len(unique_curves)),
    }]).to_csv("tables/day1_counts.csv", index=False)

    missing_rate = {
        c: (missing_counts[c] / n_rows_raw if n_rows_raw else np.nan)
        for c in core_cols
    }
    pd.Series(missing_rate, name="missing_rate").to_csv("tables/day1_missing_rate.csv")

    pct_index   = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]
    percentiles = pd.DataFrame(
        {c: pd.Series(numeric_values[c], dtype="float64").quantile(pct_index) for c in numeric_cols}
    )
    percentiles.to_csv("tables/day1_percentiles.csv")

    expected_alpha_by_re = {re: tuple(sorted(v)) for re, v in alpha_by_re.items()}
    curve_rows = []
    for (name, re_value), agg in curve_agg.items():
        n_alpha    = len(agg["angles"])
        alpha_exp  = expected_alpha_by_re.get(re_value, tuple())
        alpha_comp = (n_alpha / len(alpha_exp)) if alpha_exp else np.nan
        alpha_min  = agg["alpha_min"] if np.isfinite(agg["alpha_min"]) else np.nan
        alpha_max  = agg["alpha_max"] if np.isfinite(agg["alpha_max"]) else np.nan
        curve_rows.append({
            "name": name, "reynolds": re_value,
            "n_rows": agg["n_rows"], "n_alpha": n_alpha,
            "alpha_min": alpha_min, "alpha_max": alpha_max,
            "n_geom": len(agg["geom_hashes"]),
            "alpha_completeness": alpha_comp,
        })
    pd.DataFrame(curve_rows).to_csv("tables/day1_curve_coverage.csv", index=False)

    dup_rows = []
    for (geom_hash, angle, reynolds), da in dup_agg.items():
        cl_min, cl_max = da["cl"]["min"], da["cl"]["max"]
        cd_min, cd_max = da["cd"]["min"], da["cd"]["max"]
        cm_min, cm_max = da["cm"]["min"], da["cm"]["max"]
        dup_rows.append({
            "geom_hash": geom_hash, "angle": angle, "reynolds": reynolds, "n": da["n"],
            "cl_min": (cl_min if np.isfinite(cl_min) else np.nan),
            "cl_max": (cl_max if np.isfinite(cl_max) else np.nan),
            "cd_min": (cd_min if np.isfinite(cd_min) else np.nan),
            "cd_max": (cd_max if np.isfinite(cd_max) else np.nan),
            "cm_min": (cm_min if np.isfinite(cm_min) else np.nan),
            "cm_max": (cm_max if np.isfinite(cm_max) else np.nan),
        })
    dup = pd.DataFrame(dup_rows)
    if not dup.empty:
        dup["cl_range"] = dup["cl_max"] - dup["cl_min"]
        dup["cd_range"] = dup["cd_max"] - dup["cd_min"]
        dup["cm_range"] = dup["cm_max"] - dup["cm_min"]
        exact_dup    = dup.query("n > 1 and cl_range <= 1e-4 and cd_range <= 1e-5 and cm_range <= 1e-4")
        inconsistent = dup.query("n > 1 and (cl_range > 1e-4 or cd_range > 1e-5 or cm_range > 1e-4)")
    else:
        exact_dup    = pd.DataFrame()
        inconsistent = pd.DataFrame()

    exact_dup.to_csv("tables/day1_exact_duplicates.csv",          index=False)
    inconsistent.to_csv("tables/day1_inconsistent_duplicates.csv", index=False)

    pd.DataFrame([{
        "high_aoa_rate":      (high_aoa_n    / n_rows_raw if n_rows_raw else np.nan),
        "stall_default_rate": (stall_default_n / n_rows_raw if n_rows_raw else np.nan),
        "airfoils_with_any_stall_default": (
            sum(any_stall_by_name.values()) / len(unique_names) if unique_names else np.nan
        ),
    }]).to_csv("tables/day1_stall_frequency.csv", index=False)

    # Basic histograms
    for c in hist_cols:
        lo, hi    = hist_limits[c]
        bin_edges = np.linspace(lo, hi, hist_bins + 1)
        centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.figure(figsize=(6, 4))
        plt.bar(centers, hist_counts[c], width=(bin_edges[1] - bin_edges[0]), align="center")
        plt.title(f"Histogram of {c}")
        _save_fig(f"figures/day1_hist_{c}.png")

    cov_df = pd.DataFrame([
        {"reynolds": k[0], "angle": k[1], "n": v}
        for k, v in cov_counts.items()
    ])
    if not cov_df.empty:
        heat = cov_df.pivot(index="reynolds", columns="angle", values="n").fillna(0)
    else:
        heat = pd.DataFrame()
    heat.to_csv("tables/day1_heatmap_matrix.csv")

    plt.figure(figsize=(8, 6))
    if heat.size > 0:
        extent = [heat.columns.min(), heat.columns.max(), heat.index.min(), heat.index.max()]
        plt.imshow(heat.values, aspect="auto", origin="lower", extent=extent)
        plt.colorbar(label="Count")
    plt.title("Heatmap phu du lieu theo Reynolds va goc tan")
    plt.xlabel("Goc tan")
    plt.ylabel("Reynolds")
    _save_fig("figures/day1_heatmap_re_aoa.png")

    # -----------------------------------------------------------------------
    # Advanced diagnostics pipeline
    # -----------------------------------------------------------------------
    print("-> Dang gop du lieu va tien hanh chan doan nang cao tren tung duong cong...")
    df_all = pd.concat(all_processed_chunks, ignore_index=True)

    # Phase 1: diagnose_curve (Cd-dominant shock score + adaptive threshold)
    diagnosed_list = []
    for _, g in df_all.groupby(["name", "reynolds"], dropna=False):
        diagnosed_list.append(diagnose_curve(g, config=config))
    df_all = pd.concat(diagnosed_list, ignore_index=True)

    # Phase 2: stall onset detection
    print("-> Dang tinh stall onset score...")
    onset_list = []
    for _, g in df_all.groupby(["name", "reynolds"], dropna=False):
        onset_list.append(compute_stall_onset_score(g, config=config))
    df_all = pd.concat(onset_list, ignore_index=True)

    # Phase 3: flow regime labeling
    df_all["flow_regime"] = label_flow_regime(df_all)

    # Phase 4: quality auditing
    print("-> Tien hanh gan nhan chat luong (Quality Auditing)...")
    status_reasons = df_all.apply(
        lambda row: assign_quality(row, config=config), axis=1
    )
    df_all["quality_status"]         = [x[0] for x in status_reasons]
    df_all["reason_codes"]           = [",".join(x[1]) for x in status_reasons]
    df_all["sample_confidence_day1"] = [float(x[2]) for x in status_reasons]

    df_all["quality_status"].value_counts(dropna=False).rename("n").to_csv(
        "tables/day1_quality_status_counts.csv"
    )

    # Phase 5: confidence decomposition (vectorized)
    print("-> Dang tinh confidence decomposition...")
    df_all = _compute_confidence_vectorized(df_all, config=config)

    # Merge sample_confidence_day1 (from assign_quality row-level audit) into
    # sample_confidence (from vectorized decomposition) multiplicatively.
    # Without this, sample_confidence_day1 is computed but never used.
    if "sample_confidence_day1" in df_all.columns:
        day1_conf = pd.to_numeric(df_all["sample_confidence_day1"], errors="coerce").fillna(1.0)
        df_all["sample_confidence"] = (
            df_all["sample_confidence"] * day1_conf
        ).clip(0.0, 1.0)

    cl_conf, cd_conf, cm_conf = compute_target_confidence(df_all)
    df_all["cl_confidence"] = cl_conf
    df_all["cd_confidence"] = cd_conf
    df_all["cm_confidence"] = cm_conf

    # Phase 6: 2D density matrix + density_weight
    print("-> Dang tinh density matrix...")
    density_long, density_matrix = compute_density_matrix(df_all)
    density_matrix.to_csv("tables/day1_density_matrix.csv")
    df_all = _assign_density_weight(df_all, density_long)

    # Phase 7: target-specific sample weights
    w_cd, w_cl, w_cm = compute_sample_weights(df_all, config=config)
    df_all["sample_weight_cd"] = w_cd.values
    df_all["sample_weight_cl"] = w_cl.values
    df_all["sample_weight_cm"] = w_cm.values
    df_all["sample_weight"]    = df_all["sample_weight_cd"]   # backward-compat

    # Phase 8: derived columns
    df_all["stall_risk_score"] = compute_stall_risk_score(df_all)
    df_all["drag_ratio"]       = df_all["cd"] / (df_all["cl"] ** 2 + 0.05)

    # Phase 9: regime statistics tables
    regime_stats = df_all.groupby("flow_regime", observed=True).agg(
        n=("cd", "count"),
        mean_cd=("cd", "mean"),
        mean_cl=("cl", "mean"),
        mean_cm=("cm", "mean"),
        valid_rate=("quality_status", lambda x: (x == "valid").sum() / max(len(x), 1)),
    ).reset_index()
    regime_stats.to_csv("tables/day1_regime_statistics.csv", index=False)

    regime_metrics = compute_regime_metrics(df_all)
    regime_metrics.to_csv("tables/day1_regime_metrics.csv", index=False)

    aoa_density = compute_aoa_density(df_all)
    aoa_density.to_csv("tables/day1_aoa_density.csv", index=False)

    # Phase 10: export clean dataset
    # Keep valid + suspect; remove only invalid.
    # Drop internal-only columns before export.
    INTERNAL_COLS = ["_wedge_proxy"]
    df_clean = df_all[df_all["quality_status"] != "invalid"].copy()
    df_clean = df_clean.drop(
        columns=[c for c in INTERNAL_COLS if c in df_clean.columns],
        errors="ignore",
    )
    df_clean.to_csv("tables/deeplearwing_day1_clean.csv.gz", index=False)
    n_unique_airfoils = (
        int(df_clean["name"].astype(str).nunique()) if "name" in df_clean.columns else -1
    )
    print(
        f"-> Da xuat file sach cho Day 2 voi {len(df_clean)} / {len(df_all)} dong hop le."
    )
    print(
        f"-> Day1 clean summary: rows={len(df_clean):,}, unique_airfoils={n_unique_airfoils:,}, "
        f"output={os.path.abspath('tables/deeplearwing_day1_clean.csv.gz')}"
    )

    # -----------------------------------------------------------------------
    # New visualizations
    # -----------------------------------------------------------------------
    print("-> Dang tao visualizations...")

    # Cd vs AoA colored by quality_status
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sample = df_all.sample(min(20_000, len(df_all)), random_state=42)
        for qs, grp in sample.groupby("quality_status", observed=True):
            ax.scatter(grp["angle"], grp["cd"], s=4, alpha=0.4,
                       color=QUALITY_COLORS.get(str(qs), "#9E9E9E"), label=str(qs))
        ax.set_xlabel("AoA (deg)")
        ax.set_ylabel("Cd")
        ax.set_title("Cd vs AoA by Quality Status")
        ax.set_ylim(-0.01, 0.5)
        ax.legend(markerscale=3)
        _save_fig("figures/day1_cd_vs_aoa_quality.png")
    except Exception as e:
        print(f"[day1] Warning: cd_vs_aoa_quality: {e}"); plt.close()

    # Cd vs AoA colored by flow_regime
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Use df_clean instead of df_all to exclude invalid/infinite values
        sample = df_clean.sample(min(20_000, len(df_clean)), random_state=42)
        for reg, grp in sample.groupby("flow_regime", observed=True):
            ax.scatter(grp["angle"], grp["cd"], s=4, alpha=0.4,
                       color=REGIME_COLORS.get(str(reg), "#9E9E9E"), label=str(reg))
        ax.set_xlabel("AoA (deg)")
        ax.set_ylabel("Cd")
        ax.set_title("Cd vs AoA by Flow Regime")
        ax.set_ylim(-0.01, 0.5)
        ax.legend(markerscale=3)
        _save_fig("figures/day1_cd_vs_aoa_regime.png")
    except Exception as e:
        print(f"[day1] Warning: cd_vs_aoa_regime: {e}"); plt.close()

    # Multi-target stats by regime (3-subplot)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        regimes_ord = ["linear", "transitional", "pre_stall", "near_stall", "post_stall"]
        for ax, (col, title) in zip(axes, [("cl", "Cl"), ("cd", "Cd"), ("cm", "Cm")]):
            # Use df_clean instead of df_all to exclude invalid/infinite values
            means  = [df_clean.loc[df_clean["flow_regime"] == r, col].mean() for r in regimes_ord]
            stds   = [df_clean.loc[df_clean["flow_regime"] == r, col].std()  for r in regimes_ord]
            colors = [REGIME_COLORS.get(r, "#9E9E9E") for r in regimes_ord]
            x_pos  = list(range(len(regimes_ord)))
            ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(regimes_ord, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"Mean +/- Std {title} by Regime")
            ax.set_ylabel(title)
        plt.suptitle("Multi-Target Aerodynamic Statistics by Flow Regime")
        _save_fig("figures/day1_cl_cd_cm_by_regime.png")
    except Exception as e:
        print(f"[day1] Warning: cl_cd_cm_by_regime: {e}"); plt.close()

    # shock_score vs AoA
    try:
        ss_df = df_all.dropna(subset=["shock_score"])
        if len(ss_df) > 0:
            sample = ss_df.sample(min(15_000, len(ss_df)), random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            sc = ax.scatter(sample["angle"], sample["shock_score"], s=3, alpha=0.3,
                            c=sample["shock_score"], cmap="RdYlGn_r", vmin=0, vmax=15)
            plt.colorbar(sc, ax=ax, label="shock_score")
            ax.set_xlabel("AoA (deg)")
            ax.set_ylabel("shock_score")
            ax.set_title("Cd-Dominant Shock Score vs AoA")
            _save_fig("figures/day1_shock_score_vs_aoa.png")
        else:
            plt.close()
    except Exception as e:
        print(f"[day1] Warning: shock_score: {e}"); plt.close()

    # stall onset score histogram
    try:
        onset_vals = df_all["stall_onset_score"].dropna()
        if len(onset_vals) > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(onset_vals.clip(0, 10), bins=60, color="#F44336", alpha=0.75, edgecolor="white")
            plt.axvline(
                config["stall_onset_threshold"], color="black", linestyle="--",
                label=f"threshold={config['stall_onset_threshold']}"
            )
            plt.xlabel("stall_onset_score")
            plt.ylabel("Count")
            plt.title("Stall Onset Score Distribution")
            plt.legend()
            _save_fig("figures/day1_stall_onset_hist.png")
        else:
            plt.close()
    except Exception as e:
        print(f"[day1] Warning: stall_onset: {e}"); plt.close()

    # stall risk distribution
    try:
        risk_vals = df_all["stall_risk_score"].dropna()
        if len(risk_vals) > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(risk_vals, bins=60, color="#FF9800", alpha=0.8, edgecolor="white")
            plt.xlabel("stall_risk_score")
            plt.ylabel("Count")
            plt.title("Stall Risk Score Distribution")
            _save_fig("figures/day1_stall_risk_dist.png")
        else:
            plt.close()
    except Exception as e:
        print(f"[day1] Warning: stall_risk: {e}"); plt.close()

    # Sample weights comparison (3-subplot)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (col, color, title) in zip(axes, [
            ("sample_weight_cd", "#F44336", "Weight Cd (aggressive)"),
            ("sample_weight_cl", "#4CAF50", "Weight Cl (smooth)"),
            ("sample_weight_cm", "#2196F3", "Weight Cm (moderate)"),
        ]):
            vals = df_all[col].dropna()
            if len(vals) > 0:
                ax.hist(vals.clip(0, 8), bins=60, color=color, alpha=0.75, edgecolor="white")
                ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="mean=1.0")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.set_title(title)
                ax.legend(fontsize=8)
        plt.suptitle("Target-Specific Sample Weight Distributions")
        _save_fig("figures/day1_sample_weights_compare.png")
    except Exception as e:
        print(f"[day1] Warning: sample_weights: {e}"); plt.close()

    # Confidence components (4-subplot)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (col, color, title) in zip(axes.flat, [
            ("geometry_confidence", "#4CAF50", "Geometry Confidence"),
            ("physics_confidence",  "#2196F3", "Physics Confidence"),
            ("stall_confidence",    "#FF9800", "Stall Confidence"),
            ("solver_confidence",   "#9C27B0", "Solver Confidence"),
        ]):
            vals = df_all[col].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor="white")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.set_title(title)
        plt.suptitle("Confidence Components Distribution")
        _save_fig("figures/day1_confidence_components.png")
    except Exception as e:
        print(f"[day1] Warning: confidence_components: {e}"); plt.close()

    # Target confidence (3-subplot)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (col, color, title) in zip(axes, [
            ("cl_confidence", "#4CAF50", "Cl Confidence"),
            ("cd_confidence", "#F44336", "Cd Confidence"),
            ("cm_confidence", "#2196F3", "Cm Confidence"),
        ]):
            vals = df_all[col].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor="white")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.set_title(title)
        plt.suptitle("Target-Specific Confidence Distribution")
        _save_fig("figures/day1_target_confidence.png")
    except Exception as e:
        print(f"[day1] Warning: target_confidence: {e}"); plt.close()

    # 2D density heatmap
    try:
        if density_matrix.size > 0:
            import matplotlib.colors as mcolors
            plt.figure(figsize=(10, 5))
            max_val = float(density_matrix.values.max())
            norm = mcolors.LogNorm(vmin=1, vmax=max(max_val, 2))
            plt.imshow(
                density_matrix.values.astype(float),
                aspect="auto", origin="lower",
                norm=norm, cmap="YlOrRd",
            )
            plt.colorbar(label="n_samples (log scale)")
            plt.title("2D Density: Reynolds x AoA")
            plt.xlabel("AoA bin")
            plt.ylabel("Reynolds bin")
            _save_fig("figures/day1_density_heatmap.png")
        else:
            plt.close()
    except Exception as e:
        print(f"[day1] Warning: density_heatmap: {e}"); plt.close()

    # Cd variance by regime (boxplot)
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        regimes_ord = ["linear", "transitional", "pre_stall", "near_stall", "post_stall"]
        box_data = [
            # Use df_clean instead of df_all to exclude invalid/infinite values
            df_clean.loc[df_clean["flow_regime"] == r, "cd"].dropna().values
            for r in regimes_ord
        ]
        bp = ax.boxplot(box_data, labels=regimes_ord, patch_artist=True, showfliers=False)
        for patch, reg in zip(bp["boxes"], regimes_ord):
            patch.set_facecolor(REGIME_COLORS.get(reg, "#9E9E9E"))
            patch.set_alpha(0.75)
        ax.set_xlabel("Flow Regime")
        ax.set_ylabel("Cd")
        ax.set_title("Cd Distribution by Flow Regime")
        _save_fig("figures/day1_cd_variance_by_regime.png")
    except Exception as e:
        print(f"[day1] Warning: cd_variance_by_regime: {e}"); plt.close()

    # Drag ratio histogram
    try:
        dr_vals = df_all["drag_ratio"].dropna().clip(0, 10)
        if len(dr_vals) > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(dr_vals, bins=80, color="#FF5722", alpha=0.75, edgecolor="white")
            plt.axvline(
                config["drag_ratio_max"], color="black", linestyle="--",
                label=f"threshold={config['drag_ratio_max']}"
            )
            plt.xlabel("drag_ratio = cd / (cl^2 + 0.05)")
            plt.ylabel("Count")
            plt.title("Drag Ratio Distribution")
            plt.legend()
            _save_fig("figures/day1_drag_ratio_hist.png")
        else:
            plt.close()
    except Exception as e:
        print(f"[day1] Warning: drag_ratio: {e}"); plt.close()

    # Anomaly curve plots (existing)
    if "jump_flag" in df_all.columns:
        bad_curves = (
            df_all[df_all["jump_flag"] == True][["name", "reynolds"]]
            .drop_duplicates().head(3)
        )
        for _, row_curve in bad_curves.iterrows():
            g_sample = df_all[
                (df_all["name"] == row_curve["name"])
                & (df_all["reynolds"] == row_curve["reynolds"])
            ]
            out_img = (
                f"figures/curves/anomaly_{row_curve['name']}"
                f"_{int(row_curve['reynolds'])}.png"
            )
            plot_aoa_cl_curve(g_sample, row_curve["name"], row_curve["reynolds"], out_img)

    print("-> Day1 v2 hoan tat. Kiem tra tables/ va figures/ de xem ket qua.")


if __name__ == "__main__":
    main()
