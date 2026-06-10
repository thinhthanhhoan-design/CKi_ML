"""
day2.py

Geometry preprocessing + tabular feature engineering.

Key upgrades:
- Zero target leakage: Clear distinction between *_feature (for training) and *_audit (for auditing).
- Day1 metadata preservation: Preserves Day1 confidence, weights, regimes using multiplicative refinement.
- Dynamic fallback: Lightweight, conservative fallbacks when Day1 is missing (with warnings).
- Local geometry quality analysis: Spline oscillation and Trailing Edge quality scores.
- Improved IsolationForest input space using pure geometric indicators (zero leakage).
- Smooth Cd curve quality checks near stall to reduce false positives on monotonic drag rises.
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Import configuration from Day1 if available, otherwise define local fallback
try:
    from day1 import DAY1_CONFIG
except ImportError:
    DAY1_CONFIG: dict = {
        "curvature_spike_global": 50.0,
        "curvature_spike_te": 30.0,
        "te_osc_threshold": 5.0,
        "le_osc_threshold": 5.0,
        "te_thickness_max": 0.03,
        "wedge_angle_max_deg": 30.0,
    }


class AirfoilPreprocessingPipeline:
    def __init__(self, n_points: int = 120, contamination: float = 0.03):
        self.n_points = int(n_points)
        self.contamination = float(contamination)
        self.xg = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, self.n_points)))
        self.geometry_table: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # 1) Geometry canonicalization
    # ------------------------------------------------------------------
    def canonicalize_geometry(
        self, x_raw: np.ndarray, y_raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        x = np.asarray(x_raw, dtype=np.float64)
        y = np.asarray(y_raw, dtype=np.float64)
        if len(x) != len(y) or len(x) < 20:
            raise ValueError("invalid coordinate length")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite coordinates")

        x_min, x_max = float(np.min(x)), float(np.max(x))
        chord = x_max - x_min
        if chord <= 1e-12:
            raise ValueError("degenerate chord")

        x_norm = (x - x_min) / chord
        y_norm = y / chord

        idx_le = int(np.argmin(x_norm))
        upper_raw = np.column_stack([x_norm[: idx_le + 1], y_norm[: idx_le + 1]])[::-1]
        lower_raw = np.column_stack([x_norm[idx_le:], y_norm[idx_le:]])

        xu, yu = self._deduplicate_and_monotonize(upper_raw)
        xl, yl = self._deduplicate_and_monotonize(lower_raw)
        if len(xu) < 5 or len(xl) < 5:
            raise ValueError("too few unique points after dedup")

        method = "cubic"
        try:
            yu_res = CubicSpline(xu, yu, bc_type="natural", extrapolate=False)(
                np.clip(self.xg, xu.min(), xu.max())
            )
            yl_res = CubicSpline(xl, yl, bc_type="natural", extrapolate=False)(
                np.clip(self.xg, xl.min(), xl.max())
            )
            th = yu_res - yl_res
            if np.isnan(yu_res).any() or np.isnan(yl_res).any() or float(np.min(th)) < -1e-4:
                raise RuntimeError("cubic invalid")
        except Exception:
            method = "pchip"
            yu_res = PchipInterpolator(xu, yu, extrapolate=False)(
                np.clip(self.xg, xu.min(), xu.max())
            )
            yl_res = PchipInterpolator(xl, yl, extrapolate=False)(
                np.clip(self.xg, xl.min(), xl.max())
            )

        if not np.isfinite(yu_res).all() or not np.isfinite(yl_res).all():
            raise ValueError("interpolation produced non-finite")

        # Force shared LE point to remove tiny numeric mismatch.
        yu_res[0] = yl_res[0] = 0.5 * (yu_res[0] + yl_res[0])
        return yu_res.astype(np.float32), yl_res.astype(np.float32), method

    @staticmethod
    def _deduplicate_and_monotonize(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = coords[:, 0]
        y = coords[:, 1]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        clean_x: List[float] = []
        clean_y: List[float] = []
        last_x: float | None = None
        bucket: List[float] = []

        for xi, yi in zip(x, y):
            fx = float(xi)
            fy = float(yi)
            if last_x is None or abs(fx - last_x) < 1e-9:
                bucket.append(fy)
                last_x = fx
            else:
                clean_x.append(last_x)
                clean_y.append(float(np.median(bucket)))
                last_x = fx
                bucket = [fy]
        if last_x is not None:
            clean_x.append(last_x)
            clean_y.append(float(np.median(bucket)))

        return np.asarray(clean_x, dtype=np.float64), np.asarray(clean_y, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2) Geometry features and hard rules
    # ------------------------------------------------------------------
    def compute_geometric_features(self, yu: np.ndarray, yl: np.ndarray) -> Dict[str, float]:
        thickness = yu - yl
        camber = 0.5 * (yu + yl)

        i_tmax = int(np.argmax(thickness))
        i_cmax = int(np.argmax(camber))

        dyu = np.gradient(yu, self.xg)
        dyl = np.gradient(yl, self.xg)
        d2yu = np.gradient(dyu, self.xg)
        d2yl = np.gradient(dyl, self.xg)
        dth = np.gradient(thickness, self.xg)

        aft06 = self.xg >= 0.60
        aft = self.xg >= 0.70
        tail = self.xg >= 0.80
        mid = (self.xg >= 0.30) & (self.xg <= 0.70)
        le = self.xg <= 0.05

        # Local oscillation area masks
        le_mask = self.xg <= 0.15
        te_mask = self.xg >= 0.80

        le_slope = np.diff(thickness[:5]) / np.maximum(np.diff(self.xg[:5]), 1e-12)
        le_radius_proxy = float(np.mean(np.abs(le_slope)))

        # Use geometric angle (radians), not raw slope delta.
        trailing_edge_angle = float(abs(np.arctan(dyu[-1]) - np.arctan(dyl[-1])))
        aft_thickness = float(np.mean(thickness[aft]))
        aft_camber = float(np.mean(camber[aft]))
        curvature_energy_aft = float(np.mean(d2yu[aft] ** 2 + d2yl[aft] ** 2))
        upper_curvature_energy_aft_06 = float(np.mean(d2yu[aft06] ** 2))
        upper_aft_curvature_concentration = float(
            np.mean(np.abs(d2yu[tail])) / (np.mean(np.abs(d2yu[aft06])) + 1e-8)
        )
        slope_variance_aft = float(np.var(dyu[aft]) + np.var(dyl[aft]))
        thickness_gradient_aft = float(np.mean(dth[aft]))
        thickness_gradient_abs_aft = float(np.mean(np.abs(dth[aft])))
        upper_aft_slope_change = float(np.mean(np.abs(np.gradient(dyu[aft], self.xg[aft]))))
        lower_aft_slope_change = float(np.mean(np.abs(np.gradient(dyl[aft], self.xg[aft]))))
        wake_proxy = float(aft_thickness * trailing_edge_angle * curvature_energy_aft)
        hysteresis_proxy = float(
            upper_aft_curvature_concentration * trailing_edge_angle * max(aft_thickness, 1e-8)
        )
        aft_to_mid_thickness_ratio = float(
            aft_thickness / (float(np.mean(thickness[mid])) + 1e-8)
        )
        le_curvature_energy = float(np.mean(d2yu[le] ** 2 + d2yl[le] ** 2)) if np.any(le) else 0.0

        # --- Day4C V3 stall/separation-compatible geometry descriptors ---
        # These are geometry-only descriptors. AoA/Re-dependent interaction
        # versions are created later in assemble_final_dataset().
        sep_mid = (self.xg >= 0.10) & (self.xg <= 0.80)
        pr_aft = (self.xg >= 0.60) & (self.xg <= 0.95)
        aft_grad = (self.xg >= 0.55) & (self.xg <= 1.00)

        separation_proxy = float(np.max(np.abs(d2yu[sep_mid]))) if np.any(sep_mid) else 0.0
        upper_curvature_aft = float(np.mean(np.abs(d2yu[aft06]))) if np.any(aft06) else 0.0
        curvature_aft = float(curvature_energy_aft)  # alias expected by Day4C V2/V3

        # Pressure-recovery proxy: rear suction-side slope change + rear curvature.
        # Higher values imply stronger adverse-pressure-recovery demand near the aft body.
        if np.any(pr_aft):
            dyu_pr = dyu[pr_aft]
            d2yu_pr = d2yu[pr_aft]
            pressure_recovery_proxy = float(
                np.mean(np.abs(dyu_pr - np.median(dyu_pr))) * (1.0 + np.mean(np.abs(d2yu_pr)))
            )
            aft_pressure_recovery_proxy = float(
                np.max(np.abs(dyu_pr - dyu_pr[0])) * (1.0 + np.max(np.abs(d2yu_pr)))
            )
        else:
            pressure_recovery_proxy = 0.0
            aft_pressure_recovery_proxy = 0.0

        # Aft loading: integrated mean camber in the last 30% chord.
        aft_loading_metric = float(np.trapz(camber[aft], self.xg[aft])) if np.any(aft) else 0.0

        # Trailing-edge camber departure controls wake/pressure drag tendency.
        dcamber = np.gradient(camber, self.xg)
        d2camber = np.gradient(dcamber, self.xg)
        te_idx = int(np.argmin(np.abs(self.xg - 0.95)))
        te_curv_idx = int(np.argmin(np.abs(self.xg - 0.90)))
        te_camber_slope = float(dcamber[te_idx])
        te_camber_curvature = float(d2camber[te_curv_idx])

        # Aft curvature gradient: how abruptly rear curvature changes.
        aft_curvature_gradient = float(np.mean(np.abs(np.gradient(d2yu[aft_grad], self.xg[aft_grad])))) if np.any(aft_grad) else 0.0

        # Backward-compatible aliases expected by existing Day4C V2 utilities.
        leading_edge_radius = float(1.0 / (1.0 + le_radius_proxy))
        slope_variance = float(slope_variance_aft)

        # --- New Geometry Quality & Oscillation Metrics ---
        le_surface_oscillation = float(np.std(d2yu[le_mask]) + np.std(d2yl[le_mask])) if np.any(le_mask) else 0.0
        te_surface_oscillation = float(np.std(d2yu[te_mask]) + np.std(d2yl[te_mask])) if np.any(te_mask) else 0.0
        local_curvature_variance = float(np.var(d2yu) + np.var(d2yl))

        # Trailing edge quality score in [0, 1]
        te_gap = float(abs(yu[-1] - yl[-1]))
        te_quality = np.exp(
            -0.20 * (trailing_edge_angle / 0.5)
            - 0.30 * (te_gap / max(DAY1_CONFIG["te_thickness_max"], 0.01))
            - 0.50 * min(te_surface_oscillation / 5.0, 5.0)
        )
        trailing_edge_quality_score = float(np.clip(te_quality, 0.0, 1.0))

        # --- Moment-Aware Physics Features (Designed to avoid target leakage) ---
        front_region = self.xg <= 0.30
        mid_region = (self.xg > 0.30) & (self.xg <= 0.70)
        aft_region = self.xg > 0.70

        # Camber area and centroid
        camber_area = float(np.trapz(camber, self.xg))

        if abs(camber_area) < 1e-6:
            camber_centroid_x = 0.5
        else:
            camber_centroid_x = float(
                np.trapz(self.xg * camber, self.xg) / camber_area
            )

        # Camber regional areas
        camber_front_area = float(np.trapz(camber[front_region], self.xg[front_region])) if np.any(front_region) else 0.0
        camber_mid_area = float(np.trapz(camber[mid_region], self.xg[mid_region])) if np.any(mid_region) else 0.0
        camber_aft_area = float(np.trapz(camber[aft_region], self.xg[aft_region])) if np.any(aft_region) else 0.0

        # Camber regional slopes
        camber_slope_front = float(np.mean(dcamber[front_region])) if np.any(front_region) else 0.0
        camber_slope_mid = float(np.mean(dcamber[mid_region])) if np.any(mid_region) else 0.0
        camber_slope_aft = float(np.mean(dcamber[aft_region])) if np.any(aft_region) else 0.0

        # Camber regional curvatures
        camber_curvature_front = float(np.mean(np.abs(d2camber[front_region]))) if np.any(front_region) else 0.0
        camber_curvature_mid = float(np.mean(np.abs(d2camber[mid_region]))) if np.any(mid_region) else 0.0
        camber_curvature_aft = float(np.mean(np.abs(d2camber[aft_region]))) if np.any(aft_region) else 0.0

        # Thickness area and centroid
        thickness_area = float(np.trapz(thickness, self.xg))
        thickness_centroid_x = float(np.trapz(self.xg * thickness, self.xg) / (thickness_area + 1e-8))

        # Thickness regional areas
        thickness_front_area = float(np.trapz(thickness[front_region], self.xg[front_region])) if np.any(front_region) else 0.0
        thickness_mid_area = float(np.trapz(thickness[mid_region], self.xg[mid_region])) if np.any(mid_region) else 0.0
        thickness_aft_area = float(np.trapz(thickness[aft_region], self.xg[aft_region])) if np.any(aft_region) else 0.0

        # Thickness moment arm
        thickness_moment_arm = float(thickness_centroid_x - 0.25)

        # Thin airfoil theory pitching moment proxy
        zero_lift_moment_proxy = float(np.trapz(camber * (4.0 * self.xg - 3.0), self.xg))

        # Robust Center of Pressure proxy with division-by-zero protection
        if np.abs(camber_area) < 1e-6:
            pressure_center_proxy = 0.25
        else:
            denom_cp = np.sign(camber_area) * np.maximum(np.abs(camber_area), 1e-6)
            pressure_center_proxy = float(0.25 - zero_lift_moment_proxy / denom_cp)

        # Camber pressure distribution shape proxy
        camber_pressure_proxy = float(np.trapz(camber * self.xg * (1.0 - self.xg), self.xg))

        # Aft loading proxy
        aft_loading_proxy = float(np.trapz(camber[aft_region] * (self.xg[aft_region] - 0.70), self.xg[aft_region])) if np.any(aft_region) else 0.0

        # Variance/spread of camber moment contribution along the chord
        moment_distribution_proxy = float(np.trapz(camber * (self.xg - camber_centroid_x)**2, self.xg))

        return {
            "t_max": float(thickness[i_tmax]),
            "x_tmax": float(self.xg[i_tmax]),
            "camber_max": float(camber[i_cmax]),
            "x_cmax": float(self.xg[i_cmax]),
            "te_gap": te_gap,
            "le_radius_proxy": le_radius_proxy,
            "trailing_edge_angle": trailing_edge_angle,
            "aft_thickness": aft_thickness,
            "aft_camber": aft_camber,
            "curvature_energy_aft": curvature_energy_aft,
            "upper_curvature_energy_aft_06": upper_curvature_energy_aft_06,
            "upper_aft_curvature_concentration": upper_aft_curvature_concentration,
            "slope_variance_aft": slope_variance_aft,
            "thickness_gradient_aft": thickness_gradient_aft,
            "thickness_gradient_abs_aft": thickness_gradient_abs_aft,
            "upper_aft_slope_change": upper_aft_slope_change,
            "lower_aft_slope_change": lower_aft_slope_change,
            "wake_proxy": wake_proxy,
            "hysteresis_proxy": hysteresis_proxy,
            "aft_to_mid_thickness_ratio": aft_to_mid_thickness_ratio,
            "le_curvature_energy": le_curvature_energy,
            # Day4C V2/V3-compatible stall/separation geometry descriptors
            "separation_proxy": separation_proxy,
            "pressure_recovery_proxy": pressure_recovery_proxy,
            "aft_pressure_recovery_proxy": aft_pressure_recovery_proxy,
            "aft_curvature_gradient": aft_curvature_gradient,
            "aft_loading_metric": aft_loading_metric,
            "te_camber_slope": te_camber_slope,
            "te_camber_curvature": te_camber_curvature,
            "upper_curvature_aft": upper_curvature_aft,
            "curvature_aft": curvature_aft,
            "leading_edge_radius": leading_edge_radius,
            "slope_variance": slope_variance,
            # Exported geometry-only features
            "le_surface_oscillation": le_surface_oscillation,
            "te_surface_oscillation": te_surface_oscillation,
            "local_curvature_variance": local_curvature_variance,
            "trailing_edge_quality_score": trailing_edge_quality_score,
            # New moment-aware features
            "camber_area": camber_area,
            "camber_centroid_x": camber_centroid_x,
            "camber_front_area": camber_front_area,
            "camber_mid_area": camber_mid_area,
            "camber_aft_area": camber_aft_area,
            "camber_slope_front": camber_slope_front,
            "camber_slope_mid": camber_slope_mid,
            "camber_slope_aft": camber_slope_aft,
            "camber_curvature_front": camber_curvature_front,
            "camber_curvature_mid": camber_curvature_mid,
            "camber_curvature_aft": camber_curvature_aft,
            "thickness_centroid_x": thickness_centroid_x,
            "thickness_front_area": thickness_front_area,
            "thickness_mid_area": thickness_mid_area,
            "thickness_aft_area": thickness_aft_area,
            "thickness_moment_arm": thickness_moment_arm,
            "zero_lift_moment_proxy": zero_lift_moment_proxy,
            "pressure_center_proxy": pressure_center_proxy,
            "camber_pressure_proxy": camber_pressure_proxy,
            "aft_loading_proxy": aft_loading_proxy,
            "moment_distribution_proxy": moment_distribution_proxy,
        }

    def evaluate_hard_rules(
        self,
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        yu: np.ndarray,
        yl: np.ndarray,
    ) -> Tuple[str, List[str]]:
        flags: List[str] = []
        status = "valid"

        if len(x_raw) != len(y_raw):
            return "invalid", ["POINT_COUNT_MISMATCH"]
        if len(x_raw) < 20:
            return "invalid", ["TOO_FEW_POINTS_RAW"]
        if len(x_raw) < 50:
            flags.append("SPARSE_POINTS_SUSPECT")
            status = "suspect"
        if not np.isfinite(x_raw).all() or not np.isfinite(y_raw).all():
            return "invalid", ["NON_FINITE_COORDINATES"]
        if not np.isfinite(yu).all() or not np.isfinite(yl).all():
            return "invalid", ["NON_FINITE_INTERPOLATED_COORDINATES"]

        thickness = yu - yl
        if float(np.min(thickness)) < -1e-4:
            return "invalid", ["NEGATIVE_THICKNESS"]

        te_gap = float(abs(yu[-1] - yl[-1]))
        if te_gap > DAY1_CONFIG["te_thickness_max"]:
            flags.append("OPEN_TE_GAP_SUSPECT")
            status = "suspect"

        dyu = np.gradient(yu, self.xg)
        dyl = np.gradient(yl, self.xg)
        d2yu = np.gradient(dyu, self.xg)
        d2yl = np.gradient(dyl, self.xg)
        aft = self.xg >= 0.70
        mid_body = (self.xg >= 0.03) & (self.xg <= 0.97)

        curv_aft = float(np.mean(d2yu[aft] ** 2 + d2yl[aft] ** 2))
        te_angle = float(abs(np.arctan(dyu[-1]) - np.arctan(dyl[-1])))

        if te_angle > (DAY1_CONFIG["wedge_angle_max_deg"] * np.pi / 180.0):
            flags.append("TE_ANGLE_EXTREME_SUSPECT")
            status = "suspect"
        if curv_aft > 500.0:
            flags.append("AFT_CURVATURE_SPIKE_SUSPECT")
            status = "suspect"

        # Avoid false positives near LE/TE where thickness naturally approaches zero.
        if np.any(thickness[mid_body] < 0.002) and float(np.max(thickness)) > 0.05:
            flags.append("LOCAL_THICKNESS_COLLAPSE_SUSPECT")
            status = "suspect"

        return status, flags

    # ------------------------------------------------------------------
    # 3) PASS 2: geometry database
    # ------------------------------------------------------------------
    def process_geometry_database(self, raw_csv_path: str, chunksize: int = 50_000) -> None:
        print("-> [Day2] PASS2: build geometry table")
        processed_names: set[str] = set()
        geom_records: List[Dict[str, object]] = []

        first_row = pd.read_csv(raw_csv_path, nrows=1)
        actual_cols = first_row.columns.tolist()
        col_mapping = {}
        if "airfoil_name" in actual_cols and "name" not in actual_cols:
            col_mapping["airfoil_name"] = "name"

        csv_usecols = []
        for col in ["name", "x_coords", "y_coords"]:
            mapped_src = None
            for src, dest in col_mapping.items():
                if dest == col:
                    mapped_src = src
                    break
            if mapped_src is not None and mapped_src in actual_cols:
                csv_usecols.append(mapped_src)
            elif col in actual_cols:
                csv_usecols.append(col)

        for chunk in pd.read_csv(
            raw_csv_path,
            usecols=csv_usecols,
            chunksize=int(chunksize),
        ):
            chunk = chunk.rename(columns=col_mapping)
            for name, grp in chunk.groupby("name"):
                if pd.isna(name):
                    continue
                name_str = str(name)
                if name_str in processed_names:
                    continue
                processed_names.add(name_str)

                first_row = grp.iloc[0]
                x_raw = np.fromstring(str(first_row["x_coords"]), sep=" ")
                y_raw = np.fromstring(str(first_row["y_coords"]), sep=" ")
                if len(x_raw) == 0 or len(y_raw) == 0:
                    continue

                try:
                    yu, yl, method = self.canonicalize_geometry(x_raw, y_raw)
                    geom_vec = np.concatenate([yu, yl])
                    geom_hash = hashlib.sha1(np.round(geom_vec, 6).tobytes()).hexdigest()
                    rule_status, rule_flags = self.evaluate_hard_rules(x_raw, y_raw, yu, yl)
                    phys_feats = self.compute_geometric_features(yu, yl)

                    rec: Dict[str, object] = {
                        "name": name_str,
                        "geom_hash": geom_hash,
                        "n_points_raw": int(len(x_raw)),
                        "rule_status": rule_status,
                        "rule_flags": ",".join(rule_flags),
                        "resample_method": method,
                        **phys_feats,
                    }
                    for i, val in enumerate(geom_vec):
                        rec[f"g_{i:03d}"] = float(val)
                    geom_records.append(rec)
                except Exception as exc:
                    geom_records.append(
                        {
                            "name": name_str,
                            "geom_hash": "INVALID_GEOM",
                            "n_points_raw": int(len(x_raw)),
                            "rule_status": "invalid",
                            "rule_flags": f"GEOMETRY_PROCESSING_ERROR:{type(exc).__name__}",
                            "resample_method": "failed",
                        }
                    )

        geom_df = pd.DataFrame(geom_records)
        if geom_df.empty:
            raise RuntimeError("geometry table is empty")

        print("-> [Day2] PASS2: IsolationForest geometry outlier screening (strictly zero target leakage)")
        valid_geom_df = geom_df[geom_df["rule_status"] != "invalid"].copy()

        if len(valid_geom_df) > 10:
            g_cols = [c for c in valid_geom_df.columns if c.startswith("g_")]
            phys_cols = [
                "t_max",
                "x_tmax",
                "camber_max",
                "x_cmax",
                "te_gap",
                "le_radius_proxy",
                "trailing_edge_angle",
                "aft_thickness",
                "aft_camber",
                "curvature_energy_aft",
                "upper_curvature_energy_aft_06",
                "upper_aft_curvature_concentration",
                "slope_variance_aft",
                "thickness_gradient_aft",
                "thickness_gradient_abs_aft",
                "upper_aft_slope_change",
                "lower_aft_slope_change",
                "wake_proxy",
                "hysteresis_proxy",
                "aft_to_mid_thickness_ratio",
                "le_curvature_energy",
                # Day4C V2/V3-compatible stall/separation features
                "separation_proxy",
                "pressure_recovery_proxy",
                "aft_pressure_recovery_proxy",
                "aft_curvature_gradient",
                "aft_loading_metric",
                "te_camber_slope",
                "te_camber_curvature",
                "upper_curvature_aft",
                "curvature_aft",
                "leading_edge_radius",
                "slope_variance",
                # New geometric features in IF space
                "le_surface_oscillation",
                "te_surface_oscillation",
                "local_curvature_variance",
                "trailing_edge_quality_score",
            ]
            phys_cols = [c for c in phys_cols if c in valid_geom_df.columns]

            x_geom = valid_geom_df[g_cols].to_numpy(dtype=float)
            x_phys = valid_geom_df[phys_cols].to_numpy(dtype=float)
            x_combined = np.hstack([x_geom, x_phys])
            x_scaled = StandardScaler().fit_transform(x_combined)

            n_pca = min(15, x_scaled.shape[1], len(valid_geom_df) - 1)
            if n_pca > 1:
                x_pca = PCA(n_components=n_pca, random_state=42).fit_transform(x_scaled)
                x_iso = np.hstack([x_scaled, x_pca])
            else:
                x_iso = x_scaled

            iso = IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            preds = iso.fit_predict(x_iso)
            scores = iso.decision_function(x_iso)

            valid_geom_df["if_pred"] = preds
            valid_geom_df["iforest_keep"] = np.where(preds == 1, 1, 0)
            valid_geom_df["iforest_score"] = scores

            lo = float(np.percentile(scores, 5))
            hi = float(np.percentile(scores, 95))
            denom = max(hi - lo, 1e-12)
            scaled = np.clip((scores - lo) / denom, 0.0, 1.0)
            valid_geom_df["iforest_confidence"] = 0.3 + 0.7 * scaled
        else:
            valid_geom_df["if_pred"] = 1
            valid_geom_df["iforest_keep"] = 1
            valid_geom_df["iforest_score"] = 0.0
            valid_geom_df["iforest_confidence"] = 1.0

        geom_df = geom_df.merge(
            valid_geom_df[
                ["name", "if_pred", "iforest_keep", "iforest_score", "iforest_confidence"]
            ],
            on="name",
            how="left",
        )
        geom_df["if_pred"] = geom_df["if_pred"].fillna(-1).astype(int)
        geom_df["iforest_keep"] = geom_df["iforest_keep"].fillna(0).astype(int)
        geom_df["iforest_score"] = pd.to_numeric(geom_df["iforest_score"], errors="coerce").fillna(-999.0)
        geom_df["iforest_confidence"] = (
            pd.to_numeric(geom_df["iforest_confidence"], errors="coerce").fillna(0.3).clip(0.05, 1.0)
        )

        self.geometry_table = geom_df
        os.makedirs("tables", exist_ok=True)
        geom_df.to_csv("tables/airfoil_geometry_240.csv", index=False)
        geom_df[
            [
                "name",
                "rule_status",
                "rule_flags",
                "if_pred",
                "iforest_keep",
                "iforest_score",
                "iforest_confidence",
            ]
        ].to_csv("tables/geometry_rule_filter_report.csv", index=False)
        print(f"-> [Day2] PASS2 done: tables/airfoil_geometry_240.csv | n={len(geom_df)}")

    # ------------------------------------------------------------------
    # 4) PASS 3: assemble final tabular dataset
    # ------------------------------------------------------------------
    @staticmethod
    def assign_aoa_density_region(angle: float) -> str:
        a = abs(float(angle))
        if a < 5.0:
            return "attached_coarse"
        if a < 8.0:
            return "transition_5_8_dense"
        if a < 10.0:
            return "transition_8_10_dense"
        if a < 12.0:
            return "prestall_10_12_dense"
        if a < 14.0:
            return "stall_onset_12_14_dense"
        return "poststall_coarse"

    def assemble_final_dataset(self, raw_csv_path: str, chunksize: int = 100_000) -> None:
        print("-> [Day2] PASS3: assemble tabular dataset")
        if self.geometry_table is None:
            raise ValueError("run process_geometry_database() first")

        geom_indexed = self.geometry_table.set_index("name")
        output_csv = "deeplearwing_day2_tabular.csv"
        if os.path.exists(output_csv):
            os.remove(output_csv)
        first_chunk = True

        dtypes = {
            "name": "category",
            "angle": "float32",
            "reynolds": "float32",
            "cl": "float32",
            "cd": "float32",
            "cm": "float32",
        }
        
        # Day1 metadata columns to read and propagate
        DAY1_METADATA_COLS = [
            "flow_regime",
            "stall_risk_score",
            "shock_score",
            "adaptive_threshold",
            "drag_ratio",
            "geometry_confidence",
            "physics_confidence",
            "stall_confidence",
            "solver_confidence",
            "sample_weight_cl",
            "sample_weight_cd",
            "sample_weight_cm",
            "cl_confidence",
            "cd_confidence",
            "cm_confidence",
            "density_weight",
        ]

        first_row = pd.read_csv(raw_csv_path, nrows=1)
        actual_cols = first_row.columns.tolist()
        col_mapping = {}
        if "airfoil_name" in actual_cols and "name" not in actual_cols:
            col_mapping["airfoil_name"] = "name"
        if "alpha" in actual_cols and "angle" not in actual_cols:
            col_mapping["alpha"] = "angle"

        available_cols = actual_cols
        input_cols = list(dtypes.keys())
        if "sample_confidence" in available_cols:
            input_cols.append("sample_confidence")
        elif "sample_confidence_day1" in available_cols:
            input_cols.append("sample_confidence_day1")

        # Dynamically append Day1 metadata columns if they exist in the input file
        for col in DAY1_METADATA_COLS:
            if col in available_cols and col not in input_cols:
                input_cols.append(col)

        csv_usecols = []
        csv_dtypes = {}
        for col in input_cols:
            mapped_src = None
            for src, dest in col_mapping.items():
                if dest == col:
                    mapped_src = src
                    break
            if mapped_src is not None and mapped_src in actual_cols:
                csv_usecols.append(mapped_src)
                if col in dtypes:
                    csv_dtypes[mapped_src] = dtypes[col]
            elif col in actual_cols:
                csv_usecols.append(col)
                if col in dtypes:
                    csv_dtypes[col] = dtypes[col]

        for chunk in pd.read_csv(
            raw_csv_path,
            usecols=csv_usecols,
            dtype=csv_dtypes,
            chunksize=int(chunksize),
        ):
            chunk = chunk.rename(columns=col_mapping)
            chunk["name"] = chunk["name"].astype(str)
            joined = chunk.join(geom_indexed, on="name", how="inner")

            # Keep valid + suspect. Drop only truly invalid geometry.
            cur = joined[joined["rule_status"] != "invalid"].copy()
            if cur.empty:
                continue

            for col in ["angle", "reynolds", "cl", "cd", "cm"]:
                cur[col] = pd.to_numeric(cur[col], errors="coerce")
            cur = cur.dropna(subset=["angle", "reynolds", "cl", "cd", "cm"])
            cur = cur[(cur["cd"] > 0.0) & (cur["reynolds"] > 0.0)].copy()
            if cur.empty:
                continue

            # --- Target Leakage Prevention & Regime-Aware Feature Engineering ---
            cur["abs_aoa_feature"] = np.abs(cur["angle"]).astype(np.float32)

            # Flow Regime propagation / lightweight fallback
            if "flow_regime" in cur.columns:
                cur["flow_regime"] = cur["flow_regime"].astype(str)
            else:
                abs_a = cur["abs_aoa_feature"].values
                cur["flow_regime"] = np.select(
                    [abs_a < 4.0, abs_a < 8.0, abs_a < 12.0, abs_a < 16.0],
                    ["linear", "transitional", "pre_stall", "near_stall"],
                    default="post_stall"
                )

            # Regime Index Feature (Pure geometry + AoA)
            regime_map = {"linear": 0, "transitional": 1, "pre_stall": 2, "near_stall": 3, "post_stall": 4}
            cur["regime_index"] = cur["flow_regime"].map(regime_map).fillna(4).astype(np.int32)

            # Stall Risk Score propagation / lightweight geometry fallback
            if "stall_risk_score" in cur.columns:
                cur["stall_risk_score"] = pd.to_numeric(cur["stall_risk_score"], errors="coerce").fillna(0.0)
                cur["stall_risk_score_audit"] = cur["stall_risk_score"].astype(np.float32)
            else:
                # Fallback stall_risk_score chỉ dùng khi không có Day1, không được thay thế Day1 score.
                # Lightweight and conservative, strictly annotated as fallback.
                cur["stall_risk_score"] = np.clip((cur["abs_aoa_feature"] - 8.0) / 12.0, 0.0, 1.0).astype(np.float32)
                cur["stall_risk_score_audit"] = cur["stall_risk_score"]

            # Regime Transition Score (Calculated strictly from AoA + Aft Geometry features, NO Cl/Cd/Cm target leakage)
            wake_pr = np.abs(cur["wake_proxy"].fillna(0.0).values)
            curv_af = cur["curvature_energy_aft"].fillna(0.0).values
            q95_wake = float(np.nanpercentile(wake_pr, 95)) if len(wake_pr) else 1.0
            q95_curv = float(np.nanpercentile(curv_af, 95)) if len(curv_af) else 1.0
            wake_scaled = np.clip(wake_pr / max(q95_wake, 1e-5), 0.0, 1.0)
            curv_scaled = np.clip(curv_af / max(q95_curv, 1e-5), 0.0, 1.0)
            
            cur["regime_transition_score_feature"] = (
                0.5 * np.clip(cur["abs_aoa_feature"] / 16.0, 0.0, 1.0)
                + 0.3 * wake_scaled
                + 0.2 * curv_scaled
            ).astype(np.float32)

            # Stall-Transition Descriptors (Pure training features, no targets in calculation)
            cur["drag_growth_proxy_feature"] = (cur["wake_proxy"].fillna(0.0) * (cur["abs_aoa_feature"] / 15.0) ** 2).astype(np.float32)
            cur["lift_break_proxy_feature"] = (cur["hysteresis_proxy"].fillna(0.0) * (cur["abs_aoa_feature"] / 15.0)).astype(np.float32)
            cur["hysteresis_risk_feature"] = (cur["hysteresis_proxy"].fillna(0.0) * (cur["abs_aoa_feature"] / 15.0)).astype(np.float32)

            # Day4C V3 physics-informed stall features. These are safe:
            # they use geometry + AoA + Re only, never Cl/Cd/Cm targets.
            sep_base = pd.to_numeric(cur.get("separation_proxy", 0.0), errors="coerce").fillna(0.0)
            pr_base = pd.to_numeric(cur.get("pressure_recovery_proxy", cur["wake_proxy"]), errors="coerce").fillna(0.0)
            aft_load_base = pd.to_numeric(cur.get("aft_loading_metric", cur["aft_camber"]), errors="coerce").fillna(0.0)
            aft_grad_base = pd.to_numeric(cur.get("aft_curvature_gradient", cur["curvature_energy_aft"]), errors="coerce").fillna(0.0)
            curv_aft_base = pd.to_numeric(cur.get("curvature_aft", cur["curvature_energy_aft"]), errors="coerce").fillna(0.0)

            log_re_safe = np.log10(np.maximum(pd.to_numeric(cur["reynolds"], errors="coerce").fillna(1.0), 1.0))
            abs_aoa_norm = cur["abs_aoa_feature"] / 15.0

            cur["separation_proxy_feature"] = (sep_base * abs_aoa_norm).astype(np.float32)
            cur["pressure_recovery_proxy_feature"] = (pr_base * abs_aoa_norm).astype(np.float32)
            cur["aft_loading_feature"] = (aft_load_base * abs_aoa_norm).astype(np.float32)
            cur["aft_curvature_gradient_feature"] = (aft_grad_base * abs_aoa_norm).astype(np.float32)

            cur["abs_aoa_x_separation_proxy"] = (cur["abs_aoa_feature"] * sep_base).astype(np.float32)
            cur["abs_aoa_x_pressure_recovery_proxy"] = (cur["abs_aoa_feature"] * pr_base).astype(np.float32)
            cur["abs_aoa_x_aft_loading_metric"] = (cur["abs_aoa_feature"] * aft_load_base).astype(np.float32)
            cur["abs_aoa_x_curvature_aft"] = (cur["abs_aoa_feature"] * curv_aft_base).astype(np.float32)

            # Operating-point stall severity: geometry risk amplified by AoA and low-Re effects.
            cur["stall_severity_proxy"] = (
                np.power(cur["abs_aoa_feature"], 1.3)
                * (0.45 * sep_base + 0.35 * pr_base + 0.20 * np.abs(aft_load_base))
                * (1.0 + np.clip(aft_grad_base, 0.0, None))
                / np.maximum(log_re_safe, 1.0)
            ).astype(np.float32)
            cur["stall_severity_proxy_feature"] = cur["stall_severity_proxy"].astype(np.float32)

            # Auditing Columns (Calculated from Cl/Cd targets, strictly labeled *_audit)
            cur["drag_ratio_audit"] = (cur["cd"] / (cur["cl"] ** 2 + 0.05)).astype(np.float32)
            cur["drag_growth_proxy_audit"] = (cur["cd"] / (1.0 + np.exp(-(cur["angle"].abs() - 10.0) / 2.0))).astype(np.float32)
            cur["lift_break_proxy_audit"] = ((1.0 - cur["cl"].clip(-2, 1.8) / 1.8) * cur["stall_risk_score"]).astype(np.float32)
            cur["hysteresis_risk_audit"] = (cur["hysteresis_proxy"].fillna(0.0) * cur["stall_risk_score"] * (1.0 + cur["abs_aoa_feature"] / 15.0)).astype(np.float32)
            cur["wake_growth_proxy_audit"] = (cur["wake_proxy"].fillna(0.0) * (1.0 + cur["abs_aoa_feature"] / 10.0) * (cur["cd"] / 0.05).clip(0.1, 5.0)).astype(np.float32)

            # --- Day1 Confidence & Weight Preservation / Dynamic Refinement ---
            if "sample_confidence_day1" in cur.columns:
                base_confidence = pd.to_numeric(cur["sample_confidence_day1"], errors="coerce").fillna(1.0)
            elif "sample_confidence" in cur.columns:
                base_confidence = pd.to_numeric(cur["sample_confidence"], errors="coerce").fillna(1.0)
            else:
                base_confidence = pd.Series(1.0, index=cur.index)

            # Load Day1 target-specific confidence metrics if present
            cl_conf_in = cur["cl_confidence"].fillna(base_confidence) if "cl_confidence" in cur.columns else base_confidence
            cd_conf_in = cur["cd_confidence"].fillna(base_confidence) if "cd_confidence" in cur.columns else base_confidence
            cm_conf_in = cur["cm_confidence"].fillna(base_confidence) if "cm_confidence" in cur.columns else base_confidence

            # Geometry Refinement from Day2 features
            f_refine = np.ones(len(cur), dtype=np.float32)
            f_refine = np.where(cur["rule_status"] == "suspect", f_refine * 0.5, f_refine)
            if "iforest_confidence" in cur.columns:
                f_refine = f_refine * pd.to_numeric(cur["iforest_confidence"], errors="coerce").fillna(0.3).values
            else:
                f_refine = f_refine * 0.8

            if "iforest_keep" in cur.columns:
                f_refine = np.where(cur["iforest_keep"] == 0, f_refine * 0.7, f_refine)

            # AoA and geometry severity penalties
            abs_angle = cur["abs_aoa_feature"].values
            aoa_confidence = np.exp(-np.maximum(abs_angle - 10.0, 0.0) / 6.0)

            # Stall severity proxy (strictly geometry-based)
            log_re = np.log10(np.maximum(pd.to_numeric(cur["reynolds"], errors="coerce").fillna(1.0), 1.0))
            aft_curv_conc = np.abs(cur["upper_aft_curvature_concentration"].fillna(0.0).values)
            stall_sev_proxy = (
                np.power(abs_angle, 1.3)
                * wake_pr
                * (1.0 + aft_curv_conc)
                / np.maximum(log_re, 1.0)
            )
            q90_sev = float(np.nanpercentile(stall_sev_proxy[np.isfinite(stall_sev_proxy)], 90)) if np.any(np.isfinite(stall_sev_proxy)) else 1.0
            q90_sev = max(q90_sev, 1e-6)
            stall_sev_confidence = np.exp(-0.20 * np.clip(stall_sev_proxy / q90_sev, 0.0, 6.0))

            # Consolidated geometry-based refinement factor
            total_refine = f_refine * aoa_confidence * stall_sev_confidence

            # Target-specific Refined Confidence Metrics
            cur["cl_confidence_refined"] = (cl_conf_in * total_refine).clip(0.05, 1.0).astype(np.float32)
            cur["cd_confidence_refined"] = (cd_conf_in * total_refine).clip(0.05, 1.0).astype(np.float32)  # cd_curve_confidence is multiplied later in PASS 3.1
            cur["cm_confidence_refined"] = (cm_conf_in * total_refine * cur["trailing_edge_quality_score"]).clip(0.05, 1.0).astype(np.float32)
            
            # Backward-compatible sample_confidence (uses Cd refined confidence)
            cur["sample_confidence"] = cur["cd_confidence_refined"]

            # Target-Specific Weight Generation (Used in loss function only, not for training features)
            REGIME_TRANSITION_WEIGHT = {
                "linear": 1.0, "transitional": 1.2, "pre_stall": 1.5, "near_stall": 2.0, "post_stall": 2.2, "unknown": 1.0
            }
            MILD_REGIME_WEIGHT = {
                "linear": 1.0, "transitional": 1.1, "pre_stall": 1.2, "near_stall": 1.4, "post_stall": 1.5, "unknown": 1.0
            }
            MODERATE_REGIME_WEIGHT = {
                "linear": 1.0, "transitional": 1.1, "pre_stall": 1.4, "near_stall": 1.6, "post_stall": 1.8, "unknown": 1.0
            }

            r_trans = cur["flow_regime"].map(REGIME_TRANSITION_WEIGHT).fillna(1.0).values
            r_mild  = cur["flow_regime"].map(MILD_REGIME_WEIGHT).fillna(1.0).values
            r_mod   = cur["flow_regime"].map(MODERATE_REGIME_WEIGHT).fillna(1.0).values

            cd_weight = np.clip(cur["cd"] / 0.08, 1.0, 4.0).values
            high_aoa_weight = np.where(cur["abs_aoa_feature"] >= 8.0, 2.0, 1.0)
            density_weight = cur["density_weight"].fillna(1.0).values if "density_weight" in cur.columns else np.ones(len(cur))

            # Apply multiplicative refinement if Day1 weights are present, otherwise compute from scratch
            if "sample_weight_cl" in cur.columns:
                cur["sample_weight_cl"] = (cur["sample_weight_cl"].fillna(1.0) * total_refine).clip(0.1, 10.0).astype(np.float32)
            else:
                cur["sample_weight_cl"] = (cur["cl_confidence_refined"] * r_mild).clip(0.1, 10.0).astype(np.float32)

            if "sample_weight_cd" in cur.columns:
                cur["sample_weight_cd"] = (cur["sample_weight_cd"].fillna(1.0) * total_refine).clip(0.1, 10.0).astype(np.float32)
            else:
                cur["sample_weight_cd"] = (cur["cd_confidence_refined"] * cd_weight * high_aoa_weight * density_weight * r_trans).clip(0.1, 10.0).astype(np.float32)

            if "sample_weight_cm" in cur.columns:
                cur["sample_weight_cm"] = (cur["sample_weight_cm"].fillna(1.0) * total_refine * cur["trailing_edge_quality_score"]).clip(0.1, 10.0).astype(np.float32)
            else:
                cur["sample_weight_cm"] = (cur["cm_confidence_refined"] * r_mod * cur["trailing_edge_quality_score"]).clip(0.1, 10.0).astype(np.float32)

            # Backward-compatible sample_weight (mapped to cd weight for Day4)
            cur["sample_weight"] = cur["sample_weight_cd"]

            # Additional tabular geometry inputs (pure features)
            cur["log10_re"] = np.log10(cur["reynolds"]).astype(np.float32)
            cur["aoa_x_t"] = (cur["angle"] * cur["t_max"]).astype(np.float32)
            cur["aoa_x_camber"] = (cur["angle"] * cur["camber_max"]).astype(np.float32)
            if "trailing_edge_angle" in cur.columns:
                cur["aoa_x_te_angle"] = (cur["angle"] * cur["trailing_edge_angle"]).astype(np.float32)
            if "curvature_energy_aft" in cur.columns:
                cur["aoa_x_aft_curvature"] = (cur["angle"] * cur["curvature_energy_aft"]).astype(np.float32)
            if "wake_proxy" in cur.columns:
                cur["abs_aoa_x_wake_proxy"] = (np.abs(cur["angle"]) * cur["wake_proxy"]).astype(np.float32)
            if "upper_aft_curvature_concentration" in cur.columns:
                cur["abs_aoa_x_upper_aft_curvature_concentration"] = (
                    np.abs(cur["angle"]) * cur["upper_aft_curvature_concentration"]
                ).astype(np.float32)
            if "hysteresis_proxy" in cur.columns:
                cur["abs_aoa_x_hysteresis_proxy"] = (
                    np.abs(cur["angle"]) * cur["hysteresis_proxy"]
                ).astype(np.float32)
            # The following columns may already be created above; keep them stable
            # and ensure they exist for downstream Day4C feature discovery.
            for _col in [
                "abs_aoa_x_separation_proxy",
                "abs_aoa_x_pressure_recovery_proxy",
                "abs_aoa_x_aft_loading_metric",
                "abs_aoa_x_curvature_aft",
                "separation_proxy_feature",
                "pressure_recovery_proxy_feature",
                "aft_loading_feature",
                "aft_curvature_gradient_feature",
                "stall_severity_proxy_feature",
            ]:
                if _col not in cur.columns:
                    cur[_col] = 0.0
                cur[_col] = pd.to_numeric(cur[_col], errors="coerce").fillna(0.0).astype(np.float32)
            cur["aoa_density_region"] = cur["angle"].map(self.assign_aoa_density_region)

            # Save tabular chunk
            cur.to_csv(
                output_csv,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False

        if first_chunk:
            raise RuntimeError("No rows were written to deeplearwing_day2_tabular.csv")

        self.add_cd_curve_quality(output_csv)
        print(f"=== [DAY2] DONE: {output_csv} ===")

    # ------------------------------------------------------------------
    # 5) Cd(AoA) quality checks
    # ------------------------------------------------------------------
    def add_cd_curve_quality(self, output_csv: str) -> None:
        print("-> [Day2] PASS3.1: Cd(AoA) curve quality checks")
        if not os.path.exists(output_csv):
            raise FileNotFoundError(output_csv)

        df = pd.read_csv(output_csv, low_memory=False)
        df["cd_spike_flag"] = False
        df["cd_drop_flag"] = False
        df["cd_curve_confidence"] = 1.0
        df["cd_local_roughness"] = 0.0
        df["cd_relative_jump"] = 0.0

        for _, idx in df.groupby(["geom_hash", "reynolds"]).groups.items():
            g = df.loc[idx].sort_values("angle")
            if len(g) < 5:
                continue

            aoa = g["angle"].to_numpy(dtype=float)
            cd = g["cd"].to_numpy(dtype=float)
            valid = np.isfinite(aoa) & np.isfinite(cd) & (cd > 0.0)
            if int(np.sum(valid)) < 5:
                continue

            g = g.loc[valid].sort_values("angle")
            aoa = g["angle"].to_numpy(dtype=float)
            cd = g["cd"].to_numpy(dtype=float)
            if len(np.unique(aoa)) < 5:
                continue

            h1 = aoa[1:-1] - aoa[:-2]
            h2 = aoa[2:] - aoa[1:-1]
            good = (np.abs(h1) > 1e-8) & (np.abs(h2) > 1e-8)
            if not np.any(good):
                continue

            slope1 = np.zeros_like(h1)
            slope2 = np.zeros_like(h2)
            slope1[good] = (cd[1:-1][good] - cd[:-2][good]) / h1[good]
            slope2[good] = (cd[2:][good] - cd[1:-1][good]) / h2[good]
            local_roughness = np.abs(slope2 - slope1) / (np.abs(cd[1:-1]) + 1e-4)

            t = h1 / (h1 + h2 + 1e-12)
            local_linear = cd[:-2] + t * (cd[2:] - cd[:-2])
            relative_jump = np.abs(cd[1:-1] - local_linear) / (np.abs(cd[1:-1]) + 1e-4)

            sensitive = np.abs(aoa[1:-1]) >= 5.0
            
            # Intelligent physics-aware spike/drop detection (prevents smooth stall rise false positives)
            local_min_neighbors = np.minimum(cd[:-2], cd[2:])
            local_max_neighbors = np.maximum(cd[:-2], cd[2:])
            is_monotonic_rise = (cd[1:-1] > cd[:-2]) & (cd[2:] > cd[1:-1])
            is_stall = np.abs(aoa[1:-1]) >= 10.0

            drop = (cd[1:-1] < 0.50 * local_min_neighbors) & sensitive
            # Spike must be extremely aggressive (>2.5x) to be classified as numerical error in high-AoA zones
            spike = (cd[1:-1] > 2.50 * np.maximum(local_max_neighbors, 1e-8)) & sensitive

            lr_thr = float(np.percentile(local_roughness, 90)) if len(local_roughness) else np.inf
            severe_lr_thr = float(np.percentile(local_roughness, 97)) if len(local_roughness) else np.inf

            # Scale thresholds dynamically in case of monotonic rise or stall
            lr_threshold = np.where(is_monotonic_rise, lr_thr * 3.0, lr_thr)
            lr_threshold = np.where(is_stall & ~is_monotonic_rise, lr_threshold * 2.0, lr_threshold)
            
            jump_threshold = np.where(is_monotonic_rise, 0.80, 0.32)
            jump_threshold = np.where(is_stall & ~is_monotonic_rise, 0.60, jump_threshold)

            bad = ((relative_jump > jump_threshold) | (local_roughness > lr_threshold) | drop | spike) & sensitive
            severe = ((relative_jump > 2.0 * jump_threshold) | (local_roughness > 2.0 * lr_threshold) | drop | spike) & sensitive

            mid_idx = g.index[1:-1]
            bad_idx = mid_idx[bad]
            severe_idx = mid_idx[severe]
            df.loc[mid_idx, "cd_local_roughness"] = local_roughness
            df.loc[mid_idx, "cd_relative_jump"] = relative_jump
            df.loc[bad_idx, "cd_spike_flag"] = True
            df.loc[mid_idx[drop], "cd_drop_flag"] = True
            
            df.loc[bad_idx, "cd_curve_confidence"] = np.minimum(
                pd.to_numeric(df.loc[bad_idx, "cd_curve_confidence"], errors="coerce").fillna(1.0),
                0.25,
            )
            df.loc[severe_idx, "cd_curve_confidence"] = np.minimum(
                pd.to_numeric(df.loc[severe_idx, "cd_curve_confidence"], errors="coerce").fillna(1.0),
                0.12,
            )

        # Multiplicative Refinement of CD target-specific confidence and weight
        df["cd_confidence_refined"] = (
            pd.to_numeric(df["cd_confidence_refined"], errors="coerce").fillna(1.0)
            * pd.to_numeric(df["cd_curve_confidence"], errors="coerce").fillna(1.0)
        ).clip(0.05, 1.0).astype(np.float32)

        # Backward-compatible sample_confidence
        df["sample_confidence"] = df["cd_confidence_refined"]

        # Refine Cd Weight
        df["sample_weight_cd"] = (
            pd.to_numeric(df["sample_weight_cd"], errors="coerce").fillna(1.0)
            * pd.to_numeric(df["cd_curve_confidence"], errors="coerce").fillna(1.0)
        ).clip(0.1, 10.0).astype(np.float32)

        # Mean-normalize all three weights so each stays centered at ~1.0 (target mean).
        # The cd_curve_confidence refinement can shift the global scale; this restores it
        # without changing relative rankings. Clip after normalization for safety.
        for _wcol, _wmax in [
            ("sample_weight_cd", 10.0),
            ("sample_weight_cl",  5.0),
            ("sample_weight_cm",  5.0),
        ]:
            if _wcol in df.columns:
                _w = pd.to_numeric(df[_wcol], errors="coerce").fillna(1.0)
                _mean = float(_w.mean())
                if _mean > 1e-8:
                    _w = _w / _mean
                df[_wcol] = _w.clip(0.1, _wmax).astype(np.float32)

        # Backward-compatible sample_weight
        df["sample_weight"] = df["sample_weight_cd"]

        os.makedirs("tables", exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        # Save curve quality report
        df[
            [
                "name",
                "geom_hash",
                "reynolds",
                "angle",
                "cd",
                "cd_spike_flag",
                "cd_drop_flag",
                "cd_curve_confidence",
                "cd_local_roughness",
                "cd_relative_jump",
                "sample_confidence",
                "sample_weight",
            ]
        ].to_csv("tables/day2_cd_curve_quality_report.csv", index=False)
        print("-> [Day2] saved tables/day2_cd_curve_quality_report.csv")

        # PASS 3.2: Export tables/day2_regime_metrics.csv (multi-target statistics by regime)
        print("-> [Day2] exporting tables/day2_regime_metrics.csv")
        regime_metrics = df.groupby("flow_regime", observed=True).agg(
            n=("cd", "count"),
            mean_cl=("cl", "mean"),
            var_cl=("cl", "var"),
            mean_cd=("cd", "mean"),
            var_cd=("cd", "var"),
            mean_cm=("cm", "mean"),
            var_cm=("cm", "var"),
            mean_confidence=("sample_confidence", "mean"),
        ).reset_index()
        regime_metrics.to_csv("tables/day2_regime_metrics.csv", index=False)


if __name__ == "__main__":
    csv_input_path = "tables/deeplearwing_day1_clean.csv"
    if not os.path.exists(csv_input_path):
        csv_input_path = "DeepLearWing.csv"

    if not os.path.exists(csv_input_path):
        print(f"Warning: missing {csv_input_path}. Creating a tiny mock file for smoke testing.")
        csv_input_path = "tables/deeplearwing_day1_clean.csv"
        os.makedirs("tables", exist_ok=True)
        x_u = np.linspace(1.0, 0.0, 18)
        y_u = 0.07 * np.sqrt(np.clip(x_u, 0.0, 1.0)) * (1.0 - x_u)
        x_l = np.linspace(0.0, 1.0, 18)
        y_l = -0.07 * np.sqrt(np.clip(x_l, 0.0, 1.0)) * (1.0 - x_l)
        x_all = np.concatenate([x_u, x_l[1:]])
        y_all = np.concatenate([y_u, y_l[1:]])
        x_str = " ".join(f"{v:.6f}" for v in x_all)
        y_str = " ".join(f"{v:.6f}" for v in y_all)
        mock_data = pd.DataFrame(
            {
                "name": ["NACA0012"] * 6,
                "x_coords": [x_str] * 6,
                "y_coords": [y_str] * 6,
                "angle": [0, 2, 5, 8, 10, 12],
                "reynolds": [200000] * 6,
                "cl": [0.0, 0.2, 0.5, 0.8, 0.9, 0.7],
                "cd": [0.008, 0.009, 0.011, 0.025, 0.030, 0.060],
                "cm": [-0.01] * 6,
            }
        )
        mock_data.to_csv(csv_input_path, index=False)

    t0 = time.time()
    pipeline = AirfoilPreprocessingPipeline(n_points=120, contamination=0.03)
    pipeline.process_geometry_database(csv_input_path, chunksize=20_000)
    pipeline.assemble_final_dataset(csv_input_path, chunksize=20_000)
    print(f"Day2 total time: {time.time() - t0:.1f}s")
