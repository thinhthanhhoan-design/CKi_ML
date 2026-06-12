# generate_neuralfoil_dataset.py
import argparse
import errno
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import neuralfoil as nf


def make_aoa_grid(
    max_abs_aoa=22.0,
    step_0_5=0.25,
    step_5_8=0.25,
    step_8_12=0.20,
    step_12_16=0.10,
    step_gt_16=0.25,
):
    """
    Sinh AoA đối xứng âm/dương, mật độ cao ở vùng near-stall/stall.
    Không để sót biên: 0, 5, 8, 12, 16, max_abs_aoa đều có mặt.
    """

    segments = [
        (0.0, 5.0, step_0_5),
        (5.0, 8.0, step_5_8),
        (8.0, 12.0, step_8_12),
        (12.0, 16.0, step_12_16),
        (16.0, max_abs_aoa, step_gt_16),
    ]

    positive = []

    for i, (a, b, step) in enumerate(segments):
        vals = np.arange(a, b + step * 0.5, step)
        vals = vals[(vals >= a - 1e-9) & (vals <= b + 1e-9)]

        if i > 0:
            vals = vals[vals > a + 1e-9]

        positive.extend(vals.tolist())

    positive = np.array(sorted(set(np.round(positive, 6))))

    aoa = np.concatenate((-positive[:0:-1], positive))
    aoa = np.round(aoa, 6)

    return aoa


def aoa_regime(alpha):
    aa = abs(float(alpha))
    if aa <= 5:
        return "attached_0_5"
    if aa <= 8:
        return "pre_stall_5_8"
    if aa <= 12:
        return "near_stall_8_12"
    if aa <= 16:
        return "stall_12_16"
    return "post_stall_gt_16"


def read_dat_file(path: Path):
    xs, ys = [], []

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()

        if not line:
            continue
        if line.startswith("#"):
            continue

        parts = line.replace(",", " ").split()

        if len(parts) < 2:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            xs.append(x)
            ys.append(y)
        except ValueError:
            continue

    coords = np.column_stack([xs, ys]).astype(float)

    if len(coords) < 20:
        raise ValueError("Too few coordinate points")

    if not np.all(np.isfinite(coords)):
        raise ValueError("NaN/Inf in coordinates")

    return coords


def normalize_coords(coords):
    """
    Chuẩn hóa chord về [0, 1].
    Giữ thứ tự kiểu Selig nếu file đã đúng:
    upper TE -> LE -> lower TE.
    """

    coords = np.asarray(coords, dtype=float)

    x = coords[:, 0]
    y = coords[:, 1]

    x_min = np.min(x)
    x_max = np.max(x)
    chord = x_max - x_min

    if chord <= 1e-8:
        raise ValueError("Invalid chord length")

    coords = coords.copy()
    coords[:, 0] = (coords[:, 0] - x_min) / chord
    coords[:, 1] = coords[:, 1] / chord

    return coords


def geom_hash(coords, decimals=6):
    arr = np.round(coords, decimals=decimals)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def run_neuralfoil_one_airfoil(
    dat_path,
    reynolds_list,
    aoa_grid,
    model_size="xlarge",
    n_crit=9.0,
    xtr_upper=1.0,
    xtr_lower=1.0,
    include_coords_in_rows=False,
):
    raw_coords = read_dat_file(dat_path)
    coords = normalize_coords(raw_coords)

    gh = geom_hash(coords)

    rows = []

    for re in reynolds_list:
        alpha = np.asarray(aoa_grid, dtype=float)
        Re = np.full_like(alpha, float(re), dtype=float)

        out = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alpha,
            Re=Re,
            model_size=model_size,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
        )

        for i, a in enumerate(alpha):
            row = {
                "name": dat_path.stem,
                "source_file": dat_path.name,
                "geom_hash": gh,
                "angle": float(a),
                "abs_angle": abs(float(a)),
                "aoa_regime": aoa_regime(a),
                "reynolds": float(re),
                "log10_reynolds": math.log10(float(re)),
                "x_coords": " ".join(f"{v:.8f}" for v in coords[:, 0]),
                "y_coords": " ".join(f"{v:.8f}" for v in coords[:, 1]),
                "cl": float(np.asarray(out["CL"])[i]),
                "cd": float(np.asarray(out["CD"])[i]),
                "cm": float(np.asarray(out["CM"])[i]),
                "analysis_confidence": float(np.asarray(out.get("analysis_confidence", np.ones_like(alpha)))[i]),
                "model": "NeuralFoil",
                "model_size": model_size,
            }
            if include_coords_in_rows:
                row["x_coords"] = " ".join(f"{v:.8f}" for v in coords[:, 0])
                row["y_coords"] = " ".join(f"{v:.8f}" for v in coords[:, 1])
            rows.append(row)

    return rows, coords, gh


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dat_dir", type=str, default="uiuc_airfoils_dat")
    parser.add_argument("--output_csv", type=str, default="neuralfoil_deeplearwing_augmented.csv.gz")
    parser.add_argument("--error_log", type=str, default="neuralfoil_failed_files.csv")
    parser.add_argument("--config_out", type=str, default="neuralfoil_generation_config.json")
    parser.add_argument("--geometry_csv", type=str, default="neuralfoil_geometries.csv.gz")
    parser.add_argument("--coords_in_rows", action="store_true",
                        help="Store x_coords/y_coords in every row (very large output).")

    parser.add_argument("--reynolds", type=float, nargs="+",
                        default=[50000, 100000, 200000, 500000, 800000])

    parser.add_argument("--max_abs_aoa", type=float, default=22.0)

    parser.add_argument("--step_0_5", type=float, default=0.25)
    parser.add_argument("--step_5_8", type=float, default=0.25)
    parser.add_argument("--step_8_12", type=float, default=0.20)
    parser.add_argument("--step_12_16", type=float, default=0.10)
    parser.add_argument("--step_gt_16", type=float, default=0.25)

    parser.add_argument("--model_size", type=str, default="xlarge")
    parser.add_argument("--n_crit", type=float, default=9.0)
    parser.add_argument("--xtr_upper", type=float, default=1.0)
    parser.add_argument("--xtr_lower", type=float, default=1.0)

    parser.add_argument("--flush_every", type=int, default=20)

    args = parser.parse_args()

    dat_dir = Path(args.dat_dir)
    output_csv = Path(args.output_csv)
    error_log = Path(args.error_log)
    geometry_csv = Path(args.geometry_csv)

    dat_files = sorted(dat_dir.glob("*.dat"))

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {dat_dir}")

    aoa_grid = make_aoa_grid(
        max_abs_aoa=args.max_abs_aoa,
        step_0_5=args.step_0_5,
        step_5_8=args.step_5_8,
        step_8_12=args.step_8_12,
        step_12_16=args.step_12_16,
        step_gt_16=args.step_gt_16,
    )

    cfg = vars(args).copy()
    cfg["n_dat_files"] = len(dat_files)
    cfg["n_aoa_per_re"] = len(aoa_grid)
    cfg["aoa_min"] = float(np.min(aoa_grid))
    cfg["aoa_max"] = float(np.max(aoa_grid))
    cfg["expected_rows"] = int(len(dat_files) * len(args.reynolds) * len(aoa_grid))
    cfg["aoa_grid"] = aoa_grid.tolist()

    write_json(Path(args.config_out), cfg)

    print(f"DAT files: {len(dat_files)}")
    print(f"AoA count per Re: {len(aoa_grid)}")
    print(f"Re count: {len(args.reynolds)}")
    print(f"Expected rows: {cfg['expected_rows']:,}")
    print(f"AoA range: {aoa_grid.min()} → {aoa_grid.max()}")

    buffer = []
    geom_buffer = []
    errors = []
    wrote_header = output_csv.exists() and output_csv.stat().st_size > 0
    wrote_geom_header = geometry_csv.exists() and geometry_csv.stat().st_size > 0
    seen_geom_hashes = set()

    def csv_compression_for(path: Path):
        return "gzip" if path.suffix.lower() == ".gz" else None

    data_compression = csv_compression_for(output_csv)
    geom_compression = csv_compression_for(geometry_csv)

    def flush_buffers():
        nonlocal buffer, geom_buffer, wrote_header, wrote_geom_header
        if buffer:
            df = pd.DataFrame(buffer)
            df.to_csv(
                output_csv,
                mode="a",
                index=False,
                header=not wrote_header,
                compression=data_compression,
            )
            wrote_header = True
            buffer = []
        if geom_buffer:
            gdf = pd.DataFrame(geom_buffer)
            gdf.to_csv(
                geometry_csv,
                mode="a",
                index=False,
                header=not wrote_geom_header,
                compression=geom_compression,
            )
            wrote_geom_header = True
            geom_buffer = []

    for idx, dat_path in enumerate(tqdm(dat_files), start=1):
        try:
            rows, coords, gh = run_neuralfoil_one_airfoil(
                dat_path=dat_path,
                reynolds_list=args.reynolds,
                aoa_grid=aoa_grid,
                model_size=args.model_size,
                n_crit=args.n_crit,
                xtr_upper=args.xtr_upper,
                xtr_lower=args.xtr_lower,
                include_coords_in_rows=args.coords_in_rows,
            )
            buffer.extend(rows)
            if not args.coords_in_rows and gh not in seen_geom_hashes:
                seen_geom_hashes.add(gh)
                geom_buffer.append({
                    "name": dat_path.stem,
                    "source_file": dat_path.name,
                    "geom_hash": gh,
                    "x_coords": " ".join(f"{v:.8f}" for v in coords[:, 0]),
                    "y_coords": " ".join(f"{v:.8f}" for v in coords[:, 1]),
                })

        except Exception as e:
            errors.append({
                "file": dat_path.name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

        if (len(buffer) > 0 or len(geom_buffer) > 0) and idx % args.flush_every == 0:
            try:
                flush_buffers()
            except OSError as e:
                if getattr(e, "errno", None) == errno.ENOSPC:
                    print("\nDisk full while writing output. Partial results were kept.")
                    break
                raise

    try:
        flush_buffers()
    except OSError as e:
        if getattr(e, "errno", None) == errno.ENOSPC:
            print("\nDisk full during final flush. Partial results were kept.")
        else:
            raise

    if errors:
        pd.DataFrame(errors).to_csv(error_log, index=False)

    print("\nDONE")
    print(f"Output CSV: {output_csv.resolve()}")
    if not args.coords_in_rows:
        print(f"Geometry CSV: {geometry_csv.resolve()}")
    print(f"Failed files: {len(errors)}")
    if errors:
        print(f"Error log: {error_log.resolve()}")


if __name__ == "__main__":
    main()
