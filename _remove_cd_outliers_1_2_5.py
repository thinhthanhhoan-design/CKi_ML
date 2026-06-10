from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:\Project2").resolve()
REMOVE = {
    "10ec05e3a6d9fdbaa34ae324b42fce7686fc92cb": "cap21c",
    "8177ea687834a2cb8f589b190dd23a74df4e740d": "fx79w470a",
    "ef509a59974d9e5a916f5544c4e733f8b56dd6bf": "ah93w480b",
}
SRC_CSV = ROOT / "deeplearwing_day2_tabular.csv"
SRC_GZ = ROOT / "deeplearwing_day2_tabular.csv.gz"
TMP_CSV = ROOT / "deeplearwing_day2_tabular.filtered_tmp.csv"
TMP_GZ = ROOT / "deeplearwing_day2_tabular.filtered_tmp.csv.gz"
BAK_CSV = ROOT / "deeplearwing_day2_tabular.before_remove_1_2_5.bak.csv"
BAK_GZ = ROOT / "deeplearwing_day2_tabular.before_remove_1_2_5.bak.csv.gz"
REPORT_DIR = ROOT / "outputs" / "data_cleaning"
REPORT = REPORT_DIR / "remove_cd_outliers_1_2_5_report.json"
CHUNKSIZE = 50_000

def assert_inside(path: Path) -> None:
    resolved = path.resolve()
    if ROOT not in [resolved, *resolved.parents]:
        raise RuntimeError(f"Unsafe path outside project root: {resolved}")

def cleanup_temp() -> None:
    for p in (TMP_CSV, TMP_GZ):
        if p.exists():
            p.unlink()

def main() -> int:
    t0 = time.time()
    for p in (SRC_CSV, SRC_GZ, TMP_CSV, TMP_GZ, BAK_CSV, BAK_GZ, REPORT):
        assert_inside(p)
    if not SRC_CSV.exists() or not SRC_GZ.exists():
        raise FileNotFoundError("Missing source CSV or CSV.GZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_temp()

    total_in = 0
    total_out = 0
    removed_counts = {h: 0 for h in REMOVE}
    removed_names = {h: set() for h in REMOVE}
    first = True

    print("START_REMOVE_CD_OUTLIERS_1_2_5", flush=True)
    print("remove_hashes=" + json.dumps(REMOVE, indent=2), flush=True)

    with open(TMP_CSV, "w", encoding="utf-8", newline="") as f_csv, gzip.open(TMP_GZ, "wt", encoding="utf-8", newline="") as f_gz:
        for i, chunk in enumerate(pd.read_csv(SRC_CSV, chunksize=CHUNKSIZE, low_memory=False), 1):
            if "geom_hash" not in chunk.columns:
                raise RuntimeError("Missing geom_hash column")
            total_in += len(chunk)
            geom = chunk["geom_hash"].astype(str)
            drop_mask = geom.isin(REMOVE)
            if bool(drop_mask.any()):
                dropped = chunk.loc[drop_mask]
                for h, g in dropped.groupby(dropped["geom_hash"].astype(str), sort=False):
                    removed_counts[h] = removed_counts.get(h, 0) + int(len(g))
                    if "name" in g.columns:
                        removed_names.setdefault(h, set()).update(g["name"].dropna().astype(str).unique().tolist())
            kept = chunk.loc[~drop_mask]
            total_out += len(kept)
            kept.to_csv(f_csv, index=False, header=first)
            kept.to_csv(f_gz, index=False, header=first)
            first = False
            if i % 2 == 0:
                print(f"progress chunks={i} total_in={total_in} total_out={total_out} removed={sum(removed_counts.values())}", flush=True)

    expected_out = total_in - sum(removed_counts.values())
    if total_out != expected_out:
        raise RuntimeError(f"Row count mismatch: total_out={total_out}, expected={expected_out}")
    if any(removed_counts.get(h, 0) <= 0 for h in REMOVE):
        raise RuntimeError(f"At least one requested hash was not removed: {removed_counts}")

    report = {
        "removed_hashes": REMOVE,
        "removed_counts": removed_counts,
        "removed_names": {h: sorted(v) for h, v in removed_names.items()},
        "total_rows_before": total_in,
        "total_rows_after": total_out,
        "total_rows_removed": sum(removed_counts.values()),
        "source_csv": str(SRC_CSV),
        "source_gz": str(SRC_GZ),
        "backup_csv": str(BAK_CSV),
        "backup_gz": str(BAK_GZ),
        "elapsed_sec_write": round(time.time() - t0, 3),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Replace as a pair with backups kept for manual rollback.
    for bak in (BAK_CSV, BAK_GZ):
        if bak.exists():
            bak.unlink()
    os.replace(SRC_CSV, BAK_CSV)
    os.replace(SRC_GZ, BAK_GZ)
    os.replace(TMP_CSV, SRC_CSV)
    os.replace(TMP_GZ, SRC_GZ)

    report["elapsed_sec_total"] = round(time.time() - t0, 3)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE_REMOVE_CD_OUTLIERS_1_2_5", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED_REMOVE_CD_OUTLIERS: {exc!r}", file=sys.stderr, flush=True)
        raise
