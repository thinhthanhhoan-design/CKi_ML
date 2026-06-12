from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "navy": "#17324D",
    "blue": "#2676B8",
    "cyan": "#4DB6AC",
    "green": "#2E8B57",
    "amber": "#E69F00",
    "red": "#C33C54",
    "gray": "#66737F",
    "light": "#EEF3F6",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": COLORS["navy"],
            "axes.labelcolor": COLORS["navy"],
            "xtick.color": COLORS["navy"],
            "ytick.color": COLORS["navy"],
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "grid.color": "#D7E0E6",
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_geometry_names(dataset: Path, hashes: set[str]) -> dict[str, str]:
    if not dataset.exists() or not hashes:
        return {}

    names: dict[str, str] = {}
    requested = {"geom_hash", "name", "airfoil_name"}
    for chunk in pd.read_csv(
        dataset,
        usecols=lambda column: column in requested,
        chunksize=100_000,
        low_memory=False,
    ):
        if "geom_hash" not in chunk.columns:
            break
        matched = chunk[chunk["geom_hash"].astype(str).isin(hashes)]
        for row in matched.drop_duplicates("geom_hash").itertuples(index=False):
            row_data = row._asdict()
            name = row_data.get("name") or row_data.get("airfoil_name")
            if pd.notna(name):
                names[str(row_data["geom_hash"])] = str(name)
        if hashes.issubset(names):
            break
    return names


def plot_model_summary(model_df: pd.DataFrame, output: Path) -> None:
    row = model_df.iloc[0]
    metrics = [
        ("OOF R2", float(row["r2"]), "higher is better", COLORS["green"]),
        ("OOF RMSE", float(row["rmse"]), "lower is better", COLORS["blue"]),
        ("CLmax MAE", float(row["clmax_mae"]), "lower is better", COLORS["cyan"]),
        ("Stall AoA MAE", float(row["stall_aoa_mae"]), "degrees", COLORS["amber"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    fig.suptitle("HGB-only CL Validation Summary", fontsize=17, color=COLORS["navy"])
    for ax, (title, value, note, color) in zip(axes, metrics):
        ax.set_facecolor(COLORS["light"])
        ax.text(0.5, 0.68, title, ha="center", va="center", fontsize=12, color=COLORS["navy"])
        fmt = f"{value:.4f}" if title != "Stall AoA MAE" else f"{value:.2f} deg"
        ax.text(0.5, 0.43, fmt, ha="center", va="center", fontsize=22, weight="bold", color=color)
        ax.text(0.5, 0.19, note, ha="center", va="center", fontsize=9, color=COLORS["gray"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color(color)
    save(fig, output / "cl_hgb_validation_summary.png")


def plot_conformal(conformal: pd.DataFrame, output: Path) -> None:
    x = conformal["target_coverage_alpha"].to_numpy(float) * 100.0
    observed = conformal["PICP"].to_numpy(float) * 100.0
    half_width = conformal["MPIW"].to_numpy(float) / 2.0

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    width = 2.8
    ax.bar(x - width / 2, x, width=width, label="Target coverage", color="#B9C8D3")
    ax.bar(x + width / 2, observed, width=width, label="Observed coverage", color=COLORS["blue"])
    ax.set_xlabel("Conformal confidence level (%)")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(70, 100)
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(x, half_width, marker="o", linewidth=2.2, color=COLORS["red"], label="CL half-width")
    ax2.set_ylabel("Prediction interval half-width (CL)", color=COLORS["red"])
    ax2.tick_params(axis="y", colors=COLORS["red"])
    ax2.legend(loc="lower right")
    ax.set_title("Conformal Coverage and Interval Width")
    save(fig, output / "conformal_coverage.png")


def plot_ood_distribution(ood: pd.DataFrame, output: Path) -> None:
    scores = np.clip(ood["geometry_ood_score"].to_numpy(float), 1e-8, None)
    flagged = ood["geometry_ood_flag"].astype(bool).to_numpy()
    log_scores = np.log10(scores)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.hist(log_scores[~flagged], bins=35, alpha=0.78, color=COLORS["blue"], label="In-distribution")
    ax.hist(log_scores[flagged], bins=25, alpha=0.82, color=COLORS["red"], label="OOD flagged")
    if np.any(flagged):
        threshold = float(np.min(scores[flagged]))
        ax.axvline(np.log10(threshold), color=COLORS["amber"], linestyle="--", linewidth=2,
                   label=f"Observed flag boundary ~ {threshold:.3g}")
    ax.set_xlabel("log10(geometry OOD score)")
    ax.set_ylabel("Geometry count")
    ax.set_title("Geometry OOD Score Distribution")
    ax.grid(axis="y")
    ax.legend()
    save(fig, output / "geometry_ood_distribution.png")


def plot_trust_summary(ood: pd.DataFrame, output: Path) -> None:
    order = ["SAFE", "WARNING", "UNSAFE"]
    counts = ood["geometry_trust_class"].value_counts().reindex(order, fill_value=0)
    total = max(1, int(counts.sum()))
    colors = [COLORS["green"], COLORS["amber"], COLORS["red"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(order, counts.values, color=colors, width=0.62)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.015,
            f"{count} ({100.0 * count / total:.1f}%)",
            ha="center",
            va="bottom",
            weight="bold",
            color=COLORS["navy"],
        )
    ax.set_ylabel("Geometry count")
    ax.set_title("Geometry Trust Classification")
    ax.set_ylim(0, max(counts.max() * 1.17, 1))
    ax.grid(axis="y")
    save(fig, output / "geometry_trust_summary.png")


def plot_top_ood(ood: pd.DataFrame, dataset: Path, output: Path) -> None:
    top = ood.nlargest(15, "geometry_ood_score").copy()
    hashes = set(top["geom_hash"].astype(str))
    names = load_geometry_names(dataset, hashes)
    top["label"] = [
        names.get(str(geom_hash), str(geom_hash)[:10])
        for geom_hash in top["geom_hash"]
    ]
    top = top.sort_values("geometry_ood_score")

    color_map = {"SAFE": COLORS["green"], "WARNING": COLORS["amber"], "UNSAFE": COLORS["red"]}
    colors = [color_map.get(str(value), COLORS["gray"]) for value in top["geometry_trust_class"]]
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.barh(top["label"], top["geometry_ood_score"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Geometry OOD score (log scale)")
    ax.set_ylabel("Airfoil")
    ax.set_title("Top 15 Geometry OOD Cases")
    ax.grid(axis="x")
    save(fig, output / "top_geometry_ood_cases.png")


def write_report(root: Path, model_df: pd.DataFrame, conformal: pd.DataFrame, ood: pd.DataFrame) -> None:
    row = model_df.iloc[0]
    counts = ood["geometry_trust_class"].value_counts()
    report = f"""# CL HGB Visual Diagnostics

Generated from the partial Day4 CL output.

## Model quality

- OOF R2: {float(row['r2']):.6f}
- OOF RMSE: {float(row['rmse']):.6f}
- CLmax MAE: {float(row['clmax_mae']):.6f}
- Stall AoA MAE: {float(row['stall_aoa_mae']):.3f} deg

## Geometry trust

- SAFE: {int(counts.get('SAFE', 0))}
- WARNING: {int(counts.get('WARNING', 0))}
- UNSAFE: {int(counts.get('UNSAFE', 0))}
- OOD flagged: {int(ood['geometry_ood_flag'].astype(bool).sum())}

## Important limitation

The output does not contain `cl_model.joblib` or row-level OOF predictions. Therefore,
an exact CL true-vs-predicted scatter plot cannot be reconstructed from this package.
The existing `clmax_model.joblib` and `stall_model.joblib` are different targets and
must not be used as substitutes for the primary CL model.
"""
    (root / "reports" / "visual_diagnostics_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset", default=r"E:\Project2\deeplearwing_day2_tabular.csv")
    args = parser.parse_args()

    root = Path(args.output_root)
    tables = root / "tables"
    figures = root / "figures"
    reports = root / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(tables / "model_comparison.csv")
    conformal = pd.read_csv(tables / "conformal_coverage_report.csv")
    ood = pd.read_csv(tables / "geometry_ood_scores.csv")

    configure_style()
    plot_model_summary(model_df, figures)
    plot_conformal(conformal, figures)
    plot_ood_distribution(ood, figures)
    plot_trust_summary(ood, figures)
    plot_top_ood(ood, Path(args.dataset), figures)
    write_report(root, model_df, conformal, ood)

    print(f"Generated figures in: {figures}")
    for path in sorted(figures.glob("*.png")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
