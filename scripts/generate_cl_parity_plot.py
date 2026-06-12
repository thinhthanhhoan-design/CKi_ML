import os
import json
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from day4_cl_2 import (
    CLReliabilityConfig,
    apply_comprehensive_whitelist_v8,
    downcast_float64_to_float32,
    section_base_cl_model,
    ensure_dirs
)

def main():
    print("Initializing...")
    cfg = CLReliabilityConfig()
    cfg.dataset_path = "deeplearwing_day2_tabular.csv.gz"
    cfg.output_dir = "outputs/day4_cl_2_hgb_improved"
    paths = ensure_dirs(cfg.output_dir)

    oof_csv = Path(cfg.output_dir) / "tables" / "cl_oof_predictions.csv"
    
    if oof_csv.exists():
        print(f"Loading existing OOF predictions from {oof_csv}...")
        df_oof = pd.read_csv(oof_csv)
        y_true = df_oof["y_true"].to_numpy()
        oof_preds = df_oof["oof_preds"].to_numpy()
    else:
        print("Re-computing OOF predictions (this will take 1-2 minutes)...")
        if not os.path.exists(cfg.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")
        
        # Load dataset
        raw_df = pd.read_csv(cfg.dataset_path, low_memory=False)
        raw_df = downcast_float64_to_float32(raw_df)
        day5_features, excluded = apply_comprehensive_whitelist_v8(raw_df)
        
        clean_df = raw_df.dropna(subset=["cl", "geom_hash", "angle"]).copy().reset_index(drop=True)
        clean_df = downcast_float64_to_float32(clean_df)
        
        cfg.max_rows = 50000
        if len(clean_df) > cfg.max_rows:
            clean_df = clean_df.sample(cfg.max_rows, random_state=cfg.random_state).reset_index(drop=True)
            clean_df = downcast_float64_to_float32(clean_df)
            
        del raw_df
        gc.collect()
        
        # Run CV
        base_ctx, oof_preds, y_true = section_base_cl_model(clean_df, day5_features, cfg, paths)
        
        # Save to CSV for future use
        df_oof = pd.DataFrame({"y_true": y_true, "oof_preds": oof_preds})
        df_oof.to_csv(oof_csv, index=False)
        print(f"OOF predictions saved to {oof_csv}")

    print("Generating Parity Plot...")
    
    # Configure styling
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Randomly sample points to avoid over-saturation and match the user's plot density
    # 665,215 points is too dense for a scatter plot; 15,000 points matches the target plot perfectly
    np.random.seed(42)
    sample_size = min(15000, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    y_true_sample = y_true[indices]
    oof_preds_sample = oof_preds[indices]
    
    # Scatter plot of predictions
    # Teal color #409090 with alpha=0.5 matches the user's plot
    ax.scatter(y_true_sample, oof_preds_sample, color="#409090", alpha=0.5, s=20, edgecolors="none", zorder=3)
    
    # Diagonal line y=x in dark navy color #17324D
    ax.plot([-0.6, 2.1], [-0.6, 2.1], color="#17324D", linestyle="--", linewidth=2.0, zorder=2)
    
    # Set titles and labels
    ax.set_title("CL OOF Parity", fontsize=14, fontweight="normal", color="#333333", pad=12)
    ax.set_xlabel("True CL", fontsize=12, labelpad=8, color="#333333")
    ax.set_ylabel("Predicted CL", fontsize=12, labelpad=8, color="#333333")
    
    # Axis ticks and limits exactly from -0.5 to 2.0
    ax.set_xlim(-0.6, 2.1)
    ax.set_ylim(-0.6, 2.1)
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Configure borders (spines) to keep the box border around the plot
    for spine_name, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(1.5)
        
    # Configure tick params
    ax.tick_params(axis="both", which="major", labelsize=11, labelcolor="#333333")
    
    # Thin light-grey grid lines
    ax.grid(True, linestyle="-", color="#E0E0E0", linewidth=1.0)
    
    # Save the plot
    output_path = Path(cfg.output_dir) / "figures" / "cl_prediction_vs_truth_seaborn.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Beautiful parity plot saved to: {output_path}")

    # Also overwrite the default output file
    default_output_path = Path(cfg.output_dir) / "figures" / "cl_prediction_vs_truth.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_sample, oof_preds_sample, color="#409090", alpha=0.5, s=20, edgecolors="none", zorder=3)
    ax.plot([-0.6, 2.1], [-0.6, 2.1], color="#17324D", linestyle="--", linewidth=2.0, zorder=2)
    ax.set_title("CL OOF Parity", fontsize=14, fontweight="normal", color="#333333", pad=12)
    ax.set_xlabel("True CL", fontsize=12, labelpad=8, color="#333333")
    ax.set_ylabel("Predicted CL", fontsize=12, labelpad=8, color="#333333")
    ax.set_xlim(-0.6, 2.1)
    ax.set_ylim(-0.6, 2.1)
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    for spine_name, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=11, labelcolor="#333333")
    ax.grid(True, linestyle="-", color="#E0E0E0", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(default_output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Default prediction vs truth plot overwritten at: {default_output_path}")

if __name__ == "__main__":
    main()
