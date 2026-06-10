import sys
import os
import json
import time
import argparse
import logging
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Import helper functions from day4_cl_2
sys.path.append("e:/Project2")
import day4_cl_2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HGB_Hyperparameter_Tuning")

def main():
    parser = argparse.ArgumentParser(description="Tune HGB hyperparameters for day4_cl_2 to maximize R2.")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--max_rows", type=int, default=150000, help="Number of rows to sample for fast search. Set to 0 to use all rows.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for HGB training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Try import optuna, install if missing
    try:
        import optuna
    except ImportError:
        logger.info("Installing optuna for hyperparameter tuning...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        import optuna

    cfg = day4_cl_2.CLReliabilityConfig()
    dataset_path = cfg.dataset_path
    if not os.path.exists(dataset_path):
        # Check if uncompressed csv exists
        csv_path = dataset_path.replace(".gz", "")
        if os.path.exists(csv_path):
            dataset_path = csv_path
        else:
            logger.error(f"Dataset not found: {dataset_path}")
            sys.exit(1)

    logger.info(f"Loading dataset from {dataset_path}...")
    read_nrows = max(100_000, args.max_rows * 3) if args.max_rows else None
    raw_df = pd.read_csv(dataset_path, low_memory=False, nrows=read_nrows)
    raw_df = day4_cl_2.downcast_float64_to_float32(raw_df)
    
    day5_features, excluded = day4_cl_2.apply_comprehensive_whitelist_v8(raw_df)
    logger.info(f"Features: {len(day5_features)} selected, {len(excluded)} excluded.")
    
    clean_df = raw_df.dropna(subset=["cl", "geom_hash", "angle"]).copy().reset_index(drop=True)
    clean_df = day4_cl_2.downcast_float64_to_float32(clean_df)
    
    if args.max_rows and len(clean_df) > args.max_rows:
        logger.info(f"Sampling {args.max_rows} rows for fast search...")
        clean_df = clean_df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
        clean_df = day4_cl_2.downcast_float64_to_float32(clean_df)
    
    X = clean_df[day5_features].to_numpy(dtype=np.float32)
    y = clean_df["cl"].to_numpy(dtype=np.float32)
    groups = clean_df["geom_hash"].astype(str).to_numpy()
    sample_weights = day4_cl_2.build_cl_sample_weights(clean_df, y, cfg)
    
    logger.info("Imputing missing values once to accelerate tuning...")
    X = SimpleImputer(strategy="median").fit_transform(X)
    
    del raw_df
    gc.collect()

    def objective(trial):
        # Define hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        max_iter = trial.suggest_int("max_iter", 200, 800, step=50)
        max_depth = trial.suggest_int("max_depth", 6, 16)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 31, 255, step=16)
        l2_reg = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 100, step=10)
        
        gkf = GroupKFold(n_splits=args.n_splits)
        oof_preds = np.full(len(y), np.nan, dtype=np.float32)
        
        logger.info(f"Starting Trial {trial.number}: lr={learning_rate:.4f}, iter={max_iter}, depth={max_depth}, leaves={max_leaf_nodes}, l2={l2_reg:.4f}, min_samples={min_samples_leaf}")
        
        fold_times = []
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            t0 = time.time()
            model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                l2_regularization=l2_reg,
                min_samples_leaf=min_samples_leaf,
                random_state=args.seed,
            )
            model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
            oof_preds[va_idx] = model.predict(X[va_idx])
            fold_times.append(time.time() - t0)
            
        r2 = r2_score(y, oof_preds)
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        logger.info(f"Trial {trial.number} finished in {sum(fold_times):.1f}s | OOF R2 = {r2:.6f} | OOF RMSE = {rmse:.6f}")
        
        # We want to maximize R2
        return r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING COMPLETED")
    logger.info(f"Best OOF R2: {study.best_value:.6f}")
    logger.info("Best Parameters:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 80)
    
    # Save the best parameters to a JSON file
    out_path = "best_hgb_cl_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Saved best parameters to {out_path}")

if __name__ == "__main__":
    main()
