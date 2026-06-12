# guided_search.py
"""Utility module for guided (Bayesian) search in day5 optimizer.

Implements a simple Optuna wrapper that samples CST delta weights within
the trust‑region bounds defined in the Config object. The objective function
expects a *delta_weights* array and returns a **score** (higher is better).
We negate the score for Optuna because it minimizes by default.

The function returns the best delta_weights and the corresponding (non‑negated)
score.
"""
import logging
from typing import Tuple
import numpy as np
import optuna

logger = logging.getLogger(__name__)


def bayesian_search(cfg, base_wu, base_wl, objective_func) -> Tuple[np.ndarray, float]:
    """Run a Bayesian optimization loop using Optuna.

    Parameters
    ----------
    cfg: Config
        Configuration object containing ``random_seed`` and ``n_trials``.
    base_wu, base_wl: np.ndarray
        Baseline upper‑ and lower‑surface CST weights (unused by the search
        but kept for signature compatibility with previous calls).
    objective_func: Callable[[np.ndarray], float]
        Function that receives ``delta_weights`` (the CST perturbation) and
        returns a **score** (higher is better). ``day5``'s ``objective`` method
        already packs all aerodynamic and penalty terms and returns ``-final_score``.
        Therefore we simply call ``-objective_func`` inside Optuna.

    Returns
    -------
    Tuple[np.ndarray, float]
        ``best_weights`` – the delta_weights that achieved the best score.
        ``best_score`` – the corresponding (non‑negated) score.
    """
    # Ensure reproducibility
    np.random.seed(cfg.random_seed)

    n = cfg.cst_order + 1
    dim = 2 * n  # upper and lower surface weights

    def optuna_obj(trial):
        # Sample each delta within the trust‑region bounds
        delta = np.empty(dim)
        for i in range(dim):
            delta[i] = trial.suggest_uniform(
                f"dw_{i}", -cfg.cst_delta_max, cfg.cst_delta_max
            )
        # ``objective_func`` returns *negative* final_score (as in original code).
        # Optuna minimizes, so we return the raw value.
        return objective_func(delta)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=cfg.random_seed),
    )
    study.optimize(optuna_obj, n_trials=cfg.n_trials)

    best_weights = np.array([study.best_params[f"dw_{i}"] for i in range(dim)])
    best_score = -study.best_value  # invert because original returns -score
    logger.info(
        "Bayesian search completed – best score %.4f after %d trials",
        best_score,
        cfg.n_trials,
    )
    return best_weights, best_score
