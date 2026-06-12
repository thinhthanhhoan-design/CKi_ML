"""day4_xgb_asymmetric_cl_objective.py
Snippet custom objective cho XGBoost, sinh từ Day 3.
Over-prediction Cl được phạt nặng hơn under-prediction.
"""

import numpy as np

OVER_WEIGHT = 10.0


def asymmetric_cl_objective(y_true, y_pred):
    e = y_pred - y_true
    w = np.where(e > 0.0, OVER_WEIGHT, 1.0)
    grad = 2.0 * w * e
    hess = 2.0 * w
    return grad, hess
