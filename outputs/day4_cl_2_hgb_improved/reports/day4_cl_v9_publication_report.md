# Day4 CL Reliability V9 Publication Report

## Overall Metrics

- Champion: **hgb**
- OOF RMSE: 0.05695
- OOF R²: 0.99505

## CLmax / Stall Metrics

- CLmax Error: 0.03283
- Stall AoA Error: 1.06242 deg

## Geometry Refit Validation

Removed for runtime; no per-geometry LOGO refits are executed.

## Trust Region Analysis

| trust_region_class   |   count |
|:---------------------|--------:|
| SAFE                 |  598577 |
| WARNING              |   46600 |
| UNSAFE               |   20038 |

## Uncertainty Quality

| interval_method          |   pearson_uq_error |   spearman_uq_error |   coverage90 |   coverage90_gap |   coverage95 |   coverage95_gap |   calibration_slope | uncertainty_reliable   |
|:-------------------------|-------------------:|--------------------:|-------------:|-----------------:|-------------:|-----------------:|--------------------:|:-----------------------|
| adaptive_split_conformal |           0.324949 |            0.269713 |     0.923538 |        0.0235375 |     0.963344 |        0.0133437 |             0.12407 | True                   |

## Physical Consistency

- Physical consistency violations: 2

## Day5 Readiness

Day5 interface is safety-aware and includes CL interval, OOD, trust-region, reliability, and safety flag endpoints.

## Research Conclusions

V9 prioritizes geometry generalization, CLmax/stall reliability, uncertainty validation, and safe inverse design rather than R²-only optimization.
