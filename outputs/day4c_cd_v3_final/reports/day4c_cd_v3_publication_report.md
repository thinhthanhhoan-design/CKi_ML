# Day4C_CD_V3_1_Fixed Publication Report

## Overall metrics
| scope   |      n |      mae |     rmse |       r2 |    bias |
|:--------|-------:|---------:|---------:|---------:|--------:|
| overall | 132810 | 0.000505 | 0.001064 | 0.999825 | 4.7e-05 |

## Regime metrics
| scope           |     n |      mae |     rmse |       r2 |      bias |
|:----------------|------:|---------:|---------:|---------:|----------:|
| R1_linear       | 22230 | 0.000806 | 0.001632 | 0.988816 | -2.8e-05  |
| R2_transitional | 13680 | 0.000666 | 0.001415 | 0.99738  | -0.000112 |
| R3_pre_stall    | 11400 | 0.000381 | 0.000905 | 0.999454 |  1.5e-05  |
| R4_stall_onset  | 11400 | 0.000346 | 0.000716 | 0.999778 |  2.1e-05  |
| R5_near_stall   | 45600 | 0.000373 | 0.000672 | 0.999865 |  2.8e-05  |
| R6_post_stall   | 28500 | 0.000518 | 0.000991 | 0.999654 |  0.000235 |

## Geometry refit validation
Removed for runtime; no per-geometry refits are executed.

## Trust-region analysis
|   geometry_threshold |   flow_threshold |   mean_trust_region_score |   safe_rate |   warning_rate |   unsafe_rate |
|---------------------:|-----------------:|--------------------------:|------------:|---------------:|--------------:|
|              6.63779 |           2.8668 |                  0.602592 |    0.835539 |       0.038077 |      0.126384 |

## Physics consistency analysis
|   n_curves |   true_violating_curves |   pred_violating_curves |   true_violation_rate |   pred_violation_rate |   negative_cd_total |   post_stall_collapse_total |
|-----------:|------------------------:|------------------------:|----------------------:|----------------------:|--------------------:|----------------------------:|
|        570 |                       0 |                       0 |              0.006172 |              0.012926 |                   0 |                           0 |

## Uncertainty quality
|   pearson_uq_error |   spearman_uq_error |   calibration_slope |   coverage80 |   coverage90 |   coverage95 |   mpiw90 | uncertainty_quality   |
|-------------------:|--------------------:|--------------------:|-------------:|-------------:|-------------:|---------:|:----------------------|
|           0.555968 |            0.430099 |              0.1009 |     0.808516 |     0.906558 |     0.956434 | 0.002532 | PASS                  |

## Conformal coverage
| scope           |   level |   nominal |   coverage |      n |
|:----------------|--------:|----------:|-----------:|-------:|
| global          |      80 |      0.8  |   0.808516 | 132810 |
| R1_linear       |      80 |      0.8  |   0.826946 |  22230 |
| R2_transitional |      80 |      0.8  |   0.824342 |  13680 |
| R3_pre_stall    |      80 |      0.8  |   0.801754 |  11400 |
| R4_stall_onset  |      80 |      0.8  |   0.780789 |  11400 |
| R5_near_stall   |      80 |      0.8  |   0.800877 |  45600 |
| R6_post_stall   |      80 |      0.8  |   0.812561 |  28500 |
| global          |      90 |      0.9  |   0.906558 | 132810 |
| R1_linear       |      90 |      0.9  |   0.926226 |  22230 |
| R2_transitional |      90 |      0.9  |   0.924269 |  13680 |
| R3_pre_stall    |      90 |      0.9  |   0.907982 |  11400 |
| R4_stall_onset  |      90 |      0.9  |   0.888246 |  11400 |
| R5_near_stall   |      90 |      0.9  |   0.895877 |  45600 |
| R6_post_stall   |      90 |      0.9  |   0.906561 |  28500 |
| global          |      95 |      0.95 |   0.956434 | 132810 |
| R1_linear       |      95 |      0.95 |   0.97193  |  22230 |
| R2_transitional |      95 |      0.95 |   0.965058 |  13680 |
| R3_pre_stall    |      95 |      0.95 |   0.957982 |  11400 |
| R4_stall_onset  |      95 |      0.95 |   0.947368 |  11400 |
| R5_near_stall   |      95 |      0.95 |   0.949232 |  45600 |
| R6_post_stall   |      95 |      0.95 |   0.954737 |  28500 |

## Geometry GroupKFold CV
See `reports/geometry_groupkfold_cv_report.md` and `tables/groupkfold_*` files. This validation holds out geometry groups by fold and replaces expensive LOGO refits.

## Geometry failure root cause
No root-cause table.

## Champion selection
Champion: **stall_weighted**

| candidate      |   overall_mae |   overall_rmse |       r2 |   logo_mae |   r1_mae |   r1_rmse |   r1_bias |   r1_low_drag_mae |   r2_mae |   r2_rmse |   r2_bias |   stall_region_mae |   near_stall_mae |   post_stall_mae |   high_cd_tail_mae |   high_cd_tail_rmse |   high_cd_tail_bias |   coverage90 |   uq_pearson | uncertainty_quality   |   physics_violation_count | selection_source        |   uq_pass_rank |
|:---------------|--------------:|---------------:|---------:|-----------:|---------:|----------:|----------:|------------------:|---------:|----------:|----------:|-------------------:|-----------------:|-----------------:|-------------------:|--------------------:|--------------------:|-------------:|-------------:|:----------------------|--------------------------:|:------------------------|---------------:|
| stall_weighted |      0.002038 |       0.003476 | 0.998134 |        nan | 0.00162  |  0.003405 | -0.000264 |          0.000973 | 0.002391 |  0.004087 | -0.000244 |           0.001973 |         0.001774 |         0.002051 |           0.004106 |            0.006101 |           -0.001037 |     0.898909 |          nan | CV_P50_ONLY           |                         0 | geometry_groupkfold_oof |              0 |
| baseline       |      0.002136 |       0.003574 | 0.998027 |        nan | 0.001646 |  0.003506 | -0.000234 |          0.000994 | 0.002251 |  0.003945 | -0.00025  |           0.00214  |         0.001967 |         0.002134 |           0.003939 |            0.005966 |           -0.002218 |     0.888877 |          nan | CV_P50_ONLY           |                         0 | geometry_groupkfold_oof |              0 |

## Day5 readiness
Exports `day5_interface/day5_cd_v3_interface.pkl` and `day5_interface/day5_interface.json` with safety-aware prediction methods.

## Research conclusions
This V3 system prioritizes geometry generalization, stall-region error, uncertainty validity, trust-region safety, and physics consistency over raw R².