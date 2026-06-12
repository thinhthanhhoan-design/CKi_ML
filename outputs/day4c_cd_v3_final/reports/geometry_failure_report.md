# Geometry Failure Analysis

## Representative geometries

### Best
| geom_hash                                |      mae |     rmse |       r2 |    t_max |   x_tmax |   camber_max |   x_cmax |   te_gap |   le_radius_proxy |   trailing_edge_angle |   aft_thickness |
|:-----------------------------------------|---------:|---------:|---------:|---------:|---------:|-------------:|---------:|---------:|------------------:|----------------------:|----------------:|
| ec7851c33a4a3477102fdc81efeb3a4a60b6bee4 | 0.000207 | 0.000291 | 0.999988 | 0.078445 | 0.289023 |     0.017493 |  0.44074 |        0 |           6.54106 |              0.091349 |        0.015106 |

### Median
| geom_hash                                |      mae |     rmse |       r2 |    t_max |   x_tmax |   camber_max |   x_cmax |   te_gap |   le_radius_proxy |   trailing_edge_angle |   aft_thickness |
|:-----------------------------------------|---------:|---------:|---------:|---------:|---------:|-------------:|---------:|---------:|------------------:|----------------------:|----------------:|
| 4f51c6264df9cb32638c30f8b9e3492dadfa9fe3 | 0.000374 | 0.000588 | 0.999928 | 0.130529 | 0.427655 |     0.035835 | 0.414621 |  0.00072 |            9.3759 |              0.198124 |        0.030977 |

### Worst
| geom_hash                                |      mae |     rmse |       r2 |    t_max |   x_tmax |   camber_max |   x_cmax |   te_gap |   le_radius_proxy |   trailing_edge_angle |   aft_thickness |
|:-----------------------------------------|---------:|---------:|---------:|---------:|---------:|-------------:|---------:|---------:|------------------:|----------------------:|----------------:|
| ef509a59974d9e5a916f5544c4e733f8b56dd6bf | 0.007364 | 0.015311 | 0.956555 | 0.478479 | 0.427655 |     0.033571 | 0.427655 |  0.23392 |           32.7567 |               2.26458 |        0.336148 |

## Descriptor correlations with LOGO MAE

| feature                     |   corr_with_logo_mae |
|:----------------------------|---------------------:|
| camber_max                  |             0.568753 |
| wake_proxy                  |             0.539829 |
| separation_proxy            |             0.512328 |
| camber_area                 |             0.48056  |
| camber_pressure_proxy       |             0.479137 |
| slope_variance_aft          |             0.476292 |
| moment_distribution_proxy   |             0.437885 |
| aft_pressure_recovery_proxy |             0.437174 |
| thickness_gradient_abs_aft  |             0.426743 |
| curvature_energy_aft        |             0.424846 |
| le_radius_proxy             |             0.420341 |
| hysteresis_proxy            |             0.414177 |
| te_gap                      |             0.412542 |
| t_max                       |             0.408013 |
| aft_loading_metric          |             0.393551 |
