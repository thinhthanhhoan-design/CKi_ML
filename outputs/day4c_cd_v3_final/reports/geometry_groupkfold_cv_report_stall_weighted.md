# CD Geometry GroupKFold CV - stall_weighted

- Folds: 5
- Geometry groups: 571
- Weighted model: True
- CV model mode: p50_global_iter_180_bootstrap_1

## Overall
| scope                       |      n |      mae |     rmse |       r2 |    bias | tag            |
|:----------------------------|-------:|---------:|---------:|---------:|--------:|:---------------|
| geometry_groupkfold_overall | 665215 | 0.002038 | 0.003476 | 0.998134 | 9.5e-05 | stall_weighted |

## Fold Metrics
| scope             |      n |      mae |     rmse |       r2 |      bias |   fold | tag            | weighted   |   n_test_geometries | validation_mode     | cv_mode                         |
|:------------------|-------:|---------:|---------:|---------:|----------:|-------:|:---------------|:-----------|--------------------:|:--------------------|:--------------------------------|
| groupkfold_fold_1 | 133975 | 0.001955 | 0.003467 | 0.998127 |  0.000214 |      1 | stall_weighted | True       |                 115 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_2 | 132810 | 0.002046 | 0.003309 | 0.99826  |  9.5e-05  |      2 | stall_weighted | True       |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_3 | 132810 | 0.002094 | 0.003459 | 0.998152 |  0.000191 |      3 | stall_weighted | True       |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_4 | 132810 | 0.001964 | 0.003384 | 0.998265 | -0.000133 |      4 | stall_weighted | True       |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_5 | 132810 | 0.00213  | 0.003745 | 0.997859 |  0.000105 |      5 | stall_weighted | True       |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |

## Regime Metrics
| scope           |      n |      mae |     rmse |       r2 |      bias |   regime_id | tag            |
|:----------------|-------:|---------:|---------:|---------:|----------:|------------:|:---------------|
| R1_linear       | 111345 | 0.00162  | 0.003405 | 0.956673 | -0.000264 |           1 | stall_weighted |
| R2_transitional |  68520 | 0.002391 | 0.004087 | 0.978727 | -0.000244 |           2 | stall_weighted |
| R3_pre_stall    |  57100 | 0.002914 | 0.004494 | 0.986745 | -8.8e-05  |           3 | stall_weighted |
| R4_stall_onset  |  57100 | 0.00257  | 0.003981 | 0.993155 | -0.000117 |           4 | stall_weighted |
| R5_near_stall   | 228400 | 0.001774 | 0.002637 | 0.997909 | -5.2e-05  |           5 | stall_weighted |
| R6_post_stall   | 142750 | 0.002051 | 0.003701 | 0.995083 |  0.000929 |           6 | stall_weighted |

## Worst Geometry Groups
| scope                                    |    n |      mae |     rmse |       r2 |      bias | geom_hash                                | tag            |
|:-----------------------------------------|-----:|---------:|---------:|---------:|----------:|:-----------------------------------------|:---------------|
| f5b08930698c57af8d015c6714fa426c6a0b457d | 1165 | 0.009906 | 0.011874 | 0.970421 |  0.004925 | f5b08930698c57af8d015c6714fa426c6a0b457d | stall_weighted |
| 2c5fe8e9aa926c59341a62521e16cd16bc3d2393 | 1165 | 0.009857 | 0.015369 | 0.86744  | -0.006418 | 2c5fe8e9aa926c59341a62521e16cd16bc3d2393 | stall_weighted |
| af227cbf9adc8b20fdebaf34cd910bf02a198a66 | 1165 | 0.007884 | 0.011845 | 0.972354 | -0.003703 | af227cbf9adc8b20fdebaf34cd910bf02a198a66 | stall_weighted |
| 74c82006806abdb8e1b183d23c45a5fc5a40595f | 1165 | 0.007321 | 0.008777 | 0.989772 | -0.001371 | 74c82006806abdb8e1b183d23c45a5fc5a40595f | stall_weighted |
| 730c080d56ad11d58c83fcb2cbd84e11c9414aba | 1165 | 0.006242 | 0.009889 | 0.982523 | -0.002568 | 730c080d56ad11d58c83fcb2cbd84e11c9414aba | stall_weighted |
| 52bd83ceb37c1aab8b4f2c30e0b2cc3e69f47bc5 | 1165 | 0.006025 | 0.009922 | 0.982479 | -0.003078 | 52bd83ceb37c1aab8b4f2c30e0b2cc3e69f47bc5 | stall_weighted |
| 5f017e2f14b22bde7364907b88ad9debe03e2744 | 1165 | 0.005599 | 0.007692 | 0.989394 | -0.000174 | 5f017e2f14b22bde7364907b88ad9debe03e2744 | stall_weighted |
| a149443229ee3cbe37d17086aed88ecbf751569e | 1165 | 0.005359 | 0.008322 | 0.985597 | -0.000182 | a149443229ee3cbe37d17086aed88ecbf751569e | stall_weighted |
| 128fdeb6a620aa76381a578a1f3c105dbf0eaeac | 1165 | 0.005311 | 0.008325 | 0.98983  | -0.004274 | 128fdeb6a620aa76381a578a1f3c105dbf0eaeac | stall_weighted |
| 8669d841fe965d37094a78a440b6d017b1e74ea2 | 1165 | 0.005261 | 0.007155 | 0.992863 | -0.004687 | 8669d841fe965d37094a78a440b6d017b1e74ea2 | stall_weighted |
| d2c17d913a84654165433feb1b964577ad82d7e1 | 1165 | 0.005032 | 0.008648 | 0.985109 | -0.002595 | d2c17d913a84654165433feb1b964577ad82d7e1 | stall_weighted |
| d8f258b1a12d96a34d990a36dd243e853cd05a03 | 1165 | 0.004926 | 0.009219 | 0.981261 | -0.00213  | d8f258b1a12d96a34d990a36dd243e853cd05a03 | stall_weighted |
| b1d9de94608a669d89eabc4b6c4b67448eab5aaa | 1165 | 0.004766 | 0.006252 | 0.993517 | -0.001295 | b1d9de94608a669d89eabc4b6c4b67448eab5aaa | stall_weighted |
| ccaf8c8e6eea4b3ebf9d19e62066f52c96d7defd | 1165 | 0.004646 | 0.006229 | 0.9954   | -0.003437 | ccaf8c8e6eea4b3ebf9d19e62066f52c96d7defd | stall_weighted |
| 17c0783e33da18628b5c4abe864207e34d82cbe4 | 1165 | 0.004529 | 0.006001 | 0.993399 |  0.000175 | 17c0783e33da18628b5c4abe864207e34d82cbe4 | stall_weighted |
| 6f32aab0ff6833acf9be522c10827bec9d729383 | 1165 | 0.004512 | 0.006044 | 0.986738 |  0.003436 | 6f32aab0ff6833acf9be522c10827bec9d729383 | stall_weighted |
| bc79f88a278bbd514b8212380cfa4723b31440b9 | 1165 | 0.004479 | 0.00576  | 0.995792 | -0.000297 | bc79f88a278bbd514b8212380cfa4723b31440b9 | stall_weighted |
| 95a1517b1d019c4c667cba308dc28fab046a775c | 1165 | 0.004474 | 0.005392 | 0.995887 |  0.000423 | 95a1517b1d019c4c667cba308dc28fab046a775c | stall_weighted |
| 7c3f5e23065c39362e31761383a9b7095526db36 | 1165 | 0.004386 | 0.007371 | 0.991241 | -0.003297 | 7c3f5e23065c39362e31761383a9b7095526db36 | stall_weighted |
| 73a5162c600f295122de88ccd33c86a5bab3cecf | 1165 | 0.004334 | 0.006956 | 0.988434 | -0.00264  | 73a5162c600f295122de88ccd33c86a5bab3cecf | stall_weighted |