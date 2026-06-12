# CD Geometry GroupKFold CV - baseline

- Folds: 5
- Geometry groups: 571
- Weighted model: False
- CV model mode: p50_global_iter_180_bootstrap_1

## Overall
| scope                       |      n |      mae |     rmse |       r2 |    bias | tag      |
|:----------------------------|-------:|---------:|---------:|---------:|--------:|:---------|
| geometry_groupkfold_overall | 665215 | 0.002136 | 0.003574 | 0.998027 | 9.4e-05 | baseline |

## Fold Metrics
| scope             |      n |      mae |     rmse |       r2 |      bias |   fold | tag      | weighted   |   n_test_geometries | validation_mode     | cv_mode                         |
|:------------------|-------:|---------:|---------:|---------:|----------:|-------:|:---------|:-----------|--------------------:|:--------------------|:--------------------------------|
| groupkfold_fold_1 | 133975 | 0.002063 | 0.003567 | 0.998018 |  0.000277 |      1 | baseline | False      |                 115 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_2 | 132810 | 0.002158 | 0.00344  | 0.998119 |  0.000232 |      2 | baseline | False      |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_3 | 132810 | 0.002205 | 0.003617 | 0.99798  |  0.000132 |      3 | baseline | False      |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_4 | 132810 | 0.002079 | 0.003471 | 0.998175 | -0.000116 |      4 | baseline | False      |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |
| groupkfold_fold_5 | 132810 | 0.002174 | 0.003766 | 0.997836 | -5.6e-05  |      5 | baseline | False      |                 114 | geometry_groupkfold | p50_global_iter_180_bootstrap_1 |

## Regime Metrics
| scope           |      n |      mae |     rmse |       r2 |      bias |   regime_id | tag      |
|:----------------|-------:|---------:|---------:|---------:|----------:|------------:|:---------|
| R1_linear       | 111345 | 0.001646 | 0.003506 | 0.95407  | -0.000234 |           1 | baseline |
| R2_transitional |  68520 | 0.002251 | 0.003945 | 0.980176 | -0.00025  |           2 | baseline |
| R3_pre_stall    |  57100 | 0.002923 | 0.004561 | 0.986347 | -9.3e-05  |           3 | baseline |
| R4_stall_onset  |  57100 | 0.002844 | 0.004344 | 0.991847 | -0.000112 |           4 | baseline |
| R5_near_stall   | 228400 | 0.001967 | 0.002885 | 0.997499 | -5.7e-05  |           5 | baseline |
| R6_post_stall   | 142750 | 0.002134 | 0.003644 | 0.995231 |  0.000914 |           6 | baseline |

## Worst Geometry Groups
| scope                                    |    n |      mae |     rmse |       r2 |      bias | geom_hash                                | tag      |
|:-----------------------------------------|-----:|---------:|---------:|---------:|----------:|:-----------------------------------------|:---------|
| 2c5fe8e9aa926c59341a62521e16cd16bc3d2393 | 1165 | 0.01152  | 0.017685 | 0.824465 | -0.006855 | 2c5fe8e9aa926c59341a62521e16cd16bc3d2393 | baseline |
| f5b08930698c57af8d015c6714fa426c6a0b457d | 1165 | 0.010268 | 0.012318 | 0.968167 |  0.004729 | f5b08930698c57af8d015c6714fa426c6a0b457d | baseline |
| af227cbf9adc8b20fdebaf34cd910bf02a198a66 | 1165 | 0.008605 | 0.012553 | 0.96895  | -0.004053 | af227cbf9adc8b20fdebaf34cd910bf02a198a66 | baseline |
| 74c82006806abdb8e1b183d23c45a5fc5a40595f | 1165 | 0.007645 | 0.009326 | 0.988454 | -0.002106 | 74c82006806abdb8e1b183d23c45a5fc5a40595f | baseline |
| 730c080d56ad11d58c83fcb2cbd84e11c9414aba | 1165 | 0.006951 | 0.011085 | 0.978037 | -0.004537 | 730c080d56ad11d58c83fcb2cbd84e11c9414aba | baseline |
| 6f32aab0ff6833acf9be522c10827bec9d729383 | 1165 | 0.006115 | 0.007981 | 0.976875 |  0.00468  | 6f32aab0ff6833acf9be522c10827bec9d729383 | baseline |
| 52bd83ceb37c1aab8b4f2c30e0b2cc3e69f47bc5 | 1165 | 0.006039 | 0.010361 | 0.980895 | -0.002296 | 52bd83ceb37c1aab8b4f2c30e0b2cc3e69f47bc5 | baseline |
| 5f017e2f14b22bde7364907b88ad9debe03e2744 | 1165 | 0.005979 | 0.008037 | 0.98842  | -0.000568 | 5f017e2f14b22bde7364907b88ad9debe03e2744 | baseline |
| 128fdeb6a620aa76381a578a1f3c105dbf0eaeac | 1165 | 0.005874 | 0.008969 | 0.988197 | -0.005007 | 128fdeb6a620aa76381a578a1f3c105dbf0eaeac | baseline |
| a149443229ee3cbe37d17086aed88ecbf751569e | 1165 | 0.005601 | 0.008351 | 0.985495 |  0.000746 | a149443229ee3cbe37d17086aed88ecbf751569e | baseline |
| d8f258b1a12d96a34d990a36dd243e853cd05a03 | 1165 | 0.005302 | 0.009656 | 0.979443 | -0.001559 | d8f258b1a12d96a34d990a36dd243e853cd05a03 | baseline |
| 8669d841fe965d37094a78a440b6d017b1e74ea2 | 1165 | 0.00519  | 0.00784  | 0.991431 | -0.003843 | 8669d841fe965d37094a78a440b6d017b1e74ea2 | baseline |
| ccaf8c8e6eea4b3ebf9d19e62066f52c96d7defd | 1165 | 0.005145 | 0.00682  | 0.994486 | -0.004617 | ccaf8c8e6eea4b3ebf9d19e62066f52c96d7defd | baseline |
| d2c17d913a84654165433feb1b964577ad82d7e1 | 1165 | 0.004919 | 0.008083 | 0.986991 | -0.002633 | d2c17d913a84654165433feb1b964577ad82d7e1 | baseline |
| 95a1517b1d019c4c667cba308dc28fab046a775c | 1165 | 0.004839 | 0.00577  | 0.995289 |  0.001149 | 95a1517b1d019c4c667cba308dc28fab046a775c | baseline |
| bc79f88a278bbd514b8212380cfa4723b31440b9 | 1165 | 0.004685 | 0.005876 | 0.995622 | -0.000705 | bc79f88a278bbd514b8212380cfa4723b31440b9 | baseline |
| 7c3f5e23065c39362e31761383a9b7095526db36 | 1165 | 0.004609 | 0.007612 | 0.990658 | -0.002347 | 7c3f5e23065c39362e31761383a9b7095526db36 | baseline |
| b1d9de94608a669d89eabc4b6c4b67448eab5aaa | 1165 | 0.004481 | 0.006149 | 0.993728 | -0.002261 | b1d9de94608a669d89eabc4b6c4b67448eab5aaa | baseline |
| 17c0783e33da18628b5c4abe864207e34d82cbe4 | 1165 | 0.00446  | 0.00568  | 0.994087 |  0.000889 | 17c0783e33da18628b5c4abe864207e34d82cbe4 | baseline |
| e95814ea1ba24ef9d5bee8649e467ae959b41283 | 1165 | 0.004457 | 0.005701 | 0.995995 | -0.000862 | e95814ea1ba24ef9d5bee8649e467ae959b41283 | baseline |