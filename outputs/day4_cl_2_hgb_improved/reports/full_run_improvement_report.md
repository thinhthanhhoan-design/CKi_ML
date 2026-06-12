# Full-run CL Improvement Report

| metric                     |       old |   improved |   improvement_percent |
|:---------------------------|----------:|-----------:|----------------------:|
| OOF RMSE                   | 0.0617891 |  0.0569519 |             7.82845   |
| OOF R2                     | 0.994178  |  0.995054  |             0.0880959 |
| CLmax MAE                  | 0.0361923 |  0.0328319 |             9.28496   |
| Stall AoA MAE              | 1.25065   |  1.06242   |            15.0507    |
| Stall audit violation rate | 0.548161  |  0.0119089 |            97.8275    |
| 90% interval coverage      | 0.157552  |  0.923538  |           486.179     |
| 95% interval coverage      | 0.185863  |  0.963344  |           418.308     |

- Old stall violations: 1565/2855
- Improved stall violations: 34/2855
- UQ reliable: True
- Model family remains HGB-only; no CL feature schema or Day5 interface path was changed.
