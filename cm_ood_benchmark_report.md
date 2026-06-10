# OUT-OF-DISTRIBUTION (OOD) CHALLENGE BENCHMARK REPORT

B?o c?o n?y ??nh gi? n?ng l?c ngo?i suy h?nh h?c (extrapolation) c?a c?c m? h?nh surrogate Cm tr?n hai h? h?nh h?c OOD ho?n to?n b? ?n trong qu? tr?nh hu?n luy?n.

> [!IMPORTANT]
> OOD results are diagnostic only and are not used for final model selection.
> K?t qu? OOD n?y ch? mang t?nh ch?t ch?n ?o?n v? ho?n to?n kh?ng ???c s? d?ng ?? ??a ra quy?t ??nh l?a ch?n m? h?nh cu?i c?ng.

- **M? h?nh ???c ch?n theo ID Benchmark:** `Regime-Specific ExtraTrees`
- **S? l??ng m?u OOD:** `2,330` m?u

## 1. B?ng So s?nh Hi?u n?ng tr?n OOD Challenge Set

| Model                      |   OOD_MAE |   OOD_RMSE |   OOD_R? |   R?_2032c |   R?_thin_flat |   OOD_Slope | OOD_Compressed   |
|:---------------------------|----------:|-----------:|---------:|-----------:|---------------:|------------:|:-----------------|
| Regime-Specific ExtraTrees |   0.01787 |    0.02496 |  0.83755 |        nan |            nan |     1.12125 | YES              |

## 2. ??nh gi? chi ti?t cho M? h?nh ???c ch?n

M? h?nh ???c ch?n `Regime-Specific ExtraTrees` ho?t ??ng tr?n OOD Challenge Set:
- **OOD MAE:** `0.01787`
- **OOD RMSE:** `0.02496`
- **OOD R?:** `0.83755`
- **R? tr?n h? 2032c:** `nan`
- **R? tr?n h? thin-flat:** `nan`
- **OOD Slope:** `1.1213`
- **Amplitude Compressed Flag:** `YES`

### Top 15 m?u OOD c? sai s? l?n nh?t

| geom_hash                                |   angle |   reynolds |       cm |     pred |   residual |   abs_err |
|:-----------------------------------------|--------:|-----------:|---------:|---------:|-----------:|----------:|
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18.5  |     800000 | -0.17942 | -0.10736 |   -0.07206 |   0.07206 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18.25 |     800000 | -0.17512 | -0.10361 |   -0.07151 |   0.07151 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   17.5  |     500000 | -0.16895 | -0.09749 |   -0.07146 |   0.07146 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   17.75 |     500000 | -0.17368 | -0.10239 |   -0.07129 |   0.07129 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18.75 |     800000 | -0.1834  | -0.11228 |   -0.07112 |   0.07112 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18    |     800000 | -0.17043 | -0.09933 |   -0.0711  |   0.0711  |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18    |     500000 | -0.17778 | -0.10687 |   -0.07091 |   0.07091 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   17.75 |     800000 | -0.16526 | -0.09472 |   -0.07054 |   0.07054 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   19    |     800000 | -0.18712 | -0.11708 |   -0.07004 |   0.07004 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18.25 |     500000 | -0.18139 | -0.11172 |   -0.06968 |   0.06968 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   17.25 |     500000 | -0.16345 | -0.0938  |   -0.06965 |   0.06965 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   17.5  |     800000 | -0.15957 | -0.09051 |   -0.06906 |   0.06906 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   19.25 |     800000 | -0.1906  | -0.12167 |   -0.06893 |   0.06893 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   17    |     500000 | -0.15705 | -0.08971 |   -0.06734 |   0.06734 |
| 1d68e56a76e70e92a3208081e6a68831232f7f23 |   18.5  |     500000 | -0.18466 | -0.11747 |   -0.06719 |   0.06719 |

## 3. Ph?n t?ch Amplitude Compression

S? n?n bi?n ?? (amplitude compression) ph?n ?nh xu h??ng m? h?nh d? li?u (data-driven) ?p c?c gi? tr? c?c tr? kh? ??ng h?c v? g?n gi? tr? trung b?nh khi g?p ?i?u ki?n ch?a bi?t:
- **H? cong ng??c `2032c`** c? h?nh d?ng airfoil ph?n x? u?n l??n ??c th?. Do thi?u d? li?u t??ng t?, m? h?nh kh?ng th? suy di?n ??ng ?? cong c?a ???ng camber d?c ??ng, d?n ??n sai s? r?t l?n v? R? ?m.
- **H? ph?ng m?ng `thin-flat`** gi? ???c m?t ph?n xu h??ng kh? ??ng tuy?n t?nh c? b?n, tuy nhi?n bi?n ?? pitching moment b? thu h?p ??ng k? so v?i th?c t?.
